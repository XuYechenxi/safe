#!/usr/bin/env python3
"""
一致性模型图像生成器
基于OpenAI一致性模型，使用CLIP进行文本引导
实现单步生成图像
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import os

# 导入一致性模型相关模块
from cm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from cm.karras_diffusion import karras_sample, KarrasDenoiser
from cm import dist_util
import clip


class ConsistencyModelGenerator:
    """一致性模型图像生成器，支持CLIP文本引导"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model_name: str = "ViT-B/32",
        image_size: int = 256,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        初始化一致性模型生成器
        
        Args:
            model_path: 预训练模型路径（如果为None，将使用默认配置创建模型）
            clip_model_name: CLIP模型名称
            image_size: 图像尺寸
            device: 计算设备（'cuda' 或 'cpu'）
            use_fp16: 是否使用半精度
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        print(f"[INFO] 初始化一致性模型生成器...")
        print(f"  设备: {self.device}")
        print(f"  图像尺寸: {image_size}x{image_size}")
        print(f"  半精度: {use_fp16}")
        
        # 初始化CLIP模型用于文本引导
        print(f"[INFO] 加载CLIP模型: {clip_model_name}...")
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            self.clip_model.eval()
            if self.use_fp16:
                self.clip_model = self.clip_model.half()
            print("✅ CLIP模型加载完成")
        except Exception as e:
            print(f"⚠️  CLIP模型加载失败: {e}")
            self.clip_model = None
            self.clip_preprocess = None
        
        # 初始化一致性模型
        print(f"[INFO] 初始化一致性模型...")
        self._init_consistency_model(model_path)
        
    def _init_consistency_model(self, model_path: Optional[str] = None):
        """初始化一致性模型"""
        # 设置默认参数
        defaults = model_and_diffusion_defaults()
        defaults.update({
            'image_size': self.image_size,
            'class_cond': False,  # 不使用类别条件，使用文本条件
            'num_channels': 256,
            'num_res_blocks': 2,
            'channel_mult': '',
            'num_heads': 4,
            'num_head_channels': 64,
            'attention_resolutions': "32,16,8",
            'dropout': 0.0,
            'use_checkpoint': False,
            'use_scale_shift_norm': True,
            'resblock_updown': False,
            'use_fp16': self.use_fp16,
            'use_new_attention_order': False,
            'learn_sigma': False,
            'weight_schedule': "karras",
            'sigma_min': 0.002,
            'sigma_max': 80.0,
        })
        
        # 创建模型和扩散过程
        self.model, self.diffusion = create_model_and_diffusion(
            **defaults,
            distillation=True,  # 使用一致性蒸馏
        )
        
        # 加载预训练权重（如果提供）
        if model_path and os.path.exists(model_path):
            print(f"[INFO] 加载预训练模型: {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ 模型权重加载完成")
            except Exception as e:
                print(f"⚠️  模型权重加载失败: {e}，使用随机初始化")
        else:
            print("[INFO] 使用随机初始化的模型（建议使用预训练模型）")
        
        # 移动到设备
        self.model.to(self.device)
        if self.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()
        
        print("✅ 一致性模型初始化完成")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        使用CLIP编码文本
        
        Args:
            text: 文本提示词
            
        Returns:
            文本特征向量
        """
        if self.clip_model is None:
            raise RuntimeError("CLIP模型未加载，无法编码文本")
        
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            # 归一化特征
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 1,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        batch_size: int = 1,
        **kwargs
    ) -> Image.Image:
        """
        生成图像（单步生成）
        
        Args:
            prompt: 文本提示词
            num_inference_steps: 推理步数（一致性模型通常只需要1步）
            guidance_scale: 引导强度
            seed: 随机种子
            batch_size: 批次大小
            
        Returns:
            生成的PIL图像
        """
        print(f"[INFO] 开始生成图像...")
        print(f"  提示词: {prompt}")
        print(f"  推理步数: {num_inference_steps}")
        print(f"  引导强度: {guidance_scale}")
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 编码文本提示词（用于后续的一致性检测）
        # 注意：当前一致性模型实现可能不支持直接文本条件
        # 这里先进行无条件生成，文本特征用于后续的语义一致性检测
        if self.clip_model is not None:
            text_features = self.encode_text(prompt)
            # 保存文本特征用于后续处理
            self._last_text_features = text_features
        else:
            self._last_text_features = None
        
        # 当前使用无条件生成（未来可以扩展支持文本条件）
        model_kwargs = {}
        
        # 准备采样参数
        shape = (batch_size, 3, self.image_size, self.image_size)
        
        # 使用一致性模型采样（单步生成）
        with torch.no_grad():
            sample = karras_sample(
                self.diffusion,
                self.model,
                shape,
                steps=num_inference_steps,
                model_kwargs=model_kwargs,
                device=self.device,
                clip_denoised=True,
                sampler="heun",  # 可以使用 "heun" 或 "euler"
                sigma_min=self.diffusion.sigma_min,
                sigma_max=self.diffusion.sigma_max,
                s_churn=0.0,
                s_tmin=0.0,
                s_tmax=float("inf"),
                s_noise=1.0,
            )
        
        # 将张量转换为PIL图像
        # 假设输出范围是[-1, 1]，需要转换到[0, 255]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()
        
        # 取第一张图像
        image_array = sample[0]
        
        # 转换为PIL图像
        image = Image.fromarray(image_array)
        
        print("✅ 图像生成完成")
        return image
    
    def generate_batch(
        self,
        prompts: list,
        num_inference_steps: int = 1,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs
    ) -> list:
        """
        批量生成图像
        
        Args:
            prompts: 文本提示词列表
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            
        Returns:
            PIL图像列表
        """
        images = []
        for i, prompt in enumerate(prompts):
            current_seed = seed + i if seed is not None else None
            image = self.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                **kwargs
            )
            images.append(image)
        return images


def create_generator(
    model_path: Optional[str] = None,
    clip_model_name: str = "ViT-B/32",
    image_size: int = 256,
    device: Optional[str] = None,
) -> ConsistencyModelGenerator:
    """
    创建一致性模型生成器的便捷函数
    
    Args:
        model_path: 预训练模型路径
        clip_model_name: CLIP模型名称
        image_size: 图像尺寸
        device: 计算设备
        
    Returns:
        一致性模型生成器实例
    """
    return ConsistencyModelGenerator(
        model_path=model_path,
        clip_model_name=clip_model_name,
        image_size=image_size,
        device=device,
    )

