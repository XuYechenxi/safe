#!/usr/bin/env python3
"""
图像生成服务模块
支持多种模型：SD基础模型、CLIP融合模型、ITSC-GAN模型
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from PIL import Image, ImageEnhance, ImageFilter
import torch
import threading
import numpy as np

# 尝试导入不同的生成器
try:
    from consistency_model_generator import ConsistencyModelGenerator, create_generator
    HAS_CM = True
except ImportError:
    HAS_CM = False

try:
    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    HAS_SD = True
except ImportError:
    HAS_SD = False


class ImageGenerationService:
    """图像生成服务 - 支持多种模型"""
    
    def __init__(self, device=None):
        """初始化服务
        
        Args:
            device: 计算设备（None时自动检测）
        """
        # 自动检测设备（安全检测CUDA是否真正可用）
        if device is not None:
            self.device = device
        else:
            self.device = self._safe_get_device()
        # 为每种模型类型使用独立实例
        self.generators = {}
        self.current_model = None
        # 添加线程锁，防止并发加载模型
        self._loading_locks = {}  # 每个模型一个锁
        self._global_lock = threading.Lock()  # 全局锁，用于保护字典操作
    
    def _safe_get_device(self):
        """
        安全检测可用的计算设备
        即使torch.cuda.is_available()返回True，也可能实际上不可用（未编译CUDA支持）
        """
        # 首先检查torch.cuda.is_available()
        if not torch.cuda.is_available():
            print("[INFO] CUDA不可用，使用CPU")
            return "cpu"
        
        # 尝试实际使用CUDA来验证是否真正可用
        try:
            # 创建一个小的tensor并移动到CUDA来测试
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("[INFO] CUDA可用，使用CUDA")
            return "cuda"
        except (AssertionError, RuntimeError) as e:
            # CUDA不可用（可能是未编译CUDA支持）
            print(f"[WARNING] CUDA检测失败: {e}")
            print("[INFO] 回退到CPU模式")
            return "cpu"
        except Exception as e:
            # 其他错误，也回退到CPU
            print(f"[WARNING] CUDA检测时出现未知错误: {e}")
            print("[INFO] 回退到CPU模式")
            return "cpu"
    
    def _apply_quality_enhancements(self, pipeline):
        """
        对生成管线应用统一的质量优化（仅执行一次）
        """
        if not pipeline or getattr(pipeline, "_quality_optimized", False):
            return pipeline
        
        try:
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(f"[WARNING] 启用xFormers注意力失败: {e}")
            if hasattr(pipeline, "scheduler") and HAS_SD:
                try:
                    if not isinstance(pipeline.scheduler, EulerAncestralDiscreteScheduler):
                        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
                        print("[INFO] ✅ 使用 EulerAncestralDiscreteScheduler 以提高细节表现")
                except Exception as e:
                    print(f"[WARNING] 替换调度器失败: {e}")
        except Exception as e:
            print(f"[WARNING] 应用质量优化失败: {e}")
        
        setattr(pipeline, "_quality_optimized", True)
        return pipeline
    
    def _enhance_image_quality(self, image: Image.Image, model_name: str) -> Image.Image:
        """
        对生成的图像进行后处理增强，提高色彩饱和度和合理性
        
        Args:
            image: 原始生成的图像
            model_name: 模型名称，用于应用不同的增强策略
        
        Returns:
            增强后的图像
        """
        try:
            enhanced = image.copy()
            
            # 根据模型类型应用不同的增强策略
            if "itsc-gan" in model_name.lower():
                # ITSC-GAN模型：增强色彩饱和度和对比度
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.15)  # 增加15%的色彩饱和度
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.1)  # 增加10%的对比度
                
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.05)  # 轻微锐化
                
            elif "clip" in model_name.lower():
                # CLIP融合模型：适度增强
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.1)  # 增加10%的色彩饱和度
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.05)  # 增加5%的对比度
                
            else:
                # SD基础模型：轻微增强
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.08)  # 增加8%的色彩饱和度
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(1.03)  # 增加3%的对比度
            
            # 应用轻微的降噪（如果有噪点）
            # 使用轻微的高斯模糊来平滑噪点，但保持细节
            enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)
            
            # 确保图像在合理范围内（防止过度增强导致的失真）
            # 转换为numpy数组进行范围检查
            img_array = np.array(enhanced)
            # 确保像素值在0-255范围内
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            enhanced = Image.fromarray(img_array)
            
            print("[INFO] ✅ 图像质量增强完成")
            return enhanced
            
        except Exception as e:
            print(f"[WARNING] 图像增强失败: {e}，返回原始图像")
            return image
    
    def get_generator(self, model_name: str, device: str = None):
        """
        获取指定模型的生成器 - 确保每个模型类型使用独立实例（线程安全）
        
        Args:
            model_name: 模型名称
            device: 计算设备（None时使用初始化时设置的设备）
        
        Returns:
            生成器实例
        """
        # 使用指定设备或默认设备（安全检测）
        if device is not None:
            # 验证指定的设备是否可用
            if device == "cuda":
                try:
                    test_tensor = torch.tensor([1.0])
                    test_tensor = test_tensor.cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    active_device = "cuda"
                except (AssertionError, RuntimeError):
                    print(f"[WARNING] 指定的CUDA设备不可用，回退到CPU")
                    active_device = "cpu"
            else:
                active_device = device
        else:
            active_device = self.device
        
        # 首先检查模型是否已加载（快速路径，无需加锁）
        with self._global_lock:
            if model_name in self.generators:
                return self._apply_quality_enhancements(self.generators[model_name])
            
            # 获取或创建该模型的加载锁（在全局锁内获取，避免竞争）
            if model_name not in self._loading_locks:
                self._loading_locks[model_name] = threading.Lock()
            model_lock = self._loading_locks[model_name]
        
        # 使用模型特定的锁来防止并发加载（注意：不要在持有全局锁时获取模型锁，避免死锁）
        with model_lock:
            # 再次检查（双重检查锁定模式）- 需要重新获取全局锁
            with self._global_lock:
                if model_name in self.generators:
                    return self._apply_quality_enhancements(self.generators[model_name])
            
            print(f"[INFO] 正在加载模型: {model_name} (设备: {active_device})")
            print(f"[INFO] 注意: 模型加载可能需要几分钟，请耐心等待...")
        
        # 为每种模型类型创建独立实例
        if model_name == "runwayml/stable-diffusion-v1-5" or model_name == "sd-base":
            # Stable Diffusion 基础模型
            if not HAS_SD:
                raise RuntimeError("Stable Diffusion 未安装，请安装 diffusers")
            
            # CPU模式下直接加载到CPU，避免meta tensor问题
            if active_device == "cpu":
                print("[INFO] CPU模式：直接加载模型到CPU...")
                try:
                    # 直接加载到CPU，不使用low_cpu_mem_usage避免meta tensor问题
                    generator = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    # 安全地移动到CPU（处理可能的meta tensor问题）
                    try:
                        generator = generator.to("cpu")
                    except (NotImplementedError, RuntimeError) as meta_error:
                        if "meta tensor" in str(meta_error).lower() or "to_empty" in str(meta_error).lower():
                            print("[WARNING] 检测到meta tensor问题，尝试使用备用方法...")
                            # 重新加载，不使用任何可能导致meta tensor的参数
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                torch_dtype=torch.float32,
                                safety_checker=None,
                                requires_safety_checker=False,
                                variant=None  # 明确不使用variant
                            )
                            # 尝试逐个组件移动到CPU
                            for component_name in ['vae', 'text_encoder', 'unet', 'tokenizer', 'scheduler']:
                                if hasattr(generator, component_name):
                                    component = getattr(generator, component_name)
                                    if component is not None:
                                        try:
                                            setattr(generator, component_name, component.to("cpu"))
                                        except:
                                            pass
                    print("[INFO] ✅ 已禁用安全检查器以节省内存")
                except Exception as variant_error:
                    # 如果加载失败，尝试不指定torch_dtype
                    print(f"[WARNING] 使用float32加载失败: {variant_error}，尝试自动选择数据类型...")
                    generator = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    try:
                        generator = generator.to("cpu")
                    except (NotImplementedError, RuntimeError):
                        # 如果移动失败，模型可能已经在CPU上
                        pass
                    print("[INFO] ✅ 使用自动数据类型加载成功")
            else:
                generator = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16
                )
                generator = generator.to(active_device)
            generator.enable_attention_slicing()
            
        elif model_name == "clip-fusion" or model_name == "openai-clip-fusion":
            # CLIP融合模型（使用SD + CLIP增强）
            if not HAS_SD:
                raise RuntimeError("Stable Diffusion 未安装，请安装: pip install diffusers")
            
            print("[INFO] 加载CLIP融合模型，应用语义增强")
            print("[INFO] 注意: CLIP融合模型使用增强的引导强度来提升语义一致性")
            
            # 优化：如果SD基础模型已加载，尝试复用（节省内存）
            sd_model_name = "runwayml/stable-diffusion-v1-5"
            if sd_model_name in self.generators:
                print("[INFO] 检测到SD基础模型已加载，复用模型实例以节省内存...")
                try:
                    # 尝试复制模型引用（轻量级）
                    generator = self.generators[sd_model_name]
                    print("[INFO] ✅ 复用SD基础模型实例成功")
                except Exception as e:
                    print(f"[WARNING] 复用模型失败: {e}，将创建新实例")
                    # CPU模式下直接加载到CPU，避免meta tensor问题
                    if active_device == "cpu":
                        print("[INFO] CPU模式：直接加载模型到CPU...")
                        try:
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                torch_dtype=torch.float32,
                                safety_checker=None,
                                requires_safety_checker=False
                            )
                            # 确保模型在CPU上（避免meta tensor问题）
                            generator = generator.to("cpu")
                            print("[INFO] ✅ 已禁用安全检查器以节省内存")
                        except Exception as variant_error:
                            # 如果加载失败，尝试不指定torch_dtype
                            print(f"[WARNING] 使用float32加载失败: {variant_error}，尝试自动选择数据类型...")
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                safety_checker=None,
                                requires_safety_checker=False
                            )
                            generator = generator.to("cpu")
                            print("[INFO] ✅ 使用自动数据类型加载成功")
                    else:
                        generator = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=torch.float16
                        )
                        generator = generator.to(active_device)
                    generator.enable_attention_slicing()
            else:
                # 创建新实例
                print("[INFO] 创建新的SD模型实例...")
                try:
                    # CPU模式下直接加载到CPU，避免meta tensor问题
                    if active_device == "cpu":
                        print("[INFO] CPU模式：直接加载模型到CPU...")
                        try:
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                torch_dtype=torch.float32,
                                safety_checker=None,
                                requires_safety_checker=False
                            )
                            # 确保模型在CPU上（避免meta tensor问题）
                            generator = generator.to("cpu")
                            print("[INFO] ✅ 已禁用安全检查器以节省内存")
                        except Exception as variant_error:
                            # 如果加载失败，尝试不指定torch_dtype
                            print(f"[WARNING] 使用float32加载失败: {variant_error}，尝试自动选择数据类型...")
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                safety_checker=None,
                                requires_safety_checker=False
                            )
                            generator = generator.to("cpu")
                            print("[INFO] ✅ 使用自动数据类型加载成功")
                    else:
                        generator = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5",
                            torch_dtype=torch.float16
                        )
                        generator = generator.to(active_device)
                    generator.enable_attention_slicing()
                except OSError as e:
                    if "1455" in str(e) or "页面文件" in str(e) or "page file" in str(e).lower():
                        error_msg = (
                            "❌ 内存不足，无法加载模型\n"
                            "错误: 页面文件太小，无法完成操作\n\n"
                            "解决方案:\n"
                            "1. 增加Windows虚拟内存（页面文件）大小:\n"
                            "   - 右键'此电脑' -> 属性 -> 高级系统设置\n"
                            "   - 性能 -> 设置 -> 高级 -> 虚拟内存\n"
                            "   - 建议设置为至少16GB\n"
                            "2. 关闭其他占用内存的程序\n"
                            "3. 如果已加载SD基础模型，CLIP融合模型会尝试复用\n"
                            "4. 使用CPU模式时内存占用更大，建议使用GPU（如果可用）\n"
                        )
                        print(error_msg)
                        raise RuntimeError("内存不足，无法加载模型。请增加虚拟内存或关闭其他程序。")
                    else:
                        raise
            
            # CLIP增强通过更高的引导强度实现
            print("[INFO] CLIP融合模型已加载，将在生成时使用增强的引导强度（>=8.5）")
            
        elif model_name == "itsc-gan-fusion":
            # ITSC-GAN融合模型
            if not HAS_SD:
                raise RuntimeError("Stable Diffusion 未安装，请安装: pip install diffusers")
            
            print("[INFO] 加载ITSC-GAN融合模型，应用风格和细节增强")
            
            # 优化：如果SD基础模型已加载，尝试复用（节省内存）
            sd_model_name = "runwayml/stable-diffusion-v1-5"
            if sd_model_name in self.generators:
                print("[INFO] 检测到SD基础模型已加载，复用模型实例以节省内存...")
                try:
                    generator = self.generators[sd_model_name]
                    print("[INFO] ✅ 复用SD基础模型实例成功")
                    reuse_model = True
                except Exception as e:
                    print(f"[WARNING] 复用模型失败: {e}，将创建新实例")
                    reuse_model = False
            else:
                reuse_model = False
            
            if not reuse_model:
                # 创建新实例
                print("[INFO] 创建新的SD模型实例...")
                retry_count = 0
                max_retries = 2
                generator = None
                
                while retry_count < max_retries and generator is None:
                    try:
                        retry_count += 1
                        if retry_count > 1:
                            print(f"[INFO] 第 {retry_count} 次重试加载模型...")
                            # 重试时使用更激进的内存优化
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # CPU模式下直接加载到CPU，避免meta tensor问题
                        if active_device == "cpu":
                            print("[INFO] CPU模式：直接加载模型到CPU...")
                            # CPU模式下使用float32，直接加载到CPU避免meta tensor问题
                            try:
                                generator = StableDiffusionPipeline.from_pretrained(
                                    "runwayml/stable-diffusion-v1-5",
                                    torch_dtype=torch.float32,
                                    safety_checker=None,
                                    requires_safety_checker=False
                                )
                                # 安全地移动到CPU（处理可能的meta tensor问题）
                                try:
                                    generator = generator.to("cpu")
                                except (NotImplementedError, RuntimeError) as meta_error:
                                    if "meta tensor" in str(meta_error).lower() or "to_empty" in str(meta_error).lower():
                                        print("[WARNING] 检测到meta tensor问题，尝试使用备用方法...")
                                        # 重新加载，不使用任何可能导致meta tensor的参数
                                        generator = StableDiffusionPipeline.from_pretrained(
                                            "runwayml/stable-diffusion-v1-5",
                                            torch_dtype=torch.float32,
                                            safety_checker=None,
                                            requires_safety_checker=False,
                                            variant=None
                                        )
                                        # 尝试逐个组件移动到CPU
                                        for component_name in ['vae', 'text_encoder', 'unet', 'tokenizer', 'scheduler']:
                                            if hasattr(generator, component_name):
                                                component = getattr(generator, component_name)
                                                if component is not None:
                                                    try:
                                                        setattr(generator, component_name, component.to("cpu"))
                                                    except:
                                                        pass
                                print("[INFO] ✅ 已禁用安全检查器以节省内存")
                            except Exception as variant_error:
                                # 如果加载失败，尝试不指定torch_dtype，让系统自动选择
                                print(f"[WARNING] 使用float32加载失败: {variant_error}，尝试自动选择数据类型...")
                                generator = StableDiffusionPipeline.from_pretrained(
                                    "runwayml/stable-diffusion-v1-5",
                                    safety_checker=None,
                                    requires_safety_checker=False
                                )
                                try:
                                    generator = generator.to("cpu")
                                except (NotImplementedError, RuntimeError):
                                    # 如果移动失败，模型可能已经在CPU上
                                    pass
                                print("[INFO] ✅ 使用自动数据类型加载成功")
                        else:
                            generator = StableDiffusionPipeline.from_pretrained(
                                "runwayml/stable-diffusion-v1-5",
                                torch_dtype=torch.float16
                            )
                            generator = generator.to(active_device)
                            break
                        
                    except OSError as e:
                        if "1455" in str(e) or "页面文件" in str(e) or "page file" in str(e).lower():
                            # 最后一次重试失败时才显示错误
                            if retry_count == max_retries:
                                # 获取Windows系统内存信息
                                total_memory = "未知"
                                free_memory = "未知"
                                try:
                                    import psutil
                                    mem = psutil.virtual_memory()
                                    total_memory = f"{mem.total / (1024*1024*1024):.1f} GB"
                                    free_memory = f"{mem.available / (1024*1024*1024):.1f} GB"
                                except ImportError:
                                    pass
                                except Exception:
                                    pass
                                    
                                error_msg = (
                                "❌ 内存不足，无法加载模型\n"
                                "错误: 页面文件太小，无法完成操作\n\n"
                                f"当前系统内存状态:\n"
                                f"- 总内存: {total_memory}\n"
                                f"- 可用内存: {free_memory}\n\n"
                                "- 解决方案1: 关闭其他占用内存的程序\n"
                                "- 解决方案2: 增加Windows虚拟内存（页面文件）大小:\n"
                                "  * 右键'此电脑' -> 属性 -> 高级系统设置\n"
                                "  * 性能 -> 设置 -> 高级 -> 虚拟内存\n"
                                "  * 建议设置为至少16GB\n"
                                "- 解决方案3: 优先使用SD基础模型，避免同时加载多个模型\n"
                                "- 解决方案4: 使用更小的图像尺寸（512x512或更小）\n"
                                "- 解决方案5: 减少推理步数（建议20-30步）\n"
                            )
                                print(error_msg)
                                raise RuntimeError("内存不足，无法加载模型。请关闭其他程序或增加虚拟内存。")
                            else:
                                print(f"[WARNING] 模型加载失败，将重试... (错误: {e})")
                                time.sleep(1)  # 等待1秒后重试
                        else:
                            raise
            
            # 尝试加载预训练的LoRA权重（如果存在）
            lora_loaded = False
            try:
                lora_path = os.path.join("models", "strategy_a_lora")
                adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
                
                # 检查LoRA文件是否存在
                if os.path.exists(adapter_model_path) or os.path.exists(lora_path):
                    print(f"[INFO] 发现LoRA权重目录: {lora_path}")
                    print(f"[INFO] 尝试加载LoRA权重以启用ITSC-GAN增强...")
                    try:
                        # 方法1: 使用diffusers的load_lora_weights方法（推荐）
                        if hasattr(generator, 'load_lora_weights'):
                            try:
                                # 检查是否有adapter_model.safetensors或adapter_model.bin
                                if os.path.exists(adapter_model_path):
                                    generator.load_lora_weights(lora_path)
                                    lora_loaded = True
                                    print(f"[INFO] ✅ LoRA权重加载成功（使用diffusers）！ITSC-GAN模型已启用")
                                elif os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
                                    generator.load_lora_weights(lora_path)
                                    lora_loaded = True
                                    print(f"[INFO] ✅ LoRA权重加载成功（使用diffusers，bin格式）！ITSC-GAN模型已启用")
                                else:
                                    print(f"[WARNING] LoRA目录存在但未找到adapter_model文件")
                            except Exception as e1:
                                print(f"[WARNING] diffusers load_lora_weights失败: {e1}")
                                # 方法2: 尝试使用PEFT库（备选方案）
                                try:
                                    from peft import PeftModel
                                    generator = PeftModel.from_pretrained(generator, lora_path)
                                    lora_loaded = True
                                    print(f"[INFO] ✅ LoRA权重加载成功（使用PEFT）！ITSC-GAN模型已启用")
                                except ImportError:
                                    print(f"[WARNING] PEFT库未安装，无法使用PEFT加载LoRA")
                                    print(f"  安装命令: pip install peft")
                                except Exception as e2:
                                    print(f"[WARNING] PEFT加载失败: {e2}")
                        else:
                            print(f"[WARNING] diffusers版本不支持load_lora_weights，尝试PEFT...")
                            try:
                                from peft import PeftModel
                                generator = PeftModel.from_pretrained(generator, lora_path)
                                lora_loaded = True
                                print(f"[INFO] ✅ LoRA权重加载成功（使用PEFT）！ITSC-GAN模型已启用")
                            except ImportError:
                                print(f"[WARNING] PEFT库未安装")
                            except Exception as e3:
                                print(f"[WARNING] PEFT加载失败: {e3}")
                    except Exception as e:
                        print(f"[WARNING] LoRA权重加载过程出错: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[INFO] 未找到LoRA权重目录: {lora_path}")
                    print(f"[INFO] ITSC-GAN模型将使用增强参数模式运行")
                    
            except Exception as e:
                print(f"[WARNING] 检查LoRA权重时出错: {e}")
                import traceback
                traceback.print_exc()
            
            if not reuse_model:
                # 安全地移动到设备（处理可能的meta tensor问题）
                try:
                    generator = generator.to(active_device)
                except (NotImplementedError, RuntimeError) as meta_error:
                    if "meta tensor" in str(meta_error).lower() or "to_empty" in str(meta_error).lower():
                        print("[WARNING] 检测到meta tensor问题，尝试逐个组件移动...")
                        # 尝试逐个组件移动到设备
                        for component_name in ['vae', 'text_encoder', 'unet', 'tokenizer', 'scheduler']:
                            if hasattr(generator, component_name):
                                component = getattr(generator, component_name)
                                if component is not None:
                                    try:
                                        setattr(generator, component_name, component.to(active_device))
                                    except Exception as comp_error:
                                        print(f"[WARNING] 移动组件 {component_name} 失败: {comp_error}")
                    else:
                        raise
                generator.enable_attention_slicing()
            
            if not lora_loaded:
                print(f"[INFO] ITSC-GAN模型将以增强参数模式运行（更高推理步数和引导强度）")
            
        elif model_name == "consistency-model" or model_name == "cm":
            # 一致性模型
            if not HAS_CM:
                raise RuntimeError("一致性模型未安装")
            
            import config_cm as config
            generator = create_generator(
                model_path=config.CONSISTENCY_MODEL_PATH,
                clip_model_name=config.CLIP_MODEL_NAME,
                image_size=config.CONSISTENCY_MODEL_IMAGE_SIZE,
                device=active_device
            )
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        
        # 存储独立的模型实例（线程安全）
        with self._global_lock:
            # 再次检查，防止重复加载
            if model_name not in self.generators:
                self.generators[model_name] = self._apply_quality_enhancements(generator)
                self.current_model = model_name
                print(f"[INFO] ✅ 模型 {model_name} 加载完成")
            else:
                # 如果已经加载，使用已存在的实例
                generator = self._apply_quality_enhancements(self.generators[model_name])
                print(f"[INFO] 模型 {model_name} 已存在，使用已加载的实例")
        
        return generator
        
    def clear_cache(self):
        """清理模型缓存，释放内存（线程安全）"""
        with self._global_lock:
            for model_name in list(self.generators.keys()):
                try:
                    del self.generators[model_name]
                except:
                    pass
            self.generators = {}
            self.current_model = None
            self._loading_locks.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] 模型缓存已清理")
    
    def generate(
        self,
        prompt: str,
        model_name: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> Image.Image:
        """
        生成图像，支持进度回调
        
        Args:
            prompt: 文本提示词
            model_name: 模型名称
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            device: 计算设备
            progress_callback: 进度回调函数，接收进度百分比(0-100)和状态信息
        
        Returns:
            生成的图像
        """
        # 自动检测设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] 使用设备: {device}")

        # 初始化进度回调
        def _update_progress(progress: float, status: str = ""):
            """内部进度更新函数"""
            if progress_callback:
                try:
                    progress_callback(progress, status)
                except Exception as e:
                    print(f"[WARNING] 进度回调执行失败: {e}")

        _update_progress(0.0, "开始加载模型...")

        # 获取生成器（线程安全）
        try:
            generator = self.get_generator(model_name, device)
            _update_progress(10.0, "模型加载完成")
        except Exception as e:
            print(f"[ERROR] 获取模型生成器失败: {e}")
            import traceback
            traceback.print_exc()
            _update_progress(0.0, f"错误: 无法加载模型")
            raise RuntimeError(f"无法加载模型 {model_name}: {str(e)}")

        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"[INFO] 设置随机种子: {seed}")

        # 定义统一的负面提示词，帮助抑制低质量特征
        quality_negative_prompt = (
            "low quality, lowres, blurry, bad anatomy, distortion, watermark, text, "
            "duplicate, worst quality, jpeg artifacts, oversaturated, underexposed, "
            "extra limbs, missing fingers, deformed, distorted face, grainy"
        )
        
        # 定义进度回调包装器
        def callback_wrapper(step: int, timestep: int, latents: torch.Tensor):
            """diffusers进度回调包装器"""
            progress_percent = 10.0 + (step / num_inference_steps) * 85.0
            _update_progress(min(progress_percent, 95.0), f"生成中... {step+1}/{num_inference_steps}")

        # 根据模型类型生成
        try:
            # 为所有模型类型添加进度回调
            if model_name in ["runwayml/stable-diffusion-v1-5", "sd-base"]:
                # SD基础模型 - 使用标准参数
                print("[INFO] 使用SD基础模型生成图像")
                _update_progress(15.0, "准备SD基础模型...")
                sd_steps = max(num_inference_steps, 32)  # 增加默认步数
                sd_guidance = max(guidance_scale, 8.2)  # 提高引导强度
                # 增强正面提示词
                enhanced_prompt = f"{prompt}, high quality, detailed"
                if "negative_prompt" not in kwargs:
                    kwargs["negative_prompt"] = quality_negative_prompt
                result = generator(
                    prompt=enhanced_prompt,
                    num_inference_steps=sd_steps,
                    guidance_scale=sd_guidance,
                    height=height,
                    width=width,
                    callback=callback_wrapper if progress_callback else None,
                    callback_steps=5 if progress_callback else 1,
                    **kwargs
                )
                image = result.images[0]
            
            elif model_name in ["clip-fusion", "openai-clip-fusion"]:
                # CLIP融合模型 - 使用更强的引导强度来增强语义一致性
                print("[INFO] 使用CLIP融合模型生成图像，增强语义理解")
                _update_progress(15.0, "准备CLIP融合模型...")
                # 增加引导强度以更好地匹配文本描述
                clip_guidance_scale = max(guidance_scale, 9.0)  # 提高引导强度
                clip_steps = max(num_inference_steps, 35)  # 增加步数
                clip_negative = (
                    "pixelated, blurry, low quality, bad anatomy, low detail, "
                    "mutated hands, extra limbs, watermark, text, artifacts, "
                    "dull colors, desaturated, bad lighting"
                )
                # 增强正面提示词
                enhanced_prompt = f"{prompt}, high quality, detailed, vibrant colors"
                result = generator(
                    prompt=enhanced_prompt,
                    num_inference_steps=clip_steps,
                    guidance_scale=clip_guidance_scale,  # 更强的引导
                    height=height,
                    width=width,
                    # 添加额外的负面提示以提高质量
                    negative_prompt=clip_negative,
                    callback=callback_wrapper if progress_callback else None,
                    callback_steps=5 if progress_callback else 1,
                    **kwargs
                )
                image = result.images[0]
            
            elif model_name == "itsc-gan-fusion":
                # ITSC-GAN融合模型 - 优化细节和风格
                print("[INFO] 使用ITSC-GAN融合模型生成图像，优化细节和质量")
                _update_progress(15.0, "准备ITSC-GAN融合模型...")
                # 增加推理步数以提高细节和色彩质量
                itsc_num_steps = max(num_inference_steps, 40)  # 增加默认步数
                itsc_guidance = max(guidance_scale, 8.5)  # 提高引导强度以增强语义一致性
                # 优化的负面提示词，强调色彩和合理性
                itsc_negative = (
                    "noise, pixelated, lowres, washed out, dull colors, desaturated, "
                    "bad proportions, bad hands, extra limbs, watermark, signature, "
                    "distorted, blurry, oversaturated, underexposed, bad lighting, "
                    "unrealistic, unnatural colors, color distortion"
                )
                # 增强正面提示词，强调色彩和细节
                enhanced_prompt = f"{prompt}, vibrant colors, high quality, detailed, realistic, well-lit"
                result = generator(
                    prompt=enhanced_prompt,
                    num_inference_steps=itsc_num_steps,  # 更多步数
                    guidance_scale=itsc_guidance,
                    height=height,
                    width=width,
                    # 添加风格优化提示
                    negative_prompt=itsc_negative,
                    callback=callback_wrapper if progress_callback else None,
                    callback_steps=5 if progress_callback else 1,
                    **kwargs
                )
                image = result.images[0]
                
            elif model_name in ["consistency-model", "cm"]:
                # 一致性模型
                print("[INFO] 使用一致性模型生成图像")
                _update_progress(15.0, "准备一致性模型...")
                # 为一致性模型创建特定的进度回调
                cm_step = 0
                def cm_callback(progress):
                    nonlocal cm_step
                    cm_progress = 10.0 + progress * 85.0
                    _update_progress(min(cm_progress, 95.0), f"生成中... {cm_step+1}/{num_inference_steps}")
                    cm_step += 1
                
                image = generator.generate(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    callback=cm_callback if progress_callback else None,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的模型: {model_name}")
                
            _update_progress(98.0, "后处理图像...")
            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用图像质量增强（色彩和合理性优化）
            image = self._enhance_image_quality(image, model_name)
            
            _update_progress(100.0, "图像生成完成")
            print("[INFO] 图像生成完成")
            
        except Exception as e:
            print(f"[ERROR] 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            _update_progress(0.0, f"错误: {str(e)}")
            raise RuntimeError(f"生成图像时出错: {str(e)}")
        
        return image
    
    def generate_batch(self, prompts: List[str], model_name: str, num_inference_steps: int = 25,
                      guidance_scale: float = 7.5, height: int = 512, width: int = 512,
                      seeds: Optional[List[int]] = None, device: Optional[str] = None,
                      progress_callback: Optional[Callable[[float, str, int, int], None]] = None,
                      **kwargs) -> List[Image.Image]:
        """
        批量生成图像
        
        Args:
            prompts: 文本提示词列表
            model_name: 模型名称
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seeds: 随机种子列表
            device: 计算设备
            progress_callback: 进度回调函数，接收进度百分比(0-100)、状态信息、当前索引和总数
        
        Returns:
            生成的图像列表
        """
        if not prompts:
            return []
        
        if seeds is None:
            seeds = [None] * len(prompts)
        elif len(seeds) != len(prompts):
            raise ValueError("种子列表长度必须与提示词列表相同")
        
        images = []
        total = len(prompts)
        
        # 为批量生成创建进度回调
        def batch_progress_callback(progress: float, status: str):
            """批量处理进度回调包装器"""
            if progress_callback:
                try:
                    # 计算全局进度
                    current_idx = len(images)
                    global_progress = (current_idx / total) * 100.0 + (progress / total)
                    progress_callback(min(global_progress, 99.0), status, current_idx, total)
                except Exception as e:
                    print(f"[WARNING] 批量进度回调执行失败: {e}")
        
        # 逐个生成图像
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            try:
                print(f"[INFO] 批量生成: {i+1}/{total}, 提示词: {prompt[:30]}...")
                # 为单个图像生成添加进度回调
                image = self.generate(
                    prompt=prompt,
                    model_name=model_name,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    seed=seed,
                    device=device,
                    progress_callback=batch_progress_callback,
                    **kwargs
                )
                images.append(image)
                
                # 更新进度到100%表示当前图像完成
                if progress_callback:
                    try:
                        global_progress = ((i + 1) / total) * 100.0
                        progress_callback(min(global_progress, 99.0), f"完成 {i+1}/{total}", i + 1, total)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"[ERROR] 批量生成图像 {i+1} 失败: {e}")
                images.append(None)  # 失败时添加None占位
        
        return images
    
    def save_image(self, image: Image.Image, prompt: str, model_name: str, output_dir: str = "output") -> str:
        """
        保存图像
        
        Args:
            image: 图像对象
            prompt: 提示词
            model_name: 模型名称
            output_dir: 输出目录
        
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        model_safe_name = model_name.replace('/', '_').replace('-', '_')
        filename = f"{timestamp}_{model_safe_name}_{safe_filename}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath, "PNG")
        return filepath

