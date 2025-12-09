#!/usr/bin/env python3
"""
一致性检测服务模块
支持CLIP和ITSC-GAN检测
"""

from typing import Tuple, Dict, Any
from PIL import Image
from semantic_consistency_detector import SemanticConsistencyDetector


class ConsistencyDetectionService:
    """一致性检测服务"""
    
    def __init__(self, device: str = "cuda"):
        """初始化服务"""
        # 使用延迟加载模式，避免启动时卡住
        # CLIP模型将在首次使用时自动加载
        self.detector = SemanticConsistencyDetector(
            device=device,
            use_simple_mode=False,
            lazy_load_clip=True  # 改为延迟加载，启动更快
        )
    
    def detect(
        self,
        image: Image.Image,
        prompt: str,
        threshold: float = 0.3,
        model_name: str = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测语义一致性
        
        Args:
            image: 图像对象
            prompt: 文本提示词
            threshold: 一致性阈值
            model_name: 模型名称（用于计算模型特定的分数）
        
        Returns:
            (是否一致, 一致性分数, 详细结果)
        """
        is_consistent, score, detail = self.detector.detect_consistency(
            image, prompt, threshold, model_name=model_name
        )
        
        # 提取CLIP和融合分数
        if isinstance(detail, dict):
            clip_score = detail.get('clip_score', score)
            fused_score = detail.get('fused_score', score)
            # 获取模型特定的分数
            model_specific_score = detail.get('model_specific_score', score)
        else:
            clip_score = score
            fused_score = score
            model_specific_score = score
        
        result = {
            'is_consistent': is_consistent,
            'overall_score': model_specific_score,  # 使用模型特定分数
            'clip_score': clip_score,
            'fused_score': fused_score,
            'model_specific_score': model_specific_score,
            'model_name': model_name,
            'threshold': threshold,
            'detail': detail
        }
        
        return is_consistent, model_specific_score, result

