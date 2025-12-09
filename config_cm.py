"""
一致性模型配置文件
基于OpenAI一致性模型和CLIP的配置
"""

import os

# ==================== 模型配置 ====================
# 一致性模型配置
CONSISTENCY_MODEL_PATH = os.getenv("CONSISTENCY_MODEL_PATH", None)  # 预训练模型路径
CONSISTENCY_MODEL_IMAGE_SIZE = int(os.getenv("CONSISTENCY_MODEL_IMAGE_SIZE", 256))  # 图像尺寸

# CLIP模型配置
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")  # CLIP模型名称
# 可选: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"

# ==================== 设备配置 ====================
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ==================== 生成配置 ====================
# 一致性模型通常只需要1-4步即可生成高质量图像
DEFAULT_NUM_STEPS = int(os.getenv("CM_NUM_STEPS", 1))  # 默认单步生成
DEFAULT_GUIDANCE_SCALE = float(os.getenv("CM_GUIDANCE_SCALE", 7.5))  # 引导强度
DEFAULT_HEIGHT = int(os.getenv("CM_HEIGHT", 256))  # 默认高度
DEFAULT_WIDTH = int(os.getenv("CM_WIDTH", 256))  # 默认宽度

# ==================== 检测配置 ====================
DEFAULT_THRESHOLD = 0.3  # 语义一致性阈值
DETECTOR_USE_CLIP = True  # 使用CLIP进行语义一致性检测
DETECTOR_USE_SIMPLE_MODE = False  # 使用完整CLIP模式

# ==================== 输出配置 ====================
OUTPUT_DIR = "output"
LOG_DIR = "logs"

# ==================== 性能优化 ====================
USE_FP16 = True  # 使用半精度（需要CUDA）
ENABLE_ATTENTION_SLICING = True  # 减少显存占用
ENABLE_MEMORY_EFFICIENT = True

# ==================== 系统配置 ====================
# 一致性模型的优势
CM_ADVANTAGES = {
    "speed": "单步生成，速度提升250倍+",
    "quality": "质量与多步扩散模型相当",
    "efficiency": "内存占用更少",
    "real_time": "支持实时应用"
}

