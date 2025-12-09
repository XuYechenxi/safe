"""
ITSC-GAN相关模块实现
包含图像区域注意力模块(IRAM)和文本信息增强模块(TEM)
基于论文: ITSC-GAN: Image-Text Semantic Consistency GAN for Text-to-Image Generation
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from PIL import Image
from typing import Optional, Tuple

class SelfAttentionModule(nn.Module):
    """
    自注意力模块(SAM)
    基于Transformer中的多头注意力和前馈网络构成
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        初始化自注意力模块
        
        Args:
            dim: 输入特征维度
            num_heads: 多头注意力的头数
            ff_dim: 前馈网络的隐藏层维度，默认与输入维度相同
            dropout: dropout率
        """
        super().__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim or dim
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [sequence_length, batch_size, embed_dim]
            
        Returns:
            增强后的特征 [sequence_length, batch_size, embed_dim]
        """
        # 多头注意力
        attention_output, _ = self.attention(x, x, x)
        attention_output = self.layernorm1(x + attention_output)
        
        # 前馈网络
        ffn_output = self.ffn(attention_output)
        output = self.layernorm2(attention_output + ffn_output)
        
        return output

class ImageRegionalAttentionModule(nn.Module):
    """
    图像区域注意力模块(IRAM)
    构建生成图像区域间的联系，使得生成目标更完整，前景、背景边界更加清晰
    基于论文: ITSC-GAN: Image-Text Semantic Consistency GAN for Text-to-Image Generation
    """
    def __init__(self, 
                 in_channels: int = 3, 
                 dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化图像区域注意力模块
        
        Args:
            in_channels: 输入图像的通道数
            dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: dropout率
        """
        super().__init__()
        
        # CNN特征提取器 - 增加更多层和残差连接
        self.cnn = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, dim//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//4),
            
            # 第二层
            nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层
            nn.Conv2d(dim//4, dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//2),
            
            # 第四层
            nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第五层
            nn.Conv2d(dim//2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            
            # 第六层
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim)
        )
        
        # 残差连接层 - 更准确的空间尺寸匹配
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim//4, dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(dim//2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim//2, dim, kernel_size=3, stride=1, padding=1)
        )
        
        # 自注意力模块
        self.self_attention = SelfAttentionModule(dim, num_heads, dropout=dropout)
        
        # 使用正弦位置编码代替可学习的位置编码
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            增强后的区域特征 [batch_size, sequence_length, embed_dim]
        """
        # CNN特征提取
        features = self.cnn(x)  # [batch_size, dim, h, w]
        
        # 生成残差特征
        residual_features = self.residual_conv(x)  # [batch_size, dim, h, w]
        
        # 添加残差连接
        features = features + residual_features
        features = F.relu(features)
        
        # 重塑为序列形式
        batch_size, dim, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(batch_size, h*w, dim)  # [batch_size, h*w, dim]
        
        # 转置为注意力模块需要的形状 [sequence_length, batch_size, embed_dim]
        features = features.permute(1, 0, 2)
        
        # 添加正弦位置编码
        seq_len = features.shape[0]
        pos_emb = create_position_encoding(seq_len, dim).to(features.device)
        features += pos_emb
        
        # 自注意力增强
        attention_output = self.self_attention(features)
        
        # 转置回原始形状 [batch_size, sequence_length, embed_dim]
        attention_output = attention_output.permute(1, 0, 2)
        
        return attention_output

class CrossAttentionModule(nn.Module):
    """
    交叉注意力模块(CAM)
    包含2个连续的多头注意力，且子层是并行解码多个对象
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化交叉注意力模块
        
        Args:
            dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: dropout率
        """
        super().__init__()
        
        # 自注意力（用于文本特征）
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 交叉注意力（文本-图像）
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询向量 [sequence_length_q, batch_size, embed_dim]
            key: 键向量 [sequence_length_k, batch_size, embed_dim]
            value: 值向量 [sequence_length_v, batch_size, embed_dim]
            
        Returns:
            增强后的特征 [sequence_length_q, batch_size, embed_dim]
        """
        # 文本自注意力
        self_attention_output, _ = self.self_attention(query, query, query)
        self_attention_output = self.layernorm1(query + self_attention_output)
        
        # 文本-图像交叉注意力
        cross_attention_output, _ = self.cross_attention(
            query=self_attention_output, 
            key=key, 
            value=value
        )
        cross_attention_output = self.layernorm2(self_attention_output + cross_attention_output)
        
        return cross_attention_output

class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层(CAL)
    用于实现文本特征到图像特征的动态表示
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8, 
                 dropout: float = 0.1):
        """
        初始化交叉注意力层
        
        Args:
            dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: dropout率
        """
        super().__init__()
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 层归一化
        self.layernorm = nn.LayerNorm(dim)
        
        # 线性层用于最终输出
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: 查询向量 [sequence_length_q, batch_size, embed_dim]
            key: 键向量 [sequence_length_k, batch_size, embed_dim]
            value: 值向量 [sequence_length_v, batch_size, embed_dim]
            
        Returns:
            增强后的特征 [sequence_length_q, batch_size, embed_dim]
        """
        # 多头注意力
        attention_output, _ = self.attention(query, key, value)
        attention_output = self.layernorm(attention_output)
        
        # 线性层输出
        output = self.linear(attention_output)
        
        return output

class TextEnhancementModule(nn.Module):
    """
    文本信息增强模块(TEM)
    利用生成图像增强文本特征的表征能力，进而增强生成图像与文本描述的语义一致性
    基于论文: ITSC-GAN: Image-Text Semantic Consistency GAN for Text-to-Image Generation
    """
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1):
        # 如果没有提供ff_dim，使用dim的2倍作为默认值
        if ff_dim is None:
            ff_dim = dim * 2
        """
        初始化文本信息增强模块
        
        Args:
            dim: 特征维度
            num_heads: 多头注意力的头数
            ff_dim: 前馈网络隐藏层维度
            dropout: dropout率
        """
        super().__init__()
        
        # 交叉注意力模块（CAM）
        self.cam = CrossAttentionModule(dim, num_heads, dropout)
        
        # 交叉注意力层（CAL）
        self.cal = CrossAttentionLayer(dim, num_heads, dropout)
        
        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.layernorm = nn.LayerNorm(dim)
    
    def forward(self, 
                text_features: torch.Tensor, 
                image_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text_features: 文本特征 [sequence_length_t, batch_size, embed_dim]
            image_features: 图像特征 [sequence_length_i, batch_size, embed_dim]
            
        Returns:
            增强后的文本特征 [sequence_length_t, batch_size, embed_dim]
        """
        # 保存原始文本特征用于残差连接
        original_text_features = text_features
        
        # 交叉注意力模块（CAM）
        cam_output = self.cam(
            query=text_features, 
            key=image_features, 
            value=image_features
        )
        
        # 残差连接
        cam_output = text_features + cam_output
        cam_output = self.layernorm(cam_output)
        
        # 交叉注意力层（CAL）
        cal_output = self.cal(
            query=text_features, 
            key=cam_output, 
            value=cam_output
        )
        
        # 残差连接
        cal_output = text_features + cal_output
        cal_output = self.layernorm(cal_output)
        
        # 前馈网络
        ffn_output = self.ffn(cal_output)
        
        # 最终残差连接
        final_output = original_text_features + ffn_output
        final_output = self.layernorm(final_output)
        
        return final_output

class ITSCGANGenerator(nn.Module):
    """
    ITSC-GAN生成器
    基于Stable Diffusion，集成IRAM和TEM模块，实现更一致的图像-文本语义生成
    基于论文: ITSC-GAN: Image-Text Semantic Consistency GAN for Text-to-Image Generation
    """
    def __init__(self, 
                 stable_diffusion_pipeline, 
                 dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化ITSC-GAN生成器
        
        Args:
            stable_diffusion_pipeline: 预训练的Stable Diffusion管道
            dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: dropout率
        """
        super().__init__()
        
        # 保存Stable Diffusion管道
        self.pipeline = stable_diffusion_pipeline
        
        # 初始化图像区域注意力模块(IRAM)
        self.iram = ImageRegionalAttentionModule(dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 初始化文本信息增强模块(TEM)
        self.tem = TextEnhancementModule(dim=dim, num_heads=num_heads, dropout=dropout)
        
        # 语义一致性损失权重
        self.semantic_loss_weight = 0.1
        
    def forward(self, 
                prompt: str,
                negative_prompt: str = "",
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                height: int = 512,
                width: int = 512,
                seed: Optional[int] = None) -> Tuple[Image.Image, torch.Tensor]:
        """
        前向传播
        
        Args:
            prompt: 文本提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            
        Returns:
            生成的图像和语义一致性损失
        """
        # 使用Stable Diffusion生成图像
        generator = torch.Generator(device=self.pipeline.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        # 生成图像
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pt"  # 返回PyTorch张量以便进一步处理
        )
        
        # 获取生成的图像张量
        image_tensor = result.images[0]  # [channels, height, width]
        image_tensor = image_tensor.unsqueeze(0)  # [batch_size, channels, height, width]
        image_tensor = image_tensor.to(self.pipeline.device)
        
        # 获取文本特征
        text_embeddings = self.pipeline.text_encoder(self.pipeline.tokenizer(prompt, return_tensors="pt").input_ids.to(self.pipeline.device))[0]
        text_embeddings = text_embeddings.permute(1, 0, 2)  # [sequence_length, batch_size, embed_dim]
        
        # 使用IRAM增强图像区域特征
        image_features = self.iram(image_tensor)
        
        # 调整文本特征维度以匹配TEM的输入要求
        text_features = text_embeddings
        if text_features.shape[-1] != image_features.shape[-1]:
            text_features = F.linear(text_features, torch.randn(image_features.shape[-1], text_features.shape[-1], device=self.pipeline.device))
        
        # 使用TEM增强文本特征
        enhanced_text_features = self.tem(text_features, image_features)
        
        # 计算语义一致性损失
        semantic_loss = self._compute_semantic_consistency_loss(image_features, enhanced_text_features)
        
        # 将图像张量转换回PIL图像
        generated_image = self.pipeline.numpy_to_pil(image_tensor)[0]
        
        return generated_image, semantic_loss
    
    def _compute_semantic_consistency_loss(self, 
                                           image_features: torch.Tensor, 
                                           text_features: torch.Tensor) -> torch.Tensor:
        """
        计算语义一致性损失
        
        Args:
            image_features: 图像特征 [batch_size, sequence_length, embed_dim]
            text_features: 文本特征 [sequence_length, batch_size, embed_dim]
            
        Returns:
            语义一致性损失值
        """
        # 将文本特征转置为 [batch_size, sequence_length, embed_dim]
        text_features = text_features.permute(1, 0, 2)
        
        # 池化特征
        pooled_image_features = torch.mean(image_features, dim=1)
        pooled_text_features = torch.mean(text_features, dim=1)
        
        # 计算余弦相似度
        cosine_similarity = torch.cosine_similarity(pooled_image_features, pooled_text_features, dim=-1)
        
        # 语义一致性损失（最大化相似度）
        semantic_loss = 1.0 - cosine_similarity.mean()
        
        return semantic_loss

class ITSCGANDiscriminator(nn.Module):
    """
    ITSC-GAN判别器
    用于判断生成图像的真实性以及与文本描述的语义一致性
    基于论文: ITSC-GAN: Image-Text Semantic Consistency GAN for Text-to-Image Generation
    """
    def __init__(self, 
                 image_channels: int = 3,
                 image_feature_dim: int = 256,
                 text_feature_dim: int = 768,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化ITSC-GAN判别器
        
        Args:
            image_channels: 图像通道数
            image_feature_dim: 图像特征维度
            text_feature_dim: 文本特征维度
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力的头数
            dropout: dropout率
        """
        super().__init__()
        
        # 保存参数以便后续使用
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.hidden_dim = hidden_dim
        
        # 图像特征提取网络
        self.image_encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 第四层
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 第五层
            nn.Conv2d(512, image_feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_feature_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 文本特征提取网络
        self.text_encoder = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 交叉注意力模块，用于融合图像和文本特征
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 真实性判断输出层
        self.realism_output = nn.Linear(hidden_dim//2, 1)
        
        # 语义一致性判断输出层
        self.consistency_output = nn.Linear(hidden_dim//2, 1)
        
    def forward(self, 
                image: torch.Tensor, 
                text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            image: 输入图像 [batch_size, channels, height, width]
            text_features: 文本特征 [batch_size, sequence_length, text_feature_dim]
            
        Returns:
            真实性判断和语义一致性判断
        """
        # 提取图像特征
        image_features = self.image_encoder(image)  # [batch_size, image_feature_dim, h, w]
        
        # 池化图像特征
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1))  # [batch_size, image_feature_dim, 1, 1]
        image_features = image_features.view(image_features.size(0), -1)  # [batch_size, image_feature_dim]
        
        # 提取文本特征
        # 处理不同形状的输入
        if len(text_features.shape) == 2:
            # 如果输入是 [batch_size, text_feature_dim]，直接使用
            text_features = text_features
        elif len(text_features.shape) == 3:
            # 如果输入是 [batch_size, sequence_length, text_feature_dim]，进行平均池化
            text_features = torch.mean(text_features, dim=1)  # [batch_size, text_feature_dim]
        else:
            raise ValueError(f"Unexpected text_features shape: {text_features.shape}")
        
        # 确保设备匹配
        text_features = text_features.to(image.device)
        
        # 确保维度匹配
        expected_dim = self.text_feature_dim
        if text_features.shape[-1] != expected_dim:
            # 如果维度不匹配，添加一个线性映射层
            if not hasattr(self, 'text_feature_adapter'):
                actual_dim = text_features.shape[-1]
                # 使用 register_module 确保适配器被正确注册为子模块
                adapter = nn.Linear(actual_dim, expected_dim).to(image.device)
                self.register_module('text_feature_adapter', adapter)
            text_features = self.text_feature_adapter(text_features)
        
        text_features = self.text_encoder(text_features)  # [batch_size, hidden_dim]
        
        # 交叉注意力融合
        # 调整形状以适应多头注意力
        image_features_attention = image_features.unsqueeze(0)  # [1, batch_size, image_feature_dim]
        text_features_attention = text_features.unsqueeze(0)  # [1, batch_size, hidden_dim]
        
        # 确保特征维度匹配
        if image_features_attention.shape[-1] != text_features_attention.shape[-1]:
            image_features_attention = F.linear(image_features_attention, torch.randn(self.hidden_dim, self.image_feature_dim, device=image.device))
        
        # 执行交叉注意力
        attention_output, _ = self.cross_attention(
            query=image_features_attention,
            key=text_features_attention,
            value=text_features_attention
        )
        
        # 重塑注意力输出
        attention_output = attention_output.squeeze(0)  # [batch_size, hidden_dim]
        
        # 融合特征
        fused_features = torch.cat([image_features, attention_output], dim=1)  # [batch_size, image_feature_dim + hidden_dim]
        fused_features = self.fusion(fused_features)  # [batch_size, hidden_dim//2]
        
        # 真实性判断
        realism = torch.sigmoid(self.realism_output(fused_features))  # [batch_size, 1]
        
        # 语义一致性判断
        consistency = torch.sigmoid(self.consistency_output(fused_features))  # [batch_size, 1]
        
        return realism, consistency

# 工具函数
def create_position_encoding(sequence_length: int, dim: int) -> torch.Tensor:
    """
    创建位置编码
    
    Args:
        sequence_length: 序列长度
        dim: 特征维度
        
    Returns:
        位置编码 [sequence_length, 1, dim]
    """
    position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    
    position_encoding = torch.zeros(sequence_length, dim)
    position_encoding[:, 0::2] = torch.sin(position * div_term)
    position_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return position_encoding.unsqueeze(1)  # [sequence_length, 1, dim]