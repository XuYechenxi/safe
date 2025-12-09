#!/usr/bin/env python3
"""
生成图像伪造检测系统架构图（按照用户提供的图片样式）
"""
from PIL import Image, ImageDraw, ImageFont
import os

# 创建画布
width, height = 1800, 800
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# 定义颜色
colors = {
    'input': '#F5F5F5',
    'watermark': '#BBDEFB',
    'permutation': '#B3E5FC',
    'ps': '#FFEBEE',
    'deepfake': '#E8F5E9',
    'watermark_extraction': '#C8E6C9',
    'tampering': '#FFECB3',
    'morphological': '#FFCDD2',
    'text': '#333333',
    'border': '#212121',
    'arrow': '#757575'
}

# 定义字体（使用默认字体）
try:
    title_font = ImageFont.truetype("arial.ttf", 32)
    module_font = ImageFont.truetype("arial.ttf", 18)
    label_font = ImageFont.truetype("arial.ttf", 16)
except:
    # 如果找不到字体，使用默认字体
    title_font = ImageFont.load_default()
    module_font = ImageFont.load_default()
    label_font = ImageFont.load_default()

def draw_rounded_rect(draw, x, y, w, h, fill, outline=None, radius=8):
    """绘制圆角矩形"""
    draw.rectangle([x, y, x+w, y+h], fill=fill, outline=outline, width=2)

# 绘制模块（带圆角和标签）
def draw_module(draw, x, y, w, h, label, fill, outline=None):
    draw_rounded_rect(draw, x, y, w, h, fill=fill, outline=outline, radius=8)
    # 居中绘制标签
    bbox = draw.textbbox((0, 0), label, font=module_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (w - text_w) // 2
    text_y = y + (h - text_h) // 2
    draw.text((text_x, text_y), label, fill='black', font=module_font)

# 绘制箭头（支持水平和垂直方向）
def draw_arrow(draw, x1, y1, x2, y2, width=2):
    # 绘制直线
    draw.line([(x1, y1), (x2, y2)], fill=colors['arrow'], width=width)
    # 计算箭头角度
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length > 0:
        # 绘制箭头
        arrow_size = 15
        angle1 = -15
        angle2 = 15
        import math
        if dx != 0:
            angle = math.atan2(dy, dx)
        else:
            angle = math.pi / 2 if dy > 0 else -math.pi / 2
        
        # 箭头端点
        arrow1_x = x2 - arrow_size * math.cos(angle + math.radians(angle1))
        arrow1_y = y2 - arrow_size * math.sin(angle + math.radians(angle1))
        arrow2_x = x2 - arrow_size * math.cos(angle - math.radians(angle2))
        arrow2_y = y2 - arrow_size * math.sin(angle - math.radians(angle2))
        
        # 绘制箭头
        draw.polygon([(x2, y2), (arrow1_x, arrow1_y), (arrow2_x, arrow2_y)], fill=colors['arrow'])

# 绘制图像块（带边框的矩形）
def draw_image_block(draw, x, y, w, h, label=None):
    draw.rectangle([x, y, x+w, y+h], fill='#FFFFFF', outline=colors['border'], width=2)
    if label:
        draw.text((x + 5, y + 5), label, fill=colors['text'], font=label_font)

# 绘制模型标志
def draw_model_logo(draw, x, y, w, h, text):
    draw.rectangle([x, y, x+w, y+h], fill='#4285F4', outline=colors['border'], width=2)
    draw.text((x + w//2 - 20, y + h//2 - 10), text, fill='white', font=title_font)

# 主绘制函数
def main():
    # 绘制输入图像
    draw_image_block(draw, 50, 100, 100, 100, "y_ori")
    
    # 左侧水印模块路径
    # 水印提取模块
    draw_module(draw, 200, 100, 150, 100, "Watermark Extraction Module", fill=colors['watermark'], outline=colors['border'])
    draw.text((205, 150), "Watermark Module", fill=colors['text'], font=label_font)
    
    # 水印嵌入模块
    draw_module(draw, 380, 100, 150, 100, "Permutation Module", fill=colors['permutation'], outline=colors['border'])
    
    # 绘制水印图像
    draw_image_block(draw, 560, 100, 100, 100, "x_def")
    
    # 水平箭头连接输入到水印模块
    draw_arrow(draw, 150, 150, 200, 150)
    draw_arrow(draw, 350, 150, 380, 150)
    draw_arrow(draw, 530, 150, 560, 150)
    
    # 右侧伪造检测路径
    # 传统伪造检测 (Photoshop)
    draw_model_logo(draw, 750, 100, 80, 80, "Ps")
    draw.text((730, 180), "Traditional forgery", fill=colors['text'], font=label_font)
    
    # Deepfake 检测
    draw_arrow(draw, 830, 150, 880, 150)
    draw_module(draw, 880, 100, 120, 100, "Deepfakes", fill=colors['deepfake'], outline=colors['border'])
    
    # 绘制处理后的图像
    draw_image_block(draw, 690, 300, 100, 100, "x_imp")
    
    # 从 x_def 到 传统伪造检测的箭头
    draw_arrow(draw, 610, 150, 750, 150)
    
    # 从传统伪造检测到 x_imp 的箭头
    draw_arrow(draw, 790, 180, 740, 300)
    
    # 从 Deepfakes 到 x_imp 的箭头 (虚线表示)
    draw_arrow(draw, 990, 150, 790, 300)
    
    # 右侧水印提取模块
    draw_module(draw, 830, 300, 180, 100, "Watermark Extraction", fill=colors['watermark_extraction'], outline=colors['border'])
    
    # 绘制水印图像
    draw_image_block(draw, 1040, 300, 100, 100, "M_wm")
    
    # 篡改本地化模块
    draw_module(draw, 1040, 450, 180, 100, "Tampering Localization", fill=colors['tampering'], outline=colors['border'])
    draw.text((1010, 510), "Tampering localization for", fill=colors['text'], font=label_font)
    draw.text((1020, 530), "traditional forgery", fill=colors['text'], font=label_font)
    
    # 形态学后处理模块
    draw_module(draw, 1040, 600, 180, 100, "Morphological Post-", fill=colors['morphological'], outline=colors['border'])
    draw.text((1040, 630), "processing module", fill=colors['text'], font=label_font)
    
    # 最终输出图像
    draw_image_block(draw, 1250, 600, 100, 100, "M")
    draw.text((1255, 605), "M", fill=colors['text'], font=title_font)
    
    # 连接箭头
    draw_arrow(draw, 790, 350, 830, 350)
    draw_arrow(draw, 1010, 350, 1040, 350)
    draw_arrow(draw, 1140, 350, 1140, 450)
    draw_arrow(draw, 1130, 500, 1130, 600)
    draw_arrow(draw, 1130, 650, 1250, 650)
    
    # 从 Deepfakes 到篡改本地化的箭头
    draw_arrow(draw, 940, 200, 1130, 450)
    draw.text((980, 300), "Proactive disruption", fill=colors['text'], font=label_font)
    draw.text((990, 320), "for Deepfake", fill=colors['text'], font=label_font)
    
    # 保存图像
    output_path = "伪造检测系统架构图.png"
    img.save(output_path)
    print(f"✅ 伪造检测系统架构图已生成: {output_path}")
    print(f"   图片尺寸: {width} x {height} 像素")

if __name__ == "__main__":
    main()
