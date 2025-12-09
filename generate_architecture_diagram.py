#!/usr/bin/env python3
"""
生成系统架构图
"""
from PIL import Image, ImageDraw, ImageFont
import os

# 创建画布
width, height = 1600, 2000
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# 定义颜色
colors = {
    'input': '#E3F2FD',
    'generation': '#F3E5F5',
    'detection': '#E8F5E9',
    'analysis': '#FFF3E0',
    'output': '#FCE4EC',
    'sd': '#F093FB',
    'clip': '#4FACFE',
    'itsc': '#43E97B',
    'text': '#333333',
    'border': '#666666'
}

# 定义字体（使用默认字体）
try:
    title_font = ImageFont.truetype("arial.ttf", 40)
    header_font = ImageFont.truetype("arial.ttf", 32)
    text_font = ImageFont.truetype("arial.ttf", 24)
    small_font = ImageFont.truetype("arial.ttf", 18)
except:
    # 如果找不到字体，使用默认字体
    title_font = ImageFont.load_default()
    header_font = ImageFont.load_default()
    text_font = ImageFont.load_default()
    small_font = ImageFont.load_default()

def draw_rounded_rect(draw, x, y, w, h, fill, outline=None, radius=10):
    """绘制圆角矩形"""
    draw.rectangle([x, y, x+w, y+h], fill=fill, outline=outline, width=2)
    # 简化版：直接绘制矩形（圆角需要更复杂的实现）

def draw_module(draw, x, y, w, h, text, subtext, color, text_color='white'):
    """绘制模块"""
    # 绘制模块背景
    draw.rectangle([x, y, x+w, y+h], fill=color, outline=colors['border'], width=2)
    
    # 绘制文本
    bbox = draw.textbbox((0, 0), text, font=text_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (w - text_w) // 2
    text_y = y + (h - text_h - 20) // 2
    draw.text((text_x, text_y), text, fill=text_color, font=text_font)
    
    # 绘制子文本
    if subtext:
        bbox2 = draw.textbbox((0, 0), subtext, font=small_font)
        subtext_w = bbox2[2] - bbox2[0]
        subtext_x = x + (w - subtext_w) // 2
        subtext_y = text_y + text_h + 5
        draw.text((subtext_x, subtext_y), subtext, fill=text_color, font=small_font)

def draw_arrow(draw, x, y, length, direction='down'):
    """绘制箭头"""
    if direction == 'down':
        # 绘制竖线
        draw.line([x, y, x, y+length], fill=colors['border'], width=3)
        # 绘制箭头
        draw.polygon([(x, y+length), (x-10, y+length-15), (x+10, y+length-15)], fill=colors['border'])

# 绘制标题
title = "图像生成与语义一致性检测系统架构图"
bbox = draw.textbbox((0, 0), title, font=title_font)
title_w = bbox[2] - bbox[0]
title_x = (width - title_w) // 2
draw.text((title_x, 30), title, fill=colors['text'], font=title_font)

y_pos = 100

# 1. 输入层
section_y = y_pos
draw.rectangle([50, section_y, width-50, section_y+150], fill=colors['input'], outline=colors['border'], width=3)
draw.text((70, section_y+20), "📝 输入层 (Input Layer)", fill=colors['text'], font=header_font)

draw_module(draw, 200, section_y+70, 400, 60, "文本提示词", "Text Prompt", colors['sd'], 'white')
draw_module(draw, 700, section_y+70, 400, 60, "参数配置", "Parameters", colors['clip'], 'white')

y_pos = section_y + 180

# 箭头
draw_arrow(draw, width//2, y_pos, 40)

y_pos += 60

# 2. 生成层
section_y = y_pos
draw.rectangle([50, section_y, width-50, section_y+300], fill=colors['generation'], outline=colors['border'], width=3)
draw.text((70, section_y+20), "🎨 图像生成层 (Generation Layer)", fill=colors['text'], font=header_font)

# 三个并行模型
module_width = 450
module_height = 200
spacing = (width - 100 - 3 * module_width) // 4
start_x = 50 + spacing

# SD基础模型
draw_module(draw, start_x, section_y+70, module_width, module_height, 
           "SD基础模型", "Stable Diffusion v1.5", colors['sd'], 'white')
draw.text((start_x + 20, section_y+280), "基础生成路径", fill=colors['text'], font=small_font)

# CLIP融合模型
start_x += module_width + spacing
draw_module(draw, start_x, section_y+70, module_width, module_height,
           "OpenAI CLIP融合", "CLIP-Enhanced Generation", colors['clip'], 'white')
draw.text((start_x + 20, section_y+280), "语义增强路径", fill=colors['text'], font=small_font)

# ITSC-GAN融合模型
start_x += module_width + spacing
draw_module(draw, start_x, section_y+70, module_width, module_height,
           "ITSC-GAN融合", "ITSC-GAN Fusion (IRAM+TEM)", colors['itsc'], 'white')
draw.text((start_x + 20, section_y+280), "一致性优化路径", fill=colors['text'], font=small_font)

y_pos = section_y + 330

# 箭头
draw_arrow(draw, width//2, y_pos, 40)

y_pos += 60

# 3. 检测层
section_y = y_pos
draw.rectangle([50, section_y, width-50, section_y+250], fill=colors['detection'], outline=colors['border'], width=3)
draw.text((70, section_y+20), "🔍 语义一致性检测层 (Consistency Detection Layer)", fill=colors['text'], font=header_font)

# 检测模块
detect_x = 200
detect_y = section_y + 70
detect_w = width - 400
detect_h = 100
draw.rectangle([detect_x, detect_y, detect_x+detect_w, detect_y+detect_h], 
              fill='#FA709A', outline=colors['border'], width=2)
draw.text((detect_x + 20, detect_y + 20), "语义一致性检测模块", fill='white', font=text_font)
draw.text((detect_x + 20, detect_y + 50), "• CLIP相似度检测  • ITSC-GAN融合检测  • 模型特定分数计算", 
         fill='white', font=small_font)

# 分数显示
score_y = detect_y + detect_h + 20
score_w = 300
score_h = 50
draw_module(draw, 300, score_y, score_w, score_h, "CLIP分数", "基础检测", '#4CAF50', 'white')
draw_module(draw, width - 300 - score_w, score_y, score_w, score_h, "融合分数", "增强检测", '#4CAF50', 'white')

y_pos = section_y + 280

# 箭头
draw_arrow(draw, width//2, y_pos, 40)

y_pos += 60

# 4. 分析层
section_y = y_pos
draw.rectangle([50, section_y, width-50, section_y+150], fill=colors['analysis'], outline=colors['border'], width=3)
draw.text((70, section_y+20), "📊 模型对比与分析层 (Comparison & Analysis Layer)", fill=colors['text'], font=header_font)

# 三个分析模块
analysis_w = 450
analysis_h = 80
analysis_spacing = (width - 100 - 3 * analysis_w) // 4
analysis_start_x = 50 + analysis_spacing

draw_module(draw, analysis_start_x, section_y+60, analysis_w, analysis_h,
           "模型对比", "Multi-Model Comparison", '#30CFD0', 'white')
analysis_start_x += analysis_w + analysis_spacing
draw_module(draw, analysis_start_x, section_y+60, analysis_w, analysis_h,
           "总结分析", "Summary Analysis", '#30CFD0', 'white')
analysis_start_x += analysis_w + analysis_spacing
draw_module(draw, analysis_start_x, section_y+60, analysis_w, analysis_h,
           "数据可视化", "Data Visualization", '#30CFD0', 'white')

y_pos = section_y + 180

# 箭头
draw_arrow(draw, width//2, y_pos, 40)

y_pos += 60

# 5. 输出层
section_y = y_pos
draw.rectangle([50, section_y, width-50, section_y+200], fill=colors['output'], outline=colors['border'], width=3)
draw.text((70, section_y+20), "📤 输出层 (Output Layer)", fill=colors['text'], font=header_font)

# 输出项
output_items = ["生成的图像", "一致性检测结果", "模型对比报告", "分析总结", "历史记录"]
output_w = 250
output_h = 60
output_spacing = 30
total_output_w = len(output_items) * output_w + (len(output_items) - 1) * output_spacing
output_start_x = (width - total_output_w) // 2

for i, item in enumerate(output_items):
    x = output_start_x + i * (output_w + output_spacing)
    draw_module(draw, x, section_y+70, output_w, output_h, item, "", '#667EEA', 'white')

# 保存图片
output_path = "系统架构图.png"
img.save(output_path)
print(f"✅ 系统架构图已生成: {output_path}")
print(f"   图片尺寸: {width} x {height} 像素")




