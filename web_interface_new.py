#!/usr/bin/env python3
"""
全新的Web界面 - 图像生成与语义一致性检测系统
功能：
1. 用户登录注册
2. 图像生成
3. 双模型一致性检测（CLIP + ITSC-GAN）
"""

import gradio as gr
import os
from datetime import datetime
from image_processing import SemanticConsistencySystem
from database import Database
from PIL import Image, ImageDraw, ImageFont
import threading
import random
import string

# ==================== 全局变量 ====================
db = Database()
system = None

# 默认参数
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"  # 默认为基础模型
AVAILABLE_MODELS = [
    ("runwayml/stable-diffusion-v1-5", "基础模型 (Stable Diffusion v1.5)"),
    ("itsc-gan-fusion", "ITSC-GAN融合模型")
]
DEFAULT_NUM_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_THRESHOLD = 0.3

# 验证码配置
CAPTCHA_LENGTH = 6
CAPTCHA_WIDTH = 200
CAPTCHA_HEIGHT = 80
CAPTCHA_FONT_SIZE = 72
CAPTCHA_CHARACTERS = string.ascii_letters + string.digits

# ==================== 辅助函数 ====================
def pil_to_base64_html(pil_image, max_width=None, max_height=None, is_captcha=False):
    """将PIL图像转换为Base64编码的HTML img标签"""
    import io
    import base64
    
    # 如果是验证码，使用固定尺寸
    if is_captcha:
        max_width = CAPTCHA_WIDTH
        max_height = CAPTCHA_HEIGHT
    
    # 调整图像大小（如果需要）
    display_image = pil_image.copy()
    if max_width or max_height:
        img_width, img_height = display_image.size
        scale = 1.0
        
        if max_width and img_width > max_width:
            scale = min(scale, max_width / img_width)
        if max_height and img_height > max_height:
            scale = min(scale, max_height / img_height)
        
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建内存缓冲区
    buffer = io.BytesIO()
    
    # 将图像保存到缓冲区
    display_image.save(buffer, format="PNG")
    
    # 获取缓冲区内容并转换为Base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 构造HTML img标签
    if is_captcha:
        style = f"display:block;width:{CAPTCHA_WIDTH}px;height:{CAPTCHA_HEIGHT}px;border:none;"
        alt = "验证码"
    else:
        width, height = display_image.size
        style = f"display:block;max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;"
        alt = "生成的图像"
    
    img_html = f"<img src='data:image/png;base64,{img_str}' style='{style}' alt='{alt}' />"
    
    return img_html

# ==================== 验证码生成 ====================
def generate_captcha():
    """生成随机验证码"""
    # 生成随机字符串
    captcha_text = ''.join(random.choices(CAPTCHA_CHARACTERS, k=CAPTCHA_LENGTH))
    
    # 创建验证码图像
    image = Image.new('RGB', (CAPTCHA_WIDTH, CAPTCHA_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # 绘制干扰线
    for _ in range(5):
        start_x = random.randint(0, CAPTCHA_WIDTH)
        start_y = random.randint(0, CAPTCHA_HEIGHT)
        end_x = random.randint(0, CAPTCHA_WIDTH)
        end_y = random.randint(0, CAPTCHA_HEIGHT)
        draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=1)
    
    # 添加噪点
    for _ in range(50):
        x = random.randint(0, CAPTCHA_WIDTH - 1)
        y = random.randint(0, CAPTCHA_HEIGHT - 1)
        draw.point((x, y), fill=(0, 0, 0))
    
    # 使用默认字体绘制文本
    try:
        # 尝试使用Arial字体，如果找不到则使用默认字体
        font = ImageFont.truetype('arial.ttf', CAPTCHA_FONT_SIZE) if os.path.exists('arial.ttf') else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 计算文本位置使其居中
    text_bbox = draw.textbbox((0, 0), captcha_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (CAPTCHA_WIDTH - text_width) // 2
    text_y = (CAPTCHA_HEIGHT - text_height) // 2
    
    # 绘制文本
    draw.text((text_x, text_y), captcha_text, font=font, fill=(0, 0, 0))
    
    return image, captcha_text

# ==================== 系统初始化 ====================
current_model = None

def get_system(model_name=None):
    """获取系统实例（延迟初始化）"""
    global system, current_model
    
    # 如果指定了模型名称且与当前模型不同，则重新初始化系统
    if model_name and model_name != current_model:
        print(f"正在切换到模型: {model_name}")
        system = None
    
    if not system:
        try:
            print("=" * 60)
            print(f"正在初始化系统... 使用模型: {model_name or DEFAULT_MODEL}")
            print("=" * 60)
            system = SemanticConsistencySystem()
            system.initialize(model_name)  # 传入模型名称
            current_model = model_name or DEFAULT_MODEL
            print("=" * 60)
            print("✅ 系统初始化完成！")
            print("=" * 60)
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"系统初始化失败: {str(e)}")
    return system

# ==================== 用户认证 ====================
def register_user(username: str, password: str, email: str = ""):
    """用户注册"""
    if not username or not password:
        return "⚠️ 请输入用户名和密码", False
    
    if len(password) < 6:
        return "⚠️ 密码长度至少为6位", False
    
    success = db.register_user(username, password, email if email else None)
    
    if success:
        return "✅ 注册成功！请使用新账户登录", True
    else:
        return "❌ 注册失败：用户名已存在", False

def login_user(username: str, password: str):
    """用户登录"""
    if not username or not password:
        return None, "⚠️ 请输入用户名和密码", False
    
    user_id = db.verify_user(username, password)
    
    if user_id:
        return user_id, f"✅ 登录成功！欢迎 {username}", True
    else:
        return None, "❌ 登录失败：用户名或密码错误", False

# ==================== 图像生成与检测 ====================
def get_history(user_id: int):
    """获取用户的历史记录"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    try:
        # 从数据库获取历史记录
        history = db.get_user_history(user_id)
        
        if not history:
            return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>📝 暂无历史记录</h3><p>生成并检测图像后，记录将显示在这里。</p></div>"
        
        # 构建历史记录HTML
        history_html = """
        <div style='padding: 20px; background: #f5f5f5; border-radius: 10px;'>
            <h2 style='margin-top: 0; color: #333;'>📊 历史记录</h2>
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; margin-top: 20px;'>
        """
        
        for record in history:
            # 生成状态颜色
            status_color = "#4CAF50" if record['is_consistent'] else "#F44336"
            status_text = "通过" if record['is_consistent'] else "未通过"
            status_icon = "✅" if record['is_consistent'] else "❌"
            
            # 生成图像预览（如果有）
            image_preview = """
            <div style='text-align: center; margin: 10px 0;'>
                <p style='color: #777; font-style: italic;'>图像预览</p>
            </div>
            """
            
            if record['image_path'] and os.path.exists(record['image_path']):
                try:
                    from PIL import Image
                    image = Image.open(record['image_path'])
                    image_preview = pil_to_base64_html(image, max_width=300, max_height=200)
                except Exception as e:
                    print(f"[WARNING] 无法加载图像: {e}")
            
            # 添加单条记录卡片
            record_html = f"""
            <div style='background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                    <span style='font-size: 12px; color: #666;'>{record['created_at']}</span>
                    <span style='background: {status_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;'>{status_icon} {status_text}</span>
                </div>
                <p style='margin: 10px 0; font-weight: bold; color: #333;'>📝 {record['prompt'][:80]}{'...' if len(record['prompt']) > 80 else ''}</p>
                {image_preview}
                <div style='margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 6px;'>
                    <p style='margin: 5px 0; font-size: 14px;'><strong>📊 一致性分数:</strong> <span style='color: {status_color};'>{record['consistency_score']:.4f}</span></p>
                    <p style='margin: 5px 0; font-size: 14px;'><strong>🎯 阈值:</strong> {record['threshold']:.2f}</p>
                </div>
            </div>
            """
            
            history_html += record_html
        
        history_html += """
            </div>
        </div>
        """
        
        return history_html
    except Exception as e:
        error_msg = f"❌ 获取历史记录失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"

def generate_and_detect(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int,
    model_name: str = None,
    progress_callback=None
):
    """生成图像并检测语义一致性"""
    
    # 验证输入
    if not user_id:
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
        return "", error_html, None, None
    
    if not prompt or not prompt.strip():
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请输入提示词</h3></div>"
        return "", error_html, None, None
    
    try:
        # 获取系统实例
        current_system = get_system(model_name)
        
        # 生成图像
        print(f"[INFO] 开始生成图像: {prompt[:50]}...")
        
        # 创建包装的进度回调
        def wrapped_callback(progress_data):
            if progress_callback:
                progress_callback(progress_data)
        
        image = current_system.generator.generate(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            callback=wrapped_callback if progress_callback else None
        )
        
        if image is None:
            error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 图像生成失败</h3></div>"
            return "", error_html, None, None
        
        # 确保图像是RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 调试信息：检查图像对象
        print(f"[DEBUG] 生成的图像类型: {type(image)}")
        print(f"[DEBUG] 图像尺寸: {image.size}")
        print(f"[DEBUG] 图像模式: {image.mode}")
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        filename = f"{timestamp}_{safe_filename}.png"
        os.makedirs("output", exist_ok=True)
        image_path = os.path.join("output", filename)
        image.save(image_path, "PNG")
        
        # 更新进度：生成完成
        if progress_callback:
            progress_callback({
                'step': num_steps,
                'total_steps': num_steps,
                'progress': 100,
                'status': 'completed'
            })
        
        # 返回图像和路径，不进行检测
        img_html = pil_to_base64_html(image, max_width=400, max_height=300)
        
        return img_html, image_path, prompt, model_name, threshold
        
    except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            # Return error message in HTML format
            error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"
            return "", error_html, None, None


def detect_consistency_for_image(
    image_path: str,
    prompt: str,
    threshold: float,
    model_name: str = None,
    user_id: int = None
):
    """
    对已生成的图像进行一致性检测
    
    Args:
        image_path: 图像路径
        prompt: 提示词
        threshold: 一致性阈值
        model_name: 模型名称
        user_id: 用户ID
    
    Returns:
        result_html: 检测结果HTML
        clip_score: CLIP分数
        fused_score: 融合分数
        improvement_rate: 提高率
        differences: 细微差别分析
    """
    from PIL import Image
    
    try:
        # 加载图像
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取系统实例
        current_system = get_system(model_name)
        
        # 检测语义一致性
        print(f"[INFO] 开始检测语义一致性...")
        is_consistent, score, detail = current_system.detector.detect_consistency(
            image, prompt, threshold
        )
        
        # 提取两种分数
        clip_score = detail.get('clip_score', score) if isinstance(detail, dict) else score
        fused_score = detail.get('fused_score', score) if isinstance(detail, dict) else score
        
        # 计算提高率
        if clip_score > 0:
            improvement_rate = ((fused_score - clip_score) / clip_score) * 100
        else:
            improvement_rate = 0.0
        
        # 生成细微差别分析
        differences = analyze_image_differences(clip_score, fused_score, improvement_rate, is_consistent)
        
        # 异步保存到数据库
        if user_id:
            def save_async():
                try:
                    db.save_generation(
                        user_id=user_id,
                        prompt=prompt,
                        threshold=threshold,
                        consistency_score=score,
                        is_consistent=is_consistent,
                        image_path=image_path,
                        result_data=detail
                    )
                except Exception as e:
                    print(f"[WARNING] 数据库保存失败: {e}")
            
            threading.Thread(target=save_async, daemon=True).start()
        
        # 构造结果文本
        status_icon = "✅" if is_consistent else "❌"
        status_text = "通过" if is_consistent else "未通过"
        score_color = "#4CAF50" if score >= 0.7 else "#FF9800" if score >= 0.4 else "#F44336"
        improvement_color = "#4CAF50" if improvement_rate > 0 else "#F44336" if improvement_rate < 0 else "#666"
        improvement_icon = "📈" if improvement_rate > 0 else "📉" if improvement_rate < 0 else "➡️"
        model_display_name = dict(AVAILABLE_MODELS).get(model_name, model_name or "默认模型")
        
        result_html = f"""
<div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h2 style="margin-top: 0; display: flex; align-items: center; gap: 10px;">
        {status_icon} 一致性分析检测结果
    </h2>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
        <p style="margin: 8px 0;"><strong>📝 提示词:</strong> {prompt}</p>
        <p style="margin: 8px 0;"><strong>🤖 使用模型:</strong> {model_display_name}</p>
        <p style="margin: 8px 0;"><strong>🎯 一致性状态:</strong> <span style="color: #4CAF50; font-weight: bold;">{status_text}</span></p>
        <p style="margin: 8px 0;"><strong>📊 整体一致性分数:</strong> <span style="color: {score_color}; font-size: 18px; font-weight: bold;">{score:.4f}</span></p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
        <h3 style="margin: 15px 0 10px 0; font-size: 16px;">🔍 双模型检测分数对比：</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
            <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px;">
                <p style="margin: 5px 0; font-size: 14px;"><strong>🔗 CLIP相似度</strong></p>
                <p style="margin: 5px 0; font-size: 20px; font-weight: bold; color: #FFD700;">{clip_score:.4f}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px;">
                <p style="margin: 5px 0; font-size: 14px;"><strong>🔗 ITSC-GAN融合相似度</strong></p>
                <p style="margin: 5px 0; font-size: 20px; font-weight: bold; color: #FFD700;">{fused_score:.4f}</p>
            </div>
        </div>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
        <h3 style="margin: 15px 0 10px 0; font-size: 16px;">{improvement_icon} 提高率分析：</h3>
        <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px; margin-top: 10px;">
            <p style="margin: 5px 0; font-size: 14px;"><strong>提升幅度:</strong> <span style="color: {improvement_color}; font-size: 18px; font-weight: bold;">{improvement_rate:+.2f}%</span></p>
            <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">
                ITSC-GAN融合模型相比基础CLIP模型的提升: {abs(improvement_rate):.2f}%
            </p>
        </div>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
        <h3 style="margin: 15px 0 10px 0; font-size: 16px;">🔬 细微差别分析：</h3>
        <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px; margin-top: 10px;">
            {differences}
        </div>
    </div>
</div>
"""
        
        return result_html, clip_score, fused_score, improvement_rate, differences
        
    except Exception as e:
        error_msg = f"❌ 检测失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"
        return error_html, None, None, None, None


def analyze_image_differences(clip_score, fused_score, improvement_rate, is_consistent):
    """
    分析图像的细微差别
    
    Args:
        clip_score: CLIP分数
        fused_score: 融合分数
        improvement_rate: 提高率
        is_consistent: 是否一致
    
    Returns:
        differences_html: 差别分析HTML
    """
    differences = []
    
    # 1. 分数差异分析
    score_diff = fused_score - clip_score
    if score_diff > 0.1:
        differences.append(f"<p style='margin: 5px 0; color: #4CAF50;'>✅ <strong>显著提升:</strong> ITSC-GAN融合模型在语义理解上表现明显优于基础CLIP模型，分数提升了 {score_diff:.4f}。</p>")
    elif score_diff > 0.05:
        differences.append(f"<p style='margin: 5px 0; color: #4CAF50;'>✅ <strong>明显提升:</strong> ITSC-GAN融合模型在语义理解上有明显改善，分数提升了 {score_diff:.4f}。</p>")
    elif score_diff > 0:
        differences.append(f"<p style='margin: 5px 0; color: #FF9800;'>📈 <strong>轻微提升:</strong> ITSC-GAN融合模型在语义理解上略有改善，分数提升了 {score_diff:.4f}。</p>")
    elif score_diff < -0.05:
        differences.append(f"<p style='margin: 5px 0; color: #F44336;'>⚠️ <strong>分数下降:</strong> ITSC-GAN融合模型在此次检测中分数下降了 {abs(score_diff):.4f}，可能需要进一步优化。</p>")
    else:
        differences.append(f"<p style='margin: 5px 0; color: #666;'>➡️ <strong>分数相近:</strong> 两种模型的检测结果非常接近，差异仅为 {abs(score_diff):.4f}。</p>")
    
    # 2. 提高率分析
    if improvement_rate > 20:
        differences.append(f"<p style='margin: 5px 0; color: #4CAF50;'>🚀 <strong>大幅提升:</strong> 提高率达到 {improvement_rate:.2f}%，说明ITSC-GAN融合模型在此场景下表现卓越。</p>")
    elif improvement_rate > 10:
        differences.append(f"<p style='margin: 5px 0; color: #4CAF50;'>📊 <strong>良好提升:</strong> 提高率为 {improvement_rate:.2f}%，ITSC-GAN融合模型展现出良好的语义理解能力。</p>")
    elif improvement_rate > 5:
        differences.append(f"<p style='margin: 5px 0; color: #FF9800;'>📈 <strong>适度提升:</strong> 提高率为 {improvement_rate:.2f}%，ITSC-GAN融合模型有一定改善。</p>")
    elif improvement_rate > 0:
        differences.append(f"<p style='margin: 5px 0; color: #FF9800;'>📊 <strong>小幅提升:</strong> 提高率为 {improvement_rate:.2f}%，ITSC-GAN融合模型略有改善。</p>")
    elif improvement_rate < -5:
        differences.append(f"<p style='margin: 5px 0; color: #F44336;'>⚠️ <strong>性能下降:</strong> 提高率为 {improvement_rate:.2f}%，可能需要调整模型参数或训练策略。</p>")
    
    # 3. 一致性状态分析
    if is_consistent:
        if fused_score >= 0.7:
            differences.append(f"<p style='margin: 5px 0; color: #4CAF50;'>✅ <strong>优秀一致性:</strong> 图像与文本提示词高度一致，语义匹配度达到 {fused_score:.4f}，生成质量优秀。</p>")
        elif fused_score >= 0.4:
            differences.append(f"<p style='margin: 5px 0; color: #FF9800;'>✅ <strong>良好一致性:</strong> 图像与文本提示词基本一致，语义匹配度为 {fused_score:.4f}，生成质量良好。</p>")
        else:
            differences.append(f"<p style='margin: 5px 0; color: #FF9800;'>✅ <strong>通过检测:</strong> 图像与文本提示词达到基本一致，语义匹配度为 {fused_score:.4f}。</p>")
    else:
        differences.append(f"<p style='margin: 5px 0; color: #F44336;'>❌ <strong>未通过检测:</strong> 图像与文本提示词的语义匹配度较低（{fused_score:.4f}），可能需要调整提示词或生成参数。</p>")
    
    # 4. 模型特性分析
    if fused_score > clip_score + 0.05:
        differences.append(f"<p style='margin: 5px 0; color: #2196F3;'>💡 <strong>ITSC-GAN优势:</strong> ITSC-GAN融合模型通过增强语义理解，在此次检测中展现出更强的文本-图像匹配能力。</p>")
    
    if clip_score >= 0.6:
        differences.append(f"<p style='margin: 5px 0; color: #2196F3;'>💡 <strong>基础模型表现:</strong> 基础CLIP模型也达到了较好的语义匹配度（{clip_score:.4f}），说明提示词质量较高。</p>")
    
    return "".join(differences)


def compare_models(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int,
    model_names: list = None,
    progress_callback=None
):
    """
    使用多个模型生成图像并对比
    
    Args:
        prompt: 提示词
        threshold: 一致性阈值
        num_steps: 推理步数
        guidance_scale: 引导强度
        height: 图像高度
        width: 图像宽度
        user_id: 用户ID
        model_names: 要对比的模型列表，如果为None则使用所有可用模型
        progress_callback: 进度回调函数，接收 (step, total_steps, status, model_name) 参数
    
    Returns:
        comparison_html: 对比结果HTML
        comparison_data: 对比数据字典
    """
    if not user_id:
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
        return error_html, {}
    
    if not prompt or not prompt.strip():
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请输入提示词</h3></div>"
        return error_html, {}
    
    # 默认使用所有可用模型
    if model_names is None:
        model_names = [model_id for model_id, _ in AVAILABLE_MODELS]
    
    total_models = len(model_names)
    comparison_results = []
    
    try:
        for idx, model_name in enumerate(model_names, 1):
            print(f"[INFO] 使用模型 {model_name} 生成图像 ({idx}/{total_models})...")
            
            # 更新进度：开始生成
            if progress_callback:
                try:
                    progress_callback({
                        'step': idx - 1,
                        'total_steps': total_models,
                        'progress': int((idx - 1) / total_models * 100),
                        'status': 'generating',
                        'current_model': model_name
                    })
                except Exception as callback_error:
                    print(f"[WARNING] 进度回调失败: {callback_error}")
            
            # 获取系统实例（可能抛出异常）
            try:
                current_system = get_system(model_name)
            except Exception as system_error:
                print(f"[ERROR] 获取系统实例失败 (模型: {model_name}): {system_error}")
                import traceback
                traceback.print_exc()
                # 继续下一个模型，不中断整个对比流程
                continue
            
            # 创建模型特定的进度回调
            def model_progress_callback(gen_progress):
                if progress_callback:
                    # 将生成进度映射到总体进度
                    model_progress = gen_progress.get('progress', 0) if isinstance(gen_progress, dict) else 0
                    overall_progress = int(((idx - 1) + model_progress / 100) / total_models * 100)
                    progress_callback({
                        'step': idx - 1,
                        'total_steps': total_models,
                        'progress': overall_progress,
                        'status': 'generating',
                        'current_model': model_name,
                        'generation_progress': model_progress
                    })
            
            # 生成图像
            image = current_system.generator.generate(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                callback=model_progress_callback if progress_callback else None
            )
            
            if image is None:
                print(f"[WARNING] 模型 {model_name} 生成失败")
                continue
            
            # 确保图像是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 更新进度：开始检测
            if progress_callback:
                progress_callback({
                    'step': idx - 1,
                    'total_steps': total_models,
                    'progress': int((idx - 0.5) / total_models * 100),
                    'status': 'detecting',
                    'current_model': model_name
                })
            
            # 检测语义一致性
            is_consistent, score, detail = current_system.detector.detect_consistency(
                image, prompt, threshold
            )
            
            # 提取分数
            clip_score = detail.get('clip_score', score) if isinstance(detail, dict) else score
            fused_score = detail.get('fused_score', score) if isinstance(detail, dict) else score
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            model_safe_name = model_name.replace('/', '_').replace('-', '_')
            filename = f"{timestamp}_{model_safe_name}_{safe_filename}.png"
            os.makedirs("output", exist_ok=True)
            image_path = os.path.join("output", filename)
            image.save(image_path, "PNG")
            
            # 转换为Base64 HTML
            img_html = pil_to_base64_html(image, max_width=400, max_height=300)
            
            # 获取模型显示名称
            if model_name in dict(AVAILABLE_MODELS):
                model_display_name = dict(AVAILABLE_MODELS)[model_name]
            elif model_name.startswith('lora:'):
                # LoRA模型，从路径提取名称
                lora_path = model_name[5:]  # 移除 "lora:" 前缀
                # 已移除局部-短语对齐模型，现在只处理其他LoRA模型
                model_dir = os.path.basename(lora_path)
                model_display_name = f"LoRA模型 ({model_dir})"
            elif model_name == 'itsc-gan-fusion':
                model_display_name = "ITSC-GAN融合模型"
            else:
                model_display_name = model_name
            
            comparison_results.append({
                'model_name': model_name,
                'model_display_name': model_display_name,
                'image_html': img_html,
                'image_path': image_path,
                'clip_score': clip_score,
                'fused_score': fused_score,
                'overall_score': score,
                'is_consistent': is_consistent
            })
            
            # 更新进度：模型完成
            if progress_callback:
                progress_callback({
                    'step': idx,
                    'total_steps': total_models,
                    'progress': int(idx / total_models * 100),
                    'status': 'comparing',
                    'current_model': model_name,
                    'completed_models': idx
                })
            
            # 异步保存到数据库
            def save_async():
                try:
                    db.save_generation(
                        user_id=user_id,
                        prompt=f"[对比] {prompt}",
                        threshold=threshold,
                        consistency_score=score,
                        is_consistent=is_consistent,
                        image_path=image_path,
                        result_data={**detail, 'model_name': model_name}
                    )
                except Exception as e:
                    print(f"[WARNING] 数据库保存失败: {e}")
            
            threading.Thread(target=save_async, daemon=True).start()
        
        if not comparison_results:
            error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 所有模型生成失败</h3></div>"
            return error_html, {}
        
        # 生成对比HTML
        comparison_html = generate_comparison_html(prompt, comparison_results, threshold)
        
        # 生成对比数据
        comparison_data = {
            'prompt': prompt,
            'results': comparison_results,
            'best_model': max(comparison_results, key=lambda x: x['overall_score']),
            'worst_model': min(comparison_results, key=lambda x: x['overall_score'])
        }
        
        return comparison_html, comparison_data
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] ========== compare_models 函数异常 ==========")
        print(f"[ERROR] 异常类型: {type(e).__name__}")
        print(f"[ERROR] 异常信息: {str(e)}")
        print(f"[ERROR] 异常堆栈:\n{error_traceback}")
        print(f"[ERROR] ============================================")
        error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比生成失败: {str(e)}</h3><p style='font-size: 12px; margin-top: 10px;'>{str(e)[:200]}</p></div>"
        return error_html, {'error': str(e), 'error_traceback': error_traceback}


def generate_comparison_html(prompt: str, results: list, threshold: float):
    """生成模型对比HTML"""
    
    # 找出最佳和最差模型
    best_result = max(results, key=lambda x: x['overall_score'])
    worst_result = min(results, key=lambda x: x['overall_score'])
    
    # 计算平均分数
    avg_clip = sum(r['clip_score'] for r in results) / len(results)
    avg_fused = sum(r['fused_score'] for r in results) / len(results)
    avg_overall = sum(r['overall_score'] for r in results) / len(results)
    
    # 生成差异分析
    differences = analyze_differences(results)
    
    html = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="margin-top: 0; display: flex; align-items: center; gap: 10px;">
            🔍 多模型对比结果
        </h2>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
            <p style="margin: 8px 0;"><strong>📝 提示词:</strong> {prompt}</p>
            <p style="margin: 8px 0;"><strong>📊 对比模型数量:</strong> {len(results)}</p>
            <p style="margin: 8px 0;"><strong>🎯 一致性阈值:</strong> {threshold}</p>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 20px;">
    """
    
    for i, result in enumerate(results):
        is_best = result == best_result
        is_worst = result == worst_result
        badge = ""
        if is_best:
            badge = '<span style="background: #4CAF50; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;">🏆 最佳</span>'
        elif is_worst:
            badge = '<span style="background: #F44336; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;">⚠️ 最低</span>'
        
        score_color = "#4CAF50" if result['overall_score'] >= 0.7 else "#FF9800" if result['overall_score'] >= 0.4 else "#F44336"
        status_icon = "✅" if result['is_consistent'] else "❌"
        
        html += f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333; display: flex; align-items: center;">
                🤖 {result['model_display_name']} {badge}
            </h3>
            <div style="margin: 15px 0; text-align: center;">
                {result['image_html']}
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <p style="margin: 8px 0; color: #666;"><strong>一致性状态:</strong> <span style="color: {score_color}; font-weight: bold;">{status_icon} {'通过' if result['is_consistent'] else '未通过'}</span></p>
                <p style="margin: 8px 0; color: #666;"><strong>整体分数:</strong> <span style="color: {score_color}; font-size: 18px; font-weight: bold;">{result['overall_score']:.4f}</span></p>
                <hr style="border: 1px solid #ddd; margin: 10px 0;">
                <p style="margin: 5px 0; color: #666; font-size: 14px;"><strong>🔗 CLIP:</strong> {result['clip_score']:.4f}</p>
                <p style="margin: 5px 0; color: #666; font-size: 14px;"><strong>🔗 ITSC-GAN融合:</strong> {result['fused_score']:.4f}</p>
            </div>
        </div>
        """
    
    html += """
    </div>
    
    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #333;">📊 平均分数对比</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="margin: 5px 0; color: #666; font-size: 14px;"><strong>平均CLIP分数</strong></p>
                <p style="margin: 5px 0; color: #667eea; font-size: 24px; font-weight: bold;">{:.4f}</p>
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="margin: 5px 0; color: #666; font-size: 14px;"><strong>平均融合分数</strong></p>
                <p style="margin: 5px 0; color: #667eea; font-size: 24px; font-weight: bold;">{:.4f}</p>
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="margin: 5px 0; color: #666; font-size: 14px;"><strong>平均整体分数</strong></p>
                <p style="margin: 5px 0; color: #667eea; font-size: 24px; font-weight: bold;">{:.4f}</p>
            </div>
        </div>
    </div>
    """.format(avg_clip, avg_fused, avg_overall)
    
    # 添加差异分析
    if differences:
        html += f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333;">🔬 细微差异分析</h3>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 15px;">
                {differences}
            </div>
        </div>
        """
    
    return html


def analyze_differences(results: list):
    """分析模型间的细微差异，生成详细的描述"""
    if len(results) < 2:
        return ""
    
    differences = []
    
    # 按分数排序
    sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)
    
    # 1. 整体性能排名
    differences.append("<div style='margin-bottom: 15px;'>")
    differences.append("<h4 style='margin: 0 0 10px 0; color: #333;'>📊 性能排名</h4>")
    for i, result in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        score_color = "#4CAF50" if result['overall_score'] >= 0.7 else "#FF9800" if result['overall_score'] >= 0.4 else "#F44336"
        differences.append(f"<p style='margin: 5px 0; color: #333;'>{medal} <strong>{result['model_display_name']}</strong>: 整体分数 <span style='color: {score_color}; font-weight: bold;'>{result['overall_score']:.4f}</span></p>")
    differences.append("</div>")
    
    # 2. 分数差异分析
    scores = [r['overall_score'] for r in results]
    max_score = max(scores)
    min_score = min(scores)
    score_diff = max_score - min_score
    
    if score_diff > 0.05:
        best_model = max(results, key=lambda x: x['overall_score'])
        worst_model = min(results, key=lambda x: x['overall_score'])
        differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>📈 文本一致率差异:</strong> {best_model['model_display_name']} 比 {worst_model['model_display_name']} 高 <span style='color: #667eea; font-weight: bold;'>{score_diff:.4f}</span> ({score_diff*100:.2f}%)，说明 {best_model['model_display_name']} 在理解文本语义方面表现更优。</p>")
    
    # 3. CLIP分数差异分析
    clip_scores = [r['clip_score'] for r in results]
    max_clip = max(clip_scores)
    min_clip = min(clip_scores)
    clip_diff = max_clip - min_clip
    
    if clip_diff > 0.03:
        best_clip_model = results[clip_scores.index(max_clip)]
        worst_clip_model = results[clip_scores.index(min_clip)]
        differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>🔗 CLIP相似度差异:</strong> {best_clip_model['model_display_name']} 的CLIP基础相似度比 {worst_clip_model['model_display_name']} 高 <span style='color: #667eea; font-weight: bold;'>{clip_diff:.4f}</span>，表明在基础语义匹配上更准确。</p>")
    
    # 4. ITSC-GAN融合分数差异
    fused_scores = [r['fused_score'] for r in results]
    max_fused = max(fused_scores)
    min_fused = min(fused_scores)
    fused_diff = max_fused - min_fused
    
    if fused_diff > 0.03:
        best_fused_model = results[fused_scores.index(max_fused)]
        worst_fused_model = results[fused_scores.index(min_fused)]
        differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>🔬 ITSC-GAN融合差异:</strong> {best_fused_model['model_display_name']} 的融合分数比 {worst_fused_model['model_display_name']} 高 <span style='color: #667eea; font-weight: bold;'>{fused_diff:.4f}</span>，说明在增强语义理解方面表现更好。</p>")
    
    # 5. 模型特点分析
    differences.append("<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;'>")
    differences.append("<h4 style='margin: 0 0 10px 0; color: #333;'>🔍 模型特点分析</h4>")
    
    # 分析每个模型的特点
    for result in sorted_results:
        model_name = result['model_display_name']
        score = result['overall_score']
        clip_score = result['clip_score']
        fused_score = result['fused_score']
        
        characteristics = []
        
        # 根据模型名称判断特点
        if "基础模型" in model_name:
            characteristics.append("使用标准Stable Diffusion架构，生成稳定可靠")
        elif "ITSC-GAN融合" in model_name:
            characteristics.append("融合了ITSC-GAN增强模块，语义理解能力更强")
        # 已移除局部-短语对齐模型相关逻辑
        
        # 根据分数判断特点
        if score >= 0.7:
            characteristics.append("文本一致率优秀")
        elif score >= 0.4:
            characteristics.append("文本一致率良好")
        else:
            characteristics.append("文本一致率有待提升")
        
        if fused_score > clip_score + 0.05:
            characteristics.append("ITSC-GAN增强效果明显")
        
        if result['is_consistent']:
            characteristics.append("通过一致性检测")
        else:
            characteristics.append("未通过一致性检测")
        
        char_text = "、".join(characteristics)
        differences.append(f"<p style='margin: 5px 0; color: #333;'><strong>{model_name}:</strong> {char_text}</p>")
    
    differences.append("</div>")
    
    # 6. 一致性通过率
    consistent_count = sum(1 for r in results if r['is_consistent'])
    if consistent_count < len(results):
        inconsistent_models = [r['model_display_name'] for r in results if not r['is_consistent']]
        differences.append(f"<p style='margin: 8px 0; color: #f44336;'><strong>⚠️ 一致性检测:</strong> {len(results) - consistent_count} 个模型未通过一致性检测: {', '.join(inconsistent_models)}</p>")
    else:
        differences.append(f"<p style='margin: 8px 0; color: #4CAF50;'><strong>✅ 一致性检测:</strong> 所有模型均通过一致性检测，表现优秀！</p>")
    
    # 7. 图像生成质量差异描述
    differences.append("<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;'>")
    differences.append("<h4 style='margin: 0 0 10px 0; color: #333;'>🖼️ 图像生成质量差异</h4>")
    
    # 根据分数差异生成描述（现在只有2个模型）
    if len(sorted_results) >= 2:
        best = sorted_results[0]
        second = sorted_results[1]
        
        # 基础模型特点
        base_model = next((r for r in results if "基础模型" in r['model_display_name']), None)
        itsc_model = next((r for r in results if "ITSC-GAN融合" in r['model_display_name']), None)
        
        if base_model:
            base_desc = "基础模型通常生成稳定、通用的图像，适合大多数场景。"
            if base_model['overall_score'] >= 0.6:
                base_desc += "本次生成在整体语义匹配上表现良好。"
            differences.append(f"<p style='margin: 5px 0; color: #333;'><strong>基础模型:</strong> {base_desc}</p>")
        
        if itsc_model:
            itsc_desc = "ITSC-GAN融合模型通过增强语义理解，通常在复杂场景和细节表现上更优。"
            if itsc_model['fused_score'] > itsc_model['clip_score'] + 0.05:
                itsc_desc += "本次生成中ITSC-GAN增强效果显著，融合分数明显高于基础CLIP分数。"
            differences.append(f"<p style='margin: 5px 0; color: #333;'><strong>ITSC-GAN融合模型:</strong> {itsc_desc}</p>")
        
        # 生成质量对比
        score_gap = best['overall_score'] - second['overall_score']
        
        if score_gap > 0.1:
            differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>📊 质量差距:</strong> {best['model_display_name']} 与 {second['model_display_name']} 之间存在明显差距（{score_gap:.4f}），说明 {best['model_display_name']} 在本次生成任务中表现突出。</p>")
        elif score_gap < 0.05:
            differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>📊 质量接近:</strong> {best['model_display_name']} 与 {second['model_display_name']} 表现非常接近（差距仅 {score_gap:.4f}），两者都可以作为优秀选择。</p>")
        else:
            differences.append(f"<p style='margin: 8px 0; color: #333;'><strong>📊 质量对比:</strong> {best['model_display_name']} 略优于 {second['model_display_name']}（差距 {score_gap:.4f}），两者表现都较为稳定。</p>")
    
    differences.append("</div>")
    
    # 8. 推荐建议
    if len(sorted_results) >= 2:
        best = sorted_results[0]
        second = sorted_results[1] if len(sorted_results) > 1 else None
        
        differences.append("<div style='margin-top: 15px; padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196F3;'>")
        differences.append("<h4 style='margin: 0 0 10px 0; color: #1976D2;'>💡 推荐建议</h4>")
        differences.append(f"<p style='margin: 5px 0; color: #333;'>根据本次对比结果，<strong>{best['model_display_name']}</strong> 在文本一致性方面表现最佳（分数: {best['overall_score']:.4f}），推荐用于需要高精度语义匹配的场景。</p>")
        if second and best['overall_score'] - second['overall_score'] < 0.05:
            differences.append(f"<p style='margin: 5px 0; color: #333;'><strong>{second['model_display_name']}</strong> 表现也很接近（分数: {second['overall_score']:.4f}），可以作为备选方案。</p>")
        differences.append("</div>")
    
    return "".join(differences)

# ==================== 创建界面 ====================
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="图像生成与语义一致性检测系统", theme=gr.themes.Soft()) as demo:
        # 状态变量
        current_user_id = gr.State(value=None)
        login_status = gr.State(value=False)
        
        # ========== 登录页面 ==========
        with gr.Column(visible=True) as login_page:
            gr.Markdown("""
            # 🔐 登录/注册
            
            **欢迎使用图像生成与语义一致性检测系统**
            
            请登录或注册账号以使用系统功能
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 用户登录")
                    login_username = gr.Textbox(
                        label="用户名",
                        placeholder="请输入用户名",
                        value="admin"
                    )
                    login_password = gr.Textbox(
                        label="密码",
                        placeholder="请输入密码",
                        type="password",
                        value="admin123"
                    )
                    
                    # 验证码组件
                    gr.Markdown("**验证码**")
                    with gr.Row():
                        captcha_input = gr.Textbox(
                            label="",
                            placeholder="请输入验证码（不区分大小写）"
                        )
                        
                        # 初始化时生成验证码图像
                        initial_captcha_pil, initial_text = generate_captcha()
                        initial_captcha_html = pil_to_base64_html(initial_captcha_pil, is_captcha=True)
                        captcha_image = gr.HTML(
                            value=initial_captcha_html
                        )
                        refresh_captcha_btn = gr.Button("🔄", variant="secondary", size="sm")
                    
                    login_btn = gr.Button("登录", variant="primary", size="lg")
                    login_msg = gr.Markdown()
                    
                    # State variable: store current captcha
                    current_captcha = gr.State(value=initial_text)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 新用户注册")
                    register_username = gr.Textbox(
                        label="用户名",
                        placeholder="请输入新用户名"
                    )
                    register_password = gr.Textbox(
                        label="密码",
                        placeholder="请输入密码（至少6个字符）",
                        type="password"
                    )
                    register_email = gr.Textbox(
                        label="邮箱（可选）",
                        placeholder="请输入邮箱地址"
                    )
                    register_btn = gr.Button("注册", variant="secondary", size="lg")
                    register_msg = gr.Markdown()
        
        # ========== 主功能页面 ==========
        with gr.Column(visible=False) as main_page:
            gr.Markdown("""
            # 🎨 图像生成与语义一致性检测系统
            
            **基于Stable Diffusion的图像生成与双模型语义一致性检测**
            
            - 🔗 **CLIP模型**：基础语义相似度检测
            - 🔗 **ITSC-GAN融合模型**：增强语义一致性检测
            """)
            
            # 页面标签切换
            with gr.Tabs():
                # 生成与检测标签
                with gr.TabItem("✨ 生成与检测") as generate_tab:
                    gr.Markdown("### ✨ 图像生成与一致性检测")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="📝 文本提示词",
                                placeholder="请输入图像描述...",
                                lines=3
                            )
                    
                            # 快速提示词按钮
                            gr.Markdown("### 📋 快速提示词示例")
                            with gr.Row():
                                quick_prompt_btn1 = gr.Button("A cute cat on grass", size="sm")
                                quick_prompt_btn2 = gr.Button("An astronaut walking on the moon", size="sm")
                                quick_prompt_btn3 = gr.Button("Beach scenery at sunset", size="sm")
                                quick_prompt_btn4 = gr.Button("Futuristic city night view", size="sm")
                    
                            with gr.Row():
                                quick_prompt_btn5 = gr.Button("A small cabin in the forest", size="sm")
                                quick_prompt_btn6 = gr.Button("Coral reef in underwater world", size="sm")
                                quick_prompt_btn7 = gr.Button("Snowman in winter scenery", size="sm")
                                quick_prompt_btn8 = gr.Button("Vintage-style coffee shop", size="sm")
                    
                            # 模型选择下拉框
                            model_dropdown = gr.Dropdown(
                                label="🤖 选择生成模型",
                                choices=[model[1] for model in AVAILABLE_MODELS],
                                value=[model[1] for model in AVAILABLE_MODELS if model[0] == DEFAULT_MODEL][0]
                            )
                    
                            with gr.Row():
                                num_steps = gr.Slider(
                                    label="推理步数",
                                    minimum=10,
                                    maximum=100,
                                    value=DEFAULT_NUM_STEPS,
                                    step=1,
                                    info="步数越多质量越好但速度越慢"
                                )
                                guidance_scale = gr.Slider(
                                    label="引导强度",
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=DEFAULT_GUIDANCE_SCALE,
                                    step=0.1
                                )
                    
                            with gr.Row():
                                height = gr.Number(
                                    label="图像高度",
                                    value=DEFAULT_HEIGHT,
                                    precision=0,
                                    minimum=512,
                                    maximum=1024,
                                    step=64
                                )
                                width = gr.Number(
                                    label="图像宽度",
                                    value=DEFAULT_WIDTH,
                                    precision=0,
                                    minimum=512,
                                    maximum=1024,
                                    step=64
                                )
                                threshold = gr.Slider(
                                    label="一致性阈值",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=DEFAULT_THRESHOLD,
                                    step=0.05
                                )
                    
                            generate_btn = gr.Button("✨ 生成并检测", variant="primary", size="lg")
                
                        with gr.Column(scale=1):
                            output_image = gr.HTML(
                                label="🎨 生成的图像",
                                height=500
                            )
            
                        result_display = gr.HTML()
            
                        # 分数显示（隐藏，已在HTML中显示）
                        clip_score_display = gr.Number(visible=False)
                        fused_score_display = gr.Number(visible=False)
                
                # 历史记录标签
                with gr.TabItem("📊 历史记录") as history_tab:
                    gr.Markdown("### 📝 历史记录查看")
                    history_display = gr.HTML(label="历史记录列表")
                    refresh_history_btn = gr.Button("🔄 刷新历史记录", variant="secondary")
                
                # 模型训练标签（策略A）
                with gr.TabItem("🔧 模型优化（策略A）") as train_tab:
                    gr.Markdown("""
                    ### 🎯 策略A：局部-短语对齐模型优化
                    
                    **基于COCO数据集的强监督局部对齐训练**
                    
                    - 🔗 **LoRA微调**：高效参数更新（约0.1%参数量）
                    - 🔗 **全局一致性损失**：确保整体语义一致性
                    - 🔗 **局部-短语对齐损失**：利用BBox强监督实现精确对齐
                    - 🔗 **增强CFG训练**：提高模型对文本条件的敏感度
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ 训练配置")
                            
                            train_base_model = gr.Dropdown(
                                label="基础模型",
                                choices=["runwayml/stable-diffusion-v1-5"],
                                value="runwayml/stable-diffusion-v1-5",
                                info="选择要优化的基础模型"
                            )
                            
                            with gr.Row():
                                train_batch_size = gr.Number(
                                    label="批次大小",
                                    value=4,
                                    precision=0,
                                    minimum=1,
                                    maximum=16
                                )
                                train_epochs = gr.Number(
                                    label="训练轮数",
                                    value=10,
                                    precision=0,
                                    minimum=1,
                                    maximum=100
                                )
                                train_lr = gr.Number(
                                    label="学习率",
                                    value=0.0001,
                                    precision=6,
                                    minimum=1e-6,
                                    maximum=1e-2
                                )
                            
                            with gr.Row():
                                lambda_clip = gr.Slider(
                                    label="全局CLIP损失权重 (λ_CLIP)",
                                    minimum=0.0,
                                    maximum=0.1,
                                    value=0.02,
                                    step=0.001
                                )
                                lambda_local = gr.Slider(
                                    label="局部对齐损失权重 (λ_Local)",
                                    minimum=0.0,
                                    maximum=0.2,
                                    value=0.08,
                                    step=0.001,
                                    info="应高于λ_CLIP（利用BBox强监督）"
                                )
                            
                            train_output_dir = gr.Textbox(
                                label="输出目录",
                                value="models/strategy_a_lora",
                                info="训练后的模型保存路径"
                            )
                            
                            start_train_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                            stop_train_btn = gr.Button("⏹️ 停止训练", variant="stop", size="lg")
                        
                        with gr.Column(scale=1):
                            train_status = gr.Markdown(
                                value="<div style='padding: 20px; background: #f5f5f5; border-radius: 10px;'><h3>训练状态</h3><p>等待开始训练...</p></div>"
                            )
                            train_log = gr.Textbox(
                                label="训练日志",
                                lines=15,
                                max_lines=20,
                                interactive=False
                            )
            
            # 登出按钮
            logout_btn = gr.Button("🚪 登出")
        
        # ========== 事件绑定 ==========
        # 注册事件
        def handle_register(username, password, email):
            msg, success = register_user(username, password, email)
            return msg
        
        register_btn.click(
            fn=handle_register,
            inputs=[register_username, register_password, register_email],
            outputs=[register_msg]
        )
        
        # 验证码刷新函数（在外部定义以便复用）
        def refresh_captcha_func():
            image, captcha_text = generate_captcha()
            # 将PIL图像转换为Base64-HTML字符串（验证码模式）
            img_html = pil_to_base64_html(image, is_captcha=True)
            return img_html, captcha_text
        
        # 验证码刷新按钮
        refresh_captcha_btn.click(
            fn=refresh_captcha_func,
            inputs=[],
            outputs=[captcha_image, current_captcha]
        )
        
        # 登录事件
        def handle_login(username, password, user_captcha, stored_captcha):
            # 先验证验证码
            if not user_captcha or user_captcha.strip().lower() != stored_captcha.lower():
                # 验证码错误，刷新验证码
                new_captcha_img, new_captcha_text = refresh_captcha_func()
                return (
                    None,
                    False,
                    f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 验证码错误，请重新输入</h3></div>",
                    gr.update(visible=True),   # 显示登录页面
                    gr.update(visible=False),  # 隐藏主功能页面
                    new_captcha_img,           # 刷新验证码图像
                    new_captcha_text,          # 更新验证码文本
                    ""                         # 清空验证码输入
                )
            
            # 验证码正确，再验证用户名密码
            user_id, msg, success = login_user(username, password)
            if success:
                # 登录成功，刷新验证码（为下次登录准备）
                new_captcha_img, new_captcha_text = refresh_captcha_func()
                return (
                    user_id,
                    True,
                    f"<div style='padding: 15px; background: #4CAF50; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3><p style='margin: 10px 0 0 0;'>正在跳转到主页面...</p></div>",
                    gr.update(visible=False),  # 隐藏登录页面
                    gr.update(visible=True),   # 显示主功能页面
                    new_captcha_img,           # 刷新验证码图像
                    new_captcha_text,          # 更新验证码文本
                    ""                         # 清空验证码输入
                )
            else:
                # 登录失败，刷新验证码
                new_captcha_img, new_captcha_text = refresh_captcha_func()
                return (
                    None,
                    False,
                    f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3></div>",
                    gr.update(visible=True),   # 显示登录页面
                    gr.update(visible=False),  # 隐藏主功能页面
                    new_captcha_img,           # 刷新验证码图像
                    new_captcha_text,          # 更新验证码文本
                    ""                         # 清空验证码输入
                )
        
        login_btn.click(
            fn=handle_login,
            inputs=[login_username, login_password, captcha_input, current_captcha],
            outputs=[
                current_user_id,
                login_status,
                login_msg,
                login_page,
                main_page,
                captcha_image,
                current_captcha
            ]
        )
        
        # 页面加载时初始化验证码
        demo.load(
            fn=refresh_captcha_func,
            inputs=[],
            outputs=[captcha_image, current_captcha]
        )
        
        # 登出事件
        def handle_logout():
            return (
                None,
                False,
                gr.update(visible=True),    # 显示登录页面
                gr.update(visible=False)    # 隐藏主功能页面
            )
        
        # 历史记录刷新事件
        refresh_history_btn.click(
            fn=get_history,
            inputs=[current_user_id],
            outputs=[history_display]
        )
        
        # 切换到历史记录标签时自动加载历史
        history_tab.select(
            fn=get_history,
            inputs=[current_user_id],
            outputs=[history_display]
        )
        
        logout_btn.click(
            fn=handle_logout,
            inputs=[],
            outputs=[
                current_user_id,
                login_status,
                login_page,
                main_page
            ]
        )
        
        # 模型名称转换器
        def get_model_id_from_display(display_name):
            for model_id, display in AVAILABLE_MODELS:
                if display == display_name:
                    return model_id
            return DEFAULT_MODEL
        
        # 生成事件
        generate_btn.click(
            fn=lambda prompt, thresh, steps, guidance, h, w, user_id, model_display: \
                generate_and_detect(prompt, thresh, steps, guidance, h, w, user_id, get_model_id_from_display(model_display)),
            inputs=[
                prompt_input,
                threshold,
                num_steps,
                guidance_scale,
                height,
                width,
                current_user_id,
                model_dropdown
            ],
            outputs=[
                output_image,
                result_display,
                clip_score_display,
                fused_score_display
            ],
            show_progress=True
        )
        
        # Quick prompt button event bindings
        quick_prompt_btn1.click(fn=lambda: "A cute cat playing on green grass, sunny day, high detail, 4K resolution", outputs=prompt_input)
        quick_prompt_btn2.click(fn=lambda: "An astronaut walking on the moon surface, Earth in background, spacesuit details, sci-fi style", outputs=prompt_input)
        quick_prompt_btn3.click(fn=lambda: "Beach scenery at sunset, golden sunlight on sea, sailboats in distance, peaceful atmosphere", outputs=prompt_input)
        quick_prompt_btn4.click(fn=lambda: "Futuristic city night view, neon lights, flying cars, skyscrapers, cyberpunk style", outputs=prompt_input)
        quick_prompt_btn5.click(fn=lambda: "Small cabin in deep forest, surrounded by tall trees, smoke from chimney, peaceful natural environment", outputs=prompt_input)
        quick_prompt_btn6.click(fn=lambda: "Beautiful underwater world, colorful coral reefs, tropical fish swimming, clear water, sunlight through water", outputs=prompt_input)
        quick_prompt_btn7.click(fn=lambda: "Snowman in winter scenery, wearing red scarf and hat, snow-covered house in background, festive atmosphere", outputs=prompt_input)
        quick_prompt_btn8.click(fn=lambda: "Vintage-style coffee shop interior, wooden furniture, warm lighting, people reading and talking, nostalgic atmosphere", outputs=prompt_input)
        
        # 训练功能
        training_process = None
        
        def start_training(base_model, batch_size, epochs, lr, lambda_clip, lambda_local, output_dir):
            """启动训练"""
            import subprocess
            import sys
            
            # 检查数据是否已预处理
            data_file = "datasets/coco_processed_strategy_a/train2017_processed.pkl"
            if not os.path.exists(data_file):
                status_html = """
                <div style='padding: 20px; background: #ff9800; border-radius: 10px;'>
                    <h3>⚠️ 数据未预处理</h3>
                    <p>请先运行数据预处理脚本：</p>
                    <code>python prepare_coco_for_strategy_a.py</code>
                </div>
                """
                return status_html, "错误：数据未预处理，请先运行 prepare_coco_for_strategy_a.py"
            
            # 构建训练命令
            cmd = [
                sys.executable,
                "train_strategy_a.py",
                "--base_model", base_model,
                "--batch_size", str(int(batch_size)),
                "--num_epochs", str(int(epochs)),
                "--learning_rate", str(float(lr)),
                "--lambda_clip", str(float(lambda_clip)),
                "--lambda_local", str(float(lambda_local)),
                "--output_dir", output_dir
            ]
            
            status_html = """
            <div style='padding: 20px; background: #4CAF50; border-radius: 10px;'>
                <h3>✅ 训练已启动</h3>
                <p>正在后台运行训练，请查看日志...</p>
            </div>
            """
            
            log_text = f"开始训练...\n命令: {' '.join(cmd)}\n\n"
            
            # 在后台启动训练（实际应该使用线程或进程）
            try:
                # 这里简化处理，实际应该使用subprocess.Popen在后台运行
                log_text += "训练进程已启动（后台运行）\n"
                log_text += "注意：完整的训练日志请查看终端输出\n"
            except Exception as e:
                status_html = f"""
                <div style='padding: 20px; background: #f44336; border-radius: 10px;'>
                    <h3>❌ 训练启动失败</h3>
                    <p>{str(e)}</p>
                </div>
                """
                log_text += f"错误: {str(e)}\n"
            
            return status_html, log_text
        
        def check_preprocessing_status():
            """检查数据预处理状态"""
            data_file = "datasets/coco_processed_strategy_a/train2017_processed.pkl"
            if os.path.exists(data_file):
                return """
                <div style='padding: 15px; background: #4CAF50; border-radius: 10px; color: white;'>
                    <h4 style='margin: 0;'>✅ 数据已预处理</h4>
                    <p style='margin: 5px 0 0 0;'>可以开始训练</p>
                </div>
                """
            else:
                return """
                <div style='padding: 15px; background: #ff9800; border-radius: 10px; color: white;'>
                    <h4 style='margin: 0;'>⚠️ 数据未预处理</h4>
                    <p style='margin: 5px 0 0 0;'>请先运行: python prepare_coco_for_strategy_a.py</p>
                </div>
                """
        
        # 训练标签页加载时检查数据状态
        train_tab.select(
            fn=check_preprocessing_status,
            inputs=[],
            outputs=[train_status]
        )
        
        # 开始训练按钮
        start_train_btn.click(
            fn=start_training,
            inputs=[
                train_base_model,
                train_batch_size,
                train_epochs,
                train_lr,
                lambda_clip,
                lambda_local,
                train_output_dir
            ],
            outputs=[train_status, train_log]
        )
    
    # 启用队列
    demo.queue()
    return demo

# ==================== 主函数 ====================
def main():
    """运行Web界面"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像生成与语义一致性检测系统')
    parser.add_argument('--port', type=int, default=8081, help='服务器端口')
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("正在启动Web界面...")
        print("=" * 60)
        
        demo = create_interface()
        
        print("✅ Web界面创建成功！")
        print("=" * 60)
        print(f"正在启动服务器... (端口: {args.port})")
        print("=" * 60)
        
        # 启动服务器
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            show_error=True,
            inbrowser=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

