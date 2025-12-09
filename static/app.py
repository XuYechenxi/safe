#!/usr/bin/env python3
"""
Gradio界面 - 图像生成与语义一致性检测系统
保持HTML登录界面样式，使用Gradio实现所有功能
"""

import gradio as gr
import os
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw # 引入ImageDraw用于Mock生成
import threading
import random
import time # 引入time用于mock异步操作

# ==================== 模拟/存根依赖项 ====================
# 为了让程序能够独立运行，我们提供数据库和服务的模拟实现。

class Database:
    """模拟数据库操作"""
    def verify_user(self, username_or_email, password):
        # 简单Mock: 成功登录用户名为 'testuser'，密码为 'password'
        if username_or_email == "testuser" and password == "password":
            return 1  # Mock User ID 1
        return None

    def get_username_by_id(self, user_id):
        return f"用户_{user_id}"

    def register_user(self, username, password, email=None):
        if username == "exists":
            return False, "用户名已存在"
        return True, "注册成功"

    def save_generation(self, user_id, prompt, threshold, consistency_score, is_consistent, image_path, result_data):
        """Mock: 异步保存生成记录"""
        print(f"[DB MOCK] 异步保存记录: 用户{user_id}, 提示词: {prompt[:20]}...")
        # 实际应用中，这里会执行 Firestore 或其他数据库写入操作
        return True

    def get_user_history(self, user_id):
        """Mock: 获取用户历史记录"""
        if user_id == 1:
            return [
                {'created_at': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'), 'is_consistent': True, 'prompt': '一只在岩石岸边孤独灯塔的水彩画，意境深远，8k分辨率', 'consistency_score': 0.8972, 'threshold': 0.7, 'image_path': 'output/mock_image_path_1.png'},
                {'created_at': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), 'is_consistent': False, 'prompt': '一辆在未来赛博朋克城市中飞驰的汽车，细节丰富', 'consistency_score': 0.6120, 'threshold': 0.7, 'image_path': 'output/mock_image_path_2.png'}
            ]
        return []

    def get_dashboard_stats(self, user_id):
        """Mock: 获取仪表盘统计数据"""
        stats = {'total_generations': 150, 'today_generations': 25, 'consistent_count': 120, 'inconsistent_count': 30}
        
        # 模拟12小时数据
        hourly_stats = [{'hour': (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:00'), 'count': random.randint(5, 15)} for i in range(12)]
        hourly_stats.reverse()
        
        # 模拟模型统计数据
        model_stats = {'SD基础模型': 80, 'OpenAI CLIP融合模型': 40, 'ITSC-GAN融合模型': 30}
        
        # 模拟热门关键词
        keywords = ['灯塔', '猫咪', '城市景观', '山脉', '赛博朋克', '水彩', '摄影', '数字艺术', '油画', '抽象', '风景', '人像', '动物', '汽车'] * 5
        random.shuffle(keywords)
        
        return {'stats': stats, 'hourly_stats': hourly_stats, 'model_stats': model_stats, 'keywords': keywords}

# 初始化模拟数据库
db = Database()

# 模拟服务模块
class ImageGenerationService:
    def generate(self, prompt, model_name, num_inference_steps, guidance_scale, height, width):
        """Mock: 模拟图像生成"""
        print(f"[GEN MOCK] 正在生成: {prompt} ({num_inference_steps}步)")
        time.sleep(1 + (num_inference_steps / 50)) # 模拟耗时
        
        # 返回一个带有提示词信息的模拟图片
        img = Image.new('RGB', (width, height), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        draw.text((10,10), f"Mock Image by {model_name}\nSteps: {num_inference_steps}", fill='white')
        return img
    
    def save_image(self, image, prompt, model_id):
        """Mock: 模拟图像保存"""
        os.makedirs("output", exist_ok=True)
        filename = f"output/mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}.png"
        image.save(filename, "PNG") 
        return filename

class ConsistencyDetectionService:
    def detect(self, image, prompt, threshold):
        """Mock: 模拟一致性检测"""
        score = random.uniform(0.3, 0.95)
        is_consistent = score >= threshold
        detail = {
            'clip_score': score * random.uniform(0.8, 1.1), 
            'fused_score': score * random.uniform(0.9, 1.2),
            'model_info': 'Mock-CLIP-GAN-Fusion'
        }
        return is_consistent, score, detail

class ModelComparisonService:
    def compare_models(self, prompt, model_names, threshold, num_inference_steps, guidance_scale, height, width):
        """Mock: 模拟多模型对比"""
        results = []
        for model_id in model_names:
            model_name = dict(AVAILABLE_MODELS).get(model_id, model_id)
            score = random.uniform(0.4, 0.9)
            is_consistent = score >= threshold
            
            # 模拟生成图片
            img = Image.new('RGB', (400, 300), color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
            draw = ImageDraw.Draw(img)
            draw.text((10,10), f"Mock {model_name}", fill='white')
            
            results.append({
                'model_name': model_name,
                'overall_score': score,
                'clip_score': score * random.uniform(0.8, 1.1),
                'fused_score': score * random.uniform(0.9, 1.2),
                'is_consistent': is_consistent,
                'image': img
            })
        print(f"[COMPARE MOCK] 完成 {len(model_names)} 个模型的对比。")
        return {'prompt': prompt, 'results': results}

class SummaryAnalysisService:
    pass # 暂未使用

class DashboardService:
    def generate_dashboard_html(self, user_id):
        """调用顶级函数生成仪表盘HTML"""
        stats_data = db.get_dashboard_stats(user_id)
        return generate_dashboard_html(
            stats=stats_data['stats'],
            hourly_stats=stats_data['hourly_stats'],
            model_stats=stats_data['model_stats'],
            keywords=stats_data['keywords']
        )

# Mock implementation for the missing get_system function
def get_system():
    class MockSystem:
        def __init__(self):
            # 在一个函数调用中获取新的服务实例，保持结构一致性
            self.generator = ImageGenerationService() 
            self.detector = ConsistencyDetectionService()
    return MockSystem()

# 初始化服务 (仅用于不需频繁实例化的服务，生成和检测将使用 get_system 获取)
# 这一部分保持不变，以便其他不使用 get_system() 的函数可以继续使用这些全局实例。
comparison_service = ModelComparisonService()
summary_service = SummaryAnalysisService()
dashboard_service = DashboardService()


# ==================== 全局变量 (来自用户代码) ====================
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
AVAILABLE_MODELS = [
    ("runwayml/stable-diffusion-v1-5", "SD基础模型 (Stable Diffusion v1.5)"),
    ("openai-clip-fusion", "OpenAI CLIP融合模型"),
    ("itsc-gan-fusion", "ITSC-GAN融合模型")
]
DEFAULT_NUM_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_THRESHOLD = 0.7 # 将阈值改为0.7，更符合一致性检测的实际应用

# 反向映射：显示名称 -> 模型ID (用于修复模型选择bug)
MODEL_ID_MAP = {display_name: model_id for model_id, display_name in AVAILABLE_MODELS}

# ==================== 辅助函数 (来自用户代码) ====================
def pil_to_base64_html(pil_image, max_width=None, max_height=None):
    """将PIL图像转换为Base64编码的HTML img标签"""
    import io
    import base64
    
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
    
    buffer = io.BytesIO()
    display_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    style = f"display:block;max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;margin: 0 auto;"
    img_html = f"<img src='data:image/png;base64,{img_str}' style='{style}' />"
    return img_html

# ==================== 用户认证 (来自用户代码) ====================
def login_user(username_or_email, password):
    """用户登录"""
    try:
        user_id = db.verify_user(username_or_email, password)
        if user_id:
            username = db.get_username_by_id(user_id)
            # 注意：此处更新 iface 的 selected 索引需要额外的 click 绑定，见文件底部
            return user_id, f"✅ 登录成功！欢迎，{username}！", gr.update(visible=False), gr.update(visible=True)
        else:
            return None, "❌ 用户名或密码错误", gr.update(visible=True), gr.update(visible=False)
    except Exception as e:
        return None, f"❌ 登录失败: {str(e)}", gr.update(visible=True), gr.update(visible=False)

def register_user(username, email, password, confirm_password):
    """用户注册"""
    if not username or not password:
        return "❌ 用户名和密码不能为空", gr.update(visible=False)
    
    if len(password) < 6:
        return "❌ 密码长度至少6个字符", gr.update(visible=False)
    
    if password != confirm_password:
        return "❌ 两次输入的密码不一致", gr.update(visible=False)
    
    try:
        success, message = db.register_user(username, password, email=email)
        if success:
            return "✅ 注册成功！请返回登录页面登录。", gr.update(visible=false)
        else:
            return f"❌ {message}", gr.update(visible=false)
    except Exception as e:
        return f"❌ 注册失败: {str(e)}", gr.update(visible=false)

# ==================== 图像生成与检测 (来自用户代码) ====================
def generate_image(
    prompt: str,
    model_name: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int
):
    """生成图像"""
    if not user_id:
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
        return "", "", error_html, None, None, None
    
    try:
        # 1. 统一获取系统实例
        current_system = get_system()
        
        # 2. 修复模型显示名称到模型ID的转换
        model_id = MODEL_ID_MAP.get(model_name, DEFAULT_MODEL) 
        
        # 3. 生成图像 (使用实例)
        image = current_system.generator.generate(
            prompt=prompt,
            model_name=model_id,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        )
        
        # 4. 保存图像 (使用实例)
        image_path = current_system.generator.save_image(image, prompt, model_id)
        
        # 转换为HTML
        img_html = pil_to_base64_html(image, max_width=600, max_height=400)
        
        # 返回图片文件路径、HTML、以及用于下一步检测的输入
        return image_path, img_html, "<div style='padding: 10px; background: #2196F3; border-radius: 6px; color: white; text-align: center;'>✅ 图像生成成功，请点击检测按钮进行一致性分析。</div>", prompt, model_name, threshold
        
    except Exception as e:
        error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 生成失败: {str(e)}</h3></div>"
        return "", "", error_html, None, None, None


def detect_consistency(image_path: str, prompt: str, threshold: float, model_name: str, user_id: int):
    """检测一致性"""
    if not user_id:
        return gr.update(value="<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>")
    
    if not image_path or not os.path.exists(image_path):
        return gr.update(value="<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先成功生成图像</h3></div>")
    
    try:
        image = Image.open(image_path)
        
        # 1. 统一获取系统实例
        current_system = get_system()
        
        # 2. 进行一致性检测 (使用实例)
        is_consistent, score, detail = current_system.detector.detect(image, prompt, threshold)
        
        # 保存到数据库
        def save_async():
            try:
                # 3. 修复模型显示名称到模型ID的转换，用于数据库保存
                model_id = MODEL_ID_MAP.get(model_name, DEFAULT_MODEL)
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
        
        # 生成结果HTML
        clip_score = detail.get('clip_score', score)
        fused_score = detail.get('fused_score', score)
        status_icon = "✅" if is_consistent else "❌"
        status_color = "#4CAF50" if is_consistent else "#F44336"
        
        result_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2 style="margin-top: 0;">{status_icon} 一致性检测结果</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                <p><strong>📝 提示词:</strong> {prompt[:100]}...</p>
                <p><strong>🤖 生成模型:</strong> {model_name}</p>
                <p><strong>🎯 一致性状态:</strong> <span style="color: {status_color}; font-weight: bold;">{'通过' if is_consistent else '未通过'}</span></p>
                <p><strong>📊 整体分数:</strong> <span style="font-size: 18px; font-weight: bold;">{score:.4f}</span> (阈值: {threshold:.2f})</p>
                <p><strong>🔗 CLIP分数:</strong> {clip_score:.4f}</p>
                <p><strong>🔥 融合分数:</strong> {fused_score:.4f}</p>
            </div>
            <p style="text-align: right; margin: 10px 0 0 0;">结果已记录到历史记录。</p>
        </div>
        """
        
        return gr.update(value=result_html)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(value=f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 检测失败: {str(e)}</h3></div>")

# ==================== 多模型对比 (来自用户代码) ====================
def generate_comparison_html(comparison_results: dict) -> str:
    """生成对比结果HTML (来自用户代码)"""
    results = comparison_results.get('results', [])
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return "<div style='padding: 20px; background: #ff9800; border-radius: 10px;'>没有有效的对比结果</div>"
    
    # 生成对比卡片
    cards_html = ""
    for result in valid_results:
        model_name = result['model_name']
        overall_score = result.get('overall_score', 0)
        is_consistent = result.get('is_consistent', False)
        image = result.get('image')
        
        status_icon = "✅" if is_consistent else "❌"
        status_color = "#4CAF50" if is_consistent else "#F44336"
        
        img_html = ""
        if image:
            img_html = pil_to_base64_html(image, max_width=350, max_height=250)
        
        cards_html += f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">🤖 {model_name}</h3>
            <div style="margin: 15px 0; text-align: center;">
                {img_html}
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                <p style="margin: 5px 0;"><strong>🎯 一致性状态:</strong> <span style="color: {status_color}; font-weight: bold;">{status_icon} {'通过' if is_consistent else '未通过'}</span></p>
                <p style="margin: 5px 0;"><strong>📊 整体分数:</strong> <span style="font-size: 18px; font-weight: bold;">{overall_score:.4f}</span></p>
            </div>
        </div>
        """
    
    return f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="margin-top: 0;">🔍 多模型对比结果</h2>
        <p><strong>提示词:</strong> {comparison_results.get('prompt', '')[:100]}...</p>
        <p><strong>一致性阈值:</strong> {comparison_results.get('threshold', DEFAULT_THRESHOLD):.2f}</p>
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px;">
        {cards_html}
    </div>
    """

def compare_models_sync(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int
):
    """同步对比多个模型"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    if not prompt or not prompt.strip():
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请输入提示词</h3></div>"

    try:
        # 获取所有模型ID
        model_names = [m[0] for m in AVAILABLE_MODELS]
        
        # 对比模型 (使用全局 comparison_service，因为它不需要频繁实例化)
        comparison_results = comparison_service.compare_models(
            prompt=prompt,
            model_names=model_names,
            threshold=threshold,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        )
        comparison_results['threshold'] = threshold
        
        # 生成对比HTML
        comparison_html = generate_comparison_html(comparison_results)
        
        return comparison_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {str(e)}</h3></div>"

# ==================== 历史记录 (来自用户代码) ====================
def get_history(user_id):
    """获取用户历史记录"""
    if not user_id:
        return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    try:
        history = db.get_user_history(user_id)
        
        if not history:
            return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>📝 暂无历史记录</h3><p>生成并检测图像后，记录将显示在这里。</p></div>"
        
        history_html = """
        <div style='padding: 20px; background: #f5f5f5; border-radius: 10px;'>
            <h2 style='margin-top: 0; color: #333;'>📊 历史记录</h2>
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; margin-top: 20px;'>
        """
        
        for record in history:
            status_color = "#4CAF50" if record['is_consistent'] else "#F44336"
            status_text = "通过" if record['is_consistent'] else "未通过"
            status_icon = "✅" if record['is_consistent'] else "❌"
            
            image_preview = "<div style='text-align: center; margin: 10px 0;'><p style='color: #777; font-style: italic;'>图像预览（Mock）</p></div>"
            
            # 使用 mock 图片路径生成一个占位图片（因为实际文件可能不存在）
            mock_image = Image.new('RGB', (300, 200), color = 'gray')
            ImageDraw.Draw(mock_image).text((10, 10), "Mock Preview", fill='white')
            image_preview = pil_to_base64_html(mock_image, max_width=300, max_height=200)

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

# ==================== 多步数对比 (来自用户代码) ====================
def compare_steps_sync(
    prompt: str,
    threshold: float,
    num_steps: int, # 这个参数在这个函数中不起作用，因为它生成的是固定步数列表
    guidance_scale: float,
    user_id: int
):
    """多步数对比（对比不同推理步数的效果）"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    if not prompt or not prompt.strip():
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请输入提示词</h3></div>"
    
    # 对比不同推理步数（1, 2, 4, 8, 16步，避免过多生成）
    step_list = [1, 2, 4, 8, 16]
    results = []
    
    try:
        current_system = get_system()
        
        for steps in step_list:
            print(f"[INFO] 使用推理步数 {steps} 生成图像...")
            
            # 简化调用，使用默认宽高
            image = current_system.generator.generate(
                prompt=prompt,
                model_name=DEFAULT_MODEL,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=DEFAULT_HEIGHT,
                width=DEFAULT_WIDTH
            )
            
            if image is None:
                continue
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            is_consistent, score, detail = current_system.detector.detect(
                image, prompt, threshold
            )
            
            clip_score = detail.get('clip_score', score)
            fused_score = detail.get('fused_score', score)
            
            # Mock save
            os.makedirs("output", exist_ok=True)
            image_path = f"output/mock_steps_comp_{steps}_{datetime.now().timestamp()}.png"
            image.save(image_path, "PNG")
            
            img_html = pil_to_base64_html(image, max_width=400, max_height=300)
            
            results.append({
                'steps': steps,
                'image_html': img_html,
                'clip_score': clip_score,
                'fused_score': fused_score,
                'overall_score': score,
                'is_consistent': is_consistent
            })
            
            # 异步保存到数据库
            def save_async():
                try:
                    db.save_generation(
                        user_id=user_id,
                        prompt=f"[步数对比-{steps}步] {prompt}",
                        threshold=threshold,
                        consistency_score=score,
                        is_consistent=is_consistent,
                        image_path=image_path,
                        result_data={**detail, 'steps': steps}
                    )
                except Exception as e:
                    print(f"[WARNING] 数据库保存失败: {e}")
            
            threading.Thread(target=save_async, daemon=True).start()
        
        if not results:
            return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 所有步数生成失败</h3></div>"
        
        # 生成对比HTML
        best_result = max(results, key=lambda x: x['overall_score'])
        worst_result = min(results, key=lambda x: x['overall_score'])
        avg_clip = sum(r['clip_score'] for r in results) / len(results)
        avg_fused = sum(r['fused_score'] for r in results) / len(results)
        avg_overall = sum(r['overall_score'] for r in results) / len(results)
        
        comparison_html = f"""
<div style="padding: 20px; background: linear-gradient(135deg, #FF9800 0%, #F44336 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
    <h2 style="margin-top: 0; display: flex; align-items: center; gap: 10px;">
        🔍 推理步数效果对比 (1, 2, 4, 8, 16步)
    </h2>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
        <p style="margin: 8px 0;"><strong>📝 提示词:</strong> {prompt[:100]}...</p>
        <p style="margin: 8px 0;"><strong>🎯 一致性阈值:</strong> {threshold}</p>
    </div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px;">
"""
        
        for result in results:
            status_color = "#4CAF50" if result['is_consistent'] else "#F44336"
            status_text = "通过" if result['is_consistent'] else "未通过"
            status_icon = "✅" if result['is_consistent'] else "❌"
            
            comparison_html += f"""
    <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0; color: #333; text-align: center;">推理步数: {result['steps']} 步</h3>
        <div style="margin: 15px 0; text-align: center;">
            {result['image_html']}
        </div>
        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <p style="margin: 5px 0;"><strong>🔗 CLIP分数:</strong> <span style="color: #2196F3; font-weight: bold;">{result['clip_score']:.4f}</span></p>
            <p style="margin: 5px 0;"><strong>📊 整体分数:</strong> <span style="color: {status_color}; font-weight: bold; font-size: 18px;">{result['overall_score']:.4f}</span></p>
            <p style="margin: 5px 0;"><strong>🎯 状态:</strong> <span style="background: {status_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">{status_icon} {status_text}</span></p>
        </div>
    </div>
"""
        
        comparison_html += """
</div>

<div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px;">
    <h3 style="margin-top: 0; color: #333;">📊 统计摘要</h3>
"""
        
        comparison_html += f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
            <p style="margin: 5px 0; color: #666; font-size: 14px;">平均CLIP分数</p>
            <p style="margin: 5px 0; color: #2196F3; font-size: 24px; font-weight: bold;">{avg_clip:.4f}</p>
        </div>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
            <p style="margin: 5px 0; color: #666; font-size: 14px;">平均整体分数</p>
            <p style="margin: 5px 0; color: #2196F3; font-size: 24px; font-weight: bold;">{avg_overall:.4f}</p>
        </div>
        <div style="background: #e8f5e9; padding: 15px; border-radius: 8px;">
            <p style="margin: 5px 0; color: #666; font-size: 14px;">融合/CLIP提升率 (Mock)</p>
            <p style="margin: 5px 0; color: #4CAF50; font-size: 24px; font-weight: bold;">{((avg_fused - avg_clip) / avg_clip * 100):+.2f}%</p>
        </div>
    </div>
    <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
        <p style="margin: 5px 0; color: #856404;"><strong>🏆 最佳步数:</strong> {best_result['steps']} 步 (分数: {best_result['overall_score']:.4f})</p>
        <p style="margin: 5px 0; color: #856404;"><strong>📉 最差步数:</strong> {worst_result['steps']} 步 (分数: {worst_result['overall_score']:.4f})</p>
    </div>
</div>
"""
        
        return comparison_html
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[ERROR] 步数对比失败: {str(e)}")
        print(error_traceback)
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 步数对比失败: {str(e)}</h3></div>"

# ==================== 仪表盘功能 (来自用户代码) ====================
def get_dashboard_data(user_id):
    """获取仪表盘数据"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    try:
        # 调用 DashboardService 的 mock 实现
        dashboard_html = dashboard_service.generate_dashboard_html(user_id)
        return dashboard_html
    except Exception as e:
        error_msg = f"❌ 获取仪表盘数据失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"

def generate_dashboard_html(stats, hourly_stats, model_stats, keywords):
    """
    生成仪表盘HTML (补全用户代码中被截断的部分)
    注: 由于 Gradio 的 HTML 组件限制，这里的导航栏点击事件需要通过 Gradio TabbedInterface 间接实现，
    这里仅用于展示视觉效果。
    """
    
    # 准备12小时数据（填充缺失的小时）
    hours_data = {}
    now = datetime.now()
    for i in range(12):
        # 使用小时作为标签，去除日期
        hour_time = now - timedelta(hours=11-i)
        hour_label = hour_time.strftime('%H:00')
        hour_key = hour_time.strftime('%Y-%m-%d %H:00')
        hours_data[hour_key] = {'label': hour_label, 'count': 0}
    
    for item in hourly_stats:
        hour_key = item['hour']
        if hour_key in hours_data:
            hours_data[hour_key]['count'] = item['count']
    
    # 准备图表数据
    chart_labels = [h['label'] for h in hours_data.values()]
    chart_data = [h['count'] for h in hours_data.values()]
    
    # 模型统计数据
    model_labels = list(model_stats.keys())
    model_values = list(model_stats.values())
    
    # 关键词统计（简单统计）
    keyword_counts = {}
    for keyword in keywords:
        if len(keyword) >= 2:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # 取前20个关键词
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    dashboard_html = f"""
    <style>
        .dashboard-container {{
            min-height: 100vh;
            background: #f5f5f5;
            padding: 30px;
            font-family: 'Inter', sans-serif;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        @media (max-width: 768px) {{
            .metric-grid {{
                grid-template-columns: repeat(2, 1fr) !important;
            }}
            .chart-grid {{
                grid-template-columns: 1fr !important;
            }}
        }}
    </style>
    
    <div class="dashboard-container">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
            <h1 style="margin: 0; color: #333; font-size: 28px;">📊 仪表盘总览</h1>
        </div>
        
        <!-- 四个指标卡片 -->
        <div class="metric-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div class="metric-card" style="border-left: 4px solid #fbbf24;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 24px;">⭐</span>
                    <span style="color: #666; font-size: 14px;">总生成次数</span>
                </div>
                <div class="value">{stats['total_generations']}条</div>
            </div>
            
            <div class="metric-card" style="border-left: 4px solid #ef4444;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 24px;">🕐</span>
                    <span style="color: #666; font-size: 14px;">今日生成</span>
                </div>
                <div class="value">{stats['today_generations']}条</div>
            </div>
            
            <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 24px;">✅</span>
                    <span style="color: #666; font-size: 14px;">一致性通过</span>
                </div>
                <div class="value">{stats['consistent_count']}条</div>
            </div>
            
            <div class="metric-card" style="border-left: 4px solid #6b7280;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-size: 24px;">❌</span>
                    <span style="color: #666; font-size: 14px;">一致性未通过</span>
                </div>
                <div class="value">{stats['inconsistent_count']}条</div>
            </div>
        </div>
        
        <!-- 图表区域 -->
        <div class="chart-grid" style="display: grid; grid-template-columns: 2fr 1fr; gap: 30px;">
            <!-- 左侧 - 生成趋势图 -->
            <div class="chart-container">
                <h3 style="margin-top: 0; color: #333;">近12小时生成趋势</h3>
                <canvas id="hourlyTrendChart" width="400" height="150"></canvas>
            </div>
            
            <!-- 右侧 - 模型分布图 -->
            <div class="chart-container">
                <h3 style="margin-top: 0; color: #333;">模型使用分布</h3>
                <canvas id="modelDistributionChart" width="200" height="200"></canvas>
            </div>
        </div>
        
        <!-- 关键词云 (模拟) -->
        <div style="margin-top: 30px;" class="chart-container">
            <h3 style="margin-top: 0; color: #333;">热门关键词 (Top {len(top_keywords)})</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
"""
    
    # 关键词卡片
    for keyword, count in top_keywords:
        # 简单模拟字体大小变化
        font_size = max(12, min(30, int(count * 0.8))) 
        dashboard_html += f"""
                <span style="padding: 5px 10px; border-radius: 5px; background: #e0f7fa; color: #00796b; font-size: {font_size}px; font-weight: 500; cursor: default;">
                    {keyword}
                </span>
"""
    
    dashboard_html += f"""
            </div>
        </div>
        
        <!-- Chart.js 脚本 -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
        <script>
            // Gradio 在加载外部脚本时可能会有延迟，为了确保 Chart.js 正常工作，使用 setTimeout
            setTimeout(() => {{
                const chartLabels = {json.dumps(chart_labels)};
                const chartData = {json.dumps(chart_data)};
                const modelLabels = {json.dumps(model_labels)};
                const modelValues = {json.dumps(model_values)};
                
                // 1. 小时趋势图
                const ctxTrend = document.getElementById('hourlyTrendChart');
                if (ctxTrend) {{
                    new Chart(ctxTrend, {{
                        type: 'line',
                        data: {{
                            labels: chartLabels,
                            datasets: [{{
                                label: '生成次数',
                                data: chartData,
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fill: true,
                                tension: 0.3
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{ display: false }},
                                title: {{ display: true, text: '近12小时生成次数' }}
                            }},
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: '次数' }} }},
                                x: {{ title: {{ display: true, text: '小时' }} }}
                            }}
                        }}
                    }});
                }}

                // 2. 模型分布图
                const ctxDist = document.getElementById('modelDistributionChart');
                if (ctxDist) {{
                    const backgroundColors = ['#10b981', '#f97316', '#3b82f6', '#ef4444'];
                    new Chart(ctxDist, {{
                        type: 'doughnut',
                        data: {{
                            labels: modelLabels,
                            datasets: [{{
                                data: modelValues,
                                backgroundColor: backgroundColors.slice(0, modelLabels.length),
                                hoverOffset: 4
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: true, 
                            plugins: {{
                                legend: {{ position: 'bottom' }},
                                title: {{ display: true, text: '模型使用分布' }}
                            }}
                        }}
                    }});
                }}
            }}, 100); // 延迟100ms加载图表
        </script>
    </div>
    """
    
    return dashboard_html


# ==================== Gradio 界面定义 ====================

# 使用 gr.State 来跨组件和页面存储用户状态
user_id_state = gr.State(None)
username_state = gr.State("访客")

# 登录/注册页面
with gr.Blocks(title="登录/注册") as login_block:
    gr.Markdown("# 🎨 图像语义一致性检测系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔒 用户登录")
            login_username = gr.Textbox(label="用户名/邮箱 (Mock: testuser)", placeholder="testuser")
            login_password = gr.Textbox(label="密码 (Mock: password)", placeholder="password", type="password")
            login_button = gr.Button("登录", variant="primary")
            login_output = gr.Markdown("")
            
        with gr.Column(scale=1):
            gr.Markdown("## ✏️ 新用户注册 (Mock: 任何新用户)")
            reg_username = gr.Textbox(label="用户名")
            reg_email = gr.Textbox(label="邮箱 (可选)")
            reg_password = gr.Textbox(label="密码 (至少6位)", type="password")
            reg_confirm = gr.Textbox(label="确认密码", type="password")
            register_button = gr.Button("注册", variant="secondary")
            register_output = gr.Markdown("")
            
    # 登录逻辑
    login_button.click(
        fn=login_user,
        inputs=[login_username, login_password],
        outputs=[user_id_state, login_output, login_block]
        # 实际 Gradio 中，Blocks 之间的切换需要通过 TabbedInterface 管理，
        # 这里用 update(visible=False/True) 模拟切换，但 Gradio 不支持直接隐藏当前 Block。
        # 在 TabbedInterface 中，我们将在外部控制标签页的显示。
    )

    # 注册逻辑
    register_button.click(
        fn=register_user,
        inputs=[reg_username, reg_email, reg_password, reg_confirm],
        outputs=[register_output]
    )

# 主应用内容，仅在用户登录后可见
with gr.Blocks() as main_app_block:
    
    # 顶部状态栏
    gr.Markdown(
        f"""
        <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 5px solid #2196F3;">
            <p style="margin: 0; font-size: 16px;">
                当前用户状态: <span id="current-user-status" style="font-weight: bold;"></span> (ID: <span id="current-user-id" style="font-weight: bold;"></span>)
            </p>
        </div>
        <script>
            // 实时更新用户状态
            function updateUserInfo(userId) {{
                const statusElement = document.getElementById('current-user-status');
                const idElement = document.getElementById('current-user-id');
                if (userId && userId !== 'null') {{
                    statusElement.innerText = '已登录';
                    statusElement.style.color = '#4CAF50';
                    idElement.innerText = userId;
                }} else {{
                    statusElement.innerText = '访客 (未登录)';
                    statusElement.style.color = '#F44336';
                    idElement.innerText = 'N/A';
                }}
            }}
            // 初始加载时调用 (依赖于 Gradio 渲染 State 的值)
            setTimeout(() => updateUserInfo(null), 500); 
        </script>
        """
    )
    
    # 存储临时生成结果的 State，用于传递给检测步骤
    image_path_temp = gr.State(None)
    prompt_temp = gr.State(None)
    model_name_temp = gr.State(None)
    threshold_temp = gr.State(DEFAULT_THRESHOLD)

    with gr.Tab("✨ 语义生成图像") as tab_generation:
        gr.Markdown("## 图像生成与语义一致性检测")
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="提示词 (Prompt)", lines=3, placeholder="输入您想生成的图像描述，例如：一只太空猫在月球上打篮球")
                model_select = gr.Dropdown(label="生成模型", choices=[m[1] for m in AVAILABLE_MODELS], value=AVAILABLE_MODELS[0][1])
                threshold_slider = gr.Slider(label="一致性检测阈值", minimum=0.1, maximum=1.0, step=0.05, value=DEFAULT_THRESHOLD)
                
                with gr.Accordion("高级参数", open=False):
                    steps_slider = gr.Slider(label="推理步数 (Num Steps)", minimum=10, maximum=100, step=5, value=DEFAULT_NUM_STEPS)
                    guidance_slider = gr.Slider(label="指导尺度 (Guidance Scale)", minimum=1.0, maximum=15.0, step=0.5, value=DEFAULT_GUIDANCE_SCALE)
                    with gr.Row():
                        height_input = gr.Slider(label="图像高度 (Height)", minimum=256, maximum=1024, step=64, value=DEFAULT_HEIGHT)
                        width_input = gr.Slider(label="图像宽度 (Width)", minimum=256, maximum=1024, step=64, value=DEFAULT_WIDTH)
                        
                generate_button = gr.Button("🚀 生成图像", variant="primary")

            with gr.Column(scale=1):
                image_output_html = gr.HTML(label="生成图像", value="<div style='text-align: center; color: #666; padding: 50px; border: 1px dashed #ccc; border-radius: 8px;'>图像将显示在这里</div>")
                status_html = gr.HTML(label="状态/信息", value="等待输入...")
                detect_button = gr.Button("🔍 进行一致性检测", variant="secondary")
                consistency_output_html = gr.HTML(label="一致性检测结果", value="")

        # 生成逻辑
        generate_button.click(
            fn=generate_image,
            inputs=[prompt_input, model_select, threshold_slider, steps_slider, guidance_slider, height_input, width_input, user_id_state],
            outputs=[image_path_temp, image_output_html, status_html, prompt_temp, model_name_temp, threshold_temp]
        )
        
        # 检测逻辑
        detect_button.click(
            fn=detect_consistency,
            inputs=[image_path_temp, prompt_temp, threshold_temp, model_name_temp, user_id_state],
            outputs=[consistency_output_html]
        )

    with gr.Tab("🔍 多模型对比") as tab_comparison:
        gr.Markdown("## 多模型/多步数效果对比分析")
        with gr.Row():
            with gr.Column(scale=1):
                comp_prompt = gr.Textbox(label="对比提示词", lines=3, placeholder="输入用于多模型对比的提示词")
                comp_threshold = gr.Slider(label="一致性检测阈值", minimum=0.1, maximum=1.0, step=0.05, value=DEFAULT_THRESHOLD)
                
                with gr.Accordion("生成参数 (统一使用)", open=False):
                    comp_steps = gr.Slider(label="推理步数", minimum=10, maximum=100, step=5, value=DEFAULT_NUM_STEPS)
                    comp_guidance = gr.Slider(label="指导尺度", minimum=1.0, maximum=15.0, step=0.5, value=DEFAULT_GUIDANCE_SCALE)
                    comp_height = gr.Slider(label="高度", minimum=256, maximum=1024, step=64, value=DEFAULT_HEIGHT)
                    comp_width = gr.Slider(label="宽度", minimum=256, maximum=1024, step=64, value=DEFAULT_WIDTH)

                comp_model_button = gr.Button("🚀 模型对比 (所有模型)", variant="primary")
                comp_steps_button = gr.Button("🚀 步数对比 (1, 2, 4, 8, 16 步)", variant="secondary")

            with gr.Column(scale=2):
                comparison_output_html = gr.HTML(label="对比结果", value="<div style='text-align: center; color: #666; padding: 50px; border: 1px dashed #ccc; border-radius: 8px;'>对比结果将以卡片形式展示</div>")
        
        # 模型对比逻辑
        comp_model_button.click(
            fn=compare_models_sync,
            inputs=[comp_prompt, comp_threshold, comp_steps, comp_guidance, comp_height, comp_width, user_id_state],
            outputs=[comparison_output_html]
        )
        
        # 步数对比逻辑
        comp_steps_button.click(
            fn=compare_steps_sync,
            inputs=[comp_prompt, comp_threshold, comp_steps, comp_guidance, user_id_state],
            outputs=[comparison_output_html]
        )
        
    with gr.Tab("📊 历史记录") as tab_history:
        gr.Markdown("## 个人生成历史记录")
        history_button = gr.Button("刷新历史记录", variant="primary")
        history_output = gr.HTML(label="历史记录列表", value="")
        
        history_button.click(
            fn=get_history,
            inputs=[user_id_state],
            outputs=[history_output]
        )
        
    with gr.Tab("📈 仪表盘") as tab_dashboard:
        gr.Markdown("## 系统与个人统计仪表盘")
        dashboard_button = gr.Button("刷新仪表盘", variant="primary")
        dashboard_output = gr.HTML(label="仪表盘数据", value="")
        
        dashboard_button.click(
            fn=get_dashboard_data,
            inputs=[user_id_state],
            outputs=[dashboard_output]
        )


# 主应用入口，使用 gr.TabbedInterface 封装，并在登录成功后更新状态
app_blocks = [login_block, main_app_block]
titles = ["登录/注册", "主应用"]

# 最终 Gradio 接口
# 在Gradio 3.50.0版本中，TabbedInterface不支持selected参数
iface = gr.TabbedInterface(app_blocks, titles)

# 在 Gradio 启动时，我们不能直接控制用户是否登录，所以需要在主应用块中添加逻辑进行判断
# 此外，为了模拟登录后的跳转，我们需要在 tab_generation, tab_comparison 等被选中时，
# 检查 user_id_state 的值，如果为 None，则强制跳转回登录页。
# 由于 Gradio Blocks 的限制，这个跳转逻辑最好在前端通过JS实现，但在纯Python中我们仅能依赖 State 传递。

# 假设用户成功登录后，我们可以在 login_user 中触发一个 EventData 来模拟状态更新
# 由于 Gradio 的复杂性，最简单的方法是让用户手动切换到下一页

if __name__ == "__main__":
    # 创建 output 文件夹用于 mock 保存图片
    os.makedirs("output", exist_ok=True)
    print("Gradio App 正在启动。请使用 'testuser' / 'password' 登录以解锁完整功能。")
    # 添加share=False参数来禁用外部分享和相关分析功能，避免Google Analytics连接错误
    iface.launch(share=False)