#!/usr/bin/env python3
"""
Gradio界面 - 图像生成与语义一致性检测系统
保持HTML登录界面样式，使用Gradio实现所有功能
"""

import gradio as gr
import os
import time
import uuid
from datetime import datetime
from database import Database
from typing import Callable, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import threading
import random

# 导入服务模块
from image_generation_service import ImageGenerationService
from consistency_detection_service import ConsistencyDetectionService
from model_comparison_service import ModelComparisonService
from summary_analysis_service import SummaryAnalysisService
from dashboard_service import DashboardService

# ==================== 全局变量 ====================
db = Database()

# 服务初始化改为后台异步加载，不阻塞主程序启动
# 初始化为None，在后台线程中加载
generation_service = None
detection_service = None
comparison_service = None
summary_service = None
dashboard_service = None

def initialize_services_async():
    """在后台线程中初始化服务，不阻塞主程序"""
    global generation_service, detection_service, comparison_service, summary_service, dashboard_service
    
    print("[INFO] 正在后台初始化服务...")
    print("[INFO] 注意: CLIP模型将延迟加载，首次使用时可能需要几分钟")
    
    try:
        generation_service = ImageGenerationService()
        print("[INFO] ✅ 图像生成服务初始化成功")
    except Exception as e:
        print(f"[WARNING] 图像生成服务初始化失败: {e}")
        print("[WARNING] 登录功能仍可使用，但图像生成功能可能不可用")
        generation_service = None
    
    try:
        detection_service = ConsistencyDetectionService()
        print("[INFO] ✅ 一致性检测服务初始化成功")
    except Exception as e:
        print(f"[WARNING] 一致性检测服务初始化失败: {e}")
        print("[WARNING] 登录功能仍可使用，但一致性检测功能可能不可用")
        detection_service = None
    
    try:
        comparison_service = ModelComparisonService(db=db)
        summary_service = SummaryAnalysisService()
        dashboard_service = DashboardService()
        print("[INFO] ✅ 其他服务初始化成功")
    except Exception as e:
        print(f"[WARNING] 部分服务初始化失败: {e}")
        print("[WARNING] 登录功能仍可使用")
        comparison_service = None
        summary_service = None
        dashboard_service = None
    
    print("[INFO] 服务初始化完成（CLIP模型将在首次检测时加载）")

# 在后台线程中启动服务初始化
threading.Thread(target=initialize_services_async, daemon=True).start()
print("[INFO] 服务将在后台初始化，登录功能立即可用")

# 默认参数
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
AVAILABLE_MODELS = [
    ("runwayml/stable-diffusion-v1-5", "SD基础模型 (Stable Diffusion v1.5)"),
    ("openai-clip-fusion", "OpenAI CLIP融合模型"),
    ("itsc-gan-fusion", "ITSC-GAN融合模型")
]
DEFAULT_NUM_STEPS = 32
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_THRESHOLD = 0.3

# ==================== 辅助函数 ====================
def get_model_id_from_display(display_name: str) -> str:
    """将模型显示名称转换为模型ID"""
    for model_id, display in AVAILABLE_MODELS:
        if display == display_name:
            return model_id
    # 如果找不到，尝试直接使用传入的值（可能是模型ID）
    if display_name in [m[0] for m in AVAILABLE_MODELS]:
        return display_name
    return DEFAULT_MODEL

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
    
    style = f"display:block;max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;"
    img_html = f"<img src='data:image/png;base64,{img_str}' style='{style}' />"
    return img_html

# ==================== 用户认证 ====================
def login_user(username_or_email, password):
    """用户登录（优化：快速响应，避免阻塞）"""
    import time
    start = time.time()
    
    try:
        print(f"[DEBUG] login_user: 开始验证用户 {username_or_email}")
        
        # 快速验证参数
        if not username_or_email or not password:
            print(f"[DEBUG] login_user: 参数为空")
            return None, "❌ 请输入用户名和密码", False
        
        # 验证用户（设置超时保护）
        user_id = None
        try:
            user_id = db.verify_user(username_or_email, password)
            elapsed = time.time() - start
            print(f"[DEBUG] login_user: verify_user 完成，耗时 {elapsed:.3f}秒, user_id={user_id}")
        except Exception as db_error:
            elapsed = time.time() - start
            print(f"[ERROR] login_user: 数据库验证失败，耗时 {elapsed:.3f}秒: {db_error}")
            import traceback
            traceback.print_exc()
            return None, f"❌ 数据库连接失败: {str(db_error)}", False
        
        if user_id:
            # 获取用户名（如果失败，使用默认值）
            try:
                username = db.get_username_by_id(user_id)
                if not username:
                    username = username_or_email
            except Exception as e:
                print(f"[WARNING] 获取用户名失败: {e}，使用输入的用户名")
                username = username_or_email
            
            total_elapsed = time.time() - start
            print(f"[DEBUG] login_user: 登录成功，总耗时 {total_elapsed:.3f}秒")
            return user_id, f"✅ 登录成功！欢迎，{username}！", True
        else:
            total_elapsed = time.time() - start
            print(f"[DEBUG] login_user: 登录失败（用户名或密码错误），总耗时 {total_elapsed:.3f}秒")
            return None, "❌ 用户名或密码错误", False
            
    except Exception as e:
        total_elapsed = time.time() - start
        error_msg = f"登录过程出错: {str(e)}"
        print(f"[ERROR] login_user: {error_msg}，总耗时 {total_elapsed:.3f}秒")
        import traceback
        traceback.print_exc()
        return None, f"❌ {error_msg}", False

def register_user(username, email, password, confirm_password):
    """用户注册"""
    if not username or not password:
        return "❌ 用户名和密码不能为空", False
    
    if len(password) < 6:
        return "❌ 密码长度至少6个字符", False
    
    if password != confirm_password:
        return "❌ 两次输入的密码不一致", False
    
    try:
        success, message = db.register_user(username, password, email=email)
        if success:
            return "✅ 注册成功！请返回登录页面登录。", True
        else:
            # 使用数据库返回的具体错误消息
            return f"❌ {message}", False
    except Exception as e:
        return f"❌ 注册失败: {str(e)}", False

# ==================== 图像生成与检测 ====================
def generate_image(
    prompt: str,
    model_name: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int,
    progress_callback=None
):
    """生成图像（支持进度回调）"""
    if not user_id:
        error_html = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
        return "", error_html, None, None, None
    
    # 检查服务是否已初始化
    if generation_service is None:
        error_html = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 图像生成服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
        return "", error_html, None, None, None
    
    try:
        # 转换模型显示名称为模型ID
        model_id = get_model_id_from_display(model_name)
        print(f"[DEBUG] 模型选择: 显示名称='{model_name}' -> 模型ID='{model_id}'")
        
        # 生成图像（带进度回调）
        print(f"[INFO] 开始使用模型 '{model_id}' 生成图像...")
        image = generation_service.generate(
            prompt=prompt,
            model_name=model_id,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            progress_callback=progress_callback
        )
        
        # 保存图像
        image_path = generation_service.save_image(image, prompt, model_id)
        
        # 转换为HTML
        img_html = pil_to_base64_html(image, max_width=600, max_height=400)
        
        return img_html, image_path, prompt, model_id, threshold
        
    except Exception as e:
        error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 生成失败: {str(e)}</h3></div>"
        return "", error_html, None, None, None


def detect_consistency(image_path: str, prompt: str, threshold: float, model_name: str, user_id: int):
    """检测一致性"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    # 检查服务是否已初始化
    if detection_service is None:
        return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 一致性检测服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
    
    try:
        image = Image.open(image_path)
        # 传递模型名称给检测服务，用于计算模型特定的分数
        is_consistent, score, detail = detection_service.detect(image, prompt, threshold, model_name=model_name)
        
        # 保存到数据库
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
        
        # 生成结果HTML
        clip_score = detail.get('clip_score', score)
        fused_score = detail.get('fused_score', score)
        model_specific_score = detail.get('model_specific_score', score)
        model_name_display = detail.get('model_name', model_name)
        
        # 获取模型显示名称
        if model_name_display:
            model_display_name = dict(AVAILABLE_MODELS).get(model_name_display, model_name_display)
            if not model_display_name or model_display_name == model_name_display:
                # 如果找不到，尝试反向查找
                for model_id, display in AVAILABLE_MODELS:
                    if model_id == model_name_display:
                        model_display_name = display
                        break
                else:
                    model_display_name = model_name_display
        else:
            model_display_name = "未知模型"
        
        status_icon = "✅" if is_consistent else "❌"
        status_color = "#4CAF50" if is_consistent else "#F44336"
        
        # 计算分数差异和提升率
        score_diff = model_specific_score - score
        improvement_rate = ((fused_score - clip_score) / clip_score * 100) if clip_score > 0 else 0.0
        
        result_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin-top: 0; display: flex; align-items: center; gap: 10px;">
                {status_icon} 一致性检测结果
            </h2>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
                <p style="margin: 8px 0;"><strong>📝 提示词:</strong> {prompt}</p>
                <p style="margin: 8px 0;"><strong>🤖 使用模型:</strong> {model_display_name}</p>
                <p style="margin: 8px 0;"><strong>🎯 一致性状态:</strong> <span style="color: {status_color}; font-weight: bold; font-size: 18px;">{'通过' if is_consistent else '未通过'}</span></p>
                <p style="margin: 8px 0;"><strong>📊 一致性阈值:</strong> <span style="color: #FFD700; font-weight: bold;">{threshold:.4f}</span></p>
                <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
                <h3 style="margin: 15px 0 10px 0; font-size: 16px;">🔍 检测分数详情：</h3>
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
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 6px; margin-top: 10px;">
                    <p style="margin: 5px 0; font-size: 14px;"><strong>⭐ 最终分数</strong></p>
                    <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: #FFD700;">{model_specific_score:.4f}</p>
                </div>
                {f'<p style="margin: 10px 0; font-size: 14px; color: #90EE90;"><strong>📈 融合提升率:</strong> {improvement_rate:+.2f}%</p>' if improvement_rate != 0 else ''}
            </div>
            <p style="text-align: right; margin: 10px 0 0 0; font-size: 12px; opacity: 0.9;">✅ 结果已记录到历史记录</p>
        </div>
        """
        
        return result_html
        
    except Exception as e:
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 检测失败: {str(e)}</h3></div>"


def compare_models_sync(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int
):
    """同步对比多个模型（保留向后兼容）"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    # 检查服务是否已初始化
    if comparison_service is None:
        return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 模型对比服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
    
    try:
        # 获取所有模型
        model_names = [m[0] for m in AVAILABLE_MODELS]
        
        # 对比模型 - 使用带进度回调的版本
        progress_info = {"progress": 0, "status": "初始化", "current_model": ""}
        
        def progress_callback(progress, status="", model_name=""):
            progress_info["progress"] = progress
            progress_info["status"] = status
            progress_info["current_model"] = model_name
        
        comparison_results = comparison_service.compare_models(
            prompt=prompt,
            model_names=model_names,
            threshold=threshold,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            progress_callback=progress_callback
        )
        
        # 生成对比HTML
        comparison_html = generate_comparison_html(comparison_results)
        
        return comparison_html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {str(e)}</h3></div>"

# 全局任务ID跟踪
global_comparison_tasks = {}

def compare_models_async_web(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    user_id: int,
    progress=None
):
    """异步对比多个模型（用于Web界面）"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>", ""
    
    # 检查服务是否已初始化
    if comparison_service is None:
        error_msg = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 模型对比服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
        return error_msg, error_msg
    
    try:
        # 获取所有模型
        model_names = [m[0] for m in AVAILABLE_MODELS]
        
        # 初始化进度条（如果progress不为None）
        if progress is not None:
            try:
                progress(0, desc="开始模型对比...")
            except Exception as e:
                print(f"[WARNING] 进度条初始化失败: {e}")
        
        # 生成唯一的任务ID
        task_id = str(uuid.uuid4())
        
        # 进度回调函数（用于更新Gradio进度条）
        def progress_callback(progress_value, status=""):
            if progress is not None:
                try:
                    desc = f"处理中: {status}"
                    progress(progress_value / 100, desc=desc)
                except Exception as e:
                    print(f"[WARNING] 更新进度条失败: {e}")
        
        # 使用异步对比方法（注意：compare_models_async内部会创建自己的进度回调）
        # 我们需要通过get_comparison_status来获取进度
        comparison_service.compare_models_async(
            task_id=task_id,
            prompt=prompt,
            model_names=model_names,
            threshold=threshold,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            user_id=user_id,
            reuse_cache=True  # 启用图片复用
        )
        
        # 存储任务ID
        global_comparison_tasks[task_id] = {
            "start_time": datetime.now(),
            "prompt": prompt,
            "user_id": user_id
        }
        
        # 轮询任务状态
        max_wait_time = 600  # 最多等待10分钟
        start_time = time.time()
        last_progress = 0.0
        
        while True:
            status = comparison_service.get_comparison_status(task_id)
            
            # 更新进度条（如果提供了progress对象）
            if progress is not None:
                current_progress = status.get("progress", 0.0)
                if current_progress > last_progress:
                    progress(current_progress / 100, desc=status.get("message", "处理中..."))
                    last_progress = current_progress
            
            if status["status"] == "completed":
                if progress is not None:
                    progress(1.0, desc="对比完成！")
                # 生成对比HTML
                comparison_html = generate_comparison_html(status["result"])
                
                # 生成总结（如果服务可用）
                summary_html = ""
                if summary_service is not None:
                    try:
                        summary_result = summary_service.generate_summary(status["result"], include_charts=True)
                        summary_html = summary_result.get('summary_html', '')
                    except Exception as e:
                        print(f"[WARNING] 生成总结失败: {e}")
                
                # 清理任务
                if task_id in global_comparison_tasks:
                    del global_comparison_tasks[task_id]
                
                return comparison_html, summary_html
            elif status["status"] == "error" or status["status"] == "failed":
                error_msg = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {status.get('error', '未知错误')}</h3></div>"
                # 清理任务
                if task_id in global_comparison_tasks:
                    del global_comparison_tasks[task_id]
                return error_msg, error_msg
            elif status["status"] == "not_found":
                error_msg = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 任务不存在或已过期</h3></div>"
                return error_msg, error_msg
            
            # 检查超时
            if time.time() - start_time > max_wait_time:
                error_msg = "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 任务超时，请重试</h3></div>"
                # 清理任务
                if task_id in global_comparison_tasks:
                    del global_comparison_tasks[task_id]
                return error_msg, error_msg
            
            # 等待一小段时间再查询
            time.sleep(0.5)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {str(e)}</h3></div>"
        return error_msg, error_msg


def _create_comparison_visuals(valid_results):
    """
    基于多模型对比结果，生成两张可视化图片：
    1）多模型生成效果对比图（横向拼接）
    2）模型一致性评分对比柱状图
    """
    try:
        # 准备模型名称、图像和分数
        model_labels = []
        images = []
        scores = []
        
        for result in valid_results:
            model_id = result.get('model_name', '')
            # 获取模型显示名称（只使用中文部分，去掉英文）
            model_display_name = None
            for mid, display in AVAILABLE_MODELS:
                if mid == model_id:
                    # 提取中文名称（去掉括号和英文部分）
                    display_name = display
                    # 如果包含括号，只取括号前的中文部分
                    if '(' in display_name:
                        model_display_name = display_name.split('(')[0].strip()
                    else:
                        model_display_name = display_name
                    break
            
            # 如果没有找到，使用模型ID的简化名称
            if not model_display_name:
                if 'stable-diffusion' in model_id.lower() or 'runwayml' in model_id.lower() or model_id == 'sd-base':
                    model_display_name = "SD基础模型"
                elif 'clip' in model_id.lower() or 'openai' in model_id.lower():
                    model_display_name = "CLIP融合模型"
                elif 'itsc' in model_id.lower() or 'gan' in model_id.lower():
                    model_display_name = "ITSC-GAN融合模型"
                else:
                    model_display_name = model_id
            
            image = result.get('image')
            if image is None:
                continue
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            detail = result.get('detail', {})
            model_specific_score = detail.get('model_specific_score', result.get('overall_score', 0.0))
            
            # 清理模型显示名称，移除乱码和英文部分
            import re
            clean_display_name = model_display_name.strip()
            # 移除可能存在的乱码字符（如☐等）
            clean_display_name = re.sub(r'[☐☑☒]', '', clean_display_name)
            # 如果包含括号，只保留括号前的中文部分
            if '(' in clean_display_name:
                clean_display_name = clean_display_name.split('(')[0].strip()
            
            model_labels.append(clean_display_name)
            images.append(image)
            scores.append(float(model_specific_score))
        
        if not images:
            return None, None
        
        # 通用字体设置
        try:
            title_font = ImageFont.truetype("arial.ttf", 26)
            label_font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # ===== 方法一：生成效果对比拼接图 =====
        max_height = 320
        padding = 20
        resized_images = []
        for img in images:
            im = img.copy()
            im.thumbnail((400, max_height))
            resized_images.append(im)
        
        total_width = padding * (len(resized_images) + 1) + sum(im.width for im in resized_images)
        composite_height = max_height + 90  # 预留标题和标签空间
        composite = Image.new("RGB", (total_width, composite_height), (245, 248, 252))
        draw_comp = ImageDraw.Draw(composite)
        
        # 标题
        comp_title = "对比方法一：多模型生成效果对比图"
        bbox = draw_comp.textbbox((0, 0), comp_title, font=title_font)
        title_w = bbox[2] - bbox[0]
        draw_comp.text(
            ((total_width - title_w) // 2, 10),
            comp_title,
            fill="#333333",
            font=title_font,
        )
        
        # 逐个粘贴图像并绘制模型名称+分数
        x = padding
        for im, label, score in zip(resized_images, model_labels, scores):
            y = 50
            composite.paste(im, (x, y))
            
            # 文本：模型名称 + 分数（只显示中文名称）
            # 清理标签，移除任何特殊字符或乱码
            clean_label = label.strip()
            # 移除可能存在的乱码字符（如☐等）
            import re
            clean_label = re.sub(r'[☐☑☒]', '', clean_label)
            # 如果包含括号，只保留括号前的中文部分
            if '(' in clean_label:
                clean_label = clean_label.split('(')[0].strip()
            
            text = f"{clean_label} ({score:.2f})"
            tb = draw_comp.textbbox((0, 0), text, font=label_font)
            text_w = tb[2] - tb[0]
            tx = x + (im.width - text_w) // 2
            ty = y + im.height + 8
            draw_comp.text((tx, ty), text, fill="#111827", font=label_font)
            
            x += im.width + padding
        
        # ===== 方法二：一致性评分柱状图 =====
        chart_width, chart_height = 900, 420
        chart = Image.new("RGB", (chart_width, chart_height), (248, 250, 255))
        draw_chart = ImageDraw.Draw(chart)
        
        chart_title = "对比方法二：模型语义一致性评分对比图"
        cb = draw_chart.textbbox((0, 0), chart_title, font=title_font)
        ct_w = cb[2] - cb[0]
        draw_chart.text(
            ((chart_width - ct_w) // 2, 10),
            chart_title,
            fill="#111827",
            font=title_font,
        )
        
        # 坐标系参数
        margin_left = 80
        margin_right = 40
        margin_bottom = 80
        margin_top = 60
        
        x0 = margin_left
        y0 = margin_top
        x1 = chart_width - margin_right
        y1 = chart_height - margin_bottom
        
        # 画坐标轴
        draw_chart.line((x0, y0, x0, y1), fill="#9CA3AF", width=2)
        draw_chart.line((x0, y1, x1, y1), fill="#9CA3AF", width=2)
        
        if scores:
            max_score = max(max(scores), 0.01)
        else:
            max_score = 1.0
        
        # y 轴刻度（0.0 ~ 1.0，步长0.1）
        for i in range(0, 11):
            val = i / 10.0
            y = y1 - (y1 - y0) * (val / 1.0)
            draw_chart.line((x0 - 5, y, x0, y), fill="#D1D5DB", width=1)
            label = f"{val:.1f}"
            lb = draw_chart.textbbox((0, 0), label, font=small_font)
            lw = lb[2] - lb[0]
            lh = lb[3] - lb[1]
            draw_chart.text((x0 - 10 - lw, y - lh / 2), label, fill="#6B7280", font=small_font)
        
        # 柱子
        n = len(scores)
        if n > 0:
            bar_area_width = x1 - x0
            bar_width = bar_area_width / (n * 2.0)
            colors = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#3B82F6"]
            
            for i, (label, score) in enumerate(zip(model_labels, scores)):
                center_x = x0 + (2 * i + 1) * bar_width
                bar_height = (score / 1.0) * (y1 - y0)  # 映射到 0~1 区间高度
                bx0 = center_x - bar_width * 0.7
                bx1 = center_x + bar_width * 0.7
                by1 = y1
                by0 = y1 - bar_height
                
                color = colors[i % len(colors)]
                draw_chart.rectangle((bx0, by0, bx1, by1), fill=color, outline=color)
                
                # 分数文本
                score_text = f"{score:.2f}"
                sb = draw_chart.textbbox((0, 0), score_text, font=small_font)
                sw = sb[2] - sb[0]
                sh = sb[3] - sb[1]
                draw_chart.text(
                    (center_x - sw / 2, by0 - sh - 2),
                    score_text,
                    fill="#111827",
                    font=small_font,
                )
                
                # x 轴标签（模型简称，清理乱码并只显示中文）
                import re
                # 清理标签，移除乱码字符
                clean_label = label.strip()
                clean_label = re.sub(r'[☐☑☒]', '', clean_label)
                # 如果包含括号，只保留括号前的中文部分
                if '(' in clean_label:
                    clean_label = clean_label.split('(')[0].strip()
                # 简化标签（移除"模型"、"融合"等词）
                short_label = clean_label.replace("模型", "").replace("Stable Diffusion", "SD")
                short_label = short_label.replace("融合", "").replace("基础", "").strip()
                # 如果标签为空，使用原始标签的简化版本
                if not short_label:
                    short_label = clean_label[:10]  # 最多10个字符
                xb = draw_chart.textbbox((0, 0), short_label, font=small_font)
                xw = xb[2] - xb[0]
                draw_chart.text(
                    (center_x - xw / 2, y1 + 10),
                    short_label,
                    fill="#374151",
                    font=small_font,
                )
        
        return composite, chart
    except Exception as e:
        print(f"[WARNING] 创建对比可视化图失败: {e}")
        return None, None


def generate_comparison_html(comparison_results: dict) -> str:
    """生成对比结果HTML，显示每个模型的特定分数，并提供两种对比方法的效果图"""
    results = comparison_results.get('results', [])
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        # 所有模型都失败了，显示详细错误信息
        error_messages = []
        for result in results:
            if 'error' in result:
                model_name = result.get('model_name', '未知模型')
                error_msg = result.get('error', '未知错误')
                error_messages.append(f"<li><strong>{model_name}:</strong> {error_msg}</li>")
        
        error_html = "<div style='padding: 20px; background: #fef2f2; border-radius: 10px; border: 2px solid #fecaca;'>"
        error_html += "<h3 style='color: #dc2626; margin-top: 0;'>❌ 所有模型生成失败</h3>"
        if error_messages:
            error_html += "<p style='color: #991b1b;'><strong>错误详情：</strong></p>"
            error_html += f"<ul style='color: #991b1b;'>{''.join(error_messages)}</ul>"
        else:
            error_html += "<p style='color: #991b1b;'>所有模型都未能成功生成图像。</p>"
        
        # 检查是否是CUDA问题
        cuda_errors = [e for e in error_messages if 'CUDA' in str(e) or 'Torch not compiled' in str(e)]
        if cuda_errors:
            error_html += "<div style='margin-top: 15px; padding: 15px; background: #fff7ed; border-radius: 8px; border-left: 4px solid #f59e0b;'>"
            error_html += "<p style='color: #92400e; margin: 0;'><strong>💡 解决方案：</strong></p>"
            error_html += "<ul style='color: #92400e; margin: 5px 0 0 0;'>"
            error_html += "<li>您的PyTorch未编译CUDA支持，系统已自动切换到CPU模式</li>"
            error_html += "<li>CPU模式运行较慢，但可以正常使用</li>"
            error_html += "<li>如需使用GPU，请安装支持CUDA的PyTorch版本</li>"
            error_html += "</ul></div>"
        
        error_html += "</div>"
        return error_html
    
    # 生成两张汇总对比图（方法一：生成效果；方法二：评分柱状图）
    composite_img, chart_img = _create_comparison_visuals(valid_results)
    composite_html = ""
    chart_html = ""
    if composite_img is not None:
        composite_html = pil_to_base64_html(composite_img, max_width=1000, max_height=500)
    if chart_img is not None:
        chart_html = pil_to_base64_html(chart_img, max_width=1000, max_height=500)
    
    # 生成每个模型的详细卡片
    cards_html = ""
    for result in valid_results:
        # 获取模型显示名称（清理乱码）
        model_id = result.get('model_name', '')
        model_display_name = None
        for mid, display in AVAILABLE_MODELS:
            if mid == model_id:
                # 提取中文名称（去掉括号和英文部分）
                display_name = display
                if '(' in display_name:
                    model_display_name = display_name.split('(')[0].strip()
                else:
                    model_display_name = display_name
                break
        
        # 如果没有找到，使用模型ID的简化名称
        if not model_display_name:
            if 'stable-diffusion' in model_id.lower() or 'runwayml' in model_id.lower() or model_id == 'sd-base':
                model_display_name = "SD基础模型"
            elif 'clip' in model_id.lower() or 'openai' in model_id.lower():
                model_display_name = "CLIP融合模型"
            elif 'itsc' in model_id.lower() or 'gan' in model_id.lower():
                model_display_name = "ITSC-GAN融合模型"
            else:
                model_display_name = model_id
        
        # 清理乱码字符
        import re
        model_display_name = re.sub(r'[☐☑☒]', '', model_display_name).strip()
        
        # 获取模型特定分数
        detail = result.get('detail', {})
        model_specific_score = detail.get('model_specific_score', result.get('overall_score', 0.0))
        clip_score = result.get('clip_score', 0.0)
        fused_score = result.get('fused_score', 0.0)
        is_consistent = result.get('is_consistent', False)
        image = result.get('image')
        
        status_icon = "✅" if is_consistent else "❌"
        status_color = "#4CAF50" if is_consistent else "#F44336"
        
        img_html = ""
        if image:
            img_html = pil_to_base64_html(image, max_width=400, max_height=300)
        
        cards_html += f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">🤖 {model_display_name}</h3>
            <div style="margin: 15px 0; text-align: center;">
                {img_html}
            </div>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                <p style="margin: 5px 0;"><strong>一致性状态:</strong> <span style="color: {status_color}; font-weight: bold;">{status_icon} {'通过' if is_consistent else '未通过'}</span></p>
                <hr style="border: 1px solid #ddd; margin: 10px 0;">
                <p style="margin: 5px 0;"><strong>最终分数:</strong> <span style="font-size: 20px; font-weight: bold; color: #FF6B6B;">{model_specific_score:.4f}</span></p>
            </div>
        </div>
        """
    
    # 整体HTML：先展示两个对比方法的效果图，再展示每个模型详情
    return f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="margin-top: 0;">🔍 多模型对比结果</h2>
        <p><strong>提示词:</strong> {comparison_results.get('prompt', '')}</p>
    </div>
    
    <div style="display: flex; flex-direction: column; gap: 24px; margin-bottom: 24px;">
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h3 style="margin: 0 0 12px 0; color: #111827;">🖼️ 对比方法一：生成效果对比图</h3>
            <div style="text-align: center;">
                {composite_html}
            </div>
        </div>
        <div style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h3 style="margin: 0 0 12px 0; color: #111827;">📊 对比方法二：一致性评分对比图</h3>
            <div style="text-align: center;">
                {chart_html}
            </div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
        {cards_html}
    </div>
    """

# ==================== 历史记录 ====================
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
            
            image_preview = "<div style='text-align: center; margin: 10px 0;'><p style='color: #777; font-style: italic;'>图像预览</p></div>"
            
            if record['image_path'] and os.path.exists(record['image_path']):
                try:
                    image = Image.open(record['image_path'])
                    image_preview = pil_to_base64_html(image, max_width=300, max_height=200)
                except Exception as e:
                    print(f"[WARNING] 无法加载图像: {e}")
            
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

# ==================== 多模型对比 ====================
# 注意：detect_consistency函数已在上面定义（153行），这里不再重复定义

def compare_steps_sync_old(image_path, prompt, threshold, user_id):
    """检测图像一致性"""
    if not image_path or not os.path.exists(image_path):
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 图像不存在，请先生成图像</h3></div>"
    
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        current_system = get_system()
        
        is_consistent, score, detail = current_system.detector.detect_consistency(
            image, prompt, threshold
        )
        
        clip_score = detail.get('clip_score', score) if isinstance(detail, dict) else score
        fused_score = detail.get('fused_score', score) if isinstance(detail, dict) else score
        
        improvement_rate = ((fused_score - clip_score) / clip_score * 100) if clip_score > 0 else 0.0
        
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
        
        status_icon = "✅" if is_consistent else "❌"
        status_text = "通过" if is_consistent else "未通过"
        score_color = "#4CAF50" if score >= 0.7 else "#FF9800" if score >= 0.4 else "#F44336"
        improvement_color = "#4CAF50" if improvement_rate > 0 else "#F44336" if improvement_rate < 0 else "#666"
        improvement_icon = "📈" if improvement_rate > 0 else "📉" if improvement_rate < 0 else "➡️"
        model_display_name = "一致性模型"
        
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
        </div>
    </div>
</div>
"""
        
        return result_html
        
    except Exception as e:
        error_msg = f"❌ 检测失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"

# ==================== 多模型对比 ====================
def compare_steps_sync(
    prompt: str,
    threshold: float,
    num_steps: int,
    guidance_scale: float,
    user_id: int
):
    """多步数对比（对比不同推理步数的效果）"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    if not prompt or not prompt.strip():
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请输入提示词</h3></div>"
    
    # 对比不同推理步数（1-4步）
    step_list = [1, 2, 3, 4]
    results = []
    
    try:
        current_system = get_system()
        
        for steps in step_list:
            print(f"[INFO] 使用推理步数 {steps} 生成图像...")
            
            image = current_system.generator.generate(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )
            
            if image is None:
                continue
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            is_consistent, score, detail = current_system.detector.detect_consistency(
                image, prompt, threshold
            )
            
            clip_score = detail.get('clip_score', score) if isinstance(detail, dict) else score
            fused_score = detail.get('fused_score', score) if isinstance(detail, dict) else score
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            filename = f"{timestamp}_steps{steps}_{safe_filename}.png"
            os.makedirs("output", exist_ok=True)
            image_path = os.path.join("output", filename)
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
                        prompt=f"[对比-{steps}步] {prompt}",
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
<div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
    <h2 style="margin-top: 0; display: flex; align-items: center; gap: 10px;">
        🔍 多步数对比结果
    </h2>
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 10px 0;">
        <p style="margin: 8px 0;"><strong>📝 提示词:</strong> {prompt}</p>
        <p style="margin: 8px 0;"><strong>📊 对比步数:</strong> {len(results)} 种（1-4步）</p>
        <p style="margin: 8px 0;"><strong>🎯 一致性阈值:</strong> {threshold}</p>
    </div>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 20px;">
"""
        
        for result in results:
            status_color = "#4CAF50" if result['is_consistent'] else "#F44336"
            status_text = "通过" if result['is_consistent'] else "未通过"
            status_icon = "✅" if result['is_consistent'] else "❌"
            
            comparison_html += f"""
    <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0; color: #333;">推理步数: {result['steps']} 步</h3>
        <div style="margin: 15px 0;">
            {result['image_html']}
        </div>
        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <p style="margin: 5px 0;"><strong>🔗 CLIP分数:</strong> <span style="color: #2196F3; font-weight: bold;">{result['clip_score']:.4f}</span></p>
            <p style="margin: 5px 0;"><strong>📊 整体分数:</strong> <span style="color: {status_color}; font-weight: bold; font-size: 18px;">{result['overall_score']:.4f}</span></p>
            <p style="margin: 5px 0;"><strong>🎯 一致性状态:</strong> <span style="background: {status_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">{status_icon} {status_text}</span></p>
        </div>
    </div>
"""
        
        comparison_html += """
</div>

<div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 20px;">
    <h3 style="margin-top: 0; color: #333;">📊 对比统计</h3>
"""
        
        comparison_html += f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
            <p style="margin: 5px 0; color: #666; font-size: 14px;">平均CLIP分数</p>
            <p style="margin: 5px 0; color: #2196F3; font-size: 24px; font-weight: bold;">{avg_clip:.4f}</p>
        </div>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 8px;">
            <p style="margin: 5px 0; color: #666; font-size: 14px;">平均整体分数</p>
            <p style="margin: 5px 0; color: #2196F3; font-size: 24px; font-weight: bold;">{avg_overall:.4f}</p>
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
        print(f"[ERROR] 对比失败: {str(e)}")
        print(error_traceback)
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {str(e)}</h3></div>"

# ==================== 仪表盘功能 ====================
def get_dashboard_data(user_id):
    """获取仪表盘数据"""
    if not user_id:
        return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
    
    try:
        dashboard_html = dashboard_service.generate_dashboard_html(user_id)
        return dashboard_html
    except Exception as e:
        error_msg = f"❌ 获取仪表盘数据失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{error_msg}</h3></div>"

def generate_dashboard_html(stats, hourly_stats, model_stats, keywords):
    """生成仪表盘HTML"""
    import json
    from datetime import datetime, timedelta
    
    # 准备12小时数据（填充缺失的小时）
    hours_data = {}
    now = datetime.now()
    for i in range(12):
        hour_time = now - timedelta(hours=11-i)
        hour_key = hour_time.strftime('%Y-%m-%d %H:00')
        hours_data[hour_key] = 0
    
    for item in hourly_stats:
        hours_data[item['hour']] = item['count']
    
    # 准备图表数据
    chart_labels = list(hours_data.keys())
    chart_data = list(hours_data.values())
    
    # 模型统计数据
    model_labels = list(model_stats.keys())
    model_values = list(model_stats.values())
    
    # 关键词统计（简单统计）
    keyword_counts = {}
    for keyword in keywords:
        if len(keyword) >= 2:  # 至少2个字符
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # 取前20个关键词
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    dashboard_html = f"""
    <div style="display: flex; min-height: 100vh; background: #f5f5f5;">
        <!-- 左侧导航栏 -->
        <div style="width: 250px; background: #1e293b; color: white; padding: 20px; position: fixed; height: 100vh; overflow-y: auto;">
            <h2 style="margin: 0 0 30px 0; font-size: 24px; font-weight: bold;">🎨 图像生成系统</h2>
            <div style="margin-bottom: 30px;">
                <div style="padding: 12px; background: #3b82f6; border-radius: 8px; margin-bottom: 10px; cursor: pointer;">
                    📊 仪表盘
                </div>
                <div style="padding: 12px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; opacity: 0.8;">
                    ✨ 语义生成图像
                </div>
                <div style="padding: 12px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; opacity: 0.8;">
                    🔍 多模型对比
                </div>
                <div style="padding: 12px; border-radius: 8px; margin-bottom: 5px; cursor: pointer; opacity: 0.8;">
                    📊 历史记录
                </div>
            </div>
        </div>
        
        <!-- 主内容区域 -->
        <div style="margin-left: 250px; flex: 1; padding: 30px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
                <h1 style="margin: 0; color: #333; font-size: 28px;">仪表盘</h1>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <select style="padding: 8px 15px; border: 1px solid #ddd; border-radius: 5px;">
                        <option>Admin</option>
                    </select>
                    <span style="color: #666; font-size: 14px;">图像生成系统/仪表盘</span>
                </div>
            </div>
            
            <!-- 四个指标卡片 -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #fbbf24;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 24px;">⭐</span>
                        <span style="color: #666; font-size: 14px;">总生成次数</span>
                    </div>
                    <div style="font-size: 32px; font-weight: bold; color: #333;">{stats['total_generations']}条</div>
                </div>
                
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #ef4444;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 24px;">🕐</span>
                        <span style="color: #666; font-size: 14px;">今日生成</span>
                    </div>
                    <div style="font-size: 32px; font-weight: bold; color: #333;">{stats['today_generations']}条</div>
                </div>
                
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #8b5cf6;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 24px;">✅</span>
                        <span style="color: #666; font-size: 14px;">一致性通过</span>
                    </div>
                    <div style="font-size: 32px; font-weight: bold; color: #333;">{stats['consistent_count']}条</div>
                </div>
                
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #6b7280;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 24px;">📊</span>
                        <span style="color: #666; font-size: 14px;">平均一致性分数</span>
                    </div>
                    <div style="font-size: 32px; font-weight: bold; color: #333;">{stats['average_score']:.2f}</div>
                </div>
            </div>
            
            <!-- 12小时内数据分布折线图 -->
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;">
                <h3 style="margin: 0 0 20px 0; color: #333; font-size: 18px;">12小时内生成数据量分布</h3>
                <canvas id="lineChart" style="max-height: 300px;"></canvas>
            </div>
            
            <!-- 底部图表区域 -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px;">
                <!-- 模型使用占比环形图 -->
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0 0 20px 0; color: #333; font-size: 18px;">模型使用占比</h3>
                    <canvas id="doughnutChart" style="max-height: 300px;"></canvas>
                </div>
                
                <!-- 数据统计饼图 -->
                <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0 0 20px 0; color: #333; font-size: 18px;">数据统计占比</h3>
                    <canvas id="pieChart" style="max-height: 300px;"></canvas>
                </div>
            </div>
            
            <!-- 热词词云图 -->
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 20px 0; color: #333; font-size: 18px;">热词词云图</h3>
                <div id="wordCloud" style="min-height: 300px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center;">
"""
    
    # 添加词云关键词
    for keyword, count in top_keywords:
        size = min(24 + count * 2, 48)  # 根据频率调整大小
        color = f"hsl({hash(keyword) % 360}, 70%, 50%)"  # 根据关键词生成颜色
        dashboard_html += f'<span style="font-size: {size}px; color: {color}; font-weight: bold; padding: 5px;">{keyword}</span>'
    
    dashboard_html += """
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script>
        // 12小时内数据分布折线图
        const lineCtx = document.getElementById('lineChart');
        if (lineCtx) {
            new Chart(lineCtx, {
                type: 'line',
                data: {
                    labels: """ + json.dumps([label.split(' ')[1][:5] for label in chart_labels]) + """,
                    datasets: [{
                        label: '生成数量',
                        data: """ + json.dumps(chart_data) + """,
                        borderColor: 'rgb(59, 130, 246)',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        }
        
        // 模型使用占比环形图
        const doughnutCtx = document.getElementById('doughnutChart');
        if (doughnutCtx) {
            new Chart(doughnutCtx, {
                type: 'doughnut',
                data: {
                    labels: """ + json.dumps(model_labels) + """,
                    datasets: [{
                        data: """ + json.dumps(model_values) + """,
                        backgroundColor: [
                            'rgb(59, 130, 246)',
                            'rgb(239, 68, 68)',
                            'rgb(139, 92, 246)',
                            'rgb(251, 191, 36)',
                            'rgb(34, 197, 94)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true
                }
            });
        }
        
        // 数据统计饼图
        const pieCtx = document.getElementById('pieChart');
        if (pieCtx) {
            new Chart(pieCtx, {
                type: 'pie',
                data: {
                    labels: ['一致性通过', '一致性未通过'],
                    datasets: [{
                        data: [""" + str(stats['consistent_count']) + """, """ + str(stats['inconsistent_count']) + """],
                        backgroundColor: [
                            'rgb(34, 197, 94)',
                            'rgb(239, 68, 68)'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true
                }
            });
        }
    </script>
    """
    
    return dashboard_html

# ==================== 创建界面 ====================
def create_interface():
    """创建Gradio界面，保持HTML登录界面样式"""
    
    # 自定义CSS样式 - 精确水平分割布局
    custom_css = """
    /* Reset & 基础 */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    html, body { height: 100%; }
    
    body {
        font-family: "Inter", "Helvetica Neue", Helvetica, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
        background: linear-gradient(180deg, #eef1ff 0%, #eef1ff 100%);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: #222;
        min-height: 100vh;
        overflow: auto;
    }
    
    /* 主内容容器 */
    .main-content-container {
        display: flex;
        min-height: calc(100vh - 160px);
        gap: 20px;
        width: 100%;
        max-width: 1800px;  /* 增大最大宽度 */
        margin: 0 auto;
        padding: 0 20px;
        overflow-y: auto;
    }
    
    /* 左侧导航栏 - 优化样式 */
    .sidebar {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.06);
        min-width: 240px;
        max-width: 240px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .sidebar:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    /* 导航按钮样式 - 增强交互和视觉效果 */
    .nav-button {
        background: #ffffff !important;
        border: 2px solid #f1f5f9 !important;
        border-radius: 14px !important;
        padding: 15px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 12px !important;
        position: relative !important;
        overflow: hidden;
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 10 !important;
    }
    
    /* 添加图标装饰效果 */
    .nav-button::before {
        content: attr(data-icon);
        font-size: 18px;
        min-width: 24px;
        text-align: center;
        pointer-events: none;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-color: #e2e8f0 !important;
        color: #334155 !important;
        transform: translateX(6px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 激活状态的按钮 - 增强渐变效果 */
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3) !important;
        transform: translateX(4px) !important;
    }
    
    .nav-button.active:hover {
        transform: translateX(4px) !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* 登出按钮样式 - 优化设计 */
    .logout-button {
        margin-top: auto;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%) !important;
        color: #dc2626 !important;
        border: 2px solid #fecaca !important;
        border-radius: 14px !important;
        padding: 14px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }
    
    .logout-button:hover {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15) !important;
        border-color: #fca5a5 !important;
        transform: translateY(-2px) !important;
    }
    
    /* 右侧内容区域 - 卡片式布局优化 */
    .content-area {
        background: white;
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.06);
        overflow-y: auto;
        flex: 1;
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .content-area:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    /* 响应式设计 - 小屏幕适配 */
    @media (max-width: 1024px) {
        .main-content-container {
            flex-direction: column;
            height: auto;
            min-height: calc(100vh - 160px);
        }
        
        .sidebar {
            min-width: auto;
            max-width: none;
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: center;
            padding: 16px;
        }
        
        .nav-button {
            flex: 1 1 30%;
            min-width: 160px;
            justify-content: center;
            margin-bottom: 8px;
        }
        
        .logout-button {
            flex: 1 1 100%;
            margin-top: 8px;
        }
    }
    
    @media (max-width: 768px) {
        .nav-button {
            flex: 1 1 45%;
            min-width: 140px;
        }
    }
    
    /* 优化内容区域滚动条 */
    .content-area::-webkit-scrollbar {
        width: 8px;
    }
    
    .content-area::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    .content-area::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    .content-area::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* 生成结果区域 - 添加到对比按钮提示 */
    #add-to-compare-btn {
        border: 2px dashed #7c3aed !important;
        background: linear-gradient(90deg, #ede9fe, #e0e7ff) !important;
        color: #4c1d95 !important;
        font-weight: 700 !important;
        margin-top: 18px !important;
        position: relative;
        z-index: 5;
    }
    
    #add-to-compare-btn::after {
        content: "⬆ 生成后点击此处加入模型对比";
        display: block;
        font-size: 13px;
        color: #5b21b6;
        margin-top: 6px;
        text-align: center;
    }
    
    #add-to-compare-btn.pulse {
        animation: pulseGlow 1.5s ease;
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.4); }
        100% { box-shadow: 0 0 0 18px rgba(124, 58, 237, 0); }
    }
    
    #generated-image-preview {
        border-radius: 16px;
        border: 1px solid #e0e7ff;
        padding: 12px;
        background: #f8fafc;
    }
    
    /* 页面主容器：左右并列 */
    .login-container {
        min-height: 100vh;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px;
        gap: 32px;
        position: relative;
    }
    
    /* 中间竖线 */
    .login-container::before {
        content: '';
        position: absolute;
        width: 2px;
        height: 62%;
        left: 50%;
        top: 19%;
        transform: translateX(-50%);
        background: rgba(255, 255, 255, 0.22);
        border-radius: 2px;
        pointer-events: none;
        z-index: 1;
    }
    
    /* 左侧区域 */
    .left-panel, .right-panel {
        flex: 1 1 50%;
        max-width: 600px;
        min-width: 300px;
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .left-card {
        width: 100%;
        height: 520px;
        border-radius: 20px;
        padding: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(160deg, #5063f5 0%, #b86ff8 100%);
        box-shadow: 0 12px 40px rgba(11, 12, 30, 0.12), inset 0 -6px 40px rgba(255, 255, 255, 0.03);
        color: #fff;
    }
    
    /* 左侧卡片内容垂直排列 - 居中 */
    .left-content { 
        width: 100%; 
        max-width: 520px; 
        margin: 0 auto;
        text-align: center;
    }
    
    .logo-row { 
        display: flex; 
        align-items: center; 
        justify-content: center;
        gap: 12px; 
        margin-bottom: 18px; 
    }
    .icon { 
        width: 44px; 
        height: 44px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        font-size: 28px; 
    }
    
    /* 标题 - 居中 */
    .title {
        font-size: 28px;
        line-height: 1.25;
        font-weight: 700;
        margin-bottom: 12px;
        color: #fff;
        text-align: center;
    }
    
    /* 副标题 - 居中 */
    .subtitle {
        font-size: 16.5px;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.92);
        margin-bottom: 18px;
        text-align: center;
    }
    
    /* 功能列表 - 居中 */
    .feature-list { 
        list-style: none; 
        margin-top: 6px; 
        display: flex; 
        flex-direction: column; 
        gap: 8px; 
        font-size: 15px; 
        color: rgba(255, 255, 255, 0.95);
        text-align: left;
        max-width: 400px;
        margin-left: auto;
        margin-right: auto;
    }
    .feature-list .bullet { 
        display: inline-block; 
        width: 14px; 
        text-align: center; 
        margin-right: 8px; 
        color: rgba(255, 255, 255, 0.95); 
    }
    
    /* 右侧表单卡片 */
    .right-card {
        width: 100%;
        height: 520px;
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(10, 12, 30, 0.08);
        padding: 40px 48px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: stretch;
    }
    
    /* 确保左右卡片高度一致 */
    .left-card, .right-card {
        min-height: 520px;
        max-height: 520px;
    }
    
    /* 表单标题 */
    .form-title {
        text-align: center;
        font-size: 22px;
        margin-bottom: 28px;
        font-weight: 700;
        color: #222;
    }
    
    /* 表单样式 */
    .login-form { width: 100%; display: flex; flex-direction: column; gap: 16px; }
    
    /* 字段 */
    .field { display: flex; flex-direction: column; gap: 8px; }
    .field-label { 
        font-size: 13px; 
        color: #374151; 
        font-weight: 500;
    }
    
    /* 输入框 - 极简柔性设计语言 (Material+Neumorphism) */
    .input {
        height: 44px;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(229, 231, 235, 0.8);
        font-size: 14px;
        outline: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(135deg, #f8f9fa 0%, #f1f3f5 100%);
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 1px 2px rgba(0, 0, 0, 0.04);
        font-weight: 400;
        color: #222;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }
    .input:hover {
        background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%);
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.08),
            0 2px 4px rgba(0, 0, 0, 0.06);
        border-color: rgba(209, 213, 219, 0.9);
    }
    .input:focus { 
        border-color: transparent;
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 0 0 2px rgba(91, 103, 246, 0.2),
            0 6px 18px rgba(90, 80, 255, 0.18);
        color: #222;
    }
    .input::placeholder {
        color: #9ca3af;
    }
    
    
    /* 按钮行 */
    .btn-row {
        display: flex;
        gap: 14px;
        margin-top: 6px;
    }
    .btn { 
        flex: 1 1 50%; 
        height: 44px; 
        border-radius: 10px; 
        font-weight: 600; 
        border: none; 
        cursor: pointer; 
        font-size: 15px; 
        transition: transform .12s, box-shadow .12s; 
    }
    .btn:active { transform: translateY(1px); }
    
    /* 主按钮：渐变紫色 */
    .btn.primary {
        background: linear-gradient(90deg, #5b67f6, #a86ff7);
        color: #fff;
        box-shadow: 0 8px 18px rgba(88, 78, 255, 0.14);
    }
    .btn.primary:hover { filter: brightness(1.03); box-shadow: 0 12px 30px rgba(88, 78, 255, 0.16); }
    
    /* 次按钮：浅色风格 */
    .btn.ghost {
        background: linear-gradient(90deg, #f5f6fb, #ffffff);
        border: 1px solid #eee;
        color: #333;
    }
    .btn.ghost:hover { filter: brightness(0.98); box-shadow: 0 8px 18px rgba(20, 20, 20, 0.04); }
    
    /* Gradio 组件样式覆盖 */
    .login-card {
        width: 100%;
        max-width: 100%;
        background: transparent;
        padding: 0;
        color: #1f2937;
    }
    
    .custom-input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        height: 44px !important;
    }
    
    .custom-input::placeholder {
        color: #9ca3af !important;
    }
    
    .custom-input:hover {
        border-color: #cbd5e1 !important;
        background-color: #ffffff !important;
    }
    
    .custom-input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1), 
                    0 4px 12px rgba(102, 126, 234, 0.1) !important;
        background-color: #ffffff !important;
    }
    
    .gradio-button-primary,
    .gradio-button[variant="primary"],
    .btn.primary {
        background: linear-gradient(90deg, #5b67f6, #a86ff7) !important;
        color: #fff !important;
        box-shadow: 0 8px 18px rgba(88, 78, 255, 0.14) !important;
        border: none !important;
        border-radius: 10px !important;
        height: 44px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: transform .12s, box-shadow .12s, filter .12s !important;
    }
    
    .gradio-button-primary:hover,
    .gradio-button[variant="primary"]:hover,
    .btn.primary:hover {
        filter: brightness(1.03) !important;
        box-shadow: 0 12px 30px rgba(88, 78, 255, 0.16) !important;
        transform: none !important;
    }
    
    .gradio-button-primary:active,
    .gradio-button[variant="primary"]:active,
    .btn.primary:active {
        transform: translateY(1px) !important;
    }
    
    .gradio-textbox {
        position: relative;
    }
    
    /* Gradio 输入框 - 极简柔性设计语言 */
    .gradio-textbox {
        position: relative;
        width: 100%;
    }
    
    .gradio-textbox input,
    .gradio-textbox textarea {
        background: linear-gradient(135deg, #f8f9fa 0%, #f1f3f5 100%) !important;
        color: #222 !important;
        border: 1px solid rgba(229, 231, 235, 0.8) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        height: 44px !important;
        position: relative !important;
        font-size: 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        line-height: normal !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 1px 2px rgba(0, 0, 0, 0.04) !important;
        font-weight: 400 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    .gradio-textbox input::placeholder,
    .gradio-textbox textarea::placeholder {
        color: #9ca3af !important;
    }
    
    .gradio-textbox input:hover,
    .gradio-textbox textarea:hover {
        background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%) !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.08),
            0 2px 4px rgba(0, 0, 0, 0.06) !important;
        border-color: rgba(209, 213, 219, 0.9) !important;
    }
    
    .gradio-textbox input:focus,
    .gradio-textbox textarea:focus {
        border-color: transparent !important;
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 0 0 2px rgba(91, 103, 246, 0.2),
            0 6px 18px rgba(90, 80, 255, 0.18) !important;
        color: #222 !important;
        outline: none !important;
    }
    
    .gradio-textbox input.input,
    .gradio-textbox textarea.input {
        background: linear-gradient(135deg, #f8f9fa 0%, #f1f3f5 100%) !important;
        border: 1px solid rgba(229, 231, 235, 0.8) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 1px 2px rgba(0, 0, 0, 0.04) !important;
        color: #222 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    .gradio-textbox input.input:hover,
    .gradio-textbox textarea.input:hover {
        background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%) !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.08),
            0 2px 4px rgba(0, 0, 0, 0.06) !important;
        border-color: rgba(209, 213, 219, 0.9) !important;
    }
    
    .gradio-textbox input.input:focus,
    .gradio-textbox textarea.input:focus {
        border-color: transparent !important;
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important;
        box-shadow: 
            inset 0 0 0 1px rgba(0, 0, 0, 0.06),
            0 0 0 2px rgba(91, 103, 246, 0.2),
            0 6px 18px rgba(90, 80, 255, 0.18) !important;
        color: #222 !important;
    }
    
    .gradio-button-primary,
    .gradio-button[variant="primary"],
    .main-btn {
        background: linear-gradient(90deg, #55c0e8, #3b82f6) !important;
        color: white !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        z-index: 10 !important;
        position: relative !important;
    }
    
    .gradio-button-primary:hover,
    .gradio-button[variant="primary"]:hover,
    .main-btn:hover {
        opacity: 0.9 !important;
        box-shadow: 0 4px 18px rgba(59, 130, 246, 0.6) !important;
    }
    
    .gradio-button-primary:active,
    .gradio-button[variant="primary"]:active,
    .main-btn:active {
        transform: scale(0.98) !important;
    }
    
    .gradio-button:disabled,
    .main-btn:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    
    .gradio-textbox {
        margin-bottom: 0 !important;
    }
    
    /* 小屏适配：垂直堆叠 */
    @media (max-width: 980px) {
        .login-container { 
            flex-direction: column; 
            padding: 24px; 
        }
        .left-panel, .right-panel { 
            max-width: none; 
            width: 100%; 
        }
        .left-card, .right-card { 
            height: auto; 
            min-height: 360px; 
        }
        .login-container::before { 
            display: none; 
        }
        .left-card { 
            padding: 28px; 
            border-radius: 14px; 
        }
        .right-card { 
            padding: 28px; 
            border-radius: 14px; 
            margin-top: 18px; 
        }
    }
    
    /* Gradio容器铺满 */
    .gradio-container {
        width: 100% !important;
        height: 100vh !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    #component-0, #root {
        width: 100% !important;
        height: 100vh !important;
        max-width: 100% !important;
    }
    
    /* 标签页美化 */
    .gradio-tabs {
        width: 100% !important;
    }
    
    .gradio-tab-nav {
        justify-content: center !important;
        margin-bottom: 2rem !important;
    }
    
    .gradio-tab-nav button {
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* 表单间距优化 */
    .gradio-column {
        gap: 1.5rem !important;
    }
    
    /* 按钮美化 */
    .gradio-button-secondary,
    .btn.ghost {
        background: linear-gradient(90deg, #f5f6fb, #ffffff) !important;
        border: 1px solid #eee !important;
        border-radius: 10px !important;
        color: #333 !important;
        transition: transform .12s, box-shadow .12s, filter .12s !important;
        height: 44px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    .gradio-button-secondary:hover,
    .btn.ghost:hover {
        filter: brightness(0.98) !important;
        box-shadow: 0 8px 18px rgba(20, 20, 20, 0.04) !important;
        transform: none !important;
    }
    
    .gradio-button-secondary:active,
    .btn.ghost:active {
        transform: translateY(1px) !important;
    }
    
    #messageBox {
        text-align: center;
        font-size: 0.875rem;
        margin-top: 1.5rem;
        padding: 0.75rem;
        border-radius: 12px;
    }
    
    #messageBox.hidden {
        display: none;
    }
    
    #messageBox.bg-green-900 {
        background-color: #065f46;
        color: #6ee7b7;
    }
    
    #messageBox.bg-red-900 {
        background-color: #7f1d1d;
        color: #fca5a5;
    }
    """
    
    with gr.Blocks(title="图像生成与语义一致性检测系统", theme=gr.themes.Soft(), css=custom_css, fill_height=True) as demo:
        # 状态变量
        current_user_id = gr.State(value=None)
        login_status = gr.State(value=False)
        current_image_path = gr.State(value=None)
        current_prompt = gr.State(value=None)
        current_threshold = gr.State(value=None)
        # 待对比图像列表
        comparison_images_list = gr.State(value=[])
        
        # ========== 登录页面 ==========
        with gr.Column(visible=True) as login_page:
            # 引入 Google Fonts Inter
            gr.HTML("""
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
            """)
            
            with gr.Tabs() as auth_tabs:
                # 登录标签
                with gr.TabItem("用户登录") as login_tab:
                    with gr.Row(elem_classes=["login-container"]):
                        # 左侧美化面板
                        with gr.Column(scale=1, elem_classes=["left-panel"]):
                            gr.HTML("""
                            <div class="left-card">
                                <div class="left-content">
                                    <div class="logo-row">
                                        <div class="icon">🎨</div>
                                    </div>
                                    <h1 class="title">图像生成与语义一致性检测系统</h1>
                                    <p class="subtitle">
                                        基于 Stable Diffusion 的图像生成<br/>
                                        与双模型语义一致性检测
                                    </p>
                                    <ul class="feature-list">
                                        <li><span class="bullet">●</span> <strong>CLIP 模型</strong>：基础语义相似度检测</li>
                                        <li><span class="bullet">●</span> <strong>ITSC-GAN</strong>：增强语义一致性检测</li>
                                    </ul>
                                </div>
                            </div>
                            """)
                        
                        # 右侧登录表单
                        with gr.Column(scale=1, elem_classes=["right-panel"]):
                            with gr.Column(elem_classes=["right-card"]):
                                gr.HTML("""
                                <h2 class="form-title">用户登录</h2>
                                """)
                                
                                with gr.Column(elem_classes=["login-form"]):
                                    # 用户名或邮箱输入框
                                    gr.HTML("""
                                    <label class="field">
                                        <span class="field-label">用户名或邮箱</span>
                                    </label>
                                    """)
                                    login_username = gr.Textbox(
                                        label="",
                                        placeholder="请输入用户名或邮箱",
                                        value="admin",
                                        elem_classes=["input"],
                                        container=False
                                    )
                                    
                                    # 密码输入框
                                    gr.HTML("""
                                    <label class="field">
                                        <span class="field-label">密码</span>
                                    </label>
                                    """)
                                    login_password = gr.Textbox(
                                        label="",
                                        placeholder="请输入密码",
                                        type="password",
                                        value="admin123",
                                        elem_classes=["input"],
                                        container=False
                                    )
                                    
                                # 按钮区域
                                gr.HTML("""
                                <div class="btn-row">
                                """)
                                login_btn = gr.Button("登录", variant="primary", size="lg", elem_classes=["btn", "primary"], scale=1)
                                register_switch_btn = gr.Button("注册", variant="secondary", size="lg", elem_classes=["btn", "ghost"], scale=1)
                                gr.HTML("""
                                </div>
                                """)
                                
                                login_msg = gr.Markdown()
                                gr.HTML("""
                                <div id="messageBox" class="text-center text-sm hidden mt-6 p-3 rounded-xl" role="alert"></div>
                                """)
                
                # 注册标签
                with gr.TabItem("用户注册") as register_tab:
                    with gr.Row(elem_classes=["login-container"]):
                        # 左侧美化面板
                        with gr.Column(scale=1, elem_classes=["left-panel"]):
                            gr.HTML("""
                            <div class="left-panel-content">
                                <span class="left-panel-icon">🎨</span>
                                <h1>图像生成与语义一致性检测系统</h1>
                                <p class="subtitle">基于 Stable Diffusion 的图像生成</p>
                                <p class="subtitle">与双模型语义一致性检测</p>
                                <div class="features">
                                    <p>📌 <strong>CLIP模型：</strong>基础语义相似度检测</p>
                                    <p>📌 <strong>ITSC-GAN融合模型：</strong>增强语义一致性检测</p>
                                </div>
                            </div>
                            """)
                        
                        # 右侧注册表单
                        with gr.Column(scale=1, elem_classes=["right-panel"]):
                            with gr.Column(elem_classes=["login-card"]):
                                gr.HTML("""
                                <div style="text-align: center; margin-bottom: 2.5rem;">
                                    <h1 style="font-size: 2.5rem; font-weight: 800; color: #1f2937; margin-bottom: 0.75rem;">
                                        用户注册
                                    </h1>
                                    <p style="color: #6b7280; font-size: 1rem; font-weight: 400;">创建新账户以开始使用</p>
                                </div>
                                """)
                                
                                with gr.Column(scale=1, min_width=400):
                                    gr.HTML('<label class="block text-sm font-medium mb-2" style="color: #374151; font-weight: 600; font-size: 0.95rem;">👤 用户名</label>')
                                    register_username = gr.Textbox(
                                        label="",
                                        placeholder="请输入用户名",
                                        elem_classes=["custom-input"]
                                    )
                                    
                                    gr.HTML('<label class="block text-sm font-medium mb-2" style="color: #374151; font-weight: 600; font-size: 0.95rem;">📧 电子邮箱</label>')
                                    register_email = gr.Textbox(
                                        label="",
                                        placeholder="yourname@example.com",
                                        elem_classes=["custom-input"]
                                    )
                                    
                                    gr.HTML('<label class="block text-sm font-medium mb-2" style="color: #374151; font-weight: 600; font-size: 0.95rem;">🔒 密码</label>')
                                    register_password = gr.Textbox(
                                        label="",
                                        placeholder="请输入密码（至少6字符）",
                                        type="password",
                                        elem_classes=["custom-input"]
                                    )
                                    
                                    gr.HTML('<label class="block text-sm font-medium mb-2" style="color: #374151; font-weight: 600; font-size: 0.95rem;">🔒 确认密码</label>')
                                    register_confirm_password = gr.Textbox(
                                        label="",
                                        placeholder="再次确认密码",
                                        type="password",
                                        elem_classes=["custom-input"]
                                    )
                                
                                # 按钮区域 - 两个按钮并排
                                with gr.Row():
                                    register_btn = gr.Button("注册", variant="primary", size="lg", elem_classes=["main-btn"], scale=1)
                                    login_switch_btn = gr.Button("登录", variant="secondary", size="lg", scale=1)
                                
                                register_msg = gr.Markdown()
                                gr.HTML("""
                                <div id="registerMessageBox" class="hidden mt-6 p-3 rounded-xl" role="alert"></div>
                                """)
            
            # 添加全局样式和脚本
            global_styles = gr.HTML("""
            <style>
                /* 修复滚动问题：确保body和html可以滚动 */
                html, body {
                    height: auto !important;
                    min-height: 100vh !important;
                    overflow-x: hidden !important;
                    overflow-y: auto !important;
                }
                
                /* 确保Gradio主容器可以滚动 */
                #root, .gradio-container {
                    min-height: 100vh !important;
                    height: auto !important;
                    overflow-y: auto !important;
                    overflow-x: hidden !important;
                }
                
                /* 修复固定高度导致的滚动问题 */
                .gradio-container > div {
                    min-height: auto !important;
                    height: auto !important;
                    overflow-y: visible !important;
                }
                
                /* 确保内容区域可以滚动 */
                .gradio-tabs, .gradio-tab {
                    min-height: auto !important;
                    height: auto !important;
                    overflow-y: visible !important;
                    overflow-x: hidden !important;
                }
                
                /* 修复固定高度的卡片容器 */
                .gradio-row, .gradio-column {
                    min-height: auto !important;
                    height: auto !important;
                    overflow-y: visible !important;
                }
                
                /* 确保HTML组件内容可以滚动 */
                .gradio-html {
                    overflow-y: visible !important;
                    overflow-x: hidden !important;
                    max-height: none !important;
                }
                
                /* 修复历史记录和对比结果的滚动 */
                .gradio-html > div {
                    overflow-y: visible !important;
                    max-height: none !important;
                }
                
                .gradio-tabs {
                    background: transparent !important;
                    border: none !important;
                }
                
                .gradio-tab-nav {
                    background: transparent !important;
                    border: none !important;
                }
                
                .gradio-tab-nav button {
                    background: transparent !important;
                    color: #9ca3af !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 8px 16px !important;
                    transition: all 0.2s ease !important;
                }
                
                .gradio-tab-nav button:hover {
                    background: rgba(255, 255, 255, 0.05) !important;
                    color: #d1d5db !important;
                }
                
                .gradio-tab-nav button.selected {
                    background: rgba(85, 192, 232, 0.1) !important;
                    color: #55c0e8 !important;
                }
                
                ::-webkit-scrollbar {
                    width: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #111111;
                    border-radius: 10px;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(90deg, #55c0e8, #3b82f6);
                    border-radius: 10px;
                }
                
                .gradio-textbox input {
                    padding-left: 2.5rem !important;
                }
                
                /* 响应式：确保小屏幕也能滚动 */
                @media (max-width: 768px) {
                    html, body {
                        overflow-y: auto !important;
                    }
                    
                    .gradio-container {
                        overflow-y: auto !important;
                    }
                }
            </style>
            
            <script>
                function showRegister() {
                    const registerTab = document.querySelector('[data-testid="tab-用户注册"]');
                    if (registerTab) registerTab.click();
                    setTimeout(() => {
                        if (typeof lucide !== 'undefined') lucide.createIcons();
                    }, 100);
                }
                
                function showLogin() {
                    const loginTab = document.querySelector('[data-testid="tab-用户登录"]');
                    if (loginTab) loginTab.click();
                    setTimeout(() => {
                        if (typeof lucide !== 'undefined') lucide.createIcons();
                    }, 100);
                }
                
                // 初始化图标和导航按钮data-icon属性
                function initIcons() {
                    if (typeof lucide !== 'undefined') {
                        lucide.createIcons();
                    }
                    
                    // 设置导航按钮的图标
                    setTimeout(() => {
                        document.getElementById('dashboard-btn')?.setAttribute('data-icon', '📊');
                        document.getElementById('generate-btn')?.setAttribute('data-icon', '✨');
                        document.getElementById('detect-btn')?.setAttribute('data-icon', '🔍');
                        document.getElementById('compare-btn')?.setAttribute('data-icon', '🔄');
                        document.getElementById('summary-btn')?.setAttribute('data-icon', '📈');
                        document.getElementById('history-btn')?.setAttribute('data-icon', '📋');
                        document.getElementById('logout-btn')?.setAttribute('data-icon', '🚪');
                    }, 100);
                }
                
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', function() {
                        initIcons();
                    });
                } else {
                    initIcons();
                }
                
                // 监听Gradio组件加载完成
                document.addEventListener('DOMContentLoaded', function() {
                    const observer = new MutationObserver(function(mutations) {
                        initIcons();
                    });
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                });
                
                // 修复滚动问题：确保body可以滚动
                function fixScrolling() {
                    // 确保body和html可以滚动
                    document.body.style.overflow = 'auto';
                    document.body.style.overflowX = 'hidden';
                    document.documentElement.style.overflow = 'auto';
                    document.documentElement.style.overflowX = 'hidden';
                    
                    // 移除可能阻止滚动的样式
                    const gradioContainer = document.querySelector('.gradio-container');
                    if (gradioContainer) {
                        gradioContainer.style.overflowY = 'auto';
                        gradioContainer.style.overflowX = 'hidden';
                        gradioContainer.style.height = 'auto';
                        gradioContainer.style.minHeight = '100vh';
                    }
                    
                    // 修复所有可能阻止滚动的容器
                    const fixedHeightElements = document.querySelectorAll('[style*="overflow: hidden"], [style*="overflow:hidden"]');
                    fixedHeightElements.forEach(el => {
                        const style = el.getAttribute('style') || '';
                        if (style.includes('height: 100vh') && style.includes('overflow')) {
                            el.style.overflowY = 'auto';
                            el.style.height = 'auto';
                            el.style.minHeight = '100vh';
                        }
                    });
                }
                
                // 页面加载时执行
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', fixScrolling);
                } else {
                    fixScrolling();
                }
                
                // 定期检查并修复（Gradio可能会动态修改DOM）
                setInterval(fixScrolling, 1000);
            </script>
            """)
        
        # ========== 主功能页面 ==========
        with gr.Column(visible=False) as main_page:
            # 页面标题栏
            main_header = gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px 30px;
                border-radius: 16px;
                margin-bottom: 20px;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                color: white;
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1400px;
                margin-left: auto;
                margin-right: auto;
            ">
                <div>
                    <h1 style="margin: 0; font-size: 1.6rem; font-weight: 800;">
                        🎨 图像生成与语义一致性检测系统
                    </h1>
                    <p style="margin: 5px 0 0 0; font-size: 0.95rem; opacity: 0.95;">
                        基于Stable Diffusion的图像生成与双模型语义一致性检测
                    </p>
                </div>
            </div>
            """)
            
            # 主内容区域 - 左侧导航栏 + 右侧内容
            with gr.Row(elem_classes=["main-content-container"]):
                # 左侧导航栏 - 添加渐变背景和阴影
                with gr.Column(scale=1, min_width=230, max_width=230, elem_classes=["sidebar"]):
                    # 导航按钮 - 垂直排列，添加data-icon属性
                    dashboard_nav_btn = gr.Button("仪表盘", variant="secondary", size="lg", elem_classes=["nav-button", "active"], elem_id="dashboard-btn")
                    
                    generate_nav_btn = gr.Button("图像生成", variant="secondary", size="lg", elem_classes=["nav-button"], elem_id="generate-btn")
                    
                    detect_nav_btn = gr.Button("一致性检测", variant="secondary", size="lg", elem_classes=["nav-button"], elem_id="detect-btn")
                    
                    compare_nav_btn = gr.Button("模型对比", variant="secondary", size="lg", elem_classes=["nav-button"], elem_id="compare-btn")
                    
                    summary_nav_btn = gr.Button("总结分析", variant="secondary", size="lg", elem_classes=["nav-button"], elem_id="summary-btn")
                    
                    history_nav_btn = gr.Button("历史记录", variant="secondary", size="lg", elem_classes=["nav-button"], elem_id="history-btn")
                    
                    # 分隔线
                    gr.HTML("<div style='height: 1px; background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent); margin: 16px 0;'></div>")
                    
                    # 登出按钮 - 优化样式
                    logout_btn = gr.Button("登出", variant="secondary", elem_classes=["logout-button"], elem_id="logout-btn")
                
                # 右侧内容区域 - 添加卡片式布局
                with gr.Column(scale=5, elem_classes=["content-area"]):
                    # 仪表盘内容
                    with gr.Column(visible=True) as dashboard_content:
                        dashboard_display = gr.HTML(label="仪表盘")
                        refresh_dashboard_btn = gr.Button("🔄 刷新仪表盘", variant="secondary")
                    
                    # 生成图像内容
                    with gr.Column(visible=False) as generate_content:
                        prompt_input = gr.Textbox(
                            label="📝 文本提示词",
                            placeholder="请输入图像描述...",
                            lines=3
                        )
                        
                        # 快速提示词按钮
                        gr.Markdown("### 📋 快速提示词示例")
                        with gr.Row():
                            quick_btn1 = gr.Button("一只可爱的小猫坐在窗台上", size="sm")
                            quick_btn2 = gr.Button("一个宇航员在月球上行走", size="sm")
                            quick_btn3 = gr.Button("海滩日落景色", size="sm")
                            quick_btn4 = gr.Button("未来城市夜景", size="sm")
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="🤖 选择生成模型",
                                choices=[model[1] for model in AVAILABLE_MODELS],
                                value=[model[1] for model in AVAILABLE_MODELS if model[0] == DEFAULT_MODEL][0]
                            )
                            
                            # 添加ITSC-GAN模型内存使用提示
                            gr.Markdown("""<div style="background-color: #f0f9ff; padding: 10px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                                💡 **提示**：选择"ITSC-GAN融合模型"时，建议：
                                <br>• 关闭其他占用大量内存的程序
                                <br>• 图像尺寸使用默认的512x512
                                <br>• 推理步数设置为30-50之间
                                <br>• 如果遇到内存不足错误，请尝试增加Windows虚拟内存
                            </div>""")
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
                        
                        generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
                        # 先放置“添加到对比”按钮，再显示图像，确保按钮始终出现在图像上方
                        add_to_compare_btn = gr.Button(
                            "➕ 添加到对比",
                            variant="secondary",
                            size="lg",
                            visible=False,
                            elem_id="add-to-compare-btn"
                        )
                        output_image = gr.HTML(label="🎨 生成的图像", elem_id="generated-image-preview")
                        generate_msg = gr.Markdown()
                        # 存储当前生成的图像信息
                        current_generated_image = gr.State(value=None)
                        current_generated_prompt = gr.State(value=None)
                    
                    # 一致性检测内容
                    with gr.Column(visible=False) as detect_content:
                        gr.Markdown("### 🔍 图像语义一致性检测")
                        detect_image_input = gr.File(
                            label="上传图像",
                            file_types=["image"]
                        )
                        detect_prompt_input = gr.Textbox(
                            label="📝 文本提示词",
                            placeholder="请输入图像对应的文本描述...",
                            lines=3
                        )
                        detect_model_dropdown = gr.Dropdown(
                            label="🤖 选择模型（用于检测）",
                            choices=[model[1] for model in AVAILABLE_MODELS],
                            value=[model[1] for model in AVAILABLE_MODELS if model[0] == DEFAULT_MODEL][0]
                        )
                        detect_threshold = gr.Slider(
                            label="一致性阈值",
                            minimum=0.0,
                            maximum=1.0,
                            value=DEFAULT_THRESHOLD,
                            step=0.05
                        )
                        detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
                        detect_result = gr.HTML()
                    
                    # 多模型对比内容
                    with gr.Column(visible=False) as compare_content:
                        gr.Markdown("### 🔄 模型对比功能分区")
                        compare_images_display = gr.HTML(
                            value="<div style='padding: 15px; background: #fff3cd; border-radius: 10px; margin-bottom: 15px;'><p style='margin: 0; color: #856404;'>💡 提示：在'图像生成'功能区生成图像后，点击'添加到对比'按钮将图像添加到此列表。至少需要2张图像才能开始对比。</p></div>",
                            label="📋 待对比图像列表"
                        )
                        
                        compare_prompt = gr.Textbox(
                            label="📝 文本提示词（用于新生成对比图像）",
                            placeholder="请输入图像描述...",
                            lines=3
                        )
                        
                        with gr.Row():
                            compare_steps = gr.Slider(
                                label="推理步数",
                                minimum=10,
                                maximum=100,
                                value=DEFAULT_NUM_STEPS,
                                step=1
                            )
                            compare_guidance = gr.Slider(
                                label="引导强度",
                                minimum=1.0,
                                maximum=20.0,
                                value=DEFAULT_GUIDANCE_SCALE,
                                step=0.1
                            )
                            compare_threshold = gr.Slider(
                                label="一致性阈值",
                                minimum=0.0,
                                maximum=1.0,
                                value=DEFAULT_THRESHOLD,
                                step=0.05
                            )
                        
                        with gr.Row():
                            compare_height = gr.Number(
                                label="图像高度",
                                value=DEFAULT_HEIGHT,
                                precision=0,
                                minimum=512,
                                maximum=1024,
                                step=64
                            )
                            compare_width = gr.Number(
                                label="图像宽度",
                                value=DEFAULT_WIDTH,
                                precision=0,
                                minimum=512,
                                maximum=1024,
                                step=64
                            )
                        
                        compare_btn = gr.Button("🚀 开始模型对比", variant="primary", size="lg", interactive=False)
                        compare_result = gr.HTML()
                        compare_summary = gr.HTML()
                    
                    # 总结分析内容
                    with gr.Column(visible=False) as summary_content:
                        gr.Markdown("### 📈 模型对比总结分析")
                        summary_prompt = gr.Textbox(
                            label="📝 文本提示词（用于生成对比数据）",
                            placeholder="请输入图像描述...",
                            lines=2
                        )
                        generate_summary_btn = gr.Button("📊 生成总结报告", variant="primary", size="lg")
                        summary_display = gr.HTML()
                    
                    # 历史记录内容
                    with gr.Column(visible=False) as history_content:
                        history_display = gr.HTML(label="历史记录列表")
                        refresh_history_btn = gr.Button("🔄 刷新历史记录", variant="secondary")
        
        # ========== 事件绑定 ==========
        # 注册
        def handle_register(username, email, password, confirm_password):
            try:
                msg, success = register_user(username, email, password, confirm_password)
                # 格式化消息显示
                if success:
                    formatted_msg = f"<div style='padding: 15px; background: #4CAF50; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3></div>"
                    return formatted_msg, gr.update(visible=True), gr.update(visible=False)  # 显示登录标签
                else:
                    formatted_msg = f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3></div>"
                    return formatted_msg, gr.update(visible=False), gr.update(visible=True)  # 保持注册标签
            except Exception as e:
                error_msg = f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 注册失败: {str(e)}</h3></div>"
                return error_msg, gr.update(visible=False), gr.update(visible=True)
        
        # 注册按钮点击事件
        def register_click_handler(username, email, password, confirm_password):
            """注册按钮点击处理函数，添加调试信息"""
            print(f"[DEBUG] 注册按钮被点击，参数: username={username}, email={email}")
            try:
                result = handle_register(username, email, password, confirm_password)
                print(f"[DEBUG] 注册处理结果: {result}")
                return result
            except Exception as e:
                print(f"[ERROR] 注册处理异常: {e}")
                import traceback
                traceback.print_exc()
                error_msg = f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 注册失败: {str(e)}</h3></div>"
                return error_msg, gr.update(visible=False), gr.update(visible=True)
        
        register_btn.click(
            fn=register_click_handler,
            inputs=[register_username, register_email, register_password, register_confirm_password],
            outputs=[register_msg, login_tab, register_tab]
        )
        
        # 登录页面的注册按钮 - 切换到注册标签
        def switch_to_register():
            return gr.update(visible=False), gr.update(visible=True)
        
        register_switch_btn.click(
            fn=switch_to_register,
            inputs=[],
            outputs=[login_tab, register_tab]
        )
        
        # 注册页面的登录按钮 - 切换到登录标签
        def switch_to_login():
            return gr.update(visible=True), gr.update(visible=False)
        
        login_switch_btn.click(
            fn=switch_to_login,
            inputs=[],
            outputs=[login_tab, register_tab]
        )
        
        # 添加JavaScript切换函数
        switch_script = gr.HTML("""
        <script>
        function showRegister() {
            const loginTab = document.querySelector('[data-testid="tab-用户登录"]');
            const registerTab = document.querySelector('[data-testid="tab-用户注册"]');
            if (loginTab && registerTab) {
                loginTab.style.display = 'none';
                registerTab.style.display = 'block';
                registerTab.click();
            }
        }
        
        function showLogin() {
            const loginTab = document.querySelector('[data-testid="tab-用户登录"]');
            const registerTab = document.querySelector('[data-testid="tab-用户注册"]');
            if (loginTab && registerTab) {
                registerTab.style.display = 'none';
                loginTab.style.display = 'block';
                loginTab.click();
            }
        }
        </script>
        """)
        
        # 登录
        def handle_login(username, password):
            """处理用户登录（优化：极速响应，完全异步）"""
            import time
            import sys
            start_time = time.time()
            
            # 强制刷新输出，确保日志立即显示
            sys.stdout.flush()
            sys.stderr.flush()
            
            try:
                print(f"[DEBUG] handle_login: 开始处理登录请求")
                print(f"[DEBUG] handle_login: username={username}, password_length={len(password) if password else 0}")
                sys.stdout.flush()
                
                # 用户登录验证（只做数据库查询，不依赖任何服务）
                user_id, msg, success = login_user(username, password)
                
                elapsed = time.time() - start_time
                print(f"[DEBUG] handle_login: login_user 返回，耗时: {elapsed:.3f}秒, success={success}, user_id={user_id}")
                sys.stdout.flush()
                
                if success:
                    # 登录成功，立即返回（不等待任何服务）
                    dashboard_data = "<div style='padding: 15px; background: #e3f2fd; border-radius: 10px; color: #1976d2; text-align: center;'><h3 style='margin: 0;'>✅ 登录成功！点击上方'仪表盘'按钮查看数据</h3></div>"
                    
                    result = (
                        user_id,
                        True,
                        f"<div style='padding: 15px; background: #4CAF50; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3></div>",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        dashboard_data
                    )
                    total_elapsed = time.time() - start_time
                    print(f"[DEBUG] handle_login: 登录成功，准备返回结果，总耗时: {total_elapsed:.3f}秒")
                    sys.stdout.flush()
                    return result
                else:
                    result = (
                        None,
                        False,
                        f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>{msg}</h3></div>",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        ""
                    )
                    total_elapsed = time.time() - start_time
                    print(f"[DEBUG] handle_login: 登录失败，准备返回结果，总耗时: {total_elapsed:.3f}秒")
                    sys.stdout.flush()
                    return result
            except Exception as e:
                error_msg = f"登录过程出错: {str(e)}"
                print(f"[ERROR] handle_login: {error_msg}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                total_elapsed = time.time() - start_time
                print(f"[ERROR] handle_login: 登录异常，总耗时: {total_elapsed:.3f}秒")
                sys.stdout.flush()
                return (
                    None,
                    False,
                    f"<div style='padding: 15px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ {error_msg}</h3><p style='margin-top: 10px; font-size: 12px;'>请检查控制台日志获取详细信息</p></div>",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    ""
                )
        
        # 登录按钮点击事件（优化：添加队列和超时保护）
        login_btn.click(
            fn=handle_login,
            inputs=[login_username, login_password],
            outputs=[
                current_user_id,
                login_status,
                login_msg,
                login_page,
                main_page,
                dashboard_display
            ],
            queue=False  # 不使用队列，立即处理
        )
        
        # 登录成功后加载仪表盘数据 (通过登录按钮直接调用)
        
        # 生成图像
        def handle_generate(prompt, model_display, steps, guidance, height, width, user_id, progress=gr.Progress()):
            """处理图像生成（带进度条）"""
            try:
                # 初始化进度条
                progress(0, desc="开始图像生成...")
                
                # 转换模型显示名称为模型ID
                model_id = get_model_id_from_display(model_display)
                print(f"[DEBUG] handle_generate: 显示名称='{model_display}' -> 模型ID='{model_id}'")
                
                # 进度回调函数
                def progress_callback(progress_value, status="", step=""):
                    desc = f"图像生成中: {status}"
                    if step:
                        desc += f" - {step}"
                    progress(progress_value / 100, desc=desc)
                
                # 生成图像（带进度回调）
                progress(0.1, desc="加载模型中...")
                img_html, image_path, prompt_val, model_id, threshold_val = generate_image(
                    prompt, model_display, DEFAULT_THRESHOLD, steps, guidance, height, width, user_id,
                    progress_callback=progress_callback
                )
                
                progress(1.0, desc="图像生成完成！")
                
                if image_path:
                    scroll_script = """
                    <script>
                        const compareBtn = document.getElementById('add-to-compare-btn');
                        if (compareBtn) {
                            compareBtn.scrollIntoView({behavior: 'smooth', block: 'center'});
                            compareBtn.classList.add('pulse');
                            setTimeout(() => compareBtn.classList.remove('pulse'), 2000);
                        }
                    </script>
                    """
                    return (
                        img_html,
                        gr.update(visible=True),  # 显示"添加到对比"按钮
                        image_path,  # 保存图像路径
                        prompt_val,  # 保存提示词（使用generate_image返回的实际提示词）
                        f"<div style='padding: 15px; background: #4CAF50; border-radius: 10px; color: white;'><h3 style='margin: 0;'>✅ 图像生成成功！</h3><p>图像已保存至: {image_path}</p><p style='margin-top: 10px;'>系统已自动定位到“添加到对比”按钮。</p></div>{scroll_script}"
                    )
                return (
                    img_html,
                    gr.update(visible=False),
                    None,
                    None,
                    f"<div style='padding: 15px; background: #ff9800; border-radius: 10px; color: white;'><h3 style='margin: 0;'>⚠️ 生成完成，但未保存</h3></div>"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 生成失败: {str(e)}</h3></div>"
                return error_html, gr.update(visible=False), None, None, error_html
        
        generate_btn.click(
            fn=handle_generate,
            inputs=[prompt_input, model_dropdown, num_steps, guidance_scale, height, width, current_user_id],
            outputs=[output_image, add_to_compare_btn, current_generated_image, current_generated_prompt, generate_msg],
            show_progress=True
        )
        
        # 添加到对比功能
        def add_to_comparison(image_path, prompt, images_list):
            """将图像添加到对比列表"""
            if not image_path or not prompt:
                return images_list, gr.update(interactive=False), "<div style='padding: 15px; background: #fff3cd; border-radius: 10px; margin-bottom: 15px;'><p style='margin: 0; color: #856404;'>⚠️ 请先生成图像</p></div>"
            
            # 检查是否已存在
            for img_info in images_list:
                if img_info.get('image_path') == image_path:
                    return images_list, gr.update(interactive=len(images_list) >= 2), "<div style='padding: 15px; background: #fff3cd; border-radius: 10px; margin-bottom: 15px;'><p style='margin: 0; color: #856404;'>⚠️ 该图像已在对比列表中</p></div>"
            
            # 创建图像信息字典
            image_info = {
                "image_path": image_path,
                "prompt": prompt,
                "id": len(images_list) + 1
            }
            
            # 添加到列表
            new_list = images_list + [image_info]
            
            # 生成对比列表HTML
            list_html = "<div style='padding: 15px; background: #e8f5e9; border-radius: 10px; margin-bottom: 15px; border: 2px solid #4CAF50;'>"
            list_html += f"<h4 style='margin: 0 0 10px 0; color: #2e7d32;'>✅ 已添加 {len(new_list)} 张图像到对比列表：</h4>"
            for idx, img_info in enumerate(new_list, 1):
                list_html += f"""
                <div style='padding: 10px; background: white; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #667eea;'>
                    <p style='margin: 5px 0;'><strong>图像 {idx}:</strong> {img_info['prompt'][:50]}{'...' if len(img_info['prompt']) > 50 else ''}</p>
                    <p style='margin: 5px 0; color: #666; font-size: 12px;'>{img_info['image_path']}</p>
                </div>
                """
            if len(new_list) >= 2:
                list_html += "<p style='margin: 10px 0 0 0; color: #2e7d32; font-weight: bold;'>✓ 已满足对比条件，可以开始对比了！</p>"
            else:
                list_html += f"<p style='margin: 10px 0 0 0; color: #856404;'>还需要添加 {2 - len(new_list)} 张图像才能开始对比</p>"
            list_html += "</div>"
            
            # 如果至少有2张图像，启用对比按钮
            button_interactive = len(new_list) >= 2
            
            return new_list, gr.update(interactive=button_interactive), list_html
        
        add_to_compare_btn.click(
            fn=add_to_comparison,
            inputs=[current_generated_image, current_generated_prompt, comparison_images_list],
            outputs=[comparison_images_list, compare_btn, compare_images_display]
        )
        
        # 一致性检测（专门的功能区）
        def handle_detect_standalone(image_file, prompt, model_display, threshold, user_id):
            """专门的一致性检测功能"""
            try:
                if not image_file:
                    return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请上传图像</h3></div>"
                
                # 转换模型显示名称为模型ID
                model_id = get_model_id_from_display(model_display)
                
                # 读取上传的图像
                if isinstance(image_file, str):
                    image_path = image_file
                else:
                    image_path = image_file.name
                
                result = detect_consistency(image_path, prompt, threshold, model_id, user_id)
                return result
            except Exception as e:
                return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 检测失败: {str(e)}</h3></div>"
        
        detect_btn.click(
            fn=handle_detect_standalone,
            inputs=[detect_image_input, detect_prompt_input, detect_model_dropdown, detect_threshold, current_user_id],
            outputs=[detect_result]
        )
        
        # 模型对比
        def handle_compare(images_list, prompt, threshold, steps, guidance, height, width, user_id, progress=gr.Progress()):
            """处理模型对比（异步版本，带进度条）"""
            try:
                # 检查服务是否已初始化
                if comparison_service is None:
                    error_html = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 模型对比服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
                    return error_html, ""
                
                if not images_list or len(images_list) < 2:
                    error_html = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请至少添加2张图像到对比列表</h3></div>"
                    return error_html, ""
                
                # 如果有提示词，使用提示词生成新图像进行对比（异步版本）
                if prompt and prompt.strip():
                    # 使用异步对比方法，自动显示进度条
                    return compare_models_async_web(
                        prompt, threshold, steps, guidance, height, width, user_id, progress
                    )
                else:
                    # 使用已添加的图像列表进行对比
                    result_html = "<div style='padding: 20px; background: #4CAF50; border-radius: 10px; color: white;'><h3 style='margin: 0;'>✅ 开始对比已添加的图像</h3></div>"
                    result_html += "<div style='padding: 20px; background: white; border-radius: 10px; margin-top: 15px;'>"
                    for img_info in images_list:
                        result_html += f"<div style='padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 10px;'>"
                        result_html += f"<p><strong>图像 {img_info['id']}:</strong> {img_info['prompt']}</p>"
                        result_html += f"<p style='color: #666; font-size: 12px;'>{img_info['image_path']}</p>"
                        result_html += "</div>"
                    result_html += "</div>"
                    summary_html = "<div style='padding: 20px; background: #e3f2fd; border-radius: 10px;'><p>对比功能开发中，将对比已添加的图像...</p></div>"
                    return result_html, summary_html
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 对比失败: {str(e)}</h3></div>"
                return error_html, error_html
        
        compare_btn.click(
            fn=handle_compare,
            inputs=[comparison_images_list, compare_prompt, compare_threshold, compare_steps, compare_guidance, compare_height, compare_width, current_user_id],
            outputs=[compare_result, compare_summary],
            show_progress=True
        )
        
        # 取消对比任务按钮
        def cancel_comparison_task():
            """取消所有正在进行的对比任务"""
            for task_id in list(global_comparison_tasks.keys()):
                comparison_service.cancel_task(task_id)
                del global_comparison_tasks[task_id]
            return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 所有对比任务已取消</h3></div>"
        
        # 创建取消按钮（默认隐藏，可在需要时显示）
        cancel_btn = gr.Button("取消对比", visible=False)
        cancel_btn.click(
            fn=cancel_comparison_task,
            outputs=[compare_result]
        )
        
        # 总结分析
        def handle_summary(prompt, user_id, progress=gr.Progress()):
            """生成总结报告（带进度条）"""
            try:
                if not user_id:
                    return "<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
                
                # 检查服务是否已初始化
                if comparison_service is None or summary_service is None:
                    return "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 服务正在初始化中，请稍候...</h3><p style='margin-top: 10px; font-size: 14px;'>服务将在几秒内完成初始化，请稍后重试</p></div>"
                
                progress(0, desc="准备生成总结报告...")
                
                # 进度回调
                def progress_callback(progress_value, status="", model_name=""):
                    desc = f"{status}"
                    if model_name:
                        desc += f" - {model_name}"
                    progress(progress_value / 100, desc=desc)
                
                # 执行对比以生成数据（带进度回调）
                model_names = [m[0] for m in AVAILABLE_MODELS]
                progress(0.1, desc="开始模型对比...")
                comparison_results = comparison_service.compare_models(
                    prompt=prompt,
                    model_names=model_names,
                    threshold=DEFAULT_THRESHOLD,
                    num_inference_steps=DEFAULT_NUM_STEPS,
                    guidance_scale=DEFAULT_GUIDANCE_SCALE,
                    height=DEFAULT_HEIGHT,
                    width=DEFAULT_WIDTH,
                    progress_callback=progress_callback
                )
                
                # 生成总结
                progress(0.9, desc="正在生成总结报告...")
                summary_result = summary_service.generate_summary(comparison_results, include_charts=True)
                progress(1.0, desc="总结报告生成完成！")
                return summary_result.get('summary_html', '')
                
            except Exception as e:
                return f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 生成总结失败: {str(e)}</h3></div>"
        
        generate_summary_btn.click(
            fn=handle_summary,
            inputs=[summary_prompt, current_user_id],
            outputs=[summary_display],
            show_progress=True
        )
        
        # 导航切换函数
        def show_dashboard():
            """显示仪表盘（优化：快速响应）"""
            try:
                # 快速返回，避免阻塞
                if not current_user_id.value:
                    dashboard_html = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 请先登录</h3></div>"
                else:
                    # 获取仪表盘数据（如果dashboard_service可用）
                    if dashboard_service:
                        dashboard_html = get_dashboard_data(current_user_id.value)
                    else:
                        dashboard_html = "<div style='padding: 20px; background: #ff9800; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>⚠️ 仪表盘服务未初始化</h3></div>"
            except Exception as e:
                print(f"[ERROR] 获取仪表盘数据失败: {e}")
                dashboard_html = f"<div style='padding: 20px; background: #f44336; border-radius: 10px; color: white; text-align: center;'><h3 style='margin: 0;'>❌ 加载失败: {str(e)}</h3></div>"
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(elem_classes=["nav-button", "active"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                dashboard_html
            )
        
        def show_generate():
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button", "active"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                ""
            )
        
        def show_detect():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button", "active"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                ""
            )
        
        def show_compare():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button", "active"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                ""
            )
        
        def show_summary():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button", "active"]),
                gr.update(elem_classes=["nav-button"]),
                ""
            )
        
        def show_history():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button"]),
                gr.update(elem_classes=["nav-button", "active"]),
                ""
            )
        
        # 导航按钮点击事件
        dashboard_nav_btn.click(
            fn=show_dashboard,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        generate_nav_btn.click(
            fn=show_generate,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        detect_nav_btn.click(
            fn=show_detect,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        compare_nav_btn.click(
            fn=show_compare,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        summary_nav_btn.click(
            fn=show_summary,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        history_nav_btn.click(
            fn=show_history,
            inputs=[],
            outputs=[
                dashboard_content, generate_content, detect_content,
                compare_content, summary_content, history_content,
                dashboard_nav_btn, generate_nav_btn, detect_nav_btn,
                compare_nav_btn, summary_nav_btn, history_nav_btn,
                generate_msg
            ]
        )
        
        # 仪表盘刷新
        refresh_dashboard_btn.click(
            fn=get_dashboard_data,
            inputs=[current_user_id],
            outputs=[dashboard_display]
        )
        
        # 历史记录刷新
        refresh_history_btn.click(
            fn=get_history,
            inputs=[current_user_id],
            outputs=[history_display]
        )
        
        # 快速提示词
        quick_btn1.click(fn=lambda: "一只可爱的小猫坐在窗台上，阳光透过窗户洒在它身上", outputs=prompt_input)
        quick_btn2.click(fn=lambda: "一个宇航员在月球上行走，地球在背景中", outputs=prompt_input)
        quick_btn3.click(fn=lambda: "海滩日落景色，金色阳光洒在海面上，远处有帆船", outputs=prompt_input)
        quick_btn4.click(fn=lambda: "未来城市夜景，霓虹灯闪烁，飞行汽车，摩天大楼，赛博朋克风格", outputs=prompt_input)
        
        # 登出
        def handle_logout():
            return (
                None,
                False,
                gr.update(visible=True),
                gr.update(visible=False)
            )
        
        logout_btn.click(
            fn=handle_logout,
            inputs=[],
            outputs=[current_user_id, login_status, login_page, main_page]
        )
        
    
    demo.queue()
    return demo

# ==================== 端口管理工具 ====================
def kill_port(port):
    """
    杀死占用指定端口的进程（Windows）
    
    Args:
        port: 端口号
        
    Returns:
        是否成功杀死进程
    """
    import subprocess
    import platform
    
    if platform.system() != 'Windows':
        print(f"[WARNING] 端口杀死功能目前仅支持Windows系统")
        return False
    
    try:
        # 查找占用端口的进程
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            encoding='gbk'  # Windows中文系统使用GBK编码
        )
        
        # 查找占用指定端口的进程ID
        pid = None
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    break
        
        if pid:
            print(f"[INFO] 发现端口 {port} 被进程 {pid} 占用，正在终止...")
            # 杀死进程
            kill_result = subprocess.run(
                ['taskkill', '/F', '/PID', pid],
                capture_output=True,
                text=True,
                encoding='gbk'
            )
            
            if kill_result.returncode == 0:
                print(f"[INFO] ✅ 成功终止进程 {pid}")
                return True
            else:
                print(f"[WARNING] 终止进程失败: {kill_result.stderr}")
                return False
        else:
            print(f"[INFO] 端口 {port} 未被占用")
            return True
            
    except Exception as e:
        print(f"[WARNING] 杀死端口失败: {e}")
        return False

def find_available_port(start_port=8080, max_attempts=50):
    """查找可用端口（增加尝试次数）"""
    import socket
    
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))  # 绑定到所有接口，更准确
                return port
        except OSError:
            continue
    return None

def is_port_available(port):
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False

# ==================== 主函数 ====================
def main():
    """运行Web界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description='图像生成与语义一致性检测系统')
    parser.add_argument('--port', type=int, default=None, help='服务器端口 (默认: 自动选择)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0 允许外部访问)')
    parser.add_argument('--share', action='store_true', help='创建公共链接 (使用Gradio的share功能)')
    parser.add_argument('--local', action='store_true', help='仅本地访问 (使用127.0.0.1)')
    parser.add_argument('--auth', type=str, nargs=2, metavar=('USERNAME', 'PASSWORD'), 
                       help='启用HTTP基本认证 (格式: --auth username password)')
    parser.add_argument('--auth-message', type=str, default='请输入用户名和密码',
                       help='认证提示信息')
    args = parser.parse_args()
    
    # 如果指定了--local，则使用127.0.0.1
    if args.local:
        server_name = "127.0.0.1"
        print("⚠️  仅本地访问模式 (127.0.0.1)")
    else:
        # 默认使用0.0.0.0允许所有网络接口访问，包括localhost
        server_name = args.host if args.host != "0.0.0.0" else "0.0.0.0"
        print(f"🌐 服务器模式 (监听: {server_name})")
        if args.port:
            print(f"   本地访问: http://127.0.0.1:{args.port}")
            print(f"   局域网访问: http://<您的IP>:{args.port}")
        else:
            print(f"   本地访问: http://127.0.0.1:<自动分配端口>")
            print(f"   局域网访问: http://<您的IP>:<自动分配端口>")
            print(f"   注: 端口将在启动时自动分配")
    
    # 处理端口 - 改为自动选择可用端口
    if args.port:
        # 如果指定了端口，先检查是否可用
        if is_port_available(args.port):
            target_port = args.port
            print(f"✅ 使用指定端口: {target_port}")
        else:
            print(f"⚠️  端口 {args.port} 已被占用，尝试杀死占用进程...")
            # 尝试杀死占用端口的进程
            if kill_port(args.port):
                # 等待一下，让端口释放
                import time
                time.sleep(1)
                # 再次检查端口是否可用
                if is_port_available(args.port):
                    target_port = args.port
                    print(f"✅ 端口 {args.port} 已释放，使用该端口")
                else:
                    print(f"⚠️  端口 {args.port} 仍被占用，自动查找可用端口...")
                    target_port = find_available_port(start_port=args.port, max_attempts=50)
                    if target_port:
                        print(f"✅ 找到可用端口: {target_port}")
                    else:
                        print(f"❌ 无法找到可用端口（已尝试 {args.port}-{args.port+49}）")
                        print(f"💡 建议：关闭占用端口 {args.port} 的进程，或使用其他端口")
                        return
            else:
                print(f"⚠️  无法杀死占用进程，自动查找可用端口...")
                target_port = find_available_port(start_port=args.port, max_attempts=50)
                if target_port:
                    print(f"✅ 找到可用端口: {target_port}")
                else:
                    print(f"❌ 无法找到可用端口（已尝试 {args.port}-{args.port+49}）")
                    print(f"💡 建议：关闭占用端口 {args.port} 的进程，或使用其他端口")
                    return
    else:
        # 未指定端口，让Gradio自动选择（不设置server_port参数）
        target_port = None
        print(f"✅ 将自动选择可用端口")
    
    try:
        print("=" * 60)
        print("正在启动Gradio界面...")
        print("=" * 60)
        
        demo = create_interface()
        
        print("✅ Gradio界面创建成功！")
        print("=" * 60)
        print(f"正在启动服务器...")
        print(f"  地址: {server_name}")
        
        # 准备启动参数
        launch_kwargs = {
            'server_name': server_name,
            'share': args.share,
            'show_error': True,
            'inbrowser': True,  # 自动打开浏览器
            'favicon_path': None,
        }
        
        # 只有在指定了端口时才设置server_port，否则让Gradio自动选择
        if target_port is not None:
            launch_kwargs['server_port'] = target_port
            print(f"  端口: {target_port}")
        else:
            print(f"  端口: 自动选择")
        
        # 如果启用了认证
        if args.auth:
            launch_kwargs['auth'] = tuple(args.auth)
            launch_kwargs['auth_message'] = args.auth_message
        
        demo.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


