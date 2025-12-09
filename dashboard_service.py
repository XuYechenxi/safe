#!/usr/bin/env python3
"""
仪表盘服务模块
显示不同模型的生成图片和统计信息
"""

import os
from typing import Dict, List, Any
from database import Database
from PIL import Image
import base64
import io


class DashboardService:
    """仪表盘服务"""
    
    def __init__(self):
        """初始化服务"""
        self.db = Database()
    
    def get_dashboard_data(self, user_id: int = None) -> Dict[str, Any]:
        """
        获取仪表盘数据
        
        Args:
            user_id: 用户ID（可选，None表示所有用户）
        
        Returns:
            仪表盘数据字典
        """
        # 获取统计数据（如果user_id为None，返回空统计）
        try:
            if user_id is not None and hasattr(self.db, 'get_statistics'):
                stats = self.db.get_statistics(user_id)
            else:
                stats = {
                    'total_generations': 0,
                    'today_generations': 0,
                    'consistent_count': 0,
                    'inconsistent_count': 0,
                    'consistency_rate': 0,
                    'average_score': 0.0
                }
        except Exception as e:
            print(f"[WARNING] 获取统计数据失败: {e}")
            stats = {
                'total_generations': 0,
                'today_generations': 0,
                'consistent_count': 0,
                'inconsistent_count': 0,
                'consistency_rate': 0,
                'average_score': 0.0
            }
        
        # 获取各模型的生成记录
        model_images = self._get_model_images(user_id)
        
        # 获取最近生成的图片
        recent_images = self._get_recent_images(user_id, limit=12)
        
        return {
            'stats': stats,
            'model_images': model_images,
            'recent_images': recent_images
        }
    
    def _get_model_images(self, user_id: int = None) -> Dict[str, List[Dict]]:
        """获取各模型的生成图片"""
        # 从数据库获取历史记录
        try:
            if user_id:
                history = self.db.get_user_history(user_id)
            else:
                # 获取所有用户的历史记录（需要数据库支持）
                history = []
        except Exception as e:
            print(f"[WARNING] 获取历史记录失败: {e}")
            history = []
        
        model_images = {
            'sd-base': [],
            'clip-fusion': [],
            'itsc-gan': []
        }
        
        for record in history:
            # 尝试从result_data中解析model_name
            model_name = record.get('model_name', 'unknown')
            if model_name == 'unknown':
                # 尝试从result_data中解析
                try:
                    import json
                    result_data_str = record.get('result_data', '')
                    if result_data_str:
                        result_data = json.loads(result_data_str) if isinstance(result_data_str, str) else result_data_str
                        model_name = result_data.get('model_name', 'unknown')
                except Exception:
                    pass
            
            image_path = record.get('image_path')
            
            if not image_path or not os.path.exists(image_path):
                continue
            
            # 分类模型
            if 'stable-diffusion' in model_name.lower() or 'sd' in model_name.lower() or 'runwayml' in model_name.lower():
                model_key = 'sd-base'
            elif 'clip' in model_name.lower() or 'openai' in model_name.lower():
                model_key = 'clip-fusion'
            elif 'itsc' in model_name.lower() or 'gan' in model_name.lower():
                model_key = 'itsc-gan'
            else:
                continue
            
            try:
                image = Image.open(image_path)
                img_html = self._image_to_html(image, max_width=200, max_height=200)
                # PIL Image对象会自动管理资源，不需要显式关闭
                
                model_images[model_key].append({
                    'image_html': img_html,
                    'image_path': image_path,
                    'prompt': record.get('prompt', ''),
                    'score': record.get('consistency_score', 0),
                    'created_at': record.get('created_at', '')
                })
            except Exception as e:
                print(f"[WARNING] 无法加载图像 {image_path}: {e}")
        
        return model_images
    
    def _get_recent_images(self, user_id: int = None, limit: int = 12) -> List[Dict]:
        """获取最近生成的图片"""
        try:
            if user_id:
                history = self.db.get_user_history(user_id)
            else:
                history = []
        except Exception as e:
            print(f"[WARNING] 获取历史记录失败: {e}")
            history = []
        
        # 按时间排序，取最近的（确保history是列表且不为None）
        if not isinstance(history, list):
            history = []
        history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        recent = history[:limit]
        
        recent_images = []
        for record in recent:
            image_path = record.get('image_path')
            if not image_path or not os.path.exists(image_path):
                continue
            
            # 尝试从result_data中解析model_name
            model_name = record.get('model_name', 'unknown')
            if model_name == 'unknown':
                try:
                    import json
                    result_data_str = record.get('result_data', '')
                    if result_data_str:
                        result_data = json.loads(result_data_str) if isinstance(result_data_str, str) else result_data_str
                        model_name = result_data.get('model_name', 'unknown')
                except Exception:
                    pass
            
            try:
                image = Image.open(image_path)
                img_html = self._image_to_html(image, max_width=150, max_height=150)
                # PIL Image对象会自动管理资源，不需要显式关闭
                
                recent_images.append({
                    'image_html': img_html,
                    'image_path': image_path,
                    'prompt': record.get('prompt', ''),
                    'model_name': model_name,
                    'score': record.get('consistency_score', 0),
                    'created_at': record.get('created_at', '')
                })
            except Exception as e:
                print(f"[WARNING] 无法加载图像 {image_path}: {e}")
        
        return recent_images
    
    def _image_to_html(self, image: Image.Image, max_width: int = 300, max_height: int = 300) -> str:
        """将PIL图像转换为HTML"""
        # 调整大小
        img_width, img_height = image.size
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"<img src='data:image/png;base64,{img_str}' style='max-width: 100%; height: auto; border-radius: 8px;' />"
    
    def generate_dashboard_html(self, user_id: int = None) -> str:
        """
        生成仪表盘HTML
        
        Args:
            user_id: 用户ID
        
        Returns:
            仪表盘HTML字符串
        """
        data = self.get_dashboard_data(user_id)
        
        # 生成统计卡片
        stats_html = self._generate_stats_cards(data.get('stats', {}))
        
        # 生成模型图片展示
        model_images_html = self._generate_model_images_section(data.get('model_images', {}))
        
        # 生成最近图片
        recent_images_html = self._generate_recent_images_section(data.get('recent_images', []))
        
        dashboard_html = f"""
        <div style="padding: 20px;">
            <h1 style="margin-top: 0; color: #333;">📊 仪表盘</h1>
            {stats_html}
            {model_images_html}
            {recent_images_html}
        </div>
        """
        
        return dashboard_html
    
    def _generate_stats_cards(self, stats: Dict) -> str:
        """生成统计卡片"""
        total = stats.get('total_generations', 0)
        today = stats.get('today_generations', 0)
        consistent = stats.get('consistent_count', 0)
        avg_score = stats.get('average_score', 0.0)
        
        return f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #fbbf24;">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">总生成次数</div>
                <div style="font-size: 32px; font-weight: bold; color: #333;">{total}</div>
            </div>
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #ef4444;">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">今日生成</div>
                <div style="font-size: 32px; font-weight: bold; color: #333;">{today}</div>
            </div>
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #8b5cf6;">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">一致性通过</div>
                <div style="font-size: 32px; font-weight: bold; color: #333;">{consistent}</div>
            </div>
            <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #6b7280;">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">平均一致性分数</div>
                <div style="font-size: 32px; font-weight: bold; color: #333;">{avg_score:.2f}</div>
            </div>
        </div>
        """
    
    def _generate_model_images_section(self, model_images: Dict) -> str:
        """生成模型图片展示区域"""
        sections = []
        
        for model_key, model_name in [('sd-base', 'SD基础模型'), ('clip-fusion', 'CLIP融合模型'), ('itsc-gan', 'ITSC-GAN模型')]:
            images = model_images.get(model_key, [])
            if not images:
                continue
            
            image_grid = "".join([
                f"<div style='margin: 10px;'>{img['image_html']}</div>"
                for img in images[:6]  # 最多显示6张
            ])
            
            sections.append(f"""
            <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #333;">{model_name} 生成图片</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px;">
                    {image_grid}
                </div>
            </div>
            """)
        
        return "".join(sections) if sections else "<div style='padding: 20px; background: #f5f5f5; border-radius: 10px;'>暂无生成图片</div>"
    
    def _generate_recent_images_section(self, recent_images: List[Dict]) -> str:
        """生成最近图片区域"""
        if not recent_images:
            return "<div style='padding: 20px; background: #f5f5f5; border-radius: 10px;'>暂无最近生成的图片</div>"
        
        image_grid = "".join([
            f"""
            <div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                {img['image_html']}
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{img['prompt'][:30]}...</p>
            </div>
            """
            for img in recent_images
        ])
        
        return f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333;">🕐 最近生成的图片</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px;">
                {image_grid}
            </div>
        </div>
        """

