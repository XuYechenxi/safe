#!/usr/bin/env python3
"""
模型对比服务模块
支持多个模型生成并对比，包含进度回调和异步处理功能
"""

from typing import List, Dict, Any, Optional, Callable
from PIL import Image
from image_generation_service import ImageGenerationService
from consistency_detection_service import ConsistencyDetectionService
import threading
import time
import queue
import torch
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None

class ModelComparisonService:
    """模型对比服务，支持进度回调和异步处理"""
    
    def __init__(self, device: str = None, db=None):
        """初始化服务"""
        # 安全检测设备
        if device is None:
            device = self._safe_get_device()
        else:
            # 验证指定的设备是否可用
            if device == "cuda":
                try:
                    import torch
                    test_tensor = torch.tensor([1.0])
                    test_tensor = test_tensor.cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                except (AssertionError, RuntimeError):
                    print(f"[WARNING] 指定的CUDA设备不可用，回退到CPU")
                    device = "cpu"
        
        self.detection_service = ConsistencyDetectionService(device)
        self.device = device
        self.db = db  # 数据库实例，用于检查历史记录
        print(f"[INFO] 模型对比服务初始化，使用设备: {self.device}")
        self.lock = threading.Lock()
        # 用于异步任务的结果队列和进度跟踪
        self.result_queues = {}
        self.task_progress = {}
        # 图片缓存（内存缓存，避免重复加载）
        self._image_cache = {}
    
    def _safe_get_device(self):
        """安全检测可用的计算设备"""
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            # 尝试实际使用CUDA来验证
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.cuda()
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda"
        except (AssertionError, RuntimeError):
            return "cpu"
        except Exception:
            return "cpu"
    
    def _check_history_for_image(self, model_name: str, prompt: str, 
                                 num_inference_steps: int, guidance_scale: float,
                                 height: int, width: int, user_id: Optional[int] = None):
        """
        检查历史记录中是否有相同参数的已生成图片
        
        Returns:
            (image, image_path) 如果找到，否则 (None, None)
        """
        if not self.db or not user_id:
            return None, None
        
        try:
            # 获取用户历史记录
            history = self.db.get_user_history(user_id, limit=100)
            
            for record in history:
                # 检查prompt和模型是否匹配
                if record.get('prompt', '').strip() != prompt.strip():
                    continue
                
                # 检查result_data中的模型名称
                try:
                    result_data_str = record.get('result_data', '')
                    if result_data_str:
                        import json
                        result_data = json.loads(result_data_str) if isinstance(result_data_str, str) else result_data_str
                        record_model = result_data.get('model_name', '')
                        if record_model != model_name:
                            continue
                except:
                    pass
                
                # 检查图片文件是否存在
                image_path = record.get('image_path', '')
                if image_path and os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        # 检查尺寸是否匹配（允许小误差）
                        if abs(image.size[0] - width) <= 10 and abs(image.size[1] - height) <= 10:
                            print(f"[INFO] 找到历史图片: {image_path}")
                            return image, image_path
                    except Exception as e:
                        print(f"[WARNING] 加载历史图片失败: {e}")
                        continue
            
            return None, None
        except Exception as e:
            print(f"[WARNING] 检查历史记录失败: {e}")
            return None, None
    
    def _generate_for_model(self, model_name: str, prompt: str, 
                           num_inference_steps: int, guidance_scale: float,
                           height: int, width: int, seed: Optional[int],
                           result_dict: Dict[str, Any], index: int,
                           progress_callback: Optional[Callable[[float, str], None]] = None,
                           user_id: Optional[int] = None,
                           reuse_cache: bool = True):
        """
        为单个模型生成图像并保存结果，支持进度回调和图片复用
        
        Args:
            reuse_cache: 是否复用历史图片（如果存在）
        """
        generation_service = None
        thread_id = threading.current_thread().ident
        try:
            print(f"[INFO] [线程 {thread_id}] 使用模型 {model_name} 生成图像...")
            
            # 首先检查历史记录中是否有可复用的图片
            if reuse_cache and user_id:
                cached_image, cached_path = self._check_history_for_image(
                    model_name, prompt, num_inference_steps, guidance_scale, height, width, user_id
                )
                if cached_image is not None:
                    print(f"[INFO] [线程 {thread_id}] 复用历史图片: {cached_path}")
                    # 更新进度
                    if progress_callback:
                        try:
                            progress_callback(100.0, "复用历史图片")
                        except:
                            pass
                    
                    # 存储结果（使用锁保护）
                    with self.lock:
                        result_dict[index] = {
                            'model_name': model_name,
                            'image': cached_image,
                            'image_path': cached_path,
                            'reused': True  # 标记为复用
                        }
                    
                    print(f"[INFO] [线程 {thread_id}] 模型 {model_name} 图片复用成功")
                    return
            
            # 如果没有找到历史图片，则生成新图片
            print(f"[INFO] [线程 {thread_id}] 未找到历史图片，开始生成新图片...")
            
            # 创建独立的生成服务实例以确保独立性（每个线程使用独立实例）
            # 使用当前服务的设备设置
            generation_service = ImageGenerationService(device=self.device)
            print(f"[INFO] [线程 {thread_id}] 使用设备: {self.device}")
            print(f"[INFO] [线程 {thread_id}] 生成服务实例创建完成")
            
            # 生成图像，传递进度回调
            current_seed = seed + index if seed is not None else None
            print(f"[INFO] [线程 {thread_id}] 开始生成图像...")
            image = generation_service.generate(
                prompt=prompt,
                model_name=model_name,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=current_seed,
                device=self.device,
                progress_callback=progress_callback
            )
            
            if image is None:
                raise RuntimeError("图像生成返回None")
            
            print(f"[INFO] [线程 {thread_id}] 图像生成完成，开始保存...")
            
            # 保存图像
            image_path = generation_service.save_image(
                image, prompt, model_name
            )
            
            print(f"[INFO] [线程 {thread_id}] 图像保存完成: {image_path}")
            
            # 存储结果（使用锁保护）
            with self.lock:
                result_dict[index] = {
                    'model_name': model_name,
                    'image': image,
                    'image_path': image_path,
                    'reused': False  # 标记为新生成
                }
            
            print(f"[INFO] [线程 {thread_id}] 模型 {model_name} 图像生成成功")
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            print(f"[ERROR] [线程 {thread_id}] 模型 {model_name} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            
            with self.lock:
                result_dict[index] = {
                    'model_name': model_name,
                    'error': error_msg
                }
            
            # 如果有进度回调，通知错误
            if progress_callback:
                try:
                    progress_callback(0.0, f"错误: {str(e)}")
                except Exception as callback_error:
                    print(f"[WARNING] [线程 {thread_id}] 进度回调失败: {callback_error}")
            
        finally:
            # 清理实例和资源（确保总是执行）
            print(f"[INFO] [线程 {thread_id}] 开始清理资源...")
            if generation_service:
                try:
                    generation_service.clear_cache()
                    print(f"[INFO] [线程 {thread_id}] 服务缓存清理完成")
                except Exception as cleanup_error:
                    print(f"[WARNING] [线程 {thread_id}] 清理服务资源失败: {cleanup_error}")
            
            # 释放CUDA内存
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print(f"[INFO] [线程 {thread_id}] CUDA内存清理完成")
                except Exception as cuda_error:
                    print(f"[WARNING] [线程 {thread_id}] 清理CUDA内存失败: {cuda_error}")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            print(f"[INFO] [线程 {thread_id}] 资源清理完成")
    
    def compare_models(
        self,
        prompt: str,
        model_names: List[str],
        threshold: float = 0.3,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        device: str = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        user_id: Optional[int] = None,
        reuse_cache: bool = True
    ) -> Dict[str, Any]:
        """
        对比多个模型 - 使用线程并行处理每个模型，支持进度回调
        
        Args:
            prompt: 文本提示词
            model_names: 模型名称列表
            threshold: 一致性阈值
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            device: 计算设备
            progress_callback: 进度回调函数，接收进度百分比(0-100)和状态信息
        
        Returns:
            对比结果字典
        """
        start_time = time.time()
        
        # 更新设备信息（安全检测）
        if device is not None:
            # 验证指定的设备是否可用
            if device == "cuda":
                try:
                    test_tensor = torch.tensor([1.0])
                    test_tensor = test_tensor.cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                except (AssertionError, RuntimeError):
                    print(f"[WARNING] 指定的CUDA设备不可用，回退到CPU")
                    device = "cpu"
            self.device = device
        else:
            # 使用初始化时的设备
            device = self.device
        
        # 初始化进度回调函数
        def _update_progress(progress: float, status: str = ""):
            """内部进度更新函数"""
            if progress_callback:
                try:
                    progress_callback(progress, status)
                except Exception as e:
                    print(f"[WARNING] 进度回调执行失败: {e}")
        
        _update_progress(0.0, "开始准备模型对比...")
        
        # 第一阶段：并行生成所有模型的图像
        threads = []
        temp_results = {}
        
        print(f"[INFO] 开始并行生成 {len(model_names)} 个模型的图像...")
        _update_progress(5.0, "开始并行生成图像...")
        
        total_models = len(model_names)
        # 模型进度跟踪器
        model_progress = {model: 0.0 for model in model_names}
        progress_lock = threading.Lock()
        
        # 全局进度计算函数
        def calculate_global_progress():
            """计算所有模型的平均进度"""
            with progress_lock:
                if not model_progress:
                    return 5.0  # 初始进度
                avg_progress = sum(model_progress.values()) / len(model_progress)
                # 阶段1占50%的总进度
                return 5.0 + (avg_progress * 0.45)
        
        # 为每个模型创建线程和进度回调
        # 注意：为了避免资源竞争，每个线程使用独立的服务实例
        for i, model_name in enumerate(model_names):
            # 创建模型特定的进度回调（使用闭包捕获正确的model_name）
            def create_model_callback(model):
                def model_progress_callback(progress, status):
                    try:
                        with progress_lock:
                            model_progress[model] = progress / 100.0  # 归一化到0-1范围
                        global_progress = calculate_global_progress()
                        _update_progress(global_progress, f"模型 {model}: {status}")
                    except Exception as e:
                        print(f"[WARNING] 进度回调失败: {e}")
                return model_progress_callback
            
            # 创建线程，设置daemon=True以便主程序退出时自动清理
            thread = threading.Thread(
                target=self._generate_for_model,
                args=(model_name, prompt, num_inference_steps, guidance_scale,
                      height, width, seed, temp_results, i),
                kwargs={
                    'progress_callback': create_model_callback(model_name),
                    'user_id': user_id,
                    'reuse_cache': reuse_cache
                },
                daemon=False  # 不使用daemon，确保线程完成
            )
            threads.append(thread)
            thread.start()
            print(f"[INFO] 已启动线程 {i} 用于模型 {model_name}")
        
        # 等待所有线程完成（添加超时机制，防止卡住）
        # 注意：time 模块已在文件顶部导入，不需要重复导入
        max_wait_time = 600  # 最多等待10分钟
        wait_start_time = time.time()  # 使用不同的变量名避免与函数开头的 start_time 冲突
        
        for i, thread in enumerate(threads):
            remaining_time = max_wait_time - (time.time() - wait_start_time)
            if remaining_time <= 0:
                print(f"[WARNING] 等待线程超时，跳过剩余线程")
                break
            
            thread.join(timeout=max(remaining_time / len(threads), 30))  # 每个线程最多等待30秒或剩余时间
            if thread.is_alive():
                print(f"[WARNING] 线程 {i} 超时，可能仍在运行")
                # 继续处理，不阻塞其他线程的结果
        
        print(f"[INFO] 所有图像生成完成（耗时: {time.time() - wait_start_time:.2f}秒）")
        _update_progress(50.0, "所有图像生成完成，开始一致性检测")
        
        # 第二阶段：处理生成的图像和一致性检测
        results = []
        valid_results = []
        
        # 计算每个模型检测占总进度的比例
        if temp_results:
            detection_progress_step = 45.0 / len(temp_results)
        else:
            detection_progress_step = 0
        
        detection_progress = 50.0
        
        for i in range(len(model_names)):
            if i in temp_results:
                result = temp_results[i]
                if "error" not in result:
                    try:
                        # 检查图像是否存在
                        if 'image' not in result or result['image'] is None:
                            print(f"[WARNING] 模型 {result.get('model_name', 'unknown')} 的图像为空，跳过检测")
                            result['error'] = "图像生成失败或为空"
                            results.append(result)
                            continue
                        
                        # 更新进度
                        detection_progress += detection_progress_step * 0.2  # 20%用于准备
                        _update_progress(detection_progress, f"开始检测模型 {result['model_name']} 的一致性...")
                        
                        # 检测一致性（传递模型名称以计算模型特定分数）
                        try:
                            is_consistent, score, detail = self.detection_service.detect(
                                result['image'], prompt, threshold, model_name=result['model_name']
                            )
                        except Exception as detect_error:
                            print(f"[ERROR] 模型 {result['model_name']} 一致性检测异常: {detect_error}")
                            import traceback
                            traceback.print_exc()
                            raise
                        
                        # 更新进度
                        detection_progress += detection_progress_step * 0.8  # 80%用于检测
                        _update_progress(min(detection_progress, 95.0), 
                                        f"模型 {result['model_name']} 一致性检测完成，分数: {score:.4f}")
                        
                        # 更新结果
                        result.update({
                            'is_consistent': is_consistent,
                            'overall_score': score,
                            'clip_score': detail.get('clip_score', score) if isinstance(detail, dict) else score,
                            'fused_score': detail.get('fused_score', score) if isinstance(detail, dict) else score,
                            'detail': detail
                        })
                        valid_results.append(result)
                        
                    except Exception as e:
                        detection_progress += detection_progress_step
                        _update_progress(min(detection_progress, 95.0), 
                                        f"模型 {result.get('model_name', 'unknown')} 一致性检测失败")
                        print(f"[ERROR] 模型 {result.get('model_name', 'unknown')} 一致性检测失败: {e}")
                        import traceback
                        traceback.print_exc()
                        result['error'] = f"一致性检测失败: {str(e)}"
                results.append(result)
            else:
                # 如果某个模型没有结果，添加错误记录
                print(f"[WARNING] 模型 {model_names[i]} (索引 {i}) 没有生成结果")
                results.append({
                    'model_name': model_names[i],
                    'error': '生成失败或超时'
                })
        
        # 找出最佳和最差模型
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['overall_score'])
            worst_result = min(valid_results, key=lambda x: x['overall_score'])
        else:
            best_result = None
            worst_result = None
        
        # ====== 生成可视化对比结果图 ======
        comparison_image_path = None
        score_chart_path = None
        
        try:
            if valid_results:
                # 确保输出目录存在
                output_dir = os.path.join("outputs", "comparisons")
                os.makedirs(output_dir, exist_ok=True)
                
                # 1）生成多模型拼接效果图（对比方法一）
                try:
                    # 统一高度，横向拼接
                    resized_images = []
                    min_height = min(
                        img_result['image'].height
                        for img_result in valid_results
                        if 'image' in img_result and img_result.get('image') is not None
                    )
                    for img_result in valid_results:
                        img = img_result.get('image')
                        if img is None:
                            continue
                        if img.height != min_height:
                            new_width = int(img.width * (min_height / img.height))
                            img = img.resize((new_width, min_height), Image.LANCZOS)
                        resized_images.append((img, img_result['model_name']))
                    
                    if resized_images:
                        total_width = sum(img.width for img, _ in resized_images)
                        comparison_image = Image.new(
                            "RGB", (total_width, min_height + 40), color=(245, 245, 245)
                        )
                        
                        # 粘贴图像并在下方写模型名称
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(comparison_image)
                        try:
                            font = ImageFont.truetype("arial.ttf", 18)
                        except Exception:
                            font = ImageFont.load_default()
                        
                        x_offset = 0
                        for img, model_name in resized_images:
                            comparison_image.paste(img, (x_offset, 0))
                            text = str(model_name)
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_w = bbox[2] - bbox[0]
                            text_h = bbox[3] - bbox[1]
                            text_x = x_offset + (img.width - text_w) // 2
                            text_y = min_height + (40 - text_h) // 2
                            draw.text((text_x, text_y), text, fill=(50, 50, 50), font=font)
                            x_offset += img.width
                        
                        safe_prompt = "".join(
                            c for c in (prompt or "")[:30] if c.isalnum() or c in ("_", "-", " ")
                        ).strip().replace(" ", "_") or "comparison"
                        comparison_filename = f"comparison_grid_{safe_prompt}_{int(time.time())}.png"
                        comparison_image_path = os.path.join(output_dir, comparison_filename)
                        comparison_image.save(comparison_image_path)
                        print(f"[INFO] 已生成对比效果图: {comparison_image_path}")
                except Exception as e:
                    print(f"[WARNING] 生成拼接对比图失败: {e}")
                
                # 2）生成评分柱状图（对比方法二）
                try:
                    if matplotlib is not None and plt is not None:
                        model_labels = [str(res['model_name']) for res in valid_results]
                        scores = [float(res.get('overall_score', 0.0)) for res in valid_results]
                        
                        plt.figure(figsize=(6, 4))
                        bars = plt.bar(
                            model_labels,
                            scores,
                            color=['#667eea', '#22c55e', '#f97316', '#3b82f6'][: len(model_labels)],
                        )
                        plt.ylim(0.0, 1.0)
                        try:
                            plt.ylabel("语义一致性得分", fontproperties="SimHei")
                            plt.title("不同模型的一致性评分对比", fontproperties="SimHei")
                        except Exception:
                            plt.ylabel("Score")
                            plt.title("Model Consistency Scores")
                        
                        # 在柱子上方标注具体数值
                        for bar, score in zip(bars, scores):
                            height = bar.get_height()
                            plt.text(
                                bar.get_x() + bar.get_width() / 2,
                                height + 0.01,
                                f"{score:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )
                        
                        plt.tight_layout()
                        safe_prompt = "".join(
                            c for c in (prompt or "")[:30] if c.isalnum() or c in ("_", "-", " ")
                        ).strip().replace(" ", "_") or "comparison"
                        chart_filename = f"comparison_scores_{safe_prompt}_{int(time.time())}.png"
                        score_chart_path = os.path.join(output_dir, chart_filename)
                        plt.savefig(score_chart_path, dpi=150)
                        plt.close()
                        print(f"[INFO] 已生成评分柱状图: {score_chart_path}")
                    else:
                        print("[WARNING] matplotlib 不可用，跳过评分柱状图生成")
                except Exception as e:
                    print(f"[WARNING] 生成评分柱状图失败: {e}")
        except Exception as e:
            print(f"[WARNING] 生成可视化对比结果图时出错: {e}")
        
        comparison_time = time.time() - start_time
        _update_progress(100.0, f"模型对比完成，耗时: {comparison_time:.2f}秒")
        
        return {
            'prompt': prompt,
            'results': results,
            'best_model': best_result,
            'worst_model': worst_result,
            'total_models': len(model_names),
            'successful_models': len(valid_results),
            'comparison_time': comparison_time,
            'comparison_image_path': comparison_image_path,
            'score_chart_path': score_chart_path,
        }
    
    def compare_models_async(self, task_id: str, prompt: str, model_names: List[str],
                           **kwargs) -> str:
        """
        异步对比多个模型
        
        Args:
            task_id: 任务ID，用于标识异步任务
            prompt: 文本提示词
            model_names: 模型名称列表
            **kwargs: 其他参数，同compare_models
            
        Returns:
            任务ID，用于后续查询结果
        """
        # 创建结果队列和进度跟踪
        with self.lock:
            if task_id in self.result_queues:
                del self.result_queues[task_id]
            if task_id in self.task_progress:
                del self.task_progress[task_id]
            self.result_queues[task_id] = queue.Queue()
            self.task_progress[task_id] = {
                'current': 0.0,
                'status': '',
                'complete': False,
                'result': None,
                'error': None
            }
        
        # 定义进度回调
        def progress_callback(progress_percent, status):
            with self.lock:
                if task_id in self.task_progress:
                    self.task_progress[task_id]['current'] = progress_percent
                    self.task_progress[task_id]['status'] = status
        
        # 在后台线程中执行对比
        def async_comparison():
            try:
                result = self.compare_models(
                    prompt=prompt,
                    model_names=model_names,
                    progress_callback=progress_callback,
                    **kwargs
                )
                with self.lock:
                    if task_id in self.task_progress:
                        self.task_progress[task_id]['complete'] = True
                        self.task_progress[task_id]['result'] = result
                self.result_queues[task_id].put(('success', result))
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] 异步对比失败: {error_msg}")
                with self.lock:
                    if task_id in self.task_progress:
                        self.task_progress[task_id]['complete'] = True
                        self.task_progress[task_id]['error'] = error_msg
                self.result_queues[task_id].put(('error', error_msg))
        
        # 启动异步任务
        thread = threading.Thread(target=async_comparison, daemon=True)
        thread.start()
        
        return task_id
    
    def get_comparison_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取异步对比任务的状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        with self.lock:
            if task_id not in self.result_queues:
                return {
                    'status': 'not_found',
                    'progress': 0.0,
                    'message': '任务不存在'
                }
            
            # 检查是否有结果
            try:
                result_type, result = self.result_queues[task_id].get_nowait()
                # 清理队列
                del self.result_queues[task_id]
                if task_id in self.task_progress:
                    del self.task_progress[task_id]
                
                if result_type == 'success':
                    return {
                        'status': 'completed',
                        'progress': 100.0,
                        'result': result
                    }
                else:
                    return {
                        'status': 'error',
                        'progress': 0.0,
                        'error': result
                    }
            except queue.Empty:
                # 任务仍在进行中
                if task_id in self.task_progress:
                    progress_info = self.task_progress[task_id]
                    return {
                        'status': 'running',
                        'progress': progress_info['current'],
                        'message': progress_info['status']
                    }
                else:
                    return {
                        'status': 'running',
                        'progress': 0.0,
                        'message': '处理中...'
                    }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消异步任务（注意：只能清理状态，无法真正中断正在运行的任务）
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功取消
        """
        with self.lock:
            if task_id in self.result_queues:
                del self.result_queues[task_id]
            if task_id in self.task_progress:
                del self.task_progress[task_id]
            return True
        return False

