"""
面向文本引导生成图像的语义一致性检测系统
基于一致性模型（Consistency Model）和CLIP
主程序入口
"""
import argparse
import os
import random
from typing import Optional

import config_cm as config
from consistency_model_generator import ConsistencyModelGenerator, create_generator
from semantic_consistency_detector import SemanticConsistencyDetector
from PIL import Image
import json
from datetime import datetime


class SemanticConsistencySystem:
    """语义一致性检测系统 - 基于一致性模型"""
    
    def __init__(self):
        """初始化系统"""
        self.generator = None
        self.detector = None
        self.current_model = None
    
    def initialize(self, model_path=None):
        """初始化系统"""
        print("=" * 60)
        print("正在初始化一致性模型系统...")
        print("=" * 60)
        
        # 初始化一致性模型生成器
        print("正在初始化一致性模型生成器...")
        try:
            # 使用传入的模型路径或配置文件中的默认值
            current_model_path = model_path if model_path else config.CONSISTENCY_MODEL_PATH
            self.generator = create_generator(
                model_path=current_model_path,
                clip_model_name=config.CLIP_MODEL_NAME,
                image_size=config.CONSISTENCY_MODEL_IMAGE_SIZE,
                device=config.DEVICE,
            )
            self.current_model = "consistency-model"
            print(f"✅ 一致性模型生成器初始化完成")
            print(f"   图像尺寸: {config.CONSISTENCY_MODEL_IMAGE_SIZE}x{config.CONSISTENCY_MODEL_IMAGE_SIZE}")
            print(f"   CLIP模型: {config.CLIP_MODEL_NAME}")
            print(f"   生成步数: {config.DEFAULT_NUM_STEPS} (单步生成)")
            print("   提示: 一致性模型支持单步生成，速度极快！")
        except Exception as e:
            print(f"❌ 一致性模型生成器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 初始化语义检测器（使用CLIP）
        print("正在初始化语义一致性检测器（CLIP）...")
        try:
            self.detector = SemanticConsistencyDetector(
                device=config.DEVICE,
                use_simple_mode=not config.DETECTOR_USE_CLIP,
                lazy_load_clip=False
            )
            print(f"✅ 检测器初始化完成（CLIP模式）")
        except Exception as e:
            print(f"⚠️  CLIP检测器初始化失败: {e}")
            print("[INFO] 使用简化模式作为兜底")
            self.detector = SemanticConsistencyDetector(
                device=config.DEVICE,
                use_simple_mode=True,
                lazy_load_clip=False
            )
        
        print("=" * 60)
        print("✅ 系统初始化完成！")
        print("=" * 60)
    
    def switch_model(self, model_path=None):
        """
        切换一致性模型（重新加载）
        
        Args:
            model_path: 新的模型路径
        """
        print(f"正在重新加载一致性模型...")
        try:
            # 重新初始化生成器
            self.generator = create_generator(
                model_path=model_path or config.CONSISTENCY_MODEL_PATH,
                clip_model_name=config.CLIP_MODEL_NAME,
                image_size=config.CONSISTENCY_MODEL_IMAGE_SIZE,
                device=config.DEVICE,
            )
            self.current_model = "consistency-model"
            print(f"✅ 模型重新加载完成")
        except Exception as e:
            print(f"❌ 模型切换失败: {e}")
            raise
    
    def process_single_request(self,
                               prompt: str,
                               save_dir: str = "output",
                               best_of: Optional[int] = None,
                               seed: Optional[int] = None) -> dict:
        """
        处理单个请求
        
        Args:
            prompt: 文本提示词
            save_dir: 输出目录
            
        Returns:
            处理结果
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 多候选生成
        best_of = best_of or config.BEST_OF_SAMPLES
        print(f"\n正在生成图像，提示词: {prompt}")
        print(f"[INFO] 将生成 {best_of} 个候选图像以挑选最佳结果")

        candidates = []
        best_candidate = None

        for idx in range(best_of):
            current_seed = seed + idx if seed is not None else random.randint(0, 2**32 - 1)
            print(f"  -> 候选 {idx + 1}/{best_of}，seed={current_seed}")
            # 一致性模型生成（单步生成）
            image = self.generator.generate(
                prompt=prompt,
                num_inference_steps=config.DEFAULT_NUM_STEPS,
                guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
                seed=current_seed
            )

            print("  -> 正在检测语义一致性...")
            is_consistent, score, detail = self.detector.detect_consistency(image, prompt)
            detail['seed'] = current_seed
            detail['candidate_index'] = idx + 1

            candidate_info = {
                "index": idx + 1,
                "seed": current_seed,
                "score": score,
                "is_consistent": is_consistent,
            }
            candidates.append(candidate_info)

            if not best_candidate or score > best_candidate["score"]:
                best_candidate = {
                    "image": image,
                    "score": score,
                    "is_consistent": is_consistent,
                    "detail": detail,
                    "seed": current_seed,
                    "index": idx + 1,
                }
                print(f"  -> 当前最佳候选: index={idx + 1}, seed={current_seed}, score={score:.4f}")
            else:
                print(f"  -> 候选 {idx + 1} 得分 {score:.4f} 未超过当前最佳")

        image = best_candidate["image"]
        is_consistent = best_candidate["is_consistent"]
        score = best_candidate["score"]
        detail = best_candidate["detail"]
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{prompt[:20].replace(' ', '_')}"
        
        image_path = os.path.join(save_dir, f"{filename}.png")
        image.save(image_path)
        
        result_path = os.path.join(save_dir, f"{filename}_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果:")
        print(f"  - 最佳候选: {best_candidate['index']}/{best_of}")
        print(f"  - 最佳 seed: {best_candidate['seed']}")
        print(f"  - 一致性: {'通过' if is_consistent else '未通过'}")
        print(f"  - 分数: {score:.4f}")
        print(f"  - 图像保存至: {image_path}")
        print(f"  - 结果保存至: {result_path}")
        
        detail.update({
            "seed": best_candidate["seed"],
            "best_of": best_of,
            "generation_config": {
                "num_inference_steps": config.DEFAULT_NUM_STEPS,
                "guidance_scale": config.DEFAULT_GUIDANCE_SCALE,
                "height": config.DEFAULT_HEIGHT,
                "width": config.DEFAULT_WIDTH,
            },
            "candidates": candidates,
        })
        detail['image_path'] = image_path
        detail['result_path'] = result_path
        return detail
    
    def process_batch(self, prompts: list, save_dir: str = "output") -> dict:
        """
        批量处理请求
        
        Args:
            prompts: 提示词列表
            save_dir: 输出目录
            
        Returns:
            批量处理结果
        """
        print(f"\n开始批量处理 {len(prompts)} 个请求...")
        
        # 生成图像
        images = self.generator.generate_batch(
            prompts,
            num_inference_steps=config.DEFAULT_NUM_STEPS,
            guidance_scale=config.DEFAULT_GUIDANCE_SCALE,
            height=config.DEFAULT_HEIGHT,
            width=config.DEFAULT_WIDTH
        )
        
        # 批量检测
        batch_results = self.detector.evaluate_batch(images, prompts)
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            filename = f"{timestamp}_{i:03d}_{prompt[:20].replace(' ', '_')}"
            image_path = os.path.join(save_dir, f"{filename}.png")
            image.save(image_path)
        
        # 保存结果
        result_path = os.path.join(save_dir, f"{timestamp}_batch_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量处理完成:")
        print(f"  - 总图像数: {batch_results['total_images']}")
        print(f"  - 一致性数量: {batch_results['consistent_count']}")
        print(f"  - 一致性率: {batch_results['consistency_rate']:.2%}")
        print(f"  - 平均分数: {batch_results['average_score']:.4f}")
        print(f"  - 结果保存至: {result_path}")
        
        return batch_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='面向文本引导生成图像的语义一致性检测系统'
    )
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'interactive'],
                       default='interactive', help='运行模式')
    parser.add_argument('--prompt', type=str, help='文本提示词（单次模式）')
    parser.add_argument('--prompts', type=str, nargs='+', help='文本提示词列表（批量模式）')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.3, help='一致性阈值')
    parser.add_argument('--best-of', type=int, default=config.BEST_OF_SAMPLES,
                        help='为每个提示生成的候选数量，自动选择评分最高的结果')
    parser.add_argument('--seed', type=int, help='起始随机种子（配合 best-of 会顺序递增）')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = SemanticConsistencySystem()
    system.initialize()
    system.detector.threshold = args.threshold
    
    if args.mode == 'single':
        if not args.prompt:
            print("错误: 单次模式需要提供 --prompt 参数")
            return
        system.process_single_request(
            prompt=args.prompt,
            save_dir=args.output,
            best_of=args.best_of,
            seed=args.seed
        )
    
    elif args.mode == 'batch':
        if not args.prompts:
            print("错误: 批量模式需要提供 --prompts 参数")
            return
        system.process_batch(args.prompts, args.output)
    
    elif args.mode == 'interactive':
        print("\n=== 交互模式 ===")
        print("输入文本提示词生成图像并检测语义一致性")
        print("输入 'quit' 或 'exit' 退出")
        
        while True:
            try:
                prompt = input("\n请输入提示词: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("退出程序")
                    break
                
                if not prompt:
                    print("请输入有效的提示词")
                    continue
                
                system.process_single_request(
                    prompt=prompt,
                    save_dir=args.output,
                    best_of=args.best_of,
                    seed=args.seed
                )
                
            except KeyboardInterrupt:
                print("\n程序被中断")
                break
            except Exception as e:
                print(f"错误: {e}")


if __name__ == "__main__":
    main()

