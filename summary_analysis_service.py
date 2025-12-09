#!/usr/bin/env python3
"""
总结分析服务模块
对不同模型的结果进行汇总总结，生成表格和图表
"""

from typing import List, Dict, Any
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64


class SummaryAnalysisService:
    """总结分析服务"""
    
    def __init__(self):
        """初始化服务"""
        pass
    
    def generate_summary(
        self,
        comparison_results: Dict[str, Any],
        include_charts: bool = True
    ) -> Dict[str, Any]:
        """
        生成对比总结
        
        Args:
            comparison_results: 模型对比结果
            include_charts: 是否包含图表
        
        Returns:
            总结结果字典
        """
        results = comparison_results.get('results', [])
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        if not valid_results:
            # 生成详细的错误信息
            error_messages = []
            for err_result in error_results:
                model_name = err_result.get('model_name', '未知模型')
                error_msg = err_result.get('error', '未知错误')
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
            
            return {
                'summary_html': error_html,
                'table_html': "",
                'chart_html': ""
            }
        
        # 生成表格
        table_html = self._generate_table(valid_results)
        
        # 生成图表
        chart_html = ""
        if include_charts:
            chart_html = self._generate_charts(valid_results)
        
        # 生成总结文本
        summary_text = self._generate_summary_text(valid_results, comparison_results)
        
        # 组合HTML
        summary_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin-top: 0;">📊 模型对比总结</h2>
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                {summary_text}
            </div>
        </div>
        {table_html}
        {chart_html}
        """
        
        return {
            'summary_html': summary_html,
            'table_html': table_html,
            'chart_html': chart_html,
            'summary_text': summary_text
        }
    
    def _generate_table(self, results: List[Dict]) -> str:
        """生成对比表格"""
        table_rows = []
        for i, result in enumerate(results, 1):
            model_name = result['model_name']
            overall_score = result.get('overall_score', 0)
            clip_score = result.get('clip_score', 0)
            fused_score = result.get('fused_score', 0)
            is_consistent = result.get('is_consistent', False)
            
            status_icon = "✅" if is_consistent else "❌"
            status_color = "#4CAF50" if is_consistent else "#F44336"
            
            table_rows.append(f"""
            <tr>
                <td>{i}</td>
                <td><strong>{model_name}</strong></td>
                <td style="color: {status_color}; font-weight: bold;">{status_icon} {'通过' if is_consistent else '未通过'}</td>
                <td style="font-weight: bold;">{overall_score:.4f}</td>
                <td>{clip_score:.4f}</td>
                <td>{fused_score:.4f}</td>
            </tr>
            """)
        
        table_html = f"""
        <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="margin-top: 0; color: #333;">📋 详细对比表格</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f5f5f5;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">排名</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">模型名称</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">一致性状态</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">整体分数</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">CLIP分数</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">融合分数</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        """
        
        return table_html
    
    def _generate_charts(self, results: List[Dict]) -> str:
        """生成对比图表"""
        try:
            # 准备数据
            model_names = [r['model_name'] for r in results]
            overall_scores = [r.get('overall_score', 0) for r in results]
            clip_scores = [r.get('clip_score', 0) for r in results]
            fused_scores = [r.get('fused_score', 0) for r in results]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 柱状图
            x = range(len(model_names))
            width = 0.25
            ax1.bar([i - width for i in x], overall_scores, width, label='整体分数', color='#667eea')
            ax1.bar(x, clip_scores, width, label='CLIP分数', color='#f093fb')
            ax1.bar([i + width for i in x], fused_scores, width, label='融合分数', color='#4facfe')
            ax1.set_xlabel('模型')
            ax1.set_ylabel('分数')
            ax1.set_title('模型分数对比（柱状图）')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 折线图
            ax2.plot(model_names, overall_scores, marker='o', label='整体分数', linewidth=2, color='#667eea')
            ax2.plot(model_names, clip_scores, marker='s', label='CLIP分数', linewidth=2, color='#f093fb')
            ax2.plot(model_names, fused_scores, marker='^', label='融合分数', linewidth=2, color='#4facfe')
            ax2.set_xlabel('模型')
            ax2.set_ylabel('分数')
            ax2.set_title('模型分数对比（折线图）')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # 转换为Base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            chart_html = f"""
            <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h3 style="margin-top: 0; color: #333;">📈 可视化对比图表</h3>
                <img src="data:image/png;base64,{chart_data}" style="width: 100%; max-width: 1000px; height: auto; border-radius: 8px;" />
            </div>
            """
            
            return chart_html
            
        except Exception as e:
            print(f"[ERROR] 生成图表失败: {e}")
            return f"<div style='padding: 20px; background: #ff9800; border-radius: 10px;'>图表生成失败: {str(e)}</div>"
    
    def _generate_summary_text(self, results: List[Dict], comparison_results: Dict) -> str:
        """生成总结文本"""
        if not results:
            return "<p>没有有效的对比结果</p>"
        
        # 排序结果
        sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
        
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        # 计算平均值
        avg_overall = sum(r.get('overall_score', 0) for r in results) / len(results)
        avg_clip = sum(r.get('clip_score', 0) for r in results) / len(results)
        avg_fused = sum(r.get('fused_score', 0) for r in results) / len(results)
        
        # 计算差异
        score_diff = best.get('overall_score', 0) - worst.get('overall_score', 0)
        
        summary = f"""
        <p style="margin: 8px 0;"><strong>📊 对比模型数量:</strong> {len(results)}</p>
        <p style="margin: 8px 0;"><strong>🏆 最佳模型:</strong> {best['model_name']} (分数: {best.get('overall_score', 0):.4f})</p>
        <p style="margin: 8px 0;"><strong>📉 最低分数模型:</strong> {worst['model_name']} (分数: {worst.get('overall_score', 0):.4f})</p>
        <p style="margin: 8px 0;"><strong>📈 分数差异:</strong> {score_diff:.4f} ({score_diff*100:.2f}%)</p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
        <p style="margin: 8px 0;"><strong>📊 平均分数:</strong></p>
        <ul style="margin: 5px 0; padding-left: 20px;">
            <li>整体分数: {avg_overall:.4f}</li>
            <li>CLIP分数: {avg_clip:.4f}</li>
            <li>融合分数: {avg_fused:.4f}</li>
        </ul>
        """
        
        # 添加细微差别分析
        if len(results) >= 2:
            differences = []
            for i in range(len(sorted_results) - 1):
                current = sorted_results[i]
                next_model = sorted_results[i + 1]
                diff = current.get('overall_score', 0) - next_model.get('overall_score', 0)
                if diff > 0.01:
                    differences.append(
                        f"<li>{current['model_name']} 比 {next_model['model_name']} 高 {diff:.4f} ({diff*100:.2f}%)</li>"
                    )
            
            if differences:
                summary += """
                <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 15px 0;">
                <p style="margin: 8px 0;"><strong>🔍 细微差别分析:</strong></p>
                <ul style="margin: 5px 0; padding-left: 20px;">
                """ + "".join(differences) + "</ul>"
        
        return summary

