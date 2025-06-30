#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单因子ASTGNN模型评价脚本
包含IC分析、因子分布、预测能力、专业回测等全面评价指标
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 导入ASTGNN相关模块
from ASTGNN import ASTGNNFactorModel
from Enhanced_Factor_Analysis import ProfessionalBacktestAnalyzer
from FactorValidation import FactorValidationFramework

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class SingleFactorEvaluator:
    """单因子ASTGNN模型评价器"""
    
    def __init__(self, factor_data_file='factor_analysis_data.npz', model_file='gpu_astgnn_best_model.pth'):
        self.factor_data_file = factor_data_file
        self.model_file = model_file
        
        # 加载数据
        self.load_factor_data()
        
        # 初始化评价框架
        self.validator = FactorValidationFramework()
        self.professional_analyzer = ProfessionalBacktestAnalyzer(
            start_date='20231229',
            end_date='20240430',
            factor_names=['ASTGNN_Factor']
        )
        
        print("[完成] 单因子评价器初始化完成")
    
    def load_factor_data(self):
        """加载因子分析数据"""
        try:
            data = np.load(self.factor_data_file)
            self.factors = data['factors']  # [time, stocks, factors]
            self.targets = data['targets']  # [time, stocks]
            
            print(f"[成功] 数据加载成功:")
            print(f"  因子数据形状: {self.factors.shape}")
            print(f"  目标数据形状: {self.targets.shape}")
            print(f"  时间步数: {self.factors.shape[0]}")
            print(f"  股票数量: {self.factors.shape[1]}")
            print(f"  因子数量: {self.factors.shape[2]}")
            
        except Exception as e:
            print(f"[错误] 数据加载失败: {e}")
            raise
    
    def basic_factor_statistics(self):
        """基础因子统计分析"""
        print("\n" + "="*60)
        print("1. 基础因子统计分析")
        print("="*60)
        
        # 单因子分析（假设是第一个因子或单因子）
        if self.factors.shape[2] == 1:
            factor_values = self.factors[:, :, 0]  # [time, stocks]
            factor_name = "ASTGNN_Factor"
        else:
            factor_values = self.factors[:, :, 0]  # 选择第一个因子进行分析
            factor_name = "ASTGNN_Factor_0"
        
        # 展平为1D数组进行统计
        factor_flat = factor_values.flatten()
        target_flat = self.targets.flatten()
        
        # 移除NaN值
        mask = ~(np.isnan(factor_flat) | np.isnan(target_flat))
        factor_clean = factor_flat[mask]
        target_clean = target_flat[mask]
        
        print(f"因子名称: {factor_name}")
        print(f"有效样本数: {len(factor_clean):,}")
        print(f"数据覆盖率: {len(factor_clean)/len(factor_flat)*100:.2f}%")
        print()
        
        # 因子分布统计
        print("因子分布统计:")
        print(f"  均值: {factor_clean.mean():.6f}")
        print(f"  标准差: {factor_clean.std():.6f}")
        print(f"  中位数: {np.median(factor_clean):.6f}")
        print(f"  偏度: {stats.skew(factor_clean):.4f}")
        print(f"  峰度: {stats.kurtosis(factor_clean):.4f}")
        print(f"  最小值: {factor_clean.min():.6f}")
        print(f"  最大值: {factor_clean.max():.6f}")
        print(f"  分位数 [25%, 50%, 75%]: {np.percentile(factor_clean, [25, 50, 75])}")
        print()
        
        # 目标分布统计
        print("收益率分布统计:")
        print(f"  均值: {target_clean.mean():.6f}")
        print(f"  标准差: {target_clean.std():.6f}")
        print(f"  年化夏普比率: {target_clean.mean()/target_clean.std()*np.sqrt(252):.4f}")
        print()
        
        return factor_clean, target_clean, factor_name
    
    def correlation_analysis(self, factor_clean, target_clean):
        """相关性分析"""
        print("\n" + "="*60)
        print("2. 因子-收益率相关性分析")
        print("="*60)
        
        # 皮尔逊相关性
        pearson_corr, pearson_pval = stats.pearsonr(factor_clean, target_clean)
        
        # 斯皮尔曼相关性 (等级相关性)
        spearman_corr, spearman_pval = stats.spearmanr(factor_clean, target_clean)
        
        # Kendall's Tau
        kendall_corr, kendall_pval = stats.kendalltau(factor_clean, target_clean)
        
        print(f"皮尔逊相关系数: {pearson_corr:.6f} (p-value: {pearson_pval:.2e})")
        print(f"斯皮尔曼相关系数: {spearman_corr:.6f} (p-value: {spearman_pval:.2e})")
        print(f"Kendall Tau: {kendall_corr:.6f} (p-value: {kendall_pval:.2e})")
        
        # 相关性显著性判断
        alpha = 0.05
        print(f"\n显著性检验 (α = {alpha}):")
        print(f"  皮尔逊相关: {'显著' if pearson_pval < alpha else '不显著'}")
        print(f"  斯皮尔曼相关: {'显著' if spearman_pval < alpha else '不显著'}")
        print(f"  Kendall Tau: {'显著' if kendall_pval < alpha else '不显著'}")
        
        return {
            'pearson': (pearson_corr, pearson_pval),
            'spearman': (spearman_corr, spearman_pval),
            'kendall': (kendall_corr, kendall_pval)
        }
    
    def ic_analysis(self):
        """IC分析（信息系数）"""
        print("\n" + "="*60)
        print("3. IC分析（信息系数）")
        print("="*60)
        
        try:
            # 计算标准RankIC: 当天因子与未来10日累积收益的斯皮尔曼相关系数
            rank_ic_results = self.calculate_professional_rank_ic()
            
            # 同时保留原有的IC计算作为对比
            legacy_ic_results = self.validator.compute_information_coefficient(
                torch.from_numpy(self.factors),  # [time, stocks, factors]
                torch.from_numpy(self.targets)   # [time, stocks]
            )
            
            factor_name = 'ASTGNN_Factor' if self.factors.shape[2] == 1 else 'ASTGNN_Factor_0'
            
            print(f"因子: {factor_name}")
            print(f"\n=== 专业RankIC分析（T+1~T+11日收益，每10日采样）===")
            print(f"  RankIC均值: {rank_ic_results['rank_ic_mean']:.6f}")
            print(f"  RankIC标准差: {rank_ic_results['rank_ic_std']:.6f}")
            print(f"  RankIC_IR: {rank_ic_results['rank_ic_ir']:.6f}")
            print(f"  RankIC胜率: {rank_ic_results['rank_ic_win_rate']:.4f}")
            print(f"  有效期数: {rank_ic_results['effective_periods']}")
            print(f"  采样间隔: 每{rank_ic_results['sampling_interval']}个交易日")
            
            print(f"\n=== 传统IC分析（当期对比）===")
            print(f"  IC均值: {legacy_ic_results['ic_mean'][0]:.6f}")
            print(f"  IC标准差: {legacy_ic_results['ic_std'][0]:.6f}")
            print(f"  IC_IR: {legacy_ic_results['ic_ir'][0]:.6f}")
            print(f"  IC胜率: {legacy_ic_results['ic_win_rate'][0]:.4f}")
            
            # RankIC系列分析
            rank_ic_series = rank_ic_results['rank_ic_series']
            if len(rank_ic_series) > 0:
                print(f"\nRankIC时间序列统计:")
                print(f"  RankIC最大值: {np.max(rank_ic_series):.6f}")
                print(f"  RankIC最小值: {np.min(rank_ic_series):.6f}")
                print(f"  RankIC正值比例: {(np.array(rank_ic_series) > 0).mean():.4f}")
                print(f"  |RankIC| > 0.02 比例: {(np.abs(rank_ic_series) > 0.02).mean():.4f}")
                print(f"  |RankIC| > 0.05 比例: {(np.abs(rank_ic_series) > 0.05).mean():.4f}")
            
            # 合并结果，以RankIC为主
            combined_results = {
                'ic_mean': [rank_ic_results['rank_ic_mean']],
                'ic_std': [rank_ic_results['rank_ic_std']],
                'ic_ir': [rank_ic_results['rank_ic_ir']],
                'ic_win_rate': [rank_ic_results['rank_ic_win_rate']],
                'ic_series': [rank_ic_series],
                'effective_periods': rank_ic_results['effective_periods'],
                'rank_ic_results': rank_ic_results,
                'legacy_ic_results': legacy_ic_results
            }
            
            return combined_results
            
        except Exception as e:
            print(f"[错误] IC分析失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def calculate_professional_rank_ic(self, future_days=10, sampling_interval=10):
        """
        计算专业的RankIC指标
        RankIC = 当天因子与未来10日累积收益的斯皮尔曼相关系数
        每隔10个交易日计算一次
        
        参数:
        - future_days: 未来收益计算天数 (默认10天)
        - sampling_interval: 采样间隔 (默认10天)
        
        返回:
        - rank_ic_results: RankIC分析结果
        """
        print(f"计算专业RankIC: 未来{future_days}日收益，每{sampling_interval}日采样")
        
        time_steps, num_stocks, num_factors = self.factors.shape
        
        # 确保使用第一个因子（单因子分析）
        factor_data = self.factors[:, :, 0]  # [time, stocks]
        return_data = self.targets  # [time, stocks]
        
        rank_ic_series = []
        sampling_dates = []
        
        # 每隔sampling_interval天采样一次
        for t in range(0, time_steps - future_days, sampling_interval):
            try:
                # 当期因子值
                current_factors = factor_data[t, :]  # [stocks]
                
                # 计算未来10日累积收益 (T+1 到 T+future_days)
                future_returns = np.zeros(num_stocks)
                
                # 累积未来10日的日收益率
                for d in range(1, future_days + 1):
                    if t + d < time_steps:
                        future_returns += return_data[t + d, :]  # 累积收益
                
                # 移除NaN值
                valid_mask = ~(np.isnan(current_factors) | np.isnan(future_returns))
                if valid_mask.sum() < 10:  # 至少需要10只股票
                    continue
                
                valid_factors = current_factors[valid_mask]
                valid_returns = future_returns[valid_mask]
                
                # 计算斯皮尔曼相关系数 (RankIC)
                from scipy.stats import spearmanr
                rank_ic, p_value = spearmanr(valid_factors, valid_returns)
                
                if not np.isnan(rank_ic):
                    rank_ic_series.append(rank_ic)
                    sampling_dates.append(t)
                    
                print(f"  第{t}期: RankIC={rank_ic:.6f}, 有效股票数={valid_mask.sum()}, 未来{future_days}日累积收益范围=[{valid_returns.min():.6f}, {valid_returns.max():.6f}]")
                
            except Exception as e:
                print(f"  第{t}期计算失败: {e}")
                continue
        
        # 计算RankIC统计量
        if len(rank_ic_series) == 0:
            print("警告: 没有有效的RankIC计算结果")
            return {
                'rank_ic_mean': 0.0,
                'rank_ic_std': 0.0,
                'rank_ic_ir': 0.0,
                'rank_ic_win_rate': 0.0,
                'rank_ic_series': [],
                'effective_periods': 0,
                'sampling_interval': sampling_interval,
                'future_days': future_days
            }
        
        rank_ic_array = np.array(rank_ic_series)
        
        rank_ic_mean = np.mean(rank_ic_array)
        rank_ic_std = np.std(rank_ic_array)
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0
        rank_ic_win_rate = np.mean(rank_ic_array > 0)
        
        results = {
            'rank_ic_mean': rank_ic_mean,
            'rank_ic_std': rank_ic_std,
            'rank_ic_ir': rank_ic_ir,
            'rank_ic_win_rate': rank_ic_win_rate,
            'rank_ic_series': rank_ic_series,
            'effective_periods': len(rank_ic_series),
            'sampling_interval': sampling_interval,
            'future_days': future_days,
            'sampling_dates': sampling_dates
        }
        
        print(f"RankIC计算完成: 共{len(rank_ic_series)}期有效数据")
        return results
    
    def prediction_performance(self, factor_clean, target_clean):
        """预测性能评估"""
        print("\n" + "="*60)
        print("4. 预测性能评估")
        print("="*60)
        
        # 预测性能指标
        mse = mean_squared_error(target_clean, factor_clean)
        mae = mean_absolute_error(target_clean, factor_clean)
        rmse = np.sqrt(mse)
        
        # R²分数
        ss_res = np.sum((target_clean - factor_clean) ** 2)
        ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        print(f"预测性能指标:")
        print(f"  MSE (均方误差): {mse:.8f}")
        print(f"  MAE (平均绝对误差): {mae:.6f}")
        print(f"  RMSE (均方根误差): {rmse:.6f}")
        print(f"  R² 分数: {r2_score:.6f}")
        
        # 方向预测准确性
        factor_direction = np.sign(factor_clean)
        target_direction = np.sign(target_clean)
        direction_accuracy = (factor_direction == target_direction).mean()
        
        print(f"  方向预测准确率: {direction_accuracy:.4f}")
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2_score,
            'direction_accuracy': direction_accuracy
        }
    
    def quintile_analysis(self, factor_clean, target_clean):
        """分位数分析"""
        print("\n" + "="*60)
        print("5. 因子分位数分析")
        print("="*60)
        
        # 将数据按因子值分为5组
        df = pd.DataFrame({
            'factor': factor_clean,
            'return': target_clean
        })
        
        # 计算分位数标签
        df['quintile'] = pd.qcut(df['factor'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # 分组统计
        quintile_stats = df.groupby('quintile')['return'].agg([
            'mean', 'std', 'count'
        ]).round(6)
        
        print("分位数分组收益率统计:")
        print(quintile_stats)
        
        # 计算多空收益
        q5_return = quintile_stats.loc['Q5', 'mean']
        q1_return = quintile_stats.loc['Q1', 'mean']
        long_short_return = q5_return - q1_return
        
        print(f"\n多空策略:")
        print(f"  Q5组收益率: {q5_return:.6f}")
        print(f"  Q1组收益率: {q1_return:.6f}")
        print(f"  多空收益差: {long_short_return:.6f}")
        
        return quintile_stats, long_short_return
    
    def visualize_results(self, factor_clean, target_clean, factor_name, correlations, ic_results=None):
        """可视化分析结果"""
        print("\n" + "="*60)
        print("6. 生成可视化分析图表")
        print("="*60)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 因子分布直方图
        plt.subplot(2, 3, 1)
        plt.hist(factor_clean, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{factor_name} Distribution')
        plt.xlabel('Factor Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. 因子-收益率散点图
        plt.subplot(2, 3, 2)
        # 采样以避免过密
        sample_size = min(10000, len(factor_clean))
        indices = np.random.choice(len(factor_clean), sample_size, replace=False)
        plt.scatter(factor_clean[indices], target_clean[indices], alpha=0.6, s=1)
        plt.xlabel('Factor Value')
        plt.ylabel('Return')
        plt.title(f'Factor-Return Relationship (r={correlations["pearson"][0]:.4f})')
        
        # 添加回归线
        z = np.polyfit(factor_clean[indices], target_clean[indices], 1)
        p = np.poly1d(z)
        x_line = np.linspace(factor_clean[indices].min(), factor_clean[indices].max(), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # 3. 收益率分布
        plt.subplot(2, 3, 3)
        plt.hist(target_clean, bins=50, alpha=0.7, edgecolor='black', color='green')
        plt.title('Return Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 4. 相关性比较
        plt.subplot(2, 3, 4)
        corr_names = ['Pearson', 'Spearman', 'Kendall']
        corr_values = [correlations['pearson'][0], correlations['spearman'][0], correlations['kendall'][0]]
        colors = ['blue', 'orange', 'green']
        
        bars = plt.bar(corr_names, corr_values, color=colors, alpha=0.7)
        plt.title('Correlation Coefficients Comparison')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, corr_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 5. IC时间序列（如果有的话）
        plt.subplot(2, 3, 5)
        if ic_results:
            ic_series = np.array(ic_results['ic_series'][0])
            plt.plot(ic_series, alpha=0.8)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=ic_results['ic_mean'][0], color='green', linestyle='--', alpha=0.7, 
                       label=f'IC Mean: {ic_results["ic_mean"][0]:.4f}')
            plt.title('IC Time Series')
            plt.xlabel('Time Period')
            plt.ylabel('IC Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'IC Analysis Unavailable', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('IC Time Series')
        
        # 6. 因子分位数收益率
        plt.subplot(2, 3, 6)
        # 简化的分位数分析
        df = pd.DataFrame({'factor': factor_clean, 'return': target_clean})
        df['quintile'] = pd.qcut(df['factor'], 5, labels=[1, 2, 3, 4, 5])
        quintile_returns = df.groupby('quintile')['return'].mean()
        
        bars = plt.bar(range(1, 6), quintile_returns.values, alpha=0.7, color='purple')
        plt.title('Quintile Portfolio Returns')
        plt.xlabel('Quintile Group (1=Lowest, 5=Highest)')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{quintile_returns.iloc[i]:.5f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        import os
        os.makedirs('training_results', exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'training_results/single_factor_evaluation_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[完成] 可视化图表已保存: {save_path}")
        
        plt.show()
        
        return save_path
    
    def comprehensive_evaluation(self):
        """综合评价报告"""
        print("\n" + "[开始] 单因子ASTGNN模型综合评价")
        print("="*80)
        
        # 1. 基础统计
        factor_clean, target_clean, factor_name = self.basic_factor_statistics()
        
        # 2. 相关性分析
        correlations = self.correlation_analysis(factor_clean, target_clean)
        
        # 3. IC分析
        ic_results = self.ic_analysis()
        
        # 4. 预测性能
        performance = self.prediction_performance(factor_clean, target_clean)
        
        # 5. 分位数分析
        quintile_stats, long_short_return = self.quintile_analysis(factor_clean, target_clean)
        
        # 6. 可视化
        chart_path = self.visualize_results(factor_clean, target_clean, factor_name, correlations, ic_results)
        
        # 7. 综合评分
        self.generate_factor_score(correlations, ic_results, performance, long_short_return)
        
        print("\n" + "[完成] 单因子评价完成！")
        return {
            'correlations': correlations,
            'ic_results': ic_results,
            'performance': performance,
            'quintile_stats': quintile_stats,
            'chart_path': chart_path
        }
    
    def generate_factor_score(self, correlations, ic_results, performance, long_short_return):
        """生成因子综合评分"""
        print("\n" + "="*60)
        print("7. 因子综合评分")
        print("="*60)
        
        scores = {}
        
        # 相关性得分 (0-20分)
        abs_corr = abs(correlations['spearman'][0])
        if abs_corr > 0.1:
            corr_score = 20
        elif abs_corr > 0.05:
            corr_score = 15
        elif abs_corr > 0.02:
            corr_score = 10
        else:
            corr_score = 5
        scores['相关性'] = corr_score
        
        # IC得分 (0-25分)
        if ic_results:
            ic_ir = ic_results['ic_ir'][0]
            if ic_ir > 1.0:
                ic_score = 25
            elif ic_ir > 0.5:
                ic_score = 20
            elif ic_ir > 0.2:
                ic_score = 15
            else:
                ic_score = 10
        else:
            ic_score = 5
        scores['IC表现'] = ic_score
        
        # 预测性能得分 (0-25分)
        r2 = performance['r2']
        direction_acc = performance['direction_accuracy']
        if r2 > 0.1 and direction_acc > 0.55:
            pred_score = 25
        elif r2 > 0.05 and direction_acc > 0.52:
            pred_score = 20
        elif r2 > 0.01 and direction_acc > 0.5:
            pred_score = 15
        else:
            pred_score = 10
        scores['预测能力'] = pred_score
        
        # 多空收益得分 (0-30分)
        abs_ls_return = abs(long_short_return)
        if abs_ls_return > 0.01:
            ls_score = 30
        elif abs_ls_return > 0.005:
            ls_score = 25
        elif abs_ls_return > 0.002:
            ls_score = 20
        else:
            ls_score = 10
        scores['多空收益'] = ls_score
        
        # 计算总分
        total_score = sum(scores.values())
        
        print("因子评分详情:")
        for category, score in scores.items():
            print(f"  {category}: {score}分")
        
        print(f"\n总分: {total_score}/100分")
        
        # 等级评定
        if total_score >= 80:
            grade = "优秀 (A)"
        elif total_score >= 65:
            grade = "良好 (B)"
        elif total_score >= 50:
            grade = "一般 (C)"
        else:
            grade = "需改进 (D)"
        
        print(f"因子等级: {grade}")
        
        return scores, total_score, grade


def main():
    """主函数"""
    print("[分析] 单因子ASTGNN模型评价分析")
    print("="*60)
    
    # 创建评价器
    evaluator = SingleFactorEvaluator()
    
    # 执行综合评价
    results = evaluator.comprehensive_evaluation()
    
    print(f"\n[总结] 评价结果总结:")
    print(f"  相关性: {results['correlations']['spearman'][0]:.4f}")
    if results['ic_results']:
        print(f"  IC_IR: {results['ic_results']['ic_ir'][0]:.4f}")
    print(f"  R²分数: {results['performance']['r2']:.4f}")
    print(f"  方向准确率: {results['performance']['direction_accuracy']:.4f}")


if __name__ == "__main__":
    main() 