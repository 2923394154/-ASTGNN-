import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class EnhancedFactorAnalyzer:
    """增强的因子分析器 - 包含RankIC、ICIR、分组回测等全面指标"""
    
    def __init__(self, factor_names: Optional[List[str]] = None):
        self.factor_names = factor_names or []
        self.results = {}
    
    def comprehensive_ic_analysis(self, 
                                 factors: torch.Tensor, 
                                 returns: torch.Tensor) -> Dict:
        """全面的IC分析：包含IC、RankIC、ICIR等
        
        参数：
        - factors: 因子矩阵 [time, stocks, factors]
        - returns: 收益率矩阵 [time, stocks]
        
        返回：
        - 全面的IC分析结果
        """
        logger.info("=== 增强IC分析（IC + RankIC + ICIR）===")
        
        if factors.dim() != 3 or returns.dim() != 2:
            raise ValueError("需要时间序列数据: factors [time, stocks, factors], returns [time, stocks]")
        
        num_periods, num_stocks, num_factors = factors.shape
        
        # 存储IC和RankIC序列
        ic_series = []       # Pearson IC
        rank_ic_series = []  # Spearman RankIC
        
        for t in range(num_periods):
            factor_t = factors[t].numpy()  # [stocks, factors]
            return_t = returns[t].numpy()   # [stocks]
            
            # 计算当期IC（Pearson相关系数）和RankIC（Spearman相关系数）
            period_ic = []
            period_rank_ic = []
            
            for f in range(num_factors):
                # 移除NaN值
                valid_mask = ~(np.isnan(factor_t[:, f]) | np.isnan(return_t))
                if valid_mask.sum() < 10:  # 至少需要10个有效样本
                    period_ic.append(np.nan)
                    period_rank_ic.append(np.nan)
                    continue
                
                factor_clean = factor_t[valid_mask, f]
                return_clean = return_t[valid_mask]
                
                # Pearson IC
                ic, _ = stats.pearsonr(factor_clean, return_clean)
                period_ic.append(ic)
                
                # Spearman RankIC  
                rank_ic, _ = stats.spearmanr(factor_clean, return_clean)
                period_rank_ic.append(rank_ic)
            
            ic_series.append(period_ic)
            rank_ic_series.append(period_rank_ic)
        
        ic_series = np.array(ic_series)         # [periods, factors]
        rank_ic_series = np.array(rank_ic_series)  # [periods, factors]
        
        # 计算统计量
        results = {}
        factor_names = self.factor_names if len(self.factor_names) == num_factors else [f'Factor_{i}' for i in range(num_factors)]
        
        for f in range(num_factors):
            factor_name = factor_names[f]
            
            # IC统计
            ic_values = ic_series[:, f]
            rank_ic_values = rank_ic_series[:, f]
            
            # 移除NaN值
            ic_clean = ic_values[~np.isnan(ic_values)]
            rank_ic_clean = rank_ic_values[~np.isnan(rank_ic_values)]
            
            if len(ic_clean) == 0 or len(rank_ic_clean) == 0:
                continue
            
            # IC分析
            ic_mean = np.mean(ic_clean)
            ic_std = np.std(ic_clean)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            ic_win_rate = np.mean(ic_clean > 0)
            
            # RankIC分析  
            rank_ic_mean = np.mean(rank_ic_clean)
            rank_ic_std = np.std(rank_ic_clean)
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
            rank_ic_win_rate = np.mean(rank_ic_clean > 0)
            
            # 统计显著性检验
            ic_t_stat, ic_p_value = stats.ttest_1samp(ic_clean, 0)
            rank_ic_t_stat, rank_ic_p_value = stats.ttest_1samp(rank_ic_clean, 0)
            
            # 累积IC
            ic_cumsum = np.cumsum(ic_clean)
            rank_ic_cumsum = np.cumsum(rank_ic_clean)
            
            # 稳定性分析
            ic_stability = float(self._calculate_rolling_stability(ic_clean, window=min(6, len(ic_clean)//2)))
            rank_ic_stability = float(self._calculate_rolling_stability(rank_ic_clean, window=min(6, len(rank_ic_clean)//2)))
            
            results[factor_name] = {
                # IC指标
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'ic_win_rate': ic_win_rate,
                'ic_t_stat': ic_t_stat,
                'ic_p_value': ic_p_value,
                'ic_significant': ic_p_value < 0.05,
                'ic_cumsum': ic_cumsum,
                'ic_stability': ic_stability,
                
                # RankIC指标
                'rank_ic_mean': rank_ic_mean,
                'rank_ic_std': rank_ic_std,
                'rank_ic_ir': rank_ic_ir,
                'rank_ic_win_rate': rank_ic_win_rate,
                'rank_ic_t_stat': rank_ic_t_stat,
                'rank_ic_p_value': rank_ic_p_value,
                'rank_ic_significant': rank_ic_p_value < 0.05,
                'rank_ic_cumsum': rank_ic_cumsum,
                'rank_ic_stability': rank_ic_stability,
                
                # 原始序列
                'ic_series': ic_clean,
                'rank_ic_series': rank_ic_clean,
                'num_periods': len(ic_clean)
            }
        
        # 打印详细结果
        self._print_comprehensive_results(results)
        
        return results
    
    def _calculate_rolling_stability(self, series: np.ndarray, window: int = 6) -> float:
        """计算滚动稳定性（滚动标准差的均值）"""
        if len(series) < window or window < 2:
            return np.std(series) if len(series) > 1 else 0.0
        
        rolling_stds = []
        for i in range(window, len(series) + 1):
            window_std = np.std(series[i-window:i])
            rolling_stds.append(window_std)
        
        return np.mean(rolling_stds)
    
    def _print_comprehensive_results(self, results: Dict):
        """打印全面的分析结果"""
        logger.info("\n=== 因子IC和RankIC详细分析 ===")
        
        # 表头
        header = f"{'Factor':15} {'IC':>8} {'IC_IR':>8} {'IC_Win':>8} {'IC_Sig':>8} {'RankIC':>8} {'RIC_IR':>8} {'RIC_Win':>8} {'RIC_Sig':>8} {'评级':>8}"
        logger.info(header)
        logger.info("=" * len(header))
        
        # 按IC_IR排序
        sorted_factors = sorted(results.items(), key=lambda x: abs(x[1]['ic_ir']), reverse=True)
        
        for factor_name, metrics in sorted_factors:
            # 因子评级
            rating = self._rate_factor(metrics)
            
            line = (f"{factor_name:15} "
                   f"{metrics['ic_mean']:8.4f} "
                   f"{metrics['ic_ir']:8.4f} "
                   f"{metrics['ic_win_rate']:8.2%} "
                   f"{'Yes' if metrics['ic_significant'] else 'No':>8} "
                   f"{metrics['rank_ic_mean']:8.4f} "
                   f"{metrics['rank_ic_ir']:8.4f} "
                   f"{metrics['rank_ic_win_rate']:8.2%} "
                   f"{'Yes' if metrics['rank_ic_significant'] else 'No':>8} "
                   f"{rating:>8}")
            logger.info(line)
        
        # 整体统计
        logger.info("\n=== 整体因子表现统计 ===")
        avg_ic = np.mean([r['ic_mean'] for r in results.values()])
        avg_rank_ic = np.mean([r['rank_ic_mean'] for r in results.values()])
        avg_ic_ir = np.mean([r['ic_ir'] for r in results.values()])
        avg_rank_ic_ir = np.mean([r['rank_ic_ir'] for r in results.values()])
        
        effective_factors_ic = sum(1 for r in results.values() if abs(r['ic_mean']) > 0.02)
        effective_factors_rank_ic = sum(1 for r in results.values() if abs(r['rank_ic_mean']) > 0.02)
        significant_factors_ic = sum(1 for r in results.values() if r['ic_significant'])
        significant_factors_rank_ic = sum(1 for r in results.values() if r['rank_ic_significant'])
        
        logger.info(f"平均IC: {avg_ic:.6f}")
        logger.info(f"平均RankIC: {avg_rank_ic:.6f}")
        logger.info(f"平均IC_IR: {avg_ic_ir:.6f}")
        logger.info(f"平均RankIC_IR: {avg_rank_ic_ir:.6f}")
        logger.info(f"有效因子数量 (|IC|>0.02): {effective_factors_ic}/{len(results)}")
        logger.info(f"有效因子数量 (|RankIC|>0.02): {effective_factors_rank_ic}/{len(results)}")
        logger.info(f"显著因子数量 (IC p<0.05): {significant_factors_ic}/{len(results)}")
        logger.info(f"显著因子数量 (RankIC p<0.05): {significant_factors_rank_ic}/{len(results)}")
    
    def _rate_factor(self, metrics: Dict) -> str:
        """因子评级"""
        ic_ir = abs(metrics['ic_ir'])
        rank_ic_ir = abs(metrics['rank_ic_ir'])
        ic_sig = metrics['ic_significant']
        rank_ic_sig = metrics['rank_ic_significant']
        
        avg_ir = (ic_ir + rank_ic_ir) / 2
        
        if avg_ir > 0.5 and (ic_sig or rank_ic_sig):
            return "A+"
        elif avg_ir > 0.3 and (ic_sig or rank_ic_sig):
            return "A"
        elif avg_ir > 0.2:
            return "B+"
        elif avg_ir > 0.1:
            return "B"
        elif avg_ir > 0.05:
            return "C"
        else:
            return "D"
    
    def factor_grouping_backtest(self, 
                                factors: torch.Tensor,
                                returns: torch.Tensor,
                                factor_idx: int = 0,
                                num_groups: int = 5) -> Dict:
        """因子分组回测"""
        logger.info(f"\n=== 因子分组回测 (Factor_{factor_idx}) ===")
        
        if factors.dim() != 3 or returns.dim() != 2:
            raise ValueError("需要时间序列数据")
        
        num_periods, num_stocks, num_factors = factors.shape
        
        # 存储各组收益
        group_returns = [[] for _ in range(num_groups)]
        long_short_returns = []
        
        for t in range(num_periods):
            factor_t = factors[t, :, factor_idx].numpy()
            return_t = returns[t].numpy()
            
            # 移除NaN值
            valid_mask = ~(np.isnan(factor_t) | np.isnan(return_t))
            if valid_mask.sum() < num_groups * 5:  # 每组至少5只股票
                continue
                
            factor_valid = factor_t[valid_mask]
            return_valid = return_t[valid_mask]
            
            # 按因子值分组
            sorted_idx = np.argsort(factor_valid)
            group_size = len(sorted_idx) // num_groups
            
            period_group_returns = []
            for g in range(num_groups):
                start_idx = g * group_size
                if g == num_groups - 1:
                    end_idx = len(sorted_idx)
                else:
                    end_idx = (g + 1) * group_size
                
                group_stocks = sorted_idx[start_idx:end_idx]
                group_ret = np.mean(return_valid[group_stocks])
                group_returns[g].append(group_ret)
                period_group_returns.append(group_ret)
            
            # 多空组合 (Top - Bottom)
            if len(period_group_returns) == num_groups:
                long_short_ret = period_group_returns[-1] - period_group_returns[0]
                long_short_returns.append(long_short_ret)
        
        # 计算组合统计
        group_stats = []
        for g in range(num_groups):
            if len(group_returns[g]) > 0:
                returns_g = np.array(group_returns[g])
                stats_g = {
                    'mean_return': np.mean(returns_g),
                    'std_return': np.std(returns_g),
                    'sharpe_ratio': np.mean(returns_g) / np.std(returns_g) * np.sqrt(252) if np.std(returns_g) > 0 else 0,
                    'win_rate': np.mean(returns_g > 0),
                    'max_return': np.max(returns_g),
                    'min_return': np.min(returns_g),
                    'num_periods': len(returns_g)
                }
            else:
                stats_g = {'mean_return': np.nan, 'std_return': np.nan, 'sharpe_ratio': np.nan}
            group_stats.append(stats_g)
        
        # 多空组合统计
        if len(long_short_returns) > 0:
            ls_returns = np.array(long_short_returns)
            long_short_stats = {
                'mean_return': np.mean(ls_returns),
                'std_return': np.std(ls_returns),
                'sharpe_ratio': np.mean(ls_returns) / np.std(ls_returns) * np.sqrt(252) if np.std(ls_returns) > 0 else 0,
                'win_rate': np.mean(ls_returns > 0),
                't_stat': stats.ttest_1samp(ls_returns, 0)[0],
                'p_value': stats.ttest_1samp(ls_returns, 0)[1],
                'max_return': np.max(ls_returns),
                'min_return': np.min(ls_returns),
                'num_periods': len(ls_returns)
            }
        else:
            long_short_stats = {}
        
        # 打印结果
        logger.info(f"分组数量: {num_groups}, 有效期数: {len(long_short_returns)}")
        logger.info("\n各组表现 (从低到高分位):")
        for g in range(num_groups):
            stats_g = group_stats[g]
            logger.info(f"  Group {g+1}: 收益率={stats_g['mean_return']:7.4f}, "
                       f"夏普={stats_g['sharpe_ratio']:6.3f}, 胜率={stats_g['win_rate']:6.2%}")
        
        if long_short_stats:
            logger.info(f"\n多空组合表现:")
            logger.info(f"  收益率: {long_short_stats['mean_return']:7.4f}")
            logger.info(f"  夏普比率: {long_short_stats['sharpe_ratio']:6.3f}")
            logger.info(f"  胜率: {long_short_stats['win_rate']:6.2%}")
            logger.info(f"  t统计量: {long_short_stats['t_stat']:6.3f}")
            logger.info(f"  p值: {long_short_stats['p_value']:6.4f}")
            logger.info(f"  显著性: {'是' if long_short_stats['p_value'] < 0.05 else '否'}")
        
        return {
            'group_stats': group_stats,
            'long_short_stats': long_short_stats,
            'group_returns': group_returns,
            'long_short_returns': long_short_returns
        }
    
    def generate_optimization_suggestions(self, ic_results: Dict) -> List[str]:
        """基于IC分析结果生成优化建议"""
        suggestions = []
        
        # 统计有效因子
        effective_factors = [name for name, metrics in ic_results.items() 
                           if abs(metrics['ic_mean']) > 0.02 or abs(metrics['rank_ic_mean']) > 0.02]
        
        weak_factors = [name for name, metrics in ic_results.items() 
                       if abs(metrics['ic_mean']) < 0.01 and abs(metrics['rank_ic_mean']) < 0.01]
        
        unstable_factors = [name for name, metrics in ic_results.items() 
                          if metrics.get('ic_stability', 0) > 0.1 or metrics.get('rank_ic_stability', 0) > 0.1]
        
        # 生成建议
        suggestions.append("=== 模型优化建议 ===")
        suggestions.append(f"有效因子数量: {len(effective_factors)}/{len(ic_results)}")
        
        if len(effective_factors) < len(ic_results) * 0.6:
            suggestions.append("有效因子比例偏低，建议:")
            suggestions.append("   1. 增加训练轮数 (150 → 300)")
            suggestions.append("   2. 调整学习率 (0.002 → 0.001)")
            suggestions.append("   3. 增加因子输出数量 (8 → 16)")
            suggestions.append("   4. 调整模型架构 (增加GRU层数或GAT头数)")
        
        if weak_factors:
            suggestions.append(f"弱因子 ({len(weak_factors)}个): {', '.join(weak_factors[:3])}")
            suggestions.append("   建议: 使用因子选择或增加正则化")
        
        if unstable_factors:
            suggestions.append(f"不稳定因子 ({len(unstable_factors)}个): {', '.join(unstable_factors[:3])}")
            suggestions.append("   建议: 增加dropout或使用更强的正则化")
        
        # 数据质量建议
        avg_ic_ir = np.mean([abs(r['ic_ir']) for r in ic_results.values()])
        avg_rank_ic_ir = np.mean([abs(r['rank_ic_ir']) for r in ic_results.values()])
        
        if avg_ic_ir < 0.2 and avg_rank_ic_ir < 0.2:
            suggestions.append("整体IR偏低，建议:")
            suggestions.append("   1. 检查数据质量和标准化")
            suggestions.append("   2. 增加序列长度 (10 → 20)")
            suggestions.append("   3. 调整损失函数权重")
            suggestions.append("   4. 尝试不同的邻接矩阵构建方法")
        
        # 积极建议
        best_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]['ic_ir']), reverse=True)[:3]
        if best_factors:
            suggestions.append(f"表现最佳因子: {', '.join([f[0] for f in best_factors])}")
            suggestions.append("   建议: 分析这些因子的特征，应用到其他因子")
        
        return suggestions
    
    def plot_ic_analysis(self, ic_results: Dict, save_path: str = None):
        """绘制IC分析图表"""
        if not ic_results:
            logger.warning("没有IC结果可绘制")
            return
        
        # 确保目录存在
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Factor IC Analysis', fontsize=16, fontweight='bold')
        
        factors = list(ic_results.keys())
        ic_means = [ic_results[f]['ic_mean'] for f in factors]
        rank_ic_means = [ic_results[f]['rank_ic_mean'] for f in factors]
        ic_irs = [ic_results[f]['ic_ir'] for f in factors]
        rank_ic_irs = [ic_results[f]['rank_ic_ir'] for f in factors]
        
        # IC均值对比
        x = range(len(factors))
        axes[0,0].bar([i-0.2 for i in x], ic_means, 0.4, label='IC', alpha=0.7)
        axes[0,0].bar([i+0.2 for i in x], rank_ic_means, 0.4, label='RankIC', alpha=0.7)
        axes[0,0].set_title('IC vs RankIC Mean')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(factors, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # IR对比
        axes[0,1].bar([i-0.2 for i in x], ic_irs, 0.4, label='IC_IR', alpha=0.7)
        axes[0,1].bar([i+0.2 for i in x], rank_ic_irs, 0.4, label='RankIC_IR', alpha=0.7)
        axes[0,1].set_title('IC_IR vs RankIC_IR')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(factors, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 累积IC曲线 (选择前4个因子)
        for i, factor in enumerate(factors[:4]):
            ic_cumsum = ic_results[factor]['ic_cumsum']
            axes[1,0].plot(ic_cumsum, label=f'{factor} IC', alpha=0.8)
            
        axes[1,0].set_title('Cumulative IC')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Cumulative IC')
        
        # 累积RankIC曲线
        for i, factor in enumerate(factors[:4]):
            rank_ic_cumsum = ic_results[factor]['rank_ic_cumsum']
            axes[1,1].plot(rank_ic_cumsum, label=f'{factor} RankIC', alpha=0.8)
            
        axes[1,1].set_title('Cumulative RankIC')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Cumulative RankIC')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IC分析图表保存: {save_path}")
        
        plt.show()
    
    def comprehensive_factor_analysis(self, 
                                    factors: torch.Tensor, 
                                    returns: torch.Tensor,
                                    perform_backtest: bool = True,
                                    plot_results: bool = True) -> Dict:
        """综合因子分析 - 一站式分析"""
        logger.info("=== 启动综合因子分析 ===")
        
        results = {}
        
        # 1. IC分析
        ic_results = self.comprehensive_ic_analysis(factors, returns)
        results['ic_analysis'] = ic_results
        
        # 2. 分组回测 (选择最佳因子)
        if perform_backtest and ic_results:
            best_factor_idx = 0
            best_ic_ir = 0
            for i, (factor_name, metrics) in enumerate(ic_results.items()):
                if abs(metrics['ic_ir']) > best_ic_ir:
                    best_ic_ir = abs(metrics['ic_ir'])
                    best_factor_idx = i
            
            logger.info(f"选择因子 {best_factor_idx} 进行分组回测 (IC_IR={best_ic_ir:.4f})")
            backtest_results = self.factor_grouping_backtest(factors, returns, best_factor_idx)
            results['backtest'] = backtest_results
        
        # 3. 优化建议
        suggestions = self.generate_optimization_suggestions(ic_results)
        results['suggestions'] = suggestions
        for suggestion in suggestions:
            logger.info(suggestion)
        
        # 4. 绘制图表
        if plot_results and ic_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = f"training_results/enhanced_ic_analysis_{timestamp}.png"
            self.plot_ic_analysis(ic_results, chart_path)
            results['chart_path'] = chart_path
        
        return results


def analyze_astgnn_factors():
    """分析ASTGNN生成的因子"""
    try:
        # 加载因子分析数据
        data = np.load('factor_analysis_data.npz')
        factors = torch.from_numpy(data['factors'])  # [time, stocks, factors]
        targets = torch.from_numpy(data['targets'])  # [time, stocks]
        
        logger.info(f"加载因子数据: {factors.shape}, 目标数据: {targets.shape}")
        
        # 创建分析器
        analyzer = EnhancedFactorAnalyzer()
        
        # 综合分析
        results = analyzer.comprehensive_factor_analysis(factors, targets)
        
        return results
        
    except FileNotFoundError:
        logger.error("未找到factor_analysis_data.npz文件，请先运行训练生成因子数据")
        return None
    except Exception as e:
        logger.error(f"因子分析失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 确保结果目录存在
    os.makedirs("training_results", exist_ok=True)
    
    # 运行分析
    results = analyze_astgnn_factors()
    
    if results:
        logger.info("=== 因子分析完成 ===")
        ic_results = results.get('ic_analysis', {})
        if ic_results:
            logger.info(f"分析了 {len(ic_results)} 个因子")
            effective_factors = sum(1 for r in ic_results.values() if abs(r['ic_mean']) > 0.02)
            logger.info(f"有效因子数量: {effective_factors}")
    else:
        logger.error("因子分析失败") 