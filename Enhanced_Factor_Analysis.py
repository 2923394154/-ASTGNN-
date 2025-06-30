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
                if int(valid_mask.sum()) < 10:  # 至少需要10个有效样本
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
            if int(valid_mask.sum()) < num_groups * 5:  # 每组至少5只股票
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


class ProfessionalBacktestAnalyzer:
    """专业回撤分析器 - 严格按照量化投资标准"""
    
    def __init__(self, start_date='20231229', end_date='20240430', factor_names=None):
        """
        初始化专业回测分析器
        
        参数:
        - start_date: 回测开始日期 (YYYYMMDD格式)
        - end_date: 回测结束日期 (YYYYMMDD格式) 
        - factor_names: 因子名称列表
        """
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.factor_names = factor_names or []
        
        # 股票池配置 - 严格按照要求
        self.universe_configs = {
            'CSI_ALL': {'name': '中证全指', 'num_groups': 20},
            'HS300': {'name': '沪深300', 'num_groups': 5},
            'CSI500': {'name': '中证500', 'num_groups': 5},
            'CSI1000': {'name': '中证1000', 'num_groups': 5}
        }
        
        # 回测参数 - 严格按照专业要求
        self.return_window = 10        # T+1~T+11，未来10日收益率
        self.ic_calc_freq = 10         # 每隔10个交易日计算一次RankIC
        self.rebalance_freq = 5        # 周度调仓（每5个交易日）
        
        logger.info(f"专业回撤分析器初始化完成 - 回测期间: {start_date} 至 {end_date}")
        logger.info(f"RankIC计算频率: 每{self.ic_calc_freq}个交易日")
        logger.info(f"调仓频率: 每{self.rebalance_freq}个交易日（周度）")

    def calculate_rank_ic_with_future_returns(self, factors, prices):
        """
        计算专业RankIC - 严格按照要求：
        1. 当天因子与隔日未来十日收益率（T+1~T+11）序列计算
        2. 每隔十个交易日计算一次
        """
        logger.info("=== 开始计算专业RankIC（未来十日收益率）===")
        
        num_periods, num_stocks, num_factors = factors.shape
        
        # 计算未来十日收益率（T+1~T+11）
        future_returns = self._calculate_future_returns_t1_t11(prices)
        
        # 严格按照要求：每隔10个交易日计算一次RankIC
        rank_ic_series = []
        calculation_periods = []
        
        for t in range(0, num_periods - self.return_window, self.ic_calc_freq):
            if t + self.return_window >= num_periods:
                break
            
            # 当天因子（T日）
            factor_t = factors[t].numpy()  # [stocks, factors]
            
            # 隔日未来十日收益率（T+1~T+11）
            future_ret_t = future_returns[t].numpy()  # [stocks]
            
            # 计算当期RankIC (Spearman相关系数)
            period_rank_ic = []
            for f in range(num_factors):
                factor_clean = factor_t[:, f]
                return_clean = future_ret_t
                
                valid_mask = ~(np.isnan(factor_clean) | np.isnan(return_clean))
                if int(valid_mask.sum()) < 50:  # 至少需要50个有效样本
                    period_rank_ic.append(np.nan)
                    continue
                
                factor_valid = factor_clean[valid_mask]
                return_valid = return_clean[valid_mask]
                
                # Spearman RankIC
                try:
                    rank_ic, _ = stats.spearmanr(factor_valid, return_valid)
                    period_rank_ic.append(rank_ic if not np.isnan(rank_ic) else 0.0)
                except:
                    period_rank_ic.append(0.0)
            
            rank_ic_series.append(period_rank_ic)
            calculation_periods.append(t)
        
        if len(rank_ic_series) == 0:
            logger.warning("没有足够的数据进行RankIC计算")
            return {'results': {}}
        
        rank_ic_series = np.array(rank_ic_series)  # [calc_periods, factors]
        logger.info(f"RankIC计算完成: {len(calculation_periods)}个计算期间")
        
        # 计算专业RankIC统计量
        results = {}
        factor_names = self.factor_names if len(self.factor_names) == num_factors else [f'ASTGNN_Factor_{i}' for i in range(num_factors)]
        
        for f in range(num_factors):
            factor_name = factor_names[f]
            
            # RankIC序列
            rank_ic_values = rank_ic_series[:, f]
            rank_ic_clean = rank_ic_values[~np.isnan(rank_ic_values)]
            
            if len(rank_ic_clean) < 2:
                continue
            
            # 严格按照专业要求计算
            rank_ic_mean = np.mean(rank_ic_clean)              # RankIC均值
            rank_ic_std = np.std(rank_ic_clean, ddof=1)        # RankIC标准差
            rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0  # ICIR
            
            # 其他必要统计量
            rank_ic_win_rate = np.mean(rank_ic_clean > 0)
            rank_ic_t_stat, rank_ic_p_value = stats.ttest_1samp(rank_ic_clean, 0)
            
            results[factor_name] = {
                'rank_ic_mean': rank_ic_mean,
                'rank_ic_std': rank_ic_std,
                'rank_ic_ir': rank_ic_ir,  # ICIR = 均值/标准差
                'rank_ic_win_rate': rank_ic_win_rate,
                'rank_ic_significant': rank_ic_p_value < 0.05,
                'num_calculations': len(rank_ic_clean)
            }
        
        # 只打印核心结果
        self._print_professional_rank_ic_results(results)
        
        return {'results': results}

    def _calculate_future_returns_t1_t11(self, prices):
        """计算隔日未来十日收益率（T+1~T+11）"""
        num_periods, num_stocks = prices.shape
        future_returns = torch.zeros_like(prices)
        
        for t in range(num_periods - self.return_window - 1):
            # T+1日价格（隔日开盘）
            price_start = prices[t + 1]     
            # T+11日价格（未来10日后收盘）
            price_end = prices[t + 1 + self.return_window]  
            
            # 收益率 = (P_t+11 - P_t+1) / P_t+1
            returns = (price_end - price_start) / (price_start + 1e-8)
            future_returns[t] = returns
        
        return future_returns

    def group_backtest_analysis(self, factors, prices, universe='CSI_ALL', factor_idx=0):
        """专业分组回测分析 - 严格按照要求"""
        universe_name = self.universe_configs[universe]['name']
        num_groups = self.universe_configs[universe]['num_groups']
        
        logger.info(f"=== 开始分组回测分析（{universe}）===")
        
        # 使用原始10日收益率数据
        try:
            data = np.load('factor_analysis_data.npz')
            original_targets = torch.from_numpy(data['targets'])
            returns = original_targets  # 直接使用，无预处理
            logger.info(f"使用原始10日收益率数据: 均值={returns.mean():.6f}, 标准差={returns.std():.6f}")
        except:
            logger.warning("无法加载原始数据")
            return {}
        
        num_periods, num_stocks = factors.shape[:2]
        
        # 周度调仓的分组回测
        group_returns = [[] for _ in range(num_groups)]
        group_holdings = []
        long_short_returns = []
        top_group_returns = []  # 单独记录top组收益用于计算超额收益
        
        # 按调仓频率进行回测（每5个交易日调仓一次）
        for t in range(0, num_periods - self.rebalance_freq, self.rebalance_freq):
            if t + self.rebalance_freq >= num_periods:
                break
            
            # T日因子值用于分组
            factor_t = factors[t, :, factor_idx].numpy()
            
            # 过滤有效股票
            valid_mask = ~np.isnan(factor_t)
            if np.sum(valid_mask) < num_groups * 20:
                continue
            
            factor_valid = factor_t[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            # 因子分组（按分位数）
            factor_quantiles = np.quantile(factor_valid, np.linspace(0, 1, num_groups + 1))
            
            current_holdings = [[] for _ in range(num_groups)]
            
            # 计算调仓期间收益率（T+1日到T+rebalance_freq日）
            period_start = t + 1  # 次日开始持有
            period_end = t + self.rebalance_freq
            
            if period_end >= num_periods:
                continue
            
            for g in range(num_groups):
                if g == num_groups - 1:
                    group_mask = (factor_valid >= factor_quantiles[g])
                else:
                    group_mask = (factor_valid >= factor_quantiles[g]) & (factor_valid < factor_quantiles[g + 1])
                
                if np.sum(group_mask) > 0:
                    group_stocks = valid_indices[group_mask]
                    current_holdings[g] = group_stocks.tolist()
                    
                    # 计算该组在调仓期间的收益率
                    group_returns_period = []
                    for stock_idx in group_stocks:
                        stock_return = returns[period_start:period_end, stock_idx].mean().item()
                        group_returns_period.append(stock_return)
                    
                    group_ret = np.mean(group_returns_period) if group_returns_period else 0.0
                    group_returns[g].append(group_ret)
                    
                    # 记录top组收益
                    if g == num_groups - 1:  # top组
                        top_group_returns.append(group_ret)
                else:
                    group_returns[g].append(0.0)
                    if g == num_groups - 1:
                        top_group_returns.append(0.0)
                    current_holdings[g] = []
            
            group_holdings.append(current_holdings)
            
            # 多空对冲收益（top - bottom）
            if len(group_returns[-1]) > 0 and len(group_returns[0]) > 0:
                long_ret = group_returns[-1][-1]   # top组
                short_ret = group_returns[0][-1]   # bottom组
                long_short_returns.append(long_ret - short_ret)
        
        # 计算专业回测指标
        if len(long_short_returns) == 0:
            logger.warning("没有足够的数据进行回测")
            return {}
        
        # 多空对冲组合统计
        ls_returns = np.array(long_short_returns)
        mean_ls_return = np.mean(ls_returns)
        
        # 年化计算（基于调仓频率）
        periods_per_year = 252 / self.rebalance_freq  # 每年调仓次数
        
        if abs(mean_ls_return) < 0.5:
            ls_annual_return = (1 + mean_ls_return) ** periods_per_year - 1
        else:
            ls_annual_return = mean_ls_return * periods_per_year
        
        ls_annual_volatility = np.std(ls_returns, ddof=1) * np.sqrt(periods_per_year)
        information_ratio = ls_annual_return / ls_annual_volatility if ls_annual_volatility > 0 else 0
        
        # top组超额收益的最大回撤和年化波动率
        top_returns = np.array(top_group_returns)
        if len(top_returns) > 0:
            # 计算超额收益（top组相对于市场平均的超额）
            market_return = np.mean([np.mean(group_returns[g]) for g in range(num_groups) if len(group_returns[g]) > 0])
            excess_returns = top_returns - market_return
            
            # 基于超额收益计算回撤和波动率
            excess_cumulative = np.cumprod(1 + excess_returns)
            max_drawdown_excess = self._calculate_max_drawdown(excess_cumulative)
            annual_volatility_excess = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
        else:
            max_drawdown_excess = 0.0
            annual_volatility_excess = 0.0
        
        # 多空组合回撤
        ls_cumulative = np.cumprod(1 + ls_returns)
        ls_max_drawdown = self._calculate_max_drawdown(ls_cumulative)
        
        # 统计显著性
        t_stat, p_value = stats.ttest_1samp(ls_returns, 0)
        
        # 计算周均单边换手率（基于多头组持仓）
        turnover_rate = self._calculate_professional_turnover_rate(group_holdings, top_group_idx=num_groups-1)
        
        # 专业回测结果
        professional_results = {
            'universe': universe_name,
            'num_groups': num_groups,
            'long_short_annual_return': ls_annual_return,
            'information_ratio': information_ratio,
            'max_drawdown': ls_max_drawdown,
            'win_rate': np.mean(ls_returns > 0),
            'weekly_turnover_rate': turnover_rate['weekly_turnover_pct'],
            'top_group_excess_volatility': annual_volatility_excess,
            'top_group_excess_max_drawdown': max_drawdown_excess,
            'statistical_significant': p_value < 0.05,
            'p_value': p_value,
            'num_periods': len(ls_returns)
        }
        
        # 只打印核心专业指标
        self._print_professional_backtest_results(professional_results)
        
        return professional_results

    def _calculate_professional_turnover_rate(self, group_holdings, top_group_idx):
        """计算周均单边换手率（基于多头组持仓）"""
        if len(group_holdings) < 2:
            return {'weekly_turnover_pct': 0.0, 'num_periods': 0}
        
        turnover_rates = []
        
        for i in range(1, len(group_holdings)):
            if (top_group_idx >= len(group_holdings[i]) or 
                top_group_idx >= len(group_holdings[i-1])):
                continue
                
            prev_holdings = set(group_holdings[i-1][top_group_idx])
            curr_holdings = set(group_holdings[i][top_group_idx])
            
            # 周均单边换手率 = 新增股票数量 / 总持仓数量
            new_stocks = curr_holdings - prev_holdings
            total_stocks = len(curr_holdings)
            
            if total_stocks > 0:
                turnover_rate = len(new_stocks) / total_stocks
                turnover_rates.append(turnover_rate)
        
        weekly_turnover = np.mean(turnover_rates) if turnover_rates else 0.0
        
        return {
            'weekly_turnover_pct': weekly_turnover * 100,
            'num_periods': len(turnover_rates)
        }

    def _print_professional_rank_ic_results(self, results):
        """打印专业RankIC分析结果（精简版）"""
        logger.info("\n=== 专业RankIC分析结果 ===")
        
        header = f"{'因子名称':15} {'RankIC均值':>12} {'RankIC标准差':>12} {'ICIR':>8} {'胜率':>8} {'显著性':>8} {'计算次数':>8}"
        logger.info(header)
        logger.info("=" * len(header))
        
        # 按ICIR绝对值排序
        sorted_factors = sorted(results.items(), 
                              key=lambda x: abs(x[1]['rank_ic_ir']), 
                              reverse=True)
        
        for factor_name, metrics in sorted_factors:
            line = (f"{factor_name:15} "
                   f"{metrics['rank_ic_mean']:12.6f} "
                   f"{metrics['rank_ic_std']:12.6f} "
                   f"{metrics['rank_ic_ir']:8.4f} "
                   f"{metrics['rank_ic_win_rate']:8.2%} "
                   f"{'是' if metrics['rank_ic_significant'] else '否':>8} "
                   f"{metrics['num_calculations']:8d}")
            logger.info(line)

    def _print_professional_backtest_results(self, results):
        """打印专业回测结果（只显示核心指标）"""
        logger.info(f"\n=== {results['universe']} 专业回测结果 ===")
        logger.info(f"分组数量: {results['num_groups']}组")
        logger.info(f"")
        logger.info(f"核心指标:")
        logger.info(f"  年化收益率: {results['long_short_annual_return']:8.2%}")
        logger.info(f"  信息比率: {results['information_ratio']:8.4f}")
        logger.info(f"  最大回撤: {results['max_drawdown']:8.2%}")
        logger.info(f"  胜率: {results['win_rate']:8.2%}")
        logger.info(f"  周均单边换手率: {results['weekly_turnover_rate']:.2f}%")
        logger.info(f"  统计显著性: {'是' if results['statistical_significant'] else '否'} (p={results['p_value']:.4f})")
        logger.info(f"")

    def _calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd


def analyze_astgnn_factors():
    """分析ASTGNN生成的因子 - 包含传统分析和专业回测"""
    logger.info("=== 开始分析ASTGNN因子 ===")
    
    try:
        # 加载因子分析数据
        data = np.load('factor_analysis_data.npz')
        factors = torch.from_numpy(data['factors'])  # [time, stocks, factors]
        targets = torch.from_numpy(data['targets'])  # [time, stocks]
        
        logger.info(f"加载因子数据: {factors.shape}, 目标数据: {targets.shape}")
        
        # 创建传统分析器
        traditional_analyzer = EnhancedFactorAnalyzer(
            factor_names=[f'ASTGNN_Factor_{i}' for i in range(factors.shape[2])]
        )
        
        # 传统综合分析
        traditional_results = traditional_analyzer.comprehensive_factor_analysis(factors, targets)
        
        # 创建专业回测分析器
        professional_analyzer = ProfessionalBacktestAnalyzer(
            start_date='20231229',
            end_date='20240430',
            factor_names=[f'ASTGNN_Factor_{i}' for i in range(factors.shape[2])]
        )
        
        # 修复：正确处理收益率数据，不需要重建价格
        # targets本身就是收益率，直接使用
        logger.info("=== 开始专业回测分析 ===")
        logger.info(f"专业回测数据: factors {factors.shape}, targets {targets.shape}")
        
        # 直接使用原始收益率数据，不进行预处理
        processed_targets = targets
        
        # 方案1：如果必须有价格数据，从预处理后的收益率正确重建
        initial_price = 100.0
        prices = torch.zeros_like(processed_targets)
        prices[0] = initial_price
        
        for t in range(1, len(processed_targets)):
            # 使用预处理后的小数形式收益率
            prices[t] = prices[t-1] * (1 + processed_targets[t-1])
        
        # 运行专业RankIC分析
        rank_ic_results = professional_analyzer.calculate_rank_ic_with_future_returns(factors, prices)
        
        # 运行专业分组回测
        backtest_results = {}
        for universe in ['CSI_ALL', 'HS300']:
            backtest_results[universe] = professional_analyzer.group_backtest_analysis(
                factors, prices, 
                universe=universe, 
                factor_idx=0
            )
        
        return {
            'traditional_analysis': traditional_results,
            'professional_rank_ic': rank_ic_results,
            'professional_backtest': backtest_results
        }
        
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