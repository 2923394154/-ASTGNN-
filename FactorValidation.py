import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体"""
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 如果是macOS系统，优先使用系统字体
    import platform
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Helvetica'] + plt.rcParams['font.sans-serif']
    elif platform.system() == 'Linux':  # Linux
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
    
    print(f"matplotlib字体设置完成：{plt.rcParams['font.sans-serif'][0]}")

# 在文件开头调用字体设置
setup_chinese_font()

# 导入已有模块
from ASTGNN import ASTGNNFactorModel
from ASTGNN_Loss import ASTGNNFactorLoss
from BarraFactorSimulator import BarraFactorSimulator


class FactorValidationFramework:
    """因子有效性检测框架
    
    提供全面的因子评估指标：
    1. IC分析（信息系数）
    2. 分组回测
    3. 风险调整收益
    4. 稳定性测试
    5. 统计显著性
    """
    
    def __init__(self, factor_names: List[str] = None):
        """
        参数：
        - factor_names: 因子名称列表
        """
        self.factor_names = factor_names or []
        self.results = {}
    
    def compute_information_coefficient(self, 
                                     factors: torch.Tensor, 
                                     returns: torch.Tensor,
                                     method='pearson') -> Dict:
        """计算信息系数(IC)
        
        参数：
        - factors: 因子矩阵 [time, stocks, factors] 或 [stocks, factors]
        - returns: 收益率矩阵 [time, stocks] 或 [stocks]
        - method: 相关系数计算方法 ('pearson', 'spearman')
        
        返回：
        - ic_results: IC分析结果字典
        """
        print("=== 信息系数(IC)分析 ===")
        
        if factors.dim() == 3 and returns.dim() == 2:
            # 时间序列数据
            num_periods, num_stocks, num_factors = factors.shape
            ic_series = []
            
            for t in range(num_periods - 1):  # 避免最后一期没有未来收益
                factor_t = factors[t].numpy()  # [stocks, factors]
                return_t1 = returns[t + 1].numpy()  # [stocks] 下期收益
                
                # 计算当期因子与下期收益的相关性
                period_ic = []
                for f in range(num_factors):
                    if method == 'pearson':
                        ic, _ = stats.pearsonr(factor_t[:, f], return_t1)
                    else:  # spearman
                        ic, _ = stats.spearmanr(factor_t[:, f], return_t1)
                    
                    period_ic.append(ic)
                
                ic_series.append(period_ic)
            
            ic_series = np.array(ic_series)  # [periods-1, factors]
            
            # 计算IC统计量
            ic_mean = np.nanmean(ic_series, axis=0)
            ic_std = np.nanstd(ic_series, axis=0)
            ic_ir = ic_mean / ic_std  # IC信息比率
            ic_t_stat = ic_ir * np.sqrt(len(ic_series))  # t统计量
            
            # 计算IC胜率
            ic_win_rate = np.mean(ic_series > 0, axis=0)
            
            # 累积IC
            ic_cumsum = np.nancumsum(ic_series, axis=0)
            
        else:
            # 截面数据
            if factors.dim() == 2:
                factors = factors.numpy()
            if returns.dim() == 1:
                returns = returns.numpy()
            
            num_stocks, num_factors = factors.shape
            ic_mean = np.zeros(num_factors)
            
            for f in range(num_factors):
                if method == 'pearson':
                    ic, _ = stats.pearsonr(factors[:, f], returns)
                else:
                    ic, _ = stats.spearmanr(factors[:, f], returns)
                ic_mean[f] = ic
            
            ic_std = np.zeros_like(ic_mean)  # 单期数据无法计算std
            ic_ir = np.zeros_like(ic_mean)
            ic_t_stat = np.zeros_like(ic_mean)
            ic_win_rate = np.zeros_like(ic_mean)
            ic_series = ic_mean.reshape(1, -1)
            ic_cumsum = ic_mean.reshape(1, -1)
        
        ic_results = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_t_stat': ic_t_stat,
            'ic_win_rate': ic_win_rate,
            'ic_series': ic_series,
            'ic_cumsum': ic_cumsum,
            'num_periods': len(ic_series) if len(ic_series.shape) > 1 else 1
        }
        
        # 打印结果
        factor_names = self.factor_names if self.factor_names else [f'Factor_{i}' for i in range(len(ic_mean))]
        
        print(f"样本期数: {ic_results['num_periods']}")
        print("\n因子IC统计:")
        for i, name in enumerate(factor_names[:len(ic_mean)]):
            print(f"{name:15}: IC均值={ic_mean[i]:7.4f}, IC标准差={ic_std[i]:7.4f}, "
                  f"IC_IR={ic_ir[i]:7.4f}, 胜率={ic_win_rate[i]:7.4f}")
        
        return ic_results
    
    def factor_grouping_backtest(self, 
                                factors: torch.Tensor,
                                returns: torch.Tensor,
                                num_groups: int = 5,
                                factor_idx: int = 0) -> Dict:
        """因子分组回测
        
        参数：
        - factors: 因子矩阵 [time, stocks, factors]
        - returns: 收益率矩阵 [time, stocks]
        - num_groups: 分组数量
        - factor_idx: 要测试的因子索引
        
        返回：
        - backtest_results: 分组回测结果
        """
        print(f"\n=== 因子分组回测（因子{factor_idx}）===")
        
        if factors.dim() != 3 or returns.dim() != 2:
            raise ValueError("需要时间序列数据进行分组回测")
        
        num_periods, num_stocks, num_factors = factors.shape
        
        # 存储各组收益
        group_returns = [[] for _ in range(num_groups)]
        long_short_returns = []
        
        for t in range(num_periods - 1):
            factor_t = factors[t, :, factor_idx].numpy()  # 当期因子值
            return_t1 = returns[t + 1].numpy()  # 下期收益
            
            # 按因子值排序，分组
            sorted_idx = np.argsort(factor_t)
            group_size = len(sorted_idx) // num_groups
            
            for g in range(num_groups):
                start_idx = g * group_size
                if g == num_groups - 1:
                    end_idx = len(sorted_idx)  # 最后一组包含剩余股票
                else:
                    end_idx = (g + 1) * group_size
                
                group_stocks = sorted_idx[start_idx:end_idx]
                group_ret = np.mean(return_t1[group_stocks])
                group_returns[g].append(group_ret)
            
            # 多空组合收益（最高分位 - 最低分位）
            long_ret = np.mean(return_t1[sorted_idx[-group_size:]])
            short_ret = np.mean(return_t1[sorted_idx[:group_size]])
            long_short_returns.append(long_ret - short_ret)
        
        # 计算统计量
        group_stats = []
        for g in range(num_groups):
            returns_g = np.array(group_returns[g])  # 转换为numpy数组
            stats_g = {
                'mean_return': np.mean(returns_g),
                'std_return': np.std(returns_g),
                'sharpe_ratio': np.mean(returns_g) / np.std(returns_g) * np.sqrt(252) if np.std(returns_g) > 0 else 0,
                'cumulative_return': np.cumprod(1 + returns_g) - 1,
                'max_drawdown': self._calculate_max_drawdown(returns_g)
            }
            group_stats.append(stats_g)
        
        # 多空组合统计
        long_short_returns_array = np.array(long_short_returns)  # 转换为numpy数组
        long_short_stats = {
            'mean_return': np.mean(long_short_returns_array),
            'std_return': np.std(long_short_returns_array),
            'sharpe_ratio': np.mean(long_short_returns_array) / np.std(long_short_returns_array) * np.sqrt(252) if np.std(long_short_returns_array) > 0 else 0,
            'cumulative_return': np.cumprod(1 + long_short_returns_array) - 1,
            'max_drawdown': self._calculate_max_drawdown(long_short_returns_array),
            't_stat': stats.ttest_1samp(long_short_returns_array, 0)[0],
            'p_value': stats.ttest_1samp(long_short_returns_array, 0)[1]
        }
        
        backtest_results = {
            'group_stats': group_stats,
            'long_short_stats': long_short_stats,
            'group_returns': group_returns,
            'long_short_returns': long_short_returns,
            'num_groups': num_groups,
            'num_periods': len(long_short_returns)
        }
        
        # 打印结果
        print(f"分组数量: {num_groups}, 回测期数: {len(long_short_returns)}")
        print("\n各组统计:")
        for g in range(num_groups):
            stats_g = group_stats[g]
            print(f"第{g+1}组: 平均收益={stats_g['mean_return']:7.4f}, "
                  f"夏普比率={stats_g['sharpe_ratio']:7.4f}, "
                  f"最大回撤={stats_g['max_drawdown']:7.4f}")
        
        print(f"\n多空组合: 平均收益={long_short_stats['mean_return']:7.4f}, "
              f"夏普比率={long_short_stats['sharpe_ratio']:7.4f}, "
              f"t统计量={long_short_stats['t_stat']:7.4f}, "
              f"p值={long_short_stats['p_value']:7.4f}")
        
        return backtest_results
    
    def _calculate_max_drawdown(self, returns) -> float:
        """计算最大回撤
        
        参数：
        - returns: 收益率序列（可以是list或numpy array）
        
        返回：
        - max_drawdown: 最大回撤值
        """
        # 确保输入是numpy数组
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
        
        # 处理空数组或单个值的情况
        if len(returns) == 0:
            return 0.0
        if len(returns) == 1:
            return min(0.0, returns[0])
        
        # 计算累积收益
        cumulative = np.cumprod(1 + returns)
        
        # 计算回撤
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def factor_stability_test(self, 
                             factors: torch.Tensor,
                             returns: torch.Tensor,
                             window_size: int = 12) -> Dict:
        """因子稳定性测试
        
        参数：
        - factors: 因子矩阵 [time, stocks, factors]
        - returns: 收益率矩阵 [time, stocks]
        - window_size: 滚动窗口大小
        
        返回：
        - stability_results: 稳定性测试结果
        """
        print(f"\n=== 因子稳定性测试（窗口={window_size}）===")
        
        if factors.dim() != 3 or returns.dim() != 2:
            raise ValueError("需要时间序列数据进行稳定性测试")
        
        num_periods, num_stocks, num_factors = factors.shape
        
        # 滚动窗口IC计算
        rolling_ic = []
        rolling_periods = []
        
        for start in range(0, num_periods - window_size, window_size // 2):  # 50%重叠
            end = start + window_size
            if end >= num_periods:
                break
            
            # 计算窗口内IC
            window_factors = factors[start:end]
            window_returns = returns[start:end]
            
            ic_result = self.compute_information_coefficient(
                window_factors, window_returns, method='pearson'
            )
            
            rolling_ic.append(ic_result['ic_mean'])
            rolling_periods.append((start, end))
        
        rolling_ic = np.array(rolling_ic)  # [windows, factors]
        
        # 计算稳定性指标
        ic_stability = {
            'ic_mean_stability': np.std(rolling_ic, axis=0),  # IC均值的标准差
            'ic_sign_stability': np.mean(rolling_ic > 0, axis=0),  # IC符号稳定性
            'ic_decay': [],  # IC衰减分析
            'rolling_ic': rolling_ic,
            'rolling_periods': rolling_periods
        }
        
        # IC衰减分析：不同预测期的IC
        decay_periods = [1, 2, 3, 5, 10]
        for period in decay_periods:
            if num_periods > period:
                # 计算period期后的IC
                decay_ic = []
                for t in range(num_periods - period):
                    factor_t = factors[t].numpy()
                    return_t = returns[t + period].numpy()
                    
                    period_ic = []
                    for f in range(num_factors):
                        ic, _ = stats.pearsonr(factor_t[:, f], return_t)
                        period_ic.append(ic)
                    
                    decay_ic.append(period_ic)
                
                decay_ic = np.array(decay_ic)
                ic_stability['ic_decay'].append({
                    'period': period,
                    'ic_mean': np.nanmean(decay_ic, axis=0)
                })
        
        # 打印结果
        factor_names = self.factor_names if self.factor_names else [f'Factor_{i}' for i in range(num_factors)]
        
        print(f"滚动窗口数量: {len(rolling_ic)}")
        print("\n稳定性指标:")
        for i, name in enumerate(factor_names[:num_factors]):
            print(f"{name:15}: IC标准差={ic_stability['ic_mean_stability'][i]:7.4f}, "
                  f"正IC比例={ic_stability['ic_sign_stability'][i]:7.4f}")
        
        print("\nIC衰减分析:")
        for decay_result in ic_stability['ic_decay']:
            period = decay_result['period']
            ic_mean = decay_result['ic_mean']
            print(f"预测期{period}: ", end="")
            for i, name in enumerate(factor_names[:num_factors]):
                print(f"{name}={ic_mean[i]:6.4f}", end=" ")
            print()
        
        return ic_stability
    
    def risk_adjusted_metrics(self, 
                             factor_returns,
                             benchmark_returns = None) -> Dict:
        """风险调整收益指标
        
        参数：
        - factor_returns: 因子收益序列
        - benchmark_returns: 基准收益序列（可选）
        
        返回：
        - risk_metrics: 风险调整指标字典
        """
        # 确保输入是numpy数组
        if not isinstance(factor_returns, np.ndarray):
            returns = np.array(factor_returns)
        else:
            returns = factor_returns
        
        # 基本统计量
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        
        # 偏度和峰度
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # VaR和CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # 最大回撤
        max_dd = self._calculate_max_drawdown(returns)
        
        # Calmar比率
        calmar = mean_ret * 252 / abs(max_dd) if max_dd != 0 else 0
        
        risk_metrics = {
            'mean_return': mean_ret,
            'volatility': std_ret,
            'sharpe_ratio': sharpe,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar
        }
        
        # 如果有基准收益，计算相对指标
        if benchmark_returns is not None:
            if not isinstance(benchmark_returns, np.ndarray):
                benchmark = np.array(benchmark_returns)
            else:
                benchmark = benchmark_returns
            
            excess_returns = returns - benchmark
            
            tracking_error = np.std(excess_returns)
            information_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
            
            # Beta计算
            beta = np.cov(returns, benchmark)[0, 1] / np.var(benchmark) if np.var(benchmark) > 0 else 0
            alpha = mean_ret - beta * np.mean(benchmark)
            
            risk_metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha
            })
        
        return risk_metrics
    
    def factor_correlation_analysis(self, factors: torch.Tensor) -> Dict:
        """因子相关性分析
        
        参数：
        - factors: 因子矩阵 [time, stocks, factors] 或 [stocks, factors]
        
        返回：
        - correlation_results: 相关性分析结果
        """
        print("\n=== 因子相关性分析 ===")
        
        if factors.dim() == 3:
            # 时间序列数据：计算时间平均后的截面相关性
            time_avg_factors = torch.mean(factors, dim=0)  # [stocks, factors]
            corr_matrix = torch.corrcoef(time_avg_factors.T)
        else:
            # 截面数据
            corr_matrix = torch.corrcoef(factors.T)
        
        corr_matrix = corr_matrix.numpy()
        num_factors = corr_matrix.shape[0]
        
        # 寻找高相关因子对
        high_corr_pairs = []
        for i in range(num_factors):
            for j in range(i + 1, num_factors):
                if abs(corr_matrix[i, j]) > 0.7:  # 相关性阈值
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        # 计算相关性统计量
        off_diagonal = corr_matrix[~np.eye(num_factors, dtype=bool)]
        mean_corr = np.mean(np.abs(off_diagonal))
        max_corr = np.max(np.abs(off_diagonal))
        
        correlation_results = {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'mean_abs_correlation': mean_corr,
            'max_abs_correlation': max_corr,
            'num_high_corr_pairs': len(high_corr_pairs)
        }
        
        print(f"因子数量: {num_factors}")
        print(f"平均绝对相关系数: {mean_corr:.4f}")
        print(f"最大绝对相关系数: {max_corr:.4f}")
        print(f"高相关因子对数量(|r|>0.7): {len(high_corr_pairs)}")
        
        if high_corr_pairs:
            print("\n高相关因子对:")
            factor_names = self.factor_names if self.factor_names else [f'Factor_{i}' for i in range(num_factors)]
            for i, j, corr in high_corr_pairs[:5]:  # 只显示前5对
                name_i = factor_names[i] if i < len(factor_names) else f'Factor_{i}'
                name_j = factor_names[j] if j < len(factor_names) else f'Factor_{j}'
                print(f"  {name_i} - {name_j}: {corr:.4f}")
        
        return correlation_results
    
    def comprehensive_factor_evaluation(self,
                                      model: nn.Module,
                                      test_data: Dict,
                                      num_test_periods: int = 20) -> Dict:
        """综合因子评估
        
        参数：
        - model: 训练好的ASTGNN模型
        - test_data: 测试数据字典
        - num_test_periods: 测试期数
        
        返回：
        - evaluation_results: 综合评估结果
        """
        print("\n" + "="*50)
        print("综合因子有效性评估")
        print("="*50)
        
        factors = test_data['factors']
        returns = test_data['returns_standardized']
        adj_matrices = test_data['adjacency_matrices']
        
        # 限制测试期数
        if num_test_periods < factors.shape[0]:
            factors = factors[:num_test_periods]
            returns = returns[:num_test_periods]
            adj_matrices = adj_matrices[:num_test_periods]
        
        # 使用模型生成因子预测
        model.eval()
        seq_len = 10
        
        if factors.shape[0] > seq_len:
            # 创建序列输入
            sequences = []
            targets = []
            
            for i in range(seq_len, factors.shape[0]):
                seq_input = factors[i-seq_len:i].transpose(0, 1).transpose(0, 1)
                target = returns[i]
                sequences.append(seq_input)
                targets.append(target)
            
            sequences = torch.stack(sequences)
            targets = torch.stack(targets)
            
            with torch.no_grad():
                predictions, risk_factors, _ = model(sequences, adj_matrices[0])
            
            # 使用生成的risk_factors作为因子
            factor_data = risk_factors
            return_data = targets
        else:
            print("数据不足，使用原始因子数据")
            factor_data = factors[1:]  # 去掉第一期
            return_data = returns[1:]
        
        evaluation_results = {}
        
        # 1. IC分析
        ic_results = self.compute_information_coefficient(factor_data, return_data)
        evaluation_results['ic_analysis'] = ic_results
        
        # 2. 分组回测（测试前几个因子）
        backtest_results = []
        for factor_idx in range(min(3, factor_data.shape[-1])):  # 测试前3个因子
            backtest = self.factor_grouping_backtest(
                factor_data, return_data, num_groups=5, factor_idx=factor_idx
            )
            backtest_results.append(backtest)
        evaluation_results['backtest_analysis'] = backtest_results
        
        # 3. 稳定性测试
        if factor_data.shape[0] > 12:  # 需要足够的时间期数
            stability_results = self.factor_stability_test(
                factor_data, return_data, window_size=min(12, factor_data.shape[0]//2)
            )
            evaluation_results['stability_analysis'] = stability_results
        
        # 4. 相关性分析
        correlation_results = self.factor_correlation_analysis(factor_data)
        evaluation_results['correlation_analysis'] = correlation_results
        
        # 5. 综合评分
        score = self._calculate_comprehensive_score(evaluation_results)
        evaluation_results['comprehensive_score'] = score
        
        print(f"\n=== 综合评估得分 ===")
        print(f"总分: {score['total_score']:.2f}/100")
        print(f"  - IC质量: {score['ic_score']:.1f}/30")
        print(f"  - 收益能力: {score['return_score']:.1f}/25")
        print(f"  - 稳定性: {score['stability_score']:.1f}/25")
        print(f"  - 独立性: {score['independence_score']:.1f}/20")
        
        return evaluation_results
    
    def _calculate_comprehensive_score(self, results: Dict) -> Dict:
        """计算综合评分"""
        scores = {'ic_score': 0, 'return_score': 0, 'stability_score': 0, 'independence_score': 0}
        
        # IC质量得分 (30分)
        if 'ic_analysis' in results:
            ic_data = results['ic_analysis']
            ic_mean = np.abs(ic_data['ic_mean']).mean()
            ic_ir = np.abs(ic_data['ic_ir']).mean()
            ic_win_rate = ic_data['ic_win_rate'].mean()
            
            scores['ic_score'] = min(30, (ic_mean * 100 + ic_ir * 5 + ic_win_rate * 20))
        
        # 收益能力得分 (25分)
        if 'backtest_analysis' in results:
            sharpe_ratios = []
            for backtest in results['backtest_analysis']:
                sharpe = backtest['long_short_stats']['sharpe_ratio']
                sharpe_ratios.append(abs(sharpe))
            
            avg_sharpe = np.mean(sharpe_ratios)
            scores['return_score'] = min(25, avg_sharpe * 10)
        
        # 稳定性得分 (25分)
        if 'stability_analysis' in results:
            stability_data = results['stability_analysis']
            sign_stability = stability_data['ic_sign_stability'].mean()
            scores['stability_score'] = sign_stability * 25
        
        # 独立性得分 (20分)
        if 'correlation_analysis' in results:
            corr_data = results['correlation_analysis']
            mean_corr = corr_data['mean_abs_correlation']
            scores['independence_score'] = max(0, 20 * (1 - mean_corr))
        
        scores['total_score'] = sum(scores.values())
        return scores


def test_factor_validation():
    """测试因子验证框架"""
    print("=== 因子验证框架测试 ===")
    
    # 加载数据
    try:
        training_data = torch.load('astgnn_training_data.pt', weights_only=False)
        factors = training_data['factors'][:30]  # 使用前30期数据
        returns = training_data['returns_standardized'][:30]
        adj_matrices = training_data['adjacency_matrices'][:30]
        factor_names = training_data['factor_names']
        
        print(f"加载数据: {factors.shape[0]}期, {factors.shape[1]}只股票, {factors.shape[2]}个因子")
        
        # 创建验证框架
        validator = FactorValidationFramework(factor_names=factor_names)
        
        # 1. IC分析
        ic_results = validator.compute_information_coefficient(factors, returns)
        
        # 2. 分组回测
        backtest_results = validator.factor_grouping_backtest(
            factors, returns, num_groups=5, factor_idx=0
        )
        
        # 3. 稳定性测试
        stability_results = validator.factor_stability_test(
            factors, returns, window_size=10
        )
        
        # 4. 相关性分析
        correlation_results = validator.factor_correlation_analysis(factors)
        
        print("\n=== 测试完成 ===")
        
    except FileNotFoundError:
        print("未找到数据文件，创建模拟数据进行测试...")
        
        # 创建模拟数据
        num_periods, num_stocks, num_factors = 25, 200, 10
        
        # 生成有一定预测能力的因子
        torch.manual_seed(42)
        base_factors = torch.randn(num_periods, num_stocks, num_factors)
        
        # 第一个因子有较强的预测能力
        true_alpha = torch.tensor([0.05, 0.02, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        returns = torch.zeros(num_periods, num_stocks)
        for t in range(num_periods):
            factor_exposure = base_factors[t] @ true_alpha
            noise = torch.randn(num_stocks) * 0.1
            returns[t] = factor_exposure + noise
        
        factor_names = [f'Factor_{i+1}' for i in range(num_factors)]
        
        # 测试验证框架
        validator = FactorValidationFramework(factor_names=factor_names)
        
        print("使用模拟数据测试...")
        ic_results = validator.compute_information_coefficient(base_factors, returns)
        backtest_results = validator.factor_grouping_backtest(
            base_factors, returns, num_groups=5, factor_idx=0
        )


if __name__ == "__main__":
    test_factor_validation() 