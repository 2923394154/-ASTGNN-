import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BarraFactorSimulator:
    """Barra风险因子数据模拟器
    
    基于Barra CNE5/CNE6模型生成符合真实特征分布的模拟因子数据，包含10大类风格因子：
    Size, Beta, Trend, Liquidity, Volatility, Value, Growth, SOE, Cubic Size, Certainty
    
    """
    
    def __init__(self, num_stocks=500, seed=42):
        """
        参数：
        - num_stocks: 股票数量
        - seed: 随机种子
        """
        self.num_stocks = num_stocks
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 定义各因子的统计特性（基于Barra.md的计算方法）
        self.factor_specs = self._define_factor_specifications()
        self.factor_names = list(self.factor_specs.keys())
        
        # 生成股票基础属性
        self.stock_attrs = self._generate_stock_attributes()
        
        # 生成基础价格和收益率时间序列（用于计算衍生因子）
        self._generate_base_time_series()
        
    def _define_factor_specifications(self) -> Dict:
        """定义各因子的统计特性，基于Barra.md中的计算方法"""
        return {
            # 1. Size (市值因子)
            'size': {
                'mean': 15.8, 'std': 1.6, 'min': 12.5, 'max': 19.5,
                'description': 'ln(总市值)，其中总市值=股价×总股本',
                'formula': 'Size = ln(总市值)'
            },
            
            # 2. Beta (贝叶斯压缩过后的市场beta)
            'beta': {
                'mean': 1.0, 'std': 0.25, 'min': 0.3, 'max': 1.8,
                'description': '贝叶斯压缩过后的市场beta',
                'formula': 'β_compressed = w × β_raw + (1-w) × β_market_avg'
            },
            
            # 3. Trend (价格趋势因子)
            'trend_120': {
                'mean': 1.0, 'std': 0.12, 'min': 0.7, 'max': 1.3,
                'description': 'EWMA(halflife=20)/EWMA(halflife=120)',
                'formula': 'Trend_120 = EWMA(halflife=20) / EWMA(halflife=120)'
            },
            'trend_240': {
                'mean': 1.0, 'std': 0.18, 'min': 0.6, 'max': 1.4,
                'description': 'EWMA(halflife=20)/EWMA(halflife=240)',
                'formula': 'Trend_240 = EWMA(halflife=20) / EWMA(halflife=240)'
            },
            
            # 4. Liquidity (流动性因子)
            'turnover_volatility': {
                'mean': 0.018, 'std': 0.012, 'min': 0.003, 'max': 0.06,
                'description': '过去243天的换手率标准波动率',
                'formula': 'TO = σ(日换手率)，时间窗口：过去243个交易日'
            },
            'liquidity_beta': {
                'mean': 1.0, 'std': 0.35, 'min': 0.2, 'max': 2.2,
                'description': '个股对数换手率与市场对数换手率回归系数',
                'formula': 'ln(TO_i) = α + β × ln(TO_market) + ε'
            },
            
            # 5. Volatility (波动率因子组)
            'std_vol': {
                'mean': 0.028, 'std': 0.015, 'min': 0.008, 'max': 0.12,
                'description': '过去243天的标准波动率',
                'formula': 'StdVol = σ(日收益率)，时间窗口：过去243个交易日'
            },
            'lvff': {
                'mean': 0.024, 'std': 0.013, 'min': 0.006, 'max': 0.10,
                'description': '过去243天的Fama-French三因子特质波动率',
                'formula': 'r_i = α + β_m×r_market + β_s×SMB + β_v×HML + ε; Lvff = σ(ε)'
            },
            'range_vol': {
                'mean': 0.075, 'std': 0.035, 'min': 0.02, 'max': 0.20,
                'description': '过去243天的最高价/最低价-1的均值',
                'formula': 'Range = mean[(最高价/最低价 - 1)]，时间窗口：过去243个交易日'
            },
            'max_ret_6': {
                'mean': 0.038, 'std': 0.018, 'min': 0.008, 'max': 0.12,
                'description': '过去243天收益最高的6天的平均值',
                'formula': 'MaxRet_6 = mean(过去243天中收益最高的6天)'
            },
            'min_ret_6': {
                'mean': 0.038, 'std': 0.018, 'min': 0.008, 'max': 0.12,
                'description': '过去243天收益最低6天的收益绝对值平均值',
                'formula': 'MinRet_6 = mean(过去243天中收益最低的6天的绝对值)'
            },
            
            # 6. Value (价值因子)
            'ep_ratio': {
                'mean': 0.055, 'std': 0.12, 'min': -0.3, 'max': 0.4,
                'description': '盈利收益率 = 净利润(TTM) / 总市值',
                'formula': 'EP = 净利润(TTM) / 总市值'
            },
            'bp_ratio': {
                'mean': 0.75, 'std': 0.55, 'min': 0.05, 'max': 2.8,
                'description': '账面市值比 = 净资产 / 总市值',
                'formula': 'BP = 净资产 / 总市值'
            },
            
            # 7. Growth (成长因子)
            'delta_roe': {
                'mean': 0.015, 'std': 0.18, 'min': -0.6, 'max': 0.6,
                'description': '过去3年ROE变动的算术平均值',
                'formula': 'Delta ROE = mean(ROE_t - ROE_{t-1}, ROE_{t-1} - ROE_{t-2}, ROE_{t-2} - ROE_{t-3})'
            },
            'sales_growth': {
                'mean': 0.11, 'std': 0.28, 'min': -0.7, 'max': 1.2,
                'description': '销售收入TTM的3年复合增长率',
                'formula': 'Sales_growth = (销售收入TTM / 销售收入TTM_3年前)^(1/3) - 1'
            },
            'na_growth': {
                'mean': 0.095, 'std': 0.22, 'min': -0.5, 'max': 0.9,
                'description': '净资产TTM的3年复合增长率',
                'formula': 'Na_growth = (净资产TTM / 净资产TTM_3年前)^(1/3) - 1'
            },
            
            # 8. SOE (国有企业因子)
            'soe_ratio': {
                'mean': 0.28, 'std': 0.42, 'min': 0.0, 'max': 1.0,
                'description': '国有股东持股比例 (%)',
                'formula': 'SOE = 国有股东持股比例 (%)'
            },
            
            # 9. Cubic Size (市值幂次项)
            'cubic_size': {
                'mean': 0.0, 'std': 1.0, 'min': -3.0, 'max': 3.0,
                'description': '标准化的市值幂次项',
                'formula': 'Cubic Size = [ln(总市值)]^3，然后标准化'
            },
            
            # 10. Certainty (确定性因子组)
            'instholder_pct': {
                'mean': 0.22, 'std': 0.18, 'min': 0.0, 'max': 0.75,
                'description': '公募基金持股比例',
                'formula': 'Instholder Pct = 公募基金持股数量 / 总股本 × 100%'
            },
            'analyst_coverage': {
                'mean': 0.0, 'std': 1.0, 'min': -2.5, 'max': 2.5,
                'description': '分析师覆盖度（对市值正交化后标准化）',
                'formula': 'Cov_adj = Cov - β × ln(市值)，然后标准化'
            },
            'list_days': {
                'mean': 0.0, 'std': 1.0, 'min': -2.0, 'max': 2.0,
                'description': '标准化的上市天数',
                'formula': 'Listdays = ln(当前日期 - 上市日期)，然后标准化'
            }
        }
    
    def _generate_stock_attributes(self) -> Dict:
        """生成股票基础属性"""
        return {
            'stock_ids': [f'stock_{i:04d}' for i in range(self.num_stocks)],
            'sectors': np.random.choice(['金融', '制造业', '信息技术', '消费', '能源', '医药生物', 
                                       '房地产', '交通运输', '公用事业', '原材料'], self.num_stocks),
            'base_market_caps': np.random.lognormal(15.8, 1.5, self.num_stocks),  # 基础市值
            'is_st': np.random.choice([0, 1], self.num_stocks, p=[0.97, 0.03]),  # ST状态
            'listing_years': np.random.uniform(1, 25, self.num_stocks)  # 上市年数
        }
    
    def _generate_base_time_series(self):
        """生成基础的价格和收益率时间序列，用于计算衍生因子"""
        # 生成252天的日收益率数据（用于计算波动率因子）
        self.daily_returns = np.random.multivariate_normal(
            mean=np.zeros(self.num_stocks),
            cov=self._create_return_covariance_matrix(),
            size=252
        ).T  # [num_stocks, 252]
        
        # 生成市场收益率
        self.market_returns = np.random.normal(0.0005, 0.015, 252)
        
        # 生成价格序列（用于计算Trend因子）
        initial_prices = 10 + np.random.exponential(20, self.num_stocks)
        self.price_series = np.zeros((self.num_stocks, 252))
        self.price_series[:, 0] = initial_prices
        
        for t in range(1, 252):
            self.price_series[:, t] = self.price_series[:, t-1] * (1 + self.daily_returns[:, t])
    
    def _create_return_covariance_matrix(self) -> np.ndarray:
        """创建股票收益率的协方差矩阵"""
        # 创建基于行业的协方差结构
        correlation_base = 0.15  # 基础相关性
        intra_sector_corr = 0.4   # 行业内相关性
        
        covariance_matrix = np.eye(self.num_stocks) * 0.0008  # 对角线方差
        
        # 添加行业内相关性
        sectors = self.stock_attrs['sectors']
        for i in range(self.num_stocks):
            for j in range(i+1, self.num_stocks):
                if sectors[i] == sectors[j]:
                    corr = intra_sector_corr + np.random.normal(0, 0.1)
                else:
                    corr = correlation_base + np.random.normal(0, 0.05)
                
                corr = np.clip(corr, -0.8, 0.8)
                vol_i = np.sqrt(covariance_matrix[i, i])
                vol_j = np.sqrt(covariance_matrix[j, j])
                covariance_matrix[i, j] = corr * vol_i * vol_j
                covariance_matrix[j, i] = covariance_matrix[i, j]
        
        return covariance_matrix
    
    def _calculate_beta_with_bayesian_shrinkage(self) -> np.ndarray:
        """计算贝叶斯压缩后的Beta"""
        # 计算原始Beta
        betas_raw = np.zeros(self.num_stocks)
        beta_errors = np.zeros(self.num_stocks)
        
        for i in range(self.num_stocks):
            # 对市场收益回归计算Beta
            X = np.column_stack([np.ones(252), self.market_returns])
            y = self.daily_returns[i, :]
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                betas_raw[i] = coeffs[1]
                
                # 计算回归残差的标准差
                y_pred = X @ coeffs
                residuals = y - y_pred
                beta_errors[i] = np.std(residuals) / np.std(self.market_returns)
            except:
                betas_raw[i] = 1.0
                beta_errors[i] = 0.3
        
        # 贝叶斯压缩
        market_avg_beta = 1.0
        beta_var = np.var(betas_raw)
        error_var = np.mean(beta_errors**2)
        
        # 计算压缩权重
        shrinkage_weights = beta_var / (beta_var + error_var)
        
        # 应用贝叶斯压缩
        betas_compressed = shrinkage_weights * betas_raw + (1 - shrinkage_weights) * market_avg_beta
        
        return betas_compressed
    
    def _calculate_ewma_trend_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算EWMA趋势因子"""
        def calculate_ewma(prices, halflife):
            """计算指数加权移动平均"""
            alpha = 1 - np.exp(-np.log(2) / halflife)
            ewma = np.zeros_like(prices)
            ewma[0] = prices[0]
            
            for t in range(1, len(prices)):
                ewma[t] = alpha * prices[t] + (1 - alpha) * ewma[t-1]
            
            return ewma[-1]  # 返回最新值
        
        trend_120 = np.zeros(self.num_stocks)
        trend_240 = np.zeros(self.num_stocks)
        
        for i in range(self.num_stocks):
            prices = self.price_series[i, :]
            
            ewma_20 = calculate_ewma(prices, 20)
            ewma_120 = calculate_ewma(prices, 120)
            ewma_240 = calculate_ewma(prices, 240)
            
            trend_120[i] = ewma_20 / ewma_120 if ewma_120 > 0 else 1.0
            trend_240[i] = ewma_20 / ewma_240 if ewma_240 > 0 else 1.0
        
        return trend_120, trend_240
    
    def _calculate_volatility_factors(self) -> Dict[str, np.ndarray]:
        """计算波动率因子组"""
        vol_factors = {}
        
        # 1. 标准波动率 (StdVol)
        vol_factors['std_vol'] = np.std(self.daily_returns, axis=1)
        
        # 2. Fama-French三因子特质波动率 (Lvff)
        ff_residuals = np.zeros_like(self.daily_returns)
        
        # 构造SMB和HML因子（简化版本）
        market_caps = self.stock_attrs['base_market_caps']
        small_stocks = market_caps < np.median(market_caps)
        
        smb_factor = np.mean(self.daily_returns[small_stocks], axis=0) - \
                    np.mean(self.daily_returns[~small_stocks], axis=0)
        
        # 简化的HML因子（基于随机价值分组）
        high_value = np.random.choice(self.num_stocks, size=self.num_stocks//3, replace=False)
        low_value = np.random.choice([i for i in range(self.num_stocks) if i not in high_value], 
                                   size=self.num_stocks//3, replace=False)
        
        hml_factor = np.mean(self.daily_returns[high_value], axis=0) - \
                    np.mean(self.daily_returns[low_value], axis=0)
        
        # 三因子回归
        for i in range(self.num_stocks):
            X = np.column_stack([np.ones(252), self.market_returns, smb_factor, hml_factor])
            y = self.daily_returns[i, :]
            
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ coeffs
                ff_residuals[i, :] = y - y_pred
            except:
                ff_residuals[i, :] = self.daily_returns[i, :] - np.mean(self.daily_returns[i, :])
        
        vol_factors['lvff'] = np.std(ff_residuals, axis=1)
        
        # 3. Range波动率
        # 简化：假设高低价差约为收益率的2倍标准差
        vol_factors['range_vol'] = 2.5 * vol_factors['std_vol'] + np.random.normal(0, 0.01, self.num_stocks)
        
        # 4. 极端收益因子
        sorted_returns = np.sort(self.daily_returns, axis=1)
        vol_factors['max_ret_6'] = np.mean(sorted_returns[:, -6:], axis=1)  # 最大6个收益的均值
        vol_factors['min_ret_6'] = np.abs(np.mean(sorted_returns[:, :6], axis=1))  # 最小6个收益的绝对值均值
        
        return vol_factors
    
    def _generate_correlated_factors(self, base_factors: np.ndarray, 
                                   correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """生成具有相关性的因子数据"""
        if correlation_matrix is None:
            # 创建默认的相关性矩阵
            correlation_matrix = self._create_default_correlation_matrix()
        
        # 使用Cholesky分解生成相关数据
        try:
            L = np.linalg.cholesky(correlation_matrix)
            correlated_factors = base_factors @ L.T
        except np.linalg.LinAlgError:
            # 如果矩阵不是正定的，使用特征值分解
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.01)  # 确保正定
            sqrt_eigenvals = np.sqrt(eigenvals)
            L = eigenvecs @ np.diag(sqrt_eigenvals)
            correlated_factors = base_factors @ L.T
        
        return correlated_factors
    
    def _create_default_correlation_matrix(self) -> np.ndarray:
        """创建默认的因子相关性矩阵"""
        n_factors = len(self.factor_names)
        correlation_matrix = np.eye(n_factors)
        
        # 定义一些合理的相关性
        factor_groups = {
            'size_group': ['size', 'cubic_size', 'analyst_coverage'],
            'volatility_group': ['std_vol', 'lvff', 'range_vol', 'max_ret_6'],
            'trend_group': ['trend_120', 'trend_240'],
            'growth_group': ['delta_roe', 'sales_growth', 'na_growth'],
            'liquidity_group': ['turnover_volatility', 'liquidity_beta']
        }
        
        # 在同组内设置较高相关性
        for group_factors in factor_groups.values():
            indices = [self.factor_names.index(f) for f in group_factors 
                      if f in self.factor_names]
            for i in indices:
                for j in indices:
                    if i != j:
                        correlation_matrix[i, j] = np.random.uniform(0.3, 0.7)
        
        return correlation_matrix
    
    def generate_single_period_data(self, add_noise=True, 
                                  noise_level=0.1) -> Tuple[torch.Tensor, pd.DataFrame]:
        """生成单期因子数据，基于Barra.md中定义的计算方法
        
        参数：
        - add_noise: 是否添加噪声
        - noise_level: 噪声水平
        
        返回：
        - factor_tensor: PyTorch张量 [num_stocks, num_factors]
        - factor_df: Pandas DataFrame
        """
        factor_data = np.zeros((self.num_stocks, len(self.factor_names)))
        
        # 1. Size因子 - 基于真实市值计算
        size_values = np.log(self.stock_attrs['base_market_caps'])
        
        # 2. Beta因子 - 使用贝叶斯压缩
        beta_values = self._calculate_beta_with_bayesian_shrinkage()
        
        # 3. Trend因子 - 基于EWMA计算
        trend_120_values, trend_240_values = self._calculate_ewma_trend_factors()
        
        # 4. Volatility因子组 - 基于真实时间序列计算
        volatility_factors = self._calculate_volatility_factors()
        
        # 5. Liquidity因子 - 模拟换手率数据
        turnover_data = np.random.gamma(2, 0.01, (self.num_stocks, 243))  # 模拟243天换手率
        turnover_vol_values = np.std(turnover_data, axis=1)
        
        # 流动性Beta计算
        market_turnover = np.mean(turnover_data, axis=0)
        liquidity_beta_values = np.zeros(self.num_stocks)
        for i in range(self.num_stocks):
            try:
                # 对数化处理
                log_stock_to = np.log(turnover_data[i, :] + 1e-8)
                log_market_to = np.log(market_turnover + 1e-8)
                
                # 回归计算
                X = np.column_stack([np.ones(243), log_market_to])
                coeffs = np.linalg.lstsq(X, log_stock_to, rcond=None)[0]
                liquidity_beta_values[i] = coeffs[1]
            except:
                liquidity_beta_values[i] = 1.0
        
        # 6. Value因子 - 基于财务数据模拟
        # EP比率（允许负值）
        earnings = np.random.normal(0.06, 0.15, self.num_stocks)  # 模拟盈利率
        ep_values = earnings / np.log(self.stock_attrs['base_market_caps']) * 15  # 调整尺度
        
        # BP比率
        book_values = np.random.lognormal(0.5, 0.8, self.num_stocks)
        bp_values = book_values / (self.stock_attrs['base_market_caps'] / 1e8)
        
        # 7. Growth因子 - 基于历史增长模拟
        # Delta ROE
        base_roe = np.random.normal(0.08, 0.12, self.num_stocks)
        roe_changes = np.random.normal(0.01, 0.05, (self.num_stocks, 3))
        delta_roe_values = np.mean(roe_changes, axis=1)
        
        # Sales growth和NA growth
        sales_growth_values = np.random.normal(0.11, 0.28, self.num_stocks)
        na_growth_values = np.random.normal(0.095, 0.22, self.num_stocks)
        
        # 8. SOE因子 - 二元混合分布
        soe_values = np.random.beta(1.5, 4, self.num_stocks)  # 更符合中国市场分布
        
        # 9. Cubic Size因子 - 基于市值的三次方
        log_caps = np.log(self.stock_attrs['base_market_caps'])
        cubic_raw = log_caps ** 3
        cubic_size_values = (cubic_raw - np.mean(cubic_raw)) / np.std(cubic_raw)  # 标准化
        
        # 10. Certainty因子组
        # 机构持股比例
        instholder_values = np.random.beta(2, 6, self.num_stocks)
        
        # 分析师覆盖度（对市值正交化）
        raw_coverage = np.random.poisson(8, self.num_stocks)
        # 对市值回归
        X_coverage = np.column_stack([np.ones(self.num_stocks), log_caps])
        try:
            coeffs_coverage = np.linalg.lstsq(X_coverage, raw_coverage, rcond=None)[0]
            coverage_residual = raw_coverage - X_coverage @ coeffs_coverage
            analyst_coverage_values = (coverage_residual - np.mean(coverage_residual)) / np.std(coverage_residual)
        except:
            analyst_coverage_values = np.random.normal(0, 1, self.num_stocks)
        
        # 上市天数（标准化）
        listing_days_raw = self.stock_attrs['listing_years'] * 365
        listing_days_log = np.log(listing_days_raw + 1)
        list_days_values = (listing_days_log - np.mean(listing_days_log)) / np.std(listing_days_log)
        
        # 组装所有因子数据
        factor_dict = {
            'size': size_values,
            'beta': beta_values,
            'trend_120': trend_120_values,
            'trend_240': trend_240_values,
            'turnover_volatility': turnover_vol_values,
            'liquidity_beta': liquidity_beta_values,
            'std_vol': volatility_factors['std_vol'],
            'lvff': volatility_factors['lvff'],
            'range_vol': volatility_factors['range_vol'],
            'max_ret_6': volatility_factors['max_ret_6'],
            'min_ret_6': volatility_factors['min_ret_6'],
            'ep_ratio': ep_values,
            'bp_ratio': bp_values,
            'delta_roe': delta_roe_values,
            'sales_growth': sales_growth_values,
            'na_growth': na_growth_values,
            'soe_ratio': soe_values,
            'cubic_size': cubic_size_values,
            'instholder_pct': instholder_values,
            'analyst_coverage': analyst_coverage_values,
            'list_days': list_days_values
        }
        
        # 填充factor_data矩阵
        for i, factor_name in enumerate(self.factor_names):
            values = factor_dict[factor_name]
            
            # 应用因子规格约束
            spec = self.factor_specs[factor_name]
            values = np.clip(values, spec['min'], spec['max'])
            factor_data[:, i] = values
        
        # 添加噪声
        if add_noise:
            noise = np.random.normal(0, noise_level, factor_data.shape)
            factor_data += noise
            
            # 再次应用约束
            for i, (factor_name, spec) in enumerate(self.factor_specs.items()):
                factor_data[:, i] = np.clip(factor_data[:, i], spec['min'], spec['max'])
        
        # 创建DataFrame
        factor_df = pd.DataFrame(
            factor_data,
            columns=self.factor_names,
            index=self.stock_attrs['stock_ids']
        )
        
        # 添加额外信息
        factor_df['sector'] = self.stock_attrs['sectors']
        factor_df['is_st'] = self.stock_attrs['is_st']
        
        # 转换为张量
        factor_tensor = torch.tensor(factor_data, dtype=torch.float32)
        
        return factor_tensor, factor_df
    
    def generate_time_series_data(self, num_periods=60, start_date='2023-01-01',
                                freq='M') -> Tuple[torch.Tensor, pd.DataFrame]:
        """生成时间序列因子数据
        
        参数：
        - num_periods: 时间期数
        - start_date: 起始日期
        - freq: 频率 ('D', 'W', 'M')
        
        返回：
        - factor_tensor: [num_periods, num_stocks, num_factors]
        - factor_panel: 包含时间索引的DataFrame
        """
        # 生成时间索引
        start_dt = pd.to_datetime(start_date)
        if freq == 'D':
            dates = pd.date_range(start_dt, periods=num_periods, freq='D')
        elif freq == 'W':
            dates = pd.date_range(start_dt, periods=num_periods, freq='W')
        else:  # 'M'
            dates = pd.date_range(start_dt, periods=num_periods, freq='M')
        
        all_data = []
        factor_tensors = []
        
        for t, date in enumerate(dates):
            # 生成当期数据
            factor_tensor, factor_df = self.generate_single_period_data(
                add_noise=True, noise_level=0.05 + 0.05 * np.sin(t * 0.1)  # 时变噪声
            )
            
            # 添加时间趋势
            trend_factor = 1 + 0.01 * t / num_periods  # 轻微的时间趋势
            factor_tensor *= trend_factor
            
            # 添加季节性效应
            seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * t / 12)  # 年度季节性
            factor_tensor *= seasonal_factor
            
            # 重置索引，将stock_ids变为列
            factor_df_reset = factor_df.reset_index()
            factor_df_reset.rename(columns={'index': 'stock_id'}, inplace=True)
            factor_df_reset['date'] = date
            factor_df_reset['period'] = t
            
            all_data.append(factor_df_reset)
            factor_tensors.append(factor_tensor)
        
        # 合并所有时期数据
        factor_panel = pd.concat(all_data, ignore_index=True)
        factor_panel = factor_panel.set_index(['date', 'stock_id'])
        
        # 堆叠张量
        factor_tensor_3d = torch.stack(factor_tensors, dim=0)
        
        return factor_tensor_3d, factor_panel
    
    def generate_returns_from_factors_enhanced(self, factor_tensor: torch.Tensor,
                                             time_varying_returns=True,
                                             signal_strength=0.3) -> torch.Tensor:
        """增强版因子收益生成，支持时变因子收益和更强信号"""
        
        if time_varying_returns:
            # 时变因子收益率（更符合现实）
            num_periods = factor_tensor.shape[0] if factor_tensor.dim() == 3 else 1
            
            # 基础因子收益率（更保守，更现实）
            base_factor_returns = torch.tensor([
                -0.008,  # size
                -0.004,  # beta  
                0.06,    # trend_120 (减小)
                0.04,    # trend_240 (减小)
                -0.03,   # turnover_volatility (减小)
                -0.02,   # liquidity_beta (减小)
                -0.045,  # std_vol (减小)
                -0.035,  # lvff (减小)
                -0.025,  # range_vol (减小)
                -0.015,  # max_ret_6 (减小)
                0.015,   # min_ret_6 (减小)
                0.035,   # ep_ratio (减小)
                0.025,   # bp_ratio (减小)
                0.045,   # delta_roe (减小)
                0.055,   # sales_growth (减小)
                0.03,    # na_growth (减小)
                -0.012,  # soe_ratio (减小)
                0.004,   # cubic_size (减小)
                0.012,   # instholder_pct (减小)
                0.008,   # analyst_coverage (减小)
                -0.002   # list_days (减小)
        ], dtype=torch.float32)
            
            if factor_tensor.dim() == 3:
                # 为每个时期生成不同的因子收益率
                factor_returns_series = []
                for t in range(num_periods):
                    # 添加时间趋势和噪声
                    trend = 0.02 * np.sin(2 * np.pi * t / 12)  # 季节性
                    noise = torch.randn_like(base_factor_returns) * 0.01  # 噪声
                    time_varying_returns = base_factor_returns * (1 + trend) + noise
                    factor_returns_series.append(time_varying_returns)
                
                factor_returns_tensor = torch.stack(factor_returns_series)  # [time, factors]
                
                # 计算收益率
                returns = torch.sum(factor_tensor * factor_returns_tensor.unsqueeze(1), dim=2)
            else:
                returns = torch.matmul(factor_tensor, base_factor_returns)
        else:
            # 使用原有逻辑但降低信号强度
            factor_returns = base_factor_returns * signal_strength
            if factor_tensor.dim() == 2:
                returns = torch.matmul(factor_tensor, factor_returns)
            else:
                returns = torch.matmul(factor_tensor, factor_returns)
        
        # 添加更现实的特质收益（降低噪声比例）
        if factor_tensor.dim() == 2:
            idiosyncratic = torch.randn(factor_tensor.shape[0]) * 0.08  # 减少噪声
        else:
            idiosyncratic = torch.randn(factor_tensor.shape[:2]) * 0.08  # 减少噪声
        
                returns += idiosyncratic
        
        return returns
    
    def simulate_factor_panel_data_enhanced(self, num_periods=60):
        """生成增强版面板数据，确保因子有预测能力"""
        print("生成增强版因子面板数据...")
        
        # 生成因子时间序列
        factor_3d, factor_panel = self.generate_time_series_data(num_periods)
        
        # 生成对应的收益率（使用增强版生成方法）
        returns_3d = self.generate_returns_from_factors_enhanced(
            factor_3d, time_varying_returns=True, signal_strength=0.4
        )
        
        # 标准化收益率
        returns_standardized = torch.zeros_like(returns_3d)
        for t in range(num_periods):
            returns_t = returns_3d[t]
            mean_t = returns_t.mean()
            std_t = returns_t.std()
            returns_standardized[t] = (returns_t - mean_t) / (std_t + 1e-8)
        
        # 生成邻接矩阵
        adj_matrices = []
        for t in range(num_periods):
            factor_df_t = factor_panel.reset_index()
            factor_df_t = factor_df_t[factor_df_t['period'] == t]
            adj_t = self.create_adjacency_matrix(factor_df_t, method='sector')
            adj_matrices.append(adj_t)
        
        adj_matrices = torch.stack(adj_matrices, dim=0)
        
        # 验证数据质量
        print("验证生成数据的IC...")
        validation_framework = FactorValidationFramework(factor_names=self.factor_names)
        baseline_ic = validation_framework.compute_information_coefficient(
            factor_3d[:-1], returns_standardized[1:]
        )
        print(f"基线IC均值: {np.mean(np.abs(baseline_ic['ic_mean'])):.4f}")
        
        return {
            'factors': factor_3d,
            'returns': returns_3d,
            'returns_standardized': returns_standardized,
            'adjacency_matrices': adj_matrices,
            'factor_names': self.factor_names,
            'factor_panel': factor_panel
        }
    
    def create_adjacency_matrix(self, factor_df: pd.DataFrame,
                              method='correlation', threshold=0.3) -> torch.Tensor:
        """根据因子数据创建邻接矩阵
        
        参数：
        - factor_df: 因子数据DataFrame
        - method: 相似性计算方法 ('correlation', 'distance', 'sector')
        - threshold: 连接阈值
        
        返回：
        - adj_matrix: 邻接矩阵
        """
        # 处理不同的DataFrame格式
        if 'stock_id' in factor_df.columns:
            # 时间序列数据格式
            factor_cols = [col for col in factor_df.columns 
                          if col not in ['sector', 'is_st', 'date', 'period', 'stock_id']]
            factor_data = factor_df[factor_cols].values
            sectors = factor_df['sector'].values if 'sector' in factor_df.columns else None
        else:
            # 单期数据格式
            factor_cols = [col for col in factor_df.columns 
                          if col not in ['sector', 'is_st', 'date', 'period']]
            factor_data = factor_df[factor_cols].values
            sectors = factor_df['sector'].values if 'sector' in factor_df.columns else None
        
        if method == 'correlation':
            # 基于因子相关性
            corr_matrix = np.corrcoef(factor_data)
            adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
        
        elif method == 'distance':
            # 基于欧氏距离
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(factor_data)
            distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())
            adj_matrix = (distances_normalized < threshold).astype(float)
        
        elif method == 'sector':
            # 基于行业
            if sectors is not None:
                adj_matrix = (sectors[:, None] == sectors[None, :]).astype(float)
                
                # 添加一些因子相似性
                corr_matrix = np.corrcoef(factor_data)
                factor_sim = (np.abs(corr_matrix) > 0.5).astype(float)
                adj_matrix = np.logical_or(adj_matrix, factor_sim).astype(float)
            else:
                # 如果没有行业信息，退化为相关性方法
                corr_matrix = np.corrcoef(factor_data)
                adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
        
        # 移除自环
        np.fill_diagonal(adj_matrix, 0)
        
        return torch.tensor(adj_matrix, dtype=torch.float32)


def test_barra_simulator():
    """测试Barra因子模拟器"""
    print("=== Barra因子模拟器测试 ===")
    
    # 创建模拟器
    simulator = BarraFactorSimulator(num_stocks=200, seed=42)
    
    # 1. 生成单期数据
    print("\n1. 单期数据生成测试")
    factor_tensor, factor_df = simulator.generate_single_period_data()
    
    print(f"因子张量形状: {factor_tensor.shape}")
    print(f"DataFrame形状: {factor_df.shape}")
    print(f"因子列表: {simulator.factor_names}")
    
    # 展示统计信息
    print("\n因子统计信息:")
    print(factor_df[simulator.factor_names].describe())
    
    # 2. 生成时间序列数据
    print("\n2. 时间序列数据生成测试")
    factor_3d, factor_panel = simulator.generate_time_series_data(
        num_periods=12, freq='M'
    )
    
    print(f"3D因子张量形状: {factor_3d.shape}")
    print(f"面板数据形状: {factor_panel.shape}")
    print(f"面板数据索引: {factor_panel.index.names}")
    print(f"面板数据列: {factor_panel.columns.tolist()}")
    
    # 3. 生成收益率
    print("\n3. 基于因子生成收益率")
    returns = simulator.generate_returns_from_factors_enhanced(factor_tensor)
    print(f"收益率张量形状: {returns.shape}")
    print(f"收益率统计: 均值={returns.mean():.4f}, 标准差={returns.std():.4f}")
    
    # 4. 创建邻接矩阵
    print("\n4. 创建邻接矩阵")
    adj_matrix = simulator.create_adjacency_matrix(factor_df, method='correlation')
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"连接密度: {adj_matrix.mean():.4f}")
    
    # 5. 因子相关性分析
    print("\n5. 因子相关性分析")
    factor_corr = factor_df[simulator.factor_names].corr()
    high_corr_pairs = []
    
    for i in range(len(simulator.factor_names)):
        for j in range(i+1, len(simulator.factor_names)):
            corr_val = factor_corr.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append((
                    simulator.factor_names[i], 
                    simulator.factor_names[j], 
                    corr_val
                ))
    
    print("高相关性因子对 (|相关系数| > 0.5):")
    for f1, f2, corr in high_corr_pairs[:5]:  # 只显示前5对
        print(f"  {f1} - {f2}: {corr:.3f}")


def create_training_dataset():
    """创建用于ASTGNN训练的完整数据集"""
    print("\n=== 创建ASTGNN训练数据集 ===")
    
    # 设置参数
    num_stocks = 500
    num_periods = 60
    
    simulator = BarraFactorSimulator(num_stocks=num_stocks, seed=42)
    
    # 生成时间序列因子数据
    factor_3d, factor_panel = simulator.generate_time_series_data(num_periods)
    
    # 生成对应的收益率
    returns_3d = torch.zeros(num_periods, num_stocks)
    adj_matrices = []
    
    for t in range(num_periods):
        # 当期因子数据
        factor_t = factor_3d[t]
        
        # 生成收益率
        returns_t = simulator.generate_returns_from_factors_enhanced(factor_t)
        returns_3d[t] = returns_t
        
        # 创建邻接矩阵 - 从面板数据中提取当期数据
        factor_df_t = factor_panel.reset_index()
        factor_df_t = factor_df_t[factor_df_t['period'] == t]
        adj_t = simulator.create_adjacency_matrix(factor_df_t, method='sector')
        adj_matrices.append(adj_t)
    
    adj_matrices = torch.stack(adj_matrices, dim=0)
    
    print(f"因子数据形状: {factor_3d.shape}")
    print(f"收益率数据形状: {returns_3d.shape}")
    print(f"邻接矩阵形状: {adj_matrices.shape}")
    
    # 标准化收益率（截面标准化）
    returns_standardized = torch.zeros_like(returns_3d)
    for t in range(num_periods):
        returns_t = returns_3d[t]
        mean_t = returns_t.mean()
        std_t = returns_t.std()
        returns_standardized[t] = (returns_t - mean_t) / std_t
    
    # 保存数据
    training_data = {
        'factors': factor_3d,
        'returns': returns_3d,
        'returns_standardized': returns_standardized,
        'adjacency_matrices': adj_matrices,
        'factor_names': simulator.factor_names,
        'stock_attrs': simulator.stock_attrs
    }
    
    # 保存为PyTorch格式
    torch.save(training_data, 'astgnn_training_data.pt')
    
    # 保存为CSV格式（用于检查）
    factor_panel.to_csv('factor_panel_data.csv')
    
    print("数据集创建完成！")
    print("- PyTorch数据: astgnn_training_data.pt")
    print("- CSV数据: factor_panel_data.csv")
    
    return training_data


if __name__ == "__main__":
    # 运行测试
    test_barra_simulator()
    
    # 创建训练数据集
    training_data = create_training_dataset()
    
    print("\n=== 数据集摘要 ===")
    print(f"股票数量: {training_data['factors'].shape[1]}")
    print(f"时间期数: {training_data['factors'].shape[0]}")
    print(f"因子数量: {training_data['factors'].shape[2]}")
    print(f"平均连接度: {training_data['adjacency_matrices'].mean():.4f}") 