#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTGNN数据预处理器
整合股价数据和Barra因子数据，为ASTGNN项目提供标准化的训练数据
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ASTGNNDataPreprocessor:
    """ASTGNN数据预处理器"""
    
    def __init__(self, config: Optional[Dict] = None, use_gpu: Optional[bool] = None):
        """初始化预处理器"""
        self.config = config or self._get_default_config()
        
        # GPU加速配置
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            torch.cuda.empty_cache()
            logger.info(f"GPU加速已启用 - 设备: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info("使用CPU处理")
        
        # 文件路径
        self.stock_file = self.config.get('stock_file', 'stock_price_vol_d.txt')
        self.barra_file = self.config.get('barra_file', 'barra_Exposure(2).')
        
        # 数据缓存
        self.stock_data = None
        self.barra_data = None
        self.merged_data = None
        self.processed_data = None
        
        # 标准化器
        self.factor_scaler = StandardScaler()
        self.return_scaler = RobustScaler()
        
        # GPU批处理大小配置
        self.gpu_batch_size = self.config.get('gpu_batch_size', 1000)
        
        logger.info("数据预处理器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置 - 针对7年专业回测优化"""
        return {
            'sequence_length': 20,  # 适度增加序列长度以捕获更多模式
            'prediction_horizon': 10,  # 未来10日收益率预测
            'min_stock_history': 252,  # 最少1年交易日历史数据
            'factor_standardization': True,
            'return_standardization': True,
            'remove_outliers': True,  # 启用异常值移除以提高回测质量
            'outlier_threshold': 5.0,
            'min_correlation_threshold': 0.02,
            'adjacency_threshold': 0.15,
            'data_split_ratio': [0.7, 0.15, 0.15],  # 增加训练集比例用于长期数据
            'random_seed': 42,
            # 7年专业回测相关配置
            'full_backtest_start_date': '2017-01-01',  # 7年回测起始日期
            'full_backtest_end_date': '2024-04-30',    # 回测结束日期
            'training_start_date': '2017-01-01',       # 训练开始日期
            'training_end_date': '2023-12-28',         # 训练结束日期（回测前一天）
            'backtest_start_date': '2023-12-29',       # 实际回测开始
            'backtest_end_date': '2024-04-30',         # 实际回测结束
            'min_periods_for_backtest': 1000,          # 7年数据需要更多时间点
            'rebalance_frequency': 10,                 # 每10个交易日调仓
            'annual_periods': 252,                     # 年化期数
            # GPU加速相关配置
            'gpu_batch_size': 2000,
            'use_gpu_for_correlation': True,
            'use_gpu_for_technical_indicators': True,
            'gpu_memory_fraction': 0.9
        }
    
    def load_stock_data(self, sample_size: Optional[int] = None, 
                        target_stocks: Optional[List[str]] = None,
                        target_date_range: Optional[Tuple[str, str]] = None) -> Optional[pd.DataFrame]:
        """加载股价数据"""
        logger.info("开始加载股价数据")
        
        try:
            # 使用feather格式读取
            df = pd.read_feather(self.stock_file)
            logger.info(f"成功加载股价数据: {df.shape}")
            
            # 数据清理
            df['date'] = pd.to_datetime(df['date'])
            
            # 如果指定了目标股票，优先过滤
            if target_stocks:
                df = df[df['StockID'].isin(target_stocks)]
                logger.info(f"按目标股票过滤后: {df.shape}")
            
            # 如果指定了目标日期范围，过滤日期
            if target_date_range:
                start_date, end_date = target_date_range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                logger.info(f"按目标日期过滤后: {df.shape}")
            
            # 专业回测优化: 优先保证时间完整性，仅在股票维度进行采样
            if sample_size and len(df) > sample_size:
                logger.info(f"数据量较大({len(df):,}行)，进行优化采样以保证时间完整性")
                
                # 如果有目标时间范围，优先保证该时间范围的完整性
                if target_date_range:
                    start_target, end_target = target_date_range
                    target_period_data = df[
                        (df['date'] >= start_target) & 
                        (df['date'] <= end_target)
                    ]
                    logger.info(f"目标时间范围({start_target}至{end_target})数据量: {len(target_period_data):,}行")
                
                    # 如果目标时间范围的数据小于sample_size，直接使用
                    if len(target_period_data) <= sample_size:
                        df = target_period_data
                        logger.info(f"使用目标时间范围完整数据: {len(df):,}行")
                    else:
                        # 目标时间范围数据过多，在股票维度采样但保证时间完整性
                        all_dates = sorted(target_period_data['date'].unique())
                        all_stocks = list(target_period_data['StockID'].unique())
                        
                        # 计算可以保留多少只股票
                        avg_points_per_stock = len(target_period_data) / len(all_stocks)
                        max_stocks = min(len(all_stocks), int(sample_size / avg_points_per_stock))
                        
                        # 随机选择股票但保证每个时间点都有代表性
                        import random
                        random.seed(42)
                        selected_stocks = random.sample(all_stocks, max_stocks)
                        
                        df = target_period_data[target_period_data['StockID'].isin(selected_stocks)]
                        logger.info(f"保证时间完整性的股票采样: {len(df):,}行, {len(selected_stocks)}只股票")
                else:
                    # 没有目标时间范围，使用传统的均匀采样但优化为时间优先
                    logger.info("没有目标时间范围，使用时间优先的均匀采样")
                    all_stocks = list(df['StockID'].unique())
                    avg_points_per_stock = len(df) / len(all_stocks)
                    max_stocks = min(len(all_stocks), int(sample_size / avg_points_per_stock))
                    
                    import random
                    random.seed(42)
                    selected_stocks = random.sample(all_stocks, max_stocks)
                    df = df[df['StockID'].isin(selected_stocks)]
                    logger.info(f"时间优先采样: {len(df):,}行, {len(selected_stocks)}只股票")
            
            # 基本统计信息
            date_range = (df['date'].min(), df['date'].max())
            unique_stocks = df['StockID'].nunique()
            missing_values = df.isnull().sum().sum()
            
            logger.info(f"时间范围: {date_range[0]} 到 {date_range[1]}")
            logger.info(f"股票数量: {unique_stocks:,}")
            logger.info(f"缺失值总数: {missing_values}")
            
            self.stock_data = df
            return df
            
        except Exception as e:
            logger.error(f"加载股价数据失败: {str(e)}")
            return None
    
    def load_barra_data(self, sample_size: Optional[int] = None,
                        target_stocks: Optional[List[str]] = None,
                        target_date_range: Optional[Tuple[str, str]] = None) -> Optional[pd.DataFrame]:
        """加载Barra因子数据"""
        logger.info("开始加载Barra因子数据")
        
        try:
            # 使用parquet格式读取
            df = pd.read_parquet(self.barra_file)
            logger.info(f"成功加载Barra数据: {df.shape}")
            
            # 数据清理和重命名
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.rename(columns={
                '日期': 'date',
                '股票代码': 'StockID'
            })
            
            # 如果指定了目标股票，优先过滤
            if target_stocks:
                df = df[df['StockID'].isin(target_stocks)]
                logger.info(f"按目标股票过滤后: {df.shape}")
            
            # 如果指定了目标日期范围，过滤日期
            if target_date_range:
                start_date, end_date = target_date_range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                logger.info(f"按目标日期过滤后: {df.shape}")
            
            # 专业回测优化: 优先保证时间完整性，仅在股票维度进行采样
            if sample_size and len(df) > sample_size:
                logger.info(f"Barra数据量较大({len(df):,}行)，进行优化采样以保证时间完整性")
                
                # 如果有目标时间范围，优先保证该时间范围的完整性
                if target_date_range:
                    start_target, end_target = target_date_range
                    target_period_data = df[
                        (df['date'] >= start_target) & 
                        (df['date'] <= end_target)
                    ]
                    logger.info(f"Barra目标时间范围({start_target}至{end_target})数据量: {len(target_period_data):,}行")
                    
                    # 如果目标时间范围的数据小于sample_size，直接使用
                    if len(target_period_data) <= sample_size:
                        df = target_period_data
                        logger.info(f"使用Barra目标时间范围完整数据: {len(df):,}行")
                    else:
                        # 目标时间范围数据过多，在股票维度采样但保证时间完整性
                        all_stocks = list(target_period_data['StockID'].unique())
                        avg_points_per_stock = len(target_period_data) / len(all_stocks)
                        max_stocks = min(len(all_stocks), int(sample_size / avg_points_per_stock))
                        
                        # 随机选择股票但保证每个时间点都有代表性
                        import random
                        random.seed(42)
                        selected_stocks = random.sample(all_stocks, max_stocks)
                        
                        df = target_period_data[target_period_data['StockID'].isin(selected_stocks)]
                        logger.info(f"Barra保证时间完整性的股票采样: {len(df):,}行, {len(selected_stocks)}只股票")
                else:
                    # 没有目标时间范围，使用时间优先的均匀采样
                    logger.info("Barra没有目标时间范围，使用时间优先的均匀采样")
                    all_stocks = list(df['StockID'].unique())
                    avg_points_per_stock = len(df) / len(all_stocks)
                    max_stocks = min(len(all_stocks), int(sample_size / avg_points_per_stock))
                    
                    import random
                    random.seed(42)
                    selected_stocks = random.sample(all_stocks, max_stocks)
                    df = df[df['StockID'].isin(selected_stocks)]
                    logger.info(f"Barra时间优先采样: {len(df):,}行, {len(selected_stocks)}只股票")
            
            # 基本统计信息
            date_range = (df['date'].min(), df['date'].max())
            unique_stocks = df['StockID'].nunique()
            factor_count = len([col for col in df.columns if col.startswith('Exposure_')])
            
            logger.info(f"时间范围: {date_range[0]} 到 {date_range[1]}")
            logger.info(f"股票数量: {unique_stocks:,}")
            logger.info(f"因子数量: {factor_count}")
            
            self.barra_data = df
            return df
            
        except Exception as e:
            logger.error(f"加载Barra数据失败: {str(e)}")
            return None
    
    def merge_datasets(self, date_range: Optional[Tuple[str, str]] = None) -> Optional[pd.DataFrame]:
        """合并股价数据和Barra因子数据"""
        logger.info("开始合并数据集")
        
        if self.stock_data is None or self.barra_data is None:
            logger.error("请先加载股价数据和Barra数据")
            return None
        
        # 时间范围过滤
        if date_range:
            start_date, end_date = date_range
            stock_filtered = self.stock_data[
                (self.stock_data['date'] >= start_date) & 
                (self.stock_data['date'] <= end_date)
            ].copy()
            barra_filtered = self.barra_data[
                (self.barra_data['date'] >= start_date) & 
                (self.barra_data['date'] <= end_date)
            ].copy()
            logger.info(f"应用时间过滤: {start_date} 到 {end_date}")
        else:
            stock_filtered = self.stock_data.copy()
            barra_filtered = self.barra_data.copy()
        
        # 合并数据
        merged = pd.merge(
            stock_filtered, 
            barra_filtered,
            on=['date', 'StockID'],
            how='inner'
        )
        
        logger.info(f"合并完成: {merged.shape}")
        if len(merged) > 0:
            logger.info(f"合并后时间范围: {merged['date'].min()} 到 {merged['date'].max()}")
            logger.info(f"合并后股票数量: {merged['StockID'].nunique():,}")
        
        self.merged_data = merged
        return merged
    
    def calculate_returns_and_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算收益率和技术指标"""
        logger.info("计算收益率和技术指标")
        
        # 重置索引并按股票分组排序
        data = data.reset_index(drop=True)
        data = data.sort_values(['StockID', 'date'])
        
        # 计算多期收益率
        return_periods = [1, 5, 10, 20]
        for period in return_periods:
            col_name = f'return_{period}d'
            data[col_name] = data.groupby('StockID')['close'].pct_change(period)
        
        # 计算对数收益率
        log_returns = data.groupby('StockID')['close'].transform(lambda x: np.log(x / x.shift(1)))
        data['log_return'] = log_returns
        
        # 计算波动率指标
        data['volatility_5d'] = data.groupby('StockID')['log_return'].transform(lambda x: x.rolling(5).std())
        data['volatility_20d'] = data.groupby('StockID')['log_return'].transform(lambda x: x.rolling(20).std())
        
        # 计算价格技术指标
        data['price_ma_5'] = data.groupby('StockID')['close'].transform(lambda x: x.rolling(5).mean())
        data['price_ma_20'] = data.groupby('StockID')['close'].transform(lambda x: x.rolling(20).mean())
        data['price_ratio_ma5'] = data['close'] / data['price_ma_5']
        data['price_ratio_ma20'] = data['close'] / data['price_ma_20']
        
        # 计算成交量指标
        data['volume_ma_5'] = data.groupby('StockID')['vol'].transform(lambda x: x.rolling(5).mean())
        data['volume_ratio'] = data['vol'] / data['volume_ma_5']
        
        # 计算momentum指标
        data['momentum_5d'] = data.groupby('StockID')['close'].transform(lambda x: x / x.shift(5) - 1)
        data['momentum_20d'] = data.groupby('StockID')['close'].transform(lambda x: x / x.shift(20) - 1)
        
        logger.info("收益率和技术指标计算完成")
        return data
    
    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """移除异常值"""
        if not self.config['remove_outliers']:
            return data
        
        logger.info("开始移除异常值")
        initial_count = len(data)
        
        # 获取数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        threshold = self.config['outlier_threshold']
        
        # 对每个数值列移除异常值
        for col in numeric_cols:
            if col in ['date', 'StockID']:
                continue
            
            # 使用Z-score方法
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data = data[z_scores <= threshold]
        
        final_count = len(data)
        removed_count = initial_count - final_count
        
        logger.info(f"异常值移除完成: 移除 {removed_count} 行 ({removed_count/initial_count*100:.2f}%)")
        return data
    
    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        logger.info("开始特征标准化")
        
        # 获取因子列和收益率列
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        return_cols = [col for col in data.columns if 'return' in col and col != 'turnoverrate']
        
        # 标准化因子
        if self.config['factor_standardization'] and factor_cols:
            data[factor_cols] = self.factor_scaler.fit_transform(data[factor_cols].fillna(0))
            logger.info(f"因子标准化完成: {len(factor_cols)} 个因子")
        
        # 标准化收益率
        if self.config['return_standardization'] and return_cols:
            data[return_cols] = self.return_scaler.fit_transform(data[return_cols].fillna(0))
            logger.info(f"收益率标准化完成: {len(return_cols)} 个收益率指标")
        
        return data
    
    def filter_stocks_by_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """根据历史数据长度过滤股票"""
        min_history = self.config['min_stock_history']
        
        # 计算每只股票的历史数据长度
        stock_counts = data.groupby('StockID').size()
        valid_stocks = stock_counts[stock_counts >= min_history].index
        
        filtered_data = data[data['StockID'].isin(valid_stocks)]
        
        logger.info(f"历史数据过滤: 保留 {len(valid_stocks)} 只股票 (最少 {min_history} 天数据)")
        return filtered_data
    
    def create_sequences(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """创建时间序列数据"""
        logger.info("创建时间序列数据")
        
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        # 获取因子列
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        
        # 按日期排序
        data = data.sort_values(['date', 'StockID'])
        
        # 获取所有日期和股票
        dates = sorted(data['date'].unique())
        stocks = sorted(data['StockID'].unique())
        
        logger.info(f"时间期数: {len(dates)}, 股票数量: {len(stocks)}, 因子数量: {len(factor_cols)}")
        
        # 为每个因子分别创建pivot表，然后合并
        factor_arrays = []
        for factor in factor_cols:
            factor_pivot = data.pivot_table(
                index='date', 
                columns='StockID', 
                values=factor,
                aggfunc='first'
            )
            factor_arrays.append(factor_pivot.values)
        
        # 堆叠所有因子：[num_factors, num_dates, num_stocks]
        factor_array = np.stack(factor_arrays, axis=0)
        # 转置为：[num_dates, num_stocks, num_factors] 
        factor_array = np.transpose(factor_array, (1, 2, 0))
        
        return_data = data.pivot_table(
            index='date',
            columns='StockID',
            values='return_1d',
            aggfunc='first'
        )
        return_array = return_data.values
        
        # 处理NaN值
        factor_array = np.nan_to_num(factor_array, nan=0.0)
        return_array = np.nan_to_num(return_array, nan=0.0)
        
        logger.info(f"factor_array形状: {factor_array.shape}")
        logger.info(f"return_array形状: {return_array.shape}")
        
        # 创建序列
        sequences = []
        targets = []
        factor_targets = []  # 新增：未来因子作为目标
        
        for i in range(len(dates) - sequence_length - prediction_horizon + 1):
            # 输入序列
            seq_factors = factor_array[i:i+sequence_length]
            seq_returns = return_array[i:i+sequence_length]
            
            # 目标序列：使用未来时点的收益率
            target_returns = return_array[i+sequence_length:i+sequence_length+prediction_horizon]
            
            # 新增：未来因子值作为多因子预测目标
            target_factors = factor_array[i+sequence_length:i+sequence_length+prediction_horizon]
            
            sequences.append({
                'factors': seq_factors,
                'returns': seq_returns
            })
            targets.append(target_returns)
            factor_targets.append(target_factors)
        
        # 转换为tensor
        factor_sequences = torch.tensor([seq['factors'] for seq in sequences], dtype=torch.float32)
        return_sequences = torch.tensor([seq['returns'] for seq in sequences], dtype=torch.float32)
        target_sequences = torch.tensor(targets, dtype=torch.float32)
        factor_target_sequences = torch.tensor(factor_targets, dtype=torch.float32)  # 新增
        
        logger.info(f"序列创建完成: {factor_sequences.shape}")
        logger.info(f"因子目标序列形状: {factor_target_sequences.shape}")
        
        return {
            'factor_sequences': factor_sequences,
            'return_sequences': return_sequences,
            'target_sequences': target_sequences,
            'factor_target_sequences': factor_target_sequences,  # 新增：多因子预测目标
            'factor_names': factor_cols,
            'stock_ids': stocks,
            'dates': dates[sequence_length:len(dates)-prediction_horizon+1]
        }
    
    def create_adjacency_matrices(self, factor_sequences: torch.Tensor) -> torch.Tensor:
        """创建邻接矩阵"""
        logger.info("创建邻接矩阵")
        logger.info(f"factor_sequences形状: {factor_sequences.shape}")
        
        # 检查维度
        if len(factor_sequences.shape) == 3:
            # 形状为 [num_sequences, seq_len, num_features] 其中 num_features = num_stocks * num_factors
            num_sequences, seq_len, num_features = factor_sequences.shape
            
            # 需要重新构造以获得股票数量
            # 假设因子数量为10 (从配置或数据中获取)
            num_factors = len([col for col in self.merged_data.columns if col.startswith('Exposure_')]) if hasattr(self, 'merged_data') else 10
            num_stocks = num_features // num_factors
            
            # 重塑为 [num_sequences, seq_len, num_stocks, num_factors]
            factor_sequences = factor_sequences.view(num_sequences, seq_len, num_stocks, num_factors)
            logger.info(f"重塑后factor_sequences形状: {factor_sequences.shape}")
            
        elif len(factor_sequences.shape) == 4:
            num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
        else:
            logger.error(f"不支持的factor_sequences维度: {factor_sequences.shape}")
            return torch.zeros(1, 1, 1, 1)
        
        adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks)
        threshold = self.config['adjacency_threshold']
        
        for seq_idx in range(num_sequences):
            for t in range(seq_len):
                factors_t = factor_sequences[seq_idx, t]  # [stocks, factors]
                
                # 计算相关系数矩阵
                if not torch.isnan(factors_t).all() and factors_t.shape[0] > 1:
                    try:
                        corr_matrix = torch.corrcoef(factors_t)
                        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
                        
                        # 转换为邻接矩阵
                        adj_matrix = torch.abs(corr_matrix)
                        adj_matrix[adj_matrix < threshold] = 0
                        adj_matrix.fill_diagonal_(1.0)
                        
                        adj_matrices[seq_idx, t] = adj_matrix
                    except Exception as e:
                        logger.warning(f"计算相关矩阵失败 seq_idx={seq_idx}, t={t}: {str(e)}")
                        # 使用单位矩阵作为默认值
                        adj_matrices[seq_idx, t] = torch.eye(num_stocks)
                else:
                    # 使用单位矩阵作为默认值
                    adj_matrices[seq_idx, t] = torch.eye(num_stocks)
        
        logger.info(f"邻接矩阵创建完成: {adj_matrices.shape}")
        return adj_matrices
    
    def split_data(self, sequences: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """分割训练、验证和测试数据"""
        logger.info("分割数据集")
        
        split_ratios = self.config['data_split_ratio']
        total_sequences = sequences['factor_sequences'].shape[0]
        
        train_size = int(total_sequences * split_ratios[0])
        val_size = int(total_sequences * split_ratios[1])
        
        # 按时间顺序分割
        train_data = {
            'factor_sequences': sequences['factor_sequences'][:train_size],
            'return_sequences': sequences['return_sequences'][:train_size],
            'target_sequences': sequences['target_sequences'][:train_size],
            'factor_target_sequences': sequences.get('factor_target_sequences', sequences['target_sequences'])[:train_size]
        }
        
        val_data = {
            'factor_sequences': sequences['factor_sequences'][train_size:train_size+val_size],
            'return_sequences': sequences['return_sequences'][train_size:train_size+val_size],
            'target_sequences': sequences['target_sequences'][train_size:train_size+val_size],
            'factor_target_sequences': sequences.get('factor_target_sequences', sequences['target_sequences'])[train_size:train_size+val_size]
        }
        
        test_data = {
            'factor_sequences': sequences['factor_sequences'][train_size+val_size:],
            'return_sequences': sequences['return_sequences'][train_size+val_size:],
            'target_sequences': sequences['target_sequences'][train_size+val_size:],
            'factor_target_sequences': sequences.get('factor_target_sequences', sequences['target_sequences'])[train_size+val_size:]
        }
        
        logger.info(f"数据分割完成 - 训练: {train_data['factor_sequences'].shape[0]}, "
                   f"验证: {val_data['factor_sequences'].shape[0]}, "
                   f"测试: {test_data['factor_sequences'].shape[0]}")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def save_processed_data(self, data: Dict, filename: str = 'processed_astgnn_data.pt'):
        """保存处理后的数据"""
        logger.info(f"保存处理后的数据到: {filename}")
        
        try:
            torch.save({
                'data': data,
                'config': self.config,
                'factor_scaler': self.factor_scaler,
                'return_scaler': self.return_scaler
            }, filename)
            logger.info("数据保存成功")
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
    
    def run_preprocessing_pipeline(self, 
                                 stock_sample_size: int = 200000,
                                 barra_sample_size: int = 180000,
                                 date_range: Tuple[str, str] = ('2023-01-01', '2024-06-30')) -> Optional[Dict]:
        """运行完整的预处理流程"""
        logger.info("启动ASTGNN数据预处理流程")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 协调采样策略：先加载Barra数据确定可用股票，再加载对应的股价数据
        logger.info("第一步：加载Barra数据以确定可用股票")
        
        # 1. 先加载Barra数据（在目标时间范围内）
        barra_df = self.load_barra_data(
            sample_size=barra_sample_size,
            target_date_range=date_range
        )
        
        if barra_df is None:
            logger.error("Barra数据加载失败，流程终止")
            return None
        
        # 获取Barra数据中的股票列表
        available_stocks = list(barra_df['StockID'].unique())
        logger.info(f"Barra数据中可用股票: {len(available_stocks)} 只")
        
        # 2. 加载对应的股价数据
        logger.info("第二步：加载对应股票的股价数据")
        stock_df = self.load_stock_data(
            sample_size=stock_sample_size,
            target_stocks=available_stocks,
            target_date_range=date_range
        )
        
        if stock_df is None or barra_df is None:
            logger.error("数据加载失败，流程终止")
            return None
        
        # 2. 合并数据集
        merged_df = self.merge_datasets(date_range)
        if merged_df is None or len(merged_df) == 0:
            logger.error("数据合并失败或合并后数据为空，流程终止")
            return None
        
        logger.info(f"步骤2后数据量: {len(merged_df)} 行")
        
        # 3. 计算收益率和特征
        merged_df = self.calculate_returns_and_features(merged_df)
        logger.info(f"步骤3后数据量: {len(merged_df)} 行")
        
        # 4. 移除异常值
        merged_df = self.remove_outliers(merged_df)
        logger.info(f"步骤4后数据量: {len(merged_df)} 行")
        
        # 检查数据是否为空
        if len(merged_df) == 0:
            logger.error("异常值移除后数据为空，调整异常值阈值")
            # 重新加载数据并跳过异常值移除
            merged_df = self.merge_datasets(date_range)
            merged_df = self.calculate_returns_and_features(merged_df)
            logger.info(f"跳过异常值移除后数据量: {len(merged_df)} 行")
        
        # 5. 过滤股票 - 降低最小历史要求
        original_min_history = self.config['min_stock_history']
        self.config['min_stock_history'] = min(20, len(merged_df) // 10)  # 动态调整
        merged_df = self.filter_stocks_by_history(merged_df)
        logger.info(f"步骤5后数据量: {len(merged_df)} 行")
        
        # 如果数据仍然为空，进一步降低要求
        if len(merged_df) == 0:
            logger.warning("历史数据过滤后数据为空，降低最小历史要求")
            self.config['min_stock_history'] = 5
            merged_df = self.filter_stocks_by_history(self.calculate_returns_and_features(self.merge_datasets(date_range)))
            logger.info(f"降低要求后数据量: {len(merged_df)} 行")
        
        # 恢复原始配置
        self.config['min_stock_history'] = original_min_history
        
        # 最终检查
        if len(merged_df) == 0:
            logger.error("所有过滤步骤后数据为空，流程终止")
            return None
        
        # 6. 特征标准化
        merged_df = self.standardize_features(merged_df)
        logger.info(f"步骤6后数据量: {len(merged_df)} 行")
        
        # 7. 创建时间序列
        sequences = self.create_sequences(merged_df)
        
        # 8. 创建邻接矩阵
        adj_matrices = self.create_adjacency_matrices(sequences['factor_sequences'])
        sequences['adjacency_matrices'] = adj_matrices
        
        # 9. 分割数据集
        split_data = self.split_data(sequences)
        
        # 10. 保存处理后的数据
        final_data = {
            'sequences': split_data,
            'metadata': {
                'factor_names': sequences['factor_names'],
                'stock_ids': sequences['stock_ids'],
                'dates': sequences['dates'],
                'config': self.config
            }
        }
        
        self.save_processed_data(final_data)
        
        end_time = time.time()
        
        # 输出处理结果摘要
        train_shape = split_data['train']['factor_sequences'].shape
        logger.info("\n处理完成摘要:")
        logger.info(f"总处理时间: {end_time - start_time:.2f} 秒")
        logger.info(f"训练数据形状: {train_shape}")
        logger.info(f"因子数量: {len(sequences['factor_names'])}")
        logger.info(f"股票数量: {len(sequences['stock_ids'])}")
        logger.info(f"时间序列长度: {self.config['sequence_length']}")
        logger.info(f"预测时间跨度: {self.config['prediction_horizon']}")
        
        logger.info("\n使用建议:")
        logger.info("1. 数据已保存为 'processed_astgnn_data.pt'")
        logger.info("2. 可直接用于ASTGNN模型训练")
        logger.info("3. 包含完整的训练/验证/测试分割")
        logger.info("4. 已进行标准化和异常值处理")
        
        return final_data
    
    def diagnose_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """诊断数据质量"""
        logger.info("开始数据质量诊断")
        
        diagnosis = {
            'data_shape': data.shape,
            'date_range': (data['date'].min(), data['date'].max()),
            'unique_stocks': data['StockID'].nunique(),
            'total_stocks': len(data['StockID'].unique()),
            'missing_values': {},
            'data_coverage': {},
            'factor_analysis': {},
            'time_series_analysis': {}
        }
        
        # 检查缺失值
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = missing_count / len(data) * 100
            diagnosis['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # 检查因子列
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        diagnosis['factor_analysis'] = {
            'factor_count': len(factor_cols),
            'factor_names': factor_cols,
            'factor_stats': {}
        }
        
        # 因子统计信息
        for factor in factor_cols:
            factor_data = data[factor].dropna()
            diagnosis['factor_analysis']['factor_stats'][factor] = {
                'mean': factor_data.mean(),
                'std': factor_data.std(),
                'min': factor_data.min(),
                'max': factor_data.max(),
                'skewness': factor_data.skew(),
                'kurtosis': factor_data.kurtosis()
            }
        
        # 时间序列分析
        dates = sorted(data['date'].unique())
        diagnosis['time_series_analysis'] = {
            'total_dates': len(dates),
            'date_gaps': self._find_date_gaps(dates),
            'stocks_per_date': data.groupby('date')['StockID'].nunique().describe().to_dict(),
            'observations_per_stock': data.groupby('StockID').size().describe().to_dict()
        }
        
        # 数据覆盖率分析
        total_expected = len(dates) * diagnosis['unique_stocks']
        actual_observations = len(data)
        diagnosis['data_coverage'] = {
            'expected_observations': total_expected,
            'actual_observations': actual_observations,
            'coverage_rate': actual_observations / total_expected * 100 if total_expected > 0 else 0
        }
        
        logger.info("数据质量诊断完成")
        return diagnosis
    
    def _find_date_gaps(self, dates: List) -> List[Tuple]:
        """查找日期间隙"""
        gaps = []
        dates = pd.to_datetime(dates)
        
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i-1]).days
            if gap_days > 7:  # 假设正常间隔不超过7天
                gaps.append((dates[i-1], dates[i], gap_days))
        
        return gaps
    
    def validate_data_consistency(self, stock_data: pd.DataFrame, barra_data: pd.DataFrame) -> Dict[str, Any]:
        """验证两个数据集的一致性"""
        logger.info("验证数据一致性")
        
        validation = {
            'stock_date_range': (stock_data['date'].min(), stock_data['date'].max()),
            'barra_date_range': (barra_data['date'].min(), barra_data['date'].max()),
            'common_stocks': set(stock_data['StockID']).intersection(set(barra_data['StockID'])),
            'stock_only': set(stock_data['StockID']) - set(barra_data['StockID']),
            'barra_only': set(barra_data['StockID']) - set(stock_data['StockID']),
            'overlap_analysis': {}
        }
        
        # 分析重叠情况
        common_stocks = validation['common_stocks']
        validation['overlap_analysis'] = {
            'common_stock_count': len(common_stocks),
            'stock_only_count': len(validation['stock_only']),
            'barra_only_count': len(validation['barra_only']),
            'overlap_rate': len(common_stocks) / max(len(set(stock_data['StockID'])), len(set(barra_data['StockID']))) * 100
        }
        
        # 日期重叠分析
        stock_dates = set(stock_data['date'])
        barra_dates = set(barra_data['date'])
        common_dates = stock_dates.intersection(barra_dates)
        
        validation['date_overlap'] = {
            'common_dates': len(common_dates),
            'stock_only_dates': len(stock_dates - barra_dates),
            'barra_only_dates': len(barra_dates - stock_dates),
            'date_overlap_rate': len(common_dates) / max(len(stock_dates), len(barra_dates)) * 100
        }
        
        logger.info("数据一致性验证完成")
        return validation
    
    def print_diagnosis_report(self, diagnosis: Dict[str, Any]):
        """打印诊断报告"""
        print("\n" + "="*60)
        print("数据质量诊断报告")
        print("="*60)
        
        print(f"数据形状: {diagnosis['data_shape']}")
        print(f"时间范围: {diagnosis['date_range'][0]} 到 {diagnosis['date_range'][1]}")
        print(f"股票数量: {diagnosis['unique_stocks']}")
        print(f"因子数量: {diagnosis['factor_analysis']['factor_count']}")
        
        print(f"\n数据覆盖率: {diagnosis['data_coverage']['coverage_rate']:.2f}%")
        print(f"期望观测数: {diagnosis['data_coverage']['expected_observations']:,}")
        print(f"实际观测数: {diagnosis['data_coverage']['actual_observations']:,}")
        
        print(f"\n时间序列分析:")
        print(f"  总日期数: {diagnosis['time_series_analysis']['total_dates']}")
        print(f"  日期间隙: {len(diagnosis['time_series_analysis']['date_gaps'])} 个")
        
        print(f"\n缺失值情况:")
        high_missing_cols = [(k, v['percentage']) for k, v in diagnosis['missing_values'].items() 
                           if v['percentage'] > 10]
        if high_missing_cols:
            for col, pct in high_missing_cols[:5]:  # 显示前5个高缺失率列
                print(f"  {col}: {pct:.2f}%")
        else:
            print("  所有列缺失率都低于10%")
        
        print("="*60)

    def gpu_accelerated_calculate_returns_and_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU加速版本的收益率和技术指标计算"""
        logger.info("GPU加速计算收益率和技术指标")
        
        if not self.use_gpu or len(data) < 10000:
            return self.calculate_returns_and_features(data)
        
        data = data.reset_index(drop=True)
        data = data.sort_values(['StockID', 'date'])
        
        results = []
        stock_groups = data.groupby('StockID')
        
        # 批处理股票
        stock_list = list(stock_groups.groups.keys())
        batch_size = min(self.gpu_batch_size // 10, len(stock_list))  # 调整批量大小
        
        for i in range(0, len(stock_list), batch_size):
            batch_stocks = stock_list[i:i+batch_size]
            
            # 处理当前批次的股票
            for stock_id in batch_stocks:
                stock_data = stock_groups.get_group(stock_id).copy()
                if len(stock_data) < 2:
                    continue
                
                # 转换到GPU进行计算
                stock_data_gpu = self._gpu_process_single_stock(stock_data)
                results.append(stock_data_gpu)
            
            # 定期清理GPU内存
            if i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        final_data = pd.concat(results, ignore_index=True) if results else data
        logger.info("GPU加速收益率和技术指标计算完成")
        return final_data

    def _gpu_process_single_stock(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """使用GPU处理单只股票的技术指标"""
        # 提取价格和成交量数据到GPU
        prices = torch.tensor(stock_data['close'].values, device=self.device, dtype=torch.float32)
        volumes = torch.tensor(stock_data['vol'].values, device=self.device, dtype=torch.float32)
        
        # GPU计算各种收益率
        stock_data['return_1d'] = self._gpu_pct_change(prices, 1).cpu().numpy()
        stock_data['return_5d'] = self._gpu_pct_change(prices, 5).cpu().numpy()
        stock_data['return_10d'] = self._gpu_pct_change(prices, 10).cpu().numpy()
        stock_data['return_20d'] = self._gpu_pct_change(prices, 20).cpu().numpy()
        
        # 对数收益率
        log_returns = torch.cat([
            torch.tensor([0.0], device=self.device),
            torch.log(prices[1:] / prices[:-1])
        ])
        stock_data['log_return'] = log_returns.cpu().numpy()
        
        # 波动率指标
        stock_data['volatility_5d'] = self._gpu_rolling_std(log_returns, 5).cpu().numpy()
        stock_data['volatility_20d'] = self._gpu_rolling_std(log_returns, 20).cpu().numpy()
        
        # 移动平均线
        ma_5 = self._gpu_rolling_mean(prices, 5)
        ma_20 = self._gpu_rolling_mean(prices, 20)
        vol_ma_5 = self._gpu_rolling_mean(volumes, 5)
        
        stock_data['price_ma_5'] = ma_5.cpu().numpy()
        stock_data['price_ma_20'] = ma_20.cpu().numpy()
        stock_data['price_ratio_ma5'] = (prices / ma_5).cpu().numpy()
        stock_data['price_ratio_ma20'] = (prices / ma_20).cpu().numpy()
        stock_data['volume_ma_5'] = vol_ma_5.cpu().numpy()
        stock_data['volume_ratio'] = (volumes / vol_ma_5).cpu().numpy()
        
        # 动量指标
        stock_data['momentum_5d'] = self._gpu_momentum(prices, 5).cpu().numpy()
        stock_data['momentum_20d'] = self._gpu_momentum(prices, 20).cpu().numpy()
        
        return stock_data

    def _gpu_pct_change(self, prices: torch.Tensor, periods: int) -> torch.Tensor:
        """GPU版本的百分比变化计算"""
        if len(prices) <= periods:
            return torch.zeros_like(prices)
        
        shifted = torch.roll(prices, periods)
        shifted[:periods] = prices[0]  # 填充前几期
        returns = (prices - shifted) / shifted
        returns[:periods] = 0.0  # 前几期设为0
        return returns

    def _gpu_rolling_mean(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """GPU版本的滚动平均计算"""
        if len(data) < window:
            return torch.full_like(data, data.mean())
        
        # 使用1D卷积实现滚动平均
        data_expanded = data.unsqueeze(0).unsqueeze(0)  # [1, 1, length]
        kernel = torch.ones(1, 1, window, device=self.device) / window
        
        # 左填充以保持长度
        padded = torch.nn.functional.pad(data_expanded, (window-1, 0))
        result = torch.nn.functional.conv1d(padded, kernel)
        
        return result.squeeze()

    def _gpu_rolling_std(self, data: torch.Tensor, window: int) -> torch.Tensor:
        """GPU版本的滚动标准差计算"""
        if len(data) < window:
            return torch.full_like(data, data.std())
        
        # 计算滚动均值
        mean_rolled = self._gpu_rolling_mean(data, window)
        
        # 计算平方的滚动平均
        data_squared = data ** 2
        mean_squared = self._gpu_rolling_mean(data_squared, window)
        
        # 方差 = E[X²] - E[X]²
        variance = mean_squared - mean_rolled ** 2
        variance = torch.clamp(variance, min=0)  # 确保非负
        
        return torch.sqrt(variance)

    def _gpu_momentum(self, prices: torch.Tensor, periods: int) -> torch.Tensor:
        """GPU版本的动量计算"""
        if len(prices) <= periods:
            return torch.zeros_like(prices)
        
        shifted = torch.roll(prices, periods)
        shifted[:periods] = prices[0]
        momentum = (prices / shifted) - 1
        momentum[:periods] = 0.0
        return momentum

    def gpu_accelerated_create_adjacency_matrices(self, factor_sequences: torch.Tensor) -> torch.Tensor:
        """GPU加速版本的邻接矩阵创建"""
        logger.info("GPU加速创建邻接矩阵")
        logger.info(f"factor_sequences形状: {factor_sequences.shape}")
        
        if not self.use_gpu or not self.config.get('use_gpu_for_correlation', True):
            return self.create_adjacency_matrices(factor_sequences)
        
        # 检查维度
        if len(factor_sequences.shape) == 4:
            num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
        else:
            logger.error(f"不支持的factor_sequences维度: {factor_sequences.shape}")
            return torch.zeros(1, 1, 1, 1)
        
        # 移动到GPU
        factor_sequences_gpu = factor_sequences.to(self.device)
        adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks, device=self.device)
        threshold = self.config['adjacency_threshold']
        
        # 批量处理以节省GPU内存
        batch_size = min(50, num_sequences)
        
        for batch_start in range(0, num_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_sequences)
            
            for seq_idx in range(batch_start, batch_end):
                for t in range(seq_len):
                    factors_t = factor_sequences_gpu[seq_idx, t]  # [stocks, factors]
                    
                    if not torch.isnan(factors_t).all() and factors_t.shape[0] > 1:
                        try:
                            # 标准化特征
                            factors_norm = torch.nn.functional.normalize(factors_t, dim=1, eps=1e-8)
                            
                            # 计算相关矩阵 (GPU并行)
                            corr_matrix = torch.mm(factors_norm, factors_norm.t())
                            corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
                            
                            # 转换为邻接矩阵
                            adj_matrix = torch.abs(corr_matrix)
                            adj_matrix[adj_matrix < threshold] = 0
                            adj_matrix.fill_diagonal_(1.0)
                            
                            adj_matrices[seq_idx, t] = adj_matrix
                        except Exception as e:
                            logger.warning(f"GPU相关矩阵计算失败 seq_idx={seq_idx}, t={t}: {str(e)}")
                            adj_matrices[seq_idx, t] = torch.eye(num_stocks, device=self.device)
                    else:
                        adj_matrices[seq_idx, t] = torch.eye(num_stocks, device=self.device)
            
            # 定期清理GPU内存
            if batch_start % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        # 移回CPU
        adj_matrices_cpu = adj_matrices.cpu()
        torch.cuda.empty_cache()
        
        logger.info(f"GPU邻接矩阵创建完成: {adj_matrices_cpu.shape}")
        return adj_matrices_cpu

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        if not self.use_gpu:
            return {"message": "GPU未启用"}
        
        memory_info = {
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'gpu_cached_gb': torch.cuda.memory_reserved() / 1e9,
            'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
        
        return memory_info

    def optimize_gpu_memory(self):
        """优化GPU内存使用"""
        if self.use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU内存优化完成")

    def set_gpu_batch_size(self, batch_size: int):
        """动态调整GPU批处理大小"""
        self.gpu_batch_size = batch_size
        logger.info(f"GPU批处理大小已设置为: {batch_size}")

    def benchmark_gpu_performance(self, test_data_size: int = 50000) -> Dict[str, float]:
        """GPU性能基准测试"""
        logger.info(f"开始GPU性能基准测试 - 数据规模: {test_data_size}")
        
        # 生成测试数据
        num_stocks = 200
        num_periods = test_data_size // num_stocks
        
        test_data = pd.DataFrame({
            'StockID': np.repeat(range(num_stocks), num_periods),
            'date': pd.date_range('2020-01-01', periods=num_periods).repeat(num_stocks),
            'close': np.random.randn(test_data_size) * 100 + 1000,
            'vol': np.random.randint(1000, 100000, test_data_size)
        })
        
        results = {}
        
        # 测试CPU性能
        original_use_gpu = self.use_gpu
        self.use_gpu = False
        
        start_time = time.time()
        cpu_result = self.calculate_returns_and_features(test_data.copy())
        results['cpu_time'] = time.time() - start_time
        
        # 测试GPU性能
        if torch.cuda.is_available():
            self.use_gpu = True
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            gpu_result = self.gpu_accelerated_calculate_returns_and_features(test_data.copy())
            results['gpu_time'] = time.time() - start_time
            
            if results['gpu_time'] > 0:
                results['speedup'] = results['cpu_time'] / results['gpu_time']
            
            # GPU内存使用情况
            memory_info = self.get_gpu_memory_usage()
            results.update(memory_info)
        
        # 恢复原始设置
        self.use_gpu = original_use_gpu
        
        logger.info(f"性能测试完成: {results}")
        return results


def main():
    """主函数"""
    # 针对专业回测优化的配置
    config = {
        'sequence_length': 15,  # 缩短序列长度以获得更多时间点
        'prediction_horizon': 10,  # 未来10日收益率
        'min_stock_history': 30,  # 降低最小历史要求
        'factor_standardization': True,
        'return_standardization': True,
        'remove_outliers': False,  # 关闭异常值移除
        'outlier_threshold': 5.0,  
        'adjacency_threshold': 0.2,  
        'data_split_ratio': [0.6, 0.2, 0.2],
        'random_seed': 42,
        # 专业回测配置
        'backtest_start_date': '2023-12-29',
        'backtest_end_date': '2024-04-30',
        'min_periods_for_backtest': 100
    }
    
    # 初始化预处理器
    preprocessor = ASTGNNDataPreprocessor(config)
    
    # 运行预处理流程 - 使用包含回测期间的完整时间范围
    processed_data = preprocessor.run_preprocessing_pipeline(
        stock_sample_size=200000,
        barra_sample_size=180000,
        date_range=('2023-01-01', '2024-06-30')  # 确保包含完整回测期间
    )
    
    if processed_data:
        logger.info("ASTGNN数据预处理完成！")


if __name__ == '__main__':
    main() 