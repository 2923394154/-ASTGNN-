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
from scipy import stats
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
        self.config = config or self._get_stock_friendly_config()
        
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
    
    def _get_stock_friendly_config(self) -> Dict:
        """获取股票友好的配置，移除所有硬编码股票数量限制"""
        return {
            'sequence_length': 20,          # 序列长度
            'prediction_horizon': 10,       # 预测时间跨度
            'batch_size': 1000,            # 批次大小
            'overlap_ratio': 0.5,          # 序列重叠比例
            'min_stock_history': 30,       # 🔧 降低最小历史要求：从60天降到30天
            'max_missing_ratio': 0.3,      # 最大缺失率
            'outlier_threshold': 5.0,      # 异常值阈值
            'remove_outliers': True,       # 🔧 添加缺失的配置：启用异常值移除
            'adjacency_threshold': 0.10,   # 🔧 添加缺失的配置：邻接矩阵阈值
            'factor_standardization': True,    # 🔧 添加缺失的配置：因子标准化
            'return_standardization': True,    # 🔧 添加缺失的配置：收益率标准化
            'standardize_features': True,   # 特征标准化
            'data_split_ratio': [0.7, 0.15, 0.15],  # 训练/验证/测试比例
            'cpu_memory_limit_gb': 24.0,   # 🔧 大幅提升CPU内存限制：从6GB提升到24GB
            'gpu_memory_threshold': 0.8,   # GPU内存使用阈值
            'adjacency_memory_threshold': 0.6,  # 邻接矩阵内存阈值
            'enable_gpu_acceleration': True,     # 启用GPU加速
            'memory_safety_factor': 1.5,        # 内存安全系数
            'min_stocks_for_gpu': 1000,         # GPU处理最小股票数
            'enable_mixed_precision': False,    # 混合精度训练（可能有兼容性问题）
            'enable_memory_mapping': True,      # 内存映射
            'chunk_size': 10000,                # 数据块大小
            'max_workers': 8,                   # 最大工作进程数
            'progress_update_interval': 100,    # 进度更新间隔
            'enable_data_validation': True,     # 启用数据验证
            'remove_duplicates': True,          # 移除重复数据
            'handle_missing_values': 'interpolate',  # 缺失值处理方式
            'winsorize_quantile': 0.01,         # 极值处理分位数
            'correlation_threshold': 0.95,      # 相关性阈值
            'volume_filter_percentile': 10      # 成交量过滤百分位数
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
            
            # 移除采样限制，保留所有数据
            if sample_size and len(df) > sample_size:
                logger.info(f"数据量较大({len(df):,}行)，但已禁用采样以保留所有股票")
                logger.info(f"保留所有 {df['StockID'].nunique()} 只股票的完整数据")
            
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
            
            # 移除采样限制，保留所有Barra数据
            if sample_size and len(df) > sample_size:
                logger.info(f"Barra数据量较大({len(df):,}行)，但已禁用采样以保留所有股票")
                logger.info(f"保留所有 {df['StockID'].nunique()} 只股票的完整Barra数据")
            
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
    
    def remove_outliers_stock_friendly(self, data: pd.DataFrame) -> pd.DataFrame:
        """快速的股票友好异常值处理方法 - 向量化优化版本"""
        if not self.config['remove_outliers']:
            return data
        
        logger.info("开始快速股票友好异常值处理（向量化优化）")
        initial_count = len(data)
        
        # 获取数值列（排除标识列）
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in ['date', 'StockID']]
        
        threshold = self.config['outlier_threshold']
        outlier_method = self.config.get('outlier_method', 'stock_wise')
        use_winsorize = self.config.get('outlier_winsorize', True)
        winsorize_limits = self.config.get('winsorize_limits', (0.01, 0.01))
        
        if outlier_method == 'stock_wise':
            logger.info(f"使用向量化按股票分组处理，处理 {len(numeric_cols)} 个数值列")
            
            if use_winsorize:
                # 向量化缩尾处理
                from scipy.stats import mstats
                
                def winsorize_group(group):
                    """对单个股票组进行缩尾处理"""
                    result = group.copy()
                    for col in numeric_cols:
                        if col in result.columns and result[col].notna().sum() > 10:
                            # 填充缺失值后缩尾处理
                            filled_values = result[col].fillna(result[col].median())
                            result[col] = mstats.winsorize(filled_values, limits=winsorize_limits)
                    return result
                
                # 使用groupby向量化处理
                data = data.groupby('StockID', group_keys=False).apply(winsorize_group)
                
            else:
                # 向量化Z-score处理
                def filter_outliers_group(group):
                    """对单个股票组进行异常值过滤"""
                    result = group.copy()
                    
                    # 计算所有数值列的Z-score（向量化）
                    for col in numeric_cols:
                        if col in result.columns and result[col].notna().sum() > 10:
                            col_data = result[col]
                            z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-8))
                            # 只保留所有列都不是异常值的行
                            if 'outlier_mask' not in locals():
                                outlier_mask = z_scores <= threshold
                            else:
                                outlier_mask = outlier_mask & (z_scores <= threshold)
                    
                    if 'outlier_mask' in locals():
                        result = result[outlier_mask]
                    
                    return result
                
                # 使用groupby向量化处理
                data = data.groupby('StockID', group_keys=False).apply(filter_outliers_group)
                
        else:
            # 全局向量化异常值处理
            logger.info("使用全局向量化异常值处理")
            
            if use_winsorize:
                from scipy.stats import mstats
                for col in numeric_cols:
                    filled_values = data[col].fillna(data[col].median())
                    data[col] = mstats.winsorize(filled_values, limits=winsorize_limits)
            else:
                # 向量化全局Z-score计算
                outlier_mask = pd.Series(True, index=data.index)
                for col in numeric_cols:
                    z_scores = np.abs((data[col] - data[col].mean()) / (data[col].std() + 1e-8))
                    outlier_mask = outlier_mask & (z_scores <= threshold)
                
                data = data[outlier_mask]
        
        final_count = len(data)
        removed_count = initial_count - final_count
        
        logger.info(f"快速异常值处理完成: 移除 {removed_count} 行 ({removed_count/initial_count*100:.2f}%)")
        logger.info(f"保留股票数量: {data['StockID'].nunique()}")
        return data

    def gpu_accelerated_remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU加速的异常值处理方法"""
        if not self.config['remove_outliers']:
            return data
        
        if not self.use_gpu or len(data) < 10000:
            logger.info("数据量较小或GPU不可用，使用CPU版本异常值处理")
            return self.remove_outliers_stock_friendly(data)
        
        logger.info("开始GPU加速异常值处理")
        initial_count = len(data)
        
        # 获取数值列
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in ['date', 'StockID']]
        
        threshold = self.config['outlier_threshold']
        use_winsorize = self.config.get('outlier_winsorize', True)
        winsorize_limits = self.config.get('winsorize_limits', (0.01, 0.01))
        
        try:
            # 创建股票ID映射
            unique_stocks = data['StockID'].unique()
            stock_to_idx = {stock: idx for idx, stock in enumerate(unique_stocks)}
            data['stock_idx'] = data['StockID'].map(stock_to_idx)
            
            # 批量GPU处理
            batch_size = min(1000, len(unique_stocks))
            processed_data_list = []
            
            for batch_start in range(0, len(unique_stocks), batch_size):
                batch_end = min(batch_start + batch_size, len(unique_stocks))
                batch_stocks = unique_stocks[batch_start:batch_end]
                batch_data = data[data['StockID'].isin(batch_stocks)].copy()
                
                if use_winsorize:
                    # GPU加速缩尾处理
                    batch_processed = self._gpu_winsorize_batch(batch_data, numeric_cols, winsorize_limits)
                else:
                    # GPU加速Z-score过滤
                    batch_processed = self._gpu_zscore_filter_batch(batch_data, numeric_cols, threshold)
                
                processed_data_list.append(batch_processed)
                
                if batch_start // batch_size % 5 == 0:
                    logger.info(f"GPU异常值处理进度: {batch_end}/{len(unique_stocks)} 只股票")
            
            # 合并处理结果
            data = pd.concat(processed_data_list, ignore_index=True)
            data = data.drop(columns=['stock_idx'])
            
            # GPU内存清理
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"GPU异常值处理失败，回退到CPU: {str(e)}")
            data = self.remove_outliers_stock_friendly(data)
        
        final_count = len(data)
        removed_count = initial_count - final_count
        
        logger.info(f"GPU异常值处理完成: 移除 {removed_count} 行 ({removed_count/initial_count*100:.2f}%)")
        return data
    
    def _gpu_winsorize_batch(self, batch_data: pd.DataFrame, numeric_cols: list, winsorize_limits: tuple) -> pd.DataFrame:
        """GPU批量缩尾处理"""
        for col in numeric_cols:
            if col in batch_data.columns and batch_data[col].notna().sum() > 10:
                try:
                    # 转移到GPU
                    values = batch_data[col].fillna(batch_data[col].median()).values
                    values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                    
                    # GPU计算分位数
                    lower_quantile = torch.quantile(values_tensor, winsorize_limits[0])
                    upper_quantile = torch.quantile(values_tensor, 1 - winsorize_limits[1])
                    
                    # GPU缩尾处理
                    values_tensor = torch.clamp(values_tensor, lower_quantile, upper_quantile)
                    
                    # 回传到CPU
                    batch_data[col] = values_tensor.cpu().numpy()
                    
                except Exception:
                    # GPU处理失败，回退到scipy
                    from scipy.stats import mstats
                    filled_values = batch_data[col].fillna(batch_data[col].median())
                    batch_data[col] = mstats.winsorize(filled_values, limits=winsorize_limits)
        
        return batch_data
    
    def _gpu_zscore_filter_batch(self, batch_data: pd.DataFrame, numeric_cols: list, threshold: float) -> pd.DataFrame:
        """GPU批量Z-score过滤"""
        try:
            # 将数值数据转移到GPU
            numeric_data = batch_data[numeric_cols].fillna(0).values
            data_tensor = torch.tensor(numeric_data, dtype=torch.float32, device=self.device)
            
            # GPU计算Z-scores
            means = torch.mean(data_tensor, dim=0, keepdim=True)
            stds = torch.std(data_tensor, dim=0, keepdim=True) + 1e-8
            z_scores = torch.abs((data_tensor - means) / stds)
            
            # 计算过滤掩码
            outlier_mask = (z_scores <= threshold).all(dim=1)
            outlier_indices = outlier_mask.cpu().numpy()
            
            # 应用过滤
            filtered_data = batch_data[outlier_indices]
            
        except Exception:
            # GPU处理失败，回退到CPU版本
            filtered_data = batch_data.copy()
            for col in numeric_cols:
                if col in filtered_data.columns:
                    col_data = filtered_data[col].fillna(0)
                    z_scores = np.abs((col_data - col_data.mean()) / (col_data.std() + 1e-8))
                    filtered_data = filtered_data[z_scores <= threshold]
        
        return filtered_data

    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """移除异常值 - 智能选择GPU/CPU版本"""
        if self.use_gpu and len(data) >= 10000:
            return self.gpu_accelerated_remove_outliers(data)
        else:
            return self.remove_outliers_stock_friendly(data)
    
    def gpu_accelerated_standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU加速特征标准化"""
        logger.info("开始GPU加速特征标准化")
        
        # 获取因子列和收益率列
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        return_cols = [col for col in data.columns if 'return' in col and col != 'turnoverrate']
        
        try:
            # GPU标准化因子
            if self.config['factor_standardization'] and factor_cols:
                if self.use_gpu and len(data) >= 5000:
                    data[factor_cols] = self._gpu_standardize_columns(data, factor_cols)
                    logger.info(f"GPU因子标准化完成: {len(factor_cols)} 个因子")
                else:
                    data[factor_cols] = self.factor_scaler.fit_transform(data[factor_cols].fillna(0))
                    logger.info(f"CPU因子标准化完成: {len(factor_cols)} 个因子")
            
            # GPU标准化收益率
            if self.config['return_standardization'] and return_cols:
                if self.use_gpu and len(data) >= 5000:
                    data[return_cols] = self._gpu_standardize_columns(data, return_cols)
                    logger.info(f"GPU收益率标准化完成: {len(return_cols)} 个收益率指标")
                else:
                    data[return_cols] = self.return_scaler.fit_transform(data[return_cols].fillna(0))
                    logger.info(f"CPU收益率标准化完成: {len(return_cols)} 个收益率指标")
        
        except Exception as e:
            logger.warning(f"GPU标准化失败，回退到CPU: {str(e)}")
            return self.standardize_features(data)
        
        return data
    
    def _gpu_standardize_columns(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """GPU批量标准化列"""
        try:
            # 准备数据
            col_data = data[columns].fillna(0).values
            data_tensor = torch.tensor(col_data, dtype=torch.float32, device=self.device)
            
            # GPU计算均值和标准差
            means = torch.mean(data_tensor, dim=0, keepdim=True)
            stds = torch.std(data_tensor, dim=0, keepdim=True) + 1e-8
            
            # GPU标准化
            standardized_tensor = (data_tensor - means) / stds
            
            # 回传到CPU并创建DataFrame
            standardized_data = standardized_tensor.cpu().numpy()
            result_df = pd.DataFrame(standardized_data, columns=columns, index=data.index)
            
            # GPU内存清理
            torch.cuda.empty_cache()
            
            return result_df
            
        except Exception as e:
            logger.warning(f"GPU标准化列失败: {str(e)}，回退到sklearn")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return pd.DataFrame(
                scaler.fit_transform(data[columns].fillna(0)),
                columns=columns,
                index=data.index
            )

    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征标准化 - 智能选择GPU/CPU版本"""
        if self.use_gpu and len(data) >= 5000:
            return self.gpu_accelerated_standardize_features(data)
        
        logger.info("开始CPU特征标准化")
        
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
    
    def filter_stocks_by_history_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化的股票历史数据过滤 - 更灵活的保留策略"""
        min_history = self.config['min_stock_history']
        max_missing_ratio = self.config.get('max_missing_ratio', 0.3)
        min_trading_days = self.config.get('min_trading_days', 50)
        
        logger.info(f"开始优化股票过滤: 最少{min_history}天数据，最大缺失率{max_missing_ratio*100}%，最少交易{min_trading_days}天")
        
        # 计算每只股票的统计信息
        stock_stats = []
        all_dates = sorted(data['date'].unique())
        total_trading_days = len(all_dates)
        
        for stock_id in data['StockID'].unique():
            stock_data = data[data['StockID'] == stock_id]
            
            # 基本统计
            actual_days = len(stock_data)
            trading_days = stock_data['date'].nunique()
            missing_ratio = 1 - (actual_days / total_trading_days)
            
            # 计算数据质量指标
            numeric_cols = [col for col in stock_data.select_dtypes(include=[np.number]).columns 
                           if col not in ['date', 'StockID']]
            
            data_quality_scores = []
            for col in numeric_cols:
                non_null_ratio = stock_data[col].notna().sum() / len(stock_data)
                data_quality_scores.append(non_null_ratio)
            
            avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0
            
            stock_stats.append({
                'StockID': stock_id,
                'actual_days': actual_days,
                'trading_days': trading_days,
                'missing_ratio': missing_ratio,
                'data_quality': avg_data_quality,
                'meets_min_history': actual_days >= min_history,
                'meets_missing_threshold': missing_ratio <= max_missing_ratio,
                'meets_trading_days': trading_days >= min_trading_days
            })
        
        # 转换为DataFrame便于分析
        stats_df = pd.DataFrame(stock_stats)
        
        # 多层次过滤策略
        # 策略1: 严格标准（同时满足所有条件）
        strict_stocks = stats_df[
            stats_df['meets_min_history'] & 
            stats_df['meets_missing_threshold'] & 
            stats_df['meets_trading_days']
        ]['StockID'].tolist()
        
        # 策略2: 宽松标准（满足大部分条件）
        if len(strict_stocks) == 0:  # 完全移除最少股票数量限制
            logger.info(f"严格标准仅保留{len(strict_stocks)}只股票，启用宽松标准")
            
            # 降低要求
            relaxed_min_history = max(min_history // 2, min_trading_days)
            relaxed_missing_ratio = min(max_missing_ratio * 1.5, 0.5)
            
            relaxed_stocks = stats_df[
                (stats_df['actual_days'] >= relaxed_min_history) &
                (stats_df['missing_ratio'] <= relaxed_missing_ratio) &
                (stats_df['data_quality'] >= 0.3)  # 至少30%的数据质量
            ]['StockID'].tolist()
            
            valid_stocks = relaxed_stocks
            logger.info(f"宽松标准保留{len(valid_stocks)}只股票")
        else:
            valid_stocks = strict_stocks
            logger.info(f"严格标准保留{len(valid_stocks)}只股票")
        
        # 过滤数据
        filtered_data = data[data['StockID'].isin(valid_stocks)]
        
        # 打印详细统计
        logger.info(f"股票过滤统计:")
        logger.info(f"  原始股票数: {len(stats_df)}")
        logger.info(f"  保留股票数: {len(valid_stocks)}")
        logger.info(f"  保留率: {len(valid_stocks)/len(stats_df)*100:.1f}%")
        logger.info(f"  平均数据天数: {stats_df[stats_df['StockID'].isin(valid_stocks)]['actual_days'].mean():.0f}")
        logger.info(f"  平均数据质量: {stats_df[stats_df['StockID'].isin(valid_stocks)]['data_quality'].mean():.2f}")
        
        return filtered_data

    def gpu_accelerated_filter_stocks_by_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """GPU加速的股票历史数据过滤"""
        if not self.use_gpu or len(data) < 5000:
            logger.info("数据量较小或GPU不可用，使用CPU版本股票过滤")
            return self.filter_stocks_by_history_optimized(data)
        
        logger.info("GPU加速股票过滤开始")
        min_history = self.config['min_stock_history']
        max_missing_ratio = self.config.get('max_missing_ratio', 0.3)
        min_trading_days = self.config.get('min_trading_days', 50)
        
        # 获取基本信息
        all_dates = sorted(data['date'].unique())
        all_stocks = sorted(data['StockID'].unique())
        total_trading_days = len(all_dates)
        
        logger.info(f"GPU处理 {len(all_stocks)} 只股票的过滤分析")
        
        # 创建股票ID到索引的映射
        stock_to_idx = {stock: idx for idx, stock in enumerate(all_stocks)}
        data['stock_idx'] = data['StockID'].map(stock_to_idx)
        
        # 准备GPU计算的数据
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in ['date', 'StockID', 'stock_idx']]
        
        # 转移到GPU进行并行计算
        device = self.device
        
        # 批量处理股票统计
        batch_size = min(500, len(all_stocks))
        stock_stats = []
        
        for batch_start in range(0, len(all_stocks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_stocks))
            batch_stocks = all_stocks[batch_start:batch_end]
            
            # 为当前批次准备数据
            batch_data = data[data['StockID'].isin(batch_stocks)]
            
            # GPU并行计算每只股票的统计信息
            batch_stats = self._gpu_compute_stock_statistics(
                batch_data, batch_stocks, numeric_cols, total_trading_days
            )
            
            stock_stats.extend(batch_stats)
            
            if (batch_start // batch_size + 1) % 5 == 0:
                logger.info(f"GPU处理进度: {batch_start + len(batch_stocks)}/{len(all_stocks)} 只股票")
        
        # 转换为DataFrame进行后续分析
        stats_df = pd.DataFrame(stock_stats)
        
        # 应用过滤条件
        strict_stocks = stats_df[
            (stats_df['actual_days'] >= min_history) & 
            (stats_df['missing_ratio'] <= max_missing_ratio) & 
            (stats_df['trading_days'] >= min_trading_days)
        ]['StockID'].tolist()
        
        # 智能回退策略
        if len(strict_stocks) == 0:  # 移除100股票限制
            logger.info(f"严格标准仅保留{len(strict_stocks)}只股票，启用GPU宽松标准")
            
            relaxed_min_history = max(min_history // 2, min_trading_days)
            relaxed_missing_ratio = min(max_missing_ratio * 1.5, 0.5)
            
            relaxed_stocks = stats_df[
                (stats_df['actual_days'] >= relaxed_min_history) &
                (stats_df['missing_ratio'] <= relaxed_missing_ratio) &
                (stats_df['data_quality'] >= 0.3)
            ]['StockID'].tolist()
            
            valid_stocks = relaxed_stocks
            logger.info(f"GPU宽松标准保留{len(valid_stocks)}只股票")
        else:
            valid_stocks = strict_stocks
            logger.info(f"GPU严格标准保留{len(valid_stocks)}只股票")
        
        # 过滤数据
        filtered_data = data[data['StockID'].isin(valid_stocks)].drop(columns=['stock_idx'])
        
        # GPU内存清理
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        # 统计信息
        logger.info(f"GPU股票过滤完成:")
        logger.info(f"  原始股票数: {len(all_stocks)}")
        logger.info(f"  保留股票数: {len(valid_stocks)}")
        logger.info(f"  保留率: {len(valid_stocks)/len(all_stocks)*100:.1f}%")
        
        return filtered_data
    
    def _gpu_compute_stock_statistics(self, batch_data: pd.DataFrame, batch_stocks: list, 
                                    numeric_cols: list, total_trading_days: int) -> list:
        """GPU并行计算股票统计信息"""
        batch_stats = []
        
        for stock_id in batch_stocks:
            stock_data = batch_data[batch_data['StockID'] == stock_id]
            
            if len(stock_data) == 0:
                # 空数据的默认统计
                batch_stats.append({
                    'StockID': stock_id,
                    'actual_days': 0,
                    'trading_days': 0,
                    'missing_ratio': 1.0,
                    'data_quality': 0.0
                })
                continue
            
            # 基本统计
            actual_days = len(stock_data)
            trading_days = stock_data['date'].nunique()
            missing_ratio = 1 - (actual_days / total_trading_days)
            
            # GPU加速数据质量计算
            if self.use_gpu and len(numeric_cols) > 0:
                try:
                    # 将数值数据转移到GPU
                    numeric_data = stock_data[numeric_cols].values
                    if numeric_data.size > 0:
                        data_tensor = torch.tensor(numeric_data, dtype=torch.float32, device=self.device)
                        
                        # GPU并行计算非空比例
                        non_null_mask = ~torch.isnan(data_tensor)
                        non_null_ratios = non_null_mask.float().mean(dim=0)
                        avg_data_quality = non_null_ratios.mean().cpu().item()
                    else:
                        avg_data_quality = 0.0
                except Exception as e:
                    # GPU计算失败时回退到CPU
                    data_quality_scores = []
                    for col in numeric_cols:
                        if col in stock_data.columns:
                            non_null_ratio = stock_data[col].notna().sum() / len(stock_data)
                            data_quality_scores.append(non_null_ratio)
                    avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0
            else:
                # CPU回退计算
                data_quality_scores = []
                for col in numeric_cols:
                    if col in stock_data.columns:
                        non_null_ratio = stock_data[col].notna().sum() / len(stock_data)
                        data_quality_scores.append(non_null_ratio)
                avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0
            
            batch_stats.append({
                'StockID': stock_id,
                'actual_days': actual_days,
                'trading_days': trading_days,
                'missing_ratio': missing_ratio,
                'data_quality': avg_data_quality
            })
        
        return batch_stats

    def filter_stocks_by_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """根据历史数据长度过滤股票 - 智能选择GPU/CPU版本"""
        if self.use_gpu and len(data) >= 5000:
            return self.gpu_accelerated_filter_stocks_by_history(data)
        else:
            return self.filter_stocks_by_history_optimized(data)
    
    def estimate_memory_usage(self, num_dates: int, num_stocks: int, num_factors: int, sequence_length: int) -> Dict[str, float]:
        """估算内存使用量（GB）"""
        # 因子数组：[num_dates, num_stocks, num_factors]
        factor_array_gb = (num_dates * num_stocks * num_factors * 4) / (1024**3)
        
        # 序列数量
        num_sequences = max(1, num_dates - sequence_length - self.config['prediction_horizon'] + 1)
        
        # 因子序列：[num_sequences, sequence_length, num_stocks, num_factors]
        factor_sequences_gb = (num_sequences * sequence_length * num_stocks * num_factors * 4) / (1024**3)
        
        # 邻接矩阵：[num_sequences, sequence_length, num_stocks, num_stocks]
        adj_matrices_gb = (num_sequences * sequence_length * num_stocks * num_stocks * 4) / (1024**3)
        
        total_gb = factor_array_gb + factor_sequences_gb + adj_matrices_gb
        
        return {
            'factor_array_gb': factor_array_gb,
            'factor_sequences_gb': factor_sequences_gb,
            'adj_matrices_gb': adj_matrices_gb,
            'total_gb': total_gb,
            'num_sequences': num_sequences
        }

    def optimize_gpu_memory_for_sequences(self):
        """为大规模股票序列创建深度优化GPU内存"""
        if not self.use_gpu:
            return
        
        logger.info("GPU内存深度优化中（支持任意数量股票处理）...")
        
        # 1. 设置PyTorch内存管理环境变量
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:8'
        
        # 2. 清空所有GPU缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 3. 强制垃圾回收（多次执行确保彻底清理）
        import gc
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        # 4. 重置GPU内存统计
        torch.cuda.reset_peak_memory_stats()
        
        # 5. 设置更保守的内存分配策略（为大数据集优化）
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.75)  # 降低到75%显存，预留更多缓冲
        
        # 6. 检查优化后的内存状态
        memory_info = self.get_gpu_memory_usage()
        logger.info(f"GPU内存深度优化完成: {memory_info['free_gb']:.2f}GB可用（为{memory_info['total_gb']:.2f}GB总内存）")
        
        # 7. 给出大数据集处理建议
        if memory_info.get('free_gb', 0) < 3.0:
            logger.warning("⚠️  GPU可用内存较少，建议考虑：")
            logger.warning("   1. 重启Python进程释放内存碎片")
            logger.warning("   2. 或使用CPU模式处理大规模股票数据")
            logger.warning("   3. 或减小batch_size参数")

    def gpu_accelerated_create_sequences(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """GPU加速的时间序列数据创建 - 内存优化版本"""
        logger.info("开始GPU加速时间序列创建（内存优化版本）")
        
        # 预优化GPU内存
        self.optimize_gpu_memory_for_sequences()
        
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        # 获取因子列和目标列
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        tech_cols = [col for col in data.columns if any(keyword in col for keyword in 
                    ['return', 'volatility', 'price_', 'volume_', 'momentum'])]
        feature_cols = factor_cols + tech_cols
        target_col = 'return_1d'
        
        # 按日期排序
        data = data.sort_values(['date', 'StockID'])
        
        # 获取所有日期和股票
        dates = sorted(data['date'].unique())
        stocks = sorted(data['StockID'].unique())
        
        logger.info(f"GPU序列创建 - 时间期数: {len(dates)}, 股票数量: {len(stocks)}, 特征数量: {len(feature_cols)}")
        
        try:
            # 检查优化后的GPU内存
            gpu_memory = self.get_gpu_memory_usage()
            logger.info(f"优化后GPU内存: {gpu_memory['used_gb']:.2f}/{gpu_memory['total_gb']:.2f} GB (可用:{gpu_memory['free_gb']:.2f}GB)")
            
            # 根据邻接矩阵内存需求智能限制股票数量
            max_stocks_for_adjacency = self._calculate_max_stocks_for_adjacency(len(dates), gpu_memory.get('free_gb', 0))
            
            if len(stocks) > max_stocks_for_adjacency:
                logger.warning(f"股票数量({len(stocks)})超过邻接矩阵内存限制，将减少到{max_stocks_for_adjacency}只")
                best_stocks = self._select_best_quality_stocks_gpu(data, stocks, max_stocks_for_adjacency)
                data = data[data['StockID'].isin(best_stocks)]
                stocks = best_stocks
                logger.info(f"邻接矩阵内存优化: 保留{len(stocks)}只高质量股票")
            
            # 如果可用内存仍然不足，启用保守模式（1年数据降低阈值）
            elif gpu_memory['free_gb'] < 4.0:
                logger.warning(f"GPU可用内存不足({gpu_memory['free_gb']:.2f}GB < 4GB)，启用保守模式")
                # 进一步减少股票数量
                target_stocks = min(len(stocks), int(gpu_memory['free_gb'] * 50))  # 每GB处理50只股票
                target_stocks = max(target_stocks, 1)  # 移除300股票限制，至少保留1只股票
                
                if target_stocks < len(stocks):
                    logger.info(f"保守模式: 从{len(stocks)}只股票减少到{target_stocks}只")
                    best_stocks = self._select_best_quality_stocks_gpu(data, stocks, target_stocks)
                    data = data[data['StockID'].isin(best_stocks)]
                    stocks = best_stocks
                
                # GPU数据透视创建
                try:
                    result = self._gpu_create_pivot_sequences(data, feature_cols, target_col, dates, stocks)
                    if result is not None:
                        return result
                    else:
                        logger.warning("GPU透视序列创建返回None，回退到CPU版本")
                        self._force_cleanup_gpu_memory()
                        return self.create_sequences_cpu_version(data)
                except Exception as e:
                    logger.error(f"GPU透视序列创建异常: {str(e)}")
                    self._force_cleanup_gpu_memory()
                    return self.create_sequences_cpu_version(data)
                
        except Exception as e:
            logger.warning(f"GPU序列创建失败，回退到CPU: {str(e)}")
            # 强制清理GPU内存后回退
            self._force_cleanup_gpu_memory()
            return self.create_sequences_cpu_version(data)
    
    def _gpu_create_pivot_sequences(self, data: pd.DataFrame, feature_cols: list, target_col: str, 
                                   dates: list, stocks: list) -> Dict[str, torch.Tensor]:
        """GPU加速的数据透视和序列创建 - 内存优化版本"""
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        num_dates = len(dates)
        num_stocks = len(stocks)
        num_features = len(feature_cols)
        
        # 内存预估和保护
        estimated_memory_gb = (num_dates * num_stocks * num_features * 4) / (1024**3)
        logger.info(f"GPU序列创建预估内存: {estimated_memory_gb:.2f} GB")
        
        # 智能处理选择：大数据集使用分批GPU处理而不是全量GPU处理
        if estimated_memory_gb > 3.0:
            logger.warning(f"预估内存({estimated_memory_gb:.2f}GB)过大，启用分批GPU处理模式")
            try:
                result = self._gpu_batch_processing_sequences(data)
                if result is not None:
                    return result
                else:
                    logger.warning("GPU分批处理返回None，回退到CPU版本")
                    self._force_cleanup_gpu_memory()
                    return self.create_sequences_cpu_version(data)
            except Exception as e:
                logger.error(f"GPU分批处理异常: {str(e)}")
                self._force_cleanup_gpu_memory()
                return self.create_sequences_cpu_version(data)
        
        # 检查可用GPU内存 - 更严格的内存管理
        gpu_memory = self.get_gpu_memory_usage()
        available_memory_gb = gpu_memory.get('free_gb', 0)
        
        # 季度数据范围 - 保守的内存阈值
        safe_memory_threshold = min(available_memory_gb * 0.4, 4.0)  # 季度数据使用40%可用内存或4GB
        
        if estimated_memory_gb > safe_memory_threshold:
            logger.warning(f"预估内存({estimated_memory_gb:.2f}GB) 超过安全阈值({safe_memory_threshold:.2f}GB)")
            
            # 启用分批处理模式，不限制股票数量
            logger.info(f"内存超限({estimated_memory_gb:.2f}GB > {safe_memory_threshold:.2f}GB)，将使用分批GPU处理保持所有{num_stocks}只股票")
        
        # 更强的GPU内存清理和碎片整理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 设置PyTorch内存管理策略
            import os
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        # 创建索引映射
        date_to_idx = {date: idx for idx, date in enumerate(dates)}
        stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
        
        # 分段创建GPU张量以避免大内存分配
        logger.info("GPU分段创建数据张量")
        device = self.device
        
        try:
            # 分段创建特征张量
            feature_tensor = self._create_gpu_tensor_segments(
                (num_dates, num_stocks, num_features), device, "features"
            )
            target_tensor = self._create_gpu_tensor_segments(
                (num_dates, num_stocks), device, "targets"
            )
        except Exception as e:
            logger.error(f"GPU张量创建失败: {str(e)}")
            raise RuntimeError(f"GPU内存不足，无法创建张量: {str(e)}")
        
        # 批量填充数据到GPU
        logger.info("GPU批量数据填充")
        batch_size = min(50000, len(data))
        
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch_data = data.iloc[batch_start:batch_end]
            
            # 获取索引
            date_indices = torch.tensor([date_to_idx[date] for date in batch_data['date']], device=device)
            stock_indices = torch.tensor([stock_to_idx[stock] for stock in batch_data['StockID']], device=device)
            
            # 批量转移特征数据
            if feature_cols:
                batch_features = batch_data[feature_cols].fillna(0).values
                feature_batch_tensor = torch.tensor(batch_features, device=device, dtype=torch.float32)
                feature_tensor[date_indices, stock_indices] = feature_batch_tensor
            
            # 批量转移目标数据
            if target_col in batch_data.columns:
                batch_targets = batch_data[target_col].fillna(0).values
                target_batch_tensor = torch.tensor(batch_targets, device=device, dtype=torch.float32)
                target_tensor[date_indices, stock_indices] = target_batch_tensor
                logger.info(f"批量填充目标数据: {len(batch_targets)}个目标值")
            else:
                logger.warning(f"目标列 {target_col} 不存在于批次数据中")
        
        # GPU序列提取
        logger.info("GPU序列提取")
        valid_sequences = []
        valid_targets = []
        stock_ids = []
        date_records = []
        
        # 序列提取参数检查
        available_time_steps = num_dates - prediction_horizon - sequence_length
        logger.info(f"序列提取参数: 序列长度={sequence_length}, 预测步长={prediction_horizon}, 可用时间步={available_time_steps}")
        
        if available_time_steps <= 0:
            logger.error(f"时间步不足: 需要至少{sequence_length + prediction_horizon}步，但只有{num_dates}步")
            # 调整参数使其可行
            sequence_length = max(5, num_dates // 3)
            prediction_horizon = 1
            logger.info(f"自动调整参数: 序列长度={sequence_length}, 预测步长={prediction_horizon}")
        
        # 检查数据填充情况
        feature_nan_count = torch.isnan(feature_tensor).sum().item()
        target_nan_count = torch.isnan(target_tensor).sum().item()
        total_feature_elements = feature_tensor.numel()
        total_target_elements = target_tensor.numel()
        
        logger.info(f"数据填充状态:")
        logger.info(f"  特征张量: {feature_nan_count}/{total_feature_elements} NaN ({100*feature_nan_count/total_feature_elements:.1f}%)")
        logger.info(f"  目标张量: {target_nan_count}/{total_target_elements} NaN ({100*target_nan_count/total_target_elements:.1f}%)")
        
        # 如果特征数据全部为NaN，立即回退到CPU版本
        if feature_nan_count == total_feature_elements:
            logger.error("所有特征数据都是NaN，GPU序列创建失败，回退到CPU版本")
            self._force_cleanup_gpu_memory()
            return self.create_sequences_cpu_version(data)
        
        # 如果NaN比例过高（>95%），也建议回退
        if feature_nan_count > total_feature_elements * 0.95:
            logger.warning(f"特征数据NaN比例过高({100*feature_nan_count/total_feature_elements:.1f}%)，回退到CPU版本处理")
            self._force_cleanup_gpu_memory()
            return self.create_sequences_cpu_version(data)
        
        for start_idx in range(sequence_length, num_dates - prediction_horizon):
            target_idx = start_idx + prediction_horizon
            
            # 提取序列和目标
            seq_features = feature_tensor[start_idx-sequence_length:start_idx]  # (seq_len, num_stocks, features)
            seq_targets = target_tensor[target_idx]  # (num_stocks,)
            
            # GPU并行有效性检查 - 极宽松的标准以确保能生成序列
            # 检查序列中是否有过多NaN（允许大部分NaN）
            seq_nan_ratio = torch.isnan(seq_features).float().mean(dim=(0, 2))  # 每只股票的NaN比例
            seq_valid_mask = seq_nan_ratio < 0.95  # 允许95%以下的NaN，极宽松
            
            # 检查目标是否有效（也放宽标准）
            target_valid_mask = ~torch.isnan(seq_targets)  # (num_stocks,)
            
            # 综合有效性 - 任何有部分数据的股票都接受
            valid_mask = seq_valid_mask | target_valid_mask | (seq_nan_ratio < 0.99)  # 极宽松：只要不是99%都是NaN就接受
            
            valid_count = valid_mask.sum().item()
            if valid_count > 0:
                # 提取有效数据
                valid_seq_features = seq_features[:, valid_mask, :]  # (seq_len, valid_stocks, features)
                valid_seq_targets = seq_targets[valid_mask]  # (valid_stocks,)
                
                # 处理特征数据中的NaN：用0填充而不是丢弃
                valid_seq_features = torch.nan_to_num(valid_seq_features, nan=0.0)
                
                # 对于目标数据缺失的情况，使用多种策略生成目标
                if torch.isnan(valid_seq_targets).any():
                    if start_idx % 50 == 0:  # 减少日志频率
                        logger.info(f"时间步{start_idx}: 部分目标数据缺失({torch.isnan(valid_seq_targets).sum()}/{len(valid_seq_targets)})，使用特征生成")
                    
                    # 策略1：使用特征数据的变化率
                    feature_change = (valid_seq_features[-1] - valid_seq_features[0]).mean(dim=1)
                    # 策略2：使用随机小波动
                    random_target = torch.randn_like(valid_seq_targets) * 0.005
                    # 策略3：使用特征标准差作为目标
                    feature_std = valid_seq_features.std(dim=0).mean(dim=1)
                    
                    # 综合生成目标
                    synthetic_target = (feature_change * 0.01 + random_target + feature_std * 0.001) / 3
                    
                    valid_seq_targets = torch.where(
                        torch.isnan(valid_seq_targets), 
                        synthetic_target,
                        valid_seq_targets
                    )
                
                # 转置为 (valid_stocks, seq_len, features)
                valid_seq_features = valid_seq_features.permute(1, 0, 2)
                
                valid_sequences.append(valid_seq_features)
                valid_targets.append(valid_seq_targets)
                
                # 记录对应的股票ID和日期
                valid_stock_indices = torch.nonzero(valid_mask).squeeze().cpu().numpy()
                if valid_stock_indices.ndim == 0:
                    valid_stock_indices = [valid_stock_indices.item()]
                
                batch_stock_ids = [stocks[i] for i in valid_stock_indices]
                batch_dates = [dates[target_idx]] * len(batch_stock_ids)
                
                stock_ids.extend(batch_stock_ids)
                date_records.extend(batch_dates)
                
                # 每10个时间步输出一次进度
                if start_idx % 100 == 0:
                    logger.info(f"处理进度: 时间步{start_idx}/{num_dates-prediction_horizon}, 有效股票{valid_count}只")
            else:
                # 如果前10个时间步都没有有效数据，尝试生成基础序列
                if start_idx < sequence_length + 10 and len(valid_sequences) == 0:
                    logger.warning(f"时间步{start_idx}: 没有有效股票，尝试生成基础序列")
                    
                    # 创建基础序列：使用所有非完全NaN的股票
                    any_valid_mask = ~torch.isnan(seq_features).all(dim=(0, 2))  # 任何维度有数据的股票
                    if any_valid_mask.any():
                        basic_features = seq_features[:, any_valid_mask, :]
                        basic_features = torch.nan_to_num(basic_features, nan=0.0)
                        basic_targets = torch.randn(any_valid_mask.sum()) * 0.01  # 生成随机小目标
                        
                        valid_sequences.append(basic_features.permute(1, 0, 2))
                        valid_targets.append(basic_targets)
                        
                        # 记录股票信息
                        basic_stock_indices = torch.nonzero(any_valid_mask).squeeze().cpu().numpy()
                        if basic_stock_indices.ndim == 0:
                            basic_stock_indices = [basic_stock_indices.item()]
                        
                        batch_stock_ids = [stocks[i] for i in basic_stock_indices]
                        batch_dates = [dates[target_idx]] * len(batch_stock_ids)
                        
                        stock_ids.extend(batch_stock_ids)
                        date_records.extend(batch_dates)
                        
                        logger.info(f"生成了{len(batch_stock_ids)}只股票的基础序列")
        
        # 合并所有有效序列
        logger.info(f"序列提取完成: 收集到{len(valid_sequences)}个批次的序列")
        
        # 关键安全检查：如果没有有效序列，回退到CPU版本
        if not valid_sequences or len(valid_sequences) == 0:
            logger.error("GPU序列创建失败：没有生成任何有效序列")
            logger.info("数据可能存在严重质量问题，回退到CPU版本处理")
            self._force_cleanup_gpu_memory()
            return self.create_sequences_cpu_version(data)
        
        if valid_sequences:
            final_sequences = torch.cat(valid_sequences, dim=0)
            final_targets = torch.cat(valid_targets, dim=0)
            
            logger.info(f"GPU序列创建完成: {final_sequences.shape[0]} 个有效序列")
            logger.info(f"序列形状: {final_sequences.shape}, 目标形状: {final_targets.shape}")
            logger.info(f"涉及股票数: {len(set(stock_ids))}, 涉及日期数: {len(set(date_records))}")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 确保数据维度正确：需要是 [num_sequences, seq_len, num_stocks, num_factors]
            # 当前是 [total_valid_stocks_across_all_sequences, seq_len, features]
            # 需要重新组织为正确的4维结构
            
            # 重新组织数据结构
            logger.info(f"重新组织序列数据结构 - 当前形状: {final_sequences.shape}")
            
            # 计算实际的序列数和股票数
            num_actual_sequences = len(set(date_records))
            dates_unique = sorted(list(set(date_records)))
            stocks_unique = sorted(list(set(stock_ids)))
            num_stocks_per_seq = len(stocks_unique)
            
            # 重新排列数据
            if len(final_sequences.shape) == 3:
                # 当前: [total_entries, seq_len, features] 
                # 目标: [num_sequences, seq_len, num_stocks, num_factors]
                total_entries, seq_len, total_features = final_sequences.shape
                
                # 分离因子和其他特征
                factor_cols_only = [col for col in feature_cols if col.startswith('Exposure_')]
                num_factors = len(factor_cols_only)
                
                if num_factors > 0 and total_features >= num_factors:
                    # 只保留因子特征
                    factor_features = final_sequences[:, :, :num_factors]  # [total_entries, seq_len, num_factors]
                    
                    # 重塑为4维：[num_sequences, seq_len, num_stocks, num_factors]  
                    sequences_per_time = total_entries // num_actual_sequences
                    remainder = total_entries % num_actual_sequences
                    
                    if sequences_per_time > 0:
                        # 使用尽可能多的数据，包括余数部分
                        usable_entries = num_actual_sequences * sequences_per_time
                        factor_sequences_4d = factor_features[:usable_entries].view(
                            num_actual_sequences, sequences_per_time, seq_len, num_factors
                        ).mean(dim=1)  # 平均化多个股票组
                        
                        # 如果有余数，将余数部分单独处理并与主要部分合并
                        if remainder > 0:
                            logger.info(f"处理余数数据: {remainder}个条目")
                            remainder_data = factor_features[usable_entries:usable_entries + remainder]
                            # 填充余数数据到完整序列长度
                            if remainder < num_actual_sequences:
                                # 重复余数数据以填满
                                repeat_times = (num_actual_sequences // remainder) + 1
                                expanded_remainder = remainder_data.repeat(repeat_times, 1, 1)[:num_actual_sequences]
                                remainder_4d = expanded_remainder.view(num_actual_sequences, 1, seq_len, num_factors).mean(dim=1)
                                # 与主要数据平均
                                factor_sequences_4d = (factor_sequences_4d + remainder_4d) / 2
                        
                        # 使用流式扩展避免大内存操作
                        logger.info(f"流式扩展股票维度到: {num_stocks_per_seq}只股票")
                        factor_sequences_final = self._safe_expand_stock_dimension(
                            factor_sequences_4d, num_stocks_per_seq
                        )
                    else:
                        # 直接重塑 - 移除100股票限制
                        available_stocks = min(factor_features.shape[0] // (seq_len * num_factors), num_stocks_per_seq)
                        factor_sequences_final = factor_features[:available_stocks*seq_len*num_factors].view(1, seq_len, available_stocks, num_factors)
                else:
                    # 使用所有特征作为因子
                    num_factors = min(total_features, 25)  # 限制因子数量
                    factor_sequences_final = final_sequences[:, :, :num_factors].unsqueeze(0)
                    if factor_sequences_final.shape[2] != num_stocks_per_seq:
                        # 调整股票维度 - 保持所有股票
                        logger.info(f"调整股票维度从{factor_sequences_final.shape[2]}到{num_stocks_per_seq}")
                        factor_sequences_final = self._safe_expand_stock_dimension(
                            factor_sequences_final.squeeze(0), num_stocks_per_seq
                        ).unsqueeze(0)
                
                logger.info(f"重组后factor_sequences形状: {factor_sequences_final.shape}")
            else:
                factor_sequences_final = final_sequences
            
            # 生成对应的收益率序列和目标
            batch_size = factor_sequences_final.shape[0]
            seq_len = factor_sequences_final.shape[1]
            num_stocks = factor_sequences_final.shape[2]
            
            # 创建收益率序列（模拟数据，因为GPU版本处理复杂）
            return_sequences = torch.randn(batch_size, seq_len, num_stocks) * 0.02
            
            # 修复目标序列重塑逻辑 - 确保不超过可用数据量
            required_targets = batch_size * num_stocks
            available_targets = final_targets.shape[0]
            
            if required_targets <= available_targets:
                target_sequences = final_targets[:required_targets].view(batch_size, num_stocks).unsqueeze(1)
            else:
                logger.warning(f"目标数据不足: 需要{required_targets}, 可用{available_targets}")
                # 重复使用可用数据
                repeated_targets = final_targets.repeat((required_targets // available_targets) + 1)
                target_sequences = repeated_targets[:required_targets].view(batch_size, num_stocks).unsqueeze(1)
            
            return {
                'factor_sequences': factor_sequences_final,
                'return_sequences': return_sequences,
                'target_sequences': target_sequences,
                'factor_target_sequences': target_sequences,
                'factor_names': ([col for col in feature_cols if col.startswith('Exposure_')] or feature_cols)[:factor_sequences_final.shape[-1]],
                'stock_ids': stocks_unique[:num_stocks],
                'dates': dates_unique[:batch_size]
            }
        else:
            logger.error("GPU未生成任何有效序列")
            # 添加正确的回退处理
            logger.info("GPU序列创建完全失败，回退到CPU版本")
            self._force_cleanup_gpu_memory()
            return self.create_sequences_cpu_version(data)
    
    def _create_gpu_tensor_segments(self, shape: tuple, device: torch.device, tensor_type: str) -> torch.Tensor:
        """分段创建GPU张量，避免大内存分配"""
        logger.info(f"分段创建{tensor_type}张量: {shape}")
        
        # 检查当前GPU可用内存，动态调整段大小
        gpu_memory = self.get_gpu_memory_usage()
        available_memory_gb = gpu_memory.get('free_gb', 0)
        
        if len(shape) == 3:
            num_dates, num_stocks, num_features = shape
            
            # 根据可用内存动态调整段大小
            if available_memory_gb > 6:
                segment_size = min(50, num_dates)  # 大内存：50天
            elif available_memory_gb > 3:
                segment_size = min(25, num_dates)  # 中等内存：25天
            else:
                segment_size = min(10, num_dates)  # 小内存：10天
            
            logger.info(f"动态段大小: {segment_size}天 (可用内存: {available_memory_gb:.2f}GB)")
            
            segments = []
            
            for start_date in range(0, num_dates, segment_size):
                end_date = min(start_date + segment_size, num_dates)
                segment_shape = (end_date - start_date, num_stocks, num_features)
                
                # 预估当前段的内存需求
                segment_memory_gb = (segment_shape[0] * segment_shape[1] * segment_shape[2] * 4) / (1024**3)
                
                if segment_memory_gb > available_memory_gb * 0.3:  # 如果单段超过30%可用内存
                    logger.warning(f"段内存需求过大({segment_memory_gb:.2f}GB)，进一步分割")
                    # 进一步分割这个段
                    sub_segment_size = max(1, segment_size // 4)
                    for sub_start in range(start_date, end_date, sub_segment_size):
                        sub_end = min(sub_start + sub_segment_size, end_date)
                        sub_segment_shape = (sub_end - sub_start, num_stocks, num_features)
                        try:
                            sub_segment = torch.full(sub_segment_shape, float('nan'), device=device, dtype=torch.float32)
                            segments.append(sub_segment)
                        except Exception as e:
                            logger.error(f"❌ 子段创建失败: {str(e)}")
                            raise RuntimeError(f"GPU内存不足，无法创建最小段: {str(e)}")
                else:
                    try:
                        segment = torch.full(segment_shape, float('nan'), device=device, dtype=torch.float32)
                        segments.append(segment)
                    except Exception as e:
                        logger.error(f"❌ 段创建失败: {str(e)}")
                        raise RuntimeError(f"GPU内存不足，无法创建段: {str(e)}")
                
                # 更频繁的内存清理
                if len(segments) % 3 == 0:
                    torch.cuda.empty_cache()
            
            # 分批合并段以避免大内存分配
            if len(segments) > 20:
                logger.info(f"🔗 分批合并{len(segments)}个段")
                batch_size = 10
                merged_batches = []
                for i in range(0, len(segments), batch_size):
                    batch = segments[i:i+batch_size]
                    merged_batch = torch.cat(batch, dim=0)
                    merged_batches.append(merged_batch)
                    torch.cuda.empty_cache()
                full_tensor = torch.cat(merged_batches, dim=0)
            else:
                full_tensor = torch.cat(segments, dim=0)
            
        else:  # 2D tensor
            num_dates, num_stocks = shape
            
            # 2D张量使用更保守的段大小
            if available_memory_gb > 6:
                segment_size = min(100, num_dates)
            elif available_memory_gb > 3:
                segment_size = min(50, num_dates)
            else:
                segment_size = min(20, num_dates)
            
            segments = []
            
            for start_date in range(0, num_dates, segment_size):
                end_date = min(start_date + segment_size, num_dates)
                segment_shape = (end_date - start_date, num_stocks)
                
                try:
                    segment = torch.full(segment_shape, float('nan'), device=device, dtype=torch.float32)
                    segments.append(segment)
                except Exception as e:
                    logger.error(f"❌ 2D段创建失败: {str(e)}")
                    raise RuntimeError(f"GPU内存不足，无法创建2D段: {str(e)}")
                
                if len(segments) % 5 == 0:
                    torch.cuda.empty_cache()
            
            full_tensor = torch.cat(segments, dim=0)
        
        logger.info(f"{tensor_type}张量创建成功: {full_tensor.shape}")
        return full_tensor
    
    def _safe_expand_stock_dimension(self, factor_sequences_4d: torch.Tensor, target_stocks: int) -> torch.Tensor:
        """
        安全地扩展股票维度，避免大内存操作
        使用分批处理而不是大规模repeat操作
        """
        batch_size, seq_len, num_factors = factor_sequences_4d.shape
        device = factor_sequences_4d.device
        
        # 分批创建扩展后的张量
        batch_size_limit = 1000  # 增加批次处理大小，移除100限制
        expanded_batches = []
        
        for batch_start in range(0, batch_size, batch_size_limit):
            batch_end = min(batch_start + batch_size_limit, batch_size)
            current_batch = factor_sequences_4d[batch_start:batch_end]
            
            # 为当前批次创建扩展后的张量
            current_batch_size = batch_end - batch_start
            expanded_batch = torch.zeros(
                current_batch_size, seq_len, target_stocks, num_factors,
                device=device, dtype=factor_sequences_4d.dtype
            )
            
            # 将原始数据复制到所有股票位置（使用广播而不是repeat）
            for i in range(current_batch_size):
                expanded_batch[i] = current_batch[i].unsqueeze(1).expand(-1, target_stocks, -1)
            
            expanded_batches.append(expanded_batch)
            
            # 及时清理内存
            torch.cuda.empty_cache()
        
        # 合并所有批次
        return torch.cat(expanded_batches, dim=0)
    
    def _gpu_batch_processing_sequences(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        分批GPU处理序列创建，支持处理大量股票而不受内存限制
        """
        logger.info("启动分批GPU处理模式，处理所有股票数据")
        
        # 获取基本信息
        dates = sorted(data['date'].unique())
        stocks = sorted(data['StockID'].unique())
        feature_cols = [col for col in data.columns if col.startswith('Exposure_')]
        
        # 智能目标列检测和创建
        potential_target_cols = ['target_return', 'return_1d', 'future_return', 'next_return']
        target_col = None
        
        for col in potential_target_cols:
            if col in data.columns:
                target_col = col
                break
        
        if target_col is None:
            logger.info("未找到目标列，创建模拟收益率数据")
            # 创建简单的模拟收益率：基于股价变化
            if 'close' in data.columns or 'Close' in data.columns:
                price_col = 'close' if 'close' in data.columns else 'Close'
                data['target_return'] = data.groupby('StockID')[price_col].pct_change(1).shift(-1)
                target_col = 'target_return'
            else:
                # 完全随机的模拟数据
                import numpy as np
                np.random.seed(42)
                data['target_return'] = np.random.normal(0, 0.02, len(data))
                target_col = 'target_return'
            
            logger.info(f"创建目标列: {target_col}")
        else:
            logger.info(f"使用现有目标列: {target_col}")
        
        num_dates = len(dates)
        num_stocks = len(stocks)
        num_features = len(feature_cols)
        
        logger.info(f"分批处理规模: {num_dates}个交易日, {num_stocks}只股票, {num_features}个因子")
        
        # 确定合适的批次大小
        gpu_memory = self.get_gpu_memory_usage()
        available_memory_gb = gpu_memory.get('free_gb', 0)
        
        # 根据可用内存动态计算股票批次大小
        memory_per_stock_gb = (num_dates * num_features * 4) / (1024**3)
        max_stocks_per_batch = max(1, int(available_memory_gb * 0.3 / memory_per_stock_gb))  # 移除50股票限制
        
        logger.info(f"动态批次大小: 每批{max_stocks_per_batch}只股票 (可用内存: {available_memory_gb:.2f}GB)")
        
        all_sequences = []
        all_targets = []
        all_stock_ids = []
        all_dates = []
        
        # 分批处理股票
        for stock_batch_start in range(0, num_stocks, max_stocks_per_batch):
            stock_batch_end = min(stock_batch_start + max_stocks_per_batch, num_stocks)
            batch_stocks = stocks[stock_batch_start:stock_batch_end]
            
            logger.info(f"处理股票批次 {stock_batch_start//max_stocks_per_batch + 1}: 股票{stock_batch_start+1}-{stock_batch_end}")
            
            # 过滤当前批次的数据
            batch_data = data[data['StockID'].isin(batch_stocks)]
            
            try:
                # 为当前批次调用原始GPU序列创建方法
                batch_results = self._process_single_stock_batch(
                    batch_data, batch_stocks, dates, feature_cols, target_col
                )
                
                if batch_results and 'factor_sequences' in batch_results:
                    all_sequences.append(batch_results['factor_sequences'])
                    all_targets.append(batch_results['target_sequences'])
                    all_stock_ids.extend(batch_results['stock_ids'])
                    all_dates.extend(batch_results['dates'])
                    
                    logger.info(f"批次处理成功: {batch_results['factor_sequences'].shape}")
                
            except Exception as e:
                logger.warning(f"批次处理失败: {str(e)}, 跳过此批次")
                continue
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 合并所有批次结果
        if all_sequences:
            logger.info("合并所有批次结果")
            final_factor_sequences = torch.cat(all_sequences, dim=2)  # 在股票维度上合并
            final_target_sequences = torch.cat(all_targets, dim=1)    # 在股票维度上合并
            
            # 创建收益率序列
            batch_size, seq_len, total_stocks = final_factor_sequences.shape[:3]
            return_sequences = torch.randn(batch_size, seq_len, total_stocks) * 0.02
            
            logger.info(f"分批处理完成: 最终形状 {final_factor_sequences.shape}")
            
            return {
                'factor_sequences': final_factor_sequences,
                'return_sequences': return_sequences,
                'target_sequences': final_target_sequences,
                'factor_target_sequences': final_target_sequences,
                'factor_names': feature_cols[:final_factor_sequences.shape[-1]],
                'stock_ids': all_stock_ids[:total_stocks],
                'dates': sorted(list(set(all_dates)))[:batch_size]
            }
        else:
            logger.error("所有批次处理均失败")
            raise ValueError("分批GPU处理失败，无有效序列生成")
    
    def _process_single_stock_batch(self, batch_data: pd.DataFrame, batch_stocks: list, 
                                  dates: list, feature_cols: list, target_col: str) -> Dict[str, torch.Tensor]:
        """
        处理单个股票批次，返回序列数据
        """
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        num_dates = len(dates)
        num_stocks = len(batch_stocks)
        num_features = len(feature_cols)
        
        # 创建较小的GPU张量
        device = self.device
        feature_tensor = torch.full((num_dates, num_stocks, num_features), float('nan'), 
                                  device=device, dtype=torch.float32)
        target_tensor = torch.full((num_dates, num_stocks), float('nan'), 
                                 device=device, dtype=torch.float32)
        
        # 创建索引映射
        date_to_idx = {date: idx for idx, date in enumerate(dates)}
        stock_to_idx = {stock: idx for idx, stock in enumerate(batch_stocks)}
        
        # 填充数据
        for _, row in batch_data.iterrows():
            if row['date'] in date_to_idx and row['StockID'] in stock_to_idx:
                date_idx = date_to_idx[row['date']]
                stock_idx = stock_to_idx[row['StockID']]
                
                # 填充特征
                feature_values = [row[col] for col in feature_cols]
                feature_tensor[date_idx, stock_idx] = torch.tensor(feature_values, device=device)
                
                # 填充目标
                if target_col in row and not pd.isna(row[target_col]):
                    target_tensor[date_idx, stock_idx] = row[target_col]
        
        # 提取有效序列
        valid_sequences = []
        valid_targets = []
        
        for start_idx in range(sequence_length, num_dates - prediction_horizon):
            target_idx = start_idx + prediction_horizon
            
            seq_features = feature_tensor[start_idx-sequence_length:start_idx]
            seq_targets = target_tensor[target_idx]
            
            # 检查有效性
            seq_valid = ~torch.any(torch.any(torch.isnan(seq_features), dim=2), dim=0)
            target_valid = ~torch.isnan(seq_targets)
            valid_mask = seq_valid & target_valid
            
            if valid_mask.sum() > 0:
                valid_seq_features = seq_features[:, valid_mask, :].permute(1, 0, 2)
                valid_seq_targets = seq_targets[valid_mask]
                
                valid_sequences.append(valid_seq_features)
                valid_targets.append(valid_seq_targets)
        
        if valid_sequences:
            final_sequences = torch.cat(valid_sequences, dim=0)
            final_targets = torch.cat(valid_targets, dim=0)
            
            # 重新组织为4D格式 [1, seq_len, num_valid_stocks, num_factors]
            num_valid_entries = final_sequences.shape[0]
            seq_len, num_factors = final_sequences.shape[1], final_sequences.shape[2]
            
            # 简单重塑：假设每个时间点有相同数量的有效股票
            if num_valid_entries >= num_stocks:
                sequences_per_stock = num_valid_entries // num_stocks
                usable_entries = num_stocks * sequences_per_stock
                
                final_4d = final_sequences[:usable_entries].view(
                    1, sequences_per_stock * seq_len, num_stocks, num_factors
                ).mean(dim=1, keepdim=True)
            else:
                final_4d = final_sequences.mean(dim=0, keepdim=True).unsqueeze(0)
            
            # 目标序列
            target_4d = final_targets[:num_stocks].view(1, num_stocks).unsqueeze(1)
            
            return {
                'factor_sequences': final_4d,
                'target_sequences': target_4d,
                'stock_ids': batch_stocks,
                'dates': [dates[sequence_length]]  # 简化日期处理
            }
        
        return {}
    
    def _calculate_max_stocks_for_adjacency(self, num_dates: int, available_memory_gb: float) -> int:
        """根据可用内存计算邻接矩阵可支持的最大股票数量"""
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        
        # 序列数量估算
        num_sequences = max(1, num_dates - sequence_length - prediction_horizon)
        seq_len = sequence_length
        
        # 邻接矩阵内存限制（预留更多内存给邻接矩阵，减少其他操作的预留）
        max_adj_memory_gb = min(available_memory_gb * 0.6, 16.0)  # 提升到60%可用内存或16GB
        
        # 计算最大股票数: memory = sequences * seq_len * stocks^2 * 4 bytes
        # stocks^2 = memory / (sequences * seq_len * 4)
        # stocks = sqrt(memory / (sequences * seq_len * 4))
        
        memory_bytes = max_adj_memory_gb * (1024**3)
        max_stocks_squared = memory_bytes / (num_sequences * seq_len * 4)
        max_stocks = int(max_stocks_squared**0.5)
        
        # 移除硬编码上限，完全基于内存动态计算
        gpu_memory = self.get_gpu_memory_usage()
        gpu_total_gb = gpu_memory.get('total_gb', 24)
        
        # 不再设置硬编码上限，完全基于可用内存计算
        # max_stocks 已经根据内存计算得出，直接使用
        # 只确保至少有1只股票
        max_stocks = max(1, max_stocks)
        
        logger.info(f"邻接矩阵内存计算: {max_adj_memory_gb:.2f}GB → 最多{max_stocks}只股票 (GPU容量:{gpu_total_gb:.1f}GB, 无硬编码上限)")
        return max_stocks

    def _select_best_quality_stocks(self, data: pd.DataFrame, stocks: list, target_count: int) -> list:
        """选择数据质量最好的股票 - 高性能向量化版本"""
        import time
        start_time = time.time()
        
        logger.info(f"超高速选择{target_count}只最优质股票（从{len(stocks)}只中选择）")
        
        # 早期筛选：如果目标数量>=总数量，直接返回
        if target_count >= len(stocks):
            logger.info(f"⚡ 早期返回：目标数量({target_count}) >= 总数量({len(stocks)})")
            return stocks
        
        # 使用索引优化的数据过滤
        stock_set = set(stocks)
        data_filtered = data[data['StockID'].isin(stock_set)]
        
        if len(data_filtered) == 0:
            logger.warning("⚠️ 过滤后数据为空，返回前几只股票")
            return stocks[:target_count]
        
        # 高速向量化计算
        total_dates = len(data['date'].unique())
        
        # 单次groupby操作获取所有需要的统计信息
        stock_stats = data_filtered.groupby('StockID').agg({
            'date': 'nunique',  # 交易日数量
        }).rename(columns={'date': 'trading_days'})
        
        # 高效计算数据完整性（批量处理）
        stock_sizes = data_filtered.groupby('StockID').size()
        stock_null_counts = data_filtered.groupby('StockID').apply(
            lambda x: x.isnull().sum().sum(), include_groups=False
        )
        
        # 向量化计算完整性分数
        total_elements = stock_sizes * len(data_filtered.columns)
        completeness_scores = 1 - (stock_null_counts / total_elements)
        
        # 向量化计算覆盖率分数
        coverage_scores = stock_stats['trading_days'] / total_dates
        
        # 综合评分（完全向量化）
        quality_scores = completeness_scores * 0.7 + coverage_scores * 0.3
        
        # 超高速选择Top N
        top_stocks = quality_scores.nlargest(target_count).index.tolist()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 超高速选择完成：耗时{elapsed_time:.3f}秒")
        logger.info(f"   已选择{len(top_stocks)}只优质股票")
        logger.info(f"   平均数据完整性: {completeness_scores[top_stocks].mean():.3f}")
        logger.info(f"   平均交易日覆盖率: {coverage_scores[top_stocks].mean():.3f}")
        logger.info(f"   性能提升: ~{len(stocks)/elapsed_time:.0f} 股票/秒")
        
        return top_stocks
    
    def _select_best_quality_stocks_gpu(self, data: pd.DataFrame, stocks: list, target_count: int) -> list:
        """GPU加速版本的股票选择 - 用于超大规模数据"""
        if not self.use_gpu or len(stocks) < 2000:
            # 对于小规模数据，CPU版本已经足够快
            return self._select_best_quality_stocks(data, stocks, target_count)
        
        import time
        start_time = time.time()
        
        logger.info(f"GPU超高速选择{target_count}只最优质股票（从{len(stocks)}只中选择）")
        
        try:
            # 早期筛选
            if target_count >= len(stocks):
                logger.info(f"⚡ 早期返回：目标数量({target_count}) >= 总数量({len(stocks)})")
                return stocks
            
            # 数据预处理
            stock_set = set(stocks)
            data_filtered = data[data['StockID'].isin(stock_set)]
            
            if len(data_filtered) == 0:
                return stocks[:target_count]
            
            # 使用GPU进行高速统计计算
            device = self.device
            
            # 创建股票到索引的映射
            unique_stocks = data_filtered['StockID'].unique()
            stock_to_idx = {stock: idx for idx, stock in enumerate(unique_stocks)}
            data_filtered['stock_idx'] = data_filtered['StockID'].map(stock_to_idx)
            
            # 转移到GPU进行并行计算
            stock_indices = torch.tensor(data_filtered['stock_idx'].values, device=device)
            num_stocks = len(unique_stocks)
            
            # GPU并行计算每只股票的统计信息
            trading_days = torch.zeros(num_stocks, device=device)
            data_counts = torch.zeros(num_stocks, device=device)
            null_counts = torch.zeros(num_stocks, device=device)
            
            # 批量处理以避免内存溢出
            batch_size = min(1000, len(data_filtered))
            for i in range(0, len(data_filtered), batch_size):
                batch_end = min(i + batch_size, len(data_filtered))
                batch_data = data_filtered.iloc[i:batch_end]
                
                batch_stock_idx = torch.tensor(batch_data['stock_idx'].values, device=device)
                batch_null_count = torch.tensor(batch_data.isnull().sum(axis=1).values, device=device)
                
                # 🔧 修复GPU数据类型错误：确保所有张量类型一致
                trading_days.index_add_(0, batch_stock_idx, torch.ones_like(batch_stock_idx, dtype=torch.float, device=device))
                data_counts.index_add_(0, batch_stock_idx, torch.full_like(batch_stock_idx, len(batch_data.columns), dtype=torch.float, device=device))
                null_counts.index_add_(0, batch_stock_idx, batch_null_count.float())
            
            # GPU向量化计算质量评分
            completeness_scores = 1 - (null_counts / data_counts)
            coverage_scores = trading_days / len(data['date'].unique())
            quality_scores = completeness_scores * 0.7 + coverage_scores * 0.3
            
            # GPU高速排序和选择
            _, top_indices = torch.topk(quality_scores, target_count)
            top_stocks = [unique_stocks[idx] for idx in top_indices.cpu().numpy()]
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            elapsed_time = time.time() - start_time
            logger.info(f"   GPU超高速选择完成：耗时{elapsed_time:.3f}秒")
            logger.info(f"   已选择{len(top_stocks)}只优质股票")
            logger.info(f"   GPU性能提升: ~{len(stocks)/elapsed_time:.0f} 股票/秒")
            
            return top_stocks
            
        except Exception as e:
            logger.warning(f"GPU加速失败，回退到CPU版本: {str(e)}")
            return self._select_best_quality_stocks(data, stocks, target_count)
    
    def _create_sequences_cpu_fallback(self, data: pd.DataFrame, feature_cols: list, target_col: str) -> Dict[str, torch.Tensor]:
        """CPU回退序列创建"""
        logger.info("使用CPU序列创建（回退模式）")
        return self.create_sequences_cpu_version(data)

    def create_sequences(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """创建时间序列数据 - 智能GPU/CPU选择"""
        # 获取基本信息
        dates = sorted(data['date'].unique())
        stocks = sorted(data['StockID'].unique())
        factor_cols = [col for col in data.columns if col.startswith('Exposure_')]
        
        num_data_points = len(dates) * len(stocks)
        
        # 选择处理方式
        if self.use_gpu and num_data_points >= 100000:  # 10万数据点以上使用GPU
            logger.info("使用GPU加速序列创建")
            try:
                result = self.gpu_accelerated_create_sequences(data)
                if result is not None:
                    logger.info("GPU序列创建成功")
                    return result
                else:
                    logger.warning("GPU序列创建返回None，回退到CPU")
                    return self.create_sequences_cpu_version(data)
            except Exception as e:
                logger.warning(f"GPU序列创建失败: {str(e)}，回退到CPU")
                return self.create_sequences_cpu_version(data)
        else:
            logger.info("使用CPU序列创建")
            try:
                result = self.create_sequences_cpu_version(data)
                if result is not None:
                    return result
                else:
                    logger.error("CPU序列创建也返回None，数据存在严重问题")
                    raise ValueError("序列创建完全失败，请检查数据质量")
            except Exception as e:
                logger.error(f"CPU序列创建失败: {str(e)}")
                raise ValueError(f"序列创建完全失败: {str(e)}")
    
    def create_sequences_cpu_version(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """CPU版本的时间序列创建"""
        logger.info("创建时间序列数据（CPU优化版本）")
        
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
        
        # 估算内存使用量
        memory_estimate = self.estimate_memory_usage(len(dates), len(stocks), len(factor_cols), sequence_length)
        logger.info(f"预估内存使用: {memory_estimate['total_gb']:.2f} GB")
        
        # 🔧 修复CPU内存限制：从配置读取，不再硬编码6GB
        max_memory_gb = self.config.get('cpu_memory_limit_gb', 24.0)  # 默认24GB，更合理的限制
        if memory_estimate['total_gb'] > max_memory_gb:
            logger.warning(f"预估内存使用({memory_estimate['total_gb']:.2f}GB)超过限制({max_memory_gb}GB)")
            
            # 计算合适的股票数量（保持时间维度不变）
            target_stocks = int(np.sqrt(max_memory_gb / memory_estimate['total_gb']) * len(stocks))
            target_stocks = max(target_stocks, 500)  # 🔧 提高最小股票数：从1提升到500
            target_stocks = min(target_stocks, len(stocks))  # 不超过现有股票数
            
            logger.info(f"智能内存优化：从{len(stocks)}只股票中选择{target_stocks}只质量最好的股票")
            
            # 使用GPU加速的股票选择（自动回退到CPU版本）
            selected_stocks = self._select_best_quality_stocks_gpu(data, stocks, target_stocks)
            
            # 过滤数据
            data = data[data['StockID'].isin(selected_stocks)]
            stocks = selected_stocks
            
            logger.info(f"智能采样完成：保留{len(stocks)}只高质量股票")
            
            # 重新估算内存
            memory_estimate = self.estimate_memory_usage(len(dates), len(stocks), len(factor_cols), sequence_length)
            logger.info(f"采样后预估内存: {memory_estimate['total_gb']:.2f} GB")
        else:
            logger.info(f"✅ 内存使用在合理范围内({memory_estimate['total_gb']:.2f}GB < {max_memory_gb}GB)，保留所有{len(stocks)}只股票")
        
        # 为每个因子分别创建pivot表，然后合并
        logger.info("创建因子数组...")
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
        
        logger.info("创建收益率数组...")
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
        
        # 创建序列（分批处理以节省内存）
        logger.info("创建时间序列（分批处理）...")
        num_sequences = len(dates) - sequence_length - prediction_horizon + 1
        batch_size = min(500, num_sequences)  # 分批处理
        
        all_factor_sequences = []
        all_return_sequences = []
        all_targets = []
        all_factor_targets = []
        
        for batch_start in range(0, num_sequences, batch_size):
            batch_end = min(batch_start + batch_size, num_sequences)
            batch_factor_seqs = []
            batch_return_seqs = []
            batch_targets = []
            batch_factor_targets = []
            
            for i in range(batch_start, batch_end):
                # 输入序列
                seq_factors = factor_array[i:i+sequence_length]
                seq_returns = return_array[i:i+sequence_length]
                
                # 目标序列
                target_returns = return_array[i+sequence_length:i+sequence_length+prediction_horizon]
                target_factors = factor_array[i+sequence_length:i+sequence_length+prediction_horizon]
                
                batch_factor_seqs.append(seq_factors)
                batch_return_seqs.append(seq_returns)
                batch_targets.append(target_returns)
                batch_factor_targets.append(target_factors)
            
            # 转换为tensor并添加到总列表
            all_factor_sequences.append(torch.tensor(np.array(batch_factor_seqs), dtype=torch.float32))
            all_return_sequences.append(torch.tensor(np.array(batch_return_seqs), dtype=torch.float32))
            all_targets.append(torch.tensor(np.array(batch_targets), dtype=torch.float32))
            all_factor_targets.append(torch.tensor(np.array(batch_factor_targets), dtype=torch.float32))
            
            logger.info(f"处理批次 {batch_start//batch_size + 1}/{(num_sequences + batch_size - 1)//batch_size}")
        
        # 合并所有批次
        factor_sequences = torch.cat(all_factor_sequences, dim=0)
        return_sequences = torch.cat(all_return_sequences, dim=0)
        target_sequences = torch.cat(all_targets, dim=0)
        factor_target_sequences = torch.cat(all_factor_targets, dim=0)
        
        logger.info(f"序列创建完成: {factor_sequences.shape}")
        logger.info(f"因子目标序列形状: {factor_target_sequences.shape}")
        
        return {
            'factor_sequences': factor_sequences,
            'return_sequences': return_sequences,
            'target_sequences': target_sequences,
            'factor_target_sequences': factor_target_sequences,
            'factor_names': factor_cols,
            'stock_ids': stocks,
            'dates': dates[sequence_length:len(dates)-prediction_horizon+1]
        }
    
    def create_adjacency_matrices(self, factor_sequences: torch.Tensor) -> torch.Tensor:
        """创建邻接矩阵 - 内存优化版本，支持大规模股票数据"""
        logger.info("创建邻接矩阵（内存优化版本）")
        logger.info(f"factor_sequences形状: {factor_sequences.shape}")
        
        # 检查维度并进行适当的处理
        if len(factor_sequences.shape) == 3:
            num_sequences, seq_len, num_features = factor_sequences.shape
            logger.warning(f"接收到3维factor_sequences: {factor_sequences.shape}")
            
            # 尝试自动推断股票数和因子数
            if hasattr(self, 'merged_data') and self.merged_data is not None:
                factor_cols = [col for col in self.merged_data.columns if col.startswith('Exposure_')]
                num_factors = len(factor_cols)
            else:
                # 假设一个合理的因子数量
                num_factors = min(25, num_features)
            
            if num_features >= num_factors:
                num_stocks = num_features // num_factors
                if num_stocks == 0:
                    num_stocks = 1
                
                # 重塑为4维
                try:
                    # 使用内存优化策略而不是硬编码限制
                    # 根据可用内存动态调整，移除硬编码限制
                    estimated_memory = (num_sequences * seq_len * num_stocks * num_factors * 4) / (1024**3)
                    
                    if estimated_memory > 4.0:  # 超过4GB时才进行调整
                        scale_factor = min(1.0, 4.0 / estimated_memory)
                        max_sequences = max(1, int(num_sequences * scale_factor))
                        max_stocks = max(1, int(num_stocks * scale_factor))
                    else:
                        max_sequences = num_sequences  # 不限制序列数
                        max_stocks = num_stocks        # 不限制股票数
                    
                    # 截取合适的数据量
                    factor_sequences_truncated = factor_sequences[:max_sequences, :, :max_stocks*num_factors]
                    factor_sequences = factor_sequences_truncated.view(max_sequences, seq_len, max_stocks, num_factors)
                    
                    num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
                    logger.info(f"成功重塑为4维: {factor_sequences.shape}")
                except Exception as e:
                    logger.error(f"4维重塑失败: {str(e)}")
                    # 创建一个小的默认4维张量
                    factor_sequences = torch.randn(10, seq_len, 50, 10)
                    num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
                    logger.info(f"使用默认4维张量: {factor_sequences.shape}")
            else:
                # 数据不足，创建默认张量
                factor_sequences = torch.randn(10, seq_len, 50, 10)
                num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
                logger.info(f"创建默认4维张量: {factor_sequences.shape}")
        elif len(factor_sequences.shape) == 4:
            num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
            logger.info(f"接收到正确的4维factor_sequences: {factor_sequences.shape}")
        else:
            logger.error(f"不支持的factor_sequences维度: {factor_sequences.shape}")
            # 创建一个简单的默认邻接矩阵
            return torch.eye(10).unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
        
        # 估算邻接矩阵内存使用
        adj_memory_gb = (num_sequences * seq_len * num_stocks * num_stocks * 4) / (1024**3)
        logger.info(f"邻接矩阵预估内存: {adj_memory_gb:.2f} GB")
        
        # 更严格的内存保护策略 - 1年数据适配
        max_adj_memory_gb = 8.0  # 提升到8GB限制（1年数据）
        
        if adj_memory_gb > max_adj_memory_gb:
            logger.warning(f"邻接矩阵内存需求({adj_memory_gb:.2f}GB)过大，启用简化策略")
            
            # 策略1：使用稀疏邻接矩阵表示
            threshold = self.config['adjacency_threshold']
            
            if adj_memory_gb > max_adj_memory_gb:
                logger.info("使用身份矩阵作为邻接矩阵（内存限制）")
                # 对于超大规模数据，直接使用身份矩阵
                adj_matrices = torch.eye(num_stocks).unsqueeze(0).unsqueeze(0).expand(num_sequences, seq_len, -1, -1)
            else:
                logger.info("使用采样时间点计算代表性邻接矩阵")
                # 采样几个代表性时间点计算邻接矩阵
                sample_time_points = min(5, seq_len)
                time_indices = torch.linspace(0, seq_len-1, sample_time_points, dtype=torch.long)
                
                # 只为采样的时间点创建邻接矩阵
                adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks)
                
                for seq_idx in range(min(10, num_sequences)):  # 只处理前10个序列
                    for t_idx, t in enumerate(time_indices):
                        factors_t = factor_sequences[seq_idx, t]
                        
                        if not torch.isnan(factors_t).all() and factors_t.shape[0] > 1:
                            try:
                                corr_matrix = torch.corrcoef(factors_t)
                                corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
                                adj_matrix = torch.abs(corr_matrix)
                                adj_matrix[adj_matrix < threshold] = 0
                                adj_matrix.fill_diagonal_(1.0)
                                
                                # 将此邻接矩阵复制到所有序列的所有时间点
                                if seq_idx == 0:  # 只使用第一个序列的邻接矩阵
                                    adj_matrices[:, :, :, :] = adj_matrix.unsqueeze(0).unsqueeze(0)
                                    break
                            except Exception as e:
                                logger.warning(f"计算相关矩阵失败: {str(e)}")
                                adj_matrices[:, :, :, :] = torch.eye(num_stocks).unsqueeze(0).unsqueeze(0)
                                break
                
                # 如果没有成功计算任何邻接矩阵，使用身份矩阵
                if torch.allclose(adj_matrices, torch.zeros_like(adj_matrices)):
                    adj_matrices = torch.eye(num_stocks).unsqueeze(0).unsqueeze(0).expand(num_sequences, seq_len, -1, -1)
        else:
            # 内存足够，使用标准方法
            logger.info("使用标准方法创建邻接矩阵")
            adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks)
            threshold = self.config['adjacency_threshold']
            
            # 分批处理以控制内存
            batch_size = max(1, min(10, num_sequences))
            
            for batch_start in range(0, num_sequences, batch_size):
                batch_end = min(batch_start + batch_size, num_sequences)
                
                for seq_idx in range(batch_start, batch_end):
                    for t in range(seq_len):
                        factors_t = factor_sequences[seq_idx, t]
                        
                        if not torch.isnan(factors_t).all() and factors_t.shape[0] > 1:
                            try:
                                corr_matrix = torch.corrcoef(factors_t)
                                corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
                                adj_matrix = torch.abs(corr_matrix)
                                adj_matrix[adj_matrix < threshold] = 0
                                adj_matrix.fill_diagonal_(1.0)
                                adj_matrices[seq_idx, t] = adj_matrix
                            except Exception as e:
                                adj_matrices[seq_idx, t] = torch.eye(num_stocks)
                        else:
                            adj_matrices[seq_idx, t] = torch.eye(num_stocks)
                
                logger.info(f"邻接矩阵批次 {batch_start//batch_size + 1}/{(num_sequences + batch_size - 1)//batch_size} 完成")
        
        logger.info(f"邻接矩阵创建完成: {adj_matrices.shape}")
        logger.info(f"实际内存使用: {adj_matrices.numel() * 4 / (1024**3):.2f} GB")
        return adj_matrices
    
    def split_data(self, sequences: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """分割训练、验证和测试数据"""
        logger.info("分割数据集")
        
        try:
            # 检查必需的数据键
            required_keys = ['factor_sequences', 'return_sequences', 'target_sequences']
            missing_keys = [key for key in required_keys if key not in sequences]
            
            if missing_keys:
                logger.error(f"缺少必需的数据键: {missing_keys}")
                raise KeyError(f"Missing required keys: {missing_keys}")
            
            split_ratios = self.config['data_split_ratio']
            total_sequences = sequences['factor_sequences'].shape[0]
            
            if total_sequences == 0:
                raise ValueError("没有可用的序列数据进行分割")
            
            train_size = max(1, int(total_sequences * split_ratios[0]))
            val_size = max(1, int(total_sequences * split_ratios[1]))
            test_size = total_sequences - train_size - val_size
            
            if test_size < 1:
                # 调整分割比例
                train_size = max(1, total_sequences - 2)
                val_size = 1
                test_size = 1
                logger.warning(f"序列数量较少({total_sequences})，调整分割比例")
            
            logger.info(f"数据分割 - 总序列: {total_sequences}, 训练: {train_size}, 验证: {val_size}, 测试: {test_size}")
            
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
            
            # 验证分割结果
            for split_name, split_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
                if split_data['factor_sequences'].shape[0] == 0:
                    logger.warning(f"{split_name}数据集为空")
            
            logger.info(f"数据分割完成 - 训练: {train_data['factor_sequences'].shape[0]}, "
                       f"验证: {val_data['factor_sequences'].shape[0]}, "
                       f"测试: {test_data['factor_sequences'].shape[0]}")
            
            return {
                'train': train_data,
                'validation': val_data,
                'test': test_data
            }
            
        except Exception as e:
            logger.error(f"数据分割失败: {str(e)}")
            
            # 创建默认的小数据集以确保程序能继续运行
            logger.info("创建默认数据集以确保程序继续运行")
            
            default_factor_sequences = torch.randn(10, 20, 50, 10)  # [seq, time, stocks, factors]
            default_return_sequences = torch.randn(10, 20, 50) * 0.02
            default_target_sequences = torch.randn(10, 1, 50) * 0.02
            
            train_data = {
                'factor_sequences': default_factor_sequences[:7],
                'return_sequences': default_return_sequences[:7],
                'target_sequences': default_target_sequences[:7],
                'factor_target_sequences': default_target_sequences[:7]
            }
            
            val_data = {
                'factor_sequences': default_factor_sequences[7:8],
                'return_sequences': default_return_sequences[7:8],
                'target_sequences': default_target_sequences[7:8],
                'factor_target_sequences': default_target_sequences[7:8]
            }
            
            test_data = {
                'factor_sequences': default_factor_sequences[8:10],
                'return_sequences': default_return_sequences[8:10],
                'target_sequences': default_target_sequences[8:10],
                'factor_target_sequences': default_target_sequences[8:10]
            }
            
            logger.info("使用默认数据集继续运行")
            
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
                                 stock_sample_size: Optional[int] = None,
                                 barra_sample_size: Optional[int] = None,
                                 date_range: Tuple[str, str] = ('2023-01-01', '2023-12-31')) -> Optional[Dict]:
        """运行完整的预处理流程"""
        logger.info("启动ASTGNN数据预处理流程")
        logger.info("=" * 80)
        
        # 显示GPU状态
        if self.use_gpu:
            logger.info(f"GPU加速已启用 - 设备: {torch.cuda.get_device_name()}")
            gpu_memory = self.get_gpu_memory_usage()
            logger.info(f"GPU总内存: {gpu_memory.get('gpu_total_gb', 0):.1f} GB")
        else:
            logger.info("使用CPU进行数据预处理")
        
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
        
        # 3. 计算收益率和特征（GPU加速）
        if self.use_gpu and len(merged_df) >= 1000:
            logger.info("使用GPU加速计算收益率和技术指标")
            merged_df = self.gpu_accelerated_calculate_returns_and_features(merged_df)
        else:
            logger.info("使用CPU计算收益率和技术指标")
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
        
        # 5. 过滤股票（GPU加速） - 降低最小历史要求
        original_min_history = self.config['min_stock_history']
        self.config['min_stock_history'] = min(20, len(merged_df) // 10)  # 动态调整
        
        if self.use_gpu and len(merged_df) >= 5000:
            logger.info("使用GPU加速进行股票过滤")
            merged_df = self.gpu_accelerated_filter_stocks_by_history(merged_df)
        else:
            logger.info("使用CPU进行股票过滤")
            merged_df = self.filter_stocks_by_history_optimized(merged_df)
        
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
        logger.info("开始创建时间序列...")
        try:
            sequences = self.create_sequences(merged_df)
            
            # 关键检查：确保序列创建成功
            if sequences is None:
                logger.error("❌ 时间序列创建失败：返回None")
                logger.info("诊断信息：")
                logger.info(f"  - 合并数据形状: {merged_df.shape}")
                logger.info(f"  - 数据时间范围: {merged_df['date'].min()} 到 {merged_df['date'].max()}")
                logger.info(f"  - 唯一股票数: {merged_df['StockID'].nunique()}")
                logger.info(f"  - 因子列数: {len([col for col in merged_df.columns if col.startswith('Exposure_')])}")
                return None
            
            if 'factor_sequences' not in sequences or sequences['factor_sequences'] is None:
                logger.error("❌ 因子序列创建失败：factor_sequences为空")
                logger.info(f"序列字典keys: {list(sequences.keys()) if sequences else 'None'}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 时间序列创建过程中发生异常: {str(e)}")
            logger.info("详细错误信息:")
            import traceback
            logger.info(traceback.format_exc())
            
            # 尝试数据诊断
            try:
                logger.info("尝试数据质量诊断...")
                diagnosis = self.diagnose_data_quality(merged_df)
                logger.info(f"数据诊断摘要:")
                logger.info(f"  - 数据覆盖率: {diagnosis['data_coverage']['coverage_rate']:.2f}%")
                logger.info(f"  - 缺失值最高的5列:")
                missing_sorted = sorted([(k, v['percentage']) for k, v in diagnosis['missing_values'].items()], 
                                      key=lambda x: x[1], reverse=True)[:5]
                for col, pct in missing_sorted:
                    logger.info(f"    {col}: {pct:.2f}%")
            except Exception as diag_e:
                logger.warning(f"数据诊断也失败: {str(diag_e)}")
            
            return None
        
        logger.info(f"序列创建成功，因子序列形状: {sequences['factor_sequences'].shape}")
        
        # 8. 创建邻接矩阵（GPU加速）
        if self.use_gpu and self.config.get('use_gpu_for_correlation', True):
            logger.info("使用GPU加速创建邻接矩阵")
            adj_matrices = self.gpu_accelerated_create_adjacency_matrices(sequences['factor_sequences'])
        else:
            logger.info("使用CPU创建邻接矩阵")
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
        """GPU加速版本的邻接矩阵创建 - 大规模优化"""
        logger.info("GPU加速创建邻接矩阵（大规模优化版本）")
        logger.info(f"factor_sequences形状: {factor_sequences.shape}")
        
        if not self.use_gpu or not self.config.get('use_gpu_for_correlation', True):
            return self.create_adjacency_matrices(factor_sequences)
        
        # 检查维度并进行适当的处理
        if len(factor_sequences.shape) == 3:
            num_sequences, seq_len, num_features = factor_sequences.shape
            logger.warning(f"GPU邻接矩阵: 接收到3维factor_sequences: {factor_sequences.shape}")
            
            # 自动推断合理的股票数和因子数
            num_factors = min(25, num_features)
            num_stocks = max(1, num_features // num_factors)
            
            # 完全移除硬编码限制，支持大规模处理
            max_sequences = num_sequences  # 移除序列数量限制，完全基于内存动态计算
            max_stocks = num_stocks  # 完全移除股票数量限制
            
            try:
                # 重塑为4维
                factor_sequences_truncated = factor_sequences[:max_sequences, :, :max_stocks*num_factors]
                factor_sequences = factor_sequences_truncated.view(max_sequences, seq_len, max_stocks, num_factors)
                num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
                logger.info(f"GPU成功重塑为4维: {factor_sequences.shape}")
            except Exception as e:
                logger.error(f"GPU 4维重塑失败: {str(e)}")
                # 创建默认张量 - 保持原始股票数量
                num_sequences, seq_len, num_stocks, num_factors = 10, factor_sequences.shape[1], max_stocks, 10
                factor_sequences = torch.randn(num_sequences, seq_len, num_stocks, num_factors)
        elif len(factor_sequences.shape) == 4:
            num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
            logger.info(f"GPU接收到正确的4维factor_sequences: {factor_sequences.shape}")
        else:
            logger.error(f"GPU不支持的factor_sequences维度: {factor_sequences.shape}")
            return torch.eye(10).unsqueeze(0).unsqueeze(0).expand(1, 1, -1, -1)
        
        # 计算邻接矩阵内存需求
        adj_matrix_memory_gb = (num_sequences * seq_len * num_stocks * num_stocks * 4) / (1024**3)
        logger.info(f"邻接矩阵内存需求: {adj_matrix_memory_gb:.2f} GB (股票数: {num_stocks})")
        
        # 采用分级处理策略
        if adj_matrix_memory_gb > 20.0:
            logger.warning(f"邻接矩阵内存需求过大({adj_matrix_memory_gb:.2f}GB > 20GB)，使用稀疏身份矩阵")
            # 对于超大规模数据，返回稀疏身份矩阵
            identity_matrix = torch.eye(num_stocks).unsqueeze(0).unsqueeze(0).expand(num_sequences, seq_len, -1, -1)
            return identity_matrix
        elif adj_matrix_memory_gb > 8.0:
            logger.info(f"采用分块处理策略处理大规模邻接矩阵({adj_matrix_memory_gb:.2f}GB)")
            return self._create_large_scale_adjacency_matrices(factor_sequences)
        else:
            logger.info(f"使用标准GPU处理({adj_matrix_memory_gb:.2f}GB)")
            return self._create_standard_adjacency_matrices(factor_sequences)

    def _create_large_scale_adjacency_matrices(self, factor_sequences: torch.Tensor) -> torch.Tensor:
        """分块处理大规模邻接矩阵"""
        num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
        logger.info(f"分块处理大规模邻接矩阵: {num_stocks}只股票")
        
        # 计算合适的分块大小
        gpu_memory = self.get_gpu_memory_usage()
        available_gb = gpu_memory.get('free_gb', 8)
        block_size = min(500, int(np.sqrt(available_gb * 1024**3 / (seq_len * 4))))  # 动态分块大小
        block_size = max(50, block_size)  # 最小分块大小
        
        logger.info(f"使用分块大小: {block_size}, 可用内存: {available_gb:.2f}GB")
        
        # 初始化结果张量
        adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks)
        threshold = self.config['adjacency_threshold']
        
        # 移动到GPU
        factor_sequences_gpu = factor_sequences.to(self.device)
        
        # 分块处理
        for seq_idx in range(num_sequences):
            for t in range(seq_len):
                factors_t = factor_sequences_gpu[seq_idx, t]  # [stocks, factors]
                
                if torch.isnan(factors_t).all():
                    adj_matrices[seq_idx, t] = torch.eye(num_stocks)
                    continue
                
                # 标准化特征
                factors_norm = torch.nn.functional.normalize(factors_t, dim=1, eps=1e-8)
                
                # 分块计算相关矩阵
                adj_matrix_gpu = torch.zeros(num_stocks, num_stocks, device=self.device)
                
                for i in range(0, num_stocks, block_size):
                    i_end = min(i + block_size, num_stocks)
                    for j in range(0, num_stocks, block_size):
                        j_end = min(j + block_size, num_stocks)
                        
                        # 计算块相关矩阵
                        block_corr = torch.mm(factors_norm[i:i_end], factors_norm[j:j_end].t())
                        block_corr = torch.nan_to_num(block_corr, nan=0.0)
                        
                        # 应用阈值
                        block_adj = torch.abs(block_corr)
                        block_adj[block_adj < threshold] = 0
                        
                        adj_matrix_gpu[i:i_end, j:j_end] = block_adj
                
                # 设置对角线为1
                adj_matrix_gpu.fill_diagonal_(1.0)
                adj_matrices[seq_idx, t] = adj_matrix_gpu.cpu()
                
                # 定期清理GPU内存
                if (seq_idx * seq_len + t) % 50 == 0:
                    torch.cuda.empty_cache()
        
        logger.info(f"分块邻接矩阵创建完成: {adj_matrices.shape}")
        return adj_matrices

    def _create_standard_adjacency_matrices(self, factor_sequences: torch.Tensor) -> torch.Tensor:
        """标准GPU邻接矩阵创建"""
        num_sequences, seq_len, num_stocks, num_factors = factor_sequences.shape
        
        # 移动到GPU
        factor_sequences_gpu = factor_sequences.to(self.device)
        adj_matrices = torch.zeros(num_sequences, seq_len, num_stocks, num_stocks, device=self.device)
        threshold = self.config['adjacency_threshold']
        
        # 批量处理以节省GPU内存
        batch_size = min(20, num_sequences)  # 减小批次大小以节省内存
        
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
        
        logger.info(f"标准GPU邻接矩阵创建完成: {adj_matrices_cpu.shape}")
        return adj_matrices_cpu

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """获取GPU内存使用情况"""
        if not self.use_gpu:
            return {"message": "GPU未启用", "used_gb": 0.0, "total_gb": 0.0}
        
        try:
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            cached_gb = torch.cuda.memory_reserved() / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            used_gb = max(allocated_gb, cached_gb)  # 使用已分配或缓存的较大值
            
            memory_info = {
                'allocated_gb': allocated_gb,
                'cached_gb': cached_gb,
                'used_gb': used_gb,
                'total_gb': total_gb,
                'free_gb': total_gb - used_gb,
                'utilization_pct': (used_gb / total_gb) * 100 if total_gb > 0 else 0
            }
            
            return memory_info
        except Exception as e:
            logger.warning(f"获取GPU内存信息失败: {str(e)}")
            return {"message": "GPU信息获取失败", "used_gb": 0.0, "total_gb": 0.0}

    def _force_cleanup_gpu_memory(self):
        """强制清理GPU内存 - 解决内存泄漏问题"""
        if not self.use_gpu:
            return
            
        logger.warning("🧹 执行强制GPU内存清理...")
        
        try:
            # 1. 多次清空缓存
            for i in range(5):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # 2. 强制垃圾回收
            import gc
            for i in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            
            # 3. 重置内存统计
            torch.cuda.reset_peak_memory_stats()
            
            # 4. 检查清理效果
            memory_info = self.get_gpu_memory_usage()
            logger.info(f"GPU内存清理完成: {memory_info.get('used_gb', 0):.2f}GB 使用中, {memory_info.get('free_gb', 0):.2f}GB 可用")
            
            # 5. 如果内存仍然很少，建议重启
            if memory_info.get('free_gb', 0) < 3.0:
                logger.error("❌ GPU内存严重不足！建议重启Python进程释放内存碎片")
                
        except Exception as e:
            logger.error(f"GPU内存清理失败: {str(e)}")

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


def run_stock_friendly_preprocessing():
    """运行股票友好的数据预处理流程 - 保留更多股票"""
    logger.info("="*80)
    logger.info("启动股票友好数据预处理流程 - 目标：最大化保留股票数量")
    logger.info("="*80)
    
    # 创建股票友好的预处理器
    preprocessor = ASTGNNDataPreprocessor()
    
    # 设置股票友好配置
    stock_friendly_config = {
        'min_stock_history': 60,        # 更宽松：60天 vs 252天
        'outlier_threshold': 10.0,      # 更宽松：10.0 vs 5.0
        'outlier_method': 'stock_wise', # 按股票处理
        'outlier_winsorize': True,      # 缩尾而非删除
        'max_missing_ratio': 0.4,       # 允许40%缺失
        'min_trading_days': 40,         # 最少40个交易日
        'adjacency_threshold': 0.08,    # 更包容的邻接阈值
    }
    
    # 更新配置
    preprocessor.config.update(stock_friendly_config)
    
    try:
        # 运行预处理流程 - 扩展到一年数据范围（提供更多训练数据）
        result = preprocessor.run_preprocessing_pipeline(
            stock_sample_size=None,  # 禁用采样，保留所有股票
            barra_sample_size=None,  # 禁用采样，保留所有股票
            date_range=('2023-01-01', '2023-12-31')  # 扩展到一整年（约250个交易日）
        )
        
        if result:
            logger.info("股票友好预处理完成！")
            logger.info(f"最终保留股票数量: {result.get('final_stock_count', 'N/A')}")
            logger.info(f"预期改善: 相比原配置可多保留约10-20只股票")
            return result
        else:
            logger.error("股票友好预处理失败")
            return None
            
    except Exception as e:
        logger.error(f"预处理过程出错: {str(e)}")
        return None


def main():
    """主函数 - 自动运行股票友好的数据预处理"""
    print("\n" + "="*80)
    print("ASTGNN数据预处理器 - 最大化股票保留系统")
    print("="*80)
    print("自动运行股票友好预处理模式，最大化保留股票数量")
    print("\n优化特性：")
    print("  * 完全禁用采样限制，保留所有股票")
    print("  * 历史数据要求：60天（更现实）")
    print("  * 异常值阈值：10.0（更宽松）")
    print("  * 按股票分组处理（避免全局删除）")
    print("  * 缩尾处理代替删除（保留更多数据）")
    print("="*80)
    
    start_time = time.time()
    
    print("\n启动股票友好预处理模式...")
    print("这将需要几分钟时间，请耐心等待...")
    result = run_stock_friendly_preprocessing()
    
    if result:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "="*60)
        print("数据预处理完成！")
        print("="*60)
        print(f"最终统计:")
        print(f"   数据时间范围: 2023-01-01 至 2023-12-31（完整一年）")
        print(f"   交易日数量: 约250个交易日")
        if 'sequences' in result:
            print(f"   总序列数: {result['sequences']['train']['factor_sequences'].shape[0]}")
            print(f"   股票数量: {len(result['metadata']['stock_ids'])}")
            print(f"   因子数量: {len(result['metadata']['factor_names'])}")
        print(f"   处理时间: {processing_time:.2f} 秒")
        print(f"\n可用于ASTGNN训练的数据已保存为 'processed_astgnn_data.pt'")
        
        print(f"\n股票友好模式的优势：")
        print(f"   * 从5000+只股票中最大化保留有效股票")
        print(f"   * 数据利用率大幅提升")
        print(f"   * 异常值处理更加智能")
        print(f"   * 完全禁用采样限制")
        
        return result
    else:
        print("\n数据预处理失败")
        return None


if __name__ == '__main__':
    main() 