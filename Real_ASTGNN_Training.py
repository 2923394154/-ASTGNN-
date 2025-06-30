#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实数据训练ASTGNN模型
基于processed_astgnn_data.pt中的预处理数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
from matplotlib.font_manager import FontProperties

# GPU加速相关导入
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

# 导入ASTGNN相关模块
from ASTGNN import ASTGNNFactorModel
from ASTGNN_Loss import ASTGNNFactorLoss
from FactorValidation import FactorValidationFramework
from Enhanced_Factor_Analysis import ProfessionalBacktestAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Setup matplotlib for better plotting
def setup_matplotlib():
    """Setup matplotlib configuration"""
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    logger.info("Matplotlib configuration completed")

# Initialize matplotlib
setup_matplotlib()

class RealASTGNNTrainer:
    """使用真实数据的ASTGNN训练器"""
    
    def __init__(self, data_file: str = 'processed_astgnn_data.pt', config: Optional[Dict] = None):
        """初始化训练器"""
        self.data_file = data_file
        self.config = config or self._get_default_config()
        
        # GPU加速配置
        self._setup_gpu_acceleration()
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_count = torch.cuda.device_count()
        logger.info(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU数量: {self.gpu_count}")
            logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 混合精度训练
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("启用混合精度训练 (AMP)")
        
        # 加载数据
        self.load_processed_data()
        
        # 初始化模型组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.validator = None
        self.professional_analyzer = None
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = []
        self.val_r2_scores = []
        
        logger.info("真实数据ASTGNN训练器初始化完成")
    
    def _setup_gpu_acceleration(self):
        """配置GPU加速优化"""
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            cudnn.benchmark = True
            cudnn.deterministic = False  # 提高性能，降低可重复性
            
            # 预分配GPU内存
            torch.cuda.empty_cache()
            
            # 设置CUDA异步
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("GPU加速优化配置完成")
        else:
            logger.warning("CUDA不可用，将使用CPU训练")
    
    def _get_default_config(self) -> Dict:
        """获取默认训练配置"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 8,  # 增加批次大小以更好利用GPU
            'epochs': 100,
            'early_stopping_patience': 15,
            'gradient_clip_norm': 1.0,
            'orthogonal_penalty_weight': 0.01,
            'time_weight_decay': 0.9,
            'validation_frequency': 5,
            'save_best_model': True,
            'model_save_path': 'real_astgnn_best_model.pth',
            'plot_results': True,
            # GPU加速配置
            'use_amp': True,  # 启用混合精度训练
            'num_workers': 4,  # 数据加载器工作进程数
            'pin_memory': True,  # 锁页内存
            'compile_model': True,  # 启用模型编译优化
            'gradient_accumulation_steps': 1,  # 梯度累积步数
        }
    
    def load_processed_data(self):
        """加载预处理数据"""
        logger.info(f"加载预处理数据: {self.data_file}")
        
        try:
            data_dict = torch.load(self.data_file, map_location='cpu')
            
            # 提取数据
            self.data = data_dict['data']['sequences']
            self.metadata = data_dict['data']['metadata']
            self.preprocessing_config = data_dict['config']
            self.factor_scaler = data_dict['factor_scaler']
            self.return_scaler = data_dict['return_scaler']
            
            # 数据维度信息
            train_data = self.data['train']
            self.num_sequences = train_data['factor_sequences'].shape[0]
            self.sequence_length = train_data['factor_sequences'].shape[1]
            self.num_stocks = train_data['factor_sequences'].shape[2] 
            self.num_factors = train_data['factor_sequences'].shape[3]
            
            logger.info(f"数据加载成功:")
            logger.info(f"  训练序列数: {self.num_sequences}")
            logger.info(f"  序列长度: {self.sequence_length}")
            logger.info(f"  股票数量: {self.num_stocks}")
            logger.info(f"  因子数量: {self.num_factors}")
            logger.info(f"  因子名称: {self.metadata['factor_names']}")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器 - GPU优化版本"""
        logger.info("创建GPU优化数据加载器")
        
        # GPU加速数据加载配置
        dataloader_config = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config.get('num_workers', 4) if torch.cuda.is_available() else 0,
            'pin_memory': self.config.get('pin_memory', True) and torch.cuda.is_available(),
            'persistent_workers': True if torch.cuda.is_available() and self.config.get('num_workers', 4) > 0 else False,
        }
        
        logger.info(f"数据加载器配置: {dataloader_config}")
        
        # 对于GPU训练，数据先保持在CPU，通过pin_memory异步传输
        if torch.cuda.is_available():
            # 训练数据
            train_dataset = TensorDataset(
                self.data['train']['factor_sequences'],  # 保持在CPU
                self.data['train']['target_sequences']   # 保持在CPU
            )
            train_loader = DataLoader(
                train_dataset, 
                shuffle=True,
                **dataloader_config
            )
            
            # 验证数据
            val_dataset = TensorDataset(
                self.data['validation']['factor_sequences'],
                self.data['validation']['target_sequences']
            )
            val_loader = DataLoader(
                val_dataset, 
                shuffle=False,
                **dataloader_config
            )
            
            # 测试数据
            test_dataset = TensorDataset(
                self.data['test']['factor_sequences'],
                self.data['test']['target_sequences']
            )
            test_loader = DataLoader(
                test_dataset, 
                shuffle=False,
                **dataloader_config
            )
        else:
            # CPU训练模式（原有逻辑）
            train_dataset = TensorDataset(
                self.data['train']['factor_sequences'].to(self.device),
                self.data['train']['target_sequences'].to(self.device)
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True
            )
            
            val_dataset = TensorDataset(
                self.data['validation']['factor_sequences'].to(self.device),
                self.data['validation']['target_sequences'].to(self.device)
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False
            )
            
            test_dataset = TensorDataset(
                self.data['test']['factor_sequences'].to(self.device),
                self.data['test']['target_sequences'].to(self.device)
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False
            )
        
        logger.info(f"数据加载器创建完成:")
        logger.info(f"  训练批次: {len(train_loader)}")
        logger.info(f"  验证批次: {len(val_loader)}")
        logger.info(f"  测试批次: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self):
        """初始化模型和优化器"""
        logger.info("初始化ASTGNN模型")
        
        # 模型配置
        model_config = {
            'sequential_input_size': self.num_factors,  # Barra因子数量
            'gru_hidden_size': 64,
            'gru_num_layers': 2,
            'gat_hidden_size': 128,
            'gat_n_heads': 4,
            'res_hidden_size': 128,
            'num_risk_factors': 32,  # 生成的风险因子数量
            'tgc_hidden_size': 128,
            'tgc_output_size': 64,
            'num_tgc_layers': 2,
            'tgc_modes': ['add', 'subtract'],
            'prediction_hidden_sizes': [128, 64],
            'num_predictions': 1,  # 修改：输出单个因子
            'dropout': 0.1
        }
        
        # 创建模型
        self.model = ASTGNNFactorModel(**model_config).to(self.device)
        
        # 模型编译优化 (PyTorch 2.0+)
        if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                logger.info("模型编译优化启用")
            except Exception as e:
                logger.warning(f"模型编译失败，继续使用原始模型: {e}")
        
        # 多GPU支持
        if self.gpu_count > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"启用多GPU训练，GPU数量: {self.gpu_count}")
        
        # 改进权重初始化以增加预测方差
        self._initialize_model_weights()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8,  # 提高数值稳定性
            amsgrad=True  # 启用AMSGrad
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8
        )
        
        # 使用ASTGNN_Loss中的损失函数实现
        from ASTGNN_Loss import ASTGNNFactorLoss
        self.criterion = ASTGNNFactorLoss(
            omega=self.config.get('time_weight_decay', 0.9),
            lambda_orthogonal=self.config.get('orthogonal_penalty_weight', 0.01),
            max_periods=5,
            eps=1e-8,
            regularization_type='frobenius'
        )
        
        # 因子验证器
        self.validator = FactorValidationFramework()
        
        # 专业回测分析器
        self.professional_analyzer = ProfessionalBacktestAnalyzer(
            start_date='20231229',
            end_date='20240430',
            factor_names=['ASTGNN_Factor']  # 单因子名称
        )
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型初始化完成:")
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  模型配置: {model_config}")
    

    
    def _initialize_model_weights(self):
        """改进的权重初始化以增加预测方差"""
        logger.info("应用改进的权重初始化")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 对最后的预测层使用更大的初始化
                if 'prediction' in name.lower() or 'output' in name.lower():
                    # 增加最终预测层的权重方差
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0.0, std=0.05)
                    logger.info(f"  增强初始化预测层: {name}")
                else:
                    # 其他层使用标准Xavier初始化
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _compute_batch_loss(self, predictions, target_returns, batch_size):
        """计算批次损失"""
        batch_loss = 0.0
        for b in range(min(batch_size, 5)):
            F = predictions[b]  # [num_stocks, num_factors]
            
            # 构建未来多期收益率列表，每个元素形状为 [num_stocks]
            future_returns_list = []
            for t in range(target_returns.shape[1]):  # 遍历预测时间步
                future_returns_list.append(target_returns[b, t])  # [num_stocks]
            
            batch_loss += self.criterion(F, future_returns_list)
        
        return batch_loss / min(batch_size, 5)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch - GPU优化版本"""
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        num_batches = len(train_loader)
        
        # 梯度累积配置
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, (factor_sequences, target_returns) in enumerate(train_loader):
            # 异步数据传输到GPU
            factor_sequences = factor_sequences.to(self.device, non_blocking=True)
            target_returns = target_returns.to(self.device, non_blocking=True)
            
            # 创建邻接矩阵（简单的单位矩阵，实际应该基于因子相关性）
            batch_size = factor_sequences.shape[0]
            adj_matrix = torch.eye(self.num_stocks, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 使用混合精度训练
            if self.use_amp:
                with autocast():
                    # 前向传播
                    predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                    
                    # 计算损失
                    batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                    
                    # 梯度累积
                    batch_loss = batch_loss / accumulation_steps
            else:
                # 标准精度前向传播
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                
                # 计算损失
                batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                
                # 梯度累积
                batch_loss = batch_loss / accumulation_steps
            
            # 反向传播（支持混合精度和梯度累积）
            if self.use_amp:
                self.scaler.scale(batch_loss).backward()
                
                # 梯度累积控制
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                batch_loss.backward()
                
                # 梯度累积控制
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip_norm']
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 计算R²分数（使用单因子输出）
            with torch.no_grad():
                # 使用第一个时间步的收益率进行R²计算
                target_for_r2 = target_returns[:, 0, :]  # [batch_size, num_stocks]
                r2_score = self.calculate_r2_score(predictions.squeeze(-1), target_for_r2)  # squeeze单因子维度
            
            total_loss += (batch_loss * accumulation_steps).item()  # 还原真实损失
            total_r2 += r2_score
            
            # 打印进度
            if batch_idx % max(1, num_batches // 5) == 0:
                logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {(batch_loss * accumulation_steps).item():.6f}, R²: {r2_score:.6f}")
        
        avg_loss = total_loss / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_r2
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for factor_sequences, target_returns in val_loader:
                # 创建邻接矩阵
                batch_size = factor_sequences.shape[0]
                adj_matrix = torch.eye(self.num_stocks).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
                
                # 前向传播
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                
                # 处理目标张量维度
                # target_returns 形状: [batch_size, prediction_horizon, num_stocks]
                
                # 计算损失（类似训练时的处理）
                batch_loss = 0.0
                for b in range(min(batch_size, 5)):
                    F = predictions[b]  # [num_stocks, num_factors] 保持原来的维度
                    
                    # 构建未来多期收益率列表，每个元素形状为 [num_stocks]
                    future_returns_list = []
                    for t in range(target_returns.shape[1]):  # 遍历预测时间步
                        future_returns_list.append(target_returns[b, t])  # [num_stocks]
                    
                    batch_loss += self.criterion(F, future_returns_list)
                
                total_batch_loss = batch_loss / min(batch_size, 5)
                
                # 计算R²分数（使用第一个因子）
                # 使用第一个时间步的收益率进行R²计算
                target_for_r2 = target_returns[:, 0, :]  # [batch_size, num_stocks]
                r2_score = self.calculate_r2_score(predictions[:, :, 0], target_for_r2)
                
                total_loss += total_batch_loss.item()
                total_r2 += r2_score
        
        avg_loss = total_loss / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_r2
    
    def calculate_r2_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算R²分数"""
        # 展平张量 - 使用 reshape 避免连续性问题
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # 移除NaN值
        mask = ~(torch.isnan(pred_flat) | torch.isnan(target_flat))
        if mask.sum() == 0:
            return 0.0
        
        pred_clean = pred_flat[mask]
        target_clean = target_flat[mask]
        
        # 计算R²
        ss_res = torch.sum((target_clean - pred_clean) ** 2)
        ss_tot = torch.sum((target_clean - torch.mean(target_clean)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    
    def train_model(self):
        """完整的模型训练流程"""
        logger.info("开始ASTGNN模型训练")
        logger.info("=" * 80)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # 初始化模型
        self.initialize_model()
        
        # 训练参数
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_r2 = self.train_epoch(train_loader, epoch)
            
            # 验证
            if epoch % self.config['validation_frequency'] == 0:
                val_loss, val_r2 = self.validate_epoch(val_loader)
                
                # 记录历史
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_r2_scores.append(train_r2)
                self.val_r2_scores.append(val_r2)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                epoch_time = time.time() - epoch_start_time
                
                logger.info(f"Epoch {epoch}/{self.config['epochs']} "
                           f"[{epoch_time:.2f}s] - "
                           f"Train Loss: {train_loss:.6f}, Train R²: {train_r2:.6f}, "
                           f"Val Loss: {val_loss:.6f}, Val R²: {val_r2:.6f}")
                
                # 早停机制
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 保存最佳模型
                    if self.config['save_best_model']:
                        self.save_model(self.config['model_save_path'])
                        logger.info(f"保存最佳模型: {self.config['model_save_path']}")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"早停触发，在epoch {epoch}")
                    break
            else:
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch}/{self.config['epochs']} "
                           f"[{epoch_time:.2f}s] - "
                           f"Train Loss: {train_loss:.6f}, Train R²: {train_r2:.6f}")
        
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总耗时: {total_time:.2f}秒")
        
        # 最终测试
        self.final_evaluation(test_loader)
        
        # 绘制结果
        if self.config['plot_results']:
            self.plot_training_results()
    
    def final_evaluation(self, test_loader: DataLoader):
        """最终评估"""
        logger.info("进行最终测试评估")
        
        # 加载最佳模型
        if os.path.exists(self.config['model_save_path']):
            self.load_model(self.config['model_save_path'])
        
        test_loss, test_r2 = self.validate_epoch(test_loader)
        
        logger.info(f"最终测试结果:")
        logger.info(f"  测试损失: {test_loss:.6f}")
        logger.info(f"  测试R²: {test_r2:.6f}")
        
        # 因子分析
        self.analyze_factors(test_loader)
    
    def analyze_factors(self, test_loader: DataLoader):
        """分析生成的因子"""
        logger.info("分析生成因子的有效性")
        
        self.model.eval()
        all_factors = []  # 现在收集多个因子
        all_targets = []
        
        with torch.no_grad():
            for factor_sequences, target_returns in test_loader:
                factor_sequences = factor_sequences.to(self.device)
                batch_size = factor_sequences.size(0)
                
                # 创建邻接矩阵（单位矩阵）
                adj_matrix = torch.eye(self.num_stocks).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
                
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                
                # 现在predictions是 [batch, stocks, num_factors=8]
                all_factors.append(predictions.cpu())  # 保留所有因子维度
                all_targets.append(target_returns.cpu())
        
        # 合并所有因子和目标
        factors = torch.cat(all_factors, dim=0)  # [total_batches, stocks, num_factors]
        targets = torch.cat(all_targets, dim=0)
        
        # 处理目标张量形状：[batch, prediction_horizon, stocks] -> [batch, stocks]
        # 我们使用第一个预测期的收益率进行IC分析
        if targets.dim() == 3:
            targets = targets[:, 0, :]  # 选择第一个预测期 [batch, stocks]
        
        # 打印调试信息
        logger.info(f"因子形状: {factors.shape}")  # [time, stocks, factors]
        logger.info(f"目标形状: {targets.shape}")  # [time, stocks]
        
        # 转换为numpy
        factors_np = factors.numpy()
        targets_np = targets.numpy()
        
        # 详细诊断数据
        logger.info("=== 多因子有效性诊断 ===")
        logger.info(f"生成因子数量: {factors_np.shape[2]}")
        
        # 检查每个因子的统计信息
        for i in range(factors_np.shape[2]):
            factor_i = factors_np[:, :, i]
            logger.info(f"因子{i}统计:")
            logger.info(f"  均值: {factor_i.mean():.6f}, 标准差: {factor_i.std():.6f}")
            logger.info(f"  最小值: {factor_i.min():.6f}, 最大值: {factor_i.max():.6f}")
        
        # 检查目标值分布
        logger.info(f"目标收益率统计:")
        logger.info(f"  均值: {targets_np.mean():.6f}, 标准差: {targets_np.std():.6f}")
        logger.info(f"  最小值: {targets_np.min():.6f}, 最大值: {targets_np.max():.6f}")
        
        # 因子IC分析 - 现在是真正的多因子分析
        try:
            # factors: [time, stocks, num_factors], targets: [time, stocks]
            ic_results = self.validator.compute_information_coefficient(
                factors,    # [time, stocks, num_factors] 
                targets     # [time, stocks]
            )
            
            logger.info(f"因子IC分析结果:")
            factor_names = ['ASTGNN_Factor'] if len(ic_results['ic_mean']) == 1 else [f'Factor_{i}' for i in range(len(ic_results['ic_mean']))]
            
            for i, name in enumerate(factor_names):
                logger.info(f"{name:15}: IC均值={ic_results['ic_mean'][i]:7.4f}, "
                           f"IC标准差={ic_results['ic_std'][i]:7.4f}, "
                           f"IC_IR={ic_results['ic_ir'][i]:7.4f}, "
                           f"胜率={ic_results['ic_win_rate'][i]:7.4f}")
            
            # 总体因子表现
            avg_ic = np.mean(ic_results['ic_mean'])
            avg_ir = np.mean(ic_results['ic_ir'])
            logger.info(f"因子整体表现:")
            logger.info(f"  平均IC: {avg_ic:.6f}")
            logger.info(f"  平均IR: {avg_ir:.6f}")
            logger.info(f"  有效因子数量(|IC|>0.02): {sum(abs(ic) > 0.02 for ic in ic_results['ic_mean'])}")
                
        except Exception as e:
            logger.error(f"多因子IC计算失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        # 保存因子和目标数据用于进一步分析
        np.savez('factor_analysis_data.npz', 
                 factors=factors_np,  # 现在是多因子
                 targets=targets_np)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metadata': self.metadata,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_r2_scores': self.train_r2_scores,
            'val_r2_scores': self.val_r2_scores
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def plot_training_results(self):
        """Plot training results"""
        logger.info("Plotting training results")
        
        # Create training_results folder
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup plot style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ASTGNN Training Results', fontsize=16, fontweight='bold')
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', alpha=0.8, linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8, linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² score curves
        ax2.plot(epochs, self.train_r2_scores, 'b-', label='Train R²', alpha=0.8, linewidth=2)
        ax2.plot(epochs, self.val_r2_scores, 'r-', label='Validation R²', alpha=0.8, linewidth=2)
        ax2.set_title('R² Score Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Learning rate curve
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        ax3.plot(epochs, [lrs[0]] * len(epochs), 'g-', alpha=0.8, linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Training statistics
        stats_text = f"""Training Summary:
• Best Val Loss: {min(self.val_losses):.6f}
• Best Val R²: {max(self.val_r2_scores):.6f}
• Total Epochs: {len(self.train_losses)}
• Stocks: {self.num_stocks:,}
• Factors: {self.num_factors}
• Model Params: {sum(p.numel() for p in self.model.parameters()):,}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.7))
        ax4.set_title('Training Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save chart to training_results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'astgnn_training_results_{timestamp}.png'
        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Training results chart saved: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    # 改进的训练配置 - 解决预测方差过小问题
    config = {
        'learning_rate': 0.002,     # 提高学习率 (0.0005 -> 0.002)
        'weight_decay': 5e-5,       # 适度增加正则化
        'batch_size': 4,            # 增加批次大小 (2 -> 4)
        'epochs': 150,              # 增加训练轮数 (80 -> 150)
        'early_stopping_patience': 20,  # 增加早停容忍度 (12 -> 20)
        'gradient_clip_norm': 0.5,  # 减小梯度裁剪 (1.0 -> 0.5)
        'orthogonal_penalty_weight': 0.003,  # 减少正交惩罚 (0.008 -> 0.003)
        'time_weight_decay': 0.9,
        'validation_frequency': 5,   # 减少验证频率 (3 -> 5)
        'save_best_model': True,
        'model_save_path': 'real_astgnn_best_model.pth',
        'plot_results': True
    }
    
    # 创建训练器
    trainer = RealASTGNNTrainer(
        data_file='processed_astgnn_data.pt',
        config=config
    )
    
    # 开始训练
    trainer.train_model()
    
    # 训练完成后进行专业回测分析
    logger.info("=== 开始专业回测分析 ===")
    
    try:
        # 加载保存的因子分析数据
        import numpy as np
        data = np.load('factor_analysis_data.npz')
        factors = torch.from_numpy(data['factors'])  # [time, stocks, factors]
        targets = torch.from_numpy(data['targets'])  # [time, stocks]
        
        logger.info(f"专业回测数据: factors {factors.shape}, targets {targets.shape}")
        
        # 从目标收益率反推价格数据（用于专业回测）
        initial_price = 100.0
        prices = torch.zeros_like(targets)
        prices[0] = initial_price
        
        for t in range(1, len(targets)):
            prices[t] = prices[t-1] * (1 + targets[t-1])
        
        # 运行专业RankIC分析
        rank_ic_results = trainer.professional_analyzer.calculate_rank_ic_with_future_returns(
            factors, prices
        )
        
        # 运行专业分组回测（中证全指和沪深300）
        backtest_results = {}
        for universe in ['CSI_ALL', 'HS300']:
            backtest_results[universe] = trainer.professional_analyzer.group_backtest_analysis(
                factors, prices, 
                universe=universe, 
                factor_idx=0  # 测试第一个因子
            )
        
        logger.info("=== 专业回测分析完成 ===")
        
        # 打印总结报告
        logger.info("\n" + "="*60)
        logger.info("               专业回测分析报告")
        logger.info("="*60)
        
        # RankIC总结
        if 'results' in rank_ic_results:
            ic_results = rank_ic_results['results']
            if ic_results:
                avg_rank_ic = np.mean([abs(r['rank_ic_mean']) for r in ic_results.values()])
                avg_icir = np.mean([abs(r['rank_ic_ir']) for r in ic_results.values()])
                logger.info(f"\nRankIC分析结果:")
                logger.info(f"  平均RankIC: {avg_rank_ic:.6f}")
                logger.info(f"  平均ICIR: {avg_icir:.6f}")
                logger.info(f"  计算规范: 每隔10个交易日计算一次，使用未来10日收益率")
        
        # 分组回测总结
        for universe, bt_result in backtest_results.items():
            if 'long_short_stats' in bt_result and bt_result['long_short_stats']:
                ls_stats = bt_result['long_short_stats']
                logger.info(f"\n{bt_result['universe']}分组回测结果:")
                logger.info(f"  年化收益率: {ls_stats['annual_return']:8.2%}")
                logger.info(f"  信息比率: {ls_stats['information_ratio']:8.4f}")
                logger.info(f"  最大回撤: {ls_stats['max_drawdown']:8.2%}")
                logger.info(f"  胜率: {ls_stats['win_rate']:8.2%}")
                if 'turnover_rate' in bt_result:
                    turnover = bt_result['turnover_rate']
                    logger.info(f"  周均单边换手率: {turnover['weekly_turnover_pct']:.2f}%")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"专业回测分析失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("真实数据ASTGNN训练和专业回测分析完成!")


if __name__ == '__main__':
    main() 