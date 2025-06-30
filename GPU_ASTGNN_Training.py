#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU优化版ASTGNN训练脚本
包含混合精度训练、多GPU支持、异步数据传输等GPU加速功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

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

class GPUOptimizedASTGNNTrainer:
    """GPU优化的ASTGNN训练器"""
    
    def __init__(self, data_file: str = 'processed_astgnn_data.pt', config: Optional[Dict] = None):
        """初始化GPU优化训练器"""
        self.data_file = data_file
        self.config = config or self._get_default_config()
        
        # GPU加速配置
        self._setup_gpu_acceleration()
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_count = torch.cuda.device_count()
        self._log_gpu_info()
        
        # 混合精度训练
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("✓ 启用混合精度训练 (AMP)")
        
        # 加载数据
        self.load_processed_data()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.validator = None
        self.professional_analyzer = None
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = []
        self.val_r2_scores = []
        
        logger.info("✓ GPU优化ASTGNN训练器初始化完成")
    
    def _setup_gpu_acceleration(self):
        """配置GPU加速优化"""
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            cudnn.benchmark = True
            cudnn.deterministic = False  # 提高性能，降低可重复性
            
            # 预分配GPU内存
            torch.cuda.empty_cache()
            
            # 设置CUDA异步和TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 设置内存管理
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%显存
            
            logger.info("✓ GPU加速优化配置完成")
        else:
            logger.warning("⚠ CUDA不可用，将使用CPU训练")
    
    def _log_gpu_info(self):
        """记录GPU信息"""
        logger.info(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU数量: {self.gpu_count}")
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  显存: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  计算能力: {props.major}.{props.minor}")
    
    def _get_default_config(self) -> Dict:
        """获取GPU优化的默认训练配置"""
        return {
            'learning_rate': 0.002,
            'weight_decay': 1e-5,
            'batch_size': 16,  # 增加批次大小以更好利用GPU
            'epochs': 100,
            'early_stopping_patience': 15,
            'gradient_clip_norm': 1.0,
            'orthogonal_penalty_weight': 0.01,
            'time_weight_decay': 0.9,
            'validation_frequency': 5,
            'save_best_model': True,
            'model_save_path': 'gpu_astgnn_best_model.pth',
            'plot_results': True,
            # GPU加速配置
            'use_amp': True,  # 启用混合精度训练
            'num_workers': 8,  # 数据加载器工作进程数
            'pin_memory': True,  # 锁页内存
            'compile_model': True,  # 启用模型编译优化
            'gradient_accumulation_steps': 2,  # 梯度累积步数
            'prefetch_factor': 4,  # 预取因子
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
            
            logger.info(f"✓ 数据加载成功:")
            logger.info(f"  训练序列数: {self.num_sequences}")
            logger.info(f"  序列长度: {self.sequence_length}")
            logger.info(f"  股票数量: {self.num_stocks}")
            logger.info(f"  因子数量: {self.num_factors}")
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            raise
    
    def create_gpu_optimized_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建GPU优化的数据加载器"""
        logger.info("创建GPU优化数据加载器")
        
        # GPU优化数据加载配置
        num_workers = self.config.get('num_workers', 8) if torch.cuda.is_available() else 0
        
        dataloader_config = {
            'batch_size': self.config['batch_size'],
            'num_workers': num_workers,
            'pin_memory': self.config.get('pin_memory', True) and torch.cuda.is_available(),
            'persistent_workers': True if num_workers > 0 else False,
            'drop_last': True,  # 确保批次大小一致
        }
        
        # 只有在使用多进程时才设置prefetch_factor
        if num_workers > 0:
            dataloader_config['prefetch_factor'] = self.config.get('prefetch_factor', 4)
        
        logger.info(f"数据加载器配置: {dataloader_config}")
        
        # 训练数据加载器
        train_dataset = TensorDataset(
            self.data['train']['factor_sequences'],
            self.data['train']['target_sequences']
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_config)
        
        # 验证数据加载器
        val_dataset = TensorDataset(
            self.data['validation']['factor_sequences'],
            self.data['validation']['target_sequences']
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_config)
        
        # 测试数据加载器
        test_dataset = TensorDataset(
            self.data['test']['factor_sequences'],
            self.data['test']['target_sequences']
        )
        test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_config)
        
        logger.info(f"✓ 数据加载器创建完成:")
        logger.info(f"  训练批次: {len(train_loader)}")
        logger.info(f"  验证批次: {len(val_loader)}")
        logger.info(f"  测试批次: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def initialize_gpu_optimized_model(self):
        """初始化GPU优化的模型和优化器"""
        logger.info("初始化GPU优化ASTGNN模型")
        
        # 模型配置 - 单因子输出
        model_config = {
            'sequential_input_size': self.num_factors,
            'gru_hidden_size': 64,
            'gru_num_layers': 2,
            'gat_hidden_size': 128,
            'gat_n_heads': 4,
            'res_hidden_size': 128,
            'num_risk_factors': 32,
            'tgc_hidden_size': 128,
            'tgc_output_size': 64,
            'num_tgc_layers': 2,
            'tgc_modes': ['add', 'subtract'],
            'prediction_hidden_sizes': [128, 64],
            'num_predictions': 1,  # 单因子输出
            'dropout': 0.1
        }
        
        # 创建模型
        self.model = ASTGNNFactorModel(**model_config).to(self.device)
        
        # 模型编译优化 (PyTorch 2.0+)
        if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                logger.info("✓ 模型编译优化启用")
            except Exception as e:
                logger.warning(f"⚠ 模型编译失败，继续使用原始模型: {e}")
        
        # 多GPU支持
        if self.gpu_count > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"✓ 启用多GPU训练，GPU数量: {self.gpu_count}")
        
        # 改进权重初始化
        self._initialize_model_weights()
        
        # 优化器 - GPU优化配置
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8,
            amsgrad=True
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'] * 3,
            epochs=self.config['epochs'],
            steps_per_epoch=self.num_sequences // self.config['batch_size'],
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # 损失函数
        self.criterion = ASTGNNFactorLoss(
            omega=self.config.get('time_weight_decay', 0.9),
            lambda_orthogonal=self.config.get('orthogonal_penalty_weight', 0.01),
            max_periods=5,
            eps=1e-8,
            regularization_type='frobenius'
        )
        
        # 其他组件
        self.validator = FactorValidationFramework()
        self.professional_analyzer = ProfessionalBacktestAnalyzer(
            start_date='20231229',
            end_date='20240430',
            factor_names=['ASTGNN_Factor']
        )
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✓ 模型初始化完成:")
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  模型配置: {model_config}")
    
    def _initialize_model_weights(self):
        """改进的权重初始化"""
        logger.info("应用改进的权重初始化")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if 'prediction' in name.lower() or 'output' in name.lower():
                    nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0.0, std=0.05)
                else:
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
        """计算批次损失 - 单因子版本"""
        batch_loss = 0.0
        
        for b in range(min(batch_size, 5)):  # 限制批次数以避免内存问题
            F = predictions[b]  # [num_stocks, 1] 单因子
            
            # 构建未来多期收益率列表
            future_returns_list = []
            for t in range(target_returns.shape[1]):
                future_returns_list.append(target_returns[b, t])
            
            batch_loss += self.criterion(F, future_returns_list)
        
        return batch_loss / min(batch_size, 5)
    
    def train_epoch_gpu_optimized(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """GPU优化的训练epoch"""
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        num_batches = len(train_loader)
        
        # 梯度累积配置
        accumulation_steps = self.config.get('gradient_accumulation_steps', 2)
        
        # 预热GPU
        if epoch == 1 and torch.cuda.is_available():
            logger.info("预热GPU...")
            torch.cuda.synchronize()
        
        for batch_idx, (factor_sequences, target_returns) in enumerate(train_loader):
            # 异步数据传输到GPU
            factor_sequences = factor_sequences.to(self.device, non_blocking=True)
            target_returns = target_returns.to(self.device, non_blocking=True)
            
            batch_size = factor_sequences.shape[0]
            
            # 创建邻接矩阵（直接在GPU上）
            adj_matrix = torch.eye(self.num_stocks, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 混合精度前向传播
            if self.use_amp:
                with autocast():
                    predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                    batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                    batch_loss = batch_loss / accumulation_steps
            else:
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                batch_loss = batch_loss / accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(batch_loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                batch_loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            
            # 计算R²分数
            with torch.no_grad():
                target_for_r2 = target_returns[:, 0, :]
                r2_score = self.calculate_r2_score(predictions.squeeze(-1), target_for_r2)
            
            total_loss += (batch_loss * accumulation_steps).item()
            total_r2 += r2_score
            
            # 打印进度
            if batch_idx % max(1, num_batches // 5) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {(batch_loss * accumulation_steps).item():.6f}, "
                           f"R²: {r2_score:.6f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_r2
    
    def validate_epoch_gpu_optimized(self, val_loader: DataLoader) -> Tuple[float, float]:
        """GPU优化的验证epoch"""
        self.model.eval()
        total_loss = 0.0
        total_r2 = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for factor_sequences, target_returns in val_loader:
                # 异步数据传输
                factor_sequences = factor_sequences.to(self.device, non_blocking=True)
                target_returns = target_returns.to(self.device, non_blocking=True)
                
                batch_size = factor_sequences.shape[0]
                adj_matrix = torch.eye(self.num_stocks, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # 混合精度前向传播
                if self.use_amp:
                    with autocast():
                        predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                        batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                else:
                    predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                    batch_loss = self._compute_batch_loss(predictions, target_returns, batch_size)
                
                # 计算R²分数
                target_for_r2 = target_returns[:, 0, :]
                r2_score = self.calculate_r2_score(predictions.squeeze(-1), target_for_r2)
                
                total_loss += batch_loss.item()
                total_r2 += r2_score
        
        avg_loss = total_loss / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_r2
    
    def calculate_r2_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算R²分数"""
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        mask = ~(torch.isnan(pred_flat) | torch.isnan(target_flat))
        if mask.sum() == 0:
            return 0.0
        
        pred_clean = pred_flat[mask]
        target_clean = target_flat[mask]
        
        ss_res = torch.sum((target_clean - pred_clean) ** 2)
        ss_tot = torch.sum((target_clean - torch.mean(target_clean)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    
    def train_model_gpu_optimized(self):
        """GPU优化的完整训练流程"""
        logger.info("开始GPU优化ASTGNN模型训练")
        logger.info("=" * 80)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_gpu_optimized_data_loaders()
        
        # 初始化模型
        self.initialize_gpu_optimized_model()
        
        # 训练参数
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # GPU预热
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info("✓ GPU预热完成")
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_r2 = self.train_epoch_gpu_optimized(train_loader, epoch)
            
            # 验证
            if epoch % self.config['validation_frequency'] == 0:
                val_loss, val_r2 = self.validate_epoch_gpu_optimized(val_loader)
                
                # 记录历史
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_r2_scores.append(train_r2)
                self.val_r2_scores.append(val_r2)
                
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                logger.info(f"Epoch {epoch}/{self.config['epochs']} "
                           f"[{epoch_time:.2f}s] - "
                           f"Train Loss: {train_loss:.6f}, Train R²: {train_r2:.6f}, "
                           f"Val Loss: {val_loss:.6f}, Val R²: {val_r2:.6f}, "
                           f"LR: {current_lr:.2e}")
                
                # 早停机制
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if self.config['save_best_model']:
                        self.save_model(self.config['model_save_path'])
                        logger.info(f"✓ 保存最佳模型: {self.config['model_save_path']}")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"早停触发，在epoch {epoch}")
                    break
            else:
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{self.config['epochs']} "
                           f"[{epoch_time:.2f}s] - "
                           f"Train Loss: {train_loss:.6f}, Train R²: {train_r2:.6f}, "
                           f"LR: {current_lr:.2e}")
        
        total_time = time.time() - start_time
        logger.info(f"✓ 训练完成! 总耗时: {total_time:.2f}秒")
        
        # GPU最终同步
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 最终测试
        self.final_evaluation(test_loader)
        
        # 生成因子分析数据
        self.generate_factor_analysis_data(test_loader)
        
        # 绘制结果
        if self.config['plot_results']:
            self.plot_training_results()
    
    def final_evaluation(self, test_loader: DataLoader):
        """最终评估"""
        logger.info("进行最终测试评估")
        
        if os.path.exists(self.config['model_save_path']):
            self.load_model(self.config['model_save_path'])
        
        test_loss, test_r2 = self.validate_epoch_gpu_optimized(test_loader)
        
        logger.info(f"✓ 最终测试结果:")
        logger.info(f"  测试损失: {test_loss:.6f}")
        logger.info(f"  测试R²: {test_r2:.6f}")
    
    def generate_factor_analysis_data(self, test_loader: DataLoader):
        """生成因子分析数据用于后续评价"""
        logger.info("生成因子分析数据...")
        
        # 确保使用最佳模型
        if os.path.exists(self.config['model_save_path']):
            self.load_model(self.config['model_save_path'])
        
        self.model.eval()
        all_factors = []
        all_targets = []
        
        with torch.no_grad():
            for factor_sequences, target_returns in test_loader:
                # GPU优化数据传输
                factor_sequences = factor_sequences.to(self.device, non_blocking=True)
                target_returns = target_returns.to(self.device, non_blocking=True)
                
                # 创建邻接矩阵
                batch_size = factor_sequences.shape[0]
                adj_matrix = torch.eye(self.num_stocks, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # 前向传播获取因子预测
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
                
                # 收集因子和目标数据
                # predictions形状: [batch_size, num_stocks, 1] (单因子)
                # target_returns形状: [batch_size, prediction_horizon, num_stocks]
                
                # 转换为 [time, stocks, factors] 格式
                for b in range(batch_size):
                    # 因子数据: [num_stocks, 1] -> [1, num_stocks, 1]
                    factors_b = predictions[b].unsqueeze(0)  # [1, num_stocks, 1]
                    all_factors.append(factors_b.cpu())
                    
                    # 目标数据: [prediction_horizon, num_stocks] -> [prediction_horizon, num_stocks]
                    targets_b = target_returns[b].permute(0, 1)  # [prediction_horizon, num_stocks]
                    all_targets.append(targets_b.cpu())
        
        # 合并所有批次数据
        try:
            # 拼接因子数据: list of [1, num_stocks, 1] -> [total_time, num_stocks, 1]
            factors_tensor = torch.cat(all_factors, dim=0)
            
            # 拼接目标数据: list of [prediction_horizon, num_stocks] -> [total_batches*prediction_horizon, num_stocks]
            targets_tensor = torch.cat(all_targets, dim=0)
            
            # 调整目标数据维度以匹配因子数据的时间维度
            time_steps = factors_tensor.shape[0]
            targets_tensor = targets_tensor[:time_steps]  # 截取匹配的时间步数
            
            # 转换为numpy
            factors_np = factors_tensor.numpy()  # [time, stocks, factors]
            targets_np = targets_tensor.numpy()  # [time, stocks]
            
            logger.info(f"因子数据生成完成:")
            logger.info(f"  因子形状: {factors_np.shape}")
            logger.info(f"  目标形状: {targets_np.shape}")
            logger.info(f"  时间步数: {factors_np.shape[0]}")
            logger.info(f"  股票数量: {factors_np.shape[1]}")
            logger.info(f"  因子数量: {factors_np.shape[2]} (单因子)")
            
            # 保存因子和目标数据用于进一步分析
            np.savez('factor_analysis_data.npz', 
                     factors=factors_np,  # [time, stocks, 1] 单因子
                     targets=targets_np)  # [time, stocks]
            
            logger.info("✓ 因子分析数据已保存到 factor_analysis_data.npz")
            
        except Exception as e:
            logger.error(f"因子数据生成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
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
        if checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def plot_training_results(self):
        """绘制训练结果"""
        logger.info("绘制训练结果图表")
        
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU优化ASTGNN训练结果', fontsize=16, fontweight='bold')
        
        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失', alpha=0.8, linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失', alpha=0.8, linewidth=2)
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R²分数曲线
        ax2.plot(epochs, self.train_r2_scores, 'b-', label='训练R²', alpha=0.8, linewidth=2)
        ax2.plot(epochs, self.val_r2_scores, 'r-', label='验证R²', alpha=0.8, linewidth=2)
        ax2.set_title('R²分数曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²分数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # GPU利用率信息
        gpu_info = f"""GPU配置信息:
• 设备: {self.device}
• GPU数量: {self.gpu_count}
• 混合精度: {'启用' if self.use_amp else '禁用'}
• 批次大小: {self.config['batch_size']}
• 工作进程: {self.config.get('num_workers', 0)}
• 编译优化: {'启用' if self.config.get('compile_model') else '禁用'}
"""
        
        ax3.text(0.05, 0.95, gpu_info, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgreen", alpha=0.7))
        ax3.set_title('GPU配置信息')
        ax3.axis('off')
        
        # 训练统计
        stats_text = f"""训练摘要:
• 最佳验证损失: {min(self.val_losses):.6f}
• 最佳验证R²: {max(self.val_r2_scores):.6f}
• 总轮数: {len(self.train_losses)}
• 股票数: {self.num_stocks:,}
• 因子数: {self.num_factors}
• 模型参数: {sum(p.numel() for p in self.model.parameters()):,}
• 单因子输出: ✓
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.7))
        ax4.set_title('训练统计')
        ax4.axis('off')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'gpu_astgnn_training_results_{timestamp}.png'
        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"✓ 训练结果图表已保存: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    logger.info("启动GPU优化ASTGNN训练")
    
    # GPU优化训练配置
    config = {
        'learning_rate': 0.003,
        'weight_decay': 1e-5,
        'batch_size': 16,  # 根据GPU显存调整
        'epochs': 120,
        'early_stopping_patience': 20,
        'gradient_clip_norm': 1.0,
        'orthogonal_penalty_weight': 0.005,
        'time_weight_decay': 0.9,
        'validation_frequency': 5,
        'save_best_model': True,
        'model_save_path': 'gpu_astgnn_best_model.pth',
        'plot_results': True,
        # GPU优化配置
        'use_amp': True,
        'num_workers': 8,
        'pin_memory': True,
        'compile_model': True,
        'gradient_accumulation_steps': 2,
        'prefetch_factor': 4,
    }
    
    # 创建GPU优化训练器
    trainer = GPUOptimizedASTGNNTrainer(
        data_file='processed_astgnn_data.pt',
        config=config
    )
    
    # 开始GPU优化训练
    trainer.train_model_gpu_optimized()
    
    logger.info("✓ GPU优化训练完成")


if __name__ == "__main__":
    main() 