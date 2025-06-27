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

# 导入ASTGNN相关模块
from ASTGNN import ASTGNNFactorModel
from ASTGNN_Loss import ASTGNNFactorLoss
from FactorValidation import FactorValidationFramework

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
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载数据
        self.load_processed_data()
        
        # 初始化模型组件
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.validator = None
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = []
        self.val_r2_scores = []
        
        logger.info("真实数据ASTGNN训练器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认训练配置"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 4,  # 较小批次以适应大数据
            'epochs': 100,
            'early_stopping_patience': 15,
            'gradient_clip_norm': 1.0,
            'orthogonal_penalty_weight': 0.01,
            'time_weight_decay': 0.9,
            'validation_frequency': 5,
            'save_best_model': True,
            'model_save_path': 'real_astgnn_best_model.pth',
            'plot_results': True
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
        """创建数据加载器"""
        logger.info("创建数据加载器")
        
        # 训练数据
        train_dataset = TensorDataset(
            self.data['train']['factor_sequences'].to(self.device),
            self.data['train']['target_sequences'].to(self.device)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        # 验证数据
        val_dataset = TensorDataset(
            self.data['validation']['factor_sequences'].to(self.device),
            self.data['validation']['target_sequences'].to(self.device)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # 测试数据
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
            'num_predictions': 8,  # 修改：输出8个风险因子而不是1个收益率
            'dropout': 0.1
        }
        
        # 创建模型
        self.model = ASTGNNFactorModel(**model_config).to(self.device)
        
        # 改进权重初始化以增加预测方差
        self._initialize_model_weights()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
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
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_r2 = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (factor_sequences, target_returns) in enumerate(train_loader):
            # 创建邻接矩阵（简单的单位矩阵，实际应该基于因子相关性）
            batch_size = factor_sequences.shape[0]
            adj_matrix = torch.eye(self.num_stocks).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            
            # 前向传播
            predictions, risk_factors, attention_weights, intermediate_outputs = self.model(factor_sequences, adj_matrix)
            
            # 现在predictions是多个因子 [batch, stocks, num_factors=8]
            # 我们用这些因子来预测收益率
            
            # 处理目标张量维度
            if target_returns.dim() == 3:
                target_returns = target_returns.squeeze(1)  # [batch, stocks]
            
            # 使用ASTGNN_Loss模块中的损失函数
            # 将每个批次的因子矩阵作为F [num_stocks, num_factors]
            batch_loss = 0.0
            for b in range(min(batch_size, 5)):  # 限制批次数以避免内存问题
                F = predictions[b]  # [num_stocks, num_factors] 保持原来的维度
                future_returns_list = [target_returns[b]]  # 当前批次的收益率
                batch_loss += self.criterion(F, future_returns_list)
            
            total_batch_loss = batch_loss / min(batch_size, 5)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
            
            self.optimizer.step()
            
            # 计算R²分数（使用因子的第一个维度作为代理）
            with torch.no_grad():
                r2_score = self.calculate_r2_score(predictions[:, :, 0], target_returns)
            
            total_loss += total_batch_loss.item()
            total_r2 += r2_score
            
            # 打印进度
            if batch_idx % max(1, num_batches // 5) == 0:
                logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {total_batch_loss.item():.6f}, R²: {r2_score:.6f}")
        
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
                if target_returns.dim() == 3:
                    target_returns = target_returns.squeeze(1)  # [batch, stocks]
                
                # 计算损失（类似训练时的处理）
                batch_loss = 0.0
                for b in range(min(batch_size, 5)):
                    F = predictions[b]  # [num_stocks, num_factors] 保持原来的维度
                    future_returns_list = [target_returns[b]]
                    batch_loss += self.criterion(F, future_returns_list)
                
                total_batch_loss = batch_loss / min(batch_size, 5)
                
                # 计算R²分数（使用第一个因子）
                r2_score = self.calculate_r2_score(predictions[:, :, 0], target_returns)
                
                total_loss += total_batch_loss.item()
                total_r2 += r2_score
        
        avg_loss = total_loss / num_batches
        avg_r2 = total_r2 / num_batches
        
        return avg_loss, avg_r2
    
    def calculate_r2_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算R²分数"""
        # 展平张量
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
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
        
        # 修复目标张量形状不匹配问题
        if targets.dim() == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # 移除多余维度 [batch, 1, stocks] -> [batch, stocks]
        
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
            
            logger.info(f"多因子IC分析结果:")
            factor_names = [f'Factor_{i}' for i in range(len(ic_results['ic_mean']))]
            
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
    
    logger.info("真实数据ASTGNN训练完成!")


if __name__ == '__main__':
    main() 