import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import time
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

# 导入模型和损失函数
from ASTGNN import ASTGNNFactorModel
from ASTGNN_Loss import ASTGNNFactorLoss, FactorLossWithAdjacencyRegularization
from FactorValidation import FactorValidationFramework
from BarraFactorSimulator import BarraFactorSimulator


class ASTGNNDataset(Dataset):
    """ASTGNN数据集类"""
    
    def __init__(self, factors, returns, adj_matrices, seq_len=10, prediction_horizon=5):
        """
        参数：
        - factors: [time, stocks, factors] 因子数据
        - returns: [time, stocks] 收益率数据
        - adj_matrices: [time, stocks, stocks] 邻接矩阵
        - seq_len: 序列长度
        - prediction_horizon: 预测时间范围
        """
        self.factors = factors
        self.returns = returns
        self.adj_matrices = adj_matrices
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        
        # 计算有效样本数量
        self.total_periods = factors.shape[0]
        self.valid_start = seq_len
        self.valid_end = self.total_periods - prediction_horizon
        self.num_samples = max(0, self.valid_end - self.valid_start)
        
        print(f"数据集创建: 总期数={self.total_periods}, 有效样本={self.num_samples}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 实际的时间索引
        time_idx = self.valid_start + idx
        
        # 输入序列：过去seq_len期的因子数据
        seq_start = time_idx - self.seq_len
        seq_end = time_idx
        input_sequence = self.factors[seq_start:seq_end]  # [seq_len, stocks, factors]
        
        # 目标：未来多期收益率
        target_returns = []
        for h in range(1, self.prediction_horizon + 1):
            if time_idx + h < self.total_periods:
                target_returns.append(self.returns[time_idx + h])
        
        # 邻接矩阵
        adj_matrix = self.adj_matrices[time_idx]
        
        return {
            'input_sequence': input_sequence,
            'target_returns': target_returns,
            'adj_matrix': adj_matrix,
            'time_idx': time_idx
        }


class ASTGNNTrainer:
    """ASTGNN训练器"""
    
    def __init__(self, 
                 model: ASTGNNFactorModel,
                 loss_fn: ASTGNNFactorLoss,
                 optimizer: optim.Optimizer,
                 device: str = 'cpu',
                 validation_framework: Optional[FactorValidationFramework] = None):
        """
        参数：
        - model: ASTGNN模型
        - loss_fn: 损失函数
        - optimizer: 优化器
        - device: 设备
        - validation_framework: 验证框架
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.validation_framework = validation_framework
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_r_square_loss = 0.0
        total_orthogonal_penalty = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 数据移到设备
            input_seq = batch['input_sequence'].to(self.device)  # [batch, seq_len, stocks, factors]
            target_returns = [ret.to(self.device) for ret in batch['target_returns']]
            adj_matrix = batch['adj_matrix'].to(self.device)  # [batch, stocks, stocks]
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 模型预测（这里我们需要获取风险因子矩阵M）
            predictions, risk_factors, attention_weights, intermediate_outputs = self.model(input_seq, adj_matrix[0])
            
            # 计算损失（使用风险因子矩阵M作为特征矩阵F）
            batch_size = risk_factors.shape[0]
            total_batch_loss = 0.0
            loss_details_sum = {'r_square_loss': 0.0, 'orthogonal_penalty': 0.0}
            
            for b in range(batch_size):
                # 每个样本的风险因子和目标收益
                F_b = risk_factors[b]  # [stocks, factors]
                target_returns_b = [ret[b] for ret in target_returns]  # 多期收益率
                
                # 计算损失
                loss_b, details_b = self.loss_fn(F_b, target_returns_b, return_individual_losses=True)
                total_batch_loss += loss_b
                
                # 累积损失详情
                loss_details_sum['r_square_loss'] += details_b['r_square_loss']
                loss_details_sum['orthogonal_penalty'] += details_b['orthogonal_penalty']
            
            # 平均损失
            avg_loss = total_batch_loss / batch_size
            
            # 反向传播
            avg_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            
            # 统计
            total_loss += avg_loss.item()
            total_r_square_loss += loss_details_sum['r_square_loss'] / batch_size
            total_orthogonal_penalty += loss_details_sum['orthogonal_penalty'] / batch_size
            num_batches += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {avg_loss.item():.4f}")
        
        # 返回平均损失
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'r_square_loss': total_r_square_loss / num_batches,
            'orthogonal_penalty': total_orthogonal_penalty / num_batches
        }
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """验证模型 - 修复IC计算的时间对齐问题"""
        if len(dataloader) == 0:
            return {
                'total_loss': float('inf'),
                'r_square_loss': float('inf'), 
                'orthogonal_penalty': float('inf')
            }
        
        self.model.eval()
        total_loss = 0.0
        total_r_square_loss = 0.0
        total_orthogonal_penalty = 0.0
        num_batches = 0
        
        # 收集因子和未来收益率用于评估 - 关键修复
        all_risk_factors = []
        all_future_returns = []  # 改名强调未来收益
        
        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch['input_sequence'].to(self.device)
                target_returns = [ret.to(self.device) for ret in batch['target_returns']]
                adj_matrix = batch['adj_matrix'].to(self.device)
                
                # 前向传播
                predictions, risk_factors, attention_weights, intermediate_outputs = self.model(input_seq, adj_matrix[0])
                
                # 计算损失
                batch_size = risk_factors.shape[0]
                total_batch_loss = 0.0
                loss_details_sum = {'r_square_loss': 0.0, 'orthogonal_penalty': 0.0}
                
                for b in range(batch_size):
                    F_b = risk_factors[b]
                    target_returns_b = [ret[b] for ret in target_returns]
                    
                    loss_b, details_b = self.loss_fn(F_b, target_returns_b, return_individual_losses=True)
                    total_batch_loss += loss_b
                    
                    loss_details_sum['r_square_loss'] += details_b['r_square_loss']
                    loss_details_sum['orthogonal_penalty'] += details_b['orthogonal_penalty']
                    
                    # 收集数据用于IC计算 - 关键修复：使用当期因子预测未来收益
                    all_risk_factors.append(F_b.cpu())
                    # 使用未来的收益率（target_returns是未来多期收益）
                    if len(target_returns_b) > 0:
                        all_future_returns.append(target_returns_b[0].cpu())  # 使用第一期未来收益
                
                avg_loss = total_batch_loss / batch_size
                total_loss += avg_loss.item()
                total_r_square_loss += loss_details_sum['r_square_loss'] / batch_size
                total_orthogonal_penalty += loss_details_sum['orthogonal_penalty'] / batch_size
                num_batches += 1
        
        if num_batches == 0:
            return {
                'total_loss': float('inf'),
                'r_square_loss': float('inf'),
                'orthogonal_penalty': float('inf')
            }
        
        # 计算验证指标
        val_metrics = {
            'total_loss': total_loss / num_batches,
            'r_square_loss': total_r_square_loss / num_batches,
            'orthogonal_penalty': total_orthogonal_penalty / num_batches
        }
        
        # IC计算 - 修复版本
        if self.validation_framework and len(all_risk_factors) > 0 and len(all_future_returns) > 0:
            try:
                factors_tensor = torch.stack(all_risk_factors)  # 当期因子
                returns_tensor = torch.stack(all_future_returns)  # 未来收益
                
                print(f"IC计算数据形状: factors={factors_tensor.shape}, future_returns={returns_tensor.shape}")
                
                # 计算预测性IC
                ic_results = self.validation_framework.compute_information_coefficient(
                    factors_tensor, returns_tensor
                )
                
                # 增强IC分析：检查因子的预测能力
                self._detailed_factor_analysis(factors_tensor, returns_tensor)
                
                val_metrics['ic_mean'] = np.mean(np.abs(ic_results['ic_mean']))
                val_metrics['ic_ir'] = np.mean(np.abs(ic_results['ic_ir'])) if not np.all(ic_results['ic_ir'] == 0) else 0.0
                val_metrics['ic_win_rate'] = np.mean(ic_results['ic_win_rate'])
                
            except Exception as e:
                print(f"因子评估计算失败: {e}")
        
        return val_metrics
    
    def _detailed_factor_analysis(self, factors_tensor, returns_tensor):
        """详细的因子分析 - 诊断IC低的原因"""
        print("\n=== 详细因子分析 ===")
        
        factors_np = factors_tensor.numpy()
        returns_np = returns_tensor.numpy()
        
        # 1. 因子分布分析
        print("因子分布统计:")
        for i in range(min(5, factors_np.shape[1])):  # 分析前5个因子
            factor_i = factors_np[:, :, i].flatten()
            print(f"  因子{i}: 均值={factor_i.mean():.4f}, 标准差={factor_i.std():.4f}, "
                  f"范围=[{factor_i.min():.4f}, {factor_i.max():.4f}]")
        
        # 2. 收益率分布分析
        returns_flat = returns_np.flatten()
        print(f"\n收益率分布: 均值={returns_flat.mean():.4f}, 标准差={returns_flat.std():.4f}, "
              f"范围=[{returns_flat.min():.4f}, {returns_flat.max():.4f}]")
        
        # 3. 截面相关性分析
        print("\n截面相关性分析:")
        sample_idx = min(factors_np.shape[0] - 1, 0)  # 取一个样本
        factors_cross = factors_np[sample_idx]  # [stocks, factors]
        returns_cross = returns_np[sample_idx]   # [stocks]
        
        for i in range(min(5, factors_cross.shape[1])):
            corr = np.corrcoef(factors_cross[:, i], returns_cross)[0, 1]
            print(f"  因子{i}与收益率相关性: {corr:.4f}")
        
        # 4. 因子单调性检验
        print("\n因子单调性检验:")
        self._factor_monotonicity_test(factors_cross, returns_cross)
    
    def _factor_monotonicity_test(self, factors, returns):
        """检验因子是否具有单调性预测能力"""
        num_quantiles = 5
        
        for i in range(min(3, factors.shape[1])):
            factor_i = factors[:, i]
            
            # 按因子值分组
            quantiles = np.quantile(factor_i, np.linspace(0, 1, num_quantiles + 1))
            group_returns = []
            
            for q in range(num_quantiles):
                mask = (factor_i >= quantiles[q]) & (factor_i <= quantiles[q + 1])
                if mask.sum() > 0:
                    group_ret = returns[mask].mean()
                    group_returns.append(group_ret)
                else:
                    group_returns.append(0)
            
            # 检查单调性
            is_monotonic = all(group_returns[i] <= group_returns[i+1] for i in range(len(group_returns)-1)) or \
                          all(group_returns[i] >= group_returns[i+1] for i in range(len(group_returns)-1))
            
            spread = group_returns[-1] - group_returns[0]  # 最高组与最低组的收益差
            
            print(f"  因子{i}: 单调性={'是' if is_monotonic else '否'}, "
                  f"收益差={spread:.4f}, 分组收益={[f'{r:.4f}' for r in group_returns]}")
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int,
              save_path: str = 'astgnn_best_model.pth',
              early_stopping_patience: int = 10) -> Dict:
        """完整训练流程"""
        print("开始训练ASTGNN模型...")
        print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['total_loss'])
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['total_loss'])
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"训练损失: {train_metrics['total_loss']:.4f} "
                  f"(R²: {train_metrics['r_square_loss']:.4f}, "
                  f"正交: {train_metrics['orthogonal_penalty']:.4f})")
            print(f"验证损失: {val_metrics['total_loss']:.4f}")
            
            if 'ic_mean' in val_metrics:
                print(f"验证IC: 均值={val_metrics['ic_mean']:.4f}, "
                      f"IR={val_metrics['ic_ir']:.4f}, "
                      f"胜率={val_metrics['ic_win_rate']:.4f}")
            
            # 早停和模型保存
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_metrics': self.val_metrics
                }, save_path)
                
                print(f"保存最佳模型到 {save_path}")
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"早停触发！验证损失连续{early_stopping_patience}轮未改善")
                break
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.1f}秒")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'total_time': total_time
        }
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """绘制训练历史"""
        # 确保字体设置
        setup_chinese_font()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='训练损失', color='blue')
        axes[0, 0].plot(self.val_losses, label='验证损失', color='red')
        axes[0, 0].set_title('损失函数')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IC均值
        if len(self.val_metrics) > 0 and 'ic_mean' in self.val_metrics[0]:
            ic_means = [m.get('ic_mean', 0) for m in self.val_metrics]
            axes[0, 1].plot(ic_means, color='green')
            axes[0, 1].set_title('IC均值（绝对值）')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('|IC|')
            axes[0, 1].grid(True)
        
        # IC信息比率
        if len(self.val_metrics) > 0 and 'ic_ir' in self.val_metrics[0]:
            ic_irs = [m.get('ic_ir', 0) for m in self.val_metrics]
            axes[1, 0].plot(ic_irs, color='orange')
            axes[1, 0].set_title('IC信息比率（绝对值）')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('|IC_IR|')
            axes[1, 0].grid(True)
        
        # IC胜率
        if len(self.val_metrics) > 0 and 'ic_win_rate' in self.val_metrics[0]:
            ic_win_rates = [m.get('ic_win_rate', 0) for m in self.val_metrics]
            axes[1, 1].plot(ic_win_rates, color='purple')
            axes[1, 1].set_title('IC胜率')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图保存到: {save_path}")


def create_data_loaders(factors, returns, adj_matrices, 
                       train_ratio=0.7, val_ratio=0.2, 
                       seq_len=10, prediction_horizon=5,
                       batch_size=4, shuffle=True):
    """创建数据加载器 - 修复版本"""
    total_periods = factors.shape[0]
    min_required_periods = seq_len + prediction_horizon
    
    print(f"数据集参数: 总期数={total_periods}, 序列长度={seq_len}, 预测期={prediction_horizon}")
    print(f"最小需求期数: {min_required_periods}")
    
    # 检查数据是否足够
    if total_periods < min_required_periods:
        raise ValueError(f"数据不足：需要至少{min_required_periods}期数据，但只有{total_periods}期")
    
    # 计算有效样本数（每个数据集可以生成的样本数）
    max_samples = total_periods - min_required_periods + 1
    print(f"最大可生成样本数: {max_samples}")
    
    if max_samples < 3:  # 至少需要3个样本才能分割
        # 动态调整参数
        new_seq_len = max(3, total_periods // 3)
        new_pred_horizon = max(1, (total_periods - new_seq_len) // 2)
        print(f"参数自动调整: seq_len={seq_len}->{new_seq_len}, prediction_horizon={prediction_horizon}->{new_pred_horizon}")
        seq_len = new_seq_len
        prediction_horizon = new_pred_horizon
        min_required_periods = seq_len + prediction_horizon
        max_samples = total_periods - min_required_periods + 1
    
    # 计算数据分割点（基于时间顺序）
    train_samples = max(1, int(max_samples * train_ratio))
    val_samples = max(1, int(max_samples * val_ratio))
    test_samples = max_samples - train_samples - val_samples
    
    print(f"样本分配: 训练={train_samples}, 验证={val_samples}, 测试={test_samples}")
    
    # 时间分割点
    train_end_period = seq_len + train_samples - 1
    val_end_period = train_end_period + val_samples
    
    print(f"时间分割点: 训练=[0, {train_end_period}], 验证=[{train_end_period-seq_len+1}, {val_end_period}], 测试=[{val_end_period-seq_len+1}, {total_periods}]")
    
    # 分割数据（允许重叠以保证序列连续性）
    train_factors = factors[:train_end_period + prediction_horizon]
    train_returns = returns[:train_end_period + prediction_horizon]
    train_adj = adj_matrices[:train_end_period + prediction_horizon]
    
    val_start = max(0, train_end_period - seq_len + 1)
    val_factors = factors[val_start:val_end_period + prediction_horizon]
    val_returns = returns[val_start:val_end_period + prediction_horizon]
    val_adj = adj_matrices[val_start:val_end_period + prediction_horizon]
    
    test_start = max(0, val_end_period - seq_len + 1)
    test_factors = factors[test_start:]
    test_returns = returns[test_start:]
    test_adj = adj_matrices[test_start:]
    
    # 创建数据集
    train_dataset = ASTGNNDataset(train_factors, train_returns, train_adj, 
                                 seq_len, prediction_horizon)
    val_dataset = ASTGNNDataset(val_factors, val_returns, val_adj, 
                               seq_len, prediction_horizon)
    test_dataset = ASTGNNDataset(test_factors, test_returns, test_adj, 
                                seq_len, prediction_horizon)
    
    print(f"实际数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    
    # 确保所有数据集都有样本
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空！")
    if len(val_dataset) == 0:
        # 如果验证集为空，从训练集分割一部分作为验证集
        print("验证数据集为空，从训练数据中分割...")
        val_factors = train_factors[-seq_len-prediction_horizon-1:]
        val_returns = train_returns[-seq_len-prediction_horizon-1:]
        val_adj = train_adj[-seq_len-prediction_horizon-1:]
        val_dataset = ASTGNNDataset(val_factors, val_returns, val_adj, seq_len, prediction_horizon)
        
        # 相应缩短训练集
        train_factors = train_factors[:-prediction_horizon-1]
        train_returns = train_returns[:-prediction_horizon-1]
        train_adj = train_adj[:-prediction_horizon-1]
        train_dataset = ASTGNNDataset(train_factors, train_returns, train_adj, seq_len, prediction_horizon)
        
        print(f"重新分割后: 训练={len(train_dataset)}, 验证={len(val_dataset)}")
    
    if len(test_dataset) == 0:
        print("警告：测试数据集为空")
        # 创建一个最小的测试集
        test_dataset = ASTGNNDataset(val_factors, val_returns, val_adj, seq_len, prediction_horizon)
    
    # 动态调整batch_size
    min_dataset_size = min(len(train_dataset), len(val_dataset))
    actual_batch_size = min(batch_size, min_dataset_size)
    if actual_batch_size < batch_size:
        print(f"批量大小调整: {batch_size} -> {actual_batch_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=actual_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main_training_pipeline():
    """主训练流程 - 数据期数优化版本"""
    print("=== ASTGNN训练流程 ===")
    
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 2. 智能数据期数设置
    def determine_optimal_periods():
        """根据资源和需求确定最优期数"""
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb > 8:
            return 150  # 高内存：更多期数
        elif available_memory_gb > 4:
            return 120  # 中等内存：中等期数
        else:
            return 60   # 低内存：较少期数
    
    # 3. 加载或生成数据
    try:
        print("加载训练数据...")
        training_data = torch.load('astgnn_training_data.pt', weights_only=False)
        
        factors = training_data['factors']
        returns = training_data['returns_standardized']
        adj_matrices = training_data['adjacency_matrices']
        factor_names = training_data['factor_names']
        
        print(f"数据加载成功: {factors.shape[0]}期, {factors.shape[1]}只股票, {factors.shape[2]}个因子")
        
        # 检查是否需要重新生成更多数据
        current_periods = factors.shape[0]
        optimal_periods = determine_optimal_periods()
        
        if current_periods < optimal_periods:
            print(f"当前数据期数({current_periods})少于最优期数({optimal_periods})，重新生成...")
            raise FileNotFoundError("需要重新生成数据")
            
    except FileNotFoundError:
        optimal_periods = determine_optimal_periods()
        print(f"生成新的训练数据({optimal_periods}期)...")
        
        # 使用优化的数据生成
        simulator = BarraFactorSimulator(num_stocks=200, seed=42)
        sim_data = simulator.simulate_factor_panel_data_enhanced(num_periods=optimal_periods)
        
        factors = sim_data['factors']
        returns = sim_data['returns_standardized']
        adj_matrices = sim_data['adjacency_matrices']
        factor_names = sim_data['factor_names']
        
        # 保存数据
        torch.save({
            'factors': factors,
            'returns_standardized': returns,
            'adjacency_matrices': adj_matrices,
            'factor_names': factor_names,
            'data_quality': sim_data['data_quality']
        }, 'astgnn_training_data.pt')
        
        print(f"模拟数据生成完成: {factors.shape[0]}期, {factors.shape[1]}只股票, {factors.shape[2]}个因子")
        print(f"数据质量: IC={sim_data['data_quality']['avg_ic']:.4f}")
    
    # 4. 根据数据量优化参数
    total_periods = factors.shape[0]
    
    if total_periods >= 120:
        seq_len = 15      # 更长序列学习复杂模式
        prediction_horizon = 5
        batch_size = 8    # 更大批量
        num_epochs = 50   # 更多轮次
    elif total_periods >= 60:
        seq_len = 12
        prediction_horizon = 3
        batch_size = 6
        num_epochs = 40
    else:
        seq_len = 8
        prediction_horizon = 2
        batch_size = 4
        num_epochs = 30
    
    print(f"优化参数: 序列长度={seq_len}, 预测期={prediction_horizon}, 批量={batch_size}, 轮次={num_epochs}")
    
    # 5. 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        factors, returns, adj_matrices,
        train_ratio=0.6, val_ratio=0.2,
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size, shuffle=True
    )
    
    # 6. 创建模型
    print("创建ASTGNN模型...")
    model = ASTGNNFactorModel(
        sequential_input_size=factors.shape[2],
        gru_hidden_size=64,
        gru_num_layers=2,
        gat_hidden_size=128,
        gat_n_heads=4,
        res_hidden_size=128,
        num_risk_factors=min(32, factors.shape[2]),  # 避免因子数超过输入维度
        
        tgc_hidden_size=128,
        tgc_output_size=64,
        num_tgc_layers=2,
        tgc_modes=['add', 'subtract'],
        
        prediction_hidden_sizes=[128, 64],
        num_predictions=1,
        dropout=0.1
    )
    
    print(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. 创建损失函数
    loss_fn = ASTGNNFactorLoss(
        omega=0.9,
        lambda_orthogonal=0.05,  # 降低正交惩罚
        max_periods=prediction_horizon,
        regularization_type='frobenius'
    )
    
    # 8. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 9. 创建验证框架
    validation_framework = FactorValidationFramework(factor_names=factor_names)
    
    # 10. 创建训练器
    trainer = ASTGNNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        validation_framework=validation_framework
    )
    
    # 11. 开始训练
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path='astgnn_best_model.pth',
        early_stopping_patience=10
    )
    
    # 12. 绘制训练历史
    trainer.plot_training_history('astgnn_training_history.png')
    
    # 13. 测试集评估
    print("\n=== 测试集评估 ===")
    test_metrics = trainer.validate(test_loader)
    print(f"测试损失: {test_metrics['total_loss']:.4f}")
    if 'ic_mean' in test_metrics:
        print(f"测试IC: 均值={test_metrics['ic_mean']:.4f}, "
              f"IR={test_metrics['ic_ir']:.4f}, "
              f"胜率={test_metrics['ic_win_rate']:.4f}")
    
    # 14. 基线分析
    print("\n=== 基线IC分析 ===")
    try:
        baseline_ic = validation_framework.compute_information_coefficient(factors[:-1], returns[1:])
        print(f"原始因子IC均值: {np.mean(np.abs(baseline_ic['ic_mean'])):.4f}")
        
        # 检查个别因子的预测能力
        print("\n原始因子预测能力分析:")
        for i in range(min(5, factors.shape[2])):
            corr = np.corrcoef(factors[:-1, :, i].flatten(), returns[1:].flatten())[0, 1]
            print(f"  {factor_names[i] if i < len(factor_names) else f'Factor_{i}'}: 相关性={corr:.4f}")
    except Exception as e:
        print(f"基线分析失败: {e}")
    
    print("\n训练完成！")
    
    return trainer, training_results


if __name__ == "__main__":
    trainer, results = main_training_pipeline() 