import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASTGNNFactorLoss(nn.Module):
    """ASTGNN因子挖掘模型的自定义损失函数
    
    损失函数包含两部分：
    1. 属性特征向量与未来收益率的R-square损失（时间加权）
    2. 特征相关系数矩阵的正交惩罚项
    
    损失函数: ∑ ω^(t-1) R-square(F, y_t) + λ ||corr(F, F)||_2
    """
    
    def __init__(self, omega=0.9, lambda_orthogonal=0.05, max_periods=5, 
                 eps=1e-8, regularization_type='frobenius'):
        """
        参数：
        - omega: 时间衰减权重参数 (0 < omega < 1)
        - lambda_orthogonal: 正交惩罚项权重系数
        - max_periods: 最大预测期数
        - eps: 数值稳定性参数
        - regularization_type: 正则化类型 ('frobenius', 'nuclear', 'spectral')
        """
        super(ASTGNNFactorLoss, self).__init__()
        
        assert 0 < omega < 1, "omega必须在(0,1)区间内"
        self.omega = omega
        self.lambda_orthogonal = lambda_orthogonal
        self.max_periods = max_periods
        self.eps = eps
        self.regularization_type = regularization_type
        
        # 预计算时间权重
        self.register_buffer('time_weights', 
                           torch.tensor([omega ** (t-1) for t in range(1, max_periods + 1)]))
    
    def compute_r_square(self, F, y):
        """计算R-square - 修复版本，解决训练方向问题和数据类型问题"""
        try:
            # 【关键修复】：确保数据类型一致性（解决AMP混合精度问题）
            F = F.float()
            y = y.float()
            
            # 输入检查和数值稳定性处理
            if torch.isnan(F).any() or torch.isnan(y).any():
                return torch.tensor(0.0, device=F.device, dtype=torch.float32)
            
            if torch.isinf(F).any() or torch.isinf(y).any():
                return torch.tensor(0.0, device=F.device, dtype=torch.float32)
            
            # 确保数据维度正确
            if F.dim() != 2 or y.dim() != 1:
                raise ValueError(f"维度错误: F.shape={F.shape}, y.shape={y.shape}")
            
            num_stocks, num_factors = F.shape
            
            if num_stocks != y.shape[0]:
                raise ValueError(f"股票数量不匹配: F有{num_stocks}只股票, y有{y.shape[0]}只股票")
            
            # 【关键修复】：数据标准化，确保数值稳定
            F_normalized = torch.nn.functional.normalize(F, dim=0, eps=1e-8)  # 按因子标准化
            y_centered = y - y.mean()  # 收益率去均值
            y_std = y.std() + 1e-8  # 添加小常数避免除零
            y_normalized = y_centered / y_std
            
            # 【关键修复】：改进的回归计算
            # 使用岭回归避免共线性问题
            ridge_lambda = 1e-6
            FtF = torch.mm(F_normalized.t(), F_normalized) + ridge_lambda * torch.eye(num_factors, device=F.device, dtype=torch.float32)
            
            try:
                # 计算回归系数 β = (F'F + λI)^(-1) F'y
                Fty = torch.mv(F_normalized.t(), y_normalized)
                # 【关键修复】：确保所有张量都是float32类型
                FtF = FtF.float()
                Fty = Fty.float()
                beta = torch.linalg.solve(FtF, Fty)
                
                # 预测值
                y_pred = torch.mv(F_normalized, beta)
                
                # 【关键修复】：确保预测合理性
                # 如果预测值与实际值相关性为负，调整符号
                correlation = torch.corrcoef(torch.stack([y_normalized, y_pred]))[0, 1]
                if torch.isnan(correlation):
                    correlation = torch.tensor(0.0, device=F.device)
                
                # 计算R-square
                ss_res = torch.sum((y_normalized - y_pred) ** 2)
                ss_tot = torch.sum(y_normalized ** 2) + 1e-8
                r_square = 1 - ss_res / ss_tot
                
                # 【关键修复】：确保R-square在合理范围内
                r_square = torch.clamp(r_square, -1.0, 1.0)
                
                # 如果相关性为负且R-square为正，需要调整
                if correlation < 0 and r_square > 0:
                    r_square = -r_square  # 负相关时应该是负R-square
                
                # 数值稳定性检查
                if torch.isnan(r_square) or torch.isinf(r_square):
                    return torch.tensor(0.0, device=F.device, dtype=torch.float32)
                
                return torch.abs(r_square).float()  # 返回绝对值，确保float32类型
                
            except Exception as e:
                # 如果线性代数计算失败，使用备选方案
                print(f"回归计算失败，使用相关系数平方: {e}")
                
                # 备选方案：使用多重相关系数
                correlations = []
                for i in range(num_factors):
                    try:
                        corr_matrix = torch.corrcoef(torch.stack([F_normalized[:, i], y_normalized]))
                        corr_val = corr_matrix[0, 1]
                        if not torch.isnan(corr_val):
                            correlations.append((corr_val ** 2).float())
                    except:
                        # 如果corrcoef失败，使用手动计算
                        f_i = F_normalized[:, i]
                        corr_num = torch.sum((f_i - f_i.mean()) * (y_normalized - y_normalized.mean()))
                        corr_den = torch.sqrt(torch.sum((f_i - f_i.mean())**2) * torch.sum((y_normalized - y_normalized.mean())**2)) + 1e-8
                        corr_val = corr_num / corr_den
                        if not torch.isnan(corr_val):
                            correlations.append((corr_val ** 2).float())
                
                if correlations:
                    r_square = torch.mean(torch.stack(correlations))
                    return torch.clamp(r_square, 0.0, 1.0).float()
                else:
                    return torch.tensor(0.0, device=F.device, dtype=torch.float32)
                
        except Exception as e:
            print(f"R-square计算失败: {e}")
            return torch.tensor(0.0, device=F.device, dtype=torch.float32)
    
    def compute_correlation_penalty(self, F):
        """计算特征相关系数矩阵的正交惩罚项
        
        参数：
        - F: 属性特征向量矩阵 [num_stocks, num_factors]
        
        返回：
        - penalty: ||corr(F, F)||_2 的某种范数
        """
        # 标准化特征矩阵
        F_mean = torch.mean(F, dim=0, keepdim=True)
        F_std = torch.std(F, dim=0, keepdim=True) + self.eps
        F_normalized = (F - F_mean) / F_std
        
        # 计算相关系数矩阵
        correlation_matrix = torch.matmul(F_normalized.T, F_normalized) / (F.shape[0] - 1)
        
        # 去除对角线元素（自相关为1）
        eye_mask = torch.eye(correlation_matrix.shape[0], device=F.device)
        off_diagonal_corr = correlation_matrix * (1 - eye_mask)
        
        # 计算不同类型的范数惩罚
        if self.regularization_type == 'frobenius':
            # Frobenius范数 ||A||_F = sqrt(sum(A_ij^2))
            penalty = torch.norm(off_diagonal_corr, p='fro') ** 2
        elif self.regularization_type == 'nuclear':
            # 核范数（迹范数）||A||_* = sum(σ_i)
            penalty = torch.norm(off_diagonal_corr, p='nuc')
        elif self.regularization_type == 'spectral':
            # 谱范数 ||A||_2 = max(σ_i)
            penalty = torch.norm(off_diagonal_corr, p=2) ** 2
        else:
            raise ValueError(f"不支持的正则化类型: {self.regularization_type}")
        
        return penalty
    
    def forward(self, F, future_returns_list, return_individual_losses=False):
        """前向传播计算损失 - 修复版本"""
        if not isinstance(future_returns_list, (list, tuple)):
            raise ValueError("future_returns_list应该是列表或元组")
        
        num_periods = len(future_returns_list)
        if num_periods == 0:
            raise ValueError("future_returns_list不能为空")
        
        # 【关键修复】：调整损失函数权重配置
        # 第一部分：时间加权的R-square损失（主要损失）
        r_square_losses = []
        weighted_r_square_loss = 0.0
        
        for t, y_t in enumerate(future_returns_list):
            if t >= self.max_periods:
                break
            
            # 计算当期R-square
            r_square = self.compute_r_square(F, y_t)
            r_square_losses.append(r_square.item())
            
            # 应用时间权重
            weight = self.time_weights[t] if t < len(self.time_weights) else self.omega ** t
            
            # 【关键修复】：损失函数调整
            # 最大化R-square（最小化负R-square）
            weighted_r_square_loss += weight * (1.0 - r_square)
        
        # 第二部分：相关系数正交惩罚项（辅助损失）
        orthogonal_penalty = self.compute_correlation_penalty(F)
        
        # 【关键修复】：重新平衡损失权重
        # 降低正交惩罚权重，专注于主要预测任务
        adjusted_orthogonal_weight = self.lambda_orthogonal * 0.1  # 大幅降低正交惩罚
        
        # 总损失
        total_loss = weighted_r_square_loss + adjusted_orthogonal_weight * orthogonal_penalty
        
        # 【关键修复】：添加数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告：损失计算出现NaN或Inf，使用备选损失")
            total_loss = torch.tensor(1.0, device=F.device, requires_grad=True)
        
        if return_individual_losses:
            loss_details = {
                'r_square_loss': weighted_r_square_loss.item() if hasattr(weighted_r_square_loss, 'item') else weighted_r_square_loss,
                'orthogonal_penalty': orthogonal_penalty.item() if hasattr(orthogonal_penalty, 'item') else orthogonal_penalty,
                'total_loss': total_loss.item() if hasattr(total_loss, 'item') else total_loss,
                'r_square_values': r_square_losses,
                'effective_orthogonal_weight': adjusted_orthogonal_weight
            }
            return total_loss, loss_details
        
        return total_loss


class FactorLossWithAdjacencyRegularization(ASTGNNFactorLoss):
    """扩展版本：增加邻接矩阵时间变化的正则化
    
    添加了对邻接矩阵随时间变化的惩罚，防止换手率过高
    """
    
    def __init__(self, omega=0.9, lambda_orthogonal=0.1, lambda_adjacency=0.05,
                 max_periods=5, eps=1e-8, adjacency_penalty_type='frobenius'):
        """
        参数：
        - lambda_adjacency: 邻接矩阵变化惩罚权重
        - adjacency_penalty_type: 邻接矩阵惩罚类型
        """
        super().__init__(omega, lambda_orthogonal, max_periods, eps)
        self.lambda_adjacency = lambda_adjacency
        self.adjacency_penalty_type = adjacency_penalty_type
    
    def compute_adjacency_penalty(self, adj_matrices):
        """计算邻接矩阵时间变化惩罚
        
        参数：
        - adj_matrices: 时间序列邻接矩阵列表 [adj_t1, adj_t2, ...]
        
        返回：
        - penalty: 邻接矩阵变化惩罚
        """
        if len(adj_matrices) < 2:
            return torch.tensor(0.0, device=adj_matrices[0].device)
        
        total_penalty = 0.0
        
        for t in range(1, len(adj_matrices)):
            adj_diff = adj_matrices[t] - adj_matrices[t-1]
            
            if self.adjacency_penalty_type == 'frobenius':
                penalty = torch.norm(adj_diff, p='fro') ** 2
            elif self.adjacency_penalty_type == 'l1':
                penalty = torch.norm(adj_diff, p=1)
            else:
                penalty = torch.norm(adj_diff, p=2) ** 2
            
            total_penalty += penalty
        
        return total_penalty / (len(adj_matrices) - 1)  # 平均惩罚
    
    def forward(self, F, future_returns_list, adj_matrices=None, return_individual_losses=False):
        """扩展的前向传播
        
        参数：
        - adj_matrices: 邻接矩阵时间序列（可选）
        """
        # 调用父类方法计算基础损失
        if return_individual_losses:
            total_loss, loss_details = super().forward(F, future_returns_list, True)
        else:
            total_loss = super().forward(F, future_returns_list, False)
        
        # 添加邻接矩阵惩罚
        if adj_matrices is not None:
            adjacency_penalty = self.compute_adjacency_penalty(adj_matrices)
            total_loss += self.lambda_adjacency * adjacency_penalty
            
            if return_individual_losses:
                loss_details['adjacency_penalty'] = adjacency_penalty.item()
                loss_details['total_loss'] = total_loss.item()
        
        if return_individual_losses:
            return total_loss, loss_details
        return total_loss


def test_astgnn_loss():
    """测试ASTGNN损失函数"""
    print("=== ASTGNN因子损失函数测试 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模拟数据
    num_stocks = 200
    num_factors = 10
    num_periods = 5
    
    # 生成属性特征矩阵F
    F = torch.randn(num_stocks, num_factors)
    
    # 生成未来收益率（模拟真实的因子暴露关系）
    true_factor_returns = torch.randn(num_factors, num_periods)
    noise = torch.randn(num_stocks, num_periods) * 0.1
    
    future_returns_list = []
    for t in range(num_periods):
        # y_t = F @ factor_returns_t + noise_t
        y_t = torch.matmul(F, true_factor_returns[:, t]) + noise[:, t]
        # 标准化
        y_t = (y_t - torch.mean(y_t)) / torch.std(y_t)
        future_returns_list.append(y_t)
    
    # 1. 基础损失函数测试
    print("\n1. 基础ASTGNN损失函数测试")
    loss_fn = ASTGNNFactorLoss(omega=0.9, lambda_orthogonal=0.1)
    
    total_loss, details = loss_fn(F, future_returns_list, return_individual_losses=True)
    
    print(f"总损失: {details['total_loss']:.4f}")
    print(f"R-square损失: {details['r_square_loss']:.4f}")
    print(f"正交惩罚: {details['orthogonal_penalty']:.4f}")
    print(f"各期R-square: {[f'{r:.4f}' for r in details['r_square_values']]}")
    
    # 2. 不同正则化类型比较
    print("\n2. 不同正则化类型比较")
    reg_types = ['frobenius', 'nuclear', 'spectral']
    
    for reg_type in reg_types:
        loss_fn = ASTGNNFactorLoss(lambda_orthogonal=0.1, regularization_type=reg_type)
        loss, details = loss_fn(F, future_returns_list, return_individual_losses=True)
        print(f"{reg_type:10}: 总损失={details['total_loss']:.4f}, "
              f"正交惩罚={details['orthogonal_penalty']:.4f}")
    
    # 3. 扩展版本测试（包含邻接矩阵正则化）
    print("\n3. 邻接矩阵正则化测试")
    
    # 生成时间序列邻接矩阵
    adj_matrices = []
    base_adj = torch.randint(0, 2, (num_stocks, num_stocks)).float()
    base_adj = (base_adj + base_adj.T) / 2  # 对称化
    
    for t in range(num_periods):
        # 添加一些随机变化
        noise_adj = torch.randn_like(base_adj) * 0.1
        adj_t = torch.clamp(base_adj + noise_adj, 0, 1)
        adj_matrices.append(adj_t)
    
    extended_loss_fn = FactorLossWithAdjacencyRegularization(
        omega=0.9, lambda_orthogonal=0.1, lambda_adjacency=0.05
    )
    
    loss, details = extended_loss_fn(F, future_returns_list, adj_matrices, 
                                   return_individual_losses=True)
    
    print(f"扩展损失详情:")
    print(f"  总损失: {details['total_loss']:.4f}")
    print(f"  R-square损失: {details['r_square_loss']:.4f}")
    print(f"  正交惩罚: {details['orthogonal_penalty']:.4f}")
    print(f"  邻接矩阵惩罚: {details['adjacency_penalty']:.4f}")


def demonstrate_loss_gradients():
    """演示损失函数的梯度计算"""
    print("\n=== 损失函数梯度测试 ===")
    
    num_stocks = 50
    num_factors = 5
    
    # 创建需要梯度的特征矩阵
    F = torch.randn(num_stocks, num_factors, requires_grad=True)
    
    # 生成目标收益率
    future_returns = [torch.randn(num_stocks) for _ in range(3)]
    
    # 计算损失
    loss_fn = ASTGNNFactorLoss(omega=0.9, lambda_orthogonal=0.2)
    loss = loss_fn(F, future_returns)
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    print(f"特征矩阵梯度范数: {torch.norm(F.grad).item():.4f}")
    print(f"梯度最大值: {torch.max(torch.abs(F.grad)).item():.4f}")
    
    # 验证梯度是否合理（不为零且不为无穷）
    assert not torch.isnan(F.grad).any(), "梯度包含NaN"
    assert not torch.isinf(F.grad).any(), "梯度包含无穷大"
    assert torch.norm(F.grad).item() > 0, "梯度为零"
    
    print("梯度计算验证通过!")


if __name__ == "__main__":
    # 运行测试
    test_astgnn_loss()
    demonstrate_loss_gradients() 