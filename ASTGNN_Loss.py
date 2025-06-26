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
    
    def __init__(self, omega=0.9, lambda_orthogonal=0.1, max_periods=5, 
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
    
    def compute_r_square(self, F, y_t):
        """计算R-square
        
        R-square(F, y_t) = 1 - ||y_t - F(F^T F)^(-1) F^T y_t||^2 / ||y_t - mean(y_t)||^2
        
        参数：
        - F: 属性特征向量矩阵 [num_stocks, num_factors]
        - y_t: 第t期标准化收益率 [num_stocks]
        
        返回：
        - r_square: R平方值
        """
        if F.dim() != 2 or y_t.dim() != 1:
            raise ValueError(f"F维度应为2，y_t维度应为1，实际: F={F.dim()}, y_t={y_t.dim()}")
        
        num_stocks, num_factors = F.shape
        if y_t.shape[0] != num_stocks:
            raise ValueError(f"F和y_t的股票数量不匹配: {num_stocks} vs {y_t.shape[0]}")
        
        # 计算 F^T F + 正则化项（避免奇异）
        FTF = torch.matmul(F.T, F) + self.eps * torch.eye(num_factors, device=F.device)
        
        try:
            # 计算 (F^T F)^(-1)
            FTF_inv = torch.inverse(FTF)
        except RuntimeError:
            # 如果矩阵奇异，使用伪逆
            FTF_inv = torch.pinverse(FTF)
        
        # 计算投影: F(F^T F)^(-1) F^T y_t
        projection = torch.matmul(F, torch.matmul(FTF_inv, torch.matmul(F.T, y_t)))
        
        # 计算残差
        residual = y_t - projection
        residual_ss = torch.sum(residual ** 2)
        
        # 计算总变差 (y_t相对于其均值的变差)
        y_mean = torch.mean(y_t)
        total_ss = torch.sum((y_t - y_mean) ** 2)
        
        # R-square = 1 - RSS/TSS
        r_square = 1.0 - residual_ss / (total_ss + self.eps)
        
        # 确保R-square在合理范围内
        r_square = torch.clamp(r_square, min=-1.0, max=1.0)
        
        return r_square
    
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
        """前向传播计算损失
        
        参数：
        - F: 属性特征向量矩阵 [num_stocks, num_factors]
        - future_returns_list: 未来多期收益率列表 [y_1, y_2, ..., y_T]
                             每个y_t的形状为 [num_stocks]
        - return_individual_losses: 是否返回各项损失的详细信息
        
        返回：
        - total_loss: 总损失
        - (可选) loss_details: 损失详细信息字典
        """
        if not isinstance(future_returns_list, (list, tuple)):
            raise ValueError("future_returns_list应该是列表或元组")
        
        num_periods = len(future_returns_list)
        if num_periods == 0:
            raise ValueError("future_returns_list不能为空")
        
        # 第一部分：时间加权的R-square损失
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
            
            # 损失函数是负R-square（因为我们想最大化R-square）
            weighted_r_square_loss += weight * (1.0 - r_square)
        
        # 第二部分：相关系数正交惩罚项
        orthogonal_penalty = self.compute_correlation_penalty(F)
        
        # 总损失
        total_loss = weighted_r_square_loss + self.lambda_orthogonal * orthogonal_penalty
        
        if return_individual_losses:
            loss_details = {
                'total_loss': total_loss.item(),
                'r_square_loss': weighted_r_square_loss.item(),
                'orthogonal_penalty': orthogonal_penalty.item(),
                'individual_r_squares': r_square_losses,
                'time_weights_used': self.time_weights[:len(future_returns_list)].tolist()
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
    print(f"各期R-square: {[f'{r:.4f}' for r in details['individual_r_squares']]}")
    
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