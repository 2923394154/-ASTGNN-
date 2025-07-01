#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTGNN因子损失函数 - 专注RankIC优化版本
直接优化RankIC指标，确保因子预测方向正确
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class ASTGNNFactorLoss(nn.Module):
    """
    ASTGNN因子损失函数 - RankIC导向版本
    
    核心思想：
    1. 直接优化RankIC相关性
    2. 确保因子预测方向正确
    3. 防止因子分布偏斜
    4. 控制因子方差稳定性
    """
    
    def __init__(self, 
                 omega: float = 0.9,
                 lambda_orthogonal: float = 0.01,
                 lambda_rank_ic: float = 5.0,          # RankIC损失权重
                 lambda_distribution: float = 1.0,      # 分布正则化权重
                 lambda_variance: float = 0.5,          # 方差稳定性权重
                 max_periods: int = 3,
                 eps: float = 1e-6,
                 regularization_type: str = 'frobenius'):
        super().__init__()
        
        self.omega = omega
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_rank_ic = lambda_rank_ic
        self.lambda_distribution = lambda_distribution
        self.lambda_variance = lambda_variance
        self.max_periods = max_periods
        self.eps = eps
        self.regularization_type = regularization_type
        
        print(f"🎯 RankIC导向损失函数初始化:")
        print(f"  ω时间权重: {omega}")
        print(f"  λ正交惩罚: {lambda_orthogonal}")
        print(f"  λRankIC权重: {lambda_rank_ic}")
        print(f"  λ分布正则: {lambda_distribution}")
        print(f"  λ方差稳定: {lambda_variance}")
        
        # 保存损失权重用于动态调整
        self.current_rank_ic_weight = lambda_rank_ic
    
    def compute_rank_ic_loss(self, factors: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        直接计算RankIC损失
        
        Args:
            factors: [num_stocks] 因子值
            returns: [num_stocks] 收益率
        """
        # 确保数据有效
        valid_mask = ~(torch.isnan(factors) | torch.isnan(returns))
        if valid_mask.sum() < 10:  # 至少需要10个有效样本
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        valid_factors = factors[valid_mask]
        valid_returns = returns[valid_mask]
        
        # 计算排序
        factor_ranks = torch.argsort(torch.argsort(valid_factors)).float()
        return_ranks = torch.argsort(torch.argsort(valid_returns)).float()
        
        # 计算Spearman相关系数 (RankIC)
        n = len(factor_ranks)
        factor_ranks_centered = factor_ranks - factor_ranks.mean()
        return_ranks_centered = return_ranks - return_ranks.mean()
        
        numerator = torch.sum(factor_ranks_centered * return_ranks_centered)
        denominator = torch.sqrt(torch.sum(factor_ranks_centered**2) * torch.sum(return_ranks_centered**2))
        
        if denominator < self.eps:
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        rank_ic = numerator / denominator
        
        # 损失：我们希望最大化正向RankIC
        # 如果RankIC为负，额外惩罚以强制正向预测
        if rank_ic < 0:
            rank_ic_loss = -rank_ic + 2.0 * torch.abs(rank_ic)  # 重度惩罚负RankIC
        else:
            rank_ic_loss = -rank_ic  # 最大化正RankIC
        
        return rank_ic_loss
    
    def compute_current_rank_ic(self, factors: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        计算当前的RankIC值用于动态权重调整
        """
        # 确保数据有效
        valid_mask = ~(torch.isnan(factors) | torch.isnan(returns))
        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        valid_factors = factors[valid_mask]
        valid_returns = returns[valid_mask]
        
        # 计算排序
        factor_ranks = torch.argsort(torch.argsort(valid_factors)).float()
        return_ranks = torch.argsort(torch.argsort(valid_returns)).float()
        
        # 计算Spearman相关系数
        n = len(factor_ranks)
        factor_ranks_centered = factor_ranks - factor_ranks.mean()
        return_ranks_centered = return_ranks - return_ranks.mean()
        
        numerator = torch.sum(factor_ranks_centered * return_ranks_centered)
        denominator = torch.sqrt(torch.sum(factor_ranks_centered**2) * torch.sum(return_ranks_centered**2))
        
        if denominator < self.eps:
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        rank_ic = numerator / denominator
        return rank_ic
    
    def compute_distribution_regularization(self, factors: torch.Tensor) -> torch.Tensor:
        """
        计算分布正则化损失，防止因子分布偏斜
        """
        # 标准化因子
        factor_std = torch.std(factors) + self.eps
        factors_normalized = (factors - torch.mean(factors)) / factor_std
        
        # 计算偏度损失 (希望偏度接近0)
        skewness = torch.mean((factors_normalized)**3)
        skewness_loss = torch.abs(skewness)
        
        # 计算峰度损失 (希望峰度接近3，即正态分布)
        kurtosis = torch.mean((factors_normalized)**4)
        kurtosis_loss = torch.abs(kurtosis - 3.0)
        
        # 分布正则化损失
        distribution_loss = skewness_loss + 0.5 * kurtosis_loss
        
        return distribution_loss
    
    def compute_variance_stability_loss(self, factors: torch.Tensor) -> torch.Tensor:
        """
        计算方差稳定性损失，确保因子方差适中
        """
        factor_var = torch.var(factors) + self.eps
        
        # 我们希望方差在合理范围内 (0.5 - 2.0)
        target_var = 1.0
        var_loss = torch.abs(factor_var - target_var) / target_var
        
        return var_loss
    
    def compute_r_square_loss(self, factors: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        计算改进的R-square损失
        """
        # 确保输入维度正确
        if factors.dim() == 1:
            factors = factors.unsqueeze(-1)  # [num_stocks, 1]
        
        # 有效性检查
        valid_mask = ~(torch.isnan(factors).any(dim=-1) | torch.isnan(returns))
        if valid_mask.sum() < 5:
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        F_valid = factors[valid_mask]  # [valid_stocks, num_factors]
        y_valid = returns[valid_mask]  # [valid_stocks]
        
        try:
            # 计算 F^T F 和其逆矩阵
            FtF = torch.matmul(F_valid.t(), F_valid)  # [num_factors, num_factors]
            
            # 添加正则化项防止奇异
            ridge_reg = 1e-4 * torch.eye(FtF.shape[0], device=FtF.device)
            FtF_reg = FtF + ridge_reg
            
            # 使用伪逆替代直接求逆
            FtF_inv = torch.linalg.pinv(FtF_reg)
            
            # 计算投影: P = F(F^T F)^{-1}F^T
            P = torch.matmul(torch.matmul(F_valid, FtF_inv), F_valid.t())
            
            # 计算投影后的y: y_proj = P @ y
            y_proj = torch.matmul(P, y_valid)
            
            # 计算残差
            residual = y_valid - y_proj
            
            # R-square = 1 - ||residual||² / ||y - mean(y)||²
            ss_res = torch.sum(residual**2)
            ss_tot = torch.sum((y_valid - torch.mean(y_valid))**2) + self.eps
            
            r_square = 1.0 - ss_res / ss_tot
            
            # 损失函数：希望最大化R-square
            r_square_loss = -r_square
            
            return r_square_loss
            
        except Exception:
            # 如果计算失败，返回大的损失值
            return torch.tensor(10.0, device=factors.device, dtype=factors.dtype)
    
    def compute_orthogonal_penalty(self, factors: torch.Tensor) -> torch.Tensor:
        """
        计算因子正交性惩罚
        """
        if factors.dim() == 1 or factors.shape[-1] == 1:
            return torch.tensor(0.0, device=factors.device, dtype=factors.dtype)
        
        # 计算相关性矩阵
        factors_centered = factors - torch.mean(factors, dim=0, keepdim=True)
        cov_matrix = torch.matmul(factors_centered.t(), factors_centered)
        
        # 对角线归一化
        std_vec = torch.sqrt(torch.diag(cov_matrix) + self.eps)
        corr_matrix = cov_matrix / torch.outer(std_vec, std_vec)
        
        # 计算非对角线元素的平方和 (希望为0)
        mask = ~torch.eye(corr_matrix.shape[0], dtype=torch.bool, device=factors.device)
        
        if self.regularization_type == 'frobenius':
            penalty = torch.sum(corr_matrix[mask]**2)
        else:  # 'nuclear'
            penalty = torch.sum(torch.abs(corr_matrix[mask]))
        
        return penalty
    
    def forward(self, factors: torch.Tensor, future_returns_list: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播计算总损失
        
        Args:
            factors: [num_stocks, num_factors] 或 [num_stocks] 因子预测
            future_returns_list: 未来多期收益率列表
        """
        # 确保数据类型一致性 - 修复混合精度训练问题
        total_loss = torch.tensor(0.0, device=factors.device, dtype=factors.dtype, requires_grad=True)
        
        # 确保factors是2D
        if factors.dim() == 1:
            factors = factors.unsqueeze(-1)
        
        num_periods = min(len(future_returns_list), self.max_periods)
        
        for t in range(num_periods):
            returns_t = future_returns_list[t]
            time_weight = self.omega ** t
            
            # 对单因子情况的特殊处理
            if factors.shape[-1] == 1:
                factor_t = factors[:, 0]  # [num_stocks]
            else:
                factor_t = factors[:, 0]  # 取第一个因子
            
            # 1. RankIC损失 (最重要)
            rank_ic_loss = self.compute_rank_ic_loss(factor_t, returns_t)
            
            # 动态调整RankIC权重 - 如果当前预测效果不佳，加大惩罚
            current_rank_ic = self.compute_current_rank_ic(factor_t, returns_t)
            dynamic_rank_ic_weight = self.lambda_rank_ic
            if current_rank_ic < 0:
                dynamic_rank_ic_weight *= 2.5  # RankIC为负时权重加强
            
            # 2. R-square损失
            r_square_loss = self.compute_r_square_loss(factors, returns_t)
            
            # 3. 分布正则化损失
            distribution_loss = self.compute_distribution_regularization(factor_t)
            
            # 4. 方差稳定性损失
            variance_loss = self.compute_variance_stability_loss(factor_t)
            
            # 组合损失 - 加强RankIC权重
            period_loss = (
                dynamic_rank_ic_weight * rank_ic_loss +
                r_square_loss +
                self.lambda_distribution * distribution_loss +
                self.lambda_variance * variance_loss
            )
            
            total_loss = total_loss + time_weight * period_loss
        
        # 5. 正交性惩罚 (如果是多因子)
        if factors.shape[-1] > 1:
            orthogonal_penalty = self.compute_orthogonal_penalty(factors)
            total_loss = total_loss + self.lambda_orthogonal * orthogonal_penalty
        
        return total_loss 