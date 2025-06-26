import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
import time

# 导入模型和损失函数
from ASTGNN import ASTGNNFactorModel
from ASTGNN_Loss import ASTGNNFactorLoss
from FactorValidation import FactorValidationFramework
from BarraFactorSimulator import BarraFactorSimulator

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


def create_demo_data(num_periods=50, num_stocks=100, num_factors=21, seq_len=10):
    """创建演示数据"""
    print("=== 创建演示数据 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成基础因子数据（模拟Barra因子）
    factors = torch.randn(num_periods, num_stocks, num_factors)
    
    # 生成收益率（添加一些与因子的关系）
    factor_coefficients = torch.randn(num_factors) * 0.1
    returns = torch.zeros(num_periods, num_stocks)
    
    for t in range(num_periods):
        # 基于因子的收益率
        factor_returns = torch.matmul(factors[t], factor_coefficients)
        # 添加噪音
        noise = torch.randn(num_stocks) * 0.2
        returns[t] = factor_returns + noise
    
    # 标准化收益率
    returns_standardized = (returns - returns.mean(dim=1, keepdim=True)) / returns.std(dim=1, keepdim=True)
    
    # 生成邻接矩阵（模拟股票之间的关系）
    # 使用相关性构建邻接矩阵
    adj_matrices = torch.zeros(num_periods, num_stocks, num_stocks)
    
    for t in range(num_periods):
        # 基于因子相似性构建邻接矩阵
        factor_corr = torch.corrcoef(factors[t])
        # 将相关性转换为0-1范围的邻接矩阵
        adj_matrices[t] = torch.sigmoid(factor_corr * 2)
        # 添加自环
        adj_matrices[t].fill_diagonal_(1.0)
    
    print(f"数据创建完成:")
    print(f"  因子形状: {factors.shape}")
    print(f"  收益率形状: {returns_standardized.shape}")
    print(f"  邻接矩阵形状: {adj_matrices.shape}")
    
    return {
        'factors': factors,
        'returns_standardized': returns_standardized,
        'adjacency_matrices': adj_matrices,
        'factor_names': [f'Factor_{i+1}' for i in range(num_factors)]
    }


def prepare_training_data(data, seq_len=10, prediction_horizon=3):
    """准备训练数据"""
    print(f"\n=== 准备训练数据 ===")
    
    factors = data['factors']
    returns = data['returns_standardized']
    adj_matrices = data['adjacency_matrices']
    
    num_periods, num_stocks, num_factors = factors.shape
    
    # 创建序列数据
    sequences = []
    targets = []
    adj_seq = []
    
    valid_start = seq_len
    valid_end = num_periods - prediction_horizon
    
    for i in range(valid_start, valid_end):
        # 输入序列：过去seq_len期的因子数据
        seq_input = factors[i-seq_len:i]  # [seq_len, stocks, factors]
        
        # 目标：未来多期收益率
        target_list = []
        for h in range(1, prediction_horizon + 1):
            target_list.append(returns[i + h])
        
        # 邻接矩阵
        adj = adj_matrices[i]
        
        sequences.append(seq_input)
        targets.append(target_list)
        adj_seq.append(adj)
    
    # 转换为张量
    sequences = torch.stack(sequences)  # [samples, seq_len, stocks, factors]
    adj_matrices_batch = torch.stack(adj_seq)  # [samples, stocks, stocks]
    
    print(f"训练数据准备完成:")
    print(f"  序列数据: {sequences.shape}")
    print(f"  目标数据: {len(targets)}个样本，每个{len(targets[0])}期")
    print(f"  邻接矩阵: {adj_matrices_batch.shape}")
    
    return sequences, targets, adj_matrices_batch


def test_astgnn_architecture():
    """测试ASTGNN架构"""
    print("\n" + "="*60)
    print("ASTGNN架构测试")
    print("="*60)
    
    # 1. 创建演示数据
    data = create_demo_data(num_periods=50, num_stocks=100, num_factors=21)
    
    # 2. 准备训练数据
    sequences, targets, adj_matrices = prepare_training_data(data, seq_len=10, prediction_horizon=3)
    
    # 3. 创建模型
    print(f"\n=== 创建ASTGNN模型 ===")
    model = ASTGNNFactorModel(
        sequential_input_size=21,  # Barra因子数量
        gru_hidden_size=32,        # 减小以便快速测试
        gru_num_layers=2,
        gat_hidden_size=64,
        gat_n_heads=4,
        res_hidden_size=64,
        num_risk_factors=16,       # 风险因子数量
        
        tgc_hidden_size=64,
        tgc_output_size=32,
        num_tgc_layers=2,
        tgc_modes=['add', 'subtract'],
        
        prediction_hidden_sizes=[64, 32],
        num_predictions=1,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 测试前向传播
    print(f"\n=== 测试前向传播 ===")
    batch_size = 2
    test_sequences = sequences[:batch_size]  # [batch, seq_len, stocks, factors]
    test_adj = adj_matrices[:batch_size]     # [batch, stocks, stocks]
    
    model.eval()
    with torch.no_grad():
        predictions, risk_factors, attention_weights, intermediate_outputs = model(
            test_sequences, test_adj[0]
        )
    
    print(f"\n模型输出:")
    print(f"  预测结果: {predictions.shape}")
    print(f"  风险因子矩阵M: {risk_factors.shape}")
    print(f"  注意力权重数量: {len(attention_weights)}")
    
    # 5. 分析中间输出
    print(f"\n=== 中间输出分析 ===")
    for key, value in intermediate_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    return model, test_sequences, targets[:batch_size], test_adj


def test_loss_function(model, sequences, targets, adj_matrices):
    """测试损失函数"""
    print(f"\n=== 测试ASTGNN损失函数 ===")
    
    # 创建损失函数
    loss_fn = ASTGNNFactorLoss(
        omega=0.9,
        lambda_orthogonal=0.1,
        max_periods=3,
        regularization_type='frobenius'
    )
    
    model.eval()
    batch_size = sequences.shape[0]
    total_loss = 0.0
    loss_details_list = []
    
    with torch.no_grad():
        # 模型前向传播
        predictions, risk_factors, attention_weights, intermediate_outputs = model(
            sequences, adj_matrices[0]
        )
        
        # 计算每个样本的损失
        for b in range(batch_size):
            F_b = risk_factors[b]  # [stocks, factors] 风险因子矩阵
            target_returns_b = targets[b]  # 多期收益率列表
            
            # 计算损失
            loss_b, details_b = loss_fn(F_b, target_returns_b, return_individual_losses=True)
            total_loss += loss_b.item()
            loss_details_list.append(details_b)
            
            print(f"样本{b+1}损失详情:")
            print(f"  总损失: {details_b['total_loss']:.4f}")
            print(f"  R²损失: {details_b['r_square_loss']:.4f}")
            print(f"  正交惩罚: {details_b['orthogonal_penalty']:.4f}")
            print(f"  各期R²: {[f'{r:.4f}' for r in details_b['individual_r_squares']]}")
    
    avg_loss = total_loss / batch_size
    print(f"\n平均损失: {avg_loss:.4f}")
    
    return avg_loss, loss_details_list


def simple_training_demo(model, sequences, targets, adj_matrices, num_epochs=10):
    """简单训练演示"""
    print(f"\n=== 简单训练演示 ===")
    
    # 创建损失函数和优化器
    loss_fn = ASTGNNFactorLoss(
        omega=0.9,
        lambda_orthogonal=0.1,
        max_periods=3,
        regularization_type='frobenius'
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史
    train_losses = []
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 前向传播
        optimizer.zero_grad()
        
        predictions, risk_factors, attention_weights, intermediate_outputs = model(
            sequences, adj_matrices[0]
        )
        
        # 计算损失
        batch_size = risk_factors.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            F_b = risk_factors[b]
            target_returns_b = targets[b]
            
            loss_b = loss_fn(F_b, target_returns_b)
            total_loss += loss_b
        
        avg_loss = total_loss / batch_size
        
        # 反向传播
        avg_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        train_losses.append(avg_loss.item())
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss.item():.4f}, Time={epoch_time:.2f}s")
    
    print(f"\n训练完成！最终损失: {train_losses[-1]:.4f}")
    
    # 绘制损失曲线
    setup_chinese_font()  # 确保字体设置
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
    plt.title('ASTGNN训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('astgnn_demo_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_losses


def factor_analysis_demo(model, sequences, adj_matrices):
    """因子分析演示"""
    print(f"\n=== 因子分析演示 ===")
    
    model.eval()
    
    with torch.no_grad():
        predictions, risk_factors, attention_weights, intermediate_outputs = model(
            sequences, adj_matrices[0]
        )
    
    # 分析风险因子矩阵
    print(f"风险因子矩阵M分析:")
    print(f"  形状: {risk_factors.shape}")
    print(f"  均值: {risk_factors.mean().item():.4f}")
    print(f"  标准差: {risk_factors.std().item():.4f}")
    print(f"  最大值: {risk_factors.max().item():.4f}")
    print(f"  最小值: {risk_factors.min().item():.4f}")
    
    # 分析因子相关性
    batch_idx = 0
    factors_matrix = risk_factors[batch_idx].cpu().numpy()  # [stocks, factors]
    
    # 计算因子间相关性
    factor_corr = np.corrcoef(factors_matrix.T)  # [factors, factors]
    
    print(f"\n因子相关性分析:")
    print(f"  因子间平均相关性: {np.abs(factor_corr).mean():.4f}")
    print(f"  因子间最大相关性: {np.abs(factor_corr[~np.eye(factor_corr.shape[0], dtype=bool)]).max():.4f}")
    
    # 可视化因子相关性矩阵
    setup_chinese_font()  # 确保字体设置
    plt.figure(figsize=(10, 8))
    plt.imshow(factor_corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('风险因子相关性矩阵')
    plt.xlabel('因子编号')
    plt.ylabel('因子编号')
    plt.savefig('astgnn_factor_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return factor_corr


def main_demo():
    """主演示函数"""
    print("="*80)
    print("ASTGNN因子挖掘模型完整演示")
    print("="*80)
    print("根据论文图中的W2-GAT架构实现:")
    print("序列输入(x1,x2,...,xT) → GRU → GAT → Res-C → Full-C → NN-Layer → 输出")
    print("="*80)
    
    # 1. 架构测试
    model, sequences, targets, adj_matrices = test_astgnn_architecture()
    
    # 2. 损失函数测试
    avg_loss, loss_details = test_loss_function(model, sequences, targets, adj_matrices)
    
    # 3. 简单训练演示
    train_losses = simple_training_demo(model, sequences, targets, adj_matrices, num_epochs=20)
    
    # 4. 因子分析
    factor_corr = factor_analysis_demo(model, sequences, adj_matrices)
    
    print(f"\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("主要成果:")
    print(f"1. 成功构建W2-GAT架构（对应论文图中的Cell-X）")
    print(f"2. 实现时间图卷积（TGC）的加法和减法模式")
    print(f"3. 集成ASTGNN专用损失函数（R²损失 + 正交惩罚）")
    print(f"4. 完成端到端训练演示，最终损失: {train_losses[-1]:.4f}")
    print(f"5. 生成{factor_corr.shape[0]}个风险因子，平均相关性: {np.abs(factor_corr).mean():.4f}")
    print("="*80)


if __name__ == "__main__":
    # 运行完整演示
    main_demo() 