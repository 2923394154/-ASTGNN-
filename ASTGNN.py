import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

# 导入已有的组件
from GAT import GraphAttentionLayer, MultiHeadGAT, GAT
from Res_C import BasicResidualBlock, GraphResidualLayer
from Full_C import FullyConnectedLayer, GraphFullyConnectedLayer


class W2GATFactorExtractor(nn.Module):
    """W2-GAT因子提取器（对应图中架构）
    
    架构流程：x1, x2, ..., xT → GRU → GAT → Res-C → Full-C → 输出风险因子矩阵M
    这对应图中的Cell-X单元，用于提取风险因子
    """
    
    def __init__(self, 
                 # 输入参数
                 input_size,           # 输入特征维度（21个Barra因子）
                 
                 # GRU参数
                 gru_hidden_size=64,
                 gru_num_layers=2,
                 
                 # GAT参数
                 gat_hidden_size=128,
                 gat_n_heads=4,
                 
                 # Res_C参数
                 res_hidden_size=128,
                 
                 # Full_C参数
                 output_size=32,       # 风险因子数量K
                 
                 # 通用参数
                 dropout=0.1):
        """
        参数说明：
        - input_size: 每个时间步的输入特征维度（对应图中x_t的维度）
        - gru_hidden_size: GRU隐藏状态维度
        - gru_num_layers: GRU层数
        - gat_hidden_size: GAT输出维度
        - gat_n_heads: GAT多头注意力头数
        - res_hidden_size: 残差块隐藏维度
        - output_size: 最终输出的风险因子数量（M矩阵的列数）
        """
        super(W2GATFactorExtractor, self).__init__()
        
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        
        print(f"构建W2-GAT架构:")
        print(f"  输入维度: {input_size}")
        print(f"  GRU隐藏层: {gru_hidden_size}")
        print(f"  GAT隐藏层: {gat_hidden_size} (头数: {gat_n_heads})")
        print(f"  残差隐藏层: {res_hidden_size}")
        print(f"  输出因子数: {output_size}")
        
        # 1. GRU层 - 处理时间序列信息
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout if gru_num_layers > 1 else 0,
            batch_first=True
        )
        
        # 2. GAT层 - 处理图关系信息
        self.gat = MultiHeadGAT(
            in_features=gru_hidden_size,
            out_features=gat_hidden_size,
            n_heads=gat_n_heads,
            concat=True  # 多头拼接
        )
        
        # 3. Res_C层 - 残差连接增强特征
        self.res_block = BasicResidualBlock(
            in_features=gat_hidden_size,
            out_features=res_hidden_size
        )
        
        # 4. Full_C层 - 全连接输出风险因子矩阵M
        self.full_c = GraphFullyConnectedLayer(
            in_features=res_hidden_size,
            out_features=output_size,
            activation='tanh',  # 风险因子使用tanh激活，确保有界
            dropout=dropout,
            use_layer_norm=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # NN-Layer (额外的神经网络层，对应图中的NN-Layer)
        self.nn_layer = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size)
        )
    
    def forward(self, sequential_inputs, adj_matrix):
        """前向传播 - 实现图中的完整流程
        
        参数：
        - sequential_inputs: [batch_size, seq_len, num_stocks, input_features] 对应图中x1,x2,...xT
        - adj_matrix: [num_stocks, num_stocks] 或 [batch_size, num_stocks, num_stocks]
        
        返回：
        - risk_factors: [batch_size, num_stocks, output_size] 风险因子矩阵M
        - attention_weights: GAT注意力权重
        - intermediate_outputs: 中间输出用于分析
        """
        batch_size, seq_len, num_stocks, input_features = sequential_inputs.shape
        
        print(f"W2-GAT前向传播:")
        print(f"  输入形状: {sequential_inputs.shape}")
        
        # === 1. GRU层处理时序信息 ===
        # 重塑为 [batch_size * num_stocks, seq_len, input_features]
        gru_input = sequential_inputs.view(batch_size * num_stocks, seq_len, input_features)
        
        # GRU前向传播
        gru_output, gru_hidden = self.gru(gru_input)  # [batch*stocks, seq_len, gru_hidden]
        
        # 取最后一个时间步的输出 (对应图中h_T)
        gru_last = gru_output[:, -1, :]  # [batch*stocks, gru_hidden]
        
        # 重塑回 [batch_size, num_stocks, gru_hidden]
        gru_features = gru_last.view(batch_size, num_stocks, self.gru_hidden_size)
        gru_features = self.dropout(gru_features)
        
        print(f"  GRU输出形状: {gru_features.shape}")
        
        # === 2. GAT层处理图关系 ===
        # 处理邻接矩阵维度
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 对每个batch分别处理GAT
        gat_outputs = []
        attention_weights_list = []
        
        for i in range(batch_size):
            gat_out, attention = self.gat(gru_features[i], adj_matrix[i])
            gat_outputs.append(gat_out)
            attention_weights_list.append(attention)
        
        gat_features = torch.stack(gat_outputs, dim=0)  # [batch, stocks, gat_hidden]
        print(f"  GAT输出形状: {gat_features.shape}")
        
        # === 3. Res_C层残差连接 ===
        gat_flat = gat_features.view(batch_size * num_stocks, -1)
        res_out = self.res_block(gat_flat)
        res_features = res_out.view(batch_size, num_stocks, -1)
        
        print(f"  Res-C输出形状: {res_features.shape}")
        
        # === 4. Full_C层输出风险因子 ===
        res_flat = res_features.view(batch_size * num_stocks, -1)
        risk_factors_flat = self.full_c(res_flat)
        risk_factors = risk_factors_flat.view(batch_size, num_stocks, self.output_size)
        
        print(f"  Full-C输出形状: {risk_factors.shape}")
        
        # === 5. NN-Layer额外处理 ===
        # 对应图中的NN-Layer
        risk_factors_flat = risk_factors.view(batch_size * num_stocks, -1)
        enhanced_factors_flat = self.nn_layer(risk_factors_flat)
        enhanced_factors = enhanced_factors_flat.view(batch_size, num_stocks, self.output_size)
        
        print(f"  NN-Layer输出形状: {enhanced_factors.shape}")
        
        # 中间输出用于分析
        intermediate_outputs = {
            'gru_output': gru_features,
            'gat_output': gat_features, 
            'res_output': res_features,
            'full_c_output': risk_factors,
            'final_output': enhanced_factors
        }
        
        return enhanced_factors, attention_weights_list, intermediate_outputs


class TimeGraphConvolution(nn.Module):
    """时间图卷积层 (TGC)
    
    实现两种图模型形式：
    1. 加法形式：Z = (I + softmax(ReLU(MM^T)))XW （动量效应）
    2. 减法形式：Z = (I - softmax(ReLU(MM^T)))XW （中性化效应）
    
    其中M是风险因子矩阵，X是输入特征矩阵
    """
    
    def __init__(self, in_features, out_features, mode='add', 
                 dropout=0.1, use_bias=True):
        """
        参数：
        - in_features: 输入特征维度
        - out_features: 输出特征维度  
        - mode: 'add' 或 'subtract'，对应加法或减法形式
        - dropout: dropout概率
        - use_bias: 是否使用偏置
        """
        super(TimeGraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.dropout = dropout
        
        # 权重矩阵W
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 偏置
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 参数初始化
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def compute_similarity_matrix(self, M):
        """计算相似性矩阵：softmax(ReLU(MM^T))
        
        参数：
        - M: 风险因子矩阵 [batch_size, num_stocks, num_factors]
        
        返回：
        - similarity: 相似性矩阵 [batch_size, num_stocks, num_stocks]
        """
        # 计算MM^T
        M_transpose = M.transpose(-2, -1)  # [batch, factors, stocks]
        similarity_raw = torch.matmul(M, M_transpose)  # [batch, stocks, stocks]
        
        # 应用ReLU激活
        similarity_activated = F.relu(similarity_raw)
        
        # 应用softmax归一化（按行）
        similarity = F.softmax(similarity_activated, dim=-1)
        
        return similarity
    
    def forward(self, X, M):
        """前向传播
        
        参数：
        - X: 输入特征矩阵 [batch_size, num_stocks, in_features]
        - M: 风险因子矩阵 [batch_size, num_stocks, num_factors]
        
        返回：
        - Z: 输出特征矩阵 [batch_size, num_stocks, out_features]
        """
        batch_size, num_stocks, _ = X.shape
        
        # 1. 计算相似性矩阵
        similarity = self.compute_similarity_matrix(M)  # [batch, stocks, stocks]
        
        # 2. 线性变换 XW
        X_transformed = torch.matmul(X, self.weight)  # [batch, stocks, out_features]
        
        # 3. 根据模式选择加法或减法形式
        identity = torch.eye(num_stocks, device=X.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.mode == 'add':
            # 加法形式：Z = (I + S)XW  (动量效应)
            graph_matrix = identity + similarity
        elif self.mode == 'subtract':
            # 减法形式：Z = (I - S)XW  (中性化效应)
            graph_matrix = identity - similarity
        else:
            raise ValueError(f"不支持的模式: {self.mode}, 请选择'add'或'subtract'")
        
        # 4. 图卷积操作
        Z = torch.matmul(graph_matrix, X_transformed)  # [batch, stocks, out_features]
        
        # 5. 添加偏置
        if self.bias is not None:
            Z = Z + self.bias
        
        # 6. Dropout
        Z = self.dropout_layer(Z)
        
        return Z


class RelationalEmbeddingLayer(nn.Module):
    """关系嵌入层 - 使用多个TGC层处理特征"""
    
    def __init__(self, in_features, hidden_features, out_features,
                 num_tgc_layers=2, tgc_modes=['add', 'subtract'], 
                 dropout=0.1):
        """
        参数：
        - in_features: 输入特征维度
        - hidden_features: 隐藏层特征维度
        - out_features: 输出特征维度
        - num_tgc_layers: TGC层数
        - tgc_modes: TGC模式列表
        - dropout: dropout概率
        """
        super(RelationalEmbeddingLayer, self).__init__()
        
        self.num_tgc_layers = num_tgc_layers
        self.tgc_modes = tgc_modes
        
        # 创建多个TGC层
        self.tgc_layers = nn.ModuleList()
        
        for i in range(num_tgc_layers):
            # 确定当前层的模式
            mode = tgc_modes[i % len(tgc_modes)]
            
            # 确定输入输出维度
            if i == 0:
                in_dim = in_features
            else:
                in_dim = hidden_features
                
            if i == num_tgc_layers - 1:
                out_dim = out_features
            else:
                out_dim = hidden_features
            
            # 创建TGC层
            tgc_layer = TimeGraphConvolution(
                in_features=in_dim,
                out_features=out_dim,
                mode=mode,
                dropout=dropout
            )
            
            self.tgc_layers.append(tgc_layer)
            
        # 激活函数
        self.activation = nn.ReLU()
    
    def forward(self, X, M):
        """前向传播
        
        参数：
        - X: 输入特征矩阵 [batch_size, num_stocks, in_features]
        - M: 风险因子矩阵 [batch_size, num_stocks, num_factors]
        
        返回：
        - output: 嵌入后的特征矩阵 [batch_size, num_stocks, out_features]
        """
        output = X
        
        for i, tgc_layer in enumerate(self.tgc_layers):
            output = tgc_layer(output, M)
            
            # 最后一层不使用激活函数
            if i < len(self.tgc_layers) - 1:
                output = self.activation(output)
        
        return output


class ASTGNNFactorModel(nn.Module):
    """完整的ASTGNN因子模型 - 对应图中完整架构"""
    
    def __init__(self, 
                 # 输入维度
                 sequential_input_size,    # 时序输入维度（21个Barra因子）
                 
                 # W2-GAT参数（对应图中的Cell-X）
                 gru_hidden_size=64,
                 gru_num_layers=2,
                 gat_hidden_size=128,
                 gat_n_heads=4,
                 res_hidden_size=128,
                 num_risk_factors=32,      # 风险因子数量K
                 
                 # TGC参数
                 tgc_hidden_size=128,
                 tgc_output_size=64,
                 num_tgc_layers=2,
                 tgc_modes=['add', 'subtract'],
                 
                 # 预测层参数
                 prediction_hidden_sizes=[128, 64],
                 num_predictions=1,
                 
                 # 其他参数
                 dropout=0.1):
        """
        完整ASTGNN架构，对应论文图中的网络结构：
        
        1. W2-GAT部分（Cell-X）：Sequential Inputs → GRU → GAT → Res-C → Full-C → 风险因子M
        2. TGC部分：使用风险因子M进行图卷积
        3. 预测部分：最终输出预测结果
        """
        super(ASTGNNFactorModel, self).__init__()
        
        print(f"=== 构建完整ASTGNN模型 ===")
        print(f"输入维度: {sequential_input_size}")
        print(f"风险因子数量: {num_risk_factors}")
        print(f"TGC模式: {tgc_modes}")
        
        # 1. W2-GAT因子提取器（对应图中的Cell-X部分）
        self.factor_extractor = W2GATFactorExtractor(
            input_size=sequential_input_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            gat_hidden_size=gat_hidden_size,
            gat_n_heads=gat_n_heads,
            res_hidden_size=res_hidden_size,
            output_size=num_risk_factors,
            dropout=dropout
        )
        
        # 2. TGC关系嵌入层
        self.relational_embedding = RelationalEmbeddingLayer(
            in_features=sequential_input_size,  # 使用原始输入特征
            hidden_features=tgc_hidden_size,
            out_features=tgc_output_size,
            num_tgc_layers=num_tgc_layers,
            tgc_modes=tgc_modes,
            dropout=dropout
        )
        
        # 3. 预测层（使用已有的Full-C组件）
        prediction_layer_sizes = [tgc_output_size] + prediction_hidden_sizes + [num_predictions]
        
        self.prediction_layers = nn.ModuleList()
        for i in range(len(prediction_layer_sizes) - 1):
            is_final = (i == len(prediction_layer_sizes) - 2)
            layer = FullyConnectedLayer(
                in_features=prediction_layer_sizes[i],
                out_features=prediction_layer_sizes[i + 1],
                activation=None if is_final else 'relu',
                dropout=0 if is_final else dropout,
                batch_norm=not is_final
            )
            self.prediction_layers.append(layer)
    
    def forward(self, sequential_inputs, adj_matrix):
        """前向传播 - 实现完整的ASTGNN架构
        
        参数：
        - sequential_inputs: [batch_size, seq_len, num_stocks, input_features] 对应图中x1,x2,...xT
        - adj_matrix: [num_stocks, num_stocks] 或 [batch_size, num_stocks, num_stocks]
        
        返回：
        - predictions: [batch_size, num_stocks, num_predictions] 最终预测结果
        - risk_factors: [batch_size, num_stocks, num_risk_factors] 生成的风险因子矩阵M  
        - attention_weights: GAT注意力权重
        - intermediate_outputs: 中间结果用于分析
        """
        batch_size, seq_len, num_stocks, input_features = sequential_inputs.shape
        
        print(f"\n=== ASTGNN前向传播 ===")
        print(f"输入形状: {sequential_inputs.shape}")
        
        # 1. 使用W2-GAT提取风险因子矩阵M
        risk_factors, attention_weights, intermediate_outputs = self.factor_extractor(
            sequential_inputs, adj_matrix
        )
        
        print(f"风险因子矩阵M形状: {risk_factors.shape}")
        
        # 2. 使用当前时刻的输入特征作为X
        current_features = sequential_inputs[:, -1, :, :]  # [batch, stocks, features]
        print(f"当前特征X形状: {current_features.shape}")
        
        # 3. TGC关系嵌入
        embedded_features = self.relational_embedding(current_features, risk_factors)
        print(f"TGC嵌入后形状: {embedded_features.shape}")
        
        # 4. 预测层
        output = embedded_features
        # 重塑为2D进行全连接计算
        output_flat = output.view(batch_size * num_stocks, -1)
        
        for i, layer in enumerate(self.prediction_layers):
            output_flat = layer(output_flat)
            print(f"预测层{i+1}输出: {output_flat.shape}")
        
        # 重塑回3D
        predictions = output_flat.view(batch_size, num_stocks, -1)
        print(f"最终预测形状: {predictions.shape}")
        
        # 更新中间输出
        intermediate_outputs.update({
            'current_features': current_features,
            'embedded_features': embedded_features,
            'predictions': predictions
        })
        
        return predictions, risk_factors, attention_weights, intermediate_outputs


def test_astgnn_model():
    """测试完整的ASTGNN模型"""
    print("=== ASTGNN因子模型测试 ===")
    
    # 设置参数
    batch_size = 4
    seq_len = 10
    num_stocks = 100
    sequential_input_size = 21  # Barra因子数量
    
    # 创建模型
    model = ASTGNNFactorModel(
        sequential_input_size=sequential_input_size,
        gru_hidden_size=64,
        gat_hidden_size=128,
        gat_n_heads=4,
        num_risk_factors=32,
        tgc_hidden_size=128,
        tgc_output_size=64,
        num_tgc_layers=2,
        tgc_modes=['add', 'subtract'],
        prediction_hidden_sizes=[128, 64],
        num_predictions=1,
        dropout=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    sequential_inputs = torch.randn(batch_size, seq_len, num_stocks, sequential_input_size)
    adj_matrix = torch.randint(0, 2, (num_stocks, num_stocks)).float()
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
    adj_matrix.fill_diagonal_(1)  # 添加自环
    
    print(f"序列输入形状: {sequential_inputs.shape}")
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    
    # 前向传播
    with torch.no_grad():
        predictions, risk_factors, attention_weights, intermediate_outputs = model(sequential_inputs, adj_matrix)
    
    print(f"预测输出形状: {predictions.shape}")
    print(f"风险因子矩阵形状: {risk_factors.shape}")
    print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"风险因子范围: [{risk_factors.min():.4f}, {risk_factors.max():.4f}]")
    
    # 验证相似性矩阵计算
    print("\n=== 相似性矩阵验证 ===")
    tgc_layer = TimeGraphConvolution(64, 128, mode='add')
    
    # 计算相似性矩阵
    similarity = tgc_layer.compute_similarity_matrix(risk_factors)
    print(f"相似性矩阵形状: {similarity.shape}")
    print(f"相似性矩阵行和（应接近1）: {similarity.sum(dim=-1)[:2, :5]}")  # 前2个batch，前5个股票


def load_and_test_with_real_data():
    """使用模拟生成的数据测试模型"""
    print("\n=== 使用模拟数据测试 ===")
    
    try:
        # 加载数据
        training_data = torch.load('astgnn_training_data.pt', weights_only=False)
        
        factors = training_data['factors']  # [time, stocks, factors]
        returns = training_data['returns_standardized']  # [time, stocks]
        adj_matrices = training_data['adjacency_matrices']  # [time, stocks, stocks]
        
        print(f"因子数据形状: {factors.shape}")
        print(f"收益率数据形状: {returns.shape}")
        print(f"邻接矩阵形状: {adj_matrices.shape}")
        
        # 准备模型输入
        seq_len = 10
        num_periods, num_stocks, num_factors = factors.shape
        
        # 创建模型
        model = ASTGNNFactorModel(
            sequential_input_size=num_factors,
            num_risk_factors=32,
            tgc_modes=['add', 'subtract']
        )
        
        # 创建序列输入（使用滑动窗口）
        sequences = []
        targets = []
        adj_seq = []
        
        for i in range(seq_len, min(num_periods, seq_len + 5)):  # 只测试几个样本
            # 输入：过去seq_len期的因子数据
            seq_input = factors[i-seq_len:i].transpose(0, 1)  # [stocks, seq_len, factors]
            seq_input = seq_input.transpose(0, 1)  # [seq_len, stocks, factors]
            
            # 目标：当期的标准化收益率
            target = returns[i]  # [stocks]
            
            # 邻接矩阵
            adj = adj_matrices[i]  # [stocks, stocks]
            
            sequences.append(seq_input)
            targets.append(target)
            adj_seq.append(adj)
        
        # 转换为张量
        sequences = torch.stack(sequences)  # [batch, seq_len, stocks, factors]
        targets = torch.stack(targets)  # [batch, stocks]
        adj_matrices_batch = torch.stack(adj_seq)  # [batch, stocks, stocks]
        
        print(f"序列数据形状: {sequences.shape}")
        print(f"目标数据形状: {targets.shape}")
        print(f"邻接矩阵批次形状: {adj_matrices_batch.shape}")
        
        # 测试模型
        with torch.no_grad():
            predictions, risk_factors, attention_weights, intermediate_outputs = model(sequences, adj_matrices_batch[0])
        
        print(f"预测结果形状: {predictions.shape}")
        print(f"风险因子形状: {risk_factors.shape}")
        print(f"预测值统计: 均值={predictions.mean():.4f}, 标准差={predictions.std():.4f}")
        print(f"目标值统计: 均值={targets.mean():.4f}, 标准差={targets.std():.4f}")
        
        # 计算相关性（平展所有批次）
        pred_flat = predictions.squeeze().view(-1)
        target_flat = targets.view(-1)
        
        # 检查是否有有效的数据
        if len(pred_flat) > 1 and len(target_flat) > 1:
            correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
            print(f"预测与目标的相关系数: {correlation:.4f}")
        else:
            print("数据不足，无法计算相关系数")
        
    except FileNotFoundError:
        print("未找到训练数据文件，请先运行BarraFactorSimulator.py生成数据")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 基础模型测试
    test_astgnn_model()
    
    # 使用模拟数据测试
    load_and_test_with_real_data()