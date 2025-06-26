import torch
import torch.nn as nn
import torch.nn.functional as F

# 从GAT_test.py导入GraphAttentionLayer
try:
    from GAT_test import GraphAttentionLayer
except ImportError:
    # 如果导入失败，定义一个简化版本
    class GraphAttentionLayer(nn.Module):
        def __init__(self, in_features, out_features, dropout=0.4):
            super(GraphAttentionLayer, self).__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.dropout = dropout
        
        def forward(self, x, adj_matrix):
            out = self.linear(x)
            out = F.dropout(out, self.dropout, training=self.training)
            # 返回输出和虚拟的注意力权重
            attention = torch.ones(x.size(0), x.size(0)) / x.size(0)
            return out, attention

class FullyConnectedLayer(nn.Module):
    """全连接变换加BatchNorm层 (Full-C)
    
    这个层包含：
    1. 线性变换 (全连接层)
    2. BatchNorm归一化
    3. 可选的激活函数
    4. 可选的Dropout
    """
    
    def __init__(self, in_features, out_features, activation='relu', 
                 dropout=0.0, bias=True, batch_norm=True):
        """
        参数：
        - in_features: 输入特征维度
        - out_features: 输出特征维度
        - activation: 激活函数类型 ('relu', 'leaky_relu', 'elu', 'gelu', None)
        - dropout: dropout概率
        - bias: 是否使用偏置（如果使用BatchNorm，通常设为False）
        - batch_norm: 是否使用BatchNorm
        """
        super(FullyConnectedLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_batch_norm = batch_norm
        self.dropout_rate = dropout
        
        # 全连接层
        # 如果使用BatchNorm，通常不需要bias
        self.linear = nn.Linear(in_features, out_features, 
                               bias=bias if not batch_norm else False)
        
        # BatchNorm层
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # Dropout层
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络参数"""
        # Xavier初始化
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        参数：
        - x: 输入张量，形状为 [batch_size, in_features] 或 [num_nodes, in_features]
        
        返回：
        - 变换后的特征张量
        """
        # 线性变换
        out = self.linear(x)
        
        # BatchNorm
        if self.use_batch_norm:
            out = self.batch_norm(out)
        
        # 激活函数
        if self.activation is not None:
            out = self.activation(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class MultiLayerFullyConnected(nn.Module):
    """多层全连接网络，每层都是Full-C结构"""
    
    def __init__(self, layer_sizes, activation='relu', dropout=0.0, 
                 batch_norm=True, final_activation=False):
        """
        参数：
        - layer_sizes: 各层的神经元数量列表，如 [64, 128, 256, 10]
        - activation: 激活函数类型
        - dropout: dropout概率
        - batch_norm: 是否使用BatchNorm
        - final_activation: 最后一层是否使用激活函数
        """
        super(MultiLayerFullyConnected, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            
            # 最后一层的特殊处理
            is_final_layer = (i == len(layer_sizes) - 2)
            layer_activation = activation if not is_final_layer or final_activation else None
            layer_dropout = dropout if not is_final_layer else 0.0
            
            layer = FullyConnectedLayer(
                in_features=in_features,
                out_features=out_features,
                activation=layer_activation,
                dropout=layer_dropout,
                batch_norm=batch_norm and not is_final_layer  # 最后一层通常不用BatchNorm
            )
            
            self.layers.append(layer)
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x


class GraphFullyConnectedLayer(nn.Module):
    """适用于图神经网络的Full-C层
    
    这个变种专门为图数据设计，可以处理节点特征
    """
    
    def __init__(self, in_features, out_features, activation='relu', 
                 dropout=0.0, use_layer_norm=False):
        """
        参数：
        - use_layer_norm: 使用LayerNorm而不是BatchNorm（对图数据更合适）
        """
        super(GraphFullyConnectedLayer, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        
        # 对于图数据，LayerNorm通常比BatchNorm更合适
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.BatchNorm1d(out_features)
        
        self.use_layer_norm = use_layer_norm
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 参数初始化
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        参数：
        - x: 节点特征矩阵 [num_nodes, in_features]
        """
        # 线性变换
        out = self.linear(x)
        
        # 归一化
        if self.use_layer_norm:
            out = self.norm(out)
        else:
            # 对于BatchNorm，需要确保输入是2D的
            if out.dim() == 2:
                out = self.norm(out)
            else:
                raise ValueError("BatchNorm需要2D输入")
        
        # 激活函数
        if self.activation is not None:
            out = self.activation(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


def test_full_c_layers():
    """测试Full-C层的各种实现（不依赖GAT）"""
    print("=== Full-C 层测试 ===")
    
    # 测试数据
    batch_size = 32
    num_nodes = 100
    in_features = 64
    out_features = 128
    
    # 1. 基本Full-C层测试
    print("\n1. 基本 Full-C 层测试")
    basic_layer = FullyConnectedLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    
    print(f"输入形状: {x.shape}")
    out = basic_layer(x)
    print(f"输出形状: {out.shape}")
    print(f"参数数量: {sum(p.numel() for p in basic_layer.parameters())}")
    
    # 2. 多层Full-C网络测试
    print("\n2. 多层 Full-C 网络测试")
    multi_layer = MultiLayerFullyConnected([64, 128, 256, 128, 10])
    
    print(f"输入形状: {x.shape}")
    out = multi_layer(x)
    print(f"输出形状: {out.shape}")
    
    # 3. 图数据适用的Full-C层测试
    print("\n3. 图 Full-C 层测试（LayerNorm）")
    graph_layer = GraphFullyConnectedLayer(in_features, out_features, 
                                         use_layer_norm=True)
    x_graph = torch.randn(num_nodes, in_features)
    
    print(f"输入形状: {x_graph.shape}")
    out_graph = graph_layer(x_graph)
    print(f"输出形状: {out_graph.shape}")
    
    # 4. 不同激活函数的比较
    print("\n4. 不同激活函数比较")
    activations = ['relu', 'elu', 'leaky_relu', 'gelu']
    
    for act in activations:
        layer = FullyConnectedLayer(64, 64, activation=act)
        test_input = torch.randn(10, 64)
        output = layer(test_input)
        
        print(f"{act:12}: 输出范围 [{output.min():.3f}, {output.max():.3f}], "
              f"均值 {output.mean():.3f}")


def demonstrate_full_c_in_gat():
    """演示如何在GAT中使用Full-C层"""
    print("\n=== 在GAT中集成Full-C层 ===")
    
    class GATWithFullC(nn.Module):
        """集成了Full-C层的GAT网络"""
        
        def __init__(self, in_features, hidden_features, num_classes, n_heads=4):
            super(GATWithFullC, self).__init__()
            
            # GAT层
            self.gat1 = GraphAttentionLayer(in_features, hidden_features)
            
            # Full-C层用于特征变换
            self.full_c1 = GraphFullyConnectedLayer(
                hidden_features, hidden_features, 
                activation='elu', dropout=0.2, use_layer_norm=True
            )
            
            # 另一个GAT层
            self.gat2 = GraphAttentionLayer(hidden_features, hidden_features)
            
            # 最终的Full-C分类层
            self.classifier = GraphFullyConnectedLayer(
                hidden_features, num_classes,
                activation=None, use_layer_norm=False
            )
        
        def forward(self, x, adj_matrix):
            # 第一个GAT层
            x, attention1 = self.gat1(x, adj_matrix)
            x = F.elu(x)
            
            # Full-C层进行特征变换
            x = self.full_c1(x)
            
            # 第二个GAT层
            x, attention2 = self.gat2(x, adj_matrix)
            x = F.elu(x)
            
            # 分类层
            x = self.classifier(x)
            
            return F.log_softmax(x, dim=1), attention1, attention2
    
    # 测试网络
    model = GATWithFullC(in_features=10, hidden_features=64, num_classes=3)
    
    # 创建测试数据
    num_nodes = 50
    x = torch.randn(num_nodes, 10)
    adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
    adj_matrix.fill_diagonal_(1)  # 添加自环
    
    # 前向传播
    output, att1, att2 = model(x, adj_matrix)
    
    print(f"输入节点特征形状: {x.shape}")
    print(f"输出分类结果形状: {output.shape}")
    print(f"网络总参数数量: {sum(p.numel() for p in model.parameters())}")


def demonstrate_standalone_full_c():
    """演示独立的Full-C层使用方法"""
    print("\n=== 独立 Full-C 层使用演示 ===")
    
    # 创建测试数据
    batch_size = 16
    seq_len = 50  # 例如序列长度
    input_dim = 128
    
    # 时序数据示例
    time_series_data = torch.randn(batch_size, seq_len, input_dim)
    
    # 1. 对每个时间步应用Full-C变换
    print("\n1. 时序数据处理")
    full_c_temporal = FullyConnectedLayer(input_dim, 256, activation='gelu', dropout=0.1)
    
    # 重塑数据以适应Full-C层
    reshaped_data = time_series_data.view(-1, input_dim)  # [batch_size*seq_len, input_dim]
    transformed_data = full_c_temporal(reshaped_data)
    output_data = transformed_data.view(batch_size, seq_len, -1)  # 恢复形状
    
    print(f"输入形状: {time_series_data.shape}")
    print(f"输出形状: {output_data.shape}")
    
    # 2. 特征降维示例
    print("\n2. 特征降维")
    high_dim_features = torch.randn(100, 1024)  # 高维特征
    
    # 使用多层Full-C进行降维
    dimension_reducer = MultiLayerFullyConnected(
        layer_sizes=[1024, 512, 256, 64],
        activation='relu',
        dropout=0.2,
        batch_norm=True
    )
    
    reduced_features = dimension_reducer(high_dim_features)
    print(f"降维前: {high_dim_features.shape}")
    print(f"降维后: {reduced_features.shape}")
    
    # 3. 分类器示例
    print("\n3. 分类器应用")
    classifier = MultiLayerFullyConnected(
        layer_sizes=[64, 32, 10],  # 10类分类
        activation='relu',
        dropout=0.5,
        final_activation=False  # 最后一层不使用激活函数
    )
    
    logits = classifier(reduced_features)
    probabilities = F.softmax(logits, dim=1)
    
    print(f"分类logits形状: {logits.shape}")
    print(f"每个样本的概率和: {probabilities.sum(dim=1)[:5]}")  # 应该都接近1.0


if __name__ == "__main__":
    # 首先运行独立的Full-C层测试
    test_full_c_layers()
    demonstrate_standalone_full_c()
    
    # 然后尝试GAT集成（如果GraphAttentionLayer可用）
    try:
        demonstrate_full_c_in_gat()
    except Exception as e:
        print(f"\nGAT集成演示跳过: {e}")
        print("如需运行GAT集成示例，请确保GAT_test.py在同一目录下")
