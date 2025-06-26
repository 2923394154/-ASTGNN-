import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResidualBlock(nn.Module):
    """基本的残差块"""
    def __init__(self, in_features, out_features):
        super(BasicResidualBlock, self).__init__()
        
        # 主路径的网络层
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # 如果输入输出维度不同，需要投影层
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        # 主路径
        main_out = self.main_path(x)
        
        # 跳跃连接（残差连接）
        shortcut_out = self.shortcut(x)
        
        # 相加并激活
        out = main_out + shortcut_out
        out = F.relu(out)
        
        return out

class ConvResidualBlock(nn.Module):
    """卷积残差块（用于图像等2D数据）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 残差连接
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

class GraphResidualLayer(nn.Module):
    """在GAT中应用残差连接"""
    def __init__(self, in_features, out_features, dropout=0.4):
        super(GraphResidualLayer, self).__init__()
        
        # 主要的图注意力层
        self.gat_layer = GraphAttentionLayer(in_features, out_features, dropout)
        
        # 投影层（如果维度不匹配）
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        
        self.dropout = dropout
        
    def forward(self, x, adj_matrix):
        # 主路径：通过GAT层
        gat_out, attention = self.gat_layer(x, adj_matrix)
        
        # 残差连接
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x
        
        # 相加
        out = gat_out + residual
        
        # 可选的层归一化
        out = F.layer_norm(out, out.shape[1:])
        
        return out, attention

class ResidualGAT(nn.Module):
    """带残差连接的GAT网络"""
    def __init__(self, in_features, hidden_features, num_classes, n_heads, n_layers=3):
        super(ResidualGAT, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(
            GraphResidualLayer(in_features, hidden_features)
        )
        
        # 中间层
        for _ in range(n_layers - 2):
            self.layers.append(
                GraphResidualLayer(hidden_features, hidden_features)
            )
        
        # 输出层
        self.output_layer = GraphAttentionLayer(hidden_features, num_classes)
        
    def forward(self, x, adj_matrix):
        attentions = []
        
        # 通过残差层
        for layer in self.layers:
            x, attention = layer(x, adj_matrix)
            attentions.append(attention)
        
        # 输出层
        x, final_attention = self.output_layer(x, adj_matrix)
        attentions.append(final_attention)
        
        return F.log_softmax(x, dim=1), attentions

# 演示不同类型的残差连接
def demonstrate_residual_connections():
    print("=== 残差连接演示 ===")
    
    # 1. 基本残差块
    print("\n1. 基本残差块测试")
    basic_block = BasicResidualBlock(64, 128)
    x = torch.randn(32, 64)  # batch_size=32, features=64
    
    print(f"输入形状: {x.shape}")
    out = basic_block(x)
    print(f"输出形状: {out.shape}")
    
    # 2. 卷积残差块
    print("\n2. 卷积残差块测试")
    conv_block = ConvResidualBlock(3, 64, stride=2)
    x_img = torch.randn(8, 3, 32, 32)  # batch_size=8, channels=3, height=32, width=32
    
    print(f"输入形状: {x_img.shape}")
    out_img = conv_block(x_img)
    print(f"输出形状: {out_img.shape}")
    
    # 3. 比较有无残差连接的效果
    print("\n3. 残差连接效果对比")
    
    # 创建测试数据
    torch.manual_seed(42)
    test_input = torch.randn(10, 64)
    
    # 无残差连接的网络
    class NormalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 有残差连接的网络
    class ResidualNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 64)
            self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
        
        def forward(self, x):
            # 第一个残差块
            out1 = F.relu(self.layer1(x))
            out1 = out1 + x  # 残差连接
            
            # 第二个残差块
            out2 = F.relu(self.layer2(out1))
            out2 = out2 + out1  # 残差连接
            
            # 第三个残差块
            out3 = F.relu(self.layer3(out2))
            out3 = out3 + out2  # 残差连接
            
            return out3
    
    normal_net = NormalNetwork()
    residual_net = ResidualNetwork()
    
    # 计算梯度流
    normal_out = normal_net(test_input)
    residual_out = residual_net(test_input)
    
    loss_normal = normal_out.sum()
    loss_residual = residual_out.sum()
    
    loss_normal.backward()
    loss_residual.backward()
    
    # 检查第一层的梯度大小
    normal_grad = normal_net.layers[0].weight.grad.norm().item()
    residual_grad = residual_net.layer1.weight.grad.norm().item()
    
    print(f"普通网络第一层梯度大小: {normal_grad:.6f}")
    print(f"残差网络第一层梯度大小: {residual_grad:.6f}")
    print(f"梯度比率 (残差/普通): {residual_grad/normal_grad:.2f}")


if __name__ == "__main__":
    # 注意：这里假设已经定义了GraphAttentionLayer
    # 运行演示
    demonstrate_residual_connections()
