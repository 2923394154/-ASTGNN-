import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.4, leaky_relu_slope=0.2):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 线性变换权重矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        
        # 注意力参数向量
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        # 激活函数和正则化
        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
        self.dropout = dropout
        
        # 参数初始化
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    
    def forward(self, h, adj_matrix):
        batch_size = h.size(0)
        
        # 1. 线性变换
        h_transformed = torch.mm(h, self.W)  # [N, out_features]
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)
        
        # 2. 计算注意力分数
        # 计算源节点分数和目标节点分数
        source_scores = torch.matmul(h_transformed, self.a[:self.out_features, :])  # [N, 1]
        target_scores = torch.matmul(h_transformed, self.a[self.out_features:, :])  # [N, 1]
        
        # 广播相加得到所有节点对的分数 [N, N]
        e = source_scores + target_scores.T
        e = self.leakyrelu(e)
        
        # 3. 掩码处理（只保留存在边的节点对）
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_matrix > 0, e, connectivity_mask)
        
        # 4. 归一化得到注意力权重
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 5. 加权聚合
        h_prime = torch.matmul(attention, h_transformed)
        
        return h_prime, attention  # 返回注意力权重用于可视化

class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads, concat=True):
        super(MultiHeadGAT, self).__init__()
        self.n_heads = n_heads
        self.concat = concat
        
        if concat:
            self.out_features = out_features
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
            self.out_features = out_features
        
        # 多个注意力头
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, self.n_hidden) 
            for _ in range(n_heads)
        ])
    
    def forward(self, h, adj_matrix):
        # 并行计算多个注意力头
        head_outputs = []
        attentions = []
        
        for head in self.attention_heads:
            output, attention = head(h, adj_matrix)
            head_outputs.append(output)
            attentions.append(attention)
        
        if self.concat:
            # 拼接多头输出
            output = torch.cat(head_outputs, dim=1)
        else:
            # 平均多头输出
            output = torch.mean(torch.stack(head_outputs), dim=0)
            
        return output, attentions

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, n_heads):
        super(GAT, self).__init__()
        
        # 第一层：多头注意力，输出拼接
        self.gat1 = MultiHeadGAT(
            in_features=in_features,
            out_features=hidden_features,
            n_heads=n_heads,
            concat=True
        )
        
        # 第二层：单头注意力，用于分类
        self.gat2 = GraphAttentionLayer(
            in_features=hidden_features,
            out_features=num_classes
        )
    
    def forward(self, x, adj_matrix):
        # 第一层 + ELU激活
        x, attention1 = self.gat1(x, adj_matrix)
        x = F.elu(x)
        
        # 第二层 + Softmax输出
        x, attention2 = self.gat2(x, adj_matrix)
        return F.log_softmax(x, dim=1), attention1, attention2

def create_synthetic_graph_data(num_nodes=100, num_features=10, num_classes=3):
    """创建合成图数据用于测试"""
    
    # 生成节点特征
    node_features = torch.randn(num_nodes, num_features)
    
    # 创建邻接矩阵（随机图）
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # 添加一些随机边
    num_edges = num_nodes * 2  # 平均每个节点2条边
    for _ in range(num_edges):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # 无向图
    
    # 添加自环
    adj_matrix += torch.eye(num_nodes)
    
    # 生成标签（基于节点特征的聚类）
    with torch.no_grad():
        # 使用前几个特征来确定标签
        cluster_features = node_features[:, :3]
        labels = torch.argmax(cluster_features, dim=1) % num_classes
    
    return node_features, adj_matrix, labels

def simple_accuracy(pred, target):
    """简单的准确率计算"""
    return (pred == target).float().mean().item()

def train_gat(model, node_features, adj_matrix, labels, train_mask, val_mask, epochs=100):
    """训练GAT模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        logits, _, _ = model(node_features, adj_matrix)
        
        # 计算训练损失
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(node_features, adj_matrix)
            val_pred = val_logits[val_mask].argmax(dim=1)
            val_acc = simple_accuracy(val_pred, labels[val_mask])
            
        train_losses.append(loss.item())
        val_accuracies.append(val_acc)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def analyze_attention_weights(attention_weights, adj_matrix, num_show=5):
    """分析注意力权重分布"""
    print("\n=== 注意力权重分析 ===")
    
    # 只考虑有边连接的节点对
    mask = adj_matrix > 0
    valid_weights = attention_weights[mask]
    
    print(f"有效注意力权重数量: {valid_weights.numel()}")
    print(f"注意力权重统计:")
    print(f"  平均值: {valid_weights.mean():.4f}")
    print(f"  标准差: {valid_weights.std():.4f}")
    print(f"  最小值: {valid_weights.min():.4f}")
    print(f"  最大值: {valid_weights.max():.4f}")
    
    # 显示几个节点的注意力分布
    print(f"\n前{num_show}个节点的注意力分布:")
    for i in range(min(num_show, attention_weights.size(0))):
        neighbors = adj_matrix[i] > 0
        if neighbors.sum() > 0:
            weights = attention_weights[i][neighbors]
            print(f"节点{i}: 邻居数={neighbors.sum()}, 权重范围=[{weights.min():.3f}, {weights.max():.3f}]")

def evaluate_model(model, node_features, adj_matrix, labels, test_mask):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        logits, attention1, attention2 = model(node_features, adj_matrix)
        pred = logits[test_mask].argmax(dim=1)
        
        # 计算准确率
        accuracy = simple_accuracy(pred, labels[test_mask])
        
        # 计算每个类别的性能
        num_classes = labels.max().item() + 1
        class_acc = []
        for c in range(num_classes):
            class_mask = labels[test_mask] == c
            if class_mask.sum() > 0:
                class_pred = pred[class_mask]
                class_target = labels[test_mask][class_mask]
                acc = simple_accuracy(class_pred, class_target)
                class_acc.append(acc)
            else:
                class_acc.append(0.0)
        
    return accuracy, class_acc, attention1, attention2

def compare_with_simple_baseline(node_features, adj_matrix, labels, train_mask, test_mask):
    """与简单基线模型比较"""
    print("\n=== 基线模型比较 ===")
    
    # 简单的多层感知机作为基线
    class SimpleMLP(nn.Module):
        def __init__(self, in_features, hidden_features, num_classes):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, num_classes)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.log_softmax(self.fc2(x), dim=1)
    
    # 训练基线模型
    baseline = SimpleMLP(node_features.size(1), 64, labels.max().item() + 1)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    for epoch in range(100):
        baseline.train()
        optimizer.zero_grad()
        logits = baseline(node_features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    
    # 评估基线模型
    baseline.eval()
    with torch.no_grad():
        baseline_logits = baseline(node_features)
        baseline_pred = baseline_logits[test_mask].argmax(dim=1)
        baseline_acc = simple_accuracy(baseline_pred, labels[test_mask])
    
    print(f"基线MLP准确率: {baseline_acc:.4f}")
    return baseline_acc

# 主测试代码
if __name__ == "__main__":
    print("=== GAT 效果测试 ===")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建合成数据
    num_nodes = 50  # 减少节点数以加快训练
    num_features = 16
    num_classes = 3
    
    node_features, adj_matrix, labels = create_synthetic_graph_data(
        num_nodes, num_features, num_classes
    )
    
    print(f"图数据: {num_nodes} 个节点, {num_features} 个特征, {num_classes} 个类别")
    print(f"边数: {(adj_matrix.sum() - num_nodes) // 2:.0f}")  # 减去对角线，除以2（无向图）
    
    # 创建训练/验证/测试掩码
    num_train = num_nodes // 2
    num_val = num_nodes // 4
    
    indices = torch.randperm(num_nodes)
    train_mask = indices[:num_train]
    val_mask = indices[num_train:num_train + num_val]
    test_mask = indices[num_train + num_val:]
    
    print(f"训练集: {len(train_mask)} 节点")
    print(f"验证集: {len(val_mask)} 节点") 
    print(f"测试集: {len(test_mask)} 节点")
    
    # 初始化模型
    model = GAT(
        in_features=num_features,
        hidden_features=64,
        num_classes=num_classes,
        n_heads=4  # 减少注意力头数
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n开始训练...")
    train_losses, val_accuracies = train_gat(
        model, node_features, adj_matrix, labels, train_mask, val_mask, epochs=100
    )
    
    # 评估模型
    print("\n评估模型...")
    test_acc, class_acc, attention1, attention2 = evaluate_model(
        model, node_features, adj_matrix, labels, test_mask
    )
    
    print(f"\n=== 最终结果 ===")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"各类别准确率: {[f'{acc:.3f}' for acc in class_acc]}")
    print(f"训练过程:")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  最终验证准确率: {val_accuracies[-1]:.4f}")
    print(f"  最佳验证准确率: {max(val_accuracies):.4f}")
    
    # 分析注意力权重
    if len(attention1) > 0:
        print(f"\n第一层注意力头数: {len(attention1)}")
        analyze_attention_weights(attention1[0], adj_matrix)  # 分析第一个注意力头
    
    print(f"\n第二层注意力权重:")
    analyze_attention_weights(attention2, adj_matrix)
    
    # 与基线模型比较
    baseline_acc = compare_with_simple_baseline(
        node_features, adj_matrix, labels, train_mask, test_mask
    )
    
    print(f"\n=== 性能对比 ===")
    print(f"GAT准确率:     {test_acc:.4f}")
    print(f"基线MLP准确率: {baseline_acc:.4f}")
    print(f"改进幅度:     {test_acc - baseline_acc:.4f} ({(test_acc/baseline_acc-1)*100:.1f}%)")
    
    # 检查注意力机制是否有效工作
    print(f"\n=== 注意力机制验证 ===")
    
    # 检查注意力权重是否遵循图结构
    model.eval()
    with torch.no_grad():
        _, _, attention2 = model(node_features, adj_matrix)
        
        # 计算有边连接和无边连接的平均注意力权重
        edge_mask = adj_matrix > 0
        no_edge_mask = adj_matrix == 0
        
        edge_weights = attention2[edge_mask]
        no_edge_weights = attention2[no_edge_mask]
        
        print(f"有边连接的平均注意力权重: {edge_weights.mean():.6f}")
        print(f"无边连接的平均注意力权重: {no_edge_weights.mean():.6f}")
        print(f"权重比率: {(edge_weights.mean() / no_edge_weights.mean()).item():.1f}x")
    
    print("\n=== GAT 效果测试完成 ===")