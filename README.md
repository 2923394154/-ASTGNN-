# ASTGNN: 自适应时空图神经网络因子挖掘模型

## 项目简介

ASTGNN (Adaptive Spatio-Temporal Graph Neural Network) 是一个基于深度学习的量化因子挖掘模型，融合了图神经网络、时序建模和注意力机制，专门用于金融市场的因子提取和预测。

## 核心特性

- **W2-GAT架构**: 实现序列输入 → GRU → GAT → Res-C → Full-C → NN-Layer的完整流程
- **双模式图卷积**: 支持加法模式（动量效应）和减法模式（中性化效应）
- **专用损失函数**: 时间加权R²损失 + 正交惩罚项
- **完整训练框架**: 支持早停、模型保存、因子有效性验证
- **可视化分析**: 提供训练过程和因子相关性分析

## 模型架构

```
序列输入(x1,x2,...,xT) → GRU → GAT → Res-C → Full-C → NN-Layer → 输出
```

### 关键组件

- **GRU层**: 处理时间序列特征
- **GAT层**: 处理股票间的图关系和注意力机制
- **Res-C层**: 残差连接保持信息传递
- **Full-C层**: 全连接层进行特征映射
- **NN-Layer**: 额外的神经网络处理

## 项目结构

```
ASTGNN/
├── ASTGNN.py                      # 核心模型架构
├── ASTGNN_Training.py             # 训练框架
├── ASTGNN_Demo.py                 # 演示和测试脚本
├── ASTGNN_Loss.py                 # 专用损失函数
├── BarraFactorSimulator.py        # Barra因子模拟器
├── FactorValidation.py            # 因子有效性验证
├── GAT.py                         # 图注意力网络
├── Res_C.py                       # 残差连接层
├── Full_C.py                      # 全连接层
├── ASTGNN_Architecture_Analysis.md # 架构分析文档
└── Barra.md                       # Barra模型说明
```

## 快速开始

### 环境要求

```python
torch >= 1.9.0
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scikit-learn >= 0.24.0
```

### 安装依赖

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

### 运行演示

```bash
# 运行完整演示
python ASTGNN_Demo.py

# 运行训练流程
python ASTGNN_Training.py
```

## 使用方法

### 1. 数据准备

```python
# 准备时间序列因子数据
factors = torch.randn(periods, stocks, factors)  # [时期, 股票, 因子]
returns = torch.randn(periods, stocks)           # [时期, 股票]
adj_matrices = torch.randn(periods, stocks, stocks)  # [时期, 股票, 股票]
```

### 2. 模型训练

```python
from ASTGNN_Training import main_training_pipeline

# 运行完整训练流程
main_training_pipeline()
```

### 3. 模型推理

```python
from ASTGNN import ASTGNN

# 加载训练好的模型
model = ASTGNN(input_dim=20, hidden_dim=64, num_stocks=100)
model.load_state_dict(torch.load('astgnn_best_model.pth'))

# 进行预测
predictions, risk_factors, attention_weights, intermediate_outputs = model(factors, adj_matrices)
```

## 模型特点

### 损失函数

- **R²损失**: 衡量因子重构精度
- **正交惩罚**: 确保提取的因子相互独立
- **时间加权**: 对近期数据给予更高权重

### 图卷积模式

1. **加法模式**: `Z = (I + softmax(ReLU(MM^T)))XW`
   - 强化相似股票间的信息传递
   - 体现动量效应

2. **减法模式**: `Z = (I - softmax(ReLU(MM^T)))XW`
   - 中性化相似股票的影响
   - 提取独特性特征

## 技术创新

1. **严格按照论文图实现W2-GAT架构**
2. **实现图模型的加法和减法两种模式**
3. **集成专用损失函数**：时间加权R²损失 + 正交惩罚
4. **完整的训练框架**：支持早停、模型保存、因子有效性验证
5. **端到端可训练**：从数据处理到模型训练、验证、可视化的完整流程

## 输出结果

- **因子IC值**: 信息系数，衡量因子预测能力
- **因子相关性矩阵**: 展示提取因子间的相关关系
- **训练历史曲线**: 展示损失函数收敛过程
- **注意力权重**: 可视化股票间的关系强度

## 已知问题和解决方案

### 问题1: 激活函数不兼容
- **问题**: `ValueError: 不支持的激活函数: linear`
- **解决**: 将`activation='linear'`改为`activation=None`

### 问题2: 返回值数量不匹配
- **问题**: `ValueError: too many values to unpack`
- **解决**: 确保forward方法返回值与接收端匹配

### 问题3: 中文字体乱码
- **解决**: 设置matplotlib中文字体支持

## 参考文献

- 东方证券研究报告：融合基本面信息的ASTGNN因子挖掘模型
- Graph Attention Networks (GAT)
- Barra风险模型

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题请创建Issue或联系项目维护者。 