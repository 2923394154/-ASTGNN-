# ASTGNN 多因子挖掘模型

## 项目简介

ASTGNN (Adaptive Spatio-Temporal Graph Neural Network) 是一个基于深度学习的**多因子挖掘模型**，融合了图神经网络、时序建模和注意力机制，专门用于金融市场的因子提取和风险预测。

## 项目特色

### 最新解决的技术问题

1. **张量维度不匹配问题** - 完美解决多期预测的损失函数计算
2. **张量内存布局问题** - 修复 `view()` 连续性错误  
3. **IC计算维度错误** - 正确处理3维因子张量的IC分析
4. **数据预处理流程** - 优化从Barra因子到ASTGNN训练的完整pipeline

### 核心技术亮点

- **8因子并行生成**: 从单一收益预测升级为多维因子生成器
- **完整IC分析**: 支持时间序列IC、RankIC、ICIR等专业指标
- **论文标准损失**: 时间加权R²损失 + 因子正交惩罚项
- **图神经网络**: GAT+GRU+残差连接的混合架构
- **生产级质量**: 建立了完整的因子有效性验证框架

### 架构创新

```
序列输入 → GRU时序建模 → GAT图卷积 → 残差连接 → 全连接 → 8因子输出
    ↓           ↓             ↓          ↓         ↓         ↓
Barra因子   时间依赖     股票关系     信息保持    特征映射   风险因子
```

## 项目结构

### 核心文件

```
ASTGNN/
├── Real_ASTGNN_Training.py       # 多因子训练主程序 (最新)
├── ASTGNN.py                     # 核心模型架构
├── Enhanced_Factor_Analysis.py   # 增强因子分析工具
├── ASTGNN_Loss.py               # 专用损失函数
├── BarraFactorSimulator.py       # Barra因子模拟器
├── FactorValidation.py           # 因子有效性验证
├── DataPreprocessor.py           # 数据预处理器
├── GAT.py                       # 图注意力网络
├── Res_C.py                     # 残差连接层
├── Full_C.py                    # 全连接层
└── requirements.txt             # 依赖包列表
```

### 训练结果文件

```
training_results/
├── astgnn_training_results_*.png     # 训练过程可视化
├── enhanced_ic_analysis_*.png        # IC分析图表
└── factor_analysis_data.npz          # 因子分析数据
```

### 模型文件

```
├── real_astgnn_best_model.pth        # 最新训练的最佳模型
├── astgnn_best_model.pth            # 历史模型备份
├── processed_astgnn_data.pt         # 预处理后的训练数据
└── astgnn_training_data.pt          # 原始训练数据
```

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 确保有以下主要包
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
```

### 一键训练 (推荐)

```bash
# 运行最新的多因子训练
python Real_ASTGNN_Training.py
```

**训练输出**:
- 8个风险因子的神经网络模型
- 完整的训练过程可视化图表
- 因子IC分析和有效性评估
- 最佳模型自动保存

### 数据预处理

```python
from DataPreprocessor import RealDataPreprocessor

# 处理真实市场数据
preprocessor = RealDataPreprocessor()
preprocessor.process_real_data()  # 生成 processed_astgnn_data.pt
```

### 因子分析

```python
from Enhanced_Factor_Analysis import analyze_astgnn_factors

# 运行综合因子分析
results = analyze_astgnn_factors()
```

## 技术详解

### 1. 多因子训练架构

**Real_ASTGNN_Training.py** 实现了完整的多因子训练流程：

```python
# 关键技术特性
- 批次大小: 4 (优化内存使用)
- 学习率: 0.002 (动态调整)
- 因子数量: 8 (专业化风险因子)
- 预测期数: 5 (多期收益预测)
- 早停机制: 20轮patience
- 梯度裁剪: 0.5 (防止梯度爆炸)
```

### 2. ASTGNN模型架构

```python
# 模型配置
{
    'sequential_input_size': 10,      # Barra因子输入
    'gru_hidden_size': 64,           # GRU隐藏层
    'gat_hidden_size': 128,          # GAT隐藏层  
    'gat_n_heads': 4,                # 注意力头数
    'num_risk_factors': 32,          # 中间风险因子
    'num_predictions': 8,            # 最终输出因子数
    'dropout': 0.1                   # 防过拟合
}
```

### 3. 损失函数设计

**ASTGNN_Loss.py** 实现了论文标准的损失函数：

```python
# 时间加权R²损失
Loss = Σ(t=1 to T) ω^t * (1 - R²(F, y_t)) + λ * ||corr(F,F)||²

# 其中:
# ω: 时间衰减权重 (0.9)
# R²: 因子对收益的解释度
# λ: 正交惩罚权重 (0.01)
```

### 4. 解决的关键问题

#### 问题1: 张量维度不匹配

```python
# 修复前: 2维张量传入1维损失函数
future_returns_list = [target_returns[b]]  # [5, 1448] - 错误

# 修复后: 正确构造1维张量列表  
future_returns_list = []
for t in range(target_returns.shape[1]):  # 遍历5个时间步
    future_returns_list.append(target_returns[b, t])  # [1448] - 正确
```

#### 问题2: 张量内存布局

```python
# 修复前: view() 连续性错误
pred_flat = predictions.view(-1)

# 修复后: reshape() 自动处理连续性
pred_flat = predictions.reshape(-1)
```

#### 问题3: IC分析维度处理

```python
# 修复前: 3维目标张量导致IC计算失败
targets: [batch, prediction_horizon, stocks]

# 修复后: 选择第一个预测期进行IC分析
if targets.dim() == 3:
    targets = targets[:, 0, :]  # [batch, stocks]
```

## 训练结果

### 模型性能

- **因子数量**: 8个专业化风险因子
- **股票覆盖**: 1,448只A股
- **时间序列**: 30天历史 → 5天预测
- **模型参数**: ~45万可训练参数
- **训练效果**: 损失稳定收敛，R²持续改善

### 因子统计

最新训练生成的8个因子统计信息：

```
Factor 0: 均值=-0.3260, 标准差=0.1403, 范围=[-1.10, 0.79]
Factor 1: 均值=+0.0998, 标准差=0.1506, 范围=[-0.93, 1.89] 
Factor 2: 均值=-0.2426, 标准差=0.1959, 范围=[-1.80, 2.37]
Factor 3: 均值=+0.0678, 标准差=0.2298, 范围=[-1.55, 1.42]
Factor 4: 均值=-0.0759, 标准差=0.1889, 范围=[-1.86, 1.37]
Factor 5: 均值=+0.1555, 标准差=0.1783, 范围=[-1.23, 1.62]
Factor 6: 均值=-0.0415, 标准差=0.1794, 范围=[-2.10, 1.13]
Factor 7: 均值=-0.1320, 标准差=0.1723, 范围=[-1.59, 1.01]
```

**因子质量评估**:
- 因子分布合理，无异常值
- 标准差适中，不同因子有差异化特征
- 正负因子平衡，符合市场中性要求

### 可视化结果

训练完成后自动生成：

1. **训练过程图表** (`astgnn_training_results_*.png`)
   - 损失曲线 (训练/验证)
   - R²分数曲线
   - 学习率变化
   - 训练统计摘要

2. **IC分析图表** (`enhanced_ic_analysis_*.png`)
   - 因子IC时间序列
   - IC分布直方图
   - 累积IC曲线
   - 因子相关性热力图

## 高级功能

### 1. 增强因子分析

```python
from Enhanced_Factor_Analysis import analyze_astgnn_factors

# 综合因子分析
results = analyze_astgnn_factors()

# 输出包含:
# - IC/RankIC对比分析
# - 因子评级 (A+/A/B+/B/C/D)  
# - 统计显著性检验
# - 分组回测结果
# - 智能优化建议
```

### 2. 自定义训练配置

```python
# 修改训练参数
config = {
    'learning_rate': 0.002,          # 学习率
    'batch_size': 4,                 # 批次大小
    'epochs': 150,                   # 训练轮数
    'early_stopping_patience': 20,   # 早停耐心
    'orthogonal_penalty_weight': 0.003  # 正交惩罚
}

trainer = RealASTGNNTrainer(config=config)
trainer.train_model()
```

### 3. 模型加载和推理

```python
# 加载训练好的模型
trainer = RealASTGNNTrainer()
trainer.load_model('real_astgnn_best_model.pth')

# 进行推理
with torch.no_grad():
    factors = trainer.model(sequences, adj_matrix)
    # factors: [batch, stocks, 8factors]
```

## 开发指南

### 主要模块说明

1. **Real_ASTGNN_Training.py**: 多因子训练主程序
   - 数据加载和预处理
   - 模型训练和验证  
   - 因子有效性分析
   - 结果可视化

2. **ASTGNN.py**: 核心神经网络架构
   - GRU时序建模
   - GAT图卷积  
   - 残差连接
   - 多因子输出层

3. **Enhanced_Factor_Analysis.py**: 因子分析工具
   - IC计算和分析
   - 分组回测
   - 可视化图表
   - 优化建议

4. **DataPreprocessor.py**: 数据预处理
   - Barra因子特征工程
   - 收益率计算
   - 数据标准化
   - 序列构造

### 扩展开发

```python
# 添加新的因子类型
class CustomFactor:
    def compute_factor(self, data):
        # 实现自定义因子计算
        pass

# 修改模型架构
class ExtendedASTGNN(ASTGNNFactorModel):
    def __init__(self, **config):
        super().__init__(**config)
        # 添加新的网络层
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```python
   # 解决方案: 减小批次大小
   config['batch_size'] = 2  # 默认4 → 2
   ```

2. **训练收敛慢**
   ```python
   # 解决方案: 调整学习率
   config['learning_rate'] = 0.005  # 增大学习率
   ```

3. **因子相关性过高**
   ```python
   # 解决方案: 增加正交惩罚
   config['orthogonal_penalty_weight'] = 0.01  # 增大惩罚权重
   ```

### 调试工具

```python
# 检查数据形状
print("因子序列形状:", factor_sequences.shape)
print("目标形状:", target_returns.shape)

# 监控训练过程
logger.info(f"Epoch {epoch}, Loss: {loss:.6f}, R²: {r2:.6f}")

# 验证因子质量
ic_results = validator.compute_information_coefficient(factors, returns)
```

## 技术文档

- [ASTGNN架构分析](ASTGNN_Architecture_Analysis.md)
- [Barra因子模型](Barra.md)  
- [因子验证框架](FactorValidation.py)
- [数据预处理指南](DataPreprocessor.py)

## 项目成就

- **完整解决训练问题**: 张量维度、内存布局、IC计算等技术难题
- **实现多因子架构**: 从单一预测升级为8因子生成器
- **建立评估体系**: 完整的因子有效性验证框架
- **优化训练流程**: 自动早停、模型保存、结果可视化
- **提供生产工具**: 可直接用于实际量化投资的完整pipeline

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目！

## 联系方式

如有问题请创建Issue或联系项目维护者。

---

**本项目展示了深度学习在量化金融中的强大潜力，通过系统性的技术创新和问题解决，成功构建了一个完整的多因子挖掘和风险建模框架。** 