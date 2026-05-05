# 神经网络框架（Neural Network Framework）

这是一个基于 NumPy 的轻量级深度学习框架，适合学习、实验和小规模研究。
实现了自动求导、常用算子、网络模块、优化器、数据加载与完整训练管线。

## 已实现功能

### 核心计算
- `Tensor` 计算图与反向传播，支持 `no_grad()` 上下文
- 运算符重载：`+` `-` `*` `/` `@` `-x` `x ** n`
- 核心算子：`add` `mul` `div` `matmul` `reshape` `transpose` `broadcast` `sum` `max`

### 激活函数
`ReLU` `Sigmoid` `Tanh` `LeakyReLU` `ELU` `GELU`

### 层与模块
- 基础：`Linear` `Conv2d` `Dropout` `Flatten` `Sequential`
- 池化：`MaxPool2d` `AvgPool2d`
- 归一化：`LayerNorm` `BatchNorm1d` `BatchNorm2d`
- 注意力与 Transformer：`MultiHeadSelfAttention` `TransformerEncoderBlock` `SEBlock`
- 循环网络：`RNN` `LSTM` `GRU`

### 损失函数
`softmax_cross_entropy` `mse_loss` `bce_loss`

### 优化器
`SGD` `Adam`（均支持 `weight_decay`）

### 学习率调度
`StepLR` `CosineAnnealingLR` `MultiStepLR` `LinearWarmup` `SequentialLR`

### 训练工具
- `Trainer`（支持梯度累积 `accumulation_steps`、Early Stopping）
- `split_train_val` 数据集切分
- 梯度裁剪（`clip_grad_norm` / `clip_grad_value`）与梯度统计
- `ProgressBar` 进度条、`TrainingLogger` 日志
- `TrainingConfig` 配置类

### 模型管理
- `state_dict()` / `load_state_dict()` 参数读写
- `save_checkpoint()` / `load_checkpoint()` 完整检查点
- `save_model()` / `load_model()` 仅模型参数
- `model.zero_grad()` / `model.train()` / `model.eval()`
- `model.__repr__()` 打印模型结构

### 数据管道
- `Dataset` / `DataLoader`（支持 shuffle、seed、drop_last、transforms）
- 数据增强：`Normalize` `RandomHorizontalFlip` `RandomCrop` `Resize` `Compose`

## 快速开始

1. 安装依赖：
   ```
   pip install numpy pytest
   ```
2. 运行测试：
   ```
   pytest -q
   ```
3. 运行训练脚本：
   ```
   python test_mnist.py
   python fashion_mnist.py
   python cifar10.py
   python attention_experiment.py
   ```

## CIFAR-10 可选模型

```
python cifar10.py --model mlp
python cifar10.py --model cnn
python cifar10.py --model cnn_se
python cifar10.py --model cnn_attn
python cifar10.py --model cnn_transformer
python cifar10.py --model vit
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `tensor.py` | Tensor、计算图节点、反向传播入口 |
| `autograd.py` | 拓扑排序与梯度传播 |
| `ops.py` | 所有算子实现与梯度（含 Pooling、BatchNorm、Loss） |
| `nn.py` | 网络模块（Linear、Conv、Pool、Norm、Attention、RNN/LSTM/GRU） |
| `optimizers.py` | SGD / Adam 优化器（支持 weight_decay） |
| `data.py` | Dataset / DataLoader |
| `trainer.py` | 训练循环、EarlyStopping、梯度累积 |
| `lr_scheduler.py` | StepLR、CosineAnnealingLR、MultiStepLR、LinearWarmup、SequentialLR |
| `grad_utils.py` | 梯度裁剪与梯度统计 |
| `checkpoint.py` | 模型和训练 checkpoint 保存/加载 |
| `config.py` | TrainingConfig 配置类 |
| `transforms.py` | 数据增强 |
| `progress.py` | 进度条与训练日志 |
| `logger.py` | 日志工具 |
| `init.py` | He / Xavier 权重初始化 |
| `tests/` | 完整测试套件 |
| `test_mnist.py` | MNIST 训练脚本 |
| `fashion_mnist.py` | Fashion-MNIST 训练脚本 |
| `cifar10.py` | CIFAR-10 训练脚本 |
| `attention_experiment.py` | 注意力对比实验脚本 |

## 说明

- 训练脚本会在需要时自动下载数据集。
- 数据默认存放在 `mnist_data/` `fashion_mnist_data/` `cifar10_data/`。
- 可用 `python attention_experiment.py` 快速复现注意力机制对比实验。

## 后续可扩展方向

- `RMSprop` 优化器
- `Conv1d` / `ConvTranspose2d` 层
- 更丰富的池化（`AdaptiveAvgPool2d`）
- 分布式训练 / 混合精度
- ONNX 导出
- TensorBoard 集成
