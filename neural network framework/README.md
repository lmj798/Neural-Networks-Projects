# 神经网络框架（Neural Network Framework）

这是一个基于 NumPy 的轻量级深度学习框架，适合学习、实验和小规模研究。
当前包含自动求导、常用算子、网络模块、优化器、数据加载与训练脚本。

## 已实现功能

- `Tensor` 计算图与反向传播
- 核心算子：`add`、`mul`、`div`、`matmul`、`reshape`、`transpose`、`broadcast`、`sum`、`max`
- 激活函数：`ReLU`、`Sigmoid`、`Tanh`、`LeakyReLU`、`ELU`、`GELU`
- 损失：`softmax_cross_entropy`
- 基础模块：`Linear`、`Conv2d`、`Dropout`、`Flatten`、`Sequential`
- 注意力与 Transformer 相关模块：
  - `MultiHeadSelfAttention`
  - `LayerNorm`
  - `TransformerEncoderBlock`
  - `SEBlock`（更适合图像任务的通道注意力）
- 优化器：`SGD`、`Adam`
- 数据管道：`Dataset`、`DataLoader`
- 训练工具：`Trainer`、`split_train_val`、梯度裁剪、学习率调度器
- 模型管理：`state_dict()` / `load_state_dict()`、checkpoint 保存与恢复
- 数据增强：`Normalize`、`RandomHorizontalFlip`、`RandomCrop`、`Resize`、`Compose`

## 快速开始

1. 安装依赖：
   - `pip install numpy pytest`
2. 运行测试：
   - `pytest -q`
3. 运行训练脚本：
   - `python test_mnist.py --help`
   - `python test_mnist.py`
   - `python fashion_mnist.py`
   - `python cifar10.py`
   - `python attention_experiment.py`

## CIFAR-10 可选模型

- `python cifar10.py --model mlp`
- `python cifar10.py --model cnn`
- `python cifar10.py --model cnn_se`
- `python cifar10.py --model cnn_attn`
- `python cifar10.py --model cnn_transformer`
- `python cifar10.py --model vit`

说明：
- `cnn_se` 是 CNN + SE 注意力，通常比直接上纯 Transformer 更稳。
- `vit` 是不使用 CNN 的原生 Transformer（patch embedding + transformer blocks）。

## 项目结构

- `tensor.py`：Tensor、计算图节点、反向传播入口
- `autograd.py`：拓扑排序与梯度传播
- `ops.py`：算子实现与梯度
- `nn.py`：网络模块与层实现
- `optimizers.py`：优化器
- `data.py`：`Dataset` / `DataLoader`
- `trainer.py`：训练循环与训练/验证集切分
- `lr_scheduler.py`：`StepLR`、`CosineAnnealingLR`、`MultiStepLR`
- `grad_utils.py`：梯度裁剪与梯度统计
- `checkpoint.py`：模型和训练 checkpoint 保存/加载
- `transforms.py`：基础数据增强
- `tests/`：核心、回归、注意力、Transformer 相关测试
- `test_mnist.py`：MNIST 训练脚本
- `fashion_mnist.py`：Fashion-MNIST 训练脚本
- `cifar10.py`：CIFAR-10 训练脚本
- `attention_experiment.py`：注意力对比实验脚本

## 说明

- 训练脚本会在需要时自动下载数据集。
- 数据默认存放在：
  - `mnist_data/`
  - `fashion_mnist_data/`
  - `cifar10_data/`
- 可用 `python attention_experiment.py` 快速复现“带/不带注意力”的合成任务对比。

## 后续可扩展方向

1. 损失函数
- `mse_loss`
- `nll_loss`

2. 视觉算子
- `max_pool2d`、`avg_pool2d`

3. 工程能力
- 数值梯度检查工具
- 更规范的包结构（例如 `src/` 布局）
- lint/format/type-check 工具链

4. 优化器
- `RMSprop`
- 带 momentum / weight decay 的优化器变体
