# Neural Network Framework

基于 NumPy 的教学/实验级神经网络框架，包含自动求导、常见算子、基础模块与训练脚本。

**Features**
- Tensor 与自动求导
- 常见算子与激活函数（加/乘/矩阵乘、broadcast、reshape、sum/max、ReLU/Sigmoid/Tanh/LeakyReLU/ELU）
- 基础模块（Linear、Conv2d、Dropout、Flatten、Sequential）
- Adam 优化器
- Dataset 与 DataLoader
- MNIST / Fashion-MNIST / CIFAR-10 训练示例与基准脚本
- pytest 测试用例

**Quickstart**
1. `pip install numpy pytest`
1. `python test_mnist.py --help`
1. `python test_mnist.py`
1. `python fashion_mnist.py`
1. `python fashion_mnist_benchmark.py`
1. `python cifar10.py`

**Tests**
- `pytest -q`

**Notes**
- 训练脚本会自动下载数据集到 `mnist_data/`、`fashion_mnist_data/`、`cifar10_data/`。
- 这些数据目录与生成的权重文件已加入 `.gitignore`。

**Project Structure**
- `tensor.py`：Tensor 核心与自动求导接口
- `autograd.py`：反向传播与拓扑排序
- `ops.py`：算子与梯度
- `nn.py`：网络模块与层
- `optimizers.py`：优化器
- `data.py`：Dataset 与 DataLoader
- `test_mnist.py`：MNIST 训练脚本与参数入口
- `fashion_mnist.py`：Fashion-MNIST 训练与组件测试
- `fashion_mnist_benchmark.py`：Fashion-MNIST 基准/激活函数对比
- `cifar10.py`：CIFAR-10 训练脚本
