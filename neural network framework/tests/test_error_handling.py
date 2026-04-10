import numpy as np
import pytest

from tensor import Tensor
from nn import Linear, Sequential, ReLUModule
from optimizers import SGD, Adam
from ops import MatMul


def test_tensor_creation_errors():
    """测试张量创建的错误处理"""
    # 测试 None 输入
    with pytest.raises(ValueError, match="array cannot be None"):
        Tensor(None)
    
    # 测试无效输入
    with pytest.raises(ValueError):
        Tensor("invalid")


def test_linear_layer_errors():
    """测试线性层的错误处理"""
    # 测试无效的输入特征
    with pytest.raises(ValueError, match="in_features must be a positive integer"):
        Linear(0, 10)
    
    # 测试无效的输出特征
    with pytest.raises(ValueError, match="out_features must be a positive integer"):
        Linear(10, 0)
    
    # 测试无效的 bias 参数
    with pytest.raises(ValueError, match="bias must be a boolean"):
        Linear(10, 10, bias="True")
    
    # 测试输入维度错误
    linear = Linear(10, 5)
    with pytest.raises(ValueError, match="Expected at least 2D input"):
        linear(Tensor([1.0]))
    
    # 测试输入特征不匹配
    with pytest.raises(ValueError, match="Input features mismatch"):
        linear(Tensor(np.random.randn(2, 8)))


def test_optimizer_errors():
    """测试优化器的错误处理"""
    # 测试 SGD 学习率错误
    linear = Linear(10, 5)
    with pytest.raises(ValueError, match="Learning rate must be a positive number"):
        SGD(linear.parameters(), lr=-0.1)
    
    # 测试 Adam 学习率错误
    with pytest.raises(ValueError, match="Learning rate must be a positive number"):
        Adam(linear.parameters(), lr=-0.1)
    
    # 测试 Adam betas 错误
    with pytest.raises(ValueError, match="betas must be a list or tuple of length 2"):
        Adam(linear.parameters(), betas=[0.9])
    
    with pytest.raises(ValueError, match=r"betas must be in \(0, 1\)"):
        Adam(linear.parameters(), betas=[1.0, 0.999])
    
    # 测试 Adam eps 错误
    with pytest.raises(ValueError, match="eps must be a positive number"):
        Adam(linear.parameters(), eps=-1e-8)


def test_matmul_errors():
    """测试矩阵乘法的错误处理"""
    # 测试维度不足
    matmul = MatMul()
    with pytest.raises(ValueError, match="MatMul requires tensors with at least 2 dimensions"):
        matmul(Tensor([1.0]), Tensor([2.0]))
    
    # 测试维度不匹配
    with pytest.raises(ValueError, match="MatMul dimensions mismatch"):
        matmul(Tensor(np.random.randn(2, 3)), Tensor(np.random.randn(4, 5)))


def test_sequential_errors():
    """测试顺序模块的错误处理"""
    # 测试空模块列表
    with pytest.raises(ValueError, match="Sequential module cannot be empty"):
        Sequential()
