from typing import Any, List, Optional, Tuple, Dict
import numpy

NDArray = numpy.ndarray
TENSOR_COUNTER = 0
LAZY_MODE = False

class ABC:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()
    
    def gradient(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

class Op(ABC):
    """Operator definition."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def compute(self, *args: Tuple["NDArray"])->NDArray:
        raise NotImplementedError()

    def gradient(self, out_grad, node):
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad, node):
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class Value:
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def _init(self, op: Optional[Op], inputs: List["Value"],  *, num_outputs: int=1, cached_data: NDArray = None,
        requires_grad: Optional[bool] = None
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.get_outputs()
        return value

class Tensor(Value):
    grad: "Tensor"
    def __init__(self, array, *, dtype="float32", requires_grad=True, **kwargs):
        array_data = numpy.array(array, dtype=dtype)
        self._init(
            None,
            [],
            cached_data=array_data,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def from_numpy(numpy_array, dtype):
        return numpy.array(numpy_array, dtype=dtype)
    
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor
    
    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor_data = data.realize_cached_data()
        else:
            tensor_data = data
        tensor._init(None, [], cached_data=tensor_data, requires_grad=requires_grad)
        return tensor
    
    @property
    def data(self):
        return self.cached_data
    
    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @data.setter
    def data(self, value):
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self):
        return self.realize_cached_data().shape
    
    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    def backward(self, out_grad=None):
        if out_grad:
            out_grad = out_grad
        else:
            out_grad = Tensor(numpy.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)
        
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def sum(self, axis=None):
        return Summation(axis)(self)

    def max(self, axis=None):
        return Max(axis)(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    def argmax(self, axis=None):
        # argmax returns indices, not differentiable
        if axis is None:
            return numpy.argmax(self.realize_cached_data())
        return numpy.argmax(self.realize_cached_data(), axis=axis)

    def item(self):
        return self.realize_cached_data().item()


def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor]))) # 请自行实现拓扑排序函数
    for node in reverse_topo_order:
        # 求node的partial adjoint之和，存入属性grad
        grad_list = node_to_output_grads_list[node]
        if len(grad_list) == 1:
            node.grad = grad_list[0]
        else:
            # 累加多个梯度
            grad_sum = grad_list[0]
            for g in grad_list[1:]:
                grad_sum = grad_sum + g
            node.grad = grad_sum

        if node.is_leaf():
            continue

        # 计算node.inputs的partial adjoint
        input_grads = node.op.gradient_as_tuple(node.grad, node)
        for i, grad in enumerate(input_grads):
            input_node = node.inputs[i]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(grad)

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = set()
    topo_order = []
    topo_sort_dfs(node_list[-1], visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for pre_node in node.inputs:
        topo_sort_dfs(pre_node, visited, topo_order)
    topo_order.append(node)

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a+b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad
        grad_b = out_grad

        # 如果输入形状与输出形状不同，需要对多余维度求和
        # 处理broadcast情况
        if a.shape != out_grad.shape:
            # 对被broadcast的维度求和
            ndims_added = len(out_grad.shape) - len(a.shape)
            axes_to_sum = list(range(ndims_added))
            for i, (in_dim, out_dim) in enumerate(zip(a.shape, out_grad.shape[ndims_added:])):
                if in_dim == 1 and out_dim > 1:
                    axes_to_sum.append(i + ndims_added)
            if axes_to_sum:
                grad_a = out_grad.sum(tuple(axes_to_sum)).reshape(a.shape)

        if b.shape != out_grad.shape:
            ndims_added = len(out_grad.shape) - len(b.shape)
            axes_to_sum = list(range(ndims_added))
            for i, (in_dim, out_dim) in enumerate(zip(b.shape, out_grad.shape[ndims_added:])):
                if in_dim == 1 and out_dim > 1:
                    axes_to_sum.append(i + ndims_added)
            if axes_to_sum:
                grad_b = out_grad.sum(tuple(axes_to_sum)).reshape(b.shape)

        return grad_a, grad_b
    
class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    
    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad, )
    
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad * b, out_grad * a
    
class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)
    
class Negate(TensorOp):
    def compute(self, a):
        return numpy.negative(a)

    def gradient(self, out_grad, node):
        return MulScalar(-1)(out_grad)

class EWiseDiv(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad / self.scalar,)

class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad @ b.transpose(), a.transpose() @ out_grad

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return numpy.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        # 需要处理维度不匹配的情况
        # 例如: (1, 64) broadcast 到 (32, 64)
        # 或者: (64,) broadcast 到 (32, 64)

        # 首先对新增的维度求和
        ndims_added = len(output_shape) - len(input_shape)
        axes_to_sum = list(range(ndims_added))

        # 然后找出被广播的维度(维度为1但被扩展的)
        for i, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape[ndims_added:])):
            if in_dim == 1 and out_dim > 1:
                axes_to_sum.append(i + ndims_added)

        if axes_to_sum:
            grad = out_grad.sum(tuple(axes_to_sum))
        else:
            grad = out_grad

        # 最后reshape回输入形状
        if grad.shape != input_shape:
            grad = grad.reshape(input_shape)

        return grad

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return numpy.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad.reshape(node.inputs[0].shape)

class Summation(TensorOp):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, a: NDArray):
        return numpy.sum(a, axis=self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        if self.axis is None:
            return out_grad.broadcast_to(input_shape)
        else:
            shape = list(input_shape)
            if isinstance(self.axis, int):
                shape[self.axis] = 1
            else:
                for ax in self.axis:
                    shape[ax] = 1
            return out_grad.reshape(tuple(shape)).broadcast_to(input_shape)

class Max(TensorOp):
    def __init__(self, axis=None):
        self.axis = axis

    def compute(self, a: NDArray):
        return numpy.max(a, axis=self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        max_val = self.compute(a.realize_cached_data())
        if self.axis is None:
            mask = Tensor(a.realize_cached_data() == max_val)
            return out_grad.broadcast_to(a.shape) * mask
        else:
            shape = list(a.shape)
            if isinstance(self.axis, int):
                shape[self.axis] = 1
            else:
                for ax in self.axis:
                    shape[ax] = 1
            mask = Tensor(a.realize_cached_data() == numpy.reshape(max_val, shape))
            return out_grad.reshape(tuple(shape)).broadcast_to(a.shape) * mask

class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a: NDArray):
        if self.axes is None:
            return numpy.transpose(a)
        return numpy.transpose(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axes is None:
            return out_grad.transpose()
        # Inverse permutation
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i
        return out_grad.transpose(tuple(inv_axes))

from abc import abstractmethod

class ABC():
    pass

class Optimizer(ABC):
    def __init__(self, params):
        self.params = params
    
    @abstractmethod
    def step(self):
        pass

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def zero_grad(self):
        """清零所有参数的梯度"""
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr = 0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                # 获取梯度数据
                if isinstance(param.grad, Tensor):
                    grad_data = param.grad.realize_cached_data()
                else:
                    grad_data = param.grad

                # 更新参数: param = param - lr * grad
                param_data = param.realize_cached_data()
                new_data = param_data - self.lr * grad_data

                # 使用cached_data直接更新，避免创建新的计算图
                param.cached_data = new_data

class Adam(Optimizer):
    """Adam优化器 - 自适应学习率，效果比SGD更好"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0  # 时间步

        # 初始化动量
        self.m = []  # 一阶矩估计 (momentum)
        self.v = []  # 二阶矩估计 (velocity)
        for param in self.params:
            param_data = param.realize_cached_data()
            self.m.append(numpy.zeros_like(param_data))
            self.v.append(numpy.zeros_like(param_data))

    def step(self):
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is not None:
                # 获取梯度
                if isinstance(param.grad, Tensor):
                    grad_data = param.grad.realize_cached_data()
                else:
                    grad_data = param.grad

                # 更新一阶矩估计 (带动量的梯度)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_data

                # 更新二阶矩估计 (梯度平方的移动平均)
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_data ** 2)

                # 偏差修正
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 更新参数
                param_data = param.realize_cached_data()
                new_data = param_data - self.lr * m_hat / (numpy.sqrt(v_hat) + self.eps)

                param.cached_data = new_data

class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0)


def relu(a):
    return ReLU()(a)

import math
import numpy as np

def randn(*shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
    array = numpy.random.randn(*shape) * std + mean
    return array

def init_He(in_features, out_features):
    # Xavier/Glorot initialization for better stability
    # std = sqrt(2 / (fan_in + fan_out))
    std = numpy.sqrt(2.0 / (in_features + out_features))
    s = numpy.random.normal(0, std, (in_features, out_features))
    return s

def init_Xavier(in_features, out_features, dtype):
    v = math.sqrt(2/(in_features+out_features))
    return randn(in_features, out_features,std=v, dtype=dtype)


from abc import abstractmethod
from functools import reduce
from typing import List, Any

class ABC:
    pass

class Module(ABC):
    def __init__(self):
        self.training = True
    
    def parameters(self)->List["Tensor"]:
        return _unpack_params(self.__dict__)
    
    def _children(self):
    # 遍历当前对象属性，找出子模块（例如是 Module 类型的实例）
        children = []
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                children.append(attr)
        return children

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True


class Parameter(Tensor):
    pass

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []
    
def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        s = init_He(in_features, out_features)
        self.weight = Parameter(s, dtype="float32")
        if bias:
            self.bias = Parameter(numpy.zeros(self.out_features), dtype="float32")
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_out = X @ self.weight
        if self.bias is not None:
            # 直接相加，EWiseAdd会处理broadcast
            return X_out + self.bias
        return X_out
    
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor)->Tensor:
        for module in self.modules:
            x = module(x)
        return x
    
class ReLUModule(Module):
    def forward(self, x: Tensor)->Tensor:
        return relu(x)

class Flatten(Module):
    def forward(self, X):
        size = reduce(lambda a, b: a * b, X.shape)
        return X.reshape((X.shape[0], size // X.shape[0]))

class Dataset():
    def __init__(self, X, y, transforms: Optional[List] = None):
        self.transforms = transforms        
        self.X = X
        self.y = y

    def __getitem__(self, index) -> object:
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
        return (imgs, labels)

    def __len__(self) -> int:
        return self.X.shape[0]
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x

class DataLoader:
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False, device=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = -1
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        batch_indices = self.ordering[self.idx]
        return tuple([Tensor(x, device = self.device) for x in self.dataset[batch_indices]])



