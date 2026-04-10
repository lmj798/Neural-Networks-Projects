from contextlib import contextmanager
from typing import Any, List, Optional, Tuple
import numpy

NDArray = numpy.ndarray


class Config:
    """框架配置管理类"""
    def __init__(self):
        self._grad_mode = True
        self._lazy_mode = False
        self._tensor_counter = 0

    @property
    def grad_mode(self) -> bool:
        return self._grad_mode

    @grad_mode.setter
    def grad_mode(self, value: bool):
        self._grad_mode = value

    @property
    def lazy_mode(self) -> bool:
        return self._lazy_mode

    @lazy_mode.setter
    def lazy_mode(self, value: bool):
        self._lazy_mode = value

    @property
    def tensor_counter(self) -> int:
        return self._tensor_counter

    def increment_tensor_counter(self) -> int:
        self._tensor_counter += 1
        return self._tensor_counter


# 全局配置实例
config = Config()


def is_grad_enabled() -> bool:
    return config.grad_mode


@contextmanager
def no_grad():
    previous = config.grad_mode
    config.grad_mode = False
    try:
        yield
    finally:
        config.grad_mode = previous

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
    """计算图的基本节点类"""
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """计算并缓存数据"""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def is_leaf(self):
        """判断是否为叶子节点"""
        return self.op is None

    def _init(self, op: Optional[Op], inputs: List["Value"],  *, num_outputs: int=1, cached_data: NDArray = None,
        requires_grad: Optional[bool] = None
    ):
        """初始化节点
        
        Args:
            op: 操作符
            inputs: 输入节点列表
            num_outputs: 输出数量
            cached_data: 缓存的数据
            requires_grad: 是否需要计算梯度
        """
        if requires_grad is None:
            requires_grad = is_grad_enabled() and any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        """创建常量节点
        
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
        
        Returns:
            Value: 常量节点
        """
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
        """从操作创建节点
        
        Args:
            op: 操作符
            inputs: 输入节点列表
        
        Returns:
            Value: 新创建的节点
        """
        value = cls.__new__(cls)
        value._init(op, inputs)
        if not config.lazy_mode:
            value.realize_cached_data()
        return value

class Tensor(Value):
    """张量类，框架的核心数据结构"""
    grad: "Tensor"
    
    def __init__(self, array, *, dtype="float32", requires_grad=True, **kwargs):
        """创建张量
        
        Args:
            array: 输入数据，可以是列表、numpy数组等
            dtype: 数据类型
            requires_grad: 是否需要计算梯度
        """
        if array is None:
            raise ValueError("array cannot be None")
        try:
            array_data = numpy.array(array, dtype=dtype)
        except Exception as e:
            raise ValueError(f"Failed to create tensor from array: {e}")
        self._init(
            None,
            [],
            cached_data=array_data,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def from_numpy(numpy_array, dtype):
        """从numpy数组创建张量
        
        Args:
            numpy_array: numpy数组
            dtype: 数据类型
        
        Returns:
            Tensor: 创建的张量
        """
        return Tensor(numpy_array, dtype=dtype, requires_grad=False)
    
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        """从操作创建张量
        
        Args:
            op: 操作符
            inputs: 输入节点列表
        
        Returns:
            Tensor: 创建的张量
        """
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not config.lazy_mode:
            tensor.realize_cached_data()
        return tensor
    
    @staticmethod
    def make_const(data, requires_grad=False):
        """创建常量张量
        
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
        
        Returns:
            Tensor: 常量张量
        """
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor_data = data.realize_cached_data()
        else:
            tensor_data = data
        tensor._init(None, [], cached_data=tensor_data, requires_grad=requires_grad)
        return tensor
    
    @property
    def data(self):
        """获取张量数据"""
        return self.cached_data

    def detach(self):
        """创建一个与当前张量数据相同但不需要梯度的新张量"""
        return Tensor.make_const(self.realize_cached_data())

    @data.setter
    def data(self, value):
        """设置张量数据"""
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self):
        """获取张量形状"""
        return self.realize_cached_data().shape
    
    @property
    def dtype(self):
        """获取张量数据类型"""
        return self.realize_cached_data().dtype

    def backward(self, out_grad=None):
        """执行反向传播
        
        Args:
            out_grad: 输出梯度，默认为全1张量
        """
        if out_grad is None:
            out_grad = Tensor(numpy.ones(self.shape), requires_grad=False)
        elif not isinstance(out_grad, Tensor):
            out_grad = Tensor(out_grad, requires_grad=False)
        from autograd import compute_gradient_of_variables
        compute_gradient_of_variables(self, out_grad)

    def __add__(self, other):
        if isinstance(other, Tensor):
            from ops import EWiseAdd
            return EWiseAdd()(self, other)
        else:
            from ops import AddScalar
            return AddScalar(other)(self)
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            from ops import EWiseMul
            return EWiseMul()(self, other)
        else:
            from ops import MulScalar
            return MulScalar(other)(self)
        
    def __sub__(self, other):
        if isinstance(other, Tensor):
            from ops import EWiseAdd, Negate
            return EWiseAdd()(self, Negate()(other))
        else:
            from ops import AddScalar
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            from ops import EWiseDiv
            return EWiseDiv()(self, other)
        else:
            from ops import DivScalar
            return DivScalar(other)(self)

    def __matmul__(self, other):
        from ops import MatMul
        return MatMul()(self, other)

    def broadcast_to(self, shape):
        from ops import BroadcastTo
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        from ops import Reshape
        return Reshape(shape)(self)

    def sum(self, axis=None):
        from ops import Summation
        return Summation(axis)(self)

    def max(self, axis=None):
        from ops import Max
        return Max(axis)(self)

    def transpose(self, axes=None):
        from ops import Transpose
        return Transpose(axes)(self)

    def argmax(self, axis=None):
        if axis is None:
            return numpy.argmax(self.realize_cached_data())
        return numpy.argmax(self.realize_cached_data(), axis=axis)

    def item(self):
        return self.realize_cached_data().item()
