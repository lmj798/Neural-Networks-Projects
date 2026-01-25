from typing import Any, List, Optional, Tuple
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