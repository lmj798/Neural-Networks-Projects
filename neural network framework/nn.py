from abc import abstractmethod
from functools import reduce
from typing import List, Any
import numpy
from tensor import Tensor
from ops import relu, sigmoid, tanh, leaky_relu, elu, conv2d
from init import init_He, init_Xavier

class ABC:
    pass

class Module(ABC):
    def __init__(self):
        self.training = True
    
    def parameters(self)->List["Tensor"]:
        return _unpack_params(self.__dict__)
    
    def _children(self):
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
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype="float32",
        weight_init=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight_init is None or weight_init == "he":
            s = init_He(in_features, out_features, dtype=dtype)
        elif weight_init == "xavier":
            s = init_Xavier(in_features, out_features, dtype=dtype)
        elif callable(weight_init):
            s = weight_init(in_features, out_features, dtype=dtype)
        else:
            raise ValueError("weight_init must be None, 'he', 'xavier', or a callable")
        self.weight = Parameter(s, dtype="float32")
        if bias:
            self.bias = Parameter(numpy.zeros(self.out_features), dtype="float32")
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_out = X @ self.weight
        if self.bias is not None:
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

class SigmoidModule(Module):
    def forward(self, x: Tensor)->Tensor:
        return sigmoid(x)

class TanhModule(Module):
    def forward(self, x: Tensor)->Tensor:
        return tanh(x)

class LeakyReLUModule(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor)->Tensor:
        return leaky_relu(x, negative_slope=self.negative_slope)

class ELUModule(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor)->Tensor:
        return elu(x, alpha=self.alpha)

class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dtype="float32",
        weight_init=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_h, kernel_w = kernel_size, kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_h * kernel_w
        fan_out = out_channels * kernel_h * kernel_w

        if callable(weight_init):
            w = weight_init(out_channels, in_channels, kernel_h, kernel_w, dtype=dtype)
            w = numpy.array(w, dtype=dtype)
        elif weight_init is None or weight_init == "he":
            std = numpy.sqrt(2.0 / fan_in)
            w = numpy.random.normal(0, std, (out_channels, in_channels, kernel_h, kernel_w)).astype(dtype)
        elif weight_init == "xavier":
            std = numpy.sqrt(2.0 / (fan_in + fan_out))
            w = numpy.random.normal(0, std, (out_channels, in_channels, kernel_h, kernel_w)).astype(dtype)
        else:
            raise ValueError("weight_init must be None, 'he', 'xavier', or a callable")
        self.weight = Parameter(w, dtype=dtype)
        if bias:
            self.bias = Parameter(numpy.zeros(out_channels), dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor)->Tensor:
        out = conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(out.shape)
            out = out + bias
        return out

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability must be in [0, 1).")
        self.p = p

    def forward(self, x: Tensor)->Tensor:
        if (not self.training) or self.p == 0:
            return x
        mask = (numpy.random.rand(*x.shape) >= self.p).astype(x.dtype)
        mask = mask / (1.0 - self.p)
        return x * Tensor(mask, requires_grad=False)

class Flatten(Module):
    def forward(self, X):
        size = reduce(lambda a, b: a * b, X.shape[1:])  # 除batch维度外的所有维度
        return X.reshape((X.shape[0], size))
