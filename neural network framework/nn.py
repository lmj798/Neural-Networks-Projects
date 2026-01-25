from abc import abstractmethod
from functools import reduce
from typing import List, Any
import numpy
from tensor import Tensor
from ops import relu
from init import init_He

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
        size = reduce(lambda a, b: a * b, X.shape[1:])  # 除batch维度外的所有维度
        return X.reshape((X.shape[0], size))