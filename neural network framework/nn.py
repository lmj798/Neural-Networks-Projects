from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, List, Tuple

import numpy

from init import init_He, init_Xavier
from ops import conv2d, elu, layer_norm_last_dim, leaky_relu, relu, sigmoid, softmax, tanh
from tensor import Tensor


class Module(ABC):
    """模型模块基类"""
    def __init__(self):
        """初始化模块"""
        self.training = True

    def parameters(self) -> List["Tensor"]:
        """获取模块的所有参数
        
        Returns:
            List[Tensor]: 参数列表
        """
        return _deduplicate_by_identity(_unpack_params(self.__dict__))

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, "Tensor"]]:
        """获取模块的所有命名参数
        
        Args:
            prefix: 参数名前缀
        
        Returns:
            List[Tuple[str, Tensor]]: 命名参数列表
        """
        named = _unpack_named_params(self.__dict__, prefix)
        seen = set()
        deduped = []
        for name, param in named:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            deduped.append((name, param))
        return deduped

    def state_dict(self) -> Dict[str, numpy.ndarray]:
        """获取模块的状态字典
        
        Returns:
            Dict[str, numpy.ndarray]: 状态字典
        """
        state = {}
        for name, param in self.named_parameters():
            state[name] = numpy.array(param.realize_cached_data(), copy=True)
        return state

    def load_state_dict(self, state_dict: Dict[str, numpy.ndarray], strict: bool = True) -> Dict[str, List[str]]:
        """加载模块的状态字典
        
        Args:
            state_dict: 状态字典
            strict: 是否严格匹配
        
        Returns:
            Dict[str, List[str]]: 缺失和意外的键
        """
        named_params = dict(self.named_parameters())
        missing_keys = [name for name in named_params if name not in state_dict]
        unexpected_keys = [name for name in state_dict if name not in named_params]

        if strict and (missing_keys or unexpected_keys):
            raise KeyError(
                f"State dict mismatch. Missing keys: {missing_keys}; Unexpected keys: {unexpected_keys}"
            )

        for name, param in named_params.items():
            if name not in state_dict:
                continue
            loaded = numpy.asarray(state_dict[name], dtype=param.dtype)
            if loaded.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}'. Expected {param.shape}, got {loaded.shape}."
                )
            param.cached_data = numpy.array(loaded, copy=True)

        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def _children(self):
        """获取所有子模块"""
        return _deduplicate_by_identity(_child_modules(self.__dict__))

    def __call__(self, *args, **kwargs):
        """调用模块
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            前向传播结果
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self):
        """前向传播
        
        Returns:
            前向传播结果
        """
        pass

    def eval(self):
        """设置为评估模式"""
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        """设置为训练模式"""
        self.training = True
        for m in self._children():
            m.training = True


class Parameter(Tensor):
    pass


def _deduplicate_by_identity(items: List[Any]) -> List[Any]:
    seen = set()
    deduped = []
    for item in items:
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(item)
    return deduped


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for _, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _join_name(prefix: str, name: object) -> str:
    name_str = str(name)
    return f"{prefix}.{name_str}" if prefix else name_str


def _unpack_named_params(value: object, prefix: str = "") -> List[Tuple[str, Tensor]]:
    if isinstance(value, Parameter):
        return [(prefix, value)]
    elif isinstance(value, Module):
        return value.named_parameters(prefix=prefix)
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_named_params(v, _join_name(prefix, k))
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for i, v in enumerate(value):
            params += _unpack_named_params(v, _join_name(prefix, i))
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
        for _, v in value.items():
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
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
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
        self.weight = Parameter(s, dtype=dtype)
        if bias:
            self.bias = Parameter(numpy.zeros(self.out_features), dtype=dtype)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        if not isinstance(X, Tensor):
            raise TypeError("Input must be a Tensor")
        if len(X.shape) < 2:
            raise ValueError(f"Expected at least 2D input, got {len(X.shape)}D input")
        if X.shape[-1] != self.in_features:
            raise ValueError(f"Input features mismatch: expected {self.in_features}, got {X.shape[-1]}")
        X_out = X @ self.weight
        if self.bias is not None:
            return X_out + self.bias
        return X_out


class Sequential(Module):
    """顺序模块，按顺序执行多个模块"""
    def __init__(self, *modules):
        """初始化顺序模块
        
        Args:
            *modules: 模块列表
        """
        super().__init__()
        if not modules:
            raise ValueError("Sequential module cannot be empty")
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            Tensor: 输出张量
        """
        for module in self.modules:
            x = module(x)
        return x


class ReLUModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class SigmoidModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class TanhModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class GELUModule(Module):
    def forward(self, x: Tensor) -> Tensor:
        # tanh-based GELU approximation used in many transformer implementations.
        c = numpy.sqrt(2.0 / numpy.pi)
        x3 = x * x * x
        inner = x + (x3 * 0.044715)
        return (x * 0.5) * (tanh(inner * c) + 1.0)


class LeakyReLUModule(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, negative_slope=self.negative_slope)


class ELUModule(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        out = conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias = self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(out.shape)
            out = out + bias
        return out


class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        bias=True,
        dtype="float32",
        weight_init=None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / numpy.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, weight_init=weight_init)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, weight_init=weight_init)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, weight_init=weight_init)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, weight_init=weight_init)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = self.k_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        v = self.v_proj(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))

        scores = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        attn = softmax(scores, axis=-1)
        context = attn @ v

        context = context.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, self.embed_dim))
        return self.out_proj(context)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, dtype="float32"):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        if len(normalized_shape) != 1:
            raise ValueError("This LayerNorm currently supports only 1D normalized_shape.")
        self.normalized_shape = normalized_shape
        self.eps = eps
        dim = normalized_shape[0]
        self.weight = Parameter(numpy.ones(dim, dtype=dtype), dtype=dtype)
        self.bias = Parameter(numpy.zeros(dim, dtype=dtype), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm_last_dim(x, self.weight, self.bias, eps=self.eps)


class TransformerEncoderBlock(Module):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        dtype="float32",
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = LayerNorm(embed_dim, dtype=dtype)
        self.attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dtype=dtype)
        self.norm2 = LayerNorm(embed_dim, dtype=dtype)
        self.fc1 = Linear(embed_dim, hidden_dim, dtype=dtype)
        self.act = GELUModule()
        self.fc2 = Linear(hidden_dim, embed_dim, dtype=dtype)
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        y = self.fc2(self.act(self.fc1(self.norm2(x))))
        if self.dropout is not None:
            y = self.dropout(y)
        return x + y


class SEBlock(Module):
    def __init__(self, channels, reduction=16, dtype="float32"):
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be a positive integer.")
        hidden = max(channels // reduction, 1)
        self.channels = channels
        self.fc1 = Linear(channels, hidden, dtype=dtype)
        self.fc2 = Linear(hidden, channels, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, h, w = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        s = x.sum(axis=(2, 3)) / float(h * w)  # Global average pooling.
        z = relu(self.fc1(s))
        z = sigmoid(self.fc2(z))
        z = z.reshape((batch_size, channels, 1, 1)).broadcast_to(x.shape)
        return x * z


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability must be in [0, 1).")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p == 0:
            return x
        mask = (numpy.random.rand(*x.shape) >= self.p).astype(x.dtype)
        mask = mask / (1.0 - self.p)
        return x * Tensor(mask, requires_grad=False)


class Flatten(Module):
    def forward(self, X):
        size = reduce(lambda a, b: a * b, X.shape[1:])  # Flatten every dimension except batch.
        return X.reshape((X.shape[0], size))
