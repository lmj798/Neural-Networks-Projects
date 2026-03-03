import numpy
from tensor import Tensor, NDArray, Op

def _sum_to_shape(grad: Tensor, shape):
    if grad.shape == shape:
        return grad

    if shape == ():
        return grad.sum()

    grad_shape = grad.shape
    if len(grad_shape) < len(shape):
        raise ValueError(f"Cannot reduce gradient shape {grad_shape} to {shape}")

    ndims_added = len(grad_shape) - len(shape)
    axes_to_sum = list(range(ndims_added))

    for i, (gdim, sdim) in enumerate(zip(grad_shape[ndims_added:], shape)):
        if sdim == 1 and gdim != 1:
            axes_to_sum.append(ndims_added + i)

    if axes_to_sum:
        grad = grad.sum(tuple(sorted(set(axes_to_sum))))
    if grad.shape != shape:
        grad = grad.reshape(shape)
    return grad


def _swap_last_two_axes(t: Tensor):
    if len(t.shape) < 2:
        raise ValueError("MatMul gradient requires tensors with at least 2 dimensions.")
    axes = list(range(len(t.shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return t.transpose(tuple(axes))


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

        if a.shape != out_grad.shape:
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
        grad_a = out_grad * b
        grad_b = out_grad * a
        return _sum_to_shape(grad_a, a.shape), _sum_to_shape(grad_b, b.shape)
    
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
        grad_a = out_grad / b
        grad_b = (out_grad * a / (b * b)) * -1
        return _sum_to_shape(grad_a, a.shape), _sum_to_shape(grad_b, b.shape)

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
        if len(a.shape) == 2 and len(b.shape) == 2:
            grad_a = out_grad @ b.transpose()
            grad_b = a.transpose() @ out_grad
        else:
            grad_a = out_grad @ _swap_last_two_axes(b)
            grad_b = _swap_last_two_axes(a) @ out_grad

        return _sum_to_shape(grad_a, a.shape), _sum_to_shape(grad_b, b.shape)

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray):
        return numpy.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        ndims_added = len(output_shape) - len(input_shape)
        axes_to_sum = list(range(ndims_added))

        for i, (in_dim, out_dim) in enumerate(zip(input_shape, output_shape[ndims_added:])):
            if in_dim == 1 and out_dim > 1:
                axes_to_sum.append(i + ndims_added)

        if axes_to_sum:
            grad = out_grad.sum(tuple(axes_to_sum))
        else:
            grad = out_grad

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
        a_data = a.realize_cached_data()
        if self.axis is None:
            max_val = numpy.max(a_data)
            mask = (a_data == max_val).astype(numpy.float32)
            mask = mask / mask.sum()
            return out_grad.broadcast_to(a.shape) * Tensor(mask, requires_grad=False)

        axes = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axes = tuple(ax if ax >= 0 else ax + a_data.ndim for ax in axes)

        max_keepdims = numpy.max(a_data, axis=axes, keepdims=True)
        mask = (a_data == max_keepdims).astype(numpy.float32)
        counts = mask.sum(axis=axes, keepdims=True)
        mask = mask / counts
        return out_grad.reshape(max_keepdims.shape).broadcast_to(a.shape) * Tensor(mask, requires_grad=False)

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
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i
        return out_grad.transpose(tuple(inv_axes))

class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        grad = (a > 0).astype(a.dtype, copy=False)
        return out_grad * Tensor(grad, dtype=a.dtype, requires_grad=False)

def relu(a):
    return ReLU()(a)

def _im2col(x, kernel_h, kernel_w, stride, padding):
    x_padded = numpy.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    n, c, h, w = x_padded.shape
    out_h = (h - kernel_h) // stride + 1
    out_w = (w - kernel_w) // stride + 1

    shape = (n, c, kernel_h, kernel_w, out_h, out_w)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2],
        x_padded.strides[3],
        x_padded.strides[2] * stride,
        x_padded.strides[3] * stride,
    )
    cols = numpy.lib.stride_tricks.as_strided(
        x_padded, shape=shape, strides=strides
    )
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kernel_h, kernel_w, stride, padding, out_h, out_w):
    n, c, h, w = x_shape
    h_padded = h + 2 * padding
    w_padded = w + 2 * padding
    x_padded = numpy.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)

    cols_reshaped = (
        cols.reshape(n, out_h, out_w, c, kernel_h, kernel_w)
        .transpose(0, 3, 4, 5, 1, 2)
    )

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            x_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


class Conv2D(TensorOp):
    def __init__(self, stride=1, padding=0):
        self.stride = stride
        self.padding = padding

    def compute(self, x: NDArray, w: NDArray):
        n, c, h, w_in = x.shape
        out_channels, in_channels, kernel_h, kernel_w = w.shape
        if c != in_channels:
            raise ValueError("Input channels do not match weight channels.")

        cols, out_h, out_w = _im2col(x, kernel_h, kernel_w, self.stride, self.padding)
        w_col = w.reshape(out_channels, -1)
        out = cols @ w_col.T
        out = out.reshape(n, out_h, out_w, out_channels).transpose(0, 3, 1, 2)
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        x = node.inputs[0].realize_cached_data()
        w = node.inputs[1].realize_cached_data()
        n, c, h, w_in = x.shape
        out_channels, _, kernel_h, kernel_w = w.shape

        cols, out_h, out_w = _im2col(x, kernel_h, kernel_w, self.stride, self.padding)
        dout = out_grad.realize_cached_data().transpose(0, 2, 3, 1).reshape(-1, out_channels)

        grad_w = dout.T @ cols
        grad_w = grad_w.reshape(w.shape)

        w_col = w.reshape(out_channels, -1)
        grad_cols = dout @ w_col
        grad_x = _col2im(grad_cols, (n, c, h, w_in), kernel_h, kernel_w, self.stride, self.padding, out_h, out_w)

        return Tensor(grad_x, requires_grad=False), Tensor(grad_w, requires_grad=False)


def conv2d(a, weight, stride=1, padding=0):
    return Conv2D(stride=stride, padding=padding)(a, weight)

class SoftmaxCrossEntropy(TensorOp):
    def compute(self, logits: NDArray, targets: NDArray):
        if logits.ndim != 2:
            raise ValueError("softmax_cross_entropy expects logits with shape (batch_size, num_classes)")

        target_indices = numpy.asarray(targets, dtype=numpy.int64).reshape(-1)
        if logits.shape[0] != target_indices.shape[0]:
            raise ValueError("Targets length must match logits batch size")

        shifted = logits - numpy.max(logits, axis=1, keepdims=True)
        log_probs = shifted - numpy.log(numpy.sum(numpy.exp(shifted), axis=1, keepdims=True))
        losses = -log_probs[numpy.arange(logits.shape[0]), target_indices]
        return numpy.array(losses.mean(), dtype=logits.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        logits, targets = node.inputs
        logits_data = logits.realize_cached_data()
        target_indices = numpy.asarray(targets.realize_cached_data(), dtype=numpy.int64).reshape(-1)

        shifted = logits_data - numpy.max(logits_data, axis=1, keepdims=True)
        exp_shifted = numpy.exp(shifted)
        probs = exp_shifted / numpy.sum(exp_shifted, axis=1, keepdims=True)

        grad_logits = probs
        grad_logits[numpy.arange(logits_data.shape[0]), target_indices] -= 1.0
        grad_logits /= logits_data.shape[0]

        grad_logits_tensor = Tensor(grad_logits.astype(logits_data.dtype, copy=False), requires_grad=False) * out_grad
        grad_targets = Tensor(
            numpy.zeros(targets.shape, dtype=numpy.float32),
            requires_grad=False,
        )
        return grad_logits_tensor, grad_targets


def softmax_cross_entropy(logits, targets):
    return SoftmaxCrossEntropy()(logits, targets)

class Softmax(TensorOp):
    def __init__(self, axis=-1):
        self.axis = axis

    def compute(self, a: NDArray):
        shifted = a - numpy.max(a, axis=self.axis, keepdims=True)
        exp_shifted = numpy.exp(shifted)
        return exp_shifted / numpy.sum(exp_shifted, axis=self.axis, keepdims=True)

    def gradient(self, out_grad, node):
        probs = node.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()
        dot = numpy.sum(out_grad_data * probs, axis=self.axis, keepdims=True)
        grad = probs * (out_grad_data - dot)
        return Tensor(grad, requires_grad=False)


def softmax(a, axis=-1):
    return Softmax(axis=axis)(a)

class LayerNormLastDim(TensorOp):
    def __init__(self, eps=1e-5):
        self.eps = eps

    def compute(self, x: NDArray, gamma: NDArray, beta: NDArray):
        mean = numpy.mean(x, axis=-1, keepdims=True)
        var = numpy.var(x, axis=-1, keepdims=True)
        inv_std = 1.0 / numpy.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        return x_hat * gamma.reshape((1,) * (x.ndim - 1) + gamma.shape) + beta.reshape((1,) * (x.ndim - 1) + beta.shape)

    def gradient(self, out_grad, node):
        x, gamma, beta = node.inputs
        x_data = x.realize_cached_data()
        gamma_data = gamma.realize_cached_data()
        out_grad_data = out_grad.realize_cached_data()

        mean = numpy.mean(x_data, axis=-1, keepdims=True)
        var = numpy.var(x_data, axis=-1, keepdims=True)
        inv_std = 1.0 / numpy.sqrt(var + self.eps)
        x_centered = x_data - mean
        x_hat = x_centered * inv_std

        # Grad wrt affine params.
        reduce_axes = tuple(range(x_data.ndim - 1))
        grad_gamma = numpy.sum(out_grad_data * x_hat, axis=reduce_axes)
        grad_beta = numpy.sum(out_grad_data, axis=reduce_axes)

        # Grad wrt normalized input.
        n = x_data.shape[-1]
        gamma_b = gamma_data.reshape((1,) * (x_data.ndim - 1) + gamma_data.shape)
        dx_hat = out_grad_data * gamma_b

        dvar = numpy.sum(dx_hat * x_centered * -0.5 * (inv_std ** 3), axis=-1, keepdims=True)
        dmean = (
            numpy.sum(dx_hat * -inv_std, axis=-1, keepdims=True)
            + dvar * numpy.mean(-2.0 * x_centered, axis=-1, keepdims=True)
        )
        grad_x = dx_hat * inv_std + dvar * (2.0 * x_centered / n) + dmean / n

        return (
            Tensor(grad_x.astype(x_data.dtype, copy=False), requires_grad=False),
            Tensor(grad_gamma.astype(gamma_data.dtype, copy=False), requires_grad=False),
            Tensor(grad_beta.astype(gamma_data.dtype, copy=False), requires_grad=False),
        )


def layer_norm_last_dim(x, gamma, beta, eps=1e-5):
    return LayerNormLastDim(eps=eps)(x, gamma, beta)

class Sigmoid(TensorOp):
    def compute(self, a):
        return 1 / (1 + numpy.exp(-a))

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        sig = 1 / (1 + numpy.exp(-a))
        grad = sig * (1 - sig)
        return out_grad * Tensor(grad, dtype=a.dtype, requires_grad=False)

def sigmoid(a):
    return Sigmoid()(a)

class Tanh(TensorOp):
    def compute(self, a):
        return numpy.tanh(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        t = numpy.tanh(a)
        grad = 1 - t * t
        return out_grad * Tensor(grad, dtype=a.dtype, requires_grad=False)

def tanh(a):
    return Tanh()(a)

class LeakyReLU(TensorOp):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def compute(self, a):
        return numpy.where(a > 0, a, self.negative_slope * a)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        grad = numpy.where(a > 0, 1.0, self.negative_slope)
        return out_grad * Tensor(grad, dtype=a.dtype, requires_grad=False)

def leaky_relu(a, negative_slope=0.01):
    return LeakyReLU(negative_slope)(a)

class ELU(TensorOp):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute(self, a):
        return numpy.where(a > 0, a, self.alpha * (numpy.exp(a) - 1))

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        grad = numpy.where(a > 0, 1.0, self.alpha * numpy.exp(a))
        return out_grad * Tensor(grad, dtype=a.dtype, requires_grad=False)

def elu(a, alpha=1.0):
    return ELU(alpha)(a)
