import numpy
from tensor import Tensor, NDArray, Op

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
        inv_axes = [0] * len(self.axes)
        for i, ax in enumerate(self.axes):
            inv_axes[ax] = i
        return out_grad.transpose(tuple(inv_axes))

class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0)

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

class Sigmoid(TensorOp):
    def compute(self, a):
        return 1 / (1 + numpy.exp(-a))

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        sig = 1 / (1 + numpy.exp(-a))
        return out_grad * Tensor(sig * (1 - sig))

def sigmoid(a):
    return Sigmoid()(a)

class Tanh(TensorOp):
    def compute(self, a):
        return numpy.tanh(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        t = numpy.tanh(a)
        return out_grad * Tensor(1 - t * t)

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
        return out_grad * Tensor(grad)

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
        return out_grad * Tensor(grad)

def elu(a, alpha=1.0):
    return ELU(alpha)(a)
