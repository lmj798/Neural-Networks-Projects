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