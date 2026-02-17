import numpy as np

from tensor import Tensor
from ops import sigmoid, tanh, leaky_relu, elu


def _numeric_grad(act_fn, x, eps=1e-4):
    x = x.copy()
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        y1 = act_fn(Tensor(x, dtype="float64", requires_grad=False)).sum().realize_cached_data()
        x[idx] = orig - eps
        y2 = act_fn(Tensor(x, dtype="float64", requires_grad=False)).sum().realize_cached_data()
        grad[idx] = (y1 - y2) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def _check_grad(act_fn, x, rtol=1e-3, atol=1e-3):
    x_tensor = Tensor(x.copy(), dtype="float64", requires_grad=True)
    loss = act_fn(x_tensor).sum()
    loss.backward()
    analytic = x_tensor.grad.realize_cached_data()
    numeric = _numeric_grad(act_fn, x)
    np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)


def test_sigmoid_grad():
    x = np.array([-1.2, -0.3, 0.4, 1.1], dtype=np.float64)
    _check_grad(sigmoid, x)


def test_tanh_grad():
    x = np.array([-1.3, -0.2, 0.6, 1.4], dtype=np.float64)
    _check_grad(tanh, x)


def test_leaky_relu_grad():
    x = np.array([-0.7, -0.2, 0.1, 0.5], dtype=np.float64)
    _check_grad(lambda t: leaky_relu(t, negative_slope=0.1), x)


def test_elu_grad():
    x = np.array([-0.8, -0.1, 0.2, 0.9], dtype=np.float64)
    _check_grad(lambda t: elu(t, alpha=1.0), x)
