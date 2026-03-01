import numpy as np
import pytest

from nn import MultiHeadSelfAttention
from ops import softmax
from tensor import Tensor


def _numeric_grad(fn, x, eps=1e-5):
    x = x.copy()
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        y1 = fn(Tensor(x, dtype="float64", requires_grad=False)).realize_cached_data()
        x[idx] = orig - eps
        y2 = fn(Tensor(x, dtype="float64", requires_grad=False)).realize_cached_data()
        grad[idx] = (y1 - y2) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def test_softmax_axis_sums_to_one():
    x = Tensor(np.random.randn(4, 7).astype(np.float64), dtype="float64", requires_grad=False)
    y = softmax(x, axis=1).realize_cached_data()
    np.testing.assert_allclose(y.sum(axis=1), np.ones(4), rtol=1e-7, atol=1e-7)


def test_softmax_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, 5)).astype(np.float64)
    w_data = rng.normal(size=(3, 5)).astype(np.float64)
    w = Tensor(w_data, dtype="float64", requires_grad=False)

    def fn(t):
        return (softmax(t, axis=1) * w).sum()

    x = Tensor(x_data.copy(), dtype="float64", requires_grad=True)
    loss = fn(x)
    loss.backward()

    analytic = x.grad.realize_cached_data()
    numeric = _numeric_grad(fn, x_data)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-3)


def test_multi_head_self_attention_forward_backward():
    rng = np.random.default_rng(1)
    x_data = rng.normal(size=(2, 6, 8)).astype(np.float64)
    x = Tensor(x_data, dtype="float64", requires_grad=True)

    layer = MultiHeadSelfAttention(embed_dim=8, num_heads=2, dtype="float64")
    y = layer(x)
    assert y.shape == (2, 6, 8)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_multi_head_self_attention_invalid_head_split():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=10, num_heads=3)
