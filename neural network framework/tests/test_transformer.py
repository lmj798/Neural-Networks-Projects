import numpy as np

from cifar10 import SimpleCIFAR10CNNTransformer, SimpleCIFAR10ViT
from nn import LayerNorm, TransformerEncoderBlock
from ops import layer_norm_last_dim
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


def test_layer_norm_module_forward_backward():
    x = Tensor(np.random.randn(2, 5, 8).astype(np.float64), dtype="float64", requires_grad=True)
    ln = LayerNorm(8, dtype="float64")
    y = ln(x)
    assert y.shape == (2, 5, 8)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_layer_norm_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(2, 4)).astype(np.float64)
    gamma_data = rng.normal(size=(4,)).astype(np.float64)
    beta_data = rng.normal(size=(4,)).astype(np.float64)
    w_data = rng.normal(size=(2, 4)).astype(np.float64)

    gamma = Tensor(gamma_data, dtype="float64", requires_grad=False)
    beta = Tensor(beta_data, dtype="float64", requires_grad=False)
    w = Tensor(w_data, dtype="float64", requires_grad=False)

    def fn(t):
        return (layer_norm_last_dim(t, gamma, beta) * w).sum()

    x = Tensor(x_data.copy(), dtype="float64", requires_grad=True)
    loss = fn(x)
    loss.backward()
    analytic = x.grad.realize_cached_data()
    numeric = _numeric_grad(fn, x_data)
    np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-3)


def test_transformer_encoder_block_forward_backward():
    x = Tensor(np.random.randn(2, 16, 32).astype(np.float64), dtype="float64", requires_grad=True)
    block = TransformerEncoderBlock(embed_dim=32, num_heads=4, mlp_ratio=2.0, dtype="float64")
    y = block(x)
    assert y.shape == (2, 16, 32)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_cnn_transformer_head_shape():
    model = SimpleCIFAR10CNNTransformer(dropout=0.0, num_layers=1, num_heads=4)
    x = Tensor(np.random.randn(3, 3, 32, 32).astype(np.float32), requires_grad=False)
    y = model(x)
    assert y.shape == (3, 10)


def test_vit_head_shape():
    model = SimpleCIFAR10ViT(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    x = Tensor(np.random.randn(3, 3, 32, 32).astype(np.float32), requires_grad=False)
    y = model(x)
    assert y.shape == (3, 10)
