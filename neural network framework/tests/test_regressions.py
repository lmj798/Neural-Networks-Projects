import numpy as np
import pytest

from data import Dataset, DataLoader
from autograd import find_topo_sort
from init import init_He
from nn import Dropout, Linear, Module, Parameter, Sequential
from ops import conv2d, softmax_cross_entropy
from tensor import Op, Tensor, Value


def test_sequential_eval_disables_dropout():
    model = Sequential(Dropout(0.5))
    x_data = np.ones((32, 32), dtype=np.float32)
    x = Tensor(x_data, requires_grad=False)

    model.eval()
    out1 = model(x).realize_cached_data()
    out2 = model(x).realize_cached_data()

    np.testing.assert_allclose(out1, x_data)
    np.testing.assert_allclose(out2, x_data)

    model.train()
    out3 = model(x).realize_cached_data()
    assert not np.allclose(out3, x_data)


def test_ewise_mul_broadcast_gradient_reduces_to_input_shape():
    a = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)

    y = (a * b).sum()
    y.backward()

    np.testing.assert_equal(b.grad.shape, (3,))
    np.testing.assert_allclose(
        b.grad.realize_cached_data(),
        a.realize_cached_data().sum(axis=0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_ewise_div_broadcast_gradient_reduces_to_input_shape():
    a = Tensor(np.array([[2.0, 4.0, 8.0], [1.0, 3.0, 5.0]], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 4.0], dtype=np.float32), requires_grad=True)

    y = (a / b).sum()
    y.backward()

    expected = -(a.realize_cached_data() / (b.realize_cached_data() ** 2)).sum(axis=0)
    np.testing.assert_equal(b.grad.shape, (3,))
    np.testing.assert_allclose(b.grad.realize_cached_data(), expected, rtol=1e-5, atol=1e-5)


def test_matmul_batched_backward_handles_batch_and_reduction():
    rng = np.random.default_rng(0)
    a_data = rng.normal(size=(4, 2, 3)).astype(np.float32)
    b_data = rng.normal(size=(3, 5)).astype(np.float32)

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    y = (a @ b).sum()
    y.backward()

    expected_grad_a = np.matmul(np.ones((4, 2, 5), dtype=np.float32), b_data.T)
    expected_grad_b = np.matmul(a_data.transpose(0, 2, 1), np.ones((4, 2, 5), dtype=np.float32)).sum(axis=0)

    np.testing.assert_allclose(a.grad.realize_cached_data(), expected_grad_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.grad.realize_cached_data(), expected_grad_b, rtol=1e-5, atol=1e-5)


def test_dataloader_preserves_label_dtype():
    X = np.random.randn(4, 2).astype(np.float32)
    y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ds = Dataset(X, y, image_shape=None)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    _, batch_y = next(iter(loader))

    assert batch_y.dtype == np.float32
    np.testing.assert_allclose(batch_y.realize_cached_data(), np.array([0.1, 0.2], dtype=np.float32))


def test_linear_respects_dtype_argument():
    layer = Linear(8, 4, dtype="float64")
    assert layer.weight.dtype == np.float64
    assert layer.bias.dtype == np.float64


def test_init_he_uses_fan_in_std():
    np.random.seed(0)
    w = init_He(1024, 512, dtype="float64")
    expected_std = np.sqrt(2.0 / 1024)
    actual_std = float(w.std())
    assert np.isclose(actual_std, expected_std, rtol=0.1)


def test_value_make_from_op_realizes_without_missing_methods():
    class AddOne(Op):
        def compute(self, a):
            return a + 1

        def gradient(self, out_grad, node):
            return out_grad

    x = Tensor([1.0], requires_grad=False)
    v = Value.make_from_op(AddOne(), [x])
    np.testing.assert_allclose(v.realize_cached_data(), np.array([2.0], dtype=np.float32))


def test_max_gradient_splits_evenly_on_ties():
    x = Tensor(np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 2.0]], dtype=np.float32), requires_grad=True)
    y = x.max(axis=1).sum()
    y.backward()
    expected = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.realize_cached_data(), expected, rtol=1e-6, atol=1e-6)


def test_module_parameters_deduplicate_shared_parameter():
    class SharedParamModule(Module):
        def __init__(self):
            super().__init__()
            shared = Parameter(np.array([1.0], dtype=np.float32))
            self.p1 = shared
            self.p2 = shared

        def forward(self, x):
            return x

    module = SharedParamModule()
    params = module.parameters()
    assert len(params) == 1
    assert params[0] is module.p1


def test_module_state_dict_roundtrip_restores_parameters():
    model = Sequential(Linear(4, 3), Linear(3, 2))

    original_state = model.state_dict()

    for param in model.parameters():
        param.cached_data = np.zeros_like(param.realize_cached_data())

    result = model.load_state_dict(original_state)
    assert result["missing_keys"] == []
    assert result["unexpected_keys"] == []

    for name, param in model.named_parameters():
        np.testing.assert_allclose(param.realize_cached_data(), original_state[name])


def test_module_load_state_dict_strict_checks_missing_and_unexpected():
    model = Sequential(Linear(2, 2))

    with pytest.raises(KeyError):
        model.load_state_dict({})

    state = model.state_dict()
    state["unexpected.param"] = np.array([1.0], dtype=np.float32)
    with pytest.raises(KeyError):
        model.load_state_dict(state)


def test_find_topo_sort_visits_all_roots():
    x = Tensor([1.0], requires_grad=True)
    y1 = x + 1.0
    y2 = x * 2.0
    topo = find_topo_sort([y1, y2])
    assert x in topo
    assert y1 in topo
    assert y2 in topo


def test_softmax_cross_entropy_gradient_matches_expected():
    logits_data = np.array([[1.0, 2.0, 0.5], [0.1, -0.2, 0.3]], dtype=np.float64)
    targets_data = np.array([1, 2], dtype=np.int64)

    logits = Tensor(logits_data, dtype="float64", requires_grad=True)
    targets = Tensor(targets_data, dtype=np.int64, requires_grad=False)
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    shifted = logits_data - np.max(logits_data, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    expected_grad = probs
    expected_grad[np.arange(logits_data.shape[0]), targets_data] -= 1.0
    expected_grad /= logits_data.shape[0]

    np.testing.assert_allclose(logits.grad.realize_cached_data(), expected_grad, rtol=1e-6, atol=1e-6)


def test_conv2d_backward_computes_input_and_weight_gradients():
    x = Tensor(np.ones((1, 1, 4, 4), dtype=np.float32), requires_grad=True)
    weight = Tensor(np.ones((1, 1, 3, 3), dtype=np.float32), requires_grad=True)

    loss = conv2d(x, weight).sum()
    loss.backward()

    expected_x_grad = np.array(
        [[[[1, 2, 2, 1],
           [2, 4, 4, 2],
           [2, 4, 4, 2],
           [1, 2, 2, 1]]]],
        dtype=np.float32,
    )
    expected_weight_grad = np.full((1, 1, 3, 3), 4.0, dtype=np.float32)

    np.testing.assert_allclose(x.grad.realize_cached_data(), expected_x_grad)
    np.testing.assert_allclose(weight.grad.realize_cached_data(), expected_weight_grad)


def test_tensor_supports_scalar_left_hand_arithmetic():
    x = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=False)

    np.testing.assert_allclose((1.0 + x).realize_cached_data(), np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose((2.0 * x).realize_cached_data(), np.array([2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose((5.0 - x).realize_cached_data(), np.array([4.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose((4.0 / x).realize_cached_data(), np.array([4.0, 2.0], dtype=np.float32))


def test_tensor_data_setter_accepts_numpy_arrays():
    x = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=False)
    replacement = np.array([3.0, 4.0], dtype=np.float32)

    x.data = replacement

    np.testing.assert_allclose(x.realize_cached_data(), replacement)
