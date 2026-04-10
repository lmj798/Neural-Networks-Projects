import numpy as np
import pytest

from data import Dataset, DataLoader
from tensor import Tensor, no_grad


def test_backward_accepts_numpy_out_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()
    y.backward(np.array(1.0, dtype=np.float32))
    np.testing.assert_allclose(
        x.grad.realize_cached_data(),
        np.ones_like(x.realize_cached_data()),
    )


def test_dataset_single_and_batch_image_indexing():
    X = np.random.randn(2, 28, 28).astype(np.float32)
    y = np.array([0, 1])
    ds = Dataset(X, y)

    img, label = ds[0]
    assert img.shape == (28, 28)
    assert int(label) == 0

    img_batch, label_batch = ds[np.array([0, 1])]
    assert img_batch.shape == (2, 28, 28)
    assert label_batch.shape == (2,)


def test_dataset_flattened_indexing():
    X = np.random.randn(2, 784).astype(np.float32)
    y = np.array([3, 4])
    ds = Dataset(X, y)

    img, label = ds[0]
    assert img.shape == (28, 28)
    assert int(label) == 3

    img_batch, label_batch = ds[np.array([0, 1])]
    assert img_batch.shape == (2, 28, 28)
    assert label_batch.shape == (2,)


def test_dataloader_batch_size_none_full_batch():
    X = np.random.randn(3, 28, 28).astype(np.float32)
    y = np.array([0, 1, 2])
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=None, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (3, 28, 28)
    assert batch_y.shape == (3,)


def test_dataloader_invalid_batch_size():
    X = np.random.randn(2, 28, 28).astype(np.float32)
    y = np.array([0, 1])
    ds = Dataset(X, y)

    with pytest.raises(ValueError):
        DataLoader(ds, batch_size=0)


def test_dataloader_seed_reproducible_shuffle_order():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.int64)
    ds = Dataset(X, y, image_shape=None)

    loader1 = DataLoader(ds, batch_size=3, shuffle=True, seed=123)
    loader2 = DataLoader(ds, batch_size=3, shuffle=True, seed=123)

    order1 = np.concatenate([batch_y.realize_cached_data() for _, batch_y in loader1])
    order2 = np.concatenate([batch_y.realize_cached_data() for _, batch_y in loader2])

    np.testing.assert_array_equal(order1, order2)


def test_dataloader_drop_last_drops_incomplete_batch():
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.int64)
    ds = Dataset(X, y, image_shape=None)

    loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
    batches = list(loader)

    assert len(loader) == 2
    assert len(batches) == 2
    labels = np.concatenate([batch_y.realize_cached_data() for _, batch_y in batches])
    np.testing.assert_array_equal(labels, np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64))


def test_no_grad_disables_graph_tracking():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with no_grad():
        y = x * 2.0

    assert y.requires_grad is False

    z = (x * 2.0).sum()
    z.backward()
    np.testing.assert_allclose(x.grad.realize_cached_data(), np.array([2.0, 2.0, 2.0], dtype=np.float32))
