import numpy as np
import pytest

from data import Dataset, DataLoader
from tensor import Tensor


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
