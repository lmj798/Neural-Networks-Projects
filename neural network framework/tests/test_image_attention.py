import numpy as np
import pytest

from nn import SEBlock
from tensor import Tensor


def test_se_block_forward_backward_shape():
    x = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float64), dtype="float64", requires_grad=True)
    se = SEBlock(16, reduction=4, dtype="float64")
    y = se(x)
    assert y.shape == (2, 16, 8, 8)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_se_block_invalid_reduction():
    with pytest.raises(ValueError):
        SEBlock(16, reduction=0)
