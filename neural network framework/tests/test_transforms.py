import numpy as np
import pytest

from transforms import Normalize, RandomHorizontalFlip, RandomCrop, Resize, Compose


def test_normalize():
    """测试归一化变换"""
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    transform = Normalize(mean=(2,), std=(1,))
    result = transform(x)
    expected = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_random_horizontal_flip():
    """测试随机水平翻转"""
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    transform = RandomHorizontalFlip(p=1.0)  # 强制翻转
    result = transform(x)
    expected = np.array([[2, 1, 0], [5, 4, 3]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_random_crop():
    """测试随机裁剪"""
    x = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.float32)
    transform = RandomCrop(size=(2, 2), padding=0)
    result = transform(x)
    assert result.shape == (2, 2)


def test_resize():
    """测试 resize 变换"""
    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    transform = Resize(size=(4, 4))
    result = transform(x)
    assert result.shape == (4, 4)


def test_compose():
    """测试组合变换"""
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    transform = Compose([
        RandomHorizontalFlip(p=1.0),
        Normalize(mean=(2,), std=(1,))
    ])
    result = transform(x)
    expected = np.array([[0, -1, -2], [3, 2, 1]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
