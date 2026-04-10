from typing import List, Optional

import numpy as np

from tensor import Tensor


class Dataset:
    def __init__(self, X, y, transforms: Optional[List] = None, image_shape: Optional[tuple] = (28, 28)):
        self.transforms = transforms
        self.X = X
        self.y = y
        self.image_shape = image_shape

    def _reshape_if_needed(self, img):
        if self.image_shape is None:
            return img
        if img.shape == self.image_shape:
            return img
        expected_size = int(np.prod(self.image_shape))
        if img.size == expected_size:
            return img.reshape(self.image_shape)
        return img

    def __getitem__(self, index) -> object:
        imgs = self.X[index]
        labels = self.y[index]
        is_batch = not np.isscalar(labels) and np.ndim(labels) > 0
        if is_batch:
            batch_imgs = []
            for img in imgs:
                img = self._reshape_if_needed(img)
                img = self.apply_transforms(img)
                batch_imgs.append(img)
            imgs = np.stack(batch_imgs)
        else:
            imgs = self._reshape_if_needed(imgs)
            imgs = self.apply_transforms(imgs)
        return (imgs, labels)

    def __len__(self) -> int:
        return self.X.shape[0]

    def apply_transforms(self, x):
        if self.transforms is not None:
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ):
        if batch_size is None:
            batch_size = len(dataset)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = seed
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self.ordering = self._build_ordering(np.arange(len(dataset), dtype=np.int64))

    def _build_ordering(self, indices: np.ndarray):
        ordering = []
        total = int(indices.shape[0])
        if self.drop_last:
            total = (total // self.batch_size) * self.batch_size
        for start in range(0, total, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            if self.drop_last and batch_indices.shape[0] < self.batch_size:
                continue
            ordering.append(batch_indices)
        return ordering

    def __iter__(self):
        if self.shuffle:
            indices = self._rng.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset), dtype=np.int64)
        self.ordering = self._build_ordering(indices)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.ordering):
            raise StopIteration
        batch_indices = self.ordering[self.idx]
        self.idx += 1
        # 批量获取数据，减少索引操作次数
        batch_x, batch_y = self.dataset[batch_indices]
        batch_x_np = np.asarray(batch_x)
        batch_y_np = np.asarray(batch_y)
        # 保持原始数据类型
        return (
            Tensor(batch_x_np, dtype=batch_x_np.dtype, requires_grad=False),
            Tensor(batch_y_np, dtype=batch_y_np.dtype, requires_grad=False),
        )

    def __len__(self):
        dataset_size = len(self.dataset)
        if self.drop_last:
            return dataset_size // self.batch_size
        return (dataset_size + self.batch_size - 1) // self.batch_size
