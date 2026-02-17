from typing import Optional, List
import numpy as np
from tensor import Tensor

class Dataset():
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

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False, device=None):
        if batch_size is None:
            batch_size = len(dataset)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), range(self.batch_size, len(self.dataset), self.batch_size))
        self.idx = -1
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.ordering):
            raise StopIteration
        batch_indices = self.ordering[self.idx]
        batch_x, batch_y = self.dataset[batch_indices]
        return (Tensor(batch_x, requires_grad=False), Tensor(batch_y, dtype="int64", requires_grad=False))
