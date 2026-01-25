from typing import Optional, List
import numpy as np
from tensor import Tensor

class Dataset():
    def __init__(self, X, y, transforms: Optional[List] = None):
        self.transforms = transforms        
        self.X = X
        self.y = y

    def __getitem__(self, index) -> object:
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            # 批量处理 - 保持batch结构
            batch_imgs = []
            for img in imgs:
                if img.shape == (28, 28):
                    img_reshaped = img  # 已经是正确形状
                else:
                    img_reshaped = img.reshape(28, 28)  # 确保是28x28
                batch_imgs.append(self.apply_transforms(img_reshaped))
            imgs = np.stack(batch_imgs)
        else:
            # 单个图像处理
            if imgs.shape == (28, 28):
                img_reshaped = imgs
            else:
                img_reshaped = imgs.reshape(28, 28)
            imgs = self.apply_transforms(img_reshaped)
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