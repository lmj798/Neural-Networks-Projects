import numpy as np
from typing import Optional, Tuple


class Transform:
    """数据变换基类"""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用变换
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: 变换后的数据
        """
        raise NotImplementedError


class Normalize(Transform):
    """归一化变换"""
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        """初始化归一化变换
        
        Args:
            mean: 均值
            std: 标准差
        """
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用归一化
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: 归一化后的数据
        """
        return (x - self.mean) / self.std


class RandomHorizontalFlip(Transform):
    """随机水平翻转"""
    def __init__(self, p: float = 0.5):
        """初始化随机水平翻转
        
        Args:
            p: 翻转概率
        """
        self.p = p
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用随机水平翻转
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: 翻转后的数据
        """
        if np.random.random() < self.p:
            return np.fliplr(x)
        return x


class RandomCrop(Transform):
    """随机裁剪"""
    def __init__(self, size: Tuple[int, int], padding: Optional[int] = 0):
        """初始化随机裁剪
        
        Args:
            size: 裁剪尺寸
            padding: 填充大小
        """
        self.size = size
        self.padding = padding
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用随机裁剪
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: 裁剪后的数据
        """
        h, w = x.shape[:2]
        new_h, new_w = self.size
        
        if self.padding > 0:
            x = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
            h, w = x.shape[:2]
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        return x[top:top+new_h, left:left+new_w]


class Resize(Transform):
    """ resize 变换"""
    def __init__(self, size: Tuple[int, int]):
        """初始化 resize 变换
        
        Args:
            size: 目标尺寸
        """
        self.size = size
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用 resize
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: resize 后的数据
        """
        # 使用最近邻插值
        h, w = x.shape[:2]
        new_h, new_w = self.size
        
        if len(x.shape) == 3:
            new_x = np.zeros((new_h, new_w, x.shape[2]), dtype=x.dtype)
        else:
            new_x = np.zeros((new_h, new_w), dtype=x.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                src_i = min(int(i * h / new_h), h - 1)
                src_j = min(int(j * w / new_w), w - 1)
                new_x[i, j] = x[src_i, src_j]
        
        return new_x


class Compose(Transform):
    """组合多个变换"""
    def __init__(self, transforms: list):
        """初始化组合变换
        
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用组合变换
        
        Args:
            x: 输入数据
        
        Returns:
            np.ndarray: 变换后的数据
        """
        for t in self.transforms:
            x = t(x)
        return x
