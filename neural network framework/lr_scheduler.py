from abc import ABC, abstractmethod
from typing import List
import numpy

from tensor import Tensor


class LRScheduler(ABC):
    """学习率调度器基类"""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        """初始化学习率调度器
        
        Args:
            optimizer: 优化器
            last_epoch: 上一个 epoch 的索引
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [optimizer.lr]
    
    @abstractmethod
    def get_lr(self) -> List[float]:
        """获取新的学习率
        
        Returns:
            List[float]: 新的学习率列表
        """
        pass
    
    def step(self, epoch: int = None):
        """更新学习率
        
        Args:
            epoch: epoch 索引，默认为 last_epoch + 1
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        lrs = self.get_lr()
        for param_group, lr in zip([self.optimizer], lrs):
            param_group.lr = lr


class StepLR(LRScheduler):
    """步进式学习率衰减器
    
    每 step_size 个 epoch，学习率衰减 gamma 倍
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        """初始化步进式学习率衰减器
        
        Args:
            optimizer: 优化器
            step_size: 学习率衰减的间隔 epoch 数
            gamma: 学习率衰减系数
            last_epoch: 上一个 epoch 的索引
        """
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取新的学习率
        
        Returns:
            List[float]: 新的学习率
        """
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """余弦退火学习率调度器
    
    使用余弦函数将学习率从初始值降低到 eta_min
    """
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        """初始化余弦退火学习率调度器
        
        Args:
            optimizer: 优化器
            T_max: 最大迭代次数（半周期）
            eta_min: 最小学习率
            last_epoch: 上一个 epoch 的索引
        """
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        if eta_min < 0:
            raise ValueError("eta_min must be non-negative")
        
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取新的学习率
        
        Returns:
            List[float]: 新的学习率
        """
        if self.last_epoch == 0:
            return self.base_lrs
        
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 - numpy.cos(numpy.pi / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + numpy.cos(numpy.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    """多步进式学习率衰减器
    
    在指定的 milestones 处衰减学习率
    """
    
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        """初始化多步进式学习率衰减器
        
        Args:
            optimizer: 优化器
            milestones: 学习率衰减的 epoch 列表
            gamma: 学习率衰减系数
            last_epoch: 上一个 epoch 的索引
        """
        if not milestones:
            raise ValueError("milestones must be non-empty")
        if not all(isinstance(m, int) and m > 0 for m in milestones):
            raise ValueError("milestones must be positive integers")
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """获取新的学习率
        
        Returns:
            List[float]: 新的学习率
        """
        milestone_count = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [base_lr * (self.gamma ** milestone_count) for base_lr in self.base_lrs]
