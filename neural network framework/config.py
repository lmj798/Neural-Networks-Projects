from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 基本配置
    seed: int = 42
    epochs: int = 10
    batch_size: int = 128
    
    # 优化器配置
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    
    # 学习率调度器配置
    use_scheduler: bool = False
    scheduler_type: str = "step"  # step, cosine, multistep
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.1
    scheduler_T_max: int = 10
    scheduler_eta_min: float = 0.0
    scheduler_milestones: List[int] = field(default_factory=lambda: [5, 10, 15])
    
    # 梯度裁剪配置
    use_grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    grad_clip_norm_type: float = 2.0
    
    # 数据配置
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    test_subset_size: Optional[int] = None
    num_workers: int = 0
    pin_memory: bool = False
    
    # 日志配置
    log_every: int = 10
    save_every: int = 1
    verbose: bool = True
    use_progress_bar: bool = True
    
    # 检查点配置
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    monitor_metric: str = "val_loss"  # val_loss, val_acc, train_loss, train_acc
    monitor_mode: str = "min"  # min, max
    
    # 设备配置
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
        
        Returns:
            TrainingConfig: 配置对象
        """
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """验证配置
        
        Returns:
            List[str]: 错误信息列表
        """
        errors = []
        
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.lr <= 0:
            errors.append("lr must be positive")
        
        if self.optimizer not in ["adam", "sgd"]:
            errors.append(f"Unknown optimizer: {self.optimizer}")
        
        if self.use_scheduler:
            if self.scheduler_type not in ["step", "cosine", "multistep"]:
                errors.append(f"Unknown scheduler type: {self.scheduler_type}")
            
            if self.scheduler_type == "step" and self.scheduler_step_size <= 0:
                errors.append("scheduler_step_size must be positive")
            
            if self.scheduler_type == "cosine" and self.scheduler_T_max <= 0:
                errors.append("scheduler_T_max must be positive")
        
        if self.use_grad_clip and self.grad_clip_max_norm <= 0:
            errors.append("grad_clip_max_norm must be positive")
        
        if self.monitor_metric not in ["val_loss", "val_acc", "train_loss", "train_acc"]:
            errors.append(f"Unknown monitor metric: {self.monitor_metric}")
        
        if self.monitor_mode not in ["min", "max"]:
            errors.append(f"Unknown monitor mode: {self.monitor_mode}")
        
        return errors
