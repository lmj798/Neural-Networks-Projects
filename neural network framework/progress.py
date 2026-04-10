import sys
from typing import Optional


class ProgressBar:
    """训练进度条"""
    
    def __init__(self, total: int, desc: str = "Progress", width: int = 40):
        """初始化进度条
        
        Args:
            total: 总数量
            desc: 描述信息
            width: 进度条宽度
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
    
    def update(self, n: int = 1):
        """更新进度
        
        Args:
            n: 更新的数量
        """
        self.current += n
        self.display()
    
    def display(self):
        """显示进度条"""
        if self.total == 0:
            return
        
        fraction = self.current / self.total
        filled = int(self.width * fraction)
        remaining = self.width - filled
        
        bar = "=" * filled + ">" + "." * remaining
        percent = fraction * 100
        
        sys.stdout.write(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({percent:.1f}%)")
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    def set_description(self, desc: str):
        """设置描述信息
        
        Args:
            desc: 新的描述信息
        """
        self.desc = desc
        self.display()
    
    def finish(self):
        """完成进度条"""
        self.current = self.total
        self.display()


def format_time(seconds: float) -> str:
    """格式化时间为可读字符串
    
    Args:
        seconds: 秒数
    
    Returns:
        str: 格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_every: int = 1):
        """初始化训练日志记录器
        
        Args:
            log_every: 每隔多少个 batch 记录一次日志
        """
        self.log_every = log_every
        self.history = {
            "loss": [],
            "accuracy": [],
            "lr": []
        }
    
    def log_batch(self, batch_idx: int, loss: float, accuracy: Optional[float] = None, 
                  lr: Optional[float] = None):
        """记录 batch 日志
        
        Args:
            batch_idx: batch 索引
            loss: 损失值
            accuracy: 准确率（可选）
            lr: 学习率（可选）
        """
        if batch_idx % self.log_every == 0:
            msg = f"Batch {batch_idx}, Loss: {loss:.4f}"
            if accuracy is not None:
                msg += f", Acc: {accuracy:.4f}"
            if lr is not None:
                msg += f", LR: {lr:.6f}"
            print(msg)
        
        self.history["loss"].append(loss)
        if accuracy is not None:
            self.history["accuracy"].append(accuracy)
        if lr is not None:
            self.history["lr"].append(lr)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: Optional[float] = None, val_acc: Optional[float] = None):
        """记录 epoch 日志
        
        Args:
            epoch: epoch 索引
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失（可选）
            val_acc: 验证准确率（可选）
        """
        msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        if val_loss is not None:
            msg += f", Val Loss: {val_loss:.4f}"
        if val_acc is not None:
            msg += f", Val Acc: {val_acc:.4f}"
        print(msg)
    
    def get_history(self) -> dict:
        """获取历史记录
        
        Returns:
            dict: 历史记录
        """
        return self.history
