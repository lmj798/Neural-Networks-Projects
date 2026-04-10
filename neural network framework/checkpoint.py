from typing import Dict, Any
import numpy
import os


def save_checkpoint(filepath: str, model, optimizer=None, scheduler=None, epoch: int = 0, 
                   metrics: Dict[str, float] = None) -> Dict[str, Any]:
    """保存训练检查点
    
    Args:
        filepath: 保存路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        epoch: 当前 epoch
        metrics: 评估指标（可选）
    
    Returns:
        Dict[str, Any]: 保存的字典
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {}
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = {
            "lr": optimizer.lr,
            "params": [p.realize_cached_data() for p in optimizer.params]
        }
        
        # 保存 Adam 优化器的特殊状态
        if hasattr(optimizer, "m"):
            checkpoint["optimizer_state_dict"]["m"] = optimizer.m
        if hasattr(optimizer, "v"):
            checkpoint["optimizer_state_dict"]["v"] = optimizer.v
        if hasattr(optimizer, "t"):
            checkpoint["optimizer_state_dict"]["t"] = optimizer.t
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = {
            "last_epoch": scheduler.last_epoch,
            "base_lrs": scheduler.base_lrs
        }
        
        # 保存特定调度器的参数
        if hasattr(scheduler, "step_size"):
            checkpoint["scheduler_state_dict"]["step_size"] = scheduler.step_size
        if hasattr(scheduler, "gamma"):
            checkpoint["scheduler_state_dict"]["gamma"] = scheduler.gamma
        if hasattr(scheduler, "T_max"):
            checkpoint["scheduler_state_dict"]["T_max"] = scheduler.T_max
        if hasattr(scheduler, "eta_min"):
            checkpoint["scheduler_state_dict"]["eta_min"] = scheduler.eta_min
        if hasattr(scheduler, "milestones"):
            checkpoint["scheduler_state_dict"]["milestones"] = scheduler.milestones
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    # 保存
    numpy.savez(filepath, **checkpoint)
    
    return checkpoint


def load_checkpoint(filepath: str, model, optimizer=None, scheduler=None) -> Dict[str, Any]:
    """加载训练检查点
    
    Args:
        filepath: 文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
    
    Returns:
        Dict[str, Any]: 加载的信息
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    data = numpy.load(filepath, allow_pickle=True)
    
    # 加载 epoch 和指标
    epoch = int(data["epoch"])
    metrics = data["metrics"].item() if "metrics" in data else {}
    
    # 加载模型参数
    model_state_dict = data["model_state_dict"].item()
    model.load_state_dict(model_state_dict, strict=True)
    
    # 加载优化器状态
    if optimizer is not None and "optimizer_state_dict" in data:
        opt_state = data["optimizer_state_dict"].item()
        optimizer.lr = opt_state["lr"]
        
        # 恢复 Adam 优化器的特殊状态
        if hasattr(optimizer, "m") and "m" in opt_state:
            optimizer.m = opt_state["m"]
        if hasattr(optimizer, "v") and "v" in opt_state:
            optimizer.v = opt_state["v"]
        if hasattr(optimizer, "t") and "t" in opt_state:
            optimizer.t = int(opt_state["t"])
    
    # 加载调度器状态
    if scheduler is not None and "scheduler_state_dict" in data:
        sched_state = data["scheduler_state_dict"].item()
        scheduler.last_epoch = int(sched_state["last_epoch"])
        scheduler.base_lrs = sched_state["base_lrs"]
        
        # 恢复特定调度器的参数
        if hasattr(scheduler, "step_size") and "step_size" in sched_state:
            scheduler.step_size = int(sched_state["step_size"])
        if hasattr(scheduler, "gamma") and "gamma" in sched_state:
            scheduler.gamma = float(sched_state["gamma"])
        if hasattr(scheduler, "T_max") and "T_max" in sched_state:
            scheduler.T_max = int(sched_state["T_max"])
        if hasattr(scheduler, "eta_min") and "eta_min" in sched_state:
            scheduler.eta_min = float(sched_state["eta_min"])
        if hasattr(scheduler, "milestones") and "milestones" in sched_state:
            scheduler.milestones = list(sched_state["milestones"])
    
    return {
        "epoch": epoch,
        "metrics": metrics
    }


def save_model(filepath: str, model):
    """只保存模型参数
    
    Args:
        filepath: 保存路径
        model: 模型
    """
    state_dict = model.state_dict()
    numpy.savez(filepath, **state_dict)


def load_model(filepath: str, model):
    """只加载模型参数
    
    Args:
        filepath: 文件路径
        model: 模型
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    data = numpy.load(filepath, allow_pickle=True)
    state_dict = {key: data[key] for key in data.files}
    model.load_state_dict(state_dict, strict=True)
