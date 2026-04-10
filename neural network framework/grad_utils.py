from typing import Union
import numpy

from tensor import Tensor


def clip_grad_norm(model, max_norm: float, norm_type: float = 2.0) -> float:
    """裁剪梯度的范数
    
    Args:
        model: 模型
        max_norm: 最大范数
        norm_type: 范数类型（2 为 L2 范数，inf 为无穷范数）
    
    Returns:
        float: 实际梯度范数
    """
    parameters = model.parameters()
    
    # 收集所有梯度
    grads = []
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad.realize_cached_data()
            grads.append(grad_data.flatten())
    
    if not grads:
        return 0.0
    
    # 计算总范数
    if norm_type == numpy.inf:
        total_norm = max(numpy.max(numpy.abs(grad)) for grad in grads)
    elif norm_type == -numpy.inf:
        total_norm = min(numpy.min(numpy.abs(grad)) for grad in grads)
    else:
        total_norm = numpy.sqrt(sum(numpy.sum(grad ** 2) for grad in grads))
    
    # 如果需要裁剪
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in parameters:
            if param.grad is not None:
                grad_data = param.grad.realize_cached_data()
                param.grad.cached_data = grad_data * clip_coef
    
    return float(total_norm)


def clip_grad_value(model, clip_value: float):
    """裁剪梯度值
    
    Args:
        model: 模型
        clip_value: 裁剪阈值
    """
    if clip_value == 0:
        raise ValueError("clip_value cannot be zero")
    
    parameters = model.parameters()
    
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad.realize_cached_data()
            numpy.clip(grad_data, -clip_value, clip_value, out=grad_data)
            param.grad.cached_data = grad_data


def get_grad_stats(model) -> dict:
    """获取梯度统计信息
    
    Args:
        model: 模型
    
    Returns:
        dict: 梯度统计信息
    """
    parameters = model.parameters()
    grads = []
    
    for param in parameters:
        if param.grad is not None:
            grad_data = param.grad.realize_cached_data()
            grads.append(grad_data)
    
    if not grads:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "norm": 0.0
        }
    
    all_grads = numpy.concatenate([grad.flatten() for grad in grads])
    
    return {
        "min": float(numpy.min(all_grads)),
        "max": float(numpy.max(all_grads)),
        "mean": float(numpy.mean(all_grads)),
        "std": float(numpy.std(all_grads)),
        "norm": float(numpy.sqrt(numpy.sum(all_grads ** 2)))
    }
