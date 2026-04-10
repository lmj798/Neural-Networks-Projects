import numpy as np
import pytest

from nn import Linear, Sequential, ReLUModule
from optimizers import Adam, SGD
from lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR
from grad_utils import clip_grad_norm, clip_grad_value, get_grad_stats
from checkpoint import save_checkpoint, load_checkpoint, save_model, load_model
from progress import ProgressBar, format_time, TrainingLogger
from config import TrainingConfig
from tensor import Tensor


def test_step_lr():
    """测试 StepLR 调度器"""
    model = Linear(10, 5)
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 初始学习率
    assert optimizer.lr == 0.01
    
    # 第 5 个 epoch 后应该衰减
    for i in range(5):
        scheduler.step(epoch=i+1)
    
    assert abs(optimizer.lr - 0.001) < 1e-8


def test_cosine_annealing_lr():
    """测试 CosineAnnealingLR 调度器"""
    model = Linear(10, 5)
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    
    # 初始学习率
    initial_lr = optimizer.lr
    assert initial_lr == 0.01
    
    # 在 T_max/2 时学习率应该接近中间值
    for i in range(5):
        scheduler.step(epoch=i+1)
    
    # 学习率应该在 eta_min 和 initial_lr 之间
    assert 0.0001 <= optimizer.lr <= 0.01


def test_multistep_lr():
    """测试 MultiStepLR 调度器"""
    model = Linear(10, 5)
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    
    # 第 5 个 epoch 后应该衰减
    scheduler.step(epoch=5)
    assert abs(optimizer.lr - 0.001) < 1e-8
    
    # 第 10 个 epoch 后应该再次衰减
    scheduler.step(epoch=10)
    assert abs(optimizer.lr - 0.0001) < 1e-8


def test_clip_grad_norm():
    """测试梯度范数裁剪"""
    model = Sequential(
        Linear(10, 5),
        ReLUModule(),
        Linear(5, 2)
    )
    
    # 手动设置大梯度
    for param in model.parameters():
        param.grad = Tensor(np.ones_like(param.realize_cached_data()) * 10.0)
    
    # 裁剪梯度
    norm = clip_grad_norm(model, max_norm=1.0)
    
    # 范数应该被裁剪到 1.0 附近
    assert norm > 1.0
    
    # 裁剪后的梯度范数应该接近 1.0
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_data = param.grad.realize_cached_data()
            total_norm += np.sum(grad_data ** 2)
    total_norm = np.sqrt(total_norm)
    assert total_norm <= 1.0 + 1e-6


def test_clip_grad_value():
    """测试梯度值裁剪"""
    model = Linear(10, 5)
    
    # 手动设置大梯度
    for param in model.parameters():
        param.grad = Tensor(np.ones_like(param.realize_cached_data()) * 10.0)
    
    # 裁剪梯度
    clip_grad_value(model, clip_value=1.0)
    
    # 所有梯度值应该在 [-1, 1] 范围内
    for param in model.parameters():
        if param.grad is not None:
            grad_data = param.grad.realize_cached_data()
            assert np.all(grad_data <= 1.0)
            assert np.all(grad_data >= -1.0)


def test_get_grad_stats():
    """测试梯度统计"""
    model = Linear(10, 5)
    
    # 手动设置梯度
    for param in model.parameters():
        param.grad = Tensor(np.ones_like(param.realize_cached_data()))
    
    stats = get_grad_stats(model)
    
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert "std" in stats
    assert "norm" in stats
    
    assert stats["min"] == 1.0
    assert stats["max"] == 1.0
    assert stats["mean"] == 1.0


def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    import tempfile
    import os
    
    model = Linear(10, 5)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        filepath = f.name
    
    try:
        # 保存检查点
        save_checkpoint(
            filepath,
            model,
            optimizer=optimizer,
            epoch=5,
            metrics={"loss": 0.5, "acc": 0.8}
        )
        
        # 加载检查点
        info = load_checkpoint(filepath, model, optimizer=optimizer)
        
        assert info["epoch"] == 5
        assert info["metrics"]["loss"] == 0.5
        assert info["metrics"]["acc"] == 0.8
        assert optimizer.lr == 0.01
    finally:
        # 清理临时文件
        if os.path.exists(filepath):
            os.remove(filepath)


def test_progress_bar():
    """测试进度条"""
    import io
    import sys
    
    # 捕获输出
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        progress = ProgressBar(total=10, desc="Test")
        for i in range(10):
            progress.update()
        
        output = sys.stdout.getvalue()
        assert "Test" in output
        assert "10/10" in output
        assert "100.0%" in output
    finally:
        sys.stdout = old_stdout


def test_format_time():
    """测试时间格式化"""
    assert format_time(30) == "30.0s"
    assert format_time(90) == "1m 30s"
    assert format_time(3661) == "1h 1m"


def test_training_logger():
    """测试训练日志记录器"""
    logger = TrainingLogger(log_every=1)
    
    logger.log_batch(0, loss=0.5, accuracy=0.8, lr=0.01)
    logger.log_batch(1, loss=0.4, accuracy=0.85, lr=0.01)
    
    history = logger.get_history()
    
    assert len(history["loss"]) == 2
    assert len(history["accuracy"]) == 2
    assert len(history["lr"]) == 2


def test_training_config():
    """测试训练配置"""
    config = TrainingConfig(
        epochs=10,
        batch_size=64,
        lr=0.001,
        use_scheduler=True,
        scheduler_type="step",
        use_grad_clip=True
    )
    
    # 转换为字典
    config_dict = config.to_dict()
    assert config_dict["epochs"] == 10
    assert config_dict["batch_size"] == 64
    
    # 从字典创建
    config2 = TrainingConfig.from_dict(config_dict)
    assert config2.epochs == 10
    assert config2.batch_size == 64
    
    # 验证配置
    errors = config.validate()
    assert len(errors) == 0


def test_training_config_validation():
    """测试训练配置验证"""
    config = TrainingConfig(
        epochs=-1,  # 无效
        batch_size=0,  # 无效
        lr=-0.001  # 无效
    )
    
    errors = config.validate()
    assert len(errors) > 0
    assert "epochs must be positive" in errors
    assert "batch_size must be positive" in errors
    assert "lr must be positive" in errors
