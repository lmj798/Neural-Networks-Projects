"""
高级训练功能使用示例

本示例展示了如何使用框架的高级功能：
- 学习率调度器
- 梯度裁剪
- 检查点保存/加载
- 训练进度条
- 训练配置类
"""

import numpy as np

from data import DataLoader, Dataset
from nn import Linear, Sequential, ReLUModule, Flatten
from optimizers import Adam
from lr_scheduler import StepLR, CosineAnnealingLR
from grad_utils import clip_grad_norm
from checkpoint import save_checkpoint, load_checkpoint
from progress import ProgressBar, TrainingLogger
from config import TrainingConfig
from tensor import Tensor
from trainer import Trainer, split_train_val


def create_dummy_data(num_samples=1000, num_classes=10):
    """创建示例数据"""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (num_samples, 28, 28)).astype(np.float32)
    y = rng.integers(0, num_classes, num_samples).astype(np.int64)
    return X, y


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("基础使用示例")
    print("=" * 60)
    
    # 创建配置
    config = TrainingConfig(
        epochs=5,
        batch_size=64,
        lr=0.01,
        use_scheduler=True,
        scheduler_type="step",
        scheduler_step_size=2,
        scheduler_gamma=0.5,
        use_grad_clip=True,
        grad_clip_max_norm=1.0
    )
    
    # 创建数据
    X, y = create_dummy_data(500, 10)
    train_X, train_y, val_X, val_y = split_train_val(X, y, val_size=100)
    
    train_dataset = Dataset(train_X, train_y)
    val_dataset = Dataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 创建模型
    model = Sequential(
        Flatten(),
        Linear(784, 128),
        ReLUModule(),
        Linear(128, 10)
    )
    
    # 创建优化器
    optimizer = Adam(model.parameters(), lr=config.lr)
    
    # 创建学习率调度器
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    
    # 创建训练器
    trainer = Trainer(model, optimizer)
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(config.epochs):
        # 训练一个 epoch
        train_loss, train_acc = trainer._run_epoch(train_loader, training=True, log_every=10)
        
        # 验证
        val_loss, val_acc = trainer._run_epoch(val_loader, training=False)
        
        # 更新学习率
        scheduler.step(epoch=epoch + 1)
        
        # 打印日志
        print(f"Epoch {epoch+1}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.lr:.6f}")
        
        # 梯度裁剪
        if config.use_grad_clip:
            grad_norm = clip_grad_norm(model, max_norm=config.grad_clip_max_norm)
            print(f"  Gradient Norm: {grad_norm:.4f}")
    
    print("\n训练完成！")


def example_checkpoint_usage():
    """检查点使用示例"""
    print("\n" + "=" * 60)
    print("检查点使用示例")
    print("=" * 60)
    
    # 创建模型和优化器
    model = Sequential(
        Flatten(),
        Linear(784, 64),
        ReLUModule(),
        Linear(64, 10)
    )
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 保存检查点
    print("\n保存检查点...")
    save_checkpoint(
        "example_checkpoint.npz",
        model,
        optimizer=optimizer,
        epoch=5,
        metrics={"loss": 0.5, "accuracy": 0.85}
    )
    print("检查点已保存到：example_checkpoint.npz")
    
    # 加载检查点
    print("\n加载检查点...")
    info = load_checkpoint("example_checkpoint.npz", model, optimizer=optimizer)
    print(f"加载的检查点信息：")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Metrics: {info['metrics']}")
    print(f"  Optimizer LR: {optimizer.lr}")
    
    # 清理
    import os
    if os.path.exists("example_checkpoint.npz"):
        os.remove("example_checkpoint.npz")
        print("\n已清理示例文件")


def example_progress_bar():
    """进度条使用示例"""
    print("\n" + "=" * 60)
    print("进度条使用示例")
    print("=" * 60)
    
    progress = ProgressBar(total=20, desc="Training", width=50)
    
    for i in range(20):
        # 模拟训练
        import time
        time.sleep(0.05)
        progress.update()
    
    print("训练完成！")


def example_training_logger():
    """训练日志记录器示例"""
    print("\n" + "=" * 60)
    print("训练日志记录器示例")
    print("=" * 60)
    
    logger = TrainingLogger(log_every=1)
    
    # 模拟训练日志
    for epoch in range(5):
        train_loss = 1.0 / (epoch + 1)
        train_acc = 1.0 - 0.5 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        val_acc = 1.0 - 0.4 / (epoch + 1)
        
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc
        )
    
    # 获取历史记录
    history = logger.get_history()
    print(f"\n记录了 {len(history['loss'])} 个 epoch 的数据")


def example_config_usage():
    """配置类使用示例"""
    print("\n" + "=" * 60)
    print("配置类使用示例")
    print("=" * 60)
    
    # 创建配置
    config = TrainingConfig(
        epochs=10,
        batch_size=128,
        lr=0.001,
        optimizer="adam",
        use_scheduler=True,
        scheduler_type="cosine",
        scheduler_T_max=10,
        use_grad_clip=True,
        grad_clip_max_norm=1.0,
        verbose=True
    )
    
    # 验证配置
    errors = config.validate()
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("配置验证通过！")
    
    # 转换为字典
    config_dict = config.to_dict()
    print(f"\n配置字典包含 {len(config_dict)} 个键")
    
    # 从字典创建配置
    config2 = TrainingConfig.from_dict(config_dict)
    print(f"从字典创建配置成功：epochs={config2.epochs}, batch_size={config2.batch_size}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("高级训练功能使用示例")
    print("=" * 60)
    
    # 运行示例
    example_basic_usage()
    example_checkpoint_usage()
    example_progress_bar()
    example_training_logger()
    example_config_usage()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
