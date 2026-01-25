import argparse
import gzip
import os
import struct
import time
from urllib.request import urlretrieve

import numpy as np

from tensor import Tensor
from nn import Sequential, Linear, ReLUModule, Flatten
from optimizers import Adam
from data import Dataset, DataLoader

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(logits, targets):
    num_samples = logits.shape[0]
    log_probs = np.log(softmax(logits) + 1e-8)
    return -np.sum(log_probs[np.arange(num_samples), targets]) / num_samples

def accuracy(logits, targets):
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == targets)

class SimpleMNISTNet(Sequential):
    def __init__(self):
        super().__init__(
            Flatten(),
            Linear(784, 256),
            ReLUModule(),
            Linear(256, 128),
            ReLUModule(),
            Linear(128, 10)
        )

def download_mnist_data():
    """下载MNIST数据集"""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz', 
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('mnist_data', exist_ok=True)
    
    for name, filename in files.items():
        filepath = os.path.join('mnist_data', filename)
        if not os.path.exists(filepath):
            print(f"下载 {filename}...")
            urlretrieve(base_url + filename, filepath)
            print(f"下载完成: {filename}")
        else:
            print(f"文件已存在: {filename}")
    
    return files

def load_mnist_data(files):
    """加载MNIST数据集"""
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) / 255.0
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    train_images = load_images(os.path.join('mnist_data', files['train_images']))
    train_labels = load_labels(os.path.join('mnist_data', files['train_labels']))
    test_images = load_images(os.path.join('mnist_data', files['test_images']))
    test_labels = load_labels(os.path.join('mnist_data', files['test_labels']))
    
    return train_images, train_labels, test_images, test_labels

def generate_dummy_mnist_data(num_samples=1000, num_classes=10):
    """生成模拟MNIST数据用于测试"""
    np.random.seed(42)
    
    # 生成随机图像数据 (28x28 = 784)
    X = np.random.randn(num_samples, 28, 28) * 0.5
    # 添加一些结构使数据更真实
    for i in range(num_samples):
        digit = i % num_classes
        # 在中心区域添加数字模式
        center = 14
        for j in range(5):
            for k in range(5):
                if digit == 0 or (digit == 1 and k == 2):
                    X[i, center+j-2, center+k-2] += 1.0
                elif digit == 2:
                    if j == k or j + k == 4:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 3:
                    if k == 2 or j == 0 or j == 4:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 4:
                    if j == 0 or j == 2 or (j == 1 and k == 0):
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 5:
                    if k == 0 or k == 4 or j == 2:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 6:
                    if k == 0 or j == 2 or j == 4:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 7:
                    if j == k or k == 0:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 8:
                    if j == 2 or k == 2 or j == k or j + k == 4:
                        X[i, center+j-2, center+k-2] += 1.0
                elif digit == 9:
                    if j == 2 or (j == 0 and k == 2) or (j == 4 and k == 2):
                        X[i, center+j-2, center+k-2] += 1.0
    
    # 生成标签
    y = np.arange(num_samples) % num_classes
    
    # 添加噪声
    X += np.random.normal(0, 0.1, X.shape)
    
    return X, y

def train_mnist(
    *,
    seed: int = 42,
    num_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    train_subset_size: int = 5000,
    test_subset_size: int = 1000,
):
    np.random.seed(seed)

    print("开始MNIST神经网络训练测试...")
    print("=" * 50)

    # 下载并加载真实MNIST数据
    print("下载MNIST数据集...")
    files = download_mnist_data()
    print("加载MNIST数据集...")
    train_X, train_y, test_X, test_y = load_mnist_data(files)

    print(f"训练数据形状: {train_X.shape}, 标签形状: {train_y.shape}")
    print(f"测试数据形状: {test_X.shape}, 标签形状: {test_y.shape}")

    # 使用部分数据以加快训练
    train_subset_size = min(train_subset_size, len(train_X))
    test_subset_size = min(test_subset_size, len(test_X))

    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]

    train_dataset = Dataset(train_X_subset, train_y_subset)
    test_dataset = Dataset(test_X_subset, test_y_subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("创建神经网络模型...")
    model = SimpleMNISTNet()

    # 创建优化器
    optimizer = Adam(model.parameters(), lr=lr)
    
    print("开始训练...")
    print("-" * 50)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        start_time = time.time()
        
        # 训练循环
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            logits = model(data)
            
            # 确保标签是整数类型
            target_data = target.realize_cached_data()
            if target_data.dtype != np.int64:
                target_data = target_data.astype(np.int64)
            
            # 直接对logits进行反向传播，手动计算梯度
            # 这样梯度会正确传播到网络参数
            
            # 反向传播
            optimizer.zero_grad()
            
            # 计算损失用于显示
            probs = softmax(logits.realize_cached_data())
            log_probs = np.log(probs + 1e-8)
            loss_data = -np.sum(log_probs[np.arange(logits.shape[0]), target_data]) / logits.shape[0]
            
            # 手动计算logits的梯度（基于交叉熵损失的导数）
            grad_logits = probs.copy()
            grad_logits[np.arange(logits.shape[0]), target_data] -= 1.0
            grad_logits /= logits.shape[0]
            
            # 将梯度传播回logits
            logits.backward(Tensor(grad_logits))
            
            optimizer.step()
            
            # 统计
            epoch_loss += loss_data
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            target_data = target.realize_cached_data()
            epoch_correct += np.sum(pred == target_data)
            epoch_total += len(target_data)
            
            if batch_idx % 100 == 0:  # 减少输出频率
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss_data:.4f}")
        
        # 计算训练准确率
        train_accuracy = epoch_correct / epoch_total
        num_batches = len(train_dataset) // train_loader.batch_size + (1 if len(train_dataset) % train_loader.batch_size else 0)
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        for data, target in test_loader:
            logits = model(data)
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            target_data = target.realize_cached_data()
            test_correct += np.sum(pred == target_data)
            test_total += len(target_data)
        
        test_accuracy = test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} 完成:")
        print(f"  训练损失: {avg_loss:.4f}")
        print(f"  训练准确率: {train_accuracy:.4f}")
        print(f"  测试准确率: {test_accuracy:.4f}")
        print(f"  用时: {epoch_time:.2f}秒")
        print("-" * 50)
    
    print("训练完成!")
    print("=" * 50)
    
    # 最终测试
    print("最终模型评估...")
    model.eval()
    final_correct = 0
    final_total = 0
    
    for data, target in test_loader:
        logits = model(data)
        pred = np.argmax(logits.realize_cached_data(), axis=1)
        target_data = target.realize_cached_data()
        final_correct += np.sum(pred == target_data)
        final_total += len(target_data)
    
    final_accuracy = final_correct / final_total
    print(f"最终测试准确率: {final_accuracy:.4f}")
    
    # 保存模型参数（可选）
    print("保存模型参数...")
    try:
        params = model.parameters()
        param_data = {}
        for i, param in enumerate(params):
            param_data[f'param_{i}'] = param.realize_cached_data()
        np.savez('mnist_model_params.npz', **param_data)
        print("模型参数已保存到 mnist_model_params.npz")
    except Exception as e:
        print(f"保存模型参数时出错: {e}")
    
    return train_losses, train_accuracies, test_accuracies

def test_individual_components():
    """测试各个组件是否正常工作"""
    print("测试各个组件...")
    print("-" * 30)
    
    # 测试Tensor基本操作
    print("1. 测试Tensor基本操作")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    print(f"   {a.realize_cached_data()} + {b.realize_cached_data()} = {c.realize_cached_data()}")
    
    # 测试矩阵乘法
    print("2. 测试矩阵乘法")
    x = Tensor([[1, 2], [3, 4]])
    w = Tensor([[5, 6], [7, 8]])
    y = x @ w
    print(f"   矩阵乘法结果形状: {y.shape}")
    
    # 测试神经网络层
    print("3. 测试神经网络层")
    linear = Linear(2, 4)
    input_tensor = Tensor([[1, 2]])
    output = linear(input_tensor)
    print(f"   Linear层输出形状: {output.shape}")
    
    # 测试激活函数
    print("4. 测试激活函数")
    relu_layer = ReLUModule()
    relu_output = relu_layer(Tensor([[-1, 2, -3, 4]]))
    print(f"   ReLU输出: {relu_output.realize_cached_data()}")
    
    # 测试优化器
    print("5. 测试优化器")
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adam([param], lr=0.01)
    
    # 模拟梯度计算
    loss = Tensor(5.0, requires_grad=True)
    param.grad = Tensor([0.1, 0.2])
    optimizer.step()
    print(f"   优化后参数: {param.realize_cached_data()}")
    
    print("所有组件测试完成!")
    print("=" * 30)

def test_mnist_training():
    """测试MNIST完整训练过程"""
    try:
        train_losses, train_accuracies, test_accuracies = train_mnist()

        # 验证训练是否成功
        assert len(train_losses) == 5, f"Expected 5 epochs, got {len(train_losses)}"
        assert len(train_accuracies) == 5, f"Expected 5 training accuracies, got {len(train_accuracies)}"
        assert len(test_accuracies) == 5, f"Expected 5 test accuracies, got {len(test_accuracies)}"

        # 验证准确率合理（应该比随机猜测10%要好）
        final_test_accuracy = test_accuracies[-1]
        assert final_test_accuracy > 0.8, f"Final test accuracy too low: {final_test_accuracy}"

        print(f"PASS MNIST训练测试通过！最终测试准确率: {final_test_accuracy:.4f}")

    except Exception as e:
        print(f"FAIL MNIST训练测试失败: {e}")
        raise


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-subset", type=int, default=5000)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--skip-component-tests", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.skip_component_tests:
        test_individual_components()
        print("\n")

    # 进行MNIST训练
    try:
        train_losses, train_accuracies, test_accuracies = train_mnist(
            seed=args.seed,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_subset_size=args.train_subset,
            test_subset_size=args.test_subset,
        )

        print("\nMNIST训练总结:")
        print(f"最终训练准确率: {train_accuracies[-1]:.4f}")
        print(f"最终测试准确率: {test_accuracies[-1]:.4f}")
        print("MNIST神经网络训练测试成功完成!")

    except Exception as e:
        print(f"MNIST训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()