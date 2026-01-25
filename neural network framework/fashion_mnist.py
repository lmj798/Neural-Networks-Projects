import numpy as np
import time
import os
import gzip
import struct
from urllib.request import urlretrieve
from tensor import Tensor
from nn import Sequential, Linear, ReLUModule, Flatten
from optimizers import Adam
from data import Dataset, DataLoader

def softmax(x):
    """Softmax激活函数"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def download_fashion_mnist_data():
    """下载Fashion-MNIST数据集"""
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz', 
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('fashion_mnist_data', exist_ok=True)
    
    for name, filename in files.items():
        filepath = os.path.join('fashion_mnist_data', filename)
        if not os.path.exists(filepath):
            print(f"下载 {filename}...")
            urlretrieve(base_url + filename, filepath)
            print(f"下载完成: {filename}")
        else:
            print(f"文件已存在: {filename}")
    
    return files

def load_fashion_mnist_data(files):
    """加载Fashion-MNIST数据集"""
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) / 255.0
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    train_images = load_images(os.path.join('fashion_mnist_data', files['train_images']))
    train_labels = load_labels(os.path.join('fashion_mnist_data', files['train_labels']))
    test_images = load_images(os.path.join('fashion_mnist_data', files['test_images']))
    test_labels = load_labels(os.path.join('fashion_mnist_data', files['test_labels']))
    
    return train_images, train_labels, test_images, test_labels

class FashionMNISTNet(Sequential):
    """Fashion-MNIST专用神经网络，比MNIST更深"""
    def __init__(self):
        super().__init__(
            Flatten(),
            Linear(784, 512),
            ReLUModule(),
            Linear(512, 256),
            ReLUModule(),
            Linear(256, 128),
            ReLUModule(),
            Linear(128, 10)
        )

def train_fashion_mnist(train_subset_size=5000, test_subset_size=1000, num_epochs=8, batch_size=128, learning_rate=0.001):
    """训练Fashion-MNIST神经网络"""
    print("开始Fashion-MNIST神经网络训练测试...")
    print("=" * 60)
    
    # 下载并加载Fashion-MNIST数据
    print("下载Fashion-MNIST数据集...")
    files = download_fashion_mnist_data()
    print("加载Fashion-MNIST数据集...")
    train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)
    
    print(f"训练数据形状: {train_X.shape}, 标签形状: {train_y.shape}")
    print(f"测试数据形状: {test_X.shape}, 标签形状: {test_y.shape}")
    
    # Fashion-MNIST类别标签
    fashion_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    print(f"数据集类别: {fashion_labels}")
    
    # 使用子集数据以加快训练
    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]
    
    print(f"使用训练子集: {train_X_subset.shape}")
    print(f"使用测试子集: {test_X_subset.shape}")
    
    # 创建数据集和数据加载器
    train_dataset = Dataset(train_X_subset, train_y_subset)
    test_dataset = Dataset(test_X_subset, test_y_subset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建更深的模型以应对更复杂的Fashion-MNIST
    print("创建更深的神经网络模型...")
    model = FashionMNISTNet()
    
    # 创建优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("开始训练...")
    print("-" * 60)
    
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
            
            if batch_idx % 50 == 0:  # 输出频率
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
        print("-" * 60)
    
    print("训练完成!")
    print("=" * 60)
    
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
    
    # 保存模型参数
    print("保存模型参数...")
    try:
        params = model.parameters()
        param_data = {}
        for i, param in enumerate(params):
            param_data[f'param_{i}'] = param.realize_cached_data()
        np.savez('fashion_mnist_model_params.npz', **param_data)
        print("模型参数已保存到 fashion_mnist_model_params.npz")
    except Exception as e:
        print(f"保存模型参数时出错: {e}")
    
    return train_losses, train_accuracies, test_accuracies, final_accuracy

def test_fashion_mnist_components():
    """测试Fashion-MNIST各个组件"""
    print("测试Fashion-MNIST组件...")
    print("-" * 40)
    
    # 测试数据下载
    print("1. 测试数据下载...")
    try:
        files = download_fashion_mnist_data()
        print("PASS 数据下载成功")
    except Exception as e:
        print(f"FAIL 数据下载失败: {e}")
        return False
    
    # 测试数据加载
    print("2. 测试数据加载...")
    try:
        train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)
        print(f"PASS 训练数据: {train_X.shape}, 标签: {train_y.shape}")
        print(f"PASS 测试数据: {test_X.shape}, 标签: {test_y.shape}")
    except Exception as e:
        print(f"FAIL 数据加载失败: {e}")
        return False
    
    # 测试模型创建
    print("3. 测试模型创建...")
    try:
        model = FashionMNISTNet()
        dummy_input = Tensor(np.random.randn(1, 28, 28))
        output = model(dummy_input)
        print(f"PASS 模型输出形状: {output.shape}")
    except Exception as e:
        print(f"FAIL 模型创建失败: {e}")
        return False
    
    print("所有组件测试通过!")
    print("-" * 40)
    return True

def test_fashion_mnist_training():
    """测试Fashion-MNIST完整训练过程"""
    try:
        train_losses, train_accuracies, test_accuracies, final_acc = train_fashion_mnist(
            train_subset_size=2000,  # 使用较小数据集以加快测试
            test_subset_size=500,
            num_epochs=3,  # 减少epochs以加快测试
            batch_size=128,
            learning_rate=0.001
        )
        
        # 验证训练是否成功
        assert len(train_losses) == 3, f"Expected 3 epochs, got {len(train_losses)}"
        assert len(train_accuracies) == 3, f"Expected 3 training accuracies, got {len(train_accuracies)}"
        assert len(test_accuracies) == 3, f"Expected 3 test accuracies, got {len(test_accuracies)}"
        
        # 验证准确率合理（Fashion-MNIST较难，但应该比随机猜测10%要好）
        assert final_acc > 0.6, f"Final test accuracy too low: {final_acc}"
        
        print(f"PASS Fashion-MNIST训练测试通过！最终测试准确率: {final_acc:.4f}")
        
    except Exception as e:
        print(f"FAIL Fashion-MNIST训练测试失败: {e}")
        raise

def benchmark_training():
    """基准测试：比较不同参数设置"""
    print("Fashion-MNIST训练基准测试")
    print("=" * 60)
    
    configs = [
        {"name": "快速训练", "train_size": 1000, "test_size": 500, "epochs": 3, "batch_size": 256, "lr": 0.01},
        {"name": "标准训练", "train_size": 5000, "test_size": 1000, "epochs": 8, "batch_size": 128, "lr": 0.001},
        {"name": "深度训练", "train_size": 10000, "test_size": 2000, "epochs": 10, "batch_size": 64, "lr": 0.001}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n开始 {config['name']}...")
        try:
            train_losses, train_accuracies, test_accuracies, final_acc = train_fashion_mnist(
                train_subset_size=config['train_size'],
                test_subset_size=config['test_size'],
                num_epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['lr']
            )
            
            result = {
                'name': config['name'],
                'final_train_acc': train_accuracies[-1],
                'final_test_acc': final_acc,
                'final_loss': train_losses[-1],
                'config': config
            }
            results.append(result)
            
            print(f"{config['name']} 完成 - 测试准确率: {final_acc:.4f}")
            
        except Exception as e:
            print(f"{config['name']} 失败: {e}")
    
    # 输出结果对比
    print("\n" + "=" * 60)
    print("基准测试结果对比:")
    print("-" * 60)
    for result in results:
        print(f"{result['name']:12} | 训练: {result['final_train_acc']:.4f} | 测试: {result['final_test_acc']:.4f} | 损失: {result['final_loss']:.4f}")
    
    return results

if __name__ == "__main__":
    # 测试组件
    if not test_fashion_mnist_components():
        print("组件测试失败，退出程序")
        exit(1)
    
    print("\n选择运行模式:")
    print("1. 标准训练 (推荐)")
    print("2. 基准测试 (比较不同配置)")
    print("3. 快速训练 (仅用于验证)")
    
    choice = input("请选择 (1/2/3, 默认为1): ").strip()
    
    if choice == "2":
        benchmark_training()
    elif choice == "3":
        # 快速训练
        train_fashion_mnist(train_subset_size=1000, test_subset_size=500, num_epochs=3, batch_size=256, learning_rate=0.01)
    else:
        # 标准训练
        train_fashion_mnist()
    
    print("\nFashion-MNIST训练完成!")