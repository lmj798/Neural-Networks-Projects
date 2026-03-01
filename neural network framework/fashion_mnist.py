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
from ops import softmax_cross_entropy

def softmax(x):

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def download_fashion_mnist_data():

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
            print(f"涓嬭浇 {filename}...")
            urlretrieve(base_url + filename, filepath)
            print(f"涓嬭浇瀹屾垚: {filename}")
        else:
            print(f"鏂囦欢宸插瓨鍦? {filename}")
    
    return files

def load_fashion_mnist_data(files):

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

    print("寮€濮婩ashion-MNIST绁炵粡缃戠粶璁粌娴嬭瘯...")
    print("=" * 60)
    
    # 涓嬭浇骞跺姞杞紽ashion-MNIST鏁版嵁
    print("涓嬭浇Fashion-MNIST鏁版嵁闆?..")
    files = download_fashion_mnist_data()
    print("鍔犺浇Fashion-MNIST鏁版嵁闆?..")
    train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)
    
    print(f"璁粌鏁版嵁褰㈢姸: {train_X.shape}, 鏍囩褰㈢姸: {train_y.shape}")
    print(f"娴嬭瘯鏁版嵁褰㈢姸: {test_X.shape}, 鏍囩褰㈢姸: {test_y.shape}")
    
    # Fashion-MNIST绫诲埆鏍囩
    fashion_labels = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    print(f"鏁版嵁闆嗙被鍒? {fashion_labels}")
    
    # 浣跨敤瀛愰泦鏁版嵁浠ュ姞蹇缁?
    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]
    
    print(f"浣跨敤璁粌瀛愰泦: {train_X_subset.shape}")
    print(f"浣跨敤娴嬭瘯瀛愰泦: {test_X_subset.shape}")
    
    # 鍒涘缓鏁版嵁闆嗗拰鏁版嵁鍔犺浇鍣?
    train_dataset = Dataset(train_X_subset, train_y_subset)
    test_dataset = Dataset(test_X_subset, test_y_subset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 鍒涘缓鏇存繁鐨勬ā鍨嬩互搴斿鏇村鏉傜殑Fashion-MNIST
    print("鍒涘缓鏇存繁鐨勭缁忕綉缁滄ā鍨?..")
    model = FashionMNISTNet()
    
    # 鍒涘缓浼樺寲鍣?
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print("寮€濮嬭缁?..")
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
        
        # 璁粌寰幆
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(data)
            target_data = target.realize_cached_data().astype(np.int64)
            target_indices = Tensor(target_data, dtype=np.int64, requires_grad=False)

            loss = softmax_cross_entropy(logits, target_indices)
            loss.backward()
            optimizer.step()

            loss_data = float(loss.realize_cached_data())
            epoch_loss += loss_data
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            epoch_correct += np.sum(pred == target_data)
            epoch_total += len(target_data)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss_data:.4f}")
        
        # 璁＄畻璁粌鍑嗙‘鐜?
        train_accuracy = epoch_correct / epoch_total
        num_batches = len(train_dataset) // train_loader.batch_size + (1 if len(train_dataset) % train_loader.batch_size else 0)
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # 娴嬭瘯
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
        
        print(f"Epoch {epoch+1}/{num_epochs} 瀹屾垚:")
        print(f"  璁粌鎹熷け: {avg_loss:.4f}")
        print(f"  璁粌鍑嗙‘鐜? {train_accuracy:.4f}")
        print(f"  娴嬭瘯鍑嗙‘鐜? {test_accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 60)
    
    print("璁粌瀹屾垚!")
    print("=" * 60)
    
    # 鏈€缁堟祴璇?
    print("鏈€缁堟ā鍨嬭瘎浼?..")
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
    print(f"鏈€缁堟祴璇曞噯纭巼: {final_accuracy:.4f}")
    
    # 淇濆瓨妯″瀷鍙傛暟
    print("淇濆瓨妯″瀷鍙傛暟...")
    try:
        params = model.parameters()
        param_data = {}
        for i, param in enumerate(params):
            param_data[f'param_{i}'] = param.realize_cached_data()
        np.savez('fashion_mnist_model_params.npz', **param_data)
        print("妯″瀷鍙傛暟宸蹭繚瀛樺埌 fashion_mnist_model_params.npz")
    except Exception as e:
        print(f"淇濆瓨妯″瀷鍙傛暟鏃跺嚭閿? {e}")
    
    return train_losses, train_accuracies, test_accuracies, final_accuracy

def test_fashion_mnist_components():

    print("娴嬭瘯Fashion-MNIST缁勪欢...")
    print("-" * 40)
    
    # 娴嬭瘯鏁版嵁涓嬭浇
    print("1. 娴嬭瘯鏁版嵁涓嬭浇...")
    try:
        files = download_fashion_mnist_data()
        print("PASS 鏁版嵁涓嬭浇鎴愬姛")
    except Exception as e:
        print(f"FAIL 鏁版嵁涓嬭浇澶辫触: {e}")
        return False
    
    # 娴嬭瘯鏁版嵁鍔犺浇
    print("2. 娴嬭瘯鏁版嵁鍔犺浇...")
    try:
        train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)
        print(f"PASS 璁粌鏁版嵁: {train_X.shape}, 鏍囩: {train_y.shape}")
        print(f"PASS 娴嬭瘯鏁版嵁: {test_X.shape}, 鏍囩: {test_y.shape}")
    except Exception as e:
        print(f"FAIL 鏁版嵁鍔犺浇澶辫触: {e}")
        return False
    
    # 娴嬭瘯妯″瀷鍒涘缓
    print("3. 娴嬭瘯妯″瀷鍒涘缓...")
    try:
        model = FashionMNISTNet()
        dummy_input = Tensor(np.random.randn(1, 28, 28))
        output = model(dummy_input)
        print(f"PASS 妯″瀷杈撳嚭褰㈢姸: {output.shape}")
    except Exception as e:
        print(f"FAIL 妯″瀷鍒涘缓澶辫触: {e}")
        return False
    
    print("鎵€鏈夌粍浠舵祴璇曢€氳繃!")
    print("-" * 40)
    return True

def test_fashion_mnist_training():

    try:
        train_losses, train_accuracies, test_accuracies, final_acc = train_fashion_mnist(
            train_subset_size=2000,  # 浣跨敤杈冨皬鏁版嵁闆嗕互鍔犲揩娴嬭瘯
            test_subset_size=500,
            num_epochs=3,  # 鍑忓皯epochs浠ュ姞蹇祴璇?
            batch_size=128,
            learning_rate=0.001
        )
        
        # 楠岃瘉璁粌鏄惁鎴愬姛
        assert len(train_losses) == 3, f"Expected 3 epochs, got {len(train_losses)}"
        assert len(train_accuracies) == 3, f"Expected 3 training accuracies, got {len(train_accuracies)}"
        assert len(test_accuracies) == 3, f"Expected 3 test accuracies, got {len(test_accuracies)}"
        
        # 楠岃瘉鍑嗙‘鐜囧悎鐞嗭紙Fashion-MNIST杈冮毦锛屼絾搴旇姣旈殢鏈虹寽娴?0%瑕佸ソ锛?
        assert final_acc > 0.6, f"Final test accuracy too low: {final_acc}"
        
        print(f"PASS Fashion-MNIST璁粌娴嬭瘯閫氳繃锛佹渶缁堟祴璇曞噯纭巼: {final_acc:.4f}")
        
    except Exception as e:
        print(f"FAIL Fashion-MNIST璁粌娴嬭瘯澶辫触: {e}")
        raise

def benchmark_training():

    print("Fashion-MNIST璁粌鍩哄噯娴嬭瘯")
    print("=" * 60)
    
    configs = [
        {"name": "quick", "train_size": 1000, "test_size": 500, "epochs": 3, "batch_size": 256, "lr": 0.01},
        {"name": "standard", "train_size": 5000, "test_size": 1000, "epochs": 8, "batch_size": 128, "lr": 0.001},
        {"name": "deep", "train_size": 10000, "test_size": 2000, "epochs": 10, "batch_size": 64, "lr": 0.001},
    ]

    results = []
    
    for config in configs:
        print(f"\n寮€濮?{config['name']}...")
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
            
            print(f"{config['name']} 瀹屾垚 - 娴嬭瘯鍑嗙‘鐜? {final_acc:.4f}")
            
        except Exception as e:
            print(f"{config['name']} 澶辫触: {e}")
    
    # 杈撳嚭缁撴灉瀵规瘮
    print("\n" + "=" * 60)
    print("鍩哄噯娴嬭瘯缁撴灉瀵规瘮:")
    print("-" * 60)
    for result in results:
        print(f"{result['name']:12} | 璁粌: {result['final_train_acc']:.4f} | 娴嬭瘯: {result['final_test_acc']:.4f} | 鎹熷け: {result['final_loss']:.4f}")
    
    return results

if __name__ == "__main__":
    if not test_fashion_mnist_components():
        raise SystemExit(1)

    train_fashion_mnist()
    print("Fashion-MNIST training complete.")
