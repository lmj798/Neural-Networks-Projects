import gzip
import os
import struct
import time
from urllib.request import urlretrieve

import numpy as np

from data import DataLoader, Dataset
from nn import Flatten, Linear, ReLUModule, Sequential
from optimizers import Adam
from tensor import Tensor
from trainer import Trainer, split_train_val


def download_fashion_mnist_data():
    base_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs("fashion_mnist_data", exist_ok=True)

    for _, filename in files.items():
        filepath = os.path.join("fashion_mnist_data", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filepath)
            print(f"Downloaded: {filename}")
        else:
            print(f"Already exists: {filename}")

    return files


def load_fashion_mnist_data(files):
    def load_images(filename):
        with gzip.open(filename, "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(filename, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = load_images(os.path.join("fashion_mnist_data", files["train_images"]))
    train_labels = load_labels(os.path.join("fashion_mnist_data", files["train_labels"]))
    test_images = load_images(os.path.join("fashion_mnist_data", files["test_images"]))
    test_labels = load_labels(os.path.join("fashion_mnist_data", files["test_labels"]))

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
            Linear(128, 10),
        )


def train_fashion_mnist(
    train_subset_size=5000,
    test_subset_size=1000,
    val_subset_size=None,
    num_epochs=8,
    batch_size=128,
    learning_rate=0.001,
    seed=42,
):
    np.random.seed(seed)
    print("Preparing Fashion-MNIST training...")
    print("=" * 60)

    print("Loading Fashion-MNIST data...")
    files = download_fashion_mnist_data()
    train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)

    train_subset_size = min(train_subset_size, len(train_X))
    test_subset_size = min(test_subset_size, len(test_X))
    if train_subset_size < 2:
        raise ValueError("train_subset_size must be at least 2 so train/val split is possible.")

    if val_subset_size is None:
        val_subset_size = max(1, int(train_subset_size * 0.1))
    val_subset_size = min(max(1, val_subset_size), train_subset_size - 1)

    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]

    train_X_split, train_y_split, val_X_split, val_y_split = split_train_val(
        train_X_subset,
        train_y_subset,
        val_size=val_subset_size,
        seed=seed,
    )

    print(f"Train: {train_X_split.shape}, Val: {val_X_split.shape}, Test: {test_X_subset.shape}")

    train_dataset = Dataset(train_X_split, train_y_split)
    val_dataset = Dataset(val_X_split, val_y_split)
    test_dataset = Dataset(test_X_subset, test_y_subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building model...")
    model = FashionMNISTNet()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, optimizer)

    print("Training...")
    print("-" * 60)
    start = time.time()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=num_epochs,
        train_log_every=50,
        verbose=True,
    )
    print(f"Training finished in {time.time() - start:.2f}s")

    print("Evaluating test set...")
    final_test_loss, final_accuracy = trainer.evaluate(test_loader)
    print(f"Test loss: {final_test_loss:.4f}")
    print(f"Test accuracy: {final_accuracy:.4f}")

    print("Saving model parameters...")
    try:
        np.savez("fashion_mnist_model_params.npz", **model.state_dict())
        print("Saved to fashion_mnist_model_params.npz")
    except Exception as e:
        print(f"Failed to save model parameters: {e}")

    return history["train_loss"], history["train_acc"], history["val_acc"], final_accuracy


def test_fashion_mnist_components():
    print("Testing Fashion-MNIST components...")
    print("-" * 40)

    print("1. Testing data download...")
    try:
        files = download_fashion_mnist_data()
        print("PASS Data download successful")
    except Exception as e:
        print(f"FAIL Data download failed: {e}")
        return False

    print("2. Testing data loading...")
    try:
        train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)
        print(f"PASS Train data: {train_X.shape}, labels: {train_y.shape}")
        print(f"PASS Test data:  {test_X.shape}, labels: {test_y.shape}")
    except Exception as e:
        print(f"FAIL Data loading failed: {e}")
        return False

    print("3. Testing model forward...")
    try:
        model = FashionMNISTNet()
        dummy_input = Tensor(np.random.randn(1, 28, 28))
        output = model(dummy_input)
        print(f"PASS Model output shape: {output.shape}")
    except Exception as e:
        print(f"FAIL Model creation failed: {e}")
        return False

    print("All component tests passed.")
    print("-" * 40)
    return True


def test_fashion_mnist_training():
    try:
        train_losses, train_accuracies, val_accuracies, final_acc = train_fashion_mnist(
            train_subset_size=2000,
            test_subset_size=500,
            num_epochs=3,
            batch_size=128,
            learning_rate=0.001,
        )

        assert len(train_losses) == 3, f"Expected 3 epochs, got {len(train_losses)}"
        assert len(train_accuracies) == 3, f"Expected 3 training accuracies, got {len(train_accuracies)}"
        assert len(val_accuracies) == 3, f"Expected 3 validation accuracies, got {len(val_accuracies)}"
        assert final_acc > 0.6, f"Final test accuracy too low: {final_acc}"

        print(f"PASS Fashion-MNIST training test passed, final test accuracy: {final_acc:.4f}")
    except Exception as e:
        print(f"FAIL Fashion-MNIST training test failed: {e}")
        raise


def benchmark_training():
    print("Fashion-MNIST benchmark")
    print("=" * 60)

    configs = [
        {"name": "quick", "train_size": 1000, "test_size": 500, "epochs": 3, "batch_size": 256, "lr": 0.01},
        {"name": "standard", "train_size": 5000, "test_size": 1000, "epochs": 8, "batch_size": 128, "lr": 0.001},
        {"name": "deep", "train_size": 10000, "test_size": 2000, "epochs": 10, "batch_size": 64, "lr": 0.001},
    ]

    results = []

    for config in configs:
        print(f"\nRunning {config['name']}...")
        try:
            train_losses, train_accuracies, _, final_acc = train_fashion_mnist(
                train_subset_size=config["train_size"],
                test_subset_size=config["test_size"],
                num_epochs=config["epochs"],
                batch_size=config["batch_size"],
                learning_rate=config["lr"],
            )

            result = {
                "name": config["name"],
                "final_train_acc": train_accuracies[-1],
                "final_test_acc": final_acc,
                "final_loss": train_losses[-1],
                "config": config,
            }
            results.append(result)

            print(f"{config['name']} done - test accuracy: {final_acc:.4f}")
        except Exception as e:
            print(f"{config['name']} failed: {e}")

    print("\n" + "=" * 60)
    print("Benchmark comparison:")
    print("-" * 60)
    for result in results:
        print(
            f"{result['name']:12} | "
            f"train: {result['final_train_acc']:.4f} | "
            f"test: {result['final_test_acc']:.4f} | "
            f"loss: {result['final_loss']:.4f}"
        )

    return results


if __name__ == "__main__":
    if not test_fashion_mnist_components():
        raise SystemExit(1)

    train_fashion_mnist()
    print("Fashion-MNIST training complete.")
