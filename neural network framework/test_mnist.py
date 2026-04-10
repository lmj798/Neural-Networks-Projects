import argparse
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

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


class SimpleMNISTNet(Sequential):
    def __init__(self):
        super().__init__(
            Flatten(),
            Linear(784, 256),
            ReLUModule(),
            Linear(256, 128),
            ReLUModule(),
            Linear(128, 10),
        )


def download_mnist_data():
    """Download MNIST data files."""
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = MNIST_FILES

    os.makedirs("mnist_data", exist_ok=True)

    for _, filename in files.items():
        filepath = os.path.join("mnist_data", filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filepath)
            print(f"Downloaded: {filename}")
        else:
            print(f"Already exists: {filename}")

    return files


def load_mnist_data(files):
    """Load MNIST data arrays from local files."""

    def load_images(filename):
        with gzip.open(filename, "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(filename, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = load_images(os.path.join("mnist_data", files["train_images"]))
    train_labels = load_labels(os.path.join("mnist_data", files["train_labels"]))
    test_images = load_images(os.path.join("mnist_data", files["test_images"]))
    test_labels = load_labels(os.path.join("mnist_data", files["test_labels"]))

    return train_images, train_labels, test_images, test_labels


def _mnist_files_available(files):
    return all(os.path.exists(os.path.join("mnist_data", filename)) for filename in files.values())


def generate_dummy_mnist_data(num_samples=1000, num_classes=10, seed=42):
    """Generate synthetic MNIST-like data for testing."""
    rng = np.random.default_rng(seed)

    # Generate noisy 28x28 grayscale images.
    X = rng.normal(loc=0.0, scale=0.5, size=(num_samples, 28, 28))

    # Add simple class-dependent patterns near the center.
    for i in range(num_samples):
        digit = i % num_classes
        center = 14
        for j in range(5):
            for k in range(5):
                if digit == 0 or (digit == 1 and k == 2):
                    X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 2:
                    if j == k or j + k == 4:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 3:
                    if k == 2 or j == 0 or j == 4:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 4:
                    if j == 0 or j == 2 or (j == 1 and k == 0):
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 5:
                    if k == 0 or k == 4 or j == 2:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 6:
                    if k == 0 or j == 2 or j == 4:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 7:
                    if j == k or k == 0:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 8:
                    if j == 2 or k == 2 or j == k or j + k == 4:
                        X[i, center + j - 2, center + k - 2] += 1.0
                elif digit == 9:
                    if j == 2 or (j == 0 and k == 2) or (j == 4 and k == 2):
                        X[i, center + j - 2, center + k - 2] += 1.0

    y = np.arange(num_samples) % num_classes
    X += rng.normal(loc=0.0, scale=0.1, size=X.shape)
    return X, y


def train_mnist(
    *,
    seed: int = 42,
    num_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    train_subset_size: int = 5000,
    test_subset_size: int = 1000,
    val_subset_size: int = None,
    data_mode: str = "auto",
    save_model_params: bool = True,
):
    np.random.seed(seed)
    if data_mode not in {"auto", "download", "local", "dummy"}:
        raise ValueError("data_mode must be one of: 'auto', 'download', 'local', 'dummy'")

    print("Preparing MNIST training...")
    print("=" * 50)

    print("Loading MNIST data...")
    if data_mode == "dummy":
        train_X, train_y = generate_dummy_mnist_data(
            num_samples=max(train_subset_size, 1000),
            seed=seed,
        )
        test_X, test_y = generate_dummy_mnist_data(
            num_samples=max(test_subset_size, 500),
            seed=seed + 1,
        )
    else:
        files = MNIST_FILES
        try:
            if (data_mode in {"auto", "local"}) and _mnist_files_available(files):
                pass
            else:
                if data_mode == "local":
                    raise FileNotFoundError("MNIST local files not found under mnist_data/.")
                files = download_mnist_data()
            train_X, train_y, test_X, test_y = load_mnist_data(files)
        except Exception as e:
            if data_mode in {"download", "local"}:
                raise
            print(f"MNIST unavailable, falling back to dummy data: {e}")
            train_X, train_y = generate_dummy_mnist_data(
                num_samples=max(train_subset_size, 1000),
                seed=seed,
            )
            test_X, test_y = generate_dummy_mnist_data(
                num_samples=max(test_subset_size, 500),
                seed=seed + 1,
            )

    print(f"Train data: {train_X.shape}, labels: {train_y.shape}")
    print(f"Test data:  {test_X.shape}, labels: {test_y.shape}")

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

    train_dataset = Dataset(train_X_split, train_y_split)
    val_dataset = Dataset(val_X_split, val_y_split)
    test_dataset = Dataset(test_X_subset, test_y_subset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building model...")
    model = SimpleMNISTNet()
    optimizer = Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer)

    print("Training...")
    print("-" * 50)
    start_time = time.time()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=num_epochs,
        train_log_every=100,
        verbose=True,
    )
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")

    print("Evaluating test set...")
    final_test_loss, final_test_accuracy = trainer.evaluate(test_loader)
    print(f"Test loss: {final_test_loss:.4f}")
    print(f"Test accuracy: {final_test_accuracy:.4f}")

    if save_model_params:
        print("Saving model parameters...")
        try:
            np.savez("mnist_model_params.npz", **model.state_dict())
            print("Saved to mnist_model_params.npz")
        except Exception as e:
            print(f"Failed to save model parameters: {e}")

    return (
        history["train_loss"],
        history["train_acc"],
        history["val_acc"],
        final_test_accuracy,
    )


def test_individual_components():
    """Quick checks for basic framework pieces."""
    print("Testing individual components...")
    print("-" * 30)

    print("1. Tensor arithmetic")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    print(f"   {a.realize_cached_data()} + {b.realize_cached_data()} = {c.realize_cached_data()}")

    print("2. Matrix multiplication")
    x = Tensor([[1, 2], [3, 4]])
    w = Tensor([[5, 6], [7, 8]])
    y = x @ w
    print(f"   matmul output shape: {y.shape}")

    print("3. Linear layer")
    linear = Linear(2, 4)
    input_tensor = Tensor([[1, 2]])
    output = linear(input_tensor)
    print(f"   Linear output shape: {output.shape}")

    print("4. ReLU")
    relu_layer = ReLUModule()
    relu_output = relu_layer(Tensor([[-1, 2, -3, 4]]))
    print(f"   ReLU output: {relu_output.realize_cached_data()}")

    print("5. Optimizer step")
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Adam([param], lr=0.01)
    param.grad = Tensor([0.1, 0.2])
    optimizer.step()
    print(f"   Updated param: {param.realize_cached_data()}")

    print("Component checks finished.")
    print("=" * 30)


def test_mnist_training():
    """Test the end-to-end MNIST training pipeline."""
    try:
        train_losses, train_accuracies, val_accuracies, final_test_accuracy = train_mnist(
            num_epochs=3,
            train_subset_size=2000,
            test_subset_size=500,
            data_mode="dummy",
            save_model_params=False,
        )

        assert len(train_losses) == 3, f"Expected 3 epochs, got {len(train_losses)}"
        assert len(train_accuracies) == 3, f"Expected 3 training accuracies, got {len(train_accuracies)}"
        assert len(val_accuracies) == 3, f"Expected 3 validation accuracies, got {len(val_accuracies)}"
        assert final_test_accuracy > 0.65, f"Final test accuracy too low: {final_test_accuracy}"

        print(f"PASS MNIST training test passed, final test accuracy: {final_test_accuracy:.4f}")
    except Exception as e:
        print(f"FAIL MNIST training test failed: {e}")
        raise


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-subset", type=int, default=5000)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument(
        "--data-mode",
        type=str,
        default="auto",
        choices=["auto", "download", "local", "dummy"],
    )
    p.add_argument("--no-save-params", action="store_true")
    p.add_argument("--skip-component-tests", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.skip_component_tests:
        test_individual_components()
        print("\n")

    try:
        train_losses, train_accuracies, _, final_test_accuracy = train_mnist(
            seed=args.seed,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_subset_size=args.train_subset,
            test_subset_size=args.test_subset,
            val_subset_size=args.val_subset,
            data_mode=args.data_mode,
            save_model_params=not args.no_save_params,
        )

        print("\nMNIST training summary:")
        print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
        print(f"Final test accuracy:  {final_test_accuracy:.4f}")
        print("MNIST training completed successfully.")
    except Exception as e:
        print(f"MNIST training failed: {e}")
        import traceback

        traceback.print_exc()
