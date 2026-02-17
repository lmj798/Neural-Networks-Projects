import argparse
import gzip
import os
import struct
import time
from urllib.request import urlretrieve

import numpy as np

from tensor import Tensor
from nn import (
    Sequential,
    Linear,
    Flatten,
    ReLUModule,
    SigmoidModule,
    TanhModule,
    LeakyReLUModule,
    ELUModule,
    Dropout,
)
from optimizers import Adam
from data import Dataset, DataLoader

FASHION_MNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def download_fashion_mnist_data(data_dir="fashion_mnist_data"):
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs(data_dir, exist_ok=True)

    for filename in files.values():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(FASHION_MNIST_URL + filename, filepath)
            print(f"Downloaded {filename}")
        else:
            print(f"Found existing {filename}")

    return {k: os.path.join(data_dir, v) for k, v in files.items()}


def load_fashion_mnist_data(files):
    def load_images(filename):
        with gzip.open(filename, "rb") as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return (
                np.frombuffer(f.read(), dtype=np.uint8)
                .reshape(num, rows, cols)
                .astype(np.float32)
                / 255.0
            )

    def load_labels(filename):
        with gzip.open(filename, "rb") as f:
            _, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = load_images(files["train_images"])
    train_labels = load_labels(files["train_labels"])
    test_images = load_images(files["test_images"])
    test_labels = load_labels(files["test_labels"])

    return train_images, train_labels, test_images, test_labels


def _activation_factory(name, leaky_slope=0.01, elu_alpha=1.0):
    name = name.lower()
    if name == "relu":
        return ReLUModule
    if name == "sigmoid":
        return SigmoidModule
    if name == "tanh":
        return TanhModule
    if name == "leaky_relu":
        return lambda: LeakyReLUModule(negative_slope=leaky_slope)
    if name == "elu":
        return lambda: ELUModule(alpha=elu_alpha)
    raise ValueError(f"Unknown activation: {name}")


def _default_init_for_activation(name):
    name = name.lower()
    if name in ("sigmoid", "tanh"):
        return "xavier"
    return "he"


def build_model(
    activation_name,
    *,
    leaky_slope=0.01,
    elu_alpha=1.0,
    dropout=0.0,
    weight_init="auto",
):
    act_factory = _activation_factory(activation_name, leaky_slope, elu_alpha)
    if weight_init == "auto":
        weight_init = _default_init_for_activation(activation_name)

    layers = [
        Flatten(),
        Linear(784, 512, weight_init=weight_init),
        act_factory(),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [
        Linear(512, 256, weight_init=weight_init),
        act_factory(),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [
        Linear(256, 128, weight_init=weight_init),
        act_factory(),
    ]
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers.append(Linear(128, 10, weight_init=weight_init))

    return Sequential(*layers)


def train_once(
    train_X,
    train_y,
    test_X,
    test_y,
    *,
    activation="relu",
    seed=42,
    num_epochs=5,
    batch_size=128,
    lr=0.001,
    train_subset_size=5000,
    test_subset_size=1000,
    leaky_slope=0.01,
    elu_alpha=1.0,
    normalize=False,
    dropout=0.0,
    weight_decay=0.0,
    weight_init="auto",
    patience=0,
    min_delta=0.0,
):
    np.random.seed(seed)

    train_subset_size = min(train_subset_size, len(train_X))
    test_subset_size = min(test_subset_size, len(test_X))

    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]

    transforms = None
    if normalize:
        mean = float(train_X_subset.mean())
        std = float(train_X_subset.std()) + 1e-8

        def _normalize(x):
            return (x - mean) / std

        transforms = [_normalize]

    train_dataset = Dataset(train_X_subset, train_y_subset, transforms=transforms)
    test_dataset = Dataset(test_X_subset, test_y_subset, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(
        activation,
        leaky_slope=leaky_slope,
        elu_alpha=elu_alpha,
        dropout=dropout,
        weight_init=weight_init,
    )
    optimizer = Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    epochs_without_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            logits = model(data)

            target_data = target.realize_cached_data()
            if target_data.dtype != np.int64:
                target_data = target_data.astype(np.int64)

            optimizer.zero_grad()

            probs = softmax(logits.realize_cached_data())
            log_probs = np.log(probs + 1e-8)
            loss_data = -np.sum(log_probs[np.arange(logits.shape[0]), target_data]) / logits.shape[0]

            grad_logits = probs.copy()
            grad_logits[np.arange(logits.shape[0]), target_data] -= 1.0
            grad_logits /= logits.shape[0]

            logits.backward(Tensor(grad_logits, requires_grad=False))

            if weight_decay > 0:
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    grad_data = (
                        param.grad.realize_cached_data()
                        if isinstance(param.grad, Tensor)
                        else param.grad
                    )
                    grad_data = grad_data + weight_decay * param.realize_cached_data()
                    if isinstance(param.grad, Tensor):
                        param.grad.cached_data = grad_data
                    else:
                        param.grad = grad_data
            optimizer.step()

            epoch_loss += loss_data
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            epoch_correct += np.sum(pred == target_data)
            epoch_total += len(target_data)

            if batch_idx % 50 == 0:
                print(
                    f"[{activation}] Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx}, Loss: {loss_data:.4f}"
                )

        train_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        num_batches = len(train_dataset) // train_loader.batch_size + (
            1 if len(train_dataset) % train_loader.batch_size else 0
        )
        avg_loss = epoch_loss / num_batches

        model.eval()
        test_correct = 0
        test_total = 0
        for data, target in test_loader:
            logits = model(data)
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            target_data = target.realize_cached_data()
            test_correct += np.sum(pred == target_data)
            test_total += len(target_data)

        test_accuracy = test_correct / test_total if test_total else 0.0
        epoch_time = time.time() - start_time

        print(
            f"[{activation}] Epoch {epoch + 1}/{num_epochs} summary: "
            f"loss={avg_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"test_acc={test_accuracy:.4f}, time={epoch_time:.2f}s"
        )

        if patience > 0:
            if test_accuracy > best_acc + min_delta:
                best_acc = test_accuracy
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(
                        f"[{activation}] Early stopping at epoch {epoch + 1} "
                        f"(best_acc={best_acc:.4f})."
                    )
                    break

    return test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Benchmark activations on Fashion-MNIST.")
    parser.add_argument("--activation", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--leaky-slope", type=float, default=0.01)
    parser.add_argument("--elu-alpha", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--weight-init", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args()

    files = download_fashion_mnist_data()
    train_X, train_y, test_X, test_y = load_fashion_mnist_data(files)

    if args.activation == "all":
        activations = ["relu", "sigmoid", "tanh", "leaky_relu", "elu"]
    else:
        activations = [args.activation]

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    results = {act: [] for act in activations}
    for seed in seeds:
        for act in activations:
            print("=" * 60)
            print(f"Running activation: {act} (seed={seed})")
            acc = train_once(
                train_X,
                train_y,
                test_X,
                test_y,
                activation=act,
                seed=seed,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                train_subset_size=args.train_subset,
                test_subset_size=args.test_subset,
                leaky_slope=args.leaky_slope,
                elu_alpha=args.elu_alpha,
                normalize=args.normalize,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                weight_init=args.weight_init,
                patience=args.patience,
                min_delta=args.min_delta,
            )
            results[act].append(acc)

    print("=" * 60)
    print("Final test accuracy:")
    for act in activations:
        accs = results[act]
        if len(accs) == 1:
            print(f"{act}: {accs[0]:.4f}")
        else:
            mean = float(np.mean(accs))
            std = float(np.std(accs, ddof=0))
            joined = ", ".join(f"{a:.4f}" for a in accs)
            print(f"{act}: mean={mean:.4f} std={std:.4f} (runs: {joined})")


if __name__ == "__main__":
    main()
