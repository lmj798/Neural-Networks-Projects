import argparse
import os
import pickle
import tarfile
import time
from urllib.request import urlretrieve

import numpy as np

from tensor import Tensor
from nn import (
    Sequential,
    Linear,
    ReLUModule,
    Flatten,
    Dropout,
    Conv2d,
    Module,
    MultiHeadSelfAttention,
    TransformerEncoderBlock,
    LayerNorm,
    Parameter,
    SEBlock,
)
from optimizers import Adam
from data import Dataset, DataLoader
from ops import softmax_cross_entropy

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar10_data"
CIFAR10_BATCH_DIR = "cifar-10-batches-py"


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


def _safe_extractall(tar, path):
    abs_target = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(abs_target, member.name))
        if os.path.commonpath([abs_target, member_path]) != abs_target:
            raise ValueError(f"Blocked unsafe path in archive: {member.name}")
    tar.extractall(path=abs_target)


def download_cifar10_data(data_dir=CIFAR10_DIR):
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    batch_dir = os.path.join(data_dir, CIFAR10_BATCH_DIR)

    if not os.path.exists(archive_path):
        print(f"Downloading CIFAR-10 to {archive_path}...")
        urlretrieve(CIFAR10_URL, archive_path)
        print("Download complete.")
    else:
        print("CIFAR-10 archive already exists.")

    if not os.path.exists(batch_dir):
        print("Extracting CIFAR-10 archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extractall(tar, data_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10 archive already extracted.")

    return batch_dir


def _load_batch(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    X = data[b"data"]
    y = np.array(data[b"labels"])
    return X, y


def load_cifar10_data(data_dir=CIFAR10_DIR):
    batch_dir = os.path.join(data_dir, CIFAR10_BATCH_DIR)
    if not os.path.isdir(batch_dir):
        raise FileNotFoundError(
            f"CIFAR-10 batch directory not found: {batch_dir}. Run download_cifar10_data first."
        )

    train_X_list = []
    train_y_list = []
    for i in range(1, 6):
        batch_path = os.path.join(batch_dir, f"data_batch_{i}")
        X, y = _load_batch(batch_path)
        train_X_list.append(X)
        train_y_list.append(y)

    test_path = os.path.join(batch_dir, "test_batch")
    test_X, test_y = _load_batch(test_path)

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    train_X = train_X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_X = test_X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    return train_X, train_y, test_X, test_y


class SimpleCIFAR10MLP(Sequential):
    def __init__(self, dropout=0.0):
        layers = [
            Flatten(),
            Linear(32 * 32 * 3, 1024),
            ReLUModule(),
        ]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers += [
            Linear(1024, 512),
            ReLUModule(),
        ]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers += [
            Linear(512, 256),
            ReLUModule(),
        ]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(Linear(256, 10))
        super().__init__(*layers)


class SimpleCIFAR10CNN(Sequential):
    def __init__(self, dropout=0.0):
        layers = [
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            ReLUModule(),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            ReLUModule(),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ReLUModule(),
            Flatten(),
            Linear(128 * 8 * 8, 256),
            ReLUModule(),
        ]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(Linear(256, 10))
        super().__init__(*layers)

class SimpleCIFAR10CNNSE(Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act = ReLUModule()
        self.se2 = SEBlock(64, reduction=8)
        self.se3 = SEBlock(128, reduction=8)
        self.flatten = Flatten()
        self.fc1 = Linear(128 * 8 * 8, 256)
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.se2(x)
        x = self.act(self.conv3(x))
        x = self.se3(x)
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)

class SimpleCIFAR10CNNAttention(Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act = ReLUModule()
        self.attn = MultiHeadSelfAttention(embed_dim=128, num_heads=4)
        self.flatten = Flatten()
        self.fc1 = Linear(128 * 8 * 8, 256)
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        batch_size = x.shape[0]

        # (B, C, H, W) -> (B, H*W, C)
        x = x.transpose((0, 2, 3, 1)).reshape((batch_size, 8 * 8, 128))
        x = self.attn(x)

        # (B, H*W, C) -> (B, C, H, W)
        x = x.reshape((batch_size, 8, 8, 128)).transpose((0, 3, 1, 2))
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)

class SimpleCIFAR10CNNTransformer(Module):
    def __init__(self, dropout=0.0, num_layers=2, num_heads=4):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act = ReLUModule()

        self.embed_dim = 128
        self.num_tokens = 8 * 8
        self.pos_embed = Parameter(np.random.normal(0, 0.02, (1, self.num_tokens, self.embed_dim)).astype("float32"))
        self.blocks = tuple(
            TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(self.embed_dim)
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.head = Linear(self.embed_dim, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        batch_size = x.shape[0]
        x = x.transpose((0, 2, 3, 1)).reshape((batch_size, self.num_tokens, self.embed_dim))
        x = x + self.pos_embed.broadcast_to(x.shape)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x.sum(axis=1) / self.num_tokens
        if self.dropout is not None:
            x = self.dropout(x)
        return self.head(x)

class SimpleCIFAR10ViT(Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_side = image_size // patch_size
        self.num_tokens = self.num_patches_side * self.num_patches_side
        patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = Linear(patch_dim, embed_dim)
        self.pos_embed = Parameter(np.random.normal(0, 0.02, (1, self.num_tokens, embed_dim)).astype("float32"))
        self.blocks = tuple(
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.head = Linear(embed_dim, 10)

    def _patchify(self, x):
        # x: (B, C, H, W) -> (B, N, C*P*P)
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape((b, c, h // p, p, w // p, p))
        x = x.transpose((0, 2, 4, 1, 3, 5))
        return x.reshape((b, self.num_tokens, c * p * p))

    def forward(self, x):
        x = self._patchify(x)
        x = self.patch_proj(x)
        x = x + self.pos_embed.broadcast_to(x.shape)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x.sum(axis=1) / self.num_tokens
        if self.dropout is not None:
            x = self.dropout(x)
        return self.head(x)


def train_cifar10(
    *,
    seed: int = 42,
    num_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    train_subset_size: int = 5000,
    test_subset_size: int = 1000,
    normalize: bool = False,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    model: str = "mlp",
):
    np.random.seed(seed)

    print("Preparing CIFAR-10...")
    print("=" * 50)
    download_cifar10_data()
    train_X, train_y, test_X, test_y = load_cifar10_data()

    print(f"Train data: {train_X.shape}, labels: {train_y.shape}")
    print(f"Test data: {test_X.shape}, labels: {test_y.shape}")

    train_subset_size = min(train_subset_size, len(train_X))
    test_subset_size = min(test_subset_size, len(test_X))

    train_X_subset = train_X[:train_subset_size]
    train_y_subset = train_y[:train_subset_size]
    test_X_subset = test_X[:test_subset_size]
    test_y_subset = test_y[:test_subset_size]

    transforms = None
    if normalize:
        mean = train_X_subset.mean(axis=(0, 2, 3))
        std = train_X_subset.std(axis=(0, 2, 3)) + 1e-8

        def _normalize(x):
            return (x - mean[:, None, None]) / std[:, None, None]

        transforms = [_normalize]

    train_dataset = Dataset(train_X_subset, train_y_subset, transforms=transforms, image_shape=(3, 32, 32))
    test_dataset = Dataset(test_X_subset, test_y_subset, transforms=transforms, image_shape=(3, 32, 32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Building model...")
    if model == "mlp":
        model = SimpleCIFAR10MLP(dropout=dropout)
    elif model == "cnn":
        model = SimpleCIFAR10CNN(dropout=dropout)
    elif model == "cnn_se":
        model = SimpleCIFAR10CNNSE(dropout=dropout)
    elif model == "cnn_attn":
        model = SimpleCIFAR10CNNAttention(dropout=dropout)
    elif model == "cnn_transformer":
        model = SimpleCIFAR10CNNTransformer(dropout=dropout)
    elif model == "vit":
        model = SimpleCIFAR10ViT(dropout=dropout)
    else:
        raise ValueError("model must be 'mlp', 'cnn', 'cnn_se', 'cnn_attn', 'cnn_transformer', or 'vit'")
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Training...")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(data)
            target_data = target.realize_cached_data().astype(np.int64)
            target_indices = Tensor(target_data, dtype=np.int64, requires_grad=False)

            loss = softmax_cross_entropy(logits, target_indices)
            loss.backward()

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

            loss_data = float(loss.realize_cached_data())
            epoch_loss += loss_data
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            epoch_correct += np.sum(pred == target_data)
            epoch_total += len(target_data)

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss_data:.4f}")

        train_accuracy = epoch_correct / epoch_total if epoch_total else 0.0
        num_batches = len(train_dataset) // train_loader.batch_size + (
            1 if len(train_dataset) % train_loader.batch_size else 0
        )
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

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
        test_accuracies.append(test_accuracy)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Train accuracy: {train_accuracy:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print("-" * 50)

    return train_losses, train_accuracies, test_accuracies


def main():
    parser = argparse.ArgumentParser(description="Train an MLP on CIFAR-10.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-subset", type=int, default=5000)
    parser.add_argument("--test-subset", type=int, default=1000)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "cnn_se", "cnn_attn", "cnn_transformer", "vit"],
    )
    args = parser.parse_args()

    train_cifar10(
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_subset_size=args.train_subset,
        test_subset_size=args.test_subset,
        normalize=args.normalize,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        model=args.model,
    )


if __name__ == "__main__":
    main()


