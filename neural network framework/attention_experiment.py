import argparse
import numpy as np

from data import DataLoader, Dataset
from nn import Flatten, Linear, Module, MultiHeadSelfAttention, ReLUModule
from ops import softmax_cross_entropy
from optimizers import Adam
from tensor import Tensor


def generate_marker_match_data(
    num_samples=2000,
    seq_len=16,
    vocab_size=12,
    seed=42,
):
    rng = np.random.default_rng(seed)
    x = np.zeros((num_samples, seq_len, vocab_size + 2), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    for n in range(num_samples):
        tokens = rng.integers(0, vocab_size, size=seq_len)
        i, j = rng.choice(seq_len, size=2, replace=False)
        label = rng.integers(0, 2)

        if label == 1:
            tokens[j] = tokens[i]
        else:
            if tokens[j] == tokens[i]:
                tokens[j] = (tokens[i] + rng.integers(1, vocab_size)) % vocab_size

        x[n, np.arange(seq_len), tokens] = 1.0
        x[n, i, vocab_size] = 1.0
        x[n, j, vocab_size + 1] = 1.0
        y[n] = label

    return x, y


class BaselineSequenceClassifier(Module):
    def __init__(self, seq_len, input_dim, embed_dim=8):
        super().__init__()
        self.token_proj = Linear(input_dim, embed_dim)
        self.act = ReLUModule()
        self.seq_len = seq_len
        self.head = Linear(embed_dim, 2)

    def forward(self, x):
        x = self.token_proj(x)
        x = self.act(x)
        x = x.sum(axis=1) / self.seq_len
        return self.head(x)


class AttentionSequenceClassifier(Module):
    def __init__(self, seq_len, input_dim, embed_dim=8, num_heads=2):
        super().__init__()
        self.token_proj = Linear(input_dim, embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.act = ReLUModule()
        self.flatten = Flatten()
        self.head = Linear(seq_len * embed_dim, 2)

    def forward(self, x):
        x = self.token_proj(x)
        x = self.attn(x)
        x = self.act(x)
        x = self.flatten(x)
        return self.head(x)


def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total_count = 0
    for xb, yb in loader:
        logits = model(xb).realize_cached_data()
        pred = np.argmax(logits, axis=1)
        target = yb.realize_cached_data().astype(np.int64)
        total_correct += int(np.sum(pred == target))
        total_count += int(target.shape[0])
    return total_correct / max(total_count, 1)


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            targets = Tensor(yb.realize_cached_data().astype(np.int64), dtype=np.int64, requires_grad=False)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.realize_cached_data())
            pred = np.argmax(logits.realize_cached_data(), axis=1)
            target_np = yb.realize_cached_data().astype(np.int64)
            correct += int(np.sum(pred == target_np))
            count += int(target_np.shape[0])

        train_acc = correct / max(count, 1)
        val_acc = evaluate(model, val_loader)
        history["train_loss"].append(loss_sum / max(len(train_loader.ordering), 1))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    return history


def run_experiment(
    seed=42,
    train_size=4000,
    val_size=1000,
    seq_len=16,
    vocab_size=12,
    batch_size=128,
    epochs=12,
    lr=3e-3,
):
    np.random.seed(seed)

    x_train, y_train = generate_marker_match_data(
        num_samples=train_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    x_val, y_val = generate_marker_match_data(
        num_samples=val_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed + 1,
    )

    train_loader = DataLoader(Dataset(x_train, y_train, image_shape=None), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Dataset(x_val, y_val, image_shape=None), batch_size=batch_size, shuffle=False)

    input_dim = vocab_size + 2
    baseline = BaselineSequenceClassifier(seq_len=seq_len, input_dim=input_dim, embed_dim=8)
    attention = AttentionSequenceClassifier(seq_len=seq_len, input_dim=input_dim, embed_dim=8, num_heads=2)

    baseline_hist = train_model(baseline, train_loader, val_loader, epochs=epochs, lr=lr)
    attention_hist = train_model(attention, train_loader, val_loader, epochs=epochs, lr=lr)

    return {
        "baseline": baseline_hist,
        "attention": attention_hist,
    }


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs attention on a marker-match sequence task.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=4000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-3)
    args = p.parse_args()

    result = run_experiment(
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    b_val = result["baseline"]["val_acc"][-1]
    a_val = result["attention"]["val_acc"][-1]
    gain = a_val - b_val

    print("Experiment complete.")
    print(f"Baseline  final val_acc: {b_val:.4f}")
    print(f"Attention final val_acc: {a_val:.4f}")
    print(f"Attention gain          : {gain:+.4f}")


if __name__ == "__main__":
    main()
