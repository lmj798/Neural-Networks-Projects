from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from logger import logger
from ops import softmax_cross_entropy
from tensor import Tensor, no_grad


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(X.shape[0])
    if total != int(y.shape[0]):
        raise ValueError("X and y must have the same number of samples.")
    if total < 2:
        raise ValueError("Need at least 2 samples to split into train/val.")
    if val_size <= 0 or val_size >= total:
        raise ValueError("val_size must be in [1, total_samples - 1].")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        *,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = softmax_cross_entropy,
        grad_modifier: Optional[Callable[[object], None]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_modifier = grad_modifier

    def _run_epoch(self, loader, *, training: bool, log_every: Optional[int] = None) -> Tuple[float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(loader):
            target_np = target.realize_cached_data().astype(np.int64)
            target_indices = Tensor(target_np, dtype=np.int64, requires_grad=False)

            if training:
                self.optimizer.zero_grad()
            grad_context = nullcontext() if training else no_grad()
            with grad_context:
                logits = self.model(data)
                loss = self.loss_fn(logits, target_indices)

            if training:
                loss.backward()
                if self.grad_modifier is not None:
                    self.grad_modifier(self.model)
                self.optimizer.step()

            loss_data = float(loss.realize_cached_data())
            total_loss += loss_data

            pred = np.argmax(logits.realize_cached_data(), axis=1)
            total_correct += int(np.sum(pred == target_np))
            total_count += int(target_np.shape[0])
            num_batches += 1

            if training and log_every is not None and log_every > 0 and batch_idx % log_every == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss_data:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_count, 1)
        return avg_loss, accuracy

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int,
        train_log_every: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(
                train_loader,
                training=True,
                log_every=train_log_every,
            )
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} summary:")
                logger.info(f"  Train loss: {train_loss:.4f}")
                logger.info(f"  Train accuracy: {train_acc:.4f}")
                logger.info(f"  Val loss: {val_loss:.4f}")
                logger.info(f"  Val accuracy: {val_acc:.4f}")

        return history

    def evaluate(self, loader) -> Tuple[float, float]:
        return self._run_epoch(loader, training=False)
