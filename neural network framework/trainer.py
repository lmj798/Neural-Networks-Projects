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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, monitor="val_loss", mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, metrics: dict, epoch: int) -> bool:
        score = metrics.get(self.monitor, 0.0)
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        *,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = softmax_cross_entropy,
        grad_modifier: Optional[Callable[[object], None]] = None,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_modifier = grad_modifier
        self.accumulation_steps = accumulation_steps

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

            if training and batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()

            grad_context = nullcontext() if training else no_grad()
            with grad_context:
                logits = self.model(data)
                loss = self.loss_fn(logits, target_indices)
                if training:
                    loss = loss / float(self.accumulation_steps)

            if training:
                loss.backward()

            loss_data = float(loss.realize_cached_data()) * (self.accumulation_steps if training else 1.0)
            total_loss += loss_data

            pred = np.argmax(logits.realize_cached_data(), axis=1)
            total_correct += int(np.sum(pred == target_np))
            total_count += int(target_np.shape[0])
            num_batches += 1

            if training and ((batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(loader) - 1):
                if self.grad_modifier is not None:
                    self.grad_modifier(self.model)
                self.optimizer.step()

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
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Dict[str, list]:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        if early_stopping is not None:
            early_stopping.reset()

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

            if early_stopping is not None:
                current_metrics = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }
                should_stop = early_stopping(current_metrics, epoch + 1)
                if should_stop:
                    if verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}. "
                                    f"Best epoch: {early_stopping.best_epoch}")
                    break

        return history

    def evaluate(self, loader) -> Tuple[float, float]:
        return self._run_epoch(loader, training=False)
