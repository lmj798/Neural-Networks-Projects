import numpy as np

from data import DataLoader, Dataset
from nn import Linear, Sequential
from optimizers import Adam
from trainer import Trainer, split_train_val


def test_split_train_val_sizes_and_disjoint():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.arange(20, dtype=np.int64)

    train_X, train_y, val_X, val_y = split_train_val(X, y, val_size=5, seed=7)

    assert train_X.shape[0] == 15
    assert val_X.shape[0] == 5
    assert train_y.shape[0] == 15
    assert val_y.shape[0] == 5

    train_ids = set(train_y.tolist())
    val_ids = set(val_y.tolist())
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.union(val_ids) == set(y.tolist())


def test_trainer_fit_and_evaluate_on_simple_separable_data():
    rng = np.random.default_rng(0)

    class0 = rng.normal(loc=-1.5, scale=0.2, size=(80, 2)).astype(np.float32)
    class1 = rng.normal(loc=1.5, scale=0.2, size=(80, 2)).astype(np.float32)
    X = np.concatenate([class0, class1], axis=0)
    y = np.concatenate([
        np.zeros(80, dtype=np.int64),
        np.ones(80, dtype=np.int64),
    ])

    train_X, train_y, val_X, val_y = split_train_val(X, y, val_size=32, seed=1)

    test_class0 = rng.normal(loc=-1.5, scale=0.2, size=(40, 2)).astype(np.float32)
    test_class1 = rng.normal(loc=1.5, scale=0.2, size=(40, 2)).astype(np.float32)
    test_X = np.concatenate([test_class0, test_class1], axis=0)
    test_y = np.concatenate([
        np.zeros(40, dtype=np.int64),
        np.ones(40, dtype=np.int64),
    ])

    train_loader = DataLoader(Dataset(train_X, train_y, image_shape=None), batch_size=16, shuffle=True)
    val_loader = DataLoader(Dataset(val_X, val_y, image_shape=None), batch_size=16, shuffle=False)
    test_loader = DataLoader(Dataset(test_X, test_y, image_shape=None), batch_size=16, shuffle=False)

    model = Sequential(Linear(2, 2))
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer)

    history = trainer.fit(train_loader, val_loader, epochs=12, train_log_every=None, verbose=False)

    assert len(history["train_loss"]) == 12
    assert len(history["train_acc"]) == 12
    assert len(history["val_acc"]) == 12

    _, test_acc = trainer.evaluate(test_loader)
    assert test_acc > 0.95
