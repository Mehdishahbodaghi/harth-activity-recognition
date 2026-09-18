"""
Trains the 1D CNN and LSTM models on the raw windowed HARTH data, using the
exact same subject-wise train/test split as the classical ML models (see
splitting.py) so results are directly comparable.

Usage: python src/train_deep.py
Outputs: data/processed/deep_results.json, trained weights in data/processed/
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from splitting import subject_split, DEFAULT_TEST_SUBJECTS
from deep_models import CNN1D, LSTMClassifier

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    X = np.load(os.path.join(PROC_DIR, "windows_X.npy"))
    y = np.load(os.path.join(PROC_DIR, "windows_y.npy"))
    subjects = np.load(os.path.join(PROC_DIR, "windows_subject.npy"), allow_pickle=True)
    return X, y, subjects


def normalize(X_train, X_test):
    """Per-channel z-score normalization using TRAIN statistics only."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def train_one_model(model, train_loader, val_loader, n_epochs, lr=1e-3, patience=5):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = crit(out, yb)
                val_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:02d}/{n_epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, label_encoder):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            preds.extend(out.argmax(1).tolist())
            trues.extend(yb.tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    report = classification_report(
        trues, preds, target_names=label_encoder.classes_.astype(str), output_dict=True, zero_division=0
    )
    return acc, f1, report, preds, trues


def main():
    print("Loading windowed data...")
    X, y, subjects = load_data()
    train_mask, test_mask = subject_split(subjects, DEFAULT_TEST_SUBJECTS)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train_full, y_train_full = X[train_mask], y_enc[train_mask]
    X_test, y_test = X[test_mask], y_enc[test_mask]

    # carve out a validation slice from train (random, for early-stopping only;
    # final reported metrics always come from the held-out TEST subjects)
    rng = np.random.default_rng(SEED)
    n_train = len(X_train_full)
    val_idx = rng.choice(n_train, size=int(0.1 * n_train), replace=False)
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True

    X_val, y_val = X_train_full[val_mask], y_train_full[val_mask]
    X_train, y_train = X_train_full[~val_mask], y_train_full[~val_mask]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    X_train_n, X_test_n, mean, std = normalize(X_train, X_test)
    X_val_n = (X_val - mean) / std

    def to_loader(Xa, ya, batch_size, shuffle):
        ds = TensorDataset(torch.tensor(Xa, dtype=torch.float32), torch.tensor(ya, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train_n, y_train, 256, True)
    val_loader = to_loader(X_val_n, y_val, 512, False)
    test_loader = to_loader(X_test_n, y_test, 512, False)

    n_classes = len(le.classes_)
    results = {}

    print("\n=== Training 1D CNN ===")
    cnn = CNN1D(n_classes=n_classes)
    cnn, cnn_history = train_one_model(cnn, train_loader, val_loader, n_epochs=20, patience=5)
    acc, f1, report, preds, trues = evaluate(cnn, test_loader, le)
    print(f"CNN1D test -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    results["cnn1d"] = {"accuracy": acc, "macro_f1": f1, "history": cnn_history, "report": report}
    torch.save(cnn.state_dict(), os.path.join(PROC_DIR, "cnn1d_weights.pt"))

    print("\n=== Training LSTM ===")
    lstm = LSTMClassifier(n_classes=n_classes)
    lstm, lstm_history = train_one_model(lstm, train_loader, val_loader, n_epochs=12, patience=4)
    acc, f1, report, preds, trues = evaluate(lstm, test_loader, le)
    print(f"LSTM test -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    results["lstm"] = {"accuracy": acc, "macro_f1": f1, "history": lstm_history, "report": report}
    torch.save(lstm.state_dict(), os.path.join(PROC_DIR, "lstm_weights.pt"))

    with open(os.path.join(PROC_DIR, "deep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to data/processed/deep_results.json")


if __name__ == "__main__":
    main()
