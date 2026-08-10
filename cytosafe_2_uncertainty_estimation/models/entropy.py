"""Method 1 — Entropy uncertainty.

Trains the base MLP with CrossEntropyLoss.
Uncertainty score = Shannon entropy of the softmax output.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import build_mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(X_train: np.ndarray, y_train: np.ndarray,
          epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
          patience: int = 10, X_val=None, y_val=None) -> nn.Module:
    model = build_mlp(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if X_val is not None:
            val_loss = _val_loss(model, criterion, X_val, y_val, batch_size)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _val_loss(model, criterion, X_val, y_val, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
    )
    total_loss = total_n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, _ = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            total_n += len(yb)
    return total_loss / total_n


def uncertainty_scores(model: nn.Module, X: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """Return Shannon entropy H(p) for each sample."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )
    probs_list = []
    with torch.no_grad():
        for (xb,) in loader:
            logits, _ = model(xb.to(DEVICE))
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    # Shannon entropy: -sum(p * log(p + eps))
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return entropy
