"""Method 2 — MC Dropout uncertainty.

Uses the same trained MLP as Entropy (train once, evaluate differently).
At test time dropout is kept ON at p=0.5 (higher than training p=0.3)
to maximise disagreement between passes, and T stochastic forward passes
are averaged.
Uncertainty score = predictive entropy over T softmax distributions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from models.entropy import train  # reuse the same training loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 50                  # number of stochastic forward passes
MC_DROPOUT_P = 0.5      # dropout rate at inference (higher than training 0.3)


def _set_dropout_p(model: nn.Module, p: float):
    """Override dropout probability for all Dropout layers."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            m.train()   # keep dropout active


def uncertainty_scores(model: nn.Module, X: np.ndarray,
                       t: int = T, batch_size: int = 256) -> np.ndarray:
    """Return predictive entropy from T MC dropout passes."""
    model.eval()
    _set_dropout_p(model, MC_DROPOUT_P)

    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )

    all_probs = []
    with torch.no_grad():
        for _ in range(t):
            pass_probs = []
            for (xb,) in loader:
                logits, _ = model(xb.to(DEVICE))
                pass_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_probs.append(np.concatenate(pass_probs, axis=0))

    all_probs = np.stack(all_probs, axis=0)   # (T, N, C)
    mean_probs = all_probs.mean(axis=0)        # (N, C)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

    # Restore original dropout p after scoring
    _set_dropout_p(model, 0.3)
    return entropy
