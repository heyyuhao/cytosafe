"""Method 2 — MC Dropout uncertainty.

Uses the same trained MLP as Entropy (train once, evaluate differently).
At test time dropout is kept ON at the SAME p=0.3 as training — required
for the Gal & Ghahramani (2016) Bayesian approximation to be valid.

Uncertainty score = BALD (Bayesian Active Learning by Disagreement):
    BALD = H(E_t[p]) - E_t[H(p)]
         = entropy of mean prediction - mean of per-pass entropies

BALD specifically measures epistemic uncertainty (disagreement between
passes), not total uncertainty. It is zero when all passes agree on any
class and high when passes disagree — making it sensitive to OOD inputs
that activate inconsistent predictions across dropout masks.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from models.entropy import train  # reuse the same training loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 50


def _enable_dropout(model: nn.Module, p: float):
    """Set all Dropout layers to train mode with given p."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            m.train()


def uncertainty_scores(model: nn.Module, X: np.ndarray,
                       t: int = T, batch_size: int = 256) -> np.ndarray:
    """Return BALD score = H(E[p]) - E[H(p)] from T MC dropout passes.

    Uses the same dropout p=0.3 as training to maintain the valid
    Bayesian approximation. LayerNorm makes eval() safe to use alongside
    dropout (no batch-statistics issue).
    """
    model.eval()
    _enable_dropout(model, 0.3)   # must match training p

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

    # H(E_t[p]) — entropy of mean prediction
    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

    # E_t[H(p)] — mean of per-pass entropies
    pass_entropies = -np.sum(
        all_probs * np.log(all_probs + 1e-12), axis=2
    ).mean(axis=0)                             # (N,)

    # BALD = epistemic disagreement
    bald = pred_entropy - pass_entropies

    model.eval()   # restore clean eval state
    return bald
