"""Method 3 — BNN via Laplace approximation.

Fits a Laplace approximation post-hoc on the last linear layer of the
already-trained MLP. No retraining required.

Uncertainty score = variance of P(toxic) across N_SAMPLES weight posterior
samples. This is the native BNN uncertainty — not entropy of a point
prediction, but spread of predictions under the weight distribution.

Requires: pip install laplace-torch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES  = 100   # number of weight posterior samples


class _LogitsOnly(nn.Module):
    """Thin wrapper so laplace-torch receives a plain tensor, not a tuple."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def fit_laplace(model, X_train: np.ndarray, y_train: np.ndarray,
                batch_size: int = 256):
    """Fit a last-layer Laplace approximation and return the LA object."""
    from laplace import Laplace

    # Wrap model so laplace-torch gets plain logit tensor, not (logits, features)
    wrapped = _LogitsOnly(model).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    la = Laplace(wrapped, likelihood="classification", subset_of_weights="all",
                 hessian_structure="diag")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")
    return la


def uncertainty_scores(la, X: np.ndarray, n_samples: int = N_SAMPLES,
                       batch_size: int = 256) -> np.ndarray:
    """Return variance of P(toxic) across posterior weight samples.

    For each input x:
      - Draw n_samples weight vectors from the Laplace posterior
      - Each gives a different P(toxic) prediction
      - Variance across those predictions = epistemic uncertainty

    High variance → the posterior is uncertain about this input → likely OOD.
    Range: [0, 0.25]  (variance of a Bernoulli is p*(1-p), max at p=0.5)
    """
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )
    variances = []
    for (xb,) in loader:
        # samples shape: (n_samples, batch, n_classes)
        samples = la.predictive_samples(
            xb.to(DEVICE), pred_type="glm", n_samples=n_samples
        )
        # Take P(toxic) = class index 1, shape: (n_samples, batch)
        p_toxic = samples[:, :, 1].detach().cpu().numpy()
        # Variance across samples for each input, shape: (batch,)
        var = p_toxic.var(axis=0)
        variances.append(var)
    return np.concatenate(variances)
