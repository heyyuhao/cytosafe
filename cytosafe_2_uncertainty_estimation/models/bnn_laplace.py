"""Method 3 — BNN via Laplace approximation.

Uses last-layer Laplace with a full Hessian over the 514-parameter
final Linear(256→2) layer. This is the theoretically valid setting:
- The last layer is linear-logistic regression over learned features
- A Gaussian posterior is well-defined for linear models
- pred_type="glm" (Taylor linearization) is exact for the last layer

Full-network Laplace with diagonal Hessian over 656K params was previously
used — this is invalid because the diagonal approximation loses all
correlations, and the GLM linearization is incorrect for nonlinear layers.

Uncertainty score = entropy of the mean predictive distribution across
N_SAMPLES posterior weight samples. H(E[p]) captures how much the
averaged posterior pushes the prediction toward 0.5 — more robust than
variance (which is bounded by 0.25 and misses confidently-wrong OOD).

Requires: pip install laplace-torch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 100


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

    wrapped = _LogitsOnly(model).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    # last_layer + full Hessian: valid and efficient (only ~514 parameters)
    la = Laplace(wrapped, likelihood="classification",
                 subset_of_weights="last_layer",
                 hessian_structure="full")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")
    return la


def uncertainty_scores(la, X: np.ndarray, n_samples: int = N_SAMPLES,
                       batch_size: int = 256) -> np.ndarray:
    """Return entropy of the mean predictive distribution.

    Draws n_samples weight vectors from the last-layer posterior,
    runs each through the network via GLM linearization (valid for
    last-layer), and returns H(E[p]) — entropy of the averaged softmax.
    """
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )
    entropies = []
    for (xb,) in loader:
        # samples shape: (n_samples, batch, n_classes)
        samples = la.predictive_samples(
            xb.to(DEVICE), pred_type="glm", n_samples=n_samples
        )
        mean_p = samples.mean(dim=0).detach().cpu().numpy()   # (batch, C)
        h = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=1)
        entropies.append(h)
    return np.concatenate(entropies)
