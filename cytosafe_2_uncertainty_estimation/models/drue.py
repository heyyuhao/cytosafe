"""Method 4 — DRUE adapted for binary fingerprint inputs.

Architecture (2-block encoder):
    Encoder F:
        block1: 1024 → 512  (= m1, shallower)
        block2:  512 → 256  (= m0, deeper)
    Classifier head: Linear(256 → 2)

    Decoder G1: takes m1 (512-dim, shallower)
                512 → 512 → 1024·Sigmoid
    Decoder G0: takes m0 (256-dim, deeper)
                g0_extra: 256 → 512  (maps m0 back to m1 space)
                then G1's blocks (frozen): 512 → 512 → 1024·Sigmoid

    Uncertainty: UD(x) = MAE(G1(m1), G0(m0))
        G1 reconstructs from shallower 512-dim features (less information loss).
        G0 reconstructs from deeper 256-dim features (more information loss).
        Their difference captures the effect of the final encoder block,
        isolating true uncertainty from cumulative information loss.

Training order:
    1. train_classifier()  — CrossEntropyLoss, all params trainable
    2. train_g1()          — BCE from m1 (512-dim); encoder + head frozen
    3. train_g0()          — BCE from m0 (256-dim); encoder + head + G1 blocks frozen,
                             only g0_extra trained
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Decoder modules
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, final: bool = False):
        super().__init__()
        if final:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class G1(nn.Module):
    """Decoder from m1 (512-dim, shallower) → fingerprint (1024-dim)."""

    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.block_a = DecoderBlock(512, 512)
        self.block_b = DecoderBlock(512, input_dim, final=True)

    def forward(self, m1: torch.Tensor) -> torch.Tensor:
        return self.block_b(self.block_a(m1))


class G0(nn.Module):
    """Decoder from m0 (256-dim, deeper) → fingerprint (1024-dim).

    g0_extra maps m0 (256-dim) back to m1 space (512-dim).
    block_a and block_b are shared with G1 (frozen after G1 training).
    """

    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.g0_extra = DecoderBlock(256, 512)            # trained in phase 3
        self.block_a  = DecoderBlock(512, 512)            # shared/frozen from G1
        self.block_b  = DecoderBlock(512, input_dim, final=True)  # shared/frozen

    def forward(self, m0: torch.Tensor) -> torch.Tensor:
        z = self.g0_extra(m0)
        return self.block_b(self.block_a(z))


# ---------------------------------------------------------------------------
# Full DRUE model
# ---------------------------------------------------------------------------

class DRUEModel(nn.Module):
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim

        self.block1 = nn.Sequential(       # 1024 → 512  (= m1)
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(       # 512 → 256  (= m0)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, 2)

        self.g1 = G1(input_dim)
        self.g0 = G0(input_dim)

    def encode_both(self, x):
        m1 = self.block1(x)    # 512-dim, shallower
        m0 = self.block2(m1)   # 256-dim, deeper
        return m1, m0

    def classify(self, x):
        m1, m0 = self.encode_both(x)
        return self.head(m0), m0

    def reconstruct_g1(self, m1):
        return self.g1(m1)

    def reconstruct_g0(self, m0):
        return self.g0(m0)


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def train_classifier(model: DRUEModel, X_train, y_train,
                     X_val=None, y_val=None,
                     epochs=50, batch_size=256, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.astype(int), dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model.classify(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if X_val is not None:
            val_loss = _classifier_val_loss(model, criterion, X_val, y_val, batch_size)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"  [Classifier] Early stop at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)


def _classifier_val_loss(model, criterion, X_val, y_val, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val.astype(int), dtype=torch.long),
        ), batch_size=batch_size,
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, _ = model.classify(xb)
            total += criterion(logits, yb).item() * len(yb)
            n += len(yb)
    return total / n


def _train_decoder(model: DRUEModel, decoder: nn.Module, feature_fn,
                   X_train, epochs=50, batch_size=256, lr=1e-3, patience=10):
    params = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )

    best_loss, best_state, no_improve = float("inf"), None, 0
    model.eval()
    for epoch in range(epochs):
        decoder.train()
        epoch_loss, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                feat = feature_fn(xb)
            recon = decoder(feat)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        avg = epoch_loss / n
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  [Decoder] Early stop at epoch {epoch + 1}")
            break

    if best_state:
        decoder.load_state_dict(best_state)


def train_g1(model: DRUEModel, X_train, **kwargs):
    """Phase 2: train G1 from m1 (512-dim). Encoder frozen."""
    for p in model.block1.parameters(): p.requires_grad_(False)
    for p in model.block2.parameters(): p.requires_grad_(False)
    for p in model.head.parameters():   p.requires_grad_(False)

    def get_m1(x):
        m1, _ = model.encode_both(x)
        return m1

    _train_decoder(model, model.g1, get_m1, X_train, **kwargs)


def train_g0(model: DRUEModel, X_train, **kwargs):
    """Phase 3: train G0's g0_extra from m0 (256-dim).
    Encoder, head, and G1 shared blocks frozen.
    """
    model.g0.block_a.load_state_dict(model.g1.block_a.state_dict())
    model.g0.block_b.load_state_dict(model.g1.block_b.state_dict())
    for p in model.g0.block_a.parameters(): p.requires_grad_(False)
    for p in model.g0.block_b.parameters(): p.requires_grad_(False)

    def get_m0(x):
        _, m0 = model.encode_both(x)
        return m0

    _train_decoder(model, model.g0, get_m0, X_train, **kwargs)


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

def uncertainty_scores(model: DRUEModel, X: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """Return UD(x) = MAE(G1(m1), G0(m0)) per sample. Range [0, 1]."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
    )
    scores = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            m1, m0 = model.encode_both(xb)
            g1_out = model.reconstruct_g1(m1)
            g0_out = model.reconstruct_g0(m0)
            mae = (g1_out - g0_out).abs().mean(dim=1)
            scores.append(mae.cpu().numpy())
    return np.concatenate(scores, axis=0)
