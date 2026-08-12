"""Shared MLP backbone used by all four uncertainty methods.

Uses LayerNorm instead of BatchNorm so that MC Dropout inference is valid:
BatchNorm in eval() mode uses fixed running statistics, making all T dropout
passes identical. LayerNorm normalises per-sample and is unaffected by
eval/train mode, so each stochastic pass differs meaningfully.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 1024, num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(       # 1024 → 512  (= m1, shallower)
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(       # 512 → 256  (= m0, deeper)
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        m1 = self.block1(x)    # 512-dim, shallower
        m0 = self.block2(m1)   # 256-dim, deeper
        logits = self.head(m0)
        return logits, [m1, m0]


def build_mlp(input_dim: int = 1024, dropout: float = 0.3) -> MLP:
    return MLP(input_dim=input_dim, dropout=dropout)
