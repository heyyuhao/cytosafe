"""Evaluation: OOD detection via uncertainty scores.

One ROC per experiment: pool ID-test (label=0) + OOD-test (label=1),
rank by uncertainty score, compute AUC and AUPR.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

COLORS = {
    "Entropy":    "#1f77b4",
    "MC_Dropout": "#ff7f0e",
    "BNN":        "#2ca02c",
    "DRUE":       "#d62728",
}


def compute_ood_roc(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """ROC for OOD detection: ID=0, OOD=1, score=uncertainty.

    Higher uncertainty → predicted OOD.
    Returns fpr, tpr, AUC and avg uncertainty per split.
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    return {
        "auc":                 round(roc_auc, 4),
        "fpr":                 fpr.tolist(),
        "tpr":                 tpr.tolist(),
        "thresholds":          thresholds.tolist(),
        "n_id":                int(len(id_scores)),
        "n_ood":               int(len(ood_scores)),
        "avg_uncertainty_id":  round(float(id_scores.mean()),  6),
        "avg_uncertainty_ood": round(float(ood_scores.mean()), 6),
    }


def plot_roc(method_roc: dict, save_path: Path, title: str):
    """One figure, one curve per method."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    for method, roc in method_roc.items():
        color = COLORS.get(method)
        ax.plot(roc["fpr"], roc["tpr"], linewidth=1.8, color=color,
                label=f"{method}  AUC={roc['auc']:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def save_json(data: dict, save_path: Path):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {save_path}")
