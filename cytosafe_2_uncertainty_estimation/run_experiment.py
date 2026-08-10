"""End-to-end experiment runner — config-driven.

Usage:
    python run_experiment.py --config configs/exp1_3T3_scaffold.yaml
    python run_experiment.py --config configs/exp2_3T3_tanimoto.yaml
    python run_experiment.py --config configs/exp3_3T3_to_HEK.yaml
    python run_experiment.py --config configs/exp4_HEK_to_3T3.yaml

Outputs in results/{exp_name}/:
    ood_roc.png   — one ROC curve per method (ID=0, OOD=1, score=uncertainty)
    ood_roc.json  — AUC, avg UE per split, raw fpr/tpr per method
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from yuhao_codebase.cytosafe.cytosafe_UE.evaluate import compute_ood_roc, plot_roc, save_json

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS_DIR  = Path(__file__).parent / "data" / "splits"
RESULTS_DIR = Path(__file__).parent / "results"

TRAIN_KWARGS   = dict(epochs=100, batch_size=256, lr=1e-3, patience=15)
DECODER_KWARGS = dict(epochs=100, batch_size=256, lr=1e-3, patience=15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(cfg: dict) -> dict:
    train = cfg["train_dataset"]
    test  = cfg["test_dataset"]
    split = cfg["split_method"]
    id_dir  = SPLITS_DIR / f"{train}_{split}"
    ood_dir = SPLITS_DIR / f"{test}_{split}"
    def npy(d, name): return np.load(d / f"{name}.npy")
    return {
        "X_tr":  npy(id_dir,  "ID_train_X"), "y_tr":  npy(id_dir,  "ID_train_y"),
        "X_val": npy(id_dir,  "ID_val_X"),   "y_val": npy(id_dir,  "ID_val_y"),
        "X_id":  npy(id_dir,  "ID_test_X"),
        "X_ood": npy(ood_dir, "OOD_test_X"),
    }


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy H = -sum(p*log(p)), range [0, log(2)~0.693]."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def get_softmax(model, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            logits, _ = model(xb.to(DEVICE))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict):
    import yuhao_codebase.cytosafe.cytosafe_UE.models.entropy    as entropy_mod
    import yuhao_codebase.cytosafe.cytosafe_UE.models.mc_dropout as mc_mod
    import yuhao_codebase.cytosafe.cytosafe_UE.models.bnn_laplace as bnn_mod
    import yuhao_codebase.cytosafe.cytosafe_UE.models.drue       as drue_mod

    exp_name = cfg["exp_name"]
    data     = load_splits(cfg)
    X_tr  = data["X_tr"];  y_tr  = data["y_tr"].astype(int)
    X_val = data["X_val"]; y_val = data["y_val"].astype(int)
    X_id  = data["X_id"]
    X_ood = data["X_ood"]

    out_dir = RESULTS_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nExperiment: {exp_name}")
    print(f"  ID-train={len(X_tr)}  ID-val={len(X_val)}  "
          f"ID-test={len(X_id)}  OOD-test={len(X_ood)}")

    method_roc = {}

    # ------------------------------------------------------------------
    # Base MLP — shared by Entropy, MC Dropout, BNN
    # ------------------------------------------------------------------
    print("\n[1/4] Training base MLP ...")
    mlp = entropy_mod.train(X_tr, y_tr, X_val=X_val, y_val=y_val, **TRAIN_KWARGS)
    mlp.to(DEVICE)

    # Entropy
    print("  Scoring Entropy ...")
    id_scores  = entropy_from_probs(get_softmax(mlp, X_id))
    ood_scores = entropy_from_probs(get_softmax(mlp, X_ood))
    method_roc["Entropy"] = compute_ood_roc(id_scores, ood_scores)
    print(f"    avg UE  ID={id_scores.mean():.4f}  OOD={ood_scores.mean():.4f}  "
          f"AUC={method_roc['Entropy']['auc']}")

    # MC Dropout
    print("  Scoring MC Dropout ...")
    id_scores  = mc_mod.uncertainty_scores(mlp, X_id)
    ood_scores = mc_mod.uncertainty_scores(mlp, X_ood)
    method_roc["MC_Dropout"] = compute_ood_roc(id_scores, ood_scores)
    print(f"    avg UE  ID={id_scores.mean():.4f}  OOD={ood_scores.mean():.4f}  "
          f"AUC={method_roc['MC_Dropout']['auc']}")

    # BNN (Laplace)
    print("  Fitting Laplace (BNN) ...")
    try:
        la = bnn_mod.fit_laplace(mlp, X_tr, y_tr)
        id_scores  = bnn_mod.uncertainty_scores(la, X_id)
        ood_scores = bnn_mod.uncertainty_scores(la, X_ood)
        method_roc["BNN"] = compute_ood_roc(id_scores, ood_scores)
        print(f"    avg UE  ID={id_scores.mean():.4f}  OOD={ood_scores.mean():.4f}  "
              f"AUC={method_roc['BNN']['auc']}")
    except ImportError:
        print("    laplace-torch not installed — skipping BNN.")

    # ------------------------------------------------------------------
    # DRUE
    # ------------------------------------------------------------------
    print("\n[2/4] Training DRUE ...")
    drue = drue_mod.DRUEModel(input_dim=X_tr.shape[1]).to(DEVICE)
    print("  Phase 1: classifier ...")
    drue_mod.train_classifier(drue, X_tr, y_tr, X_val=X_val, y_val=y_val,
                              **TRAIN_KWARGS)
    print("  Phase 2: G1 decoder ...")
    drue_mod.train_g1(drue, X_tr, **DECODER_KWARGS)
    print("  Phase 3: G0 decoder ...")
    drue_mod.train_g0(drue, X_tr, **DECODER_KWARGS)

    id_scores  = drue_mod.uncertainty_scores(drue, X_id)
    ood_scores = drue_mod.uncertainty_scores(drue, X_ood)
    method_roc["DRUE"] = compute_ood_roc(id_scores, ood_scores)
    print(f"    avg UE  ID={id_scores.mean():.4f}  OOD={ood_scores.mean():.4f}  "
          f"AUC={method_roc['DRUE']['auc']}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plot_roc(method_roc, out_dir / "ood_roc.png",
             title=f"OOD Detection — {exp_name}")
    save_json(method_roc, out_dir / "ood_roc.json")
    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg  = yaml.safe_load(open(args.config))
    run(cfg)
