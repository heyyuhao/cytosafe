"""Generate all data splits for uncertainty estimation experiments.

Produces 4 split folders, each with ID_train/ID_val/ID_test/OOD_test .npy files:
    data/splits/3T3_scaffold/
    data/splits/3T3_tanimoto/
    data/splits/HEK_scaffold/
    data/splits/HEK_tanimoto/

Split ratios (of total compounds per dataset):
    ID-train  ~50%   (scaffold: 62.5% of ID pool × 80% of scaffolds)
    ID-val    ~10%   (scaffold: 12.5% of ID pool × 80% of scaffolds)
    ID-test   ~20%   (scaffold: 25%   of ID pool × 80% of scaffolds)
    OOD-test  ~20%   (scaffold: 20% of scaffolds; tanimoto: 20% least similar)

Usage:
    # Generate all 4 combinations at once
    python data/generate_splits.py

    # Or selectively
    python data/generate_splits.py --dataset 3T3 --method scaffold
    python data/generate_splits.py --dataset HEK --method tanimoto
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

DATASETS = {
    "3T3": "3T3_curated_reduced_1-5.csv",
    "HEK": "HEK_curated_reduced_1-5.csv",
}
DATA_DIR   = Path(__file__).resolve().parents[3] / "author_cytosafe" / "Datasets"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"

RADIUS            = 2
NBITS             = 1024
RANDOM_STATE      = 42
OOD_FRAC          = 0.20
ID_TEST_FRAC      = 0.25   # of the ID pool
ID_VAL_FRAC       = 0.125  # of the ID pool


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def smiles_to_ecfp4(smiles_list):
    """Return (fp_array [N,1024], rdkit_fp_list, valid_indices)."""
    fps_arr, mol_fps, valid_idx = [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, NBITS, useFeatures=False)
        arr = np.zeros(NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_arr.append(arr)
        mol_fps.append(fp)
        valid_idx.append(i)
    return np.array(fps_arr, dtype=np.float32), mol_fps, valid_idx


def stratified_3way(indices, labels, seed=RANDOM_STATE):
    """Split index array into train/val/test preserving class ratio."""
    indices  = np.array(indices)
    labels   = np.array(labels)
    rng      = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        nt = max(1, int(len(cls_idx) * ID_TEST_FRAC))
        nv = max(1, int(len(cls_idx) * ID_VAL_FRAC))
        te.extend(cls_idx[:nt])
        va.extend(cls_idx[nt:nt + nv])
        tr.extend(cls_idx[nt + nv:])
    return np.array(tr), np.array(va), np.array(te)


def save_split(out_dir, split_name, X, y):
    np.save(out_dir / f"{split_name}_X.npy", X)
    np.save(out_dir / f"{split_name}_y.npy", y)
    n_toxic = int(y.sum())
    print(f"    {split_name:12s}: {len(X):5d}  toxic={n_toxic}  non-toxic={len(X)-n_toxic}")
    return {"n_total": len(X), "n_toxic": n_toxic, "n_nontoxic": len(X) - n_toxic}


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------

def run_scaffold_split(smiles, labels, fps_arr, valid_idx, seed=RANDOM_STATE):
    rng = random.Random(seed)

    scaffold_to_pos = defaultdict(list)  # scaffold → positions in valid_idx list
    for pos, orig_idx in enumerate(valid_idx):
        smi = smiles[orig_idx]
        mol = Chem.MolFromSmiles(smi)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) \
                 if mol else ""
        except Exception:
            sc = ""
        scaffold_to_pos[sc].append(pos)

    scaffolds = list(scaffold_to_pos.keys())
    rng.shuffle(scaffolds)

    n_ood = max(1, int(len(scaffolds) * OOD_FRAC))
    ood_positions = []
    for sc in scaffolds[:n_ood]:
        ood_positions.extend(scaffold_to_pos[sc])
    id_positions = []
    for sc in scaffolds[n_ood:]:
        id_positions.extend(scaffold_to_pos[sc])

    id_positions  = np.array(id_positions)
    ood_positions = np.array(ood_positions)
    id_labels     = np.array(labels)[np.array(valid_idx)[id_positions]]

    tr_pos, va_pos, te_pos = stratified_3way(id_positions, id_labels, seed)

    return (fps_arr[tr_pos],  np.array(labels)[np.array(valid_idx)[tr_pos]],
            fps_arr[va_pos],  np.array(labels)[np.array(valid_idx)[va_pos]],
            fps_arr[te_pos],  np.array(labels)[np.array(valid_idx)[te_pos]],
            fps_arr[ood_positions], np.array(labels)[np.array(valid_idx)[ood_positions]])


# ---------------------------------------------------------------------------
# Tanimoto split
# ---------------------------------------------------------------------------

def run_tanimoto_split(smiles, labels, fps_arr, mol_fps, valid_idx, seed=RANDOM_STATE):
    n = len(mol_fps)
    print(f"    Computing max Tanimoto similarity for {n} molecules ...")
    max_sim = np.zeros(n, dtype=np.float32)
    for i in range(n):
        others = mol_fps[:i] + mol_fps[i + 1:]
        sims   = DataStructs.BulkTanimotoSimilarity(mol_fps[i], others)
        max_sim[i] = max(sims) if sims else 0.0

    order     = np.argsort(max_sim)        # ascending: least similar first
    n_ood     = max(1, int(n * OOD_FRAC))
    ood_pos   = order[:n_ood]
    id_pos    = order[n_ood:]

    print(f"    OOD max_sim: [{max_sim[ood_pos].min():.3f}, {max_sim[ood_pos].max():.3f}]")
    print(f"    ID  max_sim: [{max_sim[id_pos].min():.3f},  {max_sim[id_pos].max():.3f}]")

    id_labels = np.array(labels)[np.array(valid_idx)[id_pos]]
    tr_pos, va_pos, te_pos = stratified_3way(id_pos, id_labels, seed)

    return (fps_arr[tr_pos],  np.array(labels)[np.array(valid_idx)[tr_pos]],
            fps_arr[va_pos],  np.array(labels)[np.array(valid_idx)[va_pos]],
            fps_arr[te_pos],  np.array(labels)[np.array(valid_idx)[te_pos]],
            fps_arr[ood_pos], np.array(labels)[np.array(valid_idx)[ood_pos]])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(dataset: str, method: str):
    import pandas as pd

    print(f"\n{'='*55}")
    print(f"Dataset={dataset}  Method={method}")

    df     = pd.read_csv(DATA_DIR / DATASETS[dataset])
    smiles = df["SMILES"].tolist()
    labels = df["Outcome"].astype(int).tolist()
    print(f"  Loaded {len(df)} compounds")

    fps_arr, mol_fps, valid_idx = smiles_to_ecfp4(smiles)
    print(f"  Valid SMILES: {len(valid_idx)}")

    if method == "scaffold":
        X_tr, y_tr, X_va, y_va, X_te, y_te, X_ood, y_ood = run_scaffold_split(
            smiles, labels, fps_arr, valid_idx)
    elif method == "tanimoto":
        X_tr, y_tr, X_va, y_va, X_te, y_te, X_ood, y_ood = run_tanimoto_split(
            smiles, labels, fps_arr, mol_fps, valid_idx)
    else:
        raise ValueError(f"Unknown method: {method}")

    out_dir = SPLITS_DIR / f"{dataset}_{method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"dataset": dataset, "method": method, "splits": {}}
    for name, X, y in [("ID_train", X_tr, y_tr), ("ID_val",  X_va, y_va),
                        ("ID_test",  X_te, y_te), ("OOD_test", X_ood, y_ood)]:
        summary["splits"][name] = save_split(out_dir, name,
                                              X, y.astype(np.float32))

    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--method",  choices=["scaffold", "tanimoto"],  default=None)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    methods  = [args.method]  if args.method  else ["scaffold", "tanimoto"]

    for ds in datasets:
        for mt in methods:
            generate(ds, mt)
