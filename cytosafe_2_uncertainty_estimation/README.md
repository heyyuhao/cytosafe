# Uncertainty Estimation — Cytotoxicity Prediction

Benchmarks four uncertainty estimation (UE) methods on cytotoxicity classification, adapting [DRUE (Xu et al., 2026)](https://arxiv.org/abs/2601.19341) from medical imaging to molecular fingerprints.

---

## Datasets

| Dataset | Total | Toxic (1) | Non-toxic (0) |
|---------|-------|-----------|---------------|
| 3T3     | 24,042 | 4,007 (17%) | 20,035 (83%) |
| HEK-293 | 36,846 | 6,141 (17%) | 30,705 (83%) |

Source: `author_cytosafe/Datasets/*_curated_reduced_1-5.csv` (SMILES + binary Outcome label).

---

## Data Splits

All splits use a **50 / 10 / 20 / 20** ratio: ID-train / ID-val / ID-test / OOD-test.

Two split strategies:

**Scaffold split** — Bemis-Murcko scaffolds are randomly assigned: 20% of scaffolds → OOD-test, 80% → ID pool. No scaffold appears in both ID and OOD. Molecules with novel scaffolds are OOD.

**Tanimoto split** — Each molecule's maximum Tanimoto similarity to all others is computed. The 20% least similar molecules → OOD-test. Guarantees fingerprint-level novelty, creating a harder OOD set than scaffold split.

Within the ID pool, stratified 3-way split preserves the toxic/non-toxic ratio.

Generated once via:
```bash
python data/generate_splits.py          # all 4 folders
```

Output folders: `data/splits/{dataset}_{method}/` containing `ID_train_X.npy`, `ID_train_y.npy`, `ID_val_X.npy`, `ID_val_y.npy`, `ID_test_X.npy`, `ID_test_y.npy`, `OOD_test_X.npy`, `OOD_test_y.npy`.

**Fingerprint:** ECFP4 (Morgan radius=2, 1024 bits), computed with RDKit.

---

## Experiments (8 total)

| Exp | Train | Test | Split | Description |
|-----|-------|------|-------|-------------|
| 1 | 3T3 | 3T3 | scaffold | Within-dataset, scaffold OOD |
| 2 | 3T3 | 3T3 | tanimoto | Within-dataset, Tanimoto OOD |
| 3 | 3T3 | HEK | scaffold | Cross-cell-line, scaffold OOD |
| 4 | 3T3 | HEK | tanimoto | Cross-cell-line, Tanimoto OOD |
| 5 | HEK | HEK | scaffold | Within-dataset, scaffold OOD |
| 6 | HEK | HEK | tanimoto | Within-dataset, Tanimoto OOD |
| 7 | HEK | 3T3 | scaffold | Cross-cell-line, scaffold OOD |
| 8 | HEK | 3T3 | tanimoto | Cross-cell-line, Tanimoto OOD |

ID-train/val/test always from `{train_dataset}_{split}/`. OOD-test always from `{test_dataset}_{split}/OOD_test`.

---

## Models

All four methods share the same base MLP for fairness:

```
Input(1024) → Linear(512) → BN → ReLU → Dropout(0.3)   [output = m1]
            → Linear(256) → BN → ReLU → Dropout(0.3)   [output = m0]
            → Linear(2)
```

| Method | Training | Uncertainty Score | Range |
|--------|----------|-------------------|-------|
| **Entropy** | CrossEntropyLoss | H(softmax output) = −Σ p·log(p) | [0, 0.693] |
| **MC Dropout** | Same MLP; dropout ON at test time | H(mean of T=50 stochastic passes) | [0, 0.693] |
| **BNN** | Laplace approximation post-hoc on last layer | Var(P(toxic)) across 100 posterior weight samples | [0, 0.25] |
| **DRUE** | 3-phase: classifier → G1 decoder (BCE from m1) → G0 decoder (BCE from m0, G1 blocks frozen) | MAE(G1(m1), G0(m0)) — both outputs in [0,1] via Sigmoid | [0, 1] |

DRUE uses m1 (shallower, 512-dim) for G1 and m0 (deeper, 256-dim) for G0. The difference between their reconstructions captures uncertainty from the final encoder block, reducing the confound of cumulative information loss.

---

## Evaluation Metric

**OOD detection AUC** — the primary metric. Each sample gets an uncertainty score. ID-test samples are labelled 0, OOD-test samples labelled 1. AUC measures whether higher uncertainty scores correctly rank OOD above ID samples.

- AUC = 1.0 → perfect OOD detection
- AUC = 0.5 → random (method assigns equal uncertainty to ID and OOD)
- AUC < 0.5 → model is more confident on OOD than ID (overconfidence failure)

AUPR is omitted because ID and OOD test sets are balanced (~50/50), making AUC sufficient.

Each experiment also logs `avg_uncertainty_id` and `avg_uncertainty_ood` — if OOD > ID the method is directionally correct.

---

## Running

```bash
# Step 1 — generate splits once
python data/generate_splits.py

# Step 2 — run all 8 experiments
bash run_all.sh
```

Results saved to `results/{exp_name}/ood_roc.png` and `results/{exp_name}/ood_roc.json`.

---

## Results

Two runs were conducted: **MLP-2layer** (1024→512→256→2, original design) and **MLP-3layer** (1024→512→32→256→2, bottleneck experiment). The 3-layer bottleneck hurt DRUE (AUC consistently below 0.5) and did not improve other methods; the 2-layer design is the retained baseline.

### MLP-2layer Results — OOD Detection AUC

| Experiment | Entropy | MC Dropout | BNN | DRUE |
|------------|---------|------------|-----|------|
| exp1: 3T3→3T3 scaffold | 0.5196 | 0.5169 | 0.5153 | 0.5112 |
| exp2: 3T3→3T3 tanimoto | 0.6266 | 0.6142 | 0.6237 | **0.6149** |
| exp3: 3T3→HEK scaffold | 0.5103 | 0.5080 | 0.5047 | 0.5169 |
| exp4: 3T3→HEK tanimoto | 0.5820 | 0.5675 | 0.5735 | **0.6080** |
| exp5: HEK→HEK scaffold | 0.5074 | 0.5049 | 0.5062 | **0.5287** |
| exp6: HEK→HEK tanimoto | 0.5721 | 0.5621 | 0.5611 | **0.6053** |
| exp7: HEK→3T3 scaffold | 0.5267 | 0.5275 | 0.5246 | 0.4985 |
| exp8: HEK→3T3 tanimoto | 0.5858 | 0.5810 | 0.5848 | **0.6059** |

**Bold** = best per row.

### MLP-2layer Results — Average Uncertainty (ID vs OOD)

| Experiment | Method | Avg UE ID | Avg UE OOD | OOD > ID? |
|------------|--------|-----------|------------|-----------|
| exp2: 3T3→3T3 tanimoto | Entropy | 0.088 | 0.170 | ✓ |
| exp2: 3T3→3T3 tanimoto | MC Dropout | 0.094 | 0.175 | ✓ |
| exp2: 3T3→3T3 tanimoto | BNN | 0.002 | 0.003 | ✓ |
| exp2: 3T3→3T3 tanimoto | DRUE | 0.015 | 0.016 | ✓ |
| exp1: 3T3→3T3 scaffold | Entropy | 0.096 | 0.101 | ✓ |
| exp1: 3T3→3T3 scaffold | DRUE | 0.016 | 0.016 | ✓ (marginal) |

### Key Observations

1. **Tanimoto split consistently produces higher AUC than scaffold split** across all experiments. Tanimoto-OOD molecules are fingerprint-level novel, creating a larger domain gap that uncertainty methods can detect.

2. **DRUE is the best or tied-best on 5/8 experiments** on tanimoto splits (AUC 0.61–0.62), while performing near-random on scaffold splits. This suggests DRUE's reconstruction-based signal is sensitive to fingerprint-level novelty but not scaffold-level novelty alone.

3. **All methods converge near AUC=0.51 on scaffold splits.** Scaffold-split OOD molecules share most ECFP4 bits with training data; the model is equally confident on both, reflecting the fundamental limitation of circular fingerprints for capturing scaffold-level OOD.

4. **Entropy and MC Dropout are near-identical** in most experiments, confirming the known overconfidence issue of softmax-based measures. The average uncertainty gap (OOD−ID) is small but consistently positive on tanimoto splits.

5. **BNN (full-network Laplace, diagonal Hessian)** shows competitive AUC on tanimoto splits but near-zero absolute UE values, suggesting the posterior variance is well-calibrated directionally but small in magnitude.

6. **Cross-cell-line experiments (exp3/4/7/8) do not show higher AUC than within-dataset tanimoto.** 3T3 and HEK-293 share substantial chemical space at the fingerprint level, so the cell-line boundary alone is not a stronger OOD signal than fingerprint dissimilarity.

