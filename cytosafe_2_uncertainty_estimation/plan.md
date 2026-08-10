# Uncertainty Estimation for Cytotoxicity Prediction — Experiment Plan

## Goal

Reproduce and adapt the DRUE uncertainty estimation framework
([Xu et al., arXiv:2601.19341](https://arxiv.org/abs/2601.19341)) to a molecular
property prediction setting, and benchmark it against three baseline methods on two
cytotoxicity datasets (3T3 and HEK-293). This is the foundation for a paper submission
on uncertainty estimation in drug discovery.

---

## Datasets

Source: `author_cytosafe/Datasets/*_curated_reduced_1-5.csv` (SMILES + binary labels).
The existing `reproduction/data/fp/` files use a random 80/20 split; we discard that
split and rebuild from SMILES using a scaffold-based split.

| Dataset | Total | Toxic (1) | Non-toxic (0) |
|---------|-------|-----------|---------------|
| 3T3     | 24,042 | 4,007 (17%) | 20,035 (83%) |
| HEK-293 | 36,846 | 6,141 (17%) | 30,705 (83%) |

---

## Data Pipeline — Scaffold Split

Bemis-Murcko scaffold split is the standard OOD benchmark in cheminformatics: molecules
sharing a scaffold are grouped together, so the OOD set contains genuinely novel chemical
scaffolds unseen during training.

**Steps** (`data/scaffold_split.py`):

1. Compute Bemis-Murcko scaffold (RDKit `MurckoDecompose`) for every SMILES.
2. Assign scaffolds to ID pool (80% of scaffolds) or OOD-test (20% of scaffolds).
   Molecules whose scaffold appears in the OOD pool → OOD-test set.
3. Within the ID pool, stratified split by compound: 75% → ID-train, 25% → ID-test.
   Final ratio is approximately 60% / 20% / 20% of total compounds.
4. Compute ECFP4 (Morgan radius=2, 1024 bits, `useFeatures=False`) for each split.
5. Save as numpy `.npz` files: `{dataset}_{split}_X.npy`, `{dataset}_{split}_y.npy`.

No scaffold appears in both ID-train and OOD-test by construction.

---

## Methods

All four methods share the **same base MLP architecture** for a fair comparison.

### Shared MLP Backbone

```
Input(1024) → Linear(512) → BN → ReLU → Dropout(0.3)
            → Linear(256) → BN → ReLU → Dropout(0.3)
            → Linear(2)   → (softmax at inference)
```

### Method 1 — Entropy (baseline)

- Train: standard CrossEntropyLoss.
- Uncertainty score: Shannon entropy of the softmax output.
  `H = -sum(p * log(p))`
- No architectural changes; no retraining needed beyond the base MLP.

### Method 2 — MC Dropout

- Train: same MLP, dropout layers remain.
- Inference: keep dropout **on**, run T=50 stochastic forward passes.
- Uncertainty score: predictive entropy over the T softmax distributions.
- Same weights as Entropy (train once, evaluate differently).

### Method 3 — BNN (Laplace Approximation)

- Train: same MLP with standard CrossEntropyLoss.
- Post-hoc: fit a Laplace approximation over the last linear layer weights
  using `laplace-torch` (no retraining).
- Uncertainty score: variance of the posterior predictive distribution.

### Method 4 — DRUE (adapted from Xu et al.)

Architecture mirrors the paper but uses MLP layers instead of ResNet blocks,
and BCE loss instead of MSE (appropriate for binary fingerprints).

```
Encoder F:  Input(1024) → Linear(512) → BN → ReLU   [penultimate = m1]
                        → Linear(256) → BN → ReLU   [final = m0]

Decoder G1 (takes m1):  Linear(256) → BN → ReLU → Linear(512) → BN → ReLU → Linear(1024) → Sigmoid
Decoder G0 (takes m0):  Linear(1024, extra first block) → [shared G1 blocks, frozen] → Sigmoid
```

Training procedure (three sequential steps):
1. Train classifier (encoder F + classification head) with CrossEntropyLoss.
2. Train G1 with BCE loss reconstructing input from m1; encoder frozen.
3. Train G0's first block with BCE loss reconstructing input from m0;
   encoder frozen, G1 blocks (shared into G0) frozen.

Uncertainty score:
`UD(x) = MAE(G1(m1), G0(m0))`  — pixel-wise mean absolute difference between
the two reconstructions, both normalised to [0, 1] via Sigmoid.

---

## Evaluation

Both evaluations are performed on each of the two cell lines independently.

### Primary — OOD Detection (mirrors DRUE paper Table 1)

- Pool ID-test (label = 0, in-distribution) and OOD-test (label = 1, out-of-distribution).
- Rank all samples by uncertainty score.
- Report **AUC** (ROC) and **AUPR** (precision-recall, OOD as positive class).
- Threshold-free: AUC/AUPR sweep all possible thresholds implicitly.

### Secondary — Selective Prediction (risk-coverage curve)

- On ID-test only, sort samples by uncertainty score ascending (most confident first).
- For coverage levels 10%, 20%, …, 100%, retain the most-confident fraction and
  compute classification accuracy.
- Plot coverage (x-axis) vs accuracy (y-axis) for all four methods.
- Practical interpretation: a chemist who only acts on high-confidence predictions
  — does DRUE give a better confidence filter than Entropy?

---

## Code Structure

```
Yuhao_Experiment/
├── plan.md                     # this file
├── data/
│   └── scaffold_split.py       # SMILES → scaffold split → ECFP4 → .npy files
├── models/
│   ├── mlp.py                  # shared MLP backbone (classifier)
│   ├── entropy.py              # method 1: train + score
│   ├── mc_dropout.py           # method 2: train + score
│   ├── bnn_laplace.py          # method 3: train + score
│   └── drue.py                 # method 4: encoder + G1 + G0, train + score
├── evaluate.py                 # AUC, AUPR, selective prediction, plots
└── run_experiment.py           # end-to-end runner: one call per dataset
```

---

## Execution Order

```bash
# 1. Generate scaffold splits and fingerprints
python data/scaffold_split.py --dataset 3T3
python data/scaffold_split.py --dataset HEK293

# 2. Run all methods end-to-end
python run_experiment.py --dataset 3T3
python run_experiment.py --dataset HEK293

# Results saved to results/{dataset}/
```

---

## Key Differences from the Original DRUE Paper

| Aspect | Original DRUE | This work |
|--------|---------------|-----------|
| Domain | Medical imaging (fundus) | Drug discovery (cytotoxicity) |
| Input | RGB images (224×224) | ECFP4 binary fingerprints (1024-bit) |
| Backbone | ResNet-18 | 3-layer MLP |
| Decoder loss | MSE | BCE (appropriate for binary inputs) |
| OOD construction | Different imaging modalities | Bemis-Murcko scaffold split |
| ID task | Glaucoma detection | Cytotoxicity classification |
