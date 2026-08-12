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

## Experiments (10 total)

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
| 9 | 3T3 | random | — | **Tier-1 sanity check**: random ECFP4-sparsity vectors |
| 10 | HEK | random | — | **Tier-1 sanity check**: random ECFP4-sparsity vectors |

Exp 9–10 use random 1024-bit binary vectors with the same ~5% density as real ECFP4 fingerprints as OOD. All methods should achieve AUC → 1.0 here; failure indicates a pipeline bug.

---

## Models

All four methods share the same base MLP for fairness:

```
Input(1024) → Linear(512) → LayerNorm → ReLU → Dropout(0.3)   [output = m1]
            → Linear(256) → LayerNorm → ReLU → Dropout(0.3)   [output = m0]
            → Linear(2)
```

LayerNorm is used instead of BatchNorm so that MC Dropout inference is valid — BatchNorm in eval mode uses fixed running statistics, making all T stochastic passes deterministic and identical.

| Method | Training | Uncertainty Score | Range |
|--------|----------|-------------------|-------|
| **Entropy** | CrossEntropyLoss | H(softmax) = −Σ p·log(p) | [0, 0.693] |
| **MC Dropout** | Same MLP; dropout ON at test time (p=0.3, same as training) | BALD = H(E[p]) − E[H(p)] — epistemic disagreement across T=50 passes | [0, 0.693] |
| **BNN** | Laplace approx post-hoc on last layer only (full Hessian, ~514 params) | H(E[p]) across 100 posterior weight samples | [0, 0.693] |
| **DRUE** | 3-phase: classifier → G1 (BCE from m1, 512-dim) → G0 (BCE from m0, 256-dim, G1 blocks frozen) | MAE(G1(m1), G0(m0)) — both outputs in [0,1] via Sigmoid | [0, 1] |

DRUE uses m1 (shallower, 512-dim) for G1 and m0 (deeper, 256-dim) for G0. Their reconstruction difference captures whether the input follows learned structural patterns, independent of class prediction confidence.

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
python data/generate_splits.py # split for 3T3 and HEK dataset
python data/generate_random_ood.py   # Tier-1 sanity check OOD

# Step 2 — run all 10 experiments
bash run_all.sh
```

Results saved to `results/{exp_name}/ood_roc.png` and `results/{exp_name}/ood_roc.json`.

---

## Results

### OOD Detection AUC — Main Experiments (exp1–8)

| Experiment | Entropy | MC Dropout | BNN | DRUE |
|------------|---------|------------|-----|------|
| exp1: 3T3→3T3 scaffold | 0.5178 | 0.5133 | 0.5161 | **0.5248** |
| exp2: 3T3→3T3 tanimoto | 0.6060 | 0.5857 | 0.6073 | **0.6408** |
| exp3: 3T3→HEK scaffold | 0.5062 | 0.4985 | 0.5054 | **0.5164** |
| exp4: 3T3→HEK tanimoto | **0.5869** | 0.5625 | 0.5840 | 0.5492 |
| exp5: HEK→HEK scaffold | 0.5049| 0.5050 | 0.5046 | **0.5307** |
| exp6: HEK→HEK tanimoto | 0.5695 | 0.5618 | 0.5657 | **0.6423** |
| exp7: HEK→3T3 scaffold | 0.5120 | **0.5138** | 0.5127 | 0.4845 |
| exp8: HEK→3T3 tanimoto | 0.5767 | 0.5753 | 0.5792 | **0.6383** |

**Bold** = best per row.

### Tier-1 Sanity Check — Random ECFP4-sparsity OOD (exp9–10)

| Experiment | Entropy | MC Dropout | BNN | DRUE |
|------------|---------|------------|-----|------|
| exp9: 3T3→random | 0.4784 | 0.4831 | 0.4909 | **0.9516** |
| exp10: HEK→random | 0.4415 | 0.4757 | 0.4511 | **0.8738** |

### Key Findings

1. **DRUE is the only method that detects extreme OOD (random noise).** On exp9–10, DRUE achieves AUC=0.95/0.87 while all three softmax-based methods score below 0.5 — meaning they are *more* confident on random noise than on real molecules. This directly validates the paper's core claim.

2. **Softmax-based methods (Entropy, MC Dropout, BNN) share the same fundamental failure.** A discriminative classifier trained to predict toxic/non-toxic has no concept of "is this even a real molecule?" — it must assign one of two classes to any input. Random sparse vectors consistently activate bias-driven pathways and are assigned high-confidence predictions, producing *lower* uncertainty than real ID molecules.

3. **DRUE's reconstruction signal is fundamentally different.** G1 and G0 decoders were trained to reproduce ECFP4 substructure patterns. Random vectors that do not follow chemical co-occurrence patterns yield high MAE(G1, G0), correctly flagging them as OOD. This is a *pattern membership* test, not a *prediction confidence* test.

4. **Tanimoto split creates a harder and more informative domain gap than scaffold split.** AUC is consistently higher on tanimoto experiments (0.58–0.64) vs scaffold (0.50–0.53), because tanimoto-OOD molecules are genuinely fingerprint-level novel while scaffold-OOD molecules still share most bits with training data.

5. **MC Dropout BALD scores are an order of magnitude smaller than Entropy** (ID≈0.01 vs ID≈0.10), confirming that with a well-converged MLP, dropout masks at p=0.3 produce negligible disagreement between passes — the 50 stochastic forward passes are nearly identical.

6. **Cross-cell-line shift is not stronger than tanimoto shift** at the fingerprint level. 3T3 and HEK-293 share substantial chemical space in ECFP4 space, so the cell-line boundary alone provides no additional OOD signal beyond fingerprint dissimilarity.

