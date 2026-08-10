# CytoSafe — Cytotoxicity Prediction & Uncertainty Estimation

Research codebase studying cytotoxicity prediction and uncertainty estimation for drug discovery, using the 3T3 and HEK-293 cell line datasets from PubChem.

---

## Repository Structure

```
cytosafe/
├── author_cytosafe/          # Original author's datasets, notebooks, and pre-trained models
├── author_pipeline/          # Original author's reusable pipeline functions
├── author_DRUE/              # DRUE source code (git submodule: a-Fomalhaut-a/DRUE)
├── cytosafe_1_reproduction/  # Part 1: reproduction of the CytoSafe paper
├── cytosafe_2_uncertainty_estimation/  # Part 2: uncertainty estimation benchmark
├── environment.yml           # Conda environment (cytosafe)
└── images/
```

---

## Part 1 — CytoSafe Reproduction

**`cytosafe_1_reproduction/`**

Reproduces [Cyto-Safe: A Machine Learning Tool for Early Cytotoxicity Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01811) (J. Chem. Inf. Model., 2024).

- QSAR model: LGBM + ECFP4 fingerprints (radius=2, 1024 bits)
- Datasets: 3T3 (24,042 compounds) and HEK-293 (36,846 compounds)
- Includes: fingerprint generation, model training, evaluation against paper Table 1, and a web app for SMILES-based prediction with atom-level XAI heatmaps

See `cytosafe_1_reproduction/README.md` for details.

---

## Part 2 — Uncertainty Estimation Benchmark

**`cytosafe_2_uncertainty_estimation/`**

Adapts and benchmarks four uncertainty estimation (UE) methods on cytotoxicity classification, motivated by [DRUE (Xu et al., 2026)](https://arxiv.org/abs/2601.19341).

- Methods: Entropy, MC Dropout, BNN (Laplace approximation), DRUE
- OOD construction: Bemis-Murcko scaffold split and Tanimoto-distance split
- 8 experiments: 2 datasets × 2 split methods × within/cross-cell-line
- Evaluation: OOD detection AUC (ID-test=0, OOD-test=1, score=uncertainty)

Key finding: Tanimoto-based OOD split creates a harder and more informative domain gap than scaffold split; DRUE achieves best OOD detection AUC on tanimoto experiments.

See `cytosafe_2_uncertainty_estimation/README.md` for details.

---

## Setup

```bash
conda env create -f environment.yml
conda activate cytosafe

# Pull submodule
git submodule update --init
```
