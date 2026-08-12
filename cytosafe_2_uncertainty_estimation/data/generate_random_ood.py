"""Generate random ECFP4-like binary fingerprints as Tier-1 sanity-check OOD.

These are not real molecules — each vector has the same sparsity as real ECFP4
fingerprints (~4.6%, ≈47 bits set per 1024-bit vector) but bit positions are
chosen uniformly at random. Because real ECFP4 bits encode specific substructures,
random vectors at the same density are structurally meaningless and guaranteed to
be maximally OOD with respect to any chemistry-trained model.

Rationale: if a pipeline correctly detects these as OOD (AUC → 1.0 for all
methods), the pipeline is valid. Failure here would indicate a bug, not a
domain-gap issue.

Output:
    data/splits/random_tanimoto/OOD_test_X.npy   — (N, 1024) float32, values in {0,1}
    data/splits/random_tanimoto/OOD_test_y.npy   — (N,) float32, all zeros (no label meaning)
    data/splits/random_tanimoto/generation_info.json

N and sparsity are matched to data/splits/3T3_tanimoto/OOD_test_X.npy.
"""

import json
from pathlib import Path

import numpy as np

NBITS        = 1024
RANDOM_STATE = 42

# Match the real 3T3 tanimoto OOD split exactly
REFERENCE_X  = Path(__file__).resolve().parent / "splits" / "3T3_tanimoto" / "OOD_test_X.npy"
OUT_DIR      = Path(__file__).resolve().parent / "splits" / "random_tanimoto"


def main():
    # Measure reference sparsity
    ref_X     = np.load(REFERENCE_X)
    n_samples = ref_X.shape[0]
    avg_bits  = float(ref_X.sum(axis=1).mean())
    sparsity  = float(ref_X.mean())
    print(f"Reference: {n_samples} samples, avg bits set = {avg_bits:.2f}, "
          f"sparsity = {sparsity:.4f}")

    # Generate random fingerprints with same sparsity
    rng     = np.random.RandomState(RANDOM_STATE)
    n_bits  = int(round(avg_bits))   # ≈47 bits per vector
    X_rand  = np.zeros((n_samples, NBITS), dtype=np.float32)
    for i in range(n_samples):
        bits = rng.choice(NBITS, size=n_bits, replace=False)
        X_rand[i, bits] = 1.0

    # Labels are meaningless (no toxicity assay run on random vectors)
    # Use zeros so run_experiment.py can load without error
    y_rand = np.zeros(n_samples, dtype=np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "OOD_test_X.npy", X_rand)
    np.save(OUT_DIR / "OOD_test_y.npy", y_rand)

    info = {
        "description": "Random ECFP4-sparsity binary vectors — Tier-1 sanity-check OOD",
        "n_samples":   n_samples,
        "n_bits":      NBITS,
        "bits_set_per_vector": n_bits,
        "sparsity":    round(n_bits / NBITS, 6),
        "reference_sparsity": round(sparsity, 6),
        "random_state": RANDOM_STATE,
    }
    with open(OUT_DIR / "generation_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved {n_samples} random fingerprints → {OUT_DIR}/")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
