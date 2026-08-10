#!/usr/bin/env bash
# Run all 8 uncertainty estimation experiments sequentially.
# Usage: bash run_all.sh
# Logs: results/{exp_name}/run.log

set -euo pipefail

PYTHON=/opt/homebrew/anaconda3/envs/cytosafe/bin/python
SCRIPT=run_experiment.py
CONFIGS_DIR=configs

CONFIGS=(
    exp1_3T3_to_3T3_scaffold.yaml
    exp2_3T3_to_3T3_tanimoto.yaml
    exp3_3T3_to_HEK_scaffold.yaml
    exp4_3T3_to_HEK_tanimoto.yaml
    exp5_HEK_to_HEK_scaffold.yaml
    exp6_HEK_to_HEK_tanimoto.yaml
    exp7_HEK_to_3T3_scaffold.yaml
    exp8_HEK_to_3T3_tanimoto.yaml
)

for cfg in "${CONFIGS[@]}"; do
    exp_name="${cfg%.yaml}"
    log_dir="results/${exp_name}"
    mkdir -p "${log_dir}"
    echo "========================================"
    echo "Running: ${cfg}"
    echo "Log:     ${log_dir}/run.log"
    echo "========================================"
    $PYTHON $SCRIPT --config "${CONFIGS_DIR}/${cfg}" 2>&1 | tee "${log_dir}/run.log"
    echo "Done: ${exp_name}"
    echo ""
done

echo "All experiments complete."
