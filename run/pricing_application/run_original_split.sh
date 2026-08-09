#!/bin/bash
# Reproduce the paper's original Twin-2K-500 record split and materialize
# unit-level predictions for the pricing-application summary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export TMPDIR="/data2/tmp"
export DATASETS="Twin-2K-500"
export SEEDS="0 1 2 3 4"
export VARIANTS="x_only x_one_llm x_avg_llm x_all_llm"
export GPU_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

export RESULT_ROOT_REL="pricing_application/original_split/model_metrics"
export UNIT_PREDICTION_ROOT_REL="pricing_application/original_split/unit_predictions"
export LOG_DIR="logs/pricing_application/original_split"
export SAVE_SPLIT_RESULTS="no"
export SAVE_UNIT_PREDICTIONS="yes"
export RESULT_PRECISION="17"
export SKIP_EXISTING_RESULTS="yes"
export AUTO_CLEAR_OLD_RESULTS="no"

# Final individual-level profile used by the paper.
export MLP_HIDDEN_LAYERS="6144,3072,1536,768,384"
export MLP_ALPHA="1e-6"
export MLP_LR_INIT="0.0002"
export MLP_MAX_ITER="3500"
export MLP_BATCH_SIZE="512"
export MLP_DROPOUT="0.08"
export MLP_VALIDATION_FRACTION="0"
export MLP_N_ITER_NO_CHANGE="30"
export MLP_MIN_DELTA="1e-6"
export MLP_STANDARDIZE="y"

exec bash "$PROJECT_ROOT/run/individual/run_individual.sh"
