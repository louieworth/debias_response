#!/bin/bash
# Re-run the OpinionQA population source-robustness comparison with every
# source materialized and consumed at the same K=50 response budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export TMPDIR="/data2/tmp"

RESULT_ROOT_REL="population_source_significance_k50fix_seed0_4_precision17"
LOG_ROOT="logs/$RESULT_ROOT_REL"
SEEDS=(0 1 2 3 4)
LLM_FIELDS="claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o-mini_norm,gpt-4o_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm"

mkdir -p "$LOG_ROOT"

declare -a PIDS=()
for seed in "${SEEDS[@]}"; do
    log_file="$LOG_ROOT/OpinionQA_seed${seed}.log"
    echo "[launch] OpinionQA sources seed=$seed device=cuda:$seed"
    (
        RESULT_SUBDIR="$RESULT_ROOT_REL/OpinionQA" \
        RESULT_BASENAME="population_source_OpinionQA.csv" \
        LLM_FIELDS_CSV="$LLM_FIELDS" \
        SINGLE_SOURCE_VARIANTS="x_only x_one_llm x_avg_llm x_all_llm" \
        MULTISOURCE_VARIANTS="x_avg_llm x_all_llm" \
        MODEL_TYPES="mlp" \
        SEEDS="$seed" \
        DEVICE="cuda:$seed" \
        CLEAR_OLD_RESULTS="n" \
        REFUSE_EXISTING_RESULTS="y" \
        SAVE_SPLIT_RESULTS="n" \
        INCLUDE_SINGLE_SOURCE="y" \
        INCLUDE_SAMPLE="n" \
        INCLUDE_CONCAT="n" \
        INCLUDE_ONE_LOGPROB="n" \
        SINGLE_SOURCE_LLM_DIM="50" \
        LLM_VECTOR_TRANSFORM="raw" \
        RESULT_PRECISION="17" \
        MLP_HIDDEN_LAYERS="512,256,128" \
        MLP_ALPHA="0.01" \
        MLP_LR_INIT="0.0005" \
        MLP_MAX_ITER="1500" \
        MLP_BATCH_SIZE="64" \
        MLP_DROPOUT="0.0" \
        MLP_VALIDATION_FRACTION="0.1" \
        MLP_N_ITER_NO_CHANGE="20" \
        MLP_MIN_DELTA="0.000001" \
        MLP_STANDARDIZE="y" \
        bash "$SCRIPT_DIR/run_variants_opinionqa.sh"
    ) >"$log_file" 2>&1 &
    PIDS+=("$!")
done

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[OK] OpinionQA/sources/seed${SEEDS[$index]}"
    else
        echo "[FAIL] OpinionQA/sources/seed${SEEDS[$index]} (see $LOG_ROOT)"
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "$failed OpinionQA source tasks failed."
    exit 1
fi

echo "All ${#PIDS[@]} corrected OpinionQA source tasks completed."
