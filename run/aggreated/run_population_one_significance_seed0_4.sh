#!/bin/bash
# Reproduce the population-level paper comparison on model seeds 0--4.
# One Logprob, Mean, and Vector are tested against the matched-seed One runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export TMPDIR="/data2/tmp"

RESULT_ROOT_REL="population_one_significance_seed0_4_precision17"
LOG_ROOT="logs/population_one_significance_seed0_4_precision17"
SEEDS=(0 1 2 3 4)
DATASETS=(Twin-2K-500 OpinionQA EEDI)
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "$LOG_ROOT"

declare -a PIDS=()
declare -a TASKS=()
task_index=0

for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        Twin-2K-500) runner="$SCRIPT_DIR/run_variants_twin.sh" ;;
        OpinionQA) runner="$SCRIPT_DIR/run_variants_opinionqa.sh" ;;
        EEDI) runner="$SCRIPT_DIR/run_variants_eedi.sh" ;;
        *) echo "Unsupported dataset: $dataset"; exit 1 ;;
    esac

    for seed in "${SEEDS[@]}"; do
        gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
        log_file="$LOG_ROOT/${dataset}_seed${seed}.log"
        echo "[launch] dataset=$dataset seed=$seed device=cuda:$gpu"
        (
            RESULT_SUBDIR="$RESULT_ROOT_REL/$dataset" \
            RESULT_BASENAME="population_${dataset}.csv" \
            LLM_FIELDS_CSV="gpt-4o_norm" \
            SINGLE_SOURCE_VARIANTS="x_only x_one_llm x_avg_llm x_all_llm" \
            MULTISOURCE_VARIANTS="x_avg_llm x_all_llm" \
            MODEL_TYPES="mlp" \
            SEEDS="$seed" \
            DEVICE="cuda:$gpu" \
            CLEAR_OLD_RESULTS="n" \
            REFUSE_EXISTING_RESULTS="y" \
            SAVE_SPLIT_RESULTS="n" \
            INCLUDE_SINGLE_SOURCE="y" \
            INCLUDE_SAMPLE="n" \
            INCLUDE_CONCAT="n" \
            INCLUDE_ONE_LOGPROB="y" \
            SINGLE_SOURCE_LLM_DIM="50" \
            LLM_VECTOR_TRANSFORM="raw" \
            RESULT_PRECISION="17" \
            ONE_LOGPROB_RESULT_PRECISION="17" \
            MLP_HIDDEN_LAYERS="512,256,128" \
            MLP_ALPHA="0.01" \
            MLP_LR_INIT="0.0005" \
            MLP_MAX_ITER="1500" \
            MLP_BATCH_SIZE="64" \
            MLP_DROPOUT="0.0" \
            MLP_VALIDATION_FRACTION="0.1" \
            MLP_N_ITER_NO_CHANGE="20" \
            MLP_MIN_DELTA="1e-6" \
            MLP_STANDARDIZE="y" \
            bash "$runner"
        ) >"$log_file" 2>&1 &
        PIDS+=("$!")
        TASKS+=("$dataset/seed$seed")
        task_index=$((task_index + 1))
    done
done

failed=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${TASKS[$i]}"
    else
        echo "[FAIL] ${TASKS[$i]} (see $LOG_ROOT)"
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "$failed population tasks failed."
    exit 1
fi

echo "All ${#PIDS[@]} population dataset/seed tasks completed."
