#!/bin/bash
# Reproduce Appendix population source tables at fixed K=50 on seeds 0--4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export TMPDIR="/data2/tmp"

RESULT_ROOT_REL="population_source_significance_k50_seed0_4_precision17"
LOG_ROOT="logs/population_source_significance_k50_seed0_4_precision17"
DATASETS=(Twin-2K-500 OpinionQA EEDI)
SEEDS=(0 1 2 3 4)
GPUS=(0 1 2 3 4 5 6 7)
LLM_FIELDS="claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o-mini_norm,gpt-4o_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm"

mkdir -p "$LOG_ROOT"

run_dataset_seed() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    local dataset_key
    case "$dataset" in
        Twin-2K-500) dataset_key="Twin" ;;
        OpinionQA) dataset_key="OpinionQA" ;;
        EEDI) dataset_key="EEDI" ;;
        *) echo "Unsupported dataset: $dataset"; return 1 ;;
    esac

    local log_file="$LOG_ROOT/${dataset}_seed${seed}.log"
    echo "[launch] dataset=$dataset seed=$seed device=cuda:$gpu"
    DATASET="$dataset_key" \
    RESULT_SUBDIR="$RESULT_ROOT_REL/$dataset" \
    RESULT_BASENAME="population_source_${dataset}.csv" \
    LLM_FIELDS_CSV="$LLM_FIELDS" \
    SINGLE_SOURCE_VARIANTS="x_only x_one_llm x_avg_llm x_all_llm" \
    MODEL_TYPES="mlp" \
    SEEDS="$seed" \
    DEVICE="cuda:$gpu" \
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
    MLP_MIN_DELTA="1e-6" \
    MLP_STANDARDIZE="y" \
    bash "$SCRIPT_DIR/ablation/run_multisource_ablation.sh" \
        >"$log_file" 2>&1
    echo "[OK] dataset=$dataset seed=$seed device=cuda:$gpu"
}

declare -a PIDS=()
for gpu in "${GPUS[@]}"; do
    (
        task_index="$gpu"
        while (( task_index < 15 )); do
            dataset_index=$((task_index / 5))
            seed_index=$((task_index % 5))
            run_dataset_seed \
                "${DATASETS[$dataset_index]}" \
                "${SEEDS[$seed_index]}" \
                "$gpu"
            task_index=$((task_index + 8))
        done
    ) &
    PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "$failed GPU workers failed; inspect $LOG_ROOT."
    exit 1
fi

echo "All 15 dataset/seed groups completed (375 fits)."
