#!/bin/bash
# Re-run the population comparison while retaining per-question predictions.
# Existing paper-result files are not modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_REL="population_question_predictions_seed0_4_precision17"
LOG_ROOT="logs/$OUTPUT_REL"
SEEDS=(0 1 2 3 4)
DATASETS=(Twin-2K-500 OpinionQA EEDI)
GPUS=(0 1 2 3 4 5 6 7)
VARIANTS=(x_only x_one_llm one_logprob x_avg_llm x_all_llm)

mkdir -p "$LOG_ROOT" "results/$OUTPUT_REL"

dataset_paths() {
    case "$1" in
        Twin-2K-500)
            TRAIN_FILE="dataset/Twin-2K-500/aggreated/twin_train.parquet"
            TEST_FILE="dataset/Twin-2K-500/aggreated/twin_test.parquet"
            ;;
        OpinionQA)
            TRAIN_FILE="dataset/OpinionQA/aggreated/opinionqa_train.parquet"
            TEST_FILE="dataset/OpinionQA/aggreated/opinionqa_test.parquet"
            ;;
        EEDI)
            TRAIN_FILE="dataset/EEDI/aggreated/eedi_train.parquet"
            TEST_FILE="dataset/EEDI/aggreated/eedi_test.parquet"
            ;;
        *)
            echo "Unsupported dataset: $1" >&2
            return 1
            ;;
    esac
}

run_dataset_seed() {
    local dataset="$1"
    local seed="$2"
    local gpu="$3"
    dataset_paths "$dataset"

    local output_dir="results/$OUTPUT_REL/$dataset"
    mkdir -p "$output_dir"

    for variant in "${VARIANTS[@]}"; do
        local prediction_rel="$OUTPUT_REL/$dataset/${variant}_seed${seed}_predictions.csv"
        local result_rel="$OUTPUT_REL/$dataset/${variant}_seed${seed}_result.csv"
        local prediction_path="results/$prediction_rel"
        local result_path="results/$result_rel"

        if [[ -f "$prediction_path" && -f "$result_path" ]]; then
            echo "[reuse] dataset=$dataset seed=$seed variant=$variant"
            continue
        fi
        if [[ -e "$prediction_path" || -e "$result_path" ]]; then
            echo "[error] partial output exists for $dataset/seed$seed/$variant" >&2
            return 1
        fi

        local -a cmd=(
            python -m debias.debias_variants
            --train_file "$TRAIN_FILE"
            --test_file "$TEST_FILE"
            --variant "$variant"
            --model_type mlp
            --device "cuda:$gpu"
            --random_state "$seed"
            --result_file "$result_rel"
            --prediction_file "$prediction_rel"
            --llm_vector_transform raw
            --no_split_results
            --result_precision 17
            --hidden_layers "512,256,128"
            --mlp_alpha 0.01
            --learning_rate_init 0.0005
            --max_iter 1500
            --batch_size 64
            --mlp_dropout 0.0
            --validation_fraction 0.1
            --n_iter_no_change 20
            --min_delta 1e-6
        )

        case "$variant" in
            x_only)
                cmd+=(
                    --llm_field gpt-4o_norm
                    --llm_input_mode shared_no_llm
                    --llm_input_name x_only_shared
                )
                ;;
            x_one_llm)
                cmd+=(
                    --llm_field gpt-4o_norm
                    --llm_input_mode single_source
                    --llm_input_name gpt-4o_norm
                )
                ;;
            x_avg_llm|x_all_llm)
                cmd+=(
                    --llm_field gpt-4o_norm
                    --llm_input_mode single_source
                    --llm_input_name gpt-4o_norm
                    --llm_dim 50
                )
                ;;
            one_logprob)
                cmd+=(
                    --llm_input_mode one_logprob
                    --llm_input_name one_logprob
                )
                ;;
        esac

        echo "[run] dataset=$dataset seed=$seed variant=$variant gpu=$gpu"
        "${cmd[@]}"
    done
}

declare -a pids=()
declare -a tasks=()
task_index=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
        log_file="$LOG_ROOT/${dataset}_seed${seed}.log"
        run_dataset_seed "$dataset" "$seed" "$gpu" >"$log_file" 2>&1 &
        pids+=("$!")
        tasks+=("$dataset/seed$seed")
        task_index=$((task_index + 1))
    done
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[OK] ${tasks[$i]}"
    else
        echo "[FAIL] ${tasks[$i]} (see $LOG_ROOT)" >&2
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "$failed prediction tasks failed." >&2
    exit 1
fi

echo "All question-level prediction tasks completed."
