#!/bin/bash
# Train the paper's One and Vector estimators on the Twin complete-question split.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_FILE="dataset/Twin-2K-500/individual/syn_digits_question_holdout/individual_train.parquet"
TEST_FILE="dataset/Twin-2K-500/individual/syn_digits_question_holdout/individual_test.parquet"
RESULT_ROOT_REL="syn_digits/question_holdout/raw"
PREDICTION_ROOT_REL="syn_digits/question_holdout/unit_predictions"
LOG_DIR="logs/syn_digits/question_holdout"
SEEDS="${SEEDS:-0 1 2 3 4}"
VARIANTS="${VARIANTS:-x_one_llm x_all_llm}"
GPU_DEVICES="${GPU_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7}"
FORCE="${FORCE:-no}"

mkdir -p "results/$RESULT_ROOT_REL" "results/$PREDICTION_ROOT_REL" "$LOG_DIR"
read -r -a GPUS <<< "$GPU_DEVICES"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_DEVICES must contain at least one device" >&2
    exit 2
fi

pids=()
labels=()
job_index=0
for seed in $SEEDS; do
    for variant in $VARIANTS; do
        result_rel="$RESULT_ROOT_REL/seed${seed}_${variant}.csv"
        prediction_rel="$PREDICTION_ROOT_REL/seed${seed}_${variant}.csv"
        result_abs="results/$result_rel"
        prediction_abs="results/$prediction_rel"
        log_file="$LOG_DIR/seed${seed}_${variant}.log"
        if [ "$FORCE" != "yes" ] && [ -f "$result_abs" ] && [ -f "$prediction_abs" ]; then
            echo "[SKIP] seed=$seed variant=$variant"
            continue
        fi
        device="${GPUS[$((job_index % ${#GPUS[@]}))]}"
        job_index=$((job_index + 1))
        echo "[LAUNCH] seed=$seed variant=$variant device=$device"
        (
            python -m debias.debias_variants \
                --train_file "$TRAIN_FILE" \
                --test_file "$TEST_FILE" \
                --variant "$variant" \
                --model_type mlp \
                --device "$device" \
                --random_state "$seed" \
                --llm_vector_transform raw \
                --result_file "$result_rel" \
                --prediction_file "$prediction_rel" \
                --hidden_layers "6144,3072,1536,768,384" \
                --mlp_alpha 1e-6 \
                --learning_rate_init 0.0002 \
                --max_iter 3500 \
                --batch_size 512 \
                --mlp_dropout 0.08 \
                --validation_fraction 0 \
                --n_iter_no_change 30 \
                --min_delta 1e-6 \
                --result_precision 17 \
                --no_split_results
        ) > "$log_file" 2>&1 &
        pids+=("$!")
        labels+=("seed${seed}/${variant}")
    done
done

failures=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[OK] ${labels[$index]}"
    else
        echo "[FAIL] ${labels[$index]}" >&2
        failures=$((failures + 1))
    fi
done
if [ "$failures" -gt 0 ]; then
    echo "$failures job(s) failed" >&2
    exit 1
fi
echo "All question-holdout training jobs completed."
