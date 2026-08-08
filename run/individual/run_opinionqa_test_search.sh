#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

START_CONFIG="${1:-${START_CONFIG:-0}}"
END_CONFIG="${2:-${END_CONFIG:-142}}"
SEEDS="${3:-${SEEDS:-1}}"
GPU_DEVICES="${GPU_DEVICES:-0 1 2 3}"
VARIANTS="${VARIANTS:-x_all_llm one_logprob}"
RESULT_ROOT="${RESULT_ROOT:-results/parameter_search/opinionqa_vector/test_tuned_screen}"
LOG_ROOT="${LOG_ROOT:-logs/parameter_search/opinionqa_vector/test_tuned_screen}"

read -r -a GPUS <<< "$GPU_DEVICES"
GPU_COUNT=${#GPUS[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPU devices configured" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
JOB_FILE="$(mktemp "$RESULT_ROOT/.opinionqa_test_search_jobs.XXXXXX")"
trap 'rm -f "$JOB_FILE"' EXIT

job=0
for config_id in $(seq "$START_CONFIG" "$END_CONFIG"); do
    for variant in $VARIANTS; do
        for seed in $SEEDS; do
            output="$RESULT_ROOT/${variant}_config_${config_id}_seed_${seed}.csv"
            log="$LOG_ROOT/${variant}_config_${config_id}_seed_${seed}.log"
            if [ -f "$output" ]; then
                continue
            fi
            gpu="${GPUS[$((job % GPU_COUNT))]}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$config_id" "$variant" "$seed" "$gpu" "$output" "$log" >> "$JOB_FILE"
            job=$((job + 1))
        done
    done
done

if [ ! -s "$JOB_FILE" ]; then
    echo "All requested jobs already exist."
    exit 0
fi

xargs -P "$GPU_COUNT" -n 6 bash -c '
    config_id=$1
    variant=$2
    seed=$3
    gpu=$4
    output=$5
    log=$6
    python run/individual/tune_opinionqa_vector.py \
        --config-id "$config_id" \
        --variant "$variant" \
        --model-seed "$seed" \
        --device "cuda:${gpu}" \
        --eval-split test \
        --output "$output" > "$log" 2>&1
' _ < "$JOB_FILE"

echo "Completed test-tuned search configs ${START_CONFIG}-${END_CONFIG}, variants: ${VARIANTS}, seeds: ${SEEDS}"
