#!/bin/bash
# Reproduce the population K ablation at K=8,25,100,200 on seeds 0--4.
# K=1 and K=50 are reused from the current main population experiment by the
# companion summarizer rather than retrained here. K>50 uses a fixed
# replacement-bootstrap extension prepared from each dataset's 50 stored
# gpt-4o draws; the canonical benchmark files remain capped at K=50.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export TMPDIR="/data2/tmp"

RESULT_ROOT_REL="population_k_ablation_k1_8_25_50_100_200_seed0_4_precision17"
LOG_ROOT="logs/$RESULT_ROOT_REL"
BOOTSTRAP_INPUT_ROOT="results/$RESULT_ROOT_REL/bootstrap_inputs"
DATASETS=(Twin-2K-500 OpinionQA EEDI)
K_VALUES=(8 25 100 200)
SEEDS=(0 1 2 3 4)
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "$LOG_ROOT"
python "$SCRIPT_DIR/prepare_population_k_bootstrap_inputs.py"

dataset_files() {
    local dataset="$1"
    local k="$2"
    INPUT_MODE="population_k_ablation_native"
    INPUT_NAME="gpt-4o_K${k}"
    case "$dataset" in
        Twin-2K-500)
            if (( k > 50 )); then
                TRAIN_FILE="$BOOTSTRAP_INPUT_ROOT/Twin-2K-500_train_gpt4o_bootstrap_k200.parquet"
                TEST_FILE="$BOOTSTRAP_INPUT_ROOT/Twin-2K-500_test_gpt4o_bootstrap_k200.parquet"
                INPUT_MODE="population_k_ablation_bootstrap50"
                INPUT_NAME="gpt-4o_K${k}_bootstrap50"
            else
                TRAIN_FILE="dataset/Twin-2K-500/aggreated/twin_train.parquet"
                TEST_FILE="dataset/Twin-2K-500/aggreated/twin_test.parquet"
            fi
            ;;
        OpinionQA)
            if (( k > 50 )); then
                TRAIN_FILE="$BOOTSTRAP_INPUT_ROOT/OpinionQA_train_gpt4o_bootstrap_k200.parquet"
                TEST_FILE="$BOOTSTRAP_INPUT_ROOT/OpinionQA_test_gpt4o_bootstrap_k200.parquet"
                INPUT_MODE="population_k_ablation_bootstrap50"
                INPUT_NAME="gpt-4o_K${k}_bootstrap50"
            else
                TRAIN_FILE="dataset/OpinionQA/aggreated/opinionqa_train.parquet"
                TEST_FILE="dataset/OpinionQA/aggreated/opinionqa_test.parquet"
            fi
            ;;
        EEDI)
            if (( k > 50 )); then
                TRAIN_FILE="$BOOTSTRAP_INPUT_ROOT/EEDI_train_gpt4o_bootstrap_k200.parquet"
                TEST_FILE="$BOOTSTRAP_INPUT_ROOT/EEDI_test_gpt4o_bootstrap_k200.parquet"
                INPUT_MODE="population_k_ablation_bootstrap50"
                INPUT_NAME="gpt-4o_K${k}_bootstrap50"
            else
                TRAIN_FILE="dataset/EEDI/aggreated/eedi_train.parquet"
                TEST_FILE="dataset/EEDI/aggreated/eedi_test.parquet"
            fi
            ;;
        *)
            echo "Unsupported dataset: $1" >&2
            return 1
            ;;
    esac
}

result_is_complete() {
    local path="$1"
    local expected_k="$2"
    local expected_mode="$3"
    local expected_name="$4"
    [[ -s "$path" ]] || return 1
    python - "$path" "$expected_k" "$expected_mode" "$expected_name" <<'PY'
import sys
import pandas as pd

path, expected_k, expected_mode, expected_name = (
    sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
)
frame = pd.read_csv(path)
ok = (
    len(frame) == 1
    and frame.iloc[0]["variant"] == "x_all_llm"
    and frame.iloc[0]["model_type"] == "mlp"
    and frame.iloc[0]["llm_field"] == "gpt-4o_norm"
    and frame.iloc[0]["llm_input_mode"] == expected_mode
    and frame.iloc[0]["llm_input_name"] == expected_name
    and int(frame.iloc[0]["llm_responses_length"]) == expected_k
)
raise SystemExit(0 if ok else 1)
PY
}

run_task() {
    local dataset="$1"
    local k="$2"
    local seed="$3"
    local gpu="$4"
    local result_dir="results/$RESULT_ROOT_REL/$dataset"
    local result_path="$result_dir/population_k${k}_${dataset}_seed_${seed}.csv"
    local result_rel="$RESULT_ROOT_REL/$dataset/population_k${k}_${dataset}_seed_${seed}.csv"
    local log_file="$LOG_ROOT/${dataset}_k${k}_seed${seed}.log"

    dataset_files "$dataset" "$k"
    mkdir -p "$result_dir"
    if result_is_complete "$result_path" "$k" "$INPUT_MODE" "$INPUT_NAME"; then
        echo "[SKIP] dataset=$dataset K=$k seed=$seed"
        return 0
    fi
    if [[ -e "$result_path" ]]; then
        echo "Refusing to overwrite incomplete result: $result_path" >&2
        return 1
    fi

    echo "[RUN] dataset=$dataset K=$k seed=$seed device=cuda:$gpu"
    python -m debias.debias_variants \
        --train_file "$TRAIN_FILE" \
        --test_file "$TEST_FILE" \
        --variant x_all_llm \
        --model_type mlp \
        --llm_field gpt-4o_norm \
        --llm_input_mode "$INPUT_MODE" \
        --llm_input_name "$INPUT_NAME" \
        --llm_dim "$k" \
        --random_state "$seed" \
        --result_file "$result_rel" \
        --result_precision 17 \
        --no_split_results \
        --device "cuda:$gpu" \
        --llm_vector_transform raw \
        --hidden_layers 512,256,128 \
        --mlp_alpha 0.01 \
        --learning_rate_init 0.0005 \
        --max_iter 1500 \
        --batch_size 64 \
        --mlp_dropout 0.0 \
        --validation_fraction 0.1 \
        --n_iter_no_change 20 \
        --min_delta 0.000001 \
        >"$log_file" 2>&1

    result_is_complete "$result_path" "$k" "$INPUT_MODE" "$INPUT_NAME"
    echo "[OK] dataset=$dataset K=$k seed=$seed device=cuda:$gpu"
}

declare -a PIDS=()
for gpu in "${GPUS[@]}"; do
    (
        task_index="$gpu"
        while (( task_index < 60 )); do
            dataset_index=$((task_index / 20))
            within_dataset=$((task_index % 20))
            k_index=$((within_dataset / 5))
            seed_index=$((within_dataset % 5))
            run_task \
                "${DATASETS[$dataset_index]}" \
                "${K_VALUES[$k_index]}" \
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
    echo "$failed GPU workers failed; inspect $LOG_ROOT." >&2
    exit 1
fi

echo "All 60 K-ablation fits completed."
