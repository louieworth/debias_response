#!/bin/bash
# Slice ablation with Mean (x_avg_llm) and One (x_one_llm) heads on EEDI.
# 18 slices (8 row + 10 col) × 5 seeds × 2 heads = 180 runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_ROOT"

TRAIN_FILE="${TRAIN_FILE:-dataset/EEDI/aggreated/eedi_train.parquet}"
TEST_FILE="${TEST_FILE:-dataset/EEDI/aggreated/eedi_test.parquet}"
RESULT_DIR="${RESULT_DIR:-results/group/EEDI/slice_ablation}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cpu}"
PY="${PY:-python}"
VARIANTS="${VARIANTS:-x_avg_llm x_one_llm}"

mkdir -p "$RESULT_DIR"

run_one () {
    local VARIANT="$1"
    local SLICE="$2"
    local LLM_FIELD="$3"
    local LLM_DIM="$4"
    local SEED="$5"
    # Variant suffix: vector → keep filename as-is, avg → _mean, one → _one
    local VSUFFIX
    case "$VARIANT" in
        x_all_llm) VSUFFIX="vector" ;;
        x_avg_llm) VSUFFIX="mean" ;;
        x_one_llm) VSUFFIX="one" ;;
        *) VSUFFIX="$VARIANT" ;;
    esac
    local OUT_REL="group/EEDI/slice_ablation/aggreated_eedi_${SLICE}_${VSUFFIX}_result_seed_${SEED}.csv"
    echo "[variant=$VARIANT slice=$SLICE seed=$SEED dim=$LLM_DIM] -> $OUT_REL"
    "$PY" -m debias.debias_variants \
        --train_file "$TRAIN_FILE" \
        --test_file "$TEST_FILE" \
        --variant "$VARIANT" \
        --model_type mlp \
        --llm_field "$LLM_FIELD" \
        --llm_dim "$LLM_DIM" \
        --llm_vector_transform raw \
        --llm_input_mode single_source \
        --llm_input_name "slice_${SLICE}" \
        --random_state "$SEED" \
        --result_file "$OUT_REL" \
        --no_split_results \
        --device "$DEVICE"
}

for VARIANT in $VARIANTS; do
    for p in 1 2 3 4 5 6 7 8; do
        for SEED in $SEEDS; do
            run_one "$VARIANT" "row_p${p}" "gpt-4o_row_p${p}_norm" 10 "$SEED"
        done
    done
    for d in 0 1 2 3 4 5 6 7 8 9; do
        for SEED in $SEEDS; do
            run_one "$VARIANT" "col_d${d}" "gpt-4o_col_d${d}_norm" 8 "$SEED"
        done
    done
done

echo ""
echo "Done. CSV count:"
ls "$RESULT_DIR" | grep -c '\.csv$'
