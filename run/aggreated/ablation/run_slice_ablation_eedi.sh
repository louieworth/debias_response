#!/bin/bash
# Slice ablation on EEDI population-level: 18 slices (8 persona-row + 10 draw-col), 5 seeds, mlp head.
# Row slices: input dim = 10 (one persona's 10 stochastic decodes)
# Col slices: input dim = 8 (all 8 personas' single draw)
# Total: 18 * 5 = 90 runs.

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

mkdir -p "$RESULT_DIR"

run_one () {
    local SLICE="$1"
    local LLM_FIELD="$2"
    local LLM_DIM="$3"
    local SEED="$4"
    local OUT_REL="group/EEDI/slice_ablation/aggreated_eedi_${SLICE}_result_seed_${SEED}.csv"
    echo "[slice=$SLICE seed=$SEED dim=$LLM_DIM] -> $OUT_REL"
    "$PY" -m debias.debias_variants \
        --train_file "$TRAIN_FILE" \
        --test_file "$TEST_FILE" \
        --variant x_all_llm \
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

# 8 row slices: 1 persona × 10 draws, dim=10
for p in 1 2 3 4 5 6 7 8; do
    for SEED in $SEEDS; do
        run_one "row_p${p}" "gpt-4o_row_p${p}_norm" 10 "$SEED"
    done
done

# 10 column slices: 8 personas × 1 draw, dim=8
for d in 0 1 2 3 4 5 6 7 8 9; do
    for SEED in $SEEDS; do
        run_one "col_d${d}" "gpt-4o_col_d${d}_norm" 8 "$SEED"
    done
done

echo ""
echo "Done. CSV count:"
ls "$RESULT_DIR" | grep -c '\.csv$'
