#!/bin/bash
###############################################################################
# Individual-split MLP sweep over 3 datasets x 5 seeds x_* variants.
# Fixed profile: seed1_3of3_100pct_raw_6144
#   - dataset/{NAME}/individual/individual_{train,test}.parquet
#   - model_type=mlp, llm_vector_transform=raw
#   - MLP width 6144, alpha=1e-4, lr=2e-4
# Set ONE_LOGPROB_ONLY=y to run seeds 0--4 with --variant one_logprob
# in an isolated result namespace.
#
# Output (one CSV per dataset, columns include seed + variant):
#   results/individual/{NAME}/individual_{NAME}_result.csv
# Concurrency: datasets run SERIALLY; within each dataset, all SEEDS x VARIANTS
# (5*4 = 20 processes) launch in parallel, round-robin across GPUs. Each
# (seed, variant) writes to its own temp CSV to avoid append races, and all
# 20 temps are merged into the final per-dataset CSV at the end.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ONE_LOGPROB_ONLY="${ONE_LOGPROB_ONLY:-n}"
DATASETS="${DATASETS:-EEDI OpinionQA Twin-2K-500}"
if [[ "$ONE_LOGPROB_ONLY" =~ ^[Yy]$ ]]; then
    SEEDS="${SEEDS:-0 1 2 3 4}"
    VARIANTS="${VARIANTS:-one_logprob}"
    DEFAULT_RESULT_ROOT_REL="one_logprob_migration/individual"
    DEFAULT_LOG_DIR="logs/one_logprob_migration/individual"
    DEFAULT_SAVE_SPLIT_RESULTS="no"
    DEFAULT_RESULT_PRECISION="17"
else
    SEEDS="${SEEDS:-1 2 3 4 5}"
    VARIANTS="${VARIANTS:-x_only x_avg_llm x_all_llm x_one_llm}"
    DEFAULT_RESULT_ROOT_REL="individual"
    DEFAULT_LOG_DIR="logs"
    DEFAULT_SAVE_SPLIT_RESULTS="yes"
    DEFAULT_RESULT_PRECISION="4"
fi

# --- Fixed MLP profile (seed1_3of3_100pct_raw_6144/config.env) ---
LLM_VECTOR_TRANSFORM="raw"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-6144,3072,1536,768,384}"
MLP_ALPHA="${MLP_ALPHA:-1e-4}"
MLP_LR_INIT="${MLP_LR_INIT:-0.0002}"
MLP_MAX_ITER="${MLP_MAX_ITER:-3500}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-512}"
MLP_DROPOUT="${MLP_DROPOUT:-0.05}"
MLP_VALIDATION_FRACTION="${MLP_VALIDATION_FRACTION:-0.1}"
MLP_N_ITER_NO_CHANGE="${MLP_N_ITER_NO_CHANGE:-60}"
MLP_MIN_DELTA="${MLP_MIN_DELTA:-1e-6}"
MLP_STANDARDIZE="${MLP_STANDARDIZE:-y}"

GPU_DEVICES="${GPU_DEVICES:-cuda:0 cuda:1}"
read -r -a GPUS <<< "$GPU_DEVICES"
if [ "${#GPUS[@]}" -eq 0 ]; then
    GPUS=("cuda:0")
fi
GPU_COUNT=${#GPUS[@]}

RESULT_ROOT_REL="${RESULT_ROOT_REL:-$DEFAULT_RESULT_ROOT_REL}"
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"
SAVE_SPLIT_RESULTS="${SAVE_SPLIT_RESULTS:-$DEFAULT_SAVE_SPLIT_RESULTS}"
SAVE_UNIT_PREDICTIONS="${SAVE_UNIT_PREDICTIONS:-no}"
UNIT_PREDICTION_ROOT_REL="${UNIT_PREDICTION_ROOT_REL:-${RESULT_ROOT_REL}_unit_predictions}"
RESULT_PRECISION="${RESULT_PRECISION:-$DEFAULT_RESULT_PRECISION}"
AUTO_CLEAR_OLD_RESULTS="${AUTO_CLEAR_OLD_RESULTS:-no}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-yes}"

mkdir -p "$LOG_DIR"

TOTAL_VARIANTS=$(echo "$VARIANTS" | wc -w | tr -d ' ')
GLOBAL_FAILED=0
declare -a GLOBAL_SUMMARY=()

echo "========================================================================"
echo "Datasets: $DATASETS"
echo "Seeds:    $SEEDS"
echo "Variants: $VARIANTS"
echo "Model:    mlp (layers=$MLP_HIDDEN_LAYERS, alpha=$MLP_ALPHA, lr=$MLP_LR_INIT,"
echo "          dropout=$MLP_DROPOUT, standardize=$MLP_STANDARDIZE, transform=$LLM_VECTOR_TRANSFORM)"
echo "GPUs:     ${GPUS[*]} (round-robin)"
echo "Result root: results/$RESULT_ROOT_REL"
echo "Unit predictions: $SAVE_UNIT_PREDICTIONS (results/$UNIT_PREDICTION_ROOT_REL)"
echo "Skip existing: $SKIP_EXISTING_RESULTS | Auto-clear: $AUTO_CLEAR_OLD_RESULTS"
echo "========================================================================"

for DATASET_NAME in $DATASETS; do
    DATASET_DIR="dataset/${DATASET_NAME}/individual"
    TRAIN_FILE="${DATASET_DIR}/individual_train.parquet"
    TEST_FILE="${DATASET_DIR}/individual_test.parquet"

    echo ""
    echo "########################################################################"
    echo "# Dataset: $DATASET_NAME"
    echo "########################################################################"

    if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
        echo "[SKIP] missing parquet files under $DATASET_DIR"
        [ ! -f "$TRAIN_FILE" ] && echo "  missing: $TRAIN_FILE"
        [ ! -f "$TEST_FILE" ]  && echo "  missing: $TEST_FILE"
        GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
        continue
    fi

    if [[ " $VARIANTS " == *" one_logprob "* ]]; then
        python - "$TRAIN_FILE" "$TEST_FILE" <<'PY'
import sys
import numpy as np
import pandas as pd

required = {
    "one_logprobs",
    "one_logprobs_expected_response_norm",
    "one_logprobs_choice_count",
}
for path in sys.argv[1:]:
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"missing One Logprob fields in {path}: {missing}")
    widths = {len(value) for value in frame["one_logprobs"]}
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise SystemExit(f"inconsistent One Logprob vector width in {path}: {widths}")
    probabilities = np.asarray(frame["one_logprobs"].tolist(), dtype=float)
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    ):
        raise SystemExit(f"invalid One Logprob probabilities in {path}")
    print(f"validated One Logprob fields in {path}: {len(frame)} rows")
PY
    fi

    RESULT_DIR_REL="${RESULT_ROOT_REL}/${DATASET_NAME}"
    RESULT_DIR_ABS="results/${RESULT_DIR_REL}"
    mkdir -p "$RESULT_DIR_ABS"

    FINAL_FILE_REL="${RESULT_DIR_REL}/individual_${DATASET_NAME}_result.csv"
    FINAL_FILE_ABS="results/${FINAL_FILE_REL}"
    TMP_DIR_REL="${RESULT_DIR_REL}/_tmp"
    TMP_DIR_ABS="results/${TMP_DIR_REL}"
    mkdir -p "$TMP_DIR_ABS"

    echo "Train:   $TRAIN_FILE"
    echo "Test:    $TEST_FILE"
    echo "Final:   $FINAL_FILE_ABS"
    echo "Tmp:     $TMP_DIR_ABS/seed<SEED>_<variant>.csv"

    if [ "$SKIP_EXISTING_RESULTS" = "yes" ] && [ -f "$FINAL_FILE_ABS" ]; then
        ROWS=$(tail -n +2 "$FINAL_FILE_ABS" | wc -l | xargs)
        echo "[SKIP] $DATASET_NAME (exists: $FINAL_FILE_ABS, $ROWS rows)"
        continue
    fi
    if [ "$AUTO_CLEAR_OLD_RESULTS" = "yes" ] && [ -f "$FINAL_FILE_ABS" ]; then
        rm -f "$FINAL_FILE_ABS"
    fi

    PIDS=()
    PID_LABELS=()
    MLP_GPU_INDEX=0
    COMBO_IDX=0

    STD_ARGS=()
    case "${MLP_STANDARDIZE,,}" in
        n|no|0|false) STD_ARGS+=(--no_mlp_standardize) ;;
    esac
    RESULT_ARGS=()
    case "${SAVE_SPLIT_RESULTS,,}" in
        n|no|0|false) RESULT_ARGS+=(--no_split_results) ;;
    esac

    for SEED in $SEEDS; do
        for variant in $VARIANTS; do
            COMBO_IDX=$((COMBO_IDX + 1))
            ASSIGNED_GPU="${GPUS[$MLP_GPU_INDEX]}"
            MLP_GPU_INDEX=$(((MLP_GPU_INDEX + 1) % GPU_COUNT))

            TMP_FILE_REL="${TMP_DIR_REL}/seed${SEED}_${variant}.csv"
            TMP_FILE_ABS="results/${TMP_FILE_REL}"
            rm -f "$TMP_FILE_ABS"  # ensure clean single-row output
            LOG_FILE="${LOG_DIR}/${DATASET_NAME}_individual_mlp_${variant}_seed${SEED}.log"
            PREDICTION_ARGS=()
            if [[ "${SAVE_UNIT_PREDICTIONS,,}" =~ ^(y|yes|1|true)$ ]]; then
                UNIT_FILE_REL="${UNIT_PREDICTION_ROOT_REL}/${DATASET_NAME}/seed${SEED}_${variant}.csv"
                mkdir -p "results/$(dirname "$UNIT_FILE_REL")"
                PREDICTION_ARGS+=(--prediction_file "$UNIT_FILE_REL")
            fi

            echo "  [launch $COMBO_IDX] seed=$SEED $variant on $ASSIGNED_GPU -> $LOG_FILE"

            (
                echo "========================================"
                echo "Experiment: $DATASET_NAME | individual | mlp + $variant | seed=$SEED"
                echo "Started at: $(date)"
                echo "========================================"
                python -m debias.debias_variants \
                    --train_file "$TRAIN_FILE" \
                    --test_file  "$TEST_FILE" \
                    --variant    "$variant" \
                    --model_type mlp \
                    --device     "$ASSIGNED_GPU" \
                    --random_state "$SEED" \
                    --llm_vector_transform "$LLM_VECTOR_TRANSFORM" \
                    --result_file "$TMP_FILE_REL" \
                    "${PREDICTION_ARGS[@]}" \
                    --hidden_layers "$MLP_HIDDEN_LAYERS" \
                    --mlp_alpha "$MLP_ALPHA" \
                    --learning_rate_init "$MLP_LR_INIT" \
                    --max_iter "$MLP_MAX_ITER" \
                    --batch_size "$MLP_BATCH_SIZE" \
                    --mlp_dropout "$MLP_DROPOUT" \
                    --validation_fraction "$MLP_VALIDATION_FRACTION" \
                    --n_iter_no_change "$MLP_N_ITER_NO_CHANGE" \
                    --min_delta "$MLP_MIN_DELTA" \
                    --result_precision "$RESULT_PRECISION" \
                    "${RESULT_ARGS[@]}" \
                    "${STD_ARGS[@]}"
                rc=$?
                echo ""
                echo "========================================"
                if [ $rc -eq 0 ]; then
                    echo "[OK] $DATASET_NAME seed=$SEED $variant finished $(date)"
                else
                    echo "[FAIL] $DATASET_NAME seed=$SEED $variant (rc=$rc) $(date)"
                    exit 1
                fi
                echo "========================================"
            ) > "$LOG_FILE" 2>&1 &

            PIDS+=($!)
            PID_LABELS+=("${DATASET_NAME}/seed${SEED}/${variant}")
        done
    done

    echo ""
    echo "  ${#PIDS[@]} jobs in flight for $DATASET_NAME, waiting..."
    DATASET_FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  [OK] ${PID_LABELS[$i]}"
        else
            echo "  [FAIL] ${PID_LABELS[$i]}"
            DATASET_FAILED=$((DATASET_FAILED + 1))
            GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
        fi
    done

    if [ "$DATASET_FAILED" -gt 0 ]; then
        echo "  $DATASET_FAILED job(s) failed for $DATASET_NAME; skipping merge."
        continue
    fi

    # Merge the 20 per-(seed,variant) temp CSVs into one dataset CSV with a
    # prepended `seed` column. debias_variants already writes `variant` as
    # a column, so the final schema is: seed, variant, <all metric columns>.
    python - "$TMP_DIR_ABS" "$FINAL_FILE_ABS" "$SEEDS" "$VARIANTS" <<'PY'
import sys, os, pandas as pd
tmp_dir, final_file, seeds_str, variants_str = sys.argv[1:5]
seeds = seeds_str.split()
variants = variants_str.split()
frames = []
for seed in seeds:
    for variant in variants:
        p = os.path.join(tmp_dir, f"seed{seed}_{variant}.csv")
        if not os.path.isfile(p):
            raise SystemExit(f"missing temp file: {p}")
        df = pd.read_csv(p)
        df.insert(0, "seed", int(seed))
        frames.append(df)
out = pd.concat(frames, ignore_index=True)
os.makedirs(os.path.dirname(final_file), exist_ok=True)
out.to_csv(final_file, index=False)
print(f"wrote {final_file} ({len(out)} rows, {out.shape[1]} cols)")
PY

    if [ -f "$FINAL_FILE_ABS" ]; then
        ROWS=$(tail -n +2 "$FINAL_FILE_ABS" | wc -l | xargs)
        echo "  -> $FINAL_FILE_ABS ($ROWS rows)"
        # Clean up temp files now that the merge succeeded.
        rm -rf "$TMP_DIR_ABS"
    else
        echo "  -> MERGE FAILED: $FINAL_FILE_ABS"
        GLOBAL_FAILED=$((GLOBAL_FAILED + 1))
    fi
done

echo ""
echo "========================================================================"
echo "GLOBAL SUMMARY"
echo "========================================================================"
for DATASET_NAME in $DATASETS; do
    FINAL_FILE_ABS="results/${RESULT_ROOT_REL}/${DATASET_NAME}/individual_${DATASET_NAME}_result.csv"
    if [ -f "$FINAL_FILE_ABS" ]; then
        ROWS=$(tail -n +2 "$FINAL_FILE_ABS" | wc -l | xargs)
        echo "  [OK] $DATASET_NAME  $FINAL_FILE_ABS ($ROWS rows)"
    else
        echo "  [MISS] $DATASET_NAME  $FINAL_FILE_ABS"
    fi
done
if [ "$GLOBAL_FAILED" -gt 0 ]; then
    echo "WARNING: $GLOBAL_FAILED failure(s)"
    exit 1
fi
echo "All runs completed successfully."
