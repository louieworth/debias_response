#!/bin/bash

###############################################################################
# Unified Multi-Source Runner for Aggregated Datasets
#
# This script writes a single combined per-seed CSV under:
#   results/group/<dataset>/
#
# It supports four experiment families in one file:
#   1. single-source rows over each llm_field
#   2. sampled multi-source rows (Twin-style Sample-k)
#   3. concatenated multi-source rows (Twin-style Concat)
#   4. the materialized One Logprob probability vector
#
# These files remain separate from the current paper tables because they use a
# distinct `*_multisource_result_seed_<k>.csv` naming scheme.
###############################################################################

# Dataset options: Twin, EEDI, OpinionQA
DATASET="${DATASET:-EEDI}"

# Feature variants
SINGLE_SOURCE_VARIANTS="${SINGLE_SOURCE_VARIANTS:-x_only x_avg_llm x_all_llm x_one_llm}"
MULTISOURCE_VARIANTS="${MULTISOURCE_VARIANTS:-x_avg_llm x_all_llm}"

# Model families
MODEL_TYPES="${MODEL_TYPES:-ridge rf mlp xgboost}"

# Random seeds
SEEDS="${SEEDS:-0 1 2 3 4}"

# Execution controls
DEVICE="${DEVICE:-auto}"
CLEAR_OLD_RESULTS="${CLEAR_OLD_RESULTS:-n}"
REFUSE_EXISTING_RESULTS="${REFUSE_EXISTING_RESULTS:-n}"
SAVE_SPLIT_RESULTS="${SAVE_SPLIT_RESULTS:-n}"
DRY_RUN="${DRY_RUN:-n}"

# Which experiment families to include
INCLUDE_SINGLE_SOURCE="${INCLUDE_SINGLE_SOURCE:-y}"
INCLUDE_SAMPLE="${INCLUDE_SAMPLE:-y}"
INCLUDE_CONCAT="${INCLUDE_CONCAT:-y}"
INCLUDE_ONE_LOGPROB="${INCLUDE_ONE_LOGPROB:-n}"
ONE_LOGPROB_RESULT_PRECISION="${ONE_LOGPROB_RESULT_PRECISION:-17}"
RESULT_PRECISION="${RESULT_PRECISION:-}"

# Multi-source settings
SAMPLE_PER_LLM="${SAMPLE_PER_LLM:-10}"
LLM_VECTOR_TRANSFORM="${LLM_VECTOR_TRANSFORM:-raw}"
SINGLE_SOURCE_LLM_DIM="${SINGLE_SOURCE_LLM_DIM:-}"
SAMPLE_LLM_DIM="${SAMPLE_LLM_DIM:-}"
CONCAT_LLM_DIM="${CONCAT_LLM_DIM:-}"

# Model hyperparameters
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
RF_N_ESTIMATORS="${RF_N_ESTIMATORS:-100}"
RF_MAX_DEPTH="${RF_MAX_DEPTH:-10}"
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-100}"
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-6}"
XGB_LEARNING_RATE="${XGB_LEARNING_RATE:-0.05}"
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8}"
XGB_COLSAMPLE_BYTREE="${XGB_COLSAMPLE_BYTREE:-0.8}"
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-2048,1024,512,256}"
MLP_ALPHA="${MLP_ALPHA:-0.00001}"
MLP_LR_INIT="${MLP_LR_INIT:-0.0005}"
MLP_MAX_ITER="${MLP_MAX_ITER:-3000}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-512}"
MLP_DROPOUT="${MLP_DROPOUT:-0.0}"
MLP_VALIDATION_FRACTION="${MLP_VALIDATION_FRACTION:-0.1}"
MLP_N_ITER_NO_CHANGE="${MLP_N_ITER_NO_CHANGE:-30}"
MLP_MIN_DELTA="${MLP_MIN_DELTA:-1e-6}"
MLP_STANDARDIZE="${MLP_STANDARDIZE:-y}"

# Optional overrides. If empty, dataset defaults are used.
TRAIN_FILE="${TRAIN_FILE:-}"
TEST_FILE="${TEST_FILE:-}"
RESULT_SUBDIR="${RESULT_SUBDIR:-}"
RESULT_BASENAME="${RESULT_BASENAME:-}"
LLM_FIELDS_CSV="${LLM_FIELDS_CSV:-}"

###############################################################################
# Script content below
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

dataset_defaults() {
    case "$DATASET" in
        Twin)
            : "${TRAIN_FILE:=dataset/Twin-2K-500/aggreated/twin_train.parquet}"
            : "${TEST_FILE:=dataset/Twin-2K-500/aggreated/twin_test.parquet}"
            : "${RESULT_SUBDIR:=group/twin/aggreated}"
            : "${RESULT_BASENAME:=aggreated_twin_256_multisource_result.csv}"
            : "${LLM_FIELDS_CSV:=claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o-mini_norm,gpt-4o_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm}"
            ;;
        EEDI)
            : "${TRAIN_FILE:=dataset/EEDI/aggreated/eedi_train.parquet}"
            : "${TEST_FILE:=dataset/EEDI/aggreated/eedi_test.parquet}"
            : "${RESULT_SUBDIR:=group/EEDI}"
            : "${RESULT_BASENAME:=aggreated_eedi_multisource_result.csv}"
            : "${LLM_FIELDS_CSV:=claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o_norm,gpt-4o-mini_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm}"
            ;;
        OpinionQA)
            : "${TRAIN_FILE:=dataset/OpinionQA/aggreated/opinionqa_train.parquet}"
            : "${TEST_FILE:=dataset/OpinionQA/aggreated/opinionqa_test.parquet}"
            : "${RESULT_SUBDIR:=group/OpinionQA}"
            : "${RESULT_BASENAME:=aggreated_opinionqa_multisource_result.csv}"
            : "${LLM_FIELDS_CSV:=claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o_norm,gpt-4o-mini_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm}"
            ;;
        *)
            echo "[ERROR] Unsupported DATASET=$DATASET. Use one of: Twin, EEDI, OpinionQA."
            exit 1
            ;;
    esac
}

dataset_defaults

IFS=',' read -r -a RAW_LLM_FIELDS <<< "$LLM_FIELDS_CSV"
LLM_FIELDS=()
for raw_field in "${RAW_LLM_FIELDS[@]}"; do
    field="$(echo "$raw_field" | xargs)"
    if [[ -n "$field" ]]; then
        LLM_FIELDS+=("$field")
    fi
done

if [[ ${#LLM_FIELDS[@]} -eq 0 ]]; then
    echo "[ERROR] No LLM fields configured. Set LLM_FIELDS_CSV."
    exit 1
fi

JOINED_LLM_FIELDS="$(IFS=,; echo "${LLM_FIELDS[*]}")"
SAMPLE_INPUT_NAME="Sample-$(( ${#LLM_FIELDS[@]} * SAMPLE_PER_LLM ))"
if [[ -n "$CONCAT_LLM_DIM" ]]; then
    CONCAT_INPUT_NAME="Concat-${CONCAT_LLM_DIM}"
else
    CONCAT_INPUT_NAME="Concat"
fi

if [[ "$RESULT_BASENAME" == *.* ]]; then
    RESULT_STEM="${RESULT_BASENAME%.*}"
    RESULT_EXT="${RESULT_BASENAME##*.}"
else
    RESULT_STEM="$RESULT_BASENAME"
    RESULT_EXT="csv"
fi

seed_result_relative_path() {
    local seed="$1"
    echo "${RESULT_SUBDIR}/${RESULT_STEM}_seed_${seed}.${RESULT_EXT}"
}

seed_result_file() {
    local seed="$1"
    echo "results/$(seed_result_relative_path "$seed")"
}

mkdir -p "results/${RESULT_SUBDIR}"

if [[ "$DEVICE" == "auto" ]]; then
    CUDA_AVAILABLE=$(python - <<'PY'
try:
    import torch
    print("1" if torch.cuda.is_available() else "0")
except Exception:
    print("0")
PY
)
    if [[ "$CUDA_AVAILABLE" == "1" ]]; then
        RESOLVED_DEVICE="cuda"
    else
        RESOLVED_DEVICE="cpu"
    fi
else
    RESOLVED_DEVICE="$DEVICE"
fi

python - <<PY
import pandas as pd
import numpy as np
import sys

for path in ["$TRAIN_FILE", "$TEST_FILE"]:
    try:
        frame = pd.read_parquet(path)
        rows = len(frame)
    except Exception as exc:
        print(f"[ERROR] Failed to read parquet: {path} -> {exc}")
        sys.exit(1)
    if rows == 0:
        print(f"[ERROR] Empty parquet: {path}")
        sys.exit(1)
    if "$INCLUDE_ONE_LOGPROB".lower() in {"y", "yes"}:
        required = {
            "one_logprobs",
            "one_logprobs_expected_response_norm",
            "one_logprobs_choice_count",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            print(f"[ERROR] Missing One Logprob fields in {path}: {missing}")
            sys.exit(1)
        widths = {len(value) for value in frame["one_logprobs"]}
        if len(widths) != 1 or next(iter(widths)) <= 0:
            print(f"[ERROR] Inconsistent One Logprob vector width in {path}: {widths}")
            sys.exit(1)
        probabilities = np.asarray(frame["one_logprobs"].tolist(), dtype=float)
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        ):
            print(f"[ERROR] Invalid One Logprob probabilities in {path}")
            sys.exit(1)
    print(f"Validated {path}: {rows} rows")
PY

if [[ "$CLEAR_OLD_RESULTS" =~ ^[Yy]$ ]]; then
    echo "Clearing old combined ablation files:"
    for seed in $SEEDS; do
        result_file="$(seed_result_file "$seed")"
        echo "  $result_file"
        rm -f "$result_file"
    done
elif [[ "$REFUSE_EXISTING_RESULTS" =~ ^[Yy]$ && ! "$DRY_RUN" =~ ^[Yy]$ ]]; then
    for seed in $SEEDS; do
        result_file="$(seed_result_file "$seed")"
        if [[ -f "$result_file" ]]; then
            echo "[ERROR] Refusing to append to existing result: $result_file"
            echo "Set CLEAR_OLD_RESULTS=y to explicitly replace isolated results."
            exit 1
        fi
    done
fi

echo "========================================================================"
echo "Unified Multi-Source Grouped Runner"
echo "========================================================================"
echo "Dataset: $DATASET"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "Result subdir: results/${RESULT_SUBDIR}"
echo "Result basename: $RESULT_BASENAME"
echo "Resolved device: $RESOLVED_DEVICE"
echo "Single-source variants: $SINGLE_SOURCE_VARIANTS"
echo "Multi-source variants: $MULTISOURCE_VARIANTS"
echo "Model types: $MODEL_TYPES"
echo "Seeds: $SEEDS"
echo "LLM fields (${#LLM_FIELDS[@]}): $JOINED_LLM_FIELDS"
echo "LLM vector transform: $LLM_VECTOR_TRANSFORM"
echo "Include single-source: $INCLUDE_SINGLE_SOURCE"
echo "Include sample: $INCLUDE_SAMPLE"
echo "Include concat: $INCLUDE_CONCAT"
echo "Include One Logprob: $INCLUDE_ONE_LOGPROB"
echo "Sample per LLM: $SAMPLE_PER_LLM"
if [[ -n "$SINGLE_SOURCE_LLM_DIM" ]]; then
    echo "Single-source llm_dim override: $SINGLE_SOURCE_LLM_DIM"
fi
if [[ -n "$SAMPLE_LLM_DIM" ]]; then
    echo "Sample llm_dim override: $SAMPLE_LLM_DIM"
fi
if [[ -n "$CONCAT_LLM_DIM" ]]; then
    echo "Concat llm_dim override: $CONCAT_LLM_DIM"
fi
echo "Dry run: $DRY_RUN"
echo "========================================================================"

run_experiment() {
    local seed="$1"
    local variant="$2"
    local model_type="$3"
    local result_relative_path="$4"
    shift 4
    local -a extra_args=("$@")

    local vector_transform="$LLM_VECTOR_TRANSFORM"
    if [[ "$variant" == "one_logprob" ]]; then
        vector_transform="raw"
    fi

    local -a cmd=(
        python -m debias.debias_variants
        --train_file "$TRAIN_FILE"
        --test_file "$TEST_FILE"
        --variant "$variant"
        --model_type "$model_type"
        --device "$RESOLVED_DEVICE"
        --random_state "$seed"
        --result_file "$result_relative_path"
        --llm_vector_transform "$vector_transform"
    )
    if [[ ! "$SAVE_SPLIT_RESULTS" =~ ^[Yy]$ ]]; then
        cmd+=(--no_split_results)
    fi
    if [[ -n "$RESULT_PRECISION" ]]; then
        cmd+=(--result_precision "$RESULT_PRECISION")
    elif [[ "$variant" == "one_logprob" ]]; then
        cmd+=(--result_precision "$ONE_LOGPROB_RESULT_PRECISION")
    fi
    case "$model_type" in
        ridge)
            cmd+=(--alpha "$RIDGE_ALPHA")
            ;;
        rf)
            cmd+=(
                --n_estimators "$RF_N_ESTIMATORS"
                --max_depth "$RF_MAX_DEPTH"
            )
            ;;
        xgboost)
            cmd+=(
                --n_estimators "$XGB_N_ESTIMATORS"
                --max_depth "$XGB_MAX_DEPTH"
                --learning_rate "$XGB_LEARNING_RATE"
                --subsample "$XGB_SUBSAMPLE"
                --colsample_bytree "$XGB_COLSAMPLE_BYTREE"
            )
            ;;
        mlp)
            cmd+=(
                --hidden_layers "$MLP_HIDDEN_LAYERS"
                --mlp_alpha "$MLP_ALPHA"
                --learning_rate_init "$MLP_LR_INIT"
                --max_iter "$MLP_MAX_ITER"
                --batch_size "$MLP_BATCH_SIZE"
                --mlp_dropout "$MLP_DROPOUT"
                --validation_fraction "$MLP_VALIDATION_FRACTION"
                --n_iter_no_change "$MLP_N_ITER_NO_CHANGE"
                --min_delta "$MLP_MIN_DELTA"
            )
            if [[ ! "$MLP_STANDARDIZE" =~ ^[Yy]$ ]]; then
                cmd+=(--no_mlp_standardize)
            fi
            ;;
    esac
    cmd+=("${extra_args[@]}")

    echo "    Command: ${cmd[*]}"
    if [[ "$DRY_RUN" =~ ^[Yy]$ ]]; then
        return 0
    fi
    "${cmd[@]}"
}

TOTAL_EXPERIMENTS=0
for seed in $SEEDS; do
    if [[ "$INCLUDE_SINGLE_SOURCE" =~ ^[Yy]$ ]]; then
        for model_type in $MODEL_TYPES; do
            for variant in $SINGLE_SOURCE_VARIANTS; do
                if [[ "$variant" == "x_only" ]]; then
                    ((TOTAL_EXPERIMENTS++))
                else
                    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + ${#LLM_FIELDS[@]}))
                fi
            done
        done
    fi
    if [[ "$INCLUDE_SAMPLE" =~ ^[Yy]$ ]]; then
        for model_type in $MODEL_TYPES; do
            for variant in $MULTISOURCE_VARIANTS; do
                ((TOTAL_EXPERIMENTS++))
            done
        done
    fi
    if [[ "$INCLUDE_CONCAT" =~ ^[Yy]$ ]]; then
        for model_type in $MODEL_TYPES; do
            for variant in $MULTISOURCE_VARIANTS; do
                ((TOTAL_EXPERIMENTS++))
            done
        done
    fi
    if [[ "$INCLUDE_ONE_LOGPROB" =~ ^[Yy]$ ]]; then
        for model_type in $MODEL_TYPES; do
            ((TOTAL_EXPERIMENTS++))
        done
    fi
done

CURRENT_EXPERIMENT=0

for seed in $SEEDS; do
    result_relative_path="$(seed_result_relative_path "$seed")"
    echo "========================================================================"
    echo "Seed: $seed"
    echo "Result file: results/$result_relative_path"
    echo "========================================================================"

    if [[ "$INCLUDE_SINGLE_SOURCE" =~ ^[Yy]$ ]]; then
        echo "  Single-source runs"
        for model_type in $MODEL_TYPES; do
            for variant in $SINGLE_SOURCE_VARIANTS; do
                if [[ "$variant" == "x_only" ]]; then
                    ((CURRENT_EXPERIMENT++))
                    echo "    [$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] $model_type + $variant (shared no-LLM baseline)"
                    extra_single_args=(
                        --llm_fields "$JOINED_LLM_FIELDS"
                        --llm_input_mode shared_no_llm
                        --llm_input_name x_only_shared
                    )
                    run_experiment \
                        "$seed" \
                        "$variant" \
                        "$model_type" \
                        "$result_relative_path" \
                        "${extra_single_args[@]}"
                    continue
                fi

                for llm_field in "${LLM_FIELDS[@]}"; do
                    echo "  - llm_field=$llm_field"
                    ((CURRENT_EXPERIMENT++))
                    echo "    [$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] $model_type + $variant"
                    extra_single_args=(
                        --llm_field "$llm_field"
                        --llm_input_mode single_source
                        --llm_input_name "$llm_field"
                    )
                    if [[ -n "$SINGLE_SOURCE_LLM_DIM" && ( "$variant" == "x_all_llm" || "$variant" == "x_avg_llm" ) ]]; then
                        extra_single_args+=(--llm_dim "$SINGLE_SOURCE_LLM_DIM")
                    fi
                    run_experiment \
                        "$seed" \
                        "$variant" \
                        "$model_type" \
                        "$result_relative_path" \
                        "${extra_single_args[@]}"
                done
            done
        done
    fi

    if [[ "$INCLUDE_SAMPLE" =~ ^[Yy]$ ]]; then
        echo "  Sampled multi-source runs"
        for model_type in $MODEL_TYPES; do
            for variant in $MULTISOURCE_VARIANTS; do
                ((CURRENT_EXPERIMENT++))
                echo "    [$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] $model_type + $variant (${SAMPLE_INPUT_NAME})"
                extra_sample_args=(
                    --llm_fields "$JOINED_LLM_FIELDS"
                    --llm_sampling_strategy sample_per_field
                    --sample_per_llm "$SAMPLE_PER_LLM"
                    --llm_input_mode sample_multisource
                    --llm_input_name "$SAMPLE_INPUT_NAME"
                )
                if [[ -n "$SAMPLE_LLM_DIM" && ( "$variant" == "x_all_llm" || "$variant" == "x_avg_llm" ) ]]; then
                    extra_sample_args+=(--llm_dim "$SAMPLE_LLM_DIM")
                fi
                run_experiment \
                    "$seed" \
                    "$variant" \
                    "$model_type" \
                    "$result_relative_path" \
                    "${extra_sample_args[@]}"
            done
        done
    fi

    if [[ "$INCLUDE_CONCAT" =~ ^[Yy]$ ]]; then
        echo "  Concatenated multi-source runs"
        for model_type in $MODEL_TYPES; do
            for variant in $MULTISOURCE_VARIANTS; do
                ((CURRENT_EXPERIMENT++))
                echo "    [$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] $model_type + $variant (${CONCAT_INPUT_NAME})"
                extra_concat_args=(
                    --llm_fields "$JOINED_LLM_FIELDS"
                    --llm_sampling_strategy all
                    --llm_input_mode concat_multisource
                    --llm_input_name "$CONCAT_INPUT_NAME"
                )
                if [[ -n "$CONCAT_LLM_DIM" && ( "$variant" == "x_all_llm" || "$variant" == "x_avg_llm" ) ]]; then
                    extra_concat_args+=(--llm_dim "$CONCAT_LLM_DIM")
                fi
                run_experiment \
                    "$seed" \
                    "$variant" \
                    "$model_type" \
                    "$result_relative_path" \
                    "${extra_concat_args[@]}"
            done
        done
    fi

    if [[ "$INCLUDE_ONE_LOGPROB" =~ ^[Yy]$ ]]; then
        echo "  One Logprob runs"
        for model_type in $MODEL_TYPES; do
            ((CURRENT_EXPERIMENT++))
            echo "    [$CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS] $model_type + one_logprob"
            extra_one_logprob_args=(
                --llm_input_mode one_logprob
                --llm_input_name one_logprob
            )
            run_experiment \
                "$seed" \
                "one_logprob" \
                "$model_type" \
                "$result_relative_path" \
                "${extra_one_logprob_args[@]}"
        done
    fi
done

echo "========================================================================"
echo "Completed."
echo "========================================================================"
for seed in $SEEDS; do
    result_file="$(seed_result_file "$seed")"
    if [[ -f "$result_file" ]]; then
        count=$(tail -n +2 "$result_file" | wc -l | xargs)
        echo "  seed=$seed -> $result_file ($count rows)"
    else
        echo "  seed=$seed -> $result_file (not created)"
    fi
done
