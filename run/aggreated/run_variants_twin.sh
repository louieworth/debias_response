#!/bin/bash

###############################################################################
# Unified grouped-variant runner for Twin.
#
# Writes one combined per-seed CSV under:
#   results/group/twin/aggreated/
#
# Default output files depend on LLM_VECTOR_TRANSFORM:
#   raw    -> results/group/twin/aggreated/aggreated_twin_256_multisource_result_seed_<seed>.csv
#   sorted -> results/group/twin/aggreated/aggreated_twin_256_multisource_sorted_result_seed_<seed>.csv
#
# Supported experiment families in the same file:
#   1. single-source rows over all configured llm_field values
#   2. sampled multi-source rows (Sample-k)
#   3. concatenated multi-source rows (Concat / Concat-d)
###############################################################################

# Feature variants
SINGLE_SOURCE_VARIANTS="${SINGLE_SOURCE_VARIANTS:-x_only x_avg_llm x_all_llm x_one_llm}"
MULTISOURCE_VARIANTS="${MULTISOURCE_VARIANTS:-x_avg_llm x_all_llm}"

# Model families
MODEL_TYPES="${MODEL_TYPES:-ridge rf mlp xgboost}"

# Random seeds
SEEDS="${SEEDS:-0 1 2 3 4}"

# Which experiment families to include
INCLUDE_SINGLE_SOURCE="${INCLUDE_SINGLE_SOURCE:-y}"
INCLUDE_SAMPLE="${INCLUDE_SAMPLE:-y}"
INCLUDE_CONCAT="${INCLUDE_CONCAT:-y}"
INCLUDE_ONE_LOGPROB="${INCLUDE_ONE_LOGPROB:-n}"
ONE_LOGPROB_ONLY="${ONE_LOGPROB_ONLY:-n}"
REFUSE_EXISTING_RESULTS="${REFUSE_EXISTING_RESULTS:-n}"

# Multi-source settings
SAMPLE_PER_LLM="${SAMPLE_PER_LLM:-10}"
LLM_VECTOR_TRANSFORM="${LLM_VECTOR_TRANSFORM:-raw}"
SINGLE_SOURCE_LLM_DIM="${SINGLE_SOURCE_LLM_DIM:-}"
SAMPLE_LLM_DIM="${SAMPLE_LLM_DIM:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_SCRIPT="$SCRIPT_DIR/ablation/run_multisource_ablation.sh"

DEFAULT_LLM_FIELDS="claude-3.5-haiku_norm,deepseek-v3_norm,gpt-3.5-turbo_norm,gpt-4o-mini_norm,gpt-4o_norm,gpt-5-mini_norm,llama-3.3-70B-instruct-turbo_norm,mistral-7B-instruct-v0.3_norm"

# DEFAULT_LLM_FIELDS="gpt-4o_norm"

if [[ -z "$LLM_VECTOR_TRANSFORM" || "$LLM_VECTOR_TRANSFORM" == "raw" ]]; then
    RESULT_SUFFIX=""
else
    RESULT_SUFFIX="_${LLM_VECTOR_TRANSFORM}"
fi

DEFAULT_RESULT_BASENAME="aggreated_twin_256_multisource${RESULT_SUFFIX}_result.csv"

if [[ "$ONE_LOGPROB_ONLY" =~ ^[Yy]$ ]]; then
    MODEL_TYPES="mlp"
    INCLUDE_SINGLE_SOURCE="n"
    INCLUDE_SAMPLE="n"
    INCLUDE_CONCAT="n"
    INCLUDE_ONE_LOGPROB="y"
    REFUSE_EXISTING_RESULTS="y"
    DEFAULT_RESULT_BASENAME="aggreated_twin_one_logprob_result.csv"
fi

DATASET=Twin \
TRAIN_FILE="${TRAIN_FILE:-dataset/Twin-2K-500/aggreated/twin_train.parquet}" \
TEST_FILE="${TEST_FILE:-dataset/Twin-2K-500/aggreated/twin_test.parquet}" \
RESULT_SUBDIR="${RESULT_SUBDIR:-group/twin/aggreated}" \
RESULT_BASENAME="${RESULT_BASENAME:-$DEFAULT_RESULT_BASENAME}" \
LLM_FIELDS_CSV="${LLM_FIELDS_CSV:-${LLM_FIELDS:-$DEFAULT_LLM_FIELDS}}" \
SINGLE_SOURCE_VARIANTS="$SINGLE_SOURCE_VARIANTS" \
MULTISOURCE_VARIANTS="$MULTISOURCE_VARIANTS" \
MODEL_TYPES="$MODEL_TYPES" \
SEEDS="$SEEDS" \
DEVICE="${DEVICE:-auto}" \
CLEAR_OLD_RESULTS="${CLEAR_OLD_RESULTS:-n}" \
REFUSE_EXISTING_RESULTS="$REFUSE_EXISTING_RESULTS" \
SAVE_SPLIT_RESULTS="${SAVE_SPLIT_RESULTS:-n}" \
DRY_RUN="${DRY_RUN:-n}" \
INCLUDE_SINGLE_SOURCE="$INCLUDE_SINGLE_SOURCE" \
INCLUDE_SAMPLE="$INCLUDE_SAMPLE" \
INCLUDE_CONCAT="$INCLUDE_CONCAT" \
INCLUDE_ONE_LOGPROB="$INCLUDE_ONE_LOGPROB" \
SAMPLE_PER_LLM="$SAMPLE_PER_LLM" \
LLM_VECTOR_TRANSFORM="$LLM_VECTOR_TRANSFORM" \
SINGLE_SOURCE_LLM_DIM="$SINGLE_SOURCE_LLM_DIM" \
SAMPLE_LLM_DIM="$SAMPLE_LLM_DIM" \
CONCAT_LLM_DIM="${CONCAT_LLM_DIM:-350}" \
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}" \
RF_N_ESTIMATORS="${RF_N_ESTIMATORS:-100}" \
RF_MAX_DEPTH="${RF_MAX_DEPTH:-10}" \
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-100}" \
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-6}" \
XGB_LEARNING_RATE="${XGB_LEARNING_RATE:-0.05}" \
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8}" \
XGB_COLSAMPLE_BYTREE="${XGB_COLSAMPLE_BYTREE:-0.8}" \
MLP_HIDDEN_LAYERS="${MLP_HIDDEN_LAYERS:-512,256,128}" \
MLP_ALPHA="${MLP_ALPHA:-0.01}" \
MLP_LR_INIT="${MLP_LR_INIT:-0.0005}" \
MLP_MAX_ITER="${MLP_MAX_ITER:-1500}" \
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-64}" \
MLP_DROPOUT="${MLP_DROPOUT:-0.0}" \
MLP_VALIDATION_FRACTION="${MLP_VALIDATION_FRACTION:-0.1}" \
MLP_N_ITER_NO_CHANGE="${MLP_N_ITER_NO_CHANGE:-20}" \
MLP_MIN_DELTA="${MLP_MIN_DELTA:-1e-6}" \
MLP_STANDARDIZE="${MLP_STANDARDIZE:-y}" \
bash "$HELPER_SCRIPT"
