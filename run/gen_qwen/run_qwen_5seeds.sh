#!/usr/bin/env bash
# Extend Qwen3-4B and Qwen3-8B x_one_llm debias to 5 seeds.
# Saves CSVs at results/{group_qwen,individual_qwen}/<ds>/<model>_x_one_llm_seed<S>.csv.
# Also sweeps gpt-4o over the same 5 seeds for symmetric aggregation-level comparison.
# Runs up to MAX_PAR debias jobs in parallel on GPU3 (each uses ~2GB GPU).
set -xeuo pipefail

PROJ_ROOT=/home/jiangli/debias_response
cd "${PROJ_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODELS=("qwen3-4B" "qwen3-8B" "gpt-4o")
DATASETS=(EEDI OpinionQA Twin-2K-500)
LEVELS=(aggreated individual)
SEEDS=(2 3 4 5)   # seed 1 already exists for all these
VARIANT=x_one_llm
MAX_PAR=${MAX_PAR:-4}
declare -A AGG_STEM=( ["EEDI"]="eedi" ["OpinionQA"]="opinionqa" ["Twin-2K-500"]="twin" )
declare -A AGG_DIR=( ["EEDI"]="aggreated" ["OpinionQA"]="aggreated" ["Twin-2K-500"]="aggreated" )

LOG_DIR="${PROJ_ROOT}/logs/qwen_debias"
mkdir -p "${LOG_DIR}"

launch_one() {
  local DS=$1 LEVEL=$2 MODEL=$3 SEED=$4
  local FIELD="${MODEL}_norm"
  if [[ "${LEVEL}" == "aggreated" ]]; then
    TRAIN="dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_train.parquet"
    TEST="dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_test.parquet"
    SUB="group_qwen/${DS}"
    HIDDEN=512,256,128; ALPHA=0.01; LR=0.0005; MAX_ITER=1500; BSZ=64; DROPOUT=0.0
  else
    TRAIN="dataset/${DS}/individual/individual_train.parquet"
    TEST="dataset/${DS}/individual/individual_test.parquet"
    SUB="individual_qwen/${DS}"
    HIDDEN=6144,3072,1536,768,384; ALPHA=1e-4; LR=0.0002; MAX_ITER=3500; BSZ=512; DROPOUT=0.05
  fi
  mkdir -p "${PROJ_ROOT}/results/${SUB}"
  local REL="${SUB}/${MODEL}_${VARIANT}_seed${SEED}.csv"
  local ABS="${PROJ_ROOT}/results/${REL}"
  local LOG="${LOG_DIR}/${DS}_${LEVEL}_${MODEL}_seed${SEED}.log"
  if [[ -f "${ABS}" ]]; then
    echo "[skip] ${ABS}"
    return 0
  fi
  echo "[launch] ${DS} ${LEVEL} ${MODEL} seed=${SEED}"
  (
    python -m debias.debias_variants \
      --train_file "${TRAIN}" --test_file "${TEST}" \
      --variant "${VARIANT}" --model_type mlp \
      --device cuda:0 --random_state "${SEED}" \
      --llm_field "${FIELD}" --llm_vector_transform raw \
      --result_file "${REL}" \
      --hidden_layers "${HIDDEN}" --mlp_alpha "${ALPHA}" \
      --learning_rate_init "${LR}" --max_iter "${MAX_ITER}" \
      --batch_size "${BSZ}" --mlp_dropout "${DROPOUT}" \
      --validation_fraction 0.1 --n_iter_no_change 60 --min_delta 1e-6 \
      > "${LOG}" 2>&1
    # Move nested-prefix output if present.
    if [[ -f "${PROJ_ROOT}/results/results/${REL}" ]]; then
      mkdir -p "$(dirname "${ABS}")"
      mv "${PROJ_ROOT}/results/results/${REL}" "${ABS}"
    fi
  ) &
}

active_count() { jobs -rp | wc -l | tr -d ' '; }

for DS in "${DATASETS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        while [[ $(active_count) -ge ${MAX_PAR} ]]; do
          sleep 5
        done
        launch_one "${DS}" "${LEVEL}" "${MODEL}" "${SEED}"
      done
    done
  done
done
wait
rm -rf "${PROJ_ROOT}/results/results" 2>/dev/null || true
echo "=== run_qwen_5seeds.sh DONE ==="
