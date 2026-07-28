#!/usr/bin/env bash
# Single-LLM debias: run `x_one_llm` variant separately for qwen3-4B and
# qwen3-8B so each model is compared on its own (not combined).
#
# Output CSVs:
#   results/group_qwen/<ds>/<model>_x_one_llm_seed1.csv
#   results/individual_qwen/<ds>/<model>_x_one_llm_seed1.csv
set -xeuo pipefail

PROJ_ROOT=/home/jiangli/debias_response
cd "${PROJ_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODELS=(qwen3-4B qwen3-8B)
DATASETS=(EEDI OpinionQA Twin-2K-500)
LEVELS=(aggreated individual)
SEED=${SEED:-1}
VARIANT=x_one_llm
declare -A AGG_STEM=( ["EEDI"]="eedi" ["OpinionQA"]="opinionqa" ["Twin-2K-500"]="twin" )
declare -A AGG_DIR=( ["EEDI"]="aggreated" ["OpinionQA"]="aggreated" ["Twin-2K-500"]="aggreated" )

LOG_DIR="${PROJ_ROOT}/logs/qwen_debias"
mkdir -p "${LOG_DIR}"

for MODEL in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    for LEVEL in "${LEVELS[@]}"; do
      if [[ "${LEVEL}" == "aggreated" ]]; then
        TRAIN="dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_train.parquet"
        TEST="dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_test.parquet"
        SUB="group_qwen/${DS}"
        HIDDEN=512,256,128
        ALPHA=0.01
        LR=0.0005
        MAX_ITER=1500
        BSZ=64
        DROPOUT=0.0
      else
        TRAIN="dataset/${DS}/individual/individual_train.parquet"
        TEST="dataset/${DS}/individual/individual_test.parquet"
        SUB="individual_qwen/${DS}"
        HIDDEN=6144,3072,1536,768,384
        ALPHA=1e-4
        LR=0.0002
        MAX_ITER=3500
        BSZ=512
        DROPOUT=0.05
      fi
      mkdir -p "${PROJ_ROOT}/results/${SUB}"
      # debias_variants prepends `results/`, so pass relative to results/.
      REL="${SUB}/${MODEL}_${VARIANT}_seed${SEED}.csv"
      ABS="${PROJ_ROOT}/results/${REL}"
      LOG="${LOG_DIR}/${DS}_${LEVEL}_${MODEL}_one.log"
      if [[ -f "${ABS}" ]]; then
        echo "[skip] ${ABS}"
        continue
      fi
      echo "[debias-one] ${DS} ${LEVEL} ${MODEL} -> ${ABS}"
      python -m debias.debias_variants \
        --train_file "${TRAIN}" \
        --test_file  "${TEST}" \
        --variant    "${VARIANT}" \
        --model_type mlp \
        --device     cuda:0 \
        --random_state "${SEED}" \
        --llm_field  "${MODEL}_norm" \
        --llm_vector_transform raw \
        --result_file "${REL}" \
        --hidden_layers "${HIDDEN}" \
        --mlp_alpha "${ALPHA}" \
        --learning_rate_init "${LR}" \
        --max_iter "${MAX_ITER}" \
        --batch_size "${BSZ}" \
        --mlp_dropout "${DROPOUT}" \
        --validation_fraction 0.1 \
        --n_iter_no_change 60 \
        --min_delta 1e-6 \
        > "${LOG}" 2>&1

      # debias writes to results/${REL} and also results/{test,train}/${REL}
      # — relocate the nested-prefix artifact.
      if [[ -f "${PROJ_ROOT}/results/results/${REL}" ]]; then
        mv "${PROJ_ROOT}/results/results/${REL}" "${ABS}"
      fi
    done
  done
done

# Clean up the doubled-prefix tree if the debias script produced it.
rm -rf "${PROJ_ROOT}/results/results" 2>/dev/null || true

echo "=== run_debias_one.sh DONE ==="
