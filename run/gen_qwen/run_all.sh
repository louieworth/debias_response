#!/usr/bin/env bash
# End-to-end orchestrator for open-source LLM sample-source study.
#
# Phase 1  build gen prompts (6: 3 datasets × 2 levels)
# Phase 2  gen + inject for each of 12 configs (2 models × 3 datasets × 2 levels)
# Phase 3  run debias per (dataset, level) with qwen3-4B_norm + qwen3-8B_norm
#          as llm_fields, write per-config CSV under results/{group,individual}_qwen/
# Phase 4  build comparison CSV vs existing closed-source baseline
#
# Runs sequentially on a single GPU. Resume-friendly: skips steps whose
# output already exists.
#
# Env var overrides:
#   CUDA_VISIBLE_DEVICES default 3
#   MODELS               default "qwen3-4B qwen3-8B"
#   DATASETS             default "EEDI OpinionQA Twin-2K-500"
#   LEVELS               default "aggreated individual"
#   SEED                 default 1
#   VARIANT              default x_all_llm
set -xeuo pipefail

PROJ_ROOT=/home/jiangli/debias_response
cd "${PROJ_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export VERL_GEN_CONCURRENCY=${VERL_GEN_CONCURRENCY:-256}
# Ray OOM-killer is aggressive on this node (other tenants use a lot of RAM);
# raise the threshold to 0.99 so we don't get killed at 95%.
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}

MODELS=(${MODELS:-qwen3-4B qwen3-8B})
DATASETS=(${DATASETS:-EEDI OpinionQA Twin-2K-500})
LEVELS=(${LEVELS:-aggreated individual})
SEED=${SEED:-1}
VARIANT=${VARIANT:-x_all_llm}

LOG_GEN="${PROJ_ROOT}/logs/qwen_gen"
LOG_DEB="${PROJ_ROOT}/logs/qwen_debias"
mkdir -p "${LOG_GEN}" "${LOG_DEB}"

############ Phase 1: prompts ############
for DS in "${DATASETS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    P="${PROJ_ROOT}/dataset/gen_prompts/${DS}_${LEVEL}/prompts.parquet"
    if [[ -f "${P}" ]]; then
      echo "[skip-prompts] ${P}"
      continue
    fi
    python3 "${PROJ_ROOT}/run/gen_qwen/build_gen_prompts.py" \
      --dataset "${DS}" --level "${LEVEL}" --k_personas 50
  done
done

############ Phase 2: gen + inject ############
# MODEL_TAG -> HF model id (official Qwen3 chat models)
declare -A HF_ID=( ["qwen3-4B"]="Qwen/Qwen3-4B" ["qwen3-8B"]="Qwen/Qwen3-8B" )

for MODEL in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    for LEVEL in "${LEVELS[@]}"; do
      OUT_DIR="${PROJ_ROOT}/gen_results/qwen/${DS}_${LEVEL}"
      OUT="${OUT_DIR}/${MODEL}.parquet"
      SENTINEL_INJ="${OUT_DIR}/.injected_${MODEL}"
      LOG="${LOG_GEN}/${DS}_${LEVEL}_${MODEL}.log"

      if [[ -f "${OUT}" ]]; then
        echo "[skip-gen] ${OUT}"
      else
        echo "[gen ] ${DS} ${LEVEL} ${MODEL} -> ${OUT}"
        DATASET="${DS}" LEVEL="${LEVEL}" MODEL_TAG="${MODEL}" MODEL_PATH="${HF_ID[${MODEL}]}" \
          bash "${PROJ_ROOT}/run/gen_qwen/run_gen.sh" > "${LOG}" 2>&1 || {
            echo "[gen FAIL] ${DS} ${LEVEL} ${MODEL}, see ${LOG}"; exit 1; }
        # Tear down any lingering Ray processes before the next config so
        # we don't accumulate ~1GB of worker state per gen.
        ray stop --force >/dev/null 2>&1 || true
        pkill -9 -f "main_generation_server" 2>/dev/null || true
        pkill -9 -f "vLLMHttpServer" 2>/dev/null || true
        pkill -9 -f "ray::" 2>/dev/null || true
        sleep 2
      fi

      if [[ -f "${SENTINEL_INJ}" ]]; then
        echo "[skip-inject] ${SENTINEL_INJ}"
      else
        echo "[inject] ${DS} ${LEVEL} ${MODEL}"
        python3 "${PROJ_ROOT}/run/gen_qwen/inject_llm_column.py" \
          --dataset "${DS}" --level "${LEVEL}" \
          --model_tag "${MODEL}" --gen_parquet "${OUT}"
        touch "${SENTINEL_INJ}"
      fi
    done
  done
done

############ Phase 3: debias (Qwen3 only, x_all_llm, one seed) ############
LLM_FIELDS_QWEN=qwen3-4B_norm,qwen3-8B_norm

# Stem per dataset for aggregated parquet names.
declare -A AGG_STEM=( ["EEDI"]="eedi" ["OpinionQA"]="opinionqa" ["Twin-2K-500"]="twin" )
declare -A AGG_DIR=( ["EEDI"]="aggreated" ["OpinionQA"]="aggreated" ["Twin-2K-500"]="aggreated" )

for DS in "${DATASETS[@]}"; do
  for LEVEL in "${LEVELS[@]}"; do
    if [[ "${LEVEL}" == "aggreated" ]]; then
      TRAIN="${PROJ_ROOT}/dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_train.parquet"
      TEST="${PROJ_ROOT}/dataset/${DS}/${AGG_DIR[${DS}]}/${AGG_STEM[${DS}]}_test.parquet"
      RES_DIR="${PROJ_ROOT}/results/group_qwen/${DS}"
      # Aggregated MLP profile (smaller MLP, per run_variants_*.sh defaults).
      HIDDEN=512,256,128
      ALPHA=0.01
      LR=0.0005
      MAX_ITER=1500
      BSZ=64
      DROPOUT=0.0
    else
      TRAIN="${PROJ_ROOT}/dataset/${DS}/individual/individual_train.parquet"
      TEST="${PROJ_ROOT}/dataset/${DS}/individual/individual_test.parquet"
      RES_DIR="${PROJ_ROOT}/results/individual_qwen/${DS}"
      # Individual MLP profile (wider, per run_individual.sh defaults).
      HIDDEN=6144,3072,1536,768,384
      ALPHA=1e-4
      LR=0.0002
      MAX_ITER=3500
      BSZ=512
      DROPOUT=0.05
    fi
    mkdir -p "${RES_DIR}"
    RES_CSV="${RES_DIR}/qwen_${VARIANT}_seed${SEED}.csv"
    LOG="${LOG_DEB}/${DS}_${LEVEL}.log"

    if [[ -f "${RES_CSV}" ]]; then
      echo "[skip-debias] ${RES_CSV}"
      continue
    fi

    echo "[debias] ${DS} ${LEVEL} -> ${RES_CSV}"
    python -m debias.debias_variants \
      --train_file "${TRAIN}" \
      --test_file  "${TEST}" \
      --variant    "${VARIANT}" \
      --model_type mlp \
      --device     cuda:0 \
      --random_state "${SEED}" \
      --llm_fields "${LLM_FIELDS_QWEN}" \
      --llm_vector_transform raw \
      --result_file "results/${RES_DIR#${PROJ_ROOT}/results/}/qwen_${VARIANT}_seed${SEED}.csv" \
      --hidden_layers "${HIDDEN}" \
      --mlp_alpha "${ALPHA}" \
      --learning_rate_init "${LR}" \
      --max_iter "${MAX_ITER}" \
      --batch_size "${BSZ}" \
      --mlp_dropout "${DROPOUT}" \
      --validation_fraction 0.1 \
      --n_iter_no_change 60 \
      --min_delta 1e-6 \
      > "${LOG}" 2>&1 || {
        echo "[debias FAIL] ${DS} ${LEVEL}, see ${LOG}"; exit 1; }
  done
done

############ Phase 4: comparison summary ############
python3 "${PROJ_ROOT}/run/gen_qwen/summarize.py" || true
echo "=== run_all.sh DONE ==="
