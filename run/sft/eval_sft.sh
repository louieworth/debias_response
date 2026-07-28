#!/usr/bin/env bash
# For each trained config: merge FSDP -> HF, run vLLM generation on test set,
# run main_eval with custom hard-accuracy scorer, then aggregate to sft.json.
#
# Env var overrides:
#   CKPT_ROOT      default /data/data/jiangli/debias
#   NGPUS          default 4    (GPUs for generation server)
#   GEN_TP         default 1    (tensor-parallel size)
#   ONLY_CFG       optional, run one config only
set -xeuo pipefail

CKPT_ROOT=${CKPT_ROOT:-/data/data/jiangli/debias}
PROJ_ROOT=/home/jiangli/debias_response
GEN_DIR="${PROJ_ROOT}/gen_results"
RES_DIR="${PROJ_ROOT}/results"
mkdir -p "${GEN_DIR}" "${RES_DIR}"
NGPUS=${NGPUS:-1}
GEN_TP=${GEN_TP:-1}
PROMPT_LENGTH=${PROMPT_LENGTH:-768}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-32}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

CONFIGS=(
  "EEDI_aggreated"
  "EEDI_individual"
  "OpinionQA_aggreated"
  "OpinionQA_individual"
  "Twin-2K-500_aggreated"
  "Twin-2K-500_individual"
)

if [[ -n "${ONLY_CFG:-}" ]]; then
  CONFIGS=("${ONLY_CFG}")
fi

latest_step_dir() {
  local root="$1"
  ls -dt "${root}"/global_step_* 2>/dev/null | head -n 1
}

for CFG in "${CONFIGS[@]}"; do
  FSDP_ROOT="${CKPT_ROOT}/${CFG}"
  STEP_DIR="$(latest_step_dir "${FSDP_ROOT}")"
  if [[ -z "${STEP_DIR}" ]]; then
    echo "[skip] no checkpoint at ${FSDP_ROOT}"
    continue
  fi
  HF_CKPT="${FSDP_ROOT}/hf_merged"
  TEST="${PROJ_ROOT}/dataset/sft/${CFG}/test.parquet"
  OUT="${GEN_DIR}/${CFG}.parquet"
  EVAL_OUT="${RES_DIR}/${CFG}.json"

  # 1) Merge FSDP -> HF (idempotent)
  if [[ ! -f "${HF_CKPT}/config.json" ]]; then
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${STEP_DIR}" \
      --target_dir "${HF_CKPT}" \
      --trust-remote-code
  fi

  # 2) Generate with vLLM-backed main_generation_server
  python3 -m verl.trainer.main_generation_server \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NGPUS}" \
    actor_rollout_ref.model.path="${HF_CKPT}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n=1 \
    data.train_files="['${TEST}']" \
    data.prompt_key=messages \
    +data.output_path="${OUT}"

  # 3) Score with custom JSON-answer / hard-accuracy scorer
  python3 -m verl.trainer.main_eval \
    data.path="${OUT}" \
    custom_reward_function.path="${PROJ_ROOT}/run/sft/compute_score.py" \
    custom_reward_function.name=compute_score_data_source \
    sample_aggregation=mean \
    +output_json_path="${EVAL_OUT}" \
    +model_name="${CFG}" \
    +pass_k=1
done

# 4) Aggregate 6 per-config JSONs + MAE from gen_results -> results/sft.json
python3 "${PROJ_ROOT}/run/sft/aggregate_results.py"
