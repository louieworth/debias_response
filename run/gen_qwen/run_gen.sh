#!/usr/bin/env bash
# Run vLLM-backed generation of open-source LLM responses on a prepared
# gen_prompts parquet. The model path can be a HF id (e.g. Qwen/Qwen3-4B)
# or a local directory.
#
# Required env:
#   DATASET      EEDI | OpinionQA | Twin-2K-500
#   LEVEL        aggreated | individual
#   MODEL_TAG    short tag used in output filenames (e.g. qwen3-4B)
#   MODEL_PATH   HF id or local path to the LLM (default: Qwen/${MODEL_TAG})
# Optional env:
#   NGPUS               default 1
#   GEN_TP              default 1
#   PROMPT_LENGTH       default 768
#   RESPONSE_LENGTH     default 32
#   GPU_MEM_UTIL        default 0.9
#   VERL_GEN_CONCURRENCY default 256 (bounded asyncio; avoids port-exhaustion)
set -xeuo pipefail

: "${DATASET:?}"
: "${LEVEL:?}"
: "${MODEL_TAG:?}"
MODEL_PATH=${MODEL_PATH:-Qwen/${MODEL_TAG}}

PROJ_ROOT=/home/jiangli/debias_response
NGPUS=${NGPUS:-1}
GEN_TP=${GEN_TP:-1}
PROMPT_LENGTH=${PROMPT_LENGTH:-768}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-32}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
export VERL_GEN_CONCURRENCY=${VERL_GEN_CONCURRENCY:-256}

IN="${PROJ_ROOT}/dataset/gen_prompts/${DATASET}_${LEVEL}/prompts.parquet"
OUT_DIR="${PROJ_ROOT}/gen_results/qwen/${DATASET}_${LEVEL}"
OUT="${OUT_DIR}/${MODEL_TAG}.parquet"
mkdir -p "${OUT_DIR}"

python3 -m verl.trainer.main_generation_server \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NGPUS}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.n=1 \
  data.train_files="['${IN}']" \
  data.prompt_key=messages \
  +data.output_path="${OUT}"

echo "wrote ${OUT}"
