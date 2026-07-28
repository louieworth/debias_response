#!/usr/bin/env bash
# Sequential SFT training over the 6 (dataset, level) configs.
#
# Env var overrides:
#   MODEL_ID              default Qwen/Qwen3-4B
#   CKPT_ROOT             default /data/data/jiangli/debias
#   NGPUS                 default 1
#   FSDP_SIZE             default ${NGPUS}
#   SP_SIZE               default 1
#   TOTAL_EPOCHS          default 1
#   TRAIN_BSZ             default 16
#   MAX_LENGTH            default 768
#   MAX_TOKEN_LEN_PER_GPU default 32768
#   TORCH_PORT            default 29500     (set unique port per parallel launch)
#   OPT_OFFLOAD           default True      (offload Adam states to CPU)
#   ONLY_CFG              optional, run only one config (substring match)
set -xeuo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-4B}
CKPT_ROOT=${CKPT_ROOT:-/data/data/jiangli/debias}
PROJ_ROOT=/home/jiangli/debias_response
NGPUS=${NGPUS:-1}
FSDP_SIZE=${FSDP_SIZE:-${NGPUS}}
SP_SIZE=${SP_SIZE:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TRAIN_BSZ=${TRAIN_BSZ:-16}
MAX_LENGTH=${MAX_LENGTH:-768}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
TORCH_PORT=${TORCH_PORT:-29500}
OPT_OFFLOAD=${OPT_OFFLOAD:-True}

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

for CFG in "${CONFIGS[@]}"; do
  TRAIN="${PROJ_ROOT}/dataset/sft/${CFG}/train.parquet"
  CKPT="${CKPT_ROOT}/${CFG}"
  mkdir -p "${CKPT}"

  torchrun --nnodes=1 --nproc-per-node="${NGPUS}" \
    --rdzv_backend=c10d --rdzv_endpoint="localhost:${TORCH_PORT}" \
    -m verl.trainer.sft_trainer \
      data.train_files="${TRAIN}" \
      data.train_batch_size="${TRAIN_BSZ}" \
      data.max_length="${MAX_LENGTH}" \
      data.pad_mode=no_padding \
      data.truncation=error \
      data.use_dynamic_bsz=True \
      data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
      data.messages_key=messages \
      data.enable_thinking_key=enable_thinking \
      data.enable_thinking_default=False \
      data.ignore_input_ids_mismatch=True \
      model.path="${MODEL_ID}" \
      model.use_remove_padding=True \
      model.trust_remote_code=True \
      model.enable_gradient_checkpointing=True \
      engine=fsdp \
      optim=fsdp \
      optim.lr=1e-5 \
      optim.lr_warmup_steps_ratio=0.03 \
      optim.weight_decay=0.1 \
      optim.warmup_style=cosine \
      optim.clip_grad=1.0 \
      engine.strategy=fsdp2 \
      engine.fsdp_size="${FSDP_SIZE}" \
      engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
      engine.optimizer_offload="${OPT_OFFLOAD}" \
      engine.use_torch_compile=False \
      engine.model_dtype=bf16 \
      '+engine.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer]' \
      trainer.project_name=debias_sft \
      trainer.experiment_name="${CFG}" \
      trainer.total_epochs="${TOTAL_EPOCHS}" \
      trainer.test_freq=-1 \
      trainer.save_freq=-1 \
      trainer.max_ckpt_to_keep=1 \
      trainer.resume_mode=disable \
      trainer.logger='[console]' \
      trainer.default_local_dir="${CKPT}" \
      checkpoint.save_contents='[model,extra]'
done
