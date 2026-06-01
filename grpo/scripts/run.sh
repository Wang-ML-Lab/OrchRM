#!/usr/bin/env bash
# OrchRM GRPO training launcher.
#
# Required env vars:
#   MODEL_PATH      — path to the actor base model (HF format)
#   RM_MODEL_PATH   — path to the trained OrchRM reward model
#
# Optional env vars (all have sensible defaults):
#   TRAIN_FILE, VAL_FILE, EXP_NAME, PROJECT_NAME, GPU_IDS, ...
#
# Usage:
#   bash grpo/scripts/run.sh path/to/train.parquet path/to/val.parquet [extra verl args...]

set -euo pipefail

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Resolve repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONNOUSERSITE=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Temp directories — override via env if needed
export TMPDIR="${TMPDIR:-/tmp/orchrm}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}/ray}"
mkdir -p "${RAY_TMPDIR}"

# W&B — set WANDB_ENTITY in your environment before running
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-orchrm-grpo}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/outputs/wandb}"
mkdir -p "${WANDB_DIR}"

TRAIN_FILE="${1:-}"
VAL_FILE="${2:-}"

if [[ -z "${TRAIN_FILE}" || -z "${VAL_FILE}" ]]; then
  echo "Usage: $0 <train.parquet> <val.parquet> [extra args...]"
  exit 1
fi

(( $# >= 2 )) && shift 2

# === Required paths ===
MODEL_PATH="${MODEL_PATH:-}"
RM_MODEL_PATH="${RM_MODEL_PATH:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is required. Set it to the HF model directory."
  exit 1
fi

# === Model config ===
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
EXP_NAME="${EXP_NAME:-orchrm_grpo}"
PROJECT_NAME="${PROJECT_NAME:-OrchRM}"

# === Actor LoRA config ===
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

# === GPU ===
GPU_IDS="${GPU_IDS:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
AUTO_SET_GPUS_PER_NODE="${AUTO_SET_GPUS_PER_NODE:-0}"
NNODES="${NNODES:-1}"

# === Training config ===
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6144}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.9}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-16384}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1}"
ROLLOUT_N="${ROLLOUT_N:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-5}"

# === Reward Model config ===
USE_RM="${USE_RM:-1}"
RM_ROLLOUT_NAME="${RM_ROLLOUT_NAME:-hf}"
RM_GPU_MEM_UTIL="${RM_GPU_MEM_UTIL:-0.5}"
RM_PROMPT_LENGTH="${RM_PROMPT_LENGTH:-6144}"
RM_RESPONSE_LENGTH="${RM_RESPONSE_LENGTH:-256}"
RM_MAX_MODEL_LEN="${RM_MAX_MODEL_LEN:-8192}"
RM_MAX_NUM_SEQS="${RM_MAX_NUM_SEQS:-1}"
RM_MAX_BATCHED_TOKENS="${RM_MAX_BATCHED_TOKENS:-8192}"
RM_RESOURCE_POOL="${RM_RESOURCE_POOL:-1}"
RM_NNODES="${RM_NNODES:-1}"
RM_GPUS_PER_NODE="${RM_GPUS_PER_NODE:-1}"

CUSTOM_REWARD_FN_PATH="${REPO_ROOT}/grpo/reward_function.py"
CUSTOM_REWARD_FN_NAME="${CUSTOM_REWARD_FN_NAME:-compute_score}"
VERL_CONFIG_PATH="${REPO_ROOT}/verl/verl/trainer/config"

# === Logging ===
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/outputs/logs}"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
export LOG_FILE
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

# === Output directories ===
OUTPUT_ROOT="${REPO_ROOT}/outputs/${EXP_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUTPUT_ROOT}/rollout}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUTPUT_ROOT}/validation}"
mkdir -p "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  if [[ "${AUTO_SET_GPUS_PER_NODE}" == "1" ]]; then
    GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
  fi
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "WARN: clearing PYTORCH_CUDA_ALLOC_CONF (incompatible with vLLM memory pool)"
  unset PYTORCH_CUDA_ALLOC_CONF
fi

EXTRA_ARGS=()
if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH}")
fi

if [[ "${USE_RM}" == "1" ]]; then
  if [[ -z "${RM_MODEL_PATH}" ]]; then
    echo "ERROR: USE_RM=1 but RM_MODEL_PATH is empty. Set RM_MODEL_PATH to your trained reward model."
    exit 1
  fi

  if [[ "${RM_ROLLOUT_NAME}" != "vllm" && "${RM_ROLLOUT_NAME}" != "sglang" && "${RM_ROLLOUT_NAME}" != "trtllm" && "${RM_ROLLOUT_NAME}" != "hf" ]]; then
    echo "ERROR: Unsupported RM_ROLLOUT_NAME=${RM_ROLLOUT_NAME}. Supported: vllm, sglang, trtllm, hf."
    exit 1
  fi

  if [[ "${RM_ROLLOUT_NAME}" == "vllm" && -f "${RM_MODEL_PATH}/config.json" ]] \
    && grep -q "SequenceClassification" "${RM_MODEL_PATH}/config.json"; then
    echo "ERROR: RM is SequenceClassification — use RM_ROLLOUT_NAME=hf instead of vllm."
    exit 1
  fi

  EXTRA_ARGS+=(
    data.return_raw_chat=True
    reward.reward_model.enable=True
    "reward.custom_reward_function.path=${CUSTOM_REWARD_FN_PATH}"
    "reward.custom_reward_function.name=${CUSTOM_REWARD_FN_NAME}"
    "reward.reward_model.model_path=${RM_MODEL_PATH}"
    "reward.reward_model.rollout.name=${RM_ROLLOUT_NAME}"
    "reward.reward_model.rollout.gpu_memory_utilization=${RM_GPU_MEM_UTIL}"
    reward.reward_model.rollout.tensor_model_parallel_size=1
    "reward.reward_model.rollout.prompt_length=${RM_PROMPT_LENGTH}"
    "reward.reward_model.rollout.response_length=${RM_RESPONSE_LENGTH}"
    "reward.reward_model.rollout.max_model_len=${RM_MAX_MODEL_LEN}"
    "reward.reward_model.rollout.max_num_seqs=${RM_MAX_NUM_SEQS}"
    "reward.reward_model.rollout.max_num_batched_tokens=${RM_MAX_BATCHED_TOKENS}"
  )

  if [[ "${RM_RESOURCE_POOL}" == "1" ]]; then
    EXTRA_ARGS+=(
      reward.reward_model.enable_resource_pool=True
      "reward.reward_model.nnodes=${RM_NNODES}"
      "reward.reward_model.n_gpus_per_node=${RM_GPUS_PER_NODE}"
    )
  fi

  if [[ "${RM_ROLLOUT_NAME}" == "vllm" ]]; then
    EXTRA_ARGS+=(+reward.reward_model.rollout.logprobs_mode=null)
  fi
fi

echo "GPU plan: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} actor_gpus_per_node=${GPUS_PER_NODE} rm_gpus_per_node=${RM_GPUS_PER_NODE}"

python -m verl.trainer.main_ppo \
  --config-path="${VERL_CONFIG_PATH}" \
  --config-name=ppo_trainer.yaml \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=False \
  data.truncation=right \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  "actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}" \
  +"actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.logprobs_mode=null \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}" \
  "actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}" \
  "actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}" \
  algorithm.use_kl_in_reward=False \
  critic.enable=False \
  trainer.critic_warmup=0 \
  trainer.resume_mode=disable \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  "+ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}" \
  "${EXTRA_ARGS[@]}" \
  hydra.run.dir="${HYDRA_RUN_DIR:-${REPO_ROOT}/outputs/hydra/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}" \
  "$@"
