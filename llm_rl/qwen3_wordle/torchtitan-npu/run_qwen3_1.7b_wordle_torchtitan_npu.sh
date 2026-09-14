#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

BACKEND_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECIPE_DIR=$(cd "${BACKEND_DIR}/.." && pwd)
VENV_DIR=${VENV_DIR:-"${BACKEND_DIR}/.venv"}
SOURCES_DIR=${SOURCES_DIR:-"${BACKEND_DIR}/runtime_sources"}
PYTHON="${VENV_DIR}/bin/python3"

source_first_existing() {
    local candidate
    for candidate in "$@"; do
        if [[ -f "${candidate}" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "${candidate}"
            set -u
            return 0
        fi
    done
    return 1
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_directory() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 1
    fi
}

if ! source_first_existing \
    /home/developer/Ascend/cann-9.0.0/set_env.sh \
    /home/developer/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/cann/set_env.sh; then
    echo "CANN set_env.sh was not found." >&2
    exit 1
fi
# The pinned PyTorch wheel uses the C++11 ABI. Passing it explicitly avoids
# importing the host Python environment from ATB's setup script.
if [[ -f /home/developer/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /home/developer/Ascend/nnal/atb/set_env.sh --cxx_abi=1
    set -u
elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
    set -u
else
    echo "NNAL/ATB is required. Install the NNAL package matching CANN first." >&2
    exit 1
fi

require_file "${PYTHON}"
for source_name in verl torchtitan torchtitan-npu vllm vllm-ascend; do
    require_directory "${SOURCES_DIR}/${source_name}"
done

# Use local assets by default; each path can be overridden independently.
MODEL_PATH=${MODEL_PATH:-"${RECIPE_DIR}/models/Qwen3-1.7B-Wordle-SFT"}
TRAIN_FILE=${TRAIN_FILE:-"${RECIPE_DIR}/data/wordle_train.parquet"}
TEST_FILE=${TEST_FILE:-"${RECIPE_DIR}/data/wordle_test.parquet"}
REWARD_FILE="${RECIPE_DIR}/wordle_reward.py"
AGENT_LOOP_CONFIG="${BACKEND_DIR}/agent_loop.yaml"

require_directory "${MODEL_PATH}"
require_file "${TRAIN_FILE}"
require_file "${TEST_FILE}"
require_file "${REWARD_FILE}"
require_file "${AGENT_LOOP_CONFIG}"

MODEL_PATH=$(realpath "${MODEL_PATH}")
TRAIN_FILE=$(realpath "${TRAIN_FILE}")
TEST_FILE=$(realpath "${TEST_FILE}")
REWARD_FILE=$(realpath "${REWARD_FILE}")
AGENT_LOOP_CONFIG=$(realpath "${AGENT_LOOP_CONFIG}")

# Hardware and reproducibility.
SEED=${SEED:-42}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TORCHTITAN_FSDP_SIZE=${TORCHTITAN_FSDP_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
export PYTHONHASHSEED=${PYTHONHASHSEED:-${SEED}}

if [[ "${NNODES}" -ne 1 || "${NGPUS_PER_NODE}" -ne 2 ]]; then
    echo "This backend requires exactly one node with two NPUs." >&2
    exit 1
fi
if [[ "${TORCHTITAN_FSDP_SIZE}" -ne 2 ]]; then
    echo "TorchTitan FSDP2 shard size must be 2 for this backend." >&2
    exit 1
fi

export PATH="${VENV_DIR}/bin:${PATH}"
# Installed dependencies are resolved by the isolated environment. Keep the
# local Wordle AgentLoop importable in Ray workers.
export PYTHONPATH="${BACKEND_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Jupyter injects an inline backend that is unavailable in the isolated
# training environment and inherited Ray workers. Training is headless.
export MPLBACKEND=Agg
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}

# The aarch64 scikit-learn wheel renames its bundled OpenMP runtime. Load it
# before transformers imports sklearn so both OpenMP runtimes receive static
# TLS at process startup.
sklearn_libgomp=""
for candidate in \
    "${VENV_DIR}/lib/python3.11/site-packages/scikit_learn.libs/libgomp"*.so*; do
    if [[ -f "${candidate}" ]]; then
        sklearn_libgomp="${candidate}"
        break
    fi
done
if [[ -z "${sklearn_libgomp}" ]]; then
    echo "scikit-learn's bundled libgomp was not found." >&2
    exit 1
fi
export LD_PRELOAD="${sklearn_libgomp}${LD_PRELOAD:+:${LD_PRELOAD}}"

visible_npus=$(
    "${PYTHON}" -c \
        'import torch, torch_npu; print(torch.npu.device_count())'
)
if [[ "${visible_npus}" -ne 2 ]]; then
    echo "Expected exactly two visible NPUs, got ${visible_npus}." >&2
    exit 1
fi

# Preserve the existing launcher's GRPO and Wordle settings.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.004}
MAX_TURNS=${MAX_TURNS:-6}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.70}

PROJECT_NAME=${PROJECT_NAME:-wordle_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_wordle_torchtitan_$(date +%m%d_%H%M)}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${BACKEND_DIR}/checkpoint/${EXPERIMENT_NAME}"}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

ACTOR_MAX_TOKEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
LOG_PROB_MAX_TOKEN=$((ACTOR_MAX_TOKEN * 4))

trainer_step_args=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    trainer_step_args+=(
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    )
fi

cd "${BACKEND_DIR}"

command=(
    "${PYTHON}"
    "-m"
    "verl.trainer.main_ppo"

    # Backend and algorithm.
    "model_engine=torchtitan"
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_ctrl.kl_coef=0.0"

    # Data, reward, and model.
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.seed=${SEED}"
    "data.return_raw_chat=True"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "custom_reward_function.path=${REWARD_FILE}"
    "custom_reward_function.name=compute_score"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"

    # Actor and reference model.
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.decay_type=cosine"
    "actor_rollout_ref.actor.optim.min_lr_factor=0.1"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=5"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN}"
    "actor_rollout_ref.actor.torchtitan.param_offload=True"
    "actor_rollout_ref.actor.torchtitan.optimizer_offload=True"
    "actor_rollout_ref.actor.torchtitan.reshard_after_forward=always"
    "actor_rollout_ref.actor.torchtitan.forward_prefetch=False"
    "actor_rollout_ref.actor.torchtitan.use_torch_compile=False"
    "actor_rollout_ref.actor.torchtitan.data_parallel_replicate_size=1"
    "actor_rollout_ref.actor.torchtitan.data_parallel_shard_size=${TORCHTITAN_FSDP_SIZE}"
    "actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1"
    "actor_rollout_ref.actor.torchtitan.expert_parallel_size=1"
    "+actor_rollout_ref.actor.torchtitan.expert_tensor_parallel_size=1"
    "actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1"
    "actor_rollout_ref.actor.torchtitan.context_parallel_size=1"
    "actor_rollout_ref.actor.torchtitan.attn_type=varlen"
    "actor_rollout_ref.actor.torchtitan.max_seq_len=${ACTOR_MAX_TOKEN}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN}"
    "actor_rollout_ref.ref.torchtitan.param_offload=True"
    "actor_rollout_ref.ref.torchtitan.use_torch_compile=False"
    "actor_rollout_ref.ref.torchtitan.max_seq_len=${ACTOR_MAX_TOKEN}"

    # vLLM rollout and Wordle AgentLoop.
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.data_parallel_size=1"
    "actor_rollout_ref.rollout.expert_parallel_size=1"
    "actor_rollout_ref.rollout.pipeline_model_parallel_size=1"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.max_model_len=5120"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.multi_turn.enable=True"
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}"
    "actor_rollout_ref.rollout.multi_turn.format=hermes"
    "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable"
    "actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS}"
    "actor_rollout_ref.rollout.agent.default_agent_loop=wordle_agent"
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_LOOP_CONFIG}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.7"
    "actor_rollout_ref.rollout.val_kwargs.n=5"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"

    # Trainer.
    "trainer.logger=[\"console\",\"tensorboard\"]"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.use_legacy_worker_impl=disable"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.log_val_generations=5"
    "trainer.default_local_dir=${DEFAULT_LOCAL_DIR}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.device=npu"
    "${trainer_step_args[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
