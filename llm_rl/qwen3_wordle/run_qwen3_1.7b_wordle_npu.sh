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

# NPU env
export ASCEND_HOME_PATH=/home/developer/Ascend/ascend-toolkit
source /home/developer/Ascend/ascend-toolkit/set_env.sh
source /home/developer/Ascend/nnal/atb/set_env.sh
# export ASCEND_RT_VISIBLE_DEVICES=x,x
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

# ---- configurable ----
MODEL_PATH=${MODEL_PATH:-./models/Qwen3-1.7B-Wordle-SFT}
TRAIN_FILE=${TRAIN_FILE:-data/wordle_train.parquet}
TEST_FILE=${TEST_FILE:-data/wordle_test.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}

# Training
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ACTOR_LR=${ACTOR_LR:-1e-6}
MAX_TURNS=${MAX_TURNS:-6}
ROLLOUT_N=${ROLLOUT_N:-8}

# Rollout
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.80}

# Logging
PROJECT_NAME=${PROJECT_NAME:-wordle_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_wordle_$(date +%m%d_%H%M)}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-./checkpoint/$EXPERIMENT_NAME}

# Actor mem
ACTOR_MAX_TOKEN=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1 ))
LOG_PROB_MAX_TOKEN=$(( ACTOR_MAX_TOKEN * 4 ))

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.return_raw_chat=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=wordle_reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
    actor_rollout_ref.rollout.agent.default_agent_loop=wordle_agent \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.log_val_generations=5 \
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
    trainer.total_epochs=5 \
    trainer.device=npu \
    "$@"
