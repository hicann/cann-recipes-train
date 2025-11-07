# coding=utf-8
# Adapted from
# https://github.com/volcengine/verl/blob/v0.4.0/examples/grpo_trainer/run_qwen2-7b_math_megatron.sh
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates. All rights reserved.
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


export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HYDRA_FULL_ERROR=1                   # display the accurate error stack
export ASCEND_LAUNCH_BLOCKING=0             # debug usage, which seriously affects performance after use, but the error stack is accurate
export RAY_DEDUP_LOGS=1                     # 0: disable ray's log folding 1: enable ray's log folding

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3    # specify certain devices in the environment for use
export ASCEND_GLOBAL_EVENT_ENABLE=0         # whether to display the Ascend EVENT log; 1 is for display
export ASCEND_SLOG_PRINT_TO_STDOUT=0        # does the Ascend log screen out? 0 does not screen. By default, it is at /root/ascend/log. 1 directly screens
export ASCEND_GLOBAL_LOG_LEVEL=3            # log level: 3: ERROR  2:EVENT  1:INFO   0:DEBUG

export HCCL_CONNECT_TIMEOUT=360
export HCCL_EXEC_TIMEOUT=360
export HCCL_IF_BASE_PORT=64033
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/math/data/train.parquet \
    data.val_files=./data/math/data/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=./verl/utils/reward_score/new_math_reward.py \
    actor_rollout_ref.model.path=./model/Qwen2_5_1_5B_Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_example_math' \
    trainer.experiment_name='qwen2_5_1_5b_math' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.device=npu 2>&1 | tee ./run_log/qwen2_5_1_5b_math.log
