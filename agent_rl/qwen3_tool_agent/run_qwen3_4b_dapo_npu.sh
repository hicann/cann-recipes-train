# Adapted from
# https://github.com/volcengine/verl/blob/v0.6.0/recipe/retool/run_qwen2_7b_dapo.sh
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
source /usr/local/Ascend/nnal/atb/set_env.sh

export HYDRA_FULL_ERROR=1                   # display the accurate error stack
export ASCEND_LAUNCH_BLOCKING=0             # debug usage, which seriously affects performance after use, but the error stack is accurate
export RAY_DEDUP_LOGS=1                     # 0: disable ray's log folding 1: enable ray's log folding

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15    # specify certain devices in the environment for use
export ASCEND_GLOBAL_EVENT_ENABLE=0         # whether to display the Ascend EVENT log; 1 is for display
export ASCEND_SLOG_PRINT_TO_STDOUT=0        # does the Ascend log screen out? 0 does not screen. By default, it is at /root/ascend/log. 1 directly screens
export ASCEND_GLOBAL_LOG_LEVEL=3            # log level: 3: ERROR  2:EVENT  1:INFO   0:DEBUG

export HCCL_CONNECT_TIMEOUT=360
export HCCL_EXEC_TIMEOUT=360
export HCCL_IF_BASE_PORT=64033
export HCCL_OP_EXPANSION_MODE=AIV
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -x

export VLLM_USE_V1=1

# ================= data/model/tool =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

dapo_math_17k=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k
aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025
model_path=$HDFS_ROOT/checkpoint/multiturn-sft-qwen-3-4b-instruct/global_step_186/huggingface

train_files="['$dapo_math_17k']"
test_files="['$aime_2025']"

# tool
tool_config_path=sandbox_fusion_tool_config.yaml

# wandb
project_name=retool
experiment_name=qwen3-4b_dapo_$(data +%m%d_%H%M)
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=16
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=16
n_resp_per_prompt_val=30

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=4 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=16 \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.device=npu 2>&1 | tee log/rl_run_log/qwen3_4b_dapo_$(date +%Y%m%d_%H%M%S).log
    $@
