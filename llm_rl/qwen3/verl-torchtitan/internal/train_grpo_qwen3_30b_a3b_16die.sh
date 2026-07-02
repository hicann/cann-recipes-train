#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

set -x
export HYDRA_FULL_ERROR=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1  
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32
export MULTI_STREAM_MEMORY_RESERVE=2
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export HCCL_ALLOW_ALL_GATHER_INCONSISTENT=0

VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
NUM_GPUS=${NUM_GPUS:-16}
FSDP_SIZE=${FSDP_SIZE:-16}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-16}
VERL_EXP_NAME=${VERL_EXP_NAME:-qwen3-30B-A3B-GRPO-torchtitan}

MODEL_PATH=${MODEL_PATH:-"${HOME}/Qwen3-30B-A3B"}

python3 -m verl.trainer.main_ppo \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    data.seed=42 \
    data.train_files=/path/to/gsm8k/train.parquet \
    data.val_files=/path/to/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.min_lr_factor=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${FSDP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${EP_SIZE}" \
    actor_rollout_ref.actor.torchtitan.attn_type=sdpa \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.param_offload=True \
    actor_rollout_ref.actor.torchtitan.optimizer_offload=True \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward="always" \
    actor_rollout_ref.actor.torchtitan.mixed_precision=True \
    actor_rollout_ref.actor.torchtitan.entropy_checkpointing=True  \
    actor_rollout_ref.actor.torchtitan.forward_prefetch=True  \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=False \
    actor_rollout_ref.ref.torchtitan.param_offload=True \
    actor_rollout_ref.ref.torchtitan.optimizer_offload=True \
    actor_rollout_ref.ref.torchtitan.reshard_after_forward="always" \
    actor_rollout_ref.ref.torchtitan.mixed_precision=True \
    actor_rollout_ref.rollout.profiler.enable=False  \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.data_parallel_size=4 \
    actor_rollout_ref.rollout.expert_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_model_len=40960 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.backend='mlflow' \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger=['console','mlflow'] \
    trainer.rollout_data_dir="outputs/rollout_samples" \
    trainer.experiment_name="${VERL_EXP_NAME}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.del_local_ckpt_after_load=True \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=100 $@
