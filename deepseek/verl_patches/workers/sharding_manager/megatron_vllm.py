# Adapted from 
# https://github.com/volcengine/verl/blob/v0.4.0/verl/workers/sharding_manager/megatron_vllm.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
__all__ = [
    "MegatronVLLMShardingManager"
]

import logging
import os

import torch
import torch.distributed
from torch import nn

from verl import DataProto
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.megatron_utils import (
    get_model,
    per_tensor_generator,
    unwrap_model,
)
from vllm_ascend.patch import platform
from vllm_ascend.patch import worker
from verl.utils.torch_functional import check_cuda_is_available
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader
from verl.workers.sharding_manager.base import BaseShardingManager
from verl.workers.sharding_manager.megatron_vllm import AllGatherPPModel

from verl_patches.tools import print_memory

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronVLLMShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        actor_module: nn.ModuleList,
        rollout,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter: McoreToHFWeightConverterBase,
        module: AllGatherPPModel = None,
    ):
        from megatron.core import parallel_state as mpu

        self.actor_module = actor_module
        self.rollout = rollout
        self.inference_engine = rollout.inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.module = module
        # initialize groups for vllm inference
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.infer_tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.infer_tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.infer_tp_group = vllm_ps.get_tensor_model_parallel_group()
        if vllm_version not in ("0.5.4", "0.6.3"):
            self.infer_tp_group = self.infer_tp_group.device_group
        self.train_tp_size = mpu.get_tensor_model_parallel_world_size()
        self.train_tp_rank = mpu.get_tensor_model_parallel_rank()
        self.train_tp_group = mpu.get_tensor_model_parallel_group()
        self.train_ep_size = mpu.get_expert_model_parallel_world_size()
        self.train_ep_rank = mpu.get_expert_model_parallel_rank()
        self.train_ep_group = mpu.get_expert_model_parallel_group()
        self.need_tp_reshard = self.train_tp_size != self.infer_tp_size
        self.train_tp_larger = self.train_tp_size > self.infer_tp_size
        self.rollout_model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __enter__(self):
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            per_tensor_param = per_tensor_generator(self.actor_module, self.model_config, self.weight_converter, self.transformer_config, self.layer_name_mapping, convert_qkv_gate_up_by_simple_split=False)
            self.inference_engine.sync_model_weights(per_tensor_param, load_format="megatron")
        else:
            per_tensor_param = per_tensor_generator(
                self.actor_module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )
            print_memory("before load vllm model during training gen")
            self.rollout.onload_model_weights()
            patch_vllm_moe_model_weight_loader(self.rollout_model)

            torch.cuda.reset_peak_memory_stats()
            loaded_params = self.rollout_model.load_weights(per_tensor_param)
            logger.info(f"Check torch_npu reshard peak memory: {torch.cuda.max_memory_allocated() / 1024**3 :.3f} GB")

            if hasattr(self.rollout_model.model.layers[0].self_attn, "mla_attn"):
                self._process_mla()
            info = f"vLLM load weights, loaded_params: {len(loaded_params)}"
            logger.info(info)
            print_memory("after load vllm model during training gen")

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            self.inference_engine.offload_model_weights()
        else:
            print_memory("before offload vllm cache during training gen")
            self.rollout.free_cache_engine()
            print_memory("after offload vllm cache during training gen")
            self.rollout.offload_model_weights()
            print_memory("after offload vllm model during training gen")

        for model in self.actor_module:
            model.train()

        torch.cuda.empty_cache()

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.infer_tp_group)
        return data

    @GPUMemoryLogger(role="megatron vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.infer_tp_rank]

    def _process_mla(self):
        for i in range(self.rollout_model.model.start_layer, self.rollout_model.model.end_layer):
            mla = self.rollout_model.model.layers[i].self_attn.mla_attn.impl
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
            mla.process_weights_after_loading(None)
