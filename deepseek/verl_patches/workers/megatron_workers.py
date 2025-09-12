# coding=utf-8
# Adapted from 
# https://github.com/volcengine/verl/blob/v0.4.0/verl/workers/megatron_workers.py
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

import importlib
import logging
import os
import sys
import time
from omegaconf import DictConfig
from codetiming import Timer
import torch

from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.workers.megatron_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.model import load_megatron_gptmodel_weights
from verl.workers.actor.megatron_actor import MegatronPPOActor

from verl_patches.train_engine.initialize_training import translate_verl_train_configs_to_megatron, initialize_megatron
from verl_patches.tools import print_memory
from verl_patches.tensor_cache import TensorCache

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def actor_rollout_ref_worker_init(self, config: DictConfig, role: str):
    """
    ActorRolloutRefWorker.__init__ patch
    Switch to the complete megatron & mindspeed initialization, and use the same worker as actor_rollout and Ref
    """
    super(ActorRolloutRefWorker, self).__init__()
    self.data_info = config.data
    self.config = config.actor_rollout_ref

    # NOTE：No longer use the simple initialization of mpu.initialize_model_parallel and set_random_seed,
    # but instead use the complete initialization of meagtron and mindspeed
    megatron_config = translate_verl_train_configs_to_megatron(config)
    initialize_megatron(config=megatron_config)

    # The following code mainly references patches
    from verl_patches import verl_adaptor
    import verl.workers.megatron_workers as megatron_workers
    importlib.reload(sys.modules[__name__])
    importlib.reload(megatron_workers)

    self.role = role
    assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

    self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
    self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
    self._is_ref = self.role in ["ref", "actor_rollout_ref"]

    # will support other offload later
    self._is_offload_param = False
    self._is_offload_grad = False
    self._is_offload_optimizer = False
    self._should_load_megatron_model = True

    # NOTE: TensorCache, for D2D tensor transfer optimization between RLHF stages to speed up ray task distribution
    self.tensor_cache = TensorCache(self.config)

    # normalize config
    if self._is_actor and self._is_rollout:
        self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
        self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.actor.get("ppo_micro_batch_size", None):
            self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

        self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
        self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
        self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
    if self._is_ref:
        if self.config.ref.get("ppo_micro_batch_size", None):
            self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.ppo_micro_batch_size
        self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)


def actor_rollout_ref_build_model_optimizer(
    self, model_path, optim_config, override_model_config, override_transformer_config, is_for_actor
):
    """
    is_for_actor: Add this input parameter to prevent self.init_model from repeatedly initializing
    the actor and ref models when Actor, rolloutref and other three work together
    """
    from megatron.core.models.gpt.gpt_model import ModelType

    from verl.utils.megatron.optimizer import get_megatron_optimizer
    from verl.utils.megatron_utils import get_model, init_megatron_optim_config
    from verl.utils.model import get_generation_config, print_model_size

    from megatron.training.checkpointing import load_checkpoint
    from megatron.training import get_args

    self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config)
    self.generation_config = get_generation_config(self.local_path)

    def megatron_actor_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model

        parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, value=False, fix_moe_router=override_model_config.get("moe_config", {}).get("fix_moe_router", False))
        parallel_model.cuda()
        return parallel_model

    # Step 3: initialize the megatron model
    if self._is_actor and self._is_rollout and is_for_actor:
        print_memory("before init actor model")
        actor_module = get_model(
            megatron_actor_model_provider,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
        )
        logger.info(f"actor_module: {len(actor_module)}")
        print_memory("after init actor model")
        if self.config.actor.load_weight:
            if self.config.actor.megatron.use_dist_checkpointing:
                setattr(get_args(), "load", self.config.actor.megatron.dist_checkpointing_path)
                load_checkpoint(actor_module, None, None)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

        if self.rank == 0:
            print_model_size(actor_module[0])
        log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
    elif self._is_ref and (not is_for_actor):
        print_memory("before init ref model")
        logger.info(f"self.config.ref.load_weight: {self.config.ref.load_weight}")
        ref_module = get_model(
            model_provider_func=megatron_actor_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
        )
        print_memory("after init ref model")

        if self.config.ref.load_weight:  # should align with the actor:
            assert self.config.actor.load_weight == self.config.ref.load_weight
            logger.info("load ref weight start")
            if self.config.ref.megatron.use_dist_checkpointing:
                setattr(get_args(), "load", self.config.ref.megatron.dist_checkpointing_path)
                load_checkpoint(ref_module, None, None)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)
        log_gpu_memory_usage("After ref module init", logger=logger)
        return ref_module, self.hf_config

    if is_for_actor:
        print_memory("before init optimizer")
        optim_config = init_megatron_optim_config(optim_config)
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        print_memory("after init optimizer")
    else:
        optim_config = None
        actor_optimizer = None

    log_gpu_memory_usage("After actor optimizer init", logger=logger)

    return actor_module, actor_optimizer, self.hf_config, optim_config


def actor_rollout_ref_build_rollout(self, trust_remote_code=False):
    from torch.distributed.device_mesh import init_device_mesh

    layer_name_mapping = {
        "qkv_layer_name": "self_attention.linear_qkv.",
        "gate_proj_layer_name": "linear_fc1.weight",
    }
    if self.config.rollout.name == "vllm":
        from torch.distributed.device_mesh import init_device_mesh

        from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
        from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

        # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
        # we will reorganize their weight format when resharding from actor to rollout.

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        log_gpu_memory_usage("Before building vllm rollout", logger=None)

        global_batch_size = self.data_info.train_batch_size * self.config.rollout.n
        max_num_seqs = global_batch_size // dp

        local_path = copy_to_local(self.config.model.path)
        if vllm_mode == "customized":
            rollout = vLLMRollout(
                actor_module=self.actor_module,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
            )
        elif vllm_mode == "spmd":
            rollout = vLLMRollout(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
                max_num_seqs=max_num_seqs,
            )
        log_gpu_memory_usage("After building vllm rollout", logger=logger)

        # perform weight resharding between actor and rollout
        from verl.models.mcore import get_mcore_weight_converter
        print_memory("before init MegatronVLLMShardingManager")
        weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        sharding_manager = MegatronVLLMShardingManager(
            rollout=rollout,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            layer_name_mapping=layer_name_mapping,
            actor_module=self.actor.actor_module,
            weight_converter=weight_converter,
        )
        print_memory("after init MegatronVLLMShardingManager")
        log_gpu_memory_usage("After building sharding manager", logger=logger)
    else:
        raise NotImplementedError("Only vllmRollout is supported with Megatron now")

    return rollout, sharding_manager


@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def actor_rollout_ref_init_model(self):
    if os.getenv("ROLLOUT_REBALANCE_ENABLE", "0") != "0":
        from features.rollout_optimize.rollout_rebalance import enable_rollout_rebalance
        enable_rollout_rebalance()

    if self.config.model.get("external_lib", None) is not None:
        # This is used to import external_lib into the huggingface systems
        importlib.import_module(self.config.model.external_lib)

    from omegaconf import OmegaConf

    from verl.utils.torch_dtypes import PrecisionType

    override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
    if self._is_actor:
        override_transformer_config = OmegaConf.to_container(self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
    elif self._is_ref:
        override_transformer_config = OmegaConf.to_container(self.config.ref.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
    else:
        override_transformer_config = None
    self.param_dtype = torch.bfloat16
    log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
    self.dtype = PrecisionType.to_dtype(self.param_dtype)
    if self._is_actor or self._is_rollout:
        # we need the model for actor and rollout
        optim_config = self.config.actor.optim if self._is_actor else None
        self.actor_module, self.actor_optimizer, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            is_for_actor=True,
        )
        if self._is_offload_param:
            print_memory("before offload actor model during init")
            offload_megatron_model_to_cpu(self.actor_module)
            print_memory("after offload actor model during init")
            log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
        if self._is_offload_optimizer:
            print_memory("before offload optimizer during init")
            offload_megatron_optimizer(self.actor_optimizer)
            print_memory("after offload optimizer during init")
            log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

    if self._is_actor:
        self.actor = MegatronPPOActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

    if self._is_rollout:
        self.rollout, self.sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        log_gpu_memory_usage("After rollout init", logger=logger)

    if self._is_ref:
        self.ref_module, self.ref_model_config = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=None,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            is_for_actor=False,
        )
        log_gpu_memory_usage("After ref model init", logger=logger)
        self.ref_policy = MegatronPPOActor(
            config=self.config.ref,
            model_config=self.ref_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.ref_module,
            actor_optimizer=None,
        )
        if self._ref_is_offload_param:
            print_memory("before offload ref model during init")
            offload_megatron_model_to_cpu(self.ref_module)
            print_memory("after offload ref model during init")
            log_gpu_memory_usage("After offload ref params during init", logger=logger)

    if self._is_actor:
        self.flops_counter = FlopsCounter(self.actor_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.actor_model_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            tokenizer=self.tokenizer,
            optimizer=self.actor_optimizer,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.actor.checkpoint.contents,
        )
    torch.cuda.empty_cache()
    log_gpu_memory_usage("After init_model finish", logger=logger)


@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
@GPUMemoryLogger(role="update_actor", logger=logger)
def actor_rollout_ref_worker_update_actor(self, data: DataProto):
    assert self._is_actor
    if self._is_offload_param:
        print_memory("before load actor model during update")
        load_megatron_model_to_gpu(self.actor_module)
        print_memory("after load actor model during update")
        log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
    if self._is_offload_optimizer:
        print_memory("before load optimizer during update")
        load_megatron_optimizer(self.actor_optimizer)
        print_memory("after load optimizer during update")
        log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
    data.batch = data.batch.cuda()

    if os.getenv("D2D_DATA_TRANSFER", "0") != "0":
        # get the data used in make_minibatch_iterator for update
        keys_to_get = ["responses", "input_ids", "attention_mask", "position_ids", "ref_log_prob"]
        data_cached = self.tensor_cache.get_cached_tensors(data, keys_to_get)
        data = data.union(data_cached)

    micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
    data.meta_info["micro_batch_size"] = micro_batch_size
    dataloader = self.actor.make_minibatch_iterator(data=data)
    with Timer(name="update_policy", logger=None) as timer:
        train_start = time.time()
        metrics = self.actor.update_policy(dataloader=dataloader)
        if torch.distributed.get_rank() == 0:
            logger.info(f"update actor on rank_0 cost {time.time() - train_start} seconds.")

    delta_time = timer.last
    global_num_tokens = data.meta_info["global_token_num"]
    estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
    metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

    output = DataProto(meta_info={"metrics": metrics})
    output = output.to("cpu")

    if self._is_offload_param:
        print_memory("before offload actor grad during update")
        # generate sequences needs to use weight and not grad, only need to offload grad here
        offload_megatron_model_to_cpu(self.actor_module, False)
        print_memory("after offload actor grad during update")
        log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
    if self._is_offload_optimizer:
        print_memory("before offload optimizer during update")
        offload_megatron_optimizer(self.actor_optimizer)
        print_memory("after offload optimizer during update")
        log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

    if os.getenv("D2D_DATA_TRANSFER", "0") != "0":
        # clear the cache on NPU after update
        self.tensor_cache.clear()

    torch.cuda.empty_cache()
    print_memory("after actor update once")
    return output


@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
@GPUMemoryLogger(role="generate_sequences", logger=logger)
def actor_rollout_ref_generate_sequences(self, prompts: DataProto):
    print_memory("before gen during training")
    assert self._is_rollout
    is_validate = prompts.meta_info.get("validate", False)
    # The init_model and validate stages will offload the Megatron model,
    # it needs reload in the first step and after validation
    if self._is_offload_param and self._should_load_megatron_model:
        print_memory("before load actor model during training gen")
        load_megatron_model_to_gpu(self.actor_module, load_grad=False)
        self._should_load_megatron_model = False
        print_memory("after load actor model during training gen")
        log_gpu_memory_usage("After load actor params during generate_sequences", logger=logger)
    if is_validate:
        self._should_load_megatron_model = True
    prompts.batch = prompts.batch.cuda()
    meta_info = {
        "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
        "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
    }
    prompts.meta_info.update(meta_info)
    with self.sharding_manager:
        if self._is_offload_param:
            print_memory("before offload actor model during training gen")
            offload_megatron_model_to_cpu(self.actor_module)
            print_memory("after offload actor model during training gen")
        if self._is_offload_optimizer:
            print_memory("before offload optimizer during training gen")
            offload_megatron_optimizer(self.actor_optimizer)
            print_memory("after offload optimizer during training gen")
        log_gpu_memory_usage("After entering sharding manager", logger=logger)

        print_memory("before init vllm cache during training gen")
        self.rollout.init_cache_engine()
        print_memory("after init vllm cache during training gen")

        prompts = self.sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.sharding_manager.postprocess_data(output)

    if os.getenv("D2D_DATA_TRANSFER", "0") != "0":
        # cache tensors on NPU for subsequent stages
        if not is_validate:
            keys_to_reserve = ["responses", "attention_mask"]
            keys_no_cache = ["prompts"]
            self.tensor_cache.cache_tensors(output, keys_to_reserve=keys_to_reserve, keys_no_cache=keys_no_cache)

    output = output.to("cpu")
    # clear kv cache
    torch.cuda.empty_cache()
    print_memory("after gen during training")
    return output


@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
@GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
def actor_rollout_ref_worker_compute_ref(self, data: DataProto):
    data = data.to("cuda")

    if os.getenv("D2D_DATA_TRANSFER", "0") != "0":
        # get cached tensors needed by ref
        keys_to_get = ["responses", "input_ids", "attention_mask", "position_ids"]
        data_cached = self.tensor_cache.get_cached_tensors(data, keys_to_get)
        data = data.union(data_cached)

    assert self._is_ref
    if self._ref_is_offload_param:
        print_memory("before load ref model during training")
        load_megatron_model_to_gpu(self.ref_module, load_grad=False)
        print_memory("after load ref model during training")
        log_gpu_memory_usage("After load ref params and grad during compute_ref_log_prob", logger=logger)
    micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
    data.meta_info["micro_batch_size"] = micro_batch_size
    data.meta_info["temperature"] = self.config.rollout.temperature
    ref_start = time.time()
    output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
    if torch.distributed.get_rank() == 0:
        logger.info(f"compute ref_log_prob on rank_0 cost {time.time() - ref_start} seconds.")
    output = DataProto.from_dict(tensors={"ref_log_prob": output})

    if os.getenv("D2D_DATA_TRANSFER", "0") != "0":
        # cache the ref_log_prob for loss computation in update_actor and remove it from output
        # note that ref_log_prob is not needed by the logic on driver process
        self.tensor_cache.cache_tensors(output)

    output = output.to("cpu")
    if self._ref_is_offload_param:
        print_memory("before offload ref model during training")
        offload_megatron_model_to_cpu(self.ref_module)
        print_memory("after offload ref model during training")
        log_gpu_memory_usage("After offload ref params and grad during compute_ref_log_prob", logger=logger)
    torch.cuda.empty_cache()
    return output


# apply patches for megatron_worker
ActorRolloutRefWorker.__init__ = actor_rollout_ref_worker_init
ActorRolloutRefWorker._build_model_optimizer = actor_rollout_ref_build_model_optimizer
ActorRolloutRefWorker._build_rollout = actor_rollout_ref_build_rollout
ActorRolloutRefWorker.init_model = actor_rollout_ref_init_model
ActorRolloutRefWorker.update_actor = actor_rollout_ref_worker_update_actor
ActorRolloutRefWorker.generate_sequences = actor_rollout_ref_generate_sequences
ActorRolloutRefWorker.compute_ref_log_prob = actor_rollout_ref_worker_compute_ref
