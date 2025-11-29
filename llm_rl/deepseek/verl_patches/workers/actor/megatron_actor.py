# Adapted from
# https://github.com/volcengine/verl/blob/v0.4.0/verl/workers/actor/megatron_actor.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import logging
import os
from functools import wraps
from functools import partial
from typing import Dict, Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.megatron_actor import MegatronPPOActor

from verl_patches.profiler import NPUProfiler
from verl_patches.tools import print_memory, empty_cache

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def megatron_ppo_actor_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.iteration = 0
    return wrapper


@GPUMemoryLogger(role="megatron actor", logger=logger)
def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
    """Compute the log probability of the responses given input_ids, attention_mask and position_ids

    Args:
        data (DataProto): a DataProto containing keys

            ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
            concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

            ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

            ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

            ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

    Returns:
        DataProto: torch.Tensor: the log_prob tensor
    """
    data.batch = data.batch.contiguous()

    def compute_logprobs_fn(output, data):
        response = data["responses"]
        response_length = response.size(1)
        if self.config.use_packed_seq:
            log_probs = output["log_probs"][:, -response_length - 1: -1].contiguous()
            return {"log_probs": log_probs}
        logits = output
        logits = logits[:, -response_length - 1: -1].contiguous()
        log_probs = vocab_parallel_log_probs_from_logits(logits, response)
        return {"log_probs": log_probs}

    recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

    entropys = torch.Tensor()
    if recompute_old_log_prob:
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        input_ids = batch["input_ids"]
        batch_size = input_ids.size(0)
        response = batch["responses"]
        response_length = response.size(1)
        with torch.no_grad():
            output = self.forward_backward_batch(data, forward_only=True,
                post_process_fn=compute_logprobs_fn, calculate_entropy=calculate_entropy)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                if calculate_entropy:
                    log_probs = torch.cat([o[0]["log_probs"] for o in output], dim=0)  # (bs, seq_size)
                else:
                    log_probs = torch.cat([o["log_probs"] for o in output], dim=0)  # (bs, seq_size)
                log_probs = log_probs.to(torch.float32)
            else:
                log_probs = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)

            # broadcast across pp ranks
            torch.distributed.broadcast(
                tensor=log_probs,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
                async_op=False,
            )
            if calculate_entropy:
                # Note that o[0] is metrics, o[1] is entropy
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    entropys = torch.cat([o[1] for o in output], dim=0)
                    entropys = entropys.to(torch.float32)
                else:
                    entropys = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)
                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=entropys,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )

    # add empty cache after each compute
    empty_cache()

    return log_probs, entropys


def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
    """Make minibatch iterator for updating the actor

    Args:
        data (DataProto): a DataProto containing keys

            ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``

            ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

            ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

            ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that responses = input_ids[:, -response_length:]

            ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability of responses.

            ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of responses.
            See PPO paper for details. https://arxiv.org/abs/1707.06347

    Returns:

    """
    select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "advantages"]
    if self.config.recompute_old_log_prob:
        select_keys.append("old_log_probs")

    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")
    data = data.select(batch_keys=select_keys)
    return data.make_iterator(
        mini_batch_size=self.config.ppo_mini_batch_size,
        epochs=self.config.ppo_epochs,
        seed=self.config.data_loader_seed,
        dataloader_kwargs={"shuffle": self.config.shuffle},
    )


def forward_backward_batch(self, data: DataProto, forward_only=False, post_process_fn=None, calculate_entropy=False):
    """
    We assume:
    - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
    - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
    """
    # broadcast from last pp rank to all other pp ranks
    broadcast_dict_tensor(data.batch, src=mpu.get_pipeline_model_parallel_last_rank(), group=mpu.get_pipeline_model_parallel_group())
    # split into micro-batches
    data.batch["attention_mask"] = data.batch["attention_mask"].to(bool)

    if data.meta_info.get("micro_batch_size", None) is not None:
        batch_size = data.meta_info["micro_batch_size"]
    else:
        batch_size = self.config.ppo_micro_batch_size_per_gpu
    if self.config.use_dynamic_bsz:
        batches, _ = rearrange_micro_batches(batch=data.batch, max_token_len=self.config.max_packing_token_size)
    else:
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
    # compute input shapes for pp stages
    n_micro_batch = len(batches)
    seq_len = batches[0]["input_ids"].shape[1]

    forward_backward_func = get_forward_backward_func()

    def loss_func(output, data, meta_info):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        metrics = {}
        if self.config.use_packed_seq:
            device_type = output["log_probs"].device
        else:
            device_type = output.device
        if forward_only:
            if post_process_fn is None:
                metrics["logits"] = output
            else:
                stats = post_process_fn(output, data)
                metrics.update(stats)
            if not calculate_entropy:
                return torch.tensor(1.0, device=device_type), metrics

        responses = data["responses"]
        response_length = responses.size(1)
        attention_mask = data["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        loss_agg_mode = self.config.loss_agg_mode

        # compute policy loss
        ret_entropy = None
        if not self.config.use_packed_seq:
            logits = output[:, -response_length - 1: -1].contiguous()
        if not forward_only:
            advantages = data["advantages"]

            clip_ratio = meta_info["clip_ratio"]
            clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
            clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
            clip_ratio_c = meta_info["clip_ratio_c"]
            if self.config.use_packed_seq:
                log_prob = output["log_probs"][:, -response_length - 1: -1].contiguous()
            else:
                log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
            if self.config.recompute_old_log_prob:
                old_log_prob = data["old_log_probs"]
            else:
                old_log_prob = log_prob.detach()
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
                loss_agg_mode=loss_agg_mode,
            )
            policy_loss = pg_loss
        if calculate_entropy:
            if self.config.use_packed_seq:
                entropy = output["entropy"][:, -response_length - 1: -1].contiguous()
            else:
                entropy = vocab_parallel_entropy(logits)
            if not forward_only:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                entropy_coeff = meta_info["entropy_coeff"]
                policy_loss = pg_loss - entropy_coeff * entropy_loss
            else:
                ret_entropy = entropy

        stats = {}
        if forward_only:
            policy_loss = torch.tensor(1.0, device=device_type)
        else:
            if self.config.use_kl_loss:
                ref_log_prob = data["ref_log_prob"]
                # compute kl loss
                kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                metrics["actor/kl_loss"] = kl_loss.detach().item()
                metrics["actor/kl_coef"] = self.config.kl_loss_coef

            stats.update(
                {
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                }
            )
        append_to_dict(metrics, stats)
        if self.config.use_dynamic_bsz and not forward_only:
            policy_loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
        return policy_loss, [metrics, ret_entropy]

    def forward_step(batch_iter, model):
        batch = next(batch_iter)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        from verl.models.mcore import get_mcore_forward_fn

        forward_fn = get_mcore_forward_fn(self.hf_config)

        if not self.config.use_packed_seq:
            output = forward_fn(model, input_ids, attention_mask, position_ids, pack_seqs=self.config.use_packed_seq,
                sequence_parallel=self.tf_config.sequence_parallel)
        else:
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone().detach()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone().detach()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            def logits_processor(logits, label, label_mask):
                assert logits.shape[:2] == label.shape[:2]
                assert label.shape == label_mask.shape

                ret = {}

                if calculate_entropy:
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                log_probs = log_probs.masked_fill(~label_mask, 0.0)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label, "label_mask": label_mask}

            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(model, input_ids, attention_mask, position_ids,
                sequence_parallel=self.tf_config.sequence_parallel, pack_seqs=self.config.use_packed_seq,
                logits_processor=logits_processor, logits_processor_args=logits_processor_args)
        if forward_only:
            meta_info = None
        else:
            clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
            meta_info = {
                "clip_ratio": self.config.clip_ratio,
                "entropy_coeff": self.config.entropy_coeff,
                "clip_ratio_c": clip_ratio_c,
            }
        return output, partial(loss_func, data=batch, meta_info=meta_info)

    # batch should be a list of batches inside micro-batches
    batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

    # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=batch_size * seq_len,  # no use when input_shapes was set
            micro_batch_size=1,  # no use when input_shapes was set
            forward_only=forward_only,
        )
    else:
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=batch_size * seq_len,  # in use for pp = 1
            micro_batch_size=1,  # in use for pp = 1
            forward_only=forward_only,
        )
    # loss_reduces contains the stats returned from loss_func
    return losses_reduced


@GPUMemoryLogger(role="megatron actor", logger=logger)
def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
    """Update the policy with an iterator of DataProto

    Args:
        dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
            The keys of each data batch is described in the make_minibatch_iterator.

    Returns:
        Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
        and users have to combine the output in each dp rank manually.

    """
    metrics = {}
    self.prof.start()

    # to profile the update_actor of training step2 on NPU
    profile_flag = self.iteration == 1 and os.getenv("PROFILE_UPDATE", "0") != "0"
    if profile_flag:
        npu_prof = NPUProfiler(stage_name="update_actor", warm_up=0, active=1)
        npu_prof.start()

    for data in dataloader:
        self.actor_optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.actor_module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

        calculate_entropy = self.config.entropy_coeff != 0
        assert not calculate_entropy, "We do not calculate_entropy in verl-mindspeed"
        metric_micro_batch = self.forward_backward_batch(data, calculate_entropy=calculate_entropy)
        for metric in metric_micro_batch:
            # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
            append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.
        print_memory("before optimizer step during update")
        update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
        print_memory("after optimizer step during update")
        learning_rate = self.actor_optimizer.param_groups[-1]["lr"]
        data = {"actor/grad_norm": grad_norm, "actor/lr": learning_rate}
        append_to_dict(metrics, data)

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError
        self.prof.step()
        self.iteration += 1
    # add empty cache after each compute
    self.prof.stop_and_save()
    self.prof.stop_trace()

    if profile_flag:
        npu_prof.step()
        npu_prof.stop()

    empty_cache()
    return metrics


MegatronPPOActor.__init__ = megatron_ppo_actor_init_wrapper(MegatronPPOActor.__init__)
MegatronPPOActor.compute_log_prob = compute_log_prob
MegatronPPOActor.make_minibatch_iterator = make_minibatch_iterator
MegatronPPOActor.forward_backward_batch = forward_backward_batch
MegatronPPOActor.update_policy = update_policy
