# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List
import torch
from megatron.core import parallel_state as mpu
from vllm.distributed.parallel_state import get_ep_group


def ep_param_reshard_by_alltoallv(
    param_name,
    ep_param_train,
    num_experts,
    weight1_key_name="mlp.experts.weight1",
    weight2_key_name="mlp.experts.weight2"
):
    """Reshard EP params by AllToAllV for better memory usage and communication performance in TP_extend_EP training

    Args:
        param_name: EP param name in the training engine
        ep_param_train: EP param shard held by this rank in the training engine
        num_experts: total number of routing experts in the complete model
        weight1_key_name: key word for the expert weight1 name in the training engine
        weight2_key_name: key word for the expert weight2 name in the training engine

    For example, Train EP4PP2 and Rollout EP8PP1, after PP allgather in veRL, the communication is like below:
    train ep ranks:     0    1    2    3  |  0    1    2    3
                        | \    \                      /    /|
                        |   \    \----\       /-----/    /  |
    rollout ep ranks:   0    1    2    3     4    5    6    7

    the send tensors for global rank 0 is: [shard_to_rank0, shard_to_rank1, empty, empty]
    the recv tensors for global rank 0 is: [shard_from_rank0, empty, empty, empty]
    the send tensors for global rank 4 is: [empty, empty, empty, empty]
    the recv tensors for global rank 4 is: [empty, empty, shard_from_rank6, empty]

    NOTE: Global ranks must be consecutive within each training EP group, which is guaranteed by TP_extend_EP.
    """
    ep_size_train = mpu.get_tensor_and_expert_parallel_world_size()
    ep_rank_train = mpu.get_tensor_and_expert_parallel_rank()
    ep_group_rollout = get_ep_group().device_group
    ep_size_rollout = torch.distributed.get_world_size(ep_group_rollout)
    ep_rank_rollout = torch.distributed.get_rank(group=ep_group_rollout)
    assert ep_size_rollout % ep_size_train == 0, "EP size of rollout must be divisible by EP size of training"
    micro_ep_size = ep_size_rollout // ep_size_train

    assert num_experts % ep_size_train == 0 and num_experts % ep_size_rollout == 0
    num_experts_train = num_experts // ep_size_train
    num_experts_rollout = num_experts // ep_size_rollout

    if weight1_key_name in param_name:
        hidden_size = ep_param_train.shape[0]
        # The actual memory layout of weight `w13` is [num_experts_train, hidden_size, moe_intermediate_size],
        # view the tensor to a correct shape before using it.
        # Also, training phase and rollout phase expect different layouts for `w13`, with inversed dimension
        # order of `hidden_size` and `moe_intermediate_size`, necessiting the `transpose` and `contiguous` here.
        ep_param_train = ep_param_train.view(num_experts_train, hidden_size, -1).transpose(1, 2).contiguous()

        split_size = num_experts_train // micro_ep_size
        rollout_weight_shape = [split_size, ep_param_train.shape[1], hidden_size]

    elif weight2_key_name in param_name:
        hidden_size = ep_param_train.shape[1]
        # Similar to the handling of `w13`.
        ep_param_train = ep_param_train.view(num_experts_train, -1, hidden_size).transpose(1, 2).contiguous()

        split_size = num_experts_train // micro_ep_size
        rollout_weight_shape = [split_size, hidden_size, ep_param_train.shape[2]]
    else:
        raise NotImplementedError(f"Weight {param_name} not supported in EP param resharding yet!")

    # for send: get the corresponding rollout ep ranks of this training ep group
    ep_train_group_idx = (
        ep_rank_rollout // ep_size_train
    )  # train ep group idx within the larger rollout ep group of this rank
    ep_rank_range_rollout = list(
        range(ep_train_group_idx * ep_size_train, ep_train_group_idx * ep_size_train + ep_size_train, 1)
    )
    # for recv: get the src rollout ep rank of this rank
    recv_src_rank = ep_rank_rollout // micro_ep_size
    send_tensors = []   # sharded ep params to send to each rank in this training ep group by this rank
    recv_tensors = []   # recv buffers for this rank to recv sharded ep params from each rank in this training ep group
    split_start_idx = 0
    for rank_ep_train in range(ep_size_train):
        # update send_tensors
        rank_ep_rollout = ep_rank_range_rollout[rank_ep_train]
        if rank_ep_rollout // micro_ep_size == ep_rank_train:
            tensor_to_send = ep_param_train[split_start_idx:split_start_idx + split_size, ...]
            send_tensors.append(tensor_to_send)
            split_start_idx += split_size
        else:
            send_tensors.append(torch.zeros(0, dtype=ep_param_train.dtype, device=ep_param_train.device)) # placeholder

        # update recv_tensors
        if recv_src_rank == rank_ep_train:
            recv_tensors.append(
                torch.empty(rollout_weight_shape, dtype=ep_param_train.dtype, device=ep_param_train.device)
            )
        else:
            recv_tensors.append(torch.empty(0, dtype=ep_param_train.dtype, device=ep_param_train.device)) # placeholder

    torch.distributed.all_to_all(recv_tensors, send_tensors, group=mpu.get_tensor_and_expert_parallel_group())
    # filter out empty tensors and retain only the ep params required by this rank in rollout
    ep_params = [param for param in recv_tensors if param.numel() > 0]
    return ep_params


def get_rollout_expert_after_resharding(infer_params, model_config, is_weight1):
    """Postprocess the resharded EP parameter to return EP parameters for all ranks.
    To optimize memory, only the params for the current rank are non-empty; others are empty tensors.

    Args:
        infer_params: a tensor list but only contains one ep param of weight1 for this rank in rollout
        model_config: hugging face model config
    """
    assert len(infer_params) == 1
    rollout_ep_group = get_ep_group().device_group
    rollout_ep_size = torch.distributed.get_world_size(rollout_ep_group)
    ep_rank_rollout = torch.distributed.get_rank(group=rollout_ep_group)
    num_experts = model_config.n_routed_experts
    num_experts_rollout = num_experts // rollout_ep_size

    # expert ids held by current rank in rollout
    local_expert_ids = list(
        range(ep_rank_rollout * num_experts_rollout, ep_rank_rollout * num_experts_rollout + num_experts_rollout)
    )

    local_expert_params = infer_params[0]
    if is_weight1:
        experts_gate_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        experts_up_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        for local_idx, expert_id in enumerate(local_expert_ids):
            experts_gate_pp[expert_id], experts_up_pp[expert_id] = torch.chunk(
                local_expert_params[local_idx], chunks=2, dim=0
            )
        infer_params = [tensor for pair in zip(experts_gate_pp, experts_up_pp) for tensor in pair]
        return infer_params
    else:
        experts_down_pp = [
            torch.empty(0, dtype=local_expert_params[0].dtype, device=local_expert_params[0].device)
            for idx in range(num_experts)
        ]
        for local_idx, expert_id in enumerate(local_expert_ids):
            experts_down_pp[expert_id] = local_expert_params[local_idx]
        return experts_down_pp


def get_dp_reshard_tensor_via_alltoall(
    src_tensor: torch.Tensor,
    src_dp_size: int,
    dst_dp_size: int,
    dst_shape: List[int],
    global_megatron_dp_ranks: List[int]
):
    """ For resharding during D2D tensor transfer between generate_sequences and compute_ref_log_prob.  """
    assert src_dp_size == torch.distributed.get_world_size(), (
        "We only support src_dp_size (generate_sequences) equals world_size for now in the cached tensor resharding."
    )
    micro_dp_size = src_dp_size // dst_dp_size
    dst_dp_rank = mpu.get_data_parallel_rank()
    # this rank receives tensors from these src_ranks
    src_ranks = set(range(dst_dp_rank * micro_dp_size, (dst_dp_rank + 1) * micro_dp_size, 1))
    # output tensor buffer for AllToAllV communication
    buffer = torch.empty(dst_shape, dtype=src_tensor.dtype, device=src_tensor.device)
    src_bs = src_tensor.shape[0]
    recv_tensors = []
    send_tensors = []

    for global_rank, dp_rank in enumerate(global_megatron_dp_ranks):
        src_dp_rank = global_rank   # src_dp_size equals world_size
        if src_dp_rank in src_ranks:
            micro_dp_idx = src_dp_rank % micro_dp_size
            recv_tensors.append(buffer[micro_dp_idx * src_bs: micro_dp_idx * src_bs + src_bs, ...])
        else:
            # this rank does not recv tensor from the src_dp_rank, add empty tensor as placeholder
            recv_tensors.append(torch.empty(0, dtype=src_tensor.dtype, device=src_tensor.device))

        rank_i_src_ranks = set(range(dp_rank * micro_dp_size, (dp_rank + 1) * micro_dp_size, 1))
        if torch.distributed.get_rank() in rank_i_src_ranks:
            send_tensors.append(src_tensor)
        else:
            # this rank as src does not send tensor to the dst dp_rank, add empty tensor as placeholder
            send_tensors.append(torch.empty(0, dtype=src_tensor.dtype, device=src_tensor.device))

    torch.distributed.all_to_all(recv_tensors, send_tensors)
    return buffer
