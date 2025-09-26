# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, List
import torch
from megatron.core import parallel_state as mpu

from verl import DataProto

from verl_patches.utils.reshard import get_dp_reshard_tensor_via_alltoall


class TensorCache:
    """Cache tensors on device to enable high-speed D2D data transfer between different stages in one training step"""

    def __init__(self, config):
        self.mini_bs = config.actor.ppo_mini_batch_size * config.rollout.n  # batch_size for one training step
        self.tensor_cached = {}                                             # dict to cache tensors on NPU

    def cache_tensors(
        self,
        data: DataProto,
        keys_to_reserve: Optional[List[str]] = None,
        keys_no_cache: Optional[List[str]] = None
    ):
        """Caches specified tensors from input data while allowing fine-grained control.

        Args:
            data (DataProto): Container holding tensors to be processed.
            keys_to_reserve (List[str], optional): Tensor names that will remain in `data` after caching.
                These tensors are cached but not removed from the input.
            keys_no_cache (List[str], optional): Tensor names explicitly excluded from caching.
                These tensors will neither be stored nor removed from `data`.
        """
        if keys_to_reserve is None:
            keys_to_reserve = []
        if keys_no_cache is None:
            keys_no_cache = []

        keys = list(data.batch.keys())
        for key in keys:
            if key in keys_no_cache or key in self.tensor_cached:
                continue
            # cache tensors
            if key in keys_to_reserve:
                value = data.batch[key]     # don't pop the tensor, it will be sent to host for some computation
            else:
                value = data.batch.pop(key) # pop the tensor which will only stay in the cache on NPU
            self.tensor_cached[key] = value

    def get_cached_tensors(self, input_data: DataProto, keys_to_get: Optional[List[str]] = None):
        """Retrieves specified tensors from the cache.

        Args:
            input_data (DataProto): Input data container, the meta_info "global_megatron_dp_ranks" is needed.
            keys_to_get (List[str], optional): List of tensor names to retrieve.
                If `None` (default), returns all tensors in the cache.

        Returns:
            DataProto: A new instance containing the requested cached tensors.
        """
        if keys_to_get is None:
            keys_to_get = self.tensor_cached.keys()

        tensor_dict = {}
        for key in keys_to_get:
            if key not in self.tensor_cached:
                raise KeyError(f"{key} is not in cached keys {self.tensor_cached.keys()}")
            cached_tensor = self.tensor_cached[key]     # get the tensor cached by previous stage
            src_shape = list(cached_tensor.shape)
            src_dp_size = self.mini_bs // src_shape[0]  # get the DP size of the previous stage
            dst_dp_size = mpu.get_data_parallel_world_size()    # DP size of this stage (megatron engine)
            dst_shape = src_shape.copy()
            dst_shape[0] = self.mini_bs // dst_dp_size  # get the tensor shape at this stage
            assert src_dp_size % dst_dp_size == 0, (
                f"src_dp_size {src_dp_size} is not divisible by dst_dp_size {dst_dp_size}, "
                f"which is not supported for now"
            )
            # tensor transfer
            if dst_dp_size == src_dp_size:
                # same shape, no need to reshard the cached tensor
                tensor_dict[key] = cached_tensor
            else:
                # reshard the cached tensor on dim0 by AllToAllV communication
                # typically, this will be done between 'generate_sequences' and 'compute_ref_log_prob'
                tensor_dict[key] = get_dp_reshard_tensor_via_alltoall(
                    cached_tensor,
                    src_dp_size,
                    dst_dp_size,
                    dst_shape,
                    input_data.meta_info["global_megatron_dp_ranks"]
                )
                self.tensor_cached[key] = tensor_dict[key]  # update cache
        # return the cached tensors on NPU
        output = DataProto.from_dict(tensors=tensor_dict)
        return output

    def clear(self):
        """Clear the cache"""
        self.tensor_cached.clear()
