# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/core_r0.8.0/megatron/core/tensor_parallel/cross_entropy.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from typing import Tuple

import torch


@staticmethod
def calculate_logits_max(
    vocab_parallel_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # chunk computation to reduce peak memory
    chunk_size = 1024 # adjust according to NPU memory
    vocab_dim = vocab_parallel_logits.size(-1)
    for i in range(0, vocab_dim, chunk_size):
        end_idx = min(i + chunk_size, vocab_dim)
        vocab_parallel_logits[..., i:end_idx] = vocab_parallel_logits[..., i:end_idx].float()
    # Maximum value along vocab dimension across all GPUs.
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

    return vocab_parallel_logits, logits_max
