# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

import os
import logging
import torch
import torch_npu

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def print_memory(head_info: str, only_rank_0: bool = True):
    free, total = torch_npu.npu.mem_get_info()
    used = total - free  # used memory
    rank = torch.distributed.get_rank()
    if os.getenv("PRINT_MEMORY", "0") == "1" and ((not only_rank_0) or rank == 0):
        logger.info(
            f"-############ Check memory {head_info} on rank_{rank} : "
            f"used:{used/1024**3:.3f}GB, free:{free/1024**3:.3f}GB, total:{total/1024**3:.3f}GB"
        )


def insert_patch(patch_module, original_module):
    for key in patch_module.__all__:
        patch = getattr(patch_module, key)
        setattr(original_module, key, patch)
