# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import logging
import torch
import torch_npu
from verl.utils.device import get_torch_device

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


def empty_cache():
    """
    To use light-weight empty_cache, user should set
    `PYTORCH_NPU_ALLOC_CONF`=`expandable_segments:True`
    """
    if hasattr(get_torch_device(), 'empty_virt_addr_cache'):
        get_torch_device().empty_virt_addr_cache()
    else:
        get_torch_device().empty_cache()
