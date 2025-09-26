# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from functools import wraps

from megatron.core import parallel_state
from megatron.core.transformer import transformer_block
from megatron.core.transformer.transformer_block import get_num_layers_to_build


def get_layer_offset(pp_size, num_layer_list):
    """
    Get layer number offset for pp stage. global_layer_index = local_layer_index + layer_number_offset
    For each pp_stage, we have layer_number_offset = prefix_sum[pp_stage + 1]
    """
    prefix_sum = [0] * (pp_size + 1)
    for index, num_layers in enumerate(num_layer_list):
        prefix_sum[index + 1] = prefix_sum[index] + num_layers
    return prefix_sum


def get_num_layers_to_build_wrapper(fn):
    @wraps(fn)
    def wrapper(config):
        num_layers_to_build = fn(config)
        num_layer_list = config.num_layer_list
        if isinstance(num_layer_list, str):
            num_layer_list = num_layer_list.strip('[]')
            elements = [item.strip() for item in num_layer_list.split(',')]
            num_layer_list = [int(item) for item in elements if item]
            config.num_layer_list = num_layer_list
            config.layer_offset = get_layer_offset(config.pipeline_model_parallel_size, config.num_layer_list)
        if num_layer_list:
            pp_stage = parallel_state.get_pipeline_model_parallel_rank()
            num_layers_to_build = num_layer_list[pp_stage]
        return num_layers_to_build
    return wrapper


# apply patch
transformer_block.get_num_layers_to_build = get_num_layers_to_build_wrapper(get_num_layers_to_build)
