# Adapted from
# https://gitcode.com/ascend-mirror/MindSpeed-LLM/blob/2.1.0/mindspeed_llm/training/arguments.py
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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
