# Adapted from 
# https://github.com/volcengine/verl/blob/v0.4.0/verl/utils/model.py
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


def normalize_model_name(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    """
    Transform the model name in each model_chunk in each pp stage into the name in inference engine
    """
    from verl_patches.utils.megatron_utils import get_transformer_layer_offset

    layer_offset = get_transformer_layer_offset(pp_rank, transformer_config)

    if layer_name in name:  # belong to an intermediate layer
        split_name = name.split(".")
        # find the num next to split_name
        for i, name in enumerate(split_name):
            if name == layer_name:
                break
        layer_num_idx = i + 1
        # check the name
        assert len(split_name) >= layer_num_idx + 1, f"split_name = {split_name}"
        assert split_name[layer_num_idx].isdigit(), f"split_name = {split_name}"
        # increment layer_num_idx by layer_offset
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
        name = ".".join(split_name)  # weight name in inference_tp_model
    return name


__all__ = [
    "normalize_model_name",
]