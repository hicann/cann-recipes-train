# Adapted from 
# https://github.com/volcengine/verl/blob/main/verl/models/mcore/model_initializer.py
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

from verl.models.mcore.model_initializer import BaseModelInitializer


def get_rope_scaling_args(self) -> dict:
    """Get rope scaling args."""
    rope_scaling_args = {}
    # Currently, the megatron compatible with mindspeed does not support rope_scaling_args,
    # the parameter is passed to args through verl_patches/train_engine/initialize_training to actually take effect
    return rope_scaling_args


class DeepseekV3Model(BaseModelInitializer):
    """Initializer DeepseekV3Model."""

    def get_transformer_layer_spec(self):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        from mindspeed_llm.tasks.models.spec.deepseek_spec import layer_spec
        return layer_spec