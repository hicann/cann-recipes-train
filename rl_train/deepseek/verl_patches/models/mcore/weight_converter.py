# Adapted from
# https://github.com/volcengine/verl/blob/main/verl/models/mcore/weight_converter.py
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

import torch
from verl.models.mcore.weight_converter import McoreToHFWeightConverterBase


class McoreToHFWeightConverterDeepseekV3(McoreToHFWeightConverterBase):
    def convert_param(self, name: str, params_one_group: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if name in direct_name_mapping:
            return [direct_name_mapping[name]], [params_one_group[0]]

        if "input_layernorm.weight" in name:
            name = name.replace("input_layernorm.weight", "self_attention.linear_qkv.layer_norm_weight")

        if "self_attention" in name:
            return self._convert_attention_param(name, params_one_group)
        elif "mlp" in name:
            return self._convert_mlp_param(name, params_one_group)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_attention_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "self_attention.linear_qkv.bias" in name or "self_attention.linear_qkv.weight" in name:
            param_type = name.split(".")[-1]
            assert param_type == "bias" or param_type == "weight"
            # with q_lora_rank
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_a_proj.{param_type}")
            convert_names.append(f"model.layers.{layer_number}.self_attn.kv_a_proj_with_mqa.{param_type}")
            assert len(params) == 2
        elif "self_attention.linear_proj.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.o_proj.weight")
            assert len(params) == 1
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            convert_names.append(f"model.layers.{layer_number}.input_layernorm.weight")
            assert len(params) == 1
        elif "self_attention.k_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.kv_a_layernorm.weight")
            assert len(params) == 1
        elif "self_attention.q_layernorm.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_a_layernorm.weight")
            assert len(params) == 1
        elif "self_attention.linear_kvb.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.kv_b_proj.weight")
            assert len(params) == 1
        elif "self_attention.linear_qb.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.self_attn.q_b_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params

    def _convert_mlp_param(self, name: str, params: list[torch.Tensor]) -> tuple[list[str], list[torch.Tensor]]:
        layer_number = name.split(".")[2]
        convert_names = []
        if "pre_mlp_layernorm" in name:
            convert_names.append(f"model.layers.{layer_number}.post_attention_layernorm.weight")
            assert len(params) == 1
        elif "mlp.router.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.weight")
            assert len(params) == 1
        elif "mlp.router.expert_bias" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate.e_score_correction_bias")
            assert len(params) == 1
        elif "shared_experts.linear_fc1.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_experts.up_proj.weight")
            assert len(params) == 2
        elif "shared_experts.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.shared_experts.down_proj.weight")
            assert len(params) == 1
        elif "mlp.experts.weight1" in name:
            num_moe_experts = int(len(params) / 2)
            for expert_id in range(num_moe_experts):
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight")
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight")
        elif "mlp.experts.weight2" in name:
            num_moe_experts = len(params)
            for expert_id in range(num_moe_experts):
                convert_names.append(f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight")
        elif "mlp.linear_fc1.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.gate_proj.weight")
            convert_names.append(f"model.layers.{layer_number}.mlp.up_proj.weight")
            assert len(params) == 2
        elif "mlp.linear_fc2.weight" in name:
            convert_names.append(f"model.layers.{layer_number}.mlp.down_proj.weight")
            assert len(params) == 1
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names, params
