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

import torch
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig


class MLATransformerConfig(TransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mla_specific_param = kwargs.get('mla_specific_param', None)


def _get_base_mla_transformer_config(hf_config: PretrainedConfig, dtype: torch.dtype, **kwargs) -> TransformerConfig:
    """
    Create a base MLA TransformerConfig with common parameters across different model architectures.

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        **kwargs: Additional parameters to override defaults

    Returns:
        TransformerConfig with common parameters
    """
    from megatron.core import parallel_state as mpu

    # Common parallel state parameters
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False

    # Base configuration with common parameters
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "num_moe_experts": hf_config.n_routed_experts,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        # The current version of megatron does not have expert_tensor_parallel_world_size
        # original code is: "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_context_parallel_world_size(),
        "overlap_p2p_comm": overlap_p2p_comm,
        "batch_p2p_comm": batch_p2p_comm,
        "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    base_config.update(kwargs)

    return TransformerConfig(**base_config)


def hf_to_mcore_config_dpskv3(hf_config: PretrainedConfig, dtype: torch.dtype) -> TransformerConfig:
    # DeepseekV3ForCausalLM
    mla_transformer_config = _get_base_mla_transformer_config(hf_config=hf_config, dtype=dtype)

    mla_params = dict(
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,

        multi_head_latent_attention=True,
        q_lora_rank=hf_config.q_lora_rank,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_head_dim=hf_config.qk_nope_head_dim,
        qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        qk_layernorm=True,
        max_position_embeddings=hf_config.max_position_embeddings,
        first_k_dense_replace=hf_config.first_k_dense_replace,

        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_layer_freq=hf_config.moe_layer_freq,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.n_routed_experts,
        moe_router_load_balancing_type=hf_config.topk_method,
        topk_group=hf_config.topk_group,

        # Currently, the megatron compatible with mindspeed does not support configuration. Here,
        # the parameter is passed to args through verl_patches/train_engine/initialize_training to actually take effect
        routed_scaling_factor=hf_config.routed_scaling_factor,
        moe_router_enable_expert_bias=True,

        moe_shared_expert_overlap=True,
        moe_grouped_gemm=True,
        moe_router_score_function=hf_config.scoring_func,
        # Other optimizations
        persist_layer_norm=False,
        seq_aux=hf_config.seq_aux,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
    )
    for k, v in mla_params.items():
        setattr(mla_transformer_config, k, v)
    return mla_transformer_config
