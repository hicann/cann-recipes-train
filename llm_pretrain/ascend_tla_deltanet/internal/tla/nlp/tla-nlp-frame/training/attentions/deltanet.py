
import math
import os
import warnings
import transformers
from dataclasses import dataclass
from typing import Callable, Optional, Union


from transformers.models.llama.modeling_llama import LlamaRMSNorm

import torch

from transformers.cache_utils import Cache, EncoderDecoderCache


import sys  # 关键：必须显式导入sys模块
# 修复相对导入问题的代码（确保sys已导入）
# 计算项目根目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(
    current_file_dir,  
    '../../'  # 回退到tla-nlp-frame根目录
))
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)

# 导入delta_rule模块
try:
    from tla.ops.delta_rule import chunk_delta_rule
except ImportError as e:
    # 更友好的错误提示
    raise ImportError(
        f"\n错误：无法导入chunk_delta_rule！\n"
        f"项目根目录: {project_root}\n"
        f"请检查 {project_root}/ops/delta_rule.py 文件是否存在，且包含chunk_delta_rule函数。\n"
        f"原始错误信息: {str(e)}"
    ) from e



def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, beta=None, **kwargs):
    B, H, T, D = query.shape

    q = query.transpose(1, 2)  # [B,T,H,D]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # 单位范数
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    beta = beta.clamp(1e-4, 1 - 1e-4)
    attn_output, _ = chunk_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        scale=D ** -0.5,
        initial_state=None,
        output_final_state=False
    )
    attn_output = attn_output.to(value.dtype)

    return attn_output, None


def custom_gpt2_attention_forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward


        beta = self.beta_activation(self.b_proj(hidden_states))  # [B, T, H]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            beta=beta,
            dropout=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            **kwargs,
        )

        attn_output = self.norm(attn_output)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


def replace_attn_forward(model):

    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    import torch.nn as nn

    for module in model.modules():

        if isinstance(module, GPT2Attention):

            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = module.head_dim

            # 新增参数
            module.b_proj = nn.Linear(embed_dim, num_heads, bias=False)

            module.beta_activation = nn.Sigmoid()

            module.norm = LlamaRMSNorm(head_dim, eps=1e-6)

    # 替换 forward
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = custom_gpt2_attention_forward

    print("replace GPT2 attention forward with custom attention")
