# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen3-Next model."""

from functools import lru_cache
from typing import Optional, Union
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from transformers.cache_utils import Cache, StaticCache
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextForCausalLM,
    Qwen3NextDynamicCache,
    Qwen3NextModel,
    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRMSNorm,
    Qwen3NextMLP,
    Qwen3NextRotaryEmbedding,
    apply_rotary_pos_emb,
    load_balancing_loss_func,
    apply_mask_to_padding_states,
    l2norm,
)
from transformers.utils import logging, TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack

from ....distributed import parallel_state
from ....distributed.tensorparallel import _all_reduce
from .... import distributed
from ....distributed import parallel_state
from ....distributed.strategy import DistributedStrategy, NoOpStrategy
from ....distributed.tensorparallel import (
    reduce_from_tensor_model_parallel_region,
)
from ....distributed.tp import TPModule
from ....utils import warn0
from ...modeling_attn_mask_utils import (
    _gaudi_prepare_4d_causal_attention_mask,
)
from ...modeling_rope_utils import GaudiRotaryEmbedding
from ..modeling_all_models import KVCache, Matmul, apply_customized_rope_module


try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE  # noqa

    has_fused_rope = True
except ImportError:
    has_fused_rope = False
    print("Not using HPU fused kernel for apply_rotary_pos_emb")

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm as FusedRMSNorm

    has_fused_rms_norm = True
except ImportError:
    has_fused_rms_norm = False
    print("Not using HPU fused kernel for RMSNorm")

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore


logger = logging.get_logger(__name__)


def apply_customized_rope(q, k, cos, sin, position_ids, training=True):
    if q.device.type == "hpu" and has_fused_rope:
        return apply_customized_rope_module(q, k, cos, sin, position_ids, training)
    else:
        # keep the same implementation as Transformers v4.37.2
        return apply_rotary_pos_emb(q, k, cos[position_ids], sin[position_ids])

class GaudiQwen3NextMLP(Qwen3NextMLP):
    def pre_mlp_forward(self, x):
        input = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        output = self.down_proj(input)
        return output

    def mlp_all_reduce(self, x):
        if hasattr(self.down_proj, "all_reduce"):
            self.down_proj.all_reduce(x)

    def post_mlp_forward(self, x):
        if hasattr(self.down_proj, "post_all_reduce"):
            return self.down_proj.post_all_reduce(x)
        return x

def gaudi_qwen3next_rmsnorm_forward(self, hidden_states):
    if hidden_states.device.type == "hpu" and has_fused_rms_norm:
        # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
        if hidden_states.dtype != self.weight.dtype:
            orig_dtype = hidden_states.dtype
            hidden_states = FusedRMSNorm.apply(hidden_states.to(self.weight.dtype), self.weight, self.eps)
            return hidden_states.to(orig_dtype)
        else:
            hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.eps)
            return hidden_states
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        output = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(input_dtype)
        #variance = hidden_states.pow(2).mean(-1, keepdim=True)
        #hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        #return self.weight * hidden_states.to(input_dtype)

def gaudi_qwen3next_rmsnorm_gated_forward(self, hidden_states, gate=None):
    input_dtype = hidden_states.dtype
    if hidden_states.device.type == "hpu" and has_fused_rms_norm:
        # mixed dtypes are not good for FusedRMSNorm, both inputs need to have same dtype
        if hidden_states.dtype != self.weight.dtype:
            #orig_dtype = hidden_states.dtype
            hidden_states = FusedRMSNorm.apply(hidden_states.to(self.weight.dtype), self.weight, self.variance_epsilon)
            #return hidden_states.to(orig_dtype)
        else:
            hidden_states = FusedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
            #return hidden_states
    else:
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        hidden_states = hidden_states * (1.0 + self.weight.float())
    
    hidden_states = hidden_states * F.silu(gate.to(torch.float32))
    return hidden_states.to(input_dtype)

        #return output.type_as(input_dtype)

def gaudi_qwen3next_repeat_kv(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    n_rep: int,
):
    batch, num_key_value_heads, kv_len, head_dim = key_states.shape
    if n_rep == 1 or num_key_value_heads == 1:
        return query_states, key_states, value_states, attention_mask

    new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
    key_states = key_states.reshape(new_kv_shape)
    value_states = value_states.reshape(new_kv_shape)

    batch, _, q_len, head_dim = query_states.shape
    new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
    query_states = query_states.reshape(new_q_shape)

    if attention_mask is not None:
        # Add groups dim and set to 1
        attention_mask = attention_mask.unsqueeze(1)

    return query_states, key_states, value_states, attention_mask


# FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA, scale, attention_dropout, enable_recompute, flash_attention_fp8):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.enable_recompute = enable_recompute
        self.flash_attention_fp8 = flash_attention_fp8

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_casual,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        return self._hpu_kernel_fsdpa.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_casual,
            scale,
            softmax_mode,
            recompute_mode,
            valid_sequence_lengths,
            padding_side,
        )

def gaudi_apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    #if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:# why need bs >1??
    if attention_mask is not None and attention_mask.shape[1] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states

def gaudi_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    attn_softmax_bf16: bool = False,
    **kwargs,
):
    bsz, q_len = kwargs["input_shape"]
    query_states, key_states, value_states, attention_mask = gaudi_qwen3next_repeat_kv(
        query, key, value, attention_mask, module.num_key_value_groups
    )

    query_states = query_states * scaling
    attn_weights = module.matmul_qk(query_states, key_states.transpose(-2, -1)).float()
    htcore.mark_step()
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if attn_softmax_bf16:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=query_states.dtype)
    else:
        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = module.matmul_av(attn_weights, value_states)
    attn_output = attn_output.reshape(bsz, -1, q_len, module.head_dim)

    return attn_output, attn_weights

def gaudi_tpc_torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(torch.float32)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    #out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1).float(), bias, padding=0, groups=hidden_size)
    #optimize conv1d with tpc
    #out = (conv_state[:, 1:, :] * weight[0:-1, :].unsqueeze(0)).sum(dim=1) + hidden_states.squeeze(1) * weight[-1, :].unsqueeze(0)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out

def gaudi_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # todo: to be optimized
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    #todo: to be optimized
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        #torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        torch.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=torch.float32).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def gaudi_torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        #torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        torch.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=torch.float32).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    # todo: to be optimized
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

class GaudiQwen3NextGatedDeltaNet(Qwen3NextGatedDeltaNet):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.causal_conv1d_update = gaudi_tpc_torch_causal_conv1d_update
        self.chunk_gated_delta_rule = gaudi_torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = gaudi_torch_recurrent_gated_delta_rule
        #self.conv1d = tpc_conv1d_v2()
        self.conv_state = None
        self.recurrent_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Qwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = gaudi_apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            #cache_params is not None
            #and cache_params.has_previous_state
            use_cache
            and self.conv_state is not None
            and seq_len == 1
            #and cache_position is not None   #ignore cache_position for HPU
        )

        # getting projected states from cache if it exists
        #if cache_params is not None:
        if use_cache and self.conv_state is not None:
            #conv_state = cache_params.conv_states[self.layer_idx]
            #recurrent_state = cache_params.recurrent_states[self.layer_idx]
            conv_state = self.conv_state
            recurrent_state = self.recurrent_state

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if use_cache:
                if attention_mask is not None:
                    ori_len = attention_mask.sum(dim=1).max().item()   #TODO: consider bs>1 case for different seq lens
                    pad_len = seq_len - ori_len
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - ori_len, -pad_len)).float()
                else:
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                self.conv_state = conv_state
            #if cache_params is not None:
            #    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            #    cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        # Update cache
        #if cache_params is not None:
        #    cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        if use_cache:
            self.recurrent_state = last_recurrent_state

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output, None

    def attention_all_reduce(self, attn_output):
        if hasattr(self.out_proj, "all_reduce"):
            self.out_proj.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.out_proj, "post_all_reduce"):
            return self.out_proj.post_all_reduce(attn_output)
        return attn_output
   
class GaudiDistributedAttention(torch.nn.Module):
    def __init__(
        self, hpu_module_fsdpa: ModuleFusedSDPA, scale, attention_dropout, enable_recompute, flash_attention_fp8
    ):
        super().__init__()
        self._hpu_module_fsdpa = hpu_module_fsdpa
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            from deepspeed.sequence.layer import DistributedAttention

            self._hpu_module_fsdpa_distributed = DistributedAttention(
                self._hpu_module_fsdpa, parallel_state.get_sequence_parallel_group(), 1, 2
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        dropout_p: float,
        is_casual,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
    ):
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            return self._hpu_module_fsdpa_distributed(
                query,
                key,
                value,
                0,  # As the shape for inputs is [B, N, S, H]
                None,
                attn_mask,
                dropout_p,
                is_casual,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )
        else:
            return self._hpu_module_fsdpa(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_casual,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )


def get_gaudi_distributed_attention(
    fused_scaled_dot_product_attention, fused_scaled_dot_product_attention_distributed
):
    if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
        return fused_scaled_dot_product_attention_distributed
    else:
        return fused_scaled_dot_product_attention

class GaudiQwen3NextAttention(Qwen3NextAttention):
    def __init__(self, config: Qwen3NextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.matmul_qk = Matmul()
        self.matmul_av = Matmul()
        self.k_cache = KVCache()
        self.v_cache = KVCache()

        self.max_position_embeddings = config.max_position_embeddings
        self.inp_seq_len = -1

        self.rotary_emb = GaudiRotaryEmbedding(config=self.config)

        self.fused_scaled_dot_product_attention = (
            ModuleFusedSDPA(
                FusedSDPA,
                scale=self.scaling,
                attention_dropout=self.attention_dropout,
                enable_recompute=False,
                flash_attention_fp8=getattr(config, "flash_attention_fp8", False),
            )
            if FusedSDPA
            else None
        )
        self.fused_scaled_dot_product_attention_distributed = None
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_world_size() > 1:
            self.fused_scaled_dot_product_attention_distributed = (
                GaudiDistributedAttention(
                    self.fused_scaled_dot_product_attention,
                scale=self.scaling,
                attention_dropout=self.attention_dropout,
                enable_recompute=False,
                flash_attention_fp8=getattr(config, "flash_attention_fp8", False),
            )
            if FusedSDPA
            else None
        )

        self.num_key_value_heads = config.num_key_value_heads

    def get_k_proj_weight(self):
        """4bit quantization in GPTQ replaces the k_proj.weight with qweight."""
        if hasattr(self.k_proj, "qweight"):
            return self.k_proj.qweight
        return self.k_proj.weight

    def get_k_proj_weight_dtype(self):
        """4bit quantization in GPTQ replaces the k_proj.weight with qweight.
        Scales tensor gets the weight dtype."""
        if hasattr(self.k_proj, "qweight"):
            return self.k_proj.scales.dtype
        return self.k_proj.weight.dtype

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        cache_shape = (batch_size, self.num_key_value_heads, max_seq_len, self.head_dim)
        device = self.get_k_proj_weight().device
        dtype = self.config.torch_dtype
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def update_sincos_cache(self, seq_len):
        # Call rotary emb forward() to update cos/sin cache when infering more than self.max_position_embeddings
        # This helps in avoiding creation of these caches during actual model forward pass and
        # reduce memory consumption and improve performance.
        if seq_len > self.max_position_embeddings:
            self.max_position_embeddings = seq_len
            _, _ = self.rotary_emb(self.get_k_proj_weight(), seq_len=seq_len)

    def reorder(self, tensor, beam_idx, dim_a, dim_b):
        updated = tensor.index_select(0, beam_idx)
        tensor.copy_(updated)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        if self.k_cache.cache is None:
            return (None, None)

        head_dim = self.k_cache.cache.size(-1)
        seq_length = self.k_cache.cache.size(-2)
        self.reorder(self.k_cache.cache, beam_idx, seq_length, head_dim)
        self.reorder(self.v_cache.cache, beam_idx, seq_length, head_dim)
        return (self.k_cache.cache.shape, self.v_cache.cache.shape)

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        cache_idx: int = None,
        num_virtual_tokens: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        The only differences are:
        - add new args token_idx
        - optimize KV cache
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new arg flash_attention_causal_mask
        - add new arg flash_attention_fast_softmax
        - add new arg num_virtual_tokens
        """
        input_shape = hidden_states.shape[:-1]
        q_len = input_shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if token_idx is None:
                if hasattr(past_key_values, "get_usable_length"):
                    kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)
                else:
                    kv_seq_len += past_key_values[0].shape[-2]
            else:
                if reuse_cache and not isinstance(past_key_values[0], torch.Tensor):
                    kv_seq_len = past_key_values[0][-2]
                else:
                    if num_virtual_tokens is not None and num_virtual_tokens == past_key_values[0].shape[-2]:
                        kv_seq_len = past_key_values[0].shape[-2] + kv_seq_len
                    else:
                        kv_seq_len = past_key_values[0].shape[-2]

        seq_len = kv_seq_len
        if parallel_state.sequence_parallel_is_initialized():
            seq_len = kv_seq_len * parallel_state.get_sequence_parallel_world_size()

        cos, sin = position_embeddings
        # If sequence parallel in enabled, position_ids should be based on which part of the sequence is present in the rank
        # As we divide the inputs based on ranks, position_ids are generated to suit that part of the sequence
        if parallel_state.sequence_parallel_is_initialized() and parallel_state.get_sequence_parallel_rank() > 0:
            position_ids = torch.arange(
                kv_seq_len * parallel_state.get_sequence_parallel_rank(),
                kv_seq_len * (parallel_state.get_sequence_parallel_rank() + 1),
                dtype=torch.long,
                device=query_states.device,
            )
            position_ids = position_ids.unsqueeze(0)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if use_cache:
            # reuse k, v, self_attention
            if reuse_cache:
                if past_key_values is not None and isinstance(past_key_values[0], torch.Tensor):
                    # prefix tuning case. attach past_key_values to generate first token.
                    key_states = torch.cat((past_key_values[0], key_states), -2)
                    value_states = torch.cat((past_key_values[1], value_states), -2)
                key_states = self.k_cache(key_states, 2, token_idx)
                value_states = self.v_cache(value_states, 2, token_idx)
                past_key_values = (self.k_cache.get_shape(), self.v_cache.get_shape())
            else:
                if past_key_values is None:
                    past_key = torch.zeros(
                        key_states.shape, dtype=self.get_k_proj_weight_dtype(), device=key_states.device
                    )
                    past_value = torch.zeros(
                        key_states.shape, dtype=self.get_k_proj_weight_dtype(), device=key_states.device
                    )
                    # Return list instead of tuple
                    past_key_values = [past_key, past_value]
                if (
                    token_idx is not None
                    and num_virtual_tokens is not None
                    and num_virtual_tokens == past_key_values[0].shape[-2]
                ):
                    # prefix tuning case. attach past_key_values to generate first token.
                    key_states = torch.cat((past_key_values[0], key_states), -2)
                    value_states = torch.cat((past_key_values[1], value_states), -2)
                    past_key_values = (key_states, value_states)
                else:
                    key_states = self.k_cache.update(past_key_values[0], key_states, 2, token_idx, self.inp_seq_len)
                    value_states = self.v_cache.update(past_key_values[1], value_states, 2, token_idx, self.inp_seq_len)

                if token_idx is None:
                    past_key_values = (key_states, value_states)

            if cache_idx is not None and q_len == 1:
                key_states = key_states[:, :, :cache_idx, :]
                value_states = value_states[:, :, :cache_idx, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, :cache_idx]
                kv_seq_len = key_states.shape[-2]
        else:
            past_key_values = None
        fused_scaled_dot_product_attention = get_gaudi_distributed_attention(
            self.fused_scaled_dot_product_attention, self.fused_scaled_dot_product_attention_distributed
        )
        if use_flash_attention and FusedSDPA is not None:
            attn_weights = None
            if q_len == 1:
                # next token
                attn_output = fused_scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    0.0,
                    False,
                    None,
                    "None",
                    False,
                    None,
                    "None",
                )
            else:
                # first token
                softmax_mode = "fast" if flash_attention_fast_softmax else "None"
                if flash_attention_causal_mask:
                    attn_output = fused_scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        None,
                        0.0,
                        True,
                        None,
                        softmax_mode,
                        flash_attention_recompute,
                        valid_sequence_lengths,
                        "left",
                    )
                else:
                    attn_output = fused_scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        0.0,
                        False,
                        None,
                        softmax_mode,
                        flash_attention_recompute,
                        None,
                        "None",
                    )

        else:
            attn_output, attn_weights = gaudi_eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                #sliding_window=self.sliding_window,  # main diff with Llama
                attn_softmax_bf16=attn_softmax_bf16,
                input_shape=input_shape,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)

        if not reuse_cache and token_idx is not None and cache_idx is not None and q_len == 1:
            # Return only past key value shapes and not the tensors during decode phase (q len is 1)
            # to avoid making past key values as persistent output tensors of HPU graphs.
            past_key_values = (past_key_values[0].shape, past_key_values[1].shape)

        return attn_output, attn_weights, past_key_values

    def attention_all_reduce(self, attn_output):
        if hasattr(self.o_proj, "all_reduce"):
            self.o_proj.all_reduce(attn_output)

    def post_attn_forward(self, attn_output):
        if hasattr(self.o_proj, "post_all_reduce"):
            return self.o_proj.post_all_reduce(attn_output)
        return attn_output

@lru_cache(None)
def _is_deepspeed_initialized(self) -> bool:
    if not is_deepspeed_available():
        return False
    from deepspeed import comm as dist
    from deepspeed.module_inject.layers import LinearAllreduce

    return dist.is_initialized() and any(isinstance(m, LinearAllreduce) for _, m in self.named_modules())

class GaudiQwen3NextSparseMoeBlock(Qwen3NextSparseMoeBlock):

    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.ep_size = config.ep_size if hasattr(config, "ep_size") else 1
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        if self.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            ep_rank = dist.get_rank()
            experts_per_rank = self.num_experts // self.ep_size

            self.experts_min = experts_per_rank * ep_rank
            self.experts_max = experts_per_rank * (ep_rank + 1) - 1
            self.experts_range = range(self.experts_min, self.experts_max + 1)

            self.experts = nn.ModuleList(
                [
                    (Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size) if i in self.experts_range else None)
                    for i in range(self.num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList([Qwen3NextMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)])
            self.experts_min = 0
            self.experts_max = self.num_experts - 1
            self.experts_range = range(self.experts_min, self.experts_max + 1)

        self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        - optimize expert forward, remove dynamic control and dynamic shape
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        w1_list = [self.experts[i].gate_proj.weight.squeeze() for i in self.experts_range]
        w2_list = [self.experts[i].down_proj.weight.squeeze() for i in self.experts_range]
        w3_list = [self.experts[i].up_proj.weight.squeeze() for i in self.experts_range]

        final_hidden_states = torch.ops.hpu.mixture_of_experts(
            hidden_states=hidden_states,
            expert_routing_table=selected_experts,
            router_weights=routing_weights,
            w1=w1_list,
            w2=w3_list,
            w3=w2_list,
            permuted_weights=True,
            activation="silu",
            experts_min=self.experts_min,
            experts_max=self.experts_max,
        )
        htcore.mark_step()

        if self.ep_size > 1:
            final_hidden_states = _all_reduce(final_hidden_states)
        elif not self.training and _is_deepspeed_initialized(self):
            from deepspeed import comm as dist

            dist.all_reduce(final_hidden_states, op=dist.ReduceOp.SUM)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(-1, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
"""
class TPGaudiQwen3NextAttention(GaudiQwen3NextAttention, TPModule):
    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
    ):
        super().__init__(config, layer_idx)

        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)
        assert config.num_attention_heads % world_size == 0, "The number of heads must be divisible by world size"
        self.config = copy.deepcopy(config)

        self.pre_tp_kvheads = config.num_key_value_heads
        GaudiQwen3NextAttention.__init__(self, self.config, layer_idx)
        self.config.num_attention_heads = self.config.num_attention_heads // world_size
        self.config.num_key_value_heads = (
            (self.config.num_key_value_heads // world_size)
            if self.config.num_key_value_heads > 1
            else self.config.num_key_value_heads
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = self.config.hidden_size // world_size
        self.num_heads = self.config.num_attention_heads

        self.q_proj = torch.nn.Linear(
            config.hidden_size, self.config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = torch.nn.Linear(
            self.config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> list[str]:
        colwise_weights = ["q_proj"]
        if self.pre_tp_kvheads != 1:
            colwise_weights.append("k_proj")
            colwise_weights.append("v_proj")
        return colwise_weights

    def rowwise_param_names(self) -> list[str]:
        return ["o_proj"]

    @staticmethod
    def import_module(mha: GaudiQwen3NextAttention, layer_idx, group: ProcessGroup) -> "TPGaudiQwen3NextAttention":
        tp_mha = TPGaudiQwen3NextAttention(config=mha.config, layer_idx=layer_idx, group=group)
        return tp_mha

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flex_attention: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        cache_idx: int = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        hidden_states, attn_weights, present_key_value = GaudiQwen3NextAttention.pre_attn_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flex_attention=use_flex_attention,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            cache_idx=cache_idx,
            **kwargs,
        )

        hidden_states = reduce_from_tensor_model_parallel_region(hidden_states)
        return hidden_states, attn_weights, present_key_value
"""
class GaudiQwen3NextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super(Qwen3NextDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        # token mixer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GaudiQwen3NextGatedDeltaNet(config, layer_idx)
            #just test
            #self.linear_attn = GaudiQwen3NextAttention(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = GaudiQwen3NextAttention(config, layer_idx)
            #just test
            #self.linear_attn = GaudiQwen3NextGatedDeltaNet(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = GaudiQwen3NextSparseMoeBlock(config)
        else:
            self.mlp = GaudiQwen3NextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        if self.layer_type == "full_attention":
            self.self_attn.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        if self.layer_type == "full_attention":
            return self.self_attn.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        if self.layer_type == "full_attention":
            self.self_attn.update_sincos_cache(seq_len)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        cache_idx: int = None,
        num_virtual_tokens: int = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warn0(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`",
                state=self.accelerator.state,
            )
        residual = hidden_states
        hidden_states, _, present_key_values = self.pre_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            valid_sequence_lengths=valid_sequence_lengths,
            cache_idx=cache_idx,
            num_virtual_tokens=num_virtual_tokens,
            **kwargs,
        )
        """
        hidden_states = self.input_layernorm(hidden_states)

         # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states, present_key_value = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                token_idx=token_idx,
                attn_softmax_bf16=attn_softmax_bf16,
                reuse_cache=reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                valid_sequence_lengths=valid_sequence_lengths,
                cache_idx=cache_idx,
                num_virtual_tokens=num_virtual_tokens,
                **kwargs,
            )
        """
        if self.layer_type == "linear_attention":
            self.linear_attn.attention_all_reduce(hidden_states)
        else:
            self.self_attn.attention_all_reduce(hidden_states)
            #just test
            #self.linear_attn.attention_all_reduce(hidden_states)
        
        """
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        """
        hidden_states, residual = self.post_attn_pre_mlp(hidden_states, residual)
        hidden_states = self.post_mlp(hidden_states, residual)
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_values,)

        return outputs


    def pre_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        cache_idx: int = None,
        num_virtual_tokens: int = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = self.input_layernorm(hidden_states)

         # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states, present_key_values = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
            )
            """
            #just test
            hidden_states, _, present_key_values = self.linear_attn.pre_attn_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                token_idx=token_idx,
                attn_softmax_bf16=attn_softmax_bf16,
                reuse_cache=reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                valid_sequence_lengths=valid_sequence_lengths,
                cache_idx=cache_idx,
                num_virtual_tokens=num_virtual_tokens,
                **kwargs,
            )
            """
        elif self.layer_type == "full_attention":
            # Self Attention
            """
            #just test
            hidden_states, present_key_values = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
            )
            """
            hidden_states, _, present_key_values = self.self_attn.pre_attn_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                token_idx=token_idx,
                attn_softmax_bf16=attn_softmax_bf16,
                reuse_cache=reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                valid_sequence_lengths=valid_sequence_lengths,
                cache_idx=cache_idx,
                num_virtual_tokens=num_virtual_tokens,
                **kwargs,
            )

        return hidden_states, None , present_key_values

    def post_attn_pre_mlp(self, hidden_states, residual):
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn.post_attn_forward(hidden_states)
        else:
            hidden_states = self.self_attn.post_attn_forward(hidden_states)
            #just test
            #hidden_states = self.linear_attn.post_attn_forward(hidden_states)
        if self.training:
            hidden_states = hidden_states + residual
            residual = hidden_states
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        #hidden_states = residual + hidden_states  #do in post_mlp
        return hidden_states, residual

    def post_mlp(self, hidden_states, residual):
        if self.training:
            hidden_states = hidden_states + residual
        else:
            residual.add_(hidden_states)
            hidden_states = residual

        return hidden_states

class GaudiQwen3NextModel(Qwen3NextModel):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [GaudiQwen3NextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.rotary_emb = GaudiRotaryEmbedding(config=config)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        """
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layer = GaudiQwen3NextDecoderLayer(config, layer_idx)
            if hasattr(config, "parallel_strategy") and config.parallel_strategy is not None:
                layer = config.parallel_strategy.distribute_layer(layer, layer_idx)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        # parallel_strategy is not JSON serializable
        config.parallel_strategy = None
        """

    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        for layer in self.layers:
            layer.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return tuple(layer.reorder_kv_cache(beam_idx) for layer in self.layers)

    def update_sincos_cache(self, seq_len):
        for layer in self.layers:
            layer.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        attn_softmax_bf16: Optional[bool] = False,
        reuse_cache: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: torch.Tensor = None,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
        num_virtual_tokens: int = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        """
        Copied from Qwen3NextModel.forward: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        The only differences are:
        - add new args token_idx
        - add new args attn_softmax_bf16
        - add new args reuse_cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new arg flash_attention_causal_mask
        - add new arg flash_attention_fast_softmax
        - add new arg lazy_mode
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ignore_cache_position = True  # Ignoring cache position for HPU
        # use_new_cache = False  # Ignoring new Cache path for HPU

        past_seen_tokens = 0

        if past_key_values is not None and use_cache:  # kept for BC (cache positions)
            if reuse_cache:
                if isinstance(past_key_values[0][0], torch.Tensor):
                    past_seen_tokens = past_key_values[0][0].shape[2]
                else:
                    past_seen_tokens = past_key_values[0][0][2]
            else:
                # HPU uses legacy cache path (use_new_cache = False)
                if past_key_values[0] is not None:  ##added for (None, None)
                    past_seen_tokens = past_key_values[0][0].shape[2]

        if ignore_cache_position is False:
            if cache_position is None:
                if isinstance(past_key_values, StaticCache):
                    raise ValueError("cache_position is a required argument when using StaticCache.")
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None and cache_position:
                position_ids = cache_position.unsqueeze(0)

        else:
            if position_ids is None:
                position_ids = torch.arange(
                    past_seen_tokens, seq_length + past_seen_tokens, dtype=torch.long, device=inputs_embeds.device
                )
                position_ids = position_ids.unsqueeze(0)
            cache_position = None

        # HPU specific mask generation
        if ignore_cache_position:
            causal_mask = _gaudi_prepare_4d_causal_attention_mask(
                attention_mask,
                input_ids.shape if input_ids is not None else (batch_size, seq_length),
                inputs_embeds,
                past_seen_tokens,
            )
        else:
            mask_function = (
                create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
            )
            causal_mask = mask_function(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_seen_tokens,
                position_ids=position_ids,
            )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, position_ids[0])

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # HPU uses legacy cache path (use_new_cache = False)
        next_decoder_cache = ()

        if lazy_mode:
            htcore.mark_step()

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=None if past_key_values is None else past_key_values[layer_idx],
                use_cache=use_cache,
                cache_position=cache_position,
                token_idx=token_idx,
                attn_softmax_bf16=attn_softmax_bf16,
                reuse_cache=reuse_cache,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_causal_mask=flash_attention_causal_mask,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                valid_sequence_lengths=valid_sequence_lengths,
                cache_idx=cache_idx,
                num_virtual_tokens=num_virtual_tokens,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )


class GaudiQwen3NextForCausalLM(Qwen3NextForCausalLM):
    """
    Inherits from Qwen3NextForCausalLM: https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/mixtral/modeling_mixtral.py#L1231
    The only differences are:
    - add new args token_idx
    - add token_idx into model_inputs
    - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
    - from step2 when enable KV cache, slice next_position_ids from position_ids base on the token_idx
    """
    """
    def __init__(self, config, parallel_strategy: DistributedStrategy = NoOpStrategy):
        config.parallel_strategy = parallel_strategy
        super().__init__(config)
    """
    def allocate_kv_cache(self, batch_size, max_seq_len, inp_seq_len):
        self.model.allocate_kv_cache(batch_size, max_seq_len, inp_seq_len)
        self.kv_cache_len = max_seq_len

    def reorder_kv_cache(self, beam_idx: torch.LongTensor):
        return self.model.reorder_kv_cache(beam_idx)

    def update_sincos_cache(self, seq_len):
        self.model.update_sincos_cache(seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        trim_logits: Optional[bool] = False,
        reuse_cache: Optional[bool] = None,
        attn_softmax_bf16: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        valid_sequence_lengths: torch.Tensor = None,
        cache_idx: int = None,
        lazy_mode: Optional[bool] = True,
        num_virtual_tokens: int = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if self.generation_config.use_fused_rope is False:
            global has_fused_rope
            has_fused_rope = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            token_idx=token_idx,
            attn_softmax_bf16=attn_softmax_bf16,
            reuse_cache=reuse_cache,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_causal_mask=flash_attention_causal_mask,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            valid_sequence_lengths=valid_sequence_lengths,
            cache_idx=cache_idx,
            lazy_mode=lazy_mode,
            num_virtual_tokens=num_virtual_tokens,
        )

        hidden_states = outputs.last_hidden_state
        _, seq_len, _ = hidden_states.shape
        if seq_len > 1 and trim_logits and not self.training:
            if token_idx is not None:
                hidden_states = hidden_states.index_select(1, token_idx - 1)
            else:
                hidden_states = hidden_states[:, -1, :]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @staticmethod
    def _reorder_cache(
        past: tuple[tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        token_idx=None,
        **kwargs,
    ):
        reuse_cache = kwargs.get("reuse_cache")
        bucket_internal = kwargs.get("bucket_internal")

        if past_key_values is not None:
            if token_idx is not None:
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)
            else:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif (
                    input_ids.shape[1] != cache_position.shape[0]
                ):  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]
        elif (reuse_cache or bucket_internal) and token_idx is not None:
            # KV cache is pre allocated with reuse cache or will be padded with bucket internal
            # hence for the 1st token we can slice the inputs till token idx for the fwd pass.
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

        # keep cache_position implementation as None for HPU
        cache_position = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "trim_logits": kwargs.get("trim_logits"),
                "attn_softmax_bf16": kwargs.get("attn_softmax_bf16"),
                "reuse_cache": reuse_cache,
                "use_flash_attention": kwargs.get("use_flash_attention"),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute"),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask"),
                "flash_attention_fast_softmax": kwargs.get("flash_attention_fast_softmax"),
                "valid_sequence_lengths": kwargs.get("valid_sequence_lengths"),
                "cache_idx": kwargs.get("cache_idx"),
                "lazy_mode": kwargs.get("lazy_mode"),
                "num_virtual_tokens": kwargs.get("num_virtual_tokens"),
            }
        )
        return model_inputs
