# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
###############################################################################
# Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.mpt.modeling_mpt import (
    MptAttention,
    MptBlock,
    MptConfig,
    MptForCausalLM,
    MptModel,
)
from transformers.utils import logging

from ...modeling_attn_mask_utils import _gaudi_prepare_4d_causal_attention_mask


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None


logger = logging.get_logger(__name__)


class GaudiMptAttention(MptAttention):
    def __init__(self, config: MptConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        cache_idx: Optional[torch.Tensor] = None,
    ):
        """
        Copied from MptAttention.forward: https://github.com/huggingface/transformers/blob/v4.44.1/src/transformers/models/mpt/modeling_mpt.py
        The only differences are:
        - add new args token_idx
        - optimize KV cache
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new args cache_idx
        """

        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        if self.clip_qkv:
            mixed_qkv = mixed_qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                if token_idx is not None:
                    # copy kv to kv_cache
                    past_key_value[0].index_copy_(2, token_idx - 1, key_states)
                    past_key_value[1].index_copy_(2, token_idx - 1, value_states)

                    if cache_idx is not None:
                        key_states = past_key_value[0][:, :, :cache_idx, :]
                        value_states = past_key_value[1][:, :, :cache_idx, :]
                    else:
                        key_states = past_key_value[0]
                        value_states = past_key_value[1]
                else:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                    past_key_value = [key_states, value_states]
        else:
            if cache_idx is not None:
                cache_shape = (batch_size, self.n_heads, self.kv_cache_len, self.head_dim)
                past_key_value = [
                    torch.empty(cache_shape, dtype=key_states.dtype, device=key_states.device),
                    torch.empty(cache_shape, dtype=key_states.dtype, device=key_states.device),
                ]
                past_key_value[0][:, :, : key_states.shape[2], :] = key_states
                past_key_value[1][:, :, : key_states.shape[2], :] = value_states
            else:
                past_key_value = [key_states.clone(), value_states.clone()]

        query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

        if position_bias is not None:
            if len(position_bias.shape) != 3:
                raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
            key_length = key_states.shape[-2]

            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)

            position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

        if use_flash_attention and FusedSDPA:
            import habana_frameworks.torch.hpu as ht

            with ht.sdp_kernel(enable_recompute=flash_attention_recompute):
                attn_output = FusedSDPA.apply(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask * torch.finfo(query_states.dtype).min + position_bias.to(query_states.dtype),
                    0.0,
                    False,
                    None,
                )

            attn_weights = None
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

            if position_bias is not None:
                attention_scores = attention_scores + position_bias
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, torch.finfo(query_states.dtype).min)

            # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class GaudiMptBlock(MptBlock):
    def __init__(self, config: MptConfig):
        super().__init__(config)
        self.attn = GaudiMptAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        cache_idx: Optional[torch.Tensor] = None,
    ):
        """
        Copied from MptBlock.forward: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/mpt/modeling_mpt.py
        The only differences are:
        - add new args token_idx
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new args cache_idx
        """
        # hidden_states: [batch_size, seq_length, hidden_size]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.norm_1(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs, attn_weights, past_key_value = self.attn(
            layernorm_output,
            position_bias=position_bias,
            attention_mask=attention_mask,
            past_key_value=layer_past,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            cache_idx=cache_idx,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs) + residual

        layernorm_output = self.norm_2(hidden_states)

        # Get residual
        residual = hidden_states

        # MLP.
        output = self.ffn(layernorm_output, residual)
        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # hidden_states, present, attentions


class GaudiMptModel(MptModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        cache_idx: Optional[torch.Tensor] = None,
        **kwargs,  # NOOP kwargs, for now
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Copied from MptModel.forward: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/mpt/modeling_mpt.py
        The only differences are:
        - add new args token_idx
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new args cache_idx
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None and token_idx is None:  # because RW-cache, not standard format
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_mpt_alibi_tensor(self.num_heads, self.config.max_seq_len, device=hidden_states.device)

        causal_mask = _gaudi_prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        causal_mask = causal_mask.bool()

        for block, layer_past in zip(self.blocks, past_key_values):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    use_cache,
                    output_attentions,
                    None,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_bias=alibi,
                    token_idx=token_idx,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                    cache_idx=cache_idx,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GaudiMptForCausalLM(MptForCausalLM):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Inherits from MptForCausalLM: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/mpt/modeling_mpt.py
        The only differences are:
        - add new args token_idx
        - add token_idx into model_inputs
        - from step2 when enable KV cache, slice next_input_ids from input_ids base on the token_idx
        - support for internal bucketing
        """
        bucket_internal = kwargs.get("bucket_internal")
        cache_idx = kwargs.get("cache_idx")  # kv_cache index for slice operations
        use_flash_attention = kwargs.get("use_flash_attention", False)
        flash_attention_recompute = kwargs.get("flash_attention_recompute", False)
        # only last tokens for input_ids if past is not None
        if past_key_values is not None:
            if token_idx is None:
                past_length = past_key_values[0][0].shape[2]

                # Some generation methods already pass only the last input ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]
            else:
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)

                if bucket_internal and token_idx is not None:
                    attention_mask = attention_mask[:, :cache_idx]
        elif bucket_internal and token_idx is not None:
            # for the 1st token we can slice the inputs till token idx for the fwd pass.
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]
            # for the 1st token the cache_idx is None
            cache_idx = token_idx
            for block in self.transformer.blocks:
                block.attn.kv_cache_len = kwargs["kv_cache_len"]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,  # NITS should it be layer_past?
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "token_idx": token_idx,
                "use_flash_attention": use_flash_attention,
                "flash_attention_recompute": flash_attention_recompute,
                "cache_idx": cache_idx,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        cache_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """
        Inherits from MptForCausalLM: https://github.com/huggingface/transformers/blob/v4.32.0/src/transformers/models/mpt/modeling_mpt.py
        The only differences are:
        - add new args token_idx
        - add new args use_flash_attention
        - add new arg flash_attention_recompute
        - add new args cache_idx
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            cache_idx=cache_idx,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
