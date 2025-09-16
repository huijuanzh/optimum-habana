# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Gaudi Janus model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, StaticCache
from transformers.models.janus.modeling_janus import (
    JanusVisionAttention,
    JanusVisionEncoderLayer,
    JanusVisionEncoder,
    JanusVisionModel,
    JanusCausalLMOutputWithPast,
    JanusModel,
    JanusBaseModelOutputWithPast,
    JanusForConditionalGeneration,
)
from transformers.utils import logging, TransformersKwargs, auto_docstring, can_return_tuple
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from optimum.habana.transformers.models import GaudiLlamaForCausalLM
try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

logger = logging.get_logger(__name__)


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode):
        return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode)

# reference from:https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L270
class GaudiJanusVisionAttention(JanusVisionAttention):
    def __init__(self, config):
        super().__init__(config)
        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
 
        #import pdb;pdb.set_trace()
        if FusedSDPA is not None and use_flash_attention:
            attn_output = self.fused_scaled_dot_product_attention(
                query_states.unsqueeze(0), key_states.unsqueeze(0), value_states.unsqueeze(0), attention_mask, 0.0 if not self.training else self.attention_dropout, False, None, "None"
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states.unsqueeze(0), key_states.unsqueeze(0), value_states.unsqueeze(0), attention_mask, dropout_p=0.0 if not self.training else self.attention_dropout, is_causal=self.is_causal,
            )
            #attn_output, attn_weights = F.scaled_dot_product_attention(
            #    query_states.unsqueeze(0), key_states.unsqueeze(0), value_states.unsqueeze(0), attention_mask, dropout_p=0.0 if not self.training else self.attention_dropout, scaling=self.scale, is_causal=self.is_causal,
            #)

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output, None


# reference from:https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L366
class GaudiJanusVisionEncoderLayer(JanusVisionEncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        use_flash_attention: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_flash_attention=use_flash_attention,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

#reference from:https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L415
class GaudiJanusVisionEncoder(JanusVisionEncoder):
    @can_return_tuple
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_flash_attention: Optional[bool] = False,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                use_flash_attention=use_flash_attention,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

# reference from:https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L496
class GaudiJanusVisionModel(JanusVisionModel):
    #@auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        use_flash_attention: Optional[bool] = False,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        """
        The only differences are:
            - add use_flash_attention arg
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_flash_attention=use_flash_attention,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )    

class GaudiJanusModel(JanusModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = GaudiLlamaForCausalLM(config.text_config)
    @can_return_tuple
    #@auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_flash_attention: Optional[bool] = False,
        **kwargs,  #to check ,**kwargs used,so no need to overwrite forward??
    ):
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L1075
        - add new args use_flash_attention
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values)
            image_features = image_embeds.reshape(-1, inputs_embeds.shape[-1])
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_attention_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        import pdb;pdb.set_trace()
        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            use_flash_attention=use_flash_attention,
            **kwargs,
        )

        return JanusBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )

# reference from: https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L1124
class GaudiJanusForConditionalGeneration(JanusForConditionalGeneration):
    _supports_cache_class = True
    @can_return_tuple
    #@auto_docstring
    def forward(
         self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/janus/modeling_janus.py#L1156
        The only differences are:
        - add new args token_idx
        - add new args use_flash_attention
        """
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return JanusCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- extra custom processing
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1748
        The only differences are:
        - handle new args token_idx
        - handle new args use_flash_attention
        """
        token_idx = kwargs.get("token_idx", None)
        use_flash_attention = kwargs.get("use_flash_attention", False)
        if token_idx is not None:
            if isinstance(past_key_values, StaticCache):
                if cache_position.shape[0] > 1:
                    input_ids = input_ids[:, :token_idx]
                    attention_mask = attention_mask[:, :token_idx]
                    cache_position = cache_position[:token_idx]
                else:
                    # over-write with token idx
                    cache_position[0] = token_idx - 1

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        
        #update model_inputs
        model_inputs["token_idx"] = token_idx
        model_inputs["use_flash_attention"] = use_flash_attention

        return model_inputs