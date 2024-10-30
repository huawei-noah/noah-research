# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
# Modified version of GPT-2 into the PanGu-Alpha architecture.
# - The network include an additional head on top of the backbone model.
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
# ============================================================================
"""PyTorch Pangu Alpha GPT-2 model."""

import sys
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from source.gpt2.modeling_gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model,
    GPT2MLP
)

if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

import copy
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .generation_utils import CustomGenerationMixin

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "PanguAlphaConfig"
_TOKENIZER_FOR_DOC = "JIEBATokenizer"


class PanguAlphaHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = PanguAlphaAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        query_hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        prefix_lm_mask=None
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            query_hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix_lm_mask=prefix_lm_mask
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class PanguAlphaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.is_cross_attention = False
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, prefix_lm_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            if len(attention_mask.size()) == 3:  # this means causal mask is already given
                causal_mask = attention_mask[:, None, :, :].bool()  # expand across attn_heads
            else:
                # Means attention given is 2-D, so assuming one example per instance
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()

            if prefix_lm_mask is not None:
                if isinstance(prefix_lm_mask, torch.Tensor) and len(prefix_lm_mask.size()) == 3:   # prefix mask is given
                    prefix_lm_mask = prefix_lm_mask[:, None, :, :].bool()
                else:   # otherwise index up-to-which we consider as prefix is given
                    cond1 = prefix_lm_mask[:, None, None, None]  # broadcast prefix indexes
                    cond2 = torch.arange(causal_mask.size(-1)).to(causal_mask.device)[None, None, None, :]  # broadcast all indexes
                    prefix_lm_mask = torch.le(cond2, cond1)

                prefix_lm_mask = torch.where(prefix_lm_mask, True, causal_mask)  # expand mask so that it covers all the prefix
                attn_weights = torch.where(prefix_lm_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
            else:
                attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None and len(attention_mask.size()) == 4:  # we have expanded it before
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights),
                                   torch.zeros(1).to(attn_weights.device).to(attn_weights.dtype)[0],
                                   attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None, prefix_lm_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
                attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            if len(attention_mask.size()) == 3:  # this means causal mask is already given
                causal_mask = attention_mask[:, None, :, :].bool()  # expand across attn_heads
            else:
                # Means attention given is 2-D, so assuming one example per instance
                query_length, key_length = query.size(-2), key.size(-2)
                causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()

            if prefix_lm_mask is not None:
                if isinstance(prefix_lm_mask, torch.Tensor) and len(prefix_lm_mask.size()) == 3:  # prefix mask is given
                    prefix_lm_mask = prefix_lm_mask[:, None, :, :].bool()
                else:  # otherwise index up-to-which we consider as prefix is given
                    cond1 = prefix_lm_mask[:, None, None, None]  # broadcast prefix indexes
                    cond2 = torch.arange(causal_mask.size(-1)).to(causal_mask.device)[None, None, None,
                            :]  # broadcast all indexes
                    prefix_lm_mask = torch.le(cond2, cond1)

                prefix_lm_mask = torch.where(prefix_lm_mask, True,
                                             causal_mask)  # expand mask so that it covers all the prefix
                attn_weights = torch.where(prefix_lm_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))
            else:
                attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None and len(attention_mask.size()) == 4:  # we have expanded it before
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.where(torch.isnan(attn_weights),
                                   torch.zeros(1).to(attn_weights.device).to(attn_weights.dtype)[0],
                                   attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        query_hidden_states,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        prefix_lm_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        query = self.q_attn(query_hidden_states)
        key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask, prefix_lm_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, prefix_lm_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)



@add_start_docstrings(
    """
    The PanguAlpha Model transformer with a query layer, which resembles the transformer layer, except that an 
    additional embedding indicating the next position is used as the query vector in the attention mechanism
    """,
)
class PanguAlphaModel(GPT2PreTrainedModel, CustomGenerationMixin):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight",
                                       r"code_tokens_mask", r"doc_tokens_mask"]

    def __init__(self, config, args=None, tokenizer=None):
        super().__init__(config, args, tokenizer)

        self.transformer = GPT2Model(config)
        self.top_query_embedding = nn.Embedding(config.n_positions, config.n_embd)
        self.top_query_layer = PanguAlphaHead(config)

        self.args = args
        self.tokenizer = tokenizer
        self.config = config

        code_tokens_mask = torch.zeros((len(tokenizer),))
        doc_tokens_mask = torch.zeros((len(tokenizer),))

        if self.args.replicated_tokens_map:
            for tok_id in range(0, len(tokenizer)):
                if tok_id in self.args.replicated_tokens_map:
                    code_tokens_mask[self.args.replicated_tokens_map[tok_id]] = 1
                    doc_tokens_mask[tok_id] = 1
                # if not replicated and also not new_id
                elif (tok_id not in self.args.replicated_tokens_map.keys()) and \
                        (tok_id not in self.args.replicated_tokens_map.values()):
                    code_tokens_mask[tok_id] = 1
                    doc_tokens_mask[tok_id] = 1

        self.register_buffer('code_tokens_mask', code_tokens_mask)
        self.register_buffer('doc_tokens_mask', doc_tokens_mask)

        # self.check_code = copy.deepcopy(code_tokens_mask)
        # self.check_docstring = copy.deepcopy(doc_tokens_mask)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        # --- HACK BEGIN --- #
        prefix_lm_mask = kwargs.get("prefix_lm_mask", None)
        # --- HACK END --- #
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "prefix_lm_mask": prefix_lm_mask
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        docstr_mask=None,  # extra
        prefix_lm_mask=None  # extra
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to
            `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        tr_h = len(self.transformer.h)

        if past_key_values is not None:
            past_key_values_tr = past_key_values[:tr_h]
        else:
            past_key_values_tr = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values_tr,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefix_lm_mask=prefix_lm_mask
        )
        hidden_states = transformer_outputs.last_hidden_state

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_len)
        if position_ids is not None:
            position_ids = position_ids.view(-1, seq_len)

        if past_key_values is not None:
            past_key_values_hi = past_key_values[tr_h]
        else:
            past_key_values_hi = None

        if past_key_values_hi is None:
            past_length = 0
        else:
            past_length = past_key_values_hi[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, seq_len + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        # GPT2Attention mask.
        if attention_mask is not None and len(attention_mask.size()) == 2:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Get head mask
        head_mask = None

        # Get the query embedding
        top_query_hidden_states = self.top_query_embedding(position_ids)

        # Get output of the new layer
        hidden_states = self.top_query_layer(
            hidden_states,
            top_query_hidden_states,
            layer_past=past_key_values_hi,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            prefix_lm_mask=prefix_lm_mask
        )
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.top_query_embedding.weight.device)

        # Get logits (tied weights with embedding layer)
        lm_logits = F.linear(hidden_states[0], self.transformer.wte.weight)  # (1, vocab_size, dim)

        loss = None
        if labels is not None:
            if self.args.replicated_tokens_map:

                labels = labels.contiguous()
                docstr_mask = docstr_mask.bool().contiguous()

                # get logits and put non-valid tokens to -infinity so that softmax gives 0! :)
                lm_logits_docstr = lm_logits.masked_fill(~self.doc_tokens_mask[None, None, :].bool(), float('-inf'))
                lm_logits_code = lm_logits.masked_fill(~self.code_tokens_mask[None, None, :].bool(), float('-inf'))

                # get labels for each sequence
                labels_docstr = torch.where(docstr_mask, labels, -100)
                labels_code = torch.where(~docstr_mask, labels, -100)

                # Shift so that tokens < n predict n
                lm_logits_docstr = lm_logits_docstr[..., :-1, :].contiguous()
                labels_docstr = labels_docstr[..., 1:].contiguous()

                lm_logits_code = lm_logits_code[..., :-1, :].contiguous()
                labels_code = labels_code[..., 1:].contiguous()

                docstr_mask = docstr_mask[:, 1:]

                # Just to be safe
                loss_fct_code, loss_fct_docstr = CrossEntropyLoss(reduction='none'), CrossEntropyLoss(reduction='none')

                loss_code = loss_fct_code(
                    lm_logits_code.view(-1, lm_logits_code.size(-1)), labels_code.view(-1)
                ).view(labels_code.size())

                loss_docstr = loss_fct_docstr(
                    lm_logits_docstr.view(-1, lm_logits_docstr.size(-1)), labels_docstr.view(-1)
                ).view(labels_docstr.size())

                # Merge losses together
                loss = torch.where(docstr_mask, loss_docstr, loss_code)

                # mean across all elements
                nonzero_elements_docstr = torch.count_nonzero(labels_docstr != -100)
                nonzero_elements_code = torch.count_nonzero(labels_code != -100)
                nonzero_elements = nonzero_elements_docstr + nonzero_elements_code
                loss = loss.sum().div(nonzero_elements)

            else:
                shift_lm_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_lm_logits.view(-1, shift_lm_logits.size(-1)), shift_labels.view(-1))

        # empty labels -> we need this for generation
        else:

            if self.args.replicated_tokens_map:
                lm_logits.masked_fill_(~self.code_tokens_mask[None, None, :].bool(), float('-inf'))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # hidden_states, present, (attentions, cross_attentions)
        presents = transformer_outputs.past_key_values + (hidden_states[1],)
        all_hidden_states = (hidden_states[0],) if transformer_outputs.hidden_states is None else transformer_outputs.hidden_states + (hidden_states[0],)
        all_attentions = transformer_outputs.attentions if len(hidden_states) == 2 else transformer_outputs.attentions + (hidden_states[2][0],)
        all_cross_attentions = transformer_outputs.cross_attentions if len(hidden_states) == 2 else transformer_outputs.cross_attentions + (hidden_states[2][1],)

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if
        [`~PreTrainedModel.beam_search`] or [`~PreTrainedModel.beam_sample`] is
        called. This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

