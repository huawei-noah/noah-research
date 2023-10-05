# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaConfig
)


class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides [`RobertaConfig`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    model_type = "xlm-roberta"


class RobertaForTokenAndIntentClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_intent_labels):
        super().__init__(config, num_intent_labels)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        # Initialize weights and apply final processing
        self.post_init()
        self.num_intent_labels = num_intent_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        slot_labels=None,
        intent_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # [CLS]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)
        loss = None
        # slot_loss
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    slot_labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(slot_labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    slot_logits.view(-1, self.num_labels), slot_labels.view(-1)
                )

        # intent_loss
        if intent_labels is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = MSELoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1), intent_labels.view(-1)
                )
            else:
                intent_loss_fct = CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels),
                    intent_labels.view(-1),
                )
            loss += intent_loss

        if not return_dict:
            output = ((intent_logits, slot_logits),) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=(intent_logits, slot_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMRobertaForTokenAndIntentClassification(RobertaForTokenAndIntentClassification):
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
