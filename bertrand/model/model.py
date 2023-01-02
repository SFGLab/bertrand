from typing import Union, Tuple

import torch
from transformers import (
    BertModel,
    BertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from bertrand.model.focal_loss import FocalLoss


class BERTrand(BertPreTrainedModel):
    """
    BERTrand model class
    BERT followed by sequence classification head
    Predicts peptide:TCR binding based on the output embedding of the CLS token
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.config = config

        self.bert = BertModel(config)
        self.classifier = RobertaClassificationHead(config)  # dense->dropout->dense
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        weights: torch.Tensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], SequenceClassifierOutput
    ]:
        """

        :param input_ids: amino acid ids
        :param attention_mask: attention mask (1 for every non-padding token)
        :param token_type_ids: 0 for peptide amino acids, 1 for TCR amino acids
        :param position_ids: position in sequence for every token
        :param head_mask: mask to use only some of the attention heads
        :param inputs_embeds: user-provided input embeddings instead of input_ids (cannot be used simultaneously)
        :param labels: target for every observation (0 or 1)
        :param weights: weights for every observation
        :param output_attentions: flag to return BERT attention outputs
        :param output_hidden_states: flag to return BERT hidden states
        :param return_dict: flag to return a tuple or a dictionary (use this to override config.use_return_dict)
        :return: loss, logits, hidden states, attention outputs (either a tuple or SequenceClassifierOutput)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = FocalLoss(gamma=3, alpha=0.25, no_agg=True)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = torch.mean(loss * weights)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
