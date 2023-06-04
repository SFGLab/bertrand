from transformers import BertForSequenceClassification
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from bertrand.model.focal_loss import FocalLoss

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert'
class ProteinClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.config = config
        
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
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = FocalLoss(gamma=3, alpha=0.25, no_agg=True)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = torch.mean(loss * weights)

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
