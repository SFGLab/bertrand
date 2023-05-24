from transformers import BertForSequenceClassification
import torch.nn as nn

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert'
class ProteinClassifier(nn.Module):
    def __init__(self):
        super(ProteinClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(self.bert.config.hidden_size, 1),
                                        nn.Tanh())
        
    def forward(self, *args, **kwargs):
        output = self.bert(*args, **kwargs)
        return self.classifier(output.pooler_output)
