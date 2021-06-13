import transformers
from transformers import BertForSequenceClassification
import torch.nn as nn 
import config as cfg

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = cfg.NUM_CLASSES,
            output_attentions = False,
            output_hidden_states = False,
        )
        
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids=token_type_ids
        )
        return output
    
if __name__ == "__main__":
    model = BERTBaseUncased()
    print(model)