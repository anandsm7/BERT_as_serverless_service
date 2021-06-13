import torch
from torch.utils.data import Dataset
import config as cfg

class BertDataset(Dataset):
    def __init__(self, log, target):
        self.log = log
        self.target = target
        self.tokenizer = cfg.TOKENIZER
        self.max_len = cfg.MAX_LEN
        
    def __len__(self):
        return len(self.log)
    
    def __getitem__(self, item):
        user_log = str(self.log[item])
        user_log = " ".join(user_log.split())
        
        inputs = self.tokenizer.encode_plus(
            user_log,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        return{
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.long)
        }