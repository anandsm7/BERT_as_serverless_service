import torch
import config as cfg
import engine
from model import BERTBaseUncased

class Inference:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        #initialize model
        self.model = BERTBaseUncased()
        
        #load model weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        
    def sentence_prediction(self, sentence):
        bert_dict = dict()
        tokenizer = cfg.TOKENIZER
        max_len = cfg.MAX_LEN
        
        log = str(sentence)
        log = " ".join(log.split())
        inputs = tokenizer.encode_plus(
            log,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
        )
        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        bert_dict["ids"] = torch.tensor(ids, dtype=torch.long)
        bert_dict['mask'] = torch.tensor(mask, dtype=torch.long)
        bert_dict['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.long)
        result = engine.inference_fn(bert_dict, self.model, self.device)
        print("inference:",result['class'])
        return result
        
if __name__ == "__main__":
    infer = Inference(cfg.MODEL_PATH, cfg.DEVICE)
    
    log = "i purchased eggs, breads and apples"
    infer.sentence_prediction(log)
        
