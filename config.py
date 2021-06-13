import transformers
import torch

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
NUM_CLASSES = 5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EPOCHS = 2
BERT_PATH = './input/bert-base-uncased/'
MODEL_PATH = './model/pytorch_model.bin'
TRAINING_FILE = './input/processed.csv'
CLASS_NAME = ['food', 'transport', 'shopping', 'bills', 'credit']

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
