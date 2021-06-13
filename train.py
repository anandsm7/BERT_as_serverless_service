#train.py
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn 
from engine import train_fn, eval_fn

from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import config as cfg
from model import BERTBaseUncased
from dataset import BertDataset

def train():
  dfx = pd.read_csv(cfg.TRAINING_FILE).fillna('none')
  dfx = dfx.sample(frac=1).reset_index(drop=True)
  # dfx = dfx.iloc[:100,:]
  dfx_mapper = {
      'food':0, 'transport':1, 'shopping':2, 'bills':3, 'credit':4
  }
  dfx.cat = dfx.cat.map(dfx_mapper)
  df_train,df_valid = model_selection.train_test_split(
      dfx,
      test_size=0.1,
      random_state=42,
      stratify=dfx.cat.values
  )

  df_train = df_train.reset_index(drop=True)
  df_valid = df_valid.reset_index(drop=True)

  train_dataset = BertDataset(
      log=df_train.logs.values,
      target=df_train.cat.values
  )

  train_dataloader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=cfg.TRAIN_BATCH_SIZE,
      num_workers=3
  )

  valid_dataset = BertDataset(
      log=df_valid.logs.values,
      target=df_valid.cat.values
  )

  valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=cfg.VALID_BATCH_SIZE,
      num_workers=1
  )

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = BERTBaseUncased()
  model.to(device)
           
  #create parameters we want to optimize
  #we generally dont use any decay for bias
  #and weight layers
  param_optimizer = list(model.named_parameters())
  no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
  optimizer_parameters = [
               {
                   "params":[
                             p for n,p in param_optimizer if 
                             not any(nd in n for nd in no_decay)
                   ],
                   "weight_decay": 0.001,
               },
               {
                   "params":[
                             p for n,p in param_optimizer if 
                             any(nd in n for nd in no_decay)
                   ],
                   "weight_decay": 0.0,
               },
  ]

  #calculate the number of training steps
  #used by schduler
  num_train_steps = int(
      len(df_train) / cfg.TRAIN_BATCH_SIZE * cfg.EPOCHS
  )

  #AdamW in widely used optimizer for transformer based model
  optimizer = AdamW(optimizer_parameters, lr =3e-5)

  #fetch the schduler
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=num_train_steps
  )

  # model = nn.DataParallel(model)

  best_accuracy = 0
  for epoch in range(cfg.EPOCHS):
    print(f"Epoch: {epoch}")
    print("Model training..")
    print(len(train_dataloader))
    train_fn(train_dataloader, model,
             optimizer,device,scheduler)
    print("Model Evalutaion..")
    val_loss, outputs, targets = eval_fn(valid_dataloader,model,device)

    accuracy = metrics.accuracy_score(targets, np.argmax(outputs,axis=1))
    print(f"epoch: {epoch}, val acc: {accuracy}, val_loss: {val_loss}")

    if accuracy > best_accuracy:
      torch.save(model,cfg.MODEL_PATH)
      best_accuracy = accuracy

    
if __name__ == "__main__":
    train()