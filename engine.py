#engine.py
import torch
import torch.nn as nn 
import config as cfg
import numpy as np
from tqdm import tqdm

def loss_fn(outputs, targets):
  return nn.CrossEntropyLoss()(outputs,targets)

def train_fn(data_loader,model,optimizer,device,scheduler):
  """
  Train the model from one epoch
  :param data_loader: torch dataloader
  :param model:bert base model
  :param optimizer: adam, sgd..etc
  :param device: can be cpu or gpu
  :param scheduler: learning rate scheduler
  """
  model.train()
  #loop over all batches
  i = 0
  size = len(data_loader)
  for d in tqdm(data_loader,total=size):
    #extract ids, token type ids and mask
    ids = d['ids']
    token_type_ids = d['token_type_ids']
    mask = d['mask']
    targets = d["targets"]

    #move everything to device
    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device,dtype=torch.long)
    mask = mask.to(device,dtype=torch.long)
    targets = targets.to(device, torch.long)
    #zero-grad optimizers
    optimizer.zero_grad()
    outputs = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )[0]
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()

def eval_fn(data_loader,model,device):
  model.eval()
  fin_targets = []
  fin_outputs = []
  #to not run out of GPU & not to change gradients
  with torch.no_grad(): 
    for d in data_loader:
      ids = d["ids"]
      token_type_ids = d["token_type_ids"]
      mask = d["mask"]
      targets = d["targets"]

      ids = ids.to(device, dtype=torch.long)
      token_type_ids = token_type_ids.to(device,dtype=torch.long)
      mask = mask.to(device,dtype=torch.long)
      targets = targets.to(device, torch.long)

      outputs = model(
          ids,
          mask=mask,
          token_type_ids=token_type_ids,
      )[0]
      val_loss = loss_fn(outputs, targets)
      targets = targets.cpu().detach()
      fin_targets.extend(targets.numpy().tolist())
      outputs = outputs.cpu().detach()
      fin_outputs.extend(outputs.numpy().tolist())

    return val_loss, fin_outputs, fin_targets
  
def inference_fn(bert_dict, model, device):
  model.eval()
  final_out = dict()
  #to not run out of GPU & not to change gradients
  with torch.no_grad(): 
    ids = bert_dict["ids"]
    token_type_ids = bert_dict["token_type_ids"]
    mask = bert_dict["mask"]

    ids = ids.to(device, dtype=torch.long).view(1, -1)
    token_type_ids = token_type_ids.to(device,dtype=torch.long).view(1, -1)
    mask = mask.to(device,dtype=torch.long).view(1, -1)
    outputs = model(
        ids,
        mask=mask,
        token_type_ids=token_type_ids,
    )[0]
    targets = outputs.cpu().detach()
    final_result = np.argmax(targets).item()
    final_out['class'] = cfg.CLASS_NAME[final_result]
    

    return final_out