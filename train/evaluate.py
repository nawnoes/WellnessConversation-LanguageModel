import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm import tqdm

import torch
from transformers import (
  AdamW,
  ElectraConfig,
  ElectraTokenizer
)
from torch.utils.data import dataloader
from dataloader.wellness import WellnessTextClassificationDataset
from model.koelectra import koElectraForSequenceClassification
from model.kobert import KoBERTforSequenceClassfication
from kobert_transformers import get_tokenizer

logger = logging.getLogger(__name__)


MODEL_CLASSES ={
  "koelectra": (ElectraConfig, koElectraForSequenceClassification, ElectraTokenizer),
  "kobert": (KoBERTforSequenceClassfication)
}
CHECK_POINT ={
  "koelectra": "../checkpoint/koelectra-wellnesee-text-classification.pth",
  "kobert": "../checkpoint/kobert-wellnese.pth"
}
def get_model_and_tokenizer(model_name, device):
  save_ckpt_path = CHECK_POINT[model_name]

  if model_name== "koelectra":
    model_name_or_path = "monologg/koelectra-base-discriminator"

    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)
    electra_config = ElectraConfig.from_pretrained(model_name_or_path)
    model = koElectraForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                               config=electra_config,
                                                               num_labels=359)
  elif model_name =='kobert':
    tokenizer = get_tokenizer()
    model = KoBERTforSequenceClassfication()

  if os.path.isfile(save_ckpt_path):
      checkpoint = torch.load(save_ckpt_path, map_location=device)
      pre_epoch = checkpoint['epoch']
      # pre_loss = checkpoint['loss']
      model.load_state_dict(checkpoint['model_state_dict'])

      print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}")

  return model, tokenizer

def get_model_input(data):
  if model_name =='kobert':
    return data
  elif model_name== "koelectra":
    return {'input_ids': data['input_ids'],
              'attention_mask': data['attention_mask'],
              'labels': data['labels']
              }
def evaluate(model_name, device, batch_size, data_path):

  model, tokenizer = get_model_and_tokenizer(model_name, device)
  model.to(device)

  # WellnessTextClassificationDataset 데이터 로더
  eval_dataset = WellnessTextClassificationDataset(file_path=data_path,device=device, tokenizer=tokenizer)
  eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

  logger.info("***** Running evaluation on %s dataset *****")
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", batch_size)

  loss = 0
  acc = 0


  # model.eval()
  for data in tqdm(eval_dataloader, desc="Evaluating"):
    with torch.no_grad():
      inputs = get_model_input(data)
      outputs = model(**inputs)
      loss += outputs[0]
      logit = outputs[1]
      acc += (logit.argmax(1)==inputs['labels']).sum().item()

  return loss / len(eval_dataset), acc / len(eval_dataset)

if __name__ == '__main__':
  root_path = '..'
  data_path = f"{root_path}/data/wellness_dialog_for_text_classification_test.txt"
  checkpoint_path = f"{root_path}/checkpoint"
  save_ckpt_path = f"{checkpoint_path}/koelectra-wellnesee-text-classification.pth"
  model_name_or_path = "monologg/koelectra-base-discriminator"

  n_epoch = 50  # Num of Epoch
  batch_size = 16  # 배치 사이즈
  ctx = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(ctx)
  model_names=["kobert","koelectra"]
  for model_name in model_names:
    eval_loss, eval_acc = evaluate(model_name, device, batch_size, data_path)
    print(f'\tLoss: {eval_loss:.4f}(valid)\t|\tAcc: {eval_acc * 100:.1f}%(valid)')