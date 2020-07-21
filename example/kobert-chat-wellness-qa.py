import torch
import torch.nn as nn
import random

from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer

def load_answer():
  root_path = '..'
  answer_path = f"{root_path}/data/chatbot_wellness_category.txt"

  a_f = open(answer_path,'r')
  answer_lines = a_f.readlines()
  answer = {}

  for line_num, line_data in enumerate(answer_lines):
    data = line_data.split('    ')

    answer[data[1][:-1]] =data[0]

  return  answer

if __name__ == "__main__":
  root_path='..'
  checkpoint_path =f"{root_path}/checkpoint"
  save_ckpt_path = f"{checkpoint_path}/kobert-chatbot-wellness.pth"

  #답변 불러오기
  answer = load_answer()

  ctx = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(ctx)

  # 저장한 Checkpoint 불러오기
  checkpoint = torch.load(save_ckpt_path, map_location=device)

  model = KoBERTforSequenceClassfication(num_labels=9322)
  model.load_state_dict(checkpoint['model_state_dict'])

  model.eval()

  tokenizer = get_tokenizer()

  while 1:
    sent = input('\nQuestion: ') # '요즘 기분이 우울한 느낌이에요'
    data = kobert_input(tokenizer,sent, device,512)
    # print(data)

    output = model(**data)

    logit = output
    softmax_logit = nn.Softmax(logit).dim
    softmax_logit = softmax_logit[0].squeeze()

    max_index = torch.argmax(softmax_logit).item()
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

    print(f'Answer: {answer[str(max_index)]}, index: {max_index}, value: {max_index_value}')
    print('-'*50)
  # print('argmin:',softmax_logit[torch.argmin(softmax_logit)])




