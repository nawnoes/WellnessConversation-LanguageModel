import torch
import torch.nn as nn
import random

from model.kobert import KoBERTforSequenceClassfication, kobert_input
from kobert_transformers import get_tokenizer


def load_wellness_answer():
  root_path = '..'
  category_path = f"{root_path}/data/wellness_dialog_category.txt"
  answer_path = f"{root_path}/data/wellness_dialog_answer.txt"

  c_f = open(category_path,'r')
  a_f = open(answer_path,'r')

  category_lines = c_f.readlines()
  answer_lines = a_f.readlines()

  category = {}
  answer = {}
  for line_num, line_data in enumerate(category_lines):
    data = line_data.split('    ')
    category[data[1][:-1]]=data[0]

  for line_num, line_data in enumerate(answer_lines):
    data = line_data.split('    ')
    keys = answer.keys()
    if(data[0] in keys):
      answer[data[0]] += [data[1][:-1]]
    else:
      answer[data[0]] =[data[1][:-1]]

  return category, answer

if __name__ == "__main__":
  root_path='..'
  checkpoint_path =f"{root_path}/checkpoint"
  save_ckpt_path = f"{checkpoint_path}/kobert-wellnesee-text-classification.pth"

  #답변과 카테고리 불러오기
  category, answer = load_wellness_answer()

  ctx = "cuda" if torch.cuda.is_available() else "cpu"
  device = torch.device(ctx)

  # 저장한 Checkpoint 불러오기
  checkpoint = torch.load(save_ckpt_path, map_location=device)

  model = KoBERTforSequenceClassfication()
  model.load_state_dict(checkpoint['model_state_dict'])

  model.eval()

  tokenizer = get_tokenizer()

  while 1:
    sent = input('\nQuestion: ') # '요즘 기분이 우울한 느낌이에요'
    data = kobert_input(tokenizer,sent, device,512)
    # print(data)

    output = model(**data)

    logit = output[0]
    softmax_logit = torch.softmax(logit,dim=-1)
    softmax_logit = softmax_logit.squeeze()

    max_index = torch.argmax(softmax_logit).item()
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

    answer_list = answer[category[str(max_index)]]
    answer_len= len(answer_list)-1
    answer_index = random.randint(0,answer_len)
    print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
    print('-'*50)
  # print('argmin:',softmax_logit[torch.argmin(softmax_logit)])




