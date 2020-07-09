import os
import numpy as np
import torch
from model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

root_path='drive/My Drive/Colab Notebooks/dialogLM'
data_path = f"{root_path}/data/wellness_dialog_for_autoregressive_train.txt"
checkpoint_path =f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

tokenizer = get_kogpt2_tokenizer()

sent = '요즘 기분이 우울한 느낌이에요'#input('문장 입력: ')

tokenized_indexs = tokenizer.encode(sent)
count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수

while 1:
  input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs).unsqueeze(0)
  # set top_k to 50
  sample_output = model.generate(
      input_ids,
      do_sample=True,
      max_length=50,
      top_k=50
  )

  print("Output:\n" + 100 * '-')
  print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

for s in kss.split_sentences(sent):
    print(s)