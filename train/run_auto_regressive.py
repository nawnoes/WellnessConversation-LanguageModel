import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import dataloader
from dataloader.wellness import WellnessAutoRegressiveDataset
from model.kogpt2 import DialogKoGPT2

if __name__ == '__main__':
    data_path = "../data/wellness_dialog_for_autoregressive_train.txt"
    checkpoint_path ="../checkpoint"
    save_ckpt_path = f"{checkpoint_path}/wellnesee-auto-regressive.pth"

    n_epoch = 5         # Num of Epoch
    batch_size = 4      # 배치 사이즈
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_step = 100 # 학습 저장 주기
    learning_rate = 5e-5  # Learning Rate

    dataset= WellnessAutoRegressiveDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DialogKoGPT2()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epoch):
        count = 0
        with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = torch.stack(data)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
                data = data.transpose(1, 0)

                outputs = model(data, labels=data)
                loss, logits = outputs[:2]
                loss.backward()
                optimizer.step()

                if count % 10 == 0:
                    print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
                if (count > 0 and count % 100 == 0) or (len(data) < batch_size):
                    torch.save({
                        'epoch': epoch,
                        'train_no': count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, save_ckpt_path)
                count += 1