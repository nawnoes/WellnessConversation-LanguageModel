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

from torch.utils.data import dataloader
from dataloader.wellness import WellnessTextClassificationDataset
from model.text_classification import KoBERTforSequenceClassfication

def train(device, epoch, model, optimizer, train_loader, save_step, save_ckpt_path, train_step = 0):
    losses = []
    train_start_index = train_step+1 if train_step != 0 else 0
    total_train_step = len(train_loader) - train_start_index
    model.train()

    with tqdm(total= total_train_step, desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader, train_start_index):
            if i >= total_train_step:
                torch.save({
                    'epoch': epoch+1,  # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),  # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 저장
                    'loss': loss,  # Loss 저장
                    'train_step': 0,  # 현재 진행한 학습
                    'total_train_step': 0 # 현재 epoch에 학습 할 총 train step
                }, save_pretrain)
                model.save()
                break
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs, logits_cls, logits_lm = model(inputs, segments)


            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))

            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            if i % save_step == 0:
                torch.save({
                    'epoch': epoch,                                   # 현재 학습 epoch
                    'model_state_dict': model.state_dict(),           # 모델 저장
                    'optimizer_state_dict': optimizer.state_dict(),   # 옵티마이저 저장
                    'loss': loss,                                     # Loss 저장
                    'train_step': i,                                  # 현재 진행한 학습
                    'total_train_step': len(train_loader)             # 현재 epoch에 학습 할 총 train step
                }, save_pretrain)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)



if __name__ == '__main__':
    # Data 및 Vocab 경로

    data_path = "../data/wellness_dialog_for_text_classification.txt"
    checkpoint_path ="../checkpoint"
    save_ckpt_path = f"{checkpoint_path}/wellnesee-text-classification.pth"

    n_epoch = 20          # Num of Epoch
    batch_size = 128      # 배치 사이즈
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_step = 100 # 학습 저장 주기
    learning_rate = 5e-5  # Learning Rate

    # WellnessTextClassificationDataset 데이터 로더
    dataset = WellnessTextClassificationDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Refomer Language Model 생성
    model = KoBERTforSequenceClassfication()
    model.to(device)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pre_epoch, pre_loss, train_step = 0, 0, 0
    if os.path.isfile(save_ckpt_path):
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch']
        pre_loss = checkpoint['loss']
        train_step =  checkpoint['train_step']
        total_train_step =  checkpoint['total_train_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}, loss={pre_loss}")
        # best_epoch += 1

    losses = []
    offset = pre_epoch
    for step in range(n_epoch):
        epoch = step + offset
        loss = train(device, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader, save_step, train_step)
        losses.append(loss)

    # data
    data = {
        "loss": losses
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


