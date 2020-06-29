import torch
import torch.nn as nn
from transformers import PreTrainedModel
from kogpt2_transformers import get_kogpt2_model, get_kogpt2_tokenizer


class DialogKoGPT2(nn.Module):
  def __init__(self):
    self.kogpt2 = get_kogpt2_model()

  def save(self, epoch, count, optimizer_state_dict, loss, save_path):
    torch.save({
      'epoch': epoch,
      'train_no': count,
      'model_state_dict': self.kogpt2.state_dict(),
      'optimizer_state_dict': optimizer_state_dict,
      'loss': loss
    }, save_path + 'DialogKoGPT2_checkpoint.tar')

  def load_state_dict(self, model_state_dict):
    self.kogpt2.load_state_dict(model_state_dict)

  def forward(self, input, labels = None):
    if labels != None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs