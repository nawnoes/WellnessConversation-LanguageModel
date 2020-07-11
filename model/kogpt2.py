import torch
import torch.nn as nn
from transformers import PreTrainedModel
from kogpt2_transformers import get_kogpt2_model, get_kogpt2_tokenizer


class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model()

  def save(self, epoch, count, optimizer_state_dict, loss, save_path):
    torch.save({
      'epoch': epoch,
      'train_no': count,
      'model_state_dict': self.kogpt2.state_dict(),
      'optimizer_state_dict': optimizer_state_dict,
      'loss': loss
    }, save_path + 'wellness-dialogkogpt2-checkpoint.tar')
  def generate(self,
               input_ids,
               do_sample=True,
               max_length=50,
               top_k=0,
               temperature=0.7):
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_k=top_k,
               temperature=temperature)

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs
f

