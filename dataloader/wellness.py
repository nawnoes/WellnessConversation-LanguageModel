from torch.utils.data import Dataset # 데이터로더

from kogpt2_transformers import get_kogpt2_tokenizer
from kobert_transformers import get_tokenizer

class WellnessAutoRegressiveDataset(Dataset):
  """Wellness Auto Regressive Dataset"""

  def __init__(self, file_path,vocab,tokenizer):
    self.file_path = file_path
    self.data =[]
    self.vocab =vocab
    self.tokenizer = tokenizer
    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      toeknized_line = tokenizer(line[:-1])
      index_of_words = [vocab[vocab.bos_token],] + vocab[toeknized_line]+ [vocab[vocab.eos_token]]

      self.data.append(index_of_words)

    file.close()

  def __len__(self):
    return len(self.data)
  def __getitem__(self,index):
    item = self.data[index]
    # print(item)
    return item

if __name__ == "__main__":
