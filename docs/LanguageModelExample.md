# Language Model Example
사용 언어모델에 대한 예 모음.

## KoBERT-transformers example
**Model**
```python
>>> import torch
>>> from kobert_transformers import get_kobert_model, get_distilkobert_model
>>> model = get_kobert_model()
>>> model.eval()
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
>>> sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
>>> sequence_output[0]
tensor([[-0.2461,  0.2428,  0.2590,  ..., -0.4861, -0.0731,  0.0756],
        [-0.2478,  0.2420,  0.2552,  ..., -0.4877, -0.0727,  0.0754],
        [-0.2472,  0.2420,  0.2561,  ..., -0.4874, -0.0733,  0.0765]],
       grad_fn=<SelectBackward>
```
**Tokenizer**
```python
>>> from kobert_transformers import get_tokenizer
>>> tokenizer = get_tokenizer()
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```
## KoGPT2-transformers example
**Model**
```python
import torch
from kogpt2_transformers import get_kogpt2_model, get_kogpt2_tokenizer

torch.manual_seed(42)

#Model
model = get_kogpt2_model()

#Tokenizer
tokenizer = get_kogpt2_tokenizer()

input_ids = tokenizer.encode("안녕", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=3)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
```