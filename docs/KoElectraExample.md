# KoELECTRA Example
## Electra 란
Replaced Token Detection을 사용해, 기존의 Masked Language Model이 아닌 토큰이 `Geneator`에서 replace 되었는지
아닌지를 판단하여, 더 빠르고 작은 모델로 BERT보다 좋은 성능을 낸 언어 모델

## KoELECTRA 란?
14GB의 한국어 텍스트를 이용해 ELECTRA를 학습시킨 모

### Vocab
원 논문에서 사용한 wordpiece를 사용. 
- vocab 사이즈는 32200개 ([unused] 토큰 200개 포함)
- cased: `do_lower_case = False`를 통해 대소문자 구분

### 사용예
#### 1. Tokenizer
```python
from transformers import ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
# ['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]']

tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]'])
#[2, 18429, 41, 6240, 15229, 6204, 20894, 5689, 12622, 10690, 18, 3]
```

#### 2. Model
```python
from transformers import ElectraModel

model = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")  # KoELECTRA-Base
model = ElectraModel.from_pretrained("monologg/koelectra-small-discriminator")  # KoELECTRA-Small
model = ElectraModel.from_pretrained("monologg/koelectra-base-v2-discriminator")  # KoELECTRA-Base-v2
model = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")  # KoELECTRA-Small-v2
```