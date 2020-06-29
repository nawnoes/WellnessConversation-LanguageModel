# KoGPT2 Example

## Tokenizer
입력을 받아 id로 변환하고, id를 다시 단어로 변환할 때 사용.
```python
from kogpt2_transformers import get_kogpt2_tokenizer

tokenizer = get_kogpt2_tokenizer()

ids = tokenizer.encode("안녕하세요 KoGPT2입니다")
#ids = [28911, 4455, 527, 47534, 47794, 44113, 47465, 484]

decode_string = tokenizer.decode(ids)
# decode_string = '안녕하세요 KoGPT2입니다'
```