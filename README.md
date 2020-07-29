# Dialog Language Model
한국어 Language Model을 활용한 대화 AI. 한국어 언어 모델을 사용하여 `auto regressive`, `text classification` 테스트.   
- **KoGPT2**: **질의**가 주어졌을 때, 다음 **답변**을 생성하는 모델
- **KoELECTRA**, **KoBERT**: **질의**에 대해서 "카테고리를 예측" 하는 과제 하나와
바로 "답변을 예측" 하는 `Text Classification` 과제를 테스트.

## 사용 Language Model
KoELECTRA, KoBERT, KoGPT2

## 환경
### Data
- [AI 허브 심리상담 데이터](http://www.aihub.or.kr/keti_data_board/language_intelligence): 심리 상담 데이터의 경우 회원가입 후 신청하면 다운로드 가능.
- [songys/Chatbot_data](https://github.com/songys/Chatbot_data)
### GPU
Colab pro, P100
### Package
```
kogpt2-transformers
kobert-transformers
transformers==3.0.2
torch
```

## Task
### 1. KoELECTAR & KoBERT Text Classifcation
KoELECTAR 및 KoBERT를 이용한 텍스트 분류 모델.
#### 1.1 질의에 대한 카테고리 분류
##### 데이터
Wellness 심리 상담 데이터 사용. Wellness 데이터의 경우 **카테고리/ 질문/ 답변**으로 나누어져있다. 카테고리 별로 3개 내외의 답변을 가지고 있으므로
Wellness 데이터의 경우  질문과 카테고리 클래스의 쌍으로 만들어 학습.   
  
**카테고리 클래스** 데이터  
```txt
감정/감정조절이상    0
감정/감정조절이상/화    1
감정/걱정    2
```
**카테고리 클래스와 질의** 데이터  
```text
감정/감정조절이상    그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.
감정/감정조절이상/화    화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.
감정/감정조절이상/화    화가 너무 많이 날 때는 심호흡을 해보는 게 어떨까요? 씁- 후-
감정/걱정    당연히 걱정이 되는 상황인 것 같아요. 저도 마음이 아프네요.
감정/걱정    모든 문제는 해결되기 마련이잖아요. 마음을 편히 드세요.
```
  
**질의과 카테고리 클래스 쌍** 데이터 
```txt
근데 감정을 다스리지 못해 욱하기도하고.    0
순간순간 감정조절을 못해요.    0
예전보다 화내는 게 과격해진 거 같아.    1
화가 안 참아져.    1
나도 그런 거 아닌가 걱정돼.    2
수술한다는 말에 얼마나 걱정이 되던지…    2
```
##### 모델

###### 1.KoELECTRA
```python
class koElectraForSequenceClassification(ElectraPreTrainedModel):
  def __init__(self,
               config,
               num_labels):
    super().__init__(config)
    self.num_labels = num_labels
    self.electra = ElectraModel(config)
    self.classifier = ElectraClassificationHead(config, num_labels)

    self.init_weights()
...중략...
```
###### 2.KoBERT
> 성능 아쉬운부분은 Dense가 없는 부분. (추후 수정)
```python
class KoBERTforSequenceClassfication(BertPreTrainedModel):
  def __init__(self,
                num_labels = 359, # 분류할 라벨 갯수를 설정
                hidden_size = 768, # hidden_size
                hidden_dropout_prob = 0.1,  # dropout_prop
               ):
    super().__init__(get_kobert_config())

    self.num_labels = num_labels 
    self.kobert = get_kobert_model()
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.classifier = nn.Linear(hidden_size, num_labels)

    self.init_weights()
...중략...
```

### 2. KoGPT2 Text Generation(Auto Regressive)
GPT-2 모델을 이용한 대화 및 답변 텍스트 생성.

#### Text Generation
##### 데이터
**카테고리/ 질문/ 답변** 데이터에서 질문과 답변의 쌍으로 구성.
```text
꼭 롤러코스터 타는 것 같아요.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
꼭 롤러코스터 타는 것 같아요.    저도 그 기분 이해해요. 많이 힘드시죠?
꼭 롤러코스터 타는 것 같아요.    그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.
롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    저도 그 기분 이해해요. 많이 힘드시죠?
롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.
작년 가을부터 감정조절이 잘 안 되는 거 같아.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
작년 가을부터 감정조절이 잘 안 되는 거 같아.    저도 그 기분 이해해요. 많이 힘드시죠?
```

##### 모델
```python
class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model()
    
  
...중략...
```

##### 텍스트 생성부분
[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0) 참고 하여, Huggingface의 Generate 사용. 
```python
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
```

# References
[KoBERT](https://github.com/SKTBrain/KoBERT)  
[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)  
[KoGPT2](https://github.com/SKT-AI/KoGPT2)  
[KoGPT2-Transformers](https://github.com/taeminlee/KoGPT2-Transformers/)  
[KoELECTRA](https://github.com/monologg/KoELECTRA)  
[enlipleai/kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)  
[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0)