FROM python

# 파일 복사
## 1. 체크포인트 복사
COPY ./checkpoint/kobert-wellnesee-text-classification.pth /app/checkpoint/kobert-wellnesee-text-classification.pth
COPY ./checkpoint/koelectra-wellnesee-text-classification.pth /app/checkpoint/koelectra-wellnesee-text-classification.pth
## 2. 카테고리 및 답반 데이터 복사
COPY ./data /app/data/wellness_dialog_category.txt
COPY ./data /app/data/wellness_dialog_answer.txt
## 3. 소스파일 복
COPY ./model /app/model
COPY ./service /app/service

# 패키지 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade cython
RUN pip install -r /app/requirements.txt
EXPOSE 9900
CMD ["python", "/app/service/api.py"]
