FROM python

# 파일 복사
COPY ./checkpoint /app/checkpoint
COPY ./data /app/data
COPY ./model /app/model
COPY ./service /app/service

# 패키지 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
EXPOSE 9900
CMD ["python", "app.py"]
