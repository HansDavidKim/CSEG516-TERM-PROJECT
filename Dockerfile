FROM python:3.12.11-slim

WORKDIR /RLB-MI
COPY ./requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]
COPY . .

CMD ["python", "main.py"]
