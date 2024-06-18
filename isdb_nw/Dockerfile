FROM python:3.9-slim-buster

WORKDIR /app

COPY ./requirements.txt app/requirements.txt
RUN pip3 install -r app/requirements.txt

RUN apt-get update && apt-get -y install git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/ai-forever/sbert_large_nlu_ru /tmp/model

COPY . /app