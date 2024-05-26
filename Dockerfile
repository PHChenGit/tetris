FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

# Install Java 11
RUN apt-get update \
    && apt-get install -y openjdk-11-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ADD requirements.txt .

RUN pip install pip --upgrade \
    && pip install -r requirements.txt \
    && pip install "stable-baselines3[extra] >= 2.3.2"