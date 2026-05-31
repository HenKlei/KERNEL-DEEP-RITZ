ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

FROM ${BASE_IMAGE}

ARG TORCH_INDEX

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages torch==2.11.0 --index-url ${TORCH_INDEX} && \
    pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --break-system-packages -e .

RUN mkdir -p /app/figures
VOLUME /app/figures
