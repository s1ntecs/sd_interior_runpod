# Базовый образ с CUDA 11.8 и cuDNN8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) Системные зависимости
RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
      python3-dev python3-pip python3.10-venv \
      fonts-dejavu-core rsync git git-lfs jq moreutils \
      aria2 wget curl \
      libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
      ffmpeg libgoogle-perftools4 libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 2) Рабочая директория
WORKDIR /workspace

# 3) Копируем список зависимостей и ставим их
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 4) Копируем исходники и вспомогательные скрипты
COPY . .

# 5) Создаём папки под LoRA и чекпоинты, и загружаем модели
RUN mkdir -p loras checkpoints && \
    python3 download_checkpoints.py

# 6) Точка входа
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]