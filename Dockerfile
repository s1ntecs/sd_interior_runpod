# 1) Базовый образ с CUDA 11.8 и cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 2) Системные зависимости
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-dev python3-pip python3.10-venv \
      fonts-dejavu-core rsync git git-lfs jq moreutils \
      aria2 wget curl \
      libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
      ffmpeg libgoogle-perftools4 libtcmalloc-minimal4 procps && \
    rm -rf /var/lib/apt/lists/*

# 3) Рабочая директория
WORKDIR /workspace

# 4) Копируем только файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt
# Копирование requirements.txt перед остальным кодом позволяет Docker кэшу повторно использовать слой с зависимостями, пока requirements.txt не меняется  [oai_citation:8‡reddit.com](https://www.reddit.com/r/docker/comments/1efw2b7/best_practice_rebuild_docker_imagecache_on/?utm_source=chatgpt.com).  
# Таким образом, pip install не будет запускаться заново при изменении только кода проекта  [oai_citation:9‡stackoverflow.com](https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project?utm_source=chatgpt.com).  
# Это предотвращает повторную установку пакетов, если requirements.txt не менялся  [oai_citation:10‡forums.docker.com](https://forums.docker.com/t/requirements-txt-to-be-installed-once-for-the-first-time-when-the-env-is-created-and-when-there-ar-changes-made-init/139313?utm_source=chatgpt.com).

# 5) Копируем остальной код
COPY . .

# 6) Создаём директории и загружаем чекпоинты
RUN mkdir -p loras checkpoints && \
    python3 download_checkpoints.py

# 7) Точка входа
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]