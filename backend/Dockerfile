FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_HOME=/opt/torch
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU wheels keep the image small; switch the index URL for CUDA builds.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt
# Bake the pretrained backbone into the image so the embedding detector works offline.
RUN python -c "import torchvision; torchvision.models.resnet18(weights='DEFAULT')"
COPY chimera ./chimera
COPY scripts ./scripts
ENV CHIMERA_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "chimera.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
