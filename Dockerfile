# GPU Server — YOLOv8 detect → FastAPI dashboard
# Quadro T2000 target. CUDA 12.2 runtime, ~1.1GB image.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ponytail: no dev deps
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt && pip cache purge && rm requirements-gpu.txt

# Layer: app code
COPY gpu/ gpu/
COPY shared/ shared/
COPY edge/config.yaml edge/config.yaml

VOLUME /app/gpu/models

EXPOSE 5555 5556 8080

ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    CUDA_CACHE_MAXSIZE=2147483648

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

ENTRYPOINT ["python3", "-m", "gpu.main"]
