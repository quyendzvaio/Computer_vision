# GPU Server — ZMQ SUB → YOLOv8 detect → FastAPI dashboard
# Chạy trên Ubuntu server với NVIDIA GPU (Quadro T2000, RTX, etc.)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libgomp1 \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xfixes0 libxcb-xkb1 \
    libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY gpu/ gpu/
COPY shared/ shared/
COPY edge/config.yaml edge/config.yaml

EXPOSE 5555 5556 8080

ENV PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

ENTRYPOINT ["python3", "-m", "gpu.main"]
