# ── Build Stage ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Use headless OpenCV (no GUI libs -- saves ~200MB)
COPY requirements.txt .
RUN sed -i 's/opencv-python/opencv-python-headless/' requirements.txt && \
    sed -i 's/uvicorn\[standard\]/uvicorn/' requirements.txt && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt websockets && \
    rm -rf /opt/venv/lib/python3.11/site-packages/pip /opt/venv/lib/python3.11/site-packages/setuptools

# ── Runtime Stage ────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime system deps for numpy/OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set up data directory
RUN mkdir -p /app/data /app/models && chown -R appuser:appuser /app/data /app/models

# Use venv python
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Switch to non-root
USER appuser

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

ENTRYPOINT ["python", "main.py"]
