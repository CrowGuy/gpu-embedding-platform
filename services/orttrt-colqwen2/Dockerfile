FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip

# Core runtime
RUN python3 -m pip install --no-cache-dir \
    fastapi uvicorn[standard] pyyaml pillow numpy prometheus-client \
    onnxruntime-gpu

WORKDIR /app
COPY start.sh /app/start.sh
COPY config /app/config
COPY app /app/app

RUN chmod +x /app/start.sh

EXPOSE 8010
ENTRYPOINT ["/app/start.sh"]