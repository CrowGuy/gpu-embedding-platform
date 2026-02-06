FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip

# FastAPI + metrics + ORT GPU + HF tokenizer
RUN python3 -m pip install --no-cache-dir \
    fastapi uvicorn[standard] pyyaml numpy prometheus-client \
    onnxruntime-gpu transformers

WORKDIR /app
COPY start.sh /app/start.sh
COPY config /app/config
COPY app /app/app

RUN chmod +x /app/start.sh

EXPOSE 8002
ENTRYPOINT ["/app/start.sh"]