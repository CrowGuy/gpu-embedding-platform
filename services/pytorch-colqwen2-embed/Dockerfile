FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip

RUN python3 -m pip install --no-cache-dir \
    fastapi uvicorn[standard] numpy pillow \
    transformers

# Torch CUDA wheel (12.4)
RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY start.sh /app/start.sh
COPY app /app/app
RUN chmod +x /app/start.sh

EXPOSE 8003
ENTRYPOINT ["/app/start.sh"]