FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip

# Core runtime deps
RUN python3 -m pip install --no-cache-dir \
    fastapi uvicorn[standard] pyyaml numpy prometheus-client \
    transformers

# PyTorch: choose one
# Option A: rely on preinstalled torch in base image? (not present)
# Option B: install torch with CUDA wheels - user may prefer to pin
# NOTE: For reliability, install torch CUDA wheel explicitly.
RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY start.sh /app/start.sh
COPY config /app/config
COPY app /app/app

RUN chmod +x /app/start.sh

EXPOSE 8002
ENTRYPOINT ["/app/start.sh"]