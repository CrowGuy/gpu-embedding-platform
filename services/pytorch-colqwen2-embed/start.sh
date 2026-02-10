#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"

<<<<<<< HEAD
MODEL_DIR="${MODEL_DIR:-/models/colqwen2-v1.0}"
=======
MODEL_DIR="${MODEL_DIR:-/models/colqwen2-v1.0-merged}"
>>>>>>> feat/lab/pytorch_embed_services
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-colqwen2-embed}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

MAX_LENGTH="${MAX_LENGTH:-128}"
NORMALIZE="${NORMALIZE:-1}"

API_KEY="${API_KEY:-eslllm}"

echo "[pytorch-colqwen2-embed] HOST=$HOST PORT=$PORT"
echo "[pytorch-colqwen2-embed] MODEL_DIR=$MODEL_DIR SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "[pytorch-colqwen2-embed] DEVICE=$DEVICE DTYPE=$DTYPE TRUST_REMOTE_CODE=$TRUST_REMOTE_CODE"
echo "[pytorch-colqwen2-embed] MAX_LENGTH=$MAX_LENGTH NORMALIZE=$NORMALIZE"
echo "[pytorch-colqwen2-embed] API_KEY=${API_KEY:0:2}*** (masked)"

exec uvicorn app.main:app --host "$HOST" --port "$PORT"