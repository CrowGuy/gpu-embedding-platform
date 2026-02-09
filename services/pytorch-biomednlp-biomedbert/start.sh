#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${SERVICE_CONFIG:-/app/config/service.yaml}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

MODEL_DIR="${MODEL_DIR:-/models/biomednlp-biomedbert}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-biomednlp-biomedbert}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"

MAX_LENGTH="${MAX_LENGTH:-256}"
POOLING="${POOLING:-mean}"
NORMALIZE="${NORMALIZE:-1}"

API_KEY="${API_KEY:-eslllm}"

MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
BATCH_WAIT_MS="${BATCH_WAIT_MS:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"

if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r")) or {}
m = {
  "HOST":"host","PORT":"port",
  "MODEL_DIR":"model_dir","SERVED_MODEL_NAME":"served_model_name",
  "DEVICE":"device","DTYPE":"dtype",
  "MAX_LENGTH":"max_length","POOLING":"pooling","NORMALIZE":"normalize",
  "API_KEY":"api_key",
  "MAX_BATCH_SIZE":"max_batch_size","BATCH_WAIT_MS":"batch_wait_ms","MAX_CONCURRENCY":"max_concurrency",
}
def esc(v): return str(v).replace('"','\\"').replace("\n"," ").strip()
out=[]
for k,y in m.items():
  v=cfg.get(y)
  if v is None or v=="": continue
  out.append(f'export {k}="{esc(v)}"')
print("\n".join(out))
PY
  )"
fi

echo "[pytorch-biomednlp-biomedbert] HOST=$HOST PORT=$PORT"
echo "[pytorch-biomednlp-biomedbert] MODEL_DIR=$MODEL_DIR"
echo "[pytorch-biomednlp-biomedbert] SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "[pytorch-biomednlp-biomedbert] DEVICE=$DEVICE DTYPE=$DTYPE MAX_LENGTH=$MAX_LENGTH POOLING=$POOLING NORMALIZE=$NORMALIZE"
echo "[pytorch-biomednlp-biomedbert] BATCH max_batch=$MAX_BATCH_SIZE wait_ms=$BATCH_WAIT_MS concurrency=$MAX_CONCURRENCY"
echo "[pytorch-biomednlp-biomedbert] API_KEY=${API_KEY:0:2}*** (masked)"

exec uvicorn app.main:app --host "$HOST" --port "$PORT"