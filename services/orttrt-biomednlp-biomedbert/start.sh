#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${SERVICE_CONFIG:-/app/config/service.yaml}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

MODEL_DIR="${MODEL_DIR:-/models/biomednlp-biomedbert}"
ONNX_PATH="${ONNX_PATH:-}"

PROVIDER="${PROVIDER:-cuda}"     # tensorrt|cuda|cpu

POOLING="${POOLING:-mean}"       # mean|cls
NORM="${NORM:-1}"                # 1/0
MAX_LENGTH="${MAX_LENGTH:-256}"

API_KEY="${API_KEY:-eslllm}"

TRT_FP16="${TRT_FP16:-1}"
TRT_ENGINE_CACHE="${TRT_ENGINE_CACHE:-/cache/trt_engines}"
TRT_TIMING_CACHE="${TRT_TIMING_CACHE:-/cache/trt_timing.cache}"

if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r")) or {}
allowed = {
  "HOST":"host","PORT":"port",
  "MODEL_DIR":"model_dir","ONNX_PATH":"onnx_path",
  "PROVIDER":"provider",
  "POOLING":"pooling","NORM":"norm","MAX_LENGTH":"max_length",
  "API_KEY":"api_key",
  "TRT_FP16":"trt_fp16","TRT_ENGINE_CACHE":"trt_engine_cache","TRT_TIMING_CACHE":"trt_timing_cache",
}
def esc(v): return str(v).replace('"','\\"').replace("\n"," ").strip()
out=[]
for k,y in allowed.items():
  v=cfg.get(y)
  if v is None or v=="": continue
  out.append(f'export {k}="{esc(v)}"')
print("\n".join(out))
PY
  )"
fi

echo "[orttrt-biomednlp-biomedbert] HOST=$HOST PORT=$PORT"
echo "[orttrt-biomednlp-biomedbert] MODEL_DIR=$MODEL_DIR"
echo "[orttrt-biomednlp-biomedbert] ONNX_PATH=${ONNX_PATH:-<auto>}"
echo "[orttrt-biomednlp-biomedbert] PROVIDER=$PROVIDER POOLING=$POOLING NORM=$NORM MAX_LENGTH=$MAX_LENGTH"
echo "[orttrt-biomednlp-biomedbert] API_KEY=${API_KEY:0:2}*** (masked)"

exec uvicorn app.main:app --host "$HOST" --port "$PORT"