#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${SERVICE_CONFIG:-/app/config/service.yaml}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

MODEL_DIR="${MODEL_DIR:-/models/colqwen2-v1.0}"
ONNX_PATH="${ONNX_PATH:-}"              # optional override
PROVIDER="${PROVIDER:-tensorrt}"        # tensorrt|cuda|cpu
TRT_FP16="${TRT_FP16:-1}"
TRT_ENGINE_CACHE="${TRT_ENGINE_CACHE:-/cache/trt_engines}"
TRT_TIMING_CACHE="${TRT_TIMING_CACHE:-/cache/trt_timing.cache}"

POOLING="${POOLING:-none}"              # none|cls|mean
NORM="${NORM:-1}"                       # 1 normalize output

if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r")) or {}
allowed = {
  "HOST":"host","PORT":"port",
  "MODEL_DIR":"model_dir","ONNX_PATH":"onnx_path",
  "PROVIDER":"provider",
  "TRT_FP16":"trt_fp16",
  "TRT_ENGINE_CACHE":"trt_engine_cache",
  "TRT_TIMING_CACHE":"trt_timing_cache",
  "POOLING":"pooling","NORM":"norm",
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

echo "[orttrt-colqwen2] HOST=$HOST PORT=$PORT"
echo "[orttrt-colqwen2] MODEL_DIR=$MODEL_DIR"
echo "[orttrt-colqwen2] ONNX_PATH=${ONNX_PATH:-<auto>}"
echo "[orttrt-colqwen2] PROVIDER=$PROVIDER (TRT_FP16=$TRT_FP16)"
echo "[orttrt-colqwen2] POOLING=$POOLING NORM=$NORM"

exec uvicorn app.main:app --host "$HOST" --port "$PORT"