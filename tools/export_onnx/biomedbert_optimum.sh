#!/usr/bin/env bash
set -euo pipefail

# BioMedBERT -> ONNX exporter (Optimum CLI)
#
# Usage:
#   tools/export_onnx/biomedbert_optimum.sh \
#     --model-dir /nvme/models_active/biomednlp-biomedbert \
#     --out-dir   /nvme/models_active/biomednlp-biomedbert/onnx \
#     --opset 17
#
# Notes:
# - Requires: python3, pip, optimum[onnxruntime], transformers, torch, onnx
# - Export task: feature-extraction (encoder embeddings)
# - Output: model.onnx (plus config files)

MODEL_DIR=""
OUT_DIR=""
OPSET="17"
FP16="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --opset) OPSET="$2"; shift 2;;
    --fp16) FP16="1"; shift 1;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[FATAL] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_DIR}" ]]; then
  echo "[FATAL] --model-dir is required" >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${MODEL_DIR%/}/onnx"
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[FATAL] model dir not found: ${MODEL_DIR}" >&2
  exit 2
fi

if [[ ! -f "${MODEL_DIR%/}/config.json" ]]; then
  echo "[FATAL] ${MODEL_DIR%/}/config.json not found (is this a HF model dir?)" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] Model dir : ${MODEL_DIR}"
echo "[INFO] Out dir   : ${OUT_DIR}"
echo "[INFO] Opset     : ${OPSET}"
echo "[INFO] FP16      : ${FP16}"

# Ensure optimum-cli exists
if ! command -v optimum-cli >/dev/null 2>&1; then
  echo "[WARN] optimum-cli not found. Installing into user site-packages..."
  python3 -m pip install --user -U "pip>=23" >/dev/null
  python3 -m pip install --user -U "optimum[onnxruntime]" "transformers" "torch" "onnx" >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v optimum-cli >/dev/null 2>&1; then
  echo "[FATAL] optimum-cli still not found in PATH after install." >&2
  exit 2
fi

# Optional: show architectures (helps debug)
python3 - <<PY "${MODEL_DIR%/}/config.json" || true
import json, sys
cfg=json.load(open(sys.argv[1]))
print("[INFO] architectures:", cfg.get("architectures"))
print("[INFO] model_type:", cfg.get("model_type"))
PY

FP16_FLAG=""
if [[ "${FP16}" == "1" ]]; then
  FP16_FLAG="--fp16"
fi

# Export ONNX (encoder feature extraction)
set -x
optimum-cli export onnx \
  --model "${MODEL_DIR}" \
  --task feature-extraction \
  --opset "${OPSET}" \
  ${FP16_FLAG} \
  "${OUT_DIR}"
set +x

# Validate outputs
echo "[INFO] Export completed. Listing ONNX files:"
ls -lah "${OUT_DIR}" | sed -n '1,120p'

ONNX_FILE=""
# Common names: model.onnx / encoder_model.onnx
if [[ -f "${OUT_DIR}/model.onnx" ]]; then
  ONNX_FILE="${OUT_DIR}/model.onnx"
else
  # pick first *.onnx
  ONNX_FILE="$(ls -1 "${OUT_DIR}"/*.onnx 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${ONNX_FILE}" || ! -f "${ONNX_FILE}" ]]; then
  echo "[FATAL] No ONNX file produced under ${OUT_DIR}" >&2
  exit 2
fi

echo "[OK] ONNX ready: ${ONNX_FILE}"