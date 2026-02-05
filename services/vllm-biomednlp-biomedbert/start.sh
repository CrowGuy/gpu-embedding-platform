#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${VLLM_CONFIG:-/app/config/vllm.yaml}"

MODEL="${MODEL:-/models/biomednlp-biomedbert}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-biomednlp-biomedbert}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
API_KEY="${API_KEY:-eslllm}"

DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

# Read YAML (if present) and export known keys
if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, "r")) or {}

allowed = {
  "MODEL": "model",
  "SERVED_MODEL_NAME": "served_model_name",
  "HOST": "host",
  "PORT": "port",
  "API_KEY": "api_key",
  "DTYPE": "dtype",
  "MAX_MODEL_LEN": "max_model_len",
  "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
  "MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
  "MAX_NUM_SEQS": "max_num_seqs",
}

def esc(v: str) -> str:
  return str(v).replace('"', '\\"').replace("\n", " ").strip()

out = []
for env_k, yaml_k in allowed.items():
  v = cfg.get(yaml_k)
  if v is None or v == "":
    continue
  out.append(f'export {env_k}="{esc(v)}"')
print("\n".join(out))
PY
  )"
fi

echo "[vllm-biomednlp-biomedbert] MODEL=$MODEL"
echo "[vllm-biomednlp-biomedbert] SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "[vllm-biomednlp-biomedbert] HOST=$HOST PORT=$PORT"
echo "[vllm-biomednlp-biomedbert] DTYPE=$DTYPE"
echo "[vllm-biomednlp-biomedbert] API_KEY=${API_KEY:0:2}*** (masked)"

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then ARGS+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"); fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"); fi
if [[ -n "$MAX_NUM_SEQS" ]]; then ARGS+=(--max-num-seqs "$MAX_NUM_SEQS"); fi

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"