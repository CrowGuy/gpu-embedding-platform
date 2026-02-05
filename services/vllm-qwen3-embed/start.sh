#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${VLLM_CONFIG:-/app/config/vllm.yaml}"

# Defaults (can be overwritten by config/env)
MODEL="${MODEL:-/models/Qwen3-Embedding-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embed-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-eslllm}"

DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

# Read YAML if exists (lightweight: only keys we care about)
if [[ -f "$CONFIG_FILE" ]]; then
  python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, "r")) or {}
# Print as shell exports to stdout; caller will eval it.
def emit(k, v):
  if v is None or v == "":
    return
  # escape double quotes
  s = str(v).replace('"', '\\"')
  print(f'export {k}="{s}"')

emit("MODEL", cfg.get("model"))
emit("SERVED_MODEL_NAME", cfg.get("served_model_name"))
emit("HOST", cfg.get("host"))
emit("PORT", cfg.get("port"))
emit("API_KEY", cfg.get("api_key"))

emit("DTYPE", cfg.get("dtype"))
emit("MAX_MODEL_LEN", cfg.get("max_model_len"))
emit("GPU_MEMORY_UTILIZATION", cfg.get("gpu_memory_utilization"))
emit("MAX_NUM_BATCHED_TOKENS", cfg.get("max_num_batched_tokens"))
emit("MAX_NUM_SEQS", cfg.get("max_num_seqs"))
PY
  # shellcheck disable=SC2046
  eval "$(python3 - <<'PY' "$CONFIG_FILE"
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path, "r")) or {}
def kv(k, v):
  if v is None or v == "":
    return ""
  s = str(v).replace('"', '\\"')
  return f'export {k}="{s}"\n'
out = ""
out += kv("MODEL", cfg.get("model"))
out += kv("SERVED_MODEL_NAME", cfg.get("served_model_name"))
out += kv("HOST", cfg.get("host"))
out += kv("PORT", cfg.get("port"))
out += kv("API_KEY", cfg.get("api_key"))
out += kv("DTYPE", cfg.get("dtype"))
out += kv("MAX_MODEL_LEN", cfg.get("max_model_len"))
out += kv("GPU_MEMORY_UTILIZATION", cfg.get("gpu_memory_utilization"))
out += kv("MAX_NUM_BATCHED_TOKENS", cfg.get("max_num_batched_tokens"))
out += kv("MAX_NUM_SEQS", cfg.get("max_num_seqs"))
print(out, end="")
PY
  )"
fi

echo "[vllm-qwen3-embed] MODEL=$MODEL"
echo "[vllm-qwen3-embed] SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "[vllm-qwen3-embed] HOST=$HOST PORT=$PORT"
echo "[vllm-qwen3-embed] DTYPE=$DTYPE"
echo "[vllm-qwen3-embed] API_KEY=${API_KEY:0:2}*** (masked)"

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --dtype "$DTYPE"
)

# Optional knobs (only append if set)
if [[ -n "$MAX_MODEL_LEN" ]]; then ARGS+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"); fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"); fi
if [[ -n "$MAX_NUM_SEQS" ]]; then ARGS+=(--max-num-seqs "$MAX_NUM_SEQS"); fi

# The vllm/vllm-openai image entrypoint supports passing args directly,
# but we call the api_server module explicitly to avoid entrypoint surprises.
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"