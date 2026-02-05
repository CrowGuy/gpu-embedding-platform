Serve **Qwen3-Embedding-8B** as an OpenAI-compatible embeddings API using **vLLM**.

## Host paths (recommended)

You said you manage host storage like this:
```text
/nvme
├── docker
├── logs_recent
└── models_active
```
This service assumes:

- Host model path: `/nvme/models_active/Qwen3-Embedding-8B`
- Container model path: `/models/Qwen3-Embedding-8B`

## Build
From repo root:
```bash
docker build -t vllm-qwen3-embed ./services/vllm-qwen3-embed
```

## Run (equivalent to your current docker run)
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8000:8000 \
  --name vllm-qwen3-8b-embedding \
  vllm-qwen3-embed
```
### Override config (optional)
By default it reads /app/config/vllm.yaml inside the image.
You can override with env vars:
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8000:8000 \
  -e API_KEY=eslllm \
  -e SERVED_MODEL_NAME=qwen3-embed-8b \
  --name vllm-qwen3-8b-embedding \
  vllm-qwen3-embed
```
Or mount your own config:
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -v $(pwd)/services/vllm-qwen3-embed/config/vllm.yaml:/app/config/vllm.yaml \
  -p 8000:8000 \
  --name vllm-qwen3-8b-embedding \
  vllm-qwen3-embed
```
### API
OpenAI-compatible endpoint:
- `POST /v1/embeddings`
- Use header: `Authorization: Bearer <API_KEY>`

Example:
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embed-8b",
    "input": ["hello world", "embedding service"]
  }'
```
### Notes
- `--ipc=host` is recommended for performance.
- Tune `gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens` after you run benchmarks.
---
