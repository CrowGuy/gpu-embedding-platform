# vllm-pubmedbert-embed

Serve **pubmedbert-base-embeddings** as an OpenAI-compatible embeddings API using **vLLM**.

## Host paths (recommended)

Host storage layout:
```text
/nvme
├── docker
├── logs_recent
└── models_active
```
Expected model dir:
- Host: `/nvme/models_active/pubmedbert-base-embeddings`
- Container: `/models/pubmedbert-base-embeddings`

## Build
```bash
docker build -t vllm-pubmedbert-embed ./services/vllm-pubmedbert-embed
```

## Run
Use a different port from the Qwen3 service (e.g. 8001):
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8001:8001 \
  --name vllm-pubmedbert-embed \
  vllm-pubmedbert-embed
```

## API
OpenAI-compatible endpoint:

- `POST /v1/embeddings`
- Header: `Authorization: Bearer <API_KEY>`

Example:
```bash
curl http://localhost:8001/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pubmedbert-embed",
    "input": ["tumor suppressor gene", "antibody response in mice"]
  }'
```

## Notes
- Tune `max_num_seqs` / `max_num_batched_tokens` after benchmarking.
- If you want to override config, set env vars or mount a custom YAML to `/app/config/vllm.yaml`.