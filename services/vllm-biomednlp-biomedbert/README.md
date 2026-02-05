# vllm-biomednlp-biomedbert
Serve **biomednlp-biomedbert** (BioMedBERT/BioBERT-family encoder-only) as an OpenAI-compatible embeddings API using **vLLM**.

## Host paths
Recommended host layout:
```text
/nvme
├── docker
├── logs_recent
└── models_active
```

Expected model directory:
- Host: `/nvme/models_active/biomednlp-biomedbert`
- Container: `/models/biomednlp-biomedbert`

## Build
```bash
docker build -t vllm-biomednlp-biomedbert ./services/vllm-biomednlp-biomedbert
```

## Run
Uses port 8002 by default:
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8002:8002 \
  --name vllm-biomednlp-biomedbert \
  vllm-biomednlp-biomedbert
```

## API
OpenAI-compatible endpoint:
- `POST /v1/embeddings`
- Header: `Authorization: Bearer <API_KEY>`

Example:
```bash
curl http://localhost:8002/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biomednlp-biomedbert",
    "input": ["BRCA1 mutation and breast cancer", "immune checkpoint inhibitors"]
  }'
```

## Notes
- If you want to override defaults, either set env vars or mount a custom YAML to `/app/config/vllm.yaml`.
- Tune `max_num_seqs` / `max_num_batched_tokens` after benchmarking.