# pytorch-biomednlp-biomedbert

Serve BioMedBERT embeddings via PyTorch + FastAPI, OpenAI-compatible `/v1/embeddings`.

## Host paths (recommended)

```text
/nvme
├── docker
├── logs_recent
└── models_active
```

Expected model dir:
- Host: `/nvme/models_active/biomednlp-biomedbert`
- Container: `/models/biomednlp-biomedbert`

## Build
```bash
docker build -t pytorch-biomednlp-biomedbert ./services/pytorch-biomednlp-biomedbert
```

## Run
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8002:8002 \
  --name pytorch-biomednlp-biomedbert \
  pytorch-biomednlp-biomedbert
```
## API
- POST `/v1/embeddings`
- Header: `Authorization: Bearer <API_KEY>`

Example:
```bash
curl http://localhost:8002/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biomednlp-biomedbert",
    "input": ["hello world", "tumor suppressor gene"]
  }'
```
## Metrics
- GET `/metrics` (Prometheus format)
- GET `/healthz`

## Batching / Concurrency (MVP)
- `max_batch_size`: maximum total texts per batch
- `batch_wait_ms`: wait time to accumulate micro-batch
- `max_concurrency`: semaphore for concurrent requests

---

# Docker Run 指令（依你 /nvme 佈局）
那啟動建議用：

```bash
docker build -t pytorch-biomednlp-biomedbert ./services/pytorch-biomednlp-biomedbert

sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8002:8002 \
  --name pytorch-biomednlp-biomedbert \
  pytorch-biomednlp-biomedbert
```

Smoke test：
```bash
curl http://localhost:8002/healthz

curl http://localhost:8002/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{"model":"biomednlp-biomedbert","input":["hello world"]}'
```