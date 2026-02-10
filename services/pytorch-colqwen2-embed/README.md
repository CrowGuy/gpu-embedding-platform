# pytorch-colqwen2-embed

PyTorch + FastAPI service for ColQwen2 embeddings.

This service exposes **two endpoints**:
- `POST /v1/embeddings/text`
- `POST /v1/embeddings/image`

Both endpoints support:
- `output="pooled"`: single vector per input (OpenAI-like)
- `output="multivector"`: list of vectors per input (late-interaction friendly)

## Merge models
```bash
python tools/merge_adapters/merge_colqwen2.py \
  --base_dir /nvme/models_active/colqwen2-base \
  --adapter_dir /nvme/models_active/colqwen2-v1.0 \
  --out_dir /nvme/models_active/colqwen2-v1.0-merged \
  --dtype bf16 \
  --trust_remote_code
```

## Host paths (recommended)

```text
/nvme
├── docker
├── logs_recent
└── models_active
```
Expected model dir:
- Host: `/nvme/models_active/colqwen2-v1.0-merged`
- Container: `/models/colqwen2-v1.0-merged`

## Build
```bash
docker build -t pytorch-colqwen2-embed ./services/pytorch-colqwen2-embed
```

## Run
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -p 8003:8003 \
  --name pytorch-colqwen2-embed \
  -e API_KEY=eslllm \
  -e MODEL_DIR=/models/colqwen2-v1.0-merged \
  -e SERVED_MODEL_NAME=colqwen2-embed \
  pytorch-colqwen2-embed
```

## Health
```bash
curl http://localhost:8003/healthz
```

## API: Text embeddings
### Request
`POST /v1/embeddings/text`
```json
{
  "model": "colqwen2-embed",
  "input": ["hello world", "foo bar"],
  "output": "pooled",
  "normalize": true
}
```
- `input`: string or list of strings
- `output`:
    - `"pooled"` -> `embedding: float[]`
    - `"multivector"` -> `embedding: float[][]`
- `normalize`: optional (default uses service env NORMALIZE)

### Curl
```bash
curl http://localhost:8003/v1/embeddings/text \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{"model":"colqwen2-embed","input":["hello world"],"output":"pooled"}'
```
## API: Image embeddings
### Request
`POST /v1/embeddings/image`
```json
{
  "model": "colqwen2-embed",
  "images": ["<base64-or-data-url>"],
  "output": "pooled",
  "normalize": true
}
```
- `images`: base64 string(s)
    - supports raw base64 OR data:image/png;base64,....
- `output`:
    - `"pooled"` -> `embedding: float[]`
    - `"multivector"` -> `embedding: float[][]`

### Curl
```bash
B64=$(python3 - <<'PY'
import base64
print(base64.b64encode(open("demo.png","rb").read()).decode())
PY
)

curl http://localhost:8003/v1/embeddings/image \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"colqwen2-embed\",\"images\":[\"$B64\"],\"output\":\"pooled\"}"
```
## Notes

- Default dtype is set to `bf16` (good for RTX 4090). You can override with:
    - `-e DTYPE=fp16` or `-e DTYPE=fp32`
- This MVP is synchronous per-request (no micro-batching yet).
    - After it runs stably, add per-endpoint micro-batching (text/image queues) and metrics.

---

## 下一步（我建議你做的順序）
1) 先 build/run，確認：
   - `/healthz` OK
   - text endpoint OK
2) 再測 image endpoint（base64）
3) 跑通後，我們再把這個 service 升級成跟 biomedbert 一樣的：
   - **雙 queue micro-batching（text/image 分開）**
   - metrics（latency/batch size/OOM error counter）
   - router 串接（model name -> upstream）