# orttrt-colqwen2
Serve ColQwen2 embeddings using ONNX Runtime + TensorRT EP (with CUDA/CPU fallback).

## Host paths
```text
/nvme
├── docker
├── logs_recent
└── models_active
```

Expected directories:
- Host model dir: `/nvme/models_active/colqwen2-1.0`
- Container model dir: `/models/colqwen2-1.0`
- Host cache dir (recommended): `/nvme/docker/ort_cache` -> `/cache`

## Build
```bash
docker build -t orttrt-colqwen2 ./services/orttrt-colqwen2
```

## Run
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -v /nvme/docker/ort_cache:/cache \
  -p 8010:8010 \
  --name orttrt-colqwen2 \
  orttrt-colqwen2
```

## API
- `POST /embed` (image b64)
- `GET /healthz`
- `GET /metrics`

Example:
```bash
python3 - <<'PY'
import base64, requests
from pathlib import Path

img = Path("test.jpg").read_bytes()
payload = {
  "model": "colqwen2",
  "image_b64": base64.b64encode(img).decode("utf-8"),
  "pooling": "mean",
  "norm": True,
}
r = requests.post("http://localhost:8010/embed", json=payload, timeout=60)
print(r.status_code)
print(str(r.json())[:500])
PY
```

## Notes
- TensorRT EP is best-effort; if it cannot initialize, service falls back to CUDA EP, then CPU.
- Replace `app/preprocess.py` with the official ColQwen2 processor behavior once export spec is finalized.

---
## 你在 host 上的路徑對應（照你 /nvme 的規劃）
- 模型：`/nvme/models_active/colqwen2-1.0`  → container `/models/colqwen2-1.0`
- ORT/TRT cache：`/nvme/docker/ort_cache` → container `/cache`
  - TRT engine cache 很重要，不然每次啟動都要重新 build engine

---

## 下一步：我們要先釐清「你要 export 哪一段」
為了讓 `orttrt-colqwen2` 真正跑起來，接下來最關鍵是 ONNX graph 的 I/O：

1) **你要 image-only embedding 嗎？**（最簡，先做這個）
2) 還是 **image+text joint embedding**？
3) 輸出要 **multi-vector（L×D）** 還是 **pooled（D）**？

我建議按順序做：
- **Phase 1：image-only + pooled(mean) 跑通**（最小可用，便於壓測/觀測）
- Phase 2：multi-vector output（支援 late interaction）
- Phase 3：image+text（若你真的需要）

你如果把你現在 ColQwen2 在 PyTorch/HF 下 forward 的 **輸入 tensor keys / output tensor shape**（貼個簡短 code snippet 或 `print(model(**inputs).shape)` 的結果）給我，我就可以把 export 腳本跟 `model.py` 的 feeds/output 對齊到「可直接跑」。
::contentReference[oaicite:0]{index=0}