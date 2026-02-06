# orttrt-biomednlp-biomedbert
Serve BioMedNLP BioMedBERT embeddings via ONNX Runtime (CUDA/TRT) with an OpenAI-compatible API.

## Host paths
```text
/nvme
├── docker
├── logs_recent
└── models_active
```
Expected model dir:
- Host: `/nvme/models_active/biomednlp-biomedbert`
- Container: `/models/biomednlp-biomedbert`

Optional cache:
- Host: `/nvme/docker/ort_cache`
- Container: `/cache`

## Build
```bash
docker build -t orttrt-biomednlp-biomedbert ./services/orttrt-biomednlp-biomedbert
```

## Run
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -v /nvme/docker/ort_cache:/cache \
  -p 8002:8002 \
  --name orttrt-biomednlp-biomedbert \
  orttrt-biomednlp-biomedbert
```

## API (OpenAI-compatible)
- `POST /v1/embeddings`
- Header: `Authorization: Bearer <API_KEY>`

Example:
```bash
curl http://localhost:8002/v1/embeddings \
  -H "Authorization: Bearer eslllm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biomednlp-biomednlp-biomedbert",
    "input": ["tumor suppressor gene", "antibody response in mice"]
  }'
```

## Notes
- This service requires an ONNX export under the model dir (e.g. `model.onnx`).
- Start with `provider: cuda`. Switch to `tensorrt` after correctness is verified.

---

# 下一個關鍵：你現在有 BioMedBERT 的 ONNX 嗎？
這個服務會自動找 `MODEL_DIR` 底下的 `*.onnx`。你如果目前只有 HF 權重（`pytorch_model.bin` / `safetensors`），那我們下一步就是：

1) 用 `optimum` 或自寫 export 腳本把它匯出成 ONNX  
2) 放到 `/nvme/models_active/biomednlp-biomedbert/model.onnx`  
3) 重新啟動 service 走 ORT CUDA

如果你要我直接給你「export ONNX 的腳本」（含 dynamic axes、mean pooling 最合理的輸出），我可以照你這顆 checkpoint（你用的實際 repo 名稱/路徑）寫一份可以直接跑的。
::contentReference[oaicite:0]{index=0}