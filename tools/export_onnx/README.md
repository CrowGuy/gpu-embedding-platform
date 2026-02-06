# ONNX Export Tools

This folder contains reproducible exporters that convert Hugging Face checkpoints (downloaded locally)
into ONNX artifacts for ONNX Runtime (CUDA/TensorRT EP) serving.

## Host storage layout (recommended)
```text
/nvme
├── docker
├── logs_recent
└── models_active
```
We assume HF models are placed under:
- `/nvme/models_active/<MODEL_NAME>/` (Hugging Face local directory)
Exported ONNX artifacts will be written to:
- `/nvme/models_active/<MODEL_NAME>/onnx/`

---
## Prerequisites
### Option A (recommended): run exporters inside a Docker container
This avoids polluting the host Python environment.
You need:
- Docker
- NVIDIA driver + NVIDIA Container Toolkit (for GPU export / fp16 / quick tests)

### Option B: run on host Python
You need:
- Python 3.10+ (3.11 recommended)
- `pip install optimum[onnxruntime] transformers torch onnx pillow`

---
## Exporters
### 1) BioMedBERT / PubMedBERT-like (text encoder)
Exporter: `biomedbert_optimum.sh`
Method: `optimum-cli export onnx`
Task: `feature-extraction` (encoder embeddings)

**Usage**
```bash
bash tools/export_onnx/biomedbert_optimum.sh \
  --model-dir /nvme/models_active/biomednlp-biomedbert \
  --out-dir /nvme/models_active/biomednlp-biomedbert/onnx \
  --opset 17
```
**Output**
- `.../onnx/model.onnx` (or a *.onnx file created by Optimum)

**Notes**
- This path is intended for **BERT-family encoders**.
- vLLM does not support `BertForMaskedLM` as a serving architecture, so ORT is the preferred backend.

---
### 2) ColQwen2-like (multimodal embedding)
Exporter: `colqwen2_export.py`
Method: `torch.onnx.export` with a small wrapper that extracts an embedding tensor.

**Usage (GPU recommended)**
```bash
python3 tools/export_onnx/colqwen2_export.py \
  --model-dir /nvme/models_active/Colqwen2-1.0 \
  --out-dir /nvme/models_active/Colqwen2-1.0/onnx \
  --device cuda \
  --opset 17 \
  --max-length 128 \
  --image-size 448 \
  --trust-remote-code
```
**Output**
- `.../onnx/model.onnx`

**Notes**
- Multimodal models may require additional inputs (e.g. `image_mask`).
If export fails with missing keys, update `ExportWrapper.forward()` in the script to match the model’s forward signature.
- Always run a small ORT inference sanity check before enabling TensorRT EP.

---
## Wiring exported ONNX into ORT services
### Example: `orttrt-biomednlp-biomedbert`

In `services/orttrt-biomednlp-biomedbert/config/service.yaml`:
```yaml
model_dir: /models/biomednlp-biomedbert
onnx_path: /models/biomednlp-biomedbert/onnx/model.onnx
provider: cuda        # switch to tensorrt after correctness verified
pooling: mean
norm: 1
max_length: 256
```

Run service with host mount:
```bash
sudo docker run --rm --runtime nvidia --gpus all --ipc=host \
  -v /nvme/models_active:/models \
  -v /nvme/docker/ort_cache:/cache \
  -p 8002:8002 \
  --name orttrt-biomednlp-biomedbert \
  orttrt-biomednlp-biomedbert
```

---
## Troubleshooting
### `optimum-cli` not found

The shell script will try installing Optimum into ~/.local/ automatically.
If you prefer a venv, install manually:
```bash
pip install -U "optimum[onnxruntime]" transformers torch onnx
```

### Export succeeds but ORT service fails to load

Common causes:
- `onnx_path` points to the wrong file
- model dir is missing tokenizer files (`tokenizer.json`, `vocab.txt`, etc.)
- ORT CUDA EP unavailable inside container

### ColQwen2 export complains about missing inputs
This usually means the model’s processor/forward requires extra tensors beyond:
- `input_ids`, `attention_mask`, `pixel_values`
Inspect processor outputs and adjust wrapper accordingly.

---
## Versioning policy
- ONNX artifacts are not committed to Git.
- Keep them under `/nvme/models_active` and mount into containers at runtime.
- If you need reproducibility, pin exporter versions (torch/transformers/optimum) in your tooling container.