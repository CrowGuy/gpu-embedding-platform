from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union, Optional

from PIL import Image
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# 你已確認 colqwen2-v1.0 用的是 ColQwen2Processor
from transformers import ColQwen2Processor


def _pick_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "fp16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


def _extract_embedding_any(output: Any) -> torch.Tensor:
    """
    Conservative extractor:
      - prefer output.embeddings
      - else pooler_output
      - else last_hidden_state (CLS)
    You can tighten this after first successful run & inspecting outputs.
    """
    if hasattr(output, "embeddings") and isinstance(output.embeddings, torch.Tensor):
        return output.embeddings

    if isinstance(output, dict):
        for k in ("embeddings", "embedding", "sentence_embedding", "pooler_output"):
            v = output.get(k)
            if isinstance(v, torch.Tensor):
                return v
        v = output.get("last_hidden_state")
        if isinstance(v, torch.Tensor):
            return v

    if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
        return output.pooler_output

    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    raise RuntimeError("Cannot extract embeddings from model outputs. Please adjust extractor.")


def _to_pooled(x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Turn [B,L,D] -> [B,D] with mask-aware mean pooling.
    If already [B,D], return as-is.
    """
    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise RuntimeError(f"Unexpected embedding shape: {tuple(x.shape)}")

    if attention_mask is None:
        return x.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)  # [B,L,1]
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


class ColQwen2Embedder:
    """
    Service-friendly ColQwen2 encoder.

    Supports:
      - encode_text(texts, output=pooled|multivector)
      - encode_images(images, output=pooled|multivector)

    output:
      - pooled:  [B,D]
      - multivector: [B,L,D] (we will return as list of [L,D] per item at API layer)
    """

    def __init__(self):
        self.model_dir = os.getenv("MODEL_DIR", "/models/Colqwen2-v1.0")
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "colqwen2-embed")

        self.device_str = os.getenv("DEVICE", "cuda").lower()
        self.dtype = _pick_dtype(os.getenv("DTYPE", "bf16"))  # 4090 建議先 bf16
        self.trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "1") == "1"

        self.max_length = int(os.getenv("MAX_LENGTH", "128"))
        self.normalize_default = os.getenv("NORMALIZE", "1") == "1"

        # Load processor
        self.processor = ColQwen2Processor.from_pretrained(
            self.model_dir, trust_remote_code=self.trust_remote_code
        )

        # Resolve model class via architectures (avoid AutoModel mapping issues)
        cfg = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=self.trust_remote_code)
        archs = getattr(cfg, "architectures", None) or []
        if not archs:
            raise RuntimeError("config.architectures is empty; cannot resolve ColQwen2 model class.")
        cls_name = archs[0]

        ModelCls = get_class_from_dynamic_module(
            class_reference=cls_name,
            pretrained_model_name_or_path=self.model_dir,
        )
        self.model = ModelCls.from_pretrained(self.model_dir, trust_remote_code=self.trust_remote_code)
        self.model.eval()

        if self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
            if self.dtype in (torch.float16, torch.bfloat16):
                self.model.to(dtype=self.dtype)
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.model.to(self.device)

        self._warmup_done = False

    def warmup(self):
        if self._warmup_done:
            return
        # lightweight warmup (text only)
        _ = self.encode_text(["warmup"], output="pooled", normalize=False)
        self._warmup_done = True

    @torch.inference_mode()
    def encode_text(
        self,
        texts: List[str],
        output: str = "pooled",
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        if normalize is None:
            normalize = self.normalize_default

        batch = self.processor(
            text=texts,
            images=None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out = self.model(**batch)
        emb_any = _extract_embedding_any(out)

        # Some models return [B,D], some [B,L,D]
        if output == "pooled":
            emb = _to_pooled(emb_any, attention_mask=batch.get("attention_mask"))
        else:
            # multivector: ensure [B,L,D]
            if emb_any.ndim == 2:
                # promote to [B,1,D]
                emb = emb_any.unsqueeze(1)
            elif emb_any.ndim == 3:
                emb = emb_any
            else:
                raise RuntimeError(f"Unexpected multivector shape: {tuple(emb_any.shape)}")

        if normalize:
            # normalize last dim
            emb = F.normalize(emb, p=2, dim=-1)

        return emb

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Image.Image],
        output: str = "pooled",
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        if normalize is None:
            normalize = self.normalize_default

        # Placeholder text (many multimodal processors expect some text field)
        texts = [""] * len(images)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        out = self.model(**batch)
        emb_any = _extract_embedding_any(out)

        if output == "pooled":
            emb = _to_pooled(emb_any, attention_mask=batch.get("attention_mask"))
        else:
            if emb_any.ndim == 2:
                emb = emb_any.unsqueeze(1)
            elif emb_any.ndim == 3:
                emb = emb_any
            else:
                raise RuntimeError(f"Unexpected multivector shape: {tuple(emb_any.shape)}")

        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb