from __future__ import annotations

import os
from typing import Any, List, Optional, Literal

import torch
import torch.nn.functional as F
from PIL import Image


def _pick_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "bf16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


class ColQwen2Embedder:
    """
    ColQwen2 embedder using colpali_engine reference implementation.

    Exposes:
      - encode_text(texts, output="pooled"|"multivector")
      - encode_images(images, output="pooled"|"multivector")

    Notes:
      - output="multivector" is the natural format for ColQwen2 (late-interaction).
      - output="pooled" is provided for convenience (mean pooling across tokens/patches).
    """

    def __init__(self):
        self.model_dir = os.getenv("MODEL_DIR", "/models/ColQwen2-v1.0-merged")
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "colqwen2-embed")

        self.device_str = os.getenv("DEVICE", "cuda").lower()
        self.dtype = _pick_dtype(os.getenv("DTYPE", "bf16"))
        self.max_length = int(os.getenv("MAX_LENGTH", "128"))
        self.normalize_default = os.getenv("NORMALIZE", "1") == "1"

        # --- MUST use colpali_engine ---
        try:
            from colpali_engine.models import ColQwen2
            from colpali_engine.processors import ColQwen2Processor
        except Exception as e:
            raise RuntimeError(
                "colpali_engine import failed. Make sure you installed 'colpali-engine' "
                "inside the image. Original error: " + repr(e)
            )

        self.processor = ColQwen2Processor.from_pretrained(self.model_dir)

        # Model load (colpali_engine controls correct architecture)
        self.model = ColQwen2.from_pretrained(self.model_dir)
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
        _ = self.encode_text(["warmup"], output="pooled", normalize=False)
        self._warmup_done = True

    def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,L,D] -> [B,D] mask-aware mean pooling
        """
        if x.ndim == 2:
            return x
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected embedding shape: {tuple(x.shape)}")

        if mask is None:
            return x.mean(dim=1)

        mask = mask.unsqueeze(-1).to(dtype=x.dtype)  # [B,L,1]
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    @torch.inference_mode()
    def encode_text(
        self,
        texts: List[str],
        output: Literal["pooled", "multivector"] = "pooled",
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        if normalize is None:
            normalize = self.normalize_default

        # colpali_engine processor should provide an API similar to:
        #   batch = processor.process_queries(texts) or processor(text=..., return_tensors="pt")
        # We support both patterns via try/fallback.
        batch = None
        if hasattr(self.processor, "process_queries"):
            batch = self.processor.process_queries(texts)
        else:
            batch = self.processor(text=texts, images=None, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = self.model(**batch)

        # colpali_engine commonly returns an object/dict that contains embeddings; fallback robustly:
        if isinstance(out, dict):
            emb = out.get("embeddings") or out.get("embedding") or out.get("last_hidden_state")
        else:
            emb = getattr(out, "embeddings", None) or getattr(out, "embedding", None) or getattr(out, "last_hidden_state", None)

        if emb is None or not isinstance(emb, torch.Tensor):
            raise RuntimeError("Cannot find embeddings tensor in ColQwen2 outputs. "
                               "Please inspect output keys/attrs and adjust extraction.")

        # Ensure multivector shape [B,L,D]
        if emb.ndim == 2:
            emb_mv = emb.unsqueeze(1)
        elif emb.ndim == 3:
            emb_mv = emb
        else:
            raise RuntimeError(f"Unexpected embeddings ndim={emb.ndim}, shape={tuple(emb.shape)}")

        if output == "pooled":
            pooled = self._pool(emb_mv, mask=batch.get("attention_mask"))
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            return pooled
        else:
            if normalize:
                emb_mv = F.normalize(emb_mv, p=2, dim=-1)
            return emb_mv

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Image.Image],
        output: Literal["pooled", "multivector"] = "pooled",
        normalize: Optional[bool] = None,
    ) -> torch.Tensor:
        if normalize is None:
            normalize = self.normalize_default

        # similar to text: prefer colpali_engine process_images if present
        batch = None
        if hasattr(self.processor, "process_images"):
            batch = self.processor.process_images(images)
        else:
            texts = [""] * len(images)
            batch = self.processor(text=texts, images=images, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = self.model(**batch)

        if isinstance(out, dict):
            emb = out.get("embeddings") or out.get("embedding") or out.get("last_hidden_state")
        else:
            emb = getattr(out, "embeddings", None) or getattr(out, "embedding", None) or getattr(out, "last_hidden_state", None)

        if emb is None or not isinstance(emb, torch.Tensor):
            raise RuntimeError("Cannot find embeddings tensor in ColQwen2 outputs. "
                               "Please inspect output keys/attrs and adjust extraction.")

        if emb.ndim == 2:
            emb_mv = emb.unsqueeze(1)
        elif emb.ndim == 3:
            emb_mv = emb
        else:
            raise RuntimeError(f"Unexpected embeddings ndim={emb.ndim}, shape={tuple(emb.shape)}")

        if output == "pooled":
            pooled = self._pool(emb_mv, mask=batch.get("attention_mask"))
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            return pooled
        else:
            if normalize:
                emb_mv = F.normalize(emb_mv, p=2, dim=-1)
            return emb_mv
