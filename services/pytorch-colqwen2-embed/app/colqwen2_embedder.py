from __future__ import annotations

import os
from typing import Any, List, Literal, Optional

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
    ColQwen2 embedder service implementation aligned with Byaldi's colpali.py behavior:

    - Load model via colpali_engine.models.ColQwen2
    - Load processor via colpali_engine.models.ColQwen2Processor
    - Encode text/images by:
        batch = processor.process_queries/process_images(...)
        batch = move to device + cast float tensors to model.dtype
        out = model(**batch)
        emb = extract tensor from out using common field names (Byaldi-style)

    Outputs:
      - output="multivector": [B, L, D] (late-interaction tokens)
      - output="pooled":      [B, D]    (mean pool over L)
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
            # byaldi imports from colpali_engine.models (same as you pasted)
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except Exception as e:
            raise RuntimeError(
                "colpali_engine import failed. Ensure 'colpali-engine' is installed. "
                f"Original error: {e!r}"
            )

        # Processor
        self.processor = ColQwen2Processor.from_pretrained(
            self.model_dir,
            token=os.environ.get("HF_TOKEN"),
        )

        # Model
        # byaldi uses torch_dtype=bfloat16 and device_map='cuda' (when cuda) for from_pretrained
        # We'll follow that pattern but still keep final .to(self.device) safety.
        device_map = "cuda" if (self.device_str == "cuda" and torch.cuda.is_available()) else None
        torch_dtype = self.dtype if device_map == "cuda" else None

        self.model = ColQwen2.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=os.environ.get("HF_TOKEN"),
        ).eval()

        # Resolve device + enforce dtype when needed
        if self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            # model may already be on cuda via device_map, but keep it robust:
            self.model.to(self.device)
            if self.dtype in (torch.float16, torch.bfloat16):
                self.model.to(dtype=self.dtype)
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.model.to(self.device)

        self._warmup_done = False

    def warmup(self) -> None:
        if self._warmup_done:
            return
        _ = self.encode_text(["warmup"], output="pooled", normalize=False)
        self._warmup_done = True

    def _move_batch(self, batch: dict) -> dict:
        """
        Byaldi-style:
          - move tensors to device
          - cast float tensors (fp16/bf16/fp32) to model.dtype
          - keep integer tensors (e.g., input_ids) as-is
        """
        out: dict = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    v = v.to(self.model.dtype)
            out[k] = v
        return out

    def _extract_emb_tensor(self, out: Any) -> torch.Tensor:
        """
        Byaldi-style extraction: model(**batch) may return Tensor or ModelOutput/dataclass.
        Try common field names used across models.
        """
        if isinstance(out, torch.Tensor):
            return out

        # dict-like
        if isinstance(out, dict):
            for name in [
                "embeddings",
                "text_embeds",
                "image_embeds",
                "query_embeds",
                "dense_embeddings",
                "pooler_output",
                "sentence_embedding",
                "last_hidden_state",
            ]:
                v = out.get(name)
                if isinstance(v, torch.Tensor):
                    return v

        # object-like
        for name in [
            "embeddings",
            "text_embeds",
            "image_embeds",
            "query_embeds",
            "dense_embeddings",
            "pooler_output",
            "sentence_embedding",
            "last_hidden_state",
        ]:
            v = getattr(out, name, None)
            if isinstance(v, torch.Tensor):
                return v

        # last-resort: helpful diagnostics
        if isinstance(out, dict):
            keys = list(out.keys())
        else:
            keys = [a for a in dir(out) if not a.startswith("_")]
        raise TypeError(
            f"Expected Tensor embeddings, got {type(out)}. "
            f"Available keys/attrs(sample): {keys[:60]}"
        )

    def _pool(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,L,D] -> [B,D] mask-aware mean pooling (mask optional)
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

        batch = self.processor.process_queries(texts)
        batch = self._move_batch(batch)

        out = self.model(**batch)
        emb = self._extract_emb_tensor(out)

        # Normalize to [B,L,D]
        if emb.ndim == 2:
            emb_mv = emb.unsqueeze(1)
        elif emb.ndim == 3:
            emb_mv = emb
        else:
            raise RuntimeError(f"Unexpected embeddings shape: {tuple(emb.shape)}")

        if output == "pooled":
            pooled = self._pool(emb_mv, mask=None)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            return pooled

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

        batch = self.processor.process_images(images)
        batch = self._move_batch(batch)

        out = self.model(**batch)
        emb = self._extract_emb_tensor(out)

        # Normalize to [B,L,D]
        if emb.ndim == 2:
            emb_mv = emb.unsqueeze(1)
        elif emb.ndim == 3:
            emb_mv = emb
        else:
            raise RuntimeError(f"Unexpected embeddings shape: {tuple(emb.shape)}")

        if output == "pooled":
            pooled = self._pool(emb_mv, mask=None)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            return pooled

        if normalize:
            emb_mv = F.normalize(emb_mv, p=2, dim=-1)
        return emb_mv
