import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def _pick_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32

class BioMedBertEmbedder:
    def __init__(self):
        self.model_dir = os.getenv("MODEL_DIR", "/models/biomednlp-biomedbert")
        self.device = os.getenv("DEVICE", "cuda").lower()
        self.dtype = _pick_dtype(os.getenv("DTYPE", "fp16"))
        self.max_length = int(os.getenv("MAX_LENGTH", "256"))
        self.pooling = os.getenv("POOLING", "mean").lower()
        self.normalize = os.getenv("NORMALIZE", "1") == "1"

        # Load
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.model.eval()

        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
            # fp16/bf16 on GPU only
            if self.dtype in (torch.float16, torch.bfloat16):
                self.model.to(dtype=self.dtype)
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            # keep fp32 on cpu
            self.dtype = torch.float32
            self.model.to("cpu")

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        out = self.model(**enc)
        last_hidden = out.last_hidden_state  # [B, L, D]

        if self.pooling == "cls":
            pooled = last_hidden[:, 0, :]
        else:
            # mean pool with mask
            attn = enc.get("attention_mask")
            if attn is None:
                pooled = last_hidden.mean(dim=1)
            else:
                mask = attn.unsqueeze(-1).to(last_hidden.dtype)  # [B,L,1]
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1e-6)
                pooled = summed / denom

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled  # [B, D]