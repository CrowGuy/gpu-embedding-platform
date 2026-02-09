from __future__ import annotations

import base64
import io
import os
from typing import Optional, List

import numpy as np
from PIL import Image
from fastapi import FastAPI, Header, HTTPException

from .schemas import (
    TextEmbeddingsRequest,
    ImageEmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingItem,
    HealthzResponse,
)
from .colqwen2_embedder import ColQwen2Embedder


app = FastAPI(title="pytorch-colqwen2-embed", version="0.1.0")

API_KEY = os.getenv("API_KEY", "eslllm")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "colqwen2-embed")

embedder = ColQwen2Embedder()
embedder.warmup()


def _check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _decode_b64_image(s: str) -> Image.Image:
    s = s.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def _tensor_to_items(model: str, t, output: str) -> EmbeddingsResponse:
    """
    pooled: [B,D] -> embedding: list[float]
    multivector: [B,L,D] -> embedding: list[list[float]]
    """
    t = t.detach().cpu().float()

    data: List[EmbeddingItem] = []
    if output == "pooled":
        arr = t.numpy()  # [B,D]
        for i in range(arr.shape[0]):
            data.append(EmbeddingItem(index=i, embedding=arr[i].tolist()))
    else:
        arr = t.numpy()  # [B,L,D]
        for i in range(arr.shape[0]):
            data.append(EmbeddingItem(index=i, embedding=arr[i].tolist()))

    return EmbeddingsResponse(model=model, data=data)


@app.get("/healthz", response_model=HealthzResponse)
async def healthz():
    return HealthzResponse(
        ok=True,
        model=SERVED_MODEL_NAME,
        device=str(embedder.device),
        dtype=str(embedder.dtype),
    )


@app.post("/v1/embeddings/text", response_model=EmbeddingsResponse)
async def embeddings_text(req: TextEmbeddingsRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    if req.model != SERVED_MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. This service serves '{SERVED_MODEL_NAME}'.",
        )

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input")

    normalize = req.normalize
    t = embedder.encode_text(texts, output=req.output, normalize=normalize)
    return _tensor_to_items(req.model, t, output=req.output)


@app.post("/v1/embeddings/image", response_model=EmbeddingsResponse)
async def embeddings_image(req: ImageEmbeddingsRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)

    if req.model != SERVED_MODEL_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. This service serves '{SERVED_MODEL_NAME}'.",
        )

    imgs = [req.images] if isinstance(req.images, str) else req.images
    if not imgs:
        raise HTTPException(status_code=400, detail="Empty images")

    pil_images = [_decode_b64_image(x) for x in imgs]

    normalize = req.normalize
    t = embedder.encode_images(pil_images, output=req.output, normalize=normalize)
    return _tensor_to_items(req.model, t, output=req.output)