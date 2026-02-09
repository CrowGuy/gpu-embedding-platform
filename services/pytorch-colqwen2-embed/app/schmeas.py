rom __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union, Literal


# -----------------------
# Common response schema
# -----------------------
class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Any  # list[float] or list[list[float]] when multivector


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


# -----------------------
# Text endpoint
# -----------------------
class TextEmbeddingsRequest(BaseModel):
    """
    POST /v1/embeddings/text

    output:
      - "pooled": returns a single vector per input (OpenAI-like)
      - "multivector": returns list of vectors per input (late-interaction friendly)
    """
    model: str
    input: Union[str, List[str]]
    output: Literal["pooled", "multivector"] = "pooled"
    normalize: Optional[bool] = None  # if None -> service default


# -----------------------
# Image endpoint
# -----------------------
class ImageEmbeddingsRequest(BaseModel):
    """
    POST /v1/embeddings/image

    images:
      - base64 string (raw) OR data-url "data:image/png;base64,...."
    output:
      - "pooled": returns a single vector per image
      - "multivector": returns list of vectors per image (patch/token level)
    """
    model: str
    images: Union[str, List[str]]
    output: Literal["pooled", "multivector"] = "pooled"
    normalize: Optional[bool] = None  # if None -> service default


# -----------------------
# Health
# -----------------------
class HealthzResponse(BaseModel):
    ok: bool
    model: str
    device: str
    dtype: str