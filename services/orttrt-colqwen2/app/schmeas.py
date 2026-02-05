from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class EmbedRequest(BaseModel):
    model: Optional[str] = None

    # Base64-encoded image bytes (png/jpg). (No data URL prefix.)
    image_b64: str = Field(..., description="base64 of image bytes")

    # Optional text (for image+text)
    text: Optional[str] = None

    # Output control (overrides server default if provided)
    pooling: Optional[Literal["none", "cls", "mean"]] = None
    norm: Optional[bool] = None

class EmbedResponse(BaseModel):
    model: str
    pooling: str
    norm: bool

    # embeddings:
    # - if pooling != none: shape [D]
    # - if pooling == none: shape [L][D] (multi-vector)
    embeddings: List