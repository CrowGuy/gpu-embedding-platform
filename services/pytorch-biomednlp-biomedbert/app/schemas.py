from pydantic import BaseModel, Field
from typing import List, Union, Optional, Literal, Any

class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    user: Optional[str] = None

class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Any

class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Usage = Field(default_factory=Usage)