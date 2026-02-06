import os
from fastapi import FastAPI, Header, HTTPException
from .schemas import EmbeddingsRequest, EmbeddingsResponse, EmbeddingItem, Usage
from .model import OrtBiomedBertEmbedder
from .metrics import INFER_LATENCY, BATCH_SIZE, metrics_response

app = FastAPI(title="orttrt-biomednlp-biomedbert", version="0.1.0")
embedder = OrtBiomedBertEmbedder.from_env()

API_KEY = os.getenv("API_KEY", "eslllm")

@app.get("/healthz")
def healthz():
    return {"ok": True, "provider": embedder.provider, "model": embedder.model_id}

@app.get("/metrics")
def metrics():
    return metrics_response()

def _check_auth(authorization: str | None):
    # Expect: "Bearer <API_KEY>"
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
def embeddings(req: EmbeddingsRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input

    BATCH_SIZE.observe(len(texts))
    with INFER_LATENCY.time():
        vecs = embedder.embed_texts(texts)  # [B,D]

    # OpenAI response: data[i].embedding is list[float]
    data = []
    for i in range(vecs.shape[0]):
        data.append(EmbeddingItem(index=i, embedding=vecs[i].astype(float).tolist()))

    return EmbeddingsResponse(
        data=data,
        model=req.model,
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )