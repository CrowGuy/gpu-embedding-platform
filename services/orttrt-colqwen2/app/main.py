from fastapi import FastAPI
from .schemas import EmbedRequest, EmbedResponse
from .model import OrtTrtEmbedder
from .metrics import metrics_response

app = FastAPI(title="orttrt-colqwen2", version="0.1.0")
metrics_app = setup_metrics(app)

embedder = OrtTrtEmbedder.from_env()

@app.get("/healthz")
def healthz():
    return {"ok": True, "provider": embedder.provider, "model": embedder.model_id}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    with INFER_LATENCY.time():
        vecs = embedder.embed(req)
    return EmbedResponse(
        model=req.model or embedder.model_id,
        pooling=embedder.pooling,
        norm=embedder.norm,
        embeddings=vecs,
    )

@app.get("/metrics")
def metrics():
    return metrics_response()