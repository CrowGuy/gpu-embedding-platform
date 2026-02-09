import os
import time
import asyncio
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from .schemas import EmbeddingsRequest, EmbeddingsResponse, EmbeddingItem, Usage
from .metrics import REQ_TOTAL, INFER_LATENCY, BATCH_SIZE, metrics_response
from .model import BioMedBertEmbedder

app = FastAPI(title="pytorch-biomednlp-biomedbert", version="0.1.0")

API_KEY = os.getenv("API_KEY", "eslllm")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "biomednlp-biomedbert")

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
BATCH_WAIT_MS = int(os.getenv("BATCH_WAIT_MS", "5"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "64"))

embedder = BioMedBertEmbedder()

# ---- auth ----
def _check_auth(authorization: Optional[str]):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ---- micro-batching queue ----
class _Job:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.fut: asyncio.Future = asyncio.get_event_loop().create_future()

_queue: List[_Job] = []
_queue_lock = asyncio.Lock()
_worker_started = False
_conc_sem = asyncio.Semaphore(MAX_CONCURRENCY)

async def _batch_worker():
    global _queue
    while True:
        await asyncio.sleep(BATCH_WAIT_MS / 1000.0)

        async with _queue_lock:
            if not _queue:
                continue
            # pop up to MAX_BATCH_SIZE (but keep job boundaries)
            jobs = []
            total = 0
            while _queue and total < MAX_BATCH_SIZE:
                j = _queue[0]
                if total + len(j.texts) > MAX_BATCH_SIZE and jobs:
                    break
                jobs.append(_queue.pop(0))
                total += len(j.texts)

        # run one batched forward
        try:
            flat = []
            offsets = []
            idx = 0
            for j in jobs:
                offsets.append((idx, idx + len(j.texts)))
                flat.extend(j.texts)
                idx += len(j.texts)

            BATCH_SIZE.observe(len(flat))
            t0 = time.time()
            vecs = await asyncio.to_thread(embedder.embed_texts, flat)  # [B,D] torch tensor
            INFER_LATENCY.observe(time.time() - t0)

            vecs = vecs.detach().cpu().float().numpy()

            for (j, (a, b)) in zip(jobs, offsets):
                if not j.fut.done():
                    j.fut.set_result(vecs[a:b])
        except Exception as e:
            for j in jobs:
                if not j.fut.done():
                    j.fut.set_exception(e)

async def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    asyncio.create_task(_batch_worker())

@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": SERVED_MODEL_NAME, "device": str(embedder._device)}

@app.get("/metrics")
def metrics():
    return metrics_response()

@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def embeddings(req: EmbeddingsRequest, authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    await _ensure_worker()
    REQ_TOTAL.inc()

    if req.model != SERVED_MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. This service serves '{SERVED_MODEL_NAME}'.")

    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        raise HTTPException(status_code=400, detail="Empty input")

    async with _conc_sem:
        job = _Job(texts)
        async with _queue_lock:
            _queue.append(job)
        vecs = await job.fut  # numpy [B,D]

    data = []
    for i in range(vecs.shape[0]):
        data.append(EmbeddingItem(index=i, embedding=vecs[i].tolist()))

    return EmbeddingsResponse(
        data=data,
        model=req.model,
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )