from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

INFER_LATENCY = Histogram("biomedbert_infer_seconds", "Inference latency in seconds")
BATCH_SIZE = Histogram("biomedbert_batch_size", "Batch size", buckets=(1,2,4,8,16,32,64,128))

def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)