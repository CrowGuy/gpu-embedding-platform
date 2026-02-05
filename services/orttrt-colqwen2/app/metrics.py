from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

INFER_LATENCY = Histogram("colqwen2_infer_seconds", "Inference latency in seconds")

def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)