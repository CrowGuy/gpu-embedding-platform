import os
import glob
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def mean_pool(last_hidden: np.ndarray, attn_mask: np.ndarray) -> np.ndarray:
    # last_hidden: [B, L, D], attn_mask: [B, L]
    mask = attn_mask.astype(np.float32)[:, :, None]  # [B,L,1]
    summed = (last_hidden * mask).sum(axis=1)        # [B,D]
    denom = np.maximum(mask.sum(axis=1), 1e-6)       # [B,1]
    return summed / denom

class OrtBiomedBertEmbedder:
    def __init__(self, model_id: str, sess: ort.InferenceSession, tokenizer, pooling: str, norm: bool, max_length: int, provider: str):
        self.model_id = model_id
        self.sess = sess
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.norm = norm
        self.max_length = max_length
        self.provider = provider

        self.input_names = {i.name for i in sess.get_inputs()}
        self.output_names = [o.name for o in sess.get_outputs()]

    @staticmethod
    def _find_onnx(model_dir: str) -> str:
        cands = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if not cands:
            raise FileNotFoundError(f"No .onnx found under {model_dir}")
        return cands[0]

    @staticmethod
    def _make_session(onnx_path: str, provider: str) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        provider = provider.lower()
        if provider == "tensorrt":
            trt_fp16 = os.getenv("TRT_FP16", "1") == "1"
            trt_engine_cache = os.getenv("TRT_ENGINE_CACHE", "/cache/trt_engines")
            trt_timing_cache = os.getenv("TRT_TIMING_CACHE", "/cache/trt_timing.cache")
            os.makedirs(trt_engine_cache, exist_ok=True)
            os.makedirs(os.path.dirname(trt_timing_cache), exist_ok=True)

            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [
                {
                    "trt_fp16_enable": int(trt_fp16),
                    "trt_engine_cache_enable": 1,
                    "trt_engine_cache_path": trt_engine_cache,
                    "trt_timing_cache_enable": 1,
                    "trt_timing_cache_path": trt_timing_cache,
                },
                {},
                {},
            ]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers, provider_options=provider_options)

    @classmethod
    def from_env(cls) -> "OrtBiomedBertEmbedder":
        model_dir = os.getenv("MODEL_DIR", "/models/biomednlp-biomedbert")
        onnx_path = os.getenv("ONNX_PATH", "").strip() or cls._find_onnx(model_dir)

        pooling = os.getenv("POOLING", "mean").lower()
        norm = os.getenv("NORM", "1") == "1"
        max_length = int(os.getenv("MAX_LENGTH", "256"))
        provider = os.getenv("PROVIDER", "cuda").lower()

        # Tokenizer files should be in model_dir (HF layout)
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        sess = cls._make_session(onnx_path, provider=provider)
        model_id = os.path.basename(model_dir.rstrip("/"))
        return cls(model_id=model_id, sess=sess, tokenizer=tok, pooling=pooling, norm=norm, max_length=max_length, provider=provider)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        feeds = {}
        # Typical BERT ONNX expects input_ids, attention_mask, token_type_ids (optional)
        if "input_ids" in self.input_names:
            feeds["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in self.input_names and "token_type_ids" in enc:
            feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        outs = self.sess.run(None, feeds)
        x = np.asarray(outs[0])  # often last_hidden_state: [B,L,D] or pooled [B,D]

        if x.ndim == 2:
            pooled = x
        else:
            if self.pooling == "cls":
                pooled = x[:, 0, :]
            else:
                pooled = mean_pool(x, enc["attention_mask"])

        if self.norm:
            pooled = l2_normalize(pooled, axis=-1)

        return pooled