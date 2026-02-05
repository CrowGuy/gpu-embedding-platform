import os
import glob
import numpy as np
import onnxruntime as ort
from .schemas import EmbedRequest
from .preprocess import decode_image_b64_to_rgb, image_to_tensor

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

class OrtTrtEmbedder:
    def __init__(self, model_id: str, sess: ort.InferenceSession, provider: str, pooling: str, norm: bool):
        self.model_id = model_id
        self.sess = sess
        self.provider = provider
        self.pooling = pooling
        self.norm = norm

        self.input_names = [i.name for i in sess.get_inputs()]
        self.output_names = [o.name for o in sess.get_outputs()]

    @staticmethod
    def _find_onnx(model_dir: str) -> str:
        cands = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if not cands:
            raise FileNotFoundError(f"No .onnx found under {model_dir}")
        return cands[0]

    @classmethod
    def from_env(cls) -> "OrtTrtEmbedder":
        model_dir = os.getenv("MODEL_DIR", "/models/colqwen2-1.0")
        onnx_path = os.getenv("ONNX_PATH", "").strip() or cls._find_onnx(model_dir)

        provider = os.getenv("PROVIDER", "tensorrt").lower()
        pooling = os.getenv("POOLING", "none").lower()
        norm = os.getenv("NORM", "1") == "1"

        sess = cls._make_session(onnx_path, provider)
        model_id = os.path.basename(model_dir.rstrip("/"))
        return cls(model_id=model_id, sess=sess, provider=provider, pooling=pooling, norm=norm)

    @staticmethod
    def _make_session(onnx_path: str, provider: str) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        provider_options = []

        # TensorRT EP options (best-effort)
        if provider == "tensorrt":
            trt_fp16 = os.getenv("TRT_FP16", "1") == "1"
            trt_engine_cache = os.getenv("TRT_ENGINE_CACHE", "/cache/trt_engines")
            trt_timing_cache = os.getenv("TRT_TIMING_CACHE", "/cache/trt_timing.cache")

            os.makedirs(trt_engine_cache, exist_ok=True)
            os.makedirs(os.path.dirname(trt_timing_cache), exist_ok=True)

            providers.append("TensorrtExecutionProvider")
            provider_options.append({
                "trt_fp16_enable": int(trt_fp16),
                "trt_engine_cache_enable": 1,
                "trt_engine_cache_path": trt_engine_cache,
                "trt_timing_cache_enable": 1,
                "trt_timing_cache_path": trt_timing_cache,
            })

            # Fallback to CUDA if TRT can't init at runtime
            providers.append("CUDAExecutionProvider")
            provider_options.append({})

        elif provider == "cuda":
            providers = ["CUDAExecutionProvider"]
            provider_options = [{}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        try:
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers, provider_options=provider_options)
            return sess
        except Exception as e:
            # Hard fallback: CUDA -> CPU, TRT -> CUDA -> CPU
            if provider != "cpu":
                try:
                    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])
                    return sess
                except Exception:
                    pass
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
            return sess

    def embed(self, req: EmbedRequest):
        pooling = (req.pooling or self.pooling).lower()
        norm = self.norm if req.norm is None else bool(req.norm)

        img = decode_image_b64_to_rgb(req.image_b64)
        pixel_values = image_to_tensor(img)  # [1,3,H,W] float32

        feeds = {}
        # MVP assumes ONNX input includes image tensor name like "pixel_values"
        # If your exported graph uses different names, adjust here.
        if "pixel_values" in self.input_names:
            feeds["pixel_values"] = pixel_values
        else:
            # best-effort: use first input for image tensor
            feeds[self.input_names[0]] = pixel_values

        # Optional text input (only if graph supports it)
        if req.text is not None:
            # You will likely need tokenization -> input_ids/attention_mask
            # Leave as TODO until you finalize export format.
            pass

        outs = self.sess.run(None, feeds)
        x = outs[0]

        # x could be [1, L, D] or [1, D] depending on model/export
        x = np.asarray(x)

        if pooling != "none":
            if x.ndim == 3:  # [B,L,D]
                if pooling == "cls":
                    y = x[:, 0, :]
                else:  # mean
                    y = x.mean(axis=1)
            elif x.ndim == 2:
                y = x
            else:
                y = x.reshape((x.shape[0], -1))
            y = y[0]
            if norm:
                y = l2_normalize(y)
            return y.tolist()

        # multi-vector output
        if x.ndim == 3:
            mv = x[0]  # [L,D]
        elif x.ndim == 2:
            mv = x  # [L,D]?
        else:
            mv = x.reshape((-1, x.shape[-1]))
        if norm:
            mv = l2_normalize(mv, axis=-1)
        return mv.tolist()