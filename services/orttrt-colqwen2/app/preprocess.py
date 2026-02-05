import base64
import io
import numpy as np
from PIL import Image

def decode_image_b64_to_rgb(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img

def image_to_tensor(img: Image.Image, size: int = 448) -> np.ndarray:
    # MVP: simple resize + normalize to float32 NCHW
    img = img.resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    arr = np.expand_dims(arr, 0)                     # NCHW
    return arr