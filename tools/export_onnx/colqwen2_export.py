#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoProcessor, AutoModel


def parse_args():
    p = argparse.ArgumentParser(description="Export ColQwen2-like multimodal embedding model to ONNX.")
    p.add_argument("--model-dir", required=True, help="Local HF model directory (downloaded files).")
    p.add_argument("--out-dir", required=True, help="Output directory for ONNX.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--image-size", type=int, default=448, help="Dummy image size for export (square).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fp16", action="store_true", help="Export with fp16 weights/inputs (cuda recommended).")
    p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for custom models.")
    p.add_argument("--output-mode", default="auto", choices=["auto", "embeddings", "pooler", "cls", "mean"],
                   help="How to select embeddings from model outputs.")
    return p.parse_args()


def make_dummy_image(image_size: int) -> Image.Image:
    # simple black image
    return Image.new("RGB", (image_size, image_size), (0, 0, 0))


def select_embedding(output: Any, mode: str = "auto", attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Tries to extract a single tensor [B, D] from model outputs.
    - output can be ModelOutput / dict / tuple / tensor
    - mode can force selection
    """
    # Helper: get field if exists
    def get_field(obj, name: str):
        if hasattr(obj, name):
            v = getattr(obj, name)
            if v is not None:
                return v
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        return None

    # 1) Direct embeddings field
    if mode in ("auto", "embeddings"):
        v = get_field(output, "embeddings") or get_field(output, "embedding")
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return v

    # 2) pooler_output
    if mode in ("auto", "pooler"):
        v = get_field(output, "pooler_output")
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return v

    # 3) last_hidden_state -> CLS or mean pool
    lhs = get_field(output, "last_hidden_state")
    if isinstance(lhs, torch.Tensor) and lhs.ndim == 3:
        if mode == "cls":
            return lhs[:, 0, :]
        # mean (masked) if mask provided
        if attention_mask is not None and attention_mask.ndim == 2:
            m = attention_mask.to(lhs.dtype).unsqueeze(-1)  # [B,L,1]
            summed = (lhs * m).sum(dim=1)
            denom = m.sum(dim=1).clamp_min(1e-6)
            return summed / denom
        # fallback mean
        return lhs.mean(dim=1)

    # 4) Tuple fallback: first tensor
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                # If it's [B,L,D], reduce to [B,D]
                if item.ndim == 3:
                    return item[:, 0, :]
                if item.ndim == 2:
                    return item
                if item.ndim == 1:
                    return item.unsqueeze(0)
    # 5) Tensor output fallback
    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            return output[:, 0, :]
        if output.ndim == 2:
            return output

    raise RuntimeError("Unable to select embedding tensor from model output. "
                       "Adjust select_embedding() for this model.")


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values: torch.Tensor):
        # Many multimodal models accept these keys; some require extra keys.
        # If your model needs more (e.g., image_mask), you can extend here.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        emb = select_embedding(outputs, mode=self.mode, attention_mask=attention_mask)
        return emb


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        print("[WARN] --fp16 requested but device is cpu. Ignoring fp16.")
        args.fp16 = False

    print("[INFO] Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model.eval()

    if device.type == "cuda":
        model.to(device)
    if args.fp16:
        model.half()

    # Dummy inputs (1 sample)
    dummy_text = ["hello world"]
    dummy_img = [make_dummy_image(args.image_size)]

    batch = processor(
        text=dummy_text,
        images=dummy_img,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    # Expect keys
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise RuntimeError(f"Processor did not return input_ids/attention_mask. Keys: {list(batch.keys())}")
    # For images, different processors may use pixel_values or pixel_values_*; handle common name
    pixel_key = "pixel_values"
    if pixel_key not in batch:
        # try common alternatives
        for k in batch.keys():
            if "pixel_values" in k:
                pixel_key = k
                break
    if pixel_key not in batch:
        raise RuntimeError(f"Processor did not return pixel_values. Keys: {list(batch.keys())}")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch[pixel_key].to(device)

    if args.fp16:
        # token ids stay int64; images float16
        pixel_values = pixel_values.half()

    print("[INFO] Sanity forward...")
    wrapper = ExportWrapper(model, mode=args.output_mode)
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    print("[INFO] Output shape:", tuple(out.shape), "dtype:", out.dtype)

    # ONNX export
    onnx_path = os.path.join(args.out_dir, "model.onnx")
    print("[INFO] Exporting to:", onnx_path)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "embedding": {0: "batch"},
    }

    # Names must match wrapper.forward signature order
    input_names = ["input_ids", "attention_mask", "pixel_values"]
    output_names = ["embedding"]

    # torch.onnx.export wants a tuple of args
    export_args = (input_ids, attention_mask, pixel_values)

    torch.onnx.export(
        wrapper,
        export_args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print("[OK] Exported:", onnx_path)
    print("[INFO] Next: run ORT inference test on this ONNX before TRT EP.")


if __name__ == "__main__":
    main()