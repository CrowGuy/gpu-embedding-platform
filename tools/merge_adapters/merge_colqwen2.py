import os
import argparse
import shutil
import torch

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModel
from peft import PeftModel

def try_copy(src, dst, names):
    os.makedirs(dst, exist_ok=True)
    for n in names:
        p = os.path.join(src, n)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # Load base
    base = AutoModel.from_pretrained(
        args.base_dir,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )

    # Load adapter on base
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model = model.merge_and_unload()  # <-- merge LoRA into base weights
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)

    # Copy config/tokenizer/processor assets from base/adapter if exist
    # tokenizer/processor 可能在 base 或 adapter，兩邊都試著 copy
    try:
        tok = AutoTokenizer.from_pretrained(args.base_dir, trust_remote_code=args.trust_remote_code)
        tok.save_pretrained(args.out_dir)
    except Exception:
        pass

    try:
        proc = AutoProcessor.from_pretrained(args.base_dir, trust_remote_code=args.trust_remote_code)
        proc.save_pretrained(args.out_dir)
    except Exception:
        # 有些模型 processor 在 adapter dir
        try:
            proc = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=args.trust_remote_code)
            proc.save_pretrained(args.out_dir)
        except Exception:
            pass

    # 額外把常見 processor/config 檔案補齊（以免 save_pretrained 沒帶到某些自訂檔）
    try_copy(args.base_dir, args.out_dir, [
        "config.json", "preprocessor_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt",
    ])
    try_copy(args.adapter_dir, args.out_dir, [
        "preprocessor_config.json", "processor_config.json",
    ])

    print(f"[OK] merged model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()