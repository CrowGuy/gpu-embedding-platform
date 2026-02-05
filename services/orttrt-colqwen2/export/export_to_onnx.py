"""
Placeholder exporter.

ColQwen2 export details depend on:
- whether you export vision encoder only
- whether you include text branch
- dynamic shapes (H,W, seq_len)
- output format (multi-vector vs pooled)

Start with a minimal vision-only ONNX graph, then iterate.
"""
print("TODO: implement ColQwen2 ONNX export once input/output specs are finalized.")