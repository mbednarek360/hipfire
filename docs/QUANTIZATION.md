# Quantization Guide

## Formats

hipfire supports multiple quantization formats. The choice depends on your VRAM
budget and target throughput.

### Q8_0 (1.06 bytes/weight)
- Block: f16 scale (2B) + 32 int8 values (32B) = 34 bytes per 32 elements
- Symmetric: `scale = max(|w|) / 127`, `q = round(w / scale)`
- Dequant: `weight = scale * q` (one multiply per element)
- **Best throughput on RDNA**: 84% peak bandwidth (375 GB/s on RX 5700 XT)
- 16 VGPRs → maximum occupancy

### Q4_K (0.56 bytes/weight)
- GGML-compatible block: 144 bytes per 256 elements
- Hierarchical: super-block f16 scale/min + 8 sub-blocks with 6-bit integer scales
- Complex dequant: packed scale decode + nibble extraction + multiply + subtract
- **42% peak bandwidth** on RDNA1 due to 40 VGPRs from extraction chain
- Best compression ratio that hipfire supports natively

### Q4_F16_G64 (0.56 bytes/weight)
- Hipfire-native format: f16 scale + f16 min + 32 packed nibble bytes = 36 bytes per 64 elements
- Simpler metadata than Q4_K but same bandwidth (format doesn't determine speed — VGPRs do)
- Useful as a baseline for format experiments

### Mixed Q8+Q4_K (variable, ~0.7-0.8 bytes/weight)
- **Recommended for VRAM-constrained models**
- Attention weights (q/k/v/o projections): Q8_0 for maximum bandwidth
- FFN weights (gate/up/down): Q4_K for compression
- Embedding + lm_head: Q8_0
- Norms: F16

## Using hipfire-quantize

### Prerequisites
- HuggingFace model directory with `config.json` and `*.safetensors`
- Source weights should be FP16 or BF16 (**never quantize from pre-quantized GGUF**)

### Commands

```bash
# Full Q8 (best speed, most VRAM)
cargo run --release -p hipfire-quantize -- \
  --input /path/to/model-dir \
  --output model-q8.hfq \
  --format q8f16

# Mixed Q8+Q4_K (best speed/VRAM tradeoff) — RECOMMENDED
cargo run --release -p hipfire-quantize -- \
  --input /path/to/model-dir \
  --output model-mixed.hfq \
  --format q8-mixed

# Full Q4_F16 (maximum compression, slower)
cargo run --release -p hipfire-quantize -- \
  --input /path/to/model-dir \
  --output model-q4.hfq \
  --format q4f16

# Q8-fast: Q8 attention + Q4-as-Q8 FFN (all Q8 speed, larger file)
cargo run --release -p hipfire-quantize -- \
  --input /path/to/model-dir \
  --output model-fast.hfq \
  --format q8-fast
```

### VRAM Budget Guide

| Model Size | Q8 Full | Q8+Q4K Mixed | Q4_F16 | Fits 8GB? |
|-----------|---------|-------------|--------|-----------|
| 0.5-1B | ~0.6-1.2 GB | ~0.5-0.8 GB | ~0.3-0.6 GB | All fit |
| 3-4B | ~3.5-4.5 GB | ~2.5-3.5 GB | ~1.8-2.5 GB | All fit |
| 7-8B | ~8-9 GB | ~5.5-6.5 GB | ~4-5 GB | Mixed + Q4 fit |
| 13B+ | ~14+ GB | ~10+ GB | ~7-8 GB | Q4 only (tight) |

### Running Inference

```bash
# HFQ models
cargo run --release --example infer_hfq -- model.hfq "Your prompt here"

# GGUF models (also supported)
cargo run --release --example infer -- model.gguf "Your prompt here"
```

## Why Not Double-Quantize?

Converting Q4_K_M GGUF → F32 → Q4_F16 introduces compounding quantization error:
- Q4_K already maps each weight to one of 16 levels per sub-block
- Re-quantizing those 16 levels to a different 16-level scheme loses information
- The first few tokens may be correct, but generation quality degrades rapidly

Always quantize from the original FP16/BF16 weights. Download them from HuggingFace:

```bash
hf download Qwen/Qwen3-8B --include "*.safetensors" --local-dir ./qwen3-8b
hf download Qwen/Qwen3-8B config.json --local-dir ./qwen3-8b
```

## Format Details: .hfq

The `.hfq` (HipFire Quantized) file format:

```
Header (32 bytes):
  magic: "HFQM" (4B)
  version: u32
  architecture: u32 (0=LLaMA, 1=Qwen3, 2=Qwen3.5)
  n_tensors: u32
  metadata_offset: u64
  data_offset: u64

Metadata: JSON blob (model config + tokenizer reference)
Tensor Index: per-tensor name, quant_type, shape, data_size
Tensor Data: 4096-byte aligned, directly mmap-able

Quant types: 0=Q4_F16_G64, 1=F16, 2=F32, 3=Q8_FP16, 4=Q4_K
```
