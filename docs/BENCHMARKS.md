# hipfire Benchmarks

Hardware: AMD Radeon RX 5700 XT (8GB VRAM, RDNA1 gfx1010, 448 GB/s peak)
Branch: `restore-20ec9bb` (regression-free baseline)
Date: 2026-03-26

## Summary

| Model | Format | Size | Gen (short) | Gen (long) | Prefill |
|-------|--------|------|-------------|------------|---------|
| Qwen3-0.6B | HFQ4-G256 | 473M | **262.5 tok/s** | 235.0 | 1263 |
| Qwen3-0.6B | HFQ4 auto | 473M | 167–238 | varies | 1200–1370 |
| Qwen3-0.6B | Q8 | 775M | 219.0 | 184.2 | 358 |
| Qwen3-8B | HFQ4 auto | 4.4G | **58.1** | 54.7 | 110 |
| Qwen3-8B | HFQ4-G256 | 4.4G | **58.1** | 54.8 | 110 |

All output verified coherent. Kernel cache warm (cold compile excluded).

## Qwen3-0.6B Detail

### Q8 (default) — 775M

| Config | Gen tok/s | Prefill tok/s | Tokens | Quality |
|--------|-----------|---------------|--------|---------|
| Q8 defkv short | 219.0 | 358 | 254 | OK |
| Q8 defkv long | 184.2 | 363 | 935 | OK |
| Q8 fp32kv short | 117.6 | 32 | 254 | OK |

### HFQ4 (auto G128/G256) — 473M

| Config | Gen tok/s | Prefill tok/s | Tokens | Quality |
|--------|-----------|---------------|--------|---------|
| HFQ4 defkv short | 167.5 | 1200 | 2048 | OK |
| HFQ4 defkv long | 238.3 | 1370 | 567 | OK |
| HFQ4 fp32kv short | 183.5 | 511 | 328 | OK |

Note: HFQ4 auto picks G128 for small K (attention) and G256 for large K (FFN). Short prompt gen variance is from different token counts (model generates more with auto-mixed groups).

### HFQ4-G256 (forced) — 473M

| Config | Gen tok/s | Prefill tok/s | Tokens | Quality |
|--------|-----------|---------------|--------|---------|
| HFQ4G256 defkv short | 262.5 | 1263 | 283 | OK |
| HFQ4G256 defkv long | 235.0 | 1370 | 612 | OK |

HFQ4-G256 is the fastest config for generation on 0.6B.

## Qwen3-8B Detail

### HFQ4 (auto) — 4.4G

| Config | Gen tok/s | Prefill tok/s | Tokens | Quality |
|--------|-----------|---------------|--------|---------|
| 8B HFQ4 defkv short | 58.1 | 110 | 292 | OK |
| 8B HFQ4 defkv long | 54.7 | 109 | 920 | OK |

### HFQ4-G256 (forced) — 4.4G

| Config | Gen tok/s | Prefill tok/s | Tokens | Quality |
|--------|-----------|---------------|--------|---------|
| 8B HFQ4G256 defkv short | 58.1 | 110 | 293 | OK |
| 8B HFQ4G256 defkv long | 54.8 | 109 | 920 | OK |

8B is memory-bandwidth bound — HFQ4 auto and G256 perform identically.

## Notes

- **defkv**: Q8_0 quantized KV cache (default, 3.76x compression vs FP32)
- **fp32kv**: Unquantized FP32 KV cache (slower, for debugging)
- **short prompt**: "The meaning of life is" (philosophical, ~250–300 gen tokens)
- **long prompt**: Compiler vs interpreter explanation (technical, ~600–900 gen tokens)
- Sampling: greedy (temp=0), ChatML auto-detected
- HFQ4 embeds use Q8 (prevents quality loss on large-dim models)
