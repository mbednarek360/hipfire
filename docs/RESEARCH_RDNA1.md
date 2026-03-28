# hipfire RDNA1 Research Synthesis (2026-03-27)

Three parallel research threads, unified findings.

## The Master Constraint

**On gfx1010, VGPR count determines performance more than anything else.**

- 16 VGPRs → 84% bandwidth utilization (375 GB/s)
- 40 VGPRs → 42% bandwidth utilization (188 GB/s)
- This is why Q8 beats Q4 despite reading 2x more data

Every optimization must justify its VGPR budget.

## Immediate Wins (1 day each)

### 1. FWHT via `__shfl_xor` — 2-3x FWHT speedup, -8 VGPRs

Current: 7 `__syncthreads` passes through shared memory.
Proposed: 2 local passes (within each thread's 4 elements) + 5 `__shfl_xor` passes (wave-level, zero barriers).

```
// Each thread owns 4 of 128 elements
// Pass k: partner = __shfl_xor(val, 1 << k); val = add/sub
```

- RDNA1 wavefronts are lockstep within 32 threads — no sync needed
- Total VGPRs: 4 (values) + 4 (temps) = 8 (down from ~36 for shared memory version)
- Directly reduces turbo's per-token FWHT overhead by 2-3x

### 2. Q-vector LDS staging in attention — 15-30% attention speedup

Load Q head once into LDS (512 bytes for head_dim=128), read from LDS in inner loop.
Converts Q reads from global to LDS. Zero VGPR cost.

### 3. Batched prefill FWHT — trivially parallel

Launch `T × n_kv_heads` blocks for T prompt tokens. Each block = 32 threads = one FWHT.
Grid perfectly saturates 40 CUs. Eliminates sequential prefill bottleneck for turbo.

## Medium-Term (1-2 weeks)

### 4. Pre-RoPE K quantization — free quality win

Quantize K vectors BEFORE RoPE rotation, not after. Post-RoPE values have high-frequency
oscillations that are hard to quantize. Pre-RoPE K is smooth. One-line reorder in forward pass.

### 5. Flash Decoding — parallel KV scan for long context

Split KV cache into 64-position tiles (16KB = exactly L1 size on RDNA1). Process tiles
in parallel across wavefronts. Two-pass: partial scores per tile, then reduce.
Expected: 1.3-1.8x attention speedup at 8K+ context.

### 6. Asymmetric KV quantization (KIVI-style)

K: per-channel quantization (stable statistics across tokens)
V: per-token quantization (varies per token)
Different from turbo's uniform FWHT approach. May be complementary.

### 7. KV cache transposition — `[n_kv_heads × max_seq × head_dim]`

Gives fully coalesced K/V reads when scanning positions. 10-20% attention speedup.

## DeltaNet Key Finding

S matrix (128×128 = 64KB) fits EXACTLY in RDNA1's LDS.
Fused kernel: keep S in LDS across all tokens → 7x less global memory traffic.
This is why hipfire should target Qwen3.5 — the hardware fit is perfect.

## Context Extension Math

| Context | FP32 KV | INT4 KV | Fits 8GB? |
|---------|---------|---------|-----------|
| 2K | 603 MB | 150 MB | both |
| 8K | 2.4 GB | 600 MB | INT4 only |
| 32K | 9.4 GB | 1.2 GB | INT4 only |

KV cache quantization directly unlocks context extension. No separate strategy needed.
Qwen3-8B natively supports 32K context (RoPE freq_base). hipfire just needs to stop
capping at 2K and provide INT4 KV.

## Competitive Position

- hipfire: 59.9 tok/s generation on Qwen3-8B (1.34x llama.cpp's 44.3)
- hipfire: 108 tok/s prefill (0.57x llama.cpp's 189 via rocBLAS)
- Value proposition: "Fastest LLM inference for AMD GPUs you actually own"

## Priority Roadmap

| # | What | Impact | Effort | VGPR |
|---|------|--------|--------|------|
| 1 | FWHT via __shfl_xor | 2-3x FWHT, fixes turbo scaling | 1 day | -8 |
| 2 | Q in LDS for attention | 15-30% attention | 1 day | 0 |
| 3 | Batched prefill FWHT | T× prefill throughput | <1 day | 0 |
| 4 | Pre-RoPE K quantization | quality win, zero cost | 1 line | 0 |
| 5 | Flash Decoding | 1.3-1.8x long-ctx attention | 3-5 days | +6 |
| 6 | Embedded tokenizer + HTTP server | user viability | 1 week | — |
| 7 | DeltaNet S-in-LDS | enables Qwen3.5 | 3-5 days | 0 |

## Things NOT To Do

1. Flash Attention for prefill (16KB L1 too small, turbo is better KV solution)
2. GGUF compatibility output (format difference is the advantage)
3. RDNA3/4 matrix core paths (optimize for hardware you measure on)
4. Multi-GPU (8B fits on one card)
5. Continuous batching (single-user engine)
