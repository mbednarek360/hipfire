# Turbo3 KV Cache Design for RDNA1

## Motivation

Q8 KV cache introduces systematic norm drift in recurrent architectures (DeltaNet).
The S-state accumulates `S_{t+1} = S_t + v_t * k_t^T` — small Q8 quantization biases
compound linearly over hundreds of tokens, producing garbage output.

TurboQuant turbo3 solves this with:
1. **Exact L2 norm preservation** via corrected norm (no systematic drift)
2. **FWHT rotation** decorrelates errors across dimensions (errors grow sqrt(T) not T)
3. **Optimal Lloyd-Max centroids** for Gaussianized data (lower MSE than Q8 at 3.5 bits)

## Data Layout

### Per-head block: 128 elements (= head_dim)

```
turbo3_block (for head_dim=128):
  [f32 corrected_norm]  4 bytes
  [3-bit indices × 128] 48 bytes (packed: 128 × 3 / 8)
  Total: 52 bytes per head per position

Compression: 52B vs 512B (fp32) = 9.85x, vs 256B (fp16) = 4.92x
```

Simplification vs llama.cpp turbo3: we use a **single 128-element block** per head
(not 4 × 32-element blocks). This eliminates per-block norm overhead and matches
our head_dim=128 naturally. One norm per head, not four.

### 3-bit packing: 16 × uint24 triplets

Pack 128 × 3-bit indices into 48 bytes:
- Every 8 indices → 3 bytes (24 bits)
- 128 / 8 = 16 triplets × 3 bytes = 48 bytes
- Extraction: `idx = (packed_byte >> (bit_offset)) & 0x7`

Alternative (split 2+1 like llama.cpp): 32 bytes for 2-bit low + 16 bytes for 1-bit high.
Total = 48 bytes either way. We use the split format for simpler extraction:

```
turbo3_block (split format, head_dim=128):
  [f32 corrected_norm]         4 bytes
  [2-bit low indices × 128]   32 bytes  (4 per byte)
  [1-bit high indices × 128]  16 bytes  (8 per byte)
  Total: 52 bytes per head per position
```

### KV cache memory layout

```
k_cache[layer]: [max_seq × n_kv_heads × 52] bytes
v_cache[layer]: [max_seq × n_kv_heads × 52] bytes
```

For Qwen3-8B (32 kv_heads, 128 head_dim, 2048 ctx):
- Per layer: 2048 × 32 × 52 × 2 (K+V) = 6.5 MB
- 32 layers: 208 MB (vs 536 MB for Q8 = 2.58x savings)
- vs FP32: 1.07 GB → 208 MB = 5.15x savings

### FWHT sign tables

Two fixed sign arrays (seed=42, seed=1042), each 128 elements of ±1.
Stored as 128-bit bitmasks (16 bytes each). Total: 32 bytes constant memory.

## Centroids

Fixed at compile time (Lloyd-Max optimal for N(0, 1/128)):

```c
__constant__ float TURBO3_CENTROIDS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
```

## Kernels

### 1. `fwht_rotate_128` — FWHT rotation (in-place, 128 elements)

**Purpose**: Rotate Q vector before attention, and rotate K/V before quantization.

```
Input:  float x[128] (in registers or LDS)
Output: float x[128] (rotated in-place)

Algorithm:
  1. Apply sign_table_1: x[i] *= signs1[i]  (±1)
  2. FWHT butterfly: 7 passes, stride 1→64
     for stride in [1, 2, 4, 8, 16, 32, 64]:
       for i in [0..128 step 2*stride]:
         for j in [0..stride]:
           a = x[i+j], b = x[i+j+stride]
           x[i+j] = a + b, x[i+j+stride] = a - b
  3. Scale: x[i] *= 1/sqrt(128) = 1/11.3137...
  4. Apply sign_table_2: x[i] *= signs2[i]
```

**RDNA1 implementation**: One thread per head. 128 elements in registers (32 VGPRs
as float4). Butterfly passes are pure register shuffles — no LDS needed since one
thread owns all 128 elements. This is the key insight: head_dim=128 fits in registers.

**VGPRs**: 32 (for x[128] as float) + ~4 temps = ~36. Allows 7 waves/SIMD.

### 2. `kv_cache_write_turbo3` — Quantize and store

**Purpose**: Take fp32 K or V vector (post-RoPE), turbo3-quantize, write to cache.

```
One block per kv_head. Thread 0 does all work (128 elements is too small to parallelize).

Algorithm:
  1. Load 128 fp32 values from src
  2. Compute L2 norm: norm = sqrt(sum(x[i]^2))
  3. Normalize: x[i] /= norm
  4. FWHT rotate: fwht_rotate_128(x)
  5. Quantize to 3-bit: for each x[i], find nearest centroid index (0-7)
     - Since centroids are sorted, use binary search or threshold comparison
     - Thresholds: midpoints between adjacent centroids
  6. Compute reconstruction norm: recon = sqrt(sum(centroid[idx[i]]^2))
  7. Corrected norm: cnorm = norm / recon
  8. Pack 3-bit indices (split 2+1 format) + write cnorm
  9. Store to cache at position pos
```

**3-bit quantization shortcut**: Centroids are symmetric. For N(0, σ), the optimal
assignment boundaries are at midpoints. With 8 centroids we need 7 thresholds:

```c
float THRESHOLDS[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};
// idx = (x > T[0]) + (x > T[1]) + ... + (x > T[6])
// This is branchless: 7 comparisons, 7 additions
```

**VGPRs**: ~40 (128 values + norm + temps). Single-thread per head.

### 3. `attention_turbo3_kv` — Attention with turbo3 KV

**Purpose**: Full attention: Q·K^T (with turbo3 K dequant), softmax, weighted V sum.

```
Grid: one block per attention head. Threads: 32 (one wavefront).

Phase 0: Pre-rotate Q
  - Load Q head (128 floats) into shared memory
  - Thread 0: apply fwht_rotate_128 to Q in-place
  - __syncthreads()
  - Now Q is in rotated space, matching turbo3 K

Phase 1: Q @ K^T
  for t = tid; t < seq_len; t += nthreads:
    // Load turbo3 K block for position t
    cnorm = *(float*)(k_cache + t * kv_bytes + kv_h * 52)
    // Build centroid LUT scaled by cnorm (8 regs)
    cn[c] = CENTROIDS[c] * cnorm  for c in 0..8
    // Dot product with rotated Q
    dot = 0
    for d in 0..128:
      idx = extract_3bit(k_block, d)  // 2-bit low + 1-bit high
      dot += cn[idx] * q_rotated[d]
    scores[t] = dot * scale_attn

Phase 2: Softmax (same as existing Q8 kernel)

Phase 3: Weighted V sum
  for d = tid; d < head_dim; d += nthreads:
    val = 0
    for t in 0..seq_len:
      cnorm_v = *(float*)(v_cache + t * kv_bytes + kv_h * 52)
      idx = extract_3bit(v_block, d)
      cn_v = CENTROIDS[idx] * cnorm_v
      val += scores[t] * cn_v
    // Inverse-rotate the output dimension
    // Actually: V is stored in rotated space, output needs inverse FWHT
    out_rotated[d] = val

Phase 4: Inverse-rotate output
  Thread 0: inverse_fwht_128(out_rotated)
  Write to out[h * head_dim + ...]
```

**Critical insight for V**: Unlike K (where Q is pre-rotated to match), V's weighted
sum produces a result in rotated space. We must inverse-FWHT the output. This is the
same FWHT operation (it's its own inverse up to a scale factor).

**VGPRs**: 8 (centroid LUT) + 1 (dot accumulator) + 4 (index extraction temps) +
128/32 shared Q = ~15 per-thread. Total with softmax state: ~25-30.

### 4. `fwht_inverse_128` — Inverse rotation

Same as forward FWHT but with signs applied in reverse order:
1. Apply signs2
2. FWHT butterfly (same passes)
3. Scale by 1/sqrt(128)
4. Apply signs1

Since `(S2 · H · S1) · (S1 · H · S2) = S2 · H · H · S2 = S2 · (128·I) · S2 = 128·I`,
the inverse is `(1/128) · S1 · H · S2`. But we already scale by 1/sqrt(128) in forward,
so inverse is just: signs2 → FWHT → scale 1/sqrt(128) → signs1. Same code, swapped signs.

## Integration

### KvCache struct changes

```rust
pub struct KvCache {
    // ... existing fields ...
    pub quant_turbo3: bool,
    pub turbo3_signs1: GpuTensor,  // 128 × f32 (±1.0), uploaded once
    pub turbo3_signs2: GpuTensor,
}
```

### Forward pass changes

```rust
// After RoPE:
if kv_cache.quant_turbo3 {
    gpu.kv_cache_write_turbo3(&kv_cache.k_gpu[layer], &scratch.k, &scratch.pos_buf, ...)?;
    gpu.kv_cache_write_turbo3(&kv_cache.v_gpu[layer], &scratch.v, &scratch.pos_buf, ...)?;
    gpu.attention_turbo3_kv(
        &scratch.q, &kv_cache.k_gpu[layer], &kv_cache.v_gpu[layer],
        &scratch.attn_out, &scratch.pos_buf, pos+1, n_heads, n_kv_heads, head_dim, kv_cache.max_seq,
        &kv_cache.turbo3_signs1, &kv_cache.turbo3_signs2,
    )?;
}
```

### DeltaNet application

For DeltaNet S-state, turbo3 can be applied to the K and V vectors before the
recurrence update. The corrected norm ensures that `v_t * k_t^T` products maintain
correct scale, and the decorrelated errors prevent systematic S-state drift.

The S matrix itself stays in FP32 (it's small: head_dim × head_dim = 16KB per head).
Only K and V passing through the cache get turbo3 treatment.

## Performance Expectations

- **KV write**: ~2x slower than Q8 (FWHT + 3-bit quant vs simple int8 scale). But this
  is per-token and tiny compared to GEMV.
- **K dot product**: Similar to Q8 — centroid LUT lookup is ~same cost as int8→float.
  8 register LUT vs byte-to-float conversion. Memory reads are 52B vs 132B per head = 2.5x less.
- **V weighted sum**: Same as K, plus one inverse FWHT at the end (negligible).
- **Net**: Should be bandwidth-positive for attention at longer contexts where K/V reads
  dominate. At very short context (< 32 tokens), the FWHT overhead might be visible.

## Open Questions

1. Should V be rotated? The llama.cpp impl rotates both K and V. For V, the rotation
   decorrelates errors but requires inverse-rotating the attention output. This adds
   one FWHT per head per token. Worth it for quality, but measurable cost.

2. head_dim=64 (Qwen3-0.6B): The FWHT is 6 passes instead of 7. Block size drops to
   26 bytes (4 norm + 16 qs_low + 8 signs = 28 bytes, or 4 + 12 + 4 = 20 bytes packed).
   Need to handle both 64 and 128.

3. Batched prefill: The Q rotation and turbo3 writes need batched versions for prefill.
   Can batch the Q rotation (one FWHT per head per token in the batch). KV writes are
   already per-position.
