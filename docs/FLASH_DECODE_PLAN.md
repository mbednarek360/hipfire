# Flash Decoding Plan for hipfire RDNA1

## The Problem
Current attention: 1 block per head = 32 blocks for Qwen3-8B.
RX 5700 XT has 40 CUs × 4 SIMDs = 160 wavefront slots.
**80% of CUs are idle during attention.**

## The Solution
Split KV cache scan into tiles. Each tile = separate block.
At seq_len=2048 with tile=64: 32 heads × 32 tiles = 1024 blocks → all CUs busy.

## Two-Pass Algorithm

### Pass 1 (per-tile): online softmax + V accumulation
Each block processes [tile_start..tile_start+TILE_SIZE) positions.
Outputs: partial_out[head_dim], partial_max, partial_lse per tile.

### Pass 2 (reduction): combine tiles
Re-scale each tile's output by exp(tile_max - global_max).
Sum and normalize. Trivially fast (32 iterations per head).

## Tile Sizes (matched to 16KB L1)

| KV format | bytes/pos (K+V) | Optimal tile |
|-----------|-----------------|-------------|
| FP32 | 1024 | 32 |
| Q8_0 | 272 | 64 |
| turbo2 | 72 | 256 |
| turbo4 | 136 | 128 |

## VGPR Budget: ~31 VGPRs
- running_max, running_sum: 2
- running_out[4]: 4 (head_dim/32 per thread)
- dot product + softmax temps: 8
- K/V load regs: 14
- Loop/address: 3

At 32 VGPRs: 32 waves/SIMD → excellent occupancy.

## Expected Speedup

| seq_len | Current CU util | Flash Decode util | Attention speedup |
|---------|----------------|-------------------|-------------------|
| 128 | 20% | 40% | ~1.8× |
| 512 | 20% | ~100% | ~4× |
| 1024 | 20% | ~100% | ~4× |
| 2048 | 20% | ~100% | ~4-5× |

End-to-end impact: at 2K context where attention is ~20-30% of compute,
4× attention speedup → 1.3-1.5× overall.

## Dispatch Threshold
```rust
let use_flash = (n_heads * tiles_per_head) >= 80;  // 2× n_CUs
```
Below 80 blocks: overhead of 2nd kernel > benefit. Use original.

## TurboQuant Integration
Apply FWHT to Q once per tile block (in registers via __shfl_xor).
All positions in tile use the same rotated Q. Natural fit.

## Key ISA: gfx1010 supports global_atomic_add_f32 but CANNOT use it
for Flash Decoding (per-tile rescaling breaks atomics). Use 2-kernel approach.
