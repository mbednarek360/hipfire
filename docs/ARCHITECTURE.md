# Architecture

## Crate Structure

```
hipfire/
├── crates/
│   ├── hip-bridge/          # Safe FFI to libamdhip64.so via dlopen
│   ├── rdna-compute/        # HIP kernel compilation, dispatch, tensor ops
│   ├── engine/              # Model loading, forward pass, tokenizer
│   └── hipfire-quantize/    # Standalone quantizer (safetensors → .hfq)
└── docs/
```

### hip-bridge

Safe Rust wrapper around the HIP runtime. Loads `libamdhip64.so` at runtime
via `libloading` — no link-time dependency on ROCm. Resolves 20+ HIP API
functions (malloc, memcpy, streams, modules, kernel launch, events).

Key types:
- `HipRuntime`: loaded library + function pointers (Send + Sync)
- `DeviceBuffer`: GPU memory handle (ptr + size, Send but not Sync)
- `malloc_managed()`: unified memory that pages between VRAM and system RAM

### rdna-compute

GPU kernel management and dispatch layer. Kernels are embedded as HIP C++
string constants, compiled by `hipcc --genco` on first use, and cached as
`.hsaco` files in `/tmp/hipfire_kernels/`.

Kernels:
- **GEMV**: F32, Q4_K, Q6_K, Q8_0 (narrow 32-thread + wide 64-thread adaptive),
  Q4_F16_G64, Q4_F16_G32, Q4_LUT, Q4_WAVE, Q4-as-Q8, fused QKV, fused gate+up
- **Elementwise**: RMSNorm, RMSNorm batched, SiLU, fused SiLU×mul, add, add_inplace
- **Attention**: single-head causal with GQA support
- **Other**: RoPE, softmax, embedding lookup (F32/Q4K/Q8), argmax, F16→F32

Adaptive dispatch: `gemv_q8_0` selects between 32-thread (K>1536) and
64-thread multi-row (K≤1536) kernels based on matrix dimensions.

### engine

Model loading and inference orchestration.

Supports:
- **GGUF** format (Q4_K, Q6_K, Q8_0, F32, F16) via memory-mapped parser
- **HFQ** format (.hfq files from hipfire-quantize) with mixed quantization
- **LLaMA** architecture (TinyLlama, etc.)
- **Qwen3** architecture (QK normalization, high rope_freq_base, GPT-2 BPE)

Forward pass: embedding → [RMSNorm → QKV → QK_norm → RoPE → KV_cache →
attention → O_proj → residual → RMSNorm → gate/up → SiLU_mul → down →
residual] × n_layers → final_norm → output_GEMV → argmax

Pre-allocated scratch buffers eliminate per-layer allocation overhead.

### hipfire-quantize

Standalone binary that reads HuggingFace model directories (safetensors +
config.json) and produces `.hfq` files. Handles FP16 and BF16 source weights.

Quantization modes:
- `q8f16`: all weights Q8_0
- `q8-mixed`: Q8 attention/embedding + Q4_K FFN (recommended)
- `q4f16`: all weights Q4_F16_G64
- `q8-fast`: Q8 attention + Q4-as-Q8 FFN

Reports per-tensor quantization error (mean and max).

## Key Design Decisions

### dlopen over link-time dependency
The HIP runtime is loaded at startup via `libloading::Library::new("libamdhip64.so")`.
This means hipfire works across ROCm versions without recompilation, and the binary
can be distributed without bundling ROCm.

### Runtime kernel compilation
HIP C++ kernels are stored as `const &str` in Rust source. On first use, the source
is written to a temp file and compiled via `hipcc --genco --offload-arch=gfx1010 -O3`.
The resulting `.hsaco` is cached on disk. Subsequent runs skip compilation.

Trade-off: first inference call has a ~5 second delay per unique kernel. After that,
kernels load from cache in microseconds.

### Single-warp GEMV (32 threads)
The workhorse kernel uses one warp (32 threads on RDNA) per output row. Benefits:
- No shared memory needed (warp shuffle reduction)
- No `__syncthreads()` barriers
- Maximum blocks per CU (20 at `__launch_bounds__(32, 20)`)
- Each thread processes 8 elements per iteration (unrolled)

This is optimal for K ≥ 2048 where block count provides enough parallelism.

### Multi-row GEMV (64 threads, 2 warps)
For small matrices (K ≤ 1536), the single-warp kernel underutilizes the GPU.
The multi-row variant packs 2 warps per block, each processing a different row.
Grid = M/2. No cross-warp synchronization — each warp independently computes
its row's dot product via warp shuffle.

### Q4K embedding lookup
Large vocab models (Qwen3: 151K tokens) cannot afford F32 embedding tables
(2.5GB for 151K × 4096). GPU kernels dequantize one row at inference time
from the raw Q4K/Q8 data, reducing VRAM from 2.5GB to 334MB.

### Mixed quantization
The quantizer assigns different formats per tensor based on the tensor's role:
- Attention projections → Q8 (latency-sensitive, benefits from 84% peak BW)
- FFN projections → Q4_K (bulk of parameters, compressed for VRAM)
- Embeddings/lm_head → Q8 (large but accessed via lookup or single GEMV)
- Norms → F16 (tiny, no quantization needed)
