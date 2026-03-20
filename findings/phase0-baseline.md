# Phase 0: Baseline Findings

**Date:** 2026-03-20
**Machine:** k9lin
**GPU:** AMD Radeon RX 5700 XT (Navi 10, gfx1010, RDNA 1)

## Critical Discovery

ROCm 6.3.4 natively supports gfx1010 for compute. No `HSA_OVERRIDE_GFX_VERSION` hack needed.
The CLAUDE.md plan assumed gfx1010 was "artificially gated" — this is no longer true as of ROCm 6.3.4.

## Hardware

- GPU: AMD Radeon RX 5700 XT at PCIe 0a:00.0 (Navi 10, rev c1)
- Device ID: 0x731f, Vendor: 0x1002 (AMD)
- 40 CUs, 2 SIMDs/CU, Wavefront size 32
- Max clock: 2100 MHz, L2 cache: 4MB
- VRAM: 8GB GDDR6
- Render node: /dev/dri/renderD128
- 8 compute rings (comp_1.0.0 through comp_1.3.1)
- 7 Vulkan compute queues
- Fast F16: TRUE

## Software Stack

- **OS:** Linux 6.17.0-14-generic (k9lin)
- **ROCm:** 6.3.4 (rocm-core 6.3.4.60304-76~24.04)
- **HIP:** 6.3.42134-a9a80e791
- **Compiler:** AMD clang 18.0.0git (roc-6.3.4)
- **HSA Runtime:** 1.14.0
- **Rust:** 1.92.0 (stable)
- **Vulkan:** 1.3.275 (RADV)
- **Libraries:** rocBLAS 4.3.0, rocFFT 1.0.31, hipFFT 1.0.17, hipSPARSE 3.1.2, RCCL 2.21.5

## Harness Results

All 12 checks pass. Max Tier: 6.

| Tier | Description | Result |
|------|-------------|--------|
| 0 | Kernel driver | PASS (amdgpu loaded, renderD128 exists, 0 dmesg errors) |
| 1 | Userspace detection | PASS (rocm-smi, rocminfo gfx1010, Vulkan RADV) |
| 2 | Compute runtime init | PASS (HIP: gfx1010:xnack-, 7 Vulkan compute queues) |
| 3 | Memory operations | PASS (hipMalloc/hipMemcpy round-trip verified) |
| 4 | Compute kernel | PASS (vector_add 65536 elements, 0 errors) |
| 5 | Matrix multiply | PASS (512x512x512, 0 errors, ~457 GFLOPS naive) |
| 6 | Performance | Reached (detailed benchmarks TBD) |

## Performance Notes

- Naive matmul: ~457 GFLOPS (4.7% of 9.75 TFLOPS theoretical)
- This is expected — no shared memory tiling, no vectorization
- Significant room for optimization

## Impact on Plan

Since HIP works natively on gfx1010 with ROCm 6.3.4:
- **Phase 1** (mapping): Drastically reduced scope. Main question is now Rust FFI patterns, not "can we make the GPU work?"
- **Phase 2** (approaches): Approach B (Rust FFI to HIP/HSA via dlopen) is the clear winner
- **Phase 3** (validation): Already done — Tier 6 baseline
- **Phase 4** (build): Start immediately. Focus on hip-bridge FFI layer
