# Online Softmax

A single-pass, numerically stable softmax kernel implemented in CUDA.

## Background

Standard softmax requires **two passes** over the input:

1. First pass: find the global maximum for numerical stability
2. Second pass: compute `exp(x - max)` and normalize

**Online Softmax** merges both passes into one by maintaining a running `(max, sum)` pair that can be updated incrementally. This technique was introduced in the paper [*Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867) and is the core building block behind Flash Attention.

## Implementation

### Algorithm

Each thread maintains a local `(global_max, global_sum)` pair. As it reads each element, it updates both statistics in a single pass:

```
new_max = max(global_max, x)
global_sum = global_sum * exp(global_max - new_max) + exp(x - new_max)
global_max = new_max
```

After the loop, partial results are reduced across threads using shared memory and warp shuffle instructions.

### Key Optimizations

**1. float4 Vectorized Memory Access**

Loads 4 floats per instruction to maximize memory bandwidth utilization:

```cuda
float4 V = reinterpret_cast<const float4*>(A)[i/4];
```

**2. Two-Phase Parallel Reduction**

- Phase 1 — Shared memory tree reduction (threads > 32):

```cuda
for (int stride = blockDim.x/2; stride > 32; stride >>= 1) { ... }
```

- Phase 2 — Warp shuffle reduction (last 32 threads, no `__syncthreads()` needed):

```cuda
float other_m = __shfl_down_sync(0xffffffff, m, offset);
```

**3. Numerically Stable (max, sum) Merge**

When merging two partial results `(m1, d1)` and `(m2, d2)`:

```
new_max = max(m1, m2)
new_sum = d1 * exp(m1 - new_max) + d2 * exp(m2 - new_max)
```

This avoids overflow/underflow regardless of input magnitude.

**4. Single-Pass Write-Back**

The second loop writes results using the globally reduced `(final_max, final_sum)`, also vectorized with float4.

### Kernel Configuration

```cuda
online_softmax<<<seq_len, 256>>>(d_A, d_output, d_model);
```

- One block per row (`gridDim.x = seq_len`)
- 256 threads per block
- Each thread processes 4 elements per iteration (via float4)

## Build & Run

```bash
nvcc -O2 -o online_softmax online_softmax.cu
./online_softmax
```

Expected output (each row sum ≈ 1.0):

```
row 0 sum = 1.000000
row 1 sum = 1.000000
...
row 7 sum = 1.000000
```


## Limitations & Future Work

- Currently supports `float32` only (FP16 / BF16 support planned)
- Block size fixed at 256 threads (auto-tuning planned)
- `d_model` must be a multiple of 4 for full vectorization benefit

## Benchmark

Test environment: RTX 5070 Ti Laptop GPU, CUDA 13.1, PyTorch 2.11.0

| seq_len | d_model | PyTorch (ms) | Custom (ms) | Speedup |
|---------|---------|--------------|-------------|---------|
| 128     | 1024    | 0.0078       | 0.0077      | 1.01x   |
| 512     | 1024    | 0.0095       | 0.0100      | 0.95x   |
| 1024    | 1024    | 0.0112       | 0.0107      | 1.05x   |
| 2048    | 1024    | 0.0126       | 0.0192      | 0.66x   |
| 128     | 4096    | 0.0112       | 0.0097      | 1.15x   |
| 512     | 4096    | 0.0265       | 0.0147      | 1.81x   |
| 1024    | 4096    | 0.0468       | 0.0276      | 1.70x   |

**Key observation:** The kernel shows significant advantage at larger `d_model` (up to **1.81x** speedup at d_model=4096), where float4 vectorized memory access fully utilizes DRAM bandwidth. At small sizes, kernel launch overhead dominates.

## Nsight Compute Analysis

Profiled on input shape `4096 × 4096`:

| Metric | Value |
|--------|-------|
| DRAM Throughput | 86.84% (295 GB/s) |
| Memory Throughput | 86.84% |
| Compute Throughput | 38.50% |
| Achieved Occupancy | 96.69% |
| Branch Efficiency | 100% |

**Conclusion:** The kernel is **Memory Bound**, which is expected for softmax — the operation has low arithmetic intensity. The 86.84% DRAM utilization confirms that float4 vectorized loads are effective. Branch efficiency of 100% indicates zero warp divergence.
    

## References

- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
