# Fused Softmax CUDA Kernel

A single-pass CUDA kernel that fuses **scale**, **causal mask**, and **softmax** into one kernel, eliminating redundant global memory accesses between operations.

---

## 1. Problem Definition

In a standard Transformer attention mechanism, the attention score computation involves three sequential operations:

```
Attention(Q, K, V) = softmax(QK^T / √d_k + mask) · V
```

A naive implementation runs these as **three separate kernels**:
1. **Scale kernel** — divide every element by √d_k, write back to DRAM
2. **Mask kernel** — add causal mask, write back to DRAM  
3. **Softmax kernel** — read again, compute softmax, write back to DRAM

This means the attention score matrix is read and written **3 times** from/to DRAM. For large sequence lengths, this memory bandwidth cost dominates runtime.

**Fused Softmax** merges all three into a single kernel, so the matrix is read **once** (during the online scan) and written **once** (during write-back).

---

## 2. Mathematical Background

### Causal Mask

In autoregressive models, each token at position `i` should only attend to positions `j ≤ i`. This is enforced by setting future positions to `-∞` before softmax:

```
masked_score[i][j] = score[i][j] / √d_k       if j ≤ i
                   = -∞                          if j > i
```

Since `exp(-∞) = 0`, masked positions naturally contribute 0 after softmax, and the remaining probabilities still sum to 1.

### Online Softmax with Scale + Mask

Following the online normalizer algorithm, each thread maintains a running `(max, sum)` pair. Scale and mask are applied **on the fly** during the scan — no intermediate buffer needed:

```
x = input[i] * scale          # scale
if j > i: x = -FLT_MAX        # causal mask
new_max = max(global_max, x)
global_sum = global_sum * exp(global_max - new_max) + exp(x - new_max)
global_max = new_max
```

---

## 3. Baseline Implementation

A naive baseline would use three separate kernels:

```cuda
scale_kernel<<<...>>>(input, temp1, scale);        // Read + Write
mask_kernel<<<...>>>(temp1, temp2, seq_len);       // Read + Write
softmax_kernel<<<...>>>(temp2, output, seq_len);   // Read + Write
```

Each kernel launch incurs:
- Full DRAM read + write of the score matrix
- Kernel launch overhead
- Synchronization between kernels

Total DRAM traffic: **3 reads + 3 writes** of the score matrix.

---

## 4. Optimization

### Kernel Fusion
All three operations are merged into one kernel. The score matrix is accessed only **twice** total — once during the online scan (read), once during write-back (read + write):

```
Fused kernel: Read once → compute scale+mask+softmax in registers → Write once
```

### float4 Vectorized Memory Access
Loads 4 floats per instruction to maximize memory bandwidth:
```cuda
float4 v = reinterpret_cast<const float4*>(input)[i/4];
float x0 = v.x * scale, x1 = v.y * scale;
float x2 = v.z * scale, x3 = v.w * scale;
```

### Per-element Mask in Registers
Causal mask is applied directly in registers using column position — no separate mask matrix needed:
```cuda
if (col   > row) x0 = -FLT_MAX;
if (col+1 > row) x1 = -FLT_MAX;
if (col+2 > row) x2 = -FLT_MAX;
if (col+3 > row) x3 = -FLT_MAX;
```

### Two-Phase Reduction (Shared Memory + Warp Shuffle)
Identical to the Online Softmax kernel — shared memory tree reduction followed by warp shuffle for the final 32 threads.

---

## 5. Benchmark

> Benchmark vs PyTorch `F.scaled_dot_product_attention` (softmax component) — coming soon after Triton implementation for three-way comparison.

---

## 6. Nsight Compute Analysis

Profiled on input shape `4096 × 4096`, RTX 5070 Ti Laptop GPU, CUDA 13.1:

| Metric | online_softmax | fused_softmax |
|--------|---------------|---------------|
| Duration [us] | 413.15 | 412.80 |
| DRAM Throughput [%] | 86.84 | **87.47** |
| Memory Throughput [GB/s] | 295 | **297** |
| Achieved Occupancy [%] | 96.69 | 87.60 |
| Branch Efficiency [%] | 100 | 98.98 |
| Avg. Divergent Branches | 0 | 86.26 |

### Key Observations

**DRAM throughput is maintained (87.47%)** — kernel fusion successfully preserves memory bandwidth utilization. The fused kernel reads input only once, consistent with the theoretical expectation.

**Occupancy dropped slightly (96.69% → 87.60%)** — caused by workload imbalance across warps. The causal mask makes early rows (row 0 processes 1 element, row N processes N elements) significantly lighter than late rows, leading to uneven warp utilization within a block.

**Branch divergence appeared (Avg. Divergent Branches: 86.26)** — a direct result of the per-element causal mask `if (col > row)` check. Different threads in the same warp may take different branches when processing tiles that straddle the mask boundary. This is an expected and acceptable cost of fusing the mask.

**Conclusion:** The kernel is **Memory Bound** (87.47% DRAM utilization), consistent with the low arithmetic intensity of softmax. Kernel fusion eliminates intermediate DRAM writes at no throughput cost.

---

## 7. Summary

Implementing fused softmax deepened my understanding of two key ideas:

**Kernel fusion is about eliminating memory round-trips, not compute.** The fused kernel runs at virtually the same speed as the standalone softmax (412 vs 413 µs), but in a real attention pipeline it removes two extra kernel launches and their associated DRAM traffic for the intermediate buffers.

**Causal mask introduces unavoidable warp divergence at tile boundaries.** The divergence only occurs when a float4 tile straddles the diagonal — most tiles are either entirely unmasked or entirely masked and take no branch penalty. This is a fundamental trade-off of per-element masking in a SIMT architecture.

**Next step:** Implement this kernel in Triton and compare all three implementations (CUDA, Triton, PyTorch) in a unified benchmark.

---

## Build & Run

```bash
nvcc -O2 -o fused_softmax fused_softmax.cu
./fused_softmax
```

Expected output — each row sums to 1.0 (masked positions output 0):
```
row 0 sum = 1.000000
row 1 sum = 1.000000
...
```

## References

- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [CUDA Best Practices Guide — Kernel Launch Overhead](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
