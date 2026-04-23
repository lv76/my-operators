# 🚀 CUDA Custom Operators

A collection of hand-optimized CUDA kernels focusing on performance-critical operations in deep learning. Each operator is implemented from scratch with low-level GPU optimization techniques.

## Operator List

| Operator                            | Description                            | Key Optimizations                            | Status |
| ----------------------------------- | -------------------------------------- | -------------------------------------------- | ------ |
| [Online Softmax](./online_softmax/) | Single-pass numerically stable softmax | float4 vectorization, warp shuffle reduction | ✅ Done |

> More operators coming soon...

## Environment

- CUDA 12.9
- C++17
- Tested on NVIDIA GPU (sm_86+)

## Repository Structure

```
my-operators/
├── README.md
└── online_softmax/
    ├── online_softmax.cu
    └── README.md
```

## Author

[@lv76](https://github.com/lv76)
