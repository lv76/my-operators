#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ─── Kernel ───────────────────────────────────────────────────────────────────

__global__ void online_softmax_kernel(
    const float* A,
    float*       output,
    int          d_model
) {
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    int tid       = threadIdx.x;
    int row_start = blockIdx.x * d_model;
    int idx       = row_start + tid * 4;

    float global_max = -FLT_MAX;
    float global_sum = 0.0f;

    // Single-pass: accumulate (max, sum) pair
    for (int i = idx; i < row_start + d_model; i += blockDim.x * 4) {
        if (i + 3 < row_start + d_model) {
            float4 V = reinterpret_cast<const float4*>(A)[i / 4];
            float tile_max = fmaxf(fmaxf(V.x, V.y), fmaxf(V.z, V.w));
            float tile_sum = expf(V.x - tile_max) + expf(V.y - tile_max)
                           + expf(V.z - tile_max) + expf(V.w - tile_max);
            float new_max  = fmaxf(global_max, tile_max);
            global_sum = global_sum * expf(global_max - new_max)
                       + tile_sum  * expf(tile_max  - new_max);
            global_max = new_max;
        } else {
            for (int j = 0; j < 4; j++) {
                if (i + j < row_start + d_model) {
                    float new_max  = fmaxf(global_max, A[i + j]);
                    global_sum = global_sum * expf(global_max - new_max)
                               + expf(A[i + j] - new_max);
                    global_max = new_max;
                }
            }
        }
    }

    s_max[tid] = global_max;
    s_sum[tid] = global_sum;
    __syncthreads();

    // Shared-memory tree reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            float a       = s_max[tid + stride];
            float b       = s_sum[tid + stride];
            float new_max = fmaxf(s_max[tid], a);
            s_sum[tid]    = s_sum[tid] * expf(s_max[tid] - new_max)
                          + b          * expf(a           - new_max);
            s_max[tid]    = new_max;
        }
        __syncthreads();
    }

    // Warp shuffle reduction (last 32 threads)
    if (tid < 32) {
        float m = fmaxf(s_max[tid], s_max[tid + 32]);
        float d = s_sum[tid] * expf(s_max[tid] - m)
                + s_sum[tid + 32] * expf(s_max[tid + 32] - m);

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_m = __shfl_down_sync(0xffffffff, m, offset);
            float other_d = __shfl_down_sync(0xffffffff, d, offset);
            float new_m   = fmaxf(m, other_m);
            d = d * expf(m - new_m) + other_d * expf(other_m - new_m);
            m = new_m;
        }

        if (tid == 0) { s_max[0] = m; s_sum[0] = d; }
    }
    __syncthreads();

    float final_max = s_max[0];
    float final_sum = s_sum[0];

    // Write back (vectorized)
    for (int i = idx; i < row_start + d_model; i += blockDim.x * 4) {
        if (i + 3 < row_start + d_model) {
            float4 V   = reinterpret_cast<const float4*>(A)[i / 4];
            float4 out;
            out.x = expf(V.x - final_max) / final_sum;
            out.y = expf(V.y - final_max) / final_sum;
            out.z = expf(V.z - final_max) / final_sum;
            out.w = expf(V.w - final_max) / final_sum;
            reinterpret_cast<float4*>(output)[i / 4] = out;
        } else {
            if (i   < row_start + d_model) output[i]   = expf(A[i]   - final_max) / final_sum;
            if (i+1 < row_start + d_model) output[i+1] = expf(A[i+1] - final_max) / final_sum;
            if (i+2 < row_start + d_model) output[i+2] = expf(A[i+2] - final_max) / final_sum;
            if (i+3 < row_start + d_model) output[i+3] = expf(A[i+3] - final_max) / final_sum;
        }
    }
}

// ─── Python binding ───────────────────────────────────────────────────────────

torch::Tensor online_softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),        "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int seq_len = input.size(0);
    int d_model = input.size(1);

    auto output = torch::empty_like(input);

    online_softmax_kernel<<<seq_len, 256>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        d_model
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &online_softmax_forward, "Online Softmax forward (CUDA)");
}
