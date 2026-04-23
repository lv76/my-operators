#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>


__global__ void reduce_max_2d(
    const float* A,
    float* result,
    int d_model
) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int row_start = row * d_model;        
    int idx = row_start + tid * 4;

    float val = -FLT_MAX;

    for (int i = idx; i < row_start + d_model; i += blockDim.x * 4) {
        if (i + 3 < row_start + d_model) {
            float4 V = reinterpret_cast<const float4*>(A)[i / 4];
            val = fmaxf(fmaxf(fmaxf(V.x, V.y), fmaxf(V.z, V.w)), val);  
        } else {
            if (i   < row_start + d_model) val = fmaxf(val, A[i]);
            if (i+1 < row_start + d_model) val = fmaxf(val, A[i+1]);     
            if (i+2 < row_start + d_model) val = fmaxf(val, A[i+2]);
            if (i+3 < row_start + d_model) val = fmaxf(val, A[i+3]);
        }
    }

    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {  
        if (tid < stride) {
            s[tid] = fmaxf(s[tid], s[tid + stride]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = fmaxf(s[tid], s[tid + 32]);
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
        if (tid == 0) result[row] = val;
    }
}

__global__ void reduce_sum_exp(
    const float* A,
    const float* max_val,
    float* result,
    int d_model
) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int row_start = row * d_model;        
    int idx = row_start + tid * 4;

    float row_max = max_val[row];
    float val = 0.0f;

    for (int i = idx; i < row_start + d_model; i += blockDim.x * 4) {
        if (i + 3 < row_start + d_model) {
            float4 V = reinterpret_cast<const float4*>(A)[i / 4];
            val += expf(V.x - row_max);
            val += expf(V.y - row_max);
            val += expf(V.z - row_max);
            val += expf(V.w - row_max);
        } else {
            if (i   < row_start + d_model) val += expf(A[i]   - row_max);
            if (i+1 < row_start + d_model) val += expf(A[i+1] - row_max);
            if (i+2 < row_start + d_model) val += expf(A[i+2] - row_max);
            if (i+3 < row_start + d_model) val += expf(A[i+3] - row_max); 
        }
    }

    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        float val = s[tid] + s[tid + 32];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) result[row] = val;
    }
}

__global__ void softmax_output(
    const float* A,
    const float* max_val,
    const float* exp_sum,
    float* output,
    int d_model
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int row_start = row * d_model;
    int idx = row_start + tid * 4;

    float row_max = max_val[row];
    float row_sum = exp_sum[row];

    for (int i = idx; i < row_start + d_model; i += blockDim.x * 4) {
        if (i + 3 < row_start + d_model) {
            // 向量化读取
            float4 v = reinterpret_cast<const float4*>(A)[i / 4];

            // 计算结果写回，向量化写入
            float4 out;
            out.x = expf(v.x - row_max) / row_sum;
            out.y = expf(v.y - row_max) / row_sum;
            out.z = expf(v.z - row_max) / row_sum;
            out.w = expf(v.w - row_max) / row_sum;
            reinterpret_cast<float4*>(output)[i / 4] = out;  // 向量化写回
        } else {
            // 尾部标量处理
            if (i   < row_start + d_model) output[i]   = expf(A[i]   - row_max) / row_sum;
            if (i+1 < row_start + d_model) output[i+1] = expf(A[i+1] - row_max) / row_sum;
            if (i+2 < row_start + d_model) output[i+2] = expf(A[i+2] - row_max) / row_sum;
            if (i+3 < row_start + d_model) output[i+3] = expf(A[i+3] - row_max) / row_sum;
        }
    }
}


int main() {
    // 参数设置
    int seq_len = 8;
    int d_model = 1024;
    int total = seq_len * d_model;

    // Host 内存分配和初始化
    float *h_A = (float*)malloc(total * sizeof(float));
    float *h_output = (float*)malloc(total * sizeof(float));

    for (int i = 0; i < total; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;  // 随机初始化
    }

    // Device 内存分配
    float *d_A, *d_max, *d_sum, *d_output;
    cudaMalloc(&d_A,      total   * sizeof(float));
    cudaMalloc(&d_max,    seq_len * sizeof(float));
    cudaMalloc(&d_sum,    seq_len * sizeof(float));
    cudaMalloc(&d_output, total   * sizeof(float));

    // 数据传输 Host → Device
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

    // 三个 kernel 串行执行
    reduce_max_2d<<<seq_len, 256>>>(d_A, d_max, d_model);
    cudaDeviceSynchronize();

    reduce_sum_exp<<<seq_len, 256>>>(d_A, d_max, d_sum, d_model);
    cudaDeviceSynchronize();

    softmax_output<<<seq_len, 256>>>(d_A, d_max, d_sum, d_output, d_model);
    cudaDeviceSynchronize();

    // 数据传输 Device → Host
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证：每行的和应该等于1
    for (int row = 0; row < seq_len; row++) {
        float row_sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            row_sum += h_output[row * d_model + i];
        }
        printf("row %d sum = %f\n", row, row_sum);  // 应该接近 1.0
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_output);
    free(h_A);
    free(h_output);

    return 0;
}