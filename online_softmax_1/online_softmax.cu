#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

__global__ void online_softmax(
    const float* A,
    float *output,
    int d_model 
){
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    int tid = threadIdx.x;
    int row = blockIdx.x; 
    int row_start = blockIdx.x * d_model;
    int idx = row_start + tid * 4;

    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    //找一个d_model中的最大值
    for(int i = idx; i < row_start + d_model; i += blockDim.x * 4){
        if(i + 3 < row_start + d_model){
            float4 V = reinterpret_cast<const float4*>(A)[i/4];
            float tile_max = fmaxf(fmaxf(V.x,V.y),fmaxf(V.z,V.w));
            float tile_sum = expf(V.x - tile_max) + expf(V.y - tile_max) 
                + expf(V.w - tile_max) + expf(V.z - tile_max);
            
            float new_max = fmaxf(global_max,tile_max);
            global_sum = global_sum * expf(global_max - new_max) + tile_sum * expf(tile_max - new_max); 
            global_max = new_max;
        }else{
            for(int j = 0; j < 4; j++){
                if(i+j<row_start + d_model){
                    float new_max = fmaxf(global_max, A[i+j]);
                    global_sum =  global_sum * expf(global_max - new_max) +  expf(A[i+j] - new_max);
                    global_max = new_max;
                }
            }
        }
    }
    s_max[tid] = global_max;
    s_sum[tid] = global_sum;
    __syncthreads();
    // 归约：合并两个线程的 (max, sum) 对
    for(int stride = blockDim.x/2; stride > 32; stride >>= 1){
        if(tid < stride){   
            float a = s_max[tid + stride];
            float b = s_sum[tid + stride];
            float new_max = fmaxf(a,s_max[tid]);

            s_sum[tid] = s_sum[tid] * expf(s_max[tid] - new_max) 
                    + b * expf(a - new_max);
            s_max[tid] = new_max; 
        }
        __syncthreads();
    }

    if(tid < 32){
        float m = fmaxf(s_max[tid], s_max[tid+32]);
        float d = s_sum[tid] * expf(s_max[tid] - m) + s_sum[tid+32] * expf(s_max[tid+32]-m);

        for (int offset = 16; offset > 0 ; offset>>=1){
            float other_m = __shfl_down_sync(0xffffffff, m, offset);
            float other_d = __shfl_down_sync(0xffffffff, d, offset); 
            float new_m = fmaxf(m,other_m);
            d = d * expf(m - new_m) + other_d * expf(other_m - new_m);
            m = new_m;
        }

        if (tid == 0) {
            s_max[0] = m;
            s_sum[0] = d;
        }
        
    }
    __syncthreads();

    // 第二次遍历：写回结果
    float final_max = s_max[0];
    float final_sum = s_sum[0];

    for (int i = idx; i < row_start + d_model; i += blockDim.x*4){
        if (i + 3 < row_start + d_model){
            float4 V = reinterpret_cast<const float4*>(A)[i/4];
            float4 out ;
            out.x = expf(V.x - final_max) / final_sum;
            out.y = expf(V.y - final_max) / final_sum;
            out.z = expf(V.z - final_max) / final_sum;
            out.w = expf(V.w - final_max) / final_sum;
            reinterpret_cast<float4*>(output)[i/4] = out;
        }else{
            if (i   < row_start + d_model) output[i]   = expf(A[i]   - final_max) / final_sum;
            if (i+1 < row_start + d_model) output[i+1] = expf(A[i+1] - final_max) / final_sum;
            if (i+2 < row_start + d_model) output[i+2] = expf(A[i+2] - final_max) / final_sum;
            if (i+3 < row_start + d_model) output[i+3] = expf(A[i+3] - final_max) / final_sum;
        }
    }
}


int main() {
    int seq_len = 8;
    int d_model = 1024;
    int total = seq_len * d_model;

    // Host 内存
    float *h_A      = (float*)malloc(total   * sizeof(float));
    float *h_output = (float*)malloc(total   * sizeof(float));

    // 随机初始化
    srand(42);
    for (int i = 0; i < total; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device 内存
    float *d_A, *d_output;
    cudaMalloc(&d_A,      total * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    // Host → Device
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    online_softmax<<<seq_len, 256>>>(d_A, d_output, d_model);
    cudaDeviceSynchronize();

    // Device → Host
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证：每行 sum 应该接近 1.0
    for (int row = 0; row < seq_len; row++) {
        float row_sum = 0.0f;
        for (int col = 0; col < d_model; col++) {
            row_sum += h_output[row * d_model + col];
        }
        printf("row %d sum = %f\n", row, row_sum);
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_output);
    free(h_A);
    free(h_output);

    return 0;
}