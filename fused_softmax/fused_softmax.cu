#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>


__global__ void fused_softmax(
    const float* input, //shape(seq_len, seq_len)
    float* output,
    int seq_len,
    float scale
){
    //intinalize
    __shared__ float s_sum[256];
    __shared__ float s_max[256];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int row_start = blockIdx.x * seq_len;
    int idx = row_start + tid*4;

    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    //第一个循环：单遍扫描，scale + mask + 累积(max, sum)
    for(int i = idx; i < row_start + seq_len; i += 4 * blockDim.x){
        int col = i - row_start;
        if (i+3< row_start + seq_len){
            float4 v = reinterpret_cast<const float4*>(input)[i/4];
            float x0 = v.x * scale; float x1 = v.y * scale;
            float x2 = v.z * scale; float x3 = v.w * scale;
            if (col > row) {x0 = -FLT_MAX; x1 = -FLT_MAX; x2 = -FLT_MAX; x3 = -FLT_MAX;}
            if (col+1 > row) {x1 = -FLT_MAX; x2 = -FLT_MAX; x3 = -FLT_MAX;}
            if (col+2 > row) {x2 = -FLT_MAX; x3 = -FLT_MAX;}
            if (col+3 > row) x3 = -FLT_MAX;
            float tile_max = fmaxf(fmaxf(x0,x1),fmaxf(x2,x3));
            float tile_sum = expf(x0-tile_max) + expf(x1-tile_max) + expf(x2-tile_max) + expf(x3-tile_max);
            float new_max = fmaxf(tile_max,global_max);
            global_sum = global_sum * expf(global_max - new_max) + tile_sum * expf(tile_max - new_max); 
            global_max = new_max;
        }else{
            for(int j=0; j<4; j++){
                if(i+j< row_start + seq_len){
                    float x = input[i+j] * scale;
                    if(col + j>row) x = -FLT_MAX;
                    float new_max = fmaxf(x,global_max);
                    global_sum = expf(x-new_max) + global_sum * expf(global_max - new_max);
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
    
    float final_sum = s_sum[0];
    float final_max = s_max[0];
    for(int i = idx; i < row_start + seq_len; i += blockDim.x * 4){
        int col = i - row_start;
        if (i+3 < row_start + seq_len){
            float4 V = reinterpret_cast<const float4*>(input)[i/4];
            float4 out;
            out.x = (col   > row) ? 0.0f: expf(V.x*scale - final_max)/final_sum;
            out.y = (col+1 > row) ? 0.0f: expf(V.y*scale - final_max)/final_sum;
            out.z = (col+2 > row) ? 0.0f: expf(V.z*scale - final_max)/final_sum;
            out.w = (col+3 > row) ? 0.0f: expf(V.w*scale - final_max)/final_sum;
            reinterpret_cast<float4*>(output)[i/4] = out;
        } else {
             for(int j = 0; j < 4; j++){
                if(i+j < row_start + seq_len){
                output[i+j] = (col+j > row) ? 0.0f : expf(input[i+j]*scale - final_max)/final_sum;
                }
            }
        }
    }
}


int main(){
    int seq_len = 4096;
    int d_k = seq_len;
    float scale = 1.0f / sqrtf((float)d_k);
    int total = seq_len*seq_len;

    // Host 内存
    float *h_A      = (float*)malloc(total   * sizeof(float));
    float *h_output = (float*)malloc(total   * sizeof(float));

    //初始化
    srand(42);
    for (int i = 0; i < total; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
    }
    
    // Device 内存
    float *d_A, *d_output;
    cudaMalloc(&d_A, total*sizeof(float));
    cudaMalloc(&d_output, total*sizeof(float));

    // Host to Device
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);

    //restart kernel
    fused_softmax<<<seq_len, 256>>>(d_A, d_output, seq_len, scale);
    cudaDeviceSynchronize();

    // Device → Host
    cudaMemcpy(h_output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    // print
    for (int row = 0; row < seq_len; row++) {
        float row_sum = 0.0f;
        for (int col = 0; col < seq_len; col++) {
            row_sum += h_output[row * seq_len + col];
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