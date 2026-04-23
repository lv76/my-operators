#include <stdio.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduction_float4(const float*A, float *result,int n){
    __shared__ float s[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDimx.x * 4 + tid * 4;
    float max = 0;//softmax要求求最大值
    if(idx < n-3 ){
        float4 *V = reinterpret_cast<float4*>(A)[idx/4];
        max = fmaxf(famxf(V.x,V.y),famxf(V.z,V.w));
    }else{
        if(idx < n)max = A[idx];
        if(idx+1 < n) max = fmaxf(A[idx],A[idx+1]);
        if(idx+2 < n) max = famxf(fmaxf(A[idx],A[idx+1]),A[idx+2]);
    }
    s[tid] = max ;
    __syncthreads();

    for(int stride = blockDimx.x/2; stride>32; stride>>=1){
        if(tid<stride){
        s[tid] = fmaxf(s[tid],s[tid+stride]);
        }
        __syncthreads();
    }

    if(tid < 32){
        float val = fmaxf(s[tid],s[tid + 32]);
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
        val =fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
        if (tid == 0) result[blockIdx.x] = val;
    }

}






int main() {
    int n = 1024;
    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    // 初始化数据...

    float *d_A, *d_result, *d_final;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_result, 4 * sizeof(float));   // 4个部分和
    cudaMalloc(&d_final, 1 * sizeof(float));    // 最终结果

    float *h_A = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
    }
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    reduction<<<blocks, threads>>>(d_A, d_result, n);
    reduction<<<1, threads>>>(d_result, d_final, blocks);

    // 把结果搬回CPU
    float result;
    cudaMemcpy(&result, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum = %f\n", result);
    cudaFree(d_A);
    cudaFree(d_result);
    cudaFree(d_final);
    free(h_A);
}