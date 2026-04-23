#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduction(float *A, float *result, int n){
    __shared__ float s[256];
    int i = threadIdx.x;
    s[i] = A[blockIdx.x * blockDim.x + i];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *=2){
        if(i % (2*stride) == 0){
            s[i] += s[i+stride];
        }
        __syncthreads();
    }

    if(i == 0) {
        result[blockIdx.x] = s[0];  // 只有Thread0写回
    }
}


int main() {
    int n = 1024;
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
    // 第一次规约
    reduction<<<4, 256>>>(d_A, d_result, n);

    // 第二次规约
    reduction<<<1, 4>>>(d_result, d_final, 4);

    // 把结果搬回CPU
    float result;
    cudaMemcpy(&result, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum = %f\n", result);
}