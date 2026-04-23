#include <stdio.h>
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void vectorAdd(float *A, float *B, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <n){
        c[i] = A[i] + B[i];
    }
}

int main(){
    int n = 1024;
    size_t size = n * sizeof(int);
    //CPU上分配内存
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for(int i = 0;i <n ; i++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Host -> device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Launch Kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    cudaEventRecord(start);
    //device -> Host 
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, stop);

    printf("时间： %f ms\n", ms1);
    //printf("C[0] = %f\n", h_C[0]);
    //printf("C[1023] = %f\n", h_C[1023]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}



