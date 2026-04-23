#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 32

// 朴素矩阵乘法
__global__ void matMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Shared Memory矩阵乘法
__global__ void matMulShared(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < n / TILE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

int main() {
    int n = 4096;
    int a = TILE;
    size_t size = n * n * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(TILE, TILE);
    dim3 gridSize(n / TILE, n / TILE);

    // 朴素版计时
    float ms1 = 0;
    cudaEventRecord(start);
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop);
    printf("朴素版: %f ms\n", ms1);

    // Shared Memory版计时
    float ms2 = 0;
    cudaEventRecord(start);  // 重新开始计时
    matMulShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms2, start, stop);
    printf("Shared Memory版: %f ms\n", ms2);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("C[0][0] = %f\n", h_C[0]);  // 应该是 n*2 = 8192
    printf("TILE:%d\n",a);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}