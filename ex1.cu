#include <stdio.h>

__global__ void helloGPU() {
    printf("Hello from GPU! Block=%d, Thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    helloGPU<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}