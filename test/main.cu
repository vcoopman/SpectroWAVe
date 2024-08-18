// simple_cuda.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void helloCUDA() {
    printf("Hello from GPU!\n");
}

int main() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

