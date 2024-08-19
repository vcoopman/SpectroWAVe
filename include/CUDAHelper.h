#include <vector>

#include <cuda_runtime.h>

// TO REVIEW: May this kernels by templated?
__global__ void addArraysInPlaceKernel(float* array1, const float* array2, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    array1[id] += array2[id];
  }
}

class CUDAHelper {
  /**
   * Collection of general purpose operations powered by CUDA. 
   *
   * Keept on purpose as header only.
   */

  public:
    /**
     * array2 is added to array1 in place.
     */
    static void addArraysInPlace(float* array1, const float* array2, int size) {
      float* d_array1;
      float* d_array2;
      CUDA_RT_CALL(cudaMalloc(&d_array1, sizeof(float) * size)); 
      CUDA_RT_CALL(cudaMalloc(&d_array2, sizeof(float) * size)); 

      CUDA_RT_CALL(cudaMemcpy(d_array1, array1, sizeof(float) * size, cudaMemcpyHostToDevice));
      CUDA_RT_CALL(cudaMemcpy(d_array2, array2, sizeof(float) * size, cudaMemcpyHostToDevice));
    
      int blockSize = 1024; // 1024 is the blockSize upper limit. 
                            // It is always a good idea to choose a multiple of 32 to effiently use the warp size. 
      int gridSize = (size + blockSize) - 1 / blockSize;
      addArraysInPlaceKernel<<<gridSize, blockSize>>>(d_array1, d_array2, size);

      CUDA_RT_CALL(cudaMemcpy(array1, d_array1, sizeof(float) * size, cudaMemcpyDeviceToHost));

      CUDA_RT_CALL(cudaFree(d_array1));
      CUDA_RT_CALL(cudaFree(d_array2));
    }

  private:
    CUDAHelper() = default;
    ~CUDAHelper() = default;
};

