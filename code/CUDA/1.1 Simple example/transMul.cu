#include <iostream>
#include <cuda_runtime.h>
#include "../1.2 Test_warp/inc/utils.hpp"

__global__ void transpose(float* output, float* input, int h, int w){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    // int idx = iy * w + ix;
    if (ix < w && iy < h){
        output[ix * h + iy] = input[iy * w + ix];
    }
}

int main(){
    int h = 3;
    int w = 3;
    int size = h * w * sizeof(float);
    float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float output[9] = {0};

    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);    
    CUDA_CHECK(cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice));
    dim3 dimBlock(3, 3);
    dim3 dimGrid(1);

    transpose<<<dimGrid, dimBlock>>>(d_output, d_input, h, w);

    CUDA_CHECK(cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost));
    std::cout << "Input Matrix:" << std::endl;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            std::cout << input[i * w + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Transposed Matrix:" << std::endl;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            std::cout << output[i * h + j] << " ";
        }
        std::cout << std::endl;
    }


    cudaFree(d_input);
    cudaFree(d_output);

}