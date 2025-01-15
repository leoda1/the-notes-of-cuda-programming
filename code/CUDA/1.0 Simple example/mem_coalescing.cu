#include <iostream>
#include <cuda_runtime.h>

__global__ void copyDataNonCoalesce (float *in, float *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[(idx * 2) % n];
    }
}

__global__ void copyDataCoalesce (float *in, float *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] =in[idx];
    }
}

void initializeArray (float *arr, int n) {
    for (int i = 0; i < n; i ++) {
        arr[i] = static_cast<float>(i);  // 显式的将i变为float类型
    }
}

int main() {
    const int n = 1 << 24;
    float *in, *out;
    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);

    int blocksize = 128;
    int numblock = (n + blocksize - 1) / blocksize;

    copyDataNonCoalesce<<<numblock, blocksize>>>(in, out, n);
    cudaDeviceSynchronize();

    initializeArray(out, n);

    copyDataCoalesce<<<numblock, blocksize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);   
    cudaFree(out);

    return 0;
}

/*
nvcc编译成可执行文件用ncu的connect分析
or
nvcc编译后的可执行文件用指令
ncu --set full --export name.ncu-rep ./xxx.exe

*/