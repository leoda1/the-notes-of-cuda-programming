/******************************************************************
 * Author      : Da Liu
 * Date        : 2024-07-29
 * File Name   : GPU_Cache.cu
 * Description : 测试   L1 Cache能否使用与L2 Cache大小
 *****************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include "common.cuh"

__global__ void kernel() {}


int main() {
    int device_id = 0;
    cudaDeviceProp device_prop;
    ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);
    printf("Device %d: %s\n", device_id, device_prop.name);

    if (device_prop.globalL1CacheSupported){
        std::cout << "L1 Cache is supported" << std::endl;
    }
    else{
        std::cout << "L1 Cache is not supported" << std::endl;
    }
    std::cout << "L2 Cache size: " << device_prop.l2CacheSize / (1024 * 1024) << " MB" << std::endl;

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();

    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}

/***********************************************************************
PS C:\Users\22681\Desktop\project\cudalearn> ./build/Debug/GPU_Cache.exe      
Device 0: NVIDIA GeForce RTX 4060 Laptop GPU
L1 Cache is supported
L2 Cache size: 32 MB
 **********************************************************************/