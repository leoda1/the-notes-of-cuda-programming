/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-27
 * File: global_memory.cu
 * Description: 静态全局变量的使用.
 *****************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include "common.cuh"

__device__ int d_x = 1;
__device__ int d_y[2]; 

__global__ void kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d\n", d_x, d_y[0], d_y[1]);
}

int main(int argc, char** argv)
{
    int device_id = 0;
    cudaDeviceProp device_prop;
    ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);

    //std::cout << "运行时GPU设备:" << device_prop.name << std::endl;
    printf("the device of GPU: %s\n", device_prop.name);

    int h_y[2] = {10, 20};
    ErrorCheck(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2), __FILE__, __LINE__);
    printf("h_y[0] = %d, h_y[1] = %d\n", h_y[0], h_y[1]);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}