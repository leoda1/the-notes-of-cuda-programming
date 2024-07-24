/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-24
 * File: query.cu
 * Description: runtime API query the information of the GPU device.
 *****************************************************************/

#include <stdio.h>
#include "../cudalearn/tools/common.cuh"

int main()
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp device_prop;
    ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);

    printf("Device                                          %d: %s\n", 
    device_id, device_prop.name);
    printf("Compute Capability:                             %d.%d\n", 
            device_prop.major, device_prop.minor);
    printf("Memory Clock Rate (KHz):                        %d\n", 
        device_prop.memoryClockRate);
    printf("Memory Bus Width (bits):                        %d\n", 
        device_prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):                   %f\n", 
        2.0 * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6);
    printf("Multiprocessors:                                %d\n", 
        device_prop.multiProcessorCount);
    printf("Total Global Memory (GB):                       %f\n", 
        device_prop.totalGlobalMem / 1.0e9);
    printf("Total Constant Memory (KB):                     %d\n", 
        device_prop.totalConstMem / 1024);
    printf("Total Shared Memory per Block (KB):             %d\n", 
        device_prop.sharedMemPerBlock / 1024);
    printf("Warp Size:                                      %d\n", 
        device_prop.warpSize);
    printf("Maximum Threads per Block:                      %d\n", 
        device_prop.maxThreadsPerBlock);
    printf("Maximum Threads per Multiprocessor:             %d\n", 
        device_prop.maxThreadsPerMultiProcessor);
    printf("Maximum Grid Size:                              (%d, %d, %d)\n", 
        device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("Concurrent Kernels:                             %d\n", 
        device_prop.concurrentKernels);
    printf("ECC Enabled:                                    %d\n", 
        device_prop.ECCEnabled);
    printf("PCI Bus ID:                                     %d\n", 
        device_prop.pciBusID);
    printf("PCI Device ID:                                  %d\n", 
        device_prop.pciDeviceID);
    return 0;
}