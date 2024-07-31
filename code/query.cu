/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-24
 * File: query.cu
 * Description: runtime API query the information of the GPU device.
 *****************************************************************/

#include <stdio.h>
#include "common.cuh"

int main()
{
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp device_prop;
    ErrorCheck(cudaGetDeviceProperties(&device_prop, device_id), __FILE__, __LINE__);

    printf("Device %d:                                      %s\n", 
    device_id, device_prop.name);
        printf("Compute Capability:                             %d.%d\n", 
        device_prop.major, device_prop.minor);
    printf("Number of SMs:                                  %d\n", 
            device_prop.multiProcessorCount);
    printf("L2 Cache Size:                                  %d MB\n",
            device_prop.l2CacheSize / (1024 * 1024));
    printf("Max Threads per Multiprocessor:                 %d\n", 
            device_prop.maxThreadsPerMultiProcessor);
    printf("Stream Priorities Supported:                    %d\n", 
            device_prop.streamPrioritiesSupported);
    printf("Global Memory in L1 Cache Supported:            %d\n", 
            device_prop.globalL1CacheSupported);
    printf("Local Memory in L1 Cache Supported:             %d\n",
            device_prop.localL1CacheSupported);
    printf("Max Shared Memory per SM:                       %zd KB\n", 
            device_prop.sharedMemPerMultiprocessor / 1024);
    printf("Max 32-bit Registers per SM:                    %d\n", 
            device_prop.regsPerMultiprocessor);
    printf("max blocks per SM:                              %d\n", 
            device_prop.maxBlocksPerMultiProcessor);
    printf("Max Threads per Block:                          %d\n",
            device_prop.maxThreadsPerBlock);
    printf("GPU Memory Clock Rate (KHz):                    %d\n", 
            device_prop.memoryClockRate);
    printf("GPU Memory Bus Width (bits):                    %d\n", 
            device_prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):                   %f\n", 
            2.0 * device_prop.memoryClockRate * (device_prop.memoryBusWidth / 8) / 1.0e6);
    printf("Total Global Memory (GB):                       %f\n", 
            device_prop.totalGlobalMem / 1.0e9);
    printf("Total Constant Memory (KB):                     %zd\n", 
            device_prop.totalConstMem / 1024);
    printf("Total Shared Memory per Block (KB):             %zd\n", 
            device_prop.sharedMemPerBlock / 1024);
    printf("Warp Size:                                      %d\n", 
            device_prop.warpSize);
    printf("Max Grid Size:                                  (%d, %d, %d)\n", 
            device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("Concurrent Kernels:                             %d\n", 
            device_prop.concurrentKernels);
    printf("ECC Enabled:                                    %d\n", 
            device_prop.ECCEnabled);
    printf("PCI Bus ID:                                     %d\n", 
         device_prop.pciBusID);
    printf("PCI Device ID:                                  %d\n", 
            device_prop.pciDeviceID);
    
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}

/******************************************************************
Device 0:                                      NVIDIA GeForce RTX 4060 Laptop GPU
Compute Capability:                             8.9
Number of SMs:                                  24
L2 Cache Size:                                  32 MB
Max Threads per Multiprocessor:                 1536      ******优化重点******
Stream Priorities Supported:                    1
Global Memory in L1 Cache Supported:            1
Local Memory in L1 Cache Supported:             1
Max Shared Memory per SM:                       100 KB
Max 32-bit Registers per SM:                    65536个
max blocks per SM:                              24        ******优化重点******
Max Threads per Block:                          1024
GPU Memory Clock Rate (KHz):                    8001000
GPU Memory Bus Width (bits):                    128
Peak Memory Bandwidth (GB/s):                   256.032000
Total Global Memory (GB):                       8.585216
Total Constant Memory (KB):                     64
Total Shared Memory per Block (KB):             48
Warp Size:                                      32
Max Grid Size:                                  (2147483647, 65535, 65535)
Concurrent Kernels:                             1
ECC Enabled:                                    0
PCI Bus ID:                                     1
PCI Device ID:                                  0
 *****************************************************************/