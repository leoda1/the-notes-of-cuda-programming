#pragma once
#include<iostream>
#include<stdlib.h>
#include<stdio.h>

//查看当前可用GPU设备数 并将可用device设置为0
void setGPU(){
    int iDeviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&iDeviceCount);
    if (err!= cudaSuccess || iDeviceCount == 0)
    {
        std::cout<< "No CUDA-capable device found." << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Number of CUDA-capable devices found: " << iDeviceCount << std::endl;
    }

    int iDevice = 0;
    err = cudaSetDevice(iDevice);
    if( err != cudaSuccess)
    {
        std::cout << "Failed to set device " << iDevice << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Device " << iDevice << " set successfully." << std::endl;
    }

}

//错误检查函数
cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if(error_code != cudaSuccess)
    {
        std::cout << "CUDA error:\r\ncode=%d,name=%s,description=%s\r\nfile=%s,line=%d\r\n" << error_code 
        << cudaGetErrorName(error_code) << cudaGetErrorString(error_code) << filename << lineNumber << std::endl;
        return error_code;
    }
    return error_code;
}