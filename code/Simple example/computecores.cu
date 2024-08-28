/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-25
 * File: computcores.cu
 * Description: 计算当前显卡的核心数量.
 *****************************************************************/

#include <stdio.h>
#include "../cudalearn/tools/common.cuh"

int getSPcores(cudaDeviceProp deviceProp, const char** archName)
{
    int cores = 0;
    int mp = deviceProp.multiProcessorCount;
    switch (deviceProp.major)
    {
        case 2: // Fermi
         *archName = "Fermi";
         if (deviceProp.minor == 1) cores = mp * 48;
         else cores = mp * 32;
         break;
        case 3: // Kepler
         *archName = "Kepler";
         cores = mp * 192;
         break;
        case 5: // Maxwell
         *archName = "Maxwell";
         cores = mp * 128;
         break;
        case 6: // Pascal
         *archName = "Pascal";
         if ((deviceProp.minor == 1) || (deviceProp.minor == 2)) cores = mp * 128;
         else if (deviceProp.major == 0) cores = mp * 64;
         else printf("Unknown GPU architecture\n");
         break;
        case 7: // Volta and Turing
         *archName = "Volta";
         if ((deviceProp.minor == 0) || (deviceProp.minor == 5)) cores = mp * 64;
         else printf("Unknown GPU architecture\n");
         break;
        case 8: // Ampere
         *archName = "Ampere";
         if (deviceProp.minor == 0) cores = mp * 64;
         else if ((deviceProp.minor == 6) || (deviceProp.minor == 9)) cores = mp * 128;
         else printf("Unknown GPU architecture\n");
         break;
        case 9: //Hopper
         *archName = "Hopper";
         if (deviceProp.minor == 0) cores = mp * 128;
         else printf("Unknown GPU architecture\n");
         break;
        default:
         printf("Unknown GPU architecture\n");
         break;
    }
    return cores;
}

int main()
{
    int deviceCount = 0;
    const char* archname = NULL;
    ErrorCheck(cudaSetDevice(deviceCount), __FILE__, __LINE__);

    cudaDeviceProp deviceProp;
    ErrorCheck(cudaGetDeviceProperties(&deviceProp, deviceCount), __FILE__, __LINE__);
    printf("Device:            %s\n", deviceProp.name);
    int spCores = getSPcores(deviceProp, &archname);
    printf("GPU Architecture:  %s\n", archname);
    printf("SP cores:          %d\n", spCores);
    return 0;
}