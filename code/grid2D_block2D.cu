/******************************************************************
 * Author: Da Liu
 * Date: 2024-07-25
 * File: grid2D_block2D.cu
 * Description: 组织线程模型：二维网格二维线程块计算二维矩阵加法.
 *****************************************************************/

#include <stdio.h>
#include "../cudalearn/tools/common.cuh"

__global__ void add_matrix(int *a, int *b, int *c, const int nx, const int ny) 
{
    /* 2D网格二维/一维线程块计算二维矩阵加法 
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) {
        c[idx] = a[idx] + b[idx];
    } */

   /* 1D网格一维线程块计算二维矩阵加法 */
    int ix= blockIdx.x * blockDim.x + threadIdx.x;;
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy ++)
        {
            int idx = iy * nx + ix;
            c[idx] = a[idx] + b[idx];
        }
    }

}

int main()
{
    setGPU();                                           //设置GPU设备
    
    int nx = 16, ny = 8;                                //矩阵大小
    int nxy = nx * ny;                                  //矩阵元素个数
    size_t stBytesCount = nxy * sizeof(int);            //矩阵元素字节数

    int *ipHost_A, *ipHost_B, *ipHost_C;                //主机内存
    ipHost_A = (int* )malloc(stBytesCount);
    ipHost_B = (int* )malloc(stBytesCount);
    ipHost_C = (int* )malloc(stBytesCount);
    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        for(int i = 0; i < nxy; i++)
        {
            ipHost_A[i] = i;                             //矩阵A元素初始化为0到nxy-1
            ipHost_B[i] = i + 1;                         //矩阵B元素初始化为A元素+1
        }
        memset(ipHost_C, 0, stBytesCount);               //初始化矩阵C为0
    }
    else
    {
        printf("Memory allocation failed!\n");
        exit(-1);
    }
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;           //设备内存
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, stBytesCount), __FILE__, __LINE__);
    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }
    else
    {
        printf("Device Memory copy failed!\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(1);
    }

    dim3 block(4, 1);
    // dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);       //2Dgrid2Dblock
    // dim3 grid((nx + block.x - 1) / block.x, ny);                                 //1Dgrid1Dblock
    dim3 grid((nx + block.x -1) / block.x, 1);
    printf("Grid: %d, %d\n", grid.x, grid.y);
    printf("Block: %d, %d\n", block.x, block.y);

    add_matrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    for (int i = 0; i < 10; i++)
    {
        printf("id = %d, matrix_A = %d, matrix_B = %d, matrix_C = %d\n", i + 1, ipHost_A[i], ipHost_B[i], ipHost_C[i]);
    }
    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);
    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__);
    return 0;
}