#include<iostream>
#include "../code/common.cuh"

/*
__global__ void addGPU(float* a, float* b, float* c, const int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;
    c[idx] = a[idx] + b[idx];
}

void initialData(float* addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xff) / 10.f;
    }
    return;
}

int main()
{
    setGPU();

    int iElemCount = 512;
    size_t stByteCount = iElemCount * sizeof(float);

    float *a, *b, *c;
    a = (float*)malloc(stByteCount);
    b = (float*)malloc(stByteCount);
    c = (float*)malloc(stByteCount);
    if(a != NULL && b != NULL && c != NULL)
    {
        memset(a, 0, stByteCount);
        memset(b, 0, stByteCount);
        memset(c, 0, stByteCount);
    }
    else
    {
        std::cout << "Memory allocation failed." << std::endl;
        exit(-1);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, stByteCount);
    cudaMalloc((float**)&d_b, stByteCount);
    cudaMalloc((float**)&d_c, stByteCount);
    if(d_a != NULL && d_b != NULL && d_c != NULL)
    {
        cudaMemset(d_a, 0, stByteCount);
        cudaMemset(d_b, 0, stByteCount);
        cudaMemset(d_c, 0, stByteCount);
    }
    else
    {
        std::cout << "Memory allocation on GPU failed." << std::endl;
        free(a);
        free(b);
        free(c);
        exit(-1);
    }
    srand(666);
    initialData(a, iElemCount);
    initialData(b, iElemCount);
    
    cudaMemcpy(d_a, a, stByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, stByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, stByteCount, cudaMemcpyDeviceToHost);

    dim3 block(32);
    dim3 grid(iElemCount / 32);

    addGPU<<<grid, block>>>(d_a, d_b, d_c, iElemCount);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, stByteCount, cudaMemcpyDeviceToHost);

    for (int i = 0; i < iElemCount; i++)
    {
        std::cout << "a[" << i << "] = " << a[i] << " ";
        std::cout<<"b[" << i << "] = " << b[i] << " ";
        std::cout<<"c[" << i << "] = " << c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
*/

int main()
{
    //1.分配主机内存，并初始化
    float *fpHost_A;
    fpHost_A = (float *)malloc(4);
    memset(fpHost_A, 0, 4);//初始化为0

    //2.分配设备内存

    float *fpDev_A;
    cudaError_t err = ErrorCheck(cudaMalloc((float**)&fpDev_A, 4), __FILE__, __LINE__);
    cudaMemset(fpDev_A, 0, 4);//*fpDev_A = 0;

    //3.拷贝数据到设备内存
    ErrorCheck(cudaMemcpy(fpDev_A, fpHost_A, 4, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //释放主机和设备内存
    free(fpHost_A);
    ErrorCheck(cudaFree(fpDev_A), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);


    return 0;  
}