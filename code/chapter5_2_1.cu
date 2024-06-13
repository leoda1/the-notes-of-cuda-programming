#include <cuda_runtime.h>
#include <stdio.h>
#define N 3

__global__ void cluster_kernel(float *input, float *output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

int main() {
    float *input, *output;
    // 为输入和输出数组分配设备内存
    cudaMalloc((void**)&input, N * sizeof(float));
    cudaMalloc((void**)&output, N * sizeof(float));

    // 示例输入数据
    float h_input[N] = {1.0f, 2.0f, 3.0f};
    cudaMemcpy(input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(N, 1);
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    void *args[] = { &input, &output };
    cudaLaunchKernelExC(&config, cluster_kernel, args);
    
    cudaDeviceSynchronize();
    float h_output[N];
    cudaMemcpy(h_output, input, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; ++i) {
        printf("Output[%d] = %f\n", i, h_output[i]);
    }
    // for (int i = 0; i < 3; ++i) {
    //     std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
    // }
    // 清理设备内存
    cudaFree(input);
    cudaFree(output);
    return 0;
}
// static __inline__ __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
//                                                                const void *func,
//                                                                   void **args)