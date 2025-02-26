#include "../../cutlass/include/cutlass/cutlass.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include <iostream>

#define N 32*1024*1024
#define BLOCK_SIZE 256
// 归约核函数
template <typename ReductionOp>
__global__ void reduce_with_cutlass(
    typename ReductionOp::ElementInput* g_idata, 
    typename ReductionOp::ElementOutput* g_odata,
    int num_elements_per_block
) {
    using Reduction = cutlass::reduction::device::Reduce<
        typename ReductionOp::ElementInput,
        typename ReductionOp::ElementOutput,
        ReductionOp, 
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::arch::OpClassSimt, 
        cutlass::arch::Sm80
    >;

    // 声明线程块级别的归约对象
    __shared__ typename ReductionOp::ElementInput shared_memory[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 加载数据到共享内存
    if (idx < N) {
        shared_memory[tid] = g_idata[idx];
    } else {
        shared_memory[tid] = 0.0f;
    }
    __syncthreads();

    // 在共享内存中进行归约
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_memory[tid] += shared_memory[tid + offset];
        }
        __syncthreads();
    }

    // 输出最终的归约结果到全局内存
    if (tid == 0) {
        g_odata[blockIdx.x] = shared_memory[0];
    }
}

// 初始化归约操作
template <typename ElementInput, typename ElementOutput>
void run_reduction(
    ElementInput* input_data, 
    ElementOutput* output_data, 
    int num_elements_per_block, 
    int num_blocks
) {
    using ReductionOp = cutlass::reduction::operation::Sum<ElementInput>;
    dim3 grid(num_blocks, 1);
    dim3 block(BLOCK_SIZE, 1);

    // 启动归约核
    reduce_with_cutlass<ReductionOp><<<grid, block>>>(input_data, output_data, num_elements_per_block);
    cudaDeviceSynchronize();
}

int main() {
    // 初始化输入数据
    float *input_host = (float*)malloc(N * sizeof(float));
    float *input_device;
    cudaMalloc((void**)&input_device, N * sizeof(float));
    
    // 填充数据
    for (int i = 0; i < N; i++) input_host[i] = 2.0f;
    cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义输出
    const int block_num = 1024;
    float *output_host = (float*)malloc(block_num * sizeof(float));
    float *output_device;
    cudaMalloc((void**)&output_device, block_num * sizeof(float));

    // 执行归约
    run_reduction(input_device, output_device, N / block_num, block_num);
    
    // 获取并打印结果
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final = 0.0f;
    for (int i = 0; i < block_num; i++) {
        if (i < 10) {
            std::cout << "Block " << i << " result: " << output_host[i] << std::endl;
        }
        final += output_host[i];
    }
    
    // 输出归约后的总和
    std::cout << "Final result after reduction: " << final << std::endl;

    // 释放内存
    cudaFree(input_device);
    cudaFree(output_device);
    free(input_host);
    free(output_host);

    return 0;
}