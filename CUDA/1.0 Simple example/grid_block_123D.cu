// #include <cuda_runtime.h>
#include <stdio.h>


__global__ void print_idx_kernel(){
    printf("block idx: (%3d, %3d, %3d), thread idx: (%3d, %3d, %3d)\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ void print_dim_kernel(){
    printf("grid dimension: (%3d, %3d, %3d), block dimension: (%3d, %3d, %3d)\n",
         gridDim.z, gridDim.y, gridDim.x,
         blockDim.z, blockDim.y, blockDim.x);
}

__global__ void print_thread_idx_per_block_kernel(){
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index);
}

__global__ void print_thread_idx_per_grid_kernel(){
    int bSize  = blockDim.z * blockDim.y * blockDim.x;

    int bIndex = blockIdx.z * gridDim.x * gridDim.y + \
               blockIdx.y * gridDim.x + \
               blockIdx.x;

    int tIndex = threadIdx.z * blockDim.x * blockDim.y + \
               threadIdx.y * blockDim.x + \
               threadIdx.x;

    int index  = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n", 
         bIndex, tIndex, index);
}

__global__ void print_cord_kernel(){
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int z  = blockIdx.z * blockDim.z + threadIdx.z;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d, %3d)\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index, x, y, z);
}

void print_one_dim(){
    int inputSize = 8;
    int blockDim = 4;
    int gridDim = inputSize / blockDim;

    dim3 block(blockDim);
    dim3 grid(gridDim);
    printf("grid dimension:         %d, block dimension:            %d,\n",
           grid.x,
           block.x);
    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    // print_cord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}

void print_two_dim(){
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);
    printf("grid dimension:    (%d, %d), block dimension: (%d, %d)\n",
           grid.y, grid.x,
           block.y, block.x);
    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    // print_cord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}

void print_three_dim(){
    // 设置网格和块的三维大小
    int depth = 3;    // 深度
    int height = 3;   // 高度
    int width = 3;    // 宽度

    // 块的大小，这里设置为每个维度2个线程
    int blockDim = 2;
    
    // 计算需要的网格大小，确保覆盖所有的数据点
    dim3 block(blockDim, blockDim, blockDim);  // 每个块的维度
    dim3 grid((width + blockDim - 1) / blockDim, 
              (height + blockDim - 1) / blockDim,
              (depth + blockDim - 1) / blockDim);  // 网格的维度
    printf("grid dimension: (%d, %d, %d), block dimension: (%d, %d, %d)\n",
           grid.z, grid.y, grid.x,
           block.z, block.y, block.x);
    // 调用你的kernel函数来打印索引和其他信息
    // 你可以基于需要激活以下kernel中的一个或多个
    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    print_cord_kernel<<<grid, block>>>();

    // 确保CUDA操作完成，方便看到打印输出
    cudaDeviceSynchronize();
}

int main() {
    /*
    synchronize是同步的意思，有几种synchronize方法：
    cudaDeviceSynchronize: CPU与GPU端完成同步，CPU不执行之后的语句，知道这个语句以前的所有cuda操作结束
    cudaStreamSynchronize: 跟cudaDeviceSynchronize很像，但是这个是针对某一个stream的。只同步指定的stream中的cpu/gpu操作，其他的不管
    cudaThreadSynchronize: 现在已经不被推荐使用的方法
    __syncthreads:         线程块内同步
    */

    // print_one_dim();
    // print_two_dim();
    print_three_dim();

    return 0;
}
