#include <iostream>
#include <cuda_runtime.h>
#include <nccl.h>
#include <infiniband/verbs.h>

// 假设你有两个 GPU 和一个 InfiniBand 适配器

#define N 1024  // 数据大小（每个 GPU 上的数据大小）
#define BLOCK 256  // 每个块的大小


// 错误检查函数
#define CHECK_CUDA(call)                                   \
  {                                                         \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  }

#define CHECK_NCCL(call)                                    \
  {                                                         \
    ncclResult_t res = call;                                \
    if (res != ncclSuccess) {                               \
      std::cerr << "NCCL error: " << ncclGetErrorString(res) << std::endl; \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  }


// 简单的 NCCL 全归约操作函数
void gpudirect_rdma_example() {
    int size = N * sizeof(float);
    float *d_a, *d_b, *d_c;  // 设备内存指针
    float *h_a = new float[N]; // 主机内存指针
    float *h_b = new float[N]; // 主机内存指针

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = (i + 1) * 2.0f;
    }

    // 分配 GPU 内存
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    // 将数据从主机复制到 GPU
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 初始化 NCCL
    ncclComm_t comm;
    int device_count = 2;  // 假设有 2 个 GPU
    CHECK_NCCL(ncclCommInitAll(&comm, device_count, NULL));

    // 进行数据传输：在两个 GPU 之间进行全归约（将数据加和）
    CHECK_NCCL(ncclAllReduce(d_a, d_c, N, ncclFloat, ncclSum, comm, cudaStreamDefault));

    // 等待流完成
    CHECK_CUDA(cudaDeviceSynchronize());

    // 将计算结果从 GPU 复制回主机
    CHECK_CUDA(cudaMemcpy(h_a, d_c, size, cudaMemcpyDeviceToHost));

    // 打印一部分结果（仅查看前 10 个元素）
    for (int i = 0; i < 10; i++) {
        std::cout << "Result[" << i << "]: " << h_a[i] << std::endl;
    }

    // 释放资源
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    CHECK_NCCL(ncclCommDestroy(comm));
}

int main() {
    gpudirect_rdma_example();
    return 0;
}