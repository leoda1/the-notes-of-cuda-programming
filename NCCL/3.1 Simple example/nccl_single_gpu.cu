#include <iostream>
#include <nccl.h>
#include <cuda_runtime.h>

// 检查 CUDA 调用是否成功的宏
#define CUDA_CALL(cmd) do {                       \
    cudaError_t e = cmd;                          \
    if (e != cudaSuccess) {                       \
        std::cerr << "CUDA Error: "               \
                  << cudaGetErrorString(e)        \
                  << " at " << __FILE__           \
                  << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                       \
    }                                             \
} while(0)

// 检查 NCCL 调用是否成功的宏
#define NCCL_CALL(cmd) do {                       \
    ncclResult_t r = cmd;                         \
    if (r != ncclSuccess) {                       \
        std::cerr << "NCCL Error: "               \
                  << ncclGetErrorString(r)        \
                  << " at " << __FILE__           \
                  << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);                       \
    }                                             \
} while(0)

int main() {
    const int numElements = 4;
    float *d_sendBuff, *d_receBuff;
    float h_sendBuff[numElements] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_receBuff[numElements] = {0.0f};

    int device_id = 0;
    cudaSetDevice(device_id);

    CUDA_CALL(cudaMalloc((void **)&d_sendBuff, numElements * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&d_receBuff, numElements * sizeof(float)));

    CUDA_CALL(cudaMemcpy(d_sendBuff, h_sendBuff, numElements * sizeof(float), cudaMemcpyHostToDevice));
    ncclComm_t comm;
    ncclUniqueId id;
    NCCL_CALL(ncclGetUniqueId(&id)); // 获取 NCCL 通信唯一 ID
    NCCL_CALL(ncclCommInitRank(&comm, 1, id, 0)); // 1个线程 rank 0
    NCCL_CALL(ncclAllReduce(d_sendBuff, d_receBuff, numElements, ncclFloat,
                            ncclSum, comm, cudaStreamDefault));
    CUDA_CALL(cudaMemcpy(h_receBuff, d_receBuff, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    // 输出结果
    std::cout << "Result after NCCL All-Reduce: ";
    for (int i = 0; i < numElements; ++i) {
        std::cout << h_receBuff[i] << " ";
    }
    std::cout << std::endl;
    // 释放资源
    NCCL_CALL(ncclCommDestroy(comm));
    CUDA_CALL(cudaFree(d_sendBuff));
    CUDA_CALL(cudaFree(d_receBuff));

    return 0;

}