#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Sigmoid 激活函数
__device__ float sigmoidf(float in) {
    return 1.f / (1.f + expf(-in));
}

// ReLU 激活函数
__device__ float reluf(float in) {
    return fmaxf(0.f, in);
}

// 矩阵乘法函数
__global__ void Matmul(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float tmp = 0.f;
        for (int i = 0; i < n; i++) {
            tmp += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = tmp;
    }
}

// Sigmoid 算子
__global__ void Sigmoid(float *A, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = sigmoidf(A[idx]);
    }
}

// ReLU 算子
__global__ void ReLU(float *A, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = reluf(A[idx]);
    }
}

// 融合三个算子后的全局内核函数
__global__ void kernel_0(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算Sigmoid算子
    if (row < m && col < n) {
        float sigmoid_val = sigmoidf(A[row * n + col]);
        A[row * n + col] = sigmoid_val; // 原地更新A
    }
    __syncthreads(); // 确保所有线程完成Sigmoid计算

    // 计算矩阵乘法算子
    if (row < m && col < k) {
        float tmp = 0.f;
        for (int i = 0; i < n; i++) {
            tmp += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = tmp;
    }
    __syncthreads(); // 确保所有线程完成矩阵乘法计算
    cudaDeviceSynchronize();

    // 计算ReLU算子
    if (row < m && col < k) {
        C[row * k + col] = reluf(C[row * k + col]);
    }
}

int main() {
    int m = 1024, n = 1024, k = 128;
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * k * sizeof(float);
    size_t size_C = m * k * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    // 初始化h_A和h_B
    for (int i = 0; i < m * n; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n * k; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (k + blockDim.y - 1) / blockDim.y);

    // 测试分开执行三个算子的耗时
    float *d_temp1, *d_temp2;
    cudaMalloc(&d_temp1, size_A);
    cudaMalloc(&d_temp2, size_C);


    // 测试融合内核的耗时
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    Sigmoid<<<(m * n + 255) / 256, 256>>>(d_A, d_temp1, m * n);
    Matmul<<<gridDim, blockDim>>>(d_temp1, d_B, d_temp2, m, n, k);
    ReLU<<<(m * k + 255) / 256, 256>>>(d_temp2, d_C, m * k);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float separate_time = 0;
    cudaEventElapsedTime(&separate_time, start1, stop1);
    std::cout << "Separate kernels time: " << separate_time << " ms" << std::endl;

    // 测试融合内核的耗时
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    size_t sharedMemSize = (m * n + m * k) * sizeof(float);
    kernel_0<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, m, n, k);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float fused_time = 0;
    cudaEventElapsedTime(&fused_time, start2, stop2);
    std::cout << "Fused kernel time: " << fused_time << " ms" << std::endl;

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}