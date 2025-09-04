#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gemm_kernel(const float* Q_head, const float* K_head, float* S, 
    int N, int d_k, int ldQ, int ldK) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < N && ty < N) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i ++) {
            sum += Q_head[tx * ldQ + i] * K_head[ty * ldK + i];
        }
        S[tx * N + ty] = sum;
    }
}

__global__ void softmax(float* S, int N, float scale) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    float* row_data = S + row * N;
    __shared__ float sdata[256];

    // 找最大值
    float max_val = -1e20f;
    for (int i = tid; i < N; i += blockSize) {
        float val = row_data[i] * scale;
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();

    // 计算exp和
    float exp_sum = 0.0f;
    for (int i = tid; i < N; i += blockSize) {
        float val = expf((row_data[i] * scale) - max_val);
        row_data[i] = val;
        exp_sum += val;
    }
    sdata[tid] = exp_sum;
    __syncthreads();

    // 归约
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    exp_sum = sdata[0];

    // 归一化
    for (int i = tid; i < N; i += blockSize) {
        row_data[i] = row_data[i] / exp_sum;
    }
}

__global__ void matmul_softmax_output_kernel(float* S, const float* V, float* output_head, int N, int d_k, int ldV) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (col < d_k) {
        float* S_row = S + row * N;
        float sum = 0.0f;
        for (int k = 0; k < N; k ++) {
            sum += S_row[k] * V[k * ldV + col];
        }
        output_head[row * ldV + col] = sum;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
    int d_k = d_model / h;
    float* S = nullptr;
    cudaMalloc(&S, N * N * sizeof(float));
    float scale_val = 1.0f / sqrtf(static_cast<float>(d_k));
    dim3 gemm_block(16, 16);
    dim3 gemm_grid((N + 15) / 16, (N + 15) / 16);

    for (int head_id = 0; head_id < h; head_id ++) {
        const float* Q_head = Q + head_id * d_k;
        const float* K_head = K + head_id * d_k;
        const float* V_head = V + head_id * d_k;
        float* output_head = output + head_id * d_k;

        gemm_kernel<<<gemm_grid, gemm_block>>>(Q_head, K_head, S, N, d_k, d_model, d_model);
        softmax<<<N, 256>>>(S, N, scale_val);
        matmul_softmax_output_kernel<<<N, d_k>>>(S, V_head, output_head, N, d_k, d_model);
    }
    cudaFree(S);
}

int main() {
    int N = 4;          // 序列长度
    int d_model = 8;    // 向量维度
    int h = 2;          // 注意力头数

    size_t size = N * d_model * sizeof(float);
    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // 初始化输入
    for (int i = 0; i < N * d_model; i++) {
        h_Q[i] = 0.01f * (i+1);
        h_K[i] = 0.02f * (i+1);
        h_V[i] = 0.03f * (i+1);
    }

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    solve(d_Q, d_K, d_V, d_output, N, d_model, h);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d_model; j++) {
            printf("%.4f ", h_output[i * d_model + j]);
        }
        printf("\n");
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);

    return 0;
}

