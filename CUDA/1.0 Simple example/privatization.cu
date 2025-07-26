#include <stdio.h>
#include <cuda_runtime.h>

__global__ void WindowSumDirect (const float *in, float *out, int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = window / 2;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -half; i <= half; i ++) {
            int access = idx + i;
            if (access >= 0 && access < n) {
                sum += in[access];
            }
        }
        out[idx] = sum;
    }
}

__global__ void WindowSumPrivatized (const float *in, float *out, int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = window / 2;
    __shared__ float sharedData[1024 + 5];

    if (idx < n) {
        sharedData[threadIdx.x] = (idx < n) ? in[idx] : 0.0f;
        __syncthreads();
        float sum = 0.0f;
        for (int i = -half; i <= half; i ++) {
            int access = threadIdx.x + i;
            if (access >= 0 && access < blockDim.x && (idx + i) >= 0 && (idx + i) < n) {
                sum += sharedData[access];
            }
        }
        out[idx] = sum;
    }
}

void initializeArray (float *arr, int n) {
    for (int i = 0; i < n; i ++) {
        arr[i] = 1.0f;
    }
}

int main () {
    int n = 1 << 20;
    int window = 5;
    float *in, *out;
    float *d_in, *d_out;

    in = (float*)malloc(n * sizeof(float));
    out = (float*)malloc(n * sizeof(float));
    
    initializeArray(in, n);

    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadperblock = 256;
    int blockpergrid = (n + threadperblock - 1) / threadperblock;

    WindowSumDirect<<<blockpergrid, threadperblock>>>(d_in, d_out, n, window);
    cudaMemcpy(d_in, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    WindowSumPrivatized<<<blockpergrid, threadperblock>>>(d_in, d_out, n, window);
    cudaMemcpy(d_in, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out);
    return 0;


}