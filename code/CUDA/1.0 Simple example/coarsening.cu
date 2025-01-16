#include <stdio.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void vecadd (float *A, float *B, float *C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vecaddCoarsened(float *A, float *B, float *C) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
    if (idx + 1 < N) {
        C[idx + 1] = A[idx + 1] + B [idx + 1];
    }
    
}

void random_init (float *data, int size) {
    for (int i = 0; i < size; i ++ ) {
        data[i] = rand() / (float)RAND_MAX;
    }
}

int main() {
    float *a,   *b,   *c;
    float *d_a, *d_b, *d_c;
    int size = N * sizeof(float);

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = (float *)malloc(size); random_init(a, N);
    b = (float *)malloc(size); random_init(b, N);
    c = (float *)malloc(size);

    cudaEvent_t start, stop, startCoarsened, stopCoarsend;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startCoarsened);
    cudaEventCreate(&stopCoarsend);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //warm up
    vecadd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    vecaddCoarsened<<<(N + 2 * THREADS_PER_BLOCK - 1) / (2 * THREADS_PER_BLOCK), THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    //start kernel 1
    cudaEventRecord(start);
    vecadd<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //start kernel 2
    cudaEventRecord(startCoarsened);
    vecaddCoarsened<<<(N + 2 * THREADS_PER_BLOCK - 1) / (2 * THREADS_PER_BLOCK), THREADS_PER_BLOCK >>>(d_a, d_b, d_c);
    cudaEventRecord(stopCoarsend);
    cudaEventSynchronize(stopCoarsend);
    
    float mscends = 0, mscends_coarsened = 0;
    cudaEventElapsedTime(&mscends, start, stop);
    cudaEventElapsedTime(&mscends_coarsened, startCoarsened, stopCoarsend);
    printf ("vecadd time: %.6f ms\n", mscends);
    printf ("vecadd with coarsened: %.6f ms\n", mscends_coarsened);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startCoarsened);
    cudaEventDestroy(stopCoarsend);
    return 0;
    
}