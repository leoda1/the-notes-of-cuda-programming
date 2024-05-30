#include <iostream>

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // Number of elements
    int N = 512;

    // Allocate host memory
    float* h_A = (float*)malloc(N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));
    float* h_C = (float*)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


// #include<iostream>

// __global__ void mykernelfunc(){};

// int main()
// {
//     mykernelfunc<<<1,1>>>();
//     std::cout<<"hello world cuda!\n";
//     return 0;
// }
