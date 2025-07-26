#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

__inline__ __device__
float blockReduceMin(float val) {
	static __shared__ float shm[32];
	int tid = threadIdx.x;
	int bid = blockDim.x; 
	int lane = tid % 32, wid = tid / 32;
	for (int i = 16; i > 0; i /= 2) {
		val = fminf(val, __shfl_down_sync(0xffffffff, val, i));
	}
	if (lane == 0) shm[wid] = val;
	__syncthreads();

	val = (tid < bid / 32) ? shm[lane] : FLT_MAX;	
	if(wid == 0) {
		for (int i = 16; i >0; i /= 2) {
			val = fminf(val, __shfl_down_sync(0xffffffff, val, i));
		}
	}
	return val;
}


__global__ void reduce_min_warp(const float* __restrict__ input, float* output, int N) {
	float val = FLT_MAX;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N) val=input[idx];
	val = blockReduceMin(val);
	if(threadIdx.x ==0) {
		output[blockIdx.x]=val;
	}
}

void device_func(float* h_input, int n) {
	float *d_input, *d_output, *d_inter;
	cudaMalloc(&d_input, sizeof(float) * n);
	cudaMalloc(&d_output, sizeof(float) * n);
	cudaMalloc(&d_inter, sizeof(float) * n);
	size_t size = n * sizeof(float);
	cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(1024, 1);
	dim3 dimGrid((n + 1024 - 1) / 1024, 1);
	// first stage 
	reduce_min_warp<<<dimGrid, dimBlock, 0, nullptr>>>(d_input, d_inter, n);
	// second stage
	reduce_min_warp<<<1, 1024, 0, nullptr>>>(d_inter, d_output, 1024);
	float results;
	cudaMemcpy(&results, d_output, sizeof(float), cudaMemcpyDeviceToHost);

	printf("results : %lf\n", results);  
	cudaFree(d_input);
	cudaFree(d_inter);
	cudaFree(d_output);
}

int main() {
  const int N = 1 << 20; // 1MB
  float* h_data = new float[N];
  for (int i = 0; i < N; i ++) {
		h_data[i] = static_cast<float>(rand() % 10000);
	}
	device_func(h_data, N);
	delete h_data;
	return 0;

}
