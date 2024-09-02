#ifndef MATMUL_HPP
#define MATMUL_HPP

void Matmul_device(float* h_M, float* h_N, float* h_P, int width, int block_size);
void Matmul_device_shared(float* h_M, float* h_N, float* h_P, int width, int block_size);


#endif // MATMUL_HPP
