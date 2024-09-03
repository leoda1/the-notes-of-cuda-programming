#ifndef MATMUL_HPP
#define MATMUL_HPP

void Matmul_device(float* h_M, float* h_N, float* h_P, int width, int block_size);
void Matmul_device_shared(float* h_M, float* h_N, float* h_P, int width, int block_size, bool staticMem, bool bank_conflict);
void Matmul_device_shared_bank_conflict(float* h_M, float* h_N, float* h_P, int width, int block_size, bool staticMem, bool bank_conflict);
void Matmul_device_shared_bank_conflict_pad_resolve(float* h_M, float* h_N, float* h_P, int width, int block_size, bool staticMem, bool bank_conflict);


#endif
