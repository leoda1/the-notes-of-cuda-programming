#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

void Matmul_device(float* M_host, float* N_host, float* P_host, int size, int blocksize);
#endif // __MATMUL_HPP__