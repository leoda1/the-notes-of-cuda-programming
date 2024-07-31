#ifndef __REDUCE_HPP__
#define __REDUCE_HPP__

void ReduceOnGPUWithDivergence(float *h_idata, float *h_odata, int size, int blockSize);
void ReduceOnGPUWithoutDivergence(float *h_idata, float *h_odata, int size, int blockSize);
extern float ReduceOnCPU(float *data, int const size);

#endif 