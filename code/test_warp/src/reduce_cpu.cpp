#include "reduce.hpp"

float ReduceOnCPU(float* data, int const size)
{
    float sum = 0;
    for (int i = 0; i < size; ++i)
    {
        sum += data[i];
    }
    return sum;
}