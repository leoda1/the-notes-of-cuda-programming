#include<iostream>
#include<cuda_runtime.h>
#include <vector>
using namespace std;


static auto relu = [&] (float x) {
        return max(0.0f, x);
};


class Relu{
public:
    Relu(std::vector<std::vector<std::vector<std::vector<float>>>>& data_ref) : data(data_ref) {};
    ~Relu() = default;
    
    
    // static float relu(float x){
    //     return std::max(0.0f, x);
    // }

    // static auto relu = [&] (float x) {
    //     return max(0.0f, x);
    // };

    static void apply_relu(std::vector<std::vector<std::vector<std::vector<float>>>>& data) {
        for (auto& batch : data) {
            for (auto& channel : batch) {
                for (auto& row : channel) {
                    for (auto& element : row) {
                        element = relu(element);
                    }
                }
            }
        }
    }
    
    __global__ void relu_kernel(float* d_data, int b, int c, int h, int w) {
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix < w && iy < h) {
            for (int i = 0; i < c; i++) {
                for (int j = 0; j < b; j++) {
                    int idx = j * c * h * w + i * h * w + iy * w + ix;
                    d_data[idx] = relu(d_data[idx]);
                }
            }
        }
    }
    
    void apply_relu_gpu(std::vector<std::vector<std::vector<std::vector<float>>>>& data) {
        int b = data.size();
        int c = data[0].size();
        int h = data[0][0].size();
        int w = data[0][0][0].size();

        // Allocate memory on device
        float* d_data;
        size_t dataSize = b * c * h * w * sizeof(float);
        cudaMalloc((void**)&d_data, dataSize);

        // Prepare host data (flattened)
        std::vector<float> h_data(b * c * h * w);
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < h; k++) {
                    for (int l = 0; l < w; l++) {
                        int idx = i * c * h * w + j * h * w + k * w + l;
                        h_data[idx] = data[i][j][k][l];
                    }
                }
            }
        }

        // Copy host data to device
        cudaMemcpy(d_data, h_data.data(), dataSize, cudaMemcpyHostToDevice);

        // Launch kernel with appropriate grid and block dimensions
        int blockSize = 32;
        dim3 blockDim(blockSize, blockSize);
        dim3 gridDim((w + blockSize - 1) / blockSize, (h + blockSize - 1) / blockSize);

        relu_kernel<<<gridDim, blockDim>>>(d_data, b, c, h, w);
        cudaDeviceSynchronize();

        // Copy the result back to the host
        cudaMemcpy(h_data.data(), d_data, dataSize, cudaMemcpyDeviceToHost);

        // Copy the data back into the original 4D vector
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < c; j++) {
                for (int k = 0; k < h; k++) {
                    for (int l = 0; l < w; l++) {
                        int idx = i * c * h * w + j * h * w + k * w + l;
                        data[i][j][k][l] = h_data[idx];
                    }
                }
            }
        }

        // Free the device memory
        cudaFree(d_data);
    }

private:
    std::vector<std::vector<std::vector<std::vector<float>>>>& data;
};

int main(){
    // input data;
    std::vector<std::vector<std::vector<std::vector<float>>>> data = {
        {
            {{1.0, 2.0},{3.0, 4.0}},
            {{1.0, 2.0},{3.0, 4.0}},
            {{1.0, 2.0},{3.0, 4.0}}
        },
        {
            {{1.0, 2.0},{3.0, 4.0}},
            {{1.0, 2.0},{3.0, 4.0}},
            {{1.0, 2.0},{3.0, 4.0}}
        }
    };
    Relu relu(data);
    bool use_gpu = true;

    if (use_gpu) {
        relu.apply_relu_gpu(data);
    } else {
        Relu::apply_relu(data);
    }

    
    return 0;
}