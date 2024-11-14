#include<iostream>
#include<cuda_runtime.h>
#include <vector>


float relu(float x){
    return std::max(0.0f, x);
}

void apply_relu(std::vector<std::vector<std::vector<std::vector<float>>>>& data){
    for (auto& batch_size : data){
        for (auto& channel : batch_size){
            for (auto& hight : channel){
                for (auto& width : hight){
                    width = relu(width);
                }
            }
        }
    }

    for (const auto& batch_size : data){
        for (const auto& channel : batch_size){
            for(const auto& hight : channel){
                for (const auto& width : hight){
                    std::cout << width << ""; 
                }
            }
            std::cout << "\n";
        }
    }
}

__global__ void relu_kernel(float* d_data, int b, int c, int h, int w){

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    while( ix < w && iy < h)
        for ( int i = 0; i < c; i ++){
            for ( int j = 0; j < b; j ++ ){
                d_data[j*c*h*w + i*h*w +iy*w + ix] = relu(d_data[j*c*h*w + i*h*w +iy*w + ix]);
            }
        }

}

void apply_relu_gpu(std::vector<std::vector<std::vector<std::vector<float>>>>& data)
{
    // data dimensions
    int b = data.size();
    int c = data[0].size();
    int h = data[0][0].size();
    int w = data[0][0][0].size();

    // kernel parameters
    int blocksize = 32;
    int size = b * c * h * w * sizeof(float);

    // dataload H2D
    float* d_data;
    cudaMalloc((void**)&d_data, size);
    float* h_data = new float[b * c * h * w];

    for ( int i = 0; i < b; i ++){
        for ( int j = 0; j < c; j ++){
            for ( int k = 0; k < h; k ++){
                for ( int l = 0; l < w; l ++){
                    h_data[i*c*h*w + j*h*w + k*w + l] = data[i][j][k][l];
                }
            }
        }
    }

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // kernel function
    int gridsize = (w + blocksize - 1) / blocksize;
    dim3 blockDim(blocksize, blocksize);
    dim3 gridDim(gridsize, gridsize);

    relu_kernel<<<blockDim, gridDim, 0, nullptr>>>(d_data, b, c, h, w);
    cudaDeviceSynchronize();

    // databack D2H;
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    // print res;
    for ( int i = 0; i < b; i ++){
        for ( int j = 0; j < c; j ++){
            for ( int k = 0; k < h; k ++){
                for ( int l = 0; l < w; l ++){
                    data[i][j][k][l] = h_data[i*c*h*w + j*h*w + k*w + l];
                }
            }
        }
    }
}

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

    //apply_relu(data);
    apply_relu_gpu(data);
    return 0;
}