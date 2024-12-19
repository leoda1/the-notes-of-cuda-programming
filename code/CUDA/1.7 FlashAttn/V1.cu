#include <torch/csrc/api/include/torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/***************************************************************************************************
 *                                        Flashattention-V1
****************************************************************************************************/

__global__ void forward_kernel(const float* Q, const float* K, const float* V, const int N,
                               const int d, const int Tc, const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float* l, float* m, float* o) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset  = (bx * gridDim.y * N)     + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j ++) {
        for (int x = 0; x < d; x ++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();
        
        for (int i = 0; i < Tr; i ++) {
            for (int x = 0; x < d; x ++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }
            
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                o[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * o[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
            
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Require:
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d  = Q.size(3);
    
    // 1: Set block size Bc, Br
    const int Bc = 32; const int Br = 32;

    // 2: Initialize o, l, m in HBM
    auto o = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    // 3: Devide Q into Tr, and devide K, V into Tc
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // 
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size = 0;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    std::cout << "Max shared memoty: " << max_sram_size << "requset shared memory: " << sram_size << std::endl;

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);
    forward_kernel<<<grid_dim, block_dim, sram_size, 0>>> (
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), o.data_ptr<float>()
    );

    return o;

}

