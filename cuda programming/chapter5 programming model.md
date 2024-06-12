# Chapter 5. programming model

## 简介
英伟达 《CUDA C++ Programming Guide》 官方文档学习记录笔记 版本【Release 12.4】出版时间【 Apr 22, 2024】

## 目录
- [Chapter 5. programming model](#chapter-5-programming-model)
  - [简介](#简介)
  - [目录](#目录)
- [5.1 kernels](#51-kernels)
- [5.2 Thread Hierarchy线程分级结构](#52-thread-hierarchy线程分级结构)
      - [总结：](#总结)
  - [5.2.1 Thread Block Clusters线程块集群](#521-thread-block-clusters线程块集群)
- [5.3 Memory Hierarchy内存分级结构](#53-memory-hierarchy内存分级结构)
- [5.4 Heterogeneous Programming异构编程](#54-heterogeneous-programming异构编程)
- [5.5 Asynchronous SIMT Programming Model异步SIMD编程模型](#55-asynchronous-simt-programming-model异步simd编程模型)
  - [5.5.1 Asynchronous Operations异步操作](#551-asynchronous-operations异步操作)
- [5.6 Compute Capabilit计算能力](#56-compute-capabilit计算能力)

# 5.1 kernels
**前言：**
1.定义：CUDA C++通过定义C++函数拓展C++，称为内核（kernels）。调用该内核时，由N个不同的cuda线程并行执行N次。
2.__global__来定义内核，<<<和>>>来执行配置。
3.执行时每个线程（thread）有一个单独ID，该ID可以由内置变量再内核内访问。
```cpp
//定义kernel
#include<cuda_runtime.h>
#include<iostream>

__global__ void Vecadd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    //N个线程的内核调用，这里的每个线程都会执行一次C[i] = A[i] + B[i]的加法操作
    VecAdd<<<1, N>>>(A, B, C);
    return 0;
}
```
注意事项：
1.核函数只能访问GPU内存，不能访问主机内存。除非使用PCie总线或者API来实现数据的交互。
2.核函数写的时候需要明确参数个数。
3.核函数不能使用静态变量。
4.核函数不能使用函数指针。
5.核函数具有异步性。CPU执行main的时候完成需要等待GPU执行完毕使用cudaDeviceSynchronize()来同步数据。
# 5.2 Thread Hierarchy线程分级结构
上一个代码当中的threaIdx.x就是当前线程在块中的索引，它表示线程在线程块中的位置。线程块（thread block）：由N个线程组成的集合，每个线程块都有自己的线程ID。可以用一维，二维和三维的线程块对线程进行划分，线程块的大小由用户指定。这就有了一种自然的方式去计算向量、矩阵等等。
对于一维块，线程的索引和线程的ID是一致的；对于二维块（Dx，Dy），线程的索引是（x, y）对应的线程ID是(x + y*Dx)；对于三维块（Dx，Dy，Dz），线程的索引是（x, y, z）对应的线程ID是(x + y*Dx + z*Dx*Dy)。
```cpp
//
//kernal defination
#include <iostream>

#define N 3  //matrix size
__global__ void MatAdd(float A[N][N], float B[N][N]
                       , float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    //Kernal invocation with ONE block of N*N*1 threads
    float A[N][N] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    float B[N][N] = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };
    float C[N][N];
    
    int numThreads = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numThreads, threadsPerBlock>>>(A, B, C);
    //CUDA同步函数，执行完核函数后继续执行主机程序
    cudaDeviceSynchronize();
    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
```bash
PS C:\Users\22681\Desktop\project\cudalearn> .\build\Debug\hellocuda.exe
Matrix C:
-1.07374e+08 -1.07374e+08 -1.07374e+08
-1.07374e+08 -1.07374e+08 -1.07374e+08
-1.07374e+08 -1.07374e+08 -1.07374e+08(这里的结果不是每个元素都是10，因为threadIdx.x是线程在块中的index不是在矩阵中的index
```
每个块（block）的线程数是有限的，因为所有块都想驻留在同一个核心上。同时，一个内核可以在多个块上并行执行，这样就可以让线程总数等于块数乘以线程数。一个块当中是由一维，二维或三维的网格构成，网格中线程块的数量通常由正在处理的数据的大小决定。
<p align="center">
  <img src="img/image.png" alt="alt text" />
</p>
<p align="center">线程与网格图示</p>
在 CUDA 编程中，启动内核函数时需要指定线程块的数量或每个线程块中线程的数量。这些数量可以是简单的整数（int）或三维向量（dim3）类型。指定线程块用内置变量：blockIdx，指定线程块的维度用内置变量：blockDim。

```cpp
//kernal defination
#include <iostream>

#define N 3  //matrix size
__global__ void MatAdd(float A[N][N], float B[N][N]
                       , float C[N][N]) {
    //blockIdx.x为线程块的编号，blockDim.x为线程块的大小，threadIdx.x为线程的编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main() {
    //Kernal invocation with ONE block of N*N*1 threads
    float A[N][N] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    float B[N][N] = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };
    float C[N][N];
    //设置线程块的大小为3*3
    dim3 gridDim(3, 3);
    //设置线程块的数量为N/3
    dim3 blockDim(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<gridDim, blockDim>>>(A, B, C);
    //CUDA同步函数，执行完核函数后继续执行主机程序
    cudaDeviceSynchronize();
    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
```
这里原文PDF提到使用线程块内的线程可以通过共享内存来共享数据并通过同步它们的执行来协调内存访问。这里是通过__syncthreads()函数来实现的。__syncthreads()函数会等待所有线程都到达该函数调用处，然后再继续执行。线程同步原语参考：Cooperative Groups API，它提供了一组丰富的线程同步原语。
#### 总结：
维度|描述|块在网格的索引|线程在块的索引|线程在网格的索引
-|-|-|-|-
一维情况下 | 假设gridDim=2， blockDim=4，则每个网格中线程块数量为3，ThreadIdx.x的索引范围为0,1,2,3。|gridDim.x|blockDim.x|id = blockIdx.x * blockDim.x + threadIdx.x
二维情况下 | 假设gridDim=(2,2)， blockDim=(4,4)，则每个网格中线程块数量为4，ThreadIdx.x和ThreadIdx.y的索引范围为0,1,2,3。 | blockid = blockIdx.x + blockIdx.y * gridDim.x|Threadid =ThreadIdx.y*blockDim.x+ThreadIdx.x |id=blockid * (blockDim.x * blockDim.y) + threadId
三维情况下 | 假设gridDim=(2,2,2)， blockDim=(4,4,4)，则每个网格中线程块数量为8，ThreadIdx.x,ThreadIdx.y和ThreadIdx.z的索引范围为0,1,2,3。 | blockid = blockIdx.x +blockIdx.y * gridDim.x + gridDim.x*gridDim.y*blockIdx.z|Threadid= ThreadIdx.z*blockDim.x*blockDim.y+ThreadIdx.y*blockDim.x+ThreadIdx.x |id=blockid * (blockDim.x * blockDim.y * blockDim.z) + threadId
## 5.2.1 Thread Block Clusters线程块集群
线程块集群中的不同线程块在GPU处理集群中的共同调度与“线程块中的不同线程在流式多处理器上共同调度相似”。线程块集群也有一维，二维和三维。一个集群最多支持8个线程块。见下图：
<p align="center">
  <img src="img/fig5.png" alt="alt text" />
</p>
<p align="center">线程块集群示意图</p>
线程块集群俩种使用方式。一，使用__cluster_dims__(X, Y, Z)来指定线程块集群的大小。二，使用cuda内核启动的API，cudaLaunchKernelEx。接下来，示例代码展示用编译器时间内核启动集群。如果内核使用编译时集群大小，则在启动内核时无法修改集群大小。

```cpp
∕∕ Kernel definition 官方给的代码
∕∕ Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
_global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output) { }
int main() 
{
    float *input, *output;
    ∕∕ Kernel invocation with compile time cluster sizedim3 threadsPerBlock(16, 16); 
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    ∕∕ The grid dimension is not affected by cluster launch, and is still enumerated ∕∕ using number of blocks. ∕∕ The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output); 
}
//这里的__cluster_dims__(2, 1, 1)本人没成功但是删去后，直接定义集群内核却可以，如下代码：
//kernal defination
#include <cuda_runtime.h>
#include <iostream>
#define N 3
__global__ void cluster_kernel(float *input, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * 2; 
    }
}

int main() {
    float *input, *output;
    float hostInput[N] = {1.0, 2.0, 3.0};
    cudaMalloc((void**)&input,  N * sizeof(float));
    cudaMemcpy(input, hostInput, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&output, N * sizeof(float));
    //设置线程块的大小为3*3
    dim3 threadsPerBlock(3, 3);
    //设置线程块的数量为N/3
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
    // 在完成后记得释放内存
    cudaFree(input);
    cudaFree(output);
}
```
文档内给了另一种实现方法，用cudaLaunchKernelEx这个API，最后成功实现了使用线程块集群计算简单矩阵。
```cpp
//kernal defination
#include <cuda_runtime.h>
#include <iostream>
#define N 3

__global__ void cluster_kernel(float *input, float *output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

int main() {
    float *input, *output;
    // 为输入和输出数组分配设备内存
    cudaMalloc((void**)&input, N * sizeof(float));
    cudaMalloc((void**)&output, N * sizeof(float));

    // 示例输入数据
    float h_input[N] = {1.0f, 2.0f, 3.0f};
    cudaMemcpy(input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(N, 1);
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    void *args[] = { &input, &output };
    cudaLaunchKernelExC(&config, cluster_kernel, args);
    
    cudaDeviceSynchronize();
    float h_output[N];
    cudaMemcpy(h_output, input, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 3; ++i) {
        std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
    }

    // 清理设备内存
    cudaFree(input);
    cudaFree(output);
    return 0;
}
// static __inline__ __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
//                                                                const void *func,
//                                                                  void **args)
// 这是代码库中找到的cudaLaunchKernelExC函数，文档中的cudaLaunchKernelEx有问题。
```

# 5.3 Memory Hierarchy内存分级结构
CUDA 线程在执行过程中可以访问多个内存空间的数据，如图6。每个线程都有私有的本地内存，每个线程块对块的所有线程都有可见的共享内存，每个线程块集群中的线程块可以对彼此的共享内存执行读取、写入和同步操作。所有线程可以访问相同的全局内存。还有两个额外的只读内存空间可供所有线程访问：常量内存空间和纹理内存空间。全局、常量和纹理内存空间针对不同的内存使用进行了优化。纹理内存还为某些特定的数据格式提供不同的寻址模式以及数据过滤。全局、常量和纹理内存空间在同一应用程序启动内核时是持久的。
<p align="center">
  <img src="img/fig6.png" alt="alt text" />
</p>
<p align="center">图6 内存分级结构</p>

# 5.4 Heterogeneous Programming异构编程
如图7所示，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，该设备作为运行C++程序的主机的协处理器运行。例如，当内核在GPU上执行，而C++程序的其余部分在CPU上执行时。主机和设备各自维护自己的独立内存空间，分别称为主机内存和设备内存。主机内存是在CPU上的DRAM，而设备内存是在GPU上的DRAM。
程序通过调用CUDA运行时库（就是我前面代码里面出现的#include cuda_runtime.h）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存分配和释放，以及主机和设备内存之间的数据传输。统一内存提供了一种管理内存的方式，能够桥接主机和设备内存空间。统一内存在系统中的所有CPU和GPU上作为一个单一的、一致的内存图像进行访问，并具有一个共同的地址空间。这种能力允许设备内存的过度订阅，并且可以通过消除显式镜像主机和设备上的数据的需求，大大简化应用程序的移植任务。
<p align="center">
  <img src="img/fig7.png" alt="alt text" />
</p>
<p align="center">图7 cuda为什么是异构编程</p>

# 5.5 Asynchronous SIMT Programming Model异步SIMD编程模型
CUDA编程模型支持异步编程，即内核可以并发执行，而不必等待前一个内核执行完毕。异步编程模型可以提高程序的并发度，并减少延迟。异步编程模型的关键是将计算密集型任务分解为多个小任务，并将这些小任务分配给多个线程块，每个线程块执行一部分任务。每个线程块的执行结果可以被其他线程块使用，从而实现数据并行。在CUDA编程模型中，线程是进行计算或内存操作的最低抽象级别。从基于NVIDIA Ampere GPU架构的设备开始，CUDA编程模型通过异步编程模型提供对内存操作的加速。异步编程模型定义了异步操作相对于CUDA线程的行为。异步编程模型定义了用于CUDA线程之间同步的异步屏障的行为。该模型还解释并定义了在GPU中进行计算时，如何使用cuda:：memcpy_async从全局内存异步移动数据到设备内存，以及如何使用cuda：memcpy_dtoh从设备内存异步移动数据到全局内存。
## 5.5.1 Asynchronous Operations异步操作
异步操作被定义为由CUDA线程启动并像由另一个线程异步执行的操作。在格式良好的程序中，一个或多个CUDA线程与异步操作同步。启动异步操作的CUDA线程不需要在同步线程中。这样的异步线程（一个好像线程）总是与启动异步操作的CUDA线程相关联。异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理（例如，cuda:：memcpy_async），也可以在库内隐式管理（如，cooperative_groups:：memcpy_async）。同步对象可以是cuda:：barrier或cuda：：pipeline。在使用cuda:：pipeline的异步屏障和异步数据复制中详细解释了这些对象。这些同步对象可以在不同的线程作用域中使用。作用域定义了可以使用同步对象与异步操作同步的线程集。下表定义了CUDA C++中可用的线程作用域以及可以与每个作用域同步的线程。
Thread Scope    | Description
-------- | -----
cuda::thread_scope::thread_scope_thread  | Only the CUDA thread which initiated asynchronous operations synchronizes.
cuda::thread_scope::thread_scope_block| All or any CUDA threads within the same thread block as the initiating thread synchronizes.
cuda::thread_scope::thread_scope_device| All or any CUDA threads in the same GPU device as the initiating thread synchronizes.
cuda::thread_scope::thread_scope_system|All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.

# 5.6 Compute Capabilit计算能力
设备的计算能力由版本号表示，有时也称为它的“SM 版本”。该版本号标识 GPU 硬件支持的特征，并在运行时应用程序使用，以确定当前 GPU 上可用的哪些硬件特征和/或指令。计算能力包括主要修订数 X 和次要修订数 Y，用 X 表示。Y。具有相同主要修订数的设备具有相同的核心架构。基于 NVIDIA Hopper GPU 架构的设备的主要修订数为 9，基于 NVIDIA Ampere GPU 架构的设备为 8，基于 Volta 架构的设备为 7，基于 Pascal 架构的设备为 6，基于 Maxwell 架构的设备为 5，基于开普勒架构的设备为 3。较小的修订数对应于对核心架构的增量改进，可能包括新功能。图灵是计算能力为7.5的设备的架构，是基于Volta架构的增量更新。支持CUDA的gpu列表，所有支持CUDA的设备及其计算能力。计算能力提供了每个计算能力的技术规范。