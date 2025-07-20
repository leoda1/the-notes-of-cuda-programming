## Introduction
This repository documents the learning journey of a master's student entering the field of AI infrastructure. It serves as a personal knowledge base and practical reference covering essential components in high-performance computing for AI, including:
* CUDA programming for GPU parallelism
* TensorRT for model inference acceleration
* NCCL for multi-GPU communication
* MPI for distributed computing
* CPU multithreading and performance optimization
The notes and examples aim to bridge theoretical understanding and practical implementation in the AI infra stack.
(邮箱：20231083510920@buu.edu.cn, 电话:18621761124)
## Related note
I will update gpu-mode and cuda programming guide notes in here.
| 名称                  | 链接 |
|-----------------------|------|
| CUDA Programming Notes | [open](./cuda%20programming%20notes/) |
| GPU Mode               | [open](./cuda-mode/) |
## Project structure
```
├── code
│   ├── CPU
|   |    ├──0.0 Sync_queue 条件变量实现一个同步队列(线程安全的数据共享区)
|   |    ├──0.1 Lamba 表达式
|   |    ├──0.2 Smart Pointers(shared_ptr, weak_ptr, unique_ptr)
|   |    ├──0.3 Rvalue&Move&Perfect 右值引用 移动语义 完美转发实现性能优化
|   |    ├──0.4 Asynchronous Operation 异步操作
|   |    ├──0.5 Atomic Operation 原子变量，CAS操作和内存顺序等实现线程安全访问数据
│   ├── CUDA
|   |    ├──1.0 Simple example 官方api实现的各类功能
|   |    ├──1.1 Reduce 规约算子优化
|   |    ├──1.2 Test_warp 测试GPU预热
|   |    ├──1.3 MergeSort GPU的归并排序
|   |    ├──1.4 Matmul_shared_memory 使用共享内存对matix的tile做FMA
|   |    ├──1.5 Shared_conflict 解决1.4的bank conflict
|   |    ├──1.6 Stream 使用多个流
|   |    ├──1.7 FlashAttn 注意力领域前沿FlashAttn V1算子的复现
│   ├── TensorRT
|   |    ├──2.1 Generate-onnx 生成onnx
|   |    ├──2.2 Export-onnx 导出onnx
|   |    ├──2.3 Read-parse 解析onnx(用netron看计算图的张量和算子)
|   |    ├──2.4 Export-unsupport-node 怎么导出不支持的算子
|   |    ├──2.5 Onnxsurgeon 用surgeon来获取部分计算图
|   |    ├──2.6 Trtexec 用trtexec指令完成模型的量化部署
|   |    ├──2.7 Mnist-sample 官方的sample
|   |    ├──2.8 Load_model 加载模型
|   |    ├──2.9 Infer_model 推理模型
|   |    ├──2.10 Custom-trt-plugin 写tensorRT不支持的简称plugin
│   ├── xxxxxx.md 存放书中每章的学习笔记
│   ├── NCCL
|   |    ├──3.1 Simple example of allreduce
│   ├── MPI
|   |    ├──4.1 Peer to peer
|   |    ├──4.2 Collective
|   |    ├──4.3 Advanced collectived
│   ├── xxxxxx.md 存放书中每章的学习笔记
├── cuda_programming notes 英伟达《CUDA C++ Programming Guide》官方文档的阅读笔记
├── code-mode 这里是可以找到cuda-mode目前0-15章节的笔记以及reduce优化的实验
```

## How 2 configure your environment
### 自行配置的过程
这里在csdn写了blog记录了环境配置的过程，以及如何用cmake构建及运行上述代码, 文章链接
* [if you are windows + vscode + cuda + tensorRT，click here](https://blog.csdn.net/buuliuda/article/details/139101651?spm=1001.2014.3001.5502)。
* [if you are wsl / linux + vscode + cuda + tensorRT， click here](https://blog.csdn.net/buuliuda/article/details/144903013?spm=1001.2014.3001.5501)

### 不配置的方法
这里推荐用docker直接使用nvidia的ngc镜像:
* [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)

## How 2 install this repo
```sh
git clone https://github.com/leoda1/the-notes-of-cuda-programming.git
```
## How 2 use code
单个的cu文件用NVCC直接编译成可执行文件即可，例如运行:./code/CUDA/Simple example/GPU_Cache.cu 在terminal中输入
```sh
nvcc GPU_Cache.cu -o name
./name
```
如果是./code/CUDA/1.1-1.7节的代码，就要使用cmake来构建和编译整个项目了。例如
/code/CUDA/1.2 Test_warp用的是cmake构建, 按照下面指令运行
```sh
cmake -B build -G "Visual Studio 16 2019" #我是用的visual studio 2019来构建
cmake --build build # 编译代码
.\build\Debug\your_project.exe #运行Debug版本的可执行文件
```

## 参考资料
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
---