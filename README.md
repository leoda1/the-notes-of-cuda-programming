## 前言
愿望：希望自己能够将这个项目持续更新下去，从而能够帮助到更多的人。希望可以有更多人加入本项目分享和总结学习资料，一起维护，共同进步。

初衷：是为了记录自己学习CUDA C++编程的过程，希望能够帮助到大家。但是过程中发现学习路径不是很多，资料大多是英文为主，而我主要通过bilibili和官方文档学习。下方目录结构中的code主要包括跟着官方的samples和各类b站up主的project复现，cuda_programming_guide是官方的pdf文档的阅读笔记。

## project_directory
```
├── code
│   ├── CUDA
|   |    ├──1.0 Simple example
|   |    ├──1.1 Reduce
|   |    ├──1.2 Test_warp
|   |    ├──1.3 MergeSort
|   |    ├──1.4 Matmul_shared_memory
|   |    ├──1.5 Shared_conflict
|   |    ├──1.6 Stream
|   |    ├──1.7 Flashattn
|   |    ├──
│   ├── TensorRT
|   |    ├──2.1 Generate-onnx
|   |    ├──2.2 Export-onnx
|   |    ├──2.3 Read-parse
|   |    ├──2.4 Export-unsupport-node
|   |    ├──2.5 Onnxsurgeon
|   |    ├──2.6 Trtexec
|   |    ├──2.7 Mnist-sample
|   |    ├──2.8 Load_model
|   |    ├──2.9 Infer_model
|   |    ├──2.10 Custom-trt-plugin
├── .vscode
│   ├── settings.json 文档内的VSCode配置文件
├── cuda_programming notes 英伟达《CUDA C++ Programming Guide》官方文档的阅读笔记
│   ├── img 存放书中插图
│   ├── xxxxxx.md 存放书中每章的学习笔记
├── README.md 项目的readme文件

```
## How 2 install
```sh
git clone https://github.com/leoda1/the-notes-of-cuda-programming.git
```
## How 2 use
eg:
/code/CUDA/Simple example/GPU_Cache.cu 在terminal中
```sh
nvcc GPU_Cache.cu -o name
./name
```
eg:
/code/CUDA/Test_warp用的是cmake构建, 按照下面指令运行
```sh
cmake -B build -G #这是cmake的命令，这里可以找到你现在vs的版本
cmake -B build -G"Visual Studio 16 2019" #这是cmake的命令，具体看自己的系统
cmake --build build #编译代码
.\build\Debug\your_project.exe #运行Debug版本的可执行文件
```
这里写了blog记录了怎么用cmake运行, 具体点这里[blog](https://blog.csdn.net/buuliuda/article/details/139101651?spm=1001.2014.3001.5502)。

## 参考资料
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA CUDA Samples（官方在github上的项目，正在看）](https://github.com/NVIDIA/cuda-samples)
- [b站权双对cuda的视频讲解（我看完了）](https://www.bilibili.com/video/BV1sM4y1x7of?vd_source=85c9ce6d49ba579156fb1b41d0e606b3)
- [My CSDN（这里有些自己写的内容）](https://blog.csdn.net/buuliuda/category_12694574.html?spm=1001.2014.3001.5482)
- [b站爱学习的阿噜 (正在选择性看)](https://www.bilibili.com/video/BV1Sq4y1K7K6?vd_source=85c9ce6d49ba579156fb1b41d0e606b3)
