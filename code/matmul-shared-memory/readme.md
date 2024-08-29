[目录](#目录)
- [主函数](##主函数)
- [编译](##编译)
- [结果分析](##结果分析)

```json
//这是c_cpp_properties.json的代码，用于告诉我的vscode编译CUDA C++的代码的时候分别从workspaceFolder中的MSVC和CUDA安装目录下的NVCC。
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
            ],
            "defines": [],
            "compilerPath": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe",
            "intelliSenseMode": "windows-msvc-x64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
```
## 主函数
&emsp;&emsp;main.cpp为整个代码逻辑。先是导入了所有的头文件，库等等，定义了一个全局变量seed。这里的最核心的矩阵乘法的.cu代码我做了详细的视频解读，视频链接：[视频时长32：00，建议2x食用。](https://www.bilibili.com/video/BV1ejsgeuEt4/?spm_id_from=333.999.0.0&vd_source=290d4ccdca25416a14cdd4436003ce27)
```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#include "timer.hpp"
#include "utils.hpp"
#include "matmul.hpp"



int seed;
int main(){
    Timer timer;

    int width = 1<<12, low = 0, high = 1;
    int size = width * width;
    int block_size = 16;
    bool statMem = true;
    char str[100];

    float* h_matM = (float*)malloc(size * sizeof(float));
    float* h_matN = (float*)malloc(size * sizeof(float));
    float* h_matP = (float*)malloc(size * sizeof(float));
    float* d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM, size, low, high, seed);
    seed += 1;
    initMatrix(h_matN, size, low, high, seed);

    LOG("Input size is:%d x %d", width, width);
    /*GPU warmup*/
    timer.start_gpu();
    Matmul_device(h_matM, h_matN, h_matP, width, block_size);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu warmup");
    /*kernel function  <<<24, 32>>>*/
    timer.start_gpu();
    Matmul_device(h_matM, h_matN, d_matP, width, block_size);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu without shared memory)<<<%d, %d>>>",width / block_size, block_size);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);
    return 0;
}
/******************************************************************
(joker) PS C:\Users\22681\Desktop\project\cudalearn\notes\code\matmul-shared-memory> ./build/Debug/matmul.exe
Input size is:768 x 768
matmul in gpu warmup                                         uses 3.410112 ms
matmul in gpu without shared memory)<<<24, 32>>>             uses 3.022272 ms
(joker) PS C:\Users\22681\Desktop\project\cudalearn\notes\code\matmul-shared-memory> ./build/Debug/matmul.exe
Input size is:4096 x 4096
matmul in gpu warmup                                         uses 187.220901 ms
matmul in gpu without shared memory)<<<256, 16>>>            uses 184.976639 ms
 *****************************************************************/
```
## 编译
&emsp;&emsp;可以看到我的代码目录中有CMakeLists.txt文件，这里就是使用cmake构建整个项目。

&emsp;&emsp;代码指定了cmake版本，代码是CUDA C++，告诉了项目去哪里找CUDA的包，以及编译时的cpp,hpp,cu都在哪个目录下等等，cmake的语法不过多介绍。接下来看代码：
```md
cmake_minimum_required(VERSION 3.18)
project(matmul LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 14)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

find_package(CUDAToolkit REQUIRED)  # 查找 CUDA 库
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -arch=sm_86")   # CUDA Debug
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")                    # C++ Debug
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -arch=sm_86")       # CUDA Release
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")                    # C++ Release
endif()

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
set(INC "${PROJECT_SOURCE_DIR}/inc")

add_executable(matmul ${SRCS})
target_include_directories(matmul PRIVATE ${INC})
target_include_directories(matmul PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(matmul PRIVATE ${CUDA_LIBRARIES})
```
## 结果分析
&emsp;&emsp;接下来就是正式编译，接下来几步我的专栏中有说明，是如何使用vscode配置CUDA C++编译环境并运行的。这里第三次输入的是执行编译完成后的ereduction可执行文件，这个可执行文件无需输入参数。
```sh
cmake -B build -G "Visual Studio 16 2019"
cmake --build build
./build/Debug/matmul.exe
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06bc6147248a46e2a58c10e47a6e741a.png)