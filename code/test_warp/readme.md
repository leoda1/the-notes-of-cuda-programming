[目录](#目录)
- [配置文件](##配置文件)
- [主函数](##主函数)
- [编译](##编译)
- [性能分析](##性能分析)

## 配置文件
&emsp;&emsp;.vscode目录下，这里需要大家自行修改，这里有四个文件。
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
&emsp;&emsp;main.cpp为整个代码逻辑。先是导入了所有的头文件，库等等，定义了一个全局变量seed。执行最后的可执行文件的时候需要输入size和block_size。代码下方已经放上运行后的结果。现在请从main.cpp一行一行阅读：
```cpp
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include "utils.hpp" //错误检查函数的头文件
#include "timer.hpp" //记时函数
#include "reduce.hpp"//CPU上的分化函数
#include <cstring>
#include <memory>
#include <cmath>

int seed;
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ./build/reduction [size] [block_size]" << std::endl;//输出使用说明，告诉用户如何正确调用这个程序。
        return -1;
    }
    
    
    Timer timer;//timer类实例化
    char str[100];
    int size       = std::stoi(argv[1]);
    int block_size = std::stoi(argv[2]);
    int grid_size  = size / block_size;

    float* h_idata = nullptr;
    float* h_odata = nullptr;
    h_idata = (float*)malloc(size * sizeof(float));
    h_odata = (float*)malloc(grid_size * sizeof(float));
    
    seed = 1;
    initMatrix(h_idata, size, seed);
    memset(h_odata, 0, grid_size * sizeof(float));

    // CPU Reduction
    timer.start_cpu();
    float sumOnCPU = ReduceOnCPU(h_idata, size);
    timer.stop_cpu();
    std::sprintf(str, "reduce in cpu, results: %f", sumOnCPU);
    timer.duration_cpu<Timer::ms>(str);

    //GPU warmup
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, block_size);
    timer.stop_gpu();
    // timer.duration_gpu<Timer::ms>("reduce in gpu warmup");

    //GPU Reduction with Divergence
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, block_size);
    timer.stop_gpu();
    float sumOnGPUWithDivergence = 0;
    for (int i = 0; i < grid_size; i++)
    {
        sumOnGPUWithDivergence += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu with divergence, results: %f", sumOnGPUWithDivergence);
    timer.duration_gpu(str);

    //GPU Reduction without Divergence
    timer.start_gpu();
    ReduceOnGPUWithoutDivergence(h_idata, h_odata, size, block_size);
    timer.stop_gpu();
    float sumOnGPUWithoutDivergence = 0;
    for (int i = 0; i < grid_size; i++)
    {
        sumOnGPUWithoutDivergence += h_odata[i];
    }
    std::sprintf(str, "reduce in gpu without divergence, results: %f", sumOnGPUWithoutDivergence);
    timer.duration_gpu(str);

    free(h_idata);
    free(h_odata);
    return 0;
}

/******************************************************************
(joker) PS C:\Users\22681\Desktop\project\cudalearn\notes\code\test_warp> ./build/Debug/reduction.exe 33554432 512
reduce in cpu, results: 16777216.000000  uses 52.933100 ms
reduce in gpu with divergence, results: 16779780.000000      uses 19.392063 ms
reduce in gpu without divergence, results: 16779780.000000   uses 18.935841 ms
 *****************************************************************/
```
&emsp;&emsp;其中记时，错误检查函数（我的上一篇文章(主页点击CSDN)中有说过）的头文件分别在timer.hpp和utils.hpp，可以去查看代码看到里面声明的函数具体是什么。接下来就是如何编译这个main.cpp。
## 编译
&emsp;&emsp;可以看到我的代码目录中有CMakeLists.txt文件，这里就是使用cmake构建整个项目。.接下来看代码：
```cmake
cmake_minimum_required(VERSION 3.18)
# 项目信息
project(ReductionComputation LANGUAGES CXX CUDA)
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
# 添加可执行文件
add_executable(reduction ${SRCS})
target_include_directories(reduction PRIVATE ${INC})
target_include_directories(reduction PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_libraries(reduction PRIVATE ${CUDA_LIBRARIES})
```
&emsp;&emsp;代码指定了cmake版本，代码是CUDA C++，告诉了项目去哪里找CUDA的包，以及编译时的cpp,hpp,cu都在哪个目录下等等，cmake的语法不过多介绍。

&emsp;&emsp;接下来就是正式编译，接下来几步我的专栏(在我的CSDN)中有说明，是如何使用vscode配置CUDA C++编译环境并运行的。这里第三次输入的是执行编译完成后的ereduction可执行文件，这个可执行文件需要输入参数33554432和512。
```sh
cmake -B build -G "Visual Studio 16 2019"
cmake -build build
./build/Debug/reduction.exe 33554432 512
```
&emsp;&emsp;此时可以看到程序正确输出了结果。接下来我们使用nvidia提供的nsight system（当你下载了CUDA后你的windows开始菜单栏可以搜到这个附带的性能分析工具），这个工具替代了之前的nvprof。
![](https://i-blog.csdnimg.cn/direct/fc580c1587a6437fb07c172dc28b49e5.png)

## 性能分析
&emsp;&emsp;前面的步骤如果都没有错的话，现在进入到最后一步，对GPU执行main程序的性能进行更加详细的分析。找到编译后的可执行文件，在这个.exe文件的目录下使用管理员运行powershell,然后输入指令：
```sh
nsys profile --stats=true ./reduction.exe 33554432 512
```
&emsp;&emsp;nsys就是性能分析的nsight system，然后分析这个可执行程序，并带上这两个参数。稍等片刻会看到进度100%并返回.nsys-rep和.splite文件。

![](https://i-blog.csdnimg.cn/direct/fc44e9e614134deca39c81d1ec069170.png)

&emsp;&emsp;点击第一个nsys-rep会出现可视化界面便可以详细看到每个线程执行的时间。

![](https://i-blog.csdnimg.cn/direct/4a9a2ee935c740fcbd1936ddf0841295.png)
