cmake_minimum_required(VERSION 3.18)
project(mnist_sample LANGUAGES CXX CUDA)
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

add_executable(mnist_sample ${SRCS})     # 添加可执行文件
target_include_directories(mnist_sample PRIVATE ${INC})
target_include_directories(mnist_sample PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(mnist_sample PRIVATE ${CUDA_LIBRARIES})