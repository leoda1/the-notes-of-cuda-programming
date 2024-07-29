cmake_minimum_required(VERSION 3.18)
project(constant_memory LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成编译命令
set(CMAKE_CUDA_STANDARD 14)            # 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)   # 设置 CUDA 标准要求
set(CMAKE_CXX_STANDARD 14)             # 设置 C++ 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # 设置 C++ 标准要求

find_package(CUDAToolkit REQUIRED)  # 查找 CUDA 库

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "CUDAToolkit not found")
endif()

add_executable(constant_memory constant_memory.cu common.cuh)
target_include_directories(constant_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}) 
target_link_libraries(constant_memory PRIVATE ${CUDA_LIBRARIES})