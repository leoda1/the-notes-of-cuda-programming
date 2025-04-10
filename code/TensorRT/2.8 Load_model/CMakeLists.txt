cmake_minimum_required(VERSION 3.18)
project(load_model LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 CUDA 库
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Cannot find CUDA")
endif()

# 设置路径
set(BUILD_PATH ${PROJECT_BINARY_DIR}/build)
set(SRC_PATH ${PROJECT_SOURCE_DIR}/src/cpp)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(OPENCV_INSTALL_DIR "E:/opencv")
set(TENSORRT_INSTALL_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.1.0.27")

# 设置源文件
file(GLOB_RECURSE CXX_SRC ${SRC_PATH}/*.cpp)
file(GLOB_RECURSE KERNELS_SRC ${SRC_PATH}/*.cu)

# 编译器选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -arch=sm_86 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -arch=sm_86 -w")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w")
endif()

# 头文件包含路径
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${SRC_PATH}
    ${INC_PATH}
    "${OPENCV_INSTALL_DIR}"
    "${TENSORRT_INSTALL_DIR}/include"
)

# 链接库路径
link_directories(
    "${TENSORRT_INSTALL_DIR}/lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)

# 添加可执行文件
add_executable(load_model ${CXX_SRC} ${KERNELS_SRC})

# 链接库
target_link_libraries(load_model PRIVATE
    ${CUDA_LIBRARIES}
    cudart
    cublas
    cudnn
    nvinfer_10
    nvonnxparser_10
    # stdc++fs
    ${OpenCV_LIBRARIES}
)

# 包含路径
target_include_directories(load_model PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${SRC_PATH}
    ${INC_PATH}
    "${TENSORRT_INSTALL_DIR}/samples/common"
    "${TENSORRT_INSTALL_DIR}/samples/utils"
)

# 编译选项
target_compile_options(load_model PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS}>
)