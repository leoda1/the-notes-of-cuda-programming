cmake_minimum_required(VERSION 3.18)
project(stream LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误
set(CMAKE_CXX_STANDARD 14)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Cannot find CUDA")
endif()


file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
set(INC "${PROJECT_SOURCE_DIR}/inc")

# 头文件包含路径
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${SRC_PATH}
    ${INC_PATH}
)

# 链接库路径
link_directories(
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
)

add_executable(stream ${SRCS})
# 链接库
target_link_libraries(stream PRIVATE
    ${CUDA_LIBRARIES}
    cudart
    cublas
    cudnn
    # stdc++fs
    ${OpenCV_LIBRARIES}
)

# 包含路径
target_include_directories(stream PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${SRC_PATH}
    ${INC_PATH}
)




