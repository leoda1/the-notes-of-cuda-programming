cmake_minimum_required(VERSION 3.18)
project(HelloCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 CUDA 架构，适用于 Ampere 架构 RTX 4060
set(CMAKE_CUDA_ARCHITECTURES 86)

# 设置策略以避免开发警告
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

add_executable(hellocuda chapter5.cu)
