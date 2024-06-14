# chapter6. programming interface
## 简介
英伟达 《CUDA C++ Programming Guide》 官方文档学习记录笔记 版本【Release 12.4】出版时间【 Apr 22, 2024】
## 目录
- [chapter6. programming interface](#chapter6-programming-interface)
  - [6.1 Compilation with NVCC](#61-compilation-with-nvcc)
      - [6.1.1 Compilation Workflow](#611-compilation-workflow)
          - [6.1.1.1 Offline Compilation](#6111-offline-compilation)
          - [6.1.1.2 Just-In-Time Compilation](#6112-just-in-time-compilation)
        - [6.1.2 Binary Compatibility](#612-binary-compatibility)
        - [6.1.3 PTX Compatability](#613-ptx-compatability)
        - [6.1.4 Applicvation Compatibility](#614-applicvation-compatibility)
        - [6.1.5 C++ Compatability](#615-c-compatability)
        - [6.1.6 64-Bit Comapatibility](#616-64-bit-compapatibility)
    - [6.2 CUDA Runtime](#62-cuda-runtime)    
        - [6.2.1 Initialization](#621-initialization)
        - [6.2.2 Device Memory](#622-device-memory)
        - [6.2.3 Device Memory L2 Access Management](#623-device-memory-l2-access-management)
          - [6.2.3.1 L2 Cache Set-Aside for Persisting Access](#6231-l2-cache-set-aside-for-persisting-access)
          - [6.2.3.2 L2 Policy for Persisting Access](#6232-l2-policy-for-persisting-access)
          - [6.2.3.3 L2 Access Properties](#6233-l2-access-properties)
          - [6.2.3.4 L2 Persistence Example](#6234-l2-persistence-example)
          - [6.2.3.5 Reset L2 Access to Normal](#6235-reset-l2-access-to-normal)
          - [6.2.3.6 Manage Utilization of L2 set-aside cache](#6236-manage-utilization-of-l2-set-aside-cache)
          - [6.2.3.7 Query L2 cache Properties](#6237-query-l2-cache-properties)
          - [6.2.3.8 Control L2 Cache Set-Aside Size for Persisting Memory Access](#6238-control-l2-cache-set-aside-size-for-persisting-memory-access)
        - [6.2.4 Shared Memory](#624-shared-memory)
        - [6.2.5 Distributed Shared Memory](#625-distributed-shared-memory)
        - [6.2.6 Page-Locked Host Memory](#626-page-locked-host-memory)
          - [6.2.6.1 Portable Memory](#6261-portable-memory)
          - [6.2.6.2 Write-Combining Memory](#6262-write-combining-memory)
          - [6.2.6.3 Mapped Memory](#6263-mapped-memory)
        - [6.2.7 Memory Synchronization Domains](#627-memory-synchronization-domains)
          - [6.2.7.1 Memory Fence Interference](#6271-memory-fence-interference)
          - [6.2.7.2 Isolating Traffic with Domains](#6272-isolating-traffic-with-domains)
          - [6.2.7.3 Using Domains in CUDA](#6273-using-domains-in-cuda)
        - [6.2.8 Asynchronous Concurrent Execution](#628-asynchronous-concurrent-execution)

## 6.1 Compilation with NVCC
### 6.1.1 Compilation Workflow
#### 6.1.1.1 Offline Compilation
#### 6.1.1.2 Just-In-Time Compilation
### 6.1.2 Binary Compatibility
### 6.1.3 PTX Compatability
### 6.1.4 Applicvation Compatibility
### 6.1.5 C++ Compatability
### 6.1.6 64-Bit Comapatibility
## 6.2 CUDA Runtime
### 6.2.1 Initialization
### 6.2.2 Device Memory
### 6.2.3 Device Memory L2 Access Management
#### 6.2.3.1 L2 Cache Set-Aside for Persisting Access
#### 6.2.3.2 L2 Policy for Persisting Access
#### 6.2.3.3 L2 Access Properties
#### 6.2.3.4 L2 Persistence Example
#### 6.2.3.5 Reset L2 Access to Normal
#### 6.2.3.6 Manage Utilization of L2 set-aside cache
#### 6.2.3.7 Query L2 cache Properties
#### 6.2.3.8 Control L2 Cache Set-Aside Size for Persisting Memory Access
### 6.2.4 Shared Memory
### 6.2.5 Distributed Shared Memory
### 6.2.6 Page-Locked Host Memory
#### 6.2.6.1 Portable Memory
#### 6.2.6.2 Write-Combining Memory
#### 6.2.6.3 Mapped Memory
### 6.2.7 Memory Synchronization Domains
#### 6.2.7.1 Memory Fence Interference
#### 6.2.7.2 Isolating Traffic with Domains
#### 6.2.7.3 Using Domains in CUDA
### 6.2.8 Asynchronous Concurrent Execution
