# chapter6. programming interface 
## 简介
英伟达 《CUDA C++ Programming Guide》 官方文档学习记录笔记 版本【Release 12.4】出版时间【 Apr 22, 2024】
## 目录
- [chapter6. programming interface](#chapter6-programming-interface)
  - [简介](#简介)
  - [目录](#目录)
  - [6.1 Compilation with NVCC（NVCC 编译器）](#61-compilation-with-nvccnvcc-编译器)
    - [6.1.1 Compilation Workflow（编译工作流）](#611-compilation-workflow编译工作流)
      - [6.1.1.1 Offline Compilation （离线编译）](#6111-offline-compilation-离线编译)
      - [6.1.1.2 Just-In-Time Compilation （即时编译）](#6112-just-in-time-compilation-即时编译)
    - [6.1.2 Binary Compatibility （二进制兼容性）](#612-binary-compatibility-二进制兼容性)
    - [6.1.3 PTX Compatability （PTX 兼容性）](#613-ptx-compatability-ptx-兼容性)
    - [6.1.4 Applicvation Compatibility （应用兼容性）](#614-applicvation-compatibility-应用兼容性)
    - [6.1.5 C++ Compatability （C++ 兼容性）](#615-c-compatability-c-兼容性)
    - [6.1.6 64-Bit Comapatibility （64位兼容性）](#616-64-bit-comapatibility-64位兼容性)
  - [6.2 CUDA Runtime](#62-cuda-runtime)
    - [6.2.1 Initialization （初始化）](#621-initialization-初始化)
    - [6.2.2 Device Memory （设备内存）](#622-device-memory-设备内存)
    - [6.2.3 Device Memory L2 Access Management （设备内存L2访问管理）](#623-device-memory-l2-access-management-设备内存l2访问管理)
      - [6.2.3.1 L2 Cache Set-Aside for Persisting Access （L2缓存持久化访问）](#6231-l2-cache-set-aside-for-persisting-access-l2缓存持久化访问)
      - [6.2.3.2 L2 Policy for Persisting Access （L2持久化访问策略）](#6232-l2-policy-for-persisting-access-l2持久化访问策略)
      - [6.2.3.3 L2 Access Properties （L2访问属性）](#6233-l2-access-properties-l2访问属性)
      - [6.2.3.4 L2 Persistence Example （L2持久化示例）](#6234-l2-persistence-example-l2持久化示例)
      - [6.2.3.5 Reset L2 Access to Normal （重置L2访问到正常）](#6235-reset-l2-access-to-normal-重置l2访问到正常)
      - [6.2.3.6 Manage Utilization of L2 set-aside cache （管理L2缓存的利用率）](#6236-manage-utilization-of-l2-set-aside-cache-管理l2缓存的利用率)
      - [6.2.3.7 Query L2 cache Properties （查询L2缓存属性）](#6237-query-l2-cache-properties-查询l2缓存属性)
      - [6.2.3.8 Control L2 Cache Set-Aside Size for Persisting Memory Access （控制L2缓存持久化内存访问大小）](#6238-control-l2-cache-set-aside-size-for-persisting-memory-access-控制l2缓存持久化内存访问大小)
    - [6.2.4 Shared Memory （共享内存）](#624-shared-memory-共享内存)
    - [6.2.5 Distributed Shared Memory （分布式共享内存）](#625-distributed-shared-memory-分布式共享内存)
    - [6.2.6 Page-Locked Host Memory （页锁定主机内存）](#626-page-locked-host-memory-页锁定主机内存)
      - [6.2.6.1 Portable Memory （可移植内存）](#6261-portable-memory-可移植内存)
      - [6.2.6.2 Write-Combining Memory （写合并内存）](#6262-write-combining-memory-写合并内存)
      - [6.2.6.3 Mapped Memory （映射内存）](#6263-mapped-memory-映射内存)
    - [6.2.7 Memory Synchronization Domains （内存同步域）](#627-memory-synchronization-domains-内存同步域)
      - [6.2.7.1 Memory Fence Interference （内存栅栏干扰）](#6271-memory-fence-interference-内存栅栏干扰)
      - [6.2.7.2 Isolating Traffic with Domains （使用域隔离流量）](#6272-isolating-traffic-with-domains-使用域隔离流量)
      - [6.2.7.3 Using Domains in CUDA （在CUDA中使用域）](#6273-using-domains-in-cuda-在cuda中使用域)
    - [6.2.8 Asynchronous Concurrent Execution （异步并发执行）](#628-asynchronous-concurrent-execution-异步并发执行)

## 6.1 Compilation with NVCC（NVCC 编译器）
&emsp;&emsp;CUDA C++ 为熟悉 C++ 编程语言的用户提供了一个简单的路径，可以轻松编写设备执行的程序。它由 C++ 语言和运行时库的最小扩展集组成。核心语言扩展已在编程模型中引入。它们允许程序员将内核定义为 C++ 函数，并使用一些新的语法在每次调用函数时指定网格和块维度。所有扩展的完整描述可以在 C++ 语言扩展中找到。任何包含其中一些扩展的源文件都必须使用 nvcc 编译，如与 NVCC 编译中所述。运行时在CUDA运行时引入。它提供了在主机上执行的 C 和 C++ 函数来分配和释放设备内存、主机内存和设备内存之间的传输数据、具有多个设备的管理系统等。
&emsp;&emsp;这一节的NVCC编译流程图会在6.16节总结中。
### 6.1.1 Compilation Workflow（编译工作流）
#### 6.1.1.1 Offline Compilation （离线编译）
&emsp;&emsp;使用 nvcc 编译的源文件可以包括主机代码（即主机上执行的代码）和设备代码（即设备上执行的代码）的混合。nvcc的基本工作流程包括将设备代码与主机代码分离，然后:
▶将设备代码编译成汇编形式(PTX代码)和/或二进制形式(cubin对象)，
▶并通过替换<<<…>>>在内核中引入的语法，通过必要的CUDA运行时函数调用加载并从PTX代码或cubin对象启动每个编译的核。修改后的主机代码要么输出为 C++ 代码，该代码左使用另一个工具编译，要么通过让 NVcc 在最后一个编译阶段调用主机编译器直接编译为对象代码。然后应用程序可以：
▶链接到编译的主机代码（这是最常见的情况），
▶ 或忽略修改后的主机代码（如果有的话），并使用 CUDA 驱动程序 API加载并执行 PTX 代码或cubin对象。
#### 6.1.1.2 Just-In-Time Compilation （即时编译）
&emsp;&emsp;在运行时由应用程序加载的任何 PTX 代码都由设备驱动程序进一步编译为二进制代码。这称为即时编译。
### 6.1.2 Binary Compatibility （二进制兼容性）
&emsp;&emsp;二进制代码是特定于架构的。cubin对象是使用编译器选项生成的——代码指定目标架构：例如，使用 -code=sm_80 编译为计算能力 8.0 的设备生成二进制代码。二进制兼容性从对下一个稍作修改来保证，但不能从对前一个或重大修订的微小修改中保证。
### 6.1.3 PTX Compatability （PTX 兼容性）
&emsp;&emsp;一些 PTX 指令仅在计算能力较高的设备上支持。-archcompiler 选项指定将 C++ 编译为 PTX 代码时假设的计算能力。例如，-arch=compute_50仅在计算能力 5.0 和上述设备上支持。从为计算能力 6.0 (Pascal) 生成的 PTX 编译的计算能力 7.0 (Volta) 的二进制目标设备将不使用 Tensor Core 指令，因为这些在 Pascal 上不可用。因此，如果使用最新版本的 PTX 生成二进制，最终的二进制可能会的性能比可能差。编译为目标架构条件特征的 PTX 代码仅在完全相同的物理架构上运行，在其他架构上运行。Arch 条件 PTX 代码不是前向和后向兼容的。
### 6.1.4 Applicvation Compatibility （应用兼容性）
&emsp;&emsp;为了在特定计算能力的设备上执行代码，应用程序必须加载与二进制兼容性和 PTX 兼容性中描述的这种计算能力兼容的二进制或 PTXcode。特别是，为了能够在未来具有更高计算能力的架构上执行代码（尚未生成二进制代码），应用程序必须加载为这些设备编译的 PTX 代码（参见即时编译）。
&emsp;&emsp;注：Volta架构加入了独立的线程调度，所以需要使用编译器选项组合选择到 Pascal 的线程调度-arch=compute_60 -code=sm_70。
### 6.1.5 C++ Compatability （C++ 兼容性）
&emsp;&emsp;编译器前端处理CUDA文件依据C++语法规则。主机代码支持完整的C++代码，设备代码支持部分的C++代码。
### 6.1.6 64-Bit Comapatibility （64位兼容性）
&emsp;&emsp;64位模式编译的设备代码只支持以64位模式编译的主机代码。
**编译流程（附流程图）**
&emsp;&emsp;1.NVCC将所有代码分离为主机和设备code
&emsp;&emsp;2.设备code会编译为PTX代码，再将PTX编译为二进制cubin目标代码
&emsp;&emsp;3.编译为PTX时需要-arch = compute_xx选项，指定虚拟架构的计算能力，以便于代码可以使用cuda功能
&emsp;&emsp;4.PTX编译为cubin时，用-code = sm_xx选项指定目标架构，以便于代码可以运行在真实目标架构上，以便于确定可执行文件能够使用的GPU
<p align="center">
  <img src="img/fig8.png" alt="alt text" />
</p>
<p align="center">编译流程图</p>

## 6.2 CUDA Runtime
&emsp;&emsp;CUDA运行时提供了在主机上执行的 C 和 C++ 函数来分配和释放设备内存、主机内存和设备内存之间的传输数据、具有多个设备的管理系统等。CUDA运行时由一组函数和数据结构组成，这些函数和数据结构提供对设备内存、共享内存(share memory)、页锁定主机内存(page-locked host memory)、异步并发执行(asynchronous concurrent excution)、内存同步域、L2缓存管理、分布式共享内存等的访问。CUDA运行时还提供了一些其他功能，如错误处理(error checking、设备计时器、事件记录、流同步等。
### 6.2.1 Initialization （初始化）
&emsp;&emsp;调用cudaInitDevice()和cudaSetDevice()初始化与指定设备相关的runtime和context associated。如果没有调用，则默认使用device 0和自动初始化runtime API。
**CUDA 12.0 之前的初始化：**
在 CUDA 12.0 之前，调用 cudaSetDevice() 并不会初始化 CUDA 运行时。
因此，为了隔离运行时初始化和其他 API 活动（例如出于计时和错误处理的目的），应用程序通常会使用一个无操作的运行时调用，例如 cudaFree(0)。这个调用实际上什么都不做，但会触发 CUDA 运行时的初始化。
**CUDA 12.0 之后的初始化：**
在 CUDA 12.0 之后，调用 cudaSetDevice() 将会初始化 CUDA 运行时。这意味着不再需要通过 cudaFree(0) 来显式地初始化运行时。
&emsp;&emsp;CUDA 运行时会为系统中的每个设备创建一个 CUDA 上下文，这个上下文被称为该设备的主上下文。主上下文在第一次需要激活上下文的运行时函数调用时被初始化，并且在应用程序的所有主机线程之间共享。在创建上下文的过程中，如果需要，设备代码会进行即时编译，并加载到设备内存中，这一过程是透明的。如果需要，例如为了驱动 API 的互操作性，可以从驱动 API 访问设备的主上下文。当主机线程调用 cudaDeviceReset() 时，会销毁该主机线程当前操作的设备（即当前设备）的主上下文。下一个由任何主机线程针对该设备（作为当前设备）发出的运行时函数调用将创建一个新的主上下文。此外，CUDA 接口使用的全局状态在主机程序初始化时被创建，并在主机程序终止时被销毁。如果在程序初始化或终止过程中（例如在 main 函数之前或之后）使用这些接口，会导致未定义的行为。
### 6.2.2 Device Memory （设备内存）
### 6.2.3 Device Memory L2 Access Management （设备内存L2访问管理）
#### 6.2.3.1 L2 Cache Set-Aside for Persisting Access （L2缓存持久化访问）
#### 6.2.3.2 L2 Policy for Persisting Access （L2持久化访问策略）
#### 6.2.3.3 L2 Access Properties （L2访问属性）
#### 6.2.3.4 L2 Persistence Example （L2持久化示例）
#### 6.2.3.5 Reset L2 Access to Normal （重置L2访问到正常）
#### 6.2.3.6 Manage Utilization of L2 set-aside cache （管理L2缓存的利用率）
#### 6.2.3.7 Query L2 cache Properties （查询L2缓存属性）
#### 6.2.3.8 Control L2 Cache Set-Aside Size for Persisting Memory Access （控制L2缓存持久化内存访问大小）
### 6.2.4 Shared Memory （共享内存）
### 6.2.5 Distributed Shared Memory （分布式共享内存）
### 6.2.6 Page-Locked Host Memory （页锁定主机内存）
#### 6.2.6.1 Portable Memory （可移植内存）
#### 6.2.6.2 Write-Combining Memory （写合并内存）
#### 6.2.6.3 Mapped Memory （映射内存）
### 6.2.7 Memory Synchronization Domains （内存同步域）
#### 6.2.7.1 Memory Fence Interference （内存栅栏干扰）
#### 6.2.7.2 Isolating Traffic with Domains （使用域隔离流量）
#### 6.2.7.3 Using Domains in CUDA （在CUDA中使用域）
### 6.2.8 Asynchronous Concurrent Execution （异步并发执行）
