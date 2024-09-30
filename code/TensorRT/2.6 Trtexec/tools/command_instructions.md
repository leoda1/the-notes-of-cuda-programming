# TensorRT 构建引擎命令行工具使用说明

TensorRT 提供了多种命令行选项，允许用户在构建和运行推理引擎时进行详细的配置。以下是可用选项的详细说明：

## 基本选项

- **`--onnx=<model>`**
  - 指定输入的 ONNX 模型文件。
  
- **`--saveEngine=<file>`**
  - 指定生成的引擎文件保存路径。

## 形状控制选项

- **`--minShapes=<shapes>`**
- **`--optShapes=<shapes>`**
- **`--maxShapes=<shapes>`**
  - 为构建引擎指定输入形状的范围。这些参数在输入模型为 ONNX 格式时需要指定，以控制包括批大小在内的输入形状范围。

## 性能和内存优化选项

- **`–-memPoolSize=<pool_spec>`**
  - 指定运算策略可以使用的最大工作区大小，以及 DLA 将为每个可加载部分分配的内存池大小。支持的池类型包括 workspace, dlaSRAM, dlaLocalDRAM, dlaGlobalDRAM, 和 tacticSharedMem。一般来说1024 or 2048，2GB左右。

## 精度控制选项

- **`--fp16, --bf16, --int8, --fp8, --noTF32, --best`**
  - 指定网络级别的精度。在深度学习模型中，权重和激活通常以浮点数（如 FP32）表示。将这些值转换为 INT8 格式，意味着你需要对其进行量化，这个过程涉及到将连续的浮点值映射到有限的整数集。

- **`--timingCacheFile=<file>`**
  - 指定要加载的文件并保存时序缓存。此缓存包含 TensorRT 在引擎优化过程中探索的不同层配置的性能数据。通过使用时序缓存，TensorRT 可以在后续运行中跳过对层性能的重新评估，从而显著加快类似网络配置或使用类似硬件设置的引擎重建过程。


## 稀疏性选项

- **`--sparsity=[disable|enable|force]`**
  - 指定是否使用支持结构稀疏性的策略。
    - `disable`: 禁用所有使用结构稀疏性的策略。
    - `enable`: 启用使用结构稀疏性的策略，仅当 ONNX 文件中的权重符合结构稀疏性要求时使用。
    - `force`: 强制启用使用结构稀疏性的策略，并允许 trtexec 重写 ONNX 文件中的权重，以强制它们具有结构稀疏性模式。

## 编译和运行时选项

- **`--noCompilationCache`**
  - 禁用构建器中的编译缓存，该缓存是计时缓存的一部分。

- **`--verbose`**
  - 开启详细日志记录。

- **`--dumpLayerInfo, --exportLayerInfo=<file>`**
  - 打印或保存引擎中每一层的详细信息。使用 `--dumpLayerInfo` 在标准输出中打印信息，使用 `--exportLayerInfo=<file>` 保存信息到指定文件。

- **`--precisionConstraints=spec`**
  - 控制 TensorRT 引擎中的精度约束设置。
    - `none`: 无约束。
    - `prefer`: 尽可能满足由 `--layerPrecisions` 或 `--layerOutputTypes` 设置的精度约束。
    - `obey`: 必须满足由 `--layerPrecisions` 或 `--layerOutputTypes` 设置的精度约束，否则失败。

## 推理和性能测试

- **`--skipInference`**
  - 构建并保存引擎，但不运行推理。

- **`--profilingVerbosity=[layer_names_only|detailed|none]`**
  - 指定构建引擎时的分析详细程度。

## 层和设备类型控制

- **`--layerPrecisions=spec`**
此选项允许您对每个网络层指定精度约束。这些约束仅在设置了 `precisionConstraints` 为 `obey` 或 `prefer` 时生效。规格从左到右读取，后续规格将覆盖先前的设置。
    - **示例**:
  - `--layerPrecisions=*:fp16,layer_1:fp32`
  - 此设置将所有未指定的层的精度设置为 FP16，而将 `layer_1` 的精度设置为 FP32。

- **`--layerOutputTypes=spec`**
此选项允许您为每层的输出指定数据类型。在具有多个输出的层中，如果层具有多个输出，则可以通过 `+` 分隔符为该层提供多种类型。
    - **示例**:
  - `--layerOutputTypes=*:fp16,layer_1:fp32+fp16`
  - 此设置将所有层输出的精度设置为 FP16，除了 `layer_1`，其第一个输出将设置为 FP32，第二个输出将设置为 FP16。

- **`--layerDeviceTypes=spec`**
此选项允许您显式设置每层的设备类型为 GPU 或 DLA。与其他规格一样，这些设置从左到右读取，并且后来的设置将覆盖先前的设置。
    - **示例**:
  - `--layerDeviceTypes=layer_1:DLA,layer_2:GPU`
  - 此设置将 `layer_1` 指定为在 DLA 上运行，而 `layer_2` 在 GPU 上运行。


## 其他高级选项

- **`--dynamicPlugins=<file>`**
- **`--setPluginsToSerialize=<file>`**
  - 加载动态插件库，并在指定的情况下将插件序列化与引擎一起。

- **`--builderOptimizationLevel=N`**
  - 设置构建器优化级别，允许 TensorRT 在构建引擎时进行更多的优化选项。

- **`--maxAuxStreams=N`**
  - 设置每个推理流可使用的最大辅助流数量。如果网络中包含可以并行运行的操作，TensorRT可以使用这些流来并行运行内核，但代价是更高的内存使用。将此值设置为0以优化内存使用。有关更多信息，请参考“Within-Inference Multi-Streaming”部分。

- **`--stripWeights`**
  - 从计划中剥离权重。此标志可以与重装或具有相同权重的重装一起使用。默认设置为具有相同权重的重装；然而，你可以通过同时启用stripWeights和refit来切换到重装。

- **`--markDebug`**
  - 指定要标记为调试张量的张量名称列表。使用逗号分隔各个名称。

- **`--allowWeightStreaming`**
  - 启用可以流式传输其权重的引擎。必须与 --stronglyTyped 一起指定。TensorRT 将在运行时自动选择适当的权重流式传输预算以确保模型执行。可以通过 --weightStreamingBudget 设置特定的金额。

- **`--useDLACore=N`**
  - 使用指定的 DLA 核心执行支持 DLA 的层。

- **`--allowGPUFallback`**
  - 允许不支持 DLA 的层在 GPU 上运行。

- **`--versionCompatible, --vc`**
  - 启用版本兼容模式用于引擎构建和推理。任何在此模式下构建的引擎，都与运行在同一宿主操作系统上的较新版本的 TensorRT 兼容，前提是使用 TensorRT 的调度和精简运行时。只支持显式批处理模式。

- **`--excludeLeanRuntime`**
  - 当启用 `--versionCompatible` 时，此标志表示生成的引擎不应包含嵌入的精简运行时。如果设置此项，你必须在加载引擎时显式指定一个有效的精简运行时。仅支持显式批量和引擎内权重。

- **`--tempdir=<dir>`**
  - 覆盖 TensorRT 创建临时文件时使用的默认临时目录。更多信息请参考 IRuntime::setTemporaryDirectory API 文档。

- **`--tempfileControls=controls`**
  - 控制 TensorRT 在创建临时可执行文件时可以使用的设置。它应为逗号分隔的列表，条目格式为 [in_memory|temporary]:[allow|deny]。
    - `in_memory`: 控制是否允许 TensorRT 创建内存中的临时可执行文件。
    - `temporary`: 控制是否允许 TensorRT 在文件系统中（由 `--tempdir` 指定的目录）创建临时可执行文件。
    - 示例用法：`--tempfileControls=in_memory:allow,temporary:deny`

