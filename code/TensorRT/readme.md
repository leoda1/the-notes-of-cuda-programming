### **模型部署的基础知识：**
#### 1. 指标
#### FLOPS(Floating point number operations per second)：
- 一秒钟可以处理的浮点数运算的次数
- 衡量计算机硬件性能与计算能力的单位
![FLOPS](img/image.png)

#### TOPS(Tera operations per second)：
-  指的是一秒钟可以处理的整型运算的次数
-  衡量计算机硬件性能、计算能力的一个单位
![TOPS](img/image2.png)

#### 2. 模型部署误区
因为FLOPs只是模型计算大小的单位,还需要考虑
- 访存量
- reshape, shortcut, nchw2nhwc等等
- 前处理、后处理这些耗时操作

存在TensorRT尚未支持的算子,需要自己写自己写plugin

使用Tensor Core需要让tensor size为8或者16的倍数,不然会有性能损失
- 8的倍数： fp16
- 16的倍数：int8

对使用TensorRT得到的推理引擎做benchmark和profiling
- 使用TensorRT得到推理引擎并实现infer只是优化的第一步
- 要使用NVIDIA提供的benchmark tools进行profiling（分析模型瓶颈在哪里，分析模型可进一步优化的地方在哪里， 分析模型中多余的memory access在哪里）
- 工具：nsys, nvprof, dlprof, Nsight

#### 3. 量化
![alt text](img/image3.png)
![alt text](img/image4.png)
![alt text](img/image5.png)
![alt text](img/image6.png)