# Chapter 5. programming model

## 简介
英伟达 《CUDA C++ Programming Guide》 官方文档学习记录笔记 版本【Release 12.4】出版时间【 Apr 22, 2024】

## 目录
- [kernels](##kernels)
- [主要内容](#主要内容)
- [代码示例](#代码示例)
- [结论](#结论)
- [参考资料](#参考资料)

## kernels
CUDA C++通过定义C++函数拓展C++，称为内核（kernels）。调用该内核时，由N个不同的cuda线程并行执行N次。






### 子标题 1
详细描述子标题 1 的内容。

### 子标题 2
详细描述子标题 2 的内容。

## 代码示例
提供相关的代码示例。

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
