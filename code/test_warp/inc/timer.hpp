#include <chrono>
#include <ratio>
#include <string>
#include "cuda_runtime.h"

class Timer {
public:
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer();    // 构造函数
    ~Timer();   // 析构函数

public:
    void start_cpu();
    void stop_cpu();
    void start_gpu();
    void stop_gpu();

    template<typename span>
    void duration_cpu(std::string msg); // 用于在CPU上测量时间间隔的函数，接受一个字符串消息作为参数
    void duration_gpu(std::string msg); // 用于在GPU上测量时间间隔的函数，接受一个字符串消息作为参数

// 定义一个计时器类，用于记录和计算时间
class Timer {
private:
    // 记录CPU开始时间的时间点
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
    // 记录CPU结束时间的时间点
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
    // CUDA事件，用于记录GPU开始时间
    cudaEvent_t _gStart;
    // CUDA事件，用于记录GPU结束时间
    cudaEvent_t _gStop;
    // 记录的时间间隔，单位为秒
    float _timeElasped;
};

template<typename span>
void Timer::duration_cpu(std::string msg) {
    std::string str;

    if (std::is_same<span, s>::value) {str = "s";}
    else if (std::is_same<span, ms>::value) {str = "ms";}
    else if (std::is_same<span, us>::value) {str = "us";}
    else if (std::is_same<span, ns>::value) {str = "ns";}

    std::chrono::duration<double, span> time = _cStop - _cStart;
    LOG("%-40s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
 
}