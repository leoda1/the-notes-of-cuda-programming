/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-05
 * File Name   : Asnc_Op_exp1.cpp
 * Description : 使用std::promise 和std::future 进行线程间的数据传递
 *****************************************************************/
/***************************************************************************************************
通过`promise`传递数据的过程一共分为5步：
1. 在主线程中创建`std::promise`对象
2. 将这个`std::promise`对象通过引用的方式传递给子线程的任务函数
3. 在子线程任务函数中给`std::promise`对象赋值
4. 在主线程中通过`std::promise`对象取出绑定的`future`实例对象
5. 通过得到的`future`对象取出子线程任务函数中返回的值。
****************************************************************************************************/
#include <iostream>
#include <thread>
#include <future>
#include <string>
using namespace std;

void func (promise<string> &p)  {
    this_thread::sleep_for(chrono::seconds(2));
    p.set_value("im pandas");//设置promise的值，解除future.get()的阻塞。
    this_thread::sleep_for(chrono::seconds(1));
}

int main () {
    promise<string> pro; // 定义 promise
    // thread t1(func, ref(pro)); // // 创建线程 t1，并传递 promise 的引用
    thread t2([](promise<string>& p) {
        this_thread::sleep_for(chrono::seconds(2));
        p.set_value(" im cat");
        this_thread::sleep_for(chrono::seconds(1));
    }, ref(pro));
    future<string> f = pro.get_future();
    cout << "main thread:" << endl;
    string str = f.get();
    cout << "主线程阻塞结束" << endl;

    cout << "Return son thread data" << str << endl;
    t2.join();
    return 0;
}

/******************************************************************
main thread:
主线程阻塞结束
Return son thread data im cat
 *****************************************************************/