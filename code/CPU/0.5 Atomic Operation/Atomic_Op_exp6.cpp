/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-07
 * File Name   : Atomic_Op_exp6.cpp
 * Description : 内存顺序（注释掉的代码取消注释就可以实现正确的数据
 *               访问了，当前的代码会出现assert）
 *****************************************************************/
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

using namespace std;

int data_test = 0;
// atomic<bool> ready(false); 

void product() {
    data_test = 100;
    // ready.store(true); // Set flag
}

void consumer() {
    // while (!ready.load()) {
    //     this_thread::sleep_for(chrono::milliseconds(1));
    // }
    assert(data_test == 100);
}

int main() {
    thread t1(product);
    thread t2(consumer);
    t1.join();
    t2.join();
    return 0;
}

/******************************************************************
Atomic_Op_exp6: Atomic_Op_exp6.cpp:19: void consumer(): Assertion `
data_test == 100' failed.
Aborted (core dumped)
*******************************************************************/