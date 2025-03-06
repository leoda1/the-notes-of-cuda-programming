/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-06
 * File Name   : Atomic_Op_exp4.cpp
 * Description : compare_exchange_weak实现自旋锁,使得多个线程可以安
 *               全地累加 counter 变量
 *****************************************************************/
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
using namespace std;

int counter;
atomic<bool> lockFlag(false);

void spinlock_get() {
    bool expected = false;
    while (!lockFlag.compare_exchange_weak(expected, true)) {
        expected = false;
    }
}

void spinlock_free() {
    lockFlag.store(false);
}

void increment() {
    for (int i = 0; i < 123456; i ++) {
        spinlock_get();
        ++counter;
        spinlock_free();
    }
}

int main () {
    const int numThreads = 100;
    vector<thread> threads;
    for (int i = 0; i < numThreads; i ++) {
        threads.push_back(thread(increment));
    }

    for (auto& t : threads) {
        t.join();
    }
    cout << "Final counter value: " << counter << endl;
    return 0;
}
/******************************************************************
Final counter value: 12345600

//如果没有用自旋锁 输出的是：Final counter value: 7059450
*******************************************************************/