/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-06
 * File Name   : Atomic_Op_exp3.cpp
 * Description : 原子变量确定性操作使用 compare_exchange_strong
 *****************************************************************/
#include <iostream>
#include <thread>
#include <atomic>

using namespace std;

atomic<int> atomicInt(0);
void updateValue() {
    int expected = 0;
    int desired = 1;
    if (atomicInt.compare_exchange_strong(expected, desired)) {
        cout << "Value changed to " << desired << endl;
    } else {
        cout << "Expected " << expected << " but found " << atomicInt.load() << endl;
    }
}
int main() {
    thread t(updateValue);
    t.join();
    return 0;
}
/******************************************************************
Value changed to 1
*******************************************************************/