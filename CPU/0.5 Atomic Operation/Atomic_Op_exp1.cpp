/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-06
 * File Name   : Atomic_Op_exp1.cpp
 * Description : 测试两个线程对同一个数据执行任务
 *****************************************************************/
#include <iostream>
#include <thread>

using namespace std;

int main () {
    int sum = 0;
    auto f = [&sum] () {
        for (int i = 0; i < 12345678; i ++) {
            sum += 1;
        }
    };

    thread td1(f);
    thread td2(f);
    td1.join();
    td2.join();
    cout << "sum: " << sum << endl;

    return 0;
}
/******************************************************************
sum: 16333875    test 1
sum: 12906647    test 2
sum: 14225737    test 3
*******************************************************************/