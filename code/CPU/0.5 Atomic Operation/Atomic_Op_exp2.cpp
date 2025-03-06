/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-06
 * File Name   : Atomic_Op_exp1.cpp
 * Description : 使用atomic的公共成员函数范例
 *****************************************************************/
#include <iostream>
#include <thread>
#include <atomic>
using namespace std;

void test01()
{
    atomic_char cc('b');
    cc = 'd';
    cout << cc << endl;
    cc.store('a');
    cout << cc << endl;

    char ccc = cc.exchange('e');//返回之前的旧值
    cout << cc.load() << endl;
    cout << ccc << endl;
}


int main () {
    test01();
    return 0;
}
/******************************************************************
d
a
e
a
*******************************************************************/