/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-05
 * File Name   : Asnc_Op_exp3.cpp
 * Description :（普通函数、lambda、仿函数、成员函数、静态成员函数）
 *               包装为异步任务，并通过 std::future 获取其返回值。
 *****************************************************************/
#include <iostream>
#include <future>
#include <thread>
#include <functional>

using namespace std;

string myFunc () {
    this_thread::sleep_for(chrono::seconds(2));
    return "im a cat";
}

using funcPtr = string (*)(string, int);
// 等价于
// typedef string(*funcPtr)(string, int);

class Base {
public:
    string operator()(string msg) {
        string str = "operator() function msg: " + msg;
        return str;
    }

    operator funcPtr() { // 允许 Base 对象转换为 funcPtr（函数指针）
        return showMsg;
    }

    int getNumber(int num) {
        int number = num + 10;
        return number;
    }

    static string showMsg (string msg, int num) {
        string str = "showMsg() function msg: " + msg + ", " + to_string(num);
        return str;
    } 
};


int main() {
    packaged_task<string(void)> task1(myFunc);   // 普通函数
    packaged_task<int(int)> task2([](int arg) {  // 匿名函数
        return 100;
    });
    Base b;
    packaged_task<string(string)> task3(b);      // 仿函数
    Base bb;
    packaged_task<string(string, int)> task4(bb);// 将类对象进行转换得到的函数指针
    packaged_task<string(string, int)> task5(&Base::showMsg);//静态函数
    Base bc;
    auto obj = bind(&Base::getNumber, &bc, placeholders::_1);
    // std::placeholders::_1 表示这个参数在调用时再传入
    packaged_task<int(int)> task6(obj);          // 非静态函数
    thread t1(ref(task6), 200);
    future<int> f = task6.get_future();
    int num = f.get();
    cout << "子线程返回值: " << num << endl;
    t1.join();

    return 0;
}
/******************************************************************
子线程返回值: 210
*******************************************************************/