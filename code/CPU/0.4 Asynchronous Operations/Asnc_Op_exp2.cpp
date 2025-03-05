/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-05
 * File Name   : Asnc_Op_exp2.cpp
 * Description : 使用wait_for等待计算函数的线程完成任务
 *****************************************************************/
#include <iostream>
#include <thread>
#include <future>
#include <chrono>
using namespace std;


int main()
{
    promise<int> pr;
    thread t1([](promise<int> &p){
        this_thread::sleep_for(chrono::milliseconds(10));
        p.set_value(10);
        cout << "子线程结束了" <<endl;
    }, ref(pr));

    // future<int> ft = pr.get_future();
    auto ft = pr.get_future();
    
    //循环直到future的状态是就绪的，主线程可以乘机做其他的事情
    while(ft.wait_for(chrono::seconds(0)) != future_status::ready )
    {
        cout << "在异步等待中, 主线程在做其他的事情" << endl;
        this_thread::sleep_for(chrono::seconds(3));  
    }

    //future已经就绪了，可以获得取值
    int val = ft.get();
    cout << "val = " << val <<endl;
    t1.join();
    return 0;

}
/******************************************************************
在异步等待中, 主线程在做其他的事情
子线程结束了
val = 10
******************************************************************/