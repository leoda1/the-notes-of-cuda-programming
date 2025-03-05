/******************************************************************
 * Author      : Da Liu
 * Date        : 2025-03-05
 * File Name   : Asnc_Op_exp4.cpp
 * Description : 调用async()函数直接创建线程执行任务
 *****************************************************************/
#include <iostream>
#include <thread>
#include <future>
using namespace std;

int main() {
    cout << "主线程ID: " << this_thread::get_id() << endl;
    // 调用函数直接创建线程执行任务
    future<int> f = async(launch::deferred, [](int x) {
        cout << "子线程ID: " << this_thread::get_id() << endl;
        this_thread::sleep_for(chrono::seconds(5));
        return x += 10;
    }, 100);
    
    future_status status;
    do {
        status = f.wait_for(chrono::seconds(2));
        if (status == future_status::deferred) {
            cout << "线程还没excute..." << endl;
            //指定了launch::deferred 策略调用async()函数并
            //不会创建新的线程执行任务，当使用future类对象
            //调用了get()或者wait()方法后才开始执行任务（此
            //处一定要注意调用wait_for()函数是不行的）
            f.wait(); 
        } else if (status == future_status::ready) {
            cout << "子线程返回值: " << f.get() << endl;
        } else if (status == future_status::timeout) {
            cout << "任务还没结束，等待.... " << endl;
        }
    } while (status != future_status::ready);
    return 0;
}
/******************************************************************
(使用launch::async的时候)
主线程ID: 140143669679936
子线程ID: 140143669675584
任务还没结束，等待.... 
任务还没结束，等待.... 
子线程返回值: 110
*******************************************************************/
/******************************************************************
(使用launch::deferred的时候)
主线程ID: 140634681972544
线程还没excute...
子线程ID: 140634681972544
子线程返回值: 110
*******************************************************************/