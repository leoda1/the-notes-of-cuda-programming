@[TOC]
##  多线程|互斥锁|条件变量

> 参考资料：[https://deployment.gitbook.io/love/whitepaper/cpp/thread#hu-chi-liang](https://deployment.gitbook.io/love/whitepaper/cpp/thread#hu-chi-liang)

### 1.1 线程thread（std::thread）

在头文件中，`#include <thread>`声明。基本语法包括**构造函数**和**主要成员函数**。

**构造函数分为：**

- 默认构造函数：
    
    创建一个不表示任何线程的对象。该对象不与任何线程关联。
    
    ```cpp
    std::thread t;  // 创建一个不关联任何线程的线程对象
    ```
    
- 初始化构造函数:
    
    可以接收一个可调用对象（例如函数、函数指针、lambda、成员函数等）作为参数，并启动一个线程执行这个可调用对象。
    
    ```cpp
    std::thread t(Callable&& f, Args&&... args);
    //std::thread t(foo, 42, "Hello");  // 创建一个线程，执行 foo(42, "Hello")
    ```
    
- 拷贝构造函数:
    
    不能通过拷贝构造函数创建线程对象。
    
    ```cpp
    // 拷贝构造函数（被禁用），意味着 thread 不可被拷贝构造。
    thread(const thread&) = delete;
    ```
    
- Move构造函数:
    
    允许将一个线程对象的所有权从一个线程对象转移到另一个对象。通过移动构造函数，线程的资源可以被转移，而不是复制，这使得线程对象能够通过移动语义来有效地传递。
    
    ```cpp
    std::thread t1(foo, 42, "Hello");
    std::thread t2 = std::move(t1);  // 将 t1 的所有权转移给 t2
    ```
    

**主要成员函数分为：**

- get_id():
用于获取线程的唯一标识符。它返回一个 `std::thread::id` 对象，可以用来标识线程。
    
    ```cpp
    std::cout << "Thread ID: " << t.get_id() << std::endl;
    ```
    
- joinable():
    
    检查线程对象是否关联了一个可加入的线程。
    
- join():
    
    等待线程完成执行。调用 `join()` 后，线程对象不再关联任何线程。
    
- detach():
    
    将线程与线程对象分离 ，使线程在后台独立运行。分离后，线程对象不再关联任何线程。
    
    ```cpp
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "Thread finished." << std::endl;
    });
    
    std::cout << "Detaching thread..." << std::endl;
    t.detach();
    std::cout << "Thread detached." << std::endl;
    
    // 主线程继续执行
    std::this_thread::sleep_for(std::chrono::seconds(3));
    ```
    

**线程创建sample：**

```cpp
#include <iostream>
#include <thread>
using namespace std;

void func1()
{
    cout << "func1 into" << endl;
}

void func2(int a, int b)
{
    cout << "func2 a + b = " << a+b << endl;
}

class A
{
public:
    static void fun3(int a)
    {
        cout << "a = " << a << endl;
    }
    
    void showMsg(string name, int age)
    {
        cout << "name: " << name << ", age: " << age << endl;
    }
};

int main()
{
    thread t1(func1); // 只传递函数
    t1.join(); // 阻塞等待线程函数执行结束

    int a = 10;
    int b = 20;

    thread t2(func2, a, b); // 加上参数传递
    t2.join();

    thread t3(&A::fun3, 1); // 绑定类静态函数
    t3.join();

    A ao;
    thread t4(&A::showMsg, ao, "Mike1", 19); // 绑定类的成员函数
    t4.join();
    return 0;
}

```

**线程封装sample:**

```cpp
// zero_thread.h
#ifndef ZERO_THREAD_H
#define ZERO_THREAD_H
#include <thread>

class ZERO_Thread
{
public:
    ZERO_Thread(); // 构造函数
    virtual ~ZERO_Thread(); // 析构函数
    bool start();
    void stop();
    bool isAlive() const; // 线程是否存活.
    std::thread::id id() { return _th->get_id(); }
    std::thread* getThread() { return _th; }
    void join(); // 等待当前线程结束, 不能在当前线程上调用
    void detach(); //能在当前线程上调用
    static size_t CURRENT_THREADID();
protected:
    static void threadEntry(ZERO_Thread *pThread); // 静态函数, 线程入口
    virtual void run() = 0; // 运行
protected:
    bool _running; //是否在运行
    std::thread *_th;
};
#endif // ZERO_THREAD_H

// zero_thread.cpp
#include "zero_thread.h"
#include <sstream>
#include <iostream>
#include <exception>

ZERO_Thread::ZERO_Thread():_running(false), _th(NULL)
{
}
ZERO_Thread::~ZERO_Thread()
{
    if(_th != NULL)
    {
        //如果资源没有被detach或者被join，则自己释放
        if (_th->joinable())
        {
            _th->detach();
        }
        delete _th;
        _th = NULL;
    }
    std::cout << "~ZERO_Thread()" << std::endl;
}
bool ZERO_Thread::start()
{
    if (_running)
    {
        return false;
    }
    try
    {
        _th = new std::thread(&ZERO_Thread::threadEntry, this);
    }
    catch(...)
    {
        throw "[ZERO_Thread::start] thread start error";
    }
    return true;
}
void ZERO_Thread::stop()
{
    _running = false;
}
bool ZERO_Thread::isAlive() const
{
    return _running;
}
void ZERO_Thread::join()
{
    if (_th->joinable())
    {
        _th->join();
    }
}
void ZERO_Thread::detach()
{
    _th->detach();
}
size_t ZERO_Thread::CURRENT_THREADID()
{
    // 声明为thread_local的本地变量在线程中是持续存在的，不同于普通临时变量的生命周期，
    // 它具有static变量一样的初始化特征和生命周期，即使它不被声明为static。
    static thread_local size_t threadId = 0;
    if(threadId == 0 )
    {
        std::stringstream ss;
        ss << std::this_thread::get_id();
        threadId = strtol(ss.str().c_str(), NULL, 0);
    }
    return threadId;
}
void ZERO_Thread::threadEntry(ZERO_Thread *pThread)
{
    pThread->_running = true;
    try
    {
        pThread->run(); // 函数运行所在
    }
    catch (std::exception &ex)
    {
        pThread->_running = false;
        throw ex;
    }
    catch (...)
    {
        pThread->_running = false;
        throw;
    }
    pThread->_running = false;
}

// main.cpp
#include <iostream>
#include <chrono>
#include "zero_thread.h"

using namespace std;
class A: public ZERO_Thread
{
public:
    void run()
    {
        while (_running)
        {
            cout << "print A " << endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        cout << "----- leave A " << endl;
    }
};

class B: public ZERO_Thread
{
public:
    void run()
    {
        while (_running)
        {
            cout << "print B " << endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        cout << "----- leave B " << endl;
    }
};
int main()
{
    {
        A a;
        a.start();
        B b;
        b.start();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        a.stop();
        a.join();
        b.stop();
        b.join();
    }
    cout << "Hello World!" << endl;
    return 0;
}

```

`thread`线程类还提供了一个静态方法（`int num = thread::hardware_concurrency();`），用于获取当前计算机的CPU核心数，根据这个结果在程序中创建出数量相等的线程，每个线程独自占有一个CPU核心，这些线程就不用分时复用CPU时间片，此时程序的并发效率是最高的。

### 1.2 命名空间（this_thread）

C++11中不仅添加了线程类还有一个关于线程的命名空间。`std::this_thread`，它提供了四个公共成员函数，通过这些成员函数就可以对当前线程进行相关操作了。

- **get_id()：**
    
    程序启动，开始执行`main()`函数，此时只有一个线程也就是主线程。当创建了子线程对象`t`之后，指定的函数`func()`会在子线程中执行，这时通过调用`this_thread::get_id()`就可以得到当前线程的线程ID了。
    
    ```cpp
    #include <iostream>
    #include <thread>
    using namespace std;
    
    void func()
    {
        cout << "子线程: " << this_thread::get_id() << endl;
    }
    
    int main()
    {
        cout << "主线程: " << this_thread::get_id() << endl;
        thread t(func);
        t.join();
    }
    ```
    
- **sleep_for()：**
    
    线程被创建后有这五种状态：`创建态`，`就绪态`，`运行态`，`阻塞态(挂起态)`，`退出态(终止态)`。
    
    线程和进程的执行有很多相似之处，在计算机中启动的多个线程都需要占用CPU资源，但是CPU的个数是有限的并且每个CPU在同一时间点不能同时处理多个任务。`为了能够实现并发处理，多个线程都是分时复用CPU时间片，快速的交替处理各个线程中的任务。因此多个线程之间需要争抢CPU时间片，抢到了就执行，抢不到则无法执行`（因为默认所有的线程优先级都相同，内核也会从中调度，不会出现某个线程永远抢不到CPU时间片的情况）。
    
    命名空间`this_thread`中提供了一个休眠函数`sleep_for()`，调用这个函数的线程会马上从`运行态`变成`阻塞态`并在这种状态下休眠一定的时长，因为阻塞态的线程已经让出了CPU资源，代码也不会被执行，所以线程休眠过程中对CPU来说没有任何负担。
    
    ```cpp
    #include <iostream>
    #include <thread>
    #include <chrono>
    using namespace std;
    
    void func()
    {
        for (int i = 0; i < 10; ++i)
        {
            this_thread::sleep_for(chrono::seconds(1));
            cout << "子线程: " << this_thread::get_id() << ", i = " << i << endl;
        }
    }
    
    int main()
    {
        thread t(func);
        t.join();
    }
    ```
    
    在`func()`函数的`for`循环中使用了`this_thread::sleep_for(chrono::seconds(1));`之后，每循环一次程序都会阻塞1秒钟，也就是说每隔1秒才会进行一次输出。需要注意的是：程序休眠完成之后，会从阻塞态重新变成就绪态，就绪态的线程需要再次争抢CPU时间片，抢到之后才会变成运行态，这时候程序才会继续向下运行。
    
- **sleep_until():**
    
    指定线程阻塞到某一个指定的时间点`time_point`类型，之后解除阻塞。
    
    ```cpp
    // 获取当前系统时间点
            auto now = chrono::system_clock::now();
            // 时间间隔为2s
            chrono::seconds sec(2);
            // 当前时间点之后休眠两秒
            this_thread::sleep_until(now + sec);
    ```
    
- **yield():**
    
    在线程中调用这个函数之后，处于运行态的线程会主动让出自己已经抢到的CPU时间片，最终变为就绪态，这样其它的线程就有更大的概率能够抢到CPU时间片了。**使用这个函数的时候需要注意一点，线程调用了yield()之后会主动放弃CPU资源，但是这个变为就绪态的线程会马上参与到下一轮CPU的抢夺战中，不排除它能继续抢到CPU时间片的情况，这是概率问题。**
    
    ```cpp
    #include <iostream>
    #include <thread>
    using namespace std;
    
    void func()
    {
        for (int i = 0; i < 100000000000; ++i)
        {
            cout << "子线程: " << this_thread::get_id() << ", i = " << i << endl;
            this_thread::yield();
        }
    }
    
    int main()
    {
        thread t(func);
        thread t1(func);
        t.join();
        t1.join();
    }
    ```
    
    在上面的程序中，执行`func()`中的`for`循环会占用大量的时间，在极端情况下，如果当前线程占用CPU资源不释放就会导致其他线程中的任务无法被处理，或者该线程每次都能抢到CPU时间片，导致其他线程中的任务没有机会被执行。解决方案就是每执行一次循环，让该线程主动放弃CPU资源，重新和其他线程再次抢夺CPU时间片，如果其他线程抢到了CPU时间片就可以执行相应的任务了。
    

### 1.3 互斥量（mutex）

**mutex**又称互斥量，用于确保在任何时刻只有一个线程能够访问共享资源。它可以防止多个线程同时访问共享数据，避免数据冲突。当一个线程需要访问某个共享资源时，它会先获得这个互斥量的锁。获得锁后，其他线程就不能访问这个资源，直到当前线程释放锁。互斥量的作用是保证线程间的互斥访问，避免并发访问引发的问题。

C++ 11中与 mutex相关的类（包括锁类型）和函数都声明在 头文件中，所以如果 你需要使用 std::mutex，就必须包含`#include<mutex>`头文件。**std::mutex 是C++11 中最基本的互斥量，std::mutex 对象提供了独占所有权的特性——即不支持递归地 对 std::mutex 对象上锁，而 std::recursive_lock 则可以递归地对互斥量对象上锁。**

包括4种语义的mutex：

- std::mutex，独占的互斥量，不能递归使用。
    - 不允许拷贝构造，也不允许 move 拷贝，最初产生的 mutex 对象是处于 unlocked 状态的。
    - 成员函数`lock()`，如果该互斥量当前没 有被锁住，则调用线程将该互斥量锁住，直到调用 unlock之前。如果互斥量已经被其他线程锁定，则当前的调用线程被阻塞住。如果当前互斥量被当前调用线程锁 住，则会产生死锁(deadlock)。
    - 成员函数`unlock()`，释放已获得的互斥锁。
    - 成员函数`try_lock()`，尝试锁住互斥量，如果互斥量被其他线程占有，则当前线程也不会被阻塞。成功获取锁时返回 `true`，如果锁不可用，则返回 `false`。如果当前互斥量被当前调用线程锁住，则会产生死锁(deadlock)。
    
    ```cpp
    #include <iostream> // std::cout
    #include <thread> // std::thread
    #include <mutex> // std::mutex
    
    volatile int counter(0); // non-atomic counter
    std::mutex mtx; // locks access to counter
    void increases_10k()
    {
        for (int i=0; i<10000; ++i) {
            // 1. 使用try_lock的情况
            // if (mtx.try_lock()) { // only increase if currently not locked:
            // ++counter;
            // mtx.unlock();
            // }
            // 2. 使用lock的情况
            {
                mtx.lock();
                ++counter;
                mtx.unlock();
            }
        }
    }
    
    int main()
    {
        std::thread threads[10];
        for (int i=0; i<10; ++i)
            threads[i] = std::thread(increases_10k);
        for (auto& th : threads) th.join();
        std::cout << " successful increases of the counter " << counter << std::endl;
        return 0;
    }
    ```
    
- std::recursive_mutex，递归互斥量，不带超时功能。
    
    递归锁允许同一个线程多次获取该互斥锁，可以用来解决同一线程需要多次获取互斥量时死锁的问题。
    
- std::time_mutex，带超时的独占互斥量，不能递归使用。(比std::mutex多了两个超时获取锁的接口：try_lock_for和try_lock_until)
- std::recursive_timed_mutex，带超时的递归互斥量。

**lock_guard和unique_lock的使用和区别**

相对于手动lock和unlock，我们可以使用RAII(通过类的构造析构)来实现更好的编码方式。 

```cpp
#include <iostream> // std::cout
#include <thread> // std::thread
#include <mutex> // std::mutex, std::lock_guard
#include <stdexcept> // std::logic_error

std::mutex mtx;
void print_even (int x) {
    if (x%2==0) std::cout << x << " is even\n";
    else throw (std::logic_error("not even"));
}

void print_thread_id (int id) {
    try {
        // using a local lock_guard to lock mtx guarantees unlocking on destruction / exception:
        std::lock_guard<std::mutex> lck (mtx);
        print_even(id);
    }
    catch (std::logic_error&) {
        std::cout << "[exception caught]\n";
    }
}

int main ()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
        threads[i] = std::thread(print_thread_id,i+1);
    for (auto& th : threads) th.join();
    return 0;
}

```

- **unique_lock与lock_guard都能实现自动加锁和解锁，但是前者更加灵活，能实现更多的功能。**
- **unique_lock可以进行临时解锁和再上锁，如在构造对象之后使用lck.unlock()就可以进行解锁， lck.lock()进行上锁，而不必等到析构时自动解锁。**

```cpp
#include <iostream>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>

std::deque<int> q;
std::mutex mu;
std::condition_variable cond;

void fun1() {
    while (true) {
        std::unique_lock<std::mutex> locker(mu);
        q.push_front(count);
        locker.unlock();
        cond.notify_one();
        sleep(10);
    }
}

void fun2() {
    while (true) {
        std::unique_lock<std::mutex> locker(mu);
        cond.wait(locker, [](){return !q.empty();});
        data = q.back();
        q.pop_back();
        locker.unlock();
        std::cout << "thread2 get value form thread1: " << data << std::endl;
    }
}

int main() {
    std::thread t1(fun1);
    std::thread t2(fun2);
    t1.join();
    t2.join();
    return 0;
}
```

条件变量的目的就是，在没有获得某种提醒时长时间休眠；如果正常情况下, 我们需要一直循环 (`++sleep`), 这样的问题就是CPU消耗+时延问题，可以使用条件变量`cond.wait`这里一直休眠直到 `cond.notify_one`唤醒才开始执行下一句; 还有`cond.notify_all`接口用于唤醒所有等待的线程。

这里必须使用unique_lock的原因是：

> **条件变量在wait时会进行unlock再进入休眠, lock_guard并无该操作的接口。**
> 
- wait: 如果线程被唤醒或者超时那么会先进行lock获取锁, 再判断条件(传入的参数)是否成立, 如果成立则 wait函数返回否则释放锁继续休眠
- notify: 进行notify动作并不需要获取锁

**总结：**

**lock_guard**

1. std::lock_guard 在构造函数中进行加锁，析构函数中进行解锁。
2. 锁在多线程编程中，使用较多，因此c++11提供了lock_guard模板类；在实际编程中，我们也可以根据自己的场景编写resource_guard RAII类，避免忘掉释放资源。

**unique_lock**

1. unique_lock 是通用互斥包装器，允许延迟锁定、锁定的有时限尝试、递归锁定、所有权转移和与条件变量一同使用。
2. unique_lock比lock_guard使用更加灵活，功能更加强大。
3. 使用unique_lock需要付出更多的时间、性能成本。

<aside>
⚠️

多线程中有多少个共享资源就申请多少个mutex

</aside>

### 1.4 条件变量

互斥量是多线程间同时访问某一共享变量时，保证变量可被安全访问的手段。但单靠互斥量无法实现线程的同步。线程同步是指线程间需要按照预定的先后次序顺序进行的行为。C++11对这种行为也提供了有力的支持，这就是条件变量。条件变量位于头文件`condition_variable`下。 
**条件变量使用过程：**

1. 拥有条件变量的线程获取互斥量；
2. 循环检查某个条件，如果条件不满足则阻塞直到条件满足；如果条件满足则向下执行；
3. 某个线程满足条件执行完之后调用notify_one或notify_all唤醒一个或者所有等待线程。 条件变量提供了两类操作：wait和notify。这两类操作构成了多线程同步的基础。

<aside>
👉🏻

- 条件变量存放了被阻塞线程的线程ID
- condition_variable：需要配合std::unique_lock<std::mutex>进行wait操作，也就是阻塞线程的操作。
- condition_variable_any：可以和任意带有lock()、unlock()语义的mutex搭配使用，也就是说有四种：
    - std::mutex：独占的非递归互斥锁
    - std::timed_mutex：带超时的独占非递归互斥锁
    - std::recursive_mutex：不带超时功能的递归互斥锁
    - std::recursive_timed_mutex：带超时的递归互斥锁
</aside>

### 1.5 成员函数

1. **wait函数**
    
    **它的函数原型是：**
    
    ```cpp
    void wait (unique_lock<mutex>& lck);
    template <class Predicate>
    void wait (unique_lock<mutex>& lck, Predicate pred);
    ```
    
    包含两种重载（同一个函数名可以根据不同的参数类型或数量，定义多个不同的函数），第一种只包含unique_lock对象，另外一种包含一个Predicate对象（等待条件）。wait函数的工作原理：
    
    - 当前线程调用wait()后将被阻塞并且函数会解锁互斥量(允许其他线程访问共享资源)，直到另外某个线程调用notify_one或者 notify_all唤醒当前线程；一旦当前线程获得通知(notify)，wait()函数也是自动调用lock()，同理不能使用lock_guard对象。
    - 如果wait没有第二个参数，第一次调用默认条件不成立，直接解锁互斥量并阻塞到本行，直到某一 个线程调用notify_one或notify_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程会卡在这里，直到获取到互斥量，然后无条件地继续进行后面的操作。
    - 如果wait包含第二个参数，如果第二个参数不满足，那么wait将解锁互斥量并堵塞到本行，直到某 一个线程调用notify_one或notify_all为止，被唤醒后，wait重新尝试获取互斥量，如果得不到，线程会卡在这里，直到获取到互斥量，然后继续判断第二个参数，如果表达式为false，wait对互斥量解锁，然后休眠，如果为true，则进行后面的操作。
    
    <aside>
    👉🏻
    
    wait阻塞之前会解锁，解除阻塞之后加锁
    
    </aside>
    
2. **wait_for函数**
    
    函数原型：
    
    ```cpp
    template <class Rep, class Period>
    cv_status wait_for (unique_lock<mutex>& lck,
             const chrono::duration<Rep,Period>& rel_time);
             
    template <class Rep, class Period, class Predicate>
    bool wait_for (unique_lock<mutex>& lck,
        const chrono::duration<Rep,Period>& rel_time, Predicate
        pred);
    ```
    
    和wait不同的是，wait_for可以执行一个时间段，在线程收到唤醒通知或者时间超时之前，该线程都会 处于阻塞状态，如果收到唤醒通知或者时间超时，wait_for返回，剩下操作和wait类似。
    
3. **wait_until函数**
    
    函数原型：
    
    ```cpp
    template <class Clock, class Duration>
    cv_status wait_until (unique_lock<mutex>& lck,
        const chrono::time_point<Clock,Duration>& abs_time);
                              
    template <class Clock, class Duration, class Predicate>
    bool wait_until (unique_lock<mutex>& lck,
        const chrono::time_point<Clock,Duration>& abs_time,
        Predicate pred);
    ```
    
4. **notify_one函数**
    
    函数原型：
    
    ```cpp
    void notify_one() noexcept;
    ```
    
    解锁正在等待当前条件的线程中的一个，如果没有线程在等待，则函数不执行任何操作，如果正在等待的线程多余一个，则唤醒的线程是不确定的。
    
5. **notify_all函数**
    
    函数原型：
    
    ```cpp
    void notify_all() noexcept;
    ```
    
    解锁正在等待当前条件的所有线程，如果没有正在等待的线程，则函数不执行任何操作。
    

**范例：（使用条件变量实现一个同步队列，同步队列作为一个线程安全的数据共享区，经常用于线程之间的数据读取）**

```cpp
// sync_queue.h
#ifndef SYNC_QUEUE_H
#define SYNC_QUEUE_H
#include<list>
#include<mutex>
#include<thread>
#include<condition_variable>
#include <iostream>

template<typename T>
class SyncQueue
{
private:
    bool IsFull() const
    {
        return _queue.size() == _maxSize;
    }
    bool IsEmpty() const
    {
        return _queue.empty();
    }
public:
    SyncQueue(int maxSize) : _maxSize(maxSize)
    {
    }
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        // _notFull.wait(_mutex, [this] {return !IsFull();});
        while (IsFull())
        {
            std::cout << "full wait..." << std::endl;
            _notFull.wait(_mutex);
        }
        _queue.push_back(x);
        _notFull.notify_one();
    }
    void Take(T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        //如果只有一个任务，但是唤醒了多个消费者线程，
        //则需要消费者线程wait后判断队列是不是空的，解决方法就是将if empty改为while empty
        while (IsEmpty())
        {
            std::cout << "empty wait.." << std::endl;
            _notEmpty.wait(_mutex);
        }
        x = _queue.front();
        _queue.pop_front();
        _notFull.notify_one();
    }
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.empty();
    }
    bool Full()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size() == _maxSize;
    }
    size_t Size()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size();
    }
    int Count()
    {
        return _queue.size();
    }
private:
    std::list<T> _queue; //缓冲区
    std::mutex _mutex; //互斥量和条件变量结合起来使用
    std::condition_variable_any _notEmpty;//不为空的条件变量
    std::condition_variable_any _notFull; //没有满的条件变量
    int _maxSize; //同步队列最大的size
};
#endif // SYNC_QUEUE_H
```

```cpp
// main.cpp
#include "Sync_queue.h"

using namespace std;
SyncQueue<int> syncQueue(5);
void PutDatas()
{
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Put(888);
    }
    std::cout << "PutDatas finish\n";
}
void TakeDatas()
{
    int x = 0;
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Take(x);
        std::cout << x << std::endl;
    }
    std::cout << "TakeDatas finish\n";
}
int main(void)
{
    std::thread t1(PutDatas);
    std::thread t2(TakeDatas);
    t1.join();
    t2.join();
    std::cout << "main finish\n";
    return 0;
}
```

这里需要注意的是，wait函数中会释放mutex，而lock_guard这时还拥有mutex，它只会在出了作用域 之后才会释放mutex，所以这时它并不会释放，但执行wait时会提取释放mutex。 从语义上看这里使用lock_guard会产生矛盾，但是实际上并不会出问题，因为wait提前释放锁之后会处 于等待状态，在被notify_one或者notify_all唤醒后会先获取mutex，这相当于lock_guard的mutex在 释放之后又获取到了，因此，在出了作用域之后lock_guard自动释放mutex不会有问题。 这里应该用unique_lock，因为unique_lock不像lock_guard一样只能在析构时才释放锁，它可以随时释 放锁，因此在wait时让unique_lock释放锁从语义上更加准确。

使用unique_lock和condition_variable_variable改写上面的代码，用等待一个判 断式的方法来实现一个简单的队列。 

```cpp
// SYNC_QUEUE_UNILCK_HPP
#ifndef __SYNC_QUEUE_UNILCK_HPP__
#define __SYNC_QUEUE_UNILCK_HPP__

#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <iostream>

template<typename T>
class SimpleSyncQueue
{
public:
    SimpleSyncQueue(){}
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        _queue.push_back(x);
        _notEmpty.notify_one();
    }
    void Take(T& x)
    {
        std::unique_lock<std::mutex> locker(_mutex);
        _notEmpty.wait(locker, [this]{return !_queue.empty(); });
        x = _queue.front();
        _queue.pop_front();
    }
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.empty();
    }
    size_t Size()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size();
    }
private:
    std::list<T> _queue;
    std::mutex _mutex;
    std::condition_variable _notEmpty;
};

#endif // __SYNC_QUEUE_UNILCK_HPP__

// SYNC_QUEUE_UNILCK.cpp
#include "Sync_queue_unilck.h"
using namespace std;

SimpleSyncQueue<int> syncQueue;
void PutDatas()
{
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Put(888);
    }
}
void TakeDatas()
{
    int x = 0;
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Take(x);
        std::cout << x << std::endl;
    }
}
int main(void)
{
    std::thread t1(PutDatas);
    std::thread t2(TakeDatas);
    t1.join();
    t2.join();
    std::cout << "main finish\n";
    return 0;
}
```

> 在多线程环境中，当使用固定大小的队列（如基于数组的队列）时，我们通常需要两个条件变量：一个用于同步队列非空（`not_empty`），另一个用于同步队列未满（`not_full`）。这是因为固定大小的队列在满了之后不能再添加新元素，否则会发生溢出；同样，空了之后就不能移除元素，否则会发生下标越界。
> 
> 
> 然而，`std::list` 是一个动态数据结构，它不基于连续的内存分配，而是通过指针链接各个元素。这意味着 `std::list` 可以动态地增长和缩减，没有固定的最大容量限制（除了系统内存的大小）。因此，除非自己实现了某种形式的容量限制逻辑，否则 `std::list` 本身不会因添加元素而“溢出”。
> 

### 1.6 Call_once 和 Once_flag使用

在多线程中，有一种场景是某个任务只需要执行一次，可以用C++11中的`std::call_once`函数配合 `std::once_flag`来实现。多个线程同时调用某个函数，`std::call_once`可以保证多个线程对该函数只调用一 次。

```cpp
#include <iostream>
#include <thread>
#include <mutex>

std::once_flag flag1, flag2;
void simple_do_once()
{
    std::cout << "simple_do_once\n" ;
    std::call_once(flag1, [](){ std::cout << "Simple example: called once\n";
                              });
}
void may_throw_function(bool do_throw)
{
    if (do_throw) {
        std::cout << "throw: call_once will retry\n"; //
        throw std::exception();
    }
    std::cout << "Didn't throw, call_once will not attempt again\n"; // 保证一次
}
void do_once(bool do_throw)
{
    try {
        std::call_once(flag2, may_throw_function, do_throw);
    }
    catch (...) {
    }
}
int main()
{
    std::thread st1(simple_do_once);
    std::thread st2(simple_do_once);
    std::thread st3(simple_do_once);
    std::thread st4(simple_do_once);
    st1.join();
    st2.join();
    st3.join();
    st4.join();
    std::thread t1(do_once, false);
    std::thread t2(do_once, false);
    std::thread t3(do_once, false);
    std::thread t4(do_once, true);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

```

单例模式下的应用：

```cpp
#include <iostream>
#include <mutex>
#include <thread>
using namespace std;

once_flag g_flag;
// 编写一个单例模式的类-->懒汉模式：在第一次调用getInstance()方法时，才会创建对象
// 多线程环境下，懒汉模式是线程不安全的，多个线程同时调用getInstance()方法时，会创建多个对象
// 解决方案：加锁，但是效率低
// 解决方案：C++11之后，使用call_once()函数，保证线程安全
class Base
{
public:
    Base(const Base& obj) = delete;
    Base& operator=(const Base& obj) = delete;
    static Base* getInstance()
    {
        call_once(g_flag, [&]()
            {
                obj = new Base;
                cout << "Base实例来也!!!" << endl;
            });
        return obj;
    }

    void setName(string name)
    {
        this->name = name;
    }

    string getName()
    {
        return name;
    }
private:
    Base() {};
    static Base* obj;
    string name;
};
Base* Base::obj = nullptr;

void myFunc(string name)
{
    Base::getInstance()->setName(name);
    cout << "my name is: " << Base::getInstance()->getName() << endl;
}

int main()
{
    thread t1(myFunc, "Mike");
    thread t2(myFunc, "Conan");
    thread t3(myFunc, "Like");
    thread t4(myFunc, "Nini");
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    return 0;
}
```
