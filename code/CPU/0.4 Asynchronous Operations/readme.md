## 2.5 异步操作(asynchronous operations)
定义：异步操作指**不阻塞当前线程**的任务执行方式。传统的同步操作会一直等待结果，而异步操作允许程序**继续执行其他任务**，提高效率。简单来说：

- 同步：去餐厅点菜，等饭做好。
- 异步：点了菜，去来了一把csgo，饭好了回来吃。

C++ 提供 **多种方式** 来实现异步操作，包括 **std::thread, std::async, 线程池, 协程, Boost.Asio** 等。

## 1 std::future

### 1.1 future的理论知识

future代表的是一次性事件，从一个异步调用的角度来说，future更像是执行函数的返回值，C++标准库使用std::future为一次性事件建模，**如果一个事件需要等待特定的一次性事件，那么这线程可以获取一 个future对象来代表这个事件。** 异步调用往往不知道何时返回，但是如果异步调用的过程需要同步，或者说后一个异步调用需要使用前 一个异步调用的结果。这个时候就要用到future。 线程可以周期性的在这个future上等待一小段时间，检查future是否已经ready，如果没有，该线程可以先去做另一个任务，一旦future就绪，该future就无法复位（无法再次使用这个future等待这个事件）。
**库的头文件中声明了两种future，唯一future（std::future）和共享future（std::shared_future）**。前者的实例是仅有的一个指向其关联事件的实例，后者可以有多个实例指向同一个关联事件。当事件就绪时，所有指向同一事件的 std::shared_future实例会就绪。
std::future是一个模板，模板参数就是期待返回的类型，**虽然std::future被用于线程间通 信，但其本身却并不提供同步访问，必须通过互斥量或其他同步机制来保护访问。 std::future使用的时机是当你不需要立刻得到一个结果的时候，你可以开启一个线程帮你去做一项任务，并期待这个任务的返回，但是std::thread并没有提供这样的机制。这就需要用到std::async和std::future （都在头文件中声明），std::async返回一个std::future对象，而不是给你一个确定的值（所以当你不需要立刻使用此值的时候才需要用到这个机制）。当你需要使用这个值的时候，对std::future使用get()，线程就会阻塞直到std::future就绪，然后返回该值。**

### 1.2 future的代码定义

`future`是一个**模板类**，也就是这个类可以存储任意指定类型的数据。库文件中如下：

```cpp
template< class T > class future;
template< class T > class future<T&>;
template<>          class future<void>;
```

同时也提供了**三种重载的构造函数：**

```cpp
// 1 默认无参构造函数
future() noexcept;
// 2 移动构造函数，转移资源的所有权
future( future&& other ) noexcept;
// 3 使用=delete显示删除拷贝构造函数, 不允许进行对象之间的拷贝
future( const future& other ) = delete;
```

常用的**成员函数(public)：**

一般情况下使用`=`进行赋值操作就进行对象的拷贝，但是`future`对象不可用复制。那么就要：

- 如果`other`是右值，那么转移资源的所有权
- 如果`other`是非右值，不允许进行对象之间的拷贝（该函数被显示删除禁止使用）

```cpp
future& operator=( future&& other ) noexcept;
future& operator=( const future& other ) = delete;
```

取出`future`对象内部保存的数据，其中`void get()`是为`future<void>`准备的，此时对象内部类型就是`void`，该函数是一个阻塞函数，当子线程的数据就绪后解除阻塞就能得到传出的数值了。

```cpp
T get();
T& get();
void get();
```

因为`future`对象内部存储的是异步线程任务执行完毕后的结果，是在调用之后的将来得到的，因此可以通过调用`wait()`方法，阻塞当前线程，等待这个子线程的任务执行完毕，任务执行完毕当前线程的阻塞也就解除了。

```cpp
void wait() const;
```

如果当前线程`wait()`方法就会死等，直到子线程任务执行完毕将返回值写入到`future`对象中，调用`wait_for()`只会让线程阻塞一定的时长，但是这样并不能保证对应的那个子线程中的任务已经执行完毕了。

<aside>
⚠️

`get()` 既有等待又有获取结果的功能，而 `wait()` 只有等待的功能。

</aside>

`wait_until()`是阻塞到某一指定的时间点，`wait_until()`和`wait_for()`函数的对比代码：

```cpp
template< class Rep, class Period >
std::future_status wait_for( const std::chrono::duration<Rep,Period>& timeout_duration ) const;

template< class Clock, class Duration >
std::future_status wait_until( const std::chrono::time_point<Clock,Duration>& timeout_time ) const;

```

当`wait_until()`和`wait_for()`函数返回之后，并不能确定子线程当前的状态，因此我们需要判断函数的返回值，这样就能知道子线程当前的状态了，这个子线程当前的状态`std::future_status`返回三种常量类型：

- `future_status::deferred`， 子线程中的任务函仍未启动
- `future_status::ready`，子线程中的任务已经执行完毕，结果已就绪
- `future_status::timeout`，子线程中的任务正在执行中，超时

## 2 std::promise

### 2.1 promise的理论定义

`std::promise`是一个协助线程赋值的类，它能够将数据和`future`对象绑定起来，为获取线程函数中的某个值提供便利。

### 2.2 promise的代码定义

这也是一个模板类，我们要在线程中传递什么类型的数据，模板参数就指定为什么类型。

```cpp
// 定义于头文件 <future>
template< class R > class promise;
template< class R > class promise<R&>;
template<>          class promise<void>;
```

同样也是三种重载的构造函数：

```cpp
// 1 默认构造函数，得到一个空对象
promise();
// 2 移动构造函数
promise( promise&& other ) noexcept;
// 3 使用=delete显式删除拷贝构造函数, 不允许进行对象之间的拷贝
promise( const promise& other ) = delete;
```

在`promise`类的内部有一个`future`类对象，调用`get_future()`就可以得到这个`future`对象了

```cpp
std::future<T> get_future();
```

存储要传出的 `value` 值，并立即让状态就绪，这样数据被传出其它线程就可以得到这个数据了。重载的第四个函数是为`promise<void>`类型的对象准备的。

```cpp
void set_value( const R& value );
void set_value( R&& value );
void set_value( R& value );
void set_value();
```

存储要传出的 `value` 值，但是不立即令状态就绪。在当前线程退出时，子线程资源被销毁，再令状态就绪。

```cpp
void set_value_at_thread_exit( const R& value );
void set_value_at_thread_exit( R&& value );
void set_value_at_thread_exit( R& value );
void set_value_at_thread_exit();
```

> `set_value(value_type v)`：此方法用于立即设置 `std::promise` 持有的值。一旦调用了 `set_value`，任何与之关联的 `std::future` 或 `std::shared_future` 对象都将能够获取这个值。如果 `set_value` 被多次调用，或者与 `set_exception` 一起调用，将抛出异常。
> 

> `set_value_at_thread_exit(value_type v)`：此方法与 `set_value` 类似，但它用于设置值，但设置操作会推迟到调用线程退出时才执行。这意味着，如果你在创建 `std::promise` 的线程中调用 `set_value_at_thread_exit`，并且在此线程的 `std::promise` 对象被销毁之前没有退出，那么值将不会被设置。这在某些情况下很有用，特别是当生成值的计算可能涉及线程本身的本地资源，而这些资源只有在线程退出时才可用。
> 

---

### 2.3 promise的使用

通过`promise`传递数据的过程一共分为5步：

1. 在主线程中创建`std::promise`对象
2. 将这个`std::promise`对象通过引用的方式传递给子线程的任务函数
3. 在子线程任务函数中给`std::promise`对象赋值
4. 在主线程中通过`std::promise`对象取出绑定的`future`实例对象
5. 通过得到的`future`对象取出子线程任务函数中返回的值。

```cpp
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
```

```cpp
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
```

## **3 std::packaged_task**

### 3.1 **packaged_task理论**定义

packaged_task对象获取任务相关联的future，调用get_future()方法可以获得 std::packaged_task对象绑定的函数的返回值类型的future。**std::packaged_task的模板参数是函数签名。 PS：例如int add(int a, intb)的函数签名就是int(int, int)。**

```cpp
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

    operator funcPtr() {
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
```

`using funcPtr = string (*)(string, int);`是C++11 的 `using` 类型别名。`funcPtr` 变成了 **函数指针类型**它表示一个指向 `string (string, int)` 形式的函数指针。

| **`packaged_task` 任务** | **包装内容** | **说明** |
| --- | --- | --- |
| `task1` | `myFunc()` | **普通函数** |
| `task2` | `lambda` | **匿名函数** |
| `task3` | `Base` | **仿函数（`operator()`）** |
| `task4` | `Base` 转换 `showMsg()` | **类对象转换为函数指针** |
| `task5` | `Base::showMsg()` | **静态成员函数** |
| `task6` | `Base::getNumber()` 绑定 | **绑定非静态成员函数** |

## **4** `std::promise`、`std::packaged_task`和`std::future`的关系

std::future、std::promise和std::packaged_task都是std::async相关的几个对象。其中 std::promise和std::packaged_task的结果最终都是通过其内部的future返回出来的。

std::future提供了一个访问异步操作结果的机制，它和线程是一个级别的，属于低层次的对象，在它之上高一层的是std::packaged_task和std::promise，他们内部都有future以便访问异步操作结果，std::packaged_task包装的是一个异步操作，而std::promise包装的是一个值，都是为了方便异步操作的，因为有时我需要获取线程中的某个值，这时就用std::promise，而有时我需要获一个异步操作的返回值，这时就用std::packaged_task。

## 5 std::async

std::async是为了让用户的少费点脑子的，它让std::future、std::promise和std::packaged_task这三个对象默契的工作。**工作过程是这样的：std::async先将异步操作用std::packaged_task包装起来，然后将异步操作的结果放到std::promise中，这个过程就是创造future()的过程。外面再通过future.get()/wait()来获取future()的结果，你不用再想到底该怎么用std::future、std::promise和 std::packaged_task了，std::async已经帮你搞定一切了！**

### 5.1 std::async的原型

```cpp
async(std::launch::async | std::launch::deferred, f, args...)
```

第一个参数是线程的创建策略，有两种策略，默认的策略是立即创建线程：

- `std::launch::async`：在调用async就开始创建线程。
- `std::launch::deferred`：延迟加载方式创建线程。调用async时不创建线程，直到调用了future()的get()或者wait()时才创建线程。

第二个参数是线程函数，第三个参数是线程函数的参数，函数返回值是一个`future`对象。

---

下面是调用async()函数直接创建线程执行任务的代码：

```cpp
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
```

如果没有指定第一个创建策略，就会默认直接开始创建线程，得到上面的results，如果指定了`launch::deferred`就会得到下面的输出。注意：只有`future`类对象调用了`get()`或者`wait()`方法后才开始执行任务（此处一定要注意调用wait_for()函数是不行的）。