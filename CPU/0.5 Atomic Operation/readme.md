# 2.6 原子变量 | CAS操作 | 内存顺序

> 代码：https://github.com/leoda1/the-notes-of-cuda-programming/tree/main/code/CPU

## 1. 原子变量（Atomic Variable）

### 1.1 定义

c++11中提供的原子类型`std::atomic<T>`，内部变量通过这个原子类型管理后就变为**原子变量**。原子类型模板参数`<T>`可以指定`bool、char、int、long、指针`等类型（不支持**浮点**类型和**复合**类型）。

原子指**一系列不可被CPU上下文交换**的机器指令，这些指令组合在一起就形成了**原子操作**。在多核CPU下，当某个CPU核心开始运行原子操作时，会先暂停其它CPU内核对内存的操作，以保证原子操作不会被其它CPU内核所干扰。

**由于原子操作是通过指令提供的支持，因此它的性能相比锁和消息传递会好很多。相比较于锁而言，原子类型不需要开发者处理加锁和释放锁的问题，同时支持修改，读取等操作，还具备较高的并发性能，几乎所有的语言都支持原子类型。** 

可以看出原子类型是无锁类型，但是无锁不代表无需等待，因为原子类型内部使用了`CAS`循环，当大量的冲突发生时，该等待还是得等待！但是总归**比锁要好**。
c++11内置了`int`形的原子变量，用于更方便的使用它。多线程操作中，原子变量使用后不需要使用互斥量保护该变量。因为对原子变量的操作只能是原子操作（不会被线程调度机制打断的操作，一旦开始就一直运行到结束，中间没有上下文切换）。同时，多线程访问的共享资源造成的数据混乱（比如一个线程在修改这个数据，另一个线程也在修改相同的数据），此时使用原子变量可以很好的解决。通过使用原子操作指令，如`lock cmpxchg`，`fetch_add`等来确保数据操作不可被其他线程打断。例如：

```cpp
// Atomic_Op_exp1.cpp
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
```

这个代码当中存在`td1`和`td2`两个线程执行任务f，任务f中的lamba表达式是对sum做累加，理论上得到的结果应该是2 * 12345678，但是每次都不对，就是因为多个线程同时对 `sum` 进行读取、修改、写入的操作，会引发数据竞争（data race），导致结果不确定。

在多线程环境下，如果两个线程同时进行上述操作，就可能出现以下情况：

1. 线程 A 读取 `sum` 的值为 12345678。
2. 线程 B 读取 `sum` 的值也为 12345678。
3. 线程 A 将值加 1 并写回，`sum` 的值变为 12345679。
4. 线程 B 将值加 1 并写回，`sum` 的值也变为 12345679。

这样，两个线程各执行了一次加 1 操作，但 `sum` 的值只增加了 1。这就是数据竞争导致的错误。

### 1.2 atomic的缓存机制

atomic变量不会存储到缓存中而是运算完后直接给内存，所以先谈一下缓存机制：

| 普通变量的缓存机制 | atomic变量的缓存机制 |
| --- | --- |
| 1. 先把数据加载到CPU缓存(L1/L2/L3 cache)                 2. 在cache中修改数据，合适时间再 **写回主内存** | 变量在修改后，会直接让 **所有 CPU 核心立即看到最新的值**，不会延迟同步。 |

![image.png](attachment:5f323b22-65ae-4ac6-b24e-4ab28a9cbd03:image.png)

所以，当涉及到复合类型的数据（class 或者 struct）的时候，`atomic`无能为力。复杂数据结构的修改通常需要多个步骤（如修改多个字段），而 `std::atomic` 只能保证单个变量的 **不可中断性**。
**atomic 指针本身的修改（`++/--`）是原子的，但指针指向的对象并不是原子的**，需要额外的同步机制（如 `mutex` 或 `std::shared_ptr`）。

### 1.3 atomic**类**

**atomic类定义：**

```cpp
// 定义于头文件 <atomic>
template< class T > // so, 在使用这个模板类的时候，一定要指定模板类型。
struct atomic;
```

**atomic的构造函数是：**

```cpp
// 1 默认无参构造函数。 
atomic() noexcept = default;
// 2 使用 desired 初始化原子变量的值。
constexpr atomic( T desired ) noexcept;
// 3 使用=delete显示删除拷贝构造函数, 不允许进行对象之间的拷贝
atomic( const atomic& ) = delete;

//  exp1
std::atomic<int> x;
x.store(10);
//  exp2
std::atomic<int> x(5);
//  exp3
std::atomic<int>& x
```

**atomic的公共成员函数:**

原子类型在类内部重载了`=`操作符，并且不允许在类的外部使用 `=`进行对象的拷贝。

```cpp
T operator=( T desired ) noexcept;
T operator=( T desired ) volatile noexcept;

atomic& operator=( const atomic& ) = delete;
atomic& operator=( const atomic& ) volatile = delete;
```

以原子操作的方式，将 `desired` 作为新值存储到原子变量中，并按照 `order` 指定的内存顺序来影响内存操作。

```cpp
void store( T desired, std::memory_order order = std::memory_order_seq_cst ) noexcept;
void store( T desired, std::memory_order order = std::memory_order_seq_cst ) volatile noexcept;

```

原子地加载并返回原子变量的当前值。按照 `order` 的值影响内存。直接访问原子对象也可以得到原子变量的当前值。

```cpp
T load( std::memory_order order = std::memory_order_seq_cst ) const noexcept;
T load( std::memory_order order = std::memory_order_seq_cst ) const volatile noexcept;
```

范例:

```cpp
// Atomic_Op_exp2.cpp
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
/******************************************************************
d
a
e
a
*******************************************************************/
```

### 1.4 **特化成员函数（特化：为某些特定类型提供不同的实现）**

对于整数类型的 `std::atomic<T>`，以下运算符提供 **原子修改** 变量的功能：

| **操作符** | **重载函数** | **等价的 `fetch_*` 方法** | **适用于整数类型** | **适用于指针** |
| --- | --- | --- | --- | --- |
| `+=` | `atomic::operator+=` | `atomic::fetch_add` | ✅ 是 | ✅ 是 |
| `-=` | `atomic::operator-=` | `atomic::fetch_sub` | ✅ 是 | ✅ 是 |
| `&=` | `atomic::operator&=` | `atomic::fetch_and` | ✅ 是 | ❌ 否 |
| `|=` | `atomic::operator|=` | `fetch_or(val) | val` | ✅ 是 | ❌ 否 |
| `^=` | `atomic::operator^=` | `atomic::fetch_xor` | ✅ 是 | ❌ 否 |

对于指针类型 `std::atomic<T*>`，仅支持 `+=` 和 `-=` 操作，用于 **原子地移动指针**：

| **操作符** | **重载函数** | **等价的 `fetch_*` 方法** | **适用于整数类型** | **适用于指针** |
| --- | --- | --- | --- | --- |
| `+=` | `atomic::operator+=` | `atomic::fetch_add` | ✅ 是 | ✅ 是 |
| `-=` | `atomic::operator-=` | `atomic::fetch_sub` | ✅ 是 | ✅ 是 |
- `operator+= (ptrdiff_t val)`: **指针向前移动** `val` 个元素。
- `operator-= (ptrdiff_t val)`: **指针向后移动** `val` 个元素。

## 2 CAS操作**(compare and swap，比较并交换)**

**CAS（Compare And Swap，比较并交换）** 是 **无锁编程（Lock-Free Programming）** 中的一种 **原子操作**，用于 **实现多线程安全的数据更新**。它允许多个线程 **安全地竞争更新同一个变量，而不需要使用互斥锁（mutex）**。

没有交换操作，CAS 就不完整。交换操作可通过 exchange() 成员函数实现（前面的Atomic_Op_exp2.cpp中已经用过了）。调用 exchange() 会将原子的当前值与所需值互换，并返回原来的值。所有操作均以原子方式完成。

**CAS 主要由三个值组成：**

- **原子变量（x）**：要修改的变量，通常是 `std::atomic<T>` 类型。
- **预期值（expected）**：期望 `x` 当前的值。
- **目标值（desired）**：希望 `x` 更新成的新值。

**工作原理**

- **检查 `x` 是否等于 `expected`**（即该变量在读取之后是否仍未被其他线程修改）。
- **如果相等**，说明没有其他线程修改 `x`，那么就 **将 `x` 更新为 `desired`**，并返回 `true`。
- **如果不相等**，说明 `x` 在此期间被其他线程修改过，此时 **将 `expected` 更新为 `x` 的最新值**，并返回 `false`，通常需要重新尝试（重试循环）。

CAS 在swap前增加了一个附加条件，可通过 `compare_exchange_strong()` 和 `compare_exchange_weak()`这两个函数使用。它将一个期望值与原子变量进行比较。如果它们相等，原子变量将被交换，并返回 true。否则，原子变量的当前值将被加载到expected变量，并返回 false。这种机制允许我们创建一个重试循环，而无需再次显式地读取原子值。例如：

```cpp
std::atomic<int> x(0);
int expected = x.load();
int desired = 42;
// If x == expected, x = desired and return true
// Otherwise, expected = x and return false
while(!x.compare_exchange_strong(expected, desired));
```

CAS 会失败的原因是与其他线程的争用。在我们对原子进行初始读取以获得预期值后，我们必须确保其他线程在我们上次读取原子后没有对其进行更改。假设多个线程都在尝试 CAS，那么重试循环就会一直运行，直到我们的 CAS 比其他线程更快地完成更改。

### **2.1 在需要严格保证操作成功或失败时使用 `compare_exchange_strong`**

```cpp
// Atomic_Op_exp3.cpp
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
```

### 2.2 **使用 `compare_exchange_weak` 的场景**

`compare_exchange_weak()` 可能在某些 CPU 架构（如 ARM）在 `LL/SC` 机制下，出现伪失败。即使当前值等于预期值，也返回`false`。设计出来是为了优化在某些架构上的性能，适用于需要反复尝试的场景，如实现自旋锁或无锁数据结构。
**自旋锁的代码实现：**

```cpp
// Atomic_Op_exp5.cpp
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
```

这段代码的`lockFlag` 作为 **自旋锁（spinlock）**，初始值为 `false`（表示没有线程持有锁）。10个线程完成`increment`函数（累加1000次1），这个函数每次累加的时候会获取自旋锁和释放自旋锁。

自旋锁的实现：**如果 `lockFlag == expected(false)`**，说明 **锁是空闲的**，于是将 `lockFlag` 设为 `true`（成功加锁）。**如果 `lockFlag != expected`**（说明其他线程已经持有锁），**CAS 失败**，返回 `false`，并 **更新 `expected = lockFlag`**（但这里 `expected` 直接重新设为 `false`，以便下一次尝试）。**不断循环，直到成功获取锁**。