## **1 右值引用(Rvalue Reference)**

**定义：左值**是持久的、有明确内存地址的对象。**右值**是临时的、不可寻址的值。**C++11**中引用了右值引用和移动语义，可以避免无谓的复制，提高了程序性能。左值是表达式结束后仍然存在的持久对象，**右值是指表达式结束时就不存在的临时对象。**

**区分左值和右值：**

- 看能不能对表达式取地址，如果能则为左值，否则为右值；
- 将要被移动的对象、T&&函数返回的值、std::move返回值和转换成T&&的类型的转换函数返回值。以上都是**将亡值**。

C++11之后所有的值都必须属于**左值、将亡值、纯右值三者之一，将亡值和纯右值都属于右值。** 

### 1.1 **`&&`特性**

因为右值没有名字，所以我们只能通过引用的方式找到它。所以**右值引用就是对一个右值进行引用的类型。**无论声明左值引用还是右值引用都必须立即进行初始化，因为引用类型本身并不拥有绑定对象的内存，只是该对象的一个别名。

**通过右值引用的声明，该右值又“重获新生”，其生命周期其生命周期与右值引用类型变量的生命周期一 样，只要该变量还活着，该右值临时量将会一直存活下去。**

**在C++中，并不是所有情况下 && 都代表是一个右值引用，具体的场景体现在模板和自动类型推导中，如果是模板参数需要指定为`T&&`，如果是自动类型推导需要指定为`auto &&`，在这两种场景下 &&被称作未定的引用类型或者通用引用（universal reference）。另外还有一点需要额外注意`const T&&`表示一个右值引用，不是未定引用类型。例如：**

```cpp
int main()
{
    int x = 520, y = 1314;
    auto&& v1 = x;                   // v1 的类型是 int&
    auto&& v2 = 250;
    decltype(x)&& v3 = y;            // error, v3 的类型是 int&&
    decltype(x)&& v3 = std::move(y); // 正确
    decltype(x)&  v3 = y;            // 正确
    cout << "v1: " << v1 << ", v2: " << v2 << endl;
    return 0;
};
```

`auto&&` 是一个通用引用，它可以绑定到左值或右值。当 `auto&&` 绑定到一个左值时，它会推导出一个左值引用类型。`auto&& v1 = x;` 依赖**引用折叠**，最终 `v1` 变成了 `int&`，可以绑定到 `x`，所以不会报错。但是`decltype(x)&&`等价于`int&&`是一个右值引用不是未定引用类型，y是一个左值，不能使用左值初始化一个右值引用类型。

在C++11中**引用折叠**的规则如下：

- **通过右值推导 `T&&` 或者 `auto&&` 得到的是一个右值引用类型**
- 通过非右值（**右值引用**、左值、左值引用、**常量右值引用**、常量左值引用）推导 `T&&` 或者 `auto&&` 得到的是一个左值引用类型

范例代码1：

```cpp
#include <iostream>

int main() {
    int&& a1 = 5;
    auto&& bb = a1; //a1为右值引用，推导出的bb为左值引用类型
    auto&& bb1 = 5; //5为右值，推导出的bb1为右值引用类型

    int a2 = 5;
    int &a3 = a2;
    auto&& cc = a3; //a3为左值引用，推导出的cc为左值引用类型
    auto&& cc1 = a2;//a2为左值，推导出的cc1为左值引用类型

    const int& s1 = 100; 
    const int&& s2 = 100;
    auto&& dd = s1; //s1为常量左值引用，推导出的dd为常量左值引用类型
    auto&& ee = s2; //s2为常量右值引用，推导出的ee为常量左值引用类型

    const auto&& x = 5;//x为右值引用，不需要推导，只能通过右值初始化

    // 输出验证
    std::cout << "bb (int&): " << bb << std::endl;
    std::cout << "bb1 (int&&): " << bb1 << std::endl;
    std::cout << "cc (int&): " << cc << std::endl;
    std::cout << "cc1 (int&): " << cc1 << std::endl;
    std::cout << "dd (const int&): " << dd << std::endl;
    std::cout << "ee (const int&): " << ee << std::endl;
    std::cout << "x (const int&&): " << x << std::endl;

    return 0;
}
/******************************************************************
bb (int&): 5
bb1 (int&&): 5
cc (int&): 5
cc1 (int&): 5
dd (const int&): 100
ee (const int&): 100
x (const int&&): 5
 *****************************************************************/
```

范例代码2：

```cpp
#include <iostream>
using namespace std;

void printValue(int &i) {
    cout << "l-value: " << i << endl;
}

void printValue(int &&i) {
    cout << "r-value: " << i << endl;
}

void forward(int && k) {
    printValue(k);
}

int main() {
    int value = 500;
    printValue(value);
    printValue(500);
    forward(1000);
    return 0;
}

/******************************************************************
l-value: 500
r-value: 500
l-value: 1000
 *****************************************************************/
```

可以看到这里的函数forward()接收的是一个右值，但是在这个函数中调用函数printValue()时，参数k变成了一个命名对象，编译器会将其当做左值来处理。

**总结如下：**

- 左值和右值独立于它们的类型的，右值引用类型可能是左值也可能是右值。见范例代码1
- `auto&&` 或函数参数类型自动推导的 `T&&` 是一个未定的引用类型，被称为 `universal references`， 它可能是左值引用也可能是右值引用类型，取决于初始化的值类型。见范例代码2
- 所有的右值引用叠加到右值引用上仍然是一个右值引用，其他引用折叠都为左值引 用。当 T&& 为 模板参数时，输入左值，它会变成左值引用，而输入右值时则变为具名的右值引用。
- **编译器会将已命名的右值引用视为左值，而将未命名的右值引用视为右值。**

### 1.2 右值引用优化性能

含有堆内存的类，我们需要提供深拷贝（复制对象的所有数据，并为指针成员重新分配新的堆内存）的拷贝构造函数，如果使用默认构造函数，会导致堆内存的重复删除，比如下面的代码：

```cpp
// Rvalue_Reference_exp3.cpp
#include <iostream>
using namespace std;

class A {
public:
    A() : m_ptr(new int (42)) {// 分配堆内存
        cout << "constructor A" << endl;
    }
    ~A() {
        cout << "destructor A, m_ptr:" << m_ptr << endl;
        delete m_ptr;// 释放堆内存
        m_ptr = nullptr;
    } 
private:
    int *m_ptr;
};

A GetA(bool f) {
    A a;
    A b;
    cout << "ready return" << endl;
    if (f) return a;
    else return b;
}

int main() {
    A a = GetA(false); // 运行报错, a 接收返回值，拷贝构造触发
    return 0;
}
/******************************************************************
constructor A
constructor A
ready return
destructor A, m_ptr:0x55ce0bd692e0
destructor A, m_ptr:0x55ce0bd68eb0
destructor A, m_ptr:0x55ce0bd692e0
free(): double free detected in tcache 2
Aborted (core dumped)
 *****************************************************************/
```

这里的GetA函数结束的时候，a和b都会被销毁。返回的b是编译器创建的 **该对象的副本，**由于代码中没有写拷贝构造函数，**C++ 编译器会生成一个默认的拷贝构造函数**，它只会执行 **浅拷贝**（成员变量按值复制，指针只是复制地址，不会分配新内存）。现在返回的b会在`A a = GetA(false);`给a，现在a和b的成员变量`m_ptr`都指向一块内存，**`b` 先被析构**，释放 `m_ptr` 指向的内存。**`a` 之后被析构**，又释放了一次 `m_ptr` 指向的同一块内存（**二次释放，double free！**）。

> **深浅拷贝知识点：**
> 
- 当数据成员中没有指针时，浅拷贝是可行的；但当数据成员中有指针时，如果采用简单的浅拷贝，则两类中的两个指针将指向同一个地址，当对象快结束时，会调用两次析构函数，而导致指针悬挂现象，所以，此时，必须采用深拷贝。
- 深拷贝与浅拷贝的区别就在于深拷贝会在堆内存中另外申请空间来储存数据，从而也就解决了指针悬挂的问题。简而言之，数据成员中有指针时，必须要用深拷贝。

那么如何保证刚刚的拷贝构造的安全性呢？答案是使用深拷贝。在刚刚的类A中加上如下代码：

```cpp
// Rvalue_Reference_exp3.cpp
A (const A& a) : m_ptr(new int(*a.m_ptr)) { // A (const A& a)是 拷贝构造函数的声明，表示用已有的A对象a来创建新的A对象。
        cout << "copy construction A, new m_ptr: " << m_ptr << endl;
    }  // new int(*a.m_ptr) 在堆上分配一个新的 int，并赋值为 a.m_ptr 指向的值。
```

此时运行可执行文件可以看到析构掉每个对象的时候，内存地址是不同的。但这种拷贝构造却是不必要的，GetA函数会返回临时变量，然后通过这个临时变量拷贝构造了一个新的对象 b，临时变量在拷贝构造完成之后就销毁了，如果堆内存很大，那么，这个拷贝构造的代价会很大， 带来了额外的性能损耗。有没有办法避免临时对象的拷贝构造呢？答案是使用：移动构造（ Move Construct）。
现在class A的代码是：

```cpp
class A
{
public:
    A() :m_ptr(new int(0)) {
        cout << "constructor A" << endl;
    }
    A(A && a) :m_ptr(a.m_ptr) {
        a.m_ptr = nullptr;
        cout << "move constructor A" << endl;
    }
    ~A(){
        cout << "destructor A, m_ptr:" << m_ptr << endl;
        if(m_ptr)
            delete m_ptr;
    }
private:
    int* m_ptr;
};
```

这里的移动构造函数`A(A && a)`的参数是一个右值引用类型的参数，前面说到过右值是临时值没有内存地址的，这里的`A&&` 用来根据参数是左值还是右值来建立分支，如果是临时值，则会选择移动构造函数。移动构造函数只是将临时对象的资源做了浅拷贝，不需要对其进行深拷贝，从而避免了额外的拷贝，提高性能。这也就是所谓的移动语义，右值引用的一个重要目的是用来支持移动语义的。
**移动语义可以将资源（堆、系统对象等）通过浅拷贝方式从一个对象转移到另一个对象，这样能够减少不必要的临时对象的创建、拷贝以及销毁，可以大幅度提高 C++ 应用程序的性能，消除临时对象的维护 （创建和销毁）对性能的影响。**
接下来再举一个范例用移动语义优化性能的例子，如：

```cpp
// Rvalue_Reference_exp4.cpp
#include <iostream>
#include <vector>
#include <string.h>
using namespace std;

class mystr {
private:
    char *data;
    size_t len;
    void copy_data(const char *s) {
        data = new char[len + 1];
        memcpy(data, s, len);
        data[len] = '\0';
    }

public:
    mystr() {
        data = NULL;
        len = 0;
    }
    mystr(const char *p) {
        len = strlen(p);
        copy_data(p);
    }
    mystr(const mystr& str) {
        len = str.len;
        copy_data(str.data);
        cout << "copy constructor is called! source :" << str.data << endl;
    }

    mystr& operator=(const mystr& str) {
        if (this != &str) {
		        delete[] data;  // 释放旧内存，避免泄漏
            len = str.len;
            copy_data(str.data);
        }
        cout << "copy assignment is called! source :" << str.data << endl;
        return *this;
    }
    virtual ~mystr() {
        if (data) free(data);
    }
};

void test () {
    mystr str;
    str = mystr("Hello World");
    vector<mystr> vec;
    vec.emplace_back(mystr("bro"));
}

int main() {
    test();
    return 0;
}
/******************************************************************
copy assignment is called! source :Hello World
copy constructor is called! source :bro
 *****************************************************************/
```

看到这段代码中的`test`函数中两个字符串`Hello World`和`bro`分别调用了拷贝构造和拷贝赋值函数，然而这俩字符串都是临时值，造成了没有意义的开销。如果能够直接使用临时对象已经申请的资源，就能节省资源申请和释放的时间。现在使用右值引用来定义优化：

```cpp
//修改后的代码
#include <iostream>
#include <vector>
#include <string.h>
using namespace std;

class mystr {
private:
    char *data;
    size_t len;
    void copy_data(const char *s) {
        data = new char[len + 1];
        memcpy(data, s, len);
        data[len] = '\0';
    }

public:
    mystr() {
        data = NULL;
        len = 0;
    }
    mystr(const char *p) {
        len = strlen(p);
        copy_data(p);
    }
    // mystr(const mystr& str) {
    //     len = str.len;
    //     copy_data(str.data);
    //     cout << "copy constructor is called! source :" << str.data << endl;
    // }
    mystr(mystr&& str) noexcept {
        cout << "move constructor is called! source: " << str.data << endl;
        data = str.data;
        len = str.len;
        str.data = nullptr;
        str.len = 0;
    }
    mystr& operator=(mystr&& str) noexcept {
        cout << "move assignment is called! source: " << str.data << endl;
        if (this != &str) {
            delete[] data;
            data = str.data;
            len = str.len;
            str.data = nullptr;
            str.len = 0;
        }
        return *this;
    }
    // mystr& operator=(const mystr& str) {
    //     if (this != &str) {
    //         delete[] data;  // 释放旧内存，避免泄漏
    //         len = str.len;
    //         copy_data(str.data);
    //     }
    //     cout << "copy assignment is called! source :" << str.data << endl;
    //     return *this;
    // }
    virtual ~mystr() {
        if (data) free(data);
    }
};

void test () {
    mystr str;
    str = mystr("Hello World");
    vector<mystr> vec;
    vec.emplace_back(mystr("bro"));
}

int main() {
    test();
    return 0;
}
/******************************************************************
copy assignment is called! source :Hello World
copy constructor is called! source :bro
 *****************************************************************/

/******************************************************************
move assignment is called! source: Hello World
move constructor is called! source: bro
 *****************************************************************/
```

我们在设计和实现类时，对于需要动态申请大量资源的类，应该考虑尽量设计右值引用的拷贝构造函数和赋值函数，以提高应用程序的效率。

## 2 移动语义(Move Semantics)

经过前文已经知道移动语义是通过右值引用来匹配临时值的，**那么，普通的左值是否也能借助移动语义来优化性能呢？C++11为了解决这个问题，提供了`std::move()`方法来将左值转换为右值，从而方便应用移动语义。`std::move`是将对象的状态或者所有权从一个对象转移到另一个对象，只是转义，没有内存拷贝。**

![image.jpg](attachment:959ccce6-50ef-4fa3-8259-b7d703963e31:image.jpg)

```cpp
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string.h>

using namespace std;

class MyString {
private:
    char* m_data;
    size_t m_len;
    void copy_data(const char *s) {
        m_data = new char[m_len+1];
        memcpy(m_data, s, m_len);
        m_data[m_len] = '\0';
    }
public:
    MyString() {
        m_data = NULL;
        m_len = 0;
    }
    MyString(const char* p) {
        m_len = strlen (p);
        copy_data(p);
    }
    MyString(const MyString& str) {
        m_len = str.m_len;
        copy_data(str.m_data);
        std::cout << "Copy Constructor is called! source: " << str.m_data <<
            std::endl;
    }
    MyString& operator=(const MyString& str) {
        if (this != &str) {
            m_len = str.m_len;
            copy_data(str.m_data);
        }
        std::cout << "Copy Assignment is called! source: " << str.m_data <<
            std::endl;
        return *this;
    }
    // 用c++11的右值引用来定义这两个函数
    MyString(MyString&& str) {
        std::cout << "Move Constructor is called! source: " << str.m_data <<
            std::endl;
        m_len = str.m_len;
        m_data = str.m_data; //避免了不必要的拷贝
        str.m_len = 0;
        str.m_data = NULL;
    }
    MyString& operator=(MyString&& str) {
        std::cout << "Move Assignment is called! source: " << str.m_data <<
            std::endl;
        if (this != &str) {
            m_len = str.m_len;
            m_data = str.m_data; //避免了不必要的拷贝
            str.m_len = 0;
            str.m_data = NULL;
        }
        return *this;
    }
    virtual ~MyString() {
        if (m_data) free(m_data);
    }
};

int main()
{
    MyString a;
    a = MyString("Hello"); // move
    MyString b = a; // copy
    MyString c = std::move(a); // move， 将左值转为右值
    return 0;
}
```

## 3 完美转发(Perfect Forwarding)

**右值引用类型是独立于值的，一个右值引用作为函数参数的形参时，在函数内部转发该参数给内部其他函数时，它就变成一个左值，并不是原来的类型了。**如果需要按照参数原来的类型转发到另一个函数，可以使用C++11提供的`std::forward()`函数，该函数实现的功能称之为完美转发。

```cpp
// 函数原型
template <class T> T&& forward (typename remove_reference<T>::type& t) noexcept;
template <class T> T&& forward (typename remove_reference<T>::type&& t) noexcept;

// 精简之后的样子
std::forward<T>(t);
```

- 当T为左值引用类型时，t将被转换为T类型的左值
- 当T不是左值引用类型时，t将被转换为T类型的右值

下面通过一个例子演示一下关于`std::forward()`的使用：

```cpp
#include <iostream>
using namespace std;

template<typename T>
void printValue(T& t)
{
    cout << "l-value: " << t << endl;
}

template<typename T>
void printValue(T&& t)
{
    cout << "r-value: " << t << endl;
}

template<typename T>
void testForward(T && v)
{
    printValue(v);
    printValue(move(v));
    printValue(forward<T>(v));
    cout << endl;
}

int main()
{
    testForward(520);
    int num = 1314;
    testForward(num);
    testForward(forward<int>(num));
    testForward(forward<int&>(num));
    testForward(forward<int&&>(num));

    return 0;
}
```

## 整体范例：

```cpp
// Rvalue_Reference_exp6.cpp
#include "stdio.h"
#include <iostream>
#include<vector>
#include <string.h>
using namespace std;

class A
{
public:
    A() : m_ptr(NULL), m_nSize(0) {}

    A(int *ptr, int nSize) {
        m_nSize = nSize;
        m_ptr = new int[nSize];
        if (m_ptr)
        {
            memcpy(m_ptr, ptr, sizeof(int) * nSize);
        }
    }

    // 拷贝构造函数（深拷贝）
    A(const A& other)
    {
        m_nSize = other.m_nSize;
        if (other.m_ptr) {
            delete[] m_ptr;
            m_ptr = new int[m_nSize];
            memcpy(m_ptr, other.m_ptr, sizeof(int)* m_nSize);
        }
        else {
            m_ptr = NULL;
        }
        cout << "Copy Constructor Called" << endl;
    }
    
    // 移动构造函数
    A(A && other) noexcept {
        m_ptr   = other.m_ptr; 
        m_nSize = other.m_nSize;
        other.m_ptr = NULL;  // 让原对象失效
        other.m_nSize = 0;
        cout << "Move Constructor Called" << endl;
    }
    
    ~A() {
        if (m_ptr) {
            delete[] m_ptr;
            m_ptr = NULL;
        }
    }
    
    void deleteptr() {
        if (m_ptr) {
            delete[] m_ptr;
            m_ptr = NULL;
        }
    }
    int *m_ptr;
    int m_nSize;
};

int main()
{
    int arr[] = { 1, 2, 3 };
    A a(arr, sizeof(arr) / sizeof(arr[0]));
    cout << "m_ptr in a Addr: 0x" << a.m_ptr << endl;

    A b(a);  // 拷贝构造
    cout << "m_ptr in b Addr: 0x" << b.m_ptr << endl;

    A c(std::move(a)); // ✅ 正确使用 move
    cout << "m_ptr in c Addr: 0x" << c.m_ptr << endl;

    vector<int> vect{ 1, 2, 3, 4, 5 };
    cout << "before move vect size: " << vect.size() << endl;

    vector<int> vect1 = move(vect); // ✅ move vect
    cout << "after move vect size: " << vect.size() << endl;
    cout << "new vect1 size: " << vect1.size() << endl;
}
/******************************************************************
m_ptr in a Addr: 0x0x561c74921eb0
Copy Constructor Called
m_ptr in b Addr: 0x0x561c749222e0
Move Constructor Called
m_ptr in c Addr: 0x0x561c74921eb0
before move vect size: 5
after move vect size: 0
new vect1 size: 5
 *****************************************************************/

```