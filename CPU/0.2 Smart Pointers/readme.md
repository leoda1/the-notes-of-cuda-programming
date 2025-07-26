C++程序设计中使用堆内存是非常频繁的操作，堆内存的申请和释放都由程序员自己管理。C++11中引入了智能指针的概念，方便管理堆内存。使用普通指针，容易造成堆内存泄露（忘记释放），二次释放，程序发生异常时内存泄露等问题等，使用智能指针能更好的管理堆内存。

C++里面的四个智能指针: `~~auto_ptr~~`,`unique_ptr`,`shared_ptr`, `weak_ptr` 其中后三个是C++11支持， 第一个已经被C++11弃用。使用这些智能指针时需要引用头文件`<memory>`。

## 1 shared_ptr共享的智能指针

共享的智能指针 `std::shared_ptr`使用引用计数，每一个`shared_ptr`的拷贝都指向相同的内存。再最后一个`shared_ptr`析构的时候，内存才会被释放。`shared_ptr`共享被管理对象，同一时刻可以有多个`shared_ptr`拥有对象的所有权，当最后一个 `shared_ptr`对象销毁时，被管理对象自动销毁。简单来说，`shared_ptr`实现包含了两部分：

- 一个指向堆上创建的对象的裸指针：`raw_ptr`
- 一个指向内部隐藏的、共享的管理对象：`share_count_object`

### 1.1 基本用法

1. **初始化：**
    
    使用new初始化，使用make_shared初始化，使用另一个shared_ptr初始化，使用unique_ptr初始化后move给shared_ptr，使用自定义删除器初始化等等。这里简单说明前三种，最推荐使用第二种(**因为他更高效**)。
    
    第一种是：`std::shared_ptr<int> ptr(new int(42));`
    
    第二种是：`std::shared_ptr<int> ptr = std::make_shared<int>(42);`
    
    第三种是：
    
    ```cpp
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::shared_ptr<int> ptr2 = ptr1; // 拷贝初始化
    ```
    
    初始化的时候会涉及两个动态内存分配：
    
    1. **对象的内存分配**：这是为了分配存储对象本身的内存，这一步是`new`表达式完成的。它会分配足够内存给这个int(42)，并调用int型的构造函数来初始化这块内存。
    2. **控制块的内存分配**：`std::shared_ptr`需要一个额外的控制块来存储与对象管理相关的信息，比如引用计数和可自定义删除器。
    
    > 在`make_shared`中只进行了一次就将这两个动态内存就分配完毕了，并且此时的对象和控制块的内存是相邻连续的，会提高缓存的使用效率。而第一种先new分配内存给int(42)，再`shared_ptr` 构造函数为自己控制块分配另一块内存。
    > 
    
    对于一个未初始化的智能指针，这里会用到`shared_ptr`的第一个成员函数`reset()`方法来初始化。当智能指针有值的时候调用`reset()`会将当前管理的对象的引用计数减1。这个管理的对象的引用计数为0就会被销毁，没到0的话当前 `shared_ptr` 不再管理它。那么不管它了就结束了吗？不是，reset有三种重载形式：无参，带指针参数，以及带指针和删除器参数。定义如下：
    
    ```cpp
    //释放当前管理的对象，并将 shared_ptr 置为空指针（即不再管理任何对象）。
    void reset() noexcept; 
    
    //释放当前管理的对象，并将 shared_ptr 指向 ptr 所指向的对象。ptr 必须是一个可以转换为 T* 
    //的指针类型（T 是 shared_ptr 管理的对象类型）。
    template <class Y>
    void reset(Y* ptr);
    
    //释放当前管理的对象，并将 shared_ptr 指向 ptr 所指向的对象，同时指定一个自定义的删除器 d。
    //删除器 d 是一个可调用对象，用于在 shared_ptr 不再管理对象时释放资源。
    template <class Y, class Deleter>
    void reset(Y* ptr, Deleter d);
    ```
    
    初始化部分范例代码：
    
    ```cpp
    // init_sample.cpp
    #include <iostream>
    #include <memory>
    using namespace std;
    
    void test(shared_ptr<int> sp) {
        cout << "sp3.use_count() =" << sp.use_count() << endl;
    }
    
    int main() {
        auto sp1 = make_shared<int>(100);  //use make_shared
        shared_ptr<int> sp2(new int(100)); //use new to init
        cout << "sp1.use_count() =" << sp1.use_count() << endl;
        cout << "sp2.use_count() =" << sp2.use_count() << endl;
        shared_ptr<int> sp3(new int(100));
        test(sp3);
        cout << "sp4.use_count() =" << sp3.use_count() << endl;
    
        shared_ptr<int> p1;
        p1.reset(new int(1)); //带参数的reset会将p1指向int(1)
        shared_ptr<int> p2 = p1;
    
        // 现在p1 和 p2 都是指向int(1)
        cout << "p2.use_count() = " << p2.use_count()<< endl;//输出2
        cout << "p1.use_count() = " << p1.use_count()<< endl;//输出2
    
        p1.reset();   // 没有参数就是释放资源
        cout << "p2.use_count() = " << p2.use_count() << endl;//输出1
        cout << "p1.use_count() = " << p1.use_count()<< endl;//输出0
    
        return 0;
    }
    /* 
    sp1.use_count() =1
    sp2.use_count() =1
    sp3.use_count() =2
    sp4.use_count() =1
    p2.use_count() = 2
    p1.use_count() = 2
    p2.use_count() = 1
    p1.use_count() = 0
    */
    ```
    
    补充：这里会用到`shared_ptr`的第二个成员函数`use_count()`：它是 `std::shared_ptr` 的一个成员函数，用来返回 **当前** `shared_ptr` **所管理的对象** 被多少个 `shared_ptr` 实例共享引用。
    
2. **获取原始指针:**
    
    当需要获取原始指针时，可以通过`get`方法来返回原始指针，代码如下所示：
    
    ```cpp
    std::shared_ptr<int> ptr(new int(1));
    int *p = ptr.get(); //万一不小心 delete p;
    ```
    
    **谨慎使用`p.get()`的返回值，如果你不知道其危险性则永远不要调用get()函数。**
    
    `p.get()`的返回值就相当于一个裸指针的值，上述陷阱的所有错误都有可能发生， 遵守以下几个约定： 
    
    - 不要保存`p.get()`的返回值 ，无论是保存为裸指针还是`shared_ptr`都是错误的。保存为裸指针不知什么时候就会变成空悬指针，保存为`shared_ptr`则产生了独立指针
    - 不要`delete` `p.get()`的返回值 ，会导致对一块内存`delete`两次的错误
3. **指定删除器:**
    
    如果用`shared_ptr`管理非`new`对象或是没有析构函数的类时，应当为其写一个合适的删除器。
    
    ```cpp
    
    #include <iostream>
    #include <memory>
    using namespace std;
    
    void deleter (int *p) {
        cout << "call Deleter delete p1" << endl;
        delete p;
    }
    
    int main() {
        shared_ptr<int> p1(new int(1), deleter);
        shared_ptr<int> p2(new int(1), [](int *p) {
            cout << "call lambda1 delete p2" << endl;
            delete p;
        });
        std::shared_ptr<int> p3(new int[10], [](int *p) {
            cout << "call lambda2 delete p3" << endl;
            delete []p;
        });
        std::shared_ptr<int> p4;
        p4.reset(new int(1), [](int* ptr) {
            std::cout << "use reset init and call lambda3 delete p4" << *ptr << std::endl;
            delete ptr;
        });
        return 0;
    }
    
    /******************************************************************
    use reset init and call lambda3 delete p41
    call lambda2 delete p3
    call lambda1 delete p2
    call Deleter delete p1
     *****************************************************************/
    ```
    
    这里p1的引用计数为0时，自动调用`deleter`来释放对象的内存。删除起器也可以写成p2中的写法`lamba`表达式。
    
    > 关于
    > 
    
    当我们使用`shared_ptr`管理动态数组需要指定删除器，如p3的写法。`shared_ptr`默认删除器不支持数组对象，在C++17支持了。
    
    ```cpp
    std::shared_ptr<int> p3(new int[10], [](int *p) {
            cout << "call lambda2 delete p3" << endl;
            delete []p;
        }); // c++11
        
    std::shared_ptr<int[]>p3(new int[10]);// C++17
    ```
    
    删除数组内存除了自己写删除器，还可以使用c++提供的`std::default_delete<T>()`函数作为删除器。如：
    
    ```cpp
    std::shared_ptr<int> p3(new int[10], default_delete<int[]>());
    ```
    
    此外，可以自己封装一个`make_shared_array`来让`shared_ptr`支持数组，如下：
    
    ```cpp
    #include <iostream>
    #include <memory>
    using namespace std;
    
    template<typename T>
    shared_ptr<T> make_shared_array(size_t size) {
        // 返回匿名对象
            return shared_ptr<T>(new T[size], default_delete<T[]>());
    }
    
    int main(){
        shared_ptr<int> p1 = make_shared_array<int>(10);
        cout << p1.use_count() << endl;
        shared_ptr<char> p2 = make_shared_array<char>(128);
        cout << p2.use_count() << endl;
        return 0;
    }
    /*
    (base) joker@joker 0.2 Smart Pointers % ./make_shared_array 
    1
    1
    */
    ```
    
    从C++20开始，`std::make_shared`支持创建数组，可以直接使用`std::make_shared`来创建并管理数组。例如：
    
    ```cpp
    #include <iostream>
    #include <memory>
    
    int main() {
        // 使用 std::make_shared 创建一个共享指针，管理一个数组
        auto array = std::make_shared<int[]>(10);
    
        // 初始化数组元素
        for (int i = 0; i < 10; ++i) {
            array[i] = i;
        }
    
        // 输出数组元素
        for (int i = 0; i < 10; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    
        return 0;
    } 
    ```
    

### 1.2 什么时候需要指定删除器？

在需要 delete 以外的析构行为的时候用。因为 `shared_ptr` 在引用计数为 0 后默认调用 `delete`来释放资源; 如果不满足需求就要提供定制的删除器。这通常发生在以下几种情况：

- 资源不是通过 `new` 分配的，或者释放方式不是 `delete`，则需要指定删除器。如，使用 `malloc` 分配的内存需要通过 `free` 释放。
- `deleter`释放单个对象，如果是数组的话要指定`delete []` 。
- 智能指针不仅可以管理内存，还可以管理其他资源（如文件句柄、网络连接、数据库连接等）。这些资源的释放方式通常不是 `delete`，因此需要指定删除器。如：管理文件的话用`fclose()`。
- 某些第三方库（如 OpenGL、CUDA 等）需要调用特定的函数来释放资源。由于资源是被第三方库管理的 (第三方提供 资源获取 和 资源释放 接口, 那么要么写一个 `wrapper` 类要么就提供定制的 deleter)。
- 资源不是 `RAII` 的, 意味着析构函数不会把资源完全释放掉，也就是单纯 `delete`还不够。（`RAII`的核心思想是 资源的获取在对象的构造函数中完成， 资源的释放在对象的析构函数中完成）

### 1.3 使用shared_ptr注意事项

1. 不要用一个原始指针初始化多个shared_ptr；
    
    ```cpp
    int *p1 = new int;
    shared_ptr<int> p2(p1);
    shared_ptr<int> p3(p1); // error
    ```
    
2. 不要在函数实参中创建`shared_ptr`;
    
    ```cpp
    function(shared_ptr<int>(new int), h()); // error
    ```
    
    因为c++的函数参数的计算顺序在不同的编译器不同的约定下是不一样的。有可能会先`new int`，再`g()` 。如果恰好`g()`发生异常，而`shared_ptr`还没有创建， 则`int`没有正确释放导致内存泄漏了，正确的写法应该是先创建智能指针，代码如下：
    
    ```cpp
    shared_ptr<int> p(new int);
    function(p, g());
    ```
    
    形参和实参的区别和联系:
    
    - 形参变量只有在函数被调用时才会分配内存，调用结束后，立刻释放内存，所以形参变量只有在函数内部有效，不能在函数外部使用。
    - 实参可以是常量、变量、表达式、函数等，无论实参是何种类型的数据，在进行函数调用时，它们都必须有确定的值，以便把这些值传送给形参，所以应该提前用赋值、输入等办法使实参获得确定值。
    - 实参和形参在数量上、类型上、顺序上必须严格一致，否则会发生“类型不匹配”的错误。当然，如果能够进行自动类型转换，或者进行了强制类型转换，那么实参类型也可以不同于形参类型。
    - 函数调用中发生的数据传递是单向的，只能把实参的值传递给形参，而不能把形参的值反向地传递给实参；换句话说，一旦完成数据的传递，实参和形参就再也没有瓜葛了，所以，在函数调用过程中，形参的值发生改变并不会影响实参。
3. 通过`shared_from_this()` 返回`this`指针，不要将`this`指针作为`shared_ptr` 返回出来，因为`this`指针本质上是一个裸指针，因此，这样可能导致重复析构。事例代码如下：
    
    ```cpp
    #include <iostream>
    #include <memory>
    
    using namespace std;
    
    class A
    {
    public:
        shared_ptr<A>GetSelf()
        {
            return shared_ptr<A>(this); // 不要这么做
        }
        ~A()
        {
            cout << "Deconstruction A" << endl;
        }
    };
    
    int main()
    {
        shared_ptr<A> sp1(new A);
        shared_ptr<A> sp2 = sp1->GetSelf();
        cout << "sp1.use_count() = " << sp1.use_count()<< endl;
        cout << "sp2.use_count() = " << sp2.use_count()<< endl;
        return 0;
    }
    
    /*
    sp1.use_count() = 1
    sp2.use_count() = 1
    Deconstruction A
    Deconstruction A
    */
    ```
    
    这个例子当中，由于同一个指针`this`构造了两个智能指针sp1和sp2，而他们之间是没有任何关系的，在离开作用域后`this`将被构造的两个智能指针各自析构，导致重复析构的错误。
    
    正确返回`this`的`shared_ptr`的做法是：让目标类通过`std::enable_shared_from_this`类，然后使用基类的 成员函数`shared_from_this()`来返回`this`的`shared_ptr`，如下：
    
    ```cpp
    #include <iostream>
    #include <memory>
    
    using namespace std;
    
    class A: public std::enable_shared_from_this<A>
    {
    public:
        shared_ptr<A>GetSelf()
        {
            return shared_from_this(); //
        }
        ~A()
        {
            cout << "Deconstruction A" << endl;
        }
    };
    
    int main()
    {
        // auto spp = make_shared<A>();
        shared_ptr<A> sp1(new A);
        shared_ptr<A> sp2 = sp1->GetSelf();  // ok
        //    shared_ptr<A> sp2;
        //    {
        //        shared_ptr<A> sp1(new A);
        //        sp2 = sp1->GetSelf();  // ok
        //    }
        cout << "sp1.use_count() = " << sp1.use_count()<< endl;
        cout << "sp2.use_count() = " << sp2.use_count()<< endl;
    
        return 0;
    }
    /*
    sp1.use_count() = 2
    sp2.use_count() = 2
    Deconstruction A
    */
    ```
    
4. 避免循环引用，会导致内存泄漏。比如：

> 什么是循环引用？
> 

**是**指在coding的时候，两个或多个对象之间形成一个循环的引用关系，导致这些对象之间的内存无法被正确释放吗，从而引发内存泄漏，这种情况也被称为循环依赖或者循环关联。

```cpp
#include <iostream>
#include <memory>
using namespace std;

class A;
class B;

class A {
public:
    std::shared_ptr<B> bptr;
    ~A() {
        cout << "A is deleted" << endl;
    }
};

class B {
public:
    std::shared_ptr<A> aptr;
    ~B() {
        cout << "B is deleted" << endl;  // 析构函数后，才去释放成员变量
    }
};

int main()
{
    std::shared_ptr<A> pa;

    {
        std::shared_ptr<A> ap(new A);
        std::shared_ptr<B> bp(new B);
        ap->bptr = bp;
        bp->aptr = ap;
        pa = ap;
        // 是否可以手动释放呢？
//        ap.reset();
        ap->bptr.reset(); // 手动释放成员变量才行
    }
    cout<< "main leave. pa.use_count()" << pa.use_count() << endl;  // 循环引用导致ap bp退出了作用域都没有析构
    return 0;
}
/*
B is deleted
main leave. pa.use_count()1
A is deleted
*/
```

循环引用导致`ap`和`bp`的引用计数为2，离开作用域后，`ap`和`bp`的引用计数减为1，并不会减为0，导致两个指针都不会被析构，产生内存泄漏。解决：把A和B任何一个成员变量改为`weak_ptr`。

## 2 unique_ptr独占的智能指针

### 2.1 定义

`unique_ptr`是一个独占型的智能指针，它不允许其他的智能指针共享其内部的指针，不允许通过赋值将 一个`unique_ptr`赋值给另一个`unique_ptr`。

```cpp
unique_ptr<T> p1(new T);
unique_ptr<T> p2 = p1; // 报错，不能复制
```

但可以通过`std::move`来转移到其他的`unique_ptr`，这样它本身就不再拥有原来指针的所有权了。例如：

```cpp
unique_ptr<T> p1(new T);
unique_ptr<T> p2 = std::move(p1); // 正确
```

在c++14当中加入了`std::make_unique`，c++11中加入的是`std::make_shared`。 使用new的版本重复了被创建对象的键入，但是make_unique函数则没有。

```cpp
auto upw1(std::make_unique<Widget>()); // with make func
std::unique_ptr<Widget> upw2(new Widget); // without make func
```

### 2.2 与shared_ptr的区别

1. C++17中的shared_ptr才支持数组
2. unique_ptr需要确定删除器的类型，所以不能像shared_ptr那样直接指定删除器，得这样写：
    
    ```cpp
    std::shared_ptr<int> ptr3(new int(1), [](int *p){delete p;}); // 正确
    std::unique_ptr<int> ptr4(new int(1), [](int *p){delete p;}); // 错误
    std::unique_ptr<int, void(*)(int*)> ptr5(new int(1), [](int *p){delete p;}); //正确
    ```
    
    使用的时候会发现，这里的捕获列表加了`&`或者`=`就会报错。lambda 表达式的捕获列表(capture list)用于指定 lambda 表达式如何访问外部变量。当你为 lambda 表达式添加捕获列表（如 `&` 或 `=`）时，lambda 表达式的类型会发生变化：从函数指针(如 `void(*)(int*)`)
    变成了仿函数(闭包类型(closure type))。综上：使用`unique_ptr`的时候指定的是void(*)(int*)类型，现在的lamba表达式是闭包类型，所以报错。
    
    需要将 `std::unique_ptr` 的删除器类型改为 `std::function<void(int*)>`，因为 `std::function` 可以存储任意可调用对象（包括有捕获的 lambda 表达式）。
    
    ```cpp
    std::unique_ptr<int, std::function<void(int*)>> ptr5(new int(1), [&](int *p) {delete p;});
    ```
    
3. 应用需求不同：如果希望只有一个智能指针管理资源或者管理数组就用unique_ptr，如果希望多个智能指针管理同一个资源就用shared_ptr。

## **3 `weak_ptr` 弱引用的智能指针**

share_ptr虽然已经很好用了，但是有一点share_ptr智能指针还是有内存泄露的情况，当两个对象相互使用一个shared_ptr成员变量指向对方，会造成循环引用，使引用计数失效，从而导致内存泄漏。

weak_ptr 是一种不控制对象生命周期的智能指针, 它指向一个 shared_ptr 管理的对象。进行该对象的内存管理的是那个强引用的shared_ptr， weak_ptr只是提供了对管理对象的一个访问手段。weak_ptr 设计的目的是为配合 shared_ptr 而引入的一种智能指针来协助 shared_ptr 工作, 它只可以从一个 shared_ptr 或另一个 weak_ptr 对象构造, 它的构造和析构不会引起引用记数的增加或减少。

weak_ptr 是用来解决shared_ptr相互引用时的死锁问题，如果说两个shared_ptr相互引用，那么这两个指针的引 用计数永远不可能下降为0，资源永远不会释放。它是对对象的一种弱引用，不会增加对象的引用计数， 和shared_ptr之间可以相互转化，shared_ptr可以直接赋值给它，它可以通过调用lock函数来获得 shared_ptr。

weak_ptr没有重载操作符`*`和`->`，因为它不共享指针，不能操作资源，主要是为了通过shared_ptr获得资源的监测权，它的构造不会增加引用计数，它的析构也不会减少引用计数，纯粹只是作为一个旁观者来监视shared_ptr中管理的资源是否存在。weak_ptr还可以返回`this`指针。

### 3.1 基本用法

1. 通过use_count()方法获取当前观察资源的引用计数，如下所示：
    
    ```cpp
    shared_ptr<int> sp(new int(10));
    weak_ptr<int> wp(sp);
    cout << wp.use_count() << endl; //结果讲输出1
    ```
    
2. 通过expired()方法判断所观察资源是否已经释放，如:
    
    ```cpp
    shared_ptr<int> sp(new int(10));
    weak_ptr<int> wp(sp);
    if(wp.expired())
        cout << "weak_ptr无效,资源已释放";
    else
        cout << "weak_ptr有效";
    ```
    
3. 通过lock方法获取监视的shared_ptr，如：
    
    ```cpp
    std::weak_ptr<int> gw;
    void f()
    {
        if(gw.expired()) {
            cout << "gw无效,资源已释放";
        }
        else {
            auto spt = gw.lock(); // 尝试将 std::weak_ptr 提升为 std::shared_ptr。如果关联的 std::shared_ptr 已经释放（引用计数为 0），则返回一个空的 std::shared_ptr。
            cout << "gw有效, *spt = " << *spt << endl;
        }
    }
    
    int main()
    {
        {
            auto sp = atd::make_shared<int>(42);
            gw = sp;
            f();
        }
        f();
        return 0;
    }
    ```
    

### **3.2 weak_ptr返回this指针**

`shared_ptr`不能直接将`this`指针返回`shared_ptr`(会导致多个独立的 `shared_ptr` 实例管理同一个对象)，需要通过派生 `std::enable_shared_from_this`类，并通过其方法`shared_from_this`来返回指针，原因是 `std::enable_shared_from_this`类中有一个`weak_ptr`，这个`weak_ptr`用来观察this智能指针，调用 `shared_from_this()`方法是，会调用内部这个`weak_ptr`的`lock()`方法，将所观察的shared_ptr返回，范例:

```cpp
// weak_ptr_reutrn_this.cpp
#include <iostream>
#include <memory>

using namespace std;

class A: public std::enable_shared_from_this<A>
{
public:
    shared_ptr<A> GetSelf() {
        return shared_from_this(); // 安全：返回与现有 shared_ptr 共享引用计数的 shared_ptr
    }
    ~A()
    {
        cout << "Deconstruction A" << endl;
    }
};

int main()
{
    // auto spp = make_shared<A>();
    shared_ptr<A> sp1(new A);
    shared_ptr<A> sp2 = sp1->GetSelf();  // 安全：sp1 和 sp2 共享引用计数
    //    shared_ptr<A> sp2;
    //    {
    //        shared_ptr<A> sp1(new A);
    //        sp2 = sp1->GetSelf();  // ok
    //    }
    cout << "sp1.use_count() = " << sp1.use_count()<< endl;
    cout << "sp2.use_count() = " << sp2.use_count()<< endl;

    return 0;
}
/*
sp1.use_count() = 2
sp2.use_count() = 2
Deconstruction A
*/
```

现在通过`std::enable_shared_from_this` 这个基类模板，安全地从对象内部获取一个 `shared_ptr`。这样就不会导致一个对象的资源被多个`shared_ptr`释放。
注意：获取自身智能指针的函数要在`shared_ptr`的构造函数被调用之后才能使用，因为 `enable_shared_from_this`内部的`weak_ptr`只有通过`shared_ptr`才能构造。

### 3.3 weak_ptr解决循环引用问题

`shared_ptr`的智能指针循环引用的问题，因为智能指针的循环引用会导致内存泄漏，可以通过 weak_ptr解决该问题，只要将A或B的任意一个成员变量改为weak_ptr。例如:

```cpp
#include <iostream>
#include <memory>
using namespace std;

class A;
class B;

class A {
public:
    std::weak_ptr<B> bptr; // 修改为weak_ptr
    int *val;
    A() {
        val = new int(1);
    }
    ~A() {
        cout << "A is deleted" << endl;
        delete  val;
    }
};

class B {
public:
    std::shared_ptr<A> aptr;
    ~B() {
        cout << "B is deleted" << endl;
    }
};

//weak_ptr 是一种不控制对象生命周期的智能指针,
void test()
{
    std::shared_ptr<A> ap(new A);
    std::weak_ptr<A> wp1 = ap;
    std::weak_ptr<A> wp2 = ap;
    cout<< "ap.use_count()" << ap.use_count()<< endl;
}

void test2()
{
    std::weak_ptr<A> wp;
    {
        std::shared_ptr<A> ap(new A);
        wp = ap;
    }
    cout<< "wp.use_count()" << wp.use_count() << ", wp.expired():" << wp.expired() << endl;
    if(!wp.expired()) {
        // wp不能直接操作对象的成员、方法
        std::shared_ptr<A> ptr = wp.lock(); // 需要先lock获取std::shared_ptr<A>
        *(ptr->val) = 20;  
    }
}

int main()
{
    test2();
//    {
//        std::shared_ptr<A> ap(new A);
//        std::shared_ptr<B> bp(new B);
//        ap->bptr = bp;
//        bp->aptr = ap;
//    }
    cout<< "main leave" << endl;
    return 0;
}
/*
A is deleted
wp.use_count()0, wp.expired():1
main leave
*/
```

这样在对B的成员赋值时，即执行bp->aptr=ap;时，由于aptr是weak_ptr，它并不会增加引用计数，所以ap的引用计数仍然会是1，在离开作用域之后，ap的引用计数为减为0，A指针会被析构，析构后其内部的bptr的引用计数会被减为1，然后在离开作用域后bp引用计数又从1减为0，B对象也被析构，不会发生内存泄漏。

### 3.4 `weak_ptr`的使用注意事项

使用前需要检查是否合法，在使用`weak_ptr`前需要调用wp.expired()函数判断一下。  因为`weak_ptr`还仍旧存在，假如引用计数等于0，仍有某处“全局”性的存储块保存着这个计数信息。直到最后一个`weak_ptr`对象被析构，这块“堆”存储块才能被回收。例如：

```cpp
weak_ptr<int> wp;  // 创建一个空的 weak_ptr
shared_ptr<int> sp_ok;  // 创建一个空的 shared_ptr
{
    shared_ptr<int> sp(new int(1));  // 创建一个 shared_ptr，管理一个 int 对象
    wp = sp;  // 将 weak_ptr 绑定到 shared_ptr
    sp_ok = wp.lock();  // 通过 weak_ptr 获取 shared_ptr
}
if (wp.expired()) {
    cout << "shared_ptr is destroy" << endl;
} else {
    cout << "shared_ptr no destroy" << endl;
}
// shared_ptr no destroy
```

在`sp_ok = wp.lock();`当中由于 `sp` 仍然存在（引用计数为 `1`），`wp.lock()` 成功返回一个 `shared_ptr`，指向 `sp` 管理的对象。此时 `sp_ok` 的引用计数增加到 `2`（`sp` 和 `sp_ok` 共享引用计数）。`}`（作用域结束）后`sp` 超出作用域，它的引用计数递减到 `1`（因为 `sp_ok` 仍然存在）于引用计数仍然为 `1`，`sp` 管理的对象不会被释放。