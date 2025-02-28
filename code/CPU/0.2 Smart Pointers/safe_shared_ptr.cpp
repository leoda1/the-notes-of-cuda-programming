#include <iostream>
#include <atomic>
#include <mutex>
#include <functional>

template <typename T>
class ThreadSafeSharedPtr {
private:
    T* ptr;                          // 裸指针，指向管理的资源
    std::atomic<int>* ref_count;     // 引用计数，使用原子操作
    std::mutex* mutex;               // 互斥锁，保护关键操作
    std::function<void(T*)> deleter; // 自定义删除器

    // 增加引用计数
    void add_ref() {
        if (ref_count) {
            (*ref_count)++;
        }
    }

    // 减少引用计数，并在引用计数为零时释放资源
    void release() {
        if (ref_count && --(*ref_count) == 0) {
            std::lock_guard<std::mutex> lock(*mutex);
            if (ptr) {
                deleter ? deleter(ptr) : delete ptr;
                ptr = nullptr;
            }
            delete ref_count;
            delete mutex;
        }
    }

public:
    // 默认构造函数
    ThreadSafeSharedPtr() : ptr(nullptr), ref_count(nullptr), mutex(new std::mutex), deleter(nullptr) {}

    // 构造函数，接受裸指针
    explicit ThreadSafeSharedPtr(T* p) : ptr(p), ref_count(new std::atomic<int>(1)), mutex(new std::mutex), deleter(nullptr) {}

    // 构造函数，接受裸指针和自定义删除器
    ThreadSafeSharedPtr(T* p, std::function<void(T*)> d) : ptr(p), ref_count(new std::atomic<int>(1)), mutex(new std::mutex), deleter(d) {}

    // 拷贝构造函数
    ThreadSafeSharedPtr(const ThreadSafeSharedPtr& other) {
        std::lock_guard<std::mutex> lock(*other.mutex);
        ptr = other.ptr;
        ref_count = other.ref_count;
        mutex = other.mutex;
        deleter = other.deleter;
        add_ref();
    }

    // 赋值操作符
    ThreadSafeSharedPtr& operator=(const ThreadSafeSharedPtr& other) {
        if (this != &other) {
            release(); // 释放当前资源
            std::lock_guard<std::mutex> lock(*other.mutex);
            ptr = other.ptr;
            ref_count = other.ref_count;
            mutex = other.mutex;
            deleter = other.deleter;
            add_ref();
        }
        return *this;
    }

    // 移动构造函数
    ThreadSafeSharedPtr(ThreadSafeSharedPtr&& other) noexcept {
        std::lock_guard<std::mutex> lock(*other.mutex);
        ptr = other.ptr;
        ref_count = other.ref_count;
        mutex = other.mutex;
        deleter = other.deleter;
        other.ptr = nullptr;
        other.ref_count = nullptr;
        other.mutex = nullptr;
        other.deleter = nullptr;
    }

    // 移动赋值操作符
    ThreadSafeSharedPtr& operator=(ThreadSafeSharedPtr&& other) noexcept {
        if (this != &other) {
            release(); // 释放当前资源
            std::lock_guard<std::mutex> lock(*other.mutex);
            ptr = other.ptr;
            ref_count = other.ref_count;
            mutex = other.mutex;
            deleter = other.deleter;
            other.ptr = nullptr;
            other.ref_count = nullptr;
            other.mutex = nullptr;
            other.deleter = nullptr;
        }
        return *this;
    }

    // 析构函数
    ~ThreadSafeSharedPtr() {
        release();
    }

    // 获取引用计数
    int use_count() const {
        return ref_count ? ref_count->load() : 0;
    }

    // 解引用操作符
    T& operator*() const {
        return *ptr;
    }

    // 箭头操作符
    T* operator->() const {
        return ptr;
    }

    // 获取裸指针
    T* get() const {
        return ptr;
    }

    // 重置指针
    void reset(T* p = nullptr, std::function<void(T*)> d = nullptr) {
        release();
        ptr = p;
        ref_count = new std::atomic<int>(1);
        mutex = new std::mutex;
        deleter = d;
    }
};

// 测试代码
int main() {
    ThreadSafeSharedPtr<int> p1(new int(100));
    std::cout << "p1 use_count: " << p1.use_count() << std::endl; // 输出 1

    {
        ThreadSafeSharedPtr<int> p2 = p1;
        std::cout << "p1 use_count: " << p1.use_count() << std::endl; // 输出 2
        std::cout << "p2 use_count: " << p2.use_count() << std::endl; // 输出 2
    }

    std::cout << "p1 use_count: " << p1.use_count() << std::endl; // 输出 1

    return 0;
}
