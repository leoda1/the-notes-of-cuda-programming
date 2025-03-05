#include <iostream>
using namespace std;


// // 这里出现了两次析构 double free，因为默认拷贝构造函数是浅拷贝构造的原因
// class A {
// public:
//     A() : m_ptr(new int (42)) {// 分配堆内存
//         cout << "constructor A" << endl;
//     }
//     ~A() {
//         cout << "destructor A, m_ptr:" << m_ptr << endl;
//         delete m_ptr;// 释放堆内存
//         m_ptr = nullptr;
//     } 
// private:
//     int *m_ptr;
// };

// A GetA(bool f) {
//     A a;
//     A b;
//     cout << "ready return" << endl;
//     if (f) return a;
//     else return b;
// }

// int main() {
//     A a = GetA(false); // 运行报错, a 接收返回值，拷贝构造触发
//     return 0;
// }
// /******************************************************************
// constructor A
// constructor A
// ready return
// destructor A, m_ptr:0x55ce0bd692e0
// destructor A, m_ptr:0x55ce0bd68eb0
// destructor A, m_ptr:0x55ce0bd692e0
// free(): double free detected in tcache 2
// Aborted (core dumped)
//  *****************************************************************/


// // 自定义了深拷贝构造函数来解决double free的情况
// class A {
// private:
//     int * m_ptr;

// public:
//     A () : m_ptr(new int (42)) {cout << "construction A" << endl;}
//     A (const A& a) : m_ptr(new int(*a.m_ptr)) { // 只有这里有所不同
//         cout << "copy construction A, new m_ptr: " << m_ptr << endl;
//     }  // A (const A& a)是 拷贝构造函数的声明，表示用已有的A对象a来创建新的A对象。
//     ~A () {
//         cout << "destructor A, m_Ptr:" << m_ptr << endl;
//         delete m_ptr;
//         m_ptr = nullptr; 
//     }
// };

// A GetA(bool f) {
//     A a;
//     A b;
//     cout << "ready return" << endl;
//     if (f) return a;
//     else return b;
// }

// int main () {
//     A a = GetA(false);
//     return 0;
// }
// /******************************************************************
// construction A
// construction A
// ready return
// copy construction A, new m_ptr: 0x557bb1110300
// destructor A, m_Ptr:0x557bb11102e0
// destructor A, m_Ptr:0x557bb110feb0
// destructor A, m_Ptr:0x557bb1110300
//  *****************************************************************/


// // 移动构造避免对临时对象的深拷贝
class A {
private:
    int * m_ptr;

public:
    A () : m_ptr(new int (42)) {cout << "construction A" << endl;}
    A (A && a) : m_ptr(a.m_ptr) {
        a.m_ptr = nullptr;
        cout << "move constructor A" << endl;
    }
    ~A () {
        cout << "destructor A, m_Ptr:" << m_ptr << endl;
        delete m_ptr;
        m_ptr = nullptr; 
    }
};

A GetA(bool f) {
    A a;
    A b;
    cout << "ready return" << endl;
    if (f) return a;
    else return b;
}

int main () {
    A a = GetA(false);
    return 0;
}
/******************************************************************
construction A
construction A
ready return
move constructor A
destructor A, m_Ptr:0
destructor A, m_Ptr:0x5566481b1eb0
destructor A, m_Ptr:0x5566481b22e0
 *****************************************************************/