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
