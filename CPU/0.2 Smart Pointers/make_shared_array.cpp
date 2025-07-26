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