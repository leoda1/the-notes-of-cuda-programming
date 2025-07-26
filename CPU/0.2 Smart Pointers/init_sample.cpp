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
/******************************************************************
(base) joker@liuda:~/projects/cudalearn/notes/code/CPU/0.2 Smart Pointers$ ./init_sample 
sp1.use_count() =1
sp2.use_count() =1
sp3.use_count() =2
sp4.use_count() =1
p2.use_count() = 2
p1.use_count() = 2
p2.use_count() = 1
p1.use_count() = 0
 *****************************************************************/