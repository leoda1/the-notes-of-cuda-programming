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
<<<<<<< HEAD
}
=======
}

/******************************************************************
use reset init and call lambda3 delete p41
call lambda2 delete p3
call lambda1 delete p2
call Deleter delete p1
 *****************************************************************/
>>>>>>> 837092c (1)
