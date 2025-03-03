#include <iostream>
#include <memory>

using namespace std;

class A: public std::enable_shared_from_this<A>
{
public:
    shared_ptr<A> GetSelf()
    {
        return shared_from_this(); 
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