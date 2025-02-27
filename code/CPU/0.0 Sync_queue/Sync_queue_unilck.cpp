#include "Sync_queue_unilck.h"
using namespace std;

SimpleSyncQueue<int> syncQueue;
void PutDatas()
{
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Put(888);
    }
}
void TakeDatas()
{
    int x = 0;
    for (int i = 0; i < 20; ++i)
    {
        syncQueue.Take(x);
        std::cout << x << std::endl;
    }
}
int main(void)
{
    thread t1(PutDatas);
    thread t2(TakeDatas);
    t1.join();
    t2.join();
    cout << "main finish\n" << endl;
    return 0;
}
