#ifndef __SYNC_QUEUE_UNILCK_HPP__
#define __SYNC_QUEUE_UNILCK_HPP__

#include <thread>
#include <condition_variable>
#include <mutex>
#include <list>
#include <iostream>
using namespace std;

template<typename T>
class SimpleSyncQueue
{
public:
    SimpleSyncQueue(){}
    void Put(const T& x)
    {
        lock_guard<mutex> locker(_mutex);
        _queue.push_back(x);
        _notEmpty.notify_one();
    }
    void Take(T& x)
    {
        unique_lock<mutex> locker(_mutex);
        _notEmpty.wait(locker, [this]{return !_queue.empty(); });
        x = _queue.front();
        _queue.pop_front();
    }
    bool Empty()
    {
        lock_guard<mutex> locker(_mutex);
        return _queue.empty();
    }
    size_t Size()
    {
        lock_guard<mutex> locker(_mutex);
        return _queue.size();
    }
private:
    list<T> _queue;
    mutex _mutex;
    condition_variable _notEmpty;
};

#endif // __SYNC_QUEUE_UNILCK_HPP__