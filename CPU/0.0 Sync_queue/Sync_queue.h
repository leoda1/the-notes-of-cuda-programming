#ifndef __SYNC_QUEUE_HPP__
#define __SYNC_QUEUE_HPP__

#include<list>
#include<mutex>
#include<thread>
#include<condition_variable>
#include <iostream>

template<typename T>
class SyncQueue
{
private:
    bool IsFull() const
    {
        return _queue.size() == _maxSize;
    }
    bool IsEmpty() const
    {
        return _queue.empty();
    }
public:
    SyncQueue(int maxSize) : _maxSize(maxSize)
    {
    }
    void Put(const T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        // _notFull.wait(_mutex, [this] {return !IsFull();});
        while (IsFull())
        {
            std::cout << "full wait..." << std::endl;
            _notFull.wait(_mutex);
        }
        _queue.push_back(x);
        _notFull.notify_one();
    }
    void Take(T& x)
    {
        std::lock_guard<std::mutex> locker(_mutex);
        //如果只有一个任务，但是唤醒了多个消费者线程，
        //则需要消费者线程wait后判断队列是不是空的，解决方法就是将if empty改为while empty
        while (IsEmpty())
        {
            std::cout << "empty wait.." << std::endl;
            _notEmpty.wait(_mutex);
        }
        x = _queue.front();
        _queue.pop_front();
        _notFull.notify_one();
    }
    bool Empty()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.empty();
    }
    bool Full()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size() == _maxSize;
    }
    size_t Size()
    {
        std::lock_guard<std::mutex> locker(_mutex);
        return _queue.size();
    }
    int Count()
    {
        return _queue.size();
    }
private:
    std::list<T> _queue; //缓冲区
    std::mutex _mutex; //互斥量和条件变量结合起来使用
    std::condition_variable_any _notEmpty;//不为空的条件变量
    std::condition_variable_any _notFull; //没有满的条件变量
    int _maxSize; //同步队列最大的size
};

#endif // __SYNC_QUEUE_HPP__