/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/24 16:37
* @version: 1.0
* @description: 一个Thread对象, 记录的就是一个新线程的详细信息
********************************************************************************/
#include "Thread.h"
#include "CurrentThread.h"
#include <semaphore.h>

#include <memory>

std::atomic_int32_t Thread::numCreated_{0};


Thread::Thread(Thread::ThreadFunc func, const std::string &name) :
        joined_(false), started_(false), tid_(0), func_(std::move(func)), name_(name) {
    setDefaultName();
}

Thread::~Thread() {
    if (started_ && !joined_) {
        thread_->detach();
    }
}

void Thread::start() {
    sem_t sem;
    sem_init(&sem, false, 0);
    started_ = true;
    thread_ = std::make_shared<std::thread>([&]() {
        sem_post(&sem);
        tid_ = CurrentThread::tid();
        func_();
    });

    // 这里必须等待上面的tid生成了才能拿到.
    sem_wait(&sem);
}

void Thread::join() {
    joined_ = true;
    thread_->join();
}

void Thread::setDefaultName() {
    int num = ++numCreated_;
    if (name_.empty()) {
        char buf[32] = {0};
        snprintf(buf, sizeof buf, "Thread%d", num);
        name_ = buf;
    }
}
