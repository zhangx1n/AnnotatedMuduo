/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/24 22:00
* @version: 1.0
* @description: eventloopthreadpool是基于muduo库中Tcpserver这个类专门
* 做的一个线程池，它的模式属于半同步半异步，线程池中每一个线程都有一个
* 自己的eventloop，而每一个eventloop底层都是一个poll或者epoll，它利用了
* 自身的poll或者epoll在没有事件的时候阻塞住，在有事件发生的时候，epoll
* 监听到了事件就会去处理事件。
********************************************************************************/
#include "EventLoopThreadPool.h"
#include "EventLoopThread.h"

EventLoopThreadPool::EventLoopThreadPool(EventLoop *baseLoop, const std::string &name)
        : baseLoop_(baseLoop), name_(name), started_(false), numThreads_(0), next_(0) {

}

EventLoopThreadPool::~EventLoopThreadPool() {

}

void EventLoopThreadPool::start(const EventLoopThreadPool::ThreadInitCallback &cb) {
    started_ = true;
    if (numThreads_ == 0 && cb) {
        cb(baseLoop_);
    }
    for (int i = 0; i < numThreads_; ++i) {
        char buf[name_.size() + 32];
        snprintf(buf, sizeof buf, "%s%d", name_.c_str(), i);
        auto *t = new EventLoopThread(cb, buf);
        threads_.push_back(std::unique_ptr<EventLoopThread>(t));
        loops_.push_back(t->startLoop());
    }
}

EventLoop *EventLoopThreadPool::getNextLoop() {
    EventLoop *loop = baseLoop_;
 /************************************************************************
  轮询调度算法的原理是每一次把来自用户的请求轮流分配给内部中的服务器，从1
  开始，直到N(内部服务器个数)，然后重新开始循环。轮询调度算法假设所有服务器
  的处理性能都相同，不关心每台服务器的当前连接数和响应速度。当请求服务间隔
  时间变化比较大时，轮询调度算法容易导致服务器间的负载不平衡。
  *************************************************************************/
    if (!loops_.empty()) {
        loop = loops_[next_];
        ++next_;
        if (next_ >= loops_.size()) {
            next_ = 0;
        }
    }
    return loop;
}

std::vector<EventLoop *> EventLoopThreadPool::getAllLoops() {
    if (loops_.empty()) {
        return std::vector<EventLoop *>();
    } else {
        return loops_;
    }
}
