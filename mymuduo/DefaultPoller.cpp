/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/23 14:20
* @version: 1.0
* @description: 
********************************************************************************/
#include "Poller.h"
#include "EpollPoller.h"
#include <cstdlib>

Poller *Poller::newDefaultPoller(EventLoop *loop) {
    if (::getenv("MUDUO_USE_POOL")) {
        return nullptr;
    } else {
        return new EPollPoller(loop);
    }
}