/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/22 23:04
* @version: 1.0
* @description: 
********************************************************************************/
#include "Channel.h"
#include "EventLoop.h"
#include "Logger.h"
#include <sys/epoll.h>

const int Channel::kNoneEvent = 0;
const int Channel::kReadEvent = EPOLLIN | EPOLLPRI;
const int Channel::kWriteEvent = EPOLLOUT;


Channel::Channel(EventLoop *loop, int fd)
        : loop_(loop), fd_(fd), events_(0), revents_(0), index_(-1), tied_(false) {

}

Channel::~Channel() {

}

void Channel::handleEvent(Timestamp receiveTime) {
    std::shared_ptr<void> guard;
    if (tied_) {    // 如果tied_已经绑定了一个loop_
        guard = tie_.lock();
        if (guard) {
            handleEventWithGuard(receiveTime);
        }
    } else {
        handleEventWithGuard(receiveTime);
    }
}

// 根据poller通知的channel发生的具体事件， 由channel负责调用具体的回调操作
void Channel::handleEventWithGuard(Timestamp receiveTime)
{
    LOG_INFO("channel handleEvent revents:%d\n", revents_);

    if ((revents_ & EPOLLHUP) && !(revents_ & EPOLLIN))
    {
        if (closeCallback_)
        {
            closeCallback_();
        }
    }

    if (revents_ & EPOLLERR)
    {
        if (errorCallback_)
        {
            errorCallback_();
        }
    }

    if (revents_ & (EPOLLIN | EPOLLPRI))
    {
        if (readCallback_)
        {
            readCallback_(receiveTime);
        }
    }

    if (revents_ & EPOLLOUT)
    {
        if (writeCallback_)
        {
            writeCallback_();
        }
    }
}


void Channel::tie(const std::shared_ptr<void> &obj) {
    tie_ = obj; //tie_是弱智能指针
    tied_ = true;
}

/**
 * 当改变channel所表示fd的events事件后,update负责在Poller里面改变fd相应的事件epoll_ctl.
 */
void Channel::update() {
    loop_->updateChannel(this);
}
void Channel::remove() {
    loop_->removeChannel(this);
}
