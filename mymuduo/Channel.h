
#pragma once
#include "noncopyable.h"
#include "Timestamp.h"
#include <functional>
#include <memory>

class EventLoop;

/**
 * Channel 封装了 sockfd 和 对应的感兴趣的event, 如EPOLLIN, EPOLLOUT
 * 所有需要由 EventLoop 处理的如 Acceptor、TcpConnection 都有 Channel 成员并设置 callback 注册到 EventLoop 中。
 * Muduo 的 callback 基本都是 member function， 用 std::bind() 绑定 this 指针来注册，一些网络库会采用继承接口类来实现回调的注册。
 */
class Channel: noncopyable {
public:
    using EventCallback = std::function<void()>;
    using ReadEventCallback = std::function<void(Timestamp)>;

    Channel(EventLoop *loop, int fd);
    ~Channel();

    void handleEvent(Timestamp receiveTime);

    // 设置回调对象
    void setReadCallback(ReadEventCallback cb) {readCallback_ = std::move(cb);}
    void setWriteCallback(EventCallback cb) {writeCallback_  =std::move(cb);}
    void setCloseCallback(EventCallback cb) {closeCallback_ = std::move(cb);}
    void setErrorCallback(EventCallback cb) {errorCallback_ = std::move(cb);}

    // 防止当Channel被手动move掉, Channel还在执行回调操作. 用weakptr来监控
    void tie(const std::shared_ptr<void>&);

    int fd() const {return fd_;}
    int events() const {return events_; }
    void set_revents(int revt) {revents_ = revt;}
    // fd 到底有没有设置感兴趣的事件
    bool isNoneEvent() const {return events_ == kNoneEvent; }


// 设置fd相应的事件状态
    void enableReading() {events_ |= kReadEvent; update(); }
    void disableReading() {events_ &= ~kReadEvent; update(); }
    void enableWriting() {events_ |= kWriteEvent; update(); }
    void disableWriting() {events_ &= ~kWriteEvent; update(); }
    void disableAll() {events_ = kNoneEvent; update(); }

    bool isWriting() const {return events_ & kWriteEvent; }
    bool isReading() const {return events_ & kReadEvent; }

    EventLoop* ownerLoop() {return loop_; }
    void remove();

    int index() { return index_; }
    void set_index(int idx) { index_ = idx; }

private:
    static const int kNoneEvent;
    static const int kReadEvent;
    static const int kWriteEvent;

    EventLoop *loop_;    // 事件循环
    const int fd_;  // fd, Poller监听的对象
    int events_;    // events_ 是 Channel 关心的事件，Poller 根据这个来设置
    int revents_;   // revents_ 是 Poller 返回的已就绪的事件，handleEvent() 会调用相应的 callback 来处理。
    int index_; // 因为 epoll(2) 是记录关注的 fd 和事件的，所以不需要重复传递兴趣列表，
    // 但是区分了 EPOLL_CTL_ADD 和 EPOLL_CTL_MOD，所以 Channel 用 index_ 记录了状态是要 add 还是 mod。

    std::weak_ptr<void> tie_;
    bool tied_;

    ReadEventCallback readCallback_;
    EventCallback writeCallback_;
    EventCallback closeCallback_;
    EventCallback errorCallback_;

    void update();
    void handleEventWithGuard(Timestamp receiveTime);
};
