#pragma once

#include "Poller.h"
#include "Timestamp.h"

#include <vector>
#include <sys/epoll.h>

class Channel;

/**
 *1.这个类主要利用epoll函数，封装了epoll三个函数，
 *2.其中epoll_event.data是一个指向channel类的指针
 *这里可以等价理解为channel就是epoll_event，用于在epoll队列中注册，删除，更改的结构体
 *因为文件描述符fd，Channel，以及epoll_event结构体（只有需要添加到epoll上时才有epoll_event结构体）
 *三个都是一一对应的关系Channel.fd应该等于fd，epoll_event.data应该等于&Channel
 *如果不添加到epoll队列中，Channel和fd一一对应，就没有epoll_event结构体了
 *3.从epoll队列中删除有两种删除方法，
 *第一种暂时删除，就是从epoll队列中删除，并且把标志位置为kDeleted，但是并不从ChannelMap channels_中删除
 *第二种是完全删除，从epoll队列中删除，并且从ChannelMap channels_中也删除，最后把标志位置kNew
 *可以理解为ChannelMap channels_的作用就是：暂时不需要的，就从epoll队列中删除，但是在channels_中保留信息，类似与挂起，这样
 *下次再使用这个channel时，只需要添加到epoll队列中即可。而完全删除，就把channels_中也删除。
 */
class EPollPoller : public Poller
{
public:
    EPollPoller(EventLoop *loop);
    ~EPollPoller() override;

    // 重写基类Poller的抽象方法
    Timestamp poll(int timeoutMs, ChannelList *activeChannels) override;
    void updateChannel(Channel *channel) override;
    void removeChannel(Channel *channel) override;
private:
    static const int kInitEventListSize = 16;

    // 填写活跃的连接
    void fillActiveChannels(int numEvents, ChannelList *activeChannels) const;
    // 更新channel通道
    void update(int operation, Channel *channel);

    using EventList = std::vector<epoll_event>;

    int epollfd_;
    EventList events_;
};