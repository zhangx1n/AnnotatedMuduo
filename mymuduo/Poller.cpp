/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/23 14:04
* @version: 1.0
* @description: 
********************************************************************************/
#include "Poller.h"
#include "Channel.h"

Poller::Poller(EventLoop *loop)
        : ownerLoop_(loop)
{
}

bool Poller::hasChannel(Channel *channel) const
{
    auto it = channels_.find(channel->fd());
    return it != channels_.end() && it->second == channel;
}


