/********************************************************************************
* @author: Xin Zhang
* @email: sjhpzx@gmail.com
* @date: 2023/5/25 10:04
* @version: 1.0
* @description: 
********************************************************************************/
#include "Buffer.h"

#include <errno.h>
#include <sys/uio.h>
#include <unistd.h>

/**
 * 从fd上读取数据  Poller工作在LT模式
 * Buffer缓冲区是有大小的！ 但是从fd上读数据的时候，却不知道tcp数据最终的大小
 * 从socket读到缓冲区的方法是使用readv先读至Buffer_，
 * Buffer_空间如果不够会读入到栈上65536个字节大小的空间，然后以append的
 * 方式追加入Buffer_。既考虑了避免系统调用带来开销，又不影响数据的接收。
 */
ssize_t Buffer::readFd(int fd, int* saveErrno)
{
    char extrabuf[65536] = {0}; // 栈上的内存空间  64K

    struct iovec vec[2];

    const size_t writable = writableBytes(); // Buffer底层缓冲区剩余的可写空间大小
    vec[0].iov_base = begin() + writerIndex_;
    vec[0].iov_len = writable;

    vec[1].iov_base = extrabuf;
    vec[1].iov_len = sizeof extrabuf;

    const int iovcnt = (writable < sizeof extrabuf) ? 2 : 1;
    // readv和writev函数的功能可以概括为：对数据进行整合传输以及发送。
    // 通过writev函数可以将分散保存在多个buff的数据一并进行发送，通过readv可以由多个buff分别接受数据，
    // 适当的使用这两个函数可以减少I/O函数的调用次数。
    const ssize_t n = ::readv(fd, vec, iovcnt);
    if (n < 0)
    {
        *saveErrno = errno;
    }
    else if (n <= writable) // Buffer的可写缓冲区已经够存储读出来的数据了
    {
        writerIndex_ += n;
    }
    else // extrabuf里面也写入了数据
    {
        writerIndex_ = buffer_.size();
        append(extrabuf, n - writable);  // writerIndex_开始写 n - writable大小的数据
    }

    return n;
}

ssize_t Buffer::writeFd(int fd, int* saveErrno)
{
    ssize_t n = ::write(fd, peek(), readableBytes());
    if (n < 0)
    {
        *saveErrno = errno;
    }
    return n;
}
