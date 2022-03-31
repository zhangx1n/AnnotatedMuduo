![preview](https://pic4.zhimg.com/v2-f716c816d46792b867a6815c278f11cb_r.jpg)

## LSTM

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。



LSTM结构（图右）和普通RNN的主要输入输出区别如下所示。

![preview](https://pic4.zhimg.com/v2-e4f9851cad426dfe4ab1c76209546827_r.jpg)

RNN中的$h^t$对应LSTM中的$c^t$

首先使用LSTM的当前输入 ![[公式]](https://www.zhihu.com/equation?tex=x^t) 和上一个状态传递下来的 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}) 拼接训练得到四个状态。

<img src="https://pic4.zhimg.com/v2-15c5eb554f843ec492579c6d87e1497b_r.jpg" alt="preview" style="zoom:50%;" />

<img src="https://pic1.zhimg.com/v2-d044fd0087e1df5d2a1089b441db9970_r.jpg" alt="preview" style="zoom:50%;" />

**下面开始进一步介绍这四个状态在LSTM内部的使用。**

![preview](https://pic2.zhimg.com/v2-556c74f0e025a47fea05dc0f76ea775d_r.jpg)

![[公式]](https://www.zhihu.com/equation?tex=\odot) 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 ![[公式]](https://www.zhihu.com/equation?tex=\oplus) 则代表进行矩阵加法。



LSTM内部主要有三个阶段：

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。

    具体来说是通过计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=z^f) （f表示forget）来作为忘记门控，来控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c^{t-1}) 哪些需要留哪些需要忘。

2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入 ![[公式]](https://www.zhihu.com/equation?tex=x^t) 进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些。当前的输入内容由前面计算得到的 ![[公式]](https://www.zhihu.com/equation?tex=z+) 表示。而选择的门控信号则是由 ![[公式]](https://www.zhihu.com/equation?tex=z^i) （i代表information)来进行控制。

> 将上面两步得到的结果相加，即可得到传输给下一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c^t) 。也就是上图中的第一个公式。

3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过 ![[公式]](https://www.zhihu.com/equation?tex=z^o) 来进行控制的。并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c^o) 进行了放缩（通过一个tanh激活函数进行变化)。
3. 与普通RNN类似，输出 ![[公式]](https://www.zhihu.com/equation?tex=y^t) 往往最终也是通过 ![[公式]](https://www.zhihu.com/equation?tex=h^t) 变化得到。