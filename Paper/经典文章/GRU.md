> 不懂先看LSTM

<img src="https://pic2.zhimg.com/v2-49244046a83e30ef2383b94644bf0f31_r.jpg" alt="preview" style="zoom:50%;" />

首先，我们先通过上一个传输下来的状态 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}) 和当前节点的输入 ![[公式]](https://www.zhihu.com/equation?tex=x^t) 来获取两个门控状态。如下图2-2所示，其中 ![[公式]](https://www.zhihu.com/equation?tex=r+) 控制重置的门控（reset gate）， ![[公式]](https://www.zhihu.com/equation?tex=z) 为控制更新的门控（update gate）。

> Tips： ![[公式]](https://www.zhihu.com/equation?tex=\sigma) 为*[sigmoid](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Sigmoid_function)*函数，通过这个函数可以将数据变换为0-1范围内的数值，从而来充当门控信号。



<img src="https://pic3.zhimg.com/80/v2-7fff5d817530dada1b279c7279d73b8a_720w.jpg" alt="img" style="zoom:50%;" />



**与LSTM分明的层次结构不同，下面将对GRU进行一气呵成的介绍~~~ 请大家屏住呼吸，不要眨眼。**

得到门控信号之后，首先使用重置门控来得到**“重置”**之后的数据$\mathbf{h^{t-1'} = h^{t-1} \odot r}$， ![[公式]](https://www.zhihu.com/equation?tex={h^{t-1}}'+%3D+h^{t-1}+\odot+r+) ，再将 ![[公式]](https://www.zhihu.com/equation?tex={h^{t-1}}') 与输入 ![[公式]](https://www.zhihu.com/equation?tex=x^t+) 进行拼接，再通过一个[tanh](https://link.zhihu.com/?target=https%3A//baike.baidu.com/item/tanh)激活函数来将数据放缩到**-1~1**的范围内。即得到如下图2-3所示的 ![[公式]](https://www.zhihu.com/equation?tex=h') 。

<img src="https://pic4.zhimg.com/80/v2-390781506bbebbef799f1a12acd7865b_720w.jpg" alt="img" style="zoom:50%;" />

这里的 ![[公式]](https://www.zhihu.com/equation?tex=h'+) 主要是包含了当前输入的 ![[公式]](https://www.zhihu.com/equation?tex=x^t) 数据。有针对性地对 ![[公式]](https://www.zhihu.com/equation?tex=) 添加到当前的隐藏状态，相当于”记忆了当前时刻的状态“。类似于LSTM的选择记忆阶段（参照我的上一篇文章)。

<img src="https://pic3.zhimg.com/80/v2-5b805241ab36e126c4b06b903f148ffa_720w.jpg" alt="img" style="zoom:50%;" />



> 图2-4中的 ![[公式]](https://www.zhihu.com/equation?tex=\odot) 是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。 ![[公式]](https://www.zhihu.com/equation?tex=\oplus) 则代表进行矩阵加法操作。

------

最后介绍GRU最关键的一个步骤，我们可以称之为**”更新记忆“**阶段。

在这个阶段，我们同时进行了遗忘了记忆两个步骤。我们使用了先前得到的更新门控 ![[公式]](https://www.zhihu.com/equation?tex=z) （update gate）。

**更新表达式**： ![[公式]](https://www.zhihu.com/equation?tex=h^t+%3D+(1-z)+\odot+h^{t-1}+%2B+z\odot+h')

首先再次强调一下，门控信号（这里的 ![[公式]](https://www.zhihu.com/equation?tex=z) ）的范围为0~1。门控信号越接近1，代表”记忆“下来的数据越多；而越接近0则代表”遗忘“的越多。

> 有读者发现在pytorch里面的GRU[[链接](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dgru%23torch.nn.GRU)]写法相比原版对 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}) 多了一个映射，相当于一个GRU变体，猜测是多加多这个映射能让整体实验效果提升较大。如果有了解的同学欢迎评论指出。

GRU很聪明的一点就在于，**我们使用了同一个门控 ![[公式]](https://www.zhihu.com/equation?tex=z) 就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控）**。

- ![[公式]](https://www.zhihu.com/equation?tex=(1-z)+\odot+h^{t-1}) ：表示对原本隐藏状态的选择性“遗忘”。这里的 ![[公式]](https://www.zhihu.com/equation?tex=1-z) 可以想象成遗忘门（forget gate），忘记 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}) 维度中一些不重要的信息。
- ![[公式]](https://www.zhihu.com/equation?tex=z+\odot+h') ： 表示对包含当前节点信息的 ![[公式]](https://www.zhihu.com/equation?tex=h') 进行选择性”记忆“。与上面类似，这里的 ![[公式]](https://www.zhihu.com/equation?tex=(1-z)) 同理会忘记 ![[公式]](https://www.zhihu.com/equation?tex=h+') 维度中的一些不重要的信息。或者，这里我们更应当看做是对 ![[公式]](https://www.zhihu.com/equation?tex=h'+) 维度中的某些信息进行选择。
- ![[公式]](https://www.zhihu.com/equation?tex=h^t+%3D(1-+z)+\odot+h^{t-1}+%2B+z\odot+h') ：结合上述，这一步的操作就是忘记传递下来的 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}+) 中的某些维度信息，并加入当前节点输入的某些维度信息。

> 可以看到，这里的遗忘 ![[公式]](https://www.zhihu.com/equation?tex=z) 和选择 ![[公式]](https://www.zhihu.com/equation?tex=(1-z)) 是联动的。也就是说，对于传递进来的维度信息，我们会进行选择性遗忘，则遗忘了多少权重 （![[公式]](https://www.zhihu.com/equation?tex=z) ），我们就会使用包含当前输入的 ![[公式]](https://www.zhihu.com/equation?tex=h') 中所对应的权重进行弥补 ![[公式]](https://www.zhihu.com/equation?tex=%281-z%29) 。以保持一种”恒定“状态。

------

##  LSTM与GRU的关系

GRU是在2014年提出来的，而LSTM是1997年。他们的提出都是为了解决相似的问题，那么GRU难免会参考LSTM的内部结构。那么他们之间的关系大概是怎么样的呢？这里简单介绍一下。

大家看到 ![[公式]](https://www.zhihu.com/equation?tex=r) (reset gate)实际上与他的名字有点不符。我们仅仅使用它来获得了 ![[公式]](https://www.zhihu.com/equation?tex=h’) 。

那么这里的 ![[公式]](https://www.zhihu.com/equation?tex=h') 实际上可以看成对应于LSTM中的hidden state；上一个节点传下来的 ![[公式]](https://www.zhihu.com/equation?tex=h^{t-1}) 则对应于LSTM中的cell state。1-z对应的则是LSTM中的 ![[公式]](https://www.zhihu.com/equation?tex=z^f) forget gate，那么 z我们似乎就可以看成是选择门 ![[公式]](https://www.zhihu.com/equation?tex=z^i) 了。大家可以结合我的两篇文章来进行观察，这是非常有趣的。



## 总结

GRU输入输出的结构与普通的RNN相似，其中的内部思想与LSTM相似。

与LSTM相比，GRU内部少了一个”门控“，参数比LSTM少，但是却也能够达到与LSTM相当的功能。考虑到硬件的**计算能力**和**时间成本**，因而很多时候我们也就会选择更加”实用“的GRU啦。