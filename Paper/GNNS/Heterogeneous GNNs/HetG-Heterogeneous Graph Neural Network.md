> **paper:**Heterogeneous Graph Neural Network
>
> **code:** [chuxuzhang/KDD2019_HetGNN: code of HetGNN (github.com)](https://github.com/chuxuzhang/KDD2019_HetGNN)
>
> **conference:**KDD-2019

本文要**解决的问题**只有一个: 

*给出一个模型$F$,如何让其学到每个节点上embedding（每个节点的特征表示），这个embedding可以充分反映出异质图的结构信息和节点上的非结构信息。*

作者开篇认为之前工作有着三类问题:

* 在不同节点做聚合时，有些节点之间并没有相连，但二者之间的关系十分重要。例如下图中的author和venue，这两者并没有相连，而在传统方式中，包括上述的R-GCN中这两者之间的关系传播依赖于一个至少两层的图神经网络结构来propogation，更远的节点则需要一个更深的结构。这一传播在作者看来可能会削弱原本应该带来的影响。进一步的，对于中心某些连接紧密的节点而言，周围其实并不与其关联度有多少的邻居在聚合时会趋向于变成‘noise node’。

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112151553329.png" alt="image-20211215155358174"  />

* 节点所携带的信息可能是多元的：语音、图像、数值等都有可能，那么如何解决embedding问题也是需要考虑的，本文的算法也定义在一个单个节点可以表示多类信息的异构图上。
* Node Tpye Matter！过去的文章中没有考虑各个节点的类型在标签传播过程中产生的影响。一个典型的例子是上图中对于一个author节点而言，一个paper或author邻居对其产生的影响不可以被以同样的模式处理。

Well，作者是如何解决这三个问题呢？

**首先我们需要知道，GNN的发展中，我们一直希望能找到网络结构的一种优良表述,即如何完成graph to sequence这一过程, 翻译成大白话就是:在标签传播中这一过程中关于邻居的选择问题. 事实上在数年前,并不是一开始就是对于其一阶直接邻居做处理（GAT计算attention，GraphSAGE计算LSTM等），更古老的一种办法就是今天作者使用的：Random Walk, 本质上是shallow embedding的一种.**

作者认为对于异构图而言，这样的办法远胜标签传播，原因详见文章3.1部分开头，主要是因为当前的办法在异构图中不能分辨类别带来的影响，对邻居采样范围也过于限制。

对于每类节点邻居取出现频率最高的n个的，无向异构图的随机游走，可以被描述为：

```
#伪代码
for node in node_list:
   neighbour_of_this_node = [] #所有采样的邻居节点
   selected_node = [] #真正确定的邻居节点
   while (current_collect_number < need_to_collect) and (kinds_in_collection < all_kinds_number)
      walk_around(neighbour_of_this_node)
   if p #以p几率返回初始点
      back_to_init()
   for type in all_node_type:
      node_id = top_k(select_type(neighbour_of_this_node,type),n)
      selected_node.append(node_id)
   save(selected_node)
```

最后可以得到每个节点的邻居，覆盖周围所有可能出现的各个类型的节点，且每一类都取n个。这里一个小问题需要指出的是，在3.4部分作者给出了这样的random walk的另一个小限制，即邻居的选择应该在图中二阶相邻节点以内。

**处理节点上多种特征和节点信息聚合的问题**

对于一个节点上有着多种特征这个问题上，事实上这里的方案十分明确：用各个领域公认较强的特征抽取模型即可，换言之

> 你是图片，那就用CNN结构，你是文本，那就用Par2Vec，你是类别，那就直接one-hot编码，你是音频数据，那就....，如此将各个类型转换为向量表示

在这些特征抽取完成之后，现在在我手上的就是一堆向量，他们来自于单个节点上的不同的data feature。对于这些不同长度的feature，简单的将其输入到一个多层全连接结构的网络中，以获得最后统一长度的向量。在拿到表示各个特征的多个同一维度向量后，那我要如何将其融合呢？

简单，一个双向的RNN（LSTM）就可以。

对于LSTM单元在每一个step的输出，这里使用均值池化的操作以获得最终代表这个节点的特征向量。那么现在对于某一个节点$N$,其每一个邻居节点对应一个特征向量，那我如何来做汇聚以更新中心节点信息呢？先别急，对于这个节点$N$,它由上述Random Walk给出的邻居序列可以被写成 $\left\{N_{a 1}, N_{a 2}, N_{b 1}, N_{c 1} \cdots\right\}$,下标abc表示类别，那么我们对于每一个种类的邻居，根据我们提出的第三类问题，应该做出不同处理。

事实上作者这里依旧是如同做单个节点信息聚合一样的操作，用一个双向的LSTM加均值池化操作完成每一种类的邻居信息聚合，即一个类别邻居上的信息只用一个向量表示。这里需要注意，第二次使用LSTM聚合不同类别节点邻居信息时应该对每一个类别使用一个LSTM。这一过程如下图所示：

![image-20211215155828488](C:\Users\13505\AppData\Roaming\Typora\typora-user-images\image-20211215155828488.png)

这个过程事实上有点绕，作者分别使用了两次聚合，尽管方式均为LSTM加均值池化的操作，但含义并不相同，前者是节点层面信息的聚合，后者是不同种类内部各个邻居的聚合。

那么在获得了每个类别邻居信息的向量后，通过一个简单的attention机制来计算各个节点的重要程度并与自身节点信息聚合以完成一次标签传播。
$$
\mathcal{E}_{v}=\sum_{f_{i} \in \mathcal{F}(v)} \alpha^{v, i} f_{i}
$$
OK, 现在大功即将告成，我们回头来看这两篇文章的训练模块。

首先对图上的问题的常用训练思路做简单总结:

- 节点分类：网络产生的Embedding直接投影到类别数目长度，经过Softmax即可得到评分，这一评分与有label的结点即可计算损失, 或是将embedding另拿出来,使用评估.

> 这里还应该做一个Inductive和Transductive的细分,即测试时是否加入新的节点(训练时未见过的), 说白了就是整个图的结构是否变化了, 多提一句: 事实上当今基于message pass的GNN均支持这两类任务.
>
> 1. Transductive: 即图的结构不变化, test时预测之前在train网络时被故意遮住标签的节点
> 2. Inductive: 即预测新节点, 根据其在过去网络中的空间位置来聚合并获得embedding

- 链接预测：加入不存在的边，以某种方式从节点feature构建出一个score（如两个点上feature相乘获得一个数）来反应该边的存在概率,这一概率与边的label计算损失
- 图分类：以图池化等操作维持输出的embedding为类别长度，再由Softmax后与目标类别计算损失

