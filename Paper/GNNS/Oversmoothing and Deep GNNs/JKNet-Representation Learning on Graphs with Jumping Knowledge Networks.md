> [Representation Learning on Graphs with Jumping Knowledge Networks](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf)
>
> **PMLR 2018**
>
> **Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka**
>
> **代码：**https://github.com/mori97/JKNet-dgl

## 背景介绍

GCN对所有的节点“一视同仁”，如果你设的是K层网络，那么图中每个节点都会用第K跳聚合到的信息，但坏处是**你无法再获得第1跳到第K-1跳的任何一跳的聚合信息**。在GCN论文中有提到，网络最好的效果是在第2跳左右，随着网络层数的加深，会出现过度平滑(over-smoth)的问题。

> **什么是图神经网络的过渡平滑问题？**
> 在图神经网络的训练过程中，随着网络层数的增加和迭代次数的增加，同一连通分量内的节点的表征会趋向于收敛到同一个值（即空间上的同一个位置）

## 解决方法

为了解决上述问题，本文提出了Jumping Knowledge Networks

本文首先用一个小实验说明，**图中不同位置的结构严重影响了每个点所能够获得的信息范围大小**，也就是感受野的大小，因此不同的节点需要的信息程度是不一样的。

![image-20211130101635844](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301016182.png)

由于在图中每个节点的结构不尽相同，对于稠密的节点，从邻居传递过来的信息本来就很多，不需要经过多次的卷积迭代，也就是说稠密节点感受野较小时就能捕获到足够多它需要的信息，否则过度聚合信息反而会造成有用信息的丢失，而对于稀疏的节点，它需要获得更多全局的信息，也就是说需要更多跳数才能获得足够的聚合信息，这样可以使它的信息表达更加丰富和准确。

## 核心思想

**既然每个节点需要的信息程度不同，那我们就让模型自己去学习需要哪部分信息**。

本文提出了一种网络架构来实现上述的想法。

![image-20211130101806848](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301018914.png)

我们将每一层（每跳）的结果拼接到最后的一层，这样模型可以自己选择每个节点需要哪几跳的信息。比如，对于稠密节点，可能只需要第2跳的信息就够了，那模型就会关注第2跳的信息表达，而忽略其他跳（层）的信息表达，而如果用了第4跳的信息反而容易出现over-smoth的问题；对于稀疏的节点，他可能需要更多跳（更高层）才能获得足够的聚合信息，比如他需要第3跳和第4跳的信息。

简单来说，模型自己学习每个结点需要哪几跳的信息，这样对于不同的节点，就能自适应的聚合不同跳的信息。你这个节点想要第几跳，那就去用第几跳的信息。

顺便提一下，最后一层不同层的representation的整合策略有很多种，具体哪种整合方式好，要看具体实验结果，论文里给了三种方式 concatenation, max-pooling和LSTM-attention，并且做了实验简单对比下效果。


![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301018061.png)

## 总结

Jumping Knowledge Networks中的jump的意思是每一层的representation 并不是只输出到下一层网络，而且还直接jump到最后一层网络。不同receptive field所捕获的信息全都送到最后一层网络，这样，需要给哪个节点多大的receptive field就可以通过模型训练来确定了。


## 文中总结了几种Neighborhood aggregation 范式

### Graph Convolutional Networks (GCN)

![image-20211130105749145](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301057181.png)

这就是 普通的GCN（Kipf & Welling，2017）

### Neighborhood Aggregation with Skip Connections

![image-20211130105934415](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301059454.png)

**GraphSAGE** (Hamilton et al., 2017) uses concatenation after a feature transform. **Column Networks** (Pham et al., 2017) interpolate the neighborhood representation and the node’s previous representation, and **Gated GNN** (Li et al., 2016) uses the Gated Recurrent Unit (GRU) (Cho et al., 2014). Another wellknown variant of skip connections, **residual connections**, use the identity mapping to help signals propagate (He et al., 2016a;b)

但是这样只是针对输入单元的skip，每次都combine了，导致最终的output还是包含了每个k-hop的信息。因此，skip-connection不能适应性地调整最终层的邻域大小。

### Neighborhood Aggregation with Directional Biases

不是均等的处理邻接节点的特征，而是weigh "important" neighbors more.

**Graph Attention Networks (GAT)** (Velickovic et al. , 2018) and **VAIN** (Hoshen, 2017) learn to select the important neighbors via an attention mechanism. **The max-pooling operation in GraphSAGE** (Hamilton et al., 2017) implicitly selects the important nodes. 

## Influence score and distribution

![image-20211130154439619](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111301544722.png)