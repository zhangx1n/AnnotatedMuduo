> Tilte: [Wang X, Ji H, Shi C, et al. Heterogeneous graph attention network[C]//The World Wide Web Conference. 2019: 2022-2032.](https://arxiv.org/pdf/1903.07293.pdf)
>
> Author: Xiao Wang,Houye Ji, Chuan Shi, Bai Wang (Beijing University),  Peng Cui, P. Yu (Tsing hua University)
>
> Conference：WWW2019

## Abstract

在之前介绍的GraphSAGE和GAT都是针对同构图（homogeneous graph）的模型，它们也确实取得了不错的效果。然而在很多场景中图并不总是同构的，例如图可能有种类型的节点或者节点之间拥有不同类型的连接方式，这种图叫做异构图（heterogeneous graph）。因为异构图含有了更多的信息，往往也比同构图有更好的表现效果。这里介绍的Heterogeneous Graph Attention Network（HAN）便是经典的异构图模型，它的思想是不同类型的边应该有不同的权值，而在同一个类型的边中，不同的邻居节点又应该有不同的权值，因此它使用了节点级别的注意力（node level attention）和语义级别的注意力（semantic level attention）。其中语义级别的attention用于学习中心节点与其不同类型的邻居节点之间的重要性，语义级别的attention用于学习不同meta-path的重要性。HAN也是一个可以归纳的模型。

## Model

模型具有一个层级注意力结构(hierarchical attention structure): node-level attention --> semantic-level attention. node-level attention 学习基于meta-path邻居的权重，然后aggregate他们到semantic-specific node embedding. 然后HAN通过语义级别的注意力来分辨元路径的不同，并获得最优的权重合并semantic-specific node embedding。

![image-20211103223738369](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111032237611.png)

### Node-level Attention

由于是异构图，不同的节点有不同的属性特征，首先要给节点的特征的维度变成一样的，可以通过mlp等。$\mathbf{h}_{i}^{\prime}=\mathbf{M}_{\phi_{i}} \cdot \mathbf{h}_{i}$，$M$: transformation matrix ， $\mathbf{h}_{i}^{\prime}$: 新的节点特征，维度相同

然后用self-attention学习两个节点（这两个节点是通过mate-path$\Phi$连接的）间的注意力。$e_{i j}^{\Phi}=a t t_{n o d e}\left(\mathbf{h}_{i}^{\prime}, \mathbf{h}_{j}^{\prime} ; \Phi\right)$,

### Semantic-level Attention

