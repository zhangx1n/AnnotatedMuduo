> Paper: [Wang X, Ji H, Shi C, et al. Heterogeneous graph attention network[C]//The World Wide Web Conference. 2019: 2022-2032.](https://arxiv.org/pdf/1903.07293.pdf)
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

![image-20211216141247584](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112161412810.png)

一个节点在同一种meta-path下可能连接好几个邻居节点，但是这几个节点也应该有不同的权重。

《终结者》将与《泰坦尼克号》和《终结者2》相连，都是由导演詹姆斯·卡梅隆执导。为了更好地确定《终结者》是科幻电影，模型应该更多地关注《终结者2》，而不是《泰坦尼克号》。

### Semantic-level Attention

不同的meta-path应该对应不一样的权重，终结者和终结者2通过电影-演员-电影(都由施瓦辛格主演)或者和《鸟人》通过电影-年份-电影(均摄于1984年)联系。然而，当确定电影的类型时，很明显电影-演员-电影比电影-年份-电影更重要。

## 

![preview](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112161427369.jpeg)
