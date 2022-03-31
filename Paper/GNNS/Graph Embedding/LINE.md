> [LINE: Lare-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578)

## 概要

LINE： 大规模的图上，表示节点之间的结构信息

**一阶相似性**：局部的结构信息

**二阶相似性**：节点的邻居。共享邻居的节点可能是相似的。

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111282304472.png)

Figure 1: A toy example of information network. Edges canbe undirected, directed, and/or weighted. Vertex 6 and 7 should be placed closely in the low-dimensional space as they are connected through a strong tie. Vertex 5 and 6 should also be placed closely as they share similar neighbors.



**DeepWalk在无向图上，LINE在有向图和加权图上都可以使用**



即使找到了一个合理的目标，针对超大型网络进行优化也是具有挑战性的。近年来备受关注的一种方法是利用随机梯度梯度法进行优化。然而，我们表明，对于现实世界的信息网络，直接部署随机梯度下降是有问题的。这是因为在许多网络中，边是加权的，并且权重通常表现出很高的方差。考虑一个单词共现网络，在该网络中，单词对的权重(共现)可能从1到数十万不等。这些边的权重将乘以梯度，导致梯度爆炸，从而影响性能。针对这一问题，我们提出了一种新的边缘采样方法（edge-sampling method），提高了推理的有效性和效率。我们用与其权重成正比的概率对边进行采样，然后将采样后的边作为binary edges进行模型更新。



**LINE的问题**：1. *Low degree vertives* 上表现不好，度比较少的话就效果没有那么好

## 细节

![image-20211129090725847](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290907379.png)

 

![image-20211129091126822](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290911144.png)



**最终的embedding：**一阶相似性和二阶相似性拼接在一起

## 实验结果

![image-20211129092106172](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290921225.png)

![image-20211129092117395](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290921434.png)

![image-20211129092131615](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290921671.png)

![image-20211129092151259](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202111290921315.png)