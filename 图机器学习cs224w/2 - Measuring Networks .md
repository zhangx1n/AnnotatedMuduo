> 这是CS224W Machine Learning with Graph学习笔记第2篇-Measuring Networks。  
> 本文讲解四个网络度量指标——度分布、路径长度、集群系数和连通分量。

# 网络度量（Measuring Networks）

回忆上一讲，我们可以用节点数、边数、平均度数来描述一个网络（或图），除此之外还有以下四个指标可以用来刻画一个网络的特点：

1. Degree Distribution（度分布）： ![](https://www.zhihu.com/equation?tex=P%28k%29)

2. Path Length（路径长度）： ![](https://www.zhihu.com/equation?tex=h)

3. Clustering Coefficient（集群系数）： ![](https://www.zhihu.com/equation?tex=C)

4. Connected Components（连通分量）：![](https://www.zhihu.com/equation?tex=S)

下面以无向图为例，分别给出四个指标的定义。

## Degree Distribution（度分布）

图中所有节点的**归一化度数的分布直方图就是度分布**，比如下面的例子，图中一共有10个节点，度为1的节点有6个，所以直方图最左侧的柱形的高度是0.6。如果是有向图，可以分别统计入度和出度的分布。

度分布是一个统计指标，可以用来了解节点的度集中分布于哪些值，还可以看看有没有异常值，比如度特别大的节点。

![](https://pic1.zhimg.com/v2-bc13f0f2d1330b663f20163feaa62970_b.jpg)

Path Length（路径长度）
-----------------

**路径**是连接两个节点的全部边的序列，比如下图中，连接A到G的一条路径是 A-C-B-D-E-G。

**路径长度**其实就是最短路径（shortest path），是连接两个节点的最少边数，比如下图中，A到G的路径长度是4（A-C-D-E-G）。如果两个节点不相连，路径长度是无穷大。

如果是有向图，按照箭头方向来连接路径即可。

![](https://pic4.zhimg.com/v2-c861417138b45c94fd923d48065c55eb_b.jpg)

把路径长度推广到整张图，可以得到下面两个指标：

**直径（Diameter）**:所有节点对（node pairs）的路径长度最大值。

**平均路径长度 ![](https://www.zhihu.com/equation?tex=%5Cbar%7Bh%7D)** ：所有节点对（node pairs）的路径长度的平均值。

![](https://pic4.zhimg.com/v2-53c48f436ec3c35d31584ee836ffc38b_b.png)

这里需要注意两点：

（1）上面两个指标都没有考虑不相连的节点对；

（2） ![](https://www.zhihu.com/equation?tex=%5Cbar%7Bh%7D)的分母是 ![](https://www.zhihu.com/equation?tex=E_%7Bmax%7D)，而不是所有相连节点对的数量，因此 ![](https://www.zhihu.com/equation?tex=%5Cbar%7Bh%7D) 并不是严格意义上的算数平均值。

![](https://www.zhihu.com/equation?tex=diam) 和 ![](https://www.zhihu.com/equation?tex=%5Cbar%7Bh%7D)可以反应图的连接模式，比如，一个有大V节点（即 度很大的节点)的图，路径长度会偏小。

Clustering Coefficient（集群系数）
----------------------------

**集群系数用来衡量一个节点的所有邻居节点之间的连接紧密程度**，定义如下图， ![](https://www.zhihu.com/equation?tex=e_%7Bi%7D)是节点 ![](https://www.zhihu.com/equation?tex=i)所有邻居节点（即图中的蓝色点）之间的连边数，

 ![](https://www.zhihu.com/equation?tex=%5Cfrac%7Bk_%7Bi%7D%2A%28k_%7Bi%7D-1%29%7D%7B2%7D)是节点 ![](https://www.zhihu.com/equation?tex=i)所有邻居节点可能的最大连边数，二者相除就是集群系数。

这里存在一个边界问题，如果节点 ![](https://www.zhihu.com/equation?tex=i)只有一个邻居，集群系数为0。

![](https://pic4.zhimg.com/v2-4fcf9247d85d59e3daed760a1ae88013_b.jpg)

把集群系数推广到全图，可以计算**平均集群系数（Average Clustering Coefficient）**，即每个节点的集群系数的平均值。

Connected Components（连通分量）

连通分量的概念在上一讲介绍过，这里所指的度量指标是**最大的连通分量构成的子图，**其目的是为了剔除那些孤立节点，重点关注连通分量。比如下图中的例子，ABCD构成的子图是最大的连通分量，重点研究，FGH是一些散兵游勇，不用理会。

用BFS可以找出所有连通分量。

![](https://pic1.zhimg.com/v2-1ad6c13c23ed3838976b16cace0acffc_b.jpg)

网络度量示例：MSN网络
------------

取MSN中某个月的1.8亿活跃用户（至少发过一条消息的用户），彼此发送过消息则构成连边，构成一张13亿条边的无向图。

这张图的度分布如下图，这是一张log-log图，可见大多数用户分布在度比较小的区间里。

![](https://pic3.zhimg.com/v2-dc4054b3625292913e160cb4a7ad6c2a_b.jpg)

这张图的路径长度分布如下图，平均路径长度是6.6，超过90%的节点对的最短路径长度小于8，这也基本符合大名鼎鼎的“六度分隔”理论。

![](https://pic2.zhimg.com/v2-efe7a0d5c9b0f9638ac2522f96617685_b.jpg)

集群系数随度数的分布如下图，参考图中的绿线，在度比较小的区间内，集群系数与度数有一个近似的指数关系，整张图的平均集群系数是0.114。

![](https://pic3.zhimg.com/v2-91fa77d7e41a647c26e98a4372a8c42a_b.jpg)

连通分量的分布如下图，注意 ![](https://www.zhihu.com/equation?tex=10%5E%7B8%7D)附近的圈圈，超过99.9%的用户构成了一个超级大的连通分量，其余的散兵游勇自顾自地搞出了一些小团体。

![](https://pic1.zhimg.com/v2-b41184424dba43681799ce72f27459a4_b.jpg)

这面这个例子给出了MSN用户社交网络的度量指标，比如 平均集群系数0.114，平均路径长度6.6，如何理解这些指标？最常见的办法是用这些指标与一个基线（Baseline）做对比，图的基线是什么？答案是随机图（Random Graph），这正是下一讲的主题。

> 课程视频： [【斯坦福】CS224W：图机器学习( 中英字幕 | 2019秋)](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1Vg4y1z7Nf%3Ft%3D1391)  
> 课程主页： [CS224W | Home](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224w/)  
> 课程Notes： [Contents](https://link.zhihu.com/?target=https%3A//snap-stanford.github.io/cs224w-notes/)


**_在求知的路上，你永不独行~_**