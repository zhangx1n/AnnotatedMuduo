> 论文：Graph Convolutional Matrix Completion （GCMC） 图卷积矩阵补全
>
> 作者：来自于荷兰阿姆斯特丹大学的Rianne van den Berg, Thomas N. Kipf（GCN的作者）, Max Welling
>
> 来源：KDD 2018 Workshop
>
> 论文链接：https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf
>
> Github链接：https://github.com/riannevdberg/gc-mc

------

## 简介

基于user-item的二部图，提出了一种**图自编码器框架**，从链路预测的角度解决推荐系统中的评分预测问题。此外，为了验证所提出的消息传递方案，在标准的协同过滤任务上测试了所提出的模型，并展示出了一个有竞争力的结果。

推荐系统的一个子任务就是**矩阵补全**。文中把矩阵补全**视作在图上的链路预测**问题：users和items的交互数据可以通过一个在user和item节点之间的二部图来表示，其中观测到的评分/购买用links来表示。因此，预测评分就相当于预测在这个user-item二分图中的links。

作者提出了一个**图卷积矩阵补全（GCMC）框架**：在用深度学习处理图结构的数据的研究进展的基础上，对矩阵进行补全的一种图自编码器框架。这个自编码器通过在二部交互图中信息传递的形式生成user和item之间的隐含特征。这种user和item之间的隐含表示用于通过一个双线性的解码器重建评分links。



## 相关介绍

> reference：[ GCMC - Graph Convolutional Matrix Completion 图卷积矩阵补全 KDD 2018](https://blog.csdn.net/yyl424525/article/details/102747805)

## 在二部图中矩阵补全作为一种链接预测

![image-20220103165537957](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201031709963.png)

### Revisiting graph auto-encoders 图自编码器

![image-20220103204629331](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032046420.png)

* $[Z_u, Z_v] = f(X_u, X_v,M_1,...,M_r)$, 其中 $M_r∈\{0,1\}^{N_u×N_v}$表示评分等级类型r相关的邻接矩阵（此时为元素是1或0的矩阵，元素为1表示观测到评分等级类型，元素为0就表示没有观测到评分的）

* $Z_u$是user的embedding矩阵

* $Z_v$是item的embedding矩阵

### Graph convolutional encoder 图卷积编码器

本文针对推荐任务提出一种图自编码器的变体图卷积编码器。本文提出的编码器模型可以有效利用图形中各个位置之间的权重分配，并为每种边类型（或评分）$r \in\mathcal{R}$ 分配单独的处理通道。这种权值共享的形式受到了最近的一类卷积神经网络的启发，这些神经网络直接对图形结构的数据进行操作。图数据卷积层执行的局部操作只考虑节点的直接邻居，因此在图数据的所有位置都应用相同的转换。


局部图卷积可以看作是消息传递，其中特征值的信息被沿着图的边传递和转换

* Discriminative Embeddings of Latent Variable Models for Structured Data，ICML 2016
* Neural Message Passing for Quantum Chemistry 量子化学的神经信息传递，ICML 2017
    

文中为每个等级分配特定的转换，从item $j$ 到user $i$传递的信息$ \mu_{j \rightarrow i, r}$表示为
$$
μ_{j\rightarrow i,r}=\frac {1}{c_{ij}}W_rx{_j^v}
$$
$N(u_i)$ user i 邻居集合，c正则化常数，W 边类型的参数矩阵， x 节点的特征向量

从users到items的消息$\mu_{i \rightarrow j, r}$也以类似的方式传递。在消息传递之后，对每个节点都进行消息累计操作：对每种评分$r$下的所有邻居 $\mathcal{N}\left(u_{i}\right)$求和，并将它们累积为单个矢量表示：
$$
h_{i}=\sigma\left[\operatorname{accum}\left(\sum_{j \in \mathcal{N}_{i, 1}} \mu_{j \rightarrow i, 1}, \ldots, \sum_{j \in \mathcal{N}_{i, R}} \mu_{j \rightarrow i, R}\right)\right]
$$

* accum（·）表示一个聚合运算，例如stack（·），或sum（·）

    

$$
z_{i}^u=\sigma\left(W h_{i}^u\right)
$$

### Bilinear decoder 双线性解码器

为了在二部交互图中重建links，考虑一个双线性解码器，把每个评分等级看作是一类。$\check{M} 
$表示user$i$和item$j$之间重建的评分等级。解码器通过对可能的评分等级进行双线性运算，然后应用softmax函数生成一个概率分布：
$$
p\left(\check{M}_{i j}=r\right)=\frac{e^{\left(z_{i}^{u}\right)^{T} Q_{r} z_{j}^{v}}}{\sum_{s=1}^{R} e^{\left(z_{i}^{u}\right)^{T} Q_{s} v_{j}}}
$$

* Q 训练参数矩阵

预测的评分等级计算方式为：
$$
\check{M}_{i j}=g\left(u_{i}, v_{j}\right)=\mathbb{E}_{p\left(\check{M}_{i j}=r\right)}[r]=\sum_{r \in R} r p\left(\check{M}_{i j}=r\right)
$$
