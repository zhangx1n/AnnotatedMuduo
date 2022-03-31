> 论文：Modeling Relational Data with Graph Convolutional Networks 
>
> 作者：Michael Schlichtkrull, Thomas N. Kipf（GCN的作者）, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling 
>
> 论文链接：[https://arxiv.org/abs/1703.06103](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.06103) Github链接：[https://github.com/tkipf/relational-gcn](https://link.zhihu.com/?target=https%3A//github.com/tkipf/relational-gcn) 
>
> 期刊： ESWC 2018

## 引言

​		这篇论文是比较早的一篇关于GCN的论文，但是目前引用超 400 次，其关键在于他解决了利用GCN来处理图结构中不同边关系对节点的影响，而这也是GCN中忽略的一点，没有考虑节点之间的关系。本文中作者提出的R-GCN模型应用于**链路预测**和**实体分类**两项任务上，对于链路预测任务，通过在关系图中的多个推理步骤中使用编码器模型来积累信息，可以显著改进链路预测的模型；对于实体分类任务，则是类似于GCN论文中，即对每个节点使用一个softmax分类器，通过R-GCN来提取每个节点表示用于节点类别的预测。

​		文中举了一个例子：

![image-20211011202127913](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110112021980.png)

上图中红色字体所在的边和顶点就是链路预测和定点分类，，我们在知道“Mikhail Baryshnikov”在“Vaganova Academy”大学上学后，那么对于他和“USA”的关系我们就可以想到是“citizen of”。这就是 R-GCN 模型的联系预测，顶点标签预测也是同理。

## 主要贡献

本篇论文最主要的贡献有三：

1. 将GCN框架应用于关系数据建模，特别是链路预测和实体分类任务。

2. 在具有大量关系的多图中应用了参数共享以及实现稀疏约束的技术

3. 使用一个在关系图中执行多步信息传播的模型来加强因子分解模型。

# RGCN

## GCN

R-GCN和GCN都是利用图卷积模型来模拟信息在网络结构中的传递，因此那这个部分的一个框架可以如下所示：

$h_{i}^{(l+1)}=\sigma\left(\sum_{m \in \mathcal{M}_{i}} g_{m}\left(h_{i}^{(l)}, h_{j}^{(l)}\right)\right)$

其中：

* $g_{m}(\cdot, \cdot)$ 指的就是将传入的消息进行聚合并通过激活函数传递 
* $M_{i}$ 指的是节点$v_i$ 的传入消息集，通常选择为传入的边集
* $h_{i}^{(l)}$ 指的是节点$i$的第$l$层节点表示,
* $h_{j}^{(l)}$ 指的是节点$j$的所有邻居节点的第$l$层节点表示,

## R-GCN

基于上述的方式，论文定义了在一个关系多图的传播模型，图中节点$v_i$的更新方式如下：

$$h_{i}^{(l+1)}=\sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_{i}^{r}} \frac{1}{c_{i, r}} W_{r}^{(l)} h_{j}^{(l)}+W_{0}^{(l)} h_{i}^{(l)}\right)$$

其中：

* $N_{i}^{r}$ 表示节点的关系为 $r$ 的邻居节点集合
* $c_{i, r}$ 是一个正则化常量，其中 $c_{i, r}$ 的取值为 $\left|N_{i}^{r}\right|$
* $W_{r}^{(l)}$ 是线性转化函数，将同类型边的邻居节点，使用用一个参数矩阵 $W_{r}^{(l)}$ 进行转化。

此公式个GCN不同的是，不同边类型所连接的邻居节点，进行一个线性转化，$W_{r}^{(l)}$  的个数也就是边类型的个数，论文中称为relation-specific。当然此处还可以设置更加灵活的函数，例如多层神经网络。

![image-20211011203847208](C:\Users\13505\AppData\Roaming\Typora\typora-user-images\image-20211011203847208.png)

通过这幅图可以看清，对于中心红色的节点进行一次卷积，通过聚合邻居节点的信息来更新自身节点的表示。其中邻居节点的聚合是按照边的类型进行分类，根据边类型的不同进行相应的转换，收集的信息经过一个正则化的加和(绿色方块)，最后通过激活函数(relu)。其中每个顶点的信息更新共享参数，并行计算，同时也包括自连接，也就是说包括了节点自身表示。

## Regularization

由于R-GCN模型应用于多关系数据，因此一个核心的问题就是当图中关系与参数数目增长时，会产生对罕见关系过拟合的问题。文中对此提出了两种独立的方法对R-GCN层进行规则化：基函数分解( **basis/dictionary learning**)和块对角分解(**block diagonal matrices**)。

![image-20211011204520562](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110112045693.png)

![image-20211011204532098](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110112045221.png)

## 结点分类与链路预测

对于**节点分类**任务，通过R-GCN的卷积，可以得到每个节点的向量表示，然后再最后一层使用softmax激活函数，得到每个节点的预测类别。最后通过有标签的节点来学习模型的参数，具体的是通过最小化交叉熵损失函数：

$L=-\sum_{i \in Y} \sum_{k=1}^{K} t_{i k} l n h_{i k}^{(L)}$

其中：

* $Y$ 表示节点的索引集合
* $h_{i k}^{(L)}$表示在第L层的有标签的第i个节点的第k个
* $t_{i k}$真实标签

![image-20211011204937415](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110112049482.png)

对于**链路预测**问题：

作者使用一个图自编码器模型，其中括一个实体编码器和一个评分函数。作者用负采样训练模型。对于每一个观察到的例子，都进行负采样。通过随机破坏每个正例子的主题或对象来负采样。对交叉熵损失进行了优化，以使得模型对正样例的预测得分高于负样例。

## 参考

[知乎资料](https://zhuanlan.zhihu.com/p/157902271)

