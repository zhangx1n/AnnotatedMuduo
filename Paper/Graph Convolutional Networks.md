> reference: [Semi-Supervised Classification with Graph Convolutional ](https://arxiv.org/pdf/1609.02907.pdf?fbclid=IwAR0BgJeoKHIAvPuSE9fJ0_IQOEu5l75yxyNo7PUC08RTOFlm_IIo5YmcnQM)

# Semi-Supervised Classification with Graph Convolutional Networks

## 四个问题

1. 要解决什么问题？
    1. 半监督任务。给定一个图，其中一部节点已知标签，剩下的未知，要对整个图上的节点进行分类。
2. 用了什么方法解决？
    1. 提出了一种卷积神经网络的变种，即提出了一种新的图卷积方法。
    2. 使用谱图卷积（spectral graph convolution）的局部一阶近似，来确定卷积结构。
    3. 所提出的的网络可以学习图上局部结构的特征，并进行编码。
3. 效果如何？
    1. 在引文网络（citation network）和知识图谱（knowledge graph）等的数据集上比其之前的方法效果更好。
4. 还存在什么问题？
    1. 最大的问题就是对GPU显存的占用较大，要使用较大规模的图来训练网络只能用CPU替代。
    2. 文中的模型只是为无向图设计的，并不支持对边特征的提取。尽管能够将一个有向图看做一个无向加权联合图，但这个模型对于有向图的支持能力还是有限。

## 论文简述

### 简介

* 使用神经网络$f(X,A)$对图的结构进行编码，对所有带标签的节点进行有监督训练。
* $X$是输入数据，$A$是图邻接矩阵。
* 在图的邻接矩阵上调整$f(⋅)$能让模型从监督损失$L_0$

### 图上的快速近似卷积

* 图卷积的前向传播公式：

    $H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$

* 其中, $\tilde{A}=A+I_{N}$ 表示无向图的邻接矩阵加上self-connections, $I_{N}$ 为单位矩阵, $\tilde{D}_{i i}=\sum_{j} \tilde{A}_{i j}$ ， $W^{(l)}$ 表示可训练的权重矩阵 $H^{(l)} \in \mathbb{R}^{N \times D}$ 表示上一层的激活值， $H^{(0)}=X$ 。以下说明可以通 过一个局部谱图滤波的一阶近似来推导上式。



------



#### 推导

***GCN推导***  

与CNN类似，当前节点的更新信息由当前节点的信息和周围邻居信息累加得到，对于图结构而言，我们需要为每个节点添加一个自连接

![](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102300743.jpg)  



比如对于节点1，它的更新后的信息就是：  



$$[0.4,0.2,0.7]+[0.2,0.3,0.5]+[0.3,0.3,0.5]+[0.4,0.2,0.3]=[1.3,1.0,2.0]$$  



再比如，对于节点4，它的更新后的信息就是：  



$$[0.2,0.4,0.3]+[0.4,0.2,0.3]=[0.6,0.6,0.6]$$  



想必，你也发现问题了，对于邻居很多的节点，聚合后的数值会比其它邻居少的节点大很多，所以我们需要进行归一化，GCN是采用的归一化方式如下，对于节点$v_i,v_j$，它们的度为$d(v_i),d(v_j)$，聚合信息时，会在它们前面乘以一个权重，即度的乘积的平方根的倒数：   $\frac{1}{\sqrt{d(v_i)}\cdot \sqrt{d(v_j)}}$

所以，这时对于节点1的更新就是：  



$$\frac{1}{4}[0.4,0.2,0.7]+\frac{1}{2\sqrt{2}}[0.2,0.3,0.5]+\frac{1}{2\sqrt{3}}[0.3,0.3,0.5]+\frac{1}{2\sqrt{3}}[0.4,0.2,0.3]$$  



上面的更新操作，可以对$X$左乘一个矩阵来进行计算：$\tilde{L}_{sym}X$  

这里的$X$就是我们的因子数据，第$i$行就是第$i$个因子的向量表示，比如$X_{0,:}=[0.2,0.3,0.5]$，而  



$$\tilde{L}_{sym}=\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2},\tilde{A}=A+I,\tilde{D_{ii}}=\sum_j\tilde{A}_{ij}$$  



这里的$A$是连接矩阵，$I$是单位矩阵，所以$\tilde{A}X$，就是添加了自连接且没加权的聚合表示，即最上面的表示，如节点1的聚合  



$$[0.4,0.2,0.7]+[0.2,0.3,0.5]+[0.3,0.3,0.5]+[0.4,0.2,0.3]=[1.3,1.0,2.0]$$  



而$\tilde{D}$是$\tilde{A}$的度矩阵，它只有对角线上有值，$\tilde{D}_{ii}$的值就是$\tilde{A}$的第$i$行求和，所以$\tilde{A}$矩阵前后分别乘一个$\tilde{D}^{-1/2}$相等于乘以了上文介绍的权重$\frac{1}{\sqrt{d(v_i)}\cdot \sqrt{d(v_j)}}$

------

## 谱图卷积

### 原始GCN

![image-20211010230754608](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102307694.png)

### 加速版本GCN

![image-20211010230808674](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102308765.png)

### 线性模型

![image-20211010231036372](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102310467.png)

## 半监督节点分类

* 回到半监督任务上，前面介绍了优化后的图卷积结构。在现在的半监督任务中，作者希望通过已知的数据$X$和邻接矩阵$A$来训练图卷积网络$f ( X , A )$ 。 作者认为，在邻接矩阵中包含了一些$X$中没有的隐含的图的结构信息，而我们可以利用这些信息进行推理。
* 下图中，左图是一个GCN网络示意图，输入有$C$维特征，输出有$F$维特征，中间有若干隐藏层，$X$是训练数据，$Y$是标签。右图是使用一个两层GCN在Cora数据集上（只是用了5%的标签）得到的可视化结果。
    

![image-20211010231325029](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102313124.png)

### 实例

![image-20211010231359913](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102313995.png)

## 实验

* 数据集描述

    ![image-20211010231503995](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102315060.png)

* 半监督分类准确率

![image-20211010231516795](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110102315856.png)