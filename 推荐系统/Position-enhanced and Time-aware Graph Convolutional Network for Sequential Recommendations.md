> **Title:**[Position-enhanced and Time-aware Graph Convolutional Network for Sequential Recommendations](https://arxiv.org/ftp/arxiv/papers/2107/2107.05235.pdf)
>
> **Published:** 2022-01-14
>
> **Authors:** Liwei Huang,Yutao Ma,Yanbo Liu,Bohong,Du,Shuliang Wang,Deyi Li

## 摘要

现有的基于深度学习的顺序推荐方法大多采用递归神经网络结构或自注意力机制来建模用户历史行为之间的顺序模式和时间影响，并在特定时间学习用户的偏好。然而，这些方法有两个主要缺点。首先，他们专注于从以用户为中心的角度对User的动态状态进行建模，而总是**忽略Item的动态**。第二，大多数用户只处理一阶用户项目交互，并且**不考虑用户和项目之间的高阶连通性**，而高阶连通性最近被证明有助于序列推荐。为了解决上述问题，在本文中，我们尝试用二部图结构来建模用户项交互，并提出了一种新的基于位置增强和时间感知的图卷积网络（PTGCN）的顺序推荐方法。PTGCN通过定义位置增强和时间感知的图卷积运算，并使用自注意力聚合器在二部图上同时学习用户和项目的动态表示，对用户-项目交互之间的顺序模式和时间动态进行建模。同时，通过多层图形的叠加，实现了用户与物品之间的高阶连通。为了证明PTGCN的有效性，我们在三个不同大小的真实数据集上对PTGCN进行了综合评估，并与一些竞争基线进行了比较。实验结果表明，PTGCN在两种常用的排名评估指标方面优于几种最先进的模型。

## Introduction

### Background

以往的序列推荐算法(或模型)大多关注按交互时间排序的动作序列的单向链结构，包括两种方法:**基于马尔可夫链的方法和基于神经网络的方法**。基于马尔可夫链的方法使用L阶马尔可夫链根据最近的L行为提出建议。通过简化一些假设，该方法可以在高稀疏的情况下取得良好的效果。然而，由于对user-item交互的复杂动态建模能力有限，因此在长期推荐场景中，它的表现往往不佳。与基于马尔可夫链的推荐方法相比，基于神经网络的推荐方法，如**循环神经网络、卷积神经网络、Transformer**等，已成为网络用户行为序列模式建模的热门方法。最近，一些基于神经网络的方法试图利用用户行为的时间动态来提高特定领域的推荐性能。

### Motivation

虽然这些基于RNN和transformer的推荐方法在序列推荐任务中取得了较好的效果，但它们仍存在两个不足之处。首先，大多数只考虑了user行为的时间动态，而忽略了user属性的时间动态。正如我们所知，物品具有静态属性，不会随时间变化，也有会随时间变化(time-evolving)的属性。一个item可能会随着时间的推移显示出不同的时间动态，例如流行度的增长和社会话题的出现。有必要设计一个统一的框架来同时利用user行为和item属性的动态。其次，现有的方法在定义用于模型训练的损失函数时，只考虑了用户-商品直接交互(即一阶连通性)，而忽略了嵌入在用户-用户和商品-商品交互中的重要协同信息。因此，用户与物品的嵌入可能不足以捕捉代表用户(或物品)之间行为相似性的协作信号(或称为高阶连通性)。

![image-20220314225957119](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220314225957119.png)

## Position-enhanced and Time-aware GCN

![image-20220314230049875](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220314230049875.png)

PTGCN的体系结构如图所示，该模型框架由三个部分组成:(1)嵌入层，生成四种类型的embedding，four types of embeddings, namely user embedding, item embedding, time embedding, and position embedding; (2)卷积层，利用设计的位置增强和时间感知的图卷积对高阶连通性进行建模，细化用户和物品的嵌入;(3)预测层，将精细化的用户嵌入和项目嵌入聚合起来，并为每个用户-项目对输出一个评分。

### Embedding Layer

In particular, we use u ser neighborhood to update the user embedding and item neighborhood to update the item embedding.

#### User embedding and item embedding

首先，假设每一个user或者item的embedding只在交互发生时才会改变。为所有的users和items分别创建两个embedding matrix $\mathbf{U} \in \mathbb{R}^{|U| \times d}$, $\mathbf{V} \in \mathbb{R}^{|V| \times d}$.  对于每一次交互$i_{u_i,v_j, t}$ , 我们对user和item索引执行直接查找操作，并获得user和item相应的的embedding。（每个时间t，embedding可能会不一样）

#### Time embedding

时间信息是分析个体交互行为的必要条件。由于顺序推荐任务是依赖于时间的，我们需要从交互的连续时间特性中学习一种合适的时间表示。最直接的方法之一是直接使用原始的特性值或transformation，而不embedding。然而，由于表示的容量较低，这种方法的性能往往很差。此外，还有两种方法将时间信息嵌入到低维向量中。**场嵌入方法 (field embedding method)** 通过定义一个连续泛函 $\emptyset(\cdot)$ 来将时域的时间间隔映射到d维向量空间来学习每个数值场的单个场嵌入。**离散化方法 (discretization method)** 利用各种启发式离散化策略将数值特征转换为类别特征，然后使用分类策略进行嵌入。在一些论文中，经过的时间被分割成长度呈指数增长的间隔。例如，我们可以将时间映射到范围[0,1)，[1,2)，[2,4)，…， [2k, 2k + 1]为0,1,2，…， k + 1。不同的交互组可能有不同的时间切片粒度。然后，对分类时间特征进行直接查找，得到时间点t的时间嵌入$t$。

文章中采用的是discretization method，在每一次交互$i_{u_i,v_j, t}$，可以得到时间点t的time embedding $\mathbf{t}$, 并且利用user $u_i$ 在时间点 t 的邻居 $N_{u_i,t}$获得新的向量 $\mathbf{u}_{i,t}$ , 对item同样。

特别地，我们利用$N_{u_i}$中两个连续交互之间的时间间隔和时间点t来建模历史交互对ui当前状态的影响;同样，这与vj类似。

#### Position embedding

和transformer相同， $\mathbf{p_{i,t}}$

### Convolutional Layer

#### Position enhanced and time aware graph convolution

现有的大多数GCN模型的一个基本限制是它们**不能捕获每个节点在邻域中的位置信息**。在顺序推荐的场景中，我们需要对不同交互的时间效应进行建模，以获得用户和项目的动态表示。因此，我们提出了一种位置增强和时间感知的图卷积，结合了交互作用的时序和时间信息。
$$
\begin{gathered}
\mathbf{h}_{N_{u_{i}, t_{q}}^{(l)}}=\operatorname{AGGREGATE}\left(\left\{\left(\mathbf{v}_{j, t}^{(l-1)}, \mathbf{t}, \mathbf{p}_{j, t}\right) \mid i_{u_{i}, v_{j}, t} \in N_{u_{i}, t_{q}}\right\}\right), \\
\mathbf{u}_{i, t_{q}}^{(l)}=\mathbf{W}_{U_{2}} \cdot \sigma\left(\mathbf{W}_{U_{1}} \cdot \operatorname{CONCAT}\left(\mathbf{u}_{i, t_{q}}^{(l-1)}, \mathbf{h}_{N_{u_{i}, t}(l)}^{(l)}\right)\right)
\end{gathered}
$$

#### Aggregator architecture

大多数GCN模型忽略了节点邻域内节点的顺序，但这一特性对于为顺序推荐建模顺序模式至关重要。如上所述，同时对时序影响和时间影响进行建模也是必要的。考虑到自注意机制已经应用于顺序推荐，并取得了显著的效果，我们设计了一个使用自注意机制的聚合器。**自注意聚合器的核心思想是通过相应的时间嵌入和位置嵌入来丰富每个用户特征和物品特征。**该自注意聚合器具有K个相同的非线性层，每个非线性层包含一个自注意层、一个前馈注意层和一个普通注意层。

![image-20220315112745464](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315112745464.png)

![image-20220315112803295](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315112803295.png)

![image-20220315112816663](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315112816663.png)

#### Stacking convolutions

用于建模协作信号的高阶连通性对于评估用户的产品偏好至关重要。通过一阶连通性建模对表示进行扩充，我们可以叠加更多的位置增强和时间感知的图卷积来捕获二部图中的高阶连通性。通过堆叠$l$层图卷积，用户(或项目)可以从它的l-hop邻居接收协作信息。

### Model Prediction

$$
\hat{y}\left(u_{i}, v_{j}, t_{N}\right)=\left(\mathbf{z}_{u_{i}, t_{N}}\right)^{T} \cdot \mathbf{z}_{v_{j}, t_{N}}
$$

## Experiment and Result Analysis

实验通过回答以下四个研究问题进行:

* PTGCN在连续推荐任务方面的表现是否优于最先进的基线?

* 对项目的时间动态建模是否有利于顺序推荐?

* 高阶连通性是否有助于更好的推荐性能?

### Dataset

![image-20220315113245547](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315113245547.png)

### Evaluation Metrics

* $Recall@k$
* $NDCG@K$

### Baseline

![image-20220315113506939](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315113506939.png)

### Result

对于问题一：确实效果不错

![image-20220315113805819](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315113805819.png)

对于问题二：

![image-20220315113955134](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315113955134.png)

对于问题三：

![image-20220315114018313](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220315114018313.png)

## 总结

最大的启发就是user和item的embedding是不是可以建模成随时间变化的。