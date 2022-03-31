> 论文：[RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation](https://arxiv.org/pdf/2101.12457.pdf)
>
> 作者：Cheng Hsu and Cheng-Te Li.  National Cheng Kung University
>
> 来源：WWW 2021
>
> 代码：[retagnn/RetaGNN (github.com)](https://github.com/retagnn/RetaGNN)

------

## Motivation

顺序推荐可以精准地根据用户最近浏览的物品，为其推荐一系列物品。一个重要的任务是在不重复训练的情况下创建用户和物品的embedding。因为用户-物品的互动可能非常稀疏，另一个重要任务是创建**Transferable SR**，将一个领域得到的丰富知识转换到另一个领域。在这项工作中，我们提出了一个新的基于深度学习的模型——**关系时序注意力图神经网络（RetaGNN）**。RetaGNN的主要思想有三个方面：

1.为了具有inductive和transferable的能力，我们从局部子图上的user-item对训练了**relation attentive GNN**，其中可学习的权重矩阵位于用户、项目和属性之间的各种关系上，而不是节点或边上。

2.用户偏好的长期和短期时间模式是通过提出的**sequential self-attention mechanism**进行编码的。

3.我们设计了一个**relaton-aware regularization term**，以更好地训练RetaGNN。



当前inventing SR还没有很好的发展，比如MA-GNN，HGN都是transductive。

现有的SR模型还可以从两点改进：

* **The first is the modeling of high-order user-item interactions in long and short terms of the given sequence.** The sequential evolution of multi-hop collaborative neighbors of a user in the interaction graph can reveal how user preferences change over time.

    挖掘user-item交互图里的高阶信息

* **The second is the temporal patterns in the derived representations of sequential items.** The adoption of the next items can be influenced by recent items with different weighting contributions.

    下一个被推荐的可能受到具有不同权重贡献的近期item的影响

![image-20220109215424473](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201092154743.png)

$u：user \\	v: item$

* Conventional SR: 推荐的用户和商品都是训练集里面的
* Inductive SR: 推荐给新用户已经有的产品
* Transferable SR:推荐给新用户新产品，  可以跨域

RetaGNN：要整一个兼顾上面三者的全局模型holistic SR

![image-20220110091942249](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201100919660.png)

## Solution

![image-20220109220400455](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201092204665.png)

分为五个部分：

1. 利用 one-hidden-layer feed-forward network (FFN) 产生 users, items, attributes 的初始向量。
2.  通过在不同的time frames将一个用户和她的每个交互项目配对，为每个item对提取长期和短期ℎ-hop enclosing subgraphs
3. 建立relation attentive graph neuralnetwork (RA-GNN) 来学习user和item的表示，该层对每个子图中sequential high-order user item interactions进行编码
4. 设计了一个sequential self-attention (SSA) 层去为用户偏好的temporal patterns建模，items embedding 也在这里更新
5. 预测

### Primitive Embedding Layer

首先考虑随机初始化所有user，item and attribute 的向量。节点上随机初始化的“固定大小”向量允许我们在相同的模型参数下更新新出现的没见过的节点(用于inductive)和cross-data节点(用于transferable)。

在构造的图中学习了directional edge的模型权值，它是独立于节点和数据集的. 因此，RetaGNN可以将新到节点和跨数据节点的随机初始化向量投射到相同的向量空间中，从而达到归纳和迁移的效果。

通过将随机初始向量输入到嵌入层（FFN），我们可以为每个user，item and attribute 生成一个低维稠密的向量。$\mathrm{X} \in \mathbb{R}^{q \times d}$, d是向量维数，q是user，item and attribute数量之和。

给一个用户u的序列${\mathcal{S}_{1:t}^{u}}$,那么
$$
\mathrm{X}_{\mathcal{S}_{1:t}^{u}}=\left[\mathrm{x}_{1} \cdots \mathrm{x}_{j} \cdots \mathrm{x}_{t}\right], \text { where } \mathrm{X}_{\mathcal{S}_{u t}^{u}} \in \mathbb{R}^{t \times d}, \text { and } \mathrm{x}_{j} \in \mathbb{R}^{d}
$$
不适用onehot初始化是因为不方便拓展没有见过的的item或者user

### User-Item-Attribute Tripartite Graph

![image-20220109224724659](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201092247787.png)

### Extracting Enclosing Subgraphs

Enclosing subgraphs来自于ICLR 2020 的[IGMC](https://arxiv.org/abs/1904.12058)这篇论文，IGMC是在二部图上.

> ![image-20220109231103926](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201092311069.png)
>
> reference：Zhang M, Chen Y. Inductive matrix completion based on graph neural networks[J]. arXiv preprint arXiv:1904.12058, 2019.

![image-20220109233314094](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201092333243.png)

### Relational Attentive GNN Layer

RA-GNN就是一个三部图上的GAT

分为两部分：

* relational attention mechanism: $\alpha_{i j}^{l}=\operatorname{softmax}\left(\rho\left(\mathbf{a}^{\top}\left[\mathbf{W}_{o}^{l} \mathbf{x}_{i}^{l} \oplus \mathbf{W}_{r}^{l} \mathbf{x}_{j}^{l}\right]\right)\right)$
* message passing：

![image-20220110092927836](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201100929618.png)

![image-20220110100822863](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101008358.png)

### Sequential Self-Attention

这里的注意力机制用于不同时间步。scaled dot-product attention

![image-20220110100905446](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101009880.png)
$$
\begin{gathered}
\mathrm{Z}_{a: b}^{u}=\beta_{\mathcal{S}_{a: b}^{u}}\left(\mathrm{~V}_{a: b}^{u} \mathrm{~W}^{v a l}\right) \\
\beta_{\mathcal{S}_{a: b}^{u} t_{j}}^{t_{i}}=\frac{\exp \left(e_{V}^{t_{i} t_{j}}\right)}{\sum_{k=1}^{T} \exp \left(e_{V}^{t_{i} t_{k}}\right)} \\
e_{V}^{t_{i} t_{j}}=\frac{\left(\left(\mathrm{V}_{a: b}^{u} \mathrm{~W}^{q u e}\right)\left(\mathrm{V}_{a: b}^{u} \mathrm{~W}^{k e y}\right)^{\top}\right)_{t_{i} t_{j}}}{\sqrt{d}}+I_{t_{i} t_{j}},
\end{gathered}
$$
where $\mathbf{I} \in \mathbb{R}^{T \times T}$ is a mask matrix whose element $I_{t_{i} t_{j}}$ is either $-\infty$ or $0: I_{t_{i} t_{j}}=0$ if $a \leq t_{i} \leq t_{j} \leq b$; otherwise, $I_{t_{i} t_{j}}=-\infty$. 

### Final Embedding Generation

分成多个子序列

### Model Training

![image-20220110102918490](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101029830.png)

## Evalution

![image-20220110103010758](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101030219.png)

### Dataset

![image-20220110103155508](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101031589.png)

![image-20220110103251052](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101032134.png)

![image-20220110103410549](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101034666.png)

