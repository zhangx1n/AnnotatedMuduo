> 论文：[Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)
> 作者：William L. Hamilton, Rex Ying, Jure Leskovec
> 来源：NIPS17

## 概括

在大规模图上学习节点embedding，在很多任务中非常有效，如学习节点拓扑结构的 DeepWalk 以及同时学习邻居特征和拓扑结构的semi-GCN。

**但是现在大多数方法都是直推式学习， 不能直接泛化到未知节点。**这些方法是在一个固定的图上直接学习每个节点embedding，但是大多情况图是会演化的，当网络结构改变以及新节点的出现，直推式学习需要重新训练（复杂度高且可能会导致embedding会偏移），很难落地在需要快速生成未知节点embedding的机器学习系统上。

本文提出归纳学习—**GraphSAGE(Graph SAmple and aggreGatE)框架**，通过训练聚合节点邻居的函数（卷积层），使GCN扩展成归纳学习任务，对未知节点起到泛化作用。

> **直推式(transductive)学习**：从特殊到特殊，仅考虑当前数据。在图中学习目标是学习目标是直接生成当前节点的embedding，例如DeepWalk、LINE，把每个节点embedding作为参数，并通过SGD优化，又如GCN，在训练过程中使用图的拉普拉斯矩阵进行计算，
> **归纳(inductive)学习**：平时所说的一般的机器学习任务，从特殊到一般：目标是在未知数据上也有区分性。

## GraphSAGE框架

本文提出GraphSAGE框架的核心是如何聚合节点邻居特征信息，本章先介**绍GraphSAGE前向传播过程**（生成节点embedding），**不同的聚合函数**设定；然后介绍**无监督和有监督的损失函数**以及**参数学习。**

### 前向传播

**a. 可视化例子：**下图是GraphSAGE 生成目标节点（红色）embededing并供下游任务预测的过程：

![image-20211029224306226](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110292243335.png)

1. 先对邻居随机采样，降低计算复杂度（图中一跳邻居采样数=3，二跳邻居采样数=5）
2. 生成目标节点emebedding：先聚合2跳邻居特征，生成一跳邻居embedding，再聚合一跳邻居embedding，生成目标节点embedding，从而获得二跳邻居信息。（后面具体会讲）。
3. 将embedding作为全连接层的输入，预测目标节点的标签。

**b. 伪代码:**

![image-20211029224327923](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110292243990.png)

4-5行是核心代码，介绍卷积层操作：聚合与节点v相连的邻居（采样）k-1层的embedding，得到第k层邻居聚合特征$h_{N(v)}^{k}$,与节点v第k-1层embedding$h_{v}^{k-1}$拼接，并通过全连接层转换，得到节点v在第k层的embedding $h_{v}^{k}$

> [Weisfeiler-Lehman Isomorphism Test](https://www.davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/)：

**c. 疑问：**

第一眼看上去就是每层都是聚合的一阶邻居信息，**k层怎么就有了k阶的信息？**

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110301635694.png" alt="image-20211030163509421" style="zoom:50%;" />

其实是在第0层聚合后，**所有节点**都具有了一阶邻居节点信息，然后在在第一层聚合邻居信息后，其实是融合了一阶邻居的邻居的节点信息，此时是包含二阶信息了

> reference：[GraphSAGE：我寻思GCN也没我牛逼](https://zhuanlan.zhihu.com/p/74242097)

### 聚合函数

伪代码第5行可以使用不同聚合函数，本小节介绍五种满足排序不变量的聚合函数：平均、GCN归纳式、LSTM、pooling聚合器。（因为邻居没有顺序，聚合函数需要满足排序不变量的特性，即输入顺序不会影响函数结果）

**a.平均聚合：**先对邻居embedding中每个维度取平均，然后与目标节点embedding拼接后进行非线性转换。

$h_{N(v)}^{k}=\operatorname{mean}\left(\left\{h_{u}^{k-1}, u \in N(v)\right\}\right)$
$h_{v}^{k}=\sigma\left(W^{k} \cdot \operatorname{CONCAT}\left(h_{v}^{k-1}, h_{N(u)}^{k}\right)\right)$

**b. 归纳式聚合：**直接对目标节点和所有邻居emebdding中每个维度取平均（替换伪代码中第5、6行），后再非线性转换：

$h_{v}^{k}=\sigma\left(W^{k} \cdot \operatorname{mean}\left(\left\{h_{v}^{k-1}\right\} \cup\left\{h_{u}^{k-1}, \forall u \in N(v)\right\}\right)\right.$

**c. LSTM聚合：**LSTM函数不符合“排序不变量”的性质，需要先对邻居随机排序，然后将随机的邻居序列embedding $\left\{x_{t}, t \in N(v)\right\}$作为LSTM输入

**d. Pooling聚合器:**先对每个邻居节点上一层embedding进行非线性转换（等价单个全连接层，每一维度代表在某方面的表示（如信用情况）），再按维度应用 max/mean pooling，捕获邻居集上在某方面的突出的／综合的表现 以此表示目标节点embedding。

$h_{N(v)}^{k}=\max \left(\left\{\sigma\left(W_{p o o l} h_{u i}^{k}+b\right)\right\}, \forall u_{i} \in N(v)\right)$
$h_{v}^{k}=\sigma\left(W^{k} \cdot \operatorname{CON} C A T\left(h_{v}^{k-1}, h_{N(u)}^{k-1}\right)\right)$

### 无监督和有监督的损失设定

损失函数根据具体应用情况，可以使用**基于图的无监督损失**和**有监督损失**。

**a. 基于图的无监督损失：**希望节点u与“邻居”v的embedding也相似（对应公式第一项），而与“没有交集”的节点 $v_n$不相似（对应公式第二项)。

$J_{\mathcal{G}}\left(\mathbf{z}_{u}\right)=-\log \left(\sigma\left(\mathbf{z}_{u}^{\top} \mathbf{z}_{v}\right)\right)-Q \cdot \mathbb{E}_{v_{n} \sim P_{n}(v)} \log \left(\sigma\left(-\mathbf{z}_{u}^{\top} \mathbf{z}_{v_{n}}\right)\right)$

* $z_u$为节点u通过GraphSAGE生成的embedding。
* 节点v是节点u随机游走访达“邻居”。
* $v_{n} \sim P_{n}(u)$表示负采样：节点$v_n$是从节点u负采样分布$P_n$采样的，Q为采样样本数。
* embedding之间相似度通过向量点积计算得到

**b. 有监督损失：**无监督损失函数的设定来学习节点embedding 可以供下游多个任务使用，若仅使用在特定某个任务上，则可以替代上述损失函数符合特定任务目标，如交叉熵。

### 参数学习

通过前向传播得到节点u的embedding $z_u$,然后梯度下降（实现使用Adam优化器） **进行反向**传播优化参数 $W_k$和聚合函数内参数。

## 实验

### 实验目的

1. 比较GraphSAGE 相比baseline 算法的提升效果；
2. 比较GraphSAGE的不同聚合函数。

### 数据集及任务

1. Citation 论文引用网络（节点分类）
2. Reddit web论坛 （节点分类）
3. PPI 蛋白质网络 （graph分类）

### 比较方法

1. 随机分类器
2. 手工特征（非图特征）
3. deepwalk（图拓扑特征）
4. deepwalk+手工特征
5. GraphSAGE四个变种 ，并无监督生成embedding输入给LR 和 端到端有监督

(分类器均采用LR)

### GraphSAGE 设置

- K=2，聚合两跳内邻居特征
- S1=25，S2=10： 对一跳邻居抽样25个，二跳邻居抽样10个
- RELU 激活单元
- Adam 优化器
- 对每个节点进行步长为5的50次随机游走
- 负采样参考word2vec，按平滑degree进行，对每个节点采样20个。
- 保证公平性，：所有版本都采用相同的minibatch迭代器、损失函数、邻居抽样器。

### 运行时间和参数敏感性

1. **计算时间：**下图A中GraphSAGE中LSTM训练速度最慢，但相比DeepWalk，GraphSAGE在预测时间减少100-500倍（因为对于未知节点，DeepWalk要重新进行随机游走以及通过SGD学习embedding）
2. **邻居抽样数量：**下图B中邻居抽样数量递增，边际收益递减（F1），但计算时间也变大。 平衡F1和计算时间，将S1设为25。
3. **聚合K跳内信息**：在GraphSAGE， K=2 相比K=1 有10-15%的提升；但将K设置超过2，边际效果上只有0-5%的提升，但是计算时间却变大了10-100倍。

![image-20211029225929166](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110292259258.png)

### 效果

1. GraphSAGE相比baseline 效果大幅度提升
2. GraphSAGE有监督版本比无监督效果好。
3. LSTM和pool的效果较好