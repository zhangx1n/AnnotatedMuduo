> 论文：[Sequential Recommendation with Graph Neural Networks](https://arxiv.org/pdf/2106.14226.pdf)
>
> 代码：[tsinghua-fib-lab/SIGIR21-SURGE: Official implementation of SIGIR'2021 paper: "Sequential Recommendation with Graph Neural Networks". (github.com)](https://github.com/tsinghua-fib-lab/SIGIR21-SURGE)
>
> 作者：清华大学未来智能实验室、快手社科推荐模型组
>
> 来源：SIGIR 2021
>
> reference: [SIGIR 2021｜快手联合清华提出基于图神经网络的序列推荐新方法_腾讯新闻 (qq.com)](https://new.qq.com/omn/20210706/20210706A04TWJ00.html)

## Motivation

现存两个挑战：

- 在丰富的历史序列中，用户的**行为是隐式（点击、观看）的并且有噪声**信号在里面，从而导致无法充分的反应用户的偏好。噪声：用户可能点击了该item，但是后续没有和相似的其他item发生交互，那么用户可能对该item并不感兴趣。
- 用户的偏好是**动态变化**的，从历史数据中难以挖掘用户是模式。

本文提出 SeqUential Recommendation with Graph neural nEtworks （SURGE）来解决上述问题。

## Solution

SURGE模型主要包含四部分，分别为：

- **兴趣图构建（Interest Graph Construction）**：基于[度量学习](https://blog.csdn.net/gdengden/article/details/82715162)将松散的项目序列重新构建为紧密的项目-项目兴趣图，明确地整合和区分了长期用户行为中不同类型的偏好。
- **兴趣融合图卷积层（Interest-fusion Graph Convolutional Layer）**：通过在兴趣图上进行图卷积，动态融合用户兴趣，强化重要的行为，弱化噪声行为。
- **兴趣提取图池化层（Interest-extraction Graph Pooling Layer）**：采用动态池化的方式对用户不同时间动态变化的兴趣进行提取。
- **预测层（Predict Layer）**：对后续可能交互的item进行预测。

![image-20220110195442091](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201101954397.png)

### 兴趣图构建

通常可以采用共现关系来构图，但是共现关系的稀疏性导致不足以为每一个用户构建图。本文结合度量学习进行构图。

#### 原始图构建

为每一个交互序列构建无向图$\mathcal{G}=\{\mathcal{V}, \mathcal{E}, A\}$, $\mathcal{E}$ 表示边的集合，A 是邻接矩阵，V是item， item对应的embedding是h。目标是学习到邻接矩阵A反应item i与item j之间是否相关。 通过该图反映用户的**核心兴趣和边缘兴趣**，所谓核心兴趣就是具有更高的度（degree），连接更多的相似兴趣节点。相似兴趣的频率越高，子图就越密集。

#### 相似度度量

构建了基本图后，需要判断哪些item之间时相似的。本文采用加权余弦相似度来度量相似关系，可学习参数w和item的embedding h做哈达玛积，然后计算余弦相似度。
$$
M_{i j}=\cos \left(\overrightarrow{\mathbf{w}} \odot \vec{h}_{i}, \overrightarrow{\mathbf{w}} \odot \vec{h}_{j}\right)
$$
为了提高表达能力和稳定性，将上式扩展为多头度量，就是计算多次取平均。

#### 图稀疏化

之前通过计算两两item之间的相似度得到了一个n*n的矩阵$M$, 如果把这个矩阵作为邻接矩阵，那么这个邻接矩阵里每一个位置都有值，相当于都是邻接的，这在计算上代价高昂，并且可能引入噪声。同时图卷积无法专注于最相关的部分，因此需要对当前的图稀疏化，只提取其中最重要的关系。
$$
A_{i j}= \begin{cases}1, & M_{i j}>=\operatorname{Rank}_{\varepsilon n^{2}}(M) \\ 0, & \text { otherwise }\end{cases}
$$
**大于某个阈值的就都为1，表示强关系；小于某个阈值的，都为0。只保留 $ \varepsilon n^{2} $个强关系。**

### 兴趣融合图卷积层

#### 通过图注意力卷积进行兴趣融合

本节作者提出了聚类感知和查询感知的图注意力卷积层，在信息聚合过程中感知用户的核心兴趣（即位于聚类中心的item）和与**查询**兴趣相关的兴趣（即当前**目标item**）。查询（query）就是对应目标item。 根据下式，可以通过聚合将原有的embedding h转换为新的能够反映用户兴趣偏好的embedding h'。其中$E_{ij}$表示对齐分数，将目标节点$v_i$的重要性映射到其周围的节点$v_j$，其具体计算方式和注意力机制类似，后面介绍。w为可学习参数，聚合函数Aggregate可以是sum，max，gru等等，本文采用sum。
$$
\vec{h}_{i}^{\prime}=\sigma\left(\mathbf{W}_{\mathrm{a}} \cdot \text { Aggregate }\left(E_{i j} * \vec{h}_{j} \mid j \in \mathcal{N}_{i}\right)+\vec{h}_{i}\right)
$$
和计算相似度一样，此处为了稳定性，文中也采用了多头的方式.

#### 聚类感知和查询感知的注意力机制

通过这个注意力机制得到的权重就是我们在上一小节中用到的权重$E_{ij}$。

作者假设目标节点$v_i$及其邻居$v_j$会形成一个簇（聚类），并且簇的中心是$v_i$。定义$v_i$的k阶邻居是他的感受野，

这些邻居节点的embedding的均值为该簇的平均信息。为了判断$v_i$是否为中心，可以采用下式计算分数，
$$
\alpha_{i}=\text { Attention }_{c}\left(\mathbf{W}_{\mathbf{c}} \vec{h}_{i}\left\|\vec{h}_{i_{c}}\right\| \mathbf{W}_{\mathbf{c}} \vec{h}_{i} \odot \vec{h}_{i_{c}}\right)
$$
$h_{i_c}$就是上面的均值。

为了服务于下游的动态池化方法，学习用户兴趣对不同目标兴趣的独立演化，还应该考虑source node embedding $\vec{h}_{j}$和 target item embedding $\vec{h}_{t}$之间的相关性。
$$
\beta_{j}=\text { Attention }_{q}\left(\mathbf{W}_{\mathbf{q}} \vec{h}_{j}\left\|\vec{h}_{t}\right\| \mathbf{W}_{\mathbf{q}} \vec{h}_{j} \odot \vec{h}_{t}\right)
$$
上述两部分，第一个是聚类感知，第二个是查询感知。将这两部分结合可以得到注意力权重
$$
E_{i j}=\operatorname{softmax}_{j}\left(\alpha_{i}+\beta_{j}\right)=\frac{\exp \left(\alpha_{i}+\beta_{j}\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\alpha_{i}+\beta_{k}\right)}
$$

### 兴趣提取图池化

####  通过图池化进行兴趣提取

首先要搞一个cluster assignment matrix  $S $标记这个节点是哪个类的。

那么它就可以将节点信息池化为聚类信息。其中n是节点个数，m是超参数，表示聚类的个数。对前面的$\beta$做softmax得到$\gamma$.通过聚类矩阵S，可以将原有的embedding转换为新的embedding。
$$
\begin{aligned}
&\left\{\vec{h}_{1}^{*}, \vec{h}_{2}^{*}, \ldots, \vec{h}_{m}^{*}\right\}=S^{T}\left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{n}^{\prime}\right\} \\
&\left\{\gamma_{1}^{*}, \gamma_{2}^{*}, \ldots, \gamma_{m}^{*}\right\}=S^{T}\left\{\gamma_{1}, \gamma_{2}, \ldots, \gamma_{n}\right\}
\end{aligned}
$$
不过这里采用**soft聚类**，即每一个节点可能隶属于不同的聚类，采用这种soft的形式可以使得整个矩阵是可微的。正式因为这里可能会属于多个聚类，因此需要后续的正则项.
$$
S_{i:}=\operatorname{softmax}\left(\mathbf{W}_{\mathbf{p}} \cdot \text { Aggregate }\left(A_{i j} * \vec{h}_{j}^{\prime} \mid j \in \mathcal{N}_{i}\right)\right)
$$
通过$S^T A S$可以得到池化图的邻接矩阵$A^*$.

可以重复计算上式，进行兴趣的分层压缩。

#### 正则项

$ \left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{n}^{\prime}\right\}$反映了交互的时间信息，而聚类后的$\left\{\vec{h}_{1}^{*}, \vec{h}_{2}^{*}, \ldots, \vec{h}_{m}^{*}\right\}$无法反应时间信息，因此提出了三种正则项。

![image-20220110205929277](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201102059426.png)

## Experiment

### Dataset

* Taobao: [数据集-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
* Kuaishou: 没找到

![image-20220110212326884](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201102123113.png)

### Evaluation Mertrics

* AUC
* GAUC
* MRR
* NDCG

