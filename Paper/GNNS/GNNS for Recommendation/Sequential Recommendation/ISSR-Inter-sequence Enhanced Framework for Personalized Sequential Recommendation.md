> 论文：[Inter-sequence Enhanced Framework for Personalized Sequential Recommendation](https://arxiv.org/pdf/2004.12118.pdf)
>
> 作者：哈尔滨工业大学
>
> 来源：AAAI 2020

## Motivation

建模用户历史交互的顺序相关性是进行顺序推荐的关键。然而，**大多数的方法主要关注单个序列内的序列项相关性，而忽略了不同用户交互序列间的序列项相关性**。虽然一些研究已经意识到这个问题，但他们的方法要么简单，要么含蓄。为了更好地利用这些信息，我们提出了一个序列间增强的序列推荐(ISSR)框架。**在ISSR中，既考虑了inter-sequence的item相关性，也考虑了intra-sequence的项相关性。**首先，在 inter-sequence correlation encoder中加入图神经网络，从user-item二部图和item-item图中捕获高阶item相关性;然后，在 inter-sequence correlation encoder的基础上，在intra-sequence correlation encoder中构建GRU网络和注意力网络，对每个个体序列内的item顺序相关性和时间动态进行建模，从而预测用户对候选项目的偏好。

![image-20220107105747436](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071057605.png)

* **intra-sequence item correlation：**用户交互项目序列中两个item的序列依赖性。例如**1721和1682**

* **inter-sequence item correlation：**两个items在不同用户序列中同时出现，并且在这两个items之间存在一条带有中间节点的路径。例如**1721和1784**

    

## Solution

![image-20220107110749748](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071107885.png)

### Inter-sequence Item Correlation Encoder

![image-20220107111814240](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071118333.png)

建模了两个图：**user-item bipartite graph, item-item co-occurence graph**（这是第一篇融合这两个图的工作）

ISSR利用了来自user-item二部图和item-item共现图的item相关性。此外，ISSR还考虑了残差连接，以保持原有的item表示。最后，通过融合三种类型的项目信息，生成集成的item表示。

**user-item bipartite graph：**$GCN_B$

**item-item co-occurence graph：**一条边连接两个节点，这两个节点在某一用户的行为序列中相邻。一条边的权值表示这两项在用户行为序列中出现的次数，然后可以利用该权值进行邻域采样。这个表示为$GCN_C$

### Intra-sequence Item Correlation Encoder

$$
\begin{aligned}
&a_{u_{i} v_{j}}^{\prime}=\mathbf{W}_{1} \sigma\left(\mathbf{W}_{2}\left(\left[\mathbf{e}_{u_{i}} ; \mathbf{h}_{v_{j}}\right]\right)+\mathbf{b}_{2}\right)+\mathbf{b}_{1} \\
&a_{u_{i} v_{j}}=\frac{\exp \left(a_{u_{i} v_{j}}^{\prime}\right)}{\sum_{1 \leq t \leq L} \exp \left(a_{u_{i} v_{t}}^{\prime}\right)}
\end{aligned}
$$

$$
\mathbf{s}_{u}^{\prime}=\sum_{1 \leq j \leq L} a_{u_{i} v_{j}} \mathbf{h}_{v_{j}}
$$

$$
\mathbf{s}_{u}=\mathbf{W}_{h}\left[\mathbf{s}_{u}^{\prime} ; \mathbf{h}_{L}\right]
$$

### Prediction Decoder

$$
\hat{y}_{i}=\operatorname{softmax}\left(\mathbf{s}_{u_{i}}^{\top} \mathbf{e}_{v_{i}}\right)
$$

### Training

![image-20220107120529255](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071205398.png)

## Evalution

### Dataset

使用的数据集来源于[Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding (WSDM 2018)](https://arxiv.org/abs/1809.07426)

![image-20220107120731686](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071207751.png)

### Evaluation Metrics

$Recall$@$k,  nDCG$@$k,  HR$@$k $ and $ MRR$@$k $  for  $ k ∈ {5, 10} $

![image-20220107121439703](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071216930.png)

![image-20220107121744270](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071217397.png)

![](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071216930.png)