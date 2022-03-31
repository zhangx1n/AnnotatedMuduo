> SIGIR 2020 TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation
>
> 论文链接：https://arxiv.org/abs/2005.02844
> github：https://github.com/CRIPAC-DIG/TAGNN
>
> 来自中国科学院大学SRGNN的后序之作



## Abstract

之前的方法将会话压缩为一个固定的表示向量，而**没有考虑要预测的目标项**。之前的大多数方法的出发点就是通过对会话进行嵌入，期望能够捕获到用户意图信息完成对未来可能产生交互物品的预测。

本文的作者认为由于目标物品的多样性和用户的兴趣，**固定的向量会限制推荐模型的表示能力**。因此在本文中提出了一种新的目标注意图神经网络( TAGNN )模型用于基于会话的推荐。在TAGNN中，目标感知注意自适应地激活了用户对不同目标物品的不同兴趣。学习到的用户意图表示向量随目标物品的不同而变化，大大提高了模型的表达能力。此外，TAGNN利用图神经网络的力量来捕捉会话中的物品关系。

> **考虑一个序列：$s = v1 →v2 →v1 →v3.$ 之前的那些基于序列的方法会弄不清楚$v_1$ 和 $(v_2, v_3)$ 之间的关系**

本文的创新点如下：

1. TAGNN 将会话**建模成会话图**来捕获会话中物品的复杂关系，之后利用图神经网络的深度学习方法计算物品嵌入。
2. 为了适应用户在会话中不断变化的意图，**提出了一种针对于目标物品的注意力网络模型**，所提出的目标注意模块能够揭示特定目标物品下历史动作的相关性，进一步改善了会话表示。
   

## TAGNN

![image-20220324160933520](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220324160933520.png)

首先使用历史会话中的item来构建会话图。然后，利用图神经网络获取相应的embedding信息，以捕获基于会话图的复杂item transitions。考虑到item embeddings，我们采用目标感知的注意网络来激活特定的用户对目标物品的兴趣。在此之后，我们构建会话embedding。最后，对于每个会话，我们可以根据物品嵌入和会话嵌入来推断用户的下一步行动。

### Problem Statement and Constructing Session Graphs

在会话推荐任务中，一个匿名会话可以表示为一个列表 $s=\left[v_{s}, i\right]_{i=1}^{s_{n}}$ ，根据时间戳的不同进行排列，基于此可以通过 $V=\left\{v_{i}\right\}_{i=1}^{m}$ 作为会话中用户产生过交互物品的集合。会话推荐的核心任务是给定一个会话 $s$ 希望预测下一次最可能产生的交互对象 $v_{s, s_{n}+1 \text { 。 TAGNN }}$ 为所有候选物品产生一个概率排序列表，具有 top-k 概率值的物品将被选中进行推荐。

在 TAGNN 中选择将会话建模成有向图，表示为 $\mathcal{G}_{s}=\left(\mathcal{V}_{s}, \mathcal{E}_{s}, \mathcal{A}_{s}\right)$ ，其中 $\mathcal{V}_{s}, \mathcal{E}_{s}, \mathcal{A}_{s}$ 分别代表图节点集合、边的集合、邻接矩阵。图 中的每一个节点代表一个物品 $v_{s, i} \in V ， \mathcal{E}_{s}$ 代表用户连续与 $v_{s, i-1}$ 和 $v_{s, i}$ 产生交互。由于模型选择将会话序列建模为有向图，因此邻接 矩阵是出邻接矩阵 $\mathcal{A}^{(o u t)}$ 和入邻接矩阵 $\mathcal{A}^{(i n)}$ 的级联。并且在这两个邻接矩阵中反映了边的权重关系。此处构建会话图的方法与 SRGNN 相同。但是作者提到对于不同的基于会话的推荐场景，可以灵活地采用不同的构建会话图的机制。

### Learning Item Embeddings

首先将所有节点特征映射到同一个特征空间，同时也是转换维度的过程，经过图神经网络的结果为 $\mathbf{v}_{i} \in \mathbb{R}^{d}$ 是 $d$ 维的特征向量。然后可 以使用物品嵌入来表示每个会话 $s$ 。图神经网络( graph neural network, GNN )是一类应用广泛的深度学习模型。GNN 在图拓扑上生成节 点表示，图拓扑为复杂的物品连接建模。因此，它们特别适合基于会话的推荐。本文采用门控图神经网络来学习节点向量。其更新规则为:
$$
\begin{aligned}
\boldsymbol{a}_{s, i}^{(t)} &=A_{s, i:}\left[\boldsymbol{v}_{1}^{(t-1)}, \ldots, \boldsymbol{v}_{s_{n}}^{(t-1)}\right]^{\top} \boldsymbol{H}+\boldsymbol{b} \\
z_{s, i}^{(t)} &=\sigma\left(\boldsymbol{W}_{z} \boldsymbol{a}_{s, i}^{(t)}+\boldsymbol{U}_{z} \boldsymbol{v}_{i}^{(t-1)}\right) \\
\boldsymbol{r}_{s, i}^{(t)} &=\sigma\left(\boldsymbol{W}_{r} \boldsymbol{a}_{s, i}^{(t)}+\boldsymbol{U}_{r} \boldsymbol{v}_{i}^{(t-1)}\right) \\
\widetilde{\boldsymbol{v}_{i}^{(t)}} &=\tanh \left(\boldsymbol{W}_{o} \boldsymbol{a}_{s, i}^{(t)}+\boldsymbol{U}_{o}\left(\boldsymbol{r}_{s, i}^{(t)} \odot \boldsymbol{v}_{i}^{(t-1)}\right)\right) \\
\boldsymbol{v}_{i}^{(t)} &=\left(1-\boldsymbol{z}_{s, i}^{(t)}\right) \odot \boldsymbol{v}_{i}^{(t-1)}+\boldsymbol{z}_{s, i}^{(t)} \odot \widetilde{\boldsymbol{v}_{i}^{(t)}}
\end{aligned}
$$
其中 $t$ 代表卷积层数， $\mathbf{A}_{s, i} \in \mathbb{R}^{1 \times 2 n}$ 代表邻接矩阵的第 $i$ 行对应于节点 $v_{s, i}$ ，此处为 $2 n$ 的原因是 TAGNN 选择将会话建模成有向图， 因此邻接矩阵包含了出邻接矩阵和入邻接矩阵。 $\mathbf{H} \in \mathbb{R}^{d \times 2 d}, \mathbf{b} \in \mathbb{R}^{d}$ 分别代表可训练的权重矩阵和偏置矩阵。基于此我们可以得到会话 $s$ 的物品特征表示列表 $\left[\mathbf{v}_{\mathbf{1}}^{(\mathbf{t}-\mathbf{1})}, \ldots, \mathbf{v}_{\mathrm{s}_{\mathrm{n}}}^{(\mathbf{t}-\mathbf{1})}\right]_{\text {。 }} z_{s, i} \in \mathbb{R}^{d \times d}, r_{s, i} \in \mathbb{R}^{d \times d}$ 分别代表注意力机制的重置们和更新门。 $\sigma(\cdot)$ 代表非线性激活 函数，在本文中代表 Sigmoid 函数。 $\odot$ 代表点积。对于每一个会话图 $\mathcal{G}_{s}$ ，门控神经网络在相邻节点之间传播信息。更新和重置门分别决 定哪些信息被保留和丢弃。

### Constructing Target-Aware Embeddings

以前的工作只使用会话内的物品表示来捕获用户的兴趣。在 TAGNN 中获得了每个物品的嵌入后，开始构建目标物品嵌入，自适应地考虑 与目标物品相关的历史行为的相关性。在这里将目标物品定义为所有要预测的候选物品。通常，用户给出的推荐物品的操作只匹配部分兴趣。为了模拟这一过程，作者设计了一个新的目标注意机制来计算与每个目标物品有关的会话中所有项目的软注意分数。
作者引入了一个局部目标注意模块来计算会话 $s$ 中所有物品 $v_{i}$ 与每个目标物品 $v_{t} \in V$ 之间的注意得分。首先，对每个节点-目标对应用 一个权重矩阵 $W \in \mathbb{R}^{d \times d}$ 计算注意力得分。然后使用 Softmax 函数将注意力的分标准化:
$$
\beta_{i, t}=\operatorname{softmax}\left(e_{i, t}\right)=\frac{\exp \left(\boldsymbol{v}_{t}^{\top} \boldsymbol{W} \boldsymbol{v}_{i}\right)}{\sum_{j=1}^{m} \exp \left(\boldsymbol{v}_{t}^{\top} \boldsymbol{W} \boldsymbol{v}_{j}\right)}
$$
最后对于每一个会话 $s$ ，用户对于目标物品 $v_{t}$ 的兴趣可以表示为 $s_{\text {target }}^{t} \in \mathbb{R}^{d}$ ，表示形式如下:
$$
\boldsymbol{s}_{\text {target }}^{t}=\sum_{i=1}^{s_{n}} \beta_{i, t} \boldsymbol{v}_{i}
$$
所得到的表示用户兴趣的目标嵌入随目标物品的不同而不同。

### Generating Session Embeddings

至此已经得到了会话中物品的嵌入表示和对于每一个会话中用户对于不同目标物品的兴趣嵌入，之后进一步使用会话 $s$ 中涉及的节点表示生成用户在当前会话 $s$ 中显示的短期和长期偏好。

#### Local embedding

用户偏好的局部嵌入可以近似认为是会话中与用户产生交互的最后一个物品，所以简单地将用户的短期偏好表示为一个局部嵌入的 $s_{\text {local }} \in \mathbb{R}^{d}$ 也就是作为最后产生交互的物品 $v_{s, s_{n}}$ 的表示。

#### Global embedding

通过聚集所有涉及的节点向量，将用户的长期偏好表示为全局嵌入 $s_{g l o b a l} \in \mathbb{R}^{d}$ 。作者在此处采用另一种软注意机制，使上次访问的物品 与会话中所涉及的每一个物品之间具有相关性:
$$
\begin{aligned}
\alpha_{i} &=\boldsymbol{q}^{\top} \sigma\left(\boldsymbol{W}_{1} \boldsymbol{v}_{s_{n}}+\boldsymbol{W}_{2} \boldsymbol{v}_{i}+\boldsymbol{c}\right), \\
\boldsymbol{s}_{\text {global }} &=\sum_{i=1}^{s_{n}} \alpha_{i} \boldsymbol{v}_{i}
\end{aligned}
$$
其中 $\mathbf{q}, \mathbf{c} \in \mathbb{R}^{d}$ 并且 $\mathbf{W}_{\mathbf{1}}, \mathbf{W}_{\mathbf{2}} \in \mathbb{R}^{d \times d}$ 代表权重参数。

#### Session embedding

基于上述步骤，我们可以根据用户目标意图嵌入，局部意图嵌入，全局意图嵌入生成最终的会话嵌入表示，表示方式就是三者的级联加上 变换矩阵:
$$
\boldsymbol{s}_{t}=W_{3}\left[s_{\text {target }}^{t} ; s_{\text {local }} ; s_{\text {global }}\right]
$$
其中 $\mathbf{W}_{3} \in \mathbb{R}^{d \times 3 d}$ 将三个向量投影到一个嵌入空间 $\mathbb{R}^{d}$ 。此处需要注意的是对于每一个目标物品，生成了不同的会话嵌入表示

### Making Recommendation

在得到所有的物品嵌入和会话嵌入后，通过物品嵌入 $v_{t}$ 与会话表示 $s$ 的内积，计算每个目标物品 $v_{t} \in V$ 的推荐分数 $\hat{z}_{t}$ 。接下来，对所 有目标物品的末归一化分数 $z$ 使用 Softmax 函数，得到最终的输出向量:
$$
\begin{aligned}
\hat{z_{t}} &=\boldsymbol{s}_{t}^{\top} \boldsymbol{v}_{t} \\
\hat{\boldsymbol{y}} &=\operatorname{softmax}(\hat{z})
\end{aligned}
$$
其中 $\hat{\mathbf{y}} \in \mathbb{R}^{m}$ 表示物品在 $s$ 中作为下一个交互对象的概率。在 $\hat{\mathbf{y}}$ 中具有 top- $\mathrm{k}$ 概率的物品将被选为推荐物品。
loss为预测和标签的交叉熵:
$$
\mathcal{L}(\hat{\boldsymbol{y}})=-\sum_{i=1}^{m} \boldsymbol{y}_{i} \log \left(\hat{\boldsymbol{y}}_{i}\right)+\left(1-\boldsymbol{y}_{i}\right) \log \left(1-\hat{\boldsymbol{y}}_{i}\right)
$$
$\mathbf{y}$ 为标签项的 one-hot 编码向量。使用反向传播(BPTT)算法来训练所提出的模型。

## Experiments

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/20210219201137826.png)

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/20210219201137876.png)

