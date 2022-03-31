> Title：Global Context Enhanced Graph Neural Networks for Session-based Recommendation
>
> Author：Ziyang Wang，Wei Wei (通讯作者，来自华中科技大学),
>
> 文章发表在SIGIR 20



## 摘要

几乎所有存在的基于会话的推荐拟合用户的偏好仅仅基于当前会话，而不考虑其他会话，对于当前会话可能包括很多相关和不相关的item转换。本文提出一个新的方法，全局背景提升GNN，也就是GCEGNN，来发掘item在所有会话中的转换。特别的，GCEGNN学习两种item embedding，分别从会话图和全局图。会话图中，通过拟合pairwise item转换来学习会话级别item embedding；全局图中，则是全部的会话。在GCEGNN中，提出的新的全局级别item表示学习层是采用会话意识注意力机制递归整合全局节点邻近点的embedding。同时也设计了一个会话级别的item表示学习层，采用的是一个GNN。另外，GCEGNN通过软注意力机制聚合学习到的表达。试验证实方法不错。

## 背景

此外，几乎所有之前的研究都只基于当前会话对用户偏好进行建模，而忽略了来自其他会话的有用的***item-transition patterns***。据我们所知，**[CSRM](https://ilps.science.uva.nl/wp-content/papercite-data/pdf/wang-2019-collaborative.pdf)**是唯一一个整合了来自最近m个会话的协作信息的工作，以端到端方式丰富当前会话的表示。CSRM将会话视为最小粒度，并度量当前和最新m个会话之间的相似性，以提取协作信息。CSRM通过记忆网络将距离当前会话时间最近的m个会话中包含的相关信息进行建模，从而来获得更为准确的会话表示，以提高会话推荐的性能。然而，它可能会将其他会话的相关和不相关信息编码到当前会话嵌入中，这甚至会降低性能。

![image-20220316163153216](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316163153216.png)

在Figure 1中：

假设当前会话为session 2，基于session的推荐旨在推荐与Iphone相关的配件。从下图我们观察到:(i)利用其他session的item-transition可能有助于建模当前session的用户首选项。例如，我们可以从session 1和session 3中找到session 2相关的两两物品转换信息，例如，一个新的两两物品转换[Iphone，手机壳];(ii)直接利用整个其他session的item-transition信息，当该session编码的部分item-transition信息与当前session无关时，可能会引入噪声。

![image-20220316164712314](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316164712314.png)

## 模型

### Session graph & Global graph的构建

GCE-GNN提出构建全局图和局部图来更好的利用会话信息。如下图所示，a为局部图构造方式：一个会话构成一个局部图，局部图中的每条边都代表会话中两个相邻的项，此外局部图还包含自连接边。而全局图则在不同会话之间建立联系。如图b，对s1,s2,s3建立全局图，对于会话中的每一个物品k，以k为中心建立大小为 $ξ$ 的窗口，窗口中的其他元素与k在全局图中相连，将出现次数作为每条边的权重。（**作者在b图右侧的全局图构建出错了，v1的邻接点应该是v2、v3、v4**）

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316164932779.png" alt="image-20220316164932779"  />

![image-20220316164952227](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316164952227.png)

### 物品表征

#### 全局物品表征

全局图的attetion计算如下：

通过注意力机制进行消息传递，这里称作session-aware attention。
$$
\mathbf{h}_{\mathcal{N}_{v_{i}}^{g}}=\sum_{v_{j} \in \mathcal{N}_{v_{i}}^{g}} \pi\left(v_{i}, v_{j}\right) \mathbf{h}_{v_{j}} \\

\pi\left(v_{i}, v_{j}\right)=\mathbf{q}_{1}^{T} \operatorname{LeakyRelu}\left(\mathbf{W}_{1}\left[\left(\mathbf{s} \odot \mathbf{h}_{v_{j}}\right) \| w_{i j}\right]\right) \\

\mathbf{s}=\frac{1}{|S|} \sum_{v_{i} \in S} \mathbf{h}_{v_{i}} \\

\pi\left(v_{i}, v_{j}\right)=\frac{\exp \left(\pi\left(v_{i}, v_{j}\right)\right)}{\sum_{v_{k} \in \mathcal{N}_{v_{i}}^{g}}^{g} \exp \left(\pi\left(v_{i}, v_{k}\right)\right)}
$$
$w_{ij}$ 是边 $(v_i, v_j)$ 的权重，$W_1, q_1$ 是可训练的参数，**$s$ 可以看做是当前会话的特征，它是通过计算当前会话的项目表示的平均值得到的。**

#### 局部物品表征

会话图的attention计算：
$$
e_{i j}=\operatorname{LeakyReLU}\left(\mathbf{a}_{r_{i j}}^{\top}\left(\mathbf{h}_{v_{i}} \odot \mathbf{h}_{v_{j}}\right)\right)
$$

$$
\alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}_{r_{i j}}^{\top}\left(\mathbf{h}_{v_{i}} \odot \mathbf{h}_{v_{j}}\right)\right)\right)}{\sum_{v_{k} \in \mathcal{N}_{v_{i}}^{s}} \exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}_{r_{i k}}^{\top}\left(\mathbf{h}_{v_{i}} \odot \mathbf{h}_{v_{k}}\right)\right)\right)}
$$
$$
\mathbf{h}_{v_{i}}^{s}=\sum_{v_{j} \in \mathcal{N}_{v_{i}}^{s}} \alpha_{i j} \mathbf{h}_{v_{j}}
$$

在聚合的时候加上drop防止过拟合。然后再与局部表征相加得到最终全局与局部相结合的物品表征
$$
\begin{aligned}
\mathbf{h}_{v}^{g,(k)} &=\operatorname{dropout}\left(\mathbf{h}_{v}^{g,(k)}\right) \\
\mathbf{h}_{v}^{\prime} &=\mathbf{h}_{v}^{g,(k)}+\mathbf{h}_{v}^{s}
\end{aligned}
$$

#### 反转位置信息

然而，发现用户的主要目的和过滤噪声才是重要的，因此整合**反转位置信息**（reversed position information）和会话信息效果可能会更好。喂给gnn一个会话序列后，得到item的表达。同时使用一个**可学习的位置embedding矩阵$P = [p_1, p_2, ..., p_l ]$**, $l$ 是会话的序列长度。通过拼接和非线性转换整合位置信息
$$
\mathbf{z}_{i}=\tanh \left(\mathbf{W}_{3}\left[\mathbf{h}_{v_{i}^{s}}^{\prime} \| \mathbf{p}_{l-i+1}\right]+\mathbf{b}_{3}\right)
$$
$W_3, b_3$ 是可以学习的参数。

采用反转位置embedding的原因是会话的序列长度是不固定的，对比正向位置信息，当前item到预测的item之间的距离包含更多有效的信息。例如会话中${v_2 → v_3 →?}$，v3是第2个，对于预测的item很有影响，但对于会话${v_2 → v_3 → v_5 → v_6 → v_8 →?}$,v3的影响就不是那么大了，很小了。因此，相反的位置信息更准确,。



通过计算会话中item的平均得到会话表达：
$$
\mathbf{s^`}=\frac{1}{|S|} \sum_{v_{i} \in S} \mathbf{h^`}_{v_{i}}
$$
接下来，我们通过软注意机制学习相应的权重
$$
\beta_{i}=\mathbf{q}_{2}^{\top} \sigma\left(\mathbf{W}_{4} \mathbf{z}_{i}+\mathbf{W}_{5} \mathbf{s}^{\prime}+\mathbf{b}_{4}\right) \text {, }
$$
where $\mathbf{W}_{4}, \mathbf{W}_{5} \in \mathbb{R}^{d \times d}$ and $\mathbf{q}_{2}, \mathbf{b}_{4} \in \mathbb{R}^{d}$ are learnable parameters.
$$
\mathrm{S}=\sum_{i=1}^{l} \beta_{i} \mathbf{h}_{v_{i}^{s}}^{\prime}
$$
会话表示S由当前会话中涉及的所有item构建，其中每个item的贡献不仅由会话图中的信息决定，还由序列中的时间顺序决定。

### 预测层

和其他item做内积，然后使用的是交叉熵

## 实验

### 数据集

![image-20220316203714976](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316203714976.png)

### 评估指标

* $P@N$ 

* $MRR@N$

### Baseline

* POP: It recommends top-𝑁 frequent items of the training set.
* Item-KNN: It recommends items based on the similarity betweenitems of the current session and items of other ones.
* FPMC: It combines the matrix factorization and the first-order Markov chain for capturing both sequential effects and user preferences. By following the previous work, we also ignore the user latent representations when computing recommendation scores.
* GRU4Rec: It is RNN-based model that uses Gated Recurrent Unit (GRU) to model user sequences.
* NARM : It improves over GRU4Rec by incorporating attentions into RNN for SBR.
* STAMP: It employs attention layers to replace all RNN encoders in previous work by fully relying on the self-attention of the last item in the current session to capture the user’s short-term
    interest.
* SR-GNN: It employs a gated GNN layer to obtain item embeddings, followed by a self-attention of the last item as STAMP does to compute the session level embeddings for session-based recommendation.
* CSRM: It utilizes the memory networks to investigate the latest 𝑚 sessions for better predicting the intent of the current session.
* FGNN: It is recently proposed by designing a weighted attention graph layer to learn items embeddings, and the sessions for the next item recommendation are learnt by a graph level feature extractor.

### 结果

![image-20220316204220213](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220316204220213.png)

