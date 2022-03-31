> 论文：[Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation](https://arxiv.org/pdf/2106.15814.pdf)
>
> 作者：Yang Li , Tong Chen , Yadan Luo , Hongzhi Yin , Zi Huang The University of Queensland
>
> 来源：IJCAI 2021

------

## Motivation

![image-20220107144650740](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071446815.png)

和ISSR考虑的一样，不能只考虑一个用户的序列，得考虑他们序列之间的交互。

在这些序列方法中，由于假设每个POI序列独立于其他序列，很难从所有分散的POI序列中捕获到这种高阶协同信号(见图1(b))，从而导致模型性能下降。因此，考虑到POL-POI转换数据通常稀疏，学习后的POI表示的表达性受到严重限制，阻碍了最终的推荐有效性。

为了利用这些相关数据的信息，作者提出了seq2graph的增广方式。

## Solution

![image-20220107151048107](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071510215.png)

### Seq2Graph Augmentation

虽然check-in POI序列通常很短，但序列中的每个POI可能出现在多个POI序列中。因此，为了整合更丰富的上下文信息和缓解数据的稀疏性，我们提出合作研究不同的 visited POIs与图扩充POI序列。由于序列中每个POI中相邻节点的数量可能会带来较高的计算成本和过多的噪声，我们在SGRec中提出了一个**动态的相邻节点采样**，该采样在每个训练epoch中获得一个节点子集用于后续计算。具体来说，如图2左边所示，给定一个序列$S_u$，我们首先得到与原始序列$S_u$相关联的所有邻居节点，然后从邻居集合中均匀采样一定比例的节点，生成图。Seq2Graph Augmentation中的**采样策略将在每个epoch中重新执行**。最后，生成了图扩充的POI序列的不同变体，将不同的跨序列上下文信息带入给定的签入序列。然后将增强序列输入我们提出的SGRec模型，学习POI表示和用户偏好。

### Category-Aware Graph Attention Layer

在GNN中引入类别感知 (category-awareness) 可以使模型在建模稀疏的POI转换的基础上，从POI类别之间更密集的顺序依赖关系中学习。在SGRec中，我们设计了一种分类感知的节点信息聚合注意机制 (**category-aware attention mechanism**)，将分类侧信息无缝地注入到POI embeddings中。

首先，我们为每个category wise transition relation $c_i \rightarrow c_j$ 创建一个embedding vector $\mathbf{r}_{c_{i} \rightarrow c_{j}} \in \mathbb{R}^{D}$

对于每个POI节点的$v_p \in G$，我们不是定义一个单独的类别序列，而是将POI嵌入与其对应的类别嵌入连接起来，以构建一个统一的节点嵌入:
$$
\mathbf{v}_{p}=\left[\mathbf{p} ; \mathbf{c}_{p}\right] \in \mathbb{R}^{2 D}
$$
$\mathbf p$和$\mathbf{c_p}$分别是节点$v_p$的POI 和category latent vector。

SGRec 和 其他 GNN-based sequential recommenders 的优点在于考虑到边缘信息的异质性，即POI类别之间的关联。

两个节点之间的边是一对分类(例如，食物→酒店)。GAT [Velickovic et al.， 2018]的工作证明，有选择地从目标节点的邻居中聚合信息有利于学习高质量的嵌入。然而，它是为同构图的节点设计的。在我们的图扩充POI序列中，在两个POI节点之间存在不同的边(即不同的分类关系)，量化节点之间的两两注意力将牺牲这些关键的上下文信息。因此，我们提出了一种新的注意网络，其中两个节点之间的相互作用附加于它们的边的性质。在我们的例子中，对于每个POI节点$v_p \in G$及其邻居$v_q$，其边$q→p$的特征对应于关系嵌入$\mathbf{r}_{c_{i} \rightarrow c_{j}}$。我们首先通过非线性变换将边特征注入到邻居节点嵌入中:
$$
\widetilde{\mathbf{v}}_{q}=\mathbf{W}_{a}\left[\mathbf{v}_{q}+\operatorname{MLP}\left(\left[\mathbf{v}_{q} ; \mathbf{r}_{c_{q} \rightarrow c_{p}}\right]\right)\right]
$$
然后计算注意力分数度量target node $v_p$ 和 邻居节点$v_q$的重要性：
$$
a\left(v_{p}, v_{q}\right)=\mathbf{W}_{g a t}^{\top}\left[\mathbf{W}_{b} \mathbf{v}_{p} ; \widetilde{\mathbf{v}}_{q}\right]+b_{\text {gat }}
$$

$$
\alpha_{q}=\frac{\exp \left(\operatorname{LeakyReLU}\left(a\left(v_{p}, v_{q}\right)\right)\right)}{\sum_{v_{m} \in \mathcal{N}\left(v_{p}\right)} \exp \left(\operatorname{LeakyReLU}\left(a\left(v_{p}, v_{m}\right)\right)\right)}
$$

$$
\mathbf{h}_{p}=\sum_{v_{q} \in \mathcal{N}\left(v_{p}\right)} \alpha_{q} \Phi \mathbf{v}_{q}
$$

### Position-aware Attention Net

假设有两个序列：p1 → p2 → p4 and p1 → p2 → p3 → p5 → p6 → p4.  在这两个序列中，很明显第一个序列中p2对p4的重要性要比第二个序列高。

position embedding：$\mathbf{q}_{n-i+1}$
$$
\begin{aligned}
b\left(v_{p_{i}}, v_{p_{n}}\right)=\mathbf{W}_{p a t}^{\top} \tanh \left(\mathbf{W}_{h 1}^{\top} \mathbf{h}_{p_{n}}\right.&+\mathbf{W}_{h 2}^{\top} \mathbf{h}_{p_{i}} \\
&\left.+\mathbf{W}_{q}^{\top} \mathbf{q}_{n-i+1}\right),
\end{aligned}
$$
最后，我们将用户的短期偏好编码为单个序列级嵌入向量s
$$
\mathbf{s}=\sum_{i=1}^{n} \beta_{i} \mathbf{h}_{p_{i}}, \quad \beta_{i}=\frac{\exp \left(b\left(v_{p_{i}}, v_{p_{n}}\right)\right)}{\sum_{m=1}^{n} \exp \left(b\left(v_{p_{m}}, v_{p_{n}}\right)\right)}
$$

### Next POI Category Prediction

我们首先估计每个POI被访问的概率:(一次全连接)
$$
\widehat{\mathbf{y}}^{\text {poi }}=\operatorname{softmax}\left(\mathbf{W}_{p}\left(\left[\mathbf{h}_{p_{n}} \circ \mathbf{s} ; \mathbf{u}\right]\right)+\mathbf{b}_{p}\right)
$$
$\mathbf{u}$: user‘s long-term preference embedding

然后估计next-POI 的类别：
$$
\widehat{\mathbf{y}}^{c a t}=\operatorname{softmax}\left(\mathbf{W}_{c}\left(\left[\mathbf{c}_{p_{n}} ; \mathbf{u}\right]+\mathbf{b}_{c}\right)\right.
$$
$c_{p_n}$:last POI's category embedding

### Traning

$$
\mathcal{L}=-\frac{1}{M} \sum_{m=1}^{M} \underbrace{\mathbf{y}_{m}^{p o i \top} \log \left(\widehat{\mathbf{y}}_{m}^{p o i}\right)}_{\text {next poi loss }}+\eta \underbrace{\mathbf{y}_{m}^{c a t \top} \log \left(\widehat{\mathbf{y}}_{m}^{\mathrm{cat}}\right)}_{\text {next category loss }}+\lambda\|\Psi\|^{2},
$$

## Evalution

### Dataset

* **Foursquare**:[Dingqi YANG's Homepage (google.com)](https://sites.google.com/site/yangdingqi/home)： 基于地理位置的社交网络

* **Gowalla:**[SNAP: Network datasets: Gowalla (stanford.edu)](http://snap.stanford.edu/data/loc-gowalla.html)

    Gowalla是一个基于地理位置的社交网站，用户可以通过签到来分享他们的地理位置。该网络是无向的，使用其公共API收集，由196,591个节点和950,327条边组成。在2009年2月到2010年10月期间，我们一共收集了6.442.890个用户的签到。

    ![image-20220107210021179](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201072100266.png)

### Evaluation Metrics

* $Hit Ratio$  (HR @ K)
* $Normalised \  Discounted \ Cumulative \  Gain$ (nDCG@K)

![image-20220107210554111](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201072105209.png)