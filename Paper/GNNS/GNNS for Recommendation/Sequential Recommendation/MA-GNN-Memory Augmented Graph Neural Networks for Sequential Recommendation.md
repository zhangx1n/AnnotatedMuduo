> 论文：[Memory Augmented Graph Neural Networks for Sequential Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/5945)
>
> 作者：华为诺亚实验室
>
> 来源：AAAI 2020 
>
> 视频：[ICDM2021-Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation - YouTube](https://www.youtube.com/watch?v=n32W2HAJrgQ)

------

## Motivation

传统的序列推荐方法主要建模的是session-level的短期序列，仅能够捕获蕴含在用户近期交互物品中的短期偏好，缺乏对用户长期偏好的挖掘。因此，本文的主要动机就是考虑用户长期的交互序列，希望能够从中挖掘出用户长期稳定的偏好。除此之外，作者还希望能够显式地建模物品与物品之间的共现关系。为此，作者提出了几种方法来分别捕获用户的短期兴趣，长期兴趣以及物品之间的共现关系，并融合这些因素进行打分和推荐。

- 短期兴趣：基于短期交互序列中物品之间的转移关系图，使用GNN来捕获用户的短期兴趣。
- 长期兴趣：使用带注意力机制的记忆网络来捕获用户的长期兴趣。
- 共现关系：使用双线性函数来显式建模物品之间的共现关系。

##  Contribulation

- 为了捕获用户的短期兴趣和长期兴趣，文章提出了一种**记忆增强的图神经网络**来捕获短期上下文信息和长距离依赖。
- 为了有效地融合短期兴趣和长期兴趣，文章采用了一种**门控机制**来自适应地融合两种兴趣表征。
- 为了**显式建模物品之间的共现关系**，文章采用了双线性函数来捕获物品之间相关性。
- 在五个真实的数据集上取得了state-of-the-art的效果。

## Solution

先从总体上介绍下整个方法。整个方法实际上很像矩阵分解那一套框架。只不过分解的时候考虑了短期兴趣和长期兴趣。这里头最重要的是理解**「输入序列数据怎么规整成矩阵分解的形式，即point-wise分解和打分」** (比如：用户嵌入和物品嵌入点积)。

原始的输入数据是user-level的序列。$S^u = (I_1, I_2,...,I_{|S_u|})$，由于是user-level的序列，每条序列长度很长，如果直接建模的话，总的样本量正比于用户数，相对较少。因此需要对序列进行切割来做**「数据增强」**和**「长短期序列」**区分。可以通过在user-level的序列数据上做窗口滑动来增强，窗口**「内部的子序列」**构成了**「短期序列」**，从窗口**「左侧端点开始向左到起始点」**的子序列构成了**「长期序列」**，从窗口**「右侧端点」**开始向右的子序列构成了**「目标序列」**。这里头有好几个超参数。**「滑动窗口的大小」**$L$（即：决定了短期序列的长度），滑动窗口的左端点起始值 **「$l$（即：决定了长期序列长度的最小值），以及」**目标序列的长度$|T|$ 。

形式化地，增强后的每个sample有3段子序列，即：$S_{u,l} = [H_{u,l};L_{u,l};T_{u,l}]$，$l$是滑动窗口的左侧端点，则：$H_{u,l} = (I_1,I_2,...,I_{l-1})$是长期交互序列，$L_{u,l} = (I_l,I_{l+1},...,I_{l+|L|-1})$是滑动窗口内部的长度为$L$短期交互序列，$T_{u,l}$是大小为$T$的目标序列。

则，本文的问题是输入长期序列$H_{u,l}$和短期序列$L_{u,l}$，来输出用户$u$接下来会感兴趣的$Top-K$个物品，并在目标序列$T_{u,l}$上进行评估。命中越多目标序列$T_{u,l}$中的物品，说明模型泛化性越好。所谓的长短期，实际上就是从物品交互时间的久远来衡量，最近交互的若干个物品构成了短期交互序列，再之前的交互构成了长期交互序列。

在解决序列推荐方法上，除了物品和序列**「表征的过程」**有所差异之外，目前主流的方法都是利用物品表征和用户表征，来预测next item，即：预测所有的$N$个物品上的概率分布，推荐$K$个概率最大的，实际上是个多分类问题。但是这篇文章将多分类转成了二分类问题，即：将**「目标序列」**$T_{u,l}$中的物品$i$和用户$u$作配对，转化成$(u,i)$ 正样本对，这样就可以使用矩阵分解的方式来拟合分数。此外，此处采样了负样本，即:$(u,i,j)$三元组，$j$是采样的负样本，最后用pair-wise的BPR损失来训练。总之，**「输入」**的用户**「短期序列和长期序列」**都只是为了获取某种刻画**「用户兴趣维度」**的表征，并**「基于多样化的用户兴趣表征来多维度地联合预测分数」**。

因此，问题的关键是如何捕获输入的短期和长期序列中蕴含的用户偏好。先总体看下该方法的架构示意图。

![image-20220106195308655](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201061953303.png)

最左侧是初始的兴趣表征模块，包含了用户通用兴趣表征，短期序列中的物品表征和长期序列中的物品表征。中间是兴趣建模模块，即：如果对初始的表征进行处理和融合；右侧是基于建模得到的兴趣表征进行分数的预测，包括了3个分数来源，通用兴趣贡献分，长短期融合兴趣贡献分以及物品共现分。
$$
\hat{r}_{u, j}=\boldsymbol{p}_{u}^{T} \boldsymbol{q}_{j}+\boldsymbol{p}_{u, l}^{C^{T}} \boldsymbol{q}_{j}+\frac{1}{|L|} \sum_{i \in L_{u, l}} \boldsymbol{e}_{i}^{T} \boldsymbol{W}_{r} \boldsymbol{q}_{j}
$$
其中，$\boldsymbol{p}_{u}$是用户的通用兴趣表征，$\boldsymbol{q}_{j}$是目标物品的初始表征，$\boldsymbol{p}_{u, l}^{C^{T}}$是用户的长短期兴趣融合表征，最后一项是目标物品和用户短期交互序列中的物品的共现分数。这三项分别对应着通用兴趣建模、短期和长期兴趣建模以及物品共现建模。

### 通用兴趣建模

输入的短期序列$S_{u,l}$和长期序列$H_{u,l}$都记录着产生该行为序列的用户$u$，因此作者在做序列建模的时候，将该用户$u$也考虑进去了。作者采用随机初始化的$\boldsymbol{p}_{u}$来表征用户静态和通用的的兴趣。最后在预测层预测分数的时候，采用了简单矩阵分解策略，即：$\boldsymbol{p}_{u}^{T} \boldsymbol{q}_{j}$，$\boldsymbol{q}_{j}$是目标预测物品的embedding（实际上就是目标序列集合中的物品），该分数即：通用兴趣贡献分。

### 短期兴趣建模（使用GNN）

在将图数据输入到模型之前，由于传统的图数据是没有顺序的，所以作者先将项目序列构建了一个可以输入到网络中的图。就是在构造一个项目序列的邻接矩阵。

输入是短期序列$S_{u,l}$，输出是蕴含在短期序列中用户的兴趣表征$\boldsymbol{p}_{u, l}^{S}$，是short-term的缩小。如图所示，左下角的部分。作者采用了两层的GNN网络来捕获蕴含在序列中的局部结构信息，并形成用户短期兴趣表征。为了能够用GNN来建模，需要将序列转成session graph。策略是，短期序列中的每个物品和其后面的3个物品做连接，并对形成的邻接矩阵按照行做归一化。如下图所示：

![image-20220106200352427](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201062003532.png)

**「信息传播和聚合」**：接着，基于该**「邻接矩阵」**来进行邻域信息传播和汇聚。即：
$$
\begin{gathered}
\mathbf{h}_{i}=\tanh \left(\mathbf{W}^{(1)} \cdot\left[\sum_{k \in \mathcal{N}_{i}} \mathbf{e}_{k} A_{i, k} ; \mathbf{e}_{i}\right]\right), \forall i \in L_{u, l}, \\

\end{gathered}
$$
$\sum_{k \in \mathcal{N}_{i}} \mathbf{e}_{k} A_{i, k}$是从邻域传播的信息, $e_k$: 初始的表征，类似features；和自身$e_i$做一个拼接($||$)，再过一个非线性变换。

上述得到了序列中每个物品的表征后，需要形成**用户的短期兴趣表征**。先mean pooling得到短期序列表征，再和用户的**「通用表征」**做一个拼接并过一层非线性变换融合。即：
$$
\mathbf{p}_{u, l}^{S}=\tanh \left(\mathbf{W}^{(2)} \cdot\left[\frac{1}{|L|} \sum_{i \in L_{u, l}} \mathbf{h}_{i} ; \mathbf{p}_{u}\right]\right)
$$

### 长期兴趣建模

这一部分是本文的**亮点**。

这部分的输入是长期序列$H_{u,l}$，输出是用户的**「长期兴趣表征」**。为了能够捕获长期兴趣，通常可以采用**「外部记忆单元」**来存储用户随时间变化的动态偏好，但是如果为每个用户都存储这样的偏好，会耗费很大的存储空间，而且通过这种方式捕获到的兴趣可能和通用兴趣$p_u$相似。为了解决这些问题，作者采用了一个记忆网络来存储**「所有用户共享的隐兴趣表征」**，每种隐单元都代表着某种特定的用户隐兴趣，给定用户长期交互序列，我们可以学习到多种**「不同兴趣融合」**的用户长期兴趣表征$H_{u,l}$。记长期序列中每个物品的表征形成的表征矩阵为：$\mathbf{H}_{u, l} \in \mathbb{R}^{d \times\left|H_{u, l}\right|}$，即：第$j$列为长期序列中第个$j$物品的表征向量。记忆网络中存储着所有用户**「共享的隐兴趣表征」**，针对每一个**「用户」**以及其**「长期交互序列」**，我们需要为该用户生成与其兴趣匹配的query embedding ，然后根据该query embedding $z_{u,l}$去记忆网络中检索有用的隐兴趣表征，从而形成该用户特定的长期兴趣表征。这里面最重要的就是query embedding的产生，作者采用了多维度的注意力机制。具体而言，

- 首先模仿Transformer给序列中每个item引入了位置语义信息，$PE()$为sinusoidal positional encoding function

$$
\mathbf{H}_{u, l} =\mathbf{H}_{u, l}+\operatorname{PE}\left(H_{u, l}\right), 
$$

- 计算用户通用兴趣表征$p_u$和长期序列$H_{u,l}$**「感知」**的多维度**注意力权重矩阵MDAtt**，即：，

$$
\mathbf{S}_{u, l}=\operatorname{softmax} \left(\mathbf{W}_{a}^{(3)} \tanh \left(\mathbf{W}_{a}^{(1)} \mathbf{H}_{u, l}+\left(\mathbf{W}_{a}^{(2)} \mathbf{p}_{u}\right) \otimes \mathbf{1}_{|\mathbf{H}_{u, l}|}\right)\right)
$$

其中，$W_{a}^{(1)}, W_{a}^{(2)} \in \mathbb{R}^{d \times d}, W_{a}^{(3)} \in \mathbb{R}^{h \times d}$是可学习的注意力参数，$\otimes$是外积操作，就普通的矩阵乘法。上述注意力机制考虑了用户的**「通用兴趣表征」**和**「长期行为序列」**，因此该注意力是general-interest and long-term sequence **「aware」**的。多维度注意力机制和通常的注意力机制其实差不太多。从语义上而言，$S_{u,l}$每一行向量从某个语义角度衡量了长期行为序列中每个物品在该语义上的权重值，softmax应该是按照每行来做的，即：求每个序列中每个物品在该语义下的概率分布；基于该行向量所代表的注意力概率分布对长期序列做加权汇聚，可以得到在该语义上的用户query表征；共行，则会形成个用户query表征向量，即形成表征矩阵$\mathbf{Z}_{u, l}。$

- 具体而言，根据上述的注意力权重矩阵来对用户长期行为序列做一个聚合，形成表征矩阵$\mathbf{Z}_{u, l}$。

$$
\mathbf{Z}_{u, l} =\tanh \left(\mathbf{S}_{u, l} \cdot \mathbf{H}_{u, l}^{\top}\right),
$$



- 对上述表征矩阵按照**「行方向」**(把h维度归约掉)做mean pooling来形成最终的用户query embedding，。

$$
\mathbf{z}_{u, l} =\operatorname{avg}\left(\mathbf{Z}_{u, l}\right)
$$

实际上从语义上来讲，相当于将不同语义汇聚到的query embedding通过mean pooling汇聚在一起形成最终的query embedding。

总之，通过上述步骤，就能够形成**「用户通用兴趣和长期行为序列感知」**的检索向量。接下来就是根据该检索向量去记忆网络中检索出和该用户兴趣就相关的记忆，从而形成用户的长期兴趣表征。

记忆网络的的Key和Value矩阵分别记为：$K, V$，每一列都代表着某个维度下，所有用户共享的**「隐兴趣表征向量」**。因此，需要计算用户的query embedding和每一种隐兴趣表征的亲和度值，并转成概率分布。
$$
s_{i}=\operatorname{softmax}\left(\boldsymbol{z}_{u, l}^{T} \times \boldsymbol{k}_{i}\right)
$$
基于该概率分布对所有的隐兴趣表征（列向量）做加权汇聚。
$$
\boldsymbol{o}_{u, l}=\sum_{i} s_{i} \boldsymbol{v}_{i}
$$
最后做个skip-connection，
$$
\begin{gathered}
\boldsymbol{p}_{u, l}^{H}=\boldsymbol{z}_{u, l}+\boldsymbol{o}_{u, l}
\end{gathered}
$$

### 长短期兴趣融合

使用门控机制来融合短期兴趣和长期兴趣。这里头的做法借鉴了LSTM/GRU，实际上和SR-GNN做结点信息更新的时候的策略是类似的，不作赘述。唯一要提的点就是，这里头实际上可以直接融合长短期序列表$p_{u,l}^S$征和$p_{u,l}^H$，但是作者实际用的时候融合的是，用户长期交互序列表征以$p_{u,l}^H$及$\sum_{i \in L_{u, l}} \boldsymbol{h}_{i}$。可能是因为$p_{u,l}^S$中融入了通用兴趣表征，而最后预测分数的时候，通用兴趣表征是单独作为一项贡献分的，再融合进长短期兴趣表征显得冗余。

做法很简单，门控的输出值是近期交互行为、通用兴趣表征、长期兴趣表征感知的，
$$
\boldsymbol{g}_{u, l}=\sigma\left(\boldsymbol{W}_{g}^{(1)} \cdot \frac{1}{|L|} \sum_{i \in L_{u, l}} \boldsymbol{h}_{i}+\boldsymbol{W}_{g}^{(2)} \boldsymbol{p}_{u, l}^{H}+\boldsymbol{W}_{g}^{(3)} \boldsymbol{p}_{u}\right)
$$
基于该门控值进行融合，得到的融合后的兴趣表征为：
$$
\boldsymbol{p}_{u, l}^{C}=\boldsymbol{g}_{u, l} \odot \frac{1}{|L|} \sum_{i \in L_{u, l}} \boldsymbol{h}_{i}+\left(\boldsymbol{I}_{d}-\boldsymbol{g}_{u, l}\right) \odot \boldsymbol{p}_{u, l}^{H}
$$

### 物品共现模型

显式地对用户短期交互过的物品和目标物品做共现建模，采用了双线性函数：
$$
\boldsymbol{e}_{i}^{T} \boldsymbol{W}_{r} \boldsymbol{q}_{j}
$$
$\boldsymbol{W}_{r}$是可学习的物品相关性矩阵。$\boldsymbol{e}_{i}$是短期交互序列中的物品$i$的初始表征，$\boldsymbol{q}_{j}$是目标物品。

最后用BPR Loss来学习。

![image-20220107102929090](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071029206.png)

## Evalution

### 数据集

**Amazon-Books，Amazon-CDs：**are adopted from the Amazon review dataset with different categories, i.e., CDs and Books, which cover a large amount of user-item interaction data, e.g., user ratings and reviews

**Goodreads-Children， Goodreads-Comics：**were collected in late 2017 from the goodreads website with a focus on the genres of Children and Comics.

**MovieLens-20M：**a user-movie dataset collected from the MovieLens website

![image-20220107101036315](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071010408.png)

**数据集的处理：**为了与the  implicit feedback setting保持一致，将评级不低于4分（满分5分的话）的评级作为正反馈，并将所有其他评级视为所有数据集上的缺失条目。为了过滤噪声数据，我们只保留至少有10个评级的用户和至少有10个评级的项目。

对于每个用户，我们使用用户序列中最早的70%的交互作为训练集，并使用接下来的10%的交互作为超参数调优的验证集。剩下的20%构成报告模型性能的测试集。请注意，在测试过程中，输入序列包括训练集和验证集中的交互。所有模型的学习进行5次，以报告平均结果。

### 评估矩阵

![image-20220107101241060](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071012154.png)





实验主要包括几个部分，

**对比实验**（方法包括：BPRMF，GRU4Rec，GRU4Rec+，GC-SAN，Caser，SASRec，MARank），居然没有选SR-GNN（虽然GC-SAN论文中战胜了SR-GNN，但是很多人在很多实践中发现SR-GNN比GC-SAN好）。

![image-20220107101304483](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071013591.png)

**消融实验**：主要考察了通用兴趣，通用兴趣+短期兴趣，通用兴趣+短期兴趣+长期兴趣+gating长短期融合，通用兴趣+短期兴趣+长期兴趣+concat长短期融合，通用兴趣+短期兴趣+长期兴趣+GRU长短期融合。

![image-20220107101342009](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071013097.png)

(3)和(6)对比可以看出共现建模的好处；(1)和(2)对比看出短期兴趣建模的好处；(3)和(4)和(5)的结果说明gating机制的有效性，但是这个结果太不可思议了，gating比concat以及GRU好这么多？gating和GRU的差异主要就是有没有用吧？为了公平性，可以直接用到GRU里面来对比的。对此表示疑惑。

**记忆单元的可视化：** 验证每个记忆单元是否可以表示某种特定的兴趣。作者在MovieLens上做了case study。作者随机选了某个用户以及他观看过的电影，用其观看的电影直接作为query embedding，去计算在memory network上的注意力分数，即。期望观察到的效果是，给定不同的电影query embedding，注意力分数分布不一样，在类似的电影上，某些维度的注意力分数也应该类似。

![image-20220107101555822](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201071015908.png)

可以看到有三部Three Colors的电影的注意力分数分布挺近似的。DIe Hard是惊悚片，和其他的分布不一样。

这种可视化应该是目前论文写作的标配。

## Summarization

这篇文章总体上有一些借鉴的地方。全文最大的亮点在于长期兴趣的建模。基于长期行为序列来构造query embedding，然后去memory network中检索有用的兴趣。这种长期兴趣的建模范式可以借鉴到日常工作优化中。但是缺点是长期序列长度可能比较长，多维度注意力机制可能复杂度相对高一些。



还有这个memory network 到底是个啥

##  
