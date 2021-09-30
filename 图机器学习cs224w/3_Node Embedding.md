# Node Embedding

## Why Node Embedding



<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291600321.png" alt="image-20210929160046209" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291601304.png" alt="image-20210929160108208" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291604228.png" alt="image-20210929160410161" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291604125.png" alt="image-20210929160436040" style="zoom:50%;" />

## Example Node Embedding

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291606717.png" alt="image-20210929160634634" style="zoom:50%;" />

> **Image from：[Perozzi et al.](https://arxiv.org/pdf/1403.6652.pdf) DeepWalk: Online Learning of Social Representations. KDD 2014**

## Setup

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291609043.png" alt="image-20210929160934971" style="zoom:50%;" />

$$A$$是一个邻接矩阵。

## Embedding Nodes

### Goal

​			Goal is to encode nodes so that similarity in  the embedding space (e.g., dot product)  approximates similarity in the graph

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291613039.png" alt="image-20210929161309944" style="zoom:50%;" />

### Encoder && Decoder framework

  		1. **Encoder**：目标是将每个Node映射编码成低维的向量表示，或embedding。
  		2. **Decoder**：目标是利用Encoder输出的Embedding，来解码关于图的结构信息,比如输出一个实数，衡量两个Node在原始Graph中的相似性。

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291618046.png" alt="image-20210929161827950" style="zoom:50%;" />



3. **“Shallow" Encoding** ——一个Encoder的最简单的例子

    <img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291625339.png" alt="image-20210929162506249" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291625386.png" alt="image-20210929162537315" style="zoom:50%;" />

​	其实就是和word2vec类似，$v$是独热编码。$Z$是向量矩阵，$Z⋅v$就是对应$id$的Node Embedding

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291632497.png" alt="image-20210929163239413" style="zoom:50%;" />

4. **Framework Summary**

    ![image-20210929163326393](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291633472.png)



## Random walks

### How to Define Node Similarity

​		We will now learn node similarity definition that uses  ***random walks***, and how to optimize embeddings for  such a similarity measure.

![image-20210929163832332](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291638428.png)

### Notation

![image-20210929164151030](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291641122.png)

### Random Walk

![image-20210929172815206](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291728336.png)

### Random-Walk Embeddings

![image-20210929174113154](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291741234.png)

![image-20210929174129897](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291741987.png)

![image-20210929190634024](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291906134.png)

![image-20210929191121884](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291911990.png)

![image-20210929191216380](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291912483.png)

​		对这个目标函数的理解是：对节点$u$，我们希望其表示向量对其random walk neighborhood $N_R ( u )$ 的节点是predictive的（可以预测到它们的出现）


### Random Walk Optimization

![image-20210929211458466](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292114603.png)

![image-20210929211544421](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292115512.png)

​		这个计算概率$P\left(v \mid \mathbf{z}_{u}\right)$选用softmax的intuition就是前文所提及的，softmax会使最大元素输出靠近1，也就是在节点相似性最大时趋近于1。


![image-20210929212052747](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292120839.png)

​		但是计算这个公式**代价很大**，因为需要内外加总2次所有节点，复杂度达$O (∣V∣^2)$：


<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292123960.png" alt="image-20210929212304895" style="zoom:50%;" />

​		我们发现问题就**在于用于softmax归一化的这个分母**：

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292124781.png" alt="在这里插入图片描述" style="zoom:50%;" />

### Negative Sampling

为了解决这个分母，我们使用**negative sampling**的方法：简单来说就是原本我们是用所有节点作为归一化的负样本，现在我们只抽出一部分节点作为负样本，通过公式近似减少计算. 

**参考资料**：[word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](https://arxiv.org/pdf/1402.3722.pdf)

![Negative Sampling](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292248196.png)

![image-20210929225104549](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292251666.png)

### Stochastic Gradient Descent

![image-20210929225250863](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292252951.png)

### Summary

![image-20210929225329696](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292253794.png)

### How should we randomly walk?

 * So far we have described how to optimize  embeddings given a random walk strategy $R$

 * **What strategies should we use to run these  random walks?**

    > Reference: Perozzi et al. 2014. DeepWalk: [Online Learning of Social Representations.](https://arxiv.org/pdf/1403.6652.pdf) KDD.

     * Simplest idea: Just run fixed-length, unbiased  random walks starting from each node( [i.e.,  DeepWalk from Perozzi et al., 2013](https://arxiv.org/abs/1403.6652) )

         * The issue is that such notion of similarity is too constrained

            相似度概念受限

* **How can we generalize this?**

    * node2vec

        > Reference: Grover et al. 2016. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1403.6652). KDD.

        ![image-20210929230617629](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109292306740.png)

​	

## node2vec

> Reference: Grover et al. 2016. [node2vec: Scalable Feature Learning for Networks]([node2vec-kdd16.pdf (stanford.edu)](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)). KDD.

node2vec是一种综合考虑DFS邻域和BFS邻域的graph embedding方法。简单来说，可以看作是deepwalk的一种扩展，是结合了DFS和BFS随机游走的deepwalk。[【Graph Embedding】node2vec：算法原理，实现和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/56542707)

![image-20210930103917171](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301039281.png)

![image-20210930103954494](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301039597.png)

![image-20210930104138286](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301041362.png)

![image-20210930104225921](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301042016.png)

![image-20210930104315036](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301043115.png)

![image-20210930104359406](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301043504.png)

### Summary

![image-20210930104630046](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301046141.png)

## Other Random Walk Ideas

* **Different kinds of biased random walks:**
    * Based on node attributes ([Dong et al., 2017](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)).
    * Based on learned weights ([Abu-El-Haija et al., 2017](https://arxiv.org/abs/1710.09599))

* **Alternative optimization schemes:**
    * Directly optimize based on 1-hop and 2-hop random walk  probabilities (as in [LINE from Tang et al. 2015](https://arxiv.org/abs/1503.03578)).

* **Network preprocessing techniques:**
    * Run random walks on modified versions of the original  network (e.g., [Ribeiro et al. 2017’s struct2vec, Chen et al.  2016’s HARP](https://arxiv.org/abs/1706.07845)).

![image-20210930105147411](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301051490.png)

![image-20210930105226150](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301052233.png)

# Embedding Entire Graphs

![image-20210930105338105](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301053193.png)

## Approach 1: 聚合（加总或求平均）节点的嵌入



![image-20210930105454896](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301054980.png)

> [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292)

## Approach 2: 虚拟节点

![image-20210930105814425](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301058517.png)

创造一个假节点（virtual node），用这个节点嵌入来作为图嵌入，这个virtual node和它想嵌入的节点子集（比如全图）相连

> [Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networksJ\]. arXiv preprint arXiv:1511.05493, 2015.](https://arxiv.org/pdf/1511.05493.pdf)

## Approach 3: anonymous walk embeddings

> Anonymous Walk Embeddings, ICML 2018 https://arxiv.org/pdf/1805.11921.pdf

![image-20210930110126741](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301101837.png)

​		这种做法会使具体哪些节点被游走到这件事不可知（因此匿名）

![image-20210930110246179](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301102272.png)

​	anonymous walks的个数随walk长度**指数级增长**：

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301107124.png" alt="image-20210930110713030" style="zoom:50%;" />

### idea 1

​		独立地在途中进行固定长度l的随机游走，然后利用对应长度的anonymous walk数目来表示graph的vector。

具体采样多少次，有个很美丽的数学公式可以计算。

![image-20210930110920991](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301109083.png)

### sampling anonymous walks

![image-20210930111022480](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301110579.png)

### idea 2：walk embeddings

对每次anonymous walk进行embedding嵌入，然后把每次anonymous walk的embedding连接起来，然后作为整个graph的embedding。

> Anonymous Walk Embeddings, ICML 2018 https://arxiv.org/pdf/1805.11921.pdf

![image-20210930111301354](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301113477.png)

![image-20210930111358792](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301113889.png)

![image-20210930111421840](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301114957.png)



![image-20210930111452276](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301114377.png)

## Summary

![image-20210930111551224](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301115314.png)

![image-20210930111610575](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301116697.png)

![image-20210930111623244](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301116349.png)
