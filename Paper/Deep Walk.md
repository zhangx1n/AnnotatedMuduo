> [DeepWalk: Online Learning of Social Representations]([1403.6652.pdf (arxiv.org)](https://arxiv.org/pdf/1403.6652.pdf))

## Abstract

**Input**: a graph

**Output**: a latent representations

------

Deep Walk uses local information obtained from **truncated random walks** to learn latent representations by treating walks as the equivalent of sentences.

DeepWalk使用从截断的随机漫步中获得的局部信息，通过将漫步视为等同于句子来学习潜在的表征。

![image-20210930095646048](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109300956201.png)

 *These latent representations encode social relations in a continuous vector space with a relatively small number of dimensions.*

## Learning Social Representations

learning social representations with the following **characteristics**:

* **Adaptability**

    图是持续进化改变的，新的边新的关系不应该要求再次学习

* **Community aware**

    应该学出图之间的结构关系，比如距离和相似度

* **Low dimensional**

* **Continuous**

## Random Walks

使用**short random walks**作为抽取网络局部信息的工具

*random walk开始 随机取起始点 规定路径长度 随机漫游graph*

**优点**：

	* 可以并行化
	* 抽取的局部信息，可以适应小的改变

![image-20210930103013173](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109301030245.png)

​		我们都知道在NLP任务中，word2vec是一种常用的word embedding方法，word2vec通过语料库中的句子序列来描述词与词的共现关系，进而学习到词语的向量表示。

​		DeepWalk的思想类似word2vec，使用**图中节点与节点的共现关系**来学习节点的向量表示。那么关键的问题就是如何来描述节点与节点的共现关系，DeepWalk给出的方法是使用随机游走(RandomWalk)的方式在图中进行节点采样。

​		RandomWalk是一种**可重复访问已访问节点的深度优先遍历**算法。给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，重复此过程，直到访问序列长度满足预设条件。

​		获取足够数量的节点访问序列后，使用skip-gram model 进行向量学习。