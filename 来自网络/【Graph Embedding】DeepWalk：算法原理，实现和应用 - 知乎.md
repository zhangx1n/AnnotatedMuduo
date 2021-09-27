放假前在学校看了一点graph embedding的东西，后面打算大概用几篇文章做一些简单的记录。﻿Deepwalk之前的一些经典方法就不写了(我也没看)，按师兄的话说就是可以适当的放下历史的包袱。时间有限，我们要向前看，拥抱未来(其实现在还是在看一两年前的东西，想跟上潮流真的是难~)。

本文首先从整体介绍一下图表示学习，然后分别从原理，核心代码，应用三个部分介绍DeepWalk。

图表示学习
-----

我们都知道在数据结构中，图是一种基础且常用的结构。现实世界中许多场景可以抽象为一种图结构，如社交网络，交通网络，电商网站中用户与物品的关系等。

目前提到图算法一般指：

1\. 经典数据结构与算法层面的：最小生成树(Prim,Kruskal,...)，最短路(Dijkstra,Floyed,...)，拓扑排序，关键路径等

2\. 概率图模型，涉及图的表示，推断和学习，详细可以参考Koller的书或者公开课

3\. 图神经网络，主要包括Graph Embedding(基于随机游走)和Graph CNN(基于邻居汇聚)两部分。

这里先看下Graph Embedding的相关内容。

Graph Embedding技术将图中的节点以低维稠密向量的形式进行表达，要求在原始图中**相似**(不同的方法对相似的定义不同)的节点其在低维表达空间也接近。得到的表达向量可以用来进行下游任务，如节点分类，链接预测，可视化或重构原始图等。

DeepWalk 算法原理
-------------

虽然DeepWalk是KDD 2014的工作，但却是我们了解Graph Embedding无法绕过的一个方法。

我们都知道在NLP任务中，word2vec是一种常用的word embedding方法，word2vec通过语料库中的句子序列来描述词与词的共现关系，进而学习到词语的向量表示。

DeepWalk的思想类似word2vec，使用**图中节点与节点的共现关系**来学习节点的向量表示。那么关键的问题就是如何来描述节点与节点的共现关系，DeepWalk给出的方法是使用随机游走(RandomWalk)的方式在图中进行节点采样。

RandomWalk是一种**可重复访问已访问节点的深度优先遍历**算法。给定当前访问起始节点，从其邻居中随机采样节点作为下一个访问节点，重复此过程，直到访问序列长度满足预设条件。

获取足够数量的节点访问序列后，使用skip-gram model 进行向量学习。

![](https://pic1.zhimg.com/v2-fdcb0babe4a168df85243b548fd86d30_b.jpg)

DeepWalk 核心代码
-------------

DeepWalk算法主要包括两个步骤，第一步为随机游走采样节点序列，第二步为使用skip-gram modelword2vec学习表达向量。

①构建同构网络，从网络中的每个节点开始分别进行Random Walk 采样，得到局部相关联的训练数据； ②对采样数据进行SkipGram训练，将离散的网络节点表示成向量化，最大化节点共现，使用Hierarchical Softmax来做超大规模分类的分类器

Random Walk
-----------

我们可以通过并行的方式加速路径采样，在采用多进程进行加速时，相比于开一个进程池让每次外层循环启动一个进程，我们采用固定为每个进程分配指定数量的`num_walks`的方式，这样可以最大限度减少进程频繁创建与销毁的时间开销。

`deepwalk_walk`方法对应上一节伪代码中第6行，`_simulate_walks`对应伪代码中第3行开始的外层循环。最后的`Parallel`为多进程并行时的任务分配操作。

```python
def deepwalk_walk(self, walk_length, start_node):

    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(self.G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

def _simulate_walks(self, nodes, num_walks, walk_length,):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:           
            walks.append(self.deepwalk_walk(alk_length=walk_length, start_node=v))
    return walks

results = Parallel(n_jobs=workers, verbose=verbose, )(
    delayed(self._simulate_walks)(nodes, num, walk_length) for num in
    partition_num(num_walks, workers))

walks = list(itertools.chain(*results))
```

Word2vec
--------

这里就偷个懒直接用`gensim`里的Word2Vec了。

```python
from gensim.models import Word2Vec
w2v_model = Word2Vec(walks,sg=1,hs=1)
```

DeepWalk应用
----------

这里简单的用DeepWalk算法在wiki数据集上进行节点分类任务和可视化任务。 wiki数据集包含 2,405 个网页和17,981条网页之间的链接关系，以及每个网页的所属类别。

本例中的训练，评测和可视化的完整代码在下面的git仓库中，后面还会陆续更新**line,node2vec,sdne,struc2vec**等graph embedding算法以及一些GCN算法

```python
G = nx.read_edgelist('../data/wiki/Wiki_edgelist.txt',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])

model = DeepWalk(G,walk_length=10,num_walks=80,workers=1)
model.train(window_size=5,iter=3)
embeddings = model.get_embeddings()

evaluate_embeddings(embeddings)
plot_embeddings(embeddings)
```

分类任务结果
------

micro-F1 : 0.6674

macro-F1 : 0.5768

可视化结果
-----

![](https://pic2.zhimg.com/v2-92b7d828057a9f4719bedb5122863055_b.jpg)

* * *

本系列其他相关文章

参考资料
----

1.  [Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations\[C\]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.](https://link.zhihu.com/?target=http%3A//www.perozzi.net/publications/14_kdd_deepwalk.pdf)
2.  [Taylor Wu：Graph Neural Network Review](https://zhuanlan.zhihu.com/p/43972372)

为了方便大家学习，我把一些相关的经典文章和代码实现进行了打包汇总并放在了github仓库里，感兴趣的同学可以看看～