> 论文：Graph Convolutional Neural Networks for Web-Scale Recommender Systems
>
> 作者：Pinterest, Stanford University (William L. Hamilton , Jure Leskovec)
>
> 来源：KDD 2018 
>
> 论文链接：https://arxiv.org/pdf/1806.01973
>
> Github链接：[Graph Convolutional Neural Networks for Web-Scale Recommender Systems | Papers With Code](https://paperswithcode.com/paper/graph-convolutional-neural-networks-for-web)

------

## 简介

本文的工作与GraphSAGE紧密联系，避免了进行全图的计算以及去除了GraphSAGE需要将全图存放在GPU中的限制。

提出了一种数据高效的图卷积网络算法PinSage，同时组合了高效的随机游走以及图卷积来生成节点的embeddings（同时综合了图结构以及节点特征信息）。

![image-20220103214749558](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032147672.png)

模型的每个模块学习如何从小的邻居图上聚合信息，通过迭代这些模块，  可以从局部的网络拓扑结构中获取信息。这些局部卷积模块的参数在所有节点之间都共享，使得PinSage的方法的参数复杂度与输入图的大小无关。

### 贡献

PinSage大大提升了GCN的可扩展性：

* on-the-fly convolutions：传统的GCN通过多次图拉普拉斯矩阵来进行图卷积过程。PinSage则通过采样一个节点周围的邻居以及从采样邻居里动态构建计算图来完成卷积过程，减轻了需要在训练过程中运算全图的需求。参考GraphSAGE
* producer-consumer minibatch construction：我们构建的这种producer-consumer构架来建立minibatches的方法保证了在模型训练过程中的最大GPU使用率。一个大容量，GPU-bound producer有效地采样节点邻居以及获得必要特征来定义局部卷积，一个GPU-bound tensorflow模型使用这些预先定义的计算图来有效地运行随机梯度下降。
* efficient MapReduce inference：当给定好一个训练好的GCN模型，我们设计了一个有效的MapReduce pipeline来在减小计算量的情况下对数以亿计的节点生成embeddings。

除了对于模型的扩展性的提升，我们也具有其他训练方法以及算法的创新点，以提升被PinSage学习到的表征的质量，以及提升下流的推荐任务的表现：

* constructing convolutions via random walks：通过随机采样的方法生成节点的邻居节点是次优的，PinSage采用了short random walks来采样计算图。一个额外的收获是：每个节点现在都具有了一个重要性分数，可以被用于pooling/aggregation 步骤。
* importance pooling：基于随机游走相似度的测度我们引入了一种方法在aggregation的过程中来评判节点特征的重要性，获得了46%的性能提升。
* curriculum training：定义了一种课程学习的训练策略，来在训练过程中使用越来越难的样例，获得了12%的性能提升。

## 模型

![image-20220103215233651](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032152711.png)

该算法的结果是$u$的representation，它聚合了它自身的信息以及它局部的图邻居的信息。

**importance-based neighborhoods：**

在PinSage中，我们挖掘出了对节点$u$产生最大影响的$T$个节点。具体来说，模拟从节点 u 开始的随机游走，并计算随机游走访问的节点的 L1 归一化访问计数。节点u的邻居被定义为前T个节点具有最高的访问记数的节点。

这种基于重要性的邻居定义是两层的。首先，选择出合适的邻居数目，来限制训练的时候的空间使用量。其次，它允许算法1在聚合邻居的向量的时候考虑邻居的重要性

**stacking convolution：**

![image-20220103215656506](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032156610.png)

### 具体实例

**采样时只能选取真实的邻居节点吗？**如果构建的是一个与虚拟邻居相连的子图有什么优点？

PinSAGE 算法通过多次随机游走，按游走经过的频率选取邻居，例如下面以 0 号节点作为起始，随机进行了 4 次游走

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032200951.jpeg)

其中 5、10、11 三个节点出现的频率最高，因此我们将这三个节点与 0 号节点相连，作为 0 号节点的虚拟邻居

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032201106.jpeg)

采样时选取虚拟邻居有什么好处？可以快速获取远距离邻居的信息。实际上如果是按照 GraphSAGE 算法的方式生成子图，在聚合的过程中，非一阶邻居的信息可以通过消息传递逐渐传到中心，但是随着距离的增大，离中心越远的节点，其信息在传递过程中就越困难，甚至可能无法传递到；如果按照 PinSAGE 算法的方式生成子图，有一定的概率可以将非一阶邻居与中心直接相连，这样就可以快速聚合到多阶邻居的信息

## model training

通过使用max-margin ranking loss来通过有监督方式学习PinSage。

* loss function：

$$
J_{G}\left(\mathbf{z}_{q} \mathbf{z}_{i}\right)=\mathbb{E}_{n_{k} \sim P_{n}(q)} \max \left\{0, \mathbf{z}_{q} \cdot \mathbf{z}_{n_{k}}-\mathbf{z}_{q} \cdot \mathbf{z}_{i}+\Delta\right\}
$$

$P_n(q)$: 负样例

$\Delta$: pre-defined margin

目标是想要最大化正样例的内积。

* multi-GPU training with large minibatches：

在具有多个GPU的时候，我们将每个minibatch都划分到相同大小的部分。每个GPU都接受这样的一个部分以及使用相同的参数来完成计算。在经过反向传播之后，所有GPU上的对所有参数的梯度都被聚集在一起，一个单步的同步的SGD被计算。为了在很大数目上的样例进行计算，我们通常使用很大的batch size。为了保证模型的快速收敛以及保持训练和泛化的准确度，使用了gradual warmup procedure来从小到大地在第一个epoch上线性提升学习率，之后学习率进行指数级别的下降。

* producer-consumer minibatch construction：

在计算的时候如果需要进行GPU与CPU之间的转换的话，就会消耗大量的时间。于是，在PinSage之中，我们利用re-indexing来创建子图，只包括了目前的minibatch中的节点和它们的邻居。子图的邻接矩阵以及特征矩阵在每个minibatch的一开始就被放入GPUs中。
训练过程是GPU与CPU的迭代，模型在GPU上进行计算，但是挖掘特征，re-indexing，负采样则是放到CPU上进行计算。我们设计了producer-consumer模式来在当前iteration并行化地运行GPU计算，以及下一iteration的CPU计算。这样的做法可以将训练时间减少将近一半。

* sampling negative items：

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032211368.png" alt="image-20220103221142297" style="zoom:150%;" />


为了提升训练的效率，在每个minibbatch中采样出500个负样本来被所有训练样本共享。但将导致模型很难把负样本和相关性不大的样本区分开来。于是考虑了加入hard negative items。它们通过个性化的PageRank分数得到结果。它们较样本的相似度比随机负样本的相似度要高。

使用hard negative items会增加训练时长，为了减少收敛的时间，我们引入了课程学习的策略。在第一个epoch中，没有hard negative items加入，然后在之后的epochs中逐渐加入hard negative items，在训练的第n轮中，我们加入了n-1的hard negative items到negative items之中。


### node embeddings via MapReduce

![image-20220103221339514](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201032213585.png)

许多节点在为不同的目标节点生成embeddings的过程之中经过了重复计算。因此我们利用MapReduce来避免这种重复的计算。
1）将所有的pins投影到低维的潜在空间中，这样就可以进行aggregation操作了。
2）使用boards的id（pins的id）来连接pin的表征，board embedding通过pooling采样出的邻居节点得到。
这种mapreduce去除了冗余的计算操作，每个节点的潜在表示只被计算了一次。

### efficient nearest-neighbor lookups

给定离线训练好的PinSage，所有节点的embedding都可以通过mapreduce计算得到，有效的最近邻查询操作可以进行线上的推荐。

