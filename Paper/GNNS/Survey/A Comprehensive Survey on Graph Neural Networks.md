# 图神经网络GNN综述

> reference：[A Comprehensive Survey on Graph Neural Networks (arxiv.org)](https://arxiv.org/abs/1901.00596)
>
> [论文翻译-A Comprehensive Survey on Graph Neural Networks《图神经网络GNN综述》](https://blog.csdn.net/weixin_35479108/article/details/86308808?ops_request_misc=%7B%22request%5Fid%22%3A%22163368919516780366577474%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=163368919516780366577474&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-86308808.first_rank_v2_pc_rank_v29&utm_term=A+Comprehensive+Survey+on+Graph+Neural+Networks&spm=1018.2226.3001.4187)

## Abstract

提供了一个在数据挖掘和机器学习领域的图神经网络的全面的视角。

提出了一个新的分类方法（taxonomy）将GNNs分成了四类：

* recurent graph neural networks,
* convolutional graph neural networks
* graph autoencoders
* spatial-temporal

讨论了GNNs在个个领域的应用，总结了开源代码基准（benchmark）数据集，最后提出了潜在研究方向。

## Introduction

图数据不规则，每个图的无序节点大小是可变的，且每个节点有不同数量的邻居节点，一些卷积操作在图像数据上不容易计算。现有的机器学习算法假设数据之间是相互独立的，但是，图数据每个结点都通过一些复杂的连接信息与其他邻居相关，这些连接信息用于捕获数据之间的相互依赖关系，包括，引用，关系，交互。

本文贡献：

* New teaxonomy
* Comprehensive review
* Abundant resources
* Future direction

## Bacground & Definition

* Graph neural networks vs. network embedding 

network embedding致力于在一个低维向量空间进行网络节点表示，同时保护网络拓扑结构和节点的信息，便于后续的图像分析任务，包括分类，聚类，推荐等，能够使用简单现成的机器学习算法（例如，使用SVM分类）。许多network embedding算法都是典型的无监督算法，它们可以大致分为三种类型,，即，矩阵分解、随机游走、深度学习。

基于深度学习的network embedding属于GNN，包括图自编码算法(e.g. DNGR and SDNE)和基于无监督训练(e.g., GraphSage )的图卷积神经网络


* Graph neural networks vs. graph kernel methods

graph kernel是历史上解决图分类问题的主要方法。这些方法使用一个核函数来度量图对之间的相似性，这样基于核的算法（如支持向量机）就可以用于图的监督学习。与GNN类似，图核可以通过映射函数将图或节点嵌入到向量空间中。不同的是，这个映射函数是确定性的，而不是可学习的。图核方法由于具有对相似性计算的特点，存在计算瓶颈。一方面，GNN直接根据提取的图形表示进行图形分类，因此比graph kernel方法更有效。
![image-20211026101049845](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110261010521.png)

## Categorization & Frameworks

### Taxonomy of GNNs

1. RecGNNs

    GNN的开创性工作就是RecGNNs, 旨在学习循环神经结构的节点表示。假设图中的一个节点不断地与它的邻居交换信息，直到一个稳定的平衡点出现。RecGNNs在概念上非常重要，并启发了后来的卷积图神经网络研究。特别是消息传递的思想被基于空间的卷积图神经网络(spatial-based)所继承。

2. ConvGNNs

    ConvGNNs将传统数据的卷积算子泛化到了图数据。主要思想是生成一个节点的表征(representation)，它聚合了节点自己的特征$X_v$和所有邻居的特征$X_u$. 和RecGNNs不同，ConvGNNs堆叠多个图卷积层来提取高级节点的表征。![image-20211026103524854](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110261035982.png)

3. GAEs

    GAEs是一种无监督的方法，将节点或者图编码到一个潜在的矢量空间，并从编码信息中重建图形数据。GAEs被用来学习网络嵌入和图生成分布。对于网络嵌入，GAEs 通过重建图的结构信息（如图的邻接矩阵）来学习潜在的节点表征。对于 图的生成，有些方法是逐步生成图的节点和边。图的节点和边，而其他方法则是一次性输出一个图。

    ![image-20211026104155688](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110261041753.png)

4. STGNNs

    GSTN的核心观点是，**同时考虑空间依赖性和时间依赖性**。目前很多方法使用GCNs捕获依赖性，同时使用RNN,或者CNN建模时间依赖关系。

![image-20211026104618241](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110261046316.png)

### Frameworks

1. Graph analytics tasks

    1. Node-level

        输出是node regression 或者 node classification。RecGNNs或者ConvGNNs通过信息传播或者图卷积提取到high-level的node representation。然后接一个softmax或者mlp到输出层。这是end-to-end的方法

    2. Edge-level

        edge classification，link prediction. 用两个来自GNNs的节点表征来作为输入，咋用一个类似的神经网络去做预测。

    3. Graph-level

        graph classification。 GNNs are often combined with pooling and readout operations.

2. Training Frameworks

    1. Semi-supervised learning for node-level classification.
    2. Supervised learning for graph-level classification.
    3. Unsupervised learning for graph embedding.

## Recurrent Graph Neural Networks

* 开篇之作：[Scarselli F, Gori M, Tsoi A C, et al. The graph neural network model[J]. IEEE transactions on neural networks, 2008, 20(1): 61-80.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4700287)
* GraphESN：[Gallicchio C, Micheli A. Graph echo state networks[C]//The 2010 International Joint Conference on Neural Networks (IJCNN). IEEE, 2010: 1-8.](https://ieeexplore.ieee.org/abstract/document/5596796)
* GGNN：[Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networks[J]. arXiv preprint arXiv:1511.05493, 2015.](https://arxiv.org/abs/1511.05493)
* SSE:[Dai H, Kozareva Z, Dai B, et al. Learning steady-states of iterative algorithms over graphs[C]//International conference on machine learning. PMLR, 2018: 1106-1114.]()

## Concolutional Graph Neural Networks

Spectral-based ConvGNNs从图信号处理的角度，Spatial-based ConvGNNs从信息传播的角度。spatial-based methods 更方便，灵活，今年发展的更加迅速。

### Spectral-based ConvGNNs

谱域方法有着深厚的图信号处理(graph signal processing)的数学基础，假设**图是无向图**

归一化的**Laplacian matrix**: $\mathbf{L}=\mathbf{I}_{\mathbf{n}}-\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}$ ,其中 $\mathbf{D}_{i i}=\sum_{j}\left(\mathbf{A}_{i, j}\right)$

拉普拉斯矩阵是**对称的半正定矩阵**，所以能够**谱分解**$\mathbf{L}=\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^{T}$, 其中 $\mathbf{U}=\left[\mathbf{u}_{\mathbf{0}}, \mathbf{u}_{\mathbf{1}}, \cdots, \mathbf{u}_{\mathbf{n}-\mathbf{1}}\right] \in \mathbf{R}^{n \times n}$是特征向量，$\boldsymbol{\Lambda}$是特征值组成的对角阵，$\boldsymbol{\Lambda}_{i i}=\lambda_{i}$。归一化拉普拉斯矩阵的特征向量构成了一个**正交空间**(orthonormal space), 即$\mathbf{U}^{T} \mathbf{U}=\mathbf{I}$。

在图信号处理中，图的信号$\mathbf X \in \mathbf R^n$是节点的特征向量。信号$X$的***graph Fourier transform***(图傅里叶变换)被定义为： $\mathscr{F}(\mathbf{x})=\mathbf{U}^{T} \mathbf{X}$，***inverse graph Fourier transform***(反傅里叶变换)被定义为：$\mathscr{F}^{-1}(\hat{\mathbf{x}})=\mathbf{U} \hat{\mathbf{x}}$，$\hat{\mathbf x}$是图傅里叶变换的结果。 图傅里叶变换将输入的图形信号投射到正交空间，这个正交空间的基是$\mathbf L$特征向量。输入信号可以表示成$\mathbf{x}=\sum_{i} \hat{x}_{i} \mathbf{u}_{i}$。

输入信号$X$ 和滤波器(Filter) $g \in \mathbf R^n$的图卷积：

​																	$\begin{aligned} \mathbf{x} *_{G} \mathbf{g} &=\mathscr{F}^{-1}(\mathscr{F}(\mathbf{x}) \odot \mathscr{F}(\mathbf{g})) \\ &=\mathbf{U}\left(\mathbf{U}^{T} \mathbf{x} \odot \mathbf{U}^{T} \mathbf{g}\right) \end{aligned}$

$\odot$是Hadamard product(哈达玛积)，就是直接对应位置元素直接相乘。

> 
> $$
> \begin{gathered}
> (f * h)_{G}=U\left(\begin{array}{cc}
> \hat{h}\left(\lambda_{1}\right) & & \\
> & \ddots & \\
> & \hat{h}\left(\lambda_{n}\right)
> \end{array}\right) U^{T} f \\
> (f * h)_{G}=U\left(\left(U^{T} h\right) \odot\left(U^{T} f\right)\right)
> \end{gathered}
> $$
> 上面两个式子其实是等价的[GCN中的等式证明 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/121090537)

如果定义滤波器$\mathbf{g}_{\theta}=\operatorname{diag}\left(\mathbf{U}^{T} \mathbf{g}\right)$，那么*spectral graph convolution* 可以被简化为:

​																$\mathbf{x} *_{G} \mathbf{g}_{\theta}=\mathbf{U g}_{\theta} \mathbf{U}^{T} \mathbf{x}$

所有的Spactral-based ConvGNNs 都符合这个定义。只是滤波器$g_\theta$的选择不同。

* Spectral CNN: [Bruna J, Zaremba W, Szlam A, et al. Spectral networks and locally connected networks on graphs[J]. arXiv preprint arXiv:1312.6203, 2013.](https://arxiv.org/pdf/1312.6203.pdf%20http://arxiv.org/abs/1312.6203.pdf)
    * filter: $\mathbf{g}_{\theta}=\boldsymbol{\Theta}_{i, j}^{(k)}$是一系列可学习的参数。并且将图信号考虑成多通道
    * graph convolutino layer: $\mathbf{H}_{:, j}^{(k)}=\sigma\left(\sum_{i=1}^{f_{k-1}} \mathbf{U} \mathbf{\Theta}_{i, j}^{(k)} \mathbf{U}^{T} \mathbf{H}_{:, i}^{(k-1)}\right) \quad\left(j=1,2, \cdots, f_{k}\right)$, 其中$k$是layer index, $\mathbf{H}^{(k-1)} \in \mathbf{R}^{n \times f_{k-1}}$ 是输入的图信号，$\mathbf{H}^{(0)}=\mathbf{X}$, $f_{k-1}$是输入通道数，$f_k$是输出通道数。
    * 由于the eigen-decomposition of the Laplacian matrix, 有以下缺点：
        * `Spectral CNN`的计算依赖于拉普拉斯矩阵分解，显式地使用了特征向量矩阵$U$，而这个分解,以及矩阵相乘计算时间复杂度较高；
        * 基于拉普拉斯矩阵分解，导致卷积核不是局部化的(**spatial localization**),可以理解为一个节点的信息聚合不是来自于其邻居，而是所有节点；
* Chebyshev Spectral CNN(ChebNet): [Defferrard M, Bresson X, Vandergheynst P. Convolutional neural networks on graphs with fast localized spectral filtering[J]. Advances in neural information processing systems, 2016, 29: 3844-3852.](https://proceedings.neurips.cc/paper/2016/file/04df4d434d481c5bb723be1b6df1ee65-Paper.pdf)
    * filter:  $\mathbf{g}_{\theta}=\sum_{i=0}^{K^{\prime}} \theta_{i} T_{i}(\tilde{\Lambda})$, where $\tilde{\Lambda}=2 \boldsymbol{\Lambda} / \lambda_{\max }- \mathbf{I_n}$ , the value of $\tilde {\Lambda}$ lie in [-1, 1].
    * localized in space
* Graph Convolutional Network(GCN): [Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.](https://arxiv.org/pdf/1609.02907.pdf?fbclid=IwAR0BgJeoKHIAvPuSE9fJ0_IQOEu5l75yxyNo7PUC08RTOFlm_IIo5YmcnQM)
    * $\mathbf{x} *_{G} \mathbf{g}_{\theta}=\theta\left(\mathbf{I}_{\mathbf{n}}+\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}\right) \mathbf{x}$

* Adaptive Graph Convolution Network (AGCN): [[Adaptive Graph Convolutional Neural Networks | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/11691)]



### Spatial-based ConvGNNs

* Neural Network for Graphs (NN4G): 
    * the **first** work towards spatial-based ConvGNNs
    * $\mathbf{h}_{v}^{(k)}=f\left(\mathbf{W}^{(k)^{T}} \mathbf{x}_{v}+\sum_{i=1}^{k-1} \sum_{u \in N(v)} \mathbf{\Theta}^{(k)^{T}} \mathbf{h}_{u}^{(k-1)}\right)$
* Diffusion Convolutional Neural Network (DCNN): 
    * 假定信息以一定的传播概率从一个节点转移到其相邻的一个节点，因此 信息分布可以在几轮之后达到平衡。
    * diffusion graph convolution： $\mathbf{H}^{(k)}=f\left(\mathbf{W}^{(k)} \odot \mathbf{P}^{k} \mathbf{X}\right)$
    * probability transition matrix $\mathbf {p = D^{-1}A}$
    * output : ($concatenate(\mathbf{H}^{(1)}, \mathbf{H}^{(2)}, \cdots, \mathbf{H}^{(K)})$)
* Diffusion Graph Convolution (DGC):
    * diffusion graph convolution: $\mathbf{H}=\sum_{k=0}^{K} f\left(\mathbf{P}^{k} \mathbf{X} \mathbf{W}^{(k)}\right)$
* 

## Data sets

![image-20211028160503004](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110281605176.png)