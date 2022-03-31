> PairNorm:Tackling Oversmoothing in GNNs
>
> ICLR2020
>
> DropEdge作者团改投的期刊。分析多了带bias场景。并且给出了GCN，带bias的GCN，resGCN,以及APPNP的over-smooth分析
>
> 

众所周知，GNNs的表现随着层数的增加而有所下降，这在一定程度上归结于over-smoothing这一问题，重复图卷积这一操作会使得节点的表示最终变得不可区分。作者希望通过采取两种不同的理解方式来量化over-smoothing这一问题，并提出解决这一问题的方法。在仔细地研究过图卷积这一操作之后，作者提出了一种新奇的normalization layer---PairNorm，来防止节点的表示变得过于相似。除此之外，PairNorm具有高效且易于实现的特点，不需要对于整个模型架构做太多的改变，也不需要增加额外的参数，广泛地适用于所有的GNNs。

文章的贡献主要如下：

（1）**提出一种normalization的方法来解决GNNs over-smoothing的问题**；这一想法的关键点在于控制全部的两两节点间的特征向量的距离和为一个常数，这样可以使得距离较远的节点的特征向量的距离也比较远；

（2）**高效并具有广泛应用的价值**：直观来看（1）提出的方法，其实现过程像是需要对于每两个点之间的特征向量均计算距离，而后将其加入loss项中进行优化，但这一操作的复杂度为O(n2)，这是难以接受的。作者基于对于卷积操作前后的节点特征向量关系的理论分析，提出一种center-scale的方法，来使得这一过程变得高效，不需要增加额外的参数，并容易实现，从而具有了应用的价值；

（3）**分析怎样的问题适合更加深层的GNNs**：在现实数据集上的实验证明虽然PairNorm的方法确实可以处理over-smoothing这一问题，但GNNs的表现并不能随着层数的增加而上升，我们仍然无法从深层的GNNs中获益，据此作者分析了什么怎样的网络适合深层的GNNs，并构造数据验证了这一结论。

此外，**作者提出了一种理解over-smoothing的视角**，并从节点之间的特征向量的距离和一个节点的特征向量之间的差异来量化over-smoothing这一问题，而作者也正是从这一角度出发，提出了PairNorm的方法。同时作者还给出了节点特征向量经过图卷积这一操作之后所具有的性质---GRLS问题的近似解。



![img](https://pic2.zhimg.com/80/v2-d54584acef079ec9d11d9a1d2d422c51_720w.jpg)



Figure 1：Graph-Regularized Least Squares（GRLS）问题