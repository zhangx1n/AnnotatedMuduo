> [Simple and Deep Graph Convolutional Networks](http://proceedings.mlr.press/v119/chen20v/chen20v.pdf)
>
> 该论文由中国人民大学、复旦大学、阿里巴巴合作完成，第一作者为中国人民大学研究生陈明，通讯作者为中国人民大学教授魏哲巍。
>
> ICML2020
>
> reference: [《Simple and Deep Graph Convolutional Networks》--论文阅读笔记_LIYUO94的博客-CSDN博客](https://blog.csdn.net/LIYUO94/article/details/107498408)

## 摘要

**G**raph **C**onvolutional **N**etwork via **I**nitial residual and **I**dentity mapping(**GCNII**)，它是普通GCN模型的扩展，应用了两种简单而有效的技术：**初始残差（Initial residual）和恒等映射（Identity mapping）**

## 简介

主要说了以下几点：

1. GNN

2. 传统GCN的局限：浅层

3. 现有的一些解决方案：

    1. 使用密集跳跃连接组合每一层的输出，以保持节点表示的位置
    2. 从输入图中随机删除一些边

    另外有一些将深度传播与浅层神经网络结合：

    1. SGC试图通过在单个神经网络层中应用图卷积矩阵的 k 次幂来捕获图中的高阶信息
    2. PPNP和APPNP用自定义的PageRank矩阵代替图卷积矩阵的幂来解决过平滑问题
    3. GDC进一步扩展了APPNP，将自定义PageRank推广到任意图扩散过程

    但是，这些方法只是将每一层的相邻特征进行线性组合，而失去了深度非线性架构强大的表达能力，仍然是浅模型。

4. 本文的主要工作：
    1. 在每一层，初始残差从输入层构建一个跳跃连接，而恒等映射在权值矩阵中添加一个单位矩阵。
    2. 对多层GCN和GCNII模型进行了理论分析。已知叠加 K 层的GCN**本质上模拟了一个具有预定系数的 K 阶多项式滤波器**。之前的研究指出，该滤波器模拟了一个懒惰的随机游走（lazy random walk），最终收敛到平稳向量，从而导致过平滑。
    3. 证明了$K$ 层$GCNII$模型可以表示任意系数的 K 阶多项式谱滤波器。这个特性对于设计深度神经网络是必不可少的。
    4. 推导了平稳向量的封闭形式，并分析了普通GCN的收敛速度。分析表明，在多层GCN模型中，**度比较大的节点更有可能出现过度平滑的现象。**
       

## 对于原始GCN过平滑的分析

深层网络的训练最常用的优化方法为添加 residual connection，添加了residual connection的GCN可以表示为：$\mathbf{H}^{(\ell+1)}=\sigma\left(\tilde{\mathbf{P}} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)}\right)+\mathbf{H}^{(\ell)}$,带有 residual connection 的k层图卷积在去掉每层非线性的激活函数之后，可表示为$\mathbf{h}^{(K)}=\left(\frac{\mathbf{I}_{n}+\tilde{\mathbf{D}}^{-1 / 2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1 / 2}}{2}\right)^{K}$

因此，此类GCN仅能表达固定系数的K阶多项式滤波器。注意固定系数的K阶多项式滤波器并不一定会导致过平滑，然而GCN所对应的K阶多项式系数实际上是在模拟 lazy random walk，该类随机游走会收敛到一个与初始状态（即节点特征）无关的稳态分布。

## GCNII

本文试图对原始GCN进行最小的修改，将其改造为一个真正的深度模型。GCNII的第$l$层表示如下：
$$
\mathbf{H}^{(\ell+1)}=\sigma\left(\left(\left(1-\alpha_{\ell}\right) \tilde{\mathbf{P}} \mathbf{H}^{(\ell)}+\alpha_{\ell} \mathbf{H}^{(0)}\right)\left(\left(1-\beta_{\ell}\right) \mathbf{I}_{n}+\beta_{\ell} \mathbf{W}^{(\ell)}\right)\right)
$$
其中$\alpha_\ell, \beta_\ell$是超参数，$\tilde{\mathbf{P}}=\tilde{\mathbf{D}}^{-1 / 2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1 / 2}$

主要涉及两个修改：

1. 在每个卷积层中，将平滑后的表示$\tilde{\mathbf{P}} \mathbf{H}^{(\ell)}$与$\mathbf{H}^{(0)}$结合。
2. 在权重矩阵中增加单位映射。

在每一层中添加初始的表示使得节点本身的特征不会随着层数的增多而被稀释，避免了过度平滑。在权重矩阵中添加单位映射的思想其实也来源于ResNet，这意味着将$\tilde{\mathbf{P}} \mathbf{H}^{(\ell)}+ \mathbf{H}^{(0)}$直接映射到输出，使得我们可以在使用正则化等技术缓解过拟合的同时仍然保留住高阶信息。这里$\alpha_{\ell}$和$\beta_\ell$是两个超参数，实验中$\beta_l = log(\lambda/\ell +1)$， $\alpha_\ell$ 和 $\lambda$被设置0.1到1之间的常数。

本文还证明了经过这两项修改，GCNII可以表达任意系数$\theta_\ell$的K阶多项式滤波器

![image-20211202141948890](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021419939.png)

定理2保证了即使深层的GCNII也收敛到稳态分布，该稳态分布也将会是邻接矩阵和节点特征的函数。例

如：当$\theta_\ell = \alpha(1-\alpha)^\ell$时，该K阶多项式滤波器实际上就是APPNP。因此，深层GCNII不会出现深层GCN丢失节点特征的问题，从而彻底避免了过平滑。而相比不同hop之间线性叠加的APPNP，GCNII在每一层都包含非线性激活函数，是一个真正的深度图模型。

## 实验结果

![image-20211202142346941](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021423000.png)

![image-20211202142359777](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021423837.png)

![image-20211202142415059](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021424141.png)

![image-20211202142431461](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021424539.png)

![image-20211202142449571](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021424656.png)

![image-20211202142541600](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021425745.png)

## 总结

![image-20211202143151597](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112021431709.png)