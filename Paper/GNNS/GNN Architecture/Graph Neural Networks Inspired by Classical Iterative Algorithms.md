> 论文：[Graph Neural Networks Inspired by Classical Iterative Algorithms](http://proceedings.mlr.press/v139/yang21g/yang21g.pdf)
>
> 作者：Yongyi Yang，Tang Liu ，Yangkun Wang， Jinjing Zhou，Quan Gan , **Zhewei Wei** , Zheng Zhang, Zengfeng Huang, David Wipf
>
> 机构：AWS Shanghai AI Lab
>
> 会议：ICML 2021
>
> 代码：[FFTYYY/TWIRLS: Code for the ICML 2021 paper "Graph Neural Networks Inspired by Classical Iterative Algorithms". (github.com)](https://github.com/FFTYYY/TWIRLS)

## 简介

GNN容易被以下问题影响：

* oversmoothing 过平滑
* long-range dependencies 长程依赖
* spurious edges 假边

为了用一个简单的框架至少去部分的解决这个问题，考虑了一种新的GNN层，旨在模仿和集成两种经典迭代算法的更新规则，即近端梯度下降和迭代再加权最小二乘法( **proximal gradient descent** and **iterative reweighted least squares** )。

* 前者定义了一个可扩展的基本GNN架构，该架构不受oversmoothing的影响，但通过允许任意传播步骤来捕获long-range dependencies关系。
* 后者产生了一种新的注意机制，明确锚定到一个潜在的端到端energy function，在边缘不确定性方面提供稳定性。

这样就得到了一个既简单又健壮的模型。



当前的基于message-passing GNNs（包括运用各种attention机制的变体）都只是在homophily graph（附近的节点上有相近的labels和features）上表现不错，换到domain of heterophily 性能就急剧下降。同样，当通过对抗性攻击（adversarial attacks）或相关的方式引入边缘不确定性（edge uncertainty）时，性能也会下降。

message-passing GNNs 想要在non-local regions of the graph 需要更多的propagation layers.但是这会导致oversmoothing。之前大多数的工作都是采用skip connections 的方法去保存局部的信息。大多数更深层次的体系结构通常显示出很小的改进，但是可能需要额外的调优或正则化策略才能有效。



## 模型

### Extensible GNN Layers via Unfolding

定义energy function：
$$
\ell_{Y}(Y) \triangleq\|Y-f(X ; W)\|_{\mathcal{F}}^{2}+\lambda \operatorname{tr}\left[Y^{\top} L Y\right] \tag{1}
$$
$Y \in \mathbb{R}^{n\times d}$: a condidate embedding of $d$-dimensional features across $n$ nodes.

$f(X;W)$是一个函数：例如可以是MLP等等，$W$是参数，$X$是节点的初始特征。

$L=D-A=B^{\mathrm T} B$ : $D$ 和 $A$分别是度矩阵和邻接矩阵，$B$是关联矩阵（incidence matrix）



易得：
$$
Y^{*}(W) \triangleq \arg \min _{Y} \ell_{Y}(Y)=(I+\lambda L)^{-1} f(X ; W) \tag 2
$$
很明显$Y^*$是$W$的一个函数，这个解表示$f(X;W)$的近似，$f(X;W)$在图结构中平滑了，在(1)中可以看到它平衡了local和global的信息。因此可以将$Y^{*}(W)$含有图的结构信息可以用在下游预测任务上。
$$
\ell_{W}(W, \theta) \triangleq \sum_{i=1}^{n^{\prime}} \mathcal{D}\left(g\left[\boldsymbol{y}_{i}^{*}(W) ; \theta\right], t_{i}\right) \tag 3
$$
$t_i$: ground-truth

$\mathcal D$: discriminator function 例如交叉熵等

但是这只在小图上有用，因为大图的$I+\lambda L$的逆计算起来太难了，我们可以用(1)梯度的步骤近似$f(X;W)$

从这个意义上说，我们本质上是展开一系列梯度步骤，以形成一个可微链，可以视为一个深度架构的可训练层次。类似的策略在过去已经被使用来促进元学习，使用来自最优本身不能直接微分损失的特征。
$$
\frac{\partial \ell_{Y}(Y)}{\partial Y}=2 \lambda L Y+2 Y-2 f(X ; W) \tag 4
$$

$$
Y^{(k+1)}=Y^{(k)}-\alpha\left[Q Y^{(k)}-f(X ; W)\right], \quad Q \triangleq \lambda L+I  \tag 5
$$

步长：$\frac{\alpha}{2}$,但是如果$Q$很大，那么梯度下降收敛的很慢，preconditioning 的技术可以减少收敛时间。

预处理包括简单地缩放每个梯度步骤：

使用$(\operatorname{diag}[Q])^{-1}=(\lambda D+I)^{-1}$，$\tilde{D} \triangleq \lambda D+I$

则：
$$
Y^{(k+1)}=(1-\alpha) Y^{(k)}+\alpha \tilde{D}^{-1}\left[\lambda A Y^{(k)}+f(X ; W)\right] \tag 6
$$
我们将把这个表达式当作一个GNN模型的第k个可微传播层