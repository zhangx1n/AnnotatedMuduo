> ****
>
> **[Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training](https://arxiv.org/pdf/2104.09376.pdf)**
>
> 作者：中国电信研究院
>
> 代码：

**reference：**

1. [霸榜长达三个月！看中国电信如何在图神经网络权威榜单独占三项鳌头 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/388789950)

## 摘要

使用基于SIGN (Scalable Inception Graph Neural Networks)改进得到的**SAGN (Scalable and Adaptive Graph Neural Network)模型**，结合独创的SLE (Self-Label-Enhanced)训练方法，新模型在图神经网络权威榜单OGB (Open Graph Benchmark)的三项任务中勇夺第一，准确率提升最高达到了惊人的两个百分点。

## SAGN+SLE，高性能和高可扩展性的强力组合

SIGN模型于2020年由帝国理工学院和推特联合提出。在此基础上，AI基础能力研究团队创新性地使用精心设计的注意力结构替代了原SIGN模型中对图神经网络多级输出的简单拼接，得到的SAGN模型如下图所示。

![image-20211204104753263](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112041047454.png)

* 注意力机制使模型能够自适应地为各级输出分配不同的权重，从而更准确地捕获其中的关键信息。这一优化在保持模型复杂度的同时，大大提升了模型的表达能力与可解释性，其对模型在准确率上的提升也在榜单中清晰可见。

* 独创的SLE训练方法将模型自训练与标签传播完美结合，让模型精度更上一层楼。AI基础能力研究团队提出将基模型(Base model，本方法中即为SAGN)的输出与标签模型的输出进行叠加，以充分利用节点的标签信息。同时，在实验中发现，增加标签传播的迭代次数能够有效减弱标签溢出造成的影响，省去了为解决标签溢出而引入的标签掩码算法，从而大幅提高了该算法在图规模上的可扩展性。

![image-20211204105045012](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112041050142.png)

* 此外，SLE方法还引入了多阶段自学习的训练方式。除第一次训练迭代外，之后的每次训练都会利用模型前一次训练的输出对训练集进行增强，包括增加样本点及其对应伪标签，从而提升模型效果。

![preview](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112041052526.jpeg)

## 翻译

![preview](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112041042187.jpeg)