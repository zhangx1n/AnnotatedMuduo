> reference: [PREDICT THEN PROPAGATE: GRAPH NEURAL NETWORKS MEET PERSONALIZED PAGERANK]([1810.05997.pdf (arxiv.org)](https://arxiv.org/pdf/1810.05997.pdf)) ICML2019
>
> 参考资料：[无限层数GNN之PPNP](https://zhuanlan.zhihu.com/p/417615165)
>
> ​					[读论文 PPNP/APPNP Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://blog.csdn.net/PolarisRisingWar/article/details/118653133)
>
> 代码：[klicperajo/ppnp: PPNP & APPNP models from "Predict then Propagate: Graph Neural Networks meet Personalized PageRank" (ICLR 2019) (github.com)](https://github.com/klicperajo/ppnp)



## 摘要

随着GNN网络层数的提高，存在过平滑和参数量过大难以训练的问题。作者探索了GCN与PageRank算法之间的关系，提出了基于Personalized Page Rank 的改进版本的信息传递方式PPNP(Personalized Propagation of Neural Prediction)，以及快速近似版本APPNP。引入APPNP的模型可以大幅减少参数量，并取得较好的效果。Personalized Page Rank 增加了跳转回根节点的机会，因此可以平衡保留local信息和利用大范围邻居信息。作者还将MPNN结构中的消息传递分离出来，为每个点提供了一个较大的感受野的同时不引入参数。

## Page Rank & Personalized Page Rank

Page Rank 算法是Google创始人Larry Page 提出的，使用网络超链接进行网页重要性得分的计算算法。Page Rank的初始想法源自论文的引用，如果某篇论文被一篇非常重要的论文所引用，那么该篇论文的重要性也比较高，因为它被重要的论文背书了。一个基本的想法是某论文的重要性得分源自所有引用它的论文，而该论文又对所有它引用的论文进行重要性评估。回到互联网上的超链，Larry Page认为，重要性可以看作是某人在某个节点上以等概率跳转到该节点引用的节点，而重要性描述了最终停在每个节点上的概率。形式化的表示为<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112010958087.png" alt="image-20211201095819827" style="zoom: 50%;" />


而Page Rank 是通过随机游走的方法对得分进行迭代式计算，假定存在引用图

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112010959983.png)

转移矩阵定义为![image-20211201095938431](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112010959469.png)

初始时，所有点的重要性相同，设为![image-20211201100011213](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011000249.png)

因此，对每个点随机游走一次后，结果表示为$w_1 = TW_0$，迭代n次后的结果即为最终的节点重要性。

但是，以上的过程中存在两个问题：

* 等级泄露：某点没有出度，那么最终收敛时，该节点的重要性为1
* 等级沉默：某点没有入度，那么最终收敛时，该节点的重要性为0

为了解决该问题，Page Rank 假定网页上存在一个随机跳转按钮，点击后可以随机跳转到任意一个网页上，当，因此，Page Rank 得分形式化表示为![image-20211201100516719](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011005758.png)
其中 *μ* 描述不根据超链进行跳转的概率。使用矩阵描述以上的表达式，即为

![image-20211201100550332](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011005370.png)

其中 e表示全为 $ \frac{1}{N} $ 的向量。


Personalized Page Rank算法继承了经典PageRank算法的思想，利用图链接结构来递归地计算各结点的权重，即模拟用户通过点击链接随机访问图中结点的行为 计算稳定状态下各结点得到的随机访问概率。PPR与Page Rank的最大区别在于随机行走中的跳转行为。为了保证随机行走中各结点的访问概率能够反映出用户的偏好，PPR算法要求在随机行走中的每次跳转不可随机选择到任意结点，用户只能跳转到一些特定的结点，即代表用户偏好的那些结点。因此，在稳定状态下，用户所偏好的那些结点和相关的结点总能够获得较高的访问概率。PPR算法表示为：
![image-20211201100658933](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011006970.png)

其中$∣v∣=1 $表示用户的偏好。


## PPNP

![image-20211201100759452](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011007580.png)

## APPNP

![image-20211201100826261](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112011008314.png)

