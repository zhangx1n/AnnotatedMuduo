> **Title：**Self-Attentive Sequential Recommendation	**ICDM 2018**
>
> **Author：**Wang-Cheng Kang, Julian McAuley  （UC San Diego）
>
> **reference：**
>
> 1. [推荐算法炼丹笔记：序列化推荐算法SASRec_](https://blog.csdn.net/m0_52122378/article/details/110383706)
>
> 2. [ICDM2018｜SASRec---基于自注意力机制的序列推荐（召回）](https://blog.csdn.net/Blank_spaces/article/details/108526012)



## 背景

「Self-Attentive Sequential Recommendation」主要是针对的是**召回**的工作，提出SASRec序列推荐模型。作者受到Transformer启发，采用**自注意力机制**来对用户的历史行为信息建模，提取更为有价值的信息。最后将得到的信息分别与所有的物品embedding内容做内积，根据相关性的大小排序、筛选，得到Top-k个推荐。【当然在论文实验中，首先针对隐式数据集，完成的是一个二分类问题。在测试集中，并不是对所有的物品进行预测，这样耗时太长。针对每一个正样本，伴随了100个负样本，进行排序】


在本文中，self-attention的意义：对于下一次的物品推荐，依赖于用户行为序列的各个物品的权重是不同的，这与**「推荐场景有关」**。

* 用户物品交互较少：在一个稀疏数据集下，用户行为较少，行为相隔时间可能相差几天，甚至几个月，那么**「此时相近时间的历史物品信息表现得更为重要」**；

* 用户物品交互频繁：在一个密集型数据集下，用户行为多，例如在电商场景下，那么相近的物品信息就不是非常重要。例如，对于某个用户，他在电商场景的一个Session中，行为：手机--电脑--衣服---鼠标---裤子，那下一个用户感兴趣的是电子产品、服装都有可能，上述历史行为都很重要，这牵扯到一个**「多兴趣」**的概念了。

正如文章实验那样，「**在不同环境（数据集）下，模型的self-attention机制关注的重点是不同的**」。


## 模型

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9xUDhKUm5XNlQzclpkS0wzNXh2SU1odWJFNU1EQW5uNDhIRHNzdmQ4a29IaEtvekI5SHY5WWhnb2RXOU5KMEN4dGhzU1h5cFlpYlRTR2JBQzczNWNBTXcvNjQw)

### Embedding 层

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/79453fe720d9ea9801d25903d5e2456c.png)

> 关于这一部分其实与很多变长序列的任务都相似，需要将输入转化为一个矩阵，以方便GPU的计算。关于Embedding，在实际代码中，会「**将行为序列的特征，例如item_id、cate_id等从1开始进行表示，0作为填充值**」。之后会经过mask，因此0对应的Embedding向量并没有任何意义。

#### Positional Embedding

在Transformer中，我们知道，self-attention本身是不具备一种位置关系的，即交换序列中元素的位置，并不影响最终的结果。例如对于阿里的DIN模型，采用注意力机制对用户序列进行建模，但交换之前历史的物品，并不影响最终的预测，因为DIN只是提高与目标物品相似历史物品的权重，模型中的历史行为并没有表现一种时间上的先后关系，严格来说并不算一种序列推荐。因此Transformer中引入**「Positional Embedding（位置嵌入）」**，来表示序列中的一种先后位置关系。假设位置Embedding为 ，与行为序列的Embedding相加：

但作者提到，采用了Transformer中的Positional Embedding，结果却更差了，在实验部分有涉及讨论。

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/e5daa640eff43ea41a8efbbc96bd0713.png)

###  Self-Attention层

该部分与Transformer的**「编码层」**大体上是一致的，是由**「多个（或单个）【自注意力机制+（残差连接、LayerNormalization、Dropout）+前馈网络】」** 组成。

#### Self-Attention Block

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/b28b73a2bdb21d5626037a4af9c866a4.png)

#### Stacking Self-Attention Blocks

这里与Transformer的思想类似，认为**「叠加多个自注意力机制层能够学习更复杂的特征转换」**。

然而网络层数越深，会存在一些问题：

1. 模型更容易过拟合；
2. 训练过程不稳定（梯度消失问题等）；
3. 模型需要学习更多的参数以及需要更长的时间训练；

因此，作者在自注意力机制层和前馈网络**「加入残差连接、Layer Normalization、Dropout」**来抑制模型的过拟合。（其实依旧与Transformer类似）

### Prediction 层

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/20c2890fdceb4e52dfd7bfcbdeeac6ff.png)

使用同质(homogeneous)商品embedding的一个潜在问题是，它们的内部产品不能代表不对称的商品转换。然而，我们的模型没有这个问题，因为它学习了一个非线性变换。例如，前馈网络可以很容易地实现同项嵌入的不对称性,**经验上使用共享的商品embedding也可以大大提升模型的效果**;

**显示的用户建模**：为了提供个性化的推荐,现有的方法常用两种方法：(1).学习显示的用户embedding表示用户的喜好;(2).考虑之前的行为并且引入隐式的用户embedding。此处使用并没有带来提升。

### 训练

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/5800c516c63e2b9eea4a20d5628df898.png)

## 实验

### 数据集

文章在选取数据集也很严谨。选择了两个稀疏数据集（Amazon-Beauty、Amazon-Games）、物品平均交互多，用户平均交互少的Steam和密集型数据集（Movielens-1m），内容如下所示：

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9xUDhKUm5XNlQzclpkS0wzNXh2SU1odWJFNU1EQW5uNFlQVkNxSGliY1BKb2J0UkI5Z2NnNG9jY3JnbGFqeDVwRTR3MzZzc05mNFlmZjJRZktMUThLQmcvNjQw)

### Baseline

比较的方法主要分为3种：

1. 不考虑序列的传统推荐：PopRec、BPR；
2. 基于马尔可夫链的序列推荐：FMC、FPMC、TransRec；
3. 基于深度学习的序列推荐：GRU4Rec、GRU4Rec+、Caser；

### 实验指标

因为本文主要是针对的是召回工作，所以采用$Hit Rate@10$ 和$NDCG@10$ 两个指标。

并且为了减少大量的计算，在测试集中，对于每一个用户 ，都随机采样100个负样本进行排序。通俗来说，对于每一个样本，模型需要在101个物品中使得正样本分数更高，排得更前（不然要在所有的物品中进行召回，计算量太大）。

### 推荐效果比较

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/b44db4c144c2b0b80a91d235f8770e56.png)

- SASRec在稀疏的和dense的数据集合熵比所有的baseline都要好, 获得了6.9%的Hit Rate提升以及9.6%的NDCG提升；

### SASRec框架中不同成份的影响

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/f585a2c6a13c7cd18d5646a9b229d885.png)

* 删除PE: 删除位置embedding ,在稀疏的数据集上,删除PE效果变好,但是在稠密的数据集上,删除PE的效果变差了。
* 不共享IE(Item Embedding): 使用共享的item embedding比不使用要好很多;
* 删除RC(Residual Connection):不实用残差连接,性能会变差非常多;
* 删除Dropout: dropout可以帮助模型,尤其是在稀疏的数据集上,Dropout的作用更加明显;
* blocks的个数：没有block的时候,效果最差,在dense数据集上,相比稀疏数据多一些block的效果好一些;
* Multi-head:在我们数据集上,single-head效果最好.
  

### SASRec的训练效率和可扩展性

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/a91e23ad1305cce0a169f29e9e577ee3.png)

- SASRec是最快的;
- 序列长度可以扩展至500左右.