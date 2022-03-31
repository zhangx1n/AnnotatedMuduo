# paper

* [Efficient Estimation of Word Representations in  Vector Space]([1301.3781.pdfì— ì„œ (arxiv.org)](https://arxiv.org/pdf/1301.3781.pdfì— ì„œ))  向量空间中词表示的有效估计

* [Distributed representations ofwords and phrases and their compositionality]([Distributed Representations of Words and Phrases and their Compositionality (neurips.cc)](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)) 单词和短语的分布式表示及其组成

# A Simplified Representation of Word Vectors

![img](https://cdn-images-1.medium.com/max/1600/0*mRGKYujQkI7PcMDE.)

​	在图中，我们想象每个维度都有一个明确定义的含义。 例如，如果您想象第一个维度代表“动物”的含义或概念，那么每个词在该维度上的权重代表它与该概念的关联程度。



# 从整体来看Word2vec 

* 目的：学习出一个低维的向量表示一个词的含义

* 两个算法：
    1. Skip-grams (SG)：预测上下文
    2. Continuous Bag of Words (CBOW)：预测目标单词

* 两种高效的训练方法：
    1. Hierarchical softmax
    2. Negative sampling

![img](https://cdn-images-1.medium.com/max/1600/0*YOchANwzN9tLv0M5.)



------

**word2vec**简单来说就是一个高效的实现word embedding的算法工具。

![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291016596.jpeg)

 如上图所示，word2vec模型其实就是一个简单化的神经网络，输入是One-Hot向量，Hidden Layer没有激活函数，也就是线性的单元。Output Layer维度跟Input Layer的维度一样，用的是Softmax回归。当这个模型训练好以后，我们并不会用这个训练好的模型处理新的任务，我们真正需要的是这个模型通过训练数据所学得的参数，例如隐层的权重矩阵。

这个模型是如何定义数据的输入和输出呢？这个时候就需要提到**CBOW(Continuous Bag-of-Words** 与**Skip-Gram**两种模型。 

\- **CBOW模型**的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。　 

\- **Skip-Gram模型**和CBOW的思路是反着来的，即输入是特定的一个词的词向量，而输出是特定词对应的上下文词向量。CBOW对小型数据库比较合适，而Skip-Gram在大型语料中表现更好。

![img](https://pic2.zhimg.com/50/v2-6609552b2aa6d42c78d1d50f70173937_720w.jpg?source=1940ef5c)![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291016589.jpeg)



如上图所示，左右两张图分别从不同角度代表了输入层-隐层的权重矩阵。左图中每一列代表一个10000维的词向量和隐层单个神经元连接的权重向量。右图中，每一行实际上代表了每个单词的词向量。

如果我们将一个`1 x 10000的向量`和`10000 x 300的矩阵`相乘，它会消耗相当大的计算资源，为了高效计算，隐层权重矩阵看成了一个"查找表"（lookup table），进行矩阵计算时，直接去查输入向量中取值为1的维度下对应的那些权重值。隐层的输出就是每个输入单词的word embedding vector,如下图所示。 

![img](https://pic2.zhimg.com/50/v2-b45a5aa17bcba9ecf17a1ac65166e684_720w.jpg?source=1940ef5c)![img](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291016606.jpeg)



所以我们可以看出，word2vec得出的词向量其实就是训练后的一个神经网络的隐层的权重矩阵，在经过CBOW或者Skip-Gram模型的训练之后，词义相近的词语就会获得更为接近的权重，因此可以用向量的距离来衡量词的相似度。

# 细节

目标函数定义为所有位置的预测结果的乘积：

![hankcs.com 2017-06-07 下午2.55.51.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291024629.jpeg)

要最大化目标函数。对其取个负对数，得到损失函数——对数似然的相反数：

![hankcs.com 2017-06-07 下午2.57.28.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291024747.jpeg)

## 目标函数细节

这些术语都是一样的：Loss function = cost function = objective function，不用担心用错了。对于softmax来讲，常用的损失函数为交叉熵。

### Word2Vec细节

预测到的某个上下文条件概率p(wt+j|wt)p(wt+j|wt)可由softmax得到：

![hankcs.com 2017-06-07 下午3.07.22.png](http://wx2.sinaimg.cn/large/006Fmjmcly1fgcnjx5gssj30tu09u47t.jpg)

o是输出的上下文词语中的确切某一个，c是中间的词语。u是对应的上下文词向量，v是词向量。

### 点积

复习一下课程开头所说的baby math：

![hankcs.com 2017-06-07 下午3.09.43.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291024762.jpeg)

公式这种画风这种配色真的有点幼儿园的感觉。

点积也有点像衡量两个向量相似度的方法，两个向量越相似，其点积越大。

### Softmax function：从实数空间到概率分布的标准映射方法

![hankcs.com 2017-06-07 下午3.13.57.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291024754.jpeg)

指数函数可以把实数映射成正数，然后归一化得到概率。

softmax之所叫softmax，是因为指数函数会导致较大的数变得更大，小数变得微不足道；这种选择作用类似于max函数。

### Skipgram

![2017-06-07_15-24-56.png](http://wx1.sinaimg.cn/large/006Fmjmcly1fgco3v2ca7j30pq0j7drt.jpg)

别看这张图有点乱，但其实条理很清晰，基本一图流地说明了问题。从左到右是one-hot向量，乘以center word的W于是找到词向量，乘以另一个context word的矩阵W'得到对每个词语的“相似度”，对相似度取softmax得到概率，与答案对比计算损失。真清晰。

官方笔记里有非手写版，一样的意思：

![Skip-Gram.png](http://wx1.sinaimg.cn/large/006Fmjmcly1fgcxbwd9wdj30ca0e8ju5.jpg)

这两个矩阵都含有V个词向量，也就是说同一个词有两个词向量，哪个作为最终的、提供给其他应用使用的embeddings呢？有两种策略，要么加起来，要么拼接起来。在CS224n的编程练习中，采取的是拼接起来的策略：

```python
# concatenate the input and output word vectors
wordVectors = np.concatenate(    
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),    
    axis=0)
# wordVectors = wordVectors[:nWords,:] + wordVectors[nWords:,:]
```

他们管W中的向量叫input vector，W'中的向量叫output vector。

## 训练模型：计算参数向量的梯度

把所有参数写进向量θθ，对d维的词向量和大小V的词表来讲，有：

![2017-06-07_15-34-33.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025026.jpeg)

由于上述两个矩阵的原因，所以θθ的维度中有个22。

模型的学习当然是梯度法了，Manning还耐心地推导了十几分钟：

![hankcs.com 2017-06-07 下午5.18.54.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025270.jpeg)

![hankcs.com 2017-06-07 下午5.19.15.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025573.jpeg)

![hankcs.com 2017-06-07 下午5.20.24.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025543.jpeg)

![hankcs.com 2017-06-07 下午5.20.42.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025349.jpeg)

更清晰的公式参考：http://www.hankcs.com/nlp/word2vec.html#h3-5 

### 损失/目标函数

梯度有了，参数减去梯度就能朝着最小值走了。

![hankcs.com 2017-06-07 下午8.34.41.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025236.jpeg)

### 梯度下降、SGD

![hankcs.com 2017-06-07 下午8.42.22.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025242.jpeg)

![hankcs.com 2017-06-07 下午8.40.33.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202109291025155.jpeg)

只有一句比较新鲜，神经网络喜欢嘈杂的算法，这可能是SGD成功的另一原因。

# 优秀博客

* [Word Embedding Papers | 经典再读之Word2Vec]([(2条消息) Word Embedding Papers | 经典再读之Word2Vec_Paper weekly-CSDN博客](https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/106774021))

