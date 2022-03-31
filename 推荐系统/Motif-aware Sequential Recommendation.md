> [Motif-aware Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3404835.3463115)
>
>
> Zeyu Cui, Yinjiang Cai, Shu Wu, Xibo Ma, Liang Wang   中国科学院
>
> SIGIR 2021

**出发点：捕捉序列中的微观结构** micro-structure features

![image-20220328154440561](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220328154440561.png)

（1）是一种碰撞关系，一个人可能不会在Iphone之后再买一部手机。(2)表示单向依赖，因为用户在拥有笔记本电脑后可能需要移动HD和U盘，而很少在拥有U盘后再购买笔记本电脑。(3)是一种双向依赖关系，用户在拥有任何一个后，很有可能购买其他的产品。这些微观结构特征通过局部项之间的拓扑关系明确地模拟了下一项的概率。

三个节点可以有以下9种关系：

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/image-20220328154914896.png" alt="image-20220328154914896" style="zoom:50%;" />

### 怎么抽取这种Motif Feature 

预测下一个item $s_{t+1}^u$, 可以看成是从$s_t$到 $s_{t+1}$ 的概率。

定义一个集合set：
$$
H_{\Delta_{k}}\left(s_{t}^{u}, s_{t+1}^{u}\right)=\left[\left(s_{t}^{u}, m_{1}, s_{t+1}^{u}\right),\left(s_{t}^{u}, m_{2}, s_{t+1}^{u}\right), \ldots\right],
$$
中间的 $m$ 可以是任意的一个训练集中的一个item

下面的式子来表示一个 **$motif $**的重要性:
$$
\sigma\left(s_{t}^{u}, m, s_{t+1}^{u}\right)=A\left(s_{t}^{u}, m\right)+A\left(m, s_{t}^{u}\right)+A\left(s_{t+1}^{u}, m\right)+A\left(m, s_{t+1}^{u}\right)
$$
为每对item提取主题模型$\mathbf{X}_{k}\left(s_{t}^{u}, s_{t+1}^{u}\right)$,  $\mathbf{X}_{k}\left(s_{t}^{u}, s_{t+1}^{u}\right)$的每一个维度对应于一种主题模式
$$
\mathbf{X}_{k}\left(s_{t}^{u}, s_{t+1}^{u}\right)=\sum_{\left(s_{t}^{u}, m, s_{t+1}^{u}\right) \in H_{\Delta_{k}}\left(s_{t}^{u}, s_{t+1}^{u}\right)} \sigma\left(s_{t}^{u}, m, s_{t+1}^{u}\right)
$$

### MoSeR Framework

采用SASRec中的结构

通过叠加transformer块对用户行为序列进行建模。**序列表示和motif特征的拼接起来，作为预测层的输入**，输出被认为是用户对下一项的偏好。

