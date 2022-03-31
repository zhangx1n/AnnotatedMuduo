> 论文：[Neural Graph Collaborative Filtering (arxiv.org)](https://arxiv.org/pdf/1905.08108.pdf)
>
> 作者：何向南组
>
> 来源：SIGIR 2019

------

首先利用用户和物品的交互矩阵完成图的初始化，利用用户和物品的交互构造图：

![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040909536.png)

物品和用户的初始化Embedding:

![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910761.png)

**一层结构：**

![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910972.png)

第一步信息构建：利用i来构建u的向量（反过来也行）
![image.png](https://statics.sdk.cn/articles/img/202011/117ed0fe45ee3a444706942534_1001060.png?x-oss-process=style/thumb)



第二步：信息聚合：将所有邻居求和，再加上自己。利用邻节点来更新自己的向量

![image.png](https://statics.sdk.cn/articles/img/202011/117ed0fe45ee3a447188188145_1001060.png?x-oss-process=style/thumb)

最终：
![image20201102134752115.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910880.png)

**多层结构: **

![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910969.png)

按阶数来分别更新向量，第$I$阶用户向量为：
![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910902.png)
其中，根据上一阶的向量来初始化这一阶的向量：
![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910910.png)
其中：
![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910673.png)

**矩阵形式计算**

其中$E^{(l)}$是用户+物品向量矩阵，不断更新这个矩阵可以得到最终用户和物品的向量

![image.png](https://statics.sdk.cn/articles/img/202011/117ed0fe45ee3a440459516823_1001060.png?x-oss-process=style/thumb)

其中L和A为：A就是根据用户-物品交互矩阵R算出的

![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040916504.png)

最后L非对角非零元素为：
![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910739.png)

最后把所有阶的进行向量拼接得到最终的用户和物品向量：
![image.png](https://statics.sdk.cn/articles/img/202011/117ed0fe45ee3a442702800832_1001060.png?x-oss-process=style/thumb)

内积得到评分：
![image.png](https://statics.sdk.cn/articles/img/202011/117ed0fe45ee3a449321422310_1001060.png?x-oss-process=style/thumb)

根据BRP算Pair-wise损失：
![image.png](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202201040910889.png)