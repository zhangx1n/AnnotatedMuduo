> [Learning with Local and Global Consistency](https://proceedings.neurips.cc/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf)
>
> Dengyong Zhou, Olivier Bousquet
>
> NIPS 2003   4573次引用

[Learning with local and global consistency阅读报告NIPS2003_-CSDN博客](https://blog.csdn.net/u011070272/article/details/73606020)

这里所说的“smooth”是指：**在半监督学习问题中，算法学习到的分类目标函数，相对于标签样本和无标签样本所共同显示的内在结构，应该足够平滑（smooth）**。

算法基于两个重要的**假设**：

1. 空间中距离越近的点，越倾向于拥有同样的标签；—— local
2. 处于同一个结构（簇、流形等）的样本，倾向于拥有同样的标签。—— global



算法的核心思想：**让每一个样本的类标信息在空间中进行传递，直到达到某种合适的全局状态。**


![image-20211207213523116](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112072135323.png)

第一步： 对于一个图$G = (V, E)$,可以认为顶点$V是X, E是W$加权

第二步：D是度矩阵，确保$S$是半正定的，这是第三步迭代收敛的必要条件

第三步： 第一部分节点接收邻居的信息，第二部分保留自己原始的信息

![image-20211207221209707](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112072212817.png)

![image-20211207221654772](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112072216876.png)

![image-20211207221900841](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112072219932.png)