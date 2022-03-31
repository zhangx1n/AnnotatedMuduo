> [Graph Transformer Networks](https://arxiv.org/pdf/1911.06455v2.pdf)
>
> 韩国大学
>
> 2019



和HAN相比，GTN解决了不需要手动去定义meta-path，自动的学习出来。

对输入的异构图生成新的图结构并同时学习学到的图上的节点表示。学习的新的图结构需要识别有用的meta-path，即连接异构边的路径和多跳连接。

**Graph Transformer Layer:**
![image-20211216153843994](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112161538151.png)

A的每个通道是一种边的类型构成的邻接矩阵。

![image-20211216161622032](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202112161616140.png)

