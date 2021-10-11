# Node Classification

* Given labels of some nodes
* Let's predict labels of unlabeled nodes
* This is called semi-supervised nodes

![image-20211002153524307](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021535447.png)



**Correlations**: 

* nearby nodes have the same color (belonging to the same class)

<img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021539152.png" alt="image-20211002153926101" style="zoom:50%;" />

* Main types of dependencies that lead to correlation:

    <img src="https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021540134.png" alt="image-20211002154031083" style="zoom:50%;" />

![image-20211002154451006](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021544125.png)

> "Birds of a feather flock together"     物以类聚，人以群分

![image-20211002154616460](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021546534.png)

## Classification with Network Data

![image-20211002160014222](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021600308.png)

### Motivation

* Similar nodes are typically close together or directly connected in the network:

    ![image-20211002160500781](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021605854.png)

> 因此根据关联推定**guilt-by-association**：如果我与具有标签X的节点相连，那么我也很可能具有标签X（基于马尔科夫假设）
> 举例：互联网中的恶意/善意网页：恶意网页往往会互相关联，以增加曝光，使其看起来更可靠，并在搜索引擎中提高排名。

![image-20211002160713125](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021607189.png)

### Semi-supervised Learning

![image-20211002160849495](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021608575.png)

![image-20211002160915596](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021609677.png)

### Approach: Collective Classification

1. **collective classification的应用**

    1. Document classification
    2. 词性标注Part of speech tagging
    3. Link prediction
    4. 光学字符识别Optical character recognition
    5. Image/3D data segmentation
    6. 实体解析Entity resolution in sensor networks
    7. 垃圾邮件Spam and fraud detection
    8. collective classification概述

2. **collective classification概述**

    使用网络中的关系同时对相连节点进行分类

    概率框架propabilistic framework

    马尔科夫假设

![image-20211002162549779](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021625873.png)

​		按算法的程度由低到高，节点分类算法可以分为**Relational Classification, Iterative Classification 和 Belief Propagation三类**。这三类算法都基于相同的假设——**节点 $i$ 的label取决于节点 ![[公式]](https://www.zhihu.com/equation?tex=i) 的邻居们的label**，三类算法的计算过程也相同：

​	（1）初始化每个节点的label；

​	（2）设定相连节点之间的相互作用；

​	（3）根据连接关系在全网多次传播信息，直到全网达到收敛状态。

![image-20211002163004599](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110021630706.png)

#### Relational classifiers

![image-20211003161213941](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031612151.png)

![image-20211003164122608](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031641696.png)



![image-20211003165208271](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031652371.png)

![image-20211003165257616](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031652717.png)

![image-20211003165337079](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031653185.png)



![image-20211003165349942](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031653044.png)

![image-20211003165415123](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031654218.png)

![image-20211003165430496](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031654586.png)

![image-20211003165459201](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031654294.png)



![image-20211003165523952](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031655056.png)







#### Iterative calssification

![image-20211003165721708](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031657773.png)

![image-20211003170325158](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110031703256.png)



* 训练两个分类器：
    * $ϕ_1(f_v)$基于节点特征向量$f_v$预测节点标签
    * $ϕ_2(f_v, z_v)$基于节点特征向量$f_v$和邻居节点标签summary$z_v$预测节点标签



![image-20211007213446856](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072134997.png)

![image-20211007214010403](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072140507.png)



![image-20211007214030244](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072140357.png)

![image-20211007214221665](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072142760.png)

![image-20211007214309389](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072143492.png)

![image-20211007214801234](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072148334.png)

![image-20211007214819188](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072148280.png)

![image-20211007214831718](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072148818.png)

![image-20211007214842541](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072148643.png)

![image-20211007214859596](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072148716.png)

![image-20211007214915147](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072149245.png)

![image-20211007214928259](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072149356.png)

![image-20211007214938812](https://cdn.jsdelivr.net/gh/Zhangxin98/Note@main/img/202110072149904.png)









#### Loopy belief propagation

