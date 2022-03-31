> **标题:** [Bilinear Graph Neural Network with Neighbor Interactions](https://www.ijcai.org/Proceedings/2020/0202.pdf)
>
> **期刊/会议：**IJCAI 2020
>
> **作者：**中科大



## 摘要

图神经网络（GNN）是一个功能强大的模型，可用于学习表示形式并对图形数据进行预测。对GNN的现有工作已将图卷积定义为所连接节点的特征的加权和，以形成目标节点的表示形式。然而，加权和的运算假设相邻节点彼此独立，并且忽略它们之间可能的交互。当存在这样的交互时，例如两个邻居节点的同时出现是目标节点特征的强烈信号，现有的GNN模型可能无法捕获该信号。在这项工作中，我们认为在GNN中对相邻节点之间的交互进行建模是十分重要的。我们提出了一种新的图卷积算子，该算子通过邻居节点表示的成对交互来增加加权和。我们将此框架称为双线性图神经网络（ Bilinear Graph Neural Network ，BGNN），该框架可通过相邻节点间的双线性交互双线性来提高GNN表示能力。特别是，我们分别基于著名的GCN和GAT指定了两个名为BGCN和BGAT的BGNN模型。关于三个半监督节点分类的公开基准的实证结果证明了BGNN的有效性-BGCN（BGAT）在分类准确度方面比GCN（GAT）高1.6％（1.5％）。

## Introduction

GNN是一种在图结构上学习节点表征的神经网络。与传统的只使用节点信息特征的模型相比能够学到更全面的节点表征。图神经网络在

## Bilinear Graph Neural Network
