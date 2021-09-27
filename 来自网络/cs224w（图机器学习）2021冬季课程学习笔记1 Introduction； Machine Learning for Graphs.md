[诸神缄默不语-个人CSDN博文目录](https://blog.csdn.net/PolarisRisingWar/article/details/116396744)  
[cs224w（图机器学习）2021冬季课程学习笔记集合](https://blog.csdn.net/PolarisRisingWar/article/details/117287320)

1.  机器学习[1](#fn1)
2.  算法和图论
3.  概率论与数理统计

1.  图机器学习中的常用工具：NetworkX, PyTorch Geometric, DeepSNAP, GraphGym, SNAP.PY
2.  选择图的原因：图是用于描述并分析有关联/互动的实体的一种普适语言。它不将实体视为一系列孤立的点，而认为其互相之间有关系。它是一种很好的描述领域知识的方式。
3.  网络与图的分类
    1.  **networks / natural graphs**：自然表示为图
        1.  Social networks: Society is a collection of 7+ billion individuals
        2.  Communication and transactions: Electronic devices, phone calls, financial transactions
        3.  Biomedicine: Interactions between genes/proteins regulate life（大概是基因或蛋白质之间互动从而调节生理活动的过程）
        4.  Brain connections: Our thoughts are hidden in the connections between billions of neurons
    2.  **graphs**：作为一种表示方法
        1.  Information/knowledge are organized and linked
        2.  Software can be represented as a graph
        3.  Similarity networks: Connect similar data points
        4.  Relational structures: Molecules, Scene graphs, 3D shapes, Particle-based physics simulations
    3.  有时network和graph之间的差别是模糊的
4.  复杂领域会有丰富的关系结构，可以被表示为**关系图**relational graph，通过显式地建模关系，可以获得更好的表现
5.  但是现代深度学习工具常用于建模简单的序列sequence（如文本、语音等具有线性结构的数据）和grid（图片具有平移不变性，可以被表示为fixed size grids或fixed size standards）
    
    这些传统工具很难用于图的建模，其难点在于网络的复杂：  
    ①Arbitrary size and complex topological structure (_i.e._, no spatial locality[2](#fn2) like grids)  
    ②没有基准点，没有节点固定的顺序。没有那种上下左右的方向  
    ③经常出现动态的图，而且会有多模态的特征
    
6.  本课程中就要讲述如何将神经网络模型适用范围拓展到图上。图深度学习也是当前的新前沿领域。![](https://img-blog.csdnimg.cn/20210526125823432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    
7.  有监督机器学习全流程图：![](https://img-blog.csdnimg.cn/20210526125854173.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    在传统机器学习流程中，我们需要对原始数据进行**特征工程**feature engineering（比如手动提取特征等），但是现在我们使用**表示学习**representation learning的方式来自动学习到数据的特征，直接应用于下流预测任务。
8.  图的表示学习：大致来说就是将原始的节点（或链接、或图）表示为向量（嵌入**embedding**），图中相似的节点会被embed得靠近（指同一实体，在节点空间上相似，在向量空间上就也应当相似）![](https://img-blog.csdnimg.cn/20210526130351484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    
9.  cs224w本课程将聚焦图的机器学习和表示学习多个领域，课程大纲如下：
    1.  Traditional methods: Graphlets, Graph Kernels
    2.  Methods for node embeddings: DeepWalk, Node2Vec
    3.  Graph Neural Networks: GCN, GraphSAGE, GAT, Theory of GNNs
    4.  Knowledge graphs and reasoning: TransE, BetaE
    5.  Deep generative models for graphs
    6.  Applications to Biomedicine, Science, Industry

1.  图机器学习任务分成四类![](https://img-blog.csdnimg.cn/2021052613083731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    1.  节点级别 node level
    2.  边级别 edge level
    3.  社区 / 子图级别 community(subgraph) level
    4.  图级别，包括预测任务 graph-level prediction 和 图生成任务 graph generation
2.  各类型的典型任务
    1.  Node classification: Predict a property of a node  
        **Example**: Categorize online users / items
    2.  Link prediction: Predict whether there are missing links between two nodes  
        **Example**: Knowledge graph completion
    3.  Graph classification: Categorize different graphs  
        **Example**: Molecule property prediction
    4.  Clustering: Detect if nodes form a community  
        **Example**: Social circle detection
    5.  **Other tasks**:
        1.  Graph generation: Drug discovery
        2.  Graph evolution: Physical simulation
3.  node-level的例子：解决蛋白质折叠问题——AlphaFold[3](#fn3)  
    大致来说就是蛋白质由一系列氨基酸（氨基酸链chains of amino acids或amino acid residues）结合而成，这些氨基酸之间的交互使其形成不同的折叠方式，组成三维蛋白质结构（这个组合方式很复杂，但是一系列氨基酸交互之后形成的结构就是唯一的，所以理论上可以预测出最终结果）。学习任务就是输入一系列氨基酸，预测蛋白质的3D结构。  
    AlphaFold将一个被折叠的蛋白质视作spatial graph，residue视作节点，在相近的residue之间建立边，形成图结构，搭建深度学习模型，预测节点在空间中的位置（也就是蛋白质的三维结构）。
4.  edge-level的例子
    1.  推荐系统——PinSage[4](#fn4)  
        推荐系统的任务是向用户推荐物品![](https://img-blog.csdnimg.cn/20210526153519759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
        PinSage：基于图的推荐系统![](https://img-blog.csdnimg.cn/2021052615382957.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        在网站Pinterest上给用户推荐相关图片。  
        节点是图片和用户，组成如左下角所示的bipartite图（这个概念将在本章后文讲解）。图中的d是嵌入向量之间的距离，即任务目标是使相似节点嵌入之间的距离比不相似节点嵌入之间的距离更小。
        
    2.  在同时吃2种药的情况下预测药的副作用  
        背景：很多人需要同时吃多种药来治疗多种病症。  
        任务：输入一对药物，预测其有害副作用。
        
        Biomedical Graph Link Prediction[5](#fn5)![](https://img-blog.csdnimg.cn/20210526155010416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        ![](https://img-blog.csdnimg.cn/20210526155047922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        （预测概率最高的组合药物副作用。其中有一部分已经有论文证明存在）
        
5.  subgraph-level的例子：Traffic Prediction[6](#fn6)  
    Google Map预测一段路程的长度、耗时等：将路段建模成图，在每个子图上建立预测模型。
6.  graph-level的例子
    1.  药物发现——用图神经网络的图分类任务来从一系列备选图（分子被表示为图，节点是原子，边是化学键）中预测最有可能是抗生素的分子[7](#fn7)  
        ![](https://img-blog.csdnimg.cn/20210526182118560.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        ![](https://img-blog.csdnimg.cn/20210526182334707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
    2.  Graph Generation —— Molecule Generation / Optimization[8](#fn8)![](https://img-blog.csdnimg.cn/20210526184542718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
    3.  Graph Evolution —— Physics Simulation（粒子间的相互作用）[9](#fn9)  
        将整个物质表示为图（proximity graph），用GNN来预测粒子的下一步活动（组成一个新位置、新图）![](https://img-blog.csdnimg.cn/20210526184747990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        ![](https://img-blog.csdnimg.cn/2021052618485795.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        

1.  图的组成成分
    1.  节点（N或V）
    2.  链接 / 边（E）
    3.  网络 / 图（G）![](https://img-blog.csdnimg.cn/20210526185255926.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
2.  图是一种解决关系问题时的通用语言，各种情况下的统一数学表示。将问题抽象成图，可以用同一种机器学习算法解决所有问题。![](https://img-blog.csdnimg.cn/20210526185528849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    
3.  但为问题选择合适的表示方法是个很难的任务。  
    如图中举例，用论文间的引用作为关系就比用论文题目含有同一单词作为关系，表达能力会好很多。![](https://img-blog.csdnimg.cn/20210526185704583.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
    
4.  建图时需要考虑以什么作为节点，以什么作为边  
    对某一领域或问题选择合适的网络表示方法会决定我们能不能成功使用网络：
    1.  有些情况下只有唯一的明确做法
    2.  有些情况下可以选择很多种做法
    3.  The way you assign links will determine the nature of the question you can study
5.  以下介绍一些design choice
    1.  有向图VS无向图![](https://img-blog.csdnimg.cn/20210526190621182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
    2.  度数degree：每个点连接了多少边
        
        1.  无向图：Avg. degree: k ‾ = ⟨ k ⟩ = 1 N ∑ i = 1 N k i = 2 E N \\overline{k}=\\langle k\\rangle=\\frac{1}{N}\\sum\\limits\_{i=1}^Nk\_i=\\frac{2E}{N} k\=⟨k⟩\=N1​i\=1∑N​ki​\=N2E​
        2.  有向图：分成 in-degree 和 out-degree ，(total) degree是二者之和![](https://img-blog.csdnimg.cn/20210712113142147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
              
            参考了一下评论，感觉有向图跟无向图要作区分的话，更方便的还是看B点：在无向图中其度数就是2，在有向图中其度数就是3，所以在平均度数的计算上会有所差别。
    3.  Bipartite Graph  
        类似地，还有multipartite的情况 ![](https://img-blog.csdnimg.cn/20210526191134346.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
    4.  Folded/Projected Bipartite Graphs![](https://img-blog.csdnimg.cn/20210526203223284.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        如图所示，就是将一个bipartite图的两个节点子集分别投影  
        projection图上两个节点之间有连接：这两个节点在folded/projected bipartite graphs上至少有一个共同邻居
        
    5.  Representing Graphs
        
        1.  邻接矩阵Adjacency Matrix：每一行/列代表一个节点，如果节点之间有边就是1，没有就是0![](https://img-blog.csdnimg.cn/20210526203850817.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
            无向图的邻接矩阵天然对称
            
            网络的邻接矩阵往往是稀疏矩阵，有很多0，即其 E < < E m a x E<<E\_{max} E<<Emax​ or k < < N − 1 k<<N-1 k<<N−1  
            定义：Density of the matrix ( E N 2 \\frac{E}{N^2} N2E​)
            
        2.  Edge list![](https://img-blog.csdnimg.cn/2021052620393511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
            这种方式常用于深度学习框架中，因为可以将图直接表示成一个二维矩阵。这种表示方法的问题在于很难进行图的操作和分析，就算只是计算图中点的度数都会很难
        3.  Adjacency list![](https://img-blog.csdnimg.cn/20210526204043389.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
            对图的分析和操作更方便
    6.  节点和边的属性，可选项：
        
        1.  Weight (e.g., frequency of communication)
        2.  Ranking (best friend, second best friend…)
        3.  Type (friend, relative, co-worker)
        4.  Sign: Friend vs. Foe, Trust vs. Distrust
        5.  Properties depending on the structure of the rest of the graph: Number of common friends
    7.  Weighted / Unweighted![](https://img-blog.csdnimg.cn/20210526204842780.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        
    8.  Self-edges (self-loops) / Multigraph![](https://img-blog.csdnimg.cn/20210526204924432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
        这个multigraph有时也可被视作是weighted graph，就是说将多边的地方视作一条边的权重（在邻接矩阵上可看出效果是一样的）。但有时也可能就是想要分别处理每一条边，这些边上可能有不同的property和attribute
        
    9.  Connectivity
        
        1.  无向图的Connectivity
            1.  connected：任意两个节点都有路径相通
            2.  disconnected：由2至多个connected components构成  
                最大的子连接图：giant component  
                isolated node  
                这种图的邻接矩阵可以写成block-diagonal的形式，数字只在connected components之中出现![](https://img-blog.csdnimg.cn/20210526205325953.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
                
        2.  有向图的Connectivity
            1.  strongly connected directed graph: has a path from each node to every other node and vice versa (e.g., A-B path and B-A path)
            2.  weakly connected directed graph: is connected if we disregard the edge directions![](https://img-blog.csdnimg.cn/20210526205531620.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
                
            3.  strongly connected components![](https://img-blog.csdnimg.cn/20210526205602675.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
                

*   **Machine learning with Graphs**
    *   Applications and use cases
*   **Different types of tasks:**
    *   Node level
    *   Edge level
    *   Graph level
*   **Choice of a graph representation:**
    *   Directed, undirected, bipartite, weighted, adjacency matrix

* * *

1.  机器学习可以看李宏毅老师的课程入门。我之前看的是2017版的，2021年时看了李老师最新版深度学习课程并撰写了笔记，可以参考：[李宏毅2021春季机器学习课程视频笔记集合](https://blog.csdn.net/PolarisRisingWar/article/details/117229529) [↩︎](#fnref1)
    
2.  spatial locality空间局部性：当程序访问某存储器地址后，很可能马上访问其邻近地址的特性。（[空间局部性\_百度百科](https://baike.baidu.com/item/%E7%A9%BA%E9%97%B4%E5%B1%80%E9%83%A8%E6%80%A7/56102310?fr=aladdin)）  
    这个概念我有点没搞懂放在这里是什么意思……  
    参考资料：[局部性原理浅析——良好代码的基本素质 - Geek\_Ling - 博客园](https://www.cnblogs.com/yanlingyin/archive/2012/02/11/2347116.html) [↩︎](#fnref2)
    
3.  课程中提供的参考资料：  
    DeepMind博文：[AlphaFold: Using AI for scientific discovery](https://deepmind.com/blog/article/AlphaFold-Using-AI-for-scientific-discovery)[10](#fn10)  
    DeepMind博文：[AlphaFold: a solution to a 50-year-old grand challenge in biology](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)[11](#fn11)  
    SingularityHub文章：[DeepMind’s AlphaFold Is Close to Solving One of Biology’s Greatest Challenges](https://singularityhub.com/2020/12/15/deepminds-alphafold-is-close-to-solving-one-of-biologys-greatest-challenges/)[12](#fn12) [↩︎](#fnref3)
    
4.  Ying et al., Graph Convolutional Neural Networks for Web-Scale Recommender Systems, KDD 2018  
    [论文地址](https://arxiv.org/pdf/1806.01973.pdf)  
    是一个全网级别的大型推荐系统，使用random walk和图卷积神经网络，结合图结构和节点特征来产生节点嵌入 [↩︎](#fnref4)
    
5.  Zitnik et al., Modeling Polypharmacy Side Effects with Graph Convolutional Networks, Bioinformatics 2018  
    [论文地址](https://arxiv.org/pdf/1802.00543.pdf)  
    提出模型Decagon，建立了一个由蛋白质-蛋白质交互、蛋白质-药物交互、药物-药物交互（即对应成对药物的副作用，每一种副作用都是一种边的类型）组成的多模态图。文章提出了一种用于多模态网络多关系预测的图卷积神经网络模型 [↩︎](#fnref5)
    

7.  论文链接：[A Deep Learning Approach to Antibiotic Discovery](https://www.sciencedirect.com/science/article/pii/S0092867420301021)  
    由于抗药性细菌大量出现，所以产生了对抗生素发现的更多需求。本文建立了一种深度学习模型来找抗生素。 [↩︎](#fnref7)
    
8.  论文链接：[You et al., Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation, NeurIPS 2018](https://arxiv.org/pdf/1806.02473.pdf)  
    任务：基于某些给定条件（满足物理定律，如化学价等），生成优化某些目标（某些需要的特性，如是药的概率、人造可得性）的图结构（分子）。  
    本文提出了一个Graph Convolutional Policy Network (GCPN)，一个通过强化学习完成上述任务的图卷积网络。训练过程遵守特定领域规则，以特定领域的reward来做优化。 [↩︎](#fnref8)
    
9.  论文链接：[Sanchez-Gonzalez et al., Learning to simulate complex physics with graph networks, ICML 2020](https://arxiv.org/pdf/2002.09405.pdf) [↩︎](#fnref9)
    

11.  这篇相当于是前一篇文章的进阶版，提出了课程中所讲述的最新版本、效果最好的AlphaFold。  
    阅读笔记：  
    Christian Anfinsen于1972年就提出，蛋白质的氨基酸链可以完全决定其结构。此后五十年研究者们就一直致力于解决这一问题。  
    CASP竞赛提供的评估指标是GDT（Global Distance Test），这一指标简单来讲可以被认为是amino acid residues (beads in the protein chain)预测值与真实值空间距离小于某一误差的百分比例。超过90%可以认为是解决了这一问题，AlphaFold2成功达到这一指标。  
    下图是比较成功的预测的示例图：  
    ![](https://img-blog.csdnimg.cn/20210526142032330.gif#pic_center)
      
    模型是一个端到端的attention-based neural network system，来理解这个图结构，在其建立的隐式图上进行推理。它使用了evolutionarily related sequences, multiple sequence alignment (MSA), and a representation of amino acid residue pairs来改进图。![](https://img-blog.csdnimg.cn/20210526145932636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1BvbGFyaXNSaXNpbmdXYXI=,size_16,color_FFFFFF,t_70)
      
    原文图片配文：An overview of the main neural network model architecture. The model operates over evolutionarily related protein sequences as well as amino acid residue pairs, iteratively passing information between both representations to generate a structure. [↩︎](#fnref11)
    
12.  类似于新闻稿。  
    解释了蛋白质结构对药物发现领域的意义。原文：Nearly all of our drugs are designed to dock onto a protein, like keys to a lock. [↩︎](#fnref12)