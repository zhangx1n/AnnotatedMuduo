# Cora、Citeseer、Pubmed

| 数据集   | 来源                                                         | 图数量 | 节点  | 边    | 特征 | 标签 |
| -------- | ------------------------------------------------------------ | ------ | ----- | ----- | ---- | ---- |
| Cora     | “Collective classification in network data,” AI magazine,2008 | 1      | 2708  | 5429  | 1433 | 7    |
| CiteSeer | “Collective classification in network data,” AI magazine,2008 | 1      | 3327  | 4732  | 3703 | 6    |
| Pubmed   | “Collective classification in network data,” AI magazine,2008 | 1      | 19717 | 44338 | 500  | 3    |

```
├── gcn
│   ├── data          //图数据
│   │   ├── ind.citeseer.allx
│   │   ├── ind.citeseer.ally
│   │   ├── ind.citeseer.graph
│   │   ├── ind.citeseer.test.index
│   │   ├── ind.citeseer.tx
│   │   ├── ind.citeseer.ty
│   │   ├── ind.citeseer.x
│   │   ├── ind.citeseer.y
│   │   ├── ind.cora.allx
│   │   ├── ind.cora.ally
│   │   ├── ind.cora.graph
│   │   ├── ind.cora.test.index
│   │   ├── ind.cora.tx
│   │   ├── ind.cora.ty
│   │   ├── ind.cora.x
│   │   ├── ind.cora.y
│   │   ├── ind.pubmed.allx
│   │   ├── ind.pubmed.ally
│   │   ├── ind.pubmed.graph
│   │   ├── ind.pubmed.test.index
│   │   ├── ind.pubmed.tx
│   │   ├── ind.pubmed.ty
│   │   ├── ind.pubmed.x
│   │   └── ind.pubmed.y
│   ├── __init__.py
│   ├── inits.py    //初始化的公用函数
│   ├── layers.py   //GCN层定义
│   ├── metrics.py  //评测指标的计算
│   ├── models.py   //模型结构定义
│   ├── train.py    //训练
│   └── utils.py    //工具函数的定义
├── LICENCE
├── README.md
├── requirements.txt
└── setup.py
```

**三种数据都由以下八个文件组成：**

```
ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances 
    (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    
ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

All objects above must be saved using python pickle module.
    
以cora为例：
ind.dataset_str.x => 训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)
ind.dataset_str.tx => 测试实例的特征向量,shape:(1000, 1433)
ind.dataset_str.allx => 有标签的+无无标签训练实例的特征向量，是ind.dataset_str.x的超集，shape:(1708, 1433)

ind.dataset_str.y => 训练实例的标签，独热编码，numpy.ndarray类的实例，是numpy.ndarray对象，shape：(140, 7)
ind.dataset_str.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例,shape:(1000, 7)
ind.dataset_str.ally => 对应于ind.dataset_str.allx的标签，独热编码,shape:(1708, 7)

ind.dataset_str.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
ind.dataset_str.test.index => 测试实例的id，2157行

上述文件必须都用python的pickle模块存储
```

- Semi-Supervised Classification with Graph Convolutional Networks论文中的GCN是半监督学习，因此训练数据集中有的有标签有的没有标签

**以Cora为例**
原始数据集链接：http://linqs.cs.umd.edu/projects/projects/lbc/
数据集划分方式：https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016)

Cora数据集由机器学习论文组成，是近年来图深度学习很喜欢使用的数据集。在数据集中，论文分为以下七类之一:

* 基于案例
* 遗传算法
* 神经网络
* 概率方法
* 强化学习
* 规则学习
* 理论

论文的选择方式是，在最终语料库中，每篇论文引用或被至少一篇其他论文引用。整个语料库中有2708篇论文。

在词干堵塞和去除词尾后，只剩下1433个独特的单词。文档频率小于10的所有单词都被删除。**cora数据集包含1433个独特单词，所以特征是1433维。0和1描述的是每个单词在paper中是否存在。**

变量data是个scipy.sparse.csr.csr_matrix，类似稀疏矩阵，输出得到的是矩阵中非0的行列坐标及值


**数据格式实例**

```python
(1)--------------------------------------ind.cora.x
def load_cora():
    names = ['x']
    with open("data/ind.cora.x", 'rb') as f:
        if sys.version_info > (3, 0):
            print(f)  # <_io.BufferedReader name='data/ind.cora.x'>
            data = pkl.load(f, encoding='latin1')
            print(type(data)) #<class 'scipy.sparse.csr.csr_matrix'>

            print(data.shape)   #(140, 1433)-ind.cora.x是140行，1433列的
            print(data.shape[0]) #row:140
            print(data.shape[1]) #column:1433
            print(data[1])
  # 变量data是个scipy.sparse.csr.csr_matrix，类似稀疏矩阵，输出得到的是矩阵中非0的行列坐标及值
  # (0, 19)	1.0
  # (0, 88)	1.0
  # (0, 149)	1.0
  # (0, 212)	1.0
  # (0, 233)	1.0
  # (0, 332)	1.0
  # (0, 336)	1.0
  # (0, 359)	1.0
  # (0, 472)	1.0
  # (0, 507)	1.0
  # (0, 548)	1.0
  # ...

# print(data[100][1]) #IndexError: index (1) out of range
            nonzero=data.nonzero()
            print(nonzero)     #输出非零元素对应的行坐标和列坐标
# (array([  0,   0,   0, ..., 139, 139, 139], dtype=int32), array([  19,   81,  146, ..., 1263, 1274, 1393], dtype=int32))
            # nonzero是个tuple
            print(type(nonzero)) #<class 'tuple'>
            print(nonzero[0])    #行：[  0   0   0 ... 139 139 139]
            print(nonzero[1])    #列：[  19   81  146 ... 1263 1274 1393]
            print(nonzero[1][0])  #19
            print(data.toarray())
# [[0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 1. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]

(2)--------------------------------------ind.cora.y

def load_cora():
    with open("data/ind.cora.y", 'rb') as f:
        if sys.version_info > (3, 0):
            print(f)  #<_io.BufferedReader name='data/ind.cora.y'>
            data = pkl.load(f, encoding='latin1')
            print(type(data)) #<class 'numpy.ndarray'>
            print(data.shape)   #(140, 7)
            print(data.shape[0]) #row:140
            print(data.shape[1]) #column:7
            print(data[1]) #[0 0 0 0 1 0 0]
            
(3)--------------------------------------ind.cora.graph

def load_cora():
    with open("data/ind.cora.graph", 'rb') as f:
        if sys.version_info > (3, 0):
            data = pkl.load(f, encoding='latin1')
            print(type(data)) #<class 'collections.defaultdict'>
            print(data) 
# defaultdict(<class 'list'>, {0: [633, 1862, 2582], 1: [2, 652, 654], 2: [1986, 332, 1666, 1, 1454], 
#   , ... , 
#   2706: [165, 2707, 1473, 169], 2707: [598, 165, 1473, 2706]})


(4)--------------------------------------ind.cora.test.index

test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
print("test index:",test_idx_reorder)
#test index: [2692, 2532, 2050, 1715, 2362, 2609, 2622, 1975, 2081, 1767, 2263,..]
print("min_index:",min(test_idx_reorder))
# min_index: 1708

(5)citeseer数据集中一些孤立点的特殊处理
    #处理citeseer中一些孤立的点
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        # print("test_idx_range_full.length",len(test_idx_range_full))
        #test_idx_range_full.length 1015

        #转化成LIL格式的稀疏矩阵,tx_extended.shape=(1015,1433)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # print(tx_extended)
        #[2312 2313 2314 2315 2316 2317 2318 2319 2320 2321 2322 2323 2324 2325
        # ....
        # 3321 3322 3323 3324 3325 3326]

        #test_idx_range-min(test_idx_range):列表中每个元素都减去min(test_idx_range)，即将test_idx_range列表中的index值变为从0开始编号
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        # print(tx_extended.shape) #(1015, 3703)

        # print(tx_extended)
        # (0, 19) 1.0
        # (0, 21) 1.0
        # (0, 169) 1.0
        # (0, 170) 1.0
        # (0, 425) 1.0
        #  ...
        # (1014, 3243) 1.0
        # (1014, 3351) 1.0
        # (1014, 3472) 1.0

        tx = tx_extended
        # print(tx.shape)
        # (1015, 3703)
        #997,994,993,980,938...等15行全为0


        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
        # for i in range(ty.shape[0]):
        #     print(i," ",ty[i])
        #     # 980 [0. 0. 0. 0. 0. 0.]
        #     # 994 [0. 0. 0. 0. 0. 0.]
        #     # 993 [0. 0. 0. 0. 0. 0.]
```

* allx是训练集中的所有训练实例，包含有标签的和无标签的，从0-1707，共1708个
* ally是allx对应的标签，从1708-2707，共1000个
* citeseer的测试数据集中有一些孤立的点（test.index中没有对应的索引，15个），可把这些点当作特征全为0的节点加入到测练集tx中，并且对应的标签在ty中
* 输入是一张整图，因此将tx和allx拼起来作为feature
* 没有标签的数据的y值:[0,0,0,0,0,0,0]
* 数据集中的特征也是稀疏的，用LIL稀疏矩阵存储，格式如下
    

```python
A=np.array([[1,0,2,0],[0,0,0,0],[3,0,0,0],[1,0,0,4]])
AS=sp.lil_matrix(A)
print(AS)
# (0, 0) 1
# (0, 2) 2
# (2, 0) 3
# (3, 0) 1
# (3, 3) 4
```

# GCN的Benchmark数据集溯源

[ GCN的Benchmark数据集溯源](https://blog.csdn.net/w55100/article/details/109360817)

关于数据集的真实性，来源，第二版划分处理的合理性