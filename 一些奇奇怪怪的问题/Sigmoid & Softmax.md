Sigmoid 和 Softmax 是在逻辑回归和神经网络中常用的两个函数，初学时经常会对二者的差异和应用场景产生疑惑。

　　Sigmoid 函数形式为：

$\begin{equation}
S(x) = \frac{1}{1 + e^{-x}}
\end{equation}$

　　Sigmoid 是一个可微的有界函数，在各点均有非负的导数。当 $x \rightarrow \infty$ 时，$S(x) \rightarrow 1$；当 $x \rightarrow -\infty$ 时，$S(x) \rightarrow 0$。常用于二元分类（Binary Classification）问题，以及神经网络的激活函数（Activation Function）（把线性的输入转换为非线性的输出）。

　　Softmax 函数形式为：

$\begin{equation}
S(x_j) = \frac{e^{x_j}}{\sum_{k=1}^K e^{x_k}}, j = 1, 2, …, K
\end{equation}$

　　对于一个长度为 K 的任意实数矢量，Softmax 可以把它压缩为一个长度为 K 的、取值在 (0, 1) 区间的实数矢量，且矢量中各元素之和为 1。它在多元分类（Multiclass Classification）和神经网络中也有很多应用。Softmax 不同于普通的 max 函数：max 函数只输出最大的那个值，而 Softmax 则确保较小的值也有较小的概率，不会被直接舍弃掉，是一个比较“Soft”的“max”。

　　在二元分类的情况下，对于 Sigmod，有：

$\begin{equation}
p(y = 1 | x) = \frac{1}{1 + e^{-\theta^Tx}}
\end{equation}$

$\begin{equation}
p(y = 0 | x) = 1 – p(y = 1 | x) = \frac{e^{-\theta^Tx}}{1 + e^{-\theta^Tx}}
\end{equation}$

　　而对 $K = 2$ 的 Softmax ，有：

$\begin{equation}
p(y = 1|x) = \frac{e^{\theta_1^Tx}}{e^{\theta_0^Tx} + e^{\theta_1^Tx}} = \frac{1}{1 + e^{(\theta_0^T – \theta_1^T)x}} = \frac{1}{1 + e^{-\beta x}}
\end{equation}$

$\begin{equation}
p(y = 0|x) = \frac{e^{\theta_0^Tx}}{e^{\theta_0^Tx} + e^{\theta_1^Tx}} = \frac{e^{(\theta_0^T-\theta_1^T)x}}{1 + e^{(\theta_0^T-\theta_1^T)x}} = \frac{e^{-\beta x}}{1 + e^{-\beta x}}
\end{equation}$

　　其中

$\begin{equation}
\beta = -(\theta_0^T – \theta_1^T)
\end{equation}$

　　可见在二元分类的情况下，Softmax 退化为了 Sigmoid。



Sigmoid =多标签分类问题=多个正确答案=非独占输出

构建分类器，解决有多个正确答案的问题时，用Sigmoid函数分别处理各个原始输出值。

Softmax =多类别分类问题=只有一个正确答案=互斥输出