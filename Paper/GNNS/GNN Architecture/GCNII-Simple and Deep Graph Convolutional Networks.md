> **paper:**[Simple and Deep Graph Convolutional Networks (arxiv.org)](https://arxiv.org/pdf/2007.02133.pdf)
>
> **code:**[chennnM/GCNII: PyTorch implementation of "Simple and Deep Graph Convolutional Networks" (github.com)](https://github.com/chennnM/GCNII)



------

为了解决过平滑问题，提出了GCNII。使用了两个简单但是有效的技术：***Initial residual** and **Identity mapping***

**Model：**
$$
\mathbf{H}^{(\ell+1)}=\sigma\left(\left(\left(1-\alpha_{\ell}\right) \tilde{\mathbf{P}} \mathbf{H}^{(\ell)}+\alpha_{\ell} \mathbf{H}^{(0)}\right)\left(\left(1-\beta_{\ell}\right) \mathbf{I}_{n}+\beta_{\ell} \mathbf{W}^{(\ell)}\right)\right)
$$
其中$\alpha_\ell, \beta_\ell$是超参数，$\tilde{\mathbf{P}}=\tilde{\mathbf{D}}^{-1 / 2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1 / 2}$

