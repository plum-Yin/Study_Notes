[toc]

# 一、线性回归

Mean Squared Error (MSE):
$$
\frac{1}{n} \sum_{j=1}^{n}\left(y^{(j)}-\sum_{i=0}^{p} b_{i} x_{i}^{(j)}\right)^{2}
$$
Coefficients $b_{i}$ can be derived using calculus.

方差
$$
{\rm Var}(X)=E{\Large(}[X-E(X)]^2{\Large)}=E(X^2)-E^2(X)
$$
相关系数
$$
r=\frac{\rm cov(X,Y)}{\sqrt{{\rm Var}(X)}\sqrt{{\rm Var}(Y)}}
$$
MSE loss function

$Y=f_{\theta_0,\theta_1,...,\theta_n}(X_1,X_2,...,X_n)=f_{\theta}(\mathbf{X})$
$$
\operatorname{Loss}(\theta)=\frac{1}{n} \sum_{i=1}^{n}\left(f_{\theta}\left(\mathbf{x}_{\mathbf{i}}\right)-y_{i}\right)^{2}
$$
加上惩罚项
$$
\begin{align}
\mathcal{L}(\theta)&=\frac{1}{n} \sum_{i=1}^{n}\left(f_{\theta}\left(\mathbf{x}_{\mathbf{i}}\right)-y_{i}\right)^{2}+\frac{1}{n}\lambda\sum\limits_{i=1}^n\theta_i^2\\
\mathcal{L}(\theta)&=\frac{1}{n} \sum_{i=1}^{n}\left(f_{\theta}\left(\mathbf{x}_{\mathbf{i}}\right)-y_{i}\right)^{2}+\frac{1}{n}\lambda\sum\limits_{i=1}^n|\theta_i|
\end{align}
$$
$\lambda$

- trades off error and model complexity
- user-supplied hyper-parameter

对于 ridge 回归 $\mathbf{\hat w} =(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^Ty$

---

$$
\begin{align}
\operatorname{MSE} &= E_\mathcal{D}[\hat y − f(\mathbf{x})]^2\\
&= E_\mathcal{D}[\hat y −  E_\mathcal{D}[\hat y]]^2 + ( E_\mathcal{D}[\hat y − f(\mathbf{x})])^2\\
&={\rm (variance) + (bias)}^2
\end{align}
$$

----

# 二、分类数据



we can define a new linear regression model in which predicts not the value of $Y$, but what are called the log odds of $Y$:  
$$
\log \text{odds } Y = O d d s=b_{0}+b_{1} X_{1}+\cdots+b_{n} X_{n}
$$

- Once Odds are estimated, they can be used to calculate the probability of $Y$ :

$$
\operatorname{Pr}(Y=1)=\frac{e^{O d d s}}{\left(1+e^{O d d s}\right)}
$$

