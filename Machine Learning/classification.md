[toc]

# 一、logistic

In the case of a two-class classification problem, if we model the probability $P(Y = 1)$ of an instance x being a positive example like this:  
$$
p=P(Y=1 \mid \mathbf{x})=\frac{1}{1+e^{-\mathbf{w}^{T} \mathbf{x}}}
$$
the alternative $(1-P(Y=1))$
$$
\ln \frac{P(Y=1 \mid \mathbf{x})}{1-P(Y=1 \mid \mathbf{x})}=\mathbf{w}^{T} \mathbf{x}
$$
The quantity on the l.h.s. is called the **logit** and we are defining a linear model for the logit.

---

Unlike linear regression, no analytical maximum likelihood (ML) solution to find weights w.

An iterative gradient ascent method can be used to maximize log likelihood.  

The (conditional) log likelihood is:
$$
\sum_{i=1}^{N} y^{(i)} \log P\left(1 \mid \mathbf{x}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-P\left(1 \mid \mathbf{x}^{(i)}\right)\right)
$$

---

二、感知机







