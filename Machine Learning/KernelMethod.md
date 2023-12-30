[toc]

# SVC

<img src="https://s2.loli.net/2022/08/07/XmK6xMwakvGJjtL.png" alt="image-20220807110127810" style="zoom:80%;" />

支持向量分类器的几何。圈住的数据点是支持向量，它是最接近决策边界的训练示例。支持向量机找到使边距 $m/||w||$ 最大化的决策边界。

由于我们可以自由地重新缩放 $t$、$||\bf{w}||$ 和 $m$，因此习惯上选择 $m = 1$。然后，最大化边缘对应于最小化 $||\bf{w}||$，或者更方便地说，$\frac{1}{2}||\bf{w}||$，当然前提是没有一个训练点落在边际内。

这导致了一个二次的、有约束的优化问题
$$
w^*,t^∗ = \arg \min\limits_{w,t}\frac{1}{2}||\mathbf{w}||^2 \quad\text{subject to }\; y_i(\mathbf{w\cdot x_i} − t) \ge 1,\; 1 \le i \le n
$$
使用拉格朗日乘子法，可以推导出这个问题的对偶形式。

---

## 对偶形式

为每个训练示例添加乘子 $\alpha_i$ 的约束，得到拉格朗日函数
$$
\begin{aligned}
\Lambda\left(\mathbf{w}, t, \alpha_{1}, \ldots, \alpha_{n}\right) &=\frac{1}{2}\|\mathbf{w}\|^{2}-\sum_{i=1}^{n} \alpha_{i}\left(y_{i}\left(\mathbf{w} \cdot \mathbf{x}_{i}-t\right)-1\right) \\
&=\frac{1}{2}\|\mathbf{w}\|^{2}-\sum_{i=1}^{n} \alpha_{i} y_{i}\left(\mathbf{w} \cdot \mathbf{x}_{i}\right)+\sum_{i=1}^{n} \alpha_{i} y_{i} t+\sum_{i=1}^{n} \alpha_{i} \\
&=\frac{1}{2} \mathbf{w} \cdot \mathbf{w}-\mathbf{w} \cdot\left(\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}\right)+t\left(\sum_{i=1}^{n} \alpha_{i} y_{i}\right)+\sum_{i=1}^{n} \alpha_{i}
\end{aligned}
$$
- 通过取拉格朗日函数相对于 $t$ 的偏导数并将其设置为 $0$，我们得到 $\sum_{i=1}^{n} \alpha_{i} y_{i}=0$
- 类似地，通过取拉格朗日函数相对于 $\mathbf{w}$ 的偏导数并设置为 $0$，我们得到 $\mathbf{w}=\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}$，与我们为感知器派生的表达式相同。
- 对于感知器，实例权重 $\alpha_i$ 是非负整数，表示示例在训练中被错误分类的次数。对于支持向量机，$\alpha_i$ 是非负实数。
- 它们的共同点是，如果特定 instance $x_i$ 的 $\alpha_i = 0$，则可以从训练集中删除该 instance，而不会影响学习的决策边界。在支持向量机的情况下，这意味着 $\alpha_i>0$ 仅对支持向量：最接近决策边界的训练示例。 

这些表达式允许我们消除 $w$ 和 $t$ 并导致对偶拉格朗日量
$$
\begin{aligned}
\Lambda\left(\alpha_{1}, \ldots, \alpha_{n}\right) &=-\frac{1}{2}\left(\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}\right) \cdot\left(\sum_{i=1}^{n} \alpha_{i} y_{i} \mathbf{x}_{i}\right)+\sum_{i=1}^{n} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} \mathbf{x}_{i} \cdot \mathbf{x}_{j}+\sum_{i=1}^{n} \alpha_{i}
\end{aligned}
$$
支持向量机的对偶优化问题是在正约束和一个相等约束下，最大化对偶拉格朗日量：
$$
\begin{aligned}
\alpha_{1}^{*}, \ldots, \alpha_{n}^{*}=& \underset{\alpha_{1}, \ldots, \alpha_{n}}{\arg \max }-\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_{i} \alpha_{j} y_{i} y_{j} \mathbf{x}_{i} \cdot \mathbf{x}_{j}+\sum_{i=1}^{n} \alpha_{i} \\
& \text { subject to } \alpha_{i} \geq 0,1 \leq i \leq n \text { and } \sum_{i=1}^{n} \alpha_{i} y_{i}=0
\end{aligned}
$$

---

## Noise

到目前为止，我们假设数据是可分离的（在原始空间或变换空间中） 

错误分类的示例可能会破坏可分离性假设

![image-20220808091730978](https://s2.loli.net/2022/08/08/C2ImKnl6rtXEaYs.png)

- 引入 “slack” 变量 $\xi_i$ 以允许对实例进行错误分类
- 此 “soft margin ”允许 SVM 处理嘈杂的数据

---

## Allowing margin errors

这个想法是引入松弛变量 $\xi_i$，每个示例中的一个，允许其中一些在边际内，甚至在决策边界的错误一侧。

- 我们将这些边际误差称为 margin errors

因此，我们将约束更改为 $w\cdot x_i - t \ge 1 - \xi_i$ 并将所有可宽弛变量的总和添加到要最小化的目标函数中，

- 有以下 soft margin optimization problem

$$
\begin{align}
\mathbf{w}^*,t^*,\xi_i^* &= \arg\min\limits_{\mathbf{w},t,\xi_i}\frac{1}{2}||w||^2+C\sum^n\limits_{i=1}\xi_i\\
&\quad\quad\text{subject to }\; y_i(\mathbf{w} \cdot \mathbf{x}_i - t) ≥ 1-\xi_i\;\text{ and }\; \xi_i\ge 0, 1 \le i \le n  
\end{align}
$$

- $C$ 是用户定义的参数，用于 margin maximization 与 slack variable minimization
  - $C$ 的高值意味着 margin errors 会产生 high penalty，而低值允许更多的 margin errors（可能包括错误分类）以实现较大的利润率。 
- 如果我们允许更多的 margin errors，我们需要更少（sparse）的支持向量，因此 $C$ 在某种程度上控制 SVM 的“复杂性”，因此通常被称为复杂性参数。 
- 必须设置 $C$，例如，通过交叉验证

---













