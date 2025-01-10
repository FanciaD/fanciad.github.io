---
title: 密码学前沿问题2024 笔记
date: '2024-11-30 20:27:49'
updated: '2024-11-30 20:27:49'
tags: Fancia
permalink: OhfkaFraedda/
description: 密码学前沿问题2024
mathjax: true
---

### Lattices

#### Lattices

给 $k$ 个 **线性无关** 的 $n$ 维向量 $b_1,\cdots, b_k$，写成矩阵就是 $n\times k$ 的 $B$。定义它们生成的 lattice 为：

$$L(B)=\{\sum c_ib_i|c_i\in \Z\}$$

或者写成矩阵就是 $\{Bc|c\in \Z^k\}$。记 $n$ 是维数，$k$ 是格的秩，常见情况是 $k=n$，简称满秩。

为什么要线性无关？这样有很多简便：

1. 如果要判定一个点是否属于 $L(B)$，那只需要解方程 $Bc=x$ 然后直接看唯一解。根据 Cramer's Rule，解方程 $Ax=b$ 得到的解分子分母都是一个行列式，那么大小不是指数级的。
2. 这样定义的 Lattice 是离散的：$c\in \Z^k$ 是离散的，做完线性变换还是。如果不满足这个条件，就不一定离散了。比如 $(1),(\pi)$。这样就没法定义最短的非零向量：不存在最小值。

#### Some operations on Lattices

考虑判定两个 lattice 是否相等。显然判定 $L(B_1)?\in L(B_2)$ 只需要把每个基拿出来试一试。然后做两轮。还有一个结论：

根据刚才的做法，$L(B_1)=L(B_2)$ 时，存在一个基的整系数线性变换使得 $B_1\to B_2$。即 $\exists M\in \Z^{k\times k},B_2=B_1M$。同时交换 $1,2$ 也一样。此时 $M,M^{-1}$ 都是整系数的。因为两个 $\det$ 都得是整数，即刻得到必要条件是 $\det(M)\in \pm 1$。根据伴随矩阵的知识，$M^{-1}=M^*/det(M)$，而伴随的定义是一堆余子式，其为整数，因此这也是充分条件。

##### Dual Lattice

提前到这里。为了简便我们只考虑满秩情况。

对于一个 $L$，定义它的 Dual 是 $L^*=\{x|\forall v\in L,\langle v,x\rangle\in \Z\}$。

怎么描述一个 Dual Lattice？考虑找 $L$ 的一组基 $b_1,\cdots,b_n$，那么 $(\langle x,v_i\rangle)_{1,\cdots,n}\in \Z^n$ 可以唯一确定 $x$，反过来也一样（满秩情况下）。

因此满秩情况下，Dual 的一组基就是 $(\langle x,v_i\rangle)_{1,\cdots,n}$ 取 $e_1,\cdots,e_n$。

也可以给它写成矩阵形式，**注意到内积是个转置**，所以这里写出来是 $B$ 先转置再求逆（当然反过来做一样的）。也可以简写：$B^{-T}$。

##### Hermite Normal Form

如果我们能把 Lattice 变成一个上三角或者下三角，那它会比较好看，~~虽然这不能帮我们解决 SVP~~

Naive 的想法是直接整系数辗转相减，但这样有严重的问题：每一轮消完所有数大小翻倍，最后不能 poly 地存下所有数。

现在的问题是，我们每次消的时候下面的东西不能变小，所以最后很大。因此我们需要一些观察：

考虑整系数情况，注意到 $A*adj(A)=\det(A)I$，所以说 $\det(A)I$ 一定在 $L$ 里面（整系数下伴随也是整系数）。

那我们就可以每一步对 $\det(A)$ 取模，这样就不会太大了。如果非整系数的话，先乘一个系数再最后除回去即可。

##### Lattice Sum

给两个 Lattice，计算 $\{x_1+x_2|x_i\in L_i\}$。

先找一组极大的基，然后一个一个加进来。Naive 的加法是消元形式：进来两个先在最高位比较一下：$(a,\cdots),(b,\cdots)$ 能够合出一种组出 $\gcd(a,b)$ 的方式，和一组极小的消掉第一维的方式，后者拿下去继续做。

然后整体取模一下。

##### Lattice Intersection

考虑满秩情况。注意到 $(L_1\cap L_2)^*=L_1^*+L_2^*$：Dual 是拿限制定义的，交起来就是限制并起来。

#### SVP and CVP problems

**在一般情况下，我们默认使用 L2 norm**

对于一个 lattice，依次定义：

1. 1st SVP 是最短的非零向量。
2. 2nd SVP 是**与上一个线性无关**的最短非零向量。
3. 3rd SVP 是与上两个并起来线性无关的最短向量。

为什么要线性无关？我们只想定义 $k$ 个东西。

也可以直接定义长度：

1. $\lambda_i(L(B))$ 定义为最小的 $r$，使得 $L(B)\cap B_n(r)$ 中向量（张成空间）的秩不小于 $i$。这就是第 $i$ 个线性无关最短向量的长度。

另一个问题是，给定一个点 $t$，求出它和 lattice 的距离 $dist(t,L(B))$。我们一般不考虑和它最近的线性无关向量，因为这比较神必。

##### SVP and GapSVP

不近似的 SVP：找到一个非零向量取到 $\lambda_1(L(B))$。

近似的 SVP：找到一个 $\gamma$ 倍长度以内的。

还有判定性版本：

$\gamma$-GapSVP：这是一个 Promise Problem。给定 $d,\gamma$：

1. YES instance: $\lambda_1\leq d$
2. NO instance: $\lambda_1>\gamma d$
3. 众所周知，其余情况下可以输出任意东西。（可以通过 cutoff 避免不停机）

##### CVP and GapCVP

把所有 $\lambda_1($ 换成 $dist(t,$。

#### Reduction of different settings

##### CVP Search vs Decision

如果能做搜索版本，那显然能做对应 $\gamma$ 的判定版本。显然判定等价于求出最短距离（二分），如果 $\gamma$ 那就是一个那么多倍的近似。

考虑 $\gamma=1$ 的情况。我们希望找到 CVP。考虑把 $b_1$ 乘 $2$，如果 CVP 变大了，就说明最优解里面 $c_1$ 是奇数，此时可以给 $t$ 加上 $b_1$，然后继续算下一位。把每个数的每一位算出来即可。

可以做 $\gamma$，但是有 poly(n) 步，每步会乘一次 $\gamma$。

##### SVP Search vs Decision

这个就比较难了，因为我们不能在 GapSVP 里面加一个 $b_1$。我们需要一些工具：

1. 对于一个 Lattice，我们可以 bound SVP 每一维的上界（proof: later）
2. 我们可以求两个 lattice 的交（maybe later）

我们首先进行如下操作：选一个非常大的 $B$，然后把每一维乘上 $B^{100n}+B^i$。这样因为前面系数的存在，SVP 还是原来的组合。然后我们就可以通过 $B^i$ 部分把 SVP 的每一维的绝对值搞出来。现在离求出 SVP 只差确定符号。

然后我们使用 Thm2. 假设现在得到了 $|x_1|,|x_2|,\cdots$，那我们限制一个 $x_1|X_2|=x_2|X_1|$ 就可以判定前两个符号是不是相同。这里限制的意思是求出 Lattice Intersection，然后跑 SVP。然后继续跑下去即可。 Intersection 可以看前面。

##### GapSVP to GapCVP

我们即将看到这里唯一一个保留了 Gap 的规约。

对于一个 GapSVP，考虑如下 CVP 构造：$t=b_1$，然后把 $b_1$ 乘二。这样所有的 dist 对应 $c_1$ 为奇数的向量，那我们有概率直接判断对。

然后考虑每一维做一次。注意到 SVP 不可能全部系数都是偶数，因此总有一个对。然后 OR 起来就行了。

#### Minkowski's Theorem

定义一个 lattice 的 fundamental parallelepiped，是所有基张成的高维多面体（二维的平行四边形，三维的六面体）

定义 lattice 的 $\det$ 是上面这个东西的体积。根据经典结论这个等于 $\sqrt{\det{B^TB}}$，或者方阵的时候直接 $|\det B|$

这东西大概就代表了 Lattice 上，每个点旁边有多少空间。然后有如下结论：

Minkowski's Theorem: $\lambda_1(L)\leq O(\sqrt n)*(\det(B))^{1/n}$

证明：考虑每个点画一个半径 $\frac12\lambda_1$ 的高维球，那么根据 SVP 的定义没有球相交。因此每个球的体积不超过多面体体积。直接解得

$$
\det(B)\geq \frac 1{\sqrt{n\pi}}(\lambda_1(L)/2)^n(2\pi e/n)^{n/2}
$$

开根，只保留主项就得到了上面的界。

这个界确实是渐进紧的，可以参考作业。

#### The LLL algorithm

##### Gram-Schmidt Orthogonalization

依次算 $\tilde b_i$，每一步用之前得到的正交向量来处理掉 $b_i$ 和它们投影的部分，减去之后得到剩下的 $\tilde b_i$。

我们需要证明这个过程中不会出现特别大的数。考虑每一步的过程，我们给 $b_i$ 减去了之前 $\tilde b_i$ 的线性组合，也是一个 $b_i$ 的线性组合。考虑减了多少 $b_i$。限制是和前面正交，那可以写出 $n$ 个方程：$b_i-\sum x_jb_j$ 和 $b_j$ 正交只需要把 $b_j$ 内积上去得到方程，那么是一个 $b$ 的内积矩阵乘 $x$ 等于 $b_i$ 和前面内积的结果。

然后是解线性方程组。注意到 Cramer's rule 保证了解线性方程组的输出数字大小是 poly 的，因此每个 $\tilde b_i$ 是 poly 位的。

##### delta-reduced basis

称一组 $b_1,\cdots,b_n$ 是好的，如果其满足如下条件：

首先进行分解，得到 $b_i=\tilde b_i+\sum_{j<i}\mu_{i,j}\tilde b_j$。然后需要满足：

1. $|\mu_{i,j}|\leq 1/2$
2. $|\tilde b_{i+1}+\mu_{i+1,i}\tilde b_i|^2\geq \delta |\tilde b_i|^2$

这里 $\delta\in (1/4,1)$。

首先，如果满足这个条件，那么 $|\tilde b_{i+1}|^2\geq (\delta-1/4)|\tilde b_i|^2$，从而 $|\tilde b_i|$ 只比 $|\tilde b_j|$ 大 $(1/\sqrt{\delta-1/4})^{i-j}$ 倍以内。

考虑证明这个给了一个不错的下界。但我们不能用之前的上界来 bound 下界。注意到 SVP 不小于 $\min |\tilde b_i|$（找到最后一个非零系数，考虑 $\langle b=\sum_{j\leq i} c_jb_j,\tilde b_i\rangle=c_i|b_i|^2$），那么复用上面的界，我们不超过 $(1/\sqrt{\delta-1/4})^{n-1}$ 倍。取 $\delta\to 1$ 就是 $(2/\sqrt 3)^n$。

##### LLL algorithm

首先我们考虑怎么满足第一个条件，那考虑每个 $b_i$，再从大到小考虑 $j$。如果我们直接抄 Gram-Schmidt， $b_i -=\tilde  b_j * \langle b_i,\tilde b_j\rangle/\langle\tilde  b_j,\tilde b_j\rangle$，那就消完了。但这里首先 $\tilde b_j$ 不是整系数，然后后面乘的也不是整数。考虑：

1. 用 $b_j$ 代替 $\tilde b_j$，因为在 $\tilde b_j$ 分量上一样。
2. 把系数取整，这样 $\mu_{i,j}$ 就在 $[-1/2,1/2]$ 之间。

然后这样做一轮，就能满足第一个性质。可以证明数不会变得很大。

但这样不一定满足第二个限制，考虑如下调整法：如果不满足限制，就交换 $b_i,b_{i+1}$（保留之前消元的结果）。考虑这对 $\tilde b$ 的变化。之前是：

$$
\cdots, \tilde b_i,\tilde b_{i+1},\cdots
$$

现在先考虑 $b_{i+1}$，那么它可能少减去了 $\tilde b_i$ 的项，因此变为：

$$
\cdots,\tilde b_{i+1}+\mu_{i+1,i}\tilde b_i, \tilde b_i-sth, \cdots
$$

但根据性质，如果不满足限制，那这样第一个向量长度就变小了至少 $1/\sqrt\delta$ 倍。但另一方面，正交基长度乘起来等于行列式，所以第二个对应变大了。

那我们可以写出一个势能：$\prod |\tilde b_i|^{n-i+1}$。每一步它减少 $1/\sqrt \delta$ 倍，又注意到上述线性变换不改变 G-S 分解的结果。初始是 $V^{poly}$ 大小，因此只需要 $poly\log V\log \delta^{-1}$ 级别的步数。

#### SIS and LWE - introduction

以下部分全部在 $Z_q$ 做。

##### Short Integer Solutions

给一个 $n\times m$ 的随机矩阵 $A$，求 $Ax=0$ 的一个非平凡非零解，满足 $|x|\leq b$。

常见操作：$m\geq 2n\log q$

考虑把 $Ax=0\pmod q$ 的解写出来。考虑：

$$
\begin{bmatrix}
A_1 & A_2
\end{bmatrix} x = 0
$$

这里 $A_1$ 是前面的方阵。因为随机矩阵，我们可以找 $n$ 列满秩的，然后交换。这样就可以写成

$$
\begin{bmatrix}
I & H
\end{bmatrix} x = 0
$$

此时基就非常显然了：首先有 $m-n$ 个 $Ax=0$ 的解：$H$ 的某一列拼上对应位置的 $1$。然后模 $q$ 意义下容易再发现每个 $qe_i$ 都是解。但前 $n$ 个加进去以后，后面的就可以通过 $m-n$ 个组合出来了。得到如下基：

$$
\begin{bmatrix}
qI_n & -H\\
0 & I_{m-n}
\end{bmatrix}
$$

然后 SIS 就是这里的一个 GapSVP。

使用 Minkowski's Thm，得到 $\det=q^n$，那么 SIS 应该是 $q^{n/(2n\log q)}O(\sqrt m)=O(\sqrt m)$。因此

$$
b\geq O(\sqrt m)
$$

事实上有一个更直接的证明：考虑 $x_i\in\{0,1\}$，由 $2^{2n\log q}\geq q^n$ 可知存在像集重叠，那么两个解减一下就得到一个 $x_i\in\{-1,0,1\}$ 的 SIS，它自然满足 $b\leq \sqrt m$。

##### Learning With Errors

有一个 $s\in Z_q^n$，现在给出 $y=A^Ts+e$，其中 $A\in Z^{n\times m}$，$e$ 是某种随机分布，

考虑这个怎么用 Lattice 描述。我们需要的 Lattice 是 $\{A^Ts\pmod q\}$，然后做 CVP。类似写一下，

$$
A^Ts\equiv \begin{bmatrix}I_n\\ H\end{bmatrix} s
$$

然后需要 $\pmod q$，可以补 $m-n$ 个在后面，前面的 $\pmod q$ 可以通过加 $q$ 次上面的基然后减掉后面解决。

所以基是

$$
\begin{bmatrix}
I_n & 0\\
H & qI_{m-n}
\end{bmatrix}
$$

一般 $m\geq 2n\log q$，

仔细看看这个，如果转置得到

$$
\begin{bmatrix}
I_n & H\\
0 & qI_{m-n}
\end{bmatrix}
$$

那它和 SIS 的那个乘起来，正好 $H$ 项抵消，从而得到 $qI$。那么就有趣味性质：这两个问题的 Lattice 差常数倍是互相 Dual 的。这里 SIS 是一个 SVP，LWE 是一个 CVP。

#### Regev's reduction from LWE to GapSVP

给一个 LWE 的算法，我们可以得到一个 Quantum 的算 GapSVP 的算法。

##### Gaussian Sampling in Lattices

在 Lattice 里面采样，要求点 $x\in L$ 的系数正比于 $\exp(-\|x\|^2/r^2)$。

如果 $r$ 很小，这个问题就和 SVP 或者对应的 GapSVP 一样难。

如果 $r$ 非常大（指数级），那是能做到 Statistically Close 的，但具体做法还是非常复杂（先用 LLL，后面忘了）

##### Bounded Distance Decoding

CVP，但保证 $d\leq \lambda_1/r$，然后用 $r$ 作为参数。

$r=2^n$ 是简单的（Babai85?），$r$ 很小是猜想困难的，但还是非常难证。

这里的做法是，假设有一个 LWE 的算法，我们从一个 $L$ 的 Sampling 开始，可以得到一个 $L^*$ 的 BDD，然后得到 $L$ 的一个范围更小的 Sampling，一直做就能做原问题。

##### Sampling to BDD

BDD 的目标是，给一个 $v+e$，其中 $v\in L^*$ 然后 $e$ 很小，把它们分离出来。

直观考虑，我们搞一个 $w\in L$，然后 $\langle w,v+e\rangle =\langle w,v\rangle+\langle w,e\rangle$，后者是一个无论如何很小的数，而前者可以非常大（采样的 $r$ 很大）。如果我们取模一个比后者大的数，这就像是一个 LWE，但限制输入只能是 $L^*$ 里面的东西。因此我们把 $v$ 写成 $B^{-1}x$，然后把 $B^{-1}$ 放到 $w$ 那边去，模一个比 $\langle w,e\rangle$ 大的东西就变成了 $x$ 的 LWE。

但还需要验证两点：LWE 的 $A,e$ 是合理的分布。

这里 $e$ 部分原来是 $\langle w,e\rangle$，但那个 $e$ 是定值，如果我们假设 $w$ 是纯正 Gaussion 这就是完全正确的，一般情况也可以说明是正确的。

$A$ 部分是矩阵乘一个 $w$，然后取模。根据上面的分析不取模是 Gaussian，那只要范围够大，取模后根据对称性还是非常均匀的。显然 $r$ 太小了这不对，因此最后只能到某个 bound 的 GapSVP。

##### BDD to Sampling

考虑一个简单情况：注意到 LWE 是一个 BDD，此时目标就是对所有 SIS 的解做采样（而不是找足够小的）。

考虑先制备 $\sum |s>$，然后制备 $\sum \sqrt{pr(e)}|e>$，然后就有

$$
\sum |s> \sum \sqrt{pr(e)}|s^TA+e>
$$
这是个 LWE 形式。考虑给右边上一个 QFT，按照惯例我们得到

$$
\sum |y> w^{(s^TAy+ey)\pmod q} \sum \sqrt{pr(e)}|s>
$$

根据经典想法，如果 $y$ 是一个 $Ax\equiv 0$ 的解，$s^TAy$ 那一项就是 $0$，然后就会聚合得比较好。否则，那一项会均匀分布，然后 $\sum w^i$ 是抵消的……等等现在有个 $|s>$，不能消掉。

那考虑把 $|s>$ 干掉。这里就要用到 LWE Oracle，从右边解出 $s$ 然后异或到左边，之后再 QFT，这样就没有 $|s>$ 了。最后 $e$ 可以拿来控制 Sample rate。

#### PKE from LWE

$\{0,1\}$ 问题上 PKE CPA 的简单定义：$pk,Enc_{pk}(0)\approx pk,Enc_{pk}(1)$。

Gen: Public Key: $A,y=A^Ts+e\pmod q$. Secret Key: $s$.

Enc: 随机一个 $\{0,1\}^m$ 向量，将对应行线性组合（每一行是一个 $\langle a_i,s\rangle+e_i=y_i$），然后视情况在最后的 $y$ 加上 $b*q/2$，得到 $(a,y)$。

Dec: 拿 $s$ 带入这堆东西，误差积累很小，那么可以根据最后差的接近 $0$ 还是接近 $q/2$ 判定。

##### DLWE

Decisional LWE: 这是一个 Distributional Problem，有一半概率输入是一个合法的 LWE，有一半概率是完全随机的 $A,y$，进行判定。

显然 LWE 不比 DLWE 简单。另一个方向也是可以说明的：首先如果判定成功率是 $1$，那可以直接这样做（假设 $q\in poly$）：枚举一个 $s_i$，然后带入消到更小的 LWE 问题，然后询问。注意到如果猜的不对，那么 $A$ 这部分会变成一个随机的 $y$ 加过去，那剩下的就变成完全随机了。然后这样就可以逐位确定。也存在能做 $q$ 更大的版本。

一般情况咋办？我们需要把当前的问题变成一个完全随机的 Instance，这样才能保证正确率对。

首先考虑变动 $s$，这是直接的：随一个加上去。

但我们还需要随机 $A$，此时有如下技巧：随机一个 $01$ 矩阵乘上去。

##### Leftover Hash Lemma and Security Proof

根据 DLWE，$(A,y)$ 和随机矩阵看起来（Computational）没有区别。

再看随机 $0/1$ 乘起来的部分。根据 Leftover Hash Lemma，给一个部分随机源做某些 Hashing（比如这里）后，它的 Entropy 能保存得很好，结果就是得到的东西看起来很随机。所以 Enc 也是看起来随机的。

##### LWE from SIS

假设有一个 SIS 算法，我们就能做 DLWE，然后就能 LWE。

拿 SIS 算出一个 $Ax\equiv 0$ 的小解，然后和 $y=A^Ts+e$ 内积后，第一项就没了。只要解够小，误差就不会很大，然后 $\pmod q$ 也能看出来。

Open Problem: 如果 Error 只有 $0/1$。

#### FHE from LWE

FHE 是说，除了 $0/1$ 上的 PKE 外，我们还支持在密文上运算：给两个 $Enc(x_0),Enc(x_1)$，我们可以只用 Public Key 算出一个 $Enc(x_0\ op\ x_1)$。好处就是可以做完全隐私的计算。

这个做法来自 GSW13。

##### G Gadget

一个常见的构造问题是，我们希望有一个随机误差项系数很小，但如果它乘一个 $\Z_q$ 里面的随意元素，它就特别大了。我们希望它乘的东西小一点。因此一个想法是把东西变成二进制表示：

定义 $G^{-1}(v)$ 是把 $v$ 里面的每个元素写成二进制表示（$\{0,1\}^{\log q}$），然后拼起来。

定义 $G$ 是一个线性函数，大概是 $[1,2,4,8,\cdots,2^k]$ 然后拼对角，得到 $n\times nk$ 的矩阵。显然这两是互逆操作。

Remark:

1. $G$ 上的 SIS 是简单的：二进制分解。
2. 如果 $q=2^k$，$G$ 上的 LWE 是简单的：拿 $2^{k-1}$ 还原最低位，然后拿 $2^{k-2}$ 还原第二低位，以此类推。
3. 任意情况下，$G$ 上的 LWE 还是简单的：一个一个解，考虑 $x_1+e\equiv a$ 可以去掉取模（实在不行，枚举）得到 $x_1=a+e$，然后考虑 $2x_1+e\equiv a_2$ 的话，可以根据上一个界判断乘 $2$ 后取模掉了多少，然后就可以得到 $2x_1=a_2+e_2$，然后再看 $4$，类似的可以去掉取模。到最后就知道了答案。

##### Leveled FHE

定义：我们能做深度 $d$ 的密文运算，再做就可能爆炸了。

这里取 $d=\log n,q=\exp(\log^2 n)$。考虑如下构造：

Gen: Public Key: $A,y=A^Ts+e\pmod q$. Secret Key: $s$. 这里我们把 $A,y$ 写成一个矩阵：$A$ 在上面 $y$ 在下面，要求它和 $G$ 大小一样。

Enc：给 $B$ 做随机 $0/1$ 列变换后加上 $b\cdot G$。即 $R\in \{0,1\}^{m\times m}$, $ct\leftarrow BR+bG$.

Dec：拿 $s$ 做行变换消掉 $y$ 里面的 $A^Ts$ 部分，然后剩下 $e$ 在过了 $R$ 以后还是小的，可以拿 $bG$ 判断出来。

Eval: 考虑乘法：构造：

$$
CT_1G^{-1}CT_2=(BR_1+b_1G)G^{-1}(BR_2+b_2G)\\
=B(R_1G^{-1}(BR_2+b_2G))+b_1BR_2+b_1b_2G
$$

注意到 $G^{-1}$ 是 $0/1$ 的，所以 $B$ 后面的随机系数只是乘了 $n$ 倍。那么做 $\log q/\log n$ 轮之前我们都能还原出来。

加法就直接加，然后模 $2$ 就减去一个乘法。

##### FHE BootStrapping

我们的要求是 Dec 很好算，计算深度不超过上面的 $d$。

考虑把一个密文还原到刚加密的误差状态。

我们再发一个 $Enc(sk)$。 Open Problem：证明它的安全性。

然后干这样一件事：考虑 Dec 的 Circuit，它应该输入是 sk 和 CT，然后输出 $b$。

考虑给当前 CT 再加密一次，然后和 Enc(sk) 一起，在加密的情况下跑 Dec 的 Circuit。这样我们又得到了 Enc(b)，同时这个过程计算深度回到了一个定值。

#### Lattice Trapdoor

什么是 Trapdoor? 如果没有这个东西那原问题是困难的，但有这个东西，我们就可以解决原问题。

例如，在 RSA 里面，我们的问题是 Invert $x^e\pmod n$，而我们的 Trapdoor 就是 $e^{-1}\pmod{\varphi n}$。

还有个问题，我们允许 Adversary 是 P/poly 的，那 Adversary 记住 Trapdoor 不就行了？因此实际上所有的 Trapdoor 都是这样的：我们有一组 Function，每次可以 Sample 一个 Function 和它的 Trapdoor，这样 Adversary 不可能记住所有出现的 Trapdoor。在这一定义下：

##### Trapdoor function/permutation

一个 TrapDoor Function(TDF) 是这样的：对于每个 $n$，我们有一个 Function family $\mathcal F$，然后：

1. 对于每一个 $f\in \mathcal F$，可以 Sample 它的定义域（一般定义域就是 $\{0,1\}^{f(n)}$ 所以没问题，但我们要避免奇怪情况）
2. 我们可以快速采样一个 $f$ 和它的 Trapdoor $\tau_f$。
3. $f$ 可以快速计算，给定 $\tau_f$ 可以快速 Invert 它。
4. 整个 $\mathcal F$ 构成一个 OWF family：随机一个的情况下，任何 Adv 不能 Invert 掉它。

把 Function 换成 Permutation 就是 TDP。可以发现在合理的假设下，RSA 是非常完美的 TDP：它在一个 $Z_{p-1}\times Z_{q-1}$ 的乘法群上做了 $(\times a,\times b)$ 的映射。

Open Problem：找一个不用 RSA Assumption，不用 iO 的 TDP 构造。

##### Ajtai's function from SIS

现在我们从 Lattice 开始造一个 TDF。令 $A\in \Z_q^{n\times m}$，我们定义 Ajtai's function

$$f_{A,q}: [-B,B]^m\mapsto \Z_q^n, f_{A,q}(x)=Ax\pmod q$$

可以发现 Invert 这个就是找一个 Short Solution。由 Inhomogeneous SIS 的 Hardness（这可以从 SIS Hardness 推）可知上面是 OWF Family。

那这个 Function 有没有 Trapdoor？换言之，有没有让 SIS 变简单的 advice？根据历史经验，一个 Short Basis 可以让 Lattice 问题变简单。

##### Basis Trapdoors

直观的说，任何一个 Lattice 问题的 Basis Trapdoor 就是它的一组 Short Basis。

这里是一个 SIS，那么它的 Lattice 是 $L_q^\perp(A)=\{x|Ax\equiv 0\pmod q\}$。因此一个 Basis Trapdoor 可以写成 Basis 列向量的矩阵：一个 $m\times m$ 的矩阵 $T$，满足 $AT\equiv 0\pmod q$。为了刻画长度，定义 $\|T\|_2$ 表示它每一列 $L_2$ Norm 的最大值，我们希望这个东西不大。

根据经验，有这样一个 $T$ 我们就能 Invert 上面的东西。具体来说，考虑对于一个 $y$，先随便找一个 $Ax=y$ 的解 $x_0$。然后我们把 $x_0$ 表示成 $T$ 的线性组合：$$x_0=\sum c_it_i$$

接下来，考虑用 $T$ 去简化 $x_0$，那这个事非常直接：通过减一些整系数倍，把每个 $c_i$ 的系数调整到 $[-1/2,1/2]$ 之间。此时我们得到的解的模长不超过 $\sum \|\frac12 t_i\|=m/2|T|$。

###### Existence

首先，什么情况下有这样一组解？换言之，$\lambda_n(L_q^\perp)$ 有多大？我们拿出之前的一堆定理：

Banaszczyk 93：$1\leq \lambda_1(L)\lambda_n(L^*)\leq n$。

上次的结论：$L_q^\perp$ 的 Dual 是 $L_q/q$，其中 $L_q$ 是 $\{Ax\pmod q\}$。

Homework2 P2: $L_q$ 的 $\lambda_1$ 高概率至少是 $O(q)$：考虑 $l_{\infty}\leq q/4$ 的情况，此时如果我们全部随机，有 $q^n$ 组 $x$，对于每个，我们随机 $A$ 的时候，$Ax$ 差不多是在 $\Z_q^m$ 上独立随机的（如果权值互质那就是，不互质的情况下，它是在 $Z_q$ 的一个子环上随机，此时可以类似证明），那么每一维都在 $[-q/4,q/4]$ 里面的概率差不多是 $2^{-m}$，因此 $m=\Omega(n\log q)$ 的时候找到的概率就很小。

三个放一起即刻得到高概率 $\lambda_n(L_q^\perp)\leq O(n)$。实际上我们造的时候可能会大点，比如 poly n，但这不是大问题。

但怎么找一个 T？这还是很难，因此我们考虑另一个方向：

##### G Trapdoors

定义一个 $G\in \Z_q^{n\times nk}$ 长这样：对角线上 $n$ 个 $1\times k$ 的块，每一块形如 $1,2,4,8,\cdots,2^{k-1}$。

好消息：$G$ 的 Trapdoor 是简单的：对每一块造 $k$ 列，前 $k-1$ 列是显然的 $0,0,2,-1,0,\cdots$，最后一列放一个 $q$ 的二进制分解。

或者我们都不需要用 $G$ 的 Trapdoor 来解 SIS：直接对 $Gx=y$ 里面的 $y$ 做二进制分解就是 $x$。

我们定义一个矩阵 $M\in [-c,c]^{m\times nk}$ 是一个 G-Trapdoor，如果 $AM\equiv G$。首先它很有用：如果我们得到了 $Gx\equiv y$，那么 $A(Mx)\equiv y$，所以 $Mx$ 就是答案。这说明我们希望 $M$ 尽量小，这里实际上 $\|M\|_{\infty}=1$。

然后我们的问题变为，如何随机一个 $A$ 使得我们能造出 $AM=G$。那有一些直接的构造：

$$
A=\begin{bmatrix}A_0& A_0R+G\end{bmatrix},M=\begin{bmatrix}-R\\ I\end{bmatrix}
$$

显然满足 $AM=G$。我们只需要左边看起来很随机，也就是说 $[A_0\ A_0R]$ 看起来很随机。根据 Leftover Hash Lemma，只要 $m>O(n\log q)$，$R$ 取 $\{0,1\}$ 就是对的。

我们也可以把这个变回 Basis Trapdoor：注意到

$$
\begin{bmatrix}A_0&A_0R+G\end{bmatrix}\begin{bmatrix}I&-R\\&I\end{bmatrix}\equiv \begin{bmatrix}A_0&G\end{bmatrix}\\
\begin{bmatrix}A_0&G\end{bmatrix}\begin{bmatrix}I\\-Bin(A_0)&T_G\end{bmatrix}\equiv 0\\
$$
这里 $Bin$ 表示一个矩阵的二进制分解，根据定义这显然对。那我们只需要把两个 $\{-1,0,1\}$ 的矩阵乘起来就是对的，这样得到的 $T$ 满足 $L_{\infty}\leq m,L_2\leq m^{3/2}$。

这下就搞定了。

注意，“我们能 Sample 一个 Lattice 和它的 Trapdoor” 这件事还可以拿来证一些 Hardness，我们在 Signature 再看。

#### Signatures

一个 Signature 包含如下东西：

1. $Gen\to pk,sk$。
2. $sk$ 可以拿来 Sign：$Sign(m,sk)\to \sigma_m$。
3. $pk$ 可以拿来 Verify：$Verify(m,pk,\sigma)\to\{0,1\}$，表示对不对。

Correctness：正常操作都得到 $1$。

Security：实际上有[很多种](https://crypto.stackexchange.com/questions/44188/what-do-the-signature-security-abbreviations-like-euf-cma-mean)。最严格的是这样：Adv 可以任意的调用 $Sign(m)$，他的目标是构造任意一组合理且没有被调用过的 $(m^*,\sigma^*)$。

直观上看，我们的目的是搞一个无关人员不能伪造的东西。那考虑直接拿出 Lattice Trapdoor，Sign 就设成 Short Solution：pk 是 $A$，sk 是 Trapdoor；把 Message 变成 $y$，然后 Sign 是一个 Short 的 $x$，使得 $Ax\equiv y$……这对吗？

上面那个模型实际上很强。考虑这样一个攻击：询问得到 $m_1,m_2$ 的 Sign，那减一下得到的就是 $m_1-m_2$ 的 Sign。这就不行了。为了解决问题，我们一般假设一个 Random Oracle Model，然后把 $m$ 换成 $H(m)$，这样就不能随便线性组合了。

但还有个问题，现在的操作是，每次先解出 $Gx=H(m)$，然后输出 $Mx$。但第一步是所有人都可以直接做的，如果多来几组数据那随便哪个人都可以把 $M$ 给逆出来……

因此我们还需要做一些随机扰动。具体来说，定义 Discrete Lattice Gaussian 是：

$$
f_{L,r}(x)\sim\begin{cases}\exp(-\pi|x|^2/r^2), &x\in L\\0, &o.w.\end{cases}
$$

类似的，我们可以定义一个中心 $c$，然后把 $\exp$ 里面换成 $x-c$（但是 $x\in L$ 的条件不换）。然后这样做：上一步找到一个 $y$ 之后，按照 $f_{L,center=y}$ 随一个东西去减一下，然后输出。这样的话，我们的输出分布差不多是这样：

$$
\begin{cases}\exp(-\pi|x|^2/r^2), &Ax\equiv H(m)\\0, &o.w.\end{cases}
$$

为什么这样就行了？此时我们有另一种采样 $(H(m),x)$ 的方式：先采样 $x$（在 Gaussian 足够大的情况下，不同 $y$ 的 Normalization Factor 是差不多的，所以 $x\to rand, f\to ...$ 的情况下整个分布就像所有整点上的 Gaussian），然后直接 $H(m)=Ax$。

然后如果有一个 Adv，那我们可以自己和 Adv 交互，然后 Adv 最后就输出了一个新的 Sample。因此我们可以啥都不要直接解 SIS，这就规约了。

然后怎么采样上面那个东西？如果是 $\Z^n$，那大家都懂：系数可以拆成每一维独立，所以可以独立采样。那如果我们拿到的 Trapdoor $T$ 是一组正交基，那这也行。

但显然 $T$ 不会是一组正交基。那我们先做一个 G-S 分解：得到一组正交的 $r_i$，然后 $t_i=c_1r_1+c_2r_2+\cdots+c_ir_i$。此时可以发现， $r_n$ 前面的系数只和 $t_n$ 的系数有关，那我们先采样 $r_n$ 这一维，此时 Condition 这个 $r_{n-1}$ 也固定了，然后不断倒着采样回去就行了。

#### Simulation Paradigm

上面这个证明的启示：有些算法可以看成“给一个 A，我们就能做 B”。那如果我们能通过某种方式快速模拟 $A$，我们就能直接做 $B$。上面的例子里面，如果 Adv 可以和 Signature 交互，我们就能伪造新的东西。但我们的构造使得我们可以直接采样得到 Signature，那我们就能直接干最后一步的事情。下面还有一些例子：

1. 给定 $A,T_A,B$，其中 $A,B$ 随机，此时解 $A+B$ 的问题是困难的。

证明：采样 $A,T_A$，随机 $C$，让 $B=C-A$。因为 $A$ 是 $\approx_s$ 随机的，所以 $(A,C-A,C)\approx_s$ $A,C$ 都随机的情况，这个等于 $A,B$ 都随机的情况。

2. 给 $A,B,T_A,T_B$，其中 $A,B$ 随机，解 $A+B$ 还是困难的。

证明：上一手 Bonsai Technique，我们可以构造 $A|R,R|B$ 的 Trapdoor，然后加起来。

#### Identity-based Encryption

想象你是一个邮件服务器.jpg

我们的目标是：每个人有一个 id，有自己的 sk，可以解密自己的信息。可以通过 id 给其它人发信息（通过公共的 pk）。有两点特别：

1. 只有一个公共的 pk，但我们可以造出很多下级 sk。
2. sk 的构造和加密是无关的，可以先发信息再构造 sk。（当然，你也不能提前随机一个 sk 然后存在那）

具体来说：

1. Gen -> mpk,msk
2. Get_sk(id,msk) -> sk_id
3. Enc(m,id,mpk) -> ct
4. Dec(ct,sk_id) -> m

Correctness：这堆东西正常运转。

Security：有 poly 个人在一起，它们不能解密别的人的 message。

如果我们不仅要求解密 message，还要求 recover key 会怎么样？那就相当于有 poly 个人在一起，它们不能通过自己的 $sk_{id}$ 得到新的 $sk_{id'}$。可以发现这差不多是个 Signature。那我们复用一下之前的记号：

1. $mpk,msk$ <-> $A,T$
2. Get_sk(id,msk): 把 $id$ Hash 到一个 $H(id)$，然后按照上面的方式造 $Ax=H(id)$ 的 Short Solution。

那怎么加密解密？我们现在有 $Ax=H(id)$ 的一个解。考虑给行做一个随机线性组合，然后往右边加上 $m*q/2$，这样知道 sk 的人还是可以减一下得到答案。

但我们好像还没用到 $sk$ 很短这件事，现在随便一个 $Ax=H(id)$ 的解都可以解密，但这样的解太多了。为了考虑长度，我们这样搞：给当前这个线性方程的每一项系数加一个 $q/|sk|$ 级别的扰动。这样的话只有拿到足够段的 $sk$ 的人才能正确解密。

这安全吗？看一下左边，有一个 $A$，做了随机行线性组合之后，再每一项加一个随机值。可以发现转置一下这个就是 LWE。所以根据 DLWE Assumption 左边就是 close to random 的。右边根据 Random Oracle 自然是随机的。

但这里定义 Security 的时候，我们还有其它若干个人的 $sk$。但这就和 Signature 一样，我们可以采样出剩下每个人的 $(H(id),sk_id)$，然后就安全了。

#### Lattice Trapdoor II: Bonsai Technique

我们能不能用一个 Trapdoor 造出更多的 Trapdoor？我们已经知道，在某些情况下不行：如果有 $A$ 的 Trapdoor，那显然不能得到 $A+B$ 的 Trapdoor。

但另一种情况是可行的：$A|B$。我们可以用 $T_A$ 得到一组 $Ax=B$ 的解 $M$，然后就有

$$
\begin{bmatrix}A&B\end{bmatrix}\begin{bmatrix}T_A&-M\\&I\end{bmatrix}\equiv 0
$$

直接的结论：如果我们能够对很长的 matrix 的一部分列拿到 Trapdoor，我们就能拿到整体的更大的 Trapdoor。

一些应用：

##### Lattice Signature, revisited

Gen: 生成 $2k+1$ 个矩阵：$A_0,A_1^0,A_1^1,A_2^0,A_2^1,\cdots$。同时生成 $A_0$ 的 Trapdoor。

Sign 一个长度为 $k$ 的 Message $s$ 的时候，拼接 $A_0A_1^{s_1}\cdots$，然后解 SIS。

Security Proof：我们使用一些模拟技巧，以说明：即使 Adv 拿到其它所有信息，他也不能解最后一个 SIS。记 $x^*=00\cdots0$，那我们生成所有 $A_i^1$ 的 Trapdoor，这样就可以 Sign 任何除去 $00\cdots0$ 的东西，但它对 $0$ 显然无关。

##### Hierarchical IBE

和刚才差不多，但是所有人构成一棵树，然后每个人可以解密自己子树里面的东西。

构造：每个人的 SIS matrix $A$ 是他父亲的 $A$ 拼接上自己的一段，然后有自己这一段的 Trapdoor。

#### Witness Encryption

考虑一个 NP instance $M,x$，即我们想要找到一个 $y$ 使得 $M(x,y)=1$。要求：

1. 我们可以加密，然后存在解密算法，解密成功当且仅当输入一个 $y$ 满足 $M(x,y)=1$。
2. Security 1: 如果不存在 $y$，则没有人能解密，即 $Enc(0)\approx Enc(1)$。

还有更多可能的 Security，但大家都不会证。

iO 的构造：iO[验证输入，如果对就输出 $m$]。那如果所有东西都不对，根据 iO 定义自然 $Enc(0)\approx Enc(1)$。

奇特的构造：我们试图加密这样一个 NP Problem：给 $n$ 对矩阵 $(M_i^0,M_i^1)$，每一对里面选一个，使得乘起来是 $O$。显然这是 NP-c 的（某种 Set Cover，或者直接 3SAT）。

假设拿到了所有的矩阵 $M$，然后这样搞：

1. 随机一车 $A_0,A_1,\cdots,A_n$，和它们的 Trapdoor
2. 对于每个 $A_i$，用 Trapdoor 生成 $D_{i+1}^{0,1}$ 使得 $A_iD_{i+1}^{0,1}=M_{i+1}^{0,1}A_{i+1}$。

输出所有 $(A,D)$。可以发现 $A_0D_1D_2\cdots D_n$ 可以一路乘过去，变为 $M_1\cdots M_nA_n$。

#### Other Constructions from Lattice

OWF from lattice: 

1. Ajtai's OWF Family
2. LWE Function: $f(A,s,e)=A,A^Ts+e$。根据定义这就是 LWE Hard。（可以证明，Uniform Error 也差不多是这样的）

##### PRF Construction

Idea 1: 随机 $n$ 对矩阵 $(M_i^0,M_i^1)$，然后拿到一个 $x$ 对应乘起来。

但这个**不是** PRF，来自某次考试：

> 只考虑它的一个位置。如果我们把连乘序列分两半，那某个位置就是前面一行和后面一列的内积结果。
>
> 那我们前面随 $2n$ 个，后面随 $2n$ 个，然后两两做一遍，得到一个 $2n\times 2n$ 的矩阵。这个矩阵的每个元素是一行和一列的内积，那整个矩阵是 $2n\times n\times 2n$ 的矩阵乘法的结果。众所周知这个的 rank 最多是 $n$。
>
> 但显然随机矩阵大概率满秩。

一个 Idea：Rounding 后输出。没人会证，但也没人会证。

另一个构造：把 $q$ 放到很大，然后在最后再乘一个 $A$。

为什么这就对了？因为如果 $A$ 随机，那加一个小随机就有 $M_n^iA+e\approx U$，然后不断往前消。

这样 $e$ 往前有指数级放大，所以需要指数级的 $q$。