---
title: Expander 学习笔记
date: '2024-11-06 14:55:55'
updated: '2024-12-05 15:22:08'
tags: Fancia
permalink: Hanashirube/
description: Expander
mathjax: true
---

### Expander Views

Robustness: 删掉 $d$ 条边最多使得 $d/\phi$ 个点不连通。

Cut/Flow: Cut/Flow approx to product graph

$\leq_{cut}$：很直接的定义。

$\leq_{flow}$：不那么直接的定义：我们把每条边看成一个 $u$ 到 $v$ 的流量，然后每个图对应很多组独立的流量，它们一起在另一个图上流。

这就是为什么 $\leq_{cut}$ 不能直接导出 $\leq_{flow}$：如果你定义成任意一个 demand 的话就是对的，但是上面那个情况会差个 $\log$ 倍（证明有空补）。后面这个才是真正正确的定义。

Spectral: $\lambda_2=\Omega(1)$

Mixing Time:  $1/\lambda_2$



Open: Vertex version? Directed graph?

### Local Min Cuts

vertex (set) $x$.

Find a cluster: sparse cut, find many edges,... only looking near $x$.

"looking": see adjacent list of $u$.

例1：有向图，一个点 $x$，两个数 $v,k$：找到一个包含 $x$ 的小点集 $L$，使其向外出边不超过 $k$。然后是近似版本：要么说明大小（度数之和） $\leq v$ 的 $L$ 不存在，要么找到一个大小 $\leq O(vk)$ 的。（不近似的版本甚至是 FPT(W[1])-hard 的：很难 $n^k$ 以内）要求 constant prob 正确，复杂度 $O(vk^2)$。

保证 $v$ 相对小。为什么？如果 $vk>m$ 那找出来的解没有意义。所以 $v<m/k$。


Local: 我们的复杂度和 $|G|$ 无关。

定义：$vol$ 是每个点的双向度数之和，$\delta(D)$ 是向外的割。

Task 1: 存在一个 $L,|L|\leq v$，使得向外有 $1$ 条边，向内有 $0$ 条边。

Sol to task1: 从 $x$ 跑一个 DFS。在 $v+1$ 步之后（！），我们一定走到了外面，且停下来的点在外面。然后考虑 $x$ 到停下来的点的路径，我们把它翻过来再做 DFS，找到的东西和链相交是前缀，所以此时 DFS 到的东西一定是一个合理解（得到的东西 $\delta=1$）

Task 2: 向外有 $1$ 条边，向内可能有边。要求正确率 $99\%$

此时之前的问题在于 DFS 可能走回 $L$。但考虑 DFS $100v$ 步，然后随机选一个，此时有 $99\%$ 的概率它停在外面。

Task 3: 向外有多条边？

我们再看看翻转之后发生了啥：它使得点集向外的割数量可能减少了 $1$。这对多条边也对：如果在外面，那么进进出出最后还是 $-1$。

那我们可以做多次：重复 $k$ 次，每次 DFS $100vk$ 步，然后随机选一个终点的路径翻转。如果有一步 DFS 不动了，那我们找到了一个 $\leq 100vk$ 的合法解（原来 $\delta\leq k$）。

首先，如果有一步停下来了，那因为每次翻只会让 $\delta-1$，因此找到的一定是 $O(vk)$ 的解。

然后需要证明一定停下来。考虑保证的那个解，每一步有 $1/100k$ 的概率停在外面。正确率 $(1-1/100k)^k$。





例2：Find small (directed) cut $\leq k$。复杂度 $\tilde O(mk^2)$

考虑套上面的做法。假设有一个割的一侧度数和为 $v$。那我们可以跑上面的算法……首先起点是啥？那就随机一条边，期望 $m/v$ 次，要whp需要加 $\log$，但因为单次 $O(vk^2)$ 所以是对的。然后但我们不知道 $v$。不过可以发现，只要开大一点，上面的做法在固定一个 $v$ 的时候，能对 $[v,2v]$ 里面的东西都对。那么我们可以倍增枚举一串 $v$ 过去。

但如果 $v>m/k$，上面的近似就炸了。但此时有极其简单的做法（！!1）：同时随机 $s,t$，期望 $k$ 次正确。单次网络流 $O(mk)$。

如果图是 Expander，那每个集合都很小，同时我们可以忽略第二种情况。

### Isolating Cuts

考虑无向图。

给一个点集 $T$，对于每个 $v\in T$，计算 $(v,T\setminus v)$ 之间的最小割。要求 $O(\log T)$ 次 maxflow。

那唯一能干的事情是跑个二进制：第 $i$ 轮下标第 $i$ 位为 $0$ 的和第 $i$ 位为 $1$ 的跑个割。（这就对有向图看起来没那么道理）

然后造一个 cut 的方式看起来只有一种：每个点按照它在每一轮里面在哪边标号，然后直接按照标号不同割。

当然这显然不那么对：给一个菊花，那总有一边是把其它叶子割掉。

然后有一个猜想：找一个当前 $v\in T$ 的点集里面的一个最小割。实现方法是把向外的边连到同一个点上，然后跑小的最小割。



事实上它就是对的。那为什么呢？

Lemma 1: 对于每个点 $v$，存在极小的 Min Isolating Cut: 其它最小的都是它的超集。这里我们用点集表示一个 isolating cut.

Proof. 考虑有两个同样的最小割，那现在的图就像：

```
A(u)-v1---B
|       / |
v2  -v5-  v3
| /       |
C----v4---D
```

那么 $v_1+v_4+v_5=v_2+v_3+v_5$ 都是最小割（对应 $AB,AC$）。但 $v_1+v_2,v_3+v_4$ 也是割（对应 $A,ABC$），所以这四个只能相等 且 $v_5=0$（因为我们都没用到 $A-D$，可以忽略这部分）。那么 $A$ 自己就是更小的。一直合并下去即可。

事实上这个结论就是说这是 Submodular 的。

Lemma 2. 极小的 Isolating Cut 在我们找到的东西里面。

Proof. 直接考虑每一步。如果当前大的 Cut 把我们这个极小的东西切掉了一部分，那这一部分补回来严格更优。因此直接结束。



有向图你也可以这样分析，但最后一步可能每个点集都非常大。



##### Steiner Mincut

给一个点集 $T$，找一个划分了 $T$ 的最小割。

$T=V$ 是啥？全局最小割。

$|T|=2$ 是啥？普通最小割。

Subtask1: 保证最小割一侧只有一个 $T$ 里面的点。

那这就是 Isolating Cut.

那怎么做 $>1$？有一个技巧叫 $NP\to_r UP$：$1/k$ 概率随机选点，那 $[1/2k,2k]$ 大小的集合都有常数概率只剩下一个。

然后复刻一遍：枚举 $k=2^i$，每层采样一堆，就是 whp 正确。



例3：给一个 expander 和一个 $s$，找到所有的 $t$ 使得 $cut(s,t)$ 很小。

根据 expander，每个这样的 cut 都是 local 的。如果 $s$ 有一个，那外面的点都合法，里面的暴力。

否则，每个 $t$ 对应一个 local cut，它的大小很小。然后上 isolating cut：每次放一个 $s$，随机一堆点，如果选到一个点同时没选到邻域里面第二个点就赢了。然后确定好随的概率一直跑。



### Cut-Matching Game

有 $n$ 个点。每轮：

1. Cut Player 将点集分成两半。
2. Matching Player 给一个从一边到另一边的最大匹配（$\min(|A|,|B|)$）
3. 将匹配加入图，然后如果图变成了 expander 就结束，否则继续。

Cut Player 希望游戏尽早结束，Matching Player 则相反。



Note. 如果 $A$ 里面包含了一个 Sparse Cut $S$，那操作后 $S$ 就不是 Sparse 了。如果没有 Sparse Cut 就赢了。



考虑这样一个操作：记第 $i$ 轮加入的匹配是 $M_i$，我们从一个点开始随机游走，第 $i$ 轮里面，如果 $M_i$ 不包含这个点则不动，否则有 $1/2$ 概率走到另一侧。

记 $p_i(u,v)$ 表示 $i$ 步后 $u$ 到 $v$ 的概率，$P_i$ 表示对应矩阵。那么 $P_0=I$，$P_i$ 是 $P_{i-1}$ 右乘一些列变换：把匹配的列做平均。然后容易发现每一列总和都是 $1$。

那这有什么意义呢？

1. 每一个 $p(u,*)$ 在不同的点之间流动。初始是聚集在一个点上，最后通过这些匹配到了其它点。
2. 每条边上的流量正好是 $1$：每一列总和是 $1$。

所以 $p$ 的移动相当于一个网络流。

我们的目标是，$\forall u,v,p_k(u,v)\geq 1/2n$。这说明啥？我们取两个 $S,T$ 做最小割，那么流量至少是 $\frac{|S||T|}{2n}$。证明就用 $p_i$ 的流量。因此取一个 $K/2$：完全图，边权 $1/2n$，则 $G$ 在任意 pair 的 $flow$ 上不比 $K/2$ 差。这记作
$$
K/2 \leq_{cut} G
$$
另一方面，流量至多是 $\min(|S|,|T|)*k$，其中 $k$ 是轮数。这分析每轮匹配就可以。那么反过来有
$$
G\leq_{cut}2k\cdot K
$$
原因是 $|S|,|T|$ 中至少有一个 $\leq n/2$。那因为它和 $K$ 在 Cut 上很像，所以它是 Expander（回顾 Cut expander 的一种定义）

因此 Cut Player 的目标是，在很小（$O(\log n)$）轮里面让 $p_k(u,v)\geq 1/2n$。（我们一般认为 $1/poly\log$ 的 Expander 都是好的，但是 $1/poly(n)$ 不行）



#### Deterministic ways

On a Cut-Matching Game for the Sparsest Cut Problem

定义：

Expansion of $S$：$\Psi_G(S)|=E(S,V\setminus S)|/\min(|S|,|V\setminus S|)$（这里分母考虑点数）

Expansion of graph：$\min_S \Psi_G(S)$

$\beta$-balanced: $\min(|S|,n-|S|)\geq \beta n$



1. 如果存在一个 $1/4$-balanced but $(A,B)$ 使得 $\Psi_G(A)\leq 1/100$，则输出这个 Cut.
2. 否则，找一个 $A,B$ 使得 $|A|\leq n/4,\Psi(B)\geq 1/400$（这是个子图），输出这个割。然后我们声称 $\Psi(G)\geq \Omega(1)$.



##### Entropy Potential

定义 $\Pi_i(a)$ 表示从 $a$ 开始做上面的随机游走，$i$ 步后分布的熵。那么 $\Pi_i(a)\in[0,\log n]$。我们的目标是 $\Pi$ 很大（看起来是 $\geq \log n-1$，考虑原来的目标）

定义 $\Pi_i$ 是每个点的势能之和。



结论：在第一种情况中，每步 $\Pi_i$ 至少加 $\Omega(n)$。

证明：首先 $-p\ln p$ 在 $[0,1]$ 里上凸，所以混合不会变差，我们只需要看增加了多少。

考虑较小的那一边 $S$。那么因为 $\Phi_G(S)$ 很小，这里面出去的边数不超过 $|S|/100$。从而现在的分布里面，$S$ 里面的东西几乎都还在 $S$ 里面。但下一步之后，这里面换了一半出去，那么看起来大概每个交换都使得两边 Entropy +1。

更严谨地，我们只考虑那些几乎都还在里面的点： $p(s,S)\geq 0.9$ 的 $s$，这样的点至少有 $0.9|S|$ 个。我们只需要证明，每一个这样的点给 $\Pi$ 增加了 $\Omega(1)$。

再看一下：左边所有数加起来 $0.9$，另外一边加起来 $0.1$，然后做平均。用一些数学结果：
$$
\forall p>2q,(p+q)\log(2/(p+q))-p\log(1/p)-q\log(1/q)\geq \Omega(p)
$$
证明：纯数学。

那么增加的确实是 $\Omega(1)$：显然有很多东西满足 $p>2q$。



结论2：第一种情况结束时，我们可以找到 $|A|\leq n/4,\Psi(B)\geq 1/400$

证明：考虑每次找一个不合法的 $S$，然后把 $S$ 从图里面删掉，加到 $A$ 里面。

注意到如果每个依次对剩下的是 Sparse 的，那它们并起来对外还是 Sparse 的。所以不可能加到 $n/4$ 往上。但可能一步加过 $1/2$，所以得多算。



结论3：一个 $3n/4$ 的 $\Psi(B)>\Omega(1)$ 加上一个连到 $n/4$ 个点的匹配使得整个图 $\Psi\geq \Omega(1)$。

证明：随便取一个割，如果 $A$ 在的部分大小是 $O(B$在的部分的大小$)$，那直接用 $\Psi(B)$ 就能得到一个 Bound。否则，注意到 $A$ 向外的匹配很多，$B$ 都放不下，所以最后很多。

Note.这个 $\Omega(1)$ 是在减少的。

然后就对了。



#### Randomized ways

Graph partitioning using single commodity flows

另一种描述目标的方式：最后的矩阵接近完全平均。意思是，如果我们随一个 $r\perp 1$，然后算 $rM$（这可以线性），那结果应该接近 $0$。

那不是 $0$ 就意味着不那么对，因此一个直接的尝试是：把较大的一半和较小的一半分别拿出来做匹配。

为什么这有道理？相当于做一个随机游走的 power method 找次大特征值(!)。



##### Difference Potential

这次定义的势能直接是 $P_i$ 和全 $1/n$ 矩阵逐位差的平方和。

一开始：$\Theta(n)$

目标：$O(1/n^2)$（这样可以使得每一个都很大）



换一个看法：$n$ 维空间上有一个目标点 $(1/n,1/n,\cdots)$，有 $n$ 个起始点 $(0,0,\cdots,0,1,0,\cdots)$。每次操作是选一个匹配，然后每个匹配两个点变成平均。

看一下这样平均对势能的减少：不妨设匹配点在 $(-d,0),(d,0)$，目标点在 $(x,y)$，那么势能减少是 $2x^2-(x+d)^2-(x-d)^2=2d^2$。因此得到结论：

1. 势能不增，每步减小量是对应每个匹配距离 $d$ 贡献一个 $\frac12d^2$。

因此我们希望，每次匹配到的东西都是距离很大的。那上面那个东西又有一点很对：它还可以看成，随一个向量，然后所有目标点投影过去。显然距离很大的东西投影后距离大概率不小。

首先我们的点都在 $(1,1,\cdots,1)$ 的某个仿射正交空间上，所以可以在这个 $n-1$ 维空间上做。此时可以假设目标点是 $0$，然后所有点的平均也是 $0$。

1. $d$ 维空间上，向量随机投影到单位向量的长度期望是 $1/\sqrt d$ 倍。
2. 投影之后，按照大小排序前后匹配的距离不小于它们到原点的距离平方和。

证明2：先找个分界点拆 $(a+b)^2\geq a^2+b^2$，然后变成到某个点的距离平方和，然后用和为 $0$。



这样拆一手，高维匹配 <=> 一维匹配 >= 一维原点距离和 <=> 高维原点距离和 = 势能

仔细一看，这不是一步期望减了一半势能？但问题是这是期望。一个折中的做法是拿随机投影的 Concentration 搞一手，可以得到 whp 减少了 $1/\log n$，那么需要 $O(\log^2 n)$ 次。

做到线性：先点乘 $r$。



### Using Cut-Matching Games

什么是一个 Matching Player? 各种 Lemma 里面：”我们可以构造一个匹配“。

#### Approx Expansion

回顾定义:$\min \frac{|E(S,V\setminus S)|}{\min(|S|,|V\setminus S|)}$.

Approx:找到一个 $\Psi(S)\leq \phi$，或者说明 $\Psi_G\geq \Omega(\phi/\log^2 n)$。

怎么说明后者？可以找到一个 $\Omega(1)$ 的 Expander $X$，然后说明 $X\leq_{flow}\log^2/\phi\cdot G$。

##### Matching Embedding

为了做 Expander Embedding，我们先考虑一个简单的情况。

给一个图和两个点集 $A,B(|A|\leq |B|)$，给一个 $\phi$，然后：

1. 找到一个 $\Psi\leq \phi$ 的割，或者
2. 找到一个 $A$ 到 $B$ 的最大匹配 $M$，使得 $M\leq_{flow} G/\phi$。

Sanity Check: $/\phi$ 之后 $\Psi=1$。那考虑先把边权变成 $1/\phi$。我们假设 $1/\phi$ 是整数。

考虑真的找一个匹配：源点向 $A$ 连边，$B$ 向汇点连边，跑网络流：

1. 如果满流，那么把这 $|A|$ 条流找出来，就是一个对的匹配。咋做呢？注意到直接搜复杂度是不对的：可能 $O(nm)$。但直接搜正确性是对的，那维护一个向前的 LCT Flow 就可以 $\tilde O(m)$（当然最开始的流复杂度不那么对，后面会解决）
2. 否则，找到一个更小的割。考虑 $A$ 里面有多少个点在割的左侧，如果有 $c$ 个，那么左边到右边还剩 $|A|-1-(|A|-c)<c$ 的边，反过来也一样。此时这个割一定满足条件 $2$。



##### Running Cut-Matching Games

然后出奇地简单：Cut Player 直接开始跑，每次询问 $(A,B)$ 交给 $G$ 上的 Matching Player。

$O(\log^2)$ 轮之后，如果提前发现条件 $1$ 就结束了。否则，我们得到了一个 $\Omega(1)$-Expander，每个匹配都 $\leq_{flow} G/\phi$，所以整体 $\leq G\log^2/\phi$。

那可以做到 $O(\log^2)$ flow，$\tilde O(n)$ randomized time。



#### Balanced Sparse Cut

1. 找到一个 $\Psi\leq \phi$，且 $\beta$-balance 的割，或者，
2. 说明每一个至少 $\Omega(\beta\log^2 n)$-balance 的割至少有 $\Psi\geq \phi/\log^2 n$。

先看看怎么满足第二个条件。如何只保证大的割？删掉少量边。例如，我们可以：构造一个图，使得它加上 $O(\beta n\log^2 n)$ 条边后，图是一个 $\Omega(1)$-expander。这样的话所有大的割都还是 $\Psi\geq \Omega(1)$。

然后咋做呢？想法和之前一样：每步

1. 找一个至少 $\beta$-balance，且 $\Psi<\phi$ 的割。
2. 或者是啥？注意到上面是删边，那这里应该是一个极大匹配删掉 $O(\beta n)$ 条边，且剩下的东西能被 Embed 到 $G/\phi$ 里面。

然后需要看第一步，但第二个需要的东西已经说明了我们要干啥：重复之前的做法，看流是否至少是 $|A|-\beta n$。

1. 如果割更小，那割的每一侧都至少有 $\beta n$ 个点（因为割不掉）。
2. 否则，把剩下的流量匹配，然后补全即可。



#### Weighted Expansion

现在的问题是权值不再是整数。那我们跑上面的做法时，我们实际上得到了一堆分数权值的匹配。

可以证明这样还是对的：考虑第二个算法，直接把取平均变成带权交换（那个 flow 的形式），势能减少也是好的：前一半减少的更多。



#### d-Expansion

这次点也带权，但是一个整数，表示度数，然后把 $|S|$ 换掉。

这次我们定义的一个极大匹配是很多边，但是左侧点度数等于 $d_u$，右侧点不超过 $d_u$。

在 Matching 那边，把 $S,T$ 连向源和汇的边换掉即可。

在 Cut 那边，Naive 做法：拆点。复杂度 $O(\sum d)$。对于算 conductance: $d$ 等于实际度数 是有用的。

不 Naive 的做法：在 KRV 里面隐式拆点。算权值的时候假装自己拆一下。



#### Extensions

Vertex expansion：把割边换成割点，那就是说在外面的网络流上，每个点有流量限制。首先这能做，然后这得到的东西就是 Vertex 上的 embed。

Directed Graph：继续换流。

Hypergraph



### Expander Decomposition

Unweighted, Conductance $\Phi$（点权为度数）。

#### Intuition

给一个图，我们需要将它分成很多部分，使得每个部分是 $\phi$-expander，同时希望删的边不多。

那有一个暴力做法：每次找一个 $\Phi<\phi$ 的割，然后直接拆了递归。

根据启发式分裂可知删掉的边数是 $O(\phi m\log m)$。



#### Definition

给一张图，删掉 $\phi m\log m$ 条边，使得剩下的每个连通块都是 $\phi$-expander。

E. g. 一个网格会分成非常多小矩形，大小只和 $\phi$ 有关。



#### Algorithm

但上面那个东西做不了：首先 $\Phi$ 就不能求（NP-Hard）。

那我们可以先套一个 Conductance 的近似，用上一步的算法。现在变成：删掉 $\phi m\log^3 n$ 条边。

但还有个问题：这个分治复杂度没有保证，可能每次删一个点。

一个很好的想法是，找到最 balance 的 sparse cut。

根据上次的做法跑二分，我们可以同时 $\log^2$ 近似 Conductance 的情况下 $\log^2$ 的近似 balance，也就是说找到的割一侧大小不比最优的小 $\log^2$ 倍。



如果没有近似这件事，那每次找最大确实是最优的：如果这一步拿了一个非常小的，下一步还拿一个非常小的，那它们合并起来还是这么小的 Cond。因此下一步拿的和这一次合并起来只能是超过了 $1/2$，那么下一步是非常接近平分。



但如果 Conductance 有一个 $>1$ 的近似比 $c$，那有一个严重的问题：两次拿出来的东西合并起来还是 $c\phi$-cond 的，但我们只证明了没有 $\phi$-cond 的，所以这样不能说明拿出来的一定不小。

那一个想法就是换一个更小的 $c$ 重新做。具体来说，我们定义一个当前的层数，第 $i$ 层用 $\phi\log^{2(d-i)}n$ 做，$d$ 表示最大层数。如果在第 $i$ 层我们切出了非常小的割（大小 $k$），那换掉下一层后，连着拿出小的东西合计不会超过 $k\log^2 n$。

先考虑一个暴力的做法。初始层数是 $0$。如果一次切出来，有一边非常小，$\leq k$。那此时切到层数 $1$，不断切小的（合计 $k\log^2 n$），这之后下一次切会非常平均。因为 $d$ 不能是 $\log n$，因此平均地向下递归（和切出来的小部分）时，我们都把层数变回 $0$。（之后不会用到前面的限制。）

还有个问题，如果每次 $k=1$，那上面分得非常愉快。但 $k=m^{0.1}$ 咋办？而且即使 $k=m^{0.1}$，那也要分 $m^{0.9}$ 次。因此这里再分层一次：第 $i$ 层我们取 $m^{1-k/d}$ 作为分界，如果切出来小于这个，那说明如果走到下一层，能切点数不会超过 $m^{1-k/d}\log^2 n$。那这样 $d$ 层的话，每层最多需要 $m^{1/d}\log^2$ 次操作，否则要么切不动，要么只能平均（>1/2），要么进下一层了。

那这样的话递归深度不超过 $\log n\cdot d\cdot m^{1/d}\log^2 n=dm^{1/d}\log^3 n$，近似比例是 $\log^{2d} n$。那 $d$ 取啥？取个 $\log$ 就可以发现，一边是 $\frac 1d\log n$，一边是 $d\log\log n$，忽略高阶项就该取 $d=\sqrt{\log n}$，从而复杂度/近似比都多个 $m^{o(1)}$ 倍，差不多比 $2^{\sqrt\log}$ 多一点。



#### Another Algorithm

上一个算法：近似比（多删的边）$m^{o(1)}$，复杂度 $m^{1+o(1)}$，可以换确定性。

这一个算法：复杂度 $\tilde O(m/\phi)$，近似比 $\log^3$，随机。



#### Extensions

直接换 Sparse Cut。

$d$-expansion：定义不变，删边不变，但删的边会变成 $\phi\sum d(V)\log(d(S)/d(V))$。

Vertex Cut: 最暴力的方法是每次删点，删掉 $O(\phi n\log n)$ 个点。

Directed Graph:  Sparsity 定义为向外边和向内边的最小值。默认用 Conductance。（Note: 非 SCC 自动 $\Phi=0$）

但这个时候怎么删边？我们知道 Sparse Cut 一边很小，但另一边就比较大，全删了就太多了。但只删一边的话，删 $O(\phi m\log m)$ 条边后，我们得到了一个 DAG，每个 SCC 是一个 Expander。



### Using Expander Decomposition



#### Repeated Expander Decomposition

跑一个 $\phi$ 比近似比小的 Decomposition，剩下 $\leq m/2$ 条边。然后对剩下的边继续做。

可以得到 $O(\log n)$ 个图，每个图是 $\phi-$Expander Partition，然后所有边并起来是原图。



理论上 $\phi$ 只需要 $<1/2\log n$，实践上 $\phi=1/poly\log$ 是可以的（randomized）或者 $\phi=m^{-o(1)}$（确定性）



常见用法：对每个 Expander 做事情，然后合并。

#### Spanner with Expander

> 喜报：没打过分层。

小结论：子图的 $\times\alpha$-Spanner 可以直接合并出大的 $\times \alpha$-Spanner。

证明：把实际的最短路拆到两边。或者直接对每条边说明。



然后考虑对 Expander 做 Spanner。

简单结论：$\phi$-expander 的直径不超过 $O(\log n/\phi)$，所以随便 bfs 取 $n$ 条边就是这么多（再 $\times 2$）倍的 spanner。

证明：bfs。$(1+1/\phi)^{\log n/\phi}=\exp(\log n)$。



然后就套做法，每个 Expander 找个 bfs tree 合并。总共 $O(n\log n)$ 条边，倍数 $poly\log$。（但分层直接）



如果图带权怎么办？~~来个 Capacity Scaling~~，按照权值分类做。$\alpha$ 再乘 $2$，边数多 $\log n$ 倍。



#### Cut Sparisifer

一个 Cut Sparisifer 的定义是，$G\leq_{cut} H\leq_{cut}\alpha G$。

$H$ 是 $G$ 的 Cut Sparsifer，如果 $H$ 边数不多，且满足上面那个条件。

注意到如果 $G$ 是个完全图，那就说明 $H$ 必须带权。

那还是一样：Compose Subgraph + Solving Expanders.

首先显然 Cut 有可加性从而 Cut Sparsifer 也可以直接并。



非常高超的结论：对于每条边 $e$，记 $\lambda_e$ 为两个端点间的最小割大小，那么：

1. 考虑每条边，以 $p>O(\log^2 n/\lambda_e)$ 的概率将其加入，边权为 $1/p$。

首先它期望正确。但更高的是：它 whp 是一个 $\alpha=1+\epsilon$ 的 Sparsifier。

[A General Framework for Graph Sparsification]

我们还需要考虑期望边数。好消息是：
$$
\sum 1/\lambda_e\leq n-1
$$
证明：每次找到全局最小割，那么这上面的 $c$ 条边都是 $1/c$。然后考虑删掉这些边递归两侧。注意到删边只会减少最小割，那么每一步都是在增大这个求和。总共正好分裂 $n-1$ 次。

然后随便 bound 一下上界。



为啥上面那个没做完？$\lambda$ 很难算。那我们拿出 Expander 技术，注意到那个是度数点权：
$$
mincut(s,t)\in[\phi,1]\min(deg(S),deg(T))
$$
因此过一个 $1/\phi$ 倍近似，边数多这么多倍。

跑随机算法的话，复杂度 $\tilde O(m)$，然后算一下上面的 $\epsilon$，$(1+\epsilon)$-$\tilde O(n/\epsilon^2)$。



**简单无权无向**图

#### Mincut-preserving Contraction

Contraction: 进行一些点合并。

[Karger Throup 15] 定义 $\delta$ 为图的最小度数，则存在如下合并：

1. 合并后点数 $O(n/\delta)$，边数 $O(m/\delta)$。
2. 如果一个最小割两侧点数都 $\geq 2$，则它在合并后还存在。

注意到最小割本来不超过 $\delta$。然后

[Gabow 95] 计算 $\min(k, 一个图的最小割)$ 可以 $O(mk)$ 时间，所以最后一步容易。



我们的目标是，对于一个某种条件的 Expander，我们可以找到它的一个子集，使得任何一个满足条件的最小割都不跨过这个子集。



这怎么可能？我们必须考虑**最小**割的性质。那考虑一个调整法：如果有一个交，我们把它调整出去可以得到更小的割，从而矛盾。例如，我们可以得到：

1. 如果有一个子集 $S$，满足每个点连到 $S$ 内的边数占其度数的至少 $2/3$，则任何一个最小割和它的交要么为空，要么至少是 $\delta/3$。

证明是简单的：如果非空但小于 $\delta/3$，那把它们全部移出去，每个点对割的变化是增加了 $d/3$，减少了至少 $2d/3-\delta/3$，那就是更小的。

但还有个问题：如果割完全在 $S$ 里面怎么办？那移一个点出去……所以这就是为什么我们只能保留点数 $>1$ 的割。



但首先，它不能解决交很大的情况。比如，$S$ 为全集显然满足上述性质。为此，我们希望找到一个集合，使得每个最小割和它的交都小于 $\delta/3$。（或者说，交大于 $|S|-\delta/3$。可以取反拿到等价。）



换言之，我们希望如果一个割把 $S$ 分成了两块，且每一边都大于 $\delta/3$，则此时割至少严格大于 $\delta$……因此我们可以考虑一个点数意义下的 Expander：把 Conductance 换成 Expansion，然后让 Expansion 是 $\Omega(1)$。因此一个很好的想法是跑这个意义下的一个 Decomposition。

考虑一下这样的效果。每次分裂的时候，代价变成了点数的最小值。如果用上面的近似，那么效果为删掉了 $O(n^{1+o(1)})$ 条边。这个边数是可以接受的。



然后我们需要在 Expander 里面找一个子集，尝试满足第一个条件。但这非常困难：显然我们应该能删就删。但这样好像很难停下来：我们要求比一半多还删，这样删了之后向外的边越来越多。



那能不能只动一轮？记之前的集合是 $S$，那么保留下来的每个点到 $S$ 里面的边数至少是度数的 $2/3$。这样的好处是，每次移动时只用最开始就有的向外的边，所以外面总边数还是 $O(n^{1+o(1)})$。如果一个最小割和保留下来的点有交，考虑移动出去，这使得：

1. 到外面的边使得割增加了不超过 $d/3$。
2. 到里面的边使得割减少了 $2d/3-|S\cap C|$。然后增加了 $|S\cap C|$。

那么只要交不超过 $d/6$ 即可。



然后每个 Expander，把剩下的东西合并。我们可以认为 $d>$ 一个很大的常数。这样边数是 $O(n^{1+o(1)})=O(m/\delta*n^{o(1)})$。那点数是啥？

1. Expander 分出来只有 $n^{1+o(1)}/\delta$ 级别。

证明：如果一个 Expander 大小小于 $\delta/2$，那么每个点贡献了 $\delta/2$ 条删掉的边，但总共是 $n^{1+o(1)}$。

2. 后面的步骤只加了 $n^{1+o(1)}/\delta$ 条边。

证明：每移出来一个点会用掉原先 $O(\delta)$ 条在外面的边。



#### Cut/Flow Vertex-Sparsification

给一个无权图和点集 $T$，我们希望保留 $T$ 之间的流/割：

1. 点数 $O(|T|)$。
2. 考虑 $A,B\subset T$，那么 $A,B$ 间的最小割几乎和原来一样。（polylog 倍）
3. $T$ 之间的网络流和之前几乎一样：定义是跑这个 flow 需要的最大边容量的最小值（Congestion）



### Push-Relabel

它有一些奇妙的用途.jpg



为了简便，我们考虑无向图。



1. flow $f(u,v)$：$u$ 到 $v$ 的流。如果反过来流就是负数。
2. congestion: $\max \frac{|f(u,v)|}{c(u,v)}$
3. Demand: 每个点有一个 $dem(u)$，一个流是合法的如果 $dem(u)=\sum f(u,v)$。
4. Source-Sink：每个点有需要流的 $\Delta(u)$ 和可以接受的 $T(u)$。合法条件是最后不超过 $T(u)$。这是上一个的推广。

合法性：Maxflow-Mincut，每一个割里面需要出去的东西大于等于能用的边权。



Preflow: 先流一些，不需要满足每个点上不超过 $T(u)$ 的限制。 



算法如下：

每个点有一个 level $l$。保证：

1. 只有 $l=0$ 的点还可以接受 $T(u)$。
2. 残量网络上向下的边最多走一层。

算法：每次找一个还有多出来的流量的点，然后：

1. 如果能往下推，就推过去。
2. 否则，$l(u)$ 加一。

显然这样操作满足上述保证。考虑 $l=n$ 时停止，则此时必然出现空层，然后就把残量网络分开了。

左脚踩右脚升天.gif



如果设一个层数上界 $h$，考虑复杂度：

1. 如果 push 把边用完了，那下次刷新是升层，所以这部分 $O(mh)$。
2. 否则呢？这个点的额外流量没了，流到了下一层。因此定义势能为所有有额外流量的点的 $l$ 之和。每次操作 $-1$，每次上一种操作 $+h$，所以 $O(mh^2)$。为了做到这一点，我们需要维护所有能用的边，或者当前弧。



#### Conductance Sparse-Cut

回顾之前的问题：每个点有一个度数，给 $A,B$，找到一个 $\Phi\leq \phi$ 的割，或者找到 $A$ 到 $B$ 的度数匹配，使得它 $\leq_{flow}1/\phi\cdot G$。



之前的做法是直接流，但直接流很慢，也很难直接变快。但这里我们可以近似：把第一个改到 $\phi$ 的若干倍也是问题不大的（反正等会右侧有 $\log^2 n$ 的近似比）



取一个 $1.1\phi$，此时会发生啥？考虑取一个小的 $h$，停下来时的图。此时从 $h$ 开始，假设我们找不到这样的割，则前若干层的东西向下还没流的流量至少是 $0.1\phi\min(vol(S),vol(T\setminus S))$（需要的流量是 $vol$ 这么多，但每条边 $1/\phi$。如果我们假设 $vol(S)$ 是小的一侧，那么每次往下一层，这个东西对应的边数就加进了 $vol(S)$，那么它就是 $(1+0.05\phi)^k$（可能有反向）。这样只需要 $O(\log n/\phi)$ 层，就会让 $vol(S)$ 达到 $poly(n)$。然后我们还可以从后往前做，也是只有 $O(\log n/\phi)$ 层。所以有这样的结论：

我们只需要跑 $h=O(\log n/\phi)$，就可以近似上面的东西。一步复杂度  $O(m\log^2 n/\phi^2)$。



#### Balanced Sparse Cut

类似的，跑 $S-k$ 大小的流。或者说，如果最后剩下的小于 $k$ 就通过，否则：

剩下 $k$ 的流自动说明大小至少是 $k$。



#### LCT optimization

注意到流满一条边只有 $O(mh)$，别的流有 $O(mh^2)$。那来一个 LCT 优化，每次跑满一条边即可，这是 $O(mh)$ 次。

但还有跑不满的步骤。此时注意到上面只有 $O(mh)$ 次给一个点流量，所以这样清空一个点流量也只有 $O(mh)$ 次。所以复杂度还是 $O(mh\log n)$。

这样上一个做法复杂度就变成了 $m/\phi\cdot poly\log n$



#### Local flow

往格子里面倒水，水大都停在了最近的格子里面.jpg

1. 每个点都是一个 sink。
2. 优先填满当前点的 sink，再往外面流。



注意到我们定义 local 是 vol(S) 很小，所以定义 Sink $T(u)=deg(u)$.

每个点有初始流量 $\Delta(u)$。我们希望算法复杂度只和 $\Delta(V)$ 有关。



好消息是能 $l>0$ 的点加起来一定 $\sum deg\leq \sum \Delta$，所以 relabel（升层）复杂度自动是 $O(\Delta(V)h)$。然后就可以直接跑。



##### Local Sparse Cut

换一个初始条件：给定 $A$，保证存在一个 $\phi$-sparse Cut $S$ 使得 $vol(S\cap A)\geq \delta vol (S)$.



那每个 $A$ 里面的点放 $deg(1/\delta+1)$ 流量，然后 sink 照常是 $deg$，这个东西就流不出去。所以直接流可以找到一个割，流量至少剩下 $1$。但因为我们可能搞到整个 $A$，我们只能保证 $\Phi\leq \phi/\delta$。

然后再来一次之前的近似：如果每一层之间的割都 $\Phi\geq 1.1\phi/delta$，我们就可以加倍，从而只需要 $h=O(\log(vol/\delta)/\phi)$。



about local clustering



#### Expander Pruning

回顾一个定义：删掉 $C$ 条边大概能让 $C/\phi$ 个点不连通。

Expander Pruning 1: 给一个 $\phi$-expander，删掉一些边 $D$，现在你需要删掉一些点 $P$，满足：

1. $vol(p)=O(|D|/\phi)$
2. 删点后图是 $\Omega(\phi)$-expander



考虑做动态问题：

Dynamic Expander Pruning: 每次删一条边，然后我们再删一些点，每一步后需要满足上述条件。

Dynamic Expander Decomposition: 对每个部分套上面的东西。



Boundary induced subgraph: 把外面一圈(Boundary Vertices)也放进来。

Near Expander: 如果我们也计入外面一圈的边。

我们的目标是，给一个 Near Expander，我们删掉 $O(外部边数/\phi)$ 个点后，它变成一个 Expander。

首先怎么用这个？把删边变成两个端点连到外面。



先做一轮的情况。考虑什么情况下移除外部边不对。移掉之后相当于 vol 和割同时减少了一个数。本来是说割除以 vol 是 $\phi$。也就是说，如果 Conductance 掉下了 $\phi/10$，那割至少需要减少到本来的 $1/10$。换言之，减的东西至少得是原来 vol 的 $\phi/2$ 倍。那跑个 local flow：流量是外部边，sink是每个点的度数，找到割就删。显然删掉的大小不超过 $O(|D|/\phi)$。



#### Dynamic Local Push-Relabel and Pruning

🕊🕊🕊



### Well-linked Graphs



Vertex Well-linked: 给定一个子集 $T\subset V$，合法当且仅当 $\forall A,B\subset T, cut(A,B)\geq \alpha\min(|A|,|B|)$。可以发现 $T$ 取全集就是 expansion。

也可以定义成 Flow，如果存在一个 All-pair 的 routing：这里会多 $\log$ 倍边权，可能是因为做一个 cut-matching game。



Edge Well-linked：给一个边集，每个边看成一个点。对应全集情况是 Conductance。

#### Boundary-linked Graph

一个图里面拿一个子图，同时把所有子图向外的边拿出来。然后要求这些边是 well-linked 的。



#### Boundary-linked Decomposition

给一个 Boundary Graph 和一个 $\alpha$，将点集分成若干部分，使得每个部分是 $\alpha$-Boundary-linked。（中间切出来的边也会算）



暴力可以和之前一样，不满足条件（能割）就切，那么存在一种删多少条边的方案？区别是切完会多出来向外的边，但只多 $\alpha$ 倍。分析一下分治树。分出两边的过程本来是两边切完加上 $\alpha\min(a,b)$，我们看成（假设 $a<b$）先分成 $(1-\alpha)a,b+\alpha a$，然后给左边加上 $2\alpha a$。这样就是小的那边向下再乘一个 $\frac{1+\alpha}{1-\alpha}=1+O(\alpha)$。差不多向下 $\log |\partial|$ 次。所以我们最好需要 $\alpha=O(1/\log|\partial|)$，这样增长是常数倍，从而总边数还是 $O(\alpha|\partial|\log|\partial|)$。



#### Boundary-linked Expander Decomposition

字面含义：同时满足两个定义。

现在 $\alpha=1/O(\log n)$，我们希望删边数是 $O(原来的边界+\phi\log m)$。



如何判定 Boundary-linked + expansion？注意到自环只改变 vol 不改变割，一个充分条件是给每个外部点再加 $\alpha/\phi$ 个自环，然后跑 $\phi$-expansion，就是上面的 Expander Decomposition。



然后对着抄，同时用两边的分析证明递归下去不大。



#### General Flow/Cut Sparsifier

给一个图，给很多向外边 $\partial$，要求缩边后尽量保留 $\partial$ 上**任意两个**子集间的最小割。

跑上面的东西，然后缩每个连通块，然后：

1. 显然操作后最小割比之前大。
2. 注意到如果切到一个块里面，那么根据性质放到外面不会坏超过 $1/\alpha$ 倍（里面一个割的边权大于 $\alpha$ 倍两边 $\partial$ 大小最小值）。

理论情况下，取 $\alpha=1/\log$，那我们就得到了 $log$ 级别的近似，边数 $O(|T|)$，这个就是 Vertex Sparsifier。

然后抄写一遍实用版本的 Expander Decomposition，得到 $n^{o(1)}$ 倍近似比。

如果两个端点在同一个里面怎么考虑？那就得用 Expander 的性质。注意 Expander 里面每个割都至少是 $vol/\phi$，那就是 $\sum deg$，因此可以造一个菊花图，每条边对应原来的度数。这样是 $1/\phi$-cut sparsifer。然后把这个和上面合起来就是 $1/\alpha+1/\phi$ 的 Cut Sparsifier。那它也是加 $\log$​ 倍的 Flow Sparsifier。因为 flow 和 cut 的定义几乎可以互相转换，只差一个 log。



从 Flow 的角度也可以直接证：首先，换到缩点显然流量不会变差。然后考虑倒过来的话：

1. 外部在这个块上的流量还原回去，用 Bounded-link，$\log /\alpha$ 倍。
2. 内部点的流量，用 Expander 导出去，$\log /\phi$ 倍。



### Expander Hierarchy

我们做一个刚才的东西：先缩点得到一个小图，然后每块向里面的所有点连边。

如果对小图继续做会发生啥？我们定义 Expander Hierarchy：一直做到只剩一个点为止。然后，每相邻两层里面把度数边放进去。

然后这甚至是一棵树。



考虑现在近似比是啥。从上面每下来一层都会有一个 $\log /\alpha$ 倍，但可以发现内部的不会去乘上上面的近似比，所以是 $(\log/\alpha)^h/\phi$。那深度是多少？可以说明 $\alpha$ 比较大，$\phi$ 比较小的时候，深度是 $\log_{1/\phi} m$。所以我们让 $\alpha$ 比 $\phi$ 大，然后就绕过了 $\phi$ 自己乘自己的部分。

例如，$\alpha=1/\log,\phi=1/2^{\sqrt\log}$，这样 $h=\sqrt \log$（可能还是 tilda）。所以最后近似比是 $n^{o(1)}$。



### Flow Approximation

Flow Approx. (Ver 1.) 给一个 Demand(正负都有)，然后：

1. 要么找一个割
2. 要么找一个 $(1+\epsilon)$ Congestion 的 Flow。

Flow Approx. (Ver 2.) 给一个 Demand(正负都有)，然后：

1. 要么找一个割
2. 要么找一个部分 Flow，使得剩下的部分满足 Congestion $\leq \epsilon$。

可以后者推前者：对剩下的部分重复用。（注意到我们保证 2 下去的情况不存在小的割。）最后的东西用一个 $m$ 倍暴力（比如随便一个 MST）



#### Congestion Approximator

怎么算 Congestion？一个直接的想法是之前那个 Expander Hierarchy 拿上来，然后在 $q=n^{o(1)}$ 倍近似下变成算 Tree Flow Congestion。注意到切两块下来不如只切一块（如果多个连通块不如少一点，一个一个非子树连通块就取反），那么只需要算每个子树。



我们定义一类算法是说，只考虑一些割，然后用这些割上的 Congestion 去近似。它的近似比 quality 就定义为最差近似比。

注意到树上的 Cut-Flow 显然近似比就是 $1$，所以 quality 就是 $q$。



之前的构造是 $q=n^{o(1)}$，但我们甚至有 $polylog$ 的构造。（1411.7631）



#### Multiplicative Weight Update

在解决了上一个问题之后，相当于我们找一个 Flow，使得每个剩下的东西上 Congestion Approximator 都不超过 $\epsilon/q$。

这可以看成一个 LP：我们有一堆方程，要求每个方程都最多差一点满足。



有这样一个算法：

每个方程有一个权值 $p$，初始全 $1$。我们把方程按照权值加起来，然后要求：

1. 满足求和后的一个方程。
2. 本来每个限制的差在 $[-\rho,\rho]$ 之间。

然后，我们给每个方程 $p$ 乘上 $\exp(-\frac \epsilon \rho d)$，其中 $d$ 是限制减出来的差。这样做 $O(poly(\rho/\epsilon)\log n)$ 轮，然后输出所有解的平均数。



这为什么对？考虑 $p$ 的变化过程。显然 $p$ 就记录了每一步差的和。那么只要最后这东西不大于 $\exp(n\epsilon)$，那它最后不会少 $\epsilon$ 以上。

不妨设 $\rho=1$。考虑每一轮发生了啥，我们给每个数 $\ln p$ 变化了一个量，它们和 $p$ 点乘不大于 $0$。同时，每个变化量都在 $[-\epsilon,\epsilon]$ 之间。

注意到虽然点乘不大于 $0$，但 $\exp(x)$ 是下凸的，所以增大的部分可以变大很多。进一步可知增大的量一定不大于减少的量的 $\exp(\epsilon)$ 倍。这样很多轮之后，差不多就是 $\exp(n\epsilon)$ 为主项。这需要至少 $n=poly(1/\epsilon)$。



直接复制过来，每个限制是一个割上流的东西应该接近某个数，准确地说是差不超过割大小的 $\epsilon$ 倍，因此我们做一个归一化。带权的话，就相当于每条边上面有个系数。因此我们希望这次选的流量内积系数之后足够大，那显然的贪心是直接选最极端的方向。

注意到归一化之后，这个方程的值显然在 $[-2,2]$ 里面，所以可以不用管 $\rho$ 的问题。

如果没有解怎么办？注意到这是限制的线性组合，原来是有个确切解的，那无解显然意味着原网络流不行。但我们需要输出一个割而不是输出 `N0`。

注意到我们给边的权值有很好的性质，每次是给一个割加，那可以相当于给每个点一个点权 $\phi_i$，然后每条边的边权是 $\phi_u-\phi_v$，这样可以叠加。然后这个限制需要的流量也正好是 $\sum \phi_id_i$，因为 $01$ 的时候这就是一侧的 demand。

所以不合法实际是，有一个 $\phi$ 使得 $\sum e_i|\phi_u-\phi_v|< \sum \phi_id_i$。从大到小扫 $d$，如果每一步的割都大那加起来就该大，所以存在一个 $\phi$ 的临界，割出来就是对的。（重新拆的关键是，拆完所有东西同向。）



那全部合起来就结束了，轮数是 $poly(1/\epsilon)$，好像是 $q^2\log/\epsilon^2$





https://courses.cs.duke.edu/fall19/compsci638/fall19_notes/
