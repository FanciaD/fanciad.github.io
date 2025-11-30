---
title: Weighted Hops — Restricted Shortest Path and Combintorial Flows
date: '2025-11-30 11:03:14'
updated: '2025-11-30 11:03:14'
tags: Fancia
permalink: SongshfHaasheer/
description: Weighted Hops — Restricted Shortest Path and Combintorial Flows
mathjax: true
---

一定没有写完，但是下次再说。


我们先来看一些东西，它们的原问题很难，但是 hop-bounded 版本好像没那么难。

#### Hop-Bounded Versions

##### Bounded-Hop SSSP

很好，大家都会 $\tilde O(mh)$ 的做法。

##### Bounded-Hop flows

如题，你希望找到一个尽量大的流，使得它里面的路径平均长度不超过 $h$。

直接做非常难，所以我们允许同时近似流量和长度——你可以找一个平均长度不超过 $100h$ 的流，使得它的流量是平均长度不超过 $h$ 的最大流的 $1/100$ 就行。

和 Flow or Sparse Cut 的问题一样，做法又是万能的 push-relabel。考虑直接跑一个 $2h$ 高度的流*——写个 LCT 的话这就是 $\tilde O(mh)$ 的。那为什么这对呢？

*注意即使你在残量网络上的路径是每条都小于 $h$，也可能对消两下就变成了一短一长。所以为了和谐友善，这里写的都是平均长度。

**可以把 $2$ 换成 $1+\epsilon$，因为是 $(a-1)/2a$。

假设最好的平均 $h$-hop 流量为 $f$，但我们找到的流量只有 $f/5$。那么，可以把那个流量里面剩下的 $0.8f$ 在我们现在 push-relabel 的图上继续流过去。此时总距离不超过 $f\cdot h+ (f/5)\cdot 2h=1.4fh$，那么这部分平均距离只有 $1.4h$。但我们已经把 $2h$-hop 的push-relabel跑完了，此时剩余部分每条流量至少是 $h$ 的距离，因此矛盾。

##### Restricted Shortest Path

定义：每条边有两个权值：$l_e$ 和 $d_e$。做 SSSP，但是是在保证路径 $\sum d\leq D$ 的情况下，最小化 $\sum l$。

这回已经知道做 exact 是 NP-Hard 的（某种背包问题），所以你可以大力近似——这次还是可以同时近似 $\sum l$ 和 $\sum d$。

这东西之前最好的做法还是 $\tilde O(mn)$ 的，所以我们先假设有如下条件：已知最优解的步数不超过 $h$。此时有一个经典背包近似：还是做 DP，但是所有值上取整到 $(1+\epsilon)$ 的指数上。这样每一步会乘一个 $1+\epsilon$ 的误差，所以误差是 $(1+\epsilon)^h$。因此需要取 $1/h$ 级别的 $\epsilon$，最后复杂度 $\tilde O(mh)$。

但图不一定有这么好的性质，这怎么办？我们有一个稍微弱一点的版本：

#### Weight-Bounded Versions

##### Weight-Bounded SSSP

> 这只是一个理解用教程。但这个 idea 可以用到更多的问题上。

现在考虑每条边有一个额外权值 $w_e$，保证最短路上权值和不超过 $n$。那我们能不能做得更好？

考虑按照 hop 数（现在是 $\sum w$ DP）。直接做还是 $O(mn)$ 的，不过每条边上有一些特别重叠的转移（比如 $1\to w+1,2\to w+2,3\to w+3,\ldots$）。如果我们稍微近似一下，这东西就可以做到很稀疏：考虑不把上面那些东西全部连过来，而是只连 $w\to 2w,2w\to 3w,\ldots$。这样的话，每次要用这个转移边的时候，我们最多延迟一倍的时间再去用。那么这样做 $2n$-hop 即可完成问题，而复杂度变成了 $\sum_e n/w_e$。如果我们有一个好的 $w$，那这就比 $O(mn)$ 更快。

然后你会发现，这个 idea 可以复制到更多 hop-bounded 的想法上。

##### Weight-Bounded RSP

把 $(1+\epsilon)^k$ 看成层数，边权为 $w$ 的边只在 $w$ 的倍数层数处转移。

##### Weight-Bounded Flow

还是 push-relabel，但是边权为 $w$ 的边只在 $w$ 的倍数高度处推流。此时可能有一点标号小细节，但不重要。

#### Finding a Weight Function

现在我们希望找一个好的权值 $w$，使得路径都满足 $\sum w_e\leq n$，然后 $\sum_e n/w_e$ 尽量小。

##### Example - DAG

最简单的例子是 DAG 的情况。此时考虑拓扑排序，然后 $w$ 就是拓扑排序的差。这自然满足条件，同时 $\sum n/w_e$ 是调和级数，那就是 $n\log n$ 级别。这就是非常好的。比如，我们可以有一个 $\tilde O(n^2)$ 的 DAG Max Flow(Integer Weight)。

##### General Cases - LDD with extra edges

那一般情况怎么做呢？或者说，能不能把一般情况变为 DAG 的情况？

我们考虑 RSP 的情况（2410.17179）。此时，我们先假设我们想要的路径里面 $\sum l,\sum d$ 都差不多 $(1\pm\epsilon)$ 是 $n$（最后枚举一下缩放比例）。那就是说，我们希望所有长度是 $O(n)$ 的路径都可以边权很小。

但这不一定是可行的！考虑给一个完全图，那路径实在太多了，我们只有全部 $w=1$ 才可以对所有路径保证这些东西。这不太好。但另一方面，一个完全图里面我们其实没必要绕来绕去，不如直接走过去。

那我们能不能把完全图直接替换成直接走过去的边呢？问题是，我们想要保证这样替换的边不会爆掉之前的最优解，也就是本来走过去的最优路径一定不能太长，也就是直径比较小——因此拿出 LDD：如果我们是一个 $D$-LDD，那么我们加一个额外点，往所有点连双向 $D/2$ 的边；这样不会爆最优解，同时我们最坏只增加了 $D$ 的路径。（两种边权怎么LDD？加起来做，显然正确。）

但这样还是加了 $D$。因此考虑对大小大于 $\epsilon D$ 的东西这样 shortcut，别的暴力做，从而总误差是 $\epsilon n$，可以接受。此时，小的 SCC 内部我们设置 $w=|SCC|$，这部分对 $\sum n/w_e$ 的贡献是 $|SCC|^2\cdot n/|SCC|$，加起来就是 $n^2$。然后每个 SCC 里面对路径上 $\sum w$ 的贡献是 $|SCC|^2$，加起来是 $nD$ 级别。然后 DAG 上的边拿 DAG 做法就是 $\sum_p w,\sum_e n/w$ 都是 $n$ 的……但我们还要考虑 LDD 删掉的边，这可能导致在 DAG 上来回走很多次，所以这样搞的 $\sum_p w$ 实际上是 $n^2/D$ 级别，因为期望删掉 $n/D$ 条边（此处省略了一万个 $\tilde O$，回顾 LDD 里面删边概率是 $l/D\cdot \log^3$。）。所以取 $D=\sqrt n$，期望路径长度 $n^{1.5}$，因此复杂度 $n^{2.5}$。（再 Scale 一遍，之前 $\sum n/w_e$ 是 $n^2$ 的）

##### LDD Hierarchy

现在小的 SCC 内部太暴力了，考虑能不能更好。注意到里面的问题相当于，给每条边一个权值，使得所有 $\sum (l+d)$ 很小的路径的 $\sum w$ 都很小，然后 $\sum n/w_e$ 不大——对，这就是一模一样的问题，所以套娃。

一个问题是我们并不知道哪边是 $\sum (l+d)$ 很小的路径，所以实际上是每块都建 shortcut，然后分析的时候对于合理的部分（$\sum l+d$ 和我们期望的值差不多）递归，别的部分（$\sum l+d$ 总的是 $n$，所以大的不多）暴力 shortcut。

然后好像就做到 $n^2$ 级别了（$\tilde O$ 忽略，假装 $\epsilon$ 是常数）。具体细节下次再补。

#### Finding a Weight Function - Flow Cases

咕了。

简单来说，我们希望找一个 Weighting 使得 Flow 的 Average Weight 很小。

注意到根据经典 Approx Flow or Sparse Cut 结论*，一个 Expander 有非常小-hop 的近似 flow，所以拍一个 Expander Decomposition，甚至是 Expander Hierarchy。

*考虑给每个点 $\phi/3$ 倍流量，跑 push-relabel，跑 $\log$ 层。考虑一层到下一层的 Cut，那么根据 push-relabel，有两种边：
1. 正好是两层之间的边，可能没满流。
2. 跨越更多层的边，必定满流。
此时两类边加起来至少是 $\phi*vol(left)$，但我们又保证了总流量只有 $\phi/3*vol(left)$，所以这一层连到下一层的边至少有 $\phi/3*vol(left)$ 这么多。但这些边是会不断叠加的，然后就指数地爆炸了。

然后可能就有 2406.03648 了。再然后有人拿出了 Shortcut 技术（LDD 那种感觉），然后有了 2510.17182。