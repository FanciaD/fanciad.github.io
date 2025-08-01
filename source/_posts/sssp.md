---
title: '[paper] Negative Weight SSSP'
date: '2025-03-16 05:09:57'
updated: '2025-07-23 06:58:43'
tags: Fancia
permalink: RingonoNegoto/
description: Negative Weight SSSP
mathjax: true
---

免责声明：我发现之前写的版本有至少三处错误。

所有时间以 CST 为准。

### Real Weight SSSP V3

#### mn^8/9

简化：

1. 假设每个点只有一条负权出边，不然单独来一个点放所有负权出边。
2. 假设度数均匀，不然拆点。

##### Reweighting, Layer 3

根据经典的 Johnson Reweighting，我们可以找一个 $d(u)$，然后将边权 $w(u,v)$ 变成 $w(u,v)+d(u)-d(v)$。如果对于每条边都有 $w'(u,v)=w(u,v)+d(u)-d(v)\geq 0$，那就搞完了。

一个显然满足这东西的 $d$ 就是某个单源最短路，也可以是全源最短路，而对于求最短路我们有：

> 假设图里面只有 $k$ 条负权边，那可以搞一个分层图，记 $d^i(u)$ 表示任意一个点出发，走不超过 $i$ 条负权边到 $u$ 的最短距离。分层图形如：同层之间只有正权边，不同层之间有负权边和跳上去的边。

一个折中的方式是说，考虑建 $r$ 层图，然后最后一层连向第一层。这样只需要这样跳 $k/r$ 次。然后来一个 Reweighting：第 $i$ 层点 $u$ 就是 $d^i(u)$。这样搞完除了最后那些跳的边之外，别的都是边权非负。因此在这个图上只需要再走 $k/r$ 条负权边。但直接搞点数就 $r$ 倍了。

注意到在分层里面，如果你加一条倒过来跳的边，那不影响正确性（最多就是提前跳），不过这在多数时候是加一条负权边。但如果 $d^0=d^1=\cdots=d^r=0$，那这样加相当于全部缩点，这样反而减少了点数。因此可以这样操作：找到满足 $d^r(u)=0$ 的点，把它们缩点，然后做上面的东西。

假设度数均匀，所以总复杂度是 $m(r+k/r)+mk*(剩下的点数)/n$。那么理想情况是剩下 $n/r$ 个点，这样复杂度 $m(r+k/r)$。

但这还是很难，因为初始有 $n$ 条负权边，这样搞 $r>1$ 就很难了。

因此考虑，每次拿一部分负权边出来做上面的过程。如果我们能找到 $k$ 条边，使得走 $r$ 条边后能负权到的点数只有 $n/r$ 级别，这样每一轮就是 $m(r+k/r)$，然后做 $n/k$ 轮。



##### Reweighting, Layer 2

我们的目标是，选出一些负权边，记 $U$ 表示这些边起点的点集，我们的目标是使得尽量多的（$n-n/r$）点满足 $d^r(U,x)\geq 0$。

但上面那个目标还是不直接可能：如果有一条特别大的负权边（单向的），就全寄了。

这里的做法是直接再来一次 Reweighting：目标是操作完尽量多的 $d^r(U,x)\geq 0$。那直接来一次多源最短路……但可能 $d^r(U,U)<0$，那就炸了。

也就是说我们需要一个东西满足 $d(U)=0$，然后 $d(x)$ 尽量不大于 $d^r(U,x)$。这样在 $U$ 附近可能不大对，然后希望这样的点比较少。

那考虑取 $\min(0,\max(d^r(U,x),-d^r(x,U)))$ 来解决第一部分。注意到如果两个 metric 都对一些边满足三角形不等式，那它们的 $\min,\max$ 都满足。

1. $\max$ 满足的原因是，如果 $v$ 减了一边的，那 $u$ 那一边加上去的一定足够大。
2. $\min$ 满足的原因是，如果 $u$ 加了一边的，那 $v$ 减的那一边一定比较小。



这样一波操作完，不好的点就是那些第一步取右边的，或者说 $d^r(U,x)+d^r(x,U)<0$ 的。



##### Sandwich Construction

但这东西太复杂，因此考虑找 $(x,U,y)$，满足 $\forall u\in U,d_1(x,u),d_1(u,y)\leq0$。

然后不好的点的必要条件是，$d^{r+1}(x,v)+d^{r+1}(v,y)<0$。

首先考虑怎么找到一个很大的 $U$。先看只有 $d_1(x,u)$ 的情况。此时如果有一个大的我们可以找到 $x$，否则每个点 $x$ 都只有 $k^{2/3}$ 个 $d_1(x,u)<0$。

> 虽然我们不能算出所有 $d_1(x,u)$，但是可以采样近似集合大小。

如果是前者情况，我们再往集合里面跑第二轮放入 $y$。否则，考虑贪心选，可以得到 $k^{1/3}$ 个点，使其两两 $d_1\geq 0$。同时每个点都是一条负权边的入边。



##### Solve Far Set

如果是第二种情况，那直接同时消所有负权边，把负权边出去的权值增大，然后往外推。根据性质不会推到集合内剩下的负权边上去。然后就直接消掉 $U^{1/3}$。



##### Solve Sandwich, Reweighting Layer 1

然后考虑有一个 $(x,y)$ 的情况，此时需要让 $d^{r+1}(x,v)+d^{r+1}(v,y)<0$ 的点比较少。那显然得再做一次 reweighting。

考虑随机选一个点 $t$，然后让 $d^{r+1}(x,t),d^{r+1}(t,y)\geq0$……但这好像对别的点没啥用，尤其是如果Sandwich 就是极端的三层，每个点一条路径的时候……但即使是这样，对于每个点 $u$ 考虑它的距离：$d^{r+1}(x,t)+d^{r+1}(t,y)$。这样操作之后，相当于这个距离变成了 $0$，但这样其它的距离也增加了这个值，因为这是 Johnson Reweighting。因此这样期望让 $<0$ 的只剩一半。一般情况下，随机 $O(r\log n)$ 个点，就只剩下 $n/r$ 的大小。

然后需要做一个 Reweighting，使得一堆 $d^{r+1}(x,t),d^{r+1}(t,y)\geq0$。

然后把所有的 $d^{r+1}(x,t),d^{r+1}(t,y)$ 拿出来，做一个 $O(r\log n)$-reweighting。此时复杂度 $O(r^2m)$，使得上面任意拿出一对 $(x,y)$ 都可以满足 $\leq n/r$。

那么和外面平衡的最好结果就是每次拿进来 $k$ 条边的话，取 $r=k^{1/3}$，以 $k^{2/3}$ 的代价消边，从而除掉一个 $k^{1/3}$。因为 $k=n^{1/3}$，所以复杂度是上面那个。

#### mn^4/5

> 喜报：之前这里又是十万个错误。

考虑找更大的 Sandwich，但上面那个东西看着已经很极限了。

那考虑**定义**更好的 Sandwich。

先考虑有一组 $(s,U,t)$，然后我们可以干啥。之前我们是拿 $\max(d^r(U,x),-d^r(x,U))$ 做操作。可以发现，在之前有了 betweenness 之后，拿 $\max(d^{2r}(s,x),-d^{2r}(x,t))$ 应该是更简洁的。

现在考虑：
1. 对于 BW 之外的好点，它们满足 $\phi(x)=d^{2r}(s,x)$。
2. 对于 Sandwich 里面的点，我们至少希望它们满足 $\phi(x)=-d^{2r}(x,t)$。也就是说它们至少需要满足 $d^{2r}(s,u)+d^{2r}(u,t)\leq 0$。

根据之前的目标，我们希望好点满足 $\forall u\in U,d^r(u,x)+\phi(u)-\phi(x)\geq 0$。那现在放过来，得到：
$$
d^r(u,x)-d^{2r}(u,t)-d^{2r}(s,x)
$$
……好像没有用。但注意到 $d^r(u,x)+d^{r}(s,u)-d^{2r}(s,x)\geq 0$。那如果我们希望 $d^{r}(s,u)+d^{2r}(u,t)\leq 0$，这就对了。

这样可以发现一个新的 Sandwich 定义： $d^{r}(s,u)+d^{r}(u,t)\leq 0$……可以发现这就是另一个 Betweenness，但是是小的方向。

考虑跑 $k$ 条负边最短路。此时有如下情况：

1. 第 $k$ 轮啥都没发生，这说明我们已经求出了最短路。
2. 如果有更新，那说明 $s$ 到 $t$ 有一条路径（权值为负）经过了 $k$ 条负权边——此时可以进一步判断出如下两种情况之一：
3. 如果这些边有重复，那么有负环。
4. 否则，$s$ 到 $t$ 的最短路经过至少 $k$ 条不同负权边。

那第四条有什么用呢？注意到所有这些负权边的点都在 Sandwich 里面。我们希望说找到的东西 Sandwich 比较大，这样上面就能一步到位。这样就能说明至少有 $k$ 大小的 Sandwich……但这看起来没啥用。

考虑随机 sample 负权边进来，然后跑上述做法。如果找到一个至少经过 $k$ 条不同负权边的路径，那它原来的 Sandwich 必定很大——如果以 $1/t$ 去 Sample，那要么解决了 $n/t$ 条边，要么有一个 Sandwich，原来至少是 $kt$ 级别的。

如果是前者那就直接搞定了。如果是后者，那按照之前的方式，取 $r=(kt)^{1/3}$ 然后跑上面那堆。

然后取 $k=n^{1/5},t=n^{2/5}$，这样无论如何都解决了 $n^{3/5}$，时间 $n^{2/5}$，所以是 $mn^{4/5}$。

这里一大瓶颈是，无论如何我们都要先搞 BW Reduction，所以需要先搞 $n^{2/5}$。



#### New Hop Reducers

前情提要：focs 之前想了一个月这东西然后完全失败。同期有人搞出来一个并准备投 focs，但那东西怎么看怎么不对，然后在多方交流之后对面也觉得不大对了。挑战最速withdraw传说.jpg 然后 SODA 又有人拿出了这个新东西，这次看起来挺对的。

回顾之前的整套东西。我们说，只要有 $BW^r(s,t)\leq n/r$，再结合一个 $(s,t)$ 的 sandwich，我们就可以先做 $\max(d(s,),-d(,t))$ 的 Johnson Reweighting，此时这些负权边在 $r$ 个 hop 内能负权走到的点只有 $n/r$ 个，然后用经典的 Hop Reducer 做 $O(m(r+k/r))$。

上一个改进是更好地去找 sandwich。而这一个说，我们可以更好地做 Hop Reducer。

原来的版本：我们需要 $BW^r\leq n/r$。

这一个版本：可以只需要 $BW^1\leq n/r,BW^2\leq 2n/r,BW^4\leq 4n/r,\ldots$。


##### Simultaneously Betweenness Reduction

一个 Betweenness Reduction：随机 $r\log n$ 个点，向所有点连双向 $r$-hop$，然后这个图跑最短路。

多个 Betweenness Reduction：每个来这样一组，然后放在同一张图上跑。这样就同时满足所有限制。

每一组是 $r/2^i$ 个点，跑 $2^i$-hop。这一步是 $mr$。然后是左侧 $n$ 个点，右侧 $r$ 个点。这一步直接做是 $nr^2$。通过一些技巧可以 $nr+r^3$，大概是对右边建完全图。但这里右边每个点和左边的边权 hop 数不一样，需要稍微注意一下从小到大分析。

##### Simultaneously Reweighting

并没有理解为什么这是对的。有待继续调查

首先按照惯例，我们需要把所有的 BW 转化为说，这些点在 $a$ 个 Hop 之内，能用负权路径走到的点数只有 $n/b$。

根据之前的分析，有一个的时候，我们用一个 $\max(d(s,),-d(,t))$ 就可以满足要求。

> 回顾证明：考虑现在有 $d^a(s,u)+d^a(u,t)\leq 0$ 的 sandwich，$d^b(s,x)+d^b(x,t)\leq 0$ 的 betweenness，然后我们希望做一步 $\max(d^c(s,),-d^c(,t))$ 解决问题，使得最后 $d^d(U,)\leq 0$（$d$-hop 之内能到的点）很少。
>
> 不妨假设 $a\leq c\leq b$。因为随着 $c$ 增大，$d^c$ 只会减小，那么对于 sandwich 里面的 $u$ 和 BW 外面的 $x$，此时一定前者取到 $-d^c(u,t)$，后者一定取到 $d^c(s,x)$。那么我们希望
> $$d^d(u,x)-d^c(u,t)-d^c(s,x)\geq 0$$
> 然后因为 sandwich，我们有 $d^a(s,u)+d^c(u,t)\leq 0$，因此我们只需要证明
> $$d^d(u,x)+d^a(s,u)-d^c(s,x)\geq 0$$

##### Power Hop Reducers

现在，记 $V_i$ 表示 $2^i$-hop 内存在 $d(,u)\leq 0$ 的点 $u$ 构成的集合。我们假设 $|V_i|=O(2^in/r)$。

我们的目标是，依次对于每个 $V_i$ 内部构造 $2^i$-hop reducer，然后逐步向上。

那么，什么是一个 Hop Reducer？它要求每一个 $h$-hop 的路径都能在上面以 1-hop 表示出来。先前的做法是，我们把图复制 $h$ 份再 reweight，这样复制的一步自然保证了可以表示所有路径。

这里给出了另一种想法：如果我们直接存下所有可能的路径，那就是所有负权边之间的两两 $d^{2^i}$ 距离（$k^2$ 个），我们自然可以得到一个 Hop Reducer：把这个两两距离接到正权边的图上即可。

现在考虑把这个放到更大的图上面。那么在大图之外，我们额外放一个 $V_i$ 的小图，然后从大图往小图放那 $k^2$ 个距离，再从小图跳回去，我们就可以处理在 $V_i$ 内走了 $2^i$ 步的路径。但走一半跳出去了怎么办？那应该把 $V_i\to V\setminus V_i$ 的边也加进去。

但这样的话，可能走了一步就跳出去了，这样相当于一个 hop 就用了一步这里的边，这显然不行。

然后，注意到所有 $\to V\setminus V_i$ 的部分都是走到 $V_i$ 之外，也就是 $2^i$ 步走过去都是正权。那么，如果我们求出的真的是 $d^{2^i}$，那所有这种情况的路径权值总和都是正的。然后有一种方法把这样的东西转化为不需要 hop：把从小图跳回去的边删了，对剩下的跑一个最短路。此时从大图出发回到大图的路径都是正的，所以一次 Dijkstra 就可以求出这里的最短路。这样搞完，图里面的负权边只剩下从小图跳回去的边，而走出去是不走这种边的。这样就在大图上得到了 $2^i$-hop reducer。

一个小问题是，可能我们一不小心求出了一个比 $d^{2^i}$ 更好的东西，这样就不满足那个性质了。但这样只需要限制一下每个 $d(x,y)$。可以发现限制完至少还是 $d^{2^i}$（和之前一样的分析），所以不影响事情。

然后考虑通过这个建造 $2^{i+1}$-hop reducer，此时两个都在 $V_{i+1}$ 上。或者说，我们只需要这个图上的两步 APSP。那么考虑暴力，从 $k$ 个点开始，图的大小看起来是 $k^2+2^im/r$……那最后一层最后一项是不能接受的，$O(mk)$ 还不如暴力。但众所周知这样看是优化不了的，因为这至少是 $(\min,+)$ 矩阵乘法。

实际上也是做不了的，但我们有一些处理方式：考虑只解决 $[2^i,2^{i+1}]$-hop 的路径，这样的话，有一个高明的做法：随机 $O(\log n)\cdot k/2^i$ 个点，那么所有 $n^2$ 条路径大概率都经过了其中的一些点。那么从这些点开始跑两侧路径，再合并。这样这个 $/2^i$ 就会图大小的 $2^i$ 抵消了，每一步复杂度 $k^3+km/r$。

然后考虑怎么回到 $d^{2^i}$。想象直接把上一步的结果带回去，得到的图应该可以处理在 $V_i$ 里面跑了至少 $2^i$ 步的。那如果它跑的步数更少呢？我们就应该用更小的 $i$ 解决问题。因此记录之前每一轮搞出来的 $d$，然后把每一轮建的 $V_i$ 都加进来，这样每一个处理开头 $V_i$ 里面跑了至少 $2^i$ 步的，就总有一个能处理当前路径。

这样总复杂度就是 $k^3+km/r$，再加上 Betweenness 的 $mr$（最后用 $r$-hop reducer 算路径也是 $km/r$）（忽略所有 $\log$）。这里取 $k=n^{1/2}$（这涉及到我没看懂那里的一步问题），然后自然 $r=\sqrt k$。一步复杂度 $n^{1.5}+km/n^{0.25}$，从而复杂度是 $n^2+mn^{3/4}$。

也可以重新调参。如果 $m$ 很小，那重新算一遍，复杂度也可以是 $k^2n+nm/k^{1/2}$。这样看可以得到 $m^{4/5}n$，取 $k=m^{2/5}$。这样相当于 $m^{4/5}n$。