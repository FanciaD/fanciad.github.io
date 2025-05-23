---
title: '[paper] Parallel DFS'
date: '2024-12-04 21:11:06'
updated: '2024-12-04 21:11:06'
tags: Fancia
permalink: KaiyuugyonoGenfuukei/
description: Parallel DFS
mathjax: true
---


### Parallel DFS

考虑带 visit 的 dfs。有两种实现：

1. 出边有序，严格按照顺序进行。
2. 只需要找到一个能被 dfs 出来的序列。

#### Ordered DFS is P-complete

很遗憾的是，第一个是 P-complete 的，所以第一个是 P in? NC 的超级 Open Problem。

众所周知的 P-complete 问题是 Circuit Eval。

首先，我们需要一个可以变化的局部 dfs，那就有：

![124-1](\pic\124-1.png)

然后考虑怎么用，这里唯一区别是我们经过边的顺序不一样，那可以搞一个这样的东西：

![124-2](\pic\124-2.png)

那么，如果上面关键点被走了，它就会无事发生，否则它就会去走接下来的若干个关键点（可以通过调整顺序，使得后面即使是关键点，也会先走回来的路径，这是最关键的区别）



容易发现放两个关键点也一样，那考虑关键点被走表示 $0$，否则表示 $1$。那如果每个这样的东西的输入都是 $1$，它就会输出一堆 $1$，否则它输出 $0$。那我们就造了 AND Gate。容易发现换成 NAND Gate 就可以表示任意 Circuit 了。



定义：Depth 是最大深度，Work 是总操作次数。

#### polylog depth, poly work algorithm

众所周知，压深度只能考虑分治。

最直接的，如果你每次能找一条路径把图切成两半，然后分别递归就是 log depth。但显然这不大现实。

首先假装我们能 polylog depth 求 01 最大权匹配，这大概类似于代计的做法。

##### Separator Paths

我们的第一个目标是，在 polylog depth 里面找到 $O(1)$ 条路径，它们把图分成若干块，每块大小不超过一半。

有了这个之后可以这样做：从起点开始，每次找一条路径到上面选择路径的某个点，然后沿着路径大的方向走。这样一次一条路径长度减半，最多 $O(\log n)$ 步。如果走不动了（当前连通块里面没有别的路径），那递归求解当前连通块，然后往上看（找到上面第一个可以走到的点）。直到把路径搞完。

判定/构造一条走到给定路径上点的路径可以看成判断 $s$ 到 $t$ 的路径，可以构造匹配：每个点拆成 in/out，删掉 $in_s,out_t$，然后每个点可以不匹配（$in_s\to out_s$），也可以沿着路径匹配。



##### Construction of Separator Paths

初始有 $n$ 条路径，每次合并路径，让路径数量变为之前的 $1-\Omega(1)$ 倍。



将路径分为两组，我们在两组路径之间找一个路径匹配：每条路径连接左边的一条和右边的一条，然后这样干

![124-3](\pic\124-3.png)

虽然这样不减少路径数，但我们可以选择右边走更小的一侧（显然这里右边走哪边是一样的），从而如果我们固定右边是“短的路径”，那它一次长度减半，最后总能删空。

（找哪边更短是可以做的：把 prev 和 next 规定好，然后一边倍增 $next^{2^i}$，一边跳 $f_{u,i}=f_{u,i-1}\lor f_{next(u,2^{i-1}),i-1}$）



但为什么我们可以把左边去掉？首先考虑：在所有最大匹配中，找一个删掉的最少的方案，这相当于带权版本的最大匹配。此时不会有这种情况：![124-4](\pic\124-4.png)

那就是说，如果出现了坏的合并，这里一定不会涉及到右边没有匹配的路径。

而如果有这样一个坏的情况，右边没有匹配的路径都没用了，因为它们加起来还不到一半。因此可以这样：

1. 操作完（删掉最少的方案）后，统计每个连通块的大小，这可以邻接矩阵上 pow 几下解决。
2. 如果合法，直接继续。
3. 否则，移除右边没有匹配的路径。

因此考虑左边放 $1/3$，右边放 $2/3$，这样最后一种情况就能处理很多路径。



但如果匹配很少，比如小于左边的 $1/10$ 怎么办？此时没有匹配的任何一个左边路径和没有匹配的任何一个右边路径都是无关的。那么我们总可以选一个删掉：一边炸了那另一边一定对。



这样要么我们删掉了 $O(1)$ 比例的路径，要么右边总路径长度减少了 $O(1)$ 比例。



#### sqrt depth, near linear work algorithm

只能做**无向**图

和上一个的框架类似，但有一堆区别：

##### Choosing Separator Paths

首先还是抄上面的缩路径框架。

唯一的区别：我们现在允许一些 $L$ 的路径从中间出去（即删掉后面的部分），但不匹配到最后的 $S$ 部分。只要这个比例是更小的常数倍。

然后这样搞：把当前所有 $L$ 的路径看成一个 dfs，当前栈表示路径，然后并行跑多个 DFS。

问题是并行跑 dfs 的时候，每个点需要扩展出边，需要处理这个冲突。



Lemma 1. 最大匹配可以 npolylog work, polylog depth.

但这还不够：可能我们现在把所有出边拿出来跑了一次，选了一条走过去，但等会又回溯回来了，这样复杂度就炸了。

准确的说，我们需要保证，我们考虑的边数是被其它点用掉的点数量级的，而这在串行算法里面是自然成立的。

那这样做：每个点先拿 $1$ 条出边出来做匹配，如果不行再拿下 $2$ 条边出来做匹配，然后 $4$ 条，以此类推。如果一次没匹配上，那对应的出边就没了。

然后需要支持：删元素，查询元素，可以来一个线段树，这自然是并行的。



然后复杂度是啥？每轮每个 dfs 消耗半个点（每个点进入一次离开一次），但如果 dfs 点数很小就不行。

因此我们选 $O(\sqrt n)$ 条路径，剩下这么多个 dfs 的时候就停止。这样最后就会剩下一些停在那的路径，但它们不那么影响事情。但因为常数比例的限制，我们只能做到剩 $O(\sqrt n)$ 条路径。



##### Combining Separator Paths

考虑套之前的东西，我们需要快速找到一条路径，然后删掉这条路径，还需要树上回溯。

维护动态图连通性，里面上 Top Tree。每一步都可以是高度并行的。 Work 显然是 n polylog。