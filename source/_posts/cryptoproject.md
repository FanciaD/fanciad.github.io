---
title: '[paper] Kolmogorov Complexity and Key Agreement'
date: '2024-11-24 23:43:57'
updated: '2024-11-24 23:43:57'
tags: Fancia
permalink: HoshimatsurinoYoruni/
description: Kolmogorov Complexity and Key Agreement
mathjax: true
---

### Crypto Project

每个 $c$ 表示不同常数。

#### about Kolmogorov Complexity

$K$：常规定义

$K(a|b)$：可以免费拿到 $b$。

$K^t$：限制运行时间是 $t$。这里全部取 $t=poly(n)$，同时 $t>n^{1+\epsilon}$（我猜是因为 Simulation overhead $\log n$）

$IK$：维护两个程序跑交互，长度总和的 $\min$。

$IK^t$：字面意思

显然 $K\leq IK+c\log n$，差距是因为定义问题（字符串长度永远是一个信息！）

$RIK^tP$: Promise problem:

1. Yes 表示 $IK^t(\pi,x,y)$ 比 $K(\pi)$ 大的不多（$\leq \sigma_Y$）
2. No 表示**即使是** $K(\pi,x,y)$ 也比 $K(\pi)$ 大很多（$\geq \sigma_N$）

这记作 $RIK^tP[\sigma_Y,\sigma_N]$。

额外的，再定义一个 Promise $Q_\Delta$，表示 $K^t(x|y),K^t(y|x)$ 都不超过 $\Delta$。

$K$ 是不可计算的，但 $K^t$ 可以计算。



一些小结论：

1. $K(x,y)\leq K(x)+K(y|x)+c\log n$，因为可以拼接程序。这对 $^t$ 差不多对。
2. 如果有一个随机化程序 $D(x)$，它输出 $y$ 的概率至少是 $\Omega(1)$（比如 $1/3$），那么 $K(x,y)\leq K(x)+O(|D|)$：考虑计算所有 $D(x)$ 输出概率 $\Omega(1)$ 的元素，这只有 $O(1)$ 个，因此再记录一个下标即可。注意这就不能 $^t$ 了。
3. 如果一个随机化程序额外接受一些 bit，例如 $D(x,z)$，其中 $z$ 通过一些方式觉得，此时如果输出 $y$ 的概率是 $\Omega(1)$，则 $K(x,y)\leq K(x)+O(|D|)+|z|+c\log|z|$：直观上想，把之前的数数改成存在 $\Omega(1)$ 比例的情况能找到 $z$，那么多出 $2^{|z|}$ 倍。后面一项是因为拼接字符串。



#### Example: OWF from Kt average hardness

考虑这样一个函数：$f(x,y)$ 等于把 $y_{[1;x]}$ 当成图灵机跑 $t$ 步的结果，前面拼接上 $x$。

如果我们可以 Invert $f$ with Pr $1$，那我们可以拿到一个输出之后枚举 $x$ 去 invert 以得到答案。

那直观上看，如果 $K^t$ 是 Average 困难的（有 $1/poly$ 解不出来），那 $f$ 也应该有 $1/poly$ 不能 Invert。具体的算还略微复杂，因为 OWF 的 Invert 比例是对输入算的，但可以注意到一个 $z$ 总能找到一个直接输出它的程序（和一个长度），所以差距只有 $O(n)$ 倍。

还有一个问题是 Invert 可以是随机的，所以概率上做点 Markov。

那这得到了一个 Weak OWF（$1-1/poly$）。然后根据 Yao 的经典结果，可以得到 Strong OWF。



另一方面，如果 OWF 存在，那感觉上讲，我们存在一个 PRG，但 PRG 如果 Extend 了 $c\log n$ 位，那这些输出的 $K^t$ 都比随机的 $K^t$ 少一些 $c\log n$，那如果能算 $K^t$ 就能分辨。

问题是 PRG 的输出可能很少，然后全部撞到 $K^t$ 的 Average-case solution 解不出来的少部分里面。所以这里需要定义一个 EP-PRG 使得输出的熵差不多是 $n$（减去若干倍 $\log n$）

大概构造是说，本来的构造是 $f(x)|\langle x,r\rangle$（当然，实际上我们能 Append $O(\log n)$ 个内积），但如果有人原像特别多就寄了。

如果每个人的原像大小差不多（可以都很大），那我们可以加点类似 Quotienting 的感觉：找一个输出位数正确的 Hash function，然后把 $f(x)$ 映射到差不多表示输出个数的若干位上，把 $x$ 映射到差不多表示原像个数的若干位上。

对于一般情况，实际上我们可以限制 PRG 的输入是 $U_n$ 的一个子集，这被称为 condPRG。这个子集可能难以计算，但 OWF 显然 imply 它，且这样还能被 $K^t$ 分辨。然后 cond 原像大小在 $[2^i,2^{i+1}]$ 里面的，总有一次减少的不多。



#### KA from RIKtP|Qd

规定参数范围：$RIK^tP[c_1\log n,c_2\log n]|Q_{d\log n}$。此时要求 $c_2-c_1>O(d)$。


回顾 KA 的要求：两个人交互，然后分别得到 $x,y$。要求：

1. 有 $\alpha$ 的概率两个人得到相同的结果。
2. 对于任意一个 ppt adversary，它通过 $\pi$ 拿到 $x$ 的概率小于 $\delta$。

类似 OWF 的情况，我们也有一个 Amplification:

Thm.[“Strengthening key agreement using hard-core sets”] 如果我们能做 $\alpha=1-n^{-a},\delta=1-n^{-b}$ 且 $a>b$ 的 1-bit KA，我们就能做非常强的（$\alpha=1-negl,\delta=negl$）KA。

Thm. 多 bit KA 可以转换为 1bit KA。

Proof. 直接 Goldreich-Levin，随一个 $r$，然后两边内积起来。



Protocol 长这样：

1. A,B 分别随机一个 $[n]$ 中的长度，然后分别随机一个程序 $P_A,P_B$。
2. 两个程序模拟交互若干步，得到过程 $\pi$ 和输出 $x,y$。
3. A 随机一个 Hash function，把 $(x,h(x))$ 发过去。
4. B 随机一个 $[\log n]$ 的长度，随机一个程序 $P_c$，用这个程序过一遍 $y$ 得到 $y'$，然后检查是否 $h(x)=h(y')$。如果是，那么双方拿 $(x,y')$ 输出。否则输出 $\perp$。

我们 Hash 到 $c\log n$ 个 bit，这样碰撞概率就是 $n^{-c}$。



这何德何能对？一眼看过去，我们希望两个程序跑出来结果一样，否则输出是 $\perp$。但这个事好像没有很大的概率。但如果是这样的话……

1. 每个 $IK^t=l$ 且 $Q_{\Delta}$ 的东西有 $2^{-l-(3+d)\log n}$ 的概率被生成：我们总能枚举到正确的长度，前两个程序是 $2^{-l}$，最后一个因为 $Q_{\Delta}$ 是 $2^{-\Delta}=2^{-d\log n}$。
2. 如果对于一个 $l$，这样的东西数量很少，比 $2^{l-(c_1+3)\log n}$ 都少，那它们全部不是 Yes Instance：这样的话我们直接得到了 $K\leq l-(c_1+3)\log n+c\log n$，那么 $IK^t$ 就比 $K$ 大了超过 $c_1\log n+O(1)$：我们可以枚举所有程序一起跑决定 $IK^t$。
3. 每个 $(\pi,x,y)$ 生成概率至少是 $2^{-l-(3+d)\log n}$，数量至少是 $2^{l-(c_1+3)\log n}$，那不输出 $\perp$ 的概率就至少是 $n^{-(d+c_1+O(1))}$。只要你的 $c$ 比 $d+c_1$ 大一些，那 Adv 输出 $0$ 成功的概率就不够大。

上面的事情说明，如果输出 $0$ 的 Adv 成功概率大于啥 $1-n^{-c+1}$，那对于每个 $l$，输出非 $0$ 的情况中就没有任何一个 Yes Instance："IKt 比 K 大的不多"的东西不能太少，不然描述这个就有很小的 K。

$\pi$ 里面包含了最后验证是否成功，因此我们只需要再看相等的情况 Adv 干了啥。考虑 Adv 在相等的时候**没有**以 $1/3$ 概率找到 key 的那些输入。注意到这个条件也是可以被描述的，那我们考虑这样一个集合：

> $(\pi,x,y)$ 满足 $IK^t=l$ 且 $Q_{\Delta}$，同时 Adv 在拿到 $(\pi,h,h(x),yes)$ 的情况下有 $1/3$ 概率不输出 Key。（Over 随机 $h$）

类似上面的分析：如果这个集合（对于某个 $l$）很小，那这个集合都可以被更小的 $K$ 描述，从而不是 Yes Instance。否则，每个元素出现概率也很大，从而它们贡献了一个不小的 Adv 失败概率。这个概率至少是 $n^{-c+1}$。

因此我们得到结论：如果 Adv 失败概率小于 $n^{-c+1}$，那它对于任何一个 Yes Instance 都能 $2/3$ 概率找到 Key，因为找不到的都能被“找不到”这件事很短地描述。



然后再看 No Instance 部分。如果有一个 Adv 能对 $(\pi,h,h(x),yes)$ 常数概率（Over $h$ 和自己随机）找到 $x$，那直观上看，$h$ 是随机的，$h(x)$ 是 $c\log n$ 个不知道什么 bit，然后输出正确的概率是 $1/3$。

如果没有那个 $h(x)$，那直接 $K(\pi,x)\leq K(x)+O(1)$ 了：和之前说过的一样，考虑输出概率 $\geq 1/3$ 的那些元素。如果有一个 $c\log n$ 长度的东西，那元素多 $n^c$ 倍，可以得到 $\leq +c\log n+O(\log(c\log n))$。这样的话：
$$
K(\pi,x,y)\leq K(\pi,x)+K(y|x)\\\leq K(x)+c\log n+O(\log(c\log n))+d\log n+|Adv|\\
\leq K(x)+(c+d+O(1))\log n
$$
快速回顾，如果 $c+d+O(1)<c_2$，那这就不是 No Instance。所以我们需要 $c_1+d+O(1)\leq c\leq c_2-d-O(1)$，这就是为啥是那个限制。



Recap：

1. 做 RIKP 的时候，随机一个 $h$ 然后把造出来的 $(\pi,h,h(x),yes)$ 给 Adv。
2. 无论如何，只要 Adv 有很大概率输出 Yes，那我们都能通过 $\pi$ 加上 $|h|$ 个 bit 推出 $x$（因为很容易输出 $x$），然后用 $Q_\Delta$ 推出 $y$。这样，$K(\pi,x,y)$ 不比 $K(\pi)$ 大太多。
3. 如果 Adv 成功率很大，那一个存在 Key 但没有得到正确输出的 $\pi$ 可以直接用这句话描述。这样的集合大小是错误率除以每个元素被采样的概率。固定一个 IKt，则每个元素被采样的概率基本只和 IKt 相关（因为可以这样枚举出程序）。那么错误率比较小的时候，元素个数不比 $2^{IK^t}$ 多多少，这样 $K(\pi)$ 就可以直接通过 $IK^t$ 左右个 bit 描述：记录它在“算不出来但 IKt 很小”集合中的位置。



#### RIKtP|Qd from KA

回顾一下刚才的 $RIK^tP[c_1\log n,c_2\log n]|Q_{d\log n}$，此时 $c_2-c_1\geq 2d+O(1)$。直观上想，如果我们都能做 $RIK^tP$ 了：

1. 对于一个 KA 给出来的 $\pi$，它加上正确的 $x$ 得到的 $(\pi,x,x)$ 的 $IK^t$ 根据定义是小的，小于 $A,B$ 用到的随机 bit 数加上 $c\log n+O(1)$。
2. 如果我们转而加上一个随机的 $y$，那 $(\pi,y,y)$ 的 $K$ 几乎必然很大，是 $K(\pi)+|y|-O(1)$。考虑让 $|y|=c'\log n$。
3. 我们还需要说明，$K(\pi)$ 差不多是 $A,B$ 用到的随机 bit 数那么多。一个问题是：和之前一样，$\pi$ 的种类数可能非常小。因此这里和之前一样，定义一个 (cond)EP-KA：在某种情况下，$\pi$ 的 Entropy 和随机 bit 数只差 $c''\log n$。这样就可以说明：正确的 $x$ 大概率属于 Yes Instance，随机的 $x$ 大概率属于 No Instance。

但这还不是 KA 的定义：Distinguish Key vs Recover Key. 所以我们再来一个 Goldreich-Levin：能 predict 最后那些内积就能还原 $k$。

那么我们需要最后一步：有一个 KA，然后我们希望把它变成（Condition 某个 Event 下的）EP-KA。

算不明白，先🕊了。



#### OWF from RIKtP

和之前的区别是，我们要的是一个 Worse-case Hardness，所以需要如果能 Invert 这个 OWF，那就能完全正确。

回忆一下在 KA 里面我们如何解决 RIKtP 的：

1. 如果 Adv 对于输入能高概率还原 $x$，那加上 $x$ 后 $K$ 增大的不多，因为可以用 Adv 去生成。
2. 如果 Adv 成功率很高，但对于一些东西不能还原，同时每个东西出现概率都不低（$\exp(-IK)$），那“不能还原的东西”很少（$\exp(IK)$），这个集合可以用 $IK$ 级别的量描述 $K$。

放在这里看看，第二条看起来是不变的，这里概率是 $2^{-l}n^{-2}$。

第一条有个问题，这里需要输入 $\pi$ 还原 $(x,y)$，但如果按照之前的构造，OWF 输出两个程序 $P_A,P_B$，然后输出模拟结果 $(\pi,x,y)$，那就出问题了：信息都写在输入里面了，不能造出 Gap。如果只输出 $\pi$，那第二条里面的 $(\pi,x,y)$ 就不能定义了，这不行。

一个折中的方案是，搞一个 Hash Function，输出 $H(x,y)$。这样第一步增大的东西不超过 $H$ 的位数加上 $c\log n$，那只需要 $c_2$ 大点。对于第二步，如果 $IK^t(\pi,x,y)$ 比 $K(\pi)$ 大不超过 $c_1\log n$，那直观上 $(x,y)$ 的组数应该接近 $2^{c_1\log n}$，实际上还要再乘一些 $2^{\log n}$：大概是说，总共的程序数量是有限的，那么能造出很多组 $IK^t$ 小的 $(x,y)$ 的 $\pi$ 数量有限，那么用这个集合描述。因为 Yes Instance 的数量只和 $2^{c_1\log n}$ 差不多，那 Hash Function 到 $(c_1+c)\log n$ 位就没啥碰撞了。



#### RIKtP from OWF

类似之前的想法，先搓一个 condEP-PRG，此时随机串 K^t 很大，生成的串 K^t 相对小一点。

然后换成 IK：记 PRG 是 $\{0,1\}^n\to \{0,1\}^{n+c\log n}$，那交互前 $n$ 位，然后自己输出最后若干位，此时：

1. 根据 condEP 的构造，前 $n$ 位很随机，所以 $K$ 可以接近 $n-c’\log n$。
2. 如果是 PRG 的输出，那 IK^t 差不多是 $n$
3. 否则，IK^t 差不多是 $n+c\log n$

因此 OWF 存在可以推出 $RIK^tP$ 的 Hardness。进一步显然可以取 $\Delta=(c+O(1))\log n$。此时相当于 $c_2<d$ 情况的 Hardness。

最后我们发现，这个问题在两种不同情况下的 Hardness 分别等价于 KA 和 PRG。