---
title: '[talk] Another Proof of Chernoff Bound'
date: '2025-04-22 12:23:34'
updated: '2025-04-22 12:23:34'
tags: Fancia
permalink: Tyyneys/
description: Another Proof of Chernoff Bound
mathjax: true
---

Princeton Theory Lunch 4/18>

非常简洁的 Talk，以至于我现在还记得到。Online List Labeling 我就快忘完了。

如果你做 Theory，那你可能经常算概率，然后就会天天用 Chernoff Bound：

一个简单形式：$n$ 步操作，每次均匀随机 $+1/-1$，那么总和大于 $k\sqrt n$ 的概率不超过 $2^{-\Omega(k^2)}$。

#### The Classical Proof

虽然这里不讲这个，但我还是快速提一遍。

先上一个 Exp，把 $\Pr[\sum X_i\geq s]$ 变成 $\Pr[e^{\sum X_i}\geq e^s]$。然后用 Markov，bound $e^{(\sum X_i)-s}$。接下来，根据凸性与琴生不等式，$E[e^{\sum X_i}]=\prod e^{E[X_i]}\leq \prod E[e^X_i]$。然后单独处理右边每一项，好像可以展开 exp。

为了得到原来那个 $e^\delta/(1+\delta)^{1+\delta}$，我们需要仔细考虑第一步加入 exp 的底数，然后简单分析一下。

这样推出来形式非常抽象，第一眼看过去根本看不出上面那种简单的界。而且，这东西紧吗？直接看这个好像看不出来。

这里给一个不需要代数的证明，我们只需要下面这个东西：

#### Chebyshev Inequality

考虑分析 $E[(\sum X^2)]=n$。那么首先根据 Markov，我们可以得到：

1. $n$ 步之后，和大于 $2\sqrt n$ 的概率不超过 $1/2$。

但还可以更进一步：

2. $n$ 步**之中**，和大于过 $2\sqrt n$ 的概率不超过 $1/2$。

为什么？严谨的 Martingale 我不会。可以想象如果 $\geq 2\sqrt n$ 就停，如果停了就会贡献 $4n$ 的方差，所以停的概率是 $1/4$。

#### Naive Bound

根据上面的结论，$n$ 步之后和只有一半的概率大于过 $2\sqrt n$。

如果大于过，那么从那个点开始往后看，剩下的步里面只有不到一半的概率和能再加 $2\sqrt n$。

如此往复，和能大于 $k\sqrt n$ 的概率就至多是 $2^{-\Omega(k)}$。

#### Split and Refine

考虑将序列分成 $B$ 块，那么根据之前的结论，每一块大于 $k\sqrt{n/B}$ 的概率是 $2^{-\Omega(k)}$。

##### The Coin Flip Problem

抽象一下这个问题：有 $n$ 个变量，每个变量大于 $k$ 的概率是 $2^{-k}$（常数省略）。那这东西会怎么分布？

每个变量可以想象成一枚硬币，每次扔一下，正面就继续，否则终止。我们考虑 $n$ 枚硬币加起来能扔多少次。

期望是 $n$ 次，我们来计算超过 $2n$ 次的概率。那么把这 $3n$ 次（包含反面的）的结果写下来，总共有 $\binom{3n} n\approx 3^n$ 种情况，每种发生的概率是 $2^{-2n}=1/4^n$。因此这样的概率是 $2^{-\Omega(n)}$ 的。



那么回到原问题，这么多块加起来大于 $O(B)\sqrt{n/B}$ 的概率就是 $2^{-\Omega(B)}$ 的。可以发现，让 $B=k^2$，然后就结束了。

#### The Lower Bound

首先考虑证明 $k=1$，此时有：

1. 和大于 $0.1\sqrt n$ 的概率至少是 $0.1$。

我会爆算，别的暂时还不会。

仍然考虑分 $k^2$ 段。那么每一段都多出来 $0.1\sqrt{n/k^2}$ 的概率是 $2^{-\Theta(k^2)}$。但这样总的就多出来 $0.1k\sqrt n$。