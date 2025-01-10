---
title: '[Talk] Approx 01Knapsack Counting'
date: '2024-11-23 12:18:44'
updated: '2024-11-23 12:18:44'
tags: Fancia
permalink: Tennaki/
description: Approx 01Knapsack Counting
mathjax: true
---

### Approx #-01-Knapsack

$n$, value $t$, error $1\pm \epsilon$. Consider $\epsilon$ not too small(1/polylog)

Classical technique: approx counting -> sampling, use MCMC to approx sampling.

Mixing time for this truncated hypercube: $O(n^{4.5+o(1)}),\Omega(n^2\log)$,implies poly, but not to fast

Dyer STOC03: Approx by randomized rounding, DP, estimation.



#### Dyer03

Classical  Rounding: $w\to w/(t/poly(n))$.

Randomized Rounding: $S:=t/poly(n)$, $w\to \{\lfloor w/S\rfloor,\lceil w/S\rceil\}$, pr proportional to distance between these three value($E(w)=w/S$)

which let some combination smaller/larger, so let $t+=something$



Approx on weight does not directly mean approx counting.

1. If original is a solution, now is not.

By that rounding and adjusting t, only $\epsilon$ fraction is lost.

2. If original is not, now is

Thm. #now solutions is at most $O(n)$ times original.

If these are true, approx $old/new$ directly, Chernoff things gives $\tilde O(n/\epsilon^2)$ sample count.



With these: DP+ counting DP($\tilde O(n^{2.5})$, approx counting using few bits), then easy sampling



Why?

1. By Hoeffding, error of a subset <= $O(S\sqrt{n\log n})$ whp.  Let $t'=t+O(S\sqrt{n\log n})$



2. $O(n)$ times #sol only needs $S=t/\tilde O(n^{1.5})$.

Why? this makes $t'=S/\tilde O(n)$, so remove largest element->valid instance, count this mapping.



#### Feng-Jin25

##### Better DP

Initial Idea: D&C+FFT, $\tilde O(\sum w)$

Problem of rounding in FFT: Each term approx by $a\times 2^b$, a convolution on this is $(\max,+)$ on $b$, so not OK.

Strong (\max,+) conjecture.



Bounded Monotone (max,+): Can be $\tilde O(n\sqrt M)$

Store Approx version of prefix sum(hard details), then monotone max+ with witness count.(take some ln)



Bounded Ratio case: $w\in [t/l,2t/l]$

Split into $l$ bins, for each bin, there are $\tilde O(1)$ elements chosen whp.

With this, easy D&C: $len_k =\tilde O(t/k)/s*\sqrt{l/k}$, $k$ is number of nodes in this layer.



Different scaling: At higher layer, larger S.

Recall: Original Proof:

If Original size is large, Hoeffding bound rounding error.

If Original size is small(t+t/n), remove the largest element in $S$ to get a valid solution

Which needs $S\sqrt{n\log n}\leq t/n$. How to do layered case?

First, bounded -> $S\sqrt l$, Threshold can also be $t/l$. Therefore $t/S=l^{1.5}$.

Each layer, $k$ roundings, so can check $t/S=l\sqrt k$. Then total: $l^{1.5}$



Sample time: $l^{1.5}$ (We need to invert single term of converlution, which needs $O(size)$ time since we cannot store all s^2 terms)

Need to probe: only need $n/l$ samples.

Use a $\tilde O(n/l)$-Hitting set to hit all $l$-size set.