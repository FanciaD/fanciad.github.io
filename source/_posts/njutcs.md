---
title: NJU 计算理论之美 夏令营 笔记
date: '2024-07-16 09:26:50'
updated: '2024-07-16 09:26:50'
tags: Fancia
permalink: SailingHope/
description: NJU 计算理论之美 夏令营
mathjax: true
---

## Boolean Function Analysis

Fourier Analysis for ~~$\{0,1\}^n$~~ $\{-1,1\}^n$

Inner Product: $<f,g>=E_x f(x)g(x)$

The basis: For each $S$, $\chi_S(x)=\prod_{i\in S}x_i$. dim $2^n$

Fourier: $f=\sum_S \hat f(S)\chi_S$

$\E(f)=\hat f(\emptyset)$, $<f,f>=\sum \hat f(S)^2$. If $f: \{-1,1\}^n\mapsto \{-1,1\}$, $<f,f>=1\implies \sum \hat f(S)^2=1$

$Var=\E(f^2)-\E(f)^2=\sum_{S\neq \emptyset}f(S)^2$

Convolution: $f*g(x)=E_yf(y)g(x-y)$ $(0/1),0=1,1=-1$

Classic Results: $\hat{f*g}(S)=f(S)g(S)$

#### Linearity Test

$f:\{0,1\}^n\mapsto \{0,1\}$, test $f(x+y)\equiv f(x)+f(y)$.

As $0=1,1=-1$: $f:\{-1,1\}^n\mapsto \{-1,1\}$, test $f(xy)\equiv f(x)f(y)$.

$\chi_S$ are the only basis functions.

Test: Linear function or $\epsilon$-distance ($\%$ of different variables) from any linear.

$<f,g>=1-2dis(f,g)$.

Straightforward Test: sample $x,y$, check $f(xy)=f(x)f(y)$.

Thm. If $f$ is accepted with pr $1-\epsilon$, then $f$ is $\epsilon$-close to some linear.

Prf. $Pr[correct]=E_{x,y}(1/2+1/2f(x)f(y)f(xy))$(-1/1) $=E(1/2+1/2(f*f)f)=1/2+1/2<f*f,f>=1/2+1/2\sum \hat f(S)^3$

$1-2\epsilon\leq \sum \hat f(S)^3\leq \max f(S)(\sum f^2)=\max f(S)$

### More definitions

polynomial method: $\chi_S=\prod_{i\in S}x_i$. $deg=\max_{\hat f(S)\neq 0}|S|$

Influence: $Inf_i=\Pr_x[f(x)\neq f(x^{\oplus i})]$

$Inf_i=1/2-1/2<f,f^{\oplus i}>=1/2-1/2(\sum_{i\not\in S}\hat f^2(S)-\sum_{i\in S}\hat f^2(s))=\sum_{i\in S}\hat f(S)^2$.

Total influence $Inf(f)=\sum Inf_i=\sum \hat f(S)^2|S|$

Noise operator $T_p$. $\rho^x$: For each bit, $p$ probability keep $x$, $1-p$ probability random.

$T_p f(x)=E_{y\sim \rho^x}f(y)$

Linearity, on basis: $T_p \chi_S(x)=?$ If some bit in $S$ become random, the expectation is $0$. So the answer is $\rho^{|S|}$.

$f=\sum \hat f(S)\chi_S\to \sum \rho^{|S|}\hat f(S)\chi_S$

$p$-norms: $E_x [|f(x)|^p]^{1/p}$. $p<q, \|f\|_p\leq \|f\|_q$.

$\|T_\rho(f)\|_p=E_x[E_y|f(y)|^p]^{1/p}\leq E_x[(E_y|f(y)|^p)^{1/p}]=\|f\|_p$

$\|T_0(f)\|_p=\|f\|_1$

### Hypercontractivity 

give $p,q$, find $\rho$ that $\|T_{\rho}(f)\|_q\leq \|f\|_p$?

$[Bonami70]$: $\rho\leq\sqrt{(p-1)/(q-1)}$

If Boolean $f$ has $deg\leq k$, then $\|f\|_q\leq (q-1)^{k/2}\|f\|_2$

Let $T_{\rho}^{-1}: \sum \hat f(S)\chi_S\to \sum \rho^{-|S|}\hat f(S)\chi_S$

$\|f\|_q=\|T_{\rho}T_{\rho}^{-1}f\|_q=(p=2,\rho=\sqrt{1/(q-1)})\leq \|T_{\rho}^{-1}f\|_2\leq(expand) (q-1)^{k/2}\cdots$

$(2,p)$-Hypercontractivity:

$n=1$: $\|a+\rho bx\|_2\keq \|a+bx\|_p$ for $\rho<\sqrt{p-1}$. $x\sim\{-1,1\}$.

Prf. Let $a=1,|b|<1$. Calculate!(Binomial Expansion)

$(2,p)->(q,2)$: Holder's Inequality

Then induction on $n-1$ dim, then use $n=1$.

### Central Limit theorem

Sum of independent random variables converges to Gaussians $\mathcal N(\mu,\sigma
^2)$

$X_i$ iid, $\mu,\sigma^2$, then $\sqrt n(\frac1n(\sum X_i)-\mu)\to N(0,\sigma^2)$, independent of $X_i$'s distribution.

For $[-1,1]$:

$X_1+X_2+\cdots+X_n$($\{-1,1\}$) vs. $g_1+g_2+\cdots+g_n$(gaussian)

Berry-Esseen Theorem: let $x\sim \{-1,1\}$, the central limit theorem converges with speed $c/\sqrt n$

Hybrid argument. X1+...+Xi+Yi+1+...+Yn

Apply Taylor.

### Invariance Principle

Poly $p$, test function $\psi$, how to let $E_{x\sim \{-1,1\}}(\psi(p(x_1,\cdots,x_n)))=E_{g\sim N}(\psi(p(g_1,\cdots,g_n)))$?

1. low influence(if concentrace on 1st element?)
2. low degree(if $\prod(1+x_i)$?)

Mossel-O'Donnell-Oleszkiewicz: $$|E_{x\sim \{-1,1\}}(\psi(p(x_1,\cdots,x_n)))-E_{g\sim N}(\psi(p(g_1,\cdots,g_n)))|\leq \|\psi^{(3)}\|_{\infty}2^{O(deg(p))}*\max Inf_i(p)$$

For $\psi=[x>c]$, can take $O(d*\max inf^{1/(4d+1)})$.

Hybrid, separate polynomial, taylor to order 3, hypercontractive

### PRG and fooling halfspace game

PRG against a class of poly-time functions $C$.(Other part follows normal definition)

Fooling $C$=half space $ax\leq b$ (GKM15: $log(n/\epsilon)\log\log(n/\epsilon)^2$)

$C$=polytope: AND of $m$ inequalities. (OST18: $\log n*poly(1/\epsilon,\log m)$)

Fooling $\sum x_i\leq b$(Meka-Zukerman)

k-wise independent function: [Naor-Naor] $O(k+\log m)$

Construction: Split input to $\epsilon^2$ part, then $4$-wise independent function. Length= $O(\log m/\epsilon^2)$

Apply the proof in Berry-Esseen. First 3 order are independent, so it applies.

#### PRG and counting

Count $\sum_{x\in \{-1,1\}} w_ix_i\leq b$ solutions? SharpP-Complete.

1. find a good generator against linear classifiers
2. then count on the generator. $O(2^rpoly(r))$

So GKM 15 gives $O(n^{\log\log n/\epsilon})$

1. give a smooth approximation
2. invariance principle using Berry-Esseen
3. Anti-concentrate against smoothing(Hard)
4. low-order

<Analysis of Boolean Functions>

## Differential Privacy

Def: Changing one entry does not change output much

E.g. Query-answer, ...

Def. $(\epsilon,\delta)$-DP: For two neighboring(differs by 1 row, ...) dataset $X,X'$, for any $S$, $\Pr[M(X)\in S]\leq e^{\epsilon}\Pr[M(X')\in S]+\delta$

Add/Remove-DP(addrem 1row) / Substitution-DP(change 1row)

In substitutionDP, data size can be revealed.

Def. Privacy loss $L(m,x,x')(o)=ln(Pr[M(X)=o]/Pr[M(x')=o])$. Clearly $(\epsilon,0)$-DP iff. $\forall, L\leq \epsilon$.(discrete case)

### Noise Addition

Output $g(x)+$ random noise(Gaussian)

Discrete Sensitivity: $\max|g(X)-g(X')|$

Discrete Laplace Mechanism: $DLap(b)(i)\sim e^{-|i|/b}$, Add $DLap(\Delta/\epsilon)$ gives $\epsilon$-DP(simple)

Measure $\sqrt$ MSE. $RMSE(DLap)=O(\Delta/\epsilon)$

Multi-dim discrete: Lp sensitivity. Add $L1$ sensitivity to each dim.

Continous Case

Continous Privacy loss: use PDF.

Continous Laplace: $Lap(\Delta/\epsilon)$, similarly.

Similarly, multi-dim.

Lap: exp(-|x|)

Gaussian: exp(-x^2)

Gaussian Mechanism: Add Gaussian

Privacy Loss Distribution: The dist. of $L_{M,x,x'}(o)$ that $o\sim M(x)$.

Approximate DP condition: If $Pr[L_{o\sim PLD}>\epsilon]<\delta$, then it is $(\epsilon,\delta)$-DP

Analysis of Gaussian Mechanism: $L=...=(a-x)^2-(b-x)^2$ goes to infinity. But for a large range, it is small. The above condition gives $(\epsilon,\delta)$. It can be shown that $\sigma=2\epsilon^{-1}\sqrt{2\ln(2/\delta)}$ is OK.

Multi-dim Gaussian: use L2 sensitivity.

### Properties

Post-Processing

Query composition?

Basic Composition Theorem: The combined output remains $(\sum \epsilon,\sum \delta)$-DP.

The privacy loss sums up. => delta=0 case.

Thm. The PLDs just do convlution. => general case.

Full $\epsilon,\delta$-DP condition: $E(\max(1-\exp(\epsilon-y),0))<\delta$

Advanced Composition Theorem: $(\sqrt{2k\ln(\delta 1^{-1})}\epsilon,k\delta+\delta1)$

Proof. Concentrate PLD.

Parallel Composition Theorem: Split disjoint, then DP => sum is also $(\epsilon,\delta)$

Subsampling Theorem: Sample a subset, then run DP algorithm. => $(\epsilon,\delta)$ will decrease.

Subsampling Theorem(Sample $B$ variables): $\epsilon'=\ln(1+(B/n)(e^{\epsilon}-1)),\delta'=B/n\delta$.

Subsampling for Add/Remove DP: Sample each element with probability $B/n$ will work(exercise 3)

Proof for substitude $\delta=0$ case: Easy! For X1,X2: Originally: $exp(L)=\frac{Pr[M(X1)=o]}{Pr[M(X2)=o]}\leq e^{\epsilon}$

Sample indices. With probability $1-p$(p=B/n), the input is the same! For others, pair and use DP condition => $1+p(e^{\epsilon}-1)$

### Group Privacy

Move multiple rows? k-Neighboring.

From DP naively: $(\epsilon,\delta)^n$, then compute it.

2-add/remove => substituition

### Properties(2)

Against Reconstruction Attacks: For independent input, any adversary fails to reconstruct one entry with constant probability.(Use DP condition)

### Non-continous output: Exponential Methods

E.g.

1. Most Frequent Element: Sensitivity=n
2. Median: Sensitivity=n-1

Solution: Selete approximate output.

Selection Problem: Each element has a score(h,X), output an element that approximate the maximum score.

Score Sensitivity: $\max \delta score$. In these cases $1/2$.

Exponential Mechanisms[McSherry-Talwar]

$\epsilon$-DP, whp the score gap is $-O(\Delta\log H/\epsilon)$.

Output each $h\in H$ with prob. $pr\sim exp(0.5\epsilon*score/\Delta(scr))$.

Proof. Compute the loss(consider normalization factor!)

Utility Proof: Also consider normalization factor.

### Correlated Noise Mechanism

Cumulative Histogram Problem: Query = range sum. Neighbor => L1 norm <=2

Binary Tree Mechanism: segment tree, compute sum of each node, add noise on each node. n => log n

General Linear Queries. Data = $W$, query $y\to Wy$.

W=LR, L(Ry+noise) Goal: L is simple.

### Lowerbounds

Counting $0/1$: Sensitivity=1, DLap gives $O(1/\epsilon)$ absolute error.

Thm. Any $\epsilon-DP$ gives $\Omega(1/\epsilon)$ error.

Proof. Hybrid. From all 0 to all 1, $n\epsilon$ privacy. Take $n=\Omega(1/\epsilon)$, we must change from $0$ to $\Omega(1/\epsilon)$.

D-dim vector sum: Sensitivity = d, $O(d/\epsilon)$ per dim => $O(d^2/\epsilon)$

Thm. There are exponentially 0/1 vectors that are pairwise far(O(d))

$n\times v1,n\times v2,\cdots$.

Find some datasets that the good outcomes are pairwise disjoint. Then link together.(packing lowerbound)

Lowerbound Most frequent: exponential mechanism is optimal.

Lowerbound for range sum: gap is open.

What about approximate DP? $\delta$ becomes very large.

Def. Discrepancy of hypergraph: 2-coloring, minimize max. color difference in each edge. Or matrix way.

$disc_C$: can leave some fraction empty.

Muthukrishnan-Nikolov 12: about discrepancy.

### Local(Distributed) DP

Distributed Laplace Mechanism: each add => RMSE = $O(\sqrt n)/\epsilon$

Randomized Response: Output true with some pr, flip otherwise.(statistical correct(removing the bias))

## LP Rounding

How to relax ILP to LP?

LP: Min cTx st. Ax>=b x>=0.

polyhedron/polytope

convex combination

vertex of polytope: not convex combination of other vertices. The optimal solution is vertex.

vertex => the intersection of $n$ constraint boundary.

Solving LP: Simplex/Ellipsoid/Interior Point

Adjacent vertices: Union of $n-1$ constraints, connect 2 vertices.

Ellipsoid Method: only need to query whether $x$ is feasible. Partition through center, check, use small ellipsoid to cover one part.

Strong-poly algorithms: $poly(n)$ integer operations.

LP Dual: Max bTy st. Ay<=c y>=0 (linear combination of constraints)

Strong Duality for LP: D=P

Proof of Strong Duality:

1. Separating Hyperplane lemma(point vs. polytope)

Farkas Lemma: Ax=b,x>=0 is infeasible iff. y^TA>=0,y^Tb<0 is feasible. (b vs, (Ax,x>=0))

Farkas Lemma 2: Ax<=b,x>=0 is infeasible iff. y^TA>=0,y^Tb<0,y>=0 is feasible(<= must be non-negative, or add x')

implies strong duality(directly)

### LP and polytopes

Combintorial Optimization.

Some feasible subsets, the polytope is their convex combination.

How to write this polytope?

#### polynomial facets

Bipartite Matching Polytope:

Constraint: For each vertex, the sum of neighboring edges <=1 / xe>=0 
is the exact formulation.

Bipartite only!(proof: Hall's theorem / prove $P$ is integral)

Totally Unimodular Matrix: $det\in -1,0,1$

Thm. for TUM A, integral b, Ax>=b is integral. (Cramer's Rule, det in +-1,0)

If each row has at most one $1$, one $-1$, then it is TUM.(cycle case => det=0(add/rem))

If $0/1$, $1s$ in one row is TUM(differential it)

E. g. Integral Scheduling(simple/multi)

Bipartite Matching is TUM(flip the second part(columns), then 1/-1)

#### Separation Oracles

Cases: facets are many, but we can answer separation oracles.

then use Ellipsoid Algorithm.

E.g. Cut polytope <=> Every path from $s$ to $t$ has weight >=1

Spanning Tree polytope <=> sum xe = n-1, for each vertex set, sum xe <= |s|-1(A typical flow problem, see RabbitWorking)

General Graph Perfect Matching polytope <=> sum xe=1 + for each odd set, cut weight >=1

### Extension Complexity

Sometimes, adding extra variables makes things easier.

Extension: Adding extra variables $y$, the feasible set is $\{x|\exists y, (x,y)\in LP\}$.

E.g. Permutation polytope

$2^n$ facets: for each set, sum >= $\frac12 k(k+1)$

How to extension? Let $p_{i,j}=$ percentage of $j$ used ad position $i$. Then the constraint is just: 

1. $p$ forms a bipartite matching polytope 
2. $\sum jp_{i_j}=x_i$

This gives a polynomial size extension.

However, for some other problems(E.g. the cut polytope), there is a polynomial-time separation oracle, but the extension complexity lowerbound is exponential.

## Markov Chains

Random Process. $X_t$. Discrete time / Continous time.

We consider discrete time case.

The state space may also be discrete or continous.(Also consider discrete)

A random process is a MC iff. $\forall$,
$$
\Pr[x_{t+1}|x_0,\cdots,x_t]=\Pr[x_{t+1}|x_t]
$$
(we omit $x_{i}=v_i$ notion) (E.g. Random walk)

Only depends on the current state.(Conditional independent)

We can just write Transition matrix: $P^(t)(x_t,x_{t+1})$. Usually, we consider the case that $P$ is independent of $t$(time-homogeneous)

Row-scohastic matrix: $a_{i,j}\geq 0$, row sum =1.

State distribution $\pi$, $\pi^{t+1}=\pi^tP$.

Stationary distribuiton: $\pi=\pi P$

Thm. Such $\pi$ always exists for scohastic $\pi$.

Convergence?

1. Reducible: The MC $P$ is reducible(not SCC) => not converge to unique $\pi$.
2. Periodic: Every cycle passing $u$ has a length gcd >1 => not converge!

Convergence theorem: If MC is irreducible and ergodic(aperiodic+positive recurrent, aperiodic in finite case), then all initial state converges to the unique stationary distribution.

Each SCC has the same period(note: not simple cycle only.)

Break period? lazy MC: does nothing with pr $\alpha$, which automatically gives gcd=1. The stationary distribution retains, and becomes aperiodic.

Infinite case? transient state: the state goes back to itself for finite many times(with pr 1). recurrent state: the state always goes back to itself for infinite many times.

The returning probability could be defined. Which leads to recurrent if Pr=1 and transient if pr<1.

Positive recurrent: the expected returning time is finite.

Proof. Coupling argument.

Two MC x,y. X is the original MC; Y starts from the stationary distribution.

Run two MC simultaneously, if $X_t=Y_t$, let $X_{t+k}\leftarrow Y_{t+k}$ after that.

1. The transition low of $X$ is equivalent to original case.
2. $X_t=Y_t$ always happen since recurrent+ergodic.

So $X$ converges to $Y$.

Random walk on graph: The stationary distribution $\pi_u= deg_u/sum deg$

irreducible: connected

ergodic: non-bipartite(or, lazy walk(1/2(I+P)))

Time-reversible MC: $\pi(x)P(x,y)=\pi(y)P(y,x)$ for some $\pi$.(Detailed Balance Equation)

1. such $\pi$ is always stationary.
2. In a stationary chain, $X_i\sim X_i^R$

### Markov Chains on Colorings

Generate a random q-coloring.

colors: q Max degree: $\Delta$. If $q>\Delta$, the solution exists.

Glauber dynamics/Gibbs sampling: Each step, re-color one vertex.

If $q\geq \Delta+2$, then it is irreducible+ergodic

By Time-reversible equation, the stationary distribution is uniform over proper coloring.

Metropolis chain: random vertex/color, skip if resulting improper. $q\geq \Delta+2$.

is also uniform startionary.

### Mixing time

What is the convergence speed of MC?

Mixing time $\tau(\epsilon)$: Number of steps needed to reduce the distance of $\pi^t,\pi$ to $\epsilon$.(consider all possible initial distributions)

$\tau_{mix}=\tau(1/2e)$

Thm. difference always decrease.

The distance: statistical difference(total vatiation): sum 1/2 |pi-qi|

Coupling of random variables $x,y$: a joint distribution over $\Omega^2$ that $p(x)=\sum_y \mu(x,y), q(y)=\sum_x \mu(x,y)$

Coupling Lemma: For any coupling, $\Pr[x\neq y]\geq |p-q|$.(Easy)

Analysis of Mixing time:

### Coupling of Markov Chains

Starting from $(x,y)$, then transition to the next state $(x',y')$ is a coupling of $P(x,x')$ and $P(y,y')$

Another rule: if $x_t=y_t$, then they always equal.

If $Pr[x_t\neq y_t]<\epsilon$, then $\tau(\epsilon)\leq t$ by coupling, so we only need to bound this probability.

How to design Coupling?

#### Riffle Shuffle

1. Split to $L/n-L$, where $L\sim Binomial(n,1/2)$
2. Interleaving: if two sets has $a,b$ remains, then choose left with probability $a/(a+b)$

Any Cut-interleaving pair has weight $2^{-n}$, the original permutation has weight $(n+1)/2^n$, other possible pair has $2^{-n}$

Inverse Riffle Shuffle: Sample $0/1$ for each card, then radix sort it.

Pure symmetry => inverse and original has the same mixing time.

The coupling: both side use the same random bit for cards.

Same perm <= each card receives different random bit combination(that is a radix sort).

$2^t\approx O(n^2)$, $\tau\approx 2\log_2 n$.

#### Metropolis Dynamics

Coupling: choose the same vertex/color.

Consider the number of unmatched(color) vertices $d$

With $q-2\Delta$ choices, $d$ can $-1$. With at most $2\Delta$ choices, $d$ can $+1$.

=> If $q>4\Delta$, it is ok.

#### Glauber Dynamics

Conjecture: Mixing time for $q\geq \Delta+2$

sota: 11/6x

Coupling: choose the same vertex. Try the optimal coupling for colors.

Path coupling: Only consider $d(X,Y)=1$. Then there is a way of automatically generating coupling.

When $d=1$:

1. $-1$ with probability $1/n$
2. $+1$ with probability $\Delta/n*(some computation on difference) 1/(q-\Delta)$

$q>2\Delta$ is ok.

## High-Dimensional Expander and Markov Chains

### Change of metrics

Cutoff Phenomenon: The statistical difference may first decrease slowly, then fast, then slow. Hard to bound.

Change metric: $\phi$-divergence: $D_\phi=E_{x\sim\mu}\phi(\nu(x)/\mu(x))$, $\phi$ convex, $\phi(1)=0$.

E.g. $\phi(x)=x\log x$ means KL Divergence

$\phi(x)=\frac12|x-1|$ means statistical difference.

$\phi(x)=(x-1)^2$ means $\chi^2$-variance

We can define more.

Now we can move from SD to decay of variance($\chi^2$), and entropy($KL$).

$\phi$-entropy. Var <=> $\phi=(x-1)^2$, Ent <=> $\phi=x\log x$

Var/Ent bound => SD bound: some calculation/some inequality(Cauchy/Pinsker).

### Markov Kernels

A transition Matrix from one state space to (possibly different) one.

Adjoint operator: Inverse of transition.

Time-reversible <=> adjoint

E.g. Spanning Tree and edge exchange MC is equivalent to N(delete 1 edge)N^a(add 1 edge)

Glauber dynamic is equivalent to N(erase 1 color)N^a(color back)

Field Dynamics.

### Data processing inequalities

After adding any Markov Kernel, any $\phi$-divergence does not increase.(Any process cannot increase information)

Proof. open $\phi$-variance

Large step => NN^c => prove decay for $N$ or $N^c$

### Eigenvalues of MC

Consider reversible MCs.

1. Its orthogonal eigenvectors exists, $\lambda\in[-1,1]$ $\lambda_{max}=1$.

Relations with decay of variance: Decay rate ~ second largest eigenvalue.

### High-Dim expanders

Consider homogeneous distributions: element is from $\binom Uk$

You can change other distributions to homogeneous!(Coloring => independent set^k)

Down-Up Walk: first drop 1 element uniformly, then add sth back respect to some value.(by adjoint)

$k-l$ down-up walk: drop k-l elements.

E.g. coloring/spanning tree

$l=k-1$: pure global

$l=1$: pure local, $\lambda_2$ is easy since $AB<->BA$

local-to-global theorem[alev-lau20]: If each $t-1$ has a decay $1-\gamma_t$, then the total walk $k-l$ has $1-\prod \gamma_t$

Expected: $1-\gamma_t\leq C/(k-t)$, then it is about $O(k^C)$ time(ommited).

Proof of local-to-global theorem

## Markov Chain Analysis(3)

Sampling, Mixing time.

Goal(as usual): effectively sampling(using some weight) with low error.

Ising Model: Spin system on $\{\pm 1\}^n$, weight $\mu=\exp(H)$, $H=\sum_{i<j}J_{i,j}x_ix_j+\sum b_ix_i$(physically potential)

WLOG assume $J$ is PSD.

Change of metrics and decay method. Choose the correct metric.

Chi2 contraction is computable(matrix eigenvalue)

Entropy contraction gives better result.

Down-Up walk,eigenvalue analysis,kernels.

Spin system MC:

Simple $O(n\log n)$ lowerbound by considering each bit should be flip once.(independent case: $\theta(n\log n)$)

General $O(n\log n)$ analysis: Anari-Jain-Koehler-Pham-V 21

Decomposition into smaller chains, conditional value.

Spectral/entropic independence, entropic stability

