---
title: CTT2020 自选题题解
date: '2021-07-03 19:11:16'
updated: '2021-07-03 19:11:16'
tags: Mildia
permalink: Nightingale/
description: 2020-2021 国家集训队自选题
mathjax: true
---

#### 2020 集训队自选题 题解

Solved: **32/53**

Todo(如果我想得起来的话): #172 #256

因为某些原因，代码部分又回来了/cy

##### #108. 春天，在积雪下结一成形，抽枝发芽

###### Problem

给出 $n$ ，对于每一个 $i=1,2,...,n$，求 $i$ 阶 Simple Permutation 的数量，模 $998244353$。

称一个 $k$ 阶排列为 Simple Permutation，当且仅当它满足不存在一个长度不为 $1,n$ 的连续段。

$n\leq 10^5$

$2s,512MB$

###### Sol

设答案为 $f_1,...,f_n$

考虑一个生成函数 $G(x)=\sum_{i>0}i!x^i$，这相当于排列的生成函数。

考虑使用其它的方式计算排列，将排列分成两类，一类满足析合树上根为析点，另一类满足析合树上根为合点。

对于第一类，考虑枚举根的儿子数量 $m$ ，则根据析合树的性质，这 $m$ 个儿子相当于将排列划分成 $m$ 段，且每一段内部都是连续段，可以发现，值域一定也被划分成了 $m$ 段。由于根节点是析点，则这 $m$ 段的任意一个非平凡子区间(长度不为 $1,m$)不为连续段且 $m>3$。因此 $m$ 段值域排列的方案数为 $[m>3]f_m$。此时可以发现每一段内部任意排列不影响，因此只需要再乘上每一段长度的阶乘。

设 $F(x)=\sum_{i>3}f_ix^i$，则可以发现根为析点的排列的生成函数为 $\sum_{k>3}f_k(\sum_{i>0}i!x^i)^k$。

然后考虑第二类。这一类排列满足存在至少一个 $k<n$，使得前 $k$ 个数的值域区间为 $[1,k]$ 或者 $[n-k+1,n]$。可以发现，除去 $n=1$ 的情况，剩余情况两者一定只满足一个。因此可以算只考虑 $[1,k]$ 的，再乘 $2$。

考虑容斥，枚举一些 $k$ ，钦定这些位置满足前 $k$ 个数的值域区间为 $[1,k]$ ，假设钦定了 $m$ 个，则容斥系数为 $(-1)^{m+1}$。

可以发现，钦定一些位置之后，相当于把排列从这些位置划分成了若干段，一段 $[l,r]$ 只能填 $[l,r]$ 中的数，因此方案为每一段长度阶乘的乘积。

因此根为合点的排列的生成函数为 $2*\sum_{k>0}(-1)^{k+1}(\sum_{i>0}i!x^i)^k-x$

此时可以发现 $\sum_{k>0}([k>3]f_k+2*(-1)^{k+1})(\sum_{i>0}i!x^i)^k-x=\sum_{i>0}i!x^i$，因此：
$$
\sum_{k>0}([k>3]f_k+2*(-1)^{k+1}-[k=1])(\sum_{i>0}i!x^i)^k=x
$$
此时设 $h_k=[k>3]f_k+2*(-1)^{k+1}-[k=1]$，$H(x)=\sum_{k>0}h_kx^k$，则变为 $H(G(x))=x$。因此只需要求出 $G(x)$ 的复合逆，即可求出 $h$ 并求出 $f$。显然 $f_1=1,f_2=2,f_3=0$。

直接求复合逆的做法为 $O(n^2)/O((n\log n)^{1.5})$，难以接受，但 $G$ 存在较好的性质。

考虑牛顿迭代，设 $f(F(x))=G(F(x))-x$，则倍增时有：

$$
f(F_0(x))+f'(F_0(x))*(F(x)-F_0(x))=0\\
F(x)=F_0(x)-\frac{f(F_0(x))}{f'(F_0(x))}
$$
因此只需要快速求出多项式复合 $G(F(x))$ 以及 $G'(F(x))$ 即可，首先考虑 $G(F(x))$。

因为 $G(x)=\sum_{i>0}i!x^i$，则这个序列是P-recursive的，有 $x^2G'(x)+xG(x)+x=G(x)$。带入 $F(x)$ 后有：
$$
F^2(x)G'(F(x))+(F(x)-1)G(F(x))+F(x)=0\\
F^2(x)F'(x)G'(F(x))+(F(x)-1)F'(x)G(F(x))+F(x)F'(x)=0
$$
设 $H(x)=G(F(x))$，则有：
$$
F^2(x)H'(x)+(F(x)-1)F'(x)H(x)+F(x)F'(x)=0
$$
可以发现 $F(x)$ 的前几项为 $x-2x^2+2x^3...$，因此 $H'(x)$ 前系数的最低次数为 $2$，$H(x)$ 系数的最低次数为 $0$。因此可以分治FFT求出 $H(x)$。此时可以发现 $H'(x)=G'(F(x))*F'(x)$，因此求逆即可求出 $G'(F(x))$。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 263001
#define mod 998244353
int n,op,gr[2][N*2],rev[N*2],ntt[N],a[N],b[N],v[N],s[N],t[N],f[N],g[N],h[N],p[N],q[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int t)
{
	for(int s=0;s<2;s++)
	for(int l=2;l<=1<<t;l<<=1)
	{
		int tp=pw(3,(mod-1)/l);if(!s)tp=pw(tp,mod-2);
		int st=1;for(int i=0;i<l;i++)gr[s][l+i]=st,st=1ll*st*tp%mod;
	}
	for(int l=2;l<=1<<t;l<<=1)for(int i=0;i<l;i++)rev[i+l]=(rev[(i>>1)+l]>>1)|(i&1?l>>1:0);
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=0;j<l>>1;j++)
	{
		int s1=ntt[i+j],s2=1ll*ntt[i+j+(l>>1)]*gr[t][l+j]%mod;
		ntt[i+j]=(s1+s2)%mod,ntt[i+j+(l>>1)]=(s1-s2+mod)%mod;
	}
	int tp=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*tp%mod;
}
void cdq(int l,int r)
{
	if(l==r){g[l]=(g[l]+t[l])%mod;return;}
	int mid=(l+r)>>1;
	cdq(l,mid);
	int le=1;while(le<=(r-l)*1.5+4)le<<=1;
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<=mid-l;i++)a[i]=g[i+l];
	for(int i=1;i<=r-l;i++)b[i]=s[i];
	dft(le,a,1);dft(le,b,1);
	for(int i=0;i<le;i++)a[i]=1ll*a[i]*b[i]%mod;dft(le,a,0);
	for(int i=mid+1;i<=r;i++)g[i]=(g[i]+a[i-l])%mod;
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<=mid-l;i++)a[i]=1ll*g[i+l]*(i+l)%mod;
	for(int i=1;i<=r-l+1;i++)b[i]=v[i];
	dft(le,a,1);dft(le,b,1);
	for(int i=0;i<le;i++)a[i]=1ll*a[i]*b[i]%mod;dft(le,a,0);
	for(int i=mid+1;i<=r;i++)g[i]=(g[i]+a[i-l+1])%mod;
	cdq(mid+1,r);
}
void inv(int n,int *f,int *g)
{
	if(n==1){g[0]=pw(f[0],mod-2);return;}
	inv((n+1)>>1,f,g);
	int le=1;while(le<=n*2+1)le<<=1;
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<n;i++)a[i]=f[i],b[i]=g[i];
	dft(le,a,1);dft(le,b,1);for(int i=0;i<le;i++)a[i]=(2ll*b[i]-1ll*b[i]*b[i]%mod*a[i]%mod+mod)%mod;dft(le,a,0);
	for(int i=0;i<n;i++)g[i]=a[i];
}
void solve(int n)
{
	if(n<=2){f[1]=1;f[2]=mod-2;return;}
	solve((n+2)>>1);
	for(int i=0;i<=n+1;i++)g[i]=h[i]=p[i]=q[i]=0;
	int le=1;while(le<=n*2+5)le<<=1;
	for(int i=0;i<=n;i++)g[i]=f[i],h[i]=1ll*(i+1)*f[i+1]%mod;
	dft(le,g,1);dft(le,h,1);
	for(int i=0;i<le;i++)v[i]=1ll*g[i]*g[i]%mod,s[i]=1ll*(g[i]+mod-1)*h[i]%mod,t[i]=1ll*h[i]*g[i]%mod;
	dft(le,v,0);dft(le,s,0),dft(le,t,0);
	for(int i=0;i<le;i++)g[i]=h[i]=0;
	cdq(1,n+1);
	for(int i=0;i<=n;i++)h[i]=1ll*f[i+1]*(i+1)%mod,q[i]=1ll*g[i+1]*(i+1)%mod;
	for(int i=0;i<=n+1;i++)p[i]=0;
	inv(n+1,h,p);
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<=n;i++)a[i]=q[i],b[i]=p[i];
	dft(le,a,1);dft(le,b,1);for(int i=0;i<le;i++)a[i]=1ll*a[i]*b[i]%mod;dft(le,a,0);
	for(int i=0;i<=n;i++)h[i]=g[i];h[1]--;
	for(int i=0;i<=n;i++)g[i]=a[i];
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<=n+1;i++)a[i]=f[i],b[i]=g[i];
	dft(le,a,1);dft(le,b,1);for(int i=0;i<le;i++)a[i]=1ll*a[i]*b[i]%mod;dft(le,a,0);
	for(int i=0;i<=n;i++)h[i]=(mod+a[i]-h[i])%mod;
	for(int i=0;i<=n;i++)p[i]=0;
	inv(n+1,g,p);
	for(int i=0;i<le;i++)a[i]=b[i]=0;
	for(int i=0;i<=n+1;i++)a[i]=h[i],b[i]=p[i];
	dft(le,a,1);dft(le,b,1);for(int i=0;i<le;i++)a[i]=1ll*a[i]*b[i]%mod;dft(le,a,0);
	for(int i=0;i<=n;i++)f[i]=a[i];
}
int main()
{
	scanf("%d%d",&op,&n);
	init(18);solve(n);
	for(int i=1;i<=n;i++)f[i]=(mod+(i&1?2:mod-2)-f[i])%mod;
	f[2]=2;
	for(int i=1;i<=n;i++)if(op||i==n)printf("%d\n",f[i]);
}
```

##### #116. Guess

###### Problem

交互库有一个各位数字不同的四位整数(可以有前导0)，你可以向交互库进行两种询问：

1. 给出一个各位数字不同的四位整数(可以有前导0)，交互库返回它与答案对应相等的数位个数。
2. 给出一个各位数字不同的四位整数(可以有前导0)，交互库返回它与答案同时拥有的数位个数。

你可以向交互库进行若干轮询问，每一轮你可以给出若干个询问，然后交互库会一次性返回这一轮询问的答案。

你需要在询问轮数不超过 $q$ ，总询问轮数不超过 $p$ 的情况下得到这个数。

多组数据

$T\leq 10^5,q=10,p=2$

$2s,512MB$

###### Sol

通过随机若干个第一轮询问的情况，可以发现：

| 询问次数   | 4     | 5    | 6    | 7    | 8         | 9        |
| ---------- | ----- | ---- | ---- | ---- | --------- | -------- |
| 剩余解数量 | $144$ | $72$ | $24$ | $24$ | $\leq 16$ | $\leq 9$ |

可以发现通过 $6$ 次构造的type2询问(例如: $0134,0257,1258,1378,1379,2489$ )之后，可以求出答案包含哪些数位。

然后考虑 $4$ 次type1求出答案，经过随机方案可以发现这是可行的，一种方式为 $1324,1342,3214,3241$

因此，通过这两步可以在 $2$ 组 $10$ 次内得到答案。

因为数据组数很大，求出数位时枚举可能的情况不行，可以预处理每种答案对应哪种数位。

###### Code

```cpp
#include "guess.h"
#include<vector>
#include<algorithm>
using namespace std;
int fu[4],st[4],tp[6]={134,257,1258,1378,1379,2489},tp2[4][4]={0,2,1,3,0,2,3,1,2,1,0,3,2,1,3,0},t1[4]={1,10,100,1000},t2[4],as[41000][4],f1;
void init()
{
	for(fu[0]=0;fu[0]<7;fu[0]++)
	for(fu[1]=fu[0]+1;fu[1]<8;fu[1]++)
	for(fu[2]=fu[1]+1;fu[2]<9;fu[2]++)
	for(fu[3]=fu[2]+1;fu[3]<10;fu[3]++)
	{
		int fg=1,vl=0;
		for(int j=0;j<6&&fg;j++)
		{
			int c1=0;
			for(int s=0;s<4;s++)for(int t=0;t<4;t++)if(fu[s]==tp[j]/t1[t]%10)c1++;
			vl=vl*5+c1;
		}
		for(int j=0;j<4;j++)as[vl][j]=fu[j];
	}
	f1=1;
}
int Solve(int x,int y)
{
	if(!f1)init();
	vector<int> s1,s2;
	for(int i=0;i<6;i++)s1.push_back(-tp[i]);
	s1=Query(s1);
	int vl=0;
	for(int i=0;i<6;i++)vl=vl*5+s1[i];
	for(int i=0;i<4;i++)st[i]=as[vl][i];
	for(int i=0;i<4;i++)
	{
		int as=0;
		for(int j=0;j<4;j++)as+=st[tp2[i][j]]*t1[j];
		t2[i]=as;
	}
	for(int i=0;i<4;i++)s2.push_back(t2[i]);
	s2=Query(s2);
	do{
		int fg=1;
		for(int j=0;j<4&&fg;j++)
		{
			int c1=0;
			for(int s=0;s<4;s++)if(st[s]==t2[j]/t1[s]%10)c1++;
			if(c1!=s2[j])fg=0;
		}
		if(fg)
		{
			int as=0;
			for(int j=0;j<4;j++)as+=t1[j]*st[j];
			return as;
		}
	}while(next_permutation(st,st+4));
}
```

##### #124. Permutation

###### Problem

定义 $f_n$ 为满足如下条件的 $n$ 阶排列 $p$ 数：

1. 存在 $i$，满足 $p_i=i$
2. 存在 $i$，满足 $p_i=n-i+1$

给出 $n,p$，求出 $\oplus_{i=1}^n(f_i\bmod p)$

$p$ 不一定是质数

$n\leq 10^7,n+1\leq p\leq 10^9$

$1s,512MB$

###### Sol

考虑容斥，只需要算不满足条件1的方案数，不满足条件2的方案数和同时不满足的方案数。

显然前两个是错排，计算过程不需要除法，因此可以直接算。

考虑最后一个，相当于求满足 $p_i\neq i,p_i\neq n-i+1$ 的排列数量。

再考虑容斥，相当于钦定一些 $p_i=i,p_i=n-i+1$，计算钦定了 $k$ 个的方案数 $g_{n,k}$，答案即为 $\sum (-1)^i(n-i)!g_{n,k}$

可以发现互相冲突的限制一定会分成若干个长度为 $4$ 或 $1$ 的组，一组内为 $p_i=i,p_{n-i+1}=i,p_{n-i+1}=n-i+1,p_i=n-i+1$，且每一组中相邻两个以及首尾互相冲突。不同组不冲突。

因此可以发现， $g_{2n,k}=[x^k](1+4x+2x^2)^n,g_{2n+1,k}=[x^k](1+x)(1+4x+2x^2)^n$

为了简便，将系数翻转，则有：
$$
f_{2n}=\sum_{i=0}^{2n}(-1)^{2n-i}i![x^i](2+4x+x^2)^n\\
f_{2n+1}=\sum_{i=0}^{2n+1}(-1)^{2n+1-i}i![x^i](1+x)(2+4x+x^2)^n\\
$$
设 $v_{2n}=\sum_{i=0}^{2n}(-1)^{i}i![x^i](2+4x+x^2)^n\\v_{2n+1}=\sum_{i=0}^{2n+1}(-1)^{i}i![x^i](1+x)(2+4x+x^2)^n$，考虑计算 $v_i$。



设 $S(F(x))=\sum_{i\geq 0}[x^0](-1)^iF^{(i)}(x)$，则显然 $v_{2n}=S((2+4x+x^2)^n),v_{2n+1}=S((1+x)(2+4x+x^2)^n)$

同时，可以发现 $S(F(x))+S(G(x))=S(F(x)+G(x)),S(F(x))=[x^0]F(x)-S(F'(x))$

因此此时有：
$$
v_{2n}=S((2+4x+x^2)^n)\\
=2^n+S(n*(2+4x)*(2+4x+x^2)^{n-1})\\
=2^n+2n*S((1+x)*(2+4x+x^2)^{n-1}+(2+4x+x^2)^{n-1})\\
=2^n+2n*(v_{2n-1}+v_{2n-2})\\
v_{2n+1}=S((1+x)(2+4x+x^2)^n)\\
=2^n+S((2+4x+x^2)^n+n*(1+x)*(4+2x)*(2+4x+x^2)^{n-1})\\
=2^n+v_{2n}+n*S((4+6x+2x^2)*(2+4x+x^2)^{n-1})\\
=2^n+v_{2n}+n*S(2*(2+4x+x^2)^{n}-2x(2+4x+x^2)^{n-1})\\
=2^n+(2n+1)v_{2n}-2n*S(((1+x)-1)(2+4x+x^2)^{n-1})\\
=2^n+(2n+1)v_{2n}-2n*(v_{2n-1}-v_{2n-2})
$$

因此可以在 $O(n)$ 的时间内递推出 $v$

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
int n,mod,s1,s2,s3,t1,p2,fr,as;
int main()
{
	scanf("%d%d",&n,&mod);
	s2=1,as=1,p2=2,fr=1;
	for(int i=2;i<=n;i++)
	{
		int vl;
		if(i&1)vl=(p2+1ll*(i-1)*(s2-s3)%mod-1ll*i*s1%mod+mod)%mod,p2=2*p2%mod;
		else vl=(p2-1ll*i*(s1+s2)%mod+mod)%mod;
		s3=s2,s2=s1,s1=vl;
		t1=(1+mod-1ll*i*t1%mod)%mod;vl=(2ll*t1-vl+mod)%mod;
		if(i&1)vl=mod-vl;
		fr=1ll*fr*i%mod;
		as^=(fr+mod-vl)%mod;
	}
	printf("%d\n",as);
}
```

##### #128. A story of The Small P

###### Problem

给定 $N,m,k$ ，求有多少个正整数序列 $h$ 满足：

1. $h$ 的长度 $n$ 大于 $0$ 不超过 $N$。

2. $\forall i\in\{1,2,...,n\},1\leq h_i\leq m$。

3. 正好存在 $k$ 个 $i\in\{1,2,...,n-1\}$ 满足 $h_i<h_{i+1}$。 

答案模 $998244353$

$2\leq N,m,k\leq 2^{19},(N-k+1)*m\leq 2^{20}$

$1s,1024MB$

###### Sol

考虑直接的dp：设 $dp_{i,j,l}$ 表示放了前 $i$ 个位置，当前有 $j$ 个位置不满足 $h_i<h_{i+1}$，当前最后一个位置为 $l$ 时的方案数。

注意到第三维值域为 $[1,m]$ 中的整数，因此可以设 $dp_{i,j*m+l}$ 表示同样的状态。

此时可以发现，对于任意状态 $dp_{i,j}$，它在转移到的状态均为 $dp_{i,j+1},dp_{i,j+2},...,dp_{i,j+m}$，因此如果将 $dp$ 写成生成函数的形式，则有：
$$
dp_i=(x+x^2+...+x^m)^i
$$
此时可以发现求的是
$$
\sum_{n=1}^{N}\sum_{i=1}^m[x^{(n-k-1)m+i}](x+x^2+...+x^m)^n
$$
可以发现，在乘一个 $x^{(N-n)m}$ 后，所有 $n$ 处都求的是 $n^{(N-k+1)m+1,...,(N-k)m}$ 的系数和，因此答案为：
$$
\sum_{i=(N-k+1)+1}^{(N-k)m}[x^i]\sum_{n=1}^{N}(x+x^2+...+x^m)^nx^{m(N-n)}
$$
只需求出右侧多项式 $\bmod x^{(N-k)m+1}$ 的结果即可。
$$
\sum_{n=1}^{N}(x+x^2+...+x^m)^nx^{m(N-n)}\\=x^N*\frac{(1+x+...+x^{m-1})^{N+1}-(x^{m-1})^{N+1}}{1+x+...+x^{m-2}}-x^{Nm}
$$
考虑求 $(1+x+...+x^{m-1})^{N+1}$ ，设 $F(x)=1+x+...+x^{m-1},G(x)=F(x)^{N+1}$，则对式子求导有 $F'(x)=(N+1)G'(x)G(x)^N$，所以 $F'(x)G(x)=(N+1)G'(x)F(x)$

因此对比 $x^n$ 系数可以得到 $\sum_{i=0}^{m-1}(n-i+1)f_{n-i+1}=(N+1)\sum_{i=1}^{m-1}i*f_{n-i+1}$

因此 $(n+1)f_{n+1}=\sum_{i=1}^{m-1}((N+2)*i-n-1)f_{n-i+1}$，可以线性求出 $f$。

最后计算多项式求逆的时候，因为系数全 $1$ ，表示为递推后形如 $s_i=v_i-\sum_{j=1}^{m-2}s_{i-j}$，同样可以线性。

复杂度 $O((N-k)m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1050001
#define mod 998244353
int n,m,k,fr[N],ifr[N],v[N],su[N],st,s1,s2,as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d",&n,&m,&k);st=(n-k)*m;
	fr[0]=1;for(int i=1;i<=1<<20;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1<<20]=pw(fr[1<<20],mod-2);for(int i=1<<20;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	v[0]=1;s2=1;
	for(int i=1;i<=st;i++)
	{
		v[i]=(1ll*i*(n+1)%mod*s2-1ll*(n+2)*s1%mod+mod)%mod*ifr[i]%mod*fr[i-1]%mod;
		s1=(s1+1ll*v[i]*i)%mod;s2=(s2+v[i])%mod;
		if(i-m+1>=0)s1=(s1-1ll*v[i-m+1]*(i-m+1)%mod+mod)%mod,s2=(s2-v[i-m+1]+mod)%mod;
	}
	for(int i=st;i>=0;i--){if(i+n<=st)v[i+n]=v[i];v[i]=0;}
	if(1ll*(n+1)*m-1<=st)v[(n+1)*m-1]--;
	s1=v[0];
	for(int i=1;i<=st;i++)
	{
		v[i]=(v[i]-s1+mod)%mod;
		s1=(s1+v[i])%mod;
		if(i-m+2>=0)s1=(s1-v[i-m+2]+mod)%mod;
	}
	if(1ll*n*m<=st)v[n*m]--;
	for(int i=(n-k-1)*m+1;i<=(n-k)*m;i++)if(i>=0)as=(as+v[i])%mod;
	printf("%d\n",as);
}
```

##### #144. Game On a Circle Ⅱ

###### Problem

有一个 $n$ 个点的环，环上位置 $i$ 有一个数 $a_i$，所有数构成一个 $n$ 阶排列。有一个初始为空的序列 $b$。一个指针在环上顺时针移动，初始位置为 $1$。每次移动前，有 $p$ 的概率指针所在的位置的数从环上消失并移动到序列末尾。

求出所有数都消失时，序列 $b$ 在所有 $n$ 阶排列中的字典序排名的期望，模 $998244353$。

多组数据，$T\leq 1000,\sum n\leq 5\times 10^5$

$2s,1024MB$

###### Sol

对于一个 $n$ 阶排列 $q$ ，根据康托展开，它的字典序编号可以表示为：
$$
1+\sum_{i=1}\sum_{j>i}[q_i>q_j](n-i)!
$$
考虑暴力计算，枚举 $q_i$ 所在的位置 $x$ ，$q_j$ 所在的位置 $y$，此时剩余的系数只与比 $x$ 后出现的数的个数有关。

假设枚举了 $[1,x-1]$ 中有 $a$ 个位置在 $x$ 后出现，$[x+1,n]$ 中有 $b$ 个位置在 $x$ 后出现，考虑枚举 $x$ 消失时的操作轮数，则这种情况出现的概率为：
$$
\sum_{i>0}(1-p)^{i-1}*p*((1-p)^{i})^a*(1-(1-p)^{i})^{x-1-a}*((1-p)^{i-1})^b*(1-(1-p)^{i-1})^{n-x-b}
$$
设 $t=(1-p)^i$，则上面可以看成
$$
\sum_{i>0}t*\frac p{1-p}*t^a*(1-t)^{x-1-a}*(\frac t{1-p})^b*(1-\frac t{1-p})^{n-x-b}
$$
这可以被表示成 $\sum_{i>0}F(t)$ ，其中 $F(t)$ 是一个多项式，设 $F(t)=\sum_{i>0}f_it^i$，则这等于：
$$
\sum_{i>0}f_i\sum_{j>0}((1-p)^{i})^j=\sum_{i>0}f_i\frac 1{1-(1-p)^i}
$$
记这个概率为 $g(x,a,b)$，设 $pr_x$ 表示 $[1,x-1]$ 中有多少个 $i$ 满足 $a_i<a_x$，$su_i$ 表示 $[x+1,n]$ 中有多少个 $i$ 满足 $a_i<a_x$。

先枚举 $x$，考虑选择 $y$ 的情况。如果 $y>x$，则此时对答案的贡献为：
$$
su_x*\sum_{a=0}^{x-1}\sum_{b=0}^{n-x-1}C_{x-1}^aC_{n-x-1}^b(a+b+1)!g(x,a,b+1)
$$
如果小于，则贡献为
$$
pr_x*\sum_{a=0}^{x-2}\sum_{b=0}^{n-x}C_{x-2}^aC_{n-x}^b(a+b+1)!g(x,a+1,b)
$$
因此可以大力写出答案的式子：
$$

1+\sum_{x=1}^nsu_x\sum_{a=0}^{x-1}\sum_{b=0}^{n-x-1}C_{x-1}^aC_{n-x-1}^b(a+b+1)!\sum_{i>0}t\frac p{1-p}t^a(1-t)^{x-1-a}(\frac t{1-p})^{b+1}(1-\frac t{1-p})^{n-x-b-1}\\
+pr_x\sum_{a=0}^{x-2}\sum_{b=0}^{n-x}C_{x-2}^aC_{n-x}^b(a+b+1)!\sum_{i>0}t\frac p{1-p}t^{a+1}(1-t)^{x-1-a-1}(\frac t{1-p})^b*(1-\frac t{1-p})^{n-x-b}\\
=1+\sum_{x=1}^nsu_x\sum_{a=0}^{x-1}\sum_{b=0}^{n-x-1}C_{x-1}^aC_{n-x-1}^b(a+b+1)!\frac p{1-p}\sum_{i>0}\sum_{c=0}^{x-1-a}\sum_{d=0}^{n-x-b-1}C_{x-1-a}^cC_{n-x-b-1}^dt^{a+b+c+d+1}\frac{(-1)^{c+d}}{ {(1-p)}^{b+d+1}}\\
+pr_x\sum_{a=0}^{x-2}\sum_{b=0}^{n-x}C_{x-2}^aC_{n-x}^b(a+b+1)!\frac p{1-p}\sum_{i>0}\sum_{c=0}^{x-2-a}\sum_{d=0}^{n-x-b}C_{x-2-a}^cC_{n-x-b}^dt^{a+b+c+d+1}\frac{(-1)^{c+d}}{ {(1-p)}^{b+d}}
$$

可以发现，除去一个 $\frac 1{1-p}$ 的系数后， $su$ 一项的形式只和 $x-1$ 有关， $pr$ 一项的形式只和 $x-2$ 有关，且两项形式相同，因此可以合并两侧，此时等于：
$$
1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{a=0}^{t-1}\sum_{b=0}^{n-t-1}C_{t-1}^aC_{n-t-1}^b(a+b+1)!*\\\sum_{i>0}\sum_{c=0}^{t-1-a}\sum_{d=0}^{n-t-b-1}C_{t-1-a}^cC_{n-t-b-1}^dt^{a+b+c+d+1}*\frac{(-1)^{c+d}}{ {(1-p)}^{b+d}}\\
=1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{a=0}^{t-1}\sum_{b=0}^{n-t-1}C_{t-1}^aC_{n-t-1}^b(a+b+1)!*\\\sum_{c=0}^{t-1-a}\sum_{d=0}^{n-t-b-1}C_{t-1-a}^cC_{n-t-b-1}^d\frac 1{1-(1-p)^{a+b+c+d+1}}*\frac{(-1)^{c+d}}{ {(1-p)}^{b+d}}
$$
可以发现，中间的组合数相当于在 $t-1$ 个里面分别选出 $a,c$ 个，在 $n-t-1$ 个里面选 $b,d$ 个，考虑枚举 $x=a+c,y=b+d$，则有
$$
=1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{x=0}^{t-1}\sum_{y=0}^{n-t-1}C_{t-1}^xC_{n-t-1}^y*\frac 1{1-(1-p)^{x+y+1}}*\frac1{(1-p)^y}*\\(\sum_{a=0}^{x}\sum_{b=0}^{y}C_x^aC_y^b(a+b+1)!(-1)^{x+y-a-b})\\
=1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{x=0}^{t-1}\sum_{y=0}^{n-t-1}C_{t-1}^xC_{n-t-1}^y*\frac 1{1-(1-p)^{x+y+1}}*\frac1{(1-p)^y}*\sum_{a=0}^{x+y}C_{x+y}^a(a+1)!(-1)^{x-a}\\
=1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{x=0}^{t-1}\sum_{y=0}^{n-t-1}C_{t-1}^xC_{n-t-1}^y*\frac1{(1-p)^y}*\frac 1{1-(1-p)^{x+y+1}}*\sum_{a=0}^{x+y}C_{x+y}^a(a+1)!(-1)^{x-a}\\
$$

可以使用NTT求出对于一个 $x+y$，$\frac 1{1-(1-p)^{x+y+1}}*\sum_{a=0}^{x+y}C_{x+y}^a(a+1)!(-1)^{x-a}$ 的值，记这个值为 $v_{x+y}$。

则现在只需要求出：
$$
1+\frac p{1-p}\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*\sum_{x=0}^{t-1}\sum_{y=0}^{n-t-1}C_{t-1}^xC_{n-t-1}^y*\frac1{(1-p)^y}*v_{x+y}
$$
考虑把它看成生成函数的形式，可以发现 

$$
[s^{x+y}](s+1)^{t-1}(\frac s{1-p}+1)^{n-t-1}=\sum_{x=0}^{t-1}\sum_{y=0}^{n-t-1}C_{t-1}^xC_{n-t-1}^y*\frac1{(1-p)^y}
$$

因此只需要求出如下生成函数每一项的系数，即可求出答案：
$$
\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})(s+1)^{t-1}(\frac s{1-p}+1)^{n-t-1}
$$
先记 $s'=s+1$，则可以表示为：
$$
\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})(s')^{t-1}(\frac {s'}{1-p}+\frac p{1-p})^{n-t-1}\\
=\sum_{t=1}^{n-1}(su_t*\frac 1{1-p}+pr_{t+1})*(\frac p{1-p})^{n-t-1}*(s')^{t-1}(\frac {s'}{p}+1)^{n-t-1}
$$
可以发现这个式子可以NTT，因此可以将生成函数表示为 $\sum_{i\geq 0}f'_i(s')^{i}=\sum_{i\geq 0}f'_i(s+1)^{i}$

最后再使用一个翻转系数的NTT，即可求出原先的生成函数。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1050000
#define mod 998244353
int T,n,v[N],nt[N],fr[N],ifr[N],p,q,a[N],b[N],c[N],tr[N],tp[N],ntt[N],vl[N],f[N],g[2][N*2],rev[N*2];
void add(int x){for(int i=x;i<=n;i+=i&-i)tr[i]++;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=0;j<l>>1;j++)
	{
		int s1=ntt[i+j],s2=1ll*ntt[i+j+(l>>1)]*g[t][l+j]%mod;
		ntt[i+j]=(s1+s2)%mod,ntt[i+j+(l>>1)]=(s1-s2+mod)%mod;
	}
	int tp=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*tp%mod;
}
void solve()
{
	scanf("%d%d%d",&n,&p,&q);
	for(int i=1;i<=n;i++)tr[i]=tp[i-1]=0;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),nt[i]=v[i]-1-que(v[i]),add(v[i]);
	p=1ll*p*pw(q,mod-2)%mod;
	if(p==1)
	{
		int as=0;for(int i=1;i<=n;i++)as=(as+1ll*nt[i]*fr[n-i])%mod;
		printf("%d\n",as);return;
	}
	for(int i=1;i<=n;i++)
	{
		if(i>1)tp[i-2]=(tp[i-2]+1ll*(v[i]-1-nt[i]))%mod;
		if(i<n)tp[i-1]=(tp[i-1]+1ll*pw(mod+1-p,mod-2)*nt[i])%mod;
	}
	for(int i=0;i<=n;i++)vl[i]=pw(mod+1-pw(mod+1-p,i+1),mod-2)-1;
	for(int i=0;i<=n;i++)f[i]=0;
	int l=1;while(l<=n*2)l<<=1;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=0;i<=n;i++)a[i]=i+1,b[i]=1ll*ifr[i]*(i&1?mod-1:1)%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	for(int i=0;i<=n;i++)f[i]=1ll*fr[i]*vl[i+1]%mod*a[i]%mod;
	//sum_{i=0}^{n-2}tp_i*(x+1)^i*(vx+1)^{n-2-i}
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	q=pw(mod+1-p,mod-2);
	for(int i=0;i<n-1;i++)a[i]=1ll*tp[i]*fr[n-2-i]%mod*pw(mod+1-q,n-2-i)%mod,b[i]=1ll*ifr[i]*pw(1ll*q*pw(mod+1-q,mod-2)%mod,i)%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	for(int i=0;i<n-1;i++)c[i]=1ll*a[i]*ifr[n-2-i]%mod;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=0;i<n-1;i++)a[i]=1ll*c[n-2-i]*fr[n-2-i]%mod,b[i]=ifr[i];
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	for(int i=0;i<n-1;i++)c[n-2-i]=1ll*a[i]*ifr[n-2-i]%mod;
	int as=0;
	for(int i=0;i<n-1;i++)as=(as+1ll*c[i]*f[i])%mod;
	printf("%d\n",1ll*as*p%mod*pw(mod+1-p,mod-2)%mod+1);
}
int main()
{
	scanf("%d",&T);
	fr[0]=1;for(int i=1;i<=1e6;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1000000]=pw(fr[1000000],mod-2);for(int i=1e6;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int t=0;t<2;t++)
	for(int l=2;l<=1<<20;l<<=1)
	{
		int tp=pw(3,(mod-1)/l);if(!t)tp=pw(tp,mod-2);
		int st=1;for(int i=0;i<l;i++)g[t][l+i]=st,st=1ll*st*tp%mod;
	}
	for(int l=2;l<=1<<20;l<<=1)for(int i=0;i<l;i++)rev[i+l]=(rev[(i>>1)+l]>>1)|(i&1?l>>1:0);
	while(T--)solve();
}
```

##### #148. Tour

###### Problem

有 $n$ 个数 $v_1,...,v_n$，给定一个非负整数 $k$。

求有多少个 $n$ 阶排列 $p$ ，满足 $\forall i\in\{1,2,...,n-1\},v_{p_i}*v_{p_i+1}\leq k$。答案模 $998244353$。

$n\leq 2\times 10^5,|v_i|,k\leq 10^9$

$3s,512MB$

###### Sol

问题可以看成，如果两个数乘积小于等于 $k$ ，则在两数之间连边，求这个图的哈密顿链数量。

因为 $k\geq 0$ ，正数和负数相邻一定是可行的，因此可以分开考虑正数和负数。

首先考虑只有正数的情况，将所有数排序，可以发现对于每个数，与它右边的数一定是一段前缀，因此可以使用类似Horrible Circle的方式处理。

找到第一个位置 $i$ 满足 $v_i*v_{i+1}>n$，设 $l=i,r=i+1$，维护一个初始为空的序列，然后进行如下操作直到所有数都被放入序列：

如果当前 $v_l*v_r>n$，将 $v_l$ 放入序列，然后让 $l$ 减一。

否则，将 $v_r$ 放入序列，然后让 $r$ 加一。

这时可以发现，在第一种情况中放入序列的数一定满足它向序列中它前面的点都有连边，第二种情况中它向序列中它前面的点都没有连边。因此在这个加点顺序后，可以进行dp：

设 $dp_{i,j}$ 表示考虑序列前 $i$ 个点，当前前面的点被分成了 $j$ 条路径的方案数。

如果加入一个第一种情况的点，则考虑枚举这个点合并的路径情况，可以得到：
$$
dp_{i+1,j}=dp_{i,j-1}+2j*dp_{i,j}+(j+1)*j*dp_{i,j+1}
$$

如果加入第二种情况的点，则它无法合并路径，一定有 $dp_{i+1,j}=dp_{i,j-1}$

注意到如果把 $j$ 看成当前有 $j$ 个数的话，考虑看成给出两个不同的位置，你可以向一个位置中放一个数或者不放，放完后你再获得一个数。此时 $i\to i+1$ 的转移系数为 $1$，$i\to i$ 的转移系数为 $2i$，$i\to i-1$ 的转移系数为 $i(i-1)$，可以发现这就是第一种dp的转移系数。

可以发现第二种的转移可以看成直接获得一个数，因此问题可以看成：

你初始有 $0$ 个数，有两种点排成一列。你经过第一种点时会获得一个数，所有数互不相同。你经过第二种点时，可以选择一个数放进去，也可以不放。

求出最后你剩下 $1,2,...,n$ 个数的方案数。

记得到第 $i$ 个数时右侧还有 $t_i$ 个第二种点，考虑枚举每个数放在哪个第二种点，则可以把方案表示成一个长度为 $n$ 的序列 $v_i$ ，满足：

1. $v_i=0$ 或 $v_i\in\{1,2,...,t_i\}$
2. 若 $v_i,v_j\neq 0$ ，则 $v_i\neq v_j$

此时剩下的数个数就是 $v_i=0$ 的数的个数。

因此还可以将问题表示为：给一个 $n$ 行的棋盘，第 $i$ 行有左侧 $t_i$ 个格子。求出在棋盘上放 $0,1,...,n$ 个棋子，满足不存在两个放上去的位置同行或者同列的方案数。显然这个问题和上一个等价。

此时可以使用棋盘问题的一般解法。考虑在每一行左侧加入 $x$ 个格子，假设 $x$ 足够大，求放 $n$ 个棋子满足要求的方案数。这显然是 $\prod_{i=1}^n(x+t_i-(i-1))$

换一个角度考虑，枚举有多少个棋子在原来的棋盘上，设原来棋盘上放 $0,1,...,n$ 个的方案数为 $f_0,...,f_n$，则总的方案数为：
$$
\sum_{i=0}^nf_i*x*(x-1)*...*(x-(n-i-1))=\sum_{i=0}^nf_ix^{\underline{n-i}}
$$
因此求出 $\prod_{i=1}^n(x+t_i-(i-1))$ 后求对应的下降幂多项式就是答案。

求原来的多项式可以分治FFT，注意到下降幂多项式的系数和下降幂多项式取 $1,2,...,n$ 的点值互相为二项式反演关系，多点求值求出 $n$ 个点值后再FFT即可。这部分复杂度 $O(n\log^2 n)$

这样求出了正数所有点分成 $1,2,...,n$ 段路径的方案数，负数类似。然后考虑合并两种路径。

可以发现，假设有 $a$ 段正数路径，$b$ 段负数路径，两者能合并当且仅当 $|a-b|\leq 1$，且方案数为 $(2-|a-b|)a!b!$。因此直接算即可。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 530001
#define mod 998244353
int n,v,a,rev[N*2],g[2][N*2],ntt[N],v1[N],v2[N],v3[N],v4[N],fr[N],ifr[N],as[N],fg=130;
vector<int> s1,s2,tp[20],f1[N*2];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d)
{
	fr[0]=1;for(int i=1;i<=1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1<<d]=pw(fr[1<<d],mod-2);for(int i=(1<<d);i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int i=2;i<=1<<d;i<<=1)for(int j=0;j<i;j++)rev[i+j]=(rev[i+(j>>1)]>>1)+(j&1?i>>1:0);
	for(int t=0;t<2;t++)
	for(int i=2;i<=1<<d;i<<=1)
	{
		int tp=pw(3,mod-1+(mod-1)*(t*2-1)/i);
		for(int j=0,vl=1;j<(i>>1);j++,vl=1ll*vl*tp%mod)g[t][i+j]=vl;
	}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=i,st=l;j<i+(l>>1);j++,st++)
	{
		int v1=ntt[j],v2=1ll*ntt[j+(l>>1)]*g[t][st]%mod;
		ntt[j]=(v1+v2)%mod;ntt[j+(l>>1)]=(v1-v2+mod)%mod;
	}
	int tp=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*tp%mod;
}
vector<int> add(vector<int> a,vector<int> b)
{
	vector<int> c;
	int s1=a.size(),s2=b.size();
	for(int i=0;i<s1||i<s2;i++)c.push_back(((i<s1?a[i]:0)+(i<s2?b[i]:0))%mod);
	return c;
}
vector<int> mul(vector<int> a,vector<int> b)
{
	vector<int> c;
	int s1=a.size(),s2=b.size();
	for(int i=0;i<s1+s2-1;i++)c.push_back(0);
	if(s1+s2<=fg*4)
	{
		for(int i=0;i<s1;i++)for(int j=0;j<s2;j++)c[i+j]=(c[i+j]+1ll*a[i]*b[j])%mod;
		return c;
	}
	int l=1;while(l<=s1+s2)l<<=1;
	for(int i=0;i<l;i++)v1[i]=v2[i]=0;
	for(int i=0;i<s1;i++)v1[i]=a[i];for(int i=0;i<s2;i++)v2[i]=b[i];
	dft(l,v1,1);dft(l,v2,1);for(int i=0;i<l;i++)v1[i]=1ll*v1[i]*v2[i]%mod;dft(l,v1,0);
	for(int i=0;i<s1+s2-1;i++)c[i]=v1[i];
	return c;
}
vector<int> inv(vector<int> a,int n)
{
	while(a.size()<n)a.push_back(0);
	vector<int> b;
	for(int i=0;i<n;i++)b.push_back(0);
	int tp=min(n,fg),st=pw(a[0],mod-2);
	for(int i=0;i<tp;i++)
	{
		int vl=i?0:1;
		for(int j=0;j<i;j++)vl=(vl-1ll*b[j]*a[i-j]%mod+mod)%mod;
		b[i]=1ll*vl*st%mod;
	}
	if(n<=fg)return b;
	int x=0;while((n>>x)>=fg*2-1)x++;
	while(1)
	{
		int l1=((n-1)>>x)+1,l2=(l1>>1)+1,l=1;
		while(l<=l1*2+4)l<<=1;
		for(int i=0;i<l;i++)v1[i]=v2[i]=0;
		for(int i=0;i<l1;i++)v1[i]=a[i];
		for(int i=0;i<l2;i++)v2[i]=b[i];
		dft(l,v1,1);dft(l,v2,1);for(int i=0;i<l;i++)v1[i]=1ll*v2[i]*(2-1ll*v1[i]*v2[i]%mod+mod)%mod;dft(l,v1,0);
		for(int i=0;i<l1;i++)b[i]=v1[i];
		if(!x)return b;x--;
	}
}
vector<int> polymod(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	if(s1<s2)return a;
	if(s1<=fg*2)
	{
		while(s1>=s2)
		{
			int tp=1ll*(mod-1)*pw(b[s2-1],mod-2)%mod*a[s1-1]%mod;
			for(int i=0;i<s2;i++)a[s1-s2+i]=(a[s1-s2+i]+1ll*tp*b[i])%mod;
			a.pop_back();s1--;
		}
		return a;
	}
	int tp=s1-s2+1;
	vector<int> v3,v4,v5,as;
	for(int i=0;i<tp;i++)v3.push_back(a[s1-i-1]),v4.push_back(s2-i-1>=0?b[s2-i-1]:0);
	v4=mul(inv(v4,tp),v3);
	for(int i=0;i<tp;i++)v5.push_back(v4[tp-i-1]);
	b=mul(b,v5);
	for(int i=0;i<s2;i++)as.push_back((a[i]-b[i]+mod)%mod);
	return as;
}
void pre(int x,int l,int r)
{
	if(l==r){f1[x].clear();f1[x].push_back(mod-l);f1[x].push_back(1);return;}
	int mid=(l+r)>>1;
	pre(x<<1,l,mid);pre(x<<1|1,mid+1,r);f1[x]=mul(f1[x<<1],f1[x<<1|1]);
}
void doit(int x,int l,int r,vector<int> st)
{
	if(l==r){as[l]=st[0];return;}
	int mid=(l+r)>>1;
	doit(x<<1,l,mid,polymod(st,f1[x<<1]));
	doit(x<<1|1,mid+1,r,polymod(st,f1[x<<1|1]));
}
vector<int> solve(vector<int> t)
{
	int ct=0;
	for(int i=1;i<=19;i++)tp[i].clear();tp[1].push_back(1);
	sort(t.begin(),t.end());
	if(!t.size()){vector<int> as;as.push_back(1);return as;}
	int st=0;while(st+1<t.size()&&1ll*t[st]*t[st+1]<=v)st++;
	vector<int> v1;
	int l1=st,r1=st+1,n=t.size();
	for(int i=1;i<=n;i++)
	if(r1==n||1ll*t[l1]*t[r1]>v)v1.push_back(2),l1--;
	else v1.push_back(1),r1++;
	for(int i=1,vl=0;i<=n;i++)
	{
		vector<int> st;st.push_back((mod+vl)%mod);st.push_back(1);
		for(int t=0;((i>>t)&1)<(((i-1)>>t)&1);t++)st=mul(st,tp[ct]),ct--;
		tp[++ct]=st;
		vl--;if(v1[n-i]==2)vl+=2;
	}
	while(ct>1)tp[ct-1]=mul(tp[ct-1],tp[ct]),ct--;
	pre(1,0,n);doit(1,0,n,tp[1]);
	vector<int> t1,t2;
	for(int i=0;i<=n;i++)t1.push_back(1ll*as[i]*ifr[i]%mod),t2.push_back(1ll*(i&1?mod-1:1)*ifr[i]%mod);
	t1=mul(t1,t2);while(t1.size()>n+1)t1.pop_back();
	return t1;
}
int main()
{
	scanf("%d%d",&n,&v);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&a);
		if(a>=0)s1.push_back(a);else s2.push_back(-a);
	}
	init(19);
	vector<int> t1=solve(s1),t2=solve(s2);
	int s1=t1.size(),s2=t2.size(),as=0;
	for(int i=0;i<s1&&i<s2;i++)as=(as+1ll*t1[i]*t2[i]%mod*2*fr[i]%mod*fr[i])%mod;
	for(int i=0;i+1<s1&&i<s2;i++)as=(as+1ll*t1[i+1]*t2[i]%mod*fr[i]%mod*fr[i+1])%mod;
	for(int i=0;i<s1&&i+1<s2;i++)as=(as+1ll*t1[i]*t2[i+1]%mod*fr[i]%mod*fr[i+1])%mod;
	printf("%d\n",as);
}
```

##### #160. monster

###### Problem

有 $n$ 个数，第 $i$ 个数初始为 $a_i$。

依次考虑每一个数，考虑到数 $i$ 时，有 $p_i$ 的概率让 $a_i$ 减一，否则不变。

循环执行上面的操作，如果一次 $-1$ 时将某个数变成了 $0$，则操作停止。

求出停止时第一个数变成 $0$ 的概率，模 $998244353$。

$n\leq 5,\sum a_i\leq 5\times 10^6$

$1s,1024MB$

###### Sol

考虑枚举在第 $i+1$ 轮结束，那么一定是在前 $i$ 轮中，第一个数变成了 $1$，剩下的数还没变成 $0$。

此时枚举剩下的数被减了多少次，则可以得到答案为：
$$
\sum_{i\geq 0}\sum_{b_1=a_1-1,b_2<a_2,b_3<a_3,...,b_n<a_n}\prod_{j=1}^nC_i^{b_j}p_j^{b_j}*(1-p_j)^{i-b_j}
$$
可以发现，如果提出 $\prod_{j=1}^n(1-p_j)^i$，则右侧是一个关于 $i$ 的多项式，可以发现它的次数一定不超过 $\sum a_i$。

假设求出了这个多项式 $F(x)=\sum_{i=0}^{\sum a_i}f_ix^i$，则相当于计算
$$
\sum_{t\geq 0}\sum_{i\geq 0}f_i*t^i*(\prod_{j=1}^n(1-p_j))^t
$$
将 $F$ 表示成类似下降幂的形式：$F(x)=\sum_{i\geq 0}g_iC_x^i$，则上面的式子等于：
$$
\sum_{t\geq 0}\sum_{i\geq 0}g_i*C_t^i*(\prod_{j=1}^n(1-p_j))^t
$$
注意到 $C_1^i,C_2^i,...,C_t^i$ 对应的生成函数为 $\frac 1x(x+x^2+...)^i=\frac1x*(\frac x{1-x})^i$，令 $p=\prod_{j=1}^n(1-p_j)$，则可以发现：
$$
\sum_{t\geq 0}C_t^i*p^t=\frac 1p*(\frac p{1-p})^i
$$
因此如果求出了 $g_i$，即可求出答案。考虑 $g_i$ 的计算。

根据下降幂的定义 $F(x)=\sum_{i\geq 0}g_iC_x^i$，可以发现这是二项式反演，因此 $g_x=\sum_i(-1)^{x-i}F(i)C_x^i$。

注意到 $F(0),F(1),...,F(\sum a_i)$ 实际上只需要通过求出前 $a_i$ 轮后第一个数变成了 $1$，剩下的数还没变成 $0$ 的概率即可求出，而这个概率只需要算当前每个数满足条件的概率。对于后面的数，可以算出它之前不是 $0$ 这一轮变成 $0$ 的概率，从而算出它此时大于 $0$ 的概率。因此，可以在 $O(n*(\sum a_i))$ 的时间内算出 $F(0),F(1),...,F(\sum a_i)$

但此时直接FFT算答案不能接受，考虑一个 $F(i)$ 的贡献系数，设它为 $v_i$，则它等于：
$$
\sum_{j=i}^{\sum a_i}(-1)^{j-i}C_j^i*\frac1p*(\frac p{1-p})^j
$$

提取 $\frac 1p$ 后可以发现，一个 $j$ 对所有位置的贡献写成生成函数的形式相当于 $(\frac p{1-p}*(x-1))^j$，因此总的贡献系数的生成函数为：
$$
\sum_{i=0}^{\sum a_i}(\frac p{1-p}*(x-1))^i\\
=\frac{1-(\frac p{1-p}*(x-1))^{\sum a_i+1}}{1-(\frac p{1-p}*(x-1))}
$$
分子可以在 $O(\sum a_i)$ 的时间求出，因为分母是一次式，直接求逆即可。这里的复杂度为 $O(\sum a_i)$

复杂度 $O(n*(\sum a_i))$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 5005000
#define mod 998244353
int n,m,p=1,as,s[6][2],vl[N],fr[N],ifr[N],v2[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),s[i][1]=1ll*s[i][1]*pw(100,mod-2)%mod,p=1ll*p*(mod+1-s[i][1])%mod,m+=s[i][0];
	fr[0]=1;for(int i=1;i<=m+1;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[m+1]=pw(fr[m+1],mod-2);for(int i=m+1;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	int tp=pw(s[1][1],s[1][0]);
	for(int i=s[1][0]-1;i<=m;i++)vl[i]=1ll*fr[i]*ifr[s[1][0]-1]%mod*ifr[i+1-s[1][0]]%mod*tp%mod,tp=1ll*tp*(mod+1-s[1][1])%mod;
	for(int i=2;i<=n;i++)
	{
		int su=1,tp=pw(s[i][1],s[i][0]);
		for(int j=s[i][0];j<=m;j++)su=(su+mod-1ll*tp*fr[j-1]%mod*ifr[j-s[i][0]]%mod*ifr[s[i][0]-1]%mod)%mod,tp=1ll*tp*(mod+1-s[i][1])%mod,vl[j]=1ll*vl[j]*su%mod;
	}
	int rp=pw(p,mod-2),sp=1ll*p*pw(mod+1-p,mod-2)%mod,st=1,s1=pw(sp,m+1);
	for(int i=0;i<=m;i++)vl[i]=1ll*vl[i]*st%mod,st=1ll*st*rp%mod;
	for(int i=0;i<=m;i++)v2[i]=1ll*((m-i)&1?mod-1:1)*fr[m+1]%mod*ifr[i]%mod*ifr[m+1-i]%mod*s1%mod;
	v2[0]++;
	int t1=pw(1+sp,mod-2),t2=1ll*t1*sp%mod;
	for(int i=0;i<=m;i++)v2[i]=1ll*v2[i]*t1%mod;
	for(int i=1;i<=m;i++)v2[i]=(v2[i]+mod+1ll*v2[i-1]*t2%mod)%mod;
	for(int i=0;i<=m;i++)as=(as+1ll*v2[i]*vl[i]%mod*sp%mod*rp)%mod;
	printf("%d\n",as);
}
```

##### #180. Island Manager

###### Problem

给一个 $n\times m$ 的矩阵，每个位置有一个数 $v_{i,j}$，所有数互不相同。再给出一个长度为 $\min(n,m)$ 的序列 $s$。

定义对一个矩阵做 $k$ 阶划分的结果为：

如果矩阵为空，则结束。否则首先在矩阵中选出一个数，如果 $s_k=0$ ，则选最小的数，否则选最大的数。

将这个数所在的行和列删去，得到四个可能为空的矩阵。对每个矩阵做 $k+1$ 阶划分。

求出划分的情况。为了方便，定义一个位置的权值 $s_{i,j}$ 为：

如果这个位置在划分时没有被选中，则 $s_{i,j}=0$。

否则，设它是在第 $k$ 轮被选中，则 $s_{i,j}$ 为 $k-1$ 轮时，它所在的矩形选出的数的值。

输出所有的 $s_{i,j}$

$n\times m\leq 4\times 10^6$

$0.5s,80MB$

###### Sol

考虑暴力遍历找最大最小，可以证明暴力的复杂度不会超过 $O((nm)^{1.5})$。但因为时限很小+cache原因这样过不去。

考虑另外一种暴力，枚举每一行，然后相当于求这一行上的区间最大最小。考虑这样整个过程中询问区间max/min的次数和。

设当前矩阵大小为 $n\times m$，注意到划分时会删去 $n+m-1$ 个元素，而划分时只会询问 $n$ 次，因此总的询问次数不会超过 $nm$。

因此现在只需要维护区间max/min即可。~~四毛子~~因为空间限制，考虑将序列按照 $\log n$ 大小分块，对于块之间ST表，每次询问散块部分暴力。这样的时间复杂度 $O(n+q\log n)$，空间复杂度 $O(n)$。且这样空间常数很小。

复杂度 $O(nm\log {nm})$，空间复杂度 $O(n)$，大概只需要48~64MB

~~io优化不规范，TLE两行泪~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
namespace IO{
	const int sz=1<<17;
	char a[sz+5],b[sz+5],*p1=a,*p2=a,*t=b,p[105];
	inline char gc(){
		return p1==p2?(p2=(p1=a)+fread(a,1,sz,stdin),p1==p2?EOF:*p1++):*p1++;
	}
	template<class T> void gi(T& x){
		x=0; char c=gc();
		for(;c<'0'||c>'9';c=gc());
		for(;c>='0'&&c<='9';c=gc())
			x=x*10+(c-'0');
	}
	inline void flush(){fwrite(b,1,t-b,stdout),t=b; }
	inline void pc(char x){*t++=x; if(t-b==sz) flush(); }
	template<class T> void pi(T x,char c='\n'){
		if(x==0) pc('0'); int t=0;
		for(;x;x/=10) p[++t]=x%10+'0';
		for(;t;--t) pc(p[t]); pc(c);
	}
	struct F{~F(){flush();}}f; 
}
using IO::gi;
using IO::pi;
#define N 4000040
#define M 200020
int n,m,k,s[3050],v[N],sz=20,su,lg[M],vl[N];
int smn[M][18],smx[M][18];
int sth[N];
int qmin(int x,int y){return v[x]<v[y]?x:y;}
int qmax(int x,int y){return v[x]>v[y]?x:y;}
void init()
{
	k=n*m;su=(k-1)/sz+1;
	for(int i=2;i<=su;i++)lg[i]=lg[i>>1]+1;
	for(int i=1;i<=su;i++)
	{
		int as=(i-1)*sz+1,ls=as;
		for(int j=1;j<sz;j++)as=qmin(as,ls+j);
		smn[i][0]=as;
		int fr=(i-1)*sz+1;
		for(int j=1;j<=sz;j++)
		{
			fr=qmin(fr,(i-1)*sz+j);
			sth[(i-1)*sz+j]|=fr-(i-1)*sz;
		}
		fr=(i-1)*sz+sz;
		for(int j=sz;j>=1;j--)
		{
			fr=qmin(fr,(i-1)*sz+j);
			sth[(i-1)*sz+j]|=(fr-(i-1)*sz)<<5;
		}
	}
	for(int j=1;j<=17;j++)
	for(int i=1;i+(1<<j)-1<=su;i++)smn[i][j]=qmin(smn[i][j-1],smn[i+(1<<j-1)][j-1]);
	for(int i=1;i<=su;i++)
	{
		int as=(i-1)*sz+1,ls=as;
		for(int j=1;j<sz;j++)as=qmax(as,ls+j);
		smx[i][0]=as;
		int fr=(i-1)*sz+1;
		for(int j=1;j<=sz;j++)
		{
			fr=qmax(fr,(i-1)*sz+j);
			sth[(i-1)*sz+j]|=(fr-(i-1)*sz)<<10;
		}
		fr=(i-1)*sz+sz;
		for(int j=sz;j>=1;j--)
		{
			fr=qmax(fr,(i-1)*sz+j);
			sth[(i-1)*sz+j]|=(fr-(i-1)*sz)<<15;
		}
	}
	for(int j=1;j<=17;j++)
	for(int i=1;i+(1<<j)-1<=su;i++)smx[i][j]=qmax(smx[i][j-1],smx[i+(1<<j-1)][j-1]);
}
int querymn(int l,int r)
{
	int s1=(l-1)/sz+1,s2=(r-1)/sz+1;
	if(s1==s2)
	{
		int as=l;
		for(int i=l+1;i<=r;i++)as=qmin(i,as);
		return as;
	}
	int as=qmin((s1-1)*sz+((sth[l]>>5)&31),(s2-1)*sz+(sth[r]&31));
	s1++;s2--;
	if(s1>s2)return as;
	int tp=lg[s2-s1+1],as1=qmin(smn[s1][tp],smn[s2-(1<<tp)+1][tp]);
	return qmin(as,as1);
}
int querymx(int l,int r)
{
	int s1=(l-1)/sz+1,s2=(r-1)/sz+1;
	if(s1==s2)
	{
		int as=l;
		for(int i=l+1;i<=r;i++)as=qmax(i,as);
		return as;
	}
	int as=qmax((s1-1)*sz+((sth[l]>>15)&31),(s2-1)*sz+((sth[r]>>10)&31));
	s1++;s2--;
	if(s1>s2)return as;
	int tp=lg[s2-s1+1],as1=qmax(smx[s1][tp],smx[s2-(1<<tp)+1][tp]);
	return qmax(as,as1);
}
void solve(int l1,int r1,int l2,int r2,int ls,int k)
{
	if(l1>r1||l2>r2)return;
	int as=(l1-1)*m+l2;
	if(s[k])for(int i=l1;i<=r1;i++)as=qmax(as,querymx((i-1)*m+l2,(i-1)*m+r2));
	else for(int i=l1;i<=r1;i++)as=qmin(as,querymn((i-1)*m+l2,(i-1)*m+r2));
	vl[as]=ls;
	int sx=(as-1)/m+1,sy=(as-1)%m+1;
	solve(l1,sx-1,l2,sy-1,v[as],k+1);
	solve(l1,sx-1,sy+1,r2,v[as],k+1);
	solve(sx+1,r1,l2,sy-1,v[as],k+1);
	solve(sx+1,r1,sy+1,r2,v[as],k+1);
}
int main()
{
	gi(n);gi(m);
	for(int i=1;i<=n&&i<=m;i++)gi(s[i]);
	for(int i=1;i<=n*m;i++)gi(v[i]);
	init();
	solve(1,n,1,m,0,1);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)pi(vl[(i-1)*m+j],j==m?'\n':' ');
}
```

##### #184. 基础矩阵乘法练习题

###### Problem

给定 $r$，考虑 $r$ 阶矩阵下的操作，所有运算都在模 $998244353$ 下进行。你有一个矩阵 $x$。现在给出 $n$ 个操作，操作有两种类型：

1. 给一个可逆矩阵 $a$ ，用 $a$ 右乘 $x$。
2. 给出 $a,b,c$ ，保证 $c$ 非零，将 $x_{a,b}$ 乘以 $c$。

有 $q$ 个询问，每次给出 $x$ 的初始值和 $l,r$，求出对 $x$ 依次执行操作 $l,l+1,...,r$ 后 $x$ 的值。

$n\leq 10^4,q\leq 10^5,r\leq 5$

$4s,512MB$

###### Sol

考虑将 $r\times r$ 的矩阵看成 $1\times r^2$ 的矩阵，则此时可以发现：

操作 $1$ 后每个位置的值都是之前的值的线性组合，因此可以看成给矩阵右乘一个 $r^2\times r^2$ 的矩阵。显然操作 $2$ 也可以看成右乘一个 $r^2\times r^2$ 的矩阵。

因为给出矩阵可逆/乘数非零，可以发现得到的两种矩阵都是可逆矩阵。

问题可以看成连续乘上一个区间的矩阵。设矩阵为 $A_1,A_2,...,A_n$，因为矩阵可逆，可以发现 $A_l*...*A_r=A_l*...*A_n*A_n^R*A_{n-1}^R*...*A_{r+1}^R$

因此，求出 $B_i=A_i*A_{i+1}*...*A_n,C_i=A_n^R*A_{n-1}^R*...*A_{r+1}^R$，则答案为 $x*B_l*C_{r+1}$，单次询问顺序做这个乘法的复杂度即为 $O(r^4)$

复杂度 $O(nr^6+qr^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 25
#define M 10005
#define mod 998244353
int n,m,r,a,b,c,d;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct mat{int st[N][N];mat(){for(int i=0;i<25;i++)for(int j=0;j<25;j++)st[i][j]=0;}}tp[M],rv[M],fu[M];
struct sth{int st[N];sth(){for(int i=0;i<25;i++)st[i]=0;}}as;
mat mul(mat a,mat b)
{
	mat c;
	for(int i=0;i<r*r;i++)
	for(int k=0;k<r*r;k++)if(a.st[i][k])
	for(int j=0;j<r*r;j++)
	c.st[i][j]=(c.st[i][j]+1ll*a.st[i][k]*b.st[k][j])%mod;
	return c;
}
mat inv(mat a)
{
	mat b;
	for(int i=0;i<r*r;i++)b.st[i][i]=1;
	for(int i=0;i<r*r;i++)
	{
		int fg=i;
		for(int j=r*r-1;j>=i;j--)if(a.st[j][i])fg=j;
		for(int j=0;j<r*r;j++)swap(a.st[i][j],a.st[fg][j]),swap(b.st[i][j],b.st[fg][j]);
		for(int j=0;j<r*r;j++)if(j!=i)
		{
			int tp=1ll*pw(a.st[i][i],mod-2)*a.st[j][i]%mod*(mod-1)%mod;
			for(int k=0;k<r*r;k++)a.st[j][k]=(a.st[j][k]+1ll*tp*a.st[i][k])%mod,b.st[j][k]=(b.st[j][k]+1ll*tp*b.st[i][k])%mod;
		}
	}
	for(int i=0;i<r*r;i++){int tp=pw(a.st[i][i],mod-2);for(int j=0;j<r*r;j++)b.st[i][j]=1ll*b.st[i][j]*tp%mod;}
	return b;
}
sth mul(sth a,mat b)
{
	sth as;
	for(int i=0;i<r*r;i++)for(int j=0;j<r*r;j++)as.st[j]=(as.st[j]+1ll*a.st[i]*b.st[i][j])%mod;
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&m,&r);
	for(int i=0;i<r*r;i++)tp[m+1].st[i][i]=rv[m+1].st[i][i]=1;
	for(int i=1;i<=m;i++)
	{
		scanf("%d",&a);
		if(a==1)
		{
			for(int j=0;j<r;j++)
			for(int k=0;k<r;k++)
			{
				scanf("%d",&b);
				for(int l=0;l<r;l++)
				fu[i].st[l*r+j][l*r+k]=b;
			}
		}
		else
		{
			scanf("%d%d%d",&b,&c,&d);
			for(int j=0;j<r*r;j++)fu[i].st[j][j]=(j==b*r+c-r-1?d:1);
		}
	}
	for(int i=m;i>=1;i--)tp[i]=mul(fu[i],tp[i+1]),rv[i]=mul(rv[i+1],inv(fu[i]));
	while(n--)
	{
		scanf("%d%d",&b,&c);
		for(int i=0;i<r*r;i++)scanf("%d",&as.st[i]);
		as=mul(as,tp[b]);as=mul(as,rv[c+1]);
		for(int i=0;i<r*r;i++)printf("%d ",as.st[i]);
		printf("\n");
	}
}
```

##### #200. 小 Z 的摸鱼计划

###### Problem

交互题

有一个 $n$ 个点 $m$ 条边的简单连通图，每个点上有一个数字，它们构成一个 $n$ 阶排列 $p$。

定义一次交换为选择一条边，交换边相邻两个点的数字。

定义一个排列的代价为还原排列用的最少交换次数。

现在两人进行游戏，两人轮流进行操作：

第一个人操作时，可以选择任意两个点 $i,j$ ，交换 $p_i,p_j$。在操作后，第一个人可以选择停止游戏或者继续。

第二个人操作时，可以选择有边相连的点 $i,j$，交换 $p_i,p_j$。第二个人不能停止游戏。

第一个人希望最小化排列代价，第二个人希望最大化排列代价。求出双方都使用最优策略时排列最后的代价。

你需要实现程序模拟第一个人的操作，交互器会使用第二个人的一种最优策略。游戏的过程会以交互的形式进行。

在你选择结束时，你需要给出一种达到你给出的代价的交换方案。

交互轮数不能超过 $T$。

$n,m\leq 10^5,T=10^5+1$

$3s,512MB$

###### Sol

考虑如下第一个人的策略：

考虑一种将排列还原的方案，它一定可以被表示为若干步，每一步交换两个数字。

按照这个方式进行每一步操作，假设使用了 $k$ 步，则操作结束后，第二个人只进行了 $\max(k-1,0)$ 次交换。

此时把第一个人的操作看成改变数的数值，第二个人的操作看成交换数的位置。则可以发现此时每个数的值等于它原先所在的位置。因此只需要倒着做第二个人的操作，就一定能还原排列。

因此第一个人可以让最后的代价不大于 $\max(k-1,0)$。

考虑如下第二个人的策略：

注意到通过交换还原排列的操作数一定不小于不考虑边的限制交换还原的操作数，即点数减去看成置换后环的数量。

显然，任意一次交换只能让环的数量最多增加 $1$。因此第一个人每一步最多让环的数量 $+1$。

但如果交换的两个位置属于不同的环，则这一步交换一定让环的数量 $-1$。

可以发现，只要当前有两个环，则一定能在图上找到一条边，使得这条边的两个端点属于不同的环。

因此，设初始时环的数量为 $n-k$，则第二个人一定可以让自己操作后环的数量不大于 $n-k$。

因此此时第一个人操作一次后环数量不大于 $n-k+1$，因此代价不小于 $\max(k-1,0)$ 。

通过上面两个策略可以得到，答案即为 $\max(k-1,0)$ ，其中 $k$ 为每次交换任意两个位置，还原排列的操作数。使用上面第一个人的策略，记录当前每个数的位置即可。

复杂度 $O(n)$ 

###### Code

```cpp
#include"graph.h"
using namespace std;
#define N 104000
int vis[N],s[N][2],ct,id[N];
vector<int> as;
pair<pair<int, int>, vector<int> > Solve(int n, int m, int T, vector<int> U, vector<int> V, vector<int> p, int subtask)
{
	for(int i=0;i<n;i++)if(!vis[i])
	{
		for(int j=i;!vis[j];j=p[j])
		{
			vis[j]=1;
			if(!vis[p[j]])s[++ct][0]=j,s[ct][1]=p[j];
		}
	}
	Answer(ct?ct-1:0);
	for(int i=0;i<n;i++)id[p[i]]=i;
	for(int i=1;i<ct;i++)
	{
		int st=Swap(id[s[i][0]],id[s[i][1]]);
		swap(p[id[s[i][0]]],p[id[s[i][1]]]);swap(id[s[i][0]],id[s[i][1]]);
		as.push_back(st);
		swap(p[U[st]],p[V[st]]),swap(id[p[U[st]]],id[p[V[st]]]);
	}
	reverse(as.begin(),as.end());
	return make_pair(make_pair(id[s[ct][0]],id[s[ct][1]]),as);
}
```

##### #204. tree & prime

###### Problem

有一棵 $n$ 个点的树，每个点上有一个概率 $p_i$。

你有一个计数器 $k$，你经过一个点 $x$ 的时候，有 $p_x$ 的概率让 $k$ 加一。

对于每个点 $i$ 求出，初始计数器 $k=0$，你从 $1$ 沿最短路径走到 $i$ ，此时计数器 $k$ 为质数的概率。答案模 $998244353$

$n\leq 10^5$

$2s,512MB$

###### Sol1

考虑树分块，按照块内深度不超过 $S$ 进行分块，选出关键点，对于走到一个点，一定可以看成走到某个关键点，再向下走不超过 $S$ 步。

对于每个关键点求出，走到这个关键点上时，计数器 $k$ 满足加上 $0,1,2,...,n$ 后为质数的概率。考虑从上一个关键点处转移，先处理出从上个关键点走过来这一段计数器增加 $0,1,2,...,S$ 的概率，然后使用FFT即可求出这个点的概率。这部分复杂度为 $O(nS+\frac nS*n\log n)$

然后考虑计算答案，从每个关键点开始向下dfs，可以求出每个点走到祖先第一个关键点这一段计数器增加 $0,1,...,S$ 的概率，然后即可直接算答案。这部分复杂度 $O(nS)$

因此取 $S=O(\sqrt{n\log n})$，复杂度为 $O(n\sqrt{n\log n})$，卡常后 $1.5s$

###### Sol2

首先考虑一条链的做法，考虑分治，取出中点 $mid$，使用分治FFT预处理+FFT即可求出走到 $mid$ 时，计数器 $k$ 满足加上 $0,1,2,...,n$ 后为质数的概率。

此时即可算出 $mid$ 的答案，然后考虑分别计算两侧。注意到此时每一段长度不超过一半，因此可以只保留之前求出的 $0,1,2,...,\frac n2$ 项。然后对两侧分别做。

在分治前预处理分治FFT，则每次FFT前直接使用之前的结果即可。可以发现这样的复杂度为 $O(n\log^2 n)$

然后考虑树的情况，对树进行长链剖分。对于一条链，只要对于链顶求出了上面那个值，则使用之前的做法即可求出这条链的答案。

考虑求出了这条链后，怎么处理这条链外的儿子的答案。

设这条链长度为 $m$，这个儿子所在的位置深度为 $k$。则根据长链剖分的性质，子树内深度不超过 $m-k$。

考虑找到最大的 $s$ 满足 $m-\frac m{2^s}\leq k$。此时可以找到分治时 $[m-\frac m{2^s},m]$ 这个区间。

之前分治时求出了走到区间左端点时，计数器 $k$ 满足加上 $0,1,...$ 后为质数的概率。使用分治FFT的结果，可以求出从左端点走到深度为 $k$ 的点的这段路径上计数器增加 $0,1,...$ 的概率。通过这两个即可求出走到这个点时，计数器 $k$ 满足加上 $0,1,...$ 后为质数的概率。

设区间长度为 $l$，则这样的复杂度为 $O(l\log^2 l)$。根据分治的性质， $l$ 不会超过二倍深度。因此这部分复杂度不超过 $O(n\log^2 n)$

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 132001
#define mod 998244353
int n,head[N],cnt,a,b,v[N],g[2][N*2],rev[N*2],s1[N],s2[N],ntt[N],vl[N],is[N],fr[N],as[N],v1[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d)
{
	for(int i=2;i<=1<<d;i<<=1)for(int j=0;j<i;j++)rev[j+i]=(rev[(j>>1)+i]>>1)|(j&1?i>>1:0);
	for(int t=0;t<2;t++)
	for(int i=2;i<=1<<d;i<<=1)
	{
		int tp=pw(3,(mod-1)+(t*2-1)*(mod-1)/i),vl=1;
		for(int j=0;j<i;j++)g[t][i+j]=vl,vl=1ll*vl*tp%mod;
	}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=i,tp=l;j<i+(l>>1);j++,tp++)
	{
		int v1=ntt[j],v2=1ll*ntt[j+(l>>1)]*g[t][tp]%mod;
		ntt[j]=(v1+v2)%mod;
		ntt[j+(l>>1)]=(v1-v2+mod)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> tp[N],st[N];
void dfs0(int u,int fa)
{
	vl[u]=1;v1[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),vl[u]=max(vl[u],vl[ed[i].t]+1),v1[u]=max(v1[u],v1[ed[i].t]+1);
	if(vl[u]>=1000)is[u]=1,vl[u]=0;
}
void dfs1(int u,int fa)
{
	if(is[u])
	{
		if(u==1)
		{
			for(int i=0;i<=n;i++)tp[u].push_back(1);
			tp[u][0]=tp[u][1]=0;
			for(int i=2;i<=n;i++)for(int j=i*2;j<=n;j+=i)tp[u][j]=0;
		}
		else
		{
			int l=1;while(l<=v1[fr[u]]+1000)l<<=1;
			for(int i=0;i<l;i++)s1[i]=s2[i]=0;
			for(int i=0;i<=v1[fr[u]];i++)s1[i]=tp[fr[u]][i];
			for(int i=0;i<st[u].size();i++)s2[1000-i]=st[u][i];
			dft(l,s1,1);dft(l,s2,1);for(int i=0;i<l;i++)s1[i]=1ll*s1[i]*s2[i]%mod;dft(l,s1,0);
			for(int i=0;i<=v1[u];i++)tp[u].push_back(s1[i+1000]);
		}
		fr[u]=u;st[u].clear();st[u].push_back(1);
	}
	for(int i=0;i<st[u].size();i++)as[u]=(as[u]+1ll*st[u][i]*tp[fr[u]][i]%mod*(mod+1-v[u]))%mod;
	for(int i=0;i<st[u].size();i++)as[u]=(as[u]+1ll*st[u][i]*tp[fr[u]][i+1]%mod*v[u])%mod;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		fr[ed[i].t]=fr[u];
		for(int j=0;j<=st[u].size();j++)st[ed[i].t].push_back(0);
		for(int j=0;j<st[u].size();j++)st[ed[i].t][j]=(st[ed[i].t][j]+1ll*st[u][j]*(mod+1-v[u]))%mod,st[ed[i].t][j+1]=(st[ed[i].t][j+1]+1ll*st[u][j]*v[u])%mod;
		dfs1(ed[i].t,u);
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	for(int i=1;i<=n;i++)scanf("%d%d",&a,&b),v[i]=1ll*a*pw(b,mod-2)%mod;
	init(17);dfs0(1,0);is[1]=1;dfs1(1,0);
	for(int i=1;i<=n;i++)printf("%d\n",as[i]);
}
```

##### #208. function

###### Problem

定义 $f(x)$ 为满足下列条件的正整数 $y$ 个数：

1. $y^3\equiv 1(\bmod x)$
2. $1<y<x$

给出 $n,k$，求出满足 $1\leq i\leq n,f(i)=k$ 的正整数 $i$ 个数。

$n\leq 2\times 10^{10}$

$2s,512MB$

###### Sol

考虑计算 $f(x)+1$，即第二个限制变为 $0<y<x$ 时的答案。

设 $x=\prod p_i^{q_i}$，则显然 $\forall i,y^3\equiv 1(\bmod p_i^{q_i})$，且只要右侧成立左侧一定成立。

根据CRT，可以分别枚举 $y\bmod p_1^{q_1},y\bmod p_2^{q_2},...$，因此只需求出 $p_1^{q_1},p_2^{q_2},...$ 时的答案，再相乘就得到了 $x$ 的答案。

考虑计算 $p^q$，分情况讨论：

1. $p>2$，此时 $\bmod p^q$ 存在原根 $g$ ，且阶为 $(p-1)p^{q-1}$

此时如果 $3|(p-1)p^{q-1}$，则显然存在三个解 $g^{t*\frac13\phi(p^q)},t\in\{0,1,2\}$。可以说明只存在这三个解。

如果 $3\not|(p-1)p^{q-1}$，则反证可以说明只存在一个解 $1$。

因此，此时有三个解当且仅当 $p\equiv 1(\bmod 3)$ 或者 $p=3,q>1$。

2. $p=2$

解显然是奇数，此时如果解不为 $1$，设解为 $2^k*t+1$，其中 $0<k<q,t\equiv 1(\bmod 2)$。

则它的立方为 $t^32^{3k}+3t^22^{2k}+3t2^k+1=2^{k+1}(t^32^{2k-1}+3t^22^{k-1}+2t)+2^k+1$，可以发现它 $\bmod 2^q$ 时一定 $2^k$ 这一位为 $1$，因此矛盾。

所以此时只有一组解。

因此，有三个解当且仅当 $p\equiv 1(\bmod 3)$ 或者 $p=3,q>1$。否则只有一个解。

如果 $k+1$ 不是 $3$ 的次幂则无解。设 $k+1=3^t$，则相当于求将 $n$ 分解后正好有 $t$ 个 $p^q$ 满足条件的数个数。

设 $g(n)$ 为一个积性函数，满足：

1. $g(p^q)=x$ 当且仅当 $p\equiv 1(\bmod 3)$ 或者 $p=3,q>1$。
2. 若不满足上面的条件，则 $g(p^q)=1$

可以发现，$\forall n,\exists s\in\N,g(n)=x^s$，且此时一定满足 $f(n)+1=3^s$，因此求出 $g$ 的前缀和即可。

考虑 min_25 筛，第二部分直接做即可，第一部分因为只保留质数 $g$ 也不是完全积性不能直接做。

但注意到求的是所有 $\lfloor\frac ni\rfloor$ 的前缀中 $\bmod 3=1$ 的质数个数，考虑函数 $h(x)$：

$$
h(3t)=0,h(3t+1)=1,h(3t+2)=-1
$$

可以发现它满足完全积性，且因为$\mod 3=0$ 的质数只有一个，求出 $h$ 的前缀和后，再求出前缀质数个数即可求出 $\bmod 3=1$ 的质数个数。这两部分都可以直接筛出来。

复杂度 $O(\log k*n^{1-\omega})$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 4000050
#define ll long long
ll n,k,s;
int ch[N],pr[N],ct,p,su;
struct sth1{ll x,y;}f[N],s1[N];
sth1 operator +(sth1 a,sth1 b){return (sth1){a.x+b.x,a.y+b.y};}
sth1 operator -(sth1 a,sth1 b){return (sth1){a.x-b.x,a.y-b.y};}
sth1 operator *(sth1 a,sth1 b){return (sth1){a.x*b.x+a.y*b.y,a.x*b.y+a.y*b.x};}
sth1 fp(int p){return (sth1){p%3==1,p%3==2};}
void prime(int n)
{
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++ct]=i;
		for(int j=1;j<=ct&&1ll*i*pr[j]<=n;j++)
		{
			ch[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
	for(int i=1;i<=ct;i++)s1[i]=s1[i-1]+fp(pr[i]);
}
int getid(ll x){return x>p?su-n/x+1:x;}
ll gettid(int x){return x<=p?x:n/(su-x+1);}
void solve1()
{
	su=2*p-(1ll*p*p==n);
	for(int i=1;i<=su;i++){ll tp=gettid(i);f[i]=(sth1){(tp-1)/3,(tp+1)/3};}
	for(int i=1;i<=ct;i++)
	for(int j=su;j>=1;j--)
	{
		ll tp=gettid(j);
		if(tp<1ll*pr[i]*pr[i])break;
		f[j]=f[j]-fp(pr[i])*(f[getid(tp/pr[i])]-s1[i-1]);
	}
}
struct sth2{ll v[8];sth2(){memset(v,0,sizeof(v));}};
sth2 operator +(sth2 a,sth2 b){for(int i=0;i<8;i++)a.v[i]+=b.v[i];return a;}
sth2 operator -(sth2 a,sth2 b){for(int i=0;i<8;i++)a.v[i]-=b.v[i];return a;}
sth2 mulp(sth2 a,int p,int k){if(p%3==1||(p==3&&k>1)){for(int i=6;i>=0;i--)a.v[i+1]=a.v[i];a.v[0]=0;}return a;}
sth2 doit(ll n){sth2 as;as.v[0]=f[getid(n)].y,as.v[1]=f[getid(n)].x;if(n>=3)as.v[0]++;return as;}
sth2 solve(ll m,int k)
{
	sth2 s1;
	if(m<pr[k]||k>ct)return s1;
	s1=doit(m)-doit(pr[k-1]);
	for(int i=k;i<=ct&&1ll*pr[i]*pr[i]<=m;i++)
	for(ll j=1,st=pr[i];st<=m;j++,st*=pr[i])
	{
		sth2 as=solve(m/st,i+1);
		if(j>1)as.v[0]++;
		s1=s1+mulp(as,pr[i],j);
	}
	return s1;
}
int main()
{
	scanf("%lld%lld",&n,&k);k++;
	while(k%3==0)k/=3,s++;
	if(k!=1){printf("0\n");return 0;}
	while(1ll*p*p<=n)p++;
	prime(p*2);solve1();
	sth2 as=solve(n,1);as.v[0]++;
	printf("%lld\n",as.v[s]);
}
```

##### #212. Yet Another Permutation Problem

###### Problem

给定 $n$，有一个初始为 $1,2,...,n$ 的排列。

你可以进行若干次操作，每次操作你可以选择一个位置，将这个位置从排列中拿出，并放在排列的开头或结尾。

对于每个 $k=0,1,2,...,n-1$，求出操作不超过 $k$ 次能得到的不同排列数量。答案对给定质数 $p$ 取模。

$n\leq 1000$

$1s,1024MB$

###### Sol

考虑一个排列中没有被操作过的数，显然这些数一定构成一个连续区间，且它们必须递增。

可以发现，此时让这些数不被操作，剩下的数每个数只操作一次，一定存在一种得到这个排列的方案。

因此操作的最小次数即为 $n$ 减去最长连续上升子段的长度，可以看成求最长连续上升子段长度大于等于 $1,2,...,n$ 的排列数量。

假设当前枚举了所有的极长上升子段长度 $l_1,l_2,...,l_m$，考虑计算这样的排列数。

容斥有哪些段中间不满足前面的结尾大于后面的开头，则相当于合并 $i$ 次相邻的两个段，容斥系数为 $(-1)^i$。

设此时每一段长度为 $l'_1,l'_2,...,l'_k$，因为容斥后段中间不再有限制，段内严格递增，因此方案数即为分成大小为 $l'_1,l'_2,...,l'_k$ 的集合的方案数，为 $\frac{n!}{\prod l'_i!}$

考虑计算在最长上升子段不超过 $m$ 时，所有容斥后得到一个长度为 $k$ 的段的情况的容斥系数和，即：
$$
v_{m,k}=\sum_{a_1,...,a_l,\forall i,1\leq a_i\leq m}(-1)^{l-1}
$$
在上面的过程中，考虑先枚举容斥后的长度 $l'_1,l'_2,...,l'_k$，可以发现此时不同段内部情况是独立的，因此所有情况的容斥系数和即为每一段内的容斥系数和的乘积。因此最长上升子段不超过 $m$ 的排列数为：
$$
\sum_{l'_1,...,l'_k,\sum l'_i=n,l'_i>0}n!*\prod\frac{v_{m,l'_i}}{l'_i!}
$$
考虑计算 $-v_{m,k}$，即 $\sum_{a_1,...,a_l,\forall i,1\leq a_i\leq m}(-1)^{l}$，此时枚举最后一段，可以得到：
$$
v_{m,0}=1,(-v_{m,k})=\sum_{i=1}^m(-1)*(-v_{m,k-i})
$$

可以发现这相当于 $\sum_{i=0}^mv_{m,k-i}=0$，因此写成生成函数后相当于 $\frac 1{1+x+...+x^m}$，它等于 $\frac{1-x}{1-x^{m+1}}$。

直接展开可以发现， $v_{m,k(m+1)}=1,v_{m,k(m+1)+1}=-1,k\in\N$，其余项都是 $0$。因此有值的项只有 $O(\frac nm)$ 项。

此时考虑计算上面的答案，可以发现答案也可以写成dp，即
$$
dp_{m,0}=n!,dp_{m,i}=\sum_{j<i}dp_{m,j}*\frac{v_{m,i-j}}{(i-j)!}
$$

只转移有值的项，复杂度为 $O(\frac{n^2}m)$，因此对所有 $m$ 做一次的复杂度为 $O(n^2\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1013
int n,mod,dp[N],fr[N],ifr[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&mod);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	dp[0]=1;
	for(int i=n;i>=1;i--)
	for(int j=1;j<=n;j++)
	{
		dp[j]=0;
		for(int k=j-1;k>=0;k-=i)dp[j]=(dp[j]+1ll*dp[k]*ifr[j-k])%mod;
		for(int k=j-i;k>=0;k-=i)dp[j]=(dp[j]+mod-1ll*dp[k]*ifr[j-k]%mod)%mod;
		if(j==n)printf("%d\n",1ll*fr[n]*(mod+1-dp[n])%mod);
	}
}
```

##### #216. 大鱼洽水

###### Problem

给一个 $n$ 个点 $m$ 条边的仙人掌。你在上面随机游走，当你在一个点上时，你会随机选择一个出边走过去。

有 $c$ 个点为出口，你走到出口就会立刻停止。

求出你从点 $1,2,...,n$ 开始时，停止时期望经过的边数，答案模 $998244353$。

$n\leq 10^5,m\leq 1.5\times 10^5$，数据随机

$1s,512MB$

###### Sol

首先考虑树上的做法。

设点 $i$ 出发的答案为 $dp_i$，则设点 $i$ 的度数为 $d_i$，在 $i$ 不是出口时有 $dp_i=1+\frac 1d*\sum_{(i,v)\ exists}dp_v$。

设点 $i$ 的父亲为 $fa_i$，考虑将 $dp_i$ 表示为 $a*dp_{fa_i}+b$。对于一个点 $u$ ，如果求出了它的所有儿子的这种表示，此时可以发现 $u$ 上的方程中只包含 $dp_u,dp_{fa_u}$ 和常数，因此一定可以将 $dp_u$ 表示为 $a*dp_{fa_u}+b$ 的形式。

因此可以dfs，从下到上求出这个，最后可以通过根的方程求出根的 $dp$ 值，再dfs一次即可得到所有点的答案。

考虑仙人掌，只需要处理环的情况。 

设环上点依次为 $c_1,c_2,...,c_l$，其中 $c_1$ 为连接父亲的点。可以先求出其余点儿子的表示，此时可以发现点 $c_i(i\neq 1)$ 的方程中只包含 $c_{i-1},c_i,c_{i+1}$。

可以将 $dp_{c_2}$ 表示为 $a*dp_{c_1}+b*dp_{c_3}+c$ 的形式，然后考虑 $c_3$ 的方程可以将 $dp_{c_3}$ 表示为 $a*dp_{c_1}+b*dp_{c_4}+c$ 的形式。依次类推可以将 $dp_{c_i}$ 表示为 $a*dp_{c_1}+b*dp_{c_{i+1}}+c$ 的形式。然后通过最后一个点的方程，可以将 $dp_{c_l}$ 表示为 $a*dp_{c_1}+b$ 的形式，从而将环上所有点的 $dp$ 表示为 $a*dp_{c_1}+b$ 的形式。

如果环上遇到了出口，则可以对出口划分出的每一段分别做链的情况。

此时即可将 $dp_{c_1}$ 表示为 $a*dp_{fa_{c_1}}+b$ 的形式，因此建出圆方树后，在圆方树上使用类似之前的做法，在方点上做环的情况即可。

复杂度 $O(n\log mod)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 200050
#define mod 998244353
int n,m,k,a,is[N],s1[N],s[N][2],fa[N],f[N],dep[N],head[N],cnt,fr[N],f1[N],ct,d[N],v1[N],v2[N],v3[N],v4[N],v5[N];
vector<int> sn[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void dfs0(int u,int fa){dep[u]=dep[fa]+1,f[u]=fa;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);}
void doit(int x,int y,int f2)
{
	vector<int> s1,s2;
	while(x!=y)
	if(dep[x]>dep[y])f1[x]=f2,s1.push_back(x),x=f[x];
	else f1[y]=f2,s2.push_back(y),y=f[y];
	reverse(s1.begin(),s1.end());
	for(int i=0;i<s2.size();i++)s1.push_back(s2[i]);
	sn[f2]=s1;sn[x].push_back(f2);
}
void dfs1(int u,int fa)
{
	if(u<=n)
	{
		int s1=1,s2=pw(d[u],mod-2),s3=1;
		for(int i=0;i<sn[u].size();i++)
		{
			dfs1(sn[u][i],u);
			if(sn[u][i]<=n)s1=(s1-1ll*s2*v1[sn[u][i]]%mod+mod)%mod,s3=(s3+1ll*s2*v2[sn[u][i]])%mod;
			else
			{
				int lb=sn[sn[u][i]].front(),rb=sn[sn[u][i]].back();
				s1=(s1-1ll*s2*v1[lb]%mod+mod)%mod,s3=(s3+1ll*s2*v2[lb])%mod;
				s1=(s1-1ll*s2*v1[rb]%mod+mod)%mod,s3=(s3+1ll*s2*v2[rb])%mod;
			}
		}
		v1[u]=1ll*s2*pw(s1,mod-2)%mod;v2[u]=1ll*s3*pw(s1,mod-2)%mod;
		if(is[u])v1[u]=v2[u]=0;else fr[u]=f[u];
	}
	else
	{
		for(int i=0;i<sn[u].size();i++)dfs1(sn[u][i],u);
		int ls=-1,sz=sn[u].size();
		for(int i=0;i<sz;i++)
		{
			if(is[sn[u][i]])
			{
				if(ls==-1)continue;
				v1[sn[u][i-1]]=v3[i-1],v2[sn[u][i-1]]=v5[i-1];
				for(int j=i-2;j>=ls;j--)v1[sn[u][j]]=(v3[j]+1ll*v4[j]*v1[sn[u][j+1]])%mod,v2[sn[u][j]]=(v5[j]+1ll*v4[j]*v2[sn[u][j+1]])%mod;
				if(ls==0)for(int j=ls;j<i;j++)fr[sn[u][j]]=fa;
				ls=-1;
			}
			else
			{
				if(ls==-1)ls=i,v3[i]=(i==0)*v1[sn[u][i]],v4[i]=v1[sn[u][i]],v5[i]=v2[sn[u][i]];
				else
				{
					int tp=mod+1-1ll*v4[i-1]*v1[sn[u][i]]%mod;
					v3[i]=1ll*v1[sn[u][i]]*v3[i-1]%mod*pw(tp,mod-2)%mod;
					v4[i]=1ll*v1[sn[u][i]]*pw(tp,mod-2)%mod;
					v5[i]=(v2[sn[u][i]]+1ll*v5[i-1]*v1[sn[u][i]]%mod)*pw(tp,mod-2)%mod;
				}
			}
		}
		if(ls!=-1)
		{
			v1[sn[u][sz-1]]=(v3[sz-1]+v4[sz-1])%mod,v2[sn[u][sz-1]]=v5[sz-1];
			for(int j=sz-2;j>=ls;j--)v1[sn[u][j]]=(v3[j]+1ll*v4[j]*v1[sn[u][j+1]])%mod,v2[sn[u][j]]=(v5[j]+1ll*v4[j]*v2[sn[u][j+1]])%mod;
			for(int j=ls;j<sz;j++)fr[sn[u][j]]=fa;
		}
	}
}
void dfs2(int u){for(int i=0;i<sn[u].size();i++)v2[sn[u][i]]=(v2[sn[u][i]]+1ll*v1[sn[u][i]]*v2[u])%mod,dfs2(sn[u][i]);}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=k;i++)scanf("%d",&a),is[a]=1;
	for(int i=1;i<=m;i++)scanf("%d%d",&s[i][0],&s[i][1]),d[s[i][0]]++,d[s[i][1]]++;
	for(int i=1;i<=n;i++)fa[i]=i;
	for(int i=1;i<=m;i++)if(finds(s[i][0])!=finds(s[i][1]))adde(s[i][0],s[i][1]),s1[i]=1,fa[finds(s[i][0])]=finds(s[i][1]);
	dfs0(1,0);
	for(int i=1;i<=m;i++)if(!s1[i])doit(s[i][0],s[i][1],n+(++ct));
	for(int i=2;i<=n;i++)if(!f1[i])sn[f[i]].push_back(i);
	dfs1(1,0);
	for(int i=1;i<=n;i++)sn[i].clear();
	for(int i=1;i<=n;i++)if(fr[i])sn[fr[i]].push_back(i);
	for(int i=1;i<=n;i++)if(!fr[i])dfs2(i);
	for(int i=1;i<=n;i++)printf("%d\n",v2[i]);
}
```

##### #220. 大鱼治水

###### Problem

交互题

给一棵 $n$ 个点，以 $1$ 为根的有根树。对于一个点，你可以钦定一条它连向儿子的边为重边，也可以不钦定。

交互库首先给出这样的一棵树，你需要给出一个钦定的方案。随后交互库会进行 $q$ 次询问。

每次询问中，交互库会给出 $x$ ，你需要对钦定重边的方案进行调整，使得调整后 $x$ 到根的链上都是重边。在满足条件后，你可以再进行若干次调整。这部分调整后不需要仍然满足条件。

一次调整定义为：

1. 选择一个没有钦定重边的点，钦定一条它连向儿子的边为重边。
2. 选择一个钦定了重边的点，取消这个点的钦定。

单次询问的代价为两部分调整的次数和，你需要使得单次询问的代价不超过 $35$。

$n\leq 5\times 10^4,q\leq 5\times 10^5$

$3s,512MB$

###### Sol

考虑对树重链剖分，以这个方案为重边方案。每次询问暴力改轻边。可以发现，对于一条轻边，修改到满足条件和修改回去各需要两步，因此总共需要 $4$ 步，这样的次数为 $4*\log n=60$。

在此基础上，考虑不钦定一个点的重儿子的方式。可以发现，此时经过这个点时，一定前后共需要两次修改。因此这种方案轻儿子和重儿子都需要 $2$ 步。此时可以考虑，在重儿子大小过大的时候使用钦定重边的方案，否则不钦定。~~貌似两种方案的分解点就是黄金分割~~

考虑此时的最差步数，设 $dp_i$ 表示 $i$ 个点的子树最坏需要的步数，则有：
$$
dp_i=\max_{j\leq i-j-1}\min(\max(dp_j+4,dp_{i-j-1}),\max(dp_j+2,dp_{i-j-1}+2))
$$
可以发现此时 $dp_{50000}=40$。

考虑一个点其它的钦定方式：

>对于一个点，它可以钦定重儿子为重边，也可以不钦定。
>
>如果询问的点在重儿子子树内，则最多需要 $1$ 步钦定重边，然后可以不取消钦定，这部分需要最多 $1$ 步。
>
>如果询问的点不在重儿子子树内，则最多需要 $2$ 步钦定到这个儿子，最后用 $1$ 步取消钦定即可。这部分最多需要 $3$ 步。
>
>因此这种方式重儿子需要 $1$ 步，轻儿子需要 $3$ 步。

在加入了这种方式后， $dp$ 可以写成：
$$
dp_i=\max_{j\leq i-j-1}\min(\max(dp_j+4,dp_{i-j-1}),\max(dp_j+3,dp_{i-j-1}+1),\max(dp_j+2,dp_{i-j-1}+2))
$$
可以发现 $dp_{50000}=35$，因此使用这种方式即可。

先打表出 $dp$ 的分界点，然后可以处理出每个点使用哪种方案。这时即可dfs预处理询问每个点时的操作，询问时直接回答即可。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include"river.h"
#include<vector>
#include<algorithm>
#define N 50050
using std::make_pair;
std::vector<std::pair<int,int> > s1[N],s2[N];
int n,head[N],cnt,sz[N],v1[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int magic[37]={0,3,3,5,7,9,13,17,23,31,41,55,73,97,129,171,227,301,399,529,701,929,1231,1631,2161,2863,3793,5025,6657,8819,11683,15477,20503,27161,35981,47665,63143};
int getdp(int x){return std::lower_bound(magic+1,magic+36,x+1)-magic-1;}
void dfs1(int u,int fa){sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t];}
void dfs2(int u,int fa)
{
	if(sz[u]==1)return;
	int mx=0,fr=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&sz[ed[i].t]>mx)mx=sz[ed[i].t],fr=ed[i].t;
	int ty=(getdp(mx)-getdp(sz[u]-mx-1))/2;
	if(ty<0)ty=0;if(ty>2)ty=2;if(mx==sz[u]-1)ty=2;
	if(ty==2)v1[u]=fr;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		s1[ed[i].t]=s1[u];s2[ed[i].t]=s2[u];
		if(ed[i].t==fr)
		{
			if(ty==1)s1[ed[i].t].push_back(make_pair(u,ed[i].t));
			else if(ty==0)s1[ed[i].t].push_back(make_pair(u,ed[i].t)),s2[ed[i].t].push_back(make_pair(u,0));
		}
		else
		{
			if(ty<=1)s1[ed[i].t].push_back(make_pair(u,ed[i].t)),s2[ed[i].t].push_back(make_pair(u,0));
			else s1[ed[i].t].push_back(make_pair(u,ed[i].t)),s2[ed[i].t].push_back(make_pair(u,fr));
		}
		dfs2(ed[i].t,u);
	}
}
void doit(int x,int y)
{
	if(v1[x]==y)return;
	if(!y)set(x,0);
	else if(v1[x])set(x,0),set(x,y);
	else set(x,y);
	v1[x]=y;
}
std::vector<int> init(int m,std::vector<int> fa)
{
	n=m;
	for(int i=2;i<=n;i++)adde(fa[i-2],i);
	dfs1(1,0);dfs2(1,0);
	std::vector<int> as;
	for(int i=1;i<=n;i++)as.push_back(v1[i]);
	return as;
}
void solve(int x)
{
	for(int i=0;i<s1[x].size();i++)doit(s1[x][i].first,s1[x][i].second);
	wait();
	for(int i=0;i<s2[x].size();i++)doit(s2[x][i].first,s2[x][i].second);
}
```

##### #224. minmex

###### Problem

给一个 $n$ 个点 $m$ 条边的无向连通图，每个点上有一个权值 $v_i$，所有 $v_i$ 构成一个 $0,1,...,n-1$ 的排列。

定义 $dis_i$ 为 $i$ 到 $1$ 的最短距离，定义点 $i$ 的 $vl_i$ 为所有 $1$ 到 $i$ 的最短路中，最短路上点权 $mex$ 的最小值，即：
$$
vl_i=\min_{(1,s_1,s_2,...,s_{dis_i-1},i)是原图的一条路径}mex(v_1,v_{s_1},v_{s_2},...,v_{s_{dis_i-1}},v_t)
$$
有 $q$ 次操作，每次给出 $i,j$，交换 $v_i,v_j$，每次操作后你需要输出 $\max_{i=1}^nvl_i$

$n,q\leq 5\times 10^5,m\leq 10^6$

$2s,512MB$

###### Sol

因为只考虑所有的最短路，考虑先求出每个点的 $dis$，对于一条边 $(u,v)$，如果 $dis_u=dis_v$，则删去这条边，否则设 $dis_v=dis_u+1$，只保留 $u\to v$ 的有向边。此时可以得到一个类似于最短路DAG的DAG，可以发现所有的最短路即为DAG上从 $1$ 出发的路径。因此变为一个DAG上的问题。

可以发现，$vl_i\geq k$ 当且仅当 $\forall v\in\{0,1,...,k\}$，满足任意一条 $1$ 到 $i$ 的路径上都存在权值 $v$。这相当于对于权值为 $v$ 的点 $x$，DAG上所有 $1$ 到 $i$ 的路径都经过 $x$。

显然这是支配树的定义，考虑求出以 $1$ 为根DAG的支配树。对DAG求支配树只需要按照拓扑序做，每次把一个点支配树上的父亲设为所有连向它的点在支配树上的LCA即可。复杂度 $O(n\log n)$。

此时对于点 $i$ ，$1$ 到 $i$ 的路径上必定经过的点即为支配树上 $i$ 到根的链上的所有点。因此 $vl_i$ 即为支配树上 $i$ 到根上的所有点的点权 $mex$。

可以发现 $\max_{i=1}^nvl_i$ 等于最大的 $x$ ，满足点权为 $0,1,...,x-1$ 的点都在某个点到根的路径上。

对于每个点求出它子树对应的dfs序区间 $[l_i,r_i]$，则显然若干个点都在某个点到根的路径上等价于这些点的子树dfs序区间有交。

因此问题变为每个值 $0,1,...,n-1$ 有一个区间，求一个最大的 $x$ 满足值为 $0,1,...,x-1$ 的区间有交，支持交换两个区间。考虑使用线段树维护，每个节点记录这个区间内所有区间的交即可。

复杂度 $O((n+q)\log n+m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
#define N 500500
int n,m,q,a,b,head[N],cnt,v[N],dis[N],f[N][19],tp[N],ct,lb[N],rb[N],dep[N];
vector<int> nt[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int LCA(int x,int y){if(!x)return y;if(dep[x]<dep[y])swap(x,y);for(int i=18;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=18;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
void bfs()
{
	queue<int> st;
	for(int i=1;i<=n;i++)dis[i]=-1;
	st.push(1);dis[1]=0;
	while(!st.empty())
	{
		int s=st.front(),tp=0;st.pop();
		for(int i=0;i<nt[s].size();i++)if(dis[nt[s][i]]==dis[s]-1)tp=LCA(tp,nt[s][i]);
		if(s>1)
		{
			adde(tp,s);dep[s]=dep[tp]+1;f[s][0]=tp;
			for(int i=1;i<=18;i++)f[s][i]=f[f[s][i-1]][i-1];
		}
		for(int i=0;i<nt[s].size();i++)if(dis[nt[s][i]]==-1)dis[nt[s][i]]=dis[s]+1,st.push(nt[s][i]);
	}
}
void dfs(int u,int fa){lb[v[u]]=++ct;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);rb[v[u]]=ct;}
struct segt{
	struct node{int x,l,r,l1,r1;}e[N*4];
	void pushup(int x){e[x].l1=max(e[x<<1].l1,e[x<<1|1].l1);e[x].r1=min(e[x<<1].r1,e[x<<1|1].r1);}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].l1=lb[l],e[x].r1=rb[l];return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int v)
	{
		if(e[x].l==e[x].r){e[x].l1=lb[v],e[x].r1=rb[v];return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=v)modify(x<<1,v);else modify(x<<1|1,v);
		pushup(x);
	}
	int query(int x,int l,int r)
	{
		if(e[x].l==e[x].r)if(max(l,e[x].l1)<=min(r,e[x].r1))return e[x].l;else return e[x].l-1;
		int l1=max(l,e[x<<1].l1),r1=min(r,e[x<<1].r1);
		if(l1<=r1)return query(x<<1|1,l1,r1);
		else return query(x<<1,l,r);
	}
}tr;
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),v[i]++;
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),nt[a].push_back(b),nt[b].push_back(a);
	bfs();dfs(1,0);tr.build(1,1,n);
	while(q--)
	{
		scanf("%d%d",&a,&b);
		swap(lb[v[a]],lb[v[b]]);swap(rb[v[a]],rb[v[b]]);
		tr.modify(1,v[a]);tr.modify(1,v[b]);swap(v[a],v[b]);
		printf("%d\n",tr.query(1,1,n));
	}
}
```

##### #228. Substring Concatenation

###### Problem

给定 $n$ 以及 $n$ 个字符串 $s_1,s_2,...,s_n$，字符集大小为 $m$。

你可以选择字符串 $t_1,t_2,...,t_n$，满足 $t_i$ 是 $s_i$ 的子串。此时令 $T=t_1+t_2+...+t_n$，即所有 $t$ 顺序拼接的结果。

求能得到的本质不同的非空 $T$ 数量，模 $998244353$。

$n\leq 10^5,m,\sum |s_i|\leq 3\times 10^5$

$2s,512MB$

###### Sol

考虑判断一个 $T$ 是否能得到。显然可以贪心划分，每次选一个 $T$ 的最长前缀使得这个前缀是 $s_i$ 的子串，然后让这个前缀为 $t_i$，删去这部分继续做。

这相当于对 $n$ 个串分别建SAM，将 $T$ 从第一个串的SAM开始匹配。如果走到一位时不存在转移边，则将还没有被匹配的部分放到下一个SAM上匹配。最后能被表示出当且仅当 $T$ 能在 $n$ 个SAM中匹配完。

记 $rt$ 为SAM的根，此时可以将 $n$ 个SAM连在一起，对于一个SAM节点 $x$ 和字符 $c$，如果 $ch_{x,c}$ 不存在，则匹配到这里时，如果下一个字符为 $c$，则接下来的操作是找到后面第一个能转移 $c$ 的SAM，然后走到那个SAM上的 $ch_{rt,c}$，因此可以将 $ch_{x,c}$ 设为这个值。此时可以看成在连接所有SAM后得到的DAG上求能匹配的字符串数量，可以直接dp。但这样的转移边数量为 $O(m\sum |s_i|)$

考虑一条向后的转移边的意义，相当于在后面匹配一个以字符 $c$ 开头的字符串。考虑从后往前考虑每个字符串，设 $vl_c$ 表示只考虑当前后若干个后缀时，这部分字符串能得到的以字符 $c$ 开头的字符串数量。考虑在开头加入一个字符串，计算这个字符串SAM上的 $dp$。

此时对于一个点 $x$ 和字符 $c$，如果 $ch_{x,c}$ 存在，则方案数为 $dp_{ch_{x,c}}$，否则方案数为 $vl_c$，记录 $vl$ 的整体和即可 $O(|s_i|)$ 的推出一个SAM上的 $dp$。

然后考虑更新 $vl$， 对于所有存在的 $ch_{rt,c}$，用 $dp_{ch_{rt,c}}$ 更新 $vl_c$ 即可。这样的更新只有 $O(|s_i|)$ 次。

可以将 $T$ 倒过来，这样相当于翻转串的顺序并翻转每个串，因此可以顺序做过去，不用记录每个串再倒序做。

复杂度 $O(\sum |s_i|\log m)$，复杂度在于map建SAM。

###### Code

```cpp
#include<cstdio>
#include<map>
#include<algorithm>
using namespace std;
#define N 100500
#define mod 998244353
int n,m,s[N*3],dp[N*3],su,as;
bool cmp(int a,int b);
struct SAM{
	map<int,int> ch[N*6];
	int len[N*6],fail[N*6],ls,ct,f[N*6],tp[N*6];
	void init()
	{
		for(int i=1;i<=ct;i++)ch[i].clear(),len[i]=fail[i]=0;
		ls=ct=1;
	}
	void ins(int c)
	{
		int st=++ct,s1=ls;ls=st;len[st]=len[s1]+1;
		while(s1&&!ch[s1][c])ch[s1][c]=st,s1=fail[s1];
		if(!s1)fail[st]=1;
		else
		{
			int nt=ch[s1][c];
			if(len[nt]==len[s1]+1)fail[st]=nt;
			else
			{
				int cl=++ct;len[cl]=len[s1]+1;
				ch[cl]=ch[nt];fail[cl]=fail[nt];fail[nt]=fail[st]=cl;
				while(s1&&ch[s1][c]==nt)ch[s1][c]=cl,s1=fail[s1];
			}
		}
	}
	void solve()
	{
		for(int i=1;i<=ct;i++)tp[i]=i;
		sort(tp+1,tp+ct+1,cmp);
		for(int i=ct;i>=1;i--)
		{
			int st=tp[i];
			f[st]=(su+1)%mod;
			for(map<int,int>::iterator it=ch[st].begin();it!=ch[st].end();it++)f[st]=(f[st]+f[it->second]-dp[it->first]+1ll*mod)%mod;
		}
		as=f[1];
		for(map<int,int>::iterator it=ch[1].begin();it!=ch[1].end();it++)su=(su-dp[it->first]+f[it->second]+1ll*mod)%mod,dp[it->first]=f[it->second];
	}
}sam;
bool cmp(int a,int b){return sam.len[a]<sam.len[b];}
int main()
{
	scanf("%d%d",&n,&m);
	while(n--)
	{
		scanf("%d",&m);
		for(int i=1;i<=m;i++)scanf("%d",&s[i]);
		sam.init();
		for(int i=m;i>=1;i--)sam.ins(s[i]);
		sam.solve();
	}
	printf("%d\n",(as-1)%mod);
}
```

##### #236. 分形图

###### Problem

题意经过大幅简化。

给出正整数 $k$，一个长度为 $2^k$ 的 `01` 序列 $s$，以及 $k$ 对整数 $l_i,r_i$，保证 $s_0=1$，询问有多少个长度为 $k$ 的整数序列 $v_{1,2,...,k}$ 满足：

1. $\forall i\in\{1,2,...,k\},v_i\in[l_i,r_i]$
2. $\forall d\geq 0,s[\sum_{i=1}^k2^{i-1}[\lfloor\frac{v_i}{2^d}\rfloor\equiv 1(\bmod 2)]]=1$

答案模 $998244353$。

多组数据，$T\leq 3,k\leq 11,0\leq l_i\leq r_i\leq 10^{18}$，$l_i,r_i$ 随机，

$4.5s,512MB$

###### Sol

给出的限制 $2$ 相当于对于每一个 $d$，设 $f_{i,j}$ 表示 $v_i$ 二进制表示第 $j$ 位的值，则 $f_{1,d},...,f_{i,d}$ 满足限制 $s[\sum_{i=1}^k2^{i-1}f_{i,d}]=1$。

考虑数位 $dp$，设 $dp_{i,S}$ 表示填了较高的 $i$ 位，当前高位满足要求，且当前所有数在高位与上下界的关系为 $S$ 的方案数。其中每个数与上下界的关系有三种：等于上界，等于下界或两个都不等于。这样的状态数为 $O(3^k\log n)$

然后考虑这一位上填的数，一种做法是直接枚举这一位再计算新的状态。通过dfs确定每一位，即可在dfs的过程中确定下一位的状态。这样的复杂度为 $O(T6^k\log v)$，无法通过。

但注意到在两个都不等于时，这一位实际上没有限制也不会影响状态，因此这一位可以任意填。因此考虑只枚举剩下的位的值，此时这一位的方案数为剩下的位固定，这些位任意，满足 $s_i=1$ 要求的方案数。如果求出了这个值，则可以少一种状态，复杂度变为 $O(T5^k\log v)$。

这样的限制可以看成一个 $k$ 位三进制数，其中某一位为 $0/1$ 表示原来的二进制数上这一位必须为 $0/1$，某一位为 $2$ 表示这一位任意，相当于求有多少个满足 $s_i=1$ 要求的二进制数满足这个的限制。可以发现，这可以用一个三进制下类似FWT的转移（每一位上 $a'_2=a_0+a_1+a_2,a'_0=a_0,a'_1=a_1$ 在 $O(k3^k)$ 的时间内求出。因此总复杂度 $O(T5^k\log v)$，因为数据随机显然跑不满。

本题极其卡常，不同实现间的速度差距可以达到 $5\sim 10$ 倍并且std跑了3.5s，因此需要极其注意常数。

卡常细节：

1. 用按照位数从小到大 $dp$ 的非递归方式代替记忆化搜索。
2. 滚动 $dp$ 数组。
3. 对dfs时的转移做预处理。
4. 因为一位上填的方案数只有 $2^k$，$2^k*mod$ 也不会超过 $2^{64}$，因此在dfs枚举每一位时，可以不取模，在外侧取模一次即可。

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<vector>
using namespace std;
#define ll long long
#define mod 998244353
int T,k,dp[178901],vl[178901],v2[178901][2],dp2[178901],c1[61],pw[12]={1,3,9,27,81,243,729,2187,6561,19683,59049,177147},f1[12],ct,v1[12][2],fg[12][3],t1[12][3][2],t2[12][3][2];
ll s[12][2];
vector<int> nt[2050];
char st[2050];
void init()
{
	memset(vl,0,sizeof(vl));
	for(int i=0;i<1<<k;i++)if(st[i]=='1')
	{
		int st=0;
		for(int j=1,vl=1;j<=k;j++,vl*=3)if((i>>j-1)&1)st+=vl;
		vl[st]=1;
	}
	for(int i=0;i<pw[k];i++)
	{
		v2[i][0]=v2[i][1]=0;
		for(int j=1;j<=k;j++)if(i/pw[j-1]%3==2)v2[i][0]+=2*pw[j-1];else v2[i][1]+=1<<j-1;
	}
	for(int l=3;l<=pw[k];l*=3)
	for(int i=0;i<pw[k];i+=l)
	for(int j=i;j<i+l/3;j++)
	vl[j+l*2/3]+=vl[j]+vl[j+l/3];
	for(int i=0;i<1<<k;i++)
	{
		nt[i].clear();
		for(int j=1;j<=k;j++)if(i&(1<<j-1))nt[i].push_back(j);
	}
	for(int i=1;i<=k;i++)
	{
		ll st=s[i][0]^s[i][1];
		f1[i]=0;
		while(st>=1)st>>=1,f1[i]++;
	}
}
ll dfs(int x,int s1,int s2)
{	
	if(x==ct+1)return 1ll*vl[s2]*dp[s1];
	return dfs(x+1,s1,s2)+dfs(x+1,s1+v1[x][0],s2+v1[x][1]);
}
void dfs0(int x,int v,int f1,int f2,int c1)
{
	if(x==k+1){ct=c1;dp2[v]=dfs(1,f1,f2)%mod;return;}
	for(int s=0;s<3;s++,v+=pw[x-1])
	{
		if(fg[x][s])v1[c1+1][0]=t2[x][s][0],v1[c1+1][1]=t2[x][s][1];
		dfs0(x+1,v,f1+t1[x][s][0],f2+t1[x][s][1],c1+fg[x][s]);
	}
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d",&k);
		for(int i=1;i<=k;i++)scanf("%lld%lld",&s[i][0],&s[i][1]);
		scanf("%s",st);
		init();
		for(int i=0;i<pw[k];i++)dp[i]=1;
		for(int d=0;d<60;d++)
		{
			memset(fg,0,sizeof(fg));
			memset(t1,0,sizeof(t1));
			memset(t2,0,sizeof(t2));
			for(int x=1;x<=k;x++)t1[x][2][0]=t1[x][2][1]=2*pw[x-1];
			for(int x=1;x<=k;x++)
			for(int v=0;v<2;v++)
			{
				int t3=v;
				if(f1[x]<=d+1)
				{
					if(f1[x]<=d)t1[x][v][1]=((s[x][1]>>d)&1)*pw[x-1];
					else fg[x][v]=1,t2[x][v][0]=pw[x-1],t2[x][v][1]=pw[x-1];
				}
				else if(!t3)
				{
					if((s[x][0]>>d)&1)t1[x][v][1]=pw[x-1];
					else fg[x][v]=1,t2[x][v][0]=2*pw[x-1],t2[x][v][1]=pw[x-1];
				}
				else
				{
					if((~s[x][1]>>d)&1)t1[x][v][0]=pw[x-1];
					else t1[x][v][0]=2*pw[x-1],fg[x][v]=1,t2[x][v][0]=-pw[x-1],t2[x][v][1]=pw[x-1];
				}
			}
			dfs0(1,0,0,0,0);
			for(int v=0;v<pw[k];v++)dp[v]=dp2[v];
		}
		printf("%d\n",dp[0]);
	}
}
```

##### #240. Communication Network

###### Problem

给一棵 $n$ 个点的树，称它的边集为 $E_1$。

对于一棵生成树，设它与 $E_1$ 的公共边数量为 $x$，则它的收益为 $x2^x$，求所有生成树的收益和，模 $998244353$。

即记 $n$ 个点所有的生成树的边集的集合为 $S$，求：
$$
(\sum_{E_2\in S}|E_1\cap E_2|2^{|E_1\cap E_2|})\bmod 998244353
$$
$n\leq 2\times 10^6$

$1s,512MB$

###### Sol

考虑枚举 $|E_1\cap E_2|=S$，计算边的交集为 $S$ 的生成树方案数。相当于计算有若干条边不能选的生成树数量。

再考虑容斥，枚举剩下的一个属于 $E_1$ 的边集 $T$，钦定这些边也在 $E_2$ 中，容斥系数为 $(-1)^{|T|}$，容斥后只需要 $S,T$ 中的边都在生成树中即可。

此时相当于钦定的边连出了若干个连通块，需要把它们连起来。设有 $k$ 个连通块，大小为 $s_1,...,s_k$ 且 $\sum s_i=n$，根据prufer序列的扩展，这时的方案数为 $n^{k-2}\prod_{i=1}^ks_i$。

考虑这些系数和选的边之间的组合意义。称在 $S$ 中的边为红边，在 $T$ 中的边为蓝边，剩下的边为白边。此时连通块为只考虑红蓝边后得到的连通块。

显然 $|S|*2^{|S|}$ 相当于在红边中任意选一条，然后每条红边再乘上 $2$ 的方案数。$(-1)^{|T|}$ 相当于每条蓝边乘上 $-1$ 的方案数。

$\prod_{i=1}^ks_i$ 可以看成在每个只考虑红蓝边的连通块中选一个点的方案数， $n^{k-2}$ 可以看成给每个选择的点乘上 $n$ 的方案数，最后乘以 $\frac 1{n^2}$，也可以看成每条白边乘上 $n$ 的方案数，最后乘以 $\frac 1n$。

因此问题可以看成给每条边染色，染三种颜色分别有 $2,-1,n$ 的方案数，然后再选择一条红边，在每个红蓝边的连通块中选一个点。求总的方案数。

此时可以考虑dp，设 $dp_{i,0/1,0/1}$ 表示 $i$ 的子树中进行染色，子树内选不选红边，根的红蓝边连通块是否选择了点，子树内其他连通块都选了点的方案数，直接转移即可。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2005000
#define mod 998244353
int n,a,b,head[N],cnt,dp[N][2][2],s[2][2],t[2][2];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dfs(int u,int fa)
{
	dp[u][0][0]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs(ed[i].t,u);
		for(int j=0;j<2;j++)for(int k=0;k<2;k++)s[j][k]=t[j][k]=0;
		for(int j=0;j<2;j++)for(int k=0;k<2;k++)
		{
			if(!k)t[j][1]=(t[j][1]+2ll*dp[ed[i].t][j][k])%mod;
			t[j][k]=(t[j][k]+dp[ed[i].t][j][k])%mod;
			if(j)t[0][k]=(t[0][k]+1ll*n*dp[ed[i].t][j][k])%mod;
		}
		for(int s1=0;s1<2;s1++)for(int s2=0;s2<2;s2++)
		for(int t1=0;s1+t1<2;t1++)for(int t2=0;s2+t2<2;t2++)s[s1+t1][s2+t2]=(s[s1+t1][s2+t2]+1ll*dp[u][s1][s2]*t[t1][t2])%mod;
		for(int j=0;j<2;j++)for(int k=0;k<2;k++)dp[u][j][k]=s[j][k];
	}
	for(int j=0;j<2;j++)dp[u][1][j]=(dp[u][1][j]+dp[u][0][j])%mod;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);printf("%d\n",1ll*pw(n,mod-2)*dp[1][1][1]%mod);
}
```

##### #248. Ghost domino

###### Problem

题面经过简化（我也想会写题目背景.jpg）

有一个 $n\times m$ 的矩阵，每个格子上有一个方向 $d_{i,j}$，代表如果你走到了这个格子，则你下一步必须向这个方向走。每个格子上还有一个权值 $v_{i,j}$。

对于一个正整数 $k$，你可以选择从某个位置开始，沿着格子上的方向走，经过 $k$ 个格子后停止。这条路径需要满足：

1. 不能出界。
2. 不能经过重复格子。

你希望经过的格子的 $v_{i,j}$ 的最大值最小，给出 $q$ 次询问，每次给定 $k$，求所有合法路径中，经过的格子的 $v_{i,j}$ 的最大值的最小值，或者报告不存在这样的路径。

$n,m\leq 1500,q\leq 5\times 10^5$

$3s,512MB$

###### Sol

每个点只有一条出边，因此题可以看成给一个基环内向树森林，你需要选一条经过 $k$ 个点且不重复经过点的路径，使得路径上的点权最大值最小。

考虑对于每种点权 $v$，求出只访问点权不超过 $v$ 的点时，能走出的最长路径长度。显然这个值可以直接推出答案。

从小到大枚举 $v$，相当于每次将一个点变为可以访问，每次询问当前的最长简单路径长度。可以看成每次加入一个点。

此时不同连通块之间可以分开讨论。考虑一个树形态的连通块，相当于有一棵有根树，只能从儿子向父亲走。考虑最长路径的结尾，结尾当前的父亲一定没有被加入，否则这个结尾还能在向上走，这条路径一定不是最长路径。因此可能作为最长路径的点即为只考虑已经加入的点，每个连通块中深度最小的点。

考虑用并查集维护，树上并查集的维护方式为每个连通块合并到深度最小的点上，考虑在这个过程的基础上维护答案。对于每个深度最小的点，记录 $as_u$ 表示当前以 $u$ 结尾的最长路径长度。

考虑加入一个点的操作，可以看成加入若干条边。对于一次加边 $(u,v)$，设 $u$ 为儿子，则 $u$ 一定是之前自己连通块中深度最小的点。此时记 $v$ 所在连通块的根为 $f_v$，则考虑更新 $f_v$ 的值，可以令 $as_{f_v}=max(as_{f_v},as_u+dep_u-dep_{f_v})$，然后将 $u$ 在并查集上合并到 $f_v$ 上即可。

考虑基环内向树的情况，在环上的点全部加入之前，显然可以用树的方式维护，找到环上最后一个点加入之前这个树的根，以这个点为根计算深度即可。如果环上的点全部加入了，则可以发现一条路径走到环上后一定会绕着环走一圈结束。因此设环长为 $l$，一个环等价于删去环上所有边，走到一个原先在环上的点额外增加 $l-1$ 的贡献。

在加入点出现环的时候，可以对当前的基环内向树进行dfs（没有加入的点也要dfs，但不计算答案），重新计算每个点在不考虑环时的深度，额外增加的贡献可以等价于将这些点的深度减去 $l-1$，然后重新计算当前已经加入的点的答案。之后再按照树的做法做即可。

复杂度 $O(nm\alpha(nm)+q)$， $nm$ 部分常数非常大。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 2300100
#define M 1520
int n,m,q,a,v[N],f[N],id[N],rid[N],as[N],head[N],cnt,is[N],fa[N],vl[N],s1,dep[N],nw;
char st[M][M];
vector<int> rf[N];
struct edge{int t,next;}ed[N];
void adde(int f,int t){is[t]=1;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
bool cmp(int a,int b){return v[a]<v[b];}
void dfs(int u,int fa){dep[u]=dep[fa]+1;for(int i=head[u];i;i=ed[i].next)dfs(ed[i].t,u);}
void dfs1(int u,int f,int v,int st,int fg)
{
	if(rid[u]>nw)fg=0;
	if(is[u])dep[u]=v,st=u;else dep[u]=dep[f]+1;
	if(fg)s1=max(s1,dep[u]),fa[u]=st;
	for(int i=head[u];i;i=ed[i].next)dfs1(ed[i].t,u,v,st,fg);
	if(is[u])dep[u]=1;
}
int main()
{
	scanf("%*d%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%s",st[i]+1);
	for(int i=1;i<=n*m;i++)scanf("%d",&v[i]),id[i]=i,fa[i]=i,as[i]=77777777;
	sort(id+1,id+n*m+1,cmp);
	for(int i=1;i<=n*m;i++)rid[id[i]]=i;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)
	{
		int sx=i,sy=j;
		if(st[i][j]=='U')sx--;
		if(st[i][j]=='D')sx++;
		if(st[i][j]=='L')sy--;
		if(st[i][j]=='R')sy++;
		if(sx<1||sx>n||sy<1||sy>m)f[i*m-m+j]=i*m-m+j;
		else f[i*m-m+j]=sx*m-m+sy;
		rf[f[i*m-m+j]].push_back(i*m-m+j);
	}
	for(int i=1;i<=n*m;i++)
	{
		int tp=id[i];
		for(int j=0;j<rf[tp].size();j++)if(rid[rf[tp][j]]<rid[tp])adde(tp,rf[tp][j]),fa[finds(rf[tp][j])]=finds(tp);
		if(finds(tp)!=finds(f[tp])&&rid[f[tp]]<rid[tp])adde(f[tp],tp),fa[finds(tp)]=finds(f[tp]);
	}
	for(int i=1;i<=n*m;i++)if(!is[i])dfs(i,0);
	for(int i=1;i<=n*m;i++)fa[i]=i,is[i]=0,vl[i]=1;
	s1=1;
	for(int i=1;i<=n*m;i++)
	{
		int tp=id[i];nw=i;
		for(int j=0;j<rf[tp].size();j++)if(rid[rf[tp][j]]<rid[tp])
		{
			fa[finds(rf[tp][j])]=finds(tp);
			vl[tp]=max(vl[tp],vl[rf[tp][j]]+1);
			s1=max(s1,vl[tp]);
		}
		if(rid[f[tp]]<rid[tp])
		if(tp!=finds(f[tp]))
		{
			fa[tp]=finds(f[tp]);
			vl[finds(f[tp])]=max(vl[finds(f[tp])],vl[tp]+dep[tp]-dep[finds(f[tp])]);
			s1=max(s1,vl[finds(f[tp])]);
		}
		else
		{
			int s2=0;
			for(int j=f[tp];!is[j];j=f[j])is[j]=1,s2++;
			dfs1(tp,0,s2,0,1);
		}
		as[s1]=min(as[s1],v[tp]);
	}
	for(int i=n*m;i>=1;i--)as[i-1]=min(as[i-1],as[i]);
	scanf("%d",&q);while(q--)scanf("%d",&a),printf("%d ",as[a]);
}
```

##### #260. 不讲武德

###### Problem

给一个 $n$ 个点 $m$ 条边的连通图，每条边有边权 $a_i,b_i$。

对于每个 $k\in\{1,2,...,n-1\}$，你需要选择 $k$ 条边满足：

1. $k$ 条边不存在环。
2. 在满足上一条件的情况下，选中边的 $(\sum a_i)*(\sum b_i)$ 最小。

对于每个 $k$ 输出最小值。

$n,m\leq 1500,1\leq a_i,b_i\leq 10^5$，不存在两条边边权完全相同。

$2s,512MB$

###### Sol

考虑答案的性质。设最小的 $(\sum a_i)*(\sum b_i)=t$，则考虑二维平面上 $xy=t$ 在一象限的曲线，如果将所有的方案表示成二维平面上的点 $(\sum a_i,\sum b_i)$，则所有方案对应的点都在 $xy=t(x,y>0)$ 的上方，且存在点在这个曲线上。

考虑在这个曲线上的点 $(x',\frac t{x'})$，因为曲线是下凸的，所以可以找到一条经过这个点的切线 $(y-\frac t{x'})=-\frac t{x'^2}(x-x')$，使得曲线整体在切线上方。此时所有方案对应的点都在这个切线的上方，且只有这个点对应的方案在这个切线上。

因此对于最优方案，一定存在一条经过这个方案对应点且斜率为负的直线，使得其它方案对应的点都在这条直线上方。

设斜率为 $-d(d>0)$ ，则这样的方案相当于满足 $\sum da_i+b_i$ 最小的方案。因此对于任意 $k$，一定存在一个实数 $d$，使得选择的方案为$\sum da_i+b_i$ 最小的方案。对于一个给定的 $d$ 求这个方案相当于将所有边按照 $da_i+b_i$ 从小到大排序做最小生成树。

注意到这个过程只和所有边按照 $a_i+db_i$ 排序的顺序有关。让 $d$ 从小到大变化，考虑顺序的变化。在 $d$ 变化的过程中，顺序的变化一定是每次交换顺序相邻两条边的顺序，且这样的交换显然只有 $O(m^2)$ 次。在当前的顺序中，维护每一对相邻的边顺序改变需要的 $d$ 值，使用set维护，每次选一对改变需要的 $d$ 最小的改变，然后更新这一对相邻的每一对的值。这样即可在 $O(m^2\log m)$ 的时间内求出在 $d$ 变化的过程中，边的顺序的变化过程。

考虑一个顺序的最小生成树，显然是按照顺序加入所有边，保留加入后不形成环的边。此时选 $k$ 条边的方案一定是选前 $k$ 条加入的边。因此对于可以 $O(n)$ 求出一个顺序每个 $k$ 的答案。

考虑交换相邻两条边的顺序时，生成树以及每个 $k$ 的答案的变化。设两条边为 $x,y$，且交换前 $x$ 在前面，则：

1. 如果 $x$ 不在最小生成树中，则交换后最小生成树以及最小生成树内边的顺序一定没有变化，因此此时每个 $k$ 的答案都不会改变。
2. 如果 $x,y$ 都在最小生成树中，则交换后最小生成树不变，但最小生成树内这两条边的顺序会改变。设 $x$ 在交换前为最小生成树中的第 $k_1$ 条边，则显然 $k=k_1$ 时，最后一条选的边会从 $x$ 变成 $y$，这时的答案会改变。对于其它的 $k$ 显然答案都不变。此时可以直接求出 $k=k_1$ 新的答案，并更新最小值。
3. $x$ 在最小生成树中，$y$ 不在。因为 $y$ 之前不在最小生成树中，可以发现此时最小生成树会变化当且仅当在最小生成树中 $y$ 能代替 $x$，即 $y$ 的端点在最小生成树上的路径中包含边 $x$。如果不存在这种情况，则此时不存在任何变化。否则，最小生成树中 $y$ 会代替 $x$，其它边不变。此时整个后缀的答案都会发生变化，因此只能枚举所有 $k$ 更新答案。

判断情况3中是否会替换可以使用LCT维护最小生成树，对每条边再建一个点即可。因为需要维护所有在最小生成树中的边以及它们的边权前缀和，可以使用一个BIT维护当前所有在最小生成树中的边的位置以及前缀和，此时2操作和3操作的修改可以在BIT上直接做。

此时2操作和3操作的判断部分复杂度均为 $O(\log m)$，只有3操作的修改复杂度为 $O(m)$。

考虑替换的次数。按照 $a_i$ 从小到大加入所有边。加入一条边时，它的 $a_i$ 大于之前的所有 $a_i$，因此初始就比它优秀的一定一直比它优秀，初始比它差的一定会在某个 $d$ 值开始比它优秀。

此时每条边都相对它变得更优秀，因此如果这条边初始没有被选，则它最后一定不被选，否则一定存在一个时刻，使得这个时刻之前它被选，这个时刻它被一条边替代。因为这之后其它边相对它只会更加优秀，因此之后它不会再被选。可以发现加入一条边只可能导致一条之前初始被选的边先在不被选，再在某个时刻这条边被替代。因此总的替代次数为 $O(m-n)$ 次。

因此这样做的总复杂度为 $O(m^2\log m)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
using namespace std;
#define N 1505
#define M 3010
#define ll long long
int n,m,s[N][2],v[N][2],s1[N],is[N];
ll as[N];
struct LCT{
	int ch[M][2],fa[M],lz[M],st[M],ct;
	bool nroot(int x){return ch[fa[x]][0]==x||ch[fa[x]][1]==x;}
	void doit(int x){swap(ch[x][0],ch[x][1]);lz[x]^=1;}
	void pushdown(int x){if(lz[x])doit(ch[x][0]),doit(ch[x][1]),lz[x]=0;}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;if(ch[g][ch[g][1]==f]==f)ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;}
	void splay(int x)
	{
		st[ct=1]=x;
		for(int i=x;nroot(i);i=fa[i])st[++ct]=fa[i];
		for(int i=ct;i>=1;i--)pushdown(st[i]);
		while(nroot(x))
		{
			int f=fa[x],g=fa[f];
			if(nroot(f))rotate((ch[g][1]==f)^(ch[f][1]==x)?x:f);
			rotate(x);
		}
	}
	void access(int x){int tp=0;while(x)splay(x),ch[x][1]=tp,tp=x,x=fa[x];}
	int findroot(int x){access(x);splay(x);while(ch[x][0])x=ch[x][0];return x;}
	void makeroot(int x){access(x);splay(x);doit(x);}
	void split(int x,int y){makeroot(x);access(y);splay(y);}
	void link(int x,int y){makeroot(x);fa[x]=y;}
	void cut(int x,int y){split(x,y);fa[x]=ch[y][0]=0;}
	int query(int x){splay(x);while(ch[x][0])x=ch[x][0];return x;}
}lct;
struct BIT{
	int s1[N],sx[N],sy[N],su1[N],sux[N],suy[N];
	void modify(int x,int v1,int v2,int v3){for(int i=x;i<=m;i+=i&-i)s1[i]+=v1,sx[i]+=v2,sy[i]+=v3;}
	pair<int,ll> query(int x){int t1=0,a=0,b=0;for(int i=x;i;i-=i&-i)t1+=s1[i],a+=sx[i],b+=sy[i];return make_pair(t1,1ll*a*b);}
	void doit()
	{
		for(int i=1;i<=m;i++)su1[i]=su1[i-(i&-i)]+s1[i],sux[i]=sux[i-(i&-i)]+sx[i],suy[i]=suy[i-(i&-i)]+sy[i];
		for(int i=1;i<=m;i++)as[su1[i]]=min(as[su1[i]],1ll*sux[i]*suy[i]);
	}
}tr;
struct line{int x,y,id;};
bool operator <(line a,line b)
{
	ll v1=1ll*(v[a.y][0]-v[a.x][0])*(v[b.y][1]-v[b.x][1])-1ll*(v[a.y][1]-v[a.x][1])*(v[b.y][0]-v[b.x][0]);
	if(v1)return v1>0;
	else return a.x==b.x?a.y<b.y:a.x<b.x;
}
set<line> tp;
bool cmp(int a,int b){return v[a][0]==v[b][0]?v[a][1]<v[b][1]:v[a][0]<v[b][0];}
bool check(int x,int y){return v[x][1]<v[y][1];}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d%d%d",&v[i][0],&v[i][1],&s[i][0],&s[i][1]),s1[i]=i;
	sort(s1+1,s1+m+1,cmp);
	for(int i=1;i<=m;i++)if(lct.findroot(s[s1[i]][0])!=lct.findroot(s[s1[i]][1]))
	{
		lct.link(s[s1[i]][0],s1[i]+n);lct.link(s[s1[i]][1],s1[i]+n);
		is[s1[i]]=1;tr.modify(i,1,v[s1[i]][0],v[s1[i]][1]);
		pair<int,ll> st=tr.query(i);
		as[st.first]=st.second;
	}
	for(int i=2;i<=m;i++)if(check(s1[i],s1[i-1]))tp.insert((line){s1[i],s1[i-1],i});
	while(!tp.empty())
	{
		line l1=*tp.begin();tp.erase(l1);
		int x=l1.id;
		if(x>2)tp.erase((line){s1[x-1],s1[x-2],0});
		if(x<m)tp.erase((line){s1[x+1],s1[x],0});
		if(is[s1[x-1]]&&is[s1[x]])
		{
			tr.modify(x-1,0,v[s1[x]][0]-v[s1[x-1]][0],v[s1[x]][1]-v[s1[x-1]][1]);
			tr.modify(x,0,v[s1[x-1]][0]-v[s1[x]][0],v[s1[x-1]][1]-v[s1[x]][1]);
			pair<int,ll> st=tr.query(x-1);
			as[st.first]=min(as[st.first],st.second);
		}
		else if(is[s1[x-1]]&&!is[s1[x]])
		{
			lct.split(s[s1[x]][0],s[s1[x]][1]);
			if(lct.query(s1[x-1]+n)==lct.query(s[s1[x]][0]))
			{
				lct.cut(s1[x-1]+n,s[s1[x-1]][0]);lct.cut(s1[x-1]+n,s[s1[x-1]][1]);
				lct.link(s1[x]+n,s[s1[x]][0]);lct.link(s1[x]+n,s[s1[x]][1]);
				tr.modify(x-1,0,v[s1[x]][0]-v[s1[x-1]][0],v[s1[x]][1]-v[s1[x-1]][1]);
				tr.doit();
				is[s1[x-1]]=0;is[s1[x]]=1;
			}
			else tr.modify(x-1,-1,-v[s1[x-1]][0],-v[s1[x-1]][1]),tr.modify(x,1,v[s1[x-1]][0],v[s1[x-1]][1]);
		}
		else if(is[s1[x]]&&!is[s1[x-1]])tr.modify(x-1,1,v[s1[x]][0],v[s1[x]][1]),tr.modify(x,-1,-v[s1[x]][0],-v[s1[x]][1]);
		swap(s1[x-1],s1[x]);
		if(x>2&&check(s1[x-1],s1[x-2]))tp.insert((line){s1[x-1],s1[x-2],x-1});
		if(x<m&&check(s1[x+1],s1[x]))tp.insert((line){s1[x+1],s1[x],x+1});
	}
	for(int i=1;i<n;i++)printf("%lld ",as[i]);
}
```

##### #264. 如果会出题就好了

###### Problem

给一棵 $n$ 个点的有根树，定义 $f_i$ 为 $i$ 子树内的点到 $i$ 的最大距离（距离计算边数）。

给出 $q$ 次询问，每次给出 $x,u,v$，把 $x$ 设为根，求 $u$ 到 $v$ 的路径上所有点的 $f_i$ 的异或和。

$n,q\leq 10^6$

$2s,512MB$

###### Sol

考虑取直径中点 $y$ 为根，设直径长度的一半为 $l$，则此时直径中点存在至少两个子树深度为 $l$。

计算这时所有点的 $f_i$，考虑以 $x$ 为根时 $f_u$ 的变化：

如果 $u$ 不在 $x$ 到 $y$ 的路径上，则 $u$ 的子树没有发生改变，因此 $f_u$ 不变。

如果 $u$ 在 $x$ 到 $y$ 的路径上，考虑直径中点 $y$ 所在的子树，$u$ 这个子树内的最大深度显然为 $dep_u+l$。如果其它子树有一个深度大于 $l$ ，则将这两条路径拼起来可以得到长度为 $2l+dep_u+1$ 的路径，这大于直径长度，矛盾。因此 $f_u=dep_u+l$

因此将根换成 $x$ 时，会将所有 $x$ 到 $y$ 路径上的点的 $f_u$ 改变，且改变后的值连续。

考虑处理询问，只需要求出路径 $(u,v)$ 与路径 $(x,y)$ 的交，求出其余部分原来的 $f$ 的异或和和交部分的异或和即可。这两部分都可以树上前缀和解决。

复杂度 $O(n+q\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1040010
int n,q,a,b,c,head[N],cnt,dep[N],f1[N],f[N][20],vl[N],sl[N],tp,t1,t2,v2[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	dep[u]=dep[fa]+1;f1[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
}
void dfs1(int u,int fa)
{
	dep[u]=dep[fa]+1;f[u][0]=fa;for(int i=1;i<20;i++)f[u][i]=f[f[u][i-1]][i-1];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=t1&&ed[i].t!=t2)dfs1(ed[i].t,u),v2[u]=max(v2[u],v2[ed[i].t]+1);
}
void dfs2(int u,int fa)
{
	vl[u]=vl[fa]^v2[u];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=t1&&ed[i].t!=t2)dfs2(ed[i].t,u);
}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=19;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=19;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int solve(int x,int y)
{
	if(!x)return 0;
	int l=LCA(x,y);
	if(!l)return vl[x];
	return sl[dep[l]+tp-1]^sl[tp-1]^vl[x]^vl[l];
}
int query(int x,int y,int r)
{
	int l=LCA(x,y);
	return solve(x,r)^solve(y,r)^solve(l,r)^solve(f[l][0],r);
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	for(int i=1;i<=n;i++)sl[i]=sl[i-1]^i;
	dfs(1,0);
	int fr=1;for(int i=1;i<=n;i++)if(dep[i]>dep[fr])fr=i;
	dfs(fr,0);
	int fr2=1;for(int i=1;i<=n;i++)if(dep[i]>dep[fr2])fr2=i;
	int le=dep[fr2];tp=le/2;
	for(int i=1;i<tp;i++)fr2=f1[fr2];
	if(le&1)t1=f1[fr2];else t1=fr2,t2=f1[fr2];
	dfs1(t1,0);dfs2(t1,0);if(t2)dfs1(t2,0),dfs2(t2,0);
	while(q--)scanf("%d%d%d",&a,&b,&c),printf("%d\n",query(b,c,a));
}
```

##### #268. Sequence

###### Problem

给一个长度为 $n$ 的序列 $a$，定义区间 $[l,r]$ 是好的，当且仅当 $l<r$ 且满足 $a_l\&a_r=a_l$ 或 $a_l\& a_r=a_r$，这里的 $\&$ 为二进制与。

定义一个好的区间 $[l,r]$ 的权值为 $\max_{i=l}^ra_i$，求出所有好的区间的权值和。

$n\leq 10^5,0\leq a_i<2^{14}$

$2s,512MB$

###### Sol

可以给答案加上 $\sum a_i$ ，随后去掉 $l<r$ 的限制。

考虑对于每个 $x$，求所有 $\max$ 在这个位置取到的好区间的数量（有多个 $\max$ 时认为取最左侧一个）。

对于一个 $x$ ，找到它左侧第一个大于等于它的位置 $l_x$，右侧第一个大于它的位置 $r_x$，则 $\max$ 在这个位置当且仅当 $l\in(l_x,x],r\in[x,r_x)$。相当于求满足这个条件的好区间个数。

从最大值开始考虑这个过程，每次可以看成从最大值开始，将当前最大值所在的区间划分成两个区间。根据启发式分裂的结论，每次划分后两个区间的长度的最小值的和为 $O(n\log n)$ 级别。考虑枚举长度小的区间中的每一个元素，则相当于给定 $O(n\log n)$ 个询问，每次给出 $l,r,x$ ，询问有多少个 $i\in[l,r]$ 满足 $a_i\& x=x$ 或者 $a_i\& x=a_i$。

这个限制可以看成满足 $a_i\& x=x$ 的数量加上满足 $a_i\& x=a_i$ 的数量再减去 $a_i=x$ 的数量，考虑 $a_i\& x=x$ 的限制，剩下的类似。

再把询问拆成前缀询问，相当于如下操作：

1. 加入一个数 $a_i$。操作次数 $O(n)$
2. 给定 $x$，询问加入的数中有多少个数满足 $a_i\&x=x$。操作次数 $O(n\log n)$

考虑分块，选择一个 $d$ ，记录 $v_{x,y}$ 表示有多少个 $a_i$ 满足 $a_i$ 二进制的后 $d$ 位为 $y$ 且除去后 $d$ 位后前面是 $x$ 的超集，即满足 $a_i\equiv y(\bmod 2^d),\lfloor\frac{a_i}{2^d}\rfloor\&x=x$。

加入一个数的时候，可以枚举 $x$ 加入，复杂度为 $O(2^{14-d})$，询问时只需要枚举 $y$ 部分的值即可求出答案，复杂度 $O(2^d)$。

因此可以平均两种操作的复杂度，总复杂度 $O(n\sqrt{v\log n})$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 105000
#define ll long long
int n,v[N],c1[N],c2[N],c3[N],tp[N],ct,ls[N],rs[N];
ll as;
vector<pair<int,int> > st[N];
void add(int x)
{
	int tp=x&31;
	for(int i=tp;i<1<<14;i+=1<<5)
	{
		if((i&x)==x)c1[i]++;
		if((i&x)==i)c2[i]++;
	}
	c3[x]++;
}
int que(int x)
{
	int tp=x-(x&31),as=0;
	for(int i=tp;i<tp+32;i++)
	{
		if((i&x)==i)as+=c1[i];
		if((i&x)==x)as+=c2[i];
	}
	return as-c3[x];
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),as-=v[i];
	for(int i=1;i<=n;i++)
	{
		while(ct&&v[tp[ct]]<v[i])rs[tp[ct]]=i,ct--;
		ls[i]=tp[ct];tp[++ct]=i;
	}
	for(int i=1;i<=ct;i++)rs[tp[i]]=n+1;
	for(int i=1;i<=n;i++)if(rs[i]-i<=i-ls[i])
	for(int j=i;j<rs[i];j++)st[i].push_back(make_pair(v[j],v[i])),st[ls[i]].push_back(make_pair(v[j],-v[i]));
	else for(int j=ls[i]+1;j<=i;j++)st[rs[i]-1].push_back(make_pair(v[j],v[i])),st[i-1].push_back(make_pair(v[j],-v[i]));
	for(int i=1;i<=n;i++)
	{
		add(v[i]);
		for(int j=0;j<st[i].size();j++)as+=1ll*st[i][j].second*que(st[i][j].first);
	}
	printf("%lld\n",as);
}
```

##### #272. Find a city

###### Problem

交互题

有一个 $n$ 个点的竞赛图，你每次询问一条边的方向。

你希望找到一个出度不小于 $n-2$ 的点，输出任意一个这样的点或者报告不存在这样的点。

$n\leq 1000$，询问次数不超过 $5n$。

多组数据，$\sum n^2\leq 10^7$。

$3s,512MB$

###### Sol

满足条件的点最多只有一条入边。考虑先对于每个点，找出一条连向它的入边。

记录一个当前点 $nw$，表示当前考虑的点中还没有找到入边的点。初始 $nw=1$，从小到大考虑每个点，设考虑的点为 $x$，询问 $(x,nw)$ 间边的顺序。如果为 $x$ 连向 $nw$ 则令 $nw=x$，如果为 $nw$ 连向 $x$ 则不变。

此时只剩 $nw$ 还没有找到入边，询问它和其它所有点的边顺序，找到一条入边就结束这个过程。如果找不到这样的边，则显然 $nw$ 为合法的答案。

否则，此时对于每个点都找到了一条入边。设询问次数为 $n+k$，则可以发现这个做法对于 $k$ 个点找到了两条入边，可以删去这些点。此时只需要在 $3n$ 步操作内对剩下的点求出答案即可。

设找到的连向 $x$ 的点为 $f_x$，则对于两个点 $x,y$ 满足 $f_x\neq y,f_y\neq x$，如果询问 $(x,y)$ 的边的方向，根据结果一定能至少判断一个点不合法。

此时可以每次取这样的一对出来判断，每次都能删去一个点，直到不能删为止。可以发现，如果存在四个点，则一定存在一对点满足条件。因此最后留下的点数不超过 $3$，询问次数为 $n$ 次。

此时对于剩下的点，考虑暴力将它们和其它点询问，询问次数为 $3n$，总次数 $5n$。

但注意到达到 $5n$ 当且仅当剩下三个点组成三元环，因为上面的做法中考虑 $f$ 得到的是一个连通的基环内向树，因此初始只有这一个三元环。

这种情况下可以在前面的询问中，每次拿一个三元环上的点和一个其它点询问。可以发现这样一定能询问完。这时后面的 $3n$ 次询问一定包含了前面的 $n$ 次询问，因此可以省去 $n$ 次询问，总询问次数为 $4n$。

复杂度可以做到 $O(\sum n)$ ~~但写个n^2完全没有问题~~

###### Code

```cpp
#include "city.h"
using namespace std;
#define N 1050
int is[N][N],st[N],ct,nt[N],f1[N];
int query(int x,int y)
{
	if(is[x][y]!=-1)return is[x][y];
	is[x][y]=ask(x,y);is[y][x]=1-is[x][y];
	return is[x][y];
}
int solve(int n)
{
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)is[i][j]=-1;
	for(int i=1;i<=n;i++)f1[i]=0;
	int v1=1;
	for(int i=2;i<=n;i++)if(query(v1,i))nt[i]=v1;else nt[v1]=i,v1=i;
	for(int i=1;i<v1;i++)if(query(v1,i))nt[i]=v1;else{nt[v1]=i;v1=i;break;}
	if(!nt[v1])return v1;
	st[ct=1]=v1;f1[v1]=1;
	for(int i=nt[v1];i!=v1;i=nt[i])st[++ct]=i,f1[i]=1;
	for(int i=v1;i<=n;i++)if(!f1[i])st[++ct]=i;
	for(int i=1;i<=ct;i++)for(int j=i+1;j<=ct;j++)if(nt[st[i]]!=st[j]&&nt[st[j]]!=st[i])
	{
		int tp=query(st[i],st[j])?j:i;
		for(int l=tp;l<ct;l++)st[l]=st[l+1];
		ct--;j=i+1;
	}
	for(int i=1;i<=ct;i++)
	{
		int fg=1;
		for(int j=1;j<=n;j++)if(nt[st[i]]!=j&&st[i]!=j&&!query(st[i],j)){fg=0;break;}
		if(fg)return st[i];
	}
	return -1;
}
```

##### #276. Fierce Storm

###### Problem

有一棵 $n$ 个点的树，每个点有点权 $v_i$，每条边有一个概率 $p_i$ 。

现在第 $i$ 条边有 $p_i$ 的概率被删去，定义权值为删边后所有连通块的连通块内点权最大值的最小值。

求权值的期望，模 $10^9+7$。

$n\leq 10^5$

$2s,512MB$

###### Sol

考虑期望线性性，问题可以看成对于每种点权 $v$，求权值大于等于这个点权的概率。

此时将权值大于等于 $v$ 的看成黑点，剩下的看成白点，则相当于求删边后每个连通块中都有黑点的概率。

此时可以考虑树形 $dp$，记 $dp_{u,0/1}$ 表示 $u$ 的子树中随机删边，$u$ 所在的连通块中是否有黑点，子树内其它连通块内都有黑点的概率。转移显然。

如果 $v$ 从小到大变化，则相当于每次将一个白点变成黑点。考虑动态DP，对于一个点，如果求出了它所有轻儿子的 $dp$，重儿子的 $dp$ 到它的 $dp$ 的转移显然可以看成一个矩阵，树剖后线段树维护每条重链的矩阵，修改时每次求出当前重链顶的 $dp$，然后修改链顶父亲的矩阵再跳到父亲做即可。

使用线段树/SBT可以做到 $O(n\log^2 n)$ 或者 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define mod 1000000007
int n,a,b,c,v2[N],v[N],st[N],head[N],cnt,f[N],sz[N],sn[N],tp[N],ls[N],id[N],tid[N],vl[N][2],ct,is[N],as;
bool cmp(int a,int b){return v2[a]>v2[b];}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct sth{int x,y;sth(){x=1,y=0;}void mul(int a){if(!a)y++;else x=1ll*x*a%mod;}void div(int a){if(!a)y--;else x=1ll*x*pw(a,mod-2)%mod;}int que(){return y?0:x;}}v1[N][2];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){f,head[t]};head[t]=cnt;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void dfs1(int u,int fa)
{
	vl[u][0]=1;sz[u]=1;f[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),sn[u]=sz[sn[u]]>sz[ed[i].t]?sn[u]:ed[i].t,vl[u][0]=1ll*vl[u][0]*v[ed[i].t]%mod*vl[ed[i].t][0]%mod,sz[u]+=sz[ed[i].t];
}
int dfs2(int u,int t)
{
	tp[u]=t;id[u]=++ct;tid[ct]=u;
	if(sn[u])ls[u]=dfs2(sn[u],t);else return ls[u]=u;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=f[u]&&ed[i].t!=sn[u])
	{
		dfs2(ed[i].t,ed[i].t);
		int s0=(1ll*v[ed[i].t]*vl[ed[i].t][0]+1ll*(mod+1-v[ed[i].t])*vl[ed[i].t][1])%mod,s1=(s0+1ll*v[ed[i].t]*vl[ed[i].t][1])%mod;
		v1[u][0].mul(s0);v1[u][1].mul(s1);
	}
	return ls[u];
}
struct mat{int v[2][2];mat(){v[0][0]=v[0][1]=v[1][0]=v[1][1]=0;}};
mat operator *(mat a,mat b){mat c;for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++)c.v[i][k]=(c.v[i][k]+1ll*a.v[i][j]*b.v[j][k])%mod;return c;}
mat doit(int x)
{
	mat s1,s2;
	if(!sn[x])return s1;
	s1.v[0][0]=s1.v[1][1]=v[sn[x]];s1.v[1][0]=(mod+1-v[sn[x]])%mod;
	int t1=v1[x][0].que(),t2=v1[x][1].que();
	s2.v[0][0]=t1;s2.v[0][1]=(t2-t1+mod)%mod;s2.v[1][1]=t2;
	s1=s1*s2;
	if(is[x])for(int i=0;i<2;i++)s1.v[i][1]=(s1.v[i][1]+s1.v[i][0])%mod,s1.v[i][0]=0;
	return s1;
}
struct segt{
	struct node{int x,l,r;mat su;}e[N*4];
	void pushup(int x){e[x].su=e[x<<1|1].su*e[x<<1].su;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;e[x].su.v[0][0]=e[x].su.v[1][1]=1;if(l==r){e[x].su=doit(tid[l]);return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
	void modify(int x,int v){if(!v)return;if(e[x].l==e[x].r){e[x].su=doit(tid[e[x].l]);return;}int mid=(e[x].l+e[x].r)>>1;if(mid>=v)modify(x<<1,v);else modify(x<<1|1,v);pushup(x);}
	mat query(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].su;int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return query(x<<1|1,mid+1,r)*query(x<<1,l,mid);}
}tr;
void justdoit(int x)
{
	int s0=!is[ls[x]],s1=is[ls[x]];
	if(x!=ls[x])
	{
		mat st=tr.query(1,id[x],id[ls[x]]-1);
		s0=st.v[is[ls[x]]][0];s1=st.v[is[ls[x]]][1];
	}
	if(f[x])
	{
		int t0=(1ll*v[x]*vl[x][0]+1ll*(mod+1-v[x])*vl[x][1])%mod,t1=(t0+1ll*v[x]*vl[x][1])%mod;
		v1[f[x]][0].div(t0);v1[f[x]][1].div(t1);
		t0=(1ll*v[x]*s0+1ll*(mod+1-v[x])*s1)%mod,t1=(t0+1ll*v[x]*s1)%mod;
		v1[f[x]][0].mul(t0);v1[f[x]][1].mul(t1);
		tr.modify(1,id[f[x]]);
	}
	vl[x][0]=s0;vl[x][1]=s1;
}
void modify(int x)
{
	is[x]=1;tr.modify(1,id[x]);
	while(x)x=tp[x],justdoit(x),x=f[x];
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v2[i]),st[i]=i;
	sort(st+1,st+n+1,cmp);
	for(int i=2;i<=n;i++)scanf("%d%d%d",&a,&b,&c),v[i]=1ll*b*pw((b+c)%mod,mod-2)%mod,adde(a,i);
	dfs1(1,0);dfs2(1,1);tr.build(1,1,n);
	for(int i=1;i<=n;i++)modify(st[i]),as=(as+1ll*(v2[st[i]]-v2[st[i+1]])*vl[1][1])%mod;
	printf("%d\n",as);
}
```

##### #280. Expanding Sequence

###### Problem

使用如下的方式定义序列 $A_i$：

$A_1={1,1}$，对于 $i>1$，$A_i$ 为一个长度为 $2^{i-1}+1$ 的序列，其中：

1. $A_{i,2j}=A_{i-1,j}$
2. $A_{i,2j+1}=A_{i-1,j}+A_{i-1,j+1}$

给出 $n,k$，求出 $A_{10^{18}}$ 中第 $k$ 个小于等于 $n$ 的元素的值，保证这个元素存在。

$n\leq 10^6$

$1s,512MB$

###### Sol

考虑将序列中的每个元素看成一个二元组 $(x,y)$，初始 $A_1={(0,1),(1,1)}$，相加操作看成两维分别相加。

如果把二元组 $(x,y)$ 看成 $\frac xy$，则这个过程可以看成 Stern-Brocot Tree 上增加一层叶子的过程。因此 $A_i$ 可以看成以 $(0,1),(1,1)$ 为基础构造， Stern-Brocot Tree 上深度不超过 $i$ 的部分的dfs序。因此 $A_{10^{18}}$ 中所有小于 $n$ 的元素顺序排列的结果等于树上所有分母不超过 $n$ 的元素按dfs序排列的结果。

根据性质，这相当于将所有分母不超过 $n$ 且值在 $[0,1]$ 间的既约分数（以及 $\frac 01,\frac 11$）按照大小排序，答案即为第 $k$ 大的值的分母。

考虑在 Stern-Brocot Tree 上找第 $k$ 大，相当于若干次询问某个子树内分母不超过 $n$ 的既约分数数量。

对于一个子树，设它左侧的父亲为 $(a,x)$，右侧的父亲为 $(b,y)$，显然子树内的所有点都是由这两个二元组线性组合出来的，且每一个线性组合出来的既约分数都可以在子树中找到。因此可以发现子树内分母不超过 $n$ 的既约分数数量等于 $\sum_{i,j>0}[gcd(i,j)=1][ix+jy\leq n]$。

考虑莫比乌斯反演：
$$
\sum_{i,j>0}[gcd(i,j)=1][ix+jy\leq n]=\sum_{d\geq 1}\mu(d)\sum_{i,j>0}[ix+jy\leq\frac nd]
$$
再考虑求后面这个，这显然属于一种类欧几里得算法，因此可以做到 $O(\log n)$，数论分块后复杂度为 $O(\sqrt n\log n)$

考虑直接在树上走，显然步数不超过 $n$ 步，复杂度为 $\sum_{i=1}^nO(\sqrt\frac ni\log\frac ni)$，不超过 $O(n\log n)$

同时，在 Stern-Brocot Tree 上向下走，根据gcd的方式，每转向两次分母一定翻倍，因此可以每次二分向这个方向走多少步，只需要 $O(\log^2 n)$ 次判断。这样的复杂度不超过 $O(\sqrt n\log^3 n)$，并且常数非常小。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1000600
#define ll long long
int n,pr[N],ct,mu[N],ch[N],su[N];
ll k;
void prime()
{
	mu[1]=1;
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])mu[i]=-1,pr[++ct]=i;
		for(int j=1;j<=ct&&i*pr[j]<=n;j++)
		{
			ch[i*pr[j]]=1;mu[i*pr[j]]=-mu[i];
			if(i%pr[j]==0){mu[i*pr[j]]=0;break;}
		}
	}
	for(int i=1;i<=n;i++)su[i]=su[i-1]+mu[i];
}
ll solve(int a,int b,int c)
{	
	if(a+b>c||!a||!b)return 0;
	if(a<b)a^=b^=a^=b;
	int tp=(c-b)/a,s1=((c-b)%a+b)/b;
	if(s1>a/b)s1=a/b;
	return solve(b,a-b*s1,c-b*(tp+1)*s1)+1ll*tp*(tp+1)/2*s1;
}
ll solve2(int a,int b,int c)
{
	ll as=0;
	for(int l=1,r;l<=c;l=r+1)
	{
		r=c/(c/l);
		if(c/l<a+b)break;
		as+=(su[r]-su[l-1])*solve(a,b,c/l);
	}
	return as;
}
int query(int x,int y,ll k)
{
	ll tp=solve2(x,x+y,n);
	if(tp>=k)
	{
		int lb=2,rb=(n-y)/x,as=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)/2;
			if(solve2(x,y+mid*x,n)>=k)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		return query(x,y+as*x,k);
	}
	else if(tp+1==k)return x+y;
	else
	{
		ll s1=solve2(x+y,y,n);
		int lb=2,rb=(n-x)/y,as=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)/2;
			if(s1-solve2(x+mid*y,y,n)<k-tp-1)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		return query(x+as*y,y,k-tp-1-(s1-solve2(x+as*y,y,n)));
	}
}
int main()
{
	scanf("%d%lld",&n,&k);
	prime();
	if(k==1)printf("1\n");else printf("%d\n",query(1,0,k-1));
}
```

##### #292. teXt Editor

###### Problem

给一个长度为 $n$ 的字符串 $s$，以及 $m$ 个字符串 $t_i$，支持 $q$ 次操作：

1. 给出 $l,r$ 以及字符串 $a$ ，将 $s[l,r]$ 替换为 $a$ 无限循环后的一个长度为 $r-l+1$ 的前缀。
2. 给出 $l,r$，询问 $s[l,r]$ 中每个 $t_i$ 的出现次数之和。

$n\leq 10^6,m,q\leq 10^5,\sum |a|\leq 10^6,|t_i|\leq 50,\sum |t_i|\leq 2\times 10^5,|\sum|=62$

$2s,512MB$

###### Sol

对于一个询问，考虑对于每个后缀 $s[i,n]$，计算这个后缀包含多少个 $t_i$ 作为前缀，记这个值为 $as_i$。~~这里以及之后反过来处理完全没有问题~~

此时可以将所有串反过来建AC自动机，统计每个点的fail子树中的串数量，然后将原串反过来在AC自动机上匹配即可得到它每个后缀的答案。

因为 $|t_i|$ 很小，注意到每个后缀的答案只和后缀前 $\max |t_i|$ 个字符有关，因此对于一个循环串，除去后 $\max |t_i|$ 个位置。前面的位置的 $as$ 一定是循环的，且循环节和串的循环节相同。

考虑询问循环串中一个区间的答案，只有最后 $\max |t_i|$ 个位置的 $as$ 会发生变化，可以暴力求出这部分，复杂度 $O(\max|t_i|)$ 。对于前面的部分，因为答案循环，只需要预处理 $as$ 的前缀和即可 $O(1)$ 询问。

对于两个循环串，将它们拼接之后，只有第一个串的后 $$\max |t_i|$$ 个位置的 $as$ 会发生变化。只需要先把第二个串长度为 $\max |t_i|$ 的前缀反过来在AC自动机上匹配，再依次加入第一个串长度为 $\max |t_i|$ 的后缀即可在 $O(\max|t_i|)$ 的时间内求出。

考虑将串表示成若干个循环串拼接起来的结果。对于每个循环串处理 $as$ 以及整体答案，对于每相邻两个串处理拼接这两个串的答案。为了避免多次拼接造成多次影响，这里要求每个循环串长度不小于 $\max |t_i|$，这样就只需要处理相邻两个串的影响。

考虑询问，使用set维护每个串所在的区间，BIT维护所有的整体答案以及相邻两个串的影响，这时只有询问端点所在的两个循环串内以及相邻的影响需要重新计算，重算的复杂度显然为 $O(\max t_i)$，因此询问复杂度为 $O(\log n+\max t_i)$

考虑修改，此时会删去中间若干个串，并删去一个串的后缀和一个串的前缀。对于删去前缀后缀的情况，可以在每个串中记录它的开头结尾在循环中的位置，这样可以在不修改 $as$ 的情况删前后缀。

但此时可能导致出现长度小于 $\max |t_i|$ 的串，这时考虑从两侧的串向这个串移动若干个字符，直到长度大于等于 $\max |t_i|$。此时因为这个串长度只有 $\max |t_i|$ ，可以对这个串重新计算所有的值。如果两侧的串长度也很小，无法移动，则可以将它们合并起来再重新计算，实现时可以对于合并后长度不超过 $2\max |t_i|$ 的合并即可。

如果一次修改操作修改的区间在一个循环串的内部，则修改后得到的左右两部分循环节都与原来的串相同，如果直接复制 $as$ 等结果可能被卡。此时可以使用类似指针的方式，记录每个串使用哪个 $as$，复制这个编号即可。这样修改的复杂度即为 $O(|a|+\log n+\max|t_i|)$，其中 $O(\log n)$ 为均摊。

复杂度 $O(n+|\sum|*\sum |t_i|+q(\log n+\max |t_i|)+\sum |a|)$，有大量细节并且代码非常长。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<set>
#include<queue>
using namespace std;
#define N 200050
#define ll long long
int n,m,q,a,b,c,ct,ct1;
char s[N*5];
int getid(char s)
{
	if(s>='A'&&s<='Z')return s-'A'+1;
	if(s>='a'&&s<='z')return s-'a'+27;
	return s-'0'+53;
}
vector<int> doit2()
{
	vector<int> as;
	for(int i=1;s[i];i++)as.push_back(getid(s[i]));
	return as;
}
struct ACAM{
	int ch[N][63],ct,f[N][63],as[N],fail[N];
	void init(){ct=1;}
	void ins(vector<int> s)
	{
		reverse(s.begin(),s.end());
		int vl=1;
		for(int i=0;i<s.size();i++)
		{
			if(!ch[vl][s[i]])ch[vl][s[i]]=++ct;
			vl=ch[vl][s[i]];
		}
		as[vl]++;
	}
	void doit()
	{
		queue<int> st;
		for(int i=1;i<=62;i++)f[0][i]=f[1][i]=1;fail[0]=fail[1]=1;
		for(int i=1;i<=62;i++)if(ch[1][i])f[1][i]=ch[1][i];
		st.push(1);
		while(!st.empty())
		{
			int u=st.front();st.pop();
			for(int i=1;i<=62;i++)if(ch[u][i])
			{
				fail[ch[u][i]]=u==1?1:f[fail[u]][i];as[ch[u][i]]+=as[fail[ch[u][i]]];
				for(int j=1;j<=62;j++)f[ch[u][i]][j]=ch[ch[u][i]][j]?ch[ch[u][i]][j]:f[fail[ch[u][i]]][j];
				st.push(ch[u][i]);
			}
		}
	}
}ac;
struct st1{int l,r,id;};
bool operator <(st1 a,st1 b){return a.l==b.l?a.r<b.r:a.l<b.l;}
set<st1> fu;
vector<int> vl[N*3],st2[N*3];
vector<ll> su[N*3];
struct sth{
	int vli,sui,si;
	int le,l1,r1,s1,ls;
	ll as;
	#define vl vl[vli]
	#define su su[sui]
	#define s st2[si]
	void calc(int tp=1)
	{
		as=1ll*su[le-1]*(s1-1)-(l1?su[l1-1]:0)+su[r1];
		int s2=s1,r2=r1,t=0;
		while(t<50)
		{
			tp=ac.f[tp][s[r2]];as=as-vl[r2]+ac.as[tp];
			if(s2==1&&l1==r2)break;
			t++;if(!r2)r2=le-1,s2--;else r2--;
		}
	}
	void init(vector<int> _s,int l)
	{
		vli=sui=si=++ct1;
		s=_s;le=s.size();
		vl.clear();su.clear();
		l1=0;s1=(l-1)/le+1;r1=(l-1)%le;ls=l;
		int s1=1;
		for(int i=le-1;i>=0;i--)s1=ac.f[s1][s[i]],vl.push_back(0);
		for(int i=le-1;i>=0;i--)
		s1=ac.f[s1][s[i]],vl[i]=ac.as[s1];
		ll su1=0;
		for(int i=0;i<le;i++)su1+=vl[i],su.push_back(su1);
		calc();
	}
	ll querypr(int l)
	{
		int s2=(l1+l-1)/le+1,r2=(l1+l-1)%le;
		ll as=1ll*su[le-1]*(s2-1)-(l1?su[l1-1]:0)+su[r2];
		int t=0,tp=1;
		while(t<50)
		{
			tp=ac.f[tp][s[r2]];as=as-vl[r2]+ac.as[tp];
			if(s2==1&&l1==r2)break;
			t++;if(!r2)r2=le-1,s2--;else r2--;
		}
		return as;
	}
	ll querysu(int l,int tp=1)
	{
		int s0=(ls-l+l1)/le+1,l0=(ls-l+l1)%le;
		ll as=1ll*su[le-1]*(s1-s0)-(l0?su[l0-1]:0)+su[r1];
		int s2=s1,r2=r1,t=0;
		while(t<50)
		{
			tp=ac.f[tp][s[r2]];as=as-vl[r2]+ac.as[tp];
			if(s2==s0&&l0==r2)break;
			t++;if(!r2)r2=le-1,s2--;else r2--;
		}
		return as;
	}
	ll querylr(int l,int r)
	{
		int s0=(l+l1-1)/le+1,l0=(l+l1-1)%le;
		int s2=(l1+r-1)/le+1,r2=(l1+r-1)%le,t=0,tp=1;
		ll as=1ll*su[le-1]*(s2-s0)-(l0?su[l0-1]:0)+su[r2];
		while(t<50)
		{
			tp=ac.f[tp][s[r2]];as=as-vl[r2]+ac.as[tp];
			if(s2==s0&&l0==r2)break;
			t++;if(!r2)r2=le-1,s2--;else r2--;
		}
		return as;
	}
	void decr(int v)
	{
		ls-=v;
		int tp=r1+s1*le-v;
		s1=tp/le;r1=tp%le;
	}
	void decl(int v)
	{
		ls-=v;
		int tp=l1+v;
		s1-=tp/le;l1=tp%le;
	}
	#undef vl
	#undef su
	#undef s
}st[N*3];
ll solve(int x,int y,int le1=-1,int le2=-1)
{
	if(le1==-1)le1=st[x].ls;
	if(le2==-1)le2=st[y].ls;
	int tp[52],le=0;
	int l2=st[y].l1,r2=st[y].r1;
	while(le<50&&le<le2)
	{
		tp[++le]=st2[st[y].si][l2];
		l2++;if(l2==st[y].le)l2=0;
	}
	int v1=1;
	for(int i=le;i>=1;i--)v1=ac.f[v1][tp[i]];
	return st[x].querysu(le1,v1)-st[x].querysu(le1,1);
}
struct BIT{
	ll tr[N*5];
	void add(int x,ll y){for(int i=x;i<=n;i+=i&-i)tr[i]+=y;}
	ll que(int x){ll as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}f1,f2;
void modify(int l,int r,vector<int> s)
{
	int tp=49/s.size()+1;
	vector<int> s2;
	for(int i=1;i<=tp;i++)for(int j=0;j<s.size();j++)s2.push_back(s[j]);
	s=s2;
	set<st1>::iterator it1=fu.lower_bound((st1){l-1,n+5,0}),it2=fu.lower_bound((st1){r+1,n+5,0}),it;
	it1--;it2--;
	it=it1;
	while(1)
	{
		int id=(*it).id,l=(*it).l,r=(*it).r,id2;
		f1.add(l,-st[id].as);
		if(it==it2)break;
		it++;id2=(*it).id;
		f2.add(r,-solve(id,id2));
	}
	st1 lb=*it1,rb=*it2,tp1;
	int lf=0,rf=0;
	if(it1!=fu.begin())it1--,lf=(*it1).id,f2.add(lb.l-1,-solve(lf,lb.id));
	if((++it2)!=fu.end())rf=(*it2).id,f2.add(rb.r,-solve(rb.id,rf));
	while(1){it=fu.lower_bound(lb);tp1=*it;fu.erase(tp1);if(tp1.r==rb.r)break;}
	if(lb.id==rb.id)st[++ct]=st[rb.id],rb.id=ct;
	st[++ct].init(s,r-l+1);
	st[lb.id].decr(lb.r-l+1);
	st[rb.id].decl(r-rb.l+1);
	st1 l1=(st1){lb.l,l-1,lb.id},r1=(st1){r+1,rb.r,rb.id},nw=(st1){l,r,ct};
	st1 fr[3]={l1,nw,r1};int ct2=3;
	for(int i=0;i+1<ct2;i++)
	for(int j=ct2-1;j>i;j--)if(j<ct2)
	if(fr[j].r-fr[i].l+1<(j-i+1)*51)
	{
		vector<int> sr;
		for(int k=i;k<=j;k++)
		for(int t=fr[k].l,id=fr[k].id;t<=fr[k].r;t++)sr.push_back(st2[st[id].si][st[id].l1]),st[id].l1=(st[id].l1+1)%st[id].le;
		st[++ct].init(sr,fr[j].r-fr[i].l+1);
		fr[i]=(st1){fr[i].l,fr[j].r,ct};
		for(int k=j+1;k<ct2;k++)fr[k-j+i]=fr[k];ct2-=j-i;
	}
	if(ct2>1)
	for(int i=0;i<ct2;i++)if(fr[i].r-fr[i].l+1<50)
	{
		int rs=50-(fr[i].r-fr[i].l+1),lv=0,rv=0;
		if(i)lv=min(rs,fr[i-1].r-fr[i-1].l+1-50);
		rv=rs-lv;
		vector<int> sr;
		for(int j=1;j<=lv;j++)
		{
			int id=fr[i-1].id;
			fr[i-1].r--;
			sr.push_back(st2[st[id].si][st[id].r1]);st[id].decr(1);
		}
		reverse(sr.begin(),sr.end());
		for(int t=fr[i].l,id=fr[i].id;t<=fr[i].r;t++)sr.push_back(st2[st[id].si][st[id].l1]),st[id].l1=(st[id].l1+1)%st[id].le;
		for(int j=1;j<=rv;j++)
		{
			int id=fr[i+1].id;
			fr[i+1].l++;
			sr.push_back(st2[st[id].si][st[id].l1]);st[id].decl(1);
		}
		fr[i].l-=lv;fr[i].r+=rv;
		st[++ct].init(sr,fr[i].r-fr[i].l+1);
		fr[i].id=ct;
	}
	for(int i=0;i<ct2;i++)
	{
		fu.insert(fr[i]);st[fr[i].id].calc();
		f1.add(fr[i].l,st[fr[i].id].as);
		if(i)f2.add(fr[i-1].r,solve(fr[i-1].id,fr[i].id));
	}
	if(lf)f2.add(fr[0].l-1,solve(lf,fr[0].id));
	if(rf)f2.add(fr[ct2-1].r,solve(fr[ct2-1].id,rf));
}
ll query(int l,int r)
{
	set<st1>::iterator it1=fu.lower_bound((st1){l,n+5,0}),it2=fu.lower_bound((st1){r,n+5,0});
	it1--;it2--;
	st1 lb=*it1,rb=*it2,l1,r1;
	it1++;it2--;
	l1=*it1;r1=*it2;
	if(lb.id==rb.id)return st[lb.id].querylr(l-lb.l+1,r-lb.l+1);
	if(lb.r==rb.l-1)return solve(lb.id,rb.id,lb.r-l+1,r-rb.l+1)+st[lb.id].querysu(lb.r-l+1)+st[rb.id].querypr(r-rb.l+1);
	ll as=f1.que(rb.l-1)-f1.que(lb.r)+f2.que(rb.l-2)-f2.que(lb.r);
	as+=solve(lb.id,l1.id,lb.r-l+1,-1)+solve(r1.id,rb.id,-1,r-rb.l+1)+st[lb.id].querysu(lb.r-l+1)+st[rb.id].querypr(r-rb.l+1);
	return as;
}
int main()
{
	scanf("%d%d%d%s",&n,&m,&q,s+2);
	s[1]=s[n+2]='A';
	vector<int> se=doit2();
	ac.init();
	for(int i=1;i<=m;i++)scanf("%s",s+1),ac.ins(doit2());
	ac.doit();
	st[ct=1].init(se,n+2);fu.insert((st1){1,n+2,1});f1.add(1,st[1].as);
	while(q--)
	{
		scanf("%d%d%d",&a,&b,&c);b++;c++;
		if(a==2)scanf("%s",s+1),modify(b,c,doit2());
		else printf("%lld\n",query(b,c));
	}
}
```

##### #296. Tree & Derangement

###### Problem

给一个二分图，两侧各有 $n$ 个点，有 $m$ 条边，保证图没有环。

有一个排列 $p$，对于一条连接左边第 $i$ 个点和右边第 $j$ 个点的边，要求 $p_i\neq j$。

求满足条件的 $n$ 阶排列数，模 $998244353$。

$n\leq 10^5$

$2.5s,1024MB$

###### Sol

考虑容斥，选择若干条边，钦定这些边都满足 $p_i=j$。

如果钦定了 $k$ 条边，则容斥系数为 $(-1)^k$。如果选的限制相互冲突则显然方案数为 $0$，否则方案数为 $(n-k)!$。

可以发现限制冲突当且仅当出现 $p_i=j,p_i=k$ 或者 $p_i=j,p_k=j$，相当于有边存在公共点。

因此只需要对于每个 $k$ 求出选 $k$ 条边不存在公共点的方案数，即可求出答案。

直接的 $dp$ 为设 $dp_{i,0/1,j}$ 表示 $i$ 子树内选 $j$ 条没有公共点的边，且是否选了一条包含 $i$ 的边的方案数。这是一个子树大小相关的 $dp$ ，考虑表示成生成函数，用树剖优化。

考虑求出一条重链链顶的 $dp$，对于每个点，先做所有轻儿子的 $dp$，然后分治FFT合并，可以得到对于这个点只考虑轻儿子子树，是否选包含当前点的边时 $dp$ 的生成函数。然后在重链上分治，设 $f_{l,r,0/1,0/1}$ 表示考虑重链 $[l,r]$ 区间以及它们的轻儿子，当前重链的左右端点是否被选，此时 $dp$ 对应的生成函数，分治后FFT合并即可。

可以将分治FFT写成类似二进制分组的形式，即维护一个栈，每次加入一个生成函数。设当前是第 $i$ 次加入，将栈顶 $\log_2lowbit(i)$ 个生成函数依次乘起来。可以发现这样的复杂度显然正确，可以去掉递归。~~虽然这可能没有什么意义~~

复杂度 $O(n\log^3 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 200040
#define mod 998244353
int n,m,a,b,ct,ct2,head[N],cnt,sz[N],sn[N],ntt[N],f1[N],f2[N],g[2][N*2],rev[N*2];
vector<int> st[19];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void pre()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<17;j<<=1)
	for(int k=0;k*2<=j;k++)
	g[i][j+k]=pw(3,mod-1+(i*2-1)*(mod-1)/j*k);
	for(int j=2;j<=1<<17;j<<=1)
	for(int k=0;k<j;k++)
	rev[j+k]=(rev[(k>>1)+j]>>1)|(k&1?(j>>1):0);
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=0;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][i+st]%mod;
		ntt[k]=(v1+v2)-(v1+v2>=mod?mod:0);
		ntt[k+(i>>1)]=(v1-v2)+(v1<v2?mod:0);
	}
	int inv=pw(s,!t?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> add(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	vector<int> c;
	for(int i=0;i<s1||i<s2;i++)c.push_back(((i<s1?a[i]:0)+(i<s2?b[i]:0))%mod);
	return c;
}
vector<int> mul(vector<int> a,vector<int> b)
{
	vector<int> c;
	int s1=a.size(),s2=b.size();
	if(s1+s2<=200)
	{
		for(int i=0;i<s1+s2-1;i++)c.push_back(0);
		for(int i=0;i<s1;i++)for(int j=0;j<s2;j++)c[i+j]=(c[i+j]+1ll*a[i]*b[j])%mod;
		return c;
	}
	int l=1;while(l<s1+s2)l<<=1;
	for(int i=0;i<l;i++)f1[i]=f2[i]=0;
	for(int i=0;i<s1;i++)f1[i]=a[i];
	for(int i=0;i<s2;i++)f2[i]=b[i];
	dft(l,f1,1);dft(l,f2,1);
	for(int i=0;i<l;i++)f1[i]=1ll*f1[i]*f2[i]%mod;
	dft(l,f1,0);
	for(int i=0;i<s1+s2-1;i++)c.push_back(f1[i]);
	return c;
}
vector<int> mulx(vector<int> a){vector<int> b;b.push_back(0);for(int i=0;i<a.size();i++)b.push_back(a[i]);return b;}
struct sth{vector<int> a,b;}tp[N],stp[19],as[N];
sth operator +(sth a,sth b){return (sth){mul(a.a,b.a),add(mul(a.a,b.b),mul(a.b,b.a))};}
struct sth1{int le;vector<int> s[2][2];}stp1[19];
sth1 operator +(sth1 a,sth1 b)
{
	sth1 c;c.le=a.le+b.le;
	for(int i=0;i<2;i++)a.s[i][1]=add(a.s[i][1],a.s[i][0]),b.s[1][i]=add(b.s[1][i],b.s[0][i]);
	for(int v1=0;v1<2;v1++)for(int s2=0;s2<2;s2++)
	{
		c.s[v1][s2]=add(c.s[v1][s2],mul(a.s[v1][1],b.s[1][s2]));
		int nt1=v1|(a.le==1),nt2=s2|(b.le==1);
		c.s[nt1][nt2]=add(c.s[nt1][nt2],mulx(mul(a.s[v1][0],b.s[0][s2])));
	}
	return c;
}
void dfs1(int u,int fa){sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u];}
void dfs2(int u,int v,int fa)
{
	int c1=0,c2=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs2(ed[i].t,ed[i].t,u);
	stp[1].a.clear();stp[1].b.clear();stp[1].a.push_back(1);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])
	{
		sth v1=as[ed[i].t];
		sth v2=(sth){add(v1.a,v1.b),mulx(v1.a)};
		for(int t=0;(((c1+1)>>t)&1)<((c1>>t)&1);t++)v2=v2+stp[c2],c2--;
		c1++;stp[++c2]=v2;
	}
	while(c2>1)stp[c2-1]=stp[c2-1]+stp[c2],c2--;
	tp[u]=stp[1];
	if(sn[u])dfs2(sn[u],v,u);
	if(u==v)
	{
		c1=c2=0;
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)stp1[1].s[i][j].clear();
		stp1[1].s[0][0].push_back(1);
		for(int i=u;i;i=sn[i])
		{
			sth1 st;
			st.le=1;st.s[0][0]=tp[i].a;st.s[1][1]=tp[i].b;
			for(int t=0;(((c1+1)>>t)&1)<((c1>>t)&1);t++)st=stp1[c2]+st,c2--;
			c1++;stp1[++c2]=st;
		}
		while(c2>1)stp1[c2-1]=stp1[c2-1]+stp1[c2],c2--;
		as[u]=(sth){add(stp1[1].s[0][0],stp1[1].s[0][1]),add(stp1[1].s[1][0],stp1[1].s[1][1])};
	}
}
int main()
{
	pre();
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b+n);
	for(int i=1;i<=n*2;i++)if(!sz[i])
	{
		dfs1(i,0);dfs2(i,i,0);
		vector<int> s2=add(as[i].a,as[i].b);
		for(int t=0;(((ct+1)>>t)&1)<((ct>>t)&1);t++)s2=mul(s2,st[ct2]),ct2--;
		ct++;st[++ct2]=s2;
	}
	while(ct2>1)st[ct2-1]=mul(st[ct2-1],st[ct2]),ct2--;
	int as=0,s1=1,sz1=st[1].size();
	for(int i=n;i>=0;i--)as=(as+1ll*(i&1?mod-1:1)*s1%mod*(i>=sz1?0:st[1][i]))%mod,s1=1ll*s1*(n-i+1)%mod;
	printf("%d\n",as);
}
```

##### #305. Gem Island 2

###### Problem

有 $n$ 个人，初始每个人有一个 $v_i=1$

每一天随机选择一个人，以 $\frac{v_i}{\sum v_i}$ 的概率选中 $i$，然后将这个人的 $v_i$ 加一。

求出 $m$ 天后，$v_i$ 最大的 $k$ 个人的 $v_i$ 和，模 $998244353$。

$n,m,k\leq 1.5\times 10^7$

$1s,512MB$

###### Sol

可以将这个概率乘上 $\frac{(n+m)!}{n!}$ 变为方案数，考虑一种最后 $v_{1,2,...,n}$ 出现的方案数。

考虑每一次加一给了谁，则这部分的方案数为 $\frac{m!}{\prod(v_i-1)!}$

再考虑每一个人每次加一的系数，这部分为 $\prod(v_i-1)!$，总的方案数即为上两个相乘。因此可以发现每种可能的 $v$ 出现概率相等。

因此可以计算每种 $v_i$ 的前 $k$ 大和，再除以 $C_{n+m-1}^m$ 即为答案。

考虑期望线性性，对于每个 $x$ ，设 $v_i\geq x$ 的人有 $a$ 个，则对答案贡献为 $\min(a,k)$。

考虑计算有 $x$ 个人权值大于等于 $y$ 的情况数，记这个值为 $f_{x,y}$。

考虑 $g_{x,y}=\sum_{i\geq x}f_{i,y}C_i^x$，这相当于选 $x$ 个人，让他们的权值大于等于 $y$ 的方案数，根据插板这等于 $C_n^x*C_{n+m-1-(y-1)x}^{n-1}$。根据二项式反演有 $f_{x,y}=\sum_{i\geq x}g_{i,y}C_i^x(-1)^{i-x}$，这部分的贡献为 $\sum f_{x,y}\min(x,k)$。

考虑计算一个 $g_{x,y}$ 对答案的贡献，这等于 $\sum_{i\leq x}C_x^i(-1)^{x-i}\min(i,k)$。记这个值为 $v_{x,k}$，则：
$$
v_{n,k}=\sum_{i\leq n}(C_{n-1}^i+C_{n-1}^{i-1})(-1)^{n-i}*\min(i,k)\\
=\sum_{i\leq n}C_{n-1}^i(-1)^{n-i}*\min(i,k)+\sum_{i\leq n}C_{n-1}^{i-1}(-1)^{n-i}*\min(i,k)\\
=-\sum_{i\leq n}C_{n-1}^i(-1)^{(n-1)-i}*\min(i,k)+\sum_{i\leq n}C_{n-1}^{i-1}(-1)^{(n-1)-(i-1)}*(\min(i-1,k-1)+1)\\
=-v_{n-1,k}+v_{n-1,k-1}+\sum_{i\leq n}C_{n-1}^{i-1}(-1)^{(n-1)-(i-1)}\\
=v_{n-1,k-1}-v_{n-1,k}(n>1)\\
$$
可以发现 $v_{1,k}=1(k>0),v_{k,0}=0$，因此考虑上式的组合意义可以得到：
$$
v_{n,k}=\sum_{i=0}^{k-1}C_{n-1}^i(-1)^{n-1-i}
$$
此时可以发现：
$$
v_{n,k}=\sum_{i=0}^{k-1}(-1)^{n-1-i}(C_{n-2}^i+C_{n-2}^{i-1})(n>1)\\
v_{n,k}=-\sum_{i=0}^{k-1}(-1)^{n-2-i}C_{n-2}^i+\sum_{i=0}^{k-2}(-1)^{n-2-i}C_{n-2}^{i}\\
v_{n,k}=(-1)^{n-k}C_{n-2}^{k-1}
$$
因此， $v_{1,k}=1,v_{n,k}=(-1)^{n-k}C_{n-2}^{k-1}(n>1)$，考虑计算答案：
$$
ans=n+\sum_{y>0}\sum_{x=1}^ng_{x,y+1}v_{x,k}\\
=n+\sum_{y>0}(g_{1,y+1}+\sum_{x=2}^ng_{x,y+1}(-1)^{x-k}C_{x-2}^{k-1})\\
=n+\sum_{y>0}n*C_{n+m-1-y}^{n-1}+\sum_{y>0}\sum_{x=2}^nC_n^xC_{n+m-1-xy}^{n-1}(-1)^{x-k}C_{x-2}^{k-1}\\
=n+n*C_{n+m-1}^{n}+\sum_{T>0}C_{n+m-1-T}^{n-1}\sum_{x|T,x>1}C_n^x(-1)^{x-k}C_{x-2}^{k-1}
$$
此时问题相当于给出序列 $v_i=C_n^i(-1)^{i-k}C_{i-2}^{k-1}$，求另外一个序列 $t_i=\sum_{j|i}v_j$。

这可以使用狄利克雷前缀和，注意到如果把 $n$ 表示成分解形式 $\prod p_i^{q_i}$，则 $x|y$ 当且仅当分解出的 $q_x,q_y$ 满足 $q_x$ 的每一项都小于等于 $q_y$ 的每一项，因此可以做高维前缀和。

可以发现做高维前缀和相当于枚举所有质数 $p$ ，对于每个质数 $p$，从小到大枚举 $i$，然后令 $v_{p*i}+=v_i$。可以发现只需要枚举 $\frac np$ 项，复杂度为 $O(n\log\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 15000005
#define mod 998244353
int n,m,k,fr[N*2],ifr[N*2],as,v[N],ch[N],pr[N],ct;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int C(int x,int y){return 1ll*fr[x]*ifr[y]%mod*ifr[x-y]%mod;}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	fr[0]=1;for(int i=1;i<=n+m;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n+m]=pw(fr[n+m],mod-2);for(int i=n+m;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	as=(1ll*k*C(n+m-1,n-1)%mod+1ll*C(n-1+m,n)*n)%mod;
	for(int j=k+1;j<=n;j++)v[j]=1ll*C(n,j)*C(j-2,k-1)%mod*((j-k)&1?mod-1:1)%mod;
	for(int i=2;i<=m;i++)
	{
		if(!ch[i])
		{
			pr[++ct]=i;
			for(int j=1;i*j<=m;j++)v[i*j]=(v[i*j]+v[j])%mod;
		}
		for(int j=1;j<=ct&&i*pr[j]<=m;j++)
		{
			ch[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
	for(int i=1;i<=m;i++)as=(as+1ll*C(n-1+m-i,n-1)*v[i])%mod;
	printf("%d\n",1ll*as*pw(C(n+m-1,n-1),mod-2)%mod);
}
```

