---
title: 2021 HDU多校题解
date: '2021-09-20 16:24:05'
updated: '2021-09-20 18:58:43'
tags: Mildia
permalink: Mawaruri/
description: 2021 HDU多校
mathjax: true
---

### 2021多校题解

大概对着每一场看了一遍.jpg

不会做的：**3H** **5J** **6F** **8G** **10E**

还没看但可能去看的： **1K** **6I** **6K** **7I** **10G**

还没写的：**1B** **2G** **5A** **10F**

以下所有题目默认多测，且 $T$ 指极限数据组数。

~~sbHDU不支持行末空格~~

##### 1B Rocket land

###### Problem

给定二维平面上 $n$ 个点 $p_i(x_i,y_i)$，以及 $n$ 个 $r_i,v_i$，对于每个 $i$ 计算：
$$
as_i=\sum_{j\leq i}[dis(p_i,p_j)\leq r_j]v_j
$$
其中 $dis$ 为欧几里得距离。

$n\leq 2\times 10^5$，数据随机

$20s,512MB$

###### Sol

可以发现问题相当于给定 $n$ 个点，支持将一个圆内部的所有点加上一个权值，询问一个点的权值、

因为数据随机，可以直接使用KD-Tree解决问题，复杂度 $O(n^{1.5})$。

###### Code

没写

~~非常喜闻乐见的是我没写过KDTree~~

##### 1C Puzzle loop

###### Problem

有一个 $n\times m$ 的网格图，网格图中有 $(n-1)\times (m-1)$ 个方格，一些方格中有一个 $\in\{0,1\}$ 的数字。

你需要选择若干条网格图的边，满足如下限制：

1. 选择的边在网格图中构成若干个环。
2. 对于一个写有数字 $0$ 的方格，它周围的四条边必须选择偶数条。
3. 对于一个写有数字 $1$ 的方格，它周围的四条边必须选择奇数条。

求合法方案数，模 $998244353$

$n,m\leq 17,T\leq 10$

$1s,512MB$

###### Sol

~~状压dp没有前途~~

可以发现构成环等价于每个点的度数都是偶数。因此可以看成若干个限制，每个限制要求与某个点相邻的边选的数量为偶数。

此时可以发现所有限制都可以表示成这种形式，因此答案相当于一个异或方程组的解数。

考虑对方程组直接消元，可以发现如果有解答案一定是 $2^k$，其中 $k$ 为自由元数量。bitset消元即可。

复杂度 $O(\frac{n^3m^3}{32})$

###### Code

15ms,1.3MB

```cpp
#include<cstdio>
#include<bitset>
using namespace std;
#define N 551
#define M 25
#define mod 998244353
int T,n,m,ct,fg[N],as;
bitset<N> tp[N];
char s[M][M];
void ins(bitset<N> s1)
{
	for(int i=1;i<=ct;i++)if(s1[i]&&fg[i]!=-1)
	if(tp[i][i])s1^=tp[i];
	else {
	tp[i]=s1;return;}
	as=2*as%mod;
}
int getid(int x,int y){return x*m-m+y;}
int getid2(int x,int y){return n*m+(x-1)*(m-1)+y;}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&m);ct=n*m+(n-1)*(m-1);as=1;
		for(int i=1;i<=ct;i++)tp[i].reset(),fg[i]=0;
		for(int i=1;i<n;i++)
		{
			scanf("%s",s[i]+1);
			for(int j=1;j<m;j++)
			if(s[i][j]=='.')fg[getid2(i,j)]=-1;
			else if(s[i][j]=='1')fg[getid2(i,j)]=1;
		}
		for(int i=1;i<=n;i++)
		for(int j=1;j<m;j++)
		{
			bitset<N> s1;
			s1.set(getid(i,j),1);s1.set(getid(i,j+1),1);
			if(i>1)s1.set(getid2(i-1,j),1);if(i<n)s1.set(getid2(i,j),1);
			ins(s1);
		}
		for(int i=1;i<n;i++)
		for(int j=1;j<=m;j++)
		{
			bitset<N> s1;
			s1.set(getid(i,j),1);s1.set(getid(i+1,j),1);
			if(j>1)s1.set(getid2(i,j-1),1);if(j<m)s1.set(getid2(i,j),1);
			ins(s1);
		}
		bitset<N> s1;
		for(int i=1;i<=ct;i++)if(fg[i]!=-1&&fg[i]!=(int)s1[i])
		if(!tp[i][i])as=0;
		else s1^=tp[i];
		printf("%d\n",as);
	}
}
```

##### 1D Another thief in a Shop

###### Problem

有 $n$ 种物品，第 $i$ 种物品的重量为 $a_i$。

每种物品你都可以拿任意数量，求拿总重量为 $k$ 的物品的方案数，模 $10^9+7$

$n\leq 100,k\leq 10^{18},a_i\leq 10,T\leq 10$

$2s,256MB$

###### Sol

对于一种物品，设选的数量为 $v_i$，考虑枚举 $v_i\bmod \frac{2520}{a_i}$。这部分选的总重量不超过 $2520n$，可以直接背包。

对于剩下的部分，可以发现相当于有 $n$ 种重量均为 $2520$ 的物品。此时设剩下的重量为 $2520*m$，显然方案数为 $C_{n+m-1}^{n-1}$。直接暴力算组合数即可。

复杂度 $O(2520n^2)$

###### Code

1.279s,2.2MB

```cpp
#include<cstdio>
using namespace std;
#define N 252102
#define mod 1000000007
#define ll long long
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int T,n,dp[N],a;
ll k;
int C(ll x,int y)
{
	if(x<y)return 0;
	int as=1;
	for(int i=1;i<=y;i++)as=1ll*as*pw(i,mod-2)%mod*((x-i+1)%mod)%mod;
	return as;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%lld",&n,&k);
		for(int i=0;i<=2520*n;i++)dp[i]=0;
		dp[0]=1;
		for(int i=1;i<=n;i++)
		{
			scanf("%d",&a);
			for(int j=a;j<=2520*i;j++)dp[j]=(dp[j]+dp[j-a])%mod;
			for(int j=2520*i;j>=2520;j--)dp[j]=(dp[j]+mod-dp[j-2520])%mod;
		}
		int as=0;
		for(int i=0;i<=2520*n&&i<=k;i++)if((k-i)%2520==0)as=(as+1ll*dp[i]*C((k-i)/2520+n-1,n-1))%mod;
		printf("%d\n",as);
	}
}
```

##### 2F I love sequences

###### Problem

设 $a=(a_ka_{k-1}...a_0)_3,b=(b_kb_{k-1}...b_0)_3$，定义 $a\oplus b=(c_kc_{k-1}...c_0)_3$，其中 $c_i=\gcd(a_i,b_i)$

给定 $n$ 以及长度为 $n$ 的三个序列 $a,b,c$

定义 $d_{p,k}=\sum_{1\leq i,j\leq \frac np,i\oplus j=k}a_ib_j$，求：
$$
\sum_{p=1}^{n}\sum_{k=0}^{+\infty}d_{p,k}*c_p^k
$$
答案模 $10^9+7$

$n\leq 2\times 10^5$

$1s,512MB$

###### Sol

考虑对于每一个 $p$ 计算 $d_p$，$i\oplus j$ 可以看成一个三进制下的按位变换，可以发现变换后的位置不会超过三倍的原位置，因此总长度为 $O(n\log n)$，只需快速求出 $d_p$。

考虑每一位上的情况，可以得到如下变换：

| $a_k / b_k$ | $0$  | $1$  | $2$  |
| ----------- | ---- | ---- | ---- |
| $0$         | $0$  | $1$  | $2$  |
| $1$         | $1$  | $1$  | $1$  |
| $2$         | $2$  | $1$  | $2$  |

考虑将 $0$ 看成 $3$，则有：

| $a_k / b_k$ | $1$  | $2$  | $3$  |
| ----------- | ---- | ---- | ---- |
| $1$         | $1$  | $1$  | $1$  |
| $2$         | $1$  | $2$  | $2$  |
| $3$         | $1$  | $2$  | $3$  |

可以发现这时变换变为每一位上取 $\min$。因此使用高维后缀和变换做类似FWT的操作即可。

复杂度 $O(n\log^2 n)$

###### Code

546ms,36.9MB

```cpp
#include<cstdio>
using namespace std;
#define N 550001
#define mod 1000000007
int n,a[N],b[N],c[N],f[N],g[N],tr[N][13],h[N],as;
int main()
{
	scanf("%d",&n);
	for(int i=1,l=1;i<=12;i++,l*=3)
	for(int j=0;j<3;j++)
	for(int k=0;k<l;k++)
	tr[j*l+k][i]=tr[k][i-1]+(j+2)%3*l;
	for(int i=1;i<=n;i++)scanf("%d",&a[i]);
	for(int i=1;i<=n;i++)scanf("%d",&b[i]);
	for(int i=1;i<=n;i++)scanf("%d",&c[i]);
	for(int i=1;i<=n;i++)
	{
		int l=n/i,le=1,l1=0;
		while(le<=l)le*=3,l1++;
		for(int j=0;j<le;j++)f[j]=g[j]=0;
		for(int j=1;j<=l;j++)f[tr[j][l1]]=a[j],g[tr[j][l1]]=b[j];
		for(int l2=1,l0=1;l2<=l1;l2++,l0*=3)
		for(int j=0;j<le;j+=l0*3)
		for(int k=0;k<l0;k++)
		for(int t=1;t>=0;t--)
		f[j+k+t*l0]=(f[j+k+t*l0]+f[j+k+(t+1)*l0])%mod,g[j+k+t*l0]=(g[j+k+t*l0]+g[j+k+(t+1)*l0])%mod;
		for(int j=0;j<le;j++)f[j]=1ll*f[j]*g[j]%mod;
		for(int l2=1,l0=1;l2<=l1;l2++,l0*=3)
		for(int j=0;j<le;j+=l0*3)
		for(int k=0;k<l0;k++)
		for(int t=0;t<2;t++)
		f[j+k+t*l0]=(f[j+k+t*l0]+mod-f[j+k+(t+1)*l0])%mod;
		h[0]=1;for(int j=1;j<le;j++)h[j]=1ll*c[i]*h[j-1]%mod;
		for(int j=0;j<le;j++)
		{
			int tp=tr[tr[j][l1]][l1];
			as=(as+1ll*f[j]*h[tp])%mod;
		}
	}
	printf("%d\n",as);
}
```

##### 2G I love data structure

###### Problem

给两个长度为 $n$ 的序列 $a,b$，支持如下操作：

1. 给定 $l,r,v,0/1$，对于所有的 $i\in[l,r]$，将 $a_i$ 加上 $v$ 或者对于所有的 $i\in[l,r]$将 $b_i$ 加上 $v$
2. 给定 $l,r$，对于所有的 $i\in[l,r]$，令 $a_i,b_i\leftarrow3a_i+2b_i,3a_i-2b_i$
3. 给定 $l,r$，对于所有的 $i\in[l,r]$，令 $a_i,b_i\leftarrow b_i,a_i$
4. 给定 $l,r$，求 $\sum_{i=l}^ra_ib_i$，答案模 $10^9+7$

$n,q\leq 2\times 10^5$

$5s,256MB$

###### Sol

考虑维护 $\sum a^2,\sum ab,\sum b^2,\sum a,\sum b,\sum 1$ ，可以发现上面的所有修改操作都可以看成这六个元素之间的一个线性变换。

在线段树上处理询问，相当于区间乘矩阵区间求和，直接做即可。

复杂度 $O(6^3*q\log n)$，但是矩阵显然不满，能过。

###### Code

没写

##### 2I I love triples

###### Problem

给一个长度为 $n$ 的序列 $a$，求有多少个数对 $(i,j,k)$ 满足：

1. $1\leq i<j<k\leq n$
2. $a_ia_ja_k$ 为完全平方数

$n,a_i\leq 10^5,T\leq 6$

$1.5s,512MB$

###### Sol

显然可以对于所有 $a_i$ 去掉平方因子，考虑此时可能贡献答案的权值对，即满足 $abc$ 为完全平方数的无平方因子数对 $(a,b,c)$ 数量。

打表可以发现，如果将数对看成无序的，则 $10^5$ 内可能的数对只有不到 $1.5\times 10^7$ 个，因此可以考虑枚举这样的数对然后 $O(1)$ 计算答案。

考虑快速枚举，在满足条件的 $(a,b,c)$ 中，对于每一个质因子 $p$ ，这个质因子要么在 $a,b,c$ 中都不出现，要么正好在两个质因子中出现。

从小到大加入质因子，设 $dfs(a,b,c,k)$ 表示考虑了前 $k$ 个质数，当前三个数为 $a,b,c$，则可以递归到 $dfs(a*p_k,b*p_k,c,k+1),dfs(a*p_k,b,c*p_k,k+1),dfs(a,b*p_k,c*p_k,k+1),dfs(a,b,c,k+1)$

显然，当 $a,b,c$ 中第二大的数大于 $\frac{10^5}{p_k}$ 时，之后不可能再加入新的质因子，此时可以结束递归。

每一步递归一定至少多出一个新状态，因此复杂度即为 $O($合法数对数$)$，可以在加入第一个质因子时只取三种情况中的一种减小常数。

###### Code

514ms,2.4MB

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define ll long long
int T,n,pr[N],ch[N],s1,a,ct[N],mx,tp[N];
ll as;
void prime(int n)
{
	tp[1]=1;
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++s1]=i,tp[i]=i;
		for(int j=1;j<=s1&&i*pr[j]<=n;j++)
		{
			ch[i*pr[j]]=1,tp[i*pr[j]]=tp[i]*pr[j];
			if(i%pr[j]==0)
			{
				if(tp[i]%pr[j]==0)tp[i*pr[j]]=tp[i]/pr[j];
				break;
			}
		}
	}
}
void dfs(int d,int a,int b,int c)
{
	if(a>b)a^=b^=a^=b;
	if(b<=c)
	{
		if(a==c)as+=1ll*ct[a]*(ct[a]-1)*(ct[a]-2)/6;
		else if(a==b)as+=1ll*ct[a]*(ct[a]-1)*ct[c]/2;
		else if(b==c)as+=1ll*ct[c]*(ct[c]-1)*ct[a]/2;
		else as+=1ll*ct[a]*ct[b]*ct[c];
	}
	for(int i=d;i<=s1;i++)
	{
		if(1ll*a*pr[i]>mx||(1ll*b*pr[i]>mx&&1ll*c*pr[i]>mx))break;
		if(1ll*c*pr[i]<=mx)dfs(i+1,a*pr[i],b,c*pr[i]);
		if(1ll*b*pr[i]<=mx&&1ll*c*pr[i]<=mx&&a<b)dfs(i+1,a,b*pr[i],c*pr[i]);
		if(1ll*b*pr[i]<=mx)dfs(i+1,a*pr[i],b*pr[i],c);
	}
}
int main()
{
	prime(1e5);
	scanf("%d",&T);
	while(T--)
	{
		as=mx=0;
		scanf("%d",&n);
		for(int i=1;i<=n;i++)
		{
			scanf("%d",&a);a=tp[a];
			if(mx<a)mx=a;ct[a]++;
		}
		dfs(1,1,1,1);
		printf("%lld\n",as);
		for(int i=1;i<=mx;i++)ct[i]=0;
	}
}
```

##### 3A Bookshop

###### Problem

有一棵 $n$ 个点的树，点有点权 $c_i$。

有 $q$ 次询问，每次给定 $u_i,v_i,w_i$，你在树上从 $u_i$ 走到 $v_i$ ，你经过一个点 $x$ 时，如果 $w_i\geq c_x$，则将 $w_i$ 减去 $c_x$，否则 $w_i$ 不变。

对于每次询问，求出走完这条路径后 $w_i$ 的值。

$n,q\leq 10^5,v_i,w_i\leq 10^9,\sum n,\sum q\leq 8\times 10^5$

$12s,512MB$

###### Sol

考虑链上的问题，~~但直接做仍然有困难~~。

在链上难以一次求出一个询问的答案，考虑在 $w_i$ 减半时停止，求出停止时所在的位置，这样做 $O(\log v)$ 次即可得到答案。此时的询问有如下性质：

1. 所有满足 $c_x\leq \frac{w_i}2$ 的点一定会让 $w_i$ 减小。
2. 如果一个满足 $c_x\in[\frac{w_i}2+1,w_i]$ 的点让 $w_i$ 减小了，则会在此处停止。

因此停止时只可能有两种情况：

1. 经过的前若干个 $c_x\leq \frac{w_i}2$ 的点的和大于 $\frac{w_i}2$
2. 到达一个 $c_x\in[\frac{w_i}2+1,w_i]$ 的点，这个点的 $c_x$ 加上这个点前所有 $\leq \frac{w_i}2$ 的点权和不超过 $w_i$。

对于第一种情况，主席树或者离线后按照 $w_i$ 从大到小处理询问线段树维护即可找到停止的位置。

对于第二种情况，相当于找到最小的 $i$ 满足 $c_i+\sum_{j\in[l,i),c_i\leq\frac{w_i}2}c_j\leq w_i$ ，考虑在线段树上记录一个区间 $[l,r]$ 的：

1. $\sum_{i\in[l,r],c_i\leq\frac{w_i}2}c_i$
2. $\min_{i\in[l,r],c_i\in[\frac{w_i}2+1,w_i]}c_i+\sum_{j\in[l,i),c_i\leq\frac{w_i}2}c_j$

即可在线段树上查询。

因为空间问题，考虑离线做整个过程，显然每次处理 $w_i$ 会变小，因此可以每次拿出 $w_i$ 最大的询问，处理这个询问的一步。

复杂度 $O(n\log n+q\log n\log v)$

对于树上的情况，可以直接套树剖，变成 $O(\log n)$ 段。可以发现减半的次数不变，因此此时操作次数不会超过 $O(\log n+\log v)$

复杂度 $O(n\log n+q\log n(\log n+\log v))$

###### Code

9.5s,113.1MB

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<set>
using namespace std;
#define N 100500
#define ll long long
int T,n,q,a,b,c,head[N],cnt,v[N],sz[N],sn[N],tp[N],id[N],ct,dep[N],f[N],v1[N],as[N],c1,l,r,vl;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void dfs0(int u,int fa)
{
	sz[u]=1;sn[u]=0;f[u]=fa;dep[u]=dep[fa]+1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs0(ed[i].t,u);sz[u]+=sz[ed[i].t];
		if(sz[ed[i].t]>sz[sn[u]])sn[u]=ed[i].t;
	}
}
void dfs1(int u,int fa,int v)
{
	tp[u]=v;id[u]=++ct;
	if(sn[u])dfs1(sn[u],u,v);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs1(ed[i].t,u,ed[i].t);
}
struct sth{int l,r,fg;};
vector<sth> solve1(int x,int y)
{
	vector<sth> s1,s2;
	while(tp[x]!=tp[y])
	{
		if(dep[tp[x]]>dep[tp[y]])s1.push_back((sth){id[tp[x]],id[x],1}),x=f[tp[x]];
		else s2.push_back((sth){id[tp[y]],id[y],0}),y=f[tp[y]];
	}
	if(dep[x]>dep[y])s1.push_back((sth){id[y],id[x],1});
	else s1.push_back((sth){id[x],id[y],0});
	for(int i=s2.size();i>=1;i--)s1.push_back(s2[i-1]);
	return s1;
}
vector<sth> st[N];
struct node{int l,r;ll su,sl,sr;}e[N*4];
void pushup(int x)
{
	e[x].su=e[x<<1].su+e[x<<1|1].su;
	e[x].sl=min(e[x<<1].sl,e[x<<1|1].sl+e[x<<1].su);
	e[x].sr=min(e[x<<1|1].sr,e[x<<1].sr+e[x<<1|1].su);
}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;e[x].sl=e[x].sr=1e18;e[x].su=v1[l];
	if(l==r)return;
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	pushup(x);
}
void modify(int x,int s,ll v1,ll v2)
{
	if(e[x].l==e[x].r){e[x].su=v1;e[x].sl=e[x].sr=v2;return;}
	int mid=(e[x].l+e[x].r)>>1;
	modify(x<<1|(mid<s),s,v1,v2);
	pushup(x);
}
struct sth1{int a,b;};
sth1 query1(int x,int nw,int fg)
{
	if(e[x].l==e[x].r)
	{
		if(nw>=v1[e[x].l])nw-=v1[e[x].l];
		return (sth1){e[x].l,nw};
	}
	int ls=x<<1,rs=x<<1|1;
	ls^=fg;rs^=fg;
	if((fg?e[ls].sr:e[ls].sl)<=nw||e[ls].su>=nw-(vl/2))return query1(ls,nw,fg);
	else return query1(rs,nw-e[ls].su,fg);
}
sth1 query(int x,int nw,int fg)
{
	if(e[x].l>=l&&e[x].r<=r)
	{
		if(e[x].su<=nw-(vl/2)&&(fg?e[x].sr:e[x].sl)>nw)return (sth1){fg?e[x].l:e[x].r,nw-e[x].su};
		return query1(x,nw,fg);
	}
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)return query(x<<1,nw,fg);
	else if(mid<l)return query(x<<1|1,nw,fg);
	else if(!fg)
	{
		sth1 s1=query(x<<1,nw,fg);
		if(s1.a<mid)return s1;
		return query(x<<1|1,s1.b,fg);
	}
	else
	{
		sth1 s1=query(x<<1|1,nw,fg);
		if(s1.a>mid+1)return s1;
		return query(x<<1,s1.b,fg);
	}
}
struct sth2{int v,a,b,c;}fu[N*2];
bool operator <(sth2 a,sth2 b){return a.v==b.v?(a.a==b.a?a.b<b.b:a.a<b.a):a.v>b.v;}
vector<sth2> fu2[N*2];
void ins(sth2 f1)
{
	int vl=upper_bound(fu+1,fu+n*2+1,f1)-fu;
	fu2[vl].push_back(f1);
}
int rd(){int as=0;char c=getchar();while(c<'0'||c>'9')c=getchar();while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();return as;}
void solve()
{
	n=rd();q=rd();
	for(int i=1;i<=n;i++)head[i]=0;cnt=ct=0;
	for(int i=1;i<=n;i++)v[i]=rd();
	for(int i=1;i<n;i++)a=rd(),b=rd(),adde(a,b);
	dfs0(1,0);dfs1(1,0,1);
	for(int i=1;i<=n;i++)v1[id[i]]=v[i];
	build(1,1,n);
	for(int i=1;i<=n;i++)fu[i*2-1]=(sth2){2*v1[i]-1,-1,i,0},fu[i*2]=(sth2){v1[i]-1,-2,i,0};
	sort(fu+1,fu+n*2+1);
	for(int i=1;i<=q;i++)
	{
		a=rd(),b=rd(),c=rd();as[i]=c;
		st[i]=solve1(a,b);ins((sth2){c,i,0,id[a]});
	}
	for(int i=1;i<=n*2+1;i++)
	{
		while(!fu2[i].empty())
		{
			sth2 tp=fu2[i].back();fu2[i].pop_back();
			sth f2=st[tp.a][tp.b];
			int nw=tp.c;
			if(f2.fg)f2.r=nw;else f2.l=nw;
			l=f2.l;r=f2.r;vl=tp.v;
			sth1 f1=query(1,tp.v,f2.fg);
			as[tp.a]-=tp.v-f1.b;
			if(f1.a==(f2.fg?f2.l:f2.r)){if(tp.b+1<st[tp.a].size()){sth f3=st[tp.a][tp.b+1];ins((sth2){f1.b,tp.a,tp.b+1,f3.fg?f3.r:f3.l});}}
			else ins((sth2){f1.b,tp.a,tp.b,f1.a+(f2.fg?-1:1)});
		}
		if(i<=n*2)
		{
			sth2 tp=fu[i];
			if(tp.a==-1)modify(1,tp.b,0,v1[tp.b]);
			else if(tp.a==-2)modify(1,tp.b,0,1e18);
		}
	}
	for(int i=1;i<=q;i++)printf("%d\n",as[i]),as[i]=0;
}
int main(){T=rd();while(T--)solve();}
```

##### 3B Destinations

###### Problem

有一棵 $n$ 个点的树，给定 $m$ 对 $(s,t_1,t_2,t_3,w_1,w_2,w_3)$。

对于每一对，你需要在 $(s,t_1),(s,t_2),(s,t_3)$ 中选择一条路径，费用分别为 $w_1,w_2,w_3$

你需要满足选择的 $m$ 条路径两两不点相交，求最小总费用或输出无解。

$n\leq 2\times 10^5,m\leq 10^5,\sum n\leq 10^6,\sum m\leq 3\times 10^5$

$2.5s,512MB$

###### Sol1

对于一个子树，只考虑 $s$ 在这个子树内的部分时，这个子树内最多向外延伸一条链，且最后的方案中最多子树外向子树内延伸一条链。

设 $f_u$ 表示只考虑 $u$ 子树内部的限制，且所有路径都在 $u$ 子树内部时，这部分的最小总代价。设 $dp_{u,v}$ 表示：

1. 如果 $v$ 在 $u$ 子树内，则表示考虑 $u$ 子树内的限制，且 $v$ 到 $u$ 的路径上的点不能被覆盖时的最小代价。
2. 否则，表示表示考虑 $u$ 子树内的限制，这部分有一条路径向外延伸至 $v$ 时的最小代价。

考虑在一个点上合并儿子的 $dp$，此时有如下情况：

1. 对于 $x$ 在 $u$ 子树内的转移，此时设 $x$ 在 $u$ 的儿子 $v$ 的子树内，则显然 $dp_{u,x}=dp_{v,x}+\sum_{i\in son_u,i\neq v}f_i$
2. 对于 $x$ 在子树外的转移，考虑枚举这条路径来着哪个子树，则有：$dp_{u,x}=\min_{i\in son_u}dp_{i,x}+\sum_{j\neq i,j\in son_u}f_j$
3. 对于 $f_u$ 的转移，第一种情况是没有路径经过 $u$，此时为 $\sum_{i\in son_u}f_v$
4. 第二种情况为有一条路径经过 $u$，枚举这条路径所在的子树，可以发现此时的值为：

$$
\min_{x,y\in son_u}\min_{v\in son_x\or v\in son_y}(dp_{x,v}+dp_{y,v})+\sum_{i\in son_u,i\neq x,i\neq y}f_i
$$

此处还需处理路径端点为 $u$ 的情况，但方式类似。

可以发现，使用动态开点线段树维护 $dp$，所有不存在的位置代表值为 $+\infty$，按照dfs序作为下标，依次合并儿子的 $dp$。在合并叶子时只需要判断叶子属于哪个区间即可更新上面的情况。

最后考虑 $u$ 是某一个 $s$ 的情况，先求出不考虑这个点出发路径的 $dp_u$ 和 $f_u$，考虑一条路径 $(u,t,w)$ 的转移：

1. 如果 $t$ 在 $u$ 子树内，则相当于从 $dp_{u,t}+w$ 转移到新的 $f_u$
2. 否则相当于从 $dp_{u,u}+w$ 转移到新的 $dp_{u,t}$。

然后清空之前的部分即可。

复杂度 $O(m\log n)$

###### Sol2

可以看成选若干条点不相交的路径，要求最大化选的路径条数再最小化总和，这个问题和直接最大化总和的形式相同。

设 $dp_{u,v}(v\in subtree_u)$ 表示 $u$ 的子树内，$u$ 到 $v$ 的路径上的点不能选，此时内部的最优值。再记录 $f_u$ 表示 $u$ 

对于一条路径 $(s,t_i,w_i)$，考虑在它的LCA $l$ 处处理这条路径，设 $s,t_i$ 所在的子树为 $a,b$。此时将 $dp_{a,s}+dp_{b,t}+\sum_{v\in son_l,v\neq a,b}f_v$ 转移到 $f_u$ 即可。

如果在 $l$ 处不选路径，则转移为 $dp_{l,x}=dp_{u,x}+\sum_{v\in son_l,v\neq u}f_v$。

因为 $dp_{u}$ 中的位置为 $u$ 子树中的所有位置，如果使用dfs等方式从下往上合并，则合并时每个 $x$ 只会在一个 $dp_{u,x}$ 中出现，因此使用一个BIT维护当前每个位置在所在的 $dp_{u,x}$ 的值即可。

复杂度 $O(m\log n)$ ，常数更小

###### Code

2.293s,126.5MB

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define M 6000060
#define ll long long
int T,n,m,a,b,head[N],cnt,id[N],rb[N],ct1,is[N],s[N][3],v[N][3];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
int rt[N],ch[M][2],l1,r1,l2,r2,ct;
ll lz[M],vl[M],as,dp[N];
void doit(int x,ll v){if(!x)return;lz[x]+=v;vl[x]+=v;}
void pushdown(int x){doit(ch[x][0],lz[x]);doit(ch[x][1],lz[x]);lz[x]=0;}
void pushup(int x){vl[x]=min(vl[ch[x][0]],vl[ch[x][1]]);}
void ins(int x,int l,int r,int s,ll v)
{
	pushdown(x);
	if(l==r){vl[x]=min(vl[x],v);return;}
	int mid=(l+r)>>1,fg=0;
	if(mid>=s)fg=0,r=mid;
	else fg=1,l=mid+1;
	if(!ch[x][fg])ch[x][fg]=++ct,vl[ct]=1e17;
	ins(ch[x][fg],l,r,s,v);pushup(x);
}
int doit(int x,int l,int r,int l1,int r1)
{
	if(!x||r<l1||r1<l)return x;
	if(l==r){vl[x]=1e17;return x;}
	pushdown(x);
	int mid=(l+r)>>1;
	ch[x][0]=doit(ch[x][0],l,mid,l1,r1);
	ch[x][1]=doit(ch[x][1],mid+1,r,l1,r1);
	return x;
}
int merge(int x,int y,int l,int r)
{
	if(!y)return doit(x,l,r,l2,r2);
	if(!x)return doit(y,l,r,l1,r1);
	pushdown(x);pushdown(y);
	if(l==r)
	{
		if(l<l1||l>r2)vl[x]=min(vl[x],vl[y]);
		else if(l<=r1)as=min(as,vl[x]+vl[y]);
		else as=min(as,vl[x]+vl[y]),vl[x]=vl[y];
		return x;
	}
	int mid=(l+r)>>1;
	ch[x][0]=merge(ch[x][0],ch[y][0],l,mid);
	ch[x][1]=merge(ch[x][1],ch[y][1],mid+1,r);
	pushup(x);return x;
}
ll query(int x,int l,int r,int s)
{
	if(l==r||!x)return vl[x];
	pushdown(x);
	int mid=(l+r)>>1;
	if(mid>=s)return query(ch[x][0],l,mid,s);
	else return query(ch[x][1],mid+1,r,s);
}
void dfs0(int u,int fa)
{
	id[u]=++ct1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);
	rb[u]=ct1;
}
void dfs(int u,int fa)
{
	ll c1=0,su=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),c1+=dp[ed[i].t]>1e17,su+=dp[ed[i].t];
	if(su>1e18||c1>1){dp[u]=2e18;return;}
	l1=r1=id[u];as=1e17;rt[u]=++ct;vl[ct]=1e18;
	ins(rt[u],1,n,id[u],0);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		l2=id[ed[i].t];r2=rb[ed[i].t];
		doit(rt[ed[i].t],-dp[ed[i].t]);rt[u]=merge(rt[u],rt[ed[i].t],1,n);
		r1=r2;
	}
	doit(rt[u],su);
	dp[u]=min(as,0ll)+su;
	if(is[u])
	{
		ll ras=1e17,rt1=++ct;vl[rt1]=1e17;
		for(int i=0;i<3;i++)if(id[s[u][i]]>=id[u]&&id[s[u][i]]<=rb[u])ras=min(ras,query(rt[u],1,n,id[s[u][i]])+v[u][i]);
		else ins(rt1,1,n,id[s[u][i]],query(rt[u],1,n,id[u])+v[u][i]);
		dp[u]=ras;rt[u]=rt1;
	}
}
ll solve()
{
	for(int i=1;i<=ct;i++)ch[i][0]=ch[i][1]=vl[i]=lz[i]=0;
	for(int i=1;i<=n;i++)rt[i]=head[i]=dp[i]=is[i]=0;
	cnt=ct=ct1=0;
	scanf("%d%d",&n,&m);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	for(int i=1;i<=m;i++)
	{
		scanf("%d",&a);is[a]++;
		for(int j=0;j<3;j++)scanf("%d%d",&s[a][j],&v[a][j]);
	}
	for(int i=1;i<=n;i++)if(is[i]>1)return -1;
	dfs0(1,0);dfs(1,0);
	return dp[1]<=1e16?dp[1]:-1;
}
int main()
{
	vl[0]=7e17/3;
	scanf("%d",&T);
	while(T--)printf("%lld\n",solve());
}
```

##### 3E Kart Race

###### Problem

给一个 $n$ 个点 $m$ 条边的有向平面图。

在这个有向图上，你只能从 $x$ 小的边向 $x$ 大的边走。保证不存在一条垂直于 $x$ 轴的边。保证边只在给定的顶点相交。

对于图中的每个点，保证从 $1$ 出发能到达这个点，从这个点出发能到达 $n$。

选择一个点会有一个收益。你需要选择一些点，使得每条 $1$ 到 $n$ 的路径上最多只有一个选中点，且选中点的点权和最大。在此基础上，你需要满足选中的点的编号序列的字典序最小。

输出最小值以及方案。

$n\leq 10^5,m\leq 2n,\sum n\leq 1.5\times 10^6$

$6s,512MB$

###### Sol

问题可以看成一个平面图的DAG上的最长反链。~~然后可以猜测这东西相当于某种意义下的对偶图最短路~~

考虑将每个区域看作一个点。从 $1$ 向左从 $n$ 向右引两条射线，将外部区域分成上部和下部。

对于一个点 $(x,y)$，称这个点上部所在的区域为点 $(x,y+\epsilon)$ 所在的区域，称这个点下部所在的区域为点 $(x,y-\epsilon)$ 所在的区域。

对于平面图上的一条边。称这条边上部的区域为边界包含这条边且在这条边上方的区域，下部的区域同理。

考虑建图，对于每条边（包括两条射线），从这条边下部的区域向这条边上部的区域连一条边。则有如下结论：

两个点 $a,b$ 能同时在最短反链中（两个点互不可达）当且仅当能从一个点的上部区域开始走到另外一个点的下部区域。



###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
#define N 300500
#define M 6006000
#define ll long long
int T,n,m,a,b,s1[N][2],s2[N][2],v1,v[N],ct,s[N][2],rs[N][2];
struct pt{int x,y;}p[N];
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
ll cross(pt a,pt b){return 1ll*a.x*b.y-1ll*a.y*b.x;}
bool cmp(pair<int,int> p1,pair<int,int> p2)
{
	int a=p1.first,b=p2.first;
	pt s1=p[a]-p[v1],s2=p[b]-p[v1];
	int f1=s1.y>0||(s1.y==0&&s1.x>0),f2=(s2.y>0)||(s2.y==0&&s2.x>0);
	if(f1!=f2)return f1;
	return cross(s1,s2)>0;
}
vector<pair<int,int> > nt[N];
vector<int> f1;
int head[N],cnt,ds[N],fr[N][2],c3,as[N],in[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){in[t]++;ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;}
int rt[N],ch[M][2],c1,c2,fv[N];
ll vl[N],hv[M];
bool check1(int x,int y,int l,int r)
{
	if(hv[x]==hv[y])return 0;
	if(l==r)return (x>0)>(y>0);
	int mid=(l+r)>>1;
	if(hv[ch[x][0]]!=hv[ch[y][0]])return check1(ch[x][0],ch[y][0],l,mid);
	else return check1(ch[x][1],ch[y][1],mid+1,r);
}
bool cmp1(int a,int b)
{
	if(vl[a]!=vl[b])return vl[a]>vl[b];
	return check1(rt[a],rt[b],1,n);
}
int ins(int x,int l,int r,int v)
{
	int st=++c1;
	if(l==r){hv[st]=1;return st;}
	ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];
	int mid=(l+r)>>1;
	if(mid>=v)ch[st][0]=ins(ch[x][0],l,mid,v);
	else ch[st][1]=ins(ch[x][1],mid+1,r,v);
	hv[st]=(hv[ch[st][0]]*fv[l]+hv[ch[st][1]]*fv[r])%233333333339ll;
	return st;
}
int modify(int x,int t)
{
	rt[++c2]=ins(rt[x],1,n,t);
	vl[c2]=vl[x]+v[t];
	return c2;
}
void solve()
{
	scanf("%d%d",&n,&m);ct=0;
	fv[0]=1;for(int i=1;i<=n;i++)fv[i]=1ll*fv[i-1]*233333%10000009;
	for(int i=1;i<=n;i++)s2[i][0]=s2[i][1]=0,nt[i].clear();
	for(int i=1;i<=m;i++)s1[i][0]=s1[i][1]=0;
	for(int i=1;i<=n;i++)scanf("%d%d%d",&p[i].x,&p[i].y,&v[i]);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&a,&b);s[i][0]=a;s[i][1]=b;
		nt[a].push_back(make_pair(b,i));
		nt[b].push_back(make_pair(a,-i));
	}
	for(int i=1;i<=n;i++)
	{
		v1=i;sort(nt[i].begin(),nt[i].end(),cmp);
		int sz=nt[i].size();
		for(int j=0;j<sz;j++)
		{
			int v1=nt[i][j].second,v2=nt[i][(j+1)%sz].second;
			rs[v1*(v1>0?1:-1)][v1>0?1:0]=v2;
		}
	}
	int fu1=0;
	for(int i=1;i<=m;i++)if(!s1[i][0])
	{
		f1.clear();++ct;
		int v1=i,v2=0,nw=s[i][0];
		while(1)
		{
			s1[v1][v2]=ct;
			int nt=rs[v1][v2];
			nw^=s[v1][0]^s[v1][1];
			v2=nt>0?0:1,v1=nt*(nt>0?1:-1);
			f1.push_back(nw);
			if(nw==s[i][0])break;
		}
		int sz=f1.size();
		ll su=0;
		for(int i=1;i+1<sz;i++)su+=cross(p[f1[i]]-p[f1[0]],p[f1[i+1]]-p[f1[0]]);
		if(su>=0)fu1=ct;
		for(int i=0;i<sz;i++)
		{
			int ls=f1[(i+sz-1)%sz],nw=f1[i],nt=f1[(i+1)%sz];
			int f1=p[ls].x<p[nw].x,f2=p[nt].x<p[nw].x;
			if((f1^f2)==0)continue;
			s2[nw][f1]=ct;
		}
	}
	for(int i=1;i<=ct+2;i++)head[i]=ds[i]=0;cnt=0;
	for(int i=1;i<=n;i++)
	{
		int v1=s2[i][1],v2=s2[i][0];
		if(v1==fu1)v1=ct+1;if(v2==fu1)v2=ct+2;
		if(!v1)v1=ct+1,v2=ct+2;
		adde(v1,v2,i);
	}
	for(int i=1;i<=m;i++)
	{
		int v1=s1[i][0],v2=s1[i][1];
		if(v1==fu1)v1=ct+1;if(v2==fu1)v2=ct+2;
		adde(v1,v2,0);
	}
	for(int i=1;i<=c1;i++)ch[i][0]=ch[i][1]=hv[i]=0;
	for(int i=1;i<=c2;i++)rt[i]=vl[i]=0;
	c1=c2=1;ds[ct+1]=1;
	queue<int> fu;
	fu.push(ct+1);
	while(!fu.empty())
	{
		int x=fu.front();fu.pop();
		for(int i=head[x];i;i=ed[i].next)
		{
			int d1=ed[i].v?modify(ds[x],ed[i].v):ds[x];
			if(!ds[ed[i].t]||cmp1(d1,ds[ed[i].t]))ds[ed[i].t]=d1,fr[ed[i].t][0]=x,fr[ed[i].t][1]=ed[i].v;
			in[ed[i].t]--;if(!in[ed[i].t])fu.push(ed[i].t);
		}
	}
	printf("%lld\n",vl[ds[ct+2]]);
	c3=0;
	int nw=ct+2;
	while(nw!=ct+1)
	{
		if(fr[nw][1])as[++c3]=fr[nw][1];
		nw=fr[nw][0];
	}
	sort(as+1,as+c3+1);
	for(int i=1;i<=c3;i++)printf("%d%c",as[i],i==c3?'\n':' ');
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}

```

##### 3F New Equipments II

###### Problem

给定 $n$ 个人和 $n$ 个物品，人有权值 $a_i$，物品有权值 $b_i$。

你可以将一个物品分配给一个人，将物品 $j$ 分配给人 $i$ 的收益为 $a_i+b_j$

有 $m$ 对禁止关系 $(x_i,y_i)$，表示不能将物品 $y_i$ 分配给人 $x_i$

对于 $k=1,2,...,n$ ，求出分配正好 $k$ 个物品时的最大收益，无解输出 $-1$

$n\leq 4000,m\leq 10^4,T\leq 10$

$16s,512MB$

###### Sol

问题为最大费用流，考虑模拟每一步增广的过程。

对于每一步，因为收益为 $a_i+b_j$，与中间经过的边无关，因此只需要对于右侧的每一个还没有被增广的点，找到左侧没有被增广的点中，能到达它的点中点权最大的，再记录增广路即可完成增广过程。

这可以看成，考虑左侧之前没有增广的点，按照点权从大到小从这些点依次开始遍历图，记录每个点第一次被访问的时间。

此时从左侧连向右侧的边中，除去 $n$ 条已经使用的增广边和 $m$ 条被删去的边，剩下的每一对 $(i,j)$ 都有一条左侧 $i$ 连向右侧 $j$ 的边。而右侧连向左侧的边只有 $n$ 条增广边的反向边。

考虑bfs遍历，记录当前右侧还没有被访问到的点。如果当前在一个左侧点，则枚举右侧所有没有被访问的点，将能访问到的全部访问。因为总共只有 $n+m$ 条边不能走，因此这部分的总访问次数不超过 $2n+m$。

如果当前在一个右侧点，则它最多连出一条边，直接走即可。

因此单次增广复杂度 $O(n+m)$，总复杂度 $O(n(n+m))$

###### Code

3.588s,65MB

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 4050
#define ll long long
int T,n,m,s[N*3][2],is[N][N],v1[N],v2[N],tp[N],id[N],s1[N],s2[N],ls[N],vis[N],vis1[N],q1[N],q2[N],c1,c2;
bool cmp(int a,int b){return a>b;}
bool cmp1(int a,int b){return v1[a]>v1[b];}
bool cmp2(int a,int b){return v2[a]>v2[b];}
int doit()
{
	queue<int> qu;
	for(int i=1;i<=n;i++)ls[i]=vis[i]=vis1[i]=0;
	c1=c2=0;
	for(int i=1;i<=n;i++)q1[++c1]=i;
	for(int i=1;i<=n;i++)if(!s1[i]&&!vis1[i])
	{
		qu.push(i);vis1[i]=1;
		while(!qu.empty())
		{
			int s=qu.front();qu.pop();
			for(int j=1;j<=c1;j++)
			{
				int t=q1[j];
				if(is[s][t]||s1[s]==t)q2[++c2]=t;
				else
				{
					vis[t]=i;ls[t]=s; 
					if(s2[t]&&!vis1[s2[t]])vis1[s2[t]]=1,qu.push(s2[t]);
				}
			}
			c1=c2;c2=0;for(int j=1;j<=c1;j++)q1[j]=q2[j];
		}
	}
	int fr=0,as=-1;
	for(int i=1;i<=n;i++)if(vis[i]&&!s2[i]&&v2[i]+v1[vis[i]]>as)as=v1[vis[i]]+v2[i],fr=i;
	if(as==-1)return as;
	int tp=fr;
	while(tp)
	{
		s2[tp]=ls[tp];
		int nt=s1[s2[tp]];
		s1[s2[tp]]=tp;
		tp=nt;
	}
	return as;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&m);
		for(int i=1;i<=n;i++)scanf("%d",&v1[i]),tp[i]=i;
		for(int i=1;i<=n;i++)scanf("%d",&v2[i]);
		for(int i=1;i<=m;i++)scanf("%d%d",&s[i][0],&s[i][1]);
		sort(tp+1,tp+n+1,cmp1);sort(v1+1,v1+n+1,cmp);
		for(int i=1;i<=n;i++)id[tp[i]]=i;
		for(int i=1;i<=m;i++)s[i][0]=id[s[i][0]];
		sort(tp+1,tp+n+1,cmp2);sort(v2+1,v2+n+1,cmp);
		for(int i=1;i<=n;i++)id[tp[i]]=i;
		for(int i=1;i<=m;i++)s[i][1]=id[s[i][1]];
		for(int i=1;i<=m;i++)is[s[i][0]][s[i][1]]=1;
		for(int i=1;i<=n;i++)s1[i]=s2[i]=0;
		ll as=0;
		for(int i=1;i<=n;i++)
		{
			int tp=doit();
			if(tp==-1)as=-1;else as+=tp;
			printf("%lld\n",as);
		}
		for(int i=1;i<=m;i++)is[s[i][0]][s[i][1]]=0;
	}
}
```

##### 3L Tree Planting

###### Problem

给一个长度为 $n$ 的序列 $c$ 以及一个 $k$，定义一个长度为 $n$ 的01序列 $v$ 是合法的，当且仅当：

1. $\forall i,v_i,v_{i+1}$ 不同时为 $1$
2. $\forall i,v_i,v_{i+k}$ 不同时为 $1$

对于所有合法的序列 $v$，求和 $\prod_{i=1}^nc_i^{v_i}$，答案模 $998244353$

$4s,512MB$

$n\leq 300,T\leq 10$

###### Sol

如果 $k$ 很小，则可以设 $dp_{i,S}$ 表示考虑前 $i$ 个位置，当前最后 $k$ 个位置选的状态为 $S$ 时，前面的所有方案权值和，可以直接转移。

因为要求相邻两个位置不同时为 $1$，显然状态数不超过 $O((\frac{1+\sqrt 5}2)^k)$ 级别，复杂度为 $O(n*(\frac{1+\sqrt 5}2)^k)$

对于 $k$ 更大的情况，考虑将所有数放在宽度为 $k$ 的二维网格上，其中第 $i$ 行第 $j$ 列的位置上为第 $i*k+j$ 个数($0$ 下标)。

此时第二个限制相当于一个位置和它正下方相邻的位置不能同时为 $1$，第一个限制相当于对于除去最后一列的位置，每个位置和它正右方相邻位置不能同时为 $1$。最后一列的位置和下一行第一个位置不能同时为 $1$。

考虑按照列填数，枚举第一列填的数，然后按照列做做轮廓线dp，设 $dp_{i,j,S}$ 表示填到第 $i$ 列第 $j$ 个数。之前的 $\lceil\frac nk\rceil$ 个数状态为 $S$ 的方案数。

考虑这样的状态数，除去轮廓线上这一列和上一列的分界位置外，其余位置满足相邻两个位置不全为 $1$。

此时如果在分界位置加入一个 $0$，则限制变为相邻两个位置不全为 $1$，因此总状态数不超过 $O((\frac{1+\sqrt 5}2)^{\frac nk+1})$

因此这种方式的复杂度为 $O(n*(\frac{1+\sqrt 5}2)^{\frac {2n}k})$，可以发现上一个做法相当于对行方向做轮廓线 $dp$。

通过平衡两部分复杂度，总复杂度为 $O(n*(\frac{1+\sqrt 5}2)^{\sqrt{2n}})$

###### Code

2.324s,28.9MB

```cpp
#include<cstdio>
using namespace std;
#define N 605
#define M 75100
#define mod 1000000007
int T,n,k,v[N];
int dp[N][M],id[1<<24],st[M],ct;
int solve1()
{
	ct=1;
	for(int i=0;i<1<<k;i++)id[i]=0;
	dp[0][1]=1;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=ct;j++)if(dp[i-1][j])
	{
		int vl=st[j],nt=vl>>1;
		if(!id[nt])id[nt]=++ct,st[ct]=nt;
		dp[i][id[nt]]=(dp[i][id[nt]]+dp[i-1][j])%mod;
		if(!(vl&1)&&!(vl>>k-1))
		{
			nt=vl>>1|(1<<k-1);
			if(!id[nt])id[nt]=++ct,st[ct]=nt;
			dp[i][id[nt]]=(dp[i][id[nt]]+1ll*v[i]*dp[i-1][j])%mod;
		}
	}
	int as=mod-1;
	for(int i=1;i<=ct;i++)as=(as+dp[n][i])%mod;
	for(int i=0;i<=n;i++)for(int j=1;j<=ct;j++)dp[i][j]=0;
	return as;
}
int solve2()
{
	for(int i=n+1;i<=n*2;i++)v[i]=0;
	int le=k,he=(n-1)/k+1,as=mod-1;
	ct=0;
	for(int i=0;i<1<<he;i++)id[i]=0;
	for(int i=0;i<1<<he;i++)
	{
		int fg=1;
		for(int j=0;j<he;j++)if(((i>>j)&3)==3)fg=0;
		if(!fg)continue;
		if(!id[i])id[i]=++ct,st[ct]=i;
		dp[he][id[i]]=1;
		for(int j=2;j<=le;j++)
		for(int k=1;k<=he;k++)
		{
			int v1=v[(k-1)*le+j],nw=(j-1)*he+k;
			for(int p=1;p<=ct;p++)if(dp[nw-1][p])
			{
				int vl=st[p],nt=vl>>1;
				if(!id[nt])id[nt]=++ct,st[ct]=nt;
				dp[nw][id[nt]]=(dp[nw][id[nt]]+dp[nw-1][p])%mod;
				if(!(vl&1)&&(!(vl>>he-1)||k==1))
				{
					nt=vl>>1|(1<<he-1);
					if(!id[nt])id[nt]=++ct,st[ct]=nt;
					dp[nw][id[nt]]=(dp[nw][id[nt]]+1ll*v1*dp[nw-1][p])%mod;
				}
			}
		}
		int vl=0;
		for(int j=1;j<=ct;j++)if(((st[j]<<1)&i)==0)vl=(vl+dp[he*le][j])%mod;
		for(int j=0;j<he;j++)if((i>>j)&1)vl=1ll*vl*v[j*le+1]%mod;
		as=(as+vl)%mod;
		for(int j=1;j<=le*he;j++)
		for(int k=1;k<=ct;k++)dp[j][k]=0;
	}
	return as;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&k);
		for(int i=1;i<=n;i++)scanf("%d",&v[i]);
		if(k<n/k*2)printf("%d\n",solve1());
		else printf("%d\n",solve2());
	}
}
```

##### 4C Cycle Binary

###### Problem

对于一个长度为 $n$ 的 $01$ 串 $s$，定义 $v(s)$ 为最大的 $k$ 满足存在01串 $p,p'$ 满足：

1. $k*p+p'=s$
2. $p'$ 是 $p$ 的一个前缀。

给定 $n$ ，对于所有 $2^n$ 个 $01$ 串求和 $v(s)$，模 $998244353$

$T\leq 100,n\leq 10^9,\sum n\leq 10^{10}$

$8s,512MB$

###### Sol

记串 $s$ 的循环节长度为 $l(s)$，可以发现 $v(s)=\lfloor\frac n{l(s)}\rfloor$。

记循环节长度为 $i$ 的串数量为 $f_i$，考虑计算这个值。

可以发现，在只考虑不超过 $\frac n2$ 的循环节时，如果一个串有两个循环节 $x,y$，则它一定有一个循环节 $\gcd(x,y)$。因此如果一个串最小循环节不超过 $\frac n2$ ，则它所有的不超过 $\frac n2$ 的循环节都为最小循环节的倍数。

设 $g_i=\sum_{j|i}f_j$，显然 $g_i$ 的意义为满足 $i$ 是它的一个循环节，因此 $g_i=2^i$

此时通过反演可以得到 $f_i=\sum_{j|i}\mu(\frac ij)2^j$。考虑计算所有最小循环节不超过 $\frac n2$ 的串的答案，即为 $\sum_{i=1}^{\frac n2}f_i*\lfloor\frac ni\rfloor$

再考虑剩下的串，可以发现剩下的串答案均为 $1$。

因此答案等于：
$$
2^n+\sum_{i=1}^{\frac n2}f_i*(\lfloor\frac ni\rfloor-1)\\
=2^n+\sum_{i=1}^nf_i*(\lfloor\frac ni\rfloor-1)
$$
此时直接两重数论分块可以得到 $O(n^{\frac 34})$ 的做法，但这个做法看起来正好被时限针对了。

此时可以得到：
$$
=2^n+\sum_{i=1}^nf_i\sum_{j\leq n,i|j}1-\sum_{i=1}^nf_i\\
=2^n+\sum_{j=1}^n\sum_{i|j}f_i-\sum_{i=1}^nf_i
$$
设 $g(x)=2^x$，则 $f=\mu*g$，因此 $1*\mu*g=g$，因此答案等于：
$$
2^n+\sum_{i=1}^n2^i-\sum_{i=1}^nf_i\\
=2^{n+1}-1-\sum_{i=1}^nf_i
$$
考虑对右侧数论分块，然后相当于求 $\mu$ 的前缀和，杜教筛即可。

复杂度 $O(n^{\frac 23})$

###### Code

748ms,61.8MB

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 5005000
#define mod 998244353
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int pr[N],ch[N],mu[N],su1[N],ct,T,n,lim=5e6,vl[N],t1,as1[N];
void init(int n)
{
	mu[1]=1;
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++ct]=i,mu[i]=-1;
		for(int j=1;j<=ct&&i*pr[j]<=n;j++)
		{
			ch[i*pr[j]]=1;mu[i*pr[j]]=-mu[i];
			if(i%pr[j]==0){mu[i*pr[j]]=0;break;}
		}
	}
	for(int i=1;i<=n;i++)su1[i]=(su1[i-1]+mu[i]+mod)%mod;
}
int getid(int x){return x<=t1?x:n/x+t1;}
int getf(int x)
{
	if(x<=lim)return su1[x];
	int id=getid(x);
	if(as1[id])return as1[id];
	int as=1;
	for(int l=2,r;l<=x;l=r+1)
	{
		int v=x/l;
		r=x/v;
		as=(as+mod-1ll*getf(v)*(r-l+1)%mod)%mod;
	}
	return as1[id]=as;
}
int calc(int v)
{
	int tp=getid(v-1);
	if(!vl[tp])vl[tp]=pw(2,v);
	return vl[tp];
}
int solve(int n)
{
	int as=0,ct=0;
	t1=sqrt(n);for(int i=1;i<=2*t1;i++)vl[i]=as1[i]=0;
	for(int l=1,r;l<=n;l=r+1)
	{
		int m=n/l;
		r=n/m;
		int v1=(calc(r+1)-calc(l)+mod)%mod;
		as=(as+1ll*v1*(mod-getf(m)))%mod;
	}
	as=(as+pw(2,n+1)-1)%mod;
	return as;
}
int main()
{
	init(lim);
	scanf("%d",&T);
	while(T--)scanf("%d",&n),printf("%d\n",solve(n));
}
```

##### 4F Directed Minimum Spanning Tree

###### Problem

给一个 $n$ 个点 $m$ 条边的带权有向图，对于每个点 $u$ ，求出以 $u$ 为根的最小外向生成树边权和或输出无解。

$n\leq 10^5,m\leq 2\times 10^5,\sum n\leq 5\times 10^5,\sum m\leq 10^6$

$6s,256MB$

###### Sol

[真·模板题][http://oi-wiki.com/graph/dmst/]

以下证明是我编的，看起来就有~~非常多错误~~。

首先考虑求DMST的朱刘算法：

对于每个点 $u$ ，找到连向它的边中边权最小的一条，考虑这样选出的 $n$ 条边，这些边构成一个基环树森林。

考虑这里面的一个环 $(a_1,a_2,...,a_k)$ 以及一个点 $a_1$。对于以 $a_1$ 为根的DMST，将DMST中 $a_2,...,a_k$ 的入边删去，将这部分换成 $a_1\to a_2\to a_3\to...\to a_k$。可以发现这样一定不会变差。因此对于环上的一个点，以这个点为根的DMST一定包含这个环上从这个点前断开形成的链。

对于以其它点为根的DMST，考虑从根开始bfs，找到第一个在环上的点。

此时将已经bfs到的部分缩成一个点，则变成了上面的情况，因此上面的结论仍然成立。

因此可以发现，对于一个这样的 $k$ 个点的环，在任意的DMST中，一定会选择这个环上的 $k-1$ 条边。

考虑将这个环缩成一个点，在缩点后，需要给新的点选择一个入边。设这个入边连向了环上的 $a_i$，则在环上会选择 $a_i\to a_{i+1}\to...\to a_k\to a_1\to...\to a_{i-1}$。

此时对于一条连向这个环的边，在缩点后将它的边权加上在环上选的边的边权，然后即转换为在缩点后的图上的问题。

考虑求出缩点后每个点作为根的答案，没有被缩点的点答案显然就是缩点后的答案，对于这次缩的环上的点，显然答案为以缩点后点为根的答案加上环上选的边数。因此可以还原答案。

为了避免图不强连通导致的细节问题，可以加入 $n$ 条边 $(i\to i\bmod n+1,+\infty)$。

直接每次找环复杂度为 $O(nm)$。考虑一个更加优秀的实现：

从一个点开始，每次找连向这个点的边权最小的点，一直这样走直到走到一个之前出现过的点形成环。

在找到一个环后，将环缩起来，然后从环开始继续走。此时在环之前走过的路径上的点的入边边权不变，因此之前走过的部分可以继续使用。这样每次找到环缩起来直到只剩一个点，走的总步数为 $O(n)$。

此时只需要支持如下操作：

1. 合并两个点
2. 将一个点的所有入边边权加上某个值
3. 询问一个点入边中边权最小的一条（不考虑自环）

对于每个点，维护这个点入边的集合。缩点相当于合并集合。

在找边权最小的边时，可以直接在集合中找，如果找到缩点后变为自环的边，则删去这条边继续找。

使用线段树合并实现复杂度为 $O((n+m)\log n)$，使用可并堆实现复杂度为 $O(m+n\log n)$

###### Code

注意细节.jpg

2.745s,135.6MB

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define M 5505000
#define ll long long
int T,n,m,s1,ct,a,b,c,fa[N],rt[N],st[N],c1,c2,ins[N];
ll ch[M][2],vl[M],mn[M],tp[N][2],v1[N],as[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void doit(int x,ll v){if(x)vl[x]=min(vl[x]+v,(ll)1e18),mn[x]=min(mn[x]+v,(ll)1e18);}
void pushdown(int x){if(vl[x])doit(ch[x][0],vl[x]),doit(ch[x][1],vl[x]),vl[x]=0;}
void pushup(int x){mn[x]=min(mn[ch[x][0]],mn[ch[x][1]]);}
void insert(int x,int l,int r,int s,ll v)
{
	if(l==r){vl[x]=min(vl[x],v);mn[x]=vl[x];return;}
	if(vl[x])vl[x]=0;
	int mid=(l+r)>>1,fg=mid<s;
	if(fg)l=mid+1;else r=mid;
	if(!ch[x][fg])ch[x][fg]=++ct,vl[ct]=mn[ct]=1e18;
	insert(ch[x][fg],l,r,s,v);pushup(x);
}
int merge(int x,int y,int l,int r)
{
	if(!x||!y)return x+y;
	if(l==r){vl[x]=mn[x]=min(vl[x],vl[y]);return x;}
	pushdown(x);pushdown(y);
	int mid=(l+r)>>1;
	ch[x][0]=merge(ch[x][0],ch[y][0],l,mid);
	ch[x][1]=merge(ch[x][1],ch[y][1],mid+1,r);
	pushup(x);return x;
}
pair<int,ll> getmin(int x,int l,int r)
{
	if(l==r)return make_pair(l,vl[x]);
	pushdown(x);
	int mid=(l+r)>>1;
	if(mn[ch[x][0]]==mn[x])return getmin(ch[x][0],l,mid);
	else return getmin(ch[x][1],mid+1,r);
}
int del(int x,int l,int r,int s)
{
	if(l==r)return 1;
	pushdown(x);
	int mid=(l+r)>>1,fg=mid<s,tp;
	if(fg)l=mid+1;else r=mid;
	tp=del(ch[x][fg],l,r,s);
	if(tp)ch[x][fg]=0;
	if(ch[x][!fg])tp=0;
	pushup(x);
	return tp;
}
int rd(){int as=0;char c=getchar();while(c<'0'||c>'9')c=getchar();while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();return as;}
void solve()
{
	n=rd();m=rd();s1=n;
	for(int i=1;i<=n*2;i++)fa[i]=i,rt[i]=tp[i][0]=tp[i][1]=ins[i]=as[i]=0;
	for(int i=0;i<=ct;i++)ch[i][0]=ch[i][1]=vl[i]=mn[i]=0;
	mn[0]=1.1e18;ct=0;
	for(int i=1;i<=n;i++)rt[i]=++ct,mn[ct]=1e18;
	for(int i=1;i<=m;i++)a=rd(),b=rd(),c=rd(),insert(rt[b],1,n,a,c);
	for(int i=1;i<=n;i++)insert(rt[i],1,n,i==1?n:i-1,1e17);
	st[c1=1]=1;ins[1]=1;c2=1;
	while(c1>1||c2<n)
	{
		pair<int,ll> sr=getmin(rt[st[c1]],1,n);
		v1[c1]=min((ll)3e17,sr.second);
		int nt=finds(sr.first),ls=ins[nt];
		if(nt==finds(st[c1])){del(rt[st[c1]],1,n,sr.first);continue;}
		if(!ls){ins[nt]=++c1;st[c1]=nt;c2++;continue;}
		ll su=0;
		for(int i=ls;i<=c1;i++)su=min((ll)4e17,su+v1[i]);
		s1++;rt[s1]=++ct;
		for(int i=ls;i<=c1;i++)
		{
			ll vl=su-v1[i];
			tp[st[i]][0]=s1;tp[st[i]][1]=vl;
			doit(rt[st[i]],vl);
			rt[s1]=merge(rt[s1],rt[st[i]],1,n);
			fa[st[i]]=s1;
		}
		c1=ls;st[ls]=s1;ins[s1]=c1;
	}
	for(int i=s1-1;i>=1;i--)as[i]=min((ll)1e18,as[tp[i][0]]+tp[i][1]);
	for(int i=1;i<=n;i++)printf("%lld\n",as[i]>1e16?-1:as[i]);
}
int main(){scanf("%d",&T);while (T--)solve();}
```

##### 4J Pony Running

###### Problem

有一个 $n\times m$ 的网格。你在网格上随机游走，对于每个位置 $x$ ，给出当你在位置 $x$ 时，你下一步向上下左右走的概率 $p_{x,0},p_{x,1},p_{x,2},p_{x,3}$

有 $q$ 次操作：

1. 修改一个位置的 $p$，

2. 记 $e_x$ 为从位置 $x$ 出发走出网格的期望时间，求所有位置的 $e_x$ 的和，模 $10^9+7$

$nm,q\leq 400$，权值随机生成

$8s,512MB$

###### Sol

考虑通过翻转网格使得 $n\geq m$，此时将位置 $(x,y)$ 编号为 $x*(m-1)+y-1$，则可以发现一个点可以走到的点和它编号距离不超过 $m$。

此时可以发现这相当于一个Band Matrix，直接消元复杂度即为 $O(nm^3)$，直接每次消元即可。

事实上实现这个消元只需要在第 $i$ 次任意找一个这一行不为 $0$ ，然后记录这一行非零的位置以及这一列上非零的行，只对非零的行和非零的位置消，复杂度正确。

复杂度 $O(q(nm)^2)$

###### Code

702ms,1.9MB

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 405
#define mod 1000000007
int n,m,q,s[N][N],v[N][N][4],a,b,c,st[N],c1,as[N],d[4][2]={-1,0,1,0,0,-1,0,1};
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int getid(int x,int y)
{
	if(n<m)return (y-1)*n+x;
	else return (x-1)*m+y;
}
int solve()
{
	for(int i=1;i<=n*m+1;i++)for(int j=1;j<=n*m;j++)s[i][j]=0;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)
	{
		int tp=getid(i,j);
		s[tp][n*m+1]=mod-1;s[tp][tp]=mod-1;
		for(int c=0;c<4;c++)
		{
			int nx=i+d[c][0],ny=j+d[c][1];
			if(nx<1||nx>n||ny<1||ny>m)continue;
			s[tp][getid(nx,ny)]=v[i][j][c];
		}
	}
	for(int i=1;i<=n*m;i++)
	{
		int fr=i;
		for(int j=i;j<=n*m;j++)if(s[j][i])fr=i;
		for(int j=1;j<=n*m+1;j++)swap(s[i][j],s[fr][j]);
		c1=0;
		for(int j=1;j<=n*m+1;j++)if(s[i][j])st[++c1]=j;
		for(int j=i+1;j<=n*m;j++)if(s[j][i])
		{
			int vl=1ll*s[j][i]*pw(s[i][i],mod-2)%mod*(mod-1)%mod;
			for(int k=1;k<=c1;k++)s[j][st[k]]=(s[j][st[k]]+1ll*vl*s[i][st[k]])%mod;
		}
	}
	for(int i=n*m;i>=1;i--)
	{
		int vl=s[i][n*m+1];
		for(int j=i+1;j<=n*m;j++)vl=(vl+mod-1ll*as[j]*s[i][j]%mod)%mod;
		as[i]=1ll*vl*pw(s[i][i],mod-2)%mod;
	}
	int as1=0;for(int i=1;i<=n*m;i++)as1=(as1+as[i])%mod;
	return as1;
}
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)for(int k=0;k<4;k++)scanf("%d",&v[i][j][k]);
	while(q--)
	{
		scanf("%d",&a);
		if(a==1)
		{
			scanf("%d%d",&b,&c);
			for(int k=0;k<4;k++)scanf("%d",&v[b][c][k]);
		}
		else printf("%d\n",solve());
	}
}

```

##### 4K Travel on Tree

###### Problem

给一棵 $n$ 个点的树，$q$ 次询问：

每次询问给定 $l,r$，你需要在树上选一个点出发，以任意顺序将编号 $[l,r]$ 中的所有点全部访问一次，然后回到你的起始点。求经过边数的最小值。

$n,m\leq 10^5,\sum n,\sum m\leq 10^6$

$30s,512MB$

###### Sol

显然按照dfs序访问是最优的，答案可以看成按照dfs序排序后相邻两点距离和，也可以看成2*(包含所有点的最小连通块大小)-2。这里使用后者。

考虑以 $1$ 为根，求出每个点到根的路径的并。可以发现这个并和包含区间内所有点的最小连通块只差根到所有点LCA的路径。后半部分可以直接求出，只需要求出前半部分。

考虑从小到大枚举 $r$，记 $lb_{x,r}$ 表示在 $[1,r]$ 中，满足到根的路径包含 $x$ 的点中编号最大的点，则询问 $[l,r]$ 的并包含 $x$ 当且仅当 $lb_{x,r}\geq l$，只需维护 $lb$ 即可得到答案。

考虑 $lb_{x,r-1}$ 到 $lb_{x,r}$ 的变化，可以发现如果 $x$ 在 $r$ 到 $1$ 的根的路径上，则 $lb_{x,r}=r$，否则 $lb_{x,r}=lb_{x,r-1}$。因此 $r$ 变大时 $lb$ 的变化相当于链上赋值。

因此问题变为到根的一条链上赋值，询问点权大于某个值的点的点数。

考虑LCT，每次修改直接access，可以发现任何时刻一条实链上的点权值相同，且切换链的次数为 $O(n\log n)$。因此相当于 $O(n\log n)$ 次操作，每次将 $a_i$ 个 $b_i$ 改为 $c_i$，因为询问是权值，直接使用BIT维护出现次数即可。

复杂度 $O(n\log^2 n)$

###### Code

3.556s,30MB

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 100500
#define ll long long
int T,n,q,a,b,c,head[N],cnt,dep[N],f[N],id[N],ct,f1[N][18],dep1[N];
ll as[N],t[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void dfs(int u,int fa){dep1[u]=dep1[fa]+1;f1[u][0]=fa;for(int i=1;i<=17;i++)f1[u][i]=f1[f1[u][i-1]][i-1];id[u]=++ct;f[u]=fa;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dep[ed[i].t]=dep[u]+ed[i].v,dfs(ed[i].t,u);}
int cmin(int x,int y){return id[x]<id[y]?x:y;}
int cmax(int x,int y){return id[x]>id[y]?x:y;}
int LCA(int x,int y){if(dep1[x]<dep1[y])x^=y^=x^=y;for(int i=17;i>=0;i--)if(dep1[x]-dep1[y]>=1<<i)x=f1[x][i];if(x==y)return x;for(int i=17;i>=0;i--)if(f1[x][i]!=f1[y][i])x=f1[x][i],y=f1[y][i];return f1[x][0];}
void add(int x,int v){for(int i=x;i;i-=i&-i)t[i]+=v;}
ll que(int x){ll as=0;for(int i=x;i<=n;i+=i&-i)as+=t[i];return as;}
struct segt{
	struct node{int l,r,ls,rs;}e[N*4];
	void pushup(int x){e[x].ls=cmin(e[x<<1].ls,e[x<<1|1].ls),e[x].rs=cmax(e[x<<1].rs,e[x<<1|1].rs);}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;e[x].ls=e[x].rs=l;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
	int queryls(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].ls;int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return queryls(x<<1,l,r);else if(mid<l)return queryls(x<<1|1,l,r);else return cmin(queryls(x<<1,l,mid),queryls(x<<1|1,mid+1,r));}
	int queryrs(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].rs;int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return queryrs(x<<1,l,r);else if(mid<l)return queryrs(x<<1|1,l,r);else return cmax(queryrs(x<<1,l,mid),queryrs(x<<1|1,mid+1,r));}
}tr;
struct LCT{
	int ch[N][2],fa[N],v1[N],v2[N],mn[N],mx[N],fg[N];
	bool nroot(int x){return ch[fa[x]][0]==x||ch[fa[x]][1]==x;}
	void pushup(int x){mn[x]=min(min(mn[ch[x][0]],mn[ch[x][1]]),v2[x]);mx[x]=max(max(mx[ch[x][0]],mx[ch[x][1]]),v1[x]);}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;if(nroot(f))ch[g][ch[g][1]==f]=x;fa[x]=g;fa[ch[x][!tp]]=f;ch[f][tp]=ch[x][!tp];ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
	void splay(int x){while(nroot(x)){int f=fa[x],g=fa[f];if(nroot(f))rotate((ch[g][1]==f)^(ch[f][1]==x)?x:f);rotate(x);}}
	int doit(int x){while(ch[x][0])x=ch[x][0];return x;}
	void access(int x,int f)
	{
		int tp=0;
		while(x)
		{
			splay(x);
			int st=doit(x),t1=ch[x][1];splay(st);
			int las=fg[st];fg[st]=0;
			splay(x);ch[x][1]=0;pushup(x);
			if(las)add(las,mn[x]-mx[x]);
			if(t1){int st2=doit(t1);splay(st2);fg[st2]=las;}
			ch[x][1]=tp;tp=x;pushup(x);x=fa[x];
		}
		int st=doit(tp);splay(st);fg[st]=f;
		add(f,mx[st]);splay(tp);
	}
	void init()
	{
		for(int i=1;i<=n;i++)fa[i]=f[i],v1[i]=mx[i]=dep[i],v2[i]=mn[i]=dep[fa[i]],fg[i]=0,ch[i][0]=ch[i][1]=0;
		mx[0]=0;mn[0]=1.1e9;
	}
}lct;
vector<pair<int,int> > tp[N];
void solve()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)head[i]=0,t[i]=0,tp[i].clear();cnt=ct=0;
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
	dfs(1,0);tr.build(1,1,n);
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d",&a,&b);
		as[i]=-2*dep[LCA(tr.queryls(1,a,b),tr.queryrs(1,a,b))];
		tp[b].push_back(make_pair(a,i));
	}
	lct.init();
	for(int i=1;i<=n;i++)
	{
		lct.access(i,i);
		for(int j=0;j<tp[i].size();j++)as[tp[i][j].second]+=2*que(tp[i][j].first);
	}
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
int main(){scanf("%d",&T);while(T--)solve();}
```

##### 5A Miserable Faith

###### Problem

给一棵 $n$ 个点的有根树，每个点有一个点权 $v_i$。

对于一条边，如果它两侧的点权不同，则这条边的边权为 $1$，否则边权为 $0$。

记 $dis(u,v)$ 表示两点间路径的边权和，$d(u,v)$ 表示两点间路径经过的边数。

定义 $f_u=\max_{u\in subtree_v,dis(u,v)=0}d(u,v)$ ，支持如下操作：

1. 将一个点到根的所有点权改成一个值，保证这个值之前没有在树中出现。
2. 给定 $u,v$，询问 $dis(u,v)$
3. 给定 $u$，询问 $\sum_{v\in subtree_u}dis(i,v)$
4. 询问 $\sum_{i=1}^nf_i$

$n,q\leq 10^5,T\leq 10$

$2s,128MB$

###### Sol

可以发现操作相当于LCT的access，实边边权为 $0$，虚边边权为 $1$。

对于操作 $4$，可以发现一条实链的贡献为 $\frac{l(l-1)}2$，直接LCT上维护即可。

对于操作 $2,3$，显然边权只会改变 $O(n\log n)$ 次，操作 $2$ 直接BIT维护每个点到根的 $dis$，操作 $3$ 相当于每一条虚边的子树大小和，都可以直接维护。

复杂度 $O(n\log^2 n)$~~听说能跑得很快~~

###### Code

没写

##### 5B String Mod

###### Problem

给定 $n,k,p$，对于每一对 $0\leq a,b<p$，求有多少个长度为 $n$ ，只包含前 $k$ 种字符的字符串满足：

1. `a` 的出现次数 $\equiv a(\bmod p)$
2. `b` 的出现次数 $\equiv b(\bmod p)$

答案模 $10^9+9$

$2\leq k,n\leq 500,\sum n\leq 2000,n|mod-1$

$5.5s,256MB$

###### Sol

如果看成二元生成函数，那么就是 $(x+y+k-2)^n$

因此可以直接二维循环卷积，即：
$$
v_{i,j}=(\omega_n^i+\omega_n^j+k-2)^n\\
as_{i,j}=\frac 1{n^2}\sum_{x,y\in[0,n)}\omega^{-ix}\omega^{-jy}v_{i,j}\\
=\frac 1{n^2}\sum_{x\in[0,n)}\omega^{-ix}\sum_{y\in[0,n)}\omega^{-jy}v_{i,j}
$$
显然可以做到 $O(n^3)$，注意~~卡亿点常~~选对语言就行。~~同样的代码本地2s g++3.3s c++5.5s还T飞~~

###### Code

3.369s,3.8MB

```cpp
#include<cstdio>
using namespace std;
#define N 505
#define ll long long
#define mod 1000000009
int T,k,n,as[N][N],f[N][N],s2[N][N],g=13,tp[N];
ll m;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%lld%d",&k,&m,&n);
		for(int i=0;i<n;i++)tp[i]=pw(g,(mod-1)/n*i);
		for(int i=0;i<n;i++)for(int j=0;j<n;j++)as[i][j]=0,f[i][j]=tp[(n*n-i*j)%n];
		for(int i=0;i<n;i++)for(int j=0;j<n;j++)s2[i][j]=pw(k-2+tp[i]+tp[j],m%(mod-1))%mod;
		for(int i=0;i<n;i++)for(int j=0;j<n;j++)
		{
			int vl=0;
			for(int k=0;k<n;k++)vl=(vl+1ll*s2[i][k]*f[j][k])%mod;
			for(int k=0;k<n;k++)as[j][k]=(as[j][k]+1ll*vl*f[i][k])%mod;
		}
		int tp=pw(n,mod-3);
		for(int i=0;i<n;i++)for(int j=0;j<n;j++)as[i][j]=1ll*as[i][j]*tp%mod;
		for(int i=0;i<n;i++)for(int j=0;j<n;j++)printf("%d%c",as[i][j],j==n-1?'\n':' ');
	}
}
```

##### 5J Guess Or Not 2

###### Problem

给一个长度为 $n$ 的序列 $x_{1,2,...,n}$，使用如下方式随机生成一个序列 $a$：
$$
t_i=rand(0,1)\\
a'_i=exp((\log x_i+-\log(-\log z))/t)\\
a_i=\frac{a'_i}{\sum a'_i}
$$
给出长度为 $n$ 的序列 $z$，定义 $y_i=\frac{z_i}{\sum z_i}$，求出随机生成的概率分布在 $(y_1,y_2,...,y_n)$ 处的概率密度，模 $998244353$

$\sum n\leq 2\times 10^6,t,x_i,y_i\in[1,998244353)$

$3s,512MB$

###### Sol

做题全靠猜.jpg~~以下部分除了前几行全部没有严谨证明~~

首先考虑一个 $a'_i$ 的概率密度函数，设 $f_a(x)$ 表示 $a\leq x$ 的概率，则考虑 $a’_i$ 的计算过程，展开第一个 $exp$ 可以得到 $a'_i=\sqrt[t]{\frac{x_i}{-\log z}}$：

| $t_i=rand(0,1)$   | $f_t(x)=x(x\in[0,1])$              |
| ----------------- | ---------------------------------- |
| $t=\log t$        | $f_t(x)=e^x(x<0)$                  |
| $t=-t$            | $f_t(x)=1-e^{-x}(x>0)$             |
| $a=\frac{x_i}{t}$ | $f_t(x)=e^{-\frac{x_i}x}(x>0)$     |
| $a=\sqrt[n]a$ |$f_t(x)=e^{-\frac{x_i}{x^t}}(x>0)$|

考虑求 $\sum a'_i$ 的期望，但可以发现对于两个变量 $a,b$，$f_{a+b}(x)=\int_{t=0}^xf_a'(t)f_b(x-t)$，在本题的 $f_t$ 下积分极其困难。

最后生成的数一定满足 $\sum a_i=1$，因此这个概率密度函数在 $n$ 维下不连续。但可以猜想在 $\sum a_i=1$ 的 $n-1$ 维子空间下这东西连续。

设 $h_{a_1,...,a_{n-1}}$ 表示生成的前 $n-1$ 个数为 $a$ 的概率密度，猜想此时 $h$ 是连续且可导的。因此概率密度函数可以看成概率累计函数对每一维偏导后的结果。

希望求的位置相当于存在一个 $s$ 满足 $a_{1,2,...,n}=(sy_1,sy_2,...,sy_n)$ 的位置，可以看成 $n$ 维空间下从原点出发的射线。可以猜想答案为求射线上每个位置的概率偏导后的结果，再对射线上积分得到答案。

按照微积分的方式，考虑按照 $s$ 分成若干个小部分，对于一个部分 $s\in[k\epsilon,(k+1)\epsilon)$，求出生成的 $a'$ 在这个区域内的概率($\forall i,a'_i\in[k\epsilon*y_i,(l+1)\epsilon*y_i)$，对这个值做偏导 $\frac{\partial}{\partial a’_1}\frac{\partial}{\partial a‘_2}...\frac{\partial}{\partial a’_n}$，因为每一维取值独立，因此每一维都不超过一个给定值的概率为每一维分别不超过这个概率的乘积，每一维分别不超过的概率即为 $f_t(x)$，因此偏导后的结果可以看成：
$$
\prod_{i=1}^n((e^{-\frac{x_i}{x^t}})'|_{x=y_i*s})\\
=t^n\prod_{i=1}^n\frac{x_i}{y_i^t}*s^{-n*(t+1)}*e^{-\sum_{i=1}^n\frac{x_i}{y_i^t}*s^{-t}}
$$
但在原来是对 $a_1,...,a_{n-1}$ 求导，现在每个 $a_i'=s*a_i$。因此可以~~对着这个积分能被换元的条件~~大胆猜想再乘一个 $s^{n-1}$ 即可。此时答案相当于：
$$
t^n\prod_{i=1}^n\frac{x_i}{y_i^t}\int_{s=0}^{+\infty}s^{-n*(t+1)}*e^{-\sum_{i=1}^n\frac{x_i}{y_i^t}*s^{-t}}ds
$$
设 $v=\sum_{i=1}^n\frac{x_i}{y_i^t}$，换元 $z=vs^{-t}$，则 $dz=-vts^{-t+1}$，因此原式等于：
$$
t^n\prod_{i=1}^n\frac{x_i}{y_i^t}\int_{z=+\infty}^{0}-v^{-n}*\frac 1t*e^{-z}z^{n-1}dz\\
=t^{n-1}v^{-n}\prod_{i=1}^n\frac{x_i}{y_i^t}\int_{z=0}^{+\infty}e^{-z}z^{n-1}dz
$$
然后考虑右边的积分，~~交给wolframalpha~~分部积分一下就有：
$$
\int_{z=0}^{+\infty}e^{-z}z^{n}dz=(-e^{-z}z^n)|_{z=0}^{+\infty}-(\int_{z=0}^{+\infty}(-e^{-z})*(nz^{n-1})dz)\\
=n*\int_{z=0}^{+\infty}e^{-z}z^{n-1}dz
$$
因此 $\int_{z=0}^{+\infty}e^{-z}z^{n-1}dz=(n-1)!$，答案为：
$$
(n-1)!t^{n-1}(\sum_{i=1}^n\frac{x_i}{y_i^t})^{-n}\prod_{i=1}^n\frac{x_i}{y_i^t}
$$
复杂度 $O(k\log mod)$ 或者 $O(k)$

~~我不会任何看起来比较正确的证明~~

###### Code

936ms,2MB

```cpp
#include<cstdio>
using namespace std;
#define N 1005000
#define mod 998244353
int n,x[N],y[N],k,T;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void solve()
{
	scanf("%d%d",&n,&k);
	int su=0,as=pw(k,n-1),s1=0;
	for(int i=1;i<=n;i++)scanf("%d",&x[i]);
	for(int i=1;i<=n;i++)scanf("%d",&y[i]),su=(su+y[i])%mod;
	su=pw(su,mod-2);for(int i=1;i<=n;i++)y[i]=1ll*y[i]*su%mod;
	for(int i=1;i<=n;i++)as=1ll*as*x[i]%mod*pw(y[i],mod-k-2)%mod;
	for(int i=1;i<=n;i++)s1=(s1+1ll*x[i]*pw(y[i],mod-k-1))%mod;
	as=1ll*as*pw(s1,mod-n-1)%mod;
	for(int i=1;i<n;i++)as=1ll*as*i%mod;
	printf("%d\n",as);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 5K Jsljgame

###### Problem

有 $n$ 堆石头，第 $i$ 堆有 $a_i$ 个。

两个人轮流取石子，每个人每次可以选择一堆从中取正整数个石子，不能取的人输。

有一个特殊的限制：先手一次不能正好取 $x$ 个石子，后手一次不能正好取 $y$ 个石子。

求最优策略下谁获胜。

$T\leq 2000,n\leq 1000,a_i,x,y\leq 10^9$

$1s,256MB$

###### Sol

对于 $x=y$ 的情况，这显然是平等博弈，打表可以发现 $sg_n=x*\frac{n}{2x}+(n\bmod x)$，因此可以直接做。

对于 $x>y$ 的情况，通过打表一堆时的sg可以猜想 $y>1$ 时此时的情况和 $x=+\infty$ 的情况等价。~~证明我完全不会~~

此时如果不考虑限制先手必胜，则先手显然还是必胜。考虑剩下的情况。

如果当前没有一堆石头大于等于 $y$ ，则等价于没有限制的情况，因此算xor即可。

否则，如果有一堆石头大于等于 $y$ 且xor和为 $0$，考虑构造一种操作使得对手如果想操作这堆大于等于 $y$ 的石头就必须动 $y$ 个。设这堆石头个数为 $z$

记 $v=z\oplus (z-y)$，设 $k$ 满足 $2^k\leq v<2^{k+1}$。此时可以发现 $v$ 的第 $k$ 位一定为 $1$。因为xor和为 $0$，此时一定存在另外一堆石头 $c$ 满足 $c$ 的这一位为 $1$。，将这堆石头操作为 $c\oplus v$，可以发现此时一定有 $c\oplus v<c$。

此时对手也只能选择一堆石头 $t$，将其操作为 $t\oplus v$。此时 $t$ 的第 $k$ 位一定为 $1$。如果对手不操作最大的一堆，则接下来你还能找到另外一堆在这一位上为 $1$，且这一位上为 $1$ 的个数一定在减小，因此一定存在一个时刻对手必须操作那一堆，因此你会获胜。

结合之前的猜想，可以得到此时先手必胜当且仅当xor和不为 $0$ 或者有一个数大于等于 $y$。~~然后莫名其妙地它就对了~~

在 $y=1$ 的时候，同样可以发现上面的结论是对的。

对于 $x<y$ 的情况，先手想要获胜当且仅当第一次操作后，剩下的局面xor和为 $0$ 且每一个数都小于 $x$。

枚举操作哪一堆，由于xor和为 $0$，可能的操作只有一种，验证一下即可。

复杂度 $O(n)$

###### Code

296ms,1.2MB

```cpp
#include<cstdio>
using namespace std;
#define N 1050
int T,n,a,b,v[N];
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%d",&n,&a,&b);
		for(int i=1;i<=n;i++)scanf("%d",&v[i]);
		if(a==b)
		{
			int as=0;
			for(int i=1;i<=n;i++)as^=v[i]%a+v[i]/2/a*a;
			printf("%s\n",as?"Jslj":"yygqPenguin");
		}
		else if(a>b)
		{
			int mx=0,su=0;
			for(int i=1;i<=n;i++)mx=mx<v[i]?v[i]:mx,su^=v[i];
			printf("%s\n",mx>=b||su?"Jslj":"yygqPenguin");
		}
		else
		{
			int s1=0,s2=0,su=0;
			for(int i=1;i<=n;i++)
			{
				su^=v[i];
				if(v[i]>s1)s2=s1,s1=v[i];
				else if(s2<v[i])s2=v[i];
			}
			int fg=0;
			for(int i=1;i<=n;i++)if((v[i]==s1&&s2<a)||s1<a)
			{
				int nt=su^v[i];
				if(nt<v[i]&&nt<a&&v[i]-nt!=a)fg=1;
			}
			printf("%s\n",fg?"Jslj":"yygqPenguin");
		}
	}
}
```

##### 6B Might and Magic

###### Problem

两个人 $0,1$ 进行战斗，每个人有属性 $a_i,d_i,p_i,l_i,h_i$。同时有两个参数 $c_p,c_m$。

两个人轮流操作，你先手，在你的回合，你可以选择进行如下操作：

1. 进行物理攻击，使对方的 $h_1$ 减少 $\max(a_0-d_1,1)*c_p$ 。
2. 进行魔法攻击，使对方的 $h_1$ 减少 $p_0*c_m$，这种操作只能使用 $l_0$ 次。

在你对手的回合，他一定选择进行物理攻击，使你的 $h_0$ 减少 $\max(a_1-d_0,1)*c_p$ 。（即可以看成 $p_1=l_1=0$），$h$ 先降到 $0$ 或以下的人输。

现在给出 $a_1,d_1,h_0,n$，你可以任意分配 $a_0,d_0,p_0,l_0$，满足四个值非负且和为 $n$。

求最大的 $h_1$ 使得你能获胜。

$T\leq 10^5,a_1,d_1,h_0,n,c_p,c_m\leq 10^6$

$20s,512MB$

###### Sol

显然对手操作 $\lceil\frac {h_0}{c_p\max(a_1-d_0,1)}\rceil$ 次后你就会输。可以发现这个值可能的取值种类只有 $O(\sqrt v)$ 种，因此可以考虑枚举这个值。

枚举这个值后，可以得到你需要的 $d_0$，此时问题变为给定回合数 $k$ 以及剩余点数 $m$，你需要将 $m$ 分配给 $a,p,l$ 使得你对对方造成的伤害最大。

考虑固定进行魔法攻击的次数 $v$，则此时你造成的伤害为 $v*c_m*p+(k-v)*c_p*\max(a-d_1,1)$。

如果选择 $a<d_1$，则显然会让 $a=0$。否则因为 $a+p$ 为定值，最后的伤害为关于 $a$ 的线性函数，因此一定 $a=0$ 或者 $p=0$。

因此最优解中要么 $a=0$，要么 $p=l=0$。

如果 $p=l=0$，则方案唯一，可以直接计算。如果 $a=0$，则伤害为 $c_m*(a-l)*l+(k-l)*c_p(l\leq k)$

此时相当于找一个二次函数的最值，找到唯一的顶点 $p$ ，判断 $\lfloor p\rfloor,\lceil p\rceil$ 和取值区间端点即可。

复杂度 $O(\sqrt v)$

###### Code

2.09s,1.2MB

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define ll long long
int T,cp,cm,hp,sa,sd,n;
ll solve(int n,int r)
{
	if(n<0)return 0;
	ll as=1ll*cp*r*max(1,n-sd);
	//cm*x*(n-x)+cp*(r-x)
	//=-cmx^2+(cm*n-cp)x+cp*r
	ll val=(1ll*cm*n-cp)/cm/2;
	if(val<1)val=1;if(val>=r)val=r-1;
	for(int v=val;v<=val+1;v++)as=max(as,1ll*cm*v*(n-v)+1ll*cp*(r-v));
	return as;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		ll as=0;
		scanf("%d%d%d%d%d%d",&cp,&cm,&hp,&sa,&sd,&n);
		hp=(hp-1)/cp;
		for(int i=1;(i-1)*(i-1)<=hp;i++)as=max(as,solve(n-max(sa-i,0),hp/i+1));
		for(int i=1;(i-1)*(i-1)<=hp;i++)as=max(as,solve(n-(i==1?0:max(0,sa-hp/(i-1))),i));
		printf("%lld\n",as);
	}
}
```

##### 6C 0 tree

###### Problem

给一棵 $n$ 个点的树，点有非负点权 $a_i$，边有边权 $b_i$。

你可以进行操作，每次操作你可以给出 $u,v,c(c\geq 0)$，设树上从 $u$ 到 $v$ 的路径经过的边依次为 $e_0,e_1,...,e_l$，则进行如下操作：
$$
a_u\leftarrow a_u\oplus c,a_v\leftarrow a_v\oplus c\\
e_i\to e_i+(-1)^i*c
$$
你需要通过不超过 $4n$ 次操作，使得所有的 $a_i,b_i$ 变成 $0$。输出一种方案或者输出无解。

$n\leq 10^4,\sum n\leq 10^5$

$3s,512MB$

###### Sol

操作长度为奇数的链会增大 $\sum e_i$，操作长度为偶数的链不改变 $\sum e_i$。因此考虑使用第二种操作。

将树黑白染色，则第二种操作相当于操作两个同色的点。可以发现，此时如果进行操作 $(a,b,k),(a,b,k),(b,a,2k)$，则所有边边权不变，点 $a,b$ 的点权同时异或 $2k$。

考虑边权和为 $0$ 的情况，此时如果操作的两个点颜色不同，则边权和会增大导致不可能达到目标，因此只能操作同色点。因此可以发现如果一种颜色的点权异或和不为 $0$ 则无解。

对于树上的两条边 $(a,b),(b,c)$，通过操作 $(a,c)$ 即可将一条边的边权减小，让另外一条边的边权增大对应值。因此可以通过 $n-1$ 次操作让所有边权变为 $0$。

如果此时所有点的点权都是偶数，则通过上面的操作，每次将一个点的点权变为 $0$，操作 $3(n-2)$ 次即可将所有点权变为 $0$。

如果此时所有点点权不都是偶数，考虑所有权值 $\bmod 2$，操作变为选择一条路径，改变路径上的所有点和边权。

此时对于一条边，显然这条边的边权与这条边的子树内的所有点权和 $\bmod 2$ 不会改变，这可以说明如果在边权全部变为 $0$ 时存在点的点权为奇数，则一定无解。

再考虑边权和不为 $0$ 的情况，此时每种颜色的点权异或和一定相同，设这个值为 $v_1$，再设边权和为 $-v_2(v_2>0)$。

此时一次操作两个异色点的操作会让 $v_1$ 异或 $c$，让 $v_2$ 减小 $c$。需要找一组操作使得操作结束后 $v_1=v_2=0$。

这相当于将 $v_2$ 分成若干个数，让它们异或和为 $v_1$。显然 $v_2<v_1$ 或者 $v_1,v_2$ 奇偶性不同无解，否则可以使用操作 $v_1,\frac{v_2-v_1}2,\frac{v_2-v_1}2$。

这部分操作不影响之前 $\bmod 2$ 部分的性质，因此这部分可以直接任意选一条边并进行操作。

操作步数 $4n-4$，复杂度 $O(n)$

注意 $n=1$ 的情况

###### Code

218ms,3.7MB

```cpp
#include<cstdio>
using namespace std;
#define N 10050
#define ll long long
int T,n,a,b,ct,dep[N],head[N],cnt;
ll v1[N],v2[N],s[N*4][3];
struct edge{int t,next,id;}ed[N*2];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],id};head[t]=cnt;}
void dfs0(int u,int fa)
{
	dep[u]=dep[fa]+1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);
}
void doit(int x,int y,ll v){if(!v)return;s[++ct][0]=x;s[ct][1]=y;s[ct][2]=v;}
void dfs1(int u,int fa,int vl)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs1(ed[i].t,u,ed[i].id);
		int v=ed[i].id;
		if(v2[v]<0)doit(ed[i].t,fa,-v2[v]),v2[vl]+=v2[v],v1[ed[i].t]^=-v2[v],v1[fa]^=-v2[v],v2[v]=0;
		else if(v2[v]>0)doit(fa,ed[i].t,v2[v]),v2[vl]+=v2[v],v1[ed[i].t]^=v2[v],v1[fa]^=v2[v],v2[v]=0;
	}
}
void solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)head[i]=0;cnt=ct=0;
	for(int i=1;i<=n;i++)scanf("%lld",&v1[i]);
	for(int i=1;i<n;i++)scanf("%d%d%lld",&a,&b,&v2[i]),adde(a,b,i);
	if(n==1)
	{
		if(v1[1]==0)printf("YES\n0\n");
		else printf("NO\n");
		return;
	}
	dfs0(1,0);
	ll s1=0,s2=0,su=0,p1=1,p2=ed[head[1]].t,e1=ed[head[1]].id;
	for(int i=1;i<n;i++)su+=v2[i];
	for(int i=1;i<=n;i++)if(dep[i]&1)s1^=v1[i];else s2^=v1[i];
	if(s1!=s2||su+s1>0||(su+s1)%2){printf("NO\n");return;}
	doit(p1,p2,s1);v1[p1]^=s1;v1[p2]^=s1;v2[e1]+=s1;
	doit(p1,p2,(-su-s1)/2);doit(p1,p2,(-su-s1)/2);v2[e1]-=su+s1;
	int fr=1;
	for(int i=2;i<=n;i++)if(dep[i]>dep[fr])fr=i;
	dfs1(fr,0,0);
	for(int i=1;i<=n;i++)if(v1[i]&1){printf("NO\n");return;}
	for(int i=1;i<=n;i++)if(i!=p1&&i!=p2)
	if(dep[i]&1)doit(i,p1,v1[i]/2),doit(i,p1,v1[i]/2),doit(p1,i,v1[i]);
	else doit(i,p2,v1[i]/2),doit(i,p2,v1[i]/2),doit(p2,i,v1[i]);
	printf("YES\n%d\n",ct);
	for(int i=1;i<=ct;i++)printf("%d %d %lld\n",s[i][0],s[i][1],s[i][2]);
}
int main(){scanf("%d",&T);while(T--)solve();}
```

##### 6J Array

###### Problem

给出 $n$ 以及 $1\leq b_1\leq b_2\leq...\leq b_n\leq n+1$，询问是否存在一个长度为 $n$ 的序列序列 $a$ 满足如下条件：

$\forall l\in[1,n]$，$a$ 中所有出现过的元素都在 $a[l,r]$ 中出现过当且仅当 $r\geq b_l$。

$n\leq 2\times 10^5,\sum n\leq 4.5\times 10^6$

$5s,512MB$

###### Sol

考虑记录 $nt_{i,j}$ 表示从 $i$ 位置开始向后，第一个 $j$ 出现的位置。则显然 $r_i$ 为 $nt_i$ 中的最大值。

考虑 $nt_{i,j}$ 的构造过程，根据子序列自动机的结论有：

1. $nt_{i,j}=nt_{i,j+1}(a_i\neq j)$
2. $nt_{i,j}=i(a_i=j)$

因此从后往前考虑，相当于每次将一个值变为 $i$。

如果元素种数 $k$ 固定，则相当于如下问题：

有 $k$ 个初始为 $n+1$ 的数，第 $i$ 次操作可以将一个数变为 $n-i+1$，且这次操作后要求所有数的最大值为 $r_{n-i+1}$。

考虑贪心做这个过程，在第 $i$ 步时，称数 $x$ 是需要保留的，当且仅当它满足如下条件：

1. $i<x\leq b_{n-i+1}$
2. $x$ 在 $b$ 中出现过

可以发现，在做第 $i$ 步操作时，这些数都在之后的某一次作为max。因为 $[1,n]$ 中的每个数只会出现一次，因此不能改变这些数。对于初始的 $n+1$，可以看成有一个是需要保留的，剩下的不是需要保留的。可以发现，此时将不能保留的里面选一个最大的改变显然最优。

考虑维护两个可重集 $S_1,S_2$，表示当前在 $r$ 中出现过的元素和其它元素(其中多个 $n+1$ 只在 $S_1$ 中放一个)，然后第 $i$ 步做如下操作：

1. 如果当前 $S_1$ 中的最大元素大于 $r_{n-i+1}$，则将这个元素拿出改为 $i$。否则，将 $S_2$ 中的最大元素拿出改为 $i$。
2. 如果 $i$ 在 $b$ 中出现则放入 $S_1$，否则放入 $S_2$。
3. 判断当前是否合法

使用队列维护即可 $O(n)$ 判断。

但此时显然不能枚举 $k$，可以发现无解当且仅当出现两种情况之一：

1. 当前需要保留的数个数大于 $k$。
2. 数的数量过多导致无法让 $\max$ 足够低。

因此可以考虑找到第一个情况中需要的最小 $k$，然后判断这个 $k$。

考虑直接保留的数的最大数量，即 $[i+1,b_{n-i+1}]$ 中出现过的数种数的最大值。最小的 $k$ 即为这个值 $+1$（加上这次操作的数）

可以发现这个值就是最小值，判断这一个 $k$ 即可。

复杂度 $O(n)$

需要注意的是，最后不能剩一个 $n+1$，这相当于这种数没有出现。因此需要特判 $b_1=n+1$ 无解。

###### Code

1.107s,3.8MB

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 200500
int T,n,v[N],is[N],su[N];
bool check(int k)
{
	if(k<1)return 0;
	queue<int> s1,s2;
	for(int i=1;i<=k;i++)if(i==1&&is[n+1])s1.push(n+1);else s2.push(n+1);
	for(int i=n;i>=1;i--)
	{
		if(!s1.empty()&&s1.front()>v[i])s1.pop();else if(s2.empty())return 0;else s2.pop();
		if((!s1.empty()&&s1.front()>v[i])||(!s2.empty()&&s2.front()>v[i]))return 0;
		if(is[i])s1.push(i);else s2.push(i);
	}
	return 1;
}
bool solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	if(v[1]>n)return 0;
	for(int i=1;i<=n;i++)if(v[i]<i)return 0;
	for(int i=1;i<=n+1;i++)is[i]=su[i]=0;
	for(int i=1;i<=n;i++)is[v[i]]++;
	for(int i=1;i<=n+1;i++)su[i]=su[i-1]+(is[i]>0);
	int mx=0;
	for(int i=1;i<=n;i++)mx=max(mx,su[v[i]]-su[i]+1);
	return check(mx);
}
int main(){scanf("%d",&T);while(T--)printf("%s\n",solve()?"YES":"NO");}
```

##### 7A Fall with Fake Problem

###### Problem

给定 $n,k$ 以及一个长度为 $n$ 的字符串 $s$，找到一个字典序最小的字符串 $t$ 满足：

1. $t$ 的长度为 $n$ 且 $t$ 的字典序大于等于 $s$。
2. $t$ 中每种字符出现的次数均为 $k$ 的约数或者 $0$

字符集为小写字符。

$n,k\leq 10^5,\sum n\leq 10^6$

$10s,512MB$

###### Sol

考虑字典序最小的条件，显然的想法是找到最大的 $i$ ，使得存在以 $s[1,i-1]$ 为前缀，在第 $i$ 位更大，后面任意填的合法串。

但是因为下一个位置需要大于这个位置本来的字符，而合法条件和每种字符出现次数相关，因此这个东西不能直接二分求出。

考虑先求出最大的 $i$ ，使得存在一个以 $s[1,i-1]$ 为前缀的合法串。这个东西显然可以二分。

考虑一次check的复杂度，相当于要求每种字符至少出现若干次，求是否存在方案。

可以发现 $k$ 的约数只有 $O(\sqrt k)$ 种，因此可以直接按照每一种字符 $dp$，使用bitset优化复杂度 $O(\frac{n\sqrt k|\sum|}{32})$

设二分出的值为 $x$，如果 $x=n$ 则答案为 $s$。

考虑对于一个 $t$，是否存在以 $s[1,t-1]$ 为前缀，在第 $t$ 位更大，后面任意填的合法串：

如果 $t>x+1$，则这个串包含 $s[1,x+1]$，根据二分包含这部分一定无解，因此整体无解。

如果 $t\leq x$ 且 $\exists i\in[t,x]$ 满足 $s_i>s_t$，则考虑将答案串的第 $t$ 位换成 $s_i$，此时串的前 $t$ 位的字符集合为 $s[1,i]$ 字符集合的子集，因为后者有解，前者一定有解。

考虑上一种情况中最大的 $t$，可以发现从这个位置开始到 $x$ ，这部分的 $s$ 一定是不增的，其中最多存在 $|\sum|$ 段连续字符。考虑对于每一段连续字符分别求答案。

对于一段连续字符 $c$，相当于在段内选择一个尽量长的前缀，然后将结尾变为一个比 $c$ 大的字符，使得存在一个合法串以这个串为前缀。

这可以看成当前前缀中其它字符的出现次数固定，还需要将一种比 $c$ 大的字符出现次数 $+1$，在此基础上求 $c$ 最多出现多少次能够合法。

考虑对其它种类字符做 $dp$，设 $dp_{i,0/1,j}$ 表示考虑前 $i$ 种字符，是否选择了一种出现次数 $+1$ 的字符，这些字符出现次数和是否能为 $j$。

那么转移和之前类似，设 $ct_i$ 表示第 $i$ 种字符在前缀中的出现次数，有：
$$
dp_{i,0,j}=\or_{p|k,ct_i\leq p\leq j}dp_{i-1,0,j-p}\\
dp_{i,1,j}=(\or_{p|k,ct_i\leq p\leq j}dp_{i-1,1,j-p})\or([i>c]\and(\or_{p|k,ct_i+1\leq p\leq j}dp_{i-1,0,j-p}))
$$
$c$ 出现的最大次数即为最大的 $p|k$ 满足 $dp_{|\sum|,1,n-p}=1$，这部分单次复杂度与上一个 $dp$ 相同。

这样即可得到最大的 $t$ 满足存在以 $s[1,t-1]$ 为前缀，在第 $t$ 位更大，后面任意填的合法串。如果 $t=0$ 无解，否则只需要考虑后面怎么填字典序最小。

此时后面没有字典序限制，因此确定每种字符出现次数后显然是从小到大填。因此字典序最小相当于要求 `a` 出现尽量多，在这个基础上 `b` 出现尽量多，以此类推。

因此按照字符从大往小做一遍第一个 $dp$，从前往后按照 $dp$ 选最大的即可。

复杂度 $O(\frac 1{32}n\sqrt k|\sum|(\log n+|\sum|))$

非常喜闻乐见的是因为多测这题需要手写bitset

###### Code

3.541s,2.3MB

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define ui unsigned int
int T,n,k,v[N],ct,su[N];
char s[N];
struct bset{
	ui v[N>>5];
	void init(){for(int i=0;i<=(n>>5)+2;i++)v[i]=0;}
	int getval(int x){return (v[x>>5]>>(x&31))&1;}
	void modify(int x,int s){v[x>>5]|=1u<<(x&31);}
}tp[31],dp[31][2],f1,f2,f3;
void doit(bset &a,bset &b,int k)
{
	int tp=(k&31);
	for(int i=0;i<=((n-k)>>5)+2;i++)
	b.v[i+(k>>5)]|=(a.v[i]<<tp),b.v[i+(k>>5)+1]|=((long long)a.v[i])>>(32-tp);
}
bool check(int l,int sv=0)
{
	for(int i=0;i<=26;i++)su[i]=0;
	su[sv]++;
	for(int i=1;i<=27;i++)tp[i].init();
	tp[27].modify(0,1);
	for(int i=1;i<=l;i++)su[s[i]-'a'+1]++;
	for(int i=26;i>=1;i--)
	for(int j=1;j<=ct;j++)if(v[j]>=su[i])doit(tp[i+1],tp[i],v[j]);
	return tp[1].getval(n);
}
void solve()
{
	scanf("%d%d%s",&n,&k,s+1);
	ct=0;for(int i=0;i<=k&&i<=n;i++)if(!i||k%i==0)v[++ct]=i;
	int lb=0,rb=n,as=-1;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check(mid))as=mid,lb=mid+1;
		else rb=mid-1;
	}
	if(as==n){printf("%s\n",s+1);return;}
	if(as==-1){printf("-1\n");return;}
	int st=as;while(st>1&&s[st-1]>=s[st])st--;
	int as1=st-1;as++;
	while(st<=as)
	{
		int nw=st,t1=s[st]-'a'+1;while(s[nw+1]==s[nw]&&nw<as)nw++;
		for(int i=0;i<=26;i++)su[i]=0;
		for(int i=1;i<st;i++)su[s[i]-'a'+1]++;
		for(int i=1;i<=26;i++)for(int j=0;j<2;j++)dp[i][j].init();
		dp[0][0].modify(0,1);
		for(int i=1;i<=26;i++)
		if(i==t1)doit(dp[i-1][0],dp[i][0],0),doit(dp[i-1][1],dp[i][1],0);
		else
		{
			for(int j=1;j<=ct;j++)if(v[j]>=su[i])doit(dp[i-1][0],dp[i][0],v[j]),doit(dp[i-1][1],dp[i][1],v[j]);
			if(i>t1)for(int j=1;j<=ct;j++)if(v[j]>=su[i]+1)doit(dp[i-1][0],dp[i][1],v[j]);
		}
		int mx=-1;
		for(int j=1;j<=ct;j++)if(dp[26][1].getval(n-v[j]))mx=v[j];
		for(int i=st;i<=nw;i++)if(su[t1]+i-st<=mx)as1=i;
		st=nw+1;
	}
	if(!as1){printf("-1\n");return;}
	int nt=0;
	for(int i=s[as1]-'a'+2;i<=26;i++)if(check(as1-1,i)){nt=i;break;}
	s[as1]=nt+'a'-1;
	check(as1);
	int nw=n,t1=as1;
	for(int i=1;i<=26;i++)
	{
		int as=0;
		for(int j=1;j<=ct&&v[j]<=nw;j++)if(tp[i+1].getval(nw-v[j]))as=v[j];
		for(int j=1;j<=as-su[i];j++)s[++t1]=i+'a'-1;
		nw-=as;
	}
	printf("%s\n",s+1);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 7B Fall with Soldiers

###### Problem

给定奇数 $n$，称一个长度为 $n$ 的 `01` 串 $s$ 是好的，当且仅当它可能通过若干次操作使得串中所有字符均为 `1`。

一次操作为选择一个不在开头或结尾的位置 $i$，满足 $s_i=1$，随后删去 $s_{i-1}$ 和 $s_{i+1}$。

对于一个包含 `01?` 的串，定义它的权值为将每一个 `?` 替换为 `01` 使得得到的 `01` 串是好的的方案数。

给一个长度为 $n$ 的，包含 `01?` 的串。$q$ 次操作，每次修改这个串的一个位置，求出第一次操作前和每次操作后这个串的权值。

$n,q\leq 2\times 10^5,T\leq 11,\sum q\leq 10^6$

$8s,512MB$

###### Sol

如果当前所有字符均为 `1` ，则继续操作可以使得串中只剩下一个 `1`。可以发现合法的条件等价于操作后使得串中只剩一个 $1$。

如果在一次操作后，串中的第 $\frac{m+1}2$ 个位置为 `1` （$m$ 为当前长度），则接下来全部操作这个位置即可。

如果不存在一种方式，使得在某一步操作后，第 $\frac{m+1}2$ 个位置为 `1` ，则在剩余 $3$ 个时操作无法进行，一定不能满足要求。

因此合法当且仅当能通过操作使得当前串的正中位置为 `1` 。贪心地想，操作中一定会选择 $\frac{n+1}2$ 位置左侧的第一个 `1` 或者右侧的第一个 `1`，然后尝试将这个 `1` 移动到中间位置。设 $\frac{n+1}2$ 位置左侧第一个 `1` 的位置为 $l$，右侧第一个 `1` 的位置为 $r$。

如果将 $l$ 位置移动到中间，此时在 $l$ 或者 $l$ 左侧操作一定不优，考虑在右侧操作。如果全部操作 $r$ 位置，则最多可以在右侧操作 $n+1-r$ 次。可以发现此时可以达到目标当且仅当 $l\geq \frac{n+1-2*(n+1-r)}2$ ，整理得 $r-l\leq\frac{n-1}2$。对于将 $r$ 位置移动的情况，可以发现此时也等价于 $r-l\leq \frac{n-1}2$

还可能出现 $l,r$ 中有数不存在的情况。可以发现，在 $n>1$ 时，第一个位置和最后一个位置都不会被操作，因此将这两个位置都看成 `1` 不改变合法性。

$r-l>\frac{n-1}2$ 意味着在串中出现了连续 $\frac{n-1}2$ 个 `0`，但可以发现如果出现连续 $\frac{n-1}2$ 个 `0` 则这段连续的 `0` 一定覆盖位置 $\frac{n+1}2$，使得 $s$ 不合法。因此合法等价于将 $s_1,s_n$ 设为 `1` 后，串中不存在连续 $\frac{n-1}2$ 个 `0`。

此时可以发现，串中不可能存在两段连续的 $0$ 长度大于等于 $\frac{n-1}2$ ，因此考虑计算所有方案中 $s$ 中长度大于等于 $\frac{n-1}2$ 的极长 `0` 段数量，即为不合法的方案数。

使用类似点减边的方式，可以得到上一个值等于所有方案中 $s$ 中长度等于 $\frac{n-1}2$ 的 `0` 段数量减去所有方案中 $s$ 中长度等于 $\frac{n+1}2$ 的 `0` 段数量。考虑分别计算两个值。

此时可以求和每一个长度为 $\frac{n-1}2$ 的段是全 `0` 的方案数。设这一段内有 $a$ 个 `1`，$b$ 个 `?`，整个串中有 $su$ 个 `?`，则方案数为 $[a=0]2^{su-b}=2^{su}[a=0]2^{-b}$。

考虑使用线段树维护，包含一个位置的段一定是连续的一些，因此每次修改一定是区间修改，可以看成如下形式：

1. 区间 $a,b$ 加一或减一，保证任意时刻 $a,b$ 非负。
2. 求整体的 $\sum [a_i=0]2^{-b_i}$

那么在一个点上维护区间内的 $\min a_i$ 以及 $\sum[a_i=\min a_i]2^{-b_i}$，修改 $b$ 直接看成区间乘即可。

因为 $s_1,s_n$ 被钦定为了 `1`，所以可以看成将 $a_1,a_{\frac{n+3}2}$ 在最开始 $+1$。求 $\frac{n+1}2$ 部分同理。

上面的分析要求了 $n>1$，因此需要特判 $n=1$。

复杂度 $O(n+q\log n)$

###### Code

2.776s,12.3MB

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define mod 1000000007
int T,n,q,t[N][2],as[N],as1[N],s1[N],pw[N];
char s[N],st[10];
struct segt{
	struct node{int l,r,su,mn,lz1,lz2;}e[N*2];
	void pushup(int x)
	{
		e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);
		e[x].su=((e[x<<1].mn==e[x].mn)*e[x<<1].su+(e[x<<1|1].mn==e[x].mn)*e[x<<1|1].su)%mod;
	}
	void doit1(int x,int v){e[x].lz1=1ll*e[x].lz1*v%mod;e[x].su=1ll*e[x].su*v%mod;}
	void doit2(int x,int v){e[x].lz2+=v;e[x].mn+=v;}
	void pushdown(int x)
	{
		if(e[x].lz1!=1)doit1(x<<1,e[x].lz1),doit1(x<<1|1,e[x].lz1),e[x].lz1=1;
		if(e[x].lz2)doit2(x<<1,e[x].lz2),doit2(x<<1|1,e[x].lz2),e[x].lz2=0;
	}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].mn=e[x].lz2=0;e[x].lz1=1;e[x].su=r-l+1;
		if(l==r)return;
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	}
	void modify1(int x,int l,int r,int v)
	{
		if(e[x].r<l||e[x].l>r)return;
		if(e[x].l>=l&&e[x].r<=r){doit1(x,v);return;}
		pushdown(x);
		modify1(x<<1,l,r,v);modify1(x<<1|1,l,r,v);
		pushup(x);
	}
	void modify2(int x,int l,int r,int v)
	{
		if(e[x].r<l||e[x].l>r)return;
		if(e[x].l>=l&&e[x].r<=r){doit2(x,v);return;}
		pushdown(x);
		modify2(x<<1,l,r,v);modify2(x<<1|1,l,r,v);
		pushup(x);
	}
	int query(){return e[1].mn?0:e[1].su;}
}tr;
void doit(int k)
{
	tr.build(1,1,n-k+1);
	for(int i=1;i<=n;i++)s1[i]=s[i]=='?'?2:s[i]-'0';
	tr.modify2(1,1,1,1);tr.modify2(1,n-k+1,n-k+1,1);
	for(int i=1;i<=n;i++)if(s1[i]==1)tr.modify2(1,i>=k?i-k+1:1,i>=n-k+1?n:i,1);else if(s1[i]==2)tr.modify1(1,1,n-k+1,2),tr.modify1(1,i>=k?i-k+1:1,i>=n-k+1?n:i,(mod+1)/2);
	as1[0]=tr.query();
	for(int i=1;i<=q;i++)
	{
		int x=t[i][0],y=t[i][1];
		if(s1[x]==1)tr.modify2(1,x>=k?x-k+1:1,x>=n-k+1?n:x,-1);else if(s1[x]==2)tr.modify1(1,1,n-k+1,(mod+1)/2),tr.modify1(1,x>=k?x-k+1:1,x>=n-k+1?n:x,2);
		s1[x]=y;
		if(s1[x]==1)tr.modify2(1,x>=k?x-k+1:1,x>=n-k+1?n:x,1);else if(s1[x]==2)tr.modify1(1,1,n-k+1,2),tr.modify1(1,x>=k?x-k+1:1,x>=n-k+1?n:x,(mod+1)/2);
		as1[i]=tr.query();
	}
}
void solve()
{
	scanf("%d%d%s",&n,&q,s+1);
	pw[0]=1;for(int i=1;i<=n;i++)pw[i]=1ll*pw[i-1]*2%mod;
	int su=0;
	for(int i=1;i<=n;i++)s1[i]=s[i]=='?'?2:s[i]-'0',su+=s1[i]==2;
	as[0]=pw[su];
	for(int i=1;i<=q;i++)
	{
		scanf("%d%s",&t[i][0],st+1);
		t[i][1]=st[1]=='?'?2:st[1]-'0';
		su+=(t[i][1]==2)-(s1[t[i][0]]==2);s1[t[i][0]]=t[i][1];
		as[i]=pw[su];
	}
	if(n==1)
	{
		printf("%d\n",s1[1]!=0);
		for(int i=1;i<=q;i++)printf("%d\n",t[i][1]!=0);
		return;
	}
	doit(n>>1);for(int i=0;i<=q;i++)as[i]=(as[i]+mod-as1[i])%mod;
	doit((n+1)>>1);for(int i=0;i<=q;i++)as[i]=(as[i]+as1[i])%mod;
	for(int i=0;i<=q;i++)printf("%d\n",as[i]);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 8A X-liked Counting

###### Problem

给定 $l,r,k$ ，求有多少个 $[l,r]$ 中的整数 $x$ 满足：

1. 在十进制表示下，$x$ 的每个非空前缀和每个非空后缀都不是 $k$ 的倍数。
2. 在十进制表示下，$x$ 中不存在一位为 $7$。

$l,r\leq 10^{18},k\leq 500,T\leq 10$

$8s,128MB$

###### Sol

先拆成 $[0,r]$ 减去 $[0,l-1]$ 。对于 $[0,r]$ 的一段，它可以被拆成 $\log n*10$ 段，每一段为前若干位固定，后面任意。考虑对后面任意的部分做 $dp$。

对于一个固定的前缀，考虑计算在后 $k$ 位任意的情况下，有多少种后面任意填的方式满足在后面这一段中结束/开始的所有前缀/后缀的十进制表示都不为 $0$。

在只关心这一段内部结束/开始的部分时，可以发现对于前面的前缀只需要记录这段前缀的十进制表示值即可。同时为了判断后缀合法，可以再记录当前填了部分的后缀的值。

因此设 $dp_{d,x,y}$ 表示填 $d$ 位，这 $d$ 位前的前缀值为 $x$，这 $d$ 位内的后缀值为 $y$，满足在这 $d$ 位中结束/开始的所有前缀/后缀的十进制表示都不为 $0$ 的方案数。转移有：
$$
dp_{d,x,y}=[y\neq 0]\sum_{i=0}^9[i\neq 7][(10x+i)\bmod k\neq 0\or (x=0\and i=0)]dp_{d-1,(10x+i)\bmod k,(y-10^{d-1}*i)\bmod k}
$$
其中 $x=0$ 的状态表示前面全是 $0$。

计算答案时可以枚举 $y$，判断前面的前缀和后缀是否合法即可。

复杂度 $O(k^2\log n*10+k\log^2n*10)$

###### Code

4.134s,38.2MB

```cpp
#include<cstdio>
using namespace std;
#define N 505
#define ll long long
int T,k,pw[21];
ll dp[21][N][N],l,r,pw1[N];
ll solve(ll n)
{
	n++;
	ll as=0,as1=0;
	for(int i=0;i<=18;i++)
	{
		int st=n/pw1[i]%10;
		for(int j=0;j<st;j++)if(j!=7)
		{
			ll sv=0,f1=1,s2=0;
			for(int l=18;l>i;l--)if(n/pw1[l]%10==7)f1=0;
			if(!f1)continue;
			ll v1=1;
			for(int p=1;p<=i;p++)v1*=9;
			as1+=v1;
			for(int l=18;l>=i;l--)sv=(sv*10+(l>i?n/pw1[l]%10:j))%k,s2+=(l>i?n/pw1[l]%10:j),f1=f1&&(sv||!s2);
			if(!f1)continue;
			for(int p=0;p<k;p++)
			{
				int st=p,fg=1;
				for(int l=i;l<=18;l++)
				{
					st=(st+(l>i?n/pw1[l]%10:j)*pw[l])%k;
					if(!st)fg=0;
				}
				if(fg)as+=dp[i][sv][p];
			}
		}
	}
	return as1-as;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%lld%lld%d",&l,&r,&k);
		pw[0]=pw1[0]=1;for(int i=1;i<=18;i++)pw[i]=pw[i-1]*10%k,pw1[i]=pw1[i-1]*10;
		for(int i=1;i<=18;i++)for(int j=0;j<=k;j++)for(int l=0;l<=k;l++)dp[i][j][l]=0;
		for(int i=0;i<k;i++)dp[0][i][0]=1;
		for(int i=1;i<=18;i++)
		for(int j=0;j<k;j++)
		for(int l=0;l<k;l++)
		for(int t=0;t<10;t++)if(t!=7)
		{
			int nj=(j*10+t)%k,nl=(l+t*pw[i-1])%k;
			if((nj||(!j&&!t))&&nl)dp[i][j][nl]+=dp[i-1][nj][l];
		}
		printf("%lld\n",solve(r)-solve(l-1));
	}
}
```

##### 8J Yinyang

###### Problem

给一个 $n\times m$ 的网格，定义一种将网格中的每个格子染成黑色或者白色的方案是合法的当且仅当它满足如下条件：

1. 所有黑色格子连通
2. 所有白色格子连通
3. 不存在一个 $2\times 2$ 的连续子矩阵满足四个格子颜色相同。

现在给出一些格子的颜色，求对剩下的格子任意染色，合法的染色方案数，模 $998244353$

$T\leq 10,nm\leq 100$

$2s,512MB$

###### Sol

$\min(n,m)\leq 10$，考虑轮廓线 $dp$。在状态中记录轮廓线上的 $m+1$ 个点的连通性。

可以发现轮廓线上的 $m+1$ 个点构成一条链，因为只有两种颜色，相邻两个点连通表示它们颜色相同，不连通表示它们颜色不同。

因此只需要记录连通性以及最后一个位置的颜色，即可表示出每个位置的颜色。可以发现这样的状态数大约是 $7\times 10^4$，可以考虑直接做。

转移时需要注意如下问题：

1. 如果一个之前的连通块与当前轮廓线不连通，则接下来这种颜色不可能再出现，否则这种颜色不连通。此时剩下部分只能是另外一种颜色，但因为限制3，这种情况会出现当且仅当轮廓线到了最后一个格子处。可以发现此时合法的情况只有当前轮廓线上都是另外一种颜色。在最后一个位置特判这种转移即可。
2. 最后的状态合法当且仅当轮廓线上只有不超过两个连通块
3. 转移可以预处理，注意细节（比如每行第一个位置）和常数问题。

复杂度 $O(nm*|state|)$

###### Code

1.95s,196.5MB

```cpp
#include<cstdio>
#include<vector>
#include<cstring>
using namespace std;
#define N 105
#define M 37001
#define mod 998244353
int T,n,m,f[N][N],sr[40700070],ct,dp[N][2][M],trans[M][2][2][2][2];
vector<int> tp[M];
vector<int> doit(vector<int> x)
{
	int st[17]={0},ct=0;
	for(int i=0;i<x.size();i++)
	{
		if(!st[x[i]])st[x[i]]=++ct;
		x[i]=st[x[i]];
	}
	return x;
}
int calc(vector<int> x)
{
	int n=x.size(),as=0,tp=1;
	for(int i=1;i<n;i++)as+=(x[i]-1)*tp,tp*=(i+1);
	return as;
}
int gettrans(int st,int ls,int nw,int fi,int ed)
{
	if(trans[st][ls][nw][fi][ed])return trans[st][ls][nw][fi][ed];
	vector<int> s1=tp[st];
	if(!fi&&ls==nw&&s1[0]==s1[m]&&s1[0]==s1[m-1])return trans[st][ls][nw][fi][ed]=-1;
	vector<int> s2;s2.push_back(m+2);
	for(int i=0;i<=m;i++)s2.push_back(s1[i]);
	if(ls==nw&&!fi)s2[0]=s2[1];
	int is2=(s2[1]==s2[m+1])^(s2[m]==s2[m+1])^(ls==nw);
	if(fi)
	{
		is2=ls==nw;
		for(int i=1;i<m;i++)is2^=s2[i]!=s2[i+1];
	}
	if(is2)
	{
		int v1=s2[0],v2=s2[m];
		for(int i=0;i<=m+1;i++)if(s2[i]==v2)s2[i]=v1;
	}
	int fg=0;
	for(int i=0;i<=m;i++)if(s2[i]==s2[m+1])fg=1;
	for(int i=1;i<m;i++)if(s2[i]!=s2[0])ed=0;
	if(!fg&&!ed)return trans[st][ls][nw][fi][ed]=-1;
	s2.pop_back();
	s2=doit(s2);
	int vl=calc(s2);
	if(!sr[vl])sr[vl]=++ct,tp[ct]=s2;
	return trans[st][ls][nw][fi][ed]=sr[vl];
}
void solve()
{
	memset(sr,0,sizeof(sr));
	memset(dp,0,sizeof(dp));
	memset(trans,0,sizeof(trans));
	ct=0;
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(n>=m)scanf("%d",&f[i][j]);else scanf("%d",&f[j][i]);
	if(n<m)n^=m^=n^=m;
	for(int i=0;i<1<<m;i++)
	{
		int fg=1,vl=1;
		for(int j=1;j<=m;j++)
		if((f[1][j]==0&&((i>>j-1)&1))||(f[1][j]==1&&((~i>>j-1)&1)))fg=0;
		if(!fg)continue;
		vector<int> s1;
		for(int j=m;j>=1;j--)vl+=(j<m&&((i>>j-1)&3)%3),s1.push_back(vl);
		s1.push_back(vl);
		s1=doit(s1);
		int v1=calc(s1);
		if(!sr[v1])sr[v1]=++ct,tp[ct]=s1;
		dp[m][i>>m-1][sr[v1]]++;
	}
	for(int i=2;i<=n;i++)
	for(int j=1;j<=m;j++)
	for(int k=0;k<2;k++)
	for(int s=1;s<=ct;s++)if(dp[(i-1)*m+j-1][k][s])
	for(int t=0;t<2;t++)if(f[i][j]==-1||f[i][j]==t)
	{
		int nt=gettrans(s,k,t,j==1,i==n&&j==m);
		if(nt==-1)continue;
		dp[(i-1)*m+j][t][nt]=(dp[(i-1)*m+j][t][nt]+dp[(i-1)*m+j-1][k][s])%mod;
	}
	int as=0;
	for(int i=1;i<=ct;i++)
	{
		int fg=1;
		for(int j=0;j<=m;j++)if(tp[i][j]>2)fg=0;
		if(fg)for(int j=0;j<2;j++)as=(as+dp[n*m][j][i])%mod;
	}
	printf("%d\n",as);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}

```

##### 9A NJU emulator

###### Problem

有一个栈，支持如下操作：

1. 向栈顶加入一个 $1$
2. 向栈顶加入一个原栈顶元素
3. 删除栈顶元素
4. 交换栈顶两个元素
5. 将栈顶元素乘上某个栈中元素
6. 将栈顶元素加上某个栈中元素
7. 将栈顶元素减去某个栈中元素
8. 输出栈顶元素并结束

所有操作在 $\bmod 2^{64}$ 下进行。给定 $n$ ，你需要构造不超过 $50$ 条操作让栈输出 $n$ 并结束。

$0\leq n<2^{64},T\leq 10^4$

$1s,256MB$

###### Sol

考虑先向栈中放 $1\sim 8$ 的数，然后放一个数每次 $*8+k$，经过 $22$ 轮即可得到答案。

放入数可以2操作+5操作，这部分需要 $16$ 步，加上后面至少需要 $56$ 步。

考虑再加入减法操作，在不考虑前面退位的情况下，通过把 $+8/+9/+10/...$ 变成 $-8/-7/-6/...$ ，即可每次确定四位。

考虑退位的问题，从后往前考虑每四位，如果当前的四位是 $8\sim 15$，则向前进一位为后面的退位用。

先放入 $1\sim 8$ 再加入 $16$ 需要 $17$ 步，放一个 $1$ 后， $16$ 次 $*16+k$ 需要 $32$ 步，加上最后输出需要 $51$ 步。

但可以发现第一次 $*16$ 不是必要的，在第一次 $+k$ 的时候特判即可。

步数 $50$，复杂度 $O(\log n)$

###### Code

78ms,1.2MB

```cpp
#include<cstdio>
using namespace std;
#define ul unsigned long long
int T,st[233][2],ct;
ul n;
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%llu",&n);
		printf("p1\n");
		for(int i=2;i<=8;i++)printf("p1\nadd 1\n");
		ct=0;
		printf("dup\nmul 7\np1\n");
		for(int t=0;t<16;t++)
		{
			int vl=n&15;
			n>>=4;
			if(t==15)vl=(vl+15)&15;
			if(vl&&vl<8)st[t][0]=1,st[t][1]=9-vl;
			else if(vl)st[t][0]=2,st[t][1]=vl-7,n++;
			else st[t][0]=0;
		}
		for(int t=15;t>=0;t--)
		{
			if(t<15)printf("mul 1\n");
			if(st[t][0]==1)printf("add %d\n",st[t][1]+1);
			else if(st[t][0]==2)printf("sub %d\n",st[t][1]+1);
		}
		printf("end\n");
	}
}
```

##### 9D Into the woods

###### Problem

你在二维平面上随机游走，初始你在 $(0,0)$ ，每一步你等概率随机选择一个方向走 $1$ 单位距离。

求走 $n$ 步后，你在路径中与原点的曼哈顿距离最大值的期望，模给定质数 $p$。

$T\leq 10,n\leq 10^6$

$5s,256MB$

###### Sol

考虑旋转 $45$ 度，问题变为：

每次向 $(-1,-1),(-1,1),(1,-1),(1,1)$ 中随机选一个方向游走，求 $n$ 个时刻内 $\max(|x|,|y|)$ 的最大值期望。

可以发现此时两维上独立且相同，考虑对于一维的情况求出 $f_i$ 表示 $\max\leq i$ 的概率。随后根据期望线性性答案即为 $\sum_{i=0}^n1-f_i^2$

考虑求一个 $f_i$，相当于从 $(0,0)$ 出发，每次向右上或者右下走，要求每个时刻都在 $y\in[-i,i]$ 之间，求方案数。

考虑计算最后时刻 $y\in[-i,i]$ 的方案数，再减去中间超过这个限制的部分

如果只有不超过 $i$ 的限制，考虑在不合法路径第一次超过 $i$ 时沿着 $i+1$ 翻折，可以得到走到 $y\in[i+2,3*i+2]$ 的方案。

考虑从 $(0,0)$ 走到 $(n,y)(y\in[i+2,2*i-1])$ 的方案，这样的方案一定经过 $y=i+1$。在第一次经过的位置翻折回来，即可对应一种不合法路径。

因此这样的路径和不合法路径一一对应，这样的方案数即为 $\sum_{y=i+2}^{2*i-1}[y\equiv n(\bmod 2)]C_n^{\frac{n+y}2}$

考虑有两个方向限制的情况。可以先分别减去跨过了一条对应限制的情况，但此时同时经过两条对应限制的会被减去两次。

考虑再加上先经过一条对应限制，再经过另外一条的部分。设先跨过了 $y=i+1$，然后跨过 $y=-i-1$，考虑计算这样的方案数。

首先在第一次 $y=-i-1$ 处翻折，得到 $y\in[-i-2,-3*i-2]$，然后再沿着 $y=i+1$ 翻折，得到 $y\in[3*(i+1)+1,5*(i+1)-1]$

考虑翻折后对应的一条路径，先沿着第一次经过 $y=i+1$ 翻折回来，此时一定又经过 $y=-i-1$，然后可以再翻折一次。因此这样的方案数等于所有第一次经过 $y=i+1$ 后又经过 $y=-i-1$ 的路径数。

在加上这部分后，可以发现一条经过上-下-上的路径会被计算一次，因此可以减去交错经过三次的，再加上交错经过四次的，以此类推。

因此答案为：
$$
f_i=\frac1{2^n}(\sum_{y=-i}^i[y\equiv n(\bmod 2)]C_n^{\frac{n+y}2}+\sum_{d=1}^{+\infty}(-1)^i\sum_{y=-i+2d*(i+1)}^{i+2d*(i+1)}[y\equiv n(\bmod 2)]C_n^{\frac{n+y}2}
$$
先求出组合数的前缀和即可 $O(\frac ni)$ 计算。

复杂度 $O(n\log n)$

###### Code

2.449s,16.9MB

```cpp
#include<cstdio>
using namespace std;
#define N 1005000
int T,n,p,fr[N],ifr[N],su[N*2];
int pw(int a,int b,int p){int as=1;while(b){if(b&1)as=1ll*a*as%p;a=1ll*a*a%p;b>>=1;}return as;}
int calc(int l,int r)
{
	if(r<0||l>n*2)return 0;
	if(l<0)l=0;if(r>n*2)r=n*2;
	return (p+su[r]-(l?su[l-1]:0))%p;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&p);
		fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%p;
		ifr[n]=pw(fr[n],p-2,p);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%p;
		for(int i=0;i<=n*2;i++)su[i]=0;
		for(int i=0;i<=n;i++)su[i*2]=1ll*fr[n]*ifr[i]%p*ifr[n-i]%p;
		for(int i=1;i<=n*2;i++)su[i]=(su[i]+su[i-1])%p;
		int as=0;
		for(int i=1;i<=n;i++)
		{
			int s1=calc(n-i,n+i),tp=p-2;
			for(int t=i+1;t<=n;t=t+(i+1)*2,tp=p-tp)s1=(s1+1ll*tp*calc(n+t+1,n+t+(i+1)*2-1))%p;
			as=(as+p-1ll*s1*s1%p)%p;
		}
		as=(as+1ll*(n+1)*pw(4,n,p))%p;
		printf("%d\n",1ll*as*pw(4,p-n-1,p)%p);
	}
}
```

##### 9I Little prince and the garden of roses

###### Problem

给一个 $n\times n$ 的矩阵 $a$，每个位置上有一个数。

你需要给每个位置选择一个颜色，使得不存在两个位于同一行/同一列的元素，满足它们的值和颜色都相同，

你希望使用的颜色数量最少。输出一个方案。

$n\leq 300,T\leq 4$

$1.5s,256MB$

###### Sol

显然可以对于每种数分别考虑。考虑一个两侧各有 $n$ 个点的二分图对于一个数 $a_{i,j}$，可以看成一条左边第 $i$ 个点连向右边第 $j$ 个点的边。

考虑对这种数染色，显然一种颜色的数对应的边不能有公共点，否则不满足要求。因此相当于每种颜色的边都是一个匹配。

因此问题可以看成给一个二分图，将它的边集划分成数量尽量少的匹配。

显然个数下界就是最大度数，可以猜想下界可以取到。

一个显然的想法是对最大度数归纳，对于最大度数为 $k+1$ 的情况，只需要找到一个匹配，使得所有度数为 $k+1$ 的点都在匹配中，那么删去这个匹配后根据归纳即可在 $k$ 步之内划分剩余部分，

考虑先找一个左边度数为 $k+1$ 的点到右侧所有点的匹配。右侧每个点度数不超过 $k+1$，因此左侧 $a$ 个度数为 $k+1$ 的点至少与右侧 $a$ 个点相邻，因此根据Hall定理这样的匹配一定存在。

同理，可以对于右边所有度数为 $k+1$ 的点找到一个到左侧的匹配。考虑将这两个匹配合并起来，如果点 $i$ 的度数为 $k+1$，记 $t_i$ 表示 $i$ 在匹配中连向的点，否则记 $t_i=0$。

此时只考虑这个匹配中的边，每个点度数不超过 $2$，因此会形成若干个偶环和链。

对于偶环，直接选择所有奇数位置的边即可。对于长度为偶数的链，也可以直接选择奇数位置的边。对于长度为奇数的链，显然有一个链的端点满足 $t_i=0$，因此可以删去这个点，将剩下的点匹配。

因此存在一个匹配满足条件。直接dinic实现匹配的复杂度为 $O(n^{3.5})$

可以发现，如果当前的最大度数 $k$ 为偶数，那么将奇数度数的点连接后，可以求出若干个欧拉回路。

对于一个欧拉回路，可以将边按照回路上的顺序按照奇偶性分成两类。可以发现对于每一个点，它的所有出边都会正好被分成两个均等的部分。此时两个部分的最大度数都不超过 $\frac k2$，可以分治解决。

因此在最大度数为奇数的时候dinic找一组匹配，为偶数时分成两部分即可。

复杂度 $O(n^{2.5}\log n)$

###### Code

702ms,19.4MB

```cpp
#include<cstdio>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
#define N 305
int T,n,m,s[N][N],as[N][N],id[N],tp[N],cl[N*N],d[N*2],c11,c21;
struct sth{int a,b;};
vector<sth> sn[N*2],f1[N*N];
void dfs1(int x,int c)
{
	while(cl[sn[x].back().b])sn[x].pop_back();
	sth tp=sn[x].back();
	cl[tp.b]=c;
	if(c==1)c11++;else c21++;
	if(d[tp.a]&1)d[tp.a]=0;
	else dfs1(tp.a,3^c);
}
void dfs2(int x,int c)
{
	while(sn[x].size()&&cl[sn[x].back().b])sn[x].pop_back();
	if(sn[x].empty())return;
	sth tp=sn[x].back();
	if(c==1)c11++;else c21++;
	cl[tp.b]=c;
	dfs2(tp.a,3^c);
}
int head[N*2],cur[N*2],cnt,dis[N*2],ct,nt[N*2],fu[N*2],vis[N*2],is[N*2];
struct edge{int t,next,v;}ed[N*N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;}
bool bfs(int s,int t)
{
	for(int i=1;i<=ct;i++)cur[i]=head[i],dis[i]=-1;
	queue<int> qu;
	qu.push(s);dis[s]=0;
	while(!qu.empty())
	{
		int u=qu.front();qu.pop();
		for(int i=head[u];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]==-1)
		{
			dis[ed[i].t]=dis[u]+1;qu.push(ed[i].t);
			if(ed[i].t==t)return 1;
		}
	}
	return 0;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
vector<int> solve(vector<sth> s)
{
	int c1=0,c2=0,su=s.size(),mx1=0;
	for(int i=0;i<su;i++)
	{
		if(!id[s[i].a])id[s[i].a]=++c1,tp[c1]=s[i].a;
		s[i].a=id[s[i].a];
	}
	for(int i=1;i<=c1;i++)id[tp[i]]=0;
	for(int i=0;i<su;i++)
	{
		if(!id[s[i].b])id[s[i].b]=++c2,tp[c2]=s[i].b;
		s[i].b=id[s[i].b];
	}
	for(int i=1;i<=c2;i++)id[tp[i]]=0;
	for(int i=1;i<=c1+c2;i++)d[i]=0;
	for(int i=1;i<=c1+c2;i++)sn[i].clear();
	for(int i=0;i<su;i++)
	{
		d[s[i].a]++;d[s[i].b+c1]++;cl[i]=0;
		sn[s[i].a].push_back((sth){s[i].b+c1,i});
		sn[s[i].b+c1].push_back((sth){s[i].a,i});
	}
	for(int i=1;i<=c1+c2;i++)mx1=max(mx1,d[i]);
	int fg=1;
	for(int i=1;i<=c1+c2;i++)if(d[i]>1)fg=0;
	if(fg)
	{
		vector<int> as;
		for(int i=0;i<su;i++)as.push_back(1);
		return as;
	}
	if(mx1&1)
	{
		for(int i=1;i<=c1+c2;i++)nt[i]=fu[i]=vis[i]=is[i]=0;
		ct=c1+c2+2;
		for(int i=1;i<=ct;i++)head[i]=0;cnt=1;
		for(int i=1;i<=c1;i++)if(d[i]==mx1)adde(ct-1,i,1);
		for(int i=0;i<su;i++)adde(s[i].a,s[i].b+c1,1);
		for(int i=1;i<=c2;i++)adde(i+c1,ct,1);
		while(bfs(ct-1,ct))dfs(ct-1,ct,1e9);
		for(int i=1;i<=c1;i++)if(d[i]==mx1)for(int j=head[i];j;j=ed[j].next)if(!ed[j].v)nt[i]=ed[j].t;
		for(int i=1;i<=ct;i++)head[i]=0;cnt=1;
		for(int i=1;i<=c2;i++)if(d[i+c1]==mx1)adde(ct-1,i+c1,1);
		for(int i=0;i<su;i++)adde(s[i].b+c1,s[i].a,1);
		for(int i=1;i<=c1;i++)adde(i,ct,1);
		while(bfs(ct-1,ct))dfs(ct-1,ct,1e9);
		for(int i=1;i<=c2;i++)if(d[i+c1]==mx1)for(int j=head[i+c1];j;j=ed[j].next)if(!ed[j].v)nt[i+c1]=ed[j].t;
		for(int i=1;i<=c1+c2;i++)fu[nt[i]]=1;
		for(int i=1;i<=c1+c2;i++)if(!fu[i])
		{
			int nw=i,tp=1;
			while(nw)vis[nw]=1,is[nw]=tp,tp^=1,nw=nt[nw];
		}
		for(int i=1;i<=c1+c2;i++)if(!vis[i])
		{
			int nw=i,tp=1;
			while(nw&&!vis[nw])vis[nw]=1,is[nw]=tp,tp^=1,nw=nt[nw];
		}
		vector<int> f1,as;
		vector<sth> t1;
		for(int i=0;i<su;i++)
		{
			as.push_back(0);
			int f=s[i].a,t=s[i].b;
			if((is[f]&&nt[f]==t+c1)||(is[t+c1]&&nt[t+c1]==f))as[i]=1;
			else f1.push_back(i),t1.push_back(s[i]);
		}
		vector<int> as1=solve(t1);
		for(int i=0;i<as1.size();i++)as[f1[i]]=as1[i]+1;
		return as;
	}
	for(int i=1;i<=c1+c2;i++)if(d[i]&1)d[i]=0,dfs1(i,1);
	c11=c21=0;
	for(int i=1;i<=c1+c2;i++)
	dfs2(i,1);
	vector<sth> t1,t2;
	vector<int> s1,s2,as;
	for(int i=0;i<su;i++)if(cl[i]==1)s1.push_back(i),t1.push_back(s[i]);else s2.push_back(i),t2.push_back(s[i]);
	for(int i=0;i<su;i++)as.push_back(0);
	vector<int> v1=solve(t1),v2=solve(t2);
	int mx=0;
	for(int i=0;i<v1.size();i++)as[s1[i]]=v1[i],mx=max(mx,v1[i]);
	for(int i=0;i<v2.size();i++)as[s2[i]]=v2[i]+mx;
	return as;
}
void solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)scanf("%d",&s[i][j]);
	for(int i=1;i<=n*n;i++)f1[i].clear();
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)f1[s[i][j]].push_back((sth){i,j});
	for(int i=1;i<=n*n;i++)
	{
		vector<int> s1=solve(f1[i]);
		for(int j=0;j<s1.size();j++)as[f1[i][j].a][f1[i][j].b]=s1[j];
	}
	int mx=0,su=0;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)mx=max(mx,as[i][j]),su+=as[i][j]>1;
	printf("%d %d\n",mx-1,su);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(as[i][j]>1)printf("%d %d %d\n",i,j,as[i][j]-1);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 9K ZYB's kingdom

###### Problem

给一棵 $n$ 个点的树，给定每个点的权值 $v_i$。每个点还有一个 $c_i$，初始 $c_i=0$。

进行 $q$ 次操作，每次操作形如：

1. 给定 $k$ 以及 $k$ 个点 $s_1,...,s_k$，考虑所有的点对 $i,j(i\neq j)$，如果 $i$ 到 $j$ 的路径中没有经过任何一个给定点，则令 $c_i$ 加上 $v_j$。

2. 询问一个 $c_i$

$n,q\leq 2\times 10^5,\sum n,\sum q,\sum k\leq 10^6$

$6s,256MB$

###### Sol

对于一次修改，如果看成删去所有给定点，则剩下的每一个连通块内部的两个点都满足条件。因此每个点的 $c_i$ 会加上连通块内除去自己的 $v_i$ 的和。

此时可以看成每个点 $c_i$ 加上连通块内所有 $v_i$ 的和，随后每个点 $c_i$ 减去自己的 $v_i$。减去 $v_i$ 可以直接记录修改次数。

考虑一次修改，首先对给定点建虚树。为了简便，可以在树上加一个点 $n+1$ 作为根，每次修改都看成包含 $n+1$，显然这样不改变答案。

此时的一个连通块一定是一个给定点的儿子子树的一部分，可以将子树分成两部分：

1. 子树内部还有其它关键点。这部分子树只有 $O(k)$ 个，可以在虚树上找到这些部分。这个连通块为子树减去子树内的若干个子树部分。显然减去的子树部分数量总和是 $O(k)$ 的，因此按照dfs序排序后，这个连通块可以看成若干个区间，BIT上操作即可。

2. 子树内部没有其它关键点。这部分子树的数量可能很大，因此不能每个子树做一次。

对树进行轻重链剖分，考虑一个给定点，将它的重儿子子树直接处理，然后在这个点上打一个标记，表示它的所有轻儿子子树内部都应该做一次操作。如果一个轻儿子子树是第一种情况，则在轻儿子中打一个标记表示这个子树内应该少做一次之前的操作。

在询问的时候，答案来自于直接做的部分的结果加上标记的结果。显然每一个标记都影响一个轻边的子树，因此询问一个点时最多用到 $O(\log n)$ 个标记，直接做即可。

复杂度 $O((q+\sum k)\log n)$

###### Code

2.776s,33.1MB

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 200500
#define ll long long
int T,n,q,a,b,head[N],cnt,v[N],k,s[N],nq;
ll su[N];
int sz[N],sn[N],ct,lb[N],rb[N],f[N],dep[N],tp[N];
struct edge{int t,next;}ed[N*2];
bool cmp(int a,int b){return lb[a]<lb[b];}
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void dfs0(int u,int fa)
{
	sz[u]=1;sn[u]=0;dep[u]=dep[fa]+1;f[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u],sz[u]+=sz[ed[i].t];
}
void dfs1(int u,int fa,int v)
{
	tp[u]=v;lb[u]=++ct;
	if(sn[u])dfs1(sn[u],u,v);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs1(ed[i].t,u,ed[i].t);
	rb[u]=ct;
}
int LCA(int x,int y)
{
	while(tp[x]!=tp[y])
	{
		if(dep[tp[x]]<dep[tp[y]])y=f[tp[y]];
		else x=f[tp[x]];
	}
	return dep[x]<dep[y]?x:y;
}
int id[N],sr[N],c1,st[N],c2,is[N];
vector<int> sn1[N];
void init()
{
	sort(s+1,s+k+1,cmp);
	for(int i=1;i<=c1;i++)id[sr[i]]=0,is[i]=0,sn1[i].clear();
	c1=0;st[c2=1]=s[1];
	for(int i=1;i<=k;i++)if(!id[s[i]])id[s[i]]=++c1,sr[c1]=s[i],is[c1]=1;
	for(int i=2;i<=k;i++)
	while(1)
	{
		int l=LCA(st[c2],s[i]);
		if(l==st[c2]){st[++c2]=s[i];break;}
		else if(dep[l]<=dep[st[c2-1]])sn1[id[st[c2-1]]].push_back(id[st[c2]]),c2--;
		else
		{
			if(!id[l])id[l]=++c1,sr[c1]=l;
			sn1[id[l]].push_back(id[st[c2]]);st[c2]=l;
		}
	}
	for(int i=1;i<c2;i++)sn1[id[st[i]]].push_back(id[st[i+1]]);
}
vector<int> s1;
void dfs2(int x)
{
	if(is[x]){s1.push_back(x);return;}
	for(int i=0;i<sn1[x].size();i++)dfs2(sn1[x][i]);
}
ll v1[N],v2[N],tr[N];
void add(int x,ll v){for(int i=x;i<=n;i+=i&-i)tr[i]+=v;}
ll que(int x){ll as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
void add1(int l,int r,ll v){add(l,v);add(r+1,-v);}
void dfs3(int x)
{
	int fg=0;
	v1[sr[x]]++;
	for(int i=0;i<sn1[x].size();i++)
	{
		int t=sn1[x][i],t1=sr[t];
		while(f[t1]!=sr[x]&&tp[t1]!=tp[sr[x]])t1=tp[t1]==t1?f[t1]:tp[t1];
		if(tp[t1]==tp[sr[x]])t1=sn[sr[x]],fg=1;else v2[t1]++;
		s1.clear();dfs2(t);vector<int> s2=s1;
		ll su1=su[rb[t1]]-su[lb[t1]-1];
		for(int j=0;j<s2.size();j++)su1-=su[rb[sr[s2[j]]]]-su[lb[sr[s2[j]]]-1];
		add1(lb[t1],rb[t1],su1);
		for(int j=0;j<s2.size();j++)add1(lb[sr[s2[j]]],rb[sr[s2[j]]],-su1);
		for(int j=0;j<s2.size();j++)dfs3(s2[j]);
	}
	if(!fg&&sn[sr[x]])
	{
		int t1=sn[sr[x]];
		ll su1=su[rb[t1]]-su[lb[t1]-1];
		add1(lb[t1],rb[t1],su1);
	}
}
ll query(int x)
{
	ll as=que(lb[x]);
	while(x)
	{
		x=tp[x];
		if(!f[x])break;
		as+=(v1[f[x]]-v2[x])*(su[rb[x]]-su[lb[x]-1]);
		x=f[x];
	}
	return as;
}
int solve()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n+1;i++)head[i]=tr[i]=v1[i]=v2[i]=0;
	cnt=ct=nq=0;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	adde(n+1,n);n++;
	dfs0(n,0);dfs1(n,0,n);
	for(int i=1;i<=n;i++)su[lb[i]]=v[i];
	for(int i=1;i<=n;i++)su[i]+=su[i-1];
	while(q--)
	{
		scanf("%d",&a);
		if(a==1)
		{
			scanf("%d",&k);nq++;
			for(int i=1;i<=k;i++)scanf("%d",&s[i]),add1(lb[s[i]],lb[s[i]],v[s[i]]);
			s[++k]=n;
			init();dfs3(id[n]);
		}
		else scanf("%d",&b),printf("%lld\n",query(b)-1ll*nq*v[b]);
	}
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 10B Pty with card

###### Problem

定义 $f(n)$ 为：

有 $n$ 个人排成一队，初始时每个人手上有一张牌。随后循环进行如下操作：

设当前是第 $t$ 次操作（从 $1$ 开始计数），当前的第一个人将手上的 $x$ 张牌给下一个人，其中 $t$ 为奇数时 $x=1$，否则 $x=2$。如果当前的人手上牌不够 $x$ 张，则他把手中所有牌给下一个人。

如果当前人手上没有牌，则他离开这个队列。否则他会进入队列末尾。

如果某个时刻游戏中只剩下一个人，则 $f(n)=0$，否则可以发现所有人的牌数量会循环，$f(n)$ 即为循环节长度。

给一棵 $n$ 个点的树，点有点权 $v_i$，对于每个点 $u$，求：

$$
\sum_{i=1}^n f(v_i+dis(i,u))
$$

$1\leq n,v_i\leq 10^5,\sum n\leq 5\times 10^5$

$8s,512MB$

###### Sol

打表可以发现如下结论：

如果 $n\leq 2$，$f(n)=0$。

否则，令 $m=n-2+(n\&1)$。如果 $m=2^k(k\in \N)$，则 $f(n)=0$，否则 $f(n)=2*\frac m{lowbit(m)}$

大概按照 $\bmod 4$ 分类手玩一下就能说明。

考虑点分治，设根为 $x$，令 $ds_u$ 为 $u$ 到根的距离，则 $y$ 到 $x$ 的 $m$ 为：
$$
v_y+dis_y+dis_x-2+((v_y+dis_y+dis_x)\&1)
$$
对于 $y=u$ 的情况，可以暴力计算，否则 $v_y+dis_y\geq 2$。先不考虑 $=2^k$ 时的特判，点分时的问题相当于给若干个数 $v=v_y+dis_y-2$ ，然后每次给出一个 $x$ ，求：
$$
\sum_v\frac{v+x+((v+x)\&1)}{lowbit(v+x+((v+x)\&1))}
$$
$\&1$ 难以处理，因此可以对于一个 $v$ ，将 $v$ 和 $v+1$ 都加入一次，然后变为询问：
$$
\sum_v[2|(v+x)]\frac{v+x}{lowbit(v+x)}
$$
考虑对于每种 $lowbit$，求出等于这个 $lowbit$ 的 $v$ 的数量和和，即可求出答案。可以发现 $lowbit(v+x)=2^k$ 相当于 $2^{30}-x$ 和 $v$ 在二进制的后 $k$ 位相同，在 $2^k$ 这一位上不同。

因此将所有 $v$ 的二进制表示按照低位到高位建Trie，在Trie上查询即可 $O(\log v)$ 得到答案。

对于 $=2^k$ 的情况，记录每种数出现的次数，然后枚举 $v+x$ 即可。这种情况在上面的计算中每一个会多算 $1$，可以直接减去。

复杂度 $O(n\log n\log v)$，非常卡常~~建议连着交10遍~~。

###### Code

7.519s,57MB

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 100500
#define M 1000500
#define ll long long
int T,n,a,b,v[N],sz1[N],s1,tp,sv,head[N],cnt,dep[N],vis[N],rv[M];
ll as[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
int calc(int x)
{
	if(x<=2)return 0;x-=2;
	if(x&1)x++;
	x/=x&-x;
	return x*(x>1);
}
int ct,ch[M][2],sz[M];
ll su[M];
void ins(int x)
{
	int nw=1;rv[x]++;
	for(int i=0;i<20;i++)
	{
		int tp=(x>>i)&1;
		if(!ch[nw][tp])ch[nw][tp]=++ct;
		nw=ch[nw][tp];sz[nw]++;su[nw]+=x;
	}
}
ll query(int v)
{
	int v2=((1ll<<21)-v),nw=1;
	ll as=0;
	for(int i=2;i<1<<19;i<<=1)if(v<=i)as-=rv[i-v];
	for(int i=0;i<20;i++)
	{
		int tp=(v2>>i)&1;
		if(i)as+=(su[ch[nw][!tp]]+1ll*v*sz[ch[nw][!tp]])>>i;
		nw=ch[nw][tp];
	}
	return as;
}
vector<int> sn[N];
void dfs1(int u,int fa,int fr)
{
	sn[fr].push_back(u);
	sz1[u]=1;dep[u]=dep[fa]+1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs1(ed[i].t,u,fr),sz1[u]+=sz1[ed[i].t];
}
void dfs2(int u,int fa)
{
	int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs2(ed[i].t,u),mx=max(mx,sz1[ed[i].t]);
	mx=max(mx,sv-sz1[u]);
	if(mx<tp)tp=mx,s1=u;
}
void dfs3(int u)
{
	for(int i=1;i<=ct;i++)ch[i][0]=ch[i][1]=sz[i]=su[i]=0;ct=1;
	vis[u]=1;as[u]+=calc(v[u]);dep[u]=0;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		sn[ed[i].t].clear();
		dfs1(ed[i].t,u,ed[i].t);
		for(int j=0;j<sn[ed[i].t].size();j++)
		{
			int t=sn[ed[i].t][j];
			ins(v[t]+dep[t]-2);
			ins(v[t]+dep[t]-1);
			as[u]+=calc(v[t]+dep[t]);as[t]+=calc(v[u]+dep[t]);
		}
	}
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	for(int j=0;j<sn[ed[i].t].size();j++)
	{
		int t=sn[ed[i].t][j];
		as[t]+=query(dep[t]);
	}
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	for(int j=0;j<sn[ed[i].t].size();j++)
	{
		int t=sn[ed[i].t][j];
		rv[v[t]+dep[t]-2]--;
		rv[v[t]+dep[t]-1]--;
	}
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		for(int j=1;j<=ct;j++)ch[j][0]=ch[j][1]=sz[j]=su[j]=0;ct=1;
		for(int j=0;j<sn[ed[i].t].size();j++)
		{
			int t=sn[ed[i].t][j];
			ins(v[t]+dep[t]-2);
			ins(v[t]+dep[t]-1);
		}
		for(int j=0;j<sn[ed[i].t].size();j++)
		{
			int t=sn[ed[i].t][j];
			as[t]-=query(dep[t]);
		}
		for(int j=0;j<sn[ed[i].t].size();j++)
		{
			int t=sn[ed[i].t][j];
			rv[v[t]+dep[t]-2]--;
			rv[v[t]+dep[t]-1]--;
		}
	}
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		tp=1e7;sv=sz1[ed[i].t];
		dfs2(ed[i].t,u);
		dfs3(s1);
	}
}
void solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)as[i]=head[i]=vis[i]=0;cnt=0;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	tp=1e7;sv=n;dfs1(1,0,1);dfs2(1,0);dfs3(s1);
	for(int i=1;i<=n;i++)printf("%lld%c",as[i]*2,i==n?'\n':' ');
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```

##### 10F Pty loves lcm

###### Problem

定义 $f(x,y)=lcm(x,x+1,...,y)$

给定 $l,r$，求 $\sum_{x<y}[f(x,y)\in[l,r]]\phi(f(x,y))$，模 $2^{32}$

$T\leq 50,r\leq 10^{18}$

$5s,512MB$

###### Sol

对于 $r-l=2$ 的情况，显然 $lcm$ 是 $v^3$ 级别的，因此此时 $r\leq 1.5\times 10^6$。

枚举这个范围内的所有 $l$，然后枚举 $r$ 直到 $lcm>10^{18}$，这部分的次数应该不会超过 $2\times 10^6$。

显然 $\phi(xy)=\frac{\phi(x)\phi(y)\gcd(x,y)}{\phi(\gcd(x,y))}$，因可以快速求出 $lcm$ 的 $\phi$。

对于 $r-l=1$ 的情况，显然 $lcm(i,i+1)=i(i+1)$，那么相当于求 $\phi(i(i+1))$ 的区间和。

直接做非常困难，但是此时 $r\leq 10^9$，因此可以每 $10^6$ 个数打表。对于 $10^6$ 范围内的求和，因为 $\phi(i(i+1))=\phi(i)\phi(i+1)$，区间筛出 $\phi$ 即可。

复杂度 $O(r^{\frac 13}\log r)$

###### Code

没写

##### 10I Pty loves SegmentTree

###### Problem

给定 $a,b,k$，定义 $f_n$ 为：

考虑所有有 $n$ 个叶子，每个非叶子节点有 $2$ 个儿子且儿子有序的二叉树，求它们的权值之和。

定义一个点的权值为：

1. 叶子的权值为 $1$。
2. 如果一个非叶子节点的右子树中叶子数量为 $k$，则点权为 $b$。
3. 否则点权为 $a$。

树的权值定义为所有点权值的乘积。

$q$ 次询问，每次给出 $l,r$，询问 $\sum_{i=l}^rf_i^2$，模 $998244353$

$T\leq 5,q\leq 5\times 10^4,r,k\leq 10^7$

$15s,512MB$

###### Sol

考虑直接求出 $f$，然后 $O(1)$ 回答询问。

首先特判 $a=0$ ，此时如果 $k>1$ 显然 $f_i=[i=1]$，$k=1$ 时显然只有全部向左的链有权值，此时 $f_i=b^{i-1}$。

对于剩下的情况，可以将每个点权除以 $a$，最后乘上 $a^{n-1}$。此时可以看成只有情况2中有一个权值 $\frac ba$。

因为特殊情况只和右儿子相关，因此可以看成左兄弟右儿子，将二叉树变成一个有 $n$ 个点的有根树。此时如果一个非根的点子树大小为 $k$，则它会向它父亲展开后的某个点贡献一个 $\frac ba$。因此此时一棵树的权值为子树大小为 $k$ 的非根节点数量。

考虑去掉非根的限制，可以发现只有 $f_k$ 会多乘上一个 $\frac ba$，最后除掉即可。

首先考虑不计算 $\frac ba$ 的情况，此时设 $f$ 的生成函数为 $F(x)$，枚举根的儿子数量有：
$$
F(x)=x(1+F(x)+F^2(x)+...)\\
F(x)(1-F(x))=x\\
F^2(x)-F(x)+x=0\\
F(x)=\frac{1\pm\sqrt{1-4x}}{2}
$$
显然里面应该取减号。显然这个式子和卡特兰数相同，设 $c_k$ 表示此时的 $f_k$。

考虑计算 $\frac ba$ 的情况，此时 $F(x)$ 等于先算出 $\frac{x}{1-F(x)}$，再对 $x^k$ 项乘以 $\frac ba$。显然 $x^k$ 项的值与之前相同，因此可以将这一项乘看成加上 $c_k(\frac ba-1)x^k$。

此时的生成函数 $F(x)$ 满足（设 $c_k(\frac ba-1)=c$：
$$
F(x)=\frac x{1-F(x)}+cx^k\\
F(x)^2-(cx^k+1)F(x)+cx^k+x=0\\
F(x)=\frac{cx^k+1-\sqrt{c^2x^{2k}-2cx^k-4x+1}}{2}
$$
因此只需要计算里面的生成函数的开根即可，记 $G(x)=\sqrt{c^2x^{2k}-2cx^k-4x+1},H(x)=c^2x^{2k}-2cx^k-4x+1$，则：
$$
G'(x)=\frac12*H^{-\frac 12}(x)H'(x)\\
G'(x)H(x)=\frac12G(x)H'(x)
$$
同时取 $[x^n]$ 可以得到：
$$
(n+1)g_{n+1}-4ng_n-2c(n-k+1)g_{n-k+1}+c^2(n-2k+1)g_{n-2k+1}=\frac12(-4g_{n}-2ckg_{n-k+1}-2kc^2g_{n-2k+1})\\
(n+1)g_{n+1}=(4n-2)g_n+2c(2n-3k+2)g_{n-k+1}-c^2(n-3k+1)g_{n-2k+1}
$$
因此可以得到：
$$
g_n=\frac1n((4n-6)g_{n-1}+2c(2n-3k)g_{n-k}+c^2(3k-n)g_{n-2k})
$$
直接递推即可，需要一个线性求逆元。

复杂度 $O(q+r)$

###### Code

3.291s,157.8MB

```cpp
#include<cstdio>
using namespace std;
#define N 10005000
#define mod 998244353
int T,n=1e7,k,q,a,b,l,r,fr[N],ifr[N],inv[N],f[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void getf()
{
	if(!a)
	{
		f[1]=1;for(int i=2;i<=n;i++)f[i]=1ll*f[i-1]*(k==1?b:0)%mod;
		return;
	}
	int tp=1ll*b*pw(a,mod-2)%mod;
	int f1=1;for(int i=1;i<k;i++)f1=1ll*f1*(4*i-2)%mod*inv[i+1]%mod;
	f1=1ll*f1*(tp+mod-1)%mod;
	f[0]=1;
	for(int i=1;i<=n;i++)
	{
		int s1=1ll*f[i-1]*(mod+8*i-12)%mod;
		if(i>=k)s1=(s1+1ll*2*f1*f[i-k]%mod*(mod+2*i-3*k))%mod;
		if(i>=2*k)s1=(s1+1ll*f1*f1%mod*f[i-2*k]%mod*(mod+6*k-i*2))%mod;
		f[i]=1ll*s1*inv[i]%mod*(mod+1)/2%mod;
	}
	for(int i=0;i<=n;i++)
	{
		int vl=0;
		if(i==0)vl=1;if(i==k)vl=f1;
		f[i]=1ll*(mod+1)/2*(vl+mod-f[i])%mod;
	}
	f[k]=1ll*f[k]*pw(tp,mod-2)%mod;
	int su=1;
	for(int i=1;i<=n;i++)f[i]=1ll*f[i]*su%mod,su=1ll*su*a%mod;
}
void solve()
{
	scanf("%d%d%d%d",&q,&k,&b,&a);
	getf();
	for(int i=1;i<=n;i++)f[i]=(1ll*f[i]*f[i]+f[i-1])%mod;
	while(q--)scanf("%d%d",&l,&r),printf("%d\n",(f[r]-f[l-1]+mod)%mod);
}
int main()
{
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*i*fr[i-1]%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int i=1;i<=n;i++)inv[i]=1ll*fr[i-1]*ifr[i]%mod;
	scanf("%d",&T);while(T--)solve();
}
```

