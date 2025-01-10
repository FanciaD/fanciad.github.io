---
title: ARC 题解
date: '2022-04-12 13:10:06'
updated: '2022-04-12 13:10:06'
tags: Mildia
permalink: Festivalofillumious/
description: AtCoder Regular Contest
mathjax: true
---

#### AtCoder Regular Contest Solutions

部分证明参(chao)考(xie)了官方题解

##### ARC104E Random LIS

###### Problem

给出 $n$ 以及 $n$ 个正整数 $a_i$。

有一个长度为 $n$ 的正整数序列 $x$，其中 $x_i$ 在 $[1,a_i]$ 间等概率随机生成。

求 $x$ 的LIS长度的期望，模 $998244353$。

$n\leq 6,a_i\leq 10^9$

$2s,1024MB$

###### Sol

LIS长度只和数的相对顺序相关，即将所有数按照从小到大顺序重新赋值，且 $x_i$ 相同时给后面位置更小的值，得到的排列的LIS长度即为原序列的LIS长度。

按照 $a_i$ 将权值分成 $n$ 段，先枚举每个 $x_i$ 所在的段，这样就处理了上限的问题。

接下来考虑一段内的情况，考虑枚举这一段内的排列，求这一段内有多少种取值方式得到这个排列。

从取值过程到排列的方式相当于将值相同的放在一组，按照值从小到大给每个组内的位置标号，一个组内从后向前依次标号。

因此可以发现设标号后的排列为 $p_i$，$r_i$ 为满足 $p_{r_i}=i$ 的位置，则 $r_i,\cdots,r_j$ 构成一个组当且仅当 $r_i>r_{i+1}>\cdots>r_j$。而确定了组的个数后，方案数即为从可以取的值中选择组个数个值的方案数。因此设 $dp_{i,j}$ 表示划分到前 $i$ 个位置，划分了 $j$ 组的方案数。转移复杂度为 $O(n^3)$。



枚举所有情况后，求和每种情况的贡献即可得到答案。上述过程可以通过dfs完成。

复杂度显然不超过 $O((n!)^2*n^3)$。但可以发现实际上的情况数为 $(2n-1)!!$，因此复杂度为 $O((2n-1)!!*n^3)$

另外一种做法是枚举最后整体的组情况，再dp计算将这些组放进值的限制的方案数。这样后半部分复杂度相同，前半部分复杂度为有序划分数，即A000670(n)。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 7
#define mod 1000000007
int n,v[N],st[N],inv[N]={0,1,(mod+1)/2,(mod+1)/3,(mod+1)/4,(mod*2+1)/5,(mod+1)/6},p[N],fp[N],as;
int C(int x,int y){int as=1;for(int i=1;i<=y;i++)as=1ll*as*(x-i+1)%mod*inv[i]%mod;return as;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int calc(int n,int k,int *p)
{
	int dp[N][N]={0};
	for(int i=1;i<=n;i++)fp[p[i]]=i;
	dp[0][0]=1;
	for(int i=1;i<=n;i++)
	for(int j=0;j<i;j++)
	{
		int fg=1;
		for(int l=j+2;l<=i;l++)if(fp[l]>fp[l-1])fg=0;
		if(!fg)continue;
		for(int l=0;l<=j;l++)dp[i][l+1]=(dp[i][l+1]+dp[j][l])%mod;
	}
	int as=0;
	for(int i=1;i<=n;i++)as=(as+1ll*dp[n][i]*C(k,i))%mod;
	return as;
}
int getlis(int *p)
{
	int dp[N]={0};
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<i;j++)if(p[j]<p[i])dp[i]=max(dp[i],dp[j]);
		dp[i]++;
	}
	int as=0;
	for(int i=1;i<=n;i++)as=max(as,dp[i]);
	return as;
}
void dfs(int d,int p[N],int vl)
{
	if(!vl)return;
	if(d==n+1){for(int i=1;i<=n;i++)if(!p[i])return;
	as=(as+1ll*vl*getlis(p))%mod;return;}
	for(int i=1;i<=n;i++)if(!p[i]&&v[i]<st[d])return;
	int st2=0,ls=0;
	for(int i=1;i<=n;i++)if(!p[i])st2|=1<<i-1;else ls++;
	dfs(d+1,p,vl);
	for(int t=st2;t;t=(t-1)&st2)
	{
		int ct=0,st1[N],p1[N];
		for(int i=1;i<=n;i++)if(t&(1<<i-1))ct++,st1[ct]=i,p1[ct]=ct;
		do{
			int tp=calc(ct,st[d]-st[d-1],p1);
			for(int i=1;i<=ct;i++)p[st1[i]]=p1[i]+ls;
			dfs(d+1,p,1ll*vl*tp%mod);
			for(int i=1;i<=ct;i++)p[st1[i]]=0;
		}while(next_permutation(p1+1,p1+ct+1));
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),st[i]=v[i];
	sort(st+1,st+n+1);
	dfs(1,p,1);
	for(int i=1;i<=n;i++)as=1ll*as*pw(v[i],mod-2)%mod;
	printf("%d\n",as);
}
```



##### ARC104F Visiblity Sequence

###### Problem

给出 $n$ 以及一个长度为 $n$ 的正整数序列 $x$。

有一个长度为 $n$ 的正整数序列 $a$，满足 $\forall i,1\leq a_i\leq x_i$。

对于一个序列 $a$，定义 $p_i$ 为：

如果存在 $j$ 满足 $1\leq j<i$ 且 $p_j>p_i$，则 $p_i$ 为满足这个条件的 $j$ 中最大的一个。否则 $p_i=-1$。

求出考虑所有满足条件的 $a$ 得到的 $p$ 中，不同的序列 $p$ 类型数量，答案对 $998244353$ 取模。

$n\leq 100,x_i\leq 10^5$

$2s,1024MB$

###### Sol

对于一个序列 $a$，考虑所有 $a_i$ 最大的数中 $i$ 最大的一个，设这个位置为 $x$。

则显然 $p_x=-1$，而对于大于 $x$ 的位置 $i$，一定有 $p_x>p_i$，因此这些位置的 $p_x$ 不可能是 $-1$，且至少为 $x$。因此从这个位置分开，两部分的 $p$ 可以看成独立。考虑分别求出 $[1,x-1],[x+1,n]$ 部分的 $p$，然后将 $[x+1,n]$ 部分的 $p$ 中等于 $-1$ 的位置值变为 $x$，可以发现这就等于整个序列的 $p$。

如果对两侧的区间重复这个过程，则考虑每次选出的 $x$ 构成的树关系，这样得到的即为序列 $a$ 的笛卡尔树。

此时可以发现通过笛卡尔树可以构造序列 $p$，而对于序列 $p$，每次选出序列中最后一个 $-1$ 所在的位置 $x$，然后反过来做之前的过程，将右侧所有 $p_i=x$ 的位置值变为 $-1$，然后对两侧分别做，就可以得到笛卡尔树。

因此问题变为求可能的笛卡尔树的数量，其中存在值相同时取最右侧的最大值。



对于一棵笛卡尔树，可以从上往下贪心确定每个点的值。根节点的权值为该点的权值上限 $x_i$。如果确定了一个点的权值为 $x$，则它的左儿子权值不能超过 $x$，右儿子权值不能超过 $x-1$，因此两个儿子分别取这个上限与自己的权值上限的最小值最优。如果某个点权值最优情况还是 $0$ 则无解。

因此可以设 $dp_{l,r,k}$ 表示当前笛卡尔树的一个子树对应 $[l,r]$，且要求根节点的权值不超过 $k$ 时，子树内可能的笛卡尔树数量，转移即为：
$$
dp_{l,r,k}=\sum_{i=l}^rdp_{l,i-1,\min(k,x_i)}*dp_{i+1,r,\min(k,x_i)-1}
$$
最后可以发现权值大于 $n$ 显然没有更好的效果，可以将大于 $n$ 的 $x_i$ 都变为 $n$。这样 $dp$ 的复杂度即为 $O(n^4)$。

另外一种做法是从下往上贪心给权值，可以通过前缀和实现转移。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105
#define mod 1000000007
int dp[N][N][N],tp[N],n,vis[N][N][N];
int dfs(int l,int r,int v)
{
	if(l>r)return 1;
	if(!v)return 0;
	if(vis[l][r][v])return dp[l][r][v];
	int as=0;
	for(int i=l;i<=r;i++)as=(as+1ll*dfs(l,i-1,min(v,tp[i]))*dfs(i+1,r,min(v,tp[i])-1))%mod;
	vis[l][r][v]=1;return dp[l][r][v]=as;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&tp[i]),tp[i]=tp[i]>n?n:tp[i];
	printf("%d\n",dfs(1,n,n));
}
```



##### ARC105E Keep Graph Disconnected

###### Problem

有一个 $n$ 个点 $m$ 条边的无向图，保证 $1,n$ 不连通。

两人轮流进行操作，每次操作一个人可以选择一条当前不存在于图中的边，满足加入这条边后 $1,n$ 不连通，并加入这条边。不能加入重边自环，不能操作者输。

求双方最优操作下先手获胜还是后手获胜。

多组数据，$n,m\leq 10^5,\sum n,\sum m\leq 2\times 10^5$

$2s,1024MB$

###### Sol

考虑一个连通块中还没有被加入的边，加入这些边不影响连通性，将这些边称作空余边。

如果有两条空余边，则不难发现当前游戏的获胜情况和没有这两条空余边的情况相同，因此只需要考虑空余边数量为 $0,1$ 的情况。



考虑一个人连接了两个连通块的情况，如果两个连通块的点数都是奇数，则连接后空余边数量不变，否则空余边数量改变。因此只需要考虑每个连通块的点数奇偶性。



此时可以发现，如果一个连通块不包含 $1,n$，且点数为偶数。如果此时先手将这个连通块与另外一个连通块连接，如果操作前空余边为 $0$，则后手可以直接操作这次操作后出现的空余边使游戏变回比之前少一个偶数连通块的情况，而后手可能存在更优操作。如果操作前空余边为 $1$，则先手操作空余边得到的状态只比上一种操作得到的状态多一个偶数连通块。

因此可以发现，连接大小为偶数且不包含 $1,n$ 的连通块一定没有用，从而双方都不会在有其它可能操作的时候进行这些操作。如果当前除去 $1,n$ 所在连通块外所有连通块大小都是偶数，则可以发现此时无论如何操作，最后 $1$ 所在连通块和 $n$ 所在连通块的大小奇偶性固定，因此总边数奇偶性固定。从而可以得到在状态中直接删去这样的连通块不会改变胜负情况。



因此此时状态可以被表示为除去 $1,n$ 所在连通块外大小为奇数的连通块个数 $ct$，$1$ 所在连通块的点数奇偶性 $v_0$，$n$ 所在连通块的点数奇偶性 $v_1$ 以及当前空余边数 $s$。这样的状态有 $O(n)$ 个，转移有 $O(1)$ 种，可以求出所有状态的胜负情况。最后对给定图求出这些信息即可。

复杂度 $O(n+m)$

实际上这个 $dp$ 对于 $ct$ 有长度为 $4$ 的循环节，从而可以发现类似题解做法的分类讨论也是可行的。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105000
int T,n,m,a,b,fa[N],sz[N];
int dp[N][2][2][2];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void solve()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)fa[i]=i,sz[i]=1;
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&a,&b);
		a=finds(a);b=finds(b);
		if(a!=b)fa[b]=a,sz[a]+=sz[b];
	}
	for(int f=0;f<2;f++)for(int t=0;t<2;t++)dp[0][f][t][1]=1;
	for(int i=1;i<=n;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++)for(int s=0;s<2;s++)
	{
		int fg=0;
		if(i>=2&&!dp[i-2][j][k][s])fg=1;
		if(!dp[i-1][j^1][k][s^(!j)])fg=1;
		if(!dp[i-1][j][k^1][s^(!k)])fg=1;
		if(s&&!dp[i][j][k][0])fg=1;
		dp[i][j][k][s]=fg;
	}
	int su=0,s1=sz[finds(1)]&1,s2=sz[finds(n)]&1,ls=m&1;
	for(int i=1;i<=n;i++)if(finds(i)==i)
	{
		ls^=(1ll*sz[i]*(sz[i]-1)/2)&1;
		if((sz[i]&1)&&finds(1)!=i&&finds(n)!=i)su++;
	}
	printf("%s\n",dp[su][s1][s2][ls]?"First":"Second");
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```



##### ARC105F Lights Out on Connected Graph

###### Problem

给一张 $n$ 个点 $m$ 条边的简单连通图 $G$，现在可以任意删去 $G$ 的一些边，求 $2^m$ 种方案中有多少种方案得到的图 $G'$ 满足如下性质：

1. $G'$ 连通。
2. 假设每条边有两种状态 $0,1$，当前 $G'$ 的所有边都在状态 $0$，则可以通过若干次如下操作使得 $G'$ 种的所有边都在状态 $1$。操作为选择一个点，翻转所有与这个点相邻的边的状态。

答案模 $998244353$。

$n\leq 17$

$3s,1024MB$

###### Sol

考虑第二个限制，显然每个点操作的次数不超过 $1$。而从每条边角度考虑，可以得到一条边的两个端点操作次数之和必须为 $1$。

因此构造每个点操作次数的过程相当于二分图染色，存在解当且仅当图是二分图。



但二分图计数较为困难，考虑计数图的合法二分图染色数量总和。每个连通二分图都有两种染色，因此可以通过这个值得到答案。

考虑对连通条件容斥。这个数量满足对于两个不连通的部分，两部分的并的染色数量为两部分染色数量的乘积，因此设 $f_S$ 表示点集 $S$ 中的所有连通子图的合法染色数量，$g_S$ 表示点集中所有子图的合法染色数量，则有：
$$
f_S=g_S-\sum_{T\subset S,x\in T}f_t*g_{S-T}
$$
其中 $x$ 为任意一个属于 $S$ 的元素。

则只需要求出 $g$ 即可 $O(3^n)$ 求出 $f$。



如果直接枚举 $S$，枚举 $S$ 的染色情况再考虑每条边是否能加入，则复杂度为 $O(m3^n)$，无法通过。

以下为降智做法：

考虑再次容斥，枚举一些边钦定这些边两侧点颜色相同，剩余边没有限制。

由于 $g$ 中需要求点集的所有子图的染色数量，因此此时选中一条边有 $-1$ 的系数，不选一条边有 $2$ 的系数（在子图中没有选或者不在子图中）。可以看成选一条边有 $-\frac12$ 的系数，不选有 $1$ 的系数，最后再乘上 $2^{su}$ 即为对应的 $g$，其中 $su$ 为 $S$ 的导出子图边数。这样就可以在容斥中只考虑选中边的影响。

此时容斥一条边为钦定两侧点颜色相同，因此最后一个容斥边的连通块内颜色相同。考虑先求出对于一个连通块，这个连通块出现的所有情况中，连通块内的系数乘积和。即这个点集的所有连通子图的 $(-\frac12)^s$，其中 $s$ 为子图边数。可以发现这个容斥和上一个容斥形式类似，可以 $O(3^n)$ 求出。

然后考虑求 $g$，在选择了若干条边后，图会变为若干个连通块，枚举所有连通块的情况求和即为答案。即求出：
$$
h_S=\sum_{T_1,\cdots,T_k}[\forall i<j,T_i\cap T_j=\emptyset][\cup_i T_i=S]\prod v_{T_i}
$$
其中 $v$ 为上一步求出的系数。

这也可以通过固定一个点，枚举该点所在的连通块转移。复杂度 $O(3^n)$。



而 $h$ 即为 $g$ 中容斥的结果，因此使用 $h$ 得到 $g$ 再得到 $f$ 即可得到答案。

复杂度 $O(m2^n+3^n)$



以下为非降智做法：

考虑上面的最后一步，枚举 $S$ 中染色情况，则相当于求一侧在 $T$ 中，一侧在 $S-T$ 中的边数。可以发现，设 $d_S$ 表示两端在 $S$ 中的边数，则上述问题答案为 $d_S-d_T-d_{S-T}$。

因此fwt求出 $d$ 即可，复杂度 $O(n2^n+3^n)$ ~~code鸽子了~~

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 132001
#define mod 998244353
int n,m,a,b,s[N],vl[N],su[N],f[N],g[N],h[N];
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),s[i]=(1<<a-1)|(1<<b-1);
	for(int i=0;i<1<<n;i++)
	{
		vl[i]=1;
		for(int j=1;j<=m;j++)if((i&s[j])==s[j])vl[i]=1ll*(mod+1)/2*vl[i]%mod;
	}
	for(int i=0;i<1<<n;i++)
	{
		f[i]=vl[i];
		int lc=i&-i;
		for(int j=(i-lc);j;j=(j-1)&(i-lc))f[i]=(f[i]+mod-1ll*vl[j]*f[i^j]%mod)%mod;
	}
	for(int i=0;i<1<<n;i++)
	{
		g[i]=2*f[i]%mod;
		int lc=i&-i;
		for(int j=(i-lc);j;j=(j-1)&(i-lc))g[i]=(g[i]+2ll*f[i^j]*g[j])%mod;
	}
	for(int i=0;i<1<<n;i++)
	for(int j=1;j<=m;j++)if((i&s[j])==s[j])g[i]=2*g[i]%mod;
	for(int i=0;i<1<<n;i++)
	{
		h[i]=g[i];
		int lc=i&-i;
		for(int j=(i-lc);j;j=(j-1)&(i-lc))h[i]=(h[i]+mod-1ll*h[i^j]*g[j]%mod)%mod;
	}
	printf("%d\n",1ll*(mod+1)/2*h[(1<<n)-1]%mod);
}
```



##### ARC106E Medals

###### Problem

有 $n$ 个人，每个人有一个权值 $a_i$，第 $i$ 个人会如下行动：

从第 $1$ 天开始计算，连续工作 $a_i$ 天，然后休息 $a_i$ 天，重复这个过程。

在每一天，你可以从这一天在工作的人中选择一个人，并给他一个奖牌。如果这一天没有人工作，则你不会给出奖牌。

给定 $k$，求最小的 $m$，使得存在一种给奖牌的方式，使得 $m$ 天后每个人都拿到了至少 $k$ 个奖牌。

$n\leq 18,k,a_i\leq 10^5$

$3s,1024MB$

###### Sol

显然答案有可二分性，考虑二分答案，变为判定 $m$ 天内是否可以达到要求的问题。

如果将每个人拆成 $k$ 个点，每个点接受一个奖牌，每一天看成一个点，向来工作的人对应的那些 $k$ 个点连边，则问题可以看成是否存在一种匹配方式使得右侧 $n*k$ 个点都有匹配。因此由Hall定理可以得到有解的充分必要条件为：

对于所有人的每一个子集 $S$，$m$ 天中这个子集中至少有一个人工作的天数大于等于 $k*|S|$。

可以发现，如果 $m>2nk$，则每个人都至少工作了 $nk$ 天，上述条件一定得到满足，因此天数上界很小。在判定时，可以先枚举每一天，求出对于每一个人的子集 $S$，前 $m$ 天中来工作的人正好为集合 $S$ 的方案数。每一天的情况可以预处理。

考虑上面的条件，只需要对于每个 $S$ 求出来工作的人中不包含 $S$ 的人，即来工作的人是 $S$ 的补集的子集的天数。因此做一种高维前缀和即可。

复杂度 $O(n^2k+n(k+2^n)\log nk)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 263001
#define M 4059184
int n,k,a,fg[M],su[N],ct[N];
bool check(int l)
{
	for(int i=0;i<1<<n;i++)su[i]=0,ct[i]=ct[i>>1]+(i&1);
	for(int i=1;i<=l;i++)su[((1<<n)-1)^fg[i]]++;
	for(int i=2;i<=1<<n;i<<=1)
	for(int j=0;j<1<<n;j+=i)
	for(int k=j;k<j+(i>>1);k++)su[k]+=su[k+(i>>1)];
	for(int i=0;i<1<<n;i++)if(l-su[i]<ct[i]*k)return 0;
	return 1;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&a);
		for(int j=1;j<=n*k*2;j++)if((j-1)%(a*2)<a)fg[j]|=1<<i-1;
	}
	int lb=1,rb=n*k*2,as=rb;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%d\n",as);
}
```



##### ARC106F Figures

###### Problem

有 $n$ 个点，第 $i$ 个点有 $d_i$ 个连接位置，这些连接位置可以区分。

你可以在两个点间进行连边，具体来说，连边使用如下方式：

在两个点上分别选择一个连接位置，在两个连接位置间连边。每个连接位置在所有操作中最多被使用一次。

你希望将所有点连成一棵树，求方案数，模 $998244353$

$n\leq 2\times 10^5,1\leq d_i<998244353$

$2s,1024MB$

###### Sol

考虑树的prufer序，设点 $i$ 在prufer序中出现了 $c_i$ 次，则该点的度数为 $c_i+1$，因此该点选择连接位置的方案数为 $\prod_{k=0}^{c_i}(d_i-k)$。

可以发现 $\prod_{i=1}^nd_i$ 一定会出现，提出这部分贡献后，可以发现如果一个 $i$ 在 prufer序中出现了 $k$ 次，则贡献系数为在 $d_i-1$ 个数中依次选出 $k$ 个不同的数的方案数。

又注意到prufer序即为按照任意方式选择 $n-2$ 个数，结合上面的贡献系数，可以得到所有prufer序的贡献和为从 $\sum_i(d_i-1)$ 个数中依次选出 $n-2$ 个不同数的方案数。如果将选在前 $d_1-1$ 个位置的看成prufer序为 $1$ 的，接下来 $d_2-1$ 个位置的看成prufer序为 $2$ 的，…… 则容易发现两种方式的一一对应性。

因此记 $s=\sum_i(d_i-1)$，答案即为 $\prod_id_i*\prod_{i=1}^{n-2}(s-i+1)$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int n,a,as=1,su;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&a),as=1ll*as*a%mod,su=(su+a-1)%mod;
	for(int i=1;i<=n-2;i++)as=1ll*as*su%mod,su--;
	printf("%d\n",as);
}
```



##### ARC107E Mex Mat

###### Problem

有一个 $n\times n$ 的矩阵 $a$，每个元素都是 $[0,2]$ 间的整数。

给出矩阵的第一行第一列，对于剩余的位置 $(i,j)$，有 $a_{i,j}=mex(a_{i-1,j},a_{i,j-1})$。

求出整个矩阵中 $0,1,2$ 出现的次数。

$n\leq 5\times 10^5$

$2s,1024MB$

###### Sol

按横向处理每一行难以维护，按斜向处理每一行可以发现一些规律，但实现较为复杂。

观察一些 $n,m$ 较小的情况，可以发现除去开始的几行几列，剩余部分的每条左上到右下的对角线上元素相等。

分类讨论/暴力验证可以发现，对于这样构造的矩阵，一定有 $a_{4,4}=a_{5,5}$，因此求出前四行四列后，剩余部分可以直接被第四行第四列的值推出。模拟前四行四列的过程即可。

复杂度 $O(n)$

注意 $n<4$ 的情况。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505050
#define ll long long
int n,v1[N],v2[N],f[5][N],s[3][3]={1,2,1,2,0,0,1,0,0};
ll as[4];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v1[i]);
	v2[1]=v1[1];for(int i=2;i<=n;i++)scanf("%d",&v2[i]);
	if(n<4)
	{
		for(int i=1;i<=n;i++)f[1][i]=v1[i];
		for(int i=2;i<=n;i++)
		{
			f[i][1]=v2[i];
			for(int j=2;j<=n;j++)f[i][j]=s[f[i-1][j]][f[i][j-1]];
		}
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)as[f[i][j]]++;
		for(int i=0;i<3;i++)printf("%lld ",as[i]);
		return 0;
	}
	for(int i=1;i<=n;i++)f[1][i]=v1[i];
	for(int i=2;i<=4;i++)
	{
		f[i][1]=v2[i];
		for(int j=2;j<=n;j++)f[i][j]=s[f[i-1][j]][f[i][j-1]];
	}
	for(int i=1;i<=4;i++)for(int j=1;j<=n;j++)as[f[i][j]]++;
	for(int i=4;i<=n;i++)as[f[4][i]]+=n-i;
	for(int i=1;i<=n;i++)f[1][i]=v2[i];
	for(int i=2;i<=4;i++)
	{
		f[i][1]=v1[i];
		for(int j=2;j<=n;j++)f[i][j]=s[f[i-1][j]][f[i][j-1]];
	}
	for(int i=1;i<=4;i++)for(int j=5;j<=n;j++)as[f[i][j]]++;
	for(int i=5;i<=n;i++)as[f[4][i]]+=n-i;
	for(int i=0;i<3;i++)printf("%lld ",as[i]);
}
```



##### ARC107F Sum of Abs

###### Problem

给一张 $n$ 个点 $m$ 条边的简单图，每个点有权值 $v_i$。

你可以删去任意多个点，删去点 $i$ 的代价为 $c_i$。

在你删去点后，图会变为若干个连通块，对于每一个连通块，你可以得到这个连通块中所有点的点权和的绝对值的分数。

你的总收益为得到的分数之和减去总代价，求最大收益。

$n,m\leq 300,1\leq c_i\leq 10^6,|v_i|\leq 10^6$

$2s,1024MB$

###### Sol

因为求的是最大收益，得到每个连通块的权值和的绝对值可以看成对于每个连通块选择一个属于 $\{-1,1\}$ 的值 $x$，获得 $x$ 乘以连通块内点权和的收益。

而这可以看成给每个没有被删去的点一个属于 $\{-1,1\}$ 的值 $x_i$，最大化这些点的 $\sum x_ib_i$，满足相邻的两个点权值不能不同。

如果将删去的点看作 $x_i=0$，则问题变为，不进行删点，每个点选择权值 $-1,0,1$ 分别有 $-v_i,-c_i,v_i$ 的收益，一条边两侧的点权值差不能超过 $1$，求最大收益。



这可以看成一个最小割模型，对于每个点建一条链 $s\to a_i\to b_i\to t$，割三条边分别代表选 $-1,0,1$，可以将收益整体加一个数避免负流量上限。对于 $(i,j)$ 差不超过 $1$ 的限制，可以发现连边 $a_i\to b_j,a_j\to b_i$，边权为 $+\infty$ 即可描述这种限制。

最后为了保证一条链上只被割一条边，需要加边 $b_i\to a_i$。然后答案即为图的最小割。

复杂度 $O(n^2(n+m))$ ~~但dinic从来没有被卡满然后这东西只跑了9ms~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 616
int n,m,va[N],vb[N],as,a,b;
int su,dis[N],head[N],cnt=1,cur[N];
struct edge{int t,next,v;}ed[N*6];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;
}
bool bfs(int s,int t)
{
	for(int i=1;i<=su;i++)cur[i]=head[i],dis[i]=-1;
	dis[s]=0;
	queue<int> qu;qu.push(s);
	while(!qu.empty())
	{
		int u=qu.front();qu.pop();if(u==t)return 1;
		for(int i=head[u];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]==-1)
		dis[ed[i].t]=dis[u]+1,qu.push(ed[i].t);
	}
	return 0;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)if(dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&va[i]);
	for(int i=1;i<=n;i++)scanf("%d",&vb[i]);
	su=n*2+2;
	for(int i=1;i<=n;i++)if(vb[i]>=0)
	{
		as+=vb[i];
		adde(su-1,i,vb[i]*2);adde(i,n+i,vb[i]+va[i]);
	}
	else
	{
		vb[i]*=-1;as+=vb[i];
		adde(n+i,su,vb[i]*2);adde(i,n+i,vb[i]+va[i]);
	}
	for(int i=1;i<=n;i++)adde(n+i,i,1e9);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a+n,b,1e9),adde(b+n,a,1e9);
	while(bfs(su-1,su))as-=dfs(su-1,su,1e9);
	printf("%d\n",as);
}
```



##### ARC108E Random IS

###### Problem

有一个长度为 $n$ 的排列 $p$，考虑使用如下方式选出一个 $p$ 的上升子序列：

设选出的元素集合为 $S$，初始 $S$ 不包含任何元素，然后进行如下操作：

考虑当前所有不在 $S$ 中且在选择 $S$ 的基础上再选择该元素得到的仍然是一个上升子序列的位置。设这样的位置有 $k$ 个，如果 $k=0$，则结束这个过程，否则从 $k$ 个位置中随机选择一个加入 $S$。

求操作结束时，选出的序列长度的期望，模 $10^9+7$

$n\leq 2000$

$3s,1024MB$

###### Sol

可以发现，选择一个位置 $i$ 后，位置 $i$ 左右两侧的选择不会互相影响，即两侧的情况独立。

因此如果当前选择了若干个位置，则分出的每一段区间之间都是独立的。

考虑设 $f_{l,r}$ 表示当前选择了位置 $l,r$，中间没有选择时，$[l+1,r-1]$ 部分选择的数量的期望。考虑在排列两侧加入 $p_0=0,p_{n+1}=n+1$，选中这两个位置不会造成影响，从而答案即为 $f_{0,n+1}$。

考虑转移 $f$，设 $[l+1,r-1]$ 部分中有 $k$ 个数在 $[p_l,p_r]$ 之间。如果 $k=0$，则 $f_{l,r}=0$，否则有：
$$
f_{l,r}=1+\frac1k\sum_{l<i<r,p_l<p_i<p_r}(f_{l,i}+f_{i,r})
$$
直接做的复杂度为 $O(n^3)$，~~好像能过~~。考虑分开计算 $f_{l,i}$ 部分的和以及后面部分的和。此时相当于固定 $l$，求所有满足 $l<i<r,p_l<p_i<p_r$ 的位置的 $f_{l,i}$ 之和。

注意到按照区间dp的顺序，在计算到 $f_{l,r}$ 的时候没有计算过 $r-l$ 更大的区间，而计算了所有 $r-l$ 更小的区间，因此当前已经计算的 $f_{l,x}$ 都满足 $l<x<r$，因此可以不考虑这个限制。

对于剩下的一个限制，可以发现只需要对于每一个 $l$ 维护一个树状数组即可求和。 $r$ 部分同理可以转移。

复杂度 $O(n^2\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2053
#define mod 1000000007
int n,v[N],dp[N][N],su[N][N],inv[N];
struct BIT{
	int tr[N];
	void add(int x,int k){for(int i=x+1;i<=n+2;i+=i&-i)tr[i]=(tr[i]+k)%mod;}
	int que(int x){int as=0;for(int i=x+1;i;i-=i&-i)as=(as+tr[i])%mod;return as;}
}sl[N],sr[N];
int calc(int l,int r){return su[r][v[r]]-su[r][v[l]]-su[l][v[r]]+su[l][v[l]]-1;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)for(int j=1;j<=i;j++)if((1ll*mod*j+1)%i==0)inv[i]=(1ll*mod*j+1)/i;
	v[n+1]=n+1;
	for(int i=1;i<=n+1;i++)
	for(int j=1;j<=n+1;j++)su[i][j]=su[i-1][j]+(v[i]<=j);
	for(int l=2;l<=n+1;l++)
	for(int i=0;i+l<=n+1;i++)
	{
		int j=i+l,su=calc(i,j);
		if(!su||v[i]>v[j])continue;
		dp[i][j]=(inv[su]*(mod*3ll+sl[i].que(v[j])-sl[i].que(v[i])+sr[j].que(v[j])-sr[j].que(v[i]))+1)%mod;
		sl[i].add(v[j],dp[i][j]);
		sr[j].add(v[i],dp[i][j]);
	}
	printf("%d\n",dp[0][n+1]);
}
```



##### ARC108F Paint Tree

###### Problem

给一棵 $n$ 个点的树，你需要将每个点染成白色或者黑色中的一种颜色。

定义一种染色方式的权值为所有同色点对中两点距离的最大值。求 $2^n$ 种染色方案的权值和。答案模 $10^9+7$

$n\leq 2\times 10^5$

$2s,1024MB$

###### Sol

设 $(x,y)$ 是树的一条直径。如果 $(x,y)$ 被染成了相同颜色，则权值显然为直径长度，这种情况可以直接计算。



考虑 $(x,y)$ 颜色不同的情况，可以设 $x$ 为白色，$y$ 为黑色，求出这些情况的答案再乘二就是两种情况的答案之和。

此时有如下性质：

如果点 $a,b$ 为任意点，$x$ 为直径的一个端点，则 $dis(a,x),dis(b,x)$ 中一定有一个大于等于 $dis(a,b)$。

如果结论不成立，此时如果 $a,b,x$ 在一条链上，则只能 $x$ 在 $a,b$ 之间，但这样 $x$ 不可能是直径端点。如果 $a,b,x$ 不在一条链上，则存在一个点 $c$，使得 $c$ 到 $a,b,x$ 的路径不重复经过边。此时如果上述条件不满足，则一定有 $dis(c,x)<dis(c,a),dis(c,b)$。设直径为 $(x,y)$，则无论路径 $(x,y)$ 离开 $a,b,x$ 间路径的点在 $(c,a)$ 路径上还是 $(c,b)$ 路径上，$y$ 到另外一个点的距离一定严格大于到 $x$ 的距离，因此矛盾。所以结论成立。



因此对于白色节点，它们中两两距离的最大值一定等于 $x$ 到它们中一个点的距离的最大值。从而只需要考虑从 $x$ 出发的距离。另外一种颜色同理。

因此对于剩余节点，设 $dx_i$ 表示 $i$ 到 $x$ 的距离，$dy_i$ 表示 $i$ 到 $y$ 的距离，则一种染色方式的权值为白色点的 $dx$ 以及黑色点的 $dy$ 中的最大值。

考虑求最大值小于等于 $k$ 的方案数，显然存在方案当且仅当不存在点满足 $\min(dx_i,dy_i)>k$，此时方案数为 $2^s$，其中 $s$ 为满足 $\max(dx_i,dy_i)\leq k$ 的点数量。可以预处理再前缀和后 $O(n)$ 求出。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 205000
#define mod 1000000007
int n,a,b,head[N],cnt,dep[N],tp[N],ct[N],mn,mx,f2[N],as;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	dep[ed[i].t]=dep[u]+1,dfs(ed[i].t,u);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);int fr=1;for(int i=1;i<=n;i++)if(dep[i]>dep[fr])fr=i;
	dep[fr]=0;dfs(fr,0);
	for(int i=1;i<=n;i++)
	{
		if(dep[i]>dep[fr])fr=i;
		tp[i]=dep[i];
	}
	dep[fr]=0;dfs(fr,0);
	for(int i=1;i<=n;i++)
	{
		if(dep[i]>tp[i])dep[i]^=tp[i]^=dep[i]^=tp[i];
		if(dep[i])mn=mn<dep[i]?dep[i]:mn,ct[tp[i]]++;
		else mx=tp[i];
	}
	f2[0]=1;for(int i=1;i<=n;i++)f2[i]=2*f2[i-1]%mod;
	as=1ll*mx*f2[n-1]%mod;as=(as+2*mn)%mod;
	for(int i=1;i<=mx;i++)as=(as+2ll*(i<mn?mn:i)*(f2[ct[i]+ct[i-1]]+mod-f2[ct[i-1]]))%mod,ct[i]+=ct[i-1];
	printf("%d\n",as);
}
```



##### ARC109E 1D Reversi Builder

###### Problem

给定 $n$，有 $n$ 个格子排成一列。对于给定的 $s$，两个人进行如下游戏：

首先第一个人将每个格子独立随机地染成黑色或者白色，随后第一个人在格子 $s$ 上放一枚棋子，颜色与当前格子颜色相同。

接下来两个人轮流操作，第一个人先手。轮到一个人操作时，如果当前没有空格子，游戏结束，否则当前人可以选择一个与已经放了棋子的格子相邻的空格子，向这个格子里面放棋子，棋子颜色与放入的格子颜色相同。

在放入一枚棋子后，如果存在与这枚棋子颜色相同的棋子，则找到距离它最近的颜色相同的棋子，将两枚棋子中间的棋子全部变为与这两枚棋子相同的颜色。

第一个人希望最大化游戏结束时，黑色棋子的数量，第二个人希望最大化白色棋子的数量。双方都会以最优方式操作。

对于每个 $s$ 求出，从这个 $s$ 开始时，游戏结束时黑色棋子数量的期望。答案模 $998244353$。

$n\leq 2\times 10^5$

$2s,1024MB$

###### Sol

可以发现，在游戏过程中，每种颜色的棋子的位置都构成一段连续的区间。



如果第一个格子和最后一个格子的颜色相同，考虑一种游戏过程，假设放在 $1$ 先于放在 $n$，则放在 $1$ 后，之后的操作不可能改变 $1$ 的颜色，之后棋子的状态一定是 $[1,k]$ 中一段前缀为 $1$ 的颜色，剩余后缀为另外一种颜色。那么在 $n$ 上放棋子后，所有棋子都会变成这种颜色。

因此这种情况下最后的结果固定，这些情况出现的概率为 $\frac12$，且最后全部为黑色和白色的概率相等。



考虑剩余的情况，不妨设 $1$ 是白色，$n$ 是黑色。设 $l$ 为左侧第一个黑色格子，$r$ 为右侧开始第一个白色格子。那么有如下结论：

如果 $l-1$ 比 $r+1$ 先放，则结束时黑色棋子数量为 $n-r$。否则黑色棋子数量为 $n-l+1$。

如果 $l-1$ 比 $r+1$ 先放，则 $(l-1,r)$ 中较后操作的可以让中间整个区间变为白色。之后的操作不会再改变颜色。反过来同理。

因此先手一定尽量向右放，后手尽量向左放。从而如果 $s<\frac{l+r}2$，则后手可以先达到目标，数量为 $n-r$，否则先手可以先达到目标，数量为 $n-l+1$。



此时考虑将一种情况和全部翻转的情况配对计算。翻转后，如果 $s\leq \frac{l+r}2$，则先手可以先到达左侧，数量为 $r$，否则后手先到达，数量为 $l-1$。

此时可以发现，两种情况配对后，如果 $s\neq\frac{l+r}2$，则配对后和为 $n$，否则配对后和为 $n+(r-l+1)$。

结合之前的情况，可以发现如果计算结束时黑色棋子数量期望减去 $\frac n2$ 的值，则除去最后一种情况中的 $r-l+1$ 外，其余部分贡献总和均为 $0$。只需要计算最后一种贡献。



对于一个 $s$，只需要计算 $l+r=2s$ 的情况，对于一个这种情况，它需要满足 $1<l+1<r-1<n$，且 $[1,l+1],[r-1,n]$ 部分颜色固定，可以发现中间部分颜色任意。因此这部分贡献为：
$$
v_s=\sum_{t=1}^{\min(s-2,n-s-2)}2^{2t-1}*(2t+1)
$$
对 $2^{2t-1}*(2t+1)$ 预处理前缀和即可。最后需要除以情况数 $2^n$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 200500
#define mod 998244353
int n,v[N],ir=1;
int main()
{
	scanf("%d",&n);
	v[0]=2;for(int i=1;i<=n;i++)v[i]=4ll*v[i-1]%mod;
	for(int i=0;i<=n;i++)v[i]=1ll*(2*i+3)*v[i]%mod;
	for(int i=1;i<=n;i++)v[i]=(v[i-1]+v[i])%mod;
	for(int i=1;i<=n;i++)ir=1ll*(mod+1)/2*ir%mod;
	for(int i=1;i<=n;i++)
	{
		int as=0,tp=n+1-i>i?i:n-i+1;
		tp-=3;
		if(tp>=0)as=v[tp];
		as=(2ll*as*ir+n)%mod*(mod+1)/2%mod;
		printf("%d\n",as);
	}
}
```



##### ARC109F 1D Kingdom Builder

###### Problem

有无限个格子排成一排，编号为所有的整数。

每个格子有一个颜色，颜色为黑色或者白色。给定 $n$ 并给定格子 $1,2,\cdots,n$ 的颜色 $c_i$。所有 $i\leq 0$ 的格子 $i$ 为白色，所有 $i>n$ 的格子 $i$ 为黑色。

初始时格子上没有棋子，你可以通过如下操作，向格子上放棋子：

1. 选择黑色或者白色中的一种颜色。
2. 如果存在一个这种颜色的空格子与一个已经放了棋子的格子相邻。则向满足这个条件的格子中的一个放入棋子。否则，选择任意一个这种颜色的空格子放入棋子。

给出 $\{1,2,\cdots,n\}$ 中的一个子集 $S$，你需要让编号在 $S$ 中的格子都放上棋子。求你最少需要放的棋子数。

$n\leq 10^5$

$2s,1024MB$

###### Sol

最后放的棋子会构成若干个区间，考虑如果确定了区间，如何判定是否存在合法方案。

如果需要新增加一个区间，则需要向不相邻的位置放棋子，此时所有已经放的棋子的区间两侧必须是另外一种颜色。

而如果每个区间内当前都有棋子，那一直放棋子两侧就可以达到目标。因此如果确定了第一个棋子的位置，则接下来可以看成循环进行如下操作，直到每个最后的区间内部都放了一个棋子：

选择一个颜色 $c$，将当前棋子的区间进行扩张，使得扩张后所有区间的两侧都不是颜色 $c$ 并且扩张后不超出最后的区间，然后放一个颜色为 $c$ 的位置。



可以发现，为了尽量使操作合法，每次扩张都会尽量少地扩展，即扩展到两侧第一个满足条件的位置。

还可以发现，如果一个最后的区间内当前有两个分开的区间，那删去后放的区间只会减少之后的限制。因此每个最后的区间中当前只会存在一个区间。即某次操作向这个区间内放了一个点，之后这个区间内不再放其它点，只有第一次放的点尝试进行扩张。



再考虑每次选择的颜色，将选择相同颜色的操作看成一段操作，如果存在至少三段操作，则考虑最后三段操作，不妨设这三段操作是黑色，白色，黑色。

考虑交换前两段操作的顺序，此时对于白色这一段操作放的区间，它们在这一段结束后可能进行了扩张，接下来只有一段连续的黑色操作。但可以发现一段颜色相同的操作的扩张和一次这种颜色的扩张相同，而交换两段后白色操作之后可以看成一段黑色的操作，因此白色段如果之前能扩张，交换后仍然能扩张。而之前在第一个黑色的操作段中放的区间原来需要黑色-白色-黑色的扩张，现在没有了后两步，因此会变得更优。因此这样交换后不会变差。



从而如果存在合法方案，则存在一种合法方案使得段数不超过 $2$。此时已经可以对可能的状态进行 $dp$，但还有更优秀的性质。

不妨设操作为先黑色后白色，则：

对于最后一次黑色操作之前放的区间，它们经历了两种颜色的扩张，因此最后它们区间内部一定存在两个白色位置，且如果考虑区间两侧的两个位置，则在两个白色位置两侧分别存在一个黑色位置。

对于最后一次黑色操作放的区间，这个区间满足内部存在一个黑色位置，如果考虑上区间两侧的两个位置，则在黑色位置两侧分别存在一个黑色位置。

对于之后除去最后一次操作放的区间，它们满足内部存在一个白色位置，如果考虑上区间两侧的两个位置，则在白色位置两侧分别存在一个黑色位置。

对于最后一次操作的区间，它满足内部存在一个白色位置。

此时可以考虑，先选择最后一次黑色操作放的区间作为初始放的区间，然后全部做白色操作，除去原先最后一次操作的区间外，之前所有的区间都满足内部存在一个白色位置，两侧存在黑色位置，因此它们都可以先放一个白色位置再扩张到两侧都是黑色位置。最后用之前最后的区间结尾即可。

因此如果存在方案，则存在一种方案使得所有操作颜色相同。



考虑枚举操作的颜色，设操作颜色为白色，则有如下性质：

对于初始选的点所在的区间，它需要满足考虑上区间两侧两个位置后，存在两个黑色位置。

对于最后选的点所在的区间，它需要满足区间内部存在一个白色位置。

对于剩余区间，它们需要满足考虑上区间两侧两个位置后，存在一个子序列颜色为黑色-白色-黑色，操作时先放在白色位置，然后扩展到黑色位置内部。



此时限制都类似于区间内部存在某种颜色的子序列，因此可以进行 $dp$。$dp$ 状态为设当前考虑了前 $i$ 个位置，当前上一个位置是否在某一段中，如果在，则这一段属于的类型（第一段/最后一段/中间），这一段要求的子序列已经匹配的长度以及前面是否出现了第一段及最后一段。枚举下一个点选不选转移，如果遇到段开头额外考虑这一段怎么选。

$dp$ 的复杂度为 $O(n)$，每个 $i$ 上可能出现的状态数不超过 $30$，但也可以用多维增加一些不会访问的状态以降低代码复杂度。同时需要注意细节问题。

最后还有一种情况是不进行操作，此时相当于所有放的点构成一段，直接处理即可。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
int n,as=1e9,dp[N][5][3][3],rs[5][2][2][2][2];
char s[N],t[N];
void solve(int fg)
{
	int lb=1,rb=n;
	while(lb<=rb)
	{
		if(t[lb]!='o')lb++;
		else if(t[rb]!='o')rb--;
		else break;
	}
	as=min(as,rb-lb+1);
	for(int i=1;i<=4;i++)for(int j=0;j<2;j++)for(int p=0;p<2;p++)for(int q=0;q<2;q++)for(int f=0;f<2;f++)
	{
		int ti=0,nw=i;
		while(nw<4)
		{
			if((nw&1)&&p)nw++;
			else if((~nw&1)&&q&&!ti)ti=1,nw++;
			else if(((nw&1)^fg^j)&&!ti&&(nw!=2||f))ti=1,nw++;
			else break;
		}
		rs[i][j][p][q][f]=nw;
	}
	for(int i=0;i<=n+1;i++)for(int j=0;j<5;j++)for(int p=0;p<3;p++)for(int q=0;q<3;q++)dp[i][j][p][q]=1e9;
	dp[0][0][0][0]=0;
	for(int i=1;i<=n+1;i++)
	for(int j=0;j<5;j++)for(int p=0;p<3;p++)for(int q=0;q<3;q++)if(dp[i-1][j][p][q]<1e8)for(int f=t[i]=='o';f<2;f++)
	{
		int sv=dp[i-1][j][p][q]+f;
		if(!j&&!f)dp[i][j][p][q]=min(dp[i][j][p][q],sv);
		if(j&&!f)
		{
			if(rs[j][s[i]=='w'][p>>1][q>>1][0]!=4)continue;
			dp[i][0][min(p,1)][min(q,1)]=min(dp[i][0][min(p,1)][min(q,1)],sv);
		}
		if(!j&&f)
		{
			for(int v1=0;v1<=!p;v1++)for(int v2=0;v2<=!q;v2++)if(v1+v2<2)
			{
				int nt=rs[1][s[i-1]=='w'][v1][v2][0];
				nt=rs[nt][s[i]=='w'][v1][v2][1];
				dp[i][nt][p+v1*2][q+v2*2]=min(dp[i][nt][p+v1*2][q+v2*2],sv);
			}
		}
		if(j&&f)
		{
			int nt=rs[j][s[i]=='w'][p>>1][q>>1][1];
			dp[i][nt][p][q]=min(dp[i][nt][p][q],sv);
		}
	}
	as=min(as,dp[n+1][0][1][1]);
}
int main()
{
	scanf("%d%s%s",&n,s+2,t+2);
	s[0]=s[1]='w';s[n+2]=s[n+3]='b';n+=2;
	solve(0);solve(1);
	printf("%d\n",as);
}
```



##### ARC110E Shorten ABC

###### Problem

有一个长度为 $n$ 的，只包含 `ABC` 的字符串 $s$。你可以进行如下操作：

选择 $s$ 中相邻且不同的两个字符，将它们删去然后在原位置加入一个剩下一种字符。

求可以得到的字符串种类数，模 $10^9+7$。

$n\leq 10^6$

$2s,1024MB$

###### Sol

可以发现，对于两个相同的相邻字符（例如 `AA`），如果它们旁边存在一个其它字符，则操作它们两次后的效果相当于删去 `AA`。

从而考虑将两个相邻相同字符看成 `0`，则可以得到一个 `0ABC` 间相加的运算方式。然后可以发现这个运算同构于 $\Z_2^2$ 下的加法运算，即 $0,1,2,3$ 间的异或。



考虑将 `ABC` 分别看成 $1,2,3$。则操作后所有数的异或和不变。在操作结束后，一个字符对应原先的一段区间，由上一条，一段区间如果能操作变为一个字符，则可能得到的字符唯一且为区间的异或和对应的字符。

再考虑一段区间在什么情况下可以变为一个字符，归纳可得区间内字符只要不是全部相同或者只有一个字符就可以做到这一点。

但如果一个区间内字符全部相同且最后异或和不为 $0$，则这个区间由某种字符重复 $2k+1$ 次组成。此时如果 $s$ 中存在其它字符，则考虑将其中的 $2k$ 次向这个方向移动，直到加入这个方向第一个其它字符所在的段。经过这样操作后可以让所有段满足要求。

因此如果整个串所有字符相同，则答案为 $1$。否则可以不考虑多于一个字符时不能全部相同的限制。



此时一个串 $t$ 可能被划分出来当且仅当可以将 $s$ 划分成 $|t|$ 段，$s$ 每一段的异或和对应 $t$ 的对应字符。此时可以贪心划分，每次对 $t$ 的当前字符在 $s$ 中划分尽量短的一段。

那么预处理出前缀异或和，并对于每个位置 $x$ 和每个 $i\in\{0,1,2,3\}$ 求出 $x$ 后面第一个前缀异或和为 $i$ 的位置，即可 $O(1)$ 找到下一次划分的位置。

由于贪心的划分方案唯一，因此可以进行 $dp$，设 $dp_i$ 表示划分到 $i$ 的情况数，枚举下一个字符直接转移。转移时可以同时处理答案。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1000500
#define mod 1000000007
int n,su[N],nt[N][4],dp[N],as,fg=1;
char s[N];
int main()
{
	scanf("%d%s",&n,s+1);
	for(int i=1;i<n;i++)fg&=s[i]==s[i+1];
	if(fg){printf("1\n");return 0;}
	for(int i=1;i<=n;i++)su[i]=su[i-1]^(s[i]-'A'+1);
	for(int i=0;i<4;i++)nt[n][i]=n+1;
	for(int i=n-1;i>=0;i--)
	{
		for(int j=0;j<4;j++)nt[i][j]=nt[i+1][j];
		nt[i][su[i+1]]=i+1;
	}
	dp[0]=1;
	for(int i=0;i<n;i++)
	for(int j=0;j<4;j++)if(j!=su[i])
	{
		int r=nt[i][j];if(r>n)continue;
		dp[r]=(dp[r]+dp[i])%mod;
		if(r==n||su[r]==su[n])as=(as+dp[i])%mod;
	}
	printf("%d\n",as);
}
```



##### ARC110F Esoswap

###### Problem

给一个 $0,1,\cdots,n-1$ 的排列 $p_{0,\cdots,n-1}$，你可以进行如下操作：

选择一个数 $i$，交换 $p_i,p_{(i+p_i)\bmod n}$。

你需要在 $2\times 10^5$ 次操作中还原 $p$。构造任意方案或输出无解。

$n\leq 100$

$2s,1024MB$

###### Sol

构造可以发现一定有解。

我的构造方式：

$n=2$ 特殊处理，对于剩下的情况，考虑用 $1,2$ 将剩余序列扫一遍的效果：

考虑当前相邻四个位置为 $1,2,x,y$，如果 $x<y$，可以依次操作 $1,2$，变为 $x,1,2,y$，相当于跳过 $x$。否则，可以依次操作 $2,1,1$，变为 $y,x,1,2$，相当于进行交换。

这里的交换可以看成在环上进行，那么可以重复这样的操作，每次遍历一遍剩余序列，直到剩余序列有序。可以发现 $n$ 轮一定可以满足要求。

此时序列在环上变为有序，考虑最后一步还原，可以一直操作 $1$ 直到还原。

操作步数不超过 $3n^2$。



一种不需要输入排列的震撼构造方式：

依次考虑 $i=n-1,n-2,\cdots,0$，对于每个 $i$ 输出 $n-1$ 次 $i$。

考虑第一轮操作，如果 $p_{n-1}\neq 0$，则下一步会把 $p_{(p_{n-1}+n-1)\bmod n}$ 换过来，显然每个元素只会被换过来一次，但 $0$ 被换过来后，就不会再交换了。因此这一步操作后，一定有 $p_{n-1}=0$。

考虑之后的某一轮操作，假设前 $k$ 轮后，$p_{n-k}=0,p_{n-k+1}=1,\cdots,p_{n-1}=k-1$，考虑这一轮操作 $n-k-1$。

如果 $p_{n-k-1}\neq k$，则交换会交换到 $p_{0,\cdots,n-k-2}$ 中，此时可以看成从这部分换一个元素过来， $n-k$ 次内一定能交换到 $k$。

如果 $p_{n-k-1}=k$，则下一步会交换的数为 $k$ 和 $k-1$，然后可以发现是 $k-1,k-2$ 交换，……，一直到 $0$ 和 $1$ 交换。这之后最后 $k+1$ 个数即为 $0,1,\cdots,k$，接下来操作 $0$ 不会有变化。因此 $n-1$ 步后一定达到这一状态。

因此这样的构造一定可以将排列还原。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105
#define M 200500
int n,v[N],as[M],ct;
void doit(int x)
{
	int v1=x,v2=(x+v[x]-1)%n+1;
	swap(v[v1],v[v2]);
	as[++ct]=x%n;
}
int main()
{
	scanf("%d",&n);
	for(int i=0;i<n;i++)scanf("%d",&v[i]);
	v[n]=v[0];
	for(int i=1;i<=n;i++)if(v[i]==0)v[i]=n;
	if(n==2)
	{
		if(v[1]!=1)as[++ct]=0;
		printf("%d\n",ct);
		for(int i=1;i<=ct;i++)printf("%d\n",as[i]);
		return 0;
	}
	for(int i=1;i<=n;i++)if(v[i]==1)doit(i);
	for(int i=1;i<=n;i++)if(v[i]==2)doit(i);
	if(v[n-1]==1)doit(n-1);if(v[n]==1)doit(n);
	if(v[n]==2)doit(n);
	int ls=0;
	while(1)
	{
		int nw=2;
		while(nw<n)
		{
			if(nw==n-1||v[(nw+ls)%n+1]<v[(nw+ls+1)%n+1])doit((nw+ls-2)%n+1),doit((nw+ls-2)%n+1),nw++;
			else doit((nw+ls-1)%n+1),doit((nw+ls-2)%n+1),doit((nw+ls-1)%n+1),nw+=2;
		}
		ls=(ls+n-2)%n;
		int fg=1;
		for(int i=1;i<n;i++)if(v[(nw+i-1)%n+1]>v[(nw+i)%n+1])fg=0;
		if(fg)break;
	}
	while(1)
	{
		for(int i=1;i<=n;i++)if(v[i]==1)doit(i);
		int fg=1;
		for(int i=1;i<=n;i++)if(v[i]!=i)fg=0;
		if(fg)break;
	}
	printf("%d\n",ct);
	for(int i=1;i<=ct;i++)printf("%d\n",as[i]);
}
```

```cpp
#include<cstdio>
using namespace std;
int n;
int main()
{
    scanf("%d",&n);printf("%d\n",n*(n-1));
    for(int i=n-1;i>=0;i--)for(int j=n-1;j>=1;j--)printf("%d ",i);
}
```



##### ARC111E Simple Math 3

###### Problem

给定 $a,b,c,d$，求有多少个正整数 $i$ 满足如下条件：

$[a+bi,a+ci]$ 中的所有整数都不是 $d$ 的倍数。

一共有 $T$ 组询问。

$T\leq 10^5,1\leq a<d,0\leq b<c<d,d\leq 10^8$

$2s,1024MB$

###### Sol

如果 $(c-b)i+1\geq d$，即区间长度大于等于 $d$，此时区间内一定有一个 $d$ 的倍数，因而一定不合法。

而如果区间长度小于等于 $d$，则区间内最多有一个 $d$ 的倍数，因此可以考虑计算所有区间长度小于等于 $d$ 的区间中 $d$ 的倍数的个数，即：
$$
\sum_{i=1}^{\lfloor\frac{d-1}{c-b}\rfloor}\lfloor\frac{a+ic}d\rfloor-\lfloor\frac{a-1+ib}{d}\rfloor
$$
$\lfloor\frac{d-1}{c-b}\rfloor$ 减去该值即为答案。然后用类欧即可~~或者直接acl floor_sum~~

复杂度 $O(T\log d)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
int T,a,b,c,d;
ll floorsum(int n,int a,int b,int c)
{
	ll as=0;
	if(b>=c)as+=1ll*n*(b/c),b%=c;
	if(a>=c)as+=1ll*n*(n-1)/2*(a/c),a%=c;
	if(!a)return as;
	int li=(1ll*(n-1)*a+b)/c;
	return as+1ll*li*(n-1)-floorsum(li,c,c-b-1,a);
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%d%d",&a,&b,&c,&d);
		int li=(d-1)/(c-b);
		printf("%d\n",li-floorsum(li,c,a+c,d)+floorsum(li,b,a-1+b,d));
	}
}
```



##### ARC111F Do You like query problems?

###### Problem

考虑如下区间修改及询问问题：

有一个长度为 $n$ 的序列 $a$，初始 $a_i=0$。有一个值为 $ans$，初始 $ans=0$。有 $q$ 次操作，每次操作为以下几种之一：

1. 操作形如 $1\ l\ r\ v$，表示将所有满足 $l\leq i\leq r$ 的 $a_i$ 与 $v$ 取 $\min$。
2. 操作形如 $2\ l\ r\ v$，表示将所有满足 $l\leq i\leq r$ 的 $a_i$ 与 $v$ 取 $\max$。
3. 操作形如 $3\ l\ r$，表示你需要求出 $\sum_{i=l}^ra_i$，然后将 $ans$ 增加这个值。

这里额外给定了一个正整数 $m$，保证操作 $1,2$ 中 $v$ 为 $[0,m-1]$ 间的整数。

在所有操作结束后，你需要求出 $ans$ 的值。

可以发现，对于固定的 $n,m,q$，一共有 $(\frac{n*(n+1)}2*(2m+1))^q$ 种可能的输入。你需要求出所有可能的输入得到的输出的和，答案对 $998244353$ 取模。

$n,m,q\leq 2\times 10^5$

$4s,1024MB$

###### Sol

考虑计算随机操作下，答案的期望，最后将答案乘以输入数量即可。

考虑一个位置对答案的贡献。这个位置只会被覆盖这个位置的操作影响，可以发现需要计算如下值：

在一个位置被操作覆盖了 $k$ 次后，这个位置的值的期望。



考虑一次操作对当前值的影响。设当前值为 $x$，则如果操作为 $1\ v(v\geq x)$ 或者 $2\ v(v\leq x)$ 或者 $3$，则不会改变当前值，否则一定会改变当前值，因此当前值不变的概率为 $\frac{m+2}{2m+1}$。同时可以发现，对于剩余的每个 $y$，$x$ 正好有一种操作可以变到 $y$，因此 $x$ 变为剩余每一种值的概率都是 $\frac 1{2m+1}$。

结合初始时 $a_i=0$，可以得到如下结论：如果当前 $a_i\neq 0$，则 $a_i$ 等于 $1,2,\cdots,m-1$ 的概率相等。因此只需要计算 $a_i\neq 0$ 的概率。



此时所有操作可以分为三类。$1\ 0$ 会让值变为 $0$，$2\ x(x>0)$ 会让值变为非零，剩余的 $m+1$ 个操作不会改变是否非零。$a_i\neq 0$ 当且仅当存在第二类操作且最后一次第二类操作晚于第一类操作。

而这一条件也可以看成，存在前两类操作，且只考虑前两类操作时，最后一次操作为第二类操作。可以发现这样转化后，第一步只考虑一个操作是不是前两类，第二步只考虑前两类操作内部的情况，因此两步独立。第一步的概率为 $1-(\frac{m+1}{2m+1})^k$，显然第二步概率为 $\frac{m-1}m$，因此 $a_i\neq 0$ 的概率即为这两个值的乘积，$a_i$ 的期望为 $a_i\neq 0$ 的概率乘以 $\frac m2$。可以发现这个形式非常简单。



然后考虑一个位置的情况，即考虑 $n=1$ 时的答案。此时可以对于每个 $i$ 考虑如果操作 $i$ 是询问，则它对 $ans$ 的贡献的期望，由期望线性性将这些期望相加即可得到答案。

因此 $n=1$ 时答案为：
$$
\sum_{i=1}^{q}\frac 1{2m+1}*\frac{m-1}2*(1-(\frac{m+1}{2m+1})^{i-1})
$$
可以发现求和部分类似于等比数列求和，因此可以 $O(\log q)$ 求出。



考虑原问题，此时对于位置 $k$，一次操作有 $\frac{2k*(n-k+1)}{n*(n+1)}$ 的概率覆盖这个点。这可以看成在上面的讨论中，有 $1-\frac{2k*(n-k+1)}{n*(n+1)}$ 的概率 $a_i$ 直接不变，剩下的情况 $a_i$ 再进行上面的随机。此时上述结论仍然成立。

而此时出现两类操作的概率都变为之前的 $\frac{2k*(n-k+1)}{n*(n+1)}$ 倍，因此这部分变为 $1-(\frac{2k*(n-k+1)}{n*(n+1)}*\frac{m+1}{2m+1})^{k}$，出现一个询问且覆盖该点的概率也会乘上这个值，因此最后的答案为：
$$
\sum_{k=1}^n(\sum_{i=1}^q\frac{m-1}2*\frac{1}{2m+1}*\frac{2k*(n-k+1)}{n*(n+1)}*(1-\frac{2k*(n-k+1)}{n*(n+1)}*\frac{m+1}{2m+1})^{i-1})
$$
直接计算即可，复杂度 $O(n\log q)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int n,m,k,as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d",&n,&k,&m);
	for(int i=1;i<=n;i++)
	{
		int p1=1ll*i*(n-i+1)%mod*pw(1ll*n*(n+1)/2%mod,mod-2)%mod;
		int pr=1ll*p1*k%mod*pw(2*k+1,mod-2)%mod;
		int sp=(m+1ll*(pw(mod+1-pr,m)-1)*pw(pr,mod-2))%mod;
		as=(as+1ll*(k-1)*(mod+1)/2%mod*sp%mod*pw(2*k+1,mod-2)%mod*p1)%mod;
	}
	printf("%d\n",1ll*as*pw(2*k+1,m)%mod*pw(1ll*n*(n+1)/2%mod,m)%mod);
}
```



##### ARC112E Cigar Box

###### Problem

有一个长度为 $n$ 的排列 $p$，初始排列为 $(1,2,\cdots,n)$。

你需要进行 $k$ 次操作，每次操作为选择排列中的一个数，并选择开头或者结尾中的一个方向，然后将这个数删去，再从选择的方向加入这个数（即放到开头或者结尾）。

给定排列最后的状态，求有多少种操作方式可以使得排列最后为给定状态，答案模 $998244353$。

$n,k\leq 3000$

$2s,1024MB$

###### Sol

如果一个数被操作了多次，可以看成先把这个数删去，然后在最后一次操作时插入这个数，显然这样不改变操作的结果。



如果有一些数没有被操作过，则这些数在最后的排列中构成一段区间，所有操作过的数在区间两侧。

考虑枚举没有被操作的数构成的区间，则一个条件是给定状态中区间内的数递增。考虑计算这种情况下合法的方案数。



设区间左侧有 $l$ 个数，右侧有 $r$ 个数，则每个数最后一次操作时左侧的数必须选择左侧，右侧的数必须选择右侧，之前的操作方向可以任意选择。因此选择方向有 $2^{k-l-r}$ 种方案。显然 $l+r>k$ 不存在合法方案。

然后考虑选择操作的数的方案数，这可以看成需要在 $l+r$ 个数中选择 $k$ 次，每个数至少被选择一次，且对于左侧的数，满足左侧第一个数最后一次被操作晚于左侧第二个数，接下来同理，右侧有类似限制，求满足上述限制的方案数。

但由对称性可以发现，在所有满足每个数至少被选择一次的方案中，如果考虑所有数最后一次被选择的顺序，则所有 $(l+r)!$ 种顺序出现的概率相等。而满足条件的出现顺序有 $C_{l+r}^r$ 个，因此满足上述条件的方案数为所有满足每个数至少被选择一次的方案数乘以 $\frac{1}{l!r!}$。

因此只需要计算在 $m$ 个数中选择 $k$ 次，满足每个数都被选择一次的方案数。可以设 $dp_{m,k}$ 表示这一方案数，转移时枚举下一次选择了一个新的数或者一个之前选择过的数转移。这部分复杂度 $O(nk)$，计算这部分的贡献复杂度为 $O(n(n+k))$。



然后考虑所有数都被操作过的情况，考虑最后一次操作最早的一个数，接下来这个数可以看成固定，这样就变成了上面的情况。

枚举这个数的位置 $i$，则选择方向有 $2^{k-n+1}$ 种方案，每个数至少被选择一次的方案数为 $dp_{n,k}$，而 $n!$ 种排列方式中有 $C_{n-1}^{i-1}$ 种满足条件，因此可以得到这类情况的方案数。

复杂度 $O(n(n+k))$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 3050
#define mod 998244353
int n,k,fr[N],ifr[N],dp[N][N],p[N],as,p2[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&p[i]);
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	dp[0][0]=1;
	for(int i=0;i<=k;i++)for(int j=0;j<=n;j++)
	{
		dp[i+1][j+1]=(dp[i+1][j+1]+dp[i][j])%mod;
		dp[i+1][j]=(dp[i+1][j]+1ll*j*dp[i][j])%mod;
	}
	p2[0]=1;for(int i=1;i<=k;i++)p2[i]=2*p2[i-1]%mod;
	for(int i=1;i<=n;i++)
	for(int j=i;j<=n&&(j==i||p[j]>p[j-1]);j++)if(n-j+i-1>=0&&n-j+i-1<=k)
	as=(as+1ll*dp[k][n-j+i-1]*p2[k-(n-j+i-1)]%mod*fr[n-j+i-1]%mod*ifr[i-1]%mod*ifr[n-j])%mod;
	if(k>=n)for(int i=1;i<=n;i++)as=(as+1ll*dp[k][n]*p2[k-n+1]%mod*fr[n-1]%mod*ifr[i-1]%mod*ifr[n-i])%mod;
	printf("%d\n",as);
}
```



##### ARC112F Die Siedler

###### Problem

有 $n$ 种卡片，编号为 $1,2,\cdots,n$。初始时，你有 $c_i$ 张编号为 $i$ 的卡片，你至少有一张卡片。

有 $m$ 种卡包，第 $i$ 种卡包中有 $s_{i,j}$ 张编号为 $j$ 的卡片。

你可以进行如下操作任意次：

1. 选择一个 $i$，获得一个第 $i$ 种卡包，获得里面的所有卡片，每种卡包可以无限获取。
2. 选择一个 $i$，使用 $2i$ 张编号为 $i$ 的卡片交换一张编号为 $(i\bmod n)+1$ 的卡片。

你希望拥有的卡片数量最少，求出这个最少值。

$n\leq 16,m\leq 50$

$6s,1024MB$

###### Sol

可以发现，在操作结束后，编号为 $i$ 的卡片数量一定少于 $2i$，否则交换后更优。称这些状态为结束状态。

交换卡片的操作类似于进位，因此考虑给编号为 $i$ 的卡片权值 $\prod_{j=1}^{i-1}2j$，则除去交换时选择 $i=n$ 的情况外，交换后卡片权值总和不变。而选择 $i=n$ 会让权值总和减少 $2^nn!-1$。

因此可以发现，交换后权值总和模 $2^nn!-1$ 不变。同时可以发现，对于一种结束状态，它的权值总和一定不超过 $2^nn!-1$。同时因为你至少有一张卡片，而交换后不可能没有卡片，因此不可能达到总和为 $0$ 的情况。

从而每一种权值总和模 $2^nn!-1$ 的结果对应唯一的结束状态，而非结束状态都可以变换使得更优，因此如果确定了权值总和模 $2^nn!-1$ 的结果，就可以知道最后状态的卡片数量。

因此只需要考虑通过加入卡包可以得到的不同的权值总和模 $2^nn!-1$ 的余数。设第 $i$ 个卡包对应的权值为 $v_i$，设 $g=\gcd(2^nn!-1,v_1,\cdots,v_n)$，则设初始状态权值为 $r'$，可以发现余数 $r$ 能被达到当且仅当 $r\equiv r'(\bmod g)$。



考虑根号分治，如果 $\frac{2^nn!-1}g$ 很大，则可以直接枚举所有的 $r$ 判断，复杂度为 $O(\frac{2^nn!-1}g*n)$。

否则，考虑最后的状态，相当于需要选择尽量少的卡片，使得卡片的权值和模 $g$ 与 $r'$ 同余。考虑同余最短路，点 $x$ 表示余数为 $x$ 的状态，它向所有加入一张卡片达到的状态连边，即向所有的 $(x+\prod_{j=1}^{i-1}2j)\bmod g$ 连边。可以发现问题变为找到从 $0$ 到 $g$ 至少经过一条边的最短路长度。如果 $r'\bmod g\neq 0$，直接求最短路即可。否则可以先求出最短路，再枚举选的最后一张卡牌。复杂度 $O(gn)$。



因此总复杂度不超过 $O(n*\sqrt{2^nn!-1})$。但可以发现 $g$ 是 $2^nn!-1$ 的因子，而在 $12<n\leq 16$ 时，$2^nn!-1$ 小于等于 $\sqrt{2^nn!-1}$ 的最大因子甚至不超过 $10^4$，~~n=16时甚至是质数~~。在 $n=12$ 时，$2^nn!-1$ 不超过 $3\times 10^{12}$，因此该算法可以通过。

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 1223344
#define ll long long
int n,k,a,as=1e8,ds[N];
ll v=1,tp[17],v1,v2;
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll getv(){ll as=0;for(int i=1;i<=n;i++)scanf("%d",&a),as=(as+a*tp[i])%v;return as;}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)tp[i]=v,v*=2*i;v--;
	v2=v;
	v1=getv();
	for(int i=1;i<=k;i++)v2=gcd(v2,getv());
	if(v/v2<v2)
	{
		for(int i=0;i<v/v2;i++)
		{
			ll rs=(v1+v2*i)%v,si=0;
			if(!rs)rs=v;
			for(int j=n;j>=1;j--)si+=rs/tp[j],rs%=tp[j];
			if(si<as)as=si;
		}
	}
	else
	{
		for(int i=0;i<v2;i++)ds[i]=1e8;
		queue<int> qu;
		for(int i=1;i<=n;i++)
		{
			tp[i]%=v2;
			ds[tp[i]]=1;qu.push(tp[i]);
		}
		while(!qu.empty())
		{
			int u=qu.front();qu.pop();
			for(int i=1;i<=n;i++)
			{
				int nt=(u+tp[i])%v2;
				if(ds[nt]>1e7){ds[nt]=ds[u]+1;qu.push(nt);}
			}
		}
		as=ds[v1%v2];
	}
	printf("%d\n",as);
}
```



##### ARC113E Rvom and Rsrev

###### Problem

给一个长度为 $n$ 的，只包含 `ab` 的字符串 $s$，你可以进行如下操作：

选择 $s$ 中的两个相同字符，删去这两个字符，并将这两个字符中间的部分翻转。

求进行若干次操作后，可以得到的字典序最大的字符串。

多组数据，$T\leq 2\times 10^5,\sum |s|\leq 2\times 10^5$

$2s,1024MB$

###### Sol

设字符串中有 $c_a,c_b$ 个 `a`，`b`，为了使字典序最大，首先需要使开头的 `b` 尽量多。

考虑 `b` 的数量，可以发现，只要 $s$ 不是由一段 `a` 接着一段 `b` 组成，则最后开头 `b` 的数量至少是 $c_b-2$。

具体来说，如果字符串结尾为 `a`，可以直接将前面的 `a` 配对删除，这样 `b` 的数量即为 $c_b$。否则，根据假设一定存在相邻两个字符为 `ba`，此时操作这个 `b` 和最后一个 `b`，然后做上面的配对即可使开头 `b` 的数量为 $c_b-2$。



考虑数量为 $c_b$ 的情况，这相当于不能操作 `b`，可以发现能达到这种情况当且仅当 $s$ 满足以下条件中的一个：

1. $s$ 的结尾为 `a`。
2. $s$ 中 `a` 的数量为偶数个。

其中第二种情况为删除所有 `a`。因此能做第一种情况一定更优。因此考虑第一种情况，即 $s_n=a$。

此时的目标可以看成通过一些操作使得 $s$ 中的字符 `a` 都移动到结尾，且此时剩余的 `a` 尽量多。



考虑一种贪心方式，对于每一个长度大于等于 $2$ 的段，可以通过一次操作将这一段和结尾段拼接，使结尾段长度增加这一段长度减去 $2$。对于长度为 $1$ 的段，可以配对进行删除。设所有 `a` 段的长度为 $s_1,\cdots,s_l$，则可以发现这样操作后，`a` 的个数为 $2\lfloor\frac{s_l+\sum_{i=1}^{l-1}\max(0,s_i-2)}2\rfloor$。记这个值为 $r(s)$。

另一方面，考虑一次可能的操作，如果这次操作不减少段的数量，则这样操作后 $r(s)$ 不会变大。如果操作使得段数量减少 $1$，此时可以看成将两个长度为 $a,b$ 的段合并，合并后长度为 $a+b-2$。可以发现在保证 $s_l>0$ 的情况下，这样也不会增大 $r(s)$。如果操作使得段数减少 $2$，则一定是删去了两个长度为 $1$ 的段，此时同样不会增大。

因此可以发现，最优操作下最后结尾 `a` 的个数即为 $r(s)$。因此可以 $O(n)$ 求出这种情况的答案。



考虑剩下的情况，即有奇数个 `a`，且结尾是 `b`。由上面的讨论，可以做到开头是 $c_b-2$ 个 `b`，然后是若干个 `a`。但还有几种不操作 `b` 的特殊情况：

1. 如果 $s$ 以 `ab` 结尾，则将前面的 `a` 删去后，字符串为 $c_b-1$ 个 `b` 加上 `ab` 结尾，此时开头 `b` 的数量更优。
2. 如果 $s$ 以 `abb` 结尾，则删去前面的 `a` 后，字符串为 $c_b-2$ 个 `b` 加上 `abb` 结尾，此时结尾部分更优。

对于剩余的情况，可以发现不操作 `b` 无法做到比上面的串更优，因此一定会操作 `b`。且此时 $s$ 的结尾一定有至少三个 `b`。



由于 `b` 只能操作一次，这次操作一定为操作结尾的 `b` 以及中间某个 `ba` 中的 `b`。

考虑 $2\lfloor\frac{\sum_{i=1}^{l}\max(0,s_i-2)}2\rfloor$，可以发现无论怎么操作 `a`，这个值都不会变大。考虑操作了一次 `b` 之后，会有一段移动到结尾，此时的 $s_l+\sum_{i=1}^{l-1}\max(0,s_i-2)$ 最多比之前的 $\sum_{i=1}^{l}\max(0,s_i-2)$ 大 $2$。因此如果存在 `baa`，则在开始时直接操作这一对，就可以达到可能的最大值。

如果不存在这样的情况，则此时字符串开头可能存在一段 `a`，而剩余的每一段 `a` 长度都为 $1$。设第一段长度为 $l$。如果先操作 `b` 再做之前的操作，则最后这一段的长度为 $2\lfloor\frac{l-1}2\rfloor$。但可以发现，因为 $s$ 结尾为 `b`，因此操作 `b` 不会使两段 `a` 合并，从而使某段 `a` 的长度增加。因此不可能出现连续 $l+1$ 个 `a`。这说明直接第一次操作 `b` 是一种最优方式。

因此无论如何，第一次操作可以直接操作 `b`，且优先选择 `baa`，再选择 `ba` 作为左侧操作的 `b` 最优。接下来使用上一种情况的讨论即可。

可以发现上面的特殊情况都可以看成以下两种操作方式之一：

1. 不进行操作。
2. 从开头开始配对删除 `a`。

因此可以比较这两种情况与上面其余讨论的答案的最大值。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 200500
int T,n;
char s[N],t[N],as[N];
void doit()
{
	int fg=0;
	for(int i=1;i<=n;i++)if(t[i]<as[i])return;else if(t[i]>as[i]){fg=1;break;}
	if(fg)for(int i=1;i<=n+1;i++)as[i]=t[i];
}
void solve1()
{
	for(int i=1;i<=n+1;i++)t[i]=s[i];
	doit();
}
void solve2()
{
	int su=0;
	for(int i=1;i<=n;i++)su+=s[i]=='a';
	if(~su&1)su++;
	int ct=0;
	for(int i=1;i<=n;i++)if(s[i]=='b')t[++ct]=s[i];
	else {su--;if(!su)t[++ct]=s[i];}
	t[ct+1]=0;
	doit();
}
void solve3()
{
	if(s[n]=='b')
	{
		int lb=0,rb=n;
		for(int i=1;i<=n-2;i++)if(s[i]=='b'&&s[i+1]=='a'&&s[i+2]=='a'&&!lb)lb=i;
		for(int i=1;i<=n-1;i++)if(s[i]=='b'&&s[i+1]=='a'&&!lb)lb=i;
		if(!lb)return;
		for(int i=lb;i<rb;i++)if(lb+rb-i>i)swap(s[lb+rb-i],s[i]);
		for(int i=lb;i<n;i++)s[i]=s[i+1];
		s[n-1]=s[n]=0;n-=2;
	}
	s[0]='b';
	int sb=0,sa=0,si=0;
	for(int i=1;i<=n;i++)if(s[i]=='a')si++;else sb++;
	for(int i=1;i<=n;i++)if(s[i]=='a'&&s[i-1]=='b')
	{
		int lb=i;while(s[lb+1]=='a')lb++;
		if(lb==n)sa+=lb-i+1;
		else if(lb-i-1>0)sa+=lb-i-1;
	}
	if((si-sa)&1)sa--;
	int ct=0;
	for(int i=1;i<=sb;i++)t[++ct]='b';
	for(int i=1;i<=sa;i++)t[++ct]='a';
	t[ct+1]=0;
	doit();
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%s",s+1);n=strlen(s+1);as[1]=0;
		solve1();solve2();solve3();
		printf("%s\n",as+1);
	}
}
```



##### ARC113F Social Distance

###### Problem

有一个长度为 $n+1$ 的单调递增整数序列 $x_0,\cdots,x_n$。

现在有 $n$ 个人，第 $i$ 个人会随机出现在 $[x_{i-1},x_i]$ 中的一个实数坐标上。

求相邻两个人的距离的最小值的期望，答案模 $998244353$

$n\leq 20,0\leq x_i\leq 10^6$

$4s,1024MB$

###### Sol

考虑对于一个 $v$，计算最小值大于 $v$ 的期望，记这个值为 $f(v)$，则 $\int_{t=0}^{+\infty}f(v)dv$ 即为答案。

对于一个 $v$，问题相当于 $a_i$ 在 $[x_{i-1},x_i]$ 中随机生成，求 $\forall i,a_i+v\leq a_{i+1}$ 的概率。

考虑令 $b_i=a_i-i*v$，则相当于 $b_i$ 在 $[x_{i-1}-iv,x_i-iv]$ 中随机生成，求 $\forall i,b_i\leq b_{i+1}$ 的概率。考虑乘上 $\prod_i (x_i-x_{i-1})$，则这样相当于将 $b_i$ 在 $[x_{i-1}-iv,x_i-iv]$ 出现的概率分布都看成 $1$。

对于这样一个问题，考虑按照所有的边界将可能的取值范围分成 $2n$ 段，段按照大小排序。则每个 $b_i$ 所在的段编号必须单调不降。如果确定了每个 $b_i$ 所在的段，则可以判断是否满足 $b_i$ 的上下界要求。再考虑计算这种情况的贡献，如果有 $k$ 个 $b_i$ 在同一段内，设这一段长度为 $l$，则贡献为 $\frac{l^k}{k!}$，计算所有贡献乘积即可。

因此可以设 $dp_{i,j}$ 表示 $b_i$ 在第 $j$ 段，且 $b_{i+1}$ 不在第 $j$ 段的方案数，转移时枚举 $b_{i+1}$ 所在的段，然后枚举这一段内的元素向右到了多少即可转移。一次转移的复杂度为 $O(n^3)$。



最后需要对于所有 $v$ 求答案，但可以发现上述 $dp$ 中将上下限都看成关于 $v$ 的一次函数仍然可以 $dp$，转移时每次操作为乘一个一次式，因此单次复杂度为 $O(n^4)$。

可以发现 $dp$ 转移出现变化的时刻即为所有上下限的相对大小关系发生改变的时刻。但边界只有 $O(n)$ 个且都是一次函数，因此这样的时刻只有 $O(n^2)$ 个，也就是说可以分成 $O(n^2)$ 段，每一段内的边界相对大小关系不变，从而一段内的概率可以用同一个 $v$ 的多项式表示。

因此对于每一段 $dp$ 出答案的多项式，然后积分即可得到这一段的贡献，最后即可得到答案。

复杂度 $O(n^6)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 22
#define M 750
#define mod 998244353
int n,s[N],ct,lb[N],rb[N],su[N],si[N],sr[N*2][2],inv[N],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct fra{int a,b;}tp[M];
int gcd(int a,int b){return b?gcd(b,a%b):a;}
bool operator <(fra a,fra b){return a.a*b.b<a.b*b.a;}
int dp[N][N*2][N];
void addfrac(int x,int y)
{
	int g=gcd(x,y);
	x/=g;y/=g;
	fra t1=(fra){x,y};
	for(int i=1;i<=ct;i++)if(!(tp[i]<t1)&&!(t1<tp[i]))return;
	for(int i=1;i<=ct+1;i++)if(i==ct+1||t1<tp[i])
	{
		for(int j=ct+1;j>=i;j--)tp[j]=tp[j-1];
		tp[i]=t1;ct++;
		return;
	}
}
pair<double,int> fr[N*2];
int main()
{
	scanf("%d",&n);
	for(int i=0;i<=n;i++)scanf("%d",&s[i]);
	for(int i=1;i<=n;i++)inv[i]=pw(i,mod-2);
	for(int i=0;i<=n;i++)for(int j=i+1;j<=n;j++)
	{
		int sy=s[j]-s[i],sx=j-i;
		if(sx>1)addfrac(sy,sx-1);
		addfrac(sy,sx);addfrac(sy,sx+1);
	}
	tp[0].b=1;
	for(int i=1;i<=ct;i++)
	{
		double sp=(1.0*tp[i-1].a/tp[i-1].b+1.0*tp[i].a/tp[i].b)/2;
		for(int j=1;j<=n;j++)fr[j*2-1]=make_pair(s[j-1]-j*sp,j),fr[j*2]=make_pair(s[j]-j*sp,-j);
		sort(fr+1,fr+n*2+1);
		for(int j=1;j<=n*2;j++)
		{
			int id=fr[j].second;
			if(id>0)lb[id]=j;else rb[-id]=j;
		}
		for(int j=1;j<n*2;j++)
		{
			int sa=0,sb=0,v1=fr[j].second,v2=fr[j+1].second;
			if(v1>0)sa-=s[v1-1],sb+=v1;else sa-=s[-v1],sb+=-v1;
			if(v2>0)sa+=s[v2-1],sb-=v2;else sa+=s[-v2],sb-=-v2;
			if(sa<0)sa+=mod;if(sb<0)sb+=mod;
			sr[j][0]=sa;sr[j][1]=sb;
		}
		for(int j=0;j<=n;j++)for(int k=1;k<=n*2;k++)for(int l=0;l<=n;l++)dp[j][k][l]=0;
		dp[0][0][0]=1;
		for(int j=1;j<=n;j++)
		{
			for(int k=0;k<=n;k++)su[k]=0;
			for(int k=1;k<=n*2;k++)
			{
				for(int l=0;l<=n;l++)su[l]=(su[l]+dp[j-1][k-1][l])%mod;
				for(int l=0;l<=n;l++)si[l]=su[l];
				int v1=sr[k][0],v2=sr[k][1];
				for(int l=j;l<=n;l++)
				{
					if(k<lb[l]||k>=rb[l])break;
					for(int t=l;t>=0;t--)si[t+1]=(si[t+1]+1ll*si[t]*v2%mod*inv[l-j+1])%mod,si[t]=1ll*si[t]*v1%mod*inv[l-j+1]%mod;
					for(int t=0;t<=l;t++)dp[l][k][t]=(dp[l][k][t]+si[t])%mod;
				}
			}
		}
		for(int k=0;k<=n;k++)su[k]=0;
		for(int k=1;k<=n*2;k++)for(int l=0;l<=n;l++)su[l]=(su[l]+dp[n][k][l])%mod;
		int li=1ll*tp[i-1].a*pw(tp[i-1].b,mod-2)%mod,ri=1ll*tp[i].a*pw(tp[i].b,mod-2)%mod;
		for(int k=0;k<=n;k++)as=(as+1ll*su[k]*(mod+pw(ri,k+1)-pw(li,k+1))%mod*pw(k+1,mod-2))%mod;
	}
	for(int i=1;i<=n;i++)as=1ll*as*pw(s[i]-s[i-1],mod-2)%mod;
	printf("%d\n",as);
}
```



##### ARC114E Paper Cutting 2

###### Problem

有一个由 $n$ 行 $m$ 列的方格组成的纸片，纸片上有两个方格 $(x_1,y_1),(x_2,y_2)$ 被染成了黑色。

你会重复进行如下操作：

设当前剩余纸片为 $h$ 行 $w$ 列，则存在 $h-1$ 个横向的线，$w-1$ 个纵向的线，在这些线中随机选择一条，将纸片沿着这条线切开。

在这之后，如果两个黑色格子在不同的纸片上，则结束过程，否则保留黑色格子在的纸片，继续过程。

求你进行操作次数的期望，模 $998244353$

$2s,1024MB$

###### Sol

不妨设 $x_1\leq x_2,y_1\leq y_2$。则可以发现有 $x_2+y_2-x_1-y_1$ 条线可以切开两个黑色方格，这部分内正好会切一条线。

考虑剩余部分的每条线出现的概率，由期望线性性对每条线被切的概率求和即为答案。

操作方式等价于，将所有线随机排列，按照排列顺序考虑每条线，如果这条线当前不存在则跳过，否则切这条线。可以发现这样得到的期望与原问题的期望相同。

考虑一条横向切开了前 $a$ 行和后面行的线，其中 $a<x_1$。考虑计算这条线被切的概率。

对于这样一条线，可以发现考虑到它时它还存在当且仅当 $x_2+y_2-x_1-y_1$ 条线还没有被切，且切开横向 $[a+1,x_1]$ 位置的线（即它与黑色格子构成的矩形中间的线）还没有被切。从排列角度考虑，可以发现这条线的贡献为 $\frac{1}{x_1-a+1+(x_2+y_2-x_1-y_1)}$。

同理可以发现每条线的贡献都是 $\frac 1i$ 的概率，求出逆元即可。

复杂度 $O(n+m)$ 或者 $O((n+m)\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 200400
#define mod 998244353
int n,m,a,b,c,d,as=1;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d%d%d%d",&n,&m,&a,&b,&c,&d);
	if(a>c)a^=c^=a^=c;
	if(b>d)b^=d^=b^=d;
	int su=c-a+d-b;
	for(int i=1;i<a;i++)as=(as+pw(su+i,mod-2))%mod;
	for(int i=1;i<=n-c;i++)as=(as+pw(su+i,mod-2))%mod;
	for(int i=1;i<b;i++)as=(as+pw(su+i,mod-2))%mod;
	for(int i=1;i<=m-d;i++)as=(as+pw(su+i,mod-2))%mod;
	printf("%d\n",as);
}
```



##### ARC114F Permutation Division

###### Problem

给定一个长度为 $n$ 的排列 $p$ 和正整数 $k$，你需要进行如下操作：

1. 将 $p$ 划分成 $k$ 个非空子段 $p_1,\cdots,p_k$。
2. 将这 $k$ 个子段重新排列并拼接得到 $q$。在这一步操作中，你必须选择使得 $q$ 的字典序最大的方案。

求最后能得到的字典序最小的 $q$。

$n\leq 2\times 10^5,1\leq k\leq n$

$2s,1024MB$

###### Sol

考虑第二步的操作方式，由于 $p$ 为排列，划分得到的每一个子段的开头字符两两不同，因此可以发现字典序最大的方案即为将所有子段按照开头字符从大到小排序。

回到原问题，显然重新排列后 $q$ 的字典序不小于 $p$ 的字典序，因此考虑首先最大化 $lcp(p,q)$。

设划分的每一段的开头位置依次为 $1=l_1<l_2<\cdots<l_k$，则如果 $p_{l_1}$ 不是所有开头中最大的，则 $q$ 中开头不会是 $p_{l_1}$，此时 $lcp$ 为 $0$，否则第一段可以匹配，接下来变为后面段的问题。可以发现，找到最大的 $d$ 满足 $p_{l_1}>p_{l_2}>\cdots>p_{l_d}$，且 $p_{l_d}$ 大于 $p_{l_{d+1}},\cdots,p_{l_k}$，则 $lcp$ 为 $l_{d+1}-1$。

考虑如果确定了 $d$ 以及前 $d$ 个划分位置，如何确定是否存在方案以及最大的 $lcp$。此时需要在 $[l_{d}+1,n]$ 中找 $k-d$ 个位置 $x$ 满足 $p_{x}<p_{l_d}$，对于一种找到的方案，$lcp$ 为这些 $x$ 的最小值减一。因此可以发现，存在方案当且仅当 $[l_d+1,n]$ 中存在 $k-d$ 个小于 $p_{l_d}$ 的数，且 $lcp$ 为按位置从右往左排列第 $k-d$ 个这样数的位置减一。



因此可以发现对于固定的 $l_d$，前面的方案中 $d$ 越大越优。考虑前面的部分，相当于对于每个位置 $i$ 求出最大的 $d$，使得存在 $1=l_1<\cdots<l_d=i$，且 $p_{l_1}>\cdots>p_{l_d}$。设这个值为 $dp_i$，则可以发现 $dp_i=1+\max_{j<i,p_j>p_i}dp_j,dp_1=1$，可以树状数组转移这个 $dp$。

再考虑对于每个 $i$ 如何确定 $lcp$，需要对于每个 $i$ 维护所有满足 $p_x<p_i$ 的位置。因此可以按照 $p_x$ 从小到大考虑，使用树状数组维护这些位置，询问为询问这些位置中位置编号的第 $k$ 大值。这些部分复杂度为 $O(n\log n)$。

此时可以对于一个 $i$ 确定 $lcp$ 以及后面需要划分的段数。可以发现，因为选择了最长的 $lcp$，因此后面划分的段的开头一定是后面最小的一些权值。因此可以发现，对于多个 $i$ 得到相同 $lcp$ 的情况，后面需要划分的段数越少的方案更优。

此时可以找到一个最优的方案，并且此时后面划分的方案唯一，因此可以直接求出这种情况得到的 $q$。

有可能不存在满足条件的 $i$，此时可以看成 $lcp=0$，剩余段数为 $k$。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 265000
int n,k,p[N],rp[N],lb=1,rk,dp[N];
struct BIT1{
	int tr[N];
	void add(int x,int v){for(int i=x;i;i-=i&-i)tr[i]=max(tr[i],v);}
	int que(int x){int as=0;for(int i=x;i<=n;i+=i&-i)as=max(as,tr[i]);return as;}
}t1;
struct BIT2{
	int tr[N];
	void add(int x){for(int i=x;i<=(1<<18);i+=i&-i)tr[i]++;}
	int getkth(int k){if(!k)return 0;k--;int as=0;for(int i=18;i>=0;i--)if(tr[as+(1<<i)]<=k)as+=1<<i,k-=tr[as];return as+1;}
}t2;
int main()
{
	scanf("%d%d",&n,&k);rk=k;
	for(int i=1;i<=n;i++)scanf("%d",&p[i]),rp[p[i]]=i;
	for(int i=1;i<=n;i++)
	{
		int as=t1.que(p[i]);
		if(as||i==1)as++;
		dp[i]=as;t1.add(p[i],as);
	}
	for(int i=1;i<=n;i++)t2.add(n+1);
	for(int i=1;i<=n;t2.add(n+1-rp[i]),i++)if(dp[rp[i]]<=k&&dp[rp[i]])
	{
		int rs=n+1-t2.getkth(k-dp[rp[i]]);
		if(rs<=rp[i])continue;
		if(rs>lb||(rs==lb&&rk>k-dp[rp[i]]))lb=rs,rk=k-dp[rp[i]];
	}
	for(int i=1;i<lb;i++)printf("%d ",p[i]);
	int li=0,su=0;
	for(int i=1;i<=n;i++)if(rp[i]>=lb)
	{
		su++;
		if(su==rk){li=i;break;}
	}
	for(int i=li;i>=1;i--)if(rp[i]>=lb)
	{
		int nw=rp[i];
		while(nw<=n&&(nw==rp[i]||p[nw]>li))printf("%d ",p[nw]),nw++;
	}
}
```



##### ARC115E LEQ and NEQ

###### Problem

给出长度为 $n$ 的正整数序列 $x_1,\cdots,x_n$，求有多少个长度为 $n$ 的正整数序列 $a_{1,\cdots,n}$ 满足：

1. $\forall i,a_i\leq x_i$
2. $\forall i,a_i\neq a_{i+1}$

答案模 $998244353$

$n\leq 5\times 10^5,x_i\leq 10^9$

$2s,1024MB$

###### Sol

考虑容斥，即选定若干个 $i$，钦定这些 $i$ 满足 $a_i=a_{i+1}$，这样会划分成若干段，每一段内元素必须相等，因此每一段内的方案数为这一段 $x$ 的最小值。

因此答案为，考虑所有将序列划分成非空子段的方式，设一种方式划分成了 $k$ 段，则贡献为 $(-1)^{n-k}$ 乘上每一段内的最小值的乘积。



提出 $(-1)^n$，设 $dp_i$ 表示划分到 $i$ 的答案，则转移式为：
$$
dp_{i}=\sum_{0\leq j<i}-dp_j*(\min_{k\in[j+1,i]}x_k)
$$
考虑枚举取到 $\min$ 的位置（有多个选择最左侧一个）转移，设位置 $i$ 左侧第一个大于它的位置为 $l_i$，右侧第一个大于等于它的位置为 $r_i$，则 $i$ 位置作为 $\min$ 的转移即为 $[l_i,i-1]$ 到 $[i,r_i-1]$ 的转移。这部分可以单调栈求出。

可以发现，如果按照 $i$ 从小到大依次考虑 $i$ 作为 $\min$ 的转移，则这样转移不会破坏顺序，因此可以按照这种方式转移。由于两侧转移都是一段区间，在计算的过程中同时维护差分和前缀和即可维护转移。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 500500
#define mod 998244353
int n,v[N],lb[N],rb[N],st[N],ct,su[N],si[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)
	{
		while(ct&&v[st[ct]]>v[i])ct--;
		lb[i]=st[ct];st[++ct]=i;
	}
	ct=0;st[0]=n+1;
	for(int i=n;i>=1;i--)
	{
		while(ct&&v[st[ct]]>=v[i])ct--;
		rb[i]=st[ct];st[++ct]=i;
	}
	su[0]=n&1?mod-1:1;
	for(int i=1;i<=n;i++)
	{
		int sv=1ll*(su[i-1]-(lb[i]?su[lb[i]-1]:0)+mod)*v[i]%mod;
		si[i]=(si[i]+mod-sv)%mod;si[rb[i]]=(si[rb[i]]+sv)%mod;
		si[i]=(si[i]+si[i-1])%mod;su[i]=(su[i-1]+si[i])%mod;
	}
	printf("%d\n",si[n]);
}
```



##### ARC115F Migration

###### Problem

给一棵 $n$ 个点的树，点有点权 $v_i$。

有 $k$ 枚棋子，第 $i$ 个棋子初始在点 $s_i$，棋子之间有区别。

你可以进行若干次操作，每次操作将一个棋子沿着一条边移动一步。你需要通过操作，将第 $i$ 个棋子移动到点 $t_i$。

定义一种操作方式的代价为所有时刻中每一枚棋子所在的点的点权和（一个点上有多枚棋子时，点权计算多次）的最大值。

求所有操作方式的最小代价。

$n,k\leq 2000$

$4s,1024MB$

###### Sol

考虑二分答案，即对于一个上界 $m$，判断如果任意时刻点权和不能超过 $m$，则是否能从一个状态到达另外一个状态。

对于当前状态，考虑选择一个棋子，固定其它棋子只移动这个棋子。则可以发现这个棋子不能经过权值大于某个数的点。贪心地想，这个点一定会移动到能移动到的点中点权最小的一个点。这样移动后总权值不会变大，因此之前可以到达的部分一定被现在可以到达的部分包含，从而这样贪心不会少考虑情况。



一个棋子可以移动到的点形式与Kruskal重构树类似，考虑这棵树的Kruskal重构树，则固定其它棋子时，一个棋子可以到达的位置为重构树的一个子树。考虑记录这个棋子可以到达的子树的根 $x_i$，则记重构树中点 $i$ 子树内的权值最小值为 $mn_i$，则在贪心做法下，当前棋子可以移动到的最小权值为 $mn_{x_i}$。

设重构树上点 $i$ 权值为 $v_i$，父亲为 $f_i$，此时上述贪心可以被描述为，如果对于一个棋子 $i$ 满足 $v_{f_{x_i}}$ 加上其余棋子的 $mn_{x_i}$ 不超过上界，则当前棋子可以移动到父亲的子树内，因此将当前棋子的 $x_i$ 变为 $f_{x_i}$。



对于一个初始状态 $s$，使用贪心做法可以得到一个当前状态 $x_{1,\cdots,n}$，此时由贪心做法，可以得到如下性质：

如果一个状态 $t$ 满足存在 $i$ 使得 $t_i$ 不在 $x_i$（在重构树中的）子树内，则 $s$ 不能到达 $t$。

考虑 $s,t$ 使用贪心做法得到的状态 $x,x'$。如果 $x=x'$，则显然 $s,t$ 之间可以到达。考虑剩余的情况。

如果存在 $i$ 使得 $x_i,x_i'$ 间不存在祖先关系，则由上一条，两个状态显然不能到达。否则，对于每一对 $i$，$x_i,x_i'$ 之间都存在祖先关系。

此时如果 $s,t$ 可以到达，则 $x,x'$ 可以到达。不妨设 $x_1'$ 是 $x_1$ 的祖先，考虑 $x,x'$ 之间的路径，则一定存在一个时刻，$x$ 中有一个棋子到了它在 $x$ 中原位置的父亲。考虑第一个这样的时刻，可以发现这步移动与贪心做法的性质矛盾。因此此时 $s,t$ 不能到达。

因此 $s,t$ 可以到达当且仅当 $x=x'$。因此只需要求出 $x,x'$ 即可。



可以发现，因为总权值在变小，因此如果一个棋子的 $x$ 从 $u$ 到达了 $f_u$，则之后每一枚棋子到达 $u$ 时都可以到达 $f_u$。因此可以循环考虑每一枚棋子，考虑将这枚棋子尽量向上移动，则最多考虑 $n$ 轮，一次贪心复杂度为 $O(nk)$。总复杂度 $O(nk\log v)$

上述操作也可以看成并查集合并，如果使用set维护每个棋子还差多少可以走到父亲并合并相同节点上的棋子，则复杂度为 $O(n\log n)$，总复杂度 $O(n\log n\log v)$

最后，通过二分后问题的性质，可以发现如下结论：

考虑两个状态一起贪心，不能操作时增大 $m$（增大到可以操作为止），两个状态相同时停止，则停止时权值为最小权值。

这样复杂度可以做到 $O(nk\log n)$ 或者 $O(n\log n)$

###### Code

如果在这里面把while改成if，则上面的分析是不对的。但事实上也能过，~~因此似乎也可以证明这东西是对的。~~

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 4050
#define ll long long
int n,s[N][3],k,v[N],f[N],mn[N],fa[N],sa[N],sb[N],a,b,ta[N],tb[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
bool chk(ll li)
{
	ll su=0;
	for(int i=1;i<=k;i++)ta[i]=sa[i],su+=v[ta[i]];
	if(su>li)return 0;
	while(1)
	{
		int fg=0;
		for(int i=1;i<=k;i++)while(ta[i]<n*2-1&&su-v[ta[i]]+v[f[ta[i]]]<=li)su+=mn[f[ta[i]]]-mn[ta[i]],ta[i]=f[ta[i]],fg=1;
		if(!fg)break;
	}
	su=0;
	for(int i=1;i<=k;i++)tb[i]=sb[i],su+=v[tb[i]];
	if(su>li)return 0;
	while(1)
	{
		int fg=0;
		for(int i=1;i<=k;i++)while(tb[i]<n*2-1&&su-v[tb[i]]+v[f[tb[i]]]<=li)su+=mn[f[tb[i]]]-mn[tb[i]],tb[i]=f[tb[i]],fg=1;
		if(!fg)break;
	}
	for(int i=1;i<=k;i++)if(ta[i]!=tb[i])return 0;
	return 1;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),fa[i]=i,mn[i]=v[i];
	for(int i=1;i<n;i++)scanf("%d%d",&s[i][0],&s[i][1]),s[i][2]=max(v[s[i][0]],v[s[i][1]]);
	scanf("%d",&k);
	for(int i=1;i<=k;i++)scanf("%d%d",&sa[i],&sb[i]);
	for(int i=1;i<n;i++)
	{
		int fr=0;
		for(int j=1;j<n;j++)if(s[j][2]<s[fr][2]||!fr)fr=j;
		v[i+n]=s[fr][2];s[fr][2]=1e9+1;
		int u=finds(s[fr][0]),v=finds(s[fr][1]);
		fa[u]=fa[v]=fa[i+n]=i+n;f[u]=f[v]=i+n;mn[i+n]=min(mn[u],mn[v]);
	}
	ll lb=1,rb=1e13,as=0;
	while(lb<=rb)
	{
		ll mid=(lb+rb)/2;
		if(chk(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%lld\n",as);
}
```



##### ARC116E Spread of Infomation

###### Problem

给定一棵 $n$ 个点的树。给定正整数 $k$，你需要选择 $k$ 个点，使得每个点到选定点的最小距离的最大值最小（边没有边权）。输出这个最小值。

$n\leq 2\times 10^5$

$3s,1024MB$

###### Sol

考虑二分答案，变为判断给定距离 $d$ 是否合法。

此时可以从下往上进行贪心，考虑每个点，如果当前点不放会导致子树内有一个点不合法则放，否则不放。显然这样贪心最优。

考虑判定的问题，对于一个子树，子树内一点和子树外的距离只和它到根的距离有关，因此只需要记录选中点到达的最小距离和没有被满足的点到它的最远距离。设子树内到它最近的选中点距离为 $a$。如果存在没有满足要求的点，设到它最远的没有满足要求的点距离为 $b$。则显然 $a+b>k$。在最后的方案中这个点到子树外一个点的距离小于等于 $k$，因此子树外这个点到当前根的距离小于 $a$。因此在子树外时这个点可以完全替代距离为 $a$ 的点。

因此如果子树内没有不满足要求的点，则记录最近的选中点距离，否则记录最远的没有被满足的点距离。这样的信息显然可以直接合并，且可以判断是否必须选择当前点。单次复杂度为 $O(n)$。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
int n,k,a,b,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int ct,li;
int dfs(int u,int fa)
{
	int mx=0,mn=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		int vl=dfs(ed[i].t,u);
		mx=max(mx,vl);mn=min(mn,vl);
	}
	if(mx+mn>=1)return mx-1;
	else if(mn==-li||!fa){ct++;return li;}
	else return mn-1;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	int lb=1,rb=n,as=n;
	while(lb<=rb)
	{
		li=(lb+rb)>>1;ct=0;
		dfs(1,0);
		if(ct<=k)as=li,rb=li-1;
		else lb=li+1;
	}
	printf("%d\n",as);
}
```



##### ARC116F Deque Game

###### Problem

有 $n$ 个正整数序列 $A_i$，第 $i$ 个序列的长度为 $l_i$。

两个人进行游戏，两人轮流操作，每个人每次操作可以选择一个当前长度大于 $1$ 的序列，并选择删去序列的开头或结尾。

所有序列长度为 $1$ 时，游戏结束，游戏分数为当前所有序列的值的和。

第一个人希望最大化权值，第二个人希望最小化权值。求双方最优操作下，游戏结束时的分数。

$n\leq 2\times 10^5,\sum l_i\leq 2\times 10^5$

$2s,1024MB$

###### Sol

考虑 $l_i=3$ 的情况。如果只有一个这样的序列，则第一个人先手时分数为 $\min(\max(a_1,a_3),a_2)$，第一个人后手时分数为 $\max(\min(a_1,a_3),a_2)$。可以发现后者更大。

如果所有序列长度为 $3$，则后手可以每次先手操作后操作对应序列，使得分数为 $\min(\max(a_1,a_3),a_2)$ 之和。同时，先手可以选择如下操作策略：如果后手不操作自己刚才操作的序列，则下一步操作后手操作的序列。这样一定会使一些序列分数变为 $\max(\min(a_1,a_3),a_2)$，从而分数只会变大。因此此时游戏分数即为 $\max(\min(a_1,a_3),a_2)$ 之和。



然后考虑所有序列长度为奇数的情况，注意到双方都可以采取镜像对方的操作的策略，可以得到如下结论：

对于长度大于 $3$ 且为奇数的序列，可以只保留中间三个数。

具体来说，后手可以选择每次先手操作后操作对应序列的另外一个方向。同时先手有如下策略：

如果后手选择了上述操作，则先手按照将每个序列看成长度为 $3$ 的情况选择操作。否则，操作后手操作序列的反方向，直到后手进行了上述操作。

和之前一样，这样先手不会使分数变小。因此此时答案即为保留中间三个数的情况。



最后考虑原问题。对于一个长度为奇数的序列，如果一个人操作了这个序列，此时如果 $l=3$，则先手会有劣势，如果 $l\geq 5$，则只要后手操作反方向就不改变权值，而后手可能存在更优操作。因此操作长度为奇数的序列一定不优。因此如果存在长度为偶数的序列，则两人不会操作长度为奇数的序列。因此操作可以看成，双方轮流操作长度为偶数的序列，然后变为上一种情况。

可以发现变为上一种情况时，先手的人确定。因此可以对于每一个长度为偶数的序列求出，这个序列删去左侧或右侧后，在最后的情况中的收益的值。此时相当于有若干对 $(a_i,b_i)$，两人轮流操作，每次可以选择一对数，选择保留一个删去另外一个。显然双方的操作为将 $|a_i-b_i|$ 排序后从大到小选，因此排序后即可求出这部分答案。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,l,s[N],t1[N],t2[N],ct;
ll s1,s2;
int main()
{
	scanf("%d",&n);
	while(n--)
	{
		scanf("%d",&l);
		for(int i=1;i<=l;i++)scanf("%d",&s[i]);
		if(l==1){s1+=s[1];s2+=s[1];continue;}
		if(l==2)
		{
			s1+=min(s[1],s[2]);s2+=min(s[1],s[2]);
			t1[++ct]=max(s[1],s[2])-min(s[1],s[2]);
			t2[ct]=max(s[1],s[2])-min(s[1],s[2]);
			continue;
		}
		if(l&1)
		{
			s1+=min(max(s[l/2],s[l/2+2]),s[l/2+1]);
			s2+=max(min(s[l/2],s[l/2+2]),s[l/2+1]);
		}
		else
		{
			int v1=min(max(s[l/2],s[l/2+2]),s[l/2+1]),v2=min(max(s[l/2-1],s[l/2+1]),s[l/2]);
			s1+=min(v1,v2);t1[++ct]=max(v1,v2)-min(v1,v2);
			v1=max(min(s[l/2],s[l/2+2]),s[l/2+1]),v2=max(min(s[l/2-1],s[l/2+1]),s[l/2]);
			s2+=min(v1,v2);t2[ct]=max(v1,v2)-min(v1,v2);
		}
	}
	if(ct&1)swap(t1,t2),swap(s1,s2);
	sort(t1+1,t1+ct+1);
	for(int i=1;i<=ct;i++)if((ct-i+1)&1)s1+=t1[i];
	printf("%lld\n",s1);
}
```



##### ARC117E Zero-Sum Ranges 2

###### Problem

给定 $n,k$，考虑所有由 $n$ 个 $1$ 以及 $n$ 个 $-1$ 的序列 $a$，求有多少个 $a$ 满足如下条件：

$a$ 中正好有 $k$ 个区间满足和为 $0$。

$n\leq 30,k\leq n^2$

$5s,1024MB$

###### Sol

考虑 $a$ 的前缀和序列 $s$。则 $a$ 中和为 $0$ 的区间数量即为 $\sum_{0\leq i<j\leq 2n}[s_i=s_j]$。因此设 $c_x$ 表示前缀和等于 $s$ 的下标数量，则区间数量为 $\sum_x C_{c_x}^2$。

因此区间数量只和每一种前缀和内部的位置有关，考虑将前缀和看成折线，按照高度从高到低对折线 $dp$。

从折线上选择一个非负整数 $t$，将折线从 $x=t+\frac 12$ 位置切开，考虑上面部分的情况。即设 $dp_{i,j,k}$ 表示，上面部分一共有 $2i$ 段折线，它们在折线中构成了 $j$ 个连续的段，且高度大于等于这个数的位置贡献的区间数量为 $k$ 的方案数。

如果求出了这个值，考虑将折线从前缀和为 $0$ 位置切开，枚举两侧的折线长度和段数，再考虑两侧的段怎么合并，则答案为：
$$
\sum_{c_1=0}^n\sum_{c_2=0}^n\sum_{l_1=0}^n\sum_{s=0}^kdp_{l_1,c_1,s}*dp_{n-l_1,c_2,k-s-\frac{(c_1+c_2+1)(c_1+c_2)}2}*C_{c_1+c_2}^{c_1}
$$
这部分复杂度为 $O(n^5)$。



考虑直接向下转移，枚举 $j$ 段中合并了 $p$ 段，然后加入了 $q$ 个单点段。则状态会从 $(i,j,k)$ 转移到 $(i+j+q-p,j+q-p,i+\frac{(j*2+q-p)(j*2+q-p-1)}2)$，转移系数为 $C_{j-1}^p*C_{j-p+q}^q$。

直接转移复杂度为 $O(n^6)$，但是因为常数原因只需要25ms。

从上面的转移中可以考虑枚举 $q-p$，可以发现对于一个 $q-p$ 只有转移系数区别，而转移系数和显然为 $C_{j*2-1+q-p}^{j-1+q-p}$。因此复杂度可以降至 $O(n^5)$

也可以通过组合方式解释上一个系数。考虑枚举这次转移之后的段数 $s$。注意到每个点度数为 $2$，而这一部分点的总度数只与两侧段数有关，为 $2j+2s$，因此这部分点数固定为 $j+s$。再考虑转移系数，考虑相邻两个点的情况，这两个点间可能是上面的一段，也可能是分开的（都向下连），可以发现第一种情况正好有 $j$ 个，因此转移系数为 $C_{j+s-1}^{s-1}$。这与上一步的结果相同。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 35
#define M 905
#define ll long long
int n,k;
ll c[N][N],dp[N][N][M],as;
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=c[i-1][j]+c[i-1][j-1];
	for(int i=0;i<=n;i++)dp[i][i][i*(i-1)/2]=1;
	for(int i=1;i<n;i++)
	for(int j=1;j<=i;j++)
	for(int l=0;l<=k;l++)if(dp[i][j][l])
	for(int s=1;s<=n-i;s++)
	dp[i+s][s][l+(s+j)*(s+j-1)/2]+=dp[i][j][l]*c[j+s-1][s-1];
	for(int p=0;p<=n;p++)for(int q=0;p+q<=n;q++)
	for(int s=0;s<=n;s++)for(int t=0;t<=k-(p+q)*(p+q+1)/2;t++)
	as+=c[p+q][q]*dp[s][p][t]*dp[n-s][q][k-(p+q)*(p+q+1)/2-t];
	printf("%lld\n",as);
}
```



##### ARC117F Gateau

###### Problem

有一个长度为 $2n$ 的环，你需要向环上每一个位置写一个非负整数 $x_i$。

给定限制 $a_0,\cdots,a_{2n-1}$，你写的数需要满足如下限制：

对于每一个 $i$，$x_i+x_{i+1}+\cdots+x_{(i+n-1)\bmod 2n}\geq a_i$，即环上从这个位置开始的一半中所有数的和大于等于 $a_i$。

求所有合法方案中 $\sum x_i$ 的最小值。

$n\leq 1.5\times 10^5,a_i\leq 5\times 10^8$

$3s,1024MB$

###### Sol

考虑二分答案 $k$，限制总数为 $k$，此时环上一半的限制和对应的另外一半可以一起考虑，即限制 $a_i,a_{i+n}$ 可以放在一起考虑，可以看成 $x_i+\cdots+x_{i+n-1}\in[a_i,k-a_{i+n}]$。

此时问题相当于，你需要填序列 $a_0,\cdots,a_{2n-2}$，对于所有的 $i\in[0,n-1]$，需要满足 $x_i+\cdots+x_{i+n-1}\in[a_i,k-a_{i+n}]$。此时显然需要 $\sum_{i=0}^{2n-2}x_i\leq k$，如果满足这些条件，则让 $a_{2n-1}$ 等于剩下的值即可满足原限制。



记 $s_i=\sum_{j=i}^{n-1}x_i,t_i=\sum_{j=n}^{n+i-1}x_i$，则限制相当于：

1. $s_i,t_i\geq 0,s_i\geq s_{i-1},t_i\leq t_{i+1},t_0=0$
2. $s_i+t_i\in[a_i,k-a_{i+n}]$
3. $s_0+t_{n-1}\leq k$

可以发现上述问题有解当且仅当存在整数序列 $s,t$ 满足如上限制。

考虑对于一个固定的 $s_0$，如何确定是否合法。最后需要 $s_{n-1}\geq 0$ 且 $t_{n-1}$ 尽量小，从贪心的角度考虑，如果 $s_{i-1}+t_{i-1}$ 满足当前区间的要求，则不需要改变 $s,t$。如果 $s_{i-1}+t_{i-1}<a_i$，则只需要增大 $t_{i-1}$（改变 $s$ 一定不优），同理如果大于上界，则只需要减小 $s_{i-1}$。



可以注意到，在这个过程中，如果初始的 $s_0$ 变大，则最后的 $s_{n-1}$ 不会变小。因此判断 $s_{n-1}\geq 0$ 的条件时，可以二分找到最小的合法的 $s_0$。

同时，如果 $s_0$ 增加 $1$，考虑两个状态做贪心的过程，考虑（可能出现的）两个状态变得相同的操作，可以发现最后 $t_{n-1}$ 最多比之前的状态减少 $1$。因此 $s_0+t_{n-1}$ 最小的序列即为对有解的最小 $s_0$ 进行贪心操作的序列。

此时二分找最小的有解 $s_0$ 再判断即可，复杂度 $O(n\log^2 v)$，可以通过



但这里还可以更优。考虑 $s_i+t_i$ 的变化过程，可以发现这相当于每次给一个区间，用最短的移动距离将 $s_i+t_i$ 的值移动到区间内。

考虑这个过程，可以发现存在 $x\leq y$（考虑最大下界和最小上界），使得如果 $s_0\leq x$，则这些情况的贪心过程会通过若干次增加 $t_i$，最后在某次操作后与 $s_0=x$ 的贪心过程的 $s+t$ 序列重合。如果 $s_0\geq y$，则会通过若干次减少 $s_i$，最后与 $s_0=y$ 的序列重合。如果 $x<y$，则 $s_0\in[x,y]$ 时，$s_0$ 每增加 $1$，最后的 $t$ 不变，$s$ 增加 $1$（这一部分不会被上下界限制）。

从而可以发现，在 $s_0$ 从 $0$ 增长到 $+\infty$ 时，在一个前缀内 $s_0$ 每加一 $s_{n-1}$ 就会加一，之后再增长 $s_{n-1}$ 不变。

因此可以先做一次 $s_0=0$ 的情况，然后做 $s_0=-s_{n-1}$ 的情况即可判断。

复杂度 $O(n\log v)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 150050
int n,v[N*2];
bool chk(int li)
{
	for(int i=1;i<=n;i++)
	{
		int mn=v[i],mx=li-v[i+n];
		if(mx<mn)return 0;
	}
	int sl=0,sr=0;
	for(int i=1;i<=n;i++)
	{
		int mn=v[i],mx=li-v[i+n];
		if(sl+sr>mx)sl-=sl+sr-mx;
		if(sl+sr<mn)sr+=mn-sl-sr;
	}
	int as=-sl;
	sl=as,sr=0;
	for(int i=1;i<=n;i++)
	{
		int mn=v[i],mx=li-v[i+n];
		if(sl+sr>mx)sl-=sl+sr-mx;
		if(sl+sr<mn)sr+=mn-sl-sr;
		if(sr+as>li||sl<0)return 0;
	}
	return 1;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n*2;i++)scanf("%d",&v[i]);
	int lb=0,rb=1.01e9,as=0;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(chk(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%d\n",as);
}
```



##### ARC118E Avoid Permutations

###### Problem

对于一个 $n$ 阶排列 $p$，使用如下方式定义它的权值：

有一个 $(n+2)\times(n+2)$ 的网格，行列标号为 $0,1,\cdots,n+1$。其中对于每个 $i\in\{1,2,\cdots,n\}$，位置 $(i,p_i)$ 为障碍。

你需要从 $(0,0)$ 走到 $(n+1,n+1)$，且你只能向右向上走，排列的权值即为走的方案数。

给定序列 $a$，$a_i\in\{-1,1,2,\cdots,n\}$。如果 $a_i\neq -1$，则要求 $p_i=a_i$。求所有满足要求的 $n$ 阶排列的权值和，答案模 $998244353$。

$n\leq 200$

$2s,1024MB$

###### Sol

考虑计算一个排列的权值，除了直接的 $dp$，考虑如下容斥方式：

选择一些位置，钦定路径经过这些位置。

计算所有排列的权值可以看成考虑所有容斥的位置集合，计算所有集合的贡献。



考虑选择的位置 $(i,p_i)$ 构成的集合 $T$，一个 $T$ 可能在某个排列的容斥中出现当且仅当：

1. $T$ 中没有两个位置在同一行或者同一列。
2. $T$ 不与排列的已知部分矛盾。即如果 $a_i\neq 0$，则 $\forall(x,y)\in T$，则 $x=i$ 当且仅当 $y=a_i$。

对于一个 $T$，考虑它的贡献，贡献可以分为三部分：

1. 容斥系数 $(-1)^{|T|}$
2. 有多少个排列的容斥中存在 $T$，即有多少个排列的 $\{(i,p_i)\}$ 包含 $T$。可以发现，对于可能出现的 $T$，设 $T$ 中有 $x$ 个位置 $(i,p_i)$ 满足 $a_i=-1$，总共有 $s$ 个位置满足 $a_i=-1$，则这 $x$ 个位置被固定，剩下的位置任意，因此这部分权值为 $(s-x)!$。
3. $T$ 的贡献方案数，即只向上向右走，经过 $T$ 中所有位置的路径数。



此时再考虑对路径 $dp$，对于一条路径，枚举路径上的一个点集构成 $T$，则需要考虑如下问题：

1. $T$ 需要满足两个要求。对于第一个要求，由于路径只会向上向右，只需要记录当前行列上是否选择了数即可。对于第二个要求，可以发现由 $a$ 可以推出每个位置 $(x,y)$ 是否可以被选入 $T$。
2. 考虑 $T$ 贡献的前两部分（第三部分在枚举路径中计算），第一部分系数可以直接dp中计算，对于第二部分，只需要再记录之前选的 $|T|$ 即可。

因此设 $dp_{x,y,s,0/1,0/1}$ 表示路径走到了 $(x,y)$，当前前面选出了 $s$ 个数，当前行列内是否已经选出了一个数，此时前面所有情况的系数和。最后由 $dp_{n+1,n+1}$ 即可得到答案。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 204
#define mod 998244353
int n,a,is[N][N],fr[N],dp[N][N][N][4],as,ct;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)is[i][j]=1;
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&a);
		if(a!=-1)
		{
			for(int j=1;j<=n;j++)if(j!=i)is[j][a]=0;
			for(int j=1;j<=n;j++)if(j!=a)is[i][j]=0;
			is[i][a]=2;
		}
		else ct++;
	}
	dp[0][0][0][0]=1;
	for(int i=0;i<=n+1;i++)for(int j=0;j<=n+1;j++)
	for(int k=0;k<=n;k++)for(int t=0;t<4;t++)if(dp[i][j][k][t])
	{
		if(is[i][j]&&!t)dp[i][j][k+(is[i][j]==1)][3]=(dp[i][j][k+(is[i][j]==1)][3]+mod-dp[i][j][k][t])%mod;
		dp[i+1][j][k][t&1]=(dp[i+1][j][k][t&1]+dp[i][j][k][t])%mod;
		dp[i][j+1][k][t&2]=(dp[i][j+1][k][t&2]+dp[i][j][k][t])%mod;
	}
	int fr=1;
	for(int i=0;i<=ct;i++,fr=1ll*fr*i%mod)as=(as+1ll*fr*dp[n+1][n+1][ct-i][0])%mod;
	printf("%d\n",as);
}
```



##### ARC118F Growth Rate

###### Problem

给定 $n,m$ 以及正整数序列 $a_{1,\cdots,n}$。求有多少个长度为 $n+1$ 的正整数序列 $x$ 满足如下条件：

1. $1\leq x_i\leq m$
2. $a_ix_i\leq x_{i+1}$

答案模 $998244353$

$n\leq 1000,m\leq 10^{18},\prod a_i\leq m$

$4s,1024MB$

###### Sol

考虑 $dp_{i,a}$ 表示从后往前填到 $i$，满足 $x_i=a$ 的方案数。

则转移为：$dp_{i,a}=\sum_{j\geq a*a_i}dp_{i+1,j}$。

如果 $a_i=1$，则操作相当于后缀和。由幂和相关可知如果 $dp_{i+1}$ 是一个 $n-i$ 次多项式，则 $dp_i$ 是一个 $n-i+1$ 次多项式。

如果 $a_i>1$，则先可以看成先求 $a_i=1$ 的情况，再保留序列中 $a_i$ 的倍数。因此也可以看成多项式。

考虑原问题，$dp_{n+1,v}=[v\leq m]$，因此考虑将 $dp_{i,v}$ 表示为 $[i\leq lim]F(i)$ 的形式，其中 $F$ 为 $n-i+1$ 次多项式。

注意到 $dp_{i,a}=\sum_j dp_{i+1,j}-\sum_{j<a*a_i}dp_{i+1,j}$，前者为固定值，而多项式前缀和得到的多项式也是多项式，因此新的 $dp$ 也可以被表示为这种形式，而 $lim$ 变为 $\frac{lim}{a_i}$ 下取整。

考虑插值维护 $dp$，维护 $dp$ 的点值，则前缀和在点值上可以简单操作。对于 $a_i>1$ 需要求 $x*a_i$ 位置前缀和的部分，可以还原多项式再得到新的点值。还原一次复杂度为 $O(n^2)$，但只需要还原 $O(\log m)$ 次。

复杂度 $O(n^2\log m)$，使用快速插值等操作可以做到 $O(n\log^2n\log m)$



另外一种复杂度相同但是常数过大的做法：~~原因是我不想写插值~~

考虑类似差分的做法，设 $v_i=x_{i}-a_{i-1}x_{i-1}$，记 $a_i$ 的后缀乘积为 $b_i$，则问题相当于求满足如下条件的 $v$ 数量：

1. $v_1\geq 1,v_i\geq 0$
2. $\sum v_ib_i\leq m$

可以先将 $m$ 减去 $b_1$，变成所有数大于等于 $0$ 的限制。

考虑数位 $dp$，考虑所有本质不同的 $b$ 构成的序列 $c_1,\cdots,c_k$，则 $c_1|c_2|\cdots|c_k$。考虑将这些看成每一位。

因此对于 $v_i*c_t$，它可以被拆成 $v_{i,t}*c_t+v_{i,t+1}*c_{i+1}+\cdots+v_{i,k}*c_k$。其中对于 $t<k$ 时，有限制 $v_{i,t}<\frac{c_{t+1}}{c_t}$。这相当于混合进制下的按位表示。

回到上述问题，设 $d_i=\frac{c_i}{c_{i-1}}(c_0=1)$，对于每个 $t$，设有 $s_t$ 个 $v_ib_i$ 拆出的表示包含了 $c_{i,t}*c_t$，则上述问题可以看成求满足如下条件的 $(x_{1,1},\cdots,x_{1,s_1},x_{2,1},\cdots,x_{2,s_2},\cdots,x_{k,1},\cdots,x_{k,s_k})$ 数量：

1. $x\geq 0$
2. $\forall 1\leq t<k,\forall j,x_{i,j}<d_i$
3. $\sum_{i=1}^kc_i*\sum_{j=1}^{s_i}x_{i,j}\leq m-b_1$

此时可以发现如果从低位向高位考虑，则进位数量不超过 $n$，因此可以考虑 $dp_{i,j,0/1}$ 表示填了低的 $i$ 位，后面进位数为 $j$，低位最后的结果与 $m-b_1$ 的低位的大小关系。

此时一个状态向后转移相当于，当前数为 $j$，进行 $s_i$ 次操作，每次给它加上 $[0,d_i-1]$ 间的一个整数。最后根据 $\lfloor\frac j{d_i}\rfloor$ 以及模 $d_i$ 的余数判断接下来的状态。

考虑容斥掉加的数不超过 $d_i-1$ 的条件，设 $v_x$ 表示 $s_i$ 次相加后和为 $x$ 的方案数，则容斥得 $v_x=C_{x+s_i-1}^{s_i-1}+\sum_{t=1}^{s_i}C_{s_i}^t(-1)^tC_{x-d_i*t+s_i-1}^{s_i-1}$，

但注意到容斥时减去的都是 $d_i$ 的倍数，而减去一个 $d_i$ 状态一定从 $(i,j,k)$ 变为 $(i,j-1,k)$。因此可以将容斥放到计算 $dp$ 之后，即先不考虑上界求出 $dp'_{i,j,k}$，则一定有 $dp_{i,j,k}=\sum_{t=0}^{s_i}C_{s_i}^t(-1)^tdp'_{i,j-t-k}$。

考虑没有上界的情况，考虑一种转移，$dp_{i-1,j,0}$ 转移到 $dp_{i,k,0}$ 当且仅当加的数和在 $[k*d_i-j,k*d_i-j+(\lfloor\frac{m}{c_{i+1}}\rfloor\bmod d_i)]$ 之间。而此时加的数和为 $x$ 的方案数为 $C_{x+s_i-1}^{s_i-1}$，对 $x$ 前缀和后仍然为组合数。因此只需要计算 $O(n^2)$ 个系数，即可完成转移。另外四种情况同理。

直接计算组合数不能通过，但考虑对于一个 $k$，计算所有转移过来的 $j$ 的组合数，此时相当于求 $C_{x}^k,C_{x+1}^k,\cdots,C_{x+l}^k$，这相当于对于一个序列，求出每一个长度为 $k$ 的区间乘积，可以前缀和处理解决。因此单次转移复杂度为 $O(n^2)$

因此总复杂度为 $O(n^2\log m)$。但常数极大，需要卡常。

一种优化方式是对于小的 $c_i$ 暴力做而不容斥，对于小的 $c_i$ 复杂度可以fft做到 $O(c_i*n*\log n)$~~避免被59个2针对~~

~~可能这个做法唯一好处是任意模数的时候加上fft能做到n^2logmlogn~~

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1050
#define mod 998244353
#define ll long long
int n,v[N],dp[N][2],rs[N][2],as,c[N][N],ifr[N];
ll su=1,m;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int pr[N*2],ir[N*2],st[N*2];
void calc(ll r,int n,int k)
{
	pr[0]=1;
	for(int i=1;i<=n+k;i++)
	{
		int tp=(r-i+1+mod)%mod;
		pr[i]=pr[i-1];
		if(tp)pr[i]=1ll*pr[i]*tp%mod;
	}
	ir[n]=pw(pr[n],mod-2);
	for(int i=n;i>=1;i--)
	{
		int tp=(r-i+1+mod)%mod;
		ir[i-1]=ir[i];
		if(tp)ir[i-1]=1ll*ir[i-1]*tp%mod;
	}
	int inv=ifr[k];
	for(int i=0;i<=n;i++)
	if((r-i)%mod<k)st[i]=0;
	else st[i]=1ll*pr[i+k]*ir[i]%mod*inv%mod;
}
int main()
{
	scanf("%d%lld",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=0;i<=n+1;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n+1;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	ifr[0]=1;for(int i=1;i<=n+1;i++)ifr[i]=1ll*ifr[i-1]*pw(i,mod-2)%mod;
	dp[0][0]=1;
	for(int i=n;i>=1;i--)if(v[i]>1)
	{
		int li=m/su%v[i];su*=v[i];
		for(int j=0;j<=n;j++)rs[j][0]=rs[j][1]=0;
		int si=n-i+1;
		for(int j=0;j<=n;j++)st[j]=0;
		for(int j=0;j<=si;j++)
		{
			for(int k=0;k<=si;k++)rs[j][0]=(rs[j][0]+mod-1ll*st[k]*(dp[k][0]+dp[k][1])%mod)%mod;
			calc(1ll*v[i]*j+si+li-1,si,si);
			for(int k=0;k<=si;k++)rs[j][0]=(rs[j][0]+1ll*st[k]*dp[k][1])%mod,rs[j][1]=(rs[j][1]+mod-1ll*st[k]*dp[k][1]%mod)%mod;
			calc(1ll*v[i]*j+si+li,si,si);
			for(int k=0;k<=si;k++)rs[j][0]=(rs[j][0]+1ll*st[k]*dp[k][0])%mod,rs[j][1]=(rs[j][1]+mod-1ll*st[k]*dp[k][0]%mod)%mod;
			calc(1ll*v[i]*j+si+v[i]-1,si,si);
			for(int k=0;k<=si;k++)rs[j][1]=(rs[j][1]+1ll*st[k]*(dp[k][0]+dp[k][1]))%mod;
		}
		for(int t=0;t<2;t++)for(int j=si;j>=1;j--)for(int k=0;k<j;k++)
		rs[j][t]=(rs[j][t]+1ll*((j-k)&1?mod-1:1)*c[si][j-k]%mod*rs[k][t])%mod;
		for(int j=0;j<=si;j++)dp[j][0]=rs[j][0],dp[j][1]=rs[j][1];
	}
	for(int i=0;i<=n;i++)for(int j=0;j<2;j++)
	{
		calc(m/su-i-j+n,1,n+1);
		as=(as+1ll*dp[i][j]*st[0])%mod;
	}
	printf("%d\n",as);
}
```



##### ARC119E Pancakes

###### Problem

有一个长度为 $n$ 的序列 $a$。

你可以进行不超过一次操作，操作为选择一段区间，翻转这段区间。

你希望最小化上述过程后，$\sum_{i=1}^{n-1}|a_i-a_{i+1}|$ 的最小值。输出这个最小值。

$n\leq 3\times 10^5$

###### Sol

考虑交换 $(l,r)$ 的情况，如果 $l\neq 1,r\neq n$，则交换后这个值会减少 $|a_l-a_{l-1}|+|a_r-a_{r+1}|-|a_l-a_{r+1}|-|a_r-a_{l-1}|$。对于 $l=1$ 或者 $r=n$ 的情况，这些情况只有 $O(n)$ 种，可以特殊处理。~~此时可以直接ds维护~~



考虑 $a_l,a_{l+1}$ 的大小关系，$a_r,a_{r+1}$ 的大小关系。如果 $a_l\leq a_{l-1},a_{r+1}\leq a_r$，则交换前两对代价为 $a_{l-1}+a_{r-1}-a_l-a_r=(a_{l-1}-a_r)+(a_{r+1}-a_l)$，因此可以发现交换后代价不会变小。同理，如果 $a_l\geq a_{l-1}$ 且 $a_{r+1}\geq a_r$，交换后也不会让代价变小。

因此只需要考虑 $a_{l-1}<a_l$ 且 $a_r<a_{r+1}$ 或者两者全部反过来的情况。



对于第一种情况，可以发现如果 $[a_{l-1},a_l],[a_r,a_{r+1}]$ 两区间有交，则交换后减小的代价为区间交的两倍。如果区间没有交，则交换后代价变大。

因此这种情况下，只需要找到所有满足 $a_{l-1}<a_l$ 的区间 $[a_{l-1},a_l]$，这种情况代价减小的最大值即为这些区间中两个区间的交的最大值的两倍。而另外一种情况同理。

因此问题变为，给出若干个区间，求两两交的最大长度。

考虑将左端点从小到大排序，枚举交的左端点，则两个区间的左端点都需要在这个左端点左侧，此时交右端点为两个区间右端点的最小值，因此记录左侧最大的两个右端点扫过去即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,v[N],ct,l1,l2;
ll su,as;
int Abs(int x){return x>0?x:-x;}
pair<int,int> tp[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)su+=Abs(v[i]-v[i+1]);
	as=su;
	for(int i=1;i<n;i++)as=min(as,su+min(Abs(v[1]-v[i+1]),Abs(v[i]-v[n])-Abs(v[i]-v[i+1])));
	for(int i=1;i<n;i++)if(v[i]<=v[i+1])tp[++ct]=make_pair(v[i],v[i+1]);
	sort(tp+1,tp+ct+1);
	l1=l2=0;
	for(int i=1;i<=ct;i++)
	{
		int rb=tp[i].second;
		if(l1<rb)l2=l1,l1=rb;else if(l2<rb)l2=rb;
		as=min(as,su-2*(l2-tp[i].first));
	}
	l1=l2=ct=0;
	for(int i=1;i<n;i++)if(v[i]>v[i+1])tp[++ct]=make_pair(v[i+1],v[i]);
	sort(tp+1,tp+ct+1);
	for(int i=1;i<=ct;i++)
	{
		int rb=tp[i].second;
		if(l1<rb)l2=l1,l1=rb;else if(l2<rb)l2=rb;
		as=min(as,su-2*(l2-tp[i].first));
	}
	printf("%lld\n",as);
}
```



##### ARC119F AtCoder Express 3

###### Problem

有一排 $n+1$ 个格子，格子被标号为 $0,1,\cdots,n$。

格子 $1,2,\cdots,n-1$ 被染了颜色，颜色为红色或者蓝色。

你当前在格子 $0$，你希望到达格子 $n$。你有如下行动方式：

1. 移动到一个相邻格子。
2. 如果当前格子为红色，则可以选择移动到它左侧第一个红色格子或者移动到它右侧第一个红色格子。
3. 如果当前格子为蓝色，则可以选择移动到它左侧第一个蓝色格子或者移动到右侧第一个蓝色格子。

这里认为格子 $0,n+1$ 同时有红色和蓝色。例如从格子 $0$ 出发操作 $2,3$ 都可以选择，且如果格子 $i$ 右侧 $[i+1,n-1]$ 中没有任何一个红色格子，则认为格子 $i$ 右侧第一个红色格子为 $n$。剩余情况类似。

现在有一些格子被染好了颜色，剩下的格子可以任意染色。给定正整数 $k$，求所有可能的染色方式中，有多少种染色方式满足染色后，你可以经过不超过 $k$ 次行动从 $0$ 到达 $n$。答案对 $10^9+7$ 取模。

$n\leq 4000$

$4s,1024MB$

###### Sol

考虑从 $0$ 到 $n$ 路径的性质。如果相邻两个位置 $(i,i+1)$ 满足颜色不同，则可以发现不存在从左侧到右侧不经过这两个位置的方案，因此最短路必定经过这两个点。将这样的 $(i,i+1)$ 看作一对关键点。

由上一条性质还可以发现，对于两对关键点 $(i,i+1),(j,j+1)$，如果 $i<j$，则从 $0$ 到后一对关键点的路径一定经过上一对关键点。因此可以使用如下方式计算最短路：

依次考虑每一对关键点，从到上一对关键点的距离计算到下一对关键点的距离。

考虑相邻的一对关键点 $(i,i+1),(j,j+1)$，则 $[i+1,j]$ 为同色，$i,j+1$ 为另外一种颜色。这一对间的路径只需要考虑 $[i,j+1]$ 间的路径。可以发现有以下几种路径：

1. $i\to j+1$，长度为 $1$。
2. $i\to j+1\to j$，长度为 $2$。
3. $i+1\to i+2\to\cdots\to j$，长度为 $l-1$，其中 $l$ 为这一段长度。
4. $i+1\to\cdots\to j+1$，长度为 $l$。

由于到 $i,i+1$ 的距离不超过 $1$，因此如果 $l\geq 4$，则一定会选择上一种转移，且这种情况的转移可以看成和 $l=4$ 的转移相同。



因此可以考虑如下 $dp$：设 $dp_{i,d,s,l,c}$ 表示考虑到位置 $i$，设上一个关键位置为 $x$，则起点到 $x+1$ 的距离为 $d$，到 $x$ 的距离与到 $x+1$ 的距离差为 $s$。当前上一个关键位置 $x+1$ 与 $i$ 的距离为 $l$（即当前颜色相同的段长度），位置 $i$ 的颜色为 $c$ 时，前面染色的方案数。

初始状态为 $dp_{1,1,-1,0,c}$，转移时枚举下一个状态即可，因为 $l\geq 4$ 的情况可以看做 $l=4$，因此状态数为 $O(n^2)$。

同时可以将位置 $n$ 放入 $dp$，看作没有染色的位置，考虑所有位置 $n,n-1$ 不同的情况，可以发现这种情况的移动方式和原问题移动方式相同，且每一种原问题染色方式唯一对应一种现在的染色方式。这样即可直接使用 $dp_{n}$ 得到答案。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 4020
#define mod 1000000007
int n,k,dp[N][N][3][4][2],as;
char s[N];
int main()
{
	scanf("%d%d%s",&n,&k,s+1);s[n]='?';
	if(s[1]!='B')dp[1][1][0][0][0]=1;
	if(s[1]!='A')dp[1][1][0][0][1]=1;
	for(int i=2;i<=n;i++)
	for(int j=1;j<i;j++)
	for(int d=0;d<3;d++)
	for(int l=0;l<4;l++)
	for(int x=0;x<2;x++)if(dp[i-1][j][d][l][x])
	for(int y=(s[i]=='B');y<=(s[i]!='A');y++)
	if(x==y)dp[i][j][d][min(l+1,3)][x]=(dp[i][j][d][min(l+1,3)][x]+dp[i-1][j][d][l][x])%mod;
	else
	{
		int nj,nd;
		if(l==3)nj=j+d,nd=2;
		else
		{
			int l1=min(j+l+1,j+d),l2=min(j+l,j+d+1);
			nj=l1;nd=l2-l1+1;
		}
		dp[i][nj][nd][0][y]=(dp[i][nj][nd][0][y]+dp[i-1][j][d][l][x])%mod;
	}
	for(int j=1;j<=k;j++)for(int d=0;d<3;d++)for(int t=0;t<2;t++)as=(as+dp[n][j][d][0][t])%mod;
	printf("%d\n",as);
}
```



##### ARC120E 1D Party

###### Problem

有 $n$ 个人在一条数轴上，第 $i$ 个人在位置 $a_i$，保证 $a_i$ 为偶数且递增。

每个时刻，每个人可以在数轴上进行移动。每个人每时刻运动的距离不超过 $1$。

这些人希望在尽量短的时间内达成如下要求：

对于任意的 $1\leq i<n$，在某个时刻 $i,i+1$ 的位置重合。

求达成要求需要的最小时刻。可以证明 $a_i$ 为偶数时，答案为偶数。

$n\leq 2\times 10^5,0\leq a_i\leq 10^9$

$2s,1024MB$

###### Sol

对于每个人 $i$，考虑他第一次遇到的人，这个人一定是 $i-1$ 或者 $i+1$。

如果第一次遇到的人是 $i-1$，则可以发现这个人如果选择开始时一直向左，则他会更早遇到 $i-1$，之后他只需要向右走，而越早遇到 $i-1$ 就一定能向右走得更多。因此他会选择一直向左。

因此可以发现，在最优方案中，每个人都会选择一个方向。在开始后，每个人都会按照方向前进。在和对向的人相遇后，当前人会调转方向前进，直到达成要求。



设每个人初始移动的方向为 $d_i\in\{l,r\}$，则初始所有相邻的 `rl` 会相遇，接下来变为每一段 `l...lr...r` 之间相遇的问题。可以发现，如果将所有人划分为若干段 `l...lr...r`，设一段为 $[i,j]$，则可以发现如下情况：

如果 $i>1$，则 $i$ 会先和 $i-1$ 相遇然后转向，否则 $i$ 会直接向右。$j$ 有类似的情况。

设 $l,r$ 的分隔位置为 $k$，则 $[i+1,k]$ 间的人会先向左，遇到 $l$ 之后和 $l$ 一起向右。 $[k+1,j-1]$ 会做类似的操作。因此这一段内满足要求的时刻即为 $i,j$ 相遇的时刻。

因此这一段内的需要时刻为 $\frac12(a_{\min(r+1,n)}-a_{\max(l-1,1)})$。最后的总时刻即为所有时刻的最大值。



令 $a_0=a_1,a_{n+1}=a_n$，则相当于将序列划分成若干长度大于等于 $2$ 的段，最小化 $\max\frac12(a_{r+1}-a_{l-1})$。

直接二分答案并前缀和优化可以做到 $O(n\log v)$~~效率差不多~~。同时可以发现一个段分成两段一定更优，因此最后每一段长度为 $2$ 或 $3$，因此可以直接 $dp$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 205000
int n,v[N],dp[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),dp[i]=1e9;
	v[0]=v[1],v[n+1]=v[n];
	for(int i=1;i<=n;i++)
	{
		if(i>1)dp[i]=min(dp[i],max(dp[i-2],(v[i+1]-v[i-2])>>1));
		if(i>2)dp[i]=min(dp[i],max(dp[i-3],(v[i+1]-v[i-3])>>1));
	}
	printf("%d\n",dp[n]);
}
```



##### ARC120F Wine Thief

###### Problem

给定长度为 $n$ 的序列 $a_{1,\cdots,n}$。

给定 $k,d$，你需要从序列中选择正好 $k$ 个元素，满足如下条件：

对于序列中连续 $d$ 个元素，这 $d$ 个元素中最多被选择一个。

一种选择方案的权值为所有选择元素的和。求所有合法选择方案的权值和，模 $998244353$

$n\leq 3\times 10^5,d=2$

$3s,1024MB$

###### Sol

考虑计算 $i$ 被选中的方案数 $v_{(n,k),i}$。

在 $i$ 被选中后，$i-1,i+1$ 都不能被选中，此时序列会被分成两部分 $[1,i-2],[i+2,n]$，两部分间独立。考虑将两部分拼接在一起，则拼接后为一个长度为 $n-3$ 的序列，所有的情况对应这个序列上选择 $k-1$ 个数的如下方案：

1. 这个序列上合法的所有方案。
2. 这个序列上 $i-2,i+2$ 对应的两个相邻位置都被选中，其它位置合法的方案。

可以发现，第一种情况的方案数为 $C_{(n-3)+1-(k-1)}^{k-1}$。

而对于第二种情况，可以将这两个位置合并，合并后对应长度为 $n-4$ ，选择 $k-2$ 个的情况，同时要求 $i-2$ 位置必须被选中。

因此可以得到如下结果：
$$
v_{(n,k),i}=C_{n-k-1}^{k-1}+v_{(n-4,k-2),i-2}(3\leq i\leq n-2)
$$
因此一个 $v_{(n,k),i}$ 为若干个 $C_{(n-4t)-(k-2t)-1}^{(k-2t)-1}$ 的前缀和加上最后一个 $i=1,2$ 的情况。最后的情况特殊处理即可。

如果预处理阶乘逆元，则复杂度为 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 300500
#define mod 998244353
int n,k,fr[N],ifr[N],a,as,su[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int solve(int n,int k)
{
	if(k*2-1>n||k<0||n<0)return 0;
	return 1ll*fr[n-k+1]*ifr[k]%mod*ifr[n-2*k+1]%mod;
}
int main()
{
	scanf("%d%d%*d",&n,&k);
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=n;i++)
	{
		su[i]=solve(n-4*i+1,k-2*i+1);
		su[i]=(su[i]+su[i-1])%mod;
	}
	for(int i=1;i<=n;i++)
	{
		int li=i;if(n-i+1<li)li=n-i+1;
		int si=su[li/2];
		if(li&1)si=(si+solve(n-li*2+(i*2-1==n),k-li))%mod;
		scanf("%d",&a);
		as=(as+1ll*a*si)%mod;
	}
	printf("%d\n",as);
}
```



##### ARC120F2 Wine Thief

###### Problem

给定长度为 $n$ 的序列 $a_{1,\cdots,n}$。

给定 $k,d$，你需要从序列中选择正好 $k$ 个元素，满足如下条件：

对于序列中连续 $d$ 个元素，这 $d$ 个元素中最多被选择一个。

一种选择方案的权值为所有选择元素的和。求所有合法选择方案的权值和，模 $998244353$

$n\leq 10^6$

$10s,1024MB$

###### Sol

此时对于一个 $n,k$，方案数为 $C_{n-(k-1)(d-1)}^{k}$。

对于 $d$ 任意的情况，可以发现如果按照上面的讨论，则上述第二种情况中，不合法情况有 $\frac{d(d-1)}2$ 种，而每一种合并后减少的序列长度可能不同，因此这样递归到的状态难以一起处理。

考虑计算 $i$ 没有被选中的状态，对于这样的一个状态，考虑删去这个位置将两侧拼接，则考虑此时所有 $n-1$ 个中选 $k$ 个的方案，如果此时方案合法，则插入空位仍然合法。如果此时不合法而插入空位后合法，则当且仅当从空位分开后两侧合法，且空位左右第一个选中的数在插入空位之前距离正好为 $d-2$。

对于第二种情况，同样考虑将两侧第一个选中的数合并，考虑合并后它所在的位置。可以得到如下结果：
$$
v_{(n,k),i}=C_{n-(k-1)(d-1)}^k-C_{(n-1)-(k-1)(d-1)}^k\\-\sum_{j=i-d+1}^{i-1}v_{(n-d,k-1),i}
$$
此时直接实现可以得到复杂度 $O(\frac{n^2}d)$ 的做法。

考虑将这个过程看成多项式，由于第二部分每个 $v_{(n-d,k-1),i}$ 中 $i\leq n-d$，因此每一个值都会完整地转移到 $i+1,\cdots,i+d-1$。则可以发现答案的多项式为如下形式：
$$
F(x)=\sum_{i=0}^tv_i*(1+x+\cdots+x^{n-dt})*(-x-\cdots-x^{d-1})^t
$$
考虑乘以 $1-x$，变为 $\sum_{i=0}^tv_i*(1-x^{n-dt+1})*(-x-\cdots-x^{d-1})^t$

然后对于 $1-x^{n-dt+1}$ 中的两项分别处理，第一项显然直接分治fft即可，第二项可以使用类似的分治fft处理。

这里如果实现不当，分治fft中乘积项的次数可能为 $\lceil\frac nd\rceil*d$，而这会超过 $2^{20}$，需要注意细节。

复杂度 $O(n\log^2 n)$

###### Code

事实证明ACL比手写快一倍(5.2s->2.5s)

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 1050000
#define mod 998244353
int n,k,d,a,as,vl[N],ct,dp[N*3];
int fr[N],ifr[N],rev[N*2],gr[2][N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d=20)
{
	for(int i=2;i<=(1<<d);i<<=1)for(int j=0;j<i;j++)rev[i+j]=(rev[i+(j>>1)]>>1)|((i>>1)*(j&1));
	for(int t=0;t<2;t++)
	for(int i=2;i<=(1<<d);i<<=1)
	{
		int tp=pw(3,(mod-1)/i),vl=1;
		if(!t)tp=pw(tp,mod-2);
		for(int j=0;j<i>>1;j++)gr[t][(i>>1)+j]=vl,vl=1ll*vl*tp%mod;
	}
}
int ntt[N],f[N],g[N];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[i]=a[rev[i+s]];
	for(int l=1;l<s;l<<=1)
	for(int i=0;i<s;i+=l*2)
	for(int j=0;j<l;j++)
	{
		int v1=ntt[i+j],v2=1ll*ntt[i+j+l]*gr[t][l+j]%mod;
		ntt[i+j]=(v1+v2)%mod;ntt[i+j+l]=(v1+mod-v2)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int solve(int n,int k)
{
	if((k-1)*d+1>n||k<0||n<0)return 0;
	return 1ll*fr[n-(k-1)*(d-1)]*ifr[k]%mod*ifr[n-(k-1)*d-1]%mod;
}
void calc(int n,int k)
{
	if((k-1)*d+1>n||k<0||n<0)return;
	vl[++ct]=(solve(n,k)-solve(n-1,k)+mod)%mod;
	if(~ct&1)vl[ct]=mod-vl[ct];
	calc(n-d,k-1);
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	if(a.size()>b.size())
	{
		for(int i=0;i<b.size();i++)a[i]=(a[i]+b[i])%mod;
		return a;
	}
	else
	{
		for(int i=0;i<a.size();i++)b[i]=(a[i]+b[i])%mod;
		return b;
	}
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int l=1;while(l<=a.size()+b.size()-2)l<<=1;
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<a.size();i++)f[i]=a[i];
	for(int i=0;i<b.size();i++)g[i]=b[i];
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	vector<int> as;
	for(int i=0;i<a.size()+b.size()-1;i++)as.push_back(f[i]);
	while(as.size()&&as.back()==0)as.pop_back();
	return as;
}
struct sth{vector<int> a,b;};
sth solve1(int l,int r,int f)
{
	if(l==r)
	{
		sth tp;
		tp.a.push_back(vl[l]);
		for(int i=0;i<d;i++)tp.b.push_back(!!i);
		return tp;
	}
	int mid=(l+r)>>1;
	sth s1=solve1(l,mid,1),s2=solve1(mid+1,r,f);
	sth as;
	if(f)as.b=polymul(s1.b,s2.b);
	vector<int> ls;
	for(int i=0;i<(r-mid)*d;i++)ls.push_back(0);
	for(int i=0;i<s1.a.size();i++)ls.push_back(s1.a[i]);
	as.a=polyadd(polymul(s1.b,s2.a),ls);
	return as;
}
sth solve2(int l,int r,int f)
{
	if(l==r)
	{
		sth tp;
		tp.a.push_back(vl[l]);
		for(int i=0;i<d;i++)tp.b.push_back(!!i);
		return tp;
	}
	int mid=(l+r)>>1;
	sth s1=solve2(l,mid,1),s2=solve2(mid+1,r,f);
	sth as;
	if(f)as.b=polymul(s1.b,s2.b);
	as.a=polyadd(polymul(s1.b,s2.a),s1.a);
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&k,&d);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	calc(n,k);
	init();
	sth s1=solve1(1,ct,0);
	for(int i=0;i<s1.a.size();i++)dp[n-(ct-1)*d+1+i]=(dp[n-(ct-1)*d+1+i]+mod-s1.a[i])%mod;
	s1=solve2(1,ct,0);
	for(int i=0;i<s1.a.size();i++)dp[1+i]=(dp[1+i]+s1.a[i])%mod;
	for(int i=1;i<=n;i++)dp[i]=(dp[i]+dp[i-1])%mod;
	for(int i=1;i<=n;i++)scanf("%d",&a),as=(as+1ll*dp[i]*a)%mod;
	printf("%d\n",as);
}
```



##### ARC121E Directed Tree

###### Problem

给一棵 $n$ 个点的有根外向树，求有多少个 $n$ 阶排列 $p$ 满足如下条件：

对于任意 $i$，$p_i$ 不能通过走大于等于一条边到达 $i$。

答案模 $998244353$

$n\leq 2000$

$2s,1024MB$

###### Sol

考虑容斥，则相当于枚举一些 $i$，钦定这些 $i$ 满足 $p_i$ 是 $i$ 的祖先且 $p_i\neq i$，此时方案数会产生 $(-1)^{|S|}$ 的贡献。则只考虑这些 $i$ 填的方案数可以 $dp$ 计算，而剩下的位置可以任意填，方案数为 $(n-|S|)!$。

考虑 $dp$，设 $dp_{i,j}$ 表示 $i$ 子树内钦定了 $j$ 个点不满足限制，且这些点的 $p_i$ 都在子树内的情况数。合并不同儿子时直接背包合并。然后考虑加入子树的根，则根可以选择子树内不是根且之前没有被钦定的一个点 $x$，让 $p_x$ 为根，增加一个不满足限制的点，也可以不钦定。最后用 $dp_1$ 即可得到容斥的结果。

复杂度 $O(n^2)$

另外一种理解方式是，可以发现每一个排列与逆排列一一对应，而对应逆排列后，限制变为 $i$ 是 $p_i$ 的祖先，此时 $dp$ 状态中就不需要假设 $p_i$ 都在子树内，但 $dp$ 过程完全相同。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2050
#define mod 998244353
int n,f[N],sz[N],dp[N][N],as;
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)scanf("%d",&f[i]);
	for(int i=1;i<=n;i++)dp[i][0]=1;
	for(int i=n;i>=1;i--)
	{
		for(int j=sz[i];j>=0;j--)dp[i][j+1]=(dp[i][j+1]+mod-1ll*dp[i][j]*(sz[i]-j)%mod)%mod;
		sz[i]++;
		for(int j=sz[f[i]];j>=0;j--)
		for(int k=1;k<=sz[i];k++)
		dp[f[i]][j+k]=(dp[f[i]][j+k]+1ll*dp[f[i]][j]*dp[i][k])%mod;
		sz[f[i]]+=sz[i];
	}
	for(int i=0,fr=1;i<=n;i++,fr=1ll*fr*i%mod)as=(as+1ll*dp[1][n-i]*fr)%mod;
	printf("%d\n",as);
}
```



##### ARC121F Logical Operations on Tree

###### Problem

给定一棵 $n$ 个点的树。

现在每个点上有 $\{0,1\}$ 中的一个权值，每条边上有 $and,or$ 两种操作中的一种。

你需要进行 $n-1$ 次操作，每次操作选择一条边，将这条边的两个端点合并，合并后端点的权值为原先两个端点的权值使用这条边上的操作运算的结果。

求所有 $2^{2n-1}$ 中在不同的点权值和边操作情况中，有多少种情况满足如下条件：

存在一种操作方式，使得最后剩余的点上权值为 $1$。

答案模 $998244353$

$n\leq 10^5$

$2s,1024MB$

###### Sol

考虑一种情况的操作过程。

考虑一个叶子，如果叶子以及向上的边构成操作 $and\ 1,or\ 0$，可以发现这两种操作任何时候不改变上面点的权值，因此可以直接删去这样的叶子。

如果一个叶子构成操作 $or\ 1$，则显然可以先操作其它部分，最后操作这条边使得最后权值为 $1$。因此如果存在这样的叶子，则这种情况满足条件。

如果一个叶子构成操作 $and\ 0$，考虑其它的操作构成的操作树，加入这个叶子相当于在叶子父亲到操作树的根上选择一个位置，将这个位置变成 $0$。可以发现变成 $0$ 的位置越深越优，因此直接将叶子父亲变成 $0$ 最优。

因此操作可以看成如下方式：

不断操作除去 $or\ 1$ 之外的所有叶子，如果出现 $or\ 1$ 的叶子或者只剩一个 $1$ 的节点则达成目标。

可以发现，如果固定一个根，上述讨论仍然成立，因此操作可以看成在有根的情况下进行。

考虑计算不满足条件的方案数，不满足条件当且仅当不存在 $or\ 1$ 且最后根权值为 $0$。

设 $dp_{i,0/1}$ 表示考虑 $i$ 子树内，满足操作中不存在 $or\ 1$ 且最后根权值为 $0,1$ 的方案数。考虑转移，假设确定了每个子树根的权值，则如果根权值为 $1$，则这条边只能是 $and$，否则两种操作都可以出现。如果出现了 $and\ 0$，则无论如何根权值为 $0$，否则根权值为最初的根权值。

而只出现 $and\ 1,or\ 0$ 的方案数为 $\prod_{v\in son_u}(dp_{v,0}+dp_{v,1})$，三种情况都可以的方案数为 $\prod_{v\in son_u}(2dp_{v,0}+dp_{v,1})$。因此出现 $and\ 0$ 的方案数为后者减去前者。最后 $dp_{u,0}$ 为出现 $and\ 0$ 的方案数乘 $2$ 加上不出现的方案数，$dp_{u,1}$ 为不出现的方案数。

最后答案即为 $2^{2n-1}-dp_{u,0}$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define mod 998244353
int n,a,b,head[N],cnt,dp[N][2],as;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void dfs(int u,int fa)
{
	int s1=1,s2=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	dfs(ed[i].t,u),s1=1ll*s1*(2ll*dp[ed[i].t][0]+dp[ed[i].t][1])%mod,s2=1ll*s2*(dp[ed[i].t][0]+dp[ed[i].t][1])%mod;
	dp[u][0]=(2ll*s1+mod-s2)%mod;dp[u][1]=s2;
}
int main()
{
	scanf("%d",&n);as=2;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b),as=4ll*as%mod;
	dfs(1,0);
	printf("%d\n",(1ll*as+mod-dp[1][0])%mod);
}
```



##### ARC122E Incrasing LCMs

###### Problem

给一个长度为 $n$ 的序列 $a$，你需要将 $a$ 重排得到 $x$，使得 $x$ 满足如下性质：

令 $y_i=lcm(x_1,\cdots,x_i)$，则 $y_i$ 严格递增：$y_1<y_2<\cdots<y_n$。

如果有解，则构造任意一种合法方案，否则输出无解。

$n\leq 100,a_i\leq 10^{18}$

$2s,1024MB$

###### Sol

考虑一个 $a_i$ 能被放在结尾的条件，能放在结尾当且仅当 $lcm_{j\neq i}\ a_j<lcm_j\  a_j$。即存在 $p,q$ 使得 $p^q|a_i$ 且 $\forall j\neq i,p^q$ 不整除 $a_j$。

因此可以发现，如果当前一个数可以被放在结尾，则删除若干个数后它仍然可以被放在结尾。

因此考虑每次在可以选择的 $a_i$ 中任意选一个放在结尾，如果不能选则无解。由上一条性质可得，如果这样得到了无解，则不存在一种操作方式使得现在没有被放的数在这种操作方式中可以被放。因此直接使用上述方式即可。

直接求LCM结果可能过大，但上述式子等价于 $lcm_{j\neq i}\gcd(a_i,a_j)<a_i$，而使用后者判断则不会出现溢出问题。直接实现即可。

复杂度 $O(n^3\log v)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define ll long long
int n;
ll v[N],as[N];
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll lcm(ll a,ll b){return a/gcd(a,b)*b;}
bool chk(int x)
{
	ll s1=1;
	for(int i=1;i<=n;i++)if(i!=x&&v[i])s1=lcm(s1,gcd(v[i],v[x]));
	return s1<v[x];
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	for(int i=1;i<=n;i++)
	{
		int fr=-1;
		for(int j=1;j<=n;j++)if(v[j]&&(i==n||chk(j)))fr=j;
		if(fr==-1){printf("No\n");return 0;}
		as[n-i+1]=v[fr];v[fr]=0;
	}
	printf("Yes\n");
	for(int i=1;i<=n;i++)printf("%lld ",as[i]);
}
```



##### ARC122F Domination

###### Problem

二维平面上有 $n$ 个红点，$m$ 个蓝点，每个点坐标都是 $[0,10^9]$ 间的整数。

你可以进行多次操作，每次操作可以选择一个蓝点并移动它。将蓝点从 $(x,y)$ 移动到 $(x',y')$ 的代价为 $|x-x'|+|y-y'|$。

你需要花费最小的总代价，使得如下条件成立：

对于任意一个红点，这个红点的右上角（即 $x'\geq x,y'\geq y$ 组成的区域） 中有至少 $k$ 个蓝点。

求最小总代价。

$n,m\leq 10^5,k\leq 10$

$7s,1024MB$

###### Sol

由于红点固定，可以发现如果一个红点右上有红点，则如果右上红点满足要求则当前点一定满足要求。因此可以只保留所有右上没有其他红点的红点。

此时如果将所有剩余红点按照 $x$ 从小到大排序，则红点的 $y$ 一定递减。设排序后红点坐标为 $(x_1,y_1),\cdots,(x_l,y_l)$。

此时考虑一个蓝点可以影响到的红点。设蓝点坐标 $(x,y)$ 满足 $x_r\leq x<x_{r+1},y_{l}\leq y<y_{l-1}$，则可以发现该点影响的红点为一段区间 $[l,r]$ 内的所有红点。



考虑蓝点不动的情况，此时相当于判断 $m$ 个区间是否能将区间内每个位置覆盖 $k$ 次。

考虑一个网络流模型，设此时还有 $n$ 个红点，考虑建 $n+1$ 个点标号为 $1,2,\cdots,n+1$，点 $i+1$ 向点 $i$ 连流量不限的边。对于每一个区间 $[l,r]$，从 $l$ 向 $r+1$ 连流量为 $1$ 的边。

考虑这张图上 $1$ 到 $n+1$ 的流。如果流量大于等于 $k$，则意味着存在 $k$ 条 $1$ 到 $n+1$ 的路径使得每条区间对应的边最多被使用一次。这相当于可以在区间中选出 $k$ 个不交集合，使得每个集合中的区间能覆盖所有位置，此时每个位置一定至少被覆盖了 $k$ 次。

另一方面，考虑这个图的一个最小割，如果去掉割边后对于每个 $i$，都存在 $[1,i]$ 到 $[i+1,n+1]$ 的边，则由于不能割掉 $i+1\to i$ 的边，此时一定存在 $1\to n+1$ 的路径，与割矛盾。因此在一个最小割中，一定存在一个 $i$，使得割掉了所有 $[1,i]\to[i+1,n+1]$ 的边。因此如果这些区间满足每个位置至少被覆盖 $k$ 次，则最小割一定大于等于 $k$。

因此每个位置能被覆盖 $k$ 次当且仅当这张图的最大流大于等于 $k$。



考虑原问题中蓝点移动的部分。显然减小 $x$ 或者减小 $y$ 只会减少覆盖区域，因此只会增大 $x,y$。考虑将减小 $x,y$ 的操作看成没有代价，显然这样不会改变答案。而这样变化以后，对于最优解中的一种方案，如果一个蓝点满足 $x_r\leq x<x_{r+1},y_{l}\leq y<y_{l-1}$，则可以通过减小 $x,y$ 使得 $x=x_r,y=y_l$。因此最后每个起作用的蓝点一定会移动到形如 $(x_r,y_l)$ 的位置，而这表示它相当于一个区间 $[l,r]$。

因此考虑新建两排点，分别表示 $x,y$ 的变化。流的方向为从 $y$ 流向 $x$，区间对应的边为连接两排点的边。第一排点表示 $y$ 的变化，每个权值对应一个点。其中 $y+1$ 向 $y$ 连流量不限费用为 $1$ 的边，表示增大 $y$。$y$ 向 $y+1$ 连流量不限费用为 $0$ 的边，表示减小 $y$。

对于 $x$ 部分，建类似的一排点，但此时 $x+1\to x$ 费用为 $0$，$x\to x+1$ 费用为 $1$。

然后对于一个蓝点 $(x',y')$，从第一排点的 $y'$ 向第二排点的 $x'$ 连边，流量为 $1$ 没有费用。

此时考虑从第一排的 $y$ 出发，经过第一排的路径，再经过中间的边 $(x',y')$ ，最后到达第二排的 $x$，费用为 $\max(0,x-x')+\max(0,y-y')$。这就表示了将 $(x',y')$ 边对应蓝点移动到 $(x,y)$ 的过程。

最后考虑原先的 $n+1$ 个点，保留 $i+1\to i$ 的边，然后在原先的 $n+1$ 个点上，从点 $i$ 向 $y_i$ 对应点连边，$x_i$ 向 $i+1$ 对应点连边，流量不限。可以发现从这 $n+1$ 个点向外的一条路径就表示了一个蓝点通过移动对应到一个区间的方式。从而最后在这个图上 $1$ 到 $n+1$ 的流量为 $k$ 的最小费用流即为答案。

注意到 $x,y$ 中每一排点上只有 $n+m$ 个向外连接的点，因此可以将向外连接点中间的部分合并，这样图中点数边数均为 $O(n+m)$。



最后的问题为求最小费用流。使用primal-dual算法即可做到 $O(k(n+m)\log(n+m))$。~~可以使用atcoder::mcf_graph~~

primal-dual算法简要描述：

首先使用最短路算法求出 $s$ 到每个点的最短路 $d_i$，这里因为初始没有负边权可以直接处理。然后重复进行下面的操作：

将每条边 $(u\to v,c)$ 的权值 $c$ 改为 $c-d_v+d_u$。由三角形不等式，当前图中所有边（不考虑流量为 $0$ 而不在残量网络中的边）边权非负。而在新的图上 $s$ 到任意点的最短路为 $0$。因此在这个图上，$s$ 到 $t$ 增广最短路只需要保证经过边边权为 $0$ 即可。

因此此时可以使用dinic的方式进行增广，增加的费用为 $d_t$ 乘以流量。考虑增广之后残量网络改变对 $d_i$ 的改变。因为增广的都是边权为 $0$ 的边，可以发现它们的反向边边权也是 $0$，因此增广后在原来的 $d$ 处理后的图中，当前的所有边边权非负，此时仍然可以 $O((n+m)\log(n+m))$ 求出这个图上的最短路 $d_i'$。最后考虑 $d_i$ 的变化，显然新的 $d_i$ 为原先的 $d_i$ 加上 $d_i'$。

这样就完成了增广后对新的残量网络求最短路的过程。可以发现每一轮的复杂度为 $O(n\log n)$ 再加上dinic的复杂度。而dinic复杂度不超过 $O(F*(n+m))$，其中 $F$ 为流量。因此总复杂度不超过 $O(F*(n+m)\log(n+m))$，其中 $F$ 为需要的流量大小。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
#include<cstring>
using namespace std;
#define N 200500
int n,m,k,a,b,sx[N],sy[N],ct,tx[N],ty[N],rx[N],ry[N],su;
pair<int,int> s1[N];
int head[N*3],dis[N*3],cur[N*3],cnt=1,di[N*3];
struct edge{int t,next,v,c;}ed[N*14];
void adde(int f,int t,int v,int c){ed[++cnt]=(edge){t,head[f],v,c};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0,-c};head[t]=cnt;}
bool dij(int s,int t)
{
	for(int i=1;i<=su;i++)dis[i]=2.01e9;
	dis[s]=0;
	priority_queue<pair<int,int> > qu;
	qu.push(make_pair(0,s));
	while(!qu.empty())
	{
		int v=qu.top().second;qu.pop();
		for(int i=head[v];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]>1ll*dis[v]+ed[i].c)
		dis[ed[i].t]=dis[v]+ed[i].c,qu.push(make_pair(-dis[ed[i].t],ed[i].t));
	}
	return dis[t]<2.005e9;
}
bool bfs(int s,int t)
{
	for(int i=1;i<=su;i++)di[i]=-1;
	di[s]=0;
	queue<int> qu;
	qu.push(s);
	while(!qu.empty())
	{
		int v=qu.front();qu.pop();
		for(int i=head[v];i;i=ed[i].next)if(ed[i].v&&ed[i].c==0&&di[ed[i].t]==-1)di[ed[i].t]=di[v]+1,qu.push(ed[i].t);
	}
	return di[t]>-1;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int& i=cur[u];i;i=ed[i].next)if(ed[i].v&&ed[i].c==0&&di[ed[i].t]==di[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
long long dij_mcmf(int s,int t,int v)
{
	dij(s,t);
	for(int i=1;i<=su;i++)for(int j=head[i];j;j=ed[j].next)ed[j].c+=dis[i]-dis[ed[j].t];
	long long as=0,di=dis[t];
	while(1)
	{
		memcpy(cur,head,sizeof(head));
		if(!bfs(s,t))return as;
		int v1=dfs(s,t,v);
		v-=v1;as+=di*v1;
		if(!v||!dij(s,t))return as;
		for(int i=1;i<=su;i++)for(int j=head[i];j;j=ed[j].next)ed[j].c+=dis[i]-dis[ed[j].t];
		di+=dis[t];
	}
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++)scanf("%d%d",&a,&b),s1[i]=make_pair(a,b);
	sort(s1+1,s1+n+1);
	int mx=-233;
	for(int i=n;i>=1;i--)if(s1[i].second>mx)mx=s1[i].second,sx[++ct]=s1[i].first,sy[ct]=mx;
	reverse(sx+1,sx+ct+1);reverse(sy+1,sy+ct+1);
	for(int i=1;i<=ct;i++)rx[i]=sx[i],ry[i]=sy[i];
	for(int i=1;i<=m;i++)scanf("%d%d",&tx[i],&ty[i]),rx[i+ct]=tx[i],ry[i+ct]=ty[i];
	sort(rx+1,rx+m+ct+1);sort(ry+1,ry+m+ct+1);
	su=m*2+ct*3+2;
	int s=su-1,t=su;
	adde(s,1,1000,0);
	for(int i=2;i<=ct;i++)adde(i,i-1,1000,0);
	for(int i=2;i<=m+ct;i++)adde(i+ct,i+ct-1,1000,ry[i]-ry[i-1]),adde(i+ct*2+m-1,i+ct*2+m,1000,rx[i]-rx[i-1]);
	for(int i=2;i<=m+ct;i++)adde(i+ct-1,i+ct,1000,0),adde(i+ct*2+m,i+ct*2+m-1,1000,0);
	for(int i=1;i<=ct;i++)adde(i,lower_bound(ry+1,ry+m+ct+1,sy[i])-ry+ct,1000,0),adde(lower_bound(rx+1,rx+m+ct+1,sx[i])-rx+m+ct*2,i==ct?t:i+1,1000,0);
	for(int i=1;i<=m;i++)adde(lower_bound(ry+1,ry+m+ct+1,ty[i])-ry+ct,lower_bound(rx+1,rx+m+ct+1,tx[i])-rx+m+ct*2,1,0);
	printf("%lld\n",dij_mcmf(s,t,k));
}
```



##### ARC123E Training

###### Problem

给定正整数 $a,b,c,d$，求 $1,2,\cdots,n$ 中有多少个正整数 $t$ 满足如下限制：
$$
a+\lfloor\frac tb\rfloor=c+\lfloor\frac td\rfloor
$$
多组数据，$T\leq 2\times 10^5,a,b,c,d,n\leq 10^9$

$3s,1024MB$

###### Sol

限制相当于 $\lfloor\frac{ab+t}b\rfloor=\lfloor\frac{cd+t}d\rfloor$。

考虑这两个值 $\frac{ab+t}b,\frac{cd+t}d$，限制为它们下取整相等。显然下取整相等时，它们的差一定小于 $1$。两个函数都是线性函数，因此它们的差也是线性函数，因此满足差在一个区间内的 $t$ 也构成一个区间。因此考虑求出差属于 $(-1,0),[0,1)$ 部分的答案数，求和即为答案。

考虑差属于 $[0,1)$ 的部分，注意到如果 $a\geq b$，则取整后仍然大于等于。因此这部分内一定有 $\lfloor\frac{ab+t}b\rfloor\geq\lfloor\frac{cd+t}d\rfloor$。而因为差不超过 $1$，因此取整后差最多为 $1$。

因此求这一段内满足要求的时刻数，可以看成求这一段内的 $\sum 1-\lfloor\frac{ab+t}b\rfloor+\lfloor\frac{cd+t}d\rfloor$。

因此只需要求 $\sum_{t=l}^r\lfloor\frac tb\rfloor$。由于 $\sum_{t=0}^r\lfloor\frac tb\rfloor$ 可以 $O(1)$ 计算，因此单次询问可以 $O(1)$ 解决。

注意区间边界情况以及 $b=d$ 的情况。

复杂度 $O(T)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
int T,n,a,b,c,d;
ll calc(int n,int a,int b)
{
	n++;
	ll as=1ll*n*a;
	as+=1ll*b*(n/b)*(n/b-1)/2;
	as+=1ll*(n%b)*(n/b);
	return as;
}
int calc1(ll l,ll r,int a,int b,int c,int d)
{
	if(l>r)return 0;
	if(l<1)l=1;if(r<1)return 0;
	if(l>n)return 0;if(r>n)r=n;
	return r-l+1-calc(r,a,b)+calc(r,c,d)+calc(l-1,a,b)-calc(l-1,c,d);
}
void solve()
{
	scanf("%d%d%d%d%d",&n,&a,&c,&b,&d);
	if(c==d){printf("%d\n",a==b?n:0);return;}
	if(c<d)c^=d^=c^=d,a^=b^=a^=b;
	ll lb=1ll*(a-b-1)*c*d/(c-d)+1,md=1ll*(a-b)*c*d/(c-d),rb=1ll*(a-b+1)*c*d/(c-d);
	printf("%d\n",calc1(lb,md,a,c,b,d)+calc1(md+1,rb,b,d,a,c));
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```





##### ARC123F Insert Addition

###### Problem

定义对序列进行的一次插入操作为，在原序列每一对相邻数之间插入它们的和。即设序列为 $p_{1,2,\cdots,m}$，则插入后的序列 $f(p)=\{p_1,p_1+p_2,p_2,p_2+p_3,p_3,\cdots,p_m\}$。

给定 $a,b,n$，考虑一个初始为 $\{a,b\}$ 的序列，对其进行 $n$ 次插入操作，随后只保留序列中长度不超过 $n$ 的元素，按原顺序排列得到序列 $s$。

给定 $l,r$，求 $s_{l,\cdots,r}$。

$1\leq a,b\leq n\leq 3\times 10^5,1\leq l\leq r\leq 10^{18},r-l\leq 3\times 10^5$，保证 $|s|\geq r$

$4s,1024MB$

###### Sol

ref:Expanding Sequence

在第 $i$ 轮插入后，相邻两个数的和至少是 $i+1$。因此 $n$ 轮操作后，如果再进行插入，则插入的数一定大于 $n$。因此可以看成进行无限轮操作，但小于等于 $n$ 的数只有有限个。

可以发现这个插入操作与 Stern-Brocot Tree 有着联系。具体来说，考虑将初始序列看成 $(0,1),(1,0)$，两个二元组相加为对应元素相加。则每一次插入操作正好为树上扩展下一层的过程。而进行无限次操作即得到整个树的结构。同时，这个序列即为树的中序遍历。由 Stern-Brocot Tree 的性质，如果将二元组 $(i,j)$ 看成分数 $\frac ij$，则每个二元组都对应既约分数，且树的中序遍历即为将所有有理数按照大小顺序排序的结果。

再考虑一个二元组和原序列值的关系。可以发现，如果二元组为 $(i,j)$，则它对应原序列的值为 $aj+bi$。因此，$s$ 可以使用如下方式构造：

考虑所有满足 $i,j>0,aj+bi\leq n,\gcd(i,j)=1$ 的二元组 $(i,j)$，将它们按照对应的 $\frac ij$ 的值从小到大排序，再在序列开头加上 $(0,1)$，结尾加上 $(1,0)$，然后按顺序记录每个二元组的 $aj+bi$ 即得到 $s$。

如果求出了第 $l,r$ 个二元组，则由于二元组间按照对应分式的值排序，可以 $O(n+(r-l))$ 求出所有 $l,r$ 间的二元组，排序后即可得到答案。因此只需要支持求出 $s$ 中某个位置的二元组即可。



考虑在 Stern-Brocot Tree 上查找第 $k$ 个位置，则需要支持询问某个子树内在 $s$ 中的二元组数量。由于子树的构造与整个树的构造相同，可以发现子树内的数量相当于给一组 $a,b$，询问这组 $a,b$ 下构造的 $s$ 大小，即：
$$
\sum_{i=1}\sum_{j=1}[aj+bi\leq n][gcd(i,j)=1]1
$$
考虑反演或容斥，即变为如下结果：
$$
\sum_{d=1}^n\mu(d)*\sum_{i=1}\sum_{j=1}[aj+bi\leq \frac nd]1
$$
而后者形如 $\sum_{t=0}^n\lfloor\frac{at+b}c\rfloor$，可以类欧 $O(\log n)$ 计算。

如果枚举 $d$ 计算，则单次复杂度为 $O(\frac n{\max(a,b)}\log n)$，根据 Stern-Brocot Tree 的性质这样复杂度为 $O(n\log^2 n)$。

这里显然可以对 $d$ 数论分块，单次复杂度为 $O(\sqrt{\frac n{\max(a,b)}}\log n)$。而 $\sum_{i=1}^n \sqrt\frac ni=O(n)$，因此这样复杂度为 $O(n\log n)$。

最后，可以将在 Stern-Brocot Tree 上直接走变为二分每次向同一个方向走的次数。根据欧几里得算法，找一个二元组最多转向 $O(\log n)$ 次，分析可得复杂度不超过 $O(\sqrt n\log^2 n)$，但还是需要预处理 $\mu$ 的前缀和。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,a,b,ct,pr[N],ch[N],su[N],c1;
ll l,r;
void prime()
{
	su[1]=1;
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++c1]=i,su[i]=-1;
		for(int j=1;i*pr[j]<=n&&j<=c1;j++)
		{
			ch[i*pr[j]]=1;su[i*pr[j]]=-su[i];
			if(i%pr[j]==0){su[i*pr[j]]=0;break;}
		}
	}
	for(int i=1;i<=n;i++)su[i]+=su[i-1];
}
ll floorsum(int n,int a,int b,int c)
{
	ll as=0;
	if(b>=c)as+=1ll*n*(b/c),b%=c;
	if(a>=c)as+=1ll*n*(n-1)/2*(a/c),a%=c;
	if(!a)return as;
	int li=(1ll*(n-1)*a+b)/c;
	return as+1ll*li*(n-1)-floorsum(li,c,c-b-1,a);
}
ll calc1(int a,int b,int c){c-=a;if(c<0)return 0;return floorsum(c/a+1,a,c%a,b);}
ll calc(int a,int b,int c)
{
	ll as=0;
	for(int l=1,r;l<=c;l=r+1)
	{
		r=c/(c/l);
		if(c/l<a+b)break;
		as+=(su[r]-su[l-1])*calc1(a,b,c/l);
	}
	return as;
}
struct sth{int a,b;}as[N];
bool operator <(sth a,sth b){return 1ll*a.a*b.b>1ll*a.b*b.a;}
int gcd(int a,int b){return b?gcd(b,a%b):a;}
sth solve(int x,int y,ll k)
{
	ll tp=calc(x,x+y,n);
	if(tp>=k)
	{
		int lb=2,rb=(n-y)/x,as=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)/2;
			if(calc(x,y+mid*x,n)>=k)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		sth s1=solve(x,y+as*x,k);
		s1.a+=s1.b*as;return s1;
	}
	else if(tp+1==k)return (sth){1,1};
	else
	{
		ll s1=calc(x+y,y,n);
		int lb=2,rb=(n-x)/y,as=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)/2;
			if(s1-calc(x+mid*y,y,n)<k-tp-1)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		sth s2=solve(x+as*y,y,k-tp-1-(s1-calc(x+as*y,y,n)));
		s2.b+=s2.a*as;return s2;
	}
}
sth solve1(ll k)
{
	if(k==1)return (sth){1,0};k--;
	if(k==calc(a,b,n)+1)return (sth){0,1};
	return solve(a,b,k);
}
int main()
{
	scanf("%d%d%d%lld%lld",&a,&b,&n,&l,&r);
	prime();
	sth sl=solve1(l),sr=solve1(r);
	as[++ct]=sr;
	if(sl.a)
	for(int i=1;i*a<=n;i++)
	for(int j=(1ll*sl.b*i+sl.a-1)/sl.a;i*a+j*b<=n;j++)
	{
		if(gcd(i,j)>1)continue;
		sth st=(sth){i,j};
		if(!(st<sr))break;
		as[++ct]=st;
	}
	sort(as+1,as+ct+1);
	for(int i=1;i<=ct;i++)printf("%d ",as[i].a*a+as[i].b*b);
}
```



##### ARC124E Pass to Next

###### Problem

有 $n$ 个人排成一个环，初始第 $i$ 个人手上有 $a_i$ 个球。

现在每个人可以选择一定数量的球，然后所有人同时将选择的球传给环上下一个人。这里操作同时进行，因此每个人传给下一个人的球只能是初始时有的球。

在操作结束后，考虑每个人当前有的球数量构成的序列 $s$。求每一种可能出现的 $s$ 的 $\prod s_i$ 之和。答案模 $998244353$

$n\leq 10^5$

$2s,1024MB$

###### Sol

可以发现，如果两种传球方案使得最后每个人手上的球数量相同，则做差可得每个人传给下一个人的球数量在两种方案中的差相等。

设第 $i$ 个人的传球数量为 $c_i$，则对于 $\min c_i\neq 0$ 的方案，这种情况最后球的数量和某种 $\min c_i=0$ 的方案相同。而任意两种 $\min c_i=0$ 的传球方案得到的最后球的数量不同。因此只需要计算所有 $\min c_i=0$ 的方案的 $\prod s_i$ 即可。



考虑 $\prod s_i$ 的组合意义，即每个人在自己最后有的球中选择一个的方案数。考虑这个球是留在自己手上的还是从上一个人那里传过来的，可以对这个状态进行 $dp$。由于在环上所以需要额外记录开头的情况，设 $dp_{i,0/1,0/1,0/1}$ 表示考虑了前 $i$ 个人的操作，当前是否存在 $c_i=0$，当前 $i$ 向 $i+1$ 传的球中是否包含了 $i+1$ 最后选择的球，$n$ 向 $1$ 传的球中是否包含了 $1$ 最后选择的球，这种情况的方案数。

转移时考虑这个人的 $c_i$ 是否是 $0$，这个人自己选择的球来自哪一部分以及传给下一个人的球中是否包含被下一个人选择的球即可。可以发现每一部分的转移系数都由若干个 $C_{a_i}^{k}(k\leq 3)$ 组成。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int dp[8],rs[8],ri[8],n,a,c[3];
void calc(int a)
{
	a++;
	c[0]=a;c[1]=1ll*a*(a-1)%mod*(mod+1)/2%mod;c[2]=1ll*a*(a-1)%mod*(a-2)%mod*(mod+1)/6%mod;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)
	{
		int su=c[i+j],si=0;
		if(!j)si=i?c[0]-1:1;
		dp[i+j*2]=(su+mod-si)%mod;
		dp[i+j*2+4]=si;
	}
}
int main()
{
	scanf("%d%d",&n,&a);
	calc(a);
	for(int i=2;i<=n;i++)
	{
		for(int j=0;j<8;j++)rs[j]=dp[j];
		scanf("%d",&a);calc(a);
		for(int j=0;j<8;j++)for(int k=0;k<8;k++)
		{
			if(((j>>1)^k^1)&1)continue;
			ri[(j&5)|(k&6)]=(ri[(j&5)|(k&6)]+1ll*rs[j]*dp[k])%mod;
		}
		for(int j=0;j<8;j++)dp[j]=ri[j],ri[j]=0;
	}
	printf("%d\n",(dp[5]+dp[6])%mod);
}
```



##### ARC124F Chance Meeting

###### Problem

有一个 $n\times m$ 的网格，初始有两个人分别在 $(1,1),(n,1)$。

你可以进行 $2(n+m-2)$ 次如下操作，使得两个人分别到达 $(n,m),(1,m)$。

每次操作为以下两种操作中的一种：

1. 选择第一个人，选择向下或者向右的一种方向，让他向这个方向走一格。
2. 选择第二个人，选择向上或者向右的一种方向，让他向这个方向走一格。

这里行按照从上到下标号，列按照从左到右标号。

求有多少种合法的操作序列满足如下要求：

在某次操作后，两个人位置相同，且这样的情况正好出现一次。

答案模 $998244353$

$n,m\leq 2\times 10^5$

$2s,1024MB$

###### Sol

分开考虑上下的操作和左右的操作。对于上下操作，这部分一共有 $2n-2$ 次，且可以发现无论这部分按照什么顺序操作，一定在 $n-1$ 次操作后到下一次操作前两个人位置的这一维相同。因此可以不考虑这部分的顺序，最后乘上 $C_{2n-2}^{n-2}$。

然后考虑左右的操作，将第一个人向右看成 $+1$，第二个人向右看成 $-1$，则这部分操作为 $m-1$ 个 $+1$ 以及 $m-1$ 个 $-1$ 构成的序列，两个人这一维位置相同当且仅当当前操作的前缀和为 $0$。

如果再考虑上上一部分，相当于是对于一个这样的序列，向序列中任意插入 $2n-2$ 个位置，要求插入的第 $n-1,n$ 个位置之间正好存在一个位置在序列中前缀和为 $0$。



考虑这个前缀和为 $0$ 的位置，则这个位置需要满足如下条件：

1. 这个位置两侧分别插入了 $n-1$ 个位置。
2. 这个位置到左侧第一个前缀和为 $0$ 的位置间至少插入了一个位置。
3. 这个位置到右侧第一个前缀和为 $0$ 的位置间至少插入了一个位置。

考虑一侧的情况，设 $f_i$ 表示这一侧有 $2i$ 个数时的合法情况数。总情况数为 $v_i=C_{2i}^i*C_{2i+n-1}^{n-1}$，考虑不合法的方案，枚举右侧第一个前缀和为 $0$ 的位置的距离 $2j$，则不合法当且仅当所有 $n-1$ 个位置插入到了这个位置右侧，因此有：
$$
f_i=v_i-\sum_{j=1}^i2*c_{j-1}*v_{i-j}
$$
其中 $c_i$ 为卡特兰数，对应这一段内只有开头结尾前缀和为 $0$ 的方案数。

因此这里可以fft求出 $f$，最后答案为 $\sum_{i=0}^{m-1} f_if_{m-i}$ 再乘上之前的系数 $C_{2n-2}^{n-1}$。这样可以做到 $O(n+m\log m)$。



但仔细考虑这里的生成函数，设 $F(x)=\sum_i v_ix^i,G(x)=1-\sum_{i\geq 1}2c_{i-1}x^i$，则最后要求的为 $[x^{m-1}]F(x)^2G(x)^2$。

但卡特兰数的生成函数为 $\frac 1{2x}(1-\sqrt{1-4x})$，而 $G(x)=1-2xC(x)$，因此 $G(x)=\sqrt{1-4x}$。

因此 $F(x)^2G(x)^2=(1-4x)F(x)^2$。而 $F$ 可以 $O(n+m)$ 求出，可以 $O(m)$ 求出 $F^2$ 的一项，从而可以 $O(n+m)$ 求出答案。

复杂度 $O(n+m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 805000
#define mod 998244353
int n,m,li,fr[N],ifr[N],g[N],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int calc(int x){int as=0;for(int i=0;i<=x;i++)as=(as+1ll*g[i]*g[x-i])%mod;return as;}
int main()
{
	scanf("%d%d",&n,&m);n--;m--;li=(n+m)*2+2;
	fr[0]=1;for(int i=1;i<=li;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[li]=pw(fr[li],mod-2);for(int i=li;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int i=0;i<=m;i++)g[i]=1ll*fr[i*2+n]*ifr[i]%mod*ifr[i]%mod*ifr[n]%mod;
	printf("%d\n",1ll*(calc(m)-4ll*calc(m-1)%mod+mod)*fr[n*2]%mod*ifr[n]%mod*ifr[n]%mod);
}
```



##### ARC125E Snack

###### Problem

有 $n$ 种物品，其中第 $i$ 种物品有 $a_i$ 个。

有 $m$ 个人，你希望将这些物品中的一部分分给这些人，但需要满足如下限制：

对于第 $i$ 个人，他每一种物品最多可以得到 $b_i$ 个，并且得到的总物品数量不超过 $c_i$。

求最多可以分出多少个物品。

$n,m\leq 2\times 10^5,a_i,c_i\leq 10^{12},b_i\leq 10^7$

$2s,1024MB$

###### Sol

为了简便，这里将人和物品反过来看，即每个人最多拿 $a_i$ 个物品，第 $i$ 种物品有 $c_i$ 个且每个人最多拿 $b_i$ 个。



一种贪心做法：

考虑将所有人按照 $a_i$ 从大到小排序，然后让这些人依次拿物品。

然后考虑弱化限制：对于第 $i$ 种物品，将限制变为前 $k$ 个人最多拿 $\min(c_i,k*b_i)$ 个这种物品。

则可以证明，如果在弱化的限制下存在一种方案，则可以通过调整使得存在原问题的合法方案且选择的物品数量不变。



证明：考虑从前往后依次调整每个人选择的物品。假设当前考虑到第 $k$ 个人，这个人选的第 $i$ 种物品超过了 $b_i$ 个，但前面总共选择的这种物品没有超过 $k*b_i$。且此时前 $k-1$ 个人的选择已经满足原条件。

则前面存在一些人这种物品选择的数量更少，考虑和选择这种物品数量少于 $b_i$ 的人进行调整。设这个人为 $k'$，如果 $k'$ 选的物品数量没有达到上界，则可以直接将这个物品 $k'$ 选。否则，因为 $a_{k'}\geq a_k$，因此一定存在一种物品，使得这种物品 $k'$ 选择的数量大于 $k$ 选择的数量，此时将 $k'$ 选择的一个这种物品给 $k$ 选择，将 $k$ 超过上界的那种物品给 $k'$ 选择。这样调整一次后，新调整的物品 $k$ 选择的数量不超过 $k'$ 之前选择的数量，因此这种物品选的数量仍然不超过 $b$，而调整后超过上限的部分减少了 $1$。因此可以通过若干次调整使得前 $i$ 个人选的物品满足原限制。



因此只需要对弱化后的问题求解即可。

此时相当于每种物品会在第一个人选择时出现 $b_i$ 个，在接下来每个人选择时再出现 $b_i$ 个，直到到达上界 $c_i$ 个，且物品出现后每个人都可以选择。因此可以前缀和求和每个人选择前出现物品的数量，然后贪心选即可。复杂度在于对 $a_i$ 排序。



一种最小割做法：

考虑问题的网络流模型，则考虑人对应 $n$ 个点，物品对应 $m$ 个点，原点向人连 $a_i$ 流量的边，物品向汇点连 $c_i$ 流量的边，人向物品连物品的 $b_i$ 流量的边。则答案为图的最大流。

考虑这个图的最小割。如果割掉了一些 $a_i$ 边和一些 $c_i$ 边，则剩下需要割掉的中间部分边权值总和为没有被割掉的人数乘以没有被割的物品的 $b_i$ 之和。

考虑枚举剩余人数 $k$，则割 $a_i$ 边的代价为最小的 $n-k$ 个 $a_i$ 之和，剩余部分考虑每个物品，可以发现一个物品的代价为 $\min(c_i,k*b_i)$。预处理后枚举 $k$ 即可。



复杂度均为 $O(n\log n+m)$，但非常奇妙的是，两种做法最后的代码有某种程度的等价性。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,m;
ll a[N],b[N],c[N],su[N],as;
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%lld",&a[i]);
	for(int i=1;i<=m;i++)scanf("%lld",&b[i]);
	for(int i=1;i<=m;i++)scanf("%lld",&c[i]);
	for(int i=1;i<=m;i++)
	{
		ll tp=c[i]/b[i];if(tp>n)tp=n;
		su[n-tp+1]+=b[i]-c[i]%b[i];su[n-tp]+=c[i]%b[i];
	}
	sort(a+1,a+n+1);
	for(int i=1;i<=n;i++)su[i]+=su[i-1];
	for(int i=n;i>=1;i--)
	{
		if(a[i]>su[i])a[i]=su[i];
		as+=a[i];su[i-1]+=su[i]-a[i];
	}
	printf("%lld\n",as);
}
```



##### ARC125F Tree Degree Subset Sum

###### Problem

给一棵 $n$ 个点的树，求有多少对 $(x,y)$ 满足如下条件：

$1\leq x\leq n$，且可以在树中选择 $x$ 个不同节点，使得这些点的度数和为 $y$。

$n\leq 2\times 10^5$

$6s,1024MB$

###### Sol

设度数序列为 $d_{1,\cdots,n}$。考虑将所有 $d_i$ 减去 $1$，对应的将 $(x,y)$ 变为 $(x,y-x)$。此时问题形式不变。

此时 $\sum d_i=n-2$，但由于prufer序的性质，任何一个长度为 $n$，和为 $n-2$ 的序列都可以是序列 $d$，因此此时树不再提供任何额外限制。



由于和很小，因此序列中有大量的 $0$。具体来说，可以对于 $d$ 中每一个大于 $1$ 的元素 $x$，给它配对 $x-1$ 个 $0$。

因此可以猜想对于同一个和，在不同方案之间调整时，可以通过调整对应的 $0$ 使得不同方案中选择的数的数量不出现中断。通过猜想或者暴力计算一些小的情况，可以猜测有如下性质：

对于任意一个 $s$，设选择的数最少的方案需要 $l_s$ 个数，选择的数最多的方案需要 $r_s$ 个数，则对于任意的 $x\in[l_s,r_s]$，存在选择 $x$ 个数的方案使得这些数和为 $s$。

证明：考虑对应所有和为 $s$ 且不选择 $0$ 的方案，设这些方案最少需要 $l$ 个数，最多需要 $r$ 个数。则 $l\leq r\leq s$。而因为每一个大于 $1$ 的数 $x$ 可以和 $x-1$ 个 $0$ 配对，因此至少存在 $s-l$ 个 $0$。

从而至少有 $r-l$ 个 $0$，因此上述结论成立。



因此只需要求出 $l_s,r_s$ 即可。考虑求出 $l_s$，这类似于一个背包问题，使用 $dp$ 的方式，注意到不同的 $d_i$ 只有 $O(\sqrt n)$ 种，考虑对于每一种权值 $dp$，对于一种权值 $d$，相当于如下转移：
$$
l_s=\min_{i=0}^cl_{s-i*d}+i
$$
对于每种权值模 $d$ 的余数部分单调队列优化 $dp$ 即可。复杂度为 $O(n\sqrt n)$。$r_s$ 部分同理。

但还有常数更小的做法。考虑背包的二进制分组做法，一种权值 $d$ 可能二进制分组出 $d,2d,\cdots,2^kd$ 的物品以及最后一个单独的物品，且这些物品的重量和为原来这些点的度数和。

考虑前面部分的物品，不难发现这部分重量不超过 $k$ 的物品数量不超过 $2k$，又因为 $\sum d=n-2$，因此二进制分组出的物品数量不超过 $O(\sqrt n)$。因此这样的复杂度也是 $O(n\sqrt n)$，而不会多出 $\log$。同时这个做法常数小得多。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 205000
#define ll long long
int n,a,b,d[N],ls[N],rs[N],ct[N],st[N],lb,rb,li[N];
ll as;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),d[a]++,d[b]++;
	for(int i=1;i<=n;i++)ct[d[i]-1]++;
	for(int i=1;i<=n;i++)ls[i]=n+1;
	for(int i=1;i<=n;i++)if(ct[i])
	for(int j=0;j<i;j++)
	{
		lb=1,rb=0;
		for(int k=j;k<=n;k+=i)
		{
			if(lb<=rb&&st[lb]+ct[i]*i<k)lb++;
			while(lb<=rb&&ls[st[rb]]+(k-st[rb])/i>=ls[k])rb--;
			st[++rb]=k;
			li[k]=ls[st[lb]]+(k-st[lb])/i;
		}
		for(int k=j;k<=n;k+=i)ls[k]=li[k];
	}
	rs[0]=ct[0];
	for(int i=1;i<=n;i++)if(ct[i])
	for(int j=0;j<i;j++)
	{
		lb=1,rb=0;
		for(int k=j;k<=n;k+=i)
		{
			if(lb<=rb&&st[lb]+ct[i]*i<k)lb++;
			while(lb<=rb&&rs[st[rb]]+(k-st[rb])/i<=rs[k])rb--;
			st[++rb]=k;
			li[k]=rs[st[lb]]+(k-st[lb])/i;
		}
		for(int k=j;k<=n;k+=i)rs[k]=li[k];
	}
	for(int i=0;i<=n;i++)if(rs[i]>=ls[i])as+=rs[i]-ls[i]+1;
	printf("%lld\n",as);
}
```



##### ARC126E Infinite Operations

###### Problem

给定长度为 $n$ 的正整数序列 $a$，定义 $a$ 的权值为如下问题的答案：

你需要通过操作 $a$ 最大化自己的分数，初始分数为 $0$。你可以进行如下操作：

选择 $i\neq j$ 使得 $a_i\leq a_j$，选择非负实数 $x$ 使得 $a_j-a_i\geq 2x$。然后将 $a_i$ 加上 $x$，$a_j$ 减去 $x$，获得 $x$ 的分数。

设 $f_n$ 为进行不超过 $n$ 次操作的最大分数，可以证明 $f_n$ 收敛于一个实数，$a$ 的权值即为 $\lim_{n\to +\infty}f_n$。

现在有 $q$ 次操作，每次操作为修改一个 $a_i$。你需要求出每次修改后序列的权值。

$n,q\leq 3\times 10^5,a_i\leq 10^9$

$5s,1024MB$

###### Sol

不妨设 $a_1\leq a_2\leq\cdots\leq a_n$。

可以发现 $n=2$ 时答案为 $\frac12(a_2-a_1)$，$n=3$ 时答案为 $(a_3-a_1)$。由此猜测答案是一个线性函数，即答案为 $v_{n,1}a_1+\cdots+v_{n,n}a_n(a_1\leq a_2\leq\cdots\leq a_n)$。

答案函数满足任意一次操作后，答案的减少量不能小于等于这次操作增加的分数。考虑操作了 $i,j(i<j)$，则答案减少量为 $x$ 乘以 $v_{n,j}-v_{n,i}$。因此有 $\forall i<j,v_{n,j}\geq v_{n,i}+1$。

又因为所有数相同显然答案为 $0$，因此有 $\sum_i v_{n,i}=0$。

此时可以猜想 $v_{n,i}=i-\frac{n+1}2$。则可以使用如下方式证明这样构造的结果即为答案：

在上述构造下，可以发现操作 $i+1=j$ 的两个数时，答案函数的减少量等于增加的分数，因此这样的操作不会影响答案。

设当前答案为 $s$，则可以发现 $a_n-a_1\geq \frac s{n}$，因此存在相邻两个数差大于等于 $\frac s{n^2}$。而操作这两个数会使得 $s$ 变为至多是之前的 $1-\frac 1{2n^2}$ 倍。因此这样操作一定使得 $s\to 0$。因此存在操作方式达到这个答案。

因此只需要维护 $\sum a_i$ 以及排序后的 $\sum i*a_i$ 即可。一种简单的方式是主席树维护或者离线离散化后树状数组。

复杂度 $O(n\log n)$ 或者 $O(n\log v)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 300500
#define M 10010110
#define mod 998244353
int n,q,v[N],a,b,ct=1,su,ch[M][2];
struct sth{int a,b,c;}vl[M];
sth operator +(sth a,sth b){return (sth){a.a+b.a,(a.b+b.b)%mod,(1ll*b.b*a.a+a.c+b.c)%mod};}
void add(int x,int l,int r,int v,int t)
{
	if(l==r)
	{
		vl[x].a+=t;
		vl[x].b=1ll*vl[x].a*l%mod;
		vl[x].c=1ll*vl[x].a*(vl[x].a+1)%mod*l%mod*(mod+1)/2%mod;
		return;
	}
	int mid=(l+r)>>1,tp=mid<v;
	if(!ch[x][tp])ch[x][tp]=++ct;
	if(mid>=v)add(ch[x][0],l,mid,v,t);
	else add(ch[x][1],mid+1,r,v,t);
	vl[x]=vl[ch[x][0]]+vl[ch[x][1]];
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),add(1,1,1e9,v[i],1),su=(su+v[i])%mod;
	while(q--)
	{
		scanf("%d%d",&a,&b);
		su=(1ll*mod+su+b-v[a])%mod;
		add(1,1,1e9,v[a],-1);add(1,1,1e9,b,1);
		v[a]=b;
		printf("%d\n",(mod+vl[1].c-1ll*su*(n+1)%mod*(mod+1)/2%mod)%mod);
	}
}
```



##### ARC126F Affine Sort

###### Problem

给定一个长度为 $n$ 的序列 $v$，定义 $f_k$ 为满足如下条件的正整数对 $(a,b,c)$ 个数：

1. $1\leq c\leq k,0\leq a,b<k$
2. 令 $x_i=(v_i*a+b)\bmod c$，则 $x$ 严格单调递增：$x_1<x_2<\cdots<x_n$

可以证明 $\lim_{n\to +\infty}\frac{f_n}{n^3}$ 存在且为有理数，求这个极限模 $998244353$ 的结果。

$n\leq 1000,\sum v_i\leq 5\times 10^5$，$v_i$ 两两不同

$5s,1024MB$

###### Sol

设 $g_k$ 表示满足 $c=k$，剩余条件与 $f$ 相同的三元组 $(i,j,k)$ 数量。则考虑 $\lim_{k\to +\infty}\frac{g_k}{k^2}$，如果这个极限存在，设其值为 $c$，则可以发现 $\lim_{n\to +\infty}\frac{f_n}{n^3}=\frac 13c$。

证明：由极限定义，$\forall \epsilon\geq 0,\exists M_\epsilon\geq 0$ 使得 $\forall n\geq M_\epsilon,|\frac{g_k}{k^2}-c|\leq \epsilon$。则：
$$
\forall n\geq M_\epsilon,f_n=\sum_{i=1}^ng_i=g_{M_\epsilon}+\sum_{i=M_\epsilon+1}^ng_i\\
|f_n-c\sum_{i=1}^ni^2|\leq M_\epsilon^3+\sum_{i=M_\epsilon+1}^n|g_i-ci^2|\\
|f_n-c\sum_{i=1}^ni^2|\leq M_\epsilon^3+\sum_{i=M_\epsilon+1}^n\epsilon i^2\\
|\frac{f_n}{\sum_{i=1}^ni^2}-c|\leq \frac{M_\epsilon^3}{\sum_{i=1}^ni^2}+\epsilon*\frac{\sum_{i=M_\epsilon i+1}^ni^2}{\sum_{i=1}^ni^2}
$$
令 $n\to +\infty$，得右端趋于 $\epsilon$。则存在 $M'_\epsilon$，使得 $\forall n\geq M'_\epsilon,|\frac{f_n}{\sum_{i=1}^ni^2}-c|\leq 2\epsilon$。

因此 $\lim_{n\to +\infty}|\frac{f_n}{\sum_{i=1}^ni^2}-c|=0$。又因为 $\sum_{i=1}^ni^2=\frac13 n^3+o(n^3)$，因此原极限为 $\frac 13c$。



考虑一个 $g_k$，如果将 $a,b$ 除以 $k$，则可以看成 $x_i$ 为 $v_i*\frac ak+\frac bk$ 的小数部分。

因此 $\frac {g_k}{k^2}$ 可以看成在 $[0,1)\times[0,1)$ 中所有的 $(\frac ak,\frac bk)$ 中随机选择一对 $(x,y)$，求这对 $(x,y)$ 满足如下条件的概率：

$v_i*x+y$ 的小数部分递增。

则可以猜想，$\lim_{n\to +\infty}\frac{g_k}{k^2}$ 即为在随机选择二元实数对 $(x,y)$，它满足如下条件的概率。但这个证明需要 $(x,y)$ 合法条件的限制，因此这个证明留到最后。



考虑实数对上的问题。考虑对于一个 $x$ 如何确定合法的 $y$。

考虑相邻两个数 $i,i+1$，则要求为 $v_i*x+y$ 的小数部分小于等于 $v_{i+1}*x+y$ 的小数部分。但可以发现对于一个 $x$，两个数的差是定值。因此如果这个条件得到满足，则两个小数部分的差固定。可以发现，在这里这个差为 $(v_{i+1}-v_i)x-\lfloor(v_{i+1}-v_i)x\rfloor$。

因此如果所有的条件都得到满足，则所有小数部分的差固定。因此可以求出第一个小数部分和最后一个小数部分的差 $s$。可以发现 $s$ 一定能被表示为 $(v_n-v_1)x+k$ 的形式，其中 $k$ 为整数。

一种方式满足限制当且仅当通过上面的方式求出小数部分后，最后的小数部分都在 $[0,1)$ 之间。则不难发现，存在合法 $y$ 当且仅当 $s\leq 1$，且随机一个 $y$ 合法的概率为 $1-s$。

可以发现上面的 $k$ 变化的时候即为 $\lfloor(v_{i+1}-v_i)x\rfloor$ 变化的时候。而这样的变化点对于一对相邻的元素只有 $|v_{i+1}-v_i|$ 个，因此总的数量不超过 $O(\sum v_i)$。而对于一段内，$k$ 固定时对 $y$ 的概率为一次函数与 $0$ 取 $\max$ 的结果。因此可以积分求出每一段的贡献，即可得到总的答案。因为上面的分析最后只需要 $k$，因此经过一个分界点时可以 $O(1)$ 更新，从而计算这个值的复杂度为排序复杂度 $O((\sum v_i)\log \sum v_i)$。



然后考虑结论的正确性。继续考虑上面的情况，可以发现一段内部，一个 $y$ 合法当且仅当 $y+v_1*x$ 的小数部分不超过 $1-(v_n-v_1)x-k$。因此每一段内，$y$ 的取值范围形如一个区间 $[l,r]$ 或者两个区间 $[0,r],[l,1)$，且端点都是关于 $x$ 的一次函数。（这里不考虑边界上的问题）

因此合法区域构成 $[0,1)^2$ 上的有限多个多边形。它的边界是零测集，因此可以使用积分的方式求出合法区域的面积。上面最后一部分就完成了这个过程。

再考虑 $g$ 的部分，可以把 $g$ 的过程看成将区域两维分别 $k$ 等分，得到 $k^2$ 个区域，每一个区域内取左下角的值（是否合法），而除以 $\frac 1{k^2}$ 可以看成乘以每一个区域的面积。那么可以发现这是一个黎曼和的形式，且每一块的直径趋于 $0$。又因为积分存在，因此这个极限就等于积分值。



复杂度 $O((\sum v_i)\log\sum v_i)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 1050
#define mod 998244353
int n,v[N],as,su;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct fr{int x,y;};
bool operator <(fr a,fr b){return 1ll*a.x*b.y<1ll*a.y*b.x;}
struct sth{fr f;int b;};
bool operator <(sth a,sth b){return a.f<b.f;}
vector<sth> rs;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)
	{
		int tp=v[i]-v[i+1];
		if(tp>0)for(int j=0;j<tp;j++)rs.push_back((sth){(fr){j,tp},1});
		else for(int j=1;j<=-tp;j++)rs.push_back((sth){(fr){j,-tp},-1});
	}
	rs.push_back((sth){(fr){0,1},0});
	sort(rs.begin(),rs.end());
	rs.push_back((sth){(fr){1,1},0});
	for(int i=0;i+1<rs.size();i++)
	{
		su+=rs[i].b;
		int li=su,st=v[n]-v[1];
		li=1-li;st=-st;
		fr lb=rs[i].f,rb=rs[i+1].f;
		if(st>0&&lb<(fr){-li,st})lb=(fr){-li,st};
		if(st<0&&(fr){li,-st}<rb)rb=(fr){li,-st};
		if(rb<lb)continue;
		int vl=1ll*lb.x*pw(lb.y,mod-2)%mod,vr=1ll*rb.x*pw(rb.y,mod-2)%mod;
		as=(as+1ll*(st+mod)*(mod+1ll*vr*vr%mod-1ll*vl*vl%mod+mod)%mod*(mod+1)/2+1ll*(li+mod)*(vr-vl+mod))%mod;
	}
	printf("%d\n",1ll*as*(mod+1)/3%mod);
}
```



##### ARC127E Priority Queue

###### Problem

给定 $n,m$。考虑一个优先队列。给定一个长度为 $n+m$ 的操作序列，每次操作为插入或者删除，且正好有 $n$ 次插入。你需要按照操作序列对队列进行操作：

1. 如果当前操作为插入，则你需要从 $[1,n]$ 中选择一个正整数插入。且每个正整数必须正好被插入一次。
2. 如果当前操作为删除，则删除队列中最大元素。操作序列保证当前存在一个元素。

考虑最后队列中剩余元素构成的集合 $S$，求不同的 $S$ 数量，模 $998244353$

$n\leq 5000$

$2s,1024MB$

###### Sol

设 $s_i$ 为 $s$ 中第 $i$ 小的数。考虑 $s$ 的性质，可以发现如下结论：

设 $v_i$ 为最后一次使得当前队列中元素操作小于 $i$ 时，这次操作前插入队列的元素数量。则一定有 $s_i\leq v_i+1$。

考虑这一次操作，这次操作后队列里面有 $i-1$ 个数，之后的操作不会使得队列内元素个数少于 $i$。因此之后队列里面第 $i$ 小的数即为这次操作前的 $i-1$ 个数与之后插入的所有数中第 $i$ 小的数，从而最后第 $i$ 小的数不会大于最后插入的数中最小的数，而后面有 $n-v_i$ 个数，因此有如上结论。



上面给出了一个 $S$ 的一个必要条件，但可以发现这也是充分条件，即：

如果 $s$ 满足 $s_1<s_2<\cdots<s_{n-m}$ 且 $s_i\leq v_i+1$，则这个 $s$ 可以是最后的优先队列。

考虑把操作分成若干段：$[v_1,v_2),[v_2,v_3),\cdots$。则由上面的分析，$s$ 可以使用如下方式构造：

在 $[v_1,n+m]$ 的操作中，找到插入的最小的数，这个数即为 $s_1$。

随后删去这个数，在 $[v_2,n+m]$ 中找到剩下的插入的最小的数，这个数即为 $s_2$。剩下部分重复这个过程。

因此考虑先把 $s_i$ 放到对应段里面的插入操作，然后在 $[v_1,v_2]$ 里面剩下的插入操作中填没有被插入的大于 $s_1$ 的数，然后在 $[v_2,v_3]$ 里面剩下的插入操作中填没有被插入的大于 $s_2$ 的数。

而对于每一个 $i$，在后 $i$ 段需要填 $n-v_i+1$ 个数（加上所有 $s$），而可以选的数有 $n-s_i+1$ 个数。因此如果贪心填的时候不考虑第 $i$ 段必须大于 $s_i$，而是看成所有段都必须大于 $s_1$，则存在合法方案。而如果填到某个段需要考虑这个问题，此时前面填的最大数小于 $s_i$，因此后面可以看成一个独立的问题。由归纳可得贪心可以构造出合法方案。



因此上述条件充分必要。只需要统计满足条件的序列 $s_{1,\cdots,n-m}$。dp+前缀和统计即可。

复杂度 $O(n^2)$

###### Code

~~因为c++的pq，下面是当大根堆做的~~

```cpp
#include<cstdio>
using namespace std;
#define N 5050
#define mod 998244353
int n,m,a,s0,s1,li[N],dp[N][N];
int main()
{
	scanf("%d%d",&n,&m);
	li[0]=n;
	for(int i=1;i<=n+m;i++)
	{
		scanf("%d",&a);
		if(a==1)s0++;else s1++;
		li[s0-s1]=n-s0;
	}
	for(int i=li[0];i<=n;i++)dp[0][i]=1;
	for(int i=1;i<=n-m;i++)
	for(int j=n;j>=li[i];j--)
	dp[i][j]=(dp[i-1][j+1]+dp[i][j+1])%mod;
	printf("%d\n",dp[n-m][0]);
}
```



##### ARC127F ±AB

###### Problem

给定正整数 $a,b,m$ 和非负整数 $x_0$，有一个整数 $x$，初始 $x=x_0$。

你可以对 $x$ 进行如下操作，但任意时刻 $x$ 必须在 $[0,m]$ 之间：

1. 将 $x$ 加上或者减去 $a$
2. 将 $x$ 加上或者减去 $b$

操作可以以任意顺序执行任意次，求可以得到的不同 $x$ 种类数。

多组数据，$T\leq 10^5,1\leq a<b\leq m\leq 10^9$，$a,b$ 互质。

###### Sol

考虑一个策略：能减 $a$ 就减 $a$，否则加 $b$。

则可以发现如果 $m+1\geq a+b$，这样一定可以循环，又因为 $a,b$ 互质，因此这样可以经过 $[0,a+b-1]$ 的所有数，从而可以经过 $[0,m]$ 的所有数。因此这种情况答案为 $m+1$。



考虑 $m+1<a+b$ 的情况，此时可以发现每个 $x$ 最多只有两种可能的操作。因此如果将状态和操作看成一个图，则图由若干条链或者环组成。

因此考虑从初始状态开始，选择一个初始操作方向，然后接下来每一步操作不能选择上一次的逆操作，直到不能操作为止。记录这个方向操作的次数，再记录另外一个方向操作的次数，如果当前点所在的是一条链，则这样可以求出可以经过的点数。

可以发现两种方向分别为如下操作：

1. 能减 $a$ 就减 $a$，否则加 $b$。

2. 能减 $b$ 就减 $b$，否则加 $a$。

考虑第一种操作过程，不存在合法操作当且仅当当前数小于 $a$ 且大于 $m-b$。从 $\bmod a$ 角度考虑，可以发现这相当于找到最小的 $t\geq 0$ 使得 $(x_0+b*t)\bmod a\in[m-b,a-1]$。

因为 $a,b$ 互质且 $a+b>m+1$，这样的 $t$ 一定存在，因此这个操作一定会结束，从而一定不存在环。因此对于两个方向分别找到 $t$，即可求出两个方向的操作次数并得到答案。



因此最后的问题为，给定若干组 $a,b,c,l,r$，求最小的非负整数 $t$ 使得 $(at+b)\bmod c\in[l,r]$。

显然 $a,b$ 可以对 $c$ 取模。考虑类欧的思路，如果能做到交换 $a,c$，则通过取模操作，问题可以在 $O(\log m)$ 时间内解决。

可以发现上述式子成立当且仅当存在非负整数 $x$ 使得 $at+b\in[cx+l,cx+r]$，而这个式子等价于 $cx\in[at+b-r,at+b-l]$。

此时 $b-r,b-l$ 可能是负数，但它大于 $-c$，因此解出的 $x$ 一定大于等于 $0$。因此可以找到一个非负整数 $k=\lceil\frac {r-b}a\rceil$，将限制变为 $cx\in[at+ak+b-r,at+ak+b-l]$，且满足此时 $t\geq 0$ 时的最小合法 $x$ 为原问题答案同时 $ak+b-r\geq 0$。

而最小的 $t$ 对应最小的 $x$，因此问题已经变为了类似形式。此时 $ak+b-r<a$，因此如果 $ak+b-l<a$，而问题变为与上面相同的形式。否则，存在 $v\in[l,r]$ 使得 $at+ak+b-v=a$，即 $a(t+k-1)+b=v$。而可以发现，此时在原问题中 $x=0$ 存在合法 $t$，而这即为最小 $t$。因此这种情况可以直接求出答案。对于之前的另外一种情况递归求即可。

可以发现这样的复杂度与gcd相同，为 $O(\log m)$。

复杂度 $O(T\log m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
int T,a,b,v,m;
//ax+b in [cy+l,cy+r]
//cy in [ax+b-r,ax+b-l]
int calc(int a,int b,int c,int l,int r)
{
	a%=c;b%=c;l%=c;r%=c;
	if(b>=l&&b<=r)return 0;
	if(b<l&&(r-b)/a>(l-1-b)/a)return (l-1-b)/a+1;
	int li=calc(c,0,a,b+a-r%a,b+a-l%a);
	return (1ll*li*c+l-1-b)/a+1;
}
void solve()
{
	scanf("%d%d%d%d",&a,&b,&v,&m);
	if(a+b<=m+1){printf("%d\n",m+1);return;}
	v%=a;
	int as=0;
	int li=calc(a,v,b,m-a+1,b-1);
	as=li+(1ll*li*a+v)/b;
	li=calc(b,v,a,m-b+1,a-1);
	as+=li+(1ll*li*b+v)/a;
	printf("%d\n",as+1);
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```



##### ARC128E K Different Values

###### Problem

有 $n$ 种数，数为 $1,2,\cdots,n$，数 $i$ 有 $a_i$ 个。

给定正整数 $k$，你需要将所有数按照某种顺序排成一列 $x$，$x$ 需要满足如下限制：

$x$ 中任意相邻的 $k$ 个数两两不同。

判断是否存在解，如果存在解则求出字典序最小的解。

$n\leq 500,\sum a_i\leq 2\times 10^5$

$2s,1024MB$

###### Sol

考虑依次填每个位置，可以发现相当于需要判定如下问题：

当前已经填了 $x$ 的一个前缀，给出每种数剩余的数量 $a_i'$，判断是否有解。



记 $s=\sum a_i'$，即剩余部分的数个数。则显然有如下必要条件：

1. 因为每一种数相邻两次出现距离大于等于 $k$，因此可以发现 $a_i'$ 不能超过 $\lfloor\frac{s+k-1}k\rfloor$。
2. 记 $r=s-k*\lfloor\frac{s+k-1}k\rfloor$，在上一条的基础上，可以发现如果 $a_i'=\lfloor\frac{s+k-1}k\rfloor$，则 $i$ 必须在剩余部分的前 $r$ 个位置出现。因此满足这个条件的 $i$ 最多有 $r$ 个。
3. 再考虑已经填的数，对于之前填的最后 $k-r$ 个数，它们不能在前 $r$ 个位置出现，因此这些出现的 $i$ 必须满足 $a_i'<\lfloor\frac{s+k-1}k\rfloor$。

考虑如果满足这三个条件，如何构造接下来的序列：

考虑满足 $a_i'=\lfloor\frac{s+k-1}k\rfloor$ 的 $i$ 数量。如果这个数量等于 $r$，则选择满足这一条件的任意一个 $i$ 填，由第三条这样的选择一定满足原要求，考虑这种情况下，上述三个条件是否得到满足。

如果接下来上界减少，则此时 $r=1$，此时给这个数减一后，所有数都小于之前的上界。如果 $r>1$，则上界不变。因此第一条得到满足。

对于第二条，如果 $r=1$，则接下来上界减一，$r=k$，因此一定满足要求。否则，$r$ 减少一，但选择了一个满足条件的数，因此满足条件的数减一。所以第二条也得到满足。

对于第三条，上界减少后 $r=k$，此时不存在这样的限制。对于剩余的情况，现在的 $k-r$ 个数为之前的这些数加上这次选择的数。之前的数显然满足条件，可以发现这次选择的数剩余次数变为 $\lfloor\frac{s+k-1}k\rfloor-1$，因此满足条件。

因此这种情况下，选择后三个条件得到满足。

可以发现，如果这个数量小于 $r$，则任意选择一个可以填的数填，只有第二条可能受到影响，但因为之前数量没有达到上界，因此满足条件。

因此无论如何，可以选择一个数使得选择后三个条件都得到满足，且选择满足原要求。而剩余 $0$ 个数时一定合法。因此可以得到如下结论：

存在合法方案当且仅当满足上述要求。



再考虑上面的操作。如果达到上界的数数量小于 $r$，则可以任意选择一个填，因此选择最小的可以填的数即可。

如果数量等于 $r$，则可以发现如果选择一个没有达到上界的数，则选择后不满足第二个条件或第一个条件。因此此时只能选择一个等于上界的数。

可以发现，维护之前选择的最后 $k$ 个数以及每种数的剩余个数，则每一步可以 $O(n)$ 维护，因此复杂度为 $O(n\sum a_i)$。



事实上上面的前两条限制保证了操作后满足第三条限制，而第三条限制保证了可以进行操作。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505
int n,k,su,v[N],ls[N],is[N];
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su+=v[i];
	int ci=0,fg=0;
	for(int i=1;i<=n;i++)if(v[i]>su/k+1)fg=1;else if(v[i]==su/k+1)ci++;
	if(fg||ci>su%k){printf("-1\n");return 0;}
	while(su)
	{
		for(int i=1;i<=n;i++)is[i]=0;
		for(int i=1;i<k;i++)is[ls[i]]=1;
		int c1=0;
		for(int i=1;i<=n;i++)c1+=v[i]==su/k+1;
		int fg=c1==su%k&&c1>0,fr=0;
		for(int i=n;i>=1;i--)if(!is[i]&&(!fg||v[i]==su/k+1)&&v[i])fr=i;
		printf("%d ",fr);v[fr]--;
		for(int i=1;i<k;i++)ls[i]=ls[i+1];ls[k-1]=fr;
		su--;
	}
}
```



##### ARC128F Game against Robot

###### Problem

有 $n$ 个数，第 $i$ 个数为 $a_i$。保证 $n$ 为偶数。

两个人进行如下游戏：

游戏开始前，双方决定一个 $n$ 阶排列 $p$，双方能知道 $p$。

接下来双方轮流操作，第一个人先手：

轮到第一个人时，他可以选择一个还没有被选择过的数，获得这个数值的分数。

轮到第二个人时，这个人会找到最大的 $i$ 使得第 $p_i$ 个数还没有被选，然后选择第 $p_i$ 个数。

第一个人会最大化自己的分数。定义一个排列 $p$ 的权值为以这个排列开始时，最后第一个人的分数。求所有 $n!$ 个排列的权值和，模 $998244353$。

$n\leq 10^6$

$2s,1024MB$

###### Sol

考虑 $p=(1,2,\cdots,n)$ 的情况，考虑有哪些大小为 $\frac n2$ 的数集合可以被第一个人选择到。

因为第二个人每次选择最后一个数，因此一个集合 $S$ 合法，则最后两个数中 $S$ 最多选择一个，类似可得最后 $2k$ 个数中最多选择 $k$ 个。

同时可以发现，如果 $S$ 满足这一限制，则先手每次选择 $S$ 中最后一个数，则可以选择到这些数。因此 $S$ 合法的充分必要条件为对于任意正整数 $k$，后 $2k$ 个数中最多选择 $k$ 个。



然后考虑一个 $p$ 如何求最大值。有如下结论：

$\{S\}$ 构成一个拟阵。

考虑一个合法集合 $S$，考虑有哪些元素使得加入后不合法，设这些元素构成集合 $R_S$。则可以发现，找到最大的 $k$ 使得最后 $2k$ 个数中选择了 $k$ 个，则 $R_S$ 即为最后 $2k$ 个数中没有被选择的部分，而之前的数加入都是合法的。则 $R_s\cup S$ 中包含了最后 $2k$ 个数以及之前属于 $S$ 的数。从而对于任意一个 $R_s\cup S$ 的独立子集 $T$，$T$ 中最后 $2k$ 个数最多选择 $k$ 个，因此 $|T|\leq |S|$。

由此可以说明 $\{S\}$ 满足扩充性质，因此构成拟阵。



因此可以使用如下方式求最大值：按照权值从大到小尝试加入数，如果能加入就加入。最后得到的即为最大值。

~~如果从加入数的角度考虑，会导致自闭~~

从类似秩函数的角度考虑，对于任意 $i$，设权值最大的 $i$ 个数中最多可以选择 $r_i$ 个，则拟阵性质保证存在一种方案，使得对于任意 $i$，权值最大的 $i$ 个数中选择了 $r_i$ 个。因此设权值从大到小排序后为 $v_{1,\cdots,n}$，则这种情况的最优答案为 $\sum_i r_i*(v_i-v_{i+1})$。



考虑求 $r_i$。即当前有若干个数，求这些数中最多能选出多少个使得选出数满足限制。

考虑将选择一个数看成 $-1$，不选择看成 $+1$，这样得到一个序列。则最后 $2k$ 个数中最多选择 $k$ 个相当于对于任意 $2k$，序列最后 $2k$ 个数的和大于 $0$。可以发现这当且仅当序列的最小后缀和大于等于 $-1$。

将大于等于给定值的数看作 $-1$，剩下数变为 $+1$，则问题相当于选择尽量少的 $-1$ 变成 $+1$，使得序列最小后缀和大于等于 $-1$。可以发现每次改变最小后缀和最多增加 $2$，且每次选择最后一个 $-1$ 就可以达到这个值。因此设最小后缀和为 $s$，则最少需要不选的数为 $\lfloor\frac{-s}2\rfloor$。



回到原上面对于一个 $r_i$ 的问题，最后考虑的为 $-1,+1$ 的序列。因此回到原问题，对于所有的排列 $p$ 考虑求和不选的数的数量，则可以看成如下问题：

对于所有由 $i$ 个 $-1$，$n-i$ 个 $+1$ 构成的序列，设序列最小后缀和为 $s$，求和 $\lfloor\frac{-s}2\rfloor$。

则所有排列的 $r_i$ 之和即为 $n!*i$ 减去 $i!(n-i)!$ 乘以上述问题的答案。显然最小后缀和翻转后即为最小前缀和，因此可以考虑翻转后的问题。

根据线性性，对于正整数 $k$ 求和最小前缀和小于等于 $-2k$ 的方案数即可得到答案。如果 $n-2i\leq -2k$，则所有方案都满足这一条件。否则考虑翻折，可以在第一次经过 $y=-2k$ 处进行翻折，翻折后可以发现相当于结束时走到 $2i-4k-n$ 的方案数，即 $C_{n}^{2k-i}$。（对于终点在线下面的情况不能直接翻折，否则会多考虑翻折后不经过线的情况）。

可以发现对于一个 $i$ 需要求形如 $\sum_{i=0}^kC_n^{t+2i}$ 的结果，前缀和即可。求出 $r_i$ 即可得到答案。



复杂度 $O(n\log n)$（只有排序部分不是线性）

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1005000
#define mod 998244353
int n,fr[N],ifr[N],dp[N],v[N],as,su[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int i=0;i<=n;i++)su[i+1]=1ll*fr[n]*ifr[i]%mod*ifr[n-i]%mod;
	for(int i=2;i<=n;i++)su[i+1]=(su[i+1]+su[i-1])%mod;
	for(int i=1;i<=n/2;i++)dp[i]=(1ll*fr[n]*ifr[i]%mod*ifr[n-i]%mod*i+mod-su[i-1])%mod;
	for(int i=1;i<=n/2;i++)dp[i]=1ll*dp[i]*fr[n-i]%mod*fr[i]%mod;
	for(int i=n/2;i>=1;i--)dp[i]=(dp[i]+mod-dp[i-1])%mod;
	for(int i=n/2+1;i<=n;i++)dp[i]=(dp[1]+mod-dp[n+1-i])%mod;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	for(int i=1;i<=n;i++)as=(as+1ll*v[n-i+1]*dp[i])%mod;
	printf("%d\n",as);
}
```



##### ARC129E Yet Another Minimization

###### Problem

你需要构造一个长度为 $n$ 的序列 $x$。

序列中每个元素有 $m$ 种选择，对于第 $i$ 个数，它的第 $k$ 个选择为 $a_{i,k}$，而选择 $a_{i,k}$ 有 $c_{i,k}$ 的代价。

给定 $w_{i,j}$，在你构造了 $x$ 之后，会再产生如下代价：
$$
\sum_{1\leq i<j\leq n}w_{i,j}|x_i-x_j|
$$
你需要最小化两部分产生的代价总和，求这个最小值。

$n\leq 50,m\leq 5$

$2s,1024MB$

###### Sol

考虑最小割模型，则每个点选择一个数可以被如下构造描述：

设权值上限为 $v$，考虑一条 $v+1$ 个点的链，割掉边 $v\to v+1$ 表示选择 $v$ 的权值，因此 $v\to v+1$ 的流量为选这种值的代价。再加入边 $v+1\to v$，权值为 $+\infty$，加入这些边可以保证每条链上只被割一条边。

然后考虑如何描述第二部分代价：

一对点之间的代价为 $w_{i,j}|x_i-x_j|$。考虑对于每一个 $v$，在两个点分别链上 $v$ 对应的点间双向连权值为 $w_{i,j}$ 的边。这样如果两个点分别选择权值 $a,b(a\leq b)$，则在链中间需要割的部分为 $[a+1,b]$ 部分某个方向的边，这个代价即为 $w_{i,j}|x_i-x_j|$。



然后考虑缩点，对于一个点，考虑按照给定的 $m$ 个数将这条链分段，则每一段内部对于任意的 $(i,i+1)$，有边 $i\to i+1,i+1\to i$ 且它们权值都是 $+\infty$，因此这一段内部可以合并，两侧可能合并到原点汇点上。

考虑合并后如何处理不同链之间的代价。注意到此时每个点对应链上一个区间，可以发现两个点之间缩点前连的边数为两个区间的交大小，因此现在连区间交大小乘以 $w_{i,j}$ 流量的边即可。

最后答案即为得到的图的最小割，最大流即可。

图中点数为 $O(nm)$，边数为 $O(n^2m)$，因此理论复杂度 $O(n^4m^3)$~~但是又一次飞快~~

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 254
#define M 7
#define ll long long
int n,m,su,head[N],cnt=1,dis[N],cur[N],v[N][M],id[N][M],a;
ll c[N][M],as;
struct edge{int t,next;ll v;}ed[N*N*2];
void adde(int f,int t,ll v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;
}
bool bfs(int s,int t)
{
	queue<int> qu;
	for(int i=1;i<=su;i++)dis[i]=-1,cur[i]=head[i];
	dis[s]=0;qu.push(s);
	while(!qu.empty())
	{
		int u=qu.front();qu.pop();
		for(int i=head[u];i;i=ed[i].next)if(dis[ed[i].t]==-1&&ed[i].v)
		{
			dis[ed[i].t]=dis[u]+1;qu.push(ed[i].t);
			if(ed[i].t==t)return 1;
		}
	}
	return 0;
}
ll dfs(int u,int t,ll f)
{
	if(u==t||!f)return f;
	ll as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)if(dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		as+=tp;f-=tp;
		ed[i].v-=tp;ed[i^1].v+=tp;
		if(!f)return as;
	}
	return as;
}
int main()
{
	scanf("%d%d",&n,&m);
	su=n*(m-1)+2;
	for(int i=1;i<=n;i++)
	for(int j=0;j<=m;j++)
	if(j==0)id[i][j]=su-1;
	else if(j==m)id[i][j]=su;
	else id[i][j]=(i-1)*(m-1)+j;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)
	{
		scanf("%d%lld",&v[i][j],&c[i][j]);
		int f=id[i][j-1],t=id[i][j];
		adde(f,t,c[i][j]),adde(t,f,1e18);
	}
	for(int i=1;i<=n;i++)v[i][m+1]=1e6;
	for(int i=1;i<=n;i++)for(int j=i+1;j<=n;j++)
	{
		scanf("%d",&a);
		for(int p=0;p<=m;p++)for(int q=0;q<=m;q++)
		{
			int lb=max(v[i][p],v[j][q])+1,rb=min(v[i][p+1],v[j][q+1]);
			int f=id[i][p],t=id[j][q];
			if(lb<=rb&&f!=t)adde(f,t,1ll*(rb-lb+1)*a),adde(t,f,1ll*(rb-lb+1)*a);
		}
	}
	while(bfs(su-1,su))as+=dfs(su-1,su,1e18);
	printf("%lld\n",as);
}
```



##### ARC129F Let's Play Tag

###### Problem

数轴上有 $n+m$ 个人，其中有 $n$ 个人在原点左侧，他们与原点的距离为 $l_{1,\cdots,n}$，有 $m$ 个人在原点右侧，他们与原点的距离为 $r_{1,\cdots,m}$。$l,r$ 严格递增。

你在数轴原点，你希望抓到所有人。你会使用如下策略：

1. 选择一个由 $n$ 个 `L`，$m$ 个 `R` 组成的序列 $s$。
2. 依次考虑 $s$ 中每个元素，如果当前元素为 `L`，则向左侧移动直到抓到左侧第一个当前没有被抓到的人。否则向右侧进行类似的操作。

你每秒可以移动 $2$ 的距离，但是所有人都会以每秒 $1$ 单位距离的速度朝远离你的方向移动。

对于所有可能的序列 $s$，求和按照这个序列行动，你抓到所有人需要的时间。答案对 $998244353$ 取模。

$n,m\leq 2.5\times 10^5$

$8s,1024MB$

###### Sol

连续一串 `L` 和忽略这串 `L` 中间的人直接去抓最后一个人在时间上等价。考虑只保留那些满足抓到这个人后会转向的人和最后一个人，即序列中最后一个元素以及所有满足 $s_i\neq s_{i+1}$ 的 $i$。再记录抓这些人的顺序，考虑顺序需要满足的条件，则有如下条件：

1. 顺序中相邻的两个人不能属于同一侧。
2. 对于同一侧的人，他们在序列中出现的顺序一定是按照原先的顺序。
3. 每一侧的最后一个人必须出现。

可以发现满足这个条件的顺序一定对应合法方案。且这是一个一一对应。因此可以变为对于所有满足条件的序列求和时间。



对于这样的顺序，可以看成串为 `LRLR...` 时的问题，因此考虑求出这种情况的时间。

此时每抓一个人后一定会回到原点。考虑将操作分成若干步，每一步为从原点出发抓到某个人再回到原点。可以发现，如果当前在原点，之前用的时间为 $t$，这次要抓的人初始距离为 $d$，则他当前在 $t+d$ 位置。因此抓到他并回来需要 $2t+2d$ 时间，即这次操作后时间变为 $3t+2d$。但最后一次抓到人后不需要回来，此时为 $2t+d$。

如果不考虑最后一次的问题，则时间为 $\sum_{i=1}^k 2*3^{k-i}*d_i$。其中 $k$ 为此时串的长度。如果考虑最后一个人，则可以看成将这个时间乘以 $\frac 23$，再减去 $\frac13d_k$。这可以留到最后完成。



按照顺序，抓保留下来的人的顺序一定是一左一右交替，因此在一侧可以看成求如下值：

对于每一个 $k$，考虑在 $l_1,\cdots,l_n$ 中选择一个长度为 $k$ 且必须包含结尾的子序列的所有方式，定义一种选出子序列 $s$ 的权值为 $\sum_{i=1}^k2*9^{k-i}$ ，求和所有子序列的权值。

因为交替出现，因此每相邻两个之间会乘两次 $3$，因此为上述形式。如果求出了上述对于每个 $k$ 的答案，考虑枚举 $s_1,s_{n+m}$，再枚举一侧留下的人的数量，此时另外一侧留下的人数量确定，通过上述值即可求出这种情况下 $\sum_{i=1}^k 2*3^{k-i}*d_i$ 的和，再处理结尾即可得到时间。

因此只需要求出上述问题的答案即可得到原问题答案。

一种直接的做法是考虑分治，对于分治的每一段求出这一段内选择 $0,1,\cdots,k$ 个数时的权值和，使用fft即可合并。复杂度 $O(n\log^2 n)$，可以通过。

考虑先忽略最后一个数，前面的数没有选择的限制，则可以发现前面部分对选数没有限制。考虑前面部分一个数 $l_i$ 的贡献，枚举两侧选的数的情况，可以发现它的贡献写成生成函数为如下形式：
$$
l_i(1+x)^{i-1}(1+9x)^{n-i}
$$
需要求的即上式对于 $i$ 求和。考虑先换元 $t=x+1$，则第一部分变成单项式，此时的形式与二项式反演相同，可以直接计算，最后换元回来的形式也与二项式反演相同。因此可以 $O(n\log n)$ 求出这个生成函数，最后处理最后一个元素即可。

还有一种做法是将这部分和对答案的贡献一起考虑，最后可以得到贡献系数为一个fft可以求出的形式。

###### Code

~~做法2的实现留到133f~~

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 263501
#define mod 998244353
int n,m,v[N],as,v1,v2;
int fr[N],ifr[N],rev[N*2],gr[2][N*2],pr[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d=18)
{
	pr[0]=1;for(int i=1;i<=1<<d;i++)pr[i]=9ll*pr[i-1]%mod;
	fr[0]=1;for(int i=1;i<=1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1<<d]=pw(fr[1<<d],mod-2);for(int i=1<<d;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int l=2;l<=1<<d;l<<=1)for(int i=1;i<l;i++)rev[l+i]=(rev[l+(i>>1)]>>1)|((i&1)*(l>>1));
	for(int t=0;t<2;t++)
	for(int l=2;l<=1<<d;l<<=1)
	{
		int tp=pw(3,(mod-1)/l),vl=1;
		if(!t)tp=pw(tp,mod-2);
		for(int i=0;i<l;i++)gr[t][i+l]=vl,vl=1ll*vl*tp%mod;
	}
}
int f[N],g[N],ntt[N];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=0;j<l>>1;j++)
	{
		int v1=ntt[i+j],v2=1ll*ntt[i+j+(l>>1)]*gr[t][l+j]%mod;
		ntt[i+j]=(v1+v2)%mod;
		ntt[i+j+(l>>1)]=(v1+mod-v2)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	int sa=a.size(),sb=b.size();
	vector<int> as;
	for(int i=0;i<sa||i<sb;i++)
	as.push_back(((i<sa?a[i]:0)+(i<sb?b[i]:0))%mod);
	return as;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int sa=a.size(),sb=b.size();
	int l=1;while(l<=sa+sb)l<<=1;
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<sa;i++)f[i]=a[i];
	for(int i=0;i<sb;i++)g[i]=b[i];
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	vector<int> as;for(int i=0;i<sa+sb-1;i++)as.push_back(f[i]);
	return as;
}
vector<int> solve(int l,int r,int f)
{
	if(l==r)
	{
		vector<int> as;as.resize(2);
		as[1]=2*v[l]%mod;
		return as;
	}
	int mid=(l+r)>>1;
	vector<int> s1=solve(l,mid,0),s2=solve(mid+1,r,f);
	vector<int> t1,t2;
	if(f)t1.push_back(0);
	for(int i=f;i<=r-mid;i++)t1.push_back(1ll*fr[r-mid-f]*ifr[i-f]%mod*ifr[r-mid-i]%mod*pr[i]%mod);
	for(int i=0;i<=mid-l+1;i++)t2.push_back(1ll*fr[mid-l+1]*ifr[i]%mod*ifr[mid-l+1-i]%mod);
	return polyadd(polymul(t1,s1),polymul(t2,s2));
}
vector<int> s1,s2;
int main()
{
	scanf("%d%d",&n,&m);
	init();
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);v1=v[n];s1=solve(1,n,1);
	for(int i=1;i<=m;i++)scanf("%d",&v[i]);v2=v[m];s2=solve(1,m,1);
	for(int i=1;i<=n;i++)for(int j=i-1;j<=i;j++)if(j&&j<=m)
	{
		int c1=1ll*fr[m-1]*ifr[m-j]%mod*ifr[j-1]%mod,c2=1ll*fr[n-1]*ifr[n-i]%mod*ifr[i-1]%mod;
		as=(as+(1ll*s1[i]*c1+1ll*s2[j]*c2*3)%mod*(mod*2+2)/3%mod+mod-1ll*c1*c2%mod*v1%mod*(mod+1)/3%mod)%mod;
	}
	for(int i=1;i<=m;i++)for(int j=i-1;j<=i;j++)if(j&&j<=n)
	{
		int c1=1ll*fr[m-1]*ifr[m-i]%mod*ifr[i-1]%mod,c2=1ll*fr[n-1]*ifr[n-j]%mod*ifr[j-1]%mod;
		as=(as+(1ll*s1[j]*c1*3+1ll*s2[i]*c2)%mod*(mod*2+2)/3%mod+mod-1ll*c1*c2%mod*v2%mod*(mod+1)/3%mod)%mod;
	}
	printf("%d\n",as);
}
```



##### ARC130E Increasing Minimum

###### Problem

有一个长度为 $n$ 的正整数序列 $a_i$，对其进行 $k$ 次操作：

每次操作为，选择所有 $a_i$ 中最小的一个（有多个可以任意选择），记录选择的下标 $i$，然后将 $a_i$ 加一。

现在给出记录的序列 $i_{1,\cdots,k}$。求是否存在一个 $a_{1,\cdots,n}$ 使得它可以构造出记录序列 $i$。如果可以，则求出字典序最小的 $a$。

$n,k\leq 3\times 10^5$

###### Sol

考虑如何判断序列是否合法。如果从正向看操作过程，则记录序列一定由如下方式构造：

从小到大考虑每个 $v$，对于一个 $v$，考虑最小值从 $v$ 到 $v+1$ 的过程，可以发现这部分的操作序列为将所有初始时小于等于 $v$ 的数的下标按照某种顺序记录。

因此如果操作无限进行下去，最后的操作序列一定可以被划分成若干段，满足如下条件：

1. 每一段内不出现重复元素。
2. 每一段的元素构成的集合是下一段元素构成集合的子集。



考虑在 $k$ 次操作后，所有数相等的情况。此时可以发现一定可以将序列 $i$ 划分成若干段满足上述限制。

考虑划分的最后一段。可以发现如果一个元素 $x$ 在序列中出现，则 $x$ 必须在划分的最后一段中出现。因此划分的最后一段的长度一定为前面部分出现的元素种类数。

因此可以从后往前进行划分，可以发现如果存在划分方案，则唯一存在。且可以发现划分了最后一段后，前面部分为一个子问题，因此可以考虑 $dp$。

设 $dp_i$ 表示 $[1,i]$ 是否存在合法划分方案，则 dp 过程中，记录当前出现的数的种类数。另外一个条件可以看成最后这一段中每种数最多出现一次，因此记录每种数上次出现的位置，即可得到每种数倒数第二次出现的位置的最大值，即可处理这个限制。因此可以 $O(n+k)$ 完成 $dp$。



考虑一般的情况，可以发现在最优解中，序列中一定存在一个 $x$ 使得前 $x$ 次操作后所有数相等。（如果存在两个相同的数则显然必须存在一个划分点，否则显然全 $1$ 最优）。

考虑这个 $x$ 满足的条件，它需要满足：

1. $[1,x]$ 存在合法划分方案。
2. $[x+1,n]$ 不存在重复元素。

考虑这种情况如何构造方案，可以发现最后的数不影响操作，只需要考虑前面部分。

对于前面部分，设 $i$ 出现了 $c_i$ 次，则因为初始数是正整数，因此现在每个数至少需要是 $1+\max c_i$。因此可以发现最优解为让 $a_i=1+(\max_j c_j)-c_i$ 。

而因为 $[x+1,n]$ 中不能存在重复元素，因此对于所有可能的 $x$， $\max_j c_j$ 只有两种取值可能。对于同一种取值可能内部，可以发现 $c_i$ 越大字典序越小，因此 $x$ 越大越优秀。因此对于每一种取值考虑这种取值部分最大的合法 $x$，比较两个 $x$ 得到的字符串即可。

复杂度 $O(n+k)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 300500
int n,m,v[N],as[N],s1[N];
int ls[N],mx,is[N],ci;
int ct[N],mi,cr,lb,v1=-1,v2=-1;
void solve(int x)
{
	if(x==-1)return;
	for(int i=1;i<=n;i++)s1[i]=0;
	for(int i=1;i<=x;i++)s1[v[i]]++;
	int m2=0;
	for(int i=1;i<=n;i++)if(s1[i]>m2)m2=s1[i];
	for(int i=1;i<=n;i++)s1[i]=m2+1-s1[i];
	for(int i=1;i<=n;i++)if(as[i]<s1[i])return;else if(s1[i]<as[i])break;
	for(int i=1;i<=n;i++)as[i]=s1[i];
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d",&v[i]);
	as[1]=1e9;
	is[0]=1;
	for(int i=1;i<=m;i++)
	{
		if(!ls[v[i]])ci++;
		else if(mx<ls[v[i]])mx=ls[v[i]];
		ls[v[i]]=i;
		if(i-ci>=mx&&is[i-ci])is[i]=1;
	}
	for(int i=1;i<=m;i++)ct[v[i]]++;
	for(int i=1;i<=n;i++)if(ct[i]>mi)mi=ct[i],cr=1;else if(ct[i]==mi)cr++;
	for(int i=m;i>=1;i--)if(ct[v[i]]==mi)
	{
		cr--;
		if(!cr){lb=i-1;break;}
	}
	for(int i=mx;i<=lb;i++)if(is[i])v1=i;
	for(int i=lb+1;i<=m;i++)if(i>=mx)if(is[i])v2=i;
	solve(v1);solve(v2);
	if(as[1]>1e8)printf("-1\n");
	else for(int i=1;i<=n;i++)printf("%d ",as[i]);
}
```



##### ARC130F Replace by Average

###### Problem

给一个长度为 $n$ 的正整数序列 $a$，你可以进行如下操作：

选择 $i<j<k$ 满足 $j-i=k-j$，然后将 $a_j$ 变为 $\lfloor\frac{a_i+a_k}2\rfloor$。

你可以进行任意多次操作，你需要最小化最后的 $\sum a_i$。求这个最小值。

$n\leq 3\times 10^5$

$2s,1024MB$

###### Sol

考虑一些简单的情况。例如 $a_1=0,a_n=k(k<n),a_{2,\cdots,n-1}=M$，其中 $M$ 为大于等于 $k$ 的整数。分析可以发现如下性质：

最后的最优解为 $0,0,\cdots,0,1,2,\cdots,k$。

而如果这个结论成立，则对于任意的 $a_n=n*d+k$，如果令 $a_i'=a_i-d*i$，则可以发现最优解为 $d,2d,\cdots,(n-k)d,(n-k+1)d+1,\cdots,n*d+k$。



首先考虑必要性，即每个数至少是对应值。首先，当前所有数都非负，可以发现操作后所有数仍然非负，因此显然最后所有数非负。再考虑 $a_i'=a_i-i+(n-k)$，则这样变换后 $a_1'\geq 0,a_n'=0$，且可以发现对于一次操作，一定有 $\lfloor\frac{a_i+a_k}2\rfloor-j=\lfloor\frac{(a_i-i)+(a_k-k)}2\rfloor$，即操作在变换后形式不变。因此可以发现最后一定有 $a_i'\geq 0$，因此 $a_i\geq k+i-n$。这证明了必要性。



然后考虑充分性。只需要证明最后可以让第 $i$ 个数小于等于对应的值。考虑对 $n$ 归纳，$n=2$ 显然满足条件。对于归纳的某一步，首先考虑 $M=k$ 的情况，此时可以先对于前 $k$ 个数操作，使得最后的序列每个位置对应不超过 $0,1,\cdots,k-1,k,k,\cdots,k$。然后进行若干轮操作，每一轮操作先操作第一个 $k$ 使其小于等于 $k-1$，然后操作前一个 $k-1$，然后一直向前操作到 $1$，这之后就变为了 $0,0,1,\cdots,k-1,k,k,\cdots,k$。一直操作到最后即可满足条件。

然后考虑 $M$ 任意的情况，只需要让每个数都小于等于 $k$ 即可。如果 $n$ 是奇数，则操作 $1,n$，然后对两段分别归纳即可达到目标。否则，考虑操作 $1,n-1$ 之间的所有数，由归纳，可以让 $a_2\leq \frac M{n-2}$。然后再操作 $2,n$ 之间，可以让所有数小于等于 $\max(a_2,k)$。此时 $n\geq 4$，因此重复这个过程即可让所有数小于等于 $k$。



考虑原问题，考虑找到最小的数，设 $a_x$ 为最小数。则显然令 $a_i'=a_i-a_x$ 后，操作形式不变，因此不妨设 $a_x=0$。

然后考虑在最后的序列中这个位置向右最多能有多少个 $0$。可以发现最后一个位置 $y(y\geq x)$ 能够满足 $a_y=0$ 当且仅当它满足如下条件：

存在 $z\geq y$ 使得 $a_z\leq z-y$。

如果存在这样的 $z$，则对 $[x,z]$ 使用上面的操作即可让 $a_y=0$，因此充分性得证。再考虑必要性，如果不存在这样的 $z$，考虑令 $a_i'=a_i-i+y$，则 $\forall z\geq y$，一定有 $a_z'\geq 1$。因为 $a_i\geq 0$，可以发现左侧也有 $a_i'\geq 1$，因此最后 $a_y'$ 一定大于 $0$，即 $a_y>0$。

显然如果 $y+1$ 满足条件，则 $y$ 满足条件。考虑找到满足条件的最大的 $y$，同时找到对应的 $z$。如果有多个 $z$，则找最小的一个。则此时可以发现这个 $z$ 满足如下性质：

1. $\forall i\geq z,a_i\geq a_z+i-z$
2. $\forall i<z,a_i>a_z+i-z$

证明：对于 $i\geq z$ 的部分，如果限制不被满足，则对应的 $i$ 一定可以导致上一步更大的 $y$。对于 $i<z$ 的部分，如果限制不被满足，则可以找到更大的 $y$ 或者可以选择更小的 $x$。



此时考虑令 $a_i'=a_i-i+z$，则 $a_i'\geq a_z$，这说明在最后的结果中，$a_i\geq a_z+i-z$。再结合最初序列最小元素为 $a_x=0$，可以得最后 $a_i\geq 0$。结合这两条可以得到当前 $[i,z]$ 部分使用上面的操作得到的序列是最优的。

然后考虑 $a_i'$ 上的操作，此时 $a_z'$ 是一个最小值，而操作形式不变，因此可以从 $a_z'$ 开始向右重复这个过程，直到到达结尾。另外一个方向同理。这样得到了序列每个位置的下界，同时得到了构造达到这个下界的一种方式，因此这样得到的即为最优解。



考虑上面过程的实现。可以发现满足上面的两条限制的 $z$ 唯一，即令 $a_i-i$ 最小的 $i$，如果有多个取最前面的一个。因此如果求出 $(i,a_i)$ 的凸包，则从 $x$ 开始，向右走斜率严格小于 $1$ 的边，能走到的最右侧点即为 $z$。可以发现接下来的过程相当于每次将斜率上界增加 $1$ 然后重复上述过程。

为了避免多次出现 $z=x$，可以每次找到凸包的下一条边的斜率 $v$，然后走斜率下取整后等于 $\lfloor v\rfloor$ 的边。这样即可 $O(n)$ 完成整个过程。

上述过程中每一次向右的时候可以处理出这一段内部的最优解，因此上述过程中可以同时求出答案。

复杂度 $O(n)$



另外一种奇妙做法：

可以发现，最后的最优解一定满足 $a_i'-a_{i-1}'\geq a_{i+1}'-a_i'$，否则操作这三个数一定更优。又因为操作让数变大一定不优，因此最后序列满足 $a_i'\leq a_i$

考虑满足这个条件的一个序列，设这个序列为 $b$。则有如下性质：

令 $a_i'=a_i-b_i$，对于一次 $a$ 上的操作 $(i,j,k)$，操作后的 $a_j$ 一定大于等于 $\lfloor\frac{a_i'+a_k'}2\rfloor+b_j$，即在 $a$ 上操作后对应的 $a'$ 一定对应位置大于等于对应的 $a'$ 进行相同形式的操作得到的序列。

这一点从 $b$ 的凸性即可得到。而又因为 $a_i\geq b_i$，因此 $a_i'\geq 0$，从而任意在 $a_i'$ 上操作，最后所有数非负，因此最后一定有 $a_i-b_i\geq 0$。

则对于任意一个满足条件的 $b$，都满足 $a_i\geq b_i$。而最后最优解得到的 $a$ 也为一个满足条件的 $b$。因此只需要找到满足如下条件的 $b$ 即可：

1. $b_i\leq a_i$
2. $b_i-b_{i-1}\geq b_{i+1}-b_i$
3. 在满足上面两条的情况下，$\sum b_i$ 最大。

可以发现使用上面的方式求出的结果即为这个限制下的结果。且对这个问题考虑贪心确定也可以得到上面的方式。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 300500
#define ll long long
int n,fr;
ll v[N],as;
ll solve(vector<ll> s)
{
	ll as=0;
	vector<pair<ll,int> > s2;
	int m=0;
	for(int i=0;i<s.size();i++)
	{
		while(m>=2&&(s2[m-1].first-s2[m-2].first)*(i-s2[m-2].second)>=(s[i]-s2[m-2].first)*(s2[m-1].second-s2[m-2].second))s2.pop_back(),m--;
		s2.push_back(make_pair(s[i],i));m++;
	}
	vector<ll> vl;vl.resize(m);
	ll di=0,ls=0;
	for(int i=1;i<m;i++)
	{
		ls=i-1;
		ll v1=s2[i].first,c1=s2[i].second;
		ll v0=s2[ls].first,c0=s2[ls].second;
		vl[i]=(v1-v0)/(c1-c0);
	}
	ls=0;
	for(int i=0;i<m;i++)if(i==m-1||vl[i]!=vl[i+1])
	{
		ll v1=s2[i].first,c1=s2[i].second;
		ll v0=s2[ls].first,c0=s2[ls].second;
		as+=(c1-c0)*(c1-c0+1)/2*vl[i];
		v1-=v0+(c1-c0)*vl[i];
		as+=v1*(v1+1)/2;
		as+=v0*(c1-c0);
		ls=i;
	}
	return as;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	fr=1;for(int i=1;i<=n;i++)if(v[i]<v[fr])fr=i;
	vector<ll> s1;for(int i=fr;i<=n;i++)s1.push_back(v[i]);
	as+=solve(s1);
	vector<ll> s2;for(int i=fr;i>=1;i--)s2.push_back(v[i]);
	as+=solve(s2);
	as+=v[fr];printf("%lld\n",as);
}
```



##### ARC131E Christmas Wreath

###### Problem

给定 $n$，有一个 $n$ 个点的完全图，你需要给每条边染色，一共有三种可用的颜色。你的染色方案需要满足如下条件：

1. 图中不存在一个三角形使得三条边颜色两两不同。
2. 每种颜色的边出现的次数相同。

构造任意方案或输出无解。

$n\leq 50$

$3s,1024MB$

###### Sol

如果有解，则边数一定是 $3$ 的倍数。因此 $n=3k+2$ 一定无解。同时可以发现 $3,4$ 无解。

考虑一种显然满足第一个条件的构造方式：每次选择一个点，将这个点相邻的所有边染同一种颜色，然后考虑剩下的点。

可以发现使用这种方式之后，相当于需要将 $1,2,\cdots,n-1$ 划分成三个集合，使得三个集合的元素和相等。

可以发现 $3,4$ 显然无解。但对于 $6,7,9,10$ 非常容易构造出解：
$$
\{5\},\{1,4\},\{2,3\}\\
\{1,6\},\{2,5\},\{3,4\}\\
\{4,8\},\{5,7\},\{1,2,3,6\}\\
\{6,9\},\{7,8\},\{1,2,3,4,5\}
$$
更大的情况不好手动构造，但可以发现 $n=7$ 的构造可以用来减小 $n$。具体来说，向三个集合分别加入 $\{n-1,n-6\},\{n-2,n-5\},\{n-3,n-4\}$。即可变为 $n-6$ 的问题。因此可以发现如果 $n$ 不是 $3k+2$ 型且 $n\geq 6$，则一定可以构造出解。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<string>
using namespace std;
int n;
string as;
int main()
{
	scanf("%d",&n);
	if(n%3==2||n<=4){printf("No\n");return 0;}
	printf("Yes\n");
	while(n>10)
	{
		for(int i=1;i<=6;i++,printf("\n"))
		for(int j=1;j<=n-i;j++)printf("%c","RBWWBR"[i-1]);
		n-=6;
	}
	if(n==6)as="RWWRB";
	if(n==7)as="RBWWBR";
	if(n==9)as="RRRWBRBW";
	if(n==10)as="RRRRRWBBW";
	for(int i=1;i<n;i++,printf("\n"))
	for(int j=1;j<=n-i;j++)printf("%c",as[n-1-i]);
}
```



##### ARC131F ARC Stamps

###### Problem

你有一个长度为 $n$ 的，只包含 `ARC` 的字符串 $s$。

你可以对字符串进行不超过 $k$ 次操作，每次操作为：

选择 $s$ 中任意三个连续的位置，将这三个位置替换为 `ARC`。

给出操作后的字符串 $t$，求有多少个串可能作为 $s$。答案模 $998244353$

$n\leq 5000,k\leq 10^4$

$3s,1024MB$

###### Sol

考虑从 $t$ 开始倒着操作，每次反向操作为选择一个连续的 `ARC`，将它们替换为三个任意字符。

考虑将任意字符看成 `?`，则反向操作可以看成，选择三个连续字符，满足第一个为 `A` 或者 `?`，另外两个同理，然后将这三个字符变为 `?`。

可以发现一个 $s$ 可以作为起始字符串，当且仅当存在一种对 $T$ 反向操作的方式，使得反向操作后得到的字符串 $t'$ 满足如果 $t_i'$ 不是 `?`，则 $s_i=t_i'$。



考虑 $T$ 的操作过程。显然无意义的操作可以不做，因此可以发现每次操作可以选择连续的一个 `ARC`，`A??`，`AR?`，`??C`，`?RC`，`?R?` 变为 `???`。

如果考虑 `?` 组成的段，则第一种操作为增加一个段，接下来四种操作为对一个段进行扩展，最后一种操作为合并两个段。

如果不考虑最后一个操作，则最后得到的一段 `?` 一定满足如下条件：

这段 `?` 在 $t$ 中对应位置的串可以被划分成若干段，其中正好包含一个 `ARC`，`ARC` 左侧的段全部为 `AR`，`A`，右侧全部为 `RC`，`C`。

考虑从 $t$ 中每一个 `ARC` 开始尝试向两侧扩展。可以发现扩展方式唯一。又因为每一个这样的段都以 `A` 开头 `C` 结尾，因此可以发现扩展出的段不可能相交。



考虑一段的情况，考虑对于一个 $s$ 算出 $t$ 至少需要多少操作能够得到 $s$。如果 $s=t$，则不需要进行操作。否则，考虑 $t$ 在一段内的操作，可以发现第一次操作只能操作段内的 `ARC`，接下来每次操作可以向左操作一个 `AR` 或者 `A`，或者向右操作一个 `RC` 或者 `C`。而目标为用尽量少的操作让所有 $s_i\neq t_i$ 的位置都被 `?` 覆盖。因此这一段内部可以使用如下方式操作：

1. 先操作 `ARC`。
2. 一直向左覆盖 `AR` 或者 `A`，直到覆盖了所有 $s_i\neq t_i$ 的位置。
3. 向右进行类似操作。

因此操作次数只与左侧需要操作的次数和右侧需要操作的次数有关，而这只与两侧最后一个不相等位置的距离有关，因此可以求出每一段内部需要 $0,1,\cdots$ 次操作的 $s$ 数量。如果不考虑最后一种操作，则每一段之间独立，最后的答案计算可以看成多项式乘法。



但最后一个操作可能会影响相邻的两段。可以发现最后这种操作可以留到最后做，即对于上面求出的段，如果相邻两段之间正好有一个字符 `R`，则可以通过如下方式用 `?` 覆盖这个字符：

1. 从左侧段的 `ARC` 开始将左侧段的右边部分全部覆盖。
2. 从右侧段的 `ARC` 开始将右侧段的左边部分全部覆盖。
3. 最后使用最后一种操作覆盖中间的 `R`。

因此如果想要覆盖这个字符，则必须在两侧的段内部再进行覆盖。即如果 $s$ 在这个字符上与 $t$ 不同，则需要进行这样的覆盖。可以发现如果两段之间不是这种情况，则两段中间的字符不能被 `?` 覆盖，因此不需要考虑其余情况。



考虑设 $dp_{i,j,0/1}$ 表示考虑了前 $i$ 段，当前前面需要 $j$ 次操作，当前是否钦定要覆盖这一段和下一段之间的 `R`，此时前面合法的 $s$ 数量。

转移一段时，可以枚举它和两侧的段中间的 `R` 是否覆盖（如果存在）。每种情况下的转移都可以看成 $j$ 下标中的多项式乘法。转移时需要注意一些细节，例如 $s=t$ 的情况只在两侧都不覆盖 `R` 时可以不进行操作。

直接做的复杂度为 $O(n^2)$（可以发现最多会进行 $n$ 次操作），也可以分治fft维护dp的矩阵做到 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<vector>
using namespace std;
#define N 5050
#define mod 998244353
int n,k,f[N][2],l1,l2,p3[N],as,fr[N],l3;
char s[N];
int main()
{
	scanf("%s%d",s+1,&k);n=strlen(s+1);
	p3[0]=1;for(int i=1;i<=n;i++)p3[i]=3ll*p3[i-1]%mod;
	f[0][0]=1;
	for(int i=1;i+2<=n;i++)if(s[i]=='A'&&s[i+1]=='R'&&s[i+2]=='C')
	{
		vector<int> s1,s2;
		int ls=0,nw=i,f0=0,f1=0;
		s1.push_back(1);
		while(1)
		if(s[nw-1]=='A')ls+=1,s1.push_back((p3[ls]-p3[ls-1]+mod)%mod),nw--;
		else if(nw>1&&s[nw-2]=='A'&&s[nw-1]=='R')ls+=2,s1.push_back((p3[ls]-p3[ls-2]+mod)%mod),nw-=2;
		else {if(s[nw-1]=='R'&&nw-1==l3)f0=1;break;}
		ls=0;nw=i+2;s2.push_back(1);
		while(1)
		if(s[nw+1]=='C')ls+=1,s2.push_back((p3[ls]-p3[ls-1]+mod)%mod),nw++;
		else if(s[nw+1]=='R'&&s[nw+2]=='C')ls+=2,s2.push_back((p3[ls]-p3[ls-2]+mod)%mod),nw+=2;
		else {if(s[nw+1]=='R')f1=1,l3=nw+1;break;}
		for(int j=0;j<=l1;j++)
		for(int p=0;p<s1.size();p++)
		{
			fr[j+p+1]=(fr[j+p+1]+27ll*f[j][0]%mod*s1[p])%mod;
			if(f0)fr[j+s1.size()]=(fr[j+s1.size()]+27ll*f[j][1]%mod*s1[p])%mod;
		}
		for(int j=0;j<=l1;j++)f[j][1]=0;
		for(int j=l1+1;j>=1;j--)f[j][0]=(mod+f[j][0]-f[j-1][0])%mod;
		for(int j=0;j<=l1+s1.size();j++)
		for(int p=0;p<s2.size();p++)
		{
			f[j+p][0]=(f[j+p][0]+1ll*fr[j]*s2[p])%mod;
			if(f1)f[j+s2.size()][1]=(f[j+s2.size()][1]+2ll*fr[j]*s2[p])%mod;
		}
		l1+=s1.size()+s2.size();
		for(int j=0;j<=l1;j++)fr[j]=0;
	}
	for(int i=0;i<=n&&i<=k;i++)as=(as+f[i][0])%mod;
	printf("%d\n",as);
}
```



##### ARC132E Paw

###### Problem

有一个长度为 $n$ 的，包含 `<.>` 的字符串。

你会循环进行如下操作，直到字符串中不再包含 `.`：

1. 随机选择一个 `.`
2. 随机选择左右的方向。
3. 从选中的 `.` 出发向这个方向行走，直到走出字符串或者前面是另外一个 `.` 时停止。如果方向为向左，则将这次经过的字符全部变为 `<`，否则将这次经过的字符全部变为 `>`。（这里包含上面选中的 `.`）

求操作结束时字符串 `<` 数量的期望，答案模 $998244353$

$n\leq 10^5$

###### Sol

考虑计算所有操作方案下的 `<` 数量和，即期望乘以 $n!2^n$。

考虑最后一次操作，最后一次操作后，如果这次操作为向左，则这个位置以及左侧全部变为 `<`，因为这次操作前这个位置是 `.`，两侧操作独立，因此此时右侧的结果即为只保留右侧部分进行操作时右侧的结果。

因此归纳可得，记第 $i$ 个 `.` 的位置为 $r_i$，额外记 $r_0=0,r_{ct+1}=n+1$，则一定唯一存在 $i$ 使得：

1. $[r_i+1,r_{i+1}-1]$ 中间的字符不变。
2. $[1,r_i]$ 间字符全部变为 `<`。
3. $[r_{i+1},n]$ 间字符全部变为 `>`。



考虑达到这个状态的方案数，则可以发现如下性质：

1. $[1,r_i],[r_i+1,n]$ 之间的操作不会互相影响，即独立。
2. 对于 $[1,r_i]$ 中的操作，它们需要满足不会向右覆盖到 $[r_i+1,r_{i+1}-1]$ 部分，即如果选择一个位置 $r_k$ 时 $[r_{k+1},r_i]$ 部分已经没有 `.`，则这次操作不能选择向右。可以发现，如果满足这一限制，则最后这部分一定是全部为 `<`。因此这是充分必要条件。
3. 对于 $[r_{i+1},n]$ 部分，可以发现翻转之后就和上面完全相同。

因此考虑设 $f_i$ 表示有 $i$ 个 `.` 时，满足下列条件的操作数量：

如果当前操作的 `.` 为剩余的最后一个，则这次操作只能选择向左，否则可以任意选择方向。

则达到一个 $i$ 状态的方案数为 $C_{ct}^i*f_i*f_{ct-i}$，其中第一部分为两侧操作任意排列的系数。



考虑计算 $f$。考虑枚举第一次操作的位置。如果这次操作的位置不在最后一个，则可以发现这次操作无论怎么选择，都不影响剩余的操作，这部分有 $2(i-1)f_{i-1}$ 种方案，而如果操作的位置为最后一个，则可以发现这次必须向左，而之后的操作也是 $f_{i-1}$ 的情况，因此有 $f_i=(2i-1)f_{i-1}$。事实上如果计算小的情况也非常容易发现 $f_i=(2i-1)!!$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define mod 998244353
int n,fr[N],ifr[N],f[N],as;
char s[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	f[0]=1;for(int i=1;i<=n;i++)f[i]=1ll*f[i-1]*(2*i-1)%mod*ifr[i]%mod*fr[i-1]%mod;
	int su=0,ci=0,c2=0;
	scanf("%s",s+1);
	for(int i=1;i<=n;i++)if(s[i]=='.')c2++;
	for(int i=n;i>=0;i--)if(s[i]=='.'||i==0)
	{
		as=(as+1ll*(su+i)*f[ci]%mod*f[c2-ci]%mod*fr[c2])%mod;
		su=0;ci++;
	}
	else su+=s[i]=='<';
	as=1ll*as*ifr[c2]%mod*pw((mod+1)/2,c2)%mod;
	printf("%d\n",as);
}
```



##### ARC132F Takahashi The Strongest

###### Problem

三个人进行剪刀石头布游戏，游戏一共持续 $k$ 轮。

每个人会选择一个策略，一个策略为长度为 $k$ 的，只包含 `RPS` 的字符串。这个字符串的第 $i$ 个位置表示他在第 $i$ 轮会进行的操作。

现在给定第二个人可能的 $n$ 种策略，第三个人可能的 $m$ 种策略，这两个人会在自己的策略中任意选择一个，一共有 $nm$ 种可能的情况。

第一个人有 $3^k$ 种策略可以选择，对于每一种策略，求出如果第一个人选择这种策略，在 $nm$ 种情况中有多少种情况满足如下条件：

存在至少一轮游戏，使得这轮游戏中第一个人是唯一获胜的人。

第一个人唯一获胜，当且仅当这一轮三人的操作满足如下条件之一：

1. 第一个人出 `R`，剩余两个人都出 `S`。
2. 第一个人出 `S`，剩余两个人都出 `P`。
3. 第一个人出 `P`，剩余两个人都出 `R`。

$k\leq 12,n,m\leq 3^k$

$5s,1024MB$

###### Sol

两个人的操作情况有 $9^k$ 种，但注意到如果另外两个人的操作不同，则第一个人不可能在这轮唯一获胜。如果将这些状态合并，则可能的状态只有 $4^k$ 种。

考虑算每种状态的情况数。将 `RPS` 看成 $1,2,3$，将两个人操作不同的情况看成 $0$，则相当于定义如下运算：
$$
s\oplus t=\begin{cases}s&,s=t\\0&,s\neq t\end{cases}
$$
对于两个人的策略 $s_{1,\cdots,k},t_{1,\cdots,k}$，将它们合并即得到序列 $r_i=s_i\oplus t_i$。

这相当于某种意义下的高维卷积，而每一维的情况可以看成某种高维 $\min$ 卷积，因此可以考虑如下变换：
$$
v_0'=v_0+v_1+v_2+v_3\\
v_1'=v_1,v_2'=v_2,v_3'=v_3
$$
则可以发现两个高维的情况分布进行卷积后再变换，与变换后对应位置相乘的结果等价。也可以用容斥解释解释这个变换。

这个变换以及逆变换显然可以用和fwt类似的方式在 $O(k4^k)$ 复杂度内计算。



然后考虑通过另外两个人的操作得到答案。考虑计算不满足条件的方案数。可以发现不满足条件当且仅当对于每一轮满足如下条件：

1. 如果这一轮另外两个人操作相同，则第一个人不能出能赢的那一种操作。
2. 如果另外两个人操作不同，则第一个人可以任意操作。

因此这部分相当于每一维如下变换：
$$
v_1'=v_0+v_1+v_2\\
v_2'=v_0+v_2+v_3\\
v_3'=v_0+v_3+v_1
$$
这也可以在相同的时间内计算，最后用 $nm$ 减去不合法方案数即可。

复杂度 $O(k4^k)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 16780059
#define ll long long
int k,n,m;
ll f[N],g[N];
char s[14];
int doit()
{
	scanf("%s",s+1);
	int as=0;
	for(int i=1;i<=k;i++)
	{
		as*=4;
		if(s[i]=='P')as++;
		if(s[i]=='R')as+=2;
		if(s[i]=='S')as+=3;
	}
	return as;
}
void solve(ll *f,vector<int> d,int t)
{
	for(int l=1;l<1<<2*k;l<<=2)
	for(int i=0;i<1<<2*k;i+=l*4)
	for(int j=0;j<l;j++)
	{
		ll v[4];
		for(int t=0;t<4;t++)v[t]=f[i+j+l*t];
		for(int s=0;s<d.size();s+=2)f[i+j+d[s+1]*l]+=t*v[d[s]];
	}
}
int main()
{
	scanf("%d%d%d",&k,&n,&m);
	for(int i=1;i<=n;i++)f[doit()]++;
	for(int i=1;i<=m;i++)g[doit()]++;
	solve(f,{1,0,2,0,3,0},1);solve(g,{1,0,2,0,3,0},1);
	for(int i=0;i<1<<2*k;i++)f[i]*=g[i];
	solve(f,{1,0,2,0,3,0},-1);
	solve(f,{0,1,0,2,0,3,1,2,2,3,3,1},1);
	for(int i=0;i<1<<2*k;i++)
	{
		int fg=1,tp=i;
		for(int j=1;j<=k;j++)fg&=!!(tp&3),tp>>=2;
		if(fg)printf("%lld\n",1ll*n*m-f[i]);
	}
}
```



##### ARC133E Cyclic Medians

###### Problem

给定正整数 $n,m,v,a$。考虑两个长度为 $n,m$ 的正整数序列 $x,y$，满足序列中所有数都在 $[1,v]$ 之间。然后进行如下操作：

有一个数 $z$，初始 $z=a$。

依次考虑每个 $i=1,\cdots,nm$，对于一个 $i$，将 $z$ 变为 $z,x_{((i-1)\bmod n)+1},y_{((i-1)\bmod m)+1}$ 的中位数。

对于 $v^{n+m}$ 种可能的 $(x,y)$，求和上述操作后 $z$ 的值。答案模 $998244353$

$n,m,v,a\leq 2\times 10^5$

$2s,1024MB$

###### Sol

中位数难以处理，但对于 $0,1$ 上的问题中位数存在较好的性质。

考虑线性性，答案等于对于每一个 $x=1,\cdots,t$，求和使得最后的 $z\geq t$ 的方案数。

考虑判断是否最后会有 $z\geq t$，则可以将 $\geq t$ 的数变为 $1$，剩余数变为 $0$。因为中位数的性质，可以看成在 $01$ 序列上操作。



考虑 $01$ 上的问题，每次操作为 $z$ 变为 $z,x,y$ 的中位数。考虑 $x,y$ 的情况：

1. 如果 $x,y$ 都是 $0$，则操作之后 $z=0$。如果 $x,y$ 都是 $1$，则操作之后 $z=1$。
2. 否则操作后 $z$ 不变。

因此考虑从后往前处理，找到最后一次 $x,y$ 相同的操作，对应的值即为 $z$ 最后的值。如果不存在，则 $z$ 为初始值。

则对于一个 $t$，可以看成每个数有 $v-t+1$ 种方式为 $1$，$t-1$ 种方式为 $0$，有如下情况可以满足要求：

1. 从后往前考虑操作，存在 $x=y$ 的操作且最后一次满足 $x=y$ 的操作中 $x=1$。
2. 不存在 $x=y$ 的操作且初值大于等于 $t$。

不存在 $x=y$ 的操作的情况数容易计算。注意到如果 $n,m$ 互质，上述操作会对于每一对 $(x_i,y_j)$ 进行一次，此时存在这种情况当且仅当 $x_i$ 全部相同，$y_i$ 全部相同。对于一般的情况，令 $g=\gcd(n,m)$，则所有位置可以按照模 $g$ 分类，每一类里面是互质的情况，因此可以发现这种情况的数量为：
$$
((t-1)^{\frac ng}*(v-t+1)^{\frac mg}+(t-1)^{\frac mg}*(v-t+1)^{\frac ng})^g
$$
因此这种情况可以使用线性性 $O(n\log n)$ 求出贡献。



但剩余情况中，最后一次 $x=y$ 操作满足 $x=1$ 的情况难以计算。

但可以发现，如果交换 $0,1$，则这种情况变为 $t'=v-t$ 的问题中，存在 $x=y$ 的操作且最后一次满足 $x=y$ 的操作中 $x=0$ 的情况。因此可以发现，如果对于每个 $t=0,\cdots,v$，求和存在 $x=y$ 操作的数量，则这些情况中正好有一半的情况满足最后一次操作 $x=1$。因此可以看成这些情况每一个有 $\frac 12$ 的贡献。用之前的结果即可得到这部分贡献。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int n,m,k,v,as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int gcd(int a,int b){return b?gcd(b,a%b):a;}
int solve(int p,int f)
{
	int g=gcd(n,m);
	int tp=(1ll*pw(p,n/g)*pw(mod+1-p,m/g)+1ll*pw(p,m/g)*pw(mod+1-p,n/g))%mod;
	tp=pw(tp,g);
	int as=(tp*f+1ll*(mod+1-tp)*(mod+1)/2)%mod;
	return as;
}
int main()
{
	scanf("%d%d%d%d",&n,&m,&k,&v);
	for(int i=1;i<=k+1;i++)as=(as+solve(1ll*(k-i+1)*pw(k,mod-2)%mod,v>=i))%mod;
	printf("%d\n",1ll*as*pw(k,n+m)%mod);
}
```



##### ARC133F Random Transition

###### Problem

给定正整数 $n$。

有一个数 $x$，初始 $x$ 为 $[0,n]$ 中的一个随机整数，$x=i$ 的概率为 $p_i$。

你会进行 $k$ 次操作，每次操作为：

以 $\frac xn$ 的概率将 $x$ 减一，以 $\frac{n-x}n$ 的概率将 $x$ 加一。

对于每一个 $i=0,\cdots,n$，求出 $k$ 次操作后 $x=i$ 的概率，答案模 $998244353$

$n\leq 10^5$

$5s,1024MB$

###### Sol

设 $F(x)=\sum p_ix^i$，则一次操作对 $p$ 的生成函数的变化为：
$$
F_1(x)=\frac1n(F'(x)+nxF(x)-x^2F'(x))
$$
考虑求这个变换矩阵的特征向量，即求 $F(x)$ 满足如下条件：
$$
\frac1n(F'(x)+nxF(x)-x^2F'(x))=\lambda F(x)
$$
因此解微分方程：
$$
(1-x^2)F'(x)=(n\lambda-nx)F(x)\\
\frac{F'(x)}{F(x)}=\frac{n\lambda-nx}{1-x^2}\\
\frac{F'(x)}{F(x)}=\frac12(\frac{n\lambda+n}{1+x}+\frac{n\lambda-n}{1-x})\\
\ln F(x)=\frac12((n\lambda+n)\ln(1+x)+(n-n\lambda)\ln(1-x))+C\\
F(x)=C*(1+x)^{\frac{n\lambda+n}2}*(1-x)^{\frac{n-n\lambda}2}
$$
由此不难得到 $n+1$ 个特征向量，第 $i$ 个为 $(1+x)^i(1-x)^{n-i}$，它对应的特征值为 $\frac{2i-n}n$。

因此只需要实现对 $F(x)$ 和 $\sum_{i=0}^nv_i(1+x)^i(1-x)^{n-i}$ 的相互转换，然后先将 $F$ 转换过去，对应项乘以 $(\frac{2i-n}n)^k$，然后转换回来即可得到答案。



考虑右侧到左侧的转换。首先令 $u=1+x$，则右侧形式变为 $\sum_{i=0}^nv_iu^i(2-u)^{n-i}$。

那么将这个形式变为 $u$ 的多项式时，可以发现系数为：
$$
g_i=\sum_{j\leq i}(-1)^{i-j}2^{n-i}C_{n-j}^{n-i}v_j\\
g_i=\sum_{j\leq i}(-\frac12)^{i-j}C_{n-i}^{n-j}*(v_j*2^{n-j})
$$
这可以fft解决，同时可以发现这个形式是一种二项式反演（相当于在一个类似egf的形式上卷积一个 $e^{-\frac12x}$），因此逆操作也可以写成类似形式：
$$
v_j*2^{n-j}=\sum_{j\leq i}(\frac12)^{i-j}C_{n-i}^{n-j}*g_i
$$
这里也可以翻转系数处理，翻转系数后即为正常的二项式反演。



然后考虑 $u$ 的多项式和 $x$ 的多项式间的转化，显然这个形式为：
$$
g_i=\sum_{j\geq i}C_j^if_j
$$
显然这也可以二项式反演，因此有：
$$
f_i=\sum_{j\geq i}(-1)^{i-j}C_j^ig_j
$$
这样就完成了双向的变换，fft实现即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 263001
#define mod 998244353
int n,k,v[N];
int fr[N],ifr[N],rev[N*2],gr[2][N*2];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d=18)
{
	fr[0]=1;for(int i=1;i<=1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1<<d]=pw(fr[1<<d],mod-2);for(int i=1<<d;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int l=2;l<=1<<d;l<<=1)for(int i=1;i<l;i++)rev[l+i]=(rev[l+(i>>1)]>>1)|((i&1)*(l>>1));
	for(int t=0;t<2;t++)
	for(int l=2;l<=1<<d;l<<=1)
	{
		int tp=pw(3,(mod-1)/l),vl=1;
		if(!t)tp=pw(tp,mod-2);
		for(int i=0;i<l;i++)gr[t][i+l]=vl,vl=1ll*vl*tp%mod;
	}
}
int f[N],g[N],ntt[N];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=0;j<l>>1;j++)
	{
		int v1=ntt[i+j],v2=1ll*ntt[i+j+(l>>1)]*gr[t][l+j]%mod;
		ntt[i+j]=(v1+v2)%mod;
		ntt[i+j+(l>>1)]=(v1+mod-v2)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=0;i<=n;i++)scanf("%d",&v[i]),v[i]=1ll*v[i]*pw(1e9,mod-2)%mod;
	init(18);int l=1;while(l<=n*2)l<<=1;
	
	for(int i=0;i<=n;i++)f[i]=1ll*v[i]*fr[i]%mod,g[i]=1ll*ifr[n-i]*((n-i)&1?mod-1:1)%mod;
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	for(int i=0;i<=n;i++)v[i]=1ll*ifr[i]*f[i+n]%mod;
	
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<=n;i++)f[i]=1ll*v[i]*fr[n-i]%mod,g[i]=1ll*ifr[i]*pw((mod+1)/2,i)%mod;
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	for(int i=0;i<=n;i++)v[i]=1ll*f[i]*ifr[n-i]%mod*pw((mod+1)/2,n-i)%mod;
	
	for(int i=0;i<=n;i++)v[i]=1ll*v[i]*pw(1ll*(2*i+mod-n)*pw(n,mod-2)%mod,k)%mod;
	
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<=n;i++)f[i]=1ll*v[i]*pw(2,n-i)%mod*fr[n-i]%mod,g[i]=1ll*ifr[i]*pw((mod+1)/2,i)%mod*(i&1?mod-1:1)%mod;
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	for(int i=0;i<=n;i++)v[i]=1ll*f[i]*ifr[n-i]%mod;
	
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<=n;i++)f[i]=1ll*v[i]*fr[i]%mod,g[i]=ifr[n-i];
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	for(int i=0;i<=n;i++)v[i]=1ll*ifr[i]*f[i+n]%mod;
	for(int i=0;i<=n;i++)printf("%d ",v[i]);
}
```

