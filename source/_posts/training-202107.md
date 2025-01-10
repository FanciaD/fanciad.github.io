---
title: 2021/07 集训题解
date: '2021-07-30 18:58:43'
updated: '2021-07-30 18:58:43'
tags: Mildia
permalink: YuukokuNoKane/
description: 2021/07 南京集训
mathjax: true
---

#### 0706

##### T1 jewelry

Source: loj 6039 「雅礼集训 2017 Day5」珠宝

###### Problem

有 $n$ 个物品，每个物品有体积 $c_i$ 和价格 $v_i$。

你可以选择若干个物品满足 $\sum c_i\leq k$ 且 $\sum v_i$ 最大，每个物品只能选一次。

给定 $m$，对于每一个 $k=1,2,...,m$ 求出答案。

$n\leq 10^6,m\leq 5\times 10^4,c_i\leq 300,v_i\leq 10^9$

$2s,256MB$

###### Sol

注意到 $c$ 很小，考虑将所有物品按照 $c$ 分类，设 $dp_{i,j}$ 表示考虑了体积不超过 $i$ 的物品，选体积和不超过 $j$ 的物品的最大收益。

显然一类物品只会选收益最大的若干个，设 $su_{c,k}$ 表示体积为 $c$ 的物品中收益最大的 $k$ 个的收益和，则转移为：
$$
dp_{i,j}=\max_{k*i\leq j}dp_{i-1,j-i*k}+su_{i,k}
$$
显然 $su_c$ 为上凸序列， $2su_i\geq su_{i-1}+su_{i+1}$，因此如果将转移时的 $j$ 按照 $j\bmod i$ 分类，则每一类内部的转移系数 $f_{j,j+k*i}=su_{i,k}$ 显然满足四边形不等式，因此转移具有决策单调性。分治即可。

复杂度 $O((n+kc)\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 50050
#define M 305
#define ll long long
int n,m,a,b,ct;
ll dp[N],dp2[N],su[N],v1[N],v2[N];
vector<int> st[M];
int rd()
{
	int as=0;char c=getchar();
	while(c<'0'||c>'9')c=getchar();
	while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();
	return as;
}
void solve(int l,int r,int l1,int r1)
{
	if(l>r)return;
	int mid=(l+r)>>1;
	ll as=-1,fr=0;
	for(int i=l1;i<=r1;i++)if(i<=mid&&mid-i<=ct)
	{
		ll vl=v1[i]+su[mid-i];
		if(vl>as)as=vl,fr=i;
	}
	v2[mid]=as;
	solve(l,mid-1,l1,fr);solve(mid+1,r,fr,r1);
}
int main()
{
	freopen("jewelry.in","r",stdin);
	freopen("jewelry.out","w",stdout);
	n=rd();m=rd();
	for(int i=1;i<=n;i++)a=rd(),b=rd(),st[a].push_back(-b);
	for(int i=1;i<=300;i++)
	{
		sort(st[i].begin(),st[i].end());
		ct=st[i].size();if(ct>m)ct=m;
		for(int j=0;j<ct;j++)su[j+1]=su[j]-st[i][j];
		for(int j=0;j<i;j++)
		{
			int ct2=0,nw=j;
			while(nw<=m)v1[++ct2]=dp[nw],nw+=i;
			solve(1,ct2,1,ct2);
			for(int l=1;l<=ct2;l++)dp2[(l-1)*i+j]=v2[l];
		}
		for(int j=1;j<=m;j++)dp[j]=dp2[j];
	}
	for(int i=1;i<=m;i++)printf("%lld ",dp[i]);
}
```

##### T2 npc

###### Problem

给定 $k$ ，要求构造一个有向无环图，满足它的拓扑序数量为 $k$。

$k<2^{15}$，要求点数不超过 $50$，边数不超过 $100$

$1s,512MB$

###### Sol

考虑构造一个图，使得这个图的所有拓扑序只有两个可能的终点，且以这两个为终点的拓扑序分别有 $a,b$ 个，设这两个终点为 $x,y$。

考虑加入一个点 $z$，考虑从除去 $y$ 外的所有点向 $z$ 连边，此时可能的终点只剩下 $y,z$。

如果以 $z$ 为终点，则显然删去 $z$ 后为前面的拓扑序，因此方案数为 $a+b$.

如果以 $y$ 为终点，则删去 $y$ 后最后一个位置一定是 $z$，前面和之前一样，因此方案数为 $b$。

只考虑两个方案数的变化，则操作可以看成 $a+=b$ 或者 $b+=a$，初始值为 $a=1,b=0$，需要 $a+b=k$。

可以发现这个过程倒过来就是辗转相减，因此最后的值一定满足 $(a,b)=1$，此时~~枚举~~随机 $a$，可以发现一定存在 $a$ 使得操作次数不超过 $24$ 次。

直接连边边数不能接受，但可以发现两种操作连出的点一定分别构成一条链，因此只需要向两条链上分别连一条边即可。

点数 $25$，边数 $48$，~~应该是logn级别的但是常数无法分析~~

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<cstdlib>
using namespace std;
int n,m,s[111][2],v1,v2,l1,l2,k;
vector<int> tp;
int check(int x,int y)
{
	int ct=0;
	while(y&&ct<30)
	if(x<y)x^=y^=x^=y;
	else x-=y,ct++;
	if(x>1)ct=100;
	return ct;
}
int main()
{
	scanf("%d",&k);
	int st=1;
	while(check(st,k-st)>25)st=rand()%(k-1)+1;
	int s1=st,s2=k-st,op=1;
	while(s2)
	if(s1<s2)s1^=s2^=s1^=s2,op^=1;
	else s1-=s2,tp.push_back(op);
	n=1,v1=1;
	for(int i=tp.size()-1;i>=0;i--)
	if(tp[i]!=op){n++;if(v2)s[++m][0]=v2,s[m][1]=n;l2=v2,v2=n;if(l1)s[++m][0]=l1,s[m][1]=n;}
	else {n++;if(v1)s[++m][0]=v1,s[m][1]=n;l1=v1,v1=n;if(l2)s[++m][0]=l2,s[m][1]=n;}
	printf("%d %d\n",n,m);
	for(int i=1;i<=m;i++)printf("%d %d\n",s[i][0]-1,s[i][1]-1);
}
```

##### T3 point

Source: XVIII Open Cup GP of Ukraine K

###### Problem

数轴上有 $n$ 个点，点 $i$ 的位置为 $x_i$。

现在对于每个点，你需要将它的位置向左或者向右移动 $d$。

在移动后，你需要选择若干个线段覆盖这些点。选择一个线段 $[l,r]$ 的代价为 $a+b*(r-l)$

求最小代价和。

$1\leq n,d,x_i\leq 150$

$1s,256MB$

###### Sol

可以看成每个位置要么不移动，要么向右移动 $2d$。

对于 $a\leq b$ 的情况，可以发现最优解一定是选若干个单点区间，因此问题变为让不同位置的点数最小。

此时对于一个数 $x_i$ ，可以看成一条边 $(x_i,x_i+2d)$，目标是选个数尽量小的点，使得每条边的两个端点至少有一个被选。

可以发现所有边一定构成若干条链，一条 $n$ 个点的链的答案显然为 $\lfloor\frac n2\rfloor$，不同链答案相加即可。复杂度 $O(n)$

考虑 $a>b$ 的情况。考虑最后每个位置是否被覆盖。设变量 $v_i$，$v_i=1$ 表示 $x=i$ 的位置被覆盖，否则表示没有被覆盖。

考虑此时的代价，因为 $a>b$，因此对于两个相邻且都满足 $v_i=1$ 的点，它们一定在同一条线段内。

此时可以看成，每一个 $v_i=1$ 的点会带来 $b$ 的代价，每一个满足 $v_i=1,v_{i-1}=0$ 的点会带来额外 $a-b$ 的代价，这样对于连续的 $k$ 个 $v_i=1$ 的点，总代价为 $k*b+(a-b)=a+b*(k-1)$，即为覆盖这些点的线段的代价。

再考虑每个点的限制，对于一个 $x_i$，相当于要求 $v_{x_i},v_{x_i+2d}$ 至少有一个为 $1$，相当于如果 $v_{x_i}=0,v_{x_i+2d}=0$ 则有 $\infty$ 的代价。

首先考虑 $d$ 很小的做法。因为每个限制和两个距离不超过 $2d$ 的 $v_i$ 有关，因此可以设 $dp_{i,S}$ 表示当前考虑了前 $i$ 个位置，当前后 $2d$ 个位置的 $v_i$ 状态为 $S$，前面的最优解。转移可以直接枚举下一步的情况并判断是否合法。复杂度 $O((n+d)4^d)$。

考虑表示成最小割的形式，建 $(\max x_i)+2d+2$ 个点，分别表示 $v_{0,1,...,(\max x_i)+2d+1}$，从原点向每个点连边，每个点向汇点连边。如果$v_i$ 在割集右侧，相当于令 $v_i=0$，否则相当于 $v_i=1$。

此时可以发现单点代价可以直接看成上面边的流量。对于 $v_i=1,v_{i-1}=0$ 的限制，可以看成如果 $v_i$ 在左侧割集中但 $v_{i-1}$ 不在则有 $a-b$ 的代价，这可以看成连一条 $v_i$ 连向 $v_{i-1}$ 的边，流量为 $a-b$，此时如果出现了这种情况则需要割掉这条边，计算这个代价。

但对于 $v_{x_i}=0,v_{x_i+2d}=0$ 的限制，直接做无法处理。因为前面处理了 $d$ 很小的情况，此时 $\frac n{2d}$ 不大。考虑按照 $\frac n{2d}$ 分段，对于 $v_i$，如果 $\lfloor\frac i{2d}\rfloor\equiv 1(\bmod 2)$，则翻转 $v_i$ 值的意义。此时每个上面这种限制的两个 $v_i$ 一定在不同段中，因此这变成了一个类似 $v_{x_i}=0,v_{x_i+2d}=1$ 的限制，可以直接建边。

再考虑上面的限制，显然单点代价的建边方式不会改变，对于原来 $v_i=1,v_{i-1}=0$，如果 $i,i-1$ 在同一段内，则可以发现最后的形式仍然可以看成一条边。但如果 $i,i-1$ 不在同一段内，则限制会变为 $v_i=v_{i-1}=0$ 或者 $v_i=v_{i-1}=1$。

但这样的限制只有 $\frac n{2d}$ 个，且和所有 $v_{2d*k}$ 相关。考虑枚举所有的 $v_{2d*k}$ 的值，在图上把 $v_{2d*k}$ 不应该割的边的流量设为 $\infty$，此时 $v_{2d*k}$ 和 $v_{2d*k-1}$ 间的限制可以看成给 $v_{2d*k-1}$ 的单点代价加一个权值。剩下的限制都可以看成边，因此此时求出这个图的最小割即可。

这样的复杂度为 $O(2^{\frac n{2d}}*dinic(n+d))$，最后的复杂度即为两者最小值，可以看成类似 $O(2^{\sqrt n}*dinic(n))$。

在 $d\leq 8$ 的时候用状压，$d>8$ 的时候用网络流即可。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 465
int n,d,a,b,v[175],dp[170][65600][2],is[170];
struct stodjq{
	void solve()
	{
		for(int i=0;i<=167;i++)
		{
			is[i]=0;
			for(int j=0;j<1<<16;j++)
			for(int k=0;k<2;k++)dp[i][j][k]=1e9;
		}
		for(int i=1;i<=n;i++)is[v[i]]=1;
		dp[0][0][0]=0;
		for(int i=1;i<=167;i++)
		for(int j=0;j<1<<16;j++)
		for(int k=0;k<2;k++)if(dp[i-1][j][k]<9e8)
		for(int t=0;t<=is[i];t++)
		{
			int fg=(j&1)|t,nt=(j>>1)|((is[i]-t)<<(d-1));
			if(!fg)dp[i][nt][k]=min(dp[i][nt][k],dp[i-1][j][k]);
			else 
			{
				int vl=dp[i-1][j][k];
				if(k)vl+=b*i;
				for(int p=0;p<2;p++)dp[i][nt][p]=min(dp[i][nt][p],vl+(!p)*a-p*i*b);
			}
		}
		printf("%d\n",dp[167][0][0]);
	}
}tsk1;
int head[N],cnt,cur[N],dis[N],ct;
struct edge{int t,next,v;}ed[N*10];
struct djqorz{
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
	void init(){for(int i=1;i<=ct;i++)head[i]=0;cnt=1;}
	int dinic(){int ls=1e9,as=0;while(ls&&bfs(ct-1,ct)){int tp=dfs(ct-1,ct,ls);ls-=tp;as+=tp;}return as;}
	void solve()
	{
		int mx=0;
		for(int i=1;i<=n;i++)mx=max(mx,v[i]);
		int sn=mx+d;ct=sn+2;
		int ct2=(sn+1)/d,as=1e9;
		for(int tp=0;tp<1<<ct2;tp++)
		{
			init();
			for(int j=1;j<=sn;j++)
			{
				int v0=b,v1=0;
				if(j==1)v0=a;
				if((j/d)&1)swap(v0,v1);
				if((j+1)%d==0)
				{
					int id=(j+1)/d,vl=(tp>>(id-1))&1;
					vl^=(id&1);
					if(vl==0)if(id&1)v1+=a-b;else v0+=a-b;
				}
				if(j%d==0)
				{
					int id=j/d,vl=(tp>>(id-1))&1;
					if(vl)v0=1e9;else v1=1e9;
				}
				if(v0)adde(sn+1,j,v0);if(v1)adde(j,sn+2,v1);
			}
			for(int j=1;j<=n;j++)
			{
				int sl=v[j],sr=v[j]+d;
				if((sl/d)&1)adde(sr,sl,1e9);
				else adde(sl,sr,1e9);
			}
			for(int j=1;j<=sn;j++)
			{
				if(j==sn||(j+1)%d==0)continue;
				if((j/d)&1)adde(j+1,j,a-b);
				else adde(j,j+1,a-b);
			}
			int ras=dinic();
			if(ras<as)as=ras;
		}
		printf("%d\n",as);
	}
}tsk2;
int main()
{
	freopen("point.in","r",stdin);
	freopen("point.out","w",stdout);
	scanf("%d%d%d%d",&n,&d,&a,&b);d*=2;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);sort(v+1,v+n+1);
	int ct=0;
	for(int i=1;i<=n;i++)if(v[i]!=v[i-1])v[++ct]=v[i];
	n=ct;
	if(a<=b)
	{
		int as=0;
		for(int i=0;i<d;i++)
		{
			int l1=-1,ls=-1;
			for(int j=1;j<=n;j++)if(v[j]%d==i)
			{
				int st=v[j]/d;
				if(ls==-1)l1=ls=st;
				else if(st==ls+1)ls++;
				else as+=(ls-l1+2)/2*a,ls=l1=st;
			}
			if(ls!=-1)as+=(ls-l1+2)/2*a;
		}
		printf("%d\n",as);
		return 0;
	}
	if(d<=16)tsk1.solve();
	else tsk2.solve();
}
```

#### 0708

##### T1 random

Source: [2017~18 集训队自选题] 小C的岛屿

###### Problem

给一个 $n$ 个点的有向图 $G$，每个点都有至少一条出边，每个点上还有一个概率 $p_i$。

有一个 $n$ 个点的无向图 $H$，初始没有边。

你在 $G$ 上进行随机游走。初始你在 $1$，你循环进行如下操作：

1. 设当前点为 $x$，以 $p_x$ 的概率向 $H$ 中加入一条 $H$ 当前没有的边。如果图连通就结束。
2. 随机选择 $x$ 的一条出边，走到出边的另外一个端点。

求结束前经过边数的期望，模 $10^9+7$

$n\leq 100$

$2s,512MB$

###### Sol

随机游走部分和加边部分相对独立，考虑先对于加边部分，求出加入 $x$ 条边的概率。即求出在加入第 $x$ 条边时变为连通的情况数量。设这个值为 $v_x$。

在加入最后一条边之前，图一定被分成两个连通块。考虑枚举两边的大小以及两边的边数，设 $f_{n,m}$ 表示 $n$ 个点 $m$ 条边的连通块数量，则有：
$$
v_x=(x-1)!\sum_{i=1}^{n-1}\sum_{j=0}^mf_{i,j}f_{n-i,x-j-1}*i*(n-i)
$$
考虑求出 $f$，容斥减去不连通的情况，枚举 $1$ 所在的连通块大小则有：
$$
f_{n,m}=C_{\frac{n(n+1)}2}^m-\sum_{i=1}^{n-1}\sum_{j=0}^mf_{i,j}C_{\frac{(n-i)(n-i+1)}2}^{m-j}
$$
直接计算复杂度为 $O(n^2m^2)=O(n^6)$，但注意到计算 $f$ 以及计算 $v$ 都可以看成多项式乘法，且最后的多项式为 $O(n^2)$ 项，因此可以计算将 $v$ 看成多项式后 $1,2,...,O(m)$ 处的点值，然后通过点值插值还原多项式。

计算单个点值的复杂度为 $O(n^2)$，这部分计算所有点值和插值的复杂度均为 $O(n^4)$。

现在考虑随机游走的过程，只需要考虑每一次加边操作之间的关系。设 $e_{x}$ 表示从 $x$ 开始，到下一次加边的期望时间，$d_{x,y}$ 表示从 $x$ 开始，下一次加边时在 $y$ 的概率。

考虑计算 $e_x$，显然可以高斯消元。可以发现计算一个 $d_{x,y}$ 相当于把第 $y$ 行的常数设为 $1$，其余常数设为 $0$，做上面的高斯消元。因此考虑再加上 $n$ 个变量，分别表示每一行的常数，将新加入的变量看成常数消元即可。~~这个过程可以看成求逆~~

此时设 $dp_{i,j}$ 表示加边 $i$ 次时，不考虑停止在 $j$ 的概率。可以使用 $d$ 直接转移，复杂度 $O(n^4)$。

对答案考虑期望线性性，可以看成对于每一个 $dp_{i,j}$ 加上达到这种状态的概率乘上这种状态到达下一个状态的期望时间再乘上之前不停止的概率的结果。因此答案可以直接计算。

复杂度 $O(n^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105
#define M 5050
#define mod 1000000007
int n,m,l,a,b,pr[N],is[N][N],v[N][N*2],dp[N][M],fr[M],ifr[M],f[N][M],g[M],d[N],v1[M],v2[M],f1[M],f2[M],r1[N],inv[M*2];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void solve()
{
	for(int i=1;i<=n;i++)
	{
		int fr=i;
		for(int j=i;j<=n;j++)if(v[j][i])fr=i;
		for(int j=1;j<=n*2+1;j++)swap(v[fr][j],v[i][j]);
		for(int j=1;j<=n;j++)if(j!=i)
		{
			int tp=1ll*(mod-1)*v[j][i]%mod*pw(v[i][i],mod-2)%mod;
			for(int k=1;k<=n*2+1;k++)v[j][k]=(v[j][k]+1ll*v[i][k]*tp)%mod;
		}
	}
	for(int i=1;i<=n;i++)for(int j=n+1;j<=n*2+1;j++)v[i][j]=1ll*v[i][j]*pw(v[i][i],mod-2)%mod*(mod-1)%mod;
}
void justdoit(int n,int* v)
{
	for(int i=0;i<=n*2;i++)inv[i]=pw(mod+i-n,mod-2);
	for(int i=0;i<=n+1;i++)f1[i]=f2[i]=0;
	f1[0]=1;
	for(int i=1;i<=n;i++)
		for(int j=i;j>=0;j--)f1[j+1]=(f1[j+1]+f1[j])%mod,f1[j]=1ll*f1[j]*(mod-i)%mod;
	for(int i=1;i<=n;i++)
	{
		int vl=v[i];
		for(int j=1;j<=n;j++)if(j!=i)vl=1ll*vl*inv[i-j+n]%mod;
		for(int j=0;j<=n;j++)f1[j]=1ll*f1[j]*inv[n-i]%mod,f1[j+1]=(f1[j+1]+mod-f1[j])%mod;
		for(int j=0;j<=n;j++)f2[j]=(f2[j]+1ll*f1[j]*vl)%mod;
		for(int j=n;j>=0;j--)f1[j+1]=(f1[j+1]+f1[j])%mod,f1[j]=1ll*f1[j]*(mod-i)%mod;
	}
	for(int i=1;i<=n;i++)
	{
		int as=0,vl=1;
		for(int j=0;j<=n;j++)as=(as+1ll*vl*f2[j])%mod,vl=1ll*vl*i%mod;
	}
	for(int i=0;i<=n;i++)v[i]=f2[i];
}
void calc()
{
	l=(n-1)*n/2;
	fr[0]=ifr[0]=1;for(int i=1;i<=l;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int fu=1;fu<=l+2;fu++)
	{
		for(int i=1;i<=n;i++)r1[i]=pw(fu+1,i*(i-1)/2);
		for(int i=1;i<=n;i++)
		{
			int su=i*(i-1)/2;
			f1[i]=r1[i];
			for(int j=1;j<i;j++)
				f1[i]=(f1[i]+1ll*(mod-1)*fr[i-1]%mod*ifr[j-1]%mod*ifr[i-j]%mod*f1[j]%mod*r1[i-j])%mod;
		}
		for(int j=1;j<n;j++)
			v1[fu]=(v1[fu]+1ll*fr[n-1]*ifr[n-j]%mod*ifr[j-1]%mod*f1[j]%mod*f1[n-j]%mod*j*(n-j))%mod;
		v2[fu]=f1[n];
	}
	justdoit(l+2,v1);justdoit(l+2,v2);
	for(int i=0;i<=l;i++)
	{
		int s2=(1ll*fr[l]*ifr[i]%mod*ifr[l-i]+mod-v2[i])%mod*(l-i)%mod,s1=v1[i];
		g[i]=1ll*s1*pw(s2,mod-2)%mod;
	}
}
void doit()
{
	for(int i=1;i<=n;i++)d[i]=pw(d[i],mod-2);
	for(int j=1;j<=n;j++)for(int k=1;k<=n*2+1;k++)v[j][k]=0;
	for(int j=1;j<=n;j++)
	{
		v[j][j]=mod-1;v[j][n+1]=1;
		for(int k=1;k<=n;k++)if(is[j][k])v[j][k]=(v[j][k]+1ll*(mod+1-pr[j])*d[j])%mod,v[j][n+k+1]=(v[j][n+k+1]+1ll*pr[j]*d[j])%mod;
	}
	solve();
	dp[1][0]=1;
	for(int i=1;i<=l;i++)
	for(int j=1;j<=n;j++)
	for(int k=1;k<=n;k++)
	dp[k][i]=(dp[k][i]+1ll*dp[j][i-1]*v[j][n+k+1])%mod;
	for(int i=0;i<=l;i++)
	{
		g[i]=mod+1-g[i];
		if(i>0)g[i]=1ll*g[i-1]*g[i]%mod;
	}
	int as=v[1][n+1];
	for(int i=1;i<=l;i++)
	for(int j=1;j<=n;j++)
	as=(as+1ll*dp[j][i]*v[j][n+1]%mod*g[i-1])%mod;
	as=(as+mod-1)%mod;
	printf("%d\n",as);
}
int main()
{
	freopen("random.in","r",stdin);
	freopen("random.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&pr[i]),pr[i]=1ll*pr[i]*pw(100,mod-2)%mod;
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),is[a][b]=1,d[a]++;
	calc();doit();
}
```

##### T2 tree

Source: CF500G

###### Problem

给一棵 $n$ 个点的树，有 $q$ 次询问：

给出 $a,b,c,d$，第一个人在 $a$ 到 $b$ 的路径上循环运动，第二个人在 $c$ 到 $d$ 的路径上循环运动，两人速度都为 $1$。求两个人第一次在点上相遇的时间。

$n,q\leq 2\times 10^5$

$4s,512MB$

###### Sol

考虑两人路径的交，这显然是一条路径。

考虑求出路径的交，一种不用特判的方式是，先找到四个点的LCA中最深的，这一定是路径交（如果存在）的一个端点。此时另外一个端点一定是某两个点的LCA，枚举并找到合法且使得路径最长的另外一个端点即可。

此时只需要考虑每一段路径的长度。设交的长度为 $le$，第一个人的路径在交两侧延伸的长度为 $l_1,r_1$，第二个人的路径在交两侧的长度为 $l_2,r_2$，两条路径长度分别为 $s_1,s_2$。考虑两个人同时从左侧出发的情况，另外一种类似。

1. 两个人在同向相遇。

可以发现同向相遇一定在交的端点上，考虑左端点相遇，则当前时间 $t$ 一定满足：
$$
t\bmod 2s_1=l_1\or t\bmod 2s_1=2s_1-l_1\\
t\bmod 2s_2=l_2\or t\bmod 2s_2=2s_2-l_2
$$
枚举一种情况做excrt即可。复杂度 $O(\log n)$

2. 两个人在不同方向相遇。

考虑第一个人在从左向右，第二个人在从右向左，则需要满足如下条件：
$$
t\bmod 2s_1\in[l_1,l_1+le]\\
(t\bmod 2s_1)+(t\bmod 2s_2)=2s_2-l_2+l_1
$$
设 $t=2s_1*k_1+r_1(r_1\in [l_1,l_1+le])$，$t=2s_2*k_2+r_2(r_2\in[0,2s_2))$，则有：
$$
r_1+r_2=2s_2-l_2+l_1
$$
此时用 $1$ 式减去 $2$ 式再加上 $3$ 式即可得到：
$$
2s_1*k_1-2s_2*k_2+2r_1=2s_2-l_2+l_1
$$
如果 $l_1-l_2$ 为奇数则无解，否则可以整体除以 $2$，得到：
$$
s_1k_1-s_2k_2+r_1=s_2+\frac{l_1-l_2}2\\
s_1k_1-s_2k_2\in[(s_2+\frac{l_1-l_2}2)-(l_1+le),(s_2+\frac{l_1-l_2}2)-l1]
$$
这相当于 $s_1k_1\bmod s_2$ 在一个区间内。显然 $k_1$ 最小时的解即为 $t$ 最小的解。因此问题变为给定方程 $ax\bmod b\in [l,r]$，求最小的非负 $x$ 解。

首先可以将 $a$ 变为 $a\bmod b$，考虑 $a<b$ 的情况。

可以看成求 $ax-by\in[l,r]$ 的最小解，其中 $y$ 非负。显然如果 $y=0$ 有解，最小解一定是 $y=0$。否则，如果 $x=0$ 显然无解。

对于其余的情况，可以发现 $x$ 最小等价于 $-y$ 最小，因此可以整体取负，变为 $b(-y)-a(-x)\in[-r,-l]$，求最小的非负 $-y$。

此时得到了交换 $a,b$ 的问题，因为可以 $a=a\bmod b$，因此复杂度与gcd相同，为 $O(\log n)$。

最后还原答案即可。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,q,a,b,v[4],head[N],cnt,dep[N],f[N][18];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){dep[u]=dep[fa]+1;f[u][0]=fa;for(int i=1;i<=17;i++)f[u][i]=f[f[u][i-1]][i-1];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);}
int LCA(int x,int y){if(dep[x]<dep[y])swap(x,y);for(int i=17;i>=0;i--)if(dep[x]-dep[y]>=(1<<i))x=f[x][i];if(x==y)return x;for(int i=17;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int doit(int x,int y){return dep[x]>dep[y]?x:y;}
int getdis(int x,int y){return dep[x]+dep[y]-2*dep[LCA(x,y)];}
bool chk(int x,int y,int a){return getdis(x,a)+getdis(y,a)==getdis(x,y);}
ll exgcd(ll a,ll b,ll &x,ll &y)
{
	if(!b){x=1,y=0;return a;}
	ll g=exgcd(b,a%b,y,x);y-=a/b*x;return g;
}
ll excrt(ll b,ll a,ll d,ll c)
{
	ll x,y;
	ll g=exgcd(b,d,x,y),l=b/g*d;
	if((a-c)%g)return 1e18;
	ll as=a,tp=(c-a%d+d)%d/g;
	as=(as+tp*b%l*x)%l;as=(as+l)%l;
	return as;
}
ll doit(ll x,ll y,ll a,ll b)
{
	x%=y;
	if(a<0){ll tp=(-a)/y;tp=(tp+1)*y;a+=tp;b+=tp;}
	ll tp=a/y*y;a-=tp;b-=tp;
	if(b>=y)return 0;
	if(!x)return -1;
	if(b/x>(a-1)/x)return (a-1)/x+1;
	ll ry=doit(y,x,-b,-a);
	if(ry==-1)return -1;
	return (b+ry*y)/x;
}
ll solve(int l1,int r1,int l2,int r2,int sl,int ty)
{
//	printf("%d %d %d %d %d %d\n",l1,r1,l2,r2,sl,ty);
	ll as=1e18,s1=l1+r1+sl,s2=l2+r2+sl;
	for(int f=0;f<2;f++)
	{
		int lb,rb;
		if(!f)lb=l1,rb=ty?r2+sl-1:l2;
		else lb=sl+l1-1,rb=ty?r2:l2+sl-1;
		for(int p=-1;p<2;p+=2)
		for(int q=-1;q<2;q+=2)
		as=min(as,excrt(s1*2-2,s1*2-2+p*lb,s2*2-2,s2*2-2+q*rb));
	}
	ll cr1=s1*2-2,cr2=s2*2-2;
	//case 1: t mod cr1 + t mod cr2 = s2*2+l1-l2,t mod cr1 in [l1,l1+sl)
	//t=cr1*k1+r1=cr2*k2+(s2*2+l1-l2)-r1
	//cr1*k1-cr2*k2\in((s2*2+l1-l2)-2*(l1+sl),(s2*2+l1-l2)-2*l1]
	//if ty,vl=sl+l1+r2
	ll lb=l1,rb=l1+sl-1,vl=s2*2+l1-l2-2,as1;
	if(ty)vl=sl+l1+r2-1;
	as1=doit(cr1,cr2,vl-2*rb,vl-2*lb);
	if(vl&1)as1=-1;
	if(as1!=-1)
	{
		ll k1=as1,k2=(k1*cr1-(vl-2*rb))/cr2;
		ll r1=(cr2*k2-cr1*k1+vl)/2;
		as=min(as,k1*cr1+r1);
	}
	//case 2: t mod cr1 + t mod cr2 = s1*2+l2-l1,t mod cr1 in (s1*2-l1-sl,s1*2-l1]
	lb=s1*2-l1-sl-1,rb=s1*2-l1-2,vl=s1*2+l2-l1-2;
	if(ty)vl=s1+s2+sl+l2+r1-3;
	as1=doit(cr1,cr2,vl-2*rb,vl-2*lb);
	if(vl&1)as1=-1;
	if(as1!=-1)
	{
		ll k1=as1,k2=(k1*cr1-(vl-2*rb))/cr2;
		ll r1=(cr2*k2-cr1*k1+vl)/2;
		as=min(as,k1*cr1+r1);
	}
	if(as>1e16)as=-1;
	return as;
}
int main()
{
	freopen("tree.in","r",stdin);
	freopen("tree.out","w",stdout);
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	scanf("%d",&q);
	while(q--)
	{
		for(int i=0;i<4;i++)scanf("%d",&v[i]);
		int as=0;
		for(int i=0;i<4;i++)for(int j=i+1;j<4;j++)as=doit(as,LCA(v[i],v[j]));
		if(!chk(v[0],v[1],as)||!chk(v[2],v[3],as)){printf("-1\n");continue;}
		int s1=as,s2=as,ds=0;
		for(int i=0;i<4;i++)for(int j=i+1;j<4;j++)
		{
			int l=LCA(v[i],v[j]);
			if(!chk(v[0],v[1],l)||!chk(v[2],v[3],l))continue;
			int d1=getdis(l,s1);
			if(d1>ds)ds=d1,s2=l;
		}
		if(getdis(v[0],s1)>getdis(v[0],s2))swap(s1,s2);
//		printf("%d %d\n",s1,s2);
		int l1=getdis(v[0],s1),r1=getdis(v[1],s2),l2=getdis(v[2],s1),r2=getdis(v[3],s2),sl=getdis(s1,s2),fg=0;
		if(l2+r2>getdis(v[2],v[3]))l2-=sl,r2-=sl,fg=1,swap(l2,r2);
		printf("%lld\n",solve(l1,r1,l2,r2,sl+1,fg));
	}
}
```

##### T3 grid

###### Problem

给一个 $n\times m$ 的网格图，每条边有边权 $c_i$ 。

每个点有 $l_{i,j},r_{i,j}$，你需要对于每个点选择一个整数权值 $v_{i,j}\in[l_{i,j},r_{i,j}]$。

定义一种方式的代价为每条边的边权乘上边两侧点的点权的差的绝对值的和。求出最小代价。

$n\leq 5\times 10^4,m\leq 5,r_{i,j}\leq 10^4$

$4s,512MB$

###### Sol

考虑 $r_{i,j}\leq 1$ 的情况，此时显然可以进行插头dp，复杂度为 $O(nm2^m)$。并且显然可以通过记录 $dp$ 的转移点，在 $dp$ 时求出一种方案。

对于任意的情况，存在类似保序回归的结论：对于任意 $x$，如果让所有 $r_{i,j}\leq x$ 的点只能取 $x$，所有 $l_{i,j}>x$ 的点只能取 $x+1$，剩下的点可以取 $\{x,x+1\}$，求出此时的最优解。则存在一个整体的最优解，满足 $v_{i,j}\leq x$ 当且仅当 $(i,j)$ 在这种方案中取 $x$。

证明：[我不会](t3.pdf)。

此时考虑整体二分，记录当前的 $[l,r]$ 以及取值确定在这个区间内的位置集合。每次选出 $mid$，只考虑这些位置，取 $\{mid,mid+1\}$ 的值，求此时的最优解。根据结论，这里取 $mid$ 的最后的取值为 $[l,mid]$，取 $mid+1$ 的最后的取值为 $[mid+1,r]$。因此做类似整体二分的操作即可。

复杂度 $O(nm2^m\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#define N 50050
using namespace std;
#define ll long long
int n,m,l[N][7],r[N][7],vx[N][7],vy[N][7],as[N][7],is[N][7],as1[N][7];
ll dp[N][7][33];
int fr[N][7][33];
int getst(int x,int y,int vl)
{
	if(is[x][y])return 2;
	else if(as[x][y]<=vl)return 0;
	else return 1;
}
void doit(int l,int r,vector<pair<int,int> > sr,int vl)
{
	for(int i=l;i<=r+2;i++)
		for(int j=1;j<=m;j++)
			for(int k=0;k<1<<m;k++)dp[i][j][k]=1e14;
	int c1=0;
	for(int i=1;i<=m;i++)c1|=getst(l-1,i,vl)<<(m-i);
	dp[l][1][c1]=0;
	for(int i=l;i<=r+1;i++)
		for(int j=1;j<=m;j++)
		{
			int st=getst(i,j,vl),l1=0,r1=1;
			if(st==0)r1=0;if(st==1)l1=1;
			for(int k=0;k<1<<m;k++)
				for(int p=l1;p<=r1;p++)
				{
					ll vl=dp[i][j][k];
					int ntx=i,nty=j+1,nts=((k<<1)&((1<<m)-1))|p;
					if(nty>m)ntx++,nty=1;
					if(j>1&&(k&1)!=p)vl+=vy[i][j-1];
					if((k>>m-1)!=p)vl+=vx[i-1][j];
					if(dp[ntx][nty][nts]>vl)dp[ntx][nty][nts]=vl,fr[ntx][nty][nts]=k*2+p;
				}
		}
	ll mn=1e15;
	int sx=r+2,sy=1,f1=0;
	for(int i=0;i<1<<m;i++)if(dp[sx][sy][i]<mn)mn=dp[sx][sy][i],f1=i;
	while(sx>l||sy>1)
	{
		int lsx=sx,lsy=sy-1,f2=fr[sx][sy][f1];
		if(!lsy)lsy=m,lsx--;
		as1[lsx][lsy]=vl+(f2&1);
		f1=f2>>1;sx=lsx,sy=lsy;
	}
}
void solve(int lv,int rv,vector<pair<int,int> > fu)
{
	if(!fu.size())return;
	sort(fu.begin(),fu.end());
	if(lv==rv)
	{
		for(int i=0;i<fu.size();i++)
		{
			int lx=fu[i].first,ly=fu[i].second;
			as[lx][ly]=lv;
		}
		return;
	}
	int mid=(lv+rv)>>1;
	vector<pair<int,int> > l1,r1,f1,f2;
	for(int i=0;i<fu.size();i++)
	{
		int lx=fu[i].first,ly=fu[i].second;
		if(r[lx][ly]<=mid)l1.push_back(fu[i]),as[lx][ly]=mid;
		else if(l[lx][ly]>mid)r1.push_back(fu[i]),as[lx][ly]=mid+1;
		else f1.push_back(fu[i]),is[lx][ly]=1;
	}
	int ls=-1,lb=-1;
	for(int i=0;i<f1.size();i++)
	{
		int nx=f1[i].first;
		if(ls==-1)ls=lb=nx;
		else if(ls>=nx-1)ls=nx;
		else doit(lb,ls,f2,mid),lb=ls=nx,f2.clear();
		f2.push_back(f1[i]);
	}
	if(ls!=-1)doit(lb,ls,f2,mid);
	for(int i=0;i<f1.size();i++)
	{
		int lx=f1[i].first,rx=f1[i].second;
		is[lx][rx]=0;
		if(as1[lx][rx]<=mid)l1.push_back(f1[i]),as[lx][rx]=mid;
		else r1.push_back(f1[i]),as[lx][rx]=mid+1;
	}
	solve(lv,mid,l1);solve(mid+1,rv,r1);
}
int asb(int x){return x>0?x:-x;}
int main()
{
	freopen("grid.in","r",stdin);
	freopen("grid.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%d",&l[i][j]),as[i][j]=l[i][j];
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%d",&r[i][j]);
	for(int i=1;i<n;i++)for(int j=1;j<=m;j++)scanf("%d",&vx[i][j]);
	for(int i=1;i<=n;i++)for(int j=1;j<m;j++)scanf("%d",&vy[i][j]);
	vector<pair<int,int> > s1;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)s1.push_back(make_pair(i,j));
	solve(0,10000,s1);
	ll ras=0;
	for(int i=1;i<=n;i++)for(int j=1;j<m;j++)ras+=vy[i][j]*asb(as[i][j]-as[i][j+1]);
	for(int i=1;i<n;i++)for(int j=1;j<=m;j++)ras+=vx[i][j]*asb(as[i][j]-as[i+1][j]);
	printf("%lld\n",ras);
}
```

#### 0710

##### T1 path

###### Problem

给一棵 $n$ 个点的树，支持 $q$ 次如下操作：

1. 删除一条边再加入一条边，保证操作后为一棵树。
2. 给出 $a,b$，求有多少条从 $a$ 开始到 $b$ 结束的路径满足经过每个点不经过两次。

$n,q\leq 10^5$

$0.5s,512MB$

###### Sol

考虑 $a=b$ 的情况，以 $a$ 为根，则路径一定是从根开始向下走在向上走。对于一个点，第一次经过它一定是从上往下，因此第二次是从下往上。此时可以发现，合法的路径一定是从根到达某个点，然后再原路返回根。因此方案有 $n$ 种。

考虑 $a\neq b$ 的情况，提出 $a$ 到 $b$ 的链， 设链上的点为 $v_1,...,v_k$。可以发现，如果在链上的路径不出现折返，则在链上的每个点上都可以向这个点的其它子树内走一条路径，因此记每个点的虚子树大小为这个点所有不在链上的子树的大小和，则答案即为链上每个点虚子树大小加一的乘积。

考虑在链上出现折返的情况，因为每个点只能经过两次，因此一次折返只会向回走一步，可以发现折返一定形如 $v_{i-1}\to v_i\to v_{i+1}\to v_i\to v_{i+1}\to v_{i+2}$。可以看成这次折返覆盖了点 $(v_i,v_{i+1})$。可以发现折返一定满足如下限制：

1. 每次折返一定覆盖两个链上相邻的点。
2. 一个点只被覆盖一次。

此时如果一个点没有被覆盖，则此时它可以向它的虚子树内走，方案数为虚子树大小 $+1$，如果被覆盖则方案数为 $1$。

因此可以设 $dp_{i,0/1}$ 表示考虑链上前 $i$ 个点，是否覆盖了 $(v_i,v_{i+1})$，前面的方案数。设虚子树大小为 $lsz_i$，可以发现转移即为乘上如下 $2\times 2$ 矩阵：
$$
\begin{bmatrix}
dp_{i-1,0}\\
dp_{i-1,1}\\
\end{bmatrix}\times \begin{bmatrix}
lsz_{v_i}+1&1\\
1&0\\
\end{bmatrix} = \begin{bmatrix}
dp_{i,0}\\
dp_{i,1}\\
\end{bmatrix}
$$
考虑使用LCT维护树，如果进行 $split(u,v)$，则此时每个点的虚子树大小即为需要的大小，因此在LCT上维护这个 $dp$ 的矩阵乘法即可。

因为makeroot需要翻转，因此需要维护矩阵正向反向乘的结果，但存在如下性质：

对于矩阵 $M_1,M_2,\dots,M_k$，如果所有这些矩阵转置后都不变，则 $M_1\times M_2\times\dots\times M_k$ 转置后即为 $M_k\times M_{k-1}\times \dots\times M_1$。

~~应该是对的，证明留作练习，至少在这个题的矩阵下显然是对的~~

因此只需要维护一个矩阵即可，常数较小。~~需要卡亿点常~~

###### Code

~~NOI D2T2的代码有一部分来源于这里~~

~~另外一部分来源于下一场的T1~~

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define mod 998244353
int n,q,a,b,c,d,e,head[N],cnt,fa[N],ch[N][2],lz[N],sz[N],ls[N],st[N],c1;
struct edge{int t,next;}ed[N*2];
struct mat{int v[4];mat(){v[0]=v[3]=1;v[1]=v[2]=0;}}dp[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int f)
{
	fa[u]=f;sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=f)dfs(ed[i].t,u),sz[u]+=sz[ed[i].t];
	ls[u]=sz[u]-1;
}
mat operator *(mat a,mat b)
{
	mat c;
	c.v[0]=(1ll*a.v[0]*b.v[0]+1ll*a.v[1]*b.v[2])%mod;
	c.v[1]=(1ll*a.v[0]*b.v[1]+1ll*a.v[1]*b.v[3])%mod;
	c.v[2]=(1ll*a.v[2]*b.v[0]+1ll*a.v[3]*b.v[2])%mod;
	c.v[3]=(1ll*a.v[2]*b.v[1]+1ll*a.v[3]*b.v[3])%mod;
	return c;
}
void pushup(int x)
{
	sz[x]=ls[x]+1+sz[ch[x][0]]+sz[ch[x][1]];
	mat s1;s1.v[0]=ls[x]+1;s1.v[1]=s1.v[2]=1;s1.v[3]=0;
	dp[x]=dp[ch[x][0]]*s1*dp[ch[x][1]];
}
void doit(int x){lz[x]^=1;swap(ch[x][0],ch[x][1]);swap(dp[x].v[1],dp[x].v[2]);}
void pushdown(int x){if(lz[x])doit(ch[x][0]),doit(ch[x][1]),lz[x]=0;}
bool nroot(int x){return ch[fa[x]][0]==x||ch[fa[x]][1]==x;}
void rotate(int x)
{
	int f=fa[x],g=fa[f],tp=ch[f][1]==x;
	if(ch[g][ch[g][1]==f]==f)ch[g][ch[g][1]==f]=x;
	fa[x]=g;
	ch[f][tp]=ch[x][!tp],fa[ch[x][!tp]]=f;
	ch[x][!tp]=f,fa[f]=x;
	pushup(f),pushup(x);
}
void splay(int x)
{
	c1=0;
	int nw=x;
	while(nroot(nw))st[++c1]=nw,nw=fa[nw];
	pushdown(nw);for(int i=c1;i>=1;i--)pushdown(st[i]);
	while(nroot(x))
	{
		int f=fa[x],g=fa[f];
		if(nroot(f))rotate(((ch[f][1]==x)^(ch[g][1]==f))?x:f);
		rotate(x);
	}
}
void access(int x)
{
	int tp=0;
	while(x)
	{
		splay(x);ls[x]=ls[x]+sz[ch[x][1]]-sz[tp];
		ch[x][1]=tp;pushup(x);tp=x;x=fa[x];
	}
}
void makeroot(int x){access(x);splay(x);doit(x);splay(x);}
void split(int x,int y){makeroot(x);access(y);splay(y);}
void link(int x,int y)
{
	makeroot(x);access(y);splay(y);
	ls[y]+=sz[x];fa[x]=y;pushup(y);
}
void cut(int x,int y)
{
	split(x,y);splay(x);
	ch[x][1]=fa[y]=0;
	pushup(x);
}
int main()
{
	freopen("path.in","r",stdin);
	freopen("path.out","w",stdout);
	scanf("%d%d",&n,&q);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	for(int i=1;i<=n;i++)pushup(i);
	while(q--)
	{
		scanf("%d",&a);
		if(a==1)scanf("%d%d%d%d",&b,&c,&d,&e),cut(b,c),link(d,e);
		else
		{
			scanf("%d%d",&b,&c);
			split(b,c);
			printf("%d\n",dp[c].v[0]);
		}
	}
}
```

##### T2 danger

###### Problem

有一个 $n\times m$ 的矩阵，每个位置上有一个数 $v_{i,j}$。

有一个棋子初始在 $(1,1)$，棋子循环进行如下操作：

设棋子在 $(x,y)$，首先让 $v_{x,y}$ 变为 $\max(v_{x,y}-1,0)$，然后进行如下操作：

1. 有 $p_r$ 的概率向右移动一格。
2. 有 $p_d$ 的概率向下移动一格。
3. 有 $1-p_r-p_d$ 的概率，将 $v_{x,y}$ 变为 $0$，随后结束所有操作。

矩阵在横纵坐标上都是循环的，因此在 $(x,m)$ 向右会移动到 $(x,1)$，在 $x$ 坐标上同理。

求操作结束时，所有 $v_{i,j}$ 减少值的和的期望，模 $998244353$。 

$n,m\leq 200$

$1s,512MB$

###### Sol

考虑期望线性性，可以分别考虑每个位置对答案的贡献再求和。

对于一个位置，只有当棋子经过过它，它才会产生贡献。因此考虑求出从起点开始游走经过点 $(i,j)$ 的概率 $p_{i,j}$。

考虑第一次到达一个位置之后，因为只考虑这个位置的贡献，可以将这个位置作为原点，此时从这个位置出发，再回到这个位置的概率为 $p_{1,1}$，因此最后经过 $k$ 次这个位置的概率为 $p_{i,j}*p_{1,1}^{k-1}*(1-p_{1,1})$。

考虑这个位置的贡献，分为在这个位置停止和不停止的情况，第一种情况的贡献为 $v_{i,j}*\sum_kp_{i,j}*p_{1,1}^{k-1}*(1-p_r-p_d)$，第二种情况的贡献为 $\sum_{k}\min(k,v_{i,j})*p_{i,j}*p_{1,1}^{k-1}*(1-p_{1,1}-(1-p_r-p_d))$，显然可以 $O(\log v)$ 计算。

因此只需要求出 $p$ 即可求出答案，但直接求 $p$ 难以建立 $p$ 之间的关系。

考虑记录 $c_{i,j}$ 表示从起点出发，经过 $(i,j)$ 的期望次数。显然 $c_{i,j}=[(i,j)=(1,1)]+p_rc_{i,j-1}+p_dc_{i-1,j}$，直接消元复杂度 $O(n^3m^3)$。

考虑将所有第一排第一列的点设为关键点。对于一个非关键点 $(i,j)$ ，通过上面的递推式，一定可以将 $c_{i,j}$ 表示成关键点的 $c$ 的线性组合。

此时考虑关键点之间，可以得到 $n+m-1$ 个方程和变量，可以发现这样一定能解出 $c$，复杂度 $O((n+m)^3)$。最后可以在相同复杂度内求出所有的 $c$。

考虑通过 $c$ 求出 $v$。显然 $c_{1,1}=\sum_{k\geq 0}v_{1,1}^k$，因此 $c_{1,1}=\frac 1{1-v_{1,1}}$，可以求出 $v_{1,1}$。对于剩余的点，可以发现 $c_{i,j}=v_{i,j}*\frac 1{1-v_{1,1}}$，因此可以求出所有的 $v$。

复杂度 $O((n+m)^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 405
#define mod 998244353
int n,m,a,b,p1,p2,p3,v[N][N],s[N][N],su,s1[N],c[N][N],id[N][N],vl1[N][N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void solve()
{
	for(int i=1;i<=su;i++)
	{
		int st=i;
		for(int j=i;j<=su;j++)if(v[j][i])st=j;
		for(int j=1;j<=su+1;j++)swap(v[st][j],v[i][j]);
		for(int j=1;j<=su;j++)if(j!=i)
		{
			int vl=1ll*(mod-1)*v[j][i]%mod*pw(v[i][i],mod-2)%mod;
			for(int k=1;k<=su+1;k++)v[j][k]=(v[j][k]+1ll*v[i][k]*vl)%mod;
		}
	}
	for(int i=1;i<=su;i++)s1[i]=1ll*(mod-1)*v[i][su+1]%mod*pw(v[i][i],mod-2)%mod;
}
int main()
{
	freopen("danger.in","r",stdin);
	freopen("danger.out","w",stdout);
	scanf("%d%d",&n,&m);
	scanf("%d%d",&a,&b),p1=1ll*a*pw(b,mod-2)%mod;
	scanf("%d%d",&a,&b),p2=1ll*a*pw(b,mod-2)%mod;
	p3=(mod*2+1-p1-p2)%mod;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%d",&s[i][j]);
	su=n+m-1;
	for(int i=1;i<=n;i++)id[i][1]=i;
	for(int i=2;i<=m;i++)id[1][i]=i+n-1;
	for(int i=0;i<=su+1;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=su+1;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	for(int i=0;i<=n;i++)
		for(int j=0;j<=m;j++)
			vl1[i][j]=1ll*c[i+j][j]*pw(p1,j)%mod*pw(p2,i)%mod;
	for(int i=1;i<=n;i++)
	{
		int s1=id[i][1];
		v[s1][s1]=mod-1;
		v[s1][id[(i+n-2)%n+1][1]]=(v[s1][id[(i+n-2)%n+1][1]]+p2)%mod;
		if(m==1||i==1)v[s1][id[i][m]]=(v[s1][id[i][m]]+p1)%mod;
		else
		{
			int nx=i,ny=m;
			for(int j=2;j<=nx;j++)v[s1][id[j][1]]=(v[s1][id[j][1]]+1ll*p1*p1%mod*vl1[nx-j][ny-2])%mod;
			for(int j=2;j<=ny;j++)v[s1][id[1][j]]=(v[s1][id[1][j]]+1ll*p1*p2%mod*vl1[nx-2][ny-j])%mod;
		}
	}
	for(int i=2;i<=m;i++)
	{
		int s1=id[1][i];
		v[s1][s1]=mod-1;
		v[s1][id[1][(i+m-2)%m+1]]=(v[s1][id[1][(i+m-2)%m+1]]+p1)%mod;
		if(n==1)v[s1][id[n][i]]=(v[s1][id[n][i]]+p2)%mod;
		else
		{
			int nx=n,ny=i;
			for(int j=2;j<=nx;j++)v[s1][id[j][1]]=(v[s1][id[j][1]]+1ll*p2*p1%mod*vl1[nx-j][ny-2])%mod;
			for(int j=2;j<=ny;j++)v[s1][id[1][j]]=(v[s1][id[1][j]]+1ll*p2*p2%mod*vl1[nx-2][ny-j])%mod;
		}
	}
	v[1][su+1]=1;
	solve();
	int pr=(mod+1-pw(s1[1],mod-2))%mod,as=0;
	for(int i=1;i<=n;i++)
		for(int j=1;j<=m;j++)
		{
			int vl=0;
			if(id[i][j])vl=s1[id[i][j]];
			else
			{
				int nx=i,ny=j;
				for(int j=2;j<=nx;j++)vl=(vl+1ll*p1*vl1[nx-j][ny-2]%mod*s1[id[j][1]])%mod;
				for(int j=2;j<=ny;j++)vl=(vl+1ll*p2*vl1[nx-2][ny-j]%mod*s1[id[1][j]])%mod;
			}
			as=(as+1ll*vl*p3%mod*s[i][j])%mod;
			as=(as+1ll*vl*(mod*2+1-p3-pr)%mod*(mod+1-pw(pr,s[i][j]))%mod*pw(mod+1-pr,mod-2))%mod;
		}
	printf("%d\n",as);
}
```



##### T3 id

###### Problem

给定 $n$ 组字符串 $(a_i,b_i,c_i)$，你有三个字符串 $A,B,C$，初始全部为空。

你需要支持 $q$ 次操作，每次操作为向 $A,B,C$ 中的一个字符串的末尾加入一个字符或者删除一个字符。

在每次操作后，你需要输出当前有多少组给出的字符串 $(a_i,b_i,c_i)$ 满足 $A$ 是 $a_i$ 的前缀，$B$ 是 $b_i$ 的前缀，$C$ 是 $c_i$ 的前缀。

$n,m,\sum |a_i|+|b_i|+|c_i|\leq 5\times 10^5$

$1s,512MB$

###### Sol

考虑对所有的 $a$ 串建trie，显然 $A$ 是 $a_i$ 的前缀当且仅当 $A$ 在串上对应的节点是 $a_i$ 对应节点的祖先。

考虑维护当前 $A$ 对应的trie上节点。如果当前 $A$ 对应了一个trie上节点，则加入/删除字符时可以直接走trie的边。

如果 $A$ 不对应trie上节点，则可以记录当前 $A$ 最长的在trie上的前缀对应的位置以及长度即可。

现在问题可以看成，给三棵有根树，有 $n$ 个三元组 $(a_i,b_i,c_i)$，给 $q$ 次询问，每次给 $a,b,c$，求有多少个三元组满足 $a_i$ 在 $A$ 子树内，$b_i,c_i$ 同理。

这显然可以看成三维数点，直接做复杂度 $O(n\log^2 n)$

因为字符串总长只有 $5\times 10^5$，对于每一个三元组 $a_i,b_i,c_i$，考虑 $a_i$ 的每一个前缀，得到所有的 $(a_i[1,l],b_i,c_i)$，这样的三元组数量只有 $O(l)$ 个。

显然，如果 $(a_i,b_i,c_i)$ 对于询问 $(A,B,C)$ 满足条件，则正好存在一个 $l$，使得 $(a_i[1,l],b_i,c_i)$ 中，第一个串与 $A$ 相等，后两个串包含 $B/C$ 作为前缀。

因此可以分别考虑每一种 $A$，对于每一种做二维数点即可。

复杂度 $O((n+l)\log l)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 500500
int n,q,ch[N][26],fa[N],lb[N],rb[N],v[N][4],ct=3,le[4],l1[4],nw[4],as[N],c1;
char s[N];
struct sth{int op,x,y;};
bool operator <(sth a,sth b){return a.x==b.x?a.op<b.op:a.x<b.x;}
vector<sth> s1[N];
int ins(int rt)
{
	for(int i=1;s[i];i++)
	{
		int as=ch[rt][s[i]-'a'];
		if(!as)as=ch[rt][s[i]-'a']=++ct,fa[ct]=rt;
		rt=as;
	}
	return rt;
}
void dfs(int x)
{
	lb[x]=++c1;
	for(int i=0;i<26;i++)if(ch[x][i])dfs(ch[x][i]);
	rb[x]=c1;
}
void add1(int x,int c)
{
	if(le[x]<l1[x])l1[x]++;
	else if(!ch[nw[x]][c])l1[x]++;
	else nw[x]=ch[nw[x]][c],l1[x]++,le[x]++;
}
void del1(int x)
{
	if(le[x]<l1[x])l1[x]--;
	else nw[x]=fa[nw[x]],l1[x]--,le[x]--;
}
struct BIT{
	int tr[N];
	void modify(int x,int v){for(int i=x;i<=500000;i+=i&-i)tr[i]+=v;}
	int query(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}tr;
int main()
{
	freopen("id.in","r",stdin);
	freopen("id.out","w",stdout);
	for(int j=1;j<=3;j++)nw[j]=j;
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
		for(int j=1;j<=3;j++)
			scanf("%s",s+1),v[i][j]=ins(j);
	for(int i=1;i<=3;i++)c1=0,dfs(i);
	for(int i=1;i<=n;i++)
		for(int j=v[i][1];j;j=fa[j])s1[j].push_back((sth){-1,lb[v[i][2]],lb[v[i][3]]});
	scanf("%d",&q);
	for(int i=1;i<=q;i++)
	{
		scanf("%s",s);
		int op=s[0]=='+';
		scanf("%s",s);
		int id=s[0]-'0';
		if(op)scanf("%s",s),add1(id,s[0]-'a');
		else del1(id);
		int fg=1;
		for(int j=1;j<=3;j++)if(l1[j]>le[j])fg=0;
		if(!fg)continue;
		int v1=nw[1],v2=nw[2],v3=nw[3];
		s1[v1].push_back((sth){i,rb[v2],rb[v3]});
		s1[v1].push_back((sth){i+q,lb[v2]-1,rb[v3]});
		s1[v1].push_back((sth){i+q,rb[v2],lb[v3]-1});
		s1[v1].push_back((sth){i,lb[v2]-1,lb[v3]-1});
	}
	for(int i=1;i<=ct;i++)if(s1[i].size())
	{
		sort(s1[i].begin(),s1[i].end());
		for(int j=0;j<s1[i].size();j++)
		{
			sth t1=s1[i][j];
			if(t1.op==-1)tr.modify(t1.y,1);
			else
			{
				int t2=t1.op,vl=1;
				if(t2>q)t2-=q,vl=-1;
				as[t2]+=vl*tr.query(t1.y);
			}
		}
		for(int j=0;j<s1[i].size();j++)
		{
			sth t1=s1[i][j];
			if(t1.op==-1)tr.modify(t1.y,-1);
		}
	}
	for(int i=1;i<=q;i++)printf("%d\n",as[i]);
}
```

#### 0712

##### T1 calc

Source: CF730L

###### Problem

给一个长度为 $n$ 的，包含 `+*()` 以及数字的合法算术表达式 $S$。

$q$ 次询问，每次给出 $l,r$，求 $S[l,r]$ 求值的结果，模 $10^9+9$。如果 $S[l,r]$ 不是合法表达式则输出 $-1$。

$n,q\leq 5\times 10^5$

$3s,512MB$

###### Sol

如果 $l,r$ 不在同一个括号下，显然答案为 $-1$。可以扫一遍求出括号配对的情况，并求出每个位置属于哪一对括号。

如果 $l,r$ 在同一个括号下，则考虑它们时，可以将这个括号内的子括号全部替换为子括号内部求值的结果，因此此时相当于有一个没有括号的表达式，求一个区间求值的结果。

如果将一个数看成一个整体，则显然可以简单地使用线段树维护。线段树上每个点表示一个区间的表达式，叶子节点为每一个数。记录当前区间的运算符是否全为乘，当前左侧的连乘的值，右侧的连乘值以及中间部分的值，即可简单合并。

对于一个询问，它只会影响跨过询问端点的数的值，然后相当于求询问区间内部求值的结果。可以直接在线段树上查询中间部分，再加上两侧被修改的数。注意询问区间内部没有运算符的情况。

最后考虑处理括号求值的问题，可以看成在表达式树上dfs，从下往上求值。实现时不需要建出表达式树，只需要记录当前处理到了哪一对括号即可。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<cstring>
using namespace std;
#define N 500500
#define mod 1000000009
int n,q,l,r,st[N],ct,bel[N],pw[N],su[N],lb[N],rb[N],as[N],v1[N],sl[N],sr[N],tp[N],vl[N];
char s[N];
struct qu1{int l,r,id;};
vector<qu1> qu[N];
int calc(int l,int r){return (su[r]-1ll*su[l-1]*pw[r-l+1]%mod+mod)%mod;}
struct sth{int fg,v1,v2,v3;};
struct node{int l,r;sth tp;}e[N*4];
sth doit(sth s1,sth s2,int fg)
{
	sth s3;
	if(s1.fg&&s2.fg&&fg){s3.v1=s3.v2=1ll*s1.v1*s2.v2%mod;s3.v3=0;s3.fg=1;return s3;}
	s3.fg=0;
	if(s1.fg&&fg)s3.v1=1ll*s1.v1*s2.v1%mod;else s3.v1=s1.v1;
	if(s2.fg&&fg)s3.v2=1ll*s1.v2*s2.v2%mod;else s3.v2=s2.v2;
	s3.v3=(s1.v3+s2.v3)%mod;
	if(!s1.fg&&!s2.fg&&fg)s3.v3=(s3.v3+1ll*s1.v2*s2.v1)%mod;
	else{if(!s1.fg&&(!s2.fg||!fg))s3.v3=(s3.v3+s1.v2)%mod;if(!s2.fg&&(!s1.fg||!fg))s3.v3=(s3.v3+s2.v1)%mod;}
	return s3;
}
void pushup(int x){
	int mid=(e[x].l+e[x].r)>>1;
	e[x].tp=doit(e[x<<1].tp,e[x<<1|1].tp,s[tp[mid+1]]=='*');
}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;
	if(l==r){e[x].tp=(sth){1,vl[l],vl[l],0};return;}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);
}
sth query(int x,int l,int r)
{
	if(e[x].l==l&&e[x].r==r)return e[x].tp;
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)return query(x<<1,l,r);
	else if(mid<l)return query(x<<1|1,l,r);
	else return doit(query(x<<1,l,mid),query(x<<1|1,mid+1,r),s[tp[mid+1]]=='*');
}
void solve(int l,int r)
{
	int ct=0,ls=r;
	for(int i=r-1;i>=l;i--)if(s[i]==')')solve(lb[i],i),i=lb[i];else if(s[i]=='+'&&s[i]=='-')ls=i;else sr[i]=ls;
	ls=l;
	for(int i=l+1;i<=r;i++)if(s[i]=='(')i=rb[i];else if(s[i]=='+'||s[i]=='*')tp[++ct]=i,ls=i;else sl[i]=ls;
	tp[0]=l;tp[ct+1]=r;
	for(int i=0;i<=ct;i++)
	if(s[tp[i]+1]=='(')vl[i]=v1[tp[i]+1];else vl[i]=calc(tp[i]+1,tp[i+1]-1);
	build(1,0,ct);
	for(int i=0;i<qu[l].size();i++)
	{
		int l1=qu[l][i].l,r1=qu[l][i].r,id=qu[l][i].id;
		int l2=lower_bound(tp,tp+ct+1,l1)-tp,r2=lower_bound(tp,tp+ct+1,r1)-tp;
		if(l2==r2){as[id]=calc(l1,r1);if(s[tp[l2-1]+1]=='(')as[id]=vl[l2-1];continue;}
		int lv=calc(l1,tp[l2]-1),rv=calc(tp[r2-1]+1,r1);
		if(s[tp[l2-1]+1]=='(')lv=vl[l2-1];if(s[tp[r2-1]+1]=='(')rv=vl[r2-1];
		if(l2+1==r2)
		{
			int fg=s[tp[l2]]=='*';
			if(fg)as[id]=1ll*lv*rv%mod;
			else as[id]=(lv+rv)%mod;
			continue;
		}
		sth v0=query(1,l2,r2-2);
		v0=doit((sth){1,lv,lv,0},v0,s[tp[l2]]=='*');
		v0=doit(v0,(sth){1,rv,rv,0},s[tp[r2-1]]=='*');
		as[id]=v0.fg?v0.v1:(1ll*v0.v1+(!v0.fg)*v0.v2+v0.v3)%mod;
	}
	sth v0=query(1,0,ct);
	v1[l]=(1ll*v0.v1+(!v0.fg)*v0.v2+v0.v3)%mod;
}
int rd()
{
	int as=0;
	char c=getchar();
	while(c<'0'||c>'9')c=getchar();
	while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();
	return as;
}
void pr(int x)
{
	if(!x)return;
	int tp=x/10;
	pr(tp);putchar('0'+x-tp*10);
}
void pr1(int x)
{
	if(!x)putchar('0');
	else if(x<0)x*=-1,putchar('-');
	pr(x);
}
int main()
{
	freopen("calc.in","r",stdin);
	freopen("calc.out","w",stdout);
	n=rd();q=rd();
	scanf("%s",s+2);s[1]='(';n+=2;s[n]=')';pw[0]=1;
	for(int i=1;i<=n;i++)
	{
		pw[i]=10ll*pw[i-1]%mod;su[i]=10ll*su[i-1]%mod;bel[i]=st[ct];
		if(s[i]=='(')st[++ct]=i;
		else if(s[i]==')')rb[st[ct]]=i,lb[i]=st[ct],ct--,bel[i]=st[ct];
		else if(s[i]>='0'&&s[i]<='9')su[i]=(su[i]+s[i]-'0')%mod;
	}
	for(int i=1;i<=q;i++)
	{
		l=rd();r=rd();l++;r++;
		if(bel[l]!=bel[r]||s[l]=='*'||s[l]=='+'||s[r]=='*'||s[r]=='+'||s[l]==')'||s[r]=='(')as[i]=-1;
		else qu[bel[l]].push_back((qu1){l,r,i});
	}
	solve(1,n);
	for(int i=1;i<=q;i++)pr1(as[i]),putchar('\n');
}
```

##### T2 scc

Source: uoj451

###### Problem

给一个竞赛图，初始没有点。

有 $n$ 次加点操作，对于第 $i$ 次加点操作，加入的点编号为 $i$，给出 $c_i$ 个两两不交的区间 $[l_{i,j},r_{i,j}]$，如果 $x$ 被一个区间包含，则连边方向为 $i\to x$，否则方向为 $x\to i$。

在每次加点后，求出当前图的强连通分量数量。强制在线。

$n\leq 2\times 10^5,\sum c_i\leq 2\times 10^6$

$2s,16MB$

###### Sol

显然竞赛图缩强连通后会得到一条链，考虑维护所有点在链上的相对顺序。

设当前缩点后的链为 $v_1\to v_2\to\dots\to v_k$，考虑加入点 $x$ 的情况：

求出 $x$ 连向的点中，编号最小的点，设为 $v_{mn}$。求出连向 $x$ 的点中编号最大的点，设为 $v_{mx}$。

此时如果 $mx<mn$，则一定有 $mx+1=mn$，可以发现此时将 $x$ 插入链中 $v_{mx}\to v_{mn}$ 的位置即可。

否则，可以发现 $[mn,mx]$ 中的所有强连通分量会合并起来，同时 $x$ 也在这个强连通分量中。此时因为强连通分量内部的顺序不重要，可以直接将 $x$ 的相对顺序设在 $v_{mx}$ 后面而不影响上面的过程。

因为给出的边是若干个区间，因此考虑使用线段树维护区间内相对顺序最小和最大的点。此时在线段树上需要 $O(\sum c\log n)$ 次比较两个点的相对顺序，因此需要一个数据结构支持如下操作：

维护一个序列，支持 $O(\log n)$ 插入，$O(1)$ 比较两个元素的位置关系，强制在线（复杂度可以均摊）。

考虑一棵BST，给根节点一个区间 $[0,10^{18}]$。如果一个节点的区间为 $[l,r]$，设 $mid=\lfloor\frac{l+r}2\rfloor$，则左儿子的区间为 $[l,mid]$，右儿子的区间为 $[mid+1,r]$。此时在树深度不超过 $\log_2 10^{18}$ 的情况下，可以通过直接比较区间得到两个点的顺序关系。

此时将BST换成替罪羊树，通过重构即可使复杂度为均摊 $O(\log n)$

最后考虑将一个区间内的所有集合合并，使用并查集，可以看成每次找下一个集合合并，直到 $mn,mx$ 属于一个集合。将并查集设为以最后一个点为代表节点，在替罪羊树上找代表节点的后继即可。

复杂度 $O((n+\sum c_i)\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,t,k,a,b,f1[N],as;
pair<int,int> tp[N];
int finds(int x){return f1[x]==x?x:f1[x]=finds(f1[x]);}
int sz[N],ch[N][2],fa[N],st[N],ct;
ll lb[N],rb[N];
bool cmp(int x,int y){return lb[x]+rb[x]<lb[y]+rb[y];}
int cmin(int x,int y){return cmp(x,y)?x:y;}
int cmax(int x,int y){return cmp(x,y)?y:x;}
int mn[N*4],mx[N*4];
void pushup(int x){mn[x]=cmin(mn[x<<1],mn[x<<1|1]),mx[x]=cmax(mx[x<<1],mx[x<<1|1]);}
void modify(int x,int l1,int r1,int v)
{
	if(l1==r1){mn[x]=mx[x]=v;return;}
	int mid=(l1+r1)>>1;
	if(mid>=v)modify(x<<1,l1,mid,v);else modify(x<<1|1,mid+1,r1,v);
	pushup(x);
}
int querymn(int x,int l1,int r1,int l,int r)
{
	if(l==l1&&r==r1)return mn[x];
	int mid=(l1+r1)>>1;
	if(mid>=r)return querymn(x<<1,l1,mid,l,r);
	else if(mid<l)return querymn(x<<1|1,mid+1,r1,l,r);
	else return cmin(querymn(x<<1,l1,mid,l,mid),querymn(x<<1|1,mid+1,r1,mid+1,r));
}
int querymx(int x,int l1,int r1,int l,int r)
{
	if(l==l1&&r==r1)return mx[x];
	int mid=(l1+r1)>>1;
	if(mid>=r)return querymx(x<<1,l1,mid,l,r);
	else if(mid<l)return querymx(x<<1|1,mid+1,r1,l,r);
	else return cmax(querymx(x<<1,l1,mid,l,mid),querymx(x<<1|1,mid+1,r1,mid+1,r));
}
void dfs(int x){if(!x)return;dfs(ch[x][0]);st[++ct]=x;dfs(ch[x][1]);}
int build(int l,int r,int x,int tp)
{
	if(l>r)return 0;
	int mid=(l+r)>>1,v=st[mid];
	if(tp)rb[v]=rb[x],lb[v]=(lb[x]+rb[x])>>1;
	else lb[v]=lb[x],rb[v]=(lb[x]+rb[x])>>1;
	fa[v]=x;
	ch[v][0]=build(l,mid-1,v,0);ch[v][1]=build(mid+1,r,v,1);
	sz[v]=sz[ch[v][0]]+sz[ch[v][1]]+1;
	return v;
}
void rebuild(int x){ct=0;dfs(x);ch[fa[x]][ch[fa[x]][1]==x]=build(1,ct,fa[x],ch[fa[x]][1]==x);}
void insert(int x,int y)
{
	sz[y]=1;
	if(!ch[x][1]){ch[x][1]=y;fa[y]=x;rb[y]=rb[x];lb[y]=(lb[x]+rb[x])>>1;}
	else
	{
		x=ch[x][1];while(ch[x][0])x=ch[x][0];
		ch[x][0]=y;fa[y]=x;lb[y]=lb[x];rb[y]=(lb[x]+rb[x])>>1;
	}
	while(x)
	{
		sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1;
		if(sz[x]*0.723<max(sz[ch[x][0]],sz[ch[x][1]]))rebuild(x);
		x=fa[x];
	}
}
int getnxt(int x)
{
	if(ch[x][1])
	{
		x=ch[x][1];
		while(ch[x][0])x=ch[x][0];
		return x;
	}
	while(x)
	{
		if(ch[fa[x]][0]==x)return fa[x];
		x=fa[x];
	}
	return 0;
}
void merge(int l,int r)
{
	while(cmp(l,r))
	{
		int v1=finds(l),v2=getnxt(v1);
		f1[v1]=v2,as--;l=finds(l);
	}
}
int main()
{
	freopen("scc.in","r",stdin);
    freopen("scc.out","w",stdout);
	scanf("%d%d",&n,&t);
	ch[0][0]=n+1;rb[0]=2e18;
	lb[n+1]=0,rb[n+1]=1e18;
	lb[n+2]=5e17,rb[n+2]=1e18;ch[n+1][1]=n+2;fa[n+2]=n+1;
	for(int i=1;i<=n+2;i++)f1[i]=i;
	modify(1,1,n,1);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&k);
		for(int j=1;j<=k;j++)scanf("%d%d",&a,&b),a=(a+t*as-1)%i+1,b=(b+t*as-1)%i+1,tp[j]=make_pair(a,b);
		sort(tp+1,tp+k+1);
		int mn=n+2,mx=n+1;
		for(int j=1;j<=k;j++)mn=cmin(mn,querymn(1,1,n,tp[j].first,tp[j].second));
		tp[k+1]=make_pair(i,i);
		for(int j=0;j<=k;j++)if(tp[j+1].first-1>tp[j].second)mx=cmax(mx,querymx(1,1,n,tp[j].second+1,tp[j+1].first-1));
		mn=finds(mn);mx=finds(mx);
		if(!cmp(mx,mn))merge(mn,mx),insert(finds(mx),i),f1[finds(mx)]=i;
		else insert(finds(mx),i),as++;
		modify(1,1,n,i);
		printf("%d ",as);
	}
}
```

##### T3 draw

Source: Aizu1040

###### Problem

有一个 $n\times m$ 的网格，初始每个格子都为白色，你可以进行如下操作：

1. 选择一行或者一列上连续 $k$ 个格子，将它们全部染成黑色或者白色，费用为 $ak+b$
2. 选择一个格子，将它染成黑色或者白色，费用为 $c$

你需要满足如下限制：

1. 一个格子最多被染色两次。
2. 如果一个格子被染了白色，之后不能再将它染成黑色。
3. 如果一个格子被染了黑色，之后可以将它染成白色，且颜色会直接覆盖。

你需要网格中所有的颜色达到一个状态，求最小总费用。多组数据

$T,n,m,a,b,c\leq 40,a+b\geq c$

$1s,512MB$

###### Sol

显然是先染黑色，再染白色。并且显然一个格子不会被同一个方向同一个颜色染两次。

考虑设 $is_{i,j,0/1,0/1}$ 表示 $(i,j)$ 位置有没有被横向/纵向的黑色/白色染。那么显然如果 $is_{i,j,x,y}$ 为 $1$ 则有 $a$ 的代价。

首先，如果一个格子被横向染色了，但它的上一个格子没有被横向染同一种色，则这个格子是这一次横向染色的起点，需要额外 $b$ 的代价，因此如果 $is_{i,j,0,x}=1,is_{i,j-1,0,x}=0$ 则需要 $b$ 的代价，在纵向上可以得到如果 $is_{i,j,1,x}=1,is_{i-1,j,1,x}=0$ 则需要 $b$ 的代价。

对于格子 $(i,j)$，如果它需要是黑色的，则它不能被染白色，因此如果 $is_{i,j,x,1}=1$ 则有 $+\infty$ 的代价。如果它没有被染横向或者纵向的黑色，则它需要被单点染色，因此如果 $is_{i,j,0,0}=is_{i,j,0,1}=0$ 则额外需要 $c$ 的代价。

如果这个格子需要是白色的，首先它不能被染两次黑色，因此如果 $is_{i,j,0,0}=is_{i,j,0,1}=1$ 则需要 $+\infty$ 的代价，并且如果它没有被染白色则需要单点染色，因此如果 $is_{i,j,1,0}=is_{i,j,1,1}=0$ 则需要 $c$ 的代价。

此时还需要要求如果这个格子被染了黑色，则它必须正好被染白色。直接表示没有好的形式，但可以发现如果这个格子被横向染了黑色，再把它横向染白色，这样的方案一定不如直接把中间段染白色，两侧分开染黑色。因此最优解里面没有这种情况。此时可以表示成同向不同颜色不能同时染，如果染了黑色就必须染另外一个方向的白色。这相当于如果 $is_{i,j,x,0}=is_{i,j,x,1}=1$ 有 $+\infty$ 的代价，如果 $is_{i,j,x,0}=1,is_{i,j,1-x,1}=0$ 有 $c$ 的代价。

此时可以发现，取反 $is_{i,j,0,1},is_{i,j,1,0}$ 后，所有的限制都变成了形如 $is_{A}=1,is_{B}=0$ 时有若干额外贡献的形式，考虑最小割，对于一个点， $is_A=1$ 当且仅当它在割集原点一侧。则上面的限制可以看成连一条 $is_{A}\to is_{B}$ 的边，流量为额外代价。相当于如果割集只包含 $A$ 不包含 $B$ 则需要割这条边。对于 $is_A=1$ 的额外代价可以看成它向汇点连边的流量，$is_A=0$ 的同理。

然后对得到的图最大流即可，跑的非常快。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 42
#define M 6406
int T,n,m,a,b,c,ct,head[M],cnt=1,dis[M],cur[M];
char s[N][N];
struct edge{int t,next,v;}ed[M*20];
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
int dinic(){int ls=1e9,as=0;while(ls&&bfs(ct-1,ct)){int tp=dfs(ct-1,ct,ls);ls-=tp;as+=tp;}return as;}
int getid(int x,int y,int t1,int t2){return (x-1)*m+y+(t1*2+t2)*n*m;}
void solve()
{
	scanf("%d%d%d%d%d",&n,&m,&a,&b,&c);
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	ct=n*m*4+2;
	for(int i=1;i<=ct;i++)head[i]=0;cnt=1;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)
	{
		if(i>1)adde(getid(i,j,0,0),getid(i-1,j,0,0),b),adde(getid(i-1,j,1,0),getid(i,j,1,0),b);
		else adde(getid(i,j,0,0),ct,b),adde(ct-1,getid(i,j,1,0),b);
		if(j>1)adde(getid(i,j,1,1),getid(i,j-1,1,1),b),adde(getid(i,j-1,0,1),getid(i,j,0,1),b);
		else adde(getid(i,j,1,1),ct,b),adde(ct-1,getid(i,j,0,1),b);
		for(int p=0;p<2;p++)for(int q=0;q<2;q++)if((p+q)&1)adde(ct-1,getid(i,j,p,q),a);else adde(getid(i,j,p,q),ct,a);
		if(s[i][j]=='#')adde(getid(i,j,0,1),getid(i,j,0,0),c),adde(ct-1,getid(i,j,1,0),1e9),adde(getid(i,j,1,1),ct,1e9);
		else adde(getid(i,j,0,0),getid(i,j,0,1),1e9),adde(getid(i,j,0,0),getid(i,j,1,0),1e9),adde(getid(i,j,0,0),getid(i,j,1,1),c),
		adde(getid(i,j,1,1),getid(i,j,0,1),1e9),adde(getid(i,j,1,0),getid(i,j,0,1),c);
	}
	printf("%d\n",dinic());
}
int main(){freopen("draw.in","r",stdin);freopen("draw.out","w",stdout);scanf("%d",&T);while(T--)solve();}
```

#### 0713

##### CF1477F Nezzar and Chocolate Bars

###### Problem

有 $n$ 个数 $v_1,...,v_n$。定义一次操作为：

1. 随机选一个数，以 $\frac{v_i}{\sum v_i}$ 的概率选出 $v_i$。
2. 随机选一个 $r\in(0,v_i)$，将 $v_i$ 变为两个数 $r,v_i-r$。

当所有数都小于等于 $k$ 时停止。求停止前的期望操作次数。

$n\leq 50,k,\sum v_i\leq 2000$

$5s,1024MB$

###### Sol

第二步可以看成在长度为 $v_i$ 的线段上随机选一个分界点，因此整个操作可以看成在所有线段上随机放一个分界点，操作停止当且仅当不存在两个分界点（起点终点也算分界点）之间的距离大于 $k$。

考虑期望线性性，设 $pr_i$ 表示 $i$ 次操作前还不满足条件的概率，答案显然为 $\sum_{i\geq 0}pr_i$。

考虑一个数 $v$ 的情况，设 $g_i$ 表示当前数操作了 $i$ 次后还不满足的概率。设此时还有 $d$ 个数大于 $k$，考虑对 $d$ 容斥( $[d\geq 1]=\sum_{i\geq 1}(-1)^{i-1}C_d^i$ )，设 $f_{v,n,d}$ 表示一个数 $v$ 操作 $n$ 次后，选 $d$ 个长度大于 $k$ 的段的方案数的期望。考虑将总长度减去 $d*k$，则变为长度为 $v-d*k$ 且无限制的情况，因此可以得到：
$$
f_{v,n,d}=C_{n+1}^d(\frac{v-dk}{\sum v})^n
$$
考虑把这个看成EGF，设 $F_{v,d}(x)=\sum_{i\geq 0} f_{v,i,d}\frac 1{i!}x^i$，考虑算一个 $g_s$ ，可以枚举每一个 $v_i$ 中容斥的 $d$，记这个值为 $d_i$ ，则可以发现：
$$
g_s=s![x^s]\sum_{d_{1,2,...,n}}(-1)^{\sum d_i}\prod_{i=1}^nF_{v_i,d_i}(x)
$$
考虑一个 $F_{v,d}(x)(d>0)$，可以得到：
$$
F_{v,d}(x)=\sum_{i\geq 0}C_{i+1}^d(\frac{v-dk}{\sum v})^i\frac 1{i!}x^i\\
=\sum_{i\geq 0}(\frac{v-dk}{\sum v})^i*(i+1-d+d)*\frac 1{(i+1-d)!}*\frac 1{d!}x^i\\
=\sum_{i\geq 0}(\frac{v-dk}{\sum v}x)^i*\frac 1{(i-d)!}*\frac 1{d!}+\sum_{i\geq 0}(\frac{v-dk}{\sum v}x)^i*\frac 1{(i+1-d)!}*\frac 1{(d-1)!}\\
=(\frac{v-dk}{\sum v}x)^d*e^{\frac{v-dk}{\sum v}x}*\frac 1{d!}+(\frac{v-dk}{\sum v}x)^{d-1}*e^{\frac{v-dk}{\sum v}x}*\frac 1{(d-1)!}\\
$$
因此可以将 $F_{v,d}(x)$ 表示成若干个 $cx^ae^{bx}$ 的和的形式，乘积相当于每一个 $F$ 中选一项乘起来，对所有情况求和。

可以发现，如果只考虑第一部分，则所有这样的式子乘起来后，可以得到 $x^{\sum d}e^{\frac{v-\sum d*k}{\sum v}x}$，这只和 $\sum d$ 有关。如果选一个第二部分，则 $x$ 前面的指数会减一。

因此考虑设 $dp_{i,a,b}$ 表示考虑了前 $i$ 段，当前每一段选的项的 $d$ 之和为 $a$，当前前面 $x$ 的指数总共减了 $b$ ，前面的权值乘积和。

转移可以直接枚举这一个数的情况，复杂度为 $O(n*(\sum v)^2)$

最后可以得到 $g$ 的EGF为若干个 $x^ae^{bx}$ 求和的形式，只需要考虑每一个 $x^ae^{bx}$ 转OGF后的答案再求和即可。

考虑一个 $x^ae^{bx}$：
$$
\sum g_i\frac{x^i}{i!}=x^ae^{bx}\\
g_{i+a}=\frac{(i+a)!}{i!}b^i\\
ans=\sum_{i\geq 0}\frac{(i+a)!}{i!}b^i\\
=a!\sum_{i\geq 0}C_{i+a}^ib^i
$$
此时可以看成 $i+a$ 个数里面选 $a$ 个，剩下的每个数有 $b$ 的权值，在最开头加一个选中的数，考虑每个选中的数后面有多少个没有被选择的数，则可以发现：
$$
C_{i+a}^ib^i=[x^{i+a+1}](x+bx^2+b^2x^3+...)^{a+1}
$$
设 $H(x)=(x+bx^2+b^2x^3+...)^{a+1}$，则相当于求 $H(1)$，因此这等于：
$$
ans=a!(1+b+b^2+...)^{a+1}=\frac{a!}{(1-b)^{a+1}}
$$
对每一项求和即可。

复杂度 $O(n*(\sum v)^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2005
#define M 52
#define mod 998244353
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int n,k,v[N],su,s1,v1[N][2],dp[M][N],fr[N],ifr[N],as;
int solve(int a,int b){return 1ll*fr[a]*pw(mod+1-b,mod-2-a)%mod;}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su+=v[i];
	fr[0]=ifr[0]=1;for(int i=1;i<=su;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	dp[0][0]=1;
	for(int i=1;i<=n;i++)
	{
		int s2=(v[i]-1)/k;
		for(int j=1;j<=s2;j++)v1[j][0]=1ll*pw(1ll*(v[i]-j*k)*pw(su,mod-2)%mod,j)*ifr[j]%mod,v1[j][1]=1ll*pw(1ll*(v[i]-j*k)*pw(su,mod-2)%mod,j-1)*ifr[j-1]%mod;
		for(int p=i-1;p>=0;p--)
			for(int q=s1;q>=0;q--)
				for(int r=1;r<=s2;r++)
					for(int s=0;s<2;s++)
						dp[p+s][q+r]=(dp[p+s][q+r]+1ll*dp[p][q]*v1[r][s])%mod;
		s1+=s2;
	}
	for(int i=0;i<=n;i++)for(int j=i;j<=s1;j++)as=(as+1ll*dp[i][j]*solve(j-i,1ll*(su-j*k)*pw(su,mod-2)%mod)%mod*(j&1?1:mod-1))%mod;
	printf("%d\n",as);
}
```

#### 0714

##### T1 a

###### Problem

给一个长度为 $n$ 的排列 $p$ 和一个长度为 $4$ 的排列 $q$，求 $p$ 有多少个长度为 $4$ 的子序列 $p'_{1,2,3,4}$ 满足它的大小关系和 $q$ 同构，即：
$$
\forall 1\leq i<j\leq 4,(q_i-q_j)(p_i'-p_j')\geq 0
$$
$n\leq 2000$

$2s,512MB$

###### Sol

考虑枚举对应 $q$ 的两个位置，使得剩下两个位置间的取值独立。

考虑选择位置 $(i,j)$ 需要的条件，显然需要满足如下条件：

1. $\{i,j\}\neq \{1,2\},\{3,4\},\{1,4\}$
2. $\{p_i,p_j\}\neq \{1,2\},\{3,4\},\{1,4\}$

可以发现，如果 $(i,j)$，满足了上面两个条件，枚举对应 $i,j$ 的两个位置，此时可以发现，对于剩下的两个位置，前面的位置限制了这两个位置所在的位置区间和值域区间，且两个位置间显然独立。因此可以二维前缀和计算这种情况的贡献。复杂度 $O(n^2)$。

但可能存在排列找不到 $(i,j)$，可以发现这样的排列正好有两个，为 $(2,4,1,3),(3,1,4,2)$。

考虑 $(2,4,1,3)$，枚举 $2,4$ 位置的取值，如果不考虑位置 $1,3$ 之间的大小顺序，则可以向上面一样求出答案。此时的答案为 $(2,4,1,3)$ 的答案加上 $(1,4,2,3)$ 的答案。可以发现后者的答案可以求出，因此可以得到前者的答案。$(3,1,4,2)$ 同理。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 2005
#define ll long long
int n,p[10],v[N],su[N][N],fg;
ll as;
int calc(int l,int r,int vl,int vr)
{
	return su[r][vr]-su[l-1][vr]-su[r][vl-1]+su[l-1][vl-1];
}
bool check(int a,int b,int c,int d)
{
	if(c>d)swap(c,d);
	if(b-a==3||d-c==3)return 0;
	if(b<3||a>2||d<3||c>2)return 0;
	return 1;
}
void doit(int i,int j)
{
	fg=1;
	for(int a=1;a<=n;a++)
		for(int b=a+1;b<=n;b++)
		{
			if((p[j]-p[i])*(v[b]-v[a])<0)continue;
			int as1=1;
			for(int s=1;s<=4;s++)if(s!=i&&s!=j)
			{
				int l1=1,r1=n,l2=1,r2=n;
				if(s<i)r1=min(r1,a-1);else l1=max(l1,a+1);
				if(s<j)r1=min(r1,b-1);else l1=max(l1,b+1);
				if(p[s]<p[i])r2=min(r2,v[a]-1);else l2=max(l2,v[a]+1);
				if(p[s]<p[j])r2=min(r2,v[b]-1);else l2=max(l2,v[b]+1);
				as1*=calc(l1,r1,l2,r2);
			}
			as+=as1;
		}
}
void solve()
{
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)su[i][j]=0;
	for(int i=1;i<=n;i++)su[i][v[i]]++;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)su[i][j]=su[i-1][j]+su[i][j-1]-su[i-1][j-1]+su[i][j];
	for(int i=1;i<=4;i++)
		for(int j=i+1;j<=4;j++)
			if(!fg&&check(i,j,p[i],p[j])){as=0;doit(i,j);}
}
int main()
{
	freopen("a.in","r",stdin);
	freopen("a.out","w",stdout);
	scanf("%d",&n);
	for(int i=1;i<=4;i++)scanf("%d",&p[i]);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su[i][v[i]]++;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)su[i][j]=su[i-1][j]+su[i][j-1]-su[i-1][j-1]+su[i][j];
	solve();
	if(!fg)
	{
		doit(2,4);
		ll as1=as;as=fg=0;
		swap(p[1],p[3]);solve();
		as=as1-as;
	}
	printf("%lld\n",as);
}
```



##### T2 b

###### Problem

给一个 $n\times m$ 的网格图，每个位置的颜色为白色或者黑色。 $q$ 次操作，每次翻转一个位置的颜色，求出每次操作后图中同色连通块的数量。

强制在线

$n,m\leq 200,q\leq 10^5$

$3s,512MB$

###### Sol

做法1：

这可以看成一个平面图，因此可以考虑从每个点开始，顺时针绕着颜色的边界走直到回到原点，最后计算得到的外边界条数。

可以对于每条边的每个方向，求出如果上一步在这个方向，那么下一步会走到哪个方向。在修改时只需要改 $O(1)$ 个位置。

对于数环数的问题，可以用平衡树维护每个环，可以发现修改相当于分裂/合并环，用Splay即可。

最后的问题是可能存在一个边界被两个方向分别数一次，也有可能只被数一次。此时可以hash判掉重复的情况。

复杂度 $O((nm+q)\log nm)$

做法2：

考虑对横轴线段树，维护当前节点左右 $2m$ 的边界点的连通性以及内部连通块数量，合并暴力即可。

复杂度 $O(nq\log n)$，应该能过。

做法3：

使用做法2类似的思路，每次选择横纵中大小更大的一维分成两部分，对于每一个节点，维护这个点上下左右边界的连通性。合并的复杂度为边界长度。可以发现，查找一个点时每向下两步，边界长度减半。因此一次修改的总边界长度为 $O(n+m)$，复杂度 $O((n+m)q)$。常数非常大但是能过。

###### Code

做法3（有大量复制粘贴）：

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#define N 1020
using namespace std;
int n,m,k,a,b,ls,fa[N*8],id[N*8],is[N*8],ct=1;
char s[N][N];
struct sth{
	int h,w,s1,su;
	vector<int> v0,v1,v2,v3;//0u1d2l3r
}d[2];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct node{int lx,rx,ly,ry,op,ls,rs;}e[N*N*2];
sth mergew(sth &t1,sth &t2,int vx,int vy)
{
	sth t3;
	t3.h=t1.h;t3.w=t1.w+t2.w;
	int su=t1.s1+t2.s1;
	for(int i=1;i<=su;i++)fa[i]=i;
	for(int i=0;i<t1.h;i++)if(s[i+vx][vy]==s[i+vx][vy+1])fa[finds(t1.v3[i])]=finds(t2.v2[i]+t1.s1);
	int s1=0;
	for(int i=1;i<=su;i++)if(finds(i)==i)id[i]=++s1;
	t3.s1=s1;t3.su=t1.su+t2.su-(su-s1);
	for(int i=0;i<t1.h;i++)t3.v2.push_back(id[finds(t1.v2[i])]),t3.v3.push_back(id[finds(t2.v3[i]+t1.s1)]);
	for(int i=0;i<t1.w;i++)t3.v0.push_back(id[finds(t1.v0[i])]),t3.v1.push_back(id[finds(t1.v1[i])]);
	for(int i=0;i<t2.w;i++)t3.v0.push_back(id[finds(t2.v0[i]+t1.s1)]),t3.v1.push_back(id[finds(t2.v1[i]+t1.s1)]);
	for(int i=1;i<=s1;i++)is[i]=0;s1=0;
	for(int i=0;i<t3.v0.size();i++)if(!is[t3.v0[i]])is[t3.v0[i]]=++s1;
	for(int i=0;i<t3.v1.size();i++)if(!is[t3.v1[i]])is[t3.v1[i]]=++s1;
	for(int i=0;i<t3.v2.size();i++)if(!is[t3.v2[i]])is[t3.v2[i]]=++s1;
	for(int i=0;i<t3.v3.size();i++)if(!is[t3.v3[i]])is[t3.v3[i]]=++s1;
	t3.s1=s1;
	for(int i=0;i<t3.v0.size();i++)t3.v0[i]=is[t3.v0[i]];
	for(int i=0;i<t3.v1.size();i++)t3.v1[i]=is[t3.v1[i]];
	for(int i=0;i<t3.v2.size();i++)t3.v2[i]=is[t3.v2[i]];
	for(int i=0;i<t3.v3.size();i++)t3.v3[i]=is[t3.v3[i]];
	return t3;
}
sth mergeh(sth &t1,sth &t2,int vx,int vy)
{
	sth t3;
	t3.w=t1.w;t3.h=t1.h+t2.h;
	int su=t1.s1+t2.s1;
	for(int i=1;i<=su;i++)fa[i]=i;
	for(int i=0;i<t1.w;i++)if(s[vx][i+vy]==s[vx+1][i+vy])fa[finds(t1.v1[i])]=finds(t2.v0[i]+t1.s1);
	int s1=0;
	for(int i=1;i<=su;i++)if(finds(i)==i)id[i]=++s1;
	t3.s1=s1;t3.su=t1.su+t2.su-(su-s1);
	for(int i=0;i<t1.w;i++)t3.v0.push_back(id[finds(t1.v0[i])]),t3.v1.push_back(id[finds(t2.v1[i]+t1.s1)]);
	for(int i=0;i<t1.h;i++)t3.v2.push_back(id[finds(t1.v2[i])]),t3.v3.push_back(id[finds(t1.v3[i])]);
	for(int i=0;i<t2.h;i++)t3.v2.push_back(id[finds(t2.v2[i]+t1.s1)]),t3.v3.push_back(id[finds(t2.v3[i]+t1.s1)]);
	for(int i=1;i<=s1;i++)is[i]=0;s1=0;
	for(int i=0;i<t3.v0.size();i++)if(!is[t3.v0[i]])is[t3.v0[i]]=++s1;
	for(int i=0;i<t3.v1.size();i++)if(!is[t3.v1[i]])is[t3.v1[i]]=++s1;
	for(int i=0;i<t3.v2.size();i++)if(!is[t3.v2[i]])is[t3.v2[i]]=++s1;
	for(int i=0;i<t3.v3.size();i++)if(!is[t3.v3[i]])is[t3.v3[i]]=++s1;
	t3.s1=s1;
	for(int i=0;i<t3.v0.size();i++)t3.v0[i]=is[t3.v0[i]];
	for(int i=0;i<t3.v1.size();i++)t3.v1[i]=is[t3.v1[i]];
	for(int i=0;i<t3.v2.size();i++)t3.v2[i]=is[t3.v2[i]];
	for(int i=0;i<t3.v3.size();i++)t3.v3[i]=is[t3.v3[i]];
	return t3;
}
void mergew2(sth &t3,sth &t1,sth &t2,int vx,int vy)
{
	int su=t1.s1+t2.s1;
	for(int i=1;i<=su;i++)fa[i]=i;
	for(int i=0;i<t1.h;i++)if(s[i+vx][vy]==s[i+vx][vy+1])fa[finds(t1.v3[i])]=finds(t2.v2[i]+t1.s1);
	int s1=0;
	for(int i=1;i<=su;i++)if(finds(i)==i)id[i]=++s1;
	t3.s1=s1;t3.su=t1.su+t2.su-(su-s1);
	for(int i=0;i<t1.h;i++)t3.v2[i]=id[finds(t1.v2[i])],t3.v3[i]=id[finds(t2.v3[i]+t1.s1)];
	for(int i=0;i<t1.w;i++)t3.v0[i]=id[finds(t1.v0[i])],t3.v1[i]=id[finds(t1.v1[i])];
	for(int i=0;i<t2.w;i++)t3.v0[i+t1.w]=id[finds(t2.v0[i]+t1.s1)],t3.v1[i+t1.w]=id[finds(t2.v1[i]+t1.s1)];
	for(int i=1;i<=s1;i++)is[i]=0;s1=0;
	for(int i=0;i<t3.v0.size();i++)if(!is[t3.v0[i]])is[t3.v0[i]]=++s1;
	for(int i=0;i<t3.v1.size();i++)if(!is[t3.v1[i]])is[t3.v1[i]]=++s1;
	for(int i=0;i<t3.v2.size();i++)if(!is[t3.v2[i]])is[t3.v2[i]]=++s1;
	for(int i=0;i<t3.v3.size();i++)if(!is[t3.v3[i]])is[t3.v3[i]]=++s1;
	t3.s1=s1;
	for(int i=0;i<t3.v0.size();i++)t3.v0[i]=is[t3.v0[i]];
	for(int i=0;i<t3.v1.size();i++)t3.v1[i]=is[t3.v1[i]];
	for(int i=0;i<t3.v2.size();i++)t3.v2[i]=is[t3.v2[i]];
	for(int i=0;i<t3.v3.size();i++)t3.v3[i]=is[t3.v3[i]];
}
void mergeh2(sth &t3,sth &t1,sth &t2,int vx,int vy)
{
	int su=t1.s1+t2.s1;
	for(int i=1;i<=su;i++)fa[i]=i;
	for(int i=0;i<t1.w;i++)if(s[vx][i+vy]==s[vx+1][i+vy])fa[finds(t1.v1[i])]=finds(t2.v0[i]+t1.s1);
	int s1=0;
	for(int i=1;i<=su;i++)if(finds(i)==i)id[i]=++s1;
	t3.s1=s1;t3.su=t1.su+t2.su-(su-s1);
	for(int i=0;i<t1.w;i++)t3.v0[i]=id[finds(t1.v0[i])],t3.v1[i]=id[finds(t2.v1[i]+t1.s1)];
	for(int i=0;i<t1.h;i++)t3.v2[i]=id[finds(t1.v2[i])],t3.v3[i]=id[finds(t1.v3[i])];
	for(int i=0;i<t2.h;i++)t3.v2[i+t1.h]=id[finds(t2.v2[i]+t1.s1)],t3.v3[i+t1.h]=id[finds(t2.v3[i]+t1.s1)];
	for(int i=1;i<=s1;i++)is[i]=0;s1=0;
	for(int i=0;i<t3.v0.size();i++)if(!is[t3.v0[i]])is[t3.v0[i]]=++s1;
	for(int i=0;i<t3.v1.size();i++)if(!is[t3.v1[i]])is[t3.v1[i]]=++s1;
	for(int i=0;i<t3.v2.size();i++)if(!is[t3.v2[i]])is[t3.v2[i]]=++s1;
	for(int i=0;i<t3.v3.size();i++)if(!is[t3.v3[i]])is[t3.v3[i]]=++s1;
	t3.s1=s1;
	for(int i=0;i<t3.v0.size();i++)t3.v0[i]=is[t3.v0[i]];
	for(int i=0;i<t3.v1.size();i++)t3.v1[i]=is[t3.v1[i]];
	for(int i=0;i<t3.v2.size();i++)t3.v2[i]=is[t3.v2[i]];
	for(int i=0;i<t3.v3.size();i++)t3.v3[i]=is[t3.v3[i]];
}
sth fu[N*N*2];
void pushup(int x)
{
	if(e[x].op)fu[x]=mergew(fu[e[x].ls],fu[e[x].rs],e[x].lx,(e[x].ly+e[x].ry)>>1);
	else fu[x]=mergeh(fu[e[x].ls],fu[e[x].rs],(e[x].lx+e[x].rx)>>1,e[x].ly);
}
void pushup2(int x)
{
	if(e[x].op)mergew2(fu[x],fu[e[x].ls],fu[e[x].rs],e[x].lx,(e[x].ly+e[x].ry)>>1);
	else mergeh2(fu[x],fu[e[x].ls],fu[e[x].rs],(e[x].lx+e[x].rx)>>1,e[x].ly);
}
void build(int x,int lx,int rx,int ly,int ry)
{
	e[x].lx=lx;e[x].rx=rx;e[x].ly=ly;e[x].ry=ry;
	if(lx==rx&&ly==ry){fu[x]=d[s[lx][ly]-'0'];return;}
	e[x].ls=++ct;e[x].rs=++ct;
	if(rx-lx>ry-ly)
	{
		int mid=(lx+rx)>>1;
		build(e[x].ls,lx,mid,ly,ry);build(e[x].rs,mid+1,rx,ly,ry);
	}
	else
	{
		e[x].op=1;int mid=(ly+ry)>>1;
		build(e[x].ls,lx,rx,ly,mid);build(e[x].rs,lx,rx,mid+1,ry);
	}
	pushup(x);
}
void modify(int x,int lx,int ly)
{
	if(e[x].lx==e[x].rx&&e[x].ly==e[x].ry)return;
	int fg=0;
	if(e[x].op)fg=((e[x].ly+e[x].ry)>>1)<ly;
	else fg=((e[x].lx+e[x].rx)>>1)<lx;
	if(fg)modify(e[x].rs,lx,ly);
	else modify(e[x].ls,lx,ly);
	pushup2(x);
}
int main()
{
	freopen("b.in","r",stdin);
	freopen("b.out","w",stdout);
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	ct=1;
	for(int v=0;v<2;v++)
	{
		d[v].h=d[v].w=d[v].s1=d[v].su=1;
		d[v].v0.push_back(1);d[v].v1.push_back(1);d[v].v2.push_back(1);d[v].v3.push_back(1);
	}
	build(1,1,n,1,m);
	ls=fu[1].su;
	while(k--)
	{
		scanf("%d%d",&a,&b);
		a^=ls;b^=ls;
		s[a][b]^=1;
		modify(1,a,b);
		printf("%d\n",ls=fu[1].su);
	}
}
```

##### T3 c

###### Problem

给出二维平面上 $n$ 条直线 $l_{1,2,...,n}$，保证直线间不存在平行关系。

定义点 $s_{i,j}$ 为 $l_i,l_j$ 的交点。给出 $m$ 对 $(u_i,v_i)$，考虑除去给出的 $s_{u_i,v_i}$ 外的所有 $s_{i,j}$，求这 $\frac{n(n-1)}2-m$ 个点构成的凸包周长。

$n\leq 10^5,m\leq 50$

$1s,512MB$

###### Sol

对于一条直线，考虑所有和它相关的交点。可以发现在求凸包时，只有交点中最外侧的两个交点有用。

考虑 $m=0$ 且没有一条直线垂直于 $x$ 轴，此时相当于对于每条直线，求出它的交点中 $x$ 最小最大的。

将直线表示为 $y=k_ix+b_i$，则相当于求一个 $j\neq i$ 使得 $\frac{b_i-b_j}{k_j-k_i}$ 最小/最大。

将一条直线看成一个点 $(k_i,b_i)$，则上面的东西相当于找一个点到这个点的线段斜率最大/最小。

考虑最大且 $j<i$ 的情况，相当于在它左侧找一个这样的点。此时可以发现一定这样的点是左侧的下凸壳上的点。因此将点按照横坐标排序，从左向右维护当前前面的下凸壳，在下凸壳上二分找即可。然后对左右/上下四个方向各做一遍即可。

复杂度 $O(n\log n)$

然后考虑 $m\neq 0$ 的情况。可以发现只有 $O(m)$ 条直线会受到影响，对于剩下的直线按照上面的方式做，影响的直线暴力即可。

此时可以得到 $O(n)$ 个点，暴力求凸包即可。

本题坐标范围可以达到 $10^8$，极角排序做法大概率被卡，建议使用上下凸壳写法。

复杂度 $O(nm+n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<set>
#include<cmath>
using namespace std;
#define N 100500
int n,m,a,b,c,is[N];
set<int> fuc[N];
bool check(int x,int y){return fuc[x].find(y)==fuc[x].end();}
struct pt{double x,y;}s[N*8],f1,s2[N*8];
pt operator +(pt a,pt b){return (pt){a.x+b.x,a.y+b.y};}
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
pt operator *(pt a,double b){return (pt){a.x*b,a.y*b};}
double Abs(double x){return x>0?x:-x;}
double cross(pt a,pt b){return a.x*b.y-a.y*b.x;}
bool cmp1(pt a,pt b){return Abs(a.x-b.x)<=1e-10?a.y<b.y:a.x<b.x;}
double getdis(pt a){return sqrt(a.x*a.x+a.y*a.y);}
bool cmp(pt a,pt b){double as=cross(a-f1,b-f1);if(Abs(as)<=1e-6)return cmp1(b,a);else return as>0;}
//fuck
//void calchull(vector<pt> p1)
//{
//	sort(p1.begin(),p1.end(),cmp1);
//	vector<pt> p2;
//	p2.push_back(p1[0]);
//	for(int i=1;i<p1.size();i++)
//		if(Abs(p1[i].x-p1[i-1].x)+Abs(p1[i].y-p1[i-1].y)>1e-9)p2.push_back(p1[i]);
//	p1=p2;f1=p1[0];
//	int ct=p1.size();
//	for(int i=1;i<ct;i++)s2[i]=p1[i];
//	sort(s2+1,s2+ct,cmp);
//	int c1=1;s[1]=p1[0];
//	for(int i=1;i<ct;i++)if(i==1||Abs(cross(s2[i]-f1,s2[i-1]-f1))>1e-6)
//	{
//		pt nw=s2[i];
//		while(c1>1&&cross(nw-s[c1],nw-s[c1-1])>=-1e-9)c1--;
//		s[++c1]=nw;
//	}
//	while(c1>2&&cross(s[1]-s[c1],s[1]-s[c1-1])>=-1e-9)c1--;
//	double as=0;
//	for(int i=1;i<=c1;i++)as+=getdis(s[i]-s[i%c1+1]);
//	printf("%.10lf\n",as);
//}
pt stk[N * 10]; int top;
const double eps = 1e-8;
double slope(pt a, pt b) {
	return (b.y - a.y) / (b.x - a.x);
}

vector<pt> getDown(vector<pt> p) {
	sort(p.begin(), p.end(), [&](pt a, pt b) {
		if (fabs(a.x - b.x) > eps) return a.x < b.x;
		return a.y < b.y;
	});
	top = 0;
	for (int i = 0; i < p.size(); i++) {
		if (top && fabs(stk[top].x - p[i].x) < eps) continue;
		while (top >= 2 && slope(stk[top - 1], p[i]) < slope(stk[top - 1], stk[top])) top--;
		stk[++top] = p[i];
	}
	vector<pt> res;
	for (int i = 1; i <= top; i++) res.push_back(stk[i]);
	return res;
}

vector<pt> getUp(vector<pt> p) {
	sort(p.begin(), p.end(), [&](pt a, pt b) {
		if (fabs(a.x - b.x) > eps) return a.x > b.x;
		return a.y > b.y;
	});
	top = 0;
	for (int i = 0; i < p.size(); i++) {
		if (top && fabs(stk[top].x - p[i].x) < eps) continue;
		while (top >= 2 && slope(stk[top - 1], p[i]) < slope(stk[top - 1], stk[top])) top--;
		stk[++top] = p[i];
	}
	vector<pt> res;
	for (int i = 1; i <= top; i++) res.push_back(stk[i]);
	return res;
}
void calchull(vector<pt> lsjak)
{
	vector<pt> h1 = getDown(lsjak), h2 = getUp(lsjak);
	for (int i = 0; i < h2.size(); i++) h1.push_back(h2[i]);
	h1.push_back(h1[0]);
	double res = 0;
	for (int i = 0; i + 1 < h1.size(); i++)
		res += getdis(h1[i]- h1[i + 1]);
	printf("%.10lf\n", res);
}
struct lns{double a,b,c;}l1[N];
pt inse(lns a,lns b)
{
	double asx=0,asy=0;
	if(Abs(a.a)<1e-7)swap(a,b);
	if(Abs(b.b)>1e-7)
	{
		double tp=-a.b/b.b;
		a.a+=tp*b.a;a.b+=tp*b.b;a.c+=tp*b.c;
		asx=a.c/a.a;asy=(b.c-b.a*asx)/b.b;
	}
	else
	{
		asx=b.c/b.a;asy=(a.c-a.a*asx)/a.b;
	}
	return (pt){asx,asy};
}
vector<pt> fu;
pt s3[N];
int c3,id[N],st[N],ct,as3[N][2];
bool cmp2(int a,int b){return cmp1(s3[a],s3[b]);}
vector<int> f3[N];
void add1(int x,int y){f3[x].push_back(y);f3[y].push_back(x);}
void solve3(int fg)
{
	if(fg==0)fg=-1;
	ct=0;
	for(int i=1;i<=c3;i++)
	{
		if(ct)
		{
			int lb=2,rb=ct,as=1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(fg*cross(s3[id[i]]-s3[st[mid]],s3[st[mid-1]]-s3[st[mid]])>=0)rb=mid-1;
				else as=mid,lb=mid+1;
			}
			add1(id[i],st[as]);
		}
		while(ct>1&&fg*cross(s3[st[ct-1]]-s3[id[i]],s3[st[ct]]-s3[id[i]])>=0)ct--;
		st[++ct]=id[i];
	}
	ct=0;
	for(int i=c3;i>=1;i--)
	{
		if(ct)
		{
			int lb=2,rb=ct,as=1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(fg*cross(s3[id[i]]-s3[st[mid]],s3[st[mid-1]]-s3[st[mid]])>=0)rb=mid-1;
				else as=mid,lb=mid+1;
			}
			add1(id[i],st[as]);
		}
		while(ct>1&&fg*cross(s3[st[ct-1]]-s3[id[i]],s3[st[ct]]-s3[id[i]])>=0)ct--;
		st[++ct]=id[i];
	}
	ct=0;
}
int main()
{
	freopen("c.in","r",stdin);
	freopen("c.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d%d%d",&a,&b,&c),l1[i]=(lns){a,b,c};
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),fuc[a].insert(b),fuc[b].insert(a),is[a]=is[b]=1;
	for(int i=1;i<=n;i++)if(is[i]||Abs(l1[i].b)<=1e-7||Abs(l1[i].a)<=1e-7)
	for(int f=0;f<2;f++)
	{
		int is=0,fr=0;pt as;
		for(int j=1;j<=n;j++)if(j!=i&&check(i,j))
		{
			pt s1=inse(l1[i],l1[j]);
			if(!is)is=1,as=s1,fr=j;
			else if(cmp(s1,as)^f)as=s1,fr=j;
		}
		if(is)add1(fr,i);
	}
	else s3[i]=(pt){-l1[i].a/l1[i].b,l1[i].c/l1[i].b},id[++c3]=i;
	sort(id+1,id+c3+1,cmp2);
	for(int t=0;t<2;t++)solve3(t);
	for(int i=1;i<=n;i++)
		for(int f=0;f<2;f++)
		{
			int is=0,fr=0;pt as;
			for(int t=0;t<f3[i].size();t++)
			{
				int j=f3[i][t];
				pt s1=inse(l1[i],l1[j]);
				if(!is)is=1,as=s1,fr=j;
				else if(cmp(s1,as)^f)as=s1,fr=j;
			}
			as3[i][f]=fr;
		}
	for(int i=1;i<=n;i++)for(int f=0;f<2;f++)if(as3[i][f])
	{
		int nt=as3[i][f];
		if(as3[nt][0]==i||as3[nt][1]==i)fu.push_back(inse(l1[i],l1[nt]));
	}
	calchull(fu);
}
```

#### 0715

##### T1 binary

###### Problem

给出 $n$ 以及一个长度为 $n$ 的正整数序列 $b$，考虑如下代码：

```cpp
int cnt=0;
void solve()
{
	int l=1,r=n+1;
    while(l<r)
    {
        int mid=getmid(l,r);cnt+=b[mid];
        if(check(mid))r=mid;
        else l=mid+1；
    }
}
```

其中 $getmid(l,r)$ 可以返回 $[l,r-1]$ 中的整数，$check(x)$ 返回 $0$ 或 $1$。

你需要找到一种选择 $getmid$ 的策略，使得无论 $check$ 怎么返回，最坏情况下结束时 $cnt$ 的最大值最小。求出最小值。

$n\leq 10^6,b_i\leq 9$

$1.5s,512MB$

###### Sol

显然的暴力是设 $dp_{l,r}$ 表示当前的二分区间为 $[l,r+1]$ 时，这个区间内的最小代价。转移可以枚举区间中 $getmid$ 的结果。

如果 $getmid$ 使用正常的二分方式，则显然最坏的 $cnt$ 不会超过 $b*\log n$ 。因此考虑设 $vr_{l,x}$ 表示最大的 $r$ 满足 $dp_{l,r}\leq x$，$vl_{r,x}$ 表示最小的 $l$ 满足 $dp_{l,r}\leq x$。

考虑对于一个 $x$ 求出所有的 $vl_{i,x},vr_{i,x}$。枚举选择的分解点 $t$，则可以发现当前区间的左端点不能小于 $vr_{t-1,x-b_t}$，右端点不能大于 $vl_{t+1,x-b_t}$，且满足这个条件一定合法。

这样可以得到 $n$ 个区间，使得对于任意 $l,r$，$dp_{l,r}\leq x$ 当且仅当 $l,r$ 被某一个区间同时包含。可以通过前缀后缀和求出 $vl,vr$。当 $vr_{1,x}=n$ 时停止即可。

复杂度 $O(b_i*n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstdlib>
using namespace std;
#define N 1000500
int n,sl[10][N],sr[10][N],ti,nw;
char s[N];
int main()
{
	freopen("binary.in","r",stdin);
	freopen("binary.out","w",stdout);
	scanf("%d%s",&n,s+1);
	for(int i=0;i<10;i++)for(int j=0;j<=n+1;j++)sl[i][j]=j+1,sr[i][j]=j-1;
	while(1)
	{
		ti++;nw=(nw+1)%10;
		for(int i=1;i<=n;i++)if(s[i]-'0'<=ti)
		{
			int nt=(ti-s[i]+'0')%10;
			int lb=sl[nt][i-1],rb=sr[nt][i+1];
			sr[nw][lb]=max(sr[nw][lb],rb);sl[nw][rb]=min(sl[nw][rb],lb);
		}
		for(int i=1;i<=n;i++)if(sr[nw][i]<sr[nw][i-1])sr[nw][i]=sr[nw][i-1];
		for(int i=n;i>=1;i--)if(sl[nw][i]>sl[nw][i+1])sl[nw][i]=sl[nw][i+1];
		if(sr[nw][1]>=n)
		{
			printf("%d\n",ti);
			return 0;
		}
	}
}
```

##### T2 network

###### Problem

给一个 $n$ 个点 $m$ 条边的连通弦图，你需要选出若干个点，满足选出的点的导出子图不存在环且选择的点数尽量多。

输出一组最优解。

$n,m\leq 10^5$

$3s,512MB$

###### Sol

弦图的导出子图一定是弦图，因此无环等价于不存在三元环。

考虑求出弦图的完美消除序列 $s_{1,2,...,n}$，这个序列满足对于任意的 $i$，只考虑 $s_{i,i+1,...,n}$ 的子图时，与 $s_i$ 相邻的点构成一个团，即如果存在边 $(s_i,s_j),(s_i,s_k)(i<j<k)$，则存在边 $s_j,s_k$。

根据弦图的性质，只需要每个团中不存在三个点同时被选，这个方案就是合法的。即对于每一个 $i$，点集 $\{s_i\}\cup\{s_j|j>i,(s_i,s_j)有边\}$ 中最多只能选两个点，记这个集合为 $T_{i}$。

考虑对于点 $s_i$ ，找到最小的 $j$ 满足 $j>i$ 且 $(s_i,s_j)$ 有边。记这个 $j$ 为 $fa_i$。则有如下性质：

1. 在按照完美消除序列删点的过程中，不存在使得原本连通的图变的不连通的情况。

证明：如果删去一个点使得两个部分变的不连通，则这个点向两个部分都有连边，但这两个部分之间没有连边，矛盾。

因此可以发现 $fa$ 一定构成一棵树的形式。

2. 对于任意 $s_i$，$T_i\subset T_{s_{fa_i}}\cup\{s_i\}$

证明：考虑按照完美消除序列删点，删到 $s_i$ 时， $T_i$ 即为所有与 $s_i$ 相邻的点加上 $s_i$ 自己构成的集合，这些点构成一个团。

根据定义，$s_{fa_i}$ 一定在这个团中，且是这个团中下一个被删去的点。在删去 $s_{fa_i}$ 时，剩余的与 $s_{fa_i}$ 相邻的点一定包含这个团中除去 $s_i$ 外的所有点。因此结论成立。

3. 对于任意 $s_i$，$T_i$ 中的点一定是 $s_i$ 在树上的祖先。

证明：考虑归纳，对于 $s_i$，$T_i$ 中除去 $s_i$ 外的点都属于 $T_{fa_i}$，根据归纳这些点都是 $s_{fa_i}$ 的祖先，因此它们一定是 $s_i$ 的祖先。

此时考虑一个点 $s_i$ 的子树，$T_{s_i}$ 中的点只能选最多两个，且根据性质 $2$，这个子树内的 $T_x$ 不会包含子树外除去 $T_{s_i}$ 外的点，因此在子树外已经确定时，子树内的最优解只和 $T_{s_i}$ 中的选择情况有关。

考虑设 $dp_{u,S}$ 表示考虑点 $s_u$ 的子树，在 $T_{s_u}$ 中已经选择的点集合为 $S$ 时子树内的最优解，其中 $|S|\leq 2$。可以发现这个状态数等于图中的三元环数，为 $O(m\sqrt m)$

考虑如何得到一个 $dp_{u,S}$，考虑分别处理每个子树，因为每个儿子的 $T_v$ 最多比 $T_u$ 多一个点，可以得到：
$$
dp_{u,S}=[s_u\in S]+\sum_{fa_v=u}\max(dp_{v,S\cap T_v},dp_{v,(S\cap T_v)\cup{s_v}})
$$
先考虑只处理 $max$ 中的第一部分的情况，相当于 $dp_{u,S}=\sum_{fa_v=u}dp_{v,S\cap T_v}$ 。

考虑优化转移，因为 $|S|\leq 2$ ，一个子树的贡献可以看成如下形式：

1. 对于所有 $dp_{u,S}$ ，加上 $dp_{v,\emptyset}$。
2. 对于所有满足 $x\in S$ 的 $dp_{u,S}$，加上 $dp_{v,\{x\}}-dp_{v,\emptyset}$。
3. 对于所有满足 $x,y\in S$ 的 $dp_{u,S}$，加上 $dp_{v,\{x,y\}}-dp_{v,\{x\}}-dp_{v,\{y\}}+dp_{x,\emptyset}$。

这样一定满足要求，且可以在 $O(\sum |T_v|^2)$ 的时间内求出整体贡献。

对于第二部分，考虑先求出上面的值，然后对于所有满足 $s_u\in S$ 的 $dp_{u,S}$，用 $dp_{u,S}+1$ 转移 $dp_{u,S-\{s_u\}}$ 即可。

可以发现第一部分转移固定，只需要记录第二部分是否转移，即可在最后dfs求出方案。

复杂度 $O(m^{1.5})$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 100500
int n,m,a,b,head[N],cnt,st[N],vl[N],mx,fa[N],id[N],ct[N],sid[N],su[N],dp3[N],is3[N];
vector<int> s1[N],sn[N],f1[N],dp2[N],is[N],as;
vector<vector<int> > dp[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){f,head[t]};head[t]=cnt;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void mcs()
{
	s1[0].push_back(1);
	for(int i=1;i<=n;i++)
	{
		int nw=0;
		while(!nw)
		if(!s1[mx].size())mx--;
		else
		{
			int tp=s1[mx].back();s1[mx].pop_back();
			if(vl[tp]>=0)nw=tp;
		}
		id[nw]=i;st[i]=nw;vl[nw]=-1;
		for(int j=head[nw];j;j=ed[j].next)if(vl[ed[j].t]>=0)vl[ed[j].t]++,s1[vl[ed[j].t]].push_back(ed[j].t),mx=max(mx,vl[ed[j].t]);
	}
}
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),dp3[u]+=dp3[ed[i].t];
	for(int i=0;i<ct[u];i++)sid[f1[u][i]]=i,su[i]=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	for(int j=1;j<ct[ed[i].t];j++)
	{
		int s1=sid[f1[ed[i].t][j]];
		su[s1]+=dp2[ed[i].t][j]-dp3[ed[i].t];
	}
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	for(int j=1;j<ct[ed[i].t];j++)
	for(int k=1;k<ct[ed[i].t];k++)if(j!=k)
	{
		int s1=sid[f1[ed[i].t][j]],s2=sid[f1[ed[i].t][k]];
		dp[u][s1][s2]+=dp[ed[i].t][j][k]-dp2[ed[i].t][j]-dp2[ed[i].t][k]+dp3[ed[i].t];
	}
	for(int i=0;i<ct[u];i++)for(int j=0;j<ct[u];j++)if(i!=j)dp[u][i][j]+=su[i]+su[j]+dp3[u];
	for(int i=0;i<ct[u];i++)dp2[u][i]=su[i]+dp3[u];
	for(int i=1;i<ct[u];i++)if(dp[u][i][0]+1>dp2[u][i])dp2[u][i]=dp[u][i][0]+1,is[u][i]=1;
	if(dp2[u][0]+1>dp3[u])dp3[u]=dp2[u][0]+1,is3[u]=1;
}
void dfs2(int u,int fa,int v1,int v2)
{
	int s1=-1,s2=-1;
	for(int i=1;i<ct[u];i++)if(f1[u][i]==v1)s1=i;else if(f1[u][i]==v2)s2=i;
	if(s1<s2)swap(s1,s2);
	if(s1==-1&&is3[u])s1=0,as.push_back(u);
	else if(s1>=0&&s2==-1&&is[u][s1])s2=0,as.push_back(u);
	if(s1==-1)s1=0;else s1=f1[u][s1];
	if(s2==-1)s2=0;else s2=f1[u][s2];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs2(ed[i].t,u,s1,s2);
}
int main()
{
	freopen("network.in","r",stdin);
	freopen("network.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b);
	mcs();
	for(int i=1;i<=n;i++)
	{
		int mx=0;
		for(int j=head[i];j;j=ed[j].next)if(id[ed[j].t]<id[i]&&id[ed[j].t]>id[mx])mx=ed[j].t;
		if(i>1)sn[mx].push_back(i);
		f1[i].push_back(i);ct[i]++;
		for(int j=head[i];j;j=ed[j].next)if(id[ed[j].t]<id[i])ct[i]++,f1[i].push_back(ed[j].t);
		vector<int> st;
		for(int j=0;j<ct[i];j++)st.push_back(0);
		dp2[i]=st;is[i]=st;for(int j=0;j<ct[i];j++)dp[i].push_back(st);
	}
	for(int i=1;i<=n;i++)head[i]=0;cnt=0;
	for(int i=1;i<=n;i++)for(int j=0;j<sn[i].size();j++)adde(i,sn[i][j]);
	dfs(1,0);printf("%d\n",dp3[1]);dfs2(1,0,0,0);
	for(int i=0;i<as.size();i++)printf("%d ",as[i]);
}
```

##### T3 road

###### Problem

提交答案

给一棵 $n$ 个点的树，加入 $k$ 条边，使得树上所有点对间最短路的和尽量小。

$n=1000,k=100/300$

###### Sol

首先考虑连的边的形式，可以发现让所有边连向一个点是较为优秀的。此时随机一个点作为中心，随机连一些边可以获得 $50+$ 分。

显然此时连边均匀会更优，因此可以类似树分块的方式选点，按照大小 $\frac nk$ 分块并加入随机调整，可以获得 $75+$ 分。

此时考虑中心和连出的 $k$ 个点，这部分一共有 $k+1$ 个点，称这些点为关键点。此时一般情况下两个点的最短路形如走到最近的一个关键点，然后通过加入的边走到另外一侧最接近的关键点。

此时可以发现最短路和近似于 $n^2$ 加上 $n$ 乘上每个点到离他最近的关键点距离和，考虑最小化后面的部分，一种显然的想法是随机化爬山，每次选择一个点进行移动找更优解，可以 $O(n+k)$ 计算。然后可以枚举一个点作为中心。可以获得 $\sim97$ 分。

显然也可以对这个 $dp$，设 $dp_{u,k,l,0/1}$ 表示 $u$ 的子树内选择了 $k$ 个点，当前钦定 $u$ 到最近关键点的距离为 $l$，当前这个距离是否被满足时，子树内的最小距离和。转移可以大力做背包，复杂度 $O(nk*(\frac nk)^2)=O(\frac{n^3}k)$。

然后可以使用bitset记录转移，复杂度 $O(\frac{n^4}{kw})$，可以在 $5s$ 之内得到结果，加上一点随机化可以得到 $97\sim 100$ 分。

最后的一点问题在于直接向中心走可以少一条边，可以在 $dp$ 时再记录一维状态，这样应该能较为轻松的得到 $100$ 分。~~然而我只写了上一个做法~~

###### Code

上面倒数第二个做法的dp:

```cpp
#include<cstdio>
#include<bitset>
#include<algorithm>
#include<iostream>
using namespace std;
#define N 1050
int n,k,v,a,b,li,dp[N][11][305][2],is[N],ct,head[N],cnt,dp2[305][2],dp3[305][2];
bitset<N> tr[N][11][305][2],t2[305][2],t3[305][2];
struct edge{
	int t,next;
}ed[N*2];
void adde(int f,int t){
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
}
void dfs0(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);
	for(int l=0;l<=li;l++)for(int j=0;j<=k;j++)for(int t=0;t<2;t++)dp[u][l][j][t]=1e9;
	for(int l=0;l<=li;l++)
	{
		for(int j=0;j<=k;j++)for(int t=0;t<2;t++)t2[j][t].reset(),dp2[j][t]=1e9;
		dp2[0][0]=0;
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
		{
			for(int j=0;j<=k;j++)for(int t=0;t<2;t++)dp3[j][t]=1e9;
			for(int j=0;j<=k;j++)
				for(int t=0;t<2;t++)
				{
					int fr1=-1,fr2=-1;
					for(int p=0;p<=li;p++)
						for(int q=0;q<2;q++)
						{
							if(t==1&&(!q||p+1>l))continue;
							if(!q&&p-1<l)continue;
							if(fr1==-1||dp[ed[i].t][p][j][q]<dp[ed[i].t][fr1][j][fr2])fr1=p,fr2=q;
						}
		//			printf("%d %d %d %d\n",j,t,fr1,fr2);
					if(fr1==-1)continue;
					for(int p=0;p+j<=k;p++)
						for(int s=0;s<2;s++)
							if(dp3[p+j][s|t]>dp2[p][s]+dp[ed[i].t][fr1][j][fr2])
							{
								dp3[p+j][s|t]=dp2[p][s]+dp[ed[i].t][fr1][j][fr2];
								t3[p+j][s|t]=t2[p][s]|tr[ed[i].t][fr1][j][fr2];
							}
				}
			for(int j=0;j<=k;j++)for(int t=0;t<2;t++)dp2[j][t]=dp3[j][t],t2[j][t]=t3[j][t];
		}
		for(int j=0;j<=k;j++)for(int t=0;t<2;t++)if(dp2[j][t]<1e8)
		{
			int ntj=j,ntt=t;
			if(l==0)ntj++,ntt=1,t2[j][t].set(u,1);
			dp[u][l][ntj][ntt]=dp2[j][t],tr[u][l][ntj][ntt]=t2[j][t];
		}
	}
	for(int l=0;l<=li;l++)
		for(int i=0;i<=k;i++)
			for(int t=0;t<2;t++)dp[u][l][i][t]+=l;//printf("%d %d %d %d %d\n",u,l,i,t,dp[u][l][i][t]);
}
int main()
{
	scanf("%d%d%d",&n,&k,&v);k++;
	li=n/k;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs0(1,0);
	int fr1=0,fr2=1;
	for(int i=0;i<=li;i++)for(int j=1;j<2;j++)if(dp[1][i][k][j]<dp[1][fr1][k][fr2])fr1=i,fr2=j;
//	printf("%d %d\n",fr1,fr2);
	cerr<<dp[1][fr1][k][fr2]<<endl;
	int ls=0;
	for(int i=1;i<=n;i++)if(tr[1][fr1][k][fr2][i])printf("%d ",i);
}
```



#### 0717-0718

##### TopCoder 13459 RookGraph

考虑与一个点有边的点，其中一定是一些与它在一行，一些在一列，因此这些点构成两个团，如果不满足这个条件显然无解。

考虑从一个点开始，钦定这两个团中的一个是行方向的，然后可以通过类似dfs的方式得到这个点所在连通块内所有限制是行上的还是列上的。

此时可以判断是否有解，且如果有解可以发现这个连通块占用的行数和列数固定，内部放的方案数显然也固定。但行列可以翻转。

显然不同连通块不会使用同一行，因此问题可以看成每个连通块可以占用 $a_i$ 行 $b_i$ 列或者 $b_i$ 行 $a_i$ 列，求放下所有连通块的方案数。直接 $dp$ 即可。

复杂度 $O(n^3)$，注意判断是否有解。

##### TopCoder 12909 Seatfriends

考虑设 $dp_{i,j}$ 表示前 $i$ 个人占据了 $j$ 个连续的段的方案数，转移可以看成：

1. 加入一个段。
2. 放在一个段的端点一侧。
3. 合并两个相邻的段。

直接 $dp$ 即可。

复杂度 $O(n^2)$

##### TopCoder 13692 TwoEntrances

考虑倒着做，每次变成拿出一个物品。

考虑 $(u,v)$ 链上拿出物品的顺序。当拿出了链上一个点 $x$ 时， $x$ 的非链上子树都可以开始拿出。可以 $dp$ 出非链子树内部拿出来的方案数 $f$。

然后考虑子树内部和其余部分，可以发现这两部分之间独立，因此方案数只需要乘上从剩下的所有点中选出非链子树内点数个点的组合数即可。

因此可以设 $dp_{l,r}$ 表示当前还没有被拿出的链上部分为 $[l,r]$ 时后面的的方案数，转移形如 $dp_{l-1,r}+=dp_{l,r}*f_l*C_{(\sum_{i=l}^rsz_i)-1}^{sz_l-1}$

复杂度 $O(n^2)$

可以发现问题实际上相当于求基环树的拓扑序数量，可以发现一定有一条边没用，枚举断哪条边再做树上拓扑序即可。

##### AGC017F Zigzag

可以把每条折线看成一个长度为 $n-1$ 的二进制串。考虑 $dp_{i,j}$ 表示放了前 $i$ 条折线，当前最后一条折线为 $j$ 的方案数。

考虑转移，可以先不考虑折线的限制转移出 $dp_{i+1}$，再去掉这条折线不合法的状态。

转移相当于：
$$
dp_{i+1,T}=\sum_{折线S在折线T左侧}dp_{i,S}
$$
考虑如何处理限制。如果要求 $S$ 和 $T$ 中 $1$ 的个数相同，则可以看成如下形式：

![](zigzag.png)

此时可以发现，一定存在一个方案，每次在 $S$ 的路径中向上翻折一个方格（$01\to 10$），使得最后 $S=T$。

考虑如何不重复处理转移，显然的想法是按照一个顺序翻折。记录 $f_{i,j}$ 表示当前状态为 $i$，当前正在翻折第 $j$ 位。转移时可以做如下操作：

1. 跳过这一位，转移到 $f_{i,j+1}$。
2. 如果这一位是 $1$ 且上一位是 $0$，则变成这一位是 $0$ 上一位是 $1$，转移到 $f_{i',j-1}$。

可以发现这样转移唯一，复杂度 $O(n2^n)$。

考虑终点不同的情况。可以发现一定可以先把最后若干位变成 $1$，再做上面的转移。因此对于每个数，枚举将最后若干个 $0$ 变成 $1$ 转移，再做上面转移即可。

复杂度 $O(n^22^n)$

##### ARC078D Mole and Abandoned Mine

考虑最后的路径，其余的点一定构成若干个连通块，且每个连通块与路径上最多一个点有边。

因此可以设 $dp_{S,x}$ 表示当前考虑了点集 $S$，当前路径终点为 $T$ ，最多保留的边权和。

转移可以分为：

1. 向路径结尾加入一个点。
2. 向当前终点加入一个连通块，枚举即可。

复杂度 $O(n3^n)$

##### ARC068D Solitaire

考虑枚举在 $1$ 之后的元素集合，在删除 $1$ 后，这部分会成为一个递增序列，因此这部分有 $2^{\max(n-k-1,0)}$ 种方案。

考虑前面的情况，设之后的元素最大为 $x$，则前面可以分成从 $1$ 方向删除的和从 $x$ 方向删除的，两部分都递减，因此左侧序列一定可以分成两个递减序列，满足第二个序列的最小值大于 $x$。

只考虑前面的元素，设 $dp_{n,k}$ 表示前 $n$ 个元素的排列中，有多少个排列能被分成两个递减序列，第一个序列的结尾为最小元素，且第二个序列的结尾最大能为 $k$ 的方案数。

可以发现转移是一个类似前缀和的形式，可以做到 $O(n^2)$。

然后枚举 $dp_{n-k-1,i}$，此时这部分前 $i-1$ 个元素和之后的 $n-k$ 个元素大小关系可以任意组合，前面顺序不能改变，因此方案数即为 $C_{n-k+i-1}^{i-1}$

复杂度 $O(n^2)$

##### AGC004F Namori

不会

##### ARC097D Monochrome Cat

显然黑色的叶节点可以直接删去，此时剩余的每个点都必须经过至少一次。

考虑起点为 $u$，终点为 $v$ 的路径。显然重复走点不如只走一次并在路途中翻转，因此如果 $x$ 不在 $(u,v)$ 路径上或者 $x=v$，则它会在路径中被翻转 $d_x$ 次，否则它会被翻转 $d_x-1$ 次。

此时可以看成，每个点在路径上有一个代价，不在路径上有另外一个代价，选一条路径使得代价最小，这相当于找一条直径。可以发现每个点在路径上会让代价减少 $0$ 或 $2$。

复杂度 $O(n)$

##### TopCoder 10265 IncreasingLists

考虑按照长度分段，只需要求出 $is_{l,r,k}$ 表示 $[l,r]$ 是否能分成每个数长度都是 $k$ 的递增序列即可。

此时最高位一定不降，考虑按照最高位分段，得到的每一段一定每个数前若干位固定，后面需要段内递增。因此可以设 $dp_{l,r,k,s}$ 表示 $[l,r]$ 这一段是否能分成每个数长度都是 $k$，前 $s$ 位固定的递增序列，做一次 $dp$ 即可求出。

复杂度 $O(能过)$

##### TopCoder 9844 TreeCount

考虑枚举 $k$，再设 $dp_{u,0/1}$ 表示 $u$ 子树内选一个 $k-$ 独立集，此时根的度数为 $k$ 或者小于 $k$ 的方案数。

转移时用背包转移即可。

复杂度 $O(n^3)$

##### ARC067C Grouping

可以先看成不同组不同，然后除以每种大小数量的阶乘即可。因为转移是按照集合转移因此系数可以直接计算。

考虑直接暴力转移，复杂度显然为 $O(\sum \frac{n^2}i=n^2\log n)$

##### ARC097C Sorted and Sorted

考虑如果确定了每个数最后的位置，则可以发现交换次数等于逆序对数量。

因为最后需要黑白分别递增，因此要求的序列可以看成将黑白的序列分别归并。

因此问题可以看成归并两个序列，使得逆序对数最小。直接 $dp$ 即可。

复杂度 $O(n^2)$

##### TopCoder 10727 RabbitPuzzle

可以发现一个状态最多有三种操作：两种是中间的向外跳，一种是外面的向内跳。

对于一个状态 $(a,b,c)$，把它看成一个点，它的父亲为向内跳的方案（如果 $2b=a+c$ 则不存在父亲），它的儿子为 $(2a-b,a,c)$ 和 $(a,c,2c-b)$。

可以发现一个点的儿子向内跳一定相当于跳回来，因此一个点的儿子的父亲是它自己，因此这个结构可以看成一个无向满二叉树。

先求出两个状态的 $k$ 层父亲，如果不交则显然无解，否则问题可以看成，有一个无穷大的二叉树，给定起点和终点，求 $k$ 步内到达的方案数。

因为图是满二叉树，所以状态只和深度有关。设 $dp_{i,j,k}$ 表示 $i$ 步之后，当前点深度为 $j$，当前点和终点的LCA深度为 $k$ 的方案数。直接转移即可。

复杂度 $O(k^3)$

##### TopCoder 10664 RowGame

不会

##### TopCoder 10566 IncreasingNumber

记 $v_i=\sum_{x=1}^i10^{x-1}$，则相当于找到 $0\leq s_9\leq s_8\leq...\leq s_1=n$，使得 $\sum_{i=1}^9 v_{s_i}$ 是 $p$ 的倍数。

考虑 $v_i\bmod p$，可以发现随着 $i$ 增加这个值的变化会构成一个 Rho 型，且环长不超过 $p$。

考虑枚举最后若干位的情况，这样就变成了环上的问题。

此时考虑钦定 $s_i-s_{i+1}<$ 环长，最后再将若干个环加进去，方案是一个组合数。

可以设 $dp_{i,j,k}$ 表示 $s_i=j$ 且前面的 $\sum v_{s_i}$ 模 $p$ 等于 $k$ 的方案数，转移可以前缀和。复杂度 $O(9^2p^2)$

对最后若干位可以再做一个dp。

##### TopCoder 10773 TheCitiesAndRoadsDivOne

显然合法的图是一个基环树。

考虑枚举环上的点集，可以求出这个点集连成环的方案数。

然后相当于给一个环以及部分森林，求连出一个基环树的方案数。

这可以看成给若干个连通块，求连出一棵树的方案数。根据prufer序，答案为 $n^{k-2}\prod s_i$。

复杂度 $O(n2^n)$

##### TopCoder 10993 SpaceshipEvacuation

显然所有人在同一个点的情况最极限。因此有解相当于从每个点出发的最小割大于等于 $v$，可以发现最小割一定是一条树边或者一个环上的两条边，因此一种方案合法当且仅当以下条件被满足：

1. 每个树边从下往上的方向流量大于等于 $v$。
2. 环上从任意点出发，到达环上深度最低的点的流量大于等于 $v$。

考虑条件2，设环上点为 $1,2,...,k$，则根据最小割的性质，这个条件相当于对于任意的 $1<i<j\leq k$，$i\to i-1$ 的流量和 $j\to j+1$ 的流量之和大于等于 $v$。

设 $dp_{i,j}$ 表示填了前 $i$ 个点，当前前面所有 $x\to x-1$ 的边流量的最小值为 $j$ 时前面的最小额外代价，可以发现 $dp$ 转移可以分成若干类，每一类转移到一个区间且转移系数固定，因此可以 $O(1)$ 转移。

复杂度 $O(nv)$

##### TopCoder 10741 ColorfulMaze

显然在过程中最多触发一次陷阱。设 $dp_{i,j,S,x}$ 表示当前在 $(i,j)$，当前试过了 $S$ 中的颜色，已知颜色 $x$ 是危险的，能通过的概率。

转移枚举下一次尝试哪个颜色即可。按照 $S$ 枚举即可不用每次重算连通性。

复杂度 $O(n^22^7*8)$

##### TopCoder 10854 DrawingBlackCrosses

设 $dp_{i,j,S}$ 表示当前还剩 $i$ 行 $j$ 列，当前剩下的黑色格子集合为 $S$。这里只考虑 $S$ 中的相对位置(行列相等)关系。

转移枚举下一次选的位置。可以发现没有黑色格子的行列之间是没有区别的，转移时只需要判断不能选黑色格子以及黑色格子占据一整行一整列的情况即可。

复杂度 $O(n^32^8*8)$

##### TopCoder 10848 NextHomogeneousStrings

判断合法只和连续 $n$ 位间的相对关系有关。可以发现 $Bell(8)\leq 6\times 10^5$，考虑数位 $dp$：$dp_{i,S}$ 表示当前后面还有 $i$ 位没有填，当前前面 $n$ 位的字符的最小表示法为 $S$，填后面的方案数。$dp$ 复杂度为 $O(Bell(n-1)*l*n)$

然后逐位确定答案即可。

##### TopCoder 10902 TheMoviesLevelThreeDivOne

不会

##### TopCoder 10758 ColorfulTiles

考虑固定了答案第一行的情况下，如何填剩下的行使得答案合法。

显然 $2\times 2$ 的方格中四个格子颜色必须不同，因此如果第一行中相邻的三个位置 $(1,i-1),(1,i),(1,i+1)$ 颜色两两不同，设它们的颜色为 $A,B,C$，则 $(2,i-1),(2,i)$ 颜色为 $C,D$，$(2,i),(2,i+1)$ 颜色为 $AD$，因此 $(2,i)$ 颜色一定为 $D$。

此时从这个位置开始，可以唯一确定第二行的填法，以此类推可以唯一确定剩下每个位置的填法。且可以发现此时每一列都是两种颜色在循环。

如果不存在这种情况，可以发现此时第一行的颜色为两种颜色循环，此时可以发现剩下的行也都是两种颜色循环。因此最后的方案一定满足列循环或者行循环。可以分别计数列循环和行循环的方案数，再减去两个都循环的方案。

设 $dp_{i,j,S}$ 表示填了前 $i$ 行，当前前面需要改 $j$ 个位置，上一行循环的颜色集合为 $S$。每一行只有两种情况，直接转移即可。

复杂度 $O(n^3)$

##### TopCoder 11032 TheBoardingDivOne

因为 $222\leq 74*3$，一个人被堵三次一定不行，因此一个人最多被堵两次。

考虑哪些人能在不被堵的情况下找到位置。显然最后一个人可以，设它的位置在 $p_n$，则所有满足 $p_i-i>p_n-n$ 的人都会被他和前面被堵住的人堵住。下一个可以找到位置的人即为 $p_i-i\leq p_n-n$ 的人中编号最大的。

因此可以发现第一次能找到的一定是 $p_i-i$ 的后缀非严格最小值。

再考虑第二次，此时所有人必须不被堵找到位置，因此此时所有人必须 $p_i$ 递增。

考虑从后向前 $dp$，设 $dp_{S,x}$ 表示考虑了后 $|S|$ 个位置，这些位置上的人构成集合 $S$，当前所有第二次找到位置的人中，$p_i$ 最小的为 $x$，且后面的人能在 $T$ 时刻内结束的方案数。

转移时枚举下一个人即可。

复杂度 $O(n^22^n)$

#### 0718

##### uoj667

###### Problem

给一棵 $n$ 个点的树，对于每个 $k$，求在树上写 $1,2,...,k$，每个点上最多写一个数，在树同构意义下本质不同的方案数，模 $998244353$。

$n\leq 10^5$

$1s,512MB$

###### Sol

考虑选出重心为根，则重构只需要考虑交换子树，即可以看成有根树的情况。

设 $dp_{u,i}$ 表示将 $u$ 的子树看成以 $u$ 为根的有根树，写 $i$ 个数本质不同的方案数。

考虑通过所有儿子的 $dp$ 计算一个点的 $dp$，对于不存在儿子重构的情况，相当于所有数可以在每个子树和 $u$ 中任意分配，不会出现交换 $u$ 的儿子导致的重构情况，因此考虑写成 EGF 的形式，设 $F_u(x)=\sum_{i>0} dp_{u,i}\frac{x^i}{i!}$，则可以发现：
$$
F_u(x)+1=(1+x)\prod_{v\in son_u}(F_v(x)+1)
$$
考虑存在重构儿子的情况，设有 $k$ 个儿子两两同构，它们为 $v_1,...,v_k$，显然这部分和其它部分之间不存在影响，因此可以计算只考虑这 $k$ 个儿子部分的 $dp$ 的生成函数。记一个儿子的生成函数为 $F_v(x)$。

枚举有多少个儿子中写了数，如果有 $a$ 个儿子中写了数，则不考虑交换这个点的儿子时，这些儿子内写数的方案为 $F_v(x)^a$，可以发现考虑交换儿子相当于乘以 $\frac 1{a!}$，因此这部分考虑重构的方案数为：
$$
\sum_{i=0}^k\frac{F_v(x)^i}{i!}
$$
可以使用分治FFT计算，复杂度为 $O(k*|F_v(x)|\log^2)$

如果一个点有一些子树重构，另外一些不重构，可以发现 $F_u(x)$ 即为每一部分分别的乘积再乘上 $x+1$ 的结果。

注意到如果不存在子树重构的情况，则转移中只有乘法，可以将所有乘的项拿出来分治FFT。因此计算一个 $F_u(x)$ 时，可以从 $u$ 开始向下dfs，对于一个点的一组重构的儿子，使用上面的方式计算这部分的生成函数。对于不重构的部分继续向下dfs，每访问到一个点就再乘上 $1+x$。最后将过程中每一个重构部分的生成函数和所有的 $1+x$ 乘起来即可。

因为每一组重构子树一定至少是一对，且计算 $F$ 只需要计算其中一个子树的 $F$，因此总共需要计算的 $F$ 的大小和一定是 $O(n)$ 的，总复杂度 $O(n\log^2 n)$

对于双重心的情况，如果两边不同构，则可以直接将两侧的答案合并，否则可以看成上一种情况中的两个同构子树。

注意树hash的写法

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 132001
#define mod 998244353
#define ll long long
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int g[2][N*2],rev[N*2],ntt[N],v1[N],v2[N],v3[N],fr[N],ifr[N];
void init(int d=17)
{
	fr[0]=ifr[0]=1;for(int i=1;i<1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int l=2;l<=1<<d;l<<=1)for(int i=0;i<l;i++)rev[i+l]=((rev[(i>>1)+l])>>1)|((i&1)*(l>>1));
	for(int t=0;t<2;t++)
		for(int l=2;l<=1<<d;l<<=1)
		{
			int tp=pw(3,(mod-1)/l),st=1;
			if(!t)tp=pw(tp,mod-2);
			for(int i=0;i<l>>1;i++)g[t][i+l]=st,st=1ll*st*tp%mod;
		}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[i+s]]=a[i];
	for(int l=2;l<=s;l<<=1)
		for(int i=0;i<s;i+=l)
			for(int j=0;j<l>>1;j++)
			{
				int v1=ntt[i+j],v2=1ll*ntt[i+j+(l>>1)]*g[t][j+l]%mod;
				ntt[i+j]=(v1+v2)%mod;
				ntt[i+j+(l>>1)]=(v1-v2+mod)%mod;
			}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	vector<int> c;
	for(int i=0;i<s1+s2-1;i++)c.push_back(0);
	if(s1+s2<=200)
	{
		for(int i=0;i<s1;i++)
			for(int j=0;j<s2;j++)c[i+j]=(c[i+j]+1ll*a[i]*b[j])%mod;
		return c;
	}
	int l=1;while(l<s1+s2)l<<=1;
	for(int i=0;i<l;i++)v1[i]=v2[i]=0;
	for(int i=0;i<s1;i++)v1[i]=a[i];
	for(int i=0;i<s2;i++)v2[i]=b[i];
	dft(l,v1,1);dft(l,v2,1);for(int i=0;i<l;i++)v1[i]=1ll*v1[i]*v2[i]%mod;
	dft(l,v1,0);
	for(int i=0;i<s1+s2-1;i++)c[i]=v1[i];
	return c;
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	vector<int> c;
	for(int i=0;i<s1||i<s2;i++)c.push_back(((i<s1?a[i]:0)+(i<s2?b[i]:0))%mod);
	return c;
}
int n,a,b,head[N],cnt,sz[N],as=1e7,s1,s2;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void dfs0(int u,int fa)
{
	sz[u]=1;int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),sz[u]+=sz[ed[i].t],mx=max(mx,sz[ed[i].t]);
	mx=max(mx,n-sz[u]);
	if(mx<as)as=mx,s1=u,s2=0;
	else if(mx==as)s2=u;
}
ll sv[N];
void dfs1(int u,int fa)
{
	sz[u]=1;
	vector<ll> st;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),st.push_back(sv[ed[i].t]),sz[u]+=sz[ed[i].t];
	ll rvl=1926+st.size();
	sort(st.begin(),st.end());
	for(int i=0;i<st.size();i++)rvl=(rvl*817+st[i])%102030405060709ll;
	rvl=(rvl*2333+sz[u])%102030405060709ll;
	sv[u]=rvl;
}
vector<vector<int> > fu[N];
int su[N];
vector<int> vc;
struct sth{vector<int> a,b;};
sth doit(int l,int r)
{
	if(l==r)
	{
		sth f1;
		f1.a=f1.b=vc;
		for(int i=0;i<f1.b.size();i++)f1.b[i]=1ll*f1.b[i]*ifr[l]%mod;
		return f1;
	}
	int mid=(l+r)>>1;
	sth sl=doit(l,mid),sr=doit(mid+1,r);
	return (sth){polymul(sl.a,sr.a),polyadd(sl.b,polymul(sl.a,sr.b))};
}
vector<int> doit2(int f,int l,int r)
{
	if(l==r)return fu[f][l];
	int mid=(l+r)>>1;
	return polymul(doit2(f,l,mid),doit2(f,mid+1,r));
}
vector<int> solve(int u,int fa);
void dfs(int u,int fa,int fr)
{
	su[fr]++;
	vector<pair<ll,int> > f1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)f1.push_back(make_pair(sv[ed[i].t],ed[i].t));
	sort(f1.begin(),f1.end());
	for(int i=0;i<f1.size();i++)if(!i||(f1[i].first!=f1[i-1].first))
		if(i+1==f1.size()||f1[i].first!=f1[i+1].first)dfs(f1[i].second,u,fr);
		else
		{
			int ct=1,tp=i;
			while(tp+1<f1.size()&&f1[tp+1].first==f1[tp].first)ct++,tp++;
			vector<int> s1=solve(f1[i].second,u);
			vc=s1;vc[0]--;
			vector<int> s2=doit(1,ct).b;s2[0]=1;fu[fr].push_back(s2);
		}
}
vector<int> solve(int u,int fa)
{
	dfs(u,fa,u);
	vector<int> s0;s0.push_back(1);fu[u].push_back(s0);
	vector<int> s1=doit2(u,0,fu[u].size()-1),s2;
	for(int i=0;i<=su[u];i++)s2.push_back(1ll*fr[su[u]]*ifr[su[u]-i]%mod*ifr[i]%mod);
	return polymul(s1,s2);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	init();
	dfs0(1,0);
	dfs1(s1,s2);dfs1(s2,s1);
	vector<int> as;
	if(!s2)as=solve(s1,s2);
	else if(sv[s1]!=sv[s2])as=polymul(solve(s1,s2),solve(s2,s1));
	else
	{
		vector<int> t1=solve(s1,s2);
		vc=t1;vc[0]--;
		t1=doit(1,2).b;
		t1[0]++;
		as=t1;
	}
	for(int i=1;i<=n;i++)printf("%d ",1ll*fr[i]*as[i]%mod);
}
```

#### 0719

##### CodeChef PARADE Annual Parade

###### Problem

给一个 $n$ 个点 $m$ 条边的有向图，每条边有边权。

你需要选择若干条路径，这种方案的代价由以下几部分组成：

1. 每条路径长度的和
2. 每有一条路径的长度不等于结尾，加上 $C$ 的代价
3. 每有一个点没有被经过，加上 $C$ 的代价。

给出 $q$ 次询问，每次给出 $C$，求最小总代价。

$n\leq 250,q\leq 10^4$

$0.62s,512MB$

###### Sol

考虑依次加入每条路径，定义路径上的一个点是关键点当且仅当这个点没有被之前的路径覆盖。

可以发现，最优解中每个点至少包含一个关键点，且以关键点开头以关键点结尾。

考虑只保留路径上的关键点，在相邻两个关键点 $(i,j) $ 之间，最小距离显然是 $dis_{i\to j}$。

此时考虑一个新的图，其中 $(i\to j)$ 的边权为原图的 $dis_{i\to j}$。此时路径满足如下限制：

1. 每条路径是一条链或者一个环。
2. 每个点只被一条路径覆盖。

考虑在这个图上选路径，每条路径对应原来这个路径经过的所有关键点，可以发现这样一定存在一种方式等于最优解。

再考虑代价中关于 $C$ 的部分，一个没有被覆盖的点和一条链都会让代价 $+C$，此时可以发现，这部分代价即为 $(n-路径总边数)*C$。

可以发现，在 $C$ 不固定的时候，只需要对于每一个 $k$ ，求出选 $k$ 条满足条件的边的最小边权 $w_k$，答案即为 $\min_{i=0}^kw_i+(n-i)C$。

考虑求 $w_i$，可以发现上面的限制相当于每个点入度出度不超过 $1$，因此可以看成费用流（最大权匹配）的形式，每个点拆成 $in_u,out_u$ 两个点，连边 $(s\to in_u,1),(out_u\to t,1),(in_i\to out_j,dis_{i\to j})$ 即可。

复杂度 $O(n^3+nq)$，~~为数不多spfa费用流比dij费用流快得多的题~~

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 505
int n,m,q,a,b,c,head[N],cnt=1,dis[N],ct,cur[N],as,vis[N],ds[N][N],tp[N];
struct edge{int t,next,v,c;}ed[N*N];
void adde(int f,int t,int v,int c)
{
	ed[++cnt]=(edge){t,head[f],v,c};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0,-c};head[t]=cnt;
}
bool dij(int s,int t)
{
	for(int i=1;i<=ct;i++)dis[i]=1.01e9,vis[i]=0,cur[i]=head[i];
	priority_queue<pair<int,int> > qu;
	qu.push(make_pair(0,s));dis[s]=0;
	while(!qu.empty())
	{
		int x=qu.top().second;qu.pop();
		if(vis[x])continue;vis[x]=1;
		for(int i=head[x];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]>dis[x]+ed[i].c)
		{
			dis[ed[i].t]=dis[x]+ed[i].c;
			qu.push(make_pair(-dis[ed[i].t],ed[i].t));
		}
	}
	for(int i=1;i<=ct;i++)
		for(int j=head[i];j;j=ed[j].next)ed[j].c+=dis[i]-dis[ed[j].t];
	return dis[t]<1e9;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	vis[u]=1;
	int as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)if(!ed[i].c&&ed[i].v&&!vis[ed[i].t]&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)break;
	}
	vis[u]=0;
	return as;
}
void dij_mcmf(int s,int t)
{
	for(int i=1;i<=n;i++)tp[i]=1e9;
	int as=0,ds=0,t1=0,t2;
	while(dij(s,t))
	{
		for(int i=1;i<=ct;i++)vis[i]=0;
		ds+=dis[t],t2=dfs(s,t,1e9);
		for(int i=t1+1;i<=t1+t2;i++)tp[i]=ds+tp[i-1];
		t1+=t2,as+=ds*t2;
	}
}
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(i!=j)ds[i][j]=1e9;
	for(int i=1;i<=m;i++)scanf("%d%d%d",&a,&b,&c),ds[a][b]=min(ds[a][b],c);
	for(int k=1;k<=n;k++)for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)ds[i][j]=min(ds[i][j],ds[i][k]+ds[k][j]);
	ct=n*2+2;
	for(int i=1;i<=n;i++)adde(ct-1,i,1,0),adde(i+n,ct,1,0);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(i!=j&&ds[i][j]<1e8)adde(i,j+n,1,ds[i][j]);
	dij_mcmf(ct-1,ct);
	while(q--)
	{
		scanf("%d",&c);
		int as=c*n;
		for(int i=1;i<=n;i++)as=min(as,c*(n-i)+tp[i]);
		printf("%d\n",as);
	}
}
```

