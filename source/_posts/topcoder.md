---
title: Topcoder Selection
date: '2022-01-28 22:14:01'
updated: '2022-01-28 22:14:01'
tags: Mildia
permalink: VistoriyLess/
description: Topcoder
mathjax: true
---

2021.11(29/31)&2022.1(10/11)

可能有一些题在集训里面出现过了，这里就省略了。~~有几个题太简单了也省略了~~

默认限制为 $2s,256MB$

##### SRM712 TC14519 BinaryTreeAndPermutation

###### Problem

给一棵 $n$ 个点的有根二叉树，满足每个点有 $0$ 或 $2$ 个儿子。

你需要构造一个 $n$ 阶排列 $p$ ，满足给出的 $m$ 个限制，每个限制形如 $(a_i,b_i,c_i)$ ，表示排列需要满足
$$
LCA(p_{a_i},p_{b_i})=c_i
$$
构造任意方案或输出无解。

$n,m\leq 50$

###### Sol

首先考虑所有的 $c_i$ 都等于根的情况。此时一条限制相当于要求 $(a_i,b_i)$ 满足如下限制之一：

1. $(p_{a_i},p_{b_i})$ 中有一个是根
2. $(p_{a_i},p_{b_i})$ 在根的不同子树中

考虑枚举使得 $p_x$ 为根的 $x$ ，那么此时相当于对于不包含 $x$ 的限制，这个限制的两个位置需要在不同的子树中。

此时将限制看成边，则有解当且仅当图是二分图，且对于一个连通块，它黑白染色后的一侧必须被放在同一个子树中。

现在相当于决定将每个连通块的黑色部分放在左侧还是右侧，使得两边的大小都不会超过子树大小。因此可以 $dp$ 解决。

$dp$ 的复杂度为 $O(n^2)$ ，乘以枚举点的复杂度为 $O(n^3)$。

考虑 $c_i$ 任意的情况。一条限制 $(a_i,b_i,c_i)$ 要求了 $p_{a_i},p_{b_i}$ 都在 $c_i$ 子树内。因此考虑在树上从下往上构造，每次对于一个 $x$ ，处理所有满足 $c_i=x$ 的限制。此时与上面部分唯一的区别在于有些点可能已经确定了属于哪个子树，相当于染色时有些点颜色固定，这不影响上面的做法。

同时可以注意到，上面的做法只考虑了两个子树内部有多少个可用的位置，子树内空余位置的排列情况不影响上面的过程，因此对于一个点构造方案时任意构造一个即可。

复杂度 $O(n^3+m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 55
int n,m,ch[N][2],s[N][3],fg,fg1,fr[N],pr[N],p1[N],head[N],cnt,is[N],c0,c1,st[N][2],f1[N],vis[N],ct,dp[N][N],fr2[N][N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
vector<int> sn[N];
void dfs1(int x,int fi)
{
	int f3=0;
	for(int i=1;i<=n;i++)if(pr[i]==x)f3=1;
	if(!f3)st[fi][fr[x]-1]++;f1[x]=fi;vis[x]=1;
	for(int i=head[x];i;i=ed[i].next)
	if(!fr[ed[i].t])fr[ed[i].t]=3-fr[x],dfs1(ed[i].t,fi);
	else if(fr[ed[i].t]==fr[x])fg1=1;
}
void dfs(int x)
{
	sn[x].push_back(x);
	if(!ch[x][0])
	{
		int v1=0;
		for(int i=1;i<=m;i++)if(s[i][2]==x)
		{
			if(v1)v1=v1==s[i][0]?s[i][0]:-1;else v1=s[i][0];
			if(v1)v1=v1==s[i][1]?s[i][1]:-1;else v1=s[i][1];
		}
		if(v1==-1)fg=1;else if(v1)pr[x]=v1;
		return;
	}
	dfs(ch[x][0]);dfs(ch[x][1]);
	for(int i=1;i<=n;i++)is[i]=fr[i]=0;c0=c1=0;
	for(int i=0;i<sn[ch[x][0]].size();i++)fr[pr[sn[ch[x][0]][i]]]=1,sn[x].push_back(sn[ch[x][0]][i]),c0+=!pr[sn[ch[x][0]][i]];
	for(int i=0;i<sn[ch[x][1]].size();i++)fr[pr[sn[ch[x][1]][i]]]=2,sn[x].push_back(sn[ch[x][1]][i]),c1+=!pr[sn[ch[x][1]][i]];
	for(int i=1;i<=m;i++)if(s[i][2]==x)
	{
		is[s[i][0]]=is[s[i][1]]=1;
		if(fr[s[i][0]]&&fr[s[i][0]]==fr[s[i][1]]){fg=1;return;}
	}
	int v1=0;
	vector<int> fu;fu.push_back(0);
	for(int i=1;i<=n;i++)if(is[i]&&!fr[i])fu.push_back(i);
	for(int i=1;i<=m;i++)if(s[i][0]==s[i][1]&&s[i][2]==x)v1=v1?(v1==s[i][0]?s[i][0]:-1):s[i][0];
	if(v1==-1||(v1&&fr[v1])){fg=1;return;}
	if(v1)fu.clear(),fu.push_back(v1);
	for(int t=0;t<fu.size();t++)
	{
		int u=fu[t];
		for(int i=0;i<=n;i++)head[i]=fr[i]=f1[i]=vis[i]=st[i][0]=st[i][1]=0;cnt=fg1=ct=0;
		for(int i=0;i<sn[ch[x][0]].size();i++)fr[pr[sn[ch[x][0]][i]]]=1;
		for(int i=0;i<sn[ch[x][1]].size();i++)fr[pr[sn[ch[x][1]][i]]]=2;
		for(int i=1;i<=m;i++)if(s[i][2]==x&&s[i][0]!=u&&s[i][1]!=u)adde(s[i][0],s[i][1]);
		for(int i=1;i<=n;i++)if(is[i]&&fr[i]&&!vis[i])dfs1(i,0);
		for(int i=1;i<=n;i++)if(is[i]&&!vis[i]&&i!=u)fr[i]=1,dfs1(i,++ct);
		if(fg1)continue;
		for(int i=1;i<=ct;i++)for(int j=0;j<=n;j++)dp[i][j]=0;
		dp[0][0]=1;
		for(int i=1;i<=ct;i++)
		{
			for(int j=0;j<=n;j++)if(dp[i-1][j])dp[i][j+st[i][0]]=1,fr2[i][j+st[i][0]]=0;
			for(int j=0;j<=n;j++)if(dp[i-1][j])dp[i][j+st[i][1]]=1,fr2[i][j+st[i][1]]=1;
		}
		int su=0,fr1=-1;
		for(int i=0;i<=ct;i++)su+=st[i][0]+st[i][1];
		for(int i=0;i<=n;i++)if(dp[ct][i]&&i+st[0][0]<=c0&&su-i-st[0][0]<=c1)fr1=i;
		if(fr1==-1)continue;
		for(int i=ct;i>=1;i--)
		{
			int tp=fr2[i][fr1];
			fr1-=st[i][tp];
			if(tp)for(int j=1;j<=n;j++)if(f1[j]==i)fr[j]=3-fr[j];
		}
		for(int j=1;j<=n;j++)
		{
			int f2=0;
			for(int i=1;i<=n;i++)if(pr[i]==j)f2=1;
			if(f2)continue;
			if(j==u)pr[x]=j;
			else if(is[j]&&fr[j]==1){for(int i=0;i<sn[ch[x][0]].size();i++)if(!pr[sn[ch[x][0]][i]]){pr[sn[ch[x][0]][i]]=j;break;}}
			else if(is[j]&&fr[j]==2){for(int i=0;i<sn[ch[x][1]].size();i++)if(!pr[sn[ch[x][1]][i]]){pr[sn[ch[x][1]][i]]=j;break;}}
		}
		return;
	}
	fg=1;return;
}
struct BinaryTreeAndPermutation{
	vector<int> findPermutation(vector<int> s0,vector<int> s1,vector<int> a,vector<int> b,vector<int> c)
	{
		vector<int> as;
		n=s0.size(),m=a.size();
		for(int i=1;i<=n;i++)ch[i][0]=s0[i-1]+1,ch[i][1]=s1[i-1]+1;
		for(int i=1;i<=m;i++)s[i][0]=a[i-1]+1,s[i][1]=b[i-1]+1,s[i][2]=c[i-1]+1;
		dfs(1);
		if(fg)return as;
		for(int i=1;i<=n;i++)p1[pr[i]]=i;
		for(int i=1;i<=n;i++)if(!p1[i])for(int j=1;j<=n;j++)if(!pr[j]){pr[j]=i,p1[i]=j;break;}
		for(int i=1;i<=n;i++)as.push_back(p1[i]-1);
		return as;
	}
};
```

##### SRM643 TC13501 CasinoGame

###### Problem

有 $n$ 个数 $a_{1,...,n}$。你有一个分数，初始为 $0$。循环执行如下三步操作直到所有数都被删去：

1. 在没有被删去的数中随机选择一个，你的分数加上它当前的值，随后删去这个数。
2. 对当前每个没有被删去的数分别进行一次随机操作。

对一个数进行随机操作的方式如下：

1. 有 $33\%$ 的概率，将这个数删去，但不加入你的分数。
2. 有 $33\%$ 的概率，将这个数除以 $2$。
3. 有 $33\%$ 的概率，将这个数开根。
4. 有 $1\%$ 的概率，将这个数变回这个数初始的值。

求结束时你分数的期望，输出实数，绝对或相对误差不超过 $10^{-9}$。

$n\leq 1000,10^{-3}\leq a_i\leq 10^3$

###### Sol

设 $f_{i}$ 表示操作 $i$ 次后还有数没有被删去的概率，则下一步对分数的贡献为随机选出一个数，这个数当前的值。

因为所有数在操作上没有区别，因此在剩下的数中随机选择一个的期望值和不考虑删去的情况下所有数中随机选一个的期望值相同。考虑计算 $g_i$ 表示不考虑删除（$\frac{33}{67}$ 开根，$\frac{33}{67}$ 除以 $2$，$\frac 1{67}$ 还原）时，所有数操作 $i$ 次后的和的期望，则答案为：
$$
\sum_{i}f_ig_i*\frac 1n
$$
考虑操作一个数的情况，只考虑开根和除以 $2$ 两种操作。此时可以倒过来考虑，设之后的开根次数为 $k$，则一次除以 $2$ 相当于对于最后的值除以 $2^{\frac1{2^k}}$。这样除法就不会被（倒过来之后）之后的开根影响。

设 $dp_{i,j}$ 表示操作了 $i$ 次，且其中有 $j$ 次开根时，这些操作中除法操作对最后值的影响的乘积的和。则有：
$$
dp_{i,j}=\frac12(dp_{i-1,j-1}+dp_{i-1,j}*2^{-\frac1{2^j}})
$$
此时一个数 $v_i$ 操作 $j$ 次后值的期望为：
$$
\sum_{k=0}^jv_i^{\frac1{2^k}}dp_{j,k}
$$
再考虑还原，枚举最后一次还原的时刻，可以得到此时一个数 $v_i$ 操作 $x$ 次后值的期望为：
$$
(\frac{66}{67})^x\sum_{k=0}^xv_i^{\frac1{2^k}}dp_{x,k}+\sum_{j=0}^{x-1}(\frac{66}{67})^x*\frac 1{67}*\sum_{k=0}^jv_i^{\frac1{2^k}}dp_{j,k}
$$
对于第二部分可以前缀和，这样可以 $O(n^2)$ 求出一个数的贡献。

同时可以发现，只要预处理出 $\sum_{i=1}^nv_i^{\frac1{2^k}}$，就可以 $O(n^2)$ 求出所有数的贡献和。

最后考虑算 $f$。设 $h_{i,j}$ 表示操作 $i$ 次后剩下 $j$ 个数的概率，则：
$$
h_{i,j}=\sum_{k=j+1}^nC_{k-1}^j(\frac{33}{100})^{k-1-j}(\frac{67}{100})^jh_{i-1,k}
$$
组合系数可以预处理，这样可以得到一个小常数的 $O(n^3)$ 做法，可以通过。

在此基础上，考虑只转移 $h>\epsilon$ 的值，这样不会造成超过 $\epsilon*n^2$ 的相对误差。可以取 $\epsilon=10^{-25}$ ，这样可以大幅提升速度，计算次数接近 $O(n^2)$ 级别。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<vector>
using namespace std;
#define N 1050
int n,v[N];
double sr[N],dp[N],dp2[N],as,c[N][N],pr[N],su1[N];
double f1[N][N];
void init()
{
	pr[0]=0.5;for(int i=1;i<=n;i++)pr[i]=sqrt(pr[i-1]);
	f1[0][0]=1;
	for(int i=1;i<=n;i++)
	for(int j=0;j<=n;j++)
	{
		f1[i][j]+=f1[i-1][j]*pr[j]*33/67;
		f1[i][j+1]+=f1[i-1][j]*33/67;
	}
	for(int i=1;i<=n;i++)
	{
		double nw=v[i]*1e-3;
		for(int k=0;k<=n;k++)su1[k]+=nw,nw=sqrt(nw);
	}
	double su=0;
	for(int j=0;j<=n;j++)
	{
		double s1=0;
		for(int k=0;k<=n;k++)s1+=su1[k]*f1[j][k];
		sr[j]=su+s1;su+=s1/67;
	}
}
struct CasinoGame{
	double expectedValue(vector<int> s)
	{
		n=s.size();
		for(int i=1;i<=n;i++)v[i]=s[i-1];
		init();
		dp[n]=1;
		c[0][0]=1;for(int i=1;i<=n;i++)c[i][0]=c[i-1][0]*0.33;
		for(int i=1;i<=n;i++)for(int j=1;j<=i;j++)c[i][j]=c[i-1][j]*0.33+c[i-1][j-1]*0.67;
		for(int i=0;i<=n;i++)
		{
			for(int j=1;j<=n;j++)as+=dp[j]*sr[i]/n;
			for(int j=1;j<=n-i;j++)if(dp[j]>1e-25)
			for(int k=1;k<j;k++)dp2[k]+=dp[j]*c[j-1][k];
			for(int j=1;j<=n-i;j++)dp[j]=dp2[j],dp2[j]=0;
		}
		return as;
	}
};
```

##### SRM470 TC10737 BuildingRoads

###### Problem

有一个 $n\times m$ 的网格，其中有一些格子上有城市。一共有 $d$ 对城市。

一些格子上有障碍，一个障碍可能占据多个连通的格子。每个障碍有移除它的代价 $c_i$。

你需要移除一些障碍，使得给出的 $d$ 对城市满足每一对城市间连通。输出最小总代价。

$n,m\leq 50,d\leq 4$

###### Sol

将所有 $2d$ 个城市看作关键点。将所有障碍缩点，可以得到一个图，选择一些点有代价，要求选择一些点使得每一对给出点连通。

考虑用Steiner Tree的做法，对于每个关键点的子集，求出将这个子集连通的最小代价。然后再做一次状压dp就能求出将每一对都连通的最小代价。

注意到将一个点集连通的最小代价方案一定形如一棵树，设 $dp_{x,S}$ 表示当前的根是 $x$，当前子树中有集合 $S$ 中的关键点时，子树的最小代价。转移有两种：

1. $dp_{y,S}=min(dp_{y,S},dp_{x,S}+v_y)$，要求 $(x,y)$ 有边。其中 $v_y$ 是点权。这种转移表示儿子向父亲转移。
2. $dp_{y,S}=min(dp_{y,S},dp_{y,T}+dp_{y,S-T}-v_y)$。这种转移表示合并一个点的两个子树。

第一种转移对于一个 $S$ 相当于最短路。因此可以从小到大枚举 $S$，先做 $S$ 的第二种转移，然后做最短路。

本题中复杂度为 $O(nm(3^{2d}+2^{2d}\log nm))$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
#include<vector>
#include<string>
using namespace std;
#define N 55
#define M 2505
int n,m,fr[N][N],ct,vl[M],head[M],cnt,is,d[4][2]={-1,0,1,0,0,1,0,-1},c1[4];
char s[N][N];
struct edge{int t,next;}ed[M*4];
void adde(int f,int t){if(f==t)return;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int getid(int x,int y)
{
	char st=s[x][y];
	if(st=='.')return 0;
	if(st>='a'&&st<='z')return st-'a'+1;
	if(st>='A'&&st<='Z')return (st-'A'+1)*100;
	if(st>='1'&&st<='9')return (st-'1'+1)*10000;
	if(st=='0')return 100000;
	if(st=='#')return -1;
	if(st=='@')return -2;
	if(st=='$')return -3;
	return -4;
}
void dfs(int x,int y,int id)
{
	fr[x][y]=id;
	for(int i=0;i<4;i++)
	{
		int nx=x+d[i][0],ny=y+d[i][1];
		if(s[nx][ny]==s[x][y]&&!fr[nx][ny])dfs(nx,ny,id);
	}
}
int dp[M][257],vis[M],dp2[17];
struct BuildingRoads{
	int destroyRocks(vector<string> s1)
	{
		n=s1.size();m=s1[0].size();
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)s[i][j]=s1[i-1][j-1];
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(!fr[i][j])
		{
			fr[i][j]=++ct;vl[ct]=getid(i,j);
			if(vl[ct]>=0)dfs(i,j,ct);else is|=1<<(-1-vl[ct]);
		}
		for(int i=1;i<=n;i++)for(int j=1;j<m;j++)adde(fr[i][j],fr[i][j+1]);
		for(int i=1;i<n;i++)for(int j=1;j<=m;j++)adde(fr[i][j],fr[i+1][j]);
		for(int i=1;i<=ct;i++)for(int j=1;j<256;j++)dp[i][j]=1e9;
		for(int i=1;i<=ct;i++)if(vl[i]>=0)dp[i][0]=vl[i];else dp[i][1<<((-1-vl[i])*2+(c1[-1-vl[i]]++))]=dp[i][0]=0,vl[i]=0;
		for(int i=1;i<256;i++)
		{
			for(int j=1;j<=ct;j++)
			for(int k=(i-1)&i;k;k=(k-1)&i)dp[j][i]=min(dp[j][i],dp[j][i^k]+dp[j][k]-vl[j]);
			priority_queue<pair<int,int> > qu;
			for(int j=1;j<=ct;j++)vis[j]=0,qu.push(make_pair(-dp[j][i],j));
			while(!qu.empty())
			{
				int tp=qu.top().second;qu.pop();
				if(vis[tp])continue;vis[tp]=1;
				for(int l=head[tp];l;l=ed[l].next)if(dp[ed[l].t][i]>dp[tp][i]+vl[ed[l].t])dp[ed[l].t][i]=dp[tp][i]+vl[ed[l].t],qu.push(make_pair(-dp[ed[l].t][i],ed[l].t));
			}
		}
		for(int i=1;i<16;i++)dp2[i]=1e9;
		for(int i=1;i<=ct;i++)
		for(int j=0;j<256;j++)
		{
			int v1=0;
			for(int k=0;k<4;k++)if((j>>(k*2))%4==3)v1|=1<<k;
			dp2[v1]=min(dp2[v1],dp[i][j]);
		}
		for(int i=1;i<16;i++)for(int j=(i-1)&i;j;j=(j-1)&i)dp2[i]=min(dp2[i],dp2[j]+dp2[i^j]);
		return dp2[is];
	}
};
```

##### TCO16 Semi1 TC14447 ColorfulPath

###### Problem

有一个 $n+1$ 个点的有向图，点被编号为 $0,1,...,n$。

图中有 $m$ 条有向边 $s_i\to t_i$，这些边满足如下条件：

1. $s_i<t_i$，因此图是一个DAG。
2. 不存在 $i,j$ 使得 $a_i<a_j<b_i<b_j$>

每条边有边权 $w_i$。

除去 $0,n$ 外，每个点有一个颜色 $c_i$。你需要选择一条从 $0$ 到 $n$ 的路径，使得对于每一种颜色，这种颜色的点全部被经过或者全部没有被经过。在此基础上，你需要让路径最短。

输出满足条件的最短路径的长度，或者输出无解。

$n,m\leq 1000$

###### Sol

考虑从 $0$ 出发，每次向能走到的编号最大的点走，得到一条路径 $(0,p_1,...,p_k,n)$。那么存在如下性质：

> 对于任意一条从 $0$ 出发到 $n$ 的路径，它一定经过了点 $p_1,...,p_k$。

如果一条边 $(s_i,t_i)$ 跨过了一个点 $p_i$，因为每次都走的是编号最大的点，因而 $s_i$ 不可能等于某个 $p_j$。

因而存在 $j$ 使得 $p_j<s_i<p_{j+1}\leq p_i<t_i$ ，与条件矛盾。

因此，可能的路径一定可以分成 $(0,p_1),(p_1,p_2),...,(p_k,n)$ 这些段。

类似地可以发现，如果这样走不到 $n$ ，那么一定不存在能走到 $n$ 的路径

考虑一段内部的情况，这一段内部可以走 $(p_i,p_{i+1})$ 的边，也可以走内部其它的边。对于第二种情况，相当于这个区间内的子问题（此时不走 $(p_i,p_{i+1})$ 的边），因此可以继续使用上面的做法。

称对 $(i,j)$ 做上面做法得到的路径为 $(i,j)$ 的扩展路径。则一条路径一定可以被这样构造出来：

初始路径为 $(0,n)$ 的扩展路径，进行若干次操作，每次选择一条边删去，换为它的扩展路径。因而扩展一条路径的代价为路径边权和减去原来边的边权。

选择扩展 $(i,j)$ 前必须满足 $(i,j)$ 在路径中，即扩展了 $(i,j)$ 这条边所在的扩展路径，因此可以发现扩展路径间构成一棵树的关系，必须选择父亲才能选择儿子。

此时再考虑颜色的限制。因为扩展一条边只会增加经过的点，因此可以看成一条扩展路径经过了若干点，要求所有选择的扩展路径经过的点满足限制。限制可以看成对于一种颜色的所有点 $p_1,...,p_l$，如果经过了 $p_i$，则需要经过 $p_{i\ \bmod \ l+1}$。

此时所有的限制都可以看成形如如果选择了 $a$，则必须选择 $b$。因此问题变为最小权闭合子图的形式。将权值乘 $-1$ 后，问题可以变为最小割，$dinic$ 求即可。

复杂度 $O(dinic(n,n))$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
#define N 1050
int n,m,s[N][3],cl[N],fr[N],ct,fa[N],vl[N];
vector<pair<int,int> > nt[N];
vector<int> sr[N];
void build(int l,int r,int id,int v)
{
	vector<int> st;st.push_back(l);
	int s1=lower_bound(nt[l].begin(),nt[l].end(),make_pair(r-1,0))-nt[l].begin()-1;
	if(s1<0)return;
	int su=-nt[l][s1].second,nw=nt[l][s1].first;st.push_back(nw);
	while(nw!=r)
	{
		int s2=lower_bound(nt[nw].begin(),nt[nw].end(),make_pair(r,0))-nt[nw].begin()-1;
		if(s2<0)return;
		su-=nt[nw][s2].second,nw=nt[nw][s2].first,st.push_back(nw);
	}
	fa[++ct]=id;vl[ct]=su-v;
	sr[ct]=st;id=ct;
	for(int i=1;i+1<st.size();i++)fr[st[i]]=ct;
	for(int i=0;i+1<st.size();i++)
	{
		int l1=st[i],r1=st[i+1];
		int s2=lower_bound(nt[l1].begin(),nt[l1].end(),make_pair(r1,0))-nt[l1].begin()-1;
		build(l1,r1,id,-nt[l1][s2].second);
	}
}
int head[N],cnt=1,dis[N],cur[N],ls[N];
struct edge{int t,next,v;}ed[N*10];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;}
bool bfs(int s,int t)
{
	for(int i=0;i<=n+1;i++)cur[i]=head[i],dis[i]=-1;
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
int solve()
{
	int as=1.01e9,s2=as;
	for(int i=0;i<nt[0].size();i++)if(nt[0][i].first==n)as=min(as,-nt[0][i].second);
	build(0,n,0,1.01e9);
	for(int i=2;i<=ct;i++)adde(i,fa[i],2.02e9);
	for(int i=1;i<n;i++)if(!fr[i])fr[i]=ct+1;
	for(int i=1;i<n;i++)
	{
		if(ls[cl[i]])adde(fr[ls[cl[i]]],fr[i],2.02e9);
		ls[cl[i]]=i;
	}
	for(int i=1;i<n;i++)if(ls[cl[i]])adde(fr[ls[cl[i]]],fr[i],2.02e9),ls[cl[i]]=0;
	for(int i=1;i<=ct;i++)if(vl[i]>0)adde(i,n+1,vl[i]);else adde(0,i,-vl[i]),s2+=vl[i];
	adde(ct+1,n+1,2.02e9);
	while(bfs(0,n+1))
	{
		int vl=dfs(0,n+1,1e9);
		if(1ll*s2+vl>1.01e9)s2=1.01e9;else s2+=vl;
	}
	if(as>s2)as=s2;if(as>1e9)as=-1;
	return as;
}
struct ColorfulPath{
	int shortestPath(vector<int> a,vector<int> b,vector<int> c,vector<int> c1)
	{
		n=c1.size()+1;m=c.size();
		for(int i=1;i<=m;i++)nt[a[i-1]].push_back(make_pair(b[i-1],-c[i-1]));
		for(int i=0;i<n;i++)sort(nt[i].begin(),nt[i].end());
		for(int i=1;i<n;i++)cl[i]=c1[i-1];
		return solve();
	}
};
```

##### SRM777 TC15774 BlackAndWhiteBalls

###### Problem

有一个由黑色和白色的球组成的序列。序列被分成了 $n$ 段，第 $i$  段的长度为 $l_i$。且满足奇数段内都是白球，偶数段内都是黑球。

你需要将这个序列划分成若干个区间，然后给每个区间标记为黑色或者白色。你需要满足每个被标记成黑色的区间内都正好有 $b$ 个黑球，每个被标记成白色的区间内都正好有 $w$ 个白球。 

求方案数，对 $10^9+7$ 取模。

$n\leq 100,l_i,w,b\leq 10^9$

###### Sol

直接的暴力是设 $dp_i$ 表示前 $i$ 个球分段的方案数，然后枚举下一段的颜色，一定是转移到一段区间。这样的复杂度为 $O(nl)$。

但转移时会出现将一段的 $dp$ 平移后加到另外一段上，这部分难以优化转移。

注意到这种情况只会在连续转移同一种颜色时出现，可以考虑每次将同一种颜色的段转移完。

如果上一段是黑色，下一段是白色，此时有如下情况：

1. 划分点在黑色段内，则剩余的黑色部分对下一段白色没有影响，因此可以看成划分点在分界点的情况。
2. 划分点在白色段内或者在分界点。此时白色部分对上一段黑色没有影响，因而划分点在白色中的每一个位置的方案数相同。

因此可以设 $f_i$ 表示第一种情况转移到第 $i$ 段的开头，且下一段需要是第 $i$ 段的颜色的情况数。设 $g_i$ 表示第二种情况转移到第 $i$ 段内部的方案数。

考虑 $f_i$ 向后转移，暴力做法是每次划分一段，然后转移这一段后的情况。如果这一次划分后结尾在某一段的中间，则因为下一次转移另外一种色，因此会转移到这一段之后的 $f$。如果这一次划分后在一段结尾，则会转移到下一段的 $g$。同时如果继续向后划分相同颜色的段，则这里的划分点有 $l_i+1$ 种等价的方案，之后的方案数需要乘上 $l_i+1$。

同时，如果 $f$ 能正好划分到结尾，那么可以对答案做贡献。

显然一段内部每次是向右跳 $w$ 个或者 $b$ 个，一段内部的情况可以快速处理。转移一次的复杂度为 $O(n)$。

对于 $g$ ，暴力做法是枚举开头这一段的长度，然后做类似 $f$ 的转移。

可以发现上面转移中唯一的特殊情况是划分到一段结尾的情况，而这种情况只在这部分的长度是 $w$ 的倍数时出现。此时只有 $O(n)$ 种初始长度模 $w$ 的余数会导致特殊转移，因此，可以先不考虑特殊转移做整体转移，然后对每一种特殊情况单独转移。

在不考虑特殊转移时，可以发现在这一段中停止的方案数形如 $\sum_{i=l}^r\lfloor\frac{x+a}b\rfloor$。因为 $x$ 上没有系数，可以 $O(1)$ 直接算出。

对于特殊转移，可以发现此时每一段的方案数形如 $\sum_{i=l}^r\lfloor\frac{bx+a}b\rfloor$，也容易计算。因此单次转移的复杂度仍然是 $O(n)$。

同时因为必须转移一段这种颜色，因此转移时需要特判边界（不能转移 $0$ 段）。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
#include<vector>
using namespace std;
#define N 105
#define mod 1000000007
#define ll long long
int n,s[2],v[N],dp[N][2],as;
int calc1(ll l,int k)
{
	if(l<=0)return 0;
	return (1ll*(l/k)*(l%k+1)+1ll*(l/k)%mod*((l/k-1)%mod)/2%mod*k)%mod;
}
int calc(ll l,ll r,int k)
{
	return (mod+calc1(r,k)-calc1(l-1,k))%mod;
}
struct BlackAndWhiteBalls{
	int getNumber(vector<int> t,int a,int b)
	{
		s[1]=a;s[0]=b;n=t.size();
		for(int i=1;i<=n;i++)v[i]=t[i-1];
		dp[1][0]=dp[2][0]=1;
		for(int i=1;i<=n;i++)
		{
			ll su=0,vl=dp[i][0];
			set<int> fu;
			for(int j=i;j<=n;j+=2)
			{
				ll nt=su+v[j];
				fu.insert(nt%s[i&1]);
				dp[j+1][0]=(dp[j+1][0]+1ll*((nt-1)/s[i&1]-su/s[i&1])*vl)%mod;
				if(nt%s[i&1]==0&&nt){dp[j+1][1]=(dp[j+1][1]+vl)%mod;if(j+1<n)vl=1ll*vl*(v[j+1]+1)%mod;}
				su=nt;
			}
			if(su%s[i&1]==0&&su)as=(as+vl)%mod;
			su=-v[i],vl=dp[i][1];ll v1=1;
			for(int j=i;j<=n;j+=2)
			{
				ll nt=su+v[j];
				dp[j+1][0]=(dp[j+1][0]+1ll*(calc(nt-1,nt+v[i]-1,s[i&1])-calc(su,su+v[i],s[i&1])+mod)*vl)%mod;
				if(nt)dp[j+1][0]=(dp[j+1][0]+1ll*((nt-1)/s[i&1]-max(0ll,su/s[i&1]))*vl%mod*(v1-1))%mod;
				if(nt%s[i&1]==0&&nt){dp[j+1][1]=(dp[j+1][1]+1ll*vl*v1)%mod;if(j+1<n)v1=1ll*v1*(v[j+1]+1)%mod;}
				su=nt;
			}
			if(su%s[i&1]==0&&su)as=(as+1ll*vl*v1)%mod;
			for(set<int>::iterator it=fu.begin();it!=fu.end();it++)
			{
				ll lb=-v[i]+1,rb=-(*it),vl=dp[i][1],v1=1;
				if(lb>rb)continue;
				lb=rb-(rb-lb)/s[i&1]*s[i&1];vl=1ll*vl*((rb-lb)/s[i&1]+1)%mod;
				for(int j=i;j<=n;j+=2)
				{
					ll ls=(lb+v[j]-1)/s[i&1]-max(0ll,lb/s[i&1]),rs=(rb+v[j]-1)/s[i&1]-max(0ll,rb/s[i&1]);
					if(ls<0)ls=0;
					dp[j+1][0]=(dp[j+1][0]+1ll*1ll*(ls+rs)*(rs-ls+1)/2%mod*vl%mod*(v1-1))%mod;
					lb+=v[j],rb+=v[j];
					if(lb%s[i&1]==0){dp[j+1][1]=(dp[j+1][1]+1ll*vl*v1)%mod;if(j+1<n)v1=1ll*v1*(v[j+1]+1)%mod;}
				}
				if(lb%s[i&1]==0)as=(as+1ll*vl*v1)%mod;
			}
		}
		return as;
	}
};
```

##### SRM560 TC12294 BoundedOptimization

###### Problem

有 $n$ 个变量 $x_i$，给出 $n$ 组上下界限制 $l_i,r_i$，表示限制 $l_i\leq x_i\leq r_i$。

同时给出总上界限制 $lim$，表示要求 $\sum x_i\leq lim$。

给出 $m$ 个无序对 $(a_i,b_i)$，满足每一个无序对 $(a_i,b_i)$ 最多出现一次且 $a_i\neq b_i$。

你需要构造 $x$ 的值使得 $\sum_{i=1}^mx_{a_i}x_{b_i}$ 最大。输出最大值。

$n\leq 13$

###### Sol

考虑对一组解进行调整。在一组解中，如果存在 $i\neq j$ 使得 $x_i,x_j$ 都不等于上界或者下界，且优化项中不存在 $x_ix_j$。此时固定其它值不变，则在需要最大化的式子中， $x_i,x_j$ 的系数不会改变。

此时总上界限制相当于 $x_i+x_j\leq t$，最优化的式子形如 $ax_i+bx_j$，可以发现一定可以调整到一种不差的方案，使得 $x_i,x_j$ 中有一个达到上界或者下界。

因此如果将一个 $(x_i,x_j)$ 看成边，则此时不等于上界或下界的变量一定构成一个团。

此时再考虑需要最大化的式子中每个变量的系数，即图中所有和它相邻的变量的和。如果在团中有两个变量的系数不相等，则将系数大的增加 $\epsilon$，系数小的减小 $\epsilon$，此时可以发现答案一定会变优。因此，这个团中所有变量的系数一定相等。

又因为这部分是一个团，因此和这个点相邻的所有点为和这个点相邻的非团中的点和团中其它点。其中第一部分的变量值已经固定。如果减去团中所有点的权值和，则相当于这个变量相邻的非团中的变量的值的和减去这个变量的值是一个定值。

因而如果将某一个变量的值设为 $x$，则最优方案下其它团中变量的值都可以表示为 $x+c$ 的形式。此时 $x$ 显然越大越好，可以通过每个变量的限制和总上界求出 $x$ 是否存在以及上界。

因此只需要枚举团，再枚举剩下的每一个变量取上界还是下界，再做上面的过程即可。

复杂度 $O(n^23^n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<string>
using namespace std;
#define N 15
int n,lb[N],rb[N],su,is[N][N],vl[N],f1[N],v1[N];
double as,v2[N];
char s[740];
double solve(int s)
{
	for(int i=1;i<=n;i++)f1[i]=s%3,s/=3,vl[i]=v1[i]=0;
	for(int i=1;i<=n;i++)if(!f1[i])vl[i]=lb[i];else if(f1[i]==2)vl[i]=rb[i];
	for(int i=1;i<=n;i++)if(f1[i]==1)
	{
		for(int j=1;j<=n;j++)if(i!=j)
		if(f1[j]==1&&!is[i][j])return -1e9;
		else if(is[i][j])v1[i]+=vl[j];
	}
	int l1=-1e9,r1=1e9,s1=su,s2=0,ct=0;
	for(int i=1;i<=n;i++)if(f1[i]!=1)s1-=vl[i];
	else l1=max(l1,lb[i]-v1[i]),r1=min(r1,rb[i]-v1[i]),s2+=v1[i],ct++;
	double r2=min(1.0*r1,1.0*(s1-s2)/ct);
	if(r2+1e-9<l1)return -1e9;
	for(int i=1;i<=n;i++)if(f1[i]==1)v2[i]=v1[i]+r2;else v2[i]=vl[i];
	double as=0;
	for(int i=1;i<=n;i++)for(int j=i+1;j<=n;j++)if(is[i][j])as+=v2[i]*v2[j];
	return as;
}
struct BoundedOptimization{
	double maxValue(vector<string> sr,vector<int> l1,vector<int> r1,int s2)
	{
		n=l1.size();su=s2;
		int nw=0;
		for(int i=0;i<sr.size();i++)for(int j=0;j<sr[i].size();j++)s[++nw]=sr[i][j];
		for(int i=1;s[i];i+=3)is[s[i]-'a'+1][s[i+1]-'a'+1]=is[s[i+1]-'a'+1][s[i]-'a'+1]=1;
		for(int i=1;i<=n;i++)lb[i]=l1[i-1],rb[i]=r1[i-1];
		int tp=1;
		for(int i=1;i<=n;i++)tp*=3;
		for(int i=0;i<tp;i++)as=max(as,solve(i));
		return as;
	}
};
```

##### SRM713 TC14572 CoinsQuery

###### Problem

有 $n$ 种硬币，第 $i$ 种的硬币重量为 $w_i$，价值为 $v_i$。

给出 $q$ 次询问，每次给出 $W$，你需要选择一个硬币序列，每种硬币可以使用无限多个，使得序列中硬币的重量和正好为 $W$，且价值和最大。

求出最大的价值，以及达到最大价值的序列数。方案数对 $10^9+7$ 取模。

$n,w_i,q\leq 100,v_i,W\leq 10^9$

###### Sol

直接的暴力做法：设 $mx_i$ 表示重量和为 $i$ 的最大价值和，$dp_i$ 表示方案数，则：
$$
mx_i=\max_{j=1}^nmx_{i-w_j}+v_j\\
dp_i=\sum_{j=1}^n[mx_{i-w_j}+v_j=mx[i]]dp_{i-w_j}
$$
考虑 $mx$ 的性质，将硬币分为两类，第一类为 $\frac{v_i}{w_i}$ 最大的，第二类为其它的。

对于第一类，设这一类的 $\gcd w_i=g$，则容易发现，这一类硬币可以表示出所有大于 $w^2$ 且是 $g$ 的倍数的重量。

对于第二类，如果第二类硬币使用了超过 $g-1$ 个，考虑硬币的前缀重量和，由抽屉原理，一定存在两个前缀和模 $g$ 同余，设这一段的重量和为 $k*g$。此时，如果第一类使用的硬币重量和大于 $2w^2$，则此时一定存在一种使用硬币重量和等于之前的重量和减去 $kg$ 的方案。这样替换后，得到的价值和更优。

因此对于任意一种超过 $3w^2$ 的重量，此时第二类使用的硬币重量和不超过 $w^2$，因而第一种使用的重量和不少于 $2w^2$。因而存在一种让第一种的总重量增加 $g$ 的方案，也存在减少 $g$ 的方案。

因此，此时一定有 $mx_{i+g}=mx_i+\frac{v_ig}{w_i}$。因而这部分中 $dp$ 的转移也是每 $g$ 个一循环的。

因此可以先暴力做 $dp$，直到出现循环。将 $dp$ 转移看成矩阵，则可以求出循环节内的矩阵前缀积以及循环节乘积的 $2^0,...,2^{30}$ 次方，然后询问相当于求出一个向量乘上 $\log n$ 个矩阵后的结果。

复杂度 $O((w^3+qw^2)\log v)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 105
#define M 100500
#define ll long long
#define mod 1000000007
int n,q,s[N][2],a,k=1,vl[M],mx=1,f1[N][N][N],tp[33][N][N],v1[N],v2[N],is[M],fu;
ll dp[M];
struct CoinsQuery{
	vector<ll> query(vector<int> s0,vector<int> s1,vector<int> qu)
	{
		n=s0.size();q=qu.size();
		for(int i=1;i<=n;i++)s[i][0]=s0[i-1],s[i][1]=s1[i-1];
		for(int i=1;i<=n;i++)if(1ll*s[mx][1]*s[i][0]<1ll*s[mx][0]*s[i][1])mx=i;
		vl[0]=1;is[0]=1;
		fu=1;
		for(int i=1;i<=100;i++)
		{
			int fg=1;
			for(int j=1;j<=n;j++)if(s[j][0]%i!=0)fg=0;
			if(fg)fu=i;
		}
		while(1)
		{
			for(int i=1;i<=n;i++)if(s[i][0]<=k)
			{
				ll s1=dp[k-s[i][0]]+s[i][1],v1=vl[k-s[i][0]];
				if(!is[k-s[i][0]])continue;
				is[k]=1;
				if(s1>dp[k])dp[k]=s1,vl[k]=v1;
				else if(s1==dp[k])vl[k]=(vl[k]+v1)%mod;
			}
			int fg=k>s[mx][0]*2&&k>10000;
			if(fg)for(int i=0;i<=100;i++)if(is[k-i]&&dp[k-i]!=dp[k-i-s[mx][0]]+s[mx][1])fg=0;
			if(fg)break;else k++;
		}
		for(int t=k+1;t<=k+100;t++)
		for(int i=1;i<=n;i++)if(s[i][0]<=t)
		{
			ll s1=dp[t-s[i][0]]+s[i][1],v1=vl[t-s[i][0]];
			if(!is[t-s[i][0]])continue;
			is[t]=1;
			if(s1>dp[t])dp[t]=s1,vl[t]=v1;
			else if(s1==dp[t])vl[t]=(vl[t]+v1)%mod;
		}
		for(int i=0;i<100;i++)f1[0][i][i]=1;
		for(int i=1;i<=s[mx][0];i++)
		for(int j=0;j<100;j++)
		{
			for(int l=0;l<99;l++)f1[i][j][l+1]=f1[i-1][j][l];
			for(int l=1;l<=n;l++)if(dp[k+i]==dp[k+i-s[l][0]]+s[l][1])f1[i][j][0]=(f1[i][j][0]+f1[i-1][j][s[l][0]-1])%mod;
		}
		for(int i=0;i<100;i++)for(int j=0;j<100;j++)tp[0][i][j]=f1[s[mx][0]][i][j];
		for(int i=1;i<=30;i++)
		for(int j=0;j<100;j++)
		for(int p=0;p<100;p++)
		for(int q=0;q<100;q++)
		tp[i][j][p]=(tp[i][j][p]+1ll*tp[i-1][j][q]*tp[i-1][q][p])%mod;
		vector<ll> a1;
		for(int t=0;t<q;t++)
		{
			a=qu[t];
			if(a%fu){a1.push_back(-1);a1.push_back(-1);continue;}
			if(a<=k)
			{
				if(!is[a])a1.push_back(-1),a1.push_back(-1);
				else a1.push_back(dp[a]),a1.push_back(vl[a]);
			}
			else
			{
				for(int i=0;i<100;i++)v1[i]=v2[i]=0;
				a-=k;
				ll as1=dp[k+a%s[mx][0]];
				for(int i=0;i<100;i++)v1[i]=f1[a%s[mx][0]][i][0];
				a/=s[mx][0];as1+=1ll*a*s[mx][1];
				for(int i=0;i<=30;i++)if((a>>i)&1)
				{
					for(int p=0;p<100;p++)
					for(int q=0;q<100;q++)
					v2[p]=(v2[p]+1ll*v1[q]*tp[i][p][q])%mod;
					for(int p=0;p<100;p++)v1[p]=v2[p],v2[p]=0;
				}
				int as=0;
				for(int p=0;p<100;p++)as=(as+1ll*vl[k-p]*v1[p])%mod;
				a1.push_back(as1),a1.push_back(as);
			}
		}
		return a1;
	}
};
```

##### SRM526.5 TC11676 MagicMatchesGame

###### Problem

有 $n$ 个物品，每个物品有非负属性 $a_i,b_i,c_i$，你需要选出一些物品满足：

1. 对于选出的物品的 $a_i$ 构成的可重集，这个可重集的任意非空子集满足在子集中进行nim游戏先手必胜。
2. 在满足1的情况下，选出物品数量尽量多。
3. 在满足2的情况下，选出物品的 $(\sum b_i)*(\sum c_i)$ 尽量小。

求出最小的 $(\sum b_i)*(\sum c_i)$。

$n\leq 50,a_i\leq 10^6$

###### Sol

条件 $1$ 相当于选出的 $a_i$ 在异或意义下线性无关。因而这相当于一个Binary Matroid。因此在代价为一维的情况下，可以直接线性基求出最小代价。

对于二维的情况，将一组方案对应到一个点 $(\sum b_i,\sum c_i)$。设最优解为 $k$，可以发现 $xy=k$ 在 $x>0$ 时是一个上凸函数，因此在最优解 $(x,y)$ 点处的切线满足其余所有方案对应的点都在切线的严格上方。

因此一定存在一个斜率 $t$ ，使得最优解满足 $(\sum b_i)+t(\sum c_i)$ 最小。

对于一个固定的 $t$ ，这变成了一个一维情况，因此做法为按照代价排序从小到大选。因此直接的做法是枚举 $O(n^2)$ 个让相对顺序改变的 $t$，每次暴力求线性基，这样的复杂度为 $O(n^3\log v)$。

另外一个做法是，让 $t$ 从小到大变化，在这个过程中会出现 $O(n^2)$ 次交换两个相邻位置的相对顺序的事件。维护顺序改变时可以维护当前每一对顺序相邻的元素在什么时候会交换顺序，然后使用set维护这些时刻，每次找下一次交换的时刻即可。维护交换的复杂度为 $O(n^2\log n)$。

此时考虑交换两个元素加入顺序对线性基的影响。在加入这两个元素前以及加入后，当前加入向量组成的线性空间都是确定的，因此改变顺序不会影响之前和之后的元素是否会加入线性基，只可能这两个元素的情况出现改变。

因此可以将这两个元素从线性基中删除，然后按照改变后的顺序加入即可。

复杂度 $O(n^2(\log n+\log v))$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 105
#define ll long long
int n,v[N],s[N][2],id[N],tp[N];
ll s1[N],as;
bool cmp(int a,int b){return s1[a]<s1[b];}
bool ins(int v)
{
	for(int i=20;i>=0;i--)
	if((v>>i)&1)
	if(!tp[i]){tp[i]=v;return 1;}
	else v^=tp[i];
	return 0;
}
ll check(int v1,int v2)
{
	for(int i=1;i<=n;i++)id[i]=i,s1[i]=1ll*v1*s[i][0]*10000+1ll*v2*s[i][1]*10000+s[i][0];
	sort(id+1,id+n+1,cmp);
	int t1=0,t2=0;
	for(int i=0;i<=20;i++)tp[i]=0;
	for(int i=1;i<=n;i++)if(ins(v[id[i]]))t1+=s[id[i]][0],t2+=s[id[i]][1];
	return 1ll*t1*t2;
}
struct MagicMatchesGame{
	ll minimumArea(vector<int> v1,vector<int> s0,vector<int> s1)
	{
		n=v1.size();
		for(int i=1;i<=n;i++)v[i]=v1[i-1],s[i][0]=s0[i-1],s[i][1]=s1[i-1];
		as=min(check(1,0),check(0,1));
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(i!=j)as=min(as,check(s[i][1]-s[j][1],s[j][0]-s[i][0]));
		return as;
	}
};
```

##### SRM569 TC12389 MegaFactorial

###### Problem

定义 $n!k$ 为：

1. $n!k=1(n=0)$
2. $n!k=n(k=0)$
3. $n!k=n!(k-1)*(n-1)!k(n,k>0)$

给出 $n,k,b$ ，求出 $n!k$ 在 $b$ 进制下末尾 $0$ 的个数，模 $10^9+9$

$n\leq 10^9,k\leq 16,b\leq 10$

###### Sol

如果 $b$ 为质数 $p$ 的幂 $p^q$，则只需要求出 $n!k$ 质因数分解后 $p$ 的数量，再除以 $q$ 即为答案。

但如果 $b$ 是多个质数的幂相乘，则答案为每个质数幂的答案的最小值。由于计数取模的原因难以比较大小。

但这里 $b\leq 10$，只会出现 $b=p\times q$ 的情况，其中 $p,q$ 是两个质数。此时容易发现一定是较大的质数的答案较小。

因此问题变为求出 $n!k$ 质因数分解后 $p$ 的幂次除以 $q$ 下取整的结果，其中 $p\leq 7,q\leq 3$。

对于没有下取整的情况，设 $f_{n,k}$ 表示 $n!k$ 中 $p$ 的幂次，则有如下关系：
$$
f_{n,0}=\max_{p^a|n,a\in \N}a\\
f_{n,k}=f_{n-1,k}+f_{n,k-1}
$$
这里可以将 $\max_{p^a|n,a\in \N}a$ 拆成 $\sum_{i=1}^{+\infty}[p^i|n]$。考虑对于每一项单独算贡献。此时一项的贡献形如：
$$
f_{n,0}=[p^i|n]\\
f_{n,k}=f_{n-1,k}+f_{n,k-1}
$$
这个转移是每 $p^i$ 项循环的，可以使用类似矩阵快速幂的方式求出一段的转移。

设这一段为 $[l,r]$ ，由于转移的特殊性，可以发现 $dp_{l-1,x}$ 走到 $dp_{r,y}$ 的系数只和 $y-x$ 有关。因此可以只记录矩阵的一维，这样一次乘法的复杂度为 $O(k^2)$。

然后再做一次快速幂，再乘上最后一小段，即可得到 $f_n$。这样的复杂度为 $O(k^2\log n)$

因此总的复杂度为 $O(k^2\log^2 n)$

对于下取整的情况，可以发现上面的做法不需要求逆元，因此可以求答案模 $q$ 的结果。因此求出 $ans=(ans\bmod q))/q$ 即可。

复杂度 $O(k^2\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define ll long long
int tp[11]={0,1,2,3,4,5,3,7,8,9,5};
vector<int> sr;
struct sth{vector<int> a,b;};
vector<int> sol1(ll n,int k,int mod)
{
	if(n<=1)
	{
		vector<int> as;
		for(int i=0;i<=k;i++)as.push_back(i<=n);
		return as;
	}
	vector<int> s1=sol1(n/2,k,mod);
	vector<int> s2;
	for(int i=0;i<=k;i++)s2.push_back(0);
	for(int i=0;i<=k;i++)for(int j=0;i+j<=k;j++)s2[i+j]=(s2[i+j]+1ll*s1[i]*s1[j])%mod;
	if(n&1)for(int i=k;i>=1;i--)s2[i]=(s2[i]+s2[i-1])%mod;
	return s2;
}
sth sol2(ll n,int k,int mod)
{
	vector<int> s2,s4;
	for(int i=0;i<=k;i++)s2.push_back(0),s4.push_back(0);
	if(n==1){s2[0]=1;return (sth){s2,sr};}
	sth s1=sol2(n/2,k,mod);s2=s1.a;
	for(int i=0;i<=k;i++)for(int j=0;i+j<=k;j++)s2[i+j]=(s2[i+j]+1ll*s1.a[i]*s1.b[j])%mod;
	for(int i=0;i<=k;i++)for(int j=0;i+j<=k;j++)s4[i+j]=(s4[i+j]+1ll*s1.b[i]*s1.b[j])%mod;
	if(n&1)
	{
		vector<int> s3,s5;
		for(int i=0;i<=k;i++)s3.push_back(0),s5.push_back(0);s3[0]=1;
		for(int i=0;i<=k;i++)for(int j=0;i+j<=k;j++)s3[i+j]=(s3[i+j]+1ll*s2[i]*sr[j])%mod;
		for(int i=0;i<=k;i++)for(int j=0;i+j<=k;j++)s5[i+j]=(s5[i+j]+1ll*s4[i]*sr[j])%mod;
		s2=s3;s4=s5;
	}
	return (sth){s2,s4};
}
int doit1(ll n,int k,int p,int mod)
{
	ll tp=p,as=0;
	while(tp<=n)
	{
		sr=sol1(tp,k,mod);
		vector<int> s2=sol2(n/tp,k,mod).a,s1=sol1(n%tp+k-1,k,mod);
		for(int i=0;i<k;i++)as=(as+1ll*s1[i]*s2[k-1-i])%mod;
		tp*=p;
	}
	return as;
}
int solve(ll n,int k,int d)
{
	int p=0,ct=0,mod=1e9+9;
	for(int i=2;i<=d;i++)if(d%i==0){p=i;break;}
	while(d%p==0)ct++,d/=p;
	int as1=doit1(n,k,p,mod),as2=doit1(n,k,p,ct);
	as1=(as1+mod-as2)%mod;
	return 1ll*as1*((ct-1)*mod+1)/ct%mod;
}
int main()
{
	ll n;
	int k,p;
	scanf("%lld%d%d",&n,&k,&p);
	printf("%d\n",solve(n,k,tp[p]));
}
```

##### SRM452 TC10566 IncreasingNumber

###### Problem

求出满足下列条件的 $n$ 位十进制数数量，模 $10^9+7$：

1. 没有前导 $0$，且十进制表示从左到右不降。
2. 可以被 $m$ 整除。

$n\leq 10^{18},m\leq 500$

###### Sol

记数的数位表示为 $s_{n-1}s_{n-2}...s_1s_0$。

记 $t_i=\max_{s_k\geq i}k$，则因为十进制表示不降，因而 $s_k\geq i$ 当且仅当 $k\leq t_i$。

因此可以将数写成 $\sum_{i=0}^{n-1}\sum_{j=1}^9[s_i\geq j]10^i$，进一步可以写成 $\sum_{j=1}^9\sum_{i=0}^{t_j}10^i$。

因此现在问题可以看成每个 $[0,n]$ 间的数 $x$ 有一个权值 $\frac19(10^{x+1}-1)$，你需要选择 $0\leq t_9\leq t_8\leq...\leq t_1=n-1$，使得这些数的权值和是 $m$ 的倍数。

可以通过容斥将 $t_1=n-1$ 的限制变成 $t_1\leq n-1$ 的限制。考虑后者如何处理。此时可以看成在 $[0,n-1]$ 中选择 $9$ 个无序元素的方案。

记 $v_i=\sum_{j=0}^i10^j$， $c_i$ 表示 $[0,n-1]$ 中满足 $v_k\equiv i(\bmod m)$ 的 $k$ 个数。

对于计算 $c_i$ 的部分，注意到 $v_i=10v_{i-1}+1$，这个操作在模 $p$ 意义下最后会进入循环，进入循环后求出循环节，即可处理剩余的部分，这里可以 $O(m)$ 求出。

此时考虑最后选择了 $a$ 个满足 $v_k\equiv i(\bmod m)$ 的元素的方案数。此时所有元素无序，可以发现方案数为 $C_{c_i+a-1}^{a}$。

因此设 $dp_{i,j,k}$ 表示考虑了 $v_k\equiv 0,...,i(\bmod m)$ 的所有数，当前前面的 $\sum v_i\bmod m$ 为 $j$，前面选了 $k$ 个数的方案数。转移时枚举下一种选几个即可。

复杂度 $O(m^2*10^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505
#define ll long long
#define mod 1000000007
ll n,su[N];
int p,dp[N][N][10],inv[10],st[N];
int solve(int p)
{
	for(int i=1;i<=9;i++)for(int j=1;j<=i;j++)if((1ll*mod*j+1)%i==0)inv[i]=(1ll*mod*j+1)/i;
	for(int i=1;i<=p;i++)for(int j=0;j<p;j++)for(int k=0;k<=9;k++)dp[i][j][k]=0;
	dp[0][0][0]=1;
	for(int i=0;i<p;i++)
	for(int j=0;j<=9;j++)
	{
		int vl=1,tp=i*j%p;
		for(int k=1;k<=j;k++)vl=1ll*(su[i]+k-1)%mod*inv[k]%mod*vl%mod;
		for(int k=0;k<p-tp;k++)for(int l=0;l+j<=9;l++)dp[i+1][k+tp][l+j]=(dp[i+1][k+tp][l+j]+1ll*dp[i][k][l]*vl)%mod;
		for(int k=p-tp;k<p;k++)for(int l=0;l+j<=9;l++)dp[i+1][k+tp-p][l+j]=(dp[i+1][k+tp-p][l+j]+1ll*dp[i][k][l]*vl)%mod;
	}
	int as=0;
	for(int i=0;i<=9;i++)as=(as+dp[p][0][i])%mod;
	return as;
}
int solve1(ll n,int p)
{
	for(int i=0;i<p;i++)su[i]=0;
	int nw=1%p;
	while(n)
	{
		su[nw]++;nw=(nw*10+1)%p;n--;
		if(su[nw])break;
	}
	int ct=0;
	if(n)
	{
		for(int i=nw;i!=nw||!ct;i=(i*10+1)%p)st[++ct]=i;
		for(int i=1;i<=ct;i++)su[st[i]]+=n/ct+(n%ct>=i);
	}
	return solve(p);
}
struct IncreasingNumber{
	int countNumbers(ll n,int p){return (mod+solve1(n,p)-solve1(n-1,p))%mod;}
};
```

##### SRM562 TC12304 InducedSubgraphs

###### Problem

给定 $n,k$ 以及一棵 $n$ 个点的树，称一个 $n$ 阶排列 $p$ 是好的，当且仅当它满足如下性质：

 $\forall 1\leq i\leq n-k+1$，树中 $p_i,p_{i+1},...,p_{i+k-1}$ 构成一个连通块。

求好的排列数量，模 $10^9+7$。

$n\leq 40$

###### Sol

对 $k$ 分类讨论：

1. $k=1$

此时相当于没有限制，因此答案为 $n!$

2. $k\leq \frac n2$

考虑 $p_1,...,p_k$ 的连通块以及 $p_{n-k+1},...,p_n$ 的连通块。考虑 $i$ 增加的过程，相当于每次从第一个连通块中删去最早被加入的点，再加入一个没有被加入的点，且每个时刻都是一个连通块。

此时有如下结论：

1. 删去 $p_1,...,p_k$ 后，剩余部分连通。

如果不连通，则一定存在一个 $i>k$，使得 $p_i,...,p_{i+k-1}$ 中有来着两个连通块的点，此时不满足条件。

同理删去 $p_{n-k+1},...,p_n$ 后树也应该连通。因而这两部分分别是某一个子树。

2. 剩余部分（ $p_{k+1},...,p_{n-k}$ ）中不存在一个点度数大于等于 $3$。

如果存在，则最多只有一个与它相邻的点在排列中比它先出现，否则 $i+k-1$ 小于它时这里不连通。此时有两个与它相邻的点在排列中比它后出现。因而 $i$ 大于它时不连通。因此这种情况不存在。

因而此时的树一定形如两个大小为 $k$ 的子树被一条长度为 $n-2k$ 的链连接。

考虑此时的方案数，在前 $k$ 个点中，令向外连接的点为根，则必定有父亲在排列中后于儿子，否则删去父亲后不连通。因而此时设 $dp_u$ 表示 $u$ 子树内排列的合法方案数，可以 $O(n)$ 求出。在后 $k$ 个点中，有相同的情况。

对于中间的链，可以发现这部分方案唯一。

因此可以枚举两个子树，然后判断是否合法，再 $dp$ 算子树内的方案数。复杂度 $O(n^3)$

3. $k>\frac n2$

此时 $p_{n-k+1},...,p_k$ 在所有考虑的连通块中都出现过。此时有：

1. 这些点构成一个连通块。

如果构成多个连通块，则有一个时刻连接它们的点不在当前考虑的 $p_i,...,p_{i+k-1}$ 中，此时这些点不连通。

考虑剩下的情况，此时可以看成，剩下的点有 $n-k$ 个属于初始的连通块，有 $n-k$ 个属于结束的连通块。每次需要在初始连通块中删一个点，再加入一个结束连通块的点。

将 $p_i,...,p_{i+k-1}$ 构成的连通块看作根，则此时剩下的每一个子树内部一定全部属于一种连通块。因而这个子树内的顺序情况和上一种情况中的 $dp$ 相同。

此时考虑设 $dp_{u,p,q}$ 表示考虑以 $u$ 为根的子树，$u$ 属于 $p_i,...,p_{i+k-1}$ ，此时 $u$ 子树内有 $p$ 个点属于初始连通块，有 $q$ 个点属于结束连通块，剩余点属于 $p_i,...,p_{i+k-1}$ 的方案数。

转移时可以枚举下一个子树内的情况，分成子树内的根属于 $p_i,...,p_{i+k-1}$ 和不属于的情况即可。

这样一次 $dp$ 的复杂度上限为 $O(n^4)$ ，且常数很小。

这样一次 $dp$ 可以求出所有满足 $p_i,...,p_{i+k-1}$ 包含根的方案数，因此考虑点减边，对于每个点为根做一次，再减去对每条边做一次的结果即可。

复杂度 $O(n^5)$，常数非常小。

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<vector>
using namespace std;
#define N 55
#define mod 1000000009
int n,k,a,b,head[N],cnt,c[N][N],f[N],as;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int dp[N],sz[N];
void dfs1(int u,int fa)
{
	dp[u]=1;f[u]=fa;sz[u]=0;
	int s1=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs1(ed[i].t,u);
		sz[u]+=sz[ed[i].t];
		if(sz[ed[i].t]<n/2)s1+=sz[ed[i].t],dp[u]=1ll*dp[u]*dp[ed[i].t]%mod*c[s1][sz[ed[i].t]]%mod;
	}
	sz[u]++;
}
int dp2[N][N][N];
void dfs2(int u,int fa)
{
	dp2[u][0][0]=1;
	int sr=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs2(ed[i].t,u);
		dp2[ed[i].t][0][sz[ed[i].t]]=(dp2[ed[i].t][0][sz[ed[i].t]]+dp[ed[i].t])%mod;
		dp2[ed[i].t][sz[ed[i].t]][0]=(dp2[ed[i].t][sz[ed[i].t]][0]+dp[ed[i].t])%mod;
		for(int j=sr;j>=0;j--)
		for(int k=sr-j;k>=0;k--)if(dp2[u][j][k])
		for(int p=sz[ed[i].t];p>=0;p--)
		for(int q=sz[ed[i].t]-p;q>=0;q--)if(p||q)
		dp2[u][j+p][k+q]=(dp2[u][j+p][k+q]+1ll*dp2[u][j][k]*dp2[ed[i].t][p][q]%mod*c[j+p][p]%mod*c[k+q][q])%mod;
		sr+=sz[ed[i].t];
	}
}
struct InducedSubgraphs{
	int getCount(vector<int> s0,vector<int> s1,int k)
	{
		n=s0.size()+1;
		if(k==1)k=n;
		for(int i=1;i<n;i++)adde(s0[i-1]+1,s1[i-1]+1);
		for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
		for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
		if(k*2<=n)
		{
			for(int i=1;i<=n;i++)
			{
				dfs1(i,0);
				for(int j=1;j<=n;j++)if(sz[j]==k)
				{
					int nw=j;
					for(int t=1;t<=n-k*2;t++)nw=f[nw];
					if(sz[nw]==n-k&&f[nw]==i)as=(as+1ll*dp[i]*dp[j])%mod;
				}
			}
		}
		else
		{
			for(int i=1;i<=n;i++)
			{
				dfs1(i,0);
				memset(dp2,0,sizeof(dp2));
				dfs2(i,0);
				as=(as+dp2[i][n-k][n-k])%mod;
			}
			for(int i=1;i<=n;i++)for(int j=head[i];j;j=ed[j].next)if(ed[j].t<i)
			{
				int f=i,t=ed[j].t;
				memset(dp2,0,sizeof(dp2));
				dfs1(f,t);dfs1(t,f);
				dfs2(f,t);dfs2(t,f);
				for(int p=0;p<=n-k;p++)
				for(int q=0;q<=n-k;q++)
				as=(as+mod-1ll*dp2[f][p][q]*dp2[t][n-k-p][n-k-q]%mod*c[n-k][p]%mod*c[n-k][q]%mod)%mod;
			}
			for(int i=1;i<=k*2-n;i++)as=1ll*as*i%mod;
		}
		return as;
	}
};
```

##### SRM231 TC3942 Mixture

###### Problem

给定 $n$ 个 $m$ 维向量 $V_i=(v_{i,1},...,v_{i,n})$，每个向量有一个代价 $c_i$。

再给定一个 $m$ 维向量 $S$，你需要找到一组非负实数 $s_1,...,s_n$，满足 $S=s_1V_1+...+s_nV_n$，且最小化 $\sum s_ic_i$。

求最小代价，或者输出无解。

$n,m,v\leq 10$，所有数都是非负整数。

###### Sol

问题相当于如下线性规划：

最小化 $\sum s_ic_i$

满足 $\sum_i s_iv_{i,j}=S_j$

考虑对偶，可以得到如下形式：

最大化 $\sum S_it_i$

满足 $\sum_j t_jv_{i,j}\leq c_i$

且变量 $t_i$ 没有非负的限制。

此时可以直接不需要初始化的线性规划求解，根据对偶性质如果这个线性规划无界，则原线性规划无解。

为了避免精度问题，可以 $2^n$ 枚举每个变量的符号，然后做线性规划，这样只有 $10\times 10$ 的矩阵。然后拿long double就可以正好卡过去

复杂度不会算。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<string>
using namespace std;
#define N 15
int n,m,v1[N],s1[N][N];
double as=-1e18;
long double v[N],s[N][N];
double solve()
{
	while(1)
	{
		int fr=0;
		for(int i=1;i<=m;i++)if(v[i]>v[fr])fr=i;
		if(v[fr]<=1e-14)return -v[m+1];
		long double rb=1e20;
		int tp=0;
		for(int i=1;i<=n;i++)if(s[i][fr]>1e-12)
		{
			double r1=s[i][m+1]/s[i][fr];
			if(r1<rb)rb=r1,tp=i;
		}
		if(rb>1e19)return 1e18;
		long double t1=1/s[tp][fr];s[tp][fr]=1;
		for(int i=1;i<=m+1;i++)s[tp][i]*=t1;
		for(int i=1;i<=n;i++)if(i!=tp)
		{
			long double v2=-s[i][fr];s[i][fr]=0;
			for(int j=1;j<=m+1;j++)s[i][j]+=v2*s[tp][j];
		}
		long double v2=-v[fr];v[fr]=0;
		for(int j=1;j<=m+1;j++)v[j]+=v2*s[tp][j];
	}
}
struct Mixture{
	double mix(vector<int> vl,vector<string> sr)
	{
		m=vl.size();n=sr.size();
		for(int i=1;i<=m;i++)v1[i]=vl[i-1];
		for(int i=1;i<=n;i++)
		{
			int nw=0,ct=0;
			for(int j=0;j<sr[i-1].size();j++)
			{
				char st=sr[i-1][j];
				if(st==' ')s1[i][++ct]=nw,nw=0;
				else nw=nw*10+st-'0';
			}
			s1[i][++ct]=nw;
		}
		for(int i=0;i<1<<m;i++)
		{
			for(int j=1;j<=m+1;j++)
			{
				int fg=(i>>j-1)&1?-1:1;
				v[j]=v1[j]*fg;
				for(int k=1;k<=n;k++)s[k][j]=s1[k][j]*fg;
			}
			v[m+1]=0;
			double s1=solve();
			if(s1>as)as=s1;
		}
		if(as>1e17)as=-1;
		return as;
	}
};
```

##### SRM660 TC13696 Morphling

###### Problem

给定 $n,k$，考虑所有长度为 $n$，元素为 $[1,n]$ 间正整数且每种数出现不超过 $k$ 次的序列。称这些序列构成的集合为 $S$。

现在可以对一个这样的序列进行变换操作，一次变换操作为，指定两个 $[1,n]$ 间的不同整数 $a,b$，随后对序列 $s$ 依次执行如下操作：

1. 交换 $s_a,s_b$
2. 考虑序列中的每个位置，如果这个位置在这一步操作前为 $a$，则将其变为 $b$。如果这个位置在这一步操作前为 $b$，则将其变为 $a$。否则改变这个位置。

可以发现，对一个属于 $S$ 的序列任意进行变换操作，得到的序列也属于 $S$。

你希望找到一个 $S$ 的子集 $T$ ，使得对于 $S$ 中的任意一个序列 $s$，都存在一个 $T$ 中序列 $t$ ，使得 $t$ 可以通过若干次变换操作变为 $s$。在此基础上你希望 $|T|$ 尽量小。

求出最小的 $|T|$，答案对 $10^9+7$ 取模。

$n\leq 100,k\leq 25$

$2s,64MB$

###### Sol

操作相当于在下标上将 $a,b$ 交换，同时在值上将 $a,b$ 交换。

排列可以由若干次对换得到，因此进行若干次操作后的情况相当于给一个 $n$ 阶排列 $p$，若变换前的数列满足 $s_i=j$，则变换后 $s_{p_i}=p_j$。

可以发现在变换意义下每种本质不同的序列都必须有一个在 $T$ 中，因而可以发现最小的 $|T|$ 即为在给出的变换下本质不同的序列数量。

可能的变换即为所有的排列，共有 $n!$ 个。由Burnside引理，本质不同序列数量即为每种变换下的不动点数量之和除以 $n!$。

考虑一个排列 $p$，以及置换下的一个环 $(d_1,...,d_l)$，环上位置的值在变换前为 $s_{d_1},...,s_{d_l}$。

在下标变换后，环上位置的值为 $s_{d_2},s_{d_3},...,s_{d_l},s_{d_1}$。如果 $s$ 是一个不动点，则在值变换后，这些位置的值应该被还原，因而 $p_{s_{d_2}}=s_{d_1},...,p_{s_{d_l}}=p_{s_{d_1}},p_{s_{d_1}}=s_{d_l}$

此时可以发现 $s_{d_1},...,s_{d_l}$ 在同一个置换环中。此时这个环的长度必须是 $l$ 的因子。同时，如果这个环长度是 $l'$，则有 $l'$ 种方案。且可以发现每种环上的数出现 $\frac{l}{l'}$ 次。

因此，对于一个排列，设它对应的置换中所有的环长度为 $l_1,...,l_k$，则一种合法的方案相当于：

对于每个环 $l_i$，环上位置的值对应一个环 $l_j$，满足 $l_j|l_i$，有 $l_j$ 种方案，且环 $l_j$ 上的数出现了 $\frac{l_i}{l_j}$ 次。

最后要求每个环上的数出现次数不超过 $k$ 次。

如果把上面的关系看出 $l_i$ 向 $l_j$ 连一条有向边，则因为每个点只有一条出边，这构成了一棵基环内向树。对于所有排列求不动点数量和可以看成求构造这个内向树的方案数。

考虑dp，首先考虑树上的情况，设 $dp_{i,j,l}$ 表示当前根节点的环长为 $i$，子树内的环长和为 $j$，当前根节点的值出现次数为 $l$ 的方案数（还没有决定根节点的环的值）。

再记 $f_{i,j}$ 表示根节点的环长为 $i$，子树内的环长和为 $j$，当前根节点的值出现次数小于等于 $k$ 的方案数。枚举下一个儿子转移。因为儿子无序，因此可以钦定编号最大的点在当前儿子中，因此编号排列的系数为 $C_{b-1}^{a-1}$。但根与儿子间有区别，因此可以先不考虑根的编号排列的情况，最后乘上 $C_j^i$。
$$
dp_{i,i,0}=(n-1)!\\
dp_{i,j,l}=\sum_{i|a}\sum_{a|b,b\leq j-i}C_{j-i-1}^{b-1}f_{a,b}*dp_{i,j-b,l-\frac ai}*i
$$
最后再考虑基环树上的环的情况，相当于若干个子树连成一个环，可以发现这上面的环长度相等。记 $g_{i,j}$ 表示表示根节点的环长为 $i$，子树内的环长和为 $j$，当前根节点的值出现次数小于等于 $k-1$ 的方案数。

记 $h_{i,j,l}$ 表示选出 $l$ 个根节点环长为 $i$ 的子树，满足 $g$ 中性质的方案数，直接暴力做转移即可。最后对答案的贡献为：
$$
\sum_lh_{i,n,l}*(l-1)!
$$
因为父亲的环长度是儿子环长度的约数，因此上面的状态中只有 $i|j$ 的状态有用。

经过分析总复杂度为 $O(n^2k\log n+n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define mod 1000000007
int c[N][N],f[N][N],g[N][N],s1[N],h[N][N],v[N][N],su[N],dp[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct Morphling{
	int findsz(int n,int k)
	{
		for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
		for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
		for(int i=n;i>=1;i--)
		{
			for(int j=0;j<=n;j++)for(int l=0;l<=n;l++)g[j][l]=h[j][l]=0;
			for(int j=0;j<=n;j++)s1[j]=0;
			g[1][0]=1;for(int j=1;j<i;j++)g[1][0]=1ll*g[1][0]*j%mod;
			for(int j=1;i*j<=n;j++)
			for(int l=0;l<=k;l++)
			{
				for(int p=1;p<=l;p++)if(p*i<=n)
				for(int q=1;q*p<j;q++)
				g[j][l]=(g[j][l]+1ll*f[p*i][q]*g[j-q*p][l-p]%mod*i%mod*c[(j-1)*i-1][q*p*i-1])%mod;
				int tp=1ll*g[j][l]*c[j*i][i]%mod;
				f[i][j]=(f[i][j]+tp)%mod;
				if(l<k)s1[j]=(s1[j]+1ll*tp*i)%mod;
			}
			h[0][0]=1;
			for(int j=1;j<=n;j++)if(i*j<=n)
			for(int k=1;k<=j;k++)
			for(int p=1;p<=j;p++)
			h[j][k]=(h[j][k]+1ll*h[j-p][k-1]*(k>1?k-1:1)%mod*s1[p]%mod*c[j*i-1][p*i-1])%mod;
			for(int j=1;i*j<=n;j++)for(int k=1;k<=n;k++)su[i*j]=(su[i*j]+h[j][k])%mod;
		}
		dp[0]=1;
		for(int i=1;i<=n;i++)for(int j=0;j<i;j++)dp[i]=(dp[i]+1ll*dp[j]*su[i-j]%mod*c[i-1][i-j-1])%mod;
		int as=dp[n];
		for(int i=1;i<=n;i++)as=1ll*as*pw(i,mod-2)%mod;
		return as;
	}
};
```

##### SRM556 TC12144 OldBridges

###### Problem

给一张 $n$ 个点 $m$ 条边的无向图。其中一些边可以任意经过，一些边只能被经过两次。

你希望从 $s_1$ 到达 $t_1$ 再回到 $s_1$，这样重复 $c_1$ 次。接着从 $s_2$ 到达 $t_2$ 再回到 $s_2$，重复 $c_2$ 次。

求是否可以达到目标。

$n\leq 50$，所有起点终点两两不同。

###### Sol

记 $mincut(S,T)$ 表示集合 $S$ 到集合 $T$ 的最小割。则显然需要满足下面的限制：

1. $mincut(\{s_1\},\{t_1\})\geq 2c_1$
2. $mincut(\{s_2\},\{t_2\})\geq 2c_2$
3. $mincut(\{s_1,s_2\},\{t_1,t_2\})\geq 2c_1+2c_2$

但此时仍然不够，例如一个H的形状，不能同时从左上到右下，右上到左下。

因而可以发现还有一个限制：

4. $mincut(\{s_1,t_2\},\{s_2,t_1\})\geq 2c_1+2c_2$

可以猜想满足这四个条件的一定有解。

一个极其不冷静的证明：

考虑 $n=4$ 的情况，如果存在 $s_1-t_1$ 的边，可以优先流这条边。如果这条边没有被流完，则此时只剩另外一个方向的边，此时可以发现结论成立当且仅当满足上面四个条件，且此时优先流 $s_1-t_1$ 确实是最优的。

否则，情况可以看成 $s_1,s_2,t_1,t_2$ 的一个环，设四条边流量为 $v_1,v_2,v_3,v_4$。

再设 $s_1-s_2-t_1$ 的流量为 $l_1$，经过 $t_2$ 的流量为 $l_2$，则此时 $s_2-t_2$ 还能有的最大流量为：
$$
\min(v_1-l_1,v_4-l_2)+\min(v_2-l_1,v_3-l_2)
$$
令 $l_1+l_2=S_1,l_1-l_2=t$，可以发现上述变量都是偶数。此时上式相当于：
$$
\min(v_1-\frac{S_1}2-\frac t2,v_4-\frac{S_1}2+\frac t2)\\
+\min(v_2-\frac{S_1}2-\frac t2,v_3-\frac{S_1}2+\frac t2)
$$
此时可以发现 $t\in[\min(v_2-v_3,v_1-v_4),\max(v_2-v_3,v_1-v_4)]$ 时最优。再结合 $t\in [\max(-S_1,S_1-2\min(v_3,v_4)),\min(S_1,2\min(v_1,v_2)-S_1)]$。

可以发现 $S_1$ 最大时， $t$ 可以在的值 $\min(v_1,v_2)-\min(v_3,v_4)$ 属于最优值。因此可以分成以下几类讨论：

1. $0\in [\min(v_2-v_3,v_1-v_4),\max(v_2-v_3,v_1-v_4)]$

不妨设 $v_1\geq v_4,v_3\geq v_2$。则此时可以发现每个 $S_1$ 都可以让 $t$ 取一个最优值。因此限制变为：
$$
S_2\leq \min(v_1+v_4,v_2+v_3)-S_1
$$
即 $S_1+S_2\leq v_2+v_4$。这相当于第三条限制。可以发现如果满足上面的限制，则第三条限制是最强的限制。

如果反过来，则变成第四条限制。

2. 其它情况。

不妨设 $\min(v_2-v_3,v_1-v_4)=k>0$，则当 $S_1\leq k$ 时，由于上式是一个凸函数，可能的最优 $t$ 为 $S_1$，因而相当于：
$$
S_2\leq \min(v_1-S_1,v_4)+\min(v_2-S_2,v_3)\\
S_2\leq v_3+v_4
$$
这相当于第二条限制，此时暗含第一条限制。可以发现此时一定满足另外的两条限制。

否则，可以取到全局最优 $t$ ，此时相当于：
$$
S_2\leq\min(v_1-\frac{S_1}2-\frac k2,v_4-\frac{S_1}2+\frac k2)\\
+\min(v_2-\frac{S_1}2-\frac k2,v_3-\frac{S_1}2+\frac k2)\\
S_1+S_2\leq v_3+v_4+k=v_3+v_4\min(v_2-v_3,v_1-v_4)\\
S_1+S_2\leq \min(v_1+v_3,v_2+v_4)
$$
这相当于限制三+限制四。又因为 $S_1>k$，可以验证满足限制二。显然满足限制一。

因此对于 $n=4$ 的情况，上面四个限制为充分必要条件。

对于任意情况，考虑 $\{s_1,s_2\},\{t_1,t_2\}$ 的最小割和 $\{s_1,t_2\},\{s_2,t_1\}$ 的最小割：

![](556-1.png)



此时由割的性质，可以认为每一部分内是一个连通块。因此可以看成将每一部分看成点，变成 $n=4$ 的情况。可以得到这时的一个解。考虑在每一块内部补全剩下的情况。如果无解，则相当于可以找到一个小于需要的流量的割。此时讨论所有情况可以说明，如果存在这些情况，则之前选的割不是最小的割，因此不存在这种情况。（具体证明咕了）。

因此，上面的结论对于任意图成立。

因此只需要四次最大流即可。

复杂度 $O(dinic(n,n^2))$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
#include<vector>
#include<string>
using namespace std;
#define N 55
int n,head[N],cnt,dis[N],cur[N],s1,t1,v1,s2,t2,v2;
char s[N][N];
struct edge{int t,next,v;}ed[N*N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
bool bfs(int s,int t)
{
	for(int i=1;i<=n+2;i++)dis[i]=-1,cur[i]=head[i];
	queue<int> qu;
	qu.push(s);dis[s]=0;
	while(!qu.empty())
	{
		int u=qu.front();qu.pop();
		for(int i=head[u];i;i=ed[i].next)if(dis[ed[i].t]==-1&&ed[i].v)
		dis[ed[i].t]=dis[u]+1,qu.push(ed[i].t);
	}
	return dis[t]!=-1;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)
	if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
int dinic(int s,int t)
{
	int as=0;
	while(bfs(s,t)&&as<1e7)as+=dfs(s,t,1e7);
	return as;
}
void init()
{
	for(int i=1;i<=n+2;i++)head[i]=0;cnt=1;
	for(int i=1;i<=n;i++)
	for(int j=i+1;j<=n;j++)
	if(s[i][j]=='N')adde(i,j,1e8);
	else if(s[i][j]=='O')adde(i,j,2);
}
int solve(int s1,int s2,int t1,int t2)
{
	init();
	adde(n+1,s1,1e8);if(s2)adde(n+1,s2,1e8);
	adde(t1,n+2,1e8);if(t2)adde(t2,n+2,1e8);
	return dinic(n+1,n+2);
}
struct OldBridges{
	string isPossible(vector<string> mp,int s1,int t1,int v1,int s2,int t2,int v2)
	{
		n=mp.size();
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=mp[i-1][j-1];
		s1++;s2++;t1++;t2++;
		int fg=solve(s1,0,t1,0)>=v1*2&&solve(s2,0,t2,0)>=v2*2&&solve(s1,s2,t1,t2)>=(v1+v2)*2&&solve(s1,t2,s2,t1)>=(v1+v2)*2;
		return fg?"Yes":"No";
	}
};
```

##### TCO13 Semi2 TC12844 OneBlack

###### Problem

给一个 $n\times m$ 的网格，其中有些格子是障碍。

你需要将一些非障碍格子染黑，使得对于任意一条从左上角出发，只向右向下走，到达右下角的路径都正好经过一个黑色格子。

求方案数，模 $10^9+7$。

$n,m\leq 30$

###### Sol

考虑将所有不在任何一条合法路径中的格子看成障碍，每这样处理一个格子会使答案乘 $2$。这一部分可以 $O(n^2)$ 解决。

此时剩余的图是一个边有向的平面图，你需要选一些点满足给定条件。

可以发现选的点满足构成类似极大反链的形式，只是边变成了点。

对于平面图中的一个面，设这个面的上边界为 $s,x_1,...,x_a,t$，下边界为 $s,y_1,...,y_b,t$。注意到原图是由网格图删点得到的，因此可以发现上边界和下边界中一定至少有一个点，即 $a,b\geq 1$。

因此在这里可以使用与边的极大反链相同的做法，将一个面看成一个点，对于原图的一个点，从它下方的面向上方的面连边。

经过开始的处理后，此时从起点出发可以到达任意一个点，从任意一个点出发能到达终点。此时可以证明从下方外部到上方外部的路径与极长反链一一对应。因此求DAG路径计数即可。

在本题中，可以发现一个面即为一些八连通的障碍的外轮廓。因此可以求出所有八连通的障碍，然后即可得到平面图的信息，此时直接建上面类似对偶图的结构即可。

复杂度 $O(nm)$

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<vector>
#include<string>
using namespace std;
#define N 33
#define M 1050
#define mod 1000000007
int n,m,as=1,fg[N][N],fr[N][N],ct,d[8][2]={-1,0,1,0,0,-1,0,1,-1,1,-1,-1,1,-1,1,1},head[M],cnt,in[M],dp[M],f1[M];
char s[N][N];
struct edge{int t,next;}ed[M*4];
void adde(int f,int t){in[t]++;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void dfs(int x,int y)
{
	for(int t=0;t<8;t++)
	{
		int nx=x+d[t][0],ny=y+d[t][1];
		if(s[nx][ny]==s[x][y]&&!fr[nx][ny])fr[nx][ny]=fr[x][y],dfs(nx,ny);
	}
}
struct OneBlack{
	int countColorings(vector<string> s1)
	{
		n=s1.size();m=s1[0].size();
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)s[i][j]=s1[i-1][j-1];
		fg[1][1]=1;
		for(int i=1;i<=n;i++)
		for(int j=1;j<=m;j++)
		fg[i][j]=s[i][j]=='-'&&(fg[i-1][j]||fg[i][j-1]||fg[i][j]);
		for(int i=1;i<=n;i++)
		for(int j=1;j<=m;j++)
		if(fg[i][j])fg[i][j]=0;
		else if(s[i][j]=='-')as=2ll*as%mod,s[i][j]='#';
		fg[n][m]=1;
		for(int i=n;i>=1;i--)
		for(int j=m;j>=1;j--)
		fg[i][j]=s[i][j]=='-'&&(fg[i+1][j]||fg[i][j+1]||fg[i][j]);
		for(int i=1;i<=n;i++)
		for(int j=1;j<=m;j++)
		if(fg[i][j])fg[i][j]=0;
		else if(s[i][j]=='-')as=2ll*as%mod,s[i][j]='#';
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(s[i][j]=='#')
		if(!fr[i][j])fr[i][j]=++ct,dfs(i,j);
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(s[i][j]=='-')fr[i][j]=++ct;
		for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(s[i][j]=='-')
		{
			if(!s[i+1][j-1])adde(ct+1,fr[i][j]);
			else if(s[i+1][j-1]=='#')adde(fr[i+1][j-1],fr[i][j]);
			else if(s[i+1][j]=='#')adde(fr[i+1][j],fr[i][j]);
			else if(s[i][j-1]=='#')adde(fr[i][j-1],fr[i][j]);
			if(!s[i-1][j+1])adde(fr[i][j],ct+2);
			else if(s[i-1][j+1]=='#')adde(fr[i][j],fr[i-1][j+1]);
			else if(s[i-1][j]=='#')adde(fr[i][j],fr[i-1][j]);
			else if(s[i][j+1]=='#')adde(fr[i][j],fr[i][j+1]);
			else adde(fr[i][j],fr[i-1][j+1]);
		}
		for(int i=1;i<=n;i++)if(s[i][1]=='#'&&!f1[fr[i][1]])f1[fr[i][1]]=1,adde(ct+1,fr[i][1]);
		for(int i=1;i<=m;i++)if(s[n][i]=='#'&&!f1[fr[n][i]])f1[fr[n][i]]=1,adde(ct+1,fr[n][i]);
		for(int i=1;i<=ct;i++)f1[i]=0;
		for(int i=1;i<=n;i++)if(s[i][m]=='#'&&!f1[fr[i][m]])f1[fr[i][m]]=1,adde(fr[i][m],ct+2);
		for(int i=1;i<=m;i++)if(s[1][i]=='#'&&!f1[fr[1][i]])f1[fr[1][i]]=1,adde(fr[1][i],ct+2);
		queue<int> fu;fu.push(ct+1);dp[ct+1]=1;
		while(!fu.empty())
		{
			int u=fu.front();fu.pop();
			for(int i=head[u];i;i=ed[i].next)
			{
				dp[ed[i].t]=(dp[ed[i].t]+dp[u])%mod;
				in[ed[i].t]--;if(!in[ed[i].t])fu.push(ed[i].t);
			}
		}
		return 1ll*dp[ct+2]*as%mod;
	}
};
```

##### TCO11 Online Round 5 TC11487 RemoveGame

###### Problem

对于一个只包含 `ox` ，且第一个位置为 `o`，最后一个位置为 `x` 的串 $s$，两个人在 $s$ 上进行如下游戏。其中 `o` 是第一个人的字符，`x` 是第二个人的字符。

两人轮流操作，每个人每次操作时选择一个对手的字符，满足它两侧都有自己的字符，随后删去对手的这个字符。当有人无法操作时，游戏结束。

游戏结束时，如果剩余的第一个人的字符数量大于剩余的第二个人的字符数量，则第一个人获胜。如果小于，则第二个人获胜。否则平局。

双方都采取如下策略：

1. 如果自己能赢，则采取让自己能赢的策略中，游戏结束时自己的字符数量剩余最多的行动。
2. 否则，如果能平局，则采取能让自己平局的行动。
3. 否则，采取让对手字符剩余最少的行动。

称一个串是好的，当且仅当游戏结束时，第一个人获胜，且第一个人的字符剩余 $k$ 个以上。

给一个长度为 $n$ 的包含 `ox?` ，且第一个位置为 `o`，最后一个位置为 `x` 的串，你需要求出有多少中将每个 `?` 变为 `ox` 之一的方案，使得得到的串是好的。输出方案数。

$n\leq 40$

###### Sol

两人每次都是轮流删对方的字符，因此有如下结论：

1. 如果第一个人的字符数量大于第二个人的字符数量，则第一个人必胜。
2. 如果两人字符数量相等，则可能是平局，也可能是第一个人获胜。
3. 如果第二个人字符数量更多，则第一个人不可能获胜。

对于第一种情况，无论怎么操作第一个人一定必胜。因此此时双方变为在第一个人剩余的字符数量上博弈。

注意到每个时刻第二个人能删去的位置位于一段后缀，因此第二个人每次会删去最靠前的能删去的位置。

因而第一个人每次也会选择删去最靠前的能删去的位置。因此第二个人可以删去的数量即为对于除去第一个 `x` 外的每个 `x` 向后匹配一个 `o`，得到的最大的匹配数量。

如果序列已经确定，则可以从左到右考虑每个位置，记录当前左侧没有被匹配的 `x` 数量，已经匹配的对数以及之前的 `o` 数量。最后用 `o` 数量减去匹配数量判断即可。

因此可以将这个过程写成 $dp$，复杂度 $O(n^4)$。

考虑第二种情况，此时如果最后一次操作是第一个人，则第一个人获胜，否则平局。

则双方都希望自己最后一个操作。此时无视最左侧的全 `o` 段，无视最右侧的全 `x` 段，则可以看成如下问题：

有一个包含 `ox` 的串，最左侧一个为 `x`，最右侧一个为 `o`。双方轮流删对方的字符，且一次操作后，在最左侧的 `o` 会消失，在最右侧的 `x` 会消失，删完的人获胜。

显然删去边界上的字符后会使对方字符消失，相当于对方获得优势，因此双方都只会在最后一次删去边界上的字符。因此可以发现第一个人获胜当且仅当中间这一段中 `x` 的数量小于等于 `o` 的数量。

因此考虑枚举中间这一段的位置，此时相当于求出填中间一段使得总的 `o` 数占总数方案一半的方案数。可以直接 $dp$。

复杂度 $O(n^4)$

###### Code

```cpp
#include<cstdio>
#include<string>
using namespace std;
#define N 45
#define ll long long
int n;
ll dp[N][N][N][N],as,f[N][N];
char s[N];
struct RemoveGame{
	ll countWinning(string fu,int v)
	{
		n=fu.size();
		for(int i=1;i<=n;i++)s[i]=fu[i-1];
		dp[0][0][0][0]=1;
		for(int i=1;i<=n;i++)
		for(int j=0;j<i;j++)
		for(int k=0;k<i;k++)
		for(int l=0;l<i;l++)
		for(int t=0;t<2;t++)
		{
			if(!t&&s[i]=='x')continue;
			if(t&&s[i]=='o')continue;
			int nj=j,nk=k,nl=l;
			if(!t){if(nk>1)nk--;else nl++;nj++;}
			else nk++;
			dp[i][nj][nk][nl]+=dp[i-1][j][k][l];
		}
		for(int i=0;i<=n;i++)if(i>n-i)
		for(int j=v;j<=n;j++)
		for(int k=0;k<=n;k++)if(dp[n][i][k][j])
		as+=dp[n][i][k][j];
		if(n%2==0)
		{
			for(int i=1;i<=n;i++)
			for(int j=i+1;j<=n;j++)if(i+j<=n+1&&s[i]!='o'&&s[j]!='x'&&n-j+1>=v)
			{
				int fg=1;
				for(int p=1;p<i;p++)if(s[p]=='x')fg=0;
				for(int p=j+1;p<=n;p++)if(s[p]=='o')fg=0;
				if(!fg)continue;
				for(int p=0;p<=n;p++)
				for(int q=0;q<=n;q++)f[p][q]=0;
				f[i][1]=1;
				for(int k=i+1;k<j;k++)
				for(int l=0;l<=k-i;l++)
				{
					if(s[k]!='o')f[k][l+1]+=f[k-1][l];
					if(s[k]!='x')f[k][l]+=f[k-1][l];
				}
				as+=f[j-1][j-n/2];
			}
		}
		return as;
	}
};
```

##### SRM607 TC12965 PulleyTautLine

###### Problem

$n$ 个半径为 $r$ 的圆等间隔地排成一列，相邻两个圆圆心的距离为 $d$。圆心位置为 $(d,0),(2d,0),...,(nd,0)$。

你需要从 $(0,d)$ 出发引一条绳子到达 $((n+1)d,0)$。绳子可以重复经过同一个位置，但绳子必须在所有位置都是拉紧的。

即绳子可以按照如下方式运动：

1. 从起点出发，到达第 $1$ 个圆的左上/左下。
2. 从第 $i$ 个圆的左上/下出发，到达第 $i+1$ 个圆的左上/下（直向右）。向左同理。
3. 从第 $i$ 个圆的左上/下出发，到达第 $i+1$ 个圆的左下/上（斜向右）。向左同理。
4. 从第 $i$ 个圆的左上出发，到达第 $i$ 个圆的右下（绕 $\frac 12$ 圈），另外三种情况同理。
5. 从第 $n$ 个圆的左上/左下到达结尾。

给定 $k$，求所有路径中，长度第 $k$ 小的路径的长度。

$n\leq 50,d,r\leq 10^9,k\leq 10^{18}$

###### Sol

可以将路径分成三部分：

1. 将直着在两个圆间运动和写着看成一种，且先认为在每个位置最多转半圈，确定在不同圆间移动的次数。
2. 确定每一次不同圆间运动是直着还是写着。
3. 加上每个位置额外转的圈。

可以发现如果在不同圆间运动了 $k$ 次，则第二步就有 $2^k$ 种方案。且斜向的路径不会超过 $\sqrt 2$ 倍长度。

因此只要 $n\geq 2$，则即使不考虑转圈，则考虑一种移动 $\max(n,\log k)$ 次的方案，则长度不超过 $\max(n,\log k)*(\sqrt 2+\frac{\pi}2)d$。这个值不超过 $200$。

再考虑第三种转移以及其它情况，最后的答案一定更小。事实上这个值在极限情况下不超过 $112d$。

因此在不同圆间运动的次数有限（转圈次数可能很多）。

考虑 $dp$ 第一步，设 $dp_{n,x,y,0/1}$ 表示当前移动了 $n$ 次，在圆 $x$，之前转向了 $y$ 次，当前的方向为向左还是向右的方案数。可以只转移很少的步数，因此可以直接做。

接下来考虑二分答案，变为计算小于某个长度的路径数。

枚举转向的次数 $a$，经过的圆数量 $b$，以及经过的圆中斜向的次数 $k$。则有一个方案数 $C_{b-1}^k$。如果此时额外转了 $t$ 圈，则插板可得方案数为 $C_{t+b-1}^{b-1}$。

在枚举前三种值后，此时额外转的圈数不超过某个值 $v$，因此方案数为 $\sum_{i=0}^vC_{i+b-1}^{b-1}=C_{v+b}^b$。

因为答案超过 $10^{18}$ 后可以直接退出二分，因此预处理 $v+b$ 小的情况后，只需要最多 $O(\log v)$ 步即可结束。

因此复杂度为 $O(n\max(n^2,\log^2 k)+\max(n^3,\log^3 k)\log^2v)$，可以看成 $\log^5$ 级别。最后计算组合数的 $\log$ 常数非常小，且整体上跑不满。

###### Code

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
using namespace std;
#define N 105
#define ll long long
int d,r,n;
ll k,dp[N][55][N][2],c[N*4][N*4];
double v1,v2,v3,v4,pi=acos(-1);
double calc(double d,double r)
{
	double d1=sqrt(d*d-r*r),ag=asin(d1/d);
	return d1+r*(pi/2-ag);
}
ll calc1(int n,ll d)
{
	if(n+d<=400||c[min(n+d,400ll)][n]>=k)
	return c[min(n+d,400ll)][n];
	ll as=1;
	for(int i=1;i<=n;i++)
	{
		if(1.0*as*(d+n-i+1)/i>1e18)return 1e18;
		as=as*(d+n-i+1)/i;
	}
	return as;
}
ll solve(double v)
{
	ll as=0;
	for(int i=0;i<=100;i++)
	for(int j=0;j<=i;j++)
	for(int l=0;l<=i;l++)
	{
		ll vl=dp[i][n][l][1];if(!vl)continue;
		if(1.0*vl*c[i][j]>1e18)vl=1e18;else vl*=c[i][j];
		double ds=j*v3+(i-j)*v2+v1*2+v4*l;
		ds=v-ds;
		ll tp=floor(ds/v4/2);if(tp<0)continue;
		ll vl2=calc1(i+1,tp);
		if(1.0*vl*vl2>1e18)vl=1e18;else vl*=vl2;
		as=min(as+vl,k);
	}
	return as;
}
struct PulleyTautLine{
	double getLength(int f1,int f2,int f3,ll f4)
	{
		d=f1;r=f2;n=f3;k=(f4+1)/2;
		v1=calc(d,r);v2=d;v3=2*calc(0.5*d,r);v4=r*pi;
		if(n==1)return (k-1)*v4*2+v1*2;
		dp[0][1][0][1]=1;
		for(int i=1;i<=100;i++)
		for(int j=1;j<=n;j++)
		for(int l=0;l<i;l++)
		for(int d=0;d<2;d++)
		for(int t=0;t<2;t++)
		{
			int nt=j+(t?1:-1);if(nt<1||nt>n)continue;
			dp[i][nt][l+(d!=t)][t]=min(dp[i][nt][l+(d!=t)][t]+dp[i-1][j][l][d],k);
		}
		for(int i=0;i<=400;i++)c[i][0]=c[i][i]=1;
		for(int i=2;i<=400;i++)for(int j=1;j<i;j++)c[i][j]=min(c[i-1][j-1]+c[i-1][j],k);
		double lb=0,rb=1e12;
		for(int t=1;t<=80;t++)
		{
			double mid=(lb+rb)/2;
			if(solve(mid)<k)lb=mid;else rb=mid;
		}
		return lb;
	}
};
```

##### SRM327 TC6834 PostfixRLE

###### Problem

给一个长度为 $n$ 的后缀表达式，其中变量为所有小写字符，运算符共有 $8$ 种。

同种运算符间满足交换律和结合律，你可以使用这些规律，在不改变表达式的值的情况下对后缀表达式进行调整。但不同种运算间没有任何性质。

定义一个字符串的权值为它的连续极长字符相同段数量，你需要调整后缀表达式使得权值最小。输出最小权值。

$n\leq 2500$

###### Sol

首先建出后缀表达式的二叉树，则一次调整相当于进行如下操作之一：

1. 交换一个点的两个儿子。
2. 如果一个点的运算和它父亲相同，则进行对应的左旋/右旋操作。

可以发现此时树上一个相同运算的连通块内部的形态可以任意进行调整，且所有这个连通块的儿子间的顺序可以任意调整。使用Splay的方式，可以将操作1看成交换两个相邻儿子的顺序，操作2看成Splay。因而上面的性质显然。

此时因为它的儿子都不是这种运算符，因此它儿子的后缀表达式不可能以这种运算符结尾，也不可能以运算符开头。

因此此时考虑排成一条向右的链，则在后缀表达式中形如首先将所有儿子的后缀表达式排列，最后再加入若干个当前运算符，这样一定最优。

考虑将所有儿子分成两类：

1. 包含运算符的儿子。这类的后缀表达式一定以一个变量开头，以一种固定的运算符结尾。
2. 只包含一个变量的儿子。这类的后缀表达式只有一个字符。

此时第一类之间的顺序改变不能减小段数，但将第二类的放到某一个第一类儿子的开头可能可以减少段数。

具体来说，如果第二类中出现了字符 $c$，且第一类中有一个以 $c$ 开头的字符，则可以减少 $1$ 的段数。

考虑设 $dp_{i,c}$ 表示 $i$ 这个子树的后缀表达式，以 $c$ 结尾的最小段数。注意到不同的开头最多在后面减少 $1$ 的段数，因此如果 $dp_{i,c}$ 不等于 $dp_i$ 的最小值，则它没有意义。因此可以只记录 $dp_i$ 的最小值以及取到最小值的字符。

考虑这个点上的转移。此时每个儿子只会选择取到 $dp_i$ 最小值的字符开头。可以看成每个儿子可以选择一些字符中的一种开头，且对于第二类出现的每种字符，如果这种字符是至少一个儿子的开头字符，则有 $1$ 的收益。

因此这是一个最大匹配，可以直接dinic。

再考虑计算此时的 $dp_{i,c}$。如果 $c$ 在某种第二类儿子中出现，则将这种儿子连接的段放到开头即可，因此它一定等于 $\min dp_i$。否则，相当于要求必须有一个第一类儿子以 $c$ 开头。此时再进行一次最大匹配即可。

最后只需要先将每种相同的运算符缩点在一起，对接下来的树dfs求即可。

复杂度 $O(|\sum|*dinic(n,n))$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
#define N 2550
int n,st[N],ct,ch[N][2],fg[N],dp[N];
char s[N];
int head[N],cnt,cur[N],dis[N],cn;
struct edge{int t,next,v;}ed[N*3];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;
}
bool bfs(int s,int t)
{
	for(int i=1;i<=cn;i++)cur[i]=head[i],dis[i]=-1;
	queue<int> qu;qu.push(s);dis[s]=0;
	while(!qu.empty())
	{
		int u=qu.front();qu.pop();
		for(int i=head[u];i;i=ed[i].next)if(dis[ed[i].t]==-1&&ed[i].v)
		dis[ed[i].t]=dis[u]+1,qu.push(ed[i].t);
		if(dis[t]!=-1)return 1;
	}
	return 0;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)
	if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(ed[i].v,f))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
void dfs1(int x)
{
	if(s[x]>='a'&&s[x]<='z'){dp[x]=1;return;}
	vector<int> s1,s2;
	int f1=0,as=0;
	s1.push_back(x);
	for(int i=0;i<s1.size();i++)
	{
		if(s[s1[i]]==s[x])s1.push_back(ch[s1[i]][0]),s1.push_back(ch[s1[i]][1]);
		else
		{
			dfs1(s1[i]);
			if(dp[s1[i]]==1)f1|=1<<s[s1[i]]-'a';
			else as+=dp[s1[i]],s2.push_back(fg[s1[i]]);
		}
	}
	fg[x]=f1;
	int su=0;
	for(int i=1;i<=cn;i++)head[i]=0;cnt=1;cn=28+s2.size();
	for(int i=1;i<=26;i++)if((f1>>i-1)&1)as++,adde(1,i+2,1);
	for(int i=0;i<s2.size();i++)
	{
		adde(i+29,2,1);
		for(int j=1;j<=26;j++)if((s2[i]>>j-1)&1)adde(j+2,i+29,1);
	}
	while(bfs(1,2))su+=dfs(1,2,1e8);
	as-=su;
	for(int t=1;t<=26;t++)
	{
		int su1=0;
		for(int i=1;i<=cn;i++)head[i]=0;cnt=1;cn=28+s2.size();
		for(int i=1;i<=26;i++)if(((f1>>i-1)&1)||t==i)adde(1,i+2,1);
		for(int i=0;i<s2.size();i++)
		{
			adde(i+29,2,1);
			for(int j=1;j<=26;j++)if((s2[i]>>j-1)&1)adde(j+2,i+29,1);
		}
		while(bfs(1,2))su1+=dfs(1,2,1e8);
		if(su1>su)fg[x]|=1<<t-1;
	}
	dp[x]=as+1;
}
struct PostfixRLE{
	int getCompressedSize(vector<string> fu)
	{
		for(int i=0;i<fu.size();i++)for(int j=0;j<fu[i].size();j++)s[++n]=fu[i][j];
		for(int i=1;i<=n;i++)if(s[i]>='a'&&s[i]<='z')st[++ct]=i;
		else ch[i][0]=st[ct-1],ch[i][1]=st[ct],st[ct-1]=i,ct--;
		dfs1(n);return dp[n];
	}
};
```

##### SRM714 TC14120 Salesman

###### Problem

在一条数轴上有 $n$ 个人，每个人有一个值 $v_i$，它所在的位置为 $x_i$。如果 $v_i>0$ ，则说明这个人可以提供 $v_i$ 个物品。否则说明这个人需要 $-v_i$ 个物品。

你初始在位置 $0$，你每个时刻可以在数轴上移动 $1$ 个单位长度。你经过一个提供物品的人时，可以从那里拿任意数量的物品（但需要他还能提供这么多物品）。你经过一个需要物品的人时，可以给他一些物品。你可以拿任意多的物品。

你希望满足所有需要物品的人的要求，求你最少需要的时间。

$n\leq 4000$

###### Sol

~~现在 $n>2^{11}$ 的数据交空程序都会TLE，所以我选择认为自己过了。~~

在给一个人物品时，如果他两侧都还有没有被满足的人，则之后去满足那两个人的时候一定会经过中间的人，因此可以之后再给这个人物品。

因此给物品的顺序一定是每次给最左侧或者最右侧的。

同时可以发现，在给了最左侧和最右侧之后：

1. 如果要向两侧的人拿物品，则一定在你在最左侧/最右侧的人的时候过去拿，不会之后给中间人物品的过程中去拿。

2. 中间可以给物品的人之前一定都拿过了。

因此此时直接依次走过去即可。

因此最优的给物品方案一定是先顺序给一段前缀，再倒过来给剩下一段后缀。或者倒过来。可以枚举是走哪种方向。这里讨论第一种情况。

先考虑没有倒过来走的情况。则第一步是从开头走到最左侧。可以发现再向左的部分只会在此时去一次，可以枚举向左走到了哪，这里有 $n$ 种情况。

此时变为只有向右的问题。考虑如下类似反悔的贪心：

从左向右给每个人物品，维护当前所在的位置 $nx$，之前到达过的最靠右位置 $rx$，以及当前手上的物品数量。

从左向右考虑每个人，如果这个人是给物品的，如果之前 $rx$ 小于这个位置，则说明之前没有拿走这个位置的物品，此时一定会拿走这些物品。然后更新 $nx,rx$。

如果这个人是需要物品的，此时如果手上的物品数量大于等于需求，则可以直接给。

否则，需要去右侧拿更多的物品。此时可以从之前到达的最右的位置继续向右扩展，相当于在之前这个位置反悔，继续向右走一段再回来。此时从这个位置向右，只拿物品，直到物品足够位置。此时因为要回来，每扩展一步的代价是 $2$ 倍长度。

这样的贪心复杂度为 $O(n)$。

再考虑倒着走后缀的情况。相当于在满足某个人的需求后，接下来向右走到最右侧的需求右侧的某个地方拿够物品，再向左满足剩下的需求。

可以发现，如果最开始向左走的长度被确定后，则最后向右需要走多少才能满足所有需求是确定的，可以提前求出。因此此时可以 $O(1)$ 计算这种情况的总代价。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 2550
int n,s[N],t[N],as=1e9;
void solve()
{
	int lb=n+1,rb=0;
	for(int i=n;i>=1;i--)if(t[i]<0)lb=i;
	for(int i=1;i<=n;i++)if(t[i]<0)rb=i;
	if(lb==n+1){as=0;return;}
	for(int i=1;i<=lb;i++)
	{
		int s1=0,r1=n+1;
		for(int j=1;j<=n;j++)if(t[j]<0)s1+=t[j];
		for(int j=i;j<=n;j++)
		{
			if(t[j]>0)s1+=t[j];
			if(s1>=0&&j>=rb){r1=j;break;}
		}
		if(r1==n+1)continue;
		int ds=(s[i]>0?s[i]:-s[i]),su=0,nw=0,fr=0,l1=s[i];
		for(int j=i;j<=n;j++)if(s[j]<=0||j<=lb)su+=t[j]>0?t[j]:0;
		if(s[i]>0)nw=s[i];
		for(int j=1;j<=n;j++)if(s[j]<=nw)fr=j;
		for(int j=1;j<=n;j++)if(t[j]<0)
		{
			as=min(as,ds+s[r1]*2-s[j]-l1);
			if(nw<s[j])nw=s[j];
			while(fr<n&&s[fr+1]<=nw)fr++,su+=t[fr]>=0?t[fr]:0;
			while(su+t[j]<0)
			{
				if(fr==n){ds=1e9;break;}
				ds+=2*(s[fr+1]-nw);fr++;nw=s[fr];
				if(t[fr]>0)su+=t[fr];
			}
			su+=t[j];ds+=s[j]-l1;l1=s[j];
		}
		as=min(as,ds);
	}
}
struct Salesman{
	int minMoves(vector<int> s1,vector<int> s2)
	{
		n=s1.size();
		for(int i=1;i<=n;i++)s[i]=s1[i-1],t[i]=s2[i-1];
		solve();
		for(int i=1;i<=n;i++)s[i]*=-1;
		for(int i=1;i*2<=n;i++)swap(s[i],s[n+1-i]),swap(t[i],t[n+1-i]);
		solve();
		return as;
	}
};
```

##### SRM619 TC12742 SimilarSequencesAnother

###### Problem

称两个序列 $(a,b)$ 是相似的，当且仅当可以在两个序列中各自删去不超过两个字符，使得两个序列相同。

给出 $n,k$，只考虑所有元素为 $[1,k]$ 间正整数的长度为 $n$ 的序列，求相似的序列对数量，模 $10^9+9$

$n\leq 100,k\leq 10^9$

###### Sol

相当于要求两个序列的LCS大于等于 $n-2$。

考虑计算LCS的 $dp$：$dp_{i,j}$ 表示第一个串的前 $i$ 个字符和第二个串的前 $j$ 个字符的LCS。

考虑两个串各自向后加一个字符转移，则相当于从 $dp_{n,i},dp_{i,n}$ 转移到 $dp_{n+1,i},dp_{i,n+1}$。

又因为最后只需要判断大于等于 $n-2$ ，因此可能有用的状态只有 $dp_{n,n-2},dp_{n,n-1},dp_{n,n},dp_{n-1,n},dp_{n-2,n}$ 五个，且这些状态值也不会超过 $n-2,n-1,n,n-1,n-2$。因此只记录每个位置是否大于等于 $n-2$ 以及大于等于时的值，这样的状态不超过 $2*3*4*3*2$。

同时因为判断转移条件需要之前的字符，因此可以再记录两个串位置 $n-1,n$ 的这四个字符之间的相等关系。可以发现相等关系加上五个位置的 $dp$ 的总状态数只有 $126$ 个。

转移时考虑枚举两个序列末尾新加入的数的情况。此时可以枚举新加入的数是否与之前的四个数相等，以及如果两个数都不等于之前的四个数，则它们两个是否相等。这样的情况只有不超过 $26$ 种。然后直接转移即可。

复杂度可以做到 $O(n*126*26)$ 或者 $O(126*26+126^3\log n)$。

###### Code

```cpp
#include<cstdio>
#include<map>
using namespace std;
#define N 105
#define M 130
#define mod 1000000009
#define ll long long
int n,k,dp[N][M],ct;
struct state{
	int dp[2][3],vl[2][3];
}sr[M];
map<ll,int> fu;
int getid(state f)
{
	ll as=0,fg=0;
	for(int i=0;i<2;i++)for(int j=0;j<3;j++)if(f.dp[i][j]<3)fg=1;
	if(!fg)return 0;
	for(int i=0;i<2;i++)for(int j=0;j<3;j++)as=as*4+f.dp[i][j];
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)as=as*6+f.vl[i][j];
	if(!fu[as])fu[as]=++ct,sr[ct]=f;
	return fu[as];
}
void trans(int n,int s)
{
	for(int i=1;i<=5;i++)
	for(int j=1;j<=6;j++)
	{
		if(k<i||k<j)continue;
		int tp=1ll*(i<5?1:k-4)*(j<5?1:k-4)%mod*dp[n][s]%mod;
		if(i<5&&j==6)continue;
		if(i==5&&j==5)tp=1ll*(k-4)*dp[n][s]%mod;
		if(i==5&&j==6)tp=1ll*(k-4)*(k-5)%mod*dp[n][s]%mod;
		if(!tp)continue;
		state ls=sr[s],nw;
		for(int p=0;p<2;p++)for(int q=0;q<2;q++)nw.vl[p][q+1]=ls.vl[p][q];
		nw.vl[0][0]=i;nw.vl[1][0]=j;
		for(int p=0;p<2;p++)nw.dp[p][2]=min(ls.dp[p][1]+1,nw.vl[p^1][0]==nw.vl[p][2]?ls.dp[p][2]:3);
		for(int p=0;p<2;p++)nw.dp[p][1]=min(min(ls.dp[p][0]+1,nw.dp[p][2]),nw.vl[p^1][0]==nw.vl[p][1]?ls.dp[p][1]:3);
		nw.dp[0][0]=nw.dp[1][0]=min(min(nw.dp[0][1],nw.dp[1][1]),nw.vl[0][0]==nw.vl[1][0]?ls.dp[0][0]:3);
		int fu[9]={0},c1=0;
		for(int p=0;p<2;p++)for(int q=0;q<2;q++)
		{
			int vl=nw.vl[p][q];
			if(!fu[vl]&&vl)fu[vl]=++c1;
			nw.vl[p][q]=fu[vl];
		}
		int id=getid(nw);
		dp[n+1][id]=(dp[n+1][id]+tp)%mod;
	}
}
struct SimilarSequencesAnother{
	int getCount(int a,int b)
	{
		n=a;k=b;
		dp[0][getid(sr[0])]=1;
		for(int i=1;i<=n;i++)for(int j=1;j<=ct;j++)trans(i-1,j);
		int as=0;
		for(int j=1;j<=ct;j++)as=(as+dp[n][j])%mod;
		return as;
	}
};
```

##### SRM506 TC11360 SlimeXSlimeRancher

###### Problem

有 $n$ 个三元组 $(a_i,b_i,c_i)$。你可以花费 $1$ 的代价，将某一个三元组的某一个元素 $+1$。

你希望满足，在所有 $+1$ 的操作结束后，存在一种将三元组排序的方式，使得排序后所有三元组的 $a_i$ 单调不降， $b_i$ 单调不降，$c_i$ 单调不降。

求出最小代价。

$n\leq 150$

###### Sol

考虑先找一个顺序，那么显然需要将每个位置的值都变成对应维的前缀max。

考虑对三维上的前缀max序列做 $dp$。离散化后三维分别只有 $n$ 个值，因为前缀max都是不降的，因此它的变化可以看成每次给某一维 $+1$，直到结束。

每个三元组的代价是它加入序列时的三维前缀max之和。因此对于一个固定的三维前缀max序列，每个三元组一定是在最早的能加入（即三维的前缀max都不小于三元组对应的值）的时刻加入。

因此设 $su_{i,j,k}$ 表示三维分别小于等于 $i,j,k$ 的三元组数量，$dp_{i,j,k}$ 表示当前前缀max序列的结尾为 $(i,j,k)$，前面的最小代价。则有：
$$
dp_{i,j,k}=\min(dp_{i-1,j,k}+(i+j+k)*(su_{i,j,k}-su_{i-1,j,k}),dp_{i,j-1,k}+(i+j+k)*(su_{i,j,k}-su_{i,j-1,k}),dp_{i,j,k-1}+(i+j+k)*(su_{i,j,k}-su_{i,j,k-1}))
$$
答案为 $dp_{n,n,n}$。其中 $i+j+k$ 为离散化后的权值。

可以发现这样会计算一些不是真正的前缀max的序列的情况，但这些情况都是在某个前缀max序列上增加一些位置的值得到的，它们一定不优。而真正的前缀max序列一定会被计算。因而这样的算法正确性得到保证。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<string>
using namespace std;
#define N 153
#define ll long long
int n,v[3][N],s[3][N],su[N][N][N];
ll dp[N][N][N],sr;
struct SlimeXSlimeRancher{
	ll train(vector<string> f1,vector<string> f2,vector<string> f3)
	{
		vector<string> fr[3]={f1,f2,f3};
		for(int j=0;j<3;j++)
		{
			int nw=0,ct=0;
			for(int i=0;i<fr[j].size();i++)
			for(int k=0;k<fr[j][i].size();k++)
			{
				char st=fr[j][i][k];
				if(st==' ')v[j][++ct]=nw,nw=0;
				else nw=nw*10+st-'0';
			}
			v[j][++ct]=nw;
			n=ct;
		}
		for(int j=0;j<3;j++)for(int i=1;i<=n;i++)s[j][i]=v[j][i],sr+=s[j][i];
		for(int j=0;j<3;j++)sort(s[j]+1,s[j]+n+1);
		for(int i=1;i<=n;i++)
		{
			int tp[3]={0};
			for(int j=0;j<3;j++)tp[j]=lower_bound(s[j]+1,s[j]+n+1,v[j][i])-s[j];
			su[tp[0]][tp[1]][tp[2]]++;
		}
		for(int t=0;t<3;t++)for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)su[i][j][k]+=su[i-(t==0)][j-(t==1)][k-(t==2)];
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)
		{
			ll as=1e18;
			for(int t=0;t<3;t++)as=min(as,dp[i-(t==0)][j-(t==1)][k-(t==2)]+(su[i][j][k]-su[i-(t==0)][j-(t==1)][k-(t==2)])*(1ll*s[0][i]+s[1][j]+s[2][k]));
			dp[i][j][k]=as;
		}
		return dp[n][n][n]-sr;
	}
};
```

##### SRM592 TC12434 SplittingFoxes2

###### Problem

有一个长度为 $n$ 的包含非负整数的数组 $a$（下标从 $0$ 开始），满足 $a_i=a_{n-i}$。

现在给出长度为 $n$ 的数组 $b$，满足 $b_i=\sum_{0\leq j,k<n,(j+k)\equiv i(\bmod n)}a_ja_k$。

求出所有可能的 $a$ 中，字典序最小的一个，或者输出无解。

$n\leq 25,b_i\leq 10^6$

###### Sol

考虑DFT：$a_i'=\sum_{j=0}^{n-1}a_i\omega_{n}^{ij}$，$b_i'$ 同理。则 $b_i'=(a_i')^2$。

此时可以得到 $b_i'$ ，因而每个 $b_i'$ 可以得到两个可能的 $a_i'$。

可以注意到，如果 $a_i$ 满足 $a_i=a_{n-i}$，则DFT后也满足 $a_i'=a_{n-i}'$。因此可能的情况只有 $2^{\frac n2}$ 种。

因此可以枚举每一种情况，再IDFT求出原先的 $a$。

复杂度 $O(2^{\frac n2}n^2)$

实现时可以使用复数单位根，或者使用模质数+原根，可以使用 $10^6$ 级别的模数来避免Cipolla。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 105
#define M 1005000
int n,p,fr[N],vl[N],f[N],as[N],g,rv[M],s1[N];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*a*as%p;a=1ll*a*a%p;b>>=1;}return as;}
struct SplittingFoxes2{
	vector<int> getPattern(vector<int> v)
	{
		n=v.size();as[0]=1e8;
		for(int i=1e6;i>=1;i--)if((i-1)%n==0)
		{
			int fg=1;
			for(int j=2;j*j<=i;j++)if(i%j==0){fg=0;break;}
			if(fg){p=i;break;}
		}
		for(int i=p-1;i>=0;i--)rv[1ll*i*i%p]=i;
		for(int i=2;i<p;i++)
		{
			int fg=1,nw=1;
			for(int j=1;j<n;j++)
			{
				nw=1ll*nw*i%p;
				if(nw==1)fg=0;
			}
			if(fg&&1ll*nw*i%p==1){g=i;break;}
		}
		for(int i=0,v1=1;i<n;i++,v1=1ll*v1*g%p)
		for(int j=0,v2=1;j<n;j++,v2=1ll*v2*v1%p)
		vl[j]=(vl[j]+1ll*v[i]*v2)%p;
		int tp=n/2+1;
		for(int t=0;t<(1<<tp);t++)
		{
			int rg=pw(g,p-2),f1=1;
			for(int i=0;i<n;i++)f[i]=0;
			for(int i=0,v1=1;i<n;i++,v1=1ll*v1*rg%p)
			{
				int sv=rv[vl[i]],t1=i;
				if(t1>n/2)t1=n-t1;
				if((t>>t1)&1)sv=p-sv;
				for(int j=0,v2=1ll*sv*pw(n,p-2)%p;j<n;j++,v2=1ll*v2*v1%p)
				f[j]=(f[j]+v2)%p;
			}
			for(int i=0;i<n;i++)s1[i]=0;
			for(int i=0;i<n;i++)for(int j=0;j<n;j++)s1[(i+j)%n]=min(10000000ll,s1[(i+j)%n]+1ll*f[i]*f[j]);
			for(int i=0;i<n;i++)if(s1[i]!=v[i])f1=0;
			if(f1)
			{
				int fg=0;
				for(int i=0;i<n;i++)if(as[i]>f[i]){fg=1;break;}
				else if(as[i]<f[i])break;
				if(fg)for(int i=0;i<n;i++)as[i]=f[i];
			}
		}
		vector<int> as1;
		if(as[0]>1e7)as1.push_back(-1);
		else for(int i=0;i<n;i++)as1.push_back(as[i]);
		return as1;
	}
};
```

##### SRM406 TC8791 ShortPaths

###### Problem

给一个 $n$ 个点的有向图，边有边权 $v_i$。

保证任意两个点间只有不超过一条简单路径，且每个点最多在一个简单环中。

求出 $s$ 到 $t$ 的所有路径中，第 $k$ 短的路径的长度。

$n\leq 50,k\leq 10^{12},v_i\leq 9$

###### Sol

考虑 $s$ 到 $t$ 的链，此时链上会经过若干个环。

因为只有这一条简单路径，因此不会走除了这些链和经过的环之外的情况。又因为每个点只在一个简单环中，因此可能的路径一定形如，在第一个环上转若干圈，再到第二个环转若干圈，最后到达最后一个环。

此时问题变为，每个经过的环有一个环长 $l_i$，你可以选择每个环经过的次数 $c_i(c_i\geq 0)$，求出所有方案中 $\sum l_ic_i$ 第 $k$ 小的。设环的个数为 $m$。

如果 $m\leq 1$，则可以直接求。

如果 $m=2$，考虑每个环走 $0,...,\sqrt k$ 次，就有 $k$ 种方案。此时答案不超过 $500*\sqrt k$，因此可以二分答案后暴力check或者使用类欧几里得。

如果 $m\geq 3$，则答案不超过 $500*\sqrt[3]k$，因而可以直接设 $dp_{i,j}$ 表示考虑前 $i$ 个环后总长度为 $j$ 的方案数。此时可以通过。

复杂度 $O(?)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<string>
using namespace std;
#define N 105
#define M 5050000
#define ll long long
int n,x,y,fa[N],cr[N],le[N],ct,s1[N],c1;
ll k,l1,dp[M];
char s[N][N];
void dfs(int x)
{
	for(int i=1;i<=n;i++)if(s[x][i]!='0')
	if(!fa[i])fa[i]=x,dfs(i);
	else
	{
		int nw=x,su=s[x][i]-'0',id=++ct;cr[x]=id;
		while(nw!=i)su+=s[fa[nw]][nw]-'0',nw=fa[nw],cr[nw]=id;
		le[ct]=su;
	}
}
struct ShortPaths{
	ll getPath(vector<string> s2,ll k,int x,int y)
	{
		n=s2.size();x++;y++;
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=s2[i-1][j-1];
		fa[x]=-1;dfs(x);
		if(!fa[y])return -1;
		while(y!=x)
		{
			if(le[cr[y]])s1[++c1]=le[cr[y]],le[cr[y]]=0;
			l1+=s[fa[y]][y]-'0',y=fa[y];
		}
		if(le[cr[y]])s1[++c1]=le[cr[y]],le[cr[y]]=0;
		if(k>1&&!c1)return -1;
		if(c1<=1)l1+=1ll*s1[1]*(k-1);
		else if(c1==2)
		{
			if(s1[2]>s1[1])s1[2]^=s1[1]^=s1[2]^=s1[1];
			int lb=0,rb=1e9,as=0;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				ll su=0;
				for(int i=0;1ll*i*s1[1]<=mid&&su<=k;i++)su+=(mid-1ll*i*s1[1])/s1[2]+1;
				if(su>=k)as=mid,rb=mid-1;
				else lb=mid+1;
			}
			l1+=as;
		}
		else
		{
			dp[0]=1;
			for(int i=1;i<=c1;i++)
			{
				ll su=0;
				for(int j=s1[i];j<=5e6&&su<=k;j++)
				dp[j]+=dp[j-s1[i]],su+=dp[j];
			}
			ll su=0;
			for(int j=0;j<=5e6;j++)
			{
				su+=dp[j];
				if(su>=k){l1+=j;break;}
			}
		}
		return l1;
	}
};
```

##### SRM672 TC14040 Tdetective

###### Problem

有 $n$ 个人，每一对人之间有一个权值 $v_{i,j}$，满足 $v_{i,j}=v_{j,i}$。

你还有 $n$ 个对应每个人的权值 $c_1,...,c_n$。初始 $c_1=1,c_2=...=c_n=0$。

每一个时刻，你会进行如下操作：

1. 选择当前还没有被访问的人中，$c_i$ 最大的一个人 $x$。如果有多个最大的，则可以任意选一个。
2. 访问这个人，之后更新所有 $c_i$：$c_i\leftarrow\max(v_{x,i},c_i)$

对于每个人，求出最少需要多少次访问才能访问到这个人。

$n\leq 50$

###### Sol

将所有的 $v$ 看成边，则问题可以看成，每次选择已经访问过的点和没有访问过的点间的所有边中边权最大的一条，然后访问这条边对面的一个点。

如果所有边权两两不同，则考虑求出图的Kruskal重构树（边权大优先），由重构树性质不难得到访问重构树中某个点的子树对应的连通块的过程一定是先访问初始点所在的子树，再由重构树中这个点对应的连接两个儿子子树的边权最大的边走到另外一个子树，再访问另外一个子树。

对于边权由相同的情况，可以建出类似重构树的形式，但对于一种权值 $v$，如果加入这种权值的边时合并了多个连通块，则在此时的树中这些连通块有一个共同的父亲节点。

此时可以得到类似的结论：访问一个点的子树时，一定先访问当前点所在的儿子的子树，然后每次选择一个与已经访问过的子树有当前点权值的边相连的子树，访问这个子树。

设 $f_{u,v}$ 表示 $u$ 到 $v$ 需要的最少步数，显然 $u$ 到 $v$ 只会经过 $u,v$ 在重构树中的LCA的子树，因此考虑在树中从下往上求出所有 $f$。

考虑对于一个点 $u$，已经求出了 $u$ 的每个儿子子树内所有的 $f$，求出 $u$ 的子树内所有的 $f$ 的过程。显然最优方案中经过子树的方案一定构成一条链而不会分叉，设从 $x$ 到 $y$ 依次经过了子树 $s_1,...,s_k$，则不难发现步数为 $s_1,...,s_{k-1}$ 的子树大小和加上选择一条 $s_{k-1},s_k$ 间边权为当前重构树点权的边， $s_k$ 内边端点走到 $y$ 的最小步数之和。

对于 $s_1,\cdots,s_{k-1}$ 的部分，此时相当于找一条 $s_1$ 到 $s_{k-1}$ 的路径使得点权和最小，相当于求最短路。接着考虑枚举 $s_{k-1},s_k$ 之间的边，再枚举起点终点，即可直接求出答案。

注意到每条边只会枚举一次，因此这样的复杂度为 $O(n^4)$，最短路部分可以直接floyd，总复杂度 $O(n^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<string>
using namespace std;
#define N 55
int n,fr[N],dp[N][N],sz[N],f1[N][N],vl[N][N];
struct Tdetective{
	int reveal(vector<string> s)
	{
		n=s.size();
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)vl[i][j]=s[i-1][j-1]-'0';
		for(int i=1;i<=n;i++)fr[i]=i,sz[i]=1;
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(i!=j)dp[i][j]=1e7;else dp[i][j]=1;
		for(int v=9;v>=0;v--)
		{
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)f1[i][j]=i==j?0:1e7;
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(vl[i][j]==v&&fr[i]!=fr[j])f1[fr[i]][fr[j]]=sz[fr[j]];
			for(int k=1;k<=n;k++)for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)f1[i][j]=min(f1[i][j],f1[i][k]+f1[k][j]);
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(vl[i][j]==v&&fr[i]!=fr[j])
			for(int k=1;k<=n;k++)if(fr[j]==fr[k])
			for(int s=1;s<=n;s++)dp[s][k]=min(dp[s][k],f1[fr[s]][fr[i]]+sz[fr[s]]+dp[j][k]);
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(vl[i][j]==v&&fr[i]!=fr[j])
			{
				sz[fr[i]]+=sz[fr[j]];int tp=fr[j];
				for(int k=1;k<=n;k++)if(fr[k]==tp)fr[k]=fr[i];
			}
		}
		int as=0;
		for(int i=2;i<=n;i++)as+=(dp[1][i]-1)*(i-1);
		return as;
	}
};
```

##### TCO21 Semi2 TC17283 DoubleXorGame

###### Problem

有一个 $n$ 个点的有向图，考虑如下游戏：

每个点有一个颜色（黑/白），两人轮流操作，每个人可以选择一个黑点，并翻转这个点的颜色。如果这个点有出边，则必须选择一条出边，翻转出边另外一个点的颜色。

现在有 $k$ 个这样的图，图的结构相同，但每个图上初始的颜色不同。双方在这些图上操作，不能操作的人输。如果游戏无限进行，则认为是平局。求双方最优操作下，游戏的结果（先手胜/平局/负）。

$n\leq 12,k\leq 50$

$5s,256MB$

###### Sol

Winning Ways For Your Mathmatical Plays Chapter12 关于状态可能出现环的公平博弈

首先由如下结论：

如果状态 $x$ 的转移中有一些已经可以被表示为一个nimber，且这些nimber的 $mex$ 为 $m$。此时如果对于 $x$ 剩下的能转移到的状态，它们都可以转移到一个值为 $m$ 的状态，则状态 $x$ 等价于nimber $m$。

接下来，对于还不能被确定的状态，它们的值被称为loopy value $\infty_{abc...}$，其中 $a,b,c,...$ 为所有它能转移到的能被表示为nimber的状态。

loopy value与nimber不同，这种状态不能被表示为nim游戏中的一堆石子。

有如下结论：

一个 loopy value $\infty_{abc...}$ 加上一个nimber $x$ 时，它是先手必胜状态当且仅当 $x$ 属于 $\{a,b,c,...\}$，否则它是平局。多个nimber可以先加起来。

多个 loopy value 相加一定是平局。

证明不会，但好像可以感性理解。

考虑暴力实现上面的算法，复杂度不超过 $O(nm)$，其中 $n$ 为状态数，$m$ 为转移数。

本题中复杂度为 $O(4^nn^2)$，~~实际上只跑10ms~~

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<string>
using namespace std;
#define N 4100
int n,m,a,b,sg[N],rv[N],is[N],s2[N],k,c1,c2,su[N];
vector<int> nt[N],f1[N],lf[N];
struct DoubleXorGame{
	string solve(int n,vector<int> r1,vector<int> r2,vector<int> r3)
	{
		m=r1.size();
		for(int i=1;i<=m;i++)nt[r1[i-1]+1].push_back(r2[i-1]+1);
		for(int i=1;i<=n;i++)if(!nt[i].size())nt[i].push_back(0);
		for(int i=0;i<1<<n;i++)
		for(int j=1;j<=n;j++)if((i>>j-1)&1)
		for(int k=0;k<nt[j].size();k++)
		{
			int v1=i^(1<<j-1);
			if(nt[j][k])v1^=1<<nt[j][k]-1;
			f1[i].push_back(v1);lf[v1].push_back(i);
		}
		for(int i=0;i<1<<n;i++)sg[i]=s2[i]=-1;
		while(1)
		{
			int fg1=0;
			for(int i=0;i<1<<n;i++)if(sg[i]==-1)
			{
				for(int j=0;j<f1[i].size();j++)
				{
					int nt=f1[i][j];
					if(sg[nt]!=-1)su[sg[nt]]++;
				}
				for(int j=0;;j++)if(!su[j]){rv[i]=j;break;}
				for(int j=0;j<f1[i].size();j++)
				{
					int nt=f1[i][j];
					if(sg[nt]!=-1)su[sg[nt]]--;
				}
			}
			vector<int> t1[N],t2[N];
			for(int i=0;i<1<<n;i++)if(sg[i]==-1)t1[rv[i]].push_back(i);else t2[sg[i]].push_back(i);
			for(int i=0;i<1<<n;i++)if(t1[i].size())
			{
				for(int j=0;j<t2[i].size();j++)
				{
					int nw=t2[i][j];
					for(int k=0;k<lf[nw].size();k++)is[lf[nw][k]]=1;
				}
				for(int j=0;j<t1[i].size();j++)
				{
					int nw=t1[i][j],fg=1;
					for(int k=0;k<f1[nw].size();k++)
					{
						int t1=f1[nw][k];
						if(sg[t1]==-1&&!is[t1])fg=0;
					}
					if(fg)s2[nw]=i,fg1=1;
				}
				for(int j=0;j<t2[i].size();j++)
				{
					int nw=t2[i][j];
					for(int k=0;k<lf[nw].size();k++)is[lf[nw][k]]=0;
				}
			}
			for(int i=0;i<1<<n;i++)if(s2[i]!=-1)sg[i]=s2[i],s2[i]=-1;
			if(!fg1)break;
		}
		for(int i=0;i<r3.size();i++)
		{
			a=r3[i];
			if(sg[a]!=-1)c2^=sg[a];
			else c1=c1?-1:a;
		}
		if(c1==-1)return "draw";
		else if(!c1)return c2?"win":"lose";
		else
		{
			int fg=0;
			for(int i=0;i<f1[c1].size();i++)if(sg[f1[c1][i]]==c2)fg=1;
			return fg?"win":"draw";
		}
	}
};
```

##### SRM662 TC13854 MultiplicationTable

###### Problem

给定一个包含 $\{0,1,...,n-1\}$ 的大小为 $n$ 的集合 $S$。你需要找到一个 $S$ 上的二元运算 $*:S\times S\to S$，使得该运算满足结合律：$((a*b)*c)=(a*(b*c))$。

现在给出 $0*0,...,0*(n-1)$ 的结果，你需要构造剩余乘法的结果，使得这种乘法运算满足结合律。构造任意方案或者输出无解。

$n\leq 50$

###### Sol

考虑设 $f_a(x)=a*x$，问题相当于给出了 $f_0$，构造一组剩下的方案。

由结合律由 $a*(b*x)=(a*b)*x$，即 $f_a(f_b(x))=f_{a*b}(x)$。

令 $a=0$，即  $f_0(f_b(x))=f_{0*b}(x)$。因此设 $0^1=0,0^k=0*(0^{k-1})$，则 $f_{0^k}(x)$ 等于 $f_0$ 复合 $k$ 次的结果。记复合 $k$ 次的结果为 $f_0^k$

因此可以找到 $0^k$ 的循环节，从而可以求出这些位置的值。如果 $0^k$ 的循环节不能使 $f_0^k$ 在对应位置循环，则无解。

考虑一张图，其中 $x$ 向 $f_0(x)$ 连边，则这构成了一个基环树森林。上面的过程求出了所有 $0$ 在图中能到达的点的 $f_a$。考虑求出 $x$ 所在的连通块的所有 $f$。

设 $0^k$ 的最小循环为 $0^c=0^{c+l}$，考虑如下构造：

将连通块内所有点的 $f_a$ 表示为 $f_0^{v_a}$ 的形式，对于可以被表示为 $0^k$ 的元素，由上面可知 $v_{0^k}=k(0\leq k<c+l)$ 一定成立。对于剩下的点，设 $0*p=q$，则 $f_{0}^{v_p+1}=f_0^{v_q}$，因此考虑如果 $v_q=c$，则令 $v_p=c+l-1$，否则令 $v_p=c-1$，使用dfs的方式求出所有 $v_p$.

考虑如果得到了解（即所有 $v_p$ 非负）时，上面解的正确性。注意到上述构造满足对于任意节点 $u$，$f_{0*u}=f_0(f_u)$，而任意 $f_u$ 都能被表示为 $f_{0^k}$ 的形式，而 $f_a(b)=f_0^{v_a}(b)$，因此 $f_{a*b}=f_{f_0^{v_a}*b}=f_{0^{v_a}*b}=f_0^{v_a}(f_b)=f_a(f_b)$，其中倒数第二个等式由上一步推出。因此交换律成立。

上述构造无法得到解当且仅当删去基环树中的环后， $0$ 所在的子树内存在一个深度比 $0$ 大 $2$ 的节点。如果存在解，则对于这个节点的函数 $g$，相当于它需要满足 $f_0^{d+1}(g)=f_0^d$。注意到 $f_0^d(u)=v$ 当且仅当在基环树中 $u$ 走 $d$ 步正好到达 $v$。因此存在 $x$ 使得 $f_0^d(x)=v$ 当且仅当 $v$ 在环上或者 $v$ 子树内节点到它的最大距离大于等于 $d$。又因为整个子树内的最大距离大于等于 $d+1$，因此存在至少一个节点使得子树内节点到它的最大距离正好是 $d$，此时这个点在 $f^d$ 的值域中，但不在 $f^{d+1}$ 的值域中，因此 $f_0^{d+1}(g)=f_0^d$ 不可能成立。因此这种情况一定无解。

因此上述构造可以处理所有情况。最后再考虑 $0$ 所在连通块之外的元素，考虑对于这部分直接令 $a*b=a$，因为构造中左乘上一部分中的元素都可以看成左乘若干次 $0$，因而对于 $0$ 所在连通块之外的元素，它左乘任意一个元素后都在 $0$ 所在连通块之外。此时不难发现对于 $a$ 在连通块外，$a$ 在连通块内但 $b$ 在连通块外的情况结合律都成立，因此这样的构造保证了结合律成立。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 56
int n,v[N],as[N][N],f[N][N],dep[N],v1,is[N];
struct MultiplicationTable{
	vector<int> getMultiplicationTable(vector<int> v2)
	{
		vector<int> s1;
		n=v2.size();
		for(int i=1;i<=n;i++)v[i]=v2[i-1]+1;
		for(int i=1;i<=n;i++)f[0][i]=i;
		for(int i=1;i<=n+1;i++)for(int j=1;j<=n;j++)f[i][j]=v[f[i-1][j]];
		dep[1]=1;
		int nw=1;
		while(1)
		{
			if(dep[v[nw]]){v1=dep[v[nw]];break;}
			dep[v[nw]]=dep[nw]+1;nw=v[nw];
		}
		for(int i=1;i<=n;i++)if(f[dep[nw]+1][i]!=f[v1][i]){s1.push_back(-1);return s1;}
		for(int i=1;i<=n;i++)if(dep[i])is[i]=1;
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)if(is[v[j]]&&!is[j])dep[j]=dep[v[j]]==v1?dep[nw]:dep[v[j]]-1,is[j]=1;
		for(int i=1;i<=n;i++)
		{
			if(!is[i])for(int j=1;j<=n;j++)as[i][j]=i;
			else if(dep[i]>=0)for(int j=1;j<=n;j++)as[i][j]=f[dep[i]][j];
			else {s1.push_back(-1);return s1;}
		}
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)if(as[as[i][j]][k]!=as[i][as[j][k]]){s1.push_back(-1);return s1;}
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)s1.push_back(as[i][j]-1);
		return s1;
	}
};
```

##### TCO21 Final TC17308 BinaryTreeAutomatonHard

###### Problem

有一棵 $n$ 个点的树，每个节点最多有两个儿子。

每个节点上有 $26$ 个 $01$ 标记，编号为 $a,\cdots,z$，初始时，标记 $l,r,p$ 分别代表当前节点是否有左儿子，右儿子和父亲。

你需要构造一个自动机，一个自动机由有限个状态和有限个指令组成，它将会在二叉树上移动，初始时在根节点。每个指令形如 $(current\_state,conditions,new\_state,toggle,move)$，表示如果当前的状态为 $current\_state$，当前点上的标记满足 $conditions$，则执行这条指令，将状态改变为 $new\_state$，改变当前点的 $toggle$ 标记并移动到 $move$。其中 $conditions$ 中可以包含若干个限制，每个限制可以要求一个标记为 $0$ 或 $1$，只有这些限制全部符合时才满足要求。$move$ 可以移动到左右儿子或者父亲，或者不移动。

在执行这个自动机时，每一步会按照指令给出的顺序选择第一个可以执行的指令执行。如果当前没有指令可以执行，则结束运行过程。

你需要构造一个自动机，使得这个自动机从根节点开始执行（初始状态由你给定）时可以停止，且停止时一个节点被标记了 $x$ 当且仅当它是一个到根距离最远的节点。

你的自动机指令条数不能超过 $500$，指令总长不能超过 $6500$，运行步数不能超过 $2\times 10^8$。

$n\leq 60000$

 ###### Sol

考虑如下构造：

首先从下往上求出每个点的子树内到它的最大距离，求出根节点的这个距离后，考虑设一个值 $v$。其中 $v_{root}$ 等于子树内到它的最大距离，对于其它点 $u$，$v_u=v_{fa_u}-1$，那么一个节点是到根最远的节点当且仅当 $v_u=0$，这样即可判定答案。

考虑实现上面的过程，求出最大距离只需要dfs在回溯的过程中求，第二部分可以再dfs一次实现。因此关键在于实现dfs，并实现上面的操作。

实现dfs可以通过使用两个标记记录当前点的两个儿子是否被dfs过实现，剩余的操作中，第一部分相当于令 $v_u=\max(v_{ch_{u,0}},v_{ch_{u,1}})+1$，第二部分相当于令 $v_{u}=v_{fa_u}-1$。

第一部分的取max可以看成让一个点和它的某个儿子取max，取max可以看成二进制下的字典序比较，因此使用一个标记记录当前前面的位是否已经比较出了大小，然后从大到小枚举一遍位即可。这里同样可以使用标记记录两个儿子分别是否取过max。

第二部分中的赋值可以直接实现，因此最后的问题在于实现 $+1/-1$，可以通过维护是否进位实现，也可以直接枚举可能的情况（$+1$ 一定是选择后缀极长的 $1$ 并翻转这一段和前面的一个 $0$）实现，其中后者常数为 $1$，前者常数为 $2$。

上述过程的指令数为 $O(\log n)$，总操作步数为 $O(n\log n)$，使用的标记数为 $16+4+1+1+3=25$，理论上可以接受。但由于取max的常数较大，直接的实现指令数在 $600$ 左右。考虑一些卡常数技巧，比如如果取max时发现不需要改变就直接跳转到这一步结束的状态（少一个判断）。

最后取 $max$ 的常数为 $8$，复制值的常数为 $6$，$+1/-1$ 的常数为 $1$。如果直接每个点向两个儿子操作，则步数为 $15*2*\log n+O(1)$，可以卡进 $499$ 步。长度也可以卡进去。

这里也可以改为每个点向父亲贡献，需要在父亲处记录当前操作的是哪个儿子，因为上面的操作自动机都是在父亲和儿子间来回移动，因此这样可以减少 $\frac 14$ 的指令数。但这里没有实现过。

###### Code

```cpp
#include<cstdio>
#include<string>
#include<vector>
using namespace std;
char fu[40]="abcdefghijkmnoqstuvwyzxlrp";
/*
23-25 lrp
22 x
18 19 ldfs lmx
20 21 rdfs rmx
0-15 dp
*/
vector<string> as;
string s1;
void fuc(int v)
{
	v++;
	while(v)
	{
		int tp=v%62;
		v/=62;
		if(tp<10)s1+=tp+'0';
		else if(tp<36)s1+=tp-10+'a';
		else s1+=tp-36+'A';
	}
}
void doit(int fr,int c1,int c2,int nt,int vl,int fg)
{
	s1="";
	if(nt==1116||nt==2116)nt=0,fg=-1;
	if(nt==1516||nt==2516)nt=2;
	fuc(fr);s1+=':';
	for(int i=0;i<26;i++)if((c1>>i)&1)s1+=fu[i];
	for(int i=0;i<26;i++)if((c2>>i)&1)s1+=fu[i]+'A'-'a';
	s1+=':';fuc(nt);s1+=':';
	for(int i=0;i<26;i++)if((vl>>i)&1)s1+=fu[i];
	s1+=':';
	if(fg!=-1)s1+="plr"[fg];
	as.push_back(s1);
}
void solve()
{
	as.push_back("1");
	doit(0,1<<23,1<<18,0,1<<18,1);
	doit(0,1<<24,1<<20,0,1<<20,2);
	doit(0,(1<<23)|(1<<18),1<<19,1000,1<<19,-1);
	doit(1000,1<<17,0,1100,1<<17,1);
	doit(1000,0,0,1100,0,1);
	for(int i=0;i<16;i++)
	{
		doit(1100+i,1<<i,0,1300+i,0,0);
		doit(1100+i,0,1<<i,1200+i,0,0);
		doit(1200+i,1<<i,1<<17,1116,0,1);
		doit(1200+i,1<<i,0,1101+i,1<<i,1);
		doit(1300+i,1<<17,1<<i,1101+i,1<<i,1);
		doit(1300+i,0,1<<i,1101+i,(1<<i)|(1<<17),1);
		doit(1200+i,0,0,1101+i,0,1);
		doit(1300+i,0,0,1101+i,0,1);
	}
	doit(0,(1<<24)|(1<<20),1<<21,2000,1<<21,-1);
	doit(2000,1<<17,0,2100,1<<17,2);
	doit(2000,0,0,2100,0,2);
	for(int i=0;i<16;i++)
	{
		doit(2100+i,1<<i,0,2300+i,0,0);
		doit(2100+i,0,1<<i,2200+i,0,0);
		doit(2200+i,1<<i,1<<17,2116,0,2);
		doit(2200+i,1<<i,0,2101+i,1<<i,2);
		doit(2300+i,1<<17,1<<i,2101+i,(1<<i),2);
		doit(2300+i,0,1<<i,2101+i,(1<<i)|(1<<17),2);
		doit(2200+i,0,0,2101+i,0,2);
		doit(2300+i,0,0,2101+i,0,2);
	}
	doit(0,0,(1<<23)|(1<<19),0,1<<19,-1);
	doit(0,0,(1<<24)|(1<<21),0,1<<21,-1);
	doit(0,(1<<21)|(1<<19),0,3000,0,-1);
	for(int i=1;i<=16;i++)
	doit(3000,(1<<16)-(1<<16-i+1),1<<16-i,3001,(1<<16)-(1<<16-i),-1);
	doit(3001,1<<25,0,0,0,0);
	doit(3001,0,0,1,0,-1);
	for(int i=1;i<=16;i++)
	doit(1,1<<16-i,(1<<16)-(1<<16-i+1),2,(1<<16)-(1<<16-i),-1);
	doit(2,(1<<23)|(1<<19),0,1500,1<<19,-1);
	for(int i=0;i<16;i++)
	{
		doit(1500+i,1<<i,0,1600+i,0,1);
		doit(1500+i,0,1<<i,1700+i,0,1);
		doit(1600+i,1<<i,0,1501+i,0,0);
		doit(1600+i,0,1<<i,1501+i,1<<i,0);
		doit(1700+i,1<<i,0,1501+i,1<<i,0);
		doit(1700+i,0,1<<i,1501+i,0,0);
	}
	doit(2,(1<<24)|(1<<21),0,2500,1<<21,-1);
	for(int i=0;i<16;i++)
	{
		doit(2500+i,1<<i,0,2600+i,0,2);
		doit(2500+i,0,1<<i,2700+i,0,2);
		doit(2600+i,1<<i,0,2501+i,0,0);
		doit(2600+i,0,1<<i,2501+i,1<<i,0);
		doit(2700+i,1<<i,0,2501+i,1<<i,0);
		doit(2700+i,0,1<<i,2501+i,0,0);
	}
	doit(2,(1<<23)|(1<<18),0,1,1<<18,1);
	doit(2,(1<<24)|(1<<20),0,1,1<<20,2);
	doit(2,0,((1<<16)-1)|(1<<22),2,1<<22,-1);
	doit(2,0,(1<<18)|(1<<20),2,0,0);
}
struct BinaryTreeAutomatonHard{
	vector<string> construct(int n)
	{
		solve();
		return as;
	}
};
```

##### SRM479 TC11032 TheBoardingDivOne

###### Problem

$n$ 个人在一个 $1\times 2n$ 的网格中，有一个 $n$ 阶排列 $p$，初始第 $i$ 个人在位置 $i$，他的目标位置为 $p_i+n$。

每个时刻，从当前最右侧的人开始，如果一个人还没有到达目标位置且他右边是空位，则他会向右移动一步。如果他到达了目标位置，则他会在这里停留 $74$ 时刻，随后消失。

定义总共需要的时间为最后一个消失的人消失的时间，现在 $p$ 中一部分位置已经确定，求有多少种填满剩下位置的方案，使得总共需要的时间不超过 $t$。

$n\leq 18,t\leq 222$

###### Sol

注意到 $222=3*74$，考虑整个过程，因为 $74$ 远大于 $n$，一开始一定是一些人到达位置开始等待，后面的人排在这些人的后面。如果剩下的人的目标位置不递增，则前面的人消失后，这些人向右移动的过程中，一定有一个人被另外一个到达目标位置的人阻挡 $74$ 时刻，因此最后时刻一定大于 $3*74$。

因此除去能在前 $74$ 时刻到达的人外，剩下的人编号一定递增。考虑如何判定一个排列 $p$ 是否合法。首先考虑找到第一轮能到达的人，可以从后往前考虑，记录后面所有人在第一轮中停留的最靠前的位置 $k$。如果当前加入一个目标位置为 $p_i+n$ 的位置，则如果 $p_i+n<k$，则这个人可以第一轮到达，令 $k=p_i+n$。否则，这个人的目标位置必须是左侧所有目标位置中最大的，且此时 $k$ 等于之前的 $k$ 减一。

再考虑计算用时。第一轮的人用时容易求出。但求出第二轮的人的用时需要知道这个人右侧第一个在第一轮到达的人，在dp中加入这一维复杂度过大。但可以发现，如果一个第一轮到达的人阻挡了若干个人，因为这些人的目标位置递增，因此一定是阻挡的第一个人用时最大，只需要考虑这个人。在加入第一轮的人时，如果这个人会阻挡人，则这个人阻挡的第一个人一定是剩下的人中目标位置最大的人。而如果他的目标位置大于当前人的目标位置，则这个人一定会被阻挡，他在之后被阻挡导致的用时大于他在现在被阻挡的用时。因此可以对于每个第一轮到达的人，如果这个人左侧目标位置最大的人大于这个人的目标位置，则计算一次左侧这个人出现在左侧下一个位置时的用时。这样计算出的最大用时一定等于最大用时，且不需要额外记录状态。

因此只需要记录 $dp_{S,k}$ 表示当前后面已经考虑的人的目标位置集合为 $S$，当前最后一个人停留的位置为 $k$ 的方案数，使用上面的方式判断转移。复杂度 $O(n2^n)$。

直接开空间可能超过 $64MB$，但注意到只需要记录 $k\in[n+1,2n]$ 的位置，以及答案实际上不超过 $10^7$，即可大大减小空间常数。

 ###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define ll long long
int n,t,cr[263001],hbit[263001];
ll dp[263001][19],as;
struct TheBoardingDivOne{
	ll find(vector<int> p,int t)
	{
		n=p.size();
		dp[0][0]=1;
		for(int i=1;i<1<<n;i++)cr[i]=cr[i-(i&-i)]+1;
		for(int i=1;i<1<<n;i++)hbit[i]=hbit[i>>1]+1;
		for(int i=0;i+1<1<<n;i++)for(int j=0;j<=n;j++)if(dp[i][j])
		{
			int c1=cr[i]+1;
			for(int k=1;k<=n;k++)if(!((i>>k-1)&1)&&(p[n-c1]==-1||p[n-c1]==k))
			{
				int ti=0,ns=i|(1<<k-1),nj=j?j:n+1;
				if(k<nj)
				{
					ti=k+c1+73,nj=k;
					if(ns!=(1<<n)-(1<<k-1))
					{
						int rs=hbit[((1<<n)-1)^ns];
						ti=148+c1+rs;
					}
				}
				else
				{
					nj=nj>1?nj-1:1;
					if((ns>>k-1)!=(1<<n-k+1)-1)ti=1e9;
				}
				if(ti<=t)dp[ns][nj]+=dp[i][j];
			}
		}
		for(int i=0;i<=n;i++)as+=dp[(1<<n)-1][i];
		return as;
	}
};
```

##### SRM510 TC11465 TheLuckyBasesDivOne

###### Problem

给出数 $n$ ，求有多少个大于 $1$ 的整数 $b$，使得 $n$ 在 $b$ 进制的表示下，每一位的值在十进制表示下每一位都是 $4$ 或 $7$。有无限个输出 $-1$。

$n\leq 10^{16}$

###### Sol

如果 $n$ 的十进制表示只有 $4,7$，则显然答案为 $-1$，否则一定有 $b\leq n$，从而答案有限。

考虑枚举 $b\leq n^{\frac 13}$ 的情况，此时剩下的情况位数只可能是 $2,3$。

对于位数是 $2$ 的情况，考虑直接枚举两位的结果算答案。设枚举的位是 $x,y$，则要求 $xb+y=n$ 且 $x,y<b$，这意味着 $x\leq\sqrt n$ 且 $xy\leq n$。又因为可能的数只有 $2^{\log n}=n^{\ln 2}$ 个，因此这里考虑暴力枚举然后 $O(1)$ 计算，这样不难得到复杂度为 $O(n^{\ln 2}\log n)$。

对于位数是 $3$ 的情况，考虑暴力枚举前两位，则第一位不超过 $n^{\frac 13}$，第二位不超过 $\sqrt n$，因此这部分复杂度不超过 $O(n^{\frac 56\ln 2})$。此时相当于 $xb^2+yb+z=n$，要求 $0\leq z<b$。但可以发现因为 $x,y>0$，$b$ 改变 $1$ 时，$z$ 一定会改变至少 $2b$，因此最多有一个合法的 $z$，可以二分出这个值，然后暴力判断。这部分复杂度小于上一部分。~~事实上存在合法b的方案只有几十种，这里随便做都可以~~

总复杂度 $O(n^{\frac 13}\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<cmath>
using namespace std;
#define ll long long
ll n,as,cr,li=3e5;
vector<ll> ls;
ll solve(ll n,ll a,ll b)
{
	ll lb=1,rb=sqrt(n/a)+2,as=0;
	while(lb<=rb)
	{
		ll mid=(lb+rb)>>1;
		if(a*mid*mid+b*mid<=n)as=mid,lb=mid+1;
		else rb=mid-1;
	}
	return as;
}
struct TheLuckyBasesDivOne{
	ll find(ll n)
	{
		ls.push_back(4);ls.push_back(7);
		for(int i=0;i<ls.size();i++)
		{
			if(ls[i]==n)return -1;
			if(ls[i]*10+4<=n)ls.push_back(ls[i]*10+4);
			if(ls[i]*10+7<=n)ls.push_back(ls[i]*10+7);
		}
		for(int i=2;i<=li;i++)
		{
			ll tp=n,fg=1;
			while(tp)
			{
				ll st=tp%i;tp/=i;
				vector<ll>::iterator it=lower_bound(ls.begin(),ls.end(),st);
				if(it==ls.end()||(*it)!=st){fg=0;break;}
			}
			as+=fg;
		}
		for(int i=0;i<ls.size()&&ls[i]*ls[i]<=n;i++)
		{
			ll lb=n/(ls[i]+1)+1,rb=n/ls[i];
			if(lb<=ls[i])lb=ls[i]+1;
			if(lb<=li)lb=li+1;
			if(lb>rb)continue;
			for(int j=0;j<ls.size()&&ls[j]<rb;j++)
			{
				ll l1=max(lb,ls[j]+1),r1=rb;
				ll nw=n-ls[j];if(nw%ls[i]!=0)continue;
				nw/=ls[i];if(nw>=l1&&nw<=r1)as++;
			}
		}
		for(int i=0;i<ls.size()&&ls[i]*ls[i]*ls[i]<=n;i++)
		for(int j=0;j<ls.size()&&ls[j]*ls[j]<=n;j++)
		{
			ll lb=solve(n,ls[i],ls[j]+1)+1,rb=solve(n,ls[i],ls[j]);
			if(lb<=ls[i])lb=ls[i]+1;if(lb<=ls[j])lb=ls[j]+1;
			if(lb<=li)lb=li+1;
			if(lb>rb)continue;
			while(lb<=rb)
			{
				ll tp=n%lb,fg=1;
				while(tp)fg&=tp%10==4||tp%10==7,tp/=10;
				as+=fg;
				lb++;
			}
		}
		return as;
	}
};
```

##### SRM453 TC10689 TheSoccerDivOne

###### Problem

有 $n$ 支球队进行比赛，现在每支球队正好剩下四场比赛没有打，这些比赛的安排可以是任意的，也可以两支球队打多次。胜者得 $3$ 分，负者不得分，平局双方各得 $1$ 分。

给出目前每支球队的分数 $v_i$，定义最后一个队的排名为分数严格高于它的球队数量。求所有情况中，第一支球队的最高排名。

$n\leq 50$

###### Sol

第一支球队显然需要全部胜利，然后直接的想法是依次加入球队，考虑前面的球队和后面需要打的那些场数。设 $dp_{i,a,b,c}$ 表示考虑了前 $i$ 支球队，前面与后面的比赛中前面有 $a$ 胜 $b$ 平 $c$ 负时，前面第一支球队能在分数上不低于的球队数量的最大值。转移可以枚举当前队比赛的情况。

这样复杂度为 $O(n^4)$，但转移常数至少为 $126$，不能通过。

但可以注意到，平局部分可以看成一个匹配，而这里的匹配可以贪心，因为每个队平局数量不超过 $4$，可以只保留 $b\leq 4$ 的状态，不难发现这样保留了一种正确方案。复杂度 $O(n^3)$，可以通过。

同样的，使用与上面类似的过程可以只保留 $\min(a,c)\leq 4$ 的状态。复杂度降至 $O(n^2)$。

最后，注意到如果考虑每个人的胜场减去负场，则 $a,c$ 整体中的最大值为这个值的前缀和的 $max$ 和 $-min$ 中最大的一个。因此可以shuffle序列，使最优解中 $a,c$ 的整体最大值不超过 $O(\sqrt n)$，从而复杂度 $O(n\sqrt n)$，但在这里效果不大。

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 51
int n,v[N],dp[N*2][N*2][5],dp2[N*2][N*2][5],ct;
struct sth{int a,b,c,d,e,f;}tp[N*3];
struct TheSoccerDivOne{
	int find(vector<int> v)
	{
		for(int a=0;a<=4;a++)for(int b=0;b<=4;b++)for(int c=0;c<=4;c++)if(a+b+c<=4)
		for(int d=0;d<=4;d++)for(int e=0;e<=4;e++)for(int f=0;f<=4;f++)if(a+b+c+d+e+f==4)tp[++ct]=(sth){a,b,c,d,e,f};
		n=v.size();
		for(int i=0;i<=n*2;i++)for(int j=0;j<=n*2;j++)for(int k=0;k<=4;k++)if(i+j+k)dp[i][j][k]=-100;
		for(int t=0;t<n;t++)
		{
			for(int i=0;i<=n*2;i++)for(int j=0;j<=(i<=4?n*2:4);j++)for(int k=0;k<=4;k++)dp2[i][j][k]=-100;
			for(int i=0;i<=n*2;i++)for(int j=0;j<=(i<=4?n*2:4);j++)for(int k=0;k<=4;k++)if(dp[i][j][k]>=0)
			for(int s=1;s<=ct;s++)
			{
				if(i<tp[s].b||j<tp[s].a||k<tp[s].c)continue;
				int na=i-tp[s].b+tp[s].d,nb=j-tp[s].a+tp[s].e,nc=k-tp[s].c+tp[s].f;
				if(na>n*2||nb>n*2||nc>4)continue;
				int sc=v[t]+(tp[s].a+tp[s].d)*3+tp[s].c+tp[s].f,nv=dp[i][j][k]+(sc<=v[0]+12);
				if(dp2[na][nb][nc]<nv)dp2[na][nb][nc]=nv;
			}
			for(int i=0;i<=n*2;i++)for(int j=0;j<=(i<=4?n*2:4);j++)for(int k=0;k<=4;k++)dp[i][j][k]=dp2[i][j][k];
		}
		return n+1-dp[0][0][0];
	}
};
```

##### TCO15 Round2C TC13897 PopcountRobot

###### Problem

有一个机器人，它初始在 $(0,0)$。

接下来它会进行 $T-1$ 步移动，移动从 $0$ 开始标号，在第 $i$ 步移动中，设 $s_i$ 为 $i$ 的二进制表示中 $1$ 的位数，则：

如果 $s_i\equiv 0(\bmod 4)$，则让 $x$ 加一。

如果 $s_i\equiv 1(\bmod 4)$，则让 $y$ 加一。

如果 $s_i\equiv 2(\bmod 4)$，则让 $x$ 减一。

如果 $s_i\equiv 3(\bmod 4)$，则让 $y$ 减一。

$q$ 次询问，每次给出一个两维在 $[-m,m]$ 间的坐标，询问这个坐标有没有被机器人经过过。因为IO原因询问随机生成，输出所有询问结果的和。

$T\leq 10^{18},m\leq 10^9,q\leq 5\times 10^5$

###### Sol

类似题目：Dragon Curve~~但我忘了补WF Invitational题解~~

注意到这个移动相当于初始移动方向为 $(1,0)$，每个二进制为 $1$ 的位相当于让移动逆时针旋转 $90$ 度。

因此 $2^i$ 步的行走过程可以看成，先走 $2^{i-1}$ 步，逆时针转 $90$ 度再走 $2^{i-1}$ 步。

因此可以发现走 $2^i$ 步后移动的向量角度为 $45*i$ 度，向量长度为 $(\sqrt2)^i$（即 $(1,0),(1,1),(2,0),(-2,2),(-4,0),(-4,-4),\cdots$。

因此对于一个 $t$，$t$ 步后机器人的位置容易求出，只需要先走 $t$ 最大的二进制位，再逆时针旋转后走剩下的。

如果没有旋转，则从低位开始考虑，可以发现对于一个 $i$，$2i$ 以及之后位对应的向量两维都是 $2^i$ 的倍数，因此在低位确定的情况下，$2i-1$ 这一位可以唯一确定。又可以发现对于一个 $i$，$2i-1$ 以及之后位对应的向量两维之和都是 $2^i$ 的倍数，而 $2i-2$ 的向量一维绝对值为 $2^{i-1}$，另外一维为 $0$。因此低位也可以唯一确定它。因此没有旋转的情况下可以唯一确定一个经过它的时刻或得到无解。

再考虑有旋转的情况，旋转是高位影响低位，这难以处理，考虑先枚举 $s_t\bmod 4$ 的结果，这样总旋转次数固定，就可以看成每次选择了一个低位之后，高位的向量顺时针旋转 $90$ 度（反过来旋转）。这样枚举后每一类情况求出来的解唯一，可以直接判断是否合法。这也说明了每个位置最多被经过四次。

复杂度 $O(q\log n)$，常数较大。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
int q,m,x,y,as;
ll ti;
ll solve(ll x,ll y,ll sx,ll sy)
{
	ll as=0;
	for(int i=0;i<=62;i++)
	{
		int fg=0;
		if((i&1)&&(x&((1ll<<((i+1)/2))-1)))fg=1;
		if((~i&1)&&((x+y)&((1ll<<((i+2)/2))-1)))fg=1;
		if(fg)as+=1ll<<i,x-=sx,y-=sy,sx^=sy^=sx^=sy,sy*=-1;
		sy+=sx;sx=sx*2-sy;
	}
	return as;
}
bool chk(int x,int y,ll ti)
{
	int sx=1,sy=0;
	for(int t=0;t<4;t++)
	{
		ll st=solve(x,y,sx,sy);
		if(st<=ti)
		{
			int ct=0;
			while(st)ct++,st-=st&-st;
			if(ct%4==(t+1)%4)return 1;
		}
		sx^=sy^=sx^=sy,sx*=-1;
	}
	return 0;
}
struct PopcountRobot{
	int countBlack(ll ti,int q,int m,int x,int y)
	{
		ti--;
		while(q--)as+=chk(x,y,ti),x=(1ll*(x+m)*7180087+5205425)%(2*m+1)-m,y=(1ll*(y+m)*6132773+9326231)%(2*m+1)-m;
		return as;
	}
};
```

##### SRM663 TC13893 WarAndPeas

###### Problem

有 $n$ 张牌，第 $i$ 张牌上数字为 $i$。两个人进行如下游戏：

游戏开始时，每张牌被随机发给两个人中的一个。随后循环如下过程：

如果当前所有牌都属于同一个人，则游戏结束。否则，在所有满足 $1\leq i<j\leq n$ 的 $(i,j)$ 中随机选择一对，并将数字为 $j$ 的牌移动到持有数字 $i$ 的牌的人手中（可能不会进行任何操作）。

给出一个状态 $S$，求每一次操作后以及游戏开始时，状态为 $S$ 的时刻的期望，模 $10^9+7$。

$n\leq 1000$

###### Sol

特判 $S$ 为所有牌在同一个人手中的情况（答案为 $\frac 12$），对于不是这种情况的问题，可以认为游戏不会停止，求此时的期望经过次数。

可以观察出如下结论：对于任意一个 $i$，所有第一个人手牌数为 $i$ 的状态满足对于任意 $t$，$t$ 次操作后当前状态为这些状态的概率相等。

考虑归纳，$t=0$ 显然成立，如果 $t$ 成立，考虑 $t+1$ 的情况。对于一个 $i$，可以发现这个状态可能从上一个时刻第一个人手牌数为 $i-1,i,i+1$ 的状态转移过来，此时有：

1. 对于从 $i$ 转移过来的情况，只可能是自己到自己。而这种情况的概率是选两个在同一个人手中牌的概率，即为 $\frac{C_i^2+C_{n-i}^2}{C_n^2}$，因此每个 $i$ 这部分概率相同。
2. 对于从 $i-1$ 转移过来的情况，则这次操作选择的两个数一定现在都在第一个人手中，而每一个选择第一个人手中两张牌的方案都可以对应一种 $i-1$ 的情况以及转移，因此这部分系数为 $C_i^2$，只和 $i$ 有关。
3. 从 $i+1$ 转移的情况同理。

因此对于任意 $t$ 上述结论都成立，因此经过它们的期望显然也相等。

设 $f_i$ 表示经过所有第一个人手牌数为 $i$ 的状态的期望次数总和。则 $f_{i-1}$ 转移到 $f_i$ 时，需要计算 $f_{i-1}$ 中每个状态转移到 $i$ 的期望次数。这可以直接计算所有 $i-1$ 个 $1$，$n-i+1$ 个 $0$ 的字符串中 $10$ 子序列的期望，也可以用总转移数 $C_i^2*C_n^i$ 除以状态数 $C_n^{i-1}$，两者结果都是 $\frac{(n-i+1)(i-1)}2$。

因此可以得到如下方程：$f_i=\frac{C_n^i}{2^n}+\frac{C_i^2+C_{n-i}^2}{C_n^2}f_i+\frac{(n-i+1)(i-1)}2f_{i-1}+\frac{(i+1)(n-i-1)}2f_{i+1}$

直接消元复杂度 $O(n^3)$，可以通过。这里也可以考虑将 $f_i$ 表示为 $a*f_{i+1}+b$，从左向右处理这个形式，再从右向左解出答案，以此做到 $O(n\log mod)$ 的复杂度。

###### Code

```cpp
#include<cstdio>
#include<string>
using namespace std;
#define N 1050
#define mod 1000000007
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int n,ct,sc[N],sl[N],sr[N],sv[N],fr[N],fv[N];
struct WarAndPeas{
	int expectedPeas(string s)
	{
		n=s.size();
		for(int i=0;i<n;i++)ct+=s[i]=='A';
		if(ct==0||ct==n)return (mod+1)/2;
		sc[0]=1;for(int i=1;i<=n;i++)sc[i]=1ll*sc[i-1]*(n-i+1)%mod*pw(i,mod-2)%mod;
		for(int i=1;i<n;i++)sl[i]=(1ll*(n-1)*(mod+1)/2%mod*(i-1)-1ll*(i-1)*(i-2)/2%mod+mod)%mod,sl[i]=1ll*sl[i]*pw(n*(n-1)/2,mod-2)%mod;
		for(int i=1;i<n;i++)sr[i]=sl[n-i],sv[i]=1ll*sc[i]*pw(2,mod-1-n)%mod;
		for(int i=1;i<n;i++)
		{
			int pr=1ll*(i*(i-1)+(n-i)*(n-i-1))*pw(n*(n-1),mod-2)%mod;pr=pw(mod+1-pr,mod-2);
			sl[i]=1ll*sl[i]*pr%mod;sr[i]=1ll*sr[i]*pr%mod;sv[i]=1ll*sv[i]*pr%mod;
		}
		for(int i=1;i<n;i++)
		{
			fr[i]=sr[i];fv[i]=sv[i];
			fv[i]=(fv[i]+1ll*sl[i]*fv[i-1])%mod;
			int pr=pw(mod+1-1ll*sl[i]*fr[i-1]%mod,mod-2);
			fr[i]=1ll*fr[i]*pr%mod;fv[i]=1ll*fv[i]*pr%mod;
		}
		for(int i=n-1;i>=1;i--)fv[i]=(fv[i]+1ll*fv[i+1]*fr[i])%mod;
		return 1ll*fv[ct]*pw(sc[ct],mod-2)%mod;
	}
};
```

##### SRM776 TC15754 ThreeColorTrees

###### Problem

给出 $n$，你需要构造一棵 $n$ 个点的树，使得：

考虑对这棵树的所有节点染色，一共有三种颜色，要求任意两个染了第一种颜色的节点距离大于 $2$，没有其它要求。

这棵树的染色方案数大于等于 $2.62^n$。

$n\leq 1000$

###### Sol

做法一：

考虑根节点下面挂若干个相同的子树，则只要每个子树满足子树根节点不是第一种颜色且子树内合法，则这样的方案合法。

考虑放 $n=2$ 的子树，则方案数为 $6$，方案数近似值为 $\sqrt6^n$，但这只有 $2.44$。

考虑放 $n=3$ 的二叉树，方案数为 $18$，方案数大约为 $2.52^n$。

因此考虑继续找，考虑一个 $n=7$ 的子树，其中根节点挂三个 $n=2$ 的子树，这样的方案数为 $864$，近似值为 $2.627^n$，特判小的情况就可以搞过去。

复杂度 $O(n)$

做法二：

考虑计算染色方案数，只需要对于每个子树求子树内第一种颜色的点到根的距离为 $0,1,\geq 2$ 的方案数，就能进行合并。

考虑对这个状态找一种比较方式然后直接 $dp$，一种方式是按照三个值的和比较，这种方式可以做到 $2.59^n$。上一种方式的问题在于距离为 $0$ 的方案实际上不那么优秀，因此考虑按照 $k*(a+b+c)+b+2c$ 的大小比较，其中 $a,b,c$ 分别为距离为 $0,1,\geq 2$ 的方案数。任意取一个 $k$ 都可以稳定做到 $2.637^n$，可以通过。记录转移点即可还原方案。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 2333
int n,fr[N],fa[N];
struct sth{long double v[3];sth(){v[0]=v[1]=v[2]=0;}}dp[N];
sth operator +(sth a,sth b)
{
	sth as;
	for(int i=0;i<3;i++)for(int j=0;j<3;j++)if(i+j>=2)as.v[i>=j+1?j+1:i]+=a.v[i]*b.v[j];
	return as;
}
bool operator <(sth a,sth b)
{
	long double sa=0,sb=0;
	for(int i=0;i<3;i++)sa+=a.v[i]*(10+i),sb+=b.v[i]*(10+i);
	return sa<sb;
}
void solve(int l,int n)
{
	if(n==1)return;
	fa[l+fr[n]]=l;
	solve(l,fr[n]);solve(l+fr[n],n-fr[n]);
}
struct ThreeColorTrees{
	vector<int> construct(int n)
	{
		dp[1].v[0]=1;dp[1].v[2]=2;
		for(int i=2;i<=n;i++)
		for(int j=1;j<i;j++)
		if(dp[i]<dp[j]+dp[i-j])dp[i]=dp[j]+dp[i-j],fr[i]=j;
		solve(1,n);
		vector<int> as;
		for(int i=2;i<=n;i++)as.push_back(fa[i]-1);
		return as;
	}
};
```

##### SRM368 TC8247 BinaryCodes

###### Problem

给定字符集大小为 $n$，现在每个字符 $i$ 对应了一个长度不超过 $m$ 的 $01$ 编码 $s_i$，一个字符串对应的编码即为每个字符的编码按顺序拼接的结果。

找到一个最小长度，使得存在一个这个长度的 $01$ 串能表示为三个不同的字符串的编码，或输出不存在这样的长度。

$n\leq 30,m\leq 50$

###### Sol

使用AC自动机可以得到一个 $O((nm)^3)$ 的暴力，但难以优化。

先考虑没有不同的条件时填的过程，相当于用这 $m$ 个串拼接出三个相同的字符串。

考虑每次向当前长度最短的字符串加入一个串，则有如下结论：

任意时刻，状态都可以被表示为，存在一个串 $s_i$，当前最长的串以 $s_i$ 结尾，另外两个串以 $s_i$ 的一个前缀结尾。

证明：考虑加入的过程，如果加入后这个串没有变成最长，则之前的结论仍然成立。如果加入后变成最长，则因为原先另外两个串都不短于它，因此此时另外两个串一定至少长度到了当前串除去最后一个串外的部分，因此此时仍然可以使用上述方式表示。

此时可以设 $g_{i,l_1,l_2}$ 表示最长的串以 $s_i$ 结尾，另外两个串以长度为 $l_1,l_2$ 的前缀结尾，达到这个状态时最长串的最短长度。转移可以枚举下一个加入的串，转移可以使用最短路形式实现，复杂度 $O((nm)^2\log nm)$。

再考虑如何加入不同的条件。如果三个串不同，则要么在第一个位置三个串的字符全部不同，要么第一个位置分出两种情况，其中一种情况再分为两种。（如果第一个字符全部相同，则删去这个字符更优）。对于第一种情况，可以直接枚举开头三个字符做。对于第二种情况，相当于做两个串的dp转移过来，因此设 $f_{i,l}$ 表示长的串以 $s_i$ 结尾，另外一个串以长度为 $l_1$ 的前缀结尾，达到这个状态时最长串的最短长度。转移 $f$ 的复杂度为 $O(n^2m\log nm)$，然后从每个 $f$ 开始，枚举短串分为两种情况分别用的字符即可转移到 $g$。

总复杂度 $O(n^3m+n^2m^2\log nm)$

~~然而好像答案只有300，所以奇怪做法都能过~~

~~然后上面那个东西跑了19ms~~

###### Code

```cpp
#include<cstdio>
#include<string>
#include<queue>
#include<vector>
using namespace std;
#define N 54
#define ll long long
int n,le[N],f[N][N],g[N][N][N],visf[N][N],visg[N][N][N];
ll vl[N];
char s[N][N];
ll calc(int x,int l,int r){return (vl[x]&((1ll<<r)-1))>>(l-1);}
priority_queue<pair<int,int> > qu;
struct BinaryCodes{
	int ambiguous(vector<string> s1)
	{
		n=s1.size();
		for(int i=1;i<=n;i++)le[i]=s1[i-1].size();
		for(int i=1;i<=n;i++)if(le[i]==0)return 0;
		for(int i=1;i<=n;i++)for(int j=1;j<=le[i];j++)vl[i]+=((ll)s1[i-1][j-1]-'0')<<(j-1);
		for(int i=1;i<=n;i++)for(int j=0;j<=50;j++)f[i][j]=1.1e9;
		for(int i=1;i<=n;i++)for(int j=0;j<=50;j++)for(int k=0;k<=50;k++)g[i][j][k]=1.1e9;
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(i!=j&&le[i]<=le[j])
		{
			int lc=le[i];
			if(calc(i,1,lc)==calc(j,1,lc))f[j][lc]=le[j];
		}
		for(int i=1;i<=n;i++)for(int j=0;j<=50;j++)if(f[i][j]<=1e9)qu.push(make_pair(-f[i][j],i*100+j));
		while(!qu.empty())
		{
			int tp=qu.top().second;qu.pop();
			int lx=tp/100,ly=tp%100;if(visf[lx][ly])continue;visf[lx][ly]=1;
			if(le[lx]==ly)continue;
			for(int i=1;i<=n;i++)
			{
				int l1=min(le[lx]-ly,le[i]);
				if(calc(i,1,l1)!=calc(lx,ly+1,ly+l1))continue;
				int nx=lx,ny=ly+l1,nd=f[lx][ly];
				if(ny==le[lx])nx=i,ny=l1,nd+=le[i]-l1;
				if(f[nx][ny]>nd)f[nx][ny]=nd,qu.push(make_pair(-nd,nx*100+ny));
			}
		}
		for(int i=1;i<=n;i++)for(int j=0;j<=50;j++)if(f[i][j]<=1e9)
		for(int s=1;s<=n;s++)for(int t=1;t<=n;t++)if(s!=t&&le[s]<=le[t])
		{
			if(calc(s,1,le[s])!=calc(t,1,le[s]))continue;
			int nx=i,ny=j+le[s],nz=j+le[t],nd=f[i][j];
			int l1=min(le[i]-j,le[t]);
			if(l1&&calc(i,j+1,j+l1)!=calc(t,1,l1))continue;
			if(nz>le[nx])nx=t,ny=le[s],nz=le[i]-j,nd+=le[t]-l1;
			if(ny>nz)ny^=nz^=ny^=nz;
			g[nx][ny][nz]=min(g[nx][ny][nz],nd);
		}
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)if(i!=j&&j!=k&&i!=k&&le[i]<=le[j]&&le[j]<=le[k])
		{
			if(calc(i,1,le[i])!=calc(j,1,le[i])||calc(j,1,le[j])!=calc(k,1,le[j]))continue;
			int nx=k,ny=le[i],nz=le[j],nd=le[k];
			g[nx][ny][nz]=min(g[nx][ny][nz],nd);
		}
		for(int i=1;i<=n;i++)for(int j=0;j<=50;j++)for(int k=0;k<=50;k++)if(g[i][j][k]<=1e9)qu.push(make_pair(-g[i][j][k],i*10000+j*100+k));
		while(!qu.empty())
		{
			int tp=qu.top().second;qu.pop();
			int lx=tp/10000,ly=tp/100%100,lz=tp%100;
			if(visg[lx][ly][lz])continue;visg[lx][ly][lz]=1;
			if(ly==le[lx])return g[lx][ly][lz];
			for(int i=1;i<=n;i++)
			{
				int l1=min(le[lx]-ly,le[i]);
				if(calc(i,1,l1)!=calc(lx,ly+1,ly+l1))continue;
				int nx=lx,ny=ly+le[i],nz=lz,nd=g[lx][ly][lz];
				if(ny>le[lx])nx=i,ny=l1,nz=lz-ly,nd+=le[i]-l1;
				if(ny>nz)ny^=nz^=ny^=nz;
				if(g[nx][ny][nz]>nd)g[nx][ny][nz]=nd,qu.push(make_pair(-nd,nx*10000+ny*100+nz));
			}
		}
		return -1;
	}
};
```

##### SRM674 TC13859 ClassicProblem

###### Problem

有 $n$ 种物品，第 $i$ 种物品有 $a_i$ 个，单个物品重量为 $w_i$，价值为 $v_i$。

选一些物品，重量不超过 $m$，求最大总价值。

$n,w_i\leq 80,a_i,v_i,m\leq 10^9$

###### Sol

结论：考虑贪心按照性价比从大往小选，则最优方案与这个方案每种物品选的数量不超过 $\max w_i$ 个。

证明：如果不满足这个条件，如果应该选的某一种少选了 $\max w_i$ 个以上，设这种物品的重量为 $w$，则至少选了 $w$ 个不优秀的物品。将这些物品排成一列，考虑前缀和模 $w$，由鸽巢原理可得一定存在物品的一个非空子集，子集重量和是 $w$ 的倍数。此时将这部分替换为全选这种物品更优。另外一种情况同理。

因此可以每种物品拿出 $\max w_i$ 个，这部分做 $dp$，剩下的直接贪心。

因为物品数量较多，$dp$ 可以使用单调队列优化，复杂度 $O(n^2w^2)$。最后枚举 $dp$ 部分选择的物品数量再贪心即可，复杂度与上一部分相同。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define ll long long
#define N 83
int n,m,s[N],w[N],v[N],su,st[523001],id[N];
ll vl[523001],dp[523001],as;
bool cmp(int a,int b){return 1ll*v[a]*w[b]>1ll*v[b]*w[a];}
struct ClassicProblem{
	ll maximalValue(vector<int> s1,vector<int> s2,vector<int> s3,int li)
	{
		n=s1.size();m=li;
		for(int i=1;i<=n;i++)s[i]=s1[i-1],w[i]=s2[i-1],v[i]=s3[i-1],id[i]=i;
		sort(id+1,id+n+1,cmp);
		for(int i=1;i<=n;i++)
		{
			int c1=s[i];if(c1>80)c1=80;s[i]-=c1;
			for(int r=0;r<w[i];r++)
			{
				int nw=0;
				int lb=1,rb=0;
				while(1)
				{
					if(lb<=rb&&st[lb]<nw-c1)lb++;
					while(lb<=rb&&vl[rb]<=dp[nw*w[i]+r]-1ll*nw*v[i])rb--;
					if(nw*w[i]+r<=su)st[++rb]=nw,vl[rb]=dp[nw*w[i]+r]-1ll*nw*v[i];
					if(lb>rb)break;
					dp[nw*w[i]+r]=vl[lb]+1ll*nw*v[i];
					nw++;
				}
			}
			su+=c1*w[i];
		}
		for(int i=0;i<=su&&i<=m;i++)
		{
			ll ls=m-i,vl=dp[i];
			for(int j=1;j<=n;j++)
			{
				ll tp=min(ls/w[id[j]],1ll*s[id[j]]);
				ls-=w[id[j]]*tp,vl+=v[id[j]]*tp;
			}
			if(as<vl)as=vl;
		}
		return as;
	}
};
```

##### SRM678 TC14118 ReturnOfTheJedi

###### Problem

给出 $n$ 对 $v_i,p_i$，对于每个 $d$，求出选择 $d$ 对数，这 $d$ 对数的 $(\sum v_i)*\prod p_i$ 的最小值。绝对或者相对误差不超过 $10^{-9}$

$n\leq 400,v_i\leq 10^9,10^{-5}\leq p_i\leq 1$，$p_i$ 为五位小数。

###### Sol

类似题目：SRM526.5 1C

考虑对 $p$ 取对数，则问题可以看成给出若干对 $(a,b)$，选择 $d$ 对使得 $f(\sum a,\sum b)$ 最小。其中 $f(a,b)=ae^b$。

可以发现对于任意正数 $t$，$f(x,y)=t$ 对应的曲线都是严格上凸的（$y=\ln t-\ln x$）。因此取最优解点 $(\sum a,\sum b)$ 对应的切线，这条切线满足其它所有解都在切线的严格上方。

这意味着存在一个 $k$，使得最优解满足它是所有方案中 $\sum v_i+k\sum \ln p_i$ 最小的。这里应该取 $k\in[0,+\infty)$。

因此对于一个 $k$，上述做法的结论是将所有数按照 $v_i+k\ln p_i$ 排序，选最小的 $d$ 个即可得到方案。考虑 $k$ 增大的过程，上述做法只关心这些元素的顺序。可以发现顺序只会改变 $O(n^2)$ 次，维护相邻两个元素的交换顺序的时刻即可 $O(n^2\log n)$ 维护。每次交换两个元素时，只会改变一个 $d$ 的方案，因此可以BIT/线段树 $O(\log n)$ 求出这个方案的权值。

复杂度可以做到 $O(n^2\log n)$~~但这里写了n^3~~

上述做法对于任意满足对于任意正数 $t$，$f(x,y)=t$ 对应的曲线都严格上凸的 $f$ 都成立。更经典的例子是 $f(x,y)=xy$，另外一个例子是 $f(x,y)=\frac{x^2}y$（结果是抛物线）。同时，上述取 $k$ 个的条件也可以加入需要满足一个拟阵的限制，由于拟阵满足可以贪心选最大权独立集，因此可以沿用上述做法。一个Binary Matroid的例子是SRM526.5，一个Graph Matroid的例子是loj3412

###### Code

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
#include<vector>
using namespace std;
#define N 405
int n,v[N],t[N],id[N];
double p[N],lp[N],as[N];
bool cmp(int a,int b){return v[a]<v[b];}
struct ReturnOfTheJedi{
	vector<double> minimalExpectation(vector<int> x,vector<int> s)
	{
		n=x.size();
		for(int i=1;i<=n;i++)v[i]=x[i-1],t[i]=s[i-1];
		for(int i=1;i<=n;i++)p[i]=t[i]*0.00001,lp[i]=log(p[i]),as[i]=1e18,id[i]=i;
		sort(id+1,id+n+1,cmp);
		while(1)
		{
			double s1=0,s2=1;
			for(int i=1;i<=n;i++)s1+=v[id[i]],s2*=p[id[i]],as[i]=min(as[i],s1*s2);
			double rb=1e21;
			int fr=0;
			for(int i=1;i<n;i++)if(p[id[i+1]]<p[id[i]])
			{
				double ti=(v[id[i+1]]-v[id[i]])/(p[id[i]]-p[id[i+1]]);
				if(ti<rb)rb=ti,fr=i;
			}
			if(rb>1e20)break;
			swap(id[fr],id[fr+1]);
		}
		vector<double> s1;
		for(int i=1;i<=n;i++)s1.push_back(as[i]);
		return s1;
	}
};
```

##### SRM542 TC11054 RabbitWorking

###### Problem

给一个 $n$ 个点的完全图，边有非负边权 $v_i$。

你需要选择一些点，最大化如下权值：

记你选择的点构成的团内的边权和为 $s$，选择点的个数为 $m$，则权值为 $\frac{s}{m(200-m)}$

求出最大权值。

$n\leq 50,0\leq v_i\leq 9$

###### Sol

考虑二分答案 $a$，相当于给选择 $m$ 个点的方案减去 $am(200-m)$。这可以看成给每条边加上 $2a$ 的权值，每个点给一个 $-199a$ 的权值。要求选择一个团，使得边权加上点权的和大于 $0$。

设选中的点集为 $S$，剩下的点集为 $T$，则代价为 $\sum_{a\in S} c_a+\sum_{a,b\in S,a<b}v_{a,b}$。直接搞难以处理，考虑将每个点的点权加上所有与它相邻的边权之和，则点权和变为：
$$
\sum_{a\in S} (c_a+\sum_{b}v_{a,b})\\
=\sum_{a\in S} c_a+2\sum_{a,b\in S,a<b}v_{a,b}+2\sum_{a\in S,b\in T}v_{a,b}
$$
因此考虑加上边权的一半，令 $c_a'=c_a+\frac12\sum_{b\neq a}v_{a,b}$，则代价变为：
$$
\sum_{a\in S} c'_a-\sum_{a\in S,b\in T}v_{a,b}
$$
这可以看成一个最小割，令 $S$ 为割集左侧部分，则考虑对于每对 $(a,b)$ 连边权为 $v_{a,b}$ 的双向边，则右侧减去部分即为最小割。

再考虑加入点权，使用最大权闭合子图的形式，如果 $c'_a\geq 0$，则源点向它连边权为 $c'_a$ 的边，表示不选会放弃 $c'_a$ 的收益。如果 $c'_a<0$，则它向汇点连边权为 $-c'_a$ 的边，表示选它有 $c'_a$ 的代价。最后有大于 $0$ 的方案当且仅当最小割小于 $\sum\max(c'_a,0)$。直接流即可。

复杂度 $O(dinic(n,n^2)*\log v)$，但是跑得非常快(11ms)。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 55
int n,ds[N],vis[N],cur[N];
double v[N],f[N][N];
char s[N][N];
bool bfs()
{
	for(int i=1;i<=n;i++)ds[i]=v[i]>1e-11?1:-1,vis[i]=0,cur[i]=1;
	for(int i=1;i<=n;i++)
	{
		int fr=0;
		for(int j=1;j<=n;j++)if(ds[j]!=-1&&!vis[j]&&(!fr||ds[j]<ds[fr]))fr=j;
		if(!fr)return 0;
		vis[fr]=1;if(v[fr]<-1e-11)return 1;
		for(int j=1;j<=n;j++)if(ds[j]==-1&&f[fr][j]>1e-11)ds[j]=ds[fr]+1;
	}
	return 0;
}
double dfs(int u,double fr)
{
	double as=0;
	if(v[u]<0)
	{
		double tp=min(-v[u],fr);
		as+=tp;v[u]+=tp;fr-=tp;
	}
	while(cur[u]<=n)
	{
		if(ds[cur[u]]==ds[u]+1)
		{
			double tp=dfs(cur[u],min(fr,f[u][cur[u]]));
			f[u][cur[u]]-=tp;f[cur[u]][u]+=tp;fr-=tp;as+=tp;
		}
		if(fr<1e-11)return as;
		cur[u]++;
	}
	return as;
}
bool chk(double r)
{
	for(int i=1;i<=n;i++)v[i]=-200*r*2;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)f[i][j]=s[i][j]-'0'+r*2,v[i]+=f[i][j];
	double su=0;
	for(int i=1;i<=n;i++)if(v[i]>0)su+=v[i];
	while(bfs())
	for(int i=1;i<=n;i++)if(ds[i]==1)
	{
		double tp=dfs(i,min(su,v[i]));
		v[i]-=tp;su-=tp;
	}
	return su>1e-7;
}
struct RabbitWorking{
	double getMaximum(vector<string> p) 
	{
		n=p.size();
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=p[i-1][j-1];
		double lb=0,rb=2;
		for(int t=1;t<=50;t++)
		{
			double mid=(lb+rb)/2;
			if(chk(mid))lb=mid;
			else rb=mid;
		}
		return lb;
	}
};
```
