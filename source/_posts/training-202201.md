---
title: 2022/01 集训题解
date: '2022-01-28 22:18:03'
updated: '2022-01-28 22:18:03'
tags: Mildia
permalink: Hanawataruchi/
description: 2022/01 南京集训
mathjax: true
---

随机顺序

Source鸽了

##### Just Add One Edge

###### Problem

给一个 $n$ 个点 $m$ 条边的DAG，保证 $1,2,...,n$ 为拓扑序。

求有多少对 $(x,y)$ 满足 $x>y$，且加入一条有向边 $x\to y$ 后，图中存在一条哈密顿链。

多组数据

$T\leq 5,n,m\leq 1.5\times 10^5$

$1s,256MB$

###### Sol

考虑图中哈密顿链的形状，如果哈密顿链不经过加入的边，那么原图中一定存在 $1\to 2\to...\to n$ 的链。如果存在这样的链，答案显然为 $\frac{n(n-1)}2$。

如果不存在这样的链，那么哈密顿链一定经过加入的边，此时从加入的边断开，可以得到两条路径，有以下几种情况：

1. $1<y<x<n$，此时路径为一条从 $1$ 到 $x$ 的路径和一条从 $y$ 到 $n$ 的路径。
2. $1<y<x=n$，此时路径为一条从 $1$ 到 $n$ 的路径，一条从 $y$ 出发的路径。
3. $1=y<x<n$，这个情况和上一种类似。
4. $1=y<x=n$，此时路径为一条从 $1$ 出发的路径，一条以 $n$ 结束的路径，这与第一种类似。

考虑第一种情况，此时需要用两条路径不重复地覆盖所有点。考虑将所有点 $i$ 被一条路径覆盖，$i+1$ 被另外一条路径覆盖的状态称为关键状态，那么一个关键状态 $i$ 到另外一个关键状态 $j$ 时，两条路径一定形如 $i\to j+1,i+1\to i+2\to...\to j$。

因此，对于原图中一条 $j-i>1$ 的边 $i\to j$，如果图中存在边 $i+1\to i+2\to...\to j-1$，则它对应一个关键状态的转移 $i\to j-1$。

考虑第一个关键状态 $l$ 和最后一个关键状态 $r$，则第一种情况的路径一定满足存在这两个关键状态，且满足如下条件：

1. 存在一条 $1\to 2\to...\to l$ 的链，一条路径以它开头，另外一条路径以 $l+1$ 开头。
2. 存在一条 $r+1\to r+2\to...\to n$ 的链，一条路径以它结尾，另外一条路径以 $r$ 结尾。
3. 从 $l$ 到 $r$ 间经过了奇数个关键状态。

此时第一种情况的情况数可以看成一个DAG上可达点对计数，但是 $O(\frac{n^2}{32})$ 过不去。

继续考虑原问题的性质，可以发现，如果不存在 $i\to i+1$ 的边，那么不会存在一个关键状态转移跨过 $i$，即不存在一个转移 $l\to r$ 使得 $l<i<r$。从所有这样的 $i$ 分开，则这个DAG可以看成若干个DAG拼接而成，其中每个DAG的最后一个点是下一个DAG的第一个点。

同时，因为前两条限制，合法路径的起点一定在第一个DAG中，终点一定在最后一个DAG中。又因为如果没有这样的 $i$，则答案为 $\frac{n(n-1)}2$，因此一定存在至少两个DAG。

此时只需要求出中间的每一个DAG中是否存在起点到终点长度为奇数/偶数的路径，再求出 $x$ 到第一个DAG的终点以及最后一个DAG的起点到 $y$ 是否存在长度为奇数/偶数的路径，即可求出 $x$ 到 $y$ 是否存在满足上面要求的路径。

每种奇偶性长度的路径是否存在一共只有四种情况，可以求出到达第一个DAG的终点的路径情况是上面四种中的每一种的点的数量，然后合并。这部分复杂度为 $O(n+m)$。

此时存在一种特殊情况，题目要求加入的边满足 $x>y$，而上面的方案拆分后加入的边是从 $r$ 连向 $l+1$。因此如果只经过了一个关键状态，则 $l=r$，这样的情况不合法。这种情况出现当且仅当只有一个满足条件的 $i$，可以特判处理。

再考虑第四种情况，可以发现对于一个第一种情况的路径，删去 $x\to y$ 加入 $n\to 1$ 就能得到第四种，反过来也可以得到第一种。因此第四种存在当且仅当第一种路径存在至少一条。

然后考虑第二种和第三种。对于这类路径，删去加入边后中间的路径满足的条件和上一种类似，唯一的区别是经过的关键状态数量为偶数。

此时只需要对于开头的每个点求出这个点的关键状态出发是否存在合法路径，以及每个结尾是否存在合法路径，可以使用与上面类似的方式求出，使用上面求出的结果即可。

复杂度 $O(n+m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 150050
#define ll long long
int T,n,m;
int s[N][2],is[N],rs[N],head[N],cnt,dp[N][2];
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
ll solve()
{
	scanf("%d%d",&n,&m);
	for(int i=0;i<=n+1;i++)is[i]=rs[i]=head[i]=dp[i][0]=dp[i][1]=0;cnt=0;
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(s[i][1]==s[i][0]+1)is[s[i][0]]=1;
	}
	rs[n]=n;for(int i=n-1;i>=1;i--)rs[i]=is[i]?rs[i+1]:i;
	if(rs[1]==n)return 1ll*n*(n-1)/2;
	for(int i=1;i<=m;i++)if(s[i][1]>s[i][0]+1&&rs[s[i][0]+1]>=s[i][1]-1)adde(s[i][0],s[i][1]-1);
	int lb=0,rb=0;
	for(int i=n;i>=1;i--)if(!is[i])lb=i;
	for(int i=1;i<n;i++)if(!is[i])rb=i;
	int fg=0,ls=lb;
	for(int i=lb+1;i<=rb;i++)if(!is[i])
	{
		dp[ls][0]=1;dp[ls][1]=0;
		for(int t=ls;t<i;t++)for(int j=head[t];j;j=ed[j].next)for(int k=0;k<2;k++)dp[ed[j].t][k]|=dp[t][k^1];
		int sv=dp[i][0]+dp[i][1]*2;
		if(!sv)return 0;
		if(sv==3||fg==-1)fg=-1;else fg^=sv-1;
	}
	dp[lb][0]=1;dp[lb][1]=0;dp[rb][0]=1;dp[rb][1]=0;
	for(int i=lb-1;i>=1;i--)for(int j=head[i];j;j=ed[j].next)for(int k=0;k<2;k++)dp[i][k]|=dp[ed[j].t][k^1];
	for(int i=rb;i<n;i++)for(int j=head[i];j;j=ed[j].next)for(int k=0;k<2;k++)dp[ed[j].t][k]|=dp[i][k^1];
	ll c[4]={0,0,0,0},d[4]={0,0,0,0},as=0;
	for(int i=1;i<=lb;i++)c[dp[i][0]+dp[i][1]*2]++;
	for(int i=rb;i<n;i++)d[dp[i][0]+dp[i][1]*2]++;
	for(int i=1;i<=3;i++)for(int j=1;j<=3;j++)if(i==3||j==3||fg==-1||((i-1)^(j-1)==fg))as+=c[i]*d[j];
	if(as)as++;if(lb==rb)as--;
	for(int i=1;i<=3;i++)
	{
		if(i==3||fg==-1||d[((i-1)^fg^1)+1]||d[3])as+=c[i];
		if(i==3||fg==-1||c[((i-1)^fg^1)+1]||c[3])as+=d[i];
	}
	return as;
}
int main()
{
	scanf("%d",&T);
	while(T--)printf("%lld\n",solve());
}
```

##### Keep XOR Low

###### Problem

给定长度为 $n$ 的非负数组 $a$ 以及一个非负整数 $k$，求所有 $2^n-1$ 种从 $a$ 中选一个非空子集的方式中，有多少方式满足选出的数中任意两个数异或结果不超过 $k$。答案模 $998244353$。

$n\leq 1.5\times 10^5,a_i,k<2^{30}$

$1s,256MB$

###### Sol

令 $d$ 满足 $2^d\leq k<2^{d+1}$。考虑对所有数建Trie，则选出的数一定在一个高度为 $d+1$ 的子树中。

考虑计算一个子树中的方案数，首先考虑最高位，最高位相同的之间xor一定不会超过 $k$，只需要考虑最高位不同的，相当于分成两类 $1\cdots$ 和 $0\cdots$。

此时考虑下一位，此时有四类 $11\cdots,10\cdots,01\cdots,00\cdots$，如果 $k$ 下一位为 $1$，则 $11,01$ 两类间的xor一定都合法，$10,00$ 两类间的xor也都合法，因此只需要考虑 $11\cdots,00\cdots$ 两类间的方案数以及 $10\cdots,01\cdots$ 间的方案数，再相乘即可。

如果下一位为 $0$，则如果两类都选了，则要么只能选 $11\cdots,00\cdots$ 中的，要么只能选另外两类，可以求出方案相加。另外一种情况是之选一个大类的，即只选 $0\cdots$ 或者 $1\cdots$，这种情况也容易求出。

注意到上述做法将一个形如给出Trie上两个子树，只有两个子树之间的点对可能超过限制，求方案数的问题分为了两个形式相同的子问题。考虑设 $f_{x,y,h}$ 表示给出Trie上两个高度为 $h$ 的子树 $x,y$，两个子树的高位xor后与 $k$ 的高位相同，在这两个子树中选择一个可以为空的子集的方案数。

则如果 $k$ 在这一位上为 $1$，则有 $f_{x,y,h}=f_{ch_{x,0},ch_{y,1},h-1}*f_{ch_{x,1},ch_{y,0},h-1}$。

否则，有 $f_{x,y,h}=f_{ch_{x,0},ch_{y,0},h-1}+f_{ch_{x,1},ch_{y,1},h-1}+2^{sz_{x}}-2^{sz_{ch_{x,0}}}-2^{sz_{ch_{x,1}}}+2^{sz_{y}}-2^{sz_{ch_{y,0}}}-2^{sz_{ch_{y,1}}}+1$，前两类为选两个子树中的情况，后两类为只选一个子树内的情况， $+1$ 为考虑所有部分对空集重复计算的结果。

如果有一个点不存在，那么答案一定为 $2^{sz_x+sz_y}$，因此直接递归计算的复杂度为 $O(n\log v)$。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 150050
#define M 5910001
#define mod 998244353
int n,k,v[N],ch[M][2],sz[M],pw[N],ct=1,as,a;
void ins(int v)
{
	int nw=1;
	for(int d=29;d>=0;d--)
	{
		int tp=(v>>d)&1;
		if(!ch[nw][tp])ch[nw][tp]=++ct;
		nw=ch[nw][tp],sz[nw]++;
	}
}
int solve(int x,int y,int d)
{
	if(!x||!y||!d)return pw[sz[x]+sz[y]];
	int tp=(k>>d-1)&1;
	if(tp)return 1ll*solve(ch[x][0],ch[y][1],d-1)*solve(ch[x][1],ch[y][0],d-1)%mod;
	else return (4ll*mod+pw[sz[x]]-pw[sz[ch[x][0]]]-pw[sz[ch[x][1]]]+pw[sz[y]]-pw[sz[ch[y][0]]]-pw[sz[ch[y][1]]]+solve(ch[x][0],ch[y][0],d-1)+solve(ch[x][1],ch[y][1],d-1)+1)%mod;
}
void dfs(int x,int d)
{
	if(!x)return;
	if(!d){as=(as+pw[sz[x]]-1)%mod;return;}
	if((1<<d-1)<=k){as=(as+solve(ch[x][0],ch[x][1],d-1)-1)%mod;return;}
	dfs(ch[x][0],d-1);dfs(ch[x][1],d-1);
}
int main()
{
	scanf("%d%d",&n,&k);
	pw[0]=1;
	for(int i=1;i<=n;i++)scanf("%d",&a),pw[i]=2*pw[i-1]%mod,ins(a);
	dfs(1,30);
	printf("%d\n",(as+mod)%mod);
}
```

##### ColorfulParentheses

###### Problem

给一个长度为 $n$ 的序列 $v$，你需要给每个位置放一个括号，使得所有括号按顺序构成一个合法括号序列，且 $v_i$ 相同的位置的括号相同。求方案数。

$n\leq 50$

$1s,256MB$

###### Sol

考虑折半，dfs枚举前 $\frac n2$ 位的情况，则此时与后 $\frac n2$ 位有关的状态只有：

1. 对于所有在左右两侧 $v$ 中都出现的值，这种值在左侧对应了那种括号。
2. 将左括号看作 $1$，右括号看作 $-1$，则左侧所有括号的和。

对于后 $\frac n2$ 位，倒过来可以得到类似的结果。可以发现左右两侧可以合并当且仅当两部分的值都完全相同。

因此对于两部分分别dfs后，相当于给出两个序列 $s1,s2$，求对于每一种可能的数，两个序列中这种数出现次数的乘积的和。可以hashtable解决。

复杂度 $O(2^{\frac n2})$，事实上 $n=50$ 时状态数不超过 $5200300$。

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 55
#define ll long long
int n,v[N],l1,lr,id[N],ct;
ll as;
vector<int> s1,s2;
void dfsl(int k,int su,ll f1,ll f2)
{
	if(su<0)return;
	if(k==l1+1){s1.push_back((f1&((1<<lr)-1))+(su<<lr));return;}
	int fg=-1;
	if((f1>>v[k])&1)fg=1;
	if((f2>>v[k])&1)fg=0;
	if(fg!=1)dfsl(k+1,su-1,f1,f2|(1ll<<v[k]));
	if(fg!=0)dfsl(k+1,su+1,f1|(1ll<<v[k]),f2);
}
void dfsr(int k,int su,ll f1,ll f2)
{
	if(su<0)return;
	if(k==l1){s2.push_back((f1&((1<<lr)-1))+(su<<lr));return;}
	int fg=-1;
	if((f1>>v[k])&1)fg=1;
	if((f2>>v[k])&1)fg=0;
	if(fg!=1)dfsr(k-1,su+1,f1,f2|(1ll<<v[k]));
	if(fg!=0)dfsr(k-1,su-1,f1|(1ll<<v[k]),f2);
}
#define mod 5200331
#define M 5200400
struct hashtable{
	int hd[M],nt[M],ct,su,vl[M],sa[M];
	void add(int x)
	{
		int tp=x%mod;
		for(int i=hd[tp];i;i=nt[i])if(vl[i]==x){sa[i]++;return;}
		int st=++ct;nt[st]=hd[tp];hd[tp]=st;sa[st]=1;vl[st]=x;
	}
	int que(int x)
	{
		int tp=x%mod;
		for(int i=hd[tp];i;i=nt[i])if(vl[i]==x)return sa[i];
		return 0;
	}
}ht;
struct ColorfulParentheses{
	ll count(vector<int> v1)
	{
		n=v1.size();
		for(int i=1;i<=n;i++)v[i]=v1[i-1];
		l1=(n+1)/2;
		ll ls=0,rs=0;
		for(int i=1;i<=l1;i++)ls|=1ll<<v[i];
		for(int i=l1+1;i<=n;i++)rs|=1ll<<v[i];
		ls&=rs;
		for(int i=0;i<n;i++)if((ls>>i)&1)id[i]=++ct;
		lr=ct;
		for(int i=1;i<=n;i++)if(!id[v[i]])id[v[i]]=++ct;
		for(int i=1;i<=n;i++)v[i]=id[v[i]]-1;
		dfsl(1,0,0,0);dfsr(n,0,0,0);
		for(int i=0;i<s1.size();i++)ht.add(s1[i]);
		for(int i=0;i<s2.size();i++)as+=ht.que(s2[i]);
		return as;
	}
};
```

##### BipartiteGraphGame

###### Problem

有一个二分图，两侧分别有 $n,m$ 个点，两侧任意两个点之间都有连边。

现在左侧的每个点上都有一个红色棋子，红色棋子编号为 $1,...,n$，右侧每个点上都有一个蓝色棋子，蓝色棋子编号为 $1,...,m$。

你可以进行若干次操作，每次操作可以选择一条边，然后交换这条边两侧的棋子。

你需要使得操作结束后，左侧第 $i$ 个点上为编号为 $i$ 的红色棋子，右侧第 $i$ 个点上为编号为 $j$ 的蓝色棋子。

你的方案需要满足：

1. 操作次数不超过 $1000$。
2. 最多有一条边被使用两次，其它边最多使用一次。

构造任意方案。

$n,m\leq 100$

$2s,256MB$

###### Sol

考虑如果两个排列看成置换都只有一个环的情况，此时有如下做法：

1. 交换 $(1,1)$。
2. 用右侧 $1$ 依次与左侧环上每个点交换，还原左侧。
3. 用左侧 $1$ 依次与右侧环上每个点交换，还原右侧。
4. 交换 $(1,1)$。

这样只有一条边被使用两次，但多个环时如果需要一次这样还原，则每个环都会使得一条边被用两次。

考虑如何消去其它环，对于一个偶环，可以考虑如下方式：

设环上节点为 $(1,2,...,2k)$，进行如下步骤：

1. 交换 $(p_1,1)$。
2. 对于 $2\sim 2k$ 的每个点 $2d+s(s\in(0,1))$，交换 $(p_s,2d+s),(p_{1-s},2d+s)$。
3. 交换 $(p_2,1)$。

可以发现对于中间的每一步，$p_{1-s}$ 都是环上上一个点的棋子，而将当前的棋子放到 $p_s$，再从 $p_{1-s}$ 放过来，就可以完成排列的还原过程。且可以发现这样不会改变对面 $1,2$ 的棋子位置。

对于一个奇环，可以考虑如下方式：

1. 交换 $(p_1,1),(p_1,2)$
2. 对于 $3\sim 2k+1$ 的每个点 $2d+1+s(s\in(0,1))$，交换 $(p_s,2d+1+s),(p_{1-s},2d+1+s)$。
3. 交换 $(p_2,1)$。

可以发现这样能还原奇环，但会交换 $1,2$ 的棋子位置。

因此可以考虑两侧各找一个大于 $1$ 的环（找不到就任选两个点），然后对其它环用环上两个点还原。最后两侧一定还剩下一个环或者没有环，且两个环之间的边没有被用过。此时使用之前的做法即可。

复杂度 $O(n+m)$，操作步数不超过 $2(n+m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 105
int n,m,p[N],q[N],ct,as[N*4][2],visa[N],visb[N];
void doit(int x,int y){as[++ct][0]=x-1;as[ct][1]=y-1;swap(p[x],q[y]);}
void doit1(int x,int y){as[++ct][0]=x-1;as[ct][1]=y-1;}
int c1,c2;
vector<int> sl[N],sr[N];
struct BipartiteGraphGame{
	vector<int> getMoves(vector<int> a1,vector<int> a2)
	{
		n=a1.size();m=a2.size();
		for(int i=1;i<=n;i++)p[i]=a1[i-1]+1;
		for(int i=1;i<=m;i++)q[i]=a2[i-1]+1;
		for(int i=1;i<=n;i++)if(!visa[i]&&p[i]!=i)
		{
			c1++;sl[c1].push_back(i);
			int st=p[i];while(st!=i)sl[c1].push_back(st),visa[st]=1,st=p[st];
		}
		for(int i=1;i<=m;i++)if(!visb[i]&&q[i]!=i)
		{
			c2++;sr[c2].push_back(i);
			int st=q[i];while(st!=i)sr[c2].push_back(st),visb[st]=1,st=q[st];
		}
		int f1=0,f2=0;
		for(int i=1;i<=c1;i++)if(sl[i].size()>sl[f1].size())f1=i;
		for(int i=1;i<=c2;i++)if(sr[i].size()>sr[f2].size())f2=i;
		int v1=1,v2=2;
		if(f1)v1=sl[f1][0],v2=sl[f1][1];
		for(int i=1;i<=c2;i++)if(i!=f2)
		if(sr[i].size()%2==0)
		{
			doit(v1,sr[i][0]);
			for(int j=1;j<sr[i].size();j++)
			doit(v2,sr[i][j]),doit(v1,sr[i][j]),v1^=v2^=v1^=v2;
			doit(v1,sr[i][0]);
		}
		else
		{
			doit(v1,sr[i][0]);doit(v1,sr[i][1]);
			for(int j=2;j<sr[i].size();j++)
			doit(v2,sr[i][j]),doit(v1,sr[i][j]),v1^=v2^=v1^=v2;
			doit(v1,sr[i][0]);
		}
		v1=1,v2=2;
		if(f2)v1=sr[f2][0],v2=sr[f2][1];
		for(int i=1;i<=c1;i++)if(i!=f1)
		if(sl[i].size()%2==0)
		{
			doit(sl[i][0],v1);
			for(int j=1;j<sl[i].size();j++)
			doit(sl[i][j],v2),doit(sl[i][j],v1),v1^=v2^=v1^=v2;
			doit(sl[i][0],v1);
		}
		else
		{
			doit(sl[i][0],v1);doit(sl[i][1],v1);
			for(int j=2;j<sl[i].size();j++)
			doit(sl[i][j],v2),doit(sl[i][j],v1),v1^=v2^=v1^=v2;
			doit(sl[i][0],v1);
		}
		v1=f1?sl[f1][0]:1;v2=f2?sr[f2][0]:1;
		doit1(v1,v2);
		int nw=v2;
		while(1)
		{
			nw=q[nw];
			if(nw==v2)break;
			doit1(v1,nw);
		}
		nw=v1;
		while(1)
		{
			nw=p[nw];
			if(nw==v1)break;
			doit1(nw,v2);
		}
		doit1(v1,v2);
		vector<int> as1;
		for(int i=1;i<=ct;i++)for(int j=0;j<2;j++)as1.push_back(as[i][j]);
		return as1;
	}
};
```

##### Nasty Donchik

###### Problem

给一个长度为 $n$ 的序列 $a$，求有多少对整数三元组 $(i,j,k)$ 满足：

1. $1\leq i\leq j<k\leq n$
2. $[i,j]$ 中出现的元素种类与 $[j+1,k]$ 中出现的元素种类相同。

$n\leq 2\times 10^5$

$1.5s,256MB$

###### Sol

考虑枚举一个位置计算答案。可以发现枚举 $j$ 难以处理，因此考虑枚举 $i$。

对于一个 $i$，考虑每种元素 $a$ 导致的限制。可以发现如果 $[i,j]$ 中存在 $a$，则 $k$ 不能小于 $a$ 在 $j$ 之后下一次出现的位置。否则，$k$ 不能大于等于 $a$ 在 $j$ 之后下一次出现的位置。

如果只考虑第一种限制，则相当于对于 $i$ 之后的每个位置 $x$，限制如果 $j\geq x$，则 $k$ 不能小于 $a_x$ 下一次出现的位置。因此如果从大到小枚举 $i$，则相当于加入这样的限制。显然这样的限制会构成一个单调栈的形式，因此如果设 $lb_y$ 表示 $j=y$ 时，$k$ 的下界，则变化过程中 $lb$ 的操作相当于 $O(n)$ 次区间赋值，且任何时刻 $lb$ 单调不降。

再考虑第二种限制，相当于对于 $i$ 之后每一个第一次出现的数 $a$ 以及位置 $x$，限制如果 $j<x$，则 $k<x$。如果从小到大枚举 $i$，则可以看成每次删去 $i-1$ 位置的限制，再加入 $a_{i-1}$ 下一次出现位置的限制。但删除的限制只影响 $[1,i-1]$，而这部分不影响答案，因此可以不进行删除。此时这部分也形如一个单调栈，可以维护 $rb_y$ 表示 $j=y$ 时，$k$ 的上界。

此时可以将从大到小的区间赋值倒过来，变成 $O(n)$ 次 $i$ 从小到大时的区间赋值，这样所有的操作都可以看成 $i$ 从小到大时的区间赋值。

则问题相当于维护 $lb,rb$ ，并支持区间询问 $\sum \max(0,rb_y-lb_y+1)$。考虑一次将值相同的区间变成另外一个值的操作，因为 $lb,rb$ 都单调不降，可以线段树上二分求出这个区间内在修改前和修改后的贡献，这样可以单次 $O(\log n)$ 维护答案。因为区间染色可以拆成均摊 $O(n)$ 次将值相同的区间变成另外一个值，因此总操作次数为 $O(n)$。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<vector>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,v[N],rs[N],nt[N],st[N],ct,v1[N],ti;
struct sth{int l,r,v;};
vector<sth> s1[N],s2[N];
ll as,fu;
void solve0()
{
	set<int> v1;
	for(int i=1;i<=n;i++)rs[i]=n+1;
	for(int i=n;i>=1;i--)nt[i]=rs[v[i]],rs[v[i]]=i;
	v1.insert(n+1);
	for(int i=1;i<=n;i++)if(rs[i]<=n){int ls=*v1.lower_bound(rs[i]);if(rs[i]<ls)s1[1].push_back((sth){rs[i],ls-1,rs[i]});v1.insert(rs[i]);}
	for(int i=1;i<n;i++)if(nt[i]<=n){int ls=*v1.lower_bound(nt[i]);if(nt[i]<ls)s1[i+1].push_back((sth){nt[i],ls-1,nt[i]});v1.insert(nt[i]);}
}
void solve1()
{
	st[ct=1]=n+1;
	for(int i=n;i>=1;i--)
	{
		int rb=nt[i]-1;
		while(st[ct]<=rb)s2[i+1].push_back((sth){st[ct+1]+1,st[ct],v1[ct]}),ct--;
		s2[i+1].push_back((sth){st[ct+1]+1,rb,v1[ct]});
		st[++ct]=rb;v1[ct]=i;st[ct+1]=0;
	}
	for(int i=1;i<=ct;i++)s2[1].push_back((sth){st[i+1]+1,st[i],v1[i]});
}
struct segt{
	struct node{int l,r,lz,mn,mx;ll su;}e[N*4];
	void doit(int x,int v){e[x].lz=e[x].mn=e[x].mx=v;e[x].su=1ll*(e[x].r-e[x].l+1)*v;}
	void pushdown(int x){if(e[x].lz!=-1)doit(x<<1,e[x].lz),doit(x<<1|1,e[x].lz);e[x].lz=-1;}
	void pushup(int x){e[x].su=e[x<<1].su+e[x<<1|1].su;e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r)return;int mid=(e[x].l+e[x].r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void modify(int x,int l,int r,int v){if(e[x].r<l||e[x].l>r)return;if(e[x].l>=l&&e[x].r<=r){doit(x,v);return;}pushdown(x);modify(x<<1,l,r,v);modify(x<<1|1,l,r,v);pushup(x);}
	ll query(int x,int l,int r){if(e[x].r<l||e[x].l>r)return 0;if(e[x].l>=l&&e[x].r<=r)return e[x].su;pushdown(x);return query(x<<1,l,r)+query(x<<1|1,l,r);}
	int query1(int x,int v){if(e[x].l==e[x].r)return e[x].mx>=v?e[x].l:n+1;pushdown(x);return e[x<<1].mx>=v?query1(x<<1,v):query1(x<<1|1,v);}
	int query2(int x,int v){if(e[x].l==e[x].r)return e[x].mn<=v?e[x].l:0;pushdown(x);return e[x<<1|1].mn<=v?query2(x<<1|1,v):query2(x<<1,v);}
}tr[2];
ll calcl(int l,int r,int v)
{
	int tp=tr[1].query1(1,v);
	if(tp<l)tp=l;if(tp>r)return 0;
	return tr[1].query(1,tp,r)-1ll*(r-tp+1)*v;
}
ll calcr(int l,int r,int v)
{
	int tp=tr[0].query2(1,v);
	if(tp>r)tp=r;if(tp<l)return 0;
	return 1ll*(tp-l+1)*v-tr[0].query(1,l,tp);
}
void modifyl(int l,int r,int v)
{
	while(l<=r)
	{
		int vl=tr[0].query(1,r,r),lb=tr[0].query1(1,vl);
		if(lb<l)lb=l;
		fu+=calcl(lb,r,v)-calcl(lb,r,vl);
		tr[0].modify(1,lb,r,v);r=lb-1;
	}
}
void modifyr(int l,int r,int v)
{
	if(l<ti)l=ti;if(r>n)r=n;
	while(l<=r)
	{
		int vl=tr[1].query(1,l,l),rb=tr[1].query2(1,vl);
		if(rb>r)rb=r;
		fu+=calcr(l,rb,v)-calcr(l,rb,vl);
		tr[1].modify(1,l,rb,v);l=rb+1;
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	solve0();solve1();
	tr[0].build(1,1,n);tr[1].build(1,1,n);
	for(int i=1;i<=n;i++)
	{
		ti++;
		for(int j=0;j<s1[i].size();j++)modifyl(s1[i][j].l,s1[i][j].r,s1[i][j].v);
		for(int j=0;j<s2[i].size();j++)modifyr(s2[i][j].l,s2[i][j].r,s2[i][j].v);
		as+=fu;
		modifyr(i,i,0);
	}
	printf("%lld\n",as);
}
```

##### Message

###### Problem

给出长度为 $n,m$ 的只包含小写字符的字符串 $s,t$，可以进行若干次操作，每次操作可以选择一种字符，删去这种字符的第一次出现或者最后一次出现。

每个位置的字符有一个删去代价（位置不随着删去而改变），你需要让 $s$ 变为 $t$，求最小删除代价或输出无解。

$n,m\leq 2\times 10^5$

$2s,256MB$

###### Sol

可以看成使得保留的字符权值和最大。

考虑将 $t$ 中按每种字符第一次出现的位置前以及最后一次出现后的位置分段，则考虑一段对应 $s$ 中的段(不保留段之间的空位)，可以发现此时 $s$ 中这一段内每种字符必须全部保留或者全部不保留。

考虑 $dp_{i,j}$ 表示前 $i$ 段划分到位置 $j$ 结尾的方案数，考虑一次转移一段，对于一段，此时 $s$ 保留的字符固定，一个位置能转移当且仅当这个位置向后的 $s$ 保留字符包含 $t$ 这一段作为前缀，且此时这一段一定对应这个前缀。因此只需要~~KMP ~~Hash判断两段字符串是否相等，即可 $O(n)$ 转移。

再考虑段之间空位的转移，可以发现两段之间的部分只能有不在两边同时出现的字符，这点与上部分类似，因此可以扫一遍转移段之间的部分。

复杂度 $O(n|\sum|)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 205000
#define ll long long
#define mod 102030405060719ll
int n,m,v[N],st[N],ct,lb[N],rb[N],r1[N],rv[N];
ll su,s1[N],dp[N],dp2[N];
char s[N],t[N],s2[N];
ll pw[N],h1[N],h2[N];
ll mul(ll a,ll b,ll p)
{
	ll tp=(long double)a*b/p;
	return ((a*b-tp*p)+p)%p;
}
ll calc(ll l,ll r,ll* h)
{
	return (h[r]-mul(h[l-1],pw[r-l+1],mod)+mod)%mod;
}
int main()
{
	scanf("%s%s",s+1,t+1);
	n=strlen(s+1),m=strlen(t+1);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su+=v[i];
	for(int i=1;i<=26;i++)
	{
		int s1=0,s2=0;
		for(int j=1;j<=m;j++)if(t[j]=='a'+i-1)s1=s1?s1:j,s2=j;
		if(!s1)continue;
		lb[i]=s1,rb[i]=s2;
		st[++ct]=s1,st[++ct]=s2+1;
	}
	pw[0]=1;for(int i=1;i<=m;i++)pw[i]=pw[i-1]*131%mod,h1[i]=(h1[i-1]*131+t[i])%mod;
	sort(st+1,st+ct+1);
	for(int j=1;j<ct;j++)if(st[j+1]!=st[j])
	{
		for(int i=0;i<=n;i++)r1[i]=0,dp2[i]=-1e18;
		int fg=0,cr=0;
		for(int i=1;i<=26;i++)if(lb[i]<=st[j]&&st[j]<=rb[i])fg|=1<<i-1;
		for(int i=1;i<=n;i++)
		{
			int tp=s[i]-'a'+1;
			if(fg&(1<<tp-1))s2[++cr]=s[i],r1[i]=cr,s1[cr]=s1[cr-1]+v[i],rv[cr]=i;
		}
		r1[n+1]=cr+1;for(int i=n;i>=1;i--)r1[i]=r1[i]?r1[i]:r1[i+1];
		for(int i=1;i<=cr;i++)h2[i]=(h2[i-1]*131+s2[i])%mod;
		for(int i=0;i<n;i++)
		{
			int tp=r1[i+1];
			if(tp==cr+1)continue;
			int rb=tp+st[j+1]-st[j]-1;
			if(calc(st[j],st[j+1]-1,h1)!=calc(tp,rb,h2))continue;
			dp2[rv[rb]]=max(dp2[rv[rb]],dp[i]+s1[rb]-s1[tp-1]);
		}
		int f2=0;
		for(int i=1;i<=26;i++)if(lb[i]<=st[j]&&st[j+1]<=rb[i])f2|=1<<i-1;
		for(int i=1;i<=n;i++)if(!(f2&(1<<s[i]-'a')))dp2[i]=max(dp2[i],dp2[i-1]);
		for(int i=0;i<=n;i++)dp[i]=dp2[i];
	}
	ll as=-1;for(int i=0;i<=n;i++)as=max(as,dp[i]);
	if(as>=0)printf("%lld\n",su-as);
	else printf("You better start from scratch man...\n");
}
```

##### Halting Problem

###### Problem

有一个变量 $x$，给出 $x$ 的初值 $x_0$。

有 $n$ 条指令，第 $i$ 条指令形如：

1. 如果当前 $x=x_i$，则给 $x$ 加上 $a_i$，然后跳转到第 $b_i$ 条指令。
2. 否则，给 $x$ 加上 $c_i$，然后跳转到第 $d_i$ 条指令。

跳转到 $n+1$ 视为终止。初始时程序在第 $1$ 条指令。求程序停止前，运行的指令条数，模 $10^9+7$，或输出程序不会停止。

$n\leq 10^5,x_i,a_i,c_i\in[-10^{13},10^{13}]$

$3s,256MB$

###### Sol

考虑将所有经过 $x=x_i$ 到达的状态以及初始状态看作关键状态，考虑对于每个关键状态求出这个关键状态出发到达的下一个关键状态（或者停止）。则可以看成只走 $x\neq x_i$ 的转移，到达下一个满足 $x=x_i$ 转移条件的时刻。

如果只考虑 $x\neq x_i$ 的转移，则转移构成一个基环树。因此可以分别考虑树的情况以及走到环上的情况。

对于一个有根树的情况，令 $s_v$ 表示关键状态从 $v$ 出发的权值,设 $he_u$ 表示 $u$ 到根的路径上的转移的权值和。则从 $v$ 出发可以在 $u$ 停止当且仅当 $s_v+he_v=x_u+he_u$ 且 $v$ 在 $u$ 子树内。因此考虑对于每种 $s_v+he_v$ 分别做，对于一种相当于每个出发点找祖先中第一个出现的位置，从结束位置考虑，则相当于每个结束位置可以覆盖一个dfs序区间。因此从深到浅考虑每个位置作为结束点的情况，然后相当于找到并删去dfs序在一个区间内的点。这部分复杂度为 $O(n\log n)$。

对于一个基环树，可以先删去一条边，做树上的问题，这样剩下所有的状态都可以走到当前的根，因此环上所有状态的出发点相同。

对于环的情况，考虑环上所有边权和 $su$，可以分几种情况讨论：

1. $su=0$

此时走一圈不会改变权值，因此每个状态只需要找接下来第一个前缀和与它相同的即可。复杂度 $O(n\log n)$

2. $su>0$

此时只需要将前缀和与初始权值 $\bmod su$ 相同的放在一起考虑。对于一类中的一个初始状态，需要找权值大于等于它的状态中最小的，如果有多个最小的再找位置最小的。可以将所有位置按照（权值，位置）排序，然后二分找下一个位置。复杂度 $O(n\log n)$。

3. $su<0$

这种情况显然和上一个类似。

因此可以对于每个关键状态求出它能到达的下一个关键状态或者求出它出发会导致无法结束。最后从起点开始走关键状态，如果循环显然不会终止，因此只会走 $O(n)$ 次。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<set>
#include<map>
using namespace std;
#define N 200500
#define ll long long
#define mod 1000000007
int n,nt[N],le[N],head[N],cnt,v1[N],vis[N];
ll s[N][5],v2[N],d2[N],v0;
struct edge{int t,next;ll v;}ed[N];
void adde(int f,int t,ll v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;}
int is[N],d1[N],c1,ct,lb[N],rb[N],fr[N];
map<ll,int> tp;
vector<int> fu[N];
void dfs(int u,int f1)
{
	lb[u]=++ct;fr[u]=f1;
	for(int i=head[u];i;i=ed[i].next)
	{
		d1[ed[i].t]=d1[u]+1;d2[ed[i].t]=d2[u]+ed[i].v;
		dfs(ed[i].t,f1);
	}
	rb[u]=ct;
	ll v1=s[u][0]+d2[u];
	if(!tp[v1])tp[v1]=++c1;
	fu[tp[v1]].push_back(u);
}
int fa[N];
int finds(int u){return fa[u]==u?u:fa[u]=finds(fa[u]);}
void solve1()
{
	for(int i=1;i<=n+1;i++)fa[i]=i;
	for(int i=1;i<=n;i++)
	if(finds(s[i][4])==finds(i))is[i]=1;
	else fa[finds(s[i][4])]=finds(i),adde(s[i][4],i,s[i][3]);
	is[n+1]=1;
	for(int i=1;i<=n+1;i++)if(is[i])dfs(i,i);
	for(int i=1;i<=n+1;i++)
	{
		ll vl=v2[i]+d2[v1[i]];
		if(!tp[vl])continue;
		fu[tp[vl]].push_back(-i);
	}
	for(int i=1;i<=c1;i++)
	{
		set<pair<int,int> > f1;
		for(int j=0;j<fu[i].size();j++)if(fu[i][j]<0)
		{
			int vl=-fu[i][j];
			f1.insert(make_pair(lb[v1[vl]],vl));
		}
		for(int j=0;j<fu[i].size();j++)if(fu[i][j]>0)
		{
			int vl=fu[i][j];
			while(1)
			{
				set<pair<int,int> >::iterator it=f1.lower_bound(make_pair(lb[vl],0));
				if(it==f1.end()||(*it).first>rb[vl])break;
				int s1=(*it).second;
				nt[s1]=vl,le[s1]=d1[v1[s1]]-d1[vl];
				f1.erase(it);
			}
		}
	}
	for(int i=1;i<=n+1;i++)if(!nt[i])
	{
		le[i]+=d1[v1[i]];
		v2[i]+=d2[v1[i]];
		v1[i]=fr[v1[i]];
		if(v1[i]==n+1)nt[i]=n+2;
	}
	else if(nt[i]==n+1)nt[i]=n+2;
}
vector<int> f1[N];
vector<pair<ll,int> > f2[N];
int st[N];
void solve2()
{
	for(int i=1;i<=n+1;i++)if(!nt[i])f1[v1[i]].push_back(i);
	for(int i=1;i<=n;i++)if(is[i])
	{
		for(int j=1;j<=c1;j++)f2[j].clear();
		tp.clear();c1=0;
		int l1=0,nw=i;
		while(1)
		{
			st[++l1]=nw;
			nw=s[nw][4];
			if(nw==i)break;
		}
		ll su=0,su1=0;
		for(int j=1;j<=l1;j++)su+=s[st[j]][3];
		for(int j=1;j<=l1;j++)
		{
			ll s1=s[st[j]][0]-su1,vl=su;
			if(su<0)s1*=-1,vl*=-1;
			ll s2=s1;if(vl)s2%=vl,s2=(s2+vl)%vl;
			if(!tp[s2])tp[s2]=++c1;
			f2[tp[s2]].push_back(make_pair(s1,j));
			su1+=s[st[j]][3];
		}
		for(int j=0;j<f1[i].size();j++)
		{
			ll id=f1[i][j],s1=v2[id],vl=su;
			if(su<0)s1*=-1,vl*=-1;
			ll s2=s1;if(vl)s2%=vl,s2=(s2+vl)%vl;
			if(!tp[s2])continue;
			f2[tp[s2]].push_back(make_pair(s1,-id));
		}
		if(su<0)su*=-1;
		for(int j=1;j<=c1;j++)
		{
			sort(f2[j].begin(),f2[j].end());
			pair<ll,int> ls=make_pair(-1,-1);
			for(int l=f2[j].size()-1;l>=0;l--)
			{
				if(f2[j][l].second>0)ls=f2[j][l];
				else
				{
					if(ls.second==-1)continue;
					int id=-f2[j][l].second;
					ll s1=f2[j][l].first;
					ll ti1=ls.second-1;
					if(su)ti1+=((ls.first-s1)/su)%mod*l1;
					le[id]=(le[id]+ti1)%mod;
					nt[id]=st[ls.second];
				}
			}
		}
	}
}
int main()
{
	scanf("%d%lld",&n,&v0);
	for(int i=1;i<=n;i++)for(int j=0;j<5;j++)scanf("%lld",&s[i][j]);
	for(int i=1;i<=n;i++)v1[i]=s[i][2],v2[i]=s[i][0]+s[i][1];
	v1[n+1]=1,v2[n+1]=v0;
	solve1();
	solve2();
	int nw=n+1,as=0;
	while(1)
	{
		if(vis[nw]){printf("-1\n");return 0;}vis[nw]=1;
		if(!nt[nw]){printf("-1\n");return 0;}
		as=(as+le[nw])%mod;
		if(nt[nw]==n+2){printf("%d\n",as);return 0;}
		nw=nt[nw],as++;
	}
}
```

##### Fast Spanning Tree

###### Problem

给定 $n$ 个点，每个点有点权 $v_i$，初始图中没有边。

有 $m$ 个三元组 $(a_i,b_i,c_i)$，进行如下过程。

进行如下过程：

找到编号最小的 $i$，使得 $a_i,b_i$ 当前不在一个连通块中且两者所在连通块的点权的总和大于等于 $c_i$。

如果不存在这样的 $i$，则结束过程，否则加入边 $(a_i,b_i)$。

求出过程中所有选择的 $i$ 构成的序列。

$n,m\leq 3\times 10^5,v_i,c_i\leq 10^6$

$5s,256MB$

###### Sol

考虑模拟整个过程，则需要在合并时维护哪些三元组变为合法。如果一个三元组的两个点已经连通，则之后它不会影响整个过程，因此不需要判断哪些三元组变得不合法。

一种做法是将边定向，每次合并时将所有边修改为从合并点连出，维护每个点所有出边，这样翻转次数不超过 $O(m\sqrt n)$，复杂度为 $O(m\sqrt n\log m)$，但无法通过。

可以发现如果一条边变得合法，则它的两个点所在的连通块中至少有一个增加了需要的权值的一半。因此对于一个三元组，设当前 $c_i$ 减去两个连通块的权值为 $v$，则可以在两个连通块内各加入一对 $(\frac{v+1}2,i)$，表示在这个连通块权值增加了 $\frac{v+1}2$ 的时候再考虑这条限制。在考虑这条现在时，可以重新计算这条限制的差。每次重新考虑时差一定减半，因此只会考虑 $\log v$ 次。

可以使用set维护每个连通块内这部分的限制，然后直接启发式合并，复杂度 $O(n\log n(\log v+\log n))$

###### Code

```cpp
#include<cstdio>
#include<set>
using namespace std;
#define N 300500
int n,m,fa[N],s[N][3],as[N],ct,v[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct sth{int vl,ls,id;}rs[N][2];
bool operator <(sth a,sth b){return a.vl==b.vl?a.id<b.id:a.vl<b.vl;}
set<sth> st[N];
set<int> fr,f1;
void doit(int x)
{
	for(int i=0;i<2;i++)
	{
		sth f1=(sth){v[finds(s[x][i])]+(s[x][2]+1)/2,v[finds(s[x][i])],x};
		rs[x][i]=f1;st[finds(s[x][i])].insert(f1);fr.insert(finds(s[x][i]));
	}
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)fa[i]=i;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),fr.insert(i);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]);
		s[i][2]-=v[s[i][0]]+v[s[i][1]];
		if(s[i][2]<=0)f1.insert(i);else doit(i);
	}
	while(!fr.empty()||!f1.empty())
	{
		while(!fr.empty())
		{
			int x=*fr.begin();
			if(st[x].empty()||(*st[x].begin()).vl>v[x]){fr.erase(x);continue;}
			int id=(*st[x].begin()).id;
			for(int t=0;t<2;t++)s[id][2]-=v[finds(s[id][t])]-rs[id][t].ls,st[finds(s[id][t])].erase(rs[id][t]);
			if(s[id][2]<=0)f1.insert(id);else doit(id);
		}
		if(!f1.empty())
		{
			int x=*f1.begin();f1.erase(x);
			int v1=finds(s[x][0]),v2=finds(s[x][1]);
			if(v1==v2)continue;
			as[++ct]=x;
			if(st[v1].size()<st[v2].size())v1^=v2^=v1^=v2;
			fa[v2]=v1;v[v1]+=v[v2];if(v[v1]>1e6)v[v1]=1e6;
			for(set<sth>::iterator it=st[v2].begin();it!=st[v2].end();it++)st[v1].insert(*it);
			st[v2].clear();fr.insert(v1);
		}
	}
	printf("%d\n",ct);
	for(int i=1;i<=ct;i++)printf("%d ",as[i]);
}

```

##### GameOnGraph

###### Problem

给一个 $n$ 个点的强连通竞赛图，除去 $1$ 外的每个点上有一个棋子，点 $i$ 上的棋子编号为 $i-1$。

你可以进行若干次操作，每次可以选择一条边，满足这条边从一个有棋子的点连向一个没有棋子的点，然后将棋子沿着这条边移动过去。

给出一个 $n-1$ 阶排列 $p$，你需要使用若干次操作，使得除去 $1$ 外的每个点上有一个棋子，点 $i$ 上的棋子编号为 $p_{i-1}$。

输出任意方案，操作次数不能超过 $2500$

$3\leq n\leq 30$

$2s,256MB$

###### Sol

首先有如下结论：

如果 $n>3$，则存在一个 $1$ 之外的节点，使得删去这个节点后图仍然强连通。

证明：首先对 $n$ 归纳，不难证明 $n=4,5$ 时成立。

考虑删去任意一个点 $x$，如果剩余图连通则可以直接删去这个点。否则，剩余的点会形成若干个强连通分量，且形成一条链。如果有大于等于三个强连通分量，则因为 $x$ 向第一个scc有边，最后一个scc向 $y$ 有边，因此删去中间的一个scc中的点不影响强连通性。

否则，此时只有两个强连通分量，如果两个scc大小都大于 $2$，则取不包含 $1$ 的scc，不妨设是链上靠后的一个，这个scc中包含一个哈密顿回路，且存在一个点连向 $x$。让哈密顿回路从 $x$ 后断开，变为以 $x$ 结尾的链，然后删去链开头点，此时这部分变成一条连向 $x$ 的链，此时显然仍然强连通。

否则，只有一个强连通分量大于 $2$，这部分是 $n-2$ 的情况，此时可以使用归纳的结论。因此结论成立。

考虑每次将一个点移到位置上，然后删去这个点，直到 $n=3$。$n=3$ 的情况容易解决。

直接想法是选定位置后，每次选择一个包含这个位置和需要的棋子的环并沿着这个环移动一圈。这样单次复杂度为 $2n+\frac12n^2$，总次数为 $\frac16n^3+n^2$，极限需要 $5000$ 次以上。

考虑找到需要的棋子和位置后，找一条竞赛图上棋子到位置的最短路 $(s=x_0,x_1,...,x_{l-1},x_l=t)$。由最短路的性质，对于任意的 $i,j$，如果 $j-i>1$，则 $(x_i,x_j)$ 之间的边为 $x_j$ 连向 $x_i$。

因此考虑先将空位移动到 $t$，然后沿着最短路将空位移动到 $x_1$，然后循环如下操作：

设当前棋子在 $x_i$，空位在 $x_{i+1}$。首先棋子移动到 $x_{i+1}$，然后 $x_{i+2}$ 的棋子移动到 $x_i$。

这样 $2n$ 步后棋子就可以到达位置，因为删去这个位置后图强连通，因此空位可以在不经过这个位置的情况下回到 $1$。

这样单次操作步数不超过 $5n$，总操作次数为 $\frac52n^2$ 级别，可以通过。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
#define N 32
#define M 2550
int n,cr,as[M],pr[N],p[N],ds[N][N],di[N],fr1[N],st[N],ct,p2[N];
char s[N][N];
bool chk(vector<int> t)
{
	int l=t.size();for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)ds[i][j]=i==j?0:1e8;
	for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)if(s[t[i-1]][t[j-1]]=='Y')ds[i][j]=1;
	for(int k=1;k<=l;k++)for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)ds[i][j]=min(ds[i][j],ds[i][k]+ds[k][j]);
	for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)if(ds[i][j]>1e6)return 0;
	return 1;
}
bool chkr(vector<int> t)
{
	int l=t.size();for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)ds[i][j]=i==j?0:1e8;
	for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)if(s[t[i-1]][t[j-1]]=='N'&&i!=j)ds[i][j]=1;
	for(int k=1;k<=l;k++)for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)ds[i][j]=min(ds[i][j],ds[i][k]+ds[k][j]);
	for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)if(ds[i][j]>1e6)return 0;
	return 1;
}
void solve(vector<int> t)
{
	sort(t.begin(),t.end());
	if(t.size()==3)
	{
		if(p[t[1]]==t[1])return;
		if(s[t[1]][1]=='N')swap(t[1],t[2]);
		as[++cr]=t[1];as[++cr]=t[2];as[++cr]=t[0];
		return;
	}
	int fr=0,mx=1e9;
	for(int i=1;i<t.size();i++)
	{
		vector<int> t1;for(int j=0;j<t.size();j++)if(i!=j)t1.push_back(t[j]);
		if(chk(t1))fr=i;
	}
	chk(t);
	int r1=fr,l1;
	for(int i=0;i<t.size();i++)for(int j=0;j<t.size();j++)if(ds[i+1][r1+1]==ds[j+1][r1+1]+1&&s[t[i]][t[j]]=='Y')fr1[i]=j;
	chkr(t);
	int nw=0,ls1=0;
	while(nw!=r1)for(int i=0;i<t.size();i++)if(ds[i+1][r1+1]==ds[nw+1][r1+1]-1&&s[t[i]][t[nw]]=='Y')as[++cr]=t[i],p[t[nw]]=p[t[i]],p[t[i]]=0,nw=i;
	for(int i=0;i<t.size();i++)if(p[t[i]]==t[fr])l1=i;
	while(fr1[l1]!=nw)
	{
		int tp=l1;while(fr1[tp]!=nw)tp=fr1[tp];
		as[++cr]=t[tp],p[t[nw]]=p[t[tp]],p[t[tp]]=0;
		nw=tp;
	}
	ls1=nw;
	while(l1!=r1)
	{
		as[++cr]=t[l1],p[t[fr1[l1]]]=p[t[l1]],p[t[l1]]=0;
		if(fr1[l1]!=r1)as[++cr]=t[fr1[fr1[l1]]],p[t[l1]]=p[t[fr1[fr1[l1]]]],p[t[fr1[fr1[l1]]]]=0;
		ls1=l1;l1=fr1[l1];
	}
	nw=ls1;
	vector<int> t1;for(int j=0;j<t.size();j++)if(fr!=j)t1.push_back(t[j]);
	for(int j=0;j<t1.size();j++)if(t1[j]==t[nw]){nw=j;break;}
	chkr(t1);
	while(nw)for(int i=0;i<t1.size();i++)if(ds[i+1][1]==ds[nw+1][1]-1&&s[t1[i]][t1[nw]]=='Y')as[++cr]=t1[i],p[t1[nw]]=p[t1[i]],p[t1[i]]=0,nw=i;
	solve(t1);
}
struct GameOnGraph{
	vector<int> findSolution(vector<string> g,vector<int> p1)
	{
		n=g.size();
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=g[i-1][j-1];
		for(int i=1;i<=n;i++)pr[i]=p1[i-1],p[pr[i]+1]=i;p[1]=0;
		vector<int> tp;for(int i=1;i<=n;i++)tp.push_back(i);
		solve(tp);
		vector<int> as1;
		for(int i=1;i<=cr;i++)as1.push_back(as[i]-1);
		return as1;
	}
};
```

##### TrickyInequality

###### Problem

给出正整数 $s,t,n,m$，求满足如下条件的正整数列 $(x_1,...,x_m)$ 个数，模 $10^9+7$：

1. $\sum_{i=1}^mx_i\leq s$
2. $\forall 1\leq i\leq n,x_i\leq t$

$n,m\leq 10^9,t\leq 10^5,t*n\leq s\leq 10^{18}$

$2s,256MB$

###### Sol

如果 $n=0$，则容易发现答案为 $C_s^m$。因此如果后面 $m-n$ 个数的和为 $x$，则这部分的方案数为 $C_x^{m-n}$。

因为 $s\geq t*n$，因此左侧任意取不会出现不合法的情况，考虑先把左侧取满，然后考虑减少，可以得到如下结果：
$$
\sum_{0\leq f_i<t}C_{s-n*t+\sum f_i}^{m-n}
$$
从生成函数的角度考虑，相当于：
$$
[x^{m-n}]\sum_{0\leq f_i<t}(1+x)^{s-n*t+\sum f_i}\\
=[x^{m-n}](1+x)^{s-nt}(\sum_{0\leq k<t}(1+x)^k)^n\\
=[x^{m-n}](1+x)^{s-nt}(\frac{(1+x)^{t+1}-1}x)^n
$$
直接倍增+暴力乘法复杂度 $O((m-n)^2\log n)$，多项式exp可以做到 $O((m-n)\log(m-n))$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 105
#define ll long long
#define mod 1000000007
int n,m,t,d,as1,inv[N];
ll s;
vector<int> mul(const vector<int> &a,const vector<int> &b)
{
	vector<int> as;as.resize(d+1);
	for(int i=0;i<=d;i++)for(int j=0;j<=i;j++)
	as[i]=(as[i]+1ll*a[i-j]%mod*b[j])%mod;
	return as;
}
vector<int> a,as,b;
struct TrickyInequality{
	int countSolutions(ll s,int t,int n,int m)
	{
		d=m-n;
		a.resize(d+1);as=a;as[0]=1;
		for(int i=1;i<=d;i++)for(int j=1;j<=i;j++)if((1ll*j*mod+1)%i==0)inv[i]=(1ll*j*mod+1)/i;
		a[0]=1;
		for(int i=1;i<=d;i++)a[i]=1ll*a[i-1]*(t%mod+mod-i+1)%mod*inv[i]%mod;
		s-=1ll*t*n;
		int st=n;
		while(st)
		{
			if(st&1)as=mul(as,a);
			a=mul(a,a);st>>=1;
		}
		vector<int> b;b.resize(d+1);b[0]=1;
		for(int i=1;i<=d;i++)b[i]=1ll*b[i-1]*(s%mod+mod-i+1)%mod*inv[i]%mod;
		as=mul(as,b);
		return as[d];
	}
};
```

##### Sequence to Sequence

###### Problem

给出两个长度为 $n$ 的非负整数序列 $s,t$，你可以进行如下操作：

1. 选择一个区间，将区间内所有非零的数加一。
2. 选择一个区间，将区间内所有非零的数减一。

求 $s$ 变成 $t$ 需要的最少操作次数或输出无解。

$\sum n\leq 10^6,s_i,t_i\leq 10^9$

$1s,64MB$

###### Sol

显然无解当且仅当存在一个位置 $s_i=0,t_i>0$。

有如下结论：存在一种最优解，所有操作一定是先减再加。

证明：如果存在相邻两次操作先加再减，则：

1. 如果两个区间不相交，则可以直接交换顺序。
2. 如果两个区间相交但不包含，则可以两个区间同时不操作这部分，此时中间部分 $(+1,-1)$ 与不操作等价。
3. 如果一个包含另外一个，对于加的区间包含减的区间的情况，可以拆成两个小的加区间。另外一种情况同理。

因此归纳可得结论。此时设第 $i$ 个位置减去的次数为 $a_i$，加的次数为 $b_i$，则需要满足如下条件：

1. 如果 $t_i=0$，则需要满足 $a_i\geq s_i$
2. 否则需要满足 $a_i<s_i$ 且 $b_i-a_i=t_i-s_i$

转移代价为 $\sum \max(0,t_i-t_{i-1})+\max(0,s_i-s_{i-1})$，也可以写成 $\sum \max(0,t_{i-1}-t_i)+\max(0,s_i-s_{i-1})$

对于 $t_i>0$ 的位置，这个位置 $t_i-s_i$ 固定，因此可以设 $dp_{i,j}$ 表示 $s_i=j$ 时前面的最小代价。

考虑一次连续转移一段 $t_i=0$ 的位置，这一段中只需要考虑中间这些位置中所有 $s_i$ 的最大值 $mx$。设两侧的位置分别为 $s_l,t_l,s_r,t_r$，则如果使用第一种转移代价，则 $dp_{l,x}$ 转移到 $dp_{r,y}$ 的代价为：

$$
\max(0,y-x+(t_r-s_r)-(t_l-s_l))+\max(0,mx-x)+\max(0,y-\max(x,mx))
$$
固定 $x$ 可以发现，对于一个 $x$，它到 $y$ 的转移为分段一次函数，且斜率为 $0,1,2$，但斜率为 $2$ 难以处理。

因此考虑将写成第二种代价，则转移代价为：
$$
\max(0,x-y-(t_r-s_r)+(t_l-s_l))+\max(0,mx-x)+\max(0,y-\max(x,mx))
$$
此时关于 $x,y$ 都是一个斜率为 $-1,0,1$ 的分段线性上凸函数。由于函数的特殊性，可以发现任意时刻 $dp$ 都是一个斜率为 $-1,0,1$ 的分段线性上凸函数。

因此可以维护斜率为 $0$ 的段以及两侧的合法取值区间，转移时取出所有边界点和转移系数改变的点，暴力枚举关键的转移值转移，即可得到新的函数。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 100500
#define ll long long
int T,n,s[N],t[N];
ll l,r,as,lb,rb;
void trans(int ls,int s,int t,int v)
{
	ll as1=1e18,l1=0,r1=0,l2=1e9,r2=0;
	vector<int> fu;fu.push_back(l);fu.push_back(r);
	fu.push_back(v);fu.push_back(s-1);fu.push_back(lb);fu.push_back(rb);
	for(int i=0;i<fu.size();i++)
	{
		int x=fu[i];if(x<lb||x>rb)continue;
		vector<int> ry;
		ry.push_back(v);ry.push_back(max(0,s-t));ry.push_back(s-1);ry.push_back(x-t+s+ls);ry.push_back(x);
		for(int j=0;j<ry.size();j++)
		{
			int sx=x,sy=ry[j];ll vl=as+(x<l?l-x:0)+(x>r?x-r:0);
			if(sy>=s||sy<0)continue;
			if(sx<v)vl+=v-sx,sx=v;
			if(sx<sy)vl+=sy-sx;
			int d1=x+ls,d2=sy+t-s;
			if(d2<0)continue;
			if(d1>d2)vl+=d1-d2;
			if(as1>vl)as1=vl,l1=r1=sy;
			else if(as1==vl)l1=l1>sy?sy:l1,r1=r1<sy?sy:r1;
			if(l2>sy)l2=sy;if(r2<sy)r2=sy;
		}
	}
	l=l1,r=r1,as=as1;
	lb=l2,rb=r2;
}
ll solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&s[i]);
	for(int i=1;i<=n;i++)scanf("%d",&t[i]);
	for(int i=1;i<=n;i++)if(s[i]==0&&t[i])return -1;
	l=r=0,as=0;lb=0,rb=1e9;
	int mx=0,l1=0;
	for(int i=1;i<=n;i++)
	if(t[i]>0)
	trans(l1,s[i],t[i],mx),mx=0,l1=t[i]-s[i];
	else mx=mx<s[i]?s[i]:mx;
	trans(l1,1e9,1e9,mx);
	return as+l;
}
int main()
{
	scanf("%d",&T);while(T--)printf("%lld\n",solve());
}
```

##### Evacuation

###### Problem

有 $n+2$ 个位置排成一列，编号为 $0,...,n+1$，第 $i$ 个位置可以容纳 $a_i$ 个人。

给出 $S$，有 $q$ 次询问，每次给出区间 $l,r$，当前有 $S$ 个人在 $[l,r]$ 中的某个位置，每个人可以任意移动，需要使得 $[l,r]$ 中每个位置的人数都不超过容纳上限，定义代价为所有人移动的距离总和的最小值。求出 $S$ 个人在位置 $l,...,r$ 时，所有情况的最大代价。

$n,q\leq 2\times 10^5$

$6s,1024MB$

###### Sol

在初始位置确定后，所有人一定会找最近的空位，或者走出边界。显然所有人会从最近的边界出去。因此可以分成两个问题：对于位置在区间中点左侧的所有位置，求出从这些位置出发，边界在 $l$ 的最小代价的最大值，对于右侧类似。

对于左侧的情况，设 $f_{i,j}$ 表示所有人在位置 $i$，边界在位置 $j$ 的最小代价。则预处理后容易 $O(1)$ 求一个代价。

考虑 $f_{i,j}+f_{i-1,j-1}$ 与 $f_{i,j-1}+f_{i-1,j}$ 的关系。两者都可以看成从 $i,i-1$ 各有 $S$ 个人，两者分别有一个边界。如果看成整体，则相当于 $2S$ 个人都可以从 $j-1$ 离开，但有 $S$ 个人可以从 $j$ 离开。因此如果这 $S$ 个人初始更靠近左侧，则从 $j$ 离开的人不会变少，因此有 $f_{i,j}+f_{i-1,j-1}\geq f_{i,j-1}+f_{i-1,j}$。

因此存在决策单调性：如果可以选择的边界区间固定，则随着起始位置增加，选择的边界位置单调不降。因此可以对于所有询问线段树+分治。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,q,s[N][2];
ll v[N],su[N],si[N],li[N],ls[N],as[N],sr;
ll calc(int x,int s)
{
	if(s<=li[x])return ls[x];
	return (sr-su[x*2-s]+su[s-1])*(x-s+1)+si[x*2-s]-si[x]-(su[x*2-s]-su[x])*x+(su[x]-su[s-1])*x-(si[x]-si[s-1]);
}
int lb[N*4],rb[N*4];
void build(int x,int l,int r)
{
	lb[x]=l;rb[x]=r;
	if(l==r)return;
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
vector<pair<int,int> > fu[N*4];
void doit(int x,int l,int r,int id)
{
	if(lb[x]==l&&rb[x]==r){fu[x].push_back(make_pair(s[id][0],id));return;}
	int mid=(lb[x]+rb[x])>>1;
	if(mid>=r)doit(x<<1,l,r,id);
	else if(mid<l)doit(x<<1|1,l,r,id);
	else doit(x<<1,l,mid,id),doit(x<<1|1,mid+1,r,id);
}
void solve(int id,int l,int r,int l1,int r1)
{
	if(l>r)return;
	int mid=(l+r)>>1;
	ll mx=-1;
	int fr=0,lb=fu[id][mid].first,t1=fu[id][mid].second;
	for(int i=l1;i<=r1;i++)
	{
		ll tp=calc(i,lb);
		if(tp>mx)mx=tp,fr=i;
	}
	as[t1]=max(as[t1],mx);
	solve(id,l,mid-1,l1,fr);solve(id,mid+1,r,fr,r1);
}
void doit()
{
	for(int i=1;i<=n*4;i++)fu[i].clear();
	build(1,1,n);
	for(int i=1;i<=n;i++)su[i]=su[i-1]+v[i],si[i]=si[i-1]+v[i]*i;
	for(int i=1;i<=n;i++)
	{
		int lb=max(1,i*2-n),rb=i,as1=0;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(su[i*2-mid]-su[mid-1]>=sr)as1=mid,lb=mid+1;
			else rb=mid-1;
		}
		li[i]=0;
		if(!as1)continue;
		li[i]=as1,ls[i]=calc(i,as1+1);
	}
	for(int i=1;i<=q;i++)doit(1,s[i][0],(s[i][0]+s[i][1])/2,i);
	for(int i=1;i<=n*4;i++)if(fu[i].size())sort(fu[i].begin(),fu[i].end()),solve(i,0,fu[i].size()-1,lb[i],rb[i]);
}
int main()
{
	scanf("%d%lld",&n,&sr);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	scanf("%d",&q);
	for(int i=1;i<=q;i++)scanf("%d%d",&s[i][0],&s[i][1]);
	doit();
	for(int i=1;i*2<=n;i++)swap(v[i],v[n-i+1]);
	for(int i=1;i<=q;i++)swap(s[i][0],s[i][1]),s[i][0]=n-s[i][0]+1,s[i][1]=n-s[i][1]+1;
	doit();
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```

##### Entanglement

###### Problem

给定 $n,m,k$ 以及一个 $n\times m$ 的矩阵 $c$，其中 $c_{i,j}\in\{1,2,...,k\}$。

求出有多少个序列对 $a_{1,...,n},b_{1,...,m}$，满足所有元素属于 $\{1,2,...,k\}$ 且对于任意 $i,j$，$c_{i,j}$ 等于 $a_i$ 或 $b_j$。答案模 $10^9+7$。

$n,m\leq 300$

$3s,256MB$

###### Sol

考虑dfs，枚举一行的值，此时通过这些限制可以确定一些行列的值，而对于剩下的位置不会有限制，可以对剩下的行列继续dfs。

考虑这样的复杂度，设枚举的值为 $a_i$，则如果 $c_{i,j}\neq a_i$，则 $b_j=c_{i,j}$。因此对于所有 $a_i$ 的情况，每个 $b_j$ 只会有一种情况在第一步不能被确定，因此每个 $b_j$ 只会进入一种dfs情况。

设 $k$ 列的复杂度为 $f(k)$，如果只枚举在 $c_i$ 种出现过的行，剩下的行一起算，则相当于：
$$
f(k)=\max_{1\leq s_i,\sum s_i=k}nk+\sum(f(s_i)+nk)
$$
可以发现 $f(m)=O(nm^2)$，因此直接dfs即可。

复杂度 $O(nm^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
#define N 305
#define mod 1000000007
int n,m,k,v[N][N],s1[N],s2[N];
int solve(vector<int> v1,vector<int> v2)
{
	if(!v1.size()||!v2.size()){int as=1;for(int i=1;i<=v1.size()+v2.size();i++)as=1ll*as*k%mod;return as;}
	vector<int> ls;for(int i=0;i<v2.size();i++)ls.push_back(v[v1[0]][v2[i]]);
	sort(ls.begin(),ls.end());ls.push_back(-1);
	int su=k,as=0;
	for(int i=0;i<ls.size();i++)if(!i||ls[i]!=ls[i-1])
	{
		su--;
		int fg=1;
		for(int j=1;j<=n;j++)s1[j]=0;for(int j=1;j<=m;j++)s2[j]=0;
		queue<pair<int,int> > qu;
		s1[v1[0]]=ls[i];
		qu.push(make_pair(v1[0],0));
		while(!qu.empty()&&fg)
		{
			pair<int,int> st=qu.front();qu.pop();
			int d1=st.first,d2=st.second;
			if(d2)
			{
				for(int j=0;j<v1.size();j++)if(v[v1[j]][d1]!=s2[d1])
				{
					int f1=v[v1[j]][d1];if(s1[v1[j]]&&s1[v1[j]]!=f1){fg=0;break;}
					else if(!s1[v1[j]])s1[v1[j]]=f1,qu.push(make_pair(v1[j],0));
				}
			}
			else
			for(int j=0;j<v2.size();j++)if(v[d1][v2[j]]!=s1[d1])
			{
				int f1=v[d1][v2[j]];if(s2[v2[j]]&&s2[v2[j]]!=f1){fg=0;break;}
				else if(!s2[v2[j]])s2[v2[j]]=f1,qu.push(make_pair(v2[j],1));
			}
		}
		if(!fg)continue;
		vector<int> l1,l2;
		for(int j=0;j<v1.size();j++)if(!s1[v1[j]])l1.push_back(v1[j]);
		for(int j=0;j<v2.size();j++)if(!s2[v2[j]])l2.push_back(v2[j]);
		as=(as+1ll*(ls[i]==-1?su+1:1)*solve(l1,l2))%mod;
	}
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%d",&v[i][j]);
	vector<int> v1,v2;
	for(int i=1;i<=n;i++)v1.push_back(i);
	for(int i=1;i<=m;i++)v2.push_back(i);
	printf("%d\n",solve(v1,v2)); 
}
```

##### Giant Penguin

###### Problem

给定一个 $n$ 个点 $m$ 条边的连通图，保证每个点最多在 $k$ 个简单环中。

支持 $q$ 次操作：

1. 标记点 $x_i$。
2. 询问点 $x_i$ 到最近的标记点的距离。

$n\leq 10^5,m\leq 2\times 10^5,k\leq 10$

$3s,256MB$

###### Sol

考虑一种类似点分的方式：

每次选出当前连通块的一棵生成树的重心，bfs出重心到连通块内其它点的最短距离，然后删去重心，对剩下的每个连通块分治求解。

对于询问和修改，考虑使用类似点分树的形式，对于每个点记录点分树内子树中到它的最近距离。因为只有加入点，因此询问和修改直接遍历点分树的祖先即可。这样单次复杂度为点分树深度。

考虑点分树深度。在删去一个点后，如果一个点和删去点在同一个点双内，则经过这个点的简单环数量一定会减少。但对于任意一棵生成树，一个点双内的点在生成树中一定是一个连通块。因此删去这个点后，重心一定还在这个点双内（或者移动到点双和外面的公共点，但此时已经将树分成了两部分，且每部分大小不超过一半）。

因此这个点双会变成一个 $k-1$ -仙人掌，做 $k$ 次之后就会分成多个连通块，且每个连通块大小不超过一半。因此点分树深度为 $k\log n$，直接这样分即可。

复杂度 $O((n+q)k\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
#define N 100500
int n,m,q,a,b,head[N],cnt;
int vis[N],fa[N],de[N],ls[N],sz[N],vis1[N],ds1[N];
vector<int> ds[N];
int ts,mn,cr;
struct edge{int t,next;}ed[N*4];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void bfs(int u)
{
	queue<int> st;
	vector<int> s2;
	ds1[u]=1;st.push(u);
	while(!st.empty())
	{
		int x=st.front();st.pop();s2.push_back(x);
		ds[x].push_back(ds1[x]-1);
		for(int i=head[x];i;i=ed[i].next)if(!vis[ed[i].t]&&!ds1[ed[i].t])ds1[ed[i].t]=ds1[x]+1,st.push(ed[i].t);
	}
	for(int i=0;i<s2.size();i++)ds1[s2[i]]=0;
}
void dfs1(int u,int fa)
{
	vis1[u]=1;sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&!vis1[ed[i].t])dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t];
}
void dfs2(int u,int fa)
{
	int mx=ts-sz[u];
	vis1[u]=0;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&vis1[ed[i].t])dfs2(ed[i].t,u),mx=max(mx,sz[ed[i].t]);
	if(mx<mn)mn=mx,cr=u;
}
void dfs3(int u)
{
	vis[u]=1;ls[u]=1e7;bfs(u);
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		dfs1(ed[i].t,u);
		mn=1e7,ts=sz[ed[i].t];dfs2(ed[i].t,u);
		de[cr]=de[u]+1;fa[cr]=u;dfs3(cr);
	}
}
void modify(int x)
{
	int nw=x;
	while(nw)ls[nw]=min(ls[nw],ds[x][de[nw]]),nw=fa[nw];
}
int query(int x)
{
	int nw=x,as=1e7;
	while(nw)as=min(as,ls[nw]+ds[x][de[nw]]),nw=fa[nw];
	return as;
}
int main()
{
	scanf("%d%d%*d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs3(1);
	scanf("%d",&q);
	while(q--)
	{
		scanf("%d%d",&a,&b);
		if(a==1)modify(b);
		else printf("%d\n",query(b));
	}
}
```

##### Honorable Mention

###### Problem

给一个长度为 $n$ 的序列 $a$，$q$ 次询问：

每次询问给出 $l,r,k$，求出在 $[l,r]$ 中选择 $k$ 个非空不交子区间，所有子区间权值和的最大值。

$n,q,|a_i|\leq 35000$

$5s,256MB$

###### Sol

对于一次询问，考虑 $k$ 依次增加的过程。在 $k$ 不超过非负数的数量时，每次增加一个区间一定不会使答案变差，可以发现此时可以不考虑区间非空的限制，而这样就可以得到费用流模型，因此这部分为凸函数。

在 $k$ 大于等于非负数的数量时，显然每次会选择一个负数，因此这部分也是凸函数，同时可以发现两部分合并的结果是凸函数。

为了处理区间询问，考虑线段树，在线段树的每个节点上维护这个节点区间内，左/右端点是否必须选时，选择 $0,...,k$ 个非空不交区间的最大总权值。合并时枚举两侧的情况，然后相当于一个凸函数max+卷积，可以直接做。预处理复杂度 $O(n\log n)$。

考虑询问，由于答案关于 $k$ 是凸的，考虑wqs二分。考虑将区间在线段树上分成若干段，因为每一段内部是凸的，因而确定了二分的 $k$ 后，每一段内部可以 $O(\log n)$ 求出每种情况下的最优方案。然后dp合并，即可得到段数最少的最优方案。需要注意wqs二分要求出段数最少或最多的方案。

复杂度 $O(n\log^2n\log v)$

另外一种做法是不考虑区间之间合并段的情况找一组最优解，可以发现真正的最优解最多和这个解在每个区间的段数上相差 $O(\log n)$ 段，然后可以取出这些段做dp。复杂度为 $O(n\log^2n(\log n+\log v))$，可以参考上次的题解。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 35050
#define ll long long
int n,q,v[N],l,r,k;
vector<int> msum(vector<int> a,vector<int> b)
{
	vector<int> as;
	int s1=a.size(),s2=b.size(),v1=0,v2=0;
	as.push_back(1ll*a[0]+b[0]<-2e9?-2e9:a[0]+b[0]);
	for(int i=1;i<=s1+s2-2;i++)
	{
		if(v1==s1-1)v2++;
		else if(v2==s2-1)v1++;
		else if(a[v1+1]-a[v1]>=b[v2+1]-b[v2])v1++;
		else v2++;
		ll tp=1ll*a[v1]+b[v2];if(tp<-2e9)tp=-2e9;
		as.push_back(tp);
	}
	return as;
}
vector<int> add(vector<int> a,vector<int> b)
{
	vector<int> as;
	int s1=a.size(),s2=b.size();
	for(int i=0;i<s1||i<s2;i++)as.push_back(max(i<s1?a[i]:-2e9,i<s2?b[i]:-2e9));
	return as;
}
struct sth{vector<int> st[2][2];};
sth doit(sth l,sth r)
{
	sth as;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)
	for(int p=0;p<2;p++)for(int q=0;q<2;q++)
	{
		vector<int> fu=msum(l.st[i][p],r.st[q][j]);
		if(p&&q)for(int t=0;t+1<fu.size();t++)fu[t]=max(fu[t],fu[t+1]);
		as.st[i][j]=add(as.st[i][j],fu);
	}
	return as;
}
sth init(int v)
{
	sth as;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)
	as.st[i][j].push_back(i+j?-2e9:0),as.st[i][j].push_back(v);
	return as;
}
int lb[N*4],rb[N*4];
sth tp[N*4];
void build(int x,int l,int r)
{
	lb[x]=l;rb[x]=r;
	if(l==r){tp[x]=init(v[l]);return;}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	tp[x]=doit(tp[x<<1],tp[x<<1|1]);
}
int st[45],ct;
ll fu[45][2][2][2],dp[45][2][2];
void query(int x,int l,int r)
{
	if(lb[x]==l&&rb[x]==r){st[++ct]=x;return;}
	int mid=(lb[x]+rb[x])>>1;
	if(mid>=r)query(x<<1,l,r);
	else if(mid<l)query(x<<1|1,l,r);
	else query(x<<1,l,mid),query(x<<1|1,mid+1,r);
}
int query1(int l,int r,int k)
{
	ct=0;query(1,l,r);
	int lb=-35050,rb=1.3e9,as=0;
	while(lb<=rb)
	{
		int rv=(1ll*lb+rb)/2;
		for(int i=1;i<=ct;i++)
		for(int j=0;j<2;j++)
		for(int k=0;k<2;k++)
		{
			int l1=1,r1=tp[st[i]].st[j][k].size()-1,as1=0;
			while(l1<=r1)
			{
				int mid=(l1+r1)>>1;
				if(tp[st[i]].st[j][k][mid]-tp[st[i]].st[j][k][mid-1]>rv)as1=mid,l1=mid+1;
				else r1=mid-1;
			}
			fu[i][j][k][0]=tp[st[i]].st[j][k][as1]-1ll*as1*rv,fu[i][j][k][1]=as1;
		}
		for(int i=0;i<=ct;i++)for(int j=0;j<2;j++)if(i+j)dp[i][j][0]=-1e18;
		for(int i=1;i<=ct;i++)for(int j=0;j<2;j++)
		for(int k=0;k<2;k++)for(int l=0;l<2;l++)
		{
			ll r1=dp[i-1][j][0]+fu[i][k][l][0],r2=dp[i-1][j][1]+fu[i][k][l][1];
			if(j&&k&&rv>=0)r1+=rv,r2--;
			if(dp[i][l][0]<r1||(dp[i][l][0]==r1&&dp[i][l][1]>r2))dp[i][l][0]=r1,dp[i][l][1]=r2;
		}
		ll s1=dp[ct][0][0]+dp[ct][0][1]*rv,s2=dp[ct][0][1];
		if(s2<=k)
		as=s1+rv*(k-s2),rb=rv-1;
		else lb=rv+1;
	}
	return as;
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	build(1,1,n);
	while(q--)scanf("%d%d%d",&l,&r,&k),printf("%d\n",query1(l,r,k));
}
```

##### Even Rain

###### Problem

二维空间中有 $n$ 个位置排成一排，第 $i$ 个位置的高度为非负整数 $h_i$，所有位置宽度为 $1$。

假设当前所有位置上都有无穷多的水，接着水向两侧流动，流出这些位置时消失。定义这种高度序列的权值为最后剩余水的面积。

给定 $k$，现在可以选择 $k$ 个位置，将它们的高度变为 $0$。求有多少种方式使得最后的权值为偶数。

$n\leq 25000,k\leq 25,k\leq n-1$

$3s,512MB$

###### Sol

可以看成算水位高度总和减去每个位置原先的高度。可以发现最后一个位置的水位高度为两侧最大高度中的较小值。

考虑剩余序列中高度最大的位置，从这个位置分开，则左侧的水位高度为前缀的最大值，右侧水位高度为后缀最大值。

考虑左侧的情况，因为最多删去 $k$ 个位置，因此一个前缀此时的最大值一定是原先最大的 $k+1$ 个元素中的一个。因此可以设 $ls_{i,p,q,0/1}$ 表示考虑了左侧前 $i$ 个位置，删去了 $p$ 个位置，前缀最大值为原先前缀的第 $q$ 大位置，且左侧水量为奇数/偶数的方案数，dp可以预处理后 $O(1)$ 转移。右侧同理。

然后考虑计算答案，枚举现在的最大值所在的位置，枚举两侧的情况合并。合并的复杂度为 $O(k^3)$，但注意到只需要枚举原序列中最大的 $k+1$ 个位置，因而这部分复杂度可以接受。

对于高度相同的情况，可以预处理时钦定一个顺序。

复杂度 $O(nk^2+k^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 25050
#define K 28
#define mod 1000000007
int n,k,v[N],he[N],st[N],lmx[N][K],lid[N][K],rmx[N][K],rid[N][K],lv[N],rv[N],as;
int ldp[N][K][K][2],rdp[N][K][K][2];
bool cmp(int a,int b){return v[a]<v[b];}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),st[i]=i;
	sort(st+1,st+n+1,cmp);
	for(int i=1;i<=n;i++)he[st[i]]=i;
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=k+1;j++)lmx[i][j]=lmx[i-1][j];
		for(int j=1;j<=k+1;j++)if(!lmx[i][j]||he[lmx[i][j]]<he[i])
		{
			for(int l=k+1;l>j;l--)lmx[i][l]=lmx[i][l-1];
			lmx[i][j]=i;lv[i]=j;break;
		}
		for(int l=1;l<=k+1;l++)for(int s=l;s<=l+1;s++)if(lmx[i-1][l]==lmx[i][s])lid[i-1][l]=s;
	}
	for(int i=n;i>=1;i--)
	{
		for(int j=1;j<=k+1;j++)rmx[i][j]=rmx[i+1][j];
		for(int j=1;j<=k+1;j++)if(!rmx[i][j]||he[rmx[i][j]]<he[i])
		{
			for(int l=k+1;l>j;l--)rmx[i][l]=rmx[i][l-1];
			rmx[i][j]=i;rv[i]=j;break;
		}
		for(int l=1;l<=k+1;l++)for(int s=l;s<=l+1;s++)if(rmx[i+1][l]==rmx[i][s])rid[i+1][l]=s;
	}
	ldp[0][0][0][0]=1;
	for(int i=1;i<=n;i++)
	for(int j=0;j<=k;j++)
	for(int l=0;l<=k+1;l++)for(int f=0;f<2;f++)if(ldp[i-1][j][l][f])
	for(int s=0;s<=(j!=k);s++)
	{
		int r1=lid[i-1][l],r2=j,r3=f;
		if(!s&&(lv[i]&&(lv[i]<r1||!r1)))r1=lv[i];
		r3^=v[lmx[i][r1]]&1;
		if(!s)r3^=v[i]&1;else r2++;
		ldp[i][r2][r1][r3]=(ldp[i][r2][r1][r3]+ldp[i-1][j][l][f])%mod;
	}
	rdp[n+1][0][0][0]=1;
	for(int i=n;i>=1;i--)for(int j=0;j<=k;j++)for(int l=0;l<=k+1;l++)for(int f=0;f<2;f++)if(rdp[i+1][j][l][f])
	for(int s=0;s<=(j!=k);s++)
	{
		int r1=rid[i+1][l],r2=j,r3=f;
		if(!s&&(rv[i]&&(rv[i]<r1||!r1)))r1=rv[i];
		r3^=v[rmx[i][r1]]&1;
		if(!s)r3^=v[i]&1;else r2++;
		rdp[i][r2][r1][r3]=(rdp[i][r2][r1][r3]+rdp[i+1][j][l][f])%mod;
	}
	for(int i=1;i<=n;i++)if(he[i]>=n-k)
	for(int p=0;p<=k+1;p++)for(int q=0;q<=k+1;q++)if(he[lmx[i-1][p]]<he[i]&&he[rmx[i+1][q]]<he[i])
	for(int s=0;s<=k;s++)for(int t=0;t<2;t++)
	as=(as+1ll*ldp[i-1][s][p][t]*rdp[i+1][k-s][q][t])%mod;
	printf("%d\n",as);
}
```

##### Balanced Rainbow Sequence

###### Problem

有一个长度为 $n$ 的不一定合法的括号序列，每个括号有一个颜色，颜色为红，蓝，白中的一种。

你可以多次翻转任意一个括号，你需要使得序列满足如下条件：

1. 如果删去所有红色括号，得到的是一个合法括号序列。
2. 如果删去所有蓝色括号，得到的是一个合法括号序列。

求最少翻转次数或输出无解。

$n\leq 6000$

$2s,512MB$

###### Sol

考虑一种颜色的括号翻转的性质，显然将一个更靠左的 `)` 翻过来更优，因而可以发现每种颜色的括号一定是翻转开头的一些 `)`和结尾的一些 `(` 。

同时可以发现，如果将左侧的一个 `(` 翻转为 `)`，同时将右侧的一个 `)`翻转为 `(` ，这样一定不优。因此不存在这种情况。

考虑枚举白色括号中，翻转的左右括号数量的差。由于白色加上另外两种颜色中的任意一种都合法，因此另外两种颜色的括号翻转的两种括号之差可以确定。

此时先将两者的差翻转，接下来操作可以看成，每次选择一种颜色的括号，将这种颜色的括号的第一个 `)` 和最后一个 `(` 翻转。设三种颜色的括号翻转的次数为 $c_r,c_b,c_w$。

考虑将左右括号看作 $1,-1$，则考虑某种颜色的括号中，某一个前缀的前缀和与翻转次数的关系。

设这个前缀左侧的 `)` 有 $a$ 个，右侧的 `(` 有 $b$ 个，则翻转 $k$ 次后，前缀和增加值为 $2(\min(a,k)-\max(b-k,0))$。但注意到如果两个都取到左侧，则此时相当于 `)` 翻转到了这个位置之后，`(` 翻转到了这个位置之前，但这种情况不优，因此不会出现这种情况。

此时可以发现，令 $t=\min(a,b)$，则上述值相当于 $\min(t,k)$。

因此对于题目中要求合法的两种颜色，以及一个前缀，这个前缀和大于等于 $0$ 的条件相当于 $\min(a,c_r)+\min(b,c_w)\geq c$，这相当于三个条件 $c_r\geq v_1,c_w\geq v_2,c_r+c_w\geq v_3$。

因此可以求出所有这些条件，最后所有不等式形如：
$$
c_r\geq v_1,c_b\geq v_2,c_w\geq v_3,c_r+c_w\geq v_4,c_b+c_w\geq v_5
$$
枚举 $c_w$ 判断即可。

复杂度 $O(n^2)$，注意细节。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 6050
int n,v[N],as=1e9,su[N][2],sm[N][3],sr[N][3];
char s[N],s2[N],s3[N];
void solve()
{
	for(int i=1;i<=n;i++)s2[i]=s[i];
	int lv=0,fg=1;
	while(1)
	{
		int lv1=lv,fg1=1;
		if(!fg)break;
		for(int i=1;i<=n;i++)s3[i]=s2[i];
		for(int t=0;t<2;t++)
		{
			int s1=0;
			for(int i=1;i<=n;i++)if(v[i]!=(t^1))s1+=s3[i]=='('?1:-1;
			if(s1%2){fg1=0;break;}
			for(int i=1;i<=n;i++)if(s3[i]==')'&&v[i]==t&&s1<0)s3[i]^=1,s1+=2,lv1++;
			for(int i=n;i>=1;i--)if(s3[i]=='('&&v[i]==t&&s1>0)s3[i]^=1,s1-=2,lv1++;
			if(s1){fg1=0;break;}
		}
		fg=0;lv++;
		for(int i=n;i>=1;i--)if(s2[i]=='('&&v[i]==2&&!fg)s2[i]^=1,fg=1;
		if(!fg1)continue;
		for(int t=0;t<3;t++)
		{
			for(int i=1;i<=n;i++)
			{
				su[i][0]=su[i-1][0];su[i][1]=su[i-1][1];
				if(v[i]==t)su[i][s3[i]-'(']++;
			}
			for(int i=1;i<=n;i++)sm[i][t]=min(su[i][1],su[n][0]-su[i][0]),sr[i][t]=su[i][0]-su[i][1];
		}
		int mx[4]={0},f1=0,f2=0,f3=0,f13=0,f23=0;
		for(int i=1;i<=n;i++)
		{
			for(int t=0;t<3;t++)mx[t]=max(mx[t],sm[i][t]);
			f1=max(f1,(-sr[i][0]-sr[i][2]+1)/2-sm[i][2]);f2=max(f2,(-sr[i][1]-sr[i][2]+1)/2-sm[i][2]);
			f3=max(f3,(-sr[i][0]-sr[i][2]+1)/2-sm[i][0]);f3=max(f3,(-sr[i][1]-sr[i][2]+1)/2-sm[i][1]);
			f13=max(f13,-sr[i][0]-sr[i][2]);f23=max(f23,-sr[i][1]-sr[i][2]);
		}
		f13=(f13+1)/2;f23=(f23+1)/2;
		for(int t=f3;t<=mx[2];t++)
		{
			int l1=max(f1,f13-t),l2=max(f2,f23-t);
			if(l1>mx[0]||l2>mx[1])continue;
			as=min(as,lv1+(t+l1+l2)*2);
		}
	}
}
int main()
{
	scanf("%d%s",&n,s+1);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	solve();
	for(int i=1;i*2<=n;i++)swap(v[i],v[n-i+1]),swap(s[i],s[n-i+1]);
	for(int i=1;i<=n;i++)s[i]^=1;
	solve();
	printf("%d\n",as>n?-1:as);
}
```

##### Machine Learning

###### Problem

给出 $n$ 对 $(x_i,y_i)$，保证 $x_i$ 两两不同。

找一个连续函数 $f$ ，使得这个连续函数可以被分成两段，每一段都是一个线性函数。

最小化： $\frac1n(\sum(f(x_i)-y_i)^2)$，误差不超过 $1$。

$n\leq 10^5,x_i\leq 10^6,y_i\leq 1000$

$4s,256MB$

###### Sol

考虑枚举分界点在哪一段 $(x_i,x_{i+1})$ 中，设此时 $y$ 的值为 $a$，两侧斜率为 $b,c$，则代价为：
$$
\sum_{k=1}^i(a+b(x_k-x)-y_k)^2+\sum_{k=i+1}^n(a+c(x_k-x)-y_k)^2
$$
考虑左侧情况，$b$ 相关的项为：
$$
(\sum_{k=1}^i(x_k-x)^2)b^2+(\sum_{k=1}^i2(a-y_k)(x_k-x))b
$$
那么取最小值时有这东西导数为 $0$，得到：
$$
b=\frac{\sum_{k=1}^i(a-y_k)(x_k-x)}{\sum_{k=1}^i(x_k-x)^2}
$$
可以得到 $a$ 相关的项为：
$$
\sum_{k=1}^i(1+\frac{(x_k-x)\sum_{t=1}^i(x_t-x)}{\sum_{t=1}^i(x_t-x)^2})^2a^2-2\sum_{k=1}^i(1+\frac{(x_k-x)\sum_{t=1}^i(x_t-x)}{\sum_{t=1}^i(x_t-x)^2})*(1+\frac{(x_k-x)\sum_{t=1}^iy_t(x_t-x)}{\sum_{t=1}^i(x_t-x)^2})a
$$
另外一侧同理，通分后可得系数是一个 $8$ 次多项式，对 $a$ 求导可以发现 $a$ 的最优解可以被表示成 $8$ 次分式，再带回原式，可以将答案表示为关于 $x$ 的 $20$ 次分式。

再考虑对 $x$ 求导，则相当于判断一个 $39$ 次方程的零点，因此理论上可以做到 $O(n\log v)$，其中最后一部分是求根的复杂度。

但是这样的常数至少有 $10^4$，且不精细实现的常数可能接近 $10^6$，同时系数大概有 $10^{60}$ 级别，因此完全不行。 

因此考虑猜测答案对于 $x$ 是单峰的，这样先对 $x$ 三分，然后剩下的都可以直接算出来，可以得到常数正常的 $O(n\log v)$ 做法。

也许将上面那堆算出来就能证明，但是我不想算了。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define db long double
int n,s[N][2],tp[N];
db sx[N],sy[N],sxy[N],sx2[N],sy2[N],as=1e18;
db solve(int k,db v)
{
	db a2=n,a=-2*sy[n],ab=2*(v*k-sx[k]),b2=k*v*v-2*v*sx[k]+sx2[k],b=-2*v*sy[k]+2*sxy[k];
	db ac=2*(v*(n-k)-sx[n]+sx[k]),c2=(n-k)*v*v-2*v*(sx[n]-sx[k])+sx2[n]-sx2[k],c=-2*v*(sy[n]-sy[k])+2*(sxy[n]-sxy[k]);
	db ci=sy2[n];
	db v1=ab,v2=b;v1/=-2*b2,v2/=-2*b2;
	a2+=ab*v1,a+=ab*v2;
	a+=b*v1;ci+=b*v2;
	a2+=b2*v1*v1;a+=b2*v1*v2*2,ci+=b2*v2*v2;
	v1=ac,v2=c;v1/=-2*c2,v2/=-2*c2;
	a2+=ac*v1,a+=ac*v2;
	a+=c*v1;ci+=c*v2;
	a2+=c2*v1*v1;a+=c2*v1*v2*2,ci+=c2*v2*v2;
	if(a2<=1e-16)return ci;
	return ci-a*a/4/a2;
}
bool cmp(int a,int b){return s[a][0]<s[b][0];}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),tp[i]=i;
	sort(tp+1,tp+n+1,cmp);
	for(int i=1;i<=n;i++)
	{
		int x=s[tp[i]][0],y=s[tp[i]][1];
		sx[i]=sx[i-1]+x;sy[i]=sy[i-1]+y;sxy[i]=sxy[i-1]+x*y;
		sx2[i]=sx2[i-1]+(db)1.0*x*x,sy2[i]=sy2[i-1]+y*y;
	}
	if(n==1)as=0;
	for(int i=1;i<n;i++)
	{
		db lb=s[tp[i]][0],rb=s[tp[i+1]][0];
		db mid=(lb+rb)/2;
		for(int t=1;t<=40;t++)
		{
			db mid1=(lb*2+rb)/3,mid2=(lb+rb*2)/3;
			db as1=solve(i,mid1),as2=solve(i,mid2);
			if(as1>as2)as=min(as,as2),lb=mid1;
			else as=min(as,as1),rb=mid2;
		}
	}
	printf("%.14lf\n",(double)as/n);
}
```

##### Automorphism

###### Problem

给一棵有根树，初始只有一个节点，支持 $q$ 次操作：

1. 向树中加入一个叶子节点。
2. 给定某个点，询问只考虑这个点的子树，子树内的有根自同构数量，模 $998244353$。

$q\leq 3\times 10^5$

$8s,512MB$

###### Sol

可以发现所有的自同构都可以看成交换儿子顺序，如果求出每个子树的hash值，则方案数为对于每个点，儿子的每种hash出现次数的阶乘的乘积。

考虑找一种可以快速维护的hash方式，例如：

将子树内的所有hash相乘，再 $*a+b$，然后可以再加一个当前点儿子数量。hash对大质数取模。

这可以看成 $*a$ 再乘轻儿子的hash乘积，因此可以考虑ddp维护。先离线求出最后的树结构，然后链剖线段树即可 $O(\log^2 n)$ 加入，$O(\log n)$ 查询子树hash。

同时注意到如果当前一个点的子树大小大于父亲子树大小的一半，则这个子树不会影响自同构数量。因此考虑对于每个点只维护这个点所有子树大小不超过自己子树大小一半的儿子的hash，修改一个点时，只需要修改这个点祖先中满足这个条件的点。这样每个点祖先中只有 $O(\log n)$ 个需要修改的点。

考虑记录当前的所有关键点，使用set维护dfs序即可 $O(\log n)$ 求出单次询问需要的关键点。对于每个关键点进行修改，同时判断关键点的是否会发生变化，这里可以记录每个点当前子树大小最大的儿子，即可维护关键点。

对于每个点可以使用map维护当前所有子树大小没有超过一半的儿子的hash值，这样即可求出每个点的儿子间交换的方案数。最后的子树方案数可以看成dfs序在一段区间内的点的儿子交换方案数的乘积，因此这部分使用BIT维护即可。

注意一些常数问题，比如set的遍历，以及ddp时修改一个轻儿子后更新时，求逆元exgcd比快速幂快一倍。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
#include<vector>
using namespace std;
#define N 300500
#define ll long long
#define mod 1000000000000000003
int q,n,s[N][2],f[N],head[N],cnt,sz[N],sn[N],tp[N],id[N],dep[N],rd[N],tid[N],rb[N],ct;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs0(int u,int fa)
{
	dep[u]=dep[fa]+1;sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs0(ed[i].t,u);sz[u]+=sz[ed[i].t];
		if(sz[ed[i].t]>sz[sn[u]])sn[u]=ed[i].t;
	}
}
void dfs1(int u,int fa,int v)
{
	id[u]=++ct;tp[u]=v;tid[ct]=u;
	if(sn[u])dfs1(sn[u],u,v),rd[u]=rd[sn[u]];else rd[u]=u;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs1(ed[i].t,u,ed[i].t);
	rb[u]=ct;
}
ll exgcd(ll a,ll b,ll &x,ll &y)
{
	if(!b){x=1,y=0;return a;}
	ll r=a/b,g=exgcd(b,a-b*r,y,x);y-=r*x;return g;
}
ll getinv(ll x,ll p)
{
	ll s,t;exgcd(x,p,s,t);
	return (s%p+p)%p;
}
ll vl[N],ls[N];
int is[N],rid[N];
struct sth{ll a,b;sth(){a=1;b=0;}sth(ll a1,ll b1){a=a1,b=b1;}};
sth operator +(sth a,sth b){return (sth){(__int128)a.a*b.a%mod,((__int128)a.b*b.a+b.b)%mod};}
struct node{int l,r,is;sth su;}e[N*4];
void pushup(int x){if(e[x].is)e[x].su=e[x<<1|1].su+e[x<<1].su;}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;rid[l]=x;
	if(tp[tid[e[x].l]]==tp[tid[e[x].r]])e[x].is=1;
	if(l==r)return;
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void modify(int x,int v,sth tp)
{
	x=rid[v];e[x].su=tp;
	while(1)
	{
		x>>=1;
		if(!e[x].is)return;
		pushup(x);
	}
}
ll query(int x,int l,int r,ll v)
{
	if(e[x].l==l&&e[x].r==r)return ((__int128)e[x].su.a*v+e[x].su.b)%mod;
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)return query(x<<1,l,r,v);
	else if(mid<l)return query(x<<1|1,l,r,v);
	else return query(x<<1,l,mid,query(x<<1|1,mid+1,r,v));
}
ll query1(int x)
{
	ll vl=1+is[rd[x]]*233ll;
	if(rd[x]==x)return vl;
	return query(1,id[x],id[rd[x]]-1,vl);
}
void modify1(int x)
{
	is[x]=1;is[f[x]]++;
	modify(1,id[x],(sth){ls[x],is[x]*233ll});
	if(tp[x]!=x)modify(1,id[f[x]],(sth){ls[f[x]],is[f[x]]*233ll});
	while(x)
	{
		x=tp[x];
		ls[f[x]]=(__int128)ls[f[x]]*getinv(vl[x],mod)%mod;
		vl[x]=query1(x);
		ls[f[x]]=(__int128)ls[f[x]]*vl[x]%mod;
		x=f[x];
		if(x)modify(1,id[x],(sth){ls[x],is[x]*233ll});
	}
}
struct BIT1{
	int tr[N];
	void add(int x){for(int i=x;i<=n;i+=i&-i)tr[i]++;}
	int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}tr;
int qsz(int x){return tr.que(rb[x])-tr.que(id[x]-1);}
#define md 998244353
struct BIT2{
	int tr[N];
	void add(int x,int k){for(int i=x;i<=n;i+=i&-i)tr[i]=1ll*tr[i]*k%md;}
	int que(int x){int as=1;for(int i=x;i;i-=i&-i)as=1ll*as*tr[i]%md;return as;}
}t2;
int query2(int x){return 1ll*t2.que(rb[x])*getinv(t2.que(id[x]-1),md)%md;}
set<int> fr;
int hs[N],inv[N];
ll hv[N];
struct sth1{
	set<pair<ll,int> > fr;
	int add(ll v)
	{
		set<pair<ll,int> >::iterator it=fr.lower_bound(make_pair(v,0));
		if(it==fr.end()||(*it).first>v)return fr.insert(make_pair(v,1)),1;
		else
		{
			pair<ll,int> f1=*it;fr.erase(it);
			f1.second++;fr.insert(f1);return f1.second;
		}
	}
	int del(ll v)
	{
		set<pair<ll,int> >::iterator it=fr.lower_bound(make_pair(v,0));
		pair<ll,int> f1=*it;fr.erase(it);
		int as=inv[f1.second];
		f1.second--;if(f1.second)fr.insert(f1);return as;
	}
}st[N];
void modify2(int x)
{
	tr.add(id[x]);
	vector<int> rs;
	int nw=x;
	while(nw)
	{
		int lb=id[tp[nw]],rb=id[nw];
		set<int>::iterator it=fr.lower_bound(rb+1);
		while(1)
		{
			if(it==fr.begin())break;
			it--;
			if(*it<lb)break;
			rs.push_back(tid[*it]);
		}
		nw=f[tp[nw]];
	}
	for(int i=0;i<rs.size();i++)
	{
		int x=rs[i];
		if(hv[x])t2.add(id[f[x]],st[f[x]].del(hv[x]));
		hv[x]=query1(x);t2.add(id[f[x]],st[f[x]].add(hv[x]));
		if(hs[f[x]]&&qsz(hs[f[x]])*2<=qsz(f[x]))
		{
			hv[hs[f[x]]]=query1(hs[f[x]]);
			t2.add(id[f[x]],st[f[x]].add(hv[hs[f[x]]]));
			fr.insert(id[hs[f[x]]]);hs[f[x]]=0;
		}
		if(qsz(x)*2>qsz(f[x]))st[f[x]].del(hv[x]),hs[f[x]]=x,fr.erase(id[x]);
	}
}
int main()
{
	scanf("%d",&q);n=1;
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(s[i][0]==0)n++,f[n]=s[i][1],adde(s[i][1],n),s[i][1]=n;
	}
	dfs0(1,0);dfs1(1,0,1);build(1,1,n);
	for(int i=1;i<=n;i++)t2.tr[i]=vl[i]=ls[i]=1;
	for(int i=2;i<=n;i++)fr.insert(i);
	for(int i=1;i<=n;i++)inv[i]=getinv(i,md);
	for(int i=1;i<=q;i++)
	if(s[i][0]==0)modify1(s[i][1]),modify2(s[i][1]);
	else printf("%d\n",query2(s[i][1]));
}
```

##### TwoSquares

###### Problem

有一个 $n\times n$ 的 $01$ 矩阵，你可以进行不超过两次操作，每次选择一个正方形区域并翻转。求操作后矩阵中 $1$ 的个数的最大值。

$n\leq 100$

$4s,256MB$

###### Sol

对于操作 $0,1$ 次的情况，直接做是 $O(n^3)$ 的。

对于两次的情况，考虑分类讨论：

1. 正方形不相交

此时一定存在一条横向或者纵向的边将选择的正方形分隔开，枚举这条线即可。

复杂度 $O(n^3)$

2. 正方形完全包含

定义一个正方形的权值为翻转后增加的 $1$ 的数量。则枚举里面的正方形，相当于需要找到包含它的权值最大的正方形。

可以设 $dp_s$ 表示包含正方形 $s$ 的权值最大的正方形的权值，按照正方形大小从大到小转移。

复杂度 $O(n^3)$

3. 相交区域为正方形的一个角

直接的做法是枚举相交区域，然后相当于求以某个位置为左上或右下角，且边长大于等于某个数的最大正方形权值，可以预处理后缀和。

复杂度 $O(n^4)$

另外一种做法是枚举相交区域中两条平行的边，且钦定相交区域中距离更远的边是这一对，则此时边上每个点作为左上/右下角的最大权值已知，相当于对于每个右下角求出左上角在它左侧且距离它不超过某个数的左上角中权值最大的，扫描线+单调队列即可。

复杂度 $O(n^3)$

4. 相交部分在一条边上

设相交的点为 $(x,a),(x,b)$，则相当于需要满足如下条件：

小正方形的两条边为 $y=a,b$，且经过横坐标为 $x$ 。

大正方形的一条边为横坐标为 $x$，且纵向两条边包含区间 $[a,b]$。

枚举 $x$，只考虑大正方形在横坐标小于 $x$ 的情况，此时包含一个区间 $[a,b]$ 的正方形的最大权值可以dp预处理。

然后枚举小正方形的位置即可计算答案，旋转四次即可考虑到所有情况。复杂度 $O(n^4)$

另外一种做法是预处理大正方形的情况后，枚举小正方形的两条纵向边，此时设小正方形的第一条横向边为 $x=l$，大正方形的最后一条横向边为 $x=r$，则需要满足 $r-l\leq b-a$，且这种情况的权值为小正方形翻转的权值加上大正方形之前预处理的包含这个区间的最大权值，再减去两倍中间重复覆盖的权值。

注意到中间重复覆盖的权值可以前缀和，变为只和 $l,r$ 相关的量，因而这里仍然可以扫描线+单调队列解决，复杂度 $O(n^3)$。

总复杂度 $O(n^3)$ 或 $O(n^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<string>
using namespace std;
#define N 105
int n,v[N][N],su[N][N],as=0,ls[N][N][N],rs[N][N][N],li[N],ri[N],fr[N][N],fs[N][N][N],sc;
char s[N][N],s2[N][N];
int calc(int lx,int ly,int rx,int ry){return su[rx][ry]-su[rx][ly-1]-su[lx-1][ry]+su[lx-1][ly-1];}
void solve()
{
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)su[i][j]=su[i-1][j]+su[i][j-1]-su[i-1][j-1]+(s[i][j]=='1');
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)for(int l=1;l<=n;l++)ls[i][j][l]=rs[i][j][l]=-1e8;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	{
		int tp=min(i,j),mx=-1e9;
		for(int d=tp;d>=1;d--)
		{
			int sv=d*d-calc(i-d+1,j-d+1,i,j)*2;
			if(sv>mx)mx=sv;
			ls[i][j][d]=mx;
		}
	}
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	{
		int tp=min(n-i+1,n-j+1),mx=-1e9;
		for(int d=tp;d>=1;d--)
		{
			int sv=d*d-calc(i,j,i+d-1,j+d-1)*2;
			if(sv>mx)mx=sv;
			rs[i][j][d]=mx;
		}
	}
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	for(int p=i;p<=n;p++)for(int q=j;q<=n;q++)
	as=max(as,ls[p][q][max(p-i+1,q-j+1)]+rs[i][j][max(p-i+1,q-j+1)]-2*(p-i+1)*(q-j+1)+4*calc(i,j,p,q));
	for(int i=1;i<=n;i++)li[i]=ri[i]=-1e9;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	{
		int tp=min(i,j);
		for(int d=tp;d>=1;d--)
		{
			int sv=d*d-calc(i-d+1,j-d+1,i,j)*2;
			as=max(as,sv);
			li[i-d+1]=max(li[i-d+1],sv);
			ri[i]=max(ri[i],sv);
			fs[i][j][d]=sv;
		}
	}
	for(int i=1;i<n;i++)for(int j=i+1;j<=n;j++)as=max(as,ri[i]+li[j]);
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)for(int k=j;k<=n;k++)
		{
			fr[j][k]=-1e9;
			int li=i-k+j;
			if(li>=1)fr[j][k]=(k-j+1)*(k-j+1)-calc(li,j,i,k)*2;
		}
		for(int l=n;l>=1;l--)for(int i=1;i+l-1<=n;i++)
		{
			int j=i+l-1;
			if(i>1)fr[i][j]=max(fr[i][j],fr[i-1][j]);
			if(j<n)fr[i][j]=max(fr[i][j],fr[i][j+1]);
		}
		for(int j=1;j<=n;j++)for(int k=j;k<=n;k++)
		for(int l=max(1,i-(k-j));l<=i&&l+(k-j)<=n;l++)
		as=max(as,fr[j][k]+(k-j+1)*(k-j+1)-2*calc(l,j,l+(k-j),k)-2*(i-l+1)*(k-j+1)+4*calc(l,j,i,k));
	}
	for(int d=n;d>=1;d--)
	for(int i=d;i<=n;i++)for(int j=d;j<=n;j++)
	for(int t=0;t<2;t++)for(int s=0;s<2;s++)fs[i-t][j-s][d-1]=max(fs[i-t][j-s][d-1],fs[i][j][d]);
	for(int d=n;d>=1;d--)
	for(int i=d;i<=n;i++)for(int j=d;j<=n;j++)
	as=max(as,fs[i][j][d]-(d*d-calc(i-d+1,j-d+1,i,j)*2));
}
struct TwoSquares{
	int maxOnes(vector<string> s1)
	{
		n=s1.size();
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=s1[i-1][j-1];
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)sc+=s[i][j]=='1';
		for(int t=0;t<4;t++)
		{
			solve();
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s2[j][n-i+1]=s[i][j];
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=s2[i][j];
		}
		return as+sc;
	}
};
```

##### Bearpairs

###### Problem

给一个长度为 $n$ 的正整数序列 $s$，以及长度相同的非负整数序列 $v$。

现在有 $n$ 个元素 $\{1,2,...,n\}$，你需要将它们配成 $\frac{n-k}2$ 对，使得每个元素最多在这些对中出现一次。

对于每一对 $(i,j)$，你需要满足 $s_i,s_j$ 不同，这一对的代价为 $v_i+v_j+100|i-j|$。

求最小总配对代价，或输出无解。

$n\leq 2500,s_i,k\leq 6$

$14s,256MB$

###### Sol

考虑匹配的性质，可以得到如下结论：

1. 存在一种最优解没有交叉匹配。

如果存在交叉匹配 $(a,c),(b,d)$，则如果 $(a,b),(c,d)$ 可行，则这种方案更优。否则，可以得到 $s_a\neq s_c,s_b\neq s_d$ 且 $s_a=s_b$ 或者 $s_c=s_d$。两种情况中都不难发现 $(a,d),(b,c)$ 可行。因此存在一种最优解，其中不存在这样的交叉匹配。

此时所有的匹配可以看成区间，区间的嵌套关系可以看成树结构。考虑一条链的性质：

2. 对于最优解中若干个两两嵌套的匹配区间，它们的左端点 $s_i$ 全部相同，或者右端点 $s_i$ 全部相同。

考虑这样的若干个括号，因为匹配 $(a,d),(b,c)$ 改为 $(a,b),(c,d)$ 更优，因此最优解中这样的括号满足任意两个括号的左端点相同或者右端点相同。

此时可以看成，有一个 $n$ 个点的完全图，每条边为红色或蓝色，分别代表对应的两对匹配左端点相同或者右端点相同。此时不难得到一定有一种颜色构成连通图，因此一定有一侧全部相同。

此时可以考虑 $dp$，显然匹配的代价可以拆开，设 $f_{i,j,p,q,0/1}$ 表示处理了前 $i$ 个位置，当前跨过 $i$ 的匹配有 $j$ 对，左侧有 $p$ 个位置没有匹配。由上一个结论，这 $j$ 对匹配满足左端点相同或者右端点相同。因此 $0/1$ 表示相同的是左端点还是右端点，$q$ 表示这种端点的颜色，$f$ 表示这种情况的最优解。

但此时可以发现 $1$ 的状态转移到 $0$ 的状态时，会要求现在左侧的匹配点都满足是某种颜色，而扫过去的dp无法处理这种情况。但此时还有如下结论：

3. 存在一种最优解中不存在上一种转移，即不存在匹配 $(a,f),(b,c),(d,e)$，且dp中钦定方式为钦定 $s_c=s_f,s_d=s_a$。

如果 $s_b\neq s_a$ 且 $s_e\neq s_f$，则可以改为更优的匹配 $(a,b),(c,d),(e,f)$。否则，如果左侧成立，则可以将 $(b,c)$ 改为钦定左侧，如果右侧匹配则可以改右侧的钦定。因此一定可以消除一次这种情况。

从而，对于一棵树，一定存在一个分界点，使得分界点左侧的所有左端点都被钦定，分界点右侧的所有右端点都被钦定。且不难发现两侧钦定的颜色都只有一种。

因此上面的dp中可以只考虑 $0\to 0,0\to 1,1\to 1$ 的情况。转移形如：

对于加入一个元素的情况，有：

跳过这个元素：$f_{i,j,p,q,0/1}\to f_{i+1,j,p+1,q,0/1}$

作为树中的左端点：$f_{i,j,p,q,0/1}-100*(i+1)+v_{i+1}\to f_{i+1,j+1,p,q,0/1}$，条件为最后一维为 $0$ 且 $s_{i+1}=q$ 或者最后一维为 $1$ 且 $s_{i+1}\neq q$

作为树中的右端点：$f_{i,j,p,q,0/1}+100*(i+1)+v_{i+1}\to f_{i+1,j-1,p,q,0/1}$，条件与上一个相反。

作为一棵树的开头：$f_{i,0,p,q,0/1}-100*(i+1)+v_{i+1}\to f_{i+1,1,p,s_{i+1},0}$

加入元素后，考虑中间分界点的转移，有：

$f_{i,j,p,q,0}\to f_{i,j,p,t,1}(q\neq t)$

直接转移的复杂度为 $O(n^2s^2k)$，容易做到 $O(n^2sk)$，都可以通过。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<string>
#include<vector>
using namespace std;
#define N 2550
int n,k,v[N];
char s[N];
int dp[N][7][7][2],rs[N][7][7][2];
struct BearPairs{
	int minCost(string s1,vector<int> v1,int k)
	{
		n=s1.size();
		for(int i=1;i<=n;i++)v[i]=v1[i-1],s[i]=s1[i-1];
		for(int i=0;i<=n/2;i++)for(int j=0;j<7;j++)for(int l=0;l<=k;l++)for(int t=0;t<2;t++)if(i+j+l)dp[i][j][l][t]=1.1e9;
		for(int i=1;i<=n;i++)
		{
			for(int d=0;d<=n/2;d++)for(int j=0;j<6;j++)for(int l=0;l<=k;l++)for(int t=0;t<2;t++)rs[d][j][l][t]=1.1e9;
			for(int d=0;d<=n/2;d++)for(int j=0;j<6;j++)for(int l=0;l<=k;l++)for(int t=0;t<2;t++)if(dp[d][j][l][t]<1e9)
			for(int r=0;r<3;r++)
			{
				int nd=d,nj=j,nl=l,nt=t,nv=dp[d][j][l][t];
				if(r==1)nv+=v[i]-i*100;
				if(r==2)nv+=v[i]+i*100;
				if(!r){nl++;if(nl>k)continue;}
				else if(r==1&&d)
				{
					if(t^(s[i]-'a'!=j))continue;
					nd++;
				}
				else if(r==2&&d)
				{
					if(t^(s[i]-'a'==j))continue;
					nd--;
				}
				else if(r==2&&!d)continue;
				if(!nd)nt=nj=0;
				if(!r||d)rs[nd][nj][nl][nt]=min(rs[nd][nj][nl][nt],nv);
				else for(int p=0;p<6;p++)rs[1][p][l][s[i]-'a'!=p]=min(rs[1][p][l][s[i]-'a'!=p],nv);
				if(nd&&!nt)for(int p=0;p<6;p++)if(p!=nj)rs[nd][p][nl][1]=min(rs[nd][p][nl][1],nv);
			}
			for(int d=0;d<=n/2;d++)for(int j=0;j<6;j++)for(int l=0;l<=k;l++)for(int t=0;t<2;t++)dp[d][j][l][t]=rs[d][j][l][t];
		}
		return dp[0][0][k][0]<1e9?dp[0][0][k][0]:-1;
	}
};
```

##### Quasi-template

###### Problem

称字符串 $t$ 是字符串 $s$ 的模板，当且仅当 $t$ 是 $s$ 的子串，且 $t$ 在 $s$ 中的所有出现位置可以完全覆盖 $s$。

称 $t$ 是 $s$ 的半模板，当且仅当 $t$ 是 $s$ 的子串，且存在一个包含 $s$ 作为子串的字符串 $s'$，使得 $t$ 是 $s'$ 的模板。

给出长度为 $n$ 的字符串 $s$，求 $s$ 的半模板数量，并求出长度最小的半模板中字典序最小的。

$n\leq 2\times 10^5$

$1s,256MB$

###### Sol

半模板相当于 $t$ 在 $s$ 中所有的出现能覆盖中间部分，且前缀能被 $t$ 的后缀覆盖，后缀能被 $t$ 的前缀覆盖。

考虑中间部分，如果在若干次完整出现的中间有一个位置没有被覆盖，则因为两侧都有完整的覆盖，因而前缀后缀的覆盖不可能覆盖这个位置。因此这种情况不合法。

考虑建SAM，使用线段树合并维护endpos，则中间部分需要满足串长度大于等于相邻两个endpos的差，这容易在线段树上维护出。

考虑枚举SAM上的每一个节点判断，则线段树中可以得到endpos的第一个位置和最后一个位置，以及相邻两个差的最大值。中间部分相当于要求串的长度大于等于某个值。

考虑两侧部分的限制。此时当前点中的字符串相当于给定结束位置 $r$，且初始位置在 $[a,b]$ 间的所有字符串。

对于开头的限制，可以发现随着长度增加，需要覆盖的前缀长度会变小。且字符串是向左侧增加，因而增加字符不会影响之前该串的后缀覆盖某个前缀的过程，因此可以覆盖的前缀长度单调不减。因此这部分的限制一定相当于限制长度大于等于某个值。

考虑二分判断，则相当于求 $[l,r]$ 中是否存在一个后缀是 $s$ 的前缀满足这个前缀长度大于等于 $d$。这可以看成是否存在原串中 $[l,r-d+1]$ 中的一个后缀，满足这个后缀与原串的lcp至少从后缀开头开始延伸到了 $r$。因此预处理出延伸长度，然后相当于询问区间最大值，可以st表处理。这部分复杂度 $O(n\log n)$，且之后合法开头还是一个区间。

然后结尾的限制。此时需要覆盖的结尾部分长度固定，但向开头加字符会影响覆盖的过程，因此此时不具有可二分性。此时相当于判断 $[l,r]$ 是否存在一个前缀，使得这个前缀是 $s$ 的后缀且能覆盖 $[r+1,n]$。

考虑枚举这个前缀的结尾 $u$，则相当于如下条件：

1. $u\leq r$
2. $lcs(s[1,u],s[1,n])\geq u-l+1$
3. $u-l+1\geq n-r+1$

考虑从小到大枚举 $r$ 考虑询问，则相当于加入合法的 $u$。可以发现每个 $u$ 会影响一段以 $u$ 结尾的区间。

设 $t_l$ 表示对于一个 $l$，当前所有满足条件的 $u$ 中 $u-l+1$ 的最大值。则可以发现加入一个 $u$ 时，能影响的的区间都会以新的 $u$ 作为最大值，因此相当于将一段后缀的 $t_l$ 改为 $u-l+1$。

而合法条件是 $t_l\geq n-r+1$，可以发现随着 $r$ 增加，$t_l$ 会增加，而 $n-r+1$ 会减少，因此对于一个 $l$，$r$ 的合法具有单调性。此时可以二分答案，再在单调栈上二分，$O(n\log^2n)$ 求出对于每个 $l$ 最小的合法 $r$。

但还有更好的做法，可以分开考虑每个 $u$ 导致的合法，考虑一个 $u$ 的贡献，它一定相当于以下两种形式之一：

1. 当前使得一段区间 $[l,r]$ 合法，且接下来的每个时刻会使得区间向右扩展一位，即 $t$ 个时刻后使得位置 $r+t$ 合法。
2. 当前不使得任何位置合法，但之后的某个时刻使得 $l$ 合法，接下来每个时刻使得后面的一个位置合法。

对于区间的操作，可以set维护当前所有不是合法的点，即可区间变合法。剩下的操作相当于有若干个位置，每次这些位置变得合法，然后这些位置向右移动。因为操作 $1$ 也会在区间结尾放一个位置，因此如果一个位置当前已经合法，则因为它后面还有一个位置在移动，它之后不会再有贡献，可以直接删去。这样所有位置移动的次数为 $O(n)$。使用set维护即可 $O(n\log n)$

再考虑原询问，相当于询问 $[a,b]$ 中有多少个 $l$ 使得询问的 $r$ 小于等于它合法的最小 $r$，并找到最大的合法 $l$。操作1可以主席树或者离线线段树求出，操作2可以二分+st表。

最后判断字典序的操作和判断lcp，lcs的操作都可以使用SA或者hash解决，因为只有 $O(n)$ 次询问这里hash不影响复杂度。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<set>
using namespace std;
#define N 200500
#define M 8005000
#define ll long long
int n;
ll as,cl=-1,cr;
char sr[N];
//hash
const int md[2]={998244853,1000000009};
struct ha{
	char s[N];
	int hv[N][2],pw[N][2];
	void init(){pw[0][0]=pw[0][1]=1;for(int t=0;t<2;t++)for(int i=1;i<=n;i++)hv[i][t]=(1ll*hv[i-1][t]*233+s[i])%md[t],pw[i][t]=1ll*pw[i-1][t]*233%md[t];}
	int calc(int l,int r,int d){return (hv[r][d]-1ll*hv[l-1][d]*pw[r-l+1][d]%md[d]+md[d])%md[d];}
	bool chk(int l1,int r1,int l2,int r2){return calc(l1,r1,0)==calc(l2,r2,0)&&calc(l1,r1,1)==calc(l2,r2,1);}
	int lcp(int x,int y)
	{
		int lb=1,rb=n-max(x,y)+1,as=0;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(chk(x,x+mid-1,y,y+mid-1))as=mid,lb=mid+1;
			else rb=mid-1;
		}
		return as;
	}
}sh,rsh;
//ST Table
struct stt{
	int vl[N],lg[N],mx[N][18];
	void init()
	{
		for(int i=2;i<=n;i++)lg[i]=lg[i>>1]+1;
		for(int i=1;i<=n;i++)mx[i][0]=vl[i];
		for(int j=1;j<=17;j++)for(int i=1;i+(1<<j)-1<=n;i++)mx[i][j]=max(mx[i][j-1],mx[i+(1<<j-1)][j-1]);
	}
	int que(int l,int r)
	{
		int tp=lg[r-l+1];
		return max(mx[l][tp],mx[r-(1<<tp)+1][tp]);
	}
}st,rst;
void init_s()
{
	for(int i=1;i<=n;i++)sh.s[i]=sr[i];sh.init();
	for(int i=1;i<=n;i++)st.vl[i]=i+sh.lcp(i,1)-1;
	st.init();
}
bool querys(int l,int r,int d)
{
	if(d<=0)return 1;
	int r1=r;
	r-=d-1;if(l>r)return 0;
	return st.que(l,r)>=r1;
}
//pretree
struct pret{
	int ch[M][2],su[M],rt[N],ct,c1;
	int ins(int x,int l,int r,int v)
	{
		int st=++ct;ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];su[st]=su[x]+1;
		if(l==r)return st;
		int mid=(l+r)>>1;
		if(mid>=v)ch[st][0]=ins(ch[x][0],l,mid,v);
		else ch[st][1]=ins(ch[x][1],mid+1,r,v);
		return st;
	}
	void ins1(int x){rt[c1+1]=ins(rt[c1],1,n,x);c1++;}
	int query(int x,int l,int r,int l1,int r1)
	{
		if(l==l1&&r==r1)return su[x];
		if(!x)return 0;
		int mid=(l+r)>>1;
		if(mid>=r1)return query(ch[x][0],l,mid,l1,r1);
		else if(mid<l1)return query(ch[x][1],mid+1,r,l1,r1);
		else return query(ch[x][0],l,mid,l1,mid)+query(ch[x][1],mid+1,r,mid+1,r1);
	}
	int query1(int l,int r,int l1,int r1){return query(rt[r],1,n,l1,r1)-query(rt[l-1],1,n,l1,r1);}
}pt;
set<int> rv,ls,fv[N];
void init_rs()
{
	for(int i=1;i<=n;i++)rsh.s[i]=sr[n-i+1];rsh.init();
	for(int i=1;i<=n;i++)ls.insert(i);
	for(int i=n;i>=1;i--)
	{
		for(set<int>::iterator it=fv[i].begin();it!=fv[i].end();it++)rv.insert(*it);
		set<int> r2;
		for(set<int>::iterator it=rv.begin();it!=rv.end();it++)
		{
			int tp=*it;
			if(rst.vl[tp])continue;
			ls.erase(tp);rst.vl[tp]=i;
			r2.insert(tp-1);
		}
		rv=r2;
		int rl=rsh.lcp(i,1);
		if(rl<i-1)fv[rl+1].insert(i+rl-1);
		else
		{
			int nw=i*2-2,rb=i+rl-1;
			while(1)
			{
				set<int>::iterator it=ls.lower_bound(nw);
				if(it==ls.end())break;
				int tp=*it;
				if(tp>rb)break;
				rst.vl[tp]=i;ls.erase(tp);
			}
			rv.insert(nw-1);
		}
	}
	rst.init();
	for(int i=1;i<=n;i++)pt.ins1(rst.vl[i]);
}
int rquery_ct(int l,int r,int d)
{
	swap(l,r);l=n-l+1;r=n-r+1;d=n-d+1;
	return pt.query1(l,r,d,n);
}
int rquery_mn(int l,int r,int d)
{
	swap(l,r);l=n-l+1;r=n-r+1;d=n-d+1;
	int lb=l,rb=r,as=-1;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(rst.que(l,mid)>=d)as=mid,rb=mid-1;
		else lb=mid+1;
	}
	return as==-1?-1:n-as+1;
}
//sam+segt merge
struct SAM{
	int ch[N*2][26],fail[N*2],len[N*2],id[N*2],ct,ls,c1;
	void init(){ct=ls=1;}
	void ins(int t)
	{
		int st=++ct,s1=ls;len[ct]=len[ls]+1;id[ct]=++c1;ls=ct;
		while(s1&&!ch[s1][t])ch[s1][t]=st,s1=fail[s1];
		if(!s1)fail[st]=1;
		else
		{
			int nt=ch[s1][t];
			if(len[nt]==len[s1]+1)fail[st]=nt;
			else
			{
				int cl=++ct;len[cl]=len[s1]+1;
				for(int i=0;i<26;i++)ch[cl][i]=ch[nt][i];
				fail[cl]=fail[nt];fail[nt]=fail[st]=cl;
				while(s1&&ch[s1][t]==nt)ch[s1][t]=cl,s1=fail[s1];
			}
		}
	}
}sam;
struct sth{int l,r,mx;}dp[M];
sth operator +(sth a,sth b)
{
	if(a.l==0)return b;if(b.l==0)return a;
	return (sth){a.l,b.r,max(max(a.mx,b.mx),b.l-a.r)};
}
int ch[M][2],rt[N*2],ct;
void ins(int x,int l,int r,int v)
{
	if(l==r){dp[x]=(sth){l,l,0};return;}
	int mid=(l+r)>>1,tp=mid<v;
	if(!ch[x][tp])ch[x][tp]=++ct;
	if(tp)ins(ch[x][1],mid+1,r,v);else ins(ch[x][0],l,mid,v);
	dp[x]=dp[ch[x][0]]+dp[ch[x][1]];
}
int merge(int x,int y)
{
	if(!x||!y)return x+y;
	ch[x][0]=merge(ch[x][0],ch[y][0]);
	ch[x][1]=merge(ch[x][1],ch[y][1]);
	dp[x]=dp[ch[x][0]]+dp[ch[x][1]];
	return x;
}
int tp[N*2];
bool cmp(int a,int b){return sam.len[a]<sam.len[b];}
void solve()
{
	sam.init();for(int i=1;i<=n;i++)sam.ins(sr[i]-'a');
	int c1=sam.ct;for(int i=1;i<=c1;i++)tp[i]=i;
	sort(tp+1,tp+c1+1,cmp);
	for(int i=1;i<=c1;i++)rt[i]=i;ct=c1;
	for(int i=c1;i>1;i--)
	{
		if(sam.id[tp[i]])ins(rt[tp[i]],1,n,sam.id[tp[i]]);
		sth fr=dp[rt[tp[i]]];
		int le=fr.l,re=fr.r,lb=re-sam.len[tp[i]]+1,rb=re-sam.len[sam.fail[tp[i]]];
		rb=min(rb,re-fr.mx+1);
		if(sam.fail[tp[i]])rt[sam.fail[tp[i]]]=merge(rt[sam.fail[tp[i]]],rt[tp[i]]);
		int l1=lb,r1=rb,as1=-1;
		while(l1<=r1)
		{
			int mid=(l1+r1)>>1;
			if(querys(mid,re,le-(re-mid)-1))as1=mid,l1=mid+1;
			else r1=mid-1;
		}
		if(as1==-1)continue;rb=as1;
		int t1=rquery_ct(lb,rb,re);
		as+=t1;
		if(t1)
		{
			int ls=rquery_mn(lb,rb,re),rs=re;
			if(cl==-1)cl=ls,cr=rs;
			else if(cr-cl<rs-ls)continue;
			else if(cr-cl>rs-ls)cl=ls,cr=rs;
			else
			{
				int lc=sh.lcp(ls,cl);
				if(sr[ls+lc]<sr[cl+lc])cl=ls,cr=rs;
			}
		}
	}
	printf("%lld\n",as);
	for(int i=cl;i<=cr;i++)printf("%c",sr[i]);
}
int main()
{
	scanf("%s",sr+1);n=strlen(sr+1);
	init_s();init_rs();solve();
}
```

##### loj2462

###### Problem

给一棵 $n$ 个点，带边权 $c_i$ 的树，每个点有重量 $w_i$ 和价值 $v_i$。

称一个非空点集合是好的，当且仅当它满足如下条件：

1. 它构成一个连通块，且连通块内所有点重量和不超过 $m$。
2. 连通块内所有点价值和是所有满足上一条件的集合的价值和的最大值。

求有多少种选出 $k$ 个好的集合的方式，使得存在一个点 $x$ 满足：

1. 对于任意一个选出的好的集合 $S$，$x\in S$。
2. 对于任意一个选出的好的集合 $S$ 以及任意的 $y\in S$，$v_y*dis(x,y)\leq Max$

答案模 $5^{23}$。

$n\leq 60,m\leq 10^4,k\leq 10^9,1\leq w_i,v_i\leq 10^9,1\leq c_i\leq 10^4$

$2s,512MB$

###### Sol

首先考虑求出最大价值，因此考虑树形dp，但可以发现dfs+合并的dp是 $O(nm^2)$ 的。

考虑dfs序上dp，枚举一个点作为连通块的根，则相当于如果选了一个点，则必须选择它的所有祖先。考虑设 $dp_{i,j}$ 表示当前考虑到dfs序为 $i$ 的点，这个点的祖先全部被选了，此时前面选的重量和为 $j$ 的最大价值和。转移形如：

1. 选了这个点，则转移到 $dp_{i+1}$。
2. 不选这个点，则这个点子树内都会被跳过，设这个子树的dfs序最后一个点为 $r_i$，则转移到 $dp_{r_i+1}$。

显然这样每个包含根的连通块都会正好被算一次。且复杂度为 $O(nm)$。因此可以枚举根各算一次，得到最大价值。

然后考虑计数，对于一种选择集合的方案，如果只考虑第一条限制，则合法的 $x$ 为所有连通块的交，这显然是一个连通块。

再考虑第二条限制，可以发现对于一个 $y$，由于 $v_y$ 为正，合法的 $x$ 满足距离不超过某个值，因为边权为正，因此这部分也是一个连通块。因此最后合法的 $x$ 显然构成一个连通块。

因此可以考虑点减边容斥，对于每个点算出以选择 $x$ 的方案数，再对于每条边算出选择这条边的两个端点都合法的方案数，相减就是答案。

对于选择一个点且合法的方案数，只需要选择的每个连通块都包含 $x$ 且权值最大，且用到的点都合法。考虑以这个点为根做树形dp，只用以 $x$ 为根合法的点，算出包含 $x$ 的好的连通块数量。则方案数为 $C_{ct}^k$。

对于选择一条边两个点的方案数，可以看成将两个点一起看作根，做树形dp，因此这里的方案数形式与上面相同。

因此最后的问题在于，求 $n$ 次组合数 $C_a^b\bmod 5^{23}$，其中 $a,b\leq 10^{18}$。

考虑exlucas，即将 $n!$ 表示成 $5^p*q$，其中 $5$ 不整除 $q$，对于三个阶乘各自算一遍即可。

对于计算 $n!\bmod p^q$ 的情况，可以表示为：
$$
n!=\prod_{1\leq i\leq n,p\mid i}i*\prod_{1\leq i\leq n,p\nmid i}i\\
=\lfloor\frac np\rfloor!*p^{\lfloor\frac np\rfloor}*\prod_{1\leq i\leq n,p\nmid i}i
$$
正常的做法是对于左侧递归，右侧暴力算，复杂度为 $O(p^q\log^2 n)$，但在本题中不行。注意到复杂度在于右侧，考虑优化右侧。

考虑将右侧最后一个 $p$ 周期的暴力乘，考虑左侧的，相当于：
$$
\prod_{i=0}^k(pi+1)...(pi+(p-1))\bmod p^q
$$
注意到左侧是 $p$ 的倍数，因此如果左侧乘了 $q$ 项或以上，则 $\bmod p^q$ 后为 $0$。

因此考虑看成多项式，则只用维护前 $q$ 项。

令 $F_k(x)=\prod_{i=0}^{k-1}(x+pi+1)...(x+pi+(p-1))$，这里钦定求值时 $x$ 只会代入 $p$ 的倍数，则只需要维护多项式 $\bmod x^q$ 的结果，求值后一定等价。

考虑倍增，显然有 $F_{2k}(x)=F_k(x)*F_k(x+kp)$，右侧求出 $F_k(x)$ 后可以 $O(q^2)$ 计算。因此可以 $O(q^2\log k)$ 计算 $F_k(x)$。最后带入 $x=0$ 即可得到需要的值。这样的exlucas复杂度为 $O(q^2\log^2k+pq\log k)$，预处理后可以做到 $O(pq+q^2\log^2 k)$，本题中没有必要。

总复杂度 $O(n^2m+n^3\log^2 p)$，因为常数原因可以通过。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 63
#define M 10050
#define ll __int128
#define mod 11920928955078125
int n,m,k,w[N],v[N],a,b,c,head[N],cnt;
struct edge{int t,v,next;}ed[N*2];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,v,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,v,head[t]};head[t]=cnt;
}
ll li,ds[N],mx,as;
int fg[N][N],is[N],lb[N],rb[N],tid[N],ct;
void dfs(int u,int fa)
{
	lb[u]=++ct;tid[ct]=u;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)ds[ed[i].t]=ds[u]+ed[i].v,dfs(ed[i].t,u);
	rb[u]=ct;
}
struct sth{ll su,ct;}dp[N][M];
sth operator +(sth a,sth b)
{
	if(a.su<b.su)return b;if(a.su>b.su)return a;
	return (sth){a.su,a.ct+b.ct};
}
sth solve(int u,int t)
{
	ct=0;
	for(int i=0;i<=n;i++)for(int j=0;j<=m;j++)dp[i][j]=(sth){-10000000000000000,0};
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=t)dfs(ed[i].t,u);
	if(t)for(int i=head[t];i;i=ed[i].next)if(ed[i].t!=u)dfs(ed[i].t,t);
	int s1=w[u]+w[t],s2=v[u]+v[t];
	if(s1>m)return (sth){-10000000000000000,0};
	dp[1][s1]=(sth){s2,1};
	for(int i=1;i<=ct;i++)
	for(int j=0;j<=m;j++)
	{
		dp[rb[tid[i]]+1][j]=dp[rb[tid[i]]+1][j]+dp[i][j];
		if(j+w[tid[i]]<=m&&is[tid[i]])dp[i+1][j+w[tid[i]]]=dp[i+1][j+w[tid[i]]]+(sth){dp[i][j].su+v[tid[i]],dp[i][j].ct};
	}
	for(int j=1;j<=m;j++)dp[ct+1][j]=dp[ct+1][j]+dp[ct+1][j-1];
	return dp[ct+1][m];
}
struct st1{ll v[24];st1(){for(int i=0;i<23;i++)v[i]=0;}}r1;
st1 operator *(st1 a,st1 b)
{
	st1 as;
	for(int i=0;i<23;i++)for(int j=0;i+j<23;j++)as.v[i+j]=(as.v[i+j]+a.v[i]*b.v[j])%mod;
	return as;
}
st1 doit(st1 a,ll v)
{
	for(int i=22;i>0;i--)
	for(int j=i;j<23;j++)
	a.v[j-1]=(a.v[j-1]+v*a.v[j])%mod;
	return a;
}
st1 solve(ll n)
{
	if(n==0){st1 as;as.v[0]=1;return as;}
	if(n==1)return r1;
	st1 t1=solve(n/2);
	t1=t1*doit(t1,n/2*5);
	if(n&1)t1=t1*doit(r1,(n-1)*5);
	return t1;
}
struct st2{ll v,ct;};
st2 operator +(st2 a,st2 b){return (st2){a.v*b.v%mod,a.ct+b.ct};}
st2 solve1(ll n)
{
	if(!n)return (st2){1,0};
	st2 ls=solve1(n/5);
	ll rv=solve(n/5).v[0];
	for(ll i=n/5*5+1;i<=n;i++)rv=rv*i%mod;
	return ls+(st2){rv,n/5};
}
ll getinv(ll x)
{
	ll as=1;
	for(ll i=1;i<mod;i*=5)while(as*x%(i*5)!=1)as+=i;
	return as;
}
ll calc(ll n,ll m)
{
	if(n<m)return 0;
	st2 r1=solve1(n),r2=solve1(m)+solve1(n-m);
	ll vl=r1.v*getinv(r2.v)%mod,sv=r1.ct-r2.ct;
	if(sv>23)return 0;
	while(sv--)vl=vl*5%mod;
	return vl;
}
int main()
{
	scanf("%d%d%d%lld",&n,&m,&k,&li);
	for(int i=1;i<=n;i++)scanf("%d",&w[i]);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
	r1.v[0]=1;for(int i=1;i<5;i++)for(int j=5;j>=0;j--)r1.v[j+1]+=r1.v[j],r1.v[j]*=i;
	for(int i=1;i<=n;i++)
	{
		ds[i]=0;ct=0;dfs(i,0);
		for(int j=1;j<=n;j++)if(ds[j]*v[j]<=li)fg[i][j]=1;
	}
	mx=0;
	for(int i=1;i<=n;i++)is[i]=1;
	for(int j=1;j<=n;j++)
	{
		sth tp=solve(j,0);
		mx=max(mx,tp.su);
	}
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)is[j]=fg[i][j];
		sth tp=solve(i,0);
		if(tp.su==mx)as=(as+calc(tp.ct,k))%mod;
	}
	for(int i=1;i<=n;i++)for(int t=head[i];t;t=ed[t].next)if(ed[t].t>i)
	{
		int u=i,v=ed[t].t;
		for(int j=1;j<=n;j++)is[j]=fg[u][j]&fg[v][j];
		if(!fg[u][v]||!fg[v][u])continue;
		sth tp=solve(u,v);
		if(tp.su==mx)as=(as+mod-calc(tp.ct,k))%mod;
	}
	printf("%lld\n",(long long)as);
}
```

##### IncreasingSequence

###### Problem

给一个 $n$ 位数字串，保证开头不是 $0$。

你可以将它划分成若干段，满足如下条件：

1. 如果将每一段看成一个数字，则得到的数字严格递增。
2. 在此基础上，最后一个数字尽量小。
3. 在此基础上，第一个数字尽量大，然后第二个数字尽量大，以此类推。

此时最优方案唯一，求出得到的方案中所有数的乘积，模 $10^9+3$

$n\leq 2500$

$2s,256MB$

###### Sol

考虑 $dp$，为了确定最后一个数字的最小值，需要从前向后 $dp$。

如果当前划分了一段前缀，则可以发现最后一个数字越小越优秀。设 $sr_i$ 表示当前划分了 $[1,i]$，最后一段的最小长度。则向右转移时，需要下一段长度大于某个数。

如果不考虑 $0$，则只需要判断下一段长度与这一段相等时哪个更大，不等的情况可以直接判断长度，考虑 $0$ 时只需要对于每个位置预处理左右第一个非零元素即可。

则转移相当于后缀取max，因此可以单调队列优化，也可以类似后缀和的方法解决，只要能 $O(1)$ 判断，即可 $O(n)$ 求出dp。

这样可以确定最后一个数字的最小值，考虑从后往前dp以得到字典序最大的方案。此时可以加入限制最后一个数只能取最小值。

设 $sl_i$ 表示划分了 $[i,n]$，最后一段的最大长度。则转移也是转移到一段后缀，判断的情况和上一种类似。此时转移也是后缀取max，但因为是倒着做，因此只能单调队列优化转移。

最后使用 $sl$ 容易还原字典序最大的方案。

如果使用SA+ST表预处理，则预处理复杂度 $O(n\log n)$，dp复杂度 $O(n)$。

如果使用SA-IS+四毛子或者SAM上进行处理，则可以做到 $O(n)$。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<string>
using namespace std;
#define N 2550
#define mod 1000000003
int n;
char s[N];
struct SA{
    int n,m,v[N],b[N],sa[N],rk[N],he[N],su[N],mn[N][14],lg[N];
    void init()
    {
        for(int i=1;i<=n;i++)v[i]=s[i];m=255;
        for(int i=1;i<=m;i++)su[i]=0;
        for(int i=1;i<=n;i++)su[v[i]]++;
        for(int i=1;i<=m;i++)su[i]+=su[i-1];
        for(int i=n;i>=1;i--)sa[su[v[i]]--]=i;
        for(int l=1;l<=n;l<<=1)
        {
            int ct=0;
            for(int i=n-l+1;i<=n;i++)b[++ct]=i;
            for(int i=1;i<=n;i++)if(sa[i]>l)b[++ct]=sa[i]-l;
            for(int i=1;i<=m;i++)su[i]=0;
            for(int i=1;i<=n;i++)su[v[i]]++;
            for(int i=1;i<=m;i++)su[i]+=su[i-1];
            for(int i=n;i>=1;i--)sa[su[v[b[i]]]--]=b[i];
            for(int i=1;i<=n;i++)b[i]=v[i];
            v[sa[1]]=1;
            for(int i=2;i<=n;i++)if(b[sa[i]]==b[sa[i-1]]&&b[sa[i]+l]==b[sa[i-1]+l])v[sa[i]]=v[sa[i-1]];
            else v[sa[i]]=v[sa[i-1]]+1;
            m=v[sa[n]];
        }
        for(int i=1;i<=n;i++)rk[sa[i]]=i;
        int l=0;
        for(int i=1;i<=n;i++)
        {
            if(l)l--;if(rk[i]==1)continue;
            while(s[i+l]==s[sa[rk[i]-1]+l])l++;
            he[rk[i]]=l;
        }
        for(int i=2;i<=n;i++)mn[i][0]=he[i],lg[i]=lg[i>>1]+1;
        for(int j=1;j<=13;j++)for(int i=2;i+(1<<j)-1<=n;i++)
        mn[i][j]=min(mn[i][j-1],mn[i+(1<<j-1)][j-1]);
    }
    int lcp(int i,int j)
    {
        if(i==j)return n-i+1;
        int l=rk[i],r=rk[j];
        if(l>r)swap(l,r);l++;
        int tp=lg[r-l+1];
        return min(mn[l][tp],mn[r-(1<<tp)+1][tp]);
    }
}sa;
int dp[N],ls[N],rs[N],st[N],lb,rb,l1[N];
bool cmp(int l1,int r1,int l2,int r2)
{
	if(rs[l2]>r2)return 0;if(rs[l1]>r1)return 1;
	l1=rs[l1];l2=rs[l2];
	if(r1-l1!=r2-l2)return r1-l1<r2-l2;
	int le=sa.lcp(l1,l2);
	if(l1+le-1>=r1)return 0;
	return s[l1+le]<s[l2+le];
}
void solve()
{
	lb=1,rb=0;
	for(int i=n;i>=1;i--)
	{
		while(lb<=rb&&l1[st[lb]]>i)lb++;
		if(lb<=rb)dp[i]=max(dp[i],st[lb]-1);
		if(rs[i]==n+1)continue;
		int le=dp[i]-rs[i],l2=max(i-1-le,1);
		if(cmp(l2,i-1,rs[i],dp[i]))l2=ls[l2-1]+1;
		else l2++;
		l1[i]=l2;
		if(lb<=rb&&l1[st[rb]]<=l1[i])continue;
		st[++rb]=i;
	}
}
void solve2()
{
	for(int i=1;i<=n;i++)dp[i]=1;
	for(int i=1;i<=n;i++)
	{
		dp[i]=max(dp[i],dp[i-1]);
		int le=i-rs[dp[i]],l2=rs[i+1]+le;
		if(l2>n)continue;
		if(!cmp(dp[i],i,i+1,l2))l2++;
		dp[l2]=max(dp[l2],i+1);
	}
}
struct IncreasingSequence{
	int getProduct(vector<string> s1)
	{
		for(int i=0;i<s1.size();i++)
		for(int j=0;j<s1[i].size();j++)
		s[++n]=s1[i][j];
		sa.n=n;sa.init();
		for(int i=1;i<=n;i++)ls[i]=s[i]-'0'?i:ls[i-1];
		rs[n+1]=n+1;for(int i=n;i>=1;i--)rs[i]=s[i]-'0'?i:rs[i+1];
		int as=1;
		solve2();
		for(int i=1;i<n;i++)if(cmp(dp[i],i,i+1,n))as=i+1;
		for(int i=1;i<=n;i++)dp[i]=cmp(as,n,i,n)?0:n;
		solve();
		int nw=1,ls=-1,as1=1;
		while(nw<=n)
		{
			int fg=-1,su=0;
			for(int i=nw;i<=n;i++)if((ls==-1||cmp(ls,nw-1,nw,i))&&(i<n||!cmp(as,n,nw,i))&&(i==n||cmp(nw,i,i+1,dp[i+1])))fg=i;
			for(int i=nw;i<=fg;i++)su=(10ll*su+s[i]-'0')%mod;
			as1=1ll*as1*su%mod;
			nw=fg+1;
		}
		return as1;
	}
};
```
##### Permutation and Minimum

###### Problem

有一个长度为 $2n$ 的排列 $p$，现在 $p$ 的一些位置已经确定。

你可以任意决定剩下的位置，令 $b_i=\min(p_{2i-1},p_{2i})$，求可能的序列 $b$ 数量，模 $998244353$。

$n\leq 300$

$2s,1024MB$

###### Sol

将排列中 $p_{2i-1},p_{2i}$ 看成一对，则对于一对有如下情况：

1. 一对中两个元素都已经确定，这种情况可以直接不考虑这一对。
2. 一对中有一个元素确定。这样的所有对之间有顺序。
3. 一对中两个元素都不确定。这样的对之间也有位置关系，但可以发现交换两对也能得到合法方案。因此可以看成这些对是无序的情况，最后再乘以这样的对数的阶乘。

此时从大到小考虑将每个元素，则有如下情况：

如果这个元素是需要被填进去的，则有如下情况：

1. 它作为某一对中较大的数，等待之后的数和它组成一对。
2. 它作为某一对中较小的数和之前的组成一对，此时可以和之前有一个确定的组成一对，也可以和上一种情况中留下的组成一对（此时会有选择一个的系数）。

如果这个元素是第二种情况中固定的，则它可以作为较大的一个，也可以作为较小的一个和之前留下来的一个没有被确定的元素组成一对（看成将之前那个元素填到这里）。

因此可以设 $dp_{i,j,k}$ 表示考虑了最大的 $i$ 个数，当前前面有 $j$ 个不是固定的元素以及 $k$ 个第二种情况中固定的元素需要和后面的元素配对，此时的方案数。

由上面容易得到对于一个没有被固定的元素，转移为 $dp_{i-1,j,k}\to dp_{i,j+1,k},dp_{i,j-1,k}$，且 $k*dp_{i-1,j,k}\to dp_{i,j,k-1}$。对于一个固定的元素，转移为 $dp_{i-1,j,k}\to dp_{i,j,k+1},dp_{i,j-1,k}$。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 305
#define mod 1000000007
int n,p[N*2],fg[N*2],dp[N][N],dp2[N][N],cl;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n*2;i++)scanf("%d",&p[i]);
	for(int i=1;i<=n;i++)
	{
		if(p[i*2-1]>p[i*2])swap(p[i*2-1],p[i*2]);
		if(p[i*2-1]==-1&&p[i*2]==-1)cl++;
		else if(p[i*2-1]==-1)fg[p[i*2]]=1;
		else fg[p[i*2-1]]=fg[p[i*2]]=2;
	}
	dp[0][0]=1;
	for(int i=n*2;i>=1;i--)if(fg[i]==2)continue;
	else if(fg[i]==1)
	{
		for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)if(dp[i][j])
		{
			dp2[i][j+1]=(dp2[i][j+1]+dp[i][j])%mod;
			if(i)dp2[i-1][j]=(dp2[i-1][j]+dp[i][j])%mod;
		}
		for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)dp[i][j]=dp2[i][j],dp2[i][j]=0;
	}
	else
	{
		for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)if(dp[i][j])
		{
			dp2[i+1][j]=(dp2[i+1][j]+dp[i][j])%mod;
			if(i)dp2[i-1][j]=(dp2[i-1][j]+dp[i][j])%mod;
			if(j)dp2[i][j-1]=(dp2[i][j-1]+1ll*dp[i][j]*j)%mod;
		}
		for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)dp[i][j]=dp2[i][j],dp2[i][j]=0;
	}
	int as=dp[0][0];
	for(int i=1;i<=cl;i++)as=1ll*as*i%mod;
	printf("%d\n",as);
}
```

##### Hopes of Rowing

###### Problem

有一个 $n$ 个点 $m$ 条边的图。有一个正整数 $k$，你需要给每个点分配一个非负实数权值，使得任意一条边两个端点的权值和大于等于 $k$ 且在此基础上所有点的权值和最小。求方案。

$n\leq 500,m\leq\frac{n(n-1)}2$

$1s,64MB$

###### Sol

可以得到如下结论：存在一组最优解使得每个点点权都是 $0,\frac k2,k$ 中的一种。

证明：考虑对最优解调整，选择一个点权不是这些值的点，从这个点开始，只保留所有边两个端点和等于 $k$ 的边，考虑这个点的连通块。如果连通块中包含奇环，则不难发现连通块内点权全部为 $\frac k2$，矛盾。否则连通块为二分图，可以黑白染色。选择点数较小的一侧增加，另外一侧对应减少，一直调整到出现新的与连通块连接的两侧和为 $k$ 的边，然后对新的连通块继续调整，调整的结果一定为出现奇环 ( $\frac k2$ ) 或者调整到 $0,1$。因此结论成立。

此时可以看成每个点只能选整数权值，使得每条边两侧权值和大于等于 $2$。

如果图是二分图，则问题相当于二分图最小点覆盖（且此时一定只会选 $0,2$）。但一般图的最小点覆盖难以解决，因此考虑转化为二分图的形式。

考虑将一个点拆成两个点 $s_i,t_i$，分别在二分图的两侧，选择一个点代表让这个点的权值加一。对于一条边 $(i,j)$，考虑连边 $(s_i,t_j),(s_j,t_i)$。则此时 $i,j$ 对应的四个点中选择不超过一个一定不合法，选择三个一定合法，但选择两个的情况中 $(s_i,t_j),(s_j,t_i)$ 两种会变得不合法。

但可以发现对于一种原问题中合法的方案，可以让所有权值为 $1$ 的点都选 $s_i$，这样在这个问题中一定合法。从而这个问题的一组最优解可以对应原问题的一组最优解，原问题的一组最优解也可以对应这个问题的一组最优解。因而求出这个问题的一组最优解即可。

根据一些性质，二分图最小点覆盖等于最大匹配。考虑如下构造：

如果当前所有点都在最大匹配中，则任选一侧所有点即可。

否则，找一个不在最大匹配中的点，则所有与这个点相邻的点都在最大匹配中，认为这些点都在最小点覆盖中，接下来删去这个点以及上一步选中的点。

在删去一个点时，要么这个点被选进了最小点覆盖，要么与之相邻的点都被选进了最小点覆盖。因此这样的方案一定合法。同时删去一个选中的点时，如果最大匹配不减小，则考虑此时的增广路，再加上第一步选的点可以使得这个增广路能让最大匹配变大，因此不存在这种情况。因而每次选一个点都会使得最大匹配减小 $1$，因此这样的方案是最优解。从而找一组最大匹配即可构造方案。

复杂度 $O(n^{2.5})$

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 1050
int n,k,m,head[N],cur[N],dis[N],cnt=1,a,b,vis[N],ls[N],is[N];
struct edge{int t,next,v;}ed[N*410];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;
}
bool bfs(int s,int t)
{
	for(int i=1;i<=n*2+2;i++)cur[i]=head[i],dis[i]=-1;
	queue<int> qu;qu.push(s);dis[s]=0;
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
	for(int &i=cur[u];i;i=ed[i].next)
	if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(!f)return as;
	}
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&k,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b+n,1),adde(b,a+n,1);
	for(int i=1;i<=n;i++)adde(n*2+1,i,1),adde(i+n,n*2+2,1);
	while(bfs(n*2+1,n*2+2))dfs(n*2+1,n*2+2,n);
	for(int i=1;i<=n;i++)for(int j=head[i];j;j=ed[j].next)if(!ed[j].v&&ed[j].t<=n*2)ls[i]=ed[j].t;
	for(int i=1;i<=n;i++)for(int j=head[i+n];j;j=ed[j].next)if(ed[j].v&&ed[j].t<=n*2)ls[i+n]=ed[j].t;
	for(int i=1;i<=n*2;i++)is[i]=-1;
	queue<int> tp;
	for(int i=1;i<=n*2;i++)if(!ls[i])is[i]=0,tp.push(i);
	while(!tp.empty())
	{
		int u=tp.front();tp.pop();
		if(is[u]){if(is[ls[u]]==-1)is[ls[u]]=0,tp.push(ls[u]);}
		else for(int i=head[u];i;i=ed[i].next)if(ed[i].t<=n*2&&is[ed[i].t]==-1)is[ed[i].t]=1,tp.push(ed[i].t);
	}
	for(int i=1;i<=n*2;i++)if(is[i]==-1)is[i]=(i<=n);
	for(int i=1;i<=n;i++)printf("%.1lf\n",0.5*k*(is[i]+is[i+n]));
}
```

##### ConvenientBlock

###### Problem

给一个 $n$ 个点 $m$ 条边的有向图，边有边权。你需要找到一个边的集合，使得任意一条从 $1$ 到 $n$ 的路径（不一定简单）正好经过一条集合中的边。在此基础上最小化集合中边的边权和。输出最小边权和或输出无解。

$n\leq 100,m\leq 2500$

$2s,256MB$

###### Sol

为了简便，可以先删去不能被 $1$ 到达或者不能到达 $n$ 的点。

考虑从 $1$ 出发，不经过集合中边能到达的点集 $S$，则 $S$ 满足如下性质：

1. $1\in S,n\not\in S$
2. 如果 $u\in S$，且在有向图中 $v$ 能到达 $u$，则 $v\in S$。

第一个性质显然，对于第二个性质，如果 $v\not\in S$，则 $1$ 到 $v$ 的路径上至少有一条选中的边，但 $v$ 可以到达 $u$，而 $u$ 到 $n$ 的任意一条路径上都需要有一条选中的边，因此得到矛盾。

同时，如果 $S$ 满足这些性质，则选中所有满足 $s\in S,t\not\in S$ 的边 $s\to t$，则可以发现这样满足要求。可以发现此时的问题类似于一个最大权闭合子图的形式，即不选 $1$ 有 $+\infty$ 的代价，选 $n$ 有 $+\infty$ 的代价，对于一条边 $u\to v$，选 $v$ 必须选 $u$，选 $u$ 不选 $v$ 有一个代价。而这相当于将所有反向边的流量设为 $+\infty$ 后求原图的最小割。因此直接网络流即可。如果流为 $+\infty$ 则无解。

复杂度 $O(n^2m)$

这也说明对于一个最小割问题，如果加入反向且边权为 $+\infty$ 的边，则问题变为每条路径上最多割一条边的问题。这样可以在用最小割处理一类每个变量有若干种取值，这些取值被看成一条链，割一条边代表选这个值的问题时，可以直接避免一条链割多条边的问题。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<queue>
using namespace std;
#define N 105
#define ll long long
int n,m,s[N*27][3];
struct maxflow{
	struct edge{int t,next;ll v;}ed[N*150];
	int head[N],cnt,n,cur[N],dis[N];
	void init(int m){n=m;cnt=1;}
	void adde(int f,int t,ll v)
	{
		ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
		ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;
	}
	bool bfs(int s,int t)
	{
		for(int i=1;i<=n;i++)dis[i]=-1,cur[i]=head[i];
		queue<int> qu;qu.push(s);dis[s]=0;
		while(!qu.empty())
		{
			int x=qu.front();qu.pop();
			for(int i=head[x];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]==-1)
			{
				dis[ed[i].t]=dis[x]+1;qu.push(ed[i].t);
				if(ed[i].t==t)return 1;
			}
		}
		return 0;
	}
	ll dfs(int u,int t,ll f)
	{
		if(u==t||!f)return f;
		ll as=0,tp;
		for(int &i=cur[u];i;i=ed[i].next)
		if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
		{
			as+=tp;f-=tp;
			ed[i].v-=tp;ed[i^1].v+=tp;
			if(!f)return as;
		}
		return as;
	}
	ll dinic(int s,int t)
	{
		ll as=0;
		while(bfs(s,t)&&as<1e18)as+=dfs(s,t,1e18);
		return as;
	}
}mf;
ll vl[N],is[N],is1[N];
void dfs2(int u)
{
	is[u]=1;
	for(int i=1;i<=m;i++)if(s[i][1]==u&&!is[s[i][0]])dfs2(s[i][0]);
}
void dfs3(int u)
{
	is1[u]=1;
	for(int i=1;i<=m;i++)if(s[i][0]==u&&!is1[s[i][1]])dfs3(s[i][1]);
}
struct ConvenientBlock{
	ll minCost(int _n,vector<int> sa,vector<int> sb,vector<int> sc1)
	{
		n=_n,m=sa.size();
		for(int i=1;i<=m;i++)s[i][0]=sa[i-1]+1,s[i][1]=sb[i-1]+1,s[i][2]=sc1[i-1];
		mf.init(n);dfs2(n);dfs3(1);
		for(int i=1;i<=n;i++)is[i]&=is1[i];
		for(int i=1;i<=m;i++)if(is[s[i][0]]&&is[s[i][1]])mf.adde(s[i][0],s[i][1],s[i][2]),mf.adde(s[i][1],s[i][0],1e14);
		ll as=mf.dinic(1,n);
		return as>9e13?-1:as;
	}
};
```

##### DiscountedShortestPaths

###### Problem

有一个 $n$ 个点 $m$ 条边的无向图，每条边有一个代价，每次经过这条边都需要支付代价。

你有 $d$ 张优惠劵，第 $i$ 张的权值为 $d_i$。在经过一条边时，你可以使用一张优惠劵，设原本边的代价为 $c$，则使用后需要的代价为 $|max(c-d_i,0)|$。每张优惠劵只能被使用一次。

对于每一对 $u,v$，求出 $u$ 到 $v$，可以使用所有优惠劵的最小代价，输出它们的和。

$n\leq 20$

$3s,36MB$

###### Sol

暴力做法：

一条 $s$ 到 $t$ 的路径需要满足选择的边连通， $s,t$ 度数为奇数且剩余点度数都为偶数。且最优解中最多使用 $n-1$ 条边。

但可以发现如果去掉连通的限制，多考虑的部分相当于在路径外多出了一些环，这一定不优秀。又可以注意到如果按照边权从大到小考虑每条边，则每条边使用的优惠劵一定也是从大到小，因此可以按照边权从大到小dp。

此时可以设 $dp_{i,S}$ 表示考虑了边权前 $i$ 大的边，当前度数为奇数的点集合为 $S$ ，前面的最小代价。事实上转移时不需要要求选择的边权不增，因为这样一定不优秀。最后对于 $s,t$，答案即为 $\min_i dp_{i,\{s,t\}}$。

复杂度 $O(n^32^n)$，滚动后空间复杂度 $O(2^n)$，但常数非常小，注意一些细节可以正好卡进去。

优秀做法：

可以发现如下结论：

存在 $s$ 到 $t$ 的一条最优路径，使得这条路径是路径点集的最小生成树。

如果结论不成立，则存在一条边，使得这条边的边权小于路径上这一段内某条边的边权。此时将这一段改为走这条边，因为这一段内边权最大的边权值大于新的边，因此将两种路径上的边权从大到小排序后，对于每一个 $i$，新路径上的第 $i$ 大边权都小于等于原路径的第 $i$ 大边权。因此新路径不会更差。

因此，可以枚举每个点集求最小生成树，考虑选择的最小生成树中所有边各一次的答案，用这个答案更新点集内任意两个点的最小距离。

复杂度 $O(n^22^n)$，空间复杂度 $O(n^2)$

###### Code

做法一：

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 210
#define ll long long
int n,m,k,s[N][3],tp[N],v[N],v1[N];
ll dp[525001],rs[525001],su;
vector<int> cr[N];
bool cmp(int a,int b){return a>b;}
struct DiscountedShortestPaths{
	ll minimumCostSum(int n,vector<int> sa,vector<int> sb,vector<int> sc,vector<int> di)
	{
		m=sa.size();k=di.size();
		for(int i=1;i<=m;i++)s[i][0]=sa[i-1],s[i][1]=sb[i-1],s[i][2]=sc[i-1];
		for(int i=0;i<1<<n-1;i++)dp[i]=rs[i]=1e16;
		for(int i=1;i<=m;i++)tp[i]=((1<<s[i][0])|(1<<s[i][1]))&((1<<n-1)-1);
		for(int i=0;i<1<<n-1;i++)
		{
			int ct=0;
			for(int j=1;j<n;j++)if((i>>j-1)&1)ct++;
			cr[ct].push_back(i);
		}
		dp[0]=0;
		for(int i=1;i<=k;i++)v[i]=di[i-1];
		sort(v+1,v+k+1,cmp);
		for(int i=1;i<n;i++)
		{
			for(int l=1;l<=m;l++)v1[l]=max(s[l][2]-v[i],0);
			int s1=min(i-1,n-i+1)*2;
			for(int l=1;l<=m;l++)for(int j=0;j<(1<<n-1);j++)
			rs[j^tp[l]]=min(rs[j^tp[l]],dp[j]+v1[l]);
			s1=min(i,n-i)*2;
			for(int j=0;j<=s1;j++)
			for(int t=0;t<cr[j].size();t++)
			{
				int st=cr[j][t];
				dp[st]=min(dp[st],rs[st]),rs[st]=1e16;
			}
		}
		for(int j=1;j<=n;j++)for(int k=j+1;k<=n;k++)su+=dp[((1<<j-1)|(1<<k-1))&((1<<n-1)-1)];
		return su;
	}
};
```

做法二：

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 21
#define ll long long
int n,m,k,s[N*N][3],id[N*N],fa[N],is[N],v[N];
ll as[N][N],su;
bool cmp(int a,int b){return a>b;}
bool cmp2(int a,int b){return s[a][2]<s[b][2];}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct DiscountedShortestPaths{
	ll minimumCostSum(int n,vector<int> sa,vector<int> sb,vector<int> sc,vector<int> di)
	{
		m=sa.size();k=di.size();
		for(int i=1;i<=m;i++)s[i][0]=sa[i-1]+1,s[i][1]=sb[i-1]+1,s[i][2]=sc[i-1],id[i]=i;
		for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)as[i][j]=1e18;
		for(int i=1;i<=k;i++)v[i]=di[i-1];
		sort(v+1,v+k+1,cmp);sort(id+1,id+m+1,cmp2);
		for(int d=1;d<1<<n;d++)
		{
			int ct=0,c1=0;
			ll su=0;
			for(int i=1;i<=n;i++)fa[i]=i,is[i]=(d>>i-1)&1,ct+=is[i];
			for(int i=1;i<=m;i++)
			{
				int f=s[id[i]][0],t=s[id[i]][1],v1=s[id[i]][2];
				if(!is[f]||!is[t]||finds(f)==finds(t))continue;
				fa[finds(f)]=finds(t);c1++;su+=max(v1-v[ct-c1],0);
				if(c1==ct-1)break;
			}
			if(c1<ct-1)continue;
			for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(is[i]&&is[j])as[i][j]=min(as[i][j],su);
		}
		for(int j=1;j<=n;j++)for(int k=j+1;k<=n;k++)su+=as[j][k];
		return su;
	}
};
```
##### MagicalGirlLevelThreeDivOne

###### Problem

给出 $k$ 个 $01$ 串 $s_0,...,s_{k-1}$，对于 $n\geq k$，定义 $s_n=s_{n-1}+s_{n-k-1}+s_{n-2k-1}+\cdots+s_{n-ik+1}+\cdots$。

给出 $n,l,r$，询问 $s_n$ 中 $[l,r]$ 的子串内最长的连续 $1$ 段长度。

$k,|s_i|\leq 50,n\leq 10^9,l,r\leq 10^{15}$

$2s,64MB$

###### Sol

可以发现 $s_{n-1}$ 是 $s_n$ 的前缀，因此如果 $s_i(i<k)$ 长度超过了 $r$，就可以看成询问 $s_i$。

同时可以发现 $|s_{n+k}|=|s_0|+\cdots+|s_n|$，因此只需要 $O(k(\log r-\log k))$ 个串就可以让长度大于 $r$。极限情况下这个数量不超过 $624$。

因此可以处理出每一个串的长度。同时注意到求区间最长1段可以通过维护前缀1，后缀1，段内部最长1以及是否存在0做到 $O(1)$ 合并，因此可以对于每个串求出这些信息。这部分暴力做的复杂度为 $O(k(\log r-\log k)^2)$

接着考虑询问，对于询问 $s_n$ 上的区间考虑 $s_n$ 的组成，递归到组成 $s_n$ 的每一个串上直接计算。一个区间询问只会递归到一个前缀，一个后缀和若干个整体区间，而整体区间部分之前已经求出。因此可以发现递归次数只有 $O(k(\log r-\log k))$ 次。因此总复杂度 $O(k(\log r-\log k)^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
#define ll long long
int n,k;
ll le[655],l,r;
char s[52][52];
vector<int> rs[655];
struct sth{ll l,r,k,ls;}s0,s1,dp[655];
sth operator +(sth a,sth b)
{
	if(a.ls&&b.ls)return (sth){a.l+b.l,a.r+b.r,a.k+b.k,1};
	else if(a.ls)return (sth){a.l+b.l,b.r,max(a.l+b.l,b.k),0};
	else if(b.ls)return (sth){a.l,a.r+b.r,max(a.r+b.r,a.k),0};
	else return (sth){a.l,b.r,max(a.r+b.l,max(a.k,b.k)),0};
}
sth solve(int d,ll l,ll r)
{
	if(l==1&&r==le[d])return dp[d];
	if(d<=n)
	{
		sth as=(sth){0,0,0,1};
		for(int i=l;i<=r;i++)as=as+(s[d][i]=='1'?s1:s0);
		return as;
	}
	sth as=(sth){0,0,0,1};
	for(int i=0;i<rs[d].size();i++)
	{
		ll l1=le[rs[d][i]];
		if(l>l1)l-=l1,r-=l1;
		else if(r>0)as=as+solve(rs[d][i],l,min(r,l1)),l=1,r-=l1;
	}
	return as;
}
struct MagicalGirlLevelThreeDivOne{
	ll theMaxPower(vector<string> s2,int k,ll l,ll r)
	{
		n=s2.size();
		for(int i=1;i<=n;i++)
		{
			le[i]=s2[i-1].size();
			for(int j=1;j<=le[i];j++)s[i][j]=s2[i-1][j-1];
		}
		l++;r++;k++;
		for(int i=n+1;i<=k;i++)
		{
			rs[i].push_back(i-1);
			while(rs[i].back()>n)rs[i].push_back(rs[i].back()-n);
			for(int j=0;j<rs[i].size();j++)le[i]+=le[rs[i][j]];
			if(le[i]>=r){k=i;break;}
		}
		s1=(sth){1,1,1,1};
		for(int i=1;i<=n;i++)
		{
			dp[i]=s[i][1]=='1'?s1:s0;
			for(int j=2;j<=le[i];j++)dp[i]=dp[i]+(s[i][j]=='1'?s1:s0);
		}
		for(int i=n+1;i<=k;i++)
		{
			dp[i]=dp[i-1];
			for(int j=1;j<rs[i].size();j++)dp[i]=dp[i]+dp[rs[i][j]];
		}
		return solve(k,l,r).k;
	}
};
```

##### BlackBoxDiv1

###### Problem

一个矩形被划分为 $n\times m$ 个正方形格子，现在每个格子上都有一面镜子，镜子的摆放方式一定是某条格子的对角线。 

从矩形的 $2(n+m)$ 个边界开始向矩形内部发射激光，激光的角度垂直于进入矩形的边，这样激光一定会从某个边界位置垂直于边界出来。

对于一种摆放镜子的方案，可以记录 $2(n+m)$ 个进入位置对应的出来位置，称一种方案是好的，当且仅当可以在方案中删除一面镜子，但所有进入位置对应的出来位置不变。求好的方案数量，模 $10^9+7$。

$n,m\leq 200$

$2s,256MB$

###### Sol

考虑好方案的性质。如果把镜子看作边，得到的图存在环，则考虑删去环上的一面镜子，可以发现环内部任意一个位置发射的激光会返回原位置，因此环外侧到达这面镜子的激光会在环内部走一圈，最后离开环，这和这里有一面镜子的情况等价。因此这种情况一定合法。

否则，如果没有环，则每个位置都会被某一个位置发射的激光访问到。如果一面镜子的两侧被不同的激光访问，则删去会交换这两束激光，因此不行。否则，交换一定不影响答案。可以发现此时如果翻转这面镜子的方向，则一定有一侧的激光路径形成一个环，因此此时镜子形成一个环。

因此，一种方案合法当且仅当可以通过翻转不超过一面镜子，使得镜子构成环。考虑算不合法的方案数。

此时可以发现对于一个 $2\times 2$ 的菱形，这里面选三条边都是合法的。因此在不合法的方案中，如果出现了菱形中两条相邻的边，则另外两条边都不能选对应方向。即如果出现了 `/\`，则对应的下一行这两个位置也必须是 `/\`，另外三个方向同理。

考虑还需要什么条件，如果当前的图满足上一个条件且仍然合法，对于存在环的情况，环中一定存在一个 `/\`，满足这个图形的右下角有一个 `>` 型的拐弯。但注意到 `/\` 可以向下推，`>` 可以向左推，这一定可以推到如下形状：

```
/\
 /
```

但此时两个要求都一定不能满足，因此满足条件的图中不存在环。

对于环减去一条边的情况，可以发现减去一条边最多减少两种拐弯，因此一定可以找到上面的情况，此时仍然不满足条件。因此所有满足上一个条件的方案即为所有不合法的方案。

考虑最后一行的情况，如果这一行存在一个 `\/`，则这两列上面全部是 `\/`。考虑找到所有满足相邻两列全部是 `\/` 或者全部是 `/\` 的位置，则在这些位置中间，如果有一列不全部相同，则会出现 `<` 或者 `>` 的情况，此时会与两侧的列冲突，因此这些位置中间每一列全部相等。设第一个这种列为 $l$，最后一个为 $r$，则如果 $l<r$，中间可以唯一确定两侧的两行，方案数为 $2^{r-l}$。如果 $i=j$，则中间没有更多的行，两侧有两种方案（`/\`和`\/`）此时中间一定合法，两侧的方案数相当于求 $n$ 行 $k$ 列，最后一列全部为 `/`（另外一种情况等价），且不再存在相邻两行全部为 `/\` 或者另外一种情况的方案数。

此时注意到如果有一个 `/\`，则最后一行有一个 `/\`，则最后一行这个右侧一定有一个 `\/`，此时与最后的假设矛盾。因此没有这种情况。

如果有两个不同列位置有 `\/`，则第一行一定存在 `/\`，同样可以导出矛盾。因此一定最多有相邻两列的一段行前缀中存在 `\/`。可以发现确定了最后一个 `\/` 后，剩余位置方案固定，因此可以发现这种情况的方案数为选择一个这样位置的方案数加上不选的方案数。方案数为 $(n-1)(k-1)+1$。

因此这部分的方案数为 $\sum_{1\leq i\leq j<m}2^{j-i+[i=j]}((n-1)(i-1)+1)((n-1)(m-i-1)+1)$。

考虑接下来的情况，此时不存在相邻两列满足这个条件。如果一行中存在两个 `/\`，则可以得到最后一行存在 `\/`，这不合法。同理一行中不能存在两个 `\/`。因此所有的 `/\` 一定是相邻两列的一个后缀，`\/`同理。

分情况讨论：

如果 `/\`和 `\/` 都存在，则考虑枚举 `/\` 的第一个所在的行 $r$，`\/`最后一个所在的行 $l$。则 $l<n$ 且 $1<r$，此时有以下情况：

1. $l<r$，则两侧行的方案唯一固定，中间每行可以全部是 `/` 或者 `\`，且对 `\/` 所在的列没有要求。此时方案数为 $(m-1)^2*2^{r-l-1}$。
2. $l\geq r$，此时所有行都唯一固定，但 `/\` 和 `\/` 在同一列的情况不合法。因此方案数为 $(m-1)(m-2)$。

如果只存在一种，可以只考虑 `/\` 的情况，枚举它第一次出现的行 $l(2\leq l\leq n)$，则方案数为 $2^{l-1}*(m-1)$。另外一种情况方案数相同。

最后是都不存在的情况，显然每一行任意选一种符号都可以，方案数为 $2^n$。

这里可以直接计算每种情况的方案数，最后用 $2^{nm}$ 减去即可。

如果直接计算，复杂度为 $O((n^2+m^2)\log nm)$，可以通过。

这里也可以大力化简式子，使用w|a可以得到答案为：
$$
2^{nm}-m^2(2^n-1)-n^2(2^m-1)+\frac12nm(n+1)(m+1)-nm+1
$$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 1000000007
int n,m,as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct BlackBoxDiv1{
	int count(int n,int m)
	{
		for(int i=1;i<n;i++)for(int j=i;j<n;j++)as=(as+1ll*pw(2,j-i+(i==j))*((m-1)*(i-1)+1)*((m-1)*(n-j-1)+1))%mod;
		for(int i=1;i<m;i++)as=(as+1ll*pw(2,m-i+1)*(n-1))%mod;
		for(int i=1;i<m;i++)for(int j=1;j<m;j++)
		if(j>=i)as=(as+1ll*(n-1)*(n-1)*pw(2,j-i))%mod;
		else as=(as+1ll*(n-1)*(n-2))%mod;
		as=(as+pw(2,m))%mod;
		return (pw(2,n*m)-as+mod)%mod;
	}
};
```

```cpp
//https://www.wolframalpha.com/input/?i2d=true&i=simplify+Power%5B2%2Cn*m%5D-Sum%5BSum%5BPower%5B2%2Cj-i%5D*%5C%2840%29%5C%2840%29m-1%5C%2841%29*%5C%2840%29i-1%5C%2841%29%2B1%5C%2841%29*%5C%2840%29%5C%2840%29m-1%5C%2841%29*%5C%2840%29n-j-1%5C%2841%29%2B1%5C%2841%29%2C%7Bj%2Ci%2Cn-1%7D%5D%2C%7Bi%2C1%2Cn-1%7D%5D-Sum%5B%5C%2840%29%5C%2840%29%5C%2840%29m-1%5C%2841%29*%5C%2840%29i-1%5C%2841%29%2B1%5C%2841%29*%5C%2840%29%5C%2840%29m-1%5C%2841%29*%5C%2840%29n-i-1%5C%2841%29%2B1%5C%2841%29%5C%2841%29%2C%7Bi%2C1%2Cn-1%7D%5D-Sum%5B%5C%2840%29n-1%5C%2841%29*Power%5B2%2C%5C%2840%29m-i%2B1%5C%2841%29%5D%2C%7Bi%2C1%2Cm-1%7D%5D-Sum%5BSum%5B%5C%2840%29Power%5B2%2Cj-i%5D*Power%5B%5C%2840%29n-1%5C%2841%29%2C2%5D%5C%2841%29%2C%7Bj%2Ci%2Cm-1%7D%5D%2C%7Bi%2C1%2Cm-1%7D%5D-Sum%5BSum%5B%5C%2840%29n-1%5C%2841%29*%5C%2840%29n-2%5C%2841%29%2C%7Bi%2Cj%2B1%2Cm-1%7D%5D%2C%7Bj%2C1%2Cm-1%7D%5D-Power%5B2%2Cm%5D
#define mod 1000000007
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct BlackBoxDiv1{
	int count(int n,int m){return (80000ll*mod+pw(2,n*m)+m*n*(m*n+m+n-1)/2-1ll*m*m*(pw(2,n)-1)-1ll*n*n*(pw(2,m)-1)+1)%mod;}
};
```

##### TwiceTwiceTree

###### Problem

有一棵树，初始只有 $1$ 个节点。进行 $n$ 次操作，每次操作对于操作前树的每个节点，向这个节点加入一个叶子节点。操作后树有 $2^n$ 个节点。

求树上距离为 $d$ 的点对数量，对质数 $p$ 取模。

$n\leq 10^9,d\leq 500,503\leq p\leq 10^9+7$

$2s,256MB$

###### Sol

如果在第 $i$ 次操作中，将新加入的点编号看作父亲的编号加上 $2^{i-1}$，则所有点编号为 $0,\cdots,2^n-1$，且 $i$ 的父亲为 $i$ 减去 $i$ 的最高位的结果。这里也可以翻转二进制表示看成减去 $lowbit$，接下来考虑后者。

此时可以发现两个点的LCA即为两个点二进制表示的LCP后面补0，考虑枚举LCP，设LCP长度为 $i$，则前面有 $2^i$ 种情况，下一位情况固定。此时两个点距离为 $1$，且接下来的每一位每有一个 $1$ 会使距离加一。

因此答案可以看成：
$$
\sum_{i=0}^{n-1}2^iC_{2(n-i-1)}^{d-1}
$$
从生成函数的角度这相当于：
$$
[x^{d-1}]\sum_{i=0}^{n-1}2^i(x+1)^{2(n-i-1)}\\
=[x^{d-1}]\frac{2^n-(x+1)^{2n}}{2-(x+1)^2}
$$
分母为 $1-2x-x^2$，比较系数可以发现除以 $1-2x-x^2$ 相当于 $f'_n=f_n+2f_{n-1}+f_{n-2}$。因为 $p$ 是大于 $n$ 的质数，分子组合数可以直接推。求出分子前 $d$ 项后直接做即可。

复杂度预处理逆元可以做到 $O(d)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505
int n,d,p,f[N];
int pw(int a,int b,int p){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
struct TwiceTwiceTree{
	int sumup(int n,int d,int p)
	{
		f[0]=p-1;for(int i=1;i<=d;i++)f[i]=1ll*f[i-1]*(2*n+1-i)%p*pw(i,p-2,p)%p;
		f[0]=(f[0]+pw(2,n,p))%p;
		for(int i=1;i<=d;i++)
		{
			f[i]=(f[i]+2ll*f[i-1])%p;
			if(i>1)f[i]=(f[i]+f[i-2])%p;
		}
		return f[d-1];
	}
};
```

##### Trinity

###### Problem

有一个 $n\times m$ 的网格，每个位置可以是0或者1，求出如下序列：

1. $a_i$ 表示第 $i$ 行第一个 $1$ 的位置（如果不存在为 $0$)
2. $b_i$ 表示第 $i$ 列第一个 $1$ 的位置（如果不存在为 $0$)
3. $c_i$ 表示第 $i$ 列最后一个 $1$ 的位置（如果不存在为 $0$)

求出可能的 $(a,b,c)$ 的数量，模 $998244353$

$n\leq 8000,m\leq 200$

$6s,256MB$

###### Sol

$a,b,c$ 相当于矩阵三个方向上第一个出现的位置，因此考虑在最后一个方向上 $dp$，即每次向最后加入一列。

但如果某一行没有 $1$，则向最后加入一列仍然会影响 $a$。因此考虑设 $dp_{i,j}$ 表示 $i$ 行 $j$ 列，额外要求每一行都有 $1$ 时整体的方案数。答案显然为 $\sum_{i=0}^nC_n^idp_{i,m}$。

此时考虑向后加入一行的过程，枚举新增加了多少行，则这些行在这一列上都是 $1$，剩下的行任意，且这些行可以任意排列。设之前有 $a$ 行，加入了 $b$ 行，则 $b\neq 0$ 时，问题可以看成有 $a+b$ 个位置，先选择 $b$ 个位置作为新加入的行，然后在第一个新加入的位置及左侧任意选择一个位置（这一列第一个出现），在最后一个位置即右侧任意选择一个位置（最后一个出现）。

可以看成有 $a+b+2$ 个位置（$0$ 下标），选择 $b+2$ 个位置。选择的第一个位置表示左侧第一个出现的位置（如果是 $0$ 则表示第一个位置是新加入的第一个位置），右侧类似，此时两种方案可以一一对应。因此方案数即为 $C_{a+b+2}^{b+2}$。

对于 $b=0$ 的情况，相当于 $a$ 行内任意，然后求第一个位置和最后一个位置的方案数。这显然是 $C_a^2+a+1=C_{a+2}^2-a$。此时直接 $dp$ 复杂度即为 $O(n^2m)$。转移可以使用NTT优化，复杂度 $O(nm\log n)$。

考虑将 $dp$ 看成生成函数，设 $F_m(x)=\sum dp_{i,m}x^i$。则 $F_i$ 到 $F_{i+1}$ 的转移有两部分：先做一次 $C_{a+b+2}^{b+2}$ 的卷积，再减去 $a$。

为了简单地表示卷积组合数的形式，考虑写成EGF的形式，即 $G_m(x)=\sum\frac{dp_{i,m}}{i!}x^i$。

此时第二部分可以看成 $xG'_m(x)$，考虑第一部分。$C_{a+b+2}^{b+2}=\frac{(a+b+2)!}{a!(b+2)!}$， $a!$ 可以直接看成EGF的系数，卷积的函数为 $\sum_{i\geq 0}\frac 1{(i+2)!}x^i=\frac{e^x-x-1}{x^2}$，最后需要乘上一个 $a+b+2$ 的阶乘，而直接表示为EGF系数为 $(a+b)!$，因此可以看成乘以 $x^2$，再求导两次。因此可以得到转移为：
$$
F_m(x)=(F_{m-1}(x)*(e^x-x-1))''-xF_{m-1}'(x)
$$
初值为 $F_0(x)=1$，可以发现 $F_m(x)$ 一定可以写成 $\sum x^ae^{bx}$ 的形式，其中 $0\leq a,b\leq m$。这样直接求出 $F_m(x)$ 的表示复杂度为 $O(m^3)$。

计算答案中的卷积相当于再乘一个 $e^x$，考虑算答案，相当于求 $n![x^n]x^ae^{bx}$，这相当于 $[a\leq n]b^{n-a}\frac{n!}{(n-a)!}$，可以直接求。

复杂度 $O(m^3)$。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 205
#define mod 998244353
int n,m,f[N][N],g[N][N],h[N][N],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&m);
	f[0][0]=1;
	for(int i=1;i<=m;i++)
	{
		for(int j=0;j<=i;j++)for(int k=0;k<=i;k++)g[j][k]=h[j][k]=0;
		for(int j=0;j<i;j++)for(int k=0;k<i;k++)
		{
			g[j][k+1]=(g[j][k+1]+f[j][k])%mod;
			g[j+1][k]=(g[j+1][k]+mod-f[j][k])%mod;
			g[j][k]=(g[j][k]+mod-f[j][k])%mod;
			h[j+1][k]=(h[j+1][k]+mod-f[j][k])%mod;
		}
		for(int j=0;j<=i;j++)for(int k=0;k<=i;k++)
		{
			h[j][k]=(h[j][k]+1ll*k*g[j][k])%mod;
			if(j)h[j-1][k]=(h[j-1][k]+1ll*j*g[j][k])%mod;
		}
		for(int j=0;j<=i;j++)for(int k=0;k<=i;k++)
		{
			f[j][k]=(f[j][k]+1ll*k*h[j][k])%mod;
			if(j)f[j-1][k]=(f[j-1][k]+1ll*j*h[j][k])%mod;
		}
	}
	for(int i=0,tp=1;i<=m&&i<=n;tp=1ll*tp*(n-i)%mod,i++)for(int j=1;j<=m;j++)as=(as+1ll*f[i][j]*pw(j+1,n-i)%mod*tp)%mod;
	printf("%d\n",as);
}
```

##### LotsOfLines

###### Problem

给出 $n,m$，对于每对满足 $a\in[0,n),b\in[0,m)$ 的非负整数 $(a,b)$，有一条直线 $y=ax+b$。

求这些直线将平面划分成的区域个数。

$n,m\leq 1200$

$2s,256MB$

###### Sol

考虑依次加入直线，可以发现加入一条直线时，新增加的面数为这条直线与之前直线的不同交点数量 $+1$。

因此最后的答案为每个交点贡献（经过这个点的直线数量减一），最后再加上 $nm+1$。

设 $f_i$ 表示有 $i$ 条直线经过的交点对答案的贡献，则 $f_i=i-1$。

考虑容斥，设 $g_i$ 表示枚举 $i$ 条直线交于一点，这样的贡献系数，则有 $f_i=\sum_{j\leq i}C_i^jg_i$。

如果看成EGF，则 $F(x)=\sum_{i>1}\frac{i-1}{i!}x^i=(x-1)e^x+1$。

从EGF的角度，上述式子相当于 $F(x)=G(x)e^x$，因此 $G(x)=e^{-x}+x-1=\sum_{i\geq 2}\frac{(-1)^i}{i!}x^i$，从而 $g_i=[i\geq 2](-1)^i$。

此时考虑容斥暴力，因为 $a$ 相同的点不会相交，考虑枚举若干个 $a_1<\cdots<a_l$，钦定会有 $a$ 等于这些的边交于一点，考虑求出方案数。

即给出 $a_1x+b_1=\cdots=a_lx+b_l$ 中的所有 $a$，每个 $b$ 属于 $[0,m)$，求有多少组整数 $b_{1,\cdots,l}$ 使得存在解。因为 $b$ 是整数，因而需要所有的 $x(a_j-a_i)$ 是整数，这当且仅当所有的 $x(a_{i+1}-a_i)$ 是整数。因此这里的要求可以看成 $x*\gcd(a_2-a_1,\cdots,a_l-a_{l-1})$ 是整数。

如果满足这个条件，此时因为 $a_1<\cdots<a_l$，因而 $b$ 的最大最小值一定在 $b_1,b_l$ 取到。因此这可以看成要求 $|x(a_1-a_l)|\leq m-1$，方案数为 $m-|x(a_1-a_l)|$。设 $\gcd(a_2-a_1,\cdots,a_l-a_{l-1})=g,a_l-a_i=s$，则相当于求 $xs\in \Z,|xg|\leq m-1$ 的方案数。因为 $g|s$，令 $y=xs$，相当于对满足 $|y*\frac gs|\leq m-1$ 的整数求和 $m-1-|\frac{yg}{s}|$。这只和 $\frac gs$ 有关，设 $f_k$ 为 $\frac gs=k$ 的方案数，则有 $f_k=2*(\sum_{i=0}^{\lfloor\frac mk\rfloor}(m-ik))-m)$。

设 $dp_{a,b}$ 表示选择了若干个数，最后一个数和第一个数的差为 $a$，所有相邻两个数的差的gcd为 $b$，所有选了大于等于两个数的方案的 $(-1)^k$ 和。如果暴力转移，最后枚举 $a,b$ 直接求和（有放回原序列的 $n-a$ 系数），复杂度为 $O(n^2\log n)$，可以通过。

但可以注意到， $dp_{a,b}$ 只和 $\frac ab$ 有关，且相当于如下问题的答案：选择一列若干个数和为 $\frac ab$，所有数互质，选择 $k$ 个数的方案为 $(-1)^{k+1}$（这里的选数为选择差）。

设 $h_i$ 表示没有互质限制的答案，则考虑分界点可以发现 $h_i=[i=1]$。

设 $g_i$ 表示有限制的答案，则容易发现 $h_i=\sum_{j|i}g_i$，即 $h=1*g=\epsilon$，因此 $g=\mu$，即 $g_i=\mu(i)$。

考虑枚举 $t=\frac gs$，考虑这步到枚举 $a,b$ 间的系数，则不难得到这部分的系数为 $v_k=\sum_{i=1}^{\lfloor\frac nt\rfloor}(n-ik)$。

最后答案即为 $\sum_{t=1}^n\mu(t)f_tv_t$，复杂度 $O(n)$，也可以数论分块+杜教筛 $id^{0,1,2}*\mu$ 前缀和，做到 $O(n^{\frac 23})$。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1250
#define ll long long
int n,m;
ll f[N],g[N],as;
int gcd(int a,int b){return b?gcd(b,a%b):a;}
struct LotsOfLines{
	ll countDivisions(int n,int m)
	{
		for(int i=1;i<=n;i++)
		{
			f[i]=1;
			f[i]*=(m+m%i)*(m/i+1)-m;
			f[i]*=(n-i+n%i)*(n/i)/2;
		}
		as=n*m+1;
		g[1]=1;
		for(int i=1;i<=n;i++)
		{
			for(int j=i*2;j<=n;j+=i)g[j]-=g[i];
			as+=g[i]*f[i];
		}
		return as;
	}
};
```

##### Shoot Your Gun!

###### Problem

给出一个有 $n$ 条边，且每条边都平行于坐标轴的多边形 $M$，多边形内部有两个分别由 $s,t$ 条边，且所有边平行于坐标轴的多边形 $S,T$。

你需要选择 $S$ 边界上的一个任意位置，选择四个对角中的一个方向，向这个方向发射一个点。点碰到 $M$ 的边界会反弹，碰到一个角则会反弹回来时的方向。

你需要使得这个点在再次接触到 $S$ 之前接触到 $T$ 的边界，在此基础上接触到 $T$ 时点反弹的次数最小。求最小次数。

$n,s,t\leq 50,0\leq x_i,y_i\leq 4000$

$18s,256MB$

###### Sol

从一个单位长度的边内部单个方向出发的所有情况等价，因此可以只考虑中点出发的情况，此时可以看成横纵坐标各自乘 $2$，变成只考虑点上出发。

因为运动可以逆向，可以发现两个不同的点出发，不可能从某个时刻开始运动轨迹相同。因此每个状态只会在所有出发点的情况中被经过一次。因此可以预处理后直接暴力模拟，复杂度 $O(v^2)$，但细节还是很多。

另外一种做法是加速中间的过程，考虑每次直接找下一个碰到的边界。可以对于每条对角线处理这条对角线经过的边界，然后每次可以 $O(\log n)$ 找下一个边界。因为每个边界状态只经过一次，这样的复杂度为 $O(vn\log n)$。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 4050
int n,s1,s2,sl[N][2],sr[N][2],s[N][2],as,t,mx;
int fg[N][N];
void doit(int sl[][2],int s1,int f1)
{
	for(int i=1;i<=s1;i++)if(sl[i][0]==sl[i%s1+1][0])
	{
		int sx=sl[i][0],ly=sl[i][1],ry=sl[i%s1+1][1];
		mx=max(mx,max(sx,max(ly,ry)));
		if(ly>ry)ly^=ry^=ly^=ry;
		for(int j=ly+1;j<=ry;j++)fg[sx][j]^=f1;
	}
}
int solve(int x,int y,int dx,int dy)
{
	int ct=0;
	while(1)
	{
		if((x&1)&&(y&1))
		{
			int x1=(x+dx)>>1,y1=(y+dy)>>1,f1=0;
			for(int i=x1-1;i<=x1;i++)for(int j=y1-1;j<=y1;j++)f1|=fg[i][j];
			if(f1&2)return 1e9;if(f1&4)return ct;
		}
		int tx=((x<<1)+dx)>>2,ty=((y<<1)+dy)>>2;
		if(fg[tx][ty]&2)return 1e9;
		if(fg[tx][ty]&4)return ct;
		if(!fg[tx][ty])
		{
			ct++;
			if(x&1)dy*=-1;else if(y&1)dx*=-1;
			else
			{
				int s1=!fg[(x-dx)>>1][ty],s2=!fg[tx][(y-dy)>>1];
				if((s1^s2)==0)return 1e9;
				if(s1)dy*=-1;else dx*=-1;
			}
		}
		x+=dx,y+=dy;
	}
}
int main()
{
	while(1)
	{
		scanf("%d",&s1);if(!s1)return 0;
		scanf("%d%d",&s2,&n);mx=0;
		for(int i=1;i<=s1;i++)scanf("%d%d",&sl[i][0],&sl[i][1]);
		for(int i=1;i<=s2;i++)scanf("%d%d",&sr[i][0],&sr[i][1]);
		for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]);
		doit(sl,s1,2);doit(sr,s2,4);doit(s,n,1);
		for(int i=0;i<=mx;i++)for(int j=mx;j>=0;j--)fg[j][i]^=fg[j+1][i];
		for(int i=1;i<=10;i++,printf("\n"))
		for(int j=1;j<=10;j++)printf("%d",fg[i][j]);
		as=1e9;
		for(int i=1;i<=s1;i++)if(sl[i][0]==sl[i%s1+1][0])
		{
			int sx=sl[i][0],ly=sl[i][1],ry=sl[i%s1+1][1];
			if(ly>ry)ly^=ry^=ly^=ry;
			for(int j=ly*2;j<=ry*2;j++)for(int s=0;s<2;s++)for(int t=0;t<2;t++)as=min(as,solve(sx*2+2,j+2,s*2-1,t*2-1));
		}
		else
		{
			int sx=sl[i][1],ly=sl[i][0],ry=sl[i%s1+1][0];
			if(ly>ry)ly^=ry^=ly^=ry;
			for(int j=ly*2;j<=ry*2;j++)for(int s=0;s<2;s++)for(int t=0;t<2;t++)as=min(as,solve(j+2,sx*2+2,s*2-1,t*2-1));
		}
		if(as>1e7)as=-1;
		printf("Case %d: %d\n",++t,as);
		for(int i=0;i<=mx+1;i++)for(int j=0;j<=mx+1;j++)fg[i][j]=0;
	}
}
```

##### Three Gluttons

###### Problem

给定两个 $n$ 阶排列 $a,b$，保证 $3|n$。求有多少个 $n$ 阶排列 $c$ 满足如下如下条件：

现在有 $n$ 个数 $1,\cdots,n$，进行 $\frac n3$ 轮操作，每轮中对于每个排列找到这个排列中当前第一个没有被拿走的数，然后同时拿走三个排列对应的数。这个过程中每一轮选出的三个数都不同。

答案模 $10^9+7$

$n\leq 400$

$2s,512MB$

###### Sol

考虑第三个排列拿走的数与第三个排列间的对应关系。如果这种拿走数的方式合法，则排列中第一轮另外两个拿走的数出现在第一个被拿走的数之后，剩余的同理。显然排列中被拿走的 $\frac n3$ 个数一定顺序排列，考虑从后往前将其它的数插入进去，则每次能插入到一个后缀，且如果从后往前插入，则后面插入的后缀覆盖前面的后缀。考虑每一个插入的情况，可以得到每种拿走数的合法方案对应 $\prod_{i=1}^{\frac n3}(3i-1)(3i-2)$ 种排列。

因此问题变为计算有多少种长度为 $\frac n3$ 的序列，使得第三个排列按照这个序列拿走数，原问题合法。考虑对于一种前两个排列拿走的数的方式 $s_{1,\cdots,\frac n3},t_{1,\cdots,\frac n3}$，什么情况下可以合法。则需要如下条件：

1. 对于 $a$ 中拿走的第 $k$ 个数，它在 $b$ 中不能出现在拿走的第 $k$ 个数以及之前。另外一侧同理。
2. 对于剩下的数，设它在 $a$ 中出现在第 $k_1$ 个拿走的数之后，在 $b$ 中出现在第 $k_2$ 个拿走的数之后，则它在 $c$ 中必须在前 $\min(k_1,k_2)$ 个被拿走。

可以发现所有对 $c$ 的限制都相当于要求一个数需要在前若干个位置被拿走，因此可以依次考虑 $c$ 的每一位，记录前面还有多少个空余位置，每个限制可以填一个空余位置，即可得到方案。

但可以注意到如果记录了当前 $c$ 填了多少位，$a$ 上一个选的在排列中的位置，$b$ 上一个选的在排列中的位置，则前面有限制的 $c$ 中填的数个数为 $a[1,i],b[1,j]$ 中出现过的数种类减去 $2$ 倍 $c$ 考虑的位数。因此空余位置和考虑的长度可以只记录一个。为了更好的确定转移顺序，这里考虑记录长度。

具体来说，设 $dp_{i,j,k}$ 表示考虑了前 $i$ 轮，第 $i$ 轮拿走了 $a_j,b_k$ 的方案数。考虑向后转移到 $dp_{i+1,j',k'}$ ，则限制有 $j<j',k<k'$，$j',k'$ 满足限制1。转移系数与新加入区间内有多少个之前没有出现的数有关。

此时考虑分维转移，先增加 $j$ 再增加 $k$，每遇到一个不在之前两个前缀中出现的数时考虑填进 $c$ 的方案数，最后再判断 $j,k$ 的合法性。这样单次转移即为 $O(n^2)$。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 401
#define mod 1000000007
int n,p[N],q[N],sl[N][N],sr[N][N],su[N][N],as;
int dp[135][N][N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&p[i]);
	for(int i=1;i<=n;i++)scanf("%d",&q[i]);
	for(int i=1;i<=n;i++)
	{
		sl[i][0]=1;sr[i][0]=1;
		for(int j=1;j<=n;j++)sl[i][j]=sl[i][j-1]&(q[j]!=p[i]),sr[i][j]=sr[i][j-1]&(p[j]!=q[i]);
	}
	for(int i=0;i<=n;i++)
	{
		su[i][0]=i;
		for(int j=1;j<=n;j++)su[i][j]=su[i][j-1]+sr[j][i];
	}
	dp[0][0][0]=1;
	for(int i=1;i<=n/3;i++)
	{
		for(int j=0;j<=n;j++)for(int k=0;k<=n;k++)dp[i][j+1][k+1]=dp[i-1][j][k];
		for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)
		{
			int tp=1;
			if(sl[j][k-1])tp=(i-1)*3-su[j-1][k-1];
			dp[i][j+1][k]=(dp[i][j+1][k]+1ll*tp*dp[i][j][k])%mod;
		}
		for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)
		{
			int tp=1;
			if(sr[k][j-1])tp=(i-1)*3-su[j-1][k-1];
			dp[i][j][k+1]=(dp[i][j][k+1]+1ll*tp*dp[i][j][k])%mod;
		}
		for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)if(!sl[j][k]||!sr[k][j])dp[i][j][k]=0;
	}
	for(int j=1;j<=n;j++)for(int k=1;k<=n;k++)if(dp[n/3][j][k])
	{
		int tp=dp[n/3][j][k],ls=n-su[j][k];
		while(ls)tp=1ll*ls*tp%mod,ls--;
		as=(as+tp)%mod;
	}
	for(int i=1;i<=n/3;i++)as=1ll*as*(3*i-2)*(3*i-1)%mod;
	printf("%d\n",as);
}
```

##### Secret Passage

###### Problem

有一个长度为 $n$ 的 $01$ 串 $s$，你可以进行若干次操作，每次操作你可以删去串的前两个字符，并在这两个字符中任意选择一个字符，插入到串的任意位置。

求可以得到的不同串数量，模 $998244353$

$n\leq 300$

$2s,1024MB$

###### Sol

考虑不将字符立刻插入回去，而是看成有一些可以任意放置的字符。此时操作后的状态可以被表示为当前剩余的后缀开头 $i$，当前可以任意放置的 $0$ 数量 $j$，当前可以任意放置的 $1$ 数量 $k$。一次操作可以选择后缀的两个数，或者后缀的一个数加上一个任意放置的数，或者两个任意方式的数。因此可以使用dfs的方式找出所有可以到达的状态。这部分复杂度 $O(n^3)$。

但此时一个串还可以被多个状态表示。注意到对于一个串，考虑找到这个串中是 $s$ 的后缀的最长子序列，这个长度是唯一的，考虑在这个位置计算。如果枚举能匹配到的后缀为 $s[a,n]$，剩余 $b$ 个 $0$ 和 $c$ 个 $1$，则这种情况合法当且仅当之前的操作能到达一个状态 $(i,j,k)$，满足 $a\leq i$ 且 $(b,c)$ 加上 $s[a,i-1]$ 这一段的字符后 $01$ 个数变为 $j,k$。

可以通过对所有状态按照 $i$ 从大到小预处理，得到哪些 $(a,b,c)$ 是合法的。这部分复杂度仍然是 $O(n^3)$。

最后只需要对每个 $a,b,c$ 算方案数。考虑将串翻转，则变为找一个是 $s$ 的前缀的最长子序列。显然可以贪心匹配，因此可以对 $(a,b,c)$ 的状态进行dp，求出方案数。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 312
#define mod 998244353
int n,dp[N][N][N],vis[N][N][N],as;
char s[N];
void dfs(int x,int y,int l)
{
	if(vis[x][y][l])return;vis[x][y][l]=1;
	if(l>=2)for(int t=0;t<2;t++)dfs(x+(s[l-t]=='0'),y+(s[l-t]=='1'),l-2);
	if(l&&x)dfs(x,y,l-1),dfs(x-1+(s[l]=='0'),y+(s[l]=='1'),l-1);
	if(l&&y)dfs(x,y,l-1),dfs(x+(s[l]=='0'),y-1+(s[l]=='1'),l-1);
	if(x&&x+y>1)dfs(x-1,y,l);
	if(y&&x+y>1)dfs(x,y-1,l);
}
int main()
{
	scanf("%s",s+1);n=strlen(s+1);
	for(int i=1;i*2<=n;i++)swap(s[i],s[n-i+1]);
	dp[0][0][0]=1;
	for(int i=0;i<=n;i++)for(int j=0;i+j<=n;j++)for(int t=0;t<=i+j;t++)for(int p=0;p<2;p++)
	dp[i+!p][j+p][t+(s[t+1]==p+'0')]=(dp[i+!p][j+p][t+(s[t+1]==p+'0')]+dp[i][j][t])%mod;
	dfs(0,0,n);
	for(int i=0;i<=n;i++)
	for(int j=0;j<=n;j++)for(int k=0;k<=n;k++)if(vis[j][k][i])
	{
		int tp=s[i+1]-'0';
		int nj=j-!tp,nk=k-tp;
		if(nj>=0&&nk>=0)vis[nj][nk][i+1]=1;
	}
	for(int i=0;i<=n;i++)
	{
		int s1=0,s2=0;
		for(int j=1;j<=i;j++)if(s[j]=='1')s2++;else s1++;
		for(int j=0;j+s1<=n;j++)for(int k=0;k+s2<=n;k++)if(vis[j][k][i])
		as=(as+dp[j+s1][k+s2][i])%mod;
	}
	printf("%d\n",as);
}
```

##### O(rand)

###### Problem

有 $n$ 个数 $v_{1,\cdots,n}$，求多少种从它们中选出不超过 $k$ 个的方式，使得它们的and和为 $s$，or和为 $t$。

$n,k\leq 50,v<2^{m},m=18$

$3s,512MB$

###### Sol

通过一些预处理，容易将问题变为 $s=0,t=2^v-1$ 的形式。考虑暴力容斥，即枚举一些位让这些位or为 $0$，枚举另外一些位让它们and为 $1$。满足这些条件的数可以任意选择，因此方案数可以直接计算。直接算的复杂度为 $O(n3^m)$。

考虑先枚举所有为 $0$ 的位，进行若干处理，此时再考虑枚举一些为 $1$ 的位的问题，此时一个数会向它的二进制位的子集转移。如果使用FWT，则复杂度为 $O(m3^m)$。

但因为 $n$ 很小，考虑暴力做FWT，设一个数二进制表示有 $a$ 个 $1$，则复杂度为 $O(2^a)$，但这样的数只会在枚举 $0$ 时被枚举 $2^{n-a}$ 次，因此一个数枚举的复杂度为 $O(2^m)$。如果重编号后最后将所有位置的答案算一遍，则复杂度为 $O(3^m+n2^m)$。如果只算有值的位，复杂度即为 $O(n2^m)$，两者均可通过。

这里还有另外一种做法：

注意到枚举 $0,1$ 位后，相当于要求所有选择的数这些位都等于一个给定情况。考虑一些数使得这些数在某些位上的情况相同，则正好存在一种枚举 $0,1$ 位的情况，使得枚举的 $0,1$ 位的并为这些位且这些数合法。因此可以考虑枚举两种位的并集，考虑情况变为选择的数在这些位上的情况相同。

即考虑设 $f_s$ 表示有多少种选择数的方案，使得选择的所有数 and $s$ 的结果相同。最后答案即为 $\sum (-1)^{|s|}f_s$。计算 $f$ 容易做到 $O(n2^m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 55
#define ll long long
int n,k,d,s,t,v[N],v2[N],ct,id[263001],su[263001],f1[263001];
ll as,c[N][N],f[N];
vector<int> si;
void dfs(int v,int x)
{
	if(!v){if(!su[x])si.push_back(x);su[x]++;return;}
	int tp=v&-v;v-=tp;
	dfs(v,x);dfs(v,x+id[tp]);
}
int main()
{
	scanf("%d%d%d%d",&n,&k,&s,&t);
	if((s&t)!=s){printf("0\n");return 0;}
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)if((v[i]&s)==s&&(v[i]|t)==t)v[++d]=v[i]^s;
	n=d;t^=s;d=0;
	for(int i=1;i<=18;i++)if((t>>i-1)&1)
	{
		d++;
		for(int j=1;j<=n;j++)if((v[j]>>i-1)&1)v2[j]|=1<<d-1;
	}
	for(int j=1;j<=n;j++)v[j]=v2[j];
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=c[i-1][j]+c[i-1][j-1];
	for(int i=1;i<=n;i++)for(int j=1;j<=i&&j<=k;j++)f[i]+=c[i][j];
	f1[0]=1;for(int i=1;i<1<<18;i++)f1[i]=-f1[i-(i&-i)];
	for(int i=0;i<1<<d;i++)
	{
		ct=0;int fg=1,sc=0;
		si.clear();
		for(int j=1;j<=d;j++)if((i>>j-1)&1)fg*=-1;else id[1<<j-1]=1<<sc,sc++;
		for(int j=1;j<=n;j++)if((v[j]&i)==0)dfs(v[j],0);
		for(int j=0;j<si.size();j++)as+=fg*f1[si[j]]*f[su[si[j]]],su[si[j]]=0;
	}
	printf("%lld\n",as);
}
```

##### 新年的追逐战

###### Problem

给定 $n$ 以及正整数 $v_{1,\cdots,n}$。有 $n$ 张图 $G_1,\cdots,G_n$，第 $i$ 张图有 $v_i$ 个点。

定义图 $G_1,...,G_n$ 的直积为一张 $\prod v_i$ 个点的图，点可以被标号为 $(a_1,\cdots,a_n)(1\leq a_i\leq v_i)$，两个点 $(s_1,\cdots,s_n),(t_1,\cdots,t_n)$ 有边当且仅当在每个图中 $(s_i,t_i)$ 都有边（这里 $s_i=t_i$ 的情况视为没有边）。

现在每张图的每条边都有 $\frac 12$ 的概率存在，求图直积的连通块个数的期望乘以 $2^{\sum C_{v_i}^2}$ 的结果，模 $998244353$。

$n,v_i\leq 10^5$

$1s,512MB$

###### Sol

上述期望可以看成对所有情况求和。两个点 $(s_1,\cdots,s_n),(t_1,\cdots,t_n)$ 连通当且仅当对于存在一个 $l$，使得 $G_i$ 中 $s_i$ 到 $t_i$ 有一条长度为 $l$ 的路径。因此如果存在一个 $s_i$ 在 $G_i$ 中为孤立点，则 $(s_1,\cdots,s_n)$ 也是孤立点。

考虑剩下的点，此时每个 $s_i$ 都不是孤立点。由于可以在一条边上反复走，因此上述条件可以弱化为存在 $l\in\{0,1\}$ 使得 $s_i$ 到 $t_i$ 有一条长度模 $2$ 为 $l$ 的路径。如果 $G_i$ 中 $s_i$ 所在的连通块是一个二分图，则 $s_i$ 到连通块内任意点的路径长度模 $2$ 固定，如果不是二分图则可以任意。

又因为如果两个点 $(s_1,\cdots,s_n),(t_1,\cdots,t_n)$ 连通，则 $G_i$ 中 $s_i$ 与 $t_i$ 连通。因此考虑对于每个图找一个连通块 $S_i$，计算所有 $s_i\in S_i$ 的情况的连通块数，对于所有图的所有连通块求和即为总连通块数。

如果不考虑 $S_i$ 是单点的情况，设 $S_i$ 中有 $k$ 个是二分图，剩下的不是二分图，则将所有二分图黑白染色，一个点 $(s_1,\cdots,s_n)$ 只能走到在每个二分图中对应点和 $s_i$ 颜色全部相同或者全部相反的点。因此这里会有 $2^{\max(k-1,0)}$ 个连通块。

这可以看成每个二分图连通块有 $2$ 的贡献，如果出现过二分图则有 $\frac 12$ 的贡献。那么可以看成在每个二分图有 $2$ 的贡献的情况下，分别计算全部不为二分图的连通块数，以及可以是二分图（但不能是单点）的情况的连通块数，两者相加除以 $2$ 即为答案。不难发现全部不为二分图的连通块数即为每个图中所有情况中不为二分图的连通块个数的乘积，另外一种类似，因此考虑计数这个。

首先考虑连通块计数，设连通图的EGF为 $F(x)$，则图的EGF显然为 $\sum_{i\geq 0}\frac{F^i(x)}{i!}=e^{F(x)}$，对于计数可以考虑枚举一个连通块，那么可以发现这等于 $F(x)\sum_{i\geq 0}\frac{F^i(x)}{i!}=F(x)e^{F(x)}$。直接从生成函数出发也容易得到这一点。

那么设连通二分图的生成函数为 $G(x)$，连通非二分图的生成函数为 $H(x)$（根据上面的分析，这里不考虑单点的情况）。则一个图中连通非二分图的数量即为 $H(x)e^{F(x)}$，可以包含二分图且计算贡献的数量的情况为 $(H(x)+2G(x))e^{F(x)}$。

因此这部分的贡献即为 $\frac12(\prod_{i=1}^n[x^{v_i}](H(x)+2G(x))e^{F(x)})+(\prod_{i=1}^n([x^{v_i}]H(x)e^{F(x)}))$。

再考虑单点的贡献，这部分为 $\prod v_i$ 减去每个连通块中非单点数量的乘积。这相当于计数图中非单点数量。可以先求出没有孤立点的图的生成函数 $e^{F(x)-x}$，然后求导再乘 $x$ 即可将 $x^i$ 项系数乘以 $i$，最后再乘上单点的生成函数即可得到答案。因此图中非单点数量和的生成函数为 $x(F'(x)-1)e^{F(x)}$。

最后只需要求出 $F,G,H$，即可得到答案。

注意到 $e^{F(x)}$ 是图的生成函数，而这显然等于 $\sum_{i\geq 0}2^{\frac{i(i-1)}2}x^i$，因此求出这个生成函数的 $\ln$ 即可得到 $F$。并且显然有 $F(x)=G(x)+H(x)+x$，只需要求出 $G(x)$。

考虑 $e^{G(x)}$，这相当于不存在奇环的图的生成函数，但这仍然不好求。但二分图一定存在黑白染色，从这个角度出发，设 $a_i$ 表示 $i$ 个点的所有图，对点黑白染色，染色后图中任意边连接异色点的方案数，$A(x)=\sum_{i\geq 0}a_ix^i$，则 $\ln A(x)$ 相当于 $i$ 个点的连通图进行上面的操作。可以发现此时二分图方案数为 $2$，其它图方案数为 $0$。因此 $G(x)=(\frac12\ln A(x))-x$。

只需要求出 $A(x)$，不难得到 $a_i=\sum_{j=0}^iC_i^j2^{j(i-j)}$。这可以看成EGF上的chirp-Z变换，因此可以使用bluestein算法，将 $j(i-j)$ 变为 $C_i^2-C_{i-j}^2-C_j^2$，这样即可将 $a_i$ 看成两个相同的生成函数 $\sum_{i\geq 0}\frac 1{i!2^{C_i^2}}x^i$ 的卷积。

总复杂度 $O(n\log n)$。可以发现虽然推导使用了很多 $\exp$，但最后的结果化简后只需要进行 $\ln$ 以及基础操作。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 263001
#define mod 998244353
int n,v[N],m,rev[N*2],gr[2][N*2],as,fr[N],ifr[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init(int d)
{
	for(int i=2;i<=1<<d;i<<=1)
	{
		for(int j=1;j<i;j++)rev[i+j]=(rev[i+(j>>1)]>>1)|((i>>1)*(j&1));
		for(int t=0;t<2;t++)
		{
			int rv=pw(3,(mod-1)/i);
			if(!t)rv=pw(rv,mod-2);
			gr[t][i]=1;for(int j=1;j<i;j++)gr[t][i+j]=1ll*gr[t][i+j-1]*rv%mod;
		}
	}
	fr[0]=ifr[0]=1;for(int i=1;i<1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
}
int ntt[N];
void dft(int s,int *f,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=f[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=0;j<l>>1;j++)
	{
		int v1=ntt[i+j],v2=1ll*gr[t][l+j]*ntt[i+j+(l>>1)]%mod;
		ntt[i+j]=(v1+v2)%mod;ntt[i+j+(l>>1)]=(v1+mod-v2)%mod;
	}
	int tp=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)f[i]=1ll*ntt[i]*tp%mod;
}
int fi[N],gi[N];
void polyinv(int n,int *f,int *g)
{
	if(n==1){g[0]=pw(f[0],mod-2);return;}
	polyinv((n+1)>>1,f,g);
	int l=1;while(l<=n*2)l<<=1;
	for(int i=0;i<l;i++)fi[i]=gi[i]=0;
	for(int i=0;i<n;i++)fi[i]=f[i];
	for(int i=0;i<(n+1)>>1;i++)gi[i]=g[i];
	dft(l,fi,1);dft(l,gi,1);
	for(int i=0;i<l;i++)fi[i]=1ll*gi[i]*(2+mod-1ll*fi[i]*gi[i]%mod)%mod;
	dft(l,fi,0);
	for(int i=0;i<n;i++)g[i]=fi[i];
}
int fl[N],gl[N];
void polyln(int n,int *f,int *g)
{
	int l=1;while(l<=n*2)l<<=1;
	for(int i=0;i<l;i++)fl[i]=gl[i]=0;
	for(int i=0;i+1<n;i++)fl[i]=1ll*f[i+1]*(i+1)%mod;
	polyinv(n,f,gl);
	dft(l,fl,1);dft(l,gl,1);
	for(int i=0;i<l;i++)fl[i]=1ll*fl[i]*gl[i]%mod;
	dft(l,fl,0);
	for(int i=1;i<n;i++)g[i]=1ll*fl[i-1]*pw(i,mod-2)%mod;
	g[0]=0;
}
int f[N],g[N],f1[N],g1[N],f2[N],g2[N],v1[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)if(m<v[i])m=v[i];
	init(18);
	int l=1;while(l<=m*2)l<<=1;
	//graph
	for(int i=0;i<=m;i++)f[i]=1ll*pw(2,1ll*i*(i-1)/2%(mod-1))*ifr[i]%mod;
	polyln(m+1,f,g1);g1[1]--;
	for(int i=1;i<=m;i++)v1[i]=1ll*i*g1[i]%mod;
	dft(l,f,1);dft(l,v1,1);for(int i=0;i<l;i++)v1[i]=1ll*f[i]*v1[i]%mod;dft(l,v1,0);
	//bipartite graph
	for(int i=0;i<=m;i++)g[i]=1ll*ifr[i]*pw((mod+1)/2,1ll*i*(i-1)/2%(mod-1))%mod;
	dft(l,g,1);for(int i=0;i<l;i++)g[i]=1ll*g[i]*g[i]%mod;dft(l,g,0);
	for(int i=0;i<=m;i++)g[i]=1ll*g[i]*pw(2,1ll*i*(i-1)/2%(mod-1))%mod;
	polyln(m+1,g,g2);
	for(int i=2;i<=m;i++)g2[i]=1ll*(mod+1)/2*g2[i]%mod;g2[1]=g2[0]=0;
	//sum
	for(int i=0;i<=m;i++)f1[i]=(g1[i]+g2[i])%mod,f2[i]=(g1[i]+mod-g2[i])%mod;
	dft(l,f1,1);dft(l,f2,1);
	for(int i=0;i<l;i++)f1[i]=1ll*f[i]*f1[i]%mod,f2[i]=1ll*f[i]*f2[i]%mod;
	dft(l,f1,0);dft(l,f2,0);
	//answer
	int s1=1,s2=1;
	for(int i=1;i<=n;i++)s1=1ll*s1*pw(2,1ll*v[i]*(v[i]-1)/2%(mod-1))%mod*v[i]%mod,s2=1ll*s2*v1[v[i]]%mod*fr[v[i]]%mod;
	as=(s1+mod-s2)%mod;
	s1=s2=1;
	for(int i=1;i<=n;i++)s1=1ll*s1*fr[v[i]]%mod*f1[v[i]]%mod,s2=1ll*s2*fr[v[i]]%mod*f2[v[i]]%mod;
	as=(as+1ll*(mod+1)/2*(s1+s2))%mod;
	printf("%d\n",as);
}
```

##### Goat in the Garden 4

###### Problem

给一个 $n$ 个点的简单多边形，求多边形内部最大的圆的半径。

$n\leq 25$

$1s,64MB$

###### Sol

通过对圆进行平移，缩放以及类似的操作，可以说明存在一种最优解，这种解经过若干个顶点，并与另外若干条边相切，且点数加边数不小于 $3$。因此可以枚举三个点边，解出圆的形态。但这样的细节较为复杂。

考虑二分答案，变为是否能放下这样一个圆。此时一定存在一种解经过两个顶点，或者与两条边相切，或者与一条边相切并经过另外一个顶点。三种情况都不难解出圆心的位置。

然后考虑判断是否合法，考虑每条边容易求出圆是否与多边形相交，最后只需要判断圆心是否在多边形内部，射线法判断即可。

过程中需要若干次点到线段的最小距离，这部分可以先判断三个点是否存在钝角，不存在则相当于求三角形的高，可以面积除以底边再乘2，因此可以 $O(1)$。

复杂度 $O(n^2\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 105
int n,x,y;
struct pt{double x,y;}s[N];
pt operator +(pt a,pt b){return (pt){a.x+b.x,a.y+b.y};}
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
pt operator *(pt a,double b){return (pt){a.x*b,a.y*b};}
double mul(pt a,pt b){return a.x*b.x+a.y*b.y;}
double cross(pt a,pt b){return a.x*b.y-a.y*b.x;}
double getdis(pt a){return sqrt(a.x*a.x+a.y*a.y);}
pt intersect(pt a,pt b,pt c,pt d)
{
	double sz=cross(c-a,b-a)+cross(b-a,d-a);
	return a+(b-a)*cross(c-a,d-a)*(1/sz);
}
double chk(pt x,pt l,pt r)
{
	if(mul(x-r,l-r)<=0)return getdis(x-r);
	if(mul(x-l,r-l)<=0)return getdis(x-l);
	double sz=cross(r-x,l-x);if(sz<0)sz*=-1;
	return sz/getdis(r-l);
}
double chkr(pt x,pt l,pt r)
{
	double sz=cross(r-x,l-x);if(sz<0)sz*=-1;
	return sz/getdis(r-l);
}
bool chk1(pt x,double r)
{
	for(int i=1;i<=n;i++)if(chk(x,s[i],s[i%n+1])<r)return 0;
	pt fu=(pt){114.514,1919.810};
	int ct=0;
	for(int i=1;i<=n;i++)
	{
		int fg=1;
		pt s1=s[i]-x,s2=s[i%n+1]-x;
		if(cross(s1,fu)<0)swap(s1,s2);
		if(cross(s2,fu)>0||cross(s1,fu)<0)fg=0;
		fg&=cross(s1,s2)>0;
		ct+=fg;
	}
	return ct&1;
}
bool check(double r)
{
	for(int i=1;i<=n;i++)for(int j=i+1;j<=n;j++)if(getdis(s[i]-s[j])/2<=r)
	{
		double r1=sqrt(r*r-getdis(s[i]-s[j])*getdis(s[i]-s[j])/4);
		pt p1=(s[i]+s[j])*0.5,p2=s[i]-p1;
		swap(p2.x,p2.y),p2.x*=-1;
		p2=p2*(r1/getdis(p2));
		if(chk1(p1+p2,r-1e-5)||chk1(p1-p2,r-1e-5))return 1;
	}
	for(int i=1;i<=n;i++)for(int j=i+1;j<=n;j++)
	{
		pt r1=intersect(s[i],s[i%n+1],s[j],s[j%n+1]);
		pt f1=s[i]-r1,f2=s[j]-r1;
		if(getdis(f1)<1e-7)f1=s[i%n+1]-r1;
		if(getdis(f2)<1e-7)f2=s[j%n+1]-r1;
		f1=f1*(1/getdis(f1));f2=f2*(1/getdis(f2));
		for(int p=-1;p<2;p+=2)for(int q=-1;q<2;q+=2)
		{
			pt s1=f1*p,s2=f2*q;
			double di=chkr(r1+s1+s2,s[i],s[i%n+1]);
			pt fu=r1+(s1+s2)*(r/di);
			if(chk1(fu,r-1e-5))return 1;
		}
	}
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	{
		double d1=chkr(s[i],s[j],s[j%n+1]);
		if(d1<1e-7||d1>r*2)continue;
		double rds=sqrt(r*r-(r-d1)*(r-d1));
		double pds=sqrt(getdis(s[j]-s[i])*getdis(s[j]-s[i])-d1*d1);
		pt r1=s[j%n+1]-s[j];r1=r1*(1/getdis(r1));
		pt f1=s[j]+r1*pds,f2=s[j]-r1*pds;
		if(getdis(f2-s[i])<getdis(f1-s[i]))f1=f2;
		pt r2=s[i]-f1;r2=r2*(r/getdis(r2));
		for(int q=-1;q<2;q+=2)
		{
			pt fr=f1+r1*(q*rds)+r2;
			if(chk1(fr,r-1e-5))return 1;
		}
	}
	return 0;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&x,&y),s[i]=(pt){x,y};
	double lb=0,rb=1e4;
	for(int t=0;t<=32;t++)
	{
		double mid=(lb+rb)/2;
		if(check(mid))lb=mid;
		else rb=mid;
	}
	printf("%.2lf\n",lb);
}
```

##### Snake

###### Problem

有一个 $n\times m$ 的网格，有些位置是障碍。网格的边界上有一个位置为入口。

现在有一条长度为 $l$ 个格子的蛇要经过这个网格。初始时，蛇的头部在入口处，其它部分在网格外面。接下来的每一次移动，蛇的头部可以向一个空位或者蛇的尾部所在的格子移动，之后蛇会整体移动一位。蛇的头部移动回入口时，认为蛇离开了这个网格。

这个网格保证长度为 $18$ 的蛇不能成功离开网格。对于 $2\sim 17$ 中的每个长度，求出这个长度的蛇离开网格需要的最少移动次数，或输出不可能。

$n\leq 300,m\leq 30$

$3s,64MB$

###### Sol

考虑头部经过的位置构成的序列，这个序列只需要满足相邻两个位置相邻，且一个位置两次出现之间间隔不小于 $l$。

因为起点终点相同，因此路径一定经过了一个环，因而一定经过了一个长度大于等于 $l$ 的环。

更进一步容易发现，一种最优路径一定是走到一个环，绕环一圈，然后原路返回。因此考虑对于每种环长，对于每个点求出这个点是否在一个这种环长的环中。注意到题目保证不存在 $18$ 个点或以上的环，而 $16$ 个点的不同环形状不超过 $3000$，因此考虑先搜出所有的环形状，然后暴力判断。

复杂度为 $nm$ 乘上每种环长乘以环数量的和，这大概是 $10^9$ 级别，但可以进行很多剪枝，例如按照和入口的距离从小到大考虑，如果之后的解不可能更优就退出，以及判断时不合法就退出，这样实际效果很好。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
#define N 301
int n,m,lx,ly,ds[N][N],as[N];
char s[N][N];
int is[21][21],d[4][2]={1,0,-1,0,0,1,0,-1};
int Abs(int x){return x>0?x:-x;}
vector<vector<pair<int,int> > > le[21];
void dfs(int x,int y,int l)
{
	if(l==1&&x==11)return;
	if(x<10||(x==10&&y<10))return;
	if(l+Abs(x-10)+Abs(y-10)>16)return;
	if(x==10&&y==10)
	{
		vector<pair<int,int> > st;
		for(int i=10;i<=20;i++)for(int j=0;j<=20;j++)if(is[i][j])st.push_back(make_pair(i-10,j-10));
		le[l].push_back(st);
	}
	for(int i=0;i<4;i++)
	{
		int nx=x+d[i][0],ny=y+d[i][1];
		if(is[nx][ny])continue;
		is[nx][ny]=1;dfs(nx,ny,l+1);is[nx][ny]=0;
	}
}
vector<pair<int,int> > fr;
void bfs()
{
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)ds[i][j]=-1;
	ds[lx][ly]=0;
	queue<pair<int,int> > st;st.push(make_pair(lx,ly));
	while(!st.empty())
	{
		pair<int,int> nw=st.front();st.pop();
		fr.push_back(nw);
		int rx=nw.first,ry=nw.second,di=ds[rx][ry];
		for(int t=0;t<4;t++)
		{
			int nx=rx+d[t][0],ny=ry+d[t][1];
			if(s[nx][ny]=='X'||ds[nx][ny]!=-1)continue;
			ds[nx][ny]=di+1;st.push(make_pair(nx,ny));
		}
	}
}
int main()
{
	dfs(10,10,0);
	scanf("%d%d%d%d",&n,&m,&lx,&ly);
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	bfs();
	for(int i=1;i<=18;i++)as[i]=1e9;
	for(int i=4;i<=16;i+=2)
	{
		int ls=1e9;
		for(int j=0;j<fr.size();j++)
		{
			int tx=fr[j].first,ty=fr[j].second;
			if(ds[tx][ty]*2+i+2>ls)break;
			for(int l=0;l<le[i].size();l++)
			{
				int as=1e8;
				for(int p=0;p<i;p++)
				{
					int nx=tx+le[i][l][p].first,ny=ty+le[i][l][p].second;
					if(s[nx][ny]=='X'||nx>n||ny<1||ny>m){as=1e9;break;}
					else as=min(as,ds[nx][ny]);
				}
				if(as<1e7)
				ls=min(ls,as*2+i);
			}
		}
		as[i]=ls;
	}
	for(int i=17;i>=1;i--)as[i]=min(as[i],as[i+1]);
	for(int i=2;i<18;i++)if(as[i]>1e8)as[i]=-1;
	for(int i=2;i<18;i++)printf("%d\n",as[i]);
}
```

##### Space Poker 3

###### Problem

有一种牌，这种牌有 $13$ 种数字。

给出 $l$ 种牌型，每种牌型可以被表述为 $(a_1,...,a_l)$。一组牌满足这个牌型当且仅当存在 $l$ 种不同数字 $s_1,.\cdots,s_l$，使得牌中 $s_i$ 至少有 $a_i$ 张。

定义一个牌的分数为它能满足的牌型的最大编号，不存在为 $0$。

现在有一个游戏，游戏规则为有 $n$ 个人，每个人随机获得 $m$ 张牌，接着随机给出 $k$ 张牌，每个人的分数为他自己的牌加上给出的 $k$ 张牌后，这些牌的分数。如果分数最大的人只有一个，则这个人获胜，否则平局。

现在给出第一个人的牌以及 $k$ 张牌中的一部分，剩下的牌全部随机，求第一个人获胜的概率。

$n,m\leq 10,k\leq 5,l\leq 100$

$3s,64MB$

###### Sol

牌的分数只和每种牌出现的次数构成的可重集有关。而 $1\sim 15$ 的划分数总和不到 $700$，因此考虑将这个作为状态。可以预处理每种可重集的分数。

可以发现其它人的牌分数只会被 $k$ 张牌的出现次数构成的可重集影响。考虑对这 $k$ 张牌进行 $dp$，设 $dp_{i,s,t}$ 表示考虑了前 $i$ 种数字，当前第一个人以及这 $k$ 张牌前面的牌的出现次数的集合为 $s$，这 $k$ 张牌前面出现的部分的出现次数集合为 $t$，前面的概率。这样即可求出这部分的情况。

考虑枚举这 $k$ 张牌出现次数的可重集计算答案，只需要求出对于这种可重集，一个牌全部随机的人分数是每种值的概率。这可以使用类似的 $dp$ 计算。一种做法是直接使用上面的 $dp$，但第一个人的牌变为随机给出。因为对于一种可重集，所有不同数字的情况等价，因此不难得到可重集为 $s$ 时，一个牌全部随机的人的牌的出现次数集合为 $t$ 的方案数为 $\frac{dp_{13,t,s}}{\sum_i dp_{13,i,s}}$。在可重集和第一个人的牌的可重集确定后，第一个人获胜的概率为他赢后面一个人的概率的 $n-1$ 次方。

因此再做一次上面的 $dp$，最后即可得到答案。

复杂度为 $O(c*(s(m+k)*s(k)+s(m+k)*l))$，其中 $c$ 为数字种数，$s(n)$ 为 $1\sim n$ 的划分数总和。可以发现 $s(15)<700,s(5)<20$。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
int n,m,k,d,s1,s2,a,ct,lc;
int st[65601],rs[702],trs[702][16],c1[14],c2[14],pr[702],sz[702];
int t1[17],t2[17],st1[17];
double dp[14][702][21],f[17][17],fu[202][21],f2[202][21],as;
vector<int> ls[16],fr[702],tp[302];
void dfs(int d,int s)
{
	int tp=0;
	for(int i=d-1;i>=1;i--)tp=(tp*2+1)<<(st1[i]-st1[i-1]);
	ls[s].push_back(tp*2);
	for(int i=st1[d-1];i+s<=m+k;i++)st1[d]=i,dfs(d+1,s+i);
}
void init()
{
	st1[0]=1;dfs(1,0);
	for(int i=0;i<=m+k;i++)for(int j=0;j<ls[i].size();j++)st[ls[i][j]]=++ct,rs[ct]=ls[i][j],lc+=i<=k,sz[ct]=i;
	for(int i=1;i<=ct;i++)trs[i][0]=i;
	for(int i=1;i<=ct;i++)for(int j=1;j<=m+k;j++)
	{
		int id=rs[i],ct=j,v1=0;
		while(1)
		{
			ct-=!((id>>v1)&1);v1++;
			if(!ct)break;
		}
		int nt=(id>>v1<<v1+1)|(1<<v1)|(id&((1<<v1)-1));
		if(nt<=65536)trs[i][j]=st[nt];
	}
	for(int i=1;i<=ct;i++)
	{
		vector<int> f1;
		int tp=rs[i],ct=0;
		while(tp)
		{
			if(tp&1)f1.push_back(ct);
			else ct++;
			tp>>=1;
		}
		reverse(f1.begin(),f1.end());
		fr[i]=f1;
	}
	for(int i=0;i<=m+k;i++)for(int j=0;i+j<=m+k;j++){f[i][j]=1;for(int l=1;l<=j;l++)f[i][j]*=1.0*(i+l)/l/13;}
}
int doit(char c){for(int i=1;i<=13;i++)if(c=="23456789TJQKA"[i-1])return i;}
void solve()
{
	for(int i=0;i<=13;i++)for(int j=1;j<=ct;j++)for(int k=1;k<=lc;k++)dp[i][j][k]=0;
	dp[0][1][1]=1;
	int sa=0,sb=0;
	for(int i=0;i<13;i++,sa+=t1[i],sb+=t2[i])for(int j=1;j<=ct;j++)for(int l=1;l<=lc;l++)if(dp[i][j][l]>1e-20)
	{
		int s1=sz[j]-sz[l],s2=sz[l];
		int sa1=t1[i+1],sa2=t2[i+1];
		for(int p=0;p<=m-s1-sa1;p++)for(int q=0;q<=k-s2-sa2;q++)
		{
			dp[i+1][trs[j][p+q+sa1+sa2]][trs[l][q+sa2]]+=dp[i][j][l]*f[s1-sa][p]*f[s2-sb][q];
		}
	}
	for(int j=1;j<=ct;j++)for(int l=1;l<=lc;l++)if(sz[j]==m+k&&sz[l]==k)
	fu[pr[j]][l]+=dp[13][j][l];
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	init();
	char st=getchar();
	while(st=='\n')st=getchar();
	for(int i=1;i<=m;i++)t1[doit(st)]++,st=getchar();
	while(1)
	{
		st=getchar();
		if(st=='\n')break;
		t2[doit(st)]++;d++;
	}
	scanf("%d",&s1);
	for(int i=1;i<=s1;i++)
	{
		scanf("%d",&s2);
		while(s2--)scanf("%d",&a),tp[i].push_back(a);
		sort(tp[i].begin(),tp[i].end());reverse(tp[i].begin(),tp[i].end());
	}
	for(int i=s1;i>=1;i--)
	for(int j=1;j<=ct;j++)if(!pr[j])
	{
		int sz1=tp[i].size(),sz2=fr[j].size();
		if(sz1>sz2)continue;
		int fg=1;
		for(int l=0;l<sz1;l++)if(tp[i][l]>fr[j][l]){fg=0;break;}
		if(fg)pr[j]=i;
	}
	solve();
	for(int j=0;j<=s1;j++)for(int l=1;l<=lc;l++)f2[j][l]=fu[j][l],fu[j][l]=0;
	for(int i=0;i<=13;i++)t1[i]=t2[i]=0;
	solve();
	for(int l=1;l<=lc;l++)if(sz[l]==k)
	{
		double su=0;
		for(int j=0;j<=s1;j++)su+=fu[j][l];
		for(int j=0;j<=s1;j++)fu[j][l]/=su;
		for(int j=1;j<=s1;j++)fu[j][l]+=fu[j-1][l];
		for(int j=1;j<=s1;j++)if(f2[j][l])
		{
			double v1=fu[j-1][l],v2=1;
			for(int s=1;s<n;s++)v2*=v1;
			as+=v2*f2[j][l];
		}
	}
	printf("%.10lf\n",as);
}
```



