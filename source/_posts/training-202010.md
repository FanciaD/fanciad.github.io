---
title: 2020/10 集训题解
date: '2021-02-22 22:23:04'
updated: '2021-02-22 22:23:04'
tags: Mildia
permalink: TheDieisCast/
description: 2020/10 南京集训
mathjax: true
---

Note. 本篇的完成时间已不可考，此处时间为上一个 blog 建立的时间。

##### CF1408I Bitwise Magic

###### Problem

给 $n$ 个数 $a_{1,...,n}\in [k,2^c-1]$ ，进行 $k$ 次操作，每次随机选择一个数减一，求操作结束后所有数xor的值为 $[0,...,2^c-1]$ 的概率

$n\leq 2^c-k,k,c\leq 16$

$6s,512MB$

###### Sol

设 $a_i=s_i*16+t_i$ ，则 $a_i-t=s_i*16+(t_i-t)或(s_i-1)*16+(16+t_i-t)(t\leq k)$

考虑 $(a_i-t)\oplus a_i$ ，如果 $t_i\geq t$ ，则显然 $(a_i-t)\oplus a_i<16$ ，否则设 $lowbit(s_i)=2^{v_i}$ ,$(a_i-t)\oplus a_i =(t_1\oplus(16+t_i-t))|(16(s_i\oplus (s_i-1)))=(t_1\oplus(16+t_i-t))|(16(2^{v_i+1}-1))$

枚举 $v_i,t_i$ ，对于 $v_i,t_i$ 相同的数，所有的 $(a_i-t)\oplus a_i$ 一定是相同的

对于每一类数，可以暴力算出这一类内操作 $0,...,k$ 次时，异或值为 $0,...,2^c-1$ 的概率

对于同一类数的值，第 $4,...,v_i$ 位一定相同，所以可能的异或种类数只有 $2*16$ 种

对于同一个 $v_i$ ，可能的异或值是相同的，因此可以暴力合并，对于每一类 $v_i$ 求出这一类中操作 $0,...,k$ 次，异或值为 $0,...,2^c-1$ 的概率，这一部分复杂度为 $O(ck^5)$

然后对于不同的类，暴力fwt+背包合并即可

复杂度 $O(ck^22^c+ck^5)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 65601
#define M 17
#define mod 998244353
int n,c,k,t,v[N],ct[M][M],tp[M][M*2],st[M][M*2],f2[M][M*2],fu[M][N],s1[M][N],s3[M][N],vl[M],fr[N],ifr[N],st1[N],sc[N],as[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void doit(int n,int su,int tp,int v,int s)
{
	if(s<n)return;
	int v1=tp&((1<<t)-1),f1=1;tp^=v1;
	if(tp)tp=(tp^v1)^((1<<v+t)-1),v1=tp&((1<<t)-1),v1+=(1<<t);
	int s1=1;
	for(int i=2;i<=n;i++)if(st1[i]!=st1[i-1])f1=1ll*f1*fr[s]%mod*ifr[s1]%mod*ifr[s-s1]%mod,s-=s1,s1=1;
	else s1++;
	for(int i=1;i<=n;i++)f1=1ll*f1*ifr[st1[i]]%mod;
	if(n)f1=1ll*f1*fr[s]%mod*ifr[s1]%mod*ifr[s-s1]%mod;
	st[su][v1]=(st[su][v1]+f1)%mod;
}
void dfs(int n,int ls,int su,int tp,int v,int s)
{
	doit(n-1,su,tp,v,s);
	for(int i=ls;su+i<=k;i++)st1[n]=i,dfs(n+1,i,su+i,tp^vl[i],v,s);
}
int main()
{
	scanf("%d%d%d",&n,&k,&c);
	fr[0]=ifr[0]=1;for(int i=1;i<=65555;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	while((1<<t)<k)t++;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)
	{
		int t1=v[i]&((1<<t)-1),t2=v[i]>>t,t3=0;
		t2=t2&-t2;
		while(t2)t2>>=1,t3++;
		ct[t3][t1]++;
	}
	for(int i=0;i<1<<c;i++)fu[0][i]=1;
	for(int i=1;i<1<<c;i++)sc[i]=sc[i-(i&-i)]+1;
	for(int i=0;i<=c-t+1;i++)
	{
		memset(tp,0,sizeof(tp));
		tp[0][0]=1;
		for(int j=0;j<(1<<t);j++)if(ct[i][j])
		{
			memset(st,0,sizeof(st));
			memset(f2,0,sizeof(f2));
			for(int l=1;l<=k;l++)vl[l]=((1<<i+t-1)|j)^(((1<<i+t-1)|j)-l);
			dfs(1,1,0,0,i,ct[i][j]);
			for(int v1=0;v1<=k;v1++)
			for(int v2=0;v2<=k;v2++)
			if(v1+v2<=k)
			for(int c1=0;c1<(1<<t+1);c1++)
			for(int c2=0;c2<(1<<t+1);c2++)
			f2[v1+v2][c1^c2]=(f2[v1+v2][c1^c2]+1ll*tp[v1][c1]*st[v2][c2])%mod;
			memcpy(tp,f2,sizeof(f2));
		}
		memset(s1,0,sizeof(s1));
		memset(s3,0,sizeof(s3));
		for(int j=0;j<=k;j++)
		for(int l=0;l<1<<(t+1);l++)
		{
			int tp1=l&((1<<t)-1),v2=l^tp1;
			if(v2)tp1^=(1<<i+t)-1;
			for(int s=0;s<1<<c;s++)
			{
				int s2=1ll*tp[j][l]*(sc[tp1&s]&1?mod-1:1)%mod;
				s1[j][s]=(s1[j][s]+s2)%mod;
			}
		}
		for(int j=0;j<=k;j++)
		for(int l=0;l<=k;l++)
		if(j+l<=k)
		for(int s=0;s<1<<c;s++)
		s3[j+l][s]=(s3[j+l][s]+1ll*s1[j][s]*fu[l][s])%mod;
		memcpy(fu,s3,sizeof(s3));
	}
	for(int i=0;i<1<<c;i++)as[i]=fu[k][i];
	for(int i=2;i<=1<<c;i<<=1)
	for(int j=0;j<1<<c;j+=i)
	for(int l=j;l<j+(i>>1);l++)
	{
		int v1=(as[l]+as[l+(i>>1)])%mod,v2=(as[l]-as[l+(i>>1)]+mod)%mod;
		v1=1ll*v1*(mod+1)/2%mod,v2=1ll*v2*(mod+1)/2%mod;
		as[l]=v1;as[l+(i>>1)]=v2;
	}
	int t1=0;for(int i=1;i<=n;i++)t1^=v[i];
	for(int i=0;i<1<<c;i++)printf("%d ",1ll*fr[k]*as[i^t1]%mod*pw(pw(n,mod-2),k)%mod);
}
```

##### CF1404H Rainbow Triples

###### Problem

给定一个序列 $v$ ，你需要找到若干个三元组 $(a_i,b_i,c_i)$ ，满足

1. $v_{a_i},v_{c_i}=0,v_{b_i}>0$
2. 所有 $a_i,b_i,c_i$ 两两不同
3. 所有 $v_{b_i}$ 两两不同

求出最大的三元组数量

$n\leq 5\times 10^5$

$2s,256MB$

###### Sol

考虑二分答案，如果当前答案为 $k$ ，显然 $a_i$ 应该用前 $k$ 个0， $c_i$ 应该用后 $k$ 个0

问题相当于在每个区间中选出一个数，使得每种颜色被选不超过一次

由Hall定理，有解当且仅当对于任意 $i$ 个区间，区间的并中颜色数大于等于 $i$ 

相当于对于任意的 $i\leq j$ ，第 $i$ 个区间的左端点到第 $j$ 个区间的右端点中颜色数大于 $j-i+1$

从左向右考虑，对于每一个 $l$ 维护 $[l,x]$ 的颜色数，那么 $x=c_i$ 时需要满足 $\forall j\leq i,[a_j,c_i]$ 中颜色数大于等于 $i-j+1$ 

因为颜色数不增，可以看成对 $[1,...,a_1],[1,...,a_2],...,[1,...,a_i]$ 分别减一之后满足所有位置的值大于等于0

因此可以在维护颜色数的基础上，如果当前 $x=c_i$ 则将 $[1,...,a_i]$ 区间减一，再判断是否合法

显然 $a_{1,...,k}$ 与 $k$ 无关，若答案为 $k$ 时 $[1,...,i]$ 中有 $a$ 个 $c_j$ ，则答案为 $k-1$ 时 $[1,...,i]$ 中有 $a-1$ 个 $c_j$ ，因此答案减一时相当于将最后一个 $[1,...,a_i]$ 的区间减一的操作撤销

因此在不合法的时候将答案减一，然后继续判断即可

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 500500
int t,n,v[N],s1[N],ls[N],as,s2[N],s3[N],ct,tp;
struct edge{int l,r,mn,lz;}e[N*4];
void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn)+e[x].lz;}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;e[x].mn=e[x].lz=0;
	if(l==r)return;
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void modify(int x,int l,int r,int v)
{
	if(e[x].l==l&&e[x].r==r){e[x].mn+=v;e[x].lz+=v;return;}
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)modify(x<<1,l,r,v);
	else if(mid<l)modify(x<<1|1,l,r,v);
	else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);
	pushup(x);
}
int main()
{
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d",&n);as=ct=tp=0;
		for(int i=1;i<=n;i++)scanf("%d",&v[i]),ct+=!v[i];
		as=ct/2;
		for(int i=1;i<=n;i++)s3[i]=s2[i]=ls[i]=0;
		int tp=as;
		for(int i=1;i<=n&&tp;i++)if(!v[i])s3[as-tp+1]=i,tp--;
		tp=as;
		for(int i=n;i>=1&&tp;i--)if(!v[i])s2[i]=tp,tp--;
		build(1,1,n);
		for(int i=1;i<=n;i++)
		if(v[i])modify(1,ls[v[i]]+1,i,1),ls[v[i]]=i;
		else if(s2[i])
		{
			modify(1,1,s3[s2[i]-tp],-1);
			if(e[1].mn<0)modify(1,1,s3[s2[i]-tp],1),tp++;
		}
		printf("%d\n",as-tp);
	}
}
```

##### CF1408G Clusterization Counting

###### Problem

给定 $n$ 个点以及 $v_{i,j}(v_{i,j}=v_{j,i})$ ，所有 $v$ 两两不同

你需要将点划分成若干个集合，使得每个集合 $S$ 满足

对于 $x,y,z\in S,t\not\in S,v_{i,j}<v_{z,t}(x\neq y)$

对于每一个 $k$ ，求出划分出 $k$ 个集合的方案数 ，模 $998244353$

###### Sol

考虑两个合法的集合 $A,B$

如果存在 $a,b,c$ ，$a,b\in A,c\not\in A,a,c\in B,b\not\in B$ ，那么 $v_{a,b}<v_{a,c},v_{a,b}>v_{a,c}$ ，矛盾

因此任意两个合法集合满足不相交或者一个包含另外一个

因此所有集合的包含关系构成一棵树

因为只有 $n$ 个叶子，所以合法集合只有 $O(n)$ 个

考虑找出所有合法的集合，考虑一个点 $u$ ，将剩下的点按照 $v_{u,i}$ 排序，则包含 $u$ 的合法集合只可能是 $u$ 加上一段前缀

枚举集合大小 $d$ ，可能的集合只有 $n$ 个，可以求出每一个这样集合的hash值，只考虑出现了不少于 $d$ 次的hash值，然后暴力判断集合是否合法即可，如果一个点所在的集合被判断过了就可以跳过

复杂度 $O(n^2\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
using namespace std;
#define N 1550
#define mod 998244353
#define ul unsigned int
int n,v[N][N],tp[N],st[N],s1[N][N],ls[N],fu[N][N],is[N*2],dp[N*2][N],ct,head[N*2],cnt,sz[N*2];
ul vl[N],su[N][N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void dfs(int u)
{
	if(u>n)dp[u][0]=1;
	for(int i=head[u];i;i=ed[i].next)
	{
		dfs(ed[i].t);
		for(int j=0;j<=sz[u]+sz[ed[i].t];j++)tp[j]=0;
		for(int j=0;j<=sz[u];j++)
		for(int k=0;k<=sz[ed[i].t];k++)
		tp[j+k]=(tp[j+k]+1ll*dp[u][j]*dp[ed[i].t][k])%mod;
		sz[u]+=sz[ed[i].t];
		for(int j=0;j<=sz[u];j++)dp[u][j]=tp[j];
	}
	dp[u][1]=1;
}
bool cmp(int a,int b){return tp[a]<tp[b];}
map<ul,int> mp[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	scanf("%d",&v[i][j]);
	vl[0]=1;for(int i=1;i<=n;i++)vl[i]=19260817*vl[i-1]+998244353;
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)st[j]=j,tp[j]=v[i][j];
		sort(st+1,st+n+1,cmp);sort(v[i]+1,v[i]+n+1);
		for(int j=1;j<=n;j++)s1[i][j]=st[j],su[i][j]=su[i][j-1]+vl[st[j]],mp[j][su[i][j]]++;
	}
	ct=n;for(int i=1;i<=n;i++)sz[i]=1,ls[i]=i;
	for(int i=2;i<=n;i++)
	for(int j=1;j<=n;j++)
	if(!fu[i][j]&&mp[i][su[j][i]]>=i)
	{
		int fg=1;
		for(int t=1;t<=i;t++)fu[i][s1[j][t]]=1;
		for(int t=1;t<=i;t++)if(su[s1[j][t]][i]!=su[j][i])fg=0;
		int mx=0;
		for(int t=1;t<=i;t++)if(mx<v[s1[j][t]][i])mx=v[s1[j][t]][i];
		if(i<n)for(int t=1;t<=i;t++)if(mx>v[s1[j][t]][i+1])fg=0;
		if(!fg)continue;
		++ct;
		for(int t=1;t<=i;t++)if(!is[ls[s1[j][t]]])is[ls[s1[j][t]]]=1,adde(ct,ls[s1[j][t]]);
		for(int t=1;t<=i;t++)ls[s1[j][t]]=ct;
	}
	dfs(ct);
	for(int i=1;i<=n;i++)printf("%d ",dp[ct][i]);
}
```

##### Topcoder SRM533 Pikachu

###### Problem

有三种字符，长度分别为2,2,3

有 $n$ 种单词，第 $i$ 个单词会使用 $v_i$ 次

你需要给每个单词分配一个使用这三种字符的编码，满足没有一个编码是另外一个的前缀

定义一个单词的长度为编码中所有字符的长度和

设单词 $i$ 的长度为 $l_i$ ，求合法的编码方式中 $\sum v_il_i$ 的最小值并求出有多少种编码方式能够达到最小值，方案数对 $10^9+9$ 取模

$n\leq 30$

$2s,64MB$

###### Sol

考虑一个当前没有被匹配的前缀 $s$ ，长度为 $l_s$ 

如果它对应了一个单词，那么之后不会再出现以 $s$ 开头的单词

否则，可以看成删除这个前缀，再加入三个 $s+A,s+B,s+C$ 的前缀，然后继续进行匹配

那么可以看成初始只有一个长度为 $0$ 的后缀，所有单词都没有被对应，然后进行上面的操作直到所有单词都被对应

考虑每次拿出长度最小的 $s$ 进行操作，那么每次拿出的 $s$ 长度一定不降，因此每一次对应的一定是使用次数最多的单词

同时，显然当前没有被匹配的前缀长度差不会超过3

设 $dp_{a,b,c,k},f_{a,b,c,k}$ 表示当前有 $a$ 个长度为0的后缀， $b$ 个长度为1的后缀，$c$ 个长度为2的后缀，当前前 $k$ 个单词被对应了，此时的最小代价和方案数

如果 $a+k\geq n$ ，那么最小代价一定是0，可以直接算方案数

否则考虑枚举 $a$ 个中有多少个对应单词，若有 $i$ 个，则此时剩下 $b$ 个长度为1的，$c+2(a-i)$ 个长度为2的，$a-i$ 个长度为3的

考虑这一步的方案数，选出前缀的方案数为 $C_a^i$ ，选出单词的方案数相当于在 $v_{k,...,n}$ 中选 $i$ 个最小数的方案数，可以对于每一对 $(k,i)$ 预处理，然后配对的方案数为 $i!$

考虑将剩下的前缀长度全部减一，剩下的前缀长度就变为了0,1,2，然后再额外加上 $\sum_{j>i+k}v_i$ 的代价即可

复杂度 $O(n^5)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
#include<vector>
using namespace std;
#define N 35
#define mod 1000000009
int n,v[N],dp[N][N*3][N][N],f[N][N*3][N][N],st[N][N],fr[N*10],ifr[N*10],su[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int getst(int l,int r)
{
	int as=fr[r-l+1],f1=l,f2=l;
	while(1)
	{
		while(f2+1<=n&&v[f2+1]==v[f1])f2++;
		if(f2<r)f1=f2+1;
		else return 1ll*as*fr[f2-f1+1]%mod*ifr[r-f1+1]%mod*ifr[f2-r]%mod;
	}
}
pair<int,int> dfs(int a,int b,int c,int s)
{
	if(s+a>=n)return make_pair(0,1ll*fr[a]*ifr[a-(n-s)]%mod);
	if(!a&&!b&&!c)return make_pair(1e8,0);
	if(f[a][b][c][s]!=-1)return make_pair(f[a][b][c][s],dp[a][b][c][s]);
	int mn=1e9,st1=0;
	for(int i=0;i<=a;i++)
	{
		int t1=b,t2=c+(a-i)*2,t3=(a-i),ns=s+i,s3=1ll*fr[a]*ifr[a-i]%mod*ifr[i]%mod*st[s+1][s+i]%mod;
		pair<int,int> s1=dfs(t1,t2,t3,ns);
		int tp=su[ns+1]+s1.first,s2=1ll*s3*s1.second%mod;
		if(tp<mn)mn=tp,st1=s2;
		else if(tp==mn)st1=(st1+s2)%mod;
	}
	f[a][b][c][s]=mn,dp[a][b][c][s]=st1;
	return make_pair(mn,st1);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),v[i]*=-1;
	sort(v+1,v+n+1);
	for(int i=1;i<=n;i++)v[i]*=-1;
	fr[0]=ifr[0]=1;for(int i=1;i<=n*9;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	memset(f,-1,sizeof(f));
	for(int i=1;i<=n;i++)for(int j=i;j<=n;j++)st[i][j]=getst(i,j);
	for(int i=1;i<=n;i++)st[i][i-1]=1;
	for(int i=n;i>=1;i--)su[i]=su[i+1]+v[i];
	pair<int,int> as=dfs(1,0,0,0);
	printf("%d %d\n",as.first,as.second);
}
```

##### JAG Spring Contest 2015 H Kimagure Cleaner

###### Problem

有 $n$ 条指令，每条指令由字符 $s\in \{L,R\}$ 和一个数字 $v$ 组成

一个机器人初始在原点，面向x轴正方向，它会依次执行这 $n$ 条指令

执行第 $i$ 条指令时，如果 $s_i=L$ ，则它会向左旋转90度，否则会向右旋转90度，然后它会向前前进 $v_i$ 个单位长度

给出对于这 $n$ 条指令的限制 $t_i\in\{L,R,?\},l_i,r_i$ ，如果 $t_i=L$ ，则要求 $s_i=L$ ，如果 $t_i=R$ ，则要求 $s_i=R$ ，同时要求 $l_i\leq v_i\leq r_i$

求是否存在一组满足要求的指令，使得执行完指令后机器人在 $(x,y)$ ，输出无解或任意一组合法的解

$n\leq 50,l_i\leq r_i\leq 10^9$

$10s,1024MB$

###### Sol

设问号个数为 $t$

一种暴力做法是直接枚举所有问号处取值，之后选取 $v_i$ 可以直接解决

考虑折半搜索，处理出前一半之后可能的位置，问号取值固定后是一个矩形，所以得到的是若干矩形的并，对于后一半也可以处理出开头合法的位置

然后可以直接扫描线求出前一半后一个合法的位置，然后再搜一次还原解

这样的复杂度为 $O(n*2^{\frac t2})$

考虑另外一种做法，如果 $t_i=t_{i+1}=?$ ，则 $t_i$ 这一步的方向不会对前后的方向造成任何影响，因此可以对所有这样的单独考虑

考虑暴力枚举剩下的问号的取值，可以发现这样的问号数不超过 $min(t,n-t)$ 

然后可以先枚举这些问号的取值，然后对于剩下的移动两维独立，因此可以分别折半搜索，复杂度 $2^{min(t,n-t)+\frac n4}$

也可以先处理出两维独立部分分别合法的区间，然后 $O(\log 2^{n/2})$ 判定，复杂度为 $O(2^{min(t,n-t)}*(1+n-t-min(t,n-t)))$ ，复杂度不会超过 $O(2^{\frac n2})$ 

考虑合并两类暴力，若 $t\leq \frac {3n}4$ 使用第一种，否则使用第二种，复杂度 $O(2^{\frac{3n}8}*n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<set>
#include<algorithm>
using namespace std;
#define N 53
#define ll long long
int s[N][3],f1[N],f2[N],f3[N],v[N],n,c1,fg;
char st1[3];
ll lx,ly;
vector<pair<ll,ll> > s1,s2;
multiset<pair<ll,ll> > st;
vector<ll> t1,t2;
void dfs(int x,int t,ll v1,ll v3,int f)
{
	if(x==t+1){s1.push_back(make_pair(v1,v3));return;}
	int fg=0;
	if(!s[x][0])fg=5<<(x&1);
	else if(s[x][0]==1)fg=1<<((f+1)&3);
	else fg=1<<((f+3)&3);
	for(int i=0;i<4;i++)if(fg&(1<<i))
	{
		int v11=v1,v13=v3;
		if(i==0)v11+=s[x][1];
		if(i==1)v13+=s[x][1];
		if(i==2)v11-=s[x][2];
		if(i==3)v13-=s[x][2];
		dfs(x+1,t,v11,v13,i);
	}
}
void dfs2(int x,int s1,int t,ll v1,ll v3,int f)
{
	if(x==t+1){if(!v1&&!v3)for(int i=s1;i<=t;i++)f2[i]=f1[i];return;}
	int fg=0;
	if(!s[x][0])fg=5<<(x&1);
	else if(s[x][0]==1)fg=1<<((f+1)&3);
	else fg=1<<((f+3)&3);
	for(int i=0;i<4;i++)if(fg&(1<<i))
	{
		f1[x]=i;
		int v11=v1,v13=v3;
		if(i==0)v11-=s[x][1];
		if(i==1)v13-=s[x][1];
		if(i==2)v11+=s[x][2];
		if(i==3)v13+=s[x][2];
		dfs2(x+1,s1,t,v11,v13,i);
	}
}
void dfsx(int x,int t,ll l)
{
	if(x>t){t1.push_back(l);return;}
	if(f1[x]!=2)dfsx(x+2,t,l+s[x][1]);
	if(f1[x]!=0)dfsx(x+2,t,l-s[x][2]);
}
void dfsx2(int x,int s1,int t,ll l)
{
	if(x>t){if(!l)for(int i=s1;i<=t;i+=2)f2[i]=f3[i];return;}
	if(f1[x]!=2)f3[x]=0,dfsx2(x+2,s1,t,l-s[x][1]);
	if(f1[x]!=0)f3[x]=2,dfsx2(x+2,s1,t,l+s[x][2]);
}
void dfsy(int x,int t,ll l)
{
	if(x>t){t1.push_back(l);return;}
	if(f1[x]!=3)dfsy(x+2,t,l+s[x][1]);
	if(f1[x]!=1)dfsy(x+2,t,l-s[x][2]);
}
void dfsy2(int x,int s1,int t,ll l)
{
	if(x>t){if(!l)for(int i=s1;i<=t;i+=2)f2[i]=f3[i];return;}
	if(f1[x]!=3)f3[x]=1,dfsy2(x+2,s1,t,l-s[x][1]);
	if(f1[x]!=1)f3[x]=3,dfsy2(x+2,s1,t,l+s[x][2]);
}
bool checkx()
{
	t1.clear();t2.clear();
	int st=n/2+1;if(st&1)st++;
	dfsx(2,n/2,0);t2=t1;t1.clear();
	dfsx(st,n,0);
	sort(t1.begin(),t1.end());
	ll su=0;
	for(int i=2;i<=n;i+=2)su+=s[i][2]-s[i][1];
	int as1=-1,as2=-1;
	for(int i=0;i<t2.size();i++)
	{
		ll lb=lx-t2[i]-su,rb=lx-t2[i];
		if(lower_bound(t1.begin(),t1.end(),rb+1)-lower_bound(t1.begin(),t1.end(),lb)==0)continue;
		as1=lower_bound(t1.begin(),t1.end(),lb)-t1.begin();as2=i;
	}
	if(as1==-1)return 0;
	dfsx2(2,2,n/2,t2[as2]);
	dfsx2(st,st,n,t1[as1]);
	return 1;
}
bool checky()
{
	t1.clear();t2.clear();
	int st=n/2+1;if(~st&1)st++;
	dfsy(1,n/2,0);t2=t1;t1.clear();
	dfsy(st,n,0);
	sort(t1.begin(),t1.end());
	ll su=0;
	for(int i=1;i<=n;i+=2)su+=s[i][2]-s[i][1];
	int as1=-1,as2=-1;
	for(int i=0;i<t2.size();i++)
	{
		ll lb=ly-t2[i]-su,rb=ly-t2[i];
		if(lower_bound(t1.begin(),t1.end(),rb+1)-lower_bound(t1.begin(),t1.end(),lb)==0)continue;
		as1=lower_bound(t1.begin(),t1.end(),lb)-t1.begin();as2=i;
	}
	if(as1==-1)return 0;
	dfsy2(1,1,n/2,t2[as2]);
	dfsy2(st,st,n,t1[as1]);
	return 1;
}
void dfs3(int x)
{
	if(fg)return;
	if(x==n+1){if(checkx()&&checky())fg=1;return;}
	if(s[x][0]||s[x+1][0])
	{
		int f=f1[x-1],fg=0;
		if(!s[x][0])fg=5<<(x&1);
		else if(s[x][0]==1)fg=1<<((f+1)&3);
		else fg=1<<((f+3)&3);
		for(int i=0;i<4;i++)if(fg&(1<<i))f1[x]=i,dfs3(x+1);
	}
	else f1[x]=-1,dfs3(x+1);
}
int main()
{
	scanf("%d%lld%lld",&n,&lx,&ly);
	for(int i=1;i<=n;i++)
	{
		scanf("%s%d%d",st1+1,&s[i][1],&s[i][2]);
		if(st1[1]=='L')s[i][0]=1;
		else if(st1[1]=='R')s[i][0]=2;
		else c1++;
	}
	if(c1<=36)
	{
		int tp=0,st1=1;
		for(int i=1;i<=n;i++)
		{
			if(!s[i][0])tp++;
			if(tp<=c1/2)st1=i;
		}
		dfs(1,st1,0,0,0);
		s2=s1;s1.clear();
		dfs(st1+1,n,0,0,0);
		ll fx=0,fy=0,fx1,fy1;
		for(int i=1;i<=n;i++)if(i&1)fy+=s[i][2]-s[i][1];else fx+=s[i][2]-s[i][1];
		for(int i=0;i<s1.size();i++)s1[i].first=lx-s1[i].first,s1[i].second=ly-s1[i].second;
		swap(s1,s2);
		sort(s1.begin(),s1.end());sort(s2.begin(),s2.end());
		int l1=0,l2=0,l3=0,as1=-1,as2;
		while(l3<s2.size()&&l2<s1.size())
		{
			ll v1=l1==s1.size()?1e16:s1[l1].first,v2=s1[l2].first+fx,v3=s2[l3].first;
			if(v1<=v3&&v1<=v2)st.insert(make_pair(s1[l1].second,l1)),l1++;
			else if(v3<=v2)
			{
				multiset<pair<ll,ll> >::iterator it1=st.lower_bound(make_pair(s2[l3].second-fy,0)),it2=st.lower_bound(make_pair(s2[l3].second+1,0));
				l3++;
				if(it1==it2)continue;
				as1=(*it1).second,as2=l3-1;
				break;
			}
			else st.erase(make_pair(s1[l2].second,l2)),l2++;
		}
		if(as1==-1){printf("-1\n");return 0;}
		fg=1;
		dfs2(1,1,st1,s1[as1].first,s1[as1].second,0);
		dfs2(st1+1,st1+1,n,lx-s2[as2].first,ly-s2[as2].second,0);
	}
	else dfs3(1);
	if(!fg){printf("-1\n");return 0;}
	for(int i=1;i<=n;i++)
	{
		if(f2[i]==0)lx-=s[i][1],v[i]=s[i][1];
		if(f2[i]==1)ly-=s[i][1],v[i]=s[i][1];
		if(f2[i]==2)lx+=s[i][2],v[i]=-s[i][2];
		if(f2[i]==3)ly+=s[i][2],v[i]=-s[i][2];
	}
	for(int i=1;i<=n;i++)
	{
		ll tp;
		if(f2[i]&1)tp=min(ly,1ll*s[i][2]-s[i][1]),v[i]+=tp,ly-=tp;
		else tp=min(lx,1ll*s[i][2]-s[i][1]),v[i]+=tp,lx-=tp;
	}
	printf("%d\n",n);
	for(int i=1;i<=n;i++)
	{
		if(f2[i]==(f2[i-1]+1)%4)printf("L ");else printf("R ");
		printf("%d\n",v[i]<0?-v[i]:v[i]);
	}
}
```

##### JAG Spring Contest 2015 J New Game AI

###### Problem

给 $n$ 个人，每个人有两种属性 $a_i,b_i$

人 $i$ 能够替代人 $j$ 当且仅当 $|a_i-a_j|>c,a_i<a_j$ 或者 $|a_i-a_j|\leq c,b_i<b_j$

初始选择序列第一个人，依次考虑所有人，如果当前的人能够替代选择的人，则选择当前的人

求出有多少个人满足存在一种人的排列，使得这个人最后被选择

$n\leq 5\times 10^4$

$2s,256MB$

###### Sol

考虑当前 $a_i$ 最小的点 $k$ ，所有 $a_i>k+c$ 的点都可以被这个点替代

再考虑 $a_i\leq k+c$ 的点中 $b_i$ 最小的点 $s$ (多个取 $a_i$ 最小的)，$a_i\leq k+c$ 的点都不能替代它，因此可以先选择 $k$ ，然后处理所有 $a_i>k+c$ 的，接着选择 $s$ 再处理其它点，因此这个点可以最后被选择

对于点 $t$ ，如果存在序列 $x_1,...,x_t$ ，使得 $x_1$ 能够替代 $s$ ，$x_2$ 能够替代 $x_1$ ,..., $t$ 能够替代 $x_t$ ，那么在上面的操作中留下后面的点，然后只剩 $s$ 后再沿着这个序列操作过去，即可做到最后留下 $t$

因此从 $s$ 出发，每次从一个点走到能替代它的点，所有能到达的点都可以留下，遍历的过程可以线段树维护

然后考虑剩下的点的集合 $S$ ，设这次选的集合为 $T$ ，则 $S$ 中点不能替代任意一个 $T$ 中点

如果 $S$ 中一个点能留下，则 $T$ 中任意一个点都不能替代它

这种情况会出现当且仅当 $T$ 中点的 $b_i$ 相同，$S$ 中这个点的 $b_i$ 也与它相同且这个点的 $a_i$ 和 $T$ 中点的 $a_i$ 相差不超过 $c$

如果不存在这样的点，则 $S$ 中点不能留到最后

如果存在这样的点 ，记 $a_i$ 最大的点为 $w$，只考虑 $S$ 进行上面的操作，考虑选出的 $T^{'}$ ，如果 $w\not\in T^{'}$ ，那么 $T^{'}$ 中点都存在一种以 $w$ 开始最后选择自己结束的方式，这时先在 $w$ 后考虑 $T$ 中的点即可让 $T^{'}$ 中点获胜

如果 $w\in T^{'}$ ，考虑之前的过程中选出的 $a_i$ 最小的点 $k$ 和 $a_i\leq k+c$ 的点中 $b_i$ 最小的点 $s$，如果 $a_w>a_k+c$ ，直接先选择 $w$ 再选择 $k$ ，选择 $w$ 时考虑 $T$ 中点即可

如果 $a_w\leq a_k+c,b_w>b_k$ ，构造方式相同

如果 $b_w=b_k$ ，则选择 $k$ 时也可以处理所有 $T$ 中点

如果 $b_w<b_k,b_w>b_s$ ，则可以先选择 $k$ 再选择 $w$ 再选择 $s$

如果 $b_w=b_s$ ，选择 $s$ 时也可以处理所有 $T$ 中点

因此此时 $T^{'}$ 中点都可以获胜，因为 $w$ 是这样的点中 $a_i$ 最大的，所以剩下的点中不存在一个点不会被 $T$ 中点替代，因此剩下的点不可能是答案

考虑一直进行上面的操作，如果某一步之后不存在 $w$ ，则直接终止过程，否则考虑记录每一次的 $w$ ，如果有一个 $w$ 属于某一次选出的 $T$ ，那么剩下的点可以不考虑，否则每一步找出的点一定都满足条件

选出 $k,s$ 的过程可以用线段树维护

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 50050
int n,c,is[N],fg[N],nw=1,tp=1,v1[N],ct;
pair<int,int> st[N];
bool check(int a,int b){if(!a)return 0;if(!b)return 1;return st[a].second==st[b].second?st[a].first>st[b].first:st[a].second<st[b].second;}
bool check2(int a,int b){if(!a)return 0;if(!b)return 1;return st[a].second==st[b].second?st[a].first<st[b].first:st[a].second<st[b].second;}
struct segt{
	struct node{int l,r,mn,mx;}e[N*4];
	void pushup(int x){e[x].mn=check(e[x<<1].mn,e[x<<1|1].mn)?e[x<<1].mn:e[x<<1|1].mn;e[x].mx=check2(e[x<<1].mx,e[x<<1|1].mx)?e[x<<1].mx:e[x<<1|1].mx;}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].mn=e[x].mx=l;return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int v)
	{
		if(e[x].l==e[x].r){e[x].mn=e[x].mx=0;return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=v)modify(x<<1,v);
		else modify(x<<1|1,v);
		pushup(x);
	}
	int query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].mn;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);
		else if(mid<l)return query(x<<1|1,l,r);
		else
		{
			int v1=query(x<<1,l,mid),v2=query(x<<1|1,mid+1,r);
			if(!v1||!v2)return v1+v2;
			return check(v1,v2)?v1:v2;
		}
	}
	int query2(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].mx;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query2(x<<1,l,r);
		else if(mid<l)return query2(x<<1|1,l,r);
		else
		{
			int v1=query2(x<<1,l,mid),v2=query2(x<<1|1,mid+1,r);
			if(!v1||!v2)return v1+v2;
			return check2(v1,v2)?v1:v2;
		}
	}
}tr;
int query1(int v)
{
	int tp=lower_bound(st+1,st+n+1,make_pair(v+1,0))-st-1;
	return tr.query(1,1,tp);
}
int query2(int v)
{
	int tp=lower_bound(st+1,st+n+1,make_pair(v+1,0))-st-1;
	return tr.query2(1,1,tp);
}
void doit()
{
	while(is[nw])nw++;
	if(nw>n){tp=0;return;}
	ct=0;
	int tp1=query2(st[nw].first+c);
	if(st[tp1].second==st[nw].second)tp1=nw;
	is[tp1]=1,v1[++ct]=tp1,tr.modify(1,tp1);
	for(int i=1;i<=ct;i++)
	{
		if(fg[v1[i]])tp=0;
		while(is[nw])nw++;
		while(nw<=n&&st[nw].first<st[v1[i]].first-c)
		{
			v1[++ct]=nw;is[nw]=1;tr.modify(1,nw);
			while(is[nw])nw++;
		}
		while(1)
		{
			int tp=query1(st[v1[i]].first+c);
			if(!tp||st[tp].second>=st[v1[i]].second)break;
			v1[++ct]=tp;is[tp]=1;tr.modify(1,tp);
		}
	}
	int g1=st[v1[1]].second;
	for(int i=1;i<=ct;i++)if(st[v1[i]].second!=g1)tp=0;
	if(tp)
	{
		int mn=1e9;
		for(int i=1;i<=ct;i++)mn=min(mn,st[v1[i]].first);
		int s1=query1(mn+c);
		if(!s1||st[s1].second>g1)tp=0;
		else fg[s1]=1;
	}
}
int main()
{
	scanf("%d%d",&n,&c);
	for(int i=1;i<=n;i++)scanf("%d%d",&st[i].first,&st[i].second);
	sort(st+1,st+n+1);
	tr.build(1,1,n);
	while(tp)doit();
	int as=0;for(int i=1;i<=n;i++)as+=is[i];
	printf("%d\n",as);
}
```

##### 2015 ACM-ICPC World Finals G Pipe Stream

###### Problem

有一根长度为 $l$ 的管道，有一个物体，流速为 $[v_1,v_2]$ 单位长度每秒，你需要求出它的速度，要求误差不超过 $\frac t2$

你可以进行多次操作，每次选择一个 $[0,l]$ 内的位置 $x$ ，可以得到这时物体有没有经过这个位置，但物体只会流一次

在物体开始运动后，你可以在 $s$ 秒后进行第一次操作，之后每 $s$ 秒可以进行一次操作

求出最优策略下最坏最少需要几次操作得到速度，或者输出无解

多组数据

$l,v_1,v_2,s,t\leq 10^9,T\leq 10(100)$

$2s,256MB$

###### Sol

操作可以看成判断速度是否大于某个值

考虑一种策略一定可以看成一个树形结构，根节点为 $[v_1,v_2]$ ，每个节点有两个儿子，若一个点的区间为 $[a,b]$ ，在这个点上判断的速度为 $l$ ，则儿子区间为 $[a,l],[l,b]$ ，要求每一个叶子节点区间长度不超过 $t$

如果一个不是最左侧的叶节点长度小于 $t$ ，可以将它的长度改为 $t$ ，然后将左侧的分界点全部向左移动，这样所有判断的速度只会减少，因此一定更优

因此可能判断的速度只有 $v_2-t,...,v_2-kt$ 

考虑算出如果需要判断一个速度，最晚需要在多少次操作时判断，记这个值为 $v_1,...,v_k$ ，显然 $v_1\leq v_2\leq...\leq v_k$

那么操作相当于选出一个大于0的值，删去它将序列分成两部分，然后所有数减一，然后要求两部分都有解

如果最后序列为空则有解，若不为空且无法操作则无解

考虑记 $s=\sum \frac 1{2^{v_i}}$

如果 $s\geq \frac 12$ ，考虑最小的 $j$ 使得 $\sum_{i=1}^j \frac 1{2^{v_i}}\geq \frac 12$

则 $\sum_{i=1}^j2^{v_j-v_i}\geq 2^{v_j-1},\sum_{i=1}^{j-1}2^{v_j-v_i}<2^{v_j-1}$

因为两侧都是整数，所以一定有 $\sum_{i=1}^j \frac 1{2^{v_i}}=\frac 12$

如果 $s\geq 1$ ，则可以将序列分成两份，两份都大于等于 $\frac 12$ ，此时无论从哪一侧分，另外一侧和都大于等于 $\frac 12$ ，再全部减一后大于等于 $1$

否则，从 $j$ 处删去，则两侧都小于 $\frac 12$ ，因此之后都小于 $1$

显然只剩一个元素时 $<1$ 是有解的充分必要条件，所以有解当且仅当 $s<1$

如果可以在 $k$ 步之内得到答案，那么显然将所有大于 $k$ 的 $v_i$ 全部改成 $k$ 有解

同时右边成立显然操作步数不超过 $k$ ，因此答案不超过 $k$ 当且仅当将所有大于 $k$ 的 $v_i$ 全部改成 $k$ 后 $s<1$

因此求出每一种 $v_i$ 以及个数后二分答案即可

复杂度 $O(T\sqrt n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 233333
#define ll long long
int T,l,v1,v2,t,s,vl[N],tp[N],su1[N],ct;
ll su[N],su2[N];
bool check1(int x,ll s)
{
	if(x==0)return !s;
	if(vl[x]-vl[x-1]>55)return check1(x-1,tp[x-1]);
	else return check1(x-1,tp[x-1]+(s>>vl[x]-vl[x-1]));
}
bool check(int x)
{
	int f1=lower_bound(vl+1,vl+ct+1,x)-vl-1;
	return check1(f1,tp[f1]+(x-vl[f1]>=30?0:(su1[f1+1]>>x-vl[f1])));
}
void solve()
{
	for(int i=0;i<=ct+1;i++)vl[i]=tp[i]=su[i]=su1[i]=su2[i]=0;
	v2-=t;ct=0;
	while(v2>v1)
	{
		int st=l/v2/s,st1=l/(st+1)/s;
		if(st1<v1)st1=v1;
		int tp=(v2-st1-1)/t+1;
		vl[++ct]=st,su[ct]=tp;
		v2-=tp*t;
	}
	for(int i=ct;i>=0;i--)tp[i]=su[i],su1[i]=su1[i+1]+tp[i];
	for(int i=ct;i>=1;su2[i]=su[i],i--)if(vl[i]-vl[i-1]<=55)su[i-1]+=(su[i]>>(vl[i]-vl[i-1]));
	if(su[0]){printf("impossible\n");return;} 
	int lb=0,rb=10000;int as=vl[ct];
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%d\n",as);
}
int main()
{
	scanf("%d",&T);
	while(T--)scanf("%d%d%d%d%d",&l,&v1,&v2,&t,&s),solve();
}
```

##### ARC104F Visibility Sequence

###### Problem

给定序列 $a_i$ ，整数序列 $b$ 满足 $1\leq b_i\leq a_i$

定义 $v_i$ 为最大的 $j$ 满足 $j<i,v_j>v_i$ ，不存在则为-1

求出可能的 $v$ 的数量，模 $10^9+7$

$n\leq 100,a_i\leq 10^5$

$2s,512MB$

###### Sol

考虑找到 $v$ 中最后一个 $-1$ ，设它的位置为 $k$ ，那么 $b_k$ 为最大值中最靠右的一个

考虑对于 $[1,k-1],[k+1,n]$ 分别构造 $v_i$ ，然后将 $v_{k+1,...,n}$ 中所有的 $-1$ 改为 $k$ ，即可得到整个序列的 $v_i$

设 $f_{l,r,s}$ 表示只考虑区间 $[l,r]$ ，且每个数不能超过 $s$ 时 $v$ 的数量，考虑枚举 $k$ 的位置，显然这个位置应该尽量大，有

$f_{l,r,s}=\sum_{k=l}^rf_{l,k-1,min(s,a_i)}f_{k+1,r,min(s,a_i)-1}$

注意到如果 $\exists i,b_i>n$ ，那么进行类似离散化的操作后，得到的 $v$ 显然不变，因此只需要考虑 $b_i\leq n$ 的情况

因此答案为 $f_{1,n,n}$

复杂度 $O(n^4)$

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

##### XVII open cup GP of Japan F Right Angle Painting

###### Problem

给一个 $n\times n$ 的网格图，每个位置是障碍或者空位

给定起点，你需要找到一条从起点开始的路径，满足

1. 不经过任何障碍
2. 每个空位正好经过一次
3. 路径上相邻两个位置 $v_1,v_2$ 相邻
4. 路径上相邻三个位置 $v_1,v_2,v_3$ 不在一条直线上

求是否存在这样的路径

$n\leq 400$

$4s,256MB$

###### Sol

考虑如果知道了起点终点以及起点终点处的方向，如何判断是否存在路径

对于非起点终点的点，可以看成它连出了一条横向的边和一条纵向的边

对于起点终点，它只会连出一条边

考虑一行上的情况，有一些点会连出横向边，只有当这样的点形成的每一段长度都是偶数时存在一种方案，且方案唯一

因此确定了起点终点后，可能的路径唯一，可以 $O(n^2)$ 判断合法

显然有解的一个条件是每一行/列上需要连边的点数量为奇数

选择终点以及终点连边的方向只会改变一行或一列的点数，因此确定起点和方向后，如果有解，那么可能的终点必定在某一行/列上，因此可能的终点只有 $O(n)$ 个

因此可以枚举终点判断，复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 405
int n,s1[N],s2[N],fg,sx,sy,ct,fx[N][N],fy[N][N],vis[N][N];
char tp[N][N];
void check(int sx,int sy,int tx,int ty,int f1,int f2)
{
	for(int i=1;i<=n;i++)
	{
		int ls=0;
		for(int j=1;j<=n;j++)if(!(tp[i][j]=='#'||(i==sx&&j==sy&&!f1)||(i==tx&&j==ty&&!f2)))fy[i][j]=ls?-1:1,ls^=1;
		else if(ls)return;else fy[i][j]=0;
	}
	for(int i=1;i<=n;i++)
	{
		int ls=0;
		for(int j=1;j<=n;j++)if(!(tp[j][i]=='#'||(j==sx&&i==sy&&f1)||(j==tx&&i==ty&&f2)))fx[j][i]=ls?-1:1,ls^=1;
		else if(ls)return;else fx[j][i]=0;
	}
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)vis[i][j]=0;
	int ct1=1,tp=f1;
	while(1)
	{
		if(tp==1){if(!fy[sx][sy]){fg|=ct1==ct;return;}sy+=fy[sx][sy],tp^=1,ct1++;}
		else {if(!fx[sx][sy]){fg|=ct1==ct;return;}sx+=fx[sx][sy],tp^=1,ct1++;}
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%s",tp[i]+1);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(tp[i][j]!='#')s1[i]++,s2[j]++,ct++;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(tp[i][j]=='s')sx=i,sy=j;
	for(int t=0;t<=1;t++)
	{
		if(t)s2[sy]--;else s1[sx]--;
		int ct=0,t1,t2;
		for(int i=1;i<=n;i++)if(s1[i]&1)ct++,t1=0,t2=i;
		for(int i=1;i<=n;i++)if(s2[i]&1)ct++,t1=1,t2=i;
		if(ct!=1){if(t)s2[sy]++;else s1[sx]++;continue;}
		if(t1==0)for(int j=1;j<=n;j++)check(sx,sy,t2,j,t,0);
		else for(int j=1;j<=n;j++)check(sx,sy,j,t2,t,1);
		if(t)s2[sy]++;else s1[sx]++;
	}
	printf("%s\n",fg?"POSSIBLE":"IMPOSSIBLE");
}
```

##### XVII open cup GP of Japan C House Moving

###### Problem

给 $m$ 个家庭，第 $i$ 个有 $a_i$ 人

有 $n$ 个房子排成一列，你需要给每个家庭分配一个房子，每个房子内最多有一个家庭

求所有分配方案中 $\sum_{i<j} dis(i,j)a_ia_j$ 的最大值

$n\leq 10^6,m\leq 1000,a_i\leq 100$

$1s,256MB$

###### Sol

如果所有人的相对位置顺序确定，那么可以求出每个人位置的贡献系数

显然贡献系数是单调递增的，因此此时最优策略一定是小于0的放在最左侧，剩下的放在最右侧

对于放在左侧的，考虑交换相邻两个 $a_i,a_j$ 之后的情况，一定是更大的放在左侧最优

因此一定是左侧从大到小排序，右侧从小到大排序

考虑将所有 $a_i$ 从大到小排序，每次加入最大数，那么每次一定加到当前空位的最左侧或最右侧

在每一个 $(i,i+1)$ 处计算对 $dis(i,j)$ 的贡献，设左侧的人数为 $v$ ，则这里的贡献为 $v((\sum_{i=1}^ma_i)-v)$

考虑在左侧第 $j$ 个位置加入一个数时，计算 $(j,j+1)$ 的贡献，在右侧第 $j$ 个位置加入一个数时，计算 $(n-j,n-j+1)$ 的贡献

这样加入所有数之后只剩下中间 $n-m-1$ 个位置的贡献没有计算，且中间都是空的

注意到贡献只和一侧的所有数和有关，设 $f_{i,j}$ 表示放入了前 $i$ 个数，左侧放的数的和为 $j$ ，此时的贡献系数的最大值

记 $s_i=\sum_{j=1}^ia_i$ ，其中 $a_i$ 已经排好序，转移考虑新的数放在哪边，有

$dp_{i,j}=max(dp_{i-1,j-a_i}+j(s_n-j),dp_{i-1,j}+(s_i-j)(n-(s_i-j)))$

答案即为 $max_{i=0}^{s_n}(dp_{n,i}+(n-m-1)i(s_n-i))$

复杂度 $O(m^2v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1050
#define M 105050
#define ll long long
int n,m,v[N],su[N];
ll dp[M],dp2[M];
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d",&v[i]),v[i]*=-1;
	sort(v+1,v+m+1);
	for(int i=1;i<=m;i++)v[i]*=-1,su[i]=su[i-1]+v[i];
	for(int i=1;i<=m;i++)
	{
		for(int j=0;j<=su[i];j++)dp2[j]=-1e18;
		for(int j=0;j<=su[i-1];j++)
		{
			dp2[j+v[i]]=max(dp2[j+v[i]],dp[j]+1ll*(j+v[i])*(su[m]-j-v[i]));
			dp2[j]=max(dp2[j],dp[j]+1ll*(su[i]-j)*(su[m]-su[i]+j));
		}
		for(int j=0;j<=su[i];j++)dp[j]=dp2[j];
	}
	ll as=-1e18;
	for(int i=0;i<=su[m];i++)
	as=max(as,dp[i]+1ll*(n-m-1)*i*(su[m]-i));
	printf("%lld\n",as);
}
```

##### XVII open cup GP of Japan A Circles

###### Problem

给 $n$ 个圆，第 $i$ 个圆圆心为 $(x_i,y_i)$ ，半径为 $\sqrt {x_i^2+y_i^2}$

求出至少在一个圆内部的整点个数

$n,|x_i|,|y_i|\leq 10^5$

$1s,256MB$

###### Sol

考虑整个图形的边界

注意到所有圆都经过原点，以原点进行圆反演，半平面交即可求出所有在边界上的圆和它们的顺序

考虑一段圆弧，设这一段对应的半径为 $r$ ，圆心角为 $a$ ，那么长度为 $ra$

又因为原点在圆上，所以圆弧端点到圆心的角为 $\frac a2$

因为边界反演之后是凸的，所以所以圆弧对应角的和为 $2\pi$

所以距离和不超过 $maxr*4\pi<2\times 10^6$

考虑枚举整点的横坐标，计算边界与对应直线的交点，可以发现这样的点总和不超过边界长度

对于每一段圆弧求出横坐标范围，考虑每一个交点，如果两个相邻圆的交点是整点则特判

可以加入四个小圆避免所有圆都在一个方向的情况

复杂度 $O((n+v)\log (n+v))$

###### Code

```cpp
#include<cstdio>
#include<cmath>
#include<vector>
#include<algorithm>
using namespace std;
#define N 105050
#define ll long long
int n,s[N][2],qu[N],hd,tl,ct,is1,is2,is3,is4,m;
ll as;
vector<pair<int,int> > st[N*6];
double pi=acos(-1);
struct pt{double x,y;}q2[N],p[N];
pt operator +(pt a,pt b){return (pt){a.x+b.x,a.y+b.y};}
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
pt operator *(pt a,double b){return (pt){a.x*b,a.y*b};}
double getdis(pt a){return sqrt(a.x*a.x+a.y*a.y);}
double cross(pt a,pt b){return a.x*b.y-a.y*b.x;}
pt doit(pt a)
{
	double tp=1e10,ds=getdis(a);
	return a*(tp/ds/ds);
}
struct vec{pt s,t;int id;}tp[N],q1[N];
pt intersect(vec a,vec b)
{
	double v1=cross(b.t-a.s,b.s-a.s),v2=cross(b.t-a.s,a.t-a.s)+cross(a.t-a.s,b.s-a.s);
	return a.s+(a.t-a.s)*(v1/v2);
}
double getang(vec a){return atan2(a.t.y-a.s.y,a.t.x-a.s.x);}
bool cmp(vec a,vec b)
{
	double a1=getang(a),a2=getang(b);
	if(abs(a1-a2)<=1e-13)return cross(b.t-a.s,a.t-a.s)>0;
	return a1<a2;
}
void SI()
{
	sort(tp+1,tp+ct+1,cmp);
	q1[hd=tl=1]=tp[1];
	for(int i=2;i<=ct;i++)
	if(abs(getang(tp[i])-getang(tp[i-1]))>=1e-13)
	{
		while(hd<tl&&cross(tp[i].t-tp[i].s,q2[tl-1]-tp[i].s)<=1e-7)tl--;
		while(hd<tl&&cross(tp[i].t-tp[i].s,q2[hd]-tp[i].s)<=1e-7)hd++;
		q1[++tl]=tp[i];q2[tl-1]=intersect(q1[tl],q1[tl-1]);
	}
	while(hd<tl&&cross(q1[hd].t-q1[hd].s,q2[tl-1]-q1[hd].s)<=1e-7)tl--;
	while(hd<tl&&cross(q1[tl].t-q1[tl].s,q2[hd]-q1[tl].s)<=1e-7)hd++;
	q2[tl]=intersect(q1[hd],q1[tl]);
}
pt rot(pt a,double ag){return (pt){a.x*cos(ag)-a.y*sin(ag),a.x*sin(ag)+a.y*cos(ag)};}
pt solve1(pt a,pt b)
{
	pt tp=(pt){0,0};
	b=b-a;tp=tp-a;
	double ag=getang((vec){(pt){0,0},b});
	b=rot(b,-ag);tp=rot(tp,-ag);
	tp.y*=-1;
	tp=rot(tp,ag);tp=tp+a;
	return tp;
}
bool check2(int a,int b)
{
	for(int i=1;i<=m;i++)
	if(1ll*s[i][0]*s[i][0]+1ll*s[i][1]*s[i][1]>=1ll*(s[i][0]-a)*(s[i][0]-a)+1ll*(s[i][1]-b)*(s[i][1]-b))return 1;
	return 0;
}
int main()
{
	scanf("%d",&n);m=n;
	for(int i=1;i<=n;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(s[i][0]==1&&s[i][1]==0)is1=1;
		if(s[i][0]==-1&&s[i][1]==0)is2=1;
		if(s[i][1]==1&&s[i][0]==0)is3=1;
		if(s[i][1]==-1&&s[i][0]==0)is4=1;
	}
	if(!is1)s[++n][0]=1,s[n][1]=0;
	if(!is2)s[++n][0]=-1,s[n][1]=0;
	if(!is3)s[++n][0]=0,s[n][1]=1;
	if(!is4)s[++n][0]=0,s[n][1]=-1;
	for(int i=1;i<=n;i++)
	{
		p[i]=(pt){s[i][0],s[i][1]};
		pt s1=(pt){s[i][0]*2,0},s2=(pt){0,s[i][1]*2};
		if((s[i][0]<0&&s[i][1]>0)||(s[i][0]>0&&s[i][1]<0))swap(s1,s2);
		if(!s[i][0])s1=(pt){s[i][1],s[i][1]},s2=(pt){-s[i][1],s[i][1]};
		if(!s[i][1])s1=(pt){s[i][0],-s[i][0]},s2=(pt){s[i][0],s[i][0]};
		tp[++ct]=(vec){doit(s1),doit(s2),i};
	}
	SI();
	ct=0;
	for(int i=hd;i<=tl;i++)if(q1[i].id!=-1)qu[++ct]=q1[i].id;
	if(ct==1)
	{
		ll r=1ll*s[qu[1]][0]*s[qu[1]][0]+1ll*s[qu[1]][1]*s[qu[1]][1],as=2*(int)sqrt(r)+1;
		for(int i=1;1ll*i*i<=r;i++)as+=2*(2*(int)sqrt(r-1ll*i*i)+1);
		printf("%lld\n",as);
		return 0;
	}
	for(int i=1;i<=ct;i++)
	{
		int x=qu[i],ls=qu[i==1?ct:i-1],rs=qu[i==ct?1:i+1];
		pt v1=solve1(p[ls],p[x]),v2=solve1(p[x],p[rs]);
		if(ls==rs)v2=(pt){0,0};
		double lb=min(v1.y,v2.y),rb=max(v1.y,v2.y);
		if(getdis((pt){p[x].x,p[x].y+getdis(p[x])}-p[ls])>=getdis(p[ls])-1e-8&&getdis((pt){p[x].x,p[x].y+getdis(p[x])}-p[rs])>=getdis(p[rs])-1e-8)rb=p[x].y+getdis(p[x]);
		if(getdis((pt){p[x].x,p[x].y-getdis(p[x])}-p[ls])>=getdis(p[ls])-1e-8&&getdis((pt){p[x].x,p[x].y-getdis(p[x])}-p[rs])>=getdis(p[rs])-1e-8)lb=p[x].y-getdis(p[x]);
		for(int sy=lb;sy<=rb;sy++)
		{
			int ds=s[x][1]-sy;
			if(1ll*s[x][0]*s[x][0]+1ll*s[x][1]*s[x][1]-1ll*ds*ds<0)continue;
			double ds2=sqrt(1ll*s[x][0]*s[x][0]+1ll*s[x][1]*s[x][1]-1ll*ds*ds)+1e-9;
			pt v1=(pt){s[x][0]-ds2,sy},v2=(pt){s[x][0]+ds2,sy};
			if(getdis(v1-p[ls])>=getdis(p[ls])+1e-8&&getdis(v1-p[rs])>=getdis(p[rs])+1e-8)
			st[sy+300000].push_back(make_pair(s[x][0]-(int)(ds2),1));
			if(getdis(v2-p[ls])>=getdis(p[ls])+1e-8&&getdis(v2-p[rs])>=getdis(p[rs])+1e-8)
			st[sy+300000].push_back(make_pair(s[x][0]+(int)(ds2)+1,-1));
		}
	}
	for(int i=1;i<=ct;i++)
	{
		int x=qu[i],y=qu[i==1?ct:i-1];
		pt s1=solve1(p[x],p[y]);
		int t1=s1.x<0?s1.x-0.4:s1.x+0.4,t2=s1.y<0?s1.y-0.4:s1.y+0.4;
		if(1ll*s[x][0]*s[x][0]+1ll*s[x][1]*s[x][1]!=1ll*(s[x][0]-t1)*(s[x][0]-t1)+1ll*(s[x][1]-t2)*(s[x][1]-t2))continue;
		if(1ll*s[y][0]*s[y][0]+1ll*s[y][1]*s[y][1]!=1ll*(s[y][0]-t1)*(s[y][0]-t1)+1ll*(s[y][1]-t2)*(s[y][1]-t2))continue;
		pt v1=(pt){t1-1,t2},v2=(pt){t1+1,t2};
		bool fg1=getdis(v1-p[x])<=getdis(p[x])+1e-8||getdis(v1-p[y])<=getdis(p[y])+1e-8,fg2=getdis(v2-p[x])<=getdis(p[x])+1e-8||getdis(v2-p[y])<=getdis(p[y])+1e-8;
		if(!fg1)st[t2+300000].push_back(make_pair(t1,1));
		if(!fg2)st[t2+300000].push_back(make_pair(t1+1,-1));
	}
	for(int i=0;i<=600000;i++)if(st[i].size()&&i!=300000)
	{
		sort(st[i].begin(),st[i].end());
		int ls=-1e7,s1=0;
		for(int j=0;j<st[i].size();j++)if(!j||st[i][j]!=st[i][j-1])as+=(st[i][j].first-ls)*(s1>0),ls=st[i][j].first,s1+=st[i][j].second;
	}
	int mx=0,mn=0; 
	for(int i=1;i<=n;i++)mx=max(mx,s[i][0]),mn=min(mn,s[i][0]);
	as+=2*mx-2*mn+1;
	for(int i=-2;i<=2;i++)
	for(int j=-2;j<=2;j++)
	if((i>0?i:-i)+(j>0?j:-j)<=2)
	if(!check2(i,j))as--;
	printf("%lld\n",as);
}
```

#####  Petrozavodsk Programming Camp Winter 2015 Day3 F Saddle Point

###### Problem 

给一个 $n\times m$ 的矩阵，每个位置的数为 $[1,k]$ 间的整数

定义一个位置是好的，当且仅当这个位置是它所在行和所在列的严格最大值

求至少有一个好的位置的矩阵个数，模 $10^9+7$

$n,m\leq 500,k\leq 10$

$1s,512MB$

###### Sol

显然两个好的点不会在同一行或者同一列，因此好的点不超过 $n$ 个

考虑容斥算没有好的位置的矩阵数，假设当前钦定有 $a$ 个位置是好的，选择这些位置的方案数为 $C_n^iC_m^ii!$

选了这些位置后，可以发现交换两行或者两列不影响哪些位置是好的，因此可以通过交换使得这些好的位置为 $(1,1),(2,2),(3,3),...,(a,a)$ ，且使得 $(i,i)$ 位置上的数小于等于 $(i+1,j+1)$ 位置上的数

从小到大考虑每一个 $(i,i)$ ，可以发现它会影响 $(i+1,i),(i+2,i),...,(n,i),(i,i+1),(i,i+2),...,(i,m)$ 位置的值的上限(左上部分的已经被考虑过了)，可以在这时计算这些影响的位置的值的方案数

考虑一次处理同一种值，设 $i$ 有 $v_i$ 个，那么它们排列的方案数为 $\frac{a!}{\prod v_i!}$ ，可以在dp时计算这个系数

设 $f_{i,j}$ 表示考虑了前 $i$ 个点， $(i,i)$ 位置的值不超过 $j$ 且 $(i+1,i+1)$ 位置(如果存在)值大于 $j$ 时，前面的所有方案中已经确定取值上限的位置的值的方案数再乘上 $\frac 1{\prod v_i!}$ 的和

枚举下一种值有多少个，于是有

$f_{i,j}=\sum_{t=0}^jf_{i-t,j-1}*(j-1)^{t(t-1)+t((n-t)+(m-t))}*\frac 1{t!}$

再考虑右下角没有确定上限的位置，于是钦定 $a$ 个好的位置的方案数为 $\frac{n!}{(n-a)!}*\frac{n!}{(n-b)!}*m^{(n-a)(n-b)}$

复杂度 $O(n^2k)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505
#define mod 1000000007
int n,m,k,dp[N][11],fr[N],ifr[N],f[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	if(n>m)n^=m^=n^=m;
	fr[0]=ifr[0]=1;
	for(int i=1;i<=m;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	dp[0][0]=1;
	for(int i=1;i<=k;i++)
	for(int j=0;j<=n;j++)
	for(int l=0;l<=j;l++)
	dp[j][i]=(dp[j][i]+1ll*dp[l][i-1]*pw(i-1,(j-l)*(n+m-j*2)+(j-l)*(j-l-1))%mod*ifr[j-l])%mod;
	for(int i=0;i<=n;i++)f[i]=1ll*fr[n]%mod*ifr[n-i]%mod*fr[m]%mod*ifr[m-i]%mod*dp[i][k]%mod*pw(k,(n-i)*(m-i))%mod;
	int as=0;
	for(int i=1;i<=n;i++)as=(as+1ll*(i&1?1:mod-1)*f[i])%mod;
	printf("%d\n",as);
}
```

##### Petrozavodsk Programming Camp Winter 2015 Day4 F Fulkerson

###### Problem

给一棵 $n$ 个点的树，对于每一个 $k\in\{1,2,...,n\}$ ，求出在树上选出 $k$ 个点，使得每个点到选出点的最小距离的最大值最小

$n\leq 1.5\times 10^5$

$5s,512MB$

###### Sol

考虑这样一个贪心：从下往上考虑每个点，如果当前点子树内点数大于 $\frac nk$ ，就选择这个点，然后删去这个子树

这样操作后，如果删去所有选择点与父亲的连边，则除了根所在的块，每一块大小都大于 $\frac nk$ ，因此选择的点不超过 $k$ 个

又因为从下往上考虑，显然每一块删去选中点后子树大小都小于 $\frac nk$ ，因此如果删除所有选中点，则剩下的每一个连通块大小不超过 $\frac nk$

因此答案不超过 $\frac nk$

考虑对于答案 $s$ ，计算至少需要放多少个点，进行dfs，记录每个点子树内距离它最近的选中点距离 $a_i$ 以及距离选中点距离大于 $s$ 的点中距离根最远的距离 $b_i$

那么有

$a_u=min_{v\in son_u}a_v+1$

$b_u=max([a_u>s]0,max_{v\in son_u,b_v+1+a_u>s}b_v+1)$ (若没有满足条件的取-1)

如果 $b_u=s$ ，那么此时必须选这个点

显然贪心地往上放最优，因此这样最优

先求出 $s=1,2,...,S$ 时的 $k$ ，此时最多剩余 $\frac nS$ 个 $k$ ，可以对于每一个二分求答案

复杂度 $O(n\sqrt{n\log n})$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 150500
int n,head[N],cnt,s1[N],as[N],ls[N],tp[N],a,b,t1=600,id2[N],l1[N],r1[N],ct1;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa)
{
	l1[id2[u]]=ct1+1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)id2[ed[i].t]=++ct1;
	r1[id2[u]]=ct1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u);
}
int solve(int k)
{
	int ct=0;
	for(int u=n;u>=1;u--)
	{
		ls[u]=1e9;tp[u]=0;
		for(int i=l1[u];i<=r1[u];i++)ls[u]=min(ls[u],ls[i]+1),tp[u]=max(tp[u],tp[i]+1);
		if(tp[u]+ls[u]<=k)tp[u]=-1;
		if(tp[u]==k||(u==1&&tp[u]!=-1))ct++,tp[u]=-1,ls[u]=0;
	}
	return ct;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	id2[ct1=1]=1;dfs1(1,0);
	s1[0]=n;
	for(int i=1;i<=t1&&i<=n;i++)s1[i]=solve(i);
	for(int i=1;i<=t1;i++)
	for(int j=s1[i];j<s1[i-1];j++)as[j]=i;
	int tp=s1[t1]-1;as[0]=n;
	for(int i=tp;i>=1;i--)
	{
		int lb=as[i+1],rb=n/i,as1=rb;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(solve(mid)<=i)as1=mid,rb=mid-1;
			else lb=mid+1;
		}
		as[i]=as1;
	}
	for(int i=1;i<=n;i++)printf("%d ",as[i]);
}
```

##### Gym 100739K Easy Vector

###### Problem

给一个长度为 $n$ 的数组，你一开始在1，每个时刻你可以向左或右走一步，每经过一个位置就可以获得一次这个位置上的分数

$q$ 次询问，每次给出时间 $t$ ，求 $t$ 秒内的最大分数

$n,q\leq 10^5,t\leq 10^9$

###### Sol

考虑一种走的路径，假设最远走到了 $x$

设 $x$ 满足 $v_x+v_{x+1}$ 为所有 $v_i+v_{i+1},i+1\leq x$ 的最大值，可以将路径分成若干对 $(i,i+1)$ 

可以发现，其中至少存在一对 $(1,2),(3,4),...,(x-1-(x\bmod 2),x-(x\bmod 2))$ ，显然可以将剩下的全部换成 $(x,x+1)$ 

因此路径一定为走到某个点，然后在两个点上来回走

可以发现每一对 $(x,x+1)$ 对答案的贡献分奇偶考虑后是两个一次函数，因此可以对奇偶分别维护李超树

按照递增处理询问，这样每次加入的线段一定影响之后的所有询问，因此复杂度为 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105050
#define ll long long
int n,q,v[N],qu[N],lb=1,id[N];
ll su[N],as[N];
struct lct{
	struct node{int l,r;ll k,b;}e[N*4];
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].b=-1e17;
		if(l==r)return;
		int mid=(l+r)>>1;
		build(x<<1,l,mid),build(x<<1|1,mid+1,r);
	}
	void modify(int x,ll k,ll b)
	{
		if(x>q*4||!e[x].l)return;
		int mid=(e[x].l+e[x].r)>>1;
		if(k*qu[id[mid]]+b>e[x].k*qu[id[mid]]+e[x].b)swap(e[x].k,k),swap(e[x].b,b);
		modify(x<<1|(k>e[x].k),k,b);
	}
	ll query(int x,ll s,ll v)
	{
		if(e[x].l==e[x].r)return e[x].k*v+e[x].b;
		int mid=(e[x].l+e[x].r)>>1;
		return max(e[x].k*v+e[x].b,query(x<<1|(mid<s),s,v));
	}
}t[2];
bool cmp(int a,int b){return qu[a]<qu[b];}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su[i]=su[i-1]+v[i];
	for(int i=1;i<=q;i++)scanf("%d",&qu[i]),id[i]=i;
	sort(id+1,id+q+1,cmp);
	t[0].build(1,1,q);t[1].build(1,1,q);
	for(int i=2;i<=n+1;i++)
	{
		if(i<=n)
		{
			ll k=v[i]+v[i-1],b=2*su[i]-k*(i-1);
			t[(i-1)&1].modify(1,k,b);
		}
		if(i>2)
		{
			ll k=v[i-2]+v[i-1],b=2*(su[i-1]+v[i-2])-k*(i-1);
			t[(i-1)&1].modify(1,k,b);
		}
		while(qu[id[lb]]==i-1)as[id[lb]]=t[qu[id[lb]]&1].query(1,lb,qu[id[lb]])/2,lb++;
	}
	while(lb<=q)as[id[lb]]=t[qu[id[lb]]&1].query(1,lb,qu[id[lb]])/2,lb++;
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```

##### NAIPC 2015 D Extensive OR

###### Problem

给定 $n,k$ 以及一个01串 $s$ ，将 $s$ 循环 $k$ 次，得到的串对应的二进制数为 $r$ 

求有多少对 $(a_1,...,a_n)$ 满足

1. $a_1,...,a_n$ 互不相等
2. $0\leq a_1,...,a_n< r$
3. $a_1\oplus ...\oplus a_n=0$

答案模 $10^9+7$

$n\leq 7,k\leq 10^5,|s|\leq 50$

$3s,256MB$

###### Sol

非常显然有 $O(2^{3n}(\log k+|s|))$ 的做法

考虑没有不同的条件怎么算

枚举与上界不同的位中最高位，枚举这一位上有多少个数与上界不同

之后选一个与上界不同的数，剩下的任意填，可以调整这个数使得后面异或为0，只需要前面异或为0即可

这样的复杂度为 $O(nk|s|)$

考虑斯特林容斥，枚举相同的集合的情况，可以推出一个大小为 $i$ 的集合容斥系数为 $(-1)^{i-1}(i-1)!$ ，这部分爆搜即可

复杂度 $O(n^2k|s|+2^n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 5005005
#define mod 1000000007
int n,k,s[N],f[8],l,fr[8],ifr[8],su[N],as,t1,pw1[N],pw2[8],pw3[8],vl[N];
char st[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int solve(int n)
{
	int as=~n&1;
	for(int i=1;i<=k*l;i++)if(s[i]==1)
	{
		int tp=1ll*(n&1?1:pw1[i])*pw(su[i+1]+1,n-(n&1?1:2))%mod;
		for(int j=(n-1)%2+1;j<=n;j+=2,tp=1ll*tp*vl[i]%mod)
		as=(as+1ll*fr[n]*ifr[j]%mod*ifr[n-j]%mod*tp)%mod;
		if(n&1)break;
	}
	return as;
}
void dfs(int m,int a,int b,int tp)
{
	if(!m){as=(as+1ll*tp*f[a]%mod*pw(t1,b-a))%mod;return;}
	for(int i=1;i<=m;i++)dfs(m-i,a+(i&1),b+1,1ll*tp*fr[m-1]%mod*ifr[m-i]%mod*(i&1?1:mod-1)%mod);
}
int main()
{
	scanf("%d%d%s",&n,&k,st+1);
	l=strlen(st+1);
	for(int i=1;i<=k;i++)
	for(int j=1;j<=l;j++)
	s[(i-1)*l+j]=st[j]-'0';
	s[k*l]--;
	for(int i=k*l;i>=1;i--)if(s[i]==-1)s[i-1]--,s[i]+=2;
	for(int i=1;i<=k*l;i++)t1=(2*t1+s[i])%mod;t1++;
	for(int tp=1,i=k*l;i>0;i--,tp=2*tp%mod)
	su[i]=(su[i+1]+s[i]*tp)%mod;
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	pw1[l*k]=1;for(int i=l*k-1;i>=1;i--)pw1[i]=1ll*pw1[i+1]*2%mod;
	for(int i=1;i<=l*k;i++)vl[i]=1ll*pw1[i]*pw(su[i+1]+1,mod-2)%mod,vl[i]=1ll*vl[i]*vl[i]%mod;
	for(int i=1;i<=n;i++)f[i]=solve(i);f[0]=1;
	dfs(n,0,0,1);printf("%d\n",1ll*as*ifr[n]%mod);
}
```

##### NAIPC 2015 F Sand Art

###### Problem

有 $n$ 个区域，第 $i$ 个区域长度为 $l_i$ ，宽度为 $1$ ，高度为 $h$

有 $m$ 种沙子，第 $i$ 种有 $v_i$ 单位

要求第 $i$ 个区域内的沙子不能少于 $mn_{i,j}$ ，不能多于 $mx_{i,j}$

求所有区域中最高的沙子高度与最低的沙子高度的差的最小值

保证有解

$n,m\leq 200$

$3s,256MB$

###### Sol

首先把最小值数量的沙子都填进去，这时每个区域每种沙子相当于只有上界，每个区域有一个初始高度

显然此时存在一种最优方案使得最大高度不再变化，考虑二分最低高度，相当于每个区域有一个至少需要再填进去的沙子数量

考虑网络流，原点向每种沙子连对应的边，沙子向区域连边，区域向汇点连边，右侧满流则有解

复杂度应该是 $O(nm\sqrt{n+m}\log v)$

###### Code

注意实数网络流时流量的eps和判答案的eps不能相同

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 405
#define M 105050
int n,m,w,h,head[N],cnt,dis[N],cur[N];
double su[N],sz[N],a,s1[N],s2[N][N];
struct edge{int t,next;double v;}ed[M*2];
void adde(int f,int t,double v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;}
bool bfs(int s,int t)
{
	queue<int> st;
	for(int i=1;i<=n+m+2;i++)dis[i]=-1,cur[i]=head[i];
	st.push(s);dis[s]=0;
	while(!st.empty())
	{
		int tp=st.front();st.pop();
		for(int i=head[tp];i;i=ed[i].next)
		if(ed[i].v>1e-12&&dis[ed[i].t]==-1){dis[ed[i].t]=dis[tp]+1;st.push(ed[i].t);if(ed[i].t==t)return 1;}
	}
	return 0;
}
double dfs(int u,int t,double f)
{
	if(u==t||f<1e-12)return f;
	double as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)
	if(ed[i].v>1e-12&&dis[ed[i].t]==dis[u]+1&&((tp=dfs(ed[i].t,t,min(ed[i].v,f)))>1e-12))
	{
		as+=tp;f-=tp;
		ed[i].v-=tp;ed[i^1].v+=tp;
		if(f<1e-12)return as;
	}
	return as;
}
bool check(double v)
{
	cnt=1;
	for(int i=1;i<=n+m+2;i++)head[i]=0;
	for(int i=1;i<=m;i++)adde(n+m+1,n+i,su[i]);
	for(int i=1;i<=m;i++)
	for(int j=1;j<=n;j++)
	adde(n+i,j,s2[j][i]);
	double tp=0;
	for(int i=1;i<=n;i++)
	{
		double vl=max(0.0,sz[i]*v-s1[i]);
		tp+=vl;adde(i,n+m+2,vl);
	}
	while(bfs(n+m+1,n+m+2))
	tp-=dfs(n+m+1,n+m+2,5e7);
	return tp<=1e-6;
}
int main()
{
	scanf("%d%d%d%d",&n,&m,&w,&h);
	for(int i=1;i<=m;i++)scanf("%lf",&su[i]);
	for(int i=1;i<n;i++)scanf("%lf",&sz[i]);sz[n]=w;
	for(int i=n;i>=1;i--)sz[i]-=sz[i-1];
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%lf",&s2[i][j]),s1[i]+=s2[i][j],su[j]-=s2[i][j];
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)a=s2[i][j],scanf("%lf",&s2[i][j]),s2[i][j]-=a;
	double mx=0;
	for(int i=1;i<=n;i++)mx=max(mx,s1[i]/sz[i]);
	double lb=0,rb=mx,as=0;
	for(int i=1;i<=75;i++)
	{
		double mid=(lb+rb)/2;
		if(check(mid))as=mid,lb=mid;
		else rb=mid;
	}
	printf("%.3lf\n",mx-as);
}
```

##### ......... A Covering

###### Problem

给一个 $n$ 个点的图，对于每一对 $(p,q)(0\leq p,q\leq n)$ 求出是否可以将点集划分成 $p$ 个非空团和 $q$ 个非空独立集

$n\leq 20$

$1s,256MB$

###### Sol

团的子集显然是团，独立集的子集也是独立集

如果存在两个团 $p,q(|p\cup q|>1)$ ，显然 $p\cup q$ 可以被划分成两个团，独立集类似

设团的集合幂级数为 $A$ ，独立集的集合幂级数为 $B$ ，那么有解的条件是 $p+q\leq n$ 且or卷积意义下 $A^pB^q$ 在 $\{1,2,...,n\}$ 有值

考虑先求出fwt后类似点值的表示，然后可以 $O(2^n)$ 求出一个 $A^pB^q$ fwt后的结果，然后可以 $O(2^n)$ 算出一项的值

如果不考虑 $p+q\leq n$ 的条件，显然 $(i,j)$ 合法则 $(i,j+1)$ 合法，因此从小到大枚举 $i$ ，只需要判断 $O(n)$ 对 $(p,q)$

复杂度 $O(n2^n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 21
#define M 1050050
#define mod 998244353
int n,m,a,b,st[N],s1[M],s2[M],p1[M][N],p2[M][N],tp[N],ct[M];
bool check1(int s)
{
	for(int i=1;i<=n;i++)if(s&(1<<i-1))if(((st[i]|1<<i-1)&s)!=s)return 0;
	return 1;
}
bool check2(int s)
{
	for(int i=1;i<=n;i++)if(s&(1<<i-1))if(st[i]&s)return 0;
	return 1;
}
bool check(int a,int b)
{
	int as=0;
	for(int i=0;i<1<<n;i++)as=(as+1ll*(ct[i]&1?-1:1)*p1[i][a]*p2[i][b]);
	return as;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),st[a]|=1<<b-1,st[b]|=1<<a-1;
	for(int i=1;i<1<<n;i++)s1[i]=check1(i),s2[i]=check2(i),ct[i]=ct[i-(i&-i)]+1;
	for(int i=2;i<=1<<n;i<<=1)
	for(int j=0;j<1<<n;j+=i)
	for(int k=j;k<j+(i>>1);k++)
	s1[k+(i>>1)]+=s1[k],s2[k+(i>>1)]+=s2[k];
	for(int i=0;i<1<<n;i++)
	{
		p1[i][0]=1;for(int j=1;j<=n;j++)p1[i][j]=1ll*p1[i][j-1]*s1[i];
		p2[i][0]=1;for(int j=1;j<=n;j++)p2[i][j]=1ll*p2[i][j-1]*s2[i];
	}
	for(int i=n;i>=0;i--){tp[i]=tp[i+1];while(!check(i,tp[i]))tp[i]++;}
	for(int i=0;i<=n;i++,printf("\n"))
	for(int j=0;j<=n;j++)printf("%d",tp[i]<=j&&i+j<=n);
}
```



