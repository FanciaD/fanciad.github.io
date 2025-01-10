---
title: 校内OJ题解集合
date: '2022-04-21 15:42:19'
updated: '2022-05-15 13:02:56'
tags: Mildia
permalink: RestfortheWeary/
description: auoj 2019-2022
mathjax: true
---

### 模拟赛题解集合

题解绝大部分是20年写的，有几个题是22年补的~~有效的区分方式是22年的题解普遍比较长~~

~~同时进行了重新排版，但重写题解工作量太大就咕了~~最后还是做了大规模重构

#### SCOI2020模拟6

##### auoj2 天才数学少女

###### Problem

有 $T$ 组询问，每次给定 $a,p$，求一个在 $[0,2\times 10^{18}]$ 的正整数 $n$ 使得 $a^n\equiv n(\bmod p)$，输出任意一个解或输出无解

$T\leq 100,a,p\leq 10^9$

$1s,256MB$

###### Sol

根据欧拉定理，有 $x^{\phi(p)}\equiv 1(\bmod p)$

设 $g=gcd(p,\phi(p))$，如果求出了 $a^n\equiv n(\bmod g)$ 的解 $n$ ，那么有 $a^{n+k\phi(p)}\equiv a^n(\bmod p)$

相当于求出 $a^n\equiv n+k\phi(p)(\bmod p)$ 的一组解，因为 $a^n\equiv n(\bmod g)$ 所以同余方程一定有解，做 `exgcd` 就可以求出答案。

考虑递归下去做，只会递归 $\log n$ 次，每次直接 $O(\sqrt p)$算 $\phi$，因为一次之后每次至少除以 $2$ 所以复杂度为 $O(T\sqrt p)$

通过上面的过程可以看出一定有解

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
int t,a,b;
int solve1()
{
	a%=b;if(a==0)return b;
	int v1=1;
	for(int i=1;i<=100;i++){v1=1ll*v1*a%b;if(v1==i%b)return i;}
	return -1;
}
int phi(int x)
{
	int as=x;
	for(int i=2;1ll*i*i<=x;i++)if(x%i==0){as=1ll*as*(i-1)/i;while(x%i==0)x/=i;}
	if(x>1)as=1ll*as*(x-1)/x;
	return as;
}
int gcd(int a,int b){return b?gcd(b,a%b):a;}
int exgcd(int a,int b,int &x,int &y){if(!b){x=1,y=0;return a;}int g=exgcd(b,a%b,y,x);y=y-a/b*x;return g;}
int getinv(int a,int b){int x,y;int g=exgcd(a,b,x,y);x%=(b/g);x=(x+(b/g))%(b/g);return x;}
int pw(int a,int b,int p){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
ll solve2(int a,int b,int k)
{
	a%=b;if(a==0)return b;
	int p=phi(b);
	if(gcd(p,b)==1)
	{
		int v1=getinv(1ll*p*k%b,b);
		return 1ll*p*v1;
	}
	int st=gcd(p,b);
	long long v2=solve2(a,st,k);
	int s1=pw(a,v2%p+(v2>=p)*p,b),s2=1ll*k*v2%b,s3=(s1-s2+b)%b;
	s3/=st;
	int v3=1ll*s3*getinv(1ll*p*k%b,b)%(b/st);
	return v2+1ll*v3*p;
}
ll solve3(int a,int b)
{
	a%=b;if(!a)return b;
	if(gcd(a,b)==1)return solve2(a,b,1);
	int v1=pw(a,b,b),g=gcd(v1,b);
	if(g==b)return b;
	v1/=g;b/=g;int s1=pw(a,g,b);
	return solve2(s1,b,getinv(v1,b))*g+b*g;
}
int main()
{
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d%d",&a,&b);
		if(solve1()!=-1)printf("%d\n",solve1());
		else printf("%lld\n",solve3(a,b));
	}
}
```



##### auoj3 fake

###### Problem

给定 $n,m$ 和 $a_{1...m}$，称一个序列 $b_{1,...,k}$ 是合法的当且仅当它满足：

1. $m|k$
2. $1\leq b_1<b_2<...<b_k\leq n$
3. $\forall i,\sum_{m|(j-i)} b_j\leq a_i$

求所有合法序列中 $\sum b_i$ 的最大值

$n\leq 10^9,m\leq 2\times10^6$

$1s,1024MB$

###### Sol

考虑暴力，首先枚举 $k$，那么显然有 $\sum_{m|(j-i)}b_j-\sum_{m|(j-1)} b_j\geq (i-1)*\frac{k}{m}$

因此可以计算出 $\sum_{m|(j-1)} b_j$ 可能的最大值 $v=min_i(a_i-(i-1)*\frac{k}{m})$

如果一种方案中 $\sum_{m|(j-1)} b_j<v$，如果 $\exists y,\forall 1<i<y,\sum_{m|(j-i)}b_j -(i-1)*\frac{k}{m}=\sum_{m|(j-1)}b_j,\sum_{m|(j-y)}b_j -(y-1)*\frac{k}{m}>\sum_{m|(j-1)}b_j$，则此时一定存在 $b_{xm+j}>b_{xm+j-1}-1$。因为 $\sum_{m|(j-1)} b_j<v$，将 $b_{xm+1},...,b_{xm+j-1}$ 全部加一一定合法且答案更优。

如果不存在 $y$ 满足条件，那么因为 $\sum_{m|(j-1)} b_j<v$，一定存在 $b_{xm+m}<b_{xm+m+1}-1$ 或者 $a_{k}<n$，因此可以将 $b_{xm+1},...,b_{xm+m}$ 全部加一，使答案更优。

因此最优解一定满足 $\sum_{m|(j-1)} b_j=v$

考虑如果确定了 $b_1,b_{m+1},...$，则接下来的操作可以看成一开始 $\forall 0\leq x<\frac{k}{m},1<y\leq m,b_{xm+y}=b_{xm+y-1}+1$，然后每次可以给一段 $b_{xm+y},...,b_{xm+m}$ 加上 $1$。

容易发现操作次数上限为 $n-b_1+1-k$，因此需要 $a_1$ 尽量小。

那么可以构造一组方案，除了无解外有三种情况

1. $b_1=n-k+1$
2. $b_1>1,\sum_{m|(j-1)}b_j=v$
3. $b_1=1,\sum_{m|(j-1)}b_j=v$

大力讨论即可

解出一组 $b_1,b_{m+1},...$ 之后，接下来的操作相当于每次给一段 $b_{xm+y},...,b_{xm+m}$ 加上 $1$，加的次数有上限，并且需要满足$\forall i,\sum_{m|(j-i)} b_j\leq a_i$。

显然贪心加尽量小的 $y$ 最优，可以 $O(m)$ 求出每次应该加的，枚举 $k$ 的总复杂度是 $O(n)$。

考虑对三种情况分别求最优解：

对于第一种情况，显然 $k$ 越大越优。

对于第二种情况，只有一个 $k$ 满足这种情况，可以暴力做。

考虑第三种情况，~~打表可以发现答案是一个上凸函数。~~

注意到如果最后一个 $\sum_{m|(j-i)} b_j< a_i$ 但操作次数不够了，那么这时所有 $b_{xm+j}$ 一定全部到了上界，即 $b_{xm+j}+(m-j+1)=b_{(x+1)m+1}(b_{k+1}=n+1)$。

那么一定有 $\sum_{m|(j-i)} b_j\leq n+v-(m-j+1)\frac{k}{m}$，这是一个关于 $k$ 的上凸函数。

对于每一个 $\sum_{m|(j-i)} b_j$，它能操作到的上界为 $\min_{x>j}(a_x-\frac{k}{m}(x-j))$，注意到以 $k$ 为横轴，$\min$ 的每一部分相当于一条直线，那么最后的上界一定是一个凸壳，因此是上凸函数。

两个上凸函数取 $\min$ 还是上凸函数，上凸函数相加还是上凸函数。

因此这部分最后的答案是一个关于 $k$ 的上凸函数，因此这部分可以三分解决

复杂度 $O(m\log m)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 233333
int n,m;
long long v[N],d[N],las,las2;
long long solve(int s)
{
	if(s<0||s>m/n)return -1e18;
	d[n]=v[n];for(int i=n-1;i>=1;i--)d[i]=min(v[i],d[i+1]-s);
	int lb=1,rb=1+(s-1)*n;
	if(1ll*(lb+rb)*s/2>d[1])return -1e18;
	if(!s)return 0;
	int rb2=m-n+1,lb2=m-n+1-(s-1)*n;
	long long fuc=1ll*(lb2+rb2)*s/2,tp1=lb2-(fuc-d[1]);
	if(tp1<=0)tp1=1;
	if(fuc<d[1])return 1ll*(lb2+rb2)*(rb2-lb2+1)/2;
	long long v1=1ll*d[1]*n+1ll*s*n*(n-1)/2,v2=m-tp1-s*n+1;
	for(int i=2;i<=n;i++)d[i]=(d[i]-d[1]-s*(i-1));d[1]=0;
	for(int i=n-1;i>=2;i--)d[i]=min(d[i],d[i+1]);
	for(int i=2;i<=n;i++)
	{
		long long fuc=d[i]-d[i-1];
		if(fuc>v2)fuc=v2;
		v1+=1ll*(n-i+1)*fuc;v2-=fuc;
	}
	return v1;
}
bool check1(int s)
{
	d[n]=v[n];for(int i=n-1;i>=1;i--)d[i]=min(v[i],d[i+1]-s);
	int lb=1,rb=1+(s-1)*n;
	if(1ll*(lb+rb)*s/2>d[1])return 1;
	return 0;
}
bool check2(int s)
{
	d[n]=v[n];for(int i=n-1;i>=1;i--)d[i]=min(v[i],d[i+1]-s);
	int lb=1,rb=1+(s-1)*n;
	if(1ll*(lb+rb)*s/2>d[1])return 0;
	if(!s)return 0;
	int rb2=m-n+1,lb2=m-n+1-(s-1)*n;
	long long fuc=1ll*(lb2+rb2)*s/2,tp1=lb2-(fuc-d[1]);
	if(fuc<d[1])return 1;
	return 0;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	int lb=1,rb=m/n,st1=m/n+1;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check1(mid))st1=mid,rb=mid-1;
		else lb=mid+1;
	}
	lb=1,rb=st1-1;int st2=0;st1--;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check2(mid))st2=mid,lb=mid+1;
		else rb=mid-1;
	}
	long long as=max(0ll,max(solve(st2-1),max(solve(st2),solve(st2+1))));
	lb=st2,rb=st1;
	while(lb<rb)
	{
		int mid1=(lb+rb)>>1,mid2;
		if(mid1==rb)mid2=mid1,mid1=mid2-1;
		else mid2=mid1+1;
		long long v1=solve(mid1),v2=solve(mid2);
		if(v2>=v1)as=max(as,v2),lb=mid1+1;
		else rb=mid2-1;
	}
	as=max(as,solve(lb));
	as=max(as,solve(rb));
	printf("%lld\n",as);
}
```



##### auoj4 绰绰有余

###### Problem

有一个 $2\times (n+1)$ 的网格图，现在删掉了网格图的最右侧一条纵向边，图中还剩下 $3n$ 条边。

你有 $m$ 条链，第 $i$ 条链由 $a_i$ 条边组成，你需要用这些链覆盖网格图中的这 $3n$ 条边，且每条边正好被覆盖一次。

求是否存在合法方案，如果存在构造任意一组合法方案。

$n,m\leq 10^5$

$1s,1024MB$

###### Sol

每条链最多会贡献两个奇度数的点，总共有 $2n$ 个奇度数点，因此 $m<n$ 无解，同时显然 $\sum a_i\neq 3n$ 无解。

对于链长 $\leq 3$ 的可以直接放，对于一条链长 $>3$ 的链 $k$ ，考虑把它和 $a$ 条 $1$ 和 $b$ 条 $2$ 拼成合法的，那么一定有：

$$
k+a+2b=3(1+a+b)\\2a+b=k-3
$$

如果找到了一组 $(a,b)$，可以考虑如下构造：

```
--------
|
-----
```

其中左侧中间部分用长度为 $1$ 的链补上，后面用长度为 $2$ 的链补上。

如果无解，说明剩下短链中没有长度为 $2$ 的链，且 $k$ 为偶数。

如果有两条偶数的链，设长度为 $j,k$，考虑 $a+2j+2k=3(a+2)$，取 $a=j+k-3$ 即可，一种构造如下：

```
-------- ----
|     |  |  |
----- ----  ----
```

即一条链绕成之前的形状，两条边长度差 $1$，另外一条边从这个终点开始向右上下摆动。中间的部分用 $1$ 填。

如果只剩一条偶数的链，此时显然 $n<m$，考虑将这条链绕一圈，然后用剩下的短链去补空位即可，这种构造类似于第一种。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<queue>
using namespace std;
#define N 105000
queue<int> st[4];
int n,m,v[N],as[N][3],su,ct=0,tp;
void doit1(int a)
{
	int s=v[a]-3;ct++;as[ct][1]=as[ct][0]=as[ct][2]=a;
	while(s>1&&st[0].size())++ct,as[ct][0]=as[ct][2]=a,as[ct][1]=st[0].front(),st[0].pop(),s-=2;
	while(s&&st[1].size())++ct,as[ct][0]=a,as[ct][1]=as[ct][2]=st[1].front(),st[1].pop(),s--;
}
void doit2(int a,int b)
{
	int s=v[a]-3,v2=0;ct++;as[ct][1]=as[ct][0]=as[ct][2]=a;
	while(s>1&&st[0].size())++ct,as[ct][0]=as[ct][2]=a,as[ct][1]=st[0].front(),st[0].pop(),s-=2;
	ct++;as[ct][0]=a;as[ct][1]=as[ct][2]=b;s=v[b]-2;
	while(s>1&&st[0].size())++ct,as[ct][v2]=as[ct][v2+1]=b,as[ct][(!v2)*2]=st[0].front(),st[0].pop(),s-=2,v2^=1;
}
void doit3(int a)
{
	int s=v[a]-3;ct++;as[ct][1]=as[ct][0]=as[ct][2]=a;
	while(s>1&&st[0].size())++ct,as[ct][0]=as[ct][2]=a,as[ct][1]=st[0].front(),st[0].pop(),s-=2;
	++ct;as[ct][0]=a;as[ct][1]=st[0].front(),st[0].pop();as[ct][2]=st[0].front(),st[0].pop();
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)
	{
		scanf("%d",&v[i]);
		if(v[i]==1)st[0].push(i);
		else if(v[i]==2)st[1].push(i);
		else if(v[i]&1)st[2].push(i);
		else st[3].push(i);
		su+=v[i];
		if(su>n*3){printf("no\n");return 0;}
	}
	if(su!=n*3||m<n){printf("no\n");return 0;}
	printf("yes\n");
	while(!st[2].empty())doit1(st[2].front()),st[2].pop();
	while(!st[3].empty())if(st[1].size())doit1(st[3].front()),st[3].pop();
	else if(st[3].size()>1)tp=st[3].front(),st[3].pop(),doit2(tp,st[3].front()),st[3].pop();
	else doit3(st[3].front()),st[3].pop();
	while(ct<n)
	{
		if(st[1].size()>=3)
		{
			int v1=st[1].front();st[1].pop();
			int v2=st[1].front();st[1].pop();
			int v3=st[1].front();st[1].pop();
			as[ct+1][0]=as[ct+1][1]=v1;as[ct+2][0]=as[ct+2][1]=v2;as[ct+1][2]=as[ct+2][2]=v3;ct+=2;
		}
		else if(!st[1].size())
		{
			int v1=st[0].front();st[0].pop();
			int v2=st[0].front();st[0].pop();
			int v3=st[0].front();st[0].pop();
			as[ct+1][0]=v3;as[ct+1][1]=v1;as[ct+1][2]=v2;ct++;
		}
		else
		{
			int v1=st[0].front();st[0].pop();
			int v2=st[1].front();st[1].pop();
			as[ct+1][0]=as[ct+1][1]=v2;as[ct+1][2]=v1;ct++;
		}
	}
	for(int i=1;i<=3;i++,printf("\n"))
	for(int j=1;j<=n;j++)
	printf("%d ",as[j][i-1]);
}
```



#### SCOI2020模拟?

出题人:ywh&bh(如果我没记错)

##### auoj5 Magnolia

###### Problem

给一个长度为 $n$ 的序列，求它有多少个本质不同的子串，满足子串的元素和在 $[L,R]$ 之间

$n\leq 5\times 10^5,|v|\leq 10^9,|L|,|R|\leq 10^{18}$

$4s,1024MB$

###### Sol

首先建 `SA`，那么本质不同的子串为若干组给定 $l$, 且 $r$ 在一段后缀内的串。

记录前缀和，对于一个 $l$，相当于求 $su_{l-1}+L\leq su_r\leq su_{l-1}+R$ 的 $r$ 的数量，这相当于二维数点，离线后离散化依次加入，维护 `BIT` 即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<map>
#include<algorithm>
using namespace std;
#define N 505050
map<int,int> mp;
int n,v[N],s[N],ct,f[N],t[N],tr[N];
long long su[N],l,r,su2[N],as;
bool cmp(int a,int b){return f[a]>f[b];}
struct SA{
	int a[N],b[N*2],sa[N],vl[N],n,m,su[N],rk[N];
	void pre()
	{
		for(int i=1;i<=n;i++)su[a[i]=s[i]]++;
		for(int i=1;i<=m;i++)su[i]+=su[i-1];
		for(int i=n;i>=1;i--)sa[su[a[i]]--]=i;
		for(int l=1;l<=n;l<<=1)
		{
			int ct=0;
			for(int i=n-l+1;i<=n;i++)b[++ct]=i;
			for(int i=1;i<=n;i++)if(sa[i]>l)b[++ct]=sa[i]-l;
			for(int i=1;i<=m;i++)su[i]=0;
			for(int i=1;i<=n;i++)su[a[i]]++;
			for(int i=1;i<=m;i++)su[i]+=su[i-1];
			for(int i=n;i>=1;i--)sa[su[a[b[i]]]--]=b[i];
			for(int i=1;i<=n;i++)b[i]=a[i];
			ct=1;a[sa[1]]=1;
			for(int i=2;i<=n;i++)a[sa[i]]=(b[sa[i]]==b[sa[i-1]]&&b[sa[i]+l]==b[sa[i-1]+l])?ct:++ct;
			m=ct;if(m==n)break;
		}
		for(int i=1;i<=n;i++)rk[sa[i]]=i;
		int fu=0;s[n+1]=1e9+9;
		for(int i=1;i<=n;i++)
		{
			f[i]=i;
			if(fu)fu--;
			if(rk[i]==1)continue;
			while(s[i+fu]==s[sa[rk[i]-1]+fu])fu++;
			f[i]=i+fu;
		}
	}
}sa;
void add(int x,int k){for(int i=x;i<=n;i+=i&-i)tr[i]+=k;}
int que(int x){if(x>n)x=n;int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
int main()
{
	scanf("%d%lld%lld",&n,&l,&r);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),s[i]=mp[v[i]]?mp[v[i]]:(mp[v[i]]=++ct),t[i]=i,su[i]=su[i-1]+v[i],su2[i]=su[i];
	sa.n=n;sa.m=ct;
	sa.pre();sort(t+1,t+n+1,cmp);sort(su2+1,su2+n+1);
	int las=n;
	for(int i=1;i<=n;i++)
	{
		while(f[t[i]]<=las)add(lower_bound(su2+1,su2+n+1,su[las])-su2,1),las--;
		long long v1=l+su[t[i]-1],v2=r+su[t[i]-1]+1;
		int l1=lower_bound(su2+1,su2+n+1,v1)-su2,r1=lower_bound(su2+1,su2+n+1,v2)-su2-1;
		if(l1>r1)continue;
		as+=que(r1)-que(l1-1);
	}
	printf("%lld\n",as);
}
```



##### auoj6 Myosotis

###### Problem

有一个长度为 $n$ 的序列，定义一次操作为在序列中选出两个数，分数为前面那个数的值减去后面那个数的值，然后删去这两个数

对于每一个 $k$ ，求出最多进行 $k$ 次操作得到的最大分数

$n\leq 5\times 10^5$

$5s,256MB$

###### Sol

如果一个数被作为第一个数和第二个数分别选了一次，可以看成没选过这个数，因此可以看成在每个数能分别作为第一个数和第二个数一次的情况下求操作 $k$ 次的最大分数。

考虑一个费用流，原点向每个点连费用为这个点权值，流量 $1$ 的边，每个点向它右边的点连费用 $0$，流量 $+\infty$ 的边，每个点向汇点连费用为这个点权值的相反数，流量 $1$ 的边，进行 $k$ 次操作的答案即为流量为 $k$ 的最大费用流

考虑模拟增广，每次相当于选出一对数 $(i,j)$，使得 $i<j$ 或者 $i>j$ 并且 $(i,j)$ 间的边的流量全部大于 $0$，且 $a_i-a_j$ 尽量大，然后给这个区间的边流量加一或减一

因为每个点只会选一次，所以可以把流量放到点上

使用线段树维护，对于一个区间，记录这个区间内点的最小流量，以及如果将这个区间内流量最小的点看成流量为 $0$ 的点时，这个区间内部的答案，与区间左端点连通的部分的点的最大值和最小值，与区间有端点连通的部分的点的最大值和最小值，整个区间的点的最大值和最小值，以及这些值的位置

合并时如果两边最小值一样可以直接将左边的右侧和右边的左侧合并起来，否则 $\min$ 更大的那一边没有限制，如果右边 $\min$ 更大，可以将左边的右侧和整个右边合并起来，另外一种情况同理，合并时更新答案

最后如果整体的 $\min>0$，那么整个区间都可以互相转移，可以用整个区间内选择两个数的结果更新答案。

标记可以直接下传，复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505050
int n,v[N];
long long as;
struct sth{int mx,mn,f1,f2;};
sth doit(sth a,sth b){sth c=a;if(b.mx>c.mx)c.mx=b.mx,c.f1=b.f1;if(b.mn<c.mn)c.mn=b.mn,c.f2=b.f2;return c;}
sth doit2(sth a,sth b){if(a.mx-a.mn<b.mx-b.mn)return b;return a;}
struct segt{
	struct node{int l,r,mn,lz;sth lb,rb,su,as;}e[N*4];
	void pushup(int x)
	{
		if(e[x<<1].mn<e[x<<1|1].mn)
		{
			e[x].mn=e[x<<1].mn;
			e[x].lb=e[x<<1].lb;
			e[x].rb=doit(e[x<<1|1].su,e[x<<1].rb);
			e[x].su=doit(e[x<<1].su,e[x<<1|1].su);
			e[x].as=doit2(doit2(e[x<<1].as,e[x<<1|1].as),e[x].rb);
		}
		else if(e[x<<1].mn>e[x<<1|1].mn)
		{
			e[x].mn=e[x<<1|1].mn;
			e[x].rb=e[x<<1|1].rb;
			e[x].lb=doit(e[x<<1].su,e[x<<1|1].lb);
			e[x].su=doit(e[x<<1].su,e[x<<1|1].su);
			e[x].as=doit2(doit2(e[x<<1].as,e[x<<1|1].as),e[x].lb);
		}
		else
		{
			e[x].mn=e[x<<1].mn;
			e[x].lb=e[x<<1].lb;
			e[x].rb=e[x<<1|1].rb;
			e[x].su=doit(e[x<<1].su,e[x<<1|1].su);
			e[x].as=doit2(doit2(e[x<<1].as,e[x<<1|1].as),doit(e[x<<1].rb,e[x<<1|1].lb));
		}
		sth su=(sth){e[x<<1|1].su.mx,e[x<<1].su.mn,e[x<<1|1].su.f1,e[x<<1].su.f2};
		e[x].as=doit2(e[x].as,su);
	}
	void pushdown(int x){e[x<<1].lz+=e[x].lz;e[x<<1].mn+=e[x].lz;e[x<<1|1].mn+=e[x].lz;e[x<<1|1].lz+=e[x].lz;e[x].lz=0;}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].lb=e[x].rb=e[x].as=(sth){-1000000000,1000000000,-1,-1};e[x].su=(sth){v[l],v[l],l,l};return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int s)
	{
		if(e[x].l==e[x].r){e[x].lb=e[x].rb=e[x].as=e[x].su=(sth){-1000000000,1000000000,-1,-1};return;}
		int mid=(e[x].l+e[x].r)>>1;pushdown(x);
		if(mid>=s)modify(x<<1,s);else modify(x<<1|1,s);pushup(x);
	}
	void modify2(int x,int l,int r,int k)
	{
		if(e[x].l==l&&e[x].r==r){e[x].mn+=k;e[x].lz+=k;return;}
		int mid=(e[x].l+e[x].r)>>1;pushdown(x);
		if(mid>=r)modify2(x<<1,l,r,k);
		else if(mid<l)modify2(x<<1|1,l,r,k);
		else modify2(x<<1,l,mid,k),modify2(x<<1|1,mid+1,r,k);
		pushup(x);
	}
	void dfs(int x){if(e[x].l==e[x].r)return;pushdown(x);dfs(x<<1);dfs(x<<1|1),pushup(x);}
}tr;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	tr.build(1,1,n);
	for(int i=1;i<=n/2;i++)
	{
		sth fu=tr.e[1].as;
		if(tr.e[1].mn>0)fu=doit(tr.e[1].su,fu);
		if(fu.mx>fu.mn)
		as+=fu.mx-fu.mn;
		tr.modify(1,fu.f1);tr.modify(1,fu.f2);
		if(fu.f1>fu.f2)tr.modify2(1,fu.f2+1,fu.f1,1);
		else if(fu.f1<fu.f2) tr.modify2(1,fu.f1+1,fu.f2,-1);
		printf("%lld ",as);
	}
}
```



##### auoj7 Marigold

###### Problem

使用如下方式定义序列 $f$：

1. $f_1=K$
2. $f_n=A*f_{n-1}+B*\sum_{i=1}^{n-1}f_if_{n-i}(n>1)$

现在给出 $q$ 组询问，每次给出 $l,r(r\leq n)$，你需要求出 $\sum_{i=l}^rf_i^2$。答案模 $998244353$。

$n\leq 2\times 10^7,q\leq 5\times 10^4,1\leq K,B,0\leq A$

$3s,1024MB$

###### Sol

考虑将 $f$ 看成生成函数，则可以发现 $f$ 满足如下性质：
$$
f(x)=Ax*f(x)+B*f^2(x)+Kx
$$
则解得：
$$
f(x)=\frac 1{2B}*(1-Ax\pm\sqrt{(Ax-1)^2-4BKx})
$$
考虑 $0$ 次项的系数，可以发现应该取减号。

只需要求出这个二次式开根的结果即可，那么可以使用 `exp` 的 `ODE` 形式，即：
$$
f(x)=g^k(x)\\
f'(x)=k*g'(x)*g^{k-1}(x)\\
f'(x)g(x)=kg'(x)f(x)
$$
这里 $k,g$ 已知，这是一个关于 $f$ 的ODE，可以发现比较 $x^n$ 的系数，可以用前面的项求出 $f$ 的 $x+1$ 次项，因此可以写出递推式。预处理逆元即可。最后使用前缀和即可回答询问。

复杂度 $O(n+q)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 20509018
#define mod 998244353
int n,k,a,b,f[N],q,l,r,fr[N],ifr[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d%d",&n,&k,&a,&b);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*i*fr[i-1]%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	int b1=mod-(4ll*k*b+2*a)%mod,a1=1ll*a*a%mod;
	f[0]=mod-1;
	for(int i=0;i<n;i++)
	{
		int su=1ll*f[i]*b1%mod*(mod+1)/2%mod;
		if(i)su=(su+1ll*a1*f[i-1])%mod;
		su=(su+mod-1ll*i*b1%mod*f[i]%mod)%mod;
		if(i)su=(su+mod-1ll*(i-1)*a1%mod*f[i-1]%mod)%mod;
		f[i+1]=1ll*su*fr[i]%mod*ifr[i+1]%mod;
	}
	f[0]=0;f[1]=(f[1]+mod-a)%mod;
	int ir=pw(2*b,mod-2);
	for(int i=1;i<=n;i++)f[i]=1ll*f[i]*ir%mod;
	for(int i=1;i<=n;i++)f[i]=(f[i-1]+1ll*f[i]*f[i])%mod;
	scanf("%d",&q);
	while(q--)scanf("%d%d",&l,&r),printf("%d\n",(f[r]+mod-f[l-1])%mod);
}
```



#### SCOI2020模拟?

出题人:ljz

##### auoj8 魔王的行径

###### Problem

有 $2^n$ 个人进行比赛，比赛规则如下：

```
第一轮,在胜者组的一次比赛后,赢得比赛的人留在胜者组,输的人进入败者组
称一个组的一次比赛为,将组内所有选手按照编号排序,然后第1名和第2名比赛,比赛的编号为1,第3名和第4名比赛,比赛的编号为2...
之后的每一轮比赛情况如下
初始胜者组和败者组都有2^k位选手
败者组进行一次比赛,赢的人留在败者组,输的人被淘汰
胜者组进行一次比赛,赢的人留在胜者组
胜者组比赛中输掉x号比赛的人和败者组中赢得x号比赛的人比赛,赢的人留在(/进入)败者组,输的人被淘汰
最后胜者组和败者组各留下x位选手
如此比赛直到胜者组和败者组各留下一名选手,然后她们之间进行比赛
```

给定 $m$ 名选手，你可以决定每一场比赛的胜负，求最多有多少场比赛满足比赛中至少有一名给定选手参加

$n\leq 17$

$1s,512MB$

###### Sol

比赛可以看成一个线段树的结构，在第 $i$ 轮比赛后，线段树上一个长度为 $2^i$ 的区间内的人只会留下胜者组/败者组各一人，且同一层不同两个区间间的情况互不影响

设 $dp_{i,0/1,0/1}$ 表示考虑线段树上一个点内部，最后留下来的胜者组/败者组的人是不是给定选手时的最优答案，转移直接枚举这个点的两场比赛的胜负即可。

复杂度 $O(2^n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 530000
int dp[N][2][2],n,k,v[N],a;
void doit(int x,int l,int r)
{
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)dp[x][i][j]=-1e9;
	if(l+1==r){if(v[l]+v[r]==1)dp[x][0][1]=dp[x][1][0]=1;else if(v[l]+v[r]==0)dp[x][0][0]=0;else dp[x][1][1]=1;return;}
	int mid=(l+r)>>1;doit(x<<1,l,mid);doit(x<<1|1,mid+1,r);
	for(int v1=0;v1<2;v1++)
	for(int v2=0;v2<2;v2++)
	for(int v3=0;v3<2;v3++)
	for(int v4=0;v4<2;v4++)
	dp[x][v1|v3][v2|v4]=max(dp[x][v1|v3][v2|v4],dp[x<<1][v1][v2]+dp[x<<1|1][v3][v4]+(v1|v3)+2*(v2|v4));
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=k;i++)scanf("%d",&a),v[a]=1;
	doit(1,1,1<<n);
	printf("%d\n",max(max(dp[1][0][0],dp[1][0][1]+1),max(dp[1][1][0]+1,dp[1][1][1]+1)));
}
```



##### auoj9 在世界中心呼唤爱的野兽

咕了

##### auoj10 没有人受伤的世界

###### Problem

给一个两边各 $n$ 个点的二分图，每条边有一定概率出现，求存在完美匹配的概率，模 $10^9+7$

$n\leq 7$

$6s,256MB$

###### Sol

在一个完美匹配中，左边的 $i$ 个点一定对应了右边的 $i$ 个点。

考虑对于左边的 $i$ 个点，记录它们能与右边的哪些 $i$ 个点的集合匹配

这样的状态数看起来有 $2^{35}$ ，但实际上大概在 $30000$ 左右

设 $dp_{i,j}$ 表示考虑了左边前 $i$ 个点，这些点能匹配右边的哪些点集，这种情况的概率，暴力转移即可。

复杂度 $O(n^2*2^n*f(n))$，$f(n)$ 是状态数。

###### Code

```cpp
#include<cstdio>
#include<map>
using namespace std;
#define mod 1000000007
map<__int128,int> v[11];
__int128 st[8][40100];
int dp[8][40100],n,p[8][8],ct[8];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	scanf("%d",&p[i][j]),p[i][j]=1ll*p[i][j]*570000004%mod;
	st[0][1]=1;ct[0]=1;dp[0][1]=1;
	for(int i=1;i<=n;i++)
	for(int j=0;j<1<<n;j++)
	{
		int s1=1;
		for(int k=1;k<=n;k++)if(j&(1<<k-1))s1=1ll*s1*p[i][k]%mod;else s1=1ll*s1*(mod+1-p[i][k])%mod;
		for(int k=1;k<=ct[i-1];k++)
		{
			__int128 nt=0;
			for(int l=0;l<1<<n;l++)
			if(st[i-1][k]&(((__int128)1)<<l))
			for(int s=1;s<=n;s++)if(!(l&(1<<s-1))&&(j&(1<<s-1)))nt|=((__int128)1)<<(l|(1<<s-1));
			if(nt==0)continue;
			if(!v[i][nt])v[i][nt]=++ct[i],st[i][ct[i]]=nt;
			int tp=v[i][nt];
			dp[i][tp]=(dp[i][tp]+1ll*dp[i-1][k]*s1)%mod;
		}
	}
	printf("%d\n",dp[n][1]);
}
```



#### SCOI2020模拟?

出题人:lsj

##### auoj11 废墟

###### Problem

给定 $n,k$ ，求有多少个 $1,...,n$ 的排列满足相邻两个元素的和不超过 $k$ ，模 $998244353$

$n\leq 10^6,n<k<2n$

$1s,512MB$

###### Sol

设 $f_{i,j}$ 表示 $i$ 个数的排列，前 $j$ 个数没有限制，第 $j+1$ 个数不能和最大的数相邻，第 $j+2$ 个数不能和最大的两个数相邻，以此类推的方案数，显然答案为 $dp_{n,k-n}$

如果 $i=j,j+1$ 显然方案数为 $i!$

否则，考虑 $i$ 所在的位置：

如果 $i$ 在两侧，那么它旁边的数只有 $j$ 种方案，而删除这两个数后剩下的数与这两个数互不影响，新的状态可以看成删去一个最小的元素，删去一个最大的元素，剩余元素限制不变。因此方案数为 $2j*f_{i-2,j}$

否则，它两侧的数有 $j(j-1)$ 种方案，考虑将这三个元素合并为一个元素，因为原先两侧的数无论与谁相邻都不会违反限制，可以看成新出现的数也不会导致违反限制。此时新的状态也可以看成删去一个最小的元素，删去一个最大的元素，剩余元素限制不变，因此这种情况的方案数为 $j(j-1)*f_{i-2,j}$

因此有 $f_{i,j}=j(j+1)f_{i-2,j}(i\geq j+2)$，可以直接递推，复杂度 $O(n)$ ~~打表容易发现结论~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define mod 998244353
int n,m,as=1;
int main()
{
	scanf("%d%d",&n,&m);
	int tp=m-n;
	for(int i=1;i<=tp;i++)as=1ll*as*i%mod;
	for(int i=tp+1;i<=n;i++)as=1ll*as*(tp+((i-tp)&1))%mod;
	printf("%d\n",as);
}
```



##### auoj12 魔法

###### Problem

有一棵 $n$ 个点的树，点权构成一个 $n$ 阶排列。

有 $q$ 次修改，每次交换两个点的点权，你需要输出每次修改后有多少个 $i$ 满足所有点权小于等于 $i$ 的点构成一条链

$n,q\leq 5\times 10^5$

$3s,512MB$

###### Sol

将限制分成形成连通块和每个点度数不超过 $2$

对于第二个限制，只需要求出每个点相邻的点中的第三小点权即可

考虑分成两部分，第一部分是儿子中的第三小点权，第二部分是儿子中的第二小点权和父亲点权的 $\max$。

对于第一部分，每个点使用 `set` 维护儿子点权即可，修改时改父亲处的 `set`。

对于第二部分，对每个点再开一个 `set`，将一个点的第二小的儿子点权放到父亲的 `set` 里面，修改时先重新计算这个点作为父亲时的 $\min$，然后求它父亲的第二小儿子点权，然后更新它父亲的父亲。这部分使用 `set` 维护所有这些值的 $\min$ 即可

对于第一个限制，考虑点减边容斥，只需要求出每条边两边点权的 $\min$。

注意到因为要求是链，每个点只有点权最小的两条出边是有用的，剩下的边可以直接忽略。因此暴力改影响的 $O(1)$ 条边即可，注意细节。

最后的问题相当于区间加区间减，求区间内值为 $1$ 的位置个数，且保证任意时刻每个值大于等于 $1$。因此可以使用线段树，维护区间内最小值以及最小值出现的次数，显然标记下传不影响记录的这个值。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
#pragma GCC optimize(3)
using namespace std;
#define N 500500
int head[N],cnt,v1[N],p[N],tid[N],f[N],id2[N],n,a,b,v2[N],s[N][2],q,g2[N],is[N],g3[N],as1,tr1[N];
struct edge{int t,next,id;}ed[N*2];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],id};head[t]=cnt;}
void dfs(int u,int fa){f[u]=fa;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)id2[ed[i].t]=ed[i].id,dfs(ed[i].t,u);}
set<int> st[N];
multiset<int> st2[N];
struct sth{int a,b;};
sth doit(sth a,sth b){if(a.a<b.a)return a;if(a.a>b.a)return b;return (sth){a.a,a.b+b.b};}
struct lsjtree{
	struct node{int l,r,lz;sth tp;}e[N*4];
	void pushup(int x){e[x].tp=doit(e[x<<1].tp,e[x<<1|1].tp);}
	void pushdown(int x){if(e[x].lz)e[x<<1].tp.a+=e[x].lz,e[x<<1].lz+=e[x].lz,e[x<<1|1].tp.a+=e[x].lz,e[x<<1|1].lz+=e[x].lz,e[x].lz=0;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r){e[x].tp.a=l;e[x].tp.b=1;return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
	void modify(int x,int l,int r,int v){if(e[x].l==l&&e[x].r==r){e[x].lz+=v;e[x].tp.a+=v;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify(x<<1,l,r,v);else if(mid<l)modify(x<<1|1,l,r,v);else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);pushup(x);}
	void query(int x,int l,int r){if(l>r)return;if(e[x].l==l&&e[x].r==r){as1+=(e[x].tp.a==1)*e[x].tp.b;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)query(x<<1,l,r);else if(mid<l)query(x<<1|1,l,r);else query(x<<1,l,mid),query(x<<1|1,mid+1,r);}
}tr;
void add(int x,int k){if(!x)return;for(int i=x;i<=n;i+=i&-i)tr1[i]+=k;}
int que()
{
	int as=0;
	for(int k=18;k>=0;k--)if((as+(1<<k))<=n&&tr1[as+(1<<k)]==0)as+=1<<k;
	return as;
}
void doit2(int x)
{
	if(!x)return;
	if(v2[x]==max(p[s[x][0]],p[s[x][1]]))return;
	int v1=v2[x],v3=max(p[s[x][0]],p[s[x][1]]);
	v2[x]=max(p[s[x][0]],p[s[x][1]]);
	if(v1>v3)tr.modify(1,v3,v1-1,-1);
	else tr.modify(1,v1,v3-1,1);
}
void just_doit(int x)
{
	set<int>::iterator it=st[x].begin();
	int v3=1e9,v2=1e9+1,st3=1e9+2;
	v3=*it;
	if(v3<1e9)
	{
		it++,v2=*it;
		if(v2<1e9)
		it++,st3=*it;
		else st3=1e9+1;
	}
	if(v1[x]!=st3)add(v1[x],-1),add(v1[x]=st3,1);
	if(v3<=n)doit2(id2[tid[v3]]);
	if(v2<=n)doit2(id2[tid[v2]]);
	doit2(id2[x]);
	int v12=max(p[x],*st2[x].begin());
	if(v12!=g3[x])add(g3[x],-1),add(g3[x]=v12,1);
}
void make_your_dream_come_true(int x)
{
	if(!x)return;
	set<int>::iterator it=st[x].begin();
	int v3=1e9,v2=1e9+1,st3=1e9+2;
	v3=*it;
	if(v3<1e9)
	{
		it++,v2=*it;
		if(v2<1e9)
		it++,st3=*it;
		else st3=1e9+1;
	}
	if(v1[x]!=st3)add(v1[x],-1),add(v1[x]=st3,1);
	if(v3<=n)doit2(id2[tid[v3]]);
	if(v2<=n)doit2(id2[tid[v2]]);
	doit2(id2[x]);
	if(!f[x])return;
	if(g2[x]!=v2)st2[f[x]].erase(st2[f[x]].find(g2[x])),g2[x]=v2,st2[f[x]].insert(g2[x]);
	int v12=max(p[f[x]],*st2[f[x]].begin());
	if(v12!=g3[f[x]])add(g3[f[x]],-1),add(g3[f[x]]=v12,1);
}
int main()
{
	scanf("%d",&n);tr.build(1,1,n);
	for(int i=1;i<=n;i++)scanf("%d",&p[i]),tid[p[i]]=i,st[i].insert(1e9),st2[i].insert(1e9);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),s[i][0]=a,s[i][1]=b,adde(a,b,i);
	dfs(1,0);
	for(int i=2;i<=n;i++)st[f[i]].insert(p[i]);
	for(int i=1;i<n;i++)v2[i]=max(p[s[i][0]],p[s[i][1]]),tr.modify(1,v2[i],n,-1);
	for(int i=1;i<=n;i++)
	{
		set<int>::iterator it=st[i].begin();
		int v3=1e9,v2=1e9+1,st3=1e9+2;
		v3=*it;
		if(v3<1e9)
		{
			it++,v2=*it;
			if(v2<1e9)
			it++,st3=*it;
			else st3=1e9+1;
		}
		v1[i]=st3;add(st3,1);
		if(i==1)continue;
		g2[i]=v2;st2[f[i]].insert(g2[i]);
	}
	for(int i=1;i<=n;i++)add(g3[i]=max(p[i],*st2[i].begin()),1);
	scanf("%d",&q);
	while(q--)
	{
		scanf("%d%d",&a,&b);
		if(a==b){as1=0;tr.query(1,1,que());printf("%d\n",as1);continue;}
		if(f[a]!=f[b])
		{
			if(a>1)st[f[a]].erase(p[a]),st[f[a]].insert(p[b]);
			if(b>1)st[f[b]].erase(p[b]),st[f[b]].insert(p[a]);
		}
		p[a]^=p[b]^=p[a]^=p[b];tid[p[a]]^=tid[p[b]]^=tid[p[a]]^=tid[p[b]];
		just_doit(a);just_doit(b);
		make_your_dream_come_true(f[a]);make_your_dream_come_true(f[b]);
		as1=0;tr.query(1,1,que());printf("%d\n",as1);
	}
}
```



##### auoj13 风暴

###### Problem

给一个 $n\times m$ 的网格图，每条边有 $p$ 的概率被删掉，求最后 $(1,1),(n,m)$ 连通的概率，误差不超过 $10^{-6}$ ，多组数据

$n\leq 8,m\leq 10^{18},T\leq 50,p>0.1$

$3s,256MB$

###### Sol

考虑轮廓线 `dp`，只需要记录当前的轮廓线的连通情况（类似于集合划分）以及哪一个连通块与起点相连，这样的状态数不多（不超过 $Bell(n)*n$），由于图是平面图实际的状态数更少。

设 $dp_{i,j,S}$ 表示考虑到了 $(i,j)$，当前轮廓线状态为 $S$ 的方案数，转移时枚举这个点向上和向左的边是否连通可以直接转移，注意边界情况。

设 $as_k$ 表示 $m=k$ 时的答案，当 $m$ 很大时发现 $\frac {as_{m+1}}{as_m}$ 接近于一个定值，可以先算 $100$ 项然后快速幂算后面，精度误差在合理范围内。

复杂度 $O(Tn^2Bell(n)*100)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<cmath>
using namespace std;
#define N 3500
int id1[20767676],id2[N][11],T,n,m,st2[N],ct,id3[N],trans[N][11][2][2],las;
double dp[250][9][N],p;
long long fuc;
int justdoit()
{
	int id[21]={0,1,0},ct3=1;
	for(int i=1;i<=n;i++)if(!id[st2[i]])id[st2[i]]=++ct3,st2[i]=ct3;
	else st2[i]=id[st2[i]];
	int fg3=0;
	for(int i=1;i<=n;i++)if(st2[i]==1)fg3=1;
	if(!fg3)for(int i=1;i<=n;i++)st2[i]--;
	int ha1=0;
	for(int j=1;j<=n;j++)ha1=ha1*9+st2[j];
	if(!id1[ha1])
	{
		for(int j=1;j<=n;j++)id2[ct+1][j]=st2[j];
		id1[ha1]=++ct;id3[ct]=ha1;
	}
	return id1[ha1];
}
void fuckthisdp()
{
	memset(dp,0,sizeof(dp));p=1-p;m=100;
	if(n!=las)
	{
		for(int i=1;i<=ct;i++)id1[id3[i]]=0;ct=0;
		for(int i=1;i<=n;i++)st2[i]=i;dp[1][1][justdoit()]=1;
		memset(trans,0,sizeof(trans));
		for(int g=1;g<=ct;g++)
		for(int j=1;j<=n;j++)
		for(int k=0;k<2;k++)
		for(int l=0;l<=(j>1);l++)
		{
			for(int f=1;f<=n;f++)st2[f]=id2[g][f];
			if(!k&&!l)st2[j]=n+2;
			else if(!k)st2[j]=st2[j-1];
			else if(k&&l)
			{
				int v1=st2[j-1],v2=st2[j];
				if(v2<v1)v2^=v1^=v2^=v1;
				for(int f=1;f<=n;f++)if(st2[f]==v2)st2[f]=v1;
			}
			int fg1=0;
			for(int f=1;f<=n;f++)if(st2[f]==1)fg1=1;
			if(fg1)trans[g][j][k][l]=justdoit();
		}
	}
	las=n;
	for(int i=1;i<=n;i++)st2[i]=i;
	dp[1][1][justdoit()]=1;
	for(int i=1;i<=m;i++)
	for(int j=1;j<=n;j++)
	for(int g=1;g<=ct;g++)
	{
		if(i==1&&j==1)continue;
		int v1=j==1?i-1:i,v2=j==1?n:j-1;
		for(int k=0;k<=(i>1);k++)
		for(int l=0;l<=(j>1);l++)
		dp[i][j][trans[g][j][k][l]]+=dp[v1][v2][g]*(k?p:1-p*(i>1))*(l?p:1-p*(j>1));
	}
	double las=0,tp1,v2;
	for(int j=1;j<=m;j++)
	{
		double as=0;
		for(int i=1;i<=ct;i++)if(id2[i][n]==1)as+=dp[j][n][i];
		v2=as/las;las=as;
		if(j==fuc)tp1=as;
	}
	if(fuc>100)tp1=las*pow(v2,fuc-100);
	printf("%.10lf\n",tp1);
}
int main()
{
	scanf("%d",&T);
	while(T--)scanf("%d%lld%lf",&n,&fuc,&p),fuckthisdp();
}
```



#### SCOI2020模拟?

出题人:wkr

##### auoj14 未来

###### Problem

有一棵 $n$ 个点的有根树，每个点有一个权值 $v_i$。

对于一个 $n$ 阶排列 $p$，考虑如下操作：

从小到大考虑每个 $i$，对于一个 $i$，将这个点子树内的所有点权值加上自己当前的权值(包括自己)

定义一个排列的权值为这样操作后所有点权值的和，求出所有排列的权值和，模 $998244353$

$n\leq 10^5$，树使用 `prufer` 序方式随机生成

$1s,512MB$

###### Sol

考虑 $i$ 对 $j$ 的贡献系数，显然只有当 $i$ 是 $j$ 祖先时才有贡献，且系数只和距离有关

设路径上的点数为 $d$，将点编号为 $1,...,d$。

设初始 $v_1=1,v_2=...=v_n=0$，考虑一个排列 $p$ 的操作。设 $f_u$ 为考虑到 $u$ 时它的权值，$g_u$ 为它最后的权值，那么有：
$$
f_1=1\\
f_i=\sum_{j<i,p_j<p_i}f_j\\
g_i=2f_i+\sum_{j<i,p_j>p_i}f_j=\sum_{j<i}f_j
$$

可以发现，这相当于对于 $i<j,p_i<p_j$ 的一对点连一条有向边 $(i,j)$，$f_i$ 相当于从 $1$ 走到 $i$ 的方案数，$g_d$ 相当于从 $1$ 出发的路径数。

枚举路径经过的点数，则经过的点在排列中权值递增，因此有

$$
ans_d=n!(\sum_{i=1}^d C_{d-1}^{i-1}\frac{1}{i!})
$$

因为 `prufer` 序随机下深度是根号级别的，因此可以直接暴力求上述式子。

复杂度 $O(n\sqrt n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105000
#define mod 998244353
int n,l[N],r[N],tid[N],dep[N],head[N],cnt,as[N],fr[N],ifr[N],v[N],as1,v2,ct,a,b;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){l[u]=++ct;dep[u]=dep[fa]+1;tid[ct]=u;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);r[u]=ct;}
int main()
{
	scanf("%d",&n);
	fr[0]=ifr[0]=1;
	for(int i=1;i<=5000;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=1000;i++)
	for(int j=0;j<=i;j++)
	as[i]=(as[i]+1ll*fr[i]*ifr[i-j]%mod*ifr[j]%mod*ifr[j+1])%mod;
	as[0]=2;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&v2);
		for(int j=l[i];j<=r[i];j++)as1=(as1+1ll*v2*as[dep[tid[j]]-dep[i]])%mod;
	}
	for(int i=1;i<=n;i++)as1=1ll*as1*i%mod;
	printf("%d\n",as1);
}
```



##### auoj15 幸运

###### Problem

有一棵 $n$ 个点的有根树，给定 $q$ 次操作，操作有如下类型：

1. 在一个点下加入一个叶子节点，编号为上一个点编号 $+1$
2. 删除 $u$ 到父亲的连边
3. 撤销一个 $2$ 操作
4. 给定一个点，考虑从 $1$ 开始，每次随机向一个儿子走，走到一个叶子就返回根，走到给定点时停止。求停止时经过的边数的期望，模 $998244353$

$n,q\leq 10^6$

$7s,512MB$

###### Sol

考虑一次从 $1$ 开始，要么走到给定点，要么走到一个叶子并返回。

设 $p_i$ 表示从 $1$ 开始走，在返回之前经过 $i$ 的概率，$dis_i$ 表示 $i$ 的深度。

设给定点为 $k$，那么答案为 $dis_k+\frac{\sum_{x\in leaf,x\notin subtree\ of\ k}p_xdis_x}{1-\sum_{x\in leaf,x\notin subtree\ of\ k}p_x}$

可以发现，$\frac{1}{p_u}$ 等于它所有祖先的儿子数的乘积。

先离线处理完 $1$ 操作，然后将还没有加入的点到父亲的边看成已经被删掉了，之后的 $1$ 操作可以看成 $3$ 操作。

对于一个 $2$ 操作，它相当于删除一个子树。考虑它对 $p$ 的影响，$p$ 只和每个点的儿子数有关，而这个操作只会改变父亲的儿子数，因此可以发现这个操作会使得父亲的子树内的所有 $p$ 乘上一个值。注意此时父亲也可能成为新的叶子。

对于 $3$ 操作，相当于撤销上面的操作，倒着做即可。

考虑在 `dfs` 序上操作，则子树都是一个区间。那么需要维护数据结构，支持区间加一个删除标记，撤销区间的一个删除标记，区间乘，求区间内没有删除标记的位置的和，线段树即可维护。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2050000
#define mod 998244353
int n,m,ct,fa[N],vl[N],l[N],r[N],q[N][3],d[N],dp[N],inv[N],dep[N],head[N],cnt,tid[N],ct1;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){l[u]=++ct;tid[ct]=u;dep[u]=(dep[fa]+vl[u])%mod;dp[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);r[u]=ct;}
struct sth{int a,b,c;};
sth doit(sth a,sth b){sth c;if(a.a==b.a)c.a=a.a,c.b=(a.b+b.b)%mod,c.c=(a.c+b.c)%mod;else if(a.a<b.a)c=a;else c=b;return c;}
struct wkrtree{
	struct node{int l,r,l1,l2;sth tp;}e[N*4];
	void pushup(int x){e[x].tp=doit(e[x<<1].tp,e[x<<1|1].tp);}
	void pushdown(int x){if(e[x].l1)e[x<<1].tp.a+=e[x].l1,e[x<<1].l1+=e[x].l1,e[x<<1|1].tp.a+=e[x].l1,e[x<<1|1].l1+=e[x].l1,e[x].l1=0;
	if(e[x].l2!=1)e[x<<1].tp.b=1ll*e[x<<1].tp.b*e[x].l2%mod,e[x<<1].tp.c=1ll*e[x<<1].tp.c*e[x].l2%mod,e[x<<1].l2=1ll*e[x<<1].l2*e[x].l2%mod,
	e[x<<1|1].tp.b=1ll*e[x<<1|1].tp.b*e[x].l2%mod,e[x<<1|1].tp.c=1ll*e[x<<1|1].tp.c*e[x].l2%mod,e[x<<1|1].l2=1ll*e[x<<1|1].l2*e[x].l2%mod,e[x].l2=1;}
	void build(int x,int l,int r){e[x].l2=1;e[x].l=l;e[x].r=r;if(l==r){e[x].tp.b=dp[tid[l]];e[x].tp.c=1ll*dp[tid[l]]*dep[tid[l]]%mod;return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
	void modify1(int x,int l,int r,int v){if(e[x].l==l&&e[x].r==r){e[x].l1+=v;e[x].tp.a+=v;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify1(x<<1,l,r,v);else if(mid<l)modify1(x<<1|1,l,r,v);else modify1(x<<1,l,mid,v),modify1(x<<1|1,mid+1,r,v);pushup(x);}
	void modify2(int x,int l,int r,int v2){if(e[x].l==l&&e[x].r==r){e[x].l2=1ll*e[x].l2*v2%mod;e[x].tp.b=1ll*e[x].tp.b*v2%mod;e[x].tp.c=1ll*e[x].tp.c*v2%mod;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify2(x<<1,l,r,v2);else if(mid<l)modify2(x<<1|1,l,r,v2);else modify2(x<<1,l,mid,v2),modify2(x<<1|1,mid+1,r,v2);pushup(x);}
	sth query(int x,int l,int r){if(l>r)return (sth){1,0,0};if(e[x].l==l&&e[x].r==r)return e[x].tp;pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return doit(query(x<<1,l,mid),query(x<<1|1,mid+1,r));}
}tr;
void add(int u)
{
	int f=fa[u];
	tr.modify1(1,l[f],l[f],1);
	tr.modify1(1,l[u],r[u],-1);
	if(d[f]>0)
	{
		int tp=1ll*inv[d[f]+1]*d[f]%mod;
		tr.modify2(1,l[f],r[f],tp);
	}
	d[f]++;
}
void del(int u)
{
	int f=fa[u];
	tr.modify1(1,l[f],l[f],-1);
	tr.modify1(1,l[u],r[u],1);
	if(d[f]>1)
	{
		int tp=1ll*inv[d[f]-1]*d[f]%mod;
		tr.modify2(1,l[f],r[f],tp);
	}
	d[f]--;
}
int query(int u)
{
	if(u==1)return 0;
	sth fuc=doit(tr.query(1,1,l[u]-1),tr.query(1,r[u]+1,ct1));
	int v1=fuc.b,v2=fuc.c;
	if(fuc.a>=1)v1=v2=0;
	return (1ll*v2*pw(mod+1-v1,mod-2)%mod+dep[u])%mod;
}
int main()
{
	scanf("%d%d",&n,&m);ct1=n;
	for(int i=1;i<=n+m;i++)inv[i]=pw(i,mod-2);
	for(int i=2;i<=n;i++)scanf("%d%d",&fa[i],&vl[i]);
	for(int i=1;i<=m;i++)
	{
		scanf("%d",&q[i][0]);
		if(q[i][0]==1)scanf("%d%d",&q[i][1],&q[i][2]),fa[++ct1]=q[i][1],vl[ct1]=q[i][2];
		else scanf("%d",&q[i][1]);
	}
	for(int i=2;i<=ct1;i++)adde(i,fa[i]);
	dfs(1,0);
	tr.build(1,1,ct1);
	for(int i=2;i<=ct1;i++)tr.modify1(1,l[i],r[i],1);
	for(int i=2;i<=n;i++)add(i);
	int ct2=n;
	for(int i=1;i<=m;i++)
	if(q[i][0]==1)add(++ct2);
	else if(q[i][0]==2)del(q[i][1]);
	else if(q[i][0]==3)add(q[q[i][1]][1]);
	else printf("%d\n",query(q[i][1]));
}
```



##### auoj16 重逢

###### Problem

有 $n$ 种棋子，第 $i$ 种棋子有 $a_i$ 个。考虑这样一个取石子游戏：

两个人轮流操作，每个人每次可以取走一个石子。所有石子取完后游戏结束。

每种石子有两个属性 $p_i,v_i$。在一个游戏中，设第一个人取走了 $c_i$ 个第 $i$ 种棋子，则他的分数为 $\sum_i \lfloor\frac {c_i}{p_i}\rfloor*v_i$。

现在第一个人需要将所有石子分成大小相等的两堆，保证 $2|\sum a_i$。随后两人在第一堆上进行游戏，第一个人先手，并计算游戏的分数。接下来在第二堆上第二个人先手进行游戏。第一个人最后的分数为两轮游戏分别的分数之和。

第一个人想最大化分数，第二个人想最小化分数，求双方最优操作下最后的分数。

$\sum a_i\leq 10^6,\sum p_i\leq 2000$，所有数均为正整数。

$2s,1024MB$

###### Sol

考虑一个取石子游戏。设此时第 $i$ 种棋子有 $b_i$ 个。注意到每个人可以选择再取一个上一个人取的石子，可以发现如下性质：

1. 第一个人存在一种策略使得 $\forall i,c_i\geq \lfloor\frac{b_i}2\rfloor$。
2. 第二个人存在一种策略使得 $\forall i,c_i\leq\lceil\frac{b_i}2\rceil$。

考虑使用如下方式：

如果这是第一次操作，则任意选择一个。

否则，如果上一个人上次选择的石子还有，则选择一个这样的石子。如果没有了则任意选择一个。

可以发现只要一方使用上述策略，即可以满足对应条件。

则如果当前 $b_i$ 都是偶数，则最后的结果唯一确定。

考虑任意的情况，如果当前存在一种石子有奇数个，则如果当前人选择一种有偶数个的石子，则选择后另外一个人可以做同样的操作使得这次操作没有影响，且后手可能存在其它操作。因此先手选择偶数个的情况一定不优。

因此双方一定先操作有奇数个的种类，然后变为全是偶数的情况，可以发现第一部分为轮流选择，且如果先手选择了一个种类，最后先手可以拿到 $\frac{b_i+1}2$ 个这种棋子，否则先手可以拿到 $\frac{b_i-1}2$ 个这种棋子。

因此可以发现只有 $b_i\equiv 2p_i-1(\bmod 2p_i)$ 的情况先后拿有区别，且先手先拿能多 $c_i$ 的分数。

因此可以发现双方策略为拿满足上述条件中最大的 $c$，即将满足这个条件的 $c_i$ 从大到小排序，然后从开头开始轮流拿。

考虑将所有石子按照 $c_i$ 从大到小排序，依次考虑加入每堆石子。则可以设 $dp_{i,s,0/1,0/1}$ 表示考虑了前 $i$ 种石子，当前第一堆放了 $s$ 个，第一堆前面满足 $b_i\equiv 2p_i-1(\bmod 2p_i)$ 的个数为奇数还是偶数，第二堆满足这个条件的种类数是奇数还是偶数时，前面部分的最大分数。这样的复杂度为 $O((\sum a_i)^2)$。



注意到收益只和向一堆里面分配的数的数量模 $2p_i$ 有关，考虑变为枚举这个值。设这个值为 $r_i$，则 $r_i\in[0,2p_i-1]$，且对于一个 $r_i$，可以看成先放 $r_i$ 个进第一堆，接下来有 $\lfloor\frac{a_i-r_i}{2p_i}\rfloor$ 组石子，每组有 $2p_i$ 个，你可以任意选择一些组放进第一堆。

组数会根据 $r_i$ 变化，这难以处理，考虑看成无论如何都有 $\lfloor\frac{a_i-2p_1+1}{2p_i}\rfloor$ 组，向第一堆里面先放入的数量可以为 $[0,2p_i+(a_i\bmod 2p_i)]$。这样可以表示所有的余数，且两部分情况独立。

这样第一部分每种数的选择只有 $O(p_i)$ 个，$dp$ 复杂度为 $O((\sum p_i)^2)$。

考虑第二部分，相当于给若干个物品，求它们能不能组成某些权值。

注意到 $\sum p_i$ 很小，从而不同的石子组的大小只有 $O(\sqrt{\sum p_i})$ 个，对于每一组可以线性处理背包问题，这样这部分复杂度即为 $O(\sqrt{\sum p_i}*\sum a_i)$，可以通过。

复杂度 $O((\sum p_i)^2+\sqrt{\sum p_i}*\sum a_i)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 2050
#define M 1090180
#define ll long long
int n,m,s[N][3],p[N],ct[N],su,sp;
ll dp[N*4][2][2],nt[N*4][2][2],as,s1=-1e18;
int is[M];
bool cmp(int a,int b){return s[a][2]>s[b][2];}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]),su+=s[i][0],p[i]=i;
	sort(p+1,p+n+1,cmp);
	dp[0][0][0]=-1e18;
	for(int i=1;i<=n;i++)
	{
		int va=s[p[i]][0],vb=s[p[i]][1],vc=s[p[i]][2];
		ct[vb]+=(va-2*vb+1)/vb/2;as+=(va-2*vb+1)/vb/2*vc;
		va-=(va-2*vb+1)/vb/2*2*vb;
		for(int j=0;j<=va+sp;j++)for(int p=0;p<2;p++)for(int q=0;q<2;q++)nt[j][p][q]=-1e18;
		for(int j=0;j<=sp;j++)for(int p=0;p<2;p++)for(int q=0;q<2;q++)
		for(int s=0;s<=va;s++)
		{
			int nj=j+s,np=p,nq=q;
			ll nv=dp[j][p][q];
			if((s+1)%(vb*2)==0)nv+=(!np)*vc,np^=1;
			if((va-s+1)%(vb*2)==0)nv+=(!nq)*vc,nq^=1;
			if(s>=2*vb||va-s>=2*vb)nv+=vc;
			nt[nj][np][nq]=max(nt[nj][np][nq],nv);
		}
		sp+=va;
		for(int j=0;j<=sp;j++)for(int p=0;p<2;p++)for(int q=0;q<2;q++)dp[j][p][q]=nt[j][p][q];
	}
	su>>=1;is[0]=1;
	for(int i=1;i<=2000;i++)if(ct[i])
	for(int r=0;r<i;r++)
	{
		int ls=-1e9;
		for(int j=r;j<=(su>>1);j+=i)
		{
			if(is[j])ls=j+ct[i]*i;
			if(j<=ls)is[j]=1;
		}
	}
	for(int i=0;i<=sp&&i<=su;i++)if((su-i)%2==0&&is[(su-i)>>1])
	for(int p=0;p<2;p++)for(int q=0;q<2;q++)s1=max(s1,dp[i][p][q]);
	printf("%lld\n",as+s1);
}
```



#### SCOI2020模拟?

##### auoj17 小B的棋盘

###### Problem

棋盘上有 $n$ 枚棋子，你可以再放不超过 $k$ 枚棋子，求最多有多少个点可能成为最后所有棋子的对称中心，若有无穷个输出 $-1$，棋子可以重合但只能放在整数坐标上。

$n\leq 10^5,k\leq 20$

$1s,512MB$

###### Sol

可以发现如果确定了 $k$ 枚棋子的位置，将所有棋子按照坐标字典序排序，那么一定是排序后首尾配对，且每一对的中心都是对称中心。

如果 $n\leq k$，则对于任意点，都可以放 $n$ 个棋子对称原有棋子，使得这个点成为对称中心，因此这种情况答案为 $-1$。

否则，一定存在一对棋子是原来的棋子，因此此时答案一定有限。

因为加入棋子不超过 $k$，因此一定有一对最后配对的棋子，第一个在原序列中前 $k+1$ 个，第二个在原序列中后 $k+1$ 个，可以枚举这种情况然后判断。复杂度 $O(nk^2)$

同时可以发现每个点只可能匹配对应位置的前后 $k$ 个，因此可以枚举这些对，计算每个位置作为中心能配多少对可以做到 $O(nk\log {nk})$ 

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
using namespace std;
#define ll long long
#define N 105000
int n,k,x,y;
ll f[N];
map<ll,int> st,st2;
int main()
{
	scanf("%d%d",&n,&k);
	if(n<=k){printf("-1\n");return 0;}
	for(int i=1;i<=n;i++)
	{
		scanf("%d%d",&x,&y);
		x+=1000000000;y+=1000000000;
		f[i]=4000000000ull*x+y;
	}
	sort(f+1,f+n+1);
	for(int i=1;i<=n;i++)
	{
		st2[f[i]*2]=1;
		if(i<=k+1)
		for(int j=n-i+1-k;j<=n-i+1+k;j++)
		if(j>i&&j<=n)st[f[i]+f[j]]++;
		if(i>k+1)
		for(int j=n-i+1-k;j<=n-i+1+k;j++)
		if(j>i&&j<=n&&st.count(f[i]+f[j]))st[f[i]+f[j]]++;
	}
	int as=0;
	for(map<ll,int>::iterator it=st.begin();it!=st.end();it++)
	{
		int v1=it->second;
		if(v1<(n-k-1)/2)continue;
		int v2=st2[it->first];
		if((n+k)&1)
		{
			if(v2&&v1>=(n-k-1)/2)as++;
			if(!v2)
			{
				ll f1=it->first/4000000000ull,f2=it->first%4000000000ull;
				if((f1&1)||(f2&1))continue;
				if(v1>=(n-k+1)/2)as++;
			}
		}
		else
			if(v1>=(n-k)/2)as++;
	}
	printf("%d\n",as);
}
```



##### auoj18 小B的夏令营

###### Problem

有一个 $(n+2)\times m$ 大小的网格。

每天早上，除去第一排和最后一排外，每一排当前最左侧的格子有 $p$ 的概率被删去。每天晚上，除去第一排和最后一排外每一排当前最右侧的格子有 $p$ 的概率被删去。

求 $k$ 天后，网格连通的概率。答案模 $10^9+7$

$n,m\leq 1500,k\leq 10^5$

$1s,512MB$

###### Sol

最后每一排留下的是一个区间，且可以发现设最后一排留下的是 $[l,r]$ 的概率为 $p_{l,r}$，则 $p_{l,r}$ 可以被表示为 $v_l*v_{m-r}$ 的形式，其中 $v$ 为只考虑一侧情况的概率。

最后的图连通当且仅当任意相邻两排的区间有交。可以发现和 $[l,r]$ 相交的区间为所有区间去掉右端点小于 $l$ 的区间和左端点大于 $r$ 的区间。

因此考虑设 $f_{i,l}$ 表示前 $i$ 排连通，且最后一排区间右端点小于等于 $l$ 的概率。由对称性，可以发现下一排区间为 $[l,r]$ 且合法的概率为：

$$
p_{l,r}*(f_{i,m}-f_{i,l-1}-f_{i,m-r})
$$

考虑求 $f$ 的差分，即右端点为 $l$ 的概率，可以发现这相当于：

$$
v_{m-l}*\sum_{k\leq l}v_k*(f_{i,m}-f_{i,k-1}-f_{i,m-l})
$$

可以前缀和求出这部分，这样即可求出 $f$。最后答案为 $f_{n,m}$。

复杂度 $O(nm+k)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define M 1505
#define mod 1000000007
int n,m,a,b,k,fr[N],p[M],sul[M][M];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
int main()
{
	scanf("%d%d%d%d%d",&n,&m,&a,&b,&k),a=1ll*a*pw(b,mod-2)%mod;
	fr[0]=1;for(int i=1;i<=k;i++)fr[i]=1ll*fr[i-1]*i%mod;
	for(int i=0;i<m&&i<=k;i++)p[i]=1ll*fr[k]*pw(fr[i],mod-2)%mod*pw(fr[k-i],mod-2)%mod*pw(a,i)%mod*pw(mod+1-a,k-i)%mod;
	sul[0][m]=1;
	for(int i=1;i<=n;i++)
	for(int j=1,v1=0,v2=0;j<=m;j++)
	v1=(v1+p[j-1])%mod,v2=(v2+1ll*p[j-1]*sul[i-1][j-1])%mod,sul[i][j]=(sul[i][j-1]+1ll*p[m-j]*(sul[i-1][m]-sul[i-1][m-j]+mod)%mod*v1%mod+mod-1ll*p[m-j]*v2%mod)%mod;
	printf("%d\n",sul[n][m]);
}
```



##### auoj19 小B的图

###### Problem

有 $n$ 个点 $m$ 条边，每条边的权值为 $v_i+x$ 或者 $v_i-x$，其中 $x$ 为一个变量

多组询问，每次给定一个 $x$ 的值，求此时的最小生成树边权和

$n,m,q\leq 2\times 10^5$

$1s,512MB$

###### Sol

假设当前求出了一个 $x$ 时的生成树，当 $x$ 减小时，通过生成树的贪心做法可以看出，当前生成树中 $v_i+x$ 类型的边之后一定会被选，不在生成树中的 $v_i-x$ 类型的边之后一定不会被选

类似的，当 $x$ 增大时，生成树中 $v_i-x$ 类型的边也一定会被选，不在生成树中的 $v_i+x$ 类型的边一定不会被选

因此两边还不确定是否选的边的集合不交，考虑分治，求出当前 $mid$ 的生成树后，两侧分别有一些边必定选，这部分可以看成进行缩点，剩下可能选可能不选的边可以分治下去解决。在分治过程中记录缩点后剩余的边以及已经选的边的边权和即可。

可以对两类边分别排序，求最小生成树时归并出边按照边权排序的顺序，复杂度 $O((n+m)\log q\alpha(n))$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 105000
#pragma GCC optimize(3)
#define ll long long
struct edge{int a,b,v,c;friend bool operator <(edge a,edge b){return a.c<b.c;}};
vector<edge> e[21][2][2];
int fa[N],ct[21][2],n,a,b,q,c,d,f,is[N*2][2],id[N];
ll as[N];
struct que{int v,id;friend bool operator <(que a,que b){return a.v<b.v;}}qu[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void solve(int l,int r,int s,int t,ll v1,int v2)
{
	if(l>r)return;
	int mid=(l+r)>>1;
	for(int i=0;i<2;i++)e[s+1][0][i].clear(),e[s+1][1][i].clear(),vector<edge>().swap(e[s+1][0][i]),vector<edge>().swap(e[s+1][1][i]);
	ll as1=v1+1ll*v2*qu[mid].v;
	int m1=e[s][t][0].size(),m2=e[s][t][1].size();
	for(int i=1;i<=ct[s][t];i++)fa[i]=i;
	for(int i=1;i<=m1;i++)is[i-1][0]=0;
	for(int i=1;i<=m2;i++)is[i-1][1]=0;
	int l1=0,l2=0;
	for(int i=1;i<=m1+m2;i++)
	{
		int t1,t2;
		if(l1==m1)t1=1,t2=l2,l2++;
		else if(l2==m2)t1=0,t2=l1,l1++;
		else if(e[s][t][0][l1].c+qu[mid].v<=e[s][t][1][l2].c-qu[mid].v)t1=0,t2=l1,l1++;
		else t1=1,t2=l2,l2++;
		edge fu=e[s][t][t1][t2];
		if(finds(fu.a)!=finds(fu.b))
		as1+=fu.c+fu.v*qu[mid].v,fa[finds(fu.a)]=finds(fu.b),is[t2][t1]=1;
	}
	as[qu[mid].id]=as1;
	ll v11=v1,v21=v2;
	for(int i=1;i<=ct[s][t];i++)fa[i]=i;
	for(int i=0;i<m1;i++)if(is[i][0])fa[finds(e[s][t][0][i].a)]=finds(e[s][t][0][i].b),v21++,v11+=e[s][t][0][i].c;
	ct[s+1][0]=0;
	for(int i=1;i<=ct[s][t];i++)if(finds(i)==i)id[i]=++ct[s+1][0];
	for(int i=0;i<m1;i++)
	{
		int v1=finds(e[s][t][0][i].a),v2=finds(e[s][t][0][i].b);
		if(v1==v2)continue;
		e[s+1][0][0].push_back((edge){id[v1],id[v2],e[s][t][0][i].v,e[s][t][0][i].c});
	}
	for(int i=0;i<m2;i++)
	{
		if(!is[i][1])continue;
		int v1=finds(e[s][t][1][i].a),v2=finds(e[s][t][1][i].b);
		if(v1==v2)continue;
		e[s+1][0][1].push_back((edge){id[v1],id[v2],e[s][t][1][i].v,e[s][t][1][i].c});
	}
	ll v12=v1,v22=v2;
	for(int i=1;i<=ct[s][t];i++)fa[i]=i;
	for(int i=0;i<m2;i++)if(is[i][1])fa[finds(e[s][t][1][i].a)]=finds(e[s][t][1][i].b),v22--,v12+=e[s][t][1][i].c;
	ct[s+1][1]=0;
	for(int i=1;i<=ct[s][t];i++)if(finds(i)==i)id[i]=++ct[s+1][1];
	for(int i=0;i<m1;i++)
	{
		if(!is[i][0])continue;
		int v1=finds(e[s][t][0][i].a),v2=finds(e[s][t][0][i].b);
		if(v1==v2)continue;
		e[s+1][1][0].push_back((edge){id[v1],id[v2],e[s][t][0][i].v,e[s][t][0][i].c});
	}
	for(int i=0;i<m2;i++)
	{
		int v1=finds(e[s][t][1][i].a),v2=finds(e[s][t][1][i].b);
		if(v1==v2)continue;
		e[s+1][1][1].push_back((edge){id[v1],id[v2],e[s][t][1][i].v,e[s][t][1][i].c});
	}
	solve(l,mid-1,s+1,0,v11,v21);
	solve(mid+1,r,s+1,1,v12,v22);
	for(int i=0;i<2;i++)e[s][t][i].clear(),vector<edge>().swap(e[s][t][i]);
}
int main()
{
	scanf("%d%d%d%d",&n,&a,&b,&q);ct[0][0]=n;
	for(int i=1;i<=a;i++)scanf("%d%d%d",&c,&d,&f),e[0][0][0].push_back((edge){c,d,1,f});
	for(int i=1;i<=b;i++)scanf("%d%d%d",&c,&d,&f),e[0][0][1].push_back((edge){c,d,-1,f});
	sort(e[0][0][0].begin(),e[0][0][0].end());
	sort(e[0][0][1].begin(),e[0][0][1].end());
	for(int i=1;i<=q;i++)scanf("%d",&qu[i].v),qu[i].id=i;
	sort(qu+1,qu+q+1);
	solve(1,q,0,0,0,0);
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```



#### SCOI2020模拟?

出题人:zjk

##### auoj20 Permutation

###### Problem

给定一棵 $n$ 个点的树，对于一个 $n$ 阶排列 $p$，可以使用如下方式得到每条边的边权：

初始时每条边边权为 $0$，对于每个 $i$，将 $p_i,p_{(i\ \bmod\ n)+1}$ 路径上的所有边边权 $+1$。

给定每条边的边权，求有多少个排列得到的边权等于给定边权。答案对给定质数 $p$ 取模。保证存在至少一个合法排列。

$n\leq 5000$

$1s,256MB$

###### Sol

~~树与路径2.0~~

可以发现移动是一个环，因此排列循环位移不改变得到的边权，从而可以统计 $p_1=1$ 的排列数量，然后乘以 $n$ 即为答案。

然后可以发现，如果将树以 $1$ 为根，考虑边 $(i,fa_i)$ 的权值，可以发现权值为排列中在 $i$ 子树内的点构成的段数量的 $2$ 倍。则相当于对于每个点，这个点子树内的点在排列内构成的段数量为定值。

考虑一个点上的情况，设它有 $k$ 个儿子，边权为 $2c_1,\cdots,2c_k$，它连向父亲的边权为 $2c$，则这个点上的问题为：

将 $c_1$ 个儿子 $1$ 的段，……，$c_k$ 个儿子 $k$ 的段以及当前点自己构成的段合并成 $c$ 个非空段，要求每一段内相邻两个元素不属于同一个儿子。

可以发现不同点之间的问题独立，从而答案为每个点上的方案数乘积。



考虑一个点的问题，考虑容斥，对于每个儿子的段，考虑枚举有几个相邻段在最后的排列中相邻，则方案数为 $C_{c-1}^i$，系数为 $(-1)^i$，这样相当于合并了 $i$ 个段，还剩 $c-i$ 个段。最后的方案数为所有段任意排列。

每部分的容斥可以看成一个egf，如果将父亲的边也带入容斥，直接乘总复杂度为 $O(n^3)$。

但注意到如果只容斥所有儿子，则因为每个儿子的段数不超过子树大小，合并的总复杂度为 $O(n^2)$。因此考虑对儿子部分进行容斥，设儿子部分合并后有 $s$ 段，父亲边权为 $2c$，则父亲部分方案数相当于 $s$ 段分成 $c$ 个非空段的方案数，可以发现这等于 $C_{s-1}^{c-1}$。因此最后可以每个点 $O(n)$ 计算方案数。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 5005
int n,p,head[N],a,b,v1[N],p1[N],p2[N],p3[N],fr[N],ifr[N],ct,as=1,cnt;
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
struct edge{int t,next,id;}ed[N*3];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],id};head[t]=cnt;}
void doit1(int s)
{
	for(int i=1;i<=s;i++)p2[i]=1ll*fr[s-1]*ifr[i-1]%p*ifr[s-i]%p*ifr[i]%p*((s-i)&1?p-1:1)%p;
	for(int i=0;i<=ct+s;i++)p3[i]=0;
	for(int i=0;i<=ct;i++)
	for(int j=1;j<=s;j++)p3[i+j]=(p3[i+j]+1ll*p1[i]*p2[j])%p;
	for(int i=0;i<=ct+s;i++)p1[i]=p3[i];
	ct+=s;
}
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
	ct=1;p1[1]=0;p1[u!=1]=1;
	int st3=0,st1=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)doit1(v1[ed[i].id]/2);else st3=v1[ed[i].id]/2-1;
	if(!fa)
	for(int i=0;i<=ct;i++)st1=(st1+1ll*p1[i]*fr[i])%p;
	else for(int i=st3+1;i<=ct;i++)st1=(st1+1ll*fr[i-1]*ifr[i-st3-1]%p*ifr[st3]%p*fr[i]%p*p1[i])%p;
	as=1ll*as*st1%p;
}
int main()
{
	scanf("%d%d",&n,&p);as=n;
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&v1[i]),adde(a,b,i);
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%p,ifr[i]=pw(fr[i],p-2);
	dfs(1,0);printf("%d\n",as);
}
```



##### auoj21 LCM Game

###### Problem

有一个长度为 $k$ 的序列 $v$，其中每个数都是 $[1,n]$ 间的正整数。

求所有序列的 $lcm\ v_i$ 的和以及乘积，答案模 $10^9+7$

$n\leq 500,k\leq 50$

$2s,1024MB$

###### Sol

~~在一年后，这题被出到了n=1000,k=10^9~~

首先考虑乘积部分，这时可以计算每个质因子对乘积贡献的次数。

考虑一个质因子 $p$，可以对于每个 $k$ 求出 $[1,n]$ 间有多少个正整数是 $p^k$ 的倍数且不是 $p^{k+1}$ 的倍数。然后容易求出有多少种方案的 $lcm$ 是 $p^k$ 的倍数，由线性性对 $k\geq 1$ 求和上述结果就可以得到 $p$ 在乘积里面出现的次数。这部分的复杂度为 $O(n\log n\log mod)$。注意指数上取模需要对 $\phi(mod)$ 取模。

这样解决了乘积的问题，然后考虑求和。



首先考虑容斥，考虑对于每个 $v$，求出 $[1,n]$ 中有多少个数是 $v$ 的约数。这个数量的 $k$ 次方即为结果为 $v$ 的约数的方案数。然后再容斥即可得到结果为每种数的方案数。

考虑每个 $v$ 的容斥系数，设系数为 $g_v$，则它需要满足 $\sum_{x|v}g_x=v$，由狄利克雷卷积可以发现 $g_v=\phi(v)$。

数量的 $k$ 次方是非常难以处理，考虑一个形式幂级数，将每个数看成 $x$，这样 $\sum_{i=1}^n[i|v]$ 可以化为 $\prod_{i=1}^n[i|v]x$，然后可以看成对这样一个形式幂级数求和，即：
$$
\sum_{v\in S}\phi(v)*\prod_{i=1}^n[i|v]x
$$
（这里 $S$ 是可能的LCM构成的集合）

求出这个幂级数即可得到答案，一种简便的处理方式是插值，枚举 $x=1,\cdots,n+2$ 带入分别求出结果，再插值得到多项式即可得到答案。（这里原做法是将 $(\sum)^k$ 展开后放入 $dp$）

这时问题可以看成，有一个数组 $f$，只考虑 $S$ 中的位置，对于每一个 $i=1,\cdots,n$，将数组中 $i$ 的倍数全部乘以 $x$，最后求和 $\sum \phi(v)f_v$。



考虑一个大于 $\sqrt n$ 的质数 $p$，先操作所有是 $p$ 倍数的位置，这之后可以发现对于一个数 $x$（$p$ 不是 $x$ 的约数），$f_x,f_{px}$ 在之后的操作的变化相同。因为 $\phi$ 是积性函数，考虑将 $f_{px}$ 乘以 $\phi(p)$ 加给 $f_x$，然后删去 $f_{px}$，可以发现如果这样变化，则变化后继续这个过程，结果不变。

类似的，对于质数 $p$ 和正整数 $k$，考虑一个 $x$ 使得 $p$ 不是 $x$ 的约数，如果之后的操作不存在数是 $p^k$ 的倍数，则可以将所有的 $f_{p^kx}$ 使用类似方式加到 $f_{p^{k-1}x}$ 上也可以使答案不变。

同时，如果之前的操作不存在数是 $p^k$ 的倍数，则 $f_{p^kx}=f_{p^{k-1}x}$，因此可以只记录较小的部分，需要的时候再扩展。

因此，对于一个 $p_i$，记 $l_i$ 为左侧操作的数中 $p_i$ 最大的次数，$r_i$ 为右侧最大的次数，则只需要对于这一维记录 $p_i^{0,1,\cdots\min\{l_i,r_i\}}$ 部分。每个时刻状态数为 $\prod_i(1+\min\{l_i,r_i\})$。

记这个状态数总和为 $f_n$，可以发现将所有数按照最大质因子从大到小排序后就有 $f_{1000}\leq 1.7\times 10^6$，做 $n$ 次即可。每一步都可以通过dfs枚举状态并使用一些技巧记录状态做到状态数的线性。

一种较快的实现方式是先处理出操作序列，这样可以减少做 $n$ 次的常数。

复杂度 $O(nf_n)$，在 $n=1000$ 下需要 $0.7s$



###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<cstring>
using namespace std;
#define N 1050
#define M 23333
#define S 12
#define mod 1000000007
int n,k,a,v[N],pr[N],ct,tp1[N],vl[N],s[S][N],as[S][N],id[N],nt1[N],fg1[N],fg2[N],g1[N],g2[N];
int pw(int a,int b,int md=mod){int as=1;while(b){if(b&1)as=1ll*as*a%md;a=1ll*a*a%md;b>>=1;}return as;}
vector<int> f1[N],f2[N],nt[N],s2[N],t1[N];
vector<pair<int,int> > t2[N];
int su=1,dp[M],s1[S+2][2],s11[S+2][2],dp2[M],rb[N],rb2[N],r1[N],fu[N],fu2[N];
void dfs(int d,int n,int v)
{
	if(d==S+1){nt[v].push_back(n);return;}
	for(int i=r1[d];i<=s1[d][1];i++)dfs(nt1[d],n+rb[d]*i,v);
}
void dfs1(int d,int n,int v)
{
	if(d==S+1){t1[v].push_back(n);return;}
	for(int i=0;i<=s1[d][1];i++)dfs1(nt1[d],n+rb[d]*min(i,s11[d][1]),v);
}
void dfs2(int d,int n,int v,int s)
{
	if(d==S+1){t2[v].push_back(make_pair(n,s));return;}
	for(int i=0;i<s1[d][1];i++)dfs2(nt1[d],n+rb[d]*i,v,s);
	int v1=1ll*s*(mod+1-s1[d][0])%mod,v2=s;
	for(int i=s1[d][1];i<s11[d][1];i++)dfs2(nt1[d],n+rb[d]*s1[d][1],v,v1),v1=1ll*v1*s1[d][0]%mod,v2=1ll*v2*s1[d][0]%mod;
	dfs2(nt1[d],n+rb[d]*s1[d][1],v,v2);
}
void add(int x){for(int i=1;i<=S;i++)if(s1[i][0]==x){s1[i][1]++;return;}for(int i=1;i<=S;i++)if(!s1[i][1]){s1[i][0]=x;s1[i][1]++;return;}}
void del(int x){for(int i=1;i<=S;i++)if(s1[i][0]==x){s1[i][1]--;return;}}
bool cmp(int a,int b){return v[a]==v[b]?a<b:v[a]>v[b];}
void init(int n)
{
	for(int i=1;i<=n;i++)f1[i].clear(),f2[i].clear(),nt[i].clear(),s2[i].clear(),t1[i].clear(),t2[i].clear(),fg1[i]=fg2[i]=0;
	ct=0;
	for(int i=2;i<=n;i++)
	{
		int fg=1;
		for(int j=2;j<i;j++)if(i%j==0)fg=0;
		if(fg)pr[++ct]=i,id[i]=ct;
	}
	for(int i=1;i<=n;i++)tp1[i]=i;
	for(int i=1;i<=n;i++)
	{
		int st=i;
		for(int j=2;j*j<=st;j++)if(st%j==0)
		{
			while(st%j==0)st/=j;
			v[i]=j;
		}
		if(st>1)v[i]=st;
	}
	sort(tp1+1,tp1+n+1,cmp);
	for(int i=1;i<=n;i++)
	{
		int st=tp1[i];
		for(int j=2;j*j<=st;j++)if(st%j==0)
		{
			while(st%j==0)st/=j;
			s2[id[j]].push_back(i);
		}
		if(st>1)s2[id[st]].push_back(i);
	}
	for(int i=1;i<=ct;i++)
	{
		for(int j=0;j<s2[i].size();j++)
		{
			fu[j]=0;int tp=tp1[s2[i][j]];
			while(tp%pr[i]==0)tp/=pr[i],fu[j]++;
		}
		fu2[s2[i].size()]=0;
		for(int j=s2[i].size()-1;j>=0;j--)fu2[j]=max(fu2[j+1],fu[j]);
		int ls=0;
		for(int j=0;j<s2[i].size();j++)
		{
			while(ls<fu[j])ls++,f1[s2[i][j]].push_back(pr[i]);
			while(ls>fu2[j+1])ls--,f2[s2[i][j]].push_back(pr[i]);
		}
	}
}
void pre_solve(int n)
{
	for(int i=1;i<=S;i++)s1[i][0]=s1[i][1]=0;su=1;
	dp[0]=1;
	for(int i=1;i<=n;i++)
	{
		if(f1[i].size())fg1[i]=1;
		memcpy(s11,s1,sizeof(s11));
		for(int j=0;j<f1[i].size();j++)add(f1[i][j]);
		for(int j=1;j<=S;j++){int st=j+1;while(st<=S&&!s1[st][1])st++;nt1[j]=st;}
		rb[S]=1;for(int j=S;j>=1;j--)rb[j-1]=rb[j]*(s11[j][1]+1);
		if(fg1[i])dfs1(1,0,i);
		rb[S]=1;for(int j=S;j>=1;j--)rb[j-1]=rb[j]*(s1[j][1]+1);
		int vl1=tp1[i];
		for(int j=1;j<=S;j++)r1[j]=0;
		for(int j=1;j<=S;j++)if(s1[j][0])while(vl1%s1[j][0]==0)r1[j]++,vl1/=s1[j][0];
		dfs(1,0,i);
		if(f2[i].size())fg2[i]=1;
		memcpy(s11,s1,sizeof(s11));
		for(int j=0;j<f2[i].size();j++)del(f2[i][j]);
		rb[S]=1;for(int j=S;j>=1;j--)rb[j-1]=rb[j]*(s1[j][1]+1);
		if(fg2[i])dfs2(1,0,i,1);
	}
}
int solve(int n,int v1)
{
	for(int i=1;i<=S;i++)s1[i][0]=s1[i][1]=0;su=1;
	dp[0]=1;
	for(int i=1;i<=n;i++)
	{
		if(fg1[i])for(int j=t1[i].size()-1;j>=0;j--)dp[j]=dp[t1[i][j]];
		int tp=pw(v1,vl[tp1[i]]),vl1;
		for(int j=0;j<nt[i].size();j++)dp[nt[i][j]]=1ll*dp[nt[i][j]]*tp%mod;
		if(fg2[i])for(int j=0;j<t2[i].size();j++)vl1=dp[j],dp[j]=0,dp[t2[i][j].first]=(dp[t2[i][j].first]+1ll*vl1*t2[i][j].second)%mod;
	}
	return dp[0]-1;
}
void doit(int l,int *s,int *t)
{
	int vl[N]={0};vl[0]=1;
	for(int i=1;i<=l;i++)
	for(int j=i;j>=0;j--)vl[j+1]=(vl[j+1]+vl[j])%mod,vl[j]=1ll*vl[j]*(mod-i)%mod;
	for(int i=1;i<=l;i++)
	{
		int inv=pw(mod-i,mod-2),s1=1;
		for(int j=1;j<=l;j++)if(i!=j)s1=1ll*s1*(i-j+mod)%mod;s1=pw(s1,mod-2);
		for(int j=0;j<=l;j++)vl[j]=1ll*inv*vl[j]%mod,vl[j+1]=(vl[j+1]-vl[j]+mod)%mod,t[j]=(t[j]+1ll*s[i]*vl[j]%mod*s1)%mod;
		for(int j=l;j>=0;j--)vl[j+1]=(vl[j+1]+vl[j])%mod,vl[j]=1ll*vl[j]*(mod-i)%mod;
	}
}
int solve_pr()
{
	int st1=1;
	for(int i=2;i<=n;i++)
	{
		int fg1=0;
		for(int j=2;j<i;j++)if(i%j==0)fg1=1;
		if(fg1)continue;
		int tp1=1,tp2=i,las=0;
		while(tp2/i<=n)
		{
			int v3=pw(n-n/tp2,k,mod-1);
			tp1=1ll*tp1*pw(tp2/i,(v3-las+mod-1)%(mod-1),mod)%mod;
			tp2*=i,las=v3;
		}
		st1=1ll*st1*tp1%mod;
	}
	return st1;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)vl[i]=1;
	init(n);pre_solve(n);
	for(int i=1;i<=n+2;i++)g1[i]=solve(n,i);
	doit(n+2,g1,g2);
	int as1=0;
	for(int i=0;i<=n;i++)as1=(as1+1ll*pw(i,k)*g2[i])%mod;
	printf("%d\n",as1);printf("%d\n",solve_pr());
}
```



##### auoj22 Easy Data Structure

###### Problem

考虑一个包含 `01&^|()` 的表达式，表达式按照括号优先，同一层之间从左向右的顺序运算，其中 `&^|` 为 `C++` 中的运算符。对于一个合法表达式，可以得到它的运算结果，显然结果为 $\{0,1\}$ 中的一个数。

有一个长度为 $n$ 的表达式，表达式中包含四种字符：

1. 权值字符，每一个这种字符有 $p_0$ 的概率为 `0`，有 $p_1$ 的概率为 `1`。
2. 运算字符，每一个这种字符有 $p_0$ 的概率为 `&`，$p_1$ 的概率为 `|`，$p_2$ 的概率为 `^`。
3. 左括号 `(`。
4. 右括号 `)`。

这个表达式在任意情况下是合法的，定义它的权值为所有随机字符独立随机取值，结果为 $1$ 的概率。

有 $q$ 次修改，每次修改一个字符的随机生成概率，每次修改后求表达式的权值。

$n,q\leq 2\times 10^5$

$4s,1024MB$

###### Sol

如果不存在括号，则由于运算顺序为从左向右，每个运算可以看成一个矩阵，可以对于一段运算开头，数字结尾的区间，求出这段区间如果左侧是 $x$，最后变成 $y$ 的概率，转移矩阵乘法。对这个东西线段树维护即可。

考虑括号的情况，括号关系构成一个树，修改一个位置会修改这个位置在树上的祖先。

考虑ddp，对括号树进行重链剖分，对于每个点处理出如果重儿子括号的运算结果为 $x$，则这个括号运算结果为 $y$ 的概率。这部分对于每一个括号维护一个线段树即可求出这个矩阵，并在每一次修改轻儿子时维护矩阵的变化。

然后再对于每一个重链线段树维护这个矩阵的乘法。每次修改时从下往上跳重链，在每个链顶修改父亲的线段树即可。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 300500
#define mod 998244353
int n,q,a,ty[N],s[N][3],bel[N][2],bel2[N][2],ct,head[N],cnt,sz[N],sn[N],tp[N],ct1,dp[N][2],rt[N],snid[N],ct2,ed1[N],id[N],st[N],su[N],rb,pr=1;
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
int fu[3][2][2]={0,0,0,1,0,1,1,1,0,1,1,0};
vector<int> s1[N];
struct sth{
	int s[2][2];
	int is1,is2,v1[3],v2[3];
}v1;
sth doit(sth a,sth b)
{
	sth c;
	for(int s=0;s<2;s++)
	for(int t=0;t<2;t++)
	c.s[s][t]=0;
	for(int s=0;s<3;s++)c.v1[s]=a.v1[s],c.v2[s]=b.v2[s];
	c.is1=a.is1,c.is2=b.is2;
	if(!a.is2)
	for(int s=0;s<2;s++)for(int t=0;t<2;t++)for(int f=0;f<2;f++)
	c.s[s][t]=(c.s[s][t]+1ll*a.s[s][f]*b.s[f][t])%mod;
	else 
	{
		int f[2]={0},g[2][2]={0},h[2]={0};
		for(int s=0;s<3;s++)
		for(int t=0;t<2;t++)
		{
			int v1=1ll*a.v2[s]*b.v1[t]%mod;
			for(int v=0;v<2;v++)g[v][fu[s][v][t]]=(g[v][fu[s][v][t]]+v1)%mod;
		}
		for(int s=0;s<2;s++)
		{
			for(int t=0;t<2;t++)f[t]=a.s[s][t];
			h[0]=h[1]=0;
			for(int s1=0;s1<2;s1++)
			for(int s2=0;s2<2;s2++)
			h[s2]=(h[s2]+1ll*f[s1]*g[s1][s2])%mod;
			for(int t=0;t<2;t++)f[t]=h[t],h[t]=0;
			for(int s1=0;s1<2;s1++)
			for(int s2=0;s2<2;s2++)
			h[s2]=(h[s2]+1ll*f[s1]*b.s[s1][s2])%mod;
			for(int t=0;t<2;t++)c.s[s][t]=h[t];
		}
	}
	return c;
}
struct segt{int l,r,ls,rs;sth tp;}e[N*4];
int build(int l,int r)
{
	int st=++ct2;
	e[st].l=l;e[st].r=r;
	if(l==r)return st;
	int mid=(l+r)>>1;
	e[st].ls=build(l,mid);
	e[st].rs=build(mid+1,r);
	return st;
}
void modify(int x,int v)
{
	if(e[x].l==e[x].r){e[x].tp=v1;return;}
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=v)modify(e[x].ls,v);
	else modify(e[x].rs,v);
	e[x].tp=doit(e[e[x].ls].tp,e[e[x].rs].tp);
}
struct mat{int s[2][2];};
mat operator *(mat a,mat b){mat c;for(int i=0;i<2;i++)for(int j=0;j<2;j++){c.s[i][j]=0;for(int k=0;k<2;k++)c.s[i][j]=(c.s[i][j]+1ll*a.s[i][k]*b.s[k][j])%mod;}return c;}
struct segt2{
	struct node{int l,r;mat v;}e[N*4];
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r)return;
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	}
	void modify(int x,int v,mat s)
	{
		if(e[x].l==e[x].r){e[x].v=s;return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=v)modify(x<<1,v,s);
		else modify(x<<1|1,v,s);
		e[x].v=e[x<<1|1].v*e[x<<1].v;
	}
	mat query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].v;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);
		else if(mid<l)return query(x<<1|1,l,r);
		else return query(x<<1|1,mid+1,r)*query(x<<1,l,mid);
	}
}tr;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa){sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u],sz[u]+=sz[ed[i].t];}
void dfs2(int u,int v,int fa){id[u]=++ct1;tp[u]=v;ed1[u]=u;if(sn[u])dfs2(sn[u],v,u),ed1[u]=ed1[sn[u]];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs2(ed[i].t,ed[i].t,u);}
void justdoit(int x)
{
	if(!sn[x])return;
	mat tp;for(int i=0;i<2;i++)for(int j=0;j<2;j++)tp.s[i][j]=0;
	for(int t=0;t<2;t++)
	{
		v1.is1=1,v1.is2=0;v1.v1[t]=1,v1.v1[!t]=0;
		modify(rt[x],snid[x]);
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)tp.s[t][i]=(tp.s[t][i]+1ll*e[rt[x]].tp.v1[j]*e[rt[x]].tp.s[j][i])%mod;
	}
	tr.modify(1,id[x],tp);
}
void query(int x)
{
	int t=ed1[x];
	if(x==t)
	{
		dp[x][0]=dp[x][1]=0;
		for(int s=0;s<2;s++)for(int j=0;j<2;j++)dp[x][s]=(dp[x][s]+1ll*e[rt[x]].tp.v1[j]*e[rt[x]].tp.s[j][s])%mod;
		return;
	}
	mat st=tr.query(1,id[x],id[t]-1);
	dp[x][0]=dp[x][1]=0;
	for(int s=0;s<2;s++)for(int i=0;i<2;i++)for(int j=0;j<2;j++)dp[x][s]=(dp[x][s]+1ll*e[rt[t]].tp.v1[j]*e[rt[t]].tp.s[j][i]%mod*st.s[i][s])%mod;
}
void modify1(int x)
{
	while(x)
	{
		justdoit(x);
		x=tp[x];
		query(x);
		v1.is1=1,v1.is2=0;v1.v1[0]=dp[x][0],v1.v1[1]=dp[x][1];
		modify(rt[bel2[x][0]],bel2[x][1]);
		x=bel2[x][0];
	}
}
void pre()
{
	tr.build(1,1,n);
	v1.s[1][1]=v1.s[0][0]=1;
	v1.s[0][1]=v1.s[1][0]=0;
	for(int i=ct;i>=1;i--)
	{
		rt[i]=build(1,s1[i].size());
		for(int j=0;j<s1[i].size();j++)
		if(s1[i][j]>0)
		{
			v1.is1=ty[s1[i][j]]==1,v1.is2=!v1.is1;
			for(int k=0;k<3;k++)v1.v1[k]=v1.v2[k]=s[s1[i][j]][k];
			modify(rt[i],j+1);
		}
		else
		{
			v1.is1=1,v1.is2=0;
			for(int k=0;k<2;k++)v1.v1[k]=dp[-s1[i][j]][k];
			modify(rt[i],j+1);
		}
		ed1[i]=i,query(i);
		if(i>1)adde(i,bel2[i][0]);
	}
	dfs1(1,0);dfs2(1,1,0);
	for(int i=1;i<=ct;i++)
	for(int j=0;j<s1[i].size();j++)
	if(s1[i][j]==-sn[i])snid[i]=j+1;
	for(int i=1;i<=ct;i++)justdoit(i);
}
int main()
{
	scanf("%d%d",&n,&q);
	st[0]=1;ct=1;
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&ty[i]);
		int st1=ty[i]<3?ty[i]+1:0,su1=0;
		for(int j=0;j<st1;j++)scanf("%d",&s[i][j]),su1+=s[i][j];
		if(su1>0)pr=1ll*pr*su1%mod;
		if(ty[i]==3)st[++rb]=++ct,bel2[ct][0]=st[rb-1],bel2[ct][1]=++su[rb-1],su[rb]=0,s1[st[rb-1]].push_back(-ct);
		else if(ty[i]==4)rb--;
		else bel[i][0]=st[rb],bel[i][1]=++su[rb],s1[st[rb]].push_back(i);
	}
	pre();
	while(q--)
	{
		scanf("%d",&a);
		int su=0;
		for(int j=0;j<=ty[a];j++)su+=s[a][j];
		pr=1ll*pr*pw(su,mod-2)%mod;su=0;
		for(int j=0;j<=ty[a];j++)scanf("%d",&s[a][j]),su+=s[a][j];
		pr=1ll*su*pr%mod;
		v1.is1=ty[a]==1,v1.is2=!v1.is1;
		for(int k=0;k<3;k++)v1.v1[k]=v1.v2[k]=s[a][k];
		modify(rt[bel[a][0]],bel[a][1]);
		modify1(bel[a][0]);
		printf("%d\n",1ll*dp[1][1]*pw(pr,mod-2)%mod);
	}
}
```



#### SCOI2020模拟?

出题人:zjk&lsj

##### auoj23 树与路径

###### Problem

给一棵 $n$ 个点的树，对于每一个 $k$ 求出下列问题的答案：

你需要在树上选择 $k$ 条非空的链，使得这些链正好将每条边覆盖一次，求方案数。

对于每个 $k$ 求出答案，答案对 $998244353$ 取模。

$n\leq 10^5$

$1s,256MB$

###### Sol

考虑一个点对路径的影响，设这个点度数为 $d_i$，考虑跨过这个点的路径，相当于将这 $d_i$ 条边中的一些配对合并，可以发现选择 $k$ 对边合并的方案数为 $C_{d_i}^{2k}*(2k-1)!!$。

考虑对于每个点选择边配对合并，因为图是树，可以发现任意一种合并方式都会对应一种用链覆盖树的方式，因此对每个点的相邻边配对合并的方案和原问题的方案一一对应。

考虑每个点配对的方案，可以发现对于一个点，选择合并的边对数即为路径减少的条数。因此将每个点的方案看成关于 $k$ 的多项式，将所有点的多项式乘起来，多项式的 $x$ 次数表示路径减少的条数，这样即可得到最后对于每个 $k$ 的答案。

分治fft即可，复杂度 $O(n\log^2 n)$

###### Code

~~2022年新代码~~

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 132001
#define mod 998244353
int n,a,b,d[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int rev[N*2],gr[2][N*2],fr[N],ifr[N],vl[N];
void init(int d=17)
{
	for(int l=2;l<=1<<d;l<<=1)
	for(int i=0;i<l;i++)
	rev[l+i]=(rev[l+(i>>1)]>>1)|((i&1)*(l>>1));
	for(int t=0;t<2;t++)
	for(int l=2;l<=1<<d;l<<=1)
	{
		int tp=pw(3,(mod-1)/l);
		if(!t)tp=pw(tp,mod-2);
		int v1=1;
		for(int i=0;i<l;i++)gr[t][l+i]=v1,v1=1ll*v1*tp%mod;
	}
}
int ntt[N];
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
int f[N],g[N];
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	if(s1+s2<=100)
	{
		vector<int> as(s1+s2-1);
		for(int i=0;i<s1;i++)for(int j=0;j<s2;j++)as[i+j]=(as[i+j]+1ll*a[i]*b[j])%mod;
		return as;
	}
	int l=1;while(l<s1+s2)l<<=1;
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<s1;i++)f[i]=a[i];
	for(int i=0;i<s2;i++)g[i]=b[i];
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	vector<int> as;
	for(int i=0;i<s1+s2-1;i++)as.push_back(f[i]);
	return as;
}
vector<int> st[23];
int ct=0;
int main()
{
	scanf("%d",&n);init();
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),d[a]++,d[b]++;
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	vl[0]=1;for(int i=1;i<=n;i++)vl[i]=1ll*vl[i-1]*(2*i-1)%mod;
	for(int i=1;i<=n;i++)
	{
		vector<int> sr;
		for(int j=0;j*2<=d[i];j++)sr.push_back(1ll*fr[d[i]]*ifr[j*2]%mod*ifr[d[i]-j*2]%mod*vl[j]%mod);
		int si=i,ci=0;
		while(si&&(~si&1))si>>=1,ci++;
		while(ci--)sr=polymul(sr,st[ct]),ct--;
		st[++ct]=sr;
	}
	vector<int> as;as.push_back(1);
	for(int i=1;i<=ct;i++)as=polymul(as,st[i]);
	while(as.size()<n)as.push_back(0);
	for(int i=1;i<n;i++)printf("%d ",as[n-1-i]);
}
```



##### auoj24 树据结构

###### Problem

有一棵 $n$ 个点，有边权的有根树 $S$，$1$ 为根。

有一棵树 $T$，初始 $T=S$。接下来有 $q$ 次操作，每次操作为如下几种之一：

1. 将 $S$ 上某一条链上的边权全部加上 $v$。
2. 将 $S$ 复制一份，将这一部分中的点 $1$ 连到 $T$ 的点 $a$ 上，作为 $a$ 的一个儿子，设之前 $T$ 有 $m$ 个点，则复制部分按原编号顺序编号为 $m+1,\cdots,m+n$。同时给出连接两部分的边的边权 $v$。
3. 给出 $x,y$，询问 $T$ 中 $x$ 到 $y$ 路径的边权和。
4. 给出 $x$，询问 $T$ 中 $x$ 子树内的边权和。

强制在线

$n,q\leq 10^5$

$1s,1024MB$

###### Sol

~~2022年新做法~~

对于 $1$ 操作，可以树链剖分处理链加，然后用主席树维护dfs序上的边权修改，这样就可以维护每个复制出来的部分上的边权。

如果将每一次复制的 $S$ 看成一个点，则 $T$ 可以看成这些点构成的树，每次 $2$ 操作为加一个叶子。

对于每一个部分，记录这个部分的父亲是哪一个部分，同时记录这个部分的根到父亲部分的根的距离，这个距离可以在加入时在主席树上求出。

此时对于一个 $3$ 询问，可以让两个点先在不同部分之间向上跳到同一部分，然后再处理同一部分之间的距离。对于不同部分，可以维护部分间父亲关系的倍增，每次加入叶子可以 $O(\log n)$ 求出新的倍增数组。同时维护每个部分根到 $1$ 的距离即可处理这部分的距离。对于同一部分内的，可以使用主席树维护的信息处理。



考虑 $4$ 询问，如果 $n=1$，问题相当于加入带权叶子，求子树和。

一种简单的实现方式是平衡树维护括号序，插入叶子时在父亲的括号内插入一对叶子，子树和即为括号序中的区间求和。

对于任意的情况，与上面的区别在于一个部分内部不再是一个点。

但考虑将一个部分的所有儿子按照这一部分中的dfs序排序，这样一个点的子树仍然是括号序中的一段区间。可以使用set维护每个部分所有儿子排序后的情况，这样即可找到应该插入的位置。

这样就处理了加入的其它部分对子树和的贡献，最后当前部分的贡献相当于dfs序在一段区间内的边权和，也可以在主席树上询问。

复杂度 $O(n\log n+q\log^2 n)$，不需要写LCT所以比以前做法都短

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
#include<vector>
using namespace std;
#define N 100500
#define M 9017925
#define ll long long
int n,q,op,a,b,c;
ll x,y,z,las;
int head[N],cnt;
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;
}

//HLD
int vi[N],dep[N],sz[N],sn[N],f[N],tp[N],id[N],rid[N],rb[N],c1;
void dfs1(int u,int fa)
{
	dep[u]=dep[fa]+1;sz[u]=1;f[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		vi[ed[i].t]=ed[i].v;
		dfs1(ed[i].t,u);sz[u]+=sz[ed[i].t];
		if(sz[ed[i].t]>sz[sn[u]])sn[u]=ed[i].t;
	}
}
void dfs2(int u,int fa,int v)
{
	tp[u]=v;id[u]=++c1;rid[c1]=u;
	if(sn[u])dfs2(sn[u],u,v);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])
	dfs2(ed[i].t,u,ed[i].t);
	rb[u]=c1;
}
struct sth{int l,r;};
vector<sth> que(int x,int y)
{
	vector<sth> as;
	while(tp[x]!=tp[y])
	{
		if(dep[tp[x]]<dep[tp[y]])swap(x,y);
		as.push_back((sth){id[tp[x]],id[x]});x=f[tp[x]];
	}
	if(dep[x]>dep[y])swap(x,y);
	if(x!=y)as.push_back((sth){id[x]+1,id[y]});
	return as;
}

//pretree
struct pretree{
	int rt,lz[M],ch[M][2],ct;
	ll su[M];
	int build(int l,int r)
	{
		int st=++ct;
		if(l==r){su[st]=vi[rid[l]];return st;}
		int mid=(l+r)>>1;
		ch[st][0]=build(l,mid);ch[st][1]=build(mid+1,r);
		su[st]=su[ch[st][0]]+su[ch[st][1]];
		return st;
	}
	int modify(int x,int l,int r,int l1,int r1,int v)
	{
		int st=++ct;
		su[st]=su[x]+1ll*(r1-l1+1)*v;lz[st]=lz[x];
		ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];
		if(l==l1&&r==r1){lz[st]+=v;return st;}
		int mid=(l+r)>>1;
		if(mid>=l1)ch[st][0]=modify(ch[x][0],l,mid,l1,min(r1,mid),v);
		if(mid<r1)ch[st][1]=modify(ch[x][1],mid+1,r,max(mid+1,l1),r1,v);
		return st;
	}
	void add(int l,int r,int v){rt=modify(rt,1,n,l,r,v);}
	ll query(int x,int l,int r,int l1,int r1)
	{
		if(l==l1&&r==r1)return su[x];
		if(!x)return 0;
		int mid=(l+r)>>1;
		ll as=1ll*lz[x]*(r1-l1+1);
		if(mid>=l1)as+=query(ch[x][0],l,mid,l1,min(r1,mid));
		if(mid<r1)as+=query(ch[x][1],mid+1,r,max(mid+1,l1),r1);
		return as;
	}
}tr;
void modify(int x,int y,int v)
{
	vector<sth> s1=que(x,y);
	for(int i=0;i<s1.size();i++)tr.add(s1[i].l,s1[i].r,v);
}
ll query(int id,int x,int y)
{
	vector<sth> s1=que(x,y);
	ll as=0;
	for(int i=0;i<s1.size();i++)as+=tr.query(id,1,n,s1[i].l,s1[i].r);
	return as;
}
int ctr=1,rti[N];
int fi[N],nt[N],fv[N],d1[N],f1[N][17];
ll vl[N];
int getLCA(int x,int y)
{
	if(d1[x]<d1[y])swap(x,y);
	for(int i=16;i>=0;i--)if(d1[x]-d1[y]>=(1<<i))x=f1[x][i];
	if(x==y)return x;
	for(int i=16;i>=0;i--)if(f1[x][i]!=f1[y][i])x=f1[x][i],y=f1[y][i];
	return f1[x][0];
}

//Splay
struct Splay{
	int ch[N*2][2],fa[N*2],rt;
	ll su[N*2],vl[N*2];
	void pushup(int x){su[x]=vl[x]+su[ch[x][0]]+su[ch[x][1]];}
	void rotate(int x)
	{
		int f=fa[x],g=fa[f],tp=ch[f][1]==x;
		fa[x]=g;ch[g][ch[g][1]==f]=x;
		ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;
		ch[x][!tp]=f;fa[f]=x;
		pushup(f);pushup(x);
	}
	void splay(int x,int y=0)
	{
		while(fa[x]!=y)
		{
			int f=fa[x],g=fa[f];
			if(g!=y)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);
			rotate(x);
		}
		if(!y)rt=x;
	}
	void init(){rt=1;ch[1][1]=2;fa[2]=1;}
	void ins(int lb,int x,ll v)
	{
		splay(lb);
		int tp=ch[lb][1];while(ch[tp][0])tp=ch[tp][0];
		splay(tp,lb);
		ch[tp][0]=x;ch[x][1]=x+1;
		fa[x+1]=x;fa[x]=tp;
		vl[x]=v;splay(x);
	}
	ll query(int l,int r)
	{
		splay(l);splay(r,l);
		return su[ch[r][0]];
	}
}sp;
set<pair<int,int> > sr[N];

void addtree(int lf,int sf,int v)
{
	ctr++;
	int st=ctr;
	fi[st]=lf;nt[st]=sf;fv[st]=v;d1[st]=d1[lf]+1;
	vl[st]=vl[lf]+v+query(rti[lf],1,sf);
	f1[st][0]=lf;for(int i=1;i<=16;i++)f1[st][i]=f1[f1[st][i-1]][i-1];
	rti[st]=tr.rt;
	int fr;
	sr[lf].insert(make_pair(id[sf],st));
	set<pair<int,int> >::iterator it=sr[lf].find(make_pair(id[sf],st));
	if(it==sr[lf].begin())fr=lf*2-1;
	else it--,fr=2*(*it).second;
	sp.ins(fr,st*2-1,v+tr.query(rti[st],1,n,1,n));
}
ll query1(int l1,int s1,int l2,int s2)
{
	int lc=getLCA(l1,l2);
	ll as=0;
	if(l1!=lc)
	{
		as+=query(rti[l1],1,s1)+vl[l1];
		for(int i=16;i>=0;i--)if(d1[l1]-d1[lc]>(1<<i))l1=f1[l1][i];
		as-=vl[l1];as+=fv[l1];s1=nt[l1];
	}
	if(l2!=lc)
	{
		as+=query(rti[l2],1,s2)+vl[l2];
		for(int i=16;i>=0;i--)if(d1[l2]-d1[lc]>(1<<i))l2=f1[l2][i];
		as-=vl[l2];as+=fv[l2];s2=nt[l2];
	}
	return as+query(rti[lc],s1,s2);
}
ll query2(int l1,int s1)
{
	ll as=tr.query(rti[l1],1,n,id[s1]+1,rb[s1]);
	int li=0,ri=-1;
	set<pair<int,int> >::iterator it=sr[l1].lower_bound(make_pair(id[s1],0));
	if(it!=sr[l1].begin())it--,li=(*it).second*2;else li=l1*2-1;
	it=sr[l1].lower_bound(make_pair(rb[s1]+1,0));
	if(it!=sr[l1].begin())it--,ri=(*it).second*2;
	if(ri!=-1&&li!=ri)as+=sp.query(li,ri);
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&q,&op);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
	dfs1(1,0);dfs2(1,0,1);
	tr.rt=tr.build(1,n);rti[1]=tr.rt;
	sp.init();
	while(q--)
	{
		scanf("%d",&a);
		if(a==1)
		{
			scanf("%lld%lld%lld",&x,&y,&z);
			x^=op*las,y^=op*las,z^=op*las;
			modify(x,y,z);
		}
		else if(a==2)
		{
			scanf("%lld%lld",&x,&y);
			x^=op*las,y^=op*las;
			addtree((x-1)/n+1,(x-1)%n+1,y);
		}
		else if(a==3)
		{
			scanf("%lld%lld",&x,&y);
			x^=op*las,y^=op*las;
			printf("%lld\n",las=query1((x-1)/n+1,(x-1)%n+1,(y-1)/n+1,(y-1)%n+1));
		}
		else
		{
			scanf("%lld",&x);
			x^=op*las;
			printf("%lld\n",las=query2((x-1)/n+1,(x-1)%n+1));
		}
	}
}
```



##### auoj25 树上的数

###### Problem

有一棵 $n$ 个点的树，你需要给每个点一个正奇数权值 $v_i$，使得 $\prod v_i\leq m$。

定义树上一条路径的权值为路径上所有点的权值的 $\gcd$，定义这棵树的权值为所有路径的权值乘积。

求所有给权值的方案得到的树权值之和。答案模 $998244353$

$n\leq 100,m\leq 10^{10}$，树随机生成

$8s,1536MB$

###### Sol

设 $f(m)$ 表示乘积正好为 $m$ 时的答案。

可以发现对于所有路径 $\gcd$ 乘积的形式，不同质因子间的贡献独立，可以对每个质因子计算这个质因子贡献的次数再相乘。

同时对于一个 $f(m)$，选出乘积为 $m$ 的方案可以看成将 $m$ 分解，然后每种质因子独立进行选择。

因此可以发现不同质因子的贡献完全独立，即设 $m=\prod p_i^{q_i}$，可以得到 $f(m)=\prod f(p_i^{q_i})$，即 $f$ 是积性函数。

考虑一个 $f(p^k)$，这相当于如下问题：

你需要给这棵树每个点一个非负整数权值（表示指数），使得权值和为 $k$，考虑每条链上的权值 $\min$ 之和，如果和为 $s$，则这种方案的贡献为 $p^s$，求所有方案的贡献之和。

那么可以发现 $f(p)=n*p$，因此如果能求出所有的 $f(p^k)$，则可以使用 min_25 筛 $O(n^{1-\epsilon})$ 求出答案。



考虑对于每个 $k$，对于每个 $s$ 求出和为 $s$ 的方案，这样预处理后可以对于一个 $p^k$ 可以 $O(k^2)$ 求出答案。

考虑树上 $dp$，对于一个子树，需要记录如下信息：

1. 子树内路径的权值 $\min$ 之和
2. 子树内权值和
3. 子树内每个点到根的路径权值 $\min$ 构成的可重集（用于向上合并）

可以发现可重集中元素之和不会超过子树权值和，因此状态数不超过 $O(k^3partition(k))$，在 $k=20$ 时总状态数不超过 $10^4$。

因此可以记录这个状态进行 $dp$，可以使用如下方式，用一个二进制数表示可重集：

对于一个二进制数，从前往后考虑每一位，如果当前位是 $1$，则向 $S$ 插入一个 $1$，否则将 $S$ 中所有元素加一。

对于一个可重集，进行上述操作的逆操作即可得到对应的二进制数。

合并两个状态时，从低位到高位考虑两个二进制数，容易合并出新的状态以及增加的 $s$。

复杂度 $O(n^{1-\epsilon}+能过)$

~~复杂度难以分析，但是随机数据下完全可以接受~~

~~但这东西只能跑20，所以说这题只能填奇数，因为 $10^{10}$ 内最大的是 $3^{20},2^{33}$~~

bonus(2022)：

实际上可以发现，对于大的 $k$，可能的 $p$ 很少，因此对于大的 $k$ 部分可以不维护出现次数，而是对于每个 $p$ 求一次答案。这样可以显著加速过程。

经过一些优化，这样可以在 9s 内跑过 $k=33$。



min_25 筛做法简述：

考虑筛质数的过程，每次筛掉 $p_i$ 的倍数。设 $s_{n,k}$ 表示 $[1,n]$ 中满足最小质因子大于等于 $p_k$ 的所有数的 $f(x)$ 之和。

将这些数分为两部分考虑，第一部分为质数，这部分为所有大于等于 $p_k$ 小于等于 $n$ 的质数。设 $g_n$ 表示所有小于等于 $n$ 的质数的 $f(p)$ 之和。

第二部分为非质数，考虑枚举这些数的最小质因子 $p_i$，再枚举这个因子的次数 $j$，则接下来会转移到 $s_{\lfloor\frac n{p_i^j}\rfloor,i+1}$，因此得到如下结果：
$$
s_{n,k}=g_n-g_{p_{k-1}}+\sum_{i=k}^{+\infty}\sum_{j=1}^{+\infty}f(p_i^j)*(s_{\lfloor\frac n{p_i^j}\rfloor,i+1}+[j\geq 2])
$$
这里最后一项是为了考虑 $p^k$ 的情况。应该也可以在这里只枚举 $p_i$ 次数，将剩下的留给下一次枚举，但这和上述写法等价，且上述写法可以减少递归次数。

如果求出了 $g_n$，则可以直接按照递归式计算 $s$。可以发现 $p_k>n$ 的情况都是不需要再计算的，因此只需要枚举到小于等于 $\sqrt n$ 的质数，$j$ 也只需要枚举到 $p_i^j\leq n$ 部分。预处理部分只需要先求出这些质数。但这里可能会递归到下一个质数的状态（例如 $x=p_k*p_{k+1}$ ），虽然不会造成太大影响但还是需要注意处理。

可以证明复杂度是 $O(n^{1-\epsilon})$~~但我不会~~，一般能过 $10^{10}\sim 10^{11}$。



然后考虑计算 $g$，这里额外要求只考虑质数部分时， $f(p)$ 可以看成一个完全积性函数，例如 $f(p)=p$，一般的情况大多可以化为若干个完全积性函数的线性组合，可以对每部分分别求。

可以发现上述过程中只需要所有的 $g_{\lfloor\frac nk\rfloor}$，而这样的值只有 $O(\sqrt n)$ 个。

同样考虑每次筛质数。设 $t_{n,k}$ 表示 $[1,n]$ 中所有质数以及最小质因子大于等于 $p_k$ 的数的 $f(p)$ 之和。则 $k$ 枚举到小于等于 $\sqrt m$ 的质数时得到的即为所有质数部分的和。

考虑枚举这次删去的数，即所有是所有 $p_i$ 的倍数，且最小质因子大于等于 $p_i$，且不是质数的数。可以发现如果将这些数除以 $p_i$，则它们正好是最小质因子大于等于 $p_i$ 的所有数。因此有：
$$
t_{n,k+1}=t_{n,k}-t_{\lfloor\frac n{p_i}\rfloor,k}*f(p_k)+\sum_{j=1}^{k-1}f(p_j)[p_j\leq \frac n{p_i}]
$$
这里因为将 $f$ 看成完全积性，因此可以直接乘 $f(p_k)$。

可以发现，对于这个式子可以从小到大枚举 $k$，每次从大到小计算新的 $t$，这样在一个数组上即可完成计算。这里只需要记录所有 $\lfloor\frac mk\rfloor$ 的位置，可以按照如下方式标号：

对于小于等于 $\sqrt m$ 的位置按照原来方式标号，否则标号 $2\sqrt m-\frac mi$。

同时可以发现，对于一个 $k$，只有 $n\geq p_k^2$ 的部分会受到转移影响，因此可以只枚举后面一部分。

这时可以发现，对于小于等于 $n^{\frac 14}$ 的质数，这部分枚举复杂度不超过 $O(\frac{n^{\frac 34}}{\log n})$。对于后面部分，复杂度为 $\sum_{n^{\frac 14}\leq p_i\leq n^{\frac 12}}\frac{n}{p_i^2}$。积分可得这部分也是 $O(\frac{n^{\frac 34}}{\log n})$ 的。

这样就完成了整个 min_25 筛过程。好像可以证明对于 $n\leq 10^{13}$，总复杂度仍然是 $O(\frac{n^{\frac 34}}{\log n})$ 的。

###### Code

原std:

```cpp
#include<cstdio>
#include<cmath>
using namespace std;
#define mod 998244353
#define N 105
#define M 40100
#define K 21
#define S 200500
#define ll long long
int id[K][M],tid[N][K][M],id2[K][M],tid2[K][M],ha1[600500],dp[N][K][M],f[K][M],as2[S][K];
int n,k,a,b,head[N],cnt,s1[N][K],c1[K],as[K][N*K],mp1[2222222],mp2[5555],ct1;
ll m;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
struct state{int vl,v1;}tid1[M];
int fu1(state x){int st=x.vl*2800+x.v1;return st;}
state doit(state a,state b,int s)
{
	int r1=0,r2=0,c1=1,v2=0,s1=0,s2=0,tp=0;
	state c;c.v1=0;c.vl=a.vl+b.vl;
	a.v1=mp2[a.v1];b.v1=mp2[b.v1];
	if(!s)return c;
	while(a.v1)if(a.v1&1)r1+=1<<v2,v2++,a.v1>>=1;
	else if(c1<s)c1++,v2++,a.v1>>=1;
	else a.v1>>=1;
	c1=1;v2=0;
	while(b.v1)if(b.v1&1)r2+=1<<v2,v2++,b.v1>>=1;
	else if(c1<s)c1++,v2++,b.v1>>=1;
	else b.v1>>=1;
	tp=1;v2=0;
	while(r1+r2)
	if(r1&1)c.vl+=s2,s1+=tp,c.v1+=1<<v2,v2++,r1>>=1;
	else if(r2&1)c.vl+=s1,s2+=tp,c.v1+=1<<v2,v2++,r2>>=1;
	else r1>>=1,r2>>=1,v2++,tp++;
	c.v1=mp1[c.v1];
	return c;
}
void ins1(int d,state x,int vl)
{
	int vl1=fu1(x);
	if(!ha1[vl1])ha1[vl1]=++cnt,tid1[cnt]=x;
	int fu2=ha1[vl1];
	if(!id2[d][fu2])id2[d][fu2]=++c1[d],tid2[d][c1[d]]=fu2;
	fu2=id2[d][fu2];
	f[d][fu2]=(f[d][fu2]+vl)%mod;
}
void ins2(int u,int d,state x,int vl)
{
	int vl1=fu1(x);
	if(!ha1[vl1])ha1[vl1]=++cnt,tid1[cnt]=x;
	int fu2=ha1[vl1];
	if(!id[d][fu2])id[d][fu2]=++s1[u][d],tid[u][d][s1[u][d]]=fu2;
	fu2=id[d][fu2];
	dp[u][d][fu2]=(dp[u][d][fu2]+vl)%mod;
}
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
	for(int i=0;i<=k;i++)
	{
		state fuc;
		fuc.vl=0;fuc.v1=mp1[!i?0:(1<<i-1)];
		for(int j=0;j<=k;j++)
		{
			for(int l=1;l<=c1[j];l++)id2[j][tid2[j][l]]=0,tid2[j][l]=0,f[j][l]=0;
			c1[j]=0;
		}
		ins1(i,fuc,1);
		for(int j=head[u];j;j=ed[j].next)if(ed[j].t!=fa)
		for(int l1=k;l1>=0;l1--)
		for(int v1=1;v1<=c1[l1];v1++)
		for(int l2=1;l1+l2<=k;l2++)
		for(int v2=1;v2<=s1[ed[j].t][l2];v2++)
		ins1(l1+l2,doit(tid1[tid[ed[j].t][l2][v2]],tid1[tid2[l1][v1]],i),1ll*dp[ed[j].t][l2][v2]*f[l1][v1]%mod);
		for(int j=1;j<=k;j++)
		for(int l=1;l<=c1[j];l++)
		ins2(u,j,tid1[tid2[j][l]],f[j][l]);
	}
	for(int j=0;j<=k;j++)
	for(int l=0;l<=s1[u][j];l++)id[j][tid[u][j][l]]=0;
}
bool check(int s)
{
	int tp1=1,as=0;
	while(s)if(s&1)as+=tp1,s>>=1;
	else tp1++,s>>=1;
	return as<=k;
}
int ch[S],pr[S],ct,p,f3[S],su[S];
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
}
int getid(ll x){return x<=p?x:p*2-(1ll*p*p==m)-m/x+1;}
ll gettid(int x){if(x<=p)return x;x=2*p-(1ll*p*p==m)+1-x;return m/x;}
void init()
{
	int tp=p*2-(1ll*p*p==m);
	for(int i=1;i<=tp;i++)
	{
		ll s1=gettid(i),s2=s1+1;
		if(s1%2)s2/=2;
		else s1/=2;
		s1%=mod,s2%=mod;
		f3[i]=(1ll*s1*s2-1)%mod;
	}
	for(int i=1;i<=ct;i++)su[i]=su[i-1]+pr[i];
	for(int i=1;i<=ct;i++)
	for(int j=tp;j>=1;j--)
	{
		if(1ll*pr[i]*pr[i]>gettid(j))break;
		f3[j]=(f3[j]-1ll*f3[getid(gettid(j)/pr[i])]*pr[i]%mod+1ll*su[i-1]*pr[i]%mod+mod)%mod;
	}
}
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int solve2(ll m){return 1ll*n*(f3[getid(m)]-2*(m>1)+mod)%mod;}
int solve(int a,int p)
{
	if(a==2)return 0;
	if(as2[a][p])return as2[a][p];
	int as1=0,st=pw(a,p);
	for(int i=0;i<=p*(p-1)/2;i++)as1=(as1+1ll*as[p][i]*st)%mod,st=1ll*st*a%mod;
	return as2[a][p]=as1;
}
int solveas(ll m,ll p)
{
	if(m<=1||pr[p]>m||(p>ct&&pr[ct]>=m))return 0;
	int ans=solve2(m);
	ans=(ans-1ll*n*(su[p-1]-(p>1)*2)%mod+mod)%mod;
	for(int j=p;j<=ct&&1ll*pr[j]*pr[j]<=m;j++)
	for(ll s=pr[j],tp=1;s<=m;s*=pr[j],tp++)
	ans=(ans+1ll*((s!=pr[j])+solveas(m/s,j+1))*solve(pr[j],tp))%mod;
	return ans;
}
int main()
{
	scanf("%d%lld",&n,&m);k=20;
	for(int i=0;i<=1050000;i++)if(check(i))mp1[i]=++ct1,mp2[ct1]=i;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	for(int i=1;i<=k;i++)
	for(int j=1;j<=s1[1][i];j++)
	as[i][tid1[tid[1][i][j]].vl]=(as[i][tid1[tid[1][i][j]].vl]+dp[1][i][j])%mod;
	prime(p=sqrt(m));
	init();
	printf("%d\n",(solveas(m,1)+1)%mod);
}
```

重写版本：

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<vector>
#include<map>
using namespace std;
#define mod 998244353
#define ll long long
//tree
#define N 105
int n,a,b,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
//states
#define M 2725
#define K 21
int c1,si[M],sz[M],li=20;
map<int,int> mp;
void dfs(int k,int su,int ct,int sv)
{
	if(su==k){si[++c1]=sv;sz[c1]=k;mp[sv]=c1;return;}
	if(su>k)return;
	if(ct)dfs(k,su+ct,ct,sv<<1);
	dfs(k,su+1,ct+1,sv<<1|1);
}
vector<pair<int,int> > trs[M];
int tr1[M][K];
pair<int,int> calc_tr(int a,int b)
{
	int s1=0,s2=0,v1=0,vl=0,rs=0,r2=1;
	while(a||b)
	{
		v1++;
		while(a&1)vl+=s2,s1+=v1,rs+=r2,r2<<=1,a>>=1;
		while(b&1)vl+=s1,s2+=v1,rs+=r2,r2<<=1,b>>=1;
		r2<<=1;a>>=1;b>>=1;
	}
	return make_pair(mp[rs],vl);
}
int calc_r1(int a,int x)
{
	if(!x)return 1;
	int rs=0,r2=1,vl=0;
	while(a)
	{
		if(a&1)rs+=r2,r2<<=1;
		else
		{
			vl++;
			if(vl<x)r2<<=1;
		}
		a>>=1;
	}
	return mp[rs];
}
void init_st()
{
	for(int i=0;i<=li;i++)dfs(i,0,0,0);
	for(int i=1;i<=c1;i++)
	{
		trs[i].push_back(make_pair(0,0));
		for(int j=1;sz[i]+sz[j]<=li&&j<=c1;j++)
		trs[i].push_back(calc_tr(si[i],si[j]));
	}
	for(int i=1;i<=c1;i++)for(int j=0;j<=li;j++)tr1[i][j]=calc_r1(si[i],j);
}

struct state{int vl,cr;}st[M];
int hs[M],c2,ci[K];
void init_s1()
{
	for(int i=1;i<=c1;i++)hs[i]=i,st[i]=(state){0,i},ci[sz[i]]=i;
	c2=c1;
}
void init_s2()
{
	c2=0;li=7;
	for(int i=1;i<=c1;i++)hs[i]=0;
	for(int i=0;i<=li;i++)
	for(int j=0;j<=i*(i-1)/2;j++)
	for(int k=1;sz[k]<=i;k++)if(!hs[k+j*50])
	hs[k+j*50]=++c2,st[c2]=(state){j,k},ci[i]=c2;
}

int dp[N][K][M],fg,s1[K][M],s2[K][M];
int pw[M];
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
	for(int r1=0;r1<=li;r1++)
	{
		s1[r1][hs[mp[r1?1<<r1-1:0]]]=1;
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
		{
			int t=ed[i].t;
			for(int p=0;p<=li;p++)
			for(int v1=1;v1<=ci[p];v1++)if(s1[p][v1])
			for(int q=0;q<=li-p;q++)
			for(int v2=1;v2<=ci[q];v2++)
			{
				int vl=1ll*s1[p][v1]*dp[t][q][v2]%mod;
				if(!vl)continue;
				int rv=st[v1].vl+st[v2].vl,rs;
				pair<int,int> si=trs[st[v1].cr][tr1[st[v2].cr][r1]];
				rs=si.first;
				if(fg)vl=1ll*vl*pw[si.second]%mod;else rv+=si.second;
				s2[p+q][hs[rs+rv*50]]=(s2[p+q][hs[rs+rv*50]]+vl)%mod;
			}
			for(int p=0;p<=li;p++)
			for(int v1=1;v1<=ci[p];v1++)
			s1[p][v1]=s2[p][v1],s2[p][v1]=0;
		}
		for(int p=0;p<=li;p++)
		for(int v1=1;v1<=ci[p];v1++)dp[u][p][v1]=(dp[u][p][v1]+s1[p][v1])%mod,s1[p][v1]=0;
	}
}
//prime
ll m;
#define S 200500
int pr[S],is[S],ct,pi;
void init_pr(int n)
{
	for(int i=2;i<=n;i++)
	{
		if(!is[i])pr[++ct]=i;
		for(int j=1;j<=ct&&i*pr[j]<=n;j++)
		{
			is[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
}
//precalc f(p^k)
int sv[K*M][K*2],cr[K][M];
void calc_vl()
{
	init_s1();fg=1;
	for(int i=2;pr[i]<=17&&i<=ct;i++)
	{
		ll tp=1;li=0;
		while(pr[i]*tp<=m)tp*=pr[i],li++;
		pw[0]=1;for(int j=1;j<=li*li;j++)pw[j]=1ll*pw[j-1]*pr[i]%mod;
		for(int j=1;j<=n;j++)
		for(int p=0;p<=li;p++)
		for(int v1=1;v1<=ci[p];v1++)
		dp[j][p][v1]=0;
		dfs(1,0);
		int v1=1;
		for(int j=0;j<=li;j++)
		{
			for(int k=1;k<=ci[j];k++)sv[i][j]=(sv[i][j]+1ll*dp[1][j][k]*v1)%mod;
			v1=1ll*v1*pr[i]%mod;
		}
	}
	init_s2();fg=0;
	for(int j=1;j<=n;j++)
	for(int p=0;p<=li;p++)
	for(int v1=1;v1<=ci[p];v1++)
	dp[j][p][v1]=0;
	dfs(1,0);
	for(int j=0;j<=li;j++)
	for(int k=1;k<=ci[j];k++)
	cr[j][st[k].vl+j]=(cr[j][st[k].vl+j]+dp[1][j][k])%mod;
	for(int i=1;i<=ct;i++)if(pr[i]>17)
	{
		ll tp=1,nw=0;
		while(tp<=m)
		{
			int v1=1;
			for(int j=0;j<=nw*(nw+1)/2;j++)sv[i][nw]=(sv[i][nw]+1ll*cr[nw][j]*v1)%mod,v1=1ll*v1*pr[i]%mod;
			tp*=pr[i];nw++;
		}
	}
}

//min25
int g[S],sp[S];
int getid(ll x){return x>pi?pi*2+1-m/x:x;}
ll gettid(int x){return x>pi?m/(pi*2+1-x):x;}
void calc_g()
{
	for(int i=1;i<=ct;i++)sp[i]=(sp[i-1]+pr[i])%mod;
	for(int i=1;i<=pi*2;i++)
	{
		int vl=gettid(i)%mod;
		g[i]=1ll*vl*(vl+1)%mod*(mod+1)/2%mod-1;
	}
	for(int i=1;i<=ct;i++)
	for(int j=pi*2;j>=1;j--)
	{
		ll vl=gettid(j);
		if(1ll*pr[i]*pr[i]>vl)break;
		g[j]=(g[j]+mod-1ll*pr[i]*(g[getid(vl/pr[i])]+mod-sp[i-1])%mod)%mod;
	}
}
int getg(ll v)
{
	v=getid(v);
	return 1ll*n*(g[v]-2*(v>1)+mod)%mod;
}
int calc_f(ll n,int k)
{
	if(pr[k]>n||(k>ct&&pr[ct]>n))return 0;
	int as=(getg(n)-getg(pr[k-1])+mod)%mod;
	for(int i=k;i<=ct&&1ll*pr[i]*pr[i]<=n;i++)
	for(ll j=1,sp=pr[i];sp<=n;j++,sp*=pr[i])
	as=(as+1ll*(calc_f(n/sp,i+1)+(j>1))*sv[i][j])%mod;
	return as;
}
int main()
{
	scanf("%d%lld",&n,&m);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	init_st();
	init_pr(pi=sqrt(m));
	calc_vl();
	calc_g();
	printf("%lld\n",(calc_f(m,1)+1)%mod);
}
```

能8s跑偶数的代码：

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<vector>
#include<map>
using namespace std;
#define mod 998244353
#define ll long long
//tree
#define N 105
int n,a,b,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
//states
#define M 10151
#define K 35
int c1,sz[M],li=33;
ll si[M];
map<ll,int> mp;
ll gettrs2(ll a)
{
	int s1=0,s2=1;
	for(int i=0;i<40;i++)if((a>>i)&1)s1+=s2;else s2++;
	s1=li-s1;
	ll rs=0,r2=1,vl=0;
	while(a)
	{
		if(a&1)rs+=r2,r2<<=1;
		else
		{
			vl++;
			if(vl<s1)r2<<=1;
		}
		a>>=1;
	}
	return rs;
}
void dfs(int k,int su,int ct,ll sv)
{
	if(su==k){if(gettrs2(sv)!=sv)return;si[++c1]=sv;sz[c1]=k;mp[sv]=c1;return;}
	if(su>k)return;
	if(ct)dfs(k,su+ct,ct,sv<<1);
	dfs(k,su+1,ct+1,sv<<1|1);
}
vector<pair<int,int> > trs[M];
int tr1[M][K];
pair<int,int> calc_tr(ll a,ll b)
{
	int s1=0,s2=0,v1=0,vl=0;
	ll rs=0,r2=1;
	while(a||b)
	{
		v1++;
		while(a&1)vl+=s2,s1+=v1,rs+=r2,r2<<=1,a>>=1;
		while(b&1)vl+=s1,s2+=v1,rs+=r2,r2<<=1,b>>=1;
		r2<<=1;a>>=1;b>>=1;
	}
	rs=gettrs2(rs);
	return make_pair(mp[rs],vl);
}
int calc_r1(ll a,int x)
{
	if(!x)return 1;
	ll rs=0,r2=1,vl=0;
	while(a)
	{
		if(a&1)rs+=r2,r2<<=1;
		else
		{
			vl++;
			if(vl<x)r2<<=1;
		}
		a>>=1;
	}
	return mp[rs];
}
void init_st()
{
	for(int i=0;i<=li;i++)dfs(i,0,0,0);
	for(int i=1;i<=c1;i++)
	{
		trs[i].push_back(make_pair(0,0));
		for(int j=1;sz[i]+sz[j]<=li&&j<=c1;j++)
		trs[i].push_back(calc_tr(si[i],si[j]));
	}
	for(int i=1;i<=c1;i++)for(int j=0;j<=li;j++)tr1[i][j]=calc_r1(si[i],j);
}

struct state{int vl,cr;}st[M];
int hs[M],c2,ci[K],tpvl;
void init_s1()
{
	for(int i=1;i<=c1;i++)hs[i]=i,st[i]=(state){0,i},ci[sz[i]]=i;
	c2=c1;
}
void init_s2()
{
	c2=0;li=7;tpvl=50;
	for(int i=1;i<=c1;i++)hs[i]=0;
	for(int i=0;i<=li;i++)
	for(int j=0;j<=i*(i-1)/2;j++)
	for(int k=1;sz[k]<=i;k++)if(!hs[k+j*tpvl])
	hs[k+j*tpvl]=++c2,st[c2]=(state){j,k},ci[i]=c2;
}

int dp[N][K][M],fg,s1[K][M],s2[K][M];
int pw[M];
void dfs(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
	for(int r1=0;r1<=li;r1++)
	{
		s1[r1][hs[mp[r1?gettrs2(1ll<<r1-1):0]]]=1;
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
		{
			int t=ed[i].t;
			for(int p=0;p<=li;p++)
			for(int v1=1;v1<=ci[p];v1++)if(s1[p][v1])
			for(int q=0;q<=li-p;q++)
			for(int v2=1;v2<=ci[q];v2++)if(dp[t][q][v2])
			{
				int vl=1ll*s1[p][v1]*dp[t][q][v2]%mod;
				pair<int,int> si=trs[st[v1].cr][tr1[st[v2].cr][r1]];
				int rv=st[v1].vl+st[v2].vl,rs=si.first;
				if(fg)vl=1ll*vl*pw[si.second]%mod;else rv+=si.second,rs=hs[rs+rv*tpvl];
				s2[p+q][rs]=(s2[p+q][rs]+vl)%mod;
			}
			for(int p=0;p<=li;p++)
			for(int v1=1;v1<=ci[p];v1++)
			s1[p][v1]=s2[p][v1],s2[p][v1]=0;
		}
		for(int p=0;p<=li;p++)
		for(int v1=1;v1<=ci[p];v1++)dp[u][p][v1]=(dp[u][p][v1]+s1[p][v1])%mod,s1[p][v1]=0;
	}
}
//prime
ll m;
#define S 200500
int pr[S],is[S],ct,pi;
void init_pr(int n)
{
	for(int i=2;i<=n;i++)
	{
		if(!is[i])pr[++ct]=i;
		for(int j=1;j<=ct&&i*pr[j]<=n;j++)
		{
			is[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
}
//precalc f(p^k)
int sv[M][K],cr[K][M];
void calc_vl()
{
	init_s1();fg=1;
	for(int i=1;pr[i]<=17&&i<=ct;i++)
	{
		ll tp=1;li=0;
		while(pr[i]*tp<=m)tp*=pr[i],li++;
		pw[0]=1;for(int j=1;j<=li*li;j++)pw[j]=1ll*pw[j-1]*pr[i]%mod;
		for(int j=1;j<=n;j++)
		for(int p=0;p<=li;p++)
		for(int v1=1;v1<=ci[p];v1++)
		dp[j][p][v1]=0;
		dfs(1,0);
		int v1=1;
		for(int j=0;j<=li;j++)
		{
			for(int k=1;k<=ci[j];k++)sv[i][j]=(sv[i][j]+1ll*dp[1][j][k]*v1)%mod;
			v1=1ll*v1*pr[i]%mod;
		}
	}
	init_s2();fg=0;
	for(int j=1;j<=n;j++)
	for(int p=0;p<=li;p++)
	for(int v1=1;v1<=ci[p];v1++)
	dp[j][p][v1]=0;
	dfs(1,0);
	for(int j=0;j<=li;j++)
	for(int k=1;k<=ci[j];k++)
	cr[j][st[k].vl+j]=(cr[j][st[k].vl+j]+dp[1][j][k])%mod;
	for(int i=1;i<=ct;i++)if(pr[i]>17)
	{
		ll tp=1,nw=0;
		while(tp<=m)
		{
			int v1=1;
			for(int j=0;j<=nw*(nw+1)/2;j++)sv[i][nw]=(sv[i][nw]+1ll*cr[nw][j]*v1)%mod,v1=1ll*v1*pr[i]%mod;
			tp*=pr[i];nw++;
		}
	}
}

//min25
int g[S],sp[S];
int getid(ll x){return x>pi?pi*2+1-m/x:x;}
ll gettid(int x){return x>pi?m/(pi*2+1-x):x;}
void calc_g()
{
	for(int i=1;i<=ct;i++)sp[i]=(sp[i-1]+pr[i])%mod;
	for(int i=1;i<=pi*2;i++)
	{
		int vl=gettid(i)%mod;
		g[i]=1ll*vl*(vl+1)%mod*(mod+1)/2%mod-1;
	}
	for(int i=1;i<=ct;i++)
	for(int j=pi*2;j>=1;j--)
	{
		ll vl=gettid(j);
		if(1ll*pr[i]*pr[i]>vl)break;
		g[j]=(g[j]+mod-1ll*pr[i]*(g[getid(vl/pr[i])]+mod-sp[i-1])%mod)%mod;
	}
}
int getg(ll v)
{
	v=getid(v);
	return 1ll*n*g[v]%mod;
}
int calc_f(ll n,int k)
{
	if(pr[k]>n||(k>ct&&pr[ct]>n))return 0;
	int as=(getg(n)-getg(pr[k-1])+mod)%mod;
	for(int i=k;i<=ct&&1ll*pr[i]*pr[i]<=n;i++)
	for(ll j=1,sp=pr[i];sp<=n;j++,sp*=pr[i])
	as=(as+1ll*(calc_f(n/sp,i+1)+(j>1))*sv[i][j])%mod;
	return as;
}
int main()
{
	scanf("%d%lld",&n,&m);li=0;
	while((1ll<<li+1)<=m)li++;if(li<8)li=8;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	init_st();
	init_pr(pi=sqrt(m));
	calc_vl();
	calc_g();
	printf("%lld\n",(calc_f(m,1)+1)%mod);
}
```



#### SCOI2020模拟? 

出题人:wkr&zyw

##### auoj26 欢迎来到塞莱斯特山

###### Problem

给一棵 $n$ 个点，以 $1$ 为根的有根树，定义 $dep(i)$ 为 $1$ 到 $i$ 路径经过的点数，$S$ 为所有 $n$ 阶排列组成的集合。求：

$$
\sum_{p\in S}\prod_{i=1}^{n-1}dep(LCA(p_i,p_{i+1}))
$$

答案模 $10^9+7$

$n\leq 500$

$2s,1024MB$

###### Sol

考虑将 $\prod dep$ 拆成前缀和的形式，即：
$$
\prod_{i=2}^n(\frac{dep(i)}{dep(fa_i)})^{\sum_{i=1}^{n-1}[LCA(p_i,p_{i+1})\in subtree_i]}
$$
那么对于每个 $i$，它贡献的次数为排列中相邻两个点都在 $i$ 子树内的对数，即 $i$ 子树内点数减去 $i$ 子树内的点在排列中构成的段数。

那么考虑设 $dp_{i,j}$ 表示考虑 $i$ 子树内的点进行排列，排列成 $j$ 个有序段的所有方案，求和这些方案在子树内的贡献的结果。

考虑一个点的转移，~~和#20一样~~，对于每个子树容斥有多少个相邻段在当前点所有子树的所有段排列的过程中相邻，相当于合并 $i$ 段，系数为 $(-1)^iC_{c-1}^i$。然后将不同子树间容斥后的结果合并，可以看成多项式相乘。最后考虑子树合并了 $k$ 段，向上时合并成 $j$ 个段的方案数，可以发现系数为 $C_{k-1}^{j-1}$。最后再对于 $dp_{i,j}$ 乘上 $(\frac{dep_i}{dep_{fa_i}})^{j}$ 的贡献系数即可。

复杂度 $O(n^3)$，瓶颈为对每个子树向上时的容斥为 $O(sz^2)$。



以下是一些乱搞：

考虑除去根外的一个子树，这个子树上首先做了合并：（设 $f$ 为 $dp$ 数组）
$$
f_i=\sum_{j\geq i}C_{j-1}^{i-1}f'_i
$$
然后乘以系数
$$
f_i=f_i'*(\frac{dep_u}{dep_{fa_u}})^{sz_u-i}
$$
然后向上再做了容斥：
$$
f_i=\sum_{j\geq i}C_{j-1}^{i-1}(-1)^{j-i}f'_i
$$
考虑第一步的一个 $f_i'$ 到最后一步的 $f_i$ 的贡献，可以得到如下结果：
$$
f_i=\sum_{k\geq j\geq i}(-1)^{j-i}C_{k-1}^{j-1}C_{j-1}^{i-1}(\frac{dep_u}{dep_{fa_u}})^{sz_u-j}f_k'\\
f_i=\sum_{k\geq i}C_{k-1}^{i-1}(\frac{1}{dep_{fa_u}})^{k-i}(\frac{dep_u}{dep_{fa_u}})^{sz_u-k}f_k'
$$
这里是因为树没有边权，$dep_{fa_i}=dep_i-1$

然后这个也可以看成如下结果：
$$
f_i=\sum_{k\geq i}C_{k-1}^{i-1}(\frac{1}{dep_{fa_u}})^{sz_u-i}(dep_u)^{sz_u-k}f_k'
$$
但到这里可以发现，向上合并的过程中，每个子树都会贡献 $(\frac{1}{dep_{fa_u}})^{-i}$，如果总共有 $k$ 段，则除去当前点的一段外每一段来自子树的都有这个贡献，因此有一个 $(\frac 1{dep_{fa_u}})^{-k+1}$ 的贡献，而接下来会再乘上一个 $(dep_u)^{-k}$，这一部分会进行互相抵消，因此可以消去这里，得到：
$$
f_i=(\frac{1}{dep_{fa_u}})^{sz_u}(dep_u)^{sz_u-1}\sum_{k\geq i}C_{k-1}^{i-1}f_k'
$$
这相当于每个点的 $dp$ 再乘以一个系数，因为子树合并可以看成直接相乘，因此可以看成整体系数乘以每个系数的和。

这时可以发现，每个 $dep_u$ 会被乘 $sz_u-1$ 次，接下来会被除 $\sum_{v\in son_u}sz_v$ 次，而这正好完全抵消，从而转移可以写成：
$$
f_i=\sum_{k\geq i}C_{k-1}^{i-1}f_k'
$$
可以发现 $1$ 处也是这个转移，直接用即可。最后答案仍然为 $dp_{1,1}$。

这样复杂度还是 $O(n^3)$，因为OGF下的 $f(x)=g(x-1)$ 和EGF乘法组合起来看起来没有好的性质，难以使用GF优化。但这样的常数非常小，在极限数据下 $n=500$ 只需要 50 ms。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505
#define mod 1000000007
int n,c[N][N];
int a,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
int sz[N],dp[N][N];
void dfs(int u,int fa)
{
	sz[u]=1;dp[u][1]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs(ed[i].t,u);
		int t=ed[i].t;
		for(int j=sz[u];j>=1;j--)
		{
			for(int k=1;k<=sz[t];k++)dp[u][j+k]=(dp[u][j+k]+1ll*dp[u][j]*dp[t][k]%mod*c[j+k][k])%mod;
			dp[u][j]=0;
		}
		sz[u]+=sz[t];
	}
	for(int j=1;j<=sz[u];j++)for(int k=1;k<j;k++)dp[u][k]=(dp[u][k]+1ll*dp[u][j]*c[j-1][k-1])%mod;
}
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)scanf("%d",&a),adde(a,i);
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	dfs(1,0);
	printf("%d\n",dp[1][1]);
}
```



##### auoj27 感受清风

###### Problem

有一个 $n$ 行 $m$ 列的网格，一些位置上有箱子，初始时第 $i$ 行的左侧 $v_i$ 个位置有箱子：

$q$ 次操作，每次操作为如下几种之一：

1. 在一个原来没有箱子的位置放一个箱子。
2. 删去一个位置上的箱子，保证箱子存在。
3. 将所有箱子向左侧推，直到推不动为止。
4. 将所有箱子向右侧推，直到推不动为止。
5. 给定一个位置，询问从这个位置开始向上走，只能走有箱子的位置，求最多能走多少格。
6. 询问从一个位置开始向下，只能走有箱子的位置，最多能走多少格。

$n,m,q\leq 10^6$

$3s,1024MB$

###### Sol

考虑只有询问，则每次询问相当于从一个点开始按照某个方向走，找到第一个 $v_i<x$ 的位置。线段树即可 $O(\log n)$ 求出答案。

考虑 $1,2$ 操作，将操作的位置看成关键位置。一次询问只和一列的状态有关，考虑一列的情况。

对于一列，可以发现关键位置将这一列分成了若干个区间。考虑维护关键位置中有哪些位置没有箱子，同时中间的区间中每一个区间内部是否存在一个没有箱子（$v_i<x$）的位置。考虑每一列使用 `set `维护没有箱子的关键位置以及不能走过去的区间，则一次修改复杂度为 $O(\log n)$

对于一次询问，首先线段树上判断初始位置所在的区间内的情况，如果可以走到当前区间端点，则考虑停止的位置，它可能是一个关键位置或者一段区间中的一个位置。只需要找到 `set` 中端点往后的第一个位置，然后根据这个位置是一个区间还是关键位置可以进行判断。需要特殊处理没有关键位置的情况或者一些其它边界情况。

对于一个区间内的情况，相当于找一个区间内第一个 $v_i<x$ 的点，维护区间 $\max$ 即可 $O(\log n)$ 线段树上二分。两个方向的方式相同。

最后考虑 $3,4$ 操作，可以发现这相当于变成初始状态，只需要处理出新的 $v_i$，然后清空所有关键位置相关。通过维护的关键位置状态即可得到每一行 $v_i$ 改变的情况，且显然总共只会改变 $O(q)$ 次，可以线段树上直接修改。

对于向右推的情况，可以看成对网格进行镜像，即将接下来的这一段操作全部令 $y'=m+1-y$，然后看成向左推。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 1005090
int n,m,q,fg,a,b,v[N];
char s[11];
struct segt{
	struct node{int l,r,mn;}e[N*4];
	void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].mn=v[l];return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int s)
	{
		if(e[x].l==e[x].r){e[x].mn=v[e[x].l];return;}
		int mid=(e[x].l+e[x].r)>>1;
		modify(x<<1|(mid<s),s);
		pushup(x);
	}
	int query(int x,int l,int r,int v,int f)
	{
		if(e[x].l>r||e[x].r<l||e[x].mn>=v)return -1;
		if(e[x].l==e[x].r)return e[x].l;
		int as=query(x<<1|f,l,r,v,f);
		if(as==-1)as=query(x<<1|(!f),l,r,v,f);
		return as;
	}
}tr;
set<int> si,st[N];
set<pair<int,int> > sr[N];
void modify(int y,int x,int f)
{
	si.insert(x);
	if(sr[x].find(make_pair(y,!f))!=sr[x].end())
	{
		sr[x].erase(make_pair(y,!f));
		sr[x].insert(make_pair(y,f));
		if(f)st[x].erase(y);else st[x].insert(y);
		return;
	}
	sr[x].insert(make_pair(y,f));
	set<pair<int,int> >::iterator it=sr[x].find(make_pair(y,f));
	int lb=0,rb=0;
	if(it!=sr[x].begin())
	{
		it--;
		lb=(*it).first;
		it++;
	}
	it++;
	if(it!=sr[x].end())rb=(*it).first;
	if(lb&&rb)
	{
		int tp=tr.query(1,lb+1,rb-1,x,0);
		if(tp!=-1)st[x].erase(tp);
	}
	if(lb)
	{
		int tp=tr.query(1,lb+1,y-1,x,0);
		if(tp!=-1)st[x].insert(tp);
	}
	if(rb)
	{
		int tp=tr.query(1,y+1,rb-1,x,0);
		if(tp!=-1)st[x].insert(tp);
	}
	if(!f)st[x].insert(y);
}
int di[N];
void init()
{
	set<int> tp;
	while(si.size())
	{
		int u=*si.begin();si.erase(u);
		for(set<pair<int,int> >::iterator it=sr[u].begin();it!=sr[u].end();it++)
		{
			pair<int,int> t1=*it;
			int v1=t1.first,v2=t1.second-(v[v1]>=u);
			if(v2)di[v1]+=v2,tp.insert(v1);
		}
		sr[u].clear();st[u].clear();
	}
	while(tp.size())
	{
		int u=*tp.begin();tp.erase(u);
		v[u]+=di[u];di[u]=0;
		tr.modify(1,u);
	}
}
int queryl(int y,int x)
{
	int lb=0;
	set<pair<int,int> >::iterator it=sr[x].lower_bound(make_pair(y+1,0));
	if(it!=sr[x].begin())
	{
		it--;
		lb=(*it).first;
	}
	int v1=tr.query(1,lb+1,y,x,1);
	if(v1!=-1)return y-v1;
	set<int>::iterator it2=st[x].lower_bound(y+1);
	if(it2!=st[x].begin())
	{
		it2--;
		int vl=*it2;
		int v2=(*sr[x].lower_bound(make_pair(vl,0))).first;
		if(sr[x].find(make_pair(v2,0))!=sr[x].end())return y-v2;
		int ls=tr.query(1,1,v2-1,x,1);
		if(ls!=-1)return y-ls;
	}
	if(lb)
	{
		int l1=(*sr[x].begin()).first;
		int ls=tr.query(1,1,l1-1,x,1);
		if(ls!=-1)return y-ls;
	}
	return y;
}
int queryr(int y,int x)
{
	int rb=n+1;
	set<pair<int,int> >::iterator it=sr[x].lower_bound(make_pair(y,0));
	if(it!=sr[x].end())rb=(*it).first;
	int v1=tr.query(1,y,rb-1,x,0);
	if(v1!=-1)return v1-y;
	set<int>::iterator it2=st[x].lower_bound(y);
	if(it2!=st[x].end())
	{
		int vl=*it2;
		int v2=(*(--sr[x].lower_bound(make_pair(vl+1,0)))).first;
		if(sr[x].find(make_pair(v2,0))!=sr[x].end())return v2-y;
		int ls=tr.query(1,v2+1,n,x,0);
		if(ls!=-1)return ls-y;
	}
	if(rb!=n+1)
	{
		int r1=(*sr[x].rbegin()).first;
		int ls=tr.query(1,r1+1,n,x,0);
		if(ls!=-1)return ls-y;
	}
	return n+1-y;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	tr.build(1,1,n);
	scanf("%d",&q);
	while(q--)
	{
		scanf("%s",s+1);
		if(s[1]=='l'||s[1]=='r')
		init(),fg=s[1]=='r';
		else
		{
			scanf("%d%d",&a,&b);
			if(fg)b=m+1-b;
			if(s[1]=='a'||s[1]=='d')modify(a,b,s[1]=='a');
			else if(s[2]=='u')printf("%d\n",queryl(a,b));
			else printf("%d\n",queryr(a,b));
		}
	}
}
```



##### auoj28 我的朋友们

###### Problem

有 $n$ 个物品，每个物品有一个权值 $p_i(0<p_i<1)$。给定正整数 $k$。

两个人进行游戏，初始时第一个人拿着编号为 $[1,k]$ 的物品，游戏按照轮进行：

每一轮，第一个人会将自己拿着的每一个物品向第二个人进行询问。如果第一个人询问第 $i$ 个物品，则第二个人有 $p_i$ 的概率回答 $1$，$1-p_i$ 的概率回答 $0$。第二个人所有的回答之间独立（即使是多次询问同一个物品）。

设 $k$ 次询问的回答总和为 $s$。则第一个人会扔掉拿着的编号最小的 $s$ 个物品，并从剩下的物品中拿 $s$ 个编号最小的。如果物品不够了，则结束这个过程。即设当前第一个人拿着的物品为 $[i,i+k-1]$，则接下来他会拿物品 $[i+s,i+s+k-1]$。如果 $i+s+k-1>n$ 则结束游戏。

求游戏进行的轮数的期望值，答案模 $998244353$

$n\leq 10^5$

$3s,1024MB$

###### Sol

设 $dp_i$ 表示第一个人拿着物品 $[i,i+k-1]$ 的期望次数。则答案为 $\sum_{i=1}^{n-k+1}dp_i$。

考虑求 $dp$，则可以先计算从之前的状态转移过来的次数，再乘上再这个状态期望停留的次数，显然后者为 $\frac 1{1-\prod_{j=i}^{i+k-1}(1-p_j)}$。

然后考虑这个点向后转移，如果将 $dp$ 看成生成函数，则向后的转移可以看成这一项 $dp_ix^i$ 乘以 $\prod_{j=i}^{i+k-1}(p_ix+1-p_i)$。

考虑分治优化 $dp$。因为这里向后的转移系数都是 $k$ 次多项式，考虑将序列按照长度为 $k$ 分段，则只需要对于每一段求出一段内部的转移以及一段向下一段的转移。

按照长度为 $k$ 分段后，可以发现一段内向下一段的转移多项式为 $\sum_{i=1}^kdp_ix^i*\prod_{j=i}^{i+k-1}F_j(x)$，这里 $F_i(x)=p_ix+1-p_i$。

再考虑分治的时候 $[l,mid]$ 向右侧转移的系数，可以发现这个系数也是 $\sum_{i=l}^{mid}dp_ix^i*\prod_{j=i}^{i+k-1}F_j(x)$ 中的一些项，因此考虑分治求这个。

可以注意到，对于一个区间 $l,r$，$\sum_{i=l}^rdp_ix^i*\prod_{j=i}^{i+k-1}F_j(x)$ 有公共项 $x^l\prod_{j=r+1}^{l+k-1}F_j(x)$，因此考虑维护公共项之外的部分，即：
$$
\sum_{i=l}^rdp_ix^{i-l}*\prod_{j=i}^{r}F_j(x)*\prod_{j=l+k}^{i+k-1}F_j(x)
$$
可以发现一个区间 $[l,r]$ 的这部分项次数不超过 $2(r-l)$。记这个结果为 $G_{l,r}(x)$，则可以通过如下方式合并：
$$
G_{l,r}(x)=G_{l,mid}(x)*\prod_{i=mid+1}^rF_i(x)+G_{mid+1,r}(x)*\prod_{i=l}^{mid}F_{i+k}(x)
$$
那么可以先做一遍分治+`fft` 求出每个区间的 $\prod_{i=l}^rF_i(x),\prod_{i=l}^rF_{i+k}(x)$，然后再分治即可使用之前的结果合并。

然后考虑左侧向右侧的转移，转移需要左侧部分乘上公共项 $\prod_{j=mid+1}^{l+k-1}F_j(x)$（$x^l$ 可以忽略）。

首先可以发现，这个公共项可以分治时从上往下求出。即设 $H_{l,r}(x)=\prod_{j=r+1}^{l+k-1}F_j(x)$，则：
$$
H_{l,mid}(x)=H_{l,r}(x)*\prod_{i=mid+1}^rF_i(x)\\
H_{mid+1,r}(x)=H_{l,r}(x)*\prod_{i=l}^{mid}F_{i+k}(x)
$$
因此使用预处理的结果可以向下求出每个分治区间的 $H$。同时可以注意到，对于一个区间，只需要保留 $H$ 的前 $(r-l)$ 项即可处理区间内的转移。这样一个分治区间上的转移复杂度只和区间长度有关。

可能需要注意一些细节，防止做 $3n$ 长度的卷积。同时可以跳过不需要算的部分减少常数（例如 $k$ 很大的情况）

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 263001
#define mod 998244353
int n,k,a,b,v[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int rev[N*2],gr[2][N*2];
void init(int d=18)
{
	for(int l=2;l<=1<<d;l<<=1)
	for(int i=0;i<l;i++)
	rev[l+i]=(rev[l+(i>>1)]>>1)|((i&1)*(l>>1));
	for(int t=0;t<2;t++)
	for(int l=2;l<=1<<d;l<<=1)
	{
		int tp=pw(3,(mod-1)/l);
		if(!t)tp=pw(tp,mod-2);
		int v1=1;
		for(int i=0;i<l;i++)gr[t][l+i]=v1,v1=1ll*v1*tp%mod;
	}
}
int ntt[N];
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
	if(a.size()<b.size())swap(a,b);
	for(int i=0;i<b.size();i++)a[i]=(a[i]+b[i])%mod;
	return a;
}
int f[N],g[N];
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	if(s1+s2<=100)
	{
		vector<int> as(s1+s2-1);
		for(int i=0;i<s1;i++)for(int j=0;j<s2;j++)as[i+j]=(as[i+j]+1ll*a[i]*b[j])%mod;
		return as;
	}
	int l=1;while(l<s1+s2)l<<=1;
	for(int i=0;i<l;i++)f[i]=g[i]=0;
	for(int i=0;i<s1;i++)f[i]=a[i];
	for(int i=0;i<s2;i++)g[i]=b[i];
	dft(l,f,1);dft(l,g,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*g[i]%mod;dft(l,f,0);
	vector<int> as;
	for(int i=0;i<s1+s2-1;i++)as.push_back(f[i]);
	return as;
}
vector<int> s0[N*4],s1[N*4];
void solve1(int x,int l,int r)
{
	s0[x].clear();s1[x].clear();
	if(l==r)
	{
		s0[x].push_back((mod+1-v[l])%mod);
		s0[x].push_back(v[l]);
		s1[x].push_back((mod+1-v[l+k])%mod);
		s1[x].push_back(v[l+k]);
		return;
	}
	int mid=(l+r)>>1;
	solve1(x<<1,l,mid);solve1(x<<1|1,mid+1,r);
	s0[x]=polymul(s0[x<<1],s0[x<<1|1]);
	s1[x]=polymul(s1[x<<1],s1[x<<1|1]);
}
int dp[N],sp[N],sv[N];
vector<int> solve2(int x,int l,int r,vector<int> sr)
{
	if(l>n-k+1)return {};
	while(sr.size()>r-l+1)sr.pop_back();
	if(l==r)
	{
		dp[l]=1ll*dp[l]*pw(1+mod-sv[l],mod-2)%mod;
		vector<int> as;
		as.push_back(1ll*dp[l]*(mod+1-v[l])%mod);
		as.push_back(1ll*dp[l]*v[l]%mod);
		return as;
	}
	int mid=(l+r)>>1;
	vector<int> sp=polymul(sr,s0[x<<1|1]);
	while(sp.size()>r-l+1)sp.pop_back();
	vector<int> lv=solve2(x<<1,l,mid,sp),l2=lv;
	while(l2.size()>r-l+1)l2.pop_back();
	vector<int> ls=polymul(l2,sp);
	for(int i=mid+1;i<=r;i++)dp[i]=(dp[i]+ls[i-l])%mod;
	vector<int> sp2=polymul(sr,s1[x<<1]);
	vector<int> rv=solve2(x<<1|1,mid+1,r,sp2);
	vector<int> r2;
	for(int i=1;i<=mid-l+1;i++)r2.push_back(0);
	for(int i=0;i<rv.size();i++)r2.push_back(rv[i]);
	vector<int> as=polyadd(polymul(lv,s0[x<<1|1]),polymul(r2,s1[x<<1]));
	return as;
}
int main()
{
	init();
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d%d",&a,&b),v[i]=1ll*a*pw(b,mod-2)%mod;
	sp[0]=1;for(int i=1;i<=n;i++)sp[i]=1ll*sp[i-1]*(mod+1-v[i])%mod;
	for(int i=1;i<=n;i++)sv[i]=1ll*sp[i+k-1]*pw(sp[i-1],mod-2)%mod;
	dp[1]=1;
	for(int i=1;i+k-1<=n;i+=k)
	{
		solve1(1,i,i+k-1);
		vector<int> sr=solve2(1,i,i+k-1,{1});
		for(int j=k;j<sr.size();j++)dp[i+j]=(dp[i+j]+sr[j])%mod;
	}
	int as=0;
	for(int i=1;i<=n-k+1;i++)as=(as+dp[i])%mod;
	printf("%d\n",as);
}
```



#### SCOI2020模拟8

##### auoj29 endemic

###### Problem

有一个长度为 $n$ 的序列，有 $m$ 种操作，每种操作有三个参数 $l_i,x_i,c_i$，表示可以将任意一个长度为 $l_i$ 的区间整体加 $x_i(x_i\in\{-1,1\})$，一次的费用为 $c_i$。

求一种操作方式使得序列单调不降且总费用最小，输出最小费用或输出无解

$n,m\leq 200$

$1s,512MB$

###### Sol

考虑差分，相当于有若干个数，每次操作可以将一个数减 $1$，另外一个数加 $1$，求最小的费用使得所有数非负（边界的值看成 $+\infty$）。

考虑费用流模型，起点向每一个负数连这个数绝对值流量的边，每个正数向汇点连这个数大小的边。对于每一个操作枚举执行的区间，根据 $x$ 的正负，这次操作相当于从一个端点向另外一个端点连费用为 $c$，没有流量限制的边。

结束条件为所有数非负，因此保证满流情况下的最小费用流即为答案。不能满流则无解。

复杂度 $O(mcmf)$，但是卡不掉

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 207
#define M 80101
int n,m,v[N],dis[N],head[N],cnt=1,is[N],cur[N],su,a,b;
long long as1;
char op[4];
struct edge{int t,next,v,c;}ed[M];
void adde(int f,int t,int v,int c){ed[++cnt]=(edge){t,head[f],v,c};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0,-c};head[t]=cnt;}
bool spfa(int s,int t)
{
	memset(dis,0x3f,sizeof(dis));
	memcpy(cur,head,sizeof(cur));
	queue<int> st;
	dis[s]=0;st.push(s);is[s]=1;
	while(!st.empty())
	{
		int x=st.front();st.pop();is[x]=0;
		for(int i=head[x];i;i=ed[i].next)
		if(ed[i].v&&dis[ed[i].t]>dis[x]+ed[i].c)
		{
			dis[ed[i].t]=dis[x]+ed[i].c;
			if(!is[ed[i].t])st.push(ed[i].t),is[ed[i].t]=1;
		}
	}
	return dis[t]<=1e9;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	is[u]=1;
	int as=0,tp;
	for(int& i=cur[u];i;i=ed[i].next)
	if(!is[ed[i].t]&&dis[ed[i].t]==dis[u]+ed[i].c&&ed[i].v&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		as1+=1ll*tp*ed[i].c;ed[i].v-=tp;ed[i^1].v+=tp;f-=tp;as+=tp;
		if(!f){is[u]=0;return as;}
	}
	is[u]=0;
	return as;
}
void doit(int a,int b,int v,int c){if(a<=0||a>=n||b<0||b>n)return;if(b<=0)b=n+2;if(b>=n)b=n+2;adde(a,b,v,c);}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)if(v[i]>v[i+1])adde(n+1,i,v[i]-v[i+1],0),su+=v[i]-v[i+1];
	else adde(i,n+2,v[i+1]-v[i],0);
	for(int i=1;i<=m;i++)
	{
		scanf("%s%d%d",op,&a,&b);
		if(op[0]=='-')for(int j=0;j<n;j++)doit(j+a,j,1e9,b);
		else for(int j=0;j<n;j++)doit(j,j+a,1e9,b);
	}
	while(spfa(n+1,n+2))
	su-=dfs(n+1,n+2,1e9);
	printf("%lld\n",su?-1:as1);
}
```



##### auoj30 epidemic

###### Problem

给一个长度为 $n$ 的序列，有 $q$ 次如下操作：

1. 将一个区间整体加上一个数
2. 将一个区间整体除以一个正整数，向下取整
3. 询问区间 $\max$
4. 将一个区间还原为初始状态

$n,q\leq 10^5$

$2s,512MB$

###### Sol

考虑分块(其实也可以线段树)

相当于支持整体加，整体除以及快速下放标记

注意到进行若干次操作后的标记一定形如 $x=\lfloor\frac{x+a}{b}\rfloor+c$，其中 $a<b$

如果 $b>10^9$，无论如何前面部分不会超过 $1$，那么只需要考虑 $x+a$ 和 $b$ 的大小关系，即 $b-a$ 的值

然后就可以直接维护标记了

复杂度 $O(q\sqrt n)$

可以发现这个标记显然支持合并，因此也可以直接线段树处理。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105050
#define K 350
int n,m,a,b,c,d,bel[N],v[N],s[N],mx1[K],mx2[K],is[K],g[K][3],st=350,l[N],r[N];//v_i=g_0+(v_i+g_1)/g_2
void pushdown(int x)
{
	if(is[x])for(int i=l[x];i<=r[x];i++)s[i]=v[i],is[x]=0;
	for(int i=l[x];i<=r[x];i++)s[i]=g[x][0]+(s[i]+g[x][1])/g[x][2];
	g[x][0]=g[x][1]=0;g[x][2]=1;
}
void pushup(int x)
{
	mx2[x]=-1e9;
	for(int i=l[x];i<=r[x];i++)if(mx2[x]<s[i])mx2[x]=s[i];
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),s[i]=v[i];
	for(int i=1;i<=n;i++)bel[i]=(i-1)/st+1,r[bel[i]]=i;
	for(int i=n;i>=1;i--)l[bel[i]]=i;
	int ct=(n-1)/st+1;
	for(int i=1;i<=ct;i++)pushup(i),mx1[i]=mx2[i],g[i][2]=1;
	while(m--)
	{
		scanf("%d%d%d%d",&a,&b,&c,&d);b++,c++;
		if(a==0)
		{
			if(bel[b]==bel[c])
			{
				pushdown(bel[b]);
				for(int i=b;i<=c;i++)s[i]+=d;
				pushup(bel[b]);
			}
			else
			{
				pushdown(bel[b]);pushdown(bel[c]);
				for(int i=b;i<=r[bel[b]];i++)s[i]+=d;
				for(int i=l[bel[c]];i<=c;i++)s[i]+=d;
				for(int i=bel[b]+1;i<bel[c];i++)mx2[i]+=d,g[i][0]+=d;
				pushup(bel[b]);pushup(bel[c]);
			}
		}
		else if(a==1)
		{
			if(bel[b]==bel[c])
			{
				pushdown(bel[b]);
				for(int i=b;i<=c;i++)s[i]/=d;
				pushup(bel[b]);
			}
			else
			{
				pushdown(bel[b]);pushdown(bel[c]);
				for(int i=b;i<=r[bel[b]];i++)s[i]/=d;
				for(int i=l[bel[c]];i<=c;i++)s[i]/=d;
				pushup(bel[b]);pushup(bel[c]);
				for(int i=bel[b]+1;i<bel[c];i++)
				{
					mx2[i]/=d;
					long long d1=g[i][0]/d,d2=g[i][1]+1ll*(g[i][0]%d)*g[i][2],d3=1ll*g[i][2]*d;
					if(d3>1.5e9)
					d2=1e9+d2-d3,d3=1e9;
					if(d2<0)d2=0;
					g[i][0]=d1;g[i][1]=d2;g[i][2]=d3;
				}
			}
		}
		else if(a==2)
		{
			int as=-1e9;
			if(bel[b]==bel[c])
			{
				pushdown(bel[b]);
				for(int i=b;i<=c;i++)as=max(as,s[i]);
			}
			else
			{
				pushdown(bel[b]);pushdown(bel[c]);
				for(int i=b;i<=r[bel[b]];i++)as=max(as,s[i]);
				for(int i=l[bel[c]];i<=c;i++)as=max(as,s[i]);
				for(int i=bel[b]+1;i<bel[c];i++)as=max(as,mx2[i]);
			}
			printf("%d\n",as);
		}
		else if(a==3)
		{
			if(bel[b]==bel[c])
			{
				pushdown(bel[b]);
				for(int i=b;i<=c;i++)s[i]=v[i];
				pushup(bel[b]);
			}
			else
			{
				pushdown(bel[b]);pushdown(bel[c]);
				for(int i=b;i<=r[bel[b]];i++)s[i]=v[i];
				for(int i=l[bel[c]];i<=c;i++)s[i]=v[i];
				for(int i=bel[b]+1;i<bel[c];i++)is[i]=1,mx2[i]=mx1[i],g[i][0]=g[i][1]=0,g[i][2]=1;
				pushup(bel[b]);pushup(bel[c]);
			}
		}
	}
}
```



##### auoj31 pandemic

###### Problem

有一棵 $n$ 个点的树，你需要给每条边定向，使得对于所有路径，路径上方向与经过方向相反的边的数量的最大值尽量小，输出使得最大值最小的方案数，模 $10^9+7$

$n\leq 1000$

$1s,512MB$

###### Sol

设直径长度为 $l$ ，答案下界显然是 $s=\lfloor\frac{l+1}2\rfloor$，将直径一边从上到下另外一边从下到上就可以达到这个下界，因此这个为答案。

设 $dp_{i,j,k}$ 表示 $i$ 的子树内，到根的路径最多有 $j$ 条向下的边，最多有 $k$ 条向上的边且子树内合法的方案数

合并两个子树时有 $dp_{x,max(a,c),max(b,d)}+=dp_{y,a,c}*dp_{z,b,d}*[a+d\leq s]*[b+c\leq s]$

枚举两个max分别来自哪一侧，那么对应的另外一个合法的一定是一段前缀，对两边分别做二维前缀和和横纵的一维前缀和即可做到 $O(n^2)$ 转移

总复杂度 $O(n^3)$ ，常数很小（取直径终点为根）可以通过

std做法：

从上往下考虑，对于直径中点，设它的一个子树内到它最多有 $a$ 条向上，$b$ 条向下。则 $a+b\geq$ 子树内最大深度，且对于两个子树，有 $a+b'\leq l,b+a'\leq l$。

因此对于直径中点的两个直径所在的子树，它们一定满足 $a+b=$ 最大深度，且确定了一个子树内的 $a$ 后，可以发现它会要求其它子树的 $a',b'$ 满足 $a'\leq a,b'\leq b$。

此时可以设 $dp_{x,a,b}$ 表示需要满足这个限制时子树内的方案数，从上往下转移。可以发现每次向下时，$a+b$ 正好会减少 $1$，且初始需要的状态 $a+b=l$，因此需要的状态只有 $O(n^2)$ 个，复杂度 $O(n^2)$。因为直径所在的两个子树内合法方案一定满足 $a+b\geq l$，因此枚举一个部分内的 $a,b$ 不会算重。

对于直径中点是边的情况，可以枚举中间边的情况，再类似地枚举两侧情况。

复杂度 $O(n^2)$，代码咕了。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1005
#define mod 1000000007
int n,head[N],cnt,dep[N],as,a,b,sz[N],vl,as2,id[N],ct,is[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){dep[u]=dep[fa]+1;int mx=0;sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),mx=max(mx,sz[ed[i].t]),sz[u]+=sz[ed[i].t];mx=max(mx,n-sz[u]);if(mx<vl)as2=u,vl=mx;}
int getid(){for(int i=1;i<=ct;i++)if(!is[i]){is[i]=1;return i;}is[++ct]=1;return ct;}
struct brute3{
	int dp[304][501][501],mx[N],s1,s2,v1[501][501],v2[501][501],s3,v3[501][501];
	int k1[501][501],k2[501][501],k3[501][501],k4[501][501],k5[501][501],k6[501][501];
	void merge(int u)
	{
		s3=max(s1,s2);
		if(s3>as)s3=as;
		for(int i=0;i<=s1;i++)
		for(int j=0;j<=s1;j++)
		k1[i][j]=k2[i][j]=k3[i][j]=v1[i][j];
		for(int i=0;i<=s2;i++)
		for(int j=0;j<=s2;j++)
		k4[i][j]=k5[i][j]=k6[i][j]=v2[i][j];
		for(int i=0;i<=s1;i++)
		for(int j=0;j<=s1;j++)
		k1[i][j]=(4ll*mod+k1[i][j]+(i?k1[i-1][j]:0)+(j?k1[i][j-1]:0)-(i&&j?k1[i-1][j-1]:0))%mod;
		for(int i=0;i<=s1;i++)
		for(int j=1;j<=s1;j++)
		k2[i][j]=(k2[i][j]+k2[i][j-1])%mod;
		for(int i=1;i<=s1;i++)
		for(int j=0;j<=s1;j++)
		k3[i][j]=(k3[i][j]+k3[i-1][j])%mod;
		for(int i=0;i<=s2;i++)
		for(int j=0;j<=s2;j++)
		k4[i][j]=(4ll*mod+k4[i][j]+(i?k4[i-1][j]:0)+(j?k4[i][j-1]:0)-(i&&j?k4[i-1][j-1]:0))%mod;
		for(int i=0;i<=s2;i++)
		for(int j=1;j<=s2;j++)
		k5[i][j]=(k5[i][j]+k5[i][j-1])%mod;
		for(int i=1;i<=s2;i++)
		for(int j=0;j<=s2;j++)
		k6[i][j]=(k6[i][j]+k6[i-1][j])%mod;
		for(int i=0;i<=s3;i++)
		for(int j=0;j<=s3;j++)
		{
			int d1=min(i-1,as-j),d2=min(j-1,as-i);
			if(d1>s2)d1=s2;if(d2>s2)d2=s2;
			if(d1>=0&&d2>=0&&i<=s1&&j<=s1)v3[i][j]=(v3[i][j]+1ll*v1[i][j]*k4[d1][d2])%mod;
			d1=min(i,as-j),d2=min(j,as-i);
			if(d1>s1)d1=s1;if(d2>s1)d2=s1;
			if(i<=s2&&j<=s2)v3[i][j]=(v3[i][j]+1ll*v2[i][j]*k1[d1][d2])%mod;
			if(i+j<=as)
			{
				int d3=i-1,d4=j;
				if(d3>s2)d3=s2;if(d4>s1)d4=s1;
				if(d3>=0&&d4>=0&&i<=s1&&j<=s2)v3[i][j]=(v3[i][j]+1ll*k2[i][d4]*k6[d3][j])%mod;
				d3=i,d4=j-1;
				if(d3>s1)d3=s1;if(d4>s2)d4=s2;
				if(d3>=0&&d4>=0&&i<=s2&&j<=s1)v3[i][j]=(v3[i][j]+1ll*k5[i][d4]*k3[d3][j])%mod;
			}
		}
		mx[u]=s3;
		for(int i=0;i<=s3;i++)
		for(int j=0;j<=s3;j++)
		dp[id[u]][i][j]=v3[i][j],v3[i][j]=0;
	}
	void dfs(int u,int fa)
	{
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
		{
			int fg=0;
			dfs(ed[i].t,u);
			s1=mx[u];
			if(id[u])
			for(int j=0;j<=s1;j++)
			for(int k=0;k<=s1;k++)
			v1[j][k]=dp[id[u]][j][k];
			else v1[0][0]=1,id[u]=id[ed[i].t],fg=1;
			s2=mx[ed[i].t]+1;
			for(int j=0;j<=s2;j++)
			for(int k=0;k<=s2;k++)
			v2[j][k]=0;
			for(int j=0;j<s2;j++)
			for(int k=0;k<s2;k++)
			v2[j+1][k]=(v2[j+1][k]+dp[id[ed[i].t]][j][k])%mod,
			v2[j][k+1]=(v2[j][k+1]+dp[id[ed[i].t]][j][k])%mod;
			merge(u);
			if(!fg)is[id[ed[i].t]]=0;
		}
		if(!id[u])id[u]=getid(),dp[id[u]][0][0]=1;
	}
}br2;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	vl=n+1;dfs(1,0);
	for(int i=1;i<=n;i++)if(dep[i]>dep[as])as=i;
	dfs(as,0);
	as=0;
	for(int i=1;i<=n;i++)if(dep[i]>as)as=dep[i];
	as=as/2;
	br2.dfs(as2,0);
	int as3=0;
	for(int i=0;i<=br2.mx[as2];i++)
	for(int j=0;j<=br2.mx[as2];j++)
	as3=(as3+br2.dp[id[as2]][i][j])%mod;
	printf("%d\n",as3);
}
```



#### SCOI2020模拟7

##### auoj32 高精度

###### Problem

给一个长度为 $n$ 的字符串 $S$。有一个数，初始数为 $0$，依次考虑字符串的每一位：

如果这一位是数字，那么将这个数乘 $10$ 并加上这个数字。否则，将这个数除以 $10$，结果下取整。

求出每次操作后的当前数之和，模 $998244353$

$|S|\leq 5\times 10^5$

$3s,1024MB$

###### Sol

考虑求出在不进位的情况下，每一位被加的数值。可以无视下取整这个条件（即可以有小数），只需要在最后相加算答案时规定小数部分不能进位。

此时在将最后一位变成某个数的时候，可能之前这个位置上有数，这时可以看成在这个位置上加一个数，设第 $j$ 次加的数为 $b_j$。

设 $i$ 次操作后数的位数为 $l_i$，最低位为第 $0$ 位，那么可以发现第 $i$ 次操作加的数在第 $j$ 次操作后在 $l_j-l_i$ 位。

设 $v_i$ 为最后第 $i$ 位被加了多少，那么有 $v_i=\sum_{j\leq k,l_k-l_j=i}b_j$

因为相邻两个 $l$ 之间的差不超过 $1$，这个过程可以使用分治+`fft` 计算。

复杂度 $O(|S|\log^2|S|)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 530000
#define mod 998244353
int n,v[N],a[N],b[N],su[N],ntt[N],rev[N],as[N],g[2][N*2];
char s[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int tp=pw(3,(mod-1)/i);
		for(int j=0;j<s;j+=i)
		for(int k=j,vl=0;k<j+(i>>1);k++,vl++)
		{
			int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][i+vl]%mod;
			ntt[k]=(v1+v2)%mod;
			ntt[k+(i>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=pw(s,t==0?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void cdq(int l,int r)
{
	if(l==r){as[0]=(as[0]+v[l])%mod;return;}
	int mid=(l+r)>>1;
	cdq(l,mid);cdq(mid+1,r);
	int s=1;while(s<=r-l+2)s<<=1;
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((s>>1)*(i&1));
	for(int i=0;i<s;i++)a[i]=b[i]=0;
	int v1=n,v2=n;
	for(int i=l;i<=mid;i++)v1=min(v1,su[i]);
	for(int i=mid+1;i<=r;i++)v2=min(v2,su[i]);
	for(int i=l;i<=mid;i++)a[mid-l+1-(su[i]-v1)]+=v[i];
	for(int i=mid+1;i<=r;i++)b[su[i]-v2]++;
	dft(s,a,1);dft(s,b,1);for(int i=0;i<s;i++)a[i]=1ll*a[i]*b[i]%mod;dft(s,a,0);
	int f1=mid-l+1+v1-v2,f2=-f1;
	if(f1<0)f1=0;
	for(int i=f1;i<s;i++)as[i+f2]=(as[i+f2]+a[i])%mod;
}
int main()
{
	scanf("%d%s",&n,s+1);
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<19;j<<=1)
	{
		int tp=pw(3,(mod-1)/j),v2=1;
		if(i==0)tp=pw(tp,mod-2);
		for(int l=0;l<j>>1;l++)g[i][j+l]=v2,v2=1ll*v2*tp%mod;
	}
	for(int i=1;i<=n;i++)
	if(s[i]=='-')su[i]=su[i-1]-1;
	else
	{
		su[i]=su[i-1]+1;
		v[i]=s[i]-'0'-a[su[i]];
		a[su[i]]=s[i]-'0';
	}
	cdq(1,n);
	for(int i=1;i<=n+1;i++)as[i]=(as[i]+as[i-1]/10)%mod,as[i-1]%=10;
	int fg=0;
	for(int i=n+1;i>=0;i--)if(as[i]||fg)printf("%d",as[i]),fg=1;
	if(!fg)printf("0");
}
```



##### auoj33 最短路

###### Problem

有一张 $n$ 个点的图，一开始有 $m$ 条白边。考虑加入一些黑边，对于每一对在原图中最短路为 $2$ 的点，在它们之间加入一条黑边。

给出经过一条白边的时间 $a$ 和经过一条黑边的时间 $b$ ，求 $1$ 到每个点的最短路

$n\leq 10^5,m\leq 3\times 10^5$

$1s,1024MB$

###### Sol

考虑只有白边的一条最短路，显然除了路径上相邻的两个点剩下的点对间不可能有连边，因此最短路上两个不相邻的点一定存在一条黑边。

设最短路长度为 $d$，那么可以直接在最短路上选择一些黑边，一定存在经过 $d$ 条白边的方案和经过 $\lfloor\frac d 2\rfloor$ 条黑边，$d\bmod 2$ 条白边的方案

对于一条经过了 $a$ 条白边和 $b$ 条黑边的路径，显然有 $a+2b\geq d$。因为存在上面两种路径，因此如果 $a>0$，显然这条路径不可能比上面的两条都更优秀。

因此只需要额外考虑只经过黑边的路径，即求出只经过黑边的最短路。

考虑 `bfs`，对于每个点，暴力枚举它的出边，再暴力枚举出边那个点的出边判断这两个点是否连边

如果某条出边被访问到了，那之后遍历到这个点时就不用再走这条出边了，这部分每条边只会遍历一次。否则如果枚举到了出边但不能走，则这三个点一定构成了一个三元环。

对于每个点记录它还连向哪些没有被访问的点，因为三元环只有 $O(m\sqrt m)$ 个，因此最多会有 $O(m\sqrt m)$ 次枚举某条出边但不删除，因此总复杂度 $O(m\sqrt m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
using namespace std;
#define N 151010
struct edge{int t,next;}ed[N*4];
long long as[N];
int n,m,a,b,x,y,head[N],cnt,is[N],dis[N];
vector<int> nt[N];
vector<int> fu;
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;nt[f].push_back(t);nt[t].push_back(f);}
int main()
{
	scanf("%d%d%d%d",&n,&m,&x,&y);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b);
	queue<int> st;st.push(1);
	for(int i=2;i<=n;i++)dis[i]=-1;
	while(!st.empty())
	{
		int v=st.front();st.pop();
		for(int i=head[v];i;i=ed[i].next)if(dis[ed[i].t]==-1)dis[ed[i].t]=dis[v]+1,st.push(ed[i].t);
	}
	for(int i=2;i<=n;i++)as[i]=min(1ll*x*dis[i],1ll*y*(dis[i]/2)+x*(dis[i]&1));
	st.push(1);
	for(int i=2;i<=n;i++)dis[i]=-1;
	while(!st.empty())
	{
		int v=st.front();st.pop();
		for(int i=head[v];i;i=ed[i].next)is[ed[i].t]=1;
		for(int i=head[v];i;i=ed[i].next)
		{
			int sz=nt[ed[i].t].size();
			for(int j=0;j<sz;j++)
			{
				int vl=nt[ed[i].t][j];
				if(dis[vl]!=-1)continue;
				if(vl==v||is[vl])fu.push_back(vl);
				else if(dis[vl]==-1)dis[vl]=dis[v]+1,st.push(vl);
			}
			nt[ed[i].t].clear();
			for(int j=0;j<fu.size();j++)nt[ed[i].t].push_back(fu[j]);
			fu.clear();
		}
		for(int i=head[v];i;i=ed[i].next)is[ed[i].t]=0;
	}
	for(int i=2;i<=n;i++)as[i]=min(as[i],dis[i]>=0?1ll*dis[i]*y:1000000000000000000ll),printf("%lld\n",as[i]);
}
```



##### auoj34 网格图

###### Problem

有一个 $n\times m$ 的网格图，一开始有 $k$ 个格子是黑的。你可以进行任意次操作，每次可以选择一个白格子，如果这个格子周围四个格子中有至少两个是黑的，那么将这个格子染黑，求最后最多有多少个黑格子。

$n,m,k\leq 5\times 10^5$

$3s,512MB$

###### Sol

非矩形连通块一定可以操作为矩形，因此最后的黑色部分一定由若干个矩形组成

如果两个矩形间存在两个点曼哈顿距离不超过 $2$，那么可以将中间的格子染黑，进而将两个矩形合并起来。因此一种最优的操作方式是依次合并矩形，直到不能合并为止

将矩形按照上边界排序，两个矩形 $(x_1,y_1,x_2,y_2),(x_3,y_3,x_4,y_4)(y_2\leq y_4)$ 可以合并的条件是存在两个点距离不超过 $2$，即以下三种条件之一：

1. $[x_1,x_2]\cap[x_3,x_4]\neq\emptyset\and y_2+2\geq y_3$
2. $[x_1-1,x_2+1]\cap[x_3,x_4]\neq\emptyset\and y_2+1\geq y_3$
3. $[x_1-2,x_2+2]\cap[x_3,x_4]\neq\emptyset\and y_2\geq y_3$

这相当于 $x$ 坐标的区间上插入/删除一个矩形，询问与一个 $x$ 区间有交的矩形中最大的 $y_2$ 对应的矩形

因为加入的矩形一定是 $y$ 最大的，所以可以用一个栈或 `vector` 维护所以加入的矩形的 $y$

考虑线段树，线段树上每个点维护两个 `vector`，分别表示完全覆盖这个区间的矩形和与这个区间有交的矩形。区间询问时，在经过的区间上考虑覆盖这个区间的矩形，在覆盖的区间上考虑有交的矩形即可考虑所有相交情况。（类似于标记永久化）

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 505000
int n,m,k,is[N],fa[N],s[N][4],v[N],id[N*4],ct;
vector<int> v1[N*2],v2[N*2];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
bool cmp(int a,int b){return s[a][3]<s[b][3];}
void build(int x,int l,int r){id[x]=++ct;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
void modify(int x,int l,int r,int l1,int r1,int v){if(l1==l&&r1==r){v2[id[x]].push_back(v);return;}v1[id[x]].push_back(v);int mid=(l1+r1)>>1;if(mid>=r)modify(x<<1,l,r,l1,mid,v);else if(mid<l)modify(x<<1|1,l,r,mid+1,r1,v);else modify(x<<1,l,mid,l1,mid,v),modify(x<<1|1,mid+1,r,mid+1,r1,v);}
int query(int x,int l,int r,int l1,int r1)
{
	if(l<1)l=1;if(r>n)r=n;
	int as=0;
	while(v2[id[x]].size()&&is[v2[id[x]].back()])v2[id[x]].pop_back();
	if(v2[id[x]].size())as=v2[id[x]].back();
	if(l1==l&&r1==r)
	{
		int as2=0;
		while(v1[id[x]].size()&&is[v1[id[x]].back()])v1[id[x]].pop_back();
		if(v1[id[x]].size())as2=v1[id[x]].back();
		if(s[as][3]<s[as2][3])as=as2;
		return as;
	}
	int mid=(l1+r1)>>1,as2;
	if(mid>=r)
	{
		as2=query(x<<1,l,r,l1,mid);if(s[as][3]<s[as2][3])as=as2;
	}
	else if(mid<l)
	{
		as2=query(x<<1|1,l,r,mid+1,r1);if(s[as][3]<s[as2][3])as=as2;
	}
	else
	{
		as2=query(x<<1,l,mid,l1,mid);if(s[as][3]<s[as2][3])as=as2;
		as2=query(x<<1|1,mid+1,r,mid+1,r1);if(s[as][3]<s[as2][3])as=as2;
	}
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=k;i++)scanf("%d%d",&s[i][0],&s[i][1]),s[i][3]=s[i][1],s[i][2]=s[i][0],fa[i]=v[i]=i;
	build(1,1,n);sort(v+1,v+k+1,cmp);
	for(int i=1;i<=k;i++)
	{
		while(1)
		{
			int tp=0;
			int v1=query(1,s[v[i]][0]-2,s[v[i]][2]+2,1,n);
			if(v1&&s[v1][3]>=s[v[i]][1])tp=v1;
			else
			{
				int v1=query(1,s[v[i]][0]-1,s[v[i]][2]+1,1,n);
				if(v1&&s[v1][3]>=s[v[i]][1]-1)tp=v1;
				else
				{
					int v1=query(1,s[v[i]][0],s[v[i]][2],1,n);
					if(v1&&s[v1][3]>=s[v[i]][1]-2)tp=v1;
				}
			}
			if(!tp)break;
			is[tp]=1;fa[tp]=v[i];
			s[v[i]][0]=min(s[v[i]][0],s[tp][0]);
			s[v[i]][1]=min(s[v[i]][1],s[tp][1]);
			s[v[i]][2]=max(s[v[i]][2],s[tp][2]);
			s[v[i]][3]=max(s[v[i]][3],s[tp][3]);
		}
		modify(1,s[v[i]][0],s[v[i]][2],1,n,v[i]);
	}
	long long as=0;
	for(int i=1;i<=k;i++)if(fa[i]==i)as+=1ll*(s[i][2]-s[i][0]+1)*(s[i][3]-s[i][1]+1);
	printf("%lld\n",as);
}
```



#### SCOI2020模拟?

##### auoj35 小W数排列

###### Problem

给 $n$ 个互不相同的整数 $A_{1,...,n}$，求有多少个 $n$ 阶排列满足 $\sum_{i=1}^{n-1} |A_{p_i}-A_{p_{i+1}}|\leq k$，答案模 $10^9+7$

$n\leq 100,k\leq 1000$

$1s,512MB$

###### Sol

考虑怎么算不等式左边。一种方式是直接拆绝对值，但拆完有负数，这会导致状态数很大，不能通过。

另外一种拆法是，对于排序后相邻的每一对 $(A_i,A_{i+1})$ ，计算排列中相邻两个元素一个小于等于 $A_i$ 另外一个大于等于 $A_{i+1}$ 的对数，然后加上对数乘 $A_{i+1}-A_i$ 的贡献。这样贡献只有加法。

如果小于等于 $A_i$ 的构成了 $k$ 段，那么这样的对数为 $2k-[A_{p_1}<A_i]-[A_{p_n}<A_i]$。

设 $dp_{i,j,k,0/1,0/1}$ 表示考虑到了 $A_i$，前面的数构成了 $j$ 段，前面的贡献加起来是 $k$，开头结尾是否小于等于 $A_i$，这种情况的方案数。

转移有五种情况：

1. 在中间且合并两个段
2. 在中间且与某个段相邻
3. 在中间新增一个段
4. 在开头或结尾，且与某个段相邻
5. 在开头和结尾，且新增一个段

每种情况都可以 $O(1)$ 计算转移系数，复杂度 $O(n^2k)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105
#define M 1050
#define mod 1000000007
int dp[N][N][M][2][2],n,k,v[N],as;
int main()
{
	scanf("%d%d",&n,&k);
	if(n==1){printf("1\n");return 0;}
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	dp[1][1][0][0][0]=dp[1][1][0][0][1]=dp[1][1][0][1][0]=1;
	for(int i=2;i<=n;i++)
	for(int j=0;j<=i;j++)
	for(int l=0;l<=k;l++)
	for(int s=0;s<=1;s++)
	for(int t=0;t<=1;t++)
	if(dp[i-1][j][l][s][t])
	{
		int vl=l+(j*2-s-t)*(v[i]-v[i-1]);
		if(vl>k)continue;
		dp[i][j][vl][s][t]=(dp[i][j][vl][s][t]+1ll*dp[i-1][j][l][s][t]*(j*2-s-t))%mod;
		dp[i][j+1][vl][s][t]=(dp[i][j+1][vl][s][t]+1ll*dp[i-1][j][l][s][t]*(j+1-s-t))%mod;
		dp[i][j-1][vl][s][t]=(dp[i][j-1][vl][s][t]+1ll*dp[i-1][j][l][s][t]*(j-1))%mod;
		if(!s)dp[i][j+1][vl][1][t]=(dp[i][j+1][vl][1][t]+1ll*dp[i-1][j][l][s][t])%mod;
		if(!t)dp[i][j+1][vl][s][1]=(dp[i][j+1][vl][s][1]+1ll*dp[i-1][j][l][s][t])%mod;
		if(!s)dp[i][j][vl][1][t]=(dp[i][j][vl][1][t]+1ll*dp[i-1][j][l][s][t])%mod;
		if(!t)dp[i][j][vl][s][1]=(dp[i][j][vl][s][1]+1ll*dp[i-1][j][l][s][t])%mod;
	}
	for(int i=0;i<=k;i++)as=(as+dp[n][1][i][1][1])%mod;
	printf("%d\n",as);
}
```



##### auoj36 小W玩游戏

###### Problem

给一个 $n\times m$ 的网格，进行 $d$ 次操作，每次选择一个位置，将这一行和这一列的数全部加 $1$。

求使得最后网格中奇数个数不超过 $k$ 的操作方案数，答案模 $998244353$

$n,m\leq 2\times 10^5$

$1s,512MB$

###### Sol

行列的操作独立，因此行列显然可以分开。

考虑行的情况，显然只关心最后每行被操作了奇数次或者偶数次。相当于要求出有 $i$ 行被选了奇数次的方案数。

考虑二项式反演，设 $f_i$ 表示恰好有 $i$ 行为奇数的方案数，$g_i$ 表示对于所有方案，选出 $i$ 个奇数行的方案数总和，那么有：

$$
g_i=\sum_{j=i}^nC_j^if_j
$$

二项式反演可得：

$$
f_i=\sum_{j=i}^n(-1)^{j-i}C_j^ig_j
$$


考虑求 $g$，根据所有奇数的 `egf` 有：
$$
g_i =C_n^i[x^d](\frac{e^x-e^{-x}}2)^i(e^x)^{n-i}*d!\\
=C_n^id!\frac{1}{2^i}[x^d]\sum_{j=0}^iC_i^je^{n-2j}\\
=C_n^i\frac{1}{2^i}\sum_{j=0}^iC_i^j(n-2j)^d
$$
NTT即可求出 $g$。

可以发现，如果有 $i$ 行 $j$ 列操作了奇数次，则 $1$ 的个数为 $i(m-j)+j(n-i)$。枚举有多少行选了奇数次，那么合法的列选奇数次个数一定是一段前缀/后缀，二分+前缀和即可。

复杂度 $O(n\log n+m\log m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 530000
#define mod 998244353
int n,m,f[N],g[N],ntt[N],fr[N],ifr[N],rev[N],a[N],b[N],c[N],as;
long long p,q;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int l,int *a,int t)
{
	for(int i=0;i<l;i++)rev[i]=(rev[i>>1]>>1)|((l>>1)*(i&1)),ntt[rev[i]]=a[i];
	for(int s=2;s<=l;s<<=1)
	{
		int v1=pw(3,(mod-1)/s);
		if(t==-1)v1=pw(v1,mod-2);
		for(int i=0;i<l;i+=s)
		for(int j=i,st1=1;j<i+(s>>1);j++,st1=1ll*st1*v1%mod)
		{
			int v1=ntt[j],v2=1ll*st1*ntt[j+(s>>1)]%mod;
			ntt[j]=(v1+v2)%mod;ntt[j+(s>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=1ll*pw(l,t==-1?mod-2:0);
	for(int i=0;i<l;i++)a[i]=1ll*inv*ntt[i]%mod;
}
void solve(int n,int *as)
{
	int l=1;while(l<=n*2+2)l<<=1;for(int i=0;i<=l;i++)a[i]=b[i]=0;
	for(int i=0;i<=n;i++)a[i]=1ll*pw(mod+n-2*i,p)*ifr[i]%mod,b[i]=ifr[i];
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,-1);
	l=1;while(l<=n+200000)l<<=1;for(int i=0;i<=l;i++)c[i]=0;
	for(int i=n+1;i<=l;i++)a[i]=0;
	for(int i=0;i<=n;i++)a[i]=1ll*a[i]*fr[n]%mod*ifr[n-i]%mod*pw((mod+1)/2,i)%mod;
	for(int i=0;i<=n;i++)a[i]=1ll*a[i]*fr[i]%mod;
	for(int i=0;i<=200000;i++)c[i]=1ll*ifr[200000-i]*((i&1)?mod-1:1)%mod;
	dft(l,a,1);dft(l,c,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*c[i]%mod;dft(l,a,-1);
	for(int i=0;i<=n;i++)as[n-i]=1ll*a[i+200000]*ifr[i]%mod;
}
int main()
{
	scanf("%d%d%lld%lld",&n,&m,&p,&q);p=(p-1)%(mod-1)+1;
	fr[0]=ifr[0]=1;for(int i=1;i<=200000;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	solve(n,f);solve(m,g);
	for(int j=1;j<=m;j++)g[j]=(g[j]+g[j-1])%mod;
	for(int i=0;i<=n;i++)
	{
		long long v1=q-1ll*i*m,v2=n-2*i;
		if(v2>0)
		{
			if(v1<0)continue;
			long long v3=v1/v2;
			if(v3>m)v3=m;
			as=(as+1ll*f[i]*g[v3])%mod;
		}
		else if(v2==0){if(v1>=0)as=(as+1ll*f[i]*g[m])%mod;}
		else
		{
			if(v1>=0){as=(as+1ll*f[i]*g[m])%mod;continue;}
			v1*=-1,v2*=-1;
			long long v3=(v1-1)/v2;
			if(v3>m)continue;
			as=(as+1ll*f[i]*(g[m]-g[v3]+mod))%mod;
		}
	}
	printf("%d\n",as);
}
```



##### auoj37 小W维护序列

###### Problem

给一个长度为 $n$ 的序列，有 $q$ 次操作，每次操作为以下几种之一：

1. 给一个区间，求区间内元素去重后，选出三个不同的数的所有方案的三数乘积之和，模 $10^9+7$
2. 修改一个元素
3. 删除一个元素
4. 在某个元素后加入一个元素
5. 给一个区间，求区间内元素的种数

$n,q\leq 10^5$

$2s,128MB$

###### Sol

维护种类数的经典做法是维护 $pre$，但标号会随着插入变化。考虑求出每个数的绝对位置，即给所有操作过程中插入过的数一个顺序。

维护两个平衡树，第一个平衡树上进行所有操作，插入在一个元素后时。时在第二个平衡树上对应元素后插入，第二个平衡树上不删除元素。

可以发现，这样最后第二个平衡树上的顺序就是一个合法的绝对位置关系。

对于每个数记录这个数上次出现的位置 $pre_i$，则固定了位置后一次修改只会改 $O(1)$ 个 $pre_i$。可以发现，每一次询问都相当于一个 $i,pre_i$ 上的二维数点。

对于求选出三个数乘积和的询问，记录一个区间内选 $0\sim 3$ 个的所有方案的乘积的和，就可以快速合并两个区间的信息。

因为信息满足可加性，因此可以 `cdq` 分治，分治中维护询问的答案。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<vector>
#include<map>
#include<algorithm>
using namespace std;
#define N 205000
#define mod 1000000007
set<int> st[N];
map<int,int> fuc;
int n,q,ct,s[N][3],v[N],id[N],id2[N],v2[N];
struct sth{int a,b,c,d;}tr[N],as[N][2],tr2[N];
sth doit(sth a,sth b){sth c;c.a=a.a+b.a;c.b=(a.b+b.b)%mod;c.c=(a.c+b.c+1ll*a.b*b.b)%mod;c.d=(a.d+b.d+1ll*a.c*b.b+1ll*a.b*b.c)%mod;return c;}
sth inv(sth a){sth b;b.a=-a.a;b.b=mod-a.b;b.c=(2*mod-a.c-1ll*a.b*b.b%mod)%mod;b.d=(3ll*mod-a.d-1ll*a.b*b.c%mod-1ll*a.c*b.b%mod)%mod;return b;}
struct que{int x,y,l,r,id;};
struct pts{int x,y,z,fu;sth st;};
vector<pts> s0[21][2];
vector<que> s1[21][2];
bool cmp1(pts a,pts b){return a.y<b.y;}
bool cmp2(que a,que b){return a.y<b.y;}
struct Splay{
	int ch[N][2],fa[N],sz[N],rt;
	void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1;}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
	void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(g!=y)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);rotate(x);}if(!y)rt=x;}
	int kth(int x,int k){int tp=sz[ch[x][0]];if(k==tp+1)return x;if(k<=tp)return kth(ch[x][0],k);return kth(ch[x][1],k-tp-1);}
}tr1,tr3;
void check(){for(int i=1;i<=ct;i++){int st=tr3.kth(tr3.rt,i+1);tr3.splay(st);id[st]=i;}}
void just_doit()
{
	ct=n+2;tr1.rt=tr3.rt=1;
	for(int i=1;i<n+2;i++)tr1.sz[i]=tr3.sz[i]=n-i,tr1.ch[i][1]=tr3.ch[i][1]=i+1,tr1.fa[i+1]=tr3.fa[i+1]=i;
	tr1.sz[n+2]=tr3.sz[n+2]=1;
	for(int i=1;i<=q;i++)
	if(s[i][0]==4)
	{
		tr1.splay(tr1.kth(tr1.rt,s[i][1]+1));
		tr1.splay(tr1.kth(tr1.rt,s[i][1]+2),tr1.rt);
		int v1=tr1.ch[tr1.rt][1],v2=tr1.rt;
		tr1.ch[v1][0]=++ct;tr1.sz[ct]=1;tr1.fa[ct]=v1;tr1.splay(ct);
		tr3.splay(v2);
		tr3.splay(tr3.kth(tr3.rt,tr3.sz[tr3.ch[tr3.rt][0]]+2),tr3.rt);
		v1=tr3.ch[tr3.rt][1],v2=tr3.rt;
		tr3.ch[v1][0]=ct;tr3.sz[ct]=1;tr3.fa[ct]=v1;tr3.splay(ct);
		s[i][1]=ct;
	}
	else if(s[i][0]==3)
	{
		tr1.splay(tr1.kth(tr1.rt,s[i][1]));
		tr1.splay(tr1.kth(tr1.rt,s[i][1]+2),tr1.rt);
		int v1=tr1.ch[tr1.rt][1],v3=tr1.ch[v1][0];
		s[i][1]=v3;
		tr1.ch[v1][0]=0;tr1.fa[v3]=0;tr1.pushup(v1);tr1.splay(v1);
	}
	else if(s[i][0]==1||s[i][0]==5)
	{
		int v1=tr1.kth(tr1.rt,s[i][1]+1);
		s[i][1]=v1;tr1.splay(v1);
		v1=tr1.kth(tr1.rt,s[i][2]+1);
		s[i][2]=v1;tr1.splay(v1);
	}
	else
	{
		int v1=tr1.kth(tr1.rt,s[i][1]+1);
		s[i][1]=v1;tr1.splay(v1);
	}
	check();
	int ct3=0;
	for(int i=1;i<=n;i++)if(!fuc[v2[i]])fuc[v2[i]]=++ct3;
	for(int i=1;i<=q;i++)if(s[i][0]==2||s[i][0]==4)if(!fuc[s[i][2]])
	fuc[s[i][2]]=++ct3;
	ct-=2;
	for(int i=1;i<=ct3;i++)st[i].insert(0);
	for(int i=1;i<=q;i++)
	{
		s[i][1]=id[s[i][1]];
		if(s[i][0]!=2&&s[i][0]!=4&&s[i][0]!=3)s[i][2]=id[s[i][2]];
	}
	for(int i=1;i<=n;i++)s0[0][0].push_back((pts){0,*(--st[fuc[v2[i]]].lower_bound(id[i+1])),id[i+1],1,(sth){1,v2[i],0,0}}),v[id[i+1]]=v2[i],st[fuc[v2[i]]].insert(id[i+1]);
	for(int i=1;i<=q;i++)
	if(s[i][0]==1||s[i][0]==5)s1[0][0].push_back((que){i,s[i][1]-1,s[i][1],s[i][2],i});
	else if(s[i][0]==4)
	{
		set<int>::iterator it=st[fuc[s[i][2]]].lower_bound(s[i][1]);
		if(it!=st[fuc[s[i][2]]].end())
		{
			int v1=*it,v2=*(--it);
			s0[0][0].push_back((pts){i,v2,v1,0,(sth){1,v[v1],0,0}});
			s0[0][0].push_back((pts){i,s[i][1],v1,1,(sth){1,v[v1],0,0}});
		}
		s0[0][0].push_back((pts){i,*(--st[fuc[s[i][2]]].lower_bound(s[i][1])),s[i][1],1,(sth){1,s[i][2],0,0}});
		st[fuc[s[i][2]]].insert(s[i][1]);v[s[i][1]]=s[i][2];
	}
	else if(s[i][0]==3)
	{
		set<int>::iterator it=st[fuc[v[s[i][1]]]].lower_bound(s[i][1]+1);
		if(it!=st[fuc[v[s[i][1]]]].end())
		{
			int v1=*it;it--;it--;int v2=*it;
			s0[0][0].push_back((pts){i,s[i][1],v1,0,(sth){1,v[v1],0,0}});
			s0[0][0].push_back((pts){i,v2,v1,1,(sth){1,v[v1],0,0}});
		}
		s0[0][0].push_back((pts){i,*(--st[fuc[v[s[i][1]]]].lower_bound(s[i][1])),s[i][1],0,(sth){1,v[s[i][1]],0,0}});
		st[fuc[v[s[i][1]]]].erase(s[i][1]);
	}
	else
	{
		set<int>::iterator it=st[fuc[s[i][2]]].lower_bound(s[i][1]);
		if(it!=st[fuc[s[i][2]]].end())
		{
			int v1=*it,v2=*(--it);
			s0[0][0].push_back((pts){i,v2,v1,0,(sth){1,v[v1],0,0}});
			s0[0][0].push_back((pts){i,s[i][1],v1,1,(sth){1,v[v1],0,0}});
		}
		it=st[fuc[v[s[i][1]]]].lower_bound(s[i][1]+1);
		if(it!=st[fuc[v[s[i][1]]]].end())
		{
			int v1=*it;it--;it--;int v2=*it;
			s0[0][0].push_back((pts){i,s[i][1],v1,0,(sth){1,v[v1],0,0}});
			s0[0][0].push_back((pts){i,v2,v1,1,(sth){1,v[v1],0,0}});
		}
		s0[0][0].push_back((pts){i,*(--st[fuc[v[s[i][1]]]].lower_bound(s[i][1])),s[i][1],0,(sth){1,v[s[i][1]],0,0}});
		st[fuc[v[s[i][1]]]].erase(s[i][1]);
		s0[0][0].push_back((pts){i,*(--st[fuc[s[i][2]]].lower_bound(s[i][1])),s[i][1],1,(sth){1,s[i][2],0,0}});
		st[fuc[s[i][2]]].insert(s[i][1]);v[s[i][1]]=s[i][2];
	}
	
}
void add1(int x,sth k){for(int i=x+1;i<=ct+1;i+=i&-i)tr[i]=doit(tr[i],k);}
sth que1(int x){sth fu=(sth){0,0,0,0};for(int i=x+1;i;i-=i&-i)fu=doit(fu,tr[i]);return fu;}
void add2(int x,sth k){for(int i=x+1;i<=ct+1;i+=i&-i)tr2[i]=doit(tr2[i],k);}
sth que2(int x){sth fu=(sth){0,0,0,0};for(int i=x+1;i;i-=i&-i)fu=doit(fu,tr2[i]);return fu;}
void make_your_dream_come_true(int l,int r,int d,int s)
{
	if(l==r)
	{
		sort(s0[d][s].begin(),s0[d][s].end(),cmp1);
		sort(s1[d][s].begin(),s1[d][s].end(),cmp2);
		int l1=0;
		for(int i=0;i<s1[d][s].size();i++)
		{
			while(l1<s0[d][s].size()&&s0[d][s][l1].y<=s1[d][s][i].y)if(s0[d][s][l1].fu==1)add1(s0[d][s][l1].z,s0[d][s][l1].st),l1++;
			else add2(s0[d][s][l1].z,s0[d][s][l1].st),l1++;
			as[s1[d][s][i].id][0]=doit(as[s1[d][s][i].id][0],que1(s1[d][s][i].r));
			as[s1[d][s][i].id][1]=doit(as[s1[d][s][i].id][1],que2(s1[d][s][i].r));
			as[s1[d][s][i].id][1]=doit(as[s1[d][s][i].id][1],que1(s1[d][s][i].l-1));
			as[s1[d][s][i].id][0]=doit(as[s1[d][s][i].id][0],que2(s1[d][s][i].l-1));
		}
		for(int i=0;i<l1;i++)
		if(s0[d][s][i].fu==0)add1(s0[d][s][i].z,s0[d][s][i].st);
		else add2(s0[d][s][i].z,s0[d][s][i].st);
		return;
	}
	int mid=(l+r)>>1;
	for(int i=0;i<s0[d][s].size();i++)s0[d+1][s0[d][s][i].x>mid].push_back(s0[d][s][i]);
	for(int i=0;i<s1[d][s].size();i++)
	s1[d+1][s1[d][s][i].x>mid].push_back(s1[d][s][i]);
	make_your_dream_come_true(l,mid,d+1,0);make_your_dream_come_true(mid+1,r,d+1,1);
	sort(s0[d+1][0].begin(),s0[d+1][0].end(),cmp1);
	sort(s1[d+1][1].begin(),s1[d+1][1].end(),cmp2);
	int l1=0;
	for(int i=0;i<s1[d+1][1].size();i++)
	{
		while(l1<s0[d+1][0].size()&&s0[d+1][0][l1].y<=s1[d+1][1][i].y)if(s0[d+1][0][l1].fu==1)add1(s0[d+1][0][l1].z,s0[d+1][0][l1].st),l1++;
		else add2(s0[d+1][0][l1].z,s0[d+1][0][l1].st),l1++;
		as[s1[d+1][1][i].id][0]=doit(as[s1[d+1][1][i].id][0],que1(s1[d+1][1][i].r));
		as[s1[d+1][1][i].id][1]=doit(as[s1[d+1][1][i].id][1],que2(s1[d+1][1][i].r));
		as[s1[d+1][1][i].id][1]=doit(as[s1[d+1][1][i].id][1],que1(s1[d+1][1][i].l-1));
		as[s1[d+1][1][i].id][0]=doit(as[s1[d+1][1][i].id][0],que2(s1[d+1][1][i].l-1));
	}
	for(int i=0;i<l1;i++)
	if(s0[d+1][0][i].fu==0)add1(s0[d+1][0][i].z,s0[d+1][0][i].st);
	else add2(s0[d+1][0][i].z,s0[d+1][0][i].st);
	for(int i=0;i<2;i++)s0[d+1][i].clear(),vector<pts>().swap(s0[d+1][i]);
	for(int i=0;i<2;i++)s1[d+1][i].clear(),vector<que>().swap(s1[d+1][i]);
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v2[i]);
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(s[i][0]!=3)scanf("%d",&s[i][2]);
	}
	just_doit();
	make_your_dream_come_true(0,q,0,0);
	for(int i=1;i<=q;i++)
	if(s[i][0]==1||s[i][0]==5)
	{
		sth as1=doit(as[i][0],inv(as[i][1]));
		if(s[i][0]==1)printf("%d\n",as1.d);else printf("%d\n",as1.a);
	}
}
```



#### SCOI2020模拟?

##### auoj38 小D的奶牛

###### Problem

给一个 $n$ 个点的无向图，求团的数量。

$n\leq 50$

$10s,512MB$

###### Sol

考虑折半，处理出左边每一个合法的团以及这个团在另外一侧能加入哪些点以及右边每一个团。

使用 `dfs` 的方式枚举，位运算优化可以做到 $O(2^{\frac n2})$

然后相当于求左侧选一个 $A$，右侧选一个 $B$，满足 $R_A\cap B=B$ 的数量，可以看成求 $(U\setminus R_A)\cap B = \emptyset$ 的数量，`fwt` 求 `and` 卷积即可

复杂度 $O(2^{\frac{n}2}*n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 51
#define M 33554433
int n,s[N][N],f1[N],f2[N],as,s2[M];
long long s1[M];
char v[N][N];
void dfs(int d,int v1,int v2)
{
	if(d==n/2+1){if(v1)s1[((1<<25)-1)^v2]++,as++;return;}
	if((f1[d]&v1)==v1)dfs(d+1,v1|(1<<d-1),v2&f2[d]);
	dfs(d+1,v1,v2);
}
void dfs2(int d,int v1)
{
	if(d==n+1){if(v1)s2[v1]++,as++;return;}
	if((f2[d]&v1)==v1)dfs2(d+1,v1|(1<<(d-n/2-1)));
	dfs2(d+1,v1);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%s",v[i]+1);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)s[i][j]=v[i][j]-'0';
	as+=n-50;n=50;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n/2;j++)f1[i]|=(s[i][j]<<j-1),f2[i]|=(s[i][j+(n/2)]<<j-1);
	dfs(1,0,(1<<25)-1);dfs2(n/2+1,0);
	for(int i=2;i<=1<<25;i<<=1)
	for(int j=0;j<1<<25;j+=i)
	for(int k=j;k<j+(i>>1);k++)
	s1[k]+=s1[k+(i>>1)],s2[k]+=s2[k+(i>>1)];
	for(int i=0;i<1<<25;i++)s1[i]*=s2[i];
	for(int i=2;i<=1<<25;i<<=1)
	for(int j=0;j<1<<25;j+=i)
	for(int k=j;k<j+(i>>1);k++)
	s1[k]-=s1[k+(i>>1)];
	printf("%lld\n",s1[0]+as+1);
}
```



##### auoj39 小D的交通

###### Problem

给 $n$ 个点，对于一个 $x$ ，如果 $gcd(x+i-1,x+j-1)>1$ ，那么 $(i,j)$ 间有边，这样可以得到一个图。

求一个 $x$ 使得所有点连通或输出无解

$n\leq 10^5$

$1s,256MB$

###### Sol

对于 $n\leq 16$，可以暴力验证无解。

考虑如下乱搞方式：

首先让 $2|x$，还剩下偶数没有连通。考虑一个质数 $p$ ，可以通过调整 $x\bmod p$ ，将一个原本不与 $1$ 连通的 $q+1$ 与 $q+1-p$ 或 $q+1+p$ 连通，而因为后者是奇数，因此可以让 $p$ 与 $1$ 连通。

因为 $q+1-p,q+1+p$ 显然是奇数，所以显然这两个数与 $1$ 连通

为了保证能连上（$q+1-p\geq 1$ 或者 $q+1+p\leq n$），可以从中间向两侧开始选数。可以发现这样对于大的 $n$ 都能造出解，但对于 $n\leq 48$ 的情况存在问题。

但这种情况可以写个随机顺序加入，可以发现 $n\leq 100$ 都能跑动。

最后把所有的限制做一个高精 `CRT` 即可，需要稍微注意常数

复杂度 $O(\frac{n^2}{\omega\log n})$

###### Code

```cpp
#include<cstdio>
#include<random>
#include<algorithm>
using namespace std;
#define N 105020
int n,pr[N],ch[N],is[N],ct,ct2,tp=10,vl=1e8,s[N][2],ct3,las,v3;
int pw(int a,int b,int p){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
int p[N];
struct justdoit{
	int a[N],b[N];
	void init(){b[0]=1;ct3=0;}
	void muladd(int v)
	{
		int t2=0;
		for(int i=0;i<=tp;i++)
		{long long vl2=t2+a[i]+1ll*b[i]*v;t2=vl2/vl,a[i]=vl2%vl;}
	}
	void mul(int v)
	{
		int t2=0;
		for(int i=0;i<=tp;i++){long long vl2=t2+1ll*b[i]*v;t2=vl2/vl,b[i]=vl2%vl;}
	}
	void mula(int v)
	{
		int t2=0;
		for(int i=0;i<=tp;i++){long long vl2=t2+1ll*a[i]*v;t2=vl2/vl,a[i]=vl2%vl;}
	}
	int mod2(int x)
	{
		int t2=0;
		for(int i=tp;i>=0;i--)t2=(1ll*t2*vl+a[i])%x;
		return t2;
	}
	void add(int x,int y)
	{
		s[++ct3][0]=x;s[ct3][1]=y;
	}
	void output()
	{
		for(int i=1;i<=ct3;i++)
		{
			int v3=1;
			for(int j=1;j<=ct3;j++)if(j!=i)v3=1ll*v3*s[j][1]%s[i][1];
			int tp2=1ll*s[i][0]*pw(v3,s[i][1]-2,s[i][1])%s[i][1];
			mula(s[i][1]);muladd(tp2);mul(s[i][1]);
			while(b[tp-2])tp++;
		}
		int fg=0;
		for(int i=tp;i>=0;i--)
		if(!fg){if(a[i])printf("%d",a[i]),fg=1;}
		else printf("%08d",a[i]);
		printf("\n");
	}
}fu;
bool doit(int i)
{
	if((!is[i]&&i-pr[ct2+v3+1]<=0&&i+pr[ct2+v3+1]>n)||(ct2+v3+1>ct&&!is[i]))return 0;
	if(!is[i])
	{
		fu.add(pr[ct2+v3+1]-i%pr[ct2+v3+1]+1,pr[ct2+v3+1]);
		for(int j=i;j>0;j-=pr[ct2+v3+1])is[j]=1;
		for(int j=i;j<=n;j+=pr[ct2+v3+1])is[j]=1;ct2++;
	}
	return 1;
}
mt19937 rnd(1);
int main()
{
	scanf("%d",&n);
	if(n<=16){printf("No solution\n");return 0;}
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++ct]=i;
		for(int j=2;i*j<=n;j++)
		{
			ch[i*j]=1;
			if(i%j==0)break;
		}
	}
	if(n<=100)
	{
		for(int i=1;i<=n;i++)p[i]=i;
		while(1)
		{
			v3=1;ct2=0;
			fu.init();
			shuffle(p+1,p+n+1,rnd);
			fu.add(0,2);
			is[0]=is[n+1]=1;
			for(int i=1;i<=n;i++)is[i]=i&1;
			int fg=1;
			for(int i=1;i<=n&&fg;i++)
			fg&=doit(p[i]);
			if(fg)break;
		}
	}
	else
	{
		v3=1;
		fu.init();for(int j=1;j<=v3;j++)fu.add(0,pr[j]);
		for(int i=1;i<=n;i++)
		for(int j=1;j<=v3;j++)
		if((i-1)%pr[j]==0)is[i]=1;
		is[0]=is[n+1]=1;
		for(int i=0;i<=n/2;i++)doit((n+1)/2+i),doit((n+1)/2-i);
	}
	fu.output();
}
```



##### auoj40 N门问题

###### Problem

有 $n$ 扇门，有一扇后面有奖品，`A` 一开始随机选择一扇门，然后循环进行如下操作，直到只剩两扇门时结束：主持人随机打开一扇没有奖品且当前没有被选中的门，接下来 `A` 会选择当前是奖品的概率最大的门。这个概率在 `A` 认为主持人是随机选择的情况下计算条件概率。

现在 `B` 作为主持人，每次操作可以任意选择一扇没有奖品且当前没有被选中的门打开,但 `A` 仍然认为 `B`在随机选择。`B` 希望最小化 `A` 得到奖品的概率，求 `B` 在最优策略下 `A` 得到奖品的概率，保留 $6$ 位小数。

$n$ 由 $T$ 个同余方程 $n\equiv a_i(\bmod b_i)$ 给出，你需要先判断无解或者解出来 $n<2$ 的情况。

$T\leq 50000,lcm(b_i)\leq 10^{18}$

$1s,256MB$

###### Sol

考虑 `A` 怎么算每扇门后面有奖的条件概率，这相当于 $\frac{P(这扇门后面有奖|主持人这样操作)}{\sum_iP(第 i 扇门后面有奖|主持人这样操作)}$

分母对于所有门相同，因此只关心大小关系时不用考虑分母。

考虑分子的变化，对于当前选中的门，主持人开了门之后这个概率会乘 $\frac{1}{n-1}$。对于其它门概率会乘 $\frac{1}{n-2}$。

模拟这个过程，可以 `dfs` 得到 $n\leq 10$ 的答案。

发现 $n=11$ 时答案为 $0$，猜想 $n>10$ 答案为 $0$，即 `B` 总可以让 `A` 选不到门。

因为门越多操作空间越大，因此这看起来很对，~~但是严格证明完全不会~~。

最后 `excrt` 解出 $n$ 即可。复杂度 $O(T\log v)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
#define N 50020
ll f[N][2];
int n;
double as[11]={0,0,0.5,0.666667,0.625000,0.466667,0.416667,0.342857,0.291667,0.253968,0.225000};
ll exgcd(ll a,ll b,ll &x,ll &y){if(!b){x=1,y=0;return a;}ll g=exgcd(b,a%b,x,y);ll t=x;x=y;y=t-a/b*y;return g;}
ll mul(ll x,ll y,ll mod){ll tmp=(long double)x*y/mod;return x*y-tmp*mod;}
ll excrt()
{
	for(int i=2;i<=n;i++)
	{
		ll a=f[1][0],b=f[1][1],c=f[i][0],d=f[i][1],x,y;
		ll g=exgcd(a,c,x,y);
		if((d-b)%g)return -1;
		ll l=a/g*c;
		x=(x%l+l)%l;
		x=mul((d-b)/g,x,l);
		x=((mul(x,a,l)+b)%l+l)%l;
		f[1][0]=l,f[1][1]=x;
	}
	return f[1][1];
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%lld%lld",&f[i][1],&f[i][0]);
	ll fuc=excrt();
	if(fuc<2)printf("error\n");
	else if(fuc>10)printf("0.000000\n");
	else printf("%.6lf\n",as[fuc]);
}
```



#### SCOI2020模拟?

##### auoj41 钩子

###### Problem

有 $n$ 个钩子排成一列，$1$ 号钩子的左边和 $n$ 号钩子的右边分别有一个被占据的钩子。

$n$ 个人依次进来，每个人都会在当前与最近的一个被占据的钩子的距离最远的钩子中随机选一个占据。

求对于每一对 $(i,j)$，第 $i$ 个人占据第 $j$ 个钩子的概率，对输入大质数 $p$ 取模。

$n\leq 1000$

$1s,512MB$

###### Sol

可以发现，在大部分情况下，选择了一个段后，分裂出来的段中的最远距离一定小于原来的最远距离，这样就可以分步考虑，每次考虑最远距离最大的那些段。

但上述想法唯一的反例为长度为 $2$ 的段，考虑将长度为 $2$ 的段看成两个分开的长度为 $1$ 的段，这样操作后等价，且使得上述结论成立。

那么分步考虑，假设当前有 $k$ 个距离最大的段，一定是前 $k$ 个人选了这些段的中间部分。

有一部分段有 $2$ 的概率被选，另外一些段有 $1$ 的概率被选，可以 `dp` 求出每个人选某一段的概率，这部分复杂度 $O(n^2)$。

长度是奇数的段分成两段只有一种方式。对于长度为偶数的段，考虑钦定从中间前面断开，显然另外一种情况和这种情况完全对称，因此考虑求出一定选左侧的情况后面部分的概率，然后将每一个位置与它在这一段中对称的位置的概率取平均即可。

注意到每一个位置只会被取平均 $O(\log n)$ 次，因此复杂度 $O(n^2\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 1050
int dp[N][N],n,p,inv[N],ct,ct2,f[N],dp2[N][2],g[N][N];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
vector<pair<int,int> > st[N*20];
vector<int> st2;
void rotate(int l,int r)
{
	for(int x=1;x<=n;x++)
	for(int i=l,j=r;i<j;i++,j--)
	dp[x][i]=dp[x][j]=1ll*(dp[x][i]+dp[x][j])*(p+1)/2%p;
}
void doit(int a,int b)
{
	int n=a+b;
	for(int j=0;j<=a+b;j++)dp2[j][0]=dp2[j][1]=0;
	for(int j=0;j<=a+b;j++)for(int i=0;i<=b;i++)g[j][i]=0;
	g[0][b]=1;
	for(int i=0;i<n;i++)
	for(int j=0;j<=b;j++)
	{
		int c1=n-i-j,c2=j;
		int tp1=1ll*c1*inv[c1+c2*2]%p,tp2=2ll*c2*inv[c1+c2*2]%p;
		if(j)
		{
			g[i+1][j-1]=(g[i+1][j-1]+1ll*g[i][j]*tp2)%p;
			dp2[i][1]=(dp2[i][1]+1ll*g[i][j]*tp2)%p;
		}
		g[i+1][j]=(g[i+1][j]+1ll*g[i][j]*tp1)%p;
		dp2[i][0]=(dp2[i][0]+1ll*g[i][j]*tp1)%p;
	}
}
void solve()
{
	if(!st[ct2].size())return;
	st2.clear();
	int mx=0,v1=0,v2=0;
	for(int i=0;i<st[ct2].size();i++)mx=max(mx,(st[ct2][i].second-st[ct2][i].first+2)/2);
	for(int i=0;i<st[ct2].size();i++)
	if((st[ct2][i].second-st[ct2][i].first+2)/2==mx)
	{
		if(st[ct2][i].second==st[ct2][i].first)
		st2.push_back(st[ct2][i].first),v1++;
		else if(st[ct2][i].second==st[ct2][i].first+1)
		st2.push_back(st[ct2][i].first),st2.push_back(st[ct2][i].first+1),v1+=2;
		else
		{
			int l=st[ct2][i].first,r=st[ct2][i].second,mid=(l+r)>>1;
			st2.push_back(mid);st[ct2+1].push_back(make_pair(l,mid-1));st[ct2+1].push_back(make_pair(mid+1,r));
			if((l+r)&1)f[mid]=1,v2++;else v1++;
		}
	}
	else st[ct2+1].push_back(st[ct2][i]);
	doit(v1,v2);
	for(int i=ct+1;i<=ct+st2.size();i++)
	for(int j=0;j<st2.size();j++)
	dp[i][st2[j]]=1ll*inv[f[st2[j]]?v2:v1]*dp2[i-ct-1][f[st2[j]]]%p;
	for(int i=1;i<=n;i++)f[i]=0;
	int st1=ct2;ct+=st2.size();ct2++;
	solve();
	for(int i=0;i<st[st1].size();i++)
	if((st[st1][i].second-st[st1][i].first+2)/2==mx)
	rotate(st[st1][i].first,st[st1][i].second);
}
int main()
{
	scanf("%d%d",&n,&p);
	for(int i=1;i<=n;i++)inv[i]=pw(i,p-2);
	st[0].push_back(make_pair(1,n));solve();
	for(int i=1;i<=n;i++,printf("\n"))
	for(int j=1;j<=n;j++)printf("%d ",dp[i][j]);
}
```



##### auoj42 加减

###### Problem

有一个长度为 $n$ 的序列 $a$。对于每个 $k$，求出如下问题的答案：

在 $a$ 中选择一个长度为 $k$ 的子序列 $v$，最大化 $\sum_i(-1)^iv_i$。

$n\leq 5\times 10^5$

$1s,512MB$

###### Sol

如果 $k$ 为偶数，可以发现相当于选出 $\frac k2$ 个区间，最大化每个区间 $a_l-a_r$ 的和。

这可以看成一个费用流模型，因此所有 $k$ 为偶数部分的答案关于 $k$ 是上凸的。

同理可以发现，$k$ 为奇数的情况是类似的费用流模型，因此答案也是凸的。同理如果翻转权值，得到的结果还是凸的。

因此对于一段，考虑求出钦定这一段左侧选的个数的奇偶性以及这一段内部选的个数的奇偶性，则得到的 $dp$ 是一个凸函数。

考虑合并两个区间的 $dp$，因为上凸序列可以 $O(n+m)$ 做 $\max,+$ 卷积，因此可以 $O(n+m)$ 合并两段的 $dp$。

因此考虑分治，对于分治的每一段由两侧合并 $dp$，复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 505000
vector<long long> f[21][2][2][2];
int n,v[N];
long long g[2][N];
void merge(int d,int s,int len)
{
	for(int i=0;i<2;i++)
	for(int j=0;j<2;j++)
	{
		for(int k=0;k<2;k++)
		{
			int v1=i^k,v2=j^k,ct=1,f1=0,f2=0,s1=f[d+1][0][i][k].size(),s2=f[d+1][1][v1][v2].size();
			if(j)if(k)g[k][ct]=f[d+1][0][i][k][0],f1++,ct++;else g[k][ct]=f[d+1][1][v1][v2][0],f2++,ct++;
			else if(k)g[k][ct]=f[d+1][0][i][k][0]+f[d+1][1][v1][v2][0],f1++,f2++,ct++;
			if(s1+s2-f1-f2<len-ct+1)f[d+1][0][i][k].push_back(-2e16),s1++;
			for(int l=ct;l<=len;l++)
			if(f1==s1)g[k][ct++]=f[d+1][1][v1][v2][f2++];
			else if(f2==s2)g[k][ct++]=f[d+1][0][i][k][f1++];
			else if(f[d+1][1][v1][v2][f2]>f[d+1][0][i][k][f1])g[k][ct++]=f[d+1][1][v1][v2][f2++];
			else g[k][ct++]=f[d+1][0][i][k][f1++];
		}
		for(int k=1;k<=len;k++)g[0][k]+=g[0][k-1],g[1][k]+=g[1][k-1];
		for(int k=0;k<=len;k++)if(g[0][k]<g[1][k])g[0][k]=g[1][k];
		for(int k=1;k<=len;k++)f[d][s][i][j].push_back(g[0][k]-g[0][k-1]);
	}
	for(int i=0;i<2;i++)
	for(int j=0;j<2;j++)
	for(int k=0;k<2;k++)
	f[d+1][k][i][j].clear(),vector<long long>().swap(f[d+1][k][i][j]);
}
void solve(int l,int r,int d,int s)
{
	if(l==r)
	{
		f[d][s][0][0].push_back(-2e16);
		f[d][s][1][0].push_back(-2e16);
		f[d][s][0][1].push_back(v[l]);
		f[d][s][1][1].push_back(-v[l]);
		return;
	}
	int mid=(l+r)>>1;
	solve(l,mid,d+1,0);
	solve(mid+1,r,d+1,1);
	merge(d,s,(r-l+2)/2);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	solve(1,n,0,0);
	long long v1=0,v2=0,c1=0,c2=0;
	for(int i=1;i<=n;i++)
	if(i&1)v2+=f[0][0][0][1][c2++],printf("%lld ",v2);
	else v1+=f[0][0][0][0][c1++],printf("%lld ",v1);
}
```



#### SCOI2020模拟?

##### auoj44 矩阵求和

###### Problem

有一个 $n\times m$ 的矩阵，初始第 $i$ 行第 $j$ 列的值为 $(i-1)*m+j$，有 $q$ 次操作：

1. 交换矩阵两行
2. 交换矩阵两列
3. 求一个子矩阵做 $k$ 次二维前缀和后矩阵元素的和，模 $10^9+7$

询问独立，即不会影响矩阵的元素。

$n,m,q\leq 10^5,k\leq 10$

$2s,512MB$

###### Sol

设当前第 $i$ 行原来为第 $w_i$ 行，第 $j$ 列原来为第 $h_j$ 列，那么这个位置的元素为 $(w_i-1)*m+h_i$。

考虑多次二维前缀和的形式：

$$
v_{x,y}^{'}=\sum_{x_1\leq x,y_1\leq y}\sum_{x_2\leq x_1,y_2\leq y_1}\cdots\sum_{x_k\leq x_{k-1},y_k\leq y_{k-1}} v_{x_k,y_k}
$$


可以发现这相当于以起点开头终点结尾选出 $k$ 个二维坐标依次不降的点的方案数。

设询问矩阵边界为 $(x_1,y_1),(x_2,y_2)$，那么考虑每个点贡献，可以发现答案为

$$
\sum_{x_1\leq x\leq x_2}\sum_{y_1\leq y\leq y_2}v_{x,y}C_{x_1-x+k}^kC_{y_1-y+k}^k\\
=\sum_{x_1\leq x\leq x_2}\sum_{y_1\leq y\leq y_2}((w_x-1)*(m-1)+h_y)C_{x_1-x+k}^kC_{y_1-y+k}^k
$$

注意到点权的横纵部分可以拆开，因此这等于 $(m-1)\sum_{x_1\leq x\leq x_2}(w_x-1)C_{x_1-x+k}^k\sum_{y_1\leq y\leq y_2}C_{y_1-y+k}^k+\sum_{x_1\leq x\leq x_2}C_{x_1-x+k}^k\sum_{y_1\leq y\leq y_2}h_yC_{y_1-y+k}^k$

注意到 $C_i^k$ 是一个关于 $i$ 的 $k$ 次多项式，从而拆开后维护 $x^i,y^i,w_xx^i,h_yy^i$ 的区间和即可，可以 `BIT` 维护。

复杂度 $O(qk^2+qk\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105000
#define K 11
#define mod 1000000007
int n,m,q,p1[N],p2[N],C[K][K],a,b,c,d,e,su1[K],su2[K],inv[K],f[K][K];
char op[10];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct sth{
	int v[K][N],v2[N],v3[N];
	void add1(int k,int x,int v1){
	for(int i=x;i<=1e5;i+=i&-i)v[k][i]=(v[k][i]+v1)%mod;}
	int query(int k,int x){int as=0;for(int i=x;i;i-=i&-i)as=(as+v[k][i])%mod;return as;}
	void modify1(int x,int y,int z)
	{
		int st1=v3[x],st2=z;
		for(int i=0;i<=10;i++)add1(i,x,(st2-st1+mod)%mod),st2=1ll*st2*y%mod*(mod-1)%mod,st1=1ll*st1*v2[x]%mod*(mod-1)%mod;
		v2[x]=y;v3[x]=z;
	}
}t1,t2,t3,t4;
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n;i++)t1.modify1(i,i,i-1),p1[i]=i,t3.modify1(i,i,1);
	for(int i=1;i<=m;i++)t2.modify1(i,i,i),p2[i]=i,t4.modify1(i,i,1);
	for(int i=0;i<=10;i++)C[i][i]=C[i][0]=1;
	for(int i=2;i<=10;i++)for(int j=1;j<i;j++)C[i][j]=C[i-1][j]+C[i-1][j-1];
	for(int i=0;i<=10;i++)inv[i]=pw(i,mod-2);
	for(int i=0;i<=10;i++)
	{
		f[i][0]=1;
		int tp2=1;
		for(int j=0;j<i;j++)
		for(int k=j+1;k>0;k--)
		f[i][k]=(f[i][k]+f[i][k-1])%mod,f[i][k-1]=1ll*f[i][k-1]*(mod-j)%mod;
		for(int j=1;j<=i;j++)tp2=1ll*tp2*inv[j]%mod;
		for(int j=0;j<=i;j++)f[i][j]=1ll*f[i][j]*tp2%mod;
	}
	while(q--)
	{
		scanf("%s",op+1);
		if(op[1]=='R')
		{
			scanf("%d%d",&a,&b);
			t1.modify1(a,a,p1[b]-1);t1.modify1(b,b,p1[a]-1);
			p1[a]^=p1[b]^=p1[a]^=p1[b];
		}
		else if(op[1]=='C')
		{
			scanf("%d%d",&a,&b);
			t2.modify1(a,a,p2[b]);t2.modify1(b,b,p2[a]);
			p2[a]^=p2[b]^=p2[a]^=p2[b];
		}
		else
		{
			scanf("%d%d%d%d%d",&a,&b,&c,&d,&e);
			for(int i=0;i<=10;i++)su1[i]=(t3.query(i,c)-t3.query(i,a-1)+mod)%mod,su2[i]=(t4.query(i,d)-t4.query(i,b-1)+mod)%mod;
			int v1=0,v2=0,v3=0,v4=0;
			for(int i=0;i<=e;i++)for(int j=0,tp2=1;j<=i;j++,tp2=1ll*tp2*(c+e)%mod)v1=(v1+1ll*tp2*f[e][i]%mod*su1[i-j]%mod*C[i][j])%mod;
			for(int i=0;i<=e;i++)for(int j=0,tp2=1;j<=i;j++,tp2=1ll*tp2*(d+e)%mod)v2=(v2+1ll*tp2*f[e][i]%mod*su2[i-j]%mod*C[i][j])%mod;
			for(int i=0;i<=10;i++)su1[i]=(t1.query(i,c)-t1.query(i,a-1)+mod)%mod,su2[i]=(t2.query(i,d)-t2.query(i,b-1)+mod)%mod;
			for(int i=0;i<=e;i++)for(int j=0,tp2=1;j<=i;j++,tp2=1ll*tp2*(c+e)%mod)v3=(v3+1ll*tp2*f[e][i]%mod*su1[i-j]%mod*C[i][j])%mod;
			for(int i=0;i<=e;i++)for(int j=0,tp2=1;j<=i;j++,tp2=1ll*tp2*(d+e)%mod)v4=(v4+1ll*tp2*f[e][i]%mod*su2[i-j]%mod*C[i][j])%mod;
			int as1=(1ll*v3*v2%mod*m+1ll*v1*v4)%mod;
			printf("%d\n",as1);
		}
	}
}
```



##### auoj45 西行寺无余涅槃

###### Problem

有一个数 $x$，初始为 $0$。

给出 $k$ 个 $v_i$，有 $n$ 次操作，每次操作给出 $k$ 个 $s_i$，表示这次操作有 $v_i$ 种方式将 $x$ 异或上 $s_i(s_i<2^m)$，在每次操作中这个数必须正好被异或一次.

对于每个 $i=0,\cdots,2^m-1$，求最后 $x=i$ 的操作方案数，答案模 $998244353$。

$n*2^k\leq 10^7,m+k\leq 20$

$2s,512MB$

###### Sol

以下 `FWT` 均指 `xor-FWT`。

`FWT` 之后一个位置对另外一个位置的值的贡献系数为 $\pm 1$，又因为 $v_i$ 固定，因此一次操作得到的集合幂级数 `FWT` 之后每个位置只有 $2^k$ 种取值。考虑求出每种取值出现的次数。

设 $f_{i,S}$ 表示第 $i$ 个位置，满足`FWT` 后 $S$ 集合中元素对这个位置贡献为 $-1$，其余元素贡献为 $1$ 的操作数量。

对于一个操作，考虑在 $v_1$ 位置设为 $1$，其余位置设成 $0$，进行 `FWT`。设这时第 $i$ 个位置的值为 $s_i$，对于一个 $f_{i,S}$ ，如果 $1\in S$，那么显然 $f_{i,S}$ 对 $s_i$ 的贡献为 $-1$，否则贡献为 $1$。因此有 $s_i=\sum_{S}(-1)^{|\{1\}\cap S|}f_{i,S}$

显然有 $(a\oplus b)\and c=(a\and c)\oplus(b\and c)$，因此 `FWT` 中 $a\oplus b$ 对位置 $i$ 的贡献为 $a$ 对这个位置的贡献乘上 $b$ 对这个位置的贡献。因此：

对于一个集合 $T$，考虑在 $\oplus_{i\in T}v_i$ 的位置设成 $1$，其余位置设成 $0$，然后进行 `FWT`，那么有 $s_i=\sum_{S}(-1)^{|T\cap S|}f_{i,S}$。

设 $s_{i,T}$ 表示对 $T$ 集合进行上面的操作后第 $i$ 个位置的值，注意到 `FWT` 是线性变换，即 $a$ 的 `FWT` 加上 $b$ 的 `FWT` 等于 $a+b$ 的 `FWT`，可以将所有操作的 `FWT` 加起来一起做求出 $s$，这部分复杂度为 $O(n2^k+m2^{m+k})$

注意到 $s_{i,T}=\sum_S(-1)^{|T\cap S|}f_{i,S}$，因此 $s_i$ 等于 $f_i$ `FWT` 后的结果，因此 `IFWT` 即可求出 $f_i$，通过 $f_i$ 即可得到答案。

上述过程也可以通过对 $f_i$ 进行 `FWT`，考虑 `FWT` 后每一位的值得到结论。

复杂度 $O(n2^k+(m+k)2^{m+k})$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int f[1050000],n,m,k,s[15],v[15],g[1025],lbit[1025],v2[1025],h[1050000];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=k;i++)scanf("%d",&s[i]),v2[0]=(v2[0]+s[i])%mod;
	for(int j=1;j<1<<k;j++)
	for(int l=k;l>0;l--)if(j&(1<<l-1))lbit[j]=l;
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=k;j++)scanf("%d",&v[j]);
		g[0]=0;
		for(int j=1;j<1<<k;j++)g[j]=g[j^(1<<lbit[j]-1)]^v[lbit[j]];
		for(int j=0;j<1<<k;j++)f[(g[j]<<k)|j]++;
	}
	for(int l=2;l<=1<<m+k;l<<=1)
	for(int j=0;j<1<<m+k;j+=l)
	for(int s=j;s<j+(l>>1);s++)
	{
		int v1=f[s],v2=f[s+(l>>1)];
		f[s]=(v1+v2)%mod;
		f[s+(l>>1)]=(v1-v2+mod)%mod;
	}
	int inv=1;
	for(int i=1;i<=k;i++)inv=1ll*inv*(mod+1)/2%mod;
	for(int j=1;j<1<<k;j++)v2[j]=(v2[j^(1<<lbit[j]-1)]-2ll*s[lbit[j]]+2ll*mod)%mod;
	for(int i=0;i<1<<m;i++)
	{
		int as1=1;
		for(int j=0;j<1<<k;j++)
		as1=1ll*pw(v2[j],1ll*f[(i<<k)|j]*inv%mod)*as1%mod;
		h[i]=as1;
	}
	for(int l=2;l<=1<<m;l<<=1)
	for(int j=0;j<1<<m;j+=l)
	for(int s=j;s<j+(l>>1);s++)
	{
		int v1=h[s],v2=h[s+(l>>1)];
		h[s]=1ll*(mod+1)/2*(v1+v2)%mod;
		h[s+(l>>1)]=1ll*(mod+1)/2*(v1-v2+mod)%mod;
	}
	for(int i=0;i<1<<m;i++)printf("%d ",(h[i]+mod)%mod);
}
```



##### auoj46 鱼贯而入

###### Problem

给出 $n$ 个正整数 $a_{1,\cdots,n}$，你需要选择一个 $len$ 满足 $len\geq n$，使得在模 $len$，每次向后找第一个可以插入的位置的 `hash` 表中依次询问并插入 $n$ 个数，插入时向后遍历的次数最多，即最大化运行下面代码后的 $cnt$：

```cpp
long long cnt=0,a[N],n,len;
void insert(long long x)
{
    long long y=x%len;
    while(h[y]!=-1&&h[y]!=x)y=(y+1)%len,cnt+1;
    h[y]=x;
}
void solve()
{
    for(int i=1;i<=n;i++)insert(a[i]);
}
```

$n\leq 200,a_i\leq 10^{18}$

$8s,512MB$

###### Sol

显然如果不出现两个数模 $len$ 相同答案一定是 $0$，因此 $len$ 一定是某个 $a_j-a_i$ 的约数。

如果 $p,k*p$ 都是合法的 $len$，显然 $p$ 一定不比 $k*p$ 差，因此只需要找到所有满足小于自身的约数都小于 $n$ 的数即可。

可以发现，除了质数，每个数的最小质因子不超过 $\sqrt v$，因此质数外合法的数一定不会超过 $n^2$。

对于这部分，只需要求出每一对差的所有质因子和所有不超过 $n^2$ 的约数，暴力判断即可。事实上可以直接判断所有 $[n,n^2]$ 的数。

对于质数部分，考虑将每一对 $a_j-a_i$ 分解，`pollard-rho` 即可。

复杂度 $O(n^3\log v+n^2v^{\frac1 4})$ 或 $O(n^4\log v+n^2v^{\frac1 4})$ （判断的复杂度可以做到 $O(n)$ 或者 $O(n^2)$，但 $O(n^2)$ 做法跑不满也可以过）

注意 `pollard-rho` 写法，最好每 $63$ 次做一次 $\gcd$ 或者用类似的操作来消掉 $\log$ 因子。

###### Code

```cpp
#include<cstdio>
#include<map>
#include<set>
#include<cstdlib>
#include<ctime>
using namespace std;
#define N 205
#define ll long long
int m,n;
ll f[11]={2,3,5,7,11,13,17,19,23},tp[233],v[N],su[N],ct,vl[N];
set<ll> fu,as;
ll mul(ll x,ll y,ll mod){ll tmp=(long double)x*y/mod;return (x*y-tmp*mod+mod)%mod;}
ll pw(ll a,ll p,ll k){ll as=1;while(p){if(p&1)as=mul(as,a,k);a=mul(a,a,k);p>>=1;}return as;}
ll mrtest(ll a,ll p){ll ct=0,st2=p-1;while(~st2&1)st2>>=1,ct++;tp[0]=pw(a,st2,p);for(int i=1;i<=ct;i++)tp[i]=mul(tp[i-1],tp[i-1],p);if(tp[ct]!=1)return 0;for(int i=ct;i>0;i--)if(tp[i]==1&&(tp[i-1]>1&&tp[i-1]<p-1))return 0;else if(tp[i]!=1)return 1;return 1;}
ll mr(ll p){if(p==1)return 0;for(int i=0;i<9;i++)if(f[i]==p)return 1;for(int i=0;i<9;i++)if(!mrtest(f[i],p))return 0;return 1;}
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll pr(ll x)
{
	for(int i=0;i<9;i++)if(x%f[i]==0)return f[i];
	ll st=rand()%(x-1)+1,v1=1,v2=1,vl=1,ct=2;
	while(1)
	{
		v2=(mul(v2,v2,x)+st)%x;vl=mul(vl,v1<v2?v2-v1:v1-v2,x);
		ct++;
		if(ct%63==0)
		{
			ll g=gcd(vl,x);
			if(g>1&&g<x)return g;
			if(g==x)return x;
		}
		if((ct&-ct)==ct)
		{
			ll g=gcd(vl,x);
			if(g>1&&g<x)return g;
			if(g==x)return x;
			v1=v2;vl=1;
		}
	}
}
void justdoit(ll x)
{
	if(x==1)return;
	if(mr(x)){fu.insert(x);return;}
	ll st=x;while(st==x||st==1)st=pr(x);
	justdoit(st);justdoit(x/st);
}
void dfs(int d,ll x,ll las)
{
	if(x/las>n)return;
	if(d==ct+1){
	if(x>=n)as.insert(x);return;}
	dfs(d+1,x,las);
	for(int i=1;i<=su[d];i++)x*=v[d],dfs(d+1,x,v[d]);
}
int check(ll d)
{
	int su=0;
	map<ll,int> st;
	for(int i=1;i<=n;i++)
	{
		ll tp=vl[i]%d;
		while(st[tp])tp=(tp+1)%d,su++;
		st[tp]=1;
	}
	return su;
}
int main()
{
	scanf("%d%d",&m,&n);
	for(int i=1;i<=n;i++)scanf("%lld",&vl[i]);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	if(vl[i]>vl[j])
	{
		ct=0;fu.clear();
		justdoit(vl[i]-vl[j]);
		ll st2=vl[i]-vl[j];
		for(set<ll>::reverse_iterator it=fu.rbegin();it!=fu.rend();it++)
		{
			v[++ct]=*it;
			su[ct]=0;
			while(st2%v[ct]==0)st2/=v[ct],su[ct]++;
		}
		dfs(1,1,1);
	}
	int as1=0;
	for(set<ll>::iterator it=as.begin();it!=as.end();it++)as1=max(as1,check(*it));
	printf("%d\n",as1);
}
```



#### SCOI2020模拟?

##### auoj47 同桌与室友

###### Problem

有 $n$ 个人，有一些人住双人宿舍，一些人住单间，也就是说一些人有唯一的一个室友，有些人则没有。同时有些人会和他的同桌共用一张双人桌，另一些人则单独坐。

求出有多少个排列 $p$，满足 $i$ 换到 $p_i$ 的宿舍以及桌子上后，原本的室友以及同桌关系不变，模 $10^9+7$ 

$n\leq 2\times 10^5$

$1s,512MB$

###### Sol

将同桌关系看成连蓝边，室友关系看成连红边，那么原图由若干环和链组成，且边都是交错出现。

链有三种形式：两侧蓝色，两侧红色，两侧一蓝一红。环只有一种形式。

显然只有长度相同的环和长度形式相同的链间可以整体互换，显然不同连通块互换的方案数为每一种长度形式出现次数的阶乘的乘积。

再考虑一个连通块内部操作的情况。对于两侧相同的链，它自身可以翻转，因此每条这样的链答案再乘上 $2$

对于一个环，它可以进行旋转，且环长度一定为偶数，因此每个环答案需要乘上环长除以 $2$

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 400500
#define mod 1000000007
int ct[N][4];//1 - 1 1 - 2 2 - 2 circles
int n,k1,k2,a,b,fa[N],sz[N],as=1,s[N][2],is[N],fr[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int main()
{
	scanf("%d%d%d",&n,&k1,&k2);
	fr[0]=1;
	for(int i=1;i<=n;i++)fa[i+n]=fa[i]=i,sz[i]=2,s[i][0]=i,s[i][1]=i+n,fr[i]=1ll*fr[i-1]*i%mod;
	for(int i=1;i<=k1;i++)
	{
		scanf("%d%d",&a,&b);
		int a1=a,b1=b;a=finds(a),b=finds(b);
		sz[a]+=sz[b],fa[b]=a;
		s[a][s[a][0]!=a1]=s[b][s[b][0]==b1];
	}
	for(int i=1;i<=k2;i++)
	{
		scanf("%d%d",&a,&b);a+=n;b+=n;
		int a1=a,b1=b;a=finds(a);b=finds(b);
		if(a==b){is[a]=1;ct[sz[a]/2][3]++;
		as=1ll*as*sz[a]/2%mod;continue;}
		sz[a]+=sz[b],fa[b]=a;
		s[a][s[a][0]!=a1]=s[b][s[b][0]==b1];
	}
	for(int i=1;i<=n;i++)if(finds(i)==i&&!is[i])
	{
		int tp=(s[i][0]<=n)+(s[i][1]<=n);
		if(~tp&1)as=as*2%mod;
		ct[sz[i]/2][tp]++;
	}
	for(int i=0;i<4;i++)for(int j=1;j<=n;j++)as=1ll*as*fr[ct[j][i]]%mod;
	printf("%d\n",as);
}
```



##### auoj48 传送

###### Problem

有一棵 $n$ 个点的树，边有边权。每个点有一个区间 $[l_i,r_i]$，你可以花费 $x$ 的代价可以使得所有点的区间变成 $[l_i-x,r_i+x]$

在进行上一个操作后，你需要在每个点的区间中选择一个 $a_i$，使得对于任意的 $i,j,dis(i,j)\geq |a_i-a_j|$ 

求最小的代价使得存在合法方案

多组数据

$T\leq 3,n\leq 10^6$

$3s,512MB$

###### Sol

注意到 $|a_i-a_j|\leq |a_i-a_{k_1}|+|a_{k_1}-a_{k_2}|+...+|a_{k_l}-a_j|$，因此只需要每条树边的两端满足条件，整棵树就满足条件。

如果固定了一个 $x$，可以对于每个点求出这个点的 $a_i$ 在哪个区间中时子树内存在合法方案。一个点的区间就是所有儿子的限制（区间两侧扩展一个距离）之后的交。

注意到上述过程中区间的左边界一定形如 $v-x$，右边界一定形如 $v+x$。那么可以保留 $x$ 求出每个点的区间后再找最小的合法 $x$，省去二分的过程。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1050000
int n,l[N],r[N],t,ty,a,b,c,head[N],cnt;
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void dfs(int u,int fa){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),l[u]=max(l[u],l[ed[i].t]-ed[i].v),r[u]=min(r[u],r[ed[i].t]+ed[i].v);}
int rd(){char s=getchar();while(s<'0'||s>'9')s=getchar();int as=0;while(s>='0'&&s<='9')as=as*10+s-'0',s=getchar();return as;}
int main()
{
	scanf("%d%d",&t,&ty);
	while(t--)
	{
		scanf("%d",&n);cnt=0;for(int i=1;i<=n;i++)head[i]=0;
		for(int i=1;i<=n;i++)l[i]=rd();
		for(int i=1;i<=n;i++)r[i]=rd();
		for(int i=1;i<n;i++)a=rd(),b=rd(),c=rd(),adde(a,b,c);
		dfs(1,0);
		int as=0;
		for(int i=1;i<=n;i++)as=max(as,(l[i]-r[i]+1)/2);
		if(!ty)as=as>0;
		printf("%d\n",as);
	}
}
```



##### auoj49 生成树

###### Problem

给一个 $n$ 个点，有 $m$ 条红绿蓝三种颜色的边的图。求绿边数不超过 $g$，蓝边数不超过 $b$ 的生成树数量，模 $10^9+7$

$n\leq 40,m\leq 10^5$

$2s,512MB$

###### Sol

设 $f_{i,j}$ 表示有 $i$ 条绿边，$j$ 条蓝边的方案数。

将红边权值设为 $1$，绿边权值设为 $x$，蓝边权值设为 $y$，那么所有生成树的边权乘积的和即为 $\sum f_{i,j}x^iy^j$。

因为 $x,y$ 次数不超过 $n-1$，设 $y=x^n$，这就可以看成一元多项式，求出 $n^2$ 个点的点值后插值还原多项式即可。可以预处理每对点间边的数量避免 $O(n^2m)$ 部分。

复杂度 $O(n^5+m)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define M 42
#define K 1650
#define mod 1000000007 
int n,m,s,t,a,b,c,f[M][M],v[K],v2[K],as[K],su[M][M][3],inv[K];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
int det()
{
	int as=1;
	for(int i=1;i<n;i++)
	{
		int st2=i;
		for(int j=i;j<n;j++)
		if(f[j][i])st2=j;
		if(st2>i)as=mod-as;
		for(int j=1;j<n;j++)swap(f[st2][j],f[i][j]);
		if(!f[i][i])return 0;
		for(int j=i+1;j<n;j++)
		{
			int inv=1ll*f[j][i]*pw(f[i][i],mod-2)%mod;
			for(int k=i;k<n;k++)f[j][k]=(f[j][k]-1ll*f[i][k]*inv%mod+mod)%mod;
		}
	}
	for(int i=1;i<n;i++)as=1ll*as*f[i][i]%mod;
	return as;
}
int solve(int a,int b)
{
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	f[i][j]=0;
	for(int i=1;i<=n;i++)
	for(int j=i+1;j<=n;j++)
	{
		int vl=(su[i][j][0]+1ll*su[i][j][1]*a+1ll*su[i][j][2]*b)%mod;
		f[i][i]=(f[i][i]+vl)%mod;
		f[j][j]=(f[j][j]+vl)%mod;
		f[i][j]=(f[i][j]-vl+mod)%mod;
		f[j][i]=(f[j][i]-vl+mod)%mod;
	}
	return det();
}
void doit()
{
	int v1=1644;
	v2[0]=1;
	for(int i=1;i<=v1;i++)inv[i]=pw(i,mod-2);
	for(int i=1;i<=v1;i++)
	for(int j=i;j>=0;j--)
	v2[j+1]=(v2[j+1]+v2[j])%mod,v2[j]=1ll*v2[j]*(mod-i)%mod;
	for(int i=1;i<=v1;i++)
	{
		int st=1;
		for(int j=1;j<=v1;j++)if(j!=i)st=1ll*st*(mod+i-j)%mod;
		st=pw(st,mod-2);
		for(int j=0;j<=v1;j++)v2[j]=1ll*v2[j]*inv[i]%mod*(mod-1)%mod,v2[j+1]=(v2[j+1]-v2[j]+mod)%mod;
		for(int j=0;j<=v1;j++)as[j]=(as[j]+1ll*st*v[i]%mod*v2[j])%mod;
		for(int j=v1;j>=0;j--)v2[j+1]=(v2[j+1]+v2[j])%mod,v2[j]=1ll*v2[j]*(mod-i)%mod;
	}
}
int main()
{
	scanf("%d%d%d%d",&n,&m,&s,&t);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d%d",&a,&b,&c);
		if(a>b)a^=b^=a^=b;
		su[a][b][c-1]++;
	}
	for(int i=1;i<=1644;i++)v[i]=solve(i,pw(i,40));
	doit();
	int as1=0;
	for(int i=0;i<=s;i++)
	for(int j=0;j<=t;j++)
	as1=(as1+as[i+j*40])%mod;
	printf("%d\n",as1);
}
```



#### SCOI2020模拟?

##### auoj50 简单数学题

###### Problem

给定 $x,p$，求最小的 $a$ 使得 $fib_a\equiv x(\bmod p)$ 或输出无解。

多组数据

$T\leq 100,p\leq 2\times10^9$，$p$ 为质数且 $p\equiv 1,9(\bmod 10)$

$2s,512MB$

###### Sol

根据特征根公式，有 $fib_i=\frac{(\frac{1+\sqrt 5}2)^i-(\frac{1-\sqrt 5}2)^i}{\sqrt 5}$

因为 $p\equiv 1,9(\bmod 10)$，有 $(\frac p 5)=1$

又由二次剩余的性质，$(\frac p 5)(\frac 5 p)=(-1)^{\frac{(p-1)(5-1)} 4}=1$，所以 $(\frac 5 p)=1$。即 $5$ 存在模 $p$ 意义下的二次剩余，即 $(\frac{1+\sqrt 5}2)^i$ 可以在模 $p$ 下表示。考虑 $fib$ 的方程：

$$
(\frac{1+\sqrt 5}2)^i-(\frac{1-\sqrt 5}2)^i\equiv\sqrt 5 x(\bmod p)\\
((\frac{1+\sqrt 5}2)^i)^2-\sqrt 5 x(\frac{1+\sqrt 5}2)^i-(-1)^i\equiv 0(\bmod p)
$$

枚举 $i$ 的奇偶性，解二次方程后相当于求 $(\frac{1+\sqrt 5}2)^i\equiv t(\bmod p)$ 的最小奇数/偶数解，`BSGS` 即可。

复杂度 $O(T\sqrt p)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<queue>
using namespace std;
#define ll long long
ll x,p,T,tp1;
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
struct comp{int a,b;};
comp operator *(comp a,comp b){return (comp){(1ll*a.a*b.a%p+1ll*a.b*b.b%p*tp1%p)%p,(1ll*a.b*b.a%p+1ll*a.a*b.b%p)%p};}
comp pw(comp a,int b){comp as=(comp){1,0};while(b){if(b&1)as=as*a;a=a*a;b>>=1;}return as;}
int cipolla(int a=5)
{
	if(a==0)return 0;
	if(pw(a,(p-1)/2)==p-1)return -1;
	int tp2=1;
	while(pw(tp1=(1ll*tp2*tp2%p-a+p)%p,(p-1)/2)!=p-1)tp2=(((rand()<<15)|rand())%p+p)%p+1;
	tp1=(1ll*tp2*tp2%p-a+p)%p;
	return pw((comp){tp2,1},(p+1)/2).a;
}
struct fuc{
	#define K 1050000
	int hd[K],nt[K],vl[K],v2[K],ct;
	queue<int> st;
	void init()
	{
		while(!st.empty())hd[st.front()]=0,st.pop();
		for(int i=0;i<=ct;i++)v2[i]=nt[i]=vl[i]=0;
		ct=0;
	}
	void ins(int a,int b)
	{
		int tp1=a&1048575;st.push(tp1);
		nt[++ct]=hd[tp1];hd[tp1]=ct;v2[ct]=b;vl[ct]=a;
	}
	int que(int a)
	{
		int tp1=a&1048575,as=-1;
		for(int i=hd[tp1];i;i=nt[i])if(vl[i]==a)as=v2[i];
		return as;
	}
}fu;
pair<ll,ll> bsgs(int a,int b)
{
	ll k=sqrt(p*4+1);
	ll st1=1,as1=-1,as2=-1;
	if(b==1)as1=0;
	for(int i=1;i<=k*2;i++)
	{
		st1=1ll*st1*a%p;
		if(st1==b)
		if(as1==-1)as1=i;
		else if(as2==-1)as2=i;
	}
	if(as2!=-1)return make_pair(as1,as2);
	st1=b;
	fu.init();
	for(int i=0;i<k;i++)fu.ins(st1,i),st1=1ll*st1*a%p;
	st1=pw(a,k);
	ll v2=st1;
	for(int i=1;i<=k;i++)
	{
		ll tp1=fu.que(v2);
		if(tp1!=-1)
		{
			ll v1=i*k-tp1;
			if(as1==-1)as1=v1;
			else if(as2==-1&&v1!=as1)as2=v1;
		}
		v2=1ll*v2*st1%p;
	}
	if(!as1)return make_pair(-1,-1);
	else if(!as2)as2=as1+p-1;
	return make_pair(as1,as2);
}
ll solve()
{
	if(!x)return 0;
	ll v1=cipolla(5);
	ll tp1=1ll*(1+v1)*(p+1)/2%p,tp3=1ll*(p+1-v1)*(p+1)/2%p,fu1=1ll*x*v1%p;
	ll v11=(tp1-tp3+p)%p;
	if(v11!=v1)tp1=tp3;
	ll as=1e17;
	ll tp2=(1ll*fu1*fu1+4)%p,fuc2=cipolla(tp2);
	if(fuc2!=-1)
	{
		ll as1=1ll*(fu1+fuc2)*(p+1)/2%p,as2=1ll*(fu1-fuc2+p)*(p+1)/2%p;
		pair<ll,ll> st1=bsgs(tp1,as1),st2=bsgs(tp1,as2);
		if(st1.first%2==0)as=min(as,st1.first);
		if(st1.second%2==0)as=min(as,st1.second);
		if(st2.first%2==0)as=min(as,st2.first);
		if(st2.second%2==0)as=min(as,st2.second);
	}
	tp2=(1ll*fu1*fu1+p-4)%p,fuc2=cipolla(tp2);
	if(fuc2!=-1)
	{
		ll as1=1ll*(fu1+fuc2)*(p+1)/2%p,as2=1ll*(fu1-fuc2+p)*(p+1)/2%p;
		pair<ll,ll> st1=bsgs(tp1,as1),st2=bsgs(tp1,as2);
		if(st1.first%2==1)as=min(as,st1.first);
		if(st1.second%2==1)as=min(as,st1.second);
		if(st2.first%2==1)as=min(as,st2.first);
		if(st2.second%2==1)as=min(as,st2.second);
	}
	return as>1e16?-1:as;
}
int main()
{
	scanf("%lld",&T);while(T--)scanf("%lld%lld",&x,&p),printf("%lld\n",solve());
}
```



##### auoj51 简单图论题

###### Problem

给一个 $n$ 个点 $m$ 条边的简单无向图，你需要给每条边定向，使其满足如下条件：

1. 定向后图为DAG
2. 对于任意 $i,j,k$，如果存在边 $i\to j,i\to k$，则 $j,k$ 之间存在边。

判断是否有解，如果有解则构造任意解。

$n,m\leq 2\times 10^5$

$2s,512MB$

###### Sol

如果按照拓扑序进行删点，则可以发现这种方式合法当且仅当它是完美消除序列。

因此根据弦图相关理论合法当且仅当图是弦图，且只需要构造一个完美消除序列即可。

因此考虑 `MCS` 算法，即：

给每个点一个权值 $v_i$，初始 $v_i=0$。

重复进行下面操作 $n$ 次：

在还没有被删去的点中选择 $v_i$ 最大的一个点，将这个点放到当前的完美消除序列开头，删去这个点，将图中与这个点相邻的点 $v_i$ 加一。

可以对于每个 $x$ 使用vector或者类似的东西维护 $v_i\geq x$ 的所有点，并维护最大的 $v_i$。删除点时可以只记录这个点是否被删，在找最大值时再实际判断。

这样的复杂度为 $O(n+m)$

可以证明，对于弦图这样一定可以求出一个完美消除序列，对于非弦图这样求出的方案一定不合法。因此只需要再判断这个方案是否合法。

设方案为 $1,2,\cdots,n$，则相当于需要判断所有与 $i$ 相邻且编号大于 $i$ 的点之间都有边。

直接判断复杂度为 $O(nm)$，但可以发现如果后面的点都满足条件，则设与 $i$ 相邻，编号大于 $i$ 的点中编号最小的点为 $r_i$，则只需要判断如下条件即可判断 $i$ 是否满足条件：

$\forall j\geq r_i$，如果有边 $(i,j)$，则有边 $(r_i,j)$。

这样只需要判断 $O(m)$ 条边，复杂度 $O(n+m)$

具体证明可以参考[OI Wiki](https://oi-wiki.org/graph/chord/#mcs)

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 200500
vector<int> v[N];
int n,m,a,b,vl[N],st[N],is[N],ct,head[N],cnt,mx,id[N],fg,tp[N],s[N][2];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b),s[i][0]=a,s[i][1]=b;
	for(int i=1;i<=n;i++)v[0].push_back(i);
	while(ct<n)
	{
		while(v[mx].empty())mx--;
		int s1=v[mx].back();v[mx].pop_back();
		if(vl[s1]!=mx)continue;
		st[++ct]=s1;is[s1]=1;id[s1]=ct;
		for(int j=head[s1];j;j=ed[j].next)
		if(!is[ed[j].t])vl[ed[j].t]++,v[vl[ed[j].t]].push_back(ed[j].t);
		mx++;
	}
	for(int i=1;i<=n;i++)
	{
		int v1=0;
		for(int j=head[i];j;j=ed[j].next)if(id[ed[j].t]<id[i]&&id[v1]<id[ed[j].t])v1=ed[j].t;
		if(!v1)continue;
		for(int j=head[i];j;j=ed[j].next)tp[ed[j].t]=1;tp[v1]=2;
		for(int j=head[v1];j;j=ed[j].next)if(id[ed[j].t]<id[v1]&&tp[ed[j].t])tp[ed[j].t]=2;
		for(int j=head[i];j;j=ed[j].next)if(id[ed[j].t]<id[v1]&&tp[ed[j].t]==1)fg=1;
		for(int j=head[i];j;j=ed[j].next)tp[ed[j].t]=0;
	}
	if(fg)printf("-1\n");
	else for(int i=1;i<=m;i++)printf("%d",id[s[i][0]]<id[s[i][1]]);
}
```



##### auoj52 简单数据结构题

###### Problem

给定长度为 $n$ 的排列 $p$，称 $[l,r]$ 是一个连续段，当且仅当 $(\max_{i=l}^rp_i)-(\min_{i=l}^rp_i)=r-l$。

求有多少个连续段二元组 $[l_1,r_1],[l_2,r_2]$ 满足如下条件：

1. $|[l_1,r_1]\cap[l_2,r_2]|\geq k$
2. 两个区间中不存在一个区间包含另外一个区间。

答案模 $998244353$

$1\leq n,k\leq 3\times 10^5$

$2s,512MB$

###### Sol

显然如果建析合树，则只需要考虑每个合点的贡献，每个合点内部可以枚举交的左端点计算贡献，因此复杂度可以做到 $O(n)$。

下面考虑一个线段树做法：

考虑从左往右扫右端点 $r$，记录每个 $l$ 的 $(\max_{i=l}^rp_i)-(\min_{i=l}^rp_i)-(r-l)$。显然这个值非负。前两部分可以对 $p$ 维护两个大小的单调栈，看成 $O(n)$ 次区间修改，最后一个值也可以看成每次 $r$ 增大时做区间修改。

因此可以看成有一个序列 $a_i$，有若干次 $a$ 上的区间加，当前合法的 $l$ 为满足 $a_i=0$ 的 $i$，且任意 $a_i$ 非负。

考虑如果确定了左侧的二元组 $[l_1,r_1]$，则右侧二元组需要满足如下条件：

1. $l_1<l_2\leq r_1-k+1$
2. $r_2>r_1$

因此考虑设 $b_i$ 表示 $l=i$ 的贡献，加入一个 $r$ 的贡献可以看成如下操作：

对于每一个 $i\in[1,r-k+1]$，将 $b_i$ 加上 $[1,i-1]$ 部分合法的左端点数。

而一个 $r$ 作为右端点对答案的贡献可以看成当前所有合法左端点的 $b_i$ 之和。答案为每个 $r$ 的贡献和。

考虑使用线段树维护，在记录 $a$ 的区间修改标记外再记录如下两种标记：

1. $b$ 上的区间加标记。
2. 表示对当前区间进行考虑贡献操作的标记。

其中对一个区间考虑贡献操作为，对于区间内的每个 $i$，将 $b_i$ 加上区间左侧部分 $a_i$ 等于最小值的位置个数。

可以发现下放 $2$ 标记时，可以看成两边分别进行 $2$ 操作（注意一侧内部不存在两侧整体最小 $a_i$ 的情况），然后两侧中间的贡献可以看成右侧 $b_i$ 区间加左侧最小 $a_i$ 的数量，这部分可以看成一个 $1$ 标记。而一个区间修改也可以使用类似的方式拆成线段树上 $O(\log n)$ 个点上的标记。

再考虑记录区间内等于最小值的位置个数以及这些位置的 $b_i$ 之和，可以发现两种标记都可以直接作用到记录的信息上，因此可以直接维护。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 300500
#define ll long long
int n,k,p[N],sl[N],sr[N],cl,cr,as;
struct sth{int a,ct;ll b;};
sth operator +(sth a,sth b)
{
	if(a.a<b.a)return a;
	if(a.a>b.a)return b;
	a.b+=b.b;a.ct+=b.ct;return a;
}
struct node{int l,r,lz1,lz3;ll lz2;sth vl;}e[N*4];
void doit1(int x,int v){e[x].lz1+=v;e[x].vl.a+=v;}
void doit2(int x,ll v){e[x].lz2+=v;e[x].vl.b+=e[x].vl.ct*v;}
void doit3(int x,int v){e[x].lz3+=v;e[x].vl.b+=1ll*e[x].vl.ct*(e[x].vl.ct-1)/2*v;}
void pushdown(int x)
{
	if(e[x].lz1)doit1(x<<1,e[x].lz1),doit1(x<<1|1,e[x].lz1),e[x].lz1=0;
	if(e[x].lz2)doit2(x<<1,e[x].lz2),doit2(x<<1|1,e[x].lz2),e[x].lz2=0;
	if(e[x].lz3)
	{
		if(e[x<<1].vl.a==e[x].vl.a)doit3(x<<1,e[x].lz3),doit2(x<<1|1,1ll*e[x].lz3*e[x<<1].vl.ct);
		if(e[x<<1|1].vl.a==e[x].vl.a)doit3(x<<1|1,e[x].lz3);
		e[x].lz3=0;
	}
}
void pushup(int x){e[x].vl=e[x<<1].vl+e[x<<1|1].vl;}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;e[x].vl.ct=r-l+1;
	if(l==r)return;
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void modify1(int x,int l,int r,int v)
{
	if(e[x].l==l&&e[x].r==r){doit1(x,v);return;}
	pushdown(x);
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)modify1(x<<1,l,r,v);
	else if(mid<l)modify1(x<<1|1,l,r,v);
	else modify1(x<<1,l,mid,v),modify1(x<<1|1,mid+1,r,v);
	pushup(x);
}
int modify2(int x,int l,int r,int v)
{
	if(e[x].l==l&&e[x].r==r)
	{
		doit2(x,v);if(e[x].vl.a==0)doit3(x,1);
		return v+(e[x].vl.a==0)*e[x].vl.ct;
	}
	pushdown(x);
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)v=modify2(x<<1,l,r,v);
	else if(mid<l)v=modify2(x<<1|1,l,r,v);
	else v=modify2(x<<1,l,mid,v),v=modify2(x<<1|1,mid+1,r,v);
	pushup(x);return v;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&p[i]);
	build(1,1,n);
	for(int i=1;i<=n;i++)
	{
		while(cl&&p[sl[cl]]>p[i])modify1(1,sl[cl-1]+1,sl[cl],p[sl[cl]]),cl--;
		while(cr&&p[sr[cr]]<p[i])modify1(1,sr[cr-1]+1,sr[cr],-p[sr[cr]]),cr--;
		modify1(1,sl[cl]+1,i,-p[i]);sl[++cl]=i;
		modify1(1,sr[cr]+1,i,p[i]),sr[++cr]=i;
		if(i>1)modify1(1,1,i-1,-1);
		if(i>=k)
		{
			sth s1=e[1].vl;
			as=(as+2*s1.b)%998244353;
			modify2(1,1,i-k+1,0);
		}
	}
	printf("%d\n",as);
}
```



#### SCOI2020模拟?

##### auoj53 数一数

###### Problem

有一个 $n$ 行 $m$ 列的网格，每个位置上有一个 $\{0,1\}$ 中的数。每一列中正好有一个数为 $1$，第 $i$ 行的位置为 $1$ 的概率为 $p_i$。

对于一个网格，你可以从第一列走到最后一列，每次可以从 $(x,y)$ 走到 $(x-1,y+1),(x,y+1),(x+1,y+1)$ 中的一个。你的分数为经过的所有格子的权值和。

记一个网格的权值为你的最大分数，记 $f(m)$ 为 $n$ 行 $m$ 列网格权值的期望。求 $\lim_{m\to+\infty}\frac{f(m)}m$。答案模 $10^9+7$

$n\leq 6$

$2s,512MB$

###### Sol

对于一个确定的网格，可以设 $dp_{i,j}$ 表示第 $i$ 列第 $j$ 个位置结束的路径的最大权值，从前往后 $dp$。

考虑 $dp_{i}$ 中最大的数对应的方案，可以发现对应任意一个结束点，这个点向前倒着走 $n-1$ 步一定可以和最优方案重合。因此 $dp_i$ 中的最大最小值之差不超过 $n-1$。

同时最后的结果为最大值的分数，因此考虑将 $dp_i$ 减去最大值的结果作为状态，最大值的变化看作边权。则可以看为有一个图，每个点上会按照某个概率选择一条出边，$f(m)$ 为随机走 $m$ 次后经过的边权和的期望。

因此答案可以看成，重复随机游走过程，经过的一条边的边权的期望。

那么考虑求出随机游走后，到达一个点的概率 $p_i$，由随机游走可以得到状态数个方程，但这些方程和为 $0$，因此再补上方程 $\sum p_i=1$，即可得到随机游走后停留在一个点的概率。然后即可求出答案。

实际上状态很少，所以应该能过。

###### Code

oj上没数据，代码找不到了，咕了

##### auoj54 数二数

###### Problem

有一个 $[1,n]$ 间的正整数 $x$。你可以进行提问，每次提问你可以给出两个正整数 $l,r$，满足 $1\leq l\leq r\leq n$，你可以得到 $x$ 是否属于 $[l,r]$。

有 $2^{\frac{n(n+1)}2}$ 种可能的询问集合，求有多少种询问方式满足无论 $x$ 是多少，都可以通过询问确定 $x$。答案对给定质数取模。

$n\leq 300$

$1s,128MB$

###### Sol

可以确定 $x$ 当且仅当询问集合满足对于任意两个正整数 $x$，这些询问的结果不完全相同。

考虑容斥，枚举若干对 $(i,j)(i<j)$，钦定 $(i,j)$ 对所有询问的结果相同，即所有询问满足下列条件之一：

1. $r<i$
2. $l>j$
3. $i<l<r<j$
4. $l\leq i,r\geq j$

可以发现如下结论：

如果钦定了 $(a,c),(b,d)(a<b<c<d)$，则 $(a,b)$ 一定对此时的所有询问结果相同。

证明可以分类讨论。



将被钦定为相同的看成一个连通块，则一次钦定相当于连边 $(i,j)$，且如果将所有数排成一列，边都从上面连，则如果两条边相交，则这两条边所在的连通块会合并。

那么这些连通块之间一定相互不交或者包含。具体来说，记一个连通块覆盖的区间 $[l,r]$ 为属于这个连通块的最左侧和最右侧位置，则它满足如下性质：

任意一个连通块一定全部在 $[l,r]$ 内或者 $[l,r]$ 外。

那么对于一种连通块的划分方式，它可以使用如下方式进行分解：

考虑 $1$ 属于的连通块，记右端点为 $r$，然后考虑 $r+1$，这样一直到右端点为 $n$。

这相当于将 $[1,n]$ 划分为若干个区间 $[l,r]$，满足每个区间被一个连通块覆盖。此时可以发现剩下的连通块都在一个区间内部。

考虑一个区间，这个区间被一个连通块覆盖，考虑不属于这个连通块的位置，它们构成若干个区间。则显然不可能有一个连通块在多个区间内，否则它会和大连通块合并。因此可以将剩余部分分为若干个区间，每个区间内是一个子问题，可以重复上面过程。



对于这样一种连通块划分方式，考虑可能的询问集合数量，则只需要求出合法的询问数量。

考虑第一步中划分出的区间，设有 $k$ 个区间，则显然这一部分有 $C_{k+1}^2$ 个合法的询问。接下来不同区间间不可能再有合法询问。对于一个区间内部，经过了覆盖这个区间的连通块的询问一定不行（整个区间在上一部分算过），因此只需要再对于分出的每个区间计算即可。最后总的合法数量为每一步的 $C_{k+1}^2$ 的和。



然后考虑一种划分方式的容斥系数，考虑一个连通块，则相当于有 $m$ 个点，你需要选择一个边的集合，使得连接这些边，再按照上面的结论进行合并连通块后，这 $m$ 个点连通。选择边集 $S$ 的贡献为 $(-1)^{|S|}$。可以发现不同连通块间的问题独立，只需要考虑一个连通块的情况。

设这个值为 $v_m$，显然 $m$ 个点任意连，计算容斥系数后的权值为 $[m=1]$，考虑减去实际上不连通的方案数。枚举 $1$ 所在的连通块，则可以发现 $1$ 所在的连通块的点将所有点分成了若干个区间，每个区间内部可以任意连边。但如果一个区间长度大于等于 $2$，则这个区间内有一条边可以任意连，这种情况系数一定会相互抵消，因此只需要减去每个区间长度不超过 $1$ 的情况。枚举 $1$ 所在的连通块大小 $i$，考虑可能的连通块数，相当于将剩下 $m-i$ 个数放入 $i$ 个 $[x,x+1]$ 之间的位置，每个位置最多放一个数，因此转移为：
$$
v_m=[m=1]-\sum_{i=1}^mC_{i}^{m-i}v_i
$$
这里可以 $O(n^2)$ 计算，但可以观察到 $v_m=(-1)^{m-1}c_{m-1}$，其中 $c_i$ 为卡特兰数。



在这些基础上考虑 $dp$，设 $f_n$ 表示一个长度为 $n$ 的区间，且一个连通块覆盖了这个区间时，所有情况的权值和。设 $g_n$ 表示没有一个连通块覆盖的限制时的权值和。则答案为 $g_n$。

考虑转移 $g$，如果当前大区间被分成了 $k$ 个被连通块覆盖的区间，则有 $2^{C_{k+1}^2}$ 的贡献，因此考虑设 $sg_{n,k}$ 表示长度为 $n$ 的区间，被分成了 $k$ 个有连通块覆盖的区间的所有情况的权值和。则有：
$$
sg_{n,k}=\sum_{i=1}^nsg_{n-i,k-1}*f_i\\
g_n=\sum_{i}sg_{n,i}2^{C_{i+1}^2}
$$
然后考虑转移 $f$，去掉覆盖当前区间的连通块后，剩余若干个小的区间。这些小区间的情况为 $g_i$，且小区间之间必须有大区间的元素分隔，两侧也必须有大区间的元素。因此考虑设 $sf_{n,k}$ 表示长度为 $n$ 的区间，当前有一个连通块覆盖整个区间，且这个连通块有 $k$ 个点，还没有计算这个连通块的系数的权值和。转移考虑枚举加入的下一个小区间的长度，有：
$$
sf_{n,k}=[n=k=1]+\sum_{i=1}^nsf_{n-i,k-1}*g_{i-1}\\
f_n=\sum_i sf_{n,i}*v_i
$$
直接 $dp$ 即可，复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 305
int n,p,ca[N],f[N][N],f1[N],g[N][N],g1[N],p2[N];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&p);
	for(int i=1;i<=n;i++)p2[i]=pw(2,i*(i+1)/2);
	ca[1]=1;for(int i=2;i<=n;i++)ca[i]=1ll*ca[i-1]*(p-1)%p*(4*i-6)%p*pw(i,p-2)%p;
	f[1][1]=1;f1[1]=1;g[1][1]=g[0][0]=1;g1[1]=2;g1[0]=1;
	for(int i=2;i<=n;i++)
	{
		for(int j=1;j<=i-1;j++)
		for(int k=1;k<=j;k++)
		f[i][k+1]=(f[i][k+1]+1ll*f[j][k]*g1[i-j-1])%p;
		for(int k=1;k<=i;k++)f1[i]=(f1[i]+1ll*f[i][k]*ca[k])%p;
		for(int j=0;j<i;j++)for(int k=0;k<=j;k++)
		g[i][k+1]=(g[i][k+1]+1ll*g[j][k]*f1[i-j])%p;
		for(int k=1;k<=i;k++)g1[i]=(g1[i]+1ll*p2[k]*g[i][k])%p;
	}
	printf("%d\n",g1[n]);
}
```



##### auoj55 数三数

###### Problem

定义 $F(p)$ 为将正整数 $p$ 表示为若干个不同的斐波那契数之和的方案数。

给一个长度为 $n$ 的序列 $v$，对于每一个 $i$，求 $F(f_{v_1}+\cdots+f_{v_i})$，其中 $f_i$ 为第 $i$ 个斐波那契数。答案模 $10^9+7$

$n\leq 10^5,v_i\leq 10^9$

$1s,256MB$

###### Sol

考虑 $p$ 的斐波那契表示，即 $a_1,\cdots$ 满足 $\sum a_if_i=p$ 且 $a_i\in \N^+,a_i+a_{i+1}\leq 1$。

而 $F(p)$ 为满足如下条件的非负整数组 $b$ 个数：$\sum b_if_i=p,b_i\leq 1$。

考虑从高向低确定 $b$ 的每一位，并记录当前高位部分 $a,b$ 表示的数的差值。设当前考虑到了第 $k$ 位，则差值一定可以表示为 $c_1f_{k-1}+c_2f_{k-2}$。

考虑这个的转移，可以发现转移为：
$$
c_1'=c_1+c_2+a_i-b_i\\
c_2'=c_1+a_i-b_i
$$
则有如下性质：

1. 对于合法方案，任意时刻 $c_1,c_2\geq 0$，从而 $c_1\geq c_2$。

证明：从高位向低位考虑，直到第一个不满足条件的位。则因为 $a_i,b_i\in\{0,1\}$，一定有 $b_i=1,a_i=0$，从而 $c_1=0,c_2=0$，因此转移后 $c_1=c_2=-1$。但因为 $a$ 是斐波那契表示，因此 $\sum_{i<k}a_ip_i<f_k=f_{k-1}+f_{k-2}$，因此这种情况最后不能让和变回非负，从而无解。

2. 如果 $c_1\geq 2$，则无解。

证明：考虑找到第一个满足这个条件的位，则之前 $c_1,c_2\leq 1$，从而当前 $c_2\geq 1$。但 $\sum_{i<k}b_ip_i<p_{k-1}+p_k$（可以归纳），因此无解。

从而最多存在三种状态：$\{c_1,c_2\}=\{1,1\},\{1,0\},\{0,0\}$。因此可以进行类似数位 $dp$ 的操作。



然后考虑维护斐波那契表示。考虑加入一个 $f_p$ 的过程：

1. 如果 $a_{p-1,p,p+1}=0$，那么可以直接加入 $a_p=1$。
2. 否则，如果 $a_p=0,a_{p+1}=1$，则可以合并 $f_p+f_{p+1}=f_{p+2}$，令 $a_{p+1}=0$，看成加入 $p+2$。
3. 否则，如果 $a_{p}=0,a_{p-1}=1$，则可以类似合并 $p,p-1$。
4. 否则，如果 $a_p=1$，注意到 $2a_p=a_{p+1}+a_{p-2}$，考虑分成两部分递归。

前三种操作只有 $O(n)$ 次，但最后一种操作不能保证复杂度。一种卡满的方式是 $2,4,\cdots,2k,2k,2k+1,\cdots,3k$，这样需要 $k^2$ 次修改。~~但是原数据是 $10^9$ 内随机的，所以说暴力都能过~~

一种处理方式是维护插入点向后的连续 $01$ 交错段，把 $4$ 操作的连续一段一起处理，这样一段 $4$ 操作可以看成插入删除 $O(1)$ 个位置，然后平衡树上维护 $dp$ 的矩阵即可。

复杂度 $O(n\log n)$ ~~但这东西细节巨大多，常数巨大大，时限这么离谱看起来就过不去~~，~~好像矩阵可以变成2*2，这样就能过了~~

###### Code

~~因为不想写上面那个东西，这里就放暴力了~~

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
#include<map>
using namespace std;
#define N 200500
#define mod 1000000007
int n,s[N],tp[N],m;
set<int> sr,si;
map<int,int> vi;
void modify(int x,int f){if(f)sr.insert(x);else sr.erase(x);}
int query(int x){return sr.find(x)!=sr.end();}
void ins0(int x)
{
	si.insert(x);
	if(query(x))
	{
		modify(x,0);
		ins0(x+1);
		int tp=x-2;if(tp<1)tp++;
		if(tp>0)ins0(tp);
		return;
	}
	if(query(x+1))modify(x+1,0),ins0(x+2);
	else if(query(x-1))modify(x-1,0),ins0(x+1);
	else modify(x,1);
}
struct mat
{
	int v[3][3];
	mat(){for(int i=0;i<3;i++)for(int j=0;j<3;j++)v[i][j]=0;}
}s0[N],s1[N],r0,r1;
mat operator *(mat a,mat b)
{
	mat c;
	for(int i=0;i<3;i++)for(int k=0;k<3;k++)for(int j=0;j<3;j++)c.v[i][j]=(c.v[i][j]+1ll*a.v[i][k]*b.v[k][j])%mod;
	return c;
}
mat pw(mat a,int b)
{
	mat as;for(int i=0;i<3;i++)as.v[i][i]=1;
	while(b)
	{
		if(b&1)as=as*a;
		a=a*a;b>>=1;
	}
	return as;
}
int is[N];
struct segt{
	struct node{int l,r;mat s;}e[N*4];
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].s=s0[l];return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		e[x].s=e[x<<1|1].s*e[x<<1].s;
	}
	void modify(int x,int s)
	{
		if(e[x].l==e[x].r)
		{
			e[x].s=is[s]?s1[s]:s0[s];
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		modify(x<<1|(mid<s),s);
		e[x].s=e[x<<1|1].s*e[x<<1].s;
	}
}tr;
void modify1(int x,int f)
{
	if(f)sr.insert(x);else sr.erase(x);
	int id=vi[x];
	is[id]=f;tr.modify(1,id);
}
void ins1(int x)
{
	if(query(x))
	{
		modify1(x,0);
		ins1(x+1);
		int tp=x-2;if(tp<1)tp++;
		if(tp>0)ins1(tp);
		return;
	}
	if(query(x+1))modify1(x+1,0),ins1(x+2);
	else if(query(x-1))modify1(x-1,0),ins1(x+1);
	else modify1(x,1);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&s[i]),ins0(s[i]);
	for(set<int>::iterator it=si.begin();it!=si.end();it++)tp[++m]=*it,vi[tp[m]]=m;
	r0.v[0][0]=r0.v[1][0]=r0.v[1][2]=r0.v[2][1]=r1.v[0][0]=r1.v[0][2]=r1.v[1][2]=1;
	for(int i=1;i<=m;i++)s0[i]=pw(r0,tp[i]-tp[i-1]),s1[i]=r1*pw(r0,tp[i]-tp[i-1]-1);
	tr.build(1,1,m);
	sr.clear();
	for(int i=1;i<=n;i++)ins1(s[i]),printf("%d\n",tr.e[1].s.v[0][0]);
}
```



#### SCOI2020模拟?

##### auoj56 seed

###### Problem

有 $n$ 个种子，第 $i$ 个种子价值为 $v_i$，你有 $p_i$ 次拿到这个种子时把它放回去的机会。

你会随机拿出一个种子，如果可以放回去，你可以选择放回去，再拿一个种子并重复之前的过程。求最优策略下最后种子权值的期望，输出实数。

$n\leq 10^5,\sum p_i\leq 20$

$1s,512MB$

###### Sol

设 $dp_S$ 表示放回去的种子的可重集(放回多次算多次)为 $S$ 时的答案。

枚举拿出来的是 $p_i=0$ 的还是 $p_i>0$ 的，第一种可以预处理和后一起算，对第二种的转移记忆化搜索即可。

复杂度 $O((\sum p_i)\prod(p_i+1))$，但用二进制表示写成 $2^{\prod p_i}$ 更好写。

###### Code

```cpp
#include<cstdio>
using namespace std;
int n,a,b,v[21][2],is1[21],ct,su;
long long t1;
long double dp[1050000];
long double dfs(int k)
{
	if(dp[k]>=0)return dp[k];
	long double as=(long double)1.0*t1/n;
	int is[21];
	for(int i=1;i<=ct;i++)is[i]=0;
	for(int i=1;i<=su;i++)if(!(k&(1<<i-1)))
	{
		if(is[is1[i]])continue;
		int v2=is1[i];is[v2]=1;
		long double f1=v[v2][1],f2=dfs(k|(1<<i-1));
		if(f2>f1)f1=f2;
		as+=f1/n;
	}
	for(int i=1;i<=ct;i++)if(!is[i])as+=(long double)1.0*v[i][1]/n;
	return dp[k]=as;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%d%d",&a,&b);
		if(!a)t1+=b;
		else
		{
			v[++ct][0]=a;v[ct][1]=b;
			for(int i=su+1;i<=su+a;i++)is1[i]=ct;
			su+=a;
		}
	}
	for(int i=0;i<(1<<su);i++)dp[i]=-1;
	printf("%.4Lf\n",dfs(0));
}
```



##### auoj57 string

###### Problem

给定字符集大小 $p$，求有多少个长度不超过 $n$ 的字符串能被划分成不超过两个回文串。答案模 $998244353$

$n\leq 10^5$

###### Sol

将字符串首尾相接形成环，字符串合法当且仅当在环上存在至少一条对称轴，

考虑计算对称轴的数量。对于长度 $n$，如果 $n$ 为奇数，那么所有字符串的对称轴共有 $n*p^{\frac{n+1}2}$ 条，如果为偶数则有 $\frac n 2*(p^{\frac n 2}+p^{\frac{n+2} 2})$ 条。

接着考虑如何去重，可以发现，如果一个环存在两条对称轴，那么这个环上的字符串一定是循环的，且循环节一定是一个题目中合法的串。

设 $f_i$ 表示循环节为 $i$ 的约数的本质不同合法串数量，$g_i$ 表示循环节为 $i$ 的本质不同合法串数量，可以容斥通过 $f$ 求出 $g$。

最后枚举串长，枚举循环节，减去多算的对称轴数量即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
int n,p,f[104040],as;
#define mod 998244353
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void doit(int i,int j)
{
	as=(as-1ll*(i/j-1)*f[j]%mod*j%mod+mod)%mod;
}
int main()
{
	scanf("%d%d",&n,&p);
	for(int i=1;i<=n;i++)
	if(i&1)f[i]=pw(p,(i+1)/2);
	else f[i]=1ll*(mod+1)/2*(pw(p,i/2)+pw(p,i/2+1))%mod;
	for(int i=1;i<=n;i++)
	for(int j=i*2;j<=n;j+=i)f[j]=(f[j]-f[i]+mod)%mod;
	for(int i=1;i<=n;i++)
	if(i&1)as=(as+1ll*i*pw(p,(i+1)/2))%mod;
	else as=(as+1ll*i/2*pw(p,i/2)+1ll*i/2*pw(p,i/2+1))%mod;
	for(int i=1;i<=n;i++)
	for(int j=i;j<=n;j+=i)
	doit(j,i);
	printf("%d\n",as);
}
```



##### auoj58 tree

###### Problem

给定整数 $x$，质数 $p$，对于一条路径，定义它的权值为 $\sum x^iw_i \bmod p$，其中 $w_i$ 为经过的第 $i$ 条边的边权。

给一棵 $n$ 个点的有边权树，求有多少个三元组 $(a,b,c)$ 满足 $(a,b),(b,c),(a,c)$ 的权值全部为 $0$ 或者全部非 $0$。

$n\leq 10^5$

$3s,512MB$

###### Sol

考虑看成一个完全图，每条边 $(a,b)$ 有两种状态：路径权值为 $0$ 或非 $0$。

对于一个三元组，考虑每一对边的状态是否相同，则：

1. 对于一个合法的三元组，有 $3$ 对边相同。

2. 对于一个不合法的三元组，有 $1$ 对边相同。

因此可以通过计算合法的边对数计算答案。

只需要求出对于每个点，有多少个点到它的路径权值为 $0$，它到多少个点的路径权值为 $0$，即可计算这个点作为中点的合法边对数。

考虑点分治，如果 $p$ 不整除 $x$，可以将一条路径权值全部除以 $x^i$，使得根节点处乘的权值为 $x^0$。

然后给出一边向上/向下路径的值就可以知道另外一边向下/向上路径的权值是多少时整条路径权值为 $0$，使用 `map` 统计即可。

存在 $p|x$ 的特殊情况，但这种情况容易解决。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
using namespace std;
#define N 105000
#pragma GCC optimize(3)
int n,k,p,a,b,c,head[N],cnt,v2[N],dep[N],sz[N],as,vl,as2,s1[N],s2[N],vis[N],v[N],pw2[N],ipw2[N],f2[N];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa)
{
	sz[u]=1;int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],mx=max(mx,sz[ed[i].t]);
	mx=max(mx,vl-sz[u]);
	if(mx<as)as=mx,as2=u;
}
map<int,int> fu3,fu4;
void dfs2(int u,int fa)
{
	dep[u]=dep[fa]+1;
	fu4[f2[u]]++;fu3[v2[u]]++;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])
	{
		dep[ed[i].t]=dep[u]+1;
		f2[ed[i].t]=(f2[u]-1ll*v[ed[i].t]*ipw2[dep[ed[i].t]]%p+p)%p;
		v2[ed[i].t]=(v2[u]+1ll*v[ed[i].t]*pw2[dep[ed[i].t]])%p;
		dfs2(ed[i].t,u);
	}
}
void dfs3(int u,int fa)
{
	if(fu3.count(f2[u]))s1[u]+=fu3[f2[u]];if(fu4.count(v2[u]))s2[u]+=fu4[v2[u]];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs3(ed[i].t,u);
}
void dfs4(int u,int fa)
{
	fu4[f2[u]]++;fu3[v2[u]]++;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs4(ed[i].t,u);
}
void dfs5(int u,int fa)
{
	if(fu3.count(f2[u]))s1[u]-=fu3[f2[u]];if(fu4.count(v2[u]))s2[u]-=fu4[v2[u]];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs5(ed[i].t,u);
}
void work(int u)
{
	fu3.clear();fu4.clear();
	vis[u]=1;
	dep[0]=-1;v2[u]=0;f2[u]=(p-v[u]-(v[u]==0?p:0));dfs2(u,0);dfs3(u,0);
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	fu3.clear(),fu4.clear(),dfs4(ed[i].t,u),dfs5(ed[i].t,u);
}
void doit(int u){work(u);for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])dfs1(ed[i].t,u),vl=sz[ed[i].t],as=1e7,dfs1(ed[i].t,u),doit(as2);}
int main()
{
	scanf("%d%d%d",&n,&k,&p);
	if(k%p==0){printf("%lld\n",1ll*n*n*n);return 0;}k%=p;
	pw2[0]=ipw2[0]=1;int inv=pw(k,p-2);
	for(int i=1;i<=n;i++)pw2[i]=1ll*pw2[i-1]*k%p,ipw2[i]=1ll*ipw2[i-1]*inv%p;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),v[i]%=p;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	vl=n;as=1e7;dfs1(1,0);doit(as2);
	long long fuc=1ll*n*n*n*2;
	for(int i=1;i<=n;i++)fuc=(fuc-1ll*s1[i]*(n-s2[i])-2ll*s1[i]*(n-s1[i])-2ll*s2[i]*(n-s2[i])-1ll*s2[i]*(n-s1[i]));
	printf("%lld\n",fuc/2);
}
```



#### SCOI2020模拟?

##### auoj62 string

###### Problem

给两个长度为 $n,m$ 的字符串 $s,t$，求 $s$ 有多少个长度与 $m$ 相同的子串满足字串与 $t$ 比较，不同的位置不超过 $k$ 个

$n\leq 10^6,|\sum|=8$

$5s,512MB$

###### Sol

考虑计算每个位置开始的子串与第二个串匹配的数量，枚举每种字符，得到的形式是一个差卷积，`NTT` 即可。

复杂度 $O(n\log n*|\small\sum|)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 2100500
#define mod 998244353
int sr[2][N],n,m,k,a[N],b[N],c[N],rev[N],ntt[N],as[N];
char s[N],t[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=i,ct1=l>>1;ct1<l;j++,ct1++)
	{
		int s1=ntt[j],s2=1ll*ntt[j+(l>>1)]*sr[t][ct1]%mod;
		ntt[j]=(s1+s2)%mod;ntt[j+(l>>1)]=(s1-s2+mod)%mod;
	}
	int inv=t==0?pw(s,mod-2):1;
	for(int i=0;i<s;i++)a[i]=1ll*inv*ntt[i]%mod;
}
void doit(char st)
{
	int l=1;while(l<=n+m)l<<=1;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=1;i<=n;i++)if(s[i]==st)a[i]=1;
	for(int i=1;i<=m;i++)if(t[i]==st)b[m-i]=1;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;
	dft(l,a,0);
	for(int j=0;j<=n-m;j++)as[j]+=a[j+m];
}
int main()
{
	scanf("%d%s%s",&k,s+1,t+1);n=strlen(s+1);m=strlen(t+1);
	if(n<m){printf("0\n");return 0;}
	for(int t=0;t<=1;t++)
	for(int i=1;i<=21;i++)
	{
		sr[t][1<<i-1]=1;
		int tp1=pw(3,(mod-1)>>i),tp2=tp1;
		if(!t)tp1=pw(tp1,mod-2),tp2=tp1;
		for(int j=1;j<(1<<i-1);j++)sr[t][(1<<i-1)+j]=tp2,tp2=1ll*tp2*tp1%mod;
	}
	int l=1;while(l<=n+m)l<<=1;
	for(int i=0;i<l;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(l>>1));
	doit('s');doit('y');doit('f');
	doit('a');doit('k');
	doit('n');doit('o');doit('i');
	int as1=0;for(int j=0;j<=n-m;j++)if(as[j]>=m-k)as1++;
	printf("%d\n",as1);
}
```



##### auoj63 tree

###### Problem

对于有根树 $T_1$，可以使用以下方式构造 $T_2$：

选择一个点 $x$，对链 $(root,x)$ 上的点建一棵二叉树，使得二叉树的中序遍历等于 $(root,x)$ 依次经过的点的顺序，对于链上的一个点 $u$，对于 $u$ 的每一个不在链上的儿子，使用相同的方式递归建树，然后将建出的树的父亲设为 $u$。

给一个 $T_2$，求所有可能的 $T_1$ 中所有点深度和的最大值。

$n\leq 5000$

$2s,512MB$

###### Sol

反向操作可以看成将 $T_2$ 划分为若干二叉树，然后将一个二叉树变回链。此时链的顺序可以为二叉树的任意一个中序遍历，因此二叉树中的任意一个子树在链中是一个区间，合并两个子树时只需要知道两侧的大小，因此考虑 $dp$。

设 $dp_{i,j}$ 表示 $i$ 为根的子树中，当前根所在的二叉树大小为 $j$ 时，当前子树内的最大深度和。

设 $d_i=max(dp_{i,j})$，转移枚举根与几个儿子在二叉树上相连，再枚举儿子间在链上的顺序，有以下几种情况：

1. $dp_{u,1}=1+\sum_{v\in son_u} f_v+sz_v$
2. $dp_{u,i}=max_{v\in son_u}(dp_{v,i-1}+sz_v+1+\sum_{t\in son_u,t\neq v}(f_t+sz_t))$（连一个儿子，且 $u$ 在最后的链的上部）
3. $dp_{u,i}=max_{v\in son_u}(dp_{v,i-1}+i*(sz_u-sz_v)+\sum_{t\in son_u,t\neq v}f_t)$（连一个儿子，且 $u$ 在最后的链的下部）
4. $dp_{u,i}=max_{x,y\in son_u,x\neq y}\max_{j+k=i-1}(dp_{x,j}+dp_{y,k}+(j+1)*(sz_u-sz_x)+\sum_{t\in son_u,t\neq x,y}f_t)$（连两个儿子，此时一定一个在 $u$ 上面一个在下面）

复杂度与树上合并相同，为 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 5050
int dp[N][N],head[N],cnt,n,a,b,sz[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	vector<int> sn;
	int su=0;sz[u]=1;
	for(int i=0;i<=n;i++)dp[u][i]=-1e9;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),su+=dp[ed[i].t][0],sn.push_back(ed[i].t),sz[u]+=sz[ed[i].t];
	//choose zero
	dp[u][0]=dp[u][1]=su+sz[u];
	//choose one
	for(int i=0;i<sn.size();i++)
	for(int j=1;j<=sz[sn[i]];j++)
	dp[u][j+1]=max(dp[u][j+1],max(
	su-dp[sn[i]][0]+(j+1)*(sz[u]-sz[sn[i]])+dp[sn[i]][j],//other - u
	su-dp[sn[i]][0]+sz[u]+dp[sn[i]][j]//u - other
	));
	//choose two
	for(int i=0;i<sn.size();i++)
	for(int j=i+1;j<sn.size();j++)
	for(int k=1;k<=sz[sn[i]];k++)
	for(int l=1;l<=sz[sn[j]];l++)
	dp[u][k+l+1]=max(dp[u][k+l+1],max(
	su-dp[sn[i]][0]-dp[sn[j]][0]+(k+1)*(sz[u]-sz[sn[i]])+dp[sn[i]][k]+dp[sn[j]][l],//i - u - j
	su-dp[sn[i]][0]-dp[sn[j]][0]+(l+1)*(sz[u]-sz[sn[j]])+dp[sn[i]][k]+dp[sn[j]][l]//j - u - i
	));
	for(int i=1;i<=sz[u];i++)dp[u][0]=max(dp[u][0],dp[u][i]);
}
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)scanf("%d",&a),adde(a,i);
	dfs(1,0);
	printf("%d\n",dp[1][0]);
}
```



##### auoj64 sort

###### Problem

给一个长度为 $n$ 的正整数序列，有 $q$ 次操作：

1. 区间与一个数 `and`
2. 区间与一个数 `or`
3. 区间与一个数 `xor`
4. 区间排序

求出所有操作后的序列。

$n,q\leq 10^5,v_i<2^{32}$

$4s,512MB$

###### Sol

一次 $4$ 操作后，操作区间形成了有序段，对于一个有序段，可以用 `01-trie` 维护所有数。

对于一个段上的前三种操作，可以发现进行若干次前三种操作后，每一位的操作有四种可能的情况：不变，翻转，变成 $0$，变成 $1$。因此标记可以维护为 $\text{and}\ a\ \text{or}\ b\ \text{xor}\ c$ 的形式，可以 $O(1)$ 添加一个标记。

因此可以使用平衡树维护当前所有的有序段，以及每个有序段当前前三种操作的标记，操作可以看成区间修改。

但操作可能分裂有序段，此时因为 `01-trie` 的性质，可以直接看成 `trie` 上按照大小分裂出较小的部分。分裂有序段时分裂 `trie` 并复制标记即可。这样对于前三个操作，分裂后可以平衡树上区间修改。

对于 $4$ 操作，考虑找出所有需要合并的段并直接合并，此时需要将平衡树上的这一段标记推到 `trie` 上再合并 `trie`，需要考虑 `trie` 上标记的维护。

`trie` 下放标记时，可能有三种情况：

1. 这一位数不变，直接将标记下传到下一位即可。
2. 这一位取反，交换两个儿子并下传即可。
3. 这一位会变成一个定值，此时可以先下传剩下的标记，然后合并两个儿子。合并的方式与合并两个 `trie` 相同。

`trie` 上标记可以使用相同的方式记录，因此这部分维护标记也是 $O(1)$ 的。

复杂度 $O(n\log n+n\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
using namespace std;
#define N 505918
#define M 12021425
#define ui unsigned int
ui s1=(1ll<<32)-1;
struct sth{ui a,b,c;};
sth operator +(sth a,sth b)
{
	ui v1=b.c,v2=s1^b.a^b.b^b.c;
	b.b|=(a.b&v2)|(a.a&v1);
	b.a|=(a.a&v2)|(a.b&v1);
	b.c&=(s1^(a.a|a.b));
	b.c^=(a.c&(s1^b.a^b.b));
	return b;
}
//trie
sth lz[M];
int ch[M][2],fa[M],sz[M],ct;
int merge(int x,int y,int d);
void pushdown(int x,int d)
{
	lz[ch[x][0]]=lz[ch[x][0]]+lz[x];lz[ch[x][1]]=lz[ch[x][1]]+lz[x];
	if((lz[x].c>>d)&1)ch[x][0]^=ch[x][1]^=ch[x][0]^=ch[x][1];
	if((lz[x].b>>d)&1)ch[x][1]=merge(ch[x][0],ch[x][1],d-1),ch[x][0]=0;
	if((lz[x].a>>d)&1)ch[x][0]=merge(ch[x][0],ch[x][1],d-1),ch[x][1]=0;
	lz[x]=(sth){0,0,0};
}
void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]];}
int merge(int x,int y,int d)
{
	if(!x||!y)return x+y;
	if(d>=0)pushdown(x,d),pushdown(y,d);
	sz[x]+=sz[y];
	ch[x][0]=merge(ch[x][0],ch[y][0],d-1);
	ch[x][1]=merge(ch[x][1],ch[y][1],d-1);
	return x;
}
pair<int,int> split(int x,int d,int k)
{
	if(k<=0)return make_pair(0,x);
	if(k>=sz[x])return make_pair(x,0);
	if(d<0){++ct;sz[ct]=sz[x]-k;sz[x]=k;return make_pair(x,ct);}
	pushdown(x,d);
	int tp=++ct;
	pair<int,int> s2=split(ch[x][1],d-1,k-sz[ch[x][0]]);
	pair<int,int> s1=split(ch[x][0],d-1,k);
	ch[x][0]=s1.first,ch[x][1]=s2.first;
	ch[tp][0]=s1.second,ch[tp][1]=s2.second;
	pushup(x);pushup(tp);return make_pair(x,tp);
}
ui getkth(int x,int d,int k,ui tp)
{
	if(d<0)return tp;
	pushdown(x,d);
	if(sz[ch[x][0]]>=k)return getkth(ch[x][0],d-1,k,tp);
	else return getkth(ch[x][1],d-1,k-sz[ch[x][0]],tp|(1u<<d));
}
int n,q,a,b,c,bel[N],st[N],lb[N],rb[N];
set<int> st1;
ui v[N],d;
ui justdoit(ui x,sth y){return (x&(s1^y.a)|y.b)^y.c;}
//splay
struct Splay{
	int ch[N][2],fa[N],rt,ct;
	sth lz1[N],vl[N];
	void pushdown(int x){vl[ch[x][0]]=vl[ch[x][0]]+lz1[x];vl[ch[x][1]]=vl[ch[x][1]]+lz1[x];lz1[ch[x][0]]=lz1[ch[x][0]]+lz1[x];lz1[ch[x][1]]=lz1[ch[x][1]]+lz1[x];lz1[x]=(sth){0,0,0};}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;pushdown(f);pushdown(x);ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;}
	void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(fa[f]!=y)rotate((ch[g][1]==f)^(ch[f][1]==x)?x:f);rotate(x);}if(!y)rt=x;}
	void doit(int x)
	{
		int tp=*(--st1.upper_bound(x));if(tp==x)return;
		int s1=bel[tp];splay(s1);pushdown(s1);ct++;
		pair<int,int> v1=split(st[s1],31,x-lb[s1]);
		ch[ct][1]=ch[s1][1];fa[ch[s1][1]]=ct;ch[s1][1]=0;ch[ct][0]=s1;fa[s1]=ct;
		rb[ct]=rb[s1];rb[s1]=x-1;lb[ct]=x;st[ct]=v1.second;st[s1]=v1.first;vl[ct]=vl[s1];
		bel[x]=ct;st1.insert(x);
	}
	void modify(int l,int r,sth tp)
	{
		doit(l);doit(r+1);
		int v1=*(--st1.lower_bound(l)),v2=*(st1.lower_bound(r+1));
		v1=bel[v1];v2=bel[v2];
		splay(v1);splay(v2,v1);
		pushdown(v1);pushdown(v2);
		int t1=ch[v2][0];
		vl[t1]=vl[t1]+tp;lz1[t1]=lz1[t1]+tp;
	}
	void doit2(int x,int y)
	{
		if(!x)return;
		pushdown(x);
		lz[st[x]]=lz[st[x]]+vl[x];vl[x]=(sth){0,0,0};
		if(x!=y)st[y]=merge(st[x],st[y],31);
		doit2(ch[x][0],y);doit2(ch[x][1],y);
		ch[x][0]=ch[x][1]=0;if(x!=y)fa[x]=0;
		st1.erase(lb[x]);
	}
	void modify2(int l,int r)
	{
		doit(l);doit(r+1);
		int v1=*(--st1.lower_bound(l)),v2=*(st1.lower_bound(r+1));
		v1=bel[v1];v2=bel[v2];
		splay(v1);
		splay(v2,v1);
		pushdown(v1);pushdown(v2);
		doit2(ch[v2][0],ch[v2][0]);
		st1.insert(l);bel[l]=ch[v2][0];
		lb[ch[v2][0]]=l;rb[ch[v2][0]]=r;
	}
	void getans(int x)
	{
		if(!x)return;
		pushdown(x);getans(ch[x][0]);
		if(lb[x]&&rb[x]<=n)for(int i=lb[x];i<=rb[x];i++)printf("%u ",justdoit(getkth(st[x],31,i-lb[x]+1,0),vl[x]));
		getans(ch[x][1]);
	}
}tr;
void init()
{
	for(int i=1;i<=n;i++)
	{
		int s1=++ct;st[i+1]=s1;sz[s1]=1;
		lb[i+1]=rb[i+1]=i;
		for(int j=31;j>=0;j--)
		{
			int tp=(v[i]>>j)&1;
			ch[s1][tp]=++ct;s1=ct;sz[ct]=1;
		}
	}
	lb[n+2]=rb[n+2]=n+1;tr.ct=n+2;
	for(int i=0;i<=n+1;i++)bel[i]=i+1,st1.insert(i);
	for(int i=1;i<=n+2;i++)tr.fa[i]=i-1,tr.ch[i][1]=(i==n+2?0:i+1);
	tr.rt=1;
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%u",&v[i]);
	init();
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d%d",&a,&b,&c);
		if(a==2)scanf("%u",&d),tr.modify(b,c,(sth){s1^d,0,0});
		else if(a==1)scanf("%u",&d),tr.modify(b,c,(sth){0,d,0});
		else if(a==3)scanf("%u",&d),tr.modify(b,c,(sth){0,0,d});
		else tr.modify2(b,c);
	}
	tr.getans(tr.rt);
}
```



#### SCOI2020模拟?

##### auoj65 小B的班级

###### Problem

给一棵 $n$ 个点带边权的树，考虑如下问题：

现在在树的点上有 $m$ 个红点和 $m$ 个蓝点，你需要将红点和蓝点两两配对，使得每一对两个点的距离之和最大。

对于所有 $n^{2m}$ 种点的位置，求和最大总距离，答案模 $10^9+7$

$n,m\leq 2500$

$1s,256MB$

###### Sol

如果一条边一侧有 $i$ 个红点，$j$ 个蓝点，那么这条边最多被算 $\min(i,m-j)+\min(j,m-i)=\min(i+j,2m-i-j)$ 次。

考虑类似重心的方式，显然可以找到一个点，使得这个点的每个子树内，红点与蓝点的数量和不超过 $m$。因此考虑以这个点为根，不同子树内配对，这样一定可以配完。

从而存在一种匹配方式，使得每条边都被算 $\min(i+j,2m-i-j)$ 次。因此一种情况的总距离为所有边的这个值之和。

考虑一条边的贡献。如果一条边一侧的子树大小为 $s$，那么贡献为：

$$
w_i*(\sum_{i=0}^m\sum_{j=0}^m\min(i+j,2m-i-j)s^i(n-s)^{m-i}C_m^i*s^j(n-s)^{m-j}C_m^j)
$$

枚举 $i$，此时 $\min(i+j,2m-i-j)$ 会按照 $j$ 的大小有两种取值，对于每个 $i$ 维护 $s^j(n-s)^{m-j}C_m^j$ 和 $s^j(n-s)^{m-j}C_m^j*j$ 的前缀和即可处理出每部分的贡献。

复杂度 $O(nm)$

###### Code 

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 2505
#define mod 1000000007
int n,m,a,b,d,sz[N],head[N],cnt,c[N][N],as,as2[N],pw[N][N],f[N],suf[N],suf2[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
int solve(int s)
{
	if(n-s<s)s=n-s;
	if(as2[s])return as2[s];
	int as3=0;
	for(int i=0;i<=m;i++)f[i]=1ll*pw[s][i]*pw[n-s][m-i]%mod*c[m][i]%mod;
	suf[0]=f[0];for(int i=1;i<=m;i++)suf[i]=(suf[i-1]+f[i])%mod;
	for(int i=1;i<=m;i++)suf2[i]=(suf2[i-1]+1ll*f[i]*i)%mod;
	for(int j=0;j<=m;j++)as3=(as3+1ll*f[j]*(1ll*j*suf[m-j]%mod+suf2[m-j])%mod+2ll*f[j]*m%mod*(suf[m]-suf[m-j])-1ll*f[j]*(1ll*j*(suf[m]-suf[m-j])%mod+suf2[m]-suf2[m-j]))%mod;
	return as2[s]=(as3+mod)%mod;
}
void dfs(int u,int fa)
{
	sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),sz[u]+=sz[ed[i].t],as=(as+1ll*ed[i].v*solve(sz[ed[i].t]))%mod;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=0;i<=m;i++)c[i][i]=c[i][0]=1;
	for(int i=2;i<=m;i++)
	for(int j=1;j<i;j++)
	c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	for(int i=0;i<=n;i++)
	{
		pw[i][0]=1;
		for(int j=1;j<=m;j++)pw[i][j]=1ll*pw[i][j-1]*i%mod;
	}
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&d),adde(a,b,d);
	dfs(1,0);
	printf("%d\n",as);
}
```



##### auoj66 小B的环

###### Problem

有一个长度为 $n$ 的字符串，字符串首尾相接形成一个环。对于每一个 $k=1,\cdots,n-1$，求出是否可以从环上删去连续 $k$ 个字符，使得剩下的环没有一对相邻位置相同。

多组数据

$\sum n\leq 5\times 10^6$

$2s,512MB$

###### Sol

将环倍长，变为链上的问题。

考虑环上没有两个相邻字符相同的情况，此时不合法一定只会出现在删去字符两侧的位置，则如果剩下 $k$ 个字符的情况不存在合法方案，那么一定有 $\forall i,a_i=a_{i+k-1}$。

这相当于原串满足 $s_{1,\cdots,n}=s_{k,\cdots,n+k-1}$，可以kmp/hash对于每个 $k$ 判断。

考虑有相同字符的情况，因为不能出现相邻两个相同，因此必须删除相邻两个相同的部分。此时操作可以看成留下一个区间，区间内不能有相邻相同字符，且首尾不同。相邻相同的位置将环划分成了若干个区间，可以对于每一个区间考虑。可以发现一个区间的情况和上面类似，对于一个长度为 $l$ 的区间，里面存在留下 $k$ 个字符的方式当且仅当 $k\leq l$ 且 $s_{1,\cdots,l-k+1}\neq s_{k,\cdots,l}$。可以使用类似的方式判断。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 5000500
int n,fail[N*2],is[N],as[N],is2[N];
char s[N*2],t[N];
void solve0()
{
	for(int i=n+1;i<=n*2;i++)s[i]=s[i-n];
	for(int i=2;i<=n*2;i++)
	{
		int st=fail[i-1];
		while(st&&s[st+1]!=s[i])st=fail[st];
		if(s[st+1]==s[i])fail[i]=st+1;
		else fail[i]=0;
	}
	int st4=n*2-fail[n*2];
	if(st4>n)st4=n;
	for(int i=0;i<n;i++){int st1=i+1;if(st1%st4==0)printf("0");else printf("1");}
	printf("\n");
}
void doit(int l,int r)
{
	int le=0;
	for(int i=l;;i=(i==n?1:i+1)){t[++le]=s[i];if(i==r)break;}
	for(int i=1;i<=le;i++)is2[i]=0;
	for(int i=2;i<=le;i++)
	{
		int st=fail[i-1];
		while(st&&t[st+1]!=t[i])st=fail[st];
		if(t[st+1]==t[i])fail[i]=st+1;
		else fail[i]=0;
	}
	for(int i=le;i;i=fail[i])is2[le-i+1]=1;
	for(int i=1;i<=le;i++)as[i]|=!is2[i];
}
int main()
{
	while(~scanf("%s",s+1))
	{
		n=strlen(s+1);
		int fg=0;
		for(int i=1;i<=n;i++)is[i]=s[i]==s[i==n?1:i+1],fg|=is[i];
		if(!fg)solve0();
		else
		{
			for(int i=0;i<=n;i++)as[i]=0;
			int las=0;
			for(int i=1;i<=n;i++)if(is[i])las=i==n?1:i+1;
			for(int i=1;i<=n;i++)if(is[i])doit(las,i),las=i+1;
			for(int i=0;i<n;i++)printf("%d",as[n-i]);
			printf("\n");
		}
	}
}
```



##### auoj67 小B的农场

###### Problem

有一个 $w\times h$ 的长方形，长方形内部有 $n$ 个点。点的坐标都是整数。

你需要在长方形内部选择一个矩形，满足如下条件：

1. 矩形的边平行于坐标轴。
2. 矩形内部不存在给定点，但边界上可以存在给定点。

在此基础上，你希望得到的矩形的周长最大，输出最大周长。

$n\leq 3\times 10^5$

$2s,512MB$

###### Sol

首先可以注意到，显然一定可以选择一个长为 $w$，宽为 $1$ 的矩形，另外一个方向类似。因此答案至少是 $2\max(w,h)+2$。

从而答案大于等于 $w+h+2$，因此只需要考虑周长大于等于这个的矩形。可以发现这样的矩形满足长大于等于 $\frac {w+1}2$ 或者宽大于等于 $\frac{h+1}2$。

如果满足第一个条件，则可以发现矩形一定经过 $x=\lfloor\frac w2\rfloor$，否则矩形一定经过 $y=\lfloor\frac h2\rfloor$。

因此只需要求经过某条直线的矩形的最大周长即可。

设当前考虑的是 $x$ 方向的直线，考虑此时的最大矩形，如果决定了矩形的 $y$ 轴边界 $y_1,y_2$，则最大矩形为从直线向上找到第一个给定点作为 $x$ 的上边界，再向下找到第一个给定点作为 $x$ 的下边界。

此时可以将点分成两部分，相当于找一个区间，使得两部分点在这个区间内分别的最小高度之和加上区间长度最大。

此时有多种做法。一种做法是对 $y$ 扫描线，维护两边的单调栈，线段树找最大值。

另外一种做法是，考虑一侧找到的最小点。对于一个点，考虑它两侧第一个低于它的点，这样可以得到一个区间 $[l_i,r_i]$，则如果选择的区间是它的子区间，就可以选这个位置作为最小高度。

因此问题可以转换为，两侧各给定一些区间，每个区间有权值 $v_i$。在两侧分别选择一个区间，使得它们有交并且交长度加上两个区间权值之和最大。

考虑从左向右枚举 $l$ 和一个区间，然后相当于在另外一个方向，$l$ 在左侧的区间中选择如下两种情况之一：

1. 选择 $r$ 比当前 $r$ 小的区间，最大化 $r_i+v_i$。
2. 选择 $r$ 比当前 $r$ 大的区间，最大化 $v_i$。

那么相当于支持插入位置，求前缀或者后缀的 $\max$。使用 `set` 维护任意插入的单调栈即可。

复杂度均为 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
#include<vector>
using namespace std;
#define N 300500
int n,h,w,s[N][2],as;
struct sth{int l,r,v,f;};
bool operator <(sth a,sth b){return a.l<b.l;}
struct sta{
	set<pair<int,int> > sr;
	void init(){sr.clear();}
	int query(int x)
	{
		set<pair<int,int> >::iterator it=sr.lower_bound(make_pair(x,1e9));
		if(it==sr.begin())return -1e9;
		it--;return (*it).second;
	}
	void modify(int x,int y)
	{
		if(query(x)>=y)return;
		while(1)
		{
			set<pair<int,int> >::iterator it=sr.lower_bound(make_pair(x,0));
			if(it==sr.end()||(*it).second>y)break;
			sr.erase(*it);
		}
		sr.insert(make_pair(x,y));
	}
}t1,t2;
void solve(vector<sth> sl)
{
	t1.init();t2.init();
	for(int i=0;i<sl.size();i++)
	{
		sth s1=sl[i];
		if(s1.f)as=max(as,max(s1.v+s1.r-s1.l+t2.query(w+1-s1.r),s1.v-s1.l+t1.query(s1.r)));
		else t1.modify(s1.r,s1.r+s1.v),t2.modify(w+1-s1.r,s1.v);
	}
}
int st[N],ct,lb[N],rb[N];
vector<sth> calc_st(vector<pair<int,int> > sr,int li)
{
	sort(sr.begin(),sr.end());
	ct=0;
	for(int i=0;i<sr.size();i++)
	{
		while(ct&&sr[i].second<sr[st[ct]].second)ct--;
		lb[i]=st[ct];st[++ct]=i;
	}
	ct=0;
	for(int i=sr.size()-1;i>=0;i--)
	{
		while(ct&&sr[i].second<sr[st[ct]].second)ct--;
		rb[i]=st[ct];st[++ct]=i;
	}
	vector<sth> as;
	for(int i=1;i+1<sr.size();i++)as.push_back((sth){sr[lb[i]].first,sr[rb[i]].first,sr[i].second,0});
	for(int i=0;i+1<sr.size();i++)as.push_back((sth){sr[i].first,sr[i+1].first,li,0});
	return as;
}
void doit()
{
	vector<sth> sl;
	vector<pair<int,int> > s1,s2;
	as=max(as,h+1);
	for(int i=1;i<=n;i++)if(s[i][0]<=h/2)s1.push_back(make_pair(s[i][1],h/2-s[i][0]));
	else s2.push_back(make_pair(s[i][1],s[i][0]-h/2));
	s1.push_back(make_pair(0,0));s1.push_back(make_pair(w,0));
	s2.push_back(make_pair(0,0));s2.push_back(make_pair(w,0));
	vector<sth> f1=calc_st(s1,h/2),f2=calc_st(s2,h-h/2);
	for(int i=0;i<f1.size();i++)sl.push_back(f1[i]);
	for(int i=0;i<f2.size();i++)f2[i].f=1,sl.push_back(f2[i]);
	sort(sl.begin(),sl.end());
	solve(sl);
	for(int i=0;i<sl.size();i++)sl[i].f^=1;
	solve(sl);
}
int main()
{
	scanf("%d%d%d",&h,&w,&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]);
	doit();
	for(int i=1;i<=n;i++)swap(s[i][0],s[i][1]);
	swap(h,w);doit();
	printf("%d\n",as*2);
}
```



#### SCOI2020模拟9

##### auoj68 骨灰

###### Problem

给一个网格图，其中有 $n$ 个格子为黑色，求有多少个网格中的矩形满足

1. 矩形的长边长度是短边长度的 $2$ 倍。
2. 所有在边界上的格子均为黑色。
3. 短边长属于一个给定的集合 $S$。

$n\leq 10^6$

$2s,1024MB$

###### Sol

考虑矩形的一个角，即 `L` 型的数量。这里只考虑一个方向的 `L` 型。

对于一个黑色格子数大于 $\sqrt n$ 的行，考虑所有黑色格子，每一个黑色格子最多对应一个一条边在这行上的 `L` 型，因此这部分的数量不超过 $O(n^{1.5})$。

对于剩下的行，每一行只有 $\sqrt n$ 个黑色格子，因此每个黑格子最多向右延伸 $O(\sqrt n)$ 的长度，因此这部分也不超过 $O(n^{1.5})$。因此总的 `L` 型不超过 $O(n^{1.5})$，可以考虑枚举所有的矩形。

将每条对角线上的点拿出来考虑，预处理每个点向四个方向延伸的长度。枚举每个点，再向后枚举这个点矩形的对应顶点，用延伸长度判定是否合法，根据上面的分析复杂度为 $O(n^{1.5})$。

注意卡常。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#pragma GCC optimize("-Ofast")
using namespace std;
#define N 1060000
int n,as,xl[N],xr[N],yl[N],yr[N],ct,st[N>>1][4],st2[N>>1][4];
char st1[N];
struct pt{int x,y,id;}s[N],s2[N];
bool cmp1(pt a,pt b){return a.x==b.x?a.y<b.y:a.x<b.x;}
bool cmp2(pt a,pt b){return a.y==b.y?a.x<b.x:a.y<b.y;}
vector<pt> fu[N*3];
int main()
{
	scanf("%d%s",&n,st1+1);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i].x,&s[i].y),s[i].id=i;
	sort(s+1,s+n+1,cmp2);
	for(int i=1;i<=n;i++)s2[i]=s[i];
	for(int i=1;i<=n;i++)if(s[i].y==s[i-1].y&&s[i].x==s[i-1].x+1)xl[s[i].id]=xl[s[i-1].id]+1;
	for(int i=n;i>=1;i--)if(s[i].y==s[i+1].y&&s[i].x==s[i+1].x-1)xr[s[i].id]=xr[s[i+1].id]+1;
	sort(s+1,s+n+1,cmp1);
	for(int i=1;i<=n;i++)if(s[i].x==s[i-1].x&&s[i].y==s[i-1].y+1)yl[s[i].id]=yl[s[i-1].id]+1;
	for(int i=n;i>=1;i--)if(s[i].x==s[i+1].x&&s[i].y==s[i+1].y-1)yr[s[i].id]=yr[s[i+1].id]+1;
	for(int i=1;i<=n;i++)fu[2*s[i].x+s[i].y].push_back(s[i]);
	for(int i=1;i<=n*3;i++)if(fu[i].size()&&fu[i-1].size())
	{
		int las=0,s2=fu[i-1].size(),s1=fu[i].size();
		for(int j=0;j<s1;j++)st2[j][0]=fu[i][j].x+xr[fu[i][j].id],st2[j][1]=fu[i][j].y-yl[fu[i][j].id],st2[j][2]=fu[i][j].x,st2[j][3]=fu[i][j].y;
		for(int j=0;j<s2;j++)st[j][0]=fu[i-1][j].x,st[j][1]=fu[i-1][j].y,st[j][2]=fu[i-1][j].x-xl[fu[i-1][j].id],st[j][3]=fu[i-1][j].y+yr[fu[i-1][j].id];
		for(int j=0;j<fu[i].size();j++)
		{
			int v1=st2[j][0],v2=st2[j][1],v3=st2[j][2],v4=st2[j][3];
			while(las<s2&&st[las][0]<v3)las++;
			int tp=las;
			while(tp<s2)
			{
				if(v1<st[tp][0]||v2>st[tp][1])break;
				if(st[tp][2]<=v3&&st[tp][3]>=v4)
				if(st1[st[tp][0]-v3+1]=='1')as++;
				tp++;
			}
		}
	}
	for(int i=1;i<=n*3;i++)fu[i].clear();
	for(int i=1;i<=n;i++)s[i]=s2[i],xl[i]^=yl[i]^=xl[i]^=yl[i],xr[i]^=yr[i]^=xr[i]^=yr[i],s[i].x^=s[i].y^=s[i].x^=s[i].y;
	for(int i=1;i<=n;i++)fu[2*s[i].x+s[i].y].push_back(s[i]);
	for(int i=1;i<=n*3;i++)if(fu[i].size()&&fu[i-1].size())
	{
		int las=0,s2=fu[i-1].size(),s1=fu[i].size();
		for(int j=0;j<s1;j++)st2[j][0]=fu[i][j].x+xr[fu[i][j].id],st2[j][1]=fu[i][j].y-yl[fu[i][j].id],st2[j][2]=fu[i][j].x,st2[j][3]=fu[i][j].y;
		for(int j=0;j<s2;j++)st[j][0]=fu[i-1][j].x,st[j][1]=fu[i-1][j].y,st[j][2]=fu[i-1][j].x-xl[fu[i-1][j].id],st[j][3]=fu[i-1][j].y+yr[fu[i-1][j].id];
		for(int j=0;j<fu[i].size();j++)
		{
			int v1=st2[j][0],v2=st2[j][1],v3=st2[j][2],v4=st2[j][3];
			while(las<s2&&st[las][0]<v3)las++;
			int tp=las;
			while(tp<s2)
			{
				if(v1<st[tp][0]||v2>st[tp][1])break;
				if(st[tp][2]<=v3&&st[tp][3]>=v4)
				if(st1[st[tp][0]-v3+1]=='1')as++;
				tp++;
			}
		}
	}
	printf("%d\n",as);
}
```



##### auoj69 智子

###### Problem

给定 $k$ ，两个人在一堆 $n$ 个石子上进行博弈，两人轮流操作：

1. 先手第一次取 $1$ 个石子
2. 设对方上一次取了 $t$ 个石子，这一次当前操作的人可以取 $[1,k+t]$ 个石子

取最后一个石子的人获胜，求出在 $[l,r]$ 中有多少个 $n$ 使得先手必胜。

$l,r,k\leq 10^{14}$

$2s,1024MB$

###### Sol

设 $sg_{i,j}$ 表示还剩 $i$ 个棋子，上一个人取了 $j$ 个时的sg值，打表观察可以发现 $sg_{n,1}$ 存在循环节，且循环形如：

```
0 1 2 ... k+1
0 1
0 1 2 3
0 1 ... 7
0 1 ... 15
...
0 1 ... 2^x-1
0 1 2 ... k+1
0 1
0 1 2 3
0 1 ... 7
0 1 ... 15
...
0 1 ... 2^y-1
```

其中 $x$ 为最大的满足 $2^{x}\leq k+1$ 的数，$y$ 为最大的满足 $2^y\leq 2^{x+1}-(k+1)$ 的数。~~证明不会~~

一个循环节内先手必胜的位置为 $sg_{n-1,1}=0$ 的位置，因此只有 $\log k$ 个位置，暴力计算循环节里面每个位置出现了多少次即可

复杂度 $O(\log k)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define ll long long
vector<ll> pts;
ll k,x,y,fu;
ll solve(ll x)
{
	if(x<1)return x;
	ll as1=(x/fu)*pts.size();x%=fu;
	for(int i=0;i<pts.size();i++)if(pts[i]<=x)as1++;
	return as1;
}
int main()
{
	scanf("%lld%lld%lld",&k,&x,&y);
	fu=k+2;
	ll tp=2;while(tp<=k+1)pts.push_back(fu),fu+=tp,tp<<=1;
	pts.push_back(fu),fu+=k+2;
	ll tp2=2,tp3=tp-(k+1);
	while(tp2<=tp3)pts.push_back(fu),fu+=tp2,tp2<<=1;
	pts.push_back(fu);
	printf("%lld\n",solve(y-1)-solve(x-2));
}
```



##### auoj70 墓地

###### Problem

给定 $n$ 个物品，每种物品的数量 $k$ 相同，每种物品有重量 $w_i$，第一次选的价值 $a_i$ 和以后选择一次的价值 $b_i$。

有 $q$ 个版本，其中第 $i$ 个版本为在第 $f_i$ 个版本上修改一个物品的属性得到的，给出初始版本物品的状态以及每次修改的状态。

给定 $m$，对于每个版本求出总重量不超过 $m$ 时的最大收益。

$n,m,q\leq 3500$

$1s,1024MB$

###### Sol

考虑将版本的关系看成树，那么一次修改的物品存在的版本范围是这个版本的子树，除去子树内每一个再次修改了这个物品的子树，相当于一个 `dfs` 序区间除去若干个 `dfs` 序区间，这样得到的仍然是若干个区间。

考虑一种物品的所有修改在树上的情况，一次修改会增加一个区间，同时需要在它祖先上最近的修改这个物品处将那次修改的区间去掉当前的子树。因此每次修改只会贡献 $O(1)$ 个区间，总的区间数为 $O(n+q)$。可以 `dfs` 一次求出这些区间。

此时可以看成有若干个物品，每个物品在一段时间（`dfs` 序）区间内存在，求每个时刻的背包问题的答案。注意到背包问题容易加入一个物品，考虑线段树分治，使用单调队列/二进制分组优化加入物品时的多重背包转移即可。

复杂度 $O((n+q)m\log n)$ 或 $O((n+q)m\log n\log k)$ ~~实测第二个跑得更快~~

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 3505
struct sth{int l,r;};
vector<sth> fuc[N];
int n,m,k,l,s[N][3],q[N][4],head[N],cnt,id[N],ct,tid[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
struct sth2{int l,r,a,b,c;};
vector<sth2> fu[17][2];
long long s1[17][2][N],v2[N],as[N];
void dfs(int u,int fa)
{
	id[u]=++ct;tid[ct]=u;
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa)dfs(ed[i].t,u);
	if(fa)
	{
		int las=ct;
		while(fuc[q[u][0]].size()&&fuc[q[u][0]].back().r>=id[u])
		{
			sth tp=fuc[q[u][0]].back();fuc[q[u][0]].pop_back();
			fu[0][0].push_back((sth2){tp.r+1,las,s[q[u][0]][0],q[u][1],q[u][2]});
			las=tp.l-1;
		}
		fuc[q[u][0]].push_back((sth){id[u],ct});
		fu[0][0].push_back((sth2){id[u],las,s[q[u][0]][0],q[u][1],q[u][2]});
	}
	else
	{
		for(int i=1;i<=n;i++)
		{
			int las=ct;
			while(fuc[i].size())
			{
				sth tp=fuc[i].back();fuc[i].pop_back();
				fu[0][0].push_back((sth2){tp.r+1,las,s[i][0],s[i][1],s[i][2]});
				las=tp.l-1;
			}
			fu[0][0].push_back((sth2){1,las,s[i][0],s[i][1],s[i][2]});
		}
	}
}
void ins(int x,int y,int a,int b,int c)
{
	int fu2=l-1;
	for(int i=0;i<=k;i++)v2[i]=-1e18;
	for(int i=a;i<=k;i++)v2[i]=s1[x][y][i-a]+b;
	if(1ll*a*(l-1)>=k)for(int i=a;i<=k;i++)v2[i]=max(v2[i],v2[i-a]+c);
	else
	{
		for(int tp=1;tp<=k;tp<<=1)
		if(fu2>=tp)
		{
			fu2-=tp;
			long long s11=1ll*a*tp,s2=1ll*c*tp;
			for(int i=k;i>=s11;i--)v2[i]=max(v2[i],v2[i-s11]+s2);
		}
		long long s11=1ll*a*fu2,s2=1ll*c*fu2;
		for(int i=k;i>=s11;i--)v2[i]=max(v2[i],v2[i-s11]+s2);
	}
	for(int i=0;i<=k;i++)s1[x][y][i]=max(s1[x][y][i],v2[i]);
}
void cdq(int l,int r,int d,int s)
{
	fu[d+1][0].clear();fu[d+1][1].clear();
	for(int i=0;i<fu[d][s].size();i++)
	if((fu[d][s][i].l<=l&&fu[d][s][i].r>=r))ins(d,s,fu[d][s][i].a,fu[d][s][i].b,fu[d][s][i].c);
	if(l==r){for(int i=0;i<=k;i++)as[tid[l]]=max(as[tid[l]],s1[d][s][i]);return;}
	int mid=(l+r)>>1;
	for(int i=0;i<fu[d][s].size();i++)
	if(!(fu[d][s][i].l<=l&&fu[d][s][i].r>=r))
	{
		if(fu[d][s][i].l<=mid)fu[d+1][0].push_back(fu[d][s][i]);
		if(fu[d][s][i].r>mid)fu[d+1][1].push_back(fu[d][s][i]);
	}
	for(int i=0;i<=k;i++)s1[d+1][0][i]=s1[d+1][1][i]=s1[d][s][i];
	cdq(l,mid,d+1,0);cdq(mid+1,r,d+1,1);
}
int main()
{
	scanf("%d%d%d%d",&n,&m,&k,&l);
	for(int i=1;i<=n;i++)scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]);
	for(int i=1;i<=m;i++)scanf("%d%d%d%d",&q[i+1][3],&q[i+1][0],&q[i+1][1],&q[i+1][2]),adde(i+1,q[i+1][3]+1);
	dfs(1,0);for(int i=1;i<=k;i++)s1[0][0][i]=-1e18;
	cdq(1,m+1,0,0);
	for(int i=1;i<=m;i++)printf("%lld\n",as[i+1]);
}
```



#### NOI2020模拟六校联测1

##### auoj77 Three

###### Problem

给一个长度为 $n$ 的正整数序列 $v$，求出对于每个子区间，区间内最大的三个数的乘积的和，答案模 $10^9+7$

$n\leq 10^6$

$3s,1024MB$

###### Sol

考虑在一个第三大的数处计算这个区间的贡献，为了避免可能的问题考虑给所有数钦定一个不存在相等的顺序。

则一个数能贡献到的区间为它是第三大的区间，只需要找到它左侧比它大且最靠近它的两个数，和右侧类似的两个数就能求出这些区间。

从大到小加入所有数，加入一个数时，找出当前它左右最解决的两个数，那么它作为第三大的区间有三种情况（它和左边两个数，它和右边两个数，它和左右各一个数），且每种情况的最大的三个数都是确定的，因此分别计算贡献即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 1000005
#define mod 1000000007
multiset<int> tp;
int n,v[N],st[N],as,pr[N],nt[N];
bool cmp(int a,int b){return v[a]>v[b];}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),st[i]=i;
	sort(st+1,st+n+1,cmp);
	tp.insert(0);tp.insert(n+1);
	nt[n+1]=n+1;
	for(int i=1;i<=n;i++)
	{
		multiset<int>::iterator it=tp.lower_bound(st[i]);
		int v4=*it;nt[st[i]]=v4,pr[v4]=st[i];int v5=nt[v4],v6=nt[v5];
		it--;int v3=*it;nt[v3]=st[i],pr[st[i]]=v3;int v2=pr[v3],v1=pr[v2];
		tp.insert(st[i]);
		int v11=0;
		v11=(v11+1ll*v[v2]*v[v3]%mod*(v4-st[i])%mod*(v2-v1))%mod;
		v11=(v11+1ll*v[v3]*v[v4]%mod*(v5-v4)%mod*(v3-v2))%mod;
		v11=(v11+1ll*v[v4]*v[v5]%mod*(st[i]-v3)%mod*(v6-v5))%mod;
		as=(as+1ll*v11*v[st[i]])%mod;
	}
	printf("%d\n",as);
}
```



##### auoj78 Seat

###### Problem

有 $n$ 排座位，第 $i$ 排有 $a_i$ 个座位。

会有 $\sum a_i$ 个人进来，每个人会选择当前空着的极长段中最长的一段，并坐在这一段最靠近中间的位置。

给 $q$ 个询问，每次给一个 $k$，询问第 $k$ 个进来的人选择的段的长度。

$n\leq 10^6,a_i\leq 10^9,q\leq 10^5$

$1s,2048MB$

###### Sol

考虑所有剩余的段构成的长度可重集 $S$，则一个人进来时相当于取出 $S$ 中的一个最大元素 $x$，然后放回 $\lfloor\frac x2\rfloor,\lfloor\frac{x-1}2\rfloor$。

可以发现，对于相邻两个数 $x,x-1$ ，它们分裂之后只会产生两种不同数，且这两种数也相邻。因此一个 $x$ 只会导致出现 $O(\log v)$ 种不同的长度，从而总的长度种数不超过 $O(n\log v)$。

考虑求出每种长度出现了多少次。因为同时分裂出来的数一定小于之前的数，因此可以相同长度的一起做，这样复杂度正确。

注意到从大到小选择段，分裂产生的数也是递减的，因此维护两个队列，第一个按顺序维护原来的段，第二个按顺序维护所有分裂出来的段，每次将分裂出来的段插入第二个队列结尾即可保证顺序正确。

找最长的段时，考虑两边的队首即可。注意如果两边的长度相同，需要一起拿出来，不然如果分裂结果为 $x,x-1$，则分两次拿出来会破坏第二个队列的顺序。

复杂度 $O(n\log v)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 1050500
struct sth{int a;long long b;}v1[N],v2[N*10];
int n,x,a,b,c,m,q,l1,l2,v[N],st2[N],ct=-1,l3=1,ct2=-1,as[N];
long long qu[N],nw;
bool cmp2(int a,int b){return qu[a]<qu[b];}
void ins2(int a,long long b)
{
	if(a==0)return;
	if(ct2>=0&&v2[ct2].a==a)v2[ct2].b+=b;
	else v2[++ct2]=(sth){a,b};
}
void doit(int a,long long b)
{
	nw+=b;
	while(l3<=q&&qu[st2[l3]]<=nw)as[st2[l3]]=a,l3++;
}
int main()
{
	scanf("%d%d%d%d%d%d%d",&n,&x,&a,&b,&c,&m,&q);
	v[1]=x;for(int i=2;i<=n;i++)v[i]=(1ll*v[i-1]*v[i-1]%m*a+1ll*b*v[i-1]+c)%m+1;
	sort(v+1,v+n+1);
	for(int i=n;i>=1;i--)
	{
		if(ct>=0&&v1[ct].a==v[i])v1[ct].b++;
		else v1[++ct]=(sth){v[i],1};
	}
	for(int i=1;i<=q;i++)scanf("%lld",&qu[i]),st2[i]=i;
	sort(st2+1,st2+q+1,cmp2);
	while(l3<=q)
	{
		sth as;
		if(l1>ct)as=v2[l2++];
		else if(l2>ct2)as=v1[l1++];
		else if(v1[l1].a==v2[l2].a){as.a=v1[l1].a;as.b=v1[l1].b+v2[l2].b,l1++,l2++;}
		else if(v1[l1].a>v2[l2].a)as=v1[l1++];
		else as=v2[l2++];
		doit(as.a,as.b);
		ins2(as.a/2,as.b);
		ins2(as.a-as.a/2-1,as.b);
	}
	for(int i=1;i<=q;i++)printf("%d\n",as[i]);
}
```



##### auoj79 Minusk

###### Problem

给定 $n,k$，求 $\sum_{i=1}^ni^{-k}$，答案模 $998244353$。

$nk\leq 10^{10}$

$6s,1024MB$

###### Sol

建议参考：快速阶乘和调和级数求和，做法本质相同

复杂度 $O(\sqrt{nk}\log{nk})$，代码在写了。



#### SCOI2020模拟10

##### auoj83 菱形

###### Problem

有一个 $10^5\times 10^5$ 的网格图，现在将网格图的每个点替换为一个菱形，类似如下形状：

```
 ^    ^    ^ 
< >--< >--< >
 v    v    v
 |    |    |
 ^    ^    ^ 
< >--< >--< >
 v    v    v
 |    |    |
 ^    ^    ^ 
< >--< >--< >
 v    v    v
```

其中每个顶点都对应一个点，所有边（网格边和菱形边）边权都为 $1$。$q$ 次询问，每次给出左上角菱形中的一个点 $s$ 和另外一个点 $t$，求 $s$ 到 $t$ 的最短路长度以及最短路条数。答案模 $998244353$。

$T,n,m\leq 10^5$

$1s,1024MB$

###### Sol

考虑将菱形看成点时最短路在网格上的情况，显然最短路不可能出现向上和向左的边，因为这种情况将路径缩回来一定严格更短。因此最短路在网格上一定是向下和向右。

再考虑加入菱形之后的情况，考虑中间的每一步。如果这一步和上一步方向不同，则只需要 $2$ 的时间，否则需要 $3$ 的时间同时有两种方案。

那么可以发现中间部分一定是不同方向的移动越多越好。因此可以看成将向下和向右排成序列，使得数量较少的一种没有两个相邻。显然方案是一个组合数。

但还有开头结尾的情况，为了避免讨论可以枚举从开头菱形的哪个点出发，到结尾菱形的哪个点结束，对每种情况求答案并合并。

复杂度 $O(T+n+m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105000
#define mod 998244353
int T,a,b,c,d,fr[N],ifr[N],fg1,fg2;
struct sth{int a,b;};
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
sth operator +(sth a,sth b){if(a.a<b.a)return a;if(a.a>b.a)return b;a.b=(a.b+b.b)%mod;return a;}
sth solve(int s,int t)
{
	int v1=b,v2=c,c1=0,tp21=1;
	if(s)v1--;else v2--;
	if(t)v1--;else v2--;
	if((s^a^1)&1)
	c1++,tp21*=fg1+1;
	if((t^d^1)&1)
	c1++,tp21*=fg2+1;
	if(v1<0||v2<0)return (sth){1000000000,0};
	if(v1<=v2&&v1+1<=c)
	{
		int tp=v1+1,tp2=c;
		int d1=tp2-tp+v1,d2=v1;
		int as1=1ll*tp21*fr[d1]*ifr[d2]%mod*ifr[d1-d2]%mod;
		as1=1ll*as1*pw(2,tp2-tp)%mod;
		c1+=2*(b+c-1)+(tp2-tp)+1;
		return (sth){c1,as1};
	}
	else
	{
		int tp=v2+1,tp2=b;
		int d1=tp2-tp+v2,d2=v2;
		int as1=1ll*tp21*fr[d1]*ifr[d2]%mod*ifr[d1-d2]%mod;
		as1=1ll*as1*pw(2,tp2-tp)%mod;
		c1+=2*(b+c-1)+(tp2-tp)+1;
		return (sth){c1,as1};
	}
}
int main()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=102000;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%d%d",&a,&b,&c,&d);
		if(b==0&&c==0)
		{
			int as=(a-d+4)%4;
			if(as==3)as=1;printf("%d %d\n",as,as==2?2:1);continue;
		}
		if(b==0&&c==1)
		{
			int as=1,ct=1;
			if(a==0||a==2)as++;
			if(a==3)ct*=2,as+=2;
			if(d==0||d==2)as++;
			if(d==1)ct*=2,as+=2;
			printf("%d %d\n",as,ct);
			continue;
		}
		if(b==1&&c==0)
		{
			int as=1,ct=1;
			if(a==1||a==3)as++;
			if(a==2)ct*=2,as+=2;
			if(d==1||d==3)as++;
			if(d==0)ct*=2,as+=2;
			printf("%d %d\n",as,ct);
			continue;
		}
		int c1=0;fg1=fg2=0;
		if(a==3)a=0,c1++,fg1=1;
		if(a==2)a=1,c1++,fg1=1;
		if(d==0)d=3,c1++,fg2=1;
		if(d==1)d=2,c1++,fg2=1;
		sth as=solve(0,0)+solve(0,1)+solve(1,0)+solve(1,1);
		printf("%d %d\n",as.a+c1,as.b);
	}
}
```



##### auoj84 正方形

###### Problem

给一个 $n\times m$ 的网格图，每个格子有一个 $\in\{0,1\}$ 的权值，支持 $q$ 次操作：

1. 翻转一个格子
2. 给一个矩形，求矩形内最大的全 $0$ 矩形的边长

$m\leq n,nm\leq 4\times 10^6,q\leq 2000$

$2s,1024MB$

###### Sol

对 $n$ 这一维建线段树，考虑需要维护的信息：

因为需要合并来自两部分的正方形，因此考虑记录当前部分中以上边界的每一个位置为左上角的最大正方形边长，以及以下边界的每一个位置为左下角的最大正方形边长。

合并时考虑枚举左上角所在的列并计算每个左上角的最大边长。显然如果 $l$ 开始的正方形可以到 $r$，则 $l+1$ 开始的正方形一定可以到 $r$。因此可以维护合法正方形的双指针，只需要询问两边分别在当前区间内的 $\min$，在双指针的过程中维护单调队列即可。单次合并复杂度 $O(m)$。

同时记录区间内，以每一列作为左边界的最大正方形边长即可维护答案，这个信息也可以在上面的合并中得到。

复杂度 $O(mq\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 4050000
#define M 2050
int n,m,q,a,b,c,d,e,l[N*2],r[N*2],as[N*2],l1[M],r1[M],as1[M],st[N],ct;
struct que{
	int s[M],lb,rb,v[M];
	void init(){lb=rb=0;s[0]=1e9;}
	int getmn(){return s[lb];}
	void doitl(int x){while(lb<rb&&v[lb]<x)lb++;}
	void doitr(int x,int y){while(lb<=rb&&s[rb]>x)rb--;s[++rb]=x;v[rb]=y;}
}v1,v2;
struct fuc{int len,lb;};
fuc pushup(fuc a,fuc b,fuc c)
{
	for(int i=1;i<=m;i++)l1[i]=l[a.lb+i]+l[b.lb+i]*(l[a.lb+i]>=a.len),r1[i]=r[b.lb+i]+r[a.lb+i]*(r[b.lb+i]>=b.len),as1[i]=max(as[a.lb+i],as[b.lb+i]);
	v1.init();v2.init();
	int rb=1;v1.doitr(r[a.lb+1],1);v2.doitr(l[b.lb+1],1);
	for(int i=1;i<=m;i++)
	{
		v1.doitl(i);v2.doitl(i);
		while(rb<m&&min(r[a.lb+rb+1],v1.getmn())+min(l[b.lb+rb+1],v2.getmn())>=rb-i+2)
		rb++,v1.doitr(r[a.lb+rb],rb),v2.doitr(l[b.lb+rb],rb);
		as1[i]=max(as1[i],rb-i+1);
	}
	c.len=a.len+b.len;
	for(int i=1;i<=m;i++)l[c.lb+i]=l1[i],r[c.lb+i]=r1[i],as[c.lb+i]=as1[i];
	return c;
}
struct segt{
	struct node{int l,r;fuc tp;}e[N*4];
	void doit(int x,int y){for(int i=1;i<=m;i++)as[e[x].tp.lb+i]=l[e[x].tp.lb+i]=r[e[x].tp.lb+i]=st[(y-1)*m+i];}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].tp.lb=ct;ct+=m;
		if(l==r){e[x].tp.len=1;doit(x,l);return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		e[x].tp=pushup(e[x<<1].tp,e[x<<1|1].tp,e[x].tp);
	}
	void modify(int x,int v)
	{
		if(e[x].l==e[x].r){doit(x,v);return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=v)modify(x<<1,v);else modify(x<<1|1,v);
		e[x].tp=pushup(e[x<<1].tp,e[x<<1|1].tp,e[x].tp);
	}
	fuc query(int x,int l,int r,fuc st)
	{
		if(e[x].l==l&&e[x].r==r){st=pushup(st,e[x].tp,st);return st;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r,st);
		else if(mid<l)return query(x<<1|1,l,r,st);
		else
		{
			st=query(x<<1,l,mid,st);
			return query(x<<1|1,mid+1,r,st);
		}
	}
}tr;
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n*m;i++)scanf("%d",&st[i]);
	tr.build(1,1,n);
	while(q--)
	{
		scanf("%d",&a);
		if(a==0)
		{
			scanf("%d%d",&b,&c);
			st[b*m-m+c]^=1;
			tr.modify(1,b);
		}
		else
		{
			scanf("%d%d%d%d",&b,&c,&d,&e);
			fuc tp;tp.lb=ct;tp.len=0;for(int i=1;i<=m;i++)l[tp.lb+i]=r[tp.lb+i]=as[tp.lb+i]=0;
			tp=tr.query(1,b,d,tp);
			int as1=0;
			for(int i=c;i<=e;i++)as1=max(as1,min(as[i+tp.lb],e-i+1));
			printf("%d\n",as1);
		}
	}
}
```



##### auoj85 最小生成树

###### Problem

给一个 $n$ 个点的完全图，每条边有 $p_i$ 的概率边权为 $i$，边权有 $1,2,\cdots,k$ 这几种，有 $1-\sum_{i=1}^k p_i$ 的概率这条边不存在。

对于每一个 $v$，求这张图连通且最小生成树的边权和为 $v$ 的概率。

$n\leq 40,k\leq 4$

$1s,1024MB$

###### Sol

求最小生成树的过程可以看成按照边权从小到大考虑每条边，依次尝试加入，因此可以考虑按照权值从小到大 $dp$。

设 $dp_{n,s,k}$ 表示只考虑边权不超过 $k$ 的边，$n$ 个点的图连通且最小生成树边权和为 $s$ 的概率。

考虑加入边权为 $k+1$ 的边。一种暴力方式是枚举加入后的连通块由哪些加入前（只考虑边权小于等于 $k$ 的边）组成，然后相当于求这些连通块被连通的概率。这时的一种做法是容斥，枚举 $1$ 所在的连通块减去这部分。

然后考虑将这部分放入 $dp$，首先考虑合并连通块，设 $f_{i,j}$ 表示将若干个 $k$ 时的连通块合并，认为连接中间部分的边边权都是 $k+1$，此时 $i$ 个点的图最小生成树为 $j$ 的所有情况的概率和。

令 $p=\sum_{i>k}p_i+(1-\sum p_i)$，考虑加入一个连通块，枚举 $1$ 号点之前的连通块大小，有：

$$
f_{n,j}=\sum_{l=1}^{n-1}\sum_{s=0}^{j-k-1} C_{n-1}^{l-1}p^{l(n-l)}dp_{l,s,k}f_{n-l,j-s-k-1}
$$

然后再考虑容斥，减去不连通的情况，考虑枚举 $1$ 号点实际的连通块大小。令 $q=\sum_{i>k+1}p_i+(1-\sum p_i)$，有：

$$
dp_{n,j,k+1}=f_{n,j}-\sum_{l=1}^{n-1}\sum_{s=0}^{j-k-1} C_{n-1}^{l-1}q^{l(n-l)}dp_{l,s,k+1}f_{n-l,j-s-k-1}
$$

可以发现对所有情况求和容斥的结果和容斥求和后的结果相同，因此这样是对的。

复杂度 $O(n^4k^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 1000000007
int n,k,v[5],dp[5][41][161],c[41][41],f[5][41][161];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=0;i<=k;i++)scanf("%d",&v[i]);
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	dp[0][1][0]=1;
	for(int i=1;i<=k;i++)
	{
		int s1=v[i],s2=v[0];
		for(int t=i;t<=k;t++)s2+=v[t];
		int tp3=1ll*s2*570000004%mod,tp2=1ll*s1*570000004%mod;
		for(int j=1;j<=n;j++)
		{
			for(int l=0;l<=(j-1)*(i-1);l++)dp[i][j][l]=dp[i-1][j][l];
			for(int l=1;l<=j;l++)
			for(int s=0;s<=(l-1)*i;s++)
			for(int t=0;t<=(j-l-1)*(i-1);t++)
			dp[i][j][s+t+i]=(dp[i][j][s+t+i]+1ll*c[j-1][j-l-1]*dp[i][l][s]%mod*dp[i-1][j-l][t]%mod*pw(tp3,l*(j-l)))%mod;
		}
		for(int j=1;j<=n;j++)
		for(int l=0;l<=(j-1)*i;l++)
		f[i][j][l]=dp[i][j][l];
		for(int j=1;j<=n;j++)
		for(int l=1;l<=j;l++)
		for(int s=0;s<=(l-1)*i;s++)
		for(int t=0;t<=(j-l-1)*i;t++)
		dp[i][j][s+t+i]=(dp[i][j][s+t+i]-1ll*c[j-1][l-1]*dp[i][l][s]%mod*f[i][j-l][t]%mod*pw(mod+tp3-tp2,l*(j-l))%mod+mod)%mod;
	}
	for(int i=n-1;i<=(n-1)*k;i++)printf("%d ",dp[k][n][i]);
}
```



#### NOI2020模拟六校联测2

##### auoj86 lowbit

###### Problem

定义 $lowbit_p(x)$ 为 $p$ 进制下 $x$ 最低的有值的一位的位权。

给定 $p$。对于一个非负整数 $x$，一次操作有 $\frac ab$ 的概率将其变为 $x+lowbit_p(x)$，$1-\frac a b$ 的概率将其变为 $x-lowbit_p(x)$。其中 $a,b$ 给定。

定义 $f(x)$ 表示 $x$ 期望操作多少次变为 $0$。给定 $l,r$，求 $\sum_{i=l}^r f(i)$，答案模 $998244353$

$l,r\leq 10^{18},p\leq 10^5$

$1s,512MB$

###### Sol

首先考虑一位的情况，相当于有一个数，每次操作有一个概率 $+1$，剩下的概率 $-1$，到 $0$ 或 $p$ 时终止，求停止时在 $p$ 的概率以及停止时的期望操作次数。

设 $dp_{i}$ 为 $i$ 出发，在 $p$ 停止的概率，显然有如下转移：

$$
dp_0=0,dp_p=1\\
dp_i=dp_{i+1}*\frac ab+dp_{i-1}*(1-\frac ab)
$$

一个方程可以看成用 $dp_i,dp_{i-1}$ 推出 $dp_{i+1}$。设 $dp_1=x$，那么依次考虑 $dp_{1},...,dp_{p-2}$ 的方程，则可以用 $x$ 表示出 $dp_2,...,dp_{p-1}$，最后用 $dp_{p-1}$ 的方程可以解出 $x$。这样即可求出 $dp$。

对于期望可以写出类似的 $dp$，可以使用 $x$ 的一次函数维护所有的期望，进而求出解。这样即可 $O(p)$ 求出一位上的情况。

考虑对一个数的操作，可以看成找到当前有值的最低位，在这一位上进行一位的操作，这一位上的结果一定为向下一位进一位或者不进位。

考虑原问题，将 $[a,b]$ 按位分成若干段，其中每一段的数在 $p$ 进制下前若干位全部相同，某一位上为一段前缀或后缀，这一位之后的部分可以任意填。

对于低位任意填的部分，考虑预处理算出考虑 $[0,p^k-1]$ 中的所有数操作到低 $k$ 位都是 $0$ 为止，期望有多少个数最后进位到 $p^k$ 以及这些数的期望时间之和。

考虑从小到大用上一个 $k$ 的值求下一个 $k$ 的值，则这一位可以看成独立于下面选择，因此可以单独考虑这一位，$O(p)$ 求出下一位。这部分复杂度 $O(n\log_p n)$。某一位上为前缀/后缀的一位类似。

考虑接下来的每一位，因为这一位只有一种取值，可以用之前的信息 $O(1)$ 算出向下一位进位的期望数量和这一位上对答案的贡献。

考虑处理了 $\log_p n$ 位后，可以发现接下来高位都是 $0$，因此后面每一次的转移固定，可以 $O(1)$ 求出后面的期望步数。

复杂度 $O(p\log_p n+\log_p^2 n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define mod 998244353
#define ll long long
int f[N],g[N],a,b,k,p,s[N][2],t[N],st,fu,fu2,vl[N];
ll l,r;
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
void pre()
{
	t[1]=1;
	for(int i=2;i<k;i++)
	{
		//f_i=p*f_{i+1}+(1-p)f_{i-1}
		//f_{i+1}=1/p(f_i-(1-p)f_{i-1})
		t[i]=1ll*(t[i-1]+mod+1ll*(p-1)*t[i-2])%mod*pw(p,mod-2)%mod;
	}
	//f_{k-1}=p+(1-p)f_{k-2}
	int fu1=(t[k-1]+1ll*(p-1)*t[k-2])%mod;
	fu1=1ll*p*pw(fu1,mod-2)%mod;
	for(int i=1;i<k;i++)f[i]=1ll*t[i]*fu1%mod;f[k]=1;
	s[1][0]=1;
	for(int i=2;i<k;i++)
	{
		//g_i=p*g_{i+1}+(1-p)g_{i-1}+1
		//g_{i+1}=1/p(f_i-(1-p)f_{i-1}-1)
		s[i][0]=1ll*(s[i-1][0]+mod+1ll*(p-1)*s[i-2][0])%mod*pw(p,mod-2)%mod;
		s[i][1]=1ll*(s[i-1][1]+mod+1ll*(p-1)*s[i-2][1]+mod-1)%mod*pw(p,mod-2)%mod;
	}
	//g_{k-1}=(1-p)g_{k-2}+1
	int fu3=(s[k-1][0]+1ll*(p-1)*s[k-2][0])%mod,fu2=(1+1ll*(mod+1-p)*s[k-2][1]+1ll*(mod-1)*s[k-1][1])%mod;
	fu3=1ll*fu2*pw(fu3,mod-2)%mod;
	for(int i=1;i<k;i++)g[i]=(1ll*s[i][0]*fu3+s[i][1])%mod;
}
int solve2(ll r,ll tp,ll l,ll lg)
{
	int v0=0,v1=0,as=0;
	for(int i=0;i<=tp;i++)
	v1=(v1+1ll*fu2*f[i]+1ll*fu*f[i+1])%mod,v0=(v0+1ll*fu2*(mod+1-f[i])%mod+1ll*fu*(mod+1-f[i+1])%mod)%mod,as=(as+1ll*fu2*g[i]+1ll*fu*g[i+1])%mod;
	for(int i=l;i<=lg;i++)
	{
		int t=r%k;r/=k;
		int r1=(1ll*v1*f[t+1]+1ll*v0*f[t])%mod,r0=(1ll*v1*(mod+1-f[t+1])+1ll*v0*(mod+1-f[t]))%mod;
		as=(as+1ll*v1*g[t+1]+1ll*v0*g[t])%mod;
		v0=r0,v1=r1;
	}
	st=(st+v1)%mod;return as;
}
int solve(ll r)
{
	ll lg=1,tp=r;
	while(tp)tp/=k,lg++;
	fu=0;fu2=1;st=0;int as=0,ct=0,fg=0;
	while(r)
	{
		as=(as+solve2(r/k,r%k-1,ct,lg))%mod,ct++,r/=k;
		int v1=st,v2=f[1],v3=g[1];
		as=(as+1ll*v1*v3%mod*pw(mod+1-v2,mod-2))%mod;
		st=0;
		vl[0]=0;for(int i=1;i<=k;i++)vl[i]=fu;
		for(int i=0;i<k;i++)vl[i]=(vl[i]+fu2)%mod;
		ll su=0;
		for(int i=0;i<=k;i++)su=(su+1ll*vl[i]*g[i])%mod;
		as=(as+r%mod*su)%mod;
		fu=0;fu2=0;
		for(int i=0;i<=k;i++)fu=(fu+1ll*f[i]*vl[i])%mod;
		for(int i=0;i<=k;i++)fu2=(fu2+1ll*(mod+1-f[i])*vl[i])%mod;
	}
	return as;
}
int main()
{
	scanf("%d%d%d%lld%lld",&k,&a,&b,&l,&r);
	p=1ll*a*pw(b,mod-2)%mod;
	pre();
	int v1=solve(l),v2=solve(r+1);
	printf("%d\n",(v2-v1+mod)%mod);
}
```



##### auoj87 sequence

###### Problem

给定正整数 $n,p,q,k$，称一个正整数序列 $a_{1,\cdots,m}$ 是好的当且仅当它满足如下条件：

1. $a_i\geq a_{i+1}*\frac p q$
2. $\sum a_i=n$

求在所有好的序列中，$\sum a_i*i^k$ 的最大值，输出这个最大值模 $10^9+7$ 的结果。

多组数据，$p,q$ 在一个范围内随机生成

$n\leq 10^9,k\leq 10^6,T\leq 10$

$2s,512MB$

###### Sol

考虑一种贪心构造方式，首先让长度尽可能大，然后让最后一位尽量大，在此基础上让倒数第二位尽量大，以此类推。

设这样构造的序列为 $a_1,\cdots,a_m$，则显然所有好的序列长度不超过 $m$，如果看成序列末尾可以填 $0$，则任意一个好的序列都可以表示为长度为 $m$ 的非负整数序列 $b_1,\cdots,b_m$。

上述贪心为让最后的位尽量大，可以发现这样的选择会最大化每一个后缀和，即：

对于一个好的序列 $b_{1,\cdots,m}$，对于每个 $x$ 都有 $\sum_{i=x}^ma_i\geq\sum_{i=x}^mb_i$。

证明：考虑从大到小对 $x$ 归纳。$x=m$ 由贪心显然成立。如果对于一个 $x$ 成立，但对于 $x-1$ 不成立，则存在一种好的方案使得 $\sum_{i=x}^ma_i\geq\sum_{i=x}^mb_i$，但变成 $x-1$ 后变为小于关系。因此一定有 $a_{x-1}<b_{x-1}$。但这说明如果这一位取 $b_{x-1}$，则前面存在一种和不超过 $n-\sum_{i=x-1}^mb_i$ 的方案，而 $n-\sum_{i=x-1}^mb_i< n-\sum_{i=x-1}^ma_i$，因此在 $a$ 中将这一位变为 $b_{x-1}$，存在合法方案，这与 $a$ 的选择方式矛盾。

因此上述性质成立。因为 $i^k$ 对于 $i$ 单调不降，因此这样选择得到的权值一定最大。从而只需要使用贪心方法求一组方案即可。

对于 $\frac p q\leq 1$ 的情况，显然最优解为 $a_1=...=a_n=1$，答案为 $\sum_{i=1}^ni^k$，这是一个幂和的形式，可以插值处理。为了求出 $1,\cdots,k+1$ 位置的值，需要求出这些位置的 $k$ 次幂。这里可以对 $i^k$ 线性筛。复杂度为 $O(k)$。

考虑剩下的情况，此时一定有 $a_i>a_{i-1}$，从而项数不超过 $O(\sqrt n)$。

对于 $\frac p q>1.1$ 的部分，项数不多（不超过 $300$），此时可以从后往前二分贪心每一项的值，每次判定时可以尽量小地向前填，判断和是否小于等于 $n$。这样的复杂度为 $O(l^2\log n)$，其中 $l$ 为长度。

对于剩下的情况，此时相邻两项中某一项在原基础上 $+1$ 对和的影响差距很小，可以发现如果调整了一项的值，那么它前面的一些项大概率不能再增加，可以二分下一个可以增加的位置。复杂度 O(能过)

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1005000
#define mod 1000000007
int t,n,k,p,q;
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
struct leq1{
	int f[N],su[N],fr[N],ifr[N],ch[N],pr[N],ct,s1[N],s2[N];
	void solve()
	{
		ct=0;for(int i=0;i<=k+3;i++)ch[i]=0;
		fr[0]=ifr[0]=1;
		for(int i=1;i<=k+3;i++)fr[i]=1ll*fr[i-1]*i%mod;
		ifr[k+3]=pw(fr[k+3],mod-2);
		for(int i=k+2;i>=1;i--)ifr[i]=1ll*ifr[i+1]*(i+1)%mod;
		f[1]=1;
		for(int i=2;i<=k+3;i++)
		{
			if(!ch[i]){pr[++ct]=i;f[i]=pw(i,k);}
			for(int j=1;j<=ct&&1ll*i*pr[j]<=k+3;j++)
			{
				f[i*pr[j]]=1ll*f[i]*f[pr[j]]%mod;
				ch[i*pr[j]]=1;
				if(i%pr[j]==0)continue;
			}
		}
		for(int i=2;i<=k+3;i++)f[i]=(f[i]+f[i-1])%mod;
		int as=0;
		s1[0]=s2[k+4]=1;
		for(int i=1;i<=k+4;i++)s1[i]=1ll*s1[i-1]*(n-i+mod)%mod;
		for(int i=k+3;i>=0;i--)s2[i]=1ll*s2[i+1]*(n-i+mod)%mod;
		for(int i=1;i<=k+3;i++)as=(as+1ll*f[i]*s1[i-1]%mod*s2[i+1]%mod*ifr[i-1]%mod*ifr[(k+3)-i]%mod*(((k+3)^i)&1?mod-1:1))%mod;
		printf("%d\n",as);
	}
}s1;
struct geq11{
	int as[N];
	bool check(int l)
	{
		long long las=1;
		long long as=1;
		for(int i=2;i<=l;i++)
		{
			las=(1ll*las*p+q-1)/q;
			as+=las;
			if(as>n)return 0;
		}
		return as<=n;
	}
	int getsu(int a,int l)
	{
		long long las=a;
		long long as=a;
		for(int i=2;i<=l;i++)
		{
			las=(1ll*las*p+q-1)/q;
			as+=las;
			if(as>n)return 1e9+10;
		}
		return as;
	}
	void solve()
	{
		int lb=1,rb=100000,as2=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(check(mid))as2=mid,lb=mid+1;
			else rb=mid-1;
		}
		int las=0,su=0;
		for(int i=as2;i>=1;i--)
		{
			int tp=(1ll*las*p+q-1)/q;
			if(tp==0)tp=1;
			int lb=tp,rb=n-las;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(su+getsu(mid,i)<=n)as[i]=mid,lb=mid+1;
				else rb=mid-1;
			}
			las=as[i];su+=as[i];
		}
		int as1=0;
		for(int i=1;i<=as2;i++)as1=(as1+1ll*as[i]*pw(i,k))%mod;
		printf("%d\n",as1);
	}
}s3;
struct thispart{
	int as[N],su2[N];
	bool check(int l)
	{
		long long las=1;
		long long as=1;
		for(int i=2;i<=l;i++)
		{
			las=(1ll*las*p+q-1)/q;
			as+=las;
			if(as>n)return 0;
		}
		return as<=n;
	}
	int getsu(int a,int l)
	{
		long long las=a;
		long long as=a;
		for(int i=2;i<=l;i++)
		{
			las=(1ll*las*p+q-1)/q;
			as+=las;
			if(as>n)return 1e9+10;
		}
		return as;
	}
	void solve()
	{
		int lb=1,rb=100000,as2=1;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(check(mid))as2=mid,lb=mid+1;
			else rb=mid-1;
		}
		int fu=as2;
		su2[as2+1]=0;as[as2+1]=0;
		for(int i=as2;i>=1;i--)
		{
			as[i]=(1ll*as[i+1]*p+q-1)/q;
			if(!as[i])as[i]=1;
			su2[i]=su2[i+1]+as[i];
		}
		while(su2[1]<=n)
		{
			int lb=1,rb=fu,as3=0;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(su2[mid+1]+getsu(as[mid]+1,mid)<=n)as3=mid,lb=mid+1;
				else rb=mid-1;
			}
			if(!as3)break;
			as[as3]++;
			for(int i=as3-1;i>=1;i--)as[i]=(1ll*as[i+1]*p+q-1)/q;
			for(int i=as3;i>=1;i--)su2[i]=su2[i+1]+as[i];
		}
		int as1=0;
		for(int i=1;i<=as2;i++)as1=(as1+1ll*as[i]*pw(i,k))%mod;
		printf("%d\n",as1);
	}
}s4;
int main()
{
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d%d%d%d",&n,&k,&p,&q);
		if(p<=q)s1.solve();
		else if(1.0*p/q>=1.1)s3.solve();
		else s4.solve();
	}
}
```



##### auoj88 classroom

###### Problem

教学楼由 $t$ 层组成，每一层的结构完全相同，一层的结构为一棵 $n$ 个点的树，经过一条边时间为 $1$。

有一些点上有通向相邻层的点的边，每一层中有这样的边的点的集合是相同的，为集合 $S$。如果你在一个有可以通向相邻层的边的点，你可以用 $1$ 的时间移动到相邻层上相同的点。

多组询问，每次给定教学楼中的 $k$ 个点，你需要在 $k$ 个点中选择一个点，使得 $k$ 个点到这个点的距离和最小，求出最小距离和。

$n\leq 2\times 10^5,\sum k\leq 10^6$

$5s,1024MB$

###### Sol

考虑两个点之间的距离。如果两个点在同一层，则最短路径为直接在这一层上走。否则可以发现最短路径为走到一个 $S$ 中的点，然后走到需要的层，再走过去。

考虑中间部分的距离，设起点终点在树上为 $s,t$，则相当于在 $S$ 中找一个点 $x$，最小化 $dis(s,x)+dis(x,t)$。可以发现这个值等于 $s$ 到 $t$ 的距离加上 $S$ 中点到路径 $(s,t)$ 的最小距离的两倍。

因此考虑对于每个点求出它到最近的 $S$ 中点的距离 $d_i$，则这部分代价为路径上最小的 $d_i$ 的 $2$ 倍。

此时选一个点的总距离可以看成三部分：

1. 所有点与它的层数的差之和
2. 所有点在树上与树上它的距离之和
3. 所有与它不在同一层的点在树上和它的路径间的 $f_i$ 最小值的和的 $2$ 倍。

第一部分可以对层排序，然后扫一遍即可求出。第二部分可以将所有点在树上建虚树，虚树上dfs容易求出这部分答案。

考虑第三部分，$d_i$ 容易dfs求出。考虑先无视不同层的限制求一遍第三类贡献，然后每一层内的点再分别做一次减去额外的贡献，即可得到第三部分的贡献。

则只需要解决如下问题：给定点集 $T$，对于每一个 $T$ 中的点 $x$，求和 $T$ 中每个点到它的路径上的最小 $d_i$。

考虑按照点权建Kruskal重构树，则 $(s,t)$ 路径上最小 $d_i$ 就是两个点在重构树上的LCA处的权值。那么将所有询问点在重构树上建虚树，虚树上dfs即可求出贡献。

复杂度 $O((n+m)\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 405918
#define ll long long
int n,q,k,a,b,vl[N],s[N][2],head[N],cnt,fa[N],f1[N],f[N][19],dep[N],id[N],ct,ct2,fid[N],tid[N],st[N],rb,sz[N],dep2[N],f2[N][19],id2[N],ct3,v1[N];
ll dp1[N],dp2[N],as[N*3];
vector<pair<int,int> > tp;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs01(int u,int fa){id2[u]=++ct3;dep2[u]=dep2[fa]+1;f2[u][0]=fa;for(int i=1;i<=18;i++)f2[u][i]=f2[f2[u][i-1]][i-1];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs01(ed[i].t,u),vl[u]=min(vl[u],vl[ed[i].t]+1);}
void dfs02(int u,int fa){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)vl[ed[i].t]=min(vl[ed[i].t],vl[u]+1),dfs02(ed[i].t,u);}
struct edg{int f,t,v;friend bool operator <(edg a,edg b){return a.v>b.v;}}e[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void dfs1(int u,int fa){id[u]=++ct;dep[u]=dep[fa]+1;f[u][0]=fa;for(int i=1;i<=18;i++)f[u][i]=f[f[u][i-1]][i-1];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u);}
int LCA(int x,int y,int dep[],int f[][19]){if(dep[x]<dep[y])x^=y^=x^=y;int tp=dep[x]-dep[y];for(int i=18;i>=0;i--)if((tp>>i)&1)x=f[x][i];if(x==y)return x;for(int i=18;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
bool cmp(int a,int b){return id[a]<id[b];}
bool cmp2(int a,int b){return id2[a]<id2[b];}
vector<pair<int,int> > solve(vector<int> p,int dep[],int f[][19],int tp)
{
	vector<pair<int,int> > as;
	sort(p.begin(),p.end(),cmp);
	if(tp)sort(p.begin(),p.end(),cmp2);
	ct2=0;rb=0;
	for(int i=0;i<p.size();i++)fid[p[i]]=i+1,tid[i+1]=p[i],ct2++;
	for(int i=0;i<p.size();i++)
	{
		while(rb>1&&dep[st[rb-1]]>=dep[LCA(st[rb],p[i],dep,f)])as.push_back(make_pair(fid[st[rb]],fid[st[rb-1]])),rb--;
		int l=LCA(st[rb],p[i],dep,f);
		if(l!=st[rb])
		{
			int tp=fid[l];if(!tp)fid[l]=++ct2,tid[ct2]=l;
			as.push_back(make_pair(fid[st[rb]],fid[l])),st[rb]=l;
		}
		st[++rb]=p[i];
	}
	while(rb>1)as.push_back(make_pair(fid[st[rb]],fid[st[rb-1]])),rb--;
	return as;
}
void dfs21(int u,int fa){sz[u]=v1[tid[u]];dp2[u]=0;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs21(ed[i].t,u),sz[u]+=sz[ed[i].t],dp2[u]+=dp2[ed[i].t]+1ll*sz[ed[i].t]*(dep2[tid[ed[i].t]]-dep2[tid[u]]);}
void dfs22(int u,int fa){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dp1[ed[i].t]=dp1[u]-1ll*sz[ed[i].t]*(vl[tid[u]]-vl[tid[ed[i].t]]),dp2[ed[i].t]=dp2[u]-1ll*(2*sz[ed[i].t]-k)*(dep2[tid[ed[i].t]]-dep2[tid[u]]),dfs22(ed[i].t,u);}
void doit(vector<int> p,int dep[],int f[][19],int tp)
{
	vector<pair<int,int> > e1=solve(p,dep,f,tp);
	for(int i=1;i<=ct2;i++)head[i]=0;cnt=0;
	for(int i=0;i<ct2-1;i++)adde(e1[i].first,e1[i].second);
	int rt=1;for(int i=1;i<=ct2;i++)rt=dep[tid[i]]<dep[tid[rt]]?i:rt;
	dfs21(rt,0);dp1[rt]=1ll*vl[tid[rt]]*sz[rt];
	dfs22(rt,0);
}
ll getans(vector<pair<int,int> > pt)
{
	sort(pt.begin(),pt.end());int q=pt.size();
	vector<int> p;
	for(int i=0;i<pt.size();i++)v1[pt[i].second]++;
	for(int i=1;i<=ct2;i++)fid[tid[i]]=0,tid[i]=0;
	for(int i=0;i<pt.size();i++){int v=pt[i].second;if(!fid[v])p.push_back(v),fid[v]=1;}
	doit(p,dep,f,0);for(int i=0;i<q;i++)as[i]=2*dp1[fid[pt[i].second]];
	for(int i=1;i<=ct2;i++)fid[tid[i]]=0,tid[i]=0;doit(p,dep2,f2,1);
	for(int i=0;i<q;i++)as[i]+=dp2[fid[pt[i].second]];
	for(int i=0;i<pt.size();i++)v1[pt[i].second]--;
	ll res=0;
	for(int i=0;i<q;i++)res+=pt[i].first-pt[0].first;
	for(int i=0;i<q;res+=1ll*(2*(i+1)-q)*(pt[i+1].first-pt[i].first),i++)as[i]+=res;
	int lb=0;
	while(lb<q)
	{
		int rb=lb;while(rb<q-1&&pt[rb+1].first==pt[lb].first)rb++;
		vector<int> p;
		for(int i=1;i<=ct2;i++)fid[tid[i]]=0,tid[i]=0;
		for(int i=lb;i<=rb;i++){int v=pt[i].second;if(!fid[v])p.push_back(v),fid[v]=1;v1[pt[i].second]++;}
		doit(p,dep,f,0);for(int i=lb;i<=rb;i++)as[i]-=2*dp1[fid[pt[i].second]],v1[pt[i].second]--;
		lb=rb+1;
	}
	ll mn=1e18;for(int i=0;i<q;i++)mn=min(mn,as[i]);
	return mn;
}
int main()
{
	scanf("%*d%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&vl[i]),vl[i]=vl[i]?0:1e9;
	for(int i=1;i<n;i++)scanf("%d%d",&s[i][0],&s[i][1]),adde(s[i][0],s[i][1]);
	dfs01(1,0);dfs02(1,0);
	for(int i=1;i<n;i++)e[i].f=s[i][0],e[i].t=s[i][1],e[i].v=min(vl[s[i][0]],vl[s[i][1]]);
	sort(e+1,e+n);ct=n;
	for(int i=1;i<=n*2;i++)fa[i]=i;
	for(int i=1;i<n;i++)
	{
		int v1=finds(e[i].f),v2=finds(e[i].t);
		fa[v1]=fa[v2]=f1[v1]=f1[v2]=++ct;vl[ct]=e[i].v;
	}
	for(int i=1;i<=ct;i++)head[i]=0;cnt=0;
	for(int i=1;i<ct;i++)adde(f1[i],i);
	dfs1(ct,0);
	scanf("%d",&q);
	while(q--)
	{
		tp.clear();scanf("%d",&k);
		for(int i=1;i<=k;i++)scanf("%d%d",&a,&b),tp.push_back(make_pair(a,b));
		printf("%lld\n",getans(tp));
	}
}
```



#### SCOI2020模拟11

##### auoj89 鱼死网破

###### Problem

有一个二维平面，在 $y>0$ 的部分有 $n$ 个点和 $k$ 条平行于 $x$ 轴的线段。

$q$ 次询问，每次询问给一个在 $y<0$ 部分的点 $P$，求 $n$ 个点中有多少个点满足 $P$ 向这个点连成的线段不经过任意一条给定的线段。

强制在线

$n,m\leq 10^5,k\leq 50$

$3s,512MB$

###### Sol

考虑一个点会对哪些询问点产生贡献。因为线段都在 $y>0$ 部分而询问在 $y<0$ 部分，因此有如下性质：

对于一个点，它与询问点 $P$ 的连线上没有给定线段当且仅当它向询问点的射线不与某条线段相交。

这样对于一个给定点，它能贡献的询问点在这个点出发的 $O(k)$ 个角度区间内。

考虑一个区间，它可以被表示为一个角 $\angle P_1QP_2$。因为询问点都在当前点下方，可以只考虑 $P_1,P_2$ 不在 $Q$ 上方的情况。此时可以考虑如下差分：

将 $\vec{QP_1}$ 左侧的半平面全部 $+1$，$\vec{QP_2}$ 右侧的半平面全部 $+1$，再整体 $-1$。（不妨设以 $Q$ 为原点，$P_1$ 在 $P_2$ 右侧，即 $\vec{QP_1}\times\vec{QP_2}\geq 0$）

这样会让反方向的部分多减去 $1$，但反方向部分是 $y>0$ 的部分，因此这样差分可以对于 $y<0$ 的部分得到答案。

问题变为，给定若干个半平面，求一个点在多少个半平面内，且每一个半平面的边界必定经过某一个线段端点。考虑可以每个端点记录经过这个端点的半平面有哪些，将这些半平面极角排序，询问时枚举每个端点，二分角度即可求出这个点上的半平面的贡献。注意精度问题（不能使用实数维护）

复杂度 $O(nk\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define ll long long
#define N 105000
#define M 105
struct pt{int x,y,id;}s[N],tp,t[M],t2[M];
vector<pt> r[M];
int v[M][3],n,m,k,op,x,y,ct,las;
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y,0};}
ll cross(pt a,pt b){return 1ll*a.x*b.y-1ll*a.y*b.x;}
bool cmp(pt a,pt b){return cross(a-tp,b-tp)>0;}
int main()
{
	scanf("%d%d%d%d",&n,&k,&m,&op);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i].x,&s[i].y);
	for(int i=1;i<=k;i++)
	{
		scanf("%d%d%d",&v[i][0],&v[i][1],&v[i][2]);
		if(v[i][0]>v[i][1])v[i][0]^=v[i][1]^=v[i][0]^=v[i][1];
		t[++ct]=(pt){v[i][0],v[i][2],i};
		t[++ct]=(pt){v[i][1],v[i][2],i+k};
	}
	for(int i=1;i<=n;i++)
	{
		int ct2=0;
		for(int j=1;j<=ct;j++)if(t[j].y<s[i].y)t2[++ct2]=t[j];
		tp=s[i];sort(t2+1,t2+ct2+1,cmp);
		int su=0;
		for(int j=1;j<=ct2;j++)
		if(t2[j].id<=k)
		{
			su++;
			if(su==1)r[t2[j].id].push_back(s[i]);
		}
		else
		{
			su--;
			if(!su)r[t2[j].id].push_back(s[i]);
		}
	}
	for(int i=1;i<=ct;i++)tp=t[i],sort(r[t[i].id].begin(),r[t[i].id].end(),cmp),t2[t[i].id]=t[i];
	while(m--)
	{
		scanf("%d%d",&x,&y);x^=las*op;y^=las*op;
		int as1=0;
		for(int i=1;i<=k;i++)
		{
			pt st=t2[i]-(pt){x,y};
			int lb=0,rb=r[i].size()-1,as=-1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(cross(r[i][mid]-t2[i],st)>=0)as=mid,lb=mid+1;
				else rb=mid-1;
			}
			as1+=as+1;
		}
		for(int i=1;i<=k;i++)
		{
			pt st=t2[i+k]-(pt){x,y};
			int lb=0,rb=r[i+k].size()-1,as=-1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(cross(r[i+k][mid]-t2[i+k],st)>0)as=mid,lb=mid+1;
				else rb=mid-1;
			}
			as1-=as+1;
		}
		printf("%d\n",las=n-as1);
	}
}
```



##### auoj90 漏网之鱼

###### Problem

给一个长度为 $n$ 的非负整数序列 $v$。

$q$ 组询问，每次给一个区间 $[l,r]$，求 $[l,r]$ 的所有子区间的区间mex值之和，mex为没有出现的最小非负整数值。

$n,q\leq 10^6$

$4s,512MB$

###### Sol

考虑固定 $l$，设 $f_i$ 表示从 $l$ 开始向右， $i$ 第一次出现的位置（没有出现则为 $+\infty$）

则对于任意正整数 $i$，区间 $[l,r]$ 的mex大于等于 $i$ 当且仅当 $r\geq max_{j=0}^if_j$。因此一个 $l$ 处的贡献只和 $f$ 中的前缀最大值有关。考虑 $f$ 的单调栈，则单调栈中的每个元素会对 $r$ 的一段后缀的的答案产生贡献。

从大到小考虑 $l$，这对于 $f$ 的影响相当于每次相当于将一个 $f_i$ 改为 $l$。

考虑这个过程中单调栈的改变，这次操作会将 $f_i$ 变为当前 $f$ 中的的最小值。因此如果这个元素不在单调栈上则没有影响，否则考虑删掉这个元素，然后向右依次找可以加进去的元素，即一段区间中第一个大于某个数的位置。找位置部分可以线段树维护。因为删除次数为 $O(n)$，因此单调栈上总的修改次数为 $O(n)$，且维护的复杂度为 $O(n\log n)$。

考虑计算答案，相当于有 $n$ 个时刻，这些时刻上有 $O(n)$ 次区间加操作，每次给定一个区间，求这个区间在某一个时间段内每个时刻区间和的和。

从小到大考虑所有时刻，若当前时刻为 $t$，则此时区间加一个数 $v$ 可以看成区间加 $v(x-t+1)$，其中 $x$ 为询问时刻。线段树上维护一次函数即可。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
#include<set>
using namespace std;
#define N 1050000
#define ll long long
int n,q,v[N],vl[N],a,b;
ll as[N];
set<int> fu;
struct que{int l,r,id;};
vector<que> qu[N];
struct segt1{
	struct node{int l,r,mx;}e[N*4];
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;e[x].mx=n+1;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void modify(int x,int l,int v)
	{
		if(e[x].l==e[x].r){e[x].mx=v;return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=l)modify(x<<1,l,v);else modify(x<<1|1,l,v);
		e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);
	}
	int query(int x,int l,int r,int v)
	{
		if(e[x].mx<v)return -1;
		if(e[x].l==e[x].r)return e[x].l;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r,v);
		else if(mid<l)return query(x<<1|1,l,r,v);
		else
		{
			int as=query(x<<1,l,mid,v);
			if(as==-1)as=query(x<<1|1,mid+1,r,v);
			return as;
		}
	}
}tr;
struct segt2{
	struct node{int l,r;ll s1,s2,l1,l2;}e[N*4];
	void pushdown(int x){if(e[x].l1){e[x<<1].l1+=e[x].l1;e[x<<1].s1+=(e[x<<1].r-e[x<<1].l+1)*e[x].l1;e[x<<1|1].l1+=e[x].l1;e[x<<1|1].s1+=(e[x<<1|1].r-e[x<<1|1].l+1)*e[x].l1;e[x].l1=0;}
	if(e[x].l2){e[x<<1].l2+=e[x].l2;e[x<<1].s2+=(e[x<<1].r-e[x<<1].l+1)*e[x].l2;e[x<<1|1].l2+=e[x].l2;e[x<<1|1].s2+=(e[x<<1|1].r-e[x<<1|1].l+1)*e[x].l2;e[x].l2=0;}}
	void pushup(int x){e[x].s1=e[x<<1].s1+e[x<<1|1].s1;e[x].s2=e[x<<1].s2+e[x<<1|1].s2;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void modify(int x,int l,int r,ll s1,ll s2)
	{
		if(e[x].l==l&&e[x].r==r){e[x].s1+=1ll*s1*(e[x].r-e[x].l+1);e[x].s2+=1ll*s2*(e[x].r-e[x].l+1);e[x].l1+=s1;e[x].l2+=s2;return;}
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify(x<<1,l,r,s1,s2);
		else if(mid<l)modify(x<<1|1,l,r,s1,s2);
		else modify(x<<1,l,mid,s1,s2),modify(x<<1|1,mid+1,r,s1,s2);
		pushup(x);
	}
	pair<ll,ll> query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return make_pair(e[x].s1,e[x].s2);
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);
		else if(mid<l)return query(x<<1|1,l,r);
		else
		{
			pair<ll,ll> v1=query(x<<1,l,mid),v2=query(x<<1|1,mid+1,r);
			return make_pair(v1.first+v2.first,v1.second+v2.second);
		}
	}
}tr2;
int main()
{
	scanf("%*d%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),v[i]=v[i]>n?n:v[i];
	scanf("%d",&q);
	for(int i=1;i<=q;i++)scanf("%d%d",&a,&b),qu[a].push_back((que){a,b,i});
	fu.insert(0);tr.build(1,0,n);tr2.build(1,1,n+1);
	fu.insert(n+1);
	for(int i=0;i<=n;i++)vl[i]=n+1;
	for(int i=n;i>=1;i--)
	{
		if(!fu.count(v[i]))vl[v[i]]=i,tr.modify(1,v[i],i);
		else
		{
			int las=-1,rb=n,lb=v[i];
			set<int>::iterator it=fu.find(v[i]);
			if(it!=fu.begin())las=*(--it);
			it=fu.find(v[i]);it++;rb=(*it)-1;
			tr2.modify(1,vl[v[i]],n+1,-1ll*i*(rb-lb+1),(rb-lb+1));
			tr.modify(1,v[i],i);vl[v[i]]=i;fu.erase(v[i]);
			if(las!=-1)tr2.modify(1,vl[las],n+1,-1ll*i*(v[i]-las),(v[i]-las));
			while(1)
			{
				int tp=tr.query(1,lb,rb,vl[las]+1);
				if(tp==-1)break;
				if(las!=-1)tr2.modify(1,vl[las],n+1,1ll*i*(tp-las),-(tp-las));
				las=tp;fu.insert(tp);lb=tp+1;
			}
			tr2.modify(1,vl[las],n+1,1ll*i*(rb-las+1),-(rb-las+1));
		}
		for(int j=0;j<qu[i].size();j++)
		{
			pair<ll,ll> tp=tr2.query(1,qu[i][j].l,qu[i][j].r);
			as[qu[i][j].id]=1ll*tp.first+1ll*tp.second*(i-1);
		}
	}
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```



##### auoj91 浑水摸鱼

###### Problem

给一个长度为 $n$ 的序列 $a$，对于一个区间 $[l,r]$，定义序列的这个区间的权值序列为：

将 $a_{l,\cdots,r}$ 重新标号，使得新标号均为正整数，且两个位置的新标号相同当且仅当原先的值相同。字典序最小的标号方式得到的序列即为权值序列。

求本质不同的权值序列数量。

$n\leq 5\times 10^4$

$4s,512MB$

###### Sol

显然一个区间的权值序列可以贪心确定，即从左向右考虑每个位置，如果这种数出现过，则和之前使用相同的标号，否则使用当前能用的最小标号。

那么可以发现，如果 $l$ 确定，那么以 $l$ 开头的权值序列的情况全部相同，即如果求出了 $[l,n]$ 的权值序列，则 $[l,r]$ 的权值序列为 $[l,n]$ 权值序列的一个前缀。

因此问题相当于，有若干个字符串，求有多少个本质不同前缀。则显然将所有字符串按照字典序排序，答案即为总长度减去每一对相邻字符串的lcp之和。



但此时不能快速求出所有的字符串，因此考虑在不求出字符串的情况下快速求两个字符串的lcp及比较字典序。

两个序列相同，当且仅当它们能够重标号为相同的序列。这也可以看成对于任意两个位置，第一个序列中它们相等当且仅当第二个序列中它们相等。

考虑记录 $pre_i$ 表示原序列中，这个位置的数上一次出现的位置，那么可以发现两个区间 $[l_1,r_1],[l_2,r_2]$ 可以进行重标号，当且仅当 $\forall i\in[l_1,r_1]$，如果 $pre_i\geq l_i$，则 $i-pre_i=(i+l_2-l_1)-pre_{i+l_2-l_1}$，即 $pre_i\geq l_i$ 部分的 $pre_i$ 关系相同。

因此判断相同时，只需要判断一个区间内 $i-pre_i$ 构成的序列（将 $pre_i<l$ 的位置的值变为 $0$）是否相等。则对于每一个 $l$，它对应的序列为原 $i-pre_i$ 序列中将 $pre_i<l$ 的位置变为 $0$ 的序列。考虑 $l$ 从小到大的过程，则相当于序列中的位置依次变成 $0$，因此使用主席树即可维护每个 $l$ 的序列。然后使用主席树维护hash值即可判断相同。

最后求lcp可以二分长度，判断字典序只需要判断下一位的大小关系，此时需要知道这种数在 $l$ 后面第一次出现的位置，以及 $l$ 到这个位置之间有多少种数，这都可以主席树维护。

复杂度 $O(n\log^3 n)$，注意 `sort` 默认cmp是 $O(1)$ 的因此加入了一些优化，但这里cmp不是 $O(1)$ 的，因此 `stable_sort` 会更快。

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 50050
#define ll long long
#define mod 10000000000001ll
int n,v[N],nt[N],vl[N];
ll pw[N],v1[N];
set<int> tp[N];
struct pretree{
	ll s1[N*32];
	int s2[N*32],ch[N*32][2],rt[N],ct,ct2;
	int build(int l,int r)
	{
		int st=++ct;
		if(l==r)return st;
		int mid=(l+r)>>1;
		ch[st][0]=build(l,mid);ch[st][1]=build(mid+1,r);
		return st;
	}
	void init(){rt[0]=build(1,n+1);}
	int modify(int x,int l,int r,int v,ll v1,int v2)
	{
		int st=++ct;s1[st]=(s1[x]+v1)%mod,s2[st]=s2[x]+v2;ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];
		if(l==r)return st;
		int mid=(l+r)>>1;
		if(mid>=v)ch[st][0]=modify(ch[x][0],l,mid,v,v1,v2);
		else ch[st][1]=modify(ch[x][1],mid+1,r,v,v1,v2);
		return st;
	}
	void modify2(int v,ll v1,int v2){rt[ct2+1]=modify(rt[ct2],1,n+1,v,v1,v2);ct2++;}
	ll que1(int x1,int l,int r,int r1)
	{
		if(!x1)return 0;
		if(r==r1)return (mod-s1[x1])%mod;
		int mid=(l+r)>>1;
		if(mid>=r1)return que1(ch[x1][0],l,mid,r1);
		else return (mod-s1[ch[x1][0]]+que1(ch[x1][1],mid+1,r,r1))%mod;
	}
	int que2(int x1,int x2,int l,int r,int l1,int r1)
	{
		if(!x2)return 0;
		if(l==l1&&r==r1)return s2[x2]-s2[x1];
		int mid=(l+r)>>1;
		if(mid>=r1)return que2(ch[x1][0],ch[x2][0],l,mid,l1,r1);
		else if(mid<l1)return que2(ch[x1][1],ch[x2][1],mid+1,r,l1,r1);
		else return que2(ch[x1][0],ch[x2][0],l,mid,l1,mid)+que2(ch[x1][1],ch[x2][1],mid+1,r,mid+1,r1);
	}
}tr;
ll mul(ll a,ll b){ll tp=(long double)a*b/mod;return (a*b-tp*mod+mod)%mod;}
int lcp(int a,int b)
{
	int lb=1,rb=n-max(a,b)+1,as=1;
	if(a>b)a^=b^=a^=b;
	ll v11=tr.que1(tr.rt[a-1],1,n+1,a-1),v21=tr.que1(tr.rt[b-1],1,n+1,b-1);
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(mul(pw[b-a],mod+tr.que1(tr.rt[a-1],1,n+1,a+mid-1)+mod-v11+v1[a+mid-1]-v1[a-1])==(tr.que1(tr.rt[b-1],1,n+1,b+mid-1)+mod-v21+v1[b+mid-1]-v1[b-1]+mod)%mod)as=mid,lb=mid+1;
		else rb=mid-1;
	}
	return as;
}
bool cmp(int a,int b)
{
	int tp1=lcp(a,b);
	if(a+tp1-1==n)return 1;
	if(b+tp1-1==n)return 0;
	int v1=v[a+tp1];
	int v2=*tp[v1].lower_bound(a);
	int s1=tr.que2(tr.rt[a-1],tr.rt[v2-1],1,n+1,v2,n+1);
	int v3=v[b+tp1];
	int v4=*tp[v3].lower_bound(b);
	int s2=tr.que2(tr.rt[b-1],tr.rt[v4-1],1,n+1,v4,n+1);
	return s1<s2;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),vl[i]=n+1,tp[v[i]].insert(i);
	for(int i=n;i>=1;i--)nt[i]=vl[v[i]],vl[v[i]]=i;
	tr.init();ll st=1;pw[0]=1;for(int i=1;i<=n;i++)st=50021*st%mod,pw[i]=st,tr.modify2(nt[i],st*(nt[i]-i+1)%mod,1),v1[nt[i]]=(v1[nt[i]]+st*(nt[i]-i+1)%mod)%mod;
	for(int i=2;i<=n;i++)v1[i]=(v1[i]+v1[i-1])%mod;
	for(int i=1;i<=n;i++)vl[i]=i;
	stable_sort(vl+1,vl+n+1,cmp);
	ll as=1ll*n*(n+1)/2;
	for(int i=1;i<n;i++)as-=lcp(vl[i],vl[i+1]);
	printf("%lld\n",as);
}
```



#### SCOI2020模拟12

##### auoj92 序列

###### Problem

给一个长度为 $n$ 的序列序列 $a$ 以及 $m$ 个数，你可以从 $m$ 个数中选出一些数插入到 $a$ 中，要求相邻两个 $a$ 之间最多插入一个数，且不能插入在开头和结尾。

对于每个 $k=1,\cdots,m$，求出插入 $k$ 个数，得到的序列中相邻两个位置的差的和可能的最大值

$n\leq 10^5$

$1s,1024MB$

###### Sol

如果把 $v$ 插入 $(a_i,a_{i+1})$ 中，则贡献为 $max(0,2(min(a_{i+1},a_i)-v),2(v-max(a_i,a_{i+1})))$

显然插入不会变差，可以变成插入至多 $k$ 个，然后可以将上面的贡献变为 $max(2(min(a_{i+1},a_i)-v),2(v-max(a_i,a_{i+1})))$。

第一个 $\max$ 可以看成图分成两部分，分别表示两种转移，这样最大流会选择最大的部分。

考虑一部分的建图，则考虑 $\min$ 的变化，可以看成如下方式：

> 每一个插入元素对应点 $s1_i$，每一个原来的空位对应 $t1_i$，有边 $s1_i\to t1_j$，费用为 $2*(\min(a_{j+1},a_j)-v_i)$。

一种方式是可以将所有边连到一条链上，更进一步可以发现可以将边权变为点权，使得中间的边费用均为 $0$，即给 $s1_i$ 一个 $-2v_i$ 的权值，$t1_i$ 一个 $2\min(a_{i+1},a_i)$ 的权值，这样所有 $s\to t$ 的边边权都变为 $0$，因此可以看成 $s1$ 全部连到一个点 $x_1$ 上，边权为上一步的点权，这个点再连到所有 $t1$ 上。

对于另外一部分有类似建图，唯一区别为 $\min$ 变为 $\max$ 且权值变为相反数。为了同时处理两部分，可以考虑如下建图：

> 原点向每个 $s_i$ 连流量为 $1$ 的边，$s_i$ 向 $s1_i,s2_i$ 连边。在另外一边，$t1_i,t2_i$ 向 $t_i$ 连边，$t_i$ 向汇点连流量为 $1$ 的边。

这样每个点只会操作一次。求出这个图上每个流量的最大费用最大流即为答案。



考虑模拟增广的过程，图可以看成有两个关键点 $x_1,x_2$。考虑两部分的点。

1. 对于一个上面部分的点 $v$，如果它没有被匹配，则不能经过这个点走增广路（除非从这个点出发）。否则，假设匹配到了 $x_1$，即 $\min$ 部分，则可以看成有一条 $x_1\to x_2$ 的增广路，代价为 $4v_i$。另外一个方向的情况类似但边权相反。
2. 对于一个下面部分的点 $(a,b)$，其中 $a$ 为相邻元素的最小值，$b$ 为最大值，则不存在经过它但不在这停止的最短路。如果匹配到了 $x_1$，则可以看成一条 $x_1\to x_2$ 的增广路，代价为 $-2(a+b)$，另外一个方向的情况类似。

所有增广路一定经过 $x_1,x_2$ 中的一个，因此可以发现一条增广路一定为如下情况之一：$s\to x_1\to t,s\to x_2\to t,s\to x_1\to x_2\to t,s\to x_2\to x_1\to t$。

首先 $s\to x_1,x_2$ 和 $x_1,x_2\to t$ 的部分容易求出，只需要记录每一部分当前没有被匹配的所有点按照权值排序的结果即可。

然后考虑中间的边，显然只需要维护当前的所有匹配导致的增广路。可以发现每次匹配只会改变 $O(1)$ 个点的情况，直接修改所有的增广路即可。

虽然有两种可能的增广边，但实际上可以发现，经过上面部分的边一定没有用。考虑一条走上面部分的路径 $s\to v_1\to x_1\to v_2\to x_2\to t$，此时原先存在的匹配的一部分为 $s\to v_2\to x_1$。则如果 $s\to v_1$ 的边权更大，则之前走这条边严格更优，因此不可能是这种情况，即权值上 $v_1>v_2$。但可以发现 $v_1\to x_1\to v_2\to x_2\to v_1$ 的这个环的边权和为 $4(v_2-v_1)$，此时这个值为负，因此直接走 $s\to v_1\to x_2$ 会更优。因此一定不会走这种边。

因此只存在第二种边，可以发现只需要对于 $x_1,x_2$ 分别按照权值维护所有匹配这边的 $(a,b)$ 的 $a+b$，即可找到最大权值的增广路，同时改变匹配之后容易重新维护新的增广路。这部分维护两个优先队列即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 100060
int n,m,v[N],v1[N],f1[N],f2[N],t1[N],t2[N],is[N];
long long su,las;
bool cmp1(int a,int b){return f1[a]>f1[b];}
bool cmp2(int a,int b){return f2[a]<f2[b];}
priority_queue<int> q1,q2;
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<n;i++)f1[i]=min(v[i],v[i+1]),f2[i]=max(v[i],v[i+1]),su+=v[i]>v[i+1]?v[i]-v[i+1]:v[i+1]-v[i],t1[i]=t2[i]=i;
	for(int i=1;i<=m;i++)scanf("%d",&v1[i]);sort(v1+1,v1+m+1);
	sort(t1+1,t1+n,cmp1);
	sort(t2+1,t2+n,cmp2);
	int l1=1,r1=m,d1=1,d2=1;
	las=su;
	q1.push(-1e9);q2.push(-1e9);
	for(int i=1;i<=m;i++)
	{
		while(is[t1[d1]])d1++;
		int mx1=f1[t1[d1]];
		while(is[t2[d2]])d2++;
		int mn1=f2[t2[d2]];
		int vl1=mx1-v1[l1],vl2=v1[r1]-mn1;
		int vl3=q1.top()-v1[l1]-mn1,vl4=mx1+v1[r1]+q2.top();
		int mx=max(max(vl1,vl2),max(vl3,vl4));
		if(vl1==mx)
		{
			is[t1[d1]]=1;l1++;
			q2.push(-(f1[t1[d1]]+f2[t1[d1]]));
		}
		else if(vl2==mx)
		{
			is[t2[d2]]=1;r1--;
			q1.push(f1[t2[d2]]+f2[t2[d2]]);
		}
		else if(vl3==mx)
		{
			is[t2[d2]]=1;l1++;
			int v2=q1.top();q1.pop();q2.push(-v2);
			q1.push(f1[t2[d2]]+f2[t2[d2]]);
		}
		else
		{
			is[t1[d1]]=1;r1--;
			int v2=q2.top();q2.pop();q1.push(-v2);
			q2.push(-(f1[t1[d1]]+f2[t1[d1]]));
		}
		su+=2*mx;
		if(su<las)su=las;
		las=su;
		printf("%lld ",su);
	}
}
```



##### auoj93 小Z的树

###### Problem

有一个 $n$ 个点的有根内向森林，每个根节点有一个权值，支持 $q$ 次如下类型的操作：

1. 将一个点变为根（删去它和父亲的边），并给出它的权值
2. 修改一个点的父亲（可能将一个点变为根）
3. 定义一个点的价值为它所在树的根节点的权值，求 $[l,r]$ 内所有节点的价值和。

$n,q\leq 2\times 10^5$

$5s,1024MB$

###### Sol

对操作分块，每一块内的操作只会修改 $O(size)$ 个点的父亲。

将这些点到父亲的边断开，那么此时的每个连通块在这一块的操作中父亲都是一样的，因此它们的权值全部相同，可以对于一个连通块一起处理。

对于所有与修改的点不连通的连通块，它们的价值不会改变，可以先预处理这部分的前缀和。

对于剩下的块，这部分最多只有 $O(size)$ 个，对于修改操作可以直接维护每个块当前的父亲节点。

然后考虑询问，可以暴力求出每个块当前的根节点，只需要对于每个块求这个块有多少个点在给定区间内即可得到答案。

如果直接 `lower_bound`，则 $n,q$ 同阶情况下复杂度为 $O(n\sqrt{n\log n})$，但常数很小正好可以通过。

考虑再离线一次，然后从小到大考虑每个节点，维护前缀中每个块内的点出现了多少次，每个询问在两个端点处考虑一次，这样即可 $O(n+size^2)$ 求出每个询问中每一块在区间内的节点个数。这样复杂度即为 $O(q\sqrt n)$。

###### Code

~~写的是暴力但是懒得改了~~

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 205000
#define M 2000
int n,q,fa[N],vl[N],f1[N],f2[N],is[N],is2[N],id[N],tid[N],ct,qu[N][3],as[N];
vector<int> fu[M];
long long s1[N];
void doit(int l,int r)
{
	for(int i=1;i<=n;i++)f1[i]=f2[i]=is[i]=is2[i]=id[i]=0;ct=0;
	for(int i=1;i<=n;i++)
	{
		int st=i;
		while(1)
		{
			if(f1[st]){break;}
			if(!fa[st]){f1[st]=st;break;}
			st=fa[st];
		}
		int as=f1[st];
		st=i;
		while(!f1[st])f1[st]=as,st=fa[st];
	}
	for(int i=l;i<=r;i++)if(qu[i][0]!=2)is[f1[qu[i][1]]]=1,is2[qu[i][1]]=1;
	for(int i=1;i<=n;i++)if(!is[f1[i]])s1[i]=vl[f1[i]];else s1[i]=0;
	for(int i=2;i<=n;i++)s1[i]+=s1[i-1];
	for(int i=1;i<=n;i++)
	{
		int st=i;
		while(1)
		{
			if(f2[st]){break;}
			if(!fa[st]||is2[st]){f2[st]=st;break;}
			st=fa[st];
		}
		int as=f2[st];
		st=i;
		while(!f2[st])f2[st]=as,st=fa[st];
	}
	for(int i=1;i<=n;i++)if(is[f1[i]])
	{
		if(!id[f2[i]])id[f2[i]]=++ct,tid[ct]=f2[i];
		fu[id[f2[i]]].push_back(i);
	}
	for(int i=l;i<=r;i++)
	if(qu[i][0]==0){vl[qu[i][1]]=qu[i][2];fa[qu[i][1]]=0;}
	else if(qu[i][0]==1){fa[qu[i][1]]=qu[i][2];}
	else
	{
		long long as1=s1[qu[i][2]]-s1[qu[i][1]-1];
		for(int j=1;j<=ct;j++)as[j]=-2e9;
		for(int j=1;j<=ct;j++)
		{
			int st=tid[j];
			while(as[j]==-2e9)
			{
				if(!is[f1[st]]){as[j]=vl[f1[st]];break;}
				st=f2[st];
				if(!fa[st])as[j]=vl[st];
				else if(as[id[st]]!=-2e9)as[j]=as[id[st]];
				else st=fa[st];
			}
			as1+=1ll*as[j]*(lower_bound(fu[j].begin(),fu[j].end(),qu[i][2]+1)-lower_bound(fu[j].begin(),fu[j].end(),qu[i][1]));
		}
		printf("%lld\n",as1);
	}
	for(int j=1;j<=ct;j++)fu[j].clear();
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&vl[i]);
	for(int i=1;i<=q;i++)scanf("%d%d%d",&qu[i][0],&qu[i][1],&qu[i][2]);
	for(int i=1;i<=q;i+=500)doit(i,i+499>q?q:i+499);
}
```



##### auoj94 小Y的图

###### Problem

有一个 $n\times n$ 的网格，保证 $n$ 是 $5$ 的倍数。

你在网格上进行移动。假设你当前在 $(x,y)$，则你下一步可以移动到以下 $8$ 个位置之一：

$(x-3,y),(x-2,y-2),(x,y-3),(x+2,y-2),(x+3,y),(x+2,y+2),(x,y+3),(x-2,y+2)$

构造一种从 $(1,1)$ 出发，遍历所有格子各一次并回到原点的方案或输出无解。

$n\leq 1000$

$1s,1024MB$

###### Sol

条件提示将网格划分成若干个 $5\times 5$ 的矩形，考虑一个网格的情况。
 
对于左上角的矩形，考虑留出 $(3,3)$，将其它位置走完然后走到下面的矩形，最后从右侧的 $(3,6)$ 走过来。dfs可以发现存在一种方式，使得从 $(1,1)$ 开始，不经过 $(3,3)$ 经过其它所有点最后到达 $(5,3)$。

这样可以走到下面的 $5\times 5$ 的中心 $(5,3)\to(8,3)$。考虑从一个中心向另外一个中心移动，即从 $(3,3)$ 开始走完整个 $5\times 5$，最后停留到 $(3,5)$ 或者对称位置。

dfs可以发现存在方案，因此可以采用在 $5\times 5$ 的部分上走的方式，于是 $10|n$ 的情况可以这样构造哈密顿回路:

```
o--o--o--o--o--o
|              |
o  o--o  o--o  o
|  |  |  |  |  |
o  o  o  o  o  o
|  |  |  |  |  |
o  o  o  o  o  o
|  |  |  |  |  |
o  o  o  o  o  o
|  |  |  |  |  |
o--o  o--o  o--o
```


奇数的情况不存在哈密顿回路，考虑一种斜向的方式，即这样构造：

```
o--o--o--o--o
|         //  
o  o--o  o--o
|  |  |     |
o  o  o  o--o
|  |  |  |
o  o  o  o--o
|  |  |     |
o--o  o--o--o
```

考虑斜线上的操作，dfs可以发现存在中心开头 $(1,5)$ 结束的方案，也存在 $(4,2)$ 开头 $(3,1)$ 结尾的方案，因此可以这样构造。

最后 $n=5$ 的特殊情况可以单独处理。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2333
int n,as[N][N],st[6][6],e1,e2,fg,ct=24,as2[6][6][6][6][6][6];
void dfs(int d,int x,int y)
{
	if(d==26){fg=1;return;}
	if(x==e1&&y==e2){st[x][y]=0;return;}
	if(x>3&&st[x-3][y]==0)st[x-3][y]=d,dfs(d+1,x-3,y);if(fg)return;
	if(x<5-2&&st[x+3][y]==0)st[x+3][y]=d,dfs(d+1,x+3,y);if(fg)return;
	if(y>3&&st[x][y-3]==0)st[x][y-3]=d,dfs(d+1,x,y-3);if(fg)return;
	if(y<5-2&&st[x][y+3]==0)st[x][y+3]=d,dfs(d+1,x,y+3);if(fg)return;
	{
		if(x>2&&y>2&&st[x-2][y-2]==0)st[x-2][y-2]=d,dfs(d+1,x-2,y-2);if(fg)return;
		if(x>2&&y<5-1&&st[x-2][y+2]==0)st[x-2][y+2]=d,dfs(d+1,x-2,y+2);if(fg)return;
		if(x<5-1&&y>2&&st[x+2][y-2]==0)st[x+2][y-2]=d,dfs(d+1,x+2,y-2);if(fg)return;
		if(x<5-1&&y<5-1&&st[x+2][y+2]==0)st[x+2][y+2]=d,dfs(d+1,x+2,y+2);if(fg)return;
	}
	st[x][y]=0;
}
void justdoit(int s1,int s2,int t1,int t2)
{
	if(as2[s1][s2][t1][t2][1][1]){for(int i=1;i<=5;i++)for(int j=1;j<=5;j++)st[i][j]=as2[s1][s2][t1][t2][i][j];return;}
	e1=t1,e2=t2;fg=0;
	for(int i=1;i<=5;i++)for(int j=1;j<=5;j++)st[i][j]=0;
	st[s1][s2]=1;dfs(2,s1,s2);
	for(int i=1;i<=5;i++)for(int j=1;j<=5;j++)as2[s1][s2][t1][t2][i][j]=st[i][j];
}
void doit(int s1,int s2,int t1,int t2,int x,int y)
{
	justdoit(s1,s2,t1,t2);
	for(int j=1;j<=5;j++)
	for(int k=1;k<=5;k++)
	as[x*5-5+j][y*5-5+k]=st[j][k]+ct;
	ct+=25;
}
int main()
{
	scanf("%d",&n);
	if(n==5){printf("1 14 7 4 15\n9 22 17 12 23\n19 5 25 20 6\n2 13 8 3 16\n10 21 18 11 24");return 0;}
	as[1][1]=1;as[1][2]=8;as[1][3]=5;as[1][4]=2;as[1][5]=9;
	as[2][1]=15;as[2][2]=20;as[2][3]=23;as[2][4]=12;as[2][5]=17;
	as[3][1]=6;as[3][2]=3;as[3][3]=23333;as[3][4]=7;as[3][5]=4;
	as[4][1]=22;as[4][2]=11;as[4][3]=16;as[4][4]=21;as[4][5]=10;
	as[5][1]=14;as[5][2]=19;as[5][3]=24;as[5][4]=13;as[5][5]=18;
	for(int i=2;i<n/5;i++)doit(3,3,5,3,i,1);
	doit(3,3,3,5,n/5,1);
	for(int i=2;i<n/5;i++)doit(3,3,3,5,n/5,i);
	doit(3,3,1,3,n/5,n/5);
	for(int i=n/5-1;i>1;i--)doit(3,3,1,3,i,n/5);
	doit(3,3,3,1,1,n/5);
	if(n%10==0)
	for(int i=n/5-1;i>1;i-=2)
	{
		for(int j=1;j<n/5-1;j++)doit(3,3,5,3,j,i);
		doit(3,3,3,1,n/5-1,i);
		for(int j=n/5-1;j>1;j--)doit(3,3,1,3,j,i-1);
		doit(3,3,3,1,1,i-1);
	}
	else
	if(n>15)
	{
		for(int i=n/5-1;i>4;i-=2)
		{
			for(int j=1;j<n/5-1;j++)doit(3,3,5,3,j,i);
			doit(3,3,3,1,n/5-1,i);
			for(int j=n/5-1;j>1;j--)doit(3,3,1,3,j,i-1);
			doit(3,3,3,1,1,i-1);
		}
		doit(3,3,5,3,1,4);
		for(int j=2;j<n/5-1;j++)doit(3,3,5,3,j,4);
		doit(3,3,3,1,n/5-1,4);
		for(int j=n/5-1;j>2;j-=2)
		{
			doit(3,3,3,1,j,3);
			doit(3,3,1,3,j,2);
			doit(3,3,3,5,j-1,2);
			doit(3,3,1,3,j-1,3);
		}
		doit(3,3,3,1,2,3);
		doit(3,3,1,5,2,2);
		doit(4,2,3,1,1,3);
		doit(3,3,3,1,1,2);
	}
	else
	{
		ct-=50;
		doit(3,3,3,1,2,3);
		doit(3,3,1,5,2,2);
		doit(4,2,3,1,1,3);
		doit(3,3,3,1,1,2);
	}
	as[3][3]=ct+1;
	for(int i=1;i<=n;i++,printf("\n"))
	for(int j=1;j<=n;j++)printf("%d ",as[i][j]);
}
```



#### NOI2020模拟六校联测3

##### auoj95 数据结构

###### Problem

你有 $n$ 个数据结构，每个数据结构有两个权值 $a_i,b_i(a_i\geq b_i)$。

你可以对数据结构进行嵌套操作，可以把第 $i$ 个数据结构套在第 $j$ 个里面当且仅当 $a_i<b_j$。

每个数据结构只能直接套一个数据结构，每个数据结构只能被直接套一次，但可以进行多重嵌套。

你会一直执行嵌套操作，直到不能操作为止。求可能的嵌套方案数，模 $10^9+7$

$n\leq 300$

$1s,512MB$

###### Sol

因为 $a_i\geq b_i$，因此一个嵌套关系中的 $b_1,a_1,b_2,a_2,\cdots,b_k,a_k$ 是不降的所以只要满足了第一条就不会出现嵌套出环的情况。从而可以将所有的 $(a_i,b_i)$ 拆开，看成 $a,b$ 间配对的方案数。

则问题可以看成有 $n$ 个 $a_i$ 和 $n$ 个 $b_i$，一个 $b_i$ 只能匹配小于他的 $a_i$，求不能再匹配的匹配方案数。

因为最后要求不能再匹配，因此如果当前有一个 $a$ 最后没有被匹配，那么大于它的 $b$ 必须匹配，且可以发现如果满足这一条件，则一定不能再匹配。

设 $dp_{i,j,0/1}$ 表示从小到大考虑了前 $i$ 个数，前面有 $j$ 个 $a$ 会和后面的 $a$ 匹配，前面是否有 $a$ 最后没有匹配。

从小到大考虑每一个数（相同情况先 $b$ 后 $a$）。如果当前是 $b$，则转移为：

$$
dp_{i,j,0}=dp_{i-1,j,0}+dp_{i-1,j+1,0}*(j+1)\\
dp_{i,j,1}=dp_{i-1,j+1,1}*(j+1)
$$

如果当前是 $a$，则转移为：

$$
dp_{i,j,0}=dp_{i-1,j-1,0}\\
dp_{i,j,1}=dp_{i-1,j-1,1}+dp_{i-1,j,1}+dp_{i-1,j,0}
$$

答案为 $dp_{2n,0,0}+dp_{2n,0,1}$。

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 605
#define mod 1000000007
int n,v[N],dp[N][N][2];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%d%d",&v[i*2-1],&v[i*2]);
		v[i*2]<<=1;
		v[i*2-1]=v[i*2-1]<<1|1;
	}
	sort(v+1,v+n*2+1);
	dp[0][0][0]=1;
	for(int i=1;i<=n*2;i++)
	for(int j=0;j<=n;j++)
	for(int k=0;k<2;k++)
	if(~v[i]&1)
	{
		if(j)dp[i][j-1][k]=(dp[i][j-1][k]+1ll*dp[i-1][j][k]*j)%mod;
		if(!k)dp[i][j][k]=(dp[i][j][k]+dp[i-1][j][k])%mod;
	}
	else
	{
		dp[i][j+1][k]=(dp[i][j+1][k]+dp[i-1][j][k])%mod;
		dp[i][j][1]=(dp[i][j][1]+dp[i-1][j][k])%mod;
	}
	printf("%d\n",(dp[n*2][0][0]+dp[n*2][0][1])%mod);
}
```



##### auoj96 三国学者

###### Problem

给一棵有根树，每个点有 $[1,L]$ 间的整数权值 $s_i$。

有 $m$ 次修改，第 $i$ 次修改将第 $((i-1)\bmod n)+1$ 个点的权值为 $v_i$。

每次修改之后，你需要求出有多少个长度为 $L$ 的序列 $a_{1,...,m}$ 满足如下条件（对于所有 $i$）：

1. $1\leq a_i\leq n$
2. $a_i$ 号点的权值为 $L-i+1$
3. $a_i$ 为 $a_{i+1}$ 的祖先

记 $ans_i$ 为第 $i$ 次修改后的答案，输出 $\sum i*ans_i$ 模 $10^9+7$ 的结果。

$n,L\leq 10^6,m\leq 2\times 10^6,fa_i<i$

$1s,512MB$

###### Sol

考虑一次询问的 $dp$。设 $f_i$ 表示以 $i$ 结尾的合法序列数，那么有：

$$
f_i=
\begin{cases}
1,&s_i=L.\\
\sum_{j是i的祖先,s_j=s_u+1}f_j,& s_i<L
\end{cases}
$$

答案为 $\sum_{s_i=1}f_i$。

每次修改对答案的改变量为修改后包含这个点的序列数量减去修改前包含这个点的序列数量，考虑计算这个值。

设 $g_i$ 表示以 $i$ 开头的合法序列数，有：

$$
g_i=
\begin{cases}
1,&s_i=1.\\
\sum_{j在i子树内,s_j=s_i-1}g_j,& s_i>1
\end{cases}
$$

设 $h_{i,j}=\sum_{j在i子树内,s_j=k}g_j$，那么包含 $i$ 的序列数量为 $f_i*h_{i,s_i-1}$。

把每 $n$ 次修改看成一轮修改。考虑一轮修改的情况，因为 $fa_i<i$，因此修改到一个点时，它的祖先都被修改了，它的子树内都没有被修改。而 $f$ 只和祖先有关，$h$ 只和子树有关。因此考虑每一轮求出没有修改时每个点的 $h_{i,s_i-1}$，再对于每个点求出它的祖先都被修改了时，修改它前后这个点的 $f_i$，即可求出答案。

考虑计算 $f$，可以从上往下，对于每个 $x$ 维护 $\sum_{j是u的祖先,s_j=x}f_j$，其中 $u$ 为当前点。这样dfs每经过一条边只会改一个值，可以 $O(n)$ 求出 $f$。

考虑求 $h$，使用类似的方式对于每个 $x$ 维护答案。为了求子树内的值，可以在dfs进入子树前求一个需要的值，dfs子树后再求一个需要的值做差即可得到子树内的值。因此每一轮的复杂度均为 $O(n)$。

复杂度 $O(n+m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2000600
#define mod 1000000007
int n,m,l,fa[N],head[N],cnt,v1[N],v2[N],f1[N],f2[N],g1[N],g2[N],as[N*2],vl[N],vl2[N],q2[N*2];
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs0(int u,int fa)
{
	int f11=v1[vl[u]-1],f21=v1[vl2[u]-1];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);
	int fu=(v1[vl[u]-1]-f11+mod+(vl[u]==1))%mod;
	f2[u]=(v1[vl2[u]-1]-f21+mod+(vl2[u]==1))%mod;
	f1[u]=fu;v1[vl[u]]=(v1[vl[u]]+fu)%mod;
}
void dfs1(int u,int fa)
{
	g1[u]=v2[vl[u]+1]+(vl[u]==l);g2[u]=v2[vl2[u]+1]+(vl2[u]==l);v2[vl2[u]]=(v2[vl2[u]]+g2[u])%mod;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u);
	v2[vl2[u]]=(v2[vl2[u]]+mod-g2[u])%mod;
}
int rd(){int as=0;char c=getchar();while(c<'0'||c>'9')c=getchar();while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();return as;}
int main()
{
	scanf("%d%d%d",&n,&m,&l);
	for(int i=2;i<=n;i++)fa[i]=rd(),adde(i,fa[i]);
	for(int i=1;i<=n;i++)vl[i]=rd();
	int tp=(m-1)/n+1;
	for(int i=1;i<=tp;i++)q2[i]=1;
	for(int i=1;i<=m;i++)q2[i]=rd();
	for(int i=1;i<=tp;i++)
	{
		for(int j=1;j<=l;j++)v1[j]=v2[j]=f1[j]=f2[j]=g1[j]=g2[j]=0;
		for(int j=1;j<=n;j++)vl2[j]=q2[i*n-n+j];
		dfs0(1,0);int las=v1[l];dfs1(1,0);
		for(int j=1;j<=n;j++)las=(las+1ll*f2[j]*g2[j]-1ll*f1[j]*g1[j]%mod+mod)%mod,as[i*n-n+j]=las,vl[j]=vl2[j];
	}
	int su=0;
	for(int i=1;i<=m;i++)su=(su+1ll*i*as[i])%mod;
	printf("%d\n",su);
}
```



##### auoj97 浇花

###### Problem

（近似）通信题。

有一个 $4\times 4$ 的网格，上面有一枚棋子。

第一个人有一个非负整数 $v$，他需要在 $T$ 个时刻内将这个数字传递给第二个人。

每个时刻，有一个人可以对网格进行操作。在操作时，他可以知道当前棋子的位置，接下来他必须将棋子移动到一个相邻（上下左右）格子。

但这个时刻，另外一个人不会得到任何信息（包括另外一个人进行操作的信息），且双方不知道每个时刻操作的人是谁。但保证不会有一个人连续操作 $100$ 次以上。

你需要实现双方的策略，完成数字传递的过程。

$T=3600,v\leq 10^9$

$1s,100MB$

###### Sol

第一个困难是，一个人操作时另外一个人不知道对方进行了操作，因此需要找到一种方式使得双方确定操作顺序。

考虑每个人连续操作若干次时，不能将棋子移动回连续操作开始（对方上一次操作后）棋子的位置。每个人记录这个位置和自己上次操作后棋子的位置。则如果一个人开始操作时棋子位置与上次操作后的位置不同，则他可以知道对方进行了操作，否则对方没有进行操作。

第二个困难是先手不确定，这样即使知道了对方上一次有没有操作也无法确定操作顺序。

首先考虑如果知道先手顺序，如何传递信息。一个直接的方式是，可以从当前点向两个不同的方向分出两个 $1\times 2$ 的矩形，矩形的划分方案事先确定，只和起始位置有关。如果要传递 $0$，则走到第一个矩形上来回走，否则在第二个矩形上来回走。例如如下方式（其它情况可以通过旋转/翻转得到）：

```
x11
220
```

这样除去第一次操作外，每个人都可以通过上次连续操作时最后的位置和当前棋子位置得到对方传递的信息。



考虑如何用每次传递一位的方式确定先手。考虑如下方式：

1. 双方在自己第一次操作时传递 $0$，否则传递 $1$。
2. 如果第一个人在发送信息前既传递过了 $1$，也接受过了 $1$，则接下来开始传递 $v$。
3. 如果第二个人在接收上一个信息前既传递过了 $1$，也接受过了 $1$，则接下来开始接收 $v$。

考虑模拟两种可能的情况（设两个人为 $A,B$）：

如果 $A$ 先手，则 $A$ 传递 $0$，$B$ 传递 $0$，$A$ 传递 $1$，$B$ 传递 $1$。这一次操作后两个人都变得满足条件，接下来双方同时开始传递/接收。

否则，$B$ 传递 $0$，$A$ 传递 $0$，$B$ 传递 $1$，$A$ 传递 $1$。此时 $A$ 满足条件但还没有发送信息。接下来虽然 $B$ 接收到了 $1$，但由于接收前的限制当前 $B$ 不满足条件。接下来 $B$ 传递 $1$，然后双方同时开始操作。

这样可以使用不超过 $500$ 次操作，使得双方确定操作顺序。

但此时如果直接一位一位传递信息，则最坏需要 $200\log v$ 步，这是上限的两倍。

因此考虑一次传递两位信息。但一个问题是如果当前棋子不在中间，则第一步不可能有四种操作方式。

此时有两种解决方式。std的方式为第二个人的操作总是将棋子移动回网格中心部分。这样需要精细处理前几轮的操作。

另外一种想法是，考虑一个分步的方式，即第一步时传递一位信息，接下来连续移动时如果能再传递一位，则再传递一位信息。

可以发现，对于中心部分和边上部分，都可以第一步传递一位信息，第二步传递另外一位，一种构造方式为：

```
0330 0x13
5x14 6243
5204 6540
0660 0500
```

其中 $1,2$ 为第一步的两种操作，$3,4,5,6$ 为后面的四种操作方式。

对于角上的情况，也有类似构造：

```
x133
2440
2550
6600
```

这里可能第二位信息需要两步完成。

还有一种处理方式为，第二个人的操作保证任意时刻不移动到角上。在传递 $v$ 阶段这样做非常容易。在第一部分只需要让每个位置分出的两个 $1\times 2$ 不经过角，而这非常容易做到。

这样最坏 $101$ 步传递一位，$200$ 步传递两位。因此这部分最坏情况为 $101*(\log_2 v-1)+100+1$，这里为 $3030$ 步。

因此最多需要 $3530$ 步，可以通过。

###### Code

```cpp
char s1[16][17]={
	"0110001001000110",
	"2000021001200002",
	"2000020000200002",
	"0000000000000000",
	
	"0000011001100000",
	"0110200000020110",
	"2200200000020022",
	"0000000000000000",
};
char s2[16][17]={
	"0000001331000000",
	"0000624334260000",
	"0000654004560000",
	"0000050000500000",
	
	"0330033003300330",
	"0144501441054410",
	"2550520440250552",
	"6600066006600066",
};
void init_s()
{
	for(int i=8;i<16;i++)
	for(int j=0;j<16;j++)
	s1[i][j]=s1[15-i][j],s2[i][j]=s2[15-i][j];
}
int que1(int x,int y,int lx,int ly)
{
	if(lx<0||lx>3||ly<0||ly>3)return 0;
	return s1[x*4+lx][y*4+ly]-'0';
}
int que2(int x,int y,int lx,int ly)
{
	if(lx<0||lx>3||ly<0||ly>3)return 0;
	return s2[x*4+lx][y*4+ly]-'0';
}
int d[4][2]={-1,0,1,0,0,-1,0,1};
class Alice{
	int f1,f2,v,is,lx,ly,sx,sy;
	public:
	void initA(int ty,int vl){v=vl;f1=f2=is=0;lx=ly=-1;init_s();}
	int moveA(int x,int y)
	{
		int as=-1;x--;y--;
		if(lx==-1){sx=x;sy=y;for(int i=0;i<4;i++)if(que1(x,y,x+d[i][0],y+d[i][1])==1)as=i;}
		else if(lx!=x||ly!=y)
		{
			int v1=que1(lx,ly,x,y);
			if(v1==2)f2=1;
			if(f1&&f2)is=1;
			sx=x;sy=y;
			if(!is){f1=1;for(int i=0;i<4;i++)if(que1(x,y,x+d[i][0],y+d[i][1])==2)as=i;}
			else
			{
				for(int i=0;i<4;i++)if(que2(x,y,x+d[i][0],y+d[i][1])==(v&1)+1)as=i;
				v>>=1;
			}
		}
		else
		{
			if(!is){for(int i=0;i<4;i++)if(que1(sx,sy,x+d[i][0],y+d[i][1])==que1(sx,sy,x,y))as=i;}
			else if(que2(sx,sy,x,y)>2){for(int i=0;i<4;i++)if(que2(sx,sy,x+d[i][0],y+d[i][1])==que2(sx,sy,x,y))as=i;}
			else
			{
				for(int i=0;i<4;i++)if(que2(sx,sy,x+d[i][0],y+d[i][1])==que2(sx,sy,x,y)*2+1+(v&1))as=i;
				v>>=1;
			}
		}
		lx=x+d[as][0];ly=y+d[as][1];
		return -as-1;
	}
};
class Bob{
	int f1,f2,v,is,ct,lx,ly,sx,sy;
	public:
	void initB(int ty){f1=f2=is=0;lx=ly=-1;init_s();}
	int moveB(int x,int y)
	{
		int as=-1;x--;y--;
		if(lx==-1){sx=x;sy=y;for(int i=0;i<4;i++)if(que1(x,y,x+d[i][0],y+d[i][1])==1)as=i;}
		else if(lx!=x||ly!=y)
		{
			if(f1&&f2)is=1;
			if(!is)
			{
				int tp=que1(lx,ly,x,y);
				if(tp==2)f2=1;
				for(int i=0;i<4;i++)if(que1(x,y,x+d[i][0],y+d[i][1])==2)as=i;
				f1=1;
			}
			else
			{
				int tp=que2(lx,ly,x,y);
				if(tp<=2)v+=(tp-1)<<ct,ct++;
				else
				{
					tp-=3;
					if(tp%3)tp=3-tp;
					v+=tp<<ct;ct+=2;
				}
				if(ct>=30)return v;
				for(int i=0;i<4;i++)if(que1(x,y,x+d[i][0],y+d[i][1])==1)as=i;
			}
			sx=x;sy=y;
		}
		else for(int i=0;i<4;i++)if(que1(sx,sy,x+d[i][0],y+d[i][1])==que1(sx,sy,x,y))as=i;
		lx=x+d[as][0];ly=y+d[as][1];
		return -as-1;
	}
};
#include "chess.h"
```



#### SCOI2020模拟?

##### auoj98 ZSY家今天的饭

###### Problem

有一棵 $n$ 个点带边权的树，有 $m$ 个关键点，从中随机选出 $k$ 个点，你需要找到一条路径经过这 $k$ 个点且路径长度最短。

求最短路径的长度的期望，答案模 $998244353$

$n\leq 10^5,m,k\leq 500$

$2s,512MB$

###### Sol

使用dfs的方式容易发现，最短路径长度为最小的包含这 $k$ 个点的连通块的边数乘 $2$ 再减去连通块的直径。

对于边数部分，考虑枚举每条边算贡献，求出这条边两侧关键点的数量后可以 $O(1)$ 算这条边在连通块内的情况数。

对于直径部分，考虑枚举直径（如果有多对直径 $(u,v)$，找 $u$ 尽量小时 $v$ 尽量小的）。

对于一条直径 $(u,v)$，考虑加入哪些点时它仍然是直径，如果可以加入 $x$，则 $x$ 必须满足 $dis(u,x),dis(v,x)\leq dis(u,v)$。

而两个点的距离 $(x_1,x_2)$ 可以写成 $\max(dis(u,x_1)+dis(v,x_2)-dis(u,v),dis(u,x_2)+dis(v,x_1)-dis(u,v))$，因此两个合法点之间一定有 $dis(x_1,x_2)\leq dis(u,v)$。

再考虑路径长度相同的情况。根据上面的选择方案，可以钦定如果 $dis(u,x)=dis(u,v)$，则需要 $x>v$。$v$ 方向同理。结合上一个不等式，可以得到满足这个条件时，如果 $x_1,x_2$ 满足 $dis(x_1,x_2)=dis(u,v)$，则 $\min(x_1,x_2)\geq u,\max(x_1,x_2)\geq v$。因此这样选出的所有 $x$ 可以任意加入而不影响合法性。因此可以找出所有的 $x$，设有 $cnt$ 个，则 $(u,v)$ 的贡献次数为 $C_{cnt-2}^{m-2}$。

预处理出所有点对的距离即可。复杂度 $O((n+m^2)\log n+m^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105000
#define M 505
#define mod 998244353
int n,m,k,head[N],c[M][M],cnt,as,sz[N],v[N],is[N],f[N][18],de[N],a,b,d;
long long dep[N],dis[M][M];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void dfs(int u,int fa)
{
	f[u][0]=fa;for(int i=1;i<=17;i++)f[u][i]=f[f[u][i-1]][i-1];
	de[u]=de[fa]+1;sz[u]=is[u];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dep[ed[i].t]=dep[u]+ed[i].v;
		dfs(ed[i].t,u);sz[u]+=sz[ed[i].t];
		int tp=(1ll*c[m][k]-c[sz[ed[i].t]][k]-c[m-sz[ed[i].t]][k]+2ll*mod)%mod;
		as=(as+2ll*ed[i].v*tp)%mod;
	}
}
int LCA(int x,int y){if(de[x]<de[y])x^=y^=x^=y;for(int i=17;i>=0;i--)if(de[x]-(1<<i)>=de[y])x=f[x][i];if(x==y)return x;for(int i=17;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
long long getdis(int x,int y){return dep[x]+dep[y]-2*dep[LCA(x,y)];}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=m;i++)scanf("%d",&v[i]),is[v[i]]=1;
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&d),adde(a,b,d);
	for(int i=0;i<=m;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=m;i++)
	for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	dfs(1,0);
	for(int i=1;i<=m;i++)
	for(int j=1;j<=m;j++)dis[i][j]=getdis(v[i],v[j]);
	if(k>=2)
	for(int i=1;i<=m;i++)
	for(int j=i+1;j<=m;j++)
	{
		int ct=2;
		for(int l=1;l<=m;l++)
		{
			long long tp=max(dis[i][l],dis[j][l]);
			if(tp<dis[i][j]||(tp==dis[i][j]&&(dis[i][l]<dis[i][j]||l<j)&&(dis[j][l]<dis[i][j]||l<i)&&(l!=j&&l!=i)))ct++;
		}
		if(ct>=k)as=(as-dis[i][j]%mod*c[ct-2][k-2]%mod+mod)%mod;
	}
	printf("%d\n",1ll*as*pw(c[m][k],mod-2)%mod);
}
```



##### auoj99 划愤

###### Problem

(具体题面不见了)

求 $n$ 阶矩阵nim积行列式。

$n\leq 150,v_{i,j}<2^{64}$

###### Sol

Surrender



##### auoj100 树上的鼠

###### Problem

考虑在一棵无边权的树上进行博弈。有一个棋子，它初始在节点 $1$。双方轮流操作，每次操作时每个人需要将棋子从当前点移动到一个点。

记上次操作移动的距离为 $ls$（初始 $ls=0$），则这次操作棋子移动的距离必须严格大于 $ls$。不能操作的人输。

现在给一棵 $n$ 个点的树，求有多少个包含 $1$ 的连通块使得这个连通块上进行博弈，双方最优操作下先手胜。答案模 $998244353$

$n\leq 10^6$

$3s,512MB$

###### Sol

考虑一条链的情况，设 $1$ 两侧的长度为 $l,r$。

则如果 $l=r$，后手可以在每次先手操作后将棋子移动到当前位置关于 $1$ 的对称点。这样先手下一次只能向距离 $1$ 更大的点移动，从而后手一定可以对称移动。这样后手必胜。

而如果 $l\neq r$，先手第一次操作可以向较大的方向移动，使得除去这次移动的路径外两侧的长度相等。这之后先手就可以关于这一段对称移动，因此链上先手胜当且仅当初始点不是链中点。

考虑树的情况，类似地找到树的直径。如果初始点在直径中点，则后手在每次先手操作后，将棋子移动到直径另外一侧与当前距离相同的点。和上面一样可以得到后手必胜。

同样地，如果初始点不在直径中点，先手可以移动到直径上另外一侧与当前点和直径中点距离相同的点，然后使用后手刚才的操作。因此可以得到先手胜当且仅当 $1$ 不是直径中点。



$1$ 是直径中点当且仅当 $1$ 的子树中，有至少两个子树内的最大深度等于总的最大深度。即先手胜当且仅当存在一个 $1$ 的子树，使得这个子树内的深度大于其他所有子树内的深度。

因此如果对于每个子树，求出 $dp_{u,j}$ 表示 $u$ 为根的子树中所有包含根的连通块中最大深度为 $j$ 的连通块数量，则最后可以 $O(n)$ 计算答案。

此时问题变为给若干个子树，对每个子树算出根的 $dp$。考虑长链剖分的做法。

考虑合并两个子树的 $dp$，将深度小的子树的 $dp$ 合并到深度大的子树的 $dp$ 上，则转移为：

$$
f_i=\sum_{\max(j,k)=i}dp_{u,j}*dp_{v,k}
$$

可以发现这相当于对 $dp_u$ 的前若干位进行修改，对后面的位做一个后缀乘。因此可以单独维护后缀乘标记，需要用这一项时再传标记。

同时为了方便处理向上转移的过程，可以将 $dp_u$ 倒过来使用 `vector` 存，这样向上时只需要 `push_back(1)` 即可。

每个点只会合并一次，因此复杂度 $O(n)$。

~~还有一些奇妙的长链剖分写法，例如每次将一条长链一起转移然后用链表维护链上还有贡献的位置，复杂度也是线性~~

###### Code

奇妙的写法：

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 1005000
#define mod 998244353
int n,a,b,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
int le[N],sn[N],f[N];
void dfs1(int u,int fa)
{
	f[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs1(ed[i].t,u);
		if(le[ed[i].t]>le[u])le[u]=le[ed[i].t],sn[u]=ed[i].t;
	}
	le[u]++;
}
vector<int> dp[N];
int lb[N],nt[N],vl[N],vr[N],v0[N];
void dfs2(int u)
{
	vector<int> as;
	int nw=u;
	while(nw)
	{
		dp[nw].push_back(1);
		vector<int> ls;
		for(int i=head[nw];i;i=ed[i].next)if(ed[i].t!=f[nw]&&ed[i].t!=sn[nw])
		dfs2(ed[i].t),ls.push_back(ed[i].t);
		int v1=1;
		while(ls.size())
		{
			int vl=1;
			vector<int> s2;
			for(int i=0;i<ls.size();i++)
			{
				int v2=dp[ls[i]].back();dp[ls[i]].pop_back();
				if(dp[ls[i]].empty())v1=1ll*v1*v2%mod;
				else vl=1ll*vl*v2%mod,s2.push_back(ls[i]);
			}
			ls=s2;dp[nw].push_back(1ll*v1*vl%mod);
		}
		reverse(dp[nw].begin(),dp[nw].end());
		v0[nw]=1;vl[nw]=vr[nw]=1;nt[nw]=sn[nw];lb[nt[nw]]=nw;nw=sn[nw];
	}
	nw=u;
	while(nw)
	{
		int st=nw;
		while(st)
		{
			int v1=dp[st].back();dp[st].pop_back();
			v0[st]=v1;
			if(dp[st].empty()&&nt[st])
			{
				vl[st]=1ll*vl[st]*v0[st]%mod;v0[st]=1;
				nt[lb[st]]=nt[st];
				lb[nt[st]]=lb[st];
				vr[nt[st]]=(1ll*vr[nt[st]]*vl[st]+vr[st])%mod;
				vl[nt[st]]=1ll*vl[nt[st]]*vl[st]%mod;
			}
			st=lb[st];
		}
		int v1=1;
		for(int i=nw;i;i=lb[i])v1=(1ll*v1*vl[i]%mod*v0[i]+vr[i])%mod;
		as.push_back(v1);
		nw=sn[nw];
	}
	reverse(as.begin(),as.end());
	dp[u]=as;
}
int v1[N],as,sv=1;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs1(1,0);
	vector<int> ls;
	for(int i=head[1];i;i=ed[i].next)dfs2(ed[i].t),ls.push_back(ed[i].t),v1[ed[i].t]=1;
	while(ls.size())
	{
		vector<int> s2;
		int f1=1,f2=0;
		for(int i=0;i<ls.size();i++)
		{
			int v2=dp[ls[i]].back();dp[ls[i]].pop_back();
			f2=(1ll*f2*v1[ls[i]]+1ll*f1*(v2-v1[ls[i]]+mod))%mod;
			f1=1ll*f1*v1[ls[i]]%mod;
			v1[ls[i]]=v2;
		}
		as=(as+1ll*sv*f2)%mod;
		for(int i=0;i<ls.size();i++)
		if(!dp[ls[i]].size())sv=1ll*sv*v1[ls[i]]%mod;
		else s2.push_back(ls[i]);
		ls=s2;
	}
	printf("%d\n",as);
}
```



#### NOI2020模拟六校联测9

出题人:wkr

##### auoj101 鼠

###### Problem

给 $n+2$ 个点，前 $n$ 个点每个点有一条白色出边和一条黑色出边，出边一定连向编号更大的节点，最后两个点为终止节点。

多组询问，每次给出 $s,c,k$。询问如下问题的结果：

你初始在 $s$，颜色为 $c$。你会选择和自己颜色相同的边走，直到到达终止节点。你每经过 $k$ 条边就会改变颜色。求最后你停止的节点位置。

$n,q\leq 5\times 10^4$

$1s,512MB$

###### Sol

考虑根号分治。对于一个 $k$，可以先求出从一个点开始，向后走 $k$ 步同色边后到达的点，然后从大到小扫即可 $O(n)$ 求出这个 $k$ 所有可能的询问的答案。因此对于 $k\leq \sqrt n$ 的部分可以 $O(n\sqrt n)$ 求出答案。

对于 $k>\sqrt n$ 的询问，此时切换颜色的次数不超过 $\sqrt n$。一种做法是倍增向右走同色边的情况，这样的总复杂度为 $O(n\sqrt{n\log n})$

这里也可以使用使用分块快速幂的思想预处理，即可做到 $O(n\sqrt n)$。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 50050
#define K 305
int n,q,a,b,c,f[N][K][2],nt[N][2],g[N][16][2];
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d%d",&nt[i][0],&nt[i][1]);
	nt[n+1][0]=nt[n+1][1]=n+1;
	nt[n+2][0]=nt[n+2][1]=n+2;
	for(int i=1;i<=n+2;i++)f[i][1][0]=nt[i][0],f[i][1][1]=nt[i][1];
	for(int j=2;j<=300;j++)
	for(int i=1;i<=n+2;i++)
	f[i][j][0]=nt[f[i][j-1][0]][0],f[i][j][1]=nt[f[i][j-1][1]][1];
	for(int j=1;j<=300;j++)
	for(int i=n;i>=1;i--)
	f[i][j][0]=f[f[i][j][0]][j][1],f[i][j][1]=f[f[i][j][1]][j][0];
	for(int i=1;i<=n+2;i++)g[i][0][0]=nt[i][0],g[i][0][1]=nt[i][1];
	for(int j=1;j<=15;j++)
	for(int i=1;i<=n+2;i++)
	g[i][j][0]=g[g[i][j-1][0]][j-1][0],g[i][j][1]=g[g[i][j-1][1]][j-1][1];
	while(q--)
	{
		scanf("%d%d%d",&a,&b,&c);
		if(b<=300)printf("%d\n",f[a][b][c]-n-1);
		else
		{
			while(a<=n)
			{
				int st=b,vl=0;
				while(st){if(st&1)a=g[a][vl][c];vl++;st>>=1;}
				c^=1;
			}
			printf("%d\n",a-n-1);
		}
	}
}
```



##### auoj102 Arcahv

###### Problem

有 $2^n$ 个人，每个人有一个实力值 $v_i$，实力值构成 $2^n$ 阶排列。

所有人初始排成一列。接下来会进行 $n$ 轮比赛，在一轮比赛中，设当前有 $2^k$ 个人，将他们按顺序编号为 $1,2,\cdots,2^k$，则编号为 $2i-1,2i$ 的人会进行比赛，之后实力值低的人会被淘汰，剩下的人留下按照之前的顺序排成 $2^{k-1}$ 的一列。

有 $q$ 次询问，每次给定 $x,k$，求如下问题的答案：

你可以在原先的顺序上，进行不超过 $k$ 次交换，每次交换任意两个人的位置。求进行交换后，原来编号为 $x$ 的人最多能赢几场比赛。

询问独立，强制在线

$n\leq 19,q\leq 2\times 10^5$

$1s,16MB$

###### Sol

比赛过程可以看成线段树的形式，因此一个人能赢 $i$ 次当且仅当他的实力值是所在的长度为 $2^i$ 的区间中最大的。

考虑一个区间是否合法。设当前数为 $x$，考虑的区间为 $[l,r]$，则：

1. 如果 $x\in[l,r]$，则交换一定是将别的元素交换出去，即将更大的元素交换出去。因此合法当且仅当区间中大于 $v_x$ 的数数量不超过 $k$，且 $v_x\geq r-l+1$（否则找不到换进来的数）。
2. 如果 $x$ 不在区间内，则需要一次操作将 $x$ 交换进去，此时可以将 $x$ 和区间内最大的数交换，然后使用之前的方式。可以发现合法当且仅当满足上两个条件，且满足 $k>0$。

可以发现两种情况唯一的区别在于 $k=0$ 部分，因此特殊处理 $k=0$，此时不能进行交换，处理出初始时每个人赢的次数即可。

对于剩下的情况，所有区间等价，不需要再考虑当前的人是否在区间内。可以发现上述判断条件等价于区间内第 $k+1$ 大的数小于 $v_x$。因此只需要对于每种长度，对于每个 $k$ 求出所有这个长度的区间中的第 $k$ 大数的最小值即可维护询问。

考虑进行归并排序，这样对于每个区间求出区间内排序的结果。然后考虑记录上面需要的值。因为 $\sum_{i=1}^n 2^i=O(2^n)$，因此维护这个信息只需要 $O(2^n)$ 空间。

时间复杂度 $O(n2^n+nq)$，空间复杂度 $O(2^n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 524300
int n,k,q,t,a,b,las,fuc[N*2],tid[N],tr[N*2];
void solve(int l,int r)
{
	if(l==r)return;
	int mid=(l+r)>>1;
	solve(l,mid);solve(mid+1,r);
	int l1=l,r1=mid+1;
	for(int i=l;i<=r;i++)
	if(l1>mid)tr[n+i]=tr[r1++];
	else if(r1>r)tr[n+i]=tr[l1++];
	else if(tr[l1]<tr[r1])tr[n+i]=tr[l1++];
	else tr[n+i]=tr[r1++];
	for(int i=l;i<=r;i++)tr[i]=tr[n+i];
	for(int i=l;i<=r;i++)fuc[r-l+1+(r-i)]=min(fuc[r-l+1+(r-i)],tr[i]);
}
int query(int x,int y)
{
	int as=0;
	if(y)
	for(int i=1;i<=k;i++)
	{
		if((1<<i)>x)break;
		if((1<<i)<=y){as=i;continue;}
		if(fuc[y+(1<<i)]<=x)as=i;
		else break;
	}
	else
	{
		int st=tid[x]+n-1;
		for(int i=1;i<=k;i++)
		{
			st>>=1;
			if(tr[st]<=x)as=i;
		}
	}
	return as;
}
int main()
{
	scanf("%d%d%d",&n,&k,&t);
	for(int i=1;i<=n;i++)scanf("%d",&tr[i]),tid[tr[i]]=i;
	scanf("%d",&q);
	for(int i=0;i<=n;i++)fuc[i]=n+1;
	solve(1,n);
	for(int i=0;i<=n*2;i++)tr[i]=0;
	for(int i=1;i<=n;i++)
	{
		int st=tid[i]+n-1;
		while(st)tr[st]=i,st=st>>1;
	}
	for(int i=1;i<=q;i++)scanf("%d%d",&a,&b),a^=t*las,b^=t*las,printf("%d\n",las=query(a,b));
}
```



##### auoj103 记忆

###### Problem

有 $n$ 株草，初始时它们的高度都是 $0$。

在第 $i$ 天的早晨，第 $i$ 株草会长高 $i$ 的高度。

有一个长度为 $m$ 的非负整数序列 $h_i$，在第 $i$ 天的傍晚，你会将所有高度高于 $h_{((i-1)\bmod m)+1}$ 的草全部割到 $h_{((i-1)\bmod m)+1}$ 高度。每割一株草代价为 $1$。

有 $q$ 次询问，每次给一个 $k$，求前 $k$ 天的总代价。

$n,m,q\leq 3\times 10^5,k\leq 10^{12},h_i\leq 10^{18}$

$2s,512MB$

###### Sol

将每 $m$ 天称为一轮。考虑第 $k$ 株草在什么时候会被割，则有如下性质：

1. 如果这株草在某一轮被割过了，则之后的每一轮中，所有 $h_i$ 最小的操作中它都会被割。

因为草的增长速度大于 $0$，因此这是显然的。接下来考虑一个最小的 $h_i$ 之后的情况。不妨设 $h_1$ 是一个最小值，容易得到如下性质：

2. 在之后的每一轮中，第 $i$ 次操作会割这株草当且仅当 $h_i<h_j+k*(i-j),\forall 1\leq j<i$。

在一般情况下，容易得到类似的判断条件，即第 $i$ 次操作会割这株草当且仅当 $h_i<h_{((i-j+m-1)\bmod m)+1}+k*i,\forall 1\leq j<m$。

再考虑第一次被割的这一轮，有如下性质：

3. 第一次割这株草的操作一定满足上一条给出的条件，之后一次操作会割这株草当且仅当它在一轮中满足上一个条件。

第一次割的操作为第一个满足 $h_{((i-1)\bmod m)+1}<k*i$ 的位置，即第一个满足 $h_{((i-1)\bmod m)+1}-k*i<0$ 的位置。因此它前面的位置中这个值都大于等于 $0$。但前面可能不到一轮操作，此时注意到初始高度为 $0$，因此考虑向 $i<0$ 部分延伸，这部分一定满足 $h_{((i-1)\bmod m)+1}-k*i\geq 0$。因此这个操作前 $m$ 次操作都满足这个限制，因此它满足上一个性质。

之前说明了，在之后的每一轮中只会在满足上一个条件的操作处割，而第一次操作满足上一个条件，这次操作后可以看成变成了一轮中的某个状态，因此之后的情况一定和上述情况相同。



首先考虑每次操作在哪些 $k$ 满足条件。可以发现满足条件的 $k$ 一定是一段后缀，下界可以看成环上前面的点到它的最大斜率。因此可以在环上扫两遍求凸包求出最大斜率。这样可以求出 $l_i$，表示一轮中第 $i$ 次操作在 $l_i\leq k$ 的时候会割到第 $k$ 株草。

然后考虑求出第一次割一株草的时刻。由上一个性质可以发现如下结果：

4. 对于第 $k$ 株草，它会被第 $i$ 次操作割，当且仅当这次操作满足条件（即 $li_{((i-1)\bmod m)+1}\leq k$），同时操作满足 $k*i>h_{((i-1)\bmod m)+1}$。

证明：如果满足后面一个条件，则说明如果前面没有割过这株草，则这次会割这株草。因此第一次割这株草的操作以及之后的操作满足这个条件。因此由性质 $3$ 可以得到两个条件的总和是充分必要的。

因此对于一个 $k$，可以在满足条件的操作上二分找第一个时刻（可以先求出经过多少轮一定满足条件，然后在前一轮内二分）。然后从小到大考虑 $k$，这样可以看成依次加入合法操作。直接的做法是用权值线段树维护加入的合法操作，在线段树上二分。另外一种做法是倒过来看成删除位置，然后维护每个位置的下一个合法位置，这可以并查集实现，复杂度不超过1log。

这样即可在 $O((n+m)\log m)$ 的复杂度内求出 $l_i$ 以及每株草第一次被割的时间。显然时间满足如下性质：

1. 设第 $i$ 株草第一次被割是在第 $t_i$ 次操作，则 $t_i$ 单调不增。

由增长速度显然可以得到这个结论。因此在一个时刻，被割过的草为一段后缀。

考虑一段时间，设这段时间内被割过的草没有改变，为 $[r,n]$。则这段时间内，由性质 $3$，一轮中的第 $t$ 次操作会割最后的 $\min(n-r+1,n-l_t+1)$ 株草。因此这段时间内的代价可以快速求出。

因此将所有询问时刻和草第一次被割的时刻排序，依次处理每一段。则只需要支持 $r$ 增加 $1$，求一段区间的 $\min(n-r+1,n-l_t+1)$ 之和。如果将这个贡献看成一个关于 $r$ 的一次函数，则它只会在 $r=l_t$ 时改变一次（$n-r+1\to n-l_t+1$）。因此树状数组维护这个贡献即可得到每一段内的总代价。

复杂度 $O((n+m+q)\log m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,m,q;
ll v[N*2],qu[N],as[N];
ll sr[N];
int su[N];
vector<int> ls[N];
void calc_sr()
{
	int st[N],ct=0,nw=1;
	for(int i=1;i<=m;i++)
	{
		while(ct>1&&(__int128)(v[st[ct]]-v[st[ct-1]])*(i-st[ct-1])>=(__int128)(v[i]-v[st[ct-1]])*(st[ct]-st[ct-1]))ct--;
		st[++ct]=i;
	}
	for(int i=1;i<=m;i++)v[i+m]=v[i];
	for(int i=1;i<=m;i++)
	{
		while(ct>1&&(__int128)(v[st[ct]]-v[st[ct-1]])*(m+i-st[ct-1])>=(__int128)(v[i+m]-v[st[ct-1]])*(st[ct]-st[ct-1]))ct--;
		st[++ct]=i+m;
		sr[i]=(v[st[ct]]-v[st[ct-1]])/(st[ct]-st[ct-1])+1;
		if(sr[i]<=0)sr[i]=1;if(sr[i]>n)sr[i]=n+1;
	}
	for(int i=1;i<=m;i++)ls[sr[i]].push_back(i),su[sr[i]]++;
	for(int i=1;i<=n;i++)su[i]+=su[i-1];
}
ll ti[N];
int nt[N];
int finds(int x){return nt[x]==x?x:nt[x]=finds(nt[x]);}
void calc_ti()
{
	for(int i=0;i<=m+1;i++)nt[i]=i;
	for(int i=n;i>=1;i--)
	{
		for(int j=0;j<ls[i+1].size();j++)nt[ls[i+1][j]]=ls[i+1][j]+1;
		int lx=finds(1);
		ll li=(v[lx]-1ll*lx*i+1ll*i*m)/(1ll*i*m);
		ti[i]=li*m+lx;
		li--;
		int lb=1,rb=m,as=m+1;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			int ri=finds(mid);
			if(ri>m||(li*m+ri)*i>v[ri])as=ri,rb=mid-1;
			else lb=mid+1;
		}
		if(as<=m)ti[i]=li*m+as;
	}
}
struct BIT{
	ll tr[N];
	void add(int x,int k){for(int i=x;i<=m;i+=i&-i)tr[i]+=k;}
	ll que(int x){ll as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}t1,t2;
void add(int x,int v1,int v2){t1.add(x,v1);t2.add(x,v2);}
ll que(int l,int r,int v){return t1.que(r)-t1.que(l-1)+v*(t2.que(r)-t2.que(l-1));}
ll calc(ll l,ll r,int v)
{
	if(l>r)return 0;
	ll as=que(1,m,v)*((r-l)/m);
	int lb=(l-1)%m+1,rb=(r-1)%m+1;
	if(lb<=rb)as+=que(lb,rb,v);
	else as+=que(lb,m,v)+que(1,rb,v);
	return as;
}
pair<ll,int> tp[N*2];
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=m;i++)scanf("%lld",&v[i]);
	calc_sr();calc_ti();
	for(int i=1;i<=q;i++)scanf("%lld",&qu[i]);
	for(int i=1;i<=n;i++)tp[i]=make_pair(ti[i],i);
	for(int i=1;i<=q;i++)tp[i+n]=make_pair(qu[i]+1,-i);
	sort(tp+1,tp+n+q+1);tp[0].first=1;
	for(int i=1;i<=m;i++)if(sr[i]<=n)add(i,0,1);
	int li=n;
	ll nw=0;
	for(int i=1;i<=n+q;i++)
	{
		nw+=calc(tp[i-1].first,tp[i].first-1,n-li);
		int id=tp[i].second;
		if(id<0)as[-id]=nw;
		else
		{
			for(int j=0;j<ls[id].size();j++)add(ls[id][j],n-id+1,-1);
			li--;
		}
	}
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```



#### NOI2020模拟六校联测8

出题人:zjk

##### auoj104 Number

###### Problem

给定 $p=998244353$ 和一个 $[1,p-1]$ 间的正整数 $x$。

有一个数 $a$，初始 $a=1$。接下来会进行 $m$ 轮操作，每轮操作包含 $n$ 次操作。有一个长度为 $n$ 的正整数序列 $t_i$，在每一轮的第 $i$ 次操作中，有 $\frac 12$ 的概率将 $a$ 变为 $a*x^{t_i}\bmod p$，有 $\frac 12$ 的概率不改变 $a$。

对于每一个 $x\in[1,p-1]$，如果最后 $a$ 可能为 $x$，则求出操作后 $a=x$ 的概率。

设 $e$ 为最小的正整数满足 $x^e\equiv 1(\bmod p)$，满足 $e\leq 5\times 10^4,n\leq 10^6,\sum t_i\leq 2\times 10^7$

$5s,512MB$

###### Sol

考虑看成多项式，那么每次操作相当于乘上 $\frac12(x^{t_i}+1)$，最后求长度为 $e$ 的循环卷积，即 $\bmod(x^e-1)$。

而 $x$ 是 $\bmod p$ 下的 $e$ 次单位根。因此考虑单位根反演，只需要求出一轮操作的结果，再对点值做 $m$ 次幂即可。

考虑一轮的情况，即对于每个 $k=0,\cdots,e-1$，求出 $\prod_{i=1}^n \frac12(x^{k*t_i}+1)$ 的结果。

注意到不同的 $t_i$ 只有 $O(\sqrt{\sum v_i})$ 种，而同一种 $t_i$ 的贡献可以一起计算。这样求点值的复杂度即为 $O(e*\sqrt{\sum v_i}*\log n)$。

然后考虑点值还原答案，即循环卷积的IDFT。使用bluestein算法即可。

具体来说，需要求的形式为：

$$
g_i=\sum_{j=0}^{e-1}f_jx^{-ij}
$$

而 $-ij=C_i^2+C_j^2-C_{i+j}^2$，因此可以写成：

$$
g_i=x^{C_i^2}\sum_{j=0}^{e-1}f_jx^{C_j^2}*x^{-C_{i+j}^2}
$$

后半部分可以看成一个差卷积，ntt即可。

复杂度 $O(e*\sqrt{\sum v_i}*\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 263000
#define mod 998244353
int n,m,l,k,v[1050000],ntt[N],rev[N],g[2][N*2],a[N],b[N],st[N],ct;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
pair<int,int> fu[N];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		for(int j=0;j<s;j+=i)
		for(int k=j,vl=0;k<j+(i>>1);k++,vl++)
		{
			int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][i+vl]%mod;
			ntt[k]=(v1+v2)%mod;
			ntt[k+(i>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=pw(s,t==0?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<18;j<<=1)
	{
		int tp=pw(3,(mod-1)/j),v2=1;
		if(i==0)tp=pw(tp,mod-2);
		for(int l=0;l<j>>1;l++)g[i][j+l]=v2,v2=1ll*v2*tp%mod;
	}
	scanf("%d%d%d",&n,&l,&k);
	int v1=k,v2=1;while(v1!=1)v1=1ll*v1*k%mod,v2++;
	m=v2;
	for(int i=0;i<m;i++)st[i]=1;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	int lb=1,w=k,w1=pw(w,mod-2);
	while(lb<=n)
	{
		int rb=lb;while(v[rb+1]==v[lb]&&rb<=n)rb++;
		int v1=pw(w,v[lb]),s1=1;
		for(int i=0;i<=m;i++)
		st[i]=1ll*st[i]*pw(s1+1,rb-lb+1)%mod,s1=1ll*s1*v1%mod;
		lb=rb+1;
	}
	for(int i=0;i<m;i++)st[i]=pw(st[i],l);
	for(int i=0;i<2*m;i++)a[i]=pw(w1,1ll*i*(i-1)/2%(mod-1))%mod;
	for(int i=0;i<m;i++)b[m-i]=1ll*st[i]*pw(w,1ll*i*(i-1)/2%(mod-1))%mod;
	int s=1;while(s<=m*3)s<<=1;
	dft(s,a,1);dft(s,b,1);for(int i=0;i<s;i++)a[i]=1ll*a[i]*b[i]%mod;dft(s,a,0);
	for(int i=0;i<m;i++)st[i]=1ll*a[i+m]*pw(w,1ll*i*(i-1)/2%(mod-1))%mod*pw(m,mod-2)%mod;
	for(int i=0,tp=1;i<m;i++,tp=1ll*tp*w%mod)if(st[i])fu[++ct]=make_pair(tp,st[i]);
	sort(fu+1,fu+ct+1);
	for(int i=1;i<=ct;i++)printf("%d %d\n",fu[i].first,fu[i].second);
}
```



##### auoj105 Module

###### Problem

交互题，有一个有理数 $\frac pq$，保证 $1\leq p,q\leq 10^9$。

你需要猜出这个有理数，你可以向交互库询问一个 $[10^9+1,10^{10}]$ 间的质数 $x$，交互库会返回 $\frac pq\bmod x$。你的询问次数不能超过 $5$ 次。

多组数据，$T\leq 10^5$

$4s,512MB$

###### Sol

考虑询问两个质数（例如 $10^9+7,10^9+9$），可以得到一个 $10^{18}$ 级别的数 $m$ 以及 $t=\frac pq\bmod m$。

如果存在两个不同分数 $\frac ab,\frac cd$ 在模 $m$ 下相等，则 $ad-bc\equiv 0(\bmod m)$，但 $m>(10^9)^2$，因此只要找到 $[1,10^9]$ 内的一组 $p,q$ 满足条件，则它一定是解。

要找到一组 $[1,10^9]$ 之间的 $p,q$，相当于找到一个 $[1,10^9]$ 之间的 $p$，使得 $p*t\bmod m\in[1,10^9]$。因为 $m$ 是两个大于 $10^9$ 质数的乘积，因此不会出现 $0$ 的情况。

则如果找到一个满足这个条件的 $p,q$，则它一定是实际上的解。

Sol1:Stern-Barcot Tree

上述结果相当于找到 $t,s$，使得 $p*t-m*s\in[0,10^9]$。这相当于 $\frac pm\geq \frac st$ 且 $\frac pm-\frac st\leq \frac{10^9}{t*m}$。

因此可以发现，如果 $\frac pm\geq \frac ab<\frac cd$ 且 $b\leq d$，则 $\frac cd$ 一定不优。

因此考虑在 Stern-Barcot Tree 上找 $\frac pm$ 的逼近的过程，可能的答案一定是逼近过程上的元素。

更进一步，可能的答案一定是逼近过程上转向位置的元素。在树上只会转向 $O(\log n)$ 次，直接二分每一次走的长度复杂度 $O(T\log^2 v)$，也可以直接计算每次走的长度，复杂度 $O(T\log v)$。

Sol2:类欧几里得算法

~~那个时候我还不会这东西~~

考虑找到最小的非负整数 $p$ 使得 $p*t\bmod m\in[1,10^9]$。

然后可以发现这是类欧的经典应用。

~~以下部分抄写自arc127f题解~~

给定若干组 $a,b,c,l,r$，求最小的非负整数 $t$ 使得 $(at+b)\bmod c\in[l,r]$。

显然 $a,b$ 可以对 $c$ 取模。考虑类欧的思路，如果能做到交换 $a,c$，则通过取模操作，问题可以在 $O(\log m)$ 时间内解决。

可以发现上述式子成立当且仅当存在非负整数 $x$ 使得 $at+b\in[cx+l,cx+r]$，而这个式子等价于 $cx\in[at+b-r,at+b-l]$。

此时 $b-r,b-l$ 可能是负数，但它大于 $-c$，因此解出的 $x$ 一定大于等于 $0$。因此可以找到一个非负整数 $k=\lceil\frac {r-b}a\rceil$，将限制变为 $cx\in[at+ak+b-r,at+ak+b-l]$，且满足此时 $t\geq 0$ 时的最小合法 $x$ 为原问题答案同时 $ak+b-r\geq 0$。

而最小的 $t$ 对应最小的 $x$，因此问题已经变为了类似形式。此时 $ak+b-r<a$，因此如果 $ak+b-l<a$，而问题变为与上面相同的形式。否则，存在 $v\in[l,r]$ 使得 $at+ak+b-v=a$，即 $a(t+k-1)+b=v$。而可以发现，此时在原问题中 $x=0$ 存在合法 $t$，而这即为最小 $t$。因此这种情况可以直接求出答案。对于之前的另外一种情况递归求即可。

可以发现这样的复杂度与gcd相同，为 $O(\log m)$。



这里直接用即可，复杂度 $O(T\log v)$。

###### Code

Sol1:

```cpp
#include "module.h"
#include<algorithm>
#define ll long long
ll mul(ll x,ll y,ll mod){ll tmp=(long double)x*y/mod;return (x*y-tmp*mod+mod)%mod;}
bool check(int a,int b)
{
	return 1;
}
bool check1(ll a,ll b,ll c,ll d){return (__int128)a*b>(__int128)c*d;}
std::pair<int,int> Solve()
{
	ll x,y;
	x=Query(1e9+7),y=Query(1e9+9);
	ll tp2=1000000007ll*1000000009,tp1=(mul(500000004ll*1000000009ll,x,tp2)+mul(1000000007ll*500000004ll,y,tp2))%tp2;
	ll lb1=0,rb1=1,lb2=1,rb2=1;
	while(rb1<=2e9&&rb2<=2e9)
	{
		ll t1=mul(rb1,tp1,tp2);if(t1<=1e9&&rb1<=1e9)if(check(t1,rb1)){return std::make_pair(t1,rb1);}
		ll f1=lb1+lb2,f2=rb1+rb2;
		if(check1(f1,tp2,f2,tp1))
		{
			int lb=1,rb=(1e9-rb2)/rb1,as=1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(check1(lb1*mid+lb2,tp2,tp1,rb1*mid+rb2))as=mid,lb=mid+1;
				else rb=mid-1; 
			}
			lb2+=lb1*as,rb2+=rb1*as;
		}
		else 
		{
			int lb=1,rb=(1e9-rb1)/rb2,as=1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(!check1(lb2*mid+lb1,tp2,tp1,rb2*mid+rb1))as=mid,lb=mid+1;
				else rb=mid-1; 
			}
			lb1+=lb2*as,rb1+=rb2*as;
		}
	}
}
```

Sol2:

```cpp
#include "module.h"
#include<algorithm>
#define ll long long
ll calc(ll a,ll b,ll c,ll l,ll r)
{
	a%=c;b%=c;l%=c;r%=c;
	if(b>=l&&b<=r)return 0;
	if(b<l&&(r-b)/a>(l-1-b)/a)return (l-1-b)/a+1;
	ll li=calc(c,0,a,b+a-r%a,b+a-l%a);
	return ((__int128)li*c+l-1-b)/a+1;
}
std::pair<int,int> Solve()
{
	ll x=Query(1e9+7),y=Query(1e9+9);
	ll m=1000000007ll*1000000009,k=((__int128)500000004*1000000009*x+(__int128)1000000007*500000004*y)%m;
	ll s=calc(k,0,m,1,1e9);
	return std::make_pair((__int128)s*k%m,s);
}
```

##### auoj106 Robot

###### Problem

在二维平面上有 $n$ 个障碍，每个障碍都形如一个凸包，第 $i$ 个障碍有 $v_i$ 个点组成。

有一个机器人，它的形状为一个 $k$ 个点的凸包，机器人只能平移不能旋转，移动过程中与障碍的交不能大于 $0$。称机器人的关键点为凸包上第一个点。

$q$ 次询问，每次给定两个点 $P,Q$，询问如果初始机器人的关键点在 $P$，将机器人的关键点移动到 $Q$ 需要的最小距离。

part #1:$\sum v_i\leq 50,k\leq 6,q\leq 200$

part #2:$n\leq 50,\sum v_i\leq 1000,k\leq 4,q\leq 25$

$8s,512MB$

###### Sol

考虑如何将机器人看成一个点。将机器人的关键点放到原点得到凸包 $S$，设一个障碍为凸包 $T$，则不碰到障碍时关键点不能在的位置为：

$$
\{P-Q|P\in T,Q\in S\}
$$

可以发现这是一个闵可夫斯基和的形式，因此求出来的结果仍然是凸包，可以 $O(\sum v_i+kn)$ 求出每一个凸包。现在机器人变为一个点，障碍仍然是若干个凸包。



考虑将所有凸包顶点和询问点看作关键点，则最后一条最短路显然是经过关键点的折线。设总的点数为 $s$，则 $s=\sum v_i+kn+2q$，这里 $s\leq 2500$。

如果求出了每一对关键点之间是否可以用直线段直接相连，则这可以看成 $s$ 个点的图上的最短路，做 $q$ 次 `dijkstra` 的复杂度为 $O(qs^2)$。

然后考虑如何求出这部分。枚举一个端点看作原点，此时问题变为有若干个凸包，询问原点到所有关键点的连线是否穿过凸包。可以拆成有若干个线段作为障碍，但需要注意处理端点的情况。但凸包经过了原点的情况需要特殊处理。

则问题可以看成，有若干条线段，求原点向一个角度走遇到的第一条线段。每条线段覆盖了一个极角区间。

考虑将环变为区间，然后区间上线段树分治。则对于覆盖到某个线段树区间上的所有线段，只需要得到它们的凸包即可。

这里可以将线段先按照斜率排序（注意细节），然后按照顺序加入，这样即可线性构建凸包，这部分复杂度 $O(s\log s)$。

然后考虑询问，直接的询问方式为在线段树上 $O(\log s)$ 个覆盖询问点的区间上二分凸包上的对应位置。但这里可以按顺序询问所有点，然后在每个凸包上维护当前的对应位置，则这个对应位置只需要向右移动。这部分复杂度也是 $O(s\log s)$。

这样就可以在 $O(s^2\log s)$ 内求出所有边是否存在。复杂度 $O(s^2(q+\log s))$。

凸包拆成点需要注意端点是否存在，因此接下来的整个过程都需要特别处理端点的问题，因此会导致巨量细节。~~因此我不确定std是不是对的~~

~~同时这个题卡实数运算，因此需要全程ll~~

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;
#define N 5000
#define ll long long
struct pt{ll x,y;}p[N];
int n,m,k,q,s,a,b,ct,is[N],vis[N];
double dis[N][N],dis2[N];
pt operator +(pt a,pt b){return (pt){a.x+b.x,a.y+b.y};}
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
pt operator *(pt a,ll b){return (pt){a.x*b,a.y*b};}
bool operator ==(pt a,pt b){return a.x==b.x&&a.y==b.y;}
ll cross(pt a,pt b){return a.x*b.y-a.y*b.x;}
ll mul(pt a,pt b){return a.x*b.x+a.y*b.y;}
vector<pt> fu[N],st;
vector<pt> Minkowski_sum(vector<pt> a,vector<pt> b)
{
	int s1=a.size(),s2=b.size();
	int as1=0,as2=0;
	for(int i=0;i<s1;i++)if(a[i].x<a[as1].x||(a[i].x==a[as1].x&&a[i].y<a[as1].y))as1=i;
	for(int i=0;i<s2;i++)if(b[i].x<b[as2].x||(b[i].x==b[as2].x&&b[i].y<b[as2].y))as2=i;
	vector<pt> as;
	as.push_back(a[as1]+b[as2]);
	int tp1=0,tp2=0;
	for(int i=1;i<s1+s2;i++)
	{
		pt f1=a[(tp1+as1)%s1],f2=a[(tp1+as1+1)%s1],f3=b[(tp2+as2)%s2],f4=b[(tp2+as2+1)%s2];
		int fg=0;
		if(tp1==s1)fg=0;
		else if(tp2==s2)fg=1;
		else fg=cross(f2-f1,f4-f3)<=0;
		if(fg)tp1++;else tp2++;
		as.push_back(a[(tp1+as1)%s1]+b[(tp2+as2)%s2]);
	}
	return as;
}
struct line{pt s,t;};
pt intersect(line a,line b)
{
	ll s1=cross(b.s-a.s,a.t-a.s)+cross(a.t-a.s,b.t-a.s),s2=cross(a.t-b.s,a.s-b.s);
	if(s1<0)s1*=-1,s2*=-1;
	pt v1=b.s*s1+(b.t-b.s)*s2;
	return v1;
}
bool checka(pt a,pt b){return (__int128)a.x*b.y<(__int128)a.y*b.x;}
bool check(line a,line b,line c)
{
	if(cross(b.t-b.s,c.t-c.s)==0)
	{
		if(cross(c.t-c.s,b.t-c.s)>0)return 1;
		return 0;
	}
	pt s1=intersect(a,b),s2=intersect(b,c);
	return checka(s2,s1);
}
bool checkb(line b,line c)
{
	if(cross(b.t-b.s,c.t-c.s)==0)
	if(cross(c.t-c.s,b.t-c.s)>0)return 1;
	return 0;
}
bool checkc(line b,line c)
{
	if(cross(b.t-b.s,c.t-c.s)==0)
	if(cross(c.t-c.s,b.t-c.s)<=0)return 1;
	return 0;
}
struct segt{
	struct node{int l,r,lb,rb;vector<line> fu;}e[N*4];
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].fu.clear();e[x].lb=e[x].rb=0;
		if(l==r)return;
		int mid=(l+r)>>1;
		build(x<<1,l,mid);
		build(x<<1|1,mid+1,r);
	}
	void insert(int x,int l,int r,line st)
	{
		if(e[x].l==l&&e[x].r==r)
		{
			while(e[x].rb-e[x].lb>=2&&check(e[x].fu[e[x].rb-2],e[x].fu[e[x].rb-1],st))e[x].rb--,e[x].fu.pop_back();
			if(e[x].rb-e[x].lb>=1&&checkb(e[x].fu[e[x].rb-1],st))e[x].rb--,e[x].fu.pop_back();
			if(!(e[x].rb-e[x].lb>=1&&checkc(e[x].fu[e[x].rb-1],st)))e[x].fu.push_back(st),e[x].rb++;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)insert(x<<1,l,r,st);
		else if(mid<l)insert(x<<1|1,l,r,st);
		else insert(x<<1,l,mid,st),insert(x<<1|1,mid+1,r,st);
	}
	bool query(int x,int v,pt st)
	{
		while(e[x].lb<e[x].rb-1&&checka(intersect(e[x].fu[e[x].lb],e[x].fu[e[x].lb+1]),st))e[x].lb++;
		if(e[x].rb>e[x].lb&&cross(e[x].fu[e[x].lb].s-st,e[x].fu[e[x].lb].t-st)>0)
		return 1;
		if(e[x].l==e[x].r)return 0;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=v)return query(x<<1,v,st);
		else return query(x<<1|1,v,st);
	}
}tr;
int f1[N],id[N],fid[N],vl[N][2],id4[N],wasted[N],f2[N],su1[N],fuc2[N];pt tp2[N];
line fuc[N];
bool cmp10(int a,int b){return cross(fuc[a].t-fuc[a].s,fuc[b].t-fuc[b].s)<0;}
bool cmp(int a,int b){return cross(tp2[a],tp2[b])<0;}
int sg(ll x){return x>0?1:x?-1:0;}
void justdoit(int x)
{
	int ct7=0,fg=0;
//	printf("----------------------------\n");
	for(int i=1;i<=ct;i++)fid[i]=fuc2[i]=0;
	for(int i=1;i<=ct;i++)if(i!=x)
	if(p[i].y>p[x].y||(p[i].y==p[x].y&&p[i].x<p[x].x))
	{
		f1[++ct7]=i;tp2[i]=p[i]-p[x];fg=1;
	}
	for(int i=1;i<=m;i++)
	for(int j=0;j<fu[i].size();j++)
	if(p[x]==fu[i][j])
	{
		pt v1=fu[i][j],v2=fu[i][(j+fu[i].size()-1)%fu[i].size()],v3=fu[i][(j+1)%fu[i].size()];
		for(int l=1;l<=ct;l++)
		if(cross(p[l]-v1,v2-v1)<0&&cross(p[l]-v1,v3-v1)>0)fuc2[l]=1;
	}
	int ct2=ct7;
	for(int i=1;i<=ct;i++)if(i!=x)
	if(p[i].y<p[x].y||(p[i].y==p[x].y&&p[i].x>p[x].x))
	{
		f1[++ct7]=i;tp2[i]=p[i]-p[x];
	}
	stable_sort(f1+1,f1+ct2+1,cmp);
	stable_sort(f1+ct2+1,f1+ct7+1,cmp);
	int ct3=0;
	for(int i=1;i<=ct7;i++)
	if(i>1&&cross(tp2[f1[i]],tp2[f1[i-1]])==0&&mul(tp2[f1[i]],tp2[f1[i-1]])>0)id[i]=ct3,fid[f1[i]]=ct3;
	else id[i]=++ct3,fid[f1[i]]=ct3;
	if(!ct3)return;
	tr.build(1,1,ct3);
	int ct4=0,su=n;
	for(int i=1;i<=m;su+=fu[i].size(),i++)
	for(int j=0;j<fu[i].size();j++)
	{
		int fg1=0;
		int v2=(j+1)%fu[i].size(),v3=j,v4=(j+fu[i].size()-1)%fu[i].size(),v1=(j+2)%fu[i].size();
		if(!fid[su+v2+1]||!fid[su+v3+1])continue;
		fuc[++ct4]=(line){fu[i][v2]-p[x],fu[i][v3]-p[x]};
		if(cross(fuc[ct4].s,fuc[ct4].t)>0)v1^=v4^=v1^=v4,v2^=v3^=v2^=v3,swap(fuc[ct4].s,fuc[ct4].t);
		id4[ct4]=ct4;vl[ct4][0]=fid[su+v2+1],vl[ct4][1]=fid[su+v3+1];
		if(vl[ct4][0]==vl[ct4][1])wasted[ct4]=1;
		else
		{
			if(sg(cross(fu[i][v2]-p[x],fu[i][v1]-fu[i][v2]))*sg(cross(fu[i][v2]-p[x],fu[i][v3]-fu[i][v2]))>=0)vl[ct4][0]=vl[ct4][0]==ct3?1:vl[ct4][0]+1,fg1=1;
			if(sg(cross(fu[i][v3]-p[x],fu[i][v2]-fu[i][v3]))*sg(cross(fu[i][v3]-p[x],fu[i][v4]-fu[i][v3]))>=0)vl[ct4][1]=vl[ct4][1]==1?ct3:vl[ct4][1]-1,fg1=1;
			if(vl[ct4][1]%ct3+1==vl[ct4][0]&&fg1)wasted[ct4]=1;
			else wasted[ct4]=0;
		}
	}
//	for(int i=1;i<=ct4;i++)if(!wasted[i])printf("(%lld,%lld)->(%lld,%lld) [%d,%d]\n",fuc[i].s.x,fuc[i].s.y,fuc[i].t.x,fuc[i].t.y,vl[i][0],vl[i][1]);
	int ct5=0;
	for(int i=1;i<=ct4;i++)
	{
		pt sb=fuc[i].t-fuc[i].s;
		if(!wasted[i]&&(sb.y>0||(sb.y==0&&sb.x>0)))f2[++ct5]=i;
	}
	sort(f2+1,f2+ct5+1,cmp10);
	for(int i=1;i<=ct5;i++)
	if(vl[f2[i]][0]>vl[f2[i]][1])tr.insert(1,1,vl[f2[i]][1],fuc[f2[i]]);
	else if(fuc[f2[i]].s.y>=0||(vl[f2[i]][0]==1&&fg))tr.insert(1,vl[f2[i]][0],vl[f2[i]][1],fuc[f2[i]]);
	ct5=0;
	for(int i=1;i<=ct4;i++)
	{
		pt sb=fuc[i].t-fuc[i].s;
		if(!wasted[i]&&!(sb.y>0||(sb.y==0&&sb.x>0)))f2[++ct5]=i;
	}
	sort(f2+1,f2+ct5+1,cmp10);
	for(int i=1;i<=ct5;i++)
	if(vl[f2[i]][0]<=vl[f2[i]][1])tr.insert(1,vl[f2[i]][0],vl[f2[i]][1],fuc[f2[i]]);
	ct5=0;
	for(int i=1;i<=ct4;i++)
	{
		pt sb=fuc[i].t-fuc[i].s;
		if(!wasted[i]&&(sb.y>0||(sb.y==0&&sb.x>0)))f2[++ct5]=i;
	}
	sort(f2+1,f2+ct5+1,cmp10);
	for(int i=1;i<=ct5;i++)
	if(vl[f2[i]][0]>vl[f2[i]][1])tr.insert(1,vl[f2[i]][0],ct3,fuc[f2[i]]);
	else if(fuc[f2[i]].s.y<0&&(vl[f2[i]][0]!=1||!fg))tr.insert(1,vl[f2[i]][0],vl[f2[i]][1],fuc[f2[i]]);
	for(int i=1;i<=ct7;i++)if(fid[f1[i]])
	if(fuc2[f1[i]]||tr.query(1,fid[f1[i]],tp2[f1[i]]))dis[f1[i]][x]=dis[x][f1[i]]=1e15;
	else dis[f1[i]][x]=dis[x][f1[i]]=sqrt((p[f1[i]].x-p[x].x)*(p[f1[i]].x-p[x].x)+(p[f1[i]].y-p[x].y)*(p[f1[i]].y-p[x].y));
}
void solve(int s,int t)
{
	for(int i=1;i<=ct;i++)dis2[i]=1e16,vis[i]=0;
	dis2[s]=0;
	for(int i=1;i<=ct;i++)
	{
		double as=1e17;int as2=0;
		for(int j=1;j<=ct;j++)if(!vis[j]&&dis2[j]<as)as=dis2[j],as2=j;
		vis[as2]=1;
		for(int j=1;j<=ct;j++)if(as+dis[as2][j]<dis2[j])dis2[j]=as+dis[as2][j];
	}
	if(dis2[t]>1e14)printf("-1\n");
	else printf("%.15lf\n",dis2[t]);
}
int main()
{
	scanf("%d%d%d%d",&k,&n,&m,&q);
	for(int i=1;i<=k;i++)scanf("%d%d",&a,&b),a*=-1,b*=-1,st.push_back((pt){a,b});
	for(int i=1;i<=n;i++)scanf("%d%d",&p[i].x,&p[i].y);ct=n;
	for(int i=1;i<=m;i++)
	{
		scanf("%d",&s);
		for(int j=1;j<=s;j++)scanf("%d%d",&a,&b),fu[i].push_back((pt){a,b});
		fu[i]=Minkowski_sum(fu[i],st);
		for(int j=0;j<fu[i].size();j++)p[j+ct+1]=fu[i][j];
		ct+=fu[i].size();
	}
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=m;j++)
	{
		int fg=0;
		for(int l=0;l<fu[j].size();l++)
		if(cross(fu[j][l]-p[i],fu[j][(l+1)%fu[j].size()]-p[i])>=0)fg=1;
		if(!fg)is[i]=1;
	}
	for(int i=1;i<=ct;i++)
	if(!is[i])justdoit(i);
	for(int i=1;i<=ct;i++)if(is[i])for(int j=1;j<=ct;j++)dis[i][j]=dis[j][i]=1e15;
	while(q--)scanf("%d%d",&a,&b),solve(a,b);
}
```



#### NOI2020模拟六校联测5

出题人:zyw

##### auoj107 查拉图斯特拉如是说

###### Problem

求 $\sum_{i=0}C_n^i f(i)$，答案模 $998244353$，其中 $f$ 是一个 $m$ 次多项式

$n\leq 10^9,m\leq 10^5,m\leq n$

$2s,512MB$

###### Sol

以下为原做法：

考虑 $i^k$ 的斯特林展开（以下 $S$ 表示第二类斯特林数）：

$$
\sum_{i=0}^nC_n^i\sum_{j=0}^mf_ji^j\\
=\sum_{i=0}^nC_n^i\sum_{j=0}^mf_j\sum_{k=0}^jS_j^k*k!*C_i^k\\
=\sum_{j=0}^mf_j\sum_{k=0}^jS_j^k*k!\sum_{i=0}^nC_n^iC_i^k\\
=\sum_{j=0}^mf_j\sum_{k=0}^jS_j^k*k!*C_n^k*2^{n-k}\\
$$

到这里可以 $O(m^2)$ 处理。接下来考虑用容斥展开 $S_j^k$，且可以发现 $k$ 可以枚举到 $m$ 而不改变答案（由 $S_j^k$），即：

$$
\sum_{j=0}^mf_j\sum_{k=0}^mk!*C_n^k*2^{n-k}*S_j^k\\
=\sum_{j=0}^mf_j\sum_{k=0}^mk!*C_n^k*2^{n-k}*\frac 1{k!}\sum_{i=0}^k(-1)^{k-i}C_k^ii^j\\
=\sum_{j=0}^mf_j\sum_{k=0}^mC_n^k*2^{n-k}\sum_{i=0}^k(-1)^{k-i}C_k^ii^j\\
=\sum_{k=0}^mC_n^k*2^{n-k}\sum_{i=0}^k(-1)^{k-i}C_k^i\sum_{j=0}^mf_ji^j\\
$$ 

$ans=\sum_{k=0}^mC_n^k2^{n-k}\sum_{i=1}^k(-1)^{k-i}C_k^i\sum_{j=0}^mf_jj^i$

前面可以NTT，只需要求出所有的 $\sum f_jj^i$ ，多点求值即可

复杂度 $O(m\log^2 m)$

但事实上有更简单的方式得到类似结果。注意到 $m$ 次多项式一定可以和 $m$ 次下降幂多项式互相转换。考虑对下降幂多项式进行上面的操作，则对于一项 $C_i^x$ 有：

$$
\sum_{j=0}^nC_n^jC_j^x=2^{n-x}C_n^x
$$

因此对 $m$ 次下降幂多项式的点值做题目中的变换后，得到的仍然是 $m$ 次下降幂多项式。因此最后的结果是关于 $n$ 的 $m$ 次多项式。

因此考虑拉格朗日插值，可以得到如下结果：

$$
ans=\sum_{i=0}^m(\prod_{0\leq j\leq m,j\leq i}^m (n-j))*((-1)^{m-i}i!(m-i)!)*(\sum_{j\leq i}C_i^jf(j))\\
$$

复杂度也是 $O(m\log^2 m)$ 

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 263000
#define mod 998244353
int n,m,v[N],a[N],b[N],c[N],d[N],e[N],ntt[N],rev[N],v2[N],g[2][N*2],as[N],fr[N],ifr[N],las;
vector<int> st[N],fu;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	if(las!=s)for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1));
	las=s;
	for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		for(int j=0;j<s;j+=i)
		for(int k=j,vl=0;k<j+(i>>1);k++,vl++)
		{
			int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][i+vl]%mod;
			ntt[k]=v1+v2-(v1+v2>=mod)*mod;
			ntt[k+(i>>1)]=v1-v2+(v2>v1)*mod;
		}
	}
	int inv=pw(s,t==0?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void polyinv(int n,int *s,int *t)
{
	if(n==1){t[0]=pw(s[0],mod-2);return;}
	polyinv((n+1)>>1,s,t);
	int l=1;while(l<=n*1.5+2)l<<=1;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=0;i<n;i++)a[i]=t[i],b[i]=s[i];
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=(2ll*a[i]-1ll*a[i]*a[i]%mod*b[i]%mod+mod)%mod;dft(l,a,0);
	for(int i=0;i<n;i++)t[i]=a[i];
}
vector<int> polymod(vector<int> a,vector<int> b)
{
	int s1=a.size()-1,s2=b.size()-1;
	if(s1<s2)return a;
	int tp=s1-s2+1;
	for(int i=0;i<tp;i++)c[i]=d[i]=0;
	for(int i=0;i<=s2;i++)c[s2-i]=b[i];
	polyinv(tp,c,d);
	int l=1;while(l<=tp*2)l<<=1;
	for(int i=tp;i<l;i++)c[i]=d[i]=0;
	for(int i=0;i<tp;i++)c[i]=a[s1-i];
	dft(l,c,1);dft(l,d,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*d[i]%mod;dft(l,c,0);
	for(int i=0;i<tp;i++)e[tp-1-i]=c[i];
	l=1;while(l<=s1)l<<=1;
	for(int i=tp;i<l;i++)e[i]=0;
	for(int i=0;i<l;i++)c[i]=0;
	for(int i=0;i<=s2;i++)c[i]=b[i];
	dft(l,c,1);dft(l,e,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*e[i]%mod;dft(l,c,0);
	for(int i=0;i<=s1;i++)a[i]=(a[i]-c[i]+mod)%mod;
	while(a.size()>1&&a[a.size()-1]==0)a.pop_back();
	return a;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size()-1,s2=b.size()-1;
	int l=1;while(l<=s1+s2)l<<=1;
	for(int i=0;i<l;i++)c[i]=d[i]=0;
	for(int i=0;i<=s1;i++)c[i]=a[i];
	for(int i=0;i<=s2;i++)d[i]=b[i];
	dft(l,c,1);dft(l,d,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*d[i]%mod;dft(l,c,0);
	vector<int> f;
	for(int i=0;i<=s1+s2;i++)f.push_back(c[i]);
	return f;
}
void pre(int x,int l,int r)
{
	if(l==r){st[x].push_back(mod-v2[l]);st[x].push_back(1);return;}
	int mid=(l+r)>>1;
	pre(x<<1,l,mid);pre(x<<1|1,mid+1,r);
	st[x]=polymul(st[x<<1],st[x<<1|1]);
}
void solve(int x,int l,int r,vector<int> fu)
{
	if(l==r){as[l]=fu[0];return;}
	int mid=(l+r)>>1;
	solve(x<<1,l,mid,polymod(fu,st[x<<1]));
	solve(x<<1|1,mid+1,r,polymod(fu,st[x<<1|1]));
}
void pre()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=m;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<18;j<<=1)
	{
		int tp=pw(3,(mod-1)/j),v2=1;
		if(i==0)tp=pw(tp,mod-2);
		for(int l=0;l<j>>1;l++)g[i][j+l]=v2,v2=1ll*v2*tp%mod;
	}
}
int main()
{
	scanf("%d%d",&n,&m);pre();
	for(int i=0;i<=m;i++)scanf("%d",&v[i]),fu.push_back(v[i]);
	for(int i=0;i<=m;i++)v2[i]=i;
	pre(1,0,m);solve(1,0,m,fu);
	int l=1;while(l<=m*2)l<<=1;
	for(int i=0;i<l;i++)c[i]=d[i]=0;
	for(int i=0;i<=m;i++)c[i]=1ll*as[i]*ifr[i]%mod,d[i]=1ll*ifr[i]*(i&1?mod-1:1)%mod;
	dft(l,c,1);dft(l,d,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*d[i]%mod;dft(l,c,0);
	int tp=1,as1=0;
	for(int i=0;i<=m;i++)
	{
		as1=(as1+1ll*tp*pw(2,n-i)%mod*c[i])%mod;
		tp=1ll*tp*(n-i)%mod;
	}
	printf("%d\n",as1);
}
```



##### auoj108 橡树上的逃亡

###### Problem

给一棵 $n$ 个点的有根树，设 $i$ 的父亲节点为 $f_i$，保证 $f_i<i$。并且对于 $u<v<w$ ，如果 $w$ 在 $u$ 子树内，那么 $v$ 也在 $u$ 子树内。

$q$ 组询问，每次询问给定一个区间 $[l,r]$，求在编号在 $[l,r]$ 内的所有叶子中随机选择 $k$ 次（可以重复），包含选中的点的最小连通块的边数的期望，答案模 $998244353$

强制在线

$n,q\leq 2\times 10^5,k\leq 10$

$2s,1024MB$

###### Sol

条件相当于每个点的子树内的点编号构成一段连续的区间。

对于一个点 $x$，考虑 $x$ 到父亲的边的贡献。设询问区间的叶子数为 $a$，$x$ 子树内叶子数为 $b$，那么贡献为 $\frac 1 {a^k}(a^k-b^k-(a-b)^k)$，对于一个询问 $a$ 为定值，后面部分是一个关于 $a,b$ 的 $k$ 次多项式，因此只需要对于一组询问求出每条边的 $\sum b^i(0\leq i\leq k)$，即可 $O(k)$ 得到答案。

设 $su_i$ 表示前 $i$ 个点中有叶子的数量，$[l_i,r_i]$ 表示第 $i$ 个点的子树区间，询问区间为 $[l,r]$。考虑子树内叶子数，如果 $[l_i,r_i]\cap [l,r]=\emptyset$ 则叶子数量为 $0$，且这种情况没有贡献可以不考虑，否则设 $[l_i,r_i]\cap [l,r]=[s,t]$，则子树内叶子数为 $su_t-su_{s-1}$。

找到询问区间中最左侧的叶子 $l$ 和最右侧的叶子 $r$。显然可以将询问区间换成 $[l,r]$。设 $u=lca(l,r)$。则：

1. 对于路径 $(lca,l)$ 上的点 $x$，它的区间为 $[l,r_x]$
2. 对于路径 $(lca,r)$ 上的点 $x$，它的区间为 $[l_x,r]$
3. 对于被 $[l,r]$ 完全包含的子树的根，它们的区间为 $[l_x,r_x]$。

考虑第三部分的贡献，考虑对于每个点，所有 $i=0,\cdots,k$ 记录这个点子树内所有点的区间内叶子数和的 $i$ 次方和。

考虑暴力的计算方式，可以两侧一起向上跳直到到达 `lca`，$l$ 向上一步的贡献为父亲节点在 $l$ 右侧的子树内的贡献，$r$ 向上一步的贡献为父亲节点在 $r$ 左侧的子树内的贡献，`lca` 处特判。

那么可以对于每条边求出这条边沿着两种情况向上分别的贡献，询问时链上前缀和即可。可以发现 `lca` 处会正好将每个子树多算一次，可以用上述信息处理。

考虑链上前两部分的贡献，对于第一部分，相当于询问链上的 $\sum(su_{r_x}-su_{l-1})^k$ 之和。拆开后只需要维护树上 $su_{r_x}^k$ 的前缀和即可。另外一个方向只需要类似的维护 $su_{l_x-1}^k$ 的前缀和。

但这样直接合并复杂度为 $O(k^2)$ ~~虽然std就是这个~~，但如果将两部分贡献一起考虑，则贡献系数形如 $\sum_{i\leq j\leq k} C_j^iC_k^jp^{j-i}q^{k-j}$，而这等于 $C_k^i(p+q)^{k-i}$。因此单次询问复杂度可以变为 $O(k)$。

复杂度 $O(nk+qk+q\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<set>
#include<algorithm>
using namespace std;
#define N 200500
#define K 11
#define mod 998244353
int ty,ls,n,q,f[N],l,r,k;
vector<int> sn[N];
int is[N],su[N],lb[N],rb[N];
set<int> si;
int dep[N],f1[N][18];
void init_lca()
{
	for(int i=2;i<=n;i++)dep[i]=dep[f[i]]+1,f1[i][0]=f[i];
	for(int i=2;i<=n;i++)for(int j=1;j<=17;j++)f1[i][j]=f1[f1[i][j-1]][j-1];
}
int getlca(int x,int y)
{
	if(dep[x]<dep[y])x^=y^=x^=y;
	for(int i=17;i>=0;i--)if(dep[x]-dep[y]>=(1<<i))x=f1[x][i];
	if(x==y)return x;
	for(int i=17;i>=0;i--)if(f1[x][i]!=f1[y][i])x=f1[x][i],y=f1[y][i];
	return f1[x][0];
}
int sl[N][K],tl[N][K],sr[N][K],tr[N][K];
int v1[N][K],sv[N][K];
int ts[K],c[K][K];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int query(int x,int y,int k)
{
	int as=0;
	if(su[x-1]==su[y])return 0;
	x=*si.lower_bound(x);
	y=*(--si.lower_bound(y+1));
	int l=getlca(x,y),sz=su[y]-su[x-1],vt=pw(sz,k);
	for(int i=0;i<=k;i++)ts[i]=(3ll*mod+tl[x][i]+tr[y][i]-tl[l][i]-tr[l][i]+v1[l][i]-sv[l][i])%mod;
	as=(as+1ll*vt*ts[0]+mod-ts[k])%mod;
	int tp=1;
	for(int i=0;i<=k;i++)as=(as+1ll*ts[k-i]*tp%mod*((k-i+1)&1?mod-1:1)%mod*c[k][i])%mod,tp=1ll*tp*sz%mod;
	for(int i=0;i<=k;i++)ts[i]=(sl[x][i]+mod-sl[l][i])%mod;
	as=(as+1ll*vt*ts[0])%mod;
	tp=1;for(int i=0;i<=k;i++)as=(as+mod-1ll*ts[k-i]*tp%mod*c[k][i]%mod)%mod,tp=1ll*tp*(mod-su[x-1])%mod;
	tp=1;for(int i=0;i<=k;i++)as=(as+1ll*ts[k-i]*tp%mod*((k-i+1)&1?mod-1:1)%mod*c[k][i])%mod,tp=1ll*tp*(sz+su[x-1])%mod;
	for(int i=0;i<=k;i++)ts[i]=(sr[y][i]+mod-sr[l][i])%mod;
	as=(as+1ll*vt*ts[0])%mod;
	tp=1;for(int i=0;i<=k;i++)as=(as+mod-1ll*ts[k-i]*tp%mod*c[k][i]%mod)%mod,tp=1ll*tp*su[y]%mod;
	tp=1;for(int i=0;i<=k;i++)as=(as+1ll*ts[k-i]*tp%mod*((k-i+1)&1?mod-1:1)%mod*c[k][i])%mod,tp=1ll*tp*(sz+mod-su[y])%mod;
	return 1ll*as*pw(sz,mod-1-k)%mod;
}
int main()
{
	scanf("%d%d%d",&ty,&n,&q);
	for(int i=2;i<=n;i++)scanf("%d",&f[i]),is[f[i]]=0,sn[f[i]].push_back(i),is[i]=1;
	for(int i=1;i<=n;i++)
	{
		su[i]=su[i-1];
		if(is[i])su[i]++,si.insert(i);
	}
	init_lca();
	for(int i=1;i<=n;i++)lb[i]=rb[i]=i;
	for(int i=n;i>1;i--)rb[f[i]]=max(rb[f[i]],rb[i]);
	for(int i=1;i<=n;i++)
	{
		v1[i][0]=sl[i][0]=sr[i][0]=1;
		for(int j=1;j<=10;j++)
		v1[i][j]=1ll*v1[i][j-1]*(su[rb[i]]-su[lb[i]-1])%mod,
		sl[i][j]=1ll*sl[i][j-1]*su[rb[i]]%mod,
		sr[i][j]=1ll*sr[i][j-1]*(mod-su[lb[i]-1])%mod;
	}
	for(int i=n;i>=1;i--)
	for(int j=0;j<=10;j++)
	sv[i][j]=(sv[i][j]+v1[i][j])%mod,sv[f[i]][j]=(sv[f[i]][j]+sv[i][j])%mod;
	for(int i=1;i<=n;i++)if(sn[i].size())
	{
		for(int j=0;j+1<sn[i].size();j++)
		for(int k=0;k<=10;k++)
		tr[sn[i][j+1]][k]=(tr[sn[i][j]][k]+sv[sn[i][j]][k])%mod;
		for(int j=sn[i].size()-1;j>=1;j--)
		for(int k=0;k<=10;k++)
		tl[sn[i][j-1]][k]=(tl[sn[i][j]][k]+sv[sn[i][j]][k])%mod;
	}
	for(int i=1;i<=n;i++)for(int j=0;j<=10;j++)
	sl[i][j]=(sl[i][j]+sl[f[i]][j])%mod,
	sr[i][j]=(sr[i][j]+sr[f[i]][j])%mod,
	tl[i][j]=(tl[i][j]+tl[f[i]][j])%mod,
	tr[i][j]=(tr[i][j]+tr[f[i]][j])%mod;
	for(int i=0;i<=10;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=10;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	while(q--)
	{
		scanf("%d%d%d",&l,&r,&k);l^=ty*ls;r^=ty*ls;
		printf("%d\n",ls=query(l,r,k));
	}
}
```



##### auoj109 苏菲的世界

###### Problem

给定三维空间中 $n$ 个球，求它们的体积并。误差不超过 $10^{-3}$。

$n\leq 20,|x|,|y|,|z|,r\leq 10^3$

$3s,512MB$

###### Sol

Simpson积分+求圆并

在路上了

#### SCOI2020模拟13

##### auoj110 矩阵

###### Problem

给一个 $n$ 行 $m$ 列的矩阵，对于第 $i$ 列，如果这一列有奇数个 $1$，这一列的价值为 $a_i3^{b_i}(|a_i|=1,1\leq b_i\leq v)$，否则这一列的价值为 $0$。保证所有的 $(a_i,b_i)$ 两两不同且都是整数。

你可以删去矩阵的任意多行，求矩阵的最大价值。

$n\leq 2\times 10^5,v\leq 35,m\leq 70$

$2s,512MB$

###### Sol

考虑看成有 $2v$ 行，它们的权值分别为 $3^v,-3^v,3^{v-1},\cdots,1,-1$。

最后只关心每一列是否有奇数个 $1$，因此这可以看成行之间进行异或。因此考虑求出行之间组成的线性基，只保留不超过 $2n$ 行，它们可以组出原先的行的所有情况。

注意到 $\sum_{j<i}3^j-(-3^j)<3^i$，所以从大到小考虑 $3^i$，每次尽量让当前 $3^i$ 的系数最大，这样得到的一定是最优解。

如果在线性基中优先消前面的位，则由于线性基是上三角矩阵，因此得到的结果不会出现权值（的绝对值）小的位影响权值大的位，这样从大到小考虑就不需要考虑线性基中后面的元素的影响。

考虑当前最大的两位 $3^v,-3^v$。考虑线性基上这两位的元素，有以下情况：

1. 两位上都没有元素。此时这两位的情况固定，可以直接跳过这一位。
2. 两位上都有元素。此时无论前面的选择如何，这两位一定有唯一的方式在这两位上异或出 $10$，而这是唯一的最优方式。因此这两个元素有唯一的选择方式。
3. 只有一位上有元素。此时有两种异或方式。如果两种方式的结果不同，则最优方式唯一，这种情况和上面情况类似。但还有可能这个元素是否异或不改变这一位的贡献。此时可以看成这个元素可以任意选择。那么可以考虑将这个元素删去这两位加入之后的线性基，操作后的线性基和之前等效。

这部分最多只会插入 $m$ 次，因此复杂度为 $O(\frac{v^3}{\omega})$。

复杂度 $O(\frac{v^2(n+v)}{\omega})$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 75
#define M 200500
#define ll __int128
int n,m,k=35,a,b,id[N];
ll as=0;
char s[M][N];
ll si[N];
void ins(ll v)
{
	for(int i=1;i<=k*2;i++)if(v&((ll)1<<i-1))
	if(!si[i]){si[i]=v;return;}
	else v^=si[i];
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),id[i]=k*2-b*2+(a==1?1:2);
	for(int i=1;i<=n;i++)
	{
		ll v=0;
		for(int j=1;j<=m;j++)if(s[i][j]=='1')v|=(ll)1<<id[j]-1;
		ins(v);
	}
	ll tp=1,ls=0;
	for(int i=1;i<=k;i++)tp*=3;
	for(int i=1;i<=k;i++,tp/=3)
	if(!si[i*2-1]&&!si[i*2])
	{
		int vl=(ls>>(i*2-2))&3;
		if(vl==1)as+=tp;if(vl==2)as-=tp;
	}
	else if(si[i*2-1]&&si[i*2])
	{
		if(~(ls>>(i*2-2))&1)ls^=si[i*2-1];
		if((ls>>(i*2-1))&1)ls^=si[i*2];
		int vl=(ls>>(i*2-2))&3;
		if(vl==1)as+=tp;if(vl==2)as-=tp;
	}
	else
	{
		ll v1=ls^(si[i*2-1]+si[i*2]);
		int s1=(ls>>(i*2-2))&3,s2=(v1>>(i*2-2))&3;
		s1=s1%3?3-2*s1:0,s2=s2%3?3-2*s2:0;
		if(s1>s2)as+=s1*tp;
		else if(s1<s2)ls=v1,as+=s2*tp;
		else ins((si[i*2-1]+si[i*2])>>(i*2)<<(i*2)),as+=s1*tp;;
	}
	printf("%lld\n",(long long)as);
}
```



##### auoj111 树

###### Problem

给一棵 $n$ 个点的有根树，$1$ 为根。

每条边有边权 $d_i$ 和修改代价 $c_i$。你可以将边权修改为任意整数，修改为 $d'$ 的代价为 $c_i*|d_i-d'|$。

你需要通过修改边权，使得根到任意一个叶子的距离相等。求最小代价并输出任意一组方案。

$n\leq 2\times 10^5$

$2s,512MB$

###### Sol

如果根到每个叶子的距离相等，则任意点到它子树内所有叶子的距离相等。考虑设 $dp_{u,x}$ 表示考虑 $u$ 的子树包括 $u$ 向父亲的边，根上面到每个叶子距离都是 $x$ 时的最小代价。为了简便，可以看成根向上有一条 $c_i=d_i=0$ 的边，这样 $dp_{1,0}$ 即为答案。

则转移可以看成如下形式（先合并子树，再考虑向上的边）：

$$
f_{u,x}=\sum_{v\in son_u}dp_{u,x}\\
dp_{u,x+d_u}=\min_{v}dp_{u,x+v}+|v|*c_u
$$

可以发现第二部分相当于一个凸函数卷积，凸函数相加和卷积得到的都是凸函数，因此 $dp_u$ 一定是凸函数。

考虑维护 $dp_u$，只需要维护 $dp_{u,0}$，$-\infty$ 处的斜率以及所有斜率的改变点即可。合并 $dp$ 可以使用 `set` 启发式合并或者平衡树合并。

再考虑这个卷积，可以看成先卷积一个 $c_u*|v|$，然后平移 $d_u$。可以发现第一部分相当于将两侧斜率大于等于 $c_u$ 部分变为斜率为 $c_u$，可以看成删去斜率改变点。整体平移可以通过标记解决。

初始时的 $dp$ 为 $c_u*|v-d_u|$，因此一种斜率变化点可能有多个重叠在一起，需要记录每个位置上的出现次数。

然后考虑输出方案。过程中唯一的转移为卷积 $c_u*|v|$，考虑卷积过程中，中间斜率绝对值小于 $c_u$ 的部分的区间。则可以发现如果转移后 $x-d_u$ 在这个区间中，则转移前对应位置仍然是 $x-d_u$，否则会通过调整这条边的边权使得转移前的距离在区间的边界上（因为两侧部分可以看成转移前的区间端点通过卷积 $|v|$ 转移得到）。从上往下还原方案即可。

复杂度 $O(n\log^2 n)$ 或者 $O(n\log n)$。

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 200500
#define ll long long
int n,f[N],d[N],v[N],head[N],cnt;
int id[N];
set<pair<ll,ll> > sr[N];
ll lv[N],su[N],lb[N],rb[N],lz[N],si[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t)
{
	ed[++cnt]=(edge){t,head[f]};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t]};head[t]=cnt;
}
void ins(int u,ll x,ll y)
{
	set<pair<ll,ll> >::iterator it=sr[u].lower_bound(make_pair(x,0));
	if(it!=sr[u].end()&&(*it).first==x)
	{
		pair<ll,ll> tp=*it;
		sr[u].erase(tp);tp.second+=y;sr[u].insert(tp);
	}
	else sr[u].insert(make_pair(x,y));
}
void merge(int x,int y)
{
	if(sr[id[x]].size()<sr[id[y]].size())swap(id[x],id[y]);
	for(set<pair<ll,ll> >::iterator it=sr[id[y]].begin();it!=sr[id[y]].end();it++)
	{
		pair<ll,ll> tp=*it;
		tp.first+=lz[id[y]]-lz[id[x]];
		ins(id[x],tp.first,tp.second);
	}
	su[x]+=su[y];lv[x]+=lv[y];
}
void dfs(int u,int fa)
{
	id[u]=u;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),merge(u,ed[i].t);
	if(id[u]==u)
	{
		lb[u]=rb[u]=0;
		ins(id[u],d[u],v[u]*2);
		lv[u]=1ll*d[u]*v[u];su[u]=v[u];
		return;
	}
	if(su[u]<=v[u])lb[u]=-1e17,rb[u]=1e17;
	else
	{
		ll v1=su[u]-v[u];
		while(v1)
		{
			pair<ll,ll> tp=*sr[id[u]].begin();sr[id[u]].erase(tp);
			ll v2=min(tp.second,v1);
			v1-=v2;tp.second-=v2;lv[u]-=v2*(tp.first+lz[id[u]]);
			if(tp.second)sr[id[u]].insert(tp);
		}
		lb[u]=(*sr[id[u]].begin()).first+lz[id[u]];
		v1=su[u]-v[u];
		while(v1)
		{
			pair<ll,ll> tp=*sr[id[u]].rbegin();sr[id[u]].erase(tp);
			ll v2=min(tp.second,v1);
			v1-=v2;tp.second-=v2;
			if(tp.second)sr[id[u]].insert(tp);
		}
		if(sr[id[u]].size())rb[u]=(*sr[id[u]].rbegin()).first+lz[id[u]];
		su[u]=v[u];
	}
	lz[id[u]]+=d[u];lv[u]+=1ll*su[u]*d[u];
}
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)scanf("%d%d%d",&f[i],&d[i],&v[i]),adde(i,f[i]);
	dfs(1,0);
	printf("%lld\n",lv[1]);
	si[1]=lb[1];
	for(int i=2;i<=n;i++)
	{
		si[i]=si[f[i]]-d[i];
		if(si[i]<lb[i])si[i]=lb[i];
		if(si[i]>rb[i])si[i]=rb[i];
		printf("%lld\n",si[f[i]]-si[i]);
	}
}
```

##### auoj112 向量

###### Problem

有一棵 $n$ 个点带边权的树，保证边权 $w_i>0$ 且为整数。

你需要给每个点一个 $m$ 维向量 $v_i$ ($m$ 可以任意决定，但不能超过 $16$)，使其满足如下条件：

$$
\forall i,j,dis(i,j)=max_{k=1}^m|v_{i,k}-v_{j,k}|
$$

输出任意一组解。

$n\leq 1000,w_i\leq 10^6$

$1s,512MB$

###### Sol

有 $m=O(\log n)$，考虑树分治。分治操作可以看成选择一个点，使用一维上的操作使得跨过这个点的点对都被这一维满足。

但可以发现如果点分治，则当前点有三个儿子时这样就无法构造。因此考虑边分治。对于当前选出的一条边 $(u,v)$，将 $u$ 侧每个点的点权设为这个点到 $u$ 的距离，$v$ 侧点权为这个点到 $v$ 的距离的相反数，这样跨过 $(u,v)$ 的点在这一维上的差就是它们的距离。

边分树只有 $O(\log n)$ 层，考虑将一层的所有块放在同一维向量中。考虑如何使得不同块之间不违反限制。

显然，将同一块的点权全部加上一个数，这一块内还是满足要求。同时每一块内部的一条边 $(i,j)$ 一定满足 $dis(i,j)\geq |v_i-v_j|$

而如果每条边 $(i,j)$ 满足 $dis(i,j)\geq |v_i-v_j|$，则由绝对值不等式，任意两个点间都满足 $dis(i,j)\geq |v_i-v_j|$。

因此只需要让块之间的边满足这个限制即可。所有块形成一棵树的关系，可以通过 `dfs` 决定给每个块加上的权值使得满足上一条件。

边分治的一种实现方式是拆虚点将边权看成 $0$。而看成 $0$ 后显然一个点的所有虚点的向量必须相同，因此虚点可以直接合并，这样可以直接得到方案。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 3333
int n,m,head[N],cnt,s[N*2][3],ct,ct2,a,b,c,sz[N],bel[N][17],nw,fu[N][17],vl[N][17],as[N][17],vis[N],v3,as1,as2,f1[N],dep[N],is[N],v2[N];
struct edge{int t,next,v,id;}ed[N*2];
void adde(int f,int t,int v,int id){ed[++cnt]=(edge){t,head[f],v,id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v,id};head[t]=cnt;}
void dfs0(int u,int fa)
{
	int las=u,las2=u;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		if(las==-1)s[++ct2][0]=++ct,s[ct2][1]=las2,las=las2=ct;
		s[++ct2][0]=las,s[ct2][1]=ed[i].t,s[ct2][2]=ed[i].v;
		las=-1;dfs0(ed[i].t,u);
	}
}
void dfs1(int u,int fa,int vl)
{
	sz[u]=1;is[u]=1;bel[u][nw]=vl;
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa&&!vis[ed[i].id])dfs1(ed[i].t,u,vl),sz[u]+=sz[ed[i].t];
}
void dfs2(int u,int fa)
{
	sz[u]=1;f1[u]=fa;
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa&&!vis[ed[i].id])v2[ed[i].t]=ed[i].id,dfs2(ed[i].t,u),sz[u]+=sz[ed[i].t];
	int tp=max(sz[u],v3-sz[u]);
	if(as1>tp)as1=tp,as2=u;
}
void dfs3(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa&&!vis[ed[i].id])dep[ed[i].t]=dep[u]+ed[i].v,dfs3(ed[i].t,u);
}
void dfs4(int u,int fa)
{
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa&&!vis[ed[i].id])dep[ed[i].t]=dep[u]-ed[i].v,dfs4(ed[i].t,u);
}
void dfs5(int u,int fa)
{
	as[u][nw]=fu[u][nw]+vl[bel[u][nw]][nw];
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa)
	{
		if(bel[u][nw]!=bel[ed[i].t][nw])
		{
			int nt=as[u][nw]-fu[ed[i].t][nw];
			vl[bel[ed[i].t][nw]][nw]=nt;
		}
		dfs5(ed[i].t,u);
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c,0);
	ct=n;dfs0(1,0);
	for(int i=1;i<=n;i++)head[i]=0;
	cnt=0;
	for(int i=1;i<=ct2;i++)adde(s[i][0],s[i][1],s[i][2],i);
	for(int i=1;i<=16;i++)
	{
		nw=i;
		for(int j=1;j<=ct;j++)is[j]=dep[j]=0;
		for(int j=1;j<=ct;j++)if(!is[j])
		{
			dfs1(j,0,j);
			if(sz[j]==1){bel[j][i]=j+ct2;continue;}
			v3=sz[j];as1=1e9;
			dfs2(j,0);
			vis[v2[as2]]=1;
			int v1=as2,v3=f1[as2];
			dep[v1]=0;dfs3(v1,v3);
			dep[v3]=-ed[v2[as2]*2-1].v;dfs4(v3,v1);
		}
		for(int j=1;j<=ct;j++)fu[j][i]=dep[j];
		dfs5(1,0);
	}
	printf("16\n");
	for(int i=1;i<=n;i++,printf("\n"))
	for(int j=1;j<=16;j++)printf("%d ",as[i][j]);
}
```



#### SCOI2020模拟22

##### auoj117 随机变量

###### Problem

有一个数 $z$，操作前 $z=0$。

进行 $n$ 次操作，每次从 $0$ 到 $k$ 中随机选出一个整数，以 $p_i=\frac{a_i}{\sum a_i}$ 的概率选出 $i$ ，并给 $z$ 加上选出的数。

求 $min(m,z)$ 的期望，模 $998244353$

$n\leq 10^7,k*m\leq 5\times 10^7$

$4s,1024MB$

###### Sol

考虑对于 $i=0,\codts,m$ 求出最后 $z=i$ 的概率，即求 $(\sum p_ix^i)^n(\bmod x^m)$。

设 $F(x)=\sum p_ix^i$，则相当于求 $G(x)=F^n(x)(\bmod x^m)$

考虑求导，有：

$$
G'(x)=n*F'(x)*F^{n-1}(x)\\
G'(x)*F(x)=n*F'(x)*G(x)
$$

比较两侧 $x^a$ 项系数可以发现，如果知道了 $G$ 的 $0,1,...,a-1$ 项，则可以 $O(k)$ 得到下一项。注意逆元需要预处理。

复杂度 $O(km)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1050
#define M 10000050
#define mod 998244353
int n,m,k,p[N],p1[N],su,dp[M],st,fr[M],ifr[M];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=0;i<=m;i++)scanf("%d",&p[i]),su+=p[i];
	for(int i=0;i<=m;i++)p[i]=1ll*p[i]*pw(su,mod-2)%mod;
	for(int i=1;i<=m;i++)p1[i-1]=1ll*p[i]*i%mod;
	//g'f=gf'*n
	dp[0]=pw(p[0],n);
	st=pw(p[0],mod-2);
	fr[0]=1;for(int i=1;i<=k+1;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[k+1]=pw(fr[k+1],mod-2);
	for(int i=k;i>=0;i--)ifr[i]=1ll*ifr[i+1]*(i+1)%mod;
	for(int i=0;i<k;i++)
	{
		int s1=0,s2=0;
		for(int j=1;j<=m&&j<=i;j++)s1=(s1+1ll*p[j]*dp[i-j+1]%mod*(i-j+1))%mod;
		for(int j=0;j<m&&j<=i;j++)s2=(s2+1ll*p1[j]*dp[i-j])%mod;
		int tp=(1ll*n*s2-s1+mod)%mod;
		tp=1ll*tp*st%mod;
		dp[i+1]=1ll*tp*fr[i]%mod*ifr[i+1]%mod;
	}
	int as=0,su=0;
	for(int i=0;i<k;i++)as=(as+1ll*i*dp[i])%mod,su=(su+dp[i])%mod;
	as=(as+1ll*(mod+1-su)*k)%mod;
	printf("%d\n",as);
}
```



##### auoj118 两个整数

###### Problem

给定 $n$，以以下方式生成一个长度为 $2n$ 的 `01` 串 $s$：

从小到大考虑每一位，如果之前已经生成了 $n$ 个 `0`，那么这一位一定是 `1`，如果之前生成了 $n$ 个 `1` 那么这一位一定是 `0`，否则以 $\frac 12$ 概率生成 `1`，$\frac12$ 概率生成 `0`。

$q$ 次询问，每次给定 $k$ 个位置 $a_1,...,a_k$，求生成的串中 $s_{a_1}=s_{a_2}=...=s_{a_k}$ 的概率，答案对 $998244353$ 取模。

$n,q\leq 10^5,\sum k\leq 2\times 10^5$

$2s,1024MB$

###### Sol

只考虑最后一位是 `1` 的情况，最后答案乘 $2$ 即可。

假设最后有连续 $i$ 个 `1`，那么这种情况为前 $2n-i-1$ 位中有 $n-1$ 个 `0` 和 $n-i$ 个 `1`，接下来一个 `0`， $i$ 个 `1`，这种情况出现的概率是 $\frac{C_{2n-i-1}^{n-1}}{2^{2n-i}}$，算出这种情况下合法的方案数除以 $2^{2n-i}$ 就是这种情况对答案的贡献。

首先考虑 $a_k>2n-i$ 的情况，此时显然 $s_{a_k}=1$，因此如果存在一个 $a_j=2n-i$，那么因为这一位必须是 `0`，所以合法方案数为 $0$。

否则，设有 $p$ 个 $j$ 使得 $a_j<2n-i$，那么只需要前面部分中这些数全部是 `1`，那么方案数为 $C_{2n-i-1-p}^{n-1}$。

如果 $a_k=2n-i$，则需要前面的数全部是 `0`，方案数为 $C_{2n-i-1-(k-1)}^{n-1-(k-1)}=C_{a_k-k}^{n-k}$。$a_k=2n$ 时不存在这种情况，需要特殊处理。

如果 $a_k<2n-i$，那么前面的数可能都是 `0` 或者都是 `1`，两种情况方案数分别为 $C_{2n-i-1-k}^{n-1},C_{2n-i-1-k}^{n-k-1}$。

因此求答案只需要 $O(k)$ 次求 $\sum_{i=l}^r\frac{C_{i}^{n-1}}{2^{2n-i}}$ 以及 $O(1)$ 次求 $\sum_{i=l}^r\frac{C_{i}^{n-1-k}}{2^{2n-i}}$。

对于第一部分，相当于 $\frac 1{2^{2n}}\sum_{i=l}^r 2^iC_i^{n-1}$，可以预处理前缀和

对于第二部分，它与上一个唯一的区别为组合数上标为 $n-k-1$。注意到可能的 $k$ 只有 $O(\sqrt{\sum k})$ 个，对于每一种 $k$ 求出前缀和即可。

复杂度 $O((n+k)\log n+n\sqrt{\sum k})$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 200300
#define mod 998244353
int n,q,k,v[N],fr[N],ifr[N],fu[501][N],g[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int C(int a,int b){if(a<b||b<0)return 0;return 1ll*fr[a]*ifr[a-b]%mod*ifr[b]%mod;}
int doit(int l,int k){int tp=n-1-k;if(tp<=500)return fu[tp][l];return g[l];}
int solve(int l,int r,int k){return (doit(r,k)-1ll*pw(2,r-l+1)*doit(l-1,k)%mod+mod)%mod;}
int main()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=2e5;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	scanf("%d%d",&n,&q);
	for(int i=0;i<=500;i++)for(int j=1;j<=n*2;j++)fu[i][j]=(2ll*fu[i][j-1]+C(j,n-1-i))%mod;
	while(q--)
	{
		scanf("%d",&k);
		if(k>500)for(int j=1;j<=n*2;j++)g[j]=(2ll*g[j-1]+C(j,n-1-k))%mod;
		int as=0;
		for(int i=1;i<=k;i++)scanf("%d",&v[i]),as=(as+1ll*pw(2,2*n-v[i]+2)*solve(v[i-1]-i+1,v[i]-i-1,n-1))%mod;
		if(v[k]<n*2)as=(as+4ll*solve(v[k]-k,n*2-2-k,n-1)+4ll*solve(v[k]-k,n*2-2-k,n-1-k)+1ll*pw(2,n*2-v[k]+1)*C(v[k]-k,n-k))%mod;
		printf("%d\n",1ll*as*pw(499122177,2*n)%mod);
	}
}
```



##### auoj119 字母集合

###### Problem

考虑 $n$ 个元素构成的所有元素个数不超过 $k$ 的非空子集。你需要将这些子集构成的集合划分成若干个集合，使得它们满足如下条件：

1. 每个集合中所有子集两两不交。
2. 每个集合中所有子集大小之和不超过 $k$。

最小化划分出的集合的数量，输出方案。

$n\leq 17$

$4s,1024MB$

###### Sol

显然大小大于 $\frac k2$ 的集合不可能两个放在一起，因此这部分每个集合必须放一个单独的集合。因此 $k$ 为奇数时，答案不小于 $\sum_{i=\lceil\frac k2\rceil}^kC_n^i$，偶数时答案不超过 $\lceil\frac{C_n^k}2\rceil+\sum_{i=\frac k2+1}^kC_n^i$。

注意到 $C_n^k\geq C_n^0,C_n^{k-1}\geq C_n^1,\cdots,C_n^{k-\lfloor\frac k2\rfloor}\geq C_n^{\lfloor\frac k2\rfloor}$。因此考虑用大小为 $k-1$ 的集合中的一部分去配对所有大小为 $1$ 的集合，用 $k-2$ 的集合去配对大小为 $2$ 的集合，以此类推。则如果每一组都能配出来，则这样对于奇数的情况找到了最优方式。

则问题相当于，给定 $a,b(a<b,a+b\leq n)$，需要给每一个 $a$ 元子集配对一个 $b$ 元子集，使得：

1. 每个 $b$ 元子集最多被用一次。
2. 一个集合和配对的集合之间没有交。

首先考虑一种基础情况：$n=2m+1,a=m,b=m+1$。此时可以看成需要找一个双射。通过各种尝试可以得到如下构造：

考虑将子集看成一个长度为 $n$ 的序列，对于第 $i$ 个位置，如果子集中有这个元素则这一位为 $-1$，否则这一位为 $1$。

从 $a$ 到 $b$ 的方式为，考虑序列的前缀和，将最后一个前缀和最小的位置的下一个位置由 $1$ 变为 $-1$。由于和为 $1$，最小的位置一定不是最后一个位置。

从 $b$ 到 $a$ 的方式为，将第一个前缀和最小的位置处的 $-1$ 变为 $1$。因为上一步是在最后一个前缀和最小的位置之后操作，可以发现操作后第一个前缀和最小的位置就是操作位置。



接下来考虑 $a+b=n$ 的情况，考虑如下方式：

从 $b$ 到 $a$ 的方式为，进行 $b-a$ 次操作，每次将第一个前缀和最小的位置处的 $-1$ 变为 $1$。

由于每一步都是双射，只需要证明单向可以进行操作，就可以得到整体的双射。初始时序列的总和为 $a-b$，最小前缀和不大于 $a-b$，而每次操作后，考虑最小前缀和的上一个位置，可以得到最小前缀和最多增加 $1$。因此操作时最小前缀和都小于 $0$，因此不会出现操作开头之前的情况。



最后考虑剩余情况，如果 $b>\frac n2$，则可以发现 $b$ 可以和 $n-b$ 配对，只需要对于每个 $a$ 元子集找一个 $n-b$ 元子集配对，使得配对的集合是它的超集即可。

这里可以使用和上面类似的操作，即将最后一个前缀和最小的位置的下一个位置由 $1$ 变为 $-1$。这样是一个单射，因此可以找到这部分的配对方案。

这样就解决了 $a<b$ 部分的所有配对。预处理操作后复杂度 $O(n2^n)$。接下来考虑 $k$ 是偶数的情况。此时需要将 $k$ 元子集两两配对，使得每一对不交且配对数量尽量多。

考虑一种乱搞，把图建出来（边数不会太多），然后问题相当于一般图最大匹配。考虑类似带花树的方式，选一个未匹配点开始dfs，但是dfs中不缩花，找到解直接结束过程。

因为每个点都有 $O(n)$ 条出边（$n=2k$ 的情况除外，但这种情况非常简单），加上图有很好的对称性，这样搞可以对于范围内的所有情况找到最大匹配，并且跑得飞快。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
using namespace std;
#define N 132001
int n,k,ct,as[N][2],f[N][2],ci[N],vis[N],fi[N],ti[N];
int vl[N],nt[N];
void doit(int s)
{
	printf(" ");
	for(int i=1;i<=n;i++)if(s>>(i-1)&1)printf("%c",'a'+i-1);
}
vector<int> ls[N];
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=0;i<1<<n;i++)
	{
		ci[i]=ci[i>>1]+(i&1);
		int su=0,mn=0,fr=0;
		for(int j=1;j<=n;j++)
		{
			su+=(i>>j-1)&1?-1:1;
			if(su<mn)mn=su,fr=j;
		}
		if(fr)f[i][0]=i^(1<<fr-1);
		su=mn=fr=0;
		for(int j=1;j<=n;j++)
		{
			su+=(i>>j-1)&1?-1:1;
			if(su<=mn)mn=su,fr=j;
		}
		if(fr<n)f[i][1]=i^(1<<fr);
	}
	for(int i=0;i*2<k;i++)
	{
		for(int j=0;j<1<<n;j++)if(ci[j]==k-i)
		{
			int nw=((1<<n)-1)^j;
			for(int t=0;t<(n-k+i)*2-n;t++)nw=f[nw][0];
			vl[nw]=j;
		}
		int tp=n-k+i;if(tp*2>n)tp=n-tp;
		for(int j=1;j<1<<n;j++)if(ci[j]==i)
		{
			int nw=j;
			for(int t=0;t<tp-i;t++)nw=f[nw][1];
			as[++ct][0]=j;as[ct][1]=vl[nw];vl[nw]=0;
		}
		for(int j=0;j<1<<n;j++)if(vl[j])as[++ct][0]=vl[j],vl[j]=0;
	}
	if(k%2==0)
	{
		for(int j=0;j<1<<n;j++)if(ci[j]==k/2)
		{  
			int nw=((1<<n)-1)^j;
			for(int l=nw;l;l=(l-1)&nw)if(ci[l]==k/2&&j!=l)ls[j].push_back(l);
		}
		for(int j=0;j<1<<n;j++)if(ci[j]==k/2&&!vis[j])
		{
			int fr=0;
			queue<int> st;
			st.push(j);
			while(!st.empty())
			{
				int u=st.front();st.pop();
				for(int l=0;l<ls[u].size();l++)
				{
					int t=ls[u][l];
					if(t==j)break;
					if(!vis[t]){fi[t]=u;fr=t;break;}
					if(ti[nt[t]]==j)continue;
					ti[nt[t]]=j;fi[nt[t]]=u;
					st.push(nt[t]);
				}
				if(fr)break;
			}
			if(!fr)continue;
			vis[fr]=vis[j]=1;
			int li=fr;fr=fi[fr];
			while(1)
			{
				int ls=li;li=nt[fr];
				nt[fr]=ls;nt[ls]=fr;
				if(fr==j)break;
				fr=fi[fr];
			}
		}
		for(int j=0;j<1<<n;j++)if(ci[j]==k/2)if(!nt[j]||nt[j]>j)as[++ct][0]=j,as[ct][1]=nt[j];
	}
	printf("%d\n",ct);
	for(int i=1;i<=ct;i++)
	if(as[i][1])printf("2"),doit(as[i][0]),doit(as[i][1]),printf("\n");
	else printf("1"),doit(as[i][0]),printf("\n");
}
```

##### auoj120 最大公约数

###### Problem

求 $\sum_{i=1}^n\sum_{j=1}^n\frac i {gcd(i,j)}$，答案模 $10^9+7$

$n\leq 10^{10}$

$2s,1024MB$

###### Sol

考虑枚举 `gcd` 再反演/容斥的方式：

$$
\sum_{i=1}^n\sum_{j=1}^n\frac i {gcd(i,j)}\\
=\sum_{g=1}^n\sum_{i=1}^{\lfloor\frac ng \rfloor}\sum_{j=1}^{\lfloor\frac ng \rfloor}i[\gcd(i,j)=1]\\
=\sum_{g=1}^n\sum_{i=1}^{\lfloor\frac ng \rfloor}\sum_{j=1}^{\lfloor\frac ng \rfloor}i\sum_{k|\gcd(i,j)}\mu(k)\\
=\sum_{g=1}^n\sum_{k=1}^n\mu(k)\sum_{i=1}^{\lfloor\frac n{gk} \rfloor}\sum_{j=1}^{\lfloor\frac n{gk} \rfloor}k*i\\
=\sum_{t=1}^n(\sum_{k|t}k\mu(k))\sum_{i=1}^{\lfloor\frac nt \rfloor}\sum_{j=1}^{\lfloor\frac nt \rfloor}i\\
=\sum_{t=1}^n(\sum_{k|t}k\mu(k))\frac12\lfloor\frac nt \rfloor^2(\lfloor\frac nt \rfloor+1)
$$

对后面数论分块，则只需要算 $f(t)=\sum_{k|t}k\mu(k)$ 在所有 $t=\lfloor\frac nx\rfloor$ 位置的前缀和即可。

可以发现 $f*id=id\mu*id*1=1*(id\mu*id)=1*\epsilon=1$。因此 $f$ 满足杜教筛的条件。复杂度 $O(n^{\frac 23})$

杜教筛做法简述：

考虑求积性函数 $f$ 的前缀和，满足存在积性函数 $g$ 使得 $g,f*g$ 的前缀和容易求出。则：

$$
\sum_{i=1}^nf*g(i)=\sum_{ij\leq n}f(i)g(j)\\
f(1)=\sum_{i=1}^nf*g(i)-\sum_{ij\leq n,i>1}f(i)g(j)
$$

那么对右侧第二项数论分块，可以发现需要 $g$ 的前缀和以及 $\lfloor\frac nx\rfloor$ 位置的 $f$ 前缀和。

那么过程中只会需要 $\lfloor\frac nx\rfloor$ 位置的 $f$ 前缀和，记忆化即可。

考虑对于 $m\leq n^{\frac 23}$ 部分线性筛求出 $f$ 前缀和，对于大的部分递归计算。此时大的部分记忆化可以直接标号 $\frac nx$。

复杂度为 $\sum_{i=1}^{n^{\frac 13}}(\frac nx)^{\frac 12}$，而 $\int x^{-\frac 12}=2\sqrt x+C$，因此第二部分复杂度为 $O(\sqrt n*\sqrt{x^{\frac 13}})=O(n^{\frac 23})$，两部分复杂度相同。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 3006000
#define mod 1000000007
#define ll long long
int pr[N],ch[N],f[N],as[N],ct;
ll n;
void pre(int m)
{
	f[1]=1;
	for(int i=2;i<=m;i++)
	{
		if(!ch[i])pr[++ct]=i,f[i]=1-i;
		for(int j=1;1ll*i*pr[j]<=m&&j<=ct;j++)
		{
			ch[i*pr[j]]=1,f[i*pr[j]]=f[i]*(1-pr[j]);
			if(i%pr[j]==0){f[i*pr[j]]=f[i];break;}
		}
	}
	for(int i=2;i<=m;i++)f[i]=(f[i]+f[i-1]+3ll*mod)%mod;
}
int solve(ll x)
{
	if(x<=3e6)return f[x];
	int id=n/x;
	if(as[id])return as[id];
	int as1=x%mod;
	for(ll l=2,r;l<=x;l=r+1)
	{
		r=x/(x/l);
		ll f1=((r%mod)*(r%mod+1)/2-(l%mod)*(l%mod-1)/2%mod+mod)%mod;
		as1=(as1-f1*solve(x/l)%mod+mod)%mod;
	}
	return as[id]=as1;
}
int main()
{
	scanf("%lld",&n);pre(3e6);
	int as1=0;
	for(ll l=1,r;l<=n;l=r+1)
	{
		r=n/(n/l);
		ll tp=n/l;
		int res=(tp%mod)*(tp%mod+1)/2%mod*(tp%mod)%mod;
		as1=(as1+1ll*res*(solve(r)-solve(l-1)+mod))%mod;
	}
	printf("%d\n",as1);
}
```



##### auoj121 凸包的价值

###### Problem

对于一个严格凸包(点数大于 $2$ 且没有三点共线)，设它的顶点有 $x$ 个，它内部的点（包含边界上，不包含顶点）有 $y$ 个，外部的点有 $z$ 个，那么这个凸包的贡献为 $xa^x(a+c)^yc^z$。

给 $n$ 个不重合的点，求所有可以组成的严格凸包的贡献和，模 $10^9+7$

$n\leq 2000$

$2s,512MB$

###### Sol

设凸包顶点集合为 $A$，内部点集合为 $B$，考虑将 $(a+c)^y$ 拆开，则贡献相当于：

$$
|A|a^{|A|}(a+c)^{|B|}c^{n-|A|-|B|}=\sum_{|S|\subset |B|}|A|a^{|A|+|S|}c^{n-|A|-|S|}
$$

相当于枚举 $A$，再枚举凸包内部点的一个子集，贡献与凸包点数和枚举的点集点数有关。

注意到对于一个 $T=|A|\cup |S|$，因为 $S$ 中点都在 $A$ 形成的凸包内部，因此 $A$ 一定是点集 $T$ 的凸包上的所有点。因此考虑枚举凸包，设 $f(S)$ 为点集的凸包点数，那么贡献等于 $\sum_Ta^{|T|}c^{n-|T|}f(T)$。

注意到凸包点数相当于凸包边数，于是可以将 $f(T)$ 拆成每条边做贡献，考虑一条边的贡献，即选择一个点集使得凸包包含这条边的方案数。可以发现条件为凸包中所有点都在这条线的一侧或者线段上，且所有点不同时在线段上。

枚举一个点 $i$，将其余点按这个点极角排序。对于一条边 $(i,j)$，考虑凸包在这条边左侧的情况（另外一侧在另外一个点考虑）

则点集中的所有点必须在这条线的左侧，或者在这条线段上。假设有 $k$ 个点，那么这条边会对 $|T|=x$ 的情况贡献 $C_{k}^{x-2}$ 种方案，可以记录每个 $k$ 的情况出现了多少次，最后再对于每一种 $k$ 算贡献。

注意到这样会统计所有点在一条线上的情况，因此需要再减掉这种情况。在极角排序过程中求每条线段上有多少个点即可。

复杂度 $O(n^2\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 2333
#define ll long long
#define mod 1000000007
struct pt{int x,y;}p[N],tp2[N],tp3[N];
int n,a,b,ct1,ct2,su[N],as[N],c[N][N],f1[N];
ll cross(pt a,pt b){return 1ll*a.x*b.y-1ll*a.y*b.x;}
pt operator -(pt a,pt b){return (pt){a.x-b.x,a.y-b.y};}
bool cmp(pt a,pt b){return cross(a,b)<0;}
void solve(int x)
{
	ct1=ct2=0;
	for(int i=1;i<=n;i++)if(i!=x)
	{
		pt sb=p[i]-p[x];
		if(sb.y>0||(sb.y==0&&sb.x<0))tp2[++ct1]=sb;
		else tp3[++ct2]=sb;
	}
	sort(tp2+1,tp2+ct1+1,cmp);
	sort(tp3+1,tp3+ct2+1,cmp);
	int lb=0;
	for(int i=1;i<=ct1;i++)
	{
		int rb=i;
		while(rb<ct1&&cross(tp2[rb+1],tp2[i])==0)rb++;
		while(lb<ct2&&cross(tp2[i],tp3[lb+1])<0)lb++;
		for(int j=0;j<rb-i+1;j++)su[ct1-rb+lb+j]=(su[ct1-rb+lb+j]+1)%mod;
		for(int j=1;j<=rb-i+1;j++)f1[j+1]=(f1[j+1]+c[rb-i+1][j])%mod;
		i=rb;
	}
	lb=0;
	for(int i=1;i<=ct2;i++)
	{
		int rb=i;
		while(rb<ct2&&cross(tp3[rb+1],tp3[i])==0)rb++;
		while(lb<ct1&&cross(tp3[i],tp2[lb+1])<0)lb++;
		for(int j=0;j<rb-i+1;j++)su[ct2-rb+lb+j]=(su[ct2-rb+lb+j]+1)%mod;
		for(int j=1;j<=rb-i+1;j++)f1[j+1]=(f1[j+1]+c[rb-i+1][j])%mod;
		i=rb;
	}
}
int main()
{
	scanf("%d%d%*d%d",&n,&a,&b);
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)
	for(int j=1;j<i;j++)
	c[i][j]=(c[i-1][j-1]+c[i-1][j])%mod;
	for(int i=1;i<=n;i++)scanf("%d%d",&p[i].x,&p[i].y);
	for(int i=1;i<=n;i++)solve(i);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=i;j++)
	as[j+2]=(as[j+2]+1ll*su[i]*c[i][j])%mod;
	int as1=0;
	for(int i=3;i<=n;i++)
	{
		int tp=(as[i]+mod-f1[i])%mod;
		for(int j=1;j<=n;j++)tp=1ll*tp*(j<=i?a:b)%mod;
		as1=(as1+tp)%mod;
	}
	printf("%d\n",as1);
}
```



##### auoj122 交通网络

###### Problem

给一张 $n$ 个点完全图，其中所有 $(i,i+1)$ 的边为关键边。

对于图的一棵生成树，如果它包含 $k$ 条关键边，则它的权值为 $k2^k$。

求图的所有生成树的权值和，模 $998244353$

$n\leq 5\times 10^5$

$2s,512MB$

###### Sol

设 $f(i)$ 表示有 $i$ 条关键边的方案数，$g(i)=\sum_{j\geq i}C_j^if(j)$。则 $g$ 的组合意义相当于钦定 $i$ 条关键边必须选，所有钦定情况的生成树数量之和。

设关键边形成的连通块大小为 $a_1,...,a_{n-i}$，由 `prufer` 序方案为 $n^{n-i-2}\prod a_i$。

只需要考虑 $\prod a_i$，它的组合意义相当于在每一个连通块中选出一个点的方案数。

那么 $g(i)$ 相当于在链上交替选出 $n-i$ 个点和 $n-i-1$ 条边的方案数。考虑将边和点都看成元素，相当于在 $2n-1$ 个数中选 $2n-2i-1$ 个数，使得选择的第 $i$ 小的数与 $i$ 奇偶性相同。

钦定在 $2n$ 处有一个数，且这个数必选，考虑每一个数减去上一个数的差，显然差必须为奇数。因为钦定了最后一个数为 $2n$，此时相当于将 $2n$ 分成 $2n-2i$ 个奇数的方案数。

将每个数减一，则这相当于将 $2i$ 分成 $2n-2i$ 个偶数的方案数，于是相当于将 $i$ 分成 $2n-2i$ 个数的方案数，于是它等于 $C_{2n-i-1}^{i}$。

根据二项式反演有 $f(i)=\sum_{j\geq i}(-1)^{j-i}C_j^ig(j)$，NTT即可

复杂度 $O(n\log n)$，这样可以处理任意权值的问题。

再考虑计算一个 $g(j)$ 的贡献系数，即对 $\{n2^n\}$ 做反向的二项式反演。可以发现结果为 $\{2n\}$。即答案为 $2\sum_{i=1}^{n-1}i*n^{n-i-2}*C_{2n-i-1}^{i}$。

另外一种方式是，考虑 $k2^k$ 的组合意义，相当于在关键边中选一个子集，再独立选一条边。

考虑两部分选择的关键边的并。如果并为 $S$，有两种情况：

1. 选的边在子集内。这种情况贡献为 $|S|$。
2. 选的边在子集外，贡献也为 $|S|$。

因此每个 $S$ 的贡献为 $2|S|$，而一个 $S$ 的方案数即为这些边同时在生成树内的方案数，即上面的 $g$。因此得到相同的结果。

###### Code

Sol1:

```cpp
#include<cstdio>
using namespace std;
#define N 1050000
#define mod 998244353
int n,fr[N],ifr[N],f[N],as,a[N],b[N],rev[N],ntt[N];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%mod;a=1ll*a*a%mod;b>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int tp=pw(3,mod-1+t*(mod-1)/i);
		for(int j=0;j<s;j+=i)
		for(int l=j,st=1;l<j+(i>>1);l++,st=1ll*st*tp%mod)
		{
			int v1=ntt[l],v2=1ll*ntt[l+(i>>1)]*st%mod;
			ntt[l]=(v1+v2)%mod;
			ntt[l+(i>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=pw(s,t==-1?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d",&n);
	fr[0]=ifr[0]=1;for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<n;i++)f[n-i]=1ll*fr[n+i-1]*ifr[n-i]%mod*ifr[2*i-1]%mod*pw(n,mod-3+i)%mod;
	int l=1;while(l<=n*2)l<<=1;
	for(int i=0;i<=n;i++)a[i]=1ll*f[i]*fr[i]%mod,b[n-i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,-1);
	for(int i=1;i<n;i++)as=(as+1ll*a[i+n]*i%mod*pw(2,i)%mod*ifr[i])%mod;
	printf("%d\n",as);
}
```

Sol2:

```cpp
#include<cstdio>
using namespace std;
#define N 1005000
#define mod 998244353
int n,fr[N],ifr[N],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	fr[0]=1;for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n*2]=pw(fr[n*2],mod-2);for(int i=n*2;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	int tp=2*pw(n,mod-2)%mod;
	for(int i=n-1;i>=1;i--)as=(as+1ll*fr[n*2-i-1]*ifr[i-1]%mod*ifr[n*2-i*2-1]%mod*tp)%mod,tp=1ll*tp*n%mod;
	printf("%d\n",as);
}
```



#### SCOI2020模拟14

##### auoj123 倾尽天下

###### Problem

有一个长度为 $n$ 的数组 $a$，满足 $a_1=0,a_{ij}=a_i\oplus a_j$。则可以发现只需要确定了质数位置的值，就可以确定整个数组的值。

有 $q$ 次操作，每次操作为翻转一个 $a_p$ 处的值。求出每次操作后重新计算数组，当前数组中 $1$ 的个数。

$n,q\leq 2\times 10^5$

$3s,512MB$

###### Sol

乱搞做法：拿 `bitset` 维护+根号分治，复杂度 $O(\frac{nq}{\omega}+(n+q)\sqrt n)$。

将 $0$ 看成 $1$，$1$ 看成 $-1$，则 $a$ 是一个完全积性函数。

设 $su_i=\sum_{j=1}^ia_j$，考虑修改时对 $su_n$ 的改变，注意到只会改 $p$ 的倍数，所以可以先减去 $p$ 的倍数的贡献，算出新的贡献后把贡献加回去。

因为 $a$ 是完全积性函数，有 $\sum_{i=1}^{\lfloor\frac np\rfloor}a_{ip}=a_psu_{\lfloor\frac np\rfloor}$。因此 $su_n$ 中的这部分贡献只和 $su_{\lfloor\frac np\rfloor}$ 有关，由这个值就可以修改。

这样只会用到所有的 $su_{\lfloor\frac nk\rfloor}$，只有 $O(\sqrt n)$ 个取值，考虑同时维护维护这些位置处的前缀和。

那么先从大到小减去每个位置对应的 $a_psu_{\lfloor\frac np\rfloor}$，再计算新的贡献，从小到大加回去。

复杂度 $O(q\sqrt n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 200500
int n,vl[N],as[N],ct,q,a,tid[N],f[N];
int main()
{
	scanf("%d%d",&n,&q);
	vl[ct=1]=n;as[1]=n;
	for(int i=2;i<=n;i++)if(n/i!=n/(i-1))vl[++ct]=n/i,as[ct]=n/i;
	for(int i=1;i<=ct;i++)tid[vl[i]]=i;
	for(int i=1;i<=n;i++)f[i]=1;
	while(q--)
	{
		scanf("%d",&a);
		for(int i=1;i<=ct;i++)as[i]-=f[a]*as[tid[vl[i]/a]];
		f[a]*=-1;
		for(int i=ct;i>=1;i--)as[i]+=f[a]*as[tid[vl[i]/a]];
		printf("%d\n",(n-as[1])/2);
	}
}
```



##### auoj124 春风一顾

上次blog写了，然后我懒了，所以说直接抄写了.jpg

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

对于 $b=0$ 的情况，相当于 $a$ 行内任意，然后求第一个位置和最后一个位置的方案数。这显然是 $C_a^2+a+1=C_{a+2}^2-a$。此时直接 $dp$ 复杂度即为 $O(n^2m)$。转移可以使用NTT优化，复杂度 $O(nm\log n)$，可以通过



考虑将 $dp$ 看成生成函数，设 $F_m(x)=\sum dp_{i,m}x^i$。则 $F_i$ 到 $F_{i+1}$ 的转移有两部分：先做一次 $C_{a+b+2}^{b+2}$ 的卷积，再减去 $a$。

为了简单地表示卷积组合数的形式，考虑写成EGF的形式，即 $G_m(x)=\sum\frac{dp_{i,m}}{i!}x^i$。

此时第二部分可以看成 $xG'_m(x)$，考虑第一部分。$C_{a+b+2}^{b+2}=\frac{(a+b+2)!}{a!(b+2)!}$， $a!$ 可以直接看成EGF的系数，卷积的函数为 $\sum_{i\geq 0}\frac 1{(i+2)!}x^i=\frac{e^x-x-1}{x^2}$，最后需要乘上一个 $a+b+2$ 的阶乘，而直接表示为EGF系数为 $(a+b)!$，因此可以看成乘以 $x^2$，再求导两次。因此可以得到转移为：
$$
F_m(x)=(F_{m-1}(x)*(e^x-x-1))''-xF_{m-1}'(x)
$$
初值为 $F_0(x)=1$，可以发现 $F_m(x)$ 一定可以写成 $\sum x^ae^{bx}$ 的形式，其中 $0\leq a,b\leq m$。这样直接求出 $F_m(x)$ 的表示复杂度为 $O(m^3)$。

计算答案中的卷积相当于再乘一个 $e^x$，考虑算答案，相当于求 $n![x^n]x^ae^{bx}$，这相当于 $[a\leq n]b^{n-a}\frac{n!}{(n-a)!}$，可以直接求。

复杂度 $O(m^3)$

###### Code

Sol1:

```cpp
#include<cstdio>
using namespace std;
#define N 16400
#define M 201
#define mod 998244353
int dp[M][N],fr[N],ifr[N],n,m,rev[N],ntt[N],a[N],b[N],c[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int tp=pw(3,(mod-1)/i);
		if(t==-1)tp=pw(tp,mod-2);
		for(int j=0;j<s;j+=i)
		for(int k=j,st=1;k<j+(i>>1);k++,st=1ll*st*tp%mod)
		{
			int v1=ntt[k],v2=ntt[k+(i>>1)]*1ll*st%mod;
			ntt[k]=(v1+v2)%mod;ntt[k+(i>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=t==-1?pw(s,mod-2):1;
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d%d",&n,&m);
	fr[0]=ifr[0]=1;for(int i=1;i<=n+2;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	dp[0][0]=1;
	for(int i=1;i<=m;i++)
	{
		int l=1;while(l<=n*2)l<<=1;
		for(int j=0;j<l;j++)a[j]=b[j]=0;
		for(int j=0;j<=n;j++)a[j]=1ll*dp[i-1][j]*ifr[j]%mod,b[j]=ifr[j+2];b[0]=0;
		dft(l,a,1);dft(l,b,1);for(int j=0;j<l;j++)a[j]=1ll*a[j]*b[j]%mod;dft(l,a,-1);
		for(int j=0;j<=n;j++)dp[i][j]=(1ll*dp[i-1][j]*(1+1ll*j*(j+1)/2%mod)+1ll*a[j]*fr[j+2])%mod;
	}
	int as=0;for(int i=0;i<=n;i++)as=(as+1ll*dp[m][i]*fr[n]%mod*ifr[i]%mod*ifr[n-i])%mod;
	printf("%d\n",as);
}
```

Sol2:

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



##### auoj125 陌上花早

###### Problem

有 $n$ 个物品，每个物品有价值 $v_i$ 和种类 $c_i$。

对于每个 $k=1,\cdots,n$，求出下面问题的答案：

你需要选 $k$ 个物品，满足如果选择了一个种类的物品，则这个种类的物品至少需要选两个。求最大总价值或输出无解。

$n\leq 2\times 10^5$，保证出现的每个种类至少有两个物品。

$2s,512MB$

###### Sol

~~去掉一项再做卷积的结果不可能是凸的，即使分成几类也不是凸的~~

如果确定则每一类物品选择的数量，则最后一定是每一类内选择收益最大的。可以求出每一类选择 $2,3,\cdot$ 个的收益。

考虑选择 $k$ 个物品的一组最优解与选择 $k+1$ 个物品的最优解之间的差距。考虑记录每种物品选择数量改变的值，这样可以得到一个集合 $S$，$S$ 中元素和为 $1$。则有如下性质：

1. 存在一种从任意一个 $k$ 个物品的最优解变换到 $k+1$ 个物品的最优解的方案，使得 $S$ 中不同时存在 $>3,<-3$ 的元素。

考虑这样的两个元素 $+x,-y$。则有两种情况：

如果 $+x$ 部分的前两个元素和大于 $-y$ 部分的后两个元素和，则在 $k$ 个物品的解上 $+2,-2$，可以得到严格更优的解。

如果 $+x$ 部分的前两个元素和小于等于 $-y$ 部分的后两个元素和，则将这次的变化变为 $+(x-2),-(y-2)$，由凸性这样得到的解不会变差。

因此存在一种情况满足上述条件。

2. 存在一种方案，使得 $S$ 中不存在 $>3,<-3$ 的元素。

对于 $+4,-3$，可以沿用上一个讨论，但将第一部分换成 $+3,-3$。因为前两个数的和不小于总和的 $\frac 23$，分析可得相同的结论。因此如果必须存在 $+4$ 或更大的元素，则不能存在 $-3$。

$+4,-2$ 也可以用相同的讨论处理。剩下情况为 $+4$ 和若干个 $-1$。此时一种情况是将 $+2,-1,-1$ 的操作移到上一轮变得严格更优，另外一种情况是直接不操作这部分。

可以发现上述讨论对更大的 $+$ 也成立，因为最多让 $+$ 的数减少 $2$。

接下来考虑 $-4$ 的情况，有以下几种情况：

1. $-4,+3,+3$ 可以分出一个 $-2,+1,+1$，然后和上一步的最后一个讨论一样。
2. 如果存在 $+2$，则可以对 $-4,+2$ 使用上面的第二种讨论。
3. 如果存在两个 $+1$，则也可以分出 $-2,+1,+1$。
4. 否则，只有一个 $+1$，一个 $+3$。但此时不能抵消掉 $-4$，不存在这种情况。



此时只剩下 $+3,+2,+1,-1,-2,-3$。还可以得到如下性质：

1. 存在一种方案，使得 $S$ 中不存在一个子集非空的和为 $0$。

如果出现，分两种情况将这一对移到上一轮或者不做即可。

此时剩余的情况很少，进行分类讨论：

如果出现 $+3,+2$，则只能再有一个 $-1$，但这不可能。

如果出现 $+3,+1$，则只能再有 $-2$，因此 $+1$ 有一个，因为奇偶性此时 $+3$ 至少有两个，从而 $-2$ 有至少三个，但 $+3,+3,-2,-2,-2$ 和为 $0$。这种情况不存在。

如果只出现 $+3$，如果只有一个，则情况有 $+3,-2$ 和 $+3,-1,-1$。如果出现两个或以上 $+3$，则 $-2$ 最多有两个，$-1$ 最多有两个且 $-1,-2$ 不能同时出现，但这也不可能。

如果出现 $-3$，则不能出现 $+3$ 也不能出现三个 $+1$ 或者三个 $+2$。因此正数部分和不超过 $6$，从而最多有一个 $-3$。因此唯一情况为 $-3,+2,+2$。

还剩下出现 $+2,+1,-1,-2$ 的情况。如果出现 $+2$，则只能再出现 $-1$ 且最多有一个，唯一情况为 $+2,-1$。

如果出现 $-2$，则只能出现一个 $+1$，但这样不行。

对于 $+1,-1$ 的情况，唯一可能为单个 $+1$。

同时，$+3,-1,-1$ 可以分出 $+2,-1,-1$ 给上一轮，或者在这一轮变成 $+2,-1$。讨论可以发现可以不考虑这种情况。

因此只有四种情况：$(+1),(+2,-1),(+3,-2),(+2,+2,-3)$。

那么对于每一种变化维护每一类按照这个变化的权值改变量，从大到小排序，就可以对于每一种情况得到最优解，从而得到 $k+1$ 的最优解。

同时可以发现，所有的 $+2,+3$ 一定是对当前选择了 $0$ 个的数做的，所有的 $-2,-3$ 做完之后这一类一定剩余 $0$ 个，否则存在更简单的方案。这样后面四类可以少维护一些东西，讨论一些情况。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<set>
using namespace std;
#define N 200500
#define ll long long
int n,k,a,b,ci[N];
ll as,v2[N],v3[N];
vector<int> sr[N];
vector<ll> su[N];
set<pair<ll,int> > s1,s2,s3,r1,r2,r3;
void ins(int x)
{
	as+=ci[x]?su[x][ci[x]-1]:0;
	if(ci[x]==2)s2.insert(make_pair(v2[x],x));
	if(ci[x]==3)s3.insert(make_pair(v3[x],x));
	if(ci[x]==0)
	{
		r2.insert(make_pair(-v2[x],x));
		if(sr[x].size()>2)r3.insert(make_pair(-v3[x],x));
	}
	if(ci[x]<sr[x].size()&&ci[x])r1.insert(make_pair(-sr[x][ci[x]],x));
	if(ci[x]>2)s1.insert(make_pair(sr[x][ci[x]-1],x));
}
void del(int x)
{
	as-=ci[x]?su[x][ci[x]-1]:0;
	if(ci[x]==2)s2.erase(make_pair(v2[x],x));
	if(ci[x]==3)s3.erase(make_pair(v3[x],x));
	if(ci[x]==0)
	{
		r2.erase(make_pair(-v2[x],x));
		if(sr[x].size()>2)r3.erase(make_pair(-v3[x],x));
	}
	if(ci[x]<sr[x].size()&&ci[x])r1.erase(make_pair(-sr[x][ci[x]],x));
	if(ci[x]>2)s1.erase(make_pair(sr[x][ci[x]-1],x));
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d%d",&a,&b),sr[a].push_back(-b);
	for(int i=1;i<=k;i++)sort(sr[i].begin(),sr[i].end());
	for(int i=1;i<=k;i++)for(int j=0;j<sr[i].size();j++)sr[i][j]*=-1;
	int fg=0;
	for(int i=1;i<=k;i++)
	{
		v2[i]=sr[i][0]+sr[i][1];
		if(sr[i].size()>2)fg=1,v3[i]=sr[i][2]+v2[i];
	}
	if(!fg)
	{
		vector<int> tp;
		for(int i=1;i<=k;i++)tp.push_back(-v2[i]);
		sort(tp.begin(),tp.end());
		ll su=0;
		for(int i=1;i<=k;i++)su-=tp[i-1],printf("-1\n%lld\n",su);
		return 0;
	}
	for(int i=1;i<=k;i++)
	{
		ll si=0;
		for(int j=0;j<sr[i].size();j++)si+=sr[i][j],su[i].push_back(si);
	}
	for(int i=1;i<=k;i++)ins(i);
	int id=(*r2.begin()).second;
	del(id);ci[id]+=2;ins(id);
	printf("-1\n%lld\n",as);
	//+1
	//+2 -1
	//+3 -2
	//+2 +2 -3
	for(int i=3;i<=n;i++)
	{
		ll rv=-1e16,fr=0;
		if(r1.size())
		{
			ll v1=-(*r1.begin()).first;
			if(rv<v1)rv=v1,fr=1;
		}
		if(r2.size()&&s1.size())
		{
			ll v1=-(*r2.begin()).first-(*s1.begin()).first;
			if(rv<v1)rv=v1,fr=2;
		}
		if(r3.size()&&s2.size())
		{
			ll v1=-(*r3.begin()).first-(*s2.begin()).first;
			if(rv<v1)rv=v1,fr=3;
		}
		if(r2.size()>2&&s3.size())
		{
			ll v1=-(*r2.begin()).first-(*s3.begin()).first-(*(++r2.begin())).first;
			if(rv<v1)rv=v1,fr=4;
		}
		if(fr==1)
		{
			int v1=(*r1.begin()).second;
			del(v1);ci[v1]++;ins(v1);
		}
		if(fr==2)
		{
			int v1=(*r2.begin()).second,v2=(*s1.begin()).second;
			del(v1);ci[v1]+=2;ins(v1);
			del(v2);ci[v2]--;ins(v2);
		}
		if(fr==3)
		{
			int v1=(*r3.begin()).second,v2=(*s2.begin()).second;
			del(v1);ci[v1]+=3;ins(v1);
			del(v2);ci[v2]-=2;ins(v2);
		}
		if(fr==4)
		{
			int v1=(*r2.begin()).second,v2=(*(++r2.begin())).second,v3=(*s3.begin()).second;
			del(v1);ci[v1]+=2;ins(v1);
			del(v2);ci[v2]+=2;ins(v2);
			del(v3);ci[v3]-=3;ins(v3);
		}
		printf("%lld\n",as);
	}
}
```



#### SCOI2020模拟15

##### auoj149 第一题

###### Problem

给一个小写字母组成的字符串 $s$，长度为 $n$。

$q$ 次询问，每次给出 $k,p$，求字符串所有本质不同的子序列中字典序第 $k$ 小的子序列的后 $p$ 个字符

$q\leq 10^5,n\leq 3\times 10^5,k\leq 10^{18},\sum p\leq 10^6$

$2s,1024MB$

###### Sol

对于一个子序列，考虑将连续的字符缩在一起。这样在前 $|\sum|$ 段中，一定存在一段的字符大于下一段的字符。

删去这一段字符，考虑剩下的段，如果保留了下一段，那么后面无论怎么选都比原来的子序列字典序更小。考虑每一段选不选，因为相邻两段字符不同，只需要 $O(\log k)$ 段就存在大于 $k$ 个字典序小于这个子序列的子序列。

因此答案中相同字符构成段数不超过 $|\sum|+O(\log k)$。

考虑建出子序列自动机，考虑沿着同一种字符倍增，求出每个位置后面再接 $2^i$ 个与这个位置相同的字符时有多少个新增的小于它的子序列，并且加入这些字符后会到达哪个位置。询问时确定下一位字符后倍增确定这一段的长度即可，显然可以还原答案。

复杂度 $O(n(|\sum|+\log n)+q\log n(|\sum|+\log k))$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 300500
#define ll long long
int n,nt[N][26],f[N][19],s1[N][2],ct,q,l,s11;
long long dp[N],su[N][19],k;
char s[N];
int main()
{
	scanf("%s%d",s+1,&q);n=strlen(s+1);
	for(int i=0;i<26;i++)nt[n][i]=n+1;
	for(int i=n;i>=1;i--)
	{
		for(int j=0;j<26;j++)nt[i-1][j]=nt[i][j];
		nt[i-1][s[i]-'a']=i;
	}
	for(int i=n;i>=0;i--)
	{
		dp[i]=1;
		for(int j=0;j<26;j++)
		{
			dp[i]+=dp[nt[i][j]];
			if(dp[i]>1e18+100000)dp[i]=1e18+100000;
		}
	}
	for(int i=1;i<=n;i++)
	{
		su[i][0]=1;
		for(int j=0;j<s[i]-'a';j++)
		{
			su[i][0]+=dp[nt[i][j]];
			if(su[i][0]>1e18+1)su[i][0]=1e18+1;
		}
		f[i][0]=nt[i][s[i]-'a'];
	}
	f[n+1][0]=n+1;
	for(int i=1;i<=18;i++)
	for(int j=0;j<=n+1;j++)
	f[j][i]=f[f[j][i-1]][i-1],su[j][i]=su[j][i-1]+su[f[j][i-1]][i-1],su[j][i]=su[j][i]>1e18+1?1e18+1:su[j][i];
	while(q--)
	{
		scanf("%lld%d",&k,&l);
		int nw=0;ct=0;s11=0;
		if(k>dp[0]-1){printf("-1\n");continue;}
		while(k)
		{
			int st=0;
			for(int i=0;i<26;i++)
			if(dp[nt[nw][i]]>=k){st=i;break;}
			else k-=dp[nt[nw][i]];
			k--;nw=nt[nw][st];
			s1[++ct][0]=st;s1[ct][1]=1;
			for(int i=18;i>=0;i--)
			if(f[nw][i]<=n&&su[nw][i]<k&&su[nw][i]+dp[f[nw][i]]-1>=k)k-=su[nw][i],nw=f[nw][i],s1[ct][1]+=(1<<i);
			s11+=s1[ct][1];
		}
		s11-=l;if(s11<0)s11=0;
		for(int i=1;i<=ct;i++)
		{
			int tp=s1[i][1];if(tp>s11)tp=s11;
			s11-=tp;s1[i][1]-=tp;
			for(int j=0;j<s1[i][1];j++)printf("%c",s1[i][0]+'a');
		}
		printf("\n");
	}
}
```



##### auoj150 第二题

###### Problem

有 $n$ 张牌，每张牌有一个 $[0,d-1]$ 的颜色，保证 $d|n$，第 $i$ 张牌的分数为 $i$。

进行 $\frac nd$ 轮游戏，第 $i$ 轮拿出前 $id$ 张牌排成一个环，然后进行 $m$ 次操作，每次随机交换环上两张牌。然后第 $j(0\leq j<d)$ 个人拿走第 $j,j+d,...,j+(i-1)d$ 张牌，他这一轮的分数为所有颜色与 $j$ 相同的牌的分数和。

不同轮之间独立，即每一次的交换不会影响到初始牌的顺序。

求所有游戏后，每个人的分数期望，输出实数，相对误差不超过 $10^{-6}$。

$n\leq 3\times 10^6,d\leq 10,m\leq 10^{11}$

$2s,1024MB$

###### Sol

考虑一轮中一张牌的贡献，它有贡献当且仅当它向右移动的距离模 $d$ 结果为某个固定值。

可以发现操作对一张牌的位置来说相当于有 $\frac 1{id}$ 的概率向右一步，有 $\frac 1{id}$ 的概率向左一步，剩下的概率不移动。

看成生成函数，则这相当于 $(\frac{x+x^{-1}+id-2}{id})^m (\bmod x^d-1)$，单位根反演即可。单次复杂度 $O(d^2)$。注意到里面dft后一定虚部为 $0$，所以只算实数部分减小常数。

然后考虑一轮的情况，只需要对于所有 $i,j$ 知道颜色为 $i$ 且位置模 $d$ 余 $j$ 的牌的分数和，就可以由上述结果得到答案。前缀和即可。

复杂度 $O(nd+n\log m)$，注意常数和精度问题。

###### Code

```cpp
#include<cstdio>
#include<cmath>
using namespace std;
#define N 3005000
#define K 11
#define ll long long
int n,d;
ll m,su[K][K];
char st[N];
long double pi=acos(-1),as[K],st1[K],st2[K],w[K*K];
int main()
{
	scanf("%d%d%lld%s",&n,&d,&m,st+1);
	for(int i=0;i<=d*d;i++)w[i]=cos(2*pi/d*i);
	for(int i=1;i<=n/d;i++)
	{
		for(int j=1;j<=d;j++)su[st[i*d-d+j]-'0'][j-1]+=i*d-d+j;
		for(int j=0;j<d;j++)st1[j]=pow(1-2.0/i/d*(1-w[j]),m),st2[j]=0;
		for(int j=0;j<d;j++)for(int k=0;k<d;k++)st2[j]=st2[j]+st1[k]*w[j*k];
		for(int j=1;j<=d;j++)for(int k=0;k<d;k++)as[j]+=su[j-1][k]*st2[(j-1-k+d)%d]/d;
	}
	for(int i=1;i<=d;i++)printf("%.10lf\n",(double)as[i]);
}
```



##### auoj151 第三题

###### Problem

给两棵 $n$ 个点的有根树，它们的叶子数量相同。

在两棵树的根之间连有不可删去的边，剩余的每条树边有删去代价。

你需要将两棵树的叶子一一配对，每一对的叶子之间连上不可删去的边，然后删去一些原先的树边使得图变成一棵树。

求一种方案使得删去边的总代价和最小，输出最小总代价

$n\leq 10^5$

$2s,1024MB$

###### Sol

设各有 $m$ 个叶子，对于一个合法的方案，如果将叶子间的边全部删去，则树会变成 $m+1$ 个连通块，且每个连通块至少有一个叶子。

考虑 $m+1$ 个连通块，每个连通块至少有一个叶子。根据图的形态，最多有一个连通块有两棵树的叶子。如果存在这样一个块，可以从这一块开始扩展，优先连大小大于等于 $2$ 的，最后连剩下的，这样一定可以通过连接不同树的叶子将整个图连通。如果不存在这样的块，则考虑找两侧叶子最多的连通块连起来，然后使用上面的方式，可以发现也一定能做到。因此只需要构造满足上述条件的最小代价。

则问题变为，给一棵树，保留一些边使得有叶子的连通块数量不少于 $m+1$，求保留的最大边权和。

~~可以证明这是一个拟阵，所以说直接贪心选边就可以了。~~

考虑一个贪心，每次尝试加入最大的边，必须保证加边后有叶子的连通块不少于 $m+1$ 个。考虑证明这个贪心的正确性，一种方式是证明对于每一个 $k$，边权大于等于 $k$ 的边中这种方式选择的数量都是可能的最大值（即秩函数）

对于一个 $k$ ，考虑所有边权大于等于 $k$ 的边，它们一定连出了若干连通块。对于一个没有叶子的连通块，上述方式显然会选所有边。对于有叶子的连通块，最后可能有一些边没有被选。

在从大到小依次加入边时，如果某条边当前连接的是一个有叶子的连通块和一个没有叶子的连通块，那么考虑到这条边时一定会选。因此最后贪心选的边一定将这个连通块分成了若干个包含叶子的连通块，且没有选的边都是连接有叶子的连通块的边。

对于这些边，每选一条边包含叶子的连通块一定减一，因此没有选的边的数量一定是 $m+1$ 减去选了所有边权大于等于 $k$ 的边后包含叶子的连通块数量的结果对 $0$ 取 $\max$。

显然一种合法的方案至少要不选这么多条边，因此对于每一个 $k$，这种方案选的边权大于等于 $k$ 的边的数量一定是可能的最大值。因此这样的贪心一定最优。直接模拟贪心过程即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
int n,fa[N],su[N],s1,d[N];
long long as;
struct edge{int f,t,v;friend bool operator <(edge a,edge b){return a.v>b.v;}}e[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d%d",&e[i].f,&e[i].t,&e[i].v),d[e[i].f]++,d[e[i].t]++,as+=e[i].v;
	for(int i=2;i<=n;i++)if(d[i]==1)su[i]=su[i+n]=1,s1++;s1--;
	for(int i=n;i<n*2-1;i++)scanf("%d%d%d",&e[i].f,&e[i].t,&e[i].v),e[i].f+=n,e[i].t+=n,as+=e[i].v;
	sort(e+1,e+n*2-1);
	for(int i=1;i<=n*2;i++)fa[i]=i;fa[n+1]=1;
	for(int i=1;i<=n*2-2;i++)
	{
		int a=finds(e[i].f),b=finds(e[i].t);
		if(a==b)continue;
		if(!s1&&su[a]&&su[b])continue;
		s1-=su[a]&su[b];fa[b]=a;as-=e[i].v;su[a]|=su[b];
	}
	printf("%lld\n",as);
}
```



#### SCOI2020模拟16

##### auoj401 A

###### Problem

有 $n$ 个任务，第 $i$ 个任务只能在时间内 $[l_i,r_i]$ 内进行工作，需要完成的工作量为 $b_i$。

有 $m$ 个人进行工作，第 $i$ 个人单位时间可以完成 $k_i$ 的工作量。

每个人同一时刻只能进行一个工作，每个工作同一时刻只能被一个人进行。

找到最小的非负实数 $r$，使得将所有 $r_i$ 加上 $r$ 后，存在完成所有任务的方式。

多组数据

$T\leq 5,n,m\leq 30$

$1s,128MB$

###### Sol

显然问题有可二分性，考虑二分答案，变为判定是否合法。

将时间按照每个任务当前的端点分成 $2n$ 段，考虑一段时间内的情况，此时可以做的任务固定。

设这一段时间长度为 $l$，$v_1\geq v_2\geq v_m$。考虑每个任务在这一段内完成的工作量满足的限制，显然有如下结果：

对于任意的 $i$，任意 $i$ 个当前能做的任务完成的工作量之和不超过 $l*(v_1+\cdots+v_k)$。

可以发现这个也是充分必要条件，即只要满足这个条件，就可以找到方案。证明可以考虑贪心操作，第 $i$ 个人操作当前剩余工作量第 $i$ 大的位置，如果两个任务工作量相同了则这两个人交替做，多个任务相同时类似。~~具体细节省略~~

考虑如何简单地描述这个限制。经过一些尝试可以得到如下方式：

分成 $m$ 部分，第 $i$ 部分中需要将 $i*l*(v_i-v_{i+1})$ 的工作量分给所有工作，每个工作得到的工作量不超过 $l*(v_i-v_{i+1})$。

这样考虑 $k$ 个工作得到的工作量，在第 $i$ 部分中得到的量不超过 $l*\min(i,k)*(v_i-v_{i+1})$，相加即得到 $l*(v_1+\cdots+v_k)$。

对于一个满足之前条件的方案，考虑按照工作量从大到小从前往后，按照部分从后往前贪心分配，可以发现一定能造出方案。因此这是充分必要的。

因此可以得到一个网络流模型，判断是否能满流即可。

复杂度 $O(T*\log v*dinic)$，可以证明一个 $n^4m^2$ 的上界，但是它一次1ms都跑不到。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 1859
#define M 141800
int T,n,m,s[N][3],v[N];
int head[N],cnt,cur[N],dis[N],su;
struct edge{int t,next;double v;}ed[M];
void adde(int f,int t,double v)
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
		for(int i=head[u];i;i=ed[i].next)if(ed[i].v>1e-11&&dis[ed[i].t]==-1)
		{
			dis[ed[i].t]=dis[u]+1,qu.push(ed[i].t);
			if(ed[i].t==t)return 1;
		}
	}
	return 0;
}
double dfs(int u,int t,double f)
{
	if(u==t||!f)return f;
	double as=0,tp;
	for(int &i=cur[u];i;i=ed[i].next)
	if(dis[ed[i].t]==dis[u]+1&&ed[i].v>1e-11&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		ed[i].v-=tp;ed[i^1].v+=tp;
		as+=tp;f-=tp;
		if(f<1e-11)return as;
	}
	return as;
}
double si[N];
bool chk(double ti)
{
	for(int i=1;i<=n;i++)si[i*2-1]=s[i][0],si[i*2]=s[i][1]+ti;
	sort(si+1,si+n*2+1);
	su=(n*2-1)*m+2+n;
	cnt=1;for(int i=1;i<=su;i++)head[i]=0;
	double sr=0;
	for(int i=1;i<=n;i++)sr+=s[i][2],adde(su-n-2+i,su,s[i][2]);
	for(int i=1;i<n*2;i++)
	{
		double le=si[i+1]-si[i];
		for(int j=1;j<=m;j++)
		{
			adde(su-1,(i-1)*m+j,(m-j+1)*le*(v[j]-v[j-1]));
			for(int k=1;k<=n;k++)if(si[i]>=s[k][0]-1e-9&&si[i+1]<=s[k][1]+ti+1e-9)
			adde((i-1)*m+j,su-n-2+k,le*(v[j]-v[j-1]));
		}
	}
	while(bfs(su-1,su))
	sr-=dfs(su-1,su,1e9);
	return sr<=1e-6;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&m);
		for(int i=1;i<=n;i++)scanf("%d%d%d",&s[i][2],&s[i][0],&s[i][1]);
		for(int i=1;i<=m;i++)scanf("%d",&v[i]);
		sort(v+1,v+m+1);
		double lb=0,rb=3e6,as=rb;
		for(int t=1;t<=57;t++)
		{
			double mid=(lb+rb)/2;
			if(chk(mid))as=mid,rb=mid;
			else lb=mid;
		}
		printf("%.10lf\n",as);
	}
}
```



##### auoj402 B

###### Problem

求有多少个 $n$ 个点的有标号无根树满足如下条件：

1. 每个点度数不超过 $d$
2. 树上的每一条路径都是合法的

给定 $op\in\{0,1\}$，合法的限制为：

1. 当 $op=0$ 时，一条路径合法当且仅当这条路径经过的点的编号递增或者递减
2. 当 $op=1$ 时，一条路径合法当且仅当路径上存在一个点，这个点向两侧的路径同时递增或同时递减

答案对给定 $m$ 取模，$m$ 不一定是质数。

$n\leq 200$

$1s,128MB$

###### Sol

考虑 $op=0$ 的情况，如果有一个点度数大于 $2$，那么这个点连出的三条边中一定有两条边连向的点编号同时大于它或者同时小于它，这两个点的路径显然不合法。

因此这时路径只能是一条链，且显然编号只能递增（递减与递增的树相同），所以答案为 $1$（注意 $m=1$ 的情况）。



考虑 $op=1$，观察可以发现如下结论：树合法当且仅当存在一个点 $u$，使得 $u$ 的任意一个子树内所有点到这个点的路径全部递增或者全部递减。

满足这个条件的树显然合法。如果一棵树不满足条件，考虑一个点 $u$，存在一个点 $v$ 和 $v$ 的儿子 $w$，使得路径 $u\to\cdots\to v\to w$ 的权值以 $v$ 为转折点。

考虑以 $v$ 为根，由假设它的子树中一定有一个子树不满足递增或递减的条件。如果这个子树为 $u$ 在的子树，设路径为 $v\to\cdots\to x\to y$，其中 $x$ 为转折点。不妨设 $u\to\cdots\to v$ 的路径递减，那么 $v\to\cdots\to x$ 上递增而 $x\to y$ 递减。此时路径 $w\to v\to\cdots\to x\to y$ 先递增再递减再递增，矛盾。

否则，转折点一定在 $v$ 的子树内，考虑使用同样的方式找下去，直到变为上一种情况或者找到儿子全部为叶子的节点，此时一定是上面的情况。因此一定可以导出矛盾。

因此，一定存在一个点满足上述条件。



考虑有两个点 $u,v(u<v)$ 同时满足条件的情况。如果 $v$ 存在两个子树到它都是递增，设两个子树为的根为 $x,y$，则一定有一个子树中不含 $u$，不妨设为 $x$，则 $u\to\cdots\to v\to x$ 不满足上述条件，矛盾。因此以 $u$ 为根只有一个子树到它递增，同理以 $v$ 为根只有一个子树到它递减。

如果两个点 $u,v(u<v-1)$ 满足条件，根据上一条有所有 $[u+1,v-1]$ 的点都在 $(u,v)$ 的路径上或在路径某个点的子树上。如果一个点在子树上，设它是 $x$，它父亲为 $f$，那么因为 $u\to\cdots f->x$ 为递增路径有 $f<x$，又因为 $v\to\cdots\to f\to x$ 是递减路径有 $f>x$，矛盾。因此所有这部分点都在 $(u,v)$ 路径上，这些点显然只能顺序排列，因此这些点全部满足条件。

因此，所有满足条件的点必定是一段编号连续的点，这些点形成一条链且编号单调。

考虑点减边容斥，只需要算每个点合法的方案数之和，再对于每个 $i$ 减去边 $(i,i+1)$ 存在且两侧均合法的方案数。

考虑算 $i$ 合法的方案数。考虑 $i$ 向前的部分，限制为每个点的父亲编号大于它。从小到大加入每个点，以编号大的点为父亲，$[1,i-1]$ 这部分中每个点最多有 $d-1$ 个编号小于自己的儿子。

考虑依次加入点并决定父亲，设 $dp_{i,j}$ 表示加入了前 $i$ 个点，当前还有 $j$ 个点没有确定父亲的方案数。枚举加入点有多少儿子，那么有：

$$
dp_{i,j}=\sum_{k=0}^{j-1}dp_{i-1,j+k-1}C_{j+k-1}^k
$$

对于点 $i$，先处理两侧的情况，两侧没有决定的点都只能连向它。那么点 $i$ 的贡献为 $\sum_{j}\sum_{k}[j+k\leq d]dp_{i-1,j}dp_{n-i,k}$

对于一条边类似考虑，对于连接 $(i,i+1)$ 的边，它存在的方案数为 $\sum_{j\leq d-1}\sum_{k\leq d-1}dp_{i-1,j}dp_{n-i-1,k}$。

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 233
int n,d,m,k,dp[N][N],c[N][N];
int main()
{
	scanf("%d%d%d%d",&n,&d,&m,&k);
	if(d==1&&n>2){printf("0\n");return 0;}
	if(k==0||n==2){printf("%d\n",1%m);return 0;}
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)
	for(int j=1;j<i;j++)
	c[i][j]=(1ll*c[i-1][j]+c[i-1][j-1])%m;
	dp[0][0]=dp[1][1]=1;
	for(int i=2;i<=n;i++)
	for(int j=1;j<i;j++)
	for(int l=0;l<d&&j-l+1>0;l++)
	dp[i][j-l+1]=(dp[i][j-l+1]+1ll*dp[i-1][j]*c[j][l])%m;
	int as=0;
	for(int i=1;i<=n;i++)
	for(int j=0;j<=d;j++)
	for(int l=0;l<=d;l++)
	if(j+l<=d)
	as=(as+1ll*dp[i-1][j]*dp[n-i][l])%m;
	for(int i=1;i<n;i++)
	for(int j=0;j<d;j++)
	for(int l=0;l<d;l++)
	as=(as-1ll*dp[i-1][j]*dp[n-i-1][l]%m+m)%m;
	printf("%d\n",as);
}
```



##### auoj403 C

###### Problem

给定 $n$，你需要将 $1,1,2,2,\cdots,n,n$ 这 $2n$ 个数排成一个环，满足如下条件：

1. 对于值相同的两个数，它们在环上的距离为 $1,2$ 或者 $n$。
2. 环满足一种对称性。具体来说，如果 $u,v$ 位置颜色相同，则它们在环上对称得到的的位置 $u',v'$ 颜色必须相同。

定义一种方案的权值为：

1. 如果不存在在环上距离为 $n$ 的数对，则贡献为 $0$。
2. 否则，所有满足这个条件的数间环划分成了若干段，贡献为每一段内部元素个数的乘积（不计算划分的元素）。

求所有本质不同的方案的权值和，答案模 $998244353$。

称两种方案本质相同，如果可以通过重标号数字使得填的数字相同。

$n\leq 2\times 10^6$

$3s,512MB$

###### Sol

只考虑本质不同的方案可以看成，将 $2n$ 个位置配对连边，满足上述限制。

首先考虑暴力dp。所有跨过中心的线将环分成了若干段，相邻两段之间可能有跨过中间分界线的边。因为对称性，算方案数只用考虑环的一半的情况。

首先考虑一段内部的情况，可以发现内部一定由 `11` 和 `1212` 组成，因此有 $f_i=f_{i-2}+f_{i-4}$。

考虑从位置开始向左找到第一条距离为 $n$ 的线，从这条线开始 $dp$。枚举第一条线的端点左右两个点是否有连边，然后变为链上的情况。这样需要给第一段再额外乘一个段长度的贡献作为找开头的方案数。

设 $dp_{i,0/1}$ 表示前 $i$ 个点分成了若干段，最后一个点连向中心，最后一个点两侧的点是否连边时的所有情况权值和。转移时枚举下一段长度以及下一段连向中心的点两侧是否有连边，直接转移。

复杂度 $O(n^2)$，可以分治fft做到 $O(n\log^2 n)$。

可以发现这个递推式是常系数的，前面的 $f$ 有封闭形式，可以猜测答案也有封闭形式。

然后消元可以发现答案是一个 $16$ 阶线性递推，于是可以做到 $O(n)$

###### Code

```cpp
//n^2
/*
#include<cstdio>
using namespace std;
#define N 23333
#define mod 998244353
int n,f[N],dp[N][2][2];
int main()
{
	scanf("%d",&n);
	f[0]=f[2]=1;
	for(int i=2;i<=n;i+=2)f[i]=(f[i-2]+f[i-4])%mod;
	for(int i=1;i<=n;i++)
	if(i&1)
	dp[i+1][0][1]=dp[i+1][1][0]=1ll*(i+1)*i%mod*i%mod*f[i-1]%mod;
	else dp[i+1][0][0]=1ll*(i+1)*i%mod*i%mod*f[i]%mod,dp[i+1][1][1]=1ll*(i+1)*i%mod*i%mod*f[i-2]%mod;
	for(int i=2;i<=n;i++)
	for(int j=1;i+j+1<=n;j++)
	for(int t=0;t<2;t++)
	if(j&1)
	{
		dp[i+j+1][t][1]=(dp[i+j+1][t][1]+1ll*dp[i][t][0]*f[j-1]%mod*j%mod*j)%mod;
		dp[i+j+1][t][0]=(dp[i+j+1][t][0]+1ll*dp[i][t][1]*f[j-1]%mod*j%mod*j)%mod;
	}
	else
	{
		dp[i+j+1][t][1]=(dp[i+j+1][t][1]+1ll*dp[i][t][1]*f[j-2]%mod*j%mod*j)%mod;
		dp[i+j+1][t][0]=(dp[i+j+1][t][0]+1ll*dp[i][t][0]*f[j]%mod*j%mod*j)%mod;
	}
	printf("%d\n",(dp[n][0][0]+dp[n][1][1])%mod);
}
*/
#include<cstdio>
using namespace std;
#define mod 998244353
int n,as[2333333]={0,0,0,24,4,240,204,1316,2988,6720,26200,50248,174280,436904,1140888,3436404,8348748},tp[17]={0,0,4,8,998244352,16,998244343,4,998244341,998244305,26,998244309,15,998244337,998244349,998244349,998244352};
int main(){scanf("%d",&n);for(int i=17;i<=n;i++)for(int j=1;j<=16;j++)as[i]=(as[i]+1ll*as[i-j]*tp[j])%mod;printf("%d\n",as[n]);}
```



#### SCOI2020模拟17

##### auoj404 放送事故

###### Problem

有一张 $n$ 个点的无向图，进行如下操作：

```cpp
//初始若(i,j)有边，则 f[i][j]=1，否则 f[i][j]=0
for(int o=1;o<=lim;++o)
for(int i=1;i<=n;++i)
for(int j=1;j<=n;++j)if(i!=j)
f[i][j]=f[i][j]||(f[i][o]&&f[o][j]);
```

$T$ 组询问，每次给出 $n,m$，求有多少个 $n$ 个点的完全图满足 $lim=m$ 时执行上述代码得到的的 $f$ 与 $lim=n$ 时得到的 $f$ 相同，答案模 $10^9+7$。

$n\leq 200,T\leq 11451$

$5s,512MB$

###### Sol

可以发现操作相当于 `floyd`，第一部分相当于只执行部分 `floyd`。

根据 `floyd` 的性质，做 $k$ 轮时得到的 $f_{i,j}$ 为 $i,j$ 是否能只经过编号不超过 $k$ 的点相互到达（$i,j$ 不受编号限制）。

最后得到的结果为 $i,j$ 是否连通。因此图合法当且仅当对于任意 $i,j$，如果 $i,j$ 连通，则存在一条除去 $i,j$ 外剩余点编号不超过 $m$ 的路径连接它们。

因此考虑图的一个连通块，这个连通块中所有编号小于等于 $m$ 的点必须连通，否则不满足要求。

如果连通块中有一个编号小于等于 $m$ 的点，则有如下性质：对于任意一个编号大于 $m$ 的点，它必须和一个编号小于等于 $m$ 的点相连。如果不满足这个性质，则从这个点出发向上的路径上还有一个编号大于 $m$ 的点，矛盾。且只需要满足这个性质，就一定满足条件。

否则，如果连通块中所有点编号大于 $m$，则这个连通块内部在前 $k$ 轮不会有操作，因此这个连通块必须是一个完全图。



首先处理第二种情况，设 $f_j$ 表示 $j$ 个点分成若干完全图的方案，转移显然。

然后考虑处理第一种情况，设 $dp_{i,j}$ 表示小于等于 $m$ 的部分当前有 $i$ 个点，大于等于 $m$ 的部分当前有 $j$ 个点的方案数，显然答案为 $dp_{m,n-m}$。

使用第二种情况作为初值，则初值为 $dp_{0,i}=f_i$

考虑转移，加入一个第一种情况的连通块时，枚举有多少个小于等于 $m$ 的点，这部分内部需要连通。然后枚举有多少个大于 $m$ 的点，每个点需要和一个前面的点直接相连，这些点内部可以任意连。因此转移为：

$dp_{i,j}=\sum_{a=1}^{i-1}\sum_{b=1}^{j-1}C_{i-1}^{a-1}C_j^bdp_{i-a,j-b}g_a(2^a-1)^b2^{\frac{b(b-1)}2}$

其中 $g_a$ 表示 $a$ 个点的连通图数，可以容斥算。

复杂度 $O(n^4+T)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 205
#define mod 1000000007
int f[N],c[N][N],fr[N],T,n,m,g[N],dp[N][N],tp[N][N],tp2[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&T);
	n=200;
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)
	for(int j=1;j<i;j++)
	c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	for(int i=1;i<=n;i++)
	{
		f[i]=pw(2,i*(i-1)/2);
		for(int j=1;j<i;j++)
		f[i]=(f[i]-1ll*f[j]*c[i-1][j-1]%mod*pw(2,(i-j)*(i-j-1)/2)%mod+mod)%mod;
	}
	for(int i=1;i<=n;i++)tp2[i]=pw(2,i*(i-1)/2);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	tp[i][j]=pw(pw(2,i)-1,j);
	g[0]=1;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=i;j++)
	g[i]=(g[i]+1ll*g[i-j]*c[i-1][j-1])%mod;
	for(int i=0;i<=n;i++)dp[0][i]=g[i];
	for(int i=1;i<=n;i++)
	for(int j=0;j<=n;j++)
	if(i+j<=n)
	{
		for(int k=1;k<=i;k++)dp[i][j]=(dp[i][j]+1ll*dp[i-k][j]*f[k]%mod*c[i-1][k-1])%mod;
		for(int k=1;k<=i;k++)
		for(int l=1;l<=j;l++)
		dp[i][j]=(dp[i][j]+1ll*dp[i-k][j-l]%mod*f[k]%mod*tp[k][l]%mod*tp2[l]%mod*c[i-1][k-1]%mod*c[j][l])%mod;
	}
	while(T--)scanf("%d%d",&n,&m),printf("%d\n",dp[m][n-m]);
}
```



##### auoj405 圣经咏唱

###### Problem

给一棵 $n$ 个点，以 $1$ 为根的有根树，然后进行如下操作：

1. 选择任意条边，删掉这些边，然后删掉不与根连通的部分
2. 在每个剩余点上写一个数 $v_i$，要求 $v_u\geq\sum v_{son_{u}}$ 且 $v_1=s$

求最后不同的树的方案数模 $998244353$ ，两树不同当且仅当树形态不同或者有一个点写的数不同

$n\leq 10^5,s\leq 10^{18}$

$5s,1024MB$

###### Sol

考虑差分，设 $a_u=v_u-\sum v_{son_u}$，可以发现 $v_1=\sum a_u$，且 $a_u\geq 0$。

那么对于 $n$ 个点的树，方案数相当于将 $s$ 分成 $n$ 个数的方案数，显然是 $C_{s+n-1}^{n-1}$，可以用 $O(n\log mod)$ 的时间算出每一个 $n$ 的方案数。

问题变为对于每一个 $n$，求出最后剩下 $n$ 个点的树的方案数。



显然的暴力是设 $dp_{i,j}$ 表示 $i$ 为根的子树内有 $j$ 个点的方案数，考虑将其写成生成函数的形式，设 $f_u(x)$ 表示 $u$ 的 `dp` 对应的生成函数，转移式为：

$$
f_u(x)=1+x\prod f_{son_u}(x)
$$

$f_u(x)$ 的次数为 $u$ 的子树大小，注意到在树链剖分中，所有轻链的子树大小的和是 $O(n\log n)$ 的，考虑树剖优化转移。

设 $u$ 的重儿子为 $s_u$，对于一个点的轻儿子，可以先求出它们的 `dp`，再用分治 `FFT` 把它们的生成函数乘起来，这部分复杂度为 $O(n\log^3 n)$。

那么对于一条链，相当于 $f_u(x)=1+g_u(x)f_{s_u}(x)$，其中 $g_u(x)$ 为其它儿子的 `dp` 乘起来的结果。

注意到对于 $f_2(x)=1+g_2(x)f_1(x),f_3(x)=1+g_3(x)f_2(x)$，它等价于 $f_3(x)=(1+g_3(x))+g_2(x)g_3(x)f_1(x)$。这意味着可以将两个形如 $f_i(x)=h_i(x)+g_i(x)f_{i-1}(x)$ 的式子合并，得到同样的形式。

因为一条链的所有 $g_u(x)$ 的次数之和等于这条链上轻儿子子树大小和，因此所有链上这一部分的多项式次数和也是 $O(n\log n)$ 的，对于每条重链分治合并即可。

复杂度 $O(n\log^3 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 132001
#define mod 998244353
int n,ntt[N],rev[N],a,b,c[N],d[N],g[2][N*2],head[N],cnt,sz[N],tp[N],sn[N];
long long s;
vector<int> as[N],fu[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void pre()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<17;j<<=1)
	{
		int st=1,w=pw(3,(mod-1)/j);
		if(i==0)w=pw(w,mod-2);
		for(int k=0;k<j>>1;k++)g[i][j+k]=st,st=1ll*st*w%mod;
	}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=i;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][st]%mod;
		ntt[k]=(v1+v2)-(v1+v2>=mod?mod:0);
		ntt[k+(i>>1)]=v1-v2+(v1<v2?mod:0);
	}
	int inv=pw(s,t==0?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size()-1,s2=b.size()-1;
	if(s1+s2<=200)
	{
		for(int i=0;i<=s1+s2;i++)c[i]=0;
		for(int i=0;i<=s1;i++)
		for(int j=0;j<=s2;j++)
		c[i+j]=(c[i+j]+1ll*a[i]*b[j])%mod;
		vector<int> f;
		for(int i=0;i<=s1+s2;i++)f.push_back(c[i]);
		return f;
	}
	int l=1;while(l<=s1+s2+1)l<<=1;
	for(int i=0;i<l;i++)c[i]=d[i]=0;
	for(int i=0;i<=s1;i++)c[i]=a[i];
	for(int i=0;i<=s2;i++)d[i]=b[i];
	dft(l,c,1);dft(l,d,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*d[i]%mod;dft(l,c,0);
	vector<int> f;
	for(int i=0;i<=s1+s2;i++)f.push_back(c[i]);
	return f;
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	int s1=a.size()-1,s2=b.size()-1;
	if(s1>s2){for(int i=0;i<=s2;i++)a[i]=(a[i]+b[i])%mod;return a;}
	else {for(int i=0;i<=s1;i++)b[i]=(b[i]+a[i])%mod;return b;}
}
struct sth{vector<int> a,b;};
sth solve(int l,int r)
{
	if(l==r){sth tp;tp.b=fu[l];tp.a.push_back(1);return tp;}
	int mid=(l+r)>>1;
	sth v1=solve(l,mid),v2=solve(mid+1,r);
	return (sth){polyadd(v2.a,polymul(v1.a,v2.b)),polymul(v1.b,v2.b)};
}
vector<int> solve1(int l,int r)
{
	if(l==r)return fu[l];
	int mid=(l+r)>>1;
	return polymul(solve1(l,mid),solve1(mid+1,r));
}
void dfs0(int u,int fa)
{
	sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u],sz[u]+=sz[ed[i].t];
}
void dfs1(int u,int fa,int v)
{
	tp[u]=v;
	if(sn[u])dfs1(sn[u],u,v);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs1(ed[i].t,u,ed[i].t);
	int ct=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])fu[++ct]=as[ed[i].t];
	if(ct)as[u]=solve1(1,ct);
	else as[u].push_back(1);
	int s1=as[u].size();
	as[u].push_back(0);
	for(int i=s1-1;i>=0;i--)as[u][i+1]=as[u][i];as[u][0]=0;
	if(tp[u]==u)
	{
		int ct1=1,st=u;
		while(sn[st])st=sn[st],ct1++;
		int ct2=ct1;st=u;
		while(st)fu[ct2]=as[st],ct2--,st=sn[st];
		sth fuc=solve(1,ct1);
		as[u]=polyadd(fuc.a,fuc.b);
	}
}
int main()
{
	scanf("%d%lld",&n,&s);pre();
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs0(1,0);dfs1(1,0,1);
	int tp=1,as1=0;
	for(int i=1;i<=n;i++)
	{
		as1=(as1+1ll*tp*as[1][i])%mod;
		tp=1ll*tp*((s+i)%mod)%mod*pw(i,mod-2)%mod;
	}
	printf("%d\n",as1);
}
```



##### auoj406 闭幕雷鸡

###### Problem

提交答案。

有一个 $n\times m$ 的网格，每个位置有一个 $\in\{0,1\}$ 的权值 $v_{i,j}$。

现在对于每个位置，给出它周围 $9$ 个位置的权值和（如果出界，则对应位置权值为 $0$）。

你需要找到任意一组合法的权值。保证有解。

$n,m\leq 600$

###### Sol

暴力做法是直接 `dfs` 加上若干剪枝，可以获得 $50$ 分。

观察数据可以发现，剩下的点为了卡 `dfs`，让绝大部分中间的和都在 $4,5$ 之间，这样 `dfs` 的剪枝就很难有效。

但可以发现数据的构造方式非常像先找一个中间全 $4$ 或者中间全 $5$ 的矩阵，然后随机翻转一些位，这一点从输入中若干个 $3\times 3$ 的矩形可以看出来。

因此考虑找一个中间全 $4$ 或者中间全 $5$ 的解，优先 `dfs` 这个解的情况。可以发现全 $4$ 只能是如下 $3\times 3$ 循环：

```
011
100
100
```

全 $5$ 为翻转的情况，因此枚举 $9$ 种循环情况分别 `dfs`。可以发现找到正确方式后 `dfs` 可以 $0.1s$ 得到解。



但实际上可以发现有 $nm$ 个方程 $nm$ 个变量，因此可以考虑直接解出来。

首先考虑相邻四个 $3\times 3$，用主对角线两个之和减去另外两个之和，则可以得到 $v_{i+3,j+3}+v_{i,j}-v_{i,j+3}-v_{i+3,j}$。因为给出了边界上的值，因此这里 $i,j$ 可以从 $0$ 取到 $n-2$。

又因为左上边界外的一圈（$i=0$ 或者 $j=0$）上都是 $0$，因此这样可以解出所有 $(3x,3y)$ 位置的值。

考虑当前的第 $3i$ 行，这一行还剩 $1,2,4,5,7,8,\cdots$ 的位置没有求出来。通过比较 $(3i-2,y),(3i-1,y)$ 处的值，结合第 $0$ 行全部为 $0$，可以求出 $3i$ 行每连续三个位置的值之和。

又因为求出了上一部分，则可以求出剩余位置每相邻两个的和。除去一种情况（和全部为 $1$）外，剩余情况可以唯一确定这一行。对于最后一种情况，可以发现两种取法对每个格子周围 $9$ 个格子的和贡献相同。因此此时任意选择即可。

实际上也可以发现，直接考虑从每连续三个位置的值之和得到这一行的值，可以枚举前两个向后判断是否合法。对于任意一组满足条件的方案。它满足任意连续三个位置的和都和确定的值一样，因此任意方案满足条件，任取一组即可。

同理，对于 $3i$ 列也可以求出类似的结果。

此时还剩下模 $3$ 余 $1,2$ 的行列位置没有得到。如果只保留这些位置，则原来一个 $3\times 3$ 的矩形当前一定剩下一个 $2\times 2$ 的矩形。因此问题变为：

有一个 $n\times m$ 的 `01` 矩阵，现在给出每个 $2\times 2$ 位置的和，构成 $(n-1)\times (m-1)$ 矩阵。构造任意一个满足条件的解。这可以用类似差分约束的做法解决：

设此时矩阵为 $a$，则由这些 $2\times 2$ 进行组合，可以将 $a_{i,j}$ 表示为 $c+a_{1,j}*(-1)^i+a_{i,1}*(-1)^j+a_{1,1}*(-1)^{i+j}$ 的形式，则限制为右侧部分在 $[0,1]$ 之间。

枚举 $a_{1,1}$，则限制可以看成 $a_{1,j}*(-1)^i+a_{i,1}*(-1)^j$ 在一个区间内。因为权值的限制，可以发现可能的解数量只能是 $1,3$。对于唯一解部分，可以确定两个位置。否则限制可以看成 $a_{1,j}=b,a_{i,1}=c$ 不能被同时满足。

那么可以先确定唯一解部分，然后沿着一侧被满足的限制 `bfs`。此时剩余一些行列，剩余行列对这部分没有限制。可以发现，只要取 $a_{i,1}=(-1)^i,a_{1,i}=(-1)^{i+1}$，则这部分两项求和一定为 $0$，从而对于任意一个解数量为 $3$ 的部分这样一定合法，因此这样填即可得到解。

复杂度 $O(nm)$。有大量细节，对于 $n=3k,3k+1,3k+2$ 部分有不同需要注意的细节：

首先考虑确定 $3i$ 行部分：

1. 对于 $3k$ 情况，在确定 $3i$ 行时最后一个位置填了之后后面必须是 $0$。
2. 对于 $3k+1$ 情况，可以通过 $3i+1$ 行位置周围的值唯一确定最后一个位置，但这一行的计算需要特判。
3. 对于 $3k+2$ 情况，最后有一个位置 $3i+2$ 没有限制，因此考虑了 $3i+1$ 位置的限制后，最后可以是 $0$ 或 $1$。

然后考虑后半部分，如果行列不是 $3k+2$ 的情况，则利用原来的最后一行的和可以唯一确定最后一行的值。列同理。

从而有一个限制是固定最后一行，因此可以考虑将上面的过程对最后一行一列而不是第一行第一列做。

###### Code

~~数据里面没有 $n,m=3k+2$ 的情况，所以不太确定有没有写对~~

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 605
int n,m,as[N][N];
char s[N][N];
int sn,sm,s2[N][N],v2[N][N],vl[N][N];
void solve()
{
	for(int t=0;t<2;t++)
	{
		for(int i=1;i<=sn;i++)for(int j=1;j<=sm;j++)v2[i][j]=-1;
		v2[sn][sm]=t;
		if(n%3!=2)for(int i=sm-1;i>=1;i--)v2[sn][i]=s2[sn][i]-v2[sn][i+1];
		if(m%3!=2)for(int i=sn-1;i>=1;i--)v2[i][sm]=s2[i][sm]-v2[i+1][sm];
		for(int i=sn-1;i>=1;i--)for(int j=sm-1;j>=1;j--)
		{
			vl[i][j]=s2[i][j]-vl[i+1][j]-vl[i][j+1]-vl[i+1][j+1];
			int tp=vl[i][j]-v2[sn][sm]*((sn+sm-i-j)&1?-1:1);
			int f1=(sm-j)&1?1:-1,f2=(sn-i)&1?1:-1;
			int r1=0,r2=0;
			for(int s=0;s<2;s++)if(v2[i][sm]==-1||s==v2[i][sm])
			for(int t=0;t<2;t++)if(v2[sn][j]==-1||t==v2[sn][j])
			if(tp-f1*s-f2*t>=0&&tp-f1*s-f2*t<=1)r1|=1<<s,r2|=1<<t;
			if(v2[i][sm]==-1&&r1<3)v2[i][sm]=r1-1;
			if(v2[sn][j]==-1&&r2<3)v2[sn][j]=r2-1;
		}
		queue<pair<int,int> > qu;
		for(int i=sn-1;i>=1;i--)if(v2[i][sm]!=-1)qu.push(make_pair(i,sm));
		for(int i=sm-1;i>=1;i--)if(v2[sn][i]!=-1)qu.push(make_pair(sn,i));
		while(!qu.empty())
		{
			pair<int,int> nw=qu.front();qu.pop();
			int lx=nw.first,ly=nw.second;
			for(int t=1;t<lx||t<ly;t++)
			{
				int rx=lx,ry=ly;
				if(rx==sn)rx=t;else ry=t;
				if(rx>=sn||ry>=sm)continue;
				int i=rx,j=ry;
				vl[i][j]=s2[i][j]-vl[i+1][j]-vl[i][j+1]-vl[i+1][j+1];
				int tp=vl[i][j]-v2[sn][sm]*((sn+sm-i-j)&1?-1:1);
				int f1=(sm-j)&1?1:-1,f2=(sn-i)&1?1:-1;
				int r1=0,r2=0;
				for(int s=0;s<2;s++)if(v2[i][sm]==-1||s==v2[i][sm])
				for(int t=0;t<2;t++)if(v2[sn][j]==-1||t==v2[sn][j])
				if(tp-f1*s-f2*t>=0&&tp-f1*s-f2*t<=1)r1|=1<<s,r2|=1<<t;
				if(v2[i][sm]==-1&&r1<3&&r1)v2[i][sm]=r1-1,qu.push(make_pair(i,sm));
				if(v2[sn][j]==-1&&r2<3&&r2)v2[sn][j]=r2-1,qu.push(make_pair(sn,i));
			}
		}
		for(int i=1;i<sn;i++)if(v2[i][sm]==-1)v2[i][sm]=(sn-i+1)&1;
		for(int i=1;i<sm;i++)if(v2[sn][i]==-1)v2[sn][i]=(sm-i)&1;
		for(int i=sn-1;i>=1;i--)for(int j=sm-1;j>=1;j--)
		{
			int tp=vl[i][j]-v2[sn][sm]*((sn+sm-i-j)&1?-1:1);
			int f1=(sm-j)&1?1:-1,f2=(sn-i)&1?1:-1;
			v2[i][j]=tp-f1*v2[i][sm]-f2*v2[sn][j];
		}
		int fg=1;
		for(int i=1;i<=sn;i++)for(int j=1;j<=sm;j++)if(v2[i][j]<0||v2[i][j]>1)fg=0;
		if(!fg)continue;
		for(int i=1;i<=sn;i++)for(int j=1;j<=sm;j++)as[i+(i-1)/2][j+(j-1)/2]=v2[i][j];
		return;
	}
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	for(int i=1;i<=n;i++)s[i][m+1]=s[i][m];
	for(int i=1;i<=m;i++)s[n+1][i]=s[n][i];
	for(int i=3;i<=n;i+=3)for(int j=3;j<=m;j++)
	as[i][j]=as[i][j-3]+as[i-3][j]-as[i-3][j-3]-s[i-2][j-1]-s[i-1][j-2]+s[i-1][j-1]+s[i-2][j-2];
	if(m%3==1)for(int i=1;i<=n;i++)as[i][m/3*3+3]=as[i][m/3*3];
	if(n%3==1)for(int i=1;i<=m;i++)as[n/3*3+3][i]=as[n/3*3][i];
	for(int i=1;i<=n;i++)if(i%3&&(i<n||n%3==1))
	for(int j=3;j<=m;j+=3)as[i][j]=as[i][j-3]+s[i+1][j-1]-s[i+1][j-2]-as[i+3-i%3][j]+as[i+3-i%3][j-3];
	for(int i=1;i<=m;i++)if(i%3&&(i<m||m%3==1))
	for(int j=3;j<=n;j+=3)as[j][i]=as[j-3][i]+s[j-1][i+1]-s[j-2][i+1]-as[j][i+3-i%3]+as[j-3][i+3-i%3];
	for(int i=3;i<=n;i+=3)
	for(int t=0;t<2;t++)
	{
		int nw=t,fg=1;
		for(int j=1;j<=m;j++)if(j%3&&(j<m||m%3==1))
		nw=as[i][j]-nw,fg&=nw>=0&&nw<=1;
		fg&=!nw||m%3==2;
		if(fg)
		{
			nw=t;int nt=0;
			for(int j=1;j<=m;j++)if(j%3&&(j<m||m%3==1))
			nt=as[i][j]-nw,as[i][j]=nw,nw=nt;
			if(m%3==2)as[i][m]=nw;
			break;
		}
	}
	for(int i=3;i<=m;i+=3)
	for(int t=0;t<2;t++)
	{
		int nw=t,fg=1;
		for(int j=1;j<=n;j++)if(j%3&&(j<n||n%3==1))
		nw=as[j][i]-nw,fg&=nw>=0&&nw<=1;
		fg&=!nw||n%3==2;
		if(fg)
		{
			nw=t;int nt=0;
			for(int j=1;j<=n;j++)if(j%3&&(j<n||n%3==1))
			nt=as[j][i]-nw,as[j][i]=nw,nw=nt;
			if(n%3==2)as[n][i]=nw;
			break;
		}
	}
	sn=n-n/3;sm=m-m/3;
	for(int i=1;i<n;i++)for(int j=1;j<m;j++)
	{
		int tp=s[i+1][j+1]-'0';
		for(int k=i;k<i+3;k++)for(int l=j;l<j+3;l++)if((k%3==0)||(l%3==0))tp-=as[k][l];
		s2[i-(i-1)/3][j-(j-1)/3]=tp;
	}
	solve();
	for(int i=1;i<=n;i++,printf("\n"))for(int j=1;j<=m;j++)printf("%c",as[i][j]?'X':'O');
}
```



#### NOI2020模拟六校联测6

1000%阴间场 ~~爆搜题 超级卡常题 全场只有一个人理解对题意的题~~

~~这样一看，T1最阳间~~

##### auoj407 mst

###### Problem

给一个 $n$ 个点 $m$ 条边的图，求第 $k$ 小生成树的边权和。

$n\leq 50,m\leq 1000,k\leq 10^4$

$2s,512MB$

###### Sol

考虑更广泛的一种问题：给一个拟阵，求权值和第 $k$ 小的基。为了简便，不妨设每个元素权值不同。

则有如下性质：对于一个不是最小的基 $S$，存在一个权值和更小的基 $T$，使得 $S$ 可以删去一个元素，再加入一个元素得到 $T$。

证明考虑基交换性质，将 $S$ 中最大的元素换出，换入一个最小基里面的元素。显然 $S$ 中最大元素比最小的基中的任意元素都大。

那么可以考虑先找到最小的基，然后在之前找到的基的基础上扩展，每次换出一个元素换入一个更大的元素。

为了避免重复，可以钦定换出的元素必须是初始的元素，且换出的编号递增，同时加入的元素也递增。使用标号就可以判断是否合法。

可以对于每种情况枚举可能的换出的元素，然后从小到大考虑换入的元素，每次找到下一个能换入的元素，这种方案被取出的时候再去取下一个。

如果直接暴力向后枚举，用 `set` 维护后面的方案。如果加入元素的判定独立集是 $O(1)$ 的，则复杂度为 $O(mk\log k+nmk)$，这里 $n$ 为拟阵的秩。本题中可以使用这种方式，因为这部分跑不满可以通过。



考虑找到一个方案时有两个问题，第一个是对于此时新的换出方案找第一个能加入的边，第二个是对于当前方案找下一个能加入的边。后者可以暴力做，复杂度 $O(mk)$。考虑前者，相当于给一棵树，对于每条边求出删去这条边后编号大于某个值且最小的能加入的边。从小到大考虑后面的边，相当于将路径上没有被覆盖的边覆盖，那么维护树上并查集缩点即可。这样复杂度即为 $O(mk\log k)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<set>
using namespace std;
#define N 55
#define M 2333
#define K 10500
#define ll long long
#define mod 1000000009
int n,m,k,res,fa[N],tp[N],is[M],as1,ct,head[N],cnt,fa2[N],ct3=0,fg,vl2[M],ct5,f[N][N];
vector<int> st;
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct edge2{int f,t,v;friend bool operator <(edge2 a,edge2 b){return a.v<b.v;}}s[M];
vector<int> fu[K];
ll fu2[K];
set<pair<int,int> > vl;
set<int> fuc2;
void dfs1(int u,int fa,int v)
{
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)f[ed[i].t][v]=max(f[u][v],s[ed[i].v].v),dfs1(ed[i].t,u,v);
}
void dfs2(int u,int fa,int t)
{
	if(u==t){fg=1;return;}
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa&&!fg)
	{
		st.push_back(ed[i].v);
		dfs2(ed[i].t,u,t);
		if(!fg)st.pop_back();
	}
}
int main()
{
	for(int i=1;i<=2000;i++)vl2[i]=((1ll*rand())<<32|rand())%mod;
	scanf("%d%d%d",&n,&m,&k);res=k;
	for(int i=1;i<=m;i++)scanf("%d%d%d",&s[i].f,&s[i].t,&s[i].v);
	sort(s+1,s+m+1);
	for(int i=1;i<=n;i++)fa[i]=i;
	for(int i=1;i<=m;i++)
	{
		int f1=finds(s[i].f),f2=finds(s[i].t);
		if(f1!=f2)fa[f1]=f2,as1+=s[i].v,tp[++ct]=i,is[i]=ct;
	}
	vl.insert(make_pair(as1,0));
	while(res)
	{
		if(!vl.size()){printf("-1\n");return 0;}
		pair<int,int> fuc=*vl.begin();vl.erase(fuc);
		res--;as1=fuc.first;
		if(!res)break;
		int tp2=fuc.second;
		for(int i=1;i<=n;i++)fa[i]=i;
		for(int i=0;i<fu[tp2].size();i++)fa[finds(s[fu[tp2][i]].f)]=finds(s[fu[tp2][i]].t);
		for(int i=1;i<=n;i++)fa2[i]=finds(i),head[i]=0;cnt=0;
		for(int i=1;i<n;i++)if(~fu2[tp2]>>i&1)
		adde(fa2[s[tp[i]].f],fa2[s[tp[i]].t],tp[i]);
		for(int i=1;i<=n;i++)dfs1(i,0,i);
		int rb=1;
		if(fu[tp2].size())rb=fu[tp2][fu[tp2].size()-1]+1;
		int ha1=0;
		for(int l=0;l<fu[tp2].size();l++)ha1=(ha1+vl2[fu[tp2][l]])%mod;
		for(int i=1;i<n;i++)if(fu2[tp2]>>i&1)ha1=(mod+ha1-vl2[tp[i]])%mod;
		for(int i=rb;i<=m;i++)if(!is[i])
		{
			int v1=fa2[s[i].f],v2=fa2[s[i].t];
			if(v1==v2||(vl.size()==res&&as1+s[i].v-f[v1][v2]>=(*vl.rbegin()).first))continue;
			fg=0;dfs2(v1,0,v2);
			for(int j=0;j<st.size();j++)
			{
				int tp=st[j];
				int as2=as1+s[i].v-s[tp].v;
				if(vl.size()==res&&(*vl.rbegin()).first<=as2)continue;
				int ha2=(1ll*ha1+vl2[i]-vl2[tp]+mod)%mod;
				if(fuc2.count(ha2))continue;
				fuc2.insert(ha2);
				if(vl.size()<res)
				{
					fu[++ct3]=fu[tp2];
					fu[ct3].push_back(i);
					fu2[ct3]=fu2[tp2];
					fu2[ct3]|=1ll<<is[tp];
					vl.insert(make_pair(as2,ct3));
				}
				else
				{
					int tp1=(*vl.rbegin()).first;
					if(tp1<=as2)continue;
					int id3=(*vl.rbegin()).second;vl.erase(*vl.rbegin());
					fu2[id3]=fu2[tp2];
					fu2[id3]|=1ll<<is[tp];
					fu[id3]=fu[tp2];fu[id3].push_back(i);vl.insert(make_pair(as2,id3));
				}
			}
			st.clear();vector<int>().swap(st);
		}
	}
	printf("%d\n",as1);
}
```



##### auoj408 string

###### Problem

给一个长度为 $n$ 的 `01` 串 $s$，其中有些位置已经确定，另外一些位置没有确定。

有 $q$ 个限制，每个限制给出 $a,b,c$，需要满足 $s_{a,...,a+c-1}=s_{b,...,b+c-1}$。

求字典序最小的合法解，保证有解

$n,q\leq 10^6$

$1s,256MB$

###### Sol

直接的方式是拆成 $nq$ 个两个位置相同的限制，然后并查集。但显然不能直接做，考虑如何快速维护。

考虑类似于 `rmq` 的倍增处理方式，将每个限制拆成两段，每一段形如 $s_{a,...,a+2^i-1}=s_{b,...,b+2^i-1}$。

可能的 $2^i$ 只有 $\log n$ 种，考虑对于每一种使用并查集维护。

在 $2^i$ 这一层中，如果 $a$ 与 $b$ 在并查集中连通，说明 $s_{a,...,a+2^i-1}=s_{b,...,b+2^i-1}$。而这相当于 $s_{a,...,a+2^{i-1}-1}=s_{b,...,b+2^{i-1}-1},s_{a+2^{i-1},...,a+2^{i-1}+2^{i-1}-1}=s_{b+2^{i-1},...,b+2^{i-1}+2^{i-1}-1}$，因此可以将一层的一个标记看成下一层的两个标记。

而一层的连通性可以用 $O(n)$ 个标记表示，下传时只传有用的标记即可。求出相等关系的并查集后容易构造方案。

复杂度 $O(q\log n+n\log n\alpha(n))$。注意卡常。

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 1000600
int f[20][N],sz[20][N],f1[N],n,m,a,b,l,vl[N];
char st[N];
int finds(int d,int x){return f[d][x]==x?x:f[d][x]=finds(d,f[d][x]);}
void merge(int d,int x,int y)
{
	x=finds(d,x);y=finds(d,y);if(x==y)return;
	if(sz[d][x]<sz[d][y])x^=y^=x^=y;
	sz[d][x]+=sz[d][y];f[d][y]=x;
}
inline char gc() {
    static char buf[1000000], *p1, *p2;
    return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1000000, stdin), p1 == p2) ? EOF : *p1++;
}

template<class T> inline void rd(T &x) {
    x = 0; char c = gc();
    while (c < '0' || c > '9') c = gc();
    while ('0' <= c && c <= '9') x = x * 10 + c -'0', c = gc();
}
int main()
{
	scanf("%s",st+1);n=strlen(st+1);
	for(int i=1;i<=n;i++)
	for(int j=0;j<20;j++)
	f[j][i]=i,sz[j][i]=1;
	scanf("%d",&m);
	for(int i=1;i<=m;i++)
	{
		rd(a);rd(b);rd(l);
		int st=0;while((1ll<<(st+1))<=l)st++;
		merge(st,a,b);merge(st,a+l-(1<<st),b+l-(1<<st));
	}
	for(int i=19;i>0;i--)
	for(int j=1;j+(1<<i)-1<=n;j++)
	if(f[i][j]!=j)
	{
		int s1=j,s2=f[i][j];
		merge(i-1,s1,s2);
		merge(i-1,s1+(1<<i-1),s2+(1<<i-1));
	}
	for(int i=1;i<=n;i++)if(st[i]=='1')f1[finds(0,i)]=1;
	for(int i=1;i<=n;i++)printf("%d",f1[finds(0,i)]);
}
```



##### auoj409 tree

###### Problem

给一棵 $n$ 个点的树，每个点有点权。支持 $q$ 次操作：

1. 增加一个点的点权。
2. 给定 $x$，求只保留点权大于等于 $x$ 的点时的连通块数量。

$n,q\leq 5\times 10^5$

$2s,512MB$

###### Sol

考虑点减边，则在每个点权值处 $+1$，每条边两侧点权 $\max$ 处 $-1$，$x$ 的后缀和即为答案。

点权部分可以直接维护。考虑维护两侧点权的 $\max$，但 $\max$ 可能改变 $O(n^2)$ 次。

修改一个点点权时，对于之前就以这个点为 $\max$ 的边，可以记录这部分的次数，然后一起修改。只需要处理 $\max$ 方向切换的边。

对于一条边，设它两侧的点修改的次数为 $a,b$，则可以发现在增加的过程中，$\max$ 方向切换的次数显然不超过 $2*\min(a,b)+1$。

由于这是一棵树，考虑取一个根，将 $\min$ 放缩到儿子的修改次数，则可以发现总的切换次数不超过 $O(n+q)$。

因此只需要暴力处理切换的情况，对于每个点维护相邻但不以这个点取 $\max$ 的边，修改时从这些边中找到对侧点点权最小的尝试改变方向。

但还有一个问题，如果用 `set` 维护所有大于它的相邻点点权，则修改边时会改变所有与这个点相邻的点的这个点权，这样复杂度不行。考虑修改点权时不修改 `set` 中的对应值，在找最小的点权时，考虑找到当前 `set` 的最小元素，如果这个元素对应的点权已经被改过了，则放回真的点权重试一次。可以发现每条边导致的重试次数不超过 $2*\min(a,b)$，因此这部分次数也是 $O(q)$。

复杂度 $O((n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 500500
priority_queue<pair<int,int> > s1[N];
int n,q,a,b,c,v[N],tr[N*2],ct[N];
void modify(int x,int v){for(int i=x;i;i-=i&-i)tr[i]+=v;}
int query(int x){int as=0;for(int i=x;i<=1e6;i+=i&-i)as+=tr[i];return as;}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),modify(v[i],1);
	for(int i=1;i<n;i++)
	{
		scanf("%d%d",&a,&b);if(v[a]>v[b])a^=b^=a^=b;
		modify(v[a],-1);s1[a].push(make_pair(-v[b],b));ct[a]++;
	}
	while(q--)
	{
		scanf("%d%d",&a,&b);
		if(a==1)
		{
			scanf("%d",&c);
			while(!s1[b].empty())
			{
				pair<int,int> t1=s1[b].top();s1[b].pop();
				if(v[t1.second]!=-t1.first){t1.first=-v[t1.second];s1[b].push(t1);continue;}
				if(-t1.first>=c){s1[b].push(t1);break;}
				modify(v[b],1);modify(v[t1.second],-1);s1[t1.second].push(make_pair(-c,b));ct[t1.second]++;ct[b]--;
			}
			modify(v[b],ct[b]-1);modify(c,1-ct[b]);v[b]=c;
		}
		else printf("%d\n",query(b));
	}
}
```



##### auoj411 sequence

###### Problem

定义一个序列是好的，当且仅当 $\forall 1<i<n,a_i=a_{i-1}+a_{i+1}$。

给一个长度为 $n$ 的序列 $b$，你需要找到一个长度相同的好的序列 $a$，使得 $\sum|a_i-b_i|$ 最小，输出这个最小值。

$n\leq 3\times 10^5,|b_i|\leq 10^9$

$1s,64MB$

###### Sol

设序列前两项为 $-x,y$，那么好的序列前几项为 $-x,y,x+y,x,-y,-x-y,-x,y,x+y,x,-y,-x-y,...$

因此好的序列一定循环，且 $6$ 是一个循环节。

只考虑模 $3$ 余 $1$ 的位置的这些项，这些项对 $\sum|a_i-b_i|$ 的贡献是一个关于 $x$ 的下凸函数。

考虑模 $3$ 余 $2$ 的位置，贡献是一个关于 $y$ 的下凸函数。

对于模 $3$ 余 $0$ 的位置，贡献是关于 $x+y$ 的下凸函数。

通过对前两个凸函数做 $\min$ 卷积，可以得到 $x+y=k$ 时，前两部分的最小贡献。这也是一个下凸函数。下凸函数做 $\min$ 卷积相当于将所有斜率段按照斜率排序拼接。

然后和第三部分对应位置相加并找出整体 $\min$ 即可。下凸函数相加显然还是下凸函数，因此也可以相加后再找 $\min$。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,v[N],v1[N],v2[N],v3[N],ct1,ct2,ct3,ct11,ct21,ct31;
vector<pair<int,long long> > f1,f2,f3,f4;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&v[i]);
		if(i%6==1)v1[++ct1]=-v[i];
		if(i%6==2)v2[++ct2]=v[i];
		if(i%6==3)v3[++ct3]=v[i];
		if(i%6==4)v1[++ct1]=v[i];
		if(i%6==5)v2[++ct2]=-v[i];
		if(i%6==0)v3[++ct3]=-v[i];
	}
	sort(v1+1,v1+ct1+1);sort(v2+1,v2+ct2+1);sort(v3+1,v3+ct3+1);
	ll res=0,las=-1e9,h1=-ct1;
	for(int i=1;i<=ct1;i++)res+=1e9+v1[i];
	v1[++ct1]=1e9;
	f1.push_back(make_pair(las,res));
	for(int i=1;i<=ct1;i++)
	{
		res+=h1*(v1[i]-las);h1+=2;
		if(las!=v1[i])f1.push_back(make_pair(v1[i],res)),ct11++;
		las=v1[i];
	}
	res=0,las=-1e9,h1=-ct2;
	for(int i=1;i<=ct2;i++)res+=1e9+v2[i];
	v2[++ct2]=1e9;
	f2.push_back(make_pair(las,res));
	for(int i=1;i<=ct2;i++)
	{
		res+=h1*(v2[i]-las);h1+=2;
		if(las!=v2[i])f2.push_back(make_pair(v2[i],res)),ct21++;
		las=v2[i];
	}
	res=0,las=-1e9,h1=-ct3;
	for(int i=1;i<=ct3;i++)res+=1e9+v3[i];
	v3[++ct3]=1e9;
	f3.push_back(make_pair(las,res));
	for(int i=1;i<=ct3;i++)
	{
		res+=h1*(v3[i]-las);h1+=2;
		if(las!=v3[i])f3.push_back(make_pair(v3[i],res)),ct31++;
		las=v3[i];
	}
	ct1=ct11;ct2=ct21;ct3=ct31;
	int l0=0,r0=0;
	f4.push_back(make_pair(f1[l0].first+f2[r0].first,f1[l0].second+f2[r0].second));
	for(int i=1;i<=ct1+ct2;i++)
	{
		if(l0==ct1)r0++;
		else if(r0==ct2)l0++;
		else if(1ll*(f1[l0].second-f1[l0+1].second)*(f2[r0+1].first-f2[r0].first)>=1ll*(f2[r0].second-f2[r0+1].second)*(f1[l0+1].first-f1[l0].first))l0++;
		else r0++;
		f4.push_back(make_pair(f1[l0].first+f2[r0].first,f1[l0].second+f2[r0].second));
	}
	int ct4=ct1+ct2,v1=0,v2=0,nw;
	ll as=1e18;
	while(v1<ct4||v2<ct3)
	{
		if(f4[v1+1].first<f3[v2+1].first)v1++,nw=f4[v1].first;
		else v2++,nw=f3[v2].first;
		if(v1==ct4||v2==ct3)break;
		as=min(as,f4[v1].second+1ll*(nw-f4[v1].first)*((f4[v1+1].second-f4[v1].second)/(f4[v1+1].first-f4[v1].first))+f3[v2].second+1ll*(nw-f3[v2].first)*((f3[v2+1].second-f3[v2].second)/(f3[v2+1].first-f3[v2].first)));
	}
	printf("%lld\n",as);
}
```



##### auoj412 antimatter

###### Problem

有 $n$ 种实验，第 $i$ 种实验费用为 $c_i$，它会随机生成 $[l_i,r_i]$ 中一个整数数量的反物质。

你最多可以存储 $k$ 个单位的反物质，且你的操作必须满足不可能超过这个上界。即如果当前你有 $x$ 个单位的反物质，则只能选择 $r_i\leq k-x$ 的实验。

如果最后生成了 $x$ 个单位的反物质，则收益为 $10^9*x$ 减去实验的总费用。

你需要选择策略最大化最坏情况的收益。求最坏情况收益的最大值。

$n\leq 100,k\leq 2\times 10^6$

$2s,128MB$

###### Sol

考虑设 $dp_i$ 表示当前有 $i$ 个单位时，后面的最大收益。转移枚举接下来做哪种实验，有：

$$
dp_i=\max(i*10^9,\max_{j,i+r_j\leq m}((\min_{x=i+l_j}^{i+r_j}dp_x)-c_j))
$$

直接的想法是维护 $n$ 个单调队列，但这样空间 $O(nk)$，无法接受。~~虽然说因为数据问题，这样只需要16MB，但是deque的常数问题非常大导致TLE~~

考虑一种单调队列的替代方式。记录当前区间内的最小值位置 $nw$，对于每个位置记录它左侧第一个小于它的位置 $ls_i$。考虑区间向左一位后的情况：

1. 如果 $nw$ 不在区间内，则 $nw$ 也向左一位。
2. 如果 $ls_{nw}$ 大于等于区间左端点，则将 $nw$ 变为 $ls_{nw}$，重复这个过程直到不满足条件。
   
可以发现，进行第二步操作后，$nw$ 一定是区间前缀最小值。再结合 $nw$ 是之前区间的最小值，可以得到现在的 $nw$ 是现在的区间最小值。

这样 $ls$ 可以在 `dp` 的过程中单调栈求出。这种方式需要 $O(k)$ 的公共空间和每个单调队列 $O(1)$ 的空间。

复杂度 $O(nk)$，空间复杂度 $O(n+k)$，常数非常小。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define M 2005000
#define ll long long
int n,m,s[N][3],st[M],rb,ls[M],nw[N];
ll dp[M];
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]),nw[i]=m;
	for(int i=1;i<=m;i++)dp[i]=1e9*i;
	for(int i=m;i>=0;i--)
	{
		for(int j=1;j<=n;j++)if(i+s[j][1]<=m)
		{
			if(nw[j]>i+s[j][1])nw[j]--;
			while(ls[nw[j]]>=i+s[j][0])nw[j]=ls[nw[j]];
			if(dp[i]<dp[nw[j]]-s[j][2])dp[i]=dp[nw[j]]-s[j][2];
		}
		while(rb&&dp[st[rb]]>=dp[i])ls[st[rb]]=i,rb--;
		st[++rb]=i;
	}
	printf("%lld\n",dp[0]);
}
```



##### auoj413 游走

###### Problem

对于一个序列，考虑如下游戏：

你一开始随机出现在一个位置上，每一次你可以进行如下操作：

1. 获得序列这个位置的值的分数，游戏结束。
2. 以 $\frac12$ 的概率移动到左边，$\frac12$ 的概率移动到右边。在两端时无法选择这样操作。

给一个长度为 $n$ 的序列，对于它的每一个前缀求出，在这个前缀上进行上面的游戏，最优策略下的期望得分，答案。模 $998244353$。

$n\leq 5\times 10^5$

$2s,1024MB$

###### Sol

设 $f_i$ 表示 $i$ 处开始的期望最优答案，$v_i$ 表示这里的数。则 $f_i=\min(v_i,\frac{f_{i-1}+f_{i+1}}2)$。如果 $v_i<\frac{f_{i-1}+f_{i+1}}2$，那么这个位置一定会选择操作 $2$，否则一定选择操作 $1$。

因此策略一定可以看成，存在若干个位置，到达这些位置就停止。

考虑一个点 $i$ 出发的情况，设左右两个停止位置为 $l,r$，有 $f_l=v_l,f_r=v_r,f_i=\frac{f_{i-1}+f_{i+1}}2(l<i<r)$。

容易发现 $f_i=\frac{(r-i)v_l+(i-l)v_r}{r-l}$，那么相当于在 $(l,v_l),(r,v_r)$ 连了一条线，$f_i$ 为这条线在 $x=i$ 时的高度。这里也可以计算从这个位置出发在左侧停止的概率，概率为 $\frac {le-i}{le}$。

那么选一些位置停止相当于将这些位置对应的点连起来。问题相当于在 $n$ 个点 $(i,v_i)$ 上找一条向右的折线，使得折线下的面积最大。

那么容易发现上凸壳是最优的，直接单调栈维护即可，这样即可对于每个前缀都求出答案。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500500
#define mod 998244353
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int n,st[N],rb;
long long dp[N],v[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	dp[1]=v[1]*2;
	st[rb=1]=1;
	for(int i=2;i<=n;i++)
	{
		while(rb>1&&1ll*(v[i]-v[st[rb-1]])*(st[rb]-st[rb-1])>=1ll*(v[st[rb]]-v[st[rb-1]])*(i-st[rb-1]))rb--;
		dp[i]=1ll*(v[i]+v[st[rb]])*(i-st[rb]+1)-v[st[rb]]*2+dp[st[rb]];st[++rb]=i;
	}
	for(int i=1;i<=n;i++)printf("%d ",dp[i]%mod*pw(2*i,mod-2)%mod);
}
```



#### SCOI2020模拟20

##### auoj414 s1mple的矩阵

###### Problem

给一个 $n\times n$ 的 `01` 矩阵 $B$。$q$ 次询问，每次给一个长度为 $n-1$ 的序列 $a$，求有多少个 $n$ 阶排列 $p$ 满足如下条件：

$$
\forall 0<i<n,B_{p_i,p_{i+1}}=a_i
$$

$n\leq 17,q\leq 10^5$

$1s,512MB$

###### Sol

考虑容斥，容斥后变为要求一些 $i$ 的位置必须是 $1$，剩下的位置没有限制，求方案数。

所有 $1$ 构成了若干段，设它们的长度依次为 $s_1,...,s_k$。则要求为找一个经过所有点的路径，路径前 $s_1$ 个点中相邻两个点的边为 $1$，接下来 $s_2$ 个点中相邻两个点的边为 $1$，依次类推。不同段之间没有限制。

则 $s_i$ 之间的顺序不影响答案，所以可以只考虑 $n$ 的每一种划分对应的 $s_i$。

对于一组 $s_1,...,s_m$，先预处理出对于每个集合 $S$，找一条路径经过 $S$ 内所有点，且每条边在矩阵上都为 $1$ 的方案数，这可以状压 `dp` 做到 $O(n^22^n)$。

然后相当于找出 $k$ 个集合 $S_1,...,S_k$，使得 $\forall i,|S_i|=s_i$ 且所有集合不交，并集为全集，这样的方案数为每个集合内选路径方案数的乘积。

因为 $\sum s_i=n$，所以只要保证了并是全集，那么一定不交。因此只需要考虑并是全集的条件。

对于每一个 $i$，可以将满足 $|S|=i$ 的集合和方案数写成集合幂级数，然后将 $k$ 部分个集合幂级数乘起来即可。

先预处理出每个大小的集合幂级数 `FWT` 之后的结果，对于一组 $s_1,...,s_m$ 只需要点值相乘然后 `IFWT`。但注意到只需要全集一项的值，因此可以不做最后的 `IFWT`。这样即可求出容斥后的答案。最后容斥回去即可。

如果直接枚举划分再做上面的东西。则无论做不做 `IFWT` 复杂度都是 $O(p(n)n2^n+n^22^n)$

更加快速的方式是 `dfs` 枚举划分。但直接 `dfs` 复杂度还是会剩下一个 $n$，因为 `dfs` 到一堆 $1$ 的时候，每一项都会乘一次。考虑判掉这种情况，如果当前可以放 $1,2$，则至少有两种方案，这样之前这些操作对复杂度的影响就可以无视掉。从而复杂度可以做到 $O(p(n)2^n+n^22^n)$

###### Code

Sol1:

```cpp
#include<cstdio>
using namespace std;
#define N 18
#define M 132001
#define ll long long
int n,q,vis[M*10],bitc[M];
ll as[M*10],as1[M],dp[M][N],f[M],fu[N][M],vl[M],ct[N];
char st[N][N],qu[N];
ll solve(int tp)
{
	for(int i=1;i<=n;i++)ct[i]=0;
	int ls=1;
	for(int i=1;i<n;i++,tp>>=1)
	if(tp&1)ls++;
	else ct[ls]++,ls=1;
	ct[ls]++;
	int fg=0;
	for(int i=n;i>=1;i--,fg<<=1)
	for(int j=1;j<=ct[i];j++)fg=fg<<1|1;
	if(vis[fg])return as[fg];
	for(int i=0;i<1<<n;i++)vl[i]=1;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=ct[i];j++)
	for(int k=0;k<1<<n;k++)
	vl[k]*=fu[i][k];
	for(int j=2;j<=1<<n;j<<=1)
	for(int k=0;k<1<<n;k+=j)
	for(int l=k;l<k+(j>>1);l++)
	vl[l+(j>>1)]-=vl[l];
	vis[fg]=1;return as[fg]=vl[(1<<n)-1];
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%s",st[i]+1);
	for(int i=1;i<=n;i++)dp[1<<i-1][i]=1;
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)if(dp[i][j])
	for(int k=1;k<=n;k++)if(st[j][k]=='1'&&!(i&(1<<k-1)))dp[i|(1<<k-1)][k]+=dp[i][j];
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)f[i]+=dp[i][j];
	for(int i=1;i<1<<n;i++)bitc[i]=bitc[i>>1]+(i&1),fu[bitc[i]][i]=f[i];
	for(int i=1;i<=n;i++)
	for(int j=2;j<=1<<n;j<<=1)
	for(int k=0;k<1<<n;k+=j)
	for(int l=k;l<k+(j>>1);l++)
	fu[i][l+(j>>1)]+=fu[i][l];
	for(int i=0;i<1<<(n-1);i++)as1[i]=solve(i);
	for(int j=2;j<=1<<n;j<<=1)
	for(int k=0;k<1<<n;k+=j)
	for(int l=k;l<k+(j>>1);l++)
	as1[l]-=as1[l+(j>>1)];
	scanf("%d",&q);
	while(q--)
	{
		scanf("%s",qu+1);
		int as=0;
		for(int i=1;i<n;i++)as=as*2+qu[i]-'0';
		printf("%lld\n",as1[as]);
	}
}
```

优化版本（实际上快不了太多）：

```cpp
#include<cstdio>
using namespace std;
#define N 18
#define M 132001
#define ll long long
int n,q,bitc[M];
ll as[M*2],as1[M],dp[M][N],f[M];
ll fu[N][M],pw[N][M],ct[N];
char st[N][N],qu[N];
ll solve(int tp)
{
	for(int i=1;i<=n;i++)ct[i]=0;
	int ls=1;
	for(int i=1;i<n;i++,tp>>=1)
	if(tp&1)ls++;
	else ct[ls]++,ls=1;
	ct[ls]++;
	int fg=0;
	for(int i=n;i>=1;i--,fg<<=1)
	for(int j=1;j<=ct[i];j++)fg=fg<<1|1;
	return as[fg];
}
ll vl[N][M];
void dfs(int ls,int d)
{
	if(ls==1)
	{
		for(int i=0;i<1<<n;i++)vl[n][i]=vl[d][i]*pw[n-d][i];
		ct[1]=n-d;d=n;
	}
	else ct[1]=0;
	if(d==n)
	{
		int fg=0;
		for(int i=n;i>=1;i--,fg<<=1)
		for(int j=1;j<=ct[i];j++)fg=fg<<1|1;
		ll si=0;
		for(int i=0;i<(1<<n);i++)
		si+=vl[d][i]*((n-bitc[i])&1?-1:1);
		as[fg]=si;
		return;
	}
	if(d+ls<=n)
	{
		for(int i=0;i<1<<n;i++)vl[d+ls][i]=vl[d][i]*fu[ls][i];
		ct[ls]++;dfs(ls,d+ls);ct[ls]--;
	}
	dfs(ls-1,d);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%s",st[i]+1);
	for(int i=1;i<=n;i++)dp[1<<i-1][i]=1;
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)if(dp[i][j])
	for(int k=1;k<=n;k++)if(st[j][k]=='1'&&!(i&(1<<k-1)))dp[i|(1<<k-1)][k]+=dp[i][j];
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)f[i]+=dp[i][j];
	for(int i=1;i<1<<n;i++)bitc[i]=bitc[i>>1]+(i&1),fu[bitc[i]][i]=f[i];
	for(int i=1;i<=n;i++)
	for(int j=2;j<=1<<n;j<<=1)
	for(int k=0;k<1<<n;k+=j)
	for(int l=k;l<k+(j>>1);l++)
	fu[i][l+(j>>1)]+=fu[i][l];
	for(int i=0;i<1<<n;i++)vl[0][i]=pw[0][i]=1;
	for(int i=1;i<=n;i++)for(int j=0;j<1<<n;j++)pw[i][j]=pw[i-1][j]*fu[1][j];
	dfs(n,0);
	for(int i=0;i<1<<(n-1);i++)as1[i]=solve(i);
	for(int j=2;j<=1<<n;j<<=1)
	for(int k=0;k<1<<n;k+=j)
	for(int l=k;l<k+(j>>1);l++)
	as1[l]-=as1[l+(j>>1)];
	scanf("%d",&q);
	while(q--)
	{
		scanf("%s",qu+1);
		int as=0;
		for(int i=1;i<n;i++)as=as*2+qu[i]-'0';
		printf("%lld\n",as1[as]);
	}
}
```



##### auoj415 s2mple的字符串

###### Problem

给一个长度为 $n$ 的小写字母字符串 $s$。$q$ 次询问，每次给一对 $l,r$，求 $s[l,r]$ 在 $s$ 的每一个本质不同子串中的出现次数之和。

$n,q\leq 4\times 10^5$

$4s,1024MB$

###### Sol

可以发现出现次数相当于有多少种方式将 $s[l,r]$ 扩充为当前串。因此问题可以转化为如下形式：

求多少个字符串组 $a,b$，使得 $a+s[l,r]+b$ 是原串的一个子串。

考虑建 `SAM`，则如果只考虑向前加入，相当于在 `ch` 边组成的 `DAG` 上，当前字符串对应的点出发的路径数量。

如果只考虑向后加入，则相当于在压缩前，`fail` 树上当前字符串对应的点子树内的点数。除去当前点所在的压缩点外，子树内的其它压缩点一定被完整包含，当前点上的情况只需要考虑询问的字符串和压缩点的长度即可得到，因此这个值也容易求出。

对于两侧都有加入的情况，考虑先加入右侧再加入左侧，则答案为 `fail` 树上子树内每个点向前加入的方案数和。由于一个压缩点满足这一段路径上的点的 `ch` 转移在压缩下等价，因此这一段内向前加入的路径数量相同，一个压缩点的贡献为它内部的字符串数量乘上它的方案数。因此可以在压缩后的 `fail` 树上 `dp` 求出子树和。

询问时先在 `fail` 树上定位这个串，一种定位方式为从对应的 `endpos` 开始，向上倍增找对应长度的点。然后使用 `dp` 和当前的长度就能得到答案。

复杂度 $O(n*|\small\sum|+(n+q)\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 805918
int n,q,a,b,ch[N][26],len[N],fail[N],st[N],f[N][21],las=1,ct=1,ls[N],c1;
long long dp[N],su[N];
char s[N];
bool cmp(int a,int b){return len[a]<len[b];}
void ins(int s)
{
	int st=++ct,s1=las;las=st;len[st]=len[s1]+1;ls[++c1]=st;
	while(s1&&!ch[s1][s])ch[s1][s]=st,s1=fail[s1];
	if(!s1)fail[st]=1;
	else
	{
		int nt=ch[s1][s];
		if(len[nt]==len[s1]+1)fail[st]=nt;
		else
		{
			int cl=++ct;len[cl]=len[s1]+1;
			for(int i=0;i<26;i++)ch[cl][i]=ch[nt][i];
			fail[cl]=fail[nt];fail[nt]=fail[st]=cl;
			while(s1&&ch[s1][s]==nt)ch[s1][s]=cl,s1=fail[s1];
		}
	}
}
int main()
{
	scanf("%d%d%s",&n,&q,s+1);
	for(int i=1;i<=n;i++)ins(s[i]-'a');
	for(int i=1;i<=ct;i++)st[i]=i,dp[i]=1;
	sort(st+1,st+ct+1,cmp);
	for(int i=ct;i>=1;i--)for(int j=0;j<26;j++)dp[st[i]]+=dp[ch[st[i]][j]];
	for(int i=ct;i>=1;i--)su[fail[st[i]]]+=su[st[i]]+dp[st[i]]*(len[st[i]]-len[fail[st[i]]]);
	for(int i=1;i<=ct;i++){f[st[i]][0]=fail[st[i]];for(int j=1;j<=19;j++)f[st[i]][j]=f[f[st[i]][j-1]][j-1];}
	while(q--)
	{
		scanf("%d%d",&a,&b);
		int st=ls[b];
		for(int i=20;i>=0;i--)if(len[f[st][i]]>=b-a+1)st=f[st][i];
		printf("%lld\n",su[st]+1ll*dp[st]*(len[st]-b+a));
	}
}
```



##### auoj416 s3mple的排列

###### Problem

对于一个排列 $p$，设它的长度为 $n$，定义第 $i$ 个位置的距离为大于 $p_i$ 的元素距 $i$ 的最小距离（认为 $p_0=p_{n+1}=+\infty$，这样不会出现没有定义的情况）

多组询问，给出大质数 $mod$，每次给出 $n,m$，求有多少个长度为 $n$ 的排列满足所有位置的距离的和为 $m$，答案对 $mod$ 取模。

$n\leq 200,T\leq 10$

$1s,256MB$

###### Sol

考虑枚举排列中最大的元素的位置，那么容易得到这个位置的距离。由于两侧的元素都小于中间这个元素，因此这之后左右两侧互不影响，变成了两个子问题。

因此设 $dp_{i,j}$ 表示长度为 $i$ 的排列，距离和为 $j$ 的方案数，有

$$
dp_{n,m}=\sum_{i=1}^n\sum_{j=0}^{m-min(i,n+1-i)}dp_{i-1,j}dp_{n-i,m-j-min(i,n+1-i)}C_{n-1}^{i-1}
$$

将排列建笛卡尔树，每个位置的距离为两个子树的 `size` 的 `min`，根据树剖或者启发式合并的分析总和是 $O(n\log n)$ 的。因此第二维的上限不超过 $O(n\log n)$。

注意到转移时第二维可以看成一个卷积，考虑维护第二维的生成函数，设 $f_n(x)$ 表示 $n$ 时的生成函数，有

$$
f_n(x)=\sum_{i=1}^nf_{i-1}(x)f_{n-i}(x)x^{min(i,n+1,i)}C_{n-1}^{i-1}
$$

由于模数是质数，考虑维护 $f_n(x)$ 在 $1,2,...,m+1$ 处的点值，使用点值转移。那么对于一个 $x$ 的取值 $i$，做 `dp` 的复杂度为 $O(n^2)$。总复杂度 $O(n^3\log n)$

然后询问时拉格朗日插值还原答案即可。注意到这里需要还原系数，可以对所有情况预处理出点值对某一项系数的贡献，这样询问可以 $O(n\log n)$。

复杂度 $O(n^3\log n+n^2\log^2 n+Tn\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 205
int p,c[N][N],dp[N][750],n,k,fu[N],f1[750][750],fu1[750],f2[750],as[N][750],q1[751][N],inv[2333];
int pw(int a,int b){int as=1;while(b){if(b&1)as=1ll*as*a%p;a=1ll*a*a%p;b>>=1;}return as;}
int main()
{
	n=200;scanf("%d",&p);
	for(int i=0;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%p;
	fu[0]=0;
	for(int i=1;i<=n;i++)
	for(int k=1;k<=i;k++)
	{
		int st=min(k,i-k+1);
		fu[i]=max(fu[i],fu[k-1]+fu[i-k]+st);
	}
	for(int i=1;i<=740;i++)dp[0][i]=1;
	for(int i=1;i<=740;i++){q1[i][0]=1;for(int j=1;j<=n;j++)q1[i][j]=1ll*q1[i][j-1]*i%p;}
	for(int i=1;i<=n;i++)
	for(int k=1;k<=i;k++)
	{
		int st=min(k,i-k+1);
		for(int j=1;j<=740;j++)
		dp[i][j]=(dp[i][j]+1ll*c[i-1][k-1]*dp[k-1][j]%p*dp[i-k][j]%p*q1[j][st])%p;
	}
	fu1[0]=1;
	for(int i=1;i<=740;i++)
	for(int j=i-1;j>=0;j--)
	fu1[j+1]=(fu1[j+1]+fu1[j])%p,fu1[j]=1ll*fu1[j]*(p-i)%p;
	for(int i=0;i<=2000;i++)inv[i]=pw(p-1000+i,p-2);
	for(int i=1;i<=740;i++)
	{
		int st=1;
		for(int j=1;j<=740;j++)if(i!=j)st=1ll*st*inv[1000+i-j]%p;
		for(int j=0;j<=740;j++)f2[j]=fu1[j];
		for(int j=0;j<=740;j++)f2[j]=1ll*f2[j]*inv[1000-i]%p,f2[j+1]=(f2[j+1]-f2[j]+p)%p;
		for(int j=0;j<=740;j++)f1[i][j]=1ll*st*f2[j]%p;
	}
	while(~scanf("%d%d",&n,&k))
	if(k>fu[n])printf("0\n");
	else
	{
		int as1=0;
		for(int j=1;j<=740;j++)as1=(as1+1ll*dp[n][j]*f1[j][k])%p;
		printf("%d\n",as1);
	}
}
```



#### NOI2020模拟六校联测7

##### auoj417 极乐迪斯科

###### Problem

有一棵 $n$ 个点有根树，没有边权。有 $m$ 个摄像头，第 $i$ 个摄像头可以监视到以 $x_i$ 为根的子树内距离 $x_i$ 不超过 $d_i$ 的点，拆掉它费用为 $c_i$。

你可以拆掉一些摄像头，这之后对于每一个没有被监视到的点，你可以获得对应的 $v_i$ 的收益，求收益减去费用的最大值。

$n,m\leq 5\times 10^5$

$1s,256MB$

###### Sol

考虑先选全部 $v_i$，然后看成最少要放弃多少收益。

则如果 $a$ 能监控到 $b$，则要么用对应费用把 $a$ 拆掉，要么放弃 $b$ 的收益。可以发现这相当于一个割。

考虑从源点向每一个摄像头连流量等于它的费用的边，每个摄像头向它能监视到的点连流量 $+\infty$ 的边，每个点向汇点连流量等于它的收益的边。则放弃收益的方式为图中的割，因此相当于求出这个图的最小割，即最大流。

因此问题相当于每个摄像头有 $c_i$ 的流，可以匹配子树内距离不超过 $d_i$ 的点。每个点可以接受 $v_i$ 的流，求最大匹配。

考虑先做一个深度最深的摄像头的匹配。如果它还有可以匹配的流量，则显然匹配比不匹配更优。因为剩下的能影响到这个子树的摄像头都是它的祖先，因此对于子树内两个点 $a,b$，如果 $a$ 比 $b$ 深，则一个祖先如果能匹配到 $b$ 则一定能匹配到 $a$。因此当前点匹配深度更深的点更优。

因此考虑让这个点匹配能匹配位置的中深度最大的那些，可以发现这是最优的。因此可以从下往上考虑每个点上的摄像头，每次匹配能匹配的位置中深度最大的。

对于每个点维护子树内每个深度剩余的流量，需要支持合并子树间的情况，查一个区间内有流量的最大位置，线段树合并即可维护。

复杂度 $O((n+m)\log n)$，注意卡常。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 500500
#define ll long long
#pragma GCC optimize("-Ofast")
int n,m,lb[N],rb[N],tid[N],dep[N],vl[N],fa[N],head[N],cnt,ct,ct1,a,b,c;
ll as;
struct sth{int d,x;friend bool operator <(sth a,sth b){return a.d<b.d;}};
vector<sth> fu[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa){dep[u]=dep[fa]+1;lb[u]=++ct1;tid[ct]=u;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u);rb[u]=ct1;}
#define M 10000001
int ch[M][2],rt[N],f1;
ll su[M],f2;
int merge(int x,int y)
{
	if(!x||!y)return x+y;
	su[x]+=su[y];
	ch[x][0]=merge(ch[x][0],ch[y][0]);
	ch[x][1]=merge(ch[x][1],ch[y][1]);
	return x;
}
int query(int x,int l,int r,int l1,int r1)
{
	if(su[x]==0)return 0;
	if(l==r){f1=l;f2=su[x];return 1;}
	int mid=(l+r)>>1;
	if(mid>=r1)return query(ch[x][0],l,mid,l1,r1);
	else
	{
		int tp=query(ch[x][1],mid+1,r,l1,r1);
		if(!tp)return query(ch[x][0],l,mid,l1,r1);
		else return 1;
	}
}
void modify(int x,int l,int r,int s,int v)
{
	su[x]-=v;
	if(l==r)return;
	int mid=(l+r)>>1;
	if(mid>=s)modify(ch[x][0],l,mid,s,v);
	else modify(ch[x][1],mid+1,r,s,v);
}
void insert(int x,int l,int r,int s,int v)
{
	su[x]+=v;
	if(l==r)return;
	int mid=(l+r)>>1;
	if(mid>=s)
	{
		if(!ch[x][0])ch[x][0]=++ct;
		insert(ch[x][0],l,mid,s,v);
	}
	else
	{
		if(!ch[x][1])ch[x][1]=++ct;
		insert(ch[x][1],mid+1,r,s,v);
	}
}
void dfs2(int u,int fa)
{
	rt[u]=++ct;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs2(ed[i].t,u),rt[u]=merge(rt[u],rt[ed[i].t]);
	insert(rt[u],1,n,dep[u],vl[u]);
	sort(fu[u].begin(),fu[u].end());
	for(int i=0;i<fu[u].size();i++)
	{
		sth tp=fu[u][i];
		int lb=dep[u],rb=dep[u]+tp.d;
		if(rb>n)rb=n;
		while(tp.x)
		{
			int v1=query(rt[u],1,n,1,rb);
			if(!v1)break;
			int v2=min(f2,1ll*tp.x);
			as-=v2;tp.x-=v2;modify(rt[u],1,n,f1,v2);
		}
	}
}
inline char gc(){static char buf[1000000],*p1,*p2;return p1==p2&&(p2=(p1=buf)+fread(buf,1,1000000,stdin),p1==p2)?EOF:*p1++;}
template<class T> inline void rd(T &x){x=0;char c=gc();while(c<'0'||c>'9')c=gc();while('0'<=c&&c<='9')x=x*10+c-'0',c=gc();}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=2;i<=n;i++)rd(fa[i]),adde(i,fa[i]);
	for(int i=1;i<=n;i++)rd(vl[i]),as+=vl[i];
	for(int i=1;i<=m;i++)rd(a),rd(b),rd(c),fu[a].push_back((sth){b,c});
	dfs1(1,0);dfs2(1,0);printf("%lld\n",as);
}
```



##### auoj418 反讽

###### Problem

给定两个长度为 $n,m$，不一定合法的括号序列 $a,b$。你需要使用归并的方式将它们合并成一个序列，再加入尽量少的括号，使得序列变为合法序列。求第二步中最少需要加多少括号。

多组数据

$T\leq 10,n,m\leq 10^6$

$2s,256MB$

###### Sol

考虑将 `(` 看成 $+1$，`)` 看成 $-1$，则第二步的问题相当于最少需要插入多少个数使得前缀和非负且总和为 $0$。

可以发现 `(` 加在开头，`)` 加在结尾最优。可以发现最小需要加的数只和当前序列的总和的最小前缀和有关。因此问题变为归并两个序列，最大化最小前缀和。

~~于是这是今年论文的经典例子~~

对于一个序列，如果将它看成一个整体，则在不改变最小前缀和的结果的情况下，一定可以将它合并为两个数构成的序列 $-a,+b$。这里 $-a$ 为原序列最小前缀和，$-a+b$ 为总和，$a,b\geq 0$。

考虑两段之间怎么排列。一种方式是交换两个元素，比较前后哪一种更优。因为这个问题满足如果序列 $s$ 和 $t$ 总和相同，且 $s$ 比 $t$ 优，则向两侧加入任意字符后 $s$ 不比 $t$ 差，因此多个段时也可以考虑这样比较。

那么对于 $-a,+b$ 和 $-c,+d$，一种比较方式是比较 $\min(-a,b-a-c)$ 和 $\min(-c,d-a-c)$。

可以发现左侧结果和 $b-a$ 的符号有关，如果 $b-a\geq 0$，则这个值大于等于 $\min(-a,-c)$。否则这个值小于等于 $\min(-a,-c)$。

由此考虑将所有元素按照 $sgn(b-a)$ 分类。

1. 对于 $b>a,d>c$ 的情况，分析可以发现比较方式为 $a\leq c$。
2. 对于 $b<a,d<c$ 的情况，可以发现比较方式为 $b\geq d$。
3. 对于 $b=a,d=c$ 的情况，可以任意排列。

每一类内部都是一个好的序关系，但不同类之间如果直接按照上面比较，则存在一些问题（例如 $(1,1)=(1,2),(1,1)=(2,2)$ 但 $(1,2)<(2,2)$，不能交换）。

可以考虑不同类之间按照 $b>a,b=a,b<a$ 排序，这样得到一个序关系，且满足如果 $s<t$，则在任意一个方案中，如果 $s$ 在 $t$ 的后一个，则交换后不会变差。

这样对于将多个没有限制的段排列的问题，按照序关系排序即可得到答案。



考虑有限制的问题，更加广泛的情况是给一棵有根树作为限制。有两种做法：

1. 每次找到除去根之外的最小序列，将它与父亲合并。
2. 从下往上构造每个子树合并后的若干单调的段，先将子树合并，然后将根放在开头，尝试和后面的段合并直到单调。

可以证明在一些条件下，两个做法都是正确的，而这个问题满足这些条件。

~~但是具体证明写了5页，所以说这里就不写了~~

但这里是两条链，因此只需要如下结论：

如果 $s$ 只有一个儿子 $t$，且 $t$ 的元素比 $s$ 的元素更优，则存在一组最优解使得 $t$ 在 $s$ 的下一个。

证明：考虑其它的情况，设中间部分为 $x$，则对于 $s,x,t$ 的情况，因为 $t\leq s$，因此要么 $t\leq x$，要么 $x\leq s$，这里比较是比较对应元素。对于前者，可以将 $t,x$ 交换，显然交换后满足顺序限制，对于后者可以交换 $s,x$。

因此考虑从后往前看这个串，维护当前合并后元素组成的单调栈。加入一个元素时，考虑当前元素 $u$ 和栈顶 $x$，如果 $x\leq u$ 则合并，直到不能合并为止，再将 $u$ 放入栈顶。则存在一种最优方式使得最后单调栈中每一段在方案中连续。

最后需要将两个单调栈内的元素归并，但此时一个单调栈内部是单调的，所以直接按照权值顺序归并显然是最优的，因此直接归并即可得到答案。

复杂度 $O(T(n+m))$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1000500
struct sth{
	int a,b;
	friend bool operator <(sth a,sth b){int t1=a.a<a.b,t2=b.a<b.b;if(t1&&!t2)return 1;if(t2&&!t1)return 0;if(t1&&t2)return a.a<b.a;else return a.b>b.b;}
	friend sth operator +(sth a,sth b){return (sth){max(a.a,a.a-a.b+b.a),-a.a+a.b-b.a+b.b+max(a.a,a.a-a.b+b.a)};}
}v1[N],v2[N];
int T,n,m,l1,l2,su,as;
char s1[N],s2[N];
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%s%s",&n,&m,s1+1,s2+1);
		l1=l2=0;su=0;as=0;
		for(int i=1;i<=n;i++)
		{
			sth tp=(sth){0,0};
			if(s1[i]=='(')tp.b=1,su++;else tp.a=1,su--;
			while(l1&&tp<v1[l1])tp=v1[l1]+tp,l1--;
			v1[++l1]=tp;
		}
		for(int i=1;i<=m;i++)
		{
			sth tp=(sth){0,0};
			if(s2[i]=='(')tp.b=1,su++;else tp.a=1,su--;
			while(l2&&tp<v2[l2])tp=v2[l2]+tp,l2--;
			v2[++l2]=tp;
		}
		sth fu=(sth){0,0};
		if(su<0)fu.b=-su,as-=su;
		int r1=1,r2=1;
		for(int i=1;i<=l1+l2;i++)
		if(r1>l1)fu=fu+v2[r2++];
		else if(r2>l2)fu=fu+v1[r1++];
		else if(v1[r1]<v2[r2])fu=fu+v1[r1++];
		else fu=fu+v2[r2++];
		as+=fu.a+fu.b;
		printf("%d\n",as);
	}
}
```



##### auoj419 敏感词

###### Problem

给一个长度为 $n$ 的字符串，你可以选择一个串 $t$，将原串中每一个等于 $t$ 的子串的所有位置做上标记，要求最后有 $k$ 个位置被做过标记。

求所有满足条件的选择的串中长度最小的串或者输出无解，若有多个解选字典序最小的。

多组数据

$T\leq 10,n\leq 2\times 10^4$

$2s,256MB$

###### Sol

对于一个串，最后标记的位置个数只和这个串所有出现的位置以及这个串的长度相关。

对原串建 `SAM`，考虑 `SAM` 上的每一个点，这个点上的字符串所有出现的结尾位置即为该点的 `endpos`，设其为 $s_1<s_2<...<s_m$。

对于这个点上一个长度为 $l$ 的串，它最后标记的位置个数为 $l+\sum_{i<m}min(s_{i+1}-s_i,l)$，这显然是单调的，可以二分找到需要的 $l$。

考虑维护这部分，只需要知道所有相邻两个 `endpos` 位置的差。

考虑启发式合并维护 `endpos`，将一个位置插入另外一个 `endpos` 集合时，会删去一个相邻的差，再加入两个相邻的差。这样的次数只和启发式合并插入的次数有关。

使用线段树维护每个点相邻 `endpos` 位置的差，对于一个点，启发式合并儿子的 `endpos`，对于差的修改可以直接在较大的儿子的线段树上改，这样就可以从下往上维护每个点的线段树。

然后考虑二分 $l$，只需要知道差小于 $l$ 的部分所有差的和以及大于等于 $l$ 的差个数，可以直接在线段树上查。这里也可以线段树上二分，但因为第一部分复杂度是 $\log^2$ 的这样不能降低复杂度。

这样可以求出 $O(n)$ 个可能的字符串，然后使用 `hash` 或者其它的比较方式即可。

复杂度 $O(Tn\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 40050
#define M 8006000
#define ll long long
int n,k,T;
char s[N];
//SAM
struct SAM{
	int fail[N],len[N],ch[N][26],is[N],las,ct,fu;
	void init(){for(int i=0;i<=ct;i++){fail[i]=len[i]=is[i]=0;for(int j=0;j<26;j++)ch[i][j]=0;}las=ct=1;fu=0;}
	void insert(int x)
	{
		int st=++ct,s1=las;len[st]=len[s1]+1;is[st]=++fu;las=st;
		while(!ch[s1][x]&&s1)ch[s1][x]=st,s1=fail[s1];
		if(!s1){fail[st]=1;return;}
		if(len[ch[s1][x]]==len[s1]+1)fail[st]=ch[s1][x];
		else
		{
			int tp=ch[s1][x],cl=++ct;
			len[cl]=len[s1]+1;for(int i=0;i<26;i++)ch[cl][i]=ch[tp][i];
			fail[cl]=fail[tp];fail[tp]=fail[st]=cl;
			while(s1&&ch[s1][x]==tp)ch[s1][x]=cl,s1=fail[s1];
		}
	}
}sam;
//segt
int rt[N],ch[M][2],sz[M],ct,su[M];
struct sth{int a,b;};
sth doit(sth a,sth b){return (sth){a.a+b.a,a.b+b.b};}
void modify(int x,int l,int r,int v,int v1,int v2)
{
	sz[x]+=v1;su[x]+=v2;
	if(l==r)return;
	int mid=(l+r)>>1;
	if(mid>=v)
	{
		if(!ch[x][0])ch[x][0]=++ct;
		modify(ch[x][0],l,mid,v,v1,v2);
	}
	else
	{
		if(!ch[x][1])ch[x][1]=++ct;
		modify(ch[x][1],mid+1,r,v,v1,v2);
	}
}
sth query(int x,int l,int r,int l1,int r1)
{
	if(!x)return (sth){0,0};
	if(l==l1&&r==r1)return (sth){sz[x],su[x]};
	int mid=(l+r)>>1;
	if(mid>=r1)return query(ch[x][0],l,mid,l1,r1);
	else if(mid<l1)return query(ch[x][1],mid+1,r,l1,r1);
	else return doit(query(ch[x][0],l,mid,l1,mid),query(ch[x][1],mid+1,r,mid+1,r1));
}
//hash
int ch1=131,ch2=101,md1=998244353,md2=1e9+7,ha1[N],ha2[N],pw1[N],pw2[N],as1,as2;
void pre()
{
	for(int i=1;i<=n;i++)ha1[i]=(1ll*ha1[i-1]*ch1+s[i]-'a'+1)%md1,ha2[i]=(1ll*ha2[i-1]*ch2+s[i]-'a'+1)%md2;
	pw1[0]=pw2[0]=1;
	for(int i=1;i<=n;i++)pw1[i]=1ll*pw1[i-1]*ch1%md1,pw2[i]=1ll*pw2[i-1]*ch2%md2;
}
int gethash1(int l,int r){return (ha1[r]-1ll*ha1[l-1]*pw1[r-l+1]%md1+md1)%md1;}
int gethash2(int l,int r){return (ha2[r]-1ll*ha2[l-1]*pw2[r-l+1]%md2+md2)%md2;}
int lcp(int a,int b)
{
	int lb=0,rb=min(n-a+1,n-b+1),as=0;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(gethash1(a,a+mid-1)==gethash1(b,b+mid-1)&&gethash2(a,a+mid-1)==gethash2(b,b+mid-1))as=mid,lb=mid+1;
		else rb=mid-1;
	}
	return as;
}
void check(int l,int r)
{
	if(!as1){as1=l,as2=r;return;}
	if(as2-as1<r-l)return;
	if(as2-as1>r-l){as1=l;as2=r;return;}
	int l1=lcp(l,as1);
	if(l+l1-1>=r&&as1+l1-1>=as2)
	{if(r-l<as2-as1)as2=r,as1=l;}
	else if(l+l1-1>=r)as2=r,as1=l;
	else if(l+l1-1<r&&as1+l1-1<as2)
	if(s[l+l1]<s[as1+l1])as2=r,as1=l;
}
//endpos
set<int> fu[N];
struct edge{int t,next;}ed[N*2];
int head[N],cnt,id[N],s11[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void merge(int x,int y)
{
	if(fu[id[x]].size()<fu[id[y]].size())id[x]^=id[y]^=id[x]^=id[y],rt[x]=rt[y];
	for(set<int>::iterator it=fu[id[y]].begin();it!=fu[id[y]].end();it++)
	{
		int tp=*it;
		set<int>::iterator it2=fu[id[x]].lower_bound(tp);
		if(it2==fu[id[x]].begin())
		{
			int st=(*it2)-tp;
			modify(rt[x],1,n,st,1,st);
		}
		else if(it2==fu[id[x]].end())
		{
			it2--;int st=tp-(*it2);
			modify(rt[x],1,n,st,1,st);
		}
		else
		{
			set<int>::iterator it3=it2;it2--;
			int st=tp-(*it2),st1=(*it3)-tp,st2=(*it3)-(*it2);
			modify(rt[x],1,n,st,1,st);
			modify(rt[x],1,n,st1,1,st1);
			modify(rt[x],1,n,st2,-1,-st2);
		}
		fu[id[x]].insert(tp);
	}
}
void dfs(int u)
{
	rt[u]=++ct;if(sam.is[u])fu[id[u]].insert(sam.is[u]),s11[u]=1;
	for(int i=head[u];i;i=ed[i].next)dfs(ed[i].t),merge(u,ed[i].t),s11[u]+=s11[ed[i].t];
	int lb=sam.len[sam.fail[u]]+1,rb=sam.len[u],as=lb;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		sth tp=query(rt[u],1,n,1,mid);
		int su1=(s11[u]-tp.a)*mid+tp.b;
		if(su1<=k)as=mid,lb=mid+1;
		else rb=mid-1;
	}
	sth tp=query(rt[u],1,n,1,as);
	int su1=(s11[u]-tp.a)*as+tp.b;
	if(su1==k&&as<=rb)
	{
		int rb=*fu[id[u]].begin();
		check(rb-as+1,rb);
	}
}
void solve()
{
	scanf("%s%d",s+1,&k);n=strlen(s+1);
	sam.init();
	for(int i=1;i<=n*2;i++)fu[i].clear(),head[i]=0,id[i]=i,s11[i]=0,rt[i]=0;
	for(int i=0;i<=ct;i++)ch[i][0]=ch[i][1]=sz[i]=su[i]=0;
	cnt=0;ct=0;as1=as2=0;pre();
	for(int i=1;i<=n;i++)sam.insert(s[i]-'a');
	int c1=sam.ct;
	for(int i=2;i<=c1;i++)adde(sam.fail[i],i);
	dfs(1);
	if(!as1)printf("NOTFOUND!\n");
	else
	{
		for(int i=as1;i<=as2;i++)printf("%c",s[i]);
		printf("\n");
	}
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```



#### NOI2020模拟六校联测10

##### auoj420 Arcahv

###### Problem

给定 $n$ 个字符串，有 $q$ 次操作

1. 修改一个串的一个位置
2. 给定一个串，询问它是不是所有串的公共子序列

$n\leq 10,\sum len,q\leq 10^5,|\small\sum|=10^9$

$2s,512MB$

###### Sol

可以发现，只需要判断询问的串是不是每个串的子序列即可

考虑判断是不是一个串的子序列，可以贪心，每次只需要找某种字符在某个位置后第一次出现的位置。

对于每个串每种字符用一个 `set` 维护它的出现位置，因为字符集很大，可以记录当前有的字符种类，然后 `map` 离散化。

复杂度 $O(n(len+q)\log len)$

###### Code

```cpp
#include<cstdio>
#include<map>
#include<set>
using namespace std;
#define N 100400
set<int> fu[N*4];
map<int,int> id[N];
int n,q,v[11][N],a,b,c,d,v2[N],le[N],ct;
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&le[i]);
		for(int j=1;j<=le[i];j++)
		{
			scanf("%d",&v[i][j]);
			if(!id[i][v[i][j]])id[i][v[i][j]]=++ct;
			fu[id[i][v[i][j]]].insert(j);
		}
	}
	while(q--)
	{
		scanf("%d",&a);
		if(a==1)
		{
			scanf("%d%d%d",&b,&c,&d);
			fu[id[b][v[b][c]]].erase(c);
			v[b][c]=d;
			if(!id[b][v[b][c]])id[b][v[b][c]]=++ct;
			fu[id[b][v[b][c]]].insert(c);
		}
		else
		{
			scanf("%d",&b);
			for(int i=1;i<=b;i++)scanf("%d",&v2[i]);
			int fg=1;
			for(int i=1;i<=n;i++)
			{
				int nw=0;
				for(int j=1;j<=b;j++)
				{
					set<int>::iterator it=fu[id[i][v2[j]]].lower_bound(nw+1);
					if(it==fu[id[i][v2[j]]].end())fg=0;else nw=*it;
				}
			}
			printf("%s\n",fg?"Hikari":"Tairitsu");
		}
	}
}
```



##### auoj421 Tempestissimo

###### Problem

给出 $n,m$，从 $1,...,n$ 中随机拿出数，拿出后不放回，$1,2,...,m$ 被全部拿出后停止，求期望拿出多少个数，答案模 $998244353$

多组数据

$T\leq 10^7,n,m\leq 9\times 10^8$

$1s,256MB$

###### Sol

可以看成随机一个排列，然后从开头开始拿数，满足条件停止。

考虑每个数的贡献，前 $m$ 个数的贡献显然为 $1$。对于后面的数，如果它们在排列中比 $m$ 个中的最后一个还要靠后那么没有贡献，否则有贡献。因此贡献为 $\frac m{m+1}$。

根据线性性，答案为 $m+\frac{m(n-m)}{m+1}$

只需要线性求出所有 $m+1$ 的逆元即可。一种方式是离线，求出前缀乘积，然后求出最后一个前缀乘积的逆元，再向前乘得到所有逆元的前缀乘积，再用两个前缀乘积即可得到所有逆元。

复杂度 $O(T+\log mod)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
#define N 10000010
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int T,as,range,q[N][2],su[N],inv[N];
unsigned int seed;
inline unsigned int randint()
{
    seed^=seed<<13;
    seed^=seed>>7;
    seed^=seed<<5;
    return seed%range+1;
}
int main()
{
	scanf("%d%u%d",&T,&seed,&range);
	for(int i=1;i<=T;i++)
	{
		int n=randint(),m=randint();
		if(n<m)n^=m^=n^=m;
		q[i][0]=n;q[i][1]=m;
	}
	su[0]=1;for(int i=1;i<=T;i++)su[i]=1ll*su[i-1]*(q[i][1]+1)%mod;
	inv[T]=pw(su[T],mod-2);for(int i=T-1;i>=0;i--)inv[i]=1ll*inv[i+1]*(q[i+1][1]+1)%mod;
	for(int i=1;i<=T;i++)
	{
		int as1=(q[i][1]+1ll*(q[i][0]-q[i][1])*q[i][1]%mod*su[i-1]%mod*inv[i])%mod;
		as^=as1;
	}
	printf("%d\n",as);
}
```



##### auoj422 生成无向图

###### Problem

给一个数组 $a$，$q$ 次询问，每次给定 $l,r$，初始时有一个点，然后依次加入 $r-l+1$ 个点，每个点的父亲在之前的点中随机，这条边的边权为 $a_{i+l-1}$，求出最后得到的树上所有点对距离和的期望，模 $998244353$。

$n,q\leq 5\times 10^4$

$2s,512MB$

###### Sol

假设当前有 $n$ 个点，当前答案为 $ans$，假设当前加入一个点，边权为 $v$。

考虑与加入的这个点相关的路径，这 $n$ 条路径都经过新加的边，考虑新加的边之外的贡献，相当于这个点连向的点到其它所有点距离和。因此相当于随机选一个点，求这个点到原来的其它所有点距离和的期望，这显然是 $ans*\frac 2n$。

那么有 $ans^{'}=ans*\frac{n+2}n+nv$，$n$ 为当前点数。

那么考虑每个 $nv$ 的贡献可以得到最后的答案为 $(r-l+2)(r-l+3)\sum_{i=l}^rv_i\frac{i-l+1}{(i-l+2)(i-l+3)}$

如果没有 $r$ 的限制右边就是差卷积。考虑分块，对于 $r=s,2s,...,\lfloor\frac ns\rfloor s$ 的情况分别用 `NTT` 求出这时对于所有 $l$ 右边的值，然后对于每个询问块之间的 $O(s)$ 项暴力即可。

取 $s=O(\sqrt{n\log n})$，复杂度 $O(n\sqrt{n\log n})$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
#define N 132000
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int n,q,l,r,k,a[N],inv[N],f[N],v1[N],v2[N],rev[N],ntt[N],as1[105][N],g[2][N*2];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		for(int j=0;j<s;j+=i)
		for(int k=j,vl=0;k<j+(i>>1);k++,vl++)
		{
			int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][i+vl]%mod;
			ntt[k]=v1+v2-(v1+v2>=mod)*mod;
			ntt[k+(i>>1)]=v1-v2+(v2>v1)*mod;
		}
	}
	int inv=pw(s,t==0?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void pre()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<17;j<<=1)
	{
		int tp=pw(3,(mod-1)/j),v2=1;
		if(i==0)tp=pw(tp,mod-2);
		for(int l=0;l<j>>1;l++)g[i][j+l]=v2,v2=1ll*v2*tp%mod;
	}
}
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&a[i]);
	for(int i=1;i<=n+3;i++)inv[i]=pw(i,mod-2);
	for(int i=0;i<=n;i++)f[i]=1ll*(i+1)*inv[i+2]%mod*inv[i+3]%mod;
	pre();
	for(int i=1;i*500<=n;i++)
	{
		int st=i*500;
		int l=1;while(l<=st*2)l<<=1;
		for(int j=0;j<l;j++)v1[j]=v2[j]=0;
		for(int j=0;j<=st;j++)v1[j]=a[j],v2[st-j]=f[j];
		dft(l,v1,1);dft(l,v2,1);for(int j=0;j<l;j++)v1[j]=1ll*v1[j]*v2[j]%mod;dft(l,v1,0);
		for(int j=0;j<=st;j++)as1[i][j]=v1[j+st];
	}
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d",&l,&r);
		if(r-l<=1000)
		{
			int as=0;
			for(int i=l;i<=r;i++)as=(as+1ll*a[i]*f[i-l])%mod;
			printf("%d\n",1ll*as*(r-l+2)%mod*(r-l+3)%mod);
			continue;
		}
		k=r/500*500;
		int as=as1[k/500][l];
		for(int i=k+1;i<=r;i++)as=(as+1ll*a[i]*f[i-l])%mod;
		printf("%d\n",1ll*as*(r-l+2)%mod*(r-l+3)%mod);
	}
```



#### SCOI2020模拟19

##### auoj424 qiqi20021026 的 T1

###### Problem

给出 $n$ 个字符串 $s_1,...,s_n$。有 $q$ 次询问，每次给两个长度相等的区间 $[l_0,r_0],[l_1,r_1]$，你需要将 $s_{l_0},...,s_{r_0}$ 与 $s_{l_1},...,s_{r_1}$ 进行匹配，每一对匹配的权值为两个串的最长公共后缀，求匹配的最大权值。

$n,\sum len\leq 10^4,q\leq 5\times 10^5$

$1s,512MB$

###### Sol

考虑将所有串反过来建 `trie` 树，匹配的权值相当于 `lca` 的深度（不算根）。

那么整个匹配的权值可以看成对于每个不是根的点，记录 `lca` 在它子树中的匹配对数，每个不是根的点的匹配对数和。

对于一个点，设它子树内有 $a_1$ 个第一个区间的串，$a_2$ 个第二个区间的串，那么这个点的可能的最大匹配对数为 $\min(a_1,a_2)$。考虑从下往上贪心匹配，则显然可以对于每个点都达到这个上界。

因此只需要维护每个点的 $\min(a_1,a_2)$ 即可。因此在加入或者删除一个字符串时，可以在 `trie` 上直接改，复杂度为串长。

将 $l_0,r_0,l_1,r_1$ 移动一位的代价是对应串的长度，可以按照串长前缀和作为位置进行莫队，四维莫队即可。

复杂度 $O(\sum len*q^{\frac34})$，注意卡常

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 10500
int n,q,ls[N],su[N],ch[N][26],fa[N],ct,v1[N],as,fu=150,bel[N],as1[N*50];
int rd(){int as=0;char c=getchar();while(c<'0'||c>'9')c=getchar();while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();return as;}
char s[N];
struct que{
	int a,b,c,d,id;
	friend bool operator <(que a,que b){return bel[su[a.a]]==bel[su[b.a]]?(bel[su[a.b]]==bel[su[b.b]]?(bel[su[a.c]]==bel[su[b.c]]?(bel[su[b.c]]&1?a.d<b.d:a.d>b.d):(bel[su[a.b]]&1?a.c<b.c:a.c>b.c)):(bel[su[a.a]]&1?a.b<b.b:a.b>b.b)):a.a<b.a;}
}qu[N*50];
void addl(int x)
{
	for(int i=ls[x];fa[i];i=fa[i])
	{
		v1[i]++;
		if(v1[i]<=0)as++;
	}
}
void dell(int x)
{
	for(int i=ls[x];fa[i];i=fa[i])
	{
		v1[i]--;
		if(v1[i]<0)as--;
	}
}
void addr(int x)
{
	for(int i=ls[x];fa[i];i=fa[i])
	{
		v1[i]--;
		if(v1[i]>=0)as++;
	}
}
void delr(int x)
{
	for(int i=ls[x];fa[i];i=fa[i])
	{
		v1[i]++;
		if(v1[i]>0)as--;
	}
}
int main()
{
	scanf("%d%d",&n,&q);ct=1;
	for(int i=1;i<=n;i++)
	{
		su[i]=su[i-1];ls[i]=1;
		scanf("%s",s+1);
		int le=strlen(s+1);
		for(int j=1;j<le-j+1;j++)swap(s[j],s[le-j+1]);
		for(int j=1;s[j];j++)
		{
			su[i]++;
			int tp=s[j]-'a';
			if(!ch[ls[i]][tp])ch[ls[i]][tp]=++ct,fa[ct]=ls[i];
			ls[i]=ch[ls[i]][tp];
		}
	}
	for(int i=1;i<=su[n];i++)bel[i]=(i-1)/fu+1;
	for(int i=1;i<=q;i++)qu[i].a=rd(),qu[i].b=rd(),qu[i].c=rd(),qu[i].d=rd(),qu[i].id=i;
	sort(qu+1,qu+q+1);
	int la=1,ra=0,lb=1,rb=0;
	for(int i=1;i<=q;i++)
	{
		while(ra<qu[i].b)addl(++ra);
		while(la>qu[i].a)addl(--la);
		while(rb<qu[i].d)addr(++rb);
		while(lb>qu[i].c)addr(--lb);
		while(ra>qu[i].b)dell(ra--);
		while(la<qu[i].a)dell(la++);
		while(rb>qu[i].d)delr(rb--);
		while(lb<qu[i].c)delr(lb++);
		as1[qu[i].id]=as;
	}
	for(int i=1;i<=q;i++)printf("%d\n",as1[i]);
}
```



##### auoj425 xuanyiming 的 T2

###### Problem

对于一个图，定义它的权值为图中是树的连通块个数的 $k$ 次方。

求所有 $n$ 个点有标号无向图的权值和，模 $998244353$

多组数据

$n\leq 5\times 10^4,k\leq 20,T\leq 10^5$

$2s,512MB$

###### Sol

可以将 $n^k$ 拆成下降幂的形式，拆成下降幂后只需要求出在图中选出 $k$ 个树的方案数。

设 $f_{n,m}$ 表示 $n$ 个点由 $m$ 棵树组成的图的方案数，枚举 $1$ 所在的树的点数，那么有 $f_{n,m}=\sum_{i=1}^nf_{n-i,m-1}C_{n-1}^{i-1}i^{i-2}$，$m$ 次 `NTT` 即可。

设 $g_{n,m}$ 表示 $n$ 个点的图选出 $m$ 棵树的方案数，那么有 $g_{n,m}=\sum_{i=0}^nC_n^if_{i,m}$，也可以 `NTT`。

最后对于每个询问用 $g_{n,m}$ 和 $n^k$ 的斯特林下降幂展开即可得到答案。

复杂度 $O(nk\log n+Tk+k^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 132001
#define K 21
#define mod 998244353
int f[N][K],fr[N],ifr[N],t,n,k,s[K][K],rev[N],ntt[N],g[2][N*2],a[N],b[N],las;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void pre()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<17;j<<=1)
	{
		int st=pw(3,(mod-1)/j),vl=1;
		if(!i)st=pw(st,mod-2);
		for(int k=0;k<j>>1;k++)
		g[i][j+k]=vl,vl=1ll*vl*st%mod;
	}
}
void dft(int s,int *a,int t)
{
	if(las!=s)for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1));las=s;
	for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=i;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][st]%mod;
		ntt[k]=(v1+v2)%mod;
		ntt[k+(i>>1)]=(v1-v2+mod)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=5e4;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	s[0][0]=1;
	for(int i=1;i<=20;i++)
	for(int j=1;j<=20;j++)
	s[i][j]=(1ll*j*s[i-1][j]+s[i-1][j-1])%mod;
	f[0][0]=1;
	for(int i=1;i<=5e4;i++)f[i][1]=pw(i,i==1?1:i-2);
	pre();
	for(int j=0;j<131072;j++)b[j]=0;
	for(int j=1;j<=5e4;j++)b[j]=1ll*f[j][1]*ifr[j-1]%mod;
	dft(131072,b,1);
	for(int i=2;i<=20;i++)
	{
		int l=131072;
		for(int j=0;j<l;j++)a[j]=0;
		for(int j=1;j<=5e4;j++)a[j]=1ll*f[j][i-1]*ifr[j]%mod;
		dft(l,a,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
		for(int j=1;j<=5e4;j++)f[j][i]=1ll*fr[j-1]*a[j]%mod;
	}
	for(int j=0;j<131072;j++)b[j]=0;
	for(int j=0;j<=5e4;j++)b[j]=1ll*pw(2,1ll*j*(j-1)/2%(mod-1))*ifr[j]%mod;
	dft(131072,b,1);
	for(int i=1;i<=20;i++)
	{
		int l=131072;
		for(int j=0;j<l;j++)a[j]=0;
		for(int j=0;j<=5e4;j++)a[j]=1ll*f[j][i]*ifr[j]%mod;
		dft(l,a,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
		for(int j=1;j<=5e4;j++)f[j][i]=1ll*fr[j]*a[j]%mod;
	}
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d%d",&n,&k);
		int as=0;
		for(int j=1;j<=k;j++)
		as=(as+1ll*s[k][j]*f[n][j]%mod*fr[j])%mod;
		printf("%d\n",as);
	}
}
```



##### auoj426 daklqw 的 T3

###### Problem

有一棵 $n$ 个点的树，经过每条边的时间为 $1$，有一些边为奖励边，每经过一次奖励边分数加 $c$。

有 $m$ 个任务，每个任务在点 $x_i$，持续时间为 $[a_i,a_i+b_i]$，完成任务可以得到 $d_i$ 的分数。如果要完成这个任务，需要在这段时间中一直停留在这个点上且做这个任务，且同一时间只能做一个任务（可以同时完成多个 $b_i=0$ 且 $a_i$ 相同的任务）。

求从每个点在时刻 $0$ 时出发，时刻 $T$ 时的最大分数。

$n,m\leq 10^5,T\leq 10^8$

$4s,512MB$ (因为某些原因，auoj上时限为 $16s$)

###### Sol

由于原先的点数和时间都很大，考虑将每个任务看成一个点，处理任务间的关系。

为了简便处理询问，可以看成在每个点时刻 $0$ 都有一个持续时间 $0$，分数 $0$ 的任务，每个点的答案即为从这个任务开始的最大分数。

设 $dp_i$ 表示从任务 $i$ 开始的最大收益，有两种情况:

1. 之后再也不做任务
2. 做下一个任务

对于第二种情况，考虑从一个任务到另外一个任务，需要处理出用 $l$ 个时间，从 $a$ 走到 $b$，最多能得到多少奖励边的分。

只需要求出距离路径 $(a,b)$ 最近的奖励边距离即可，先求出距离每个点最近的奖励边距离，然后相当于路径 $\min$。位置这个值后也容易求出第一种情况的分数。

考虑点分树处理，注意到即使 $a->x->b$ 距离奖励边可能比 $a->b$ 更近，但近的距离不会超过多走的距离，因此这样一定不优，因此点分树上不用考虑必须与当前的点不在同一个子树的限制。

对于两个点 $a,b$ 和一个分治中心 $x$，设 $a$ 的任务结束后到达 $x$ 的时间为 $t_1$，路径上奖励边数量为 $v_1$，距离路径最近的奖励边距离为 $d_1$。再设 $b$ 的任务开始时间减去 $b$ 到达 $x$ 的距离为 $t_2$，路径上奖励边数量为 $v_2$，距离路径最近的奖励边距离为 $d_2$。那么最多能走的奖励边数量为 $v_1+v_2+max(0,2\lfloor\frac{t_2-t_1-\min(d_1,d_2)}2\rfloor)$

对于与 $0$ 取 $\max$ 的部分，考虑分开处理这种情况。$0$ 相当于不走奖励边，因此只需要满足 $t_2\geq t_1$ 即可，这种情况可以每个点上维护一个平衡树解决。

因为前面是减法同时取 $\max$，因此 $\min(d_1,d_2)$ 可以通过枚举哪一个作为 $\min$ 来去掉，然后相当于

1. 给定 $v,t+d$，查询最大的 $dp_x+c(v+v_x+2\lfloor\frac{t_x-t-d}2\rfloor)$ 且 $t_x\geq t$
2. 给定 $v,t$，查询最大的 $dp_x+c(v+v_x+2\lfloor\frac{t_x-d_x-t}2\rfloor)$ 且 $t_x\geq t$

可以将奇数和偶数分开维护，然后平衡树维护即可。

因此每个点需要开 $5$ 个平衡树，分别维护不走奖励边，情况 $1$ 且 $t_x$ 为奇数/偶数，情况 $2$ 且 $t_x-d_x$ 为奇数/偶数的情况。

复杂度 $O(n\log^2 n)$。但因为oj的奇妙原因，现在最快的代码要 $10s$。

~~说不定改成set就能跑了，但是下次再说~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
#include<queue>
#define ll long long
using namespace std;
#define N 200500
map<int,int> id[N];
struct edge{int t,next,v;}ed[N*2];
int n,q,t,s,head[N],cnt,a,b,c,d,dis[N],qu[N],ct1,sb[N],ct2,ds[N][22],mn[N][22],v3[N][22],f1[N],dep[N],sz[N],vis[N],vl1,as1,as2,ls[N];
long long as[N],fu[N][4];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
bool cmp(int a,int b){return fu[a][1]==fu[b][1]?fu[a][2]==fu[b][2]?a>b:fu[a][2]<fu[b][2]:fu[a][1]<fu[b][1];}
void dfs1(int u,int fa)
{
	sz[u]=1;int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],mx=mx<sz[ed[i].t]?sz[ed[i].t]:mx;
	if(mx<vl1-sz[u])mx=vl1-sz[u];
	if(as1>mx)as1=mx,as2=u;
}
void dfs2(int u,int fa,int d,int di,int mi,int v)
{
	di++,mi=min(mi,dis[u]);
	ds[u][d]=di;mn[u][d]=mi;v3[u][d]=v;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs2(ed[i].t,u,d,di,mi,v+ed[i].v);
}
void dfs3(int u,int d)
{
	vis[u]=1;dep[u]=d;
	dfs2(u,0,d,-1,1e9,0);
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]){dfs1(ed[i].t,u);vl1=sz[ed[i].t];as1=1e9;dfs1(ed[i].t,u);f1[as2]=u;dfs3(as2,d+1);}
}
#define M 12330000
int ch[M][2],fa[M],ct;
ll vl[M],v[M],mx[M];
struct Splay{
	int rt;
	void init(){rt=++ct;ch[ct][1]=ct+1;fa[ct+1]=ct;ct++;vl[ct]=1e18;v[ct]=v[ct-1]=mx[ct]=mx[ct-1]=-1e18;vl[ct-1]=-1e18;}
	void pushup(int x){mx[x]=max(v[x],max(mx[ch[x][0]],mx[ch[x][1]]));}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
	void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(g!=y)rotate((ch[g][1]==f)^(ch[f][1]==x)?x:f);rotate(x);}if(!y)rt=x;}
	int getpre(int x,ll v){if(!x)return 0;if(vl[x]>v)return getpre(ch[x][0],v);int st=getpre(ch[x][1],v);return st?st:x;}
	void insert(ll v1,ll v2)
	{
		if(v1<0)return;
		int tp=getpre(rt,v1);
		splay(tp);
		if(vl[tp]==v1){v[tp]=max(v[tp],v2);pushup(tp);return;}
		int st=ch[tp][1];
		while(ch[st][0])st=ch[st][0];
		splay(st,tp);
		ch[st][0]=++ct;v[ct]=mx[ct]=v2;fa[ct]=st;vl[ct]=v1;
		pushup(st);pushup(tp);
	}
	ll query(ll st){ll tp=getpre(rt,st-1);splay(tp);return mx[ch[tp][1]];}
}tr[N][5];
int main()
{
	mx[0]=-1e18;
	scanf("%d%d%d%d",&n,&q,&t,&s);
	for(int i=1;i<=n;i++)dis[i]=1e9;
	queue<int> st;
	for(int i=1;i<n;i++)
	{
		scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
		if(c)dis[a]=dis[b]=1,st.push(a),st.push(b);
	}
	while(!st.empty())
	{
		int f=st.front();st.pop();
		for(int i=head[f];i;i=ed[i].next)if(dis[ed[i].t]>1e8)dis[ed[i].t]=dis[f]+1,st.push(ed[i].t);
	}
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d%d%d",&a,&b,&c,&d);
		if(c||!id[a][b]){fu[++ct2][0]=a;fu[ct2][1]=b;fu[ct2][2]=c+b;fu[ct2][3]=d;if(!c)id[a][b]=ct2;}
		else fu[id[a][b]][3]+=d;
	}
	dfs3(1,1);
	for(int i=1;i<=n;i++)fu[i+ct2][0]=i,ls[i]=i+ct2;
	ct2+=n;
	for(int i=1;i<=ct2;i++)sb[i]=i;
	sort(sb+1,sb+ct2+1,cmp);
	for(int i=ct2;i>=1;i--)
	{
		ll res=fu[sb[i]][3],v1=dis[fu[sb[i]][0]],v2=t-fu[sb[i]][2];
		v2-=v1-1;if(v2<0)v2=0;
		res+=v2*s;
		as[sb[i]]=res;
	}
	for(int i=1;i<=n;i++)for(int j=0;j<5;j++)tr[i][j].init();
	for(int i=ct2;i>=1;i--)
	{
		int tp=sb[i];
		for(int j=fu[tp][0];j;j=f1[j])
		{
			int de=dep[j];
			ll t1=fu[tp][2]+ds[fu[tp][0]][de],v1=mn[fu[tp][0]][de]-1,f1=v3[fu[tp][0]][de];
			ll as1=tr[j][4].query(t1);
			as[tp]=max(as[tp],as1+fu[tp][3]+f1*s);
			ll as2=tr[j][1].query(t1),as3=tr[j][3].query(t1);
			if(t1&1)
			{
				as[tp]=max(as[tp],as2-(t1+1)/2*s*2+f1*s+fu[tp][3]);
				as[tp]=max(as[tp],as3-t1/2*s*2+f1*s+fu[tp][3]);
			}
			else
			{
				as[tp]=max(as[tp],as2-t1/2*s*2+f1*s+fu[tp][3]);
				as[tp]=max(as[tp],as3-t1/2*s*2+f1*s+fu[tp][3]);
			}
			ll as4=tr[j][0].query(t1+v1*2),as5=tr[j][2].query(t1+v1*2);
			if((t1+2*v1)&1)
			{
				as[tp]=max(as[tp],as4-(t1+2*v1+1)/2*s*2+f1*s+fu[tp][3]);
				as[tp]=max(as[tp],as5-(t1+2*v1)/2*s*2+f1*s+fu[tp][3]);
			}
			else
			{
				as[tp]=max(as[tp],as4-(t1+2*v1)/2*s*2+f1*s+fu[tp][3]);
				as[tp]=max(as[tp],as5-(t1+2*v1)/2*s*2+f1*s+fu[tp][3]);
			}
		}
		if(tp<=ct2-n)
		for(int j=fu[tp][0];j;j=f1[j])
		{
			int de=dep[j];
			ll t1=fu[tp][1]-ds[fu[tp][0]][de],v1=mn[fu[tp][0]][de]-1,f1=v3[fu[tp][0]][de];
			tr[j][4].insert(t1,as[tp]+f1*s);
			tr[j][(t1&1)<<1].insert(t1,as[tp]+t1/2*s*2+f1*s);
			tr[j][((t1-2*v1)&1)<<1|1].insert(t1-2*v1,as[tp]+(t1-2*v1)/2*s*2+f1*s);
		}
	}
	for(int i=1;i<=n;i++)printf("%lld ",as[ls[i]]);
}
```



#### NOI2020模拟六校联测11

##### auoj427 异或树

###### Problem

给一个 $n$ 个点的完全图，每个点有一个 $[0,2^m)$ 中的权值，每条边的权值为两个点权值的异或。

求出对于所有 $2^{nm}$ 种点权，这个图的最小生成树的边权和的和，答案对 $p$ 取模。

$n\leq 50,m\leq 8,10^8\leq p\leq 10^9$

$1s,512MB$

###### Sol

考虑点权的最高位。如果所有点在这一位上都相同，那么可以直接删去最高位。

否则，根据最小生成树的贪心思路，一定是先让两侧分别连通，然后中间加入一条边。

设 $f_{i,j}$ 表示 $i$ 个点，权值上限为 $2^j-1$ 的所有方案的权值和。由期望线性性，中间的边的权值可以和两侧的权值分开求和，转移时只需要考虑左侧有 $a$ 个点，右侧有 $b$ 个点时，所有情况中间这条边的最小权值的和。

相当于给 $a+b$ 个在 $[0,2^j)$ 的数，求所有情况下前 $a$ 个数与后 $b$ 个数两两异或得到的最小数的和。

设 $dp_{a,b,j,k}$ 表示给 $a+b$ 个在 $[0,2^j)$ 的数，前 $a$ 个数与后 $b$ 个数两两异或得到的最小数为 $k$ 的方案数，要求 $a,b>0$，考虑转移。

考虑从最高位分开，分成两部分。如果两部分都有 $a,b>0$ ，那么 $\min$ 的最高位一定不为 $1$，因此这一部分的转移为：

$$
dp1_{a,b,j,k}=\sum_{s=1}^{a-1}\sum_{t=1}^{b-1}\sum_{p=0}^{2^{j-1}-1}\sum_{q=0}^{2^{j-1}-1}[min(p,q)=k]dp_{s,t,j-1,p}dp_{i-s,j-t,j-1,q}C_i^sC_j^t
$$

可以使用前缀和优化 $\min$ 部分，复杂度 $O(n^42^m)$。

如果一个部分 $a,b>0$，另外一个部分有 $=0$ 的，那么 $\min$ 一定在第一个部分取到，有：

$$
dp2_{a,b,j,k}=dp_{a,b,j-1,k}+2(\sum_{i=1}^{a-1}dp_{i,b,j-1,k}2^{(j-1)(a-i)}C_a^i+\sum_{i=1}^{b-1}dp_{a,i,j-1,k}2^{(j-1)(b-i)}C_b^i)
$$

否则，一定是一部分是前 $a$ 个数，另外一部分是后 $b$ 个数，除去最高位后变成了子问题，因此有：

$$
dp3_{a,b,j,k}=2dp_{a,b,j-1,k-2^{j-1}}
$$

三部分相加即可。最后用 $dp$ 容易 $O(n^2m)$ 求出 $f$。

复杂度 $O(n^42^m)$，常数很小。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 53
int n,m,mod,dp[8][N][N][257],f[N][N][8],g[N][9],c[N][N],pw[N*N];
int main()
{
	scanf("%d%d%d",&n,&m,&mod);
	if(n==1){printf("0\n");return 0;}
	for(int i=1;i<=n;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	pw[0]=1;for(int i=1;i<=n*m;i++)pw[i]=pw[i-1]*2%mod;
	for(int i=1;i<=n;i++)
	for(int j=1;i+j<=n;j++)
	dp[0][i][j][0]=1;
	for(int i=1;i<m;i++)
	{
		//type 1
		for(int j=1;j<=n;j++)
		for(int k=1;j+k<=n;k++)
		for(int p=1;p+j+k<=n;p++)
		for(int q=1;p+q+j+k<=n;q++)
		{
			int st=1ll*c[j+p][p]*c[k+q][q]%mod,s1=0,s2=0;
			for(int l=(1<<i-1)-1;l>=0;l--)
			s1+=dp[i-1][p][q][l],dp[i][j+p][k+q][l]=(dp[i][j+p][k+q][l]+(1ll*dp[i-1][j][k][l]*s1+1ll*dp[i-1][p][q][l]*s2)%mod*st)%mod,s2+=dp[i-1][j][k][l],s1-=(s1>=mod?mod:0),s2-=(s2>=mod?mod:0);
		}
		//type 2
		for(int j=1;j<=n;j++)
		for(int k=1;j+k<=n;k++)
		for(int l=(1<<i-1)-1;l>=0;l--)
		{
			int res=2ll*dp[i-1][j][k][l]%mod;
			for(int p=0;p+j+k<=n;p++)
			dp[i][j+p][k][l]=(dp[i][j+p][k][l]+1ll*c[j+p][p]*res)%mod,dp[i][j][k+p][l]=(dp[i][j][k+p][l]+(p>0)*1ll*c[k+p][p]*res)%mod,res=1ll*res*pw[i-1]%mod;
		}
		//type 3
		for(int j=1;j<=n;j++)
		for(int k=1;j+k<=n;k++)
		for(int l=(1<<i-1)-1;l>=0;l--)
		dp[i][j][k][l+(1<<i-1)]=2ll*dp[i-1][j][k][l]%mod;
	}
	for(int i=0;i<m;i++)
	for(int j=1;j<=n;j++)
	for(int k=1;j+k<=n;k++)
	for(int l=0;l<(1<<i);l++)
	f[j][k][i]=(f[j][k][i]+1ll*(l+pw[i])*dp[i][j][k][l])%mod;
	for(int j=1;j<=m;j++)
	for(int i=1;i<=n;i++)
	{
		g[i][j]=2*g[i][j-1]%mod;
		for(int k=1;k<i;k++)
		g[i][j]=(g[i][j]+1ll*c[i][k]*(1ll*g[k][j-1]*pw[(i-k)*(j-1)]%mod+1ll*g[i-k][j-1]*pw[k*(j-1)]%mod)%mod+1ll*c[i][k]*f[k][i-k][j-1])%mod;
	}
	printf("%d\n",g[n][m]);
}
```



##### auoj428 密码

###### Problem

给定长度为 $n$ 的 $S$ 与长度为 $m$ 的 $T$ ，它们都是数字串。

$S$ 将会以某种概率生成，对于第 $i$ 位，它有 $p_{i,j}$ 的概率为 $j$ 。

对于每一个可能的 $i$，求出 $S_{i,...,i+m-1}=T$ 的概率，误差不超过 $10^{-9}$

$n\leq 2\times 10^5,m\leq 5\times 10^4$

$2s,512MB$

###### Sol

对于一个 $i$，找到 $p_{i,j}$ 最大的一个 $j$。则如果这一位上不是这个数，那么概率至少乘 $\frac 12$。

那么如果最后答案大于 $10^{-6}$，则最多会有 $\log$ 次与最大的不同。如果超过了 $\log$ 次可以直接输出 $0$。

考虑向右匹配与最大的相同的情况，求能匹配的长度相当于求两个后缀的 `lcp`，求这一段的概率可以线段树维护乘积。`lcp` 也可以暴力二分加上 `hash` 判断。

从一个点开始最多匹配 $O(\log v)$ 次就可以结束，复杂度 $O(n\log m\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ch 131
#define md 1000000009
int n,m,st[N][10],fu[N],s[N],pw[N],h1[N],h2[N];
char s1[N];
struct segt{
	struct node{int l,r;double su;}e[N*4];
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r){e[x].su=st[l][fu[l]]/1e9;return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);e[x].su=e[x<<1].su*e[x<<1|1].su;}
	double query(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].su;int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return query(x<<1,l,mid)*query(x<<1|1,mid+1,r);}
}tr;
int lcp(int i,int j)
{
	int lb=1,rb=min(m-i,n-j),as=0;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if((h2[i+mid-1]-1ll*h2[i-1]*pw[mid]%md+md)%md==(h1[j+mid-1]-1ll*h1[j-1]*pw[mid]%md+md)%md)as=mid,lb=mid+1;
		else rb=mid-1;
	}
	return as;
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)
	for(int j=0;j<10;j++)
	scanf("%d",&st[i][j]),fu[i]=st[i][fu[i]]<st[i][j]?j:fu[i];
	tr.build(1,1,n);
	scanf("%s",s1+1);
	for(int i=1;i<=m;i++)s[i]=s1[i]-'0';
	pw[0]=1;for(int i=1;i<=n;i++)pw[i]=1ll*pw[i-1]*ch%md;
	for(int i=1;i<=n;i++)h1[i]=(1ll*h1[i-1]*ch+fu[i])%md;
	for(int i=1;i<=m;i++)h2[i]=(1ll*h2[i-1]*ch+s[i])%md;
	for(int i=1;i<=n-m+1;i++)
	{
		int v1=1,v2=i;
		double as=1;
		while(v1<=m)
		{
			int tp=lcp(v1,v2);
			if(tp)as*=tr.query(1,v2,v2+tp-1);
			v1+=tp,v2+=tp;
			as*=st[v2][s[v1]]/1e9;v1++,v2++;
			if(as<1e-11)break;
		}
		printf("%.15lf\n",as);
	}
}
```



##### auoj429 排列

###### Problem

对于一个排列，定义它的权值为有多少个 $i$ 满足 $|p_i-i|=k$ 

对于每个 $0\leq i\leq n$ ，求出权值为 $i$ 的 $n$ 阶排列数量。

$n,k\leq 10^5$

$1s,512MB$

###### Sol

设 $f_i$ 表示权值为 $i$ 的 $n$ 阶排列数量，$g_i=\sum_{j\geq i}f_jC_j^i$。

考虑 $g_i$ 的组合意义，相当于选出 $i$ 个满足条件的位置的方案数。

建立 $2n$ 个点，定义选择一条 $(i,j+n)$ 的边表示让 $p_i=j$，那么会连所有的 $(i,i+n+k)(i+k\leq n),(i,i+n-k)(i>k)$。

选择一个位置满足某种条件相当于选择一条边。显然一个点只能与一条选中的边相邻，而满足这个条件就是合法的。因此 $g_i$ 相当于在上面的图中选出 $i$ 条不相邻的边的方案数，再乘上 $(n-i)!$。

可以发现这个图由若干条链构成，对于一条 $i$ 条边的链，选出 $j$ 条边的方案数为 $C_{i-j+1}^j$

容易发现图中只有两种长度 $\lfloor\frac nm\rfloor,\lfloor\frac nm\rfloor-1$ 的链，那么可以点值处理一种长度的快速幂，合并点值再 `IDFT` 求出方案数，最后二项式反演回去即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<map>
using namespace std;
#define N 530001
#define mod 998244353
int n,m,ntt[N],rev[N],fr[N],ifr[N],g[2][N*2],f1[N],f2[N],f3[N],t1[N],t2[N];
map<int,int> ct;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void pre()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<1<<19;j<<=1)
	{
		int st=pw(3,(mod-1)/j),vl=1;
		if(i==0)st=pw(st,mod-2);
		for(int k=0;k<j>>1;k++)g[i][j+k]=vl,vl=1ll*vl*st%mod;
	}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=i;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][st]%mod;
		ntt[k]=(v1+v2)%mod,ntt[k+(i>>1)]=(v1-v2+mod)%mod;
	}
	int inv=pw(s,t?0:mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d%d",&n,&m);pre();
	fr[0]=ifr[0]=1;
	for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	int l=1;while(l<=n*2+3)l<<=1;
	for(int i=0;i<l;i++)t1[i]=1;
	for(int i=1;i<=m;i++)ct[(n-i)/m]+=2;
	for(map<int,int>::iterator it=ct.begin();it!=ct.end();it++)
	{
		int s1=it->first,s2=it->second;
		if(s1<1)continue;
		for(int i=0;i*2<=s1+1;i++)t2[i]=1ll*fr[s1+1-i]*ifr[i]%mod*ifr[s1+1-i*2]%mod;
		dft(l,t2,1);
		for(int i=0;i<l;i++)t1[i]=1ll*t1[i]*pw(t2[i],s2)%mod,t2[i]=0;
	}
	dft(l,t1,0);
	for(int i=0;i<=n;i++)t1[i]=1ll*t1[i]*fr[n-i]%mod;
	for(int i=0;i<l;i++)f1[i]=f2[i]=f3[i]=0;
	for(int i=0;i<=n;i++)f1[i]=1ll*t1[i]*fr[i]%mod,f2[n-i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	dft(l,f1,1);dft(l,f2,1);for(int i=0;i<l;i++)f3[i]=1ll*f1[i]*f2[i]%mod;dft(l,f3,0);
	for(int i=0;i<=n;i++)printf("%d\n",1ll*f3[i+n]*ifr[i]%mod);
}
```



#### NOI2020模拟六校联测12

##### auoj430 字符串计数

###### Problem

给定 $n,m,k$，求在字符集大小为 $k$ 的情况下，有多少对字符串 $s,t$ 满足如下条件：

1. $|s|=n,|t|=m$
2. $t$ 是 $s$ 的子串。

答案模 $10^9+7$

$n\leq 200,m\leq 50$

$1s,512MB$

###### Sol

考虑确定了 $t$ 后，如何确定 $s$ 的数量。

考虑计算 $s$ 不包含 $t$ 作为子串的方案数。一种 `dp` 方式为记录当前 $s$ 后缀匹配 $t$ 前缀的最长长度。但这样转移时需要 $t$ 的 `border` 以及下一个字符，而下一个字符的信息种类数数量为集合划分，这不能接受。

考虑容斥计算，枚举一些位置，钦定这些位置是 $t$。那么设 $dp_i$ 表示上一个钦定的结尾为 $i$ 的方案数，则转移为：

$$
dp_i=-\sum_{j=0}^{i-1}dp_j*v_{j,i}
$$

其中 $v_{j,i}$ 为转移系数。当 $i-j\geq m$ 时，$v_{j,i}=k^{i-j-m}$，否则 $v_{j,i}=[t[1,m-i+j]=t[i-j,m]]$。

这个转移只和 $t$ 的 `border` 集合有关。可以发现可能的 `border` 集合数量不多，只要能枚举所有 `border` 集合以及出现次数，就可以得到答案。



那么考虑如何枚举 `border` 集合。

考虑从小到大枚举 `border`，但显然不是所有可能的集合都是合法的。可以发现有如下性质：

如果 $x,y$ 都是 `border`，且 $x<y,2x-y>0$，则 $2x-y$ 必须是 `border`。

证明：由 `border` 的性质，有 $s[1,x]=s[m-x+1,m],s[1,y]=s[m-y+1,m]$，则两个前缀的后 $x$ 位相同，因此 $s[y-x+1,y]=s[1,x]$。而 $2x-y>0$，因此两个串有公共部分，从而两个串开头 $2x-y$ 位，结尾 $2x-y$ 位全部相等。由此可以得到 $2x-y$ 是一个 `border`。

满足这个条件的串数量已经很少（$3\tiems 10^4$ 以内），因此 `dfs` 是可行的，但还需要考虑方案数的问题。

假设当前枚举的最后一个 `border` 为 $l$，考虑记录 $[1,l]$ 内满足 `border` 是当前枚举的情况的方案数。枚举下一个 `border` 为 $x$。考虑计算填 $[1,x]$ 使得前面的 `border` 仍然存在的方案数。

如果前面的 `border` 仍然存在，则只需要 $s[1,l]=s[x-l+1,x]$。如果两段相交，则可能的方案唯一，合法当且仅当满足上面的条件，即对于任意小的 `border` $l$ 都有 $2l-x\leq 0$ 或者 $2l-x$ 是 `border`。否则，中间可以任意填，可以乘上 $k^{l-2x}$ 的方案数。

但这样会计算一些 $[l+1,x-1]$ 部分有 `border` 的情况，考虑如何减去这部分。一种方式是枚举实际上这段出现的第一个 `border` $y$，则需要减去的方案为前 $x$ 位中，满足 $[1,l]$ 为之前的 `border`，$y$ 是 $l$ 后的第一个 `border`，同时 $x$ 也是 `border` 的方案。

考虑在 `dfs` 时，对于每个 $x$ 求出，有多少种填前 $x$ 位的方案满足前 $l$ 位的 `border` 情况和当前情况相同，则上面需要减去的情况为前 $y$ 位的 `border` 集合为当前枚举的集合加上 $y$ 的情况的方案数，向下 `dfs` 的过程会求出这个值。如果从小到大考虑每个 $x$，则可以记录前面部分对于每个 $x$ 多算的量，这样就可以求出实际上的情况数。

考虑如何维护方案数，一种方式是枚举下一个 `border` 的位置，然后将每一个下个位置的方案数求和，最后加上当前第 $l$ 位的情况。可以发现上一步中记录每个 $x$ 多算的量部分与这里求和的值一样，因此上面过程结束后就可以得到当前情况的结果。

复杂度 $O(n^2f(m)+m^2s(m))$，$f(m)$ 为不同 `border` 数量，这里为 `A005434`，$m=50$ 时不超过 $3000$。$s(m)$ 为 `dfs` 状态数，直接实现状态数不超过 $3\times 10^4$，还可以剪枝变得更优，所以能过。

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 233
#define mod 1000000007
int n,m,a,as1,st[N],dp[N],is[N],rb;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int solve()
{
	for(int i=0;i<=n;i++)dp[i]=0;
	for(int i=m;i<=n;i++)
	{
		dp[i]=pw(a,i-m);
		for(int j=m;j<i;j++)
		{
			if(j+m<=i)dp[i]=(dp[i]-1ll*dp[j]*pw(a,i-m-j)%mod+mod)%mod;
			else
			{
				int st1=m-(i-j);
				if(is[st1])dp[i]=(dp[i]-dp[j]+mod)%mod;
			}
		}
	}
	int as=0;
	for(int i=m;i<=n;i++)as=(as+1ll*dp[i]*pw(a,n-i))%mod;
	return as;
}
vector<int> dfs(int fu)
{
	if(is[m])
	as1=(as1+1ll*fu*solve())%mod;
	int las=st[rb];
	vector<int> tp;
	tp.resize(m+1);
	tp[las]=(tp[las]+fu)%mod;
	for(int i=las+1;i<=m;i++)
	{
		st[++rb]=i;
		int fuc=fu,fg=1;
		for(int j=1;j<=rb-1;j++)
		{
			int v1=st[j]*2-i;
			if(v1<=0)continue;
			if(!is[v1])fg=0;
		}
		if(fg)
		{
			int tp1=i-las*2;
			if(tp1>0)fuc=1ll*fuc*pw(a,tp1)%mod;
			fuc=(fuc-tp[i]+mod)%mod;
			if(fuc)
			{
				is[i]=1;
				vector<int> tp1=dfs(fuc);
				for(int i=0;i<=m;i++)tp[i]=(tp[i]+tp1[i])%mod;
			}
		}
		is[i]=0;rb--;
	}
	return tp;
}
int main()
{
	scanf("%d%d%d",&n,&m,&a);
	dfs(1);printf("%d\n",as1);
}
```



##### auoj431 两人距离的概算

###### Problem

给一张 $n$ 个点 $m$ 条边的连通图，满足 $m-n\leq 9$。

对于每个点，求出所有点到它的最短路长度的平均数，答案模 $10^9+7$

$n\leq 10^5$

$2s,512MB$

###### Sol

首先找出图的一棵生成树，然后对于剩下的边，记录它两侧的点将所有这些点建虚树，定义虚树上的所有点为关键点。

显然这些关键点将图分成了若干棵树，且每棵树最多和两个关键点相邻。关键点只有 $O(m-n)$ 个。

可以将只和一个关键点相邻且相邻的关键点相同的树看成一一个部分，这样只有 $O(m-n)$ 个部分。

除了起点所在的部分外，到达其它的部分都一定要经过关键点。

先求出起点到每个关键点的距离，考虑除了起点所在的树外的每一个部分，如果这个部分只和一个关键点相邻，那么只需要记录所有点到关键点的距离和以及这个部分的点数即可求出起点到这部分内的距离。这可以预处理时 `dfs` 求出。

如果这个部分和两个关键点相邻，对于一个点，设它到两个关键点的距离为 $x,y$，两个关键点到起点的距离为 $d_1,d_2$，那么最后的距离为 $min(x+d_1,y+d_2)$，这只和 $x-y,d_2-d_1$ 的大小关系有关。因此可以将每个这样的部分的 $x-y$ 排序，然后维护前缀和，询问时二分即可。

然后考虑起点所在的部分，如果这个部分只和一个关键点相邻，那么显然路径不会走到这个部分外面去（可能到关键点上）。因此相当于求一个点到树上所有点的距离和，对每一部分两次 `dfs` 即可。

如果这个部分和两个关键点相邻，则可能从一个关键点出去到另外一个关键点回来。先求出这两个关键点的距离，只考虑这部分内的点和关键点间的路径则可以看成一个基环树。

把环拿出来，先处理环上每个点不在环上的子树到这个点的距离和，然后考虑环上的路径。考虑环上一个点出发的情况，每个点一定是向环上一个区间中的点走时走顺时针方向，另外一个区间时走逆时针方向，可以 $O(1)$ 找到两种路径的分界点，然后需要维护环上形如 $\sum size_i*(i-k+1)$ 的和，维护 $size_i,size_i*i$ 的前缀和即可，最后再从上往下 `dfs` 一次求出不在环上的点到这个部分其它点的距离和。

复杂度 $O(n(m-n)^2+n(m-n)\log n)$，细节非常多。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 100500
#define K 81
#define mod 1000000007
#define ll long long
int n,m,q,head[N],cnt=1,fa[N],s[N][2],used[N],x,dis[K][N],bel[N],dp[N],ct,ct2,id[N],tid[N],f1[N],dis1[K][K],ct1,lb[N],is[N],sz[N],st[N],f[N][18],dep[N],sta[N],rb,rt,ds2[N],su2[N],su3[N];
int fu1[N],st1[N];
ll as1[N],as2[N],sz1[N],sz2[N],vl[N],as[N],vl2[N];
vector<int> sth[K],su1[K];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
bool cmp(int a,int b){return lb[a]<lb[b];}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs0(int u,int fa){lb[u]=++ct1;f[u][0]=fa;for(int i=1;i<=17;i++)f[u][i]=f[f[u][i-1]][i-1];dep[u]=dep[fa]+1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);}
void dfs1(int u,int fa,int v){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dis[v][ed[i].t]=dis[v][u]+1,dfs1(ed[i].t,u,v);}
void dfs2(int u,int fa,int id1){bel[u]=id1;sz[u]=1;vl[u]=0;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!id[ed[i].t])dfs2(ed[i].t,u,id1),sz[u]+=sz[ed[i].t],vl[u]+=vl[ed[i].t]+sz[ed[i].t];else if(id[ed[i].t])is[i/2]=1;}
void dfs3(int u,int fa,int s1){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!id[ed[i].t])vl[ed[i].t]=vl[u]+(s1-2*sz[ed[i].t]),dfs3(ed[i].t,u,s1);}
void dfs4(int u,int fa,int id1){bel[u]=id1;sz[u]=1;vl[u]=0;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!id[ed[i].t])dfs4(ed[i].t,u,id1),sz[u]+=sz[ed[i].t],vl[u]+=vl[ed[i].t]+sz[ed[i].t];}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=17;i>=0;i--)if(dep[x]-dep[y]>=(1<<i))x=f[x][i];if(x==y)return x;for(int i=17;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int main()
{
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=n;i++)fa[i]=i;
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(finds(s[i][0])!=finds(s[i][1]))used[i]=1,fa[finds(s[i][0])]=finds(s[i][1]),adde(s[i][0],s[i][1]);
	}
	if(m==n-1)
	{
		dfs2(1,0,1);dfs3(1,0,sz[1]);
		while(q--){scanf("%d",&x);printf("%d\n",vl[x]%mod*pw(n,mod-2)%mod);}
		return 0;
	}
	dfs0(1,0);
	for(int i=1;i<=m;i++)if(!used[i])st[++ct]=s[i][0],st[++ct]=s[i][1];
	sort(st+1,st+ct+1,cmp);
	for(int i=1;i<=ct*2;i++)for(int j=1;j<=ct*2;j++)if(i!=j)dis1[i][j]=1e9;
	for(int i=1;i<=ct;i++)if(st[i]!=st[i-1])
	{
		if(!id[st[i]])id[st[i]]=++ct2,tid[ct2]=st[i];
		if(!rb){sta[++rb]=st[i];continue;}
		int tp=LCA(st[i],sta[rb]);
		if(!id[tp])id[tp]=++ct2,tid[ct2]=tp;
		while(rb&&dep[sta[rb]]>dep[tp])
		{
			f1[id[sta[rb]]]=id[sta[rb-1]];
			if(dep[tp]>dep[sta[rb-1]])f1[id[sta[rb]]]=id[tp];
			dis1[id[sta[rb]]][f1[id[sta[rb]]]]=dis1[f1[id[sta[rb]]]][id[sta[rb]]]=dep[sta[rb]]-dep[tid[f1[id[sta[rb]]]]];
			rb--;
		}
		if(dep[sta[rb]]<dep[tp])sta[++rb]=tp;
		if(dep[sta[rb]]<dep[st[i]])sta[++rb]=st[i];
	}
	while(rb)
	{
		dis1[id[sta[rb]]][id[sta[rb-1]]]=dis1[id[sta[rb-1]]][id[sta[rb]]]=dep[sta[rb]]-dep[sta[rb-1]];
		f1[id[sta[rb]]]=id[sta[rb-1]];
		rb--;
	}
	for(int i=1;i<=m;i++)if(!used[i])dis1[id[s[i][0]]][id[s[i][1]]]=dis1[id[s[i][1]]][id[s[i][0]]]=min(dis1[id[s[i][0]]][id[s[i][1]]],1);
	for(int k=1;k<=ct2;k++)for(int i=1;i<=ct2;i++)for(int j=1;j<=ct2;j++)dis1[i][j]=min(dis1[i][j],dis1[i][k]+dis1[k][j]);
	for(int i=1;i<=ct2;i++)dfs1(tid[i],0,i);
	for(int t=1;t<=ct2;t++)if(f1[t]&&!id[f[tid[t]][0]])
	{
		int ct=0,ssz=0;
		dfs2(f[tid[t]][0],tid[t],t+ct2);
		int f0=tid[t],f2=tid[f1[t]],d1=dis1[t][f1[t]];
		while(f0!=f2)st1[ct++]=f0,fu1[f0]=1,f0=f[f0][0];
		st1[ct]=f2;fu1[f2]=1;
		ssz=ct-1;
		for(int i=1;i<ct;i++)sz[st1[i]]=1,bel[st1[i]]=t+ct2,vl[st1[i]]=0;
		for(int i=1;i<ct;i++)
		for(int j=head[st1[i]];j;j=ed[j].next)if(!fu1[ed[j].t])
		{
			dfs4(ed[j].t,st1[i],t+ct2),ssz+=sz[ed[j].t];
			sz[st1[i]]+=sz[ed[j].t],vl2[st1[i]]+=vl[ed[j].t]+sz[ed[j].t];
		}
		ll su1=0;
		for(int i=1;i<ct;i++)su1+=vl2[st1[i]],su2[i]=su2[i-1]+sz[st1[i]],su3[i]=su3[i-1]+sz[st1[i]]*i;
		for(int i=1;i<ct;i++)
		{
			vl[st1[i]]+=su1;
			int v1=(2*i-ct-d1)/2;
			if(v1>0)vl[st1[i]]+=su3[v1]+su2[v1]*(ct+d1-i);else v1=0;
			vl[st1[i]]+=i*(su2[i]-su2[v1])-su3[i]+su3[v1];
			int v3=(ct+d1+2*i+1)/2;
			if(v3<ct)vl[st1[i]]+=(ct+d1+i)*(su2[ct-1]-su2[v3-1])-su3[ct-1]+su3[v3-1];else v3=ct;
			vl[st1[i]]+=su3[v3-1]-su3[i]-i*(su2[v3-1]-su2[i]);
		}
		for(int i=1;i<ct;i++)
		for(int j=head[st1[i]];j;j=ed[j].next)if(!fu1[ed[j].t])
		vl[ed[j].t]=vl[st1[i]]+(ssz-2*sz[ed[j].t]),dfs3(ed[j].t,st1[i],ssz);
	}
	else if(f1[t]){for(int j=head[tid[t]];j;j=ed[j].next)if(ed[j].t==tid[f1[t]])is[j/2]=1;}
	for(int i=1;i<=ct2;i++)
	for(int j=head[tid[i]];j;j=ed[j].next)if(!is[j/2])
	dfs4(ed[j].t,tid[i],i),as1[i]+=vl[ed[j].t]+sz[ed[j].t],sz1[i]+=sz[ed[j].t];
	for(int i=1;i<=ct2;i++)
	for(int j=head[tid[i]];j;j=ed[j].next)if(!is[j/2])
	vl[ed[j].t]=as1[i]+sz1[i]-2*sz[ed[j].t],dfs3(ed[j].t,tid[i],sz1[i]);
	for(int i=1;i<=n;i++)if(bel[i]>ct2)
	{
		int tp=bel[i]-ct2,tp1=f1[tp];
		int ds1=dis[tp][i],ds2=dis[tp1][i];
		as2[tp+ct2]+=ds1;sth[tp].push_back(ds1-ds2);sz1[tp+ct2]++;
	}
	for(int i=1;i<=ct2;i++)sort(sth[i].begin(),sth[i].end());
	for(int i=1;i<=ct2;i++)if(sth[i].size())
	{
		su1[i].push_back(sth[i][0]);
		for(int j=1;j<sth[i].size();j++)su1[i].push_back((su1[i][j-1]+sth[i][j])%mod);
	}
	while(q--)
	{
		scanf("%d",&x);
		if(as[x]){printf("%d\n",as[x]);continue;}
		ll as4=vl[x];
		for(int i=1;i<=ct2;i++)ds2[i]=dis[i][x];
		for(int i=1;i<=ct2;i++)
		for(int j=1;j<=ct2;j++)
		ds2[j]=min(ds2[j],ds2[i]+dis1[i][j]);
		for(int i=1;i<=ct2;i++)as4+=ds2[i];
		for(int i=1;i<=ct2;i++)if(i!=bel[x])
		as4+=as1[i]+1ll*sz1[i]*ds2[i];
		for(int i=1;i<=ct2;i++)if(i+ct2!=bel[x]&&f1[i]&&sth[i].size())
		{
			int tp1=i,tp2=f1[i],fu=ds2[tp2]-ds2[tp1];
			as4+=as2[i+ct2]+1ll*sz1[i+ct2]*ds2[i];
			int lb=0,rb=sth[i].size()-1,as=rb+1;
			while(lb<=rb)
			{
				int mid=(lb+rb)>>1;
				if(sth[i][mid]>fu)as=mid,rb=mid-1;
				else lb=mid+1;
			}
			if(as<sth[i].size())
			{
				int ct=sth[i].size()-as,vl=su1[i][sth[i].size()-1];
				if(as>0)vl=(vl-su1[i][as-1]+mod)%mod;
				vl=(-vl+1ll*ct*fu)%mod;
				as4+=vl;
			}
		}
		as4%=mod;as[x]=1ll*as4*pw(n,mod-2)%mod;
		printf("%d\n",as[x]);
	}
}
```



##### auoj432 随机变换的子串

###### Problem

称两个只包含 `ab` 的字符串 $s,t$ 是等价的，当且仅当可以通过如下操作将 $s$ 变成 $t$：

1. 插入或者删除一个连续的 `aa`。
2. 插入或者删除一个连续的 `bbb`。
3. 插入或者删除一个连续的 `ababab`。

给一个长度为 $n$ 的字符串 $s$，$q$ 次询问，每次给定 $l_1,r_1,l_2,r_2$，求 $s[l_2,r_2]$ 中有多少个子串和 $s[l_1,r_1]$ 等价。相同子串出现多次算多次。

$n\leq 10^5,q\leq 5\times 10^4$

$2s,512MB$

###### Sol

题解告诉我们，这个操作变换是一个正四面体的群。考虑一个四面体，以 $1$ 为上方顶点时，下面三个顶点按照逆时针排序。则有如下构造方式：

令 `b` 操作为，固定上方顶点不变，将正四面体旋转 $60$ 度。如果看成置换，则它相当于 $(1,3,4,2)$。显然 `bbb` 和不操作相同。

令 `a` 操作为，从一条边的角度看四面体，然让四面体向左逆时针转 $180$ 度，改变这条边的方向。如果选择 $(1,2)$ 边，则可以发现它相当于变换 $(2,1,4,3)$。显然 `aa` 和不操作相同。

可以发现，`ab` 等于置换 $(3,1,2,4)$，因此 `ababab` 和不操作相同。

可以发现这样的变换可以表示所有偶置换，因此这样可以得到群 $A_4$，而这也是一个正四面体在不镜像的情况下的所有变换（因为有手性）。



那么首先有如下结论：如果两个字符串等价，则它们看成置换后的结果相等。

为了证明这是充分必要的，只需要证明任何看成置换后等于 $(1,2,3,4)$ 的序列都可以变成空，对于剩下的情况，可以对每种串钦定一个表示，对于一个串，通过一些加入操作使得结尾为钦定的表示，前面为等价于 $(1,2,3,4)$ 的情况。

然后考虑按照长度从小到大归纳，只需要证明长度不超过 $12$ 的串都可以做到，枚举一下暴力验证可以发现这是对的。~~当然也可以不证明直接rush~~

考虑线段树维护，对于一个区间，记录区间内所有子串的情况，所有前缀的情况，所有后缀的情况，直接合并即可。预处理置换相乘的结果就可以 $12^2$ 的合并。

复杂度 $O((n+q*\log n)*12^2)$

###### Code

在写了

#### SCOI2020模拟21

##### auoj457 矩形

不会surreal number，跑路了

##### auoj458 序列

###### Problem

给一个长度为 $n$ 的序列 $v$。定义 $f(k)$ 为如下问题的答案：

你可以对序列进行 $k$ 次操作，每次操作为选择一个数将它减一，可以减到负数。你希望最小化操作后的最小非空子段和，求这个值。

给定 $m$，求 $\sum_{i=1}^mf(i)$，答案模 $998244353$。

$n\leq 10^5,m\leq 10^{13}$

$2s,1024MB$

###### Sol

显然 $f(k)$ 是单调不增的，考虑对于一个 $v$，求出至少需要操作多少次才能使得最小非空子段和小于等于 $v$。设这个值为 $g(v)$。

考虑一种最优的操作方式。对于任意一个操作的位置，一定存在一个包含这个位置的区间，使得这个区间当前的和为 $v$，否则删去这次操作更优。

考虑如果选出的两个区间重合，则可以删去被包含的区间。如果两个区间 $[l_1,r_1],[l_2,r_2]$ 相交但不重合（$l_1<l_2\leq r_1<r_2$），则因为这是合法方案，则 $[l_1,r_2],[l_2,r_1]$ 部分和都小于等于 $v$，但因为 $[l_1,r_1],[l_2,r_2]$ 区间的和都等于 $v$，且两部分两个区间的和加起来相等，因此 $[l_1,r_2],[l_2,r_1]$ 的区间和都是 $v$。因此可以换成两个区间的并。

因此在最优方式中可以选择若干个不交区间覆盖所有操作位置，且每个区间当前的和都是 $v$，因此操作次数为选出的区间的总和减去 $v$ 乘以选出的区间数量。

从另外一个方向考虑，如果选出若干个不交区间，则每个区间的和都不能超过 $v$，因此对于任意一种选区间的方式操作次数不小于选出的区间的总和减去 $v$ 乘以选出的区间数量。

而上面说明了最优解能取到一个这样的值，因此最少的操作次数即为所有方案中这个值的最大值。

首先考虑对于每个 $i$，求出选择 $i$ 个不交非空区间，区间和的最大值。如果没有非空条件，则这是一个费用流模型。加上非空条件后，可以发现在 $i$ 不超过正数个数的情况下可以无视非空条件，接下来的过程一定是从大到小依次选负数，因此这还是凸的。一种方式是使用线段树维护不考虑非空的增广，支持区间取反求最大子段和，最后特判负数部分。

另外一种方式是，通过费用流可以发现这是一个凸函数，考虑分治，设 $f_{l,r,k,0/1,0/1}$ 表示区间 $[l,r]$ 中选择 $k$ 个不交区间，钦定左右端点是否必须选时的最大总和。根据上述讨论这也是凸的，因此两个上凸函数做 $\max,+$ 卷积可以线性，这样合并时枚举两侧情况再卷积合并，也可以在 $O(n\log n)$ 的复杂度内求出每个 $i$ 的答案。

然后考虑原问题。设 $i$ 个不交区间的最大总和为 $s_i$，则 $g(v)=\max_{i=1}^ns_i-v*i$。这和 $(i,s_i)$ 构成的上凸壳有关，而 $s_i$ 本来就是上凸的，因此非严格凸壳可以保留所有点。

然后可以发现随着 $v$ 从初始的最大子段和开始变小，取 $\max$ 的点只会向后走。可以发现 $\max$ 取某个点的情况为 $v$ 在 $s_{i+1}-s_i$ 到 $s_i-s_{i-1}$ 的这一段。而答案可以看成 $v*m$ 减去对于每个小于最大子段和的 $x$，使得最大子段和小于等于 $x$ 的 $k$ 数量，即 $\max(k-g(v)+1,0)$。因此对于一段内部，这个值是一个一次函数，可以直接求和。这样即可得到答案。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 100500
#define ll long long
#define mod 998244353
int n,v[N],as;
ll k;
vector<ll> conv(vector<ll> a,vector<ll> b)
{
	int sa=a.size(),sb=b.size(),l1=0,l2=0;
	vector<ll> as;as.push_back(a[0]+b[0]);
	for(int i=1;i<sa+sb-1;i++)
	{
		int fg=0;
		if(l1+1==sa)fg=1;else if(l2+1!=sb&&a[l1+1]-a[l1]<b[l2+1]-b[l2])fg=1;
		if(fg)l2++;else l1++;
		as.push_back(a[l1]+b[l2]);
	}
	return as;
}
struct sth{vector<ll> dp[2][2];};
sth operator +(sth a,sth b)
{
	sth as;
	for(int s=0;s<2;s++)for(int t=0;t<2;t++)
	{
		vector<ll> s1=conv(a.dp[s][0],b.dp[0][t]),s2=conv(a.dp[s][1],b.dp[1][t]);
		for(int i=1;i<s2.size();i++)s1[i-1]=max(s1[i-1],s2[i]);
		as.dp[s][t]=s1;
	}
	return as;
}
sth solve(int l,int r)
{
	if(l==r)
	{
		sth as;
		for(int s=0;s<2;s++)for(int t=0;t<2;t++)as.dp[s][t].push_back(s+t?-1e16:0),as.dp[s][t].push_back(v[l]);
		return as;
	}
	int mid=(l+r)>>1;
	return solve(l,mid)+solve(mid+1,r);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	vector<ll> sr=solve(1,n).dp[0][0];
	scanf("%lld",&k);
	as=1ll*(k%mod)*(sr[1]%mod)%mod;
	k++;
	for(int i=1;i<=n;i++)
	{
		ll li;
		if(i<n)li=sr[i]*2-sr[i-1]-sr[i+1];
		else li=1e16;
		ll tp=k/i;
		if(tp>li)tp=li;
		as=(as+mod-1ll*(k-i+k-tp*i)%mod*(tp%mod)%mod*(mod+1)/2%mod)%mod;k-=i*tp;
		if(li>tp)break;
	}
	printf("%d\n",as);
}
```



##### auoj459 有向图

###### Problem

给一个 $n$ 个点 $m$ 条边有边权的有向图，保证 $1$ 能到达所有点。对于每个 $i$ 求解如下问题：

找到从 $1$ 到 $i$ 的两条边不相交路径，使得两条路径上的边权和最小。求最小值或输出无解。

$n\leq 10^5,m\leq 3\times 10^5$

$3s,1024MB$

###### Sol

问题可以看成求从 $1$ 到 $i$ 的流量为 $2$ 的最小费用流。但显然直接 `spfa-mamf` 是完全不行的。

考虑 `primal-dual` 的做法，先求出 $1$ 到每个点的最短路 $d_i$，然后将一条边 $(f\to t,c)$ 的边权变为 $c-d_t+d_f$。这样之后第一次流只会走边权为 $0$ 的边，将这条路径反向之后，$1$ 到 $x$ 当前的最短路就是第二条路径的增广路，答案为 $2d_x$ 加上当前的最短路。

考虑图的一个最短路树，可以钦定第一次流的是树上的路径。而有根树上将根到一个点的路径反向相当于将这个点变为根，因此可以看成如下问题：

有一棵树，树边边权为 $0$。还有一些有向边，边权非负（`primal-dual` 的性质）。对于每个 $i$ 求出，将树上边按照以 $i$ 为根，外向定向后，图中 $1$ 到 $i$ 的最短路。



图中没有非负边，因此对于任意一个 $i$ 的问题，根到任意点的距离不存在负数。因此对于根的某一个 $u$ 的子树内的问题，根到其它子树都有边权为 $0$ 的边，因此到这些子树内的距离都是 $0$。而根到 $u$ 子树内的路径一定是到其它子树中，再通过非树边进入 $u$ 子树。因为其它子树距离都是 $0$，因此进入后不会再出来，否则不优。那么可以找到所有其它子树连向这个子树的边，设一条边为 $(f\to t,c)$，则在所有 $u$ 子树内的问题中，都可以看成有一条根到 $t$ 边权为 $c$ 的路径。

然后考虑这个子树内部的情况。上面讨论了所有经过外部子树的情况，因此剩下的路径只能在子树内完成。考虑找到当前和根距离最近的点 $x$。则因为所有边权非负，根到达 $x$ 的路径不可能更短。接下来考虑以 $x$ 为根，再做上面的操作。记 $di_x$ 为在之前的操作后，$1$ 到 $x$ 的最小距离，则找 $x$ 相当于找 $di_x$ 最小的点，接下来的操作相当于找到所有 $x$ 的一个子树到另外一个子树的非树边 $(f\to t,c)$，然后将 $di_t$ 和 $di_x+c$ 取 $\min$，接下来对于每个子树分治处理。

为了简便，也可以在整体上每次找最小的点进行操作，因此可以看成如下过程：

初始 $di_1=0$，剩下的点 $di_x=+\infty$。

每次操作找到没有被访问过的点中 $di$ 最小的，设这个点为 $u$。则当前处理的为 $u$ 开始不经过已访问点能到达的连通块。

在连通块中找到起点终点不在 $u$ 的同一个子树内的非树边 $(f\to t,c)$，将 $di_t$ 变为 $\min(di_t,di_u+c)$。

重复上述过程，直到访问所有点。



每次找 $u$ 可以优先队列，但不能每次直接找连通块，否则复杂度不对。考虑将一条非树边看成树上的路径 $(f,t)$，则找到的非树边一定满足 $(f,t)$ 路径经过 $u$。而一条路径如果被找到一次，则接下来会被已访问点分隔，从而接下来不会考虑这条边。因此可以看成如下操作：

每次找到 $u$，处理所有路径 $(f,t)$ 经过了 $u$ 的边 $f\to t$，并删去这些路径。

考虑记录 `dfs` 序，枚举路径的一侧在 $u$ 的哪个儿子中，相当于求 `dfs` 序在一个区间内的点中，有没有一个点的一条非树边连向 `dfs` 序不在这个区间内的点。

那么只需要对于每个点维护与其有非树边相连的点中 `dfs` 序最大最小的，然后 `dfs` 序维护区间 $\max,\min$，找大于或者小于某个值的位置即可。这样就可以 $O(\log n)$ 删除，查找。

复杂度 $O((n+m)\log m)$



还有一种更加标准的做法，被称为 Suurballe`s algorithm。注意到分治的时候如果不访问一个子树，仍然可以通过其它子树连出去的边访问到所有跨过 $u$ 的非树边。考虑将度数作为权值，分治时跳过权值最大的子树，则这样和启发式分裂一样，复杂度 $O((n+m)\log m)$。这样常数可能更小，尤其是 $m$ 更大的情况下。~~但我没有仔细分析过如何正确找到子树权值~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
#include<set>
using namespace std;
#define N 100500
#define ll long long
int n,m,head[N],cnt,is[N*3];
struct edge{int t,next,v;}ed[N*3];
void adde(int f,int t,int v)
{
	ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;
}
ll dis[N],rs[N],vis[N],fr[N],s[N*3][3];
void dij()
{
	for(int i=2;i<=n;i++)dis[i]=1e18;
	priority_queue<pair<ll,int> > qu;qu.push(make_pair(0,1));
	while(!qu.empty())
	{
		int u=qu.top().second;qu.pop();
		if(vis[u])continue;vis[u]=1;
		for(int i=head[u];i;i=ed[i].next)if(dis[ed[i].t]>dis[u]+ed[i].v)
		dis[ed[i].t]=dis[u]+ed[i].v,qu.push(make_pair(-dis[ed[i].t],ed[i].t)),fr[ed[i].t]=i;
	}
}
int id[N],rb[N],ct;
set<pair<int,int> > sr[N];
void dfs(int u,int fa)
{
	id[u]=++ct;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
	rb[u]=ct;
}
struct segt{
	struct node{int l,r,mn,mx;}e[N*4];
	void pushup(int x)
	{
		e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);
		e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);
	}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r)
		{
			if(sr[l].empty())e[x].mn=1e9,e[x].mx=-1e9;
			else e[x].mn=(*sr[l].begin()).first,e[x].mx=(*sr[l].rbegin()).first;
			return;
		}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int s)
	{
		if(e[x].l==e[x].r)
		{
			int l=e[x].l;
			if(sr[l].empty())e[x].mn=1e9,e[x].mx=-1e9;
			else e[x].mn=(*sr[l].begin()).first,e[x].mx=(*sr[l].rbegin()).first;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		modify(x<<1|(mid<s),s);
		pushup(x);
	}
	int querymn(int x,int l,int r,int v)
	{
		if(e[x].mn>=v||e[x].l>r||e[x].r<l)return -1;
		if(e[x].l==e[x].r)return (*sr[e[x].l].begin()).second;
		int as=querymn(x<<1,l,r,v);
		if(as==-1)as=querymn(x<<1|1,l,r,v);
		return as;
	}
	int querymx(int x,int l,int r,int v)
	{
		if(e[x].mx<=v||e[x].l>r||e[x].r<l)return -1;
		if(e[x].l==e[x].r)return (*sr[e[x].l].rbegin()).second;
		int as=querymx(x<<1,l,r,v);
		if(as==-1)as=querymx(x<<1|1,l,r,v);
		return as;
	}
}tr;
priority_queue<pair<ll,int> > q2;
void doit(int x,int l,int r)
{
	while(1)
	{
		int fr=tr.querymn(1,l,r,l);
		if(fr==-1)fr=tr.querymx(1,l,r,r);
		if(fr==-1)return;
		if(rs[s[fr][1]]>rs[x]+s[fr][2])
		rs[s[fr][1]]=rs[x]+s[fr][2],q2.push(make_pair(-rs[s[fr][1]],s[fr][1]));
		sr[id[s[fr][0]]].erase(make_pair(id[s[fr][1]],fr));
		sr[id[s[fr][1]]].erase(make_pair(id[s[fr][0]],fr));
		tr.modify(1,id[s[fr][0]]);tr.modify(1,id[s[fr][1]]);
	}
}
void dij2()
{
	for(int i=1;i<=n;i++)rs[i]=1e18,vis[i]=0;
	rs[1]=0;
	q2.push(make_pair(0,1));
	while(!q2.empty())
	{
		int u=q2.top().second;q2.pop();
		if(vis[u])continue;vis[u]=1;
		doit(u,id[u],id[u]);
		for(int i=head[u];i;i=ed[i].next)if(id[ed[i].t]>id[u])
		doit(u,id[ed[i].t],rb[ed[i].t]);
	}
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]),adde(s[i][0],s[i][1],s[i][2]);
	dij();
	for(int i=1;i<=n;i++)head[i]=0;cnt=0;
	for(int i=2;i<=n;i++)is[fr[i]]=1,adde(s[fr[i]][0],s[fr[i]][1],0),adde(s[fr[i]][1],s[fr[i]][0],0);
	dfs(1,0);
	for(int i=1;i<=m;i++)if(!is[i])
	{
		s[i][2]+=dis[s[i][0]]-dis[s[i][1]];
		sr[id[s[i][0]]].insert(make_pair(id[s[i][1]],i));
		sr[id[s[i][1]]].insert(make_pair(id[s[i][0]],i));
	}
	tr.build(1,1,n);
	dij2();
	for(int i=2;i<=n;i++)
	printf("%lld\n",rs[i]>1e17?-1:dis[i]*2+rs[i]);
}
```



#### SCOI2020模拟23

##### auoj460 无根树

###### Problem

给一个 $n$ 阶排列 $p$，求有多少棵有标号无根树满足如果 $(i,j)$ 有边，则 $(p_i,p_j)$ 有边。答案模 $998244353$

多组数据

$T\leq 100,\sum n\leq 5\times 10^5$

$1s,1024MB$

###### Sol

这里排列可以看成置换，考虑将置换看成若干个环。

考虑在长度为 $a$ 的环与长度为 $b$ 的环之间的边的情况，由上述限制可以发现最后如果两侧两个点编号模 $\gcd(a,b)$ 同余，则它们之间连了边。因此为了不出现环，只能有 $a=\gcd(a,b)$ 或 $b=\gcd(a,b)$。

可以发现此时一个小的环与一个大的环相连时，不会改变小的环上的连通性。因此每个长度大于 $1$ 的环必须向更小的环连边或者内部连边。

如果一个长度为 $a$ 的环连向了两个长度小于它的环，只考虑这三个环，则一共有 $2a$ 条边，最多有 $a+\frac a2+\frac a2=2a$ 个点，因此得到的一定不是树。从而一个环最多连向一个长度小于它的环。

如果环内部有连边，那么内部会形成若干环，只有当 $a$ 为偶数，且是相对的点连边时不会出现环。

如果 $a=2$ 环上的点就连通了，否则此时环上的点不连通，需要连向一个更小的环。考虑这两个环之间的边以及大环内的边，边数为 $a+\frac a2$，点数不超过 $a+\frac a2$，因此一定不是树。

因此这种情况只在 $a=2$ 时可能出现，接下来分情况讨论：

如果有长度为 $1$ 的环，那么长度为 $2$ 的环一定会通过若干个长度为 $2$ 的环，然后与长度为 $1$ 的环相连，因此此时长度为 $2$ 的环不能内部连边。

此时可以发现对于任意一组方案，每个长度大于 $1$ 的环要么连到一个长度相同的环上，要么自己连到一个长度更小的环上。否则这个环上的点不能连通。

对于长度为 $a$ 的环，只考虑连向小于等于它的环的边，一定是先将一些这种长度的环相连连成树，连两个环有 $a$ 的方案数，然后每个连通块选一个点向更小的环的连边。

设长度为 $i$ 的环有 $c_i$ 个，那么一个长度为 $i$ 的环连向一个长度更小的环的方案数显然为 $\sum_{j|i,j<i}c_j*j$。这可以直接 $O(n\log n)$ 求出。

这时相当于 $c_i+1$ 个点，其中有一个关键点，连向关键点的边权为 $v_i$，其余边边权为 $i$，求所有生成树的边权乘积和。在 `prufer` 序中，每个元素出现了 $d_u-1$ 次，其中 $d_u$ 为这个点的度数。对于一棵树，考虑以关键点为根，除去则每个点的 $d_i-1$ 表示这个点向儿子的连边，根会少算一条边。而只有根向儿子的连边边权为 $v_i$。因此可以看成 `prufer` 序中每出现一个关键点就乘 $v_i$，否则乘上 $i$，最后额外乘一个 $v_i$。因此总的方案数为 $v_i(v_i+c_i*i)^{c_i-1}$，最后将每种长度的方案数相乘即可。

对于没有出现长度为 $1$ 的环的情况，如果有长度为 $2$ 的环，可以发现最后一定是将所有 $2$ 的环连通，再选择一个环做内部连边，因此先做之前的方案，再乘上 $2$ 的环个数即可。

如果长度为 $1,2$ 的环都没有出现，则考虑长度最小的一个环，这个环上的这些点与其它环相连不能改变连通性（也不能内部连边），因此这种情况无解。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 500050
#define mod 998244353
int T,n,t[N],vis[N],ct[N],vl[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int solve()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&t[i]),vis[i]=ct[i]=vl[i]=0;
	vis[2]=ct[2]=vl[2]=0;
	for(int i=1;i<=n;i++)if(!vis[i])
	{
		int st=i,c1=0;
		while(!vis[st])vis[st]=1,c1++,st=t[st];
		ct[c1]++;
	}
	if(!ct[1]&&!ct[2])return 0;
	int as=1;
	for(int i=1;i<=n;i++)if(ct[i])
	for(int j=i*2;j<=n;j+=i)vl[j]=(vl[j]+1ll*ct[i]*i)%mod;
	for(int i=3;i<=n;i++)if(ct[i])as=1ll*as*pw((vl[i]+1ll*i*ct[i])%mod,ct[i]-1)%mod*vl[i]%mod%mod;
	if(!ct[1])as=1ll*as*ct[2]%mod*pw(ct[2],ct[2]==1?1:ct[2]-2)%mod*pw(2,ct[2]-1)%mod;
	else as=1ll*as*pw(ct[1],ct[1]==1?1:ct[1]-2)%mod*(ct[2]?1ll*pw(vl[2]+2*ct[2],ct[2]-1)*vl[2]%mod:1)%mod;
	return as;
}
int main()
{
	scanf("%d",&T);while(T--)printf("%d\n",solve());
}
```



##### auoj461 数列

###### Problem

给定长度为 $n$ 的两个序列 $a,b$，你可以进行任意次操作，每次将一个 $a_i$ 减少 $1$，费用为 $1$。

对于 $i$，如果满足 $a_i\geq \max_{j=1}^ia_j$，即 $a_i$ 为前缀最大值，那么有 $b_i$ 的收益。

求收益减去费用的最大值。

$n\leq 5\times 10^5,a_i,b_i\leq 10^9$

$3s,1024MB$

###### Sol

设 $dp_{i,j}$ 表示前 $i$ 个数，当前前面数的 $\max$ 不超过 $j$，前面的最大收益。显然只有当某个 $a_i=j$ 的 $j$ 是有用的，否则可以调整到更优（增加所有等于这个值的位置的权值）。这样可以看成只有 $n$ 种权值。

那么转移有：

$$
dp_{i,j}=\max_{k\leq j}((dp_{i-1,k}+b_i-a_i+k)[k\leq a_i],dp_{i-1,k}[k>a_i])
$$

这相当于给前缀加上一段一次函数，然后取前缀 $\max$。

因为取 $\max$ 后的 `dp` 递增，前半段加的一次函数递增，因此加之后前半段递增。只需要找到后面一段小于 $a_i$ 位置修改后的值的区间，做区间赋值即可。

考虑线段树维护区间加一次函数和区间赋值操作，然后线段树上二分即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 505000
#define ll long long
int n,v1[N],v2[N],v3[N],id[N];
struct segt{
	struct node{int l,r,lz1;ll lz2,lz3,mx;}e[N*4];
	void pushdown(int x){if(e[x].lz3>-1e18)e[x<<1].mx=e[x<<1|1].mx=e[x<<1].lz3=e[x<<1|1].lz3=e[x].lz3,e[x].lz3=-2e18,e[x<<1].lz1=e[x<<1].lz2=e[x<<1|1].lz1=e[x<<1|1].lz2=0;e[x<<1].lz1+=e[x].lz1;e[x<<1].lz2+=e[x].lz2;e[x<<1].mx+=e[x].lz2+1ll*v3[e[x<<1].r]*e[x].lz1;e[x<<1|1].lz1+=e[x].lz1;e[x<<1|1].lz2+=e[x].lz2;e[x<<1|1].mx+=e[x].lz2+1ll*v3[e[x<<1|1].r]*e[x].lz1;e[x].lz1=e[x].lz2=0;}
	void pushup(int x){e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void modify1(int x,int l,int r,int v1,int v2)
	{
		if(e[x].l==l&&e[x].r==r){e[x].lz1+=v1;e[x].lz2+=v2;e[x].mx+=v1*v3[e[x].r]+v2;return;}
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify1(x<<1,l,r,v1,v2);
		else if(mid<l)modify1(x<<1|1,l,r,v1,v2);
		else modify1(x<<1,l,mid,v1,v2),modify1(x<<1|1,mid+1,r,v1,v2);
		pushup(x);
	}
	void modify2(int x,int l,int r,ll v)
	{
		if(e[x].l==l&&e[x].r==r){e[x].lz1=e[x].lz2=0;e[x].lz3=e[x].mx=v;return;}
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify2(x<<1,l,r,v);
		else if(mid<l)modify2(x<<1|1,l,r,v);
		else modify2(x<<1,l,mid,v),modify2(x<<1|1,mid+1,r,v);
		pushup(x);
	}
	int query(int x,ll v){if(e[x].l==e[x].r)return e[x].mx<v?e[x].l:0;pushdown(x);if(e[x<<1].mx>=v)return query(x<<1,v);else return max(e[x<<1].r,query(x<<1|1,v));}
	ll que(int x,int v){if(e[x].l==e[x].r)return e[x].mx;pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=v)return que(x<<1,v);else return que(x<<1|1,v);}
}tr;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v1[i]),v3[i]=v1[i];
	for(int i=1;i<=n;i++)scanf("%d",&v2[i]);
	sort(v3+1,v3+n+1);
	for(int i=1;i<=n;i++)id[i]=lower_bound(v3+1,v3+n+1,v1[i])-v3;
	tr.build(1,1,n);
	for(int i=1;i<=n;i++)
	{
		ll tp=tr.que(1,id[i])+v2[i];
		int st=tr.query(1,tp);
		tr.modify1(1,1,id[i],1,v2[i]-v3[id[i]]);
		if(id[i]<st)tr.modify2(1,id[i]+1,st,tp);
	}
	printf("%lld\n",tr.que(1,n));
}
```



##### auoj462 排列

###### Problem

给定一个 $n$ 阶排列 $p$，排列从 $0$ 开始标号。你可以对排列进行操作，一次操作由如下几步组成：

1. 选择 $p$ 的一个子序列 $s$。
2. 将 $s$ 中所有奇数按顺序组成序列 $s_1$，所有偶数按顺序组成 $s_2$。
3. 选择 $x\in \{0,1\}$，如果 $x=0$，则令 $t=s_1+s_2$，否则令 $t=s_2+s_1$。
4. 将 $t$ 按照 $s$ 中位置填回序列。即对于每个 $1\leq i\leq |s|$，设 $s$ 的第 $i$ 个元素是从 $p$ 的第 $a_i$ 个位置选出来的，则现在将 $p$ 的第 $a_i$ 个位置变为 $t_i$。

你需要通过不超过 $30$ 次操作，将排列排序。输出任意一组方案。

多组数据

$T\leq 10,n\leq 1.5\times 10^4$

$2s,1024MB$

###### Sol

操作可以看成选出一些数，将它们按照最后一位排序再放回去。因此如果最后一位是有序的，则情况更加简单，有更多的操作空间。

考虑一次操作将所有偶数放到前 $\lceil\frac n2\rceil$ 个位置，奇数放到后面的位置。考虑找到当前前 $\lceil\frac n2\rceil$ 个位置中所有的奇数，和后面位置中所有的偶数，显然这两部分数数量相同，因此考虑选出这些数，进行一次 $x=0$ 的操作，这样就满足了要求。

考虑在满足这个条件的情况下，要使得最后能快速变成 $(0,1,\cdots,p)$，则排列应有的形式。考虑将偶数中小于 $\lceil\frac n2\rceil$ 的放在正确的位置上，剩下的从小到大填左侧剩下的位置，奇数中大于等于 $\lceil\frac n2\rceil$ 的放在正确的位置上，剩下的从小到大填右侧的位置。接下来选出所有不在正确位置上的数，做一次 $x=1$ 的操作，则所有属于右侧的偶数会按顺序填到右侧，左侧同理，这样就排好了序。

因此只需要通过操作使得两侧分别达到某个顺序。可以看成两个部分分别有一个排列，需要通过中间的操作将两个排列同时排序。可以发现此时两侧的排列已经与上面的操作没有关系。

操作次数为 $O(\log_2 n)$。考虑基数排序，第 $i$ 轮将两侧排列的二进制从低到高第 $i$ 位排好序，同时保证这一位相同的按照之前的顺序排序，最后就可以将两侧同时排序。可以发现上面的操作能很好地保证相同的按照之前排序的性质。

此时可以看成两侧各有一些 $0,1$。如果左侧的 $1$ 个数和右侧的 $0$ 个数相同，则存在如下的方式：

1. 选择左侧所有 $1$ 和右侧所有 $0$，做 $x=1$ 的操作让右侧（奇数）部分来到左侧，左侧来到右侧。因为两部分数量相同，此时右侧所有 $0$ 都在原先的边界左侧，左侧所有 $1$ 都在原先的边界右侧。
2. 考虑选择所有数做一次 $x=0$ 操作，这样所有数回到对应侧，可以发现对于左侧部分，所有 $0$ 在 $1$ 前面，两部分内部满足原先顺序，因此这样达到了要求。

注意为了求后面的操作方式，需要模拟基数排序的过程或者模拟操作。这样的操作次数是 $2\lceil\log_2 n\rceil$，满足要求。但不一定左侧 $1$ 个数和右侧 $0$ 个数相同。

考虑给两侧重标号，使得对于每一位都满足这个条件。注意到两侧数的数量最多相差 $1$，考虑如下方式：

考虑从上往下建立 `01-trie`，最后用 `01-trie` 得到编号。在向下扩展一层的时候，需要将每个点当前的大小分成两部分。考虑将偶数直接平均分，对于一个奇数 $x$，可以分 $\lceil\frac x2\rceil,\lfloor\frac x2\rfloor$ 两部分，两部分大小相差 $1$。考虑第一个奇数让左侧更大，第二个奇数让右侧更大，如此交替。这样可以使得一层左侧有 $\lceil\frac x2\rceil$ 个，右侧有 $\lfloor\frac x2\rfloor$ 个，其中 $x$ 为总共的数个数。反过来操作可以交换两侧的情况。

如果两侧数的数量相同，则直接左侧不变，右侧反过来操作即可。对于不同的情况，考虑 $2x+1,2x$ 的情况，可以让两侧都正向做，则个数为 $x+1,x,x,x$，中间两个相等。如果为 $2x,2x-1$，可以发现也是两侧都正向做。这样就完成了编号过程。

复杂度 $O(Tn\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 17925
int T,n,m,p[N],rp[N],ct,f1[N];
vector<int> fr[N];
void doit(vector<int> t,int k)
{
	fr[++ct]=t;f1[ct]=k;
	vector<int> s1,s2;
	for(int i=0;i<t.size();i++)
	{
		if(rp[t[i]]&1)s2.push_back(rp[t[i]]);
		else s1.push_back(rp[t[i]]);
	}
	if(k)swap(s1,s2);
	for(int i=0;i<s2.size();i++)s1.push_back(s2[i]);
	for(int i=0;i<t.size();i++)rp[t[i]]=s1[i];
}
int sp[N],rs[N];
vector<int> getid(int n,int f,int d)
{
	vector<pair<int,int> > st;st.push_back(make_pair(0,n));
	for(int i=0;i<d;i++)
	{
		int nw=f;
		vector<pair<int,int> > s2;
		for(int i=0;i<st.size();i++)
		{
			int id=st[i].first,ct=st[i].second;
			s2.push_back(make_pair(id*2,ct/2));
			s2.push_back(make_pair(id*2+1,ct/2));
			if(ct&1)s2[s2.size()-2+nw].second++,nw^=1;
		}
		st=s2;
	}
	vector<int> as;
	for(int i=0;i<st.size();i++)if(st[i].second)as.push_back(st[i].first);
	return as;
}
void solve()
{
	scanf("%d%*d",&n);m=(n+1)/2;ct=0;
	for(int i=0;i<n;i++)scanf("%d",&p[i]),rp[i]=p[i];
	vector<int> s1;
	for(int i=0;i<n;i++)if((i>=m)^(p[i]&1))s1.push_back(i);
	doit(s1,0);
	for(int i=0;i<n;i++)p[i]=rp[i];
	for(int i=0;i<n;i++)sp[i]=i;
	int lb=1,rb=m+(m&1);
	while(rb<n)swap(sp[lb],sp[rb]),lb+=2,rb+=2;
	for(int i=0;i<n;i++)rs[sp[i]]=i;
	for(int i=0;i<n;i++)p[i]=rs[p[i]];
	int d=0,t1=1;
	while(t1<m)d++,t1<<=1;
	vector<int> id1=getid(m,m*2==n,d),id2=getid(n-m,0,d);
	for(int i=0;i<m;i++)p[i]=id1[p[i]];
	for(int i=m;i<n;i++)p[i]=id2[p[i]-m];
	for(int t=0;t<d;t++)
	{
		vector<int> s1,s4;
		for(int i=0;i<n;i++)if((i>=m)^((p[i]>>t)&1))s1.push_back(i);
		doit(s1,1);
		for(int i=0;i<n;i++)s4.push_back(i);
		doit(s4,0);
		vector<int> tp;
		for(int i=0;i<m;i++)if(!((p[i]>>t)&1))tp.push_back(p[i]);
		for(int i=0;i<m;i++)if(((p[i]>>t)&1))tp.push_back(p[i]);
		for(int i=m;i<n;i++)if(!((p[i]>>t)&1))tp.push_back(p[i]);
		for(int i=m;i<n;i++)if(((p[i]>>t)&1))tp.push_back(p[i]);
		for(int i=0;i<n;i++)p[i]=tp[i];
	}
	vector<int> s2;
	for(int i=0;i<n;i++)if(sp[i]!=i)s2.push_back(i);
	doit(s2,1);
	printf("%d\n",ct);
	for(int i=1;i<=ct;i++)
	{
		printf("%d ",f1[i]);
		vector<int> is(n);
		for(int j=0;j<fr[i].size();j++)is[fr[i][j]]=1;
		for(int j=0;j<n;j++)printf("%d",is[j]);
		printf("\n");
	}
}
int main()
{
	scanf("%d",&T);
	while(T--)solve();
}
```



#### NOI2020 模拟测试1

~~数据强度：tan(pi/2)~~

##### auoj466 工厂

###### Problem

有 $m$ 种零件，你每天可以生产每种零件各一个。

如果当前有 $m$ 个种类两两不同，且生产日期两两不同的零件，则你可以将它们组合成一个产品。

有 $n$ 个订单，第 $i$ 个要求在第 $a_i$ 天前得到一个产品。

你希望违约的订单数量尽量小，输出这个最小值。

$n\leq 200,m\leq 100,a_i\leq 1000$

$2s,256MB$

###### Sol

~~因为数据强度问题，随便搞都能过，比如认为只有 $a_i<m$ 会违约~~

考虑二分答案，变为一个判定性问题。

考虑无视种类不同的限制，只要求生产日期两两不同，则如果原问题有解，则这个问题有解。

考虑这个问题中的一个方案，将每天生产的零件看成二分图左侧的点，每个产品看成二分图右侧的点，产品选择了一天生产的零件相当于连边，则这个二分图有如下性质：

1. 左侧每个点度数不超过 $m$
2. 右侧每个点度数等于 $m$

因为满足生产日期不同，图中没有重边。而种类不同的条件可以看成需要将二分图的边分成 $m$ 个匹配。

可以发现，一定可以将边分成 $m$ 个匹配。考虑归纳，只需要找到一个匹配。使得删去匹配后两侧点度数都小于等于 $m-1$。

考虑找到右侧所有度数为 $m$ 的点向左侧点的匹配。此时任意 $k$ 个右侧这样的点向左连边数量为 $mk$，而左侧点度数不超过 $m$，因此右侧这样的 $k$ 个点至少和左侧 $k$ 个点相邻。由 `Hall` 定理，一定存在一个匹配包含右侧所有度数为 $m$ 的点。同理存在一个匹配包含左侧度数为 $m$ 的点。

考虑将两个匹配拼接起来，可以得到若干个链和偶环。长度为奇数的链和偶环都可以构造一个匹配使得所有点都在匹配内，但偶数链可能有一个点不在匹配内。但可以发现上面两个匹配中，每个度数为 $m$ 的点都对应了一个匹配点，因此链上两个端点一定有一个不是度数为 $m$ 的点（考虑链的方向），因此可以合并匹配使得剩下的点在匹配中。

因此存在一个匹配，包含两侧所有度数为 $m$ 的点。因此结论成立。



所以可以不考虑种类不同的限制。二分答案后从前往后考虑需要处理的每个订单，处理到当前订单时需要选择 $m$ 个日期，从这 $m$ 个日期剩余的零件中分别选择一个。记录每个日期的零件剩余的数量，则显然的贪心是选择剩余数量最大的 $m$ 个日期，这样显然是对的。

一种实现方式是记录每种剩余数量的个数，这样加入若干天可以 $O(1)$，选择最大的 $m$ 个减一可以 $O(m)$，复杂度 $O(nm\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 205
int n,m,k,v[N],ci[N],cr[N];
bool chk(int x)
{
	int ls=0;
	for(int i=1;i<=m;i++)ci[i]=0;
	for(int i=x;i<=n;i++)
	{
		ci[m]+=v[i]-ls;ls=v[i];
		int tp=m;
		for(int j=m;j>=1;j--)
		{
			cr[j]=ci[j];
			if(cr[j]>tp)cr[j]=tp;
			tp-=cr[j];
		}
		if(tp)return 0;
		for(int j=1;j<=m;j++)ci[j-1]+=cr[j],ci[j]-=cr[j],cr[j]=0;
	}
	return 1;
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	int lb=1,rb=n,as=n+1;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(chk(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%d\n",k*(as-1));
}
```



##### auoj467 未来程序·不改

###### Problem

有 $26$ 个函数，标记为 `A` 到 `Z`。还有 $26$ 种运算，标记为 `a` 到 `z`。

有 $n$ 条调用关系，每条调用关系属于一个函数，它包含一个调用列表，列表长度为 $d_i$，列表中每个元素是运算或者函数。

运行一个函数时，如果不存在这个函数的调用关系则结束函数运行，否则随机选择一个它的调用列表，按顺序执行列表元素。

程序运行时，使用栈的方式维护调用的过程。具体来说，初始栈中只有初始调用的函数（初始调用的函数给定），接下来重复如下过程直到栈为空：

取出栈顶的元素，如果元素为运算则执行运算。如果元素为函数，则运行函数。

运行函数的方式为，如果存在这个函数的调用关系，则随机选择一个调用列表，将这个列表逆序插入到栈顶。

给 $q$ 组询问，每次给定两个函数 $s,t$，询问如果函数 $s$ 会被运行，则函数 $t$ 是否一定会在有限时间内被运行。

$n\leq 100,d_i\leq 48,q\leq 10$

$1s,256MB$

###### Sol

原 `std` 是假的，强烈谴责。

~~这直接导致写类似 `std` 的做法或者乱搞随便过，但正解写挂一点就20pts~~

设函数数量为 $m$，这里 $m=26$。

考虑什么情况下这个命题不成立，这当且仅当存在一种运行程序的方式能在有限步之内运行 $s$，但不能在有限步之内运行到 $t$。

此时有两种情况，第一种是可以在有限步内结束，且经过了 $s$ 不经过 $t$。第二种是不经过 $t$，经过 $s$ 且循环。因此接下来的所有情况都认为不能经过 $t$。

首先考虑停止的情况，记 $f_i$ 表示运行函数 $i$ 后，能不能在不经过 $t$ 的情况下结束，$g_i$ 表示能不能在不经过 $t$ 且经过 $s$ 的情况下结束。

考虑如何转移。如果 $i$ 没有调用关系显然 $f_i=1$，如果 $i$ 存在一个调用关系，使得调用关系中不存在 $t$ 且每一个调用函数 $x$ 都满足 $f_x=1$，则 $f_i=1$。

对于 $g_i$，如果 $i=s,f_i=1$ 则 $g_i=1$，否则如果在上面的转移中，存在一个调用函数 $x$ 满足 $g_x=1$，则也有 $g_i=1$。

但这个转移是循环的，考虑一直做，每一轮用之前的 $f,g$ 计算下一轮的 $f,g$，直到 $f,g$ 不再改变，可以发现剩余的点的转移必定循环，因此这样得到的是真的 $g$。这样最多做 $2m$ 轮，复杂度 $O(nmd)$。

如果 $g_s=1$，则这样就找到了第一种情况的路径。



考虑第二种情况，考虑如果必须循环，则路径可以怎么走。

如果当前调用到一个函数 $u$，在调用列表中调用到 $x$ 时，接下来不再从 $x$ 出来，则看成从 $u$ 向 $x$ 连边。可以发现合法的循环方式一定可以表示成这样的路径，且路径最后成为一个环。

考虑如何处理边。对于一个调用列表，能向第 $i$ 个函数连边当且仅当满足如下条件：

1. 这个函数以及之前的调用函数中不存在 $t$。
2. 之前的调用函数都满足 $f_x=1$（否则之前就无法结束）

可以 $O(nd)$ 得到这个图。那么经过 $s$ 的方案有两种情况：

1. 上面的路径中经过 $s$。
2. 在路径之外的部分经过了 $s$。即到达一个函数 $u$ 时，路径上下一步走 $x$，而 $x$ 前面的一个函数可以经过 $s$ 并返回。

考虑这个有向图上进行一次 `floyd`，可以对于每一对 $(i,j)$ 得到 $i$ 是否能到达 $j$，且 $i=j$ 时可以得到是否存在环经过 $i$。则对于第一种情况，只需要判断起点能否走到 $s$，然后枚举环上一个点 $t$，判断 $s$ 能否到 $t$，$t$ 能否到 $t$ 即可。

对于第二种情况，枚举调用列表，从前往后考虑每个函数，记录之前是否存在一个函数可以经过 $s$ 并返回（即 $g_x=1$），然后考虑走当前出边，像上面一样判断环即可。

单次询问复杂度 $O(nmd+m^3)$，总复杂度 $O(qnmd+qm^3)$

可以将所有 $t$ 相同的询问一起做，复杂度可以降至 $O(nm^2d+m^4+qm)$，但这里没有必要。

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 105
char s[6];
int s1,n,m=26,q,a,b,fr[N],is[N],is2[N];
vector<int> sr[N];
int si[N][N],fg[N];
bool query(int s,int t)
{
	for(int i=1;i<=m;i++)is[i]=1,fg[i]=0;
	for(int i=1;i<=m;i++)for(int j=1;j<=m;j++)si[i][j]=0;
	for(int i=1;i<=n;i++)is[fr[i]]=0;
	is[t]=0;if(is[s])is[s]=3;
	while(1)
	{
		for(int i=1;i<=m;i++)is2[i]=is[i];
		for(int i=1;i<=n;i++)
		{
			int fg=1;
			for(int j=0;j<sr[i].size();j++)
			if(!is[sr[i][j]]){fg=0;break;}
			else fg|=is[sr[i][j]];
			is2[fr[i]]|=fg;
		}
		is2[t]=0;if(is2[s])is2[s]=3;
		int fg=0;
		for(int i=1;i<=m;i++)if(is[i]!=is2[i])fg=1,is[i]=is2[i];
		if(!fg)break;
	}
	if(is2[s1]==3)return 0;
	for(int i=1;i<=n;i++)if(fr[i]!=t)
	for(int j=0;j<sr[i].size();j++)
	{
		if(sr[i][j]==t)break;
		si[fr[i]][sr[i][j]]=1;
		if(!is[sr[i][j]])break;
	}
	for(int k=1;k<=m;k++)for(int i=1;i<=m;i++)for(int j=1;j<=m;j++)si[i][j]|=si[i][k]&si[k][j];
	for(int i=1;i<=m;i++)if(i==s&&(si[s1][i]||i==s1))
	for(int j=1;j<=m;j++)if(si[i][j]&&si[j][j])return 0;
	for(int i=1;i<=n;i++)if(si[s1][fr[i]])
	{
		int fg=0;
		for(int j=0;j<sr[i].size();j++)
		{
			if(sr[i][j]==t)break;
			if(fg)
			{
				int nt=sr[i][j];
				for(int k=1;k<=m;k++)if(si[nt][k]&&si[k][k])return 0;
			}
			if(is[sr[i][j]]==3)fg=1;
			if(!is[sr[i][j]])break;
		}
	}
	return 1;
}
int main()
{
	scanf("%s",s+1);s1=s[1]-'A'+1;
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%s",s+1);fr[i]=s[1]-'A'+1;
		while(getchar()!='\n')
		{
			scanf("%s",s+1);
			if(s[1]<'A'||s[1]>'Z')continue;
			sr[i].push_back(s[1]-'A'+1);
		}
	}
	scanf("%d",&q);
	while(q--)
	{
		scanf("%s",s+1);a=s[1]-'A'+1;
		scanf("%s",s+1);b=s[1]-'A'+1;
		printf("%s\n",query(a,b)?"Yes":"No");
	}
}
```



##### auoj468 扫除积雪

###### Problem

有 $n$ 个位置排成一列，每个位置有雪的高度 $h_i$。

你有一个清理积雪的工具，它有一个功率。对于一个 $k$，如果你从 $k$ 开始清理积雪，则过程如下：

首先清理 $k$ 位置的积雪，将功率设为 $h_k$，这次操作不用时间。接下来每次操作可以选择如下操作的一种，直到所有位置被清理：

假设当前已经清理的区间为 $[l,r]$，则操作有三种：

1. 如果 $l>1$ 且当前功率大于等于 $h_{l-1}$，则可以花费 $1$ 的时间清理 $l-1$ 位置。
2. 如果 $r<n$ 且当前功率大于等于 $h_{r+1}$，则可以花费 $1$ 的时间清理 $r+1$ 位置。
3. 可以花费 $c$ 的时间，将当前功率变为 $\min(h_{l-1},h_{r+1})$。

这里认为 $h_0=h_{n+1}=+\infty$。

设 $f_i$ 表示从 $i$ 开始清理需要的最少时间。有 $q$ 次操作，每次操作为如下两种类型之一：

1. 交换两个相邻的 $h_i$。
2. 给定 $l,r$，询问 $[l,r]$ 内所有位置的 $f_i$ 之和。

$n\leq 10^5,q\leq 2\times 10^5$

$2s,128MB$

###### Sol

考虑清雪的过程，显然只有能变大功率时会做操作 $3$，且可以发现此时必须做操作 $3$，因此容易确定操作的方式，即能做前两种操作就做，否则进行操作 $3$。

可以发现前两种操作的次数和一定为 $n-1$，这部分代价固定，只需要考虑增大功率的次数。可以发现这个次数为从这个位置向两侧向上的单调栈中元素的种类数，但这样不好计算。

考虑计算每种权值的贡献。可以发现一个权值 $x$ 能对一个位置 $i$ 做贡献，当且仅当满足如下条件：

存在一个 $h_j=x$ 的位置 $j$，使得 $i,j$ 之间所有位置的 $h$ 小于等于 $j$，且 $h_j>h_i$。

如果去掉最后一个限制，则每个位置会多算自己的 $h_i$ 一次，且只会多算这一次。因此考虑去掉这个限制，最后每个位置答案减去 $c$ 即可。

然后考虑这样的贡献形式。对于一个位置 $i$，设 $l_i$ 为 $i$ 向左第一个大于 $h_i$ 的位置，$r_i$ 为右侧第一个大于 $h_i$ 的位置，则在上面的贡献形式中，$i$ 可以对 $[l_i+1,r_i-1]$ 中的位置做贡献。

但还有一个问题是一种权值可以由多个合法的 $j$。可以发现如果有两个合法的 $j$，则这两个 $j$ 中间的位置高度不超过这两个位置的高度，因此这两个 $j$ 对应的区间相同。因此考虑对于这种情况只留下一个 $j$，即只保留满足 $[i+1,r_i-1]$ 中间不存在一个 $j$ 使得 $h_j=h_i$ 的 $i$，这些 $i$ 贡献即为所有的贡献。



考虑修改对区间的影响。设修改位置为 $i,i+1$。不妨设 $h_i<h_{i+1}$，大于的情况可以看成倒过来操作，等于的情况可以跳过操作。

则对于大于等于 $h_{i+1}$ 或者小于 $h_i$ 的权值，交换这两个元素不影响这些权值向两侧找第一个大于它的位置的过程（因为两个位置同时大于这个权值或者同时小于等于这个权值），这些区间不会改变。

对于权值等于 $h_i$ 的区间，因为 $i,i+1$ 进行了交换，可能本来左侧有这个区间，右侧没有，交换后变为了右侧有左侧没有。考虑找到 $i-1$ 以及左侧第一个大于 $h_i$ 的位置 $l$，如果 $[l+1,i-1]$ 中有一个等于 $h_i$ 的位置，则交换过去后左侧的区间仍然存在，只是端点改变了 $1$ 的距离，否则左侧区间不再存在，可以看成 $[l+1,i]$ 区间减一。对于右侧，考虑找 $i+2$ 向右第一个大于 $h_i$ 的位置 $r$，然后类似讨论即可（是否存在等于 $h_i$ 的位置）。因为这段区间中一定没有大于 $h_i$ 的位置，因此这里判断只需要查区间 $\max$。

然后考虑权值在 $[h_i+1,h_{i+1}-1]$ 的区间。考虑左侧的情况，如果有一个权值 $v\in [h_i+1,h_{i+1}-1]$ 满足如下条件：

存在一个位置 $j<i$，使得 $h_j=v$ 且 $[j+1,i-1]$ 区间中任意 $h$ 小于 $v$。

则这个权值的区间之前右端点在 $i$，交换过去后右端点在 $i-1$，因此会使得 $i$ 位置的次数减少 $1$。可以发现权值在这个区间内且在左侧的区间中，只有满足这个条件的区间能够覆盖到 $i$，因此左侧只有这些改变。

因此 $i$ 位置减少的次数相当于从 $i$ 向左，依次找以 $i$ 结束的后缀最大值，权值大于等于 $h_{i+1}$ 时停止，这样找到的在范围内的后缀最大值数量。

对于右侧类似，即 $i+1$ 位置增加的次数相当于换过去后，从 $i+1$ 向右找从 $i+1$ 开始的前缀最大值，大于等于 $h_{i}$ 时停止，这样找到的数量。 

这样就考虑了所有变化。考虑如何维护。首先对覆盖次数的影响只有区间加以及单点加。可以用 `BIT` 区间加区间求和。

上面的第一部分需要从某个位置向一个方向找第一个大于某个值的位置，可以线段树上维护 $\max$，然后线段树上二分。

第二部分相当于需要询问区间单调栈长度，因此两个方向都做线段树维护单调栈即可。值的上界问题可以在线段树上询问的内部处理，也可以先用上面的询问找到需要的区间端点直接处理掉。

复杂度 $O((n+q)\log^2 n)$

线段树维护单调栈的做法：

考虑从左向右的情况。记 $mx_x$ 为线段树节点内最大值，记 $v_x$ 为如下值：

设前面已经有的 $\max$ 为 $mx_{ls_x}$，接下来的部分为 $rs_x$ 的区间，在这个区间中还有多少前缀 $\max$。

如果得到了这个式子，考虑询问，询问的形式为，如果当前前面的 $\max$ 为 $m_1$，当前考虑 $x$ 节点的区间，则区间内还有多少前缀 $\max$。

如果 $m_1\geq mx_{ls_x}$，则左儿子不会有贡献，可以直接跳过，只询问右侧儿子。否则，考虑询问左侧儿子，这之后前面的 $\max$ 为 $mx_{ls_x}$，因此右侧部分直接使用 $v_x$ 的结果即可。这样询问复杂度 $O(\log n)$。

然后考虑维护 $v_x$，从下往上求，这样直接做一次询问即可得到 $v_x$。因此修改一个点会做 $O(\log n)$ 次询问，复杂度 $O(\log^2 n)$。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define ll long long
int n,v[N],a,l,r,k;
struct BIT{
	ll tr[N];
	void add(int x,int k){for(int i=x;i<=n;i+=i&-i)tr[i]+=k;}
	ll que(int x){ll as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}t1,t2;
ll calc(int l,int r)
{
	ll s1=t1.que(r)+t2.que(r)*r,s2=t1.que(l-1)+t2.que(l-1)*(l-1);
	return s1-s2;
}
void radd(int l,int r,int v)
{
	if(l<1)l=1;if(l>r)return;
	t2.add(l,v);t1.add(l,(1-l)*v);
	t2.add(r,-v);t1.add(r,r*v);
}
struct sth{int a,b;};
struct segt{
	struct node{int l,r,mx,ls,rs;}e[N*4];
	int quel(int x,int v)
	{
		if(e[x].l==e[x].r)return e[x].mx>v;
		if(e[x<<1|1].mx>=v)return quel(x<<1|1,v)+e[x].ls;
		return quel(x<<1,v);
	}
	int quer(int x,int v)
	{
		if(e[x].l==e[x].r)return e[x].mx>v;
		if(e[x<<1].mx>=v)return quer(x<<1,v)+e[x].rs;
		return quer(x<<1|1,v);
	}
	void pushup(int x)
	{
		e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);
		e[x].ls=quel(x<<1,e[x<<1|1].mx);
		e[x].rs=quer(x<<1|1,e[x<<1].mx);
	}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].mx=v[l];return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int s)
	{
		if(e[x].l==e[x].r){e[x].mx=v[e[x].l];return;}
		int mid=(e[x].l+e[x].r)>>1;
		modify(x<<1|(mid<s),s);
		pushup(x);
	}
	int que(int x,int l,int r,int v,int f)//0:left 1:right
	{
		if(e[x].l>r||e[x].r<l)return -1;
		if(e[x].mx<v)return -1;
		if(e[x].l==e[x].r)return e[x].l;
		int as=que(x<<1|(!f),l,r,v,f);
		if(as==-1)as=que(x<<1|f,l,r,v,f);
		return as;
	}
	sth queryl(int x,int l,int r,int nw,int mx)
	{
		if(e[x].mx<=nw||e[x].l>r||e[x].r<l||nw>mx)return (sth){nw,0};
		if(e[x].l>=l&&e[x].r<=r&&e[x].mx<mx)return (sth){e[x].mx,quel(x,nw)};
		if(e[x].l==e[x].r)return (sth){mx+1,0};
		sth v1=queryl(x<<1|1,l,r,nw,mx),v2=queryl(x<<1,l,r,v1.a,mx);
		v2.b+=v1.b;return v2;
	}
	sth queryr(int x,int l,int r,int nw,int mx)
	{
		if(e[x].mx<=nw||e[x].l>r||e[x].r<l||nw>mx)return (sth){nw,0};
		if(e[x].l>=l&&e[x].r<=r&&e[x].mx<mx)return (sth){e[x].mx,quer(x,nw)};
		if(e[x].l==e[x].r)return (sth){mx+1,0};
		sth v1=queryr(x<<1,l,r,nw,mx),v2=queryr(x<<1|1,l,r,v1.a,mx);
		v2.b+=v1.b;return v2;
	}
}tr;
void init()
{
	tr.build(1,0,n+1);
	for(int i=1;i<=n;i++)
	{
		int ls=tr.que(1,0,i-1,v[i],0),rs=tr.que(1,i+1,n+1,v[i]+1,1);
		if(v[ls]!=v[i])radd(ls+1,rs-1,1);
	}
}
void modify(int x)
{
	if(v[x]==v[x+1])return;
	int fg=v[x]<v[x+1],rv=calc(x+fg,x+fg),r1=rv+1,las=calc(x+(!fg),x+(!fg)),vl=v[x+(!fg)],vr=v[x+fg];
	int ls=tr.que(1,0,x-1,vl,0),rs=tr.que(1,x+2,n+1,vl,1);
	if(v[ls]!=vl)radd(ls+1,x-1,fg?-1:1);
	if(v[rs]!=vl)radd(x+2,rs-1,fg?1:-1);
	if(fg)r1+=tr.queryr(1,x+2,n+1,vl,vr).b,t1.add(x,rv-las),t1.add(x+1,r1-rv);
	else r1+=tr.queryl(1,0,x-1,vl,vr).b,t1.add(x+1,rv-las),t1.add(x,r1-rv);
	v[x]^=v[x+1]^=v[x]^=v[x+1];
	tr.modify(1,x);tr.modify(1,x+1);
}
int main()
{
	scanf("%d%d",&n,&k);v[0]=v[n+1]=1e9+1;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	init();
	while(~scanf("%d",&a))
	if(a==1)scanf("%d",&l),modify(l);
	else scanf("%d%d",&l,&r),printf("%lld\n",calc(l,r)*k+1ll*(r-l+1)*(n-1-k));
}
```



#### NOI2020 模拟测试2

##### auoj471 咕

###### Problem

有一个长度为 $n$ 的随机排列 $p$，你一开始在排列的位置 $1$。在位置 $i$ 上，你可以知道 $p_i$ 是否是排列的一个前缀最小值，接下来你可以选择如下操作：

1. 在这个位置上结束游戏，如果这个位置的值为 $1$ 则获胜。
2. 走到下一个位置。

你希望最大化获胜的概率，求最优策略下获胜的概率，模 $998244353$

多组数据

$T\leq 10^5,n\leq 10^6$

$1s,256MB$

###### Sol

显然只可能在某一个是前缀最小值的位置选择停止。考虑当前在一个前缀最小值 $p_i$ 的情况，因为只关心最小值的位置，同时得到的信息也只关于前缀最小值，因此可以发现之前位置的顺序对后面没有影响，因此可以不考虑前面的信息。

则需要考虑的只有当前位置 $i$，因此策略可以看成对于每个 $i$ 决定如果当前是前缀最小值，那么应该继续还是结束。

考虑从后往前第一个选择继续的位置，则这个位置满足向后找到下一个前缀最小就停止时获胜的概率大于等于在当前位置直接停止的概率。

这里计算概率需要计算条件概率，即满足当前位置是前缀最小值的情况的概率。考虑分别计算满足当前位置是前缀最小值，且选择继续/停止时能获胜的方案数，因为两个条件概率的条件相同，因此比较它们只需要比较两个方案数的大小。

考虑停止时获胜的方案数，显然这是 $(n-1)!$。

考虑继续且获胜的方案数，设下一个前缀最小值的位置为 $j$，则这种情况的数量为使得 $j$ 是前缀最小值，且 $i$ 是 $[1,j-1]$ 中的最小值的排列数量。这种情况的数量为 $(n-1)!*\frac 1{j-1}$。因此总的数量为 $n!*\sum_{j=i}^{n-1}\frac 1j$。

因此从后往前，第一个选择继续的位置为第一个满足 $\sum_{j=i}^{n-1}\frac 1j\geq 1$ 的位置 $i$。可以发现在 $n\geq 2$ 时，取 $i=1$，则左侧和一定大于等于 $1$，因此一定存在这样的位置。但还有 $n=1$ 的情况，因此需要特判 $n=1$。

继续向前考虑，前面的位置都满足向后找到下一个前缀就停止比在这里停止更优，因此这些位置一定选择继续。

因此找到第一个选择继续的位置 $x$，则策略为超过 $x$ 后，找到第一个前缀最小值就停止。

考虑计算获胜的概率，枚举停止的位置 $y$，则在 $[x+1,y-1]$ 部分不停止相当于 $[1,y-1]$ 的最小值在 $[1,x]$ 中，这个概率为 $\frac x{y-1}$。因此答案为：

$$
\frac 1n*\sum_{j=x}^{n-1}\frac xj
$$

考虑预处理 $\frac 1i$ 的前缀和，这样二分即可找到 $x$，同时预处理 $\frac 1i$ 取模后的和即可得到答案。

复杂度 $O((n+T)\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1040000
#define mod 998244353
int T,n,fr[N],ifr[N],inv[N],s2[N];
double su2[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	fr[0]=ifr[0]=1;
	for(int i=1;i<=1e6;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1000000]=pw(fr[1000000],mod-2);
	for(int i=999999;i>0;i--)ifr[i]=1ll*ifr[i+1]*(i+1)%mod;
	for(int i=1;i<=1000000;i++)inv[i]=1ll*fr[i-1]*ifr[i]%mod;
	for(int i=1;i<=1e6;i++)s2[i]=(s2[i-1]+inv[i])%mod,su2[i]=su2[i-1]+1.0/i;
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d",&n);
		if(n==1){printf("1\n");continue;}
		int lb=1,rb=n-1,as=0;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(su2[n-1]-su2[mid]>=1)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		printf("%d\n",1ll*inv[n]*(as+1)%mod*(mod+s2[n-1]-s2[as])%mod);
	}
}
```



##### auoj472 凋朱颜

###### Problem

给定 $n,m$，有一个 $m^n\times m^n$ 的矩阵，行列下标从 $0$ 开始。

定义 $f(a,b)$ 为 $a,b$ 在 $m$ 进制下的不进位加法的结果，矩阵中 $(a,b)$ 处的数为 $f(a,b)$，求有多少条矩阵上的路径满足

1. 路径上相邻两个位置在矩阵上相邻
2. 路径上相邻两个位置后一个位置的数比前一个大 $1$

答案模 $998244353$

$n,m\leq 10^9,m\geq 2$

$1s,256MB$

###### Sol

对于一条路径，如果它经过的位置的值的最低 $k$ 位上有不同的值，而更高的位都相同，则定义这样的路径为 $k$ 级路径。

对于第 $d$ 级路径，因为它的高位全部相同，可以只考虑 $m^d\times m^d$ 中的路径数量，然后将这个数量乘上 $m^{2n-2d}$。

将 $m^d\times m^d$ 的部分按照最高位分成 $m\times m$ 的块，每一块大小为 $m^{d-1}\times m^{d-1}$，这一级路径一定至少经过了 $2$ 个块。

对于 $m>2$ 的情况，容易发现在不同块之间的移动只能向下或者向右。

考虑不同块之间的移动。如果要从一个块走出去，则最后一步所在位置的值在 $m$ 进制下低 $d-1$ 位的值必须是 $m^{d-1}-1$（因为走到下一个格子上一位会发生变化），但如果一位上的和模 $m$ 余 $m-1$，则只有可能是 $0+(m-1)$。因此边界上满足的位置只有两个角 $(0,m^{d-1}-1),(m^{d-1}-1,0)$。再考虑下一步到达的位置，从一个角的 $(0,m^{d-1}-1)$ 出发只能到达右侧的 $(0,0)$ 或者上面的 $(m^{d-1}-1,m^{d-1}-1)$，而 $m>2$ 时只有第一个可能是合法的。因此从另外一个块走过来时，只能到 $(0,0)$。从而如果路径开头结尾都不在这个块，则路径为 $(0,0)$ 到 $(0,m^{d-1}-1),(m^{d-1}-1,0)$。而这一块内部需要在 $m^{d-1}$ 位不变的情况下低位从 $0$ 增加到 $m^{d-1}-1$，因此只能走 $m^{d-1}-1$ 步，可以发现唯一的合法方式为全部向右或者全部向下。

因此在路径中间的一块上，可以走到下面或者右侧，方案数都是 $1$。而向下或者向右走时，只要不越过这一位 $a+b=m$ 的线，$m^{d-1}$ 位上一定增加。因此可以分成左上角部分和右下角部分，每一部分内不同块间向下向右都是合法的。

那么这一级上的路径可以分为三部分：

1. 从起点所在的块内走到块角上
2. 在不同块间向下向右走
3. 从一个块左上角开始，在块内结束不走出这个块

考虑第一部分，考虑走到左下角 $(m^{d-1},0)$ 的方案数，可以看成从这个位置开始倒着走的方案数。假设当前走到 $(x,0)(x>0)$，下一步需要让当前位置的值为 $x-1$。走到 $(x-1,0)$ 显然合法，$(x+1,0)$ 显然不合法。考虑 $(x,1)$ 位置，如果 $x$ 的最低位不是 $m-1$，则这个位置值是 $x+1$，不合法。如果最低位是 $m-1$，则当前位置值为 $x-m+1$，但 $m>2$，因此这种情况也不合法。从而到达左下角只能全部向下，方案数为 $m^{d-1}$。可以看成有 $m^{d-1}$ 的系数，然后再在这个块上决定一次方向（向下还是向右）。

考虑第三部分，相当于在一个 $m^{d-1}\times m^{d-1}$ 的部分上从左上角开始走的方案数。此时当前层为 $m^{d-2}$ 位，在这一层上的方案可以看成选择一个块结束，走到这个块的左上角，然后进入低一层的路径。

考虑一层上的路径数，由上面的分析只要不到达 $a+b=m$，可以任意向下和向右，这相当于最多走 $m-1$ 步，可以发现走 $m-1$ 步一定不会离开矩形，因此任意的 $m-1$ 个方向都是合法的，因此这一层的方案数为 $\sum_{i=0}^{m-1}2^i=2^m-1$。

而每一层情况相同，因此这部分方案数为 $(2^m-1)^{d-1}$。

考虑第二部分，根据上面的分析，可以分成左上和右下部分。

对于左上部分，考虑一个 $x+y=i$ 的位置作为起点，则它接下来只要走的步数不超过 $m-i$ 就是合法的，但必须走至少一步。而 $x+y=i$ 的点有 $i+1$ 个，因此这部分的路径数为 $\sum_{i=0}^m(i+1)*(2^{m-i}-2)$

对于右下部分，考虑倒着看路径，这样变为和左上相同的情况，但三角形大小减少了 $1$，因此这部分为 $\sum_{i=0}^{m-1}(i+1)*(2^{m-1-i}-2)$

考虑把这个东西表示成简单的形式，经过计算可以得到结果为 $3*2^{m+1}-2m^2-4m-6$。

回到原问题，枚举每一层求和，考虑进行化简：

$$
\sum_{d=1}^n(3*2^{m+1}-2m^2-4m-6)*m^{d-1}*(2^m-1)^{d-1}*m^{2n-2d}\\
=(3*2^{m+1}-2m^2-4m-6)m^{2n-2}\sum_{d=1}^n(\frac{2^m-1}m)^d\\
=(3*2^{m+1}-2m^2-4m-6)*((\frac{2^m-1}m)^{n+1}-1)*\frac 1{\frac{2^m-1}m-1}
$$

然后就可以快速算了，复杂度 $O(\log n+\log mod)$

对于 $m=2$ 的情况可以类似地推一遍。但还有一种直接的方式，可以发现上面的结果中只有一个指数和 $n$ 有关，这是线性递推的形式，因此猜测 $m=2$ 也是线性递推。

设这种情况答案为 $a_n$，打表可得 $a_n=13a_{n-1}-36a_{n-2},a_n=\frac{4*9^n+4^n}5$，直接求即可。

复杂度 $O(\log n+\log mod)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define mod 998244353
int n,m;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&m);
	if(m>2)
	{
		int as=pw(m,n*2),s1=1;
		int f2=0,f1=0;
		f2=pw(2,m)-1;
		f1=(6ll*pw(2,m)%mod-2ll*m*m%mod-4ll*m%mod-6+3ll*mod)%mod;
		int v1=1ll*f1*s1%mod*pw(1ll*m*m%mod,n-1)%mod,v2=1ll*f2*pw(m,mod-2)%mod;
		if(v2==1)as=(as+1ll*n*v1)%mod;
		else as=(as+1ll*v1*(pw(v2,n)-1)%mod*pw(v2-1,mod-2)%mod+mod)%mod;
		printf("%d\n",as);
	}
	else printf("%d\n",(4ll*pw(9,n)+pw(4,n))*pw(5,mod-2)%mod);
}
```



##### auoj473 简单题

###### Problem

给两张 $n$ 个点的有边权无向图，定义 $dis_1(i,j)$ 为第一张图上两点所有路径上的边权最大值的最小值，$dis_2(i,j)$ 为第二张图上的这个值。

求 $\sum_{i<j}dis_1(i,j)dis_2(i,j)$，模 $998244353$

$n\leq 2\times 10^5,m\leq 5 \times 10^5$

$3s,1024MB$

###### Sol

考虑对两个图建 `kruskal` 重构树，那么两个点在这个图上的 `dis` 即为对应重构树上 `lca` 的权值。

考虑枚举第一棵树上的 `lca` 算贡献。对于第一棵重构树上的每个点，维护这个点子树内所有叶子（非重构树新增的点）的集合。重构树上每个点只有两个儿子，而 `lca` 在这个点上的情况即为两个点分别在两个子树的情况，因此考虑在合并这两个儿子的集合时计算贡献，即对于两个儿子的集合 $S_1,S_2$，计算：

$$
\sum_{i\in S_1}\sum_{j\in S_2}dis_2(i,j)
$$

考虑启发式合并，枚举小的集合中的每一个元素，询问大的集合中每个元素与它在第二棵树上的 `lca` 权值和。

这相当于需要维护若干个集合，求一个点与集合内每一个点在第二棵树上的 `lca` 的权值和，并支持合并两个集合。

考虑如何处理 `lca` 权值。考虑差分，对于一个点，记它的贡献为 $val_u-val_{fa_u}$，其中 $val$ 为重构树点权，同时每个点有一个价值，初始价值全部为 $0$。考虑两个点 $a,b$，首先将 $a$ 到根上的每个点的价值加上这个点的贡献 $val_u-val_{fa_u}$，考虑此时 $b$ 到根上每个点的价值和，可以发现这正好等于两个点 `lca` 处的点权。多个点和一个点算贡献时类似，只需要对每个点做一次修改，然后再处理询问点即可。

因此一个集合的贡献系数可以看成对每个点维护一个 $a_i$，初始 $a_i=0$，集合内加入一个点的操作为将它到根的路径上的所有 $a_i$ 加一，询问为询问一个点到根的路径上所有点的 $a_i(val_i-val_{fa_i})$ 之和。需要支持合并两个集合的贡献系数。

链上的情况可以线段树解决，树上用树剖拆成 `dfs` 序区间即可。考虑每个集合维护一个动态开点线段树，区间加使用标记永久化，合并使用线段树合并。

复杂度 $O(n\log^3 n+m\log n)$，但有两个 $\log$ 是启发式合并和树剖，因此常数很小，可以通过。

有一些复杂度更优的方式，比如重构树上边分治，但是咕咕咕了。

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 400500
#define mod 998244353
#define ll long long
int n,m,a,b,c,as,id[N];
struct ed2{int f,t,v;friend bool operator <(ed2 a,ed2 b){return a.v<b.v;}};
struct edge{int t,next;};
struct kruskaltree{
	int f[N],fa[N],vl[N],id[N],tp[N],sz[N],sn[N],tid[N],head[N],cnt,ct2,ct;
	ll su[N];
	edge ed[N*2];
	ed2 e[N*3];
	int finds(int x){return f[x]==x?x:f[x]=finds(f[x]);}
	void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
	void dfs1(int u,int f){fa[u]=f;sz[u]=1;for(int i=head[u];i;i=ed[i].next)dfs1(ed[i].t,u),sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u],sz[u]+=sz[ed[i].t];}
	void dfs2(int u,int v){id[u]=++ct;tp[u]=v;tid[ct]=u;if(sn[u])dfs2(sn[u],v);for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=sn[u])dfs2(ed[i].t,ed[i].t);}
	void solve()
	{
		sort(e+1,e+m+1);ct2=n;
		for(int i=1;i<=n;i++)f[i]=i;
		for(int i=1;i<=m;i++)if(finds(e[i].f)!=finds(e[i].t))
		{
			int st=++ct2;vl[st]=e[i].v;
			adde(st,finds(e[i].f));adde(st,finds(e[i].t));
			f[st]=f[finds(e[i].f)]=f[finds(e[i].t)]=st;
		}
		dfs1(ct2,0);dfs2(ct2,ct2);
		for(int i=1;i<=ct2;i++)su[id[i]]=vl[i]-vl[fa[i]];
		for(int i=1;i<=ct2;i++)su[i]+=su[i-1];
	}
}tr1,tr2;
#define M 44005000
int rt[N],ct;
set<int> fu[N];
int ch[M][2];
int vl[M],lz[M];
void ins(int x,int l,int r,int l1,int r1)
{
	if(l==l1&&r==r1){lz[x]++;return;}
	int mid=(l+r)>>1;
	vl[x]=(1ll*vl[x]+tr2.su[r1]-tr2.su[l1-1]%mod+mod)%mod;
	if(mid>=r1){if(!ch[x][0])ch[x][0]=++ct;ins(ch[x][0],l,mid,l1,r1);}
	else if(mid<l1){if(!ch[x][1])ch[x][1]=++ct;ins(ch[x][1],mid+1,r,l1,r1);}
	else
	{
		{if(!ch[x][0])ch[x][0]=++ct;ins(ch[x][0],l,mid,l1,mid);}
		{if(!ch[x][1])ch[x][1]=++ct;ins(ch[x][1],mid+1,r,mid+1,r1);}
	}
}
int merge(int x,int y)
{
	if(!x||!y)return x+y;
	vl[x]=(vl[x]+vl[y])%mod;lz[x]+=lz[y];
	ch[x][0]=merge(ch[x][0],ch[y][0]);
	ch[x][1]=merge(ch[x][1],ch[y][1]);
	return x;
}
int query(int x,int l,int r,int l1,int r1)
{
	if(!x)return 0;
	int tp=1ll*lz[x]*(tr2.su[r1]%mod-tr2.su[l1-1]%mod+mod)%mod;
	if(l==l1&&r==r1)return (tp+vl[x])%mod;
	int mid=(l+r)>>1;
	if(mid>=r1)return (tp+query(ch[x][0],l,mid,l1,r1))%mod;
	else if(mid<l1)return (tp+query(ch[x][1],mid+1,r,l1,r1))%mod;
	else return (1ll*tp+query(ch[x][0],l,mid,l1,mid)+query(ch[x][1],mid+1,r,mid+1,r1))%mod;
}
void modify(int x,int y)
{
	while(y)
	{
		ins(rt[x],1,n*2,tr2.id[tr2.tp[y]],tr2.id[y]);
		y=tr2.fa[tr2.tp[y]];
	}
}
int que2(int x,int y)
{
	int as=0;
	while(y)
	{
		as=(as+query(rt[x],1,n*2,tr2.id[tr2.tp[y]],tr2.id[y]))%mod;
		y=tr2.fa[tr2.tp[y]];
	}
	return as;
}
void doit(int x)
{
	rt[x]=++ct;
	if(x<=n){modify(x,x);fu[x].insert(x);id[x]=x;return;}
	int ls=tr1.ed[tr1.head[x]].t,rs=tr1.ed[tr1.ed[tr1.head[x]].next].t;
	doit(ls);doit(rs);
	if(fu[id[ls]].size()<fu[id[rs]].size())id[ls]^=id[rs]^=id[ls]^=id[rs];
	for(set<int>::iterator it=fu[id[rs]].begin();it!=fu[id[rs]].end();it++)
	{
		int tp=*it;
		as=(as+1ll*tr1.vl[x]*que2(id[ls],tp))%mod;
		fu[id[ls]].insert(tp);
	}
	id[x]=id[ls];rt[id[x]]=merge(rt[id[ls]],rt[id[rs]]);
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d%d",&a,&b,&c),tr1.e[i]=(ed2){a,b,c};
	for(int i=1;i<=m;i++)scanf("%d%d%d",&a,&b,&c),tr2.e[i]=(ed2){a,b,c};
	tr1.solve();tr2.solve();doit(n*2-1);printf("%d\n",as);
}
```



#### NOI2020 模拟测试3

##### auoj474 气象学

###### Problem

三维空间中有一个点 $(x_1,y_1,z_1)$ 作为光源，有一个球，这个球的球心坐标为 $(x_2,y_2,z_2)$，半径为 $r$。有一个平面 $ax+by+cz+d=0$，求平面被球遮挡的阴影面积，相对误差不超过 $10^{-6}$。

保证阴影存在且面积有限，球，光源，平面不相交。多组数据

$T\leq 30$，所有坐标绝对值不超过 $10^4$

$1s,1024MB$

###### Sol

考虑通过旋转平移平面，将平面转成类似 $x=0$ 的形式。

显然交换两维坐标，答案不变，可以交换到 $b\neq 0$，然后在这一维上将所有物体整体平移使得 $d=0$。

然后考虑旋转平面，首先考虑 $x,y$ 两维，以 $x=y=0$ 为中心进行旋转，将所有图形在 $x,y$ 平面上逆时针旋转 $(\frac a{\sqrt{a^2+b^2}},\frac b{\sqrt{a^2+b^2}})$ 的角度，可以使得平面方程变为满足 $b=0$。旋转的一种简单实现方式是看成复数相乘。

然后考虑 $y,z$ 两维，进行相同的操作，可以将平面变为 $x=0$ 的形式。

由于平面为 $x=0$，可以将光源和球整体在 $y,z$ 上平移。然后可以将球心坐标平移到 $y=z=0$，再以 $y=z=0$ 为轴进行旋转，让光源坐标变为 $z=0$。

注意到阴影形状相当于圆锥的一个切面，因此一定是一个椭圆。

点与球相切的所有点一定构成一个圆。考虑 $z=0$ 的截面部分，这部分是点和一个圆相切，可以算出切点位置和中心位置。

通过几何观察可以发现椭圆的长轴的两个端点一定是光源与圆上 $x=0$ 的点的连线与平面的交点，可以直接求出

对于短轴的两个端点，它们的连线垂直于长轴，因此只需要确定圆上一条与上面的两个点的连线垂直的线，它与圆的两个交点就是对应的两个点。考虑通过确定切点在相切得到的圆上，如果将圆看成 $x^2+y^2=1$，则这两个点的 $x$ 坐标位置（从 $-1$ 到 $1$），通过 $z=0$ 的平面部分可以得到这个位置的 $x$ 坐标，通过圆上的几何观察可以得到 $z$ 坐标，而最后两个投影点只有 $z$ 坐标不同，因此这样求出 $z$ 坐标之差即可得到答案。

椭圆是凸的，可以发现选一条线得到的距离是单峰的，三分即可。这里也可以直接用力算出来。

复杂度 $O(T\log v)$，注意细节

###### Code

下面的代码在旋转部分有一些区别

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
using namespace std;
int T,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11;
double a1,b1,c1,a2,b2,c2,r,a,b,c,d,pi=acos(-1);
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%d%d%d%d%d%d%d%d%d",&f1,&f2,&f3,&f4,&f5,&f6,&f7,&f8,&f9,&f10,&f11);
		a1=f1;b1=f2;c1=f3;a2=f4;b2=f5;c2=f6;r=f7;a=f8;b=f9;c=f10;d=f11;
		if(abs(a)<=0.1)swap(a1,b1),swap(a2,b2),swap(a,b);
		if(abs(a)<=0.1)swap(a1,c1),swap(a2,c2),swap(a,c);
		a1+=d/a;a2+=d/a;d=0;
		if(abs(b)>1e-8)
		{
			double f1=abs(a/b),f2=sqrt(f1*f1+1),ag1=(1+f2*f2-f1*f1)/2/f2;
			double t1=sin(acos(ag1));
			if(b/a<0)t1*=-1;
			double g1=ag1*a1-t1*b1,g2=a1*t1+b1*ag1;b1=g1,a1=g2;
			g1=ag1*a2-t1*b2,g2=a2*t1+b2*ag1;b2=g1,a2=g2;a=(b>0?1:-1)*sqrt(a*a+b*b);
		}
		swap(b1,c1),swap(b2,c2),swap(b,c);
		if(abs(b)>1e-8)
		{
			double f1=abs(a/b),f2=sqrt(f1*f1+1),ag1=(1+f2*f2-f1*f1)/2/f2;
			double t1=sin(acos(ag1));
			if(b/a<0)t1*=-1;
			double g1=ag1*a1-t1*b1,g2=a1*t1+b1*ag1;b1=g1,a1=g2;
			g1=ag1*a2-t1*b2,g2=a2*t1+b2*ag1;b2=g1,a2=g2;
		}
		b=c=0;
		double d2=sqrt((b1-b2)*(b1-b2)+(c1-c2)*(c1-c2));
		b2=c2=c1=0;b1=d2;
		double d1=sqrt(b1*b1+(a1-a2)*(a1-a2));
		double ds1=r*r/d1;
		double p1=a2+(a1-a2)*ds1/d1,p2=b1*ds1/d1,p3=0,r1=sqrt(r*r-ds1*ds1);
		double fu2=-a1+p1,fu1=-p2+b1;
		double ds2=sqrt(fu1*fu1+fu2*fu2);fu1*=r1/ds2;fu2*=r1/ds2;
		double s1=p1+fu1,s2=p2+fu2,v1=(b1-s2)*a1/(a1-s1);
		s1=p1-fu1,s2=p2-fu2;
		double v2=(b1-s2)*a1/(a1-s1);
		v2-=v1;if(v2<0)v2*=-1;v2/=2;
		double lb=-1,rb=1,as=0;
		for(int i=1;i<=100;i++)
		{
			double mid1=(lb*2+rb)/3,mid2=(lb+rb*2)/3;
			double f11=p1+fu1*mid1,f12=p2+fu2*mid1,f21=p1+fu1*mid2,f22=p2+fu2*mid2;
			double as1=a1/(a1-f11)*sqrt(1-mid1*mid1)*r1,as2=a1/(a1-f21)*sqrt(1-mid2*mid2)*r1;
			if(as1>as2)as=as1,rb=mid2;
			else as=as2,lb=mid1;
		}
		printf("%.12lf\n",pi*v2*as);
	}
}
```



##### auoj475 巴塞罗那
  
###### Problem

考虑如下问题：

有 $n$ 堆硬币，每一堆都有 $k$ 枚，其中有一堆硬币比别的硬币重，这一堆硬币的重量相同，其余硬币重量相同，但你不知道两种硬币的重量，重量可以是实数。

你可以进行 $m$ 次操作，有一个天平，每次操作你可以在两侧各放一些硬币，数量任意，但不能超过你有的数量。天平会给出两侧硬币重量的差。

多组询问，每次给出 $m,k$，求最大的 $n$ 使得存在操作方式一定能够分辨出哪一堆硬币是更重的硬币。答案模 $998244353$。

$T\leq 10,m,k\leq 10^5$

$1s,1024MB$

###### Sol

对于一堆硬币，设它在第 $i$ 次操作中，两侧放的硬币的数量差为 $v_i$，则显然有 $v_i\in[-k,k]$。记第 $i$ 堆硬币的序列为 $v^i$。

考虑要求每次操作两侧的硬币数量都相同，这样差值就是额外的重量乘上这次操作重的硬币在两侧放的数量的差。

如果这堆硬币更重，且额外重量为 $x$。则每次称量两侧的重量差为 $v_1^ix,v_2^ix,\cdots,v_k^ix$。

如果能分辨出是哪一堆硬币，则一定是对于每一个可能的重量差的结果，可以唯一确定哪一堆硬币可以得到这个结果。因此不存在两个正数 $x,y$ 以及两堆硬币的上述序列 $v^a,v^b$，使得 $\for 1\leq i\leq k,v_i^ax=v_i^by$。

对于一个序列 $v_{1,\cdots,k}^a$，令 $g_a=\gcd_{i=1}^k|v_i^a|$（如果 $v^a$ 全零，则令 $g_a=1$）。考虑序列 $\frac{v_1^a}{g_a},\cdots,\frac{v_k^a}{g_a}$，如果两个序列 $v^a,v^b$ 使得存在正数 $a,b$ 满足 $\for 1\leq i\leq k,v_i^ax=v_i^by$，可以发现这当且仅当序列 $\frac{v_1^a}{g_a},\cdots,\frac{v_k^a}{g_a}$ 和序列 $\frac{v_1^b}{g_b},\cdots,\frac{v_k^b}{g_b}$ 相等。

因此 $n$ 的一个上界是可能的序列 $\frac{v_1^a}{g_a},\cdots,\frac{v_k^a}{g_a}$ 的数量，而如果 $v_{1,\cdots,m}^a$ 满足每个元素都在 $[-k,k]$ 内，则 $\frac{v_1^a}{g_a},\cdots,\frac{v_k^a}{g_a}$ 也在 $[-k,k]$ 内。而这样的序列 $\frac{v^a}{g_a}$ 满足所有元素为 $0$ 或者所有元素绝对值的 $\gcd$ 为 $1$，可以发现这是充分必要条件。

因此这个上界可以看成长度为 $m$，每个元素在 $[-k,k]$ 内，且满足如下条件的序列 $v$ 个数：

$v$ 满足元素全零或者所有元素的绝对值的 $\gcd$ 为 $1$。

可以发现满足这个条件的序列全部乘 $-1$，得到的仍然是满足条件的序列，因此所有这样的序列的总和每个位置都是 $0$。从而如果按照这个序列构造方案，则每次操作天平两侧的硬币数量都相同。因此这个上界是可以达到的。

考虑计数这个式子，设 $f_i$ 表示 $\gcd$ 等于 $i$ 的序列数量，则答案为 $f_1+1$。

设 $g_i$ 表示满足 $\gcd$ 是 $i$ 的倍数且序列非零的序列数量，则可以发现 $g_i=(2\lfloor\frac ki\rfloor+1)^m-1$。

然后有 $g_i=\sum_{i|j}f_j$，根据莫比乌斯反演有 $f_i=\sum_{i|j}\mu(\frac ji)g_j$，那么可以 $O(k+\sqrt k\log m)$ 计算，也可以直接容斥计算，复杂度 $O(k\log k+k\log m)$。

当然这里也可以对 $j$ 数论分块，然后相当于求 $\mu$ 的前缀和，然后可以杜教筛。

复杂度 $O(Tk(\log k+\log m))$ 或者 $O(Tk^{\frac 23})$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define mod 998244353
int T,n,k,f[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&k);
		for(int i=1;i<=k;i++)f[i]=pw(k/i*2+1,n)-1;
		for(int i=k;i>=1;i--)for(int j=i*2;j<=k;j+=i)f[i]=(f[i]-f[j]+mod)%mod;
		printf("%d\n",f[1]+1);
	}
}
```



##### auoj476 重映射

###### Problem

定义积性函数 $f$ 满足 $f(p^a)=2^a$。

给定 $n$，求 $\sum_{i=1}^nf(i)$，答案对给定大质数 $p$ 取模。

$n\leq 10^{14},p\leq 10^9$

$5s,1024MB$

###### Sol

考虑 $f'=f*\mu$，那么有 $f'(p^a)=2^{a-1}(a>0),f'(1)=1$

考虑 $h=f'*\mu$，那么有 $h(p^a)=2^{a-2}(a>1),h(p)=0,h(1)=1$

这样 $h(p)=0$，因此 $h$ 于是只有所有的 `powerful number` 处有值，`powerful number` 定义为每个出现的质因子出现至少两次的数。

根据反演，$f=h*1*1$。那么可以发现 $h(i)$ 对 $\sum_{i=1}^nf(i)$ 的贡献次数为 $\sum_{1\leq a,b}[abi\leq n]$

设 $g(n)=\sum_{1\leq a,b}[ab\leq n]$，那么 $g(n)=\sum_{i=1}^n\lfloor\frac ni\rfloor$，上面的贡献次数为 $g(\lfloor\frac ni\rfloor)$。

考虑 $O(\sqrt n)$ 算这个东西。显然 $a,b$ 中一个数小于等于 $\sqrt n$，枚举每一个 $i\leq \sqrt n$，考虑计算 $a,b$ 中有一个为 $i$，另外一个大于等于 $i$ 的方案数。首先如果一个等于 $i$，则方案数有 $\lfloor\frac ni\rfloor$。而 $a,b$ 都可以等于 $i$，因此需要乘 $2$。然后考虑多算的情况，一种情况是有一个数小于 $i$，另外一个等于 $i$，另外一种情况是两个都等于 $i$，因此需要减去 $2i-1$ 个多算的情况。

因此 $g(n)=\sum_{i=1}^{\sqrt n}(2\lfloor\frac ni\rfloor-2i+1)$。这样比数论分块常数小一倍以上。~~这就是为什么数论分块会T成60~~

`powerful number` 的每个质因子都出现两侧，预处理 $\sqrt n$ 内的质数，暴搜所有 `powerful number`，暴力计算 $h,g$ 计算答案。考虑这样的复杂度，`powerful number` 可以被表示成 $a^2b^3$，因此复杂度不超过 $\sum_{a=1}^n\sum_{b=1}^n\lfloor\sqrt\frac n{a^2 b^3}\rfloor$。只考虑 $a$ 时这个东西是 $O(\sqrt n\log n)$，只考虑 $b$ 时是 $O(\sqrt n)$，可以发现总复杂度 $O(\sqrt n\log n)$。

可以通过预处理小的 $g(n)$ 和用上面方式代替数论分块来减少常数。

这样的 `powerful number` 筛相当于给 $f$ 卷一个函数，使得 $f*g(p)=0$，然后枚举所有有值的 `powerful number` 位置，相当于计算 $inv_g$ 的前缀和，如果这个前缀和可以数论分块或者其它方式快速计算。`powerful number` 的数量为 $O(\sqrt n)$，因此这样可以更快，一种直接的例子是 $f(p)=0,1,2$ 的情况，取 $\mu$ 或者 $\mu^2$ 即可。

###### Code

可以线性筛算小的 $g$，但是咕了

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 10050000
#define ll long long
ll n;
int as,mod,pr[N],ch[N],ct,f[65],as1[4000050];
void getpr()
{
	int p=sqrt(n);
	for(int i=2;i<=p;i++)
	{
		if(!ch[i])pr[++ct]=i;
		for(int j=1;j<=ct&&1ll*i*pr[j]<=p;j++)
		{
			ch[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
}
int getp(ll f)
{
	ll tp,as=0;
	if(f<4e6)return as1[f];
	for(int i=1;1ll*i*i<=f;i++)tp=f/i,as=(as+2*tp-i*2+1);
	return as%mod;
}
void dfs(ll x,int d,int v,int tp)
{
	if(tp)as=(as+1ll*f[v]*getp(x))%mod;
	if(d>ct||1ll*pr[d]*pr[d]>x)return;
	dfs(x,d+1,v,0);
	x/=1ll*pr[d]*pr[d];
	for(int i=2;x;i++,x/=pr[d])dfs(x,d+1,v+i-2,1);
}
int main()
{
	for(int i=1;i<=4e6;i++)for(int j=i;j<=4e6;j+=i)as1[j]++;
	for(int i=1;i<=4e6;i++)as1[i]+=as1[i-1];
	scanf("%lld%d",&n,&mod);
	f[0]=1;for(int i=1;i<=60;i++)f[i]=2*f[i-1]%mod;
	getpr();dfs(n,1,0,1);printf("%d\n",as);
}
```



#### NOI2020 模拟测试4

##### auoj477 清理通道

###### Problem

有 $T$ 个物品，每个物品有两个属性 $a_i,b_i$。

你有 $n$ 个人，第 $i$ 个人有一个属性 $s_i$，他每个时刻可以拿走一个满足 $a<s_i$ 的物品。

你还有 $m$ 个人，第 $i$ 个人有一个属性 $t_i$，他每个时刻可以拿走一个满足 $b<t_i$ 的物品。

求最少需要多少时刻使得所有物品都被拿走，或输出无法拿走所有物品。

$n,m\leq 5\times 10^4,T\leq 5\times 10^5$

$3s,64MB$

###### Sol

无解当且仅当有一个物品不能被拿走，这容易判断。

考虑二分答案 $as$，根据 `Hall` 定理，存在在 $as$ 时刻中拿走所有物品的方案当且仅当满足如下条件：对于每一个物品的集合 $S$ ，能够拿走集合中至少一个物品的人数乘上 $as$ 大于等于 $|S|$。

对于一个物品的集合，可以发现影响能够拿走集合中至少一个物品的人数的只有这些物品中 $a$ 的最小值和 $b$ 的最小值。因此可以发现合法的条件等价于对于每一对 $(a,b)$，满足 $a_i\geq a,b_i\geq b$ 的物品数量不超过 $as$ 乘上第一类人中 $s_i>a$ 的人数和第二类人中 $t_i>b$ 的人数的和。

这些都可以看成二维平面上的区间加，物品看成给 $(a_i,b_i)$ 的左下角全部 $-1$，人看成给 $x<a$ 或者 $x<b$ 的部分加上 $as$，合法相当于平面上所有位置的值非负，扫描线+线段树维护即可。

复杂度 $O((n+m+T)\log T\log n)$，非常卡常

另一种做法是，二分后考虑贪心，先考虑前 $n$ 个人，按照 $s_i$ 从小到大考虑，每个人优先拿能拿的里面 $b$ 最大的物品。然后再考虑后 $m$ 个人，能拿完就说明可以。可以发现前半部分这样的贪心是最优的。

可以使用 `set` 维护所有物品的 $b$，这样第一部分就容易维护。复杂度与上面相同，但是常数小得多。~~代码又咕了~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 50050
#define M 500500
#define ll long long
int n,m,k,s1[N],s2[N],s[M][2],id[M],fu[M];
bool cmp(int a,int b){return s[a][0]>s[b][0];}
struct segt{
	struct node{int l,r;ll mn,su;}e[N*4];
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].mn=e[x].su=0;
		if(l==r)return;
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	}
	void pushup(int x){e[x].mn=min(e[x<<1].mn+e[x<<1|1].su,e[x<<1|1].mn);e[x].su=e[x<<1].su+e[x<<1|1].su;}
	void modify(int x,int l,ll v)
	{
		if(e[x].l==e[x].r){e[x].mn+=v;e[x].su+=v;return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=l)modify(x<<1,l,v);else modify(x<<1|1,l,v);
		pushup(x);
	}
}tr;
bool check(int v)
{
	tr.build(1,0,k);
	for(int i=1;i<=k;i++)tr.modify(1,i-1,v);
	int st=1;
	for(int i=m;i>=0;i--)
	{
		while(st<=n&&s[id[st]][0]>=s1[i])tr.modify(1,fu[id[st]]-1,-1),st++;
		if(tr.e[1].mn<0)return 0;
		tr.modify(1,k,v);
	}
	return 1;
}
int main()
{
	scanf("%d%d%d",&m,&k,&n);
	for(int i=1;i<=m;i++)scanf("%d",&s1[i]);
	for(int i=1;i<=k;i++)scanf("%d",&s2[i]);
	sort(s1+1,s1+m+1);sort(s2+1,s2+k+1);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),id[i]=i,fu[i]=lower_bound(s2+1,s2+k+1,s[i][1]+1)-s2;
	sort(id+1,id+n+1,cmp);
	int lb=1,rb=n,as=-1;
	if(check(100))as=100,rb=as-1;else lb=101;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check(mid))as=mid,rb=mid-1;
		else lb=mid+1;
	}
	printf("%d\n",as);
}
```



##### auoj478 砰砰博士

###### Problem

在直线上有 $n$ 个红点 $m$ 个蓝点，你可以连接两个异色的点，代价为这两个点的距离

你需要让每个点都至少和一个异色点相连，求最小代价

$n,m\leq 10^5$，点的坐标在 $[0,10^9]$ 直接。

$1s,256MB$

###### Sol

考虑模拟费用流，将每个点拆成两部分，一部分有 $1$ 个点，必须匹配（或者匹配额外代价 $-\infty$），另一部分有 $+\infty$ 个点，匹配额外代价为 $0$。

假设当前位置 $x$ 的红点匹配了一个蓝点，代价为 $x-y$，那么之有两种情况：

1. 红点反悔这次操作，然后来一个蓝点与它匹配，设新的匹配点位置为 $s$，那么之后这次操作的代价为 $s-x-(x-y)$，相当于加入一个位置在 $2x-y$ 的红点。
2. 蓝点反悔这次操作，如果红点是必须匹配的点，那么无法反悔。否则设新的匹配点位置为 $s$，那么之后这次操作的代价为 $s-y-(x-y)$，相当于加入一个位置在 $x$ 的蓝点。

对于一个点，先将必须匹配的匹配，对于有 $+\infty$ 个点的部分，先一直匹配直到总代价不再减少，然后再加入 $\infty$ 个这种点即可，可以使用 `pair` 记录当前这个点有 $1$ 个还是 $+\infty$ 个，显然不会两个 $+\infty$ 匹配、

如果一个匹配两个点都反悔了，那么匹配一定存在交叉，因此一定不优，所以总的反悔次数为 $O(n)$。

复杂度 $O(n\log n)$

另外可以考虑将序列分成若干段，每一段内部先是红点后是蓝点或者反过来，相邻两段可能共用一个点，这样可以得到一种 `dp`，分析转移系数发现可以线性。~~具体细节又咕了~~

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 105000
#define ll long long
priority_queue<pair<ll,int> > q0,q1;
int n,m,v1[N],v2[N];
ll as;
void add0(int v)
{
	pair<ll,int> f1=q1.top();q1.pop();
	as+=v-f1.first;q0.push(make_pair(2*v-f1.first,0));if(f1.second)q1.push(f1);
	while(1)
	{
		pair<ll,int> f1=q1.top();
		if(v>=f1.first)break;
		q1.pop();as+=v-f1.first;q0.push(make_pair(2*v-f1.first,0));q1.push(make_pair(v,0));if(f1.second)q1.push(f1);
	}
	q0.push(make_pair(v,1));
}
void add1(int v)
{
	pair<ll,int> f1=q0.top();q0.pop();
	as+=v-f1.first;q1.push(make_pair(2*v-f1.first,0));if(f1.second)q0.push(f1);
	while(1)
	{
		pair<ll,int> f1=q0.top();
		if(v>=f1.first)break;
		q0.pop();as+=v-f1.first;q1.push(make_pair(2*v-f1.first,0));q0.push(make_pair(v,0));if(f1.second)q0.push(f1);
	}
	q1.push(make_pair(v,1));
}
int main()
{
	q0.push(make_pair(-1e13,1));q1.push(make_pair(-1e13,1));
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v1[i]);
	for(int i=1;i<=m;i++)scanf("%d",&v2[i]);
	int s1=1,s2=1;
	for(int i=1;i<=n+m;i++)
	if(s1>n)add1(v2[s2++]);
	else if(s2>m)add0(v1[s1++]);
	else if(v1[s1]<v2[s2])add0(v1[s1++]);
	else add1(v2[s2++]);
	printf("%lld\n",as);
}
```



##### auoj479 怪盗之翼

###### Problem

有一个 $n\times m$ 的网格图，有些格子上有障碍

你需要画若干条回路，使得所有回路不经过障碍点，回路两两没有公共点且每个非障碍点都在一个回路上。

定义一个回路上一个点是转角当且仅当这个点和回路上前一个点，后一个点不在一条直线上。如果一个点 $(i,j)$ 是转角，则这个点有 $v_{i,j}$ 的贡献，否则这个点的贡献为 $0$。

求一种回路的方式使得总贡献最大，输出最大贡献，或者输出无解。

$n\leq 150,m\leq 30$

$2s,128MB$

###### Sol

选出若干回路覆盖所有非障碍点相当于选出若干条非障碍点之间的边，使得每个非障碍点度数为 $2$。因为网格图是二分图，直接建网络流即可判断是否有解。

定义横向的边为红边，纵向的点为蓝边，对于一个点，如果它是转角，那么它连出的边一定是一蓝一红，否则它连出去的边颜色相同。

考虑费用流模型对于每个点建三个点，第一个点与原点或汇点连边，第二个点连出所有的红边，第三个点连出所有的蓝边。第一个点与第二个点连两条边，两条边流量为 $1$，第一条费用为 $0$，第二条费用为 $v_{i,j}$。第一个点与第三个点同理。

这时如果满流说明合法，如果一个点是转角，那么这个点处连边为一蓝一红，因此费用为 $0$，否则两条边颜色相同，费用为 $v_{i,j}$。因此满流情况下的最小费用流即为最少需要减去多少贡献。

复杂度为 `mcmf` 复杂度，如果写 `primal-dual` 则复杂度是 $O(n^2m^2\log nm)$，复杂度理论上正确，~~但很有可能跑不过spfa费用流~~

###### Code

代码里面写的是删边，但本质相同

`primal-dual` 的代码可以参考 Topcoder 中的 `CurvyonRails`。~~虽然说那题题解里面没放代码~~

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 13505
#define M 233333
#define ll long long
int n,m,head[N],cnt=1,dis[N],cur[N],is[N],s[155][33],t[155][33],d[4][2]={-1,0,1,0,0,1,0,-1},as,ct,ct1,ct2,as1;
struct edge{int t,next,v,c;}ed[M];
void adde(int f,int t,int v,int c){ed[++cnt]=(edge){t,head[f],v,c};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0,-c};head[t]=cnt;}
bool spfa(int s,int t)
{
	memset(dis,0x3f,sizeof(dis));
	memcpy(cur,head,sizeof(cur));
	queue<int> st;
	dis[s]=0;st.push(s);is[s]=1;
	while(!st.empty())
	{
		int x=st.front();st.pop();is[x]=0;
		for(int i=head[x];i;i=ed[i].next)
		if(ed[i].v&&dis[ed[i].t]>dis[x]+ed[i].c)
		{
			dis[ed[i].t]=dis[x]+ed[i].c;
			if(!is[ed[i].t])st.push(ed[i].t),is[ed[i].t]=1;
		}
	}
	return dis[t]<=1e9;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	is[u]=1;
	int as=0,tp;
	for(int& i=cur[u];i;i=ed[i].next)
	if(!is[ed[i].t]&&dis[ed[i].t]==dis[u]+ed[i].c&&ed[i].v&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
	{
		as1+=1ll*tp*ed[i].c;ed[i].v-=tp;ed[i^1].v+=tp;f-=tp;as+=tp;
		if(!f){is[u]=0;return as;}
	}
	is[u]=0;
	return as;
}
int getid(int x,int y){return x*m-m+y;}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)scanf("%d",&s[i][j]),s[i][j]^=1;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)scanf("%d",&t[i][j]);
	int s1=3*n*m+1,t1=3*n*m+2;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=m;j++)
	if(s[i][j])
	if((i+j)&1)
	{
		int tp=getid(i,j),ct=0;
		for(int k=0;k<4;k++)if(s[i+d[k][0]][j+d[k][1]])ct++,adde((k/2+1)*n*m+tp,(k/2+1)*n*m+getid(i+d[k][0],j+d[k][1]),1,0);
		if(ct<2){printf("-1\n");return 0;}
		if(ct==2){if(s[i+1][j]^s[i-1][j])as+=t[i][j];}
		if(ct==3){adde(s1,tp,1,0);ct1++;as+=t[i][j];for(int k=0;k<4;k++)if(s[i+d[k^1][0]][j+d[k^1][1]]==0)adde(tp,(k/2+1)*n*m+tp,1,t[i][j]),adde(tp,(2-k/2)*n*m+tp,2,0);}
		if(ct==4)
		{
			adde(tp,n*m+tp,1,0);adde(tp,n*m+tp,1,t[i][j]);
			adde(tp,2*n*m+tp,1,0);adde(tp,2*n*m+tp,1,t[i][j]);
			adde(s1,tp,2,0);ct1+=2;as+=t[i][j];
		}
	}
	else
	{
		int tp=getid(i,j),ct=0;
		for(int k=0;k<4;k++)if(s[i+d[k][0]][j+d[k][1]]==1)ct++;
		if(ct<2){printf("-1\n");return 0;}
		if(ct==2){if(s[i+1][j]^s[i-1][j])as+=t[i][j];}
		if(ct==3){adde(tp,t1,1,0);ct2++;as+=t[i][j];for(int k=0;k<4;k++)if(s[i+d[k^1][0]][j+d[k^1][1]]==0)adde((k/2+1)*n*m+tp,tp,1,t[i][j]),adde((2-k/2)*n*m+tp,tp,2,0);}
		if(ct==4)
		{
			adde(n*m+tp,tp,1,0);adde(n*m+tp,tp,1,t[i][j]);
			adde(2*n*m+tp,tp,1,0);adde(2*n*m+tp,tp,1,t[i][j]);
			adde(tp,t1,2,0);ct2+=2;as+=t[i][j];
		}
	}
	if(ct1!=ct2){printf("-1\n");return 0;}
	while(spfa(s1,t1))ct1-=dfs(s1,t1,1e8);
	if(ct1){printf("-1\n");return 0;}
	printf("%d\n",as-as1);
}
```



#### NOI2020 模拟测试5

##### auoj480 黑白沙漠

###### Problem

有一个数轴，初始时只有区间 $[L,R]$ 是无风的，在这个区间中有 $n$ 个建筑，第 $i$ 个建筑的位置是 $a_i$，能在风中坚持 $b_i$ 的时间，之后它就会倒下。

在 $[L,R]$ 中随机选定一个点 $x$，之后无风的区间开始以每个单位时间缩小 $1$ 个单位长度的速度缩小，直至缩小至点 $x$ 处。缩小的方式为，任意时刻当前无风的区间 $[L,R]$ 满足 $\frac{R-x}{x-L}$ 不变。

对于每一个建筑，求出它最后一个倒下的概率，误差不超过 $10^{-9}$

$n\leq 2\times 10^5,|L|,|R|\leq 10^6$，建筑位置两两不同。

$2s,1024MB$

###### Sol

可以通过平移使得 $L=0$，因此不妨设 $L=0$。

考虑建筑 $i$ 坚持的时间，如果 $a_i<x$，则坚持的时间为 $b_i+(R-L)*\frac{a_i}{x}$。

考虑这部分的两个建筑 $i,j$，建筑 $i$ 坚持更久当且仅当 $b_ix+(R-L)a_i>b_jx+(R-L)a_j$。

只考虑左侧建筑的话，从左到右考虑 $x$，相当于支持加入一条直线，求出当前 $x$ 位置处最高的一条直线。

考虑依次加入直线，并维护直线形成的上凸壳。注意到后加入的直线的 $a_i$ 更大，因此如果某一条直线被加入的时候比某一条已经加入的直线低，因为它的 $a_i$ 更高，所以它的 $b_i$ 更低，因此它以后不会超过之前的直线，可以直接不加入这条直线。

否则，因为它在当前是最高的，所以它替换掉的一定是上凸壳的一段后缀，维护一个栈即可处理加入直线的操作，同时记录栈顶元素被下一个元素超过的时间即可。

这样可以得到 $O(n)$ 个 $x$ 的区间，每个区间内的 $x$ 左侧留到最后的建筑是相同的。

对右侧做相同的操作，可以得到 $O(n)$ 个 $x$ 的区间，每个区间内的 $x$ 两侧留到最后的建筑是相同的。

注意到 $b_i+(R-L)*\frac{a_i}{x}$ 是一个单调递减的函数，而右侧的坚持时间则是一个单调递增函数，因此可以二分求出每个区间内部哪一部分左侧最后倒下，哪一部分右侧最后倒下。

也可以考虑直接解出来，通分后是一个一次方程，注意细节，再归并两侧的区间即可做到线性。~~代码又咕了~~

复杂度 $O(n\log v+n\log n)$

###### Code

```cpp
include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 400500
int n,l,r,a[N],b[N],st[N],rb,fu[N],fu2[N],ct1,ct2,ct;
double ti[N],ti2[N],as[N];
double calc1(int x,int y){return 1.0*r*(a[y]-a[x])/(b[x]-b[y]);}
bool check(int x,int y,int z){return 1ll*(a[x]-a[y])*(b[z]-b[y])>=1ll*(a[y]-a[z])*(b[y]-b[x]);}
struct sth{double t;int x,y;friend bool operator <(sth a,sth b){return a.t<b.t;}}t[N*2];
double solve(int x,double t)
{
	if(a[x]<t)return b[x]+r-r*(t-a[x])/t;
	else return b[x]+r-r*(a[x]-t)/(r-t);
}
int main()
{
	scanf("%*d%d%d%d",&n,&l,&r);
	for(int i=1;i<=n;i++)scanf("%d",&a[i]),a[i]-=l;r-=l;
	for(int i=1;i<=n;i++)scanf("%d",&b[i]);
	int lb=1;
	double las=0;
	while(lb<=n||las<=r)
	{
		double st1=1e10,st2=lb>n?1e11:a[lb];
		if(rb>=2)st1=calc1(st[rb-1],st[rb]);
		if(st1>5e9&&st2>5e9){ti[++ct1]=r;break;}
		if(st1<st2)
		{
			ti[++ct1]=st1;rb--;fu[ct1]=st[rb];
			if(ti[ct1]>r)ti[ct1]=r;
		}
		else
		{
			if(!rb||st2*b[lb]+1.0*a[lb]*r>st2*b[st[rb]]+1.0*a[st[rb]]*r)
			{
				ti[++ct1]=st2;fu[ct1]=lb;
				while(rb&&b[st[rb]]<=b[lb])rb--;
				while(rb>=2&&check(lb,st[rb],st[rb-1]))rb--;
				st[++rb]=lb;
			}
			lb++;
		}
	}
	for(int i=1;i<=n;i++)a[i]=r-a[i];
	lb=n;rb=0;las=0;
	while(lb||las<=r)
	{
		double st1=1e10,st2=lb==0?1e11:a[lb];
		if(rb>=2)st1=calc1(st[rb-1],st[rb]);
		if(st1>5e9&&st2>5e9){ti2[++ct2]=r;break;}
		if(st1<st2)
		{
			ti2[++ct2]=st1;rb--;fu2[ct2]=st[rb];
			if(ti2[ct2]>r)ti2[ct2]=r;
		}
		else
		{
			if(!rb||st2*b[lb]+1.0*a[lb]*r>st2*b[st[rb]]+1.0*a[st[rb]]*r)
			{
				ti2[++ct2]=st2;fu2[ct2]=lb;
				while(rb&&b[st[rb]]<=b[lb])rb--;
				while(rb>=2&&check(lb,st[rb],st[rb-1]))rb--;
				st[++rb]=lb;
			}
			lb--;
		}
	}
	for(int i=1;i<=n;i++)a[i]=r-a[i];
	for(int i=1;i<ct1;i++)if(abs(ti[i]-ti[i+1])>1e-9)t[++ct]=(sth){ti[i],fu[i],1};
	for(int i=1;i<ct2;i++)if(abs(ti2[i]-ti2[i+1])>1e-9)t[++ct]=(sth){r-ti2[i+1],fu2[i],2};
	sort(t+1,t+ct+1);
	double ls=0;
	int v1=0,v2=0;
	t[++ct].t=r;
	for(int i=1;i<=ct;i++)
	{
		double l=ls,r=t[i].t;ls=t[i].t;
		if(v1==0)as[v2]+=r-l;
		else if(v2==0||v1==v2)as[v1]+=r-l;
		else
		{
			double l1=l,r1=r;
			for(int j=1;j<=75;j++)
			{
				double mid=(l+r)/2;
				if(solve(v1,mid)>=solve(v2,mid))l=mid;
				else r=mid;
			}
			as[v1]+=l-l1;as[v2]+=r1-r;
		}
		if(t[i].y==1)v1=t[i].x;else v2=t[i].x;
	}
	for(int i=1;i<=n;i++)printf("%.15lf\n",as[i]/r);
}
```



##### auoj481 荒野聚餐

###### Problem

有一个二分图，两侧各有 $n$ 个点，边有边权 $w_{i,j}$。

多组询问，每次给一个 $C$，你可以花费 $s$ 的代价，使得所有边权降低 $\frac sC$。

然后你需要给每个点分配一个权值 $v_i$，使得对于每条边 $(i,j)$，满足 $v_i+v_j\geq w_{i,j}$。

对于每个询问求出最小代价，保留一位小数

$n\leq 500,q\leq 5000$

$1s,1024MB$

###### Sol

先不考虑 $C$，考虑将后面的问题写成线性规划，相当于：

最小化 $\sum v_i$

满足 $\forall(i,j),v_i+v_j\geq w_{i,j}$

考虑对偶，则相当于：

最大化 $\sum w_{i,j}*s_{i,j}$，其中 $s_{i,j}$ 为这条边对应的变量

满足 $\forall i,\sum_j s_{i,j}\leq 1$ 且 $\forall j,\sum_i s_{i,j}\leq 1$

这相当于实数边权的最大权匹配，可以证明这东西等于整数边权的情况，即正常的二分图最大权匹配。

可以发现 `#458` 和 `arc130f` 也可以看成广义上的对偶，因此也可以像那两个题一样给这东西一个组合一点的解释。



考虑枚举最后的匹配中有多少条边，求有 $i$ 条边的最大权匹配。

设 $i$ 条边的最大匹配权值为 $v_i$，那么可以发现答案为 $\min_xCx+max_i(v_i-ix)$

右边相当于一个下凸壳，询问时直接枚举凸壳上的点即可，也可以排序后扫过去。

如果写 `spfa` 费用流，会被针对导致过不去。但奇怪的写法（每个点保留50条出边）能过。

如果写 `primal-dual`，复杂度为正确的 $O(n^3+nq)$，但常数非常离谱~~实测10s~~，完全过不去。主要问题在于链表存稠密图太慢。

upd: 魔改的 `primal-dual` 能过，虽然那东西魔改成了和 `KM` 非常像的形式。

如果写 `KM`，则常数较小可以通过。~~但我不会~~

###### Code

放弃操作.jpg

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
#include<vector>
using namespace std;
#define N 1005
#define ll long long
int n,q,a,head[N],cnt=1,ct,cur[N],as,vis[N];
ll tp[N],dis[N];
struct edge{int t,next,v,c;}ed[N*N];
void adde(int f,int t,int v,int c)
{
	ed[++cnt]=(edge){t,head[f],v,c};head[f]=cnt;
	ed[++cnt]=(edge){f,head[t],0,-c};head[t]=cnt;
}
vector<int> ls[N];
bool dij(int s,int t)
{
	for(int i=1;i<=ct;i++)dis[i]=1.01e17,vis[i]=0,cur[i]=head[i];
	dis[s]=0;
	while(1)
	{
		int x=0;
		for(int i=1;i<=ct;i++)if(!vis[i]&&(!x||dis[i]<dis[x]))x=i;
		if(dis[x]>1e17||!x)break;
		vis[x]=1;
		for(int i=head[x];i;i=ed[i].next)if(ed[i].v&&dis[ed[i].t]>dis[x]+ed[i].c)dis[ed[i].t]=dis[x]+ed[i].c;
	}
	for(int i=1;i<=ct;i++)ls[i].clear();
	for(int i=1;i<=ct;i++)
	for(int j=head[i];j;j=ed[j].next)
	{
		ed[j].c+=dis[i]-dis[ed[j].t];
		if(ed[j].c==0)ls[i].push_back(j);
	}
	return dis[t]<1e17;
}
bool bfs(int s,int t)
{
	for(int i=1;i<=ct;i++)dis[i]=-1,cur[i]=0;
	queue<int> qu;qu.push(s);dis[s]=0;
	while(!qu.empty())
	{
		int x=qu.front();qu.pop();
		for(int d=0;d<ls[x].size();d++)
		{
			int i=ls[x][d];
			if(ed[i].v&&dis[ed[i].t]==-1)
			{
				dis[ed[i].t]=dis[x]+1;qu.push(ed[i].t);
				if(ed[i].t==t)return 1;
			}
		}
	}
	return 0;
}
int dfs(int u,int t,int f)
{
	if(u==t||!f)return f;
	int as=0,tp;
	for(int& d=cur[u];d<ls[u].size();d++)
	{
		int i=ls[u][d];
		if(ed[i].v&&dis[ed[i].t]==dis[u]+1&&(tp=dfs(ed[i].t,t,min(f,ed[i].v))))
		{
			ed[i].v-=tp;ed[i^1].v+=tp;
			as+=tp;f-=tp;
			if(!f)return as;
		}
	}
	return as;
}
int dinic(int s,int t){int as=0;while(bfs(s,t))as+=dfs(s,t,1e9);return as;}
void dij_mcmf(int s,int t)
{
	for(int i=1;i<=n;i++)tp[i]=1e18;
	int as=0,ds=0,t1=0,t2;
	while(dij(s,t))
	{
		ds+=dis[t],t2=dinic(s,t);
		for(int i=t1+1;i<=t1+t2;i++)tp[i]=ds+tp[i-1];
		t1+=t2,as+=ds*t2;
	}
	for(int i=1;i<=n;i++)tp[i]=1e9*i-tp[i];
}
int main()
{
	scanf("%*d%d%d",&n,&q);
	ct=n*2+2;
	for(int i=1;i<=n;i++)adde(ct-1,i,1,0),adde(i+n,ct,1,0);
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)scanf("%d",&a),adde(i,j+n,1,1e9-a);
	dij_mcmf(ct-1,ct);
	while(q--)
	{
		scanf("%d",&a);
		ll as=tp[n];
		for(int i=0;i<n;i++)as=min(as,1ll*(a-i)*(tp[i+1]-tp[i])+tp[i]);
		printf("%lld.0\n",as);
	}
}
```



##### auoj482 火星在住

###### Problem

给一棵 $n$ 个点的带边权树和 $l,r$。

对于每一个 $k\in[l,r]$，求出在树上选 $k$ 条边，满足任意两条边没有公共端点，边权和的最大值，或输出无解。

$n\leq 2\times 10^5$

$2s,1024MB$

###### Sol

设 $dp_{u,0/1,i}$ 表示 $u$ 的子树内，$u$ 不能被匹配/是否匹配均可时，匹配了 $i$ 条边的最大权值。

如果没有根节点是否被选的限制，可以将问题转换成费用流，因此 $dp_{u,1,i}$ 是凸函数。

如果根节点不能被选，相当于每个儿子的函数做 $\max,+$ 卷积，凸函数卷积仍然是凸函数，所以 $dp_{u,0,i}$ 也是凸函数。

考虑给 $u$ 加入一个儿子 $v$，转移为：

$$
dp_{u,0}^{'}=dp_{u,0}*dp_{v,1}\\
dp_{u,1}^{'}=\max(dp_{u,1}*dp_{v,1},dp_{u,0}*dp_{v,0}*\{0,w_{u,v}\})
$$

其中 $*$ 表示凸函数卷积，$\max$ 表示对应位置取 $\max$，$\{0,w_{u,v}\}$ 表示一个只有两项的凸函数。但直接做复杂度是 $O(n^2)$，因为凸函数卷积复杂度和两个函数的和相关。

注意到 $dp_{u,0/1}$ 的项数不超过子树大小，考虑树链剖分。对于一条重链，先分治合并算出每个点只考虑所有轻儿子的 $dp$，然后在链上分治，设 $f_{u,v,0/1,0/1,i}$ 表示考虑 $u$ 到 $v$ 的重链，$u$ 不能被匹配/是否匹配均可，$v$ 不能被匹配/是否匹配均可时，匹配 $i$ 条边的最大权值，这仍然是一个凸函数，可以直接合并。

如果直接分治，复杂度 $O(n\log^2 n)$。但如果两部分都按照子树大小分治，则复杂度和 `SBT` 一样，为 $O(n\log n)$。~~只能快一点~~

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define ll long long
#define N 200500
vector<ll> dp[N][2],fu[N*4][2][2],f1[N*4][2],s[N][2];
int n,l,r,a,b,c,sn[N],st[N],sz[N],vl[N],head[N],cnt,v1[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
vector<ll> merge(vector<ll> a,vector<ll> b)
{
	int s1=a.size(),s2=b.size(),l1=0,r1=0;
	vector<ll> as;
	if(!s1||!s2)return as;
	as.push_back(0);
	for(int i=1;i<s1+s2-1;i++)
	{
		if(l1+1==s1)r1++;
		else if(r1+1==s2)l1++;
		else if(a[l1+1]+b[r1]>a[l1]+b[r1+1])l1++;
		else r1++;
		as.push_back(a[l1]+b[r1]);
	}
	return as;
}
vector<ll> doit(vector<ll> a,int b)
{
	vector<ll> as;as.push_back(0);
	for(int i=0;i<a.size();i++)as.push_back(a[i]+b);
	return as;
}
vector<ll> getmx(vector<ll> a,vector<ll> b)
{
	int s1=a.size(),s2=b.size();
	vector<ll> as;
	for(int i=0;i<s1||i<s2;i++)
	{
		if(i>=s1)as.push_back(b[i]);
		else if(i>=s2)as.push_back(a[i]);
		else as.push_back(max(a[i],b[i]));
	}
	return as;
}
int su[N];
void solve1(int x,int l,int r)
{
	if(l==r){f1[x][0]=s[l][0];f1[x][1]=s[l][1];return;}
	int mn=1e9,si=su[r]-su[l-1],mid=(l+r)>>1;
	for(int i=l;i<r;i++)
	{
		int tp=max(su[i]-su[l-1],su[r]-su[i]);
		if(tp<mn)mn=tp,mid=i;
	}
	solve1(x<<1,l,mid);solve1(x<<1|1,mid+1,r);
	f1[x][0]=merge(f1[x<<1][0],f1[x<<1|1][0]);
	f1[x][1]=getmx(merge(f1[x<<1][0],f1[x<<1|1][1]),merge(f1[x<<1|1][0],f1[x<<1][1]));
}
void solve2(int x,int l,int r)
{
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)fu[x][i][j].clear(),vector<ll>().swap(fu[x][i][j]);
	if(l==r){fu[x][0][0]=dp[st[l]][0];fu[x][1][1]=dp[st[l]][1];return;}
	int mn=1e9,si=su[r]-su[l-1],mid=(l+r)>>1;
	for(int i=l;i<r;i++)
	{
		int tp=max(su[i]-su[l-1],su[r]-su[i]);
		if(tp<mn)mn=tp,mid=i;
	}
	solve2(x<<1,l,mid);solve2(x<<1|1,mid+1,r);
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)
	for(int k=0;k<2;k++)for(int l1=0;l1<2;l1++)
	{
		fu[x][i][l1]=getmx(fu[x][i][l1],merge(fu[x<<1][i][j],fu[x<<1|1][k][l1]));
		if(!j&&!k)
		{
			int nt1=i|(mid-l==0),nt4=l1|(r-mid==1);
			fu[x][nt1][nt4]=getmx(fu[x][nt1][nt4],getmx(merge(fu[x<<1][i][j],fu[x<<1|1][k][l1]),doit(merge(fu[x<<1][i][j],fu[x<<1|1][k][l1]),vl[mid])));
		}
	}
}
void dfs1(int u,int fa){sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u];}
void dfs2(int u,int fa)
{
	if(sn[u])dfs2(sn[u],u);
	else {dp[u][0].push_back(0);return;}
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t==sn[u])v1[u]=ed[i].v;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs2(ed[i].t,u);
	int ct=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])s[++ct][0]=getmx(dp[ed[i].t][0],dp[ed[i].t][1]),s[ct][1]=getmx(dp[ed[i].t][0],doit(dp[ed[i].t][0],ed[i].v));
	if(ct)
	{
		for(int i=1;i<=ct;i++)su[i]=su[i-1]+s[i][0].size();
		solve1(1,1,ct),dp[u][0]=f1[1][0],dp[u][1]=f1[1][1];
	}
	else dp[u][0].push_back(0);
	if(sn[fa]!=u)
	{
		int ct2=0,st1=u;
		while(st1)st[++ct2]=st1,vl[ct2]=v1[st1],st1=sn[st1];
		for(int i=1;i<=ct2;i++)su[i]=su[i-1]+dp[st[i]][0].size();
		solve2(1,1,ct2);
		dp[u][0]=getmx(fu[1][0][0],fu[1][0][1]);
		dp[u][1]=getmx(fu[1][1][0],fu[1][1][1]);
	}
}
int main()
{
	scanf("%*d%d%d%d",&n,&l,&r);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c+1e9);
	dfs1(1,0);dfs2(1,0);
	vector<ll> as=getmx(dp[1][0],dp[1][1]);
	for(int i=l;i<=r;i++)
	if(i>=as.size())printf("- ");
	else printf("%lld ",as[i]-1000000000ll*i);
}
```



#### NOI2020 模拟测试6

##### auoj818 锁

###### Problem

有 $n$ 个人，每个人有一个权值 $v_i$，还有一个权值下限 $m$。

你可以选择钥匙的种类数 $k$，给每个人这 $k$ 种钥匙中的任意多种（一种钥匙可以给多个人），使得满足如下条件：

对于任意一个人的集合 $S$，$S$ 中的人加起来有每一种钥匙（即对于每一种钥匙，$S$ 中存在一个人有这种钥匙）当且仅当 $S$ 中人的权值和大于等于 $m$。

求满足条件需要的最小的 $k$。

$n\leq 20$

$1s,512MB$

###### Sol

对于一个集合 $S$，如果 $S$ 中人的权值和小于 $m$，那么一定存在一种钥匙使得 $S$ 中的人都没有。

考虑所有满足 $T$ 中人的权值和小于 $m$，但再加入任意一个人权值和就大于等于 $m$ 的集合 $T$，显然每个集合 $T$ 至少要对应一把 $T$ 中的人都没有的钥匙。

如果两个不同的集合 $T_1,T_2$ 对应的钥匙相同，则 $T_1\cup T_2$ 中的所有人都没有这种钥匙。但根据上面的性质，$T_1\cup T_2$ 中所有人的权值和大于等于 $m$，与条件矛盾。

因此每个集合 $T$ 对应的钥匙不同，容易发现给每个集合 $T$，给集合 $T$ 外的人 $T$ 对应的这种钥匙，得到的方案是合法的（如果有一种钥匙 $S$ 中的人都没有，这种情况当且仅当 $S$ 是某个 $T$ 的子集，因此得证），因此答案即为 $T$ 的数量。

枚举所有集合判断即可，复杂度 $O(n2^n)$ 或者 $O(2^n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 22
int n,m,v[N],as;
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<1<<n;i++)
	{
		long long su=0,mn=1e9;
		for(int j=1;j<=n;j++)if(i&(1<<j-1))su+=v[j];else mn=mn>v[j]?v[j]:mn;
		if(su<m&&su+mn>=m)as++;
	}
	printf("%d\n",as);
}
```



##### auoj819 bwt

###### Problem

给定字符串 $S$，定义 $f(S)$ 为：

将 $S$ 的所有循环位移串按照字典序从小到大排序，排序后按顺序取每个字符串的最后一个字符得到 $f(S)$。

给定一个随机 `01` 串 $s$，多组询问，每次随机生成 $a,b,c,d$，询问 $f(s_{a,...,b}),f(s_{c,...,d})$ 的字典序大小关系。

$n,q\leq 10^5$

$1s,512MB$

###### Sol

这个东西被称为 `Burrows-Wheeler Transform`，这也是题目名称的来源。

而这个东西的一个性质为，两个串变换后相同当且仅当它们循环同构，因此不同的串至少有 $\frac{2^n}n$ 个，因此可以看成这些串都分布的很随机。因此期望只需要比较 $f(s_{a,...,b}),f(s_{c,...,d})$ 的前 $(\log n)$ 位。

类似的，对于两个随机后缀，期望只需要 $O(\log n)$ 位的比较就可以得到它们的大小关系。因此求 $f(S)$ 的时候，除了最后 $O(\log n)$ 位作为开头的串外，剩下的串都可以直接比较后缀的大小。为了保证高的正确性可以取 $60$ 位。~~但实际上取20也能过~~

因为随机，比较两个循环位移串的大小也只需要 $O(\log n)$ ，可以先将后 $O(\log n)$ 个串排序

考虑如何找出所有循环位移串中最小的，第二小的，...

如果能将串分成两部分，对于每一部分支持询问第 $k$ 小串，那么使用类似归并的方式就可以每次找出需要的串。

对于第一部分，后缀排序后按照每个后缀的排名建主席树，查区间第 $k$ 大即可。这里后缀排序可以直接暴力。

对于第二部分，只有 $O(\log n)$ 个串，考虑直接排序。但如果还是 $O(\log n)$ 比较则太慢，考虑将这个位置向后的 $60$ 位循环串看成一个二进制数，可以 $O(\log n)$ 求出一次询问时这些位置中每一个向后的二进制数，然后就可以 $O(1)$ 比较。

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105000
#define M 2133333
int n,q,a,b,c,d,s,s1[N],tp[N],rk[N];
bool cmp(int a,int b){while(1){if(a>n)return 1;if(b>n)return 0;if(s1[a]<s1[b])return 1;if(s1[b]<s1[a])return 0;a++;b++;}}
struct pretree{
	int rt[N],ch[M][2],sz[M],ct;
	int build(int l,int r)
	{
		int st=++ct;
		if(l==r)return st;
		int mid=(l+r)>>1;
		ch[st][0]=build(l,mid);ch[st][1]=build(mid+1,r);
		return st;
	}
	void init(){rt[0]=build(0,n);}
	int modify(int x,int l,int r,int s)
	{
		int st=++ct;ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];sz[st]=sz[x]+1;
		if(l==r)return st;
		int mid=(l+r)>>1;
		if(mid>=s)ch[st][0]=modify(ch[x][0],l,mid,s);
		else ch[st][1]=modify(ch[x][1],mid+1,r,s);
		return st;
	}
	void modify2(int x,int v){rt[x]=modify(rt[x-1],0,n,v);}
	int query(int x,int y,int l,int r,int k)
	{
		if(l==r)return l;
		int mid=(l+r)>>1;
		if(sz[ch[x][0]]-sz[ch[y][0]]>k)return query(ch[x][0],ch[y][0],l,mid,k);
		else return query(ch[x][1],ch[y][1],mid+1,r,k-sz[ch[x][0]]+sz[ch[y][0]]);
	}
	int getkth(int l,int r,int k){return tp[query(rt[r],rt[l-1],0,n,k-1)];}
}tr;
bool check(int l,int r,int x,int y){for(int i=1;i<=r-l+1;i++){if(s1[x]<s1[y])return 1;if(s1[y]<s1[x])return 0;x=x==r?l:x+1;y=y==r?l:y+1;}return 0;}
struct sth{
	int l,r,l1,s2[75],s3[75],ct,c1,c2,d1,d2;
	long long vl[N];
	void doit(int l1,int r1)
	{
		if(l1==r1)return;
		int mid=(l1+r1)>>1;
		doit(l1,mid);doit(mid+1,r1);
		int l2=l1,r2=mid+1;
		for(int i=l1;i<=r1;i++)
		if(l2==mid+1)s3[i]=s2[r2++];
		else if(r2==r1+1)s3[i]=s2[l2++];
		else if(vl[s2[l2]]<=vl[s2[r2]])s3[i]=s2[l2++];
		else s3[i]=s2[r2++];
		for(int i=l1;i<=r1;i++)s2[i]=s3[i];
	}
	void pre(int l2,int r2)
	{
		l=l2;r=r2;c1=c2=0;d1=d2=0;ct=0;
		l1=r-60;if(l1<l)l1=l;
		long long v1=s1[r2],nw=r2;
		for(int i=1;i<60;i++)nw=nw==r2?l2:nw+1,v1=v1*2+s1[nw];
		vl[r2]=v1;
		for(int i=r2-1;i>=l1;i--)v1=(v1>>1)|(1ll*s1[i]<<59),vl[i]=v1;
		for(int i=l1;i<=r;i++)s2[++c1]=i;
		doit(1,c1);
		c2=l1-l;
	}
	int query()
	{
		if(ct==r-l+1)return -1;
		ct++;
		int v1,v2,s3;
		if(d1==c2)s3=s2[++d2];
		else if(d2==c1)s3=tr.getkth(l,l1-1,++d1);
		else
		{
			v1=s2[d2+1],v2=tr.getkth(l,l1-1,d1+1);
			if(check(l,r,v1,v2))d2++,s3=v1;
			else d1++,s3=v2;
		}
		return s3==l?s1[r]:s1[s3-1];
	}
}f1,f2;
int gen(){s=(s*100000005ll+20150609)%998244353;return s;}
int main()
{
	scanf("%d%d%d",&n,&q,&s);
	for(int i=1;i<=n;i++)s1[i]=gen()%2,tp[i]=i;
	sort(tp+1,tp+n+1,cmp);for(int i=1;i<=n;i++)rk[tp[i]]=i;
	tr.init();for(int i=1;i<=n;i++)tr.modify2(i,rk[i]);
	while(q--)
	{
		a=gen()%n+1;b=gen()%n+1;c=gen()%n+1;d=gen()%n+1;
		if(a>b)a^=b^=a^=b;if(c>d)c^=d^=c^=d;
		f1.pre(a,b);f2.pre(c,d);
		while(1)
		{
			int a1=f1.query(),a2=f2.query();
			if(a1==-1&&a2==-1){printf("0\n");break;}
			if(a1==-1){printf("-1\n");break;}
			if(a2==-1){printf("1\n");break;}
			if(a1<a2){printf("-1\n");break;}
			if(a1>a2){printf("1\n");break;}
		}
	}
}
```



##### auoj820 robot

###### Problem

给定 $l$，有一个长度为 $l$ 的指令，指令包含 `LRUD`，表示四个方向，机器人会循环执行指令，执行到对应指令时会向对应方向走 $1$ 单位距离。

给定 $n$ 个限制，第 $i$ 个限制为 $(t_i,x_i,y_i)$，表示在时刻 $t_i$ 机器人必须在 $(x_i,y_i)$。

求满足限制的指令序列数量，模 $10^9+7$

$n\leq 2\times 10^5,l\leq 2\times 10^6$

$1s,512MB$

###### Sol

将所有点旋转 $45$ 度，变成斜向行走。可以发现，斜向选一个方向行走相当于两维分别决定 $+1$ 或 $-1$，因此可以将两维分开考虑。

考虑一维的情况，限制形如 $(t_i,x_i)$，设执行了长度为 $l$ 的指令后机器人移动的有向距离为 $s$，时刻 $i(i\leq l)$ 时机器人的位置是 $v_i$，那么需要满足：

1. $v_0=0,v_l=s$
2. $v_i-v_{i-1}\in\{-1,1\}$ 
3. $\lfloor\frac{t_i}l\rfloor*s+v_{t_i\bmod l}=x_i$

则第一类和第三类限制都形如 $v_i=a*s+b$，考虑这些限制中下标相邻的两个 $v_i=as+b,v_j=cs+d$，可能有解当且仅当 $(c-a)s+(d-b)$ 与 $j-i$ 的奇偶性相同，且 $|(c-a)s+(d-b)|\leq j-i$，此时这一段的方案数为 $C_{j-i}^{\frac{(c-a)s+(d-b)+(j-i)}2}$。

显然所有相邻段的限制满足了整体限制就满足了，如果确定了一个 $s$，方案数即为每一段方案数的乘积。

考虑所有相邻段，可以得到最后 $s$ 可能的奇偶性以及可能的值域。

定义一段的长度为 $j-i$，如果这一段的 $c-a\neq 0$，那么这一段内合法的值域长度不会超过 $j-i$。

因此，如果有 $k$ 段的 $c-a\neq 0$，那么值域不会超过 $\frac lk$，因此可以枚举值域，枚举每一个 $c-a\neq 0$ 的段算贡献，复杂度 $O(l)$。也可以先不考虑奇偶性直接算。

显然 $c-a=0$ 的段对于每一个 $s$ 的方案数是相同的，可以预处理，这部分不影响复杂度。

复杂度 $O(n\log n+l)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 2005000
#define ll long long
#define mod 1000000007
int n,l,fr[N],ifr[N],st[N],as[N];
ll s[N][3],t[N][3];
bool cmp(int a,int b){return t[a][0]<t[b][0];}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int C(int i,int j){if(i<j||j<0)return 0;return 1ll*fr[i]*ifr[j]%mod*ifr[i-j]%mod;}
int solve()
{
	ll lb=-l,rb=l,is0=1,is1=1,tp=1;
	t[n+1][0]=l;t[n+1][1]=-1;st[n+1]=n+1;
	for(int i=1;i<=n+1;i++)
	{
		ll v1=t[st[i]][0]-t[st[i-1]][0],v2=t[st[i-1]][1]-t[st[i]][1],v3=t[st[i]][2]-t[st[i-1]][2];
		if(v2&1)if((v1^v3)&1)is0=0;else is1=0;
		else if((v1^v3)&1)return 0;
		if(v2==0)
		{
			if((v1^v3)&1)return 0;
			tp=1ll*tp*C(v1,(v1+v3)/2)%mod;continue;
		}
		double l1=1.0*(-v3-v1)/v2,r1=1.0*(-v3+v1)/v2;
		if(l1>r1)swap(l1,r1);
		ll l2=l1,r2=r1;
		if(l2<l1)l2++;if(r2>r1)r2--;
		if(lb<l2)lb=l2;if(rb>r2)rb=r2;
	}
	if(lb>rb)return 0;
	int as1=0;
	if(is0)
	{
		int tp=lb%2?lb+1:lb,ct=(rb-tp+2)/2;
		for(int j=1;j<=ct;j++)as[j]=1;
		for(int i=1;i<=n+1;i++)
		{
			ll v1=t[st[i]][0]-t[st[i-1]][0],v2=t[st[i-1]][1]-t[st[i]][1],v3=t[st[i]][2]-t[st[i-1]][2];
			if(v2==0)continue;
			for(int j=1;j<=ct;j++)
			{
				int tp1=tp+(j-1)*2;
				as[j]=1ll*as[j]*C(v1,(v3+v1+v2*tp1)/2)%mod;
			}
		}
		for(int j=1;j<=ct;j++)as1=(as1+as[j])%mod;
	}
	if(is1)
	{
		int tp=lb%2==0?lb+1:lb,ct=(rb-tp+2)/2;
		for(int j=1;j<=ct;j++)as[j]=1;
		for(int i=1;i<=n+1;i++)
		{
			ll v1=t[st[i]][0]-t[st[i-1]][0],v2=t[st[i-1]][1]-t[st[i]][1],v3=t[st[i]][2]-t[st[i-1]][2];
			if(v2==0)continue;
			for(int j=1;j<=ct;j++)
			{
				int tp1=tp+(j-1)*2;
				as[j]=1ll*as[j]*C(v1,(v3+v1+v2*tp1)/2)%mod;
			}
		}
		for(int j=1;j<=ct;j++)as1=(as1+as[j])%mod;
	}
	return 1ll*as1*tp%mod;
}
int main()
{
	fr[0]=1;for(int i=1;i<=2e6;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[2000000]=pw(fr[2000000],mod-2);
	for(int i=1999999;i>=0;i--)ifr[i]=1ll*ifr[i+1]*(i+1)%mod;
	scanf("%d%d",&n,&l);
	for(int i=1;i<=n;i++)scanf("%lld%lld%lld",&s[i][0],&s[i][1],&s[i][2]),t[i][0]=s[i][0]%l,t[i][1]=s[i][0]/l,st[i]=i;
	sort(st+1,st+n+1,cmp);
	int as1=1;
	for(int i=1;i<=n;i++)t[i][2]=s[i][1]+s[i][2];
	as1=1ll*as1*solve()%mod;
	for(int i=1;i<=n;i++)t[i][2]=s[i][1]-s[i][2];
	as1=1ll*as1*solve()%mod;
	printf("%d\n",as1);
}
```



#### NOI2020 模拟测试7

##### auoj821 白鱼赤乌

###### Problem

定义一个数字串是好的，当且仅当它每一位上的数字的总和和是 $10$ 的倍数。

定义一个数字串是优美的，当且仅当对于每一位，都存在一个包含这一位的子串，使得这个子串是好的。

给定 $n$，求长度不超过 $n$ 的数字串（可以以 $0$ 开头）中优美的串的数量，答案对 $10^9+7$ 取模。

多组数据

$T\leq 2\times 10^4,n<2^{31}$

$1s,512MB$

###### Sol

对于一个长度为 $n$ 的数字串，设 $s_i$ 表示 $[1,i]$ 位置的数字和模 $10$ 的余数。如果 $s_i=s_j$，那么 $[i+1,j]$ 就是好的。

因此，对于一个 $i$，如果 $s_{0,...,i-1}$ 与 $s_{i,...,n}$ 中存在一对相同元素，那么这个位置一定被一个好的子串覆盖，否则一定不被一个好的子串覆盖。

可以发现，每一个长度为 $n$ 的数字串和长度为 $n$，每个元素在 $[0,9]$ 间的序列 $s$ 一一对应，因此可以看成计算满足条件的的 $s$ 数量。

考虑容斥，钦定一些位置不合法，那么这些位置将 $s_{0,...,n}$ 划分成若干段，任意两段中不能有相同元素。

设 $dp_{i,j,k}$ 表示考虑了前 $i$ 位，前面已经被划分出的段内共有 $j$ 种元素，当前最后一个还没有被划分的段内有 $k$ 种元素的方案数乘上容斥系数之和

初始 $dp_{0,0,1}=1$，转移有如下情况：

1. 不分段，加一种新的元素: $dp_{i-1,j,k}*(10-j-k)\to dp_{i,j,k+1}$
2. 不分段，加一个在最后一段中已经存在的元素: $dp_{i-1,j,k}*k\to dp_{i,j,k+1}$
3. 分段，再在新的一段加一个新的元素: $-dp_{i-1,j,k}*(10-j-k)\to dp_{i,j+k,1}$

转移可以写成矩阵乘法，加上统计答案用的位置，矩阵大小为 $56$，下面设这个大小为 $m$。答案即为矩阵的 $n$ 次幂中某一项的值。

考虑分块预处理，设矩阵为 $S$，处理出 $S^k,S^{2^8k},S^{2^{16}k},S^{2^{24}k}(k=0,\cdots,2^8-1)$，然后一次询问相当于求一个 $1\times m$ 的矩阵乘上四个 $m\times m$ 的矩阵，可以做到 $O(m^2*4)$。因此复杂度 $O(T*k*m^2+k*m*3n^{\frac 1k})$。这里相当于取 $k=4$。

可以发现按照某种顺序排序后矩阵中只存在从前往后的转移，这样矩阵乘法的常数可以小很多。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 59
#define M 11
#define mod 1000000007
int T,n,id[M][M],f[N][N],ct,dp[N],dp2[N],s1[N];
struct mat{int s[N][N];mat(){for(int i=1;i<=ct;i++)for(int j=1;j<=ct;j++)s[i][j]=0;};}s,f1[256],f2[256],f3[256],f4[256];
mat operator *(mat a,mat b)
{
	mat c;
	for(int i=1;i<=ct;i++)
	for(int k=1;k<=ct;k++)
	if(a.s[i][k])
	for(int j=1;j<=ct;j++)
	c.s[i][j]=(c.s[i][j]+1ll*a.s[i][k]*b.s[k][j])%mod;
	return c;
}
void pre()
{
	for(int i=0;i<=10;i++)
	for(int j=1;j<=10;j++)
	if(i+j<=10)id[i][j]=++ct;
	for(int i=0;i<=10;i++)
	for(int j=1;j<=10;j++)
	if(id[i][j])
	{
		f[id[i][j]][id[i][j]]=j;
		if(id[i+j][1])f[id[i][j]][id[i+j][1]]=mod-(10-i-j);
		if(id[i][j+1])f[id[i][j]][id[i][j+1]]=10-i-j;
	}
	for(int i=1;i<=ct;i++)f[i][ct+1]=1;ct++;f[ct][ct]=1;
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=ct;j++)s.s[i][j]=f[i][j];
	f1[2]=s*s;
	for(int i=1;i<=ct;i++)f1[0].s[i][i]=f2[0].s[i][i]=f3[0].s[i][i]=f4[0].s[i][i]=1;
	f1[1]=s;for(int i=2;i<256;i++)f1[i]=f1[i-1]*f1[1];
	f2[1]=f1[1]*f1[255];for(int i=2;i<256;i++)f2[i]=f2[i-1]*f2[1];
	f3[1]=f2[1]*f2[255];for(int i=2;i<256;i++)f3[i]=f3[i-1]*f3[1];
	f4[1]=f3[1]*f3[255];for(int i=2;i<256;i++)f4[i]=f4[i-1]*f4[1];
}
int query(int n)
{
	for(int i=1;i<ct;i++)dp[i]=f[1][i];dp[ct]=0;
	int v1=n&255;n>>=8;
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=ct;j++)dp2[j]=(dp2[j]+1ll*dp[i]*f1[v1].s[i][j])%mod;
	for(int i=1;i<=ct;i++)dp[i]=dp2[i],dp2[i]=0;
	v1=n&255;n>>=8;
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=ct;j++)dp2[j]=(dp2[j]+1ll*dp[i]*f2[v1].s[i][j])%mod;
	for(int i=1;i<=ct;i++)dp[i]=dp2[i],dp2[i]=0;
	v1=n&255;n>>=8;
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=ct;j++)dp2[j]=(dp2[j]+1ll*dp[i]*f3[v1].s[i][j])%mod;
	for(int i=1;i<=ct;i++)dp[i]=dp2[i],dp2[i]=0;
	v1=n&255;n>>=8;
	for(int i=1;i<=ct;i++)
	for(int j=1;j<=ct;j++)dp2[j]=(dp2[j]+1ll*dp[i]*f4[v1].s[i][j])%mod;
	for(int i=1;i<=ct;i++)dp[i]=dp2[i],dp2[i]=0;
	return dp[ct];
}
int main()
{
	pre();
	scanf("%d",&T);while(T--)scanf("%d",&n),printf("%d\n",query(n));
}
```



##### auoj822 祸及池鱼

###### Problem

给 $n$ 个二进制数 $a_{1,\cdots,n}$ 的二进制表示，第 $i$ 个表示的长度为 $l_i$。

称若干个数组成的序列是好的，当且仅当它们的二进制 `or` 和等于它们的和，即每一个二进制位上最多有一个数在这一位为 $1$。

你需要找到一个序列 $b_{1,\cdots,n}$，使得它满足如下条件：

1. $\forall i,a_i\leq b_i$
2. $b_{1,\cdots,n}$ 是好的。
3. $\sum b_i$ 尽量小。

求最小的 $\sum b_i$，输出二进制表示形式。

$n,\sum l_i\leq 3\times 10^5$

$2s,512MB$

###### Sol

等会再说

##### auoj823 鲁鱼陶阴

###### Problem

给定一个小写字符串 $s$ 和 $op\in\{0,1\}$，$q$ 次询问，每次给定 $l,r$。

如果 $op=1$，则需要求出将 $s_{l,...,r}$ 中的字符全部替换成 `Y` 后，$s$ 中的本质不同子串数量。

否则，需要求出将 $s_{l,...,r}$ 中的字符全部替换成对应的大写字符后，$s$ 中的本质不同子串数量。

询问之间独立

$n,q\leq 2\times 10^5$

$1s,512MB$

###### Sol

考虑将子串分成三类

1. 同时包含大写和小写字符的串
2. 只包含小写字符的串
3. 只包含大写字符的串

对于第一类，因为大写字符只有一段，显然任意位置不同的两个这样的串不可能相同，可以直接计算这部分数量。

对于第二类，相当于求出一个前缀和一个后缀部分的本质不同子串数量。

考虑一个子串，设它第一次出现的结尾位置是 $a$，最后一次出现的开头位置是 $b$，那么如果询问 $[l,r]$ 满足 $l\leq a,r\geq b$，那么这个子串不会在这次询问的部分中出现，否则会出现。

考虑 `SAM` 的 `fail` 树/后缀树上的点，对于一个 `SAM` 中的点，可以发现它对应的所有子串的 $a$ 是相同的，$b$ 是连续的一段。

考虑计算不出现的子串数，按 $l$ 从大到小做扫描线，维护 $r$ 的贡献，相当于区间加上一个一次函数并单点询问，也可以差分后变成区间加区间询问，复杂度 $O(n\log n)$。



对于第三类，$op=1$ 时数量就是 $r-l+1$，否则数量是这个区间内的本质不同子串数量。

考虑将询问按照 $r$ 排序，维护每个子串在 $[1,r]$ 中最后一次出现的开头位置 $s$，区间本质不同子串数量相当于查 $s\geq l$ 的子串数量。

考虑每个子串在 $[1,r]$ 中最后一次出现的结尾位置 $t$，显然每个 `SAM` 上的点对应的所有串的 $t$ 是相同的。

对于一个 $r$，以 $r$ 位置结尾的子串在 `fail` 树上就是 $s_{1,...,r}$ 对应的位置到根的一条链，因此 $r$ 增加时的操作相当于将一条链上的 $t$ 改成 $r$。

对于一条 $u$ 到祖先 $v$ 的链，如果它们的 $t$ 相同，那么它们的 $s$ 一定构成一段连续的区间。

注意到到根的一条链赋值可以看成 `LCT` 的 `access` 操作，使用 `LCT` 维护，在 `access` 的时候可以找出所有需要改变 $t$ 且之前 $t$ 相同的段（每次实边构成的链），由 `LCT` 的复杂度分析这样的段数量为 $O(n\log n)$，然后每一段在线段树上修改，询问在线段树上查询即可。

复杂度 $O(n\log^2 n)$

可以发现第二类的数量可以由后缀+分隔符+前缀的本质不同子串数量减去跨过分隔符的子串数量得到，因此也可以用区间本质不同子串的做法维护。

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 400500
#define ll long long
int n,q,a,b,ty,f1[N][2],s2[N],tp[N][3],s3[N];
ll as[N];
char st[N];
bool cmp(int a,int b){return f1[a][1]<f1[b][1];}
bool cmp2(int a,int b){return f1[a][0]>f1[b][0];}
bool cmp3(int a,int b){return tp[a][0]>tp[b][0];}
struct segt{
	struct node{int l,r,lz;ll su;}e[N*2];
	void build(int x,int l,int r){e[x].lz=e[x].su=0;e[x].l=l;e[x].r=r;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void pushup(int x){e[x].su=e[x<<1].su+e[x<<1|1].su;}
	void doit(int x,int v){e[x].lz+=v;e[x].su+=1ll*v*(e[x].r-e[x].l+1);}
	void pushdown(int x){doit(x<<1,e[x].lz);doit(x<<1|1,e[x].lz);e[x].lz=0;}
	void modify(int x,int l,int r,int v){if(l>r)return;if(e[x].l==l&&e[x].r==r){doit(x,v);return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify(x<<1,l,r,v);else if(mid<l)modify(x<<1|1,l,r,v);else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);pushup(x);}
	ll query(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].su;pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return query(x<<1,l,mid)+query(x<<1|1,mid+1,r);}
}tr;
struct SAM{
	int ch[N][26],fail[N],len[N],in[N],f1[N],f2[N],las,ct,ls[N];
	void init(){for(int i=1;i<=n*2;i++)f1[i]=1e7,f2[i]=0;las=ct=1;}
	void ins(int s,int tp)
	{
		int st=++ct,s1=las;len[st]=len[s1]+1;las=ct;ls[tp]=st;
		while(!ch[s1][s])ch[s1][s]=st,s1=fail[s1];
		f1[st]=f2[st]=tp;
		if(!s1)fail[st]=1;
		else
		{
			int nt=ch[s1][s];
			if(len[nt]==len[s1]+1)fail[st]=nt;
			else
			{
				int cl=++ct;len[cl]=len[s1]+1;
				for(int i=0;i<26;i++)ch[cl][i]=ch[nt][i];
				fail[cl]=fail[nt];fail[nt]=fail[st]=cl;
				while(ch[s1][s]==nt)ch[s1][s]=cl,s1=fail[s1];
			}
		}
	}
	void pre()
	{
		queue<int> st;
		for(int i=1;i<=ct;i++)in[fail[i]]++;
		for(int i=1;i<=ct;i++)if(!in[i])st.push(i);
		while(!st.empty())
		{
			int s=st.front();st.pop();
			f1[fail[s]]=min(f1[fail[s]],f1[s]);
			f2[fail[s]]=max(f2[fail[s]],f2[s]);
			in[fail[s]]--;if(!in[fail[s]])st.push(fail[s]);
		}
	}
}s;
struct LCT{
	int ch[N][2],fa[N],v1[N],v2[N],mn[N],mx[N],fg[N];
	bool nroot(int x){return ch[fa[x]][0]==x||ch[fa[x]][1]==x;}
	void pushup(int x){mn[x]=min(min(mn[ch[x][0]],mn[ch[x][1]]),v1[x]);mx[x]=max(max(mx[ch[x][0]],mx[ch[x][1]]),v2[x]);}
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
			if(las)
			tr.modify(1,las-mx[x]+1,las-mn[x]+1,-1);
			if(t1){int st2=doit(t1);splay(st2);fg[st2]=las;}
			ch[x][1]=tp;tp=x;pushup(x);x=fa[x];
		}
		int st=doit(tp);splay(st);fg[st]=f;
		tr.modify(1,1,f,1);splay(tp);
	}
	void init()
	{
		int ct=s.ct;
		for(int i=1;i<=ct;i++)v1[i]=mn[i]=s.len[s.fail[i]]+1,v2[i]=mx[i]=s.len[i],fa[i]=s.fail[i];
		mx[0]=0;mn[0]=1e9;
	}
}lct;
int main()
{
	scanf("%d%d%s%d",&n,&ty,st+1,&q);
	s.init();for(int i=1;i<=n;i++)s.ins(st[i]-'a',i);
	tr.build(1,1,n);lct.init();
	for(int i=1;i<=q;i++)scanf("%d%d",&f1[i][0],&f1[i][1]),s2[i]=i;
	sort(s2+1,s2+q+1,cmp);
	int lb=1;
	for(int i=1;i<=n;i++)
	{
		lct.access(s.ls[i],i);
		while(lb<=q&&f1[s2[lb]][1]==i)as[s2[lb]]+=tr.query(1,f1[s2[lb]][0],n),lb++;
	}
	if(ty==1)for(int i=1;i<=q;i++)as[i]=f1[i][1]-f1[i][0]+1;
	for(int i=1;i<=q;i++)as[i]+=1ll*(f1[i][0]-1)*(n-f1[i][0]+1)+1ll*(f1[i][1]-f1[i][0]+1)*(n-f1[i][1]);
	s.pre();tr.build(1,1,n);
	ll su=0;
	for(int i=2;i<=s.ct;i++)s3[i]=i,tp[i][0]=s.f1[i],tp[i][1]=s.f2[i]-s.len[i]+1,tp[i][2]=s.f2[i]-s.len[s.fail[i]],su+=s.len[i]-s.len[s.fail[i]];
	sort(s3+2,s3+s.ct+1,cmp3);
	sort(s2+1,s2+q+1,cmp2);
	int l1=1,r1=2;
	for(int i=n;i>=1;i--)
	{
		while(r1<=s.ct&&tp[s3[r1]][0]==i)tr.modify(1,tp[s3[r1]][1],tp[s3[r1]][2],1),r1++;
		while(l1<=q&&f1[s2[l1]][0]==i)as[s2[l1]]+=su-tr.query(1,1,f1[s2[l1]][1]),l1++;
	}
	for(int i=1;i<=q;i++)printf("%lld\n",as[i]);
}
```



#### NOI2020 模拟测试8

##### auoj824 林海的密码

###### Problem

给定 $c$，你需要构造一个可以有重边的有向图，满足点数不超过 $60$，边数不超过 $220$，满足以 $n$ 为根的外向生成树数量为 $c$。

$c\leq 10^{18}$

$1s,512MB$

###### Sol

考虑造一条链然后加入边。如果加入多条边，则这些边互相影响可能导致生成树个数难以直接得到。

考虑一种简单的避免影响的方式，比如让所有加入边都连向 $1$，这样一个生成树最多选一条这样的边。

那么考虑一条链，对于每个 $i$ 有边 $i\to i+1,i+1\to i$，这样已经有 $1$ 的生成树数量。然后考虑加入边 $x\to 1$，可以发现包含这条边的生成树只可能是如下情况：

$n$ 连到 $x$，$x$ 连向 $1$。考虑链上剩余部分，一定存在一个分界点，$1$ 连向分界点上面，$x$ 连向分界点下面，因此方案数为 $x-1$。

这样可以构造出 $O(n^2)$ 内的所有数，但数量级完全不够。

考虑加入两条 $i\to i+1$ 的边，这样如果中间部分为 $1\to t-1,x\to t$，则方案数为 $2^{t-2}$。求和可以发现 $x\to 1$ 边的贡献为 $2^{x-1}-1$。

此时考虑按位从高到低构造，这样点数为 $\log_2 c$，但边数最大可能达到 $4\log_2 c$，实际上容易卡到 $235$ 以上。~~但是数据没卡~~

上面的方式为从小到大每一步 $*2$，考虑变成 $*2$ 和 $*3$ 中随机选择一个，这样从高到低构造时可能一条边需要加两次，但 $\frac 6{\log 3}$ 实际上比 $\frac 4{\log 2}$ 更小，实际上 $*3$ 更优。随机选择一个后使用之前的方式构造，随机情况下边数不超过 $190$，而且多次随机完全卡不掉。

复杂度 $O(\log c)$

###### Code

```cpp
#include<cstdio>
#include<random>
using namespace std;
#define N 73
long long n,su[N];
int m,v[N];
mt19937 rnd(1);
int main()
{
	scanf("%lld",&n);n--;
	while(1)
	{
		long long ct=0,tp=n;
		m=0;su[0]=1;
		while(su[m]<=n)
		{
			m++;v[m]=2+rnd()%2;ct+=v[m]+1;
			su[m]=su[m-1]*v[m];
		}
		ct-=v[m]+1;m--;ct++;
		for(int i=1;i<=m;i++)su[i]+=su[i-1];
		for(int i=m;i>=0;i--)ct+=tp/su[i],tp%=su[i];
		if(ct<=220)
		{
			printf("%d %d\n",m+2,ct);v[m+1]=0;
			for(int i=1;i<=m+1;i++)
			{
				printf("%d %d\n",i+1,i);
				for(int j=1;j<=v[i];j++)printf("%d %d\n",i,i+1);
			}
			for(int i=m;i>=0;i--)while(n>=su[i])printf("%d %d\n",i+2,1),n-=su[i];
			return 0;
		}
	}
}
```

##### auoj825 ⽪卡丘

###### Problem

给一棵初始只有根节点的树，每一个时刻，如果一个点的儿子数不足 $2$ 个，则这个点上会长出若干个叶子直到儿子数为 $2$。

有 $q$ 次操作，每次操作为以下两种之一：

1. 将时间向后 $t$ 个时刻。
2. 给一个 `LR` 组成的序列 $s$，从根开始按照这个序列向下走可以走到一个点（保证这个点存在），然后删除这个点以及这个点的子树，这个操作不改变时间。

每一次操作后求出当前树的点数，答案模 $998244353$

多组数据

$\sum q,\sum t,\sum |s|\leq 10^6$

$1s,512MB$

###### Sol

对于每个点 $i$，设 $f_i$ 表示 $i$ 子树内节点数，$g_i$ 表示 $i$ 子树内下一时刻增加的叶子数。

则每经过一个时刻，$f_i^{'}=f_i+g_i,g_i^{'}=2g_i$。多个时刻的变化也容易维护。

考虑所有 $2$ 操作序列形成的 `trie`，则非 `trie` 部分不会受到修改影响，这部分容易求出。只考虑 `trie` 节点的 $f,g$。

对于 $1$ 操作，直接增加当前时间，更新根的答案即可。

对于 $2$ 操作，考虑更新这条链上的所有点。一个问题是不能每次更新子树内的所有点，因此考虑类似标记永久化的操作。

对于每个点记录这个点上次被更新的时刻 $t_i$，以及这个点上次被删除的时刻。

首先从上往下考虑每个点，记录这个点到根的路径上上次删除操作的时刻 $s$，如果 $s<t_i$，那么这个点从上次更新后没有任何操作影响它，因此直接用增加的时刻数量更新 $f,g$ 即可。

如果 $s>t_i$，说明与在这个点上次更新后，删除了它的一个祖先。记录删除的时刻和深度就可以得到当前点当前的深度，这样可以得到新的 $f,g$。

这样就求出了这条链上当前时刻的 $f,g$。然后考虑删除一个点的子树对 $f,g$ 的影响，显然只会影响这个点到根的路径。

先对于链上的点，每个点减去它需要被修改的儿子的贡献，然后将删除的点设为 $f=0,g=1$，再向上更新即可。

复杂度 $O(\sum q\log t+\sum |s|\log t)$ 或者 $O(\sum q+\sum t+\sum |s|)$（预处理 $2$ 的幂）。

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1006000
#define mod 998244353
int T,n,s1[N][2],ct,ch[N][2],fa[N],las[N],dp[N],su[N],ti[N],dep[N],nt,st[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
char s[N],t[N];
int ins()
{
	int st=1;
	for(int i=1;t[i];i++)
	{
		int s1=t[i]=='R';
		if(!ch[st][s1])ch[st][s1]=++ct,fa[ct]=st,dep[ct]=dep[st]+1;
		st=ch[st][s1];
	}
	return st;
}
void modify(int x)
{
	int ct=0,s2=x;
	while(s2)st[++ct]=s2,s2=fa[s2];
	int ls=-1e9,de=0;
	for(int i=ct;i>0;i--)
	{
		if(las[st[i]]>ls)ls=las[st[i]],de=dep[st[i]];
		if(ls>ti[st[i]])
		{
			int fu=nt-ls-dep[st[i]]+de;
			dp[st[i]]=pw(2,fu),su[st[i]]=dp[st[i]]-1;ti[st[i]]=nt;
		}
		else
		{
			int fu=nt-ti[st[i]];
			su[st[i]]=(su[st[i]]+1ll*dp[st[i]]*(pw(2,fu)-1))%mod;
			dp[st[i]]=1ll*dp[st[i]]*pw(2,fu)%mod;
			ti[st[i]]=nt;
		}
	}
	for(int i=ct;i>1;i--)su[st[i]]=(su[st[i]]-su[st[i-1]]+mod)%mod,dp[st[i]]=(dp[st[i]]-dp[st[i-1]]+mod)%mod;
	las[x]=nt;dp[x]=1;su[x]=0;
	for(int i=2;i<=ct;i++)su[st[i]]=(su[st[i]]+su[st[i-1]])%mod,dp[st[i]]=(dp[st[i]]+dp[st[i-1]])%mod;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		for(int i=1;i<=ct;i++)las[i]=dp[i]=su[i]=dep[i]=ch[i][0]=ch[i][1]=fa[i]=0;
		ct=1;las[1]=-1;nt=0;ti[1]=0;su[1]=1;dp[1]=2;
		scanf("%d",&n);
		for(int i=1;i<=n;i++)
		{
			scanf("%s",s+1);
			if(s[1]=='G')s1[i][0]=0,scanf("%d",&s1[i][1]);
			else s1[i][0]=1,scanf("%s",t+1),s1[i][1]=ins();
		}
		for(int i=2;i<=ct;i++)ti[i]=-2333,las[i]=-1e7;
		for(int i=1;i<=n;i++)
		if(s1[i][0]==0)
		{
			nt+=s1[i][1];
			su[1]=(su[1]+1ll*dp[1]*(pw(2,s1[i][1])-1)%mod+mod)%mod;ti[1]=nt;
			dp[1]=1ll*dp[1]*pw(2,s1[i][1])%mod;
			printf("%d\n",su[1]);
		}
		else modify(s1[i][1]),printf("%d\n",su[1]);
	}
}
```



##### auoj826 我永远喜欢

###### Problem

有 $n$ 种数，第 $i$ 种数有 $a_i$ 个。

你需要以任意顺序将它们合并成一个序列，定义一个序列的权值为：

初始权值为 $1$，将序列首尾相接形成一个环，对于环上每一个极长的元素相同的段，设它的长度为 $l$，则权值乘上 $\frac 1{l!}$。

求出所有不同的序列的权值和，模 $998244353$

$a_i>0,\sum a_i\leq 2\times 10^5$

$3s,512MB$

###### Sol

考虑只计算开头与结尾不同的序列，对于任意一个序列，可以将结尾连续的若干个与开头相同的数放到开头，变成开头结尾不同的序列且权值不变。因此计算开头与结尾不同的序列时，需要额外乘上开头第一段长度的权值。

而对于这样的序列，可以发现计算权值的时候只需要考虑链上的情况。

首先考虑不在开头的颜色，相当于将 $a_i$ 个数分成若干段，每一段的贡献为 $\frac 1{l!}$。

假设分成了 $k$ 段，长度分别为 $l_1,...,l_k$，权值为 $\frac 1{\prod l_i!}$。考虑乘上 $a_i!$，那么权值变成一个多重组合，即序列中放 $l_1$ 个 $1$，$l_2$ 个 $2$，...，$l_k$ 个 $k$ 的方案数。

那么对于一个 $k$，所有方案的权值和乘上 $a_i!$ 相当于求有多少个长度为 $a_i$ 的序列满足每个元素都在 $[1,k]$ 之间，且每个元素都出现过。

这可以容斥没有出现的颜色数量解决，容斥可以用 `FFT` 优化。

对于在开头的颜色，相当于需要在第一段中再选一个数出来，考虑上一步的操作，选出一个数可以看成在上面的序列中把一个 $1$ 变成 $0$，这时有两种情况：

1. 还有 $1$ 出现，相当于长度为 $a_i-1$ 的序列，满足每个元素都在 $[1,k]$ 之间，且每个元素都出现过，然后在任意位置插入 $0$ ，因此再乘上 $a_i$。
2. 没有 $1$ 出现，相当于长度为 $a_i-1$ 的序列，满足每个元素都在 $[2,k]$ 之间，且每个元素都出现过，同样需要插入 $0$，方案数乘上 $a_i$。

这些都可以容斥求，同时为了保证开头结尾不同，考虑在结尾再加一个长度为 $0$ 的段，并保证这种数分出来的段中第一段在最开头，最后一段在结尾。



如果确定了哪种颜色作为开头，并且确定了每种颜色的段数，接下来还需要算将这些段排列，使得相邻两段不相邻，且开头段的颜色和结尾段相同的方案数。

考虑容斥，对于一种颜色，假设分成了 $k$ 段，钦定有 $a$ 个段相邻，方案数乘上容斥系数为 $(-1)^aC_{k-1}^a$，之后还剩下 $k-a$ 段可以任意放。

考虑对于每种颜色作为开头/不作为开头时，分成若干段，再容斥变成 $i$ 段任意放，所有方案第一步的权值和乘上第二步容斥系数的值的和。这相当于在之前的生成函数上再做一次 `FFT`。

考虑容斥后将所有颜色的段放在一起，设第 $i$ 种颜色有 $x_i$ 段，第一种颜色是开头的颜色，则第一种颜色同时需要是结尾，因此方案数为 $\frac{(\sum x_i-2)!}{(x_1-2)!\prod{i>1}x_i!}$。

对于一种颜色 $c$，设 $f_{c,i}$ 表示它不在开头时，两步之后分成 $i$ 段的权值和乘上 $\frac 1{i!}$ 的结果。$g_{c,i}$ 表示它在开头时，两步之后分成 $i$ 段的权值和乘上 $\frac 1{(i-2)!}$ 的结果。将它们看成生成函数，即 $F_c(x)=\sum f_{c,i}x^i,G_c(x)=\sum g_{c,i}x^i$。

那么只需要求出 $\sum_{i=1}^nG_i(x)\prod_{j\neq i}F_j(x)$，然后每一项乘上一个系数 $\frac 1{(i-2)!}$ 求和即可得到答案。

考虑分治，对于每一个区间 $[l,r]$，求出 $\sum G_i(x)\prod_{j\neq i}F_j(x)$ 和 $\prod F_j(x)$，然后 `FFT` 合并即可。

复杂度 $O(\sum a_i\log n\log \sum a_i)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 530001
#define mod 998244353
int n,s[N],g[2][N*2],rev[N*2],a[N],b[N],c[N],d[N],e[N],f[N],fr[N],ifr[N],ntt[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void pre()
{
	for(int d=0;d<2;d++)
	for(int i=2;i<=1<<19;i<<=1)
	for(int j=0;j<(i>>1);j++)
	g[d][i+j]=pw(3,mod-1+(d*2-1)*(mod-1)/i*j);
	for(int i=2;i<=1<<19;i<<=1)
	for(int j=0;j<i;j++)rev[i+j]=(rev[i+(j>>1)]>>1)|((j&1)?(i>>1):0);
	fr[0]=ifr[0]=1;for(int i=1;i<=2e5;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=i;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g[t][st]%mod;
		ntt[k]=(v1+v2)-(v1+v2>=mod?mod:0);ntt[k+(i>>1)]=v1-v2+(v1<v2?mod:0);
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	vector<int> c;
	int s1=a.size(),s2=b.size();
	for(int i=0;i<s1||i<s2;i++)c.push_back(((i<s1?a[i]:0)+(i<s2?b[i]:0))%mod);
	return c;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	if(s1+s2<=200)
	{
		for(int i=0;i<s1+s2;i++)c[i]=0;
		for(int i=0;i<s1;i++)
		for(int j=0;j<s2;j++)
		c[i+j]=(c[i+j]+1ll*a[i]*b[j])%mod;
		vector<int> as;
		for(int i=0;i<s1+s2-1;i++)as.push_back(c[i]);
		return as;
	}
	int l=1;while(l<=s1+s2)l<<=1;
	for(int i=0;i<l;i++)c[i]=d[i]=0;
	for(int i=0;i<s1;i++)c[i]=a[i];
	for(int i=0;i<s2;i++)d[i]=b[i];
	dft(l,c,1);dft(l,d,1);for(int i=0;i<l;i++)c[i]=1ll*c[i]*d[i]%mod;dft(l,c,0);
	vector<int> as;
	for(int i=0;i<s1+s2-1;i++)as.push_back(c[i]);
	return as;
}
vector<int> doit0(int s)
{
	int l=1;while(l<=s*2+5)l<<=1;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=0;i<=s;i++)a[i]=1ll*ifr[i]*pw(i,s)%mod;
	for(int i=0;i<=s;i++)b[i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	dft(l,a,1);dft(l,b,1);
	for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;
	dft(l,a,0);
	for(int i=s+1;i<l;i++)a[i]=0;
	for(int i=0;i<l;i++)b[i]=0;
	for(int i=0;i<=s;i++)b[s-i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	for(int i=1;i<=s;i++)a[i]=1ll*a[i]*fr[i]%mod*fr[i-1]%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	vector<int> as;as.push_back(0);
	for(int i=1;i<=s;i++)as.push_back(1ll*a[i+s]*ifr[i-1]%mod*ifr[i]%mod*ifr[s]%mod);
	return as;
}
vector<int> doit1(int s)
{
	int l=1;while(l<=s*2+5)l<<=1;
	for(int i=0;i<l;i++)a[i]=b[i]=0;
	for(int i=0;i<=s;i++)a[i]=1ll*ifr[i]*pw(i,s-1)%mod;
	for(int i=0;i<=s;i++)b[i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	for(int i=s+1;i<l;i++)a[i]=0;
	for(int i=s;i>=1;i--)a[i]=1ll*a[i]*fr[i]%mod;
	for(int i=0;i<l;i++)b[i]=0;
	for(int i=s;i>=1;i--)a[i]=(a[i]+a[i-1])%mod;
	for(int i=s+1;i>=1;i--)a[i]=a[i-1];
	a[1]=0;
	for(int i=0;i<=s;i++)b[s-i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	for(int i=1;i<=s+1;i++)a[i]=1ll*a[i]*fr[i-1]%mod;
	dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
	vector<int> as;as.push_back(0);
	for(int i=1;i<=s;i++)as.push_back(1ll*a[i+s+1]*ifr[i-1]%mod*ifr[i]%mod*ifr[s-1]%mod);
	return as;
}
struct sth{vector<int> a,b;};
sth cdq(int l,int r)
{
	if(l==r){return (sth){doit0(s[l]),doit1(s[l])};}
	int mid=(l+r)>>1;
	sth a=cdq(l,mid),b=cdq(mid+1,r);
	return (sth){polymul(a.a,b.a),polyadd(polymul(a.b,b.a),polymul(a.a,b.b))};
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&s[i]);
	pre();vector<int> st=cdq(1,n).b;
	int as=0;for(int i=1;i<st.size();i++)as=(as+1ll*st[i]*fr[i-1])%mod;
	printf("%d\n",as);
}
```



#### NOI2020 模拟测试9

##### auoj827 有向无环图

###### Problem

给一个 $n$ 个点 $m$ 条边的 `DAG`，初始每个点的权值都是 $0$ ，支持 $q$ 次如下类型的操作：

1. 给定 $u,x$，将 `DAG` 上所有 $u$ 可以到达的点的权值改为 $x$。
2. 给定 $u,x$，将 `DAG` 上所有 $u$ 可以到达的点的权值与 $x$ 取 $\min$。
3. 询问一个点的权值

$n,m,q\leq 10^5$

$3s,512MB$

###### Sol

考虑按询问分块，求出每 $kB$ 个时刻时所有点的点权。对于一次询问，从上一个 $kB$ 时刻开始，如果能求出每个询问点能否到达当前点，就能 $O(\sqrt n)$ 求出每一个询问的答案，这部分可以 `bitset` 维护每个点能到达的点处理。

考虑如何从 $kB$ 时刻的点权到 $(k+1)B$ 时刻的点权，如果只有 $1$ 操作，可以 `dfs` 一次求出每个点最后一次被覆盖的时间，这个时间对应的操作权值就是答案。

对于一个时刻 $t$ 在点 $x$ 的 $2$ 操作，它能影响到点 $u$ 当且仅当 $x$ 能到达 $u$ 且 $u$ 最后一次被覆盖的时间小于 $t$。

将所有点按照覆盖时间排序，从后往前考虑每个修改并按照覆盖时间加入点，可以得到每个点能影响的点的集合。

注意到如果按照 $x$ 从小到大考虑每一个 $2$ 操作，那么一个点只会被修改一次。按照权值依次考虑每一个 $2$ 操作，记录当前还有哪些点没有被修改，然后可以求出这次操作可能修改的点的集合。这时每个点只会被改一次，暴力改即可。

上述操作都可以 `bitset` 优化，单个询问复杂度 $O(\frac n{\omega})$。

直接 `bitset` 维护每个点可以到达的点空间不能接受（$O(\frac{n^2}{32})$），可以将所有点分成三部分，一次做一部分。

复杂度 $O(n\sqrt {n\log n}+\frac {n(m+q)}{\omega})$，可以将第一部分的 `dfs` 换成和上面第二步类似的操作去掉排序，但应该不会更快。

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<queue>
#include<algorithm>
using namespace std;
#define N 101500
int as1[256],ct2;
int que(unsigned int fu)
{
	int as=0;
	while(fu>=256)fu>>=8,as+=8;
	return as+as1[fu];
}
struct sth{
	unsigned int st[N][1051];
	void reset(int x){for(int i=1;i<=1050;i++)st[x][i]=0;}
	void modify(int x,int y){int tp=(y>>5)+1,tp2=y-((y>>5)<<5);st[x][tp]|=1u<<tp2;}
	void doit1(int x,int y){for(int i=1;i<=1050;i++)st[x][i]|=st[y][i];}
	void doit2(int x,int y){for(int i=1;i<=1050;i++)st[x][i]&=st[y][i];}
	void doit3(int x,int y){for(int i=1;i<=1050;i++)st[x][i]^=st[y][i];}
	int query(int x,int y){int tp=(y>>5)+1,tp2=y-((y>>5)<<5);return (st[x][tp]>>tp2)&1;}
	void modify2(int x,int y){int tp=(y>>5)+1,tp2=y-((y>>5)<<5);if((st[x][tp]>>tp2)&1)st[x][tp]^=1u<<tp2;}
	void copy(int x,int y){for(int i=1;i<=1050;i++)st[x][i]=st[y][i];}
	vector<int> doit(int x)
	{
		vector<int> as;
		for(int i=1;i<=1050;i++)
		if(st[x][i])
		{
			unsigned int t1=st[x][i];
			int t=(i-1)*32;
			while(t1)
			{
				unsigned int t2=t1&(-t1);
				t1-=t2;as.push_back(t+que(t2));
			}
		}
		return as;
	}
}fu;
int n,m,q,a,b,s[N][3],vl[N],ti[N],v1[N],head[N],cnt,s1[N],L,R,vis[N],as[N],in[N],in2[N],s2[N],ct,f2[N];
bool cmp1(int a,int b){return ti[a]>ti[b];}
bool cmp3(int a,int b){return s[a][2]<s[b][2];}
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
void dfs(int u)
{
	if(vis[u])return;
	if(u>=L&&u<=R)fu.modify(u,u-L);
	for(int i=head[u];i;i=ed[i].next)dfs(ed[i].t),fu.doit1(u,ed[i].t);
	vis[u]=1;
}
void solve(int l,int r)
{
	for(int i=1;i<=n;i++)ti[i]=v1[i]=s1[i]=vis[i]=0;
	for(int i=n+1;i<=n+r-l+3;i++)fu.reset(i);
	for(int i=l;i<=r;i++)
	if(s[i][0]==3)
	{
		if(s[i][1]<L||s[i][1]>R)continue;
		int tp=s[i][1],v1=vl[tp];
		for(int j=l;j<i;j++)
		if(fu.query(s[j][1],s[i][1]-L))
		if(s[j][0]==1)v1=s[j][2];
		else if(s[j][0]==2)v1=min(v1,s[j][2]);
		as[i]=v1;
	}
	else if(s[i][0]==1)ti[s[i][1]]=i,v1[s[i][1]]=s[i][2];
	for(int j=1;j<=n;j++)
	{
		int x=f2[j];
		for(int i=head[x];i;i=ed[i].next)
		if(ti[ed[i].t]<ti[x])ti[ed[i].t]=ti[x],v1[ed[i].t]=v1[x];
	}
	for(int i=L;i<=R;i++)if(v1[i])vl[i]=v1[i];
	for(int i=L;i<=R;i++)s1[i]=i;
	sort(s1+L,s1+R+1,cmp1);
	int lb=L,tp2=n+r-l+2;
	for(int i=l;i<=r;i++)if(s[i][0]==2)fu.copy(n+i-l+1,s[i][1]);
	for(int i=L;i<=R;i++)fu.modify(tp2,i-L);
	for(int i=r;i>=l;i--)
	{
		while(lb<=R&&ti[s1[lb]]>=i)fu.modify2(tp2,s1[lb]-L),lb++;
		if(s[i][0]==2)fu.doit2(n+i-l+1,tp2);
	}
	ct=0;for(int i=l;i<=r;i++)if(s[i][0]==2)s2[++ct]=i;
	sort(s2+1,s2+ct+1,cmp3);
	for(int i=L;i<=R;i++)fu.modify(tp2,i-L);
	for(int i=1;i<=ct;i++)
	{
		fu.doit2(n+s2[i]-l+1,tp2);
		fu.doit3(tp2,n+s2[i]-l+1);
		vector<int> t1=fu.doit(n+s2[i]-l+1);
		for(int j=0;j<t1.size();j++)
		vl[t1[j]+L]=min(vl[t1[j]+L],s[s2[i]][2]);
	}
}
int main()
{
	for(int i=1,as3=0;i<256;i<<=1,as3++)as1[i]=as3;
	scanf("%d%d%d",&n,&m,&q);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b),in[b]++;
	for(int i=1;i<=q;i++)
	{
		scanf("%d%d",&s[i][0],&s[i][1]);
		if(s[i][0]<3)scanf("%d",&s[i][2]);
	}
	int ct4=0;
	queue<int> st;
	for(int i=1;i<=n;i++)if(!in[i])st.push(i);
	while(!st.empty())
	{
		int x=st.front();st.pop();f2[++ct4]=x;
		for(int i=head[x];i;i=ed[i].next)
		{
			in[ed[i].t]--;if(!in[ed[i].t])st.push(ed[i].t);
		}
	}
	for(int i=1;i<=n;i+=33600)
	{
		L=i,R=i+33600-1;
		if(R>n)R=n;
		for(int j=1;j<=n;j++)fu.reset(j),vis[j]=0;
		for(int j=1;j<=n;j++)
		dfs(j);
		for(int j=1;j<=q;j+=1200)solve(j,min(j+1200-1,q));
	}
	for(int i=1;i<=q;i++)if(s[i][0]==3)printf("%d\n",as[i]);
}
```



##### auoj828 序列

###### Problem

给一个长度为 $n$ 的整数序列 $a$ 和 $m$，满足 $a_i\in[0,2^m)$，求有多少个长度为 $n$ 的序列满足如下条件：

1. $b_i\in [0,2^m)$
2. $a_i \and b_i\leq a_{i+1} \and b_{i+1}$
3. $a_i \or b_i\geq a_{i+1} \or b_{i+1}$

这里 $\and,\or$ 指位运算中的按位与/或。

答案模 $10^9+7$

$n\leq 100,m\leq 30$

$2s,1024MB$

###### Sol

设 $a_i \and b_i=c_i,a_i \or b_i=d_i$，限制只和 $c_i,d_i$ 有关。

对于 $a_i$ 中为 $1$ 的一位，$b_i$ 中这一位的值只会影响 $c_i$ 的值，$d_i$ 这一位固定为 $1$。对于 $a_i$ 中为 $0$ 的一位，$b_i$ 这一位的值只会影响 $d_i$ 的值，$c_i$ 这一位固定为 $1$。

因此 $c_i,d_i$ 的取值是独立的，可以发现一组满足要求的 $c_i,d_i$ 可以还原出 $b$，因此考虑分开计算合法的 $c,d$ 数量并相乘，得到的结果即为答案。

考虑 $c_i$ 的 `dp`。考虑 $c_i$ 的最高位，显然最高位满足一段前缀是 $0$，剩下是 $1$。可以枚举分界点，枚举分界点后然后两侧区间互不影响，可以看成子问题。

因此设 $dp_{l,r,k}$ 表示区间 $[l,r]$，只考虑后 $k$ 位时满足条件的 $c_i$ 数量。枚举分界点即可转移。$d_i$ 的转移类似。

复杂度 $O(n^3m)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define mod 1000000007
int n,m,f[N][N],g[N][N],v[N];
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=0;i<=n;i++)for(int j=0;j<=n;j++)f[i][j]=g[i][j]=1;
	for(int i=0;i<m;i++)
	for(int le=n-1;le>=0;le--)
	for(int l=1;l+le<=n;l++)
	{
		int r=l+le;
		int v1=f[l][r],v2=g[l][r];
		for(int j=r;j>=l&&((v[j]>>i)&1);j--)v1=(v1+1ll*f[l][j-1]*f[j][r])%mod;
		for(int j=r;j>=l&&!((v[j]>>i)&1);j--)v2=(v2+1ll*g[l][j-1]*g[j][r])%mod;
		f[l][r]=v1;g[l][r]=v2;
	}
	printf("%d\n",1ll*f[1][n]*g[1][n]%mod);
}
```



##### auoj829 回文串

###### Problem

定义一个串是好的，当且仅当它能被划分成两个非空回文串

给定 $n$ 和字符集大小 $c$ ，求长度不超过 $n$ 的好的串的数量，模 $10^9+7$

$n,c\leq 10^9$

$2s,512MB$

###### Sol

咕了，下次再说。

#### NOI2020 模拟测试10

##### auoj836 计数

###### Problem

给一个长度为 $n$ 的字符串 $s$，求有多少个字符串序列 $t_{1,...,k}$ 满足如下条件：

1. $t_1=s$
2. $|t_i|=|t_{i-1}|-1$
3. $t_i$ 是 $t_{i-1}$ 的子串

答案模 $998244353$

$n\leq 5\times 10^5$

$2s,1024MB$

###### Sol

显然 $t_i$ 是在 $t_{i-1}$ 上删掉开头或者结尾得到的。考虑记录每次删掉的是开头还是结尾，可以得到一个长度为 $n-1$ 的 `LR` 序列。

考虑什么情况下会算重，即两种不同操作序列得到相同的字符串序列。注意到如果删掉开头或结尾得到的字符串相同，那么整个串的所有字符必须全部相同。

那么可以看成一个 $n$ 层的三角形结构，从上往下第 $i$ 层有 $i$ 个点，第 $i$ 层第 $j$ 个点对应 $[j,j+n-i]$。如果当前点对应的串有多种字符，那么可以向两个方向走，否则只能向一个方向走。

所有只包含一种字符的串在三角形中一定形成了若干个底边上的小三角形，可以看成走到这些小三角形后只能向左走。

考虑先任意走，方案数 $2^n-1$。然后考虑多算的情况，即进入三角形后还有向右走的方案。

枚举走进去之前的最后一个点，可能的点的种类数只有 $O(n)$ 个。可以发现一个小三角形之外的点的上方不可能存在另外一个小三角形内的点，因此走到小三角形外任意一个点的方案数都是一个组合数。因此容易算出从一个点走到某个小三角形内部的方案数。

到三角形内部后，多算的情况是有向右走的方案数，可以发现这个结果是 $2^l-l-1$，其中 $l$ 为当前高度。

注意特判所有字符相同的情况，此时整个区域都被三角形覆盖，因此不存在走进去之前的点。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 500050
#define mod 998244353
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int n,fr[N],ifr[N],as,rs[N],p2[N];
char s[N];
int main()
{
	scanf("%s",s+1);n=strlen(s+1);
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[n]=pw(fr[n],mod-2);for(int i=n;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	p2[0]=1;for(int i=1;i<=n;i++)p2[i]=2*p2[i-1]%mod;
	for(int t=0;t<2;t++)
	{
		rs[n]=n;for(int i=n-1;i>=1;i--)rs[i]=s[i]==s[i+1]?rs[i+1]:i;
		for(int i=1;i<=n;i++)if(rs[i]<n)
		{
			int l=i-1,r=n-rs[i]-1;
			as=(as+1ll*fr[l+r]*ifr[l]%mod*ifr[r]%mod*(p2[rs[i]-i+1]-(rs[i]-i+1)-1))%mod;
		}
		for(int i=1;i*2<=n;i++)s[i]^=s[n+1-i]^=s[i]^=s[n+1-i];
	}
	as=(p2[n]-1-as+1ll*mod)%mod;
	if(rs[1]==n)as=n;
	printf("%d\n",as);
}
```



##### auoj837 博弈

###### Problem

有 $n$ 个人分 $m$ 枚硬币，使用如下方式：

所有人按照 $1,2,\cdots,n,1,2,\cdots,n,\cdots$ 的顺序进行提议，在提议时，当前的人会给出一个硬币的分配方案，然后所有人进行表决。

如果所有人都同意这个提议，则按照这个提议分配，结束过程。否则丢弃 $k$ 枚硬币，换下一个人进行提议。

在进行表决时，对于一个人，如果否决这个表决后他得到的硬币数量不少于当前方案他得到的硬币数量，则否决提议，否则通过提议。

在进行提议时，每个人会最大化最后自己获得的硬币数。

求出最后所有人获得的硬币数。

$n\leq 2\times 10^5,m,k\leq 10^{18}$

$2s,1024MB$

###### Sol

设 $f_{s,x}$ 表示剩余 $s$ 枚硬币时，每个人最后能得到多少枚硬币，$g_s$ 表示此时所有人得到的硬币数和。

假设当前还有 $a$ 枚硬币，如果 $a<g_{a-k}+n$，那么无论怎么分，都有人得到的硬币数量不多于否决这一次后得到的数量，因此这次一定会被否决，即 $f_a=f_{a-k},g_a=g_{a-k}$。

否则，当前分的人可以选择给每人多一枚硬币，多出来的部分给自己，这样最优。

设 $d=\lfloor\frac mk\rfloor$，考虑先计算 $f_{m-dk}$，然后依次递推 $f_{m-(d-1)k},...,f_m$。记录当前的 $a-g_{a-k}=s$，则初始 $s=m-dk$，接下来每次倒推的操作相当于：

1. 如果 $s\geq n$，所有人得到的硬币数加一，当前操作的人再得到 $s-n$ 枚硬币，然后令 $s=0$。
2. 将 $s$ 加上 $k$，操作的人变成上一个人。

可以发现，在进行了第一次 $1$ 操作后，之后每次一定是向前 $\lceil\frac nk\rceil$ 位，然后进行一次 $1$ 操作，从而这部分每次 $1$ 操作时的 $s$ 是相同的。每次操作为向前若干步，因此可以找到循环节，这样就可以算出每个位置额外加了多少以及总共加了多少次 $1$。

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define ll long long
#define N 200010
ll n,m,k,v[N],lz,st[N],ct;
int main()
{
	scanf("%lld%lld%lld",&n,&m,&k);
	ll tp=m/k,ls=m%k,nw=tp%n+1;
	ll fu=(n-ls-1)/k+1;if(fu<0)fu=0;
	if(n<ls)fu=0;
	ls+=k*fu;tp-=fu;nw-=fu;
	if(tp<0){for(int i=1;i<=n;i++)printf("0 ");return 0;}
	nw=(nw%n+n-1)%n+1;
	lz=1;v[nw]=ls-n;ls=0;
	ll v1=(n-1)/k+1,v2=v1*k-n;
	nw=(nw-v1+n-1)%n+1;
	tp/=v1;lz+=tp;
	st[0]=nw;ct=0;
	for(int i=1;i<=2e5;i++)st[i]=(st[i-1]-v1+n-1)%n+1;
	int f1=0;
	for(int i=2e5;i>0;i--)if(st[i]==nw)f1=i;
	for(int i=0;i<f1&&i<tp;i++)v[st[i]]+=((tp-i-1)/f1+1)*v2;
	for(int i=1;i<=n;i++)printf("%lld ",v[i]+lz);
}
```



##### auoj838 划分

###### Problem

给定 $n,k$ 和一个长度为 $n$ 的序列 $a$。你需要将序列划分成 $k$ 段，记第 $i$ 段的元素和为 $s_i$，最大值为 $m_i$，你需要满足：

$$
\forall i,\max(m_i,m_{i+1})\leq |s_i-s_{i+1}|
$$

输出一个方案或输出无解。

$n,k\leq 2\times 10^5$

$4s,1024MB$

###### Sol

考虑一个方案如果不合法，那么对于不合法的两段，一定可以将一段一侧的一个元素给另外一段，使得两段的差更接近。可以发现，这样调整后所有段的平方和会严格变小。

因此考虑找到 $\sum s_i^2$ 最小的划分方式，它一定合法。问题变为找到最小的划分成 $k$ 段的方式。

如果只求最小值，则可以直接 `wqs` 二分+斜率优化求出。

`wqs` 二分不一定能直接求出方案。但如果转移满足四边形不等式，则一定可以求出方案。

具体来说，`wqs` 二分可以求出一个 $k$，使得如果将每一段的代价增加 $k$ 后，最优方案中，段数最少的方案需要 $l$ 段，最大的方案需要 $r$ 段，且 $l\leq k\leq r$。

如果 $l=k$ 或者 $r=k$，则直接找到了方案，否则考虑用最大最小的方案去构造。

此时 $r>l+1$，因此一定存在一个 $l$ 方案的段完全包含 $r$ 方案的段。设两段为 $[l_1,r_1],[l_2,r_2](l_1\leq l_2\leq r_2\leq r_1)$，则由四边形不等式，$[l_1,r_2],[l_2,r_1]$ 两个转移次数和小于等于 $[l_1,r_1],[l_2,r_2]$ 的和。考虑取一个划分，左侧取 $l$ 方案，中间取 $[l_1,r_2]$，右侧取 $r$ 方案，另外一个划分中间取 $[l_2,r_1]$，两侧类似，则这两个方案的代价和小于等于之前两个方案的代价和。因此这两个方案都是最优方案。

这样就可以用之前的方案构造新的方案，从左向右考虑，可以发现一定有一个位置能构造出 $k$，这样就得到了方案。

复杂度 $O(n\log v)$

注意这里序列可能有 $0$，因此斜率优化处有一些细节，求段数最小/最大的方案更需要注意细节。

###### Code

看起来非常正确的写法：

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define ll __int128
int n,k,v[N],fr[N],ct[N],c1,c2,s1[N],s2[N],st[N],rb,lb;
ll dp[N],su[N];
void solve(ll v,int fg)
{
	st[lb=rb=1]=0;
	for(int i=1;i<=n;i++)
	{
		while(lb+1<=rb&&dp[st[lb+1]]+(su[i]-su[st[lb+1]])*(su[i]-su[st[lb+1]])<dp[st[lb]]+(su[i]-su[st[lb]])*(su[i]-su[st[lb]])+fg)lb++;
		fr[i]=st[lb],ct[i]=ct[st[lb]]+1;dp[i]=dp[st[lb]]+(su[i]-su[st[lb]])*(su[i]-su[st[lb]])+v;
		while(lb+1<=rb&&(dp[i]+su[i]*su[i]-dp[st[rb]]-su[st[rb]]*su[st[rb]])*(su[st[rb]]-su[st[rb-1]])<(dp[st[rb]]+su[st[rb]]*su[st[rb]]-dp[st[rb-1]]-su[st[rb-1]]*su[st[rb-1]])*(su[i]-su[st[rb]])+fg)rb--;
		if(su[st[rb]]!=su[i])st[++rb]=i;
		if(su[i]==su[i-1]&&v-fg<0)st[rb]=i,fr[i]=i-1;
	}
	int s2=n;c1=0;
	while(s2)s1[++c1]=s2,s2=fr[s2];
	for(int i=1;i*2<=c1;i++)swap(s1[i],s1[c1-i+1]);
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),su[i]=su[i-1]+v[i];
	ll lb=-1e24,rb=1e24,as=-1e24;
	while(lb<=rb)
	{
		ll mid=(lb+rb)/2;
		solve(mid,0);
		if(c1<=k)rb=mid-1,as=mid;
		else lb=mid+1;
	}
	solve(as,1);
	c2=c1;for(int i=1;i<=c2;i++)s2[i]=s1[i];
	solve(as,0);
	int l1=1,v1,v2;
	for(int i=0;i<=c2;i++)
	{
		while(s1[l1]<=s2[i]&&l1<c1)l1++;
		if(i+c1-l1+1==k)v1=i,v2=l1;
	}
	for(int j=1;j<=v1;j++)printf("%d ",s2[j]);
	for(int j=v2;j<c1;j++)printf("%d ",s1[j]);
}
```

