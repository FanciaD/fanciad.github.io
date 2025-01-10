---
title: 2020 ZR集训 题解
date: '2021-02-22 22:23:03'
updated: '2021-02-22 22:23:03'
tags: Mildia
permalink: Feast-Kyouen/
description: 2020 ZR集训
mathjax: true
---

Note. 本篇的完成时间已不可考，此处时间为上一个 blog 建立的时间。


### ZROI2020 部分题目总结

###### 前言

暂时只有我打过的场

目前咕咕咕的：十连测 day 1 day 5 day 7 三月集训 day 3

目前不想写的：十连测 day 4 

#### 十连测 day 2

##### T1 选拔赛

###### Problem

有 $n$ 个人参加了一场选拔赛，这场选拔赛会进行若干轮，每轮会出一道题让大家做，做不出来的选手被淘汰，其他选手进入下一轮。特别地，如果所有人都没做出这道题，那么就没人被淘汰。选拔赛会进行到只剩下一个选手为止

现在已知每个选手做出题的概率都是 $p$，由于主办方经费有限，所以主办方想知道期望进行几轮能结束比赛

你需要输出答案对 $998244353$ 取模后的值

$n\leq 10^5$

###### Sol

设 $f_i$ 表示剩下 $i$ 个人时的期望轮数

那么有 $f_i=1+\sum_{j=1}^{i-1}C_i^j*p^j*(1-p)^{i-j}*f_j+(p^i+(1-p)^i)f_i$

$f_i=(1+\sum_{j=1}^{i-1}C_i^j*p^j*(1-p)^{i-j}*f_j)/(1-p^i-(1-p)^i)$

$f_i=(1+(1-p)^i\sum_{j=1}^{i-1}C_i^j*(p/(1-p))^j*f_j)/(1-p^i-(1-p)^i)$

注意到中间是一个卷积形式，分治FFT即可

复杂度 $O(n\log^2n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 263000
#define mod 998244353
int n,p,v1,v2,a[N],b[N],c[N],ntt[N],rev[N],fr[N],ifr[N];
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+(i&1)*s/2,ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int st=pw(3,t==-1?mod-1-(mod-1)/i:(mod-1)/i);
		for(int j=0;j<s;j+=i)
		for(int k=j,s1=1;k<j+(i>>1);k++,s1=1ll*s1*st%mod)
		{
			int a1=ntt[k],a2=1ll*ntt[k+(i>>1)]*s1%mod;
			ntt[k]=(a1+a2)%mod;
			ntt[k+(i>>1)]=(a1-a2+mod)%mod;
		}
	}
	int inv=pw(s,t==-1?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void cdq(int l,int r)
{
	if(l==r)
	{
		int tp=(1-pw(p,l)-pw(mod+1-p,l))%mod;
		tp=(tp+mod)%mod;
		a[l]=1ll*(a[l]+ifr[l])*pw(tp,mod-2)%mod;
		return;
	}
	int mid=(l+r)>>1;
	cdq(l,mid);
	int s=1;while(s<=(r-l+1)*1.5+10)s<<=1;
	for(int i=0;i<s;i++)b[i]=c[i]=0;
	for(int i=l;i<=mid;i++)b[i-l]=1ll*a[i]*pw(p,i)%mod;
	for(int i=l;i<=r;i++)c[i-l]=1ll*pw(mod+1-p,i-l)*ifr[i-l]%mod;
	dft(s,b,1);dft(s,c,1);for(int i=0;i<s;i++)b[i]=1ll*b[i]*c[i]%mod;dft(s,b,-1);
	for(int i=mid+1;i<=r;i++)a[i]=(a[i]+b[i-l])%mod;
	cdq(mid+1,r);
}
int main()
{
	scanf("%d%d%d",&n,&v1,&v2);p=1ll*v1*pw(v2,mod-2)%mod;
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	cdq(1,n);printf("%lld\n",1ll*fr[n]*a[n]%mod);
}
```

##### T2 bitrev

###### Problem

设 $g(x)$ 为将 $x$ 的二进制表示进行翻转后得到的数。

也就是如果 $x$ 的二进制表示是 $a_1a_2...a_n$（其中 $a1=1$），那么他翻转过来就是 $a_na_{n−1}...a_1$

例如 $g(2)=1$，$g(6)=3$，$g(10)=5$

设 $f(x)$ 为 $x$ 的二进制表示下 $1$ 的个数

给定 $R$，求 $\sum_{i=1}^Rf(i+g(i))$

$R\leq 10^{14}$

###### Sol

考虑枚举二进制位数然后统计

首先考虑位数小于 $log_2 R$ 的情况，这种情况对位没有其它的限制

考虑数位dp,但因为rev的原因，需要从开头和结尾同时dp

这时相当于从开头和结尾开始确定 $i+g(i)$ 的每一位，但开头还需要考虑后面的进位

设 $dp_{i,j,k,0/1}$ 表示考虑了前后 $i$ 位，如果第 $l-i-1$ 位向第 $l-i$ 位有进位，那么当前第 $l-i$ 位及以前和后 $i$ 位有 $k$ 个1，如果没有进位有 $j$ 个1，当前第 $i$ 位有没有向上进位

最后枚举中间一个或者两个算答案

然后考虑位数等于的情况，这时需要考虑限制

设 $dp_{i,j,k,0/1,0/1,0/1}$ ，最后两个分别表示前 $i$ 维与 $R$ 的关系是小于/等于，后 $i$ 位是小于等于/大于

复杂度 $O(\log^4 R * 2^4 + log^3 R *2^7)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 62
long long dp[N][N][N][2],dp2[N][N][N][2][2][3];
long long n;
long long solve1(int l)
{
	memset(dp,0,sizeof(dp));
	long long as=0;
	dp[0][0][1][0]=1;
	for(int i=1;i*2<l;i++)
	for(int j=0;j<=i*2+4;j++)
	for(int k=0;k<=i*2+4;k++)
	for(int s=0;s<2;s++)
	if(dp[i-1][j][k][s])
	for(int v1=0;v1<2;v1++)
	for(int v2=0;v2<2;v2++)
	{
		if(i==1&&v1==0)continue;
		int fuc=v1+v2;
		int nj=j,nk=k,ns=0;
		int t1=fuc+s,fg;
		ns=(t1>=2);fg=t1&1;
		if(fuc>=2)nj=k+fg+(fuc&1);
		else nj=j+fg+(fuc&1);
		fuc++;
		if(fuc>=2)nk=k+fg+(fuc&1);
		else nk=j+fg+(fuc&1);
		dp[i][nj][nk][ns]=dp[i][nj][nk][ns]+dp[i-1][j][k][s];
	}
	int fu=(l-1)>>1;
	if(l&1)
	for(int j=0;j<=fu*2+4;j++)
	for(int k=0;k<=fu*2+4;k++)
	for(int s=0;s<2;s++)
	for(int v=0;v<2;v++)
	{
		if(!fu&&!v)continue;
		int v2=2*v+s;
		if(v2>=2)
		{
			int as1=k+(v2&1);
			as=as+as1*dp[fu][j][k][s];
		}
		else
		{
			int as1=j+(v2&1);
			as=as+as1*dp[fu][j][k][s];
		}
	}
	else
	for(int j=0;j<=fu*2+4;j++)
	for(int k=0;k<=fu*2+4;k++)
	for(int s=0;s<2;s++)
	for(int v=0;v<2;v++)
	for(int v2=0;v2<2;v2++)
	{
		if(!fu&&!v)continue;
		int s1=v+v2,s2=v+v2+s,tp=0;
		if(s2>=2)s1++,s2-=2;
		tp=(s1&1)+(s2&1);
		if(s1>=2)tp+=k;else tp+=j;
		as=as+tp*dp[fu][j][k][s];
	}
	return as;
}
long long solve2(int l)
{
	long long as=0;
	dp2[0][0][1][0][1][1]=1;
	for(int i=1;i*2<l;i++)
	for(int j=0;j<=i*2+4;j++)
	for(int k=0;k<=i*2+4;k++)
	for(int s=0;s<2;s++)
	for(int f1=0;f1<2;f1++)//0 < 1 =
	for(int f2=0;f2<3;f2++)// 0 < 1 = 2 >
	if(dp2[i-1][j][k][s][f1][f2])
	for(int v1=0;v1<2;v1++)
	for(int v2=0;v2<2;v2++)
	{
		if(i==1&&v1==0)continue;
		int nf1=f1,nf2=f2;
		if((n&(1ll<<l-i))&&!v1)nf1=0;
		if(!(n&(1ll<<l-i))&&v1&&f1)continue;
		if((n&(1ll<<i-1))&&!v2)nf2=0;
		if(!(n&(1ll<<i-1))&&v2)nf2=2;
		int fuc=v1+v2;
		int nj=j,nk=k,ns=0;
		int t1=fuc+s,fg;
		ns=(t1>=2);fg=t1&1;
		if(fuc>=2)nj=k+fg+(fuc&1);
		else nj=j+fg+(fuc&1);
		fuc++;
		if(fuc>=2)nk=k+fg+(fuc&1);
		else nk=j+fg+(fuc&1);
		dp2[i][nj][nk][ns][nf1][nf2]=dp2[i][nj][nk][ns][nf1][nf2]+dp2[i-1][j][k][s][f1][f2];
	}
	int fu=(l-1)>>1;
	if(l&1)
	for(int j=0;j<=fu*2+4;j++)
	for(int k=0;k<=fu*2+4;k++)
	for(int s=0;s<2;s++)
	for(int v=0;v<2;v++)
	for(int f1=0;f1<2;f1++)
	for(int f2=0;f2<3;f2++)
	{
	if(dp2[fu][j][k][s][f1][f2])
	{
		if(!fu&&!v)continue;
		int nf2=f2;
		if((n&(1ll<<fu))&&!v)nf2=0;
		if(!(n&(1ll<<fu))&&v)nf2=2;
		if(f1&&nf2==2)continue;
		int v2=2*v+s;
		if(v2>=2)
		{
			int as1=k+(v2&1);
			as=as+as1*dp2[fu][j][k][s][f1][f2];
		}
		else
		{
			int as1=j+(v2&1);
			as=as+as1*dp2[fu][j][k][s][f1][f2];
		}
	}
	}
	else
	for(int j=0;j<=fu*2+4;j++)
	for(int k=0;k<=fu*2+4;k++)
	for(int s=0;s<2;s++)
	for(int f1=0;f1<2;f1++)
	for(int f2=0;f2<3;f2++)
	if(dp2[fu][j][k][s][f1][f2])
	for(int v=0;v<2;v++)
	for(int v2=0;v2<2;v2++)
	{
		if(!fu&&!v)continue;
		int nf2=f2;
		if((n&(1ll<<fu))&&!v2)nf2=0;
		if(!(n&(1ll<<fu))&&v2)nf2=2;
		if((n&(1ll<<fu+1))&&!v)nf2=0;
		if(!(n&(1ll<<fu+1))&&v)nf2=2;
		if(f1&&nf2==2)continue;
		int s1=v+v2,s2=v+v2+s,tp=0;
		if(s2>=2)s1++,s2-=2;
		tp=(s1&1)+(s2&1);
		if(s1>=2)tp+=k;else tp+=j;
		as=as+tp*dp2[fu][j][k][s][f1][f2];
	}
	return as;
}
int main()
{
	scanf("%lld",&n);
	int l=1;while((1ll<<l)-1<n)l++;
	long long as=solve2(l);
	for(int i=1;i<l;i++)as=as+solve1(i);
	printf("%lld\n",as);
}
```

##### T3 LCA on tree

###### Problem

给定一棵 $n$ 个点的有根树，其中 $1$ 号点是根，定义 $f(x)=\sum_{i\leq j,lca(i,j)=x}(w_i+w_j)$，其中 $lca(x,y)$ 为两个点的最近公共祖先。

现在你需要支持两种操作：

- 1 x v：对于所有 x 子树内和 x 距离不超过 2 的点 y，令 $w_y=w_y+v$
- 2 x ：询问 $f(x)$ 的值

由于答案可能很大，你只需要输出答案对 $2^{32}$ 取模后的值即可。

$n\leq 3\times10^5,1s$

###### Sol

考虑1操作一个点对其它点答案的影响

对于 $x$ 的儿子的儿子 $y$ ,贡献是 $size_y*v$

对于 $x$ 的儿子 $y$ ,贡献是 $(\sum_{z\in son_y}(size_y-size_z)+size_y)*v$

对于 $x$ 贡献是 $(\sum_{z\in son_x}(|son_z|+1)*(size_x-size_z)+size_x)*v$

这三个的系数都可以预处理

对于 $x$ 的祖先 $y$ ，贡献是 $(size_y-size_z)*su*d$ ，其中 $z\in son_y$ 的子树包含 $x$ ， $su$ 为 $x$ 子树内和 $x$ 距离不超过 $2$ 的点数量

考虑树剖，对于每个点维护重儿子对应的系数，那么有 $\log n$ 个点的系数需要修改，对于其余的，相当于到一条到根的链加，单点求和，可以变为单点加区间求和，使用树状数组维护

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 305000
#define ui unsigned long long
int n,q,head[N],cnt,sz[N],tp[N],son[N],id[N],tid[N],f[N],ct,lb[N],rb[N],v[N],a,b,c;
ui vl1[N],vl2[N],vl3[N],as[N],lz[N],s1[N],s2[N],sz1[N];
ui tr[N];
void add(int x,ui b){for(int i=x;i<=n;i+=i&-i)tr[i]+=b;}
ui que(int x){if(x<0)return 0;ui as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa)
{
	sz[u]=1;f[u]=fa;s2[u]=s1[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs1(ed[i].t,u);
		if(sz[ed[i].t]>sz[son[u]])son[u]=ed[i].t;
		sz[u]+=sz[ed[i].t],sz1[u]+=sz1[ed[i].t];
		s1[u]++;s2[u]+=s1[ed[i].t];
	}
	vl1[u]=vl2[u]=vl3[u]=sz[u]+1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)vl2[u]+=(sz[u]-sz[ed[i].t]),vl3[u]+=(sz[u]-sz[ed[i].t])*s1[ed[i].t];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=son[u])as[u]+=sz1[ed[i].t]*(sz[u]-sz[ed[i].t]);
}
void dfs2(int u,int v,int fa)
{
	lb[u]=id[u]=++ct;tid[ct]=u;tp[u]=v;
	if(son[u])dfs2(son[u],v,u);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=son[u])dfs2(ed[i].t,ed[i].t,u);
	rb[u]=ct;
}
void modify(int x,int v)
{
	lz[x]+=v;
	ui tp1=v*s2[x];
	int st=x;
	add(id[st],tp1);
	while(st)
	{
		st=tp[st];
		as[f[st]]+=tp1*(sz[f[st]]-sz[st]);
		st=f[st];
	}
}
ui query(int x)
{
	ui tp1=que(rb[son[x]])-que(id[son[x]]-1);
	ui as1=as[x]+tp1*(sz[x]-sz[son[x]])+v[x]*(sz[x]+1);
	as1+=vl1[x]*lz[f[f[x]]]+vl2[x]*lz[f[x]]+vl3[x]*lz[x];
	return as1;
}
int rd()
{
	char f=getchar();
	int as=0;
	while(f<'0'||f>'9')f=getchar();
	while(!(f<'0'||f>'9'))as=as*10+f-'0',f=getchar();
	return as;
}
int main()
{
	n=rd();q=rd();
	for(int i=2;i<=n;i++)a=rd(),adde(a,i);
	for(int i=1;i<=n;i++)v[i]=rd(),sz1[i]=v[i];
	dfs1(1,0);dfs2(1,1,0);
	for(int i=1;i<=n;i++)add(id[i],v[i]);
	while(q--)
	{
		a=rd();
		if(a==1)b=rd(),c=rd(),modify(b,c);
		else b=rd(),printf("%u\n",query(b));
	}
}
```

#### 十连测 day 3

##### T1 选拔赛

###### Problem

有 $n$ 个人，每个人有两个分数 $a_i,b_i$ ,保证所有 $a_i$ 互不相同，所有 $b_i$ 互不相同。你现在知道所有 $a_i$ ,以及 $b_i$ 打乱之后的 $c_i$ ，求有多少种 $b_i$ 与 $c_i$ 的对应方式满足

对于每一个 $a_i$ 在所有 $a_i$ 中排前 $k$ 大的选手，他的 $a_i+b_i$ 不低于任意一个 $a_i$ 在所有 $a_i$ 中排后 $n-k$ 大的选手的 $a_i+b_i$

模 $998244353$

$n,k\leq 100$

###### Sol

只有不超过 $n^2$ 种 $a_i+b_i$ ,考虑枚举前 $k$ 个人中最小的 $a_i+b_i$

考虑容斥，算出前 $k$ 个人都大于等于 $s$ 且后 $n-k$ 个都小于等于 $s$ 的方案数，减去前 $k$ 个人都大于 $s$ 且后 $n-k$ 个都小于等于 $s$ 的方案数就是前 $k$ 个人的min是 $s$ 且合法的方案数

对于每一次计算，发现前 $k$ 个人可以取的都是 $c$ 排序后的一段前缀，后 $n-k$ 个可以取的是一段后缀

如果只有前缀是很好处理的，就是 $\prod (r_i-i+1)$

考虑对后面的容斥，变成所有的减去一段前缀

将所有可能的 $n$ 种前缀排序，对于前 $k$ 个对应的限制必须要满足，对于其余的可以选择满足或者不管

设 $dp_{i,j}$ 表示考虑了前 $i$ 个限制，满足了 $j$ 个的方案数和,转移时乘上这个限制对应的 $r_i-j+1$ 的系数

最后考虑那些容斥时没有选的限制，它们都可以任选，因此答案是 $\sum dp_{n,j}*(n-j)!*(-1)^{j-i}$

复杂度 $O(n^4)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 105
#define M 10403
#define mod 1000000007
int n,q,a[N],b[N],dp[N][N],fr[N],su[M],as;
struct lim{int x,v;friend bool operator <(lim a,lim b){return a.x==b.x?a.v>b.v:a.x<b.x;}}s[N];
int main()
{
	scanf("%d%d",&n,&q);q=n-q;
	fr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod;
	if(!q){printf("%d\n",fr[n]);return 0;}
	for(int i=1;i<=n;i++)scanf("%d",&a[i]);
	for(int i=1;i<=n;i++)scanf("%d",&b[i]);
	sort(a+1,a+n+1);
	sort(b+1,b+n+1);
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)su[i*n-n+j]=a[i]+b[j];
	sort(su+1,su+n*n+1);
	for(int i=1;i<=n*n;i++)
	if(i==1||su[i]!=su[i-1])
	{
		for(int j=1;j<=n;j++)
		{
			if(j<=q)
			{
				int lb=1,rb=n,as=0;
				while(lb<=rb)
				{
					int mid=(lb+rb)>>1;
					if(a[j]+b[mid]<=su[i])as=mid,lb=mid+1;
					else rb=mid-1;
				}
				s[j]=(lim){as,1};
			}
			else
			{
				int lb=1,rb=n,as=1;
				while(lb<=rb)
				{
					int mid=(lb+rb)>>1;
					if(a[j]+b[mid]>=su[i])as=mid,rb=mid-1;
					else lb=mid+1;
				}
				s[j]=(lim){as,2};
			}
		}
		sort(s+1,s+n+1);
		memset(dp,0,sizeof(dp));
		dp[0][0]=1;
		for(int j=1;j<=n;j++)
		for(int k=0;k<=j;k++)
		if(dp[j-1][k])
		{
			if(s[j].v==1)dp[j][k+1]=(dp[j][k+1]+1ll*dp[j-1][k]*(s[j].x-k))%mod;
			else
			{
				dp[j][k+1]=(dp[j][k+1]+1ll*dp[j-1][k]*(s[j].x-1-k)%mod*(mod-1))%mod;
				dp[j][k]=(dp[j][k]+dp[j-1][k])%mod;
			}
		}
		int as1=0;
		for(int j=0;j<=n;j++)as1=(as1+1ll*dp[n][j]*fr[n-j])%mod;
		as=(as+as1)%mod;
		for(int j=1;j<=n;j++)
		{
			if(j<=q)
			{
				int lb=1,rb=n,as=0;
				while(lb<=rb)
				{
					int mid=(lb+rb)>>1;
					if(a[j]+b[mid]<su[i])as=mid,lb=mid+1;
					else rb=mid-1;
				}
				s[j]=(lim){as,1};
			}
			else
			{
				int lb=1,rb=n,as=1;
				while(lb<=rb)
				{
					int mid=(lb+rb)>>1;
					if(a[j]+b[mid]>=su[i])as=mid,rb=mid-1;
					else lb=mid+1;
				}
				s[j]=(lim){as,2};
			}
		}
		sort(s+1,s+n+1);
		memset(dp,0,sizeof(dp));
		dp[0][0]=1;
		for(int j=1;j<=n;j++)
		for(int k=0;k<=j;k++)
		if(dp[j-1][k])
		{
			if(s[j].v==1)dp[j][k+1]=(dp[j][k+1]+1ll*dp[j-1][k]*(s[j].x-k))%mod;
			else
			{
				dp[j][k+1]=(dp[j][k+1]+1ll*dp[j-1][k]*(s[j].x-1-k)%mod*(mod-1))%mod;
				dp[j][k]=(dp[j][k]+dp[j-1][k])%mod;
			}
		}
		as1=0;
		for(int j=0;j<=n;j++)as1=(as1+1ll*dp[n][j]*fr[n-j])%mod;
		as=(as-as1+mod)%mod;
	}
	printf("%d\n",as);
}
```

##### T2 跳跃

###### Problem

有长度为 $n$ 的序列 $a$

你能从 $i$ 跳到 $j$ 当且仅当 $|i-j|\leq a_i$

定义 $d_{x->y}$ 为从 $x$ 跳到 $y$ 的最小步数

你想要找到一对 $x,y$ ,最大化 $min(d_{x->y},d_{y->x})$ ，输出这个最大值

$n\leq 2\times 10^5$

###### Sol

每个点跳 $k$ 步能走到的一定是一个区间

考虑二分答案，假设对于每个点求出了它能到达的区间 $[l_i,r_i]$ ，只需要判断有没有一对 $x,y$ ,满足 $x\notin[l_y,r_y],y\notin[l_x,,r_x]$

设 $x<y$ ，考虑枚举 $x$ ，那么只需要满足 $x<l_y,y>r_x$

相当于求出 $y>r_x$ 中最大的 $l_y$ ，处理后缀max即可

这一部分复杂度为 $O(n\log n)$

考虑倍增求这个，设 $l_{i,j}$ 表示 $i$ 跳 $2^j$ 步后跳到的区间左边界， $r_{i,j}$ 同理，那么有 $l_{i,j+1}=min_{k=l_{i,j}}^{r_{i,j}}l_{k,j}$, $r$ 同理

这一部分可以对每一层RMQ解决

将每一层的RMQ存下来，就可以 $O(n\log n)$ 求出 $k$ 步时每一个位置的 $l,r$ 

复杂度 $O(n\log^2 n)$

~~如果写OnO1RMQ+倍增代替二分答案是可以 $O(n\log n)$ 的~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200500
#define K 18
int lg[N],l[N][K],r[N][K],st[2][K][K][N],n,a,l1[N],r1[N],rmx[N];
int quemin(int a,int b,int l,int r)
{
	int tp=lg[r-l+1];
	return min(st[b][a][tp][l],st[b][a][tp][r-(1<<tp)+1]);
}
int quemax(int a,int b,int l,int r)
{
	int tp=lg[r-l+1];
	return max(st[b][a][tp][l],st[b][a][tp][r-(1<<tp)+1]);
}
void pre(int k)
{
	for(int i=1;i<=n;i++)st[0][k][0][i]=l[i][k],st[1][k][0][i]=r[i][k];
	for(int i=1;i<=lg[n];i++)
	for(int j=1;j+(1<<i)-1<=n;j++)
	st[0][k][i][j]=min(st[0][k][i-1][j],st[0][k][i-1][j+(1<<i-1)]);
	for(int i=1;i<=lg[n];i++)
	for(int j=1;j+(1<<i)-1<=n;j++)
	st[1][k][i][j]=max(st[1][k][i-1][j],st[1][k][i-1][j+(1<<i-1)]);
}
bool check(int mid)
{
	for(int i=1;i<=n;i++)l1[i]=r1[i]=i;
	for(int i=0;i<=lg[n];i++)if(mid&(1<<i))for(int j=1;j<=n;j++){int f1=quemin(i,0,l1[j],r1[j]),f2=quemax(i,1,l1[j],r1[j]);l1[j]=f1,r1[j]=f2;}
	rmx[n+1]=0;
	for(int i=n;i>=1;i--)rmx[i]=max(rmx[i+1],l1[i]);
	for(int i=1;i<=n;i++)if(rmx[r1[i]+1]>i)return 1;
	return 0;
}
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)lg[i]=lg[i>>1]+1;
	for(int i=1;i<=n;i++)scanf("%d",&a),l[i][0]=max(i-a,1),r[i][0]=min(i+a,n);
	pre(0);
	for(int i=1;i<=lg[n];i++)
	{
		for(int j=1;j<=n;j++)l[j][i]=quemin(i-1,0,l[j][i-1],r[j][i-1]),r[j][i]=quemax(i-1,1,l[j][i-1],r[j][i-1]);
		pre(i);
	}
	int lb=1,rb=n,as=0;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(check(mid))as=mid,lb=mid+1;
		else rb=mid-1;
	}
	printf("%d\n",as+1);
}
```

##### T3 切蛋糕

###### Problem

有一个半径为 $R$ 的圆，在上面切 $n$ 刀，每一刀给定离圆心的最近距离和最近点的角度，然后切掉没有圆心的部分

![](C:\Users\zz\Documents\pic\53.png)

你需要求出最后蛋糕边界直线部分和圆周部分的长度

$n\leq 20,R\leq 5$

###### Sol

只考虑每一刀的直线，它们组成了一个多边形，这一部分可以半平面交解决

为了防止奇怪的情况，可以在圆外面加一个正方形的边界

因为这是一个包含圆心的凸多边形，所以可以分别考虑每一条线段的贡献

那么只需要求线段与圆的交

这东西可以三分最近点+二分交点解决，注意判一些情况

复杂度 $O(n^2+n\log R)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 105
struct point{double x,y;}q2[N],p[N];
point operator +(point a,point b){return (point){a.x+b.x,a.y+b.y};}
point operator -(point a,point b){return (point){a.x-b.x,a.y-b.y};}
point operator *(point a,double b){return (point){a.x*b,a.y*b};}
struct vec{point s,t;}ed[N],q[N];
int hd=1,tl,n,r,d,ct,ct1;
double cross(point x,point y,point z,point l){return (y.x-x.x)*(l.y-z.y)-(l.x-z.x)*(y.y-x.y);}
bool cmp(vec a,vec b)
{
	double tmp=atan2(a.t.y-a.s.y,a.t.x-a.s.x),tmp2=atan2(b.t.y-b.s.y,b.t.x-b.s.x);
	if(abs(tmp-tmp2)<=1e-8)return cross(a.s,a.t,a.s,b.t)<0;
	return tmp<tmp2;
}
point ins(vec a,vec b){return a.s+(a.t-a.s)*(cross(a.s,b.s,a.s,b.t)/(cross(a.s,b.s,a.s,a.t)+cross(a.s,a.t,a.s,b.t)));}
bool si()
{
	int s1=ct1;
	ed[++s1]=(vec){r*2,r*2,-r*2,r*2};
	ed[++s1]=(vec){-r*2,r*2,-r*2,-r*2};
	ed[++s1]=(vec){-r*2,-r*2,r*2,-r*2};
	ed[++s1]=(vec){r*2,-r*2,r*2,r*2};
	sort(ed+1,ed+s1+1,cmp);
	q[1]=ed[1];hd=tl=1;
	for(int i=2;i<=s1;i++)
	if(abs(atan2(ed[i-1].t.y-ed[i-1].s.y,ed[i-1].t.x-ed[i-1].s.x)-atan2(ed[i].t.y-ed[i].s.y,ed[i].t.x-ed[i].s.x))>=1e-8)
	{
		while(hd<tl&&cross(ed[i].s,ed[i].t,ed[i].s,q2[tl-1])<0)tl--;
		while(hd<tl&&cross(ed[i].s,ed[i].t,ed[i].s,q2[hd])<0)hd++;
		q[++tl]=ed[i];
		if(hd<tl)q2[tl-1]=ins(q[tl-1],q[tl]);
	}
	while(hd<tl&&cross(q[hd].s,q[hd].t,q[hd].s,q2[tl-1])<0)tl--;
	while(hd<tl&&cross(q[tl].s,q[tl].t,q[tl].s,q2[hd])<0)hd++;
	if(tl-hd<=1)return 0;
	q2[tl]=ins(q[hd],q[tl]);
	int ct=0;
	for(int i=hd;i<=tl;i++)
	{
		point a=q2[i],b=q2[i==tl?hd:i+1];
		if(!(abs(a.x-b.x)<=1e-7&&abs(a.y-b.y)<=1e-7))ct++;
	}
	return ct>=3;
}
struct sth{double a;int b;friend bool operator <(sth a,sth b){return a.a<b.a;}}v[N];
double as1,as2=0,vl,pi=acos(-1);
double dis(point x){return sqrt(x.x*x.x+x.y*x.y);}
double solve(vec tp,double r)
{
	double d1=dis(tp.s),d2=dis(tp.t);
	if(d1<=r+1e-6&&d2<=r+1e-6)return dis(tp.s-tp.t);
	if(d1>d2)swap(tp.s,tp.t),swap(d1,d2);
	if(d1<=r+1e-6&&d2>r)
	{
		double lb=0,rb=1,as=0;
		for(int i=1;i<=60;i++)
		{
			double mid=(lb+rb)/2;
			point pt=tp.t*mid+tp.s*(1-mid);
			if(dis(pt)<=r)as=mid,lb=mid;
			else rb=mid;
		}
		point pt=tp.t*as+tp.s*(1-as);
		return dis(tp.s-pt);
	}
	else
	{
		double lb=0,rb=1;
		for(int i=1;i<=80;i++)
		{
			double mid1=(lb*4+rb*3)/7,mid2=(lb*3+rb*4)/7;
			if(dis(tp.s*mid1+tp.t*(1-mid1))<dis(tp.s*mid2+tp.t*(1-mid2)))rb=mid2;
			else lb=mid1;
		}
		if(dis(tp.s*lb+tp.t*(1-lb))>r)return 0;
		return solve((vec){tp.s,tp.s*lb+tp.t*(1-lb)},r)+solve((vec){tp.t,tp.s*lb+tp.t*(1-lb)},r);
	}
}
int main()
{
	scanf("%d%d",&n,&r);
	for(int i=1;i<=n;i++)
	{
		scanf("%d%lf",&d,&vl);
		if(abs(vl-r)<=1e-8)continue;
		point pt=(point){vl*cos(d/180.0*pi),vl*sin(d/180.0*pi)};
		double dis=sqrt(r*r-vl*vl);
		point pt2=(point){-pt.y,pt.x};
		double len=dis/sqrt(pt2.x*pt2.x+pt2.y*pt2.y);
		pt2=pt2*len;
		point pt3=pt+pt2,pt4=pt-pt2;
		ed[++ct1]=(vec){pt4,pt3};
		double f1=d/180.0*pi,f2=acos(vl/r);
		double f3=f1-f2,f4=f1+f2;
		if(f3<0)v[++ct]=(sth){0,1},v[++ct]=(sth){f4,-1},v[++ct]=(sth){f3+2*pi,1},v[++ct]=(sth){pi*2,-1};
		else if(f4>2*pi)v[++ct]=(sth){0,1},v[++ct]=(sth){f4-2*pi,-1},v[++ct]=(sth){f3,1},v[++ct]=(sth){pi*2,-1};
		else v[++ct]=(sth){f3,1},v[++ct]=(sth){f4,-1};
	}
	int vl=0;
	sort(v+1,v+ct+1);v[ct+1].a=2*pi;
	for(int i=1;i<=ct+1;i++)
	{
		if(!vl)as2+=v[i].a-v[i-1].a;
		vl+=v[i].b;
	}
	as2*=r;
	si();
	for(int i=hd;i<=tl;i++)
	{
		vec fuc=(vec){q2[i],q2[i==tl?hd:i+1]};
		as1+=solve(fuc,r);
	}
	printf("%.10lf %.10lf\n",as1,as2);
}
```

#### 十连测 day 4

咕咕咕

#### 十连测 day 6

##### T1 网格

###### Problem

一个无限大的网格图， $n$ 次操作，每次删去一个格子，求剩下部分的连通块数

强制在线

$n\leq 10^5$

###### Sol

考虑欧拉定理 $V-E+F=1+C$

将删去的点间八连通连边，连通块数 $C$ ，点数 $V$ 和边数 $E$ 容易计算, $F$ 就是答案

但是如果出现了一个小三角形，它不应该被算入答案，因此要减1

如果出现了一个小正方形，这时如果将两个对角线都连起来，它就不是平面图了，因此这时应该不连新加入的点的那条对角线，

![](C:/users/zz/documents/pic/54.png)

用map存坐标，复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<map>
using namespace std;
map<long long,int> st;
long long id(int x,int y){return x*3000000000ll+y;}
int n,t,las=0,as=1,a,b,fa[100500],f[3][3];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int main()
{
	scanf("%d%d",&n,&t);
	for(int i=1;i<=n;i++)
	{
		scanf("%d%d",&a,&b);a^=t*las,b^=t*las;
		st[id(a,b)]=i;fa[i]=i;
		for(int j=-1;j<=1;j++)
		for(int k=-1;k<=1;k++)
		{
			int a1=st[id(a+j,b+k)];
			f[j+1][k+1]=a1;
			if(!a1)continue;
			if(finds(a1)!=finds(i))fa[finds(i)]=finds(a1),as--;
		}
		if(f[0][1])as++;
		if(f[2][1])as++;
		if(f[1][0])as++;
		if(f[1][2])as++;
		if(f[0][0]&&!f[0][1]&&!f[1][0])as++;
		if(f[0][2]&&!f[0][1]&&!f[1][2])as++;
		if(f[2][0]&&!f[2][1]&&!f[1][0])as++;
		if(f[2][2]&&!f[2][1]&&!f[1][2])as++;
		if(f[0][1]&&f[1][0])as--;
		if(f[0][1]&&f[1][2])as--;
		if(f[2][1]&&f[1][0])as--;
		if(f[2][1]&&f[1][2])as--;
		printf("%d\n",las=as);
	}
}
```

##### T2 拍卖

###### Problem

![](C:/users/zz/documents/pic/55.png)

$n\leq 10^6$

###### Sol

考虑暴力dp：设 $dp_{i,j}$ 表示当前拍到第 $i$ 件，还可以拍下 $j$ 件，先手减后手的值

那么有 $dp_{i,j}=max(a_i-dp_{i+1,j-1},min(dp_{i+1,j-1}-a_i,dp_{i+1,j}))$

三种情况分别表示先手拍已及先手不拍，后手进行选择

可以看出 $0\leq dp_{i,j}\leq 1$

考虑先手如果每次到有价值的就拍下，那么如果当前是一件无价值的，后手去拍不会有影响，这时先手拍两件有价值的中间后手最多拍一件有价值的，所以有 $dp_{i,j}\geq 0$

类似的，如果后手这样做，有相同的结论，可以得到 $dp_{i,j}\leq 1$

考虑上面的dp，当 $a_i=1$时，有 $dp_{i,j}=max(1-dp_{i+1,j-1},min(dp_{i+1,j-1}-1,dp_{i+1,j}))$

显然min里面第一个更小，于是

 $dp_{i,j}=max(1-dp_{i+1,j-1},dp_{i+1,j-1}-1)$

 $dp_{i,j}=1-dp_{i+1,j-1}$

如果 $a_i=0$ ,有 $dp_{i,j}=max(-dp_{i+1,j-1},min(dp_{i+1,j-1},dp_{i+1,j}))$

$dp_{i,j}=min(dp_{i+1,j-1},dp_{i+1,j})$

考虑 $dp_i$ 的每一个01段，可以发现，进行一轮 $dp_{i,j}=min(dp_{i+1,j-1},dp_{i+1,j})$ 的操作后，每一个0段长度+1,1段长度-1,同时可能在开头加入一个0段

使用平衡树维护所有的0段长度，1段长度，开头是0/1,倒着处理， $a_i=1$ 时直接交换， $a_i=0$ 时，找出所有的长度为1的1段，将其删除并且合并对应的0段。为了找到对应的段需要维护size

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<bitset>
using namespace std;
#define N 2000500
int v[N],mn[N],sz[N],fa[N],ch[N][2],lz[N],n,k,a,is0,l1,l2,ct,as[N],lb=1;
void pushdown(int x){v[ch[x][0]]+=lz[x],v[ch[x][1]]+=lz[x],mn[ch[x][0]]+=lz[x],mn[ch[x][1]]+=lz[x],lz[ch[x][0]]+=lz[x],lz[ch[x][1]]+=lz[x],lz[x]=0;}
void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1,mn[x]=min(v[x],min(mn[ch[x][0]],mn[ch[x][1]]));}
void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;pushdown(f);pushdown(x);ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
struct Splay{
	int rt;
	void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(g!=y)rotate(((ch[f][1]==x)^(ch[g][1]==f))?x:f);rotate(x);}if(!y)rt=x;}
	int find1(int x){if(v[x]==1)return x;pushdown(x);if(mn[ch[x][0]]==1)return find1(ch[x][0]);else return find1(ch[x][1]);}
	int kth(int x,int k){if(k<=sz[ch[x][0]])return kth(ch[x][0],k);if(k==sz[ch[x][0]]+1)return x;return kth(ch[x][1],k-sz[ch[x][0]]-1);}
}t[2];
char s[N];
bitset<10064> as1;
int main()
{
	scanf("%d%d%s",&n,&k,s+1);mn[0]=1e9;
	if(n<=5000)
	{
		for(int i=n;i>=1;i--)
		if(s[i]=='0')as1=as1&(as1<<1);
		else as1=~(as1<<1);
		while(k--)scanf("%d",&a),printf("%d\n",(int)as1[a-1]);
		return 0;
	}
	t[0].rt=ct=1;v[1]=n;
	for(int i=n;i>=1;i--)
	if(s[i]=='0')
	{
		if(!t[!is0].rt)continue;
		if(l1==1)
		{
			if(!t[is0].rt)t[is0].rt=++ct,sz[ct]=1;
			else
			{
				int st=t[is0].rt;
				while(ch[st][0])st=ch[st][0];
				t[is0].splay(st);pushdown(st);
				ch[st][0]=++ct;sz[ct]=1;fa[ct]=st;pushup(st);
			}
		}
		if(l2==1)
		{
			if(!t[is0].rt)t[is0].rt=++ct,sz[ct]=1;
			else
			{
				int st=t[is0].rt;
				while(ch[st][1])st=ch[st][1];
				t[is0].splay(st);pushdown(st);
				ch[st][1]=++ct;sz[ct]=1;fa[ct]=st;pushup(st);
			}
		}
		l1=l2=0;
		lz[t[is0].rt]++;v[t[is0].rt]++;mn[t[is0].rt]++;
		while(mn[t[!is0].rt]==1)
		{
			int v3=t[!is0].find1(t[!is0].rt);
			t[!is0].splay(v3);pushdown(v3);
			int s1=sz[ch[v3][0]]+1;
			int v1=t[is0].kth(t[is0].rt,s1),v2=t[is0].kth(t[is0].rt,s1+1);
			t[is0].splay(v1);t[is0].splay(v2,v1);
			pushdown(v1);pushdown(v2);
			v[v1]+=v[v2];fa[ch[v2][1]]=v1;ch[v1][1]=ch[v2][1];pushup(v1);
			if(!ch[v3][0])t[!is0].rt=ch[v3][1],fa[ch[v3][1]]=0;
			else if(!ch[v3][1])t[!is0].rt=ch[v3][0],fa[ch[v3][0]]=0;
			else
			{
				int lb=ch[v3][0],rb=ch[v3][1];
				while(ch[lb][1])lb=ch[lb][1];
				while(ch[rb][0])rb=ch[rb][0];
				t[!is0].splay(lb),t[!is0].splay(rb,lb);
				pushdown(lb);pushdown(rb);ch[rb][0]=fa[v3]=0;pushup(rb);pushup(lb);
			}
		}
		lz[t[!is0].rt]--;mn[t[!is0].rt]--;v[t[!is0].rt]--;
	}
	else
	{
		if(l1==1)
		{
			if(!t[is0].rt)t[is0].rt=++ct,v[ct]=sz[ct]=mn[ct]=1;
			else
			{
				int st=t[is0].rt;
				while(ch[st][0])st=ch[st][0];
				t[is0].splay(st);pushdown(st);
				ch[st][0]=++ct;v[ct]=sz[ct]=mn[ct]=1;fa[ct]=st;pushup(st);
			}
		}
		else
		{
			int st=t[is0].rt;
			while(ch[st][0])st=ch[st][0];
			t[is0].splay(st);v[st]++;pushdown(st);pushup(st);
		}
		is0^=1,l1=1,l2^=1;
	}
	while(lb<=n)
	{
		int st=t[is0^l1].rt;
		while(ch[st][0])st=ch[st][0];
		t[is0^l1].splay(st);pushdown(st);
		for(int i=lb;i<=lb+v[st]-1&&i<=n;i++)as[i]=l1;lb+=v[st];
		t[is0^l1].rt=ch[st][1],fa[ch[st][1]]=0;
		l1^=1;
	}
	while(k--)scanf("%d",&a),printf("%d\n",as[a]);
}
```

##### T3 迷宫

###### Problem

交互

![](C:/users/zz/documents/pic/56.png)

$n\leq 200,L\leq 14m,k\leq 3$

###### Sol

首先考虑dfs一遍，建出dfs树

对于每一个经过的点，把它的标记设为2，这样很容易完成dfs

在dfs的时候，额外记录树上相邻两个点的边在两个点的出边中分别的编号以及每个点出发哪些边是返祖边

dfs的步数是 $4m$

考虑如果还原剩下的边

一种暴力是每次枚举一个点染成3，然后dfs枚举每一条返祖边判断，这样的步数是 $2nm+4m$

考虑三进制染色，进行5次，每次给每个点染它这一位上的值，这样就可以还原出所有边

步数是 $5*2m+4m=14m$

建好图以后暴力算答案

###### Code

```cpp
#include "maze.h"
#include<vector>
#include<queue>
using namespace std;
#define N 205
#define M 45500
struct edge{int t,next;}ed[M];
int head[N],cnt,dis[N],n,is[N][N],st[N][N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
vector<int> as;
struct edge2{int t,next,v;}ed2[N*3];
int head2[N],cnt2=1,ct=1,sz[N];
void adde2(int f,int t,int v1,int v2){ed2[++cnt2]=(edge2){t,head2[f],v1};head2[f]=cnt2;ed2[++cnt2]=(edge2){f,head2[t],v2};head2[t]=cnt2;}
void dfs1(int u,int fa,int f)
{
	n++;
	sz[u]=get_edge_number();
	set_label(3);
	for(int i=1;i<=sz[u];i++)if(i!=f)
	{
		move(i);
		int tp=get_label();
		if(tp==2)move(get_coming_edge());
		else if(tp==3)is[u][i]=1,move(get_coming_edge());
		else
		{
			int tp2=get_coming_edge();
			adde2(u,++ct,i,tp2);adde(u,ct);
			dfs1(ct,u,tp2);
		}
	}
	set_label(2);
	if(fa)move(f);
}
void dfs3(int u,int fa,int f,int d)
{
	for(int i=1;i<=sz[u];i++)if(is[u][i]){move(i);int tp=get_label()-1;move(get_coming_edge());for(int j=1;j<=d;j++)tp*=3;st[u][i]+=tp;}
	int st1=u;for(int i=1;i<=d;i++)st1/=3;
	set_label(st1%3+1);
	for(int i=head2[u];i;i=ed2[i].next)if(ed2[i].t!=fa)move(ed2[i].v),dfs3(ed2[i].t,u,ed2[i^1].v,d);
	if(fa)move(f);
}
void doit(int s)
{
	for(int i=1;i<=n;i++)dis[i]=-1;dis[s]=0;
	queue<int> fu;
	fu.push(s);
	while(!fu.empty())
	{
		int r=fu.front();fu.pop();
		for(int i=head[r];i;i=ed[i].next)
		if(dis[ed[i].t]==-1)dis[ed[i].t]=dis[r]+1,fu.push(ed[i].t);
	}
	for(int j=s;j<=n;j++)as[dis[j]]++;
}
std::vector<int> solve(int k, int L)
{
    dfs1(1,0,0);
    for(int i=0;i<=4;i++)dfs3(1,0,0,i);
    for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)if(st[i][j])adde(i,st[i][j]);
    for(int i=1;i<=n;i++)as.push_back(0);
    for(int i=1;i<=n;i++)doit(i);
    return as;
}
```

#### 十连测 day 7.5

##### T1 Trie树

###### Problem

给 $n$ 个01?串 $s_1,...,s_n$ ,求对于每一种将所有?替换成01的方案，所有串构成的trie树节点数之和模 $998244353\ $

$n\leq 20,|s_i|\leq 50$

###### Sol

trie树节点数相当于不同的前缀数

对于没有?的情况考虑容斥，答案为 $\sum |s_i|+\sum_{S}(-1)^{|S|}lcp_{i\in S}S_i$

对于原问题，把每一个 $S$ 提出来，考虑一个 $S$ 的一位

如果这一位上 $S$ 包含的字符串都是?,这一位相同的方案数是2

否则，如果01都有，方案数是0，否则是1

直接暴力对于每一位做高维前缀和复杂度为 $O(|s|*n2^n)$ ,因为只有 $n$ 个值，考虑对于每个集合每次减去lowbit算，这样复杂度就是 $O(|s|*2^n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 22
#define M 52
#define K 1050000
#define mod 998244353
int su[K][3],dp[K],dp2[K],is[K],n,ct,ipw[N],fu[K],as;
char s[N][M];
int main()
{
	scanf("%d",&n);
	ipw[0]=1;for(int i=1;i<=n;i++)ipw[i]=1ll*ipw[i-1]*499122177%mod;
	for(int i=1;i<1<<n;i++)fu[i]=fu[i>>1]+1;
	for(int i=1;i<1<<n;i++)dp2[i]=1,dp[i]=1;
	for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
	for(int i=1;i<=50;i++)
	{
		memset(su,0,sizeof(su));
		memset(is,0,sizeof(is));
		for(int j=1;j<1<<n;j++)
		{
			int lbit=j&-j,st=j-lbit,vl=fu[lbit];
			for(int k=0;k<3;k++)su[j][k]=su[st][k];
			is[j]|=is[st];
			if(s[vl][i]==0)is[j]=1;
			else if(s[vl][i]=='0')su[j][0]++;
			else if(s[vl][i]=='1')su[j][1]++;
			else su[j][2]++;
			if(is[j])dp2[j]=0;
			if(su[j][0]&&su[j][1])dp2[j]=0;
			if(!su[j][0]&&!su[j][1])dp2[j]=1ll*ipw[su[j][2]-1]*dp2[j]%mod,dp[j]=(dp[j]+dp2[j])%mod;
			else dp2[j]=1ll*ipw[su[j][2]]*dp2[j]%mod,dp[j]=(dp[j]+dp2[j])%mod;
		}
	}
	fu[0]=-1;
	for(int i=1;i<1<<n;i++)
	{
		fu[i]=fu[i>>1]*(i&1?-1:1);
		as=(as+1ll*dp[i]*fu[i]+mod*2ll)%mod;
	}
	for(int i=1;i<=50;i++)for(int j=1;j<=n;j++)if(s[j][i]=='?')as=1ll*as*2%mod;
	printf("%d\n",as);
}
```

##### T2 独立集

###### Problem

给定 $n$ 求对于所有 $1\leq i,j\leq n$ ,有多少棵有根，儿子有序的无标号树满足有 $i$ 个点且独立集为 $j$

模 $998244353\ $

$n\leq 500$

###### Sol

考虑算树独立集的dp

设 $f_i$ 表示 $i$ 的子树，不选 $i$ 的最大独立集， $g_i$ 表示 $i$ 的子树的独立集，转移显然

那么有 $0\leq g_i-f_i\leq 1$

考虑类似dp套dp，设 $dp_{i,j,0/1}$ 表示 $i$ 个点的树，根节点的 $f$ 为 $j$ ,$g-f$ 为0/1,当前的方案数

转移直接枚举下一个儿子,考虑独立集dp过程有 

$dp_{i,j,0}=\sum_{k=1}^{i-1}\sum_{l=0}^{j}dp_{k,l,0}*dp_{i-k,j-l,0}+\sum_{k=1}^{i-1}\sum_{l=0}^{j-1}dp_{k,l,1}*dp_{i-k,j-l-1,0}+\sum_{k=1}^{i-1}\sum_{l=0}^{j-1}dp_{k,l,1}*dp_{i-k,j-l-1,1}$

$dp_{i,j,1}=\sum_{k=1}^{i-1}\sum_{l=0}^{j}dp_{k,l,0}*dp_{i-k,j-l,1}$

初值 $dp_{1,0,1}=1$

暴力复杂度 $O(n^4)$ ,对第二维使用FFT可以做到 $O(n^3\log n)$

考虑转移时只会涉及到多项式乘一个 $x$ ,并且无论如何多项式次数小于 $n$ ,因此可以存dp第二维的点值

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 515
#define mod 998244353
int dp[N][N][2],dp2[N][N][2],n,pw[N],as[N][N][2],f[N],rev[N];
int pw1(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	int w=pw1(3,(mod-1)/512);
	pw[0]=1;for(int i=1;i<512;i++)pw[i]=1ll*pw[i-1]*w%mod;
	for(int i=0;i<512;i++)dp[1][i][1]=pw[i],dp2[0][i][0]=dp2[0][i][1]=1;
	for(int i=2;i<=n;i++)
	{
		for(int j=1;j<i;j++)
		for(int k=0;k<512;k++)
		for(int s=0;s<=1;s++)
		dp2[i-1][k][s]=(dp2[i-1][k][s]+1ll*dp2[i-1-j][k][s]*dp[j][k][s])%mod;
		for(int k=0;k<512;k++){dp[i][k][1]=1ll*pw[k]*dp2[i-1][k][0]%mod;dp[i][k][0]=(dp2[i-1][k][1]-dp2[i-1][k][0]+mod)%mod;dp[i][k][1]=(dp[i][k][1]+dp[i][k][0])%mod;}
	}
	for(int j=0;j<512;j++)rev[j]=rev[j/2]/2+(j&1)*256;
	for(int i=1;i<=n;i++)
	for(int k=0;k<2;k++)
	{
		for(int j=0;j<512;j++)f[rev[j]]=dp[i][j][k];
		for(int l=2;l<=512;l<<=1)
		{
			int st=pw1(3,(mod-1)/l);
			st=pw1(st,mod-2);
			for(int j=0;j<512;j+=l)
			for(int t=j,v1=1;t<j+(l>>1);t++,v1=1ll*v1*st%mod)
			{
				int v11=f[t],v21=1ll*f[t+(l>>1)]*v1%mod;
				f[t]=(v11+v21)%mod;
				f[t+(l>>1)]=(v11-v21+mod)%mod;
			}
		}
		for(int j=0;j<512;j++)as[i][j][k]=1ll*f[j]*pw1(512,mod-2)%mod;
	}
	for(int i=1;i<=n;i++,printf("\n"))
	for(int j=0;j<=n;j++)printf("%d ",as[i][j][1]);
}
```

##### T3 数据结构

###### Problem

对于数列 $A$，定义其权值为它的本质不同的子序列个数模 $998244353$  

现在给出一个长度为 $n$ 的序列 $A$ 和一个整数 $K$，现在对这个序列进行了 $m$ 次操作：

1. 给出 $l,r,x,\forall x\in[l,r]$，把 $A_i$ 变成 $(A_i+x)\mod K$
2. 给出 $l,r,x,\forall x\in[l,r]$，把 $A_i$ 变成 $(A_i*x)\mod K$
3. 给出 $l,r$，询问数组 $A$ 的区间 $[l,r]$ 形成的序列的权值

多组数据

$n\leq 30000,K\leq 5,T\leq 2$

###### Sol

首先考虑暴力dp，设 $dp_i$ 表示以 $i$ 结尾的子序列个数，$dp_{K+1}$ 表示 $\sum_{i=1}^K dp_i$

那么一个字符的转移可以写成一个矩阵

首先考虑操作1，这相当于对一个区间的矩阵的行列进行一个置换，这个可以通过在线段树上打区间置换标记解决

然后考虑操作2，在 $K=1,2,3,5$ 的情况下，相当于进行区间置换或者区间推平

预处理出每个长度的区间推平后的矩阵即可

在 $K=4$ 时，还有一种操作是区间乘2模4

这个时候相当于有三种标记，~~然后直接写我场上讨论到自闭~~

考虑维护两棵线段树，第二棵记录原序列区间乘2之后的矩阵乘积，更新时同时更新

复杂度$O(nK^3\log n)$

###### Code

~~特别难写~~

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 30050
#define mod 998244353
int T,n,m,k,v[N],a,b,c,d,p[6],p2[N]; 
struct task1
{
	struct mat{int s[6][6];}pw[N];
	struct node{int l,r,p[6],lz;mat dp;}e[N*4];
	friend mat operator *(mat a,mat b){mat c;for(int i=0;i<=k;i++)for(int j=0;j<=k;j++)c.s[i][j]=0;for(int l=0;l<=k;l++)for(int i=0;i<=k;i++)if(a.s[i][l])for(int j=0;j<=k;j++)c.s[i][j]=(c.s[i][j]+1ll*a.s[i][l]*b.s[l][j])%mod;return c;}
	void pushup(int x){e[x].dp=e[x<<1].dp*e[x<<1|1].dp;}
	void pushdown(int x){
	if(e[x].lz){for(int i=1;i<=k;i++)e[x<<1].p[i]=e[x<<1|1].p[i]=i;e[x<<1].lz=e[x<<1|1].lz=1;e[x].lz=0;e[x<<1].dp=pw[e[x<<1].r-e[x<<1].l+1],e[x<<1|1].dp=pw[e[x<<1|1].r-e[x<<1|1].l+1];}
	int fg=0;for(int i=1;i<=k;i++)if(e[x].p[i]!=i)fg=1;
	node fuc;
	if(fg)
	{
		fuc=e[x<<1];for(int i=1;i<=k;i++)fuc.p[i]=e[x].p[e[x<<1].p[i]],fuc.dp.s[e[x].p[i]-1][k]=e[x<<1].dp.s[i-1][k],fuc.dp.s[k][e[x].p[i]-1]=e[x<<1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[e[x].p[i]-1][e[x].p[j]-1]=e[x<<1].dp.s[i-1][j-1];e[x<<1]=fuc;
		fuc=e[x<<1|1];for(int i=1;i<=k;i++)fuc.p[i]=e[x].p[e[x<<1|1].p[i]],fuc.dp.s[e[x].p[i]-1][k]=e[x<<1|1].dp.s[i-1][k],fuc.dp.s[k][e[x].p[i]-1]=e[x<<1|1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[e[x].p[i]-1][e[x].p[j]-1]=e[x<<1|1].dp.s[i-1][j-1];e[x<<1|1]=fuc;
	}
	for(int i=1;i<=k;i++)e[x].p[i]=i;}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].lz=0;
		for(int i=1;i<=k;i++)e[x].p[i]=i;
		if(l==r){memset(e[x].dp.s,0,sizeof(e[x].dp.s));for(int i=0;i<=k;i++)e[x].dp.s[i][i]=1;int st=v[l];e[x].dp.s[st][st]=0;e[x].dp.s[k][k]=2;e[x].dp.s[st][k]=mod-1;e[x].dp.s[k][st]=1;return;}
		int mid=(e[x].l+e[x].r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify1(int x,int l,int r)
	{
		if(e[x].l!=e[x].r)pushdown(x);
		if(e[x].l==l&&e[x].r==r)
		{
			node fuc=e[x];
			for(int i=1;i<=k;i++)fuc.p[i]=p[e[x].p[i]],fuc.dp.s[p[i]-1][k]=e[x].dp.s[i-1][k],fuc.dp.s[k][p[i]-1]=e[x].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[p[i]-1][p[j]-1]=e[x].dp.s[i-1][j-1];e[x]=fuc;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify1(x<<1,l,r);
		else if(mid<l)modify1(x<<1|1,l,r);
		else modify1(x<<1,l,mid),modify1(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify2(int x,int l,int r)
	{
		if(e[x].l!=e[x].r)pushdown(x);
		if(e[x].l==l&&e[x].r==r)
		{
			e[x].lz=1;
			e[x].dp=pw[r-l+1];
			for(int i=1;i<=k;i++)e[x].p[i]=i;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify2(x<<1,l,r);
		else if(mid<l)modify2(x<<1|1,l,r);
		else modify2(x<<1,l,mid),modify2(x<<1|1,mid+1,r);
		pushup(x);
	}
	mat query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].dp;
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);
		else if(mid<l)return query(x<<1|1,l,r);
		else return query(x<<1,l,mid)*query(x<<1|1,mid+1,r);
	}
	void solve()
	{
		for(int i=0;i<=k;i++)pw[0].s[i][i]=1;
		memset(pw[1].s,0,sizeof(pw[1].s));
		for(int i=0;i<=k;i++)pw[1].s[i][i]=1;
		pw[1].s[0][k]=mod-1;pw[1].s[k][0]=1;pw[1].s[0][0]=0;pw[1].s[k][k]=2;
		for(int i=2;i<=n;i++)pw[i]=pw[i-1]*pw[1];
		for(int i=1;i<=n;i++)scanf("%d",&v[i]);
		build(1,1,n);
		while(m--)
		{
			scanf("%d",&a);
			if(a==1)
			{
				scanf("%d%d%d",&b,&c,&d);
				for(int i=1;i<=k;i++)p[i]=(i-1+d)%k+1;
				modify1(1,b,c);
			}
			if(a==2)
			{
				scanf("%d%d%d",&b,&c,&d);
				if(d)
				{
					for(int i=1;i<=k;i++)p[i]=((i-1)*d)%k+1;
					modify1(1,b,c);
				}
				else modify2(1,b,c);
			}
			if(a==3)
			{
				scanf("%d%d",&b,&c);
				int as=query(1,b,c).s[k][k];
				as=(as-1+mod)%mod;
				printf("%d\n",as);
			}
		}
	}
}s1;
struct task2
{
	struct mat{int s[5][5];}pw[N];
	struct node{int l,r,p[5],lz,lz2;mat dp;}e[N*4],f[N*4];
	friend mat operator *(mat a,mat b){mat c;for(int i=0;i<=k;i++)for(int j=0;j<=k;j++)c.s[i][j]=0;for(int l=0;l<=k;l++)for(int i=0;i<=k;i++)if(a.s[i][l])for(int j=0;j<=k;j++)c.s[i][j]=(c.s[i][j]+1ll*a.s[i][l]*b.s[l][j])%mod;return c;}
	void pushup(int x){e[x].dp=e[x<<1].dp*e[x<<1|1].dp;f[x].dp=f[x<<1].dp*f[x<<1|1].dp;}
	void pushdown(int x){
	if(e[x<<1].lz2){if(e[x<<1].r>e[x<<1].l)e[x<<2].lz2=e[x<<2|1].lz2=1;e[x<<1]=f[x<<1];for(int i=1;i<=k;i++)f[x<<1].p[i]=i;f[x<<1].lz=1;f[x<<1].dp=pw[f[x<<1].r-f[x<<1].l+1];e[x<<1].lz2=0;}
	if(e[x<<1|1].lz2){if(e[x<<1|1].r>e[x<<1|1].l)e[x<<2|2].lz2=e[x<<2|3].lz2=1;e[x<<1|1]=f[x<<1|1];for(int i=1;i<=k;i++)f[x<<1|1].p[i]=i;f[x<<1|1].lz=1;f[x<<1|1].dp=pw[f[x<<1|1].r-f[x<<1|1].l+1];e[x<<1|1].lz2=0;}
	if(e[x].lz){for(int i=1;i<=k;i++)e[x<<1].p[i]=e[x<<1|1].p[i]=i;e[x<<1].lz=e[x<<1|1].lz=1;e[x].lz=0;e[x<<1].dp=pw[e[x<<1].r-e[x<<1].l+1],e[x<<1|1].dp=pw[e[x<<1|1].r-e[x<<1|1].l+1];}
	int fg=0;for(int i=1;i<=k;i++)if(e[x].p[i]!=i)fg=1;
	node fuc;
	if(fg)
	{
		fuc=e[x<<1];for(int i=1;i<=k;i++)fuc.p[i]=e[x].p[e[x<<1].p[i]],fuc.dp.s[e[x].p[i]-1][k]=e[x<<1].dp.s[i-1][k],fuc.dp.s[k][e[x].p[i]-1]=e[x<<1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[e[x].p[i]-1][e[x].p[j]-1]=e[x<<1].dp.s[i-1][j-1];e[x<<1]=fuc;
		fuc=e[x<<1|1];for(int i=1;i<=k;i++)fuc.p[i]=e[x].p[e[x<<1|1].p[i]],fuc.dp.s[e[x].p[i]-1][k]=e[x<<1|1].dp.s[i-1][k],fuc.dp.s[k][e[x].p[i]-1]=e[x<<1|1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[e[x].p[i]-1][e[x].p[j]-1]=e[x<<1|1].dp.s[i-1][j-1];e[x<<1|1]=fuc;
	}
	for(int i=1;i<=k;i++)e[x].p[i]=i;
	if(f[x].lz){for(int i=1;i<=k;i++)f[x<<1].p[i]=f[x<<1|1].p[i]=i;f[x<<1].lz=f[x<<1|1].lz=1;f[x].lz=0;f[x<<1].dp=pw[f[x<<1].r-f[x<<1].l+1],f[x<<1|1].dp=pw[f[x<<1|1].r-f[x<<1|1].l+1];}
	fg=0;for(int i=1;i<=k;i++)if(f[x].p[i]!=i)fg=1;
	if(fg)
	{
		fuc=f[x<<1];for(int i=1;i<=k;i++)fuc.p[i]=f[x].p[f[x<<1].p[i]],fuc.dp.s[f[x].p[i]-1][k]=f[x<<1].dp.s[i-1][k],fuc.dp.s[k][f[x].p[i]-1]=f[x<<1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[f[x].p[i]-1][f[x].p[j]-1]=f[x<<1].dp.s[i-1][j-1];f[x<<1]=fuc;
		fuc=f[x<<1|1];for(int i=1;i<=k;i++)fuc.p[i]=f[x].p[f[x<<1|1].p[i]],fuc.dp.s[f[x].p[i]-1][k]=f[x<<1|1].dp.s[i-1][k],fuc.dp.s[k][f[x].p[i]-1]=f[x<<1|1].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[f[x].p[i]-1][f[x].p[j]-1]=f[x<<1|1].dp.s[i-1][j-1];f[x<<1|1]=fuc;
	}
	for(int i=1;i<=k;i++)f[x].p[i]=i;}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;e[x].lz=0;e[x].lz2=0;
		f[x].l=l;f[x].r=r;f[x].lz=0;
		for(int i=1;i<=k;i++)e[x].p[i]=f[x].p[i]=i;
		if(l==r){memset(e[x].dp.s,0,sizeof(e[x].dp.s));for(int i=0;i<=k;i++)e[x].dp.s[i][i]=1;int st=v[l];e[x].dp.s[st][st]=0;e[x].dp.s[k][k]=2;e[x].dp.s[st][k]=mod-1;e[x].dp.s[k][st]=1;
		memset(f[x].dp.s,0,sizeof(f[x].dp.s));for(int i=0;i<=k;i++)f[x].dp.s[i][i]=1;st=2*v[l]%k;f[x].dp.s[st][st]=0;f[x].dp.s[k][k]=2;f[x].dp.s[st][k]=mod-1;f[x].dp.s[k][st]=1;return;}
		int mid=(e[x].l+e[x].r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify1(int x,int l,int r)
	{
		if(e[x].l!=e[x].r)pushdown(x);
		if(e[x].l==l&&e[x].r==r)
		{
			node fuc=e[x];
			for(int i=1;i<=k;i++)fuc.p[i]=p[e[x].p[i]],fuc.dp.s[p[i]-1][k]=e[x].dp.s[i-1][k],fuc.dp.s[k][p[i]-1]=e[x].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[p[i]-1][p[j]-1]=e[x].dp.s[i-1][j-1];e[x]=fuc;
			fuc=f[x];for(int i=1;i<=k;i++)fuc.p[i]=p2[f[x].p[i]],fuc.dp.s[p2[i]-1][k]=f[x].dp.s[i-1][k],fuc.dp.s[k][p2[i]-1]=f[x].dp.s[k][i-1];for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)fuc.dp.s[p2[i]-1][p2[j]-1]=f[x].dp.s[i-1][j-1];f[x]=fuc;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify1(x<<1,l,r);
		else if(mid<l)modify1(x<<1|1,l,r);
		else modify1(x<<1,l,mid),modify1(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify2(int x,int l,int r)
	{
		if(e[x].l!=e[x].r)pushdown(x);
		if(e[x].l==l&&e[x].r==r)
		{
			e[x].lz=1;
			e[x].dp=pw[r-l+1];
			for(int i=1;i<=k;i++)e[x].p[i]=i;
			f[x].lz=1;
			f[x].dp=pw[r-l+1];
			for(int i=1;i<=k;i++)f[x].p[i]=i;
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify2(x<<1,l,r);
		else if(mid<l)modify2(x<<1|1,l,r);
		else modify2(x<<1,l,mid),modify2(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify3(int x,int l,int r)
	{
		if(e[x].l!=e[x].r)pushdown(x);
		if(e[x].l==l&&e[x].r==r)
		{
			e[x]=f[x];if(x<=60000)e[x<<1].lz2=1,e[x<<1|1].lz2=1;
			for(int i=1;i<=k;i++)f[x].p[i]=i;f[x].lz=1;f[x].dp=pw[f[x].r-f[x].l+1];
			return;
		}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify3(x<<1,l,r);
		else if(mid<l)modify3(x<<1|1,l,r);
		else modify3(x<<1,l,mid),modify3(x<<1|1,mid+1,r);
		pushup(x);
	}
	mat query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].dp;
		pushdown(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);
		else if(mid<l)return query(x<<1|1,l,r);
		else return query(x<<1,l,mid)*query(x<<1|1,mid+1,r);
	}
	void solve()
	{
		for(int i=0;i<=k;i++)pw[0].s[i][i]=1;
		memset(pw[1].s,0,sizeof(pw[1].s));
		for(int i=0;i<=k;i++)pw[1].s[i][i]=1;
		pw[1].s[0][k]=mod-1;pw[1].s[k][0]=1;pw[1].s[0][0]=0;pw[1].s[k][k]=2;
		for(int i=2;i<=n;i++)pw[i]=pw[i-1]*pw[1];
		for(int i=1;i<=n;i++)scanf("%d",&v[i]);
		build(1,1,n);
		while(m--)
		{
			scanf("%d",&a);
			if(a==1)
			{
				scanf("%d%d%d",&b,&c,&d);
				for(int i=1;i<=k;i++)p[i]=(i-1+d)%k+1,p2[i]=(i-1+2ll*d)%k+1;
				modify1(1,b,c);
			}
			if(a==2)
			{
				scanf("%d%d%d",&b,&c,&d);
				if(d!=2&&d)
				{
					for(int i=1;i<=k;i++)p[i]=((i-1)*d)%k+1;
					for(int i=1;i<=k;i++)p2[i]=((i-1)*d)%k+1;
					modify1(1,b,c);
				}
				else if(!d)modify2(1,b,c);
				else modify3(1,b,c);
			}
			if(a==3)
			{
				scanf("%d%d",&b,&c);
				int as=query(1,b,c).s[k][k];
				as=(as-1+mod)%mod;
				printf("%d\n",as);
			}
		}
	}
}s2;
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d%d",&n,&k,&m);
		if(k==4)s2.solve();else s1.solve();
	}
}
```

#### 十连测 day 8

##### T1 Alienation

###### Problem

你有一个平面图，每个点坐标为整数，它被划分成了若干三角形，点1在 $(0,0)$ ,点2在 $(x,0)$ ,所有点纵坐标非负

现在给出平面图每条边的边长和连接的点,求出每个点的坐标

多组数据

$n\leq 10000,T\leq 10$

###### Sol

首先搞出所有三元环，这部分是 $O(n^{1.5})$ 的

对于一个三元环，容易根据两点的坐标解三角形求第三个点的坐标

问题在于确定这个点在另外两个点的哪个方向

一个想法是bfs，每次钦定它在左侧，但是有这种情况

![](C:\Users\zz\Documents\pic\57.png)

可以发现，对于一个点，它被正确更新的方式中，每一个三角形的面积max都小于任意一种错误更新方式中面积最大的三角形的面积

因此可以每次选一个面积最小的三角形解

复杂度 $O(Tn^{1.5}\log n)$ ,卡卡能过

###### Code

```cpp
#include<cstdio>
#include<cmath>
#include<cstring>
#include<queue>
#include<vector>
#include<map>
using namespace std;
#define N 40030
int T,n,m,in[N],cr[N*100][3],fu1[N*100][3],s[N][2],head[N],cnt,ct,is[N],vis[N],is2[N],is3[N];
vector<int> fu[N];
struct point{double x,y;}as[N];
double ds[N*100][3],di[N],sz[N*100];
point operator +(point a,point b){return (point){a.x+b.x,a.y+b.y};}
point operator -(point a,point b){return (point){a.x-b.x,a.y-b.y};}
point operator *(point a,double b){return (point){a.x*b,a.y*b};}
struct edge{int t,next,id;}ed[N*2];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;}
double cross(point x,point y){return x.y*y.x-x.x*y.y;}
void addc(int d,int e,int f)
{
	++ct;
	int su=s[d][0]^s[d][1];
	fu1[ct][0]=d;fu1[ct][1]=e;fu1[ct][2]=f;
	if(s[e][0]==s[d][0]||s[e][0]==s[d][1])su^=s[e][1];else su^=s[e][0];
	cr[ct][0]=s[d][0]^s[d][1]^s[f][0]^s[f][1]^su;
	cr[ct][1]=s[d][0]^s[d][1]^s[e][0]^s[e][1]^su;
	cr[ct][2]=s[e][0]^s[e][1]^s[f][0]^s[f][1]^su;
	ds[ct][0]=di[d];ds[ct][1]=di[e];ds[ct][2]=di[f];
	fu[d].push_back(ct);fu[e].push_back(ct);fu[f].push_back(ct);
	double a1=(di[d]*di[d]+di[e]*di[e]-di[f]*di[f])/2/di[e]/di[d];
	a1=sqrt(1-a1*a1);
	sz[ct]=di[d]*di[e]*a1;
}
map<long long,int> fuc2,fuc3;
void pre()
{
	fuc3.clear();
	memset(in,0,sizeof(in));
	memset(head,0,sizeof(head));
	memset(is,0,sizeof(is));
	memset(is2,0,sizeof(is2));
	memset(is3,0,sizeof(is3));
	cnt=ct=0;
	for(int i=1;i<=m;i++)in[s[i][0]]++,in[s[i][1]]++,fu[i].clear();
	for(int i=1;i<=m;i++)
	if(in[s[i][0]]<in[s[i][1]]||(in[s[i][0]]==in[s[i][1]]&&s[i][0]<s[i][1]))adde(s[i][0],s[i][1],i);
	else adde(s[i][1],s[i][0],i);
	for(int i=1;i<=n;i++)
	{
		for(int j=head[i];j;j=ed[j].next)vis[ed[j].t]=ed[j].id;
		for(int j=head[i];j;j=ed[j].next)for(int k=head[ed[j].t];k;k=ed[k].next)if(vis[ed[k].t])addc(ed[j].id,ed[k].id,vis[ed[k].t]);
		for(int j=head[i];j;j=ed[j].next)vis[ed[j].t]=0;
	}
}
struct mod{int x,y,id;double sz;};
struct cmp{
	bool operator () (mod a,mod b)
	{
		return a.sz>b.sz;
	}
}; 
priority_queue<mod,vector<mod>,cmp> st;
point build(point a,point b,double d1,double d2,double d3)
{
	double ang1=acos((d1*d1+d2*d2-d3*d3)/2/d1/d2);
	point st=b-a;
	double fuc=d2/d1;st=st*fuc;
	point st2;st2.x=st.x*cos(ang1)-st.y*sin(ang1),st2.y=st.x*sin(ang1)+st.y*cos(ang1);
	return a+st2;
}
void solve()
{
	double ti=1e-9;
	int c1=0;
	fuc2.clear();
	int s0=fu[1].size();
	for(int i=0;i<s0;i++)
	st.push((mod){1,2,fu[1][i],sz[fu[1][i]]+ti*c1}),c1++;
	is2[1]=is2[2]=1;
	while(!st.empty())
	{
		mod f1=st.top();st.pop();
		if(is[f1.id])continue;is[f1.id]=1;
		int s1=f1.x,s2=f1.y,id=f1.id,s3=f1.x^f1.y^cr[id][0]^cr[id][1]^cr[id][2];
		double d1,d2,d3;
		if(!(cr[id][0]^cr[id][1]^s1^s2))d1=ds[id][0];
		if(!(cr[id][1]^cr[id][2]^s1^s2))d1=ds[id][1];
		if(!(cr[id][2]^cr[id][0]^s1^s2))d1=ds[id][2];
		if(!(cr[id][0]^cr[id][1]^s1^s3))d2=ds[id][0];
		if(!(cr[id][1]^cr[id][2]^s1^s3))d2=ds[id][1];
		if(!(cr[id][2]^cr[id][0]^s1^s3))d2=ds[id][2];
		if(!(cr[id][0]^cr[id][1]^s3^s2))d3=ds[id][0];
		if(!(cr[id][1]^cr[id][2]^s3^s2))d3=ds[id][1];
		if(!(cr[id][2]^cr[id][0]^s3^s2))d3=ds[id][2];
		point ls=build(as[s1],as[s2],d1,d2,d3);
		ls.x=(int)(ls.x+0.3*(ls.x>0?1:-1));
		ls.y=(int)(ls.y+0.3*(ls.y>0?1:-1));
		if(!is2[s3])as[s3]=ls,is2[s3]=1;
		int u1,u2;
		for(int i=0;i<3;i++)if((s[fu1[id][i]][0]^s[fu1[id][i]][1])==(s1^s3))u1=fu1[id][i];
		for(int i=0;i<3;i++)if((s[fu1[id][i]][0]^s[fu1[id][i]][1])==(s2^s3))u2=fu1[id][i];
		if(!is3[u1])
		{
			is3[u1]=1;
			int tp4=u1;
			int s0=fu[tp4].size();
			for(int i=0;i<s0;i++)if(!is[fu[tp4][i]])
			st.push((mod){s1,s3,fu[tp4][i],sz[fu[tp4][i]]+ti*c1}),c1++;
		}
		if(!is3[u2])
		{
			is3[u2]=1;
			int tp4=u2;
			int s0=fu[tp4].size();
			for(int i=0;i<s0;i++)if(!is[fu[tp4][i]])
			st.push((mod){s3,s2,fu[tp4][i],sz[fu[tp4][i]]+ti*c1}),c1++;
		}
	}
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&m);
		for(int i=1;i<=m;i++)scanf("%d%d%lf",&s[i][0],&s[i][1],&di[i]);
		as[1].x=as[1].y=as[2].y=0;
		as[2].x=di[1];
		pre();solve();
		for(int i=1;i<=n;i++)printf("%.0lf %.0lf\n",as[i].x,as[i].y);
		printf("\n");
	}
}
```

##### T2 Antarctica

###### Problem

有一个 $W\times H$ 的长方形，里面有一个每条边都平行于边界的多边形洞，点数为 $n$

有 $q$ 组询问。每次给一个起始点，一个弹珠从这个点开始以 $(1,1)/s$ 的速度运动，碰到边界后反弹，求落入洞的时间和位置或者不会落入洞

$n\leq 1000,m\leq 100,W,H\leq 5\times 10^8$

###### Sol

首先将矩形横纵各沿边界镜像一次，这样之后变成了两个坐标都是循环的

这时考虑每条边，可以看成求一个 $l\leq ax \bmod b\leq r$ 的最小非负 $x$

显然 $a≥b$ 时可以用 $a \bmod b$ 代替 $a$

如果 $l≤ax≤r$ 有解，那么它一定是最小解

当 $b=0$ 时，如果在上一步没有求出解，那么一定无解

考虑 $a<b$ 的情况，发现 $x$ 的最小非负整数解相当于求 $y$ 的最大非正整数解

将方程取反，得 $-r≤b(-y)+a(-x)≤-l$

$a∗⌈r/a⌉-r≤b(-y)+a(-x+⌈r/a⌉)≤a∗⌈r/a⌉-l$

这时 $y$ 的最大非正整数解等价于 $-y$ 的最小非负整数解

又因为 $l≤ax≤r$ 没有求出解，所以一定有 $0≤a∗⌈r/a⌉-r≤a∗⌈r/a⌉-l<a$

如果方程的解 $x<⌈r/a⌉$ ,那么 $ax<r$ ,因为 $l≤r<b$ ,那么 $l≤ax+by≤r$ 的解必然满足 $y=0$ ,而这种情况无解，因此可以认为 $-x+⌈r/a⌉≤0$

经过这些操作，得到了一个交换 $a,b$ 的方程

根据gcd的复杂度分析，这里是 $O(\log W)$ 的

复杂度 $O(nm\log W)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 1050
int pt[N][2],a,b,w,h,n,m;
long long as;
int solve(int l,int r,int n,int m)// l <= xn mod m <= r
{
	n%=m;
	if(!n)return l?-1:0;
	if(r/n-(l-1)/n)return (l-1)/n+1;
	// l <= xn+km <= r
	//-r <= -km+-xn <= -l
	int as1=solve((n-r%n)%n,(n-l%n)%n,m,n);
	if(as1==-1)return -1;
	int as2=(r+1ll*as1*m)/n;
	return as2;
}
long long solve2(int x,int l,int r,int a,int b,int w,int h)
{
	if(x<a)x+=2*w;
	int t2=(b+x-a)%(2*h);
	if(t2>=l&&t2<=r)return x-a;
	int l1=(l+2*h-t2)%(2*h),r1=(r+2*h-t2)%(2*h);
	int tp4=solve(l1,r1,2*w,2*h);
	if(tp4==-1)return -1;
	return 2ll*tp4*w+x-a;
}
long long solve3(int a,int b,int c,int d,int x,int y)
{
	if(a==c)return solve2(a,b,d,x,y,w,h);
	else return solve2(b,a,c,y,x,h,w);
}
void doit(int a,int b,int c,int d,int x,int y)
{
	if(a>c)a^=c^=a^=c;
	if(b>d)b^=d^=b^=d;
	long long tp=solve3(a,b,c,d,x,y);
	if(tp!=-1)as=min(as,tp);
}
int main()
{
	scanf("%d%d%d",&w,&h,&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&pt[i][0],&pt[i][1]);
	scanf("%d",&m);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&a,&b);
		as=1e18;
		for(int j=1;j<=n;j++)
		doit(pt[j][0],pt[j][1],pt[j%n+1][0],pt[j%n+1][1],a,b),
		doit(2*w-pt[j][0],pt[j][1],2*w-pt[j%n+1][0],pt[j%n+1][1],a,b),
		doit(pt[j][0],2*h-pt[j][1],pt[j%n+1][0],2*h-pt[j%n+1][1],a,b),
		doit(2*w-pt[j][0],2*h-pt[j][1],2*w-pt[j%n+1][0],2*h-pt[j%n+1][1],a,b);
		if(as>=9e17)printf("-1\n");
		else
		{
			long long fix=a+as,fiy=b+as;
			fix%=2*w;
			if(fix>w)fix=2*w-fix;
			fiy%=2*h;
			if(fiy>h)fiy=2*h-fiy;
			printf("%lld %lld %lld\n",as,fix,fiy);
		}
	}
}
```

##### T3 Aristocrat

###### Problem

有 $n$ 张桌子，第 $i$ 张桌子可以坐 $a_i$ 人

有 $m$ 个团队要来，每个团队的人数在 $[1,k]$ 间随机

每个团队会选择没有人且能坐下的一张桌子坐下，如果没有就离开

求最后所有桌子上人数总和的期望

$n,m\leq 100,k\leq 200$

###### Sol

将 $a_i$ 排序

考虑最后的情况，一定是有一些段被占满了

考虑 $dp_{i,j}$ 表示 $[i,j]$ 被占满的情况，但注意到在 $j=n$ 时会出现这个人数区间内有大于区间长度个团队的情况

一种写法是在右边再加 $m$ 张 $a_i=k$ 的桌子，这些桌子上的人数不计，这样就不考虑离开

设 $f_{i,j}$ 表示 $j-i+1$ 个团队占满 $[i,j]$ 的桌子，且区间左边的那张桌子没有团队的方案数， $g_{i,j}$ 表示所有方案桌子上的人数和

因为精度问题，可以存 $f_{i,j}/k^{j-i+1}$ ,相当于概率

转移时枚举最后一个团队占了哪张桌子，之前的 $j-i$ 个团队，可以发现左侧和右侧是独立的，所以可以乘上组合数合并，再乘上最后这个团队可能的人数

算出 $f,g$ 后，设 $dp_{i,j}$ 表示前 $i$ 张桌子，有 $j$ 张桌子有人的概率以及概率乘人数

转移时可能一张桌子没人，或者连续一段有人

复杂度 $O((n+m)^3)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 205
int n,t,m,v[N];
double dp[N][N],su[N][N],dp2[N][N],su2[N][N],c[N][N];
int main()
{
	scanf("%d%d%d",&n,&t,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=0;i<=n+m;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=n+m;i++)
	for(int j=1;j<i;j++)
	c[i][j]=c[i-1][j]+c[i-1][j-1];
	sort(v+1,v+n+1);
	for(int i=1;i<=n;i++)if(v[i]>t)v[i]=t;
	for(int i=n+1;i<=n+m;i++)v[i]=t;
	for(int i=1;i<=n+m;i++)dp[i][i-1]=1;
	for(int l=0;l<n+m;l++)
	for(int i=1;i+l<=n+m;i++)
	if(v[i+l]-v[i-1])
	{
		int j=i+l;
		for(int k=i;k<=j;k++)
		{
			double f1=c[j-i][k-i]*(v[k]-v[i-1])/t,f2=k<=n?((v[k]+v[i-1]+1)/2.0):0;
			dp[i][j]+=dp[i][k-1]*dp[k+1][j]*f1;
			su[i][j]+=f1*(f2*dp[i][k-1]*dp[k+1][j]+su[i][k-1]*dp[k+1][j]+su[k+1][j]*dp[i][k-1]);
		}
	}
	dp2[0][0]=1;
	for(int i=0;i<=n+m;i++)
	for(int j=0;j<=m;j++)
	{
		dp2[i+1][j]+=dp2[i][j];
		su2[i+1][j]+=su2[i][j];
		for(int k=i+1;k<=n+m;k++)
		if(j+(k-i)<=m&&(v[k]-v[i]))
		{
			dp2[k+1][j+k-i]+=dp2[i][j]*dp[i+1][k]*c[j+k-i][j];
			su2[k+1][j+k-i]+=(su2[i][j]*dp[i+1][k]+su[i+1][k]*dp2[i][j])*c[j+k-i][j];
		}
	}
	printf("%.10lf\n",su2[n+m+1][m]);
}
```

#### 十连测 Day 9

##### T1 Farm of Monsters

###### Problem

有 $n$ 个怪物，第 $i$ 个血量为 $v_i$

两个人轮流攻击，怪物血量 $\leq 0$ 时死亡，最后一次攻击的人得1分

你一次攻击造成的伤害为 $a$ ,你对手一次攻击造成的伤害是 $b$

你对手每次只会攻击最左边活着的一个

你先手，求最优策略下你的最大得分

$n\leq 10^5$

###### Sol

对于每个怪，只有两种策略

1.让对手攻击 $⌈v_i/b⌉$ 次，对手得1分

2.你攻击 $⌈(((v_i-1)\bmod b)+1)/a⌉$ 次，对手攻击 $⌈(v_i-⌈(((v_i-1)\bmod
b)+1)/a⌉∗a)/b⌉$  次且你攻击最后一次得1分

考虑哪些方案是合法的，假设对于一种策略，对于第 $i$ 个你要攻击 $a_i$ 次，对手要攻击 $b_i$ 次

可以发现，如果对于每一个 $i$ ,$\sum_{j=1}^i a_j-b_j \leq -1$ 的话，总可以每次找到最左边一个需要攻击的，使得最后达到目标

否则，如果存在 $i,\sum_{j=1}^ia_j-b_j>-1$ ，在对手进行了前 $\sum_{j=1}^i b_j$ 步后，你需要进行 $\sum_{j=1}^i a_j$ 步才能拿到前 $i$ 个中你想要拿到的分，而这不可能

设策略1中对手攻击 $b_i$ 次，策略2中你要攻击 $a_j^{'}$ 次，对手要攻击 $b_j^{'}$ 次

因此可以看成选一些数，使得对于每一个 $i$ ，你选的小于等于 $i$ 的数满足 $\sum a_j^{'}-b_j^{'}+b_j\leq -1+\sum_{j=1}^i b_j$

可以看成有 $n$ 个 $s_i$ 和 $f_i\ $，选一些数，使得对于每一个 $i$ 选的小于等于 $i$ 的数满足 $\sum s_j\leq f_i \ $

一种做法是，维护当前选的集合，从小到大考虑，每次尝试加入这个数，再弹掉最小的直到合法

因此，可以发现最小的一定优先选，因此也可以线段树，每次尝试加入当前的数并判断合法

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 300140
int n,a,b,v[N],s[N][3],v2[N];
struct segt{int l,r;long long vl,mn;}e[N*4];
void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1].vl+e[x<<1|1].mn);e[x].vl=e[x<<1].vl+e[x<<1|1].vl;}
bool cmp(int a,int b){return s[a][0]==s[b][0]?a<b:s[a][0]<s[b][0];}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;
	if(l==r){e[x].vl=e[x].mn=s[l][2];return;}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	pushup(x);
}
void modify(int x,int s,int v)
{
	if(e[x].l==e[x].r){e[x].vl+=v;e[x].mn+=v;return;}
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=s)modify(x<<1,s,v);
	else modify(x<<1|1,s,v);
	pushup(x);
}
int main()
{
	scanf("%d%d%d",&n,&a,&b);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int i=1;i<=n;i++)s[i][0]=(v[i]-1)%b/a+1,s[i][1]=(v[i]-1)/b,s[i][2]=(v[i]-1)/b+1,v2[i]=i;
	sort(v2+1,v2+n+1,cmp);build(1,1,n);
	int as=0;
	for(int i=1;i<=n;i++)
	{
		modify(1,v2[i],-1-s[v2[i]][0]);
		if(e[1].mn<-1)modify(1,v2[i],1+s[v2[i]][0]);
		else as++;
	}
	printf("%d\n",as);
}
```

##### T2 Airplane Cliques

###### Problem

给一棵 $n$ 个点的树，对于每个 $i$ ，求有多少个大小为 $i$ 的集合满足任意两个点距离不超过 $k$ ，模 $998244353 \ $

$n\leq 3\times 10^5,6s$

###### Sol

考虑找代表元计数

考虑这样的顺序：深度不同按深度排，深度相同任意

可以发现bfs序满足这个顺序

对于一个集合，在顺序下的最后一个点计数

考虑可能出现在包含这个点的集合里面的数，显然是顺序在它之前且和它距离不超过 $k$ 的

对于当前点 $x$ 和两个之前的点 $a,b$ ，考虑三种情况

![](C:\Users\zz\Documents\pic\58.png)

因为 $dep_a,dep_b\leq dep_x$ ，如果 $dis_{a,b}>k$ ,可以推出 $dis_{a,x}>k,dis_{b,x}>k$ 中的一个，矛盾

因此，所有这样的点都可以任意出现在包含这个点的集合中

如果对于每个 $u$ 求出了这样的点数 $c_u$ ，那显然有 $ans_i=\sum_{u=1}^nC_{c_u}^{i-1}$

这显然是一个卷积形式，可以fft解决

那么只需要求出所有的 $c_u$

考虑动态点分治，按bfs序加入点，每次在点分树上查 $dis\leq k$ 的点数

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 1050501
#define mod 998244353
int n,d,a,b,dep[N],f[N],as[N],head[N],cnt,id[N],tid[N],vl,f1,f2,vis[N],ds[N][31],sz[N],fr[N],ifr[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void bfs()
{
	queue<int> tp;
	tp.push(1);
	int ct=0;
	while(!tp.empty())
	{
		int s=tp.front();tp.pop();
		id[s]=++ct;tid[ct]=s;
		for(int i=head[s];i;i=ed[i].next)if(!id[ed[i].t])tp.push(ed[i].t);
	}
}
void dfs1(int u,int fa)
{
	sz[u]=1;int mx=0;
	for(int i=head[u];i;i=ed[i].next)
	if(!vis[ed[i].t]&&ed[i].t!=fa)
	{
		dfs1(ed[i].t,u);
		sz[u]+=sz[ed[i].t];
		if(sz[ed[i].t]>mx)mx=sz[ed[i].t];
	}
	if(vl-sz[u]>mx)mx=vl-sz[u];
	if(mx<f1)f1=mx,f2=u;
}
void dfs2(int u,int fa,int de)
{
	ds[u][de]=ds[fa][de]+1;
	sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&ed[i].t!=fa)dfs2(ed[i].t,u,de),sz[u]+=sz[ed[i].t];
}
void dfs3(int u)
{
	ds[0][dep[u]]=-1;dfs2(u,0,dep[u]);
	vis[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		f1=1e7,vl=sz[ed[i].t];dfs1(ed[i].t,u);
		dep[f2]=dep[u]+1;f[f2]=u;dfs3(f2);
	}
}
#define M 19260817
int ch[M][2],fa[M],v[M],val[M],su[M],ct3;
struct Splay{
	int rt;
	void pushup(int x){su[x]=su[ch[x][0]]+su[ch[x][1]]+val[x];}
	void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
	void splay(int x){while(fa[x]){int f=fa[x],g=fa[f];if(g)rotate((ch[f][0]==x)^(ch[g][0]==f)?x:f);rotate(x);}rt=x;}
	void ins(int x,int v1,int v2){bool tp=v[x]<v1;if(!ch[x][tp]){ch[x][tp]=++ct3;fa[ct3]=x;v[ct3]=v1;val[ct3]=v2;su[ct3]=v2;splay(ct3);return;}ins(ch[x][tp],v1,v2);}
}tr1[N],tr2[N];
int v1[N],v2[N],ntt[N],rev[N];
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*s/2),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int vl1=pw(3,(mod-1+t*(mod-1)/i)%(mod-1));
		for(int j=0;j<s;j+=i)
		for(int k=j,st1=1;k<j+(i>>1);k++,st1=1ll*st1*vl1%mod)
		{
			int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*st1%mod;
			ntt[k]=(v1+v2)%mod;
			ntt[k+(i>>1)]=(v1-v2+mod)%mod;
		}
	}
	int inv=t==-1?pw(s,mod-2):1;
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d%d",&n,&d);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dep[1]=1;bfs();dfs3(1);
	for(int i=1;i<=n;i++){tr1[i].rt=++ct3;v[ct3]=1e9;tr2[i].rt=++ct3;v[ct3]=1e9;}
	for(int i=1;i<=n;i++)
	{
		int st=tid[i],s1=st,ls=0;as[i]=1;
		while(st)
		{
			int fuc=d-ds[s1][dep[st]];
			int v1=tr1[st].rt,las=0;
			while(v1)
			{
				las=v1;
				if(v[v1]<=fuc)as[i]+=val[v1]+su[ch[v1][0]],v1=ch[v1][1];
				else v1=ch[v1][0];
			}
			if(las)tr1[st].splay(las);
			if(ls)
			{
				int v1=tr2[ls].rt,las=0;
				while(v1)
				{
					las=v1;
					if(v[v1]<=fuc)as[i]-=val[v1]+su[ch[v1][0]],v1=ch[v1][1];
					else v1=ch[v1][0];
				}
				if(las)tr2[ls].splay(las);
				tr2[ls].ins(tr2[ls].rt,ds[s1][dep[st]],1);
			}
			tr1[st].ins(tr1[st].rt,ds[s1][dep[st]],1);
			ls=st;st=f[st];
		}
	}
	fr[0]=ifr[0]=1;
	for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=n;i++)v1[as[i]-1]=(v1[as[i]-1]+fr[as[i]-1])%mod,v2[n-i]=ifr[i];
	v2[n]=1;
	int l=1;while(l<n*2)l<<=1;
	dft(l,v1,1);dft(l,v2,1);for(int i=0;i<l;i++)v1[i]=1ll*v1[i]*v2[i]%mod;dft(l,v1,-1);
	for(int i=0;i<n;i++)printf("%d ",1ll*v1[i+n]*ifr[i]%mod);
}
```

##### T3 Horrible Circles

###### Problem

有一个二分图，两边各有 $n$ 个点，左边的第 $i$ 个点连向了右边的前 $a_i$ 个点，求简单环数，模 $998244353$

$n\leq 5000$

###### Sol

可以通过这两种操作建出图：

1.向右侧加入一个孤立点

2.向左侧加入一个点，这个点向右边所有点连边

考虑一个环在这个过程中的变化

对于1操作，它可能向这个环中加入了一个孤立的部分

对于2操作，它可能合并了两个之前的部分

注意到这样的每一部分都是一条链，因此2操作也可以将一条链连成一个环

注意到转移只和部分的数量有关，和长度无关

设 $dp_{i,j}$ 表示考虑了前 $i$ 个操作，当前有 $j$ 个部分的方案数

对于操作1，有 $dp_{i,j}=dp_{i-1,j}+dp_{i-1,j-1}$ ,表示是否加入

对于操作2，有 $dp_{i,j}=dp_{i-1,j}+dp_{i-1,j-1}*j*(j-1)$ ,表示选两个出来合并，然后答案加上 $dp_{i-1,1}$

这样算的是有向环的数量，因此最后答案要除以2

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 5050
#define mod 998244353
int n,dp[N],as,v[N];
void add0(int a){for(int i=a;i>=1;i--)dp[i]=(dp[i]+dp[i-1])%mod;}
void add1(int a){as=(as+dp[1])%mod;for(int i=1;i<=a;i++)dp[i]=(dp[i]+1ll*dp[i+1]*(i+1)*i)%mod;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),as=(as-v[i]+mod)%mod;
	sort(v+1,v+n+1);dp[0]=1;
	for(int i=1;i<=n;i++){for(int j=v[i-1]+1;j<=v[i];j++)add0(j);add1(v[i]);}
	printf("%d\n",1ll*as*499122177%mod);
}
```

#### 十连测 Day 10

##### T1 Easy Win

###### Problem

有 $n$ 堆石子，第 $i$ 堆有 $a_i$ 个，求出对于 $[1,n]$ 的所有 $k$ ，进行每个人最多取 $k$ 个的nim游戏，双方最优策略下先手胜还是后手胜

 $n,a_i\leq 5\times 10^5$

###### Sol

考虑一堆的sg值，显然是 $a_i \bmod (k+1)$

因此只需要算出这个的异或值

设 $v_i=(\sum_{j=1}^n[a_j==i])\bmod 2$ ,那么 $ans_k$ 等于所有 $v_i*(i\bmod (k+1))$ 的异或

考虑拆成若干段形如 $v_i*(i-l),i\in[l,r]$ 的异或和的形式，因为调和级数所有 $k$ 这样的段数只有 $O(n\log n)$ 段

考虑一段 $[l,r]$ 设 $a=\log_2 (r-l+1)$  

分两部分考虑

第一部分是 $[l,l+2^a-1]$ 的值

第二部分是 $v_i*(i-l),i\in[l+2^a,r]$ 的异或

显然在右边 $i-l$ 的最高位是 $2^a$ 

考虑 $2^a$ 位上的贡献，这只和 $\sum_{i=l+2^a}^r v_i$ 的奇偶性有关

考虑低位的贡献，可以直接把 $i-l$ 中的 $2^a$ 位去掉，得到 $v_i*(i-l-2^a),i\in[l+2^a,r]$ 的异或

这显然是一个子问题，只会递归 $O(\log n)$ 次

对于第一部分的值倍增预处理，总复杂度 $O(n\log^2 n)$ ,实际上因为很多区间都很小使得常数很小

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 505000
int n,a,v[N],su[N],lg[N],f[N][20];
int query(int l,int r)
{
	int as=0;
	if(l==r)return 0;
	for(int i=lg[r-l+1];i>=0;i--)
	if(l+(1<<i)-1<=r)as^=f[l][i]^((su[r]^su[l+(1<<i)-1])<<i),l+=1<<i;
	return as;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&a),v[a]^=1;
	for(int i=2;i<=n+1;i++)lg[i]=lg[i>>1]+1;
	for(int i=1;i<=n;i++)su[i]=su[i-1]^v[i];
	for(int i=1;i<=19;i++)
	for(int j=0;j+(1<<i)-1<=n;j++)
	f[j][i]=f[j][i-1]^f[j+(1<<i-1)][i-1]^((su[j+(1<<i)-1]^su[j+(1<<i-1)-1])<<i-1);
	query(0,6);
	for(int i=1;i<=n;i++)
	{
		int s1=0;
		for(int j=0;j<=n;j+=i+1)
		{
			int r=j+i;
			if(r>n)r=n;
			s1^=query(j,r);
		}
		printf("%s ",s1?"Alice":"Bob");
	}
}
```

##### T2 Cells Blocking

不会

##### T3 Giant Penguin

###### Problem

给一棵仙人掌，每个点最多在 $k$ 个简单环中，两种操作

1.标记一个节点

2.给一个点，询问最近的标记节点的距离

$n\leq 10^5,m\leq 2\times 10^5,k\leq 10,3s$

###### Sol

考虑 $k=0$ 的情况，这是一个经典的点分治问题，只需要对于每一层维护最近点即可

因为这里没有删除，复杂度是 $O(n\log n)$

考虑到仙人掌上，一种标准的做法是，每次找一棵生成树的重心，再选 $k-1$ 个点，将剩下的点分开

然后bfs算每个选的点到其它点的距离，最后询问部分和点分类似

复杂度 $O((n+m)k\log n)$ 

~~问题在于很难写~~

有一种奇妙的写法：

每次任取一棵生成树，取重心，bfs更新这个点到其它点的距离，然后删掉这个点，对于剩下的连通块做

这样的复杂度大概也是 $O((n+m)k\log n)$ 的，~~而且很好写~~

###### Code

```cpp
#include<cstdio>
#include<queue>
using namespace std;
#define N 105050
struct edge{int t,next;}ed[N*4];
int n,m,k,head[N],cnt,is[N],dis[N],a,b,q,ds1[801][N],vis1[N],f1[N],sz[N],vl,as,s1,vis[N],dep[N],fu[N],ti;
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa)
{
	int mx=0;sz[u]=1;vis1[u]=ti;
	for(int i=head[u];i;i=ed[i].next)
	if(!vis[ed[i].t]&&ed[i].t!=fa&&vis1[ed[i].t]!=ti)
	{
		dfs1(ed[i].t,u);sz[u]+=sz[ed[i].t];
		if(mx<sz[ed[i].t])mx=sz[ed[i].t];
	}
	if(mx<vl-sz[u])mx=vl-sz[u];
	if(as>mx)as=mx,s1=u;
}
void bfs2(int u,int d)
{
	queue<int> st;
	ds1[d][u]=1;st.push(u);
	while(!st.empty())
	{
		int s=st.front();st.pop();
		for(int i=head[s];i;i=ed[i].next)if(!vis[ed[i].t]&&!ds1[d][ed[i].t])ds1[d][ed[i].t]=ds1[d][s]+1,st.push(ed[i].t);
	}
}
void dfs3(int u)
{
	bfs2(u,dep[u]);vis[u]=1;
	queue<int> fu1;
	int fuc=ti;ti++;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&vis1[ed[i].t]<=fuc){ti++;dfs1(ed[i].t,u);ti++;vl=sz[ed[i].t];as=1e9;dfs1(ed[i].t,u);dep[s1]=dep[u]+1;f1[s1]=u;fu1.push(s1);}
	while(!fu1.empty())dfs3(fu1.front()),fu1.pop();
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=n;i++)fu[i]=1e8;
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b);
	dep[1]=1;dfs3(1);
	scanf("%d",&q);
	while(q--)
	{
		scanf("%d%d",&a,&b);
		if(a==1)
		{for(int i=b;i;i=f1[i])if(fu[i]>ds1[dep[i]][b])fu[i]=ds1[dep[i]][b];} 
		else
		{
			int as=1e8;
			for(int i=b;i;i=f1[i])if(as>ds1[dep[i]][b]+fu[i])as=ds1[dep[i]][b]+fu[i];
			printf("%d\n",as-2);
		}
	}
}
```

#### 寒假集训 Day 1

##### T1 垃圾题

###### Problem

开幕雷击

给定一个长度为 $n$ 的数列 $a_1,a_2,…,a_n$ 和长度为 $5$ 的数列 $b_1,b_2,…,b_5$

定义两个数列 $x_1,x_2,…,x_p$ 与 $y_1,y_2,…,y_q$ 匹配当且仅当$p=q$且 $xi=xj⇔yi=yj (1≤i,j≤p)$。也就是说 $y$ 中元素的相同关系和 $x$ 中元素的相同关系一致。

请问 $a$ 中有多少个子序列和 $b$ 能匹配。两个子序列不同当且仅当存在至少一个位置在原序列的下标不同

$n\leq 3000,2s$

###### Sol

开幕雷击

设 $b$ 中有 $k$ 种不同元素

$k=1$ 直接做 $O(k)$

$k=2$ 考虑枚举两种颜色，把那些位置提出来，做子序列dp 复杂度 $O(k^2)$

$k=5$ 相当于选出5个不同数的方案数 $O(k)$

$k=3$ 考虑找到一个只出现了一次的颜色 $c$ ，在 $k=2$ 的做法上，把其它的都看做第三种颜色，做子序列dp

如果把相邻的颜色3缩起来，复杂度还是 $O(k^2)$

$k=4$ 考虑找到两个只出现过一次且相邻的颜色 $c,d$

考虑容斥，算这两个可能相同可能不相同的方案减去相同的方案，相同的就是 $k=3$ 的情况，对于不相同的，枚举另外两种颜色 $a,b$ ，把其它的看做第三种，那么相当于这两个位置都是第三种颜色，做一个子序列dp

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<map>
using namespace std;
#define N 6050
vector<int> fu[N];
int su[N][N],f[N][N],n,v[N],b[N],ct,v2[N],v3[N],v4[N],sb[N],ct2,is[N],fl[N][4],fr[N][4],ai[N],ct3,as,st[N],st2[N],sz[N],b2[N];
long long g[N][N],dp[N][10];
map<int,int> q;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),fu[v[i]].push_back(i);
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)su[i][j]=su[i-1][j];
		su[i][v[i]]++;
	}
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)v2[j]=0;
		int as1=0;long long as2=0;
		for(int j=i;j<=n;j++)
		{
			if(!v2[v[j]])as1++;
			as2+=1+2*v2[v[j]];
			v2[v[j]]++;
			f[i][j]=as1,g[i][j]=as2;
		}
	}
	for(int j=1;j<=n;j++)v2[j]=0;
	for(int i=1;i<=5;i++)
	{
		scanf("%d",&b[i]);
		if(q[b[i]])b[i]=q[b[i]];
		else q[b[i]]=++ct2,b[i]=ct2;
	}
	long long as=0;
	if(ct2==1)
	{
		for(int i=1;i<=n;i++)
		{
			int s1=fu[i].size();
			as+=1ll*s1*(s1-1)/2*(s1-2)/3*(s1-3)*(s1-4)/20;
		}
		printf("%lld\n",as);
		return 0;
	}
	if(ct2==2)
	{
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)
		if(i!=j)
		{
			int s1=fu[i].size(),s2=fu[j].size();
			ct=s1+s2;
			int l1=0,l2=0;
			for(int k=1;k<=ct;k++)
			if(l1==s1)v2[k]=fu[j][l2],v3[k]=2,l2++;
			else if(l2==s2)v2[k]=fu[i][l1],v3[k]=1,l1++;
			else if(fu[j][l2]>fu[i][l1])v2[k]=fu[i][l1],v3[k]=1,l1++;
			else v2[k]=fu[j][l2],v3[k]=2,l2++;
			for(int k=0;k<=ct;k++)
			for(int l=0;l<=7;l++)dp[k][l]=0;
			dp[0][0]=1;
			for(int k=1;k<=ct;k++)
			{
				for(int l=0;l<=5;l++)dp[k][l]=dp[k-1][l];
				for(int l=1;l<=5;l++)if(v3[k]==b[l])dp[k][l]+=dp[k-1][l-1];
			}
			as+=dp[ct][5];
		}
		printf("%lld\n",as);
		return 0;
	}
	if(ct2==3)
	{
		int ct3=0,fg2=0,ct=0;
		for(int i=1;i<=5;i++)
		for(int j=i+1;j<=5;j++)
		if(b[i]==3&&b[j]==3)fg2^=3,ct++;
		for(int i=1;i<=5;i++)
		for(int j=i+1;j<=5;j++)
		if(b[i]==2&&b[j]==2)fg2^=2,ct++;
		for(int i=1;i<=5;i++)
		for(int j=i+1;j<=5;j++)
		if(b[i]==1&&b[j]==1)fg2^=1,ct++;
		if(ct==3)fg2=fg2%3+1;
		for(int i=1;i<=5;i++)if(b[i]==fg2)b[++ct3]=3;else b[++ct3]=b[i]-(b[i]>fg2);
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)
		if(i!=j)
		{
			int s1=fu[i].size(),s2=fu[j].size();
			ct=s1+s2;
			int l1=0,l2=0;
			for(int k=1;k<=ct;k++)
			if(l1==s1)v2[k]=fu[j][l2],v3[k]=2,l2++;
			else if(l2==s2)v2[k]=fu[i][l1],v3[k]=1,l1++;
			else if(fu[j][l2]>fu[i][l1])v2[k]=fu[i][l1],v3[k]=1,l1++;
			else v2[k]=fu[j][l2],v3[k]=2,l2++;
			for(int k=ct;k>=1;k--)v2[k*2]=v2[k],v2[k]=0,v3[k*2]=v3[k],v3[k]=0,sz[k*2]=1;
			v3[ct*2+1]=3,sz[ct*2+1]=n-v2[ct*2];
			for(int k=1;k<=ct;k++)v3[k*2-1]=3,sz[k*2-1]=v2[k*2]-v2[k*2-2]-1;
			ct=ct*2+1;
			for(int k=0;k<=ct;k++)
			for(int l=0;l<=7;l++)dp[k][l]=0;
			dp[0][0]=1;
			for(int k=1;k<=ct;k++)
			{
				for(int l=0;l<=5;l++)dp[k][l]=dp[k-1][l];
				for(int l=1;l<=5;l++)if(v3[k]==b[l])dp[k][l]+=dp[k-1][l-1]*sz[k];
			}
			as+=dp[ct][5];
		}
		printf("%lld\n",as);
		return 0;
	}
	if(ct2==4)
	{
		int fg2=0;
		for(int i=1;i<=5;i++)for(int j=i+1;j<=5;j++)if(b[i]==b[j])fg2=b[i];
		for(int i=1;i<=5;i++)if(b[i]==fg2)b[++ct3]=1;else b[++ct3]=b[i]-(b[i]>fg2)+1;
		for(int i=1;i<=5;i++)b2[i]=b[i];
		for(int i=1;i<=5;i++)if(b[i]==4)b[i]=2;
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)
		if(i!=j)
		{
			int s1=fu[i].size(),s2=fu[j].size();
			ct=s1+s2;
			int l1=0,l2=0;
			for(int k=1;k<=ct;k++)
			if(l1==s1)v2[k]=fu[j][l2],v3[k]=3,l2++;
			else if(l2==s2)v2[k]=fu[i][l1],v3[k]=1,l1++;
			else if(fu[j][l2]>fu[i][l1])v2[k]=fu[i][l1],v3[k]=1,l1++;
			else v2[k]=fu[j][l2],v3[k]=3,l2++;
			for(int k=ct;k>=1;k--)v2[k*2]=v2[k],v2[k]=0,v3[k*2]=v3[k],v3[k]=0,sz[k*2]=1;
			v3[ct*2+1]=2,sz[ct*2+1]=n-v2[ct*2];
			for(int k=1;k<=ct;k++)v3[k*2-1]=2,sz[k*2-1]=v2[k*2]-v2[k*2-2]-1;
			ct=ct*2+1;
			for(int k=0;k<=ct;k++)
			for(int l=0;l<=7;l++)dp[k][l]=0;
			dp[0][0]=1;
			for(int k=1;k<=ct;k++)
			{
				for(int l=0;l<=5;l++)dp[k][l]=dp[k-1][l];
				for(int l=1;l<=5;l++)if(v3[k]==b[l])dp[k][l]+=dp[k-1][l-1]*sz[k];
			}
			as+=dp[ct][5];
		}
		for(int i=1;i<=n;i++)
		for(int j=1;j<=n;j++)
		if(i!=j)
		{
			int s1=fu[i].size(),s2=fu[j].size();
			ct=s1+s2;
			int l1=0,l2=0;
			for(int k=1;k<=ct;k++)
			if(l1==s1)v2[k]=fu[j][l2],v3[k]=2,l2++;
			else if(l2==s2)v2[k]=fu[i][l1],v3[k]=1,l1++;
			else if(fu[j][l2]>fu[i][l1])v2[k]=fu[i][l1],v3[k]=1,l1++;
			else v2[k]=fu[j][l2],v3[k]=2,l2++;
			for(int k=ct;k>=1;k--)v2[k*2]=v2[k],v2[k]=0,v3[k*2]=v3[k],v3[k]=0,sz[k*2]=1;
			v3[ct*2+1]=3,sz[ct*2+1]=n-v2[ct*2];
			for(int k=1;k<=ct;k++)v3[k*2-1]=3,sz[k*2-1]=v2[k*2]-v2[k*2-2]-1;
			ct=ct*2+1;
			for(int k=0;k<=ct;k++)
			for(int l=0;l<=7;l++)dp[k][l]=0;
			dp[0][0]=1;
			for(int k=1;k<=ct;k++)
			{
				for(int l=0;l<=5;l++)dp[k][l]=dp[k-1][l];
				for(int l=1;l<=5;l++)if(v3[k]==b[l])dp[k][l]+=dp[k-1][l-1]*sz[k];
			}
			as-=dp[ct][5];
		}
		printf("%lld\n",as);
		return 0;
	}
	if(ct2==5)
	{
		long long dp[7]={1,0,0,0,0,0,0};
		for(int i=1;i<=n;i++)v3[v[i]]++;
		for(int k=1;k<=n;k++)for(int l=5;l>0;l--)dp[l]+=dp[l-1]*v3[k];
		printf("%lld\n",dp[5]);
	}
}
```

##### T2 数论题

###### Problem

给定一个素数 $p$，记 $f(i)(2≤i≤p-1)$ 表示 $i$ 关于 $p$ 的逆元，也就是 $i∗f(i)≡1(\bmod p)$ ，并且要求 $1≤f(i)≤p-1$

求出所有的 $i$ 满足 $f(i)=min_{j=2}^i f(j)$

多组数据，$T≤500,p≤10^{16},2s$ ,数据随机

###### Sol

如果 $k$ 满足 $k|p+1$ ,那么 $f(k)=(p+1)/k$

因为 $f(i)≥(p+1)/k$ ,所以这样的 $k$ 一定满足要求

因此， $p+1$ 的因子全部满足要求

在处理完 $p+1$ 的因子后，下一个最有可能的是 $2p+1$ ，然后是 $3p+1$ ,…

考虑枚举到 $kp+1$ ,如果对于当前答案相邻的两个 $x,y ,f(x)≤(kp+1)/(y-1)$ ,那么这一段中间不可能再有数了

这样最后可以发现只会枚举到几十

分解用 Pollard-Rho,复杂度 $O(Tn^{1/4}∗k)$

###### Code

~~太丑了就不放了~~

~~1.96s~~

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<set>
#include<map>
using namespace std;
#define ll long long
#define LL long long
ll p,awsl;
int T;
ll mul(ll x,ll y,ll mod){ll tmp=(long double)x*y/mod;return (x*y-tmp*mod+mod)%mod;}
vector<pair<ll,ll> > fu1,fu2;
ll pw(ll a,ll p,ll k){ll as=1;while(p){if(p&1)as=mul(as,a,k);a=mul(a,a,k);p>>=1;}return as;}
set<ll> st;
set<pair<ll,ll> > as;
ll f[11]={2,3,7,61,24251};
bool mrtest(ll f,ll x){for(ll tmp=x-1;tmp>1;tmp>>=1){ll tmp2=pw(f,tmp,x);if(tmp2==x-1)return 0;if(tmp2!=1)return 1;if(tmp&1)return 0;}}
bool mr(ll x){if(x==1)return 0;if(x==3||x==7||x==2||x==61||x==24251)return 1;for(int i=0;i<5&&f[i]<x;i++)if(mrtest(f[i],x))return 0;return 1;}
struct lsjrho{
	inline LL _rand(LL x, LL c, LL mod) {
    return (mul(x, x, mod) + c) % mod;
}

inline LL _rand() {
    return (LL)rand() << 48 | (LL)rand() << 32 | rand() << 16 | rand();
}

inline LL _abs(LL x) {
    return x >= 0 ? x : -x;
}

LL gcd(LL a, LL b) {
    return b ? gcd(b, a % b) : a;
}

inline LL Pollard_Rho(LL n) {
    LL s = 0, t = 0, c = _rand() % (n - 1) + 1, val = 1;
    for (int cir = 1; ; cir <<= 1, s = t, val = 1) {
        for (int i = 0; i < cir; i++) {
            t = _rand(t, c, n), val = mul(val, _abs(t - s), n);
            if (i % 127 == 0) {
                LL g = gcd(val, n);
                if (g != 1) return g;
            } 
        }
        LL g = gcd(val, n);
        if (g != 1) return g;
    }
}


inline void Factor(LL n) {
    if (n == 1) return;
    if (mr(n)) return st.insert(n), void();
    LL d = n;
    while (d == n) d = Pollard_Rho(n);
    while (n % d == 0) n /= d;
    Factor(n), Factor(d);
}
}d;
void dfs(ll x,int dep)
{
	set<ll>::iterator it=st.begin();
	for(int i=1;i<dep;i++)it++;
	if(it==st.end()){if(x>1&&x<p)as.insert(make_pair(x,pw(x,p-2,p)));return;}
	ll tp=*it,ct=0,s2=awsl;
	while(s2%tp==0)s2/=tp,ct++;
	for(int i=0;i<=ct;i++)dfs(x,dep+1),x*=tp;
}
int main()
{
	scanf("%d",&T);
	while(T--)
	{
		scanf("%lld",&p);
		st.clear();as.clear();
		for(int i=1;;i++)
		{
			st.clear();awsl=p*i+1;
			d.Factor(p*i+1);
			dfs(1,1);
			long long las=1;
			double f=0;
			for(set<pair<ll,ll> >::iterator it=as.begin();it!=as.end();it++)
			{
				f=max(f,1.0*las*((*it).first)/p);las=(*it).second;
			}
			if(f<=1.0*(i+1)||i>30+(p<1e15)*5)break;
		}
		long long as1=1e17;fu1.clear();fu2.clear();
		for(set<pair<ll,ll> >::iterator it=as.begin();it!=as.end();it++)
		{
			ll inv=(*it).second,i=(*it).first;
			if(i>inv)
			break;
			if(inv<as1){as1=inv;fu2.push_back(make_pair(i,inv));if(i!=inv)fu1.push_back(make_pair(inv,i));}
		}
		printf("%d\n",fu1.size()+fu2.size());
		int s2=fu2.size();
		for(int i=0;i<s2;i++)printf("%lld %lld\n",fu2[i].first,fu2[i].second);
		int s1=fu1.size();
		for(int i=s1-1;i>=0;i--)printf("%lld %lld\n",fu1[i].first,fu1[i].second);
	}
}
```

##### T3 二分题

###### Problem

给你一个 $n$ 个点的树，每条边有边权。求 $k$ 个**不同**的点 $p_1,p_2,…,p_k$ 满足

$∑_{i=1}^kdis(p_i,p_{i\bmod k+1})$

最大。这里 $dis(u,v)$ 表示 $u$ 和 $v$ 在树上的距离。

输出答案。

$n,k\leq 2\times 10^5,2|k$

###### Sol

对于一个点集，每条边最多走两边点数的min*2次

因此最优解一定是找到一个重心，答案是每个点到重心的距离和2倍

考虑选一个重心，然后相当于选 $k$ 个点，每个子树里面最多选一半，可以贪心求出答案

注意到如果最远的 $k$ 个中，有一个子树里面有大于 $k/2$ 个，那么将重心移过去，因为贪心得到的一定是那边选了 $k/2\ $ 个，所以这时不会有权值改变，因此移过去一定不会变差

因此考虑点分治确定重心，复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
using namespace std;
#define N 205000
int n,head[N],vis[N],sz[N],vl,tp,as,as1,v[N],a,b,c,cnt,sz2[N],k,v2[N],f1[N],ct[N];
long long dis[N],fu,as11,as12=0;
bool cmp(int i,int j){return dis[i]>dis[j];}
struct edge{int t,next,l;}ed[N*2];
void adde(int f,int t,int l){ed[++cnt]=(edge){t,head[f],l};head[f]=cnt;ed[++cnt]=(edge){f,head[t],l};head[t]=cnt;}
void dfs1(int u,int fa)
{
	sz[u]=1;
	int mx=0;
	for(int i=head[u];i;i=ed[i].next)
	if(!vis[ed[i].t]&&ed[i].t!=fa)dfs1(ed[i].t,u),mx=max(mx,sz[ed[i].t]),sz[u]+=sz[ed[i].t];
	mx=max(mx,vl-sz[u]);
	if(mx<tp)tp=mx,as=u;
}
void dfs2(int u,int fa,int fr){f1[u]=fr;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dis[ed[i].t]=dis[u]+ed[i].l,dfs2(ed[i].t,u,!fr?ed[i].t:fr);}
void dfs3(int u,int fa){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs3(ed[i].t,u),sz2[u]+=sz2[ed[i].t];}
void solve(int x)
{
	int ct3=k;
	dis[x]=0;dfs2(x,0,0);vis[x]=1;
	for(int i=1;i<=n;i++)v2[i]=i,sz2[i]=0,ct[i]=0;
	sort(v2+1,v2+n+1,cmp);
	as11=0;
	for(int i=1;i<=k;i++)sz2[v2[i]]=1;
	for(int i=1;i<=n&&ct3;i++){if(ct[f1[v2[i]]]+1>k/2)continue;as11+=dis[v2[i]],++ct[f1[v2[i]]];ct3--;}
	if(!ct3&&as12<as11)as12=as11;
	dfs3(x,0);
	for(int i=head[x];i;i=ed[i].next)
	if(sz2[x]<2*sz2[ed[i].t]&&!vis[ed[i].t])
	{
		dfs1(x,0);
		tp=1e7,vl=sz[ed[i].t];dfs1(ed[i].t,0);
		solve(as);
		break;
	}
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
	solve(1);printf("%lld\n",as12*2);
}
```

#### 寒假集训 Day 2

##### T1 我要打数论

###### Problem

求 $∑_{i=1}^n∑_{j=1}^nmin⁡(n,lcm(i,j)+gcd⁡(i,j)) \bmod p$ .多组数据， $p$ 给定且不变

$n≤10^6$

###### Sol

先枚举gcd

$ans=∑_{g=1}^n∑_{i=1}^{n/g}∑_{j=1}^{n/g}min⁡(n,gij+g)
[gcd⁡(i,j)=1]\ $

这个时候再拆就去世了

考虑枚举 $ij+1$

$ans=\sum_{g=1}^n(\sum_{k=1}^{n/g}\sum_{i=1}^{n/g}\sum_{j=1}^{n/g}[gcd(i,j)==1][ij+1==k]gk+n*((n/g)^2-\sum_{k=1}^{n/g}\sum_{i=1}^{n/g}\sum_{j=1}^{n/g}[gcd(i,j)==1][ij+1==k]1))$

显然对于合法的 $i,j$ ,$i,j<ij+1\leq n/g$ ，所以不用处理上界

设 $f(k)=\sum_{i=1}^{n}\sum_{j=1}^{n}[gcd(i,j)==1][ij+1==k]$

容易发现，它等于 $2^{k-1的质因子数}$

因此可以线性筛出 $f$

然后设 $g(s)=\sum_{k=1}^{s}\sum_{i=1}^{s}\sum_{j=1}^{s}[gcd(i,j)==1][ij+1==k]k,h(s)=s^2-\sum_{k=1}^{s}\sum_{i=1}^{s}\sum_{j=1}^{s}[gcd(i,j)==1][ij+1==k]1$

$g(s)=\sum_{k=1}^s k*f(k),h(s)=s^2-\sum_{k=1}^s f(s)$ ,可以前缀和求

那么有 $ans=\sum_{i=1}^n(i*g(n/i)+n*h(n/i))$

枚举 $i$ 和 $n/i$ ,可以发现对应的 $n$ 是一段区间，前缀和即可

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 1006000
int T,p,n,ct,pw[N],ch[N],pr[N],s1[N],f[N],g[N];
long long su1[N],su2[N];
void prime(int n)
{
	for(int i=2;i<=n;i++)
	{
		if(!ch[i])pr[++ct]=i,s1[i]=1;
		for(int j=1;j<=ct&&1ll*i*pr[j]<=n;j++)
		{
			ch[i*pr[j]]=1;
			s1[i*pr[j]]=s1[i]+1;
			if(i%pr[j]==0){s1[i*pr[j]]=s1[i];break;}
		}
	}
}
int main()
{
	scanf("%d%d",&T,&p);
	prime(1e6);
	pw[0]=1;for(int i=1;i<=3000;i++)pw[i]=2ll*pw[i-1]%p;
	for(int i=2;i<=1e6;i++)f[i]=pw[s1[i-1]],g[i]=1ll*f[i]*i%p;
	for(int i=1;i<=1e6;i++)f[i]=(f[i]+f[i-1])%p,g[i]=(g[i]+g[i-1])%p;
	for(int i=1;i<=1e6;i++)
	for(int j=i,ct=1;j<=1e6;j+=i,ct++)
	{
		su1[j]=(su1[j]+1ll*g[ct]*i);
		if(j+i<=1e6)su1[j+i]=(su1[j+i]-1ll*g[ct]*i);
		su2[j]=(su2[j]+f[ct]);
		if(j+i<=1e6)su2[j+i]=(su2[j+i]-f[ct]);
	}
	for(int i=1;i<=1e6;i++)su1[i]=(su1[i]+su1[i-1])%p,su2[i]=(su2[i]+su2[i-1])%p;
	while(T--)
	{
		scanf("%d",&n);
		int as=(su1[n]+p)%p,c1=(1ll*n*n%p-su2[n]+p)%p;
		printf("%lld\n",(as+1ll*c1*n)%p);
	}
}
```

##### T2 我要打矩形

###### Problem

有 $n$ 个矩形排成了一个序列，定义一个矩形序列的权值为相邻两个矩形交的整点个数的乘积，如果只有一个矩形那么权值为1

求所有子序列的权值和模 $998244353$

$n\leq 10^5,3s$

###### Sol

设 $dp_i$ 表示以 $i$ 结尾的子序列的权值和

那么 $dp_i=1+\sum_{j=1}^{i-1}dp_j*intersect(i.j)$

矩形交内整点个数有一个方便的处理方法

对于每个 $i$ ，对它的矩形内部每个点加上 $dp_i$

然后计算一个 $dp_i$ 时询问这个矩形内部点权值和

考虑cdq分治，每次计算一段对另外一段的贡献

那么问题变为二维加二维查

对询问差分，相当于求一个左下角的区间权值和

再考虑对修改差分，变成若干个向右上角的修改

那么一个询问 $(a,b)$ ,修改 $(c,d)$ 权值为 $[a\geq c][b\geq d]ab-ad-bc+bd$

这是一个二维数点，按x坐标排序，然后分别树状数组维护 $1,a,b,ab$ 的系数即可

复杂度 $O(n\log^2 n)$

###### Code

~~特别难写~~

```cpp
#include<cstdio>
#include<vector>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 200050
#define M 20
#define mod 998244353
int n,s[N][4],as[N],v21[N];
struct doit{int x,l,r,v3;};
vector<doit> st[M][2];
struct que{int x,y,t,id,is1;friend bool operator <(que a,que b){return a.x<b.x;}};
vector<que> st2[M][2];
struct tr{
	int t[N];
	void init(){memset(t,0,sizeof(t));} 
	void add(int x,int y){if(y<0)y+=mod;for(int i=x;i<=n*2+1;i+=i&-i)t[i]=(t[i]+y)-(t[i]+y>=mod?mod:0);}
	int que(int x){long long as=0;for(int i=x;i;i-=i&-i)as+=t[i];return as%mod;}
}tr[4];
int calc(int i,int j)
{
	int l1=min(s[i][2],s[j][2])-max(s[i][0],s[j][0])+1,l2=min(s[i][3],s[j][3])-max(s[i][1],s[j][1])+1;
	if(l1<0||l2<0)return 0;
	return 1ll*l1*l2%mod;
}
void cdq(int l,int r,int d,int x)
{
	if(l==r)
	{
		as[l]=(as[l]+1)%mod;
		st[d][x].push_back((doit){s[l][0],s[l][1],s[l][3],as[l]});
		st[d][x].push_back((doit){s[l][2]+1,s[l][1],s[l][3],mod-as[l]});
		return;
	}
	int mid=(l+r)>>1;
	st2[d+1][0].clear();st2[d+1][1].clear();
	st[d+1][0].clear();st[d+1][1].clear();
	int s1=st2[d][x].size();
	for(int i=0;i<s1;i++)
	if(st2[d][x][i].id<=mid)st2[d+1][0].push_back(st2[d][x][i]);
	else st2[d+1][1].push_back(st2[d][x][i]);
	cdq(l,mid,d+1,0);
	int l1=0;s1=st[d+1][0].size();int s2=st2[d+1][1].size();
	for(int i=0;i<s2;i++)
	{
		while(l1<s1&&st[d+1][0][l1].x<=st2[d+1][1][i].x)
		{
			int v1=st[d+1][0][l1].x,v2=st[d+1][0][l1].l,v3=st[d+1][0][l1].v3;
			v1--,v2--;
			int fl=lower_bound(v21+1,v21+n*2+1,st[d+1][0][l1].l)-v21,fr=lower_bound(v21+1,v21+n*2+1,st[d+1][0][l1].r+1)-v21;
 			tr[0].add(fl,1ll*v1*v2%mod*v3%mod);
			tr[0].add(fr,mod-1ll*v1*st[d+1][0][l1].r%mod*v3%mod);
			tr[1].add(fl,1ll*v3*(mod-v2)%mod);
			tr[1].add(fr,1ll*v3*st[d+1][0][l1].r%mod);
			tr[2].add(fl,1ll*v3*(mod-v1)%mod);
			tr[2].add(fr,1ll*v3*v1%mod);
			tr[3].add(fl,v3);
			tr[3].add(fr,mod-v3);
			l1++;
		}
		int fy=lower_bound(v21+1,v21+n*2+1,st2[d+1][1][i].y)-v21;
		int as1=(tr[0].que(fy)+1ll*st2[d+1][1][i].x*tr[1].que(fy)+1ll*st2[d+1][1][i].y*tr[2].que(fy)+1ll*st2[d+1][1][i].x*st2[d+1][1][i].y%mod*tr[3].que(fy))%mod;
		as1=(as1*st2[d+1][1][i].t+mod)%mod;
		as[st2[d+1][1][i].id]=(as[st2[d+1][1][i].id]+as1)%mod;
	}
	for(int i=0;i<l1;i++)
	{
		int v1=st[d+1][0][i].x,v2=st[d+1][0][i].l,v3=st[d+1][0][i].v3;
		v1--,v2--;
		int fl=lower_bound(v21+1,v21+n*2+1,st[d+1][0][i].l)-v21,fr=lower_bound(v21+1,v21+n*2+1,st[d+1][0][i].r+1)-v21;
 		tr[0].add(fl,mod-1ll*v1*v2%mod*v3%mod);
		tr[0].add(fr,1ll*v1*st[d+1][0][i].r%mod*v3%mod);
		tr[1].add(fl,1ll*v3*v2%mod);
		tr[1].add(fr,mod-1ll*v3*st[d+1][0][i].r%mod);
		tr[2].add(fr,1ll*v3*(mod-v1)%mod);
		tr[2].add(fl,1ll*v3*v1%mod);
		tr[3].add(fr,v3);
		tr[3].add(fl,mod-v3);
	}
	cdq(mid+1,r,d+1,1);
	s1=st[d+1][0].size();s2=st[d+1][1].size();
	l1=0;int l2=0;
	for(int i=1;i<=s1+s2;i++)
	if(l1==s1)st[d][x].push_back(st[d+1][1][l2++]);
	else if(l2==s2)st[d][x].push_back(st[d+1][0][l1++]);
	else if(st[d+1][0][l1].x<st[d+1][1][l2].x)st[d][x].push_back(st[d+1][0][l1++]);
	else st[d][x].push_back(st[d+1][1][l2++]);
}
//(x-x0+1)(y-y0+1)
//xy+(x0-1)(y0-1)-xy0-yx0
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d%d%d",&s[i][0],&s[i][1],&s[i][2],&s[i][3]),v21[i*2-1]=s[i][1],v21[i*2]=s[i][3];
	sort(v21+1,v21+n*2+1);
	for(int i=0;i<4;i++)tr[i].init();
	for(int i=1;i<=n;i++)
	{
		st2[1][0].push_back((que){s[i][0]-1,s[i][1]-1,1,i});
		st2[1][0].push_back((que){s[i][0]-1,s[i][3],-1,i,0});
		st2[1][0].push_back((que){s[i][2],s[i][1]-1,-1,i});
		st2[1][0].push_back((que){s[i][2],s[i][3],1,i,0});
	}
	sort(st2[1][0].begin(),st2[1][0].end());
	cdq(1,n,1,0);
	int as1=0;
	for(int i=1;i<=n;i++)as1=(as1+as[i])%mod;
	printf("%d\n",as1);
}
```

##### T3 我覀

###### Problem

我们有一个序列 $a_{1...n}$，我们定义 $F(x)$ 为有几个整数序列 $b_{1...n}$，满足对于所有 $i$ 都有 $0≤b_i≤a_i$，且 $b_{1...n}$ xor 起来等于 $x$

现在你需要支持修改操作：将 $a_x$ 修改成 $y$ ，每次修改完后你需要输出 $F(a_1 xor a_2 ..xor a_n)$ 的值

由于答案可能很大，你只需要输出答案对 $998244353$ 取模后的值

$n,q\leq 30000,a_i,y\leq 10^9$

###### Sol

首先考虑单组询问

考虑最高位，有两种情况

1.每一个元素都最高位都与上界相同，这时先检查这一位的异或是否合法，如果合法，那么之后就与这一位无关了，可以删掉这一位继续，不合法这部分答案就是0

2.存在至少一个元素，这一位上界是1，这一位填的是0

设这一位是 $2^x$ ,那么这个元素可以填的范围是 $[0.2^x-1]$

那么可以发现，拿出一个这样的元素，剩下的任意填，只要 $2^x$ 位上合法，都可以通过调整拿出来的这个数调整到答案上

那么可以设 $dp_{i,0/1,0/1}$ 表示当前 $2^x$ 位是 $0/1$ ，当前有没有选一个元素出来

转移时枚举每个数 $2^x$ 位是 $0/1$ 即可

还可以不计第三维，算完减去没有选任何一个元素的方案

这样需要做 $O(\log v)$ 次，单次复杂度 $O(n\log v)$

注意到dp是一个很小的矩阵，因此可以将转移写成 $2\times 2$ 的矩阵乘法，这样就可以快速修改

复杂度 $O((n+q\log n)\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 30060
#define mod 998244353
struct segt{int l,r,su,v1,v2;}e[29][N*4];
int n,m,a,b,v[N],su[33],v2,inv[N];
void pushup(int d,int x){e[d][x].su=1ll*e[d][x<<1].su*e[d][x<<1|1].su%mod;e[d][x].v1=(1ll*e[d][x<<1].v1*e[d][x<<1|1].v1+1ll*e[d][x<<1].v2*e[d][x<<1|1].v2)%mod;e[d][x].v2=(1ll*e[d][x<<1].v1*e[d][x<<1|1].v2+1ll*e[d][x<<1].v2*e[d][x<<1|1].v1)%mod;}
void build(int d,int x,int l,int r)
{
	e[d][x].l=l;e[d][x].r=r;
	if(l==r)return;
	int mid=(l+r)>>1;
	build(d,x<<1,l,mid);build(d,x<<1|1,mid+1,r);
}
void modify(int d,int x,int s,int v1,int v2)
{
	if(e[d][x].l==e[d][x].r){e[d][x].su=e[d][x].v1=v1;e[d][x].v2=v2;return;}
	int mid=(e[d][x].l+e[d][x].r)>>1;
	if(mid>=s)modify(d,x<<1,s,v1,v2);
	else modify(d,x<<1|1,s,v1,v2);
	pushup(d,x);
}
int query(int s)
{
	int as=1;
	for(int i=27;i>=0;i--)
	{
		if((su[i]&1)^((s>>i)&1))as=(as+1ll*e[i][1].v2*inv[i])%mod;
		else as=(as+1ll*(e[i][1].v1-e[i][1].su+mod)*inv[i])%mod;
		if((su[i]&1)^((s>>i)&1))break;
	}
	return as;
}
int main()
{
	scanf("%d%d",&n,&m);
	inv[0]=1;
	for(int i=1;i<=n;i++)inv[i]=1ll*inv[i-1]*499122177%mod;
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),v2^=v[i];
	for(int i=0;i<=27;i++)build(i,1,1,n);
	for(int i=1;i<=n;i++)
	for(int j=0;j<=27;j++)
	if(v[i]&(1<<j))modify(j,1,i,1+(v[i]&((1<<j+1)-1))-(1<<j),1<<j),su[j]++;
	else modify(j,1,i,(v[i]&((1<<j+1)-1))+1,0);
	while(m--)
	{
		scanf("%d%d",&a,&b);v2^=v[a]^b;
		for(int j=0;j<=27;j++)if(v[a]&(1<<j))su[j]--;
		v[a]=b;
		for(int j=0;j<=27;j++)if(v[a]&(1<<j))su[j]++;
		for(int j=0;j<=27;j++)
		if(v[a]&(1<<j))modify(j,1,a,1+(v[a]&((1<<j+1)-1))-(1<<j),1<<j);
		else modify(j,1,a,(v[a]&((1<<j+1)-1))+1,0);
		printf("%d\n",query(v2));
	}
}
```

#### 寒假集训 Day 3

##### T1 点分治

###### Problem

给一棵树，求可能的点分树数量模 $998244353$

$n\leq 5000$

###### Sol

考虑合并两棵点分树

设在原图中加入一条边 $(u,v)$ ，可以发现只有点分树中覆盖 $u,v$ 的点可能受影响，这样的点是 $u,v$ 的祖先

可以发现，加入一条边的时候，原来两条链上的顺序不会变，而两条链之间的顺序可以任意安排

设 $dp_{i,j}$ 表示 $i$ 的子树， $i$ 深度为 $j$ 的方案数，转移有

$dp_{u,j}^{'}=\sum_{i=1}^j\sum_{k=i-j}^ndp_{u,i}*dp_{v,k}*C_{j-1}^{i-1}$

后缀和之后复杂度即为 $O(n^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 5050
#define mod 1000000007
int dp[N][N],c[N][N],n,head[N],cnt,sz[N],a,b;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	dp[u][1]=1;sz[u]=1;
	for(int w=head[u];w;w=ed[w].next)if(ed[w].t!=fa){dfs(ed[w].t,u);for(int i=sz[u];i>=1;i--){for(int j=1;j<=sz[ed[w].t];j++)dp[u][i+j]=(dp[u][i+j]+1ll*dp[u][i]*dp[ed[w].t][j]%mod*c[i+j-1][i-1])%mod;dp[u][i]=1ll*dp[u][i]*dp[ed[w].t][0]%mod;}sz[u]+=sz[ed[w].t];}
	for(int i=sz[u];i>=0;i--)dp[u][i]=(dp[u][i]+dp[u][i+1])%mod;
}
int main()
{
	scanf("%d",&n);
	for(int i=0;i<=n;i++)c[i][i]=c[i][0]=1;
	for(int i=2;i<=n;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	printf("%d\n",dp[1][1]);
}
```

##### T2 身份证

###### Problem

有 $n$ 个字符串三元组和三个初始为空的串 $s_1,s_2,s_3$，有以下操作：

1.向一个串末尾加入一个字符

2.向一个串末尾删除字符

求每次操作后，有多少个三元组满足 $s_1$ 是第一个的前缀, $s_2$ 是第二个的前缀, $s_3$ 是第三个的前缀

$n,q,\sum |S|\leq 10^6$

###### Sol

显然可以进行trie+三位数点，但是过不去   ~~我卡了一下午过了~~

因为输入串，所以 trie上 $\sum size$ 不大

对于第一个trie上的点，处理出它子树中所有叶节点三元组后两个上的dfs序

对于询问，在第一个trie上节点处询问后两个的二维数点即可

离线后复杂度 $O((n+q)\log |S|)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 505050
struct modify{int b,c;friend bool operator <(modify a,modify b){return a.b<b.b;}};
struct query{int b,l,r,id;friend bool operator <(query a,query b){return a.b<b.b;}};
char s[N],v1[10],v21[12];
int n,m,is[N],ds[N],ct[N],v2[N][3],a,id[N];
struct trie{
	int ch[N][26],l[N],r[N],ct=1,id[N],ct3,fa[N];
	void ins(int st2)
	{
		int st=1;
		for(int i=1;s[i];i++)
		{
			if(!ch[st][s[i]-'a'])ch[st][s[i]-'a']=++ct,fa[ct]=st;
			st=ch[st][s[i]-'a'];
		}
		id[st2]=st;
	}
	void dfs(int x)
	{
		l[x]=++ct3;
		for(int i=0;i<26;i++)if(ch[x][i])dfs(ch[x][i]);r[x]=ct3;
	}
}t[3];
vector<modify> st[N];
vector<query> s2[N];
int tr[N],as[N];
void add(int x,int k){for(int i=x;i<=500050;i+=i&-i)tr[i]+=k;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)for(int j=0;j<3;j++)scanf("%s",s+1),t[j].ins(i);
	for(int i=0;i<3;i++)t[i].dfs(1);
	for(int i=1;i<=n;i++)for(int j=t[0].id[i];j;j=t[0].fa[j])st[j].push_back((modify){t[1].l[t[1].id[i]],t[2].l[t[2].id[i]]});
	is[0]=is[1]=is[2]=1;
	scanf("%d",&m);
	for(int i=1;i<=m;i++)
	{
		scanf("%s%d",v1+1,&a);a--;
		if(v1[1]=='-')
		if(ds[a]==ct[a])ds[a]--,ct[a]--,is[a]=t[a].fa[is[a]];
		else ct[a]--;
		else
		{
			scanf("%s",v21+1);
			if(ds[a]!=ct[a])ct[a]++;
			else
			{
				if(!t[a].ch[is[a]][v21[1]-'a'])ct[a]++;
				else ct[a]++,ds[a]++,is[a]=t[a].ch[is[a]][v21[1]-'a'];
			}
		}
		int fg=1;
		for(int i=0;i<3;i++)if(ds[i]!=ct[i])fg=0;
		if(fg)
		{
			s2[is[0]].push_back((query){t[1].l[is[1]]-1,t[2].r[is[2]],t[2].l[is[2]]-1,i});
			s2[is[0]].push_back((query){t[1].r[is[1]],t[2].l[is[2]]-1,t[2].r[is[2]],i});
		}
	}
	for(int i=1;i<=t[0].ct;i++)
	{
		sort(st[i].begin(),st[i].end());
		sort(s2[i].begin(),s2[i].end());
		int v1=st[i].size(),v2=s2[i].size(),l1=0;
		for(int j=0;j<v2;j++)
		{
			while(l1<v1&&st[i][l1].b<=s2[i][j].b)add(st[i][l1].c,1),l1++;
			as[s2[i][j].id]+=que(s2[i][j].r)-que(s2[i][j].l);
		}
		for(int j=0;j<l1;j++)add(st[i][j].c,-1);
	}
	for(int i=1;i<=m;i++)printf("%d\n",as[i]);
}
```

卡过去的 $\log^2$

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 505050
struct modify{int a,b,c;friend bool operator <(modify a,modify b){return a.b<b.b;}};
struct query{int a,b,l,r,id;friend bool operator <(query a,query b){return a.b<b.b;}};
char s[N],v1[10],v21[12];
int n,m,is[N],ds[N],ct[N],v2[N][3],a,id[N],f1,f2;
struct trie{
	int ch[N][26],l[N],r[N],ct=1,id[N],ct3,fa[N];
	void ins(int st2)
	{
		int st=1;
		for(int i=1;s[i];i++)
		{
			if(!ch[st][s[i]-'a'])ch[st][s[i]-'a']=++ct,fa[ct]=st;
			st=ch[st][s[i]-'a'];
		}
		id[st2]=st;
	}
	void dfs(int x)
	{
		l[x]=++ct3;
		for(int i=0;i<26;i++)if(ch[x][i])dfs(ch[x][i]);r[x]=ct3;
	}
}t[3];
vector<modify> st[2][22];
vector<query> s2[2][22];
vector<int> fu[N];
int tr[N],as[N];
void add(int x,int k){for(int i=x;i<=f2;i+=i&-i)tr[i]+=k;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
void cdq(int l,int r,int d,int x)
{
	if(!s2[d][x].size())
	{
		for(int w=l;w<=r;w++)
		{
			int s0=fu[w].size();
			for(int i=0;i<s0;i++)
			st[d][x].push_back((modify){t[id[0]].l[t[id[0]].id[fu[w][i]]],t[id[1]].l[t[id[1]].id[fu[w][i]]],t[id[2]].l[t[id[2]].id[fu[w][i]]]});
		}
		sort(st[d][x].begin(),st[d][x].end());
		return;
	}
	if(l==r)
	{
		int s0=fu[l].size();
		for(int i=0;i<s0;i++)
		st[d][x].push_back((modify){t[id[0]].l[t[id[0]].id[fu[l][i]]],t[id[1]].l[t[id[1]].id[fu[l][i]]],t[id[2]].l[t[id[2]].id[fu[l][i]]]});
		sort(st[d][x].begin(),st[d][x].end());
		int s3=s2[d][x].size(),s4=st[d][x].size();
		int l1=0;
		for(int i=0;i<s3;i++)
		{
			while(l1<s4&&s2[d][x][i].b>=st[d][x][l1].b)add(st[d][x][l1].c,1),l1++;
			as[s2[d][x][i].id]+=que(s2[d][x][i].r)-que(s2[d][x][i].l);
		}
		s2[d][x].clear();
		for(int i=1;i<l1;i++)add(st[d][x][i].c,-1);
		return;
	}
	int mid=(l+r)>>1;
	int s1=s2[d][x].size();
//	s2[0][x+1].clear();s2[1][x+1].clear();
	for(int i=0;i<s1;i++)if(s2[d][x][i].a<=mid)s2[0][x+1].push_back(s2[d][x][i]);else s2[1][x+1].push_back(s2[d][x][i]);
	s2[d][x].clear();
	vector<query>().swap(s2[d][x]);
	cdq(l,mid,0,x+1);
	int s3=s2[1][x+1].size(),s4=st[0][x+1].size();
	int l1=0;
	for(int i=0;i<s3;i++)
	{
		while(l1<s4&&s2[1][x+1][i].b>=st[0][x+1][l1].b)
		add(st[0][x+1][l1].c,1),l1++;
		as[s2[1][x+1][i].id]+=que(s2[1][x+1][i].r)-que(s2[1][x+1][i].l);
	}
	for(int i=1;i<l1;i++)add(st[0][x+1][i].c,-1);
	cdq(mid+1,r,1,x+1);
	s3=st[0][x+1].size(),s4=st[1][x+1].size();
	int l3=0,l4=0;
	for(int i=1;i<=s3+s4;i++)
	if(l3==s3)st[d][x].push_back(st[1][x+1][l4++]);
	else if(l4==s4)st[d][x].push_back(st[0][x+1][l3++]);
	else if(st[1][x+1][l4].b<st[0][x+1][l3].b)st[d][x].push_back(st[1][x+1][l4++]);
	else st[d][x].push_back(st[0][x+1][l3++]);
	st[0][x+1].clear();st[1][x+1].clear();
	vector<modify>().swap(st[0][x+1]);
	vector<modify>().swap(st[1][x+1]);
}
void reads()
{
	int ct=0;
	char s1=getchar();
	while(s1<'a'||s1>'z')s1=getchar();
	while(!(s1<'a'||s1>'z'))s[++ct]=s1,s1=getchar();
	s[ct+1]=0;
}
int main()
{
	scanf("%d",&n);id[0]=0;id[1]=1;id[2]=2;
	for(int i=1;i<=n;i++)for(int j=0;j<3;j++)reads(),t[j].ins(i);
	if(t[id[2]].ct<t[id[0]].ct)swap(id[0],id[2]);
	if(t[id[1]].ct<t[id[0]].ct)swap(id[0],id[1]);
	if(t[id[2]].ct>t[id[1]].ct)swap(id[1],id[2]);
	for(int i=0;i<3;i++)t[i].dfs(1);
	for(int i=1;i<=n;i++)fu[t[id[0]].l[t[id[0]].id[i]]].push_back(i);
	is[0]=is[1]=is[2]=1;
	scanf("%d",&m);
	for(int i=1;i<=m;i++)
	{
		v1[1]=getchar();while(v1[1]!='+'&&v1[1]!='-')v1[1]=getchar();
		v1[4]=getchar();while(v1[4]<'1'||v1[4]>'9')v1[4]=getchar();a=v1[4]-'0';
		a--;
		if(v1[1]=='-')
		if(ds[a]==ct[a])ds[a]--,ct[a]--,is[a]=t[a].fa[is[a]];
		else ct[a]--;
		else
		{
			v21[1]=getchar();while(v21[1]<'a'||v21[1]>'z')v21[1]=getchar();
			if(ds[a]!=ct[a])ct[a]++;
			else
			{
				if(!t[a].ch[is[a]][v21[1]-'a'])ct[a]++;
				else ct[a]++,ds[a]++,is[a]=t[a].ch[is[a]][v21[1]-'a'];
			}
		}
		int fg=1;
		for(int i=0;i<3;i++)if(ds[i]!=ct[i])fg=0;
		if(fg)
		{
			s2[0][1].push_back((query){t[id[0]].l[is[id[0]]]-1,t[id[1]].l[is[id[1]]]-1,t[id[2]].l[is[id[2]]]-1,t[id[2]].r[is[id[2]]],i});
			s2[0][1].push_back((query){t[id[0]].l[is[id[0]]]-1,t[id[1]].r[is[id[1]]],t[id[2]].r[is[id[2]]],t[id[2]].l[is[id[2]]]-1,i});
			s2[0][1].push_back((query){t[id[0]].r[is[id[0]]],t[id[1]].l[is[id[1]]]-1,t[id[2]].r[is[id[2]]],t[id[2]].l[is[id[2]]]-1,i});
			s2[0][1].push_back((query){t[id[0]].r[is[id[0]]],t[id[1]].r[is[id[1]]],t[id[2]].l[is[id[2]]]-1,t[id[2]].r[is[id[2]]],i});
		}
	}
	f1=t[id[0]].ct;f2=t[id[2]].ct;
	sort(s2[0][1].begin(),s2[0][1].end());
	cdq(1,f1*1.44,0,1);
	for(int i=1;i<=m;i++)printf("%d\n",as[i]);
}
```

##### T3 Fib与Gcd

###### Problem

多组询问 $gcd⁡(a∗F_n+b∗F_{n+1},c∗F_n+d∗F_{n+1} )\bmod 998244353$

$T≤10^5,n≤10^9,a,b,c,d≤10^3$

###### Sol

首先，可以对 $b,d$ 辗转相除，消掉 $b,d$ 中的一个

这时相当于求 $gcd⁡(a∗F_n,c∗F_n+d∗F_{n+1})$

如果 $a=0$ ,那么答案就是 $c∗F_n+d∗F_{n+1}$

如果 $a>0$ ,可以发现 $gcd⁡(a∗b,n)|gcd⁡(a,n)∗gcd⁡(b,n)$

因此，只需要求出 $gcd⁡(a,c∗F_n+d∗F_{n+1})$ 和 $gcd⁡(F_n,c∗F_n+d∗F_{n+1})$

对于第一个，可以先求出$(c∗F_n+d∗F_{n+1})\bmod a$ , 然后和 $a$ 求 $gcd$ 即可

对于第二个，有 $gcd⁡(F_n,c∗F_n+d∗F_{n+1})=gcd⁡(F_n,d∗F_{n+1})$

显然 $gcd⁡(F_n,F_{n+1})=1$ ,所以它等于 $gcd⁡(F_n,d)$

然后只需要求 $c∗F_n+d∗F{n+1}$ 与两个 $gcd$ 乘积的 $gcd$ 即可

复杂度 $O(T \log ⁡n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
struct mat{int a[2][2];};
mat mul(mat a,mat b,int c){return (mat){(1ll*a.a[0][0]*b.a[0][0]+1ll*a.a[0][1]*b.a[1][0])%c,(1ll*a.a[0][0]*b.a[0][1]+1ll*a.a[0][1]*b.a[1][1])%c,(1ll*a.a[1][0]*b.a[0][0]+1ll*a.a[1][1]*b.a[1][0])%c,(1ll*a.a[1][0]*b.a[0][1]+1ll*a.a[1][1]*b.a[1][1])%c};}
mat pw(mat a,long long b,int p){mat as;as.a[0][0]=as.a[1][1]=1;as.a[0][1]=as.a[1][0]=0;while(b){if(b&1)as=mul(as,a,p);a=mul(a,a,p);b>>=1;}return as;}
int fib(long long a,int p){mat sb=pw((mat){1,1,1,0},a,p);return sb.a[1][0];}
int gcd(int a,int b){if(a<0)a=-a;if(b<0)b=-b;return b?gcd(b,a%b):a;}
int solve(int a,int b,int c,int d,long long n)
{
	if(a==0&&b==0)return (1ll*c*fib(n,998244353)+1ll*d*fib(n+1,998244353))%998244353;
	if(c==0&&d==0)return (1ll*a*fib(n,998244353)+1ll*b*fib(n+1,998244353))%998244353;
	if(a==0&&c==0)return 1ll*gcd(b,d)*fib(n+1,998244353)%998244353;
	if(b==0&&d==0)return 1ll*gcd(a,c)*fib(n,998244353)%998244353;
	if(d==0)a^=c^=a^=c,b^=d^=b^=d;
	if(b==0)
	{
		if(a<0)a=-a;
		int v1=(1ll*c*fib(n,a)+1ll*d*fib(n+1,a))%a+a,v2=fib(n,d)+d;
		v1=gcd(a,v1);v2=gcd(d,v2);
		v1*=v2;
		return gcd(((1ll*c*fib(n,v1)+1ll*d*fib(n+1,v1))%v1+v1)%v1+v1,v1);
	}
	if(b<d)b^=d^=b^=d,a^=c^=a^=c;
	return solve(a-(b/d)*c,b%d,c,d,n);
}
int main()
{
	int T,a,b,c,d;long long n;
	scanf("%d",&T);
	while(T--)scanf("%lld%d%d%d%d",&n,&a,&b,&c,&d),printf("%d\n",(solve(a,b,c,d,n)+998244353)%998244353);
}
```

#### 三月集训 Day 1

##### T1 Bitset Master

###### Problem

给一棵树，有 $n$ 个集合，一开始 $S_u = \{u\}$

有两种操作

1.给一条树边 $u,v$ ，令 $S_u,S_v$ 变成 $S_u∪S_v$

2.给一个 $i$ ，询问有多少个集合包含 $i$

$n\leq 2\times 10^5,m\leq 6\times 10^5,6s$

###### Sol

考虑 $i\in S_j$ 的条件，可以发现是 $i$ 到 $j$ 的路径上存在一个时间递增的操作序列

考虑点分治，对于当前这一层，首先要求出每个点到中心的最短时间

考虑dp，设 $dp_{i,j}$ 表示 $i$ 从时刻 $j$ 开始多早能到中心

注意到只有 $i$ 和父亲连边上所有修改的时刻的dp值是有用的，所有只有 $O(m)$ 个有用dp值

更新时考虑dfs，对于每个点，考虑它到父亲的边上的每一次修改，对于这个修改在父亲的dp上lower_bound查，复杂度 $O(m\log m)$

然后这时候相当于多个从 $t_1$ 时刻开始，有多少个点能够在 $t_2$ 时刻到达的询问

考虑按 $t_1$ 从大到小处理询问，每次减小时，从中心开始尝试往下更新，对于每个点，记录它所有向儿子连的边的修改时间排序，访问到这个点后二分这个区间，找到所有需要更新的往下更新

可以发现一共会更新 $O(m)$ 次，加上二分和树状数组维护答案是 $O(m\log m)$

求出后再对于每个子树，再做一遍减去子树内部额外计算的贡献

加上点分治复杂度 $O(m\log m\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 600500
vector<int> id[N],id2[N];
int n,m,head[N],cnt,a,b,dep[N],sz[N],vl,as,as2,dp[N],f1[N],v[N],vis[N],dp2[N],dp3[N],fr[N],as1[N],tid[N];
struct que{int x,t,t2,id;};
vector<que> fu[N],tp1,fu2[N],tp2[N];
struct edge{int t,next,id;}ed[N];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],id};head[t]=cnt;}
void dfs1(int u,int fa)
{
	sz[u]=1;int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],mx=mx<sz[ed[i].t]?sz[ed[i].t]:mx;
	mx=mx<vl-sz[u]?vl-sz[u]:mx;
	if(mx<as2)as2=mx,as=u;
}
void pre(int u){vis[u]=1;for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])dfs1(ed[i].t,u),vl=sz[ed[i].t],as2=1e9,dfs1(ed[i].t,u),f1[as]=u,pre(as);}
void dfs2(int u,int fa,int f2)
{
	dp3[u]=1e9;fr[u]=f2;
	for(int i=0;i<id[v[u]].size();i++)
	{
		vector<int>::iterator it=lower_bound(id[v[fa]].begin(),id[v[fa]].end(),id[v[u]][i]);
		if(it==id[v[fa]].end())dp[id[v[u]][i]]=1e9;
		else dp[id[v[u]][i]]=max(id[v[u]][i],dp[*it]);
		if(!fa)dp[id[v[u]][i]]=0;
	}
	if(id[v[u]].size())dp2[u]=dp[id[v[u]][0]];else dp2[u]=1e9;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&ed[i].t!=fa)v[ed[i].t]=ed[i].id,dfs2(ed[i].t,u,f2?f2:ed[i].t);
	id2[u].clear();
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])
	for(int j=0;j<id[ed[i].id].size();j++)id2[u].push_back(id[ed[i].id][j]);
	sort(id2[u].begin(),id2[u].end());
}
bool cmp(que a,que b){return a.t>b.t;}
struct tr1{
	long long tr[N];
	void add(int x,int k){for(int i=x;i<=m;i+=i&-i)tr[i]+=k;}
	long long que(int x){long long as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}tr;
void dfs3(int u,int fa,int t)
{
	if(t>=dp3[u])return;
	int las=dp3[u];
	tr.add(dp3[u]+1,-1);dp3[u]=t;tr.add(dp3[u]+1,1);
	vector<int>::iterator it=lower_bound(id2[u].begin(),id2[u].end(),t);
	if(it==id2[u].end())return;
	while(it!=id2[u].end()&&(*it)<las)
	{
		int v2=tid[*it],s1=ed[v2*2].t;
		if(s1==u)s1=ed[v2*2-1].t;
		dfs3(s1,u,*it);
		it++;
	}
}
void dfs4(int u,int fa)
{
	tr.add(dp3[u]+1,-1);dp3[u]=1e9;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&ed[i].t!=fa)dfs4(ed[i].t,u);
}
void work(int u)
{
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])fu2[ed[i].t].clear(),tp2[ed[i].t].clear();
	id[0].clear();id[0].push_back(m+1);v[u]=0;dfs2(u,0,0);
	for(int i=0;i<fu[u].size();i++)fu[u][i].t=dp2[fu[u][i].x];
	sort(fu[u].begin(),fu[u].end(),cmp);
	tp1.clear();
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])for(int j=0;j<id[ed[i].id].size();j++)tp1.push_back((que){ed[i].t,id[ed[i].id][j],0,0}),tp2[ed[i].t].push_back((que){ed[i].t,id[ed[i].id][j],0,0});
	sort(tp1.begin(),tp1.end(),cmp);
	dp3[u]=0;tr.add(1,1);
	int l1=0;
	for(int i=0;i<fu[u].size();i++)
	{
		while(l1<tp1.size()&&tp1[l1].t>=fu[u][i].t)dfs3(tp1[l1].x,u,tp1[l1].t),l1++;
		if(fu[u][i].t>fu[u][i].t2)continue;
		as1[fu[u][i].id]+=tr.que(fu[u][i].t2);
		if(fu[u][i].x!=u)fu2[fr[fu[u][i].x]].push_back(fu[u][i]);
	}
	dfs4(u,0);dp3[u]=0;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])
	{
		sort(tp2[ed[i].t].begin(),tp2[ed[i].t].end(),cmp);
		sort(fu2[ed[i].t].begin(),fu2[ed[i].t].end(),cmp);
		int l1=0;
		for(int j=0;j<fu2[ed[i].t].size();j++)
		{
			while(l1<tp2[ed[i].t].size()&&tp2[ed[i].t][l1].t>=fu2[ed[i].t][j].t)dfs3(tp2[ed[i].t][l1].x,u,tp2[ed[i].t][l1].t),l1++;
			as1[fu2[ed[i].t][j].id]-=tr.que(fu2[ed[i].t][j].t2);
		}
		dfs4(ed[i].t,0);
	}
}
void doit(int u){vis[u]=1;work(u);for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])dfs1(ed[i].t,u),vl=sz[ed[i].t],as2=1e9,dfs1(ed[i].t,u),f1[as]=u,doit(as);}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b,i);
	pre(1);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d",&a,&b);
		if(a==1)
		{
			int st=b;
			while(st)fu[st].push_back((que){b,0,i,i}),st=f1[st];
		}
		else id[b].push_back(i),tid[i]=b;
	}
	for(int i=1;i<=n;i++)vis[i]=0;doit(1);
	for(int i=1;i<=m;i++)if(as1[i])printf("%d\n",as1[i]);
}
```

##### T2 Knowledge-Oriented Problem

不会

##### T3 LCM Sum

###### Problem

求 $\sum_{i=1}^n lcm(i,i+1,...,i+k)$ ,对 $1e9+7$ 取模

$n\leq 10^{18},k\leq 30,6s,1GB$

###### Sol

设 $c_i=i*(i+1)*...*(i+k)/lcm(i,...,i+k)$

容易发现， $c_i$ 里面不会有大于 $k$ 的质因子

对于一个质数 $p$ ,设 $s$ 满足 $p^s\leq k,p^{s+1}>k$ ，那么容易发现 $c_i$ 中含有质因子 $p$ 的数量只和 $c_i \bmod p^s$ 有关

设所有 $p^s$ 的乘积是 $L$ ，在 $k\leq 16$ 时 $L\leq 720720$,在 $k\leq 30$ 时则在 $2\times 10^{12}$ 左右

设 $f(i)=\sum_{j=1,j\bmod L=i}^n i*(i+1)*...*(i+k)$ 

可以发现，对于 $[1,n\bmod L]$ 和 $[(n\bmod L)+1,L]$ 两部分，它各自是一个 $k+1$ 次多项式，可以使用插值 $O(k^3)$ 求出

对于 $L$ 小的部分可以暴力枚举，对于 $L$ 大的部分，考虑折半搜索，将 $L$ 分成两部分，每一部分枚举余数

考虑合并，设分成了 $L_1,L_2$ ,考虑crt，如果 $i≡x_1(\bmod L_1),i≡x_2(\bmod L_2)$ ,可以得到 $i≡v_1x_1+v_2x_2(\bmod L)$ ，其中 $x_1,x_2$ 可以暴力求

然后将所有 $[1,L_2]$ 数按照 $(v_2x_2-1)\bmod L+1$ 排序，注意到关于 $a+b$ 的多项式可以通过维护 $a^i,b^i$ 来求出，因此可以求出对于右边 $b^i$ 乘上对应系数的前缀和

然后枚举 $[1,L_1]$ ，讨论 $a+b$ 于 $n\bmod L,L,L+n\bmod L$ 的关系，二分出三段，每一段各自 $O(k^2)$ 合并即可

注意到 $v_1|x_2$ ，可以发现所有$(v_2x_2-1)\bmod L+1$ 排序后为 $v_1,2v_1,3v_1,...,v2v1=L$,可以通过类似求逆的方式省去排序

然后注意到复杂度是 $O(L_1*k^2+L_2*k)$ ，因此可以适当调整做到接近 $O(L*k^{1.5})$

然后用力卡常可以卡到5s上下

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define mod 1000000007
#define N 6420001
#define K 35
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int fr[K],ifr[K],s1[K],s2[K],su1[K],su2[K],v2[K],vl[K],vl2[K],dp[K],su[N][K],st[N],fuc[K*3],dp1[N],dp2[N],k;
long long n,f2,t1,t2,l1,l2,st2[N];
int gcd(int a,int b){return b?gcd(b,a%b):a;}
long long gcd2(long long a,long long b){return b?gcd2(b,a%b):a;}
int solve(int v,long long d)
{
	if(d<0)return 0;
	int as=0;
	long long vl2=d;
	for(int i=0;i<=k+3;i++)
	{
		long long tp1=(1ll*i*(f2%mod)+v)%mod,tp2=1;
		for(int j=0;j<=k;j++)tp2=1ll*tp2*(tp1+j)%mod;
		s1[i]=i?(s1[i-1]+tp2)%mod:tp2;
	}
	vl2%=mod;
	if(vl2<=k+3)return s1[vl2];
	su1[0]=su2[k+4]=1;
	for(int i=1;i<=k+3;i++)su1[i]=1ll*su1[i-1]*(mod+vl2-i)%mod;
	for(int i=k+3;i>=1;i--)su2[i]=1ll*su2[i+1]*(mod+vl2-i)%mod;
	for(int i=1;i<=k+3;i++)as=(as+1ll*s1[i]*su1[i-1]%mod*su2[i+1]%mod*ifr[i-1]%mod*ifr[k+3-i]%mod*(((k+3-i)&1)?-1:1))%mod;
	as=(as+mod)%mod;
	return as;
}
void solve2(long long d)
{
	for(int i=1;i<=k+3;i++)
	{
		s2[i]=solve(i,d);
		for(int j=0;j<=k+3;j++)dp[j]=0;
		dp[0]=1;
		int st5=1;
		for(int j=1;j<=k+3;j++)if(i-j)
		{
			for(int l=k+2;l>=0;l--)dp[l+1]=(dp[l+1]+dp[l])%mod,dp[l]=1ll*dp[l]*(mod-j)%mod;
			st5=1ll*st5*pw((mod+i-j),mod-2)%mod;
		}
		for(int j=0;j<=k+3;j++)vl2[j]=(vl2[j]+1ll*dp[j]*s2[i]%mod*st5)%mod;
	}
}
void doit(int p)
{
	int p2=pw(p,mod-2);
	int st1=0,v2=p,g[K];
	while(v2<=k+1)st1++,v2*=p;v2/=p;
	for(int i=1;i<=v2+k;i++)
	{
		fuc[i]=0;int st2=i;
		while(st2%p==0)st2/=p,fuc[i]++;
		if(fuc[i]>st1)fuc[i]=st1;
		fuc[i]+=fuc[i-1];
	}
	for(int i=1;i<=v2;i++)g[i]=pw(p2,fuc[i+k]-fuc[i-1]-st1);
	for(int i=1,j=1;j<=t2;j++,i=i==v2?1:i+1)dp1[j]=1ll*dp1[j]*g[i]%mod;
}
void doit2(int p)
{
	int p2=pw(p,mod-2);
	int st1=0,v2=p,g[K];
	while(v2<=k+1)st1++,v2*=p;v2/=p;
	for(int i=1;i<=v2+k;i++)
	{
		fuc[i]=0;int st2=i;
		while(st2%p==0)st2/=p,fuc[i]++;
		if(fuc[i]>st1)fuc[i]=st1;
		fuc[i]+=fuc[i-1];
	}
	for(int i=1;i<=v2;i++)g[i]=pw(p2,fuc[i+k]-fuc[i-1]-st1);
	for(int i=1,j=1;j<=t1;j++,i=i==v2?1:i+1)dp2[j]=1ll*dp2[j]*g[i]%mod;
}
int justdoit()
{
	int as=0;
	for(int i=1;i<=t2;i++)dp1[i]=dp2[i]=1;
	int inv=0;
	while(1ll*inv*(l2/t1)%t2!=1)inv++;
	for(int i=1;i<=t2;i++){st2[i]=1ll*i*t1;st[i]=(1ll*i*inv-1)%t2+1;}
	for(int i=2;i<=k;i++)
	{
		int fg1=0;
		for(int j=2;j<i;j++)if(i%j==0)fg1=1;
		if(!fg1)if(t1%i)doit(i);else doit2(i);
	}
	for(int i=1;i<=t2;i++)
	{
		int st3=dp1[st[i]],f2=st2[i]%mod;
		for(int j=0;j<=k+1;j++)su[i][j]=(su[i-1][j]+st3)%mod,st3=1ll*st3*f2%mod;
	}
	for(int i=0;i<=k+1;i++)vl[i]=1ll*vl[i]*fr[i]%mod,vl2[i]=1ll*vl2[i]*fr[i]%mod;
	int v3[K],v4[K],v5[K],v6[K];
	for(int i=0;i<=k+1;i++)v6[i]=su[t2][i];
	for(int i=1;i<=t1;i++)
	{
		long long v1=(1ll*i*l1-1)%f2+1,st=n%f2;
		int as1=0,tp1=v1%mod,tp2=(v1-f2)%mod+mod,st3=dp2[i],as2=0;
		v4[0]=v5[0]=st3;
		for(int j=1;j<=k+1;j++)v4[j]=1ll*v4[j-1]*tp1%mod,v5[j]=1ll*v5[j-1]*tp2%mod;
		for(int j=2;j<=k+1;j++)v4[j]=1ll*v4[j]*ifr[j]%mod,v5[j]=1ll*v5[j]*ifr[j]%mod;
		if(v1>st)
		{
			as1=(f2-v1)/t1;
			for(int j=0;j<=k+1;j++)v3[j]=1ll*su[as1][j]*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v4[l]*v3[j-l])%mod;  
				as=(as+1ll*as2*vl[j])%mod;as2=0;
			}
			int las=as1;
			as1=(f2+st-v1)/t1;
			for(int j=0;j<=k+1;j++)v3[j]=1ll*(su[as1][j]-su[las][j]+mod)*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v5[l]*v3[j-l])%mod;
				as=(as+1ll*as2*vl2[j])%mod;as2=0;
			}
			for(int j=0;j<=k+1;j++)v3[j]=1ll*(v6[j]-su[as1][j]+mod)*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v5[l]*v3[j-l])%mod;
				as=(as+1ll*as2*vl[j])%mod;as2=0;
			}
		}
		else
		{
			as1=(st-v1)/t1;
			for(int j=0;j<=k+1;j++)v3[j]=1ll*su[as1][j]*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v4[l]*v3[j-l])%mod;
				as=(as+1ll*as2*vl2[j])%mod;as2=0;
			}
			int las=as1;
			as1=(f2-v1)/t1;
			for(int j=0;j<=k+1;j++)v3[j]=1ll*(su[as1][j]-su[las][j]+mod)*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v4[l]*v3[j-l])%mod;
				as=(as+1ll*as2*vl[j])%mod;as2=0;
			}
			for(int j=0;j<=k+1;j++)v3[j]=1ll*(v6[j]-su[as1][j]+mod)*ifr[j]%mod;
			for(int j=0;j<=k+1;j++)
			{
				for(int l=0;l<=j;l++)as2=(as2+1ll*v5[l]*v3[j-l])%mod;
				as=(as+1ll*as2*vl2[j])%mod;as2=0;
			}
		}
	}
	return as;
}
int solve2(int v)
{
	for(int i=0;i<=k;i++)fuc[i]=gcd2(v+i,2329089562800ll);
	int fuc1=1,as=1;
	long long st1=1;
	for(int i=0;i<=k;i++){long long tp=gcd2(st1,fuc[i]);fuc1=1ll*fuc1*(tp%mod)%mod;st1=st1/tp*fuc[i];as=1ll*as*(v+i)%mod;}
	return 1ll*as*pw(fuc1,mod-2)%mod;
}
int main()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=34;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	scanf("%lld%d",&n,&k);
	if(n<=10000)
	{
		int as1=0;
		for(int i=1;i<=n;i++)as1=(as1+solve2(i))%mod;
		printf("%d\n",as1);
		return 0;
	}
	f2=1;
	for(int i=2;i<=k;i++)
	{
		int fg1=1;
		for(int j=2;j<i;j++)if(i%j==0)fg1=0;
		if(fg1)
		{
			int st=1,st2=k;while(st2>=i)st2/=i,st*=i;
			f2*=st;
		}
	}
	if(f2==2)f2=6;
	t1=pow(1.0*f2,0.45);
	while(f2%t1||gcd(t1,f2/t1)!=1)t1++;
	t2=f2/t1;
	if(t1>t2)t1^=t2^=t1^=t2;
	l1=t2;while(l1%t1!=1)l1+=t2;
	l2=t1;while(l2%t2!=1)l2+=t1;
	solve2(n/f2-1);
	for(int i=0;i<=k+1;i++)vl[i]=vl2[i],vl2[i]=0;
	solve2(n/f2);
	printf("%d\n",justdoit());
}
```



#### 三月集训 Day 2

##### T1 调兵遣将

###### Problem

你有一个长度为 $n$ 的序列 $a$ ,你需要选出一些不交区间，使得每个区间的 $a_i\ $的gcd相同，求对于每个元素，满足它被一个区间覆盖的方案数模 $998244353$

$n\leq 50000,a_i\leq 10^9$ 

###### Sol

考虑固定 $r$ 求gcd

注意到gcd每改变一次至少减半，因此只会有 $n\log a_i$ 种gcd,可以通过倍增 $O(n\log n\log a_i)$ 求出每种gcd对应的区间

考虑枚举gcd算答案，对于一个gcd，考虑算每个元素不包含它的方案，容斥即可得到答案

设 $f_i$ 表示 $[1,i]$ 的方案数，注意到每一种gcd都是固定 $r$ ，$l$ 是一个区间

那么对于每一个方案，可以在线段树上区间查，然后更新dp

设所有 $r$ 为关键点，那么可以发现只有关键点的 $f_i$ 和 $f_{i-1}$ 不同，可以快速求出所有的 $f$

设 $g_i$ 表示 $[i,n]$ 的方案数，如果将开头的过程反过来做，那么相当于有一些固定 $l$ ，$r$ 是一个区间的方案，可以使用同样的方式更新

然后，$ans_i=f_n-f_{i-1}*g_{i+1}$ ,注意到只有两个关键点集合的并的点的值会发生改变，所以可以搞出所有区间然后区间加

复杂度 $O(n\log n\log a_i)$

###### Code

没有

我场上写了个 $O(n^2\log a_i)$ (n*所有gcd种数) ，它过了

```cpp
#include<cstdio>
#include<map>
#include<vector>
#include<vector>
using namespace std;
#define N 50050
#define mod 998244353
int n,v[N],f[N][18],ct,as1[N],v2[N],v3[N];
map<int,int> fuc;
struct sth{int v,l,r;};
vector<sth> tp[N*40],tp2[N*40];
struct segt{
	struct node{int l,r,su,lz,l2;}e[N*4];
	void pushdown(int x)
	{
		if(e[x].l2)e[x<<1].su=e[x<<1|1].su=e[x<<1|1].lz=e[x<<1].lz=0,e[x<<1].l2=e[x<<1|1].l2=1,e[x].l2=0;
		if(e[x].lz)e[x<<1].su=(e[x<<1].su+1ll*e[x].lz*(e[x<<1].r-e[x<<1].l+1))%mod,e[x<<1|1].su=(e[x<<1|1].su+1ll*e[x].lz*(e[x<<1|1].r-e[x<<1|1].l+1))%mod,
		e[x<<1].lz=(e[x<<1].lz+e[x].lz)%mod,e[x<<1|1].lz=(e[x<<1|1].lz+e[x].lz)%mod,e[x].lz=0;
	}
	void pushup(int x){e[x].su=(e[x<<1].su+e[x<<1|1].su)%mod;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	int que(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].su;pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return que(x<<1,l,r);else if(mid<l)return que(x<<1|1,l,r);else return (que(x<<1,l,mid)+que(x<<1|1,mid+1,r))%mod;}
	void modify(int x,int l,int r,int v){if(e[x].l==l&&e[x].r==r){e[x].su=(e[x].su+1ll*v*(e[x].r-e[x].l+1))%mod;e[x].lz=(e[x].lz+v)%mod;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify(x<<1,l,r,v);else if(mid<l)modify(x<<1|1,l,r,v);else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);pushup(x);}
	void init(){e[1].l2=1;e[1].lz=0;e[1].su=0;modify(1,0,0,1);}
	void init2(){e[1].l2=1;e[1].lz=0;e[1].su=0;modify(1,n+1,n+1,1);}
	void dfs(int x){if(e[x].l==e[x].r){v2[e[x].l]=e[x].su;return;}pushdown(x);dfs(x<<1);dfs(x<<1|1);}
}tr1,tr2,tr3;
int gcd(int a,int b){return b?gcd(b,a%b):a;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),f[i][0]=v[i];
	for(int j=1;j<=17;j++)
	for(int i=1;i+(1<<j)-1<=n;i++)
	f[i][j]=gcd(f[i][j-1],f[i+(1<<j-1)][j-1]);
	tr1.build(1,0,n+1);
	tr2.build(1,0,n+1);
	tr3.build(1,0,n+1);
	for(int i=1;i<=n;i++)
	{
		int v1=v[i],s1=i,s2=i;
		while(s1<=n)
		{
			for(int j=17;j>=0;j--)
			if(f[s2][j]&&gcd(f[s2][j],v1)==v1)s2=s2+(1<<j);
			if(!fuc[v1])fuc[v1]=++ct;
			int st=fuc[v1];
			tp[st].push_back((sth){i,s1,s2-1});
			s1=s2;
			v1=gcd(v1,v[s2]);
		}
	}
	for(int i=n;i>=1;i--)
	{
		int v1=v[i],s1=i,s2=i;
		while(s1)
		{
			for(int j=17;j>=0;j--)
			if(s2-(1<<j)+1>=1&&gcd(f[s2-(1<<j)+1][j],v1)==v1)s2=s2-(1<<j);
			if(!fuc[v1])fuc[v1]=++ct;;
			int st=fuc[v1];
			tp2[st].push_back((sth){i,s2+1,s1});
			s1=s2;
			v1=gcd(v1,v[s2]);
		}
	}
	int su=0;
	for(int i=1;i<=ct;i++)
	{
		int s1=tp[i].size();
		tr1.init();
		for(int j=0;j<s1;j++)
		{
			int v1=tr1.que(1,0,tp[i][j].v-1);
			tr1.modify(1,tp[i][j].l,tp[i][j].r,v1);
		}
		su=tr1.que(1,0,n+1);
		int s2=tp2[i].size();
		tr3.init2();
		for(int j=0;j<s2;j++)
		{
			int v1=tr3.que(1,tp2[i][j].v+1,n+1);
			tr3.modify(1,tp2[i][j].l,tp2[i][j].r,v1);
		}
		tr1.dfs(1);
		for(int j=0;j<=n+1;j++)v3[j]=v2[j];
		tr3.dfs(1);
		for(int j=1;j<=n+1;j++)v3[j]=(v3[j]+v3[j-1])%mod;
		for(int j=n;j>=0;j--)v2[j]=(v2[j]+v2[j+1])%mod;
		for(int j=1;j<=n;j++)as1[j]=(as1[j]+su-1ll*v3[j-1]*v2[j+1]%mod+mod)%mod;
	}
	for(int i=1;i<=n;i++)printf("%d ",(as1[i]%mod+mod)%mod);
}
```

##### T2 一掷千金

###### Problem

![](C:\Users\zz\Documents\pic\59.png)

$n,k\leq 10^5,m\leq 10^9$

###### Sol

考虑这样一个游戏：将操作改为在点上拿掉一个棋子，在链上每个点放一个棋子

显然每个点是独立的，并且如果有两个点，根据sg异或的性质，这等同于没有点，因此可以看成翻转

因此只需要算这个情况的sg值

打个表发现 $sg((x,y))=lowbit(max(x,y))$

然后写个矩形并，线段树维护y轴上的情况，对于前 $n$ 列中的第 $i$ 列，前 $j$ 行贡献是 $lowbit(i)$ ，后面是 $lowbit(j)$ ，可以两部分在线段树上查

对于后面的，考虑扫描线，对于每一段 $y$ 求助里面线段树的值后，只需要求 $lowbit(l),lowbit(l+1),...,lowbit(r)$ 的异或和，这东西可以 $O(\log m)$ 或者 $O(1)$ 解决

复杂度 $O((n+k)\log m)$ 

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<map>
#include<algorithm>
using namespace std;
#define N 100500
struct sth{int a,b,c;};
struct segt{int l,r,lz,su;sth tp;}e[N*4];
sth doit(sth a,sth b){if(a.a<b.a)return a;else if(a.a>b.a)return b;else return (sth){a.a,a.b^b.b,a.c+b.c};}
void pushup(int x){e[x].tp=doit(e[x<<1].tp,e[x<<1|1].tp);}
void pushdown(int x){e[x<<1].tp.a+=e[x].lz;e[x<<1|1].tp.a+=e[x].lz;e[x<<1].lz+=e[x].lz;e[x<<1|1].lz+=e[x].lz;e[x].lz=0;}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;
	if(l==r){e[x].tp.b=(l&-l);e[x].tp.c=1;return;}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	pushup(x);
}
void modify(int x,int l,int r,int v)
{
	if(e[x].l==l&&e[x].r==r){e[x].tp.a+=v;e[x].lz+=v;return;}
	pushdown(x);
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)modify(x<<1,l,r,v);
	else if(mid<l)modify(x<<1|1,l,r,v);
	else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);
	pushup(x);
}
sth query(int x,int l,int r)
{
	if(l>r)return (sth){1000,0,0};
	if(e[x].l==l&&e[x].r==r)return e[x].tp;
	pushdown(x);
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)return query(x<<1,l,r);
	else if(mid<l)return query(x<<1|1,l,r);
	else return doit(query(x<<1,l,mid),query(x<<1|1,mid+1,r));
}
int solve(int l,int r)
{
	if(l>r)return 0;
	int v1=0,v2=1;
	while(r)
	{
		if(((r+1)/2-l/2)&1)v1^=v2;
		v2<<=1;l=(l+1)>>1;r>>=1;
	}
	return v1;
}
map<int,int> fuc;
vector<sth> tp[N*6];
int k,n,m,a,b,c,d,v[N*4],ct;
int main()
{
	scanf("%d%d%d",&k,&n,&m);
	build(1,1,n);
	for(int i=1;i<=k;i++)
	{
		scanf("%d%d%d%d",&a,&b,&c,&d);
		if(!fuc[b])fuc[b]=++ct,v[ct]=b;
		if(!fuc[d+1])fuc[d+1]=++ct,v[ct]=d+1;
		b=fuc[b],d=fuc[d+1];
		tp[b].push_back((sth){a,c,1});
		tp[d].push_back((sth){a,c,-1});
	}
	for(int i=1;i<=n+1;i++)if(!fuc[i])fuc[i]=++ct,v[ct]=i;
	sort(v+1,v+ct+1);
	int as=0;
	for(int i=1;i<ct;i++)
	if(v[i]<=n)
	{
		int f1=fuc[v[i]];
		for(int j=0;j<tp[f1].size();j++)modify(1,tp[f1][j].a,tp[f1][j].b,tp[f1][j].c);
		int ti=v[i];
		sth v1=query(1,1,ti),v2=query(1,ti+1,n);
		int st1=ti-v1.c*(v1.a==0);
		if(st1&1)
		as^=(v[i]&-v[i]);
		int st2=solve(ti+1,n)^(v2.a?0:v2.b);
		as^=st2;
	}
	else
	{
		int f1=fuc[v[i]];
		for(int j=0;j<tp[f1].size();j++)modify(1,tp[f1][j].a,tp[f1][j].b,tp[f1][j].c);
		sth v1=query(1,1,n);
		int st1=n-v1.c*(v1.a==0);
		if(st1&1)as^=solve(v[i],v[i+1]-1);
	}
	printf("%d\n",as);
}
```

##### T3 树拓扑序

###### Problem

给一棵有根树，1为根，儿子向父亲连边，求所有拓扑序的逆序对总数模 $1e9+7$

$n\leq 500$

###### Sol

考虑算每一对数贡献的概率，再乘上总方案数

设 $dp_{i,j,k}$ 表示 $i$ 的子树中的拓扑序， $j$ 排在第 $k$ 个的方案数， $f_u$ 表示 $u$ 子树的拓扑序数

考虑合并子树转移，枚举另外一个加了多少个，有 $dp_{u,j,k}^{'}=\sum_{i=1}^{size_u}dp_{u,j,i}*f_v*C_{k-1}^{i-1}*C_{size_u-i+size_v-(k-i)}^{size_v-(k-i)}$

注意到如果枚举 $i,k-i$ 转移，可以发现复杂度是 $O(size_u*size_v)$ 的，这部分的复杂度不超过 $O(n^3)$

对于每个点，先合并所有子树，再加上根放在最后

然后考虑计算答案

考虑枚举两个子树 $u$ 对 $v$ 的贡献，那么相当于 $\sum_{i\in son_u}\sum_{j\in son_v}[i>j]\sum_{k=1}^{size_u}\sum_{l=1}^{size_v}dp_{u,i,k}*dp_{v,j,l}*g(size_u,size_v,k,l)$

这里 $g(a,b,c,d)$ 表示左边有 $a$ 个，右边有 $b$ 个，进行任意归并，左边第 $c$ 个在右边第 $d$ 个前面的方案数

然后要除以 $f_u*f_v*C_{size_u+size_v}^{size_u}$

直接暴力算是 $O(n^5)$

考虑先算出每个 $g(size_u,size_v,k,l)$ 的系数

如果暴力做前面的四重循环复杂度是 $O(n^4)$

考虑对 $son_v$ 中 $j$ 这一维做前缀和，这样就是 $O(n^3)$

然后考虑算 $g$

有 $g(a,b,c,d)=\sum_{i=0}^{d-1}C_{c-1+i}^iC_{a-c+d-i}^{d-i}$

按 $d$ 这一维递增做，就可以 $O(1)$ 求出每一个

复杂度 $O(n^3)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 505
#define mod 1000000007
int n,fr[N],ifr[N],a,b,head[N],cnt,dp[N][N],su[N][N],as,g[N][N],dp1[N],sz[N],c[N][N];
vector<int> sn[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int C(int i,int j){return c[i][j];}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void merge(int a,int b)
{
	int s1=sn[a].size(),s2=sn[b].size();
	for(int i=0;i<s2;i++)
	{
		int las=i?sn[b][i-1]:0,nw=sn[b][i];
		for(int j=1;j<=s2;j++)su[nw][j]=(dp[nw][j]+su[las][j])%mod;
	}
	for(int i=1;i<=s1;i++)
	for(int j=1;j<=s2;j++)
	g[i][j]=0;
	for(int i=0;i<s1;i++)
	{
		vector<int>::iterator it=lower_bound(sn[b].begin(),sn[b].end(),sn[a][i]);
		if(it==sn[b].begin())continue;
		int vl=*(--it);
		for(int j=1;j<=s1;j++)
		for(int k=1;k<=s2;k++)
		g[j][k]=(g[j][k]+1ll*dp[sn[a][i]][j]*su[vl][k])%mod;
	}
	int sb1=1ll*pw(dp1[a],mod-2)*pw(dp1[b],mod-2)%mod*pw(C(s1+s2,s1),mod-2)%mod;
	for(int j=1;j<=s1;j++)
	{
		int vl2=1ll*C(s1+s2-j,s2)*sb1%mod;
		for(int k=1;k<=s2;k++)
		{
			as=(as+1ll*g[j][k]*vl2)%mod;
			vl2=(vl2+1ll*C(j+k-1,k)*C(s2-k+s1-j,s2-k)%mod*sb1)%mod;
		}
	}
}
void dfs(int u,int fa)
{
	dp1[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),dp1[u]=1ll*dp1[u]*dp1[ed[i].t]%mod*C(sz[u]+sz[ed[i].t],sz[ed[i].t])%mod,sz[u]+=sz[ed[i].t];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	for(int j=head[u];j;j=ed[j].next)if(ed[j].t!=fa&&j!=i)merge(ed[j].t,ed[i].t);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	for(int j=0;j<sn[ed[i].t].size();j++)
	{
		int t=sn[ed[i].t][j];
		if(t>u)as=(as+1)%mod;
		int v1=sz[ed[i].t];
		for(int k=head[u];k;k=ed[k].next)if(ed[k].t!=fa&&ed[k].t!=ed[i].t)
		{
			int s1=sz[ed[k].t];
			for(int l1=v1;l1>0;l1--)
			for(int l2=s1;l2>=0;l2--)
			dp[t][l1+l2]=(dp[t][l1+l2]+1ll*dp[t][l1]*(1ll*C(l1-1+l2,l2)*C(v1-l1+s1-l2,s1-l2)%mod*dp1[ed[k].t]%mod-(l2==0)))%mod;
			v1+=s1;
		}
		sn[u].push_back(t);
	}
	sz[u]++;
	sn[u].push_back(u);dp[u][sz[u]]=dp1[u];
	sort(sn[u].begin(),sn[u].end());
}
int main()
{
	scanf("%d",&n);
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=0;i<=n;i++)for(int j=0;j<=i;j++)c[i][j]=1ll*fr[i]*ifr[j]%mod*ifr[i-j]%mod;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	printf("%d\n",1ll*dp1[1]*as%mod);
}
```

#### 三月集训 Day 4

##### T1 Manager

###### Problem

给一棵有根树，每个点有权值 $a_i$ ，定义树的权值为每个点子树中所有 $a_i$ 的中位数，偶数个取两个中较小的一个

求对于每个点，如果令 $a_i=10^5$ ,整个树的权值

$n\leq 2\times 10^5,a_i\leq 10^5$

###### Sol

对于每个点，将它子树内的 $a$ 排序，设中位数为 $a_{mid}$

如果这个点是叶子，改它的贡献一定是 $10^5-a_i$

否则，如果 $a_{mid}=a_{mid+1}$ ,一定没有贡献

否则，如果改了 $a_{1,...,mid}$ ，贡献为 $a_{mid+1}-a_{mid}$ ，改后面的贡献为0

那么可以维护一棵权值线段树，每次将两个子树线段树合并，然后进行区间加

复杂度 $O(n\log n)\ $

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200050
#define M 4004000
int n,v[N],tid[N],a,head[N],cnt,sz1[N],id[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int rt[N],sz[M],v2[M],ch[M][2],ct;
long long lz[M],su;
void pushdown(int x){lz[ch[x][0]]+=lz[x];lz[ch[x][1]]+=lz[x];lz[x]=0;}
void ins(int x,int l,int r,int v,int v1)
{
	sz[x]++;
	if(l==r){v2[x]=v1;return;}
	int mid=(l+r)>>1,tp=mid<v;
	if(!ch[x][tp])ch[x][tp]=++ct;
	if(tp)ins(ch[x][tp],mid+1,r,v,v1);
	else ins(ch[x][tp],l,mid,v,v1);
}
void modify(int x,int l,int r,int k,int v1)
{
	if(!x)return;
	if(l==r){lz[x]+=v1;return;}
	pushdown(x);int mid=(l+r)>>1;
	if(sz[ch[x][0]]<k)lz[ch[x][0]]+=v1,modify(ch[x][1],mid+1,r,k-sz[ch[x][0]],v1);
	else modify(ch[x][0],l,mid,k,v1);
}
int getkth(int x,int l,int r,int k)
{
	if(l==r)return v2[x];
	pushdown(x);int mid=(l+r)>>1;
	if(sz[ch[x][0]]<k)return getkth(ch[x][1],mid+1,r,k-sz[ch[x][0]]);
	else return getkth(ch[x][0],l,mid,k);
}
int merge(int x,int y)
{
	if(!x||!y)return x+y;
	pushdown(x);pushdown(y);sz[x]+=sz[y];
	ch[x][0]=merge(ch[x][0],ch[y][0]);
	ch[x][1]=merge(ch[x][1],ch[y][1]);
	return x;
}
long long que(int x,int l,int r,int v)
{
	if(l==r)return lz[x];
	pushdown(x);
	int mid=(l+r)>>1;
	if(mid>=v)return que(ch[x][0],l,mid,v);
	else return que(ch[x][1],mid+1,r,v);
}
void dfs(int u,int fa)
{
	rt[u]=++ct;sz1[u]=1;
	ins(rt[u],1,n,tid[u],v[u]);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),sz1[u]+=sz1[ed[i].t],rt[u]=merge(rt[u],rt[ed[i].t]);
	if(sz1[u]==1)modify(rt[u],1,n,1,100000-v[u]),su+=v[u];
	else
	{
		int t1=(sz1[u]+1)/2,t2=t1+1;
		int s1=getkth(rt[u],1,n,t1),s2=getkth(rt[u],1,n,t2);
		su+=s1;
		if(s1==s2)return;
		modify(rt[u],1,n,t1,s2-s1);
	}
}
bool cmp(int i,int j){return v[i]<v[j];}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),id[i]=i;
	sort(id+1,id+n+1,cmp);
	for(int i=1;i<=n;i++)tid[id[i]]=i;
	for(int i=1;i<n;i++)scanf("%d",&a),adde(i+1,a);
	dfs(1,0);
	for(int i=1;i<=n;i++)printf("%lld\n",que(rt[1],1,n,tid[i])+su);
}
```

##### T2 GCD再放送

###### Problem

定义一个序列的权值为所有区间gcd的和

现在有 $n$ 个区间，求所有 $n!$ 种拼接方式得到的序列权值和，模 $1e9+7$

$n.\sum k,a_i\leq 10^5,2s$

###### Sol

考虑套路：对于每个 $k$ 求出gcd是 $k$ 的倍数的总区间数，然后容斥得到答案

对于每个区间内部的情况，枚举 $l$ ，倍增找到 $r$ 变化的log个点求

假设当前枚举的是 $g$ ,每个区间有两种情况

1.所有数都是 $g$ 的倍数

2.有一段前缀和一段后缀是 $g$ 的倍数

考虑暴力，枚举两段算两段的贡献

设枚举的是 $a,b$ ,且 $a$ 在 $b$ 前面，则可能的贡献值是 $a$ 后缀长度乘上 $b$ 前缀长度

发现可能贡献的方案数只和剩下的所有数都是 $g$ 的倍数段数有关

注意到对于不同的 $a,b$ ，这个数值只有三种

对于每一种，可以枚举中间有几个暴力算

然后考虑这三种情况

1.两个一段前缀和一段后缀是 $g$ 的倍数的段拼起来，注意到除了一段自己的前缀和自己的后缀以外，其余都正好贡献一次，所以只需要记录前缀的长度和，后缀的长度和以及每一段前缀长度乘后缀长度的和即可

2.一个一段前缀和一段后缀是 $g$ 的倍数的段和一个所有数都是 $g$ 的倍数的段，只需记录每一个所有数都是 $g$ 的倍数的区间的长度和，另外两个和之前已经维护了

3.两个所有数都是 $g$ 的倍数的段，相当于所有数都是 $g$ 的倍数的区间的长度和的平方减去每一个区间长度平方，维护每一段长度的平方和

算两个之间转移系数需要再维护有多少个所有数都是 $g$ 的倍数的段

显然前缀后缀都是0的段没有意义，所以分解第一个数和最后一个数求上述贡献即可

复杂度 $O(\sum k*\sqrt a_i+a_i*\log a_i)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 100500
#define mod 1000000007
int n,su[N][6];//sul,sur,su(l*r),su_1,su_1 len su_1 len^2
int k,v[N],is[N],st,f1,f2,f3,fr[N],ifr[N],as[N],as1,f[N][19];
int gcd(int a,int b){return b?gcd(b,a%b):a;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void check(int x)
{
	if(is[x])return;is[x]=1;
	if(st%x==0){su[x][3]++;su[x][4]=(su[x][4]+k)%mod;su[x][5]=(su[x][5]+1ll*k*k)%mod;return;}
	int l1=1,r1=k;
	for(;v[l1]%x==0;l1++);l1--;
	for(;v[r1]%x==0;r1--);r1++;
	int v1=l1,v2=k+1-r1;
	su[x][0]=(su[x][0]+v1)%mod;
	su[x][1]=(su[x][1]+v2)%mod;
	su[x][2]=(su[x][2]+1ll*v1*v2)%mod;
}
void doit(int x){for(int i=1;i<=320;i++)if(x%i==0)check(i),check(x/i);}
void doit2(int x){for(int i=1;i<=320;i++)if(x%i==0)is[i]=is[x/i]=0;}
void solve1()
{
	for(int i=1;i<=k+1;i++)
	for(int j=0;j<=18;j++)
	f[i][j]=0;
	for(int i=1;i<=k;i++)f[i][0]=v[i];
	for(int j=1;j<=18;j++)
	for(int i=1;i+(1<<j)-1<=k;i++)
	f[i][j]=gcd(f[i][j-1],f[i+(1<<j-1)][j-1]);
	for(int i=1;i<=k;i++)
	{
		int tp1=v[i],tp2=i,las=i;
		while(las<=k)
		{
			for(int j=18;j>=0;j--)
			if(f[tp2][j]&&f[tp2][j]%tp1==0)tp2+=(1<<j);
			as1=(as1+1ll*(tp2-las)*tp1%mod*fr[n])%mod;
			tp1=gcd(tp1,v[tp2]);las=tp2;
		}
	}
}
int main()
{
	scanf("%d",&n);
	fr[0]=ifr[0]=1;for(int i=1;i<=100000;i++)fr[i]=1ll*i*fr[i-1]%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&k);st=0;
		for(int j=1;j<=k;j++)scanf("%d",&v[j]),st=gcd(v[j],st);
		doit(v[1]);doit(v[k]);
		doit2(v[1]);doit2(v[k]);
		solve1();
	}
	for(int i=1;i<=100000;i++)
	{
		f1=f2=f3=0;
		int t=su[i][3];
		for(int j=0;j<=t&&j<n;j++)f1=(f1+1ll*fr[n-j-1]*fr[t]%mod*ifr[t-j])%mod;
		for(int j=0;j<t&&j<n;j++)f2=(f2+1ll*fr[n-j-1]*fr[t-1]%mod*ifr[t-1-j])%mod;
		for(int j=0;j<t-1&&j<n;j++)f3=(f3+1ll*fr[n-j-1]*fr[t-2]%mod*ifr[t-2-j])%mod;
		int tp1=(1ll*su[i][0]*su[i][1]+mod-su[i][2])%mod,tp2=1ll*(su[i][0]+su[i][1])*su[i][4]%mod,tp3=(1ll*su[i][4]*su[i][4]-su[i][5]+mod)%mod;
		as[i]=(as[i]+1ll*tp1*f1+1ll*tp2*f2+1ll*tp3*f3)%mod;
	}
	for(int i=100000;i>=1;i--)
	for(int j=i*2;j<=100000;j+=i)as[i]=(as[i]-as[j]+mod)%mod;
	for(int i=1;i<=100000;i++)as1=(as1+1ll*as[i]*i)%mod;
	printf("%d\n",as1);
}
```

##### T3 dict

###### Problem

给定一个 $1...n$ 的排列 $p_{1...n}$ ，定义两个大小为 $n$ 的不可重集合 $A,B$ 的字典序比较方式为：

先比较 $A$ 和 $B$ 的第 $p_1$ 小的元素，较小的那个字典序较小，否则就比较第 $p_2$ 小的元素，以此类推

现在给定 $p_{1...n}$和一个大小为 $n$ 的不可重集合 $B$，求有几个值在 $1...m$，大小为 $n$ 的不可重集合 $A$ 满足 $A$ 的字典序比 $B$ 小

由于答案可能很大，你只需要输出答案对 $998244353$ 取模后的值

$n,m\leq 2\times 10^5\ $

###### Sol

考虑求对于前 $i-1$ 个位置都和 $B$ 相同且第 $i$ 个位置小于 $B$ 的方案数

注意到每一个等于的限制相当于把序列分成了若干个独立的部分，所以可以对于每个部分分开考虑

注意到每次只会改变一个区间，所以可以维护每个区间的方案数，每次计算时除掉这个区间的组合数，之后乘上分裂后两个区间的方案数

因此现在只需要算一个区间中有一个小于限制的方案数

设这个区间是 $[l,r]$ ,值为 $v_l,v_r$ ,限制为 $v_x< s$

那么暴力计算方式是 $\sum_{i=v_l+(x-l)}^{s-1} C_{i-v_l-1}^{x-l-1}C_{v_r-i-1}^{r-x-1}$

这样是 $O(n^2)$ 的

但注意到，也可以算不合法的方案数，这等价于 $C_{v_r-v_l-1}^{r-l-1}-\sum_{i=s}^{v_r-(r-x)} C_{i-v_l-1}^{x-l-1}C_{v_r-i-1}^{r-x-1}$

注意到这样的复杂度是 $min((B_x-x)-(B_l-l),(B_r-r)-(B_x-x))$

容易发现，这个复杂度等价于反向启发式合并(启发式分裂)，因此复杂度为 $O(m\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 200500
#define mod 998244353
int n,m,v[N],fr[N],ifr[N],p[N],as,as1;
set<int> fuc;
int C(int i,int j){return 1ll*fr[i]*ifr[j]%mod*ifr[i-j]%mod;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);sort(v+1,v+n+1);
	for(int i=1;i<=n;i++)scanf("%d",&p[i]);
	fr[0]=ifr[0]=1;for(int i=1;i<=m;i++)fr[i]=1ll*i*fr[i-1]%mod,ifr[i]=pw(fr[i],mod-2);
	fuc.insert(0);fuc.insert(n+1);v[n+1]=m+1;
	as1=C(m,n);
	for(int i=1;i<=n;i++)
	{
		int l=*(--(fuc.lower_bound(p[i]))),r=*(fuc.lower_bound(p[i])),t=p[i];
		as1=1ll*as1*pw(C(v[r]-v[l]-1,r-l-1),mod-2)%mod;
		int v1=v[t]-v[l]-(t-l),v2=(v[r]-v[t])-(r-t);
		if(v1<=v2)
		{
			int st1=0;
			for(int j=v[l]+(t-l);j<v[t];j++)st1=(st1+1ll*C(j-v[l]-1,t-l-1)*C(v[r]-j-1,r-t-1))%mod;
			as=(as+1ll*st1*as1)%mod;
		}
		else
		{
			int st1=C(v[r]-v[l]-1,r-l-1);
			for(int j=v[t];j<=v[r]-(r-t);j++)st1=(st1-1ll*C(j-v[l]-1,t-l-1)*C(v[r]-j-1,r-t-1)%mod+mod)%mod;
			as=(as+1ll*st1*as1)%mod;
		}
		as1=1ll*as1*C(v[t]-v[l]-1,t-l-1)%mod*C(v[r]-v[t]-1,r-t-1)%mod;
		fuc.insert(p[i]);
	}
	printf("%d\n",as);
}
```

#### 三月集训 Day 5

##### T1 买到

###### Problem

![](C:\Users\zz\Documents\pic\60.png)

$n\leq 20,T\leq 10000,4s$

###### Sol

显然对于每个点，越早到达越好

对于每个点和每条边，容易求出在 $0,...,T-1$ 时刻到达，最早什么时候能够完成

然后直接状压dp

复杂度 $O(n^2T+n^22^n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 24
#define M 202
#define K 1050000
#define S 10050
int dp[N][K],f[N][S],f1[N][S],g[M][S],g1[M][S],n,t,q,a,id[N][N],ct,ti[N][N][S],as[K];
int main()
{
	scanf("%d%d",&n,&t);
	for(int i=1;i<=n;i++)
	for(int j=i+1;j<=n;j++)
	id[i][j]=id[j][i]=++ct;
	for(int i=1;i<=n;i++)
	{
		int mn=1e9;
		for(int j=0;j<t;j++)scanf("%d",&f[i][j]),mn=min(mn,f[i][j]+j);
		for(int j=t-1;j>=0;j--)mn=min(mn+1,f[i][j]),f1[i][j]=mn;
	}
	for(int i=1;i<=n*(n-1)/2;i++)
	{
		int mn=1e9;
		for(int j=0;j<t;j++)scanf("%d",&g[i][j]),mn=min(mn,g[i][j]+j);
		for(int j=t-1;j>=0;j--)mn=min(mn+1,g[i][j]),g1[i][j]=mn;
	}
	for(int j=1;j<=n;j++)
	for(int k=1;k<=n;k++)
	if(j!=k)
	{
		int st=id[j][k];
		for(int s=0;s<t;s++)
		{
			int ti1=f1[j][s],ti2=(ti1+s)%t;
			ti[j][k][s]=ti1+g1[st][ti2];
		}
	}
	memset(dp,0x3f,sizeof(dp));
	memset(as,0x3f,sizeof(as));
	for(int i=1;i<=n;i++)dp[i][1<<i-1]=0;
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)
	if(i&(1<<j-1))
	{
		int t1=dp[j][i]%t;
		for(int k=1;k<=n;k++)
		if(!(i&(1<<k-1)))
		dp[k][i|(1<<k-1)]=min(dp[k][i|(1<<k-1)],dp[j][i]+ti[j][k][t1]);
	}
	for(int i=1;i<1<<n;i++)
	for(int j=1;j<=n;j++)
	if(i&(1<<j-1))
	as[i]=min(as[i],dp[j][i]+f1[j][dp[j][i]%t]);
	scanf("%d",&q);
	while(q--)scanf("%d",&a),printf("%d\n",as[a]);
}
```

##### T2 口罩

###### Problem

有一棵树，你可以执行以下操作 $k$ 次

删去一条边，加入一条边，要求这之后它还是一棵树

求最后可能的树的数量

$n,k\leq 5000$

###### Sol

考虑容斥，设 $f_i$ 表示钦定 $i$ 条边保留，剩下边任意的方案数，最后的容斥系数是一个类似于二项式反演的组合数

根据prufer序，如果当前 $n$ 个点被分成了 $k$ 个连通块，每个连通块大小为 $a_1,...,a_k$ ，那么方案数为 $n^{k-2}\prod a_i$

考虑右边的组合意义，相当于每个连通块选一个点的方案数

设 $dp_{i,j,0/1}$ 表示 $i$ 的子树，当前分了 $j$ 个连通块(钦定了 $size_i-j$ 条边)，当前有没有选点的方案数

转移暴力判断断不断边

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 5050
#define mod 1000000007
int dp[N][N][2],n,k,head[N],cnt,f[N],a,b,sz[N],fr[N],ifr[N];
int pw(int a,int p){if(p<0)return 1;int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	sz[u]=1;dp[u][0][0]=dp[u][0][1]=1;
	for(int i=head[u];i;i=ed[i].next)
	if(ed[i].t!=fa)
	{
		dfs(ed[i].t,u);
		for(int j=sz[u]-1;j>=0;j--)
		{
			int f1=dp[u][j][0],f2=dp[u][j][1];
			dp[u][j][0]=dp[u][j][1]=0;
			for(int k=0;k<sz[ed[i].t];k++)
			{
				dp[u][j+k][1]=(dp[u][j+k][1]+1ll*f1*dp[ed[i].t][k][1]+1ll*f2*dp[ed[i].t][k][0])%mod;
				dp[u][j+k][0]=(dp[u][j+k][0]+1ll*f1*dp[ed[i].t][k][0])%mod;
				dp[u][j+k+1][0]=(dp[u][j+k+1][0]+1ll*f1*dp[ed[i].t][k][1])%mod;
				dp[u][j+k+1][1]=(dp[u][j+k+1][1]+1ll*f2*dp[ed[i].t][k][1])%mod;
			}
		}
		sz[u]+=sz[ed[i].t];
	}
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)
	scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	for(int i=0;i<n;i++)f[i]=1ll*dp[1][i][1]*pw(n,i-1)%mod;
	f[0]=1;
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=0;i<n;i++)
	for(int j=0;j<i;j++)
	f[i]=(f[i]+1ll*(mod-1)*f[j]%mod*fr[n-1-j]%mod*ifr[i-j]%mod*ifr[n-1-i])%mod;
	int as1=0;
	for(int i=0;i<=k;i++)as1=(as1+f[i])%mod;
	printf("%d\n",as1);
}
```

##### T3 了吗

###### Problem

![](C:\Users\zz\Documents\pic\61.png)

$n,s\leq 10^6,3s$

###### Sol

对于类型0的点，相当于给 $V_1$ 加上 $0,...,k$ 中的一个

对于类型1的点，相当于将子树内值乘2

因此，对于每一个类型0的点，设它到根有 $d$ 个1，那么相当于加上 $0,2^d,...,k*2^d$

显然只有 $d\leq 19$ 的点有用，相当于有19种生成函数，每一种为 $\sum_{i=0}^kx^{2^di}$

对于一个 $d $ ,设 $y=x^{2^d}$ ,那么只需要算到 $y^{s/2^d}$

考虑每一个式子相当于 $(1-y^{k+1})/(1-y)$

这相当于 $(1-y^{k+1})*(1+y+y^2+...)$

设有 $m$ 个这样的式子相乘，那么相当于 $(1-y^{k+1})^m*(1+y+y^2+...)^m$

两边的系数都可以组合数算出，然后fft合并

最后，每次把 $x^{2^{k+1}},x^{2^k}$ 对应的合并，这样的复杂度为 $O(s\log s)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 2100555
#define mod 998244353
int n,k,s,head[N],cnt,a[N],b[N],c[N],e[N],ct[29],dep[N],v[N],f,fr[N],ifr[N],ntt[N],rev[N],as[N],as1[N],st[2][N],ct2;
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	dep[u]=dep[fa]+v[u];
	if(!v[u]&&dep[u]<20)ct[dep[u]]++;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
}
int pw(int a,int p){if(p<0)return 1;int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
	for(int l=2;l<=s;l<<=1)
	for(int i=0;i<s;i+=l)
	for(int j=i,ct1=0;j<i+(l>>1);j++,ct1++)
	{
		int v1=ntt[j],v2=1ll*ntt[j+(l>>1)]*st[t][ct1+(l>>1)]%mod;
		ntt[j]=(v1+v2)-(v1+v2>=mod?mod:0);
		ntt[j+(l>>1)]=(v1-v2+mod)-(v1>=v2?mod:0);
	}
	int inv=!t?pw(s,mod-2):1;
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void solve(int s,int a)
{
	int l=1;while(l<=s*2)l<<=1;
	for(int i=0;i<l;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(l>>1));
	if(k==1)
	{
		for(int i=0;i<=s&&i<=a;i++)as1[i]=1ll*fr[a]*ifr[i]%mod*ifr[a-i]%mod;
		for(int i=a+1;i<=s;i++)as1[i]=0;
		return;
	}
	for(int i=0;i<l;i++)c[i]=e[i]=0;
	for(int i=0;i<=a&&i*(k+1)<=s;i++)c[i*(k+1)]=1ll*fr[a]*ifr[i]%mod*ifr[a-i]%mod*((i&1)?mod-1:1)%mod;
	for(int i=0;i<=s;i++)e[i]=1ll*fr[i+a-1]*ifr[a-1]%mod*ifr[i]%mod;
	dft(l,c,1);dft(l,e,1);
	for(int i=0;i<l;i++)c[i]=1ll*c[i]*e[i]%mod;
	dft(l,c,0);
	for(int i=0;i<=s;i++)as1[i]=c[i];
}
int main()
{
	scanf("%d%d%d",&n,&k,&s);
	for(int i=2;i<=n;i++)scanf("%d",&f),adde(f,i);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	fr[0]=ifr[0]=1;for(int i=1;i<=2100000;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[2100000]=pw(fr[2100000],mod-2);
	for(int i=2099999;i>=1;i--)ifr[i]=1ll*ifr[i+1]*(i+1)%mod;
	dfs(1,0);
	as[0]=1;
	for(int t=0;t<=1;t++)
	for(int i=1;i<=21;i++)
	{
		int tp1=pw(3,(mod-1)>>i),st1=1;
		if(!t)tp1=pw(tp1,mod-2);
		for(int j=0;j<1<<i-1;j++)
		st[t][(1<<i-1)+j]=st1,st1=1ll*st1*tp1%mod;
	}
	for(int i=19;i>=0;i--)
	{
		int tp1=s>>i;
		if(!tp1||!ct[i])continue;
		solve(tp1,ct[i]);
		for(int j=(tp1>>1);j>0;j--)as[j*2]=as[j],as[j]=0;
		int l=1;while(l<=tp1*2)l<<=1;
		for(int j=tp1+1;j<l;j++)as1[j]=0;
		dft(l,as1,1);dft(l,as,1);for(int j=0;j<l;j++)as[j]=1ll*as[j]*as1[j]%mod;dft(l,as,0);
		for(int j=tp1+1;j<l;j++)as[j]=0;
	}
	int as2=0;
	for(int i=0;i<=s;i++)as2^=as[i];
	printf("%d\n",as2);
}
```
#### ZROI 暑期十连测

##### Day1 T1 H2O

###### Problem

有一个 $n\times m$ 的网格图，每个位置有一个高度 $h_{i,j}$ ，网格图外的高度为无穷大，初始时所有位置上都有积水，且水位均高于地势最高的点的高度

每一天你可以在某一个格子放一个海绵，海绵可以吸走所有这个格子上以及流入这个格子的水

这个过程持续 $k$ 天，你的策略是先最小化第1天结束时的剩余水量，再最小化第2天结束时的剩余水量，以此类推

设第 $i$ 天结束时的水量为 $a_i$ ，求出所有 $a_i$ 的异或和

$n,m\leq 500,k\leq nm$

$1s,512MB$

###### Sol

如果在A处放了海绵，考虑B处的水位

记A到B的所有路径中最高点高度的最小值为 $d$ ，如果水位高于 $d$ ，则水可以从B流到A，因此水位不超过B

如果有多个位置A，则最后的水位为所有 $d$ 的最小值

对网格图建Kruskal重构树，那么一个点的水位为所有放置了海绵的格子与它LCA中最深的一个的LCA处的权值

设重构树上点 $i$ 的高度为 $d_i$ ，父亲为 $f_i$ ，子树内为初始的 $nm$ 个点中的点的个数为 $sz_i$

可以将一个点放海绵看成给这个点到根的路径打标记，然后要求的相当于每个点祖先中最深的有标记的点的高度

显然有标记的点是它到根的链上包含根的一段，考虑差分，记点 $i$ 的权值为 $d_{f_i}-d_i$ ，那么一个点减少的水位就是这个点到根的路径上所有被标记点的权值和

那么一个点被标记的贡献为 $sz_i*(d_{f_i}-d_i)$ ，每一天可以选择一条到根的链，标记链上所有点

每次选最长的链一定是最优的，因此每次选的链一定是长链剖分后的一条，因此将剖出来的链按长度排序即可

复杂度 $O(nm\log nm)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 505
#define M 500500
#define ll long long
int n,m,k,s[N][N],fa[M],ct,f[M],ls[M],rs[M],sz[M],cnt,ct1,ct2;
ll v1[M],vl[M],dp[M],s1[M],as,res;
struct edge{int f,t,v;friend bool operator <(edge a,edge b){return a.v<b.v;}}e[M];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void dfs(int x)
{
	if(x<=n*m){sz[x]=1;dp[x]=vl[x]=(v1[f[x]]-v1[x]);return;}
	dfs(ls[x]);dfs(rs[x]);sz[x]=sz[ls[x]]+sz[rs[x]];
	if(dp[ls[x]]<dp[rs[x]])dp[x]=dp[rs[x]]+sz[x]*(v1[f[x]]-v1[x]),s1[++ct1]=dp[ls[x]];
	else dp[x]=dp[ls[x]]+sz[x]*(v1[f[x]]-v1[x]),s1[++ct1]=dp[rs[x]];
}
int main()
{
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=2*n*m;i++)fa[i]=i;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)scanf("%d",&s[i][j]),v1[i*m-m+j]=s[i][j],res-=s[i][j];
	for(int i=1;i<n;i++)for(int j=1;j<=m;j++){int tp=max(s[i][j],s[i+1][j]);e[++cnt]=(edge){i*m-m+j,i*m+j,tp};}
	for(int i=1;i<=n;i++)for(int j=1;j<m;j++){int tp=max(s[i][j],s[i][j+1]);e[++cnt]=(edge){i*m-m+j,i*m-m+j+1,tp};}
	sort(e+1,e+cnt+1);ct2=n*m;
	for(int i=1;i<=cnt;i++)
	{
		int v11=finds(e[i].f),v2=finds(e[i].t),v3=e[i].v;
		if(v11==v2)continue;
		ct2++;ls[ct2]=v11;rs[ct2]=v2;v1[ct2]=v3;fa[v11]=fa[v2]=ct2;f[v11]=f[v2]=ct2;
	}
	v1[0]=v1[ct2];dfs(ct2);s1[++cnt]=dp[ct2];sort(s1+1,s1+cnt+1);
	res+=v1[0]*n*m;for(int i=1;i<=k;i++)res-=s1[cnt-i+1],as^=res;
	printf("%lld\n",as);
}
```

##### Day1 T2 W2B

###### Problem

给定 $n,m$ ，求有多少个 $n\times m$ 的矩阵，其中每一个位置是黑色或者白色，并且满足不存在一个不同的矩阵，使得对于任意一行或任意一列，这一行/列上两个矩阵的黑色格子数量相同，答案模 $998244353$

$n,m\leq 10^5$

$1s,512MB$

###### Sol

如果两个矩阵满足题目中给的条件，找出颜色不同的位置的集合，显然每一行/列在集合中的位置中，黑色和白色的数量相同

对于一个矩阵，如果能找到这样的集合，显然存在另外一个矩阵满足题目中的要求，因此这样的矩阵不合法

考虑一个有向的二分图，两侧分别有 $n,m$ 个点，如果矩阵第 $i$ 行 $j$ 列为黑色，则左边第 $i$ 个点向右边第 $j$ 个点连有向边，否则右边第 $j$ 个点向左边第 $i$ 个点连有向边

如果这个图中有环，考虑所有边对应位置组成的集合，对于这个环上所有点，显然每个点连出的边数量和连入的边的数量相同，因此每一行/列中在集合中的位置中黑白数量相同

因此有环的图一定不合法，如果一个矩阵不合法，那么一定存在一个环，因此没有环的矩阵一定合法，只需要计数合法的图个数

没有环的矩阵一定存在拓扑序，将左边的点看做 `L` ，右边的点看做 `R` ，那么最后的拓扑序一定形如 `L...LR...RL...L...`

对于一段连续的相同字符，它们间的顺序不会对边的方向造成影响

将连续一段看成一个集合，如果两个方案对应的集合不同或者顺序不同，那么图一定不同

枚举两边分别分成了 $x,y$ 个集合，那么合法的方案一定有 $|x-y|\leq 1$ ，对于一个 $x,y$ ，方案数为 $S(n,x)S(m,y)x!y!(1+[x=y])$ 

容斥+FFT求出第二类斯特林数即可

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 263001
#define mod 998244353
int n,m,fr[N],ifr[N],f[N],g[N],h[N],as,ntt[N],rev[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)|((i&1)*(s>>1)),ntt[rev[i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	{
		int st=pw(3,mod-1+t*(mod-1)/i);
		for(int j=0;j<s;j+=i)
		for(int k=j,v1=1;k<j+(i>>1);k++,v1=1ll*v1*st%mod)
		{
			int f1=ntt[k],f2=1ll*ntt[k+(i>>1)]*v1%mod;
			ntt[k]=(f1+f2)%mod;ntt[k+(i>>1)]=(f1-f2+mod)%mod;
		}
	}
	int inv=pw(s,t==-1?mod-2:0);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
	scanf("%d%d",&n,&m);fr[0]=ifr[0]=1;for(int i=1;i<=1e5;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=n;i++)f[i]=1ll*pw(i,n)*ifr[i]%mod;
	for(int i=0;i<=n;i++)h[i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	int l=1;while(l<=n*2)l<<=1;dft(l,f,1);dft(l,h,1);for(int i=0;i<l;i++)f[i]=1ll*f[i]*h[i]%mod;dft(l,f,-1);
	for(int i=1;i<=n;i++)f[i]=1ll*f[i]*fr[i]%mod;for(int i=n+1;i<l;i++)f[i]=0;
	for(int i=0;i<l;i++)h[i]=0;
	for(int i=1;i<=m;i++)g[i]=1ll*pw(i,m)*ifr[i]%mod;
	for(int i=0;i<=m;i++)h[i]=1ll*(i&1?mod-1:1)*ifr[i]%mod;
	l=1;while(l<=m*2)l<<=1;dft(l,g,1);dft(l,h,1);for(int i=0;i<l;i++)g[i]=1ll*g[i]*h[i]%mod;dft(l,g,-1);
	for(int i=1;i<=m;i++)g[i]=1ll*g[i]*fr[i]%mod;for(int i=m+1;i<l;i++)g[i]=0;
	int as=0;for(int i=1;i<=n||i<=m;i++)as=(as+2ll*f[i]*g[i]+1ll*f[i]*g[i-1]+1ll*f[i-1]*g[i])%mod;
	printf("%d\n",as);
}
```

##### Day2 T1 Stars

###### Problem

给定 $k$ 维空间中的 $n$ 个点 $p_{1,...,n}$ ，定义一个区间 $[l,r]$ 是合法的，当且仅当存在一个点 $x$ ，使得对于 $i\in [l,r]$ ，$p_i,x$ 坐标至少有一维相同

求出所有合法区间的长度之和

$n\leq 10^5,k\leq 5$

$1s,512MB$

###### Sol

考虑如何判断一个区间是否合法

依次遍历每一个元素，同时确定 $x$ 某些维的坐标，如果当前元素与 $x$ 已经确定的坐标有相同，那么可以直接跳过，否则， $x$ 未确定的某一维一定与这个元素相同，可以枚举是哪一维，这样单次复杂度为 $O(nk!)$

也可以枚举每一次确定哪一维，设第 $i$ 次用 $p_i$ 维，枚举排列即可

注意到对于一个区间，只要考虑每一个元素的顺序固定，上面的做法就是正确的

枚举排列，考虑分治，对于跨过分治中心的区间，考虑这样一个顺序：先从分治中心开始向左依次考虑，然后再从分治中心开始向右依次考虑

从分治中心开始向左考虑，对于每一个 $i$ ，第 $i$ 次选择的数是唯一的，因此只有 $k$ 种情况：选择了 $1,2,...,k$ 维

对于每一种情况，计算它可能的最靠右的右端点，直接扫过去即可

这样复杂度为 $O(nk!k^2\log n)$

考虑一个优化：假设当前分治到的区间为 $[l,r]$ ,如果左端点为 $l$ 时当前求出来的最大合法右端点大于等于 $r$ ，那么这个区间显然不用更新

分治操作可以看成这样一个过程：初始时集合中有一个区间 $[1,n]$ ，每次从集合中拿出一个区间，进行操作，然后将接下来需要继续操作的区间放入集合

对于区间 $[l,r]$ ，记 $r_1$ 为最大的满足 $[l,r_1]$ 合法的数， $l_1$ 为最小的满足 $[l_1,r]$ 合法的数，记 $S$ 为集合中所有区间 $(r-l_1)+(r_1-l)$ 的和

考虑拿出一个区间 $[l,r]$ ，设分治中心为 $mid$ ，分情况讨论

如果 $r_1<mid<l_1$ ，那么如果这一次遍历到了 $[l_2,r_2]$ ，之后 $[l,mid-1]$ 区间的 $l_1$ 为 $l_2$ ，$[mid+1,r]$ 区间的 $r_1$ 为 $r_2$ ，之后两个区间都要继续处理，因此 $S$ 增加的值不超过这次遍历的长度

如果 $r_1\geq mid$ ，则之后左侧区间不需要继续操作，标记左侧区间，在 $[l,r]$ 遍历的长度不超过 $O(左侧区间长度+C)$， $S$ 减少的值也不超过 $O(左侧区间长度+C)$

如果 $l_1\leq mid$ ，与上面类似

最后 $S=0$ ，所以第一类情况遍历长度和为 $O(所有标记区间的长度和)+O(n)$ ，第二类长度和也是 $O(所有标记区间的长度和)+O(n)$

因为标记一个区间后不会再处理它的子区间，所以标记区间长度和为 $O(n)$

所以一次更新遍历次数为 $O(n)$ ,复杂度 $O(nk!k^2)$ ，常数很小

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
int n,k,s[N][7],t[N][7],p[7],vl[7],id[7],tr[N],tp[N];
void modify(int x,int v){for(int i=x;i<=n;i+=i&-i)if(tr[i]<v)tr[i]=v;}
int query(int x){int as=0;for(int i=x;i;i-=i&-i)if(tr[i]>as)as=tr[i];return as;}
void solve(int l,int r)
{
	if(query(l)>=r)return;
	int mid=(l+r)>>1,ct=0,is=0;
	for(int i=1;i<=k+1;i++)id[i]=0;
	for(int i=mid;i>=l;i--)
	{
		int fg=0;
		for(int j=1;j<=ct;j++)if(t[i][j]==vl[j])fg=1;
		if(!fg){if(ct==k){id[ct+1]=i;is=1;break;}ct++;vl[ct]=t[i][ct];id[ct]=i;}
	}
	if(!is)id[ct+1]=l-1;
	for(int i=ct;i>=1;i--)
	{
		int ct2=i,st1=id[i+1]+1,f1=1,as=mid;
		for(int s=mid+1;s<=r;s++)
		{
			int fg=0;
			for(int j=1;j<=ct2;j++)if(t[s][j]==vl[j])fg=1;
			if(!fg){if(ct2==k){f1=0;as=s-1;break;}ct2++;vl[ct2]=t[s][ct2];}
		}
		if(f1)as=r;
		modify(st1,as);
	}
	solve(l,mid);solve(mid+1,r);
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)for(int j=1;j<=k;j++)scanf("%d",&s[i][j]);
	for(int i=1;i<=k;i++)p[i]=i;
	do{
		for(int i=1;i<=n;i++)for(int j=1;j<=k;j++)t[i][j]=s[i][p[j]];
		solve(1,n);
	}while(next_permutation(p+1,p+k+1));
	long long as=0;for(int i=1;i<=n;i++){int st=query(i);as+=1ll*(st-i+1)*(st-i+2)/2;}
	printf("%lld\n",as);
}
```

##### Day2 T2 Decode

###### Problem

给一个二进制串 $S$，定义好的子串为满足如下条件的子串

1. 首位是1
2. 如果将这个串看成二进制数，那么它是一个完全平方数

求好的子串的长度最大值

串的生成方式为先随机一个平方数，然后在两侧加入一些随机字符

$n\leq 5\times 10^5$

$5s,512MB$

###### Sol

考虑如何判断一个串是不是平方数，考虑随机模一些质数，如果这个串的值模这个质数后不是二次剩余，那么它一定不是完全平方数

因此不是完全平方数的串一次有大约1/2的概率判断出来

直接判断 $\frac{n(n+1)}2$ 个串不可行，考虑利用随机性减少需要的串

首先只用判断开头是1的串，并且一个偶数的平方数一定是某个奇数的平方数乘上4的若干次方，也就是在末尾加上偶数个0，所以可以只考虑奇数的平方数对应的串，然后dp算出所有答案

因为奇数平方后三位为001，所以这样可以让判断数量除以16

然后考虑对奇质数 $p$ 取模，设 $s_i=\sum_{j=1}^iS_j2^{n-j}$ ，那么 $(l,r)$ 的数值为 $\frac{s_r-s_{l-1}}{2^{n-r}}$

根据 $n-r$ 的奇偶性，判断这个值是不是二次剩余等价于判断 $s_r-s_{l-1}$ 或 $2(s_r-s_{l-1})$ 是不是二次剩余

那么与 $r$ 相关的只有 $r\bmod 2,s_r\bmod p$ 的值，这样的值只有 $2p$ 个，可以对于每一个预处理出哪些 $l$ 是合法的

选择10个左右的小质数，然后枚举可能的 $r$ ，求出模这些质数下可能的 $l$ 集合的交，这时合法的 $l$ 不会太多，然后用之前的方法判断即可

复杂度 $O(\frac{n^2}{???})$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<bitset>
using namespace std;
#define N 500500
#define ll long long
int pw(int a,int p,int mod){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
vector<pair<int,int> > s1,s2;
char s[N];
int n,as,p1[N],p2[N],sr[N],v1[N],v2[N],is[N],pr[N],ch[N],ct,dp[N],s11[N],is2[N],pr1[12]={0,3,5,7,11,13,17,19,23,29,31,37},su1[12][N],is3[12],f1[N],ct2;
bitset<280050> is1[12][75],as1;
bool check(ll tp){int t=sqrt(tp)+0.3;if(1ll*t*t==tp||1ll*(t-1)*(t-1)==tp||1ll*(t+1)*(t+1)==tp)return 1;return 0;}
void getpr(int s)
{
	for(int i=2;i<=s;i++)
	{
		if(!ch[i])pr[++ct]=i;
		for(int j=1;j<=ct&&1ll*i*pr[j]<=s;j++)
		{
			ch[i*pr[j]]=1;
			if(i%pr[j]==0)break;
		}
	}
}
void doit()
{
	for(int i=1;i<=700;i++)ch[i]=0;ct=0;
	getpr(700);
	for(int i=1;i<=45;i++)
	{
		int tp=rand()%700;
		while(tp<30||ch[tp])tp=rand()%700;
		ch[tp]=1;p1[0]=1;
		for(int j=0;j<tp;j++)is2[j]=0;
		for(int j=0;j<tp;j++)is2[j*j%tp]=1;
		for(int j=1;j<=n;j++)s11[j]=(2*s11[j-1]+(s[j]-'0'))%tp,p1[j]=p1[j-1]*2%tp;
		for(int j=0;j<s1.size();j++)
		{
			int l=s1[j].first,r=s1[j].second;
			int vl=(s11[r]-1ll*s11[l-1]*p1[r-l+1]%tp+tp)%tp;
			if(is2[vl])s2.push_back(make_pair(l,r));
		}
		s1=s2;s2.clear();vector<pair<int,int> >().swap(s2);
	}
	for(int i=0;i<s1.size();i++)dp[s1[i].second]=max(dp[s1[i].second],s1[i].second-s1[i].first+1);
	s1.clear();vector<pair<int,int> >().swap(s1);
}
int main()
{
	for(int j=1;j<=11;j++)if(pw(2,pr1[j]/2,pr1[j])==pr1[j]-1)is3[j]=1;
	scanf("%d%s",&n,s+1);
	for(int i=1;i<=n;i++)if(s[i]=='1')
	{
		ll su=0;
		for(int j=i;j<=n&&j<=i+50;j++)
		{
			su=2*su+s[j]-'0';
			if(check(su))dp[j]=max(dp[j],j-i+1);
		}
	}
	for(int i=1;i<n;i++)if(s[i]=='1')v1[i]=++ct2,f1[ct2]=i;
	for(int i=3;i<=n;i++)if(s[i]=='1'&&s[i-1]=='0'&&s[i-2]=='0')v2[i]=1;
	for(int i=1;i<=11;i++)
	{
		p1[0]=1;for(int j=1;j<=n;j++)p1[j]=2*p1[j-1]%pr1[i];
		for(int j=0;j<pr1[i]*2;j++)is2[j]=0;
		for(int j=0;j<pr1[i];j++)is2[j*j%pr1[i]]=is2[j*j%pr1[i]+pr1[i]]=1;
		for(int j=1;j<=n;j++)su1[i][j]=(su1[i][j-1]+(s[j]-'0')*p1[n-j])%pr1[i];
		for(int j=1;j<=n;j++)if(v1[j])
		for(int k=0;k<pr1[i];k++)if(k==su1[i][j-1])is1[i][k].set(v1[j],1),is1[i][k+pr1[i]].set(v1[j],1);else if(is2[k+pr1[i]-su1[i][j-1]])is1[i][k].set(v1[j],1);else is1[i][k+pr1[i]].set(v1[j],1);
	}
	for(int i=1;i<=n;i++)
	if(v2[i])
	{
		as1.reset();as1.flip();
		for(int j=1;j<=11;j++)as1&=is1[j][su1[j][i]+((n-i)&1)*pr1[j]*is3[j]];
		if(as1.none())continue;
		int st=0;
		while(1)
		{
			st=as1._Find_next(st+1);
			if(f1[st]>=i-40)break;
			s1.push_back(make_pair(f1[st],i));
		}
	}
	doit();
	for(int i=1;i<n-1;i++)if(dp[i]&&s[i+1]=='0'&&s[i+2]=='0')dp[i+2]=max(dp[i+2],dp[i]+2);
	for(int i=1;i<=n;i++)as=max(as,dp[i]);
	printf("%d\n",as);
}
```

##### Day3 T1 Good Subsegments

###### Problem

给一个序列，第 $i$ 个位置的数为 $2^{a_i}$ ，求有多少个区间的和是某个2的次幂

$n\leq 3\times 10^5$

$2s,512MB$

###### Sol

对原序列分治，考虑跨过分治中心的所有区间是否合法

设分治中心为 $mid$ ，若 $x\leq mid$ ，记 $s_x=\sum_{i=x}^{mid}2^{a_i}$ ，否则记 $s_x=\sum_{i=mid+1}^r2^{a_i}$

那么一个合法的区间 $(l,r)(l\leq mid<r)$ 满足 $\exists v,2^v=s_l+s_r$

考虑 $s_l\geq s_r$ 的情况，设 $t_l$ 为最小的 $x\in Z$ 满足 $2^x>s_l$ ，那么可能的 $s_l+s_r\in \{2^{t_l},2^{t_l-1}\}$ ，枚举是哪一个，对于一个质数 $p$ ,可以快速求出 $s_i\mod p$ ，使用hash表找出模 $p$ 意义下和相等的区间(不需要考虑 $s_l\geq s_r$ 的限制)，再模几个质数判断即可

直接使用set可以求出 $t_l$ ，但是常数过大

对于 $l$ ，设 $v_l=max_{i=l}^{mid}a_i$ ，只考虑 $a_i+\log n\geq v_l$ 的数，它们的和可以快速求出，对于剩下的数，因为它们的和不超过 $2^{v_l}$ ，所以这样求出的 $t_l$ 最多比真实的 $t_l$ 少1，因此判断 $\{2^{t_l},2^{t_l+1},2^{t_l+2}\}$ 即可

$s_l<s_r$ 的情况同理

如果将hash表看成单次 $O(1)$ 则复杂度为 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 200500
#define f1 1050000011
#define ll long long
int pw1(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%f1;a=1ll*a*a%f1;p>>=1;}return as;}
int pw(int a,int p,int mod){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int su[N],mxbit[N],n,v[N],su1[6][N],pr[6]={0,1050000031},as,f11[N],g1[N],st5[N],ct3,f12[N],g2[N],as1;
bool check(int l,int r,int fu,int tp1){for(int i=1;i<=1;i++)if((su1[i][r]-su1[i][l-1]+pr[i])%pr[i]!=tp1)return 0;return 1;}
struct sth{int a,b;friend bool operator <(sth a,sth b){return a.a==b.a?a.b<b.b:a.a<b.a;}}fu[N];
struct ht{
	int hd[N*6],nt[N*6],v[N*6],s[N*6],ct;
	queue<int> st1;
	ht(){ct=0;}
	void add(int x,int y){s[++ct]=y;v[ct]=x;nt[ct]=hd[x&1048575];hd[x&1048575]=ct;st1.push(x&1048575);}
	void init(){ct=0;while(!st1.empty())hd[st1.front()]=0,st1.pop();}
	void finds(int x){for(int i=hd[x&1048575];i;i=nt[i])if(v[i]==x)st5[++ct3]=s[i];}
}fuc;
struct ht2{
	int hd[N*6],nt[N*25],ct;
	ll v[N*25];
	void add(ll x){v[++ct]=x;nt[ct]=hd[x&1048575];hd[x&1048575]=ct;}
	bool finds(ll x){for(int i=hd[x&1048575];i;i=nt[i])if(v[i]==x)return 1;return 0;}
}fu1;
void ins(ll x){if(!fu1.finds(x))fu1.add(x),as1++;}
void solve(int l,int r)
{
	if(l>r)return;
	int mid=(l+r)>>1,s1=0,ct=0;
	fuc.init();
	for(int i=mid;i>=l;i--)s1=(s1+f11[i])%f1,fuc.add(s1,i);
	sort(fu+1,fu+ct+1);
	s1=0;mxbit[mid]=0;
	int vl=0;
	for(int i=mid+1;i<=r;i++)
	{
		int st=v[i],v2=f11[i],tp1;s1=(s1+f11[i])%f1;
		if(st>mxbit[i-1])
		{
			if(st-mxbit[i-1]>22)vl=0;
			else vl>>=st-mxbit[i-1];
			vl+=1<<20;
			mxbit[i]=st,g1[i]=v2,g2[i]=tp1=f12[i];
		}
		else 
		{
			int tp=st-mxbit[i-1]+20;
			if(tp>=0)vl+=1<<tp;
			mxbit[i]=mxbit[i-1],g1[i]=g1[i-1],g2[i]=tp1=g2[i-1];
		}
		if(vl>=(1<<21))vl>>=1,mxbit[i]++,g1[i]=2*g1[i]%f1,tp1=2*tp1%1050000031,g2[i]=tp1;
		for(int s=1,v1=g1[i];s<4;s++,v1=v1*2%f1)
		{
			int a1=(2ll*v1+f1-s1)%f1;
			ct3=0;fuc.finds(a1);
			tp1=2*tp1%1050000031;
			for(int j=1;j<=ct3;j++){if(check(st5[j],i,mxbit[i]+s,tp1))ins(st5[j]*1000001ll+i);}
		}
	}
	ct=0;s1=0;
	for(int i=mid+1;i<=r;i++)s1=(s1+f11[i])%f1,fuc.add(s1,i);
	s1=0;sort(fu+1,fu+ct+1);
	mxbit[mid+1]=0;vl=0;
	for(int i=mid;i>=l;i--)
	{
		int st=v[i],v2=f11[i],tp1;s1=(s1+f11[i])%f1;
		if(st>mxbit[i+1])
		{
			if(st-mxbit[i+1]>22)vl=0;
			else vl>>=st-mxbit[i+1];
			vl+=1<<20;
			mxbit[i]=st,g1[i]=v2,g2[i]=tp1=f12[i];
		}
		else 
		{
			int tp=st-mxbit[i+1]+20;
			if(tp>=0)vl+=1<<tp;
			mxbit[i]=mxbit[i+1],g1[i]=g1[i+1],g2[i]=tp1=g2[i+1];
		}
		if(vl>=(1<<21))vl>>=1,mxbit[i]++,g1[i]=2*g1[i]%f1,tp1=2*tp1%1050000031,g2[i]=tp1;
		for(int s=1,v1=g1[i];s<4;s++,v1=v1*2%f1)
		{
			int a1=(2ll*v1+f1-s1)%f1;
			ct3=0;fuc.finds(a1);
			tp1=2*tp1%1050000031;
			for(int j=1;j<=ct3;j++){if(check(i,st5[j],mxbit[i]+s,tp1))ins(i*1000001ll+st5[j]);}
		}
	}
	if(l==r)return;
	solve(l,mid);solve(mid+1,r);
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),f11[i]=pw1(2,v[i]),f12[i]=pw(2,v[i],1050000031);
	for(int i=1;i<=1;i++)for(int j=1;j<=n;j++)su1[i][j]=(su1[i][j-1]+pw(2,v[j],pr[i]))%pr[i];
	solve(1,n);
	printf("%d\n",as1+n);
}
```

##### Day3 T2 Easy Sum

###### Problem

给定 $n$ 对 $a_i,b_i(0\leq a_i,b_i<n)$ ，对于每一个 $0\leq k<n$ ，求出 $\sum_{i=1}^nC_{a_i+b_i-k}^{a_i}\bmod 998244353$

$n\leq 10^5$

$8s,512MB$

###### Sol

将答案看成生成函数 $F$ ，那么有 $F(x)=\sum_{i=1}^n\sum C_{a_i+b_i-k}^{a_i} x^k$

$F(x)=\sum_{i=1}^n\sum C_{a_i+k}^{a_i} x^{b_i-k}$

$F(x)=\sum_{i=1}^nx^{b_i}\sum C_{a_i+k}^{k} x^{-k}$

设 $y=x^{-1}$ ，那么 $F(x)=\sum_{i=1}^ny^{-b_i}\sum C_{a_i+k}^{k} y^k$

$F(x)=\sum_{i=1}^ny^{-b_i}\frac 1{(1-y)^{a_i+1}}$

$F(x)=\frac{\sum_{i=1}^ny^{-b_i}(1-y)^{n-a_i-1}}{(1-y)^n}$

需要求的是 $x^0,x^1,...,x^{n-1}$ 项的系数，相当于 $y^0,y^{-1},...,y^{-(n-1)}$ 项的系数

记 $G(x)=y^nF(x)$ ，只需要求出 $G$ $y^1,...,y^n$ 的系数即可

$G(x)=\frac{\sum_{i=1}^ny^{n-b_i}(1-y)^{n-a_i-1}}{(1-y)^n}$

按 $n-a_i-1$ 分块，考虑求出 $n-a_i-1\in[kB,(k+1)B)$ 的所有对对分子的贡献，相当于 $(\sum y^{n-b_i}(1-y)^{n-a_i-1-kB})(1-y)^{kB}$

第一部分每一个 $(1-y)$ 的次数不超过 $B$ ，暴力展开求和后和右边相乘即可

$B=O(\sqrt{n\log n})$ 时复杂度为 $O(n\sqrt{n\log n})$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 263001
#define mod 998244353
int n,s[N][2],tp[N],ntt[N],a[N],b[N],c[N],as[N],fu[N],g[2][N*2],rev[N*2],fr[N],ifr[N],C[2111][2111];
bool cmp(int a,int b){return s[a][0]<s[b][0];}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init()
{
	for(int i=0;i<2;i++)
	for(int j=2;j<=1<<18;j<<=1)
	for(int k=0;k*2<=j;k++)
	g[i][j+k]=pw(3,mod-1+(i?-1:1)*(mod-1)/j*k);
	for(int j=2;j<=1<<18;j<<=1)
	for(int k=0;k<j;k++)
	rev[j+k]=(rev[(k>>1)+j]>>1)|(k&1?(j>>1):0);
	fr[0]=ifr[0]=1;for(int i=1;i<=2e5;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=0;i<=2000;i++)C[i][0]=C[i][i]=1;
	for(int i=2;i<=2000;i++)
	for(int j=1;j<i;j++)C[i][j]=(C[i-1][j]+C[i-1][j-1])%mod;
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
int main()
{
	scanf("%d",&n);init();
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),s[i][0]++,s[i][0]=(n-s[i][0]),tp[i]=i;
	sort(tp+1,tp+n+1,cmp);
	int lb=1;b[0]=1;
	int l=1;while(l<=n*2)l<<=1;
	for(int i=0;i<=1000;i++)c[i]=1ll*C[1000][i]*(i&1?mod-1:1)%mod;dft(l,c,1);
	for(int tp1=1000;lb<=n;tp1+=1000)
	{
		int rb=lb;while(rb<n&&s[tp[rb+1]][0]<=tp1)rb++;
		for(int i=0;i<l;i++)a[i]=0;
		for(int i=lb;i<=rb;i++){int st=s[tp[i]][0]-tp1+1000,s1=s[tp[i]][1];for(int j=0;j<=s1&&j<=st;j++)a[n-s1+j]=(a[n-s1+j]+1ll*C[st][j]*(j&1?mod-1:1))%mod;}
		dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,0);
		for(int i=0;i<=n;i++)as[i]=(as[i]+a[i])%mod;for(int i=0;i<l;i++)b[i]=1ll*c[i]*b[i]%mod;dft(l,b,0);for(int i=n+1;i<l;i++)b[i]=0;lb=rb+1;
	}
	for(int i=0;i<l;i++)a[i]=0;
	for(int i=0;i<=n;i++)a[i]=1ll*fr[n+i-1]*ifr[i]%mod*ifr[n-1]%mod;
	dft(l,a,1);dft(l,as,1);for(int i=0;i<l;i++)as[i]=1ll*as[i]*a[i]%mod;dft(l,as,0);
	for(int i=0;i<n;i++)printf("%d ",as[n-i]);
}
```

##### Day3 T3 Funny Cost

###### Problem

对于一个长度为 $n-1$ 的序列 $p_{1,...,n-1}$，定义它的权值为

考虑一个 $n$ 个点的图，点编号为 $0,...,n-1$ ，对于 $i<j$ ,$i,j$ 之间连有一条边，权值为 $max_{k=i+1}^jp_k$ ，这个图的最大权完美匹配的权值

给一个序列 $a$ ，将其随机排列，求出权值的期望乘上 $(n-1)!$ 的答案，模 $998244353$

$n\leq 10^5$

$1s,256MB$

###### Sol

对于每一个 $d$ ，考虑匹配中边权大于等于 $d$ 的数的个数

所有大于等于 $d$ 的数将 $n+1$ 个点划分成若干段，每一段内部的边权小于 $d$ ，其余的边权大于等于 $d$

设 $k=\frac{n+1}2$ ，如果有一个区间长度大于 $k$ ，设区间长度为 $x$ ，那么至少有 $x-k$ 条匹配边权小于 $x$ ，否则可以使所有匹配边权都大于等于 $x$

考虑匹配 $(0,k),(1,k+1),...,(k-1,n)$ ，可以发现，对于每一个 $d$ ，匹配中大于等于 $d$ 的边的数量都是可能的最大值，因此这样的匹配是最优匹配

因此只需要求一个长度为 $k$ 的区间的最大值的期望再乘上 $k$ 即可

复杂度 $O(n)$ 或 $O(n\log mod)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define mod 998244353
int n,v[N],fr[N],ifr[N],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	fr[0]=ifr[0]=1;for(int i=1;i<=n;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=(n+1)/2;i<=n;i++)as=(as+1ll*fr[n/2]*fr[n/2+1]%mod*fr[i-1]%mod*ifr[i-1-n/2]%mod*ifr[n/2]%mod*v[i]%mod*(n/2+1))%mod;
	printf("%d\n",as);
}
```

##### Day4 T1 Alternating Paths

###### Problem

有一个图，边有黑白两种颜色，有一些点为关键点，关键点有点权

定义一条路径是好的，当且仅当起点终点都是关键点，不连续经过两条颜色相同的路径且不经过重复点，这条路径的权值为起点和终点的点权之和

在图中找出一些路径，使得每个点最多出现在一条路径中，并最大化所有路径的权值和，求出最大权值和

$n\leq 100$

$1s,512MB$

###### Sol

对于一个点，它有三种情况：作为起点/终点，作为一个路径中间的点或者不是路径上的点

第一种情况需要连1条边，第二种情况需要连黑白各一条边，第三种情况不能连边

对于后两种情况，考虑将一个点拆成黑白两个点，两个点间连边，对于原图的黑边，在两个点对应的黑点间连边，白边在对应的白点间连边

考虑这个图的一个完美匹配，如果一个点对应的两个点直接匹配了，相当于这个点不在路径上，否则黑点白点都会和其它对应颜色的点匹配，相当于这个点在某个路径上

考虑第一种情况，对于每一个关键点，新建一个点，这个点和关键点对应的黑点白点各连一条边，边权为关键点点权，如果这个点和黑点白点中的一个匹配了，那么这个点对应的点中只会再匹配一条边，这对应这个点作为一个路径开头/结尾的情况

为了处理新建点可以不匹配的情况，可以将所有新建点间连边，如果关键点数量为奇数就再加一个点连边，这些边权值都是0

然后写一个一般图最大权完美匹配即可

复杂度 $O(n^3)$

###### Code

没写

##### Day4 T2 Dispatch Money

###### Problem

给一个排列，你可以把它划分成若干段，若分成了 $k$ 段，则代价为 $kx$

然后你需要使得每一段内部递增，你每次可以交换一段内相邻的两个数，代价为1

你需要最小化两部分代价总和，求最小代价和

$n\leq 3\times 10^5$

$5s,512MB$

###### Sol

设 $f_{l,r}$ 表示划分出 $(l,r)$ 后这一段排好序的最小代价，显然这等于这一段的逆序对数

因此可以发现 $f$ 满足四边形不等式，因此存在决策单调性

因为 $f$ 不能直接快速求，所以不能使用单调栈做法

考虑先cdq分治，每一个区间上再使用决策单调性的分治做法，可以发现，在进行第二层分治的时候，依次考虑所有询问的区间端点，可以发现 $l,r$ 的变化量之和是 $O(n\log n)$ 的，因此总的变化量是 $O(n\log^2 n)$ 的

相当于需要支持在序列两侧加入或删除一个数，询问当前序列逆序对数

一种做法是直接BIT，复杂度 $O(n\log^3 n)$ ，因为BIT极小的常数，可以通过

注意到在两侧加入或删除一个数时 $f$ 的变化量相当于求在 $[l,r]$ 区间中有多少个小于 $x$ 的数，可以拆成若干个求在 $[1,r]$ 区间中有多少个小于 $x$ 的数

如果离线的话可以使用根号平衡做到 $O(\sqrt n)$ 修改 $O(1)$ 查询，把这个东西可持久化即可

复杂度 $O(n\sqrt n+n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,k,tr[N],ct,l=1,r,v[N];
ll as,dp[N];
void modify(int x,int v){for(int i=x;i<=n;i+=i&-i)tr[i]+=v;}
int query(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
void addl(int x){as+=query(x);modify(x,1);ct++;}
void addr(int x){as+=ct-query(x);modify(x,1);ct++;}
void dell(int x){modify(x,-1);ct--;as-=query(x);}
void delr(int x){modify(x,-1);ct--;as-=ct-query(x);}
ll calc(int l1,int r1){while(l>l1)addl(v[--l]);while(r<r1)addr(v[++r]);while(l<l1)dell(v[l++]);while(r>r1)delr(v[r--]);return as;}
void solve(int l,int r,int l1,int r1)
{
	if(l1>r1)return;
	int mid=(l1+r1)>>1;
	ll as=1e18,fr=0;
	for(int i=l;i<=r;i++)
	{
		ll tp=dp[i]+calc(i+1,mid)+k;
		if(tp<as)as=tp,fr=i;
	}
	dp[mid]=min(dp[mid],as);
	solve(l,fr,l1,mid-1);solve(fr,r,mid+1,r1);
}
void cdq(int l,int r)
{
	if(l==r)return;
	int mid=(l+r)>>1;
	cdq(l,mid);solve(l,mid,mid+1,r);cdq(mid+1,r);
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]),dp[i]=1e18;
	cdq(0,n);printf("%lld\n",dp[n]);
}
```

##### Day4 T3 Exercise

###### Problem

随机生成 $n$ 对 $(x_i,y_i)$ ，给定 $k$ ，你需要找到一对 $i,j$ ，满足 $x_ix_j+y_iy_j=k$ ，保证有解

$n\leq 2\times 10^5$

$6s,512MB$

###### Sol

选取一个质数 $p$ ，考虑 $x_ix_j+y_iy_j\equiv k(\bmod p)$ 的对

可以发现，在随机的情况下，这个方程成立的概率是 $\frac 1p$

枚举 $x_i,x_j,y_i\bmod p$ ，那么有 $y_iy_j\equiv k-x_ix_j(\bmod p)$ ，可以求出所有 $y_j\bmod p$

对于每一对 $(x,y)$ ，记录 $x_i\equiv x,y_i\equiv y(\bmod p)$ 的所有 $(x_i,y_i)$ ，判断所有模 $p$ 后相等的对即可

$p$ 取 $O(\sqrt n)$ 时复杂度为 $O(n\sqrt n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 205001
#define M 350
#define mod 331
#define ll long long
int n,s[N][2],inv[N];
vector<int> f[M][M];
ll k;
ll cross(int a,int b){return 1ll*s[a][0]*s[b][0]+1ll*s[a][1]*s[b][1];}
void check(int a,int b,int c,int d)
{
	int s1=f[a][b].size(),s2=f[c][d].size();
	for(int i=0;i<s1;i++)
	for(int j=0;j<s2;j++)
	if(cross(f[a][b][i],f[c][d][j])==k){printf("%d %d\n",f[a][b][i],f[c][d][j]);exit(0);}
}
int main()
{
	scanf("%d%lld",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),f[s[i][0]%mod][s[i][1]%mod].push_back(i);
	for(int i=1;i<mod;i++)
	for(int j=1;j<mod;j++)
	if(i*j%mod==1)inv[i]=j;
	int tp=k%mod;
	for(int i=0;i<mod;i++)
	for(int j=0;j<mod;j++)
	for(int l=0;l<mod;l++)
	{
		int f1=(tp-i*j%mod+mod)%mod;
		if(l>0)
		{
			int f2=f1*inv[l]%mod;
			check(i,l,j,f2);
		}
		else if(f1==0)for(int s=0;s<mod;s++)check(i,l,j,s);
	}
}
```

##### Day5 T1 Interstellar

###### Problem

给一棵有根树，边有边权，第 $i$ 条边的边权为 $2^{a_i}$ ，定义从 $x$ 开始的旅行包含以下两步

1. 考虑 $x$ 的每个儿子 $v$ ，从 $x$ 走到 $v$ ，这一步会产生这条边权的代价，进行一次从 $v$ 开始的旅行，再回到 $x$
2. 选择 $x$ 的一个儿子 $v$ ，从 $x$ 走到 $v$ ，这一步会产生这条边权的代价，进行一次从 $v$ 开始的旅行，再回到 $x$

求从根出发的旅行的最大代价，对 $998244353$ 取模

$n\leq 10^5,a_i\leq 10^{18}$

$1s,512MB$

###### Sol

设 $x$ 父亲到它的边权为 $2^{v_x}$ , $dp_x$ 表示从 $x$ 的父亲走到它，，进行一次旅行再返回的最大代价

有 $dp_x=\sum_{y\in son_x}dp_y+max_{x\in son_x}dp_y+2^{v_x}$

相当于给最大的一个子树dp值乘上2，再合并

可以发现，设 $sz_x$ 为 $x$ 子树内点数，则将 $dp_x$ 写成二进制数，得到的表示中1的个数不会超过 $sz_x$ ，因此可以维护所有 $dp$ 的二进制表示中所有1的位置

设两个二进制数分别有 $a,b$ 个1，使用set维护位置，可以使用 $O(min(a,b)\log n)$ 的时间比较两个数的大小，因此找出每个点dp值最大的儿子复杂度为 $O(n\log^2 n)$

将一个二进制数乘2相当于将所有1的位置向前一位，相当于set中元素整体加1，可以直接打标记，然后启发式合并即可

复杂度 $O(n\log^2 n)$

###### Code

```cpp
#include<cstdio>
#include<set>
using namespace std;
#define N 100500
#define mod 998244353
#define ll long long
set<ll> st[N];
int n,head[N],cnt,id[N],lz[N],a;
ll b;
struct edge{int t,next;long long v;}ed[N];
int pw(int a,long long p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void adde(int f,int t,ll v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;}
bool cmp(int a,int b)
{
	set<ll>::reverse_iterator it1=st[a].rbegin(),it2=st[b].rbegin();
	while(1)
	{
		if(it1==st[a].rend())return 1;
		if(it2==st[b].rend())return 0;
		ll v1=(*it1)+lz[a],v2=(*it2)+lz[b];
		if(v1<v2)return 1;
		if(v2<v1)return 0;
		it1++;it2++;
	}
}
void insert(int a,ll b)
{
	while(st[a].count(b))st[a].erase(b),b++;
	st[a].insert(b);
}
int merge(int a,int b)
{
	if(st[a].size()<st[b].size())a^=b^=a^=b;
	for(set<ll>::iterator it=st[b].begin();it!=st[b].end();it++)
	insert(a,(*it)+lz[b]-lz[a]);
	return a;
}
void dfs(int u,ll vl)
{
	id[u]=u;st[u].insert(vl);
	int mx=0;
	for(int i=head[u];i;i=ed[i].next)
	{
		dfs(ed[i].t,ed[i].v);
		if(!mx||cmp(id[mx],id[ed[i].t]))mx=ed[i].t;
	}
	for(int i=head[u];i;i=ed[i].next)
	if(mx==ed[i].t)lz[id[ed[i].t]]++,id[u]=merge(id[u],id[ed[i].t]);
	else id[u]=merge(id[u],id[ed[i].t]);
}
int main()
{
	scanf("%d",&n);
	for(int i=2;i<=n;i++)scanf("%d%lld",&a,&b),adde(a,i,b);
	dfs(1,0);
	int as=0;
	for(set<ll>::iterator it=st[id[1]].begin();it!=st[id[1]].end();it++)
	as=(as+1ll*pw(2,(*it)+lz[id[1]]))%mod;
	as=(as+mod-1)%mod;
	printf("%d\n",as);
}
```

##### Day5 T2 K-bag Sequence

###### Problem

给定 $n,k$ ，定义一个序列是好的当且仅当它是有若干个 $1,2,...,k$ 的排列拼接而成的，求有多少个序列满足

1. 长度为 $n$
2. 它是一个好的序列的子串

答案对 $998244353$ 取模

多组数据

$n\leq 10^9,k\leq 60,T\leq 5$

$2s,512MB$

###### Sol

如果一个序列合法，相当于存在一个 $[1,k]$ 中的位置，从这个位置左侧划分开，之后每 $k$ 个位置划分开，使得划分的每一段都不存在相同元素

特判 $n\leq k$ 的情况，对于剩下的情况，以 $[1,k]$ 中的位置开头划分的方案是不同的

考虑容斥，对于每一个 $S\subset\{1,2,...,k\}$ ，考虑计算对于每一个 $i\in S$ ，从 $i$ 开头都存在合法划分方案的序列数量

考虑所有划分方案的所有划分点，考虑这些划分点分出的若干段，对于不是开头也不是结尾的一段 $[i,j]$ ，如果 $j+k\leq n$ ，一定存在一段 $[i+k,j+k]$

因为 $[i,i+k-1]$ 是一个排列， $[j+1,j+k]$ 是一个排列，因此 $[i,j]$ 的元素集合和 $[i+k,j+k]$ 的元素集合相同

显然，如果不考虑开头结尾，对于每一个中间没有划分点，且两侧都是划分点的的段 $[l,r]$ ，$[l,r],[l+k,r+k]$ 内元素集合相同，那么这样的序列一定满足在每一个 $i\in S$ 开头划分都合法

那么这部分的方案数为考虑所有元素集合不同的段，先将元素分到这些段中，然后每一段每一次出现内部都可以任意排列

对于每一个 $i\in S$ ，找到集合中第一个大于它的数 $j$ (如果是最大的数就找最小的) ，可以得到 $|S|$ 个 $[i,j)$ ，可以发现这些区间内部没有别的划分点，且这些区间的元素集合就是所有可能的元素集合

因此只需要枚举每一段，计算这一段的出现次数，设这一段长度为 $l$ ，出现次数为 $c$ ，那么这一段的贡献为 $\frac 1{l!}*(l!)^c=(l!)^{c-1}$ ，最后的方案数就是每一个这样的段的方案数的乘积再乘上 $k!$

然后考虑开头结尾，对于开头，找到集合中最大最小的元素 $mx,mn$ ，那么开头有 $mn-1$ 个元素，且开头的 $mx-1$ 个元素互不相同，可以发现这样的方案数为 $\frac{(k-mx+1)!}{(k-mn+1)!}$

对于结尾，可以发现要找的是 $(n+1-i)\bmod k$ 的最大最小值，然后和上面类似

如果 $[i,j]\subset[1,n\bmod k+1]$ ，那么 $[i,j]$ 出现次数一定是 $\lfloor \frac nk\rfloor+1$ ，如果 $[i,j]\subset (n\bmod k+1,k]$ ，那么出现次数是 $\lfloor \frac nk\rfloor$ 

可以发现，开头结尾的贡献只和 $[1,n\bmod k+1]$ 中的最大最小数， $(n\bmod k+1,k]$ 中的最大最小数有关，考虑预处理出 $[1,n\bmod k+1]$ 中长度为 $l$ 的段，划分成若干段，这部分的权值乘上容斥系数的和，以及 $(n\bmod k+1,k]$ 中长度为 $l$ 的段，划分成若干段，这部分的权值乘上容斥系数的和 ，这部分可以 $O(k^2)$ 求出

然后枚举 $[1,n\bmod k+1]$ 中的最大最小数， $(n\bmod k+1,k]$ 中的最大最小数，每一部分内部的所有方案的权值和就是上面的东西，两部分之间的段的贡献和开头结尾的贡献可以直接算

复杂度 $O(Tk^4)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 65
#define mod 998244353
int T,n,k,fr[N],ifr[N],f[N],g[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int main()
{
	fr[0]=ifr[0]=1;for(int i=1;i<=60;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&n,&k);
		if(n<=k)
		{
			int as=1ll*fr[k]*ifr[k-n]%mod;
			for(int i=1;i<n;i++)as=(as+1ll*fr[k]*ifr[k-i]%mod*i%mod*fr[k-1]%mod*ifr[k-1-(n-i-1)])%mod;
			printf("%d\n",(as+mod)%mod);
		}
		else
		{
			int as=0;
			for(int i=0;i<=k;i++)f[i]=g[i]=0;
			f[0]=g[0]=mod-1;
			for(int i=1;i<k;i++)for(int j=0;j<=i-1;j++)f[i]=(f[i]+1ll*f[j]*pw(fr[i-j],n/k)%mod*(mod-1))%mod;
			for(int i=1;i<k;i++)for(int j=0;j<=i-1;j++)g[i]=(g[i]+1ll*g[j]*pw(fr[i-j],n/k-1)%mod*(mod-1))%mod;
			int tp=n%k;
			for(int i=0;i<=tp;i++)
			for(int j=i;j<=tp;j++)
			as=(as+1ll*pw(fr[k+i-j],(n-i)/k)%mod*ifr[k-j]%mod*fr[k-(tp-i)+(tp-j)]%mod*ifr[k-(tp-i)]%mod*f[j-i])%mod;
			for(int i=tp+1;i<k;i++)
			for(int j=i;j<k;j++)
			as=(as+1ll*pw(fr[k+i-j],(n-i)/k)%mod*ifr[k-j]%mod*fr[k-(tp-i+k)+(tp-j+k)]%mod*ifr[k-(tp-i+k)]%mod*g[j-i])%mod;
			for(int i=0;i<=tp;i++)
			for(int j=i;j<=tp;j++)
			for(int s=tp+1;s<k;s++)
			for(int t=s;t<k;t++)
			as=(as+1ll*pw(fr[k+i-t],(n-i)/k)%mod*ifr[k-t]%mod*fr[k-(tp-s+k)+(tp-j)]%mod*ifr[k-(tp-s+k)]%mod*f[j-i]%mod*g[t-s]%mod*pw(fr[s-j],(n-s)/k))%mod;
			printf("%d\n",1ll*fr[k]*(mod-as)%mod);
		}
	}
}
```

##### Day5 T3 String Cheese

###### Problem

给定长度为 $n$ 的串 $S$ ，你有一个串 $T$ ，初始是空串

你每次可以在 $T$ 的开头或结尾加入一个字符，这之后你会得到此时 $T$ 在 $S$ 中出现次数的分数

求最后分数的最大值

$n\leq 5\times 10^5$

$3s,1024MB$

###### Sol

考虑建SAM，那么向后加字符相当于SAM上DAG边的转移，向前加字符相当于后缀树上的转移

发现如果 $S$ 与 $xS$ 的出现次数相同，那么对于任意的字符串 $T$ ，$ST$ 的出现次数一定和 $xST$ 的出现次数相同

因为加字符出现次数不会变多，因此如果在前面加字符可以使得出现次数不变那么在前面加一定最优

因此如果当前字符串在SAM上某个点，则一定会一直向左加字符直到长度等于这个点的最长字符串长度

设 $dp_i$ 表示当前在点 $i$ ，当前串长等于这个点的最长字符串长度，当前的最大分数

转移时拓扑排序后同时转移两种边即可

复杂度 $O(n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<assert.h>
using namespace std;
#define N 1000500
#define ll long long
int in[N],tp[N],le[N],vl[N];
char s[N];
ll dp[N],as;
struct SAM{
	int ch[N][26],fail[N],len[N],las,ct;
	void ins(int t)
	{
		int s1=las,st=++ct;las=ct;len[st]=len[s1]+1;
		vl[st]=1;
		while(!ch[s1][t]&&s1)ch[s1][t]=st,s1=fail[s1];
		if(!ch[s1][t]){fail[st]=1;return;}
		int tp=ch[s1][t];
		if(len[tp]==len[s1]+1)fail[st]=tp;
		else
		{
			int cl=++ct;len[cl]=len[s1]+1;
			for(int i=0;i<26;i++)ch[cl][i]=ch[tp][i];
			fail[cl]=fail[tp];fail[tp]=fail[st]=cl;
			while(s1&&ch[s1][t]==tp)ch[s1][t]=cl,s1=fail[s1];
		}
	}
}sa;
bool cmp(int a,int b){return le[a]<le[b];}
int main()
{
	scanf("%s",s+1);sa.las=sa.ct=1;
	for(int i=1;s[i];i++)sa.ins(s[i]-'a');
	for(int i=1;i<=sa.ct;i++)tp[i]=i,le[i]=sa.len[i];
	sort(tp+1,tp+sa.ct+1,cmp);
	for(int i=sa.ct;i>0;i--)vl[sa.fail[tp[i]]]+=vl[tp[i]];
	for(int i=1;i<=sa.ct;i++)
	{
		int st=tp[i];
		dp[st]=max(dp[st],dp[sa.fail[st]]+1ll*vl[st]*(sa.len[st]-sa.len[sa.fail[st]]));as=max(as,dp[st]);
		for(int j=0;j<26;j++)if(sa.ch[st][j])dp[sa.ch[st][j]]=max(dp[sa.ch[st][j]],dp[st]+1ll*vl[sa.ch[st][j]]*(sa.len[sa.ch[st][j]]-sa.len[st]));
	}
	printf("%lld\n",as);
}
```

##### Day6 T1 Colorful Tree

###### Problem

给一棵树，你需要给每个点染上 $m$ 种颜色中的一种，使得每条边两边的颜色不同

给定点集 $S$  ，对于一种染色方案，设 $S$ 中的点一共被染了 $a$ 种不同的颜色，则这种方案的权值为 $a^k$

求所有合法染色方案的权值和，模 $998244353$

$n\leq 10^5,k\leq 80$

$2s,512MB$

###### Sol

考虑将 $a^k$ 拆成斯特林数的形式，只需要对于每个 $k$ 求出给定 $k$ 种颜色，对于每种给定的颜色， $S$ 中至少有一个点染了这个颜色的方案数

考虑容斥，变为对于每个 $a$ ，求出 $S$ 中的点不能染这 $a$ 种颜色的染色方案数

设 $dp_{i,0/1}$ 表示考虑 $i$ 的子树，$i$ 父亲染的是 $a$ 种颜色之一或不是 $a$ 种颜色之一，子树内的染色方案数

如果 $i$ 染的是 $a$ 种颜色之一(如果是 $S$ 中的点则不能染 $a$ 种颜色之一)，那么对 $dp$ 的贡献为

$dp_{i,0}+a\prod_{x\in son_i}dp_{x,1}->dp_{i,0}$

$dp_{i,1}+(a-1)\prod_{x\in son_i}dp_{x,1}->dp_{i,1}$

否则，对 $dp$ 的贡献为

$dp_{i,0}+(m-a-1)\prod_{x\in son_i}dp_{x,0}->dp_{i,0}$

$dp_{i,1}+(m-a)\prod_{x\in son_i}dp_{x,0}->dp_{i,1}$

复杂度 $O(nk+k^2)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 105000
#define K 95
#define mod 998244353
int n,m,k,d,f[K],g[K],s[K][K],c[K][K],head[N],cnt,is[N],a,b,fr[N],ifr[N],dp[N][2],vis[N][2];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int dfs(int u,int fa,int d,int k)
{
	if(vis[u][d])return dp[u][d];
	int as=0;
	if(!is[u])
	{
		int as1=k-d*(fa>0);if(as1<0)as1=0;
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)as1=1ll*as1*dfs(ed[i].t,u,1,k)%mod;
		as=(as+as1)%mod;
	}
	int as2=m-k-(1-d)*(fa>0);if(as2<0)as2=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)as2=1ll*as2*dfs(ed[i].t,u,0,k)%mod;
	as=(as+as2)%mod;
	vis[u][d]=1;
	return dp[u][d]=as;
}
int main()
{
	scanf("%d%d%d%d",&n,&m,&k,&d);
	for(int i=1;i<=d;i++)scanf("%d",&a),is[a]=1;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	for(int i=0;i<=k;i++)if(i<=m)
	memset(vis,0,sizeof(vis)),f[i]=dfs(1,0,0,i);
	s[0][0]=1;for(int i=1;i<=k;i++)for(int j=1;j<=k;j++)s[i][j]=(s[i-1][j-1]+1ll*s[i-1][j]*j)%mod;
	for(int i=0;i<=k;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=k;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	fr[0]=ifr[0]=1;for(int i=1;i<=m;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
	for(int i=1;i<=k;i++)
	for(int j=0;j<=i;j++)
	g[i]=(g[i]+1ll*(j&1?mod-1:1)*f[j]%mod*c[i][j])%mod;
	int as=0;for(int i=1;i<=k;i++)if(i<=m)as=(as+1ll*s[k][i]%mod*g[i]%mod*fr[m]%mod*ifr[m-i])%mod;
	printf("%d\n",as);
}
```

##### Day6 T2 2-cut

###### Problem

给一棵以1为根的有根树，然后再加入若干条边，满足后加入的边的两个端点在原树上LCA为1，得到一个图

定义一个边的集合为割，当且仅当

1. 边集合中正好包含两条树边，设这两条树边为 $(u_1,v_1),(u_2,v_2)$
2. 删去集合中所有边后， $u_1,v_1$ 不连通，$u_2,v_2$ 不连通

对于每一条树边，求包含这条树边的割的最小大小

$n\leq 5\times 10^4,m\leq 2\times 10^5$

$4s,64MB$

###### Sol

在树边 $(u,v)$ 的表示中，认为 $u$ 是 $v$ 的父亲

考虑两条树边的位置关系，如果两条树边中一条为另外一条的祖先，设 $(u_1,v_1)$ 是更靠近根的，那么因为所有非树边的LCA为1，所以删去两条树边后中间的部分连出的非树边必须删掉

设 $s_u$ 表示 $u$ 子树中连出的非树边数，那么删去 $(u_1,v_1),(u_2,v_2)$ 后还需要删去 $s_{v_1}-s_{v_2}$ 条边

因此这种情况一定是删去相邻的两条边，可以 $O(n)$ 求出这种情况的答案

如果两条边没有祖先关系，那么这时树分成了三部分： $v_1$ 子树， $v_2$ 子树和剩余的部分

前两部分间的边可以保留，但前两部分与第三部分的边必须删去

因此这样需要额外删去的边数为 $s_{v_1}+s_{v_2}-2*(两个端点分别在v_1,v_2子树中的非树边数)$

如果枚举 $v_1$ ，那么可以看成在所有 $v_1$ 子树内连出的非树边的另外一个端点做一个标记(一个点可以有多个标记)，然后最小化 $s_{v_2}-2*(v_2子树内标记数)$

可以看成对于每一条连出的边，在另外一个端点到根的路径上的所有点值-2，然后找一个最小值

考虑以 $s_u$ 作为子树大小树剖，从根开始dfs，dfs一个点时先dfs它的所有轻儿子，每dfs完一个儿子就撤销儿子中的所有修改，然后dfs重儿子，再加入重儿子外的所有修改

修改可以直接树剖+线段树，复杂度 $O(m\log m\log^2 n)$ ，能过，也可以SBT做到 $O(m\log m\log n)$ ，空间复杂度 $O(n+m)$

###### Code

```cpp
#include<cstdio>
#include<stack>
#include<set>
#include<algorithm>
using namespace std;
#define N 50050
#define M 3051000
int n,m,head[N],cnt,a,b,id[N],sz[N],sn[N],fid[N],as[N],as2[N],dp[N],f1[N],tp[N],lb[N],rb[N],ct;
multiset<int> st[N];
struct edge{int t,next,id;}ed[N*2];
void adde(int f,int t,int id){ed[++cnt]=(edge){t,head[f],id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],id};head[t]=cnt;}
void dfs1(int u,int fa){for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)fid[ed[i].t]=ed[i].id,dfs1(ed[i].t,u),sz[u]+=sz[ed[i].t],sn[u]=sz[sn[u]]<=sz[ed[i].t]?ed[i].t:sn[u];}
void dfs2(int u,int fa,int v){lb[u]=id[u]=++ct;f1[u]=fa;tp[u]=v;if(sn[u])dfs2(sn[u],u,v);for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs2(ed[i].t,u,ed[i].t);rb[u]=ct;}
void dfs3(int u,int fa)
{
	int mn1=1e9,fr1=0,mn2=1e9,fr2=0;
	dp[u]=sz[u];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs3(ed[i].t,u);dp[u]=min(dp[u],dp[ed[i].t]);
		if(mn1>dp[ed[i].t])fr2=fr1,mn2=mn1,mn1=dp[ed[i].t],fr1=ed[i].t;
		else if(mn2>dp[ed[i].t])fr2=ed[i].t,mn2=dp[ed[i].t];
	}
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		int tp=mn1;if(fr1==ed[i].t)tp=mn2;tp+=sz[ed[i].t];
		as[ed[i].t]=min(as[ed[i].t],tp);
		as[u]=min(as[u],sz[u]-sz[ed[i].t]);
		as[ed[i].t]=min(as[ed[i].t],sz[u]-sz[ed[i].t]);
	}
}
struct segt{
	struct node{int l,r,mn,lz;}e[N*4];
	void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);}
	void pushdown(int x){e[x<<1].mn+=e[x].lz;e[x<<1].lz+=e[x].lz;e[x<<1|1].mn+=e[x].lz;e[x<<1|1].lz+=e[x].lz;e[x].lz=0;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
	void modify1(int x,int s,int v){if(e[x].l==e[x].r){e[x].mn=v;return;}int mid=(e[x].l+e[x].r)>>1;if(mid>=s)modify1(x<<1,s,v);else modify1(x<<1|1,s,v);pushup(x);}
	void modify2(int x,int l,int r,int v){if(e[x].l==l&&e[x].r==r){e[x].mn+=v;e[x].lz+=v;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify2(x<<1,l,r,v);else if(mid<l)modify2(x<<1|1,l,r,v);else modify2(x<<1,l,mid,v),modify2(x<<1|1,mid+1,r,v);pushup(x);}
	int query(int x,int l,int r){if(l>r)return 1e9;if(e[x].l==l&&e[x].r==r)return e[x].mn;pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return min(query(x<<1,l,mid),query(x<<1|1,mid+1,r));}
}tr;
stack<pair<int,int> > st1;
void modify(int x,int v){while(x){tr.modify2(1,lb[tp[x]],lb[x],v);x=f1[tp[x]];}}
void doit(){while(!st1.empty())modify(st1.top().first,-st1.top().second),st1.pop();}
void modify2(int x,int v){modify(x,v);st1.push(make_pair(x,v));}
void dfs4(int u,int fa)
{
	if(!sn[u])
	{
		id[u]=u;
		for(multiset<int>::iterator it=st[u].begin();it!=st[u].end();it++)
		modify2(*it,-2);
	}
	else
	{
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs4(ed[i].t,u),doit();
		dfs4(sn[u],u);
		id[u]=id[sn[u]];
		for(multiset<int>::iterator it=st[u].begin();it!=st[u].end();it++)
		modify2(*it,-2),st[id[u]].insert(*it);
		for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])
		for(multiset<int>::iterator it=st[id[ed[i].t]].begin();it!=st[id[ed[i].t]].end();it++)
		modify2(*it,-2),st[id[u]].insert(*it);
        for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])st[id[ed[i].t]].clear();
	}
	int tp=min(tr.query(1,2,lb[u]-1),tr.query(1,rb[u]+1,n));
	as[u]=min(as[u],tp+sz[u]);
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)as[i]=1e7;
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b,i);
	for(int i=n;i<=m;i++)scanf("%d%d",&a,&b),sz[a]++,sz[b]++,st[a].insert(b),st[b].insert(a);
	dfs1(1,0);dfs2(1,0,1);for(int i=head[1];i;i=ed[i].next)dfs3(ed[i].t,1);
	tr.build(1,1,n);
	for(int i=2;i<=n;i++)tr.modify1(1,id[i],sz[i]);
	for(int i=head[1];i;i=ed[i].next)dfs4(ed[i].t,1),doit();
	for(int i=2;i<=n;i++)as2[fid[i]]=as[i];
	for(int i=1;i<n;i++)printf("%d ",as2[i]+2);
}
```

##### Day6 T3 Palindrome

###### Problem

定义一个字符串的权值为它的回文子串数量(相同子串出现多次算多次)

给一个字符串，字符集大小为 $k$ ，求将它重新排列后，得到的字符串的最大权值，以及有多少个不同的串能够达到这个权值，方案数对 $10^9+7$ 取模

$k\leq 80$

$2s,512MB$

###### Sol

一个回文串的开头结尾必定相同，设 $ct_i$ 表示第 $i$ 种字符的出现次数，回文串个数不会超过 $\sum \frac{ct_i(ct_i+1)}2$

将所有相同字符放在一起就可以得到这个结果，因此最大权值就是 $\sum \frac{ct_i(ct_i+1)}2$

取到最大值的串一定满足每一对相同字符直接都是回文串，如果有一种字符 `A` 出现了大于三次，显然每相邻两个 `A` 之间的字符排序后是相同的，如果有一个字符 `B` 在每两个相邻 `A` 中出现大于一次，这时字符串形如 `...A...B...B...A...B...B...A...` ，考虑第一个和第三个 `B` 之间， `A` , `B` 各出现一次，因此不可能数回文串

因此对于每一个出现次数大于2的字符 `C` ，相邻两个 `C` 之间每种字符只出现一次，因此最多有一个字符

于是如果不存在出现次数为2的字符，那么最后只可能有若干个形如 `ababab...ab` ，`ababab...aba` 的段组成，且对于一个这样的串， `a` , `b` 这两种字符都不能在其他地方出现

然后考虑出现次数为2的，首先可以让两个字符相邻，否则中间必定是一个回文串且它只能放在极长的 `ababab...aba` 的两侧，因此中间一定是一个完整的形如 `ababab...aba` 的串或者0/1个字符

可以发现只有出现次数相差不超过1的两种字符可以合并，将出现过的 $ct_i$ 从大到小排序，设 $dp_{a,b,c,d}$ 表示考虑了最大的 $k$ 种 $ct_i$ ，当前前面拼出了 $b$ 个 `ababab...aba` 的串， $c$ 个 `ababab...ab` 的串，出现次数为第 $k$ 种 $ct_i$ 的字符有几个需要和下一种 $ct_i$ 的字符配对，转移时枚举有多少种字符不合并，有多少个合并成 `ababab...ab` ，复杂度 $O(k^5)$

然后特殊处理出现1,2次的字符的使用情况即可

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 85
#define mod 1000000007
int n,a,ct[N],su[N],c[N][N],dp[N][N][N][N],v[N][2],c1,f1,f2,fr[N],g[N],p2[N],as,as2;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d",&a),ct[a]++;
	sort(ct+1,ct+80+1);
	for(int i=0;i<=80;i++)c[i][0]=c[i][i]=1;
	for(int i=2;i<=80;i++)for(int j=1;j<i;j++)c[i][j]=(c[i-1][j]+c[i-1][j-1])%mod;
	for(int i=1;i<=80;i++)as2+=ct[i]*(ct[i]+1)/2;
	for(int i=1;i<=80;i++)if(ct[i]==1)f1++,ct[i]=0;else if(ct[i]==2)f2++;
	for(int i=1;i<=80;i++)if(ct[i]!=ct[i-1])v[++c1][0]=ct[i],v[c1][1]=1;else v[c1][1]++;
	fr[0]=1;for(int i=1;i<=80;i++)fr[i]=1ll*fr[i-1]*i%mod;
	g[0]=1;for(int i=2;i<=80;i++)g[i]=1ll*g[i-2]*(i-1)%mod;
	p2[0]=1;for(int i=1;i<=80;i++)p2[i]=2*p2[i-1]%mod;
	dp[c1+1][0][0][0]=1;
	for(int i=c1;i>0;i--)
	for(int j=0;j<=80;j++)
	for(int k=0;k<=80;k++)
	for(int l=0;l<=80;l++)
	if(dp[i+1][j][k][l])
	{
		if(l&&(v[i][0]!=v[i+1][0]-1||v[i][1]<l))continue;
		int st=1ll*dp[i+1][j][k][l]*fr[l]%mod*c[v[i][1]][l]%mod;
		for(int s=0;s<=v[i][1]-l;s+=2)
		for(int t=0;t<=v[i][1]-l-s;t++)
		dp[i][j+(s>>1)+t+l][k+t+l][v[i][1]-l-s-t]=(dp[i][j+(s>>1)+t+l][k+t+l][v[i][1]-l-s-t]+1ll*st*g[s]%mod*c[v[i][1]-l][s]%mod*c[v[i][1]-l-s][t]%mod*p2[s>>1])%mod;
	}
	if(v[1][0]>2)
	{
		for(int j=0;j<=80;j++)
		for(int k=0;k<=80;k++)as=(as+1ll*dp[1][j][k][0]*fr[j+f1])%mod;
	}
	else
	{
		for(int j=0;j<=80;j++)
		for(int k=0;k<=80;k++)
		for(int l=0;l<=80;l++)
		if(dp[1][j][k][l])
		as=(as+1ll*dp[1][j][k][l]*fr[j+f1]%mod*(l?c[l+f1+k-1][l]:1)%mod*fr[l])%mod;
	}
	printf("%d %d\n",as2,as);
}
```

##### Day7 T1 K shortest path on tree

###### Problem

给定一棵带边权树，点 $i$ 的父亲在 $[1,i-1]$ 中随机，给定 $k$ ，求出对于每个点，所有点到它的距离中的第 $k$ 小值

$n\leq 2\times 10^5,v_i\leq 10^4$

$4s,512MB$

###### Sol

两点 $u,v$ 的距离为 $dis_u+dis_v-2*dis_{lca(u,v)}$ ，其中dis为点到根的距离

考虑 $u$ 从 $x$ 变换到 $x$ 的儿子 $s$ 时，哪些 $v$ 的 $dis_v-2*dis_{lca(u,v)}$ 会发生变化

可以发现，只有 $s$ 子树内的 $v$ 的 $dis_v-2*dis_{lca(u,v)}$ 会发生变化

因为树随机，所以所有点的子树大小之和为 $O(n\log n)$ 级别

因为值域不大，直接BIT维护即可

复杂度 $O(n\log n\log v)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 200500
#define M 6006000
int n,k,a,b,c,head[N],cnt,dep[N],as[N],tr[M];
vector<int> fu[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void insert(int x,int v){x+=3e6;for(int i=x;i<=6e6;i+=i&-i)tr[i]+=v;}
int getkth(int k){int as=0;for(int i=22;i>=0;i--)if(as+(1<<i)<=6000000&&tr[as+(1<<i)]<k)as+=1<<i,k-=tr[as];return as+1-3e6;}
void dfs1(int u,int fa)
{
	fu[u].push_back(u);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dep[ed[i].t]=dep[u]+ed[i].v;dfs1(ed[i].t,u);
		for(int j=0;j<fu[ed[i].t].size();j++)fu[u].push_back(fu[ed[i].t][j]);
	}
}
void dfs2(int u,int fa)
{
	for(int i=0;i<fu[u].size();i++)insert(dep[fu[u][i]]-2*dep[u],1);
	as[u]=getkth(k)+dep[u];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		for(int j=0;j<fu[ed[i].t].size();j++)insert(dep[fu[ed[i].t][j]]-2*dep[u],-1);
		dfs2(ed[i].t,u);
		for(int j=0;j<fu[ed[i].t].size();j++)insert(dep[fu[ed[i].t][j]]-2*dep[u],1);
	}
	for(int i=0;i<fu[u].size();i++)insert(dep[fu[u][i]]-2*dep[u],-1);
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c);
	dfs1(1,0);dfs2(1,0);for(int i=1;i<=n;i++)printf("%d\n",as[i]);
}
```

##### Day7 T2 Access

###### Problem

给一棵 $n$ 个点的有根树，你可以进行不超过 $k$ 次操作，每次选择一个点，对这个点进行LCT的access操作

求最后可能出现的不同虚实边情况的数量，模 $998244353$

$n\leq 10^4,k\leq 500$

$2s,512MB$

###### Sol

对于一个虚实边情况，考虑至少需要access多少个点

如果一个点 $u$ 到某个儿子的边是实边，那么如果access了 $u$ ，考虑最后一次access $u$ ，此时 $u$ 与儿子的边都是虚边，之后一定会access $u$ 子树内的某个点

但access $u$ 再access $u$ 的某个儿子和直接access $u$ 的某个儿子是等价的，因此没有必要access $u$

因此只有没有实儿子的点可能需要access

如果一个点 $u$ 没有实儿子，并且它到父亲的边是实边，如果没有access过 $u$ ，那么一定只可能access过 $u$ 子树内的某个点才可能使得它到父亲的边是实边，但这时 $u$ 一定存在一个实儿子，矛盾

因此如果一个点没有实儿子，并且它到父亲的边是实边，它必须被access

类似可以得到如果一个点没有实儿子，并且它子树内存在实边，它也必须被access

显然，如果一个点不满足上面两种情况，那么它子树内以及它到父亲的边都是虚边，因此不需要access

因此一个点被access当且仅当它没有实儿子，并且它到父亲的边是实边或它子树内存在实边

设 $f_{u,i}$ 表示如果 $u$ 到父亲的边是实边， $u$ 子树内有 $i$ 个点必须要access时， $u$ 子树内的虚实边情况数, $g_{u,i}$ 表示如果 $u$ 到父亲的边是虚边， $u$ 子树内有 $i$ 个点必须要access时， $u$ 子树内的虚实边情况数

转移时设 $s_{i,0/1}$ 表示考虑了前若干个儿子，前面的儿子中有 $i$ 个点必须要access，当前有没有一条到儿子的边为实边

转移一个儿子 $v$ 时有 

$s_{i,0}^{'}=\sum_{j=0}^is_{j,0}g_{v,i-j}$

$s_{i,1}^{'}=\sum_{j=0}^is_{j,0}f_{v,i-j}+s_{j,1}g_{v,i-j}$

如果 $u$ 到父亲的边是实边，那么只要 $u$ 没有虚儿子， $u$ 一定必须被access，因此有 $f_{u,i}=s_{i-1,0}+s_{i,1}$

否则，只有当 $u$ 子树内有另外一个点被access过时才需要access $u$ ，因此有 $g_{u,i}=s_{i-1,0}+s_{i,1}(i>1),g_{u,1}=s_{1,1},g_{u,0}=s_{0,0}$

因为 $f,g$ 的第二维不会超过对应点的子树大小，也不会超过 $k$ ，复杂度 $O(nk)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 10005
#define K 505
#define mod 998244353
int dp[N][K],f[K][2],g[K][2],sz[N],n,k,a,b,head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa)
{
	sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u),sz[u]+=sz[ed[i].t];
	for(int i=0;i<=k;i++)f[i][0]=f[i][1]=0;
	f[0][0]=1;int su=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		for(int j=0;j<=k;j++)g[j][0]=g[j][1]=0;
		for(int j=0;j<=su;j++)
		for(int l=0;j+l<=k&&l<=sz[ed[i].t];l++)
		{
			g[j+l][1]=(g[j+l][1]+1ll*f[j][0]*dp[ed[i].t][l]*(l>0)+1ll*f[j][1]*(dp[ed[i].t][l]-(l==1)))%mod;
			g[j+l][0]=(g[j+l][0]+1ll*f[j][0]*(dp[ed[i].t][l]-(l==1)))%mod;
		}
		su+=sz[ed[i].t];if(su>k)su=k;
		for(int j=0;j<=su;j++)f[j][0]=g[j][0],f[j][1]=g[j][1];
	}
	dp[u][0]=1;
	for(int i=1;i<=k;i++)dp[u][i]=(f[i-1][0]+f[i][1])%mod;
}
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs(1,0);
	int as=0;for(int i=1;i<=k;i++)as=(as+dp[1][i])%mod;printf("%d\n",as);
}
```

##### Day7 T3 XOR Problem

###### Problem

求 $\sum_{b_0=0}^{a_0}\sum_{b_1=0}^{a_1}\sum_{b_2=0}^{a_2}\sum_{b_3=0}^{a_3}\sum_{b_4=0}^{a_4}\sum_{b_5=0}^{a_5}max(|b_0-b_3|,|b_1-b_4|,|b_2-b_5|)\oplus b_0\oplus b_1\oplus b_2\oplus b_3\oplus b_4\oplus b_5$ ，对 $2^{64}$ 取模

$a_i\leq 3\times 10^4$

$3s,512MB$

###### Sol

考虑计算每一位的贡献

对于每一位，考虑对于 $b_0,b_3$ ，求出对于每个 $i$ ，所有 $|b_0-b_3|=i$ 的情况中，有多少对 $b_0,b_3$ 满足 $b_0\oplus b_3$ 在这一位上是0，有多少对在这一位上是1

这可以直接枚举 $b_0,b_3$ 在这一位上的值，然后做差卷积

同样的方式可以求出 $b_1,b_4$ ，$b_2,b_5$ 的结果

然后可以对于每一位算出对于每一个 $i$ ，max不超过 $i$ 的所有情况中 $b_0\oplus b_3$ 有多少种情况在这一位上是0，多少种情况在这一位上是1

于是可以对于每一位得到对于每一个 $i$ ，max不超过 $i$ 的所有情况中 $b_0\oplus b_1\oplus b_2\oplus b_3\oplus b_4\oplus b_5$  有多少种情况在这一位上是0，多少种情况在这一位上是1

然后容斥就得到了max等于 $i$ 的所有情况，然后就算出了每一位的贡献

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 70010
#define mod 998244353
#define ul unsigned long long
int v1,v2,v3,v4,v5,v6,a[N],b[N],c[N],d[N],rev[N],ntt[N],g2[2][N*2],f[N][17][2],g[N][17][2],h[N][17][2];
ul f1[N][2],g1[N][2],h1[N][2],t1[N][2],as;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*a*as%mod;a=1ll*a*a%mod;p>>=1;}return as;}
void init()
{
	for(int i=0;i<1<<16;i++)rev[i]=(rev[i>>1]>>1)|((i&1)<<15);
	for(int d=0;d<2;d++)
	for(int i=2;i<=1<<16;i<<=1)
	for(int j=0;j<i>>1;j++)
	g2[d][i+j]=pw(3,mod-1+(d?1:-1)*(mod-1)/i*j);
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[i]=a[rev[i]];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=j,st=i;k<j+(i>>1);k++,st++)
	{
		int v1=ntt[k],v2=1ll*ntt[k+(i>>1)]*g2[t][st]%mod;
		ntt[k]=(v1+v2)%mod,ntt[k+(i>>1)]=(v1-v2+mod)%mod;
	}
	int inv=pw(s,t?0:mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void doit(int x,int y,int s[][17][2])
{
	int l=1<<16;
	for(int i=0;i<16;i++)
	{
		for(int j=0;j<l;j++)a[j]=b[j]=c[j]=d[j]=0;
		for(int j=0;j<=x;j++)a[j]=(j>>i)&1,b[j]=1-a[j];
		for(int j=0;j<=y;j++)c[y-j]=(j>>i)&1,d[y-j]=1-c[y-j];
		dft(l,a,1);dft(l,b,1);dft(l,c,1),dft(l,d,1);
		for(int j=0;j<l;j++)
		{
			int v1=(1ll*a[j]*c[j]+1ll*b[j]*d[j])%mod,v2=(1ll*a[j]*d[j]+1ll*b[j]*c[j])%mod;
			a[j]=v1;b[j]=v2;
		}
		dft(l,a,0);dft(l,b,0);
		for(int j=0;j<l;j++)
		{
			int tp=y>j?y-j:j-y;
			s[tp][i][0]=s[tp][i][0]+a[j];
			s[tp][i][1]=s[tp][i][1]+b[j];
		}
	}
}
int main()
{
	scanf("%d%d%d%d%d%d",&v1,&v2,&v3,&v4,&v5,&v6);init();
	doit(v1,v4,f);doit(v2,v5,g);doit(v3,v6,h);
	for(int i=0;i<16;i++)
	{
		for(int t=0;t<2;t++)
		for(int j=0;j<1<<16;j++)
		{
			f1[j][t]=f[j][i][t];g1[j][t]=g[j][i][t];h1[j][t]=h[j][i][t];
			if(j)f1[j][t]+=f1[j-1][t],g1[j][t]+=g1[j-1][t],h1[j][t]+=h1[j-1][t];
		}
		for(int j=0;j<1<<16;j++)t1[j][0]=f1[j][0]*g1[j][0]*h1[j][0]+f1[j][1]*g1[j][1]*h1[j][0]+f1[j][1]*g1[j][0]*h1[j][1]+f1[j][0]*g1[j][1]*h1[j][1];
		for(int j=0;j<1<<16;j++)t1[j][1]=f1[j][1]*g1[j][1]*h1[j][1]+f1[j][0]*g1[j][0]*h1[j][1]+f1[j][0]*g1[j][1]*h1[j][0]+f1[j][1]*g1[j][0]*h1[j][0];
		for(int j=(1<<16)-1;j>0;j--)t1[j][0]-=t1[j-1][0],t1[j][1]-=t1[j-1][1];
		for(int j=0;j<1<<16;j++)
		{
			int tp=(j>>i)&1;tp=1-tp;
			ul as1=t1[j][tp];
			as+=as1*(1<<i);
		}
	}
	printf("%llu\n",as);
}
```

##### Day8 T1 Nim

###### Problem

有 $n$ 堆石子，第 $i$ 堆石子个数在 $[l_i,r_i]$ 之间，两个人轮流操作，每次每个人可以选择不超过两堆石子，并从中拿走若干个石子(不能不拿)，不能操作的人输

求有多少种情况使得后手必胜，模 $998244353$

$n\leq 10,r_i< 2^{30}$

$2s,512MB$

###### Sol

由k-nim的结论可以得到后手必胜当且仅当对于每个二进制位，数量满足这一位是1的石子堆数是3的倍数

考虑暴力容斥，容斥后变为只有上界的限制，设 $dp_{i,S}$ 表示填了所有数的前 $i$ 位，填的前 $i$ 位与上界前 $i$ 位相同的数的集合为 $S$ 的方案数

转移可以枚举这一位还有限制的集合填的数，剩余没有限制的数填的方式可以dp出来，复杂度 $O(6^n\log v)$ 

也可以每一位上依次考虑每个数，设 $f_{i,j,S}$ 表示填到第 $j$ 个数的第 $i$ 为，前面填的与上界前面相同的数的集合为 $S$ 的方案数，复杂度 $O(3*n4^n\log v)$

注意到每个数只有两种上界 $r_i,l_i-1$ ，设 $dp_{i,S_1,S_2}$ 表示填了所有数的前 $i$ 位，填的前 $i$ 位与上界前 $i$ 位相同且为第一种上界的数的集合为 $S_1$ ，填的前 $i$ 位与上界前 $i$ 位相同且为第二种上界的数的集合为 $S_2$ 的方案数

转移时枚举 $S_1,S_2$ 内的数怎么填，剩下的一定没有限制 ，复杂度 $O(4^n\log v)$ ，可以过

也可以每一位依次考虑每个数，复杂度 $O(3*n3^n\log v)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 12
#define M 1050
#define K 33
#define mod 998244353
int dp[K][M][M],n,l[N],r[N],is[K][M][M],ct[M],ct2[M],dp1[N][4],f1[M][K],f2[M][K];
int dfs(int d,int s1,int s2)
{
	if(d==-1)return 1;
	if(is[d][s1][s2])return dp[d][s1][s2];
	long long as=0;
	int v1=f1[s1][d],v2=f2[s2][d];
	int las=ct[((1<<n)-1)^s1^s2];
	for(int i=v1;;i=(i-1)&v1)
	{
		for(int j=v2;;j=(j-1)&v2)
		{
			int f1=ct2[i|j];
			int tp=dp1[las][f1];
			as+=1ll*tp*dfs(d-1,s1^v1^i,s2^v2^j);
			if(!j)break;
		}
		if(!i)break;
	}
	as%=mod;
	is[d][s1][s2]=1;return dp[d][s1][s2]=as;
}
int main()
{
	scanf("%d",&n);
	for(int i=0;i<1<<n;i++)
	{
		int tp=0;
		for(int j=1;j<=n;j++)if(i&(1<<j-1))tp++;
		ct[i]=tp;ct2[i]=tp%3;
	}
	dp1[0][0]=1;
	for(int i=1;i<=n;i++)
	dp1[i][0]=dp1[i-1][0]+dp1[i-1][1],dp1[i][1]=dp1[i-1][1]+dp1[i-1][2],dp1[i][2]=dp1[i-1][2]+dp1[i-1][0];
	for(int i=1;i<=n;i++)scanf("%d%d",&l[i],&r[i]),l[i]--;
	for(int d=0;d<=30;d++)for(int s1=0;s1<1<<n;s1++)for(int i=1;i<=n;i++)if(s1&(1<<i-1))if((r[i]>>d)&1)f1[s1][d]|=1<<i-1;
	for(int d=0;d<=30;d++)for(int s1=0;s1<1<<n;s1++)for(int i=1;i<=n;i++)if(s1&(1<<i-1))if((l[i]>>d)&1)f2[s1][d]|=1<<i-1;
	int as=0;
	for(int i=0;i<1<<n;i++)
	{
		int st=((1<<n)-1)-i;
		for(int j=st;;j=(j-1)&st)
		{
			dp[1][i][j]=dfs(1,i,j);
			if(!j)break;
		}
	}
	for(int i=2;i<=30;i++)
	for(int s1=0;s1<1<<n;s1++)
	{
		int st=((1<<n)-1)-s1;
		for(int s2=st;;s2=(s2-1)&st)
		{
			long long as1=0;
			int v1=f1[s1][i],v2=f2[s2][i];
			int las=ct[((1<<n)-1)^s1^s2];
			for(int g1=v1;;g1=(g1-1)&v1)
			{
				for(int j=v2;;j=(j-1)&v2)
				{
					int f1=ct2[g1|j];
					int tp=dp1[las][f1];
					as1+=1ll*tp*dp[i-1][s1^v1^g1][s2^v2^j];
					if(!j)break;
				}
				if(!g1)break;
			}
			as1%=mod;dp[i][s1][s2]=as1;
			if(!s2)break;
		}
	}
	for(int i=0;i<1<<n;i++)
	{
		int st=1;
		for(int j=1;j<=n;j++)if(i&(1<<j-1)){st*=-1;if(l[j]==-1)st=0;}
		if(st==0)continue;
		st+=mod;
		as=(as+1ll*st*dp[30][((1<<n)-1)^i][i])%mod;
	}
	printf("%d\n",as);
}
```

##### Day8 T2 Subsequence

###### Problem

给定序列 $a_{1,...,n}$ ，求有多少个序列组 $b_1,...,b_k$ 满足

1. $b_1=a$
2. $b_i$ 是 $b_{i-1}$ 的非空子序列

$n,k\leq 100$

$2s,512MB$

###### Sol

考虑给每个 $[1,n]$ 的位置一个权值 $v_i$ ，$b_i$ 中只保留所有满足 $v_j\geq i$ 的位置 $j$ 

这样能够统计所有的 $b_1,...,b_k$ ，但是会算重，考虑什么情况下会重复计数

可以发现如果 $b_{i+1}$ 在 $b_i$ 中作为子序列出现多次就会算重，考虑在出现多次时，只计算第一次出现的子序列

考虑这种情况下哪些 $v_{1,...,n}$ 是合法的，对于一对 $b_i,b_{i+1}$ ，如果对于 $b_{i+1}$ 中相邻的两个位置 $s,t$ ，存在一个属于 $b_i$ 的位置 $x$ ，满足 $s<x<t$ ，且 $x$ 位置的值和 $t$ 位置的值相同，那么将 $t$ 换成 $x$ 可以得到更先出现的子序列，因此 $b_{i+1}$ 一定不是第一次出现的子序列

显然不存在满足上面条件的 $v$ 一定是合法的，因此合法条件为不存在 $i<j,a_i=a_j,c_i<c_j$ 且 $\forall i<k<j,c_k\leq c_i$ 

考虑序列中 $c_i=k$ 的位置中最小的 $d$ ，可以发现，如果上面的条件中 $i<d<j$ ，那么如果 $c_i<c_j$ ，一定有 $c_k>c_i$ ，因此不会有跨过中间的限制

对于一个最左侧的最大值位置 $d$ ，接下来只需要分别考虑 $[1,d-1],[d+1,n]$ 内部的限制，且 $[1,d-1]$ 部分判断合法时需要额外考虑 $d$  

设 $dp_{l,r,i}$ 表示区间 $[l,r]$ 中 $c_i$ 最大值为 $i$ ，且 $c_{r+1}>i$ ，$c_{r+1}$ 已经确定，使得 $[l,r+1]$ 内部合法的方案数

枚举最左侧的最大值 $d$ ，显然有 $a_d\neq a_{r+1}$ ，于是之后 $d$ 不会参与判断合法，，于是有 $dp_{l,r,i}=\sum_{j=l}^j[a_j\neq a_{r+1}](\sum_{s=1}^{i-1}dp_{l,j-1,s})(\sum_{s=1}^idp_{j+1,r,s})$

前缀和即可做到 $O(n^3k)$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define mod 998244353
int n,k,v[N],f[N][N][N],su[N][N][N];
int main()
{
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	for(int j=1;j<=n+1;j++)for(int i=1;i<=k;i++)f[j][j-1][i]=su[j][j-1][i]=1;
	for(int i=1;i<=n;i++)for(int j=i;j<=n;j++)
	{
		int fg=0;
		for(int s=i;s<=j;s++)if(v[s]==v[j+1])fg=1;
		su[i][j][1]=f[i][j][1]=!fg;
	}
	for(int t=2;t<=k;t++)
	{
		for(int le=0;le<=n;le++)
		for(int l=1;l+le<=n;l++)
		{
			int r=l+le;
			for(int s=l;s<=r;s++)if(v[s]!=v[r+1])
			f[l][r][t]=(f[l][r][t]+1ll*su[l][s-1][t-1]*su[s+1][r][t])%mod;
			su[l][r][t]=(su[l][r][t-1]+f[l][r][t])%mod;
		}
	}
	printf("%d\n",f[1][n][k]);
}
```

##### Day8 T3 K-best-subsequence

###### Problem

给一个长度为 $n$ 的序列 $a$ ，多组询问，每次给定 $l,r,k$ ，你需要在 $a_{l,...,r}$ 中选一个长度为 $k$ 的子序列，使得子序列相邻两项的和(包括开头和结尾的和)的最大值最小，求出最小值

$n,q\leq 10^5$

$5s,512MB$

###### Sol

考虑二分答案，如果存在长度大于 $k$ 的合法方案，那么删掉最大的一些数，剩下的方案一定合法

设当前二分的答案为 $s$ ，考虑先选所有不超过 $\lfloor\frac s2\rfloor$ 的数，然后每相邻两个数直接能插入就插入

考虑在这个方案上，如果删去一个不超过 $\lfloor\frac s2\rfloor$ 的数 ，那么左右两个不超过 $\lfloor\frac s2\rfloor$ 的数中间最多只能有一个其它的数，因此这样一定不会更优

因此上面的方案就是最优方案

考虑随着 $s$ 的增加，必选的数会不断变多，每插入一个数时，可以计算序列上新分出的两个段里面能够选出数需要的最小的 $s$ ，可以记录这部分所有的变化

对于一次询问，二分答案 $s$ ，可以找到这时区间中第一个和最后一个不超过 $\lfloor\frac s2\rfloor$ 的数，两者中间的就是序列上的情况，这部分看成一个数点问题，对于不在序列上的段，可以发现只有一个，暴力判断即可

复杂度 $O(q\log^2 n)$

###### Code

没写

##### Day9 T1 lxl题

###### Problem

给一棵有根树，点有点权，对于每个点，求出在这个点子树外选两个点(可以相同)，使得这两个点点权异或值最大，对于每个点求出最大值(选不出来则是0)

$n\leq 5\times 10^5,v\leq 10^{18}$

$5s,1024MB$

###### Sol

考虑找出异或值最大的一对 $(x,y)$

可以发现，除了 $x$ 到根的路径上的点和 $y$ 到根的路径上的点外，剩余的点的答案都是 $v_x\oplus v_y$ 

对于一个点到根的路径，考虑从根开始向下，维护trie和当前最大异或值，每向下一步就把新的在子树外的点加入trie并更新答案，这样做一次的复杂度为 $O(n\log v)$

因此总复杂度 $O(n\log v)$

###### Code

没写

##### Day9 T3 魔法题

###### Problem

给一个DAG，满足 $1,2,...,n$ 是一个合法的拓扑序，从1连出的边不超过 $d$ 条，对于每个 $i$ 求出 $1$ 到 $i$ 的最小割

$n\leq 10^5,m\leq 2\times 10^5,d\leq 10$

$1s,1024MB$

###### Sol

考虑这样一个做法：给从起点出发的每条边一个随机的 $d$ 维向量，对于每个点，它的每一条出边的权值为它所有入边进行随机线性组合的结果

对于每个点，它所有入边形成的线性基的大小就是它的最小割

考虑如果最小割是 $S$ ，那么 $S$ 将点集割成两部分，可以发现第二部分的边权都是这 $|S|$ 条边的向量进行线性组合得到的，因此线性基的大小不会超过 $|S|$

根据神秘的定理这东西的正确性很大.jpg

复杂度 $O(md^2)$

###### Code

没写

##### Day10 T3 lxl题2

###### Problem

给一个序列，支持如下操作：

1. 将一个区间中的数同时对某个数取gcd
2. 求区间和

$n\leq 2\times 10^5,q\leq 5\times 10^5,v_i\leq 10^{18}$

$4s,1024MB$

###### Sol

一个数取gcd后如果减小一定至少减半，因此数改变的的次数不超过 $O(n\log v)$ ，只需要找到这些数即可

对于一个区间，考虑这个区间的lcm $s$

设取gcd的数为 $x$ ，可以发现当 $s|x$ 时整个区间内的数取gcd后都不会变，否则一定有一个数会变

考虑线段树维护，因为 $x\leq 10^{18}$ ，如果区间的lcm超过 $10^{18}$ ，可以将其看成inf

线段树上最多会遍历 $O(n\log n\log v+q\log n)$ 个点，考虑pushup时更新lcm的复杂度

注意到实际上做的是 $O(n\log v+q)$ 次更新一个点到根的路径的lcm，可以看成一个数连续与多个数取lcm

考虑一般的lcm做法 $lcm(x,y)=\frac {xy}{gcd(x,y)}$ ，gcd每递归两次lcm就会乘2，因此实际上最多只会进行 $O(\log v)$ 次就会超过 $v$

复杂度 $O((n\log v+q)(\log n+\log v))$

###### Code

```cpp
#include<cstdio>
using namespace std;
#define N 205000
#define ll long long
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
int n,m,a,b,c;
ll d,inf=1000000000000000010ll,v[N];
ll lcm(ll a,ll b){if(a>=inf||b>=inf)return inf;ll tp=b/gcd(a,b);if((long double)a*tp>1e18)return inf;return a*tp;}
struct segt{
	struct node{int l,r;ll v;unsigned int su;}e[N*4];
	void pushup(int x){e[x].v=lcm(e[x<<1].v,e[x<<1|1].v);e[x].su=e[x<<1].su+e[x<<1|1].su;}
	void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(e[x].l==e[x].r){e[x].v=v[l];e[x].su=v[l];return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
	void modify(int x,int l,int r,ll vl){if(vl%e[x].v==0)return;if(e[x].l==e[x].r){e[x].v=gcd(e[x].v,vl);e[x].su=e[x].v;return;}int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify(x<<1,l,r,vl);else if(mid<l)modify(x<<1|1,l,r,vl);else modify(x<<1,l,mid,vl),modify(x<<1|1,mid+1,r,vl);pushup(x);}
	unsigned int query(int x,int l,int r){if(e[x].l==l&&e[x].r==r)return e[x].su;int mid=(e[x].l+e[x].r)>>1;if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);else return query(x<<1,l,mid)+query(x<<1|1,mid+1,r);}
}tr;
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]);
	tr.build(1,1,n);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d%d",&a,&b,&c);
		if(a==1)scanf("%lld",&d),tr.modify(1,b,c,d);
		else printf("%u\n",tr.query(1,b,c));
	}
}
```

~~完结撒花~~



