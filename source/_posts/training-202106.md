---
title: 2021/06 集训题解
date: '2021-07-30 18:58:42'
updated: '2021-07-30 18:58:42'
tags: Mildia
permalink: Rakuenzu/
description: 2021/06 南京集训
mathjax: true
---


### 0602~0617 集训

当我还有一大堆题解没写完然后又开了新坑.jpg

#### 6.2

##### T1 trident

###### Problem

给一张 $n$ 个点 $m$ 条边的简单图，每条边有一个 $\{0,1\}$ 的边权。

你可以选择三条不同的边，满足这三条边有一个公共端点。然后你可以将这三条边的边权翻转。

构造一种步数不超过 $10m$ 的方案使得所有边权变为 $0$，或输出无解。

$n\leq 2\times 10^5,m\leq 10^5$

$2s,512MB$

###### Sol

考虑只操作公共端点在某一个点上的边集的效果。

如果这个点度数不超过 $2$ ，则无法操作。

如果这个点度数为 $3$，则只有一种同时翻转三条边的操作。

如果度数大于等于 $4$ ，可以发现通过进行类似 $123,124,134$ 的操作，可以翻转一条边。

因此可以忽略所有端点中至少有一个度数大于等于 $4$ 的边，只考虑度数为 $3$ 的点的操作，最后再将这些边使用上面的方式处理。

因为每个三度点只能整体翻转，因此可以看成给每个三度点一个 $\{0,1\}$ 权值，要求所有两端点度数不超过 $3$ 的边边权等于端点权值的异或和。

因此每条边相当于限制两个点的点权相同或者相反，并查集/dfs染色即可。

最后构造方案可以直接构造，显然操作次数不超过 $3m$。

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 200500
int n,m,d[N],s[N][3],cl[N],head[N],cnt,fg,st[N*10][3],ct;
vector<int> nt[N];
struct edge{int t,next,v;}ed[N*4];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void dfs(int u)
{
	for(int i=head[u];i;i=ed[i].next)if(cl[ed[i].t]==-1)cl[ed[i].t]=cl[u]^ed[i].v,dfs(ed[i].t);
	else if(cl[ed[i].t]!=cl[u]^ed[i].v)fg=1;
}
int rd()
{
	int as=0;
	char c=getchar();
	while(c<'0'||c>'9')c=getchar();
	while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();
	return as;
}
void wt(int x)
{
	if(!x)return;
	int tp=x/10;
	wt(tp);putchar('0'+x-tp*10);
}
int main()
{
	freopen("trident.in","r",stdin);
	freopen("trident.out","w",stdout);
	n=rd();m=rd();
	for(int i=1;i<=n;i++)cl[i]=-1;
	for(int i=1;i<=m;i++)s[i][0]=rd(),s[i][1]=rd(),s[i][2]=rd(),nt[s[i][0]].push_back(i),nt[s[i][1]].push_back(i);
	for(int i=1;i<=n;i++)d[i]=nt[i].size();
	for(int i=1;i<=m;i++)
	{
		if(d[s[i][0]]>d[s[i][1]])s[i][0]^=s[i][1]^=s[i][0]^=s[i][1];
		if(d[s[i][1]]>3)continue;
		if(d[s[i][1]]<3)
		{
			if(!s[i][2])continue;
			printf("-1\n");return 0;
		}
		if(d[s[i][0]]==3)adde(s[i][0],s[i][1],s[i][2]);
		else adde(n+1,s[i][1],s[i][2]);
	}
	dfs(n+1);
	for(int i=1;i<=n;i++)if(cl[i]==-1)cl[i]=0,dfs(i);
	if(fg){printf("-1\n");return 0;}
	for(int i=1;i<=n;i++)if(d[i]==3&&cl[i])
	{
		ct++;
		for(int j=0;j<3;j++)st[ct][j]=nt[i][j],s[nt[i][j]][2]^=1;
	}
	for(int i=1;i<=m;i++)if(s[i][2])
	{
		if(d[s[i][1]]<4)return 998244353;
		int nw=i,s1=0,s2=0,s3=0;
		for(int j=0;!s3;j++)
		{
			int tp=nt[s[i][1]][j];
			if(tp!=i)if(!s1)s1=tp;else if(!s2)s2=tp;else s3=tp;
		}
		st[ct+1][0]=i;st[ct+1][1]=s1;st[ct+1][2]=s2;
		st[ct+2][0]=i;st[ct+2][1]=s1;st[ct+2][2]=s3;
		st[ct+3][0]=i;st[ct+3][1]=s2;st[ct+3][2]=s3;
		ct+=3;s[i][2]=0;
	}
	printf("%d\n",ct);for(int i=1;i<=ct;i++,printf("\n"))for(int j=0;j<3;j++)wt(st[i][j]),putchar(' ');
}
```

##### T2 sa

###### Problem

定义字符集为 $\N^+$。定义一个字符串是好的，当且仅当：

设出现的最大字符为 $x$，则字符 $1,2,...,x$ 都出现过。

给一个长度为 $n$ 的排列 $p$，求有多少个好的字符串满足字符串的后缀数组为 $p$，答案摸 $998244353$

$p$ 以如下方式给出：

初始 $p$ 为 $\{1,2,...,n\}$，然后给出 $m$ 次操作：每次操作为：

1. 选一个区间，将这个区间平移到最开头。
2. 选一个区间，翻转区间。

$n\leq 10^9,m\leq 10^5$

$1s,512MB$

###### Sol

设字符串为 $s$，显然 $s_{sa_1}\leq s_{sa_2}\leq...\leq s_{sa_n}$。

根据题目的限制，显然 $s_{sa_{i}}\geq s_{sa_{i-1}}+1$，因此只需要考虑所有不等式是否取等。

考虑相邻两个位置，如果 $s_{sa_i}<s_{sa_{i+1}}$，则显然这两个后缀满足字典序的顺序。

否则，如果 $s_{sa_i}=s_{sa_{i+1}}$，则比较两个后缀的字典序大小时会接着比较 $s[sa_i+1,n],s[sa_{i+1}+1,n]$。这可以在SA上找到这两个后缀的顺序。

因此满足要求当且仅当对于所有 $i$，以下两者至少有一个被满足：

1. $s_{sa_i}<s_{sa_{i+1}}$
2. $rk_{sa_i+1}\leq rk_{sa_{i+1}+1}$（这里认为 $rk_{n+1}=0$）

可以发现此时满足条件 $2$ 的位置等号可以任意，剩余的位置必须不取等号。

因此设 $k$ 为满足 $rk_{sa_i+1}\leq rk_{sa_{i+1}+1}$ 的位置个数，则答案为 $2^k$。

考虑使用Splay维护当前所有连续的段，记录每一段的值域以及是否翻转，每次操作前分裂需要的段，剩余部分为基础splay操作。

最后对于每一个值域连续的段内部，可以发现除去 $sa$ 最大的一个位置外，其余位置的 $rk_{sa_i+1}$ 也在这一段内，可以发现这样一个连续的段内部除去最大的位置后其余位置一定满足 $rk_{sa_i+1}\leq rk_{sa_{i+1}+1}$。

对于每个段最大的位置以及两段中间的位置，这样的位置只有 $O(m)$ 个，可以每次二分找到一个位置的 $rk$。

复杂度 $O(m\log m)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
using namespace std;
#define N 300500
#define mod 998244353
int n,q,a,b,c,as,ct,st[N],cnt,s[N][3],su[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int rt,fa[N],ch[N][2],sz[N],lz[N],lb[N],rb[N],fg[N];
set<pair<int,int> > fu;
void doit(int x){lz[x]^=1;fg[x]^=1;swap(ch[x][0],ch[x][1]);}
void pushdown(int x){if(lz[x])doit(ch[x][0]),doit(ch[x][1]),lz[x]=0;}
void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+rb[x]-lb[x]+1;}
void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
void Splay(int x,int y=0)
{
	int c1=0,tp=x;
	while(tp!=y)st[++c1]=tp,tp=fa[tp];
	if(y)pushdown(y);while(c1)pushdown(st[c1--]);
	while(fa[x]!=y)
	{
		int f=fa[x],g=fa[f];
		if(g!=y)rotate((ch[g][1]==x)^(ch[x][1]==f)?x:f);
		rotate(x);
	}
	if(!y)rt=x;
}
int kth(int x,int k)
{
	pushdown(x);
	if(k<=sz[ch[x][0]])return kth(ch[x][0],k);
	else if(k<=sz[ch[x][0]]+rb[x]-lb[x]+1)return x;
	return kth(ch[x][1],k-sz[ch[x][0]]-(rb[x]-lb[x]+1));
}
int pre(int x)
{
	Splay(x);
	int tp=ch[x][0];
	while(1)
	{
		pushdown(tp);
		if(!ch[tp][1])break;
		tp=ch[tp][1];
	}
	return tp;
}
int nxt(int x)
{
	Splay(x);
	int tp=ch[x][1];
	while(1)
	{
		pushdown(tp);
		if(!ch[tp][0])break;
		tp=ch[tp][0];
	}
	return tp;
}
void dfs(int x)
{
	if(!x)return;pushdown(x);dfs(ch[x][0]);
	if(lb[x]<=rb[x])
	{
		++cnt;
		s[cnt][0]=lb[x];s[cnt][1]=rb[x];s[cnt][2]=fg[x];
	}
	dfs(ch[x][1]);
}
int getrk(int k)
{
	if(k>n)return 0;
	int tp=(*(--fu.lower_bound(make_pair(k,cnt+1)))).second;
	if(s[tp][2])return su[tp-1]+(s[tp][1]-k+1);
	else return su[tp-1]+(k-s[tp][0]+1);
}
int main()
{
	freopen("sa.in","r",stdin);
	freopen("sa.out","w",stdout);
	scanf("%d%d",&n,&q);
	ct=3;rt=1;lb[1]=1;rb[1]=n;sz[1]=n;lb[2]=lb[3]=1;
	ch[1][0]=2;ch[1][1]=3;fa[2]=fa[3]=1;
	while(q--)
	{
		scanf("%d%d%d",&a,&b,&c);
		int tp=kth(rt,b);
		Splay(tp);
		if(sz[ch[tp][0]]!=b-1)
		{
			int cl=++ct,ls=b-1-sz[ch[tp][0]];fg[cl]=fg[tp];sz[cl]=ls;
			if(fg[tp])rb[cl]=rb[tp],lb[cl]=rb[cl]-ls+1,rb[tp]=rb[cl]-ls;
			else lb[cl]=lb[tp],rb[cl]=lb[cl]+ls-1,lb[tp]=lb[cl]+ls;
			int st=pre(tp);Splay(st,tp);
			ch[st][1]=cl;fa[cl]=st;pushup(st);pushup(tp);
		}
		int tp2=kth(rt,c);
		Splay(tp2);
		if(sz[ch[tp2][1]]!=n-c)
		{
			int cl=++ct,ls=n-c-sz[ch[tp2][1]];fg[cl]=fg[tp2];sz[cl]=ls;
			if(!fg[tp2])rb[cl]=rb[tp2],lb[cl]=rb[cl]-ls+1,rb[tp2]=rb[cl]-ls;
			else lb[cl]=lb[tp2],rb[cl]=lb[cl]+ls-1,lb[tp2]=lb[cl]+ls;
			int st=nxt(tp2);Splay(st,tp2);
			ch[st][0]=cl;fa[cl]=st;pushup(st);pushup(tp2);
		}
		int lb1=pre(tp),rb1=nxt(tp2);
		Splay(lb1);Splay(rb1,lb1);
		int st=ch[rb1][0];
		if(a==0)
		{
			ch[rb1][0]=0;fa[st]=0;pushup(rb1);pushup(lb1);
			int t1=nxt(2);
			Splay(t1,2);ch[t1][0]=st;fa[st]=t1;pushup(t1);pushup(2);Splay(st);
		}
		else doit(st);
	}
	dfs(rt);
	for(int i=1;i<=cnt;i++)fu.insert(make_pair(s[i][0],i)),su[i]=su[i-1]+s[i][1]-s[i][0]+1;
	for(int i=1;i<=cnt;i++)if(s[i][0]<s[i][1])
	{
		as+=s[i][1]-s[i][0]-1;
		as+=s[i][2]^(getrk(s[i][1])<getrk(s[i][1]+1));
	}
	for(int i=1;i<cnt;i++)
	{
		int v1=s[i][!s[i][2]],v2=s[i+1][s[i+1][2]];
		as+=getrk(v1+1)<getrk(v2+1);
	}
	printf("%d\n",pw(2,as));
}
```

##### T3 mahjong

###### Problem

有大小为 $1,2,...,n$ 的牌，每种大小的牌有 $12$ 张，其中有三种花色 $1,2,3$，每种花色各有 $4$ 张。

定义一个大小为 $3$ 的牌的集合为面子，当且仅当三张牌花色相同，且大小相同或者大小构成 $i,i+1,i+2$ 的形式。

定义一个大小为 $2$ 的牌的集合为对子，当且仅当两张牌大小花色都相同。

定义一个牌的集合是好的，当且仅当能在集合中划分出 $4$ 个面子加上 $1$ 个对子。

定义一个牌的集合在只考虑花色 $i,j$ 时是好的，当且仅当只考虑这些花色的牌时它们是好的。

你初始有 $13$ 张牌，剩余 $12n-13$ 张牌会以随机顺序依次出现。

你会在当前出现的牌加上你的牌组成的集合满足某个条件时停止。对于以下四种条件，求出你停止的期望时间，模 $998244353$：

1. 集合在只考虑花色 $2,3$ 时是好的。
2. 集合在只考虑花色 $1,3$ 时是好的。
3. 集合在只考虑花色 $1,2$ 时是好的。
4. 集合满足上面三个条件中的至少一个。

$n\leq 40$

$5s,1024MB$

因为数据问题，实际的第 $4$ 个条件为：

集合在考虑花色 $1,2,3$ 时是好的。

###### Sol

因为转移没有自环，根据期望线性性，可以计算每一个还不满足条件的状态乘上这个状态出现概率的和，这个值即为答案。

显然不同花色间独立，考虑计算一个花色的情况。

假设确定了状态中每种大小牌的数量，考虑计算最后能组成多少个集合。

考虑一个 $dp$：设 $dp_{i,j,k,0/1}$ 表示当前考虑了前 $i$ 种大小，当前最后两种大小的牌还有 $j,k$ 张没有配对，当前有没有配对对子时，能最多配多少个面子。

考虑 $dp$ 的不同状态数。通过dfs一次可以发现，能达到的不同状态数只有 $934$ 种。

设 $f_{i,j,k}$ 表示考虑了前 $i$ 种大小，当前前面额外选了 $j$ 张牌，当前前面的 $dp$ 状态为 $k$，前面排列的方案数和。

此时可以求出一种花色的情况。对于一种花色，可以发现只需要知道对于这种花色，在这个花色中选/不选对子时最多能配多少个面子。设这样的值为 $x,y$（可能出现 $y=-\infty$），可以发现这样的状态只有 $30$ 个。

对于前三问，考虑枚举钦定的两部分选的数量以及两部分的状态。此时剩下的部分相当于给出 $a,b$ ，求出随机一个排列，期望有多少个前缀满足前缀中正好出现了给定的 $a$ 个元素中的 $b$ 个。这个可以预处理解决。复杂度 $O((30*4n)^2)$

对于最后一问，因为只有三种花色，可以先枚举前两个的状态，可以算出只用这两个状态是否已经满足条件。接下来枚举第三个状态时，需要知道这个状态和之前两个状态中的任意一个合并是否一定不满足条件。设前两个状态为 $(x_1,y_1),(x_2,y_2)$，则可以发现把它们在之后的判断中和 $(\max(x_1,x_2),\max(y_1,y_2))$ 等价。因此可以再做一个类似的dp即可。对于修改后的问题，改成 $(x_1+x_2,\max(x_2+y_1,x_1+y_2))$ 即可。

复杂度 $O(n^2)$ ~~常数大概是1e4级别的~~

###### Code

改第四问只需要改 `calcnt`

```cpp
#include<cstdio>
#include<cstring>
#include<map>
#include<algorithm>
using namespace std;
#define ll long long
#define mod 998244353
//Mahjong Automaton/cy
int ct,trans[1001][5],su[3][41],dp[41][161][1001],vl[3][161][31],n,a,b,fr[485],ifr[485],st1[1001],is2[31][31],su1[3];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct sth{int dp[2][5][5];sth(){for(int t=0;t<2;t++)for(int i=0;i<5;i++)for(int j=i;j<5;j++)dp[t][i][j]=-1;}}fu[1001];
map<ll,int> mp;
ll calchash(sth a)
{
	if(a.dp[1][0][0]==4)return -1;
	ll val=0;
	for(int t=0;t<2;t++)
	for(int i=0;i<5;i++)
	for(int j=i;j<5;j++)
	val=(val*7+a.dp[t][i][j]+1)%1000000000000000003ll;
	return val;
}
sth calctrans(sth a,int b)
{
	sth tp;
	for(int s=0;s<2;s++)
	for(int v=0;v<=b-2*s;v++)
	for(int t=0;s+t<2;t++)
	for(int i=0;i<5;i++)
	for(int j=i;j<5;j++)
	{
		int ls=a.dp[t][i][j],mn=min(v,i);
		if(ls==-1)continue;
		for(int p=0;p<=mn;p++)
		for(int q=0;q*3+p<=v;q++)
		{
			int nw=min(ls+p+q,4),va=j-p,vb=v-p-3*q;
			if(va>vb)va=vb;
			tp.dp[s+t][va][vb]=max(tp.dp[s+t][va][vb],nw);
		}
	}
	return tp;
}
void init_automaton()
{
	sth st;st.dp[0][0][0]=0;
	mp[calchash(st)]=1;fu[1]=st;
	st.dp[1][0][0]=st.dp[0][0][0]=4;
	mp[calchash(st)]=2;fu[2]=st;ct=2;
	for(int i=1;i<=ct;i++)
	for(int j=0;j<=4;j++)
	{
		sth tp=calctrans(fu[i],j);
		ll vl=calchash(tp);
		if(!mp[vl])mp[vl]=++ct,fu[ct]=tp;
		trans[i][j]=mp[vl];
	}
	for(int i=1;i<=ct;i++)
	{
		int mx1=fu[i].dp[0][0][0],mx2=fu[i].dp[1][0][0]+1;
		st1[i]=mx2*5+mx1;
	}
}
bool check2(int a,int b)
{
	if(is2[a][b])return is2[a][b]-1;
	int a1=a/5-1,a2=a%5,b1=b/5-1,b2=b%5;
	if(a1!=-1&&a1+b2>=4){is2[a][b]=2;return 1;}
	if(b1!=-1&&b1+a2>=4){is2[a][b]=2;return 1;}
	is2[a][b]=1;return 0;
}
int calcnt(int a,int b)
{
	int a1=a/5-1,a2=a%5,b1=b/5-1,b2=b%5;
	if(a1!=-1)a1=min(4,a1+b2);
	if(b1!=-1)b1=min(4,b1+a2);
	a2=min(4,a2+b2);
	if(a1<b1)a1=b1;if(a2<b2)a2=b2;
	a1++;return a1*5+a2;
}
int solve(int a,int b)
{
	int as=0;
	int vl2[405]={0},s1=su1[a]+su1[b],s2=su1[3-a-b];
	for(int i=0;i<=s1;i++)
	for(int j=0;j<=s2;j++)
	vl2[i]=(vl2[i]+1ll*fr[i+j]*ifr[i]%mod*ifr[j]%mod*fr[s1+s2-i-j]%mod*ifr[s1-i]%mod*ifr[s2-j]%mod*fr[s2])%mod;
	for(int j=0;j<30;j++)
	for(int l=0;l<30;l++)
	if(!check2(j,l))
	for(int i=0;i<=su1[a];i++)
	for(int k=0;k<=su1[b];k++)
	as=(as+1ll*vl[a][i][j]*vl[b][k][l]%mod*fr[i+k]%mod*ifr[i]%mod*ifr[k]%mod*fr[s1-i-k]%mod*vl2[i+k])%mod;
	as=1ll*as*ifr[12*n-13]%mod;
	return as;
}
int dp2[405][31];
int solve2()
{
	int s2=12*n-13;
	for(int j=0;j<30;j++)
	for(int l=0;l<30;l++)
	if(!check2(j,l))
	{
		int nt=calcnt(j,l);
		for(int i=0;i<=4*n;i++)
		for(int k=0;k<=4*n;k++)
		dp2[i+k][nt]=(dp2[i+k][nt]+1ll*vl[0][i][j]*vl[1][k][l]%mod*fr[i+k]%mod*ifr[i]%mod*ifr[k])%mod;
	}
	int as=0;
	for(int j=0;j<30;j++)
	for(int l=0;l<30;l++)
	if(!check2(j,l))
	for(int i=0;i<=su1[0]+su1[1];i++)
	for(int k=0;k<=su1[2];k++)
	as=(as+1ll*dp2[i][j]*vl[2][k][l]%mod*fr[i+k]%mod*ifr[i]%mod*ifr[k]%mod*fr[s2-i-k])%mod;
	as=1ll*as*ifr[s2]%mod;
	return as;
}
int main()
{
	freopen("mahjong.in","r",stdin);
	freopen("mahjong.out","w",stdout);
	init_automaton();
	fr[0]=1;for(int i=1;i<=480;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[480]=pw(fr[480],mod-2);for(int i=480;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	scanf("%d",&n);for(int i=0;i<3;i++)su1[i]=4*n;
	for(int i=1;i<=13;i++)scanf("%d%d",&a,&b),su[a-1][b]++,su1[a-1]--;
	for(int t=0;t<3;t++)
	{
		memset(dp,0,sizeof(dp));
		dp[0][0][1]=1;
		for(int i=1;i<=n;i++)
		for(int j=0;j<=(i-1)*4;j++)
		for(int k=1;k<=ct;k++)if(dp[i-1][j][k])
		for(int s=0;s<=4-su[t][i];s++)
		{
			int nt=trans[k][s+su[t][i]],v1=1ll*fr[j+s]*ifr[s]%mod*ifr[j]%mod*fr[4-su[t][i]]%mod*ifr[4-su[t][i]-s]%mod;
			dp[i][j+s][nt]=(dp[i][j+s][nt]+1ll*dp[i-1][j][k]*v1)%mod;
		}
		for(int j=0;j<=n*4;j++)for(int k=1;k<=ct;k++)vl[t][j][st1[k]]=(vl[t][j][st1[k]]+dp[n][j][k])%mod;
	}
	printf("%d\n%d\n%d\n",solve(1,2),solve(0,2),solve(0,1));
	printf("%d\n",solve2());
}
```

#### 6.5

##### 0604T2 string

###### Problem

给 $n$ 个长度为 $m$ 的串 $t_1,t_2,...,t_n$，字符集大小为 $k$。

给出 $p_1,...,p_k$，定义 $f(S)$ 为：

当前字符串为 $S$，每次操作向 $S$ 末尾加入一个字符。以 $p_1$ 的概率加入 $1$，$p_2$ 的概率加入 $2$，...

$f(S)$ 为使得存在一个 $t_i$ 在 $S$ 中出现的操作次数期望。

给出长度为 $R$ 的字符串 $s$。对于所有 $i$ 求出 $f(s[1,i])$。

$n,m\leq 100,R\leq 10^4$

$1s,1024MB$

###### Sol

考虑[SDOI2017]硬币游戏的做法。首先考虑空串的 $f$。

设 $f_{i,j}$ 表示在第 $i$ 次操作后满足了条件，且此时字符串以 $t_j$ 结尾的概率。$g_i$ 表示在第 $i$ 次操作后还没有满足条件的概率。

考虑计算长度为 $i+m$，以 $t_j$ 结尾，且在前 $i$ 次操作后没有满足条件的字符串数量。显然这个东西等于 $g_i*\prod_{k=1}^mpr_{t_{j,k}}$

考虑另外一种方法计算，枚举实际上满足条件的位置 $i+k$ 以及它以哪个字符串 $t_l$ 结束，考虑这种情况的概率。

因为字符串以 $t_j$ 结尾，显然需要满足 $t_l[m-k+1,m]=t_j[1,k]$。

填好前面的概率即为 $g_{i+k,l}$，后面的字符已经固定，因此这部分为 $\prod_{k'=k+1}^mpr_{t_{j,k'}}$

因此这样计算的概率即为:
$$
\sum_{k=1}^m\sum_{l=1}^n[t_l[m-k+1,m]=t_j[1,k]]g_{i+k,l}\prod_{k'=k+1}^mpr_{t_{j,k'}}
$$
因此这东西和上面那个相等。

考虑把所有的 $f_{i,j},g_i$ 看成概率生成函数：$F_j(x)=\sum x^if_{i,j},G(x)=\sum x^ig_i$，则上面的式子可以看成：
$$
G(x)x^m\prod_{k=1}^mpr_{t_{j,k}}=\sum_{k=1}^m\sum_{l=1}^n[t_l[m-k+1,m]=t_j[1,k]]F_l(x)x^{m-k}\prod_{k'=k+1}^mpr_{t_{j,k'}}
$$

因为期望停止次数显然是 $\sum g_i=G(1)$，考虑直接带入 $x=1$，有：
$$
G(1)\prod_{k=1}^mpr_{t_{j,k}}=\sum_{k=1}^m\sum_{l=1}^n[t_l[m-k+1,m]=t_j[1,k]]F_l(1)\prod_{k'=k+1}^mpr_{t_{j,k'}}
$$
这样有 $n$ 个方程，又因为显然有 $\sum_{l=1}^nF_l(1)=1$，因此可以消元求出 $G(1)$

考虑初始不为空串的情况，设初始串为 $s$，长度为 $le$。只考虑 $s$ 不会使得操作立刻结束的情况。

考虑上面的式子，等式两边都相当于计算所有操作了 $i+m$ 次，以 $t_j$ 结尾，且在前 $i$ 次操作后没有满足条件的字符串数量。

但由于初始串的存在，可能在前 $m$ 次操作中就已经以 $t_j$ 结尾。考虑此时不限定 $i\geq 0$，则可以发现求和后等式右边不变，而因为 $g$ 是从 $0$ 开始计算，因此左侧不会考虑前 $m$ 次操作中就已经以 $t_j$ 结尾。此时将这部分算上，则有：
$$
G(x)x^m\prod_{k=1}^mpr_{t_{j,k}}+\sum_{i=1}^m[s[le-i+1,le]=t_j[1,i]]x^{m-i}\prod_{k'=i+1}^mpr_{t_{j,k'}}=\sum_{k=1}^m\sum_{l=1}^n[t_l[m-k+1,m]=t_j[1,k]]F_l(x)x^{m-k}\prod_{k'=k+1}^mpr_{t_{j,k'}}
$$
带入之后有：
$$
G(1)\prod_{k=1}^mpr_{t_{j,k}}+\sum_{i=1}^m[s[le-i+1,le]=t_j[1,i]]\prod_{k'=i+1}^mpr_{t_{j,k'}}=\sum_{k=1}^m\sum_{l=1}^n[t_l[m-k+1,m]=t_j[1,k]]F_l(1)\prod_{k'=k+1}^mpr_{t_{j,k'}}
$$
可以发现，这相当于在原来的方程中，只改变每个方程的常数项，求新的解。

考虑将方程的常数项分别看成 $n+1$ 个变量 $x_1,...,x_{n+1}$，然后把这些变量看成常数做消元。

可以发现，做完之后，可以得到 $G(1)$ 通过 $x_1,...,x_{n+1}$ 的线性表示。(相当于加一个伴随矩阵做消元)

最后暴力算出每个常数项即可。判断字符串相等可以hash。

复杂度 $O(n^3+n^2m+nmR)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 105
#define M 10050
#define mod 1000000007
int n,m,k,p[N],su[N][M],f[N][N*2],vl[N][M],pr[M],v1[M];
char s[N][M],r[M];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int calc(int x,int l,int r){return (vl[x][r]-1ll*vl[x][l-1]*pr[r-l+1]%mod+mod)%mod;}
int calc(int l,int r){return (v1[r]-1ll*v1[l-1]*pr[r-l+1]%mod+mod)%mod;}
void solve(int n)
{
	for(int i=1;i<=n;i++)
	{
		int st=i;
		for(int j=n;j>=i;j--)if(f[j][i])st=j;
		for(int j=1;j<=n*2;j++)swap(f[st][j],f[i][j]);
		for(int j=1;j<=n;j++)if(i!=j)
		{
			int tp=1ll*(mod-1)*f[j][i]%mod*pw(f[i][i],mod-2)%mod;
			for(int k=1;k<=n*2;k++)f[j][k]=(f[j][k]+1ll*f[i][k]*tp)%mod;
		}
	}
	for(int j=1;j<=n;j++)for(int k=n+1;k<=n*2;k++)f[j][k]=1ll*f[j][k]*pw(f[j][j],mod-2)%mod;
}
int main()
{
	freopen("string.in","r",stdin);
	freopen("string.out","w",stdout);
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=k;i++)scanf("%d",&p[i]),p[i]=1ll*p[i]*pw(100,mod-2)%mod;
	pr[0]=1;for(int i=1;i<=10000;i++)pr[i]=1ll*pr[i-1]*131%mod;
	for(int i=1;i<=n;i++)
	{
		scanf("%s",s[i]+1);
		su[i][m+1]=1;
		for(int j=m;j>=1;j--)su[i][j]=1ll*su[i][j+1]*p[s[i][j]-'a'+1]%mod;
		for(int j=1;j<=m;j++)vl[i][j]=(1ll*vl[i][j-1]*131+s[i][j]-'a'+2)%mod;
	}
	for(int i=1;i<=n;i++)
	{
		f[i][n+1]=mod-su[i][1];
		for(int j=1;j<=n;j++)
		for(int l=1;l<=m;l++)
		if(calc(i,1,l)==calc(j,m-l+1,m))
		f[i][j]=(f[i][j]+su[i][l+1])%mod;
	}
	for(int i=1;i<=n;i++)f[n+1][i]=1;
	for(int i=1;i<=n+1;i++)f[i][n+1+i]=1;
	solve(n+1);
	scanf("%s",r+1);
	for(int i=1;r[i];i++)
	{
		v1[i]=(1ll*v1[i-1]*131+r[i]-'a'+2)%mod;
		int as=f[n+1][(n+1)*2]+i;
		for(int j=1;j<=n;j++)
		for(int l=1;l<=m&&l<=i;l++)if(calc(j,1,l)==calc(i-l+1,i))as=(as+1ll*su[j][l+1]*f[n+1][n+1+j])%mod;
		printf("%d\n",as);
	}
}
```

##### DS1: [Ynoi2015] 世上最幸福的女孩

###### Problem

整体加，区间最大子段和。

$n,q\leq 3\times 10^5$，可能出现的结果不超过 $10^{18}$。

$1s,128MB$

###### Sol

设整体加的数为 $k$。

考虑一个区间的最大后缀和，它一定形如 $\max_{i=0}^nsu_i+k*i$，可以发现如果将 $k$ 值看成横坐标，最大值看成纵坐标，则它一定是一个凸壳的形式。可以发现区间的最大后缀和同理。

对于区间最大子段和，设 $f_i$ 表示在没有整体加的时候，长度为 $i$ 的最大子段和，则答案形如 $\max_{i=0}^nf_i+k*i$。因此这个和上面类似。

考虑线段树处理询问。只需要知道在线段树上分出的区间中，每个区间当前的最大前缀/后缀/子段和，即可知道答案。因此只需要求出线段树上每个点上面的凸壳。

最大前/后缀和的凸壳可以直接求，考虑求最大子段和的凸壳。显然最大子段和有两种情况：

1. 某个儿子的最大子段和
2. 左儿子的最大后缀和+右儿子的最大前缀和

第一种情况直接把两个凸壳并起来即可。对于第二种情况，考虑 $k$ 从小到大扫，记录最大值位置的变化即可（实际上相当于求凸壳的闵可夫斯基和）

如果在凸壳上二分，则总复杂度为 $O(n\log^2 n)$，但可以将询问离线，按照询问时加的 $k$ 从小到大排序。此时询问的 $k$ 一定单调不降，因此一个凸壳上的最优点一定从左侧向右侧单调移动。维护每个凸壳的决策点即可。复杂度 $O(n\log n)$

但直接这样做的空间复杂度为 $O(n\log n)$，不使用技巧优化会MLE。下面是一个~~神必~~做法。

通过上面的过程可以知道，在 $k$ 从小到大变化的过程中，所有线段树上区间的最大前缀/后缀/子段和的最优决策点的变化次数是 $O(n\log n)$ 次。显然，一个点的最优决策点一定由儿子的最优决策点转移来。考虑记录当前线段树上每个点的最大前缀/后缀/子段和的最优决策点，以及在儿子的最优决策点不变的情况下， $k$ 至少需要多大才会导致这个点的决策变化。

维护每个点子树中发生变化的最小时刻，在每次询问之前，线段树上遍历当前所有需要变化的点并更新即可。

这样的空间复杂度为 $O(n)$，但因为改变一个点的决策需要在线段树上找到这个点，直接的实现复杂度为 $O(n\log^2 n)$。

但实际上这个复杂度不满，使用一定的常数优化后可以卡过去。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 300500
#define ll long long
int n,q,a,b,c,ct,id[N*2],s[N*2][2];
ll v[N],su[N],nw,mn,as1[N*2],ti,as,tp[N*2];
double inv[N];
bool cmp(int a,int b){return tp[a]<tp[b];}
struct node{int lf,rf,v1;ll v2,ti,st;}e[N*4];
void pushup(int x)
{
	ll as=1e16;
	int l1=e[x<<1].lf,l2=e[x<<1|1].lf;
	if(su[l1-1]+l1*ti<su[l2-1]+l2*ti)e[x].lf=l1;else e[x].lf=l2,as=min(as,(ll)((su[l1-1]-su[l2-1])*inv[l2-l1]));
	int r1=e[x<<1].rf,r2=e[x<<1|1].rf;
	if(r2*ti+su[r2]>r1*ti+su[r1])e[x].rf=r2;else e[x].rf=r1,as=min(as,(ll)((su[r1]-su[r2])*inv[r2-r1]));
	ll f1=r2-l1+1,f2=su[r2]-su[l1-1];
	e[x].v1=f1,e[x].v2=f2;
	for(int i=(x<<1);i<=(x<<1|1);i++)
	{
		ll s1=e[i].v1,s2=e[i].v2;
		if(s1*ti+s2>e[x].v1*ti+e[x].v2)e[x].v1=s1,e[x].v2=s2;
	}
	for(int i=(x<<1);i<=(x<<1|1);i++)if(e[i].v1>e[x].v1&&e[i].v2<e[x].v2)as=min(as,(ll)((e[x].v2-e[i].v2)*inv[e[i].v1-e[x].v1]));
	if(e[x].v1<f1)as=min(as,(ll)((e[x].v2-f2)*inv[f1-e[x].v1]));
	e[x].ti=as;e[x].st=min(e[x].ti,min(e[x<<1].st,e[x<<1|1].st));
}
void build(int x,int l,int r)
{
	if(l==r){e[x].lf=e[x].rf=l;e[x].v1=1;e[x].v2=v[l];e[x].ti=e[x].st=1e16;return;}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	pushup(x);
}
void doit(int x)
{
	if(e[x<<1].st<=ti)doit(x<<1);
	if(e[x<<1|1].st<=ti)doit(x<<1|1);
	pushup(x);
}
void solve(int x,int l,int r,int l1,int r1)
{
	if(l1==l&&r1==r)
	{
		as=max(as,e[x].v1*ti+e[x].v2);
		as=max(as,e[x].rf*ti+su[e[x].rf]-mn);
		mn=min(mn,e[x].lf*ti-ti+su[e[x].lf-1]);
		return;
	}
	int mid=(l1+r1)>>1;
	if(mid>=r)solve(x<<1,l,r,l1,mid);
	else if(mid<l)solve(x<<1|1,l,r,mid+1,r1);
	else solve(x<<1,l,mid,l1,mid),solve(x<<1|1,mid+1,r,mid+1,r1);
}
ll rd()
{
	ll as=0;int fg=1;
	char c='%';
	while(c<'0'||c>'9'){c=getchar();if(c=='-')fg=-1;}
	while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();
	return as*fg;
}
void wt(ll x)
{
	if(!x)putchar('0');
	char st[21]={0},ct=0;
	while(x)st[++ct]=x%10+'0',x/=10;
	for(int i=ct;i>=1;i--)putchar(st[i]);
}
int main()
{
	n=rd();q=rd();
	for(int i=1;i<=n;i++)v[i]=rd(),inv[i]=1.0/i;
	for(int i=1;i<=q;i++)
	{
		a=rd();
		if(a==1)b=rd(),nw+=b;
		else b=rd(),c=rd(),ct++,s[ct][0]=b,s[ct][1]=c,mn=min(mn,nw),id[ct]=ct,tp[ct]=nw;
	}
	for(int i=1;i<=n;i++)v[i]+=mn,su[i]=su[i-1]+v[i];
	for(int i=1;i<=ct;i++)tp[i]-=mn;
	sort(id+1,id+ct+1,cmp);
	build(1,1,n);
	for(int i=1;i<=ct;i++)
	{
		ti=tp[id[i]],doit(1);
		as=0,mn=1e18;
		solve(1,s[id[i]][0],s[id[i]][1],1,n);
		as1[id[i]]=as;
	}
	for(int i=1;i<=ct;i++)wt(as1[i]),putchar('\n');
}
```

##### DS3: [Ynoi2008] rdCcot

###### Problem

给一棵树以及一个 $C$，定义两个点是C-连通的当且仅当两个点树上距离不超过 $C$。

定义两个点 $x,y$ 属于一个C-块，当且仅当存在 $v_1,v_2,...,v_k$ ，满足序列 $x,v_1,...,v_k,y$ 满足相邻两个点C-连通。

$q$ 组询问，每次给出 $l,r$ ，求出只考虑 $[l,r]$ 中的点时，C-块的数量。

$n\leq 3\times 10^5,q\leq 6\times 10^5$

$2s,512MB$

###### Sol

首先有如下结论：

对于一棵树上深度最大的一个点 $u$ 以及两个点 $a,b$ ，如果 $dis(u,a)\leq C,dis(u,b)\leq C$，则 $dis(a,b)\leq C$。

容易证明这个结论。考虑任取一个点为根做bfs，记录bfs序。如果按照bfs序加入每个点，则可以发现在加入一个点时，在只考虑已经加入的点时，这个点在树上的深度是当前最大的。

因此，对于一个C-块，设它的所有点按照bfs序排序后为 $v_1,...,v_m$，则根据上面的结论，对于 $i<j<k$，如果 $v_i,v_k$ C-连通，且 $v_j,v_k$ C-连通，则 $v_i,v_j$ C-连通。

此时可以发现，对于 $i>1$ 的 $v_i$，一定满足存在 $j<i$，使得 $v_i,v_j$ C-连通，否则可以说明这个C-块不 C-连通。

因此，C-块的数量即为满足如下条件的点 $x$ 数量：

不存在点 $i$ 满足 $i$ 的bfs序小于 $x$，且 $dis(i,x)\leq C$

显然只保留编号在一个区间内的点，上面的结论仍然成立。

因为每次询问都是一个区间，因此考虑求出如下值：

满足 $i$ 的bfs序小于 $x$，且 $dis(i,x)\leq C$ 并且 $i<x$ 的点中，编号最大的点 $l_x$

满足 $i$ 的bfs序小于 $x$，且 $dis(i,x)\leq C$ 并且 $i>x$ 的点中，编号最小的点 $r_x$

对于一个询问 $[l,r]$，可以发现 $x$ 会对答案造成贡献当且仅当 $l\in(l_x,x]$ 且 $r\in[x,r_x)$。因此求出所有 $l_x,r_x$ 后，只需要做二维数点即可对于所有询问求出答案。

考虑点分树，按照bfs序依次加入每个点，则在点分树的一个点上可以看成如下操作：

1. 给出一个点 $u$ 以及它到点分中心的距离 $ds_u$，加入这个点。
2. 给出 $k,x$，求出已经加入的，满足 $ds_u\leq k$ 的点中，满足 $u<x$ 的最大 $u$ 以及满足 $u>x$ 的最小 $u$。

维护一个线段树/平衡树，对于询问在线段树上二分即可。~~用平衡树可能被卡常~~

复杂度 $O(n\log^2 n+m\log n)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<queue>
#include<set>
using namespace std;
#define N 300500
int n,q,c,a,b,ct,head[N],cnt,qu[N],vis[N],fr[22][N],ds[22][N],as[N*2],sz[N],mn1,st,t1,s[N][2];
struct modify{int l,r,t;};
struct query{int x,id;};
vector<modify> s1[N];
vector<query> s2[N];
int tr[N];
void add(int x,int k){for(int i=x;i<=n;i+=i&-i)tr[i]+=k;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs0(int u,int d,int f)
{
	fr[d][u]=f;sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t]&&!fr[d][ed[i].t])ds[d][ed[i].t]=ds[d][u]+1,dfs0(ed[i].t,d,f),sz[u]+=sz[ed[i].t];
}
void dfs1(int u,int fa)
{
	int mx=0;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&!vis[ed[i].t])dfs1(ed[i].t,u),mx=max(mx,sz[ed[i].t]);
	mx=max(mx,t1-sz[u]);
	if(mn1>mx)mn1=mx,st=u;
}
void dfs2(int u,int d)
{
	vis[u]=1;dfs0(u,d,u);
	for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])mn1=1e7,t1=sz[ed[i].t],dfs1(ed[i].t,u),dfs2(st,d+1);
}
struct sth{int v1,v2,x;};
vector<sth> tp[N];
void bfs()
{
	for(int i=1;i<=n;i++)vis[i]=0;
	queue<int> st;
	st.push(1);
	while(!st.empty())
	{
		int u=st.front();st.pop();
		qu[++ct]=u;vis[u]=1;
		for(int i=head[u];i;i=ed[i].next)if(!vis[ed[i].t])st.push(ed[i].t);
	}
}
int ch[N][2],fa[N],id[N],vl[N],mn[N],rt,c1;
void pushup(int x){mn[x]=min(vl[x],min(mn[ch[x][0]],mn[ch[x][1]]));}
void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;mn[x]=mn[f];pushup(f);}
void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(g!=y)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);rotate(x);}if(!y)rt=x;}
int main()
{
	scanf("%d%d%d",&n,&q,&c);
	for(int i=2;i<=n;i++)scanf("%d",&a),adde(i,a);
	dfs0(1,0,1);
	mn1=1e9;t1=n;dfs1(1,0);
	dfs2(st,1);bfs();
	for(int i=1;i<=n;i++)
	{
		int u=qu[i];
		s[u][0]=0;s[u][1]=n+1;
		for(int j=21;j>=1;j--)if(fr[j][u]&&ds[j][u]<=c)
		tp[fr[j][u]].push_back((sth){c-ds[j][u],ds[j][u],u});
	}
	mn[0]=1e9;
	for(int i=1;i<=n;i++)
	{
		if(tp[i].size()<=c)
		{
			set<int> fu;
			fu.insert(0);fu.insert(n+1);
			for(int j=0;j<tp[i].size();j++)
			{
				set<int>::iterator it=fu.upper_bound(tp[i][j].x);
				s[tp[i][j].x][1]=min(s[tp[i][j].x][1],*it);
				it--;
				s[tp[i][j].x][0]=max(s[tp[i][j].x][0],*it);
				fu.insert(tp[i][j].x);
			}
			continue;
		}
		for(int j=1;j<=c1;j++)ch[j][0]=ch[j][1]=fa[j]=id[j]=vl[j]=mn[j]=0;
		c1=2;rt=1;
		ch[1][1]=2;fa[2]=1;id[2]=n+1;
		for(int j=0;j<tp[i].size();j++)
		{
			int s1=rt,as=0,ls=0;
			while(s1)
			{
				ls=s1;
				if(id[s1]<tp[i][j].x)as=s1,s1=ch[s1][1];
				else s1=ch[s1][0];
			}
			splay(ls);
			s1=as;
			splay(s1);
			int s2=ch[s1][1];while(ch[s2][0])s2=ch[s2][0];
			splay(s2,s1);
			ch[s2][0]=++c1;fa[c1]=s2;id[c1]=tp[i][j].x;vl[c1]=mn[c1]=tp[i][j].v2;
			splay(c1);
			int l1=s1,l2=s2,v1=0,v2=n+1;
			s1=ch[c1][0];
			while(s1)
			{
				l1=s1;
				if(vl[s1]<=tp[i][j].v1)v1=id[s1];
				if(mn[ch[s1][1]]<=tp[i][j].v1)s1=ch[s1][1];else if(v1>=id[s1])break;else s1=ch[s1][0];
			}
			s2=ch[c1][1];
			while(s2)
			{
				l2=s2;
				if(vl[s2]<=tp[i][j].v1)v2=id[s2];
				if(mn[ch[s2][0]]<=tp[i][j].v1)s2=ch[s2][0];else if(v2<=id[s2])break;else s2=ch[s2][1];
			}
			splay(l2);splay(l1);
			s[tp[i][j].x][0]=max(s[tp[i][j].x][0],v1);
			s[tp[i][j].x][1]=min(s[tp[i][j].x][1],v2);
		}
	}
	for(int i=1;i<=n;i++)s1[s[i][0]+1].push_back((modify){i,s[i][1]-1,1}),s1[i+1].push_back((modify){i,s[i][1]-1,-1});
	for(int i=1;i<=q;i++)scanf("%d%d",&a,&b),s2[a].push_back((query){b,i});
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<s1[i].size();j++)add(s1[i][j].l,s1[i][j].t),add(s1[i][j].r+1,-s1[i][j].t);
		for(int j=0;j<s2[i].size();j++)as[s2[i][j].id]=que(s2[i][j].x);
	}
	for(int i=1;i<=q;i++)printf("%d\n",as[i]);
}
```

#### 6.6

##### T1 tree

###### Problem

给一棵以 $1$ 为根的有根树，每个点有点权 $v_i$。

你需要将一些点放入集合 $A$，剩下的点放入集合 $B$。记 $asc(i,j)$ 表示 $i$ 是不是 $j$ 的祖先，则你需要最小化：
$$
\sum_{i,j\in A,i\neq j}[asc(i,j)=1,v_i>v_j]+[asc(i,j)=0,asc(j,i)=0,i<j]+\\
\sum_{i,j\in B,i\neq j}[asc(i,j)=1,v_i<v_j]+\sum_{i\in A}d_i
$$
其中 $d_i$ 为点的深度。

对于每个 $k$ ，你需要求出 $|B|=k$ 时，上面这个值的最小值。

$n,v_i\leq 5\times 10^5$

$2s,1024MB$

###### Sol

考虑第一个求和，用 $\frac{|A|(|A|-1)}2$ 减去这里面的东西，则原式等于：
$$
\frac{|A|(|A|-1)}2+\sum_{i,j\in B,i\neq j}[asc(i,j)=1,v_i<v_j]-\sum_{i,j\in A,i\neq j}[asc(i,j)=1,v_i\leq v_j]+\sum_{i\in A}d_i
$$
考虑在固定的 $A,B$ 下，将一个元素 $x$ 从 $A$ 移动到 $B$ 时上式的变化。不考虑上面的 $\frac{|A|(|A|-1)}2$，则变化量为：
$$
\sum_{i\in B}[asc(i,x)=1,v_i<v_x]+[asc(x,i)=1,v_x<v_i]\\+\sum_{i\in A,i\neq x}[asc(i,x)=1,v_i\leq v_x]+[asc(x,i)=1,v_x\leq v_i]-d_x\\
=\sum_{i}[asc(i,x)=1,v_i<v_x]+[asc(x,i)=1,v_x<v_i]-d_x\\+\sum_{i\in A,i\neq x}[asc(i,x)=1,v_i=v_x]+[asc(x,i)=1,v_x=v_i]
$$
除去最后部分外的值只与 $x$ 有关，最后一部分当且仅当有多个 $v_i$ 相同时会出现，且影响的只有存在祖先关系的点。因此在权值不同的时候，可以求出每个点从 $A$ 移动到 $B$ 时答案的变化量，然后排序后加入即可。

考虑多个 $v_i$ 相同时的先后顺序。如果两个点不是祖先关系则不用考虑，只需要考虑一对值相同的 $i,j$ 满足 $i$ 是 $j$ 的祖先的情况。

此时显然满足 $asc(i,x)=1||asc(x,i)=1$ 的点集显然包含 $asc(i,y)=1||asc(y,i)=1$ 的点集。因此在第二部分中一定 $y$ 增加的最少。

考虑第一部分，唯一的问题在于 $(i,j)$ 路径上的点 $x$ 在 $i$ 处有贡献当且仅当 $v_i<v_x$，而在 $j$ 处有贡献当且仅当 $v_j>v_x$。其余的点贡献系数不会降低。可以发现这样的点数量不超过 $dis(i,j)=d_j-d_i$ 个，因此此时一定先操作 $j$ 更优。

因此对于权值相同的点，移动一个点时它的子树内一定没有权值相同的点没有被移动，而它祖先内权值相同的点一定没有被移动。因此可以求出此时每个点从 $A$ 移动到 $B$ 时答案的变化量。可以发现按照变化量排序后一定儿子优于祖先，因此直接按照这个排序即可。

复杂度 $O(n\log n)$

###### Code

```cpp
// stO djq Orz 
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500500
#define ll long long
int n,m=5e5,a,b,v[N],head[N],cnt,as[N],dep[N];
ll tp;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
struct BIT{
	int tr[N];
	void add(int x,int v){for(int i=x;i<=m;i+=i&-i)tr[i]+=v;}
	int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
}tr1,tr2;
void dfs(int u,int fa)
{
	tp-=tr1.que(v[u]);tp+=dep[u];
	as[u]+=tr1.que(v[u])-(tr2.que(m)-tr2.que(v[u]));
	tr1.add(v[u],1);tr2.add(v[u],1);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dep[ed[i].t]=dep[u]+1,dfs(ed[i].t,u);
	tr1.add(v[u],-1);
	as[u]+=tr2.que(m)-tr2.que(v[u]);
	as[u]-=dep[u];
}
int rd()
{
	int as=0;
	char c=getchar();
	while(c<'0'||c>'9')c=getchar();
	while(c>='0'&&c<='9')as=as*10+c-'0',c=getchar();
	return as;
}
void wt1(ll x)
{
	if(!x)return;
	ll tp=x/10;
	wt1(tp);putchar('0'+x-tp*10);
}
void wt(ll x)
{
	if(!x)putchar('0');
	else wt1(x);
	putchar('\n');
}
int main()
{
	freopen("tree.in","r",stdin);
	freopen("tree.out","w",stdout);
	n=rd();
	for(int i=1;i<=n;i++)v[i]=rd();
	for(int i=1;i<n;i++)a=rd(),b=rd(),adde(a,b);
	tp=1ll*n*(n-1)/2;dfs(1,0);
	sort(as+1,as+n+1);
	wt(tp);
	for(int i=1;i<=n;i++)tp=tp-(n-i)+as[i],wt(tp);
}
```

##### T2 travel

###### Problem

给一个 $n$ 个点 $m$ 条边的简单有向图，保证每个点最多在一个简单环内。

给定 $k$ ，求有多少个路径对 $(P_1,P_2)$ 满足：

1. 每个点至少被两个路径中的一个覆盖。
2. 每个点被两个路径覆盖的次数和不超过 $k$。

这里的路径可以不是简单路径，答案模 $998244353$

$n\leq 2000,m\leq 4000$

$1s,1024MB$

###### Sol

首先考虑图是DAG的情况。考虑DAG的拓扑序，显然两条路径都是拓扑序上的一个子序列。

为了方便，可以新建一个起点 $n+1$ 和一个终点 $n+2$，起点连向所有点，所有点连向终点。然后可以看成固定起点终点，走两条路径(注意判 $k=1$)。

此时可以设 $dp_{i,j,k}$ 表示考虑了拓扑序前 $i$ 个点，当前路径只考虑前 $i$ 个点这部分满足要求，且考虑前 $i$ 个点后两条路径的下一个点为 $j,k$ 的方案数。

转移时按照 $i$ 从小到大转移，可以直接枚举 $j,k$ 的出边。因为每个点至少被覆盖一次，因此只有 $\min(j,k)=i+1$ 的状态合法。

因此可以改为设 $dp_{i,j}$ 表示当前两条路径结尾在 $i,j$，拓扑序前 $\min(i,j)-1$ 个点都满足要求的方案数。转移同理。复杂度 $O(m^2)$

考虑原问题，条件限制相当于原图的每个强连通分量都是一个有向环。因此考虑一个环一个环的 $dp$。设 $dp_{i,j}$ 表示当前两条路径结尾在 $i,j$，当前路径还没有在环上走过，拓扑序小于这两个点所在环的环上都是合法的的方案数。

从一个环到下一个环的转移与上面同理，只需要考虑在一个环上的转移。(一个点不需要如下转移)

如果当前 $j,k$ 只有一个点在环上，不妨设其为点 $j$ 。考虑这个点在环上走的部分的结尾 $l$。可以发现，如果 $l$ 不是 $j$ 在环上的前一个点，则如果不绕圈则不能覆盖所有点，因此有 $k-1$ 种方案，否则可以发现正好有 $k$ 种方案，这部分转移可以做到 $O(n^2)$。

考虑两个点都在环上走的部分，不妨设环长为 $l$ ，所有点编号为 $0,1,...,l-1$。

假设起点为 $a,b$，终点为 $c,d$。则考虑把环分成 $[a,b-2],b-1,[b,a-2],a-1$ 四段，考虑 $c,d$ 在环上的哪一段，可以发现如果确定了 $c,d$ 所在的段，则方案数是固定的，只可能是 $\frac{k(k-1)}2,\frac{k(k-1)}2-1,\frac{k(k+1)}2,\frac{k(k+1)}2-1$ 中的一种（$k=1$ 特判）。

因此可以分出 $O(1)$ 个二维区间，每个区间内的转移系数相同。二维前缀和优化即可。实际上可以先整体加一个最小值，再加上大于最小值的部分，注意细节。

复杂度 $O(m^2)$

###### Code

```cpp
// stO djq Orz 
#include<cstdio>
#include<stack>
#include<algorithm>
using namespace std;
#define N 2050
#define mod 998244353
int n,m,k,head[N],cnt,dfn[N],low[N],scc[N],dp[N][N],s[N*2][2],s2[N*3][2],id[N],ct,c1,c2,c3,f1[N],sz[N];
struct edge{int t,next;}ed[N*3];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;}
stack<int> st;
void dfs(int u)
{
	st.push(u);dfn[u]=low[u]=++c1;
	for(int i=head[u];i;i=ed[i].next)if(!dfn[ed[i].t])dfs(ed[i].t),low[u]=min(low[u],low[ed[i].t]);
	else if(!scc[ed[i].t])low[u]=min(low[u],dfn[ed[i].t]);
	if(low[u]==dfn[u])
	{
		int s=-1;++c2;
		while(s!=u)s=st.top(),st.pop(),scc[s]=c2,sz[c2]++,f1[c2]=s;
	}
}
int vl[N][N],sl[N][N],sr[N][N],s1[N][N],as[N][N];
int main()
{
	freopen("travel.in","r",stdin);
	freopen("travel.out","w",stdout);
	scanf("%d%d%d",&n,&m,&k);
	if(k==0){printf("0\n");return 0;}
	for(int i=1;i<=m;i++)scanf("%d%d",&s[i][0],&s[i][1]),adde(s[i][0],s[i][1]);
	for(int i=1;i<=n;i++)if(!dfn[i])dfs(i);
	for(int i=c2;i>=1;i--)
	{
		int u=f1[i];
		while(!id[u])
		{
			id[u]=++ct;
			for(int i=head[u];i;i=ed[i].next)if(scc[ed[i].t]==scc[u]){u=ed[i].t;break;}
		}
	}
	for(int i=1;i<=m;i++)if(scc[s[i][0]]!=scc[s[i][1]])s2[++c3][0]=id[s[i][0]],s2[c3][1]=id[s[i][1]];
	for(int i=1;i<=n;i++)s2[++c3][0]=i,s2[c3][1]=n+1,head[i]=0;cnt=0;
	for(int i=1;i<=c3;i++)adde(s2[i][0],s2[i][1]);
	int ls=sz[c2],su=0;
	for(int i=1;i<=ls;i++)for(int j=1;j<=n+1;j++)dp[i][j]=dp[j][i]=1;
	sz[0]=1;
	for(int t=c2;t>=1;t--)
	{
		int lb=su+1,rb=su+sz[t];su+=sz[t];
		int nt=su+sz[t-1];
		for(int i=rb+1;i<=n+1;i++)
		{
			int su=0;
			for(int j=lb;j<=rb;j++)su=(su+dp[j][i])%mod;
			su=1ll*su*(k-1)*(rb>lb)%mod;
			for(int j=lb;j<=rb;j++)
			{
				int fu=dp[j==rb?lb:j+1][i];
				for(int k=head[j];k;k=ed[k].next)if(ed[k].t<=nt||i<=nt)dp[ed[k].t][i]=(dp[ed[k].t][i]+su+1ll*fu)%mod;
			}
		}
		for(int i=rb+1;i<=n+1;i++)
		{
			int su=0;
			for(int j=lb;j<=rb;j++)su=(su+dp[i][j])%mod;
			su=1ll*su*(k-1)*(rb>lb)%mod;
			for(int j=lb;j<=rb;j++)
			{
				int fu=dp[i][j==rb?lb:j+1];
				for(int k=head[j];k;k=ed[k].next)if(ed[k].t<=nt||i<=nt)dp[i][ed[k].t]=(dp[i][ed[k].t]+su+1ll*fu)%mod;
			}
		}
		int l=rb-lb+1,sp=0;
		for(int i=0;i<=l+1;i++)for(int j=0;j<=l+1;j++)s1[i][j]=vl[i][j]=sl[i][j]=sr[i][j]=as[i][j]=0;
		for(int i=1;i<=l;i++)
		{
			int v1=i==1?l:i-1,fu=dp[i+lb-1][i+lb-1];
			int s1=1ll*k*(k-1)/2%mod;if(rb==lb)s1=1;if(k==1)s1=1;
			sp=(sp+1ll*fu*(s1+mod-1))%mod;
			if(k>1)
			{
				sl[v1][1]=(sl[v1][1]+fu)%mod;sr[1][v1]=(sr[1][v1]+fu)%mod;
				vl[v1][v1]=(vl[v1][v1]+mod-fu)%mod;
			}
		}
		for(int i=1;i<=l;i++)
		for(int j=i+1;j<=l;j++)
		{
			int v1=i==1?l:i-1,v2=j==1?l:j-1,fu=dp[i+lb-1][j+lb-1];
			int se=1ll*k*(k-1)/2%mod;
			int st=1ll*k*fu%mod;
			vl[v2][v1]=(vl[v2][v1]+fu)%mod;
			if(k>1&&rb>lb)
			{
				sp=(sp+1ll*fu*(se+mod-1))%mod;
				s1[i][j]=(s1[i][j]+st)%mod;s1[j][j]=(s1[j][j]+mod-st)%mod;
				s1[i][1]=(s1[i][1]+st)%mod;s1[j][1]=(s1[j][1]+mod-st)%mod;
				s1[i][i]=(s1[i][i]+mod-st)%mod;s1[j][i]=(s1[j][i]+st)%mod;
				st=fu;
				s1[j][i]=(s1[j][i]+st)%mod;s1[j][j]=(s1[j][j]+mod-st)%mod;
				s1[1][i]=(s1[1][i]+st)%mod;s1[1][j]=(s1[1][j]+mod-st)%mod;
				s1[i][i]=(s1[i][i]+mod-st)%mod;s1[i][j]=(s1[i][j]+st)%mod;
				sr[j][v1]=(sr[j][v1]+st)%mod;sr[1][v1]=(sr[1][v1]+st)%mod;sr[i][v1]=(sr[i][v1]+mod-st)%mod;
				sl[v2][i]=(sl[v2][i]+st)%mod;sl[v2][j]=(sl[v2][j]+mod-st)%mod;
				sr[i][v2]=(sr[i][v2]+st)%mod;sr[j][v2]=(sr[j][v2]+mod-st)%mod;
				sl[v1][j]=(sl[v1][j]+st)%mod;sl[v1][1]=(sl[v1][1]+st)%mod;sl[v1][i]=(sl[v1][i]+mod-st)%mod;
				vl[v1][v1]=(vl[v1][v1]+mod-st)%mod;vl[v2][v2]=(vl[v2][v2]+mod-st)%mod;
			}
		}
		for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)s1[i][j]=(3ll*mod+s1[i][j]-s1[i-1][j-1]+s1[i][j-1]+s1[i-1][j])%mod,sl[i][j]=(sl[i][j]+sl[i][j-1])%mod,sr[i][j]=(sr[i][j]+sr[i-1][j])%mod,
		as[i][j]=(as[i][j]+1ll*s1[i][j]+sl[i][j]+sr[i][j]+vl[i][j])%mod;
		for(int i=0;i<=l+1;i++)for(int j=0;j<=l+1;j++)s1[i][j]=vl[i][j]=sl[i][j]=sr[i][j]=0;
		for(int i=1;i<=l;i++)
		for(int j=i+1;j<=l;j++)
		{
			int v1=i==1?l:i-1,v2=j==1?l:j-1,fu=dp[j+lb-1][i+lb-1];
			int se=1ll*k*(k-1)/2%mod;
			int st=1ll*k*fu%mod;
			vl[v2][v1]=(vl[v2][v1]+fu)%mod;
			if(k>1&&rb>lb)
			{
				sp=(sp+1ll*fu*(se+mod-1))%mod;
				s1[i][j]=(s1[i][j]+st)%mod;s1[j][j]=(s1[j][j]+mod-st)%mod;
				s1[i][1]=(s1[i][1]+st)%mod;s1[j][1]=(s1[j][1]+mod-st)%mod;
				s1[i][i]=(s1[i][i]+mod-st)%mod;s1[j][i]=(s1[j][i]+st)%mod;
				st=fu;
				s1[j][i]=(s1[j][i]+st)%mod;s1[j][j]=(s1[j][j]+mod-st)%mod;
				s1[1][i]=(s1[1][i]+st)%mod;s1[1][j]=(s1[1][j]+mod-st)%mod;
				s1[i][i]=(s1[i][i]+mod-st)%mod;s1[i][j]=(s1[i][j]+st)%mod;
				sr[j][v1]=(sr[j][v1]+st)%mod;sr[1][v1]=(sr[1][v1]+st)%mod;sr[i][v1]=(sr[i][v1]+mod-st)%mod;
				sl[v2][i]=(sl[v2][i]+st)%mod;sl[v2][j]=(sl[v2][j]+mod-st)%mod;
				sr[i][v2]=(sr[i][v2]+st)%mod;sr[j][v2]=(sr[j][v2]+mod-st)%mod;
				sl[v1][j]=(sl[v1][j]+st)%mod;sl[v1][1]=(sl[v1][1]+st)%mod;sl[v1][i]=(sl[v1][i]+mod-st)%mod;
				vl[v1][v1]=(vl[v1][v1]+mod-st)%mod;vl[v2][v2]=(vl[v2][v2]+mod-st)%mod;
			}
		}
		for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)s1[i][j]=(3ll*mod+s1[i][j]-s1[i-1][j-1]+s1[i][j-1]+s1[i-1][j])%mod,sl[i][j]=(sl[i][j]+sl[i][j-1])%mod,sr[i][j]=(sr[i][j]+sr[i-1][j])%mod,
		as[j][i]=(as[j][i]+1ll*s1[i][j]+sl[i][j]+sr[i][j]+vl[i][j]+sp)%mod;
		for(int i=1;i<=l;i++)for(int j=1;j<=l;j++)dp[i+lb-1][j+lb-1]=as[i][j];
		for(int i=lb;i<=rb;i++)for(int j=lb;j<=rb;j++)
		for(int p=head[i];p;p=ed[p].next)for(int q=head[j];q;q=ed[q].next)
		{
			int t1=ed[p].t,t2=ed[q].t;
			if(t1<=nt||t2<=nt)dp[t1][t2]=(dp[t1][t2]+dp[i][j])%mod;
		}
	}
	printf("%d\n",dp[n+1][n+1]);
}
```

##### T3 math

###### Problem

给一个正整数 $k$，定义 $p=k*10^{-5}$

定义一个正整数 $n$ 是好的，当且仅当将 $n$ 以十进制表示后(不考虑前导 $0$)，存在相邻三位的值递增。

求出最小的 $n$ ，满足 $1,2,...,n$ 中好的整数占比大于等于 $p$。

$p\leq 1-10^{-5}$

$2s,1024MB$

###### Sol

答案不超过 $10^{90}$。

可以看成一个好的数有一个权值 $1-p$，一个不好的数有一个权值 $-p$，求最小的前缀满足前缀和非负。

考虑数位 $dp$，因为判断是否是好的只和相邻三位有关，考虑设 $dp_{i,j,k},f_{i,j,k}$ 表示当前还剩最后 $i$ 位没有填，当前填了的最后两位为 $j,k$（如果这一位不存在可以看成 $9$），且当前前缀没有满足要求时，后面的 $10^i$ 个数排成一列，内部权值的最小前缀和以及权值和。

考虑转移，枚举下一位填的数，则可以将所有数分成 $10$ 段，每一段如果当前的前缀变为合法，则权值可以直接求出，否则这一段为一个子问题。使用之前的 $dp_{i-1}$ 即可。

最后考虑逐位确定答案，使用 $dp,f$ 即可判断一段内的最小前缀和。

$dp,f$ 的实现需要高精，可以维护所有数的二进制，负数用补码表示，可以发现补码表示后可以直接进行加法操作。

设答案大小为 $v$，则复杂度为 $O(10^2*\log^2 v)$

###### Code

```cpp
// stO djq Orz 
#include<cstdio>
#include<algorithm>
using namespace std;
int n;
double tp;
struct bigint{
	int v[13];
	bigint(){for(int i=0;i<12;i++)v[i]=0;}
}dp[105][10][10],su[105][10][10],pw[105],st,rv;
bigint operator +(bigint a,bigint b)
{
	int vl=0;
	for(int i=0;i<12;i++)
	{
		vl=vl+a.v[i]+b.v[i];
		a.v[i]=vl&((1<<24)-1);
		vl>>=24;
	}
	return a;
}
int sgn(bigint a){return a.v[11]>>23?-1:1;}
bool operator <(bigint a,bigint b)
{
	int v1=sgn(a),v2=sgn(b);
	if(v1>v2)return 0;if(v1<v2)return 1;
	for(int i=11;i>=0;i--)if(a.v[i]!=b.v[i])return a.v[i]<b.v[i];
	return 0;
}
bigint getrev(bigint a)
{
	int vl=1;
	for(int i=0;i<12;i++)
	{
		vl=vl+(a.v[i]^((1<<24)-1));
		a.v[i]=vl&((1<<24)-1);
		vl>>=24;
	}
	return a;
}
bigint doit(int a){bigint b;b.v[0]=a;return b;}
int main()
{
	freopen("math.in","r",stdin);
	freopen("math.out","w",stdout);
	scanf("%lf",&tp);
	n=tp*100000;
	st=getrev(doit(n));
	for(int i=0;i<10;i++)for(int j=0;j<10;j++)dp[0][i][j]=su[0][i][j]=st;
	for(int i=0;i<9;i++)rv=rv+st;
	pw[0]=doit(100000-n);
	for(int i=1;i<=100;i++)for(int j=0;j<10;j++)pw[i]=pw[i]+pw[i-1];
	for(int i=1;i<=100;i++)
	{
		for(int j=0;j<10;j++)
		for(int k=0;k<10;k++)
		{
			bigint nw,mx;
			mx.v[11]=1<<23;
			for(int l=0;l<10;l++)
			{
				bigint s1=su[i-1][k][l],v1=dp[i-1][k][l];
				if(j<k&&k<l)s1=v1=pw[i-1];
				v1=v1+nw;nw=nw+s1;
				if(mx<v1)mx=v1;
			}
			dp[i][j][k]=mx;su[i][j][k]=nw;
		}
		for(int t=1;t<=9;t++)if(sgn(rv+dp[i][9][t])>=0)
		{
			bigint nw=rv;
			int v1=9,v2=t,fg=0;printf("%d",t);
			for(int j=i;j>=1;j--)
			for(int k=0;k<10;k++)
			{
				bigint s1=su[j-1][v2][k],vl=dp[j-1][v2][k];
				if(fg||(v1<v2&&v2<k))s1=vl=pw[j-1];
				if(sgn(nw+vl)>=0){printf("%d",k);fg|=(v1<v2&&v2<k);v1=v2;v2=k;break;}
				else nw=nw+s1;
			}
			return 0;
		}
		else rv=rv+su[i][9][t];
	}
}
```

#### 6.7

##### DS2: [Ynoi2013] Ynoi

###### Problem

区间排序，区间异或，询问区间xor和。

$n,q\leq 10^5,v\leq 10^8$

$1.5s,32MB$

###### Sol

首先不考虑空间限制做这个问题。

对于区间排序的问题，考虑将区间内的所有数合并为一个段。一个段内的数为排过序的。初始时可以看成每个数一段。

在区间操作的过程中，可能出现一次操作只覆盖了一个段的一部分，此时考虑将这个段分裂。可以发现这个分裂相当于在段内找出最小的若干个数将它们分裂到一个段，剩下的留在原来的段。因此接下来只需要考虑一个操作恰好覆盖了若干个段的情况。

对于每个段，可以用一个01Trie维护。显然01Trie上所有数有序，且支持合并/按照大小分裂。

对于区间xor操作，考虑直接在覆盖的段上打标记。

考虑使用线段树维护每个段。对于一个段，设它在原序列中位置为 $[l,r]$，则可以将它放在线段树上位置 $l$ 的叶子结点上。线段树上只需要维护这个段xor操作被打的标记，以及这一段的大小和异或和。

此时分裂可以直接在线段树上找kth找到需要分裂的位置，然后分裂后将分出来的放在某个叶子结点即可。区间xor/询问直接使用线段树的操作。区间排序也可以直接找到区间内所有段进行合并。

因为一个段的排序操作在xor操作之前，因此xor/询问/分裂时不能把标记下放到01Trie内。只有在排序操作时对于需要排序的每一段先下放标记再合并。

因为分裂次数只有 $O(q)$，01Trie部分的复杂度只有 $O((n+q)\log v)$，因此这样的时间复杂度为 $O((n+q)\log)$

但直接01Trie的空间复杂度为 $O(n\log v)$，32MB显然无法接受。

此时可以考虑压缩Trie的方式，即可将空间复杂度降到 $O(n)$。~~实现留作练习.jpg~~

###### Code

```cpp
#include<cstdio>
#include<queue>
#include<set>
using namespace std;
#define N 100500
int n,q,a,b,c,d,fu[256],v[N];
int ch[N*2][2],tp[N*2][2],sz[N*2],lz[N*2],su[N*2],len[N*2];
queue<int> qu;
void pushdown(int x)
{
	int vl=lz[x],le=len[x];
	for(int i=0;i<2;i++)if(ch[x][i])
	{
		int s2=len[ch[x][i]];
		lz[ch[x][i]]^=vl&((1<<s2)-1);tp[x][i]^=vl>>s2<<s2;
		if(sz[ch[x][i]]&1)su[ch[x][i]]^=vl&((1<<s2)-1);
	}
	if(le&&((vl>>le-1)&1))swap(ch[x][0],ch[x][1]),swap(tp[x][0],tp[x][1]);
	lz[x]=0;
}
void pushup(int x)
{
	sz[x]=sz[ch[x][0]]+sz[ch[x][1]];
	su[x]=su[ch[x][0]]^su[ch[x][1]];
	if(sz[ch[x][0]]&1)su[x]^=tp[x][0];
	if(sz[ch[x][1]]&1)su[x]^=tp[x][1];
	if(!ch[x][0])tp[x][0]=0;if(!ch[x][1])tp[x][1]=0;
}
int calchb(int tp)
{
	if(!fu[1])for(int i=1;i<256;i++)fu[i]=fu[i>>1]+1;
	int as=0;
	while(tp>=256)tp>>=8,as+=8;
	return as+fu[tp];
}
void clr(int x){sz[x]=su[x]=lz[x]=len[x]=0;for(int i=0;i<2;i++)ch[x][i]=tp[x][i]=0;}
int newnode(){int st=qu.front();qu.pop();return st;}
int merge(int x,int y)
{
	pushdown(x);pushdown(y);
	int le=len[x];
	if(!le){sz[x]+=sz[y],qu.push(y);clr(y);return x;}
	for(int i=0;i<2;i++)
	{
		if(!ch[y][i])continue;
		else if(!ch[x][i])ch[x][i]=ch[y][i],tp[x][i]=tp[y][i];
		else
		{
			int l1=tp[x][i],l2=tp[y][i],s1=len[ch[x][i]],s2=len[ch[y][i]];
			if(s1<s2)swap(l1,l2),swap(s1,s2),swap(ch[x][i],ch[y][i]),swap(tp[x][i],tp[y][i]);
			if(!((l1^l2)>>s1))
			if(s1==s2)ch[x][i]=merge(ch[x][i],ch[y][i]);
			else
			{
				int st=newnode();
				len[st]=s1;
				int v2=tp[y][i]&((1<<s1)-1),fg=v2>>s1-1;
				ch[st][fg]=ch[y][i];tp[st][fg]=v2;
				ch[x][i]=merge(ch[x][i],st);
			}
			else
			{
				int t2=calchb(l1^l2);
				if(l1>l2)swap(l1,l2),swap(ch[x][i],ch[y][i]);
				int st=newnode();
				len[st]=t2;
				ch[st][0]=ch[x][i];ch[st][1]=ch[y][i];
				tp[st][0]=l1&((1<<t2)-1);tp[st][1]=l2&((1<<t2)-1);
				tp[x][i]=l1>>t2<<t2;ch[x][i]=st;
				pushup(st);
			}
		}
	}
	qu.push(y);clr(y);
	pushup(x);return x;
}
struct sth{int a,b;};
sth split(int x,int k)
{
	if(!k)return (sth){0,x};
	if(k==sz[x])return (sth){x,0};
	if(!len[x]){int tp=newnode();sz[tp]=sz[x]-k;sz[x]=k;return (sth){x,tp};}
	pushdown(x);
	if(k<=sz[ch[x][0]])
	{
		sth v1=split(ch[x][0],k);
		int st=newnode();
		ch[st][0]=v1.a;tp[st][0]=tp[x][0];len[st]=len[x];pushup(st);
		ch[x][0]=v1.b;pushup(x);return (sth){st,x};
	}
	else
	{
		sth v1=split(ch[x][1],k-sz[ch[x][0]]);
		int st=newnode();
		ch[st][1]=v1.b;tp[st][1]=tp[x][1];len[st]=len[x];pushup(st);
		ch[x][1]=v1.a;pushup(x);return (sth){x,st};
	}
}
void modify(int x,int nt)
{
	int ls=0,t1=0;
	while(len[x]>0)
	{
		pushdown(x);
		if((ch[x][0]&&ch[x][1])||!ls)ls=x,t1=nt^(!ch[x][nt]),x=ch[x][t1];
		else
		{
			int st=!ch[x][0];
			tp[ls][t1]^=tp[x][st];ch[ls][t1]=ch[x][st];
			qu.push(x);clr(x);x=ch[ls][t1];
		}
	}
}
sth split2(int x,int k)
{
	sth v1=split(x,k);
	modify(v1.a,1);
	modify(v1.b,0);
	return v1;
}
struct segt{
	int ls,lz1;
	struct node{int l,r,sz,rt,lz,su;}e[N*4];
	void doit(int x,int k){e[x].su^=(e[x].sz&1)*k;e[x].lz^=k;}
	void pushdown1(int x){if(e[x].lz)doit(x<<1,e[x].lz),doit(x<<1|1,e[x].lz),e[x].lz=0;}
	void pushup1(int x){e[x].su=e[x<<1].su^e[x<<1|1].su;e[x].sz=e[x<<1].sz+e[x<<1|1].sz;}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r)
		{
			e[x].sz=1;e[x].su=v[l];
			int s1=newnode(),s2=newnode();e[x].rt=s1;
			len[s1]=27;ch[s1][v[l]>>26]=s2;tp[s1][v[l]>>26]=v[l];
			su[s1]=v[l];sz[s1]=sz[s2]=1;return;
		}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup1(x);
	}
	int split1(int x,int k)
	{
		if(e[x].l==e[x].r)
		{
			sth v1=split2(e[x].rt,k-1);
			e[x].rt=v1.a;
			lz1=e[x].lz;
			e[x].sz=sz[e[x].rt];e[x].lz=lz1;e[x].su=su[e[x].rt]^((e[x].sz&1)*lz1);
			return v1.b;
		}
		int as;pushdown1(x);
		if(e[x<<1].sz>=k)as=split1(x<<1,k);
		else as=split1(x<<1|1,k-e[x<<1].sz);
		pushup1(x);return as;
	}
	void modify(int x,int v,int s)
	{
		if(e[x].l==e[x].r){e[x].rt=s;e[x].sz=sz[s];e[x].lz=lz1;e[x].su=su[s]^((sz[s]&1)*lz1);lz1=0;return;}
		pushdown1(x);
		int mid=(e[x].l+e[x].r)>>1;
		modify(x<<1|(mid<v),v,s);
		pushup1(x);
	}
	void splitk(int k)
	{
		if(k==n||!k)return;
		int tp=split1(1,k+1);
		modify(1,k+1,tp);
	}
	void modifym(int x,int l,int r)
	{
		if(!e[x].sz)return;
		if(e[x].l==e[x].r)
		{
			lz[e[x].rt]^=e[x].lz;su[e[x].rt]^=(sz[e[x].rt]&1)*e[x].lz;
			if(ls)ls=merge(ls,e[x].rt);else ls=e[x].rt;
			e[x].rt=e[x].sz=e[x].su=e[x].lz=0;
			return;
		}
		pushdown1(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modifym(x<<1,l,r);else if(mid<l)modifym(x<<1|1,l,r);
		else modifym(x<<1,l,mid),modifym(x<<1|1,mid+1,r);
		pushup1(x);
	}
	void modify1(int l,int r){splitk(l-1);splitk(r);ls=0;modifym(1,l,r);modify(1,l,ls);}
	void modifyx(int x,int l,int r,int k)
	{
		if(e[x].l==l&&e[x].r==r){doit(x,k);return;}
		pushdown1(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modifyx(x<<1,l,r,k);else if(mid<l)modifyx(x<<1|1,l,r,k);
		else modifyx(x<<1,l,mid,k),modifyx(x<<1|1,mid+1,r,k);
		pushup1(x);
	}
	void modify2(int l,int r,int k){splitk(l-1);splitk(r);modifyx(1,l,r,k);}
	int query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].su;
		pushdown1(x);
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query(x<<1,l,r);else if(mid<l)return query(x<<1|1,l,r);
		else return query(x<<1,l,mid)^query(x<<1|1,mid+1,r);
	}
	int query1(int l,int r){splitk(l-1);splitk(r);return query(1,l,r);}
}tr;
int main()
{
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n*2+233;i++)qu.push(i);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	tr.build(1,1,n);
	while(q--)
	{
		scanf("%d%d%d",&a,&b,&c);
		if(a==1)scanf("%d",&d),tr.modify2(b,c,d);
		else if(a==2)tr.modify1(b,c);
		else printf("%d\n",tr.query1(b,c));
	}
}
```

##### Fractional Cascading

给出 $n$ 个长度为 $m$ 的有序的序列，每次询问一个数在每个序列上的后继。

要求预处理时间空间 $O(nm)$，询问 $O(\log m+n)$

考虑有序序列上找后继这个问题。

对于一个有序序列，考虑拿出其所有偶数位置构成的序列。如果在后者上求出了一个数的后继，则在原序列上从这个后继开始调整，显然只会调整 $O(1)$ 步就可以找到新的答案。

再考虑在两个序列上找后继的问题。将两个序列归并，对于每个位置，记录这个位置之后的第一个来自序列 $1$ 的元素和第一个来着序列 $2$ 的元素。那么只要在归并后的序列上求出了后继，即可 $O(1)$ 求出两个序列分别的后继。

因此考虑将前两个序列归并，再使用前面的方式让长度减半，再归并，再减半，一直持续这个过程。可以发现每一步的序列长度不会超过 $2m$。

为了定位后继，在序列上可以记录 $las_i$ 表示这个元素在上一个序列中的位置。

询问的时候只需要在最后一个序列上面二分出答案，然后向前倒推即可。每次倒推都是 $O(1)$ 的。

事实上也可以看成在最后做 $\log m$ 次折半，让长度变为 $1$。

预处理时间空间 $O(nm)$，询问 $O(\log m+n)$

###### Code(luogu 6466)

```cpp
#include<cstdio>
using namespace std;
#define N 105
#define M 20040
int n,k,q,d,a,vl[N][M],nt[N][M],ls[N][M],v1[M],v2[M],sz[N],las,f1[M];
int main()
{
	scanf("%d%d%d%d",&n,&k,&q,&d);
	for(int i=1;i<=n;i++)scanf("%d",&vl[1][i]),nt[1][i]=i;sz[1]=n;
	for(int i=2;i<=k;i++)
	{
		for(int j=2;j<=sz[i-1];j+=2)v1[j>>1]=vl[i-1][j];
		for(int j=1;j<=n;j++)scanf("%d",&v2[j]);
		int l1=1,l2=1,r2=n,r1=sz[i-1]>>1,s1=0,s2=0;sz[i]=r1+r2;
		for(int j=1;j<=sz[i];j++)
		{
			int fg=0;
			if(l1==r1+1)fg=1;else if(l2==r2+1)fg=0;else fg=v1[l1]>v2[l2];
			if(fg)vl[i][j]=v2[l2++];else vl[i][j]=v1[l1++];
			f1[j]=fg;
			while(s2<sz[i-1]&&vl[i-1][s2]<vl[i][j])s2++;
			ls[i][j]=s2;
		}
		for(int j=sz[i];j>=1;j--)nt[i][j]=f1[j]?j:nt[i][j+1];
	}
	for(int t=1;t<=q;t++)
	{
		scanf("%d",&a),a^=las;
		int lb=1,rb=sz[k],as=sz[k];
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(vl[k][mid]>=a)as=mid,rb=mid-1;
			else lb=mid+1;
		}
		int nw=as;as=0;
		for(int i=k;i>=1;i--)
		{
			as^=(a>vl[i][nw]?0:vl[i][nt[i][nw]]);
			nw=ls[i][nw];
			while(nw>1&&vl[i-1][nw-1]>=a)nw--;
		}
		las=as;if(t%d==0)printf("%d\n",as);
	}
}
```



#### 6.8

~~Source等会再说~~

##### Problem 1 Algebra

###### Problem

给定正整数 $n,m,k$，求有多少对正整数 $a,b$ 满足：

1. $|a|,|b|\leq k$
2. $x^n+ax+b=0$ 正好有 $k$ 个有理根

多组询问，$\sum m\leq 5\times 10^5$

$5s,512MB$

###### Sol

如果 $n=1$ ，则方程为 $(a+1)x=-b$，有 $1$ 个解当且仅当 $a+1\neq 0,b\neq 0$，否则为无解或无限解。

对于 $n>1$ 的情况，设解为 $\frac pq(\gcd(p,q)=1)$，则 $p^n+apq^{n-1}+q^n=0$

此时有 $p^n\equiv 0(\bmod q)$，因此 $\gcd(p^n,q)=q$，此时一定有 $p=0$ 或者 $q=1$。因此有理根一定是整数。

对于 $n=2$ 的情况，考虑枚举 $a$，则解为 $\frac{-a\pm\sqrt{a^2-4b}}2$，有两个整根当且仅当 $a^2-4b$ 是大于 $0$ 的平方数，有一个整根当且仅当 $a^2-4b=0$，可以直接算出可能的解数。

对于 $n\geq 3$ 的情况，可以发现根的范围一定不超过 $O(m^{\frac 1{n-1}})$

考虑枚举 $a$，再枚举根，此时只有唯一的 $b$ 满足要求。可以考虑所有根，求出每个根对应的 $b$，然后即可判断每个 $b$ 对应多少组解。

复杂度 $O(m^{1.5}\log m)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<cmath>
#include<algorithm>
using namespace std;
#define N 1005000
#define ll long long
int n,m,k,vl[N];
vector<int> s1;
int main()
{
	while(~scanf("%d%d%d",&n,&m,&k))
	{
		if(k>3)printf("0\n");
		else if(n==1)
		{
			if(k!=1)printf("0\n");
			else printf("%lld\n",1ll*2*m*(2*m+1));
		}
		else if(n==2)
		{
			ll as=0;
			for(int i=0;i<=m;i++)
			{
				ll tp=1ll*i*i,fg=i?2:1;
				if(k==1)as+=fg*(tp%4==0&&tp/4>=-m&&tp/4<=m);
				else if(k==2)
				{
					ll lb=tp-4*m,rb=tp+4*m;
					if(lb<=0)lb=1;
					ll s1=ceil(sqrt(lb)),s2=floor(sqrt(rb));
					if((s1-i)&1)s1++;if((s2-i)&1)s2--;
					as=(as+fg*((s2-s1+2)/2));
				}
			}
			printf("%lld\n",as);
		}
		else
		{
			ll as=0;
			int tp=pow(m,1.0/(n-1));
			while(pow(tp+1,n)-(tp+1)*m-m<=0)tp++;
			for(int i=-tp;i<=tp;i++)
			{
				int as=1;
				for(int j=1;j<=n;j++)as=as*i;
				vl[i+tp]=as;
			}
			for(int i=-m;i<=m;i++)
			{
				for(int j=-tp;j<=tp;j++)
				{
					int v1=vl[j+tp]+i*j;v1*=-1;
					if(v1<-m||v1>m)continue;
					s1.push_back(v1);
				}
				int ls=-m-1,ct=-2;
				sort(s1.begin(),s1.end());
				for(int j=0;j<s1.size();j++)
				{
					if(ls!=s1[j])as+=ct==k,ct=1,ls=s1[j];
					else ct++;
				}
				as+=ct==k;s1.clear();
			}
			printf("%lld\n",as);
		}
	}
}
```

##### Problem 2 Number Theory

###### Problem

定义 $F(n)=\sum_{j=0}^{n-1}10^j(n>0)$

给出一个正整数 $n$，你每次可以选择一个 $k$，将 $n$ 加上或减去 $F(k)$，一次操作的代价为 $k$。

求将 $n$ 变为 $0$ 的最小代价。

多组数据，$n\leq 10^{5000}$，所有 $n$ 的位数和不超过 $5\times 10^4$

$7s,512MB$

###### Sol

显然 $10*F(n)=F(n+1)-F(1)$，因此可以发现如果某个 $F(n)$ 使用了 $\geq 6$ 次，则使用上面的调整，可以将 $6n$ 的代价变为 $5n+2$，因此 $F(n)(n>1)$ 最多使用 $5$ 次，$F(1)$ 最多使用 $6$ 次。

考虑将所有操作 $*9$，一次操作变为改变 $10^{k+1}-1$。

考虑枚举最后所有操作中操作的系数之和 $x$，则相当于找到整数序列 $s$ 满足：

1. $\sum s_i10^{i+1}-x=n$
2. $\sum s_i=x$ 
3. $\sum |s_i|*i$ 最小

因为每次都是加 $10^{i+1}$，因此可以考虑从后向前填 $s_i$。

因为 $|s_i|\leq 6$，因此除去最后 $\log_{10} (6*n)$ 位后，左侧做加法时的进位数量在 $[-1,0]$ 间。

考虑从后向前填数，则当前只需要记录三个值：

1. 当前的位数 $i$
2. 当前前面还需要的 $\sum s_i=x$
3. 当前后面做加法的进位数 $s$

则可以记录 $dp_{i,x,s}$ 表示当前填了后 $i$ 个 $s_i$，当前前面还需要 $\sum s_i=x$，当前后面加法进位数为 $s$ 时，前面最小的 $\sum |s_i|*i$。

转移时可以枚举这一位的 $s_i$，因为需要在这一位上相同，因此只有 $\frac 1{10}$ 的 $s_i$ 合法，同时可能的选择只有 $13$ 种。因此转移最多两种。

同时除去最后几位后，状态数不超过 $2*11*\frac 12n^2$ 级别，容易发现 $\sum s_i10^{i+1}\equiv \sum s_i(\bmod 9)$，因此可能的状态只有 $\frac 19$ 合法，因此最后的常数很小。~~但是还是需要卡常~~

复杂度 $O(n^2)$

###### Code

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 5100
int n,v[N],as=0,dp[N][N*5/4][2];
char s[N];
int dfs(int m,int d,int f)
{
	if(m>n&&!d&&!f)return 0;
	if((d>0?d:-d)>(n-m+5)*5)return 1e8;
	int fg=1,v2=0,v1=(d+(n-m+5)*5)/9;
	if(f<-1||f>0)fg=0;else v2=f+1;
	if(fg&&dp[m][v1][v2]<1e9)return dp[m][v1][v2];
	int vl=(v[m]-f)%10,as1=1e8;
	if(vl<-5)vl+=10;if(vl>4)vl-=10;
	int lb=vl,rb=5;
	if(m==2)lb-=10,rb+=10;
	for(int i=lb;i<=rb;i+=10)
	{
		int nt=(f+i-v[m])/10;
		as1=min(as1,(i>0?i:-i)*(m-1)+dfs(m+1,d-i,nt));
	}
	if(fg)dp[m][v1][v2]=as1;
	return as1;
}
int main()
{
	while(~scanf("%s",s+1))
	{
		n=strlen(s+1);
		for(int i=1;i<=n;i++)v[i]=s[n-i+1]-'0';
		for(int i=n+1;i<=n+50;i++)v[i]=0;
		int ls=0;
		for(int i=1;i<=n||ls;i++)
		{
			v[i]=v[i]*9+ls;
			ls=v[i]/10;v[i]-=ls*10;
			if(i>n)n=i;
		}
		for(int i=0;i<=n+5;i++)
		for(int j=0;j<=(n+5-i)*10/9;j++)
		for(int l=0;l<2;l++)
		dp[i][j][l]=1e9;
		as=1e8;
		int st=n<=500?n:n/4;
		for(int i=-5*st;i<=5*st;i++)if((i-v[1])%10==0)
		{
			int tp=(i-v[1])/10;
			as=min(as,dfs(2,-i,tp));
		}
		printf("%d\n",as);
	}
}
```

##### Problem 3 Simple Hull

###### Problem

给出二维平面上 $n$ 个点，在点 $i$ 和 $(i\bmod n)+1$ 间有连边。保证每条连边均与一个坐标轴平行，且相邻两条边一定是X方向和Y方向交错。

找一个面积最小的简单多边形(内部不存在空洞，且不存在边界相交)，使得它包含所有边。输出最小面积的下界。

$n\leq 10^5,v\leq 10^6$

$5s,512MB$

###### Sol

考虑从左下角开始，沿着边界走一圈，可以发现这样得到的一定是最小的简单多边形。

考虑模拟走的过程， 维护当前正在走的边，考虑在走到结尾前会不会碰到其它边导致转向。这相当于找到第一个遇到的穿过当前线段且与当前线段方向垂直的线段。

考虑当前在 $(x,y)$，向 $x$ 增加方向走到 $x'$ 的过程，相当于找到一条 $y$ 方向线段 $(x_1,y_1),(x_1,y_2)$ 满足：

1. $x\leq x_1\leq x'$
2. $y_1\leq y< y_2$
3. $x_1$ 在满足上述条件的线段中最小

按照 $y_1$ 将线段排序，依次加入，然后相当于求某个时刻横坐标一个区间内第一个满足 $y_2>y$ 的位置。

因此可以使用可持久化线段树维护。另外三个方向分别开一个即可。

如果找不到这样的线段，则一定会向前走到端点，然后判断是否存在一条继续向前延伸的线段，存在则继续走这条线段，否则按照给出的顺序转向下一条边即可。

得到外轮廓后即可直接求出答案。

复杂度 $O(n\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define M 2250000
int n,s[N][2],as[N*4][2],ct,c1,c2,fr,f1[4]={2,3,1,0};
struct sth{int a,b,c,d;}s1[N],s2[N];
bool cmp(sth a,sth b){return a.a<b.a;}
struct pretree{
	int ch[M][2],mx[M],id[M],rt[M],ct,nw;
	void pushup(int x){mx[x]=max(mx[ch[x][0]],mx[ch[x][1]]);}
	int modify(int x,int l,int r,int d,int s,int v)
	{
		int st=++ct;
		ch[st][0]=ch[x][0];ch[st][1]=ch[x][1];mx[st]=mx[x];id[st]=id[x];
		if(l==r){if(s>mx[st])mx[st]=s,id[st]=v;return st;}
		int mid=(l+r)>>1;
		if(mid>=d)ch[st][0]=modify(ch[x][0],l,mid,d,s,v);
		else ch[st][1]=modify(ch[x][1],mid+1,r,d,s,v);
		pushup(st);return st;
	}
	int query(int x,int l,int r,int l1,int r1,int t)
	{
		if(mx[x]<t||l1>r1)return 0;
		if(l==r)return id[x];
		int mid=(l+r)>>1;
		if(mid>=r1)return query(ch[x][0],l,mid,l1,r1,t);
		else if(mid<l1)return query(ch[x][1],mid+1,r,l1,r1,t);
		else
		{
			int v1=query(ch[x][0],l,mid,l1,mid,t);
			if(v1)return v1;
			return query(ch[x][1],mid+1,r,mid+1,r1,t);
		}
	}
}tr[4];
//0U1D2R3L
void doit(sth s1[],int c1,int id)
{
	sort(s1+1,s1+c1+1,cmp);
	int l1=1;
	for(int i=1;i<=1e6;i++)
	{
		while(l1<=c1&&s1[l1].a<=i)tr[id].nw=tr[id].modify(tr[id].nw,1,1e6,id?s1[l1].c:1e6+1-s1[l1].c,s1[l1].b,s1[l1].d),l1++;
		tr[id].rt[i]=tr[id].nw;
	}
	for(int i=1;i<=c1;i++)swap(s1[i].a,s1[i].b);
	sort(s1+1,s1+c1+1,cmp);
	l1=c1;
	for(int i=1e6;i>=1;i--)
	{
		while(l1>=1&&s1[l1].a>=i)tr[id+1].nw=tr[id+1].modify(tr[id+1].nw,1,1e6,!id?s1[l1].c:1e6+1-s1[l1].c,1e6+1-s1[l1].b,s1[l1].d),l1--;
		tr[id+1].rt[i]=tr[id+1].nw;
	}
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]);
	for(int i=1;i<=n;i++)
	{
		int a1=s[i][0],a2=s[i][1],b1=s[i%n+1][0],b2=s[i%n+1][1];
		if(a1>b1)a1^=b1^=a1^=b1;if(a2>b2)a2^=b2^=a2^=b2;
		if(a1==b1)s1[++c1]=(sth){a2,b2,a1,i};else s2[++c2]=(sth){a1,b1,a2,i};
	}
	doit(s1,c1,0);doit(s2,c2,2);
	for(int i=1;i<=n;i++)if(!fr||(s[i][0]<s[fr][0]||(s[i][0]==s[fr][0]&&s[i][1]<s[fr][1])))fr=i;
	int nwx=s[fr][0],nwy=s[fr][1],dir=2,tp;
	if(s[fr%n+1][1]==nwy)tp=fr;
	else tp=(fr+n-2)%n+1;
	while(1)
	{
		as[++ct][0]=nwx;as[ct][1]=nwy;
		int t1,l1,l2,nt;
		if(dir>=2){t1=s[tp][1],l1=s[tp][0],l2=s[tp%n+1][0],nt=dir==2?tp%n+1:(tp+n-2)%n+1;if(l1>l2)swap(l1,l2),nt=dir==3?tp%n+1:(tp+n-2)%n+1;}
		else{t1=s[tp][0],l1=s[tp][1],l2=s[tp%n+1][1],nt=dir==0?tp%n+1:(tp+n-2)%n+1;if(l1>l2)swap(l1,l2),nt=dir==1?tp%n+1:(tp+n-2)%n+1;}
		if(dir==0)l1=nwy+1;
		if(dir==1)l2=nwy-1;
		if(dir==2)l1=nwx+1;
		if(dir==3)l2=nwx-1;
		int d1=f1[dir];
		int fu=tr[d1].query(tr[d1].rt[t1],1,1e6,d1%3==0?1e6+1-l2:l1,d1%3==0?1e6+1-l1:l2,(d1&1?1e6+1-t1:t1)+1);
		if(fu)
		{
			tp=fu;
			if(dir>=2)nwx=s[tp][0];
			else nwy=s[tp][1];
			dir=d1;
		}
		else
		{
			if(dir>=2)nwx=s[nt][0];
			else nwy=s[nt][1];
			int ft=dir>=2?nwx:nwy;
			int fu2=tr[dir].query(tr[dir].rt[ft],1,1e6,dir%3==0?1e6+1-t1:t1,dir%3==0?1e6+1-t1:t1,dir&1?1e6+2-ft:ft+1);
			if(fu2)tp=fu2;
			else tp=nt,dir=d1^1;
		}
		if(nwx==s[fr][0]&&nwy==s[fr][1])break;
	}
	long long as1=0;
	for(int i=1;i<=ct;i++)as1+=1ll*as[i][0]*as[i%ct+1][1]-1ll*as[i][1]*as[i%ct+1][0];
	printf("%lld\n",as1/2);
}
```

#### 6.9

##### T1 button

###### Problem

你有一个显示器，显示器上有五个数位，能显示 $0\sim 99999$ 间的整数。初始数为 $0$。

你需要猜一个数，已知这个数为 $[l,r]$ 间的正整数，你可以进行如下操作：

1. 改变显示器上一个数位的值。
2. 进行询问，询问会返回显示器当前的数与答案的大小关系。

你需要最坏情况下操作次数最少。求出最优策略下最坏情况的最少操作次数。

多组数据

$T\leq 50,0<l\leq r\leq 99999$

$10s,1024MB$

###### Sol

考虑暴力dp，可以设 $dp_{l,r,0/1}$ 表示当前数可能在区间 $[l,r]$ 中，当前显示器上的数为 $l-1/r+1$，剩余部分的最小操作次数。记 $f(i,j)$ 表示从 $i$ 到 $j$ 需要变的数位数，则转移为：
$$
dp_{l,r,0}=\min_{i\in[l,r]}f(l-1,i)+\max(dp_{l,i-1,1},dp_{i+1,r,0})
$$
另外一侧类似。注意到 $dp$ 的值不超过 $42$，且显然在 $l$ 固定时，$dp_{l,r,0}$ 随着 $r$ 增加单调，在 $r$ 固定时， $dp_{l,r,1}$ 随着 $l$ 减小单调。

考虑记录分界点，设 $sl_{l,v}$ 表示最大的 $r$ 满足 $dp_{l,r,0}\leq v$，$sr_{r,v}$ 表示最小的 $l$ 满足 $dp_{l,r,1}\leq v$。转移考虑枚举下一次询问的数：
$$
sl_{l,v}=\max[sr_{i-1,v-1-f(l-1,i)}\leq l]sl_{i+1,v-1-f(l-1,i)}
$$
$sr$ 同理。但直接转移无法通过。

考虑枚举 $i$，再枚举 $i$ 在变为 $l-1$ 个过程中改变了哪些位。此时合法的 $l$ 为大于等于某个值且 $l-1$ 在没有被改变的位上与 $i$ 相同的所有 $l$。

再考虑枚举 $l-1$ 有哪些位没有改变，此时可以看成两者能够配对当且仅当两者分别把改变的位换成 `?` 后相等，可能的串只有 $(10+1)^5$ 个。因此可以记录每个串当前最大的转移值。

因为上面的每一种情况影响的是一段后缀，因此可以看成在某个位置加入这个转移。先处理出所有的转移，然后再从小到大枚举 $l$ 以及改变的情况计算新的 $sl$，计算 $sr$ 的情况同理。

最后计算答案时只需要枚举第一次询问的数，求一个 $dp$ 值可以在对应的 $sl/sr$ 上二分。

设位数为 $d$，显然答案为 $O(d)$ 级别，复杂度为 $O(d*10^d*2^d+10^d*q*\log d)$

###### Code

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 100050
int T,l,r,sl[N][46],sr[N][46],tp[N*2],nw,s1,pw[6]={1,10,100,1000,10000,100000},pw11[6]={1,11,121,1331,14641,161051};
struct sth{int a,b;};
vector<sth> t1[N],t2[N];
void dfs1(int x,int d,int v,int ti)
{
	if(ti<0)return;
	if(d==5){int lb=sl[x-1][ti],rb=sr[x+1][ti];t1[lb].push_back((sth){v,rb});t2[rb].push_back((sth){v,lb});return;}
	int tp=x/pw[d]%10;
	dfs1(x,d+1,v+tp*pw11[d],ti);
	dfs1(x,d+1,v+10*pw11[d],ti-1);
}
void dfs2(int x,int d,int v)
{
	if(d==5){s1=max(s1,tp[v]);return;}
	int tp=x/pw[d]%10;
	dfs2(x,d+1,v+tp*pw11[d]);
	dfs2(x,d+1,v+10*pw11[d]);
}
void dfs3(int x,int d,int v)
{
	if(d==5){s1=min(s1,tp[v]);return;}
	int tp=x/pw[d]%10;
	dfs3(x,d+1,v+tp*pw11[d]);
	dfs3(x,d+1,v+10*pw11[d]);
}
int getdpl(int x,int l)
{
	int lb=0,rb=43,as=45;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(sl[x][mid]<=l)as=mid,rb=mid-1;
		else lb=mid+1;
	}
	return as;
}
int getdpr(int x,int r)
{
	int lb=0,rb=43,as=45;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(sr[x][mid]>=r)as=mid,rb=mid-1;
		else lb=mid+1;
	}
	return as;
}
int getf(int x,int y)
{
	int as=0;
	while(x||y)
	{
		int v1=x%10,v2=y%10;
		if(v1!=v2)as++;
		x/=10;y/=10;
	}
	return as;
}
int main()
{
	freopen("button.in","r",stdin);
	freopen("button.out","w",stdout);
	for(int i=0;i<1e5;i++)sl[i][0]=sr[i][0]=i;
	sr[100000][0]=99999;
	for(nw=1;nw<=43;nw++)
	{
		for(int i=0;i<=1e5;i++)sl[i][nw]=sl[i][nw-1],sr[i][nw]=sr[i][nw-1],t1[i].clear(),t2[i].clear();
		for(int i=1;i<1e5;i++)dfs1(i,0,0,nw-1);
		for(int i=0;i<162000;i++)tp[i]=0;
		for(int i=0;i<1e5;i++)
		{
			for(int j=0;j<t1[i].size();j++)tp[t1[i][j].a]=max(tp[t1[i][j].a],t1[i][j].b);
			if(i>0)
			{
				s1=0;dfs2(i-1,0,0);
				sr[i][nw]=max(sr[i][nw],s1);
			}
		}
		for(int i=0;i<162000;i++)tp[i]=1e5;
		for(int i=1e5;i>0;i--)
		{
			for(int j=0;j<t2[i].size();j++)tp[t2[i][j].a]=min(tp[t2[i][j].a],t2[i][j].b);
			if(i<1e5)
			{
				s1=1e5;dfs3(i+1,0,0);
				sl[i][nw]=min(sl[i][nw],s1);
			}
		}
	}
	scanf("%d",&T);
	while(T--)
	{
		scanf("%d%d",&l,&r);
		int as=1e9;
		for(int i=l;i<=r;i++)as=min(as,max(getdpl(i-1,l),getdpr(i+1,r))+getf(0,i)+1);
		printf("%d\n",as);
	}
}
```

##### T2 machine

###### Problem

给出三维空间中 $n$ 个长方体，第 $i$ 个长方体由所有满足 $x\in[x_{i,1},x_{i,2}],y\in[y_{i,1},y_{i,2}],z\in[z_{i,1},z_{i,2}]$ 的点组成。

你需要选择三个整数 $a,b,c$ ，满足每个给出的长方体至少和平面 $x=a,y=b,z=c$ 中的一个相交。

输出任意一组方案或输出无解。

$n\leq 10^5$

$5s,1024MB$

###### Sol

考虑枚举 $a$，把选择 $b,c$ 看成二维平面上的一个点，考虑所有长方体的限制。

如果 $a\in[x_{i,1},x_{i,2}]$，显然没有限制，否则可以发现限制为 $b\in[y_{i,1},y_{i,2}]\cup c\in[z_{i,1},z_{i,2}]$，这相当于二维平面上限制了四个角的区域不能选。 

考虑对 $x$ 一维做线段树分治，则相当于在二维平面上支持三种操作：

1. 加入一个限制，表示 $v_1x\leq a,v_2y\leq b(v_1,v_2\in\{-1,1\})$ 的区域不能选。
2. 询问当前是否有合法的点。
3. 回退一个1操作。

考虑使用线段树维护当前合法的 $y$ 区间，支持加入一个不合法区间以及撤销。再使用四个线段树维护四个方向上的限制。

考虑加入一个限制的情况，不妨设加入的是左下角的限制 $(x,y)$，考虑找到加入限制后 $y$ 变为不合法的部分。

可以发现这个部分为左上部分中满足当前 $y$ 右侧有一个横坐标限制小于等于 $x+1$ 的部分，以及右上部分中满足当前 $y$ 左侧有一个横坐标限制小于等于 $x+1$ 的部分的并与加入限制覆盖的 $y$ 部分的交。

前两个显然是一个前缀和一个后缀，可以线段树上求出，因此这个部分可以被不超过两个区间表示。

线段树中加入一个限制可以只做单点修改，因此可以直接回退。

复杂度 $O(n\log^2 n)$ ，常数非常大。

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 200500
int n,sx[N],sy[N],sz[N],v[N][6],fg,asx,asy,asz;
struct segt{
	struct node{int l,r,mx;}e[N*4];
	void pushup(int x){e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;if(l==r)return;
		int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	}
	int modify(int x,int d,int v,int f)
	{
		int as=0;
		while(1)
		{
			if(e[x].l==e[x].r){as=e[x].mx;e[x].mx=max(e[x].mx,v);if(f)e[x].mx=v;break;}
			int mid=(e[x].l+e[x].r)>>1;
			x=x<<1|(mid<d);
		}
		x>>=1;
		while(x)pushup(x),x>>=1;
		return as;
	}
	int query(int x,int v,int fg)
	{
		if(e[x].mx<v)return fg?e[x].l-1:e[x].r+1;
		while(1)
		{
			if(e[x].l==e[x].r)return e[x].l;
			if(e[x<<1|fg].mx>=v)x=x<<1|fg;
			else x=x<<1|(!fg);
		}
	}
	int query1(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r)return e[x].mx;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)return query1(x<<1,l,r);
		else if(mid<l)return query1(x<<1|1,l,r);
		else return max(query1(x<<1,l,mid),query1(x<<1|1,mid+1,r));
	}
}tr[4];
struct segt2{
	struct node{int l,r,mn,fr,lz;}e[N*4];
	void doit(int x,int v){e[x].lz+=v;e[x].mn+=v;}
	void pushup(int x)
	{
		e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);
		if(e[x<<1].mn==e[x].mn)e[x].fr=e[x<<1].fr;
		else e[x].fr=e[x<<1|1].fr;
	}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){e[x].fr=l;return;}
		int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int l,int r,int v)
	{
		if(e[x].l==l&&e[x].r==r){doit(x,v);return;}
		if(e[x].lz)doit(x<<1,e[x].lz),doit(x<<1|1,e[x].lz),e[x].lz=0;
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)modify(x<<1,l,r,v);else if(mid<l)modify(x<<1|1,l,r,v);
		else modify(x<<1,l,mid,v),modify(x<<1|1,mid+1,r,v);
		pushup(x);
	}
}t1;
struct sth{int l,r,x,y,op;};
vector<sth> tp[23];
struct sth1{int x,y;};
struct sth2{int x,y,v;};
void solve(int l,int r,int d)
{
	if(fg)return;
	vector<sth1> md1[4];
	vector<sth2> md2;
	for(int i=0;i<tp[d].size();i++)if(tp[d][i].l<=l&&tp[d][i].r>=r)
	{
		int sx=tp[d][i].x,sy=tp[d][i].y,op=tp[d][i].op,sl,sr;
		if(~op&1)sl=1,sr=sx;else sl=sx,sr=n*2;
		int lb=0,rb=n*2+1;
		if(op<2)lb=tr[2].query(1,n*2+1-sy,1),rb=tr[3].query(1,n*2+1-sy,0);
		else lb=tr[0].query(1,sy,1),rb=tr[1].query(1,sy,0);
		if(lb>=sl)
		{
			t1.modify(1,sl,min(sr,lb),1);
			md2.push_back((sth2){sl,min(sr,lb),1});
		}
		if(rb<=sr)
		{
			t1.modify(1,max(sl,rb),sr,1);
			md2.push_back((sth2){max(sl,rb),sr,1});
		}
		int ls=tr[op].modify(1,sx,op<2?sy:n*2+1-sy,0);
		md1[op].push_back((sth1){sx,ls});
	}
	if(l==r)
	{
		if(!fg&&t1.e[1].mn==0)
		{
			fg=1;
			asx=l,asy=t1.e[1].fr;
			asz=max(tr[0].query1(1,asy,n*2),tr[1].query1(1,1,asy));
		}
	}
	else if(t1.e[1].mn==0)
	{
		int mid=(l+r)>>1;
		tp[d+1].clear();
		for(int i=0;i<tp[d].size();i++)if(!(tp[d][i].l<=l&&tp[d][i].r>=r))if(tp[d][i].l<=mid&&tp[d][i].r>=l)tp[d+1].push_back(tp[d][i]);
		solve(l,mid,d+1);
		tp[d+1].clear();
		for(int i=0;i<tp[d].size();i++)if(!(tp[d][i].l<=l&&tp[d][i].r>=r))if(tp[d][i].l<=r&&tp[d][i].r>=mid+1)tp[d+1].push_back(tp[d][i]);
		solve(mid+1,r,d+1);
	}
	for(int i=0;i<4;i++)if(md1[i].size())
	for(int j=md1[i].size()-1;j>=0;j--)tr[i].modify(1,md1[i][j].x,md1[i][j].y,1);
	for(int j=0;j<md2.size();j++)t1.modify(1,md2[j].x,md2[j].y,-md2[j].v);
}
int main()
{
	freopen("machine.in","r",stdin);
	freopen("machine.out","w",stdout);
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<6;j++)scanf("%d",&v[i][j]);
		for(int j=0;j<2;j++)sx[i*2+j-1]=v[i][j],sy[i*2+j-1]=v[i][j+2],sz[i*2+j-1]=v[i][j+4];
	}
	sort(sx+1,sx+2*n+1);
	sort(sy+1,sy+2*n+1);
	sort(sz+1,sz+2*n+1);
	for(int i=1;i<=n;i++)
	for(int j=0;j<2;j++)
	v[i][j]=lower_bound(sx+1,sx+n*2+1,v[i][j])-sx,v[i][j+2]=lower_bound(sy+1,sy+n*2+1,v[i][j+2])-sy,v[i][j+4]=lower_bound(sz+1,sz+n*2+1,v[i][j+4])-sz;
	for(int i=1;i<=n;i++)
	{
		int lx=v[i][0],rx=v[i][1],ly=v[i][2],ry=v[i][3],lz=v[i][4],rz=v[i][5];
		if(lx>1)
		{
			if(ly>1)tp[0].push_back((sth){1,lx-1,ly-1,lz,0});
			if(ly>1&&rz<n*2)tp[0].push_back((sth){1,lx-1,ly-1,rz+1,2});
			if(ry<n*2)tp[0].push_back((sth){1,lx-1,ry+1,lz,1});
			if(ry<n*2&&rz<n*2)tp[0].push_back((sth){1,lx-1,ry+1,rz+1,3});
		}
		if(rx<n*2)
		{
			if(ly>1)tp[0].push_back((sth){rx+1,n*2,ly-1,lz,0});
			if(ly>1&&rz<n*2)tp[0].push_back((sth){rx+1,n*2,ly-1,rz+1,2});
			if(ry<n*2)tp[0].push_back((sth){rx+1,n*2,ry+1,lz,1});
			if(ry<n*2&&rz<n*2)tp[0].push_back((sth){rx+1,n*2,ry+1,rz+1,3});
		}
	}
	for(int i=0;i<4;i++)tr[i].build(1,1,n*2);
	t1.build(1,1,n*2);
	solve(1,n*2,0);
	if(fg)printf("YES\n%d %d %d\n",sx[asx],sy[asy],sz[asz]);
	else printf("NO\n");
}
```

##### T3 xor

###### Problem

有 $n$ 个数以及 $n$ 个限制 $a_1,...,a_n$，表示第 $i$ 个数 $v_i$ 的取值为 $\{0,1,...,a_i\}$。

有 $q$ 次操作：

1. 修改一个 $a_i$。
2. 给出 $l,r,l_1,r_1$，询问只考虑 $[l,r]$ 之间的数，考虑它们的所有取值可能，其中满足 $\oplus_{i=l}^rv_i\in[l_1,r_1]$ 的方案个数，模 $998244353$。

$n,q\leq 10^5,v\leq 10^9$

$2s,512MB$

###### Sol

考虑一组询问，根据经典套路，将所有数一起从高位向低位填，找到第一个有数脱离上界的位 $k$。

可以发现，之后无论其它数怎么填，这个数后面的位遍历 $\{0,1,...,2^k-1\}$ 即可让所有数后面的异或遍历 $\{0,1,...,2^k-1\}$，因此后面的位异或得到每一种结果的方案数相同，同时前面显然是固定的。

枚举 $k$，此时可以知道高于这一位的一定等于上界，考虑这一位上的情况：

设 $dp_{i,0/1,0/1}$ 表示考虑了前 $i$ 个数，当前这一位的异或值为 $0/1$，当前前面是否有一个数脱离上界时，前面的每个数后面在不超过上界的情况下任意填的方案数。

此时可以发现这一位上的情况得到的结果一定形如前若干位固定，后 $k$ 位任意，容易算出这 $2^k$ 种方案中有多少个在 $[l_1,r_1]$ 区间中。

转移显然，同时可以发现转移可以可以看成类似矩阵的形式，因此可以线段树优化。

因此可以对每一位开一个线段树维护每一位的dp，修改在每一个线段树上单点修改，询问则依次对每个线段树询问。为了降低常数可以将所有的线段树放在一起。

复杂度 $O(n\log n\log v)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100500
#define mod 998244353
int n,m,v[N],a,l,r,v1,v2;
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
struct sth{int s1,s2,vl,tp;}as[30];
sth operator +(sth a,sth b)
{
	sth c;
	c.s1=(1ll*a.s1*b.s1+1ll*a.s2*b.s2)%mod;
	c.s2=(1ll*a.s2*b.s1+1ll*a.s1*b.s2)%mod;
	c.vl=1ll*a.vl*b.vl%mod;
	c.tp=a.tp^b.tp;
	return c;
}
struct segt{
	struct node{int l,r;sth vl[30];}e[N*4];
	void pushup(int x){for(int d=0;d<30;d++)e[x].vl[d]=e[x<<1].vl[d]+e[x<<1|1].vl[d];}
	void build(int x,int l,int r)
	{
		e[x].l=l;e[x].r=r;
		if(l==r){for(int d=0;d<30;d++)if((v[l]>>d)&1)e[x].vl[d]=(sth){(v[l]&((1<<d)-1))+1,1<<d,(v[l]&((1<<d)-1))+1,1};else e[x].vl[d]=(sth){(v[l]&((1<<d)-1))+1,0,(v[l]&((1<<d)-1))+1,0};return;}
		int mid=(l+r)>>1;
		build(x<<1,l,mid);build(x<<1|1,mid+1,r);
		pushup(x);
	}
	void modify(int x,int t)
	{
		if(e[x].l==e[x].r){int l=e[x].l;for(int d=0;d<30;d++)if((v[l]>>d)&1)e[x].vl[d]=(sth){(v[l]&((1<<d)-1))+1,1<<d,(v[l]&((1<<d)-1))+1,1};else e[x].vl[d]=(sth){(v[l]&((1<<d)-1))+1,0,(v[l]&((1<<d)-1))+1,0};return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=t)modify(x<<1,t);else modify(x<<1|1,t);
		pushup(x);
	}
	void query(int x,int l,int r)
	{
		if(e[x].l==l&&e[x].r==r){for(int d=0;d<30;d++)as[d]=as[d]+e[x].vl[d];return;}
		int mid=(e[x].l+e[x].r)>>1;
		if(mid>=r)query(x<<1,l,r);else if(mid<l)query(x<<1|1,l,r);
		else query(x<<1,l,mid),query(x<<1|1,mid+1,r);
	}
}tr;
int doit(int l1,int r1,int l2,int r2)
{
	int tp=min(r1,r2)-max(l1,l2)+1;
	return tp<0?0:tp;
}
int main()
{
	freopen("xor.in","r",stdin);
	freopen("xor.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	tr.build(1,1,n);
	while(m--)
	{
		scanf("%d%d%d",&a,&l,&r);
		if(a==1)v[l]=r,tr.modify(1,l);
		else
		{
			scanf("%d%d",&v1,&v2);
			int l1=0,r1=(1<<30)-1;
			int as1=0,su=0;
			for(int i=29;i>=0;i--)as[i]=(sth){1,0,1,0};
			tr.query(1,l,r);
			for(int i=29;i>=0;i--)
			{
				su^=((as[i].tp&1)<<i);
				as1=1ll*as1*(mod+1)/2%mod;
				int s2=doit(l1^(1<<i),r1,v1,v2),s1=doit(l1,r1^(1<<i),v1,v2);
				if(as[i].tp&1)swap(s1,s2),l1^=1<<i;else r1^=1<<i;
				as1=(as1+1ll*s2*as[i].s2)%mod;
				as1=(as1+1ll*s1*(as[i].s1-as[i].vl+mod))%mod;
			}
			if(v1<=su&&su<=v2)as1=(as1+1)%mod;
			printf("%d\n",as1);
		}
	}
}
```

#### 6.10

##### Problem 1 Princess and Her Shadow

Source: CF317E

###### Problem

有一个无穷大的二维网格，其中有 $m$ 个格子是障碍。

两个人在网格上进行游戏，给出两人的初始位置。第一个人可以向相邻且不是障碍的格子移动。每次第一个人移动时，如果第二个人相同方向的下一个格子不是障碍，则第二个人会向与第一个人相同的方向移动一个位置。在第二个人移动完后，如果两个人位置重合，则游戏结束。

求出一种第一个人移动步数不超过 $10^6$ 的方案使得游戏结束，或输出无解。

$m\leq 400$，障碍坐标范围不超过 $[-100,100]$

$1s,256MB$

###### Sol

显然两人不在一个连通块内无解，考虑剩下的情况：

分情况讨论：

Case1: 两人都在外部的连通块

此时可以先将第一个人移到外部，然后再通过第一个人的操作让第二个人也到达外部。这里的实现可以bfs。

此时考虑让两人都不再进入内部，只通过每个方向上最外侧的一个障碍调整位置。

如果第一个人当前在第二个人上方，则可以让第二个人移动到上方最外侧的一个障碍，然后第一个人开始向下走，这时第二个人无法向下，可以做到让两人横坐标相同。另外三个方向上同理。这样显然可以做到。

操作步数 $O(v)$，直接实现复杂度 $O(v^2)$

Case2: 两人在一个封闭的区域内

考虑如下做法：

找到当前第一个人到第二个人的最短路，让第一个人沿着最短路走。如果走到后没有结束，则重复这个过程。

如果在走的过程中，第二个人有一步不能走，则此时两个人的最短路长度一定会减少 $1$。

如果没有出现这种情况，则两个人会循环走下去。显然 $v$ 轮一定遇到障碍，因此最坏情况下一轮的步数为 $O(vm)$ 级别~~事实上好像是v+m但我不会证~~

因此这样的步数不超过 $O(vm^2)$，使用bfs求每一步，找到终点就停止，复杂度不超过 $O(l*v)$ ，其中 $l$ 为答案长度。实际上完全跑不满，原数据最坏25000步。

###### Code

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 205
int s1,s2,t1,t2,m,a,b,is[N][N],fr[N][N],ct,d[4][2]={-1,0,1,0,0,-1,0,1},f1[N][N];
char di[6]="LRDU";
void dfs(int x,int y)
{
	for(int i=0;i<4;i++)
	{
		int nx=x+d[i][0],ny=y+d[i][1];
		if(nx<0||ny<0||nx>202||ny>202||fr[nx][ny]||is[nx][ny])continue;
		f1[nx][ny]=i^1;
		fr[nx][ny]=fr[x][y];dfs(nx,ny);
	}
}
void doit(int s)
{
	printf("%c",di[s]);
	s1+=d[s][0];s2+=d[s][1];
	int nx=t1+d[s][0],ny=t2+d[s][1];
	if(nx<0||ny<0||nx>202||ny>202||!is[nx][ny])t1=nx,t2=ny;
}
bool chk1(int x,int y){return ((x<=0||x>=202)||(y<=0||y>=202));}
void fuc1(int s){if(chk1(s1+d[s][0],s2+d[s][1])&&chk1(t1+d[s][0],t2+d[s][1]))doit(s);}
int main()
{
	scanf("%d%d%d%d%d",&s1,&s2,&t1,&t2,&m);
	if(!m){printf("-1\n");return 0;}
	s1+=101;s2+=101;t1+=101;t2+=101;
	while(m--)scanf("%d%d",&a,&b),is[a+101][b+101]=1;
	for(int i=0;i<=202;i++)
	for(int j=0;j<=202;j++)if(!fr[i][j]&&!is[i][j])fr[i][j]=++ct,dfs(i,j);
	if(fr[s1][s2]!=fr[t1][t2]){printf("-1\n");return 0;}
	if(fr[s1][s2]==1)
	{
		while(!((s1<=0||s1>=202)||(s2<=0||s2>=202)))doit(f1[s1][s2]);
		for(int i=0;i<500;i++)doit(3);
		while(!((t1<=0||t1>=202)||(t2<=0||t2>=202)))doit(f1[t1][t2]);
		for(int i=1;i<=345;i++)fuc1(3);for(int i=1;i<=345;i++)fuc1(1);for(int i=1;i<=345;i++)fuc1(3);for(int i=1;i<=345;i++)fuc1(0);for(int i=1;i<=345;i++)fuc1(3);
		if(s1<t1)while(t1>=0)doit(0);else while(t1<=202)doit(1);
		if(s2<t2)while(t2>=0)doit(2);else while(t2<=202)doit(3);
		int tx=-1,ty;
		for(int i=0;i<=202;i++)
		for(int j=0;j<=202;j++)if(tx==-1)
		{
			int sx=i,sy=j;
			if(s1>=t1)sx=202-sx;
			if(s2>=t2)sy=202-sy;
			if(is[sx][sy])tx=sx,ty=sy;
		}
		if(s1<t1)
		{
			while(t2<ty)doit(3);
			while(t2>ty)doit(2);
			while(s1<tx-1)doit(1);
			for(int i=0;i<200;i++)doit(0);
			
		}
		else
		{
			while(t2>ty)doit(2);
			while(t2<ty)doit(3);
			while(s1>tx+1)doit(0);
			for(int i=0;i<200;i++)doit(1);
		}
		if(s2<t2)for(int i=0;i<200;i++)doit(2);
		else for(int i=0;i<200;i++)doit(3);
		if(s2<t2)
		{
			while(t1>tx)doit(0);
			while(t1<tx)doit(1);
			while(s2>ty+1)doit(2);
			while(s2<ty-1)doit(3);
		}
		else
		{
			while(t1>tx)doit(0);
			while(t1<tx)doit(1);
			while(s2>ty+1)doit(2);
			while(s2<ty-1)doit(3);
		}
	}
	else
	{
		while(s1!=t1||s2!=t2)
		{
			memset(fr,0,sizeof(fr));
			memset(f1,0,sizeof(f1));
			fr[t1][t2]=1;dfs(t1,t2);
			int v1=t1,v2=t2;
			while(s1!=v1||s2!=v2)doit(f1[s1][s2]);
		}
	}
}
```

##### Problem 2 Program within a Program

Source: GCJ 11 Worldfinal C

###### Problem

有一个长度为无穷的双向延伸序列，每个位置上的数初始为 $0$。

你有一个机器人，初始位置为 $0$，你希望它移到到 $n$ 并停止。机器人有若干种状态，初始状态为 $0$。

你可以给机器人输入一个程序。这个程序只能包含如下指令：

如果当前机器人所在位置数为 $a$ ，当前机器人状态为 $b$，则下一步：

1. 将当前数设置为 $v$，将自己的状态设置为 $s$，并向左/右移动。
2. 停止。

你需要用不超过 $30$ 条指令完成目标。如果出现了指令中没有的状态则算作失败。

机器人执行指令的次数不能超过 $1.5\times 10^5$ 次。

多组数据

$T\leq 15,n\leq 5000$

$60s,1024MB$

###### Sol

考虑如下思路：

在当前位置附近以某种方式记录 $n$ ，每次将记录值向右移动一位并减一。

考虑从当前位置向右 $\log n$ 位从低到高记录当前的 $n$ 的二进制表示(为了区分空白，可以记录 $1/2$)，只考虑减一操作。

显然减一操作相当于将前缀翻转，直到遇到一个 $1$。如果走到空白还没有遇到 $1$，说明 $n=0$，可以直接在当前位置停止。

因此在不向右移动的情况下，可以记录两个状态，表示当前有没有遇到 $1$，再记录一个状态表示向回走，然后转移类似于：

```
s0 0 -> 1 s0 R
s0 1 -> 0 s1 R
s0 空白 -> 空白 rt L
s1 0 -> 0 s1 R
s1 1 -> 1 s1 R
s1 空白 -> 空白 rt L
rt 0 -> 0 rt L
rt 1 -> 1 rt L
rt 空白 -> 空白 s0 R
```

因为操作次数的限制非常紧，考虑在向右的过程中同时完成右移的操作。

注意到如果只右移可以记录如下状态：

```
t0 0 -> 0 t0 R
t0 1 -> 0 t1 R
t1 0 -> 1 t0 R
t1 1 -> 1 t1 R
```

考虑结合上面两个方式，可以得到：

```
s01 0 -> 1 s01 R
s01 1 -> 0 s10 R
s10 0 -> 0 s10 R
s10 1 -> 0 s11 R
s11 0 -> 1 s10 R
s11 1 -> 1 s11 R
s01 空白 -> 停止
s10 空白 -> 0 rt L
s11 空白 -> 1 rt L
rt 0 -> 0 rt L
rt 1 -> 1 rt L
rt 空白 -> 空白 s R
s 0 -> 空白 s01 R
s 1 -> 空白 s10 R
```

然后考虑初始化，只需要从起点开始依次向左，每次写一位。这部分只需要 $O(\log n)$ 步操作。

状态数 $14+\log_2 n$，操作步数 $2n\log_2 n$，大约为 $13.5\times 10^5$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
int T,n,c1;
struct sth{int x,y,s1,s2,s3;}s[32];
void addstate(int x,int y,int s1,int s2,int s3){s[++c1]=(sth){x,y,s1,s2,s3};}
int main()
{
	scanf("%d",&T);
	for(int t=1;t<=T;t++)
	{
		scanf("%d",&n);c1=0;
		if(n==0)addstate(0,0,-1,-1,-1);
		else
		{
			addstate(1,1,4,0,1);addstate(1,2,2,0,1);
			addstate(2,1,2,1,1);addstate(2,2,3,1,1);addstate(2,0,5,1,-1);
			addstate(3,1,2,2,1);addstate(3,2,3,2,1);addstate(3,0,5,2,-1);
			addstate(4,1,4,2,1);addstate(4,2,2,2,1);addstate(4,0,-1,-1,-1);
			addstate(5,1,5,1,-1);addstate(5,2,5,2,-1);addstate(5,0,1,0,1);
			int ls=5;
			for(int i=12;i>=0;i--)addstate(i==12?0:ls,0,i?ls+1:5,(((n-1)>>i)&1)+1,-1),ls++;
		}
		printf("Case #%d: %d\n",t,c1);
		for(int i=1;i<=c1;i++)
		{
			printf("%d %d -> ",s[i].x,s[i].y);
			if(s[i].s1==-1)printf("R\n");
			else printf("%c %d %d\n",s[i].s3==1?'E':'W',s[i].s1,s[i].s2);
		}
	}
}
```

##### Problem 3 Cycling

在路上了

#### 6.11

##### T1 game

###### Problem

有 $n$ 个人和 $m$ 个任务。每个人初始体力为 $k$。

在一天中，你可以选择一个人，再选择一个 $a$ ，满足当前人的体力大于等于 $a^2$。随后你可以让他在今天内完成 $a$ 个任务，随后这个人体力减少 $a^2$，剩余的人体力减少 $a$。一天只能选一个人。

求最优策略下完成所有任务的最少天数或输出无解。

$n\leq 50,m\leq 1000,k\leq 5\times 10^4$

$1s,512MB$

###### Sol

考虑最后一次做任务的人，他最后体力非负，因此把给他的任务都放到最后一定不会让他做不完，且对其他人更优。

因此存在一个最优解，满足每个人做任务的时间都是连续的若干天。

考虑求 $f_{i,j}$ 表示 $i$ 天做 $j$ 个任务的最小体力，显然最优解是分成若干 $\lfloor\frac ji\rfloor.\lceil\frac ji\rceil$，可以 $O(1)$ 求出。

因此可以求出 $g_{i,j}$ 表示当前还有 $i$ 的体力，做 $j$ 个任务的最少时间。

最后设 $dp_{i,j}$ 表示当前做了 $j$ 个任务，还剩 $i$ 个人的最优时间，则：
$$
dp_{i,j}=\max_l dp_{i-1,j+l}+g_{k-j,l}
$$
复杂度 $O(nm^2+mk)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
int f[1050][40050],dp[1055][52],n,m,k;
int main()
{
    freopen("game.in","r",stdin);
	freopen("game.out","w",stdout);
	scanf("%d%d%d",&n,&m,&k);
	for(int i=1;i<=m;i++)for(int j=0;j<=k;j++)f[i][j]=1e9;
	for(int i=1;i<=m;i++)
	for(int j=i;j>=1;j--)
	{
		int s1=i/j,s2=s1+1,c2=i%j,c1=j-c2;
		int vl=s1*s1*c1+s2*s2*c2;
		if(vl<=k)f[i][vl]=j;
	}
	for(int i=1;i<=m;i++)for(int j=1;j<=k;j++)f[i][j]=min(f[i][j],f[i][j-1]);
	for(int i=0;i<=n;i++)for(int j=0;j<m;j++)dp[j][i]=1e8;
	for(int j=m-1;j>=0;j--)for(int l=j+1;l<=m;l++)for(int i=1;i<=n;i++)dp[j][i]=min(dp[j][i],dp[l][i-1]+f[l-j][k-j]);
	if(dp[0][n]>1e7)dp[0][n]=-1;
	printf("%d\n",dp[0][n]);
}
```

##### T2 matrix

###### Problem

给一个 $n$ 行 $m$ 列的 $01$ 矩阵，你可以将矩阵的列重新排列。

你需要满足对于矩阵的每一行，这一行的所有 $1$ 形成一个区间。

输出一个方案或输出无解。

$n,m\leq 1500$

$2s,512MB$

###### Sol

对于任意两行，可以求出这两行中 $1$ 的交，这个的大小即为最后这两行的 $1$ 所在区间的交的大小。

称一行对应的区间为这一行最后所有 $1$ 所在的区间。

称两行是相关的当且仅当两行对应的区间不存在包含关系且不相离。

可以发现，对于相关的两行，如果确定了一行的区间位置，则另外一行的区间位置只有两种(在左侧或右侧)。

同时，如果确定了两个相关行的区间的相对位置，考虑一个与它们中的至少一个相关的行，通过这两行的限制，可以得到这一行唯一的合法位置。~~证明省略~~

因此，如果把相关看成连通关系，考虑一个连通块内的所有行的相对顺序。在只有两个区间时，考虑钦定一个在另外一个前，之后可以推出所有连通的区间的相对位置。

因为反过来钦定相当于整体翻转，且无论是否翻转不影响答案，因此可以任选一种做。

考虑区间不连通的情况，此时一定是一个连通块内的区间被另外一个连通块内的一个小区间完全包含。

考虑按照长度从大到小做区间，对于一个连通块，考虑找到之前包含它的小区间并放进去即可。可以使用hash加速这个过程。

预处理复杂度 $O(\frac{n^2m}{32})$，后面所有部分复杂度 $O(nm)$。总复杂度 $O(\frac{n^2m}{32})$

更好的做法是使用PQ-tree。

###### Code

```cpp
#include<cstdio>
#include<bitset>
#include<algorithm>
#include<vector>
#include<queue>
using namespace std;
#define N 1510
#define mod 998244853
int n,m,sz[N],vl[N][N],fa[N],f2[N],sz1[N],lb[N],is[N][N],fu[N],v1[N],fr[N],fg[N][N],v2[N],as[N],f3[N];
char s[N][N],s2[N][N];
bitset<N> f[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
bool cmp(int a,int b){return sz1[a]>sz1[b];}
int check(int l,int r,int l1,int r1){int as=min(r,r1)-max(l,l1)+1;return as>=0?as:0;}
void solve(vector<int> st)
{
	int f1=st[0],fg1=1;
	for(int i=0;i<st.size();i++)lb[st[i]]=-1,fr[st[i]]=-1;
	lb[f1]=0;
	queue<int> sr;
	for(int i=0;i<st.size();i++)if(fg[f1][st[i]]&&(st[i]!=f1))
	{
		lb[st[i]]=sz[f1]-vl[f1][st[i]];
		fr[st[i]]=f1;fr[f1]=st[i];
		sr.push(st[i]);sr.push(f1);
		break;
	}
	while(!sr.empty()&&fg1)
	{
		int tp=sr.front();sr.pop();
		for(int i=0;i<st.size();i++)if(fg[tp][st[i]])
		{
			int v1=fr[tp],v2=tp,v3=st[i];
			int l1=lb[v2]+sz[v2]-vl[v2][v3],l2=lb[v2]+vl[v2][v3]-sz[v3];
			if(lb[v3]==-1)fr[v3]=v2,sr.push(v3);
			if(check(lb[v1],lb[v1]+sz[v1]-1,l1,l1+sz[v3]-1)==vl[v1][v3])fg1&=lb[v3]==l1||lb[v3]==-1,lb[v3]=l1;
			else if(check(lb[v1],lb[v1]+sz[v1]-1,l2,l2+sz[v3]-1)==vl[v1][v3])fg1&=lb[v3]==l2||lb[v3]==-1,lb[v3]=l2;
			else fg1=0;
		}
	}
	int mn=0;for(int i=0;i<st.size();i++)mn=min(mn,lb[st[i]]);
	for(int i=0;i<st.size();i++)lb[st[i]]=lb[st[i]]-mn;
	if(fg1)return;
	printf("-1\n");exit(0);
}
int main()
{
    freopen("matrix.in","r",stdin);
	freopen("matrix.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%s",s2[i]+1),fa[i]=f2[i]=i;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(s2[i][j]=='1')sz1[i]++;
	sort(f2+1,f2+n+1,cmp);
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)s[i][j]=s2[f2[i]][j],f3[f2[i]]=i;
	fu[0]=1;for(int i=1;i<=n;i++)fu[i]=7ll*fu[i-1]%mod;
	for(int i=1;i<=n;i++)for(int j=1;j<=m;j++)if(s[i][j]=='1')f[i].set(j,1),sz[i]++,v1[j]=(v1[j]+fu[i])%mod;
	for(int i=1;i<=n;i++)for(int j=1;j<=n;j++)
	{
		if(i<=j)vl[i][j]=(f[i]&f[j]).count();else vl[i][j]=vl[j][i];
		if(vl[i][j]&&vl[i][j]<sz[i]&&vl[i][j]<sz[j])fa[finds(j)]=finds(i),fg[i][j]=1;
		if(vl[i][j]==sz[j]&&(sz[i]>sz[j]||i<j))is[j][i]=1;
	}
	for(int i=1;i<=n;i++)if(!lb[i])
	{
		vector<int> fu1;
		for(int j=1;j<=n;j++)if(finds(j)==finds(i))fu1.push_back(j);
		solve(fu1);
		int as=0,fu3=0;
		for(int j=1;j<=n;j++){int fg=1;for(int k=0;k<fu1.size();k++)fg&=is[fu1[k]][j];fu3=(fu3+fg*fu[j])%mod;}
		for(int j=1;j<=n;j++)if(v2[j]==fu3){as=j;break;}
		for(int k=0;k<fu1.size();k++)lb[fu1[k]]+=as;
		for(int k=0;k<fu1.size();k++)for(int j=lb[fu1[k]];j<lb[fu1[k]]+sz[fu1[k]];j++)v2[j]=(v2[j]+fu[fu1[k]])%mod;
	}
	for(int i=1;i<=m;i++)for(int j=1;j<=m;j++)if(v2[i]==v1[j]){as[i]=j,v1[j]=-1;break;}
	for(int j=1;j<=m;j++)printf("%d ",as[j]);
}
```

##### T3 counting

###### Problem

给一个 $n$ 个点 $m$ 条边图，每个点有一个权值 $a_i$。给定 $C$，求有多少个整数序列 $b_{1,2,...,n}$ 满足：

1. $0\leq b_i\leq a_i$
2. 对于每一条边 $(u,v)$，$b_u\neq b_v$
3. $\oplus_{i=1}^nb_i=C$

答案模 $998244353$。

$n\leq 17$

$2s,512MB$

###### Sol

考虑暴力做法，枚举一个边集容斥，相当于要求这个边集中每条边相邻两个点相等。可以发现这会将图划分成若干个连通块，每个连通块内所有点点权相等。

考虑计算一个连通块的容斥系数，即所有只连出这个连通块的方案的容斥系数和。设这个值为 $f_S$。

考虑对 $S$ 连通的条件容斥，首先考虑 $S$ 内部的边任意连的方案。设 $v_S$ 表示 $S$ 的导出子图的边数，则任意连的方案数等于 $[v_S=0]$。

然后容斥，记 $l_S$ 表示 $S$ 中的最小元素，枚举 $l_S$ 实际上所在的连通块，有：

$$
f_S=[v_S=0]-\sum_{T\subset S,T\neq S,l_S\in T}f_T[v_{S-T}=0]
$$

这样的复杂度为 $O(3^n)$

考虑另外一个暴力，枚举最后得到的连通块，即点集的一个划分。则容斥系数为每个连通块系数的乘积。

考虑计算此时的方案数。对于一个连通块，可以发现它内部取值的上界为连通块内每个点上界的最小值。

如果一个连通块大小为偶数，则它任意填不改变异或和，因此可以直接乘上一个最小值+1的系数。

否则，可以看成只有一个数，它的上界等于整个点集上界的最小值。

那么方案数相当于一个与原问题类似的问题，但此时没有2限制。

此时是经典问题，枚举在哪一位上脱离限制，在这一位上做 $dp$。具体细节可以参考6.9 T3。

注意到可以看成对于一个大小为奇数的连通块只保留上限最小的点，因此这时剩下的点一定是一个子集，对于每个子集暴力求的复杂度为 $O(n*2^n*\log v)$

考虑使用 $dp$ 的方式优化枚举集合划分，设 $dp_{S,T}$ 表示当前还没有被划分的点集为 $S$，当前留下来的点集为 $T$，划分 $S$ 后所有方案的权值和。

转移考虑枚举 $l_S$ 所在的集合，直接转移即可。

注意到如果将点按照 $a_i$ 从小到大排序，则留下来的点一定是 $l_S$，因此排序后 $dp$ 时，一定有 $S$ 中最大的元素小于 $T$ 中最小的元素。

此时可以发现状态数为 $O(n*2^n)$，考虑计算枚举子集的复杂度。

考虑枚举 $S\cup T$，设 $|S\cup T|=k$，则枚举它的子集的复杂度为 $\sum_{i=0}^k2^i=O(2^k)$，因此总的枚举子集复杂度为 $O(3^n)$。

复杂度 $O(3^n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 132001
#define M 18
#define ll long long
#define mod 998244353
int n,m,a,b,dp[N][M],ct[N],v1[N],is[N],vl[N],hb[N],tp[M],id[M],vis[N][M];
ll c,v[N];
bool cmp(int a,int b){return v[a]>=v[b];}
int solve(int s)
{
	int as=0,rv=1;
	for(int i=1;i<=60;i++)rv=1ll*rv*(mod+1)/2%mod;
	for(int i=60;i>=0;i--)
	{
		int s1=1,s2=0,su=0,tp=1,fg=(c>>i)&1;
		for(int j=1;j<=n;j++)if((s>>j-1)&1)
		{
			ll v1,v2,f1=(v[j]>>i)&1;
			if(f1)su^=1,v2=1ll<<i,v1=v[j]&((1ll<<i)-1);
			else v1=v[j]&((1ll<<i)-1),v2=0;
			v1++;
			tp=v1%mod*tp%mod;
			int t1=(v1%mod*s1+v2%mod*s2)%mod,t2=(v1%mod*s2+v2%mod*s1)%mod;
			s1=t1,s2=t2;
		}
		s1=(s1+mod-tp)%mod;
		if(su)swap(s1,s2);
		if(fg)as=(as+1ll*s2*rv)%mod;else as=(as+1ll*s1*rv)%mod;
		rv=2*rv%mod;
		if(su^fg)return as;
	}
	return (as+1)%mod;
}
int dfs(int s,int k)
{
	if(!k)return v1[s];
	if(vis[s][k])return dp[s][k];
	vis[s][k]=1;
	int v1=s&((1<<k)-1);
	for(int i=v1;i>=(1<<k-1);i=(i-1)&v1)
	{
		int nt=s^i,nt2=hb[v1^i]+1,st=vl[i];
		if(ct[i]&1)nt^=1<<k-1;else st=(v[k]+1)%mod*st%mod;
		dp[s][k]=(dp[s][k]+1ll*st*dfs(nt,nt2))%mod;
	}
	return dp[s][k];
}
int main()
{
	freopen("counting.in","r",stdin);
	freopen("counting.out","w",stdout);
	scanf("%d%d%lld",&n,&m,&c);
	for(int i=1;i<=n;i++)scanf("%lld",&v[i]),tp[i]=i;
	sort(tp+1,tp+n+1,cmp);sort(v+1,v+n+1);for(int i=1;i<=n;i++)id[tp[i]]=i;
	for(int i=1;i*2<=n;i++)swap(v[i],v[n-i+1]);
	for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),is[(1<<id[a]-1)|(1<<id[b]-1)]=1;
	for(int i=1;i<1<<n;i++)for(int j=1;j<=n;j++)if(i&(1<<j-1))is[i]|=is[i^(1<<j-1)];
	hb[0]=-1;
	for(int i=1;i<1<<n;i++)
	{
		hb[i]=hb[i>>1]+1;ct[i]=ct[i>>1]+(i&1);
		vl[i]=!is[i];
		for(int j=(i-1)&i;j>=(1<<hb[i]);j=(j-1)&i)if(!is[i^j])vl[i]=(vl[i]+mod-vl[j])%mod;
	}
	for(int i=0;i<1<<n;i++)v1[i]=solve(i);
	printf("%d\n",dfs((1<<n)-1,n));
}
```



#### 6.12

##### PQ tree

大量细节警告

给定 $n$ 以及 $m$ 个 $\{1,2,...,n\}$ 的子集 $S_1,...,S_m$，构造一个 $n$ 阶排列 $p$ 满足：

对于任意一个 $S_i$ ，$p$ 中属于 $S_i$ 的元素组成一段连续的区间。

PQ tree可以在 $O(nm)$ 的时间内求出所有解的形式，因此可以求出一个解/解的数量/字典序最小的解(?)

PQ tree的形式析合树类似。在PQ tree中，一共有 $n$ 个叶节点，分别表示排列的元素 $1,2,...,n$，对于一个非叶节点，它子树内的点一定在答案中排成连续一段，这种点有两种形式：

1. P类点，这类点表示它的所有子树之间可以任意排列。
2. Q类点，这类点表示它的子树只能顺序排列或倒序排列。

显然在没有任何限制时，可以看成一个P类点作为根，它的儿子为所有的叶子。

考虑加入一个限制 $S_i$ 时，这个树的变化：

称在 $S_i$ 中的叶节点是黑的，不在 $S_i$ 中的叶节点是白的。对于一个非叶节点：

如果它子树中所有叶节点都是黑的，则它是黑的。如果它子树中所有叶节点都是白的，则它是白的。否则称它是灰的。

考虑从根节点开始考虑整个树，如果当前点只有一个儿子不是白色，显然这个限制只会影响这个子树内，不会影响其他子树。因此可以只考虑这个儿子内。

如果这个点的儿子有若干个白色的点和若干个黑色的点，则可以发现限制相当于黑色的子树必须相邻。

如果当前点为Q类点，则如果当前黑色的点不形成一段显然无解，否则这个限制一定被满足。

如果当前点为P类点，则相当于强制黑色的子树连续，然后再任意排列。

设当前点为 $x$，可以新建一个P类点 $y$，$y$ 的儿子为原先 $x$ 的所有黑色子树，$x$ 的儿子为原先 $x$ 的所有白色子树以及 $y$。

考虑加上灰色点的情况。如果有三个或者更多的灰色点显然无解，否则一定只能在黑色段的两侧分别最多放一个灰色点，且需要满足这个灰色点内部所有黑色的点在最右侧/最左侧。

此时考虑新建一个Q类点 $z$，设灰色点为 $a_1,a_2$，则可以考虑如下操作：

$x$ 的儿子为原先 $x$ 的所有白色子树以及 $z$，$z$ 的儿子依次为：$a_1$ 子树内白色部分，$a_1$ 子树内黑色部分，$y$，$a_2$ 子树内黑色部分，$a_2$ 子树内白色部分，$y$ 的儿子为原先 $x$ 的所有黑色子树。

因为Q类点可以翻转，因此这样可以表示所有灰色点分布的情况。只要能将一个灰色点按照黑白分开且不改变其余性质即可。

因此相当于对于一个灰点，要求它内部黑白分开，求加上这个限制后对应的PQ tree。

如果根节点有大于等于 $2$ 个灰点显然无解，否则分类讨论：

如果当前点是Q类点，则有解当且仅当当前是白-灰-黑。此时所有白点必定有序，所有黑点必定有序且它们和灰点也有序。考虑两个序列 $v_1,v_2$，将所有的白色儿子按顺序放在 $v_1$ 最右侧，所有黑色儿子按顺序放在 $v_2$ 最左侧，然后再对灰儿子递归做这个过程。

如果当前点是P类点，则考虑新建两个Q类点 $s_1,s_2$，$s_1$ 的儿子为这个点的所有白色儿子，$s_2$ 的儿子为这个点的所有黑色儿子，然后将 $s_1$ 放在 $v_1$ 最右侧，$s_2$ 放在 $v_2$ 最左侧。

可以发现，如果此时建一个Q类点，Q类点的儿子为 $v_1$ 拼接 $v_2$ 的结果。则可以证明，这样得到的一个PQ tree为原PQ tree加上这个限制的PQ tree。~~证明过程较为显然，留作练习~~

因此对一个灰点做这样的分裂，然后将上面的白色部分换为 $v_1$，黑色部分换为 $v_2$ 即可。另外一个方向的分裂可以先按照这个方向分裂，再翻转即可。

可以发现上面过程即可得到现在的PQ tree或者得到无解。

可以发现，对于一个儿子数为 $1$ 的点，把它缩掉不影响树的性质。因此在上面的过程后，可以再dfs一次，删去没有叶子的节点并缩掉儿子数为 $1$ 的点。这样即可保证每个点儿子数大于等于 $2$，即点数不超过 $O(m)$。

复杂度 $O(nm)$

###### Code(CF243E)

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 1550
int n,m,ct,ty[N*4],cl[N*4],s1[N*4],is[N*4],sz[N*4],id[N*4],ty2[N*4],as[N],c1;
char s[N][N];
vector<int> sn[N*4],tp[N],sn2[N*4];
void dfs(int x)
{
	if(x<=m)return;
	cl[x]=0;
	for(int i=0;i<sn[x].size();i++)dfs(sn[x][i]),cl[x]|=cl[sn[x][i]];
}
void dfs2(int x)
{
	if(x<=m){sz[x]=1;return;}sz[x]=0;
	for(int i=0;i<sn[x].size();i++)
	{
		while(sn[x][i]>m&&sn[sn[x][i]].size()==1)is[sn[x][i]]=0,sn[x][i]=sn[sn[x][i]][0];
		dfs2(sn[x][i]),sz[x]+=sz[sn[x][i]];
	}
}
void failed(){printf("NO\n");exit(0);}
void solvel(int x,vector<int> &v1,vector<int> &v2)
{
	if(!ty[x])
	{
		int ls=sn[x][0],rs=sn[x].back();
		if((cl[rs]==2&&cl[ls]!=2)||(cl[rs]==3&&cl[ls]==1))reverse(sn[x].begin(),sn[x].end());
		vector<int> s1;
		int fg=1;
		for(int i=0;i<sn[x].size();i++)
		if(cl[sn[x][i]]==2)
		{
			if(!fg)failed();
			v1.push_back(sn[x][i]);
		}
		else if(cl[sn[x][i]]==3)
		{
			if(!fg)failed();fg=0;
			solvel(sn[x][i],v1,v2);
		}
		else
		{
			if(fg)fg=0;
			v2.push_back(sn[x][i]);
		}
	}
	else
	{
		int f1=++ct,f2=++ct,c1=0;ty[f1]=ty[f2]=1;
		for(int i=0;i<sn[x].size();i++)
		if(cl[sn[x][i]]==2)sn[f1].push_back(sn[x][i]);
		else if(cl[sn[x][i]]==1)sn[f2].push_back(sn[x][i]);
		v1.push_back(f1);
		for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==3)
		{
			c1++;if(c1>1)failed();
			solvel(sn[x][i],v1,v2);
		}
		v2.push_back(f2);
	}
	sn[x].clear();
}
void solver(int x,vector<int> &v1,vector<int> &v2)
{
	vector<int> s1,s2;
	solvel(x,s1,s2);
	reverse(s1.begin(),s1.end());reverse(s2.begin(),s2.end());
	for(int i=0;i<s1.size();i++)v1.push_back(s1[i]);
	for(int i=0;i<s2.size();i++)v2.push_back(s2[i]);
}
void solve(int x)
{
	if(cl[x]!=3)return;
	int c1=0,c2=0,ls=0;
	for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==3)c2++;else if(cl[sn[x][i]]==2)c1++;
	if(c2>2)failed();
	if(!c1&&c2==1)
	{
		for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==3)solve(sn[x][i]);
		return;
	}
	if(!ty[x])
	{
		int fg=0;
		vector<int> s1,s3,s2;
		for(int i=0;i<sn[x].size();i++)
		if(cl[sn[x][i]]==1)
		{
			if(fg==1)fg=2;
			if(fg==0)s1.push_back(sn[x][i]);else s3.push_back(sn[x][i]);
		}
		else if(cl[sn[x][i]]==2)
		{
			if(fg==0)fg=1;if(fg==2)failed();
			s2.push_back(sn[x][i]);
		}
		else
		{
			if(fg==0)fg=1,solver(sn[x][i],s2,s1);
			else if(fg==1)fg=2,solvel(sn[x][i],s2,s3);
			else failed();
		}
		for(int i=0;i<s2.size();i++)s1.push_back(s2[i]);for(int i=0;i<s3.size();i++)s1.push_back(s3[i]);
		sn[x]=s1;return;
	}
	int st=++ct,sb=++ct,sw=++ct;ty[st]=0;ty[sw]=ty[sb]=1;
	vector<int> s1,s2,s3,s4;
	for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==1)s2.push_back(sn[x][i]);else if(cl[sn[x][i]]==2)sn[sb].push_back(sn[x][i]);
	for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==3&&!ls)solver(sn[x][i],sn[st],s3),ls=sn[x][i];
	sn[st].push_back(sb);
	for(int i=0;i<sn[x].size();i++)if(cl[sn[x][i]]==3&&ls!=sn[x][i])solvel(sn[x][i],sn[st],s4);
	for(int i=0;i<sn[st].size();i++)s3.push_back(sn[st][i]);for(int i=0;i<s4.size();i++)s3.push_back(s4[i]);
	sn[st]=s3;
	s2.push_back(st);sn[x]=s2;
}
void dfs3(int x)
{
	if(x<=m){as[++c1]=x;return;}
	for(int i=0;i<sn[x].size();i++)dfs3(sn[x][i]);
}
int main()
{
	scanf("%d",&n);m=n;
	for(int i=1;i<=n;i++){scanf("%s",s[i]+1);for(int j=1;j<=m;j++)if(s[i][j]=='1')tp[i].push_back(j);}
	ty[m+1]=1;for(int i=1;i<=m;i++)sn[m+1].push_back(i);
	ct=m+1;
	for(int i=1;i<=n;i++)
	{
		int c1=0;
		for(int j=1;j<=m;j++)cl[j]=1;
		for(int j=0;j<tp[i].size();j++)cl[tp[i][j]]=2;
		dfs(m+1);solve(m+1);
		for(int j=1;j<=ct;j++)is[j]=1,sz[j]=0;
		dfs2(m+1);
		for(int j=1;j<=ct;j++)if(!sz[j])is[j]=0;else if(is[j])
		{
			ty2[++c1]=ty[j];id[j]=c1;
			for(int l=0;l<sn[j].size();l++)if(sz[sn[j][l]]>0)sn2[c1].push_back(sn[j][l]);
		}
		for(int j=1;j<=ct;j++)sn[j].clear();
		for(int j=1;j<=c1;j++)ty[j]=ty2[j],sn[j]=sn2[j],sn2[j].clear();
		for(int j=1;j<=c1;j++)for(int l=0;l<sn[j].size();l++)sn[j][l]=id[sn[j][l]];
		ct=c1;
	}
	dfs3(m+1);
	printf("YES\n");
	for(int i=1;i<=n;i++,printf("\n"))for(int j=1;j<=m;j++)printf("%c",s[i][as[j]]);
}
```

##### 0526T2 Color

###### Problem

给一棵 $n$ 个点的树。

选择一个点集 $S$，一个边集 $T$。称这种方式是合法的当且仅当对于 $T$ 中的每条边，这条边两侧部分与 $S$ 的交大小不同。

一种合法方式的贡献为 $m^{|T|}$，不合法方式的贡献为 $0$。

求出所有选择 $S,T$ 的方案的贡献和，模 $998244353$。

$n\leq 8\times 10^4,m\leq 10^9$

$5s,512MB$

###### Sol

djq:这个做法看起来就非常难写，卷阶乘倒数那么好写~~我看不懂，但我大受震撼~~

考虑固定 $S$ 后，哪些边是不合法的。显然，对于 $S\neq \emptyset$ 的情况，不合法的边一定连通且形成一条链。设不合法的边集大小为 $k$，则总的贡献显然是 $(m+1)^{n-1-k}$。

首先特判 $m=998244352$ 的情况，此时 $S$ 为空集是一种情况，对于 $S\neq \emptyset$ 的情况，显然只有当树是一条链且选了两个端点的时候才会使所有边不合法。因此一条链答案为 $2$ ，否则为 $1$。

对于剩余的情况，可以看成初始贡献为 $(m+1)^{n-1}$，每一条不合法的边让贡献乘 $\frac 1{m+1}$。

考虑先计算存在不合法边的方案数（不考虑 $S\neq \emptyset$），再计算这些方案的贡献和，即可求出答案。

考虑枚举不合法的链 $(u,v)$，此时需要满足：（以 $(u,v)$ 路径上的某个位置为根）

1. 所有选择的点都在 $u$ 或 $v$ 的子树内，且在两个子树内的点数量相等。
2. 不存在 $u$ 子树内的其它点 $w$ ，满足 $u$ 子树内选择的点都在 $w$ 子树内。 $v$ 同理。

考虑在满足条件 $2$ 的情况下，计算 $u$ 子树内选若干个点的方案数。把这个看成生成函数，设 $sz_u$ 表示 $u$ 的子树大小，容斥后显然有：
$$
F_u(x)=(1+x)^{sz_u}-\sum_{i\in son_u}(1+x)^{sz_i}
$$
记 $F_{u,v}(x)$ 表示以 $v$ 为根时 $u$ 的上述生成函数，$S(F(x)=\sum f_ix^i,G(x)=\sum g_ix^i)=\sum f_ig_i$，则贡献和为：
$$
\sum_{u\neq v}S(F_{u,v}(x),F_{v,u}(x))*(\frac 1{m+1})^{dis(u,v)}
$$
以一个点为根，考虑分成两种情况计算：

1.  $u,v$ 不是祖先关系

此时考虑枚举LCA，设LCA为 $l$。如果枚举 $u,v$ 分别在哪个儿子内，则将 $(\frac 1{m+1})^{dis(u,v)}$ 拆成 $(\frac 1{m+1})^{dis(u,l)}*(\frac 1{m+1})^{dis(v,l)}$ 后，显然 $S$ 具有类似可加性的形式，因此两侧独立，可以两侧分别求和。

因此设：
$$
G_u(x)=\sum_{v\in subtree\ of \ u,v\neq u}F_v(x)*(\frac 1{m+1})^{dis(u,v)}
$$
则这部分的答案为：
$$
\sum_{u}\sum_{i,j\in son_u,i<j}S(G_i(x),G_j(x))
$$
注意到 $G$ 可以写成类似 $dp$ 的转移：
$$
G_u(x)=(1+x)^{sz_u}+\sum_{i\in son_u}G_v(x)*(\frac 1{m+1}-1)
$$
显然生成函数的次数等于子树大小，因此考虑轻重链剖分优化转移及计算答案的过程。

对于一条重链，考虑先算出它所有轻儿子的 $G$ 和贡献。考虑这条重链上的贡献，第一种情况是两个轻儿子父亲为重链上同一个点，此时可以直接 $O(sz_v)$ 考虑这个子树的生成函数并计算。

考虑第二种情况，设重链上的点为 $1,2,...,n$，可以写成如下式子：
$$
G_i(x)=(1+x)^{sz_i}+G_{i-1}(x)*\frac{-m}{m+1}+P_i(x)\\
as=\sum S(Q_i(x),G_i(x))
$$
考虑把 $(1+x)^{sz_i}$ 放入分治过程，称为 $H_i(x)$，这样每次相当于给它乘上一个轻儿子子树大小级别的多项式，因此转移中的多项式长度总和为 $O(n\log n)$。

进行分治，计算 $G$ 只需要将其表示为 $G_r(x)=G_{l-1}(x)*v+H_{l-1}(x)*A(x)+B(x),H_{r}(x)=H_{l-1}x*C(x)$ 的形式，每次乘即可。对于算答案的过程，可以求出只考虑 $[l,r]$ 区间的计算答案时， $G_{l-1}(x),H_{l-1}(x)$ 中 $x^i$ 项的贡献。合并时先算左侧的常数对右侧的贡献，然后考虑算整体的贡献次数，相当于右侧的贡献次数与左侧的转移次数做翻转卷积，再与左侧系数相加。

复杂度 $O(n\log^3 n)$

2.  $u,v$ 是祖先关系

不妨设 $u$ 为祖先，对于每条重链，考虑 $u$ 在这条重链上的答案。显然可以将一个点的所有轻儿子一起考虑，相当于所有轻儿子的 $F(x)$ 的和。

此时的答案形式为：
$$
G_i(x)=(1+x)^{sz_i}+G_{i-1}(x)*\frac{-m}{m+1}+P_i(x)\\
as=\sum S(G_{i-1}(x),F_{i+1,i}(x))\\
F_{i+1,i}(x)=(1+x)^{n-sz_i+1}-\sum_{u\in son_i,u\neq i-1}(1+x)^{sz_u}-(1+x)^{n-sz_{i+1}+1}
$$
可以将 $F_{i+1,i}(x)$ 中的第二部分单独用上面的方式做，但因为另外两部分生成函数次数很大，这里不能直接做上面的做法。

注意到这个逆卷积可以拆开，即 $S(P(x),(1+x)^{a+b})=S(S(P(x),(1+x)^b),(1+x)^{a})$

设 $R_i(x)$ 表示当前前 $i$ 个位置的答案可以被表示成 $S(R_i(x),(1+x)^{n-sz_{i+1}+1})$ 的形式，则可以写成：
$$
R_i(x)=S(R_{i-1}(x)+G_{i-1}(x),(1+x)^{sz_{i+1}-sz_i})+G_{i-1}(x)
$$
如果把 $S$ 看成乘上一个 $F(\frac 1x)$，则这个东西也可以放入分治FFT。因为转移式保证一定是先做正向卷积，然后做翻转卷积，因此直接看成乘 $F(\frac 1x)$，不会出现翻转卷积中次数变为负数的式子影响答案的情况。

此时可以把 $G,H,R$ 的转移看成一个 $3\times 3$ 加上常数的矩阵，其中每个位置是一个多项式。对这个分治FFT即可。

实际上这里和上面的分治FFT大体相同，可以把两个放在一起做。

复杂度 $O(n\log^3 n)$

算方案数显然可以把上面的 $\frac 1{m+1}$ 去掉再做一次，但这样常数极大非常容易过不去。

因为不合法的边一定是一条链，因此可以做类似点减边容斥。

对于一条边的情况，相当于在两侧选出个数相同的数，如果两边为 $a,b$，则显然方案数为 $C_{a+b}^b$。

对于一个点的情况，先把轻儿子相关的暴力做，然后重儿子和父亲部分可以也用一个组合数。

这样这一步复杂度 $O(n\log n)$。

复杂度 $O(n\log^3 n)$，转移有 $10$ 个多项式，常数极大。

###### Code

```cpp
//stO djq Orz
//为啥djq那么快啊/ll/ll/ll
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 263003
#define mod 998244353
int n,m,a,b,head[N],cnt,d[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;d[f]++;d[t]++;}
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
int rev[N*2],g[2][N*2],ntt[N],f1[N],f2[N],fr[N],ifr[N];
void init(int d)
{
	fr[0]=1;for(int i=1;i<=1<<d;i++)fr[i]=1ll*fr[i-1]*i%mod;
	ifr[1<<d]=pw(fr[1<<d],mod-2);for(int i=1<<d;i>=1;i--)ifr[i-1]=1ll*ifr[i]*i%mod;
	for(int i=2;i<=1<<d;i<<=1)for(int j=0;j<i;j++)rev[i+j]=(rev[i+(j>>1)]>>1)+(j&1)*(i>>1);
	for(int t=0;t<2;t++)
	for(int i=2;i<=1<<d;i<<=1)
	{
		int tp=pw(3,(mod-1)/i);
		if(!t)tp=pw(tp,mod-2);
		int st=1;
		for(int j=0;j<i>>1;j++)g[t][i+j]=st,st=1ll*st*tp%mod;
	}
}
void dft(int s,int *a,int t)
{
	for(int i=0;i<s;i++)ntt[rev[s+i]]=a[i];
	for(int i=2;i<=s;i<<=1)
	for(int j=0;j<s;j+=i)
	for(int k=0;k<i>>1;k++)
	{
		int v1=ntt[j+k],v2=1ll*ntt[j+k+(i>>1)]*g[t][i+k]%mod;
		ntt[j+k]=(v1+v2)%mod;ntt[j+k+(i>>1)]=(v1-v2+mod)%mod;
	}
	int inv=t?1:pw(s,mod-2);
	for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
vector<int> polyadd(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	vector<int> c;
	for(int i=0;i<s1||i<s2;i++)c.push_back(((i<s1?a[i]:0)+(i<s2?b[i]:0))%mod);
	return c;
}
vector<int> polymul(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	int l=1;while(l<s1+s2)l<<=1;
	for(int i=0;i<l;i++)f1[i]=f2[i]=0;
	for(int i=0;i<s1;i++)f1[i]=a[i];for(int i=0;i<s2;i++)f2[i]=b[i];
	dft(l,f1,1);dft(l,f2,1);for(int i=0;i<l;i++)f1[i]=1ll*f1[i]*f2[i]%mod;dft(l,f1,0);
	vector<int> c;
	for(int i=0;i<s1+s2-1;i++)c.push_back(f1[i]);
	return c;
}
vector<int> polymul(vector<int> a,int b)
{
	int s1=a.size();
	for(int i=0;i<s1;i++)a[i]=1ll*a[i]*b%mod;
	return a;
}
vector<int> polyrev(vector<int> a,vector<int> b)
{
	int s1=a.size(),s2=b.size();
	reverse(b.begin(),b.end());
	vector<int> c=polymul(a,b),d;
	for(int i=0;i<s1;i++)if(c.size()>i+s2-1)d.push_back(c[i+s2-1]);
	return d;
}
int sz[N],sn[N],su[N];
void dfs0(int u,int fa)
{
	sz[u]=1;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),sz[u]+=sz[ed[i].t],sn[u]=sz[sn[u]]<sz[ed[i].t]?ed[i].t:sn[u];
}
int vl,as,tp[N];
vector<int> fu[N],t1[N];
vector<pair<int,int> > sn1[N];
struct sth{vector<int> s21,s33,s23,s13,a1,a2,c1,c3;int sz,s11;};
vector<int> doit2(vector<int> a,int b){vector<int> as;for(int i=1;i<=b;i++)as.push_back(0);for(int i=0;i<a.size();i++)as.push_back(a[i]);return as;}
vector<int> doit3(vector<int> a,int b){vector<int> as;for(int i=b;i<a.size();i++)as.push_back(a[i]);return as;}
sth justdoit(int l,int r)
{
	if(l==r)
	{
		int s1=tp[l],sz1=1;
		vector<int> su,s2;
		for(int i=0;i<sn1[s1].size();i++)
		{
			int t=sn1[s1][i].second;
			sz1+=sz[t],su=polyadd(su,fu[t]);
			vector<int> s3;
			for(int i=0;i<=sz[t];i++)s3.push_back(1ll*fr[sz[t]]*ifr[i]%mod*ifr[sz[t]-i]%mod);
			s2=polyadd(s2,s3);
		}
		vector<int> v1;
		for(int i=0;i<=sz1;i++)v1.push_back(1ll*fr[sz1]*ifr[i]%mod*ifr[sz1-i]%mod);
		sth fu;
		fu.sz=sz1;
		fu.s21=v1;fu.s21[0]--;fu.s11=vl;
		fu.c1=polyadd(polymul(su,vl),polymul(s2,mod-1));
		if(fu.c1.size())fu.c1[0]=0;
		fu.s33=v1;fu.s13=polymul(v1,vl);fu.s13.pop_back();
		fu.a1=polymul(fu.c1,vl);
		return fu;
	}
	int lb=l,rb=r-1,v1=(l+r)>>1;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(su[mid]-su[l-1]>=(su[r]-su[l-1])/2)v1=mid,rb=mid-1;
		else lb=mid+1;
	}
	sth sl=justdoit(l,v1),sr=justdoit(v1+1,r),fu;
	fu.sz=sl.sz+sr.sz;
	fu.s33=polymul(sl.s33,sr.s33);fu.s11=1ll*sl.s11*sr.s11%mod;
	fu.s21=polyadd(polymul(sl.s21,sr.s11),polymul(sl.s33,sr.s21));
	fu.s13=polyadd(polymul(sl.s13,sr.s33),polymul(doit2(sr.s13,sl.sz),sl.s11));
	fu.s23=polyadd(polymul(sl.s21,sr.s13),polymul(sl.s33,sr.s23));
	fu.s23=polyadd(doit2(fu.s23,sl.sz),polymul(sl.s23,sr.s33));
	fu.c1=polyadd(sr.c1,polymul(sl.c1,sr.s11));
	fu.c3=polyadd(polymul(sl.c3,sr.s33),polymul(sl.c1,sr.s13));
	fu.c3=polyadd(sr.c3,doit3(fu.c3,sr.sz));
	fu.a1=polyadd(sl.a1,polymul(sr.a1,sl.s11));
	fu.a2=polyadd(sl.a2,polyrev(sr.a2,sl.s33));
	fu.a2=polyadd(fu.a2,polyrev(sr.a1,sl.s21));
	for(int i=0;i<sr.a1.size()&&i<sl.c1.size();i++)as=(as+1ll*sr.a1[i]*sl.c1[i])%mod;
	return fu;
}
void dfs1(int u,int fa,int tp1)
{
	fu[u].clear();t1[u].clear();sn1[u].clear();
	if(sn[u])dfs1(sn[u],u,tp1);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs1(ed[i].t,u,ed[i].t),sn1[u].push_back(make_pair(sz[ed[i].t],ed[i].t));
	sort(sn1[u].begin(),sn1[u].end());
	vector<int> f2;
	for(int i=0;i<sn1[u].size();i++)
	{
		int t=sn1[u][i].second;
		vector<int> s1=polymul(fu[t],vl),s2;
		for(int j=0;j<s1.size();j++)as=(as+mod-1ll*s1[j]*s1[j]%mod*(mod+1)/2%mod)%mod;
		for(int j=0;j<s1.size()&&j<=n-sz[t];j++)as=(as+1ll*s1[j]*fr[n-sz[t]]%mod*ifr[n-sz[t]-j]%mod*ifr[j])%mod;
		for(int j=0;j<s1.size()&&j<=sz[t];j++)as=(as+1ll*s1[j]*fr[sz[t]]%mod*ifr[sz[t]-j]%mod*ifr[j])%mod;
		for(int j=0;j<=sz[t];j++)s2.push_back(1ll*fr[sz[t]]*ifr[sz[t]-j]%mod*ifr[j]%mod);
		for(int j=0;j<s1.size()&&j<=sz[sn[u]];j++)as=(as+mod-1ll*s1[j]*fr[sz[sn[u]]]%mod*ifr[sz[sn[u]]-j]%mod*ifr[j]%mod)%mod;
		for(int j=0;j<s1.size()&&j<=n-sz[u];j++)as=(as+mod-1ll*s1[j]*fr[n-sz[u]]%mod*ifr[n-sz[u]-j]%mod*ifr[j]%mod)%mod;
		t1[u]=polyadd(t1[u],s1);f2=polyadd(f2,s2);
	}
	for(int i=0;i<sn1[u].size();i++)
	{
		int t=sn1[u][i].second;
		vector<int> s1=polymul(fu[t],vl);
		for(int j=0;j<s1.size()&&j<f2.size();j++)as=(as+mod-1ll*s1[j]*f2[j]%mod)%mod;
	}
	for(int j=0;j<t1[u].size();j++)as=(as+1ll*t1[u][j]*t1[u][j]%mod*(mod+1)/2)%mod;
	if(u==tp1)
	{
		int ct=0;
		for(int i=u;i;i=sn[i])tp[++ct]=i;
		for(int i=1;i*2<=ct;i++)swap(tp[i],tp[ct+1-i]);
		su[1]=1;
		for(int i=2;i<=ct;i++)
		{
			su[i]=su[i-1]+1;
			for(int j=0;j<sn1[tp[i]].size();j++)su[i]+=sz[sn1[tp[i]][j].second];
		}
		if(ct==1){fu[u].push_back(0);fu[u].push_back(1);return;}
		sth fuc=justdoit(2,ct);
		vector<int> a1,a2;a1.push_back(0);a1.push_back(1);a2=a1;a2[0]=1;
		as=(1ll*as+(fuc.a1.size()>1?fuc.a1[1]:0)+(fuc.a2.size()>1?fuc.a2[1]:0)+(fuc.a2.size()>0?fuc.a2[0]:0))%mod;
		vector<int> r1=polyadd(fuc.c1,polyadd(polymul(a1,fuc.s11),polyadd(fuc.s21,doit2(fuc.s21,1))));
		vector<int> r3=polyadd(fuc.c3,doit3(polyadd(doit2(fuc.s13,1),polyadd(fuc.s23,doit2(fuc.s23,1))),fuc.sz));
		for(int i=0;i<r3.size()&&i<=n-sz[u];i++)as=(as+1ll*r3[i]*fr[n-sz[u]]%mod*ifr[n-sz[u]-i]%mod*ifr[i])%mod;
		fu[u]=r1;
	}
}
void dfs2(int u,int fa,int tp1)
{
	if(!sn[u])return;
	dfs2(sn[u],u,tp1);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=sn[u])dfs2(ed[i].t,u,ed[i].t);
	vector<int> f2;
	for(int i=0;i<sn1[u].size();i++)
	{
		int t=sn1[u][i].second;
		vector<int> s2;
		for(int j=0;j<=sz[t];j++)s2.push_back(1ll*fr[sz[t]]*ifr[sz[t]-j]%mod*ifr[j]%mod);
		for(int j=1;j<s2.size();j++)as=(as+1ll*s2[j]*s2[j]%mod*(mod+1)/2%mod)%mod;
		f2=polyadd(f2,s2);
	}
	for(int j=1;j<f2.size();j++)
	{
		as=(as+mod-1ll*f2[j]*f2[j]%mod*(mod+1)/2%mod)%mod;
		if(j<=sz[sn[u]])as=(as+mod-1ll*fr[sz[sn[u]]]*ifr[j]%mod*ifr[sz[sn[u]]-j]%mod*f2[j]%mod)%mod;
		if(j<=n-sz[u])as=(as+mod-1ll*fr[n-sz[u]]*ifr[j]%mod*ifr[n-sz[u]-j]%mod*f2[j]%mod)%mod;
	}
	if(n-sz[u])as=(as+mod-1ll*fr[sz[sn[u]]+n-sz[u]]*ifr[n-sz[u]]%mod*ifr[sz[sn[u]]]%mod+1)%mod;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)as=(as+1ll*fr[n]*ifr[sz[ed[i].t]]%mod*ifr[n-sz[ed[i].t]]-1)%mod;
}
int main()
{
	freopen("color.in","r",stdin);
	freopen("color.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	if(m==mod-1)
	{
		int fg=1;
		for(int i=1;i<=n;i++)if(d[i]>2)fg=0;
		printf("%d\n",fg+1);
		return 0;
	}
	init(18);dfs0(1,0);
	vl=pw(m+1,mod-2);dfs1(1,0,1);
	int s1=as;
	as=0;dfs2(1,0,1);
	int s2=1ll*pw(m+1,n-1)*(1ll*pw(2,n)+s1-as-1+mod*2)%mod;
	printf("%d\n",s2+1);
}
```

#### 6.13

##### T1 training

###### Problem

有一棵 $n$ 个点的树，在树上加入若干条边，得到一个一共有 $m$ 条边的图。

额外加入的边有一个代价，你可以花费这条边的代价以删除这条边。原树边不可删除。

你需要使得图中不存在一个长度为偶数的简单环。求出删边的最小花费。

$n\leq 1000,m\leq 5000$，每个点的度数不超过 $10$。

$1s,512MB$

###### Sol

问题相当于保留的非树边权值和最大。

给树黑白染色，如果一条非树边的两个端点颜色不同，显然加入这条边一定有偶环，因此必须删掉这条边。

此时剩下的每条边都在树上加入了一个奇环。可以发现，如果两条非树边对应的原路径在树上边相交，则考虑将两个环除去交以外的部分拼起来，则可以得到一个偶环。因此只有当所有留下的非树边端点在树上的路径不存在边相交时可能合法。可以发现此时图构成了一个边仙人掌的结构，因此这样一定合法。

此时相当于树上有若干条路径，你需要选一些路径，满足两两不重合且选的路径权值和最大。

考虑设 $dp_u$ 表示只考虑 $u$ 子树内的最大权值，$f_{u,i}$ 表示 $u$ 子树内，不选 $i$ 到 $u$ 路径上的边的最大权值。

考虑 $u$ 上的转移。如果不选子树内经过 $u$ 的路径，则显然 $dp_u=\sum_{y\in son_u}dp_v$。考虑选择一条 $(x,y)$ 的路径($x,y$ 的LCA为 $u$)，设 $x,y$ 分别在 $u$ 的儿子 $v_1,v_2$ 的子树内，则只选择这一条的最优解为 $vl+f_{v_1,x}-dp_{v_1}+f_{v_2,y}-dp_{v_2}+\sum_{y\in son_u}dp_v$。对于 $x=u$ 或者 $y=u$ 的情况，可以看成此时不存在 $v_2$。

对于选多条的情况，显然不能有两个端点在同一子树内，且此时每选一条都会给一个 $vl+f_{v_1,x}-dp_{v_1}+f_{v_2,y}-dp_{v_2}$ 的增加量。

因此一条路径可以看成一条 $v_1$ 到 $v_2$ 的边，边权为 $vl+f_{v_1,x}-dp_{v_1}+f_{v_2,y}-dp_{v_2}$，可能 $v_2$ 不存在。这相当于求一个最大权匹配，此时一个点不选可能也有收益。因为度数很小，直接状压即可。

可以发现转移 $f$ 时相当于求不匹配一个点的最大权匹配，可以使用上面的结果。

复杂度 $O(m\log n+nd^22^d)$

###### Code

```cpp
//Orz lsj
//lsj ak ioi
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 1050
int n,m,s[N*5][3],f[N][12],dep[N],head[N],cnt;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs0(int u,int fa)
{
	f[u][0]=fa;dep[u]=dep[fa]+1;
	for(int i=1;i<=11;i++)f[u][i]=f[f[u][i-1]][i-1];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u);
}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=11;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=11;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int su,dp[N],vl[N],fu[N],id[N];
vector<int> sn[N],tp[N];
void dfs1(int u,int fa)
{
	int ct=0;
	sn[u].push_back(u);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	{
		dfs1(ed[i].t,u);id[ed[i].t]=++ct;
		for(int j=0;j<sn[ed[i].t].size();j++)sn[u].push_back(sn[ed[i].t][j]);
		dp[u]+=dp[ed[i].t];
	}
	for(int i=0;i<1<<ct;i++)fu[i]=0;
	for(int i=0;i<tp[u].size();i++)
	{
		int st=tp[u][i];
		int s1=s[st][0],s2=s[st][1],v1=s[st][2]+vl[s1]+vl[s2];
		for(int j=11;j>=0;j--)
		{
			if(dep[s1]-dep[u]>1<<j)s1=f[s1][j];
			if(dep[s2]-dep[u]>1<<j)s2=f[s2][j];
		}
		int f1=0;
		if(id[s1])f1|=1<<id[s1]-1;if(id[s2])f1|=1<<id[s2]-1;
		fu[f1]=max(fu[f1],v1);
	}
	for(int i=1;i<1<<ct;i++)
	{
		for(int j=1;j<=ct;j++)if(i&(1<<j-1))fu[i]=max(fu[i],fu[i^(1<<j-1)]+fu[1<<j-1]);
		for(int j=1;j<=ct;j++)if(i&(1<<j-1))
		for(int k=j+1;k<=ct;k++)if(i&(1<<k-1))fu[i]=max(fu[i],fu[i^(1<<j-1)^(1<<k-1)]+fu[(1<<j-1)^(1<<k-1)]);
	}
	dp[u]+=fu[(1<<ct)-1];
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)
	for(int j=0;j<sn[ed[i].t].size();j++)vl[sn[ed[i].t][j]]+=fu[((1<<ct)-1)^(1<<id[ed[i].t]-1)]-fu[(1<<ct)-1];
}
int main()
{
	freopen("training.in","r",stdin);
	freopen("training.out","w",stdout);
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)
	{
		scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]);su+=s[i][2];
		if(!s[i][2])adde(s[i][0],s[i][1]);
	}
	dfs0(1,0);
	for(int i=1;i<=m;i++)if(s[i][2])
	{
		if((dep[s[i][0]]^dep[s[i][1]])&1)continue;
		tp[LCA(s[i][0],s[i][1])].push_back(i);
	}
	dfs1(1,0);printf("%d\n",su-dp[1]);
}
```

##### T2 sequence

###### Problem

给一个长度为 $n$ 的序列 $v$，$q$ 次询问：

给出 $l,r,k$，只考虑 $v_{l,...,r}$，你需要选出 $k$ 个不相交的非空子段，使得它们的和最大，求出最大的和。

$n,q\leq 5\times 10^4,|v_i|\leq 10^4$

$4s,1024MB$

###### Sol

考虑没有非空限制的情况，这是一个费用流模型，考虑如下建边：

原点向 $i$ ，$i$ 向汇点连边。 $i$ 向 $i+1$ 连流量为 $1$，费用为 $v_i$ 的边。

可以发现这上面流量为 $k$ 的最大费用流即为不考虑非空的答案，所以设 $v_k$ 表示选 $k$ 段的最大收益，则 $v$ 是一个上凸序列。

可以发现，在 $k$ 大于正数个数之前，非空的限制一定没有影响。在 $k$ 大于正数个数之后，一定每次选一个负数。因此原问题的答案也是一个上凸序列。

考虑线段树维护，对于线段树的一个节点，记录 $dp_{0/1,0/1,x}$ 表示钦定最左侧和最右侧分别选不选，选 $x$ 个非空段的最大收益。

显然 $dp_{0/1,0/1}$ 是一个上凸序列，线段树上合并时枚举两边的情况，再考虑中间是否合并，可以看成做若干次max+卷积。维护线段树的复杂度为 $O(n\log n)$

考虑询问，可以分成若干个线段树上的区间。但此时不能直接合并。~~其实这里更简单的做法是直接wqs~~

最直接的想法是暴力枚举每一段的状态，此时相当于求它们的max+卷积的某一项。因为这是一个凸函数，所以一定是选差分后最大的若干个。因为只有 $O(\log n)$ 段，因此合并中间最多减少 $O(\log n)$ 个子段，询问的项的位置一定在 $[k,k+O(\log n)]$ 之间。

又因为钦定端点必须选相当于钦定费用流模型中一个位置必须流，只影响 $1$ 的流量，因此设 $v_{0/1,0/1,x}=dp_{0/1,0/1,x}-dp_{0/1,0/1,x-1}$，则有：$dp_{0,0,x-1}\geq dp_{i,j,x}\geq dp_{0,0,x}(x>1)$

在 $x=1$ 时不一定满足。因此对于每一段选择 $i,j$ 后找某一项的值，每一段选的个数和对于每一段选择 $0,0$ 后找这一项的值，每一段选的个数相差不超过 $1$。

先找到每一段选择 $0,0$ ，总共选 $k$ 段的最优解。这个可以二分，复杂度 $O(q\log^2 n\log v)$ 。可以发现，此时每一段实际上选的个数不少于这里选的个数减一。因此这样固定了 $k-O(\log n)$ 段，最多剩余 $O(\log n)$ 段需要分配。

因此对于每一段，选的个数在之前选的个数减一到之前选的个数加上 $O(\log n)$ 个之间，可以拿出这些大力dp。可以发现单组询问的复杂度为 $O(\log^3 n)$，实际上需要拿出 $3\log n$ 个，因此常数较大。

复杂度 $O(n\log n+q\log^2 n(\log n+\log v))$

###### Code

```cpp
//Orz lsj
//lsj ak ioi
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
#define N 50050
#define M 35
int n,q,v[N],l,r,k,sr[M],ct,f1[M],f2[M],dp[M][M*3][2],sz[M],rs[M][2][2][M*3];
struct node{int l,r;vector<int> f[2][2];}e[N*4];
vector<int> doit(vector<int> v1,vector<int> v2)
{
	int s1=v1.size(),s2=v2.size();
	int l1=0,r1=0;
	vector<int> as;as.push_back(v1[0]+v2[0]);
	for(int i=1;i<s1+s2-1;i++)
	{
		int fg=0;
		if(l1==s1-1)fg=1;else if(r1==s2-1)fg=0;else fg=v1[l1+1]+v2[r1]<v1[l1]+v2[r1+1];
		if(fg)r1++;else l1++;
		as.push_back(v1[l1]+v2[r1]);
	}
	if(as[0]<-1.01e9)as[0]=-1.01e9;
	return as;
}
void pushup(int x)
{
	for(int i=0;i<2;i++)
	for(int j=0;j<2;j++)
	{
		vector<int> v1=doit(e[x<<1].f[i][0],e[x<<1|1].f[0][j]),v2=doit(e[x<<1].f[i][1],e[x<<1|1].f[1][j]);
		for(int l=1;l<v2.size();l++)v1[l-1]=max(v1[l-1],v2[l]);
		e[x].f[i][j]=v1;
	}
}
void build(int x,int l,int r)
{
	e[x].l=l;e[x].r=r;
	if(l==r)
	{
		vector<int> s1;s1.push_back(0);s1.push_back(v[l]);
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)
		{
			if(i||j)s1[0]=-1.01e9;
			e[x].f[i][j]=s1;
		}
		return;
	}
	int mid=(l+r)>>1;
	build(x<<1,l,mid);build(x<<1|1,mid+1,r);
	pushup(x);
}
void query0(int x,int l,int r)
{
	if(e[x].l==l&&e[x].r==r){sr[++ct]=x;return;}
	int mid=(e[x].l+e[x].r)>>1;
	if(mid>=r)query0(x<<1,l,r);
	else if(mid<l)query0(x<<1|1,l,r);
	else query0(x<<1,l,mid),query0(x<<1|1,mid+1,r);
}
int justdoit(int x,int k)
{
	int lb=1,rb=sz[x],as=0;
	while(lb<=rb)
	{
		int mid=(lb+rb)>>1;
		if(e[sr[x]].f[0][0][mid]-e[sr[x]].f[0][0][mid-1]>=k)as=mid,lb=mid+1;
		else rb=mid-1;
	}
	return as;
}
int main()
{
	freopen("sequence.in","r",stdin);
	freopen("sequence.out","w",stdout);
	scanf("%d%d",&n,&q);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	build(1,1,n);
	while(q--)
	{
		scanf("%d%d%d",&l,&r,&k);
		ct=0;query0(1,l,r);
		int lb=-1e9,rb=1e9,as=0,s1=0,tp1=max(k-ct,0);
		for(int i=1;i<=ct;i++)sz[i]=e[sr[i]].f[0][0].size()-1;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			int s1=0;
			for(int i=1;i<=ct;i++)s1+=justdoit(i,mid);
			if(s1>=tp1)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		for(int i=1;i<=ct;i++)f1[i]=justdoit(i,as+1),f2[i]=justdoit(i,as),s1+=f1[i];
		for(int i=1;i<=ct;i++)
		{
			int tp=min(tp1-s1,f2[i]-f1[i]);
			f1[i]+=tp;s1+=tp;
		}
		for(int i=0;i<=ct;i++)for(int j=0;j<=ct*3;j++)for(int k=0;k<2;k++)dp[i][j][k]=-1e9;
		dp[0][ct][0]=0;
		for(int i=1;i<=ct;i++)
		{
			for(int p=0;p<2;p++)
			for(int q=0;q<2;q++)
			{
				int mx1=e[sr[i]].f[p][q].size()-1-f1[i];
				for(int j=0;j<=ct*3;j++)rs[i][p][q][j]=j>mx1?-1e9:e[sr[i]].f[p][q][f1[i]+j];
			}
			for(int q=0;q<2;q++)for(int j=0;j<=ct*3;j++)rs[i][0][q][j]=max(rs[i][0][q][j],rs[i][1][q][j]);
		}
		for(int i=1;i<=ct;i++)
		{
			for(int j=0;j<=ct*3-i+1;j++)if(dp[i-1][j][0]>-6e8)
			{
				for(int q=0;q<2;q++)
				for(int fu=0;j+fu<=ct*3-i;fu++)
				dp[i][j+fu][q]=max(dp[i][j+fu][q],dp[i-1][j][0]+rs[i][0][q][fu]);
				for(int q=0;q<2;q++)
				for(int fu=0;j+fu-1<=ct*3-i;fu++)
				dp[i][j+fu-1][q]=max(dp[i][j+fu-1][q],dp[i-1][j][1]+rs[i][1][q][fu]);
			}
			for(int j=0;j<=ct*3-i+1;j++)dp[i][j][0]=max(dp[i][j][0],dp[i][j][1]);
		}
		printf("%d\n",dp[ct][ct+(k-tp1)][0]);
	}
}
```

##### T3 archery

###### Problem

有 $2n$ 个人，每个人有一个能力值，能力值构成一个 $2n$ 阶排列。初始所有人排成一个序列。

有 $n$ 个场地，每个场地中会有两个人。在时刻 $0$ ，场地 $i$ 中的人为序列中位置 $2i-1,2i$ 的人。

每一个时刻，场地中的两个人会进行比赛，能力值小的人获胜。

对于 $1$ 号场地，这里胜利的人不动，失败的人移动到 $n$ 号场地。

对于 $i$ 号场地 $i>1$，胜利的人移动到 $i-1$ 号场地，失败的人不动。

现在你是第 $1$ 个人，其它人都已经排好了，你需要选择一个位置插入，使得：

1. 在 $k$ 轮后，你所在的场地编号最小。
2. 在满足上面条件的情况下，你初始所在的场地编号最大。

输出你初始所在的场地位置。

$n\leq 2\times 10^5$

$2s,512MB$

###### Sol

显然，最后最强的人会留在 $1$ 号场地，最弱的 $n-1$ 个人会留在后面的场地，中间 $n$ 个人会进行循环运动。

可以发现 $2n$ 轮后一定会变成上述状态，因此 $2n$ 轮后一定开始循环。

考虑对于一个起始位置，算出最后你所在的位置。分情况讨论：

1. 你的能力值小于等于 $n+1$。

等于 $1$ 的情况显然，考虑后面的情况。

因为一定会循环经过位置 $1$ ，因此只需要求出第一个时刻大于 $2n$ 且经过 $1$ 的时刻，即可求出最后的答案。

将你的能力值看做 $1$，你能赢的看做 $0$，你不能赢的看做 $2$。显然 $0,2$ 内部的顺序没有意义。此时游戏变为两人比赛能力值大的赢，相同的情况其中一个赢。

考虑在第 $1$ 轮后进入场地 $1$ 的人，这显然是场地 $2$ 获胜的人。在第 $2$ 轮后进入场地 $1$ 的人为场地 $2$ 失败的人与场地 $3$ 获胜的人中最强的一个。继续分析可以得到如下结论：

对于在第 $i$ 轮中在场地 $1$ 失败进入场地 $n$ 的人，看做初始在场地 $i+n$ 有一个这个人。则：

第 $i$ 轮进入场地 $1$ 的人为场地 $2,3,...,i+1$ 中，除去之前进入场地 $1$ 的人后最强的人。

证明：在 $n$ 号场地后，每个场地只有一个人，因此这部分一定每次向前，直到到达 $n$ 号场地。而这个时刻正好是这个人原先到达 $n$ 号场地的时刻。因此看成这样不影响答案。

第 $i$ 轮进入场地 $1$ 的人显然一定不会在场地 $i+1$ 后。记通过上面方式找到的人为 $x$。则可以证明如下引理：

在第 $j$ 轮前，$x$ 所在的场地编号不会大于 $i+2-j$。

证明：初始情况显然，在一轮中，如果当前他所在的编号等于 $i+2-j$，则通过在前面的结论中对 $i$ 归纳，可以说明在他之前进入场地 $1$ 的人，此时所在的场地编号一定不会大于 $i+1-j$，因此这一轮他一定获胜，向左移动。如果不等于，则即使这一轮不移动，也不影响下一轮这个条件满足。

因此引理成立，可以发现这个人一定能进入场地 $1$。

因此从左往右扫，记录当前可用的 $0,1,2$ 的个数。每一轮将初始在下一个场地上的人加入可用的部分，然后从可用的部分选择一个最强的进入场地 $1$，再处理场地 $1$ 的比赛即可。

根据上面的结论只需要模拟不超过 $3n$ 轮，因此单次的复杂度为 $O(n)$。

2. 你的能力值大于 $n+1$。

此时你会在某个后面的位置停止。因为你不会循环经过位置 $1$，因此不能使用上面的做法。

注意到只需要求 $2n$ 轮后每个位置上停留的人即可，对于场地 $i(i>1)$，在 $j$ 轮后留在这个场地的人一定是这个位置原来的人加上 $j$ 轮中进入这个场地的人中最弱的。$1$ 号场地上即为最强的。

对于场地 $n$ ，可以求出在第 $1$ 轮后在这个位置停留的人。此时可以发现剩下的人会在前 $2$ 轮进入场地 $i-1$。

同理，求出在第 $x$ 轮前进入 $n-x+1$ 的所有人，即可求出 $x$ 轮时这个位置留下的人以及 $x+1$ 轮前所有进入 $n-x$ 的人。

然后考虑再做一轮。之前求出 $[2,n+1]$ 轮后进入场地 $n$ 的人，和第 $1$ 轮后留在场地 $1$ 的人，可以求出 $[3,n+2]$ 轮中进入场地 $n-1$ 的人，然后继续做这个过程，做到 $1$ 停止。此时即可求出每个位置最后停下的人。

具体过程中，可以同样记录当前向前的人中 $0,1,2$ 的个数，每次把当前位置的人拿出来，再在整体里面选一个最大/最小的留在这里。

这样即可单次 $O(n)$ 求一个起始位置的答案。

称最后的链上位置为你最后的场地编号减去 $n$ 乘你从 $1$ 到 $n$ 的次数，可以发现这相当于把所有移动都看成一条链后的位置。

可以发现如果初始位置向前移动，最后的链上位置一定不会向后移动。且初始位置转一圈时，最后的链上位置也会移动 $n$。

相当于在一个极差为 $n$ 的单调序列中找一个位置使得这个位置模 $n$ 最小，可以直接二分。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 200500
int n,k,v[N*2],v1[N*2],tp[N*4][2],st[N],vl1,as1;
pair<int,int> solve1()
{
	memset(tp,0,sizeof(tp));
	for(int i=1;i<=n;i++)
	for(int j=0;j<2;j++)tp[i][j]=v1[i*2-1+j];
	int c[5]={0},nw=0,as,ct=0;
	int s1=max(tp[1][0],tp[1][1]),s2=min(tp[1][0],tp[1][1]);
	nw=s1;tp[1+n][0]=s2;if(s2==1)ct++;
	for(int i=2;i<=n*3;i++)
	{
		for(int j=0;j<2;j++)c[tp[i][j]]++;
		int mx=0;for(int j=2;j>=0;j--)if(c[j]){mx=j;c[j]--;break;}
		if(mx==1&&i>=n*2){as=i;break;}
		int t1=min(mx,nw);nw=max(mx,nw);
		tp[i+n][0]=t1;
		if(t1==1)ct++;
	}
	ct+=(k+2*n-as)/n;
	int st=(k+n-as)%n;
	return make_pair((n-st-1)%n+1,ct);
}
pair<int,int> solve2()
{
	int st1=0;
	memset(tp,0,sizeof(tp));
	for(int i=1;i<=n;i++)
	for(int j=0;j<2;j++)tp[i][j]=v1[i*2-1+j];
	int c[5]={0};
	for(int i=n;i>=1;i--)
	{
		for(int j=0;j<2;j++)c[tp[i][j]]++;
		int mn=0;for(int j=2;j>=0;j--)if(c[j]){mn=j;if(i==1)break;}
		c[mn]--;st[i]=mn;
	}
	if(c[1])st1=1;
	for(int i=n;i>=1;i--)
	{
		c[st[i]]++;
		int mn=0;for(int j=2;j>=0;j--)if(c[j]){mn=j;if(i==1)break;}
		c[mn]--;st[i]=mn;
	}
	for(int i=1;i<=n;i++)if(st[i]==1)return make_pair(i,st1);
}
pair<int,int> solve(int x)
{
	for(int i=1;i<x*2;i++)v1[i]=v[i+1];
	v1[x*2]=v[1];for(int i=x*2+1;i<=n*2;i++)v1[i]=v[i];
	for(int i=1;i<=n*2;i++)if(v1[i]<v[1])v1[i]=2;else if(v1[i]>v[1])v1[i]=0;else v1[i]=1;
	if(v[1]<=n+1)return solve1();else return solve2();
}
int main()
{
	freopen("archery.in","r",stdin);
	freopen("archery.out","w",stdout);
	scanf("%d%d",&n,&k);vl1=n+1;
	for(int i=1;i<=n*2;i++)scanf("%d",&v[i]);
	int l1=1;
	while(l1<=n)
	{
		int lb=l1,rb=n,as=l1;
		pair<int,int> s1=solve(lb);
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(solve(mid)==s1)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		if(s1.first<=vl1)vl1=s1.first,as1=as;
		lb=l1,rb=n,as=l1;
		while(lb<=rb)
		{
			int mid=(lb+rb)>>1;
			if(solve(mid).second==s1.second)as=mid,lb=mid+1;
			else rb=mid-1;
		}
		l1=as+1;
	}
	printf("%d\n",as1);
}
```

#### 6.14 

##### Problem 1 Falling Sands

Source: CF1534F2

###### Problem

有一个 $n\times m$ 的网格，某些格子中有沙子。

你可以让一个沙子开始下落，如果一个正在下落的箱子当前在 $(x,y)$，则在 $(x,y-1),(x,y+1),(x-1,y),(x+1,y)$ 位置的沙子(如果有)也会开始下落。

给定序列 $a_{1,...,m}$，你需要选择若干个沙子开始下落，使得最后第 $i$ 列至少有 $a_i$ 个沙子下落。输出选择沙子数量的最小值。

$nm\leq 4\times 10^5$

$2s,1024MB$

###### Sol

显然一个位置的沙子开始下落后下面的沙子也会一起下落，因此限制可以看成要求第 $i$ 列从下往上第 $a_i$ 个沙子必须开始下落。

将 $n\times m$ 个位置看成点，考虑加入如下有向边：

1. $(i,j)$ 连向 $(i+1,j)$
2. 如果 $(i-1,j)$ 是沙子，则 $(i,j)$ 连向 $(i-1,j)$。
3. 如果 $(i,j-1)$ 是沙子，则 $(i,j)$ 连向 $(i,j-1)$。如果 $(i,j+1)$ 是沙子，则 $(i,j)$ 连向 $(i,j+1)$。

此时如果从一个沙子的位置开始遍历这个图，则第一类边相当于沙子下落的过程，后两类边相当于沙子导致周围的沙子下落的过程。可以发现，如果让一个位置的沙子开始下落，则它会导致图上所有它能到达的点上的沙子开始下落。

对于每一列，显然只可能选择这一列最上面的沙子让它开始下落。

对于 $i<j<k$，对于第 $k$ 列的一个位置，如果选择了第 $i$ 列最上面的沙子可以到达它，考虑到达它的路径，显然路径上会经过一个第 $j$ 列的位置，根据边的性质经过的这个位置不可能位于第 $j$ 列最上面的沙子上方，因此选择第 $j$ 列最上面的沙子一定可以到达这个位置，因而到达第 $k$ 列的那个位置。

称列 $i$ 能到达位置 $x$ 当且仅当第 $i$ 列最上面的沙子能够到达 $x$，则可以发现对于每个位置，能到达它的列编号一定形成一段区间。从左到右从每一列的位置开始dfs，再反向做一次即可对于每个位置求出能到达它的列。

最后的问题相当于给出若干个区间，需要选数量最少的数，使得每个区间内都至少有一个数被选。直接贪心即可。

复杂度 $O(nm)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 400500
int n,m,a[N],sl[N],sr[N],vl[N],ti;
char s[N];
int getid(int x,int y){if(x>n||x<1||y>m||y<1)return 0;return x*m-m+y;}
void dfs1(int x,int y)
{
	if(x<1||x>n||y<1||y>m)return;
	if(sl[getid(x,y)])return;
	sl[getid(x,y)]=ti;
	dfs1(x+1,y);
	if(s[getid(x,y-1)]=='#')dfs1(x,y-1);
	if(s[getid(x-1,y)]=='#')dfs1(x-1,y);
	if(s[getid(x,y+1)]=='#')dfs1(x,y+1);
}
void dfs2(int x,int y)
{
	if(x<1||x>n||y<1||y>m)return;
	if(sr[getid(x,y)])return;
	sr[getid(x,y)]=ti;
	dfs2(x+1,y);
	if(s[getid(x,y-1)]=='#')dfs2(x,y-1);
	if(s[getid(x-1,y)]=='#')dfs2(x-1,y);
	if(s[getid(x,y+1)]=='#')dfs2(x,y+1);
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=n;i++)scanf("%s",s+(i-1)*m+1);
	for(int i=1;i<=m;i++)scanf("%d",&a[i]);
	for(ti=1;ti<=m;ti++)for(int j=1;j<=n;j++)if(s[getid(j,ti)]=='#')dfs1(j,ti);
	for(ti=m;ti>=1;ti--)for(int j=1;j<=n;j++)if(s[getid(j,ti)]=='#')dfs2(j,ti);
	for(int i=0;i<=m+1;i++)vl[i]=m+1;
	for(int i=1;i<=m;i++)
	{
		int su=0;
		for(int j=n;j>=1;j--)
		{
			if(s[getid(j,i)]=='#')su++;
			if(su==a[i]&&s[getid(j,i)]=='#')
			{
				int lb=sl[getid(j,i)],rb=sr[getid(j,i)];
				vl[lb-1]=min(vl[lb-1],rb);
			}
		}
	}
	for(int i=m;i>=0;i--)vl[i]=min(vl[i],vl[i+1]);
	int as=0,nw=0;while(nw<m+1)nw=vl[nw],as++;
	printf("%d\n",as-1);
}
```

##### Problem 2 New Beginning

Source: CF1534G

###### Problem

在二维平面上有 $n$ 个点 $(x_i,y_i)$。

你初始在原点，你可以选择一条只向右和向上走的折线。

定义一个点的代价为折线上的点与它的切比雪夫距离的最小值。你需要最小化所有点的代价和。

求最小的总代价。

$n\leq8\times 10^5,0\leq x_i,y_i\leq 10^9$

$7s,1024MB$

###### Sol

首先考虑折线上距离一个点切比雪夫距离最小的点位置。

设点位置在 $(x,y)$，考虑折线上与这个点在同一左上到右下对角线上的点，设这个点位置为 $(x+d,y-d)$。

不妨设 $d\geq 0$，则这个点到 $(x,y)$ 的切比雪夫距离为 $d$。

对于折线上之前的点，它们的纵坐标不超过 $y-d$。对于折线上之后的点，它们的横坐标不小于 $x+d$。因此这些点到 $(x,y)$ 的切比雪夫距离不小于 $d$。$d<0$ 的情况类似。因此最小位置一定可以取对角线上的位置。

因此可以在每走到一个点时，统计当前在同一左上到右下对角线上的点的代价。

考虑倒过来走折线，设 $dp_{a,b}$ 表示当前走到 $x+y=a,y=b$ 的点时， $x+y\geq a$ 部分统计的代价的最小值。此时有两种转移：

1. 折线向前走一步，此时 $dp_{a-1,b}=\min(dp_{a,b},dp_{a,b+1})$
2. 统计一个 $x+y=a$ 的 $(x,y)$ 的代价，此时 $dp_{a,b}+=|y-b|$

如果只有2操作显然 $dp$ 是一个凸序列，可以发现加入1操作后 $dp$ 仍然是一个凸序列。

考虑维护凸序列，记录当前斜率为 $0$ 的一段 $[l,r]$ 以及两侧所有的斜率变化点。

对于 $1$ 操作，相当于将 $l$ 以及左侧所有斜率变化点向左平移 $1$，可以整体打标记。显然连续的多个 $1$ 操作可以一起处理。

对于 $2$ 操作，相当于加入两个斜率变化点，根据 $y$ 与 $[l,r]$ 的位置关系分情况维护变化即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define ll long long
#define N 805000
multiset<int> s1,s2;
int n,lz,sl,sr,x,y,ls;
ll as;
pair<int,int> tp[N];
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)scanf("%d%d",&x,&y),tp[i]=make_pair(x+y,y);
	sort(tp+1,tp+n+1);
	for(int i=1;i<=n;i++)s1.insert(0),s2.insert(1e9);
	sl=0,sr=1e9;ls=tp[n].first;
	for(int i=n;i>=1;i--)
	{
		int vl=ls-tp[i].first;ls=tp[i].first;
		sl-=vl;lz+=vl;
		int sy=tp[i].second;
		if(sl<=sy&&sy<=sr)s1.insert(sy+lz),s2.insert(sy),sl=sr=sy;
		else if(sl>sy)
		{
			as+=sl-sy;
			int lb=(*s1.rbegin())-lz;s1.erase(s1.find(lb+lz));
			s2.insert(sl);
			s1.insert(sy+lz);s1.insert(sy+lz);
			sr=sl;sl=(*s1.rbegin())-lz;
		}
		else
		{
			as+=sy-sr;
			int rb=*s2.begin();s2.erase(s2.find(rb));
			s1.insert(sr+lz);
			s2.insert(sy);s2.insert(sy);
			sl=sr;sr=*s2.begin();
		}
	}
	lz+=ls;
	while((*s1.rbegin())-lz>0){int tp=*s1.rbegin();as+=tp-lz;s1.erase(s1.find(tp));}
	while((*s2.begin())<0){int tp=*s2.begin();as-=tp;s2.erase(s2.find(tp));}
	printf("%lld\n",as);
}
```

##### Problem 3 Lost Nodes

Source: CF1534H

###### Problem

交互题

给一棵 $n$ 个点的树，你需要进行交互以猜出两个点 $a,b$，交互方式如下：

首先交互库给你一个点 $f$，满足 $f$ 在 $a,b$ 的路径上。随后你可以进行询问。

每次询问你给出一个点 $x$，交互库返回以 $x$ 为根 $a,b$ 的LCA。

你希望询问次数最小，求出最优策略最坏情况下的询问次数最大值。

求出最大值后，你需要在这个次数内与交互库完成一次交互。

$n\leq 10^5$

$5s,1024MB$

###### Sol

设 $dp_u$ 表示已知当前路径的一个端点为 $u$ 或者 $u$ 的子树外，另外一个端点在 $u$ 的子树内，求出子树内端点最坏情况下需要的最少操作次数。

假设另外一个端点为 $x$，如果询问一个 $u$ 子树内的端点 $y$，显然如果 $x,y$ 不在 $u$ 的同一个儿子的子树内，则询问的结果一定是 $u$，否则询问的结果一定不是 $u$。

为了判断 $x$ 所在的子树(或者判断 $x=u$)，显然需要在每个儿子的子树中至少询问一次。设儿子询问的顺序为 $c_1,c_2,...,c_k$。如果最后得到 $x$ 在 $c_i$ 的子树内，则前 $i-1$ 次询问对判断 $x$ 在子树内的位置没有作用，但最后一次可以通过询问 $dp_{c_i}$ 时最优策略的第一步，因此可以发现这种情况的 $dp_u$ 为：
$$
dp_u=\max(k,\max_{i=1}^kdp_{c_i}+(i-1))
$$
显然将儿子按照 $dp_{c_i}$ 排序，按照 $dp$ 从大到小询问最优。可以 $O(n\log n)$ 算出以一个点为根时所有的 $dp$。

如果 $f$ 给定，考虑以 $f$ 为根做 $dp$，设两个端点为 $x,y$。与上面类似的，如果询问点 $z$ 和 $x,y$ 都不在同一个 $f$ 的儿子的子树中，则询问一定会返回 $x$，否则一定不会返回 $x$。

此时仍然需要按照某个顺序询问儿子，直到找到两个端点所在的子树。显然找到两个子树后就变为了两个子树内部互相独立的问题，与上面类似的，对于一个询问顺序 $c$，答案为：
$$
ans=\max(\max_{i=1}^kdp_{c_i}+(k-1),\max_{1\leq i<j\leq k}dp_{c_i}+dp_{c_j}+(j-2))
$$
显然也是按照 $dp$ 从大到小询问最优，因此第二部分取最大值时一定 $i=1$，可以 $O(k\log k)$ 计算，其中 $k$ 为儿子数量。

上面的东西只需要以 $f$ 为根时每个儿子子树的 $dp$ 值，因此以 $1$ 为根求 $dp$ 后，只需要再求出对于每个点以它为根它父亲子树的 $dp$ 即可。

再做一次dfs，考虑当前在 $u$，已经求出了 $u$ 的值，现在需要求出它所有儿子的值。此时考虑它所有儿子的 $dp$ 以及以它为根它父亲的 $dp$，求一个儿子的值相当于删去一个 $dp$，再求上面 $dp$ 转移式的值。只需要记录排序后的前后缀值的max即可。

这样即可求出答案，对于交互过程，以 $f$ 为根做一次 $dp$ ，然后按照 $dp$ 的策略做交互即可。

复杂度 $O(n\log n)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 100500
int n,a,b,head[N],cnt,dp[N],dp2[N],pr[N],su[N],f,vl[N],as1[N],f1[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
vector<pair<int,int> > st[N],s2[N];
void dfs0(int u,int fa)
{
	st[u].clear();f1[u]=fa;
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs0(ed[i].t,u),st[u].push_back((make_pair(dp[ed[i].t],ed[i].t)));
	if(!st[u].size()){dp[u]=1;vl[u]=u;return;}
	sort(st[u].begin(),st[u].end());
	int tp=-1;
	for(int i=0;i<st[u].size();i++)
	{
		tp++;
		if(tp<=st[u][i].first)vl[u]=vl[st[u][i].second],tp=st[u][i].first;
	}
	dp[u]=tp;
}
void dfs1(int u,int fa)
{
	s2[u]=st[u];
	if(u!=1&&dp2[u]==0)dp2[u]=1;
	if(u!=1)s2[u].push_back(make_pair(dp2[u],fa));
	sort(s2[u].begin(),s2[u].end());
	int sz=s2[u].size();
	for(int i=0;i<=sz;i++)pr[i]=su[i]=0;
	for(int i=0;i<sz;i++)pr[i]=max((i?pr[i-1]:0),s2[u][i].first+(sz-i)-1);
	for(int i=sz-1;i>=0;i--)su[i]=max(su[i+1],s2[u][i].first+(sz-i)-1);
	for(int i=0;i<sz;i++)if(s2[u][i].second!=fa)dp2[s2[u][i].second]=max((i?pr[i-1]-1:0),su[i+1]);
	for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs1(ed[i].t,u);
}
int query(int x){if(as1[x])return as1[x];printf("? %d\n",x);fflush(stdout);int as=0;scanf("%d",&as);return as1[x]=as;}
int doit(int x)
{
	for(int i=0;i<st[x].size();i++)if(query(vl[st[x][i].second])!=x)return doit(st[x][i].second);
	return x;
}
int main()
{
	scanf("%d",&n);
	for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
	dfs0(1,0);dfs1(1,0);
	int mx=0;
	for(int u=1;u<=n;u++)if(s2[u].size())
	{
		int s1=s2[u].size();
		for(int i=0;i+1<s2[u].size();i++)s1=max(s1,s2[u][i].first-1+((int)s2[u].size()-i));
		s1+=s2[u][s2[u].size()-1].first-1;
		mx=max(mx,s1);
	}
	printf("%d\n",mx);fflush(stdout);
	scanf("%d",&f);dfs0(f,0);
	int v1=0,v2=0;
	for(int i=1;i<=n;i++)reverse(st[i].begin(),st[i].end());
	for(int i=0;i<st[f].size();i++)
	{
		int s1=query(vl[st[f][i].second]);
		if(s1!=f){if(!v1)v1=st[f][i].second;else v2=st[f][i].second;}
		if(v2)break;
	}
	if(!v1)v1=f;else v1=doit(v1);
	if(!v2)v2=f;else v2=doit(v2);
	printf("! %d %d\n",v1,v2);
}
```

##### Problem 4 Domination

Source: ARC112F

###### Problem

给出二维平面上 $n$ 个红点，$m$ 个蓝点以及正整数 $k$。

你可以移动蓝点，将蓝点从 $(x,y)$ 移动到 $(x',y')$ 的代价为 $|x-x'|+|y-y'|$。

你需要通过移动蓝点，使得对于每一个红点 $(x,y)$，满足 $x'\geq x,y'\geq y$ 的蓝点 $(x',y')$ 数量大于等于 $k$。

求出最小代价和。

$n,m\leq 10^5,k\leq 10$

$7s,1024MB$

###### Sol

对于两个红点 $(x_1,y_1),(x_2,y_2)$，如果 $x_1\leq x_2,y_1\leq y_2$，则显然第二个红点满足要求第一个一定满足要求。可以删去第一个红点。

假设剩余 $n'$ 个红点，将它们按照横坐标从小到大排序后为 $(x_1,y_1),(x_2,y_2),...,(x_{n'},y_{n'})$，则一定有 $x_1<x_2<...<x_{n'},y_1>y_2>...>y_{n'}$。此时对于一个蓝点，它左下角的红点一定编号形成一段区间。且设当前蓝点坐标为 $(x',y')$，满足 $y_{l-1}<y'\leq y_l,x_r\leq x'<x_{r+1}$，则它左下角的红点为 $[l,r]$ 。

考虑不移动蓝点，如何判断是否合法。相当于有若干个区间，判断是否每一个位置被覆盖了至少 $k$ 次。这等价于选出 $k$ 个区间集合，满足每个集合覆盖了所有点。考虑一个 $n'+2$ 个点的网络流模型，点 $0$ 向点 $1$ 连边，点 $i$ 向 $i-1$ 连边，这些边流量不限。对于一个区间 $[l,r]$，从点 $l$ 向点 $r+1$ 连边，流量限制为 $1$。考虑 $0$ 到 $n'+1$ 的最大流。满足要求当且仅当最大流大于等于 $k$。

现在考虑移动蓝点的情况，显然只会让 $x,y$ 增加。注意到增加 $y$ 相当于使这个点覆盖区间的 $l$ 减少，增加 $x$ 相当于使这个点覆盖区间的 $r$ 增加。

考虑新建两排点，分别表示 $y$ 的变化以及 $x$ 的变化。每一排中将所有出现的 $x/y$ 坐标看做点，相邻两个值 $v_i,v_{i+1}$ 之间，小的向大的连费用为 $v_{i+1}-v_i$ 的边，大的向小的连费用为 $0$ 的边。对于一个蓝点 $(x',y')$，从 $y$ 这一排对应的 $y'$ 点向 $x$ 这一排对应的 $x'$ 点连流量为 $1$ 的边。对于一个红点 $(x'',y'')$，从这个点向对应的 $y''$ 点连边，再从 $x''$ 对应的点向这个点连边。

可以发现，上面这部分中从一个红点出发，走 $y$ 这排点上的路径，再走一个蓝点对应的路径，再走 $x$ 这排点上的路径到一个红点的路径相当于将一个蓝点向上向右移动，再在原先的网络流模型上走这个蓝点覆盖的区间的边。移动的费用即为 $x,y$ 两排点上走的费用。

此时答案即为这个图上流量为 $k$ 的最小费用流。因为图中初始没有负边权，因此直接原始对偶+dijkstra费用流即可。

复杂度 $O(k(n+m)\log(n+m))$

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

#### 6.16

##### T1 match

###### Problem

有 $n$ 个人进行比赛，每个人有一个能力值 $v_i$，比赛的过程可以看成一个二叉树。

这棵二叉树有 $n$ 个叶子，每个叶子表示一个人。每个非叶子节点表示一场比赛，两个儿子上的人进行比赛，$v_i$ 大的人获胜，到达这个点。这场比赛的代价为两个人的能力值差 $|v_i-v_j|$。

你需要选择一个比赛方案满足：

1. 比赛对应的二叉树深度不超过 $k$。
2. 满足条件1的情况下总代价最小。

输出最小代价。

$n\leq 1000,k\leq 200,2^k\geq n$

$2s,512MB$

###### Sol

显然，排序后如果增大相邻两个数间的差距，剩余的差距不变，则对于任意 $k$ 的限制，答案一定不会变小，且可以发现答案减去所有数的极差的结果也不会变小。

因此对于二叉树的一个点，将它内部的叶子按照大小分开一定最优。因此存在一种最优方式满足二叉树的叶子按照能力值排序。

因此可以设 $dp_{l,r,k}$ 表示当前子树为排序后 $[l,r]$ 的人，当前要求深度不超过 $k$ 的最小代价。转移显然为：
$$
dp_{l,r,k}=\min_{i\in [l,r]}dp_{l,i,k-1}+dp_{i+1,r,k-1}+v_r-v_i
$$
可以发现这个转移满足四边形不等式，因此它满足决策单调性。可以使用区间dp上的 $O(n^2)$ 决策单调性做法~~也可以直接决策单调性带log过去~~

复杂度 $O(n^2k)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1050
int n,k,v[N],dp[N][N],fg;
void solve(int l,int r,int l1,int r1,int x)
{
	if(l>r)return;
	int mn=6e8,fr=l,mid=(l+r)>>1;
	if(fg==1)fg=0,mid=l;
	for(int i=max(mid,l1);i<=r1;i++)
	{
		int v2=dp[mid][i]+dp[i+1][x]+v[x]-v[i];
		if(v2<=mn)mn=v2,fr=i;
	}
	solve(l,mid-1,l1,fr,x);dp[mid][x]=mn;solve(mid+1,r,fr,r1,x);
}
int main()
{
	freopen("match.in","r",stdin);
	freopen("match.out","w",stdout);
	scanf("%d%d",&n,&k);
	for(int i=1;i<=n;i++)scanf("%d",&v[i]);
	sort(v+1,v+n+1);
	for(int i=1;i<=n;i++)for(int j=i;j<=n;j++)dp[i][j]=i==j?0:5e8;
	for(int t=1;t<=k;t++)
	for(int j=n;j>=1;j--)
	{
		int st=max(1,j-(t>=20?n+1:(1<<t))+1);
		fg=1;solve(st,j-1,st,j-1,j);
	}
	printf("%d\n",dp[1][n]);
}
```

##### T2 chess

###### Problem

给定 $n,k$，你需要将 $n$ 阶完全图的所有边划分成若干个边集，满足：

1. 每个边集大小不超过 $k$。
2. 每个边集中不存在两条边有公共点。
3. 边集的数量最小。

输出方案。如果你的方案不是最优解但边集个数在 $1.5$ 倍内，则可以获得 $40\%$ 的分数。

$n\leq 1000$

$1s,512MB$

###### Sol

首先考虑 $k=\lfloor\frac n2\rfloor$ 的情况，此时可以看成没有边集大小的限制。

对于 $n$ 为奇数的情况，此时要求用 $n$ 个边集。

可以发现距离为 $1,2,...,\frac{n-1}2$ 的边都正好有 $n$ 个，考虑在每个边集中放一个距离为 $1,2,...,\frac{n-1}2$ 的边。

此时考虑对于点 $i$，构造边集 $\{(i-1,i+1),(i-2,i+2),...,(i-\frac{n-1}2,i+\frac{n-1}2)\}$（考虑循环意义下的下标），可以发现这个边集满足要求。

对于偶数的情况，需要 $n-1$ 个边集，考虑先选出 $n-1$ 个点构造 $n-1$ 个上面的边集，可以发现每个边集有一个点没有使用，将这个点与点 $n$ 的连边加入边集即可。这样正好使用所有边。

对于 $k$ 任意的情况，考虑通过上面构造出的若干个不存在两条边有公共点的边集构造答案。

考虑如下过程：

维护当前没有被使用的边的集合 $S$，这些边满足不存在两条边有公共点。

向集合 $S$ 中加入一个上面的集合，然后分出若干个大小为 $k$ 的边集，使得剩下的边满足不存在两条边有公共点。随后重复上面的过程。

如果能完成这样的过程，则显然分出的边集除去最后一个大小都为 $k$，显然达到理论下界。

考虑一步这样的操作的过程：

此时原边集和加入的边集都满足不存在两边有公共点。因此合并后得到的图中每个点度数不超过 $2$，显然可以得到若干个链和环，同时显然环是偶环。

对于一条链，考虑将所有边按照位置奇偶性分成两部分 $S_1,S_2$，钦定 $|S_1|\geq |S_2|$，可以发现每个集合内部的两条边不存在公共点。

对于一个环，因为环是偶环，可以使用相同的操作进行划分。

这样对于每个原图的连通块，可以得到 $S_1,S_2$，此时可以发现存在公共点的两条边只可能是同一个连通块划分出的 $S_1,S_2$ 中各选一条边，只要最后划分出的边集满足边集中不存在两个边分别来自同一个连通块划分出的 $S_1,S_2$ ，即可保证不存在公共边。

显然有 $|S_2|\leq |S_1|\leq |S_2|+1$，且因为加入的集合大小为 $\lfloor\frac n2\rfloor\geq k$， $S_1$ 大小的和不会小于加入的集合大小，因此 $\sum |S_1|\geq k$。

如果此时 $\sum |S_2|\geq k$ ，则可以在最后的一些 $S_2$ 中选 $k$ 条边组成一个边集。此时对于一对 $S_1,S_2$，因为它们在原图对应一条链或者一个环，因此删去 $S_2$ 若干个元素后，连通块会分成一条链和若干条独立的边。此时这些部分都满足上面的限制。

如果 $\sum |S_2|<k$，可以发现此时一定有 $\sum |S_1|\geq k$。考虑对于每个连通块，在 $S_1,S_2$ 中选择一个加入构造的集合，剩下一个不动。如果找到了这样的方案，则两部分显然都合法。

注意到只需要加入的集合大小之和等于 $k$ 即可，因为 $|S_2|\leq |S_1|\leq |S_2|+1$，初始让每个连通块都选 $S_2$，然后每次让一个连通块的选择变成 $S_1$，每次只会让总大小增加 $1$，则变化的过程中一定有一个时刻满足当前选出的集合大小和为 $k$。此时即满足条件。

最后只会剩下若干独立的边，可以任选 $k$ 个组成集合，直到不能再选为止。这样便完成了这个过程。

实现精细即可做到复杂度 $O(n^2)$

一种更加阳间的做法：

考虑将上面的边集按照 $i$ 从小到大，每个边集内部按照顺序写成一个序列（偶数构造中加入的放在最开头），根据这个构造方式，可以发现任意两条有公共点的边之间的距离至少是 $\lfloor\frac n2\rfloor+1$。

于是可以直接在序列上划分~~我看不懂，但我大受震撼~~

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 1050
int n,k,ct,c1,c2,c3,head[N],cnt,d[N],ty1[N];
struct edge{int t,next;}ed[N*4];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;d[f]++;d[t]++;}
struct sth{int a,b;}s1[N*10];
vector<sth> st[505000],tp[N];
struct sth1{
	vector<int> fu;
	int sz,s1,s2,ty;
}v1[N];
void wt(int x)
{
	if(!x)return;
	int tp=x/10;
	wt(tp);putchar('0'+x-tp*10);
}
int main()
{
	freopen("chess.in","r",stdin);
	freopen("chess.out","w",stdout);
	scanf("%d%d",&n,&k);
	int m=n-(n%2==0);
	for(int i=1;i<=m;i++)
	{
		c2=i;
		for(int j=1;j<=m/2;j++)
		{
			int s1=(i-1+m-j)%m+1,s2=(i-1+j)%m+1;
			tp[c2].push_back((sth){s1,s2});
		}
		if(n%2==0)tp[c2].push_back((sth){i,n});
	}
	for(int i=1;i<=c2;i++)
	{
		for(int j=1;j<=n;j++)head[j]=d[j]=0;cnt=0;
		for(int j=1;j<=c1;j++)adde(s1[j].a,s1[j].b);
		c1=c3=0;
		for(int j=0;j<tp[i].size();j++)adde(tp[i][j].a,tp[i][j].b);
		for(int j=1;j<=n;j++)if(d[j]==1)
		{
			vector<int> sr;
			int ls=j,ct=0,nw=j;
			while(1)
			{
				sr.push_back(nw);d[nw]=0;ct++;
				int nt=0;
				for(int l=head[nw];l;l=ed[l].next)if(ed[l].t!=ls)nt=ed[l].t;
				if(!nt)break;
				ls=nw;nw=nt;
			}
			v1[++c3].fu=sr;v1[c3].sz=ct-1;v1[c3].ty=0;
			v1[c3].s1=v1[c3].sz-v1[c3].sz/2;v1[c3].s2=v1[c3].sz/2;
		}
		for(int j=1;j<=n;j++)if(d[j]==2)
		{
			vector<int> sr;
			int ls=j,ct=0,nw=j;
			while(1)
			{
				sr.push_back(nw);d[nw]=0;ct++;
				int nt=0;
				for(int l=head[nw];l;l=ed[l].next)if(ed[l].t!=ls)nt=ed[l].t;
				if(!nt||d[nt]==0)break;
				ls=nw;nw=nt;
			}
			v1[++c3].fu=sr;v1[c3].sz=ct;v1[c3].ty=1;
			v1[c3].s1=v1[c3].s2=ct/2;
		}
		int su2=0;
		for(int j=1;j<=c3;j++)su2+=v1[j].s2;
		while(su2>=k)
		{
			vector<sth> fu1;
			int s3=k;su2-=k;
			while(s3)
			{
				vector<int>& vi=v1[c3].fu;
				int s2=v1[c3].s2;
				if(s2<=s3)
				{
					for(int p=1;p+1<vi.size();p+=2)fu1.push_back((sth){vi[p],vi[p+1]});
					for(int p=0;p+1<vi.size();p+=2)s1[++c1]=(sth){vi[p],vi[p+1]};
					if(v1[c3].ty)
					{
						int rb=vi.size()-1;
						if(rb&1)fu1.push_back((sth){vi[rb],vi[0]});
						else s1[++c1]=(sth){vi[rb],vi[0]};
					}
					c3--;s3-=s2;
				}
				else
				{
					int ls1=vi[vi.size()-1],ls2=vi[vi.size()-2];
					for(int p=1;p<=s3;p++)
					{
						int sz=vi.size(),l1=vi[sz-1],l2=vi[sz-2];
						if(p>1||!v1[c3].ty)s1[++c1]=(sth){l1,l2};
						fu1.push_back((sth){vi[sz-2],vi[sz-3]});
						vi.pop_back(),vi.pop_back();
					}
					if(v1[c3].ty)
					{
						vector<int> v2;
						v2.push_back(ls2);v2.push_back(ls1);
						for(int q=0;q<vi.size();q++)v2.push_back(vi[q]);
						vi=v2;
					}
					v1[c3].ty=0;v1[c3].sz=vi.size()-1;
					v1[c3].s1=v1[c3].sz-v1[c3].sz/2;v1[c3].s2=v1[c3].sz/2;
					s3=0;
				}
			}
			st[++ct]=fu1;
		}
		for(int j=1;j<=c3;j++)ty1[j]=0;
		int ret=k-su2-c1;if(ret<0)ret=0;
		for(int j=1;j<=c3;j++)if(v1[j].s1>v1[j].s2&&ret)ret--,ty1[j]=1;
		vector<sth> fu1;
		for(int j=1;j<=c3;j++)
		{
			vector<int>& vi=v1[j].fu;
			int fg=ty1[j];
			for(int p=!fg;p+1<vi.size();p+=2)fu1.push_back((sth){vi[p],vi[p+1]});
			for(int p=fg;p+1<vi.size();p+=2)s1[++c1]=(sth){vi[p],vi[p+1]};
			if(v1[j].ty)
			{
				int rb=vi.size()-1;
				if(rb&1)fu1.push_back((sth){vi[rb],vi[0]});
				else s1[++c1]=(sth){vi[rb],vi[0]};
			}
		}
		if(fu1.size()==k)st[++ct]=fu1,fu1.clear();
		int sz1=fu1.size();
		for(int i=1;i<=c1;i++)
		{
			fu1.push_back(s1[i]);
			if(fu1.size()==k)st[++ct]=fu1,fu1.clear();
		}
		for(int i=0;i<fu1.size();i++)s1[i+1]=fu1[i];
		c1=fu1.size();
		if(i==c2&&fu1.size())st[++ct]=fu1;
	}
	wt(ct);putchar('\n');
	for(int i=1;i<=ct;i++)
	{
		wt(st[i].size());putchar('\n');
		for(int j=0;j<st[i].size();j++)wt(st[i][j].a),putchar(' '),wt(st[i][j].b),putchar('\n');
	}
	
}
```

##### T3 sequence

###### Problem

给定一个长度为 $n$ 的正单调序列 $a_{1,...,n}$，对于每个 $k$，求出有多少个非负单调序列 $b_{1,...,n}$ 满足：

1. $0\leq b_i\leq a_i$，$b_i\leq b_{i+1}$
2. 正好有 $k$ 个 $i$ 满足 $a_i=b_i$

答案模 $998244353$

$n,a_i\leq 2.5\times 10^5$

$15s,1024MB$

###### Sol

考虑记录 $c_i=a_i-b_i$，则这个值只需要满足 $0\leq c_i\leq c_{i-1}+(a_i-a_{i-1})$

此时可以看成有一个初始为 $0$ 的数，做 $n$ 轮如下操作，求操作2后值等于 $0$ 的次数为 $k$ 次方案数：

1. 加上 $a_i-a_{i-1}$
2. 减去任意非负整数，但不能小于 $0$。

这也可以看成一个括号序列的问题，操作1相当于加入若干个左括号，操作2相当于加入任意多个右括号。

考虑反过来做，可以看成如下操作：

对于加任意个右括号的操作，看成加入一个大的右括号，大右括号表示一组任意个右括号。 

对于加一个左括号的操作，考虑枚举这个左括号对应的右括号属于哪一个大右括号。则此时可以发现这个大右括号之后的大右括号此时一定是空的，可以删去，因此此时可以看成删去任意数量个大右括号。

可以发现，这个过程结束时，如果留下了 $k$ 个大右括号，可以发现按照原顺序做的时候这 $k$ 个大括号正好是所有能把左括号弹空的位置。因此这就是 $k$ 次为 $0$ 的答案。

此时结构和原问题相同，考虑再变回原问题，相当于给另外一个序列 $a$，求 $b_n=0,1,...$ 的方案数。

这也相当于给一个网格图，第 $i$ 列只有下面 $a_i$ 行，只能向右向上走。从左下角开始在右侧结束，求在右侧每个点结束的方案数。

此时可以分治，选一个边界上的点，从这个点出发向右向下的点将矩形分成三部分。

首先可以分治求出左下部分的答案，因为右下部分是一个矩形，这部分内的情况可以直接卷积求出。此时可以得到到达右侧下部的方案数以及到达中间横向的线上的每个点的方案数。此时相当于对于右上部分，给出下侧每个点进入的方案数，求出右侧每个点走出的方案数。因此对这个分治即可。

复杂度 $O(n\log^2 n)$

###### Code

没写

#### 6.17

##### Problem 1 Sum of Digits 2

###### Problem

给定 $k$ ，称 $k$ 进制下正整数 $n$ 是好的，当且仅当令 $n$ 的 $k$ 进制表示中所有数位的和为 $a$，所有数位平方和为 $b$，则 $n$ 是所有满足 $k$ 进制表示中所有数位的和为 $a$，所有数位平方和为 $b$ 的数中最小的。

定义一个匹配串为一个包含 `x` 和 `x*` 的字符串，其中 `x*` 代表匹配任意多（可以为 $0$ ）个 `x`。

你需要找到若干个匹配串，满足：

1. 一个 $k$ 进制串能被至少一个匹配串匹配当且仅当它是好的。
2. 匹配串的数量最小。
3. 匹配串的长度和最小。

按照匹配串的字典序大小输出。

 $2\leq k\leq 36$

$5s,64MB$

###### Sol

因为重排数位 $a,b$ 不变，显然在最终串中，所有数位一定单调不降。

可以发现，如果存在 `a*b` 的形式，满足 $a+1<b$，则考虑如下构造：

将一个 $a$ 和一个 $b$ 换成 $a+1,b-1$，此时和不变，平方和减少 $2(b-a-1)$。

随后将 $2(b-a-1)$ 个 $a$ 换成 $b-a-1$ 个 $a-1,a+1$，此时和不变，平方和增加 $2(b-a-1)$。

可以发现这样变化后从 $2(b-a)-1$ 个 $a$ 和一个 $b$ 变成了 $b-a-1$ 个 $a-1,a+1$ 和一个 $b-1$，此时排序后数会变小。

因此最后的数有这几种情况：

1. 存在 $a*(a+1)*$，以及若干个小于 $a$ ，没有 $*$ 的数。
2. 存在 $a*$，若干个 $a+1$，以及若干个小于 $a$ ，没有 $*$ 的数。
3. 不存在 $*$。

考虑对于一个没有 $*$ 的串，判断它是否满足要求。

考虑记 $dp_{a,b}$ 表示和为 $a$，平方和为 $b$ 时的最小位数。对于给定 $a,b$ 考虑构造答案，则每次一定是选择最小的 $i$，满足 $dp_{a-i,b-i^2}=dp_{a,b}-1$，将当前最高位设为 $i$，然后继续构造。

那么一个数满足要求当且仅当从后向前考虑每一个后缀，它满足上述条件。且一个满足要求的串显然和一对 $(a,b)$ 对应。

考虑处理带 $*$ 的情况，可以看成将 $*$ 看成若干个这个数，显然个数越多越容易不合法。在本题数据范围中取 $13$ 个即可判断正确。

然后显然答案不会太多，考虑dfs找答案，根据上面的分析从低位向高位一位一位搜即可判断合法。因为上面三种情况中显然前面的串可能包含后面的串，因此可以顺序搜每种情况，对于一个串，需要判断它是否可能被前面的某个匹配串匹配。因为 $*$ 的情况非常特殊，可以大力判掉 ~~细节留作练习~~

复杂度 $O(?)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<iostream>
#include<string>
#include<map>
using namespace std;
char dp[1102][36004];
int k,ct,ti;
char st[]="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
map<int,int> fuc[42],fuc2[42],fuc3;
string orzdjq[48000];
int dfs(int x,int y)
{
	if(x<=0||y<=0)return x==0&&y==0?0:120;
	if(dp[x][y])return dp[x][y];
	int as=120;
	for(int i=1;i<k;i++)as=min(as,dfs(x-i,y-i*i)+1);
	return dp[x][y]=as;
}
int getid(int x,int y){return x*50000+y;}
void doit(int x,int y,int sx,int sy)
{
	string f1;fuc[ti][getid(x,y)]=1;fuc3[getid(x,y)]=1;
	while(x||y)for(int i=1;i<k;i++)if(dfs(x-i,y-i*i)==dfs(x,y)-1){f1+=st[i];x-=i;y-=i*i;break;}
	f1+=st[sx];f1+='*';f1+=st[sy];f1+='*';
	orzdjq[++ct]=f1;
}
int chk(int x,int y){if(x>1100||y>36000)return 0;for(int i=1;i<k;i++)if(dfs(x-i,y-i*i)==dfs(x,y)-1)return i;}
void dfs(int x,int y,int sx,int sy)
{
	if(sy>=k)return;
	doit(x,y,sx,sy);
	for(int i=1;i<sx;i++)
	{
		int tx=x+i,ty=y+i*i;
		if(chk(tx,ty)==i&&chk(tx+sx*13+sy*13,ty+sx*sx*13+sy*sy*13)==i)dfs(tx,ty,sx,sy);
	}
}
void doit0(int x,int y,int sx)
{
	string f1;fuc[ti][getid(x,y)]=1;fuc3[getid(x,y)]=1;
	while(x||y)for(int i=1;i<k;i++)if(dfs(x-i,y-i*i)==dfs(x,y)-1){f1+=st[i];x-=i;y-=i*i;break;}
	f1+=st[sx];f1+='*';
	orzdjq[++ct]=f1;
}
void dfs0(int x,int y,int sx,int c0,int ls)
{
	if(c0>12)return;
	if(!fuc[ti-1][getid(x-c0*ls,y-c0*ls*ls)]&&!fuc[ti][getid(x,y)])doit0(x,y,sx);
	for(int i=1;i<sx;i++)
	{
		int tx=x+i,ty=y+i*i;
		if(chk(tx+sx*13,ty+sx*sx*13)==i)dfs0(tx,ty,sx,ls==i?c0+1:c0,ls);
	}
}
void doit1(int x,int y,int sx,int r)
{
	string f1;fuc2[ti][getid(x,y)+((sx+1)*12+r)*10000003]=1;
	while(x||y)for(int i=1;i<k;i++)if(dfs(x-i,y-i*i)==dfs(x,y)-1){f1+=st[i];x-=i;y-=i*i;break;}
	f1+=st[sx];f1+='*';
	for(int i=1;i<=r;i++)f1+=st[sx+1];
	orzdjq[++ct]=f1;
}
void dfs1(int x,int y,int sx,int r)
{
	if(!fuc[ti][getid(x,y)])doit1(x,y,sx,r);
	for(int i=1;i<sx;i++)
	{
		int tx=x+i,ty=y+i*i;
		if(chk(tx+sx*13+(sx+1)*r,ty+sx*sx*13+(sx+1)*(sx+1)*r)==i)dfs1(tx,ty,sx,r);
	}
}
int t1,v1,t2,v2;
void doit2(int x,int y)
{
	string f1;
	while(x||y)for(int i=1;i<k;i++)if(dfs(x-i,y-i*i)==dfs(x,y)-1){f1+=st[i];x-=i;y-=i*i;break;}
	for(int i=1;i<=v2;i++)f1+=st[t2];
	for(int i=1;i<=v1;i++)f1+=st[t1];
	orzdjq[++ct]=f1;
}
void dfs2(int x,int y)
{
	if(!fuc[ti-1][getid(x,y)]&&!fuc[ti][getid(x+t2*v2,y+t2*t2*v2)]&&!fuc2[ti-1][getid(x,y)+(t1*12+v1)*10000003]&&!fuc3[getid(x+t1*v1+t2*v2,y+v1*t1*t1+v2*t2*t2)])doit2(x,y);
	for(int i=1;i<t2;i++)
	{
		int tx=x+i,ty=y+i*i;
		if(chk(tx+t1*v1+t2*v2,ty+v1*t1*t1+v2*t2*t2)==i)dfs2(tx,ty);
	}
}
int main()
{
	scanf("%d",&k);
	ti++;
	for(int i=1;i<k;i++)
	{
		ti++,dfs(0,0,i,i+1);
		if(i+1<k)for(int j=1;j<=10;j++)dfs1(0,0,i,j);else dfs0(0,0,i,0,i-1);
	}
	ti=2;
	for(int i=2;i<k;i++)
	{
		ti++;t1=i,t2=i-1;
		for(int j=1;j<=10;j++)for(int l=0;j+l<=12;l++)v1=j,v2=l,dfs2(0,0);
	}
	sort(orzdjq+1,orzdjq+ct+1);
	for(int i=1;i<=ct;i++)cout<<orzdjq[i]<<endl;
}
```

##### Problem 2 Recursive Ants

###### Problem

有一个 $2^n\times 2^n$ 的矩形，其中有 $m$ 个格子为障碍。

你需要从左上角 $(0,0)$ 开始走，经过每个非障碍格子正好一次，最后停留在边界的一个格子上。

你的路径必须是递归的，递归的定义为：

你进入一个 $2^k\times 2^k$ 的区域时，将它分为四个 $2^{k-1}\times 2^{k-1}$ 的小区域，你必须递归地走完一个区域才能进入下一个区域。每个区域内也需要满足递归的性质。

对于四个方向中的每一个，求出你必须停留在这个方向上的边界时，你最后停留的位置。可以证明只有唯一解或者无解。

$n\leq 30,m\leq 50$

$0.1s,512MB$

###### Sol

存在如下性质：固定方向后，走的方案唯一。

假设你初始在左上角的小区域内。对于其他情况，可以翻转旋转矩形。考虑每一个小区域都有空位的情况。

可以发现，此时大区域中只有两种走的方式：
$$
1\ 2\\
3\ 4\\
1\to2\to4\to3\\
1\to3\to4\to2
$$
可以发现，第一种方式能停留在左下方向，第二种方式能停留在右上方向。因此确定方向后，大区域上的路径唯一。

此时可以发现对于每一个小区域上的方向固定，因此归纳即可证明。

对于有一个位置全是障碍的情况，此时路径唯一，显然方案也唯一。

对于全部没有障碍的一个小区域，手玩可以发现，如果初始在 $(0,x)(x<2^{k-1})$，则最后四个方向的解分别为 $(0,2^n-1-x),(x,2^n-1),(2^n-1-x,0),(2^n-1,x)$。因此只需要对于存在障碍的区域进行递归。

可以发现递归树只有 $m$ 个叶子，深度为 $n$，因此复杂度 $O(nm)$

###### Code

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
int n,m,x,y,f[4][4]={0,3,1,0,2,0,0,1,0,0,0,3,0,0,2,0};
struct st1{int x,y;};
vector<st1> fu;
//0u1d2l3r
int solve(int n,int sd,int td,vector<st1> fu,int sv)
{
	int fg=0;
	if(!fu.size())
	{
		int tp=(sd^td)&1;
		if(!tp)sv=(1<<n)-1-sv;
		return sv;
	}
	if(sd>1)
	{
		sd^=2;td^=2;
		for(int i=0;i<fu.size();i++)swap(fu[i].x,fu[i].y);
	}
	if(sd==1)
	{
		sd^=1,td^=1;fg=1;sv=(1<<n)-sv-1;
		for(int i=0;i<fu.size();i++)fu[i].x=(1<<n)-1-fu[i].x,fu[i].y=(1<<n)-1-fu[i].y;
	}
	if(sv>=(1<<n-1))
	{
		sv=(1<<n)-sv-1;
		if(td>=2)td^=1;else fg^=1;
		for(int i=0;i<fu.size();i++)fu[i].y=(1<<n)-1-fu[i].y;
	}
	for(int i=0;i<fu.size();i++)if(fu[i].x==0&&fu[i].y==sv)return -1;
	vector<st1> s[2][2];
	for(int i=0;i<fu.size();i++)
	{
		int sx=fu[i].x,sy=fu[i].y;
		int v1=sx>>(n-1),v2=sy>>(n-1);sx-=v1<<n-1;sy-=v2<<n-1;
		s[v1][v2].push_back((st1){sx,sy});
	}
	int sz=n>10?23333:1<<(2*n-2),st=0,c1=0;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)if(s[i][j].size()==sz)st|=1<<(i*2+j),c1++;
	int ct=1,s1[6]={0};
	if(!st)ct=2,s1[2]=(td%3==0?2:1),st|=1<<s1[2];
	st|=1;
	while(c1+ct<4)
	{
		int nt,nw=s1[ct];
		if(!((st>>(nw^1))&1))nt=nw^1;
		else if(!((st>>(nw^2))&1))nt=nw^2;
		else return -1;
		s1[++ct]=nt;st|=1<<nt;
	}
	for(int i=1;i<=ct;i++)
	{
		int ns=0,nv=s1[i];
		if(i<ct)ns=f[s1[i]][s1[i+1]];else ns=td;
		sv=solve(n-1,sd,ns,s[nv>>1][nv&1],sv);
		if(sv==-1)return -1;
		sd=ns^1;
	}
	if(sd>1){sv+=(s1[ct]>=2)<<n-1;if((s1[ct]^sd^1)&1)return -1;}
	else {sv+=(s1[ct]&1)<<n-1;if((s1[ct]^((sd^1)<<1))&2)return -1;}
	if(fg)sv=(1<<n)-1-sv;
	return sv;
}
void doit(int d)
{
	int as=solve(n,0,d,fu,0);
	if(as==-1)printf("NIE\n");
	else
	{
		int sl=0,sr=as;
		if(d&1)sl=(1<<n)-1;
		if(d&2)sl^=sr^=sl^=sr;
		printf("%d %d\n",sl,sr);
	}
}
int main()
{
	scanf("%d%d",&n,&m);
	for(int i=1;i<=m;i++)scanf("%d%d",&x,&y),fu.push_back((st1){x,y});
	doit(0);doit(3);doit(1);doit(2);
}
```

##### Problem 3 Necklaces

###### Problem

给你两个压缩后的串，判断他们是否循环同构。

压缩方式为，给出 $n$ 个部分，每个部分包含一个字符串 $s_i$ 和 $l_i$，表示这部分为 $s_i$ 循环 $l_i$ 次。最后的字符串为所有串拼接起来。

$n\leq 1000,\sum |s_i|\leq 10^4,l_i\leq 10^5$

$0.1s,512MB$

###### Sol

显然只需要找到使字典序最小的起始位置，再hash即可判断合法。

但直接找较为困难，考虑找到可能是最小表示法的位置。

考虑一个循环部分内的所有起始位置。枚举起始位置在 $s_i$ 中的下标，考虑起始位置所在的不同的循环次数，得到的字符串的形式。

可以发现，这相当于有两个字符串 $A,B$，得到的是 $A...AB,A...ABA,...,BA...A$ 这些串。

可以发现，如果 $AB=BA$，则上面所有串相等。如果 $AB<BA$，则可以发现相邻两个串之间，前者字典序更小，因此第一个串字典序最小。否则可以发现一定最后一个串字典序最小。

因此可能是最小的起始点的位置只有循环中的第一个和最后一个串。因此只有 $O(\sum |s_i|)$ 个位置。

通过预处理，可以 $O(n)$ 或者 $O(1)$ 计算一个位置的hash值。最后判断这些hash值是否存在相同即可。

模数取 $10^{18}$ 可以让错误率降至 $10^{-10}$。

复杂度 $O(n\log l+\sum |s_i|)$ 或者 $O(n\sum |s_i|)$

###### Code

```cpp
#include<cstdio>
#include<vector>
#include<algorithm>
#include<map>
#define N 1050
#define ll long long
using namespace std;
const ll mod=(ll)(1e18)+3;
int n,le[N],vl[N];
char s[N*10];
vector<char> st[N];
struct sth{ll a,b;}fu[N],f2[N];
ll mul(ll a,ll b){ll st=(long double)a*b/mod,as=a*b-mod*st;return as;}
sth doit(sth a,sth b){return (sth){mul(a.a,b.a),(mul(a.b,b.a)+b.b)%mod};}
sth pw(sth a,ll b){sth as;as.a=1;as.b=0;while(b){if(b&1)as=doit(as,a);a=doit(a,a);b>>=1;}return as;}
vector<sth> sl[N],sr[N],v1[N];
map<ll,int> fuc;
int main()
{
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&le[i]);
		scanf("%s",s+1);
		scanf("%d",&vl[i]);
		for(int j=1;j<=le[i];j++)v1[i].push_back((sth){131ll,s[j]-'a'+10});
		sth s1=(sth){1,0};
		for(int j=1;j<=le[i];j++)s1=doit(s1,v1[i][j-1]),sl[i].push_back(s1);
		s1=(sth){1,0};
		for(int j=le[i];j>=1;j--)s1=doit(v1[i][j-1],s1),sr[i].push_back(s1);
		reverse(sr[i].begin(),sr[i].end());
		fu[i]=pw(s1,vl[i]);f2[i]=pw(s1,vl[i]-1);
	}
	for(int i=1;i<=n;i++)
	for(int j=1;j<=le[i];j++)
	{
		sth s1=sr[i][j-1];
		s1=doit(s1,f2[i]);
		for(int l=i%n+1;l!=i;l=l%n+1)s1=doit(s1,fu[l]);
		if(j>1)s1=doit(s1,sl[i][j-2]);
		fuc[s1.b]=1;
	}
	for(int i=1;i<=n;i++)
	for(int j=1;j<=le[i];j++)
	{
		sth s1=sr[i][j-1];
		for(int l=i%n+1;l!=i;l=l%n+1)s1=doit(s1,fu[l]);
		s1=doit(s1,f2[i]);
		if(j>1)s1=doit(s1,sl[i][j-2]);
		fuc[s1.b]=1;
	}
	scanf("%d",&n);
	for(int i=1;i<=n;i++)
	{
		scanf("%d",&le[i]);
		scanf("%s",s+1);
		scanf("%d",&vl[i]);
		sl[i].clear();sr[i].clear();v1[i].clear();
		for(int j=1;j<=le[i];j++)v1[i].push_back((sth){131ll,s[j]-'a'+10});
		sth s1=(sth){1,0};
		for(int j=1;j<=le[i];j++)s1=doit(s1,v1[i][j-1]),sl[i].push_back(s1);
		s1=(sth){1,0};
		for(int j=le[i];j>=1;j--)s1=doit(v1[i][j-1],s1),sr[i].push_back(s1);
		reverse(sr[i].begin(),sr[i].end());
		fu[i]=pw(s1,vl[i]);f2[i]=pw(s1,vl[i]-1);
	}
	for(int i=1;i<=n;i++)
	for(int j=1;j<=le[i];j++)
	{
		sth s1=sr[i][j-1];
		s1=doit(s1,f2[i]);
		for(int l=i%n+1;l!=i;l=l%n+1)s1=doit(s1,fu[l]);
		if(j>1)s1=doit(s1,sl[i][j-2]);
		if(fuc[s1.b]){printf("TAK\n");return 0;}
	}
	for(int i=1;i<=n;i++)
	for(int j=1;j<=le[i];j++)
	{
		sth s1=sr[i][j-1];
		for(int l=i%n+1;l!=i;l=l%n+1)s1=doit(s1,fu[l]);
		s1=doit(s1,f2[i]);
		if(j>1)s1=doit(s1,sl[i][j-2]);
		if(fuc[s1.b]){printf("TAK\n");return 0;}
	}
	printf("NIE\n");
}
```



