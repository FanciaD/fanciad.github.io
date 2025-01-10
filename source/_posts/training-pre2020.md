---
title: 2019/02~07 JZ集训 & ZR集训 题解
date: '2021-02-22 22:23:02'
updated: '2021-02-22 22:23:02'
tags: Mildia
permalink: AcientTimes/
description: 2019/02~07 JZ集训 & ZR集训
mathjax: true
---

Note. 本篇的完成时间已不可考，此处时间为上一个 blog 建立的时间。

Note2. 本篇为最早写作的文章，同时我的说话水平大概是递增的。但我选择把它原样留下来。你也可以看到，在这几个月之间发生了怎样的变化。


~~其实是这几个月的题目总结~~

~~因为这个人太菜了,不会的题全部跳过~~

### Part 1 JZ Onsite

一套好题，但我还是太菜了。。。。。。。。。

#### R1T1 树上四次求和

##### 题面 from jzoj

![](/pic\1.png)

![](/pic\2.png)

![](/pic\3.png)

![](/pic\4.png)

##### 冷静分析

题目的式子相当于 $ \sum_{i=1}^k \sum_{j=i}^k \sum_{x=i}^j \sum_{y=a}^j dis(a_x,a_y)$

考虑一个$dis(a_x,a_y) (x<y)$ 对一个$k$的贡献

需要满足$ 1<=i<=x $且$y< =j<=k$ 

所以会贡献$x*(k-y+1)$次$(y<=k)$

所以是$\sum_{x=1}^k \sum_{y=i}^k dis(a_x,a_y)*x*(k-y+1)$

可以$O(n^2q)$获得0分的好成绩

考虑$k++​$对答案的改变

是$\sum_{x=1}^k \sum_{y=i}^k dis(a_x,a_y)*x​$

仍然是$O(n^2q)$

考虑计算改变值的改变值（相当于差分两次）

然后就变成了$\sum_{x=1}^k dis(a_x,a_k)*x$

然后暴力就可以$O(nqlogn)$获得30分

考虑如何快速维护这个东西

路径信息考虑点分治

因为需要一个一个加点所以我选择动态点分治

在一层分治中心$s$上需要维护$\sum (dis(x,s)+dis(a,s))*a$

维护$\sum a$  和$\sum a*dis(a,s)$

然后默写板子（

$O(nlogn)$

##### 代码

~~点分树开小自闭~~

```c++
#include<cstdio>
#pragma comment(linker, "/STACK:256000000,256000000")
using namespace std;
#define N 100050
#define mod 998244353
int head[N],cnt,sz[N],as,pt,vl,fa[N],dep[N],dis[N],ds[N][31],sum1[N][2],sum2[N][2],n,m,ans[N],q,a,b,mxds,vis[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
inline int Max(int a,int b){return a>b?a:b;}
void dfs1(int u,int fa)
{
    sz[u]=1;int mx=0;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&!vis[ed[i].t])
    dfs1(ed[i].t,u),mx=Max(mx,sz[ed[i].t]),sz[u]+=sz[ed[i].t];
    mx=Max(sz[u],vl-sz[u]);
    if(as>mx)as=mx,pt=u;
}
void dfs2(int u,int fa,int dep)
{
    ds[u][dep]=dis[u]=dis[fa]+1;mxds=Max(mxds,dis[u]);
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&!vis[ed[i].t])
    dfs2(ed[i].t,u,dep);
}
void work(int u)
{
    vis[u]=1;ds[u][dep[u]]=0;dis[u]=0;
    for(int i=head[u];i;i=ed[i].next)
    if(!vis[ed[i].t])
    mxds=0,dfs2(ed[i].t,u,dep[u]);
}
void dfs3(int u)
{
    work(u);
    for(int i=head[u];i;i=ed[i].next)
    if(!vis[ed[i].t])
    vl=sz[ed[i].t],as=1e9,dfs1(ed[i].t,u),fa[pt]=u,dep[pt]=dep[u]+1,dfs3(pt);
}
void modify(int x,int f,int dep,int s,int v)
{
    int d=ds[s][dep];
    sum1[x][0]=(sum1[x][0]+v)%mod;sum2[x][0]=(sum2[x][0]+1ll*v*d)%mod;
    if(f)sum1[f][1]=(sum1[f][1]+v)%mod,sum2[f][1]=(sum2[f][1]+1ll*v*d)%mod;
    if(dep>1)modify(fa[x],x,dep-1,s,v);
}
int query(int x,int f,int dep,int s)
{
    int d=ds[s][dep];
    int ans=(1ll*(sum1[x][0]-sum1[f][1])*d%mod+sum2[x][0]-sum2[f][1])%mod;
    if(dep>1)ans=(ans+query(fa[x],x,dep-1,s))%mod;
    return ans;
}
int main()
{
    freopen("sumsumsum.in","r",stdin);
    freopen("sumsumsum.out","w",stdout);
    scanf("%d%d",&n,&q);
    for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
    dep[1]=1;dfs1(1,0);dfs3(1);
    int tp=0;
    for(int i=1;i<=n;i++)
    scanf("%d",&a),tp=(tp+query(a,0,dep[a],a))%mod,ans[i]=(ans[i-1]+tp)%mod,modify(a,0,dep[a],a,i);
    while(q--)scanf("%d",&a),printf("%d\n",(ans[a]+mod)%mod);
}
```

#### R1T2 cubelia

##### 题面

![](/pic\9.png)

![](/pic\10.png)

![](/pic\11.png)

![](/pic\12.png)

##### #%@&!&^*!@$

自闭题

设前缀和$sum[i]=\sum_{k=1}^i a[k]$

$[l,r]$的最大前缀和是$max_{i=l}^r(sum[i])-sum[l-1]$

所以$[l,r]​$的答案是$\sum_{i=l}^r \sum_{j=i}^r max_{k=i}^j (sum[k])-\sum_{i=l-1}^{r-1}sum[i]*(r-i)​$

后面的可以前缀和

所以关键要求$\sum_{i=l}^r \sum_{j=i}^r max_{k=i}^j (sum[k])​$

直接分治可以得到10分

题解太神仙了$orz​$

设$ans[i][j]=\sum_{i=l}^r \sum_{j=i}^r max_{k=i}^j (sum[k])$

先找到最大值

然后跨过最大值的答案都显然是max

然后考虑最大值左边的

计算$ ans[i][n]-ans[j+1][n]$

这时候还要减去左端点在$[i,j]$右端点在$[j+1,n]$的答案

因为$j+1$是区间最大值所以$[i,j]$对区间最大值的贡献可以无视

所以就是$(j-i+1)*(\sum_{k=j+1}^n \max sum[j+1 ... k])$

后面那个是可以单调栈算的

然后发现$ans[i][n]$就是那个的后缀和

后面那部分就可以算了

然后考虑前面部分

把序列倒过来做一遍

直接rmq可以做到$O(nlogn+q)$

通过优秀rmq可以$O(n+q)$

但是4s可以过

##### emmm

```c++
#include <bits/stdc++.h>
using namespace std;
#pragma comment(linker, "/STACK:256000000,256000000")
inline int R() {
    int a;scanf("%d",&a);return a;
}
#define mod 998244353
stack<int> st1,st2;
int a[2000007],lg2[2000200];
long long sum[2000050],rm[2000050][22],s1[2000050],s2[2000050],as,pre[2000050],suf[2000050],spre[2000050],ssuf[2000050];
int n, q;
int S, A, B, P, tp;
long long lastans;
inline int Rand() {
    S = (S * A % P + (B ^ (tp * lastans))) % P;
    S = S < 0 ? -S : S;
    return S;
}

int querymax(int l,int r){
return sum[rm[l][lg2[r-l+1]]]>sum[rm[r-(1<<lg2[r-l+1])+1][lg2[r-l+1]]]?rm[l][lg2[r-l+1]]:rm[r-(1<<lg2[r-l+1])+1][lg2[r-l+1]];
}
long long solve2(int l, int r)
{
    if(l>r)return 0;
    return ssuf[l]-ssuf[r+1]-(r-l+1)*suf[r+1];
}
long long solve3(int l, int r)
{
    if(l>r)return 0;
    return spre[r]-spre[l-1]-(r-l+1)*pre[l-1];
}
long long solve(int l, int r) {
    if(l>r)return 0;
    int tp=querymax(l,r);
    long long ans=sum[tp]*(r-tp+1)*(tp-l+1);
    return ans+solve2(l,tp-1)+solve3(tp+1,r);
}
int main() {
    freopen("cubelia.in", "r", stdin);
    freopen("cubelia.out", "w", stdout);
    n = R(), q = R();
    for (int i = 1; i <= n; ++i) a[i] = R();
    S = R(), A = R(), B = R(), P = R(), tp = R();
    for(int i=1;i<=n;i++)sum[i]=sum[i-1]+a[i],rm[i][0]=i,s1[i]=s1[i-1]+sum[i],s2[i]=s2[i-1]+sum[i]*i;
    for(int i=2;i<=n;i++)lg2[i]=lg2[i>>1]+1;
    for(int i=1;i<=21;i++)
    for(int j=1;j+(1<<i)-1<=n;j++)
    rm[j][i]=sum[rm[j][i-1]]>sum[rm[j+(1<<i-1)][i-1]]?rm[j][i-1]:rm[j+(1<<i-1)][i-1];
    sum[n+1]=1e17;st1.push(n+1);
    long long tmp=0;
    for(int i=n;i>=1;i--)
    {
        while(sum[st1.top()]<sum[i]){int st=st1.top();st1.pop();tmp-=sum[st]*(st1.top()-st);}
        tmp+=sum[i]*(st1.top()-i);st1.push(i);
        suf[i]=tmp;ssuf[i]=ssuf[i+1]+suf[i];
    }
    sum[0]=1e17;st2.push(0);
    long long tmp1=0;
    for(int i=1;i<=n;i++)
    {
        while(sum[st2.top()]<sum[i]){int st=st2.top();st2.pop();tmp1-=sum[st]*(st-st2.top());}
        tmp1+=sum[i]*(i-st2.top());st2.push(i);
        pre[i]=tmp1;spre[i]=spre[i-1]+pre[i];
    }
    for (; q; --q) {
        int l = Rand() % n + 1, r = Rand() % n + 1;
        if (l > r)
            swap(l, r);
        long long tmp = solve(l, r);
        if(l==1)l=2;
        tmp-=(s1[r]-s1[l-2])*r-s2[r]+s2[l-2];
        as=(as+tmp)%mod;lastans=tmp;
    }
    printf("%d\n",(as+mod)%mod);
    return 0;
}
```



#### R1T3 cuvelia

##### 题面

![](/pic\5.png)

![](/pic\6.png)

![](/pic\7.png)

![](/pic\8.png)

##### 分析

如果$k=1$答案一定是$n$

然后就有10分了

然后对于$k>=2$的情况，实际上可以拆成很多$k=2$的情况

因为等于有传递性

所以只需要做$k=2$

首先如果$dis(a,b) \mod 2=1$的话显然无解

否则的话答案是全集减路径中点为根a和b所在的子树

于是这是几段dfs序区间

求覆盖k-1次的数量

扫描线

$O(n+klogn+klogk)$

##### 代码

$k=0$毒瘤

自闭

upd:貌似数据改了续了30分233

```c++
#include<cstdio>
#pragma comment(linker, "/STACK:256000000,256000000")
using namespace std;
#define N 300050
struct edge{int t,next;}ed[N*2];
int head[N],cnt,dep[N],f[N][21],n,m,k,a,b,lb[N],rb[N],ct;
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){lb[u]=++ct;f[u][0]=fa;dep[u]=dep[fa]+1;for(int i=1;i<=19;i++)f[u][i]=f[f[u][i-1]][i-1];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);rb[u]=ct;}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=19;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=19;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int dis(int x,int y){return dep[x]+dep[y]-2*dep[LCA(x,y)];}
int jmp(int x,int s){for(int i=19;i>=0;i--)if(s&(1<<i))x=f[x][i];return x;}
struct segt{int l,r,l1,l2;}e[N*4];
void pushdown(int x){if(e[x].l2)e[x<<1].l2=e[x<<1|1].l2=1,e[x<<1].l1=e[x<<1|1].l1=0,e[x].l2=0;}
void build(int x,int l,int r)
{
    e[x].l=l;e[x].r=r;
    if(l==r)return;
    int mid=(l+r)>>1;
    build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void add(int x,int l,int r,int s)
{
    if(e[x].l==l&&e[x].r==r){e[x].l1+=s;return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)add(x<<1,l,r,s);
    else if(mid<l)add(x<<1|1,l,r,s);
    else add(x<<1,l,mid,s),add(x<<1|1,mid+1,r,s);
}
int query(int x,int s)
{
    s-=e[x].l1;
    if(e[x].l2){return s==0?e[x].r-e[x].l+1:0;}
    return query(x<<1,s)+query(x<<1|1,s);
}
int main()
{
    freopen("cuvelia.in","r",stdin);
    freopen("cuvelia.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
    dfs(1,0);build(1,1,n);
    while(m--)
    {
        int fg=1;
        scanf("%d",&k);
        if(k==0){printf("%d\n",n);continue;}
        scanf("%d",&a);
        if(k==1)printf("%d\n",n);
        else
        {
            e[1].l2=1;e[1].l1=0;
            int tmp=k;
            k--;
            while(k--)
            {
                scanf("%d",&b);
                int ds=dis(a,b);
                if(ds&1)fg=0;
                else
                {
                    if(dep[a]<dep[b])a^=b^=a^=b;
                    int tp=jmp(a,ds/2);
                    if(dep[a]==dep[b])
                    {
                        add(1,1,n,1);
                        int s1=jmp(a,ds/2-1),s2=jmp(b,ds/2-1);
                        add(1,lb[s1],rb[s1],-1);add(1,lb[s2],rb[s2],-1);
                    }
                    else
                    {
                        int s1=jmp(a,ds/2-1);
                        add(1,lb[tp],rb[tp],1);add(1,lb[s1],rb[s1],-1);
                    }
                }
            }
            if(!fg)printf("0\n");
            else printf("%d\n",query(1,tmp-1));
        }
    }
}
```


#### R2T1 什么什么仙人掌(cactus)（不记得题目名了）

##### 题意

有一棵n个点树，现在要加m条边，每条边有pi概率不会被加上，求加上去的边中只存在于一个简单环的边数的期望值 mod 998244353

部分分：

n,m<=20

pi=0

n,m<=500

n,m<=5000

m<=5000

n,m<=1e5

##### 玄(du)学(liu)

第一个点直接搜

然后考虑一条边什么时候会有贡献

发现是当它在树上覆盖的路径与其他路径没有边相交

然后对于每一条边暴力是m^2的

考虑优化这个过程

注意到路径交是路径

然后路径上长度为1的子路径个数-长度为2的子路径个数等于路径是否有边

于是在树上打标记前缀和，长度为2的在LCA处map特判

因为有0，所以需要把0分开讨论

然后十分毒瘤

##### 代码

????

```c++
#include<cstdio>
#include<algorithm>
#include<map>
using namespace std;
#define N 1000050
#define mod 998244353
map<long long,pair<int,int> > c1;
int head[N],cnt,f[N][22],dep[N],l1[N][2],l2[N][2],n,m,a,b,c,l[N][3];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs(int u,int fa){f[u][0]=fa;dep[u]=dep[fa]+1;for(int i=1;i<=21;i++)f[u][i]=f[f[u][i-1]][i-1];for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=21;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=21;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int jmp(int x,int s){for(int i=21;i>=0;i--)if(s&(1<<i))x=f[x][i];return x;}
int dfs2(int u,int fa)
{
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)dfs2(ed[i].t,u),l1[u][0]=1ll*l1[u][0]*l1[ed[i].t][0]%mod,l2[u][0]=1ll*l2[u][0]*l2[ed[i].t][0]%mod,l1[u][1]+=l1[ed[i].t][1],l2[u][1]+=l2[ed[i].t][1];
}
int dfs3(int u,int fa)
{
    if(fa)l1[u][0]=1ll*l1[u][0]*l1[fa][0]%mod,l2[u][0]=1ll*l2[u][0]*l2[fa][0]%mod,l1[u][1]+=l1[fa][1],l2[u][1]+=l2[fa][1];
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)dfs3(ed[i].t,u);
}
int main()
{
    freopen("cactus.in","r",stdin);
    freopen("cactus.out","w",stdout); 
    scanf("%d%d",&n,&m);
    for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
    dfs(1,0);
    for(int i=0;i<=n;i++)l1[i][0]=l2[i][0]=1;
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d%d",&a,&b,&c),l[i][0]=a,l[i][1]=b,l[i][2]=c;
        int lc=LCA(a,b),f1=jmp(a,dep[a]-dep[lc]-1),f2=jmp(b,dep[b]-dep[lc]-1);if(f1>f2)f1^=f2^=f1^=f2;
        if(c)l2[a][0]=1ll*l2[a][0]*c%mod,l2[b][0]=1ll*l2[b][0]*c%mod,l2[lc][0]=1ll*l2[lc][0]*pw(c,mod-2)%mod,l2[lc][0]=1ll*l2[lc][0]*pw(c,mod-2)%mod;
        else l2[a][1]++,l2[b][1]++,l2[lc][1]--,l2[lc][1]--;
        if(a!=lc&&b!=lc)if(c)c1[1ll*f1*1000000+f2].first=c1[1ll*f1*1000000+f2].first?1ll*c1[1ll*f1*1000000+f2].first*c%mod:c;else c1[1ll*f1*1000000+f2].second++;
        if(dep[a]-dep[lc]>=2)
        {
            int nt=jmp(a,dep[a]-dep[lc]-1);
            if(c)l1[a][0]=1ll*l1[a][0]*c%mod,l1[nt][0]=1ll*l1[nt][0]*pw(c,mod-2)%mod;
            else l1[a][1]++,l1[nt][1]--;
        }
        if(dep[b]-dep[lc]>=2)
        {
            int nt=jmp(b,dep[b]-dep[lc]-1);
            if(c)l1[b][0]=1ll*l1[b][0]*c%mod,l1[nt][0]=1ll*l1[nt][0]*pw(c,mod-2)%mod;
            else l1[b][1]++,l1[nt][1]--;
        }
    }
    dfs2(1,0);
    dfs3(1,0);
    int ans=0;
    for(int i=1;i<=m;i++)
    {
        a=l[i][0],b=l[i][1],c=l[i][2];
        int lc=LCA(a,b),f1=jmp(a,dep[a]-dep[lc]-1),f2=jmp(b,dep[b]-dep[lc]-1);if(f1>f2)f1^=f2^=f1^=f2;
        int as=1ll*pw(1ll*l2[a][0]*l2[b][0]%mod*pw(l2[lc][0],mod*2-4)%mod,mod-2)%mod,as2=-(l2[a][1]+l2[b][1]-l2[lc][1]*2);
        if(a!=lc&&b!=lc)as=1ll*as*(c1[1ll*f1*1000000+f2].first?c1[1ll*f1*1000000+f2].first:1)%mod,as2+=c1[1ll*f1*1000000+f2].second;
        if(dep[a]-dep[lc]>=2)
        {
            int nt=jmp(a,dep[a]-dep[lc]-1);
            as=1ll*as*l1[a][0]%mod*pw(l1[nt][0],mod-2)%mod;
            as2+=l1[a][1]-l1[nt][1];
        }
        if(dep[b]-dep[lc]>=2)
        {
            int nt=jmp(b,dep[b]-dep[lc]-1);
            as=1ll*as*l1[b][0]%mod*pw(l1[nt][0],mod-2)%mod;
            as2+=l1[b][1]-l1[nt][1];
        }
        if(c==0)as2++;else as=1ll*c*as%mod;
        if(as2)as=0;
        ans=(ans+1ll*pw(as,mod-2)*(mod+1-c))%mod;
//		printf("%d\n",1ll*pw(as,mod-2)*(mod+1-c)%mod);
    }
    printf("%d\n",ans);
}
```


#### R3T1 碱基配对 (base)

##### 题意

定义A串第i位与B串第j位可以匹配为在A的$[i-k,i+k]$中有一位与B的第j位相同

A串长度n,B串长度m,求A串中有多少长度为m的子串可以与B匹配

字符集大小为4(Z,P,S,B)

35% n,m,k<=500

65% n,m,k<=5000

100% n,m,k<=1e5

##### 分析

35%的话枚举暴力匹配就可以了 $O(nmk)$

注意到可以处理字符的前缀和

然后就可以$O(1)$判断 复杂度$O(nm)$ 65分

~~其实数据水这样比标算快几十倍~~

考虑优化判断

先处理A串的每一位能匹配哪些字符

这个可以扫一遍

因为A串以x开头的子串需要与B匹配的话，需要满足对于$0<=i<m,a_{x+i} matches b_i$

也就是说对于所有匹配点$(i,j)$，其中正好有m个满足$i-j==x$

暴力是不行的

考虑把4种字符分开讨论

对于每种字符s，需要统计对于所有i，有多少j满足$a_{j+i}=b_j=s$

这很像一个卷积，只不过是差

所以把一个反过来做NTT/FFT

$O(4nlogn)$还没有暴力快（滑稽

放代(ban)码(zi)

```c++
#include<cstdio>
#include<cstring>
using namespace std;
#define mod 998244353
#define N 525000
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int a[N],b[N],k,ntt[N],as[N],n,m,rev[N],tmp,vl[N];
char s[N],t[N];
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)
    rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(3,(mod-1)/l);
        if(t==-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        for(int k=j,st=1;k<j+(l>>1);k++,st=1ll*st*s1%mod)
        tmp=ntt[k],ntt[k]=(tmp+1ll*ntt[k+(l>>1)]*st)%mod,ntt[k+(l>>1)]=(tmp-1ll*ntt[k+(l>>1)]*st%mod+mod)%mod;
    }
    int tp=t==-1?pw(s,mod-2):1;
    for(int i=0;i<s;i++)a[i]=1ll*tp*ntt[i]%mod;
}
void solve(char st)
{
    int l=1;while(l<=n+m+1)l<<=1;
    for(int i=0;i<l;i++)vl[i]=a[i]=b[i]=0;
    for(int i=0;i<n;i++)
    if(s[i]==st)
    {
        if(i-k<0)vl[0]++;else vl[i-k]++;
        vl[i+k+1]--;
    }
    for(int i=1;i<n;i++)vl[i]+=vl[i-1];
    for(int i=0;i<n;i++)if(vl[i])a[i]=1;
    for(int i=0;i<m;i++)if(t[i]==st)b[m-i]=1;
    dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)a[i]=1ll*a[i]*b[i]%mod;dft(l,a,-1);
    for(int i=0;i<=n-m;i++)as[i]+=a[i+m];
}
int main()
{
    freopen("base.in","r",stdin);
    freopen("base.out","w",stdout);
    scanf("%d%s%s",&k,s,t);n=strlen(s);m=strlen(t);
    solve('Z');solve('P');solve('S');solve('B');
    int ans=0;
    for(int i=0;i<n+m;i++)if(as[i]>=m)ans++;
    printf("%d\n",ans);
}
```



#### R3T2 小凯的疑惑 (xor)

##### 题意

有n个点，每个点点权在 $[0,2^c-1]$ 之间

m次操作，每次给出x，全部点权变为$(a+x)\mod 2^c$然后输出异或最小生成树

部分分：

c<=7

q<=10

x%32==0

c<=11

c<=14,n,m<=2e4

显然可以去重

因为点权相同的点间边权一定是0

然后询问可以取前缀和

然后继续去重

这样n,m都是2^c了（subtask2除外）

subtask1:暴力kruskal$O(2^{3c})$

考虑更优秀的异或最小生成树做法

建trie树

显然边的最高位要尽量低

所以每次在两边取数使异或最小然后分治做

这样是$O(nlog^2n)$的

然后就是$O(c^22^c)$

这样就可以过2,4了

然后发现这个程序需要30s

考虑优化

首先因为我写丑了，所以3号点要2.4s(时限2s)

然后我发现加2^(c-1)是没有意义的

因为这相当于交换最高位的01

所以询问可以/2

然后还是10+s

考虑复杂度在启发式合并那里

因为3层的子树只有2^8个

所以可以记录每个的答案和两两之间合并的答案

然后启发式合并就可以8个8个找

复杂度/8

然后。。。TLE 2.5s

自闭

于是决定更加毒瘤

展开上两层

对于后c-2为相同的一起考虑

每次先合并下c-2层

然后分类讨论2(4)种情况

分开合并上两层

可以再优化

然后TLE 2.1s

然后函数改非递归+inline就过了23333

$O(2^{2c}c^2/32)$应该是这个吧

##### 代码

5.3kb快成最长题了

```c++
#include<cstdio>
using namespace std;
#define N 33333
#define M 555
#pragma GCC optimize("unroll-loops,3")
#define ZYW 1
#define AK +
#define IOI 1
/* ZYW is dsb !!!!!!! ZYW tql !!!!!!!*/
int s[N],q[N],s2[N],n,c,m,sum[N],ans,a,as[N],tp[N],dis[M][M],sp[N],use[N];
inline int Min(int a,int b){return a>b?b:a;}
int mnxor(int l,int r,int s)
{
    if(l==r)return l^s;
    int mid=(l+r)>>1;
    int lbit=l^(mid+1);
    if(s&lbit)if(sum[r]-sum[mid]==0)return mnxor(l,mid,s);else return mnxor(mid+1,r,s);
    else if(sum[mid]-(l==0?0:sum[l-1])==0)return mnxor(mid+1,r,s);else return mnxor(l,mid,s);
}
int mnxor2(int l,int r,int s,int v)
{
    if(sum[r]-((l==0)?0:sum[l-1])==0)return 1e9;
    if(r-l==7)return dis[v][sp[l>>3]];
    int mid=(l+r)>>1;
    int lbit=l^(mid+1);
    if(s&lbit)if(sum[r]-sum[mid]==0)return mnxor2(l,mid,s,v)+lbit;else return mnxor2(mid+1,r,s,v);
    else if(sum[mid]-(l==0?0:sum[l-1])==0)return mnxor2(mid+1,r,s,v)+lbit;else return mnxor2(l,mid,s,v);
}
inline int mnxor3(int l,int r,int s,int v)
{
    if(sum[r]-((l==0)?0:sum[l-1])==0)return 1e9;
    int ans=0;
    while(1)
    {
        if(r-l==7)return ans+dis[v][sp[l>>3]];
        int mid=(l+r)>>1,l2,r2;
        int lbit=l^(mid+1);
        if(s&lbit)if(sum[r]-sum[mid]==0)r2=mid,l2=l,ans+=lbit;else l2=mid+1,r2=r;
        else if(sum[mid]-(l==0?0:sum[l-1])==0)l2=mid+1,r2=r,ans+=lbit;else r2=mid,l2=l;
        l=l2,r=r2;
    }
}
int checkmn(int l,int r,int l1,int r1,int ad)
{
    if(sum[r]-(l==0?0:sum[l-1])==0)return 0;
    if(sum[r1]-(l1==0?0:sum[l1-1])==0)return 0;
    int a=1e9;
    for(register int i=l;i<r;i+=8)
    if(sp[i>>3])
    a=Min(a,mnxor3(l1,r1,i,sp[i>>3]));
    return a+ad;
}
int checkmn2(int l,int r,int l0,int r0,int l1,int r1,int l2,int r2,int ad)
{
    if(sum[r]-(l==0?0:sum[l-1])+sum[r0]-(l0==0?0:sum[l0-1])==0)return 0;
    if(sum[r1]-(l1==0?0:sum[l1-1])+sum[r2]-(l2==0?0:sum[l2-1])==0)return 0;
    int a=1e9;
    if(sum[r1]-(l1==0?0:sum[l1-1])!=0&&sum[r]-(l==0?0:sum[l-1])!=0)a=Min(a,checkmn(l,r,l1,r1,0)+ad/2);
    if(sum[r2]-(l2==0?0:sum[l2-1])!=0&&sum[r]-(l==0?0:sum[l-1])!=0)a=Min(a,checkmn(l,r,l2,r2,0));
    if(sum[r1]-(l1==0?0:sum[l1-1])!=0&&sum[r0]-(l0==0?0:sum[l0-1])!=0)a=Min(a,checkmn(l0,r0,l1,r1,0));
    if(sum[r2]-(l2==0?0:sum[l2-1])!=0&&sum[r0]-(l0==0?0:sum[l0-1])!=0)a=Min(a,checkmn(l0,r0,l2,r2,0)+ad/2);
    return a+ad;
}
void solve(int l,int r)
{
    if(sum[r]-(l==0?0:sum[l-1])==0)return;
    if(r-l==7){int tmp=0;
    for(int i=r;i>=l;i--)tmp=tmp*2+s2[i];
    ans+=tp[tmp];sp[l>>3]=tmp;return;}
    if(l==r)return;
    int mid=(l+r)>>1;
    if(sum[mid]-(l==0?0:sum[l-1])==0){solve(mid+1,r);return;}
    if(sum[r]-sum[mid]==0){solve(l,mid);return;}
    solve(l,mid);solve(mid+1,r);
    ans+=checkmn(l,mid,mid+1,r,mid-l+1);
}
void solve2(int l,int r)
{
    if(l==r)return;
    int mid=(l+r)>>1;
    if(sum[mid]-(l==0?0:sum[l-1])==0){solve2(mid+1,r);return;}
    if(sum[r]-sum[mid]==0){solve2(l,mid);return;}
    solve2(l,mid);solve2(mid+1,r);
    int a=1e9;
    for(int i=l;i<=mid;i++)
    if(s2[i])
    a=Min(a,mnxor(mid+1,r,i));
    ans+=a;
}
int mnxorb(int l,int r,int s)
{
    if(l==r)return l^s;
    int mid=(l+r)>>1;
    int lbit=l^(mid+1);
    if(s&lbit)if(sum[r]-sum[mid]==0)return mnxorb(l,mid,s);else return mnxorb(mid+1,r,s);
    else if(sum[mid]-(l==0?0:sum[l-1])==0)return mnxorb(mid+1,r,s);else return mnxorb(l,mid,s);
}
void solveb(int l,int r)
{
    if(l==r)return;
    int mid=(l+r)>>1;
    if(sum[mid]-(l==0?0:sum[l-1])==0){solveb(mid+1,r);return;}
    if(sum[r]-sum[mid]==0){solveb(l,mid);return;}
    solveb(l,mid);solveb(mid+1,r);
    int a=1e9;
    for(int i=l;i<=mid;i++)
    if(s2[i])
    a=Min(a,mnxorb(mid+1,r,i));
    ans+=a;
}
int main()
{
    freopen("xor.in","r",stdin);
    freopen("xor.out","w",stdout);
    scanf("%d%d",&n,&c);
    for(int i=1;i<=n;i++)
    scanf("%d",&a),s[a]=1;
    scanf("%d",&m);
    for(int i=1;i<=m;i++)scanf("%d",&q[i]),q[i]=(q[i]+q[i-1])%(1<<c-1),use[q[i]]=1;
    for(int i=0;i<1<<8;i++)
    {
        for(int j=0;j<8;j++)s2[j]=bool(i&(1<<j));
        sum[0]=s2[0];
        for(int j=1;j<8;j++)sum[j]=sum[j-1]+s2[j];
        ans=0;
        solve2(0,7);
        tp[i]=ans;
    }
    for(int i=0;i<1<<8;i++)
    {
        for(int j=0;j<1<<8;j++)
        {
            if(i==0||j==0){dis[i][j]=1e9;continue;}
            int ans=1e9;
            for(int k=0;k<8;k++)if(i&(1<<k))for(int l=0;l<8;l++)if(j&(1<<l))ans=Min(ans,k^l);
            dis[i][j]=ans;
        }
    }
    if(c<=7)
    {
        for(int i=0;i<(1<<c-1);i++)
        {
            for(int j=0;j<1<<c;j++)
            s2[(j+i)&((1<<c)-1)]=s[j];
            sum[0]=s2[0];
            for(int j=1;j<1<<c;j++)sum[j]=sum[j-1]+s2[j];
            ans=0;
            solveb(0,(1<<c)-1);
            as[i]=ans;
        }
        for(int i=1;i<=m;i++)printf("%d\n",as[q[i]]);
    }
    else
    for(int i=1;i<=m;i++)
    {
        if(as[q[i]]){printf("%d\n",as[q[i]]);continue;}
        int tp=q[i]%(1<<c-2);
        for(int j=0;j<1<<c;j++)
        s2[(j+tp)&((1<<c)-1)]=s[j];
        sum[0]=s2[0];sp[0]=0;
        for(int j=1;j<1<<c;j++)sum[j]=sum[j-1]+s2[j],sp[j]=0;
        ans=0;
        solve(0,(1<<c-2)-1);solve(1<<c-2,(1<<c-1)-1);solve(1<<c-1,(3<<c-2)-1);solve(3<<c-2,(1<<c)-1);
        if(use[tp])as[tp]=ans+checkmn(0,(1<<c-2)-1,1<<c-2,(1<<c-1)-1,1<<c-2)+checkmn(1<<c-1,(3<<c-2)-1,3<<c-2,(1<<c)-1,1<<c-2)+checkmn(1<<c-1,(1<<c)-1,0,(1<<c-1)-1,1<<c-1);
        if(use[tp+(1<<c-2)])as[tp+(1<<c-2)]=ans+checkmn(1<<c-1,(3<<c-2)-1,1<<c-2,(1<<c-1)-1,1<<c-2)+checkmn(0,(1<<c-2)-1,3<<c-2,(1<<c)-1,1<<c-2)+checkmn2(1<<c-2,(1<<c-1)-1,1<<c-1,(3<<c-2)-1,0,(1<<c-2)-1,3<<c-2,(1<<c)-1,1<<c-1);
        printf("%d\n",as[q[i]]);
    }
}
```

#### R3T3 false-false-true(fft)

##### 题面

~~Fast Fast TLE~~

有n+m道题，其中有n道答案为true，m道为false

每答完一道后会告诉你是否正确

求最优决策下错误题目期望值mod 998244353

部分分

n,m<=2000

n,m<=1e5,n=m

n,m<=5e5

###### 神仙题

发现剩余x道true y道false时

如果x>y一定选true

x<y一定选false

所以可以n^2dp处理每个点(x,y)经过概率再乘min(i,j)/(i+j)

设n>m

然后全选true答案是m

所以答案<=m

注意到这样其实是说“如果错一道，剩下错的数量会-1”

但是发现当x=y时，无论对错 错的数量都会-1

所以经过x=y的点时，期望答案会减少1/2

所以答案是$min(n,m)-\sum_{i=1}^{min(n,m)} 经过(i,i)的概率/2$

算个组合数

##### 代码

极其简短

~~与上一题形成鲜明对比~~

```c++
#include<cstdio>
using namespace std;
#define N 1100050
#define mod 998244353
int fr[N],n,m,ans;
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int main()
{
    scanf("%d%d",&n,&m);
    fr[0]=1;for(int i=1;i<=n+m;i++)fr[i]=1ll*fr[i-1]*i%mod;
    for(int i=1;i<=n&&i<=m;i++)
    ans=(ans+1ll*fr[i*2]*pw(fr[i],mod*2-4)%mod*fr[n+m-i*2]%mod*pw(fr[n-i],mod-2)%mod*pw(fr[m-i],mod-2))%mod;
    printf("%d\n",((n>m?m:n)-1ll*ans*fr[n]%mod*fr[m]%mod*pw(fr[n+m],mod-2)%mod*499122177%mod+mod)%mod);
}
```


不要问我为什么没有R4。。菜鸡只会R4T1

#### R5T1 计算(calc)

##### 题意

给定正整数n,**非负整数**m,我们说k是特殊的，当且仅当k∈[1,n]且将k看成一个字符串后，不含m这个子串。例如：k=123321,m=2332 我们就说k含有m这个子串。

对于所有的k，求$\sum_k e^{k/n}$

部分分：

n<=1e6

n<=1e7

m<=9

n,m<=1e9

时限0.1s

##### 正常分析

注意：以下log以10为底

暴力1：枚举判断

复杂度nlognlogm 可以过n<=1e6

dfs是记录上logm层点可以nlogm 过n<=1e7

考虑数位dp

记录每一位

设$dp[i][0/1]$表示在第i位，是否与上限相等，答案

然后一位一位做

获得m<=9的30分

m不止1位是考虑kmp

记录当前匹配到第j位

O(log^2n*10) 100分

但是其实只有90

因为会有m=0然后不算前缀0

再开一位记录是否前面有值

$dp[i][j][0/1][0/1]$记录在第i位，匹配到m上第j位，是否与上限相等，前面是否有值

O(log^2n*40) 真实100分

还有一个暴力的想法

(因为我不想写kmp)

答案=匹配0次的答案-匹配1次的答案+匹配第2次答案.......

然后2^(logn-logm)枚举匹配做有钦定位的数位dp

O(2^logn * log^2n * 40)

实际上5ms以内

##### 代码

```c++
#include<cstdio>
#include<cmath>
#define N 21
#define double long double
using namespace std;
int n,m,dn,dm,st=1,st2=1,s[N],v[N],y[N];
double dp[N][2][2];
double dfs(int dep,int is,int is2)
{
    if(dp[dep][is][is2]>=0)return dp[dep][is][is2];
    double ans=0;
    if(dep==0)return 1;
    for(int i=0;i<=9;i++)
    if((y[dep]==-1||y[dep]==i||(i==0&&is2==1))&&(is==0||s[dep]>=i)&&((is2==0||i)||y[dep]==-1))
    {
        if(is==0||s[dep]>i)ans+=exp(pow(10,dep-1)/n*i)*dfs(dep-1,0,i==0?is2:0);
        else ans+=exp(pow(10,dep-1)/n*i)*dfs(dep-1,1,i==0?is2:0);
    }
    return dp[dep][is][is2]=ans;
}
int sol(int k){int ans=1;while(k){if(k&1)ans=-ans;k>>=1;}return ans;}
int main()
{
    scanf("%d%d",&n,&m);if(m>n)m=n+1;
    dn=1;s[1]=n%10;while(st<=n)dn++,st*=10,s[dn]=n/st%10;
    dm=1;v[1]=m%10;while(st2<=m)dm++,st2*=10,v[dm]=m/st2%10;
    if(m==0)dm=2;
    double as=0;
    for(int i=0;i<1<<(dn-dm+1);i++)
    {
        int fg=0;
        for(int j=1;j<=dn;j++)y[j]=-1;
        for(int j=0;j<=dn;j++)dp[j][0][0]=dp[j][1][0]=dp[j][0][1]=dp[j][1][1]=-1;
        for(int j=1;j<=dn-dm+1;j++)
        if(i&(1<<j-1))
        {
            for(int k=j;k<j+dm-1;k++)
            {
                if(y[k]==-1)y[k]=v[k-j+1];
                else if(y[k]!=v[k-j+1])fg=1;
            }
        }
        if(fg==1)continue;
        as+=sol(i)*dfs(dn,1,1);
    }
    printf("%.3Lf\n",as-1);
}
```

#### R5T2 移动(move)

##### 玄学题面

n个位置形成环，一开始第i张牌在ai位置上，牌移动一格花费为1，求最小花费

部分分：

n<=3

n<=9

n<=16

n<=100

n<=3000

n<=1e5

n<=1e6

##### .

这题很费用流

然而直接写只有30-50分

如果n条边都有流的话

当它形成环，显然不优

否则，n条边至少有n点流量，至少一个点没有入度有额外一点流量

所以不合法

因此一定可以断开

暴力枚举是n^2的50分

设前缀和sum[i]

根据均分纸牌的套路

如果在第i处断开

则答案是$\sum  abs(sum[j]-sum[i]-j+i+(j<i?sum[n]-n:0))$

发现sum[n]=n

所以无视后面一项

所以要取所有$sum[i]-i$的中位数

然后模拟

nlogn

```c++
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1000050
int a[N],n;long long ans;
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&a[0]),a[a[0]]++;a[0]=0;
    for(int i=1;i<=n;i++)a[i]=a[i]+a[i-1]-1;
    sort(a+1,a+n+1);
    for(int i=1;i<=n;i++)ans+=a[i]<a[(n+1)/2]?a[(n+1)/2]-a[i]:a[i]-a[(n+1)/2];
    printf("%lld\n",ans);
}
```



#### R5T3 分离(partition)

有n个数求1到n的集合划分成两个集合且每个集合中都不存在互异元素a,b,c,使a*b=c

求划分方案数%m

多组数据，每次给定n,m

部分分：

n<=3

n<=15

n<=50

n<=1000

n<=1e5,T<=10

##### ？？？

2^n打表可以30分

写个玄学dfs可以打50分

考虑对于后一半的数一定不会对后面的数造成影响

所以只用搜一半，另一半直接算方案数

然后慢慢搜

到n=96时发现是0

得出结论n>95是答案为0

打表

##### 代码

暴力程序

```c++
#include<cstdio>
#include<cstring>
using namespace std;
#define N 233
int sa[N],sb[N],n;
long long ans,as[N];
int checka(int s){for(int i=2;i*i<s;i++)if(s%i==0&&sa[i]&&sa[s/i])return 0;return 1;}
int checkb(int s){for(int i=2;i*i<s;i++)if(s%i==0&&sb[i]&&sb[s/i])return 0;return 1;}
void dfs(int dep,int n)
{
    if(dep==n/2+1)
    {
        long long tp=1;
        for(int i=n/2+1;i<=n;i++)
        tp=tp*(checka(i)+checkb(i));
        ans+=tp;return;
    } 
    if(checka(dep))
    {
        sa[dep]=1;
        dfs(dep+1,n);
        sa[dep]=0;
    }
    if(checkb(dep))
    {
        sb[dep]=1;
        dfs(dep+1,n);
        sb[dep]=0;
    }
}
int check(int s)
{
    if(s<10)return 233;
    int tp=0;
    for(int i=2;i<s;i++)
    for(int j=i+1;j<s;j++)
    if(i*j==s)tp++;
    return tp;
}
int main()
{
    for(int i=1;i<=100;i++)
    {ans=0;
    if(check(i)==0)
    ans=as[i-1]*2;
    else dfs(1,i);printf("ans[%d]=%lldll;\n",i,as[i]=ans);}
}
```

跑2h就有了下面这个程序

```c++
#include<cstdio>
using namespace std;
long long ans[1100000];
int n,m,T;
int main()
{
    ans[1]=2ll;
    ans[2]=4ll;
    ans[3]=8ll;
    ans[4]=16ll;
    ans[5]=32ll;
    ans[6]=48ll;
    ans[7]=96ll;
    ans[8]=144ll;
    ans[9]=288ll;
    ans[10]=432ll;
    ans[11]=864ll;
    ans[12]=960ll;
    ans[13]=1920ll;
    ans[14]=2880ll;
    ans[15]=4320ll;
    ans[16]=7296ll;
    ans[17]=14592ll;
    ans[18]=16512ll;
    ans[19]=33024ll;
    ans[20]=34368ll;
    ans[21]=53120ll;
    ans[22]=79680ll;
    ans[23]=159360ll;
    ans[24]=111360ll;
    ans[25]=222720ll;
    ans[26]=334080ll;
    ans[27]=541440ll;
    ans[28]=685440ll;
    ans[29]=1370880ll;
    ans[30]=887040ll;
    ans[31]=1774080ll;
    ans[32]=2200320ll;
    ans[33]=3655680ll;
    ans[34]=5483520ll;
    ans[35]=8232192ll;
    ans[36]=10851840ll;
    ans[37]=21703680ll;
    ans[38]=32555520ll;
    ans[39]=54259200ll;
    ans[40]=73958400ll;
    ans[41]=147916800ll;
    ans[42]=75340800ll;
    ans[43]=150681600ll;
    ans[44]=207636480ll;
    ans[45]=355000320ll;
    ans[46]=532500480ll;
    ans[47]=1065000960ll;
    ans[48]=348364800ll;
    ans[49]=696729600ll;
    ans[50]=952197120ll;
    ans[51]=1586995200ll;
    ans[52]=2221793280ll;
    ans[53]=4443586560ll;
    ans[54]=2113413120ll;
    ans[55]=3808788480ll;
    ans[56]=3901685760ll;
    ans[57]=6502809600ll;
    ans[58]=9754214400ll;
    ans[59]=19508428800ll;
    ans[60]=9057484800ll;
    ans[61]=18114969600ll;
    ans[62]=20379340800ll;
    ans[63]=31701196800ll;
    ans[64]=53778816000ll;
    ans[65]=115644672000ll;
    ans[66]=54021427200ll;
    ans[67]=108042854400ll;
    ans[68]=152165744640ll;
    ans[69]=253609574400ll;
    ans[70]=398529331200ll;
    ans[71]=797058662400ll;
    ans[72]=489104179200ll;
    ans[73]=978208358400ll;
    ans[74]=1467312537600ll;
    ans[75]=2445520896000ll;
    ans[76]=3423729254400ll;
    ans[77]=5579410636800ll;
    ans[78]=2575112601600ll;
    ans[79]=5150225203200ll;
    ans[80]=2575112601600ll;
    ans[81]=5150225203200ll;
    ans[82]=7725337804800ll;
    ans[83]=15450675609600ll;
    ans[84]=7023034368000ll;
    ans[85]=13042778112000ll;
    ans[86]=19564167168000ll;
    ans[87]=32606945280000ll;
    ans[88]=26085556224000ll;
    ans[89]=52171112448000ll;
    ans[90]=34780741632000ll;
    ans[91]=57967902720000ll;
    ans[92]=81155063808000ll;
    ans[93]=135258439680000ll;
    ans[94]=202887659520000ll;
    ans[95]=376791367680000ll;
    scanf("%d",&T);
    while(T--)
    scanf("%d%d",&n,&m),printf("%lld\n",ans[n]%m);
}
```

#### R6T1 One？One! (one)

##### 题面

定义oneness(i)=i大于1且每一位都是1的约数个数

求$\sum_{i=1}^n oneless(i)$

n有len位，随机生成

给定$s_0,d_i=(s_i>>10)\mod 10,s_i=(747796405s_{i-1}-1403630843) \mod2^{32} $

n是$d_0d_1d_2 \cdots  d_n$从高位到低位

部分分：

len<=300

len<=2000

len<=10000

len<=250000

##### 我也不知道要写啥

枚举每个约数，那么答案是$\sum_{d=2}^{len}n/111...1​$

暴力高精度除法n^3

使用多项式除法可以n^2

考虑分子分母乘9

$\sum_{d=2}^{len+1}n*9/(10^{len}-1)$

考虑怎么计算

$s/(10^x-1)=s/10^x+(s/10^x+s\mod10^x)/(10^x-1)$

于是可以递归

计算前半段

考虑第i位对答案第j位的贡献

发现在计算len=s时所有x-y%s==0都会有贡献

所以是x-y的约数个数和-1

于是可以NTT/FFT

因为约数个数和是nlnn的，所以ntt不会爆

然后考虑后半段

对于len=s

将从低到高每s位加起来再%(10^len-1)

~~高精度加法~~

因为n随机，取前15位加起来即可

O(nlogn*15)

##### 代码

```c++
#include<cstdio>
using namespace std;
#define N 555000
#define mod 998244353
unsigned int st;
int l,s[N],t[N],ans[N],rev[N],ntt[N],as[N];
int pw(int a,long long p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)
    rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(3,(mod-1)/l);
        if(t==-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        for(int k=j,st=1;k<j+(l>>1);k++,st=1ll*st*s1%mod)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*st%mod;
            ntt[k]=s1+s2-(s1+s2>=mod?mod:0);
            ntt[k+(l>>1)]=s1-s2+(s1<s2?mod:0);
        }
    }
    int tp=t==-1?pw(s,mod-2):1;
    for(int i=0;i<s;i++)a[i]=1ll*tp*ntt[i]%mod;
}
void NTT(int *a,int *b,int l,int *as){dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)as[i]=1ll*a[i]*b[i]%mod;dft(l,a,-1);dft(l,b,-1);dft(l,as,-1);}
void solve2(int x)
{
    long long tp1=0,tp2=0,sp=1;
    for(int i=1;i<=x&&i<=15;i++)sp*=10;sp=sp-1;
    for(int i=x-1;i-x<l;i+=x)
    {
        long long st=0;
        for(int j=i;i-j<15&&i-j<x&&j>=0;j--)st=st*10+s[j];
        tp1+=st;
        tp2+=tp1/sp;tp1%=sp;
    }
    ans[0]+=tp2;
}
int main()
{
    freopen("one.in","r",stdin);
    freopen("one.out","w",stdout);
    scanf("%d%u",&l,&st);
    for(int i=0;i<l;i++)
    {
        s[l-i-1]=st/1024%10;
        st=st*747796405-1403630843;
    }
    for(int i=0;i<l;i++)s[i]*=9;
    for(int i=0;i<l;i++)s[i+1]+=s[i]/10,s[i]%=10;
    if(s[l])l++;
    for(int i=2;i<=l;i++)
    for(int j=i;j<=l;j+=i)t[l-j]++;
    int q=1;while(q<=l*2)q<<=1;
    NTT(s,t,q,as);
    for(int i=0;i<l;i++)ans[i]+=as[i+l];
    for(int i=2;i<=l;i++)solve2(i);
    for(int i=0;i<=l;i++)ans[i+1]+=ans[i]/10,ans[i]%=10;
    int fg=0;
    for(int i=l;i>=0;i--)
    {
        fg|=ans[i];
        if(fg)printf("%d",ans[i]);
    }
}
```



#### R6T2 Two？Two! (two)

给一个长度为n的序列，需要分成两个子序列

最小化$\sum max(a_1...a_x)-a_x$+$\sum max(b_1...b_x)-b_x$

例如 5,1,6,2,7,3，划分为5,6,7和1,2,3答案为0

子任务：

n<=10

n<=500

n<=2000

n<=1e5

n<=5e5

2s

##### .

先离散化

设$dp[i][j]$表示第一队max为i，第j队max为j，最小答案

转移显然，枚举当前这个人在哪边

n^3

由于显然有一边max为当前max

所以可以压一位

dp转移：

如果f[i]<=mx

$1<=j<=f[i] ,dp[i][j]=dp[i-1][j]+v[mx]-v[i]$//小于i的放到上面

$dp[i][f[i]]=min(dp[i-1][1...f[i]])$//小于i的放到下面

$f[i]<j<=mx,dp[i][j]=dp[i-1][j]+v[j]-v[i]$//大于i的一定放到下面最优

如果f[i]>mx

显然小于mx的dp不变（放到上面）

$dp[i][f[i]]=min(dp[i-1][1..mx])$其他的当前放到下面

然后mx=f[i]

这样就是区间加，单点赋值，区间求min，**区间加一次函数**

~~这做个鬼啊。。。~~

考虑暴力分块，每次维护凸包/斜率优化队列

边角块暴力重构

nsqrtn可以75分

因为重构常数很大，size可以小一点（size=90）

正解维护玄学线段树，不会

##### 75代码

```c++
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500050
#pragma GCC optimize("unroll-loops,-Ofast")
inline long double Abs(long double a){return a>0?a:-a;}
int n,fr[N],sz=90,hd[N],tl[N],q[N],mx;long long l[N],l2[N],as[N],v[N],t[N];
inline bool check(long long a,long long b,long long c,long long d){if(Abs(1.0*a*d-1.0*c*b)>1e-7)return 1.0*a*d-1.0*c*b>0;return (a*d-b*c)>=0;}
void rebuild(int x)
{
//	return;
    int lb=(x-1)*sz+1,rb=min(x*sz,n);
    hd[x]=lb+1;tl[x]=lb;
    for(int i=rb;i>=lb;i--)
    {
        while(hd[x]<tl[x]&&!check(as[q[tl[x]]]-as[q[tl[x]-1]],v[q[tl[x]-1]]-v[q[tl[x]]],as[q[tl[x]]]-as[i],v[i]-v[q[tl[x]]]))tl[x]--;
        q[++tl[x]]=i;
    }
    while(hd[x]<tl[x]&&(as[q[hd[x]+1]]-as[q[hd[x]]])<=0)hd[x]++;
}
void pushdown(int x)
{
    int lb=(x-1)*sz+1,rb=min(x*sz,n);
    for(int i=lb;i<=rb;i++)as[i]+=l[x]+l2[x]*v[i];
    l[x]=l2[x]=0;
}
long long querymn(int r)
{
    long long as1=1e17;
    for(int i=1;i<fr[r];i++)
    as1=min(as1,as[q[hd[i]]]+l[i]+l2[i]*v[q[hd[i]]]);
    pushdown(fr[r]);rebuild(fr[r]);
    for(int i=sz*(fr[r]-1)+1;i<=r;i++)
    as1=min(as1,as[i]);
    return as1;
}
void qadd(int lb,int rb,long long ad)
{
    if(fr[rb]-fr[lb]<2)
    {
        pushdown(fr[lb]);pushdown(fr[rb]);
        for(int i=lb;i<=rb;i++)as[i]+=ad;
        rebuild(fr[lb]);rebuild(fr[rb]);
        return;
    }
    for(int i=fr[lb]+1;i<fr[rb];i++)l[i]+=ad;
    pushdown(fr[lb]);pushdown(fr[rb]);
    for(int i=lb;i<=fr[lb]*sz;i++)as[i]+=ad;
    for(int i=fr[rb]*sz-sz+1;i<=rb;i++)as[i]+=ad;
    rebuild(fr[lb]);rebuild(fr[rb]);
}
void qadd2(int lb,int rb,long long ad)
{
    if(fr[rb]-fr[lb]<2)
    {
        pushdown(fr[lb]);pushdown(fr[rb]);
        for(int i=lb;i<=rb;i++)as[i]+=v[i];
        rebuild(fr[lb]);rebuild(fr[rb]);
        return;
    }
    for(int i=fr[lb]+1;i<fr[rb];i++)
    {
        l2[i]++;
        int x=i;
        while(hd[x]<tl[x]&&(as[q[hd[x]+1]]-as[q[hd[x]]])<=l2[x]*(v[q[hd[x]]]-v[q[hd[x]+1]]))hd[x]++;
    }
    pushdown(fr[lb]);pushdown(fr[rb]);
    for(int i=lb;i<=fr[lb]*sz;i++)as[i]+=v[i];
    for(int i=fr[rb]*sz-sz+1;i<=rb;i++)as[i]+=v[i];
    rebuild(fr[lb]);rebuild(fr[rb]);
}
int main()
{
    freopen("two.in","r",stdin);
    freopen("two.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%lld",&t[i]),v[i]=t[i];
    sort(v+1,v+n+1);int s=unique(v+1,v+n+1)-v-1;
    for(int i=1;i<=n;i++)t[i]=lower_bound(v+1,v+s+1,t[i])-v;
    mx=t[1];
    for(int i=1;i<=n;i++)fr[i]=(i-1)/sz+1,as[i]=i==1?0:1e18;
    for(int i=1;i<=fr[n];i++)rebuild(i);
    for(int i=2;i<=n;i++)
    {
        if(t[i]>mx)
        {
            long long mn=querymn(mx);
            pushdown(fr[mx]);as[mx]=mn;rebuild(fr[mx]);mx=t[i];
        }
        else
        {
            long long mn=querymn(t[i]);
            qadd(1,t[i],v[mx]-v[t[i]]);
            pushdown(fr[t[i]]);as[t[i]]=min(as[t[i]],mn);rebuild(fr[t[i]]);
            qadd(t[i]+1,mx,-v[t[i]]);
            qadd2(t[i]+1,mx,1);
        }
    }
    for(int i=1;i<=fr[n];i++)pushdown(i);
    printf("%lld\n",querymn(mx));
}
```



#### R6T3 More?More! (more)

##### 题意

n个人两两比赛，编号小的人打编号大的人胜率为p

求对于所有i，可以选出一个大小为i的人集合使得集合内的人赢集合外的人的概率

mod 998244353

部分分：

n<=3

n<=10

n<=2000

n<=1e5

**n<=1e6**

1s

##### 分析

显然无法同时选出多个大小相同的集合满足条件

暴力dp表示前i个钦定j个概率

直接转移

考虑一种选择方案的贡献

设选择为1，否则为0

则方案为一个01串

贡献是(1-p)^(01子序列数量)* p^(10子序列数量)

对于答案式子的一个i，这两种子序列和一定是n*(n-i)

所以就是$\sum ((1-p)/p)^{01子序列数量}*p^{n*(n-i)}​$

考虑怎么计算01子序列数量

每一个1的贡献等于1的下标和减去前面1的数量（包含自己）

对于一个i，后面的和一定为i*(i+1)/2

然后前面可以生成函数NTT求

求$((1-p)/p)^i*x+1$的乘积

再除以$((1-p)/p)^{n*(n+1)/2}$

再乘上$$p^{n*(n-i)}$$

分治NTT nlog^2n 70分

貌似倍增就过了

设前一半为f(x)，要求后一半g(x)（奇数的可以不管最后乘）

发现g(x)是将f(x)中的x替换为$((1-p)/p)^{n/2}x$

所以g(x)可以O(n)求

然后NTT

O(nlogn)可以过

标程神仙

考虑在前面或后面插入

$dp[i][j]=dp[i-1][j-1]*p^{i-j}+dp[i-1][j]*(1-p)^{i-j}$

$dp[i][j]=dp[i-1][j-1]*(1-p)^{i-j}+dp[i-1][j]*p^{i-j}$

所以对于n，$dp[n][i]=dp[n][i-1]*(p^{n-i+1}-(1-p)^{n-i+1})/(p^i-(1-p)^i)$

特判p==499122177(1/2)

和倍增一样都是nlogn

##### 代码

分治NTT

```c++
#include<cstdio>
using namespace std;
#define N 2100050
#define mod 998244353
int pw(int a,long long p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int fr[N],n,p,ntt[N],rev[N],st[22][2][N],s1[111][2],tp;
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)
    rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(3,(mod-1)/l);
        if(t==-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        for(int k=j,st=1;k<j+(l>>1);k++,st=1ll*st*s1%mod)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*st%mod;
            ntt[k]=s1+s2-(s1+s2>=mod?mod:0);
            ntt[k+(l>>1)]=s1-s2+(s1<s2?mod:0);
        }
    }
    int tp=t==-1?pw(s,mod-2):1;
    for(int i=0;i<s;i++)a[i]=1ll*tp*ntt[i]%mod;
}
void NTT(int *a,int *b,int l,int *as){dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)as[i]=1ll*a[i]*b[i]%mod;dft(l,as,-1);}
void cdq(int d,int l,int r,int s)
{
    if(l==r){st[d][s][0]=1,st[d][s][1]=pw(tp,l);s1[d][s]=2;return;}
    int mid=(l+r)>>1;
    cdq(d+1,l,mid,0);cdq(d+1,mid+1,r,1);
    s1[d][s]=s1[d+1][0]+s1[d+1][1];int tp=1;while(tp<s1[d+1][0]+s1[d+1][1])tp<<=1;
    for(int i=s1[d+1][0];i<tp;i++)st[d+1][0][i]=0;
    for(int i=s1[d+1][1];i<tp;i++)st[d+1][1][i]=0;
    NTT(st[d+1][0],st[d+1][1],tp,st[d][s]);
}
int main()
{
    freopen("more.in","r",stdin);
    freopen("more.out","w",stdout);
    scanf("%d%d",&n,&p);
    fr[0]=1;for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod;
    if(p==499122177){for(int i=1;i<n;i++)printf("%d ",1ll*fr[n]*pw(fr[n-i],mod-2)%mod*pw(fr[i],mod-2)%mod*pw(pw(p,n-i),i)%mod);return 0;}
    if(p==0||p==1){for(int i=1;i<n;i++)printf("1 ");return 0;}
    tp=1ll*p*pw(mod+1-p,mod-2)%mod;
    cdq(1,1,n,0);
    for(int i=1;i<n;i++)printf("%d ",1ll*st[1][0][i]*pw(pw(tp,mod-2),1ll*i*(i+1)/2)%mod*pw(mod+1-p,1ll*(n-i)*i)%mod);
}
```

倍增（卡常）

```c++
#include<cstdio>
using namespace std;
#define N 2221111
#define mod 998244353
int pw(int a,long long p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int fr[N],n,p,ntt[N],rev[N],s1,tp,sa[N],sb[N],as[N],ans[N],lg[N],ps[2][22][N];
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)
    rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        for(int j=0;j<s;j+=l)
        for(int k=j;k<j+(l>>1);k++)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*ps[(t+1)>>1][lg[l]][k-j]%mod;
            ntt[k]=s1+s2-(s1+s2>=mod?mod:0);
            ntt[k+(l>>1)]=s1-s2+(s1<s2?mod:0);
        }
    }
    int tp=t==-1?pw(s,mod-2):1;
    for(int i=0;i<s;i++)a[i]=1ll*tp*ntt[i]%mod;
}
void NTT(int *a,int *b,int l,int *as){dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)as[i]=1ll*a[i]*b[i]%mod;dft(l,as,-1);}
void cdq(int r)
{
    if(r==1){as[0]=1,as[1]=tp;s1=2;return;}
    cdq(r>>1);
    s1=s1*2-1;int t=1;while(t<s1)t<<=1;int tp3=1,tp4=pw(tp,r>>1);
    for(int i=0;i<t;i++)sa[i]=as[i],sb[i]=1ll*as[i]*tp3%mod,tp3=1ll*tp3*tp4%mod;
    NTT(sa,sb,t,as);
    if(r&1)
    {
        int tp3=pw(tp,r);
        for(int i=s1;i>=0;i--)
        as[i+1]=(as[i+1]+1ll*tp3*as[i])%mod;
        s1++;
    }
}
int main()
{
    freopen("more.in","r",stdin);
    freopen("more.out","w",stdout);
    scanf("%d%d",&n,&p);
    fr[0]=1;for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod;
    if(p==499122177){for(int i=1;i<n;i++)printf("%d ",1ll*fr[n]*pw(fr[n-i],mod-2)%mod*pw(fr[i],mod-2)%mod*pw(pw(p,n-i),i)%mod);return 0;}
    if(p==0||p==1){for(int i=1;i<n;i++)printf("1 ");return 0;}
    tp=1ll*p*pw(mod+1-p,mod-2)%mod;
    for(int i=2;i<=n*2;i++)lg[i]=lg[i>>1]+1;
    for(int i=1;i<=20;i++)
    for(int j=0;j<2;j++)
    {
        int sp=pw(3,(mod-1)>>i);
        if(j==0)sp=pw(sp,mod-2);
        ps[j][i][0]=1;
        for(int k=1;k<1<<i;k++)ps[j][i][k]=1ll*ps[j][i][k-1]*sp%mod;
    }
    int itp=pw(tp,mod-2);
    cdq(n);
    for(int i=1;i<=(n+1)/2;i++)ans[i]=ans[n-i]=1ll*as[i]*pw(itp,1ll*i*(i+1)/2)%mod*pw(mod+1-p,1ll*(n-i)*i)%mod;
    for(int i=1;i<n;i++)printf("%d ",ans[i]);
}
```

std写法（飞快）（但是好像没有我卡过常的倍增快）

```c++
#include<cstdio>
using namespace std;
#define N 2000040
#define mod 998244353
int pw(int a,long long p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int fr[N],n,p;
int main()
{
    freopen("more.in","r",stdin);
    freopen("more.out","w",stdout);
    scanf("%d%d",&n,&p);
    fr[0]=1;for(int i=1;i<=n*2;i++)fr[i]=1ll*fr[i-1]*i%mod;
    if(p==499122177){for(int i=1;i<n;i++)printf("%d ",1ll*fr[n]*pw(fr[n-i],mod-2)%mod*pw(fr[i],mod-2)%mod*pw(pw(p,n-i),i)%mod);return 0;}
    if(p==0||p==1){for(int i=1;i<n;i++)printf("1 ");return 0;}
    int tp=1;
    for(int i=1;i<n;i++)
    tp=1ll*tp*(pw(p,n-i+1)-pw(mod+1-p,n-i+1)+mod)%mod*pw((pw(p,i)-pw(mod+1-p,i)+mod)%mod,mod-2)%mod,printf("%d ",tp);
}
```



#### R7T1 全连(fc)

##### 题面

~~题面与全连没有关系~~

n个音符，第i个音符在第i时刻出现，且有两个属性$t_i,d_i$

意思是如果接住这个音符，收益为$t_i*d_i$，且不能接$(i-t_i,i+t_i)$内的其他音符

求最大收益

子任务：

n<=20

n<=5000

n<=1e5

n<=2e5

n<=5e5

n<=8e5

所有t相等,n<=1e5/1e6

n<=1e6

**1s**（其实就是想说需要读优）

##### 分析

设$dp[i]$表示接住第i个音符的最大收益

则$dp[i]=min(dp[j][j<=i-t[i]][i>=j+t[j]])$

于是就有了n^2算法

发现第一个限制很好做，但第二个不好做

所以每次求完以后在j+t[j]时刻加进BIT即可

O(nlogn)

##### 代码

良心题

```c++
#include<cstdio>
#include<vector>
using namespace std;
#define N 1000050
long long tr[N*2];
int n,s[N],v[N];
struct sth{int b;long long c;};
vector<sth> st[N];
inline long long Max(long long a,long long b){return a>b?a:b;}
void add(int x,long long y){for(int i=x;i<=n;i+=i&-i)tr[i]=Max(tr[i],y);}
long long que(int x){long long ans=0;for(int i=x;i;i-=i&-i)ans=Max(ans,tr[i]);return ans;}
int read(){char c=getchar();while(c<'0'||c>'9')c=getchar();int ans=0;while(c>='0'&&c<='9')ans=ans*10+c-'0',c=getchar();return ans;}
int main()
{
    freopen("fc.in","r",stdin);
    freopen("fc.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)s[i]=read();
    for(int i=1;i<=n;i++)v[i]=read();
    for(int i=1;i<=n;i++)
    {
        int sz=st[i].size();
        for(int j=0;j<sz;j++)
        {
            sth t=st[i][j];
            add(t.b,t.c);
        }
        long long mx=0;
        if(s[i]<i)mx=que(i-s[i]);
        mx+=1ll*s[i]*v[i];
        st[i+s[i]>n?n+1:i+s[i]].push_back((sth){i,mx});
    }
    {
        int sz=st[n+1].size();
        for(int j=0;j<sz;j++)
        {
            sth t=st[n+1][j];
            add(t.b,t.c);
        }
    }
    printf("%lld\n",que(n));
}
```



#### R7T2 原样输出(copy)

##### 极其简化的题意

有n个串，每个串取一个可以为空的子串按顺序拼起来输出

求输出不同输出数或者输出所有输出和输出数

部分分：

n=1

n<=2

输入文件小于1MB，输出文件小于200MB

n<=1e6

字符集大小4(AGCT)

##### 输出题

首先这输出就很毒瘤

选一个子串，考虑上SAM

于是就变成了n个SAM，每一个节点可以向后面的SAM有那个转移的SAM转移

例如一个SAM根节点有一个A的转移

那么前面一个点就可以向这个转移点“连”A的转移

但是这样一个点会有很多字符相同的转移

考虑只需要记录输出

所以可以贪心地连

如果这个点在当前SAM中有这个转移就不向外连

否则只向最近的SAM连

可以发现这是正确的

于是这又是一个优秀的DAG了

然后在上面dfs

找最近可以用类似子序列自动机的方案做

##### 代码

我觉得难点在于SAM板子和200MB的输出

~~开个char[2e8]fwrite成功MLE~~

~~开1e8就可以了~~

```c++
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 2000500
#define outsz 105000000 
#define mod 1000000007
int ch[N][4],len[N],fail[N],cnt,n,nxt[N][4],l,dp[N],k,ct=-1;
char s[N],st[N],q[outsz];
struct sam{
    int rt,last;
    void init(){rt=last=++cnt;}
    void insert(char v)
    {
        int tp;
        if(v=='A')tp=0;else if(v=='C')tp=1;else if(v=='G')tp=2;else tp=3;
        int st=++cnt,s1=last;len[st]=len[s1]+1;last=st;
        while(!ch[s1][tp]&&s1)ch[s1][tp]=st,s1=fail[s1];
        if(s1==0){fail[st]=rt;return;}
        int nt=ch[s1][tp];
        if(len[nt]==len[s1]+1)fail[st]=nt;
        else
        {
            int cl=++cnt;len[cl]=len[s1]+1;
            for(int i=0;i<4;i++)ch[cl][i]=ch[nt][i];
            fail[st]=cl;fail[cl]=fail[nt];fail[nt]=cl;
            while(ch[s1][tp]==nt&&s1)ch[s1][tp]=cl,s1=fail[s1];
        }
    }
}t[N>>1];
FILE *p1=fopen("copy.out","w");
int dfs1(int s,int fr)
{
    if(dp[s]!=-1)return dp[s];
    int as=1;
    for(int i=0;i<4;i++)
    {
        if(ch[s][i])
        as=(as+dfs1(ch[s][i],fr))%mod;
        else
        if(nxt[fr][i]!=-1)
        as=(as+dfs1(ch[t[nxt[fr][i]].rt][i],nxt[fr][i]))%mod;
    }
    return dp[s]=as;
}
int dfs2(int s,int fr,int dep)
{
    for(int i=1;i<dep;i++)q[++ct]=st[i];q[++ct]='\n';
    if(ct>1e8)fwrite(q,1,ct+1,p1),ct=0;
    int as=1;
    for(int i=0;i<4;i++)
    {
        if(i==0)st[dep]='A';if(i==1)st[dep]='C';if(i==2)st[dep]='G';if(i==3)st[dep]='T';
        if(ch[s][i])
        as=(as+dfs2(ch[s][i],fr,dep+1))%mod;
        else
        if(nxt[fr][i]!=-1)
        as=(as+dfs2(ch[t[nxt[fr][i]].rt][i],nxt[fr][i],dep+1))%mod;
        st[dep]=0;
    }
    return dp[s]=as;
}
int main()
{
    freopen("copy.in","r",stdin);
//	freopen("copy.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        scanf("%s",s+1);l=strlen(s+1);
        t[i].init();
        for(int j=1;j<=l;j++)t[i].insert(s[j]);
    }
    for(int i=0;i<4;i++)nxt[n][i]=-1;
    for(int i=n-1;i>=1;i--)
    {
        for(int j=0;j<4;j++)
        nxt[i][j]=ch[t[i+1].rt][j]==0?nxt[i+1][j]:i+1;
    }
    scanf("%d",&k);
    memset(dp,-1,sizeof(dp));
    if(k==1)
    dfs2(t[1].rt,1,1);
    else dfs1(t[1].rt,1);
    int st=dp[t[1].rt];
    int fg=0,tp=1e9;
    while(tp)
    {
        int r=st/tp%10;
        fg|=r;
        if(fg||tp==1)q[++ct]=r+'0';
        tp/=10;
    }
    fwrite(q,1,ct+1,p1);
}
```

#### R7T3 不同的缩写 (diff)

##### 简短题面

有n个字符串，现在要给每个字符串分配一个简称，使得：

简称是自己的子序列

简称互不相同

求是否存在方案，若存在，输出最长简称长度的最小值以及方案

部分分：

n<=4

n>=100，数据随机

n<=300

##### 分析

~~真正的子序列自动机来了~~

考虑每个串建子序列自动机，这样就可以快速找出子序列 

考虑二分长度，每个串向长度小于二分值的子序列连边，用hash自然溢出存（长度300卡不掉）

但是子序列可能非常多

根据二分图匹配的性质，只需要连n个即可保证匹配最大

最后记录每个点连向了哪个hash值再搜一遍

复杂度n^3左右

##### 代码(992ms/1s)

巨慢无比

```c++
#include<cstdio>
#include<map>
#include<algorithm>
#include<cstring>
#include<queue>
using namespace std;
#define N 305
#define M 1222222
#define c 37
#define ull unsigned long long
#pragma GCC optimize("unroll-loops,-Ofast")
map<int,ull> tle;
map<ull,int> mle;
int ch[N][N][27],n,l[N],head[N*300],cur[N*300],dep[N*300],cnt,ct[N],ct2;
char v[N][N],st[N];
ull ans[N];
struct edge{int t,next,v;}ed[M*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;}
bool bfs(int s,int t)
{
    memset(dep,-1,sizeof(dep));
    memcpy(cur,head,sizeof(cur));
    queue<int> tp;
    tp.push(s);dep[s]=1;
    while(!tp.empty())
    {
        int r=tp.front();tp.pop();
        for(int i=head[r];i;i=ed[i].next)
        if(ed[i].v&&dep[ed[i].t]==-1)
        {
            dep[ed[i].t]=dep[r]+1,tp.push(ed[i].t);
            if(ed[i].t==t)return 1;
        }
    }
    return 0;
}
int dfs(int u,int t,int v)
{
    if(!v)return 0;
    if(u==t)return v;
    int ans=0,tmp;
    for(int& i=cur[u];i;i=ed[i].next)
    if(dep[ed[i].t]==dep[u]+1&&(tmp=dfs(ed[i].t,t,min(v,ed[i].v))))
    {
        ans+=tmp,v-=tmp;
        ed[i].v-=tmp;ed[i^1].v+=tmp;
        if(!v)return ans;
    }
    return ans;
}
int dinic(int s,int t){int ans=0;while(bfs(s,t))ans+=dfs(s,t,1e9);return ans;}
void dfs(int s,ull h,int dep,int mx,int f)
{
    if(ct[f]>=n)return;
    if(dep)
    {
        ct[f]++;
        int st=mle[h]==0?mle[h]=++ct2:mle[h];
        tle[st]=h;
        adde(f+2,st,1);
    }
    if(dep<mx)
    for(int i=0;i<26;i++)
    if(ch[f][s][i])
    dfs(ch[f][s][i],h*c+i+1,dep+1,mx,f);
}
bool check(int d)
{
    for(int i=1;i<=n;i++)ct[i]=0;
    memset(head,0,sizeof(head));
    tle.clear();mle.clear();
    cnt=1;ct2=n+2;
    for(int i=1;i<=n;i++)adde(1,i+2,1);
    for(int i=1;i<=n;i++)dfs(0,0,0,d,i);
    for(int i=n+3;i<=ct2;i++)adde(i,2,1);
    if(dinic(1,2)==n)
    {
        for(int i=1;i<=n;i++)
        {
            for(int j=head[i+2];j;j=ed[j].next)
            if(!ed[j].v)
            ans[i]=tle[ed[j].t];
        }
        return 1;
    }
    return 0;
}
void dfs2(int s,ull h,int dep,int mx,int f)
{
    if(ct[f]>=n)return;
    if(dep)
    {
        ct[f]++;
        if(h==ans[f])
        {
            printf("%s\n",st);
            ct[f]=1926*817;
            return;
        } 
    }
    if(dep<mx)
    for(int i=0;i<26;i++)
    if(ch[f][s][i])
    st[dep]=i+'a',dfs2(ch[f][s][i],h*c+i+1,dep+1,mx,f),st[dep]=0;
}
int main()
{
    freopen("diff.in","r",stdin);
    freopen("diff.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    scanf("%s",v[i]+1),l[i]=strlen(v[i]+1);
    for(int i=1;i<=n;i++)
    {
        for(int j=0;j<26;j++)ch[i][l[i]][j]=0;
        for(int j=l[i]-1;j>-2;j--)
        {
            for(int k=0;k<26;k++)ch[i][j][k]=ch[i][j+1][k];
            ch[i][j][v[i][j+1]-'a']=j+1;
        }
    }
    int lb=1,rb=300,as=-1;
    while(lb<=rb)
    {
        int mid=(lb+rb)>>1;
        if(check(mid))as=mid,rb=mid-1;
        else lb=mid+1;
    }
    if(as==-1){printf("-1\n");return 0;}
    printf("%d\n",as);
    for(int i=1;i<=n;i++)ct[i]=0;
    for(int i=1;i<=n;i++)dfs2(0,0,0,as,i);
}
```



#### R8T1 河(river)

##### 题面

n条直线，每一条可以用y=kx+b表示，且起点都为y轴

你可以沿着河向x轴正方向走，求有多少个出发河流集合使得从这个集合中的河出发可以到达所有河，mod1e9+7

部分分：

n<=20

n<=1e5

##### 神仙题

暴力就直接枚举

考虑到了x接近inf的时候，河之间位置关系是斜率关系

先按b排序，然后处理出每条直线最后的相对位置

然后考虑每条直线可以覆盖哪些直线

发现覆盖一定是一个区间

左端点为它一开始左边直线的斜率最大值

右端点为它一开始右边直线的斜率最小值

然后就是一个区间覆盖，因为左右端点一定单调不降，所以只需要BIT维护区间和

##### 代码

```c++
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 555555
#define mod 1000000007
int n,f[N],g[N],g2[N],d[N],s[N],lb[N],rb[N],tr[N],f2[N];
bool cmp(int a,int b){return s[a]==s[b]?d[a]<d[b]:s[a]<s[b];}
bool cmp2(int a,int b){return d[a]<d[b];}
void ad(int x,int s){for(int i=x;i<=n+1;i+=i&-i)tr[i]=(tr[i]+s)%mod;}
int qu(int x){int tp=0;for(int i=x;i;i-=i&-i)tp=(tp+tr[i])%mod;return tp;}
int main()
{
    freopen("river.in","r",stdin);
    freopen("river.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d%d",&d[i],&s[i]),f[i]=f2[i]=i;
    sort(f+1,f+n+1,cmp);sort(f2+1,f2+n+1,cmp2);
    for(int i=1;i<=n;i++)g2[f2[i]]=i;
    for(int i=1;i<=n;i++)g[g2[f[i]]]=i;
    int mx=0;
    for(int i=1;i<=n;i++)
    mx=max(g[i],mx),rb[i]=mx;
    int mn=1e9;
    for(int i=n;i>=1;i--)
    mn=min(mn,g[i]),lb[i]=mn;
    ad(1,1);
    for(int i=1;i<=n;i++)
    ad(rb[i]+1,(qu(n+1)-qu(lb[i]-1)+mod)%mod);
    printf("%d\n",(qu(n+1)-qu(n)+mod)%mod);
}
```

#### R8T2 铁路(train)

##### 题面

有一颗n个点的树，有m条线路，所有线路上同时开出一辆火车，速度相等，求相遇火车对数

部分分

m<=2000

n,m<=1e5

##### 毒瘤题

暴力直接LCA

考虑如何计算对数

可以把路径拆成两段

计算上升与上升的贡献，上升与下降的贡献

可以证明这样不会算错

对于上升与上升路径，两条路径需要点相交并且起点dep相等

这一段可以线段树合并或者set启发式合并

对于后半段，两条路径一定只会相交一次

考虑树剖，分成log段，每一段内：

如果是上升，id[x]+dis为一个定值

如果是下降，id[x]-dis为一个定值

斜线的相交不好算

考虑旋转45度，然后变成(x-y,x+y)，然后变成横线和竖线的交

然后就可以扫描线加上树状数组

##### 代码

~~一开始写了个忘了启发的合并，直接MLE~~

```c++
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 100050
int sz[N],son[N],id[N],tid[N],tp[N],n,m,s[N][2],f[N][21],dep[N],head[N],cnt,ct,c1,c2,tr[N*6],a,b,fr[N];
long long ans;
multiset<int> tle[N],del[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void dfs1(int u,int fa)
{
    sz[u]=1;f[u][0]=fa;dep[u]=dep[fa]+1;
    for(int i=1;i<=20;i++)f[u][i]=f[f[u][i-1]][i-1];
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)
    dfs1(ed[i].t,u),son[u]=sz[son[u]]>sz[ed[i].t]?son[u]:ed[i].t,sz[u]+=sz[ed[i].t];
}
int LCA(int x,int y){if(dep[x]<dep[y])x^=y^=x^=y;for(int i=20;i>=0;i--)if(dep[x]-dep[y]>=1<<i)x=f[x][i];if(x==y)return x;for(int i=20;i>=0;i--)if(f[x][i]!=f[y][i])x=f[x][i],y=f[y][i];return f[x][0];}
int jmp(int x,int s){if(s<0)return -(ct++);for(int i=19;i>=0;i--)if(s&(1<<i))x=f[x][i];return x;}
void dfs2(int u,int v)
{
    id[u]=++ct;tid[ct]=u;tp[u]=v;
    if(son[u])dfs2(son[u],v);
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=son[u]&&ed[i].t!=f[u][0])dfs2(ed[i].t,ed[i].t);
}
void dfs3(int u,int fa)
{
    int st=0,as=0,tp;
    for(multiset<int>::iterator it=tle[fr[u]].begin();it!=tle[fr[u]].end();it++)
    tp=*it,as=tp==st?as+1:1,ans+=as-1,st=tp;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)
    {
        dfs3(ed[i].t,u);
        if(tle[fr[u]].size()<tle[fr[ed[i].t]].size())fr[u]^=fr[ed[i].t]^=fr[u]^=fr[ed[i].t];
        for(multiset<int>::iterator it=tle[fr[ed[i].t]].begin();it!=tle[fr[ed[i].t]].end();it++)
        ans+=tle[fr[u]].count(*it);
        for(multiset<int>::iterator it=tle[fr[ed[i].t]].begin();it!=tle[fr[ed[i].t]].end();it++)
        tle[fr[u]].insert(*it);
        tle[fr[ed[i].t]].clear();
    }
    for(multiset<int>::iterator it=del[u].begin();it!=del[u].end();it++)
    tle[fr[u]].erase(tle[fr[u]].find(*it));
}
void add(int x,int s){for(int i=x;i<=n*5;i+=i&-i)tr[i]+=s;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
struct ask{int c,x,s;}q[N*80];
struct qu{int c,l,r;}e[N*40];
bool cmp(ask a,ask b){return a.x<b.x;}
bool cmp2(qu a,qu b){return a.c<b.c;}
void adda(int c,int l,int r){q[++c1]=(ask){c+n,2*l-c+n-1,1};q[++c1]=(ask){c+n,2*r-c+n+1,-1};}
void addb(int c,int l,int r,int t){e[++c2]=(qu){c+n,2*l-c+n-t,2*r-c+n};}
void query(int x,int y)
{
    int s=0;
    while(tp[x]!=tp[y])
    {
        addb(id[x]+s,id[tp[x]],id[x],1);
        s+=dep[x]-dep[tp[x]]+1;
        x=f[tp[x]][0];
    }
    addb(id[x]+s,id[y],id[x],0);
}
void query2(int x,int y,int s)
{
    while(tp[x]!=tp[y])
    {
        adda(id[x]-s,id[tp[x]],id[x]);
        s-=dep[x]-dep[tp[x]]+1;
        x=f[tp[x]][0];
    }
    adda(id[x]-s,id[y],id[x]);
}
void solve()
{
    sort(q+1,q+c1+1,cmp);
    sort(e+1,e+c2+1,cmp2);
    int nt=1;
    for(int i=1;i<=c2;i++)
    {
        while(nt<=c1&&q[nt].x<=e[i].c)add(q[nt].c,q[nt].s),nt++;
        ans+=que(e[i].r)-que(e[i].l-1);
    }
}
int main()
{
    freopen("train.in","r",stdin);
    freopen("train.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)fr[i]=i;
    for(int i=1;i<n;i++)scanf("%d%d",&a,&b),adde(a,b);
    dfs1(1,0);dfs2(1,0);
    scanf("%d",&m);
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d",&a,&b);
        int l=LCA(a,b);
        tle[a].insert(dep[a]);
        del[l].insert(dep[a]);
        query(a,l);
        if(b!=LCA(a,b))
        {
            int tp=jmp(b,dep[b]-dep[l]-1);
            query2(b,tp,dep[a]+dep[b]-2*dep[l]);
        }
    }
    dfs3(1,0);
    solve();
    printf("%lld\n",ans);
}
```

#### R8T3 桥(bridge)

##### 题意

m条平行河道划分出m+1条平行线

有n个人，一开始在第i条线上的第j个位置，要去往第k条线上的第l个位置

现在要在河上建m座桥，每座桥连接i与i+1，求最小总距离和

部分分：

n<=30,m<=4

n,m<=200

nm<=1e7

n,m<=1e5

##### ~

暴力直接做

dp i j 表示第i层的桥建在j位置

考虑一层一层转移

第一层一定是凸函数

设当前函数为f(x)

转移下一层时：

设当前两座桥间经过了j个人

首先，f2(i)=min(f(k)+j*abs(k-i))

考虑这个东西是什么

前面是个凸函数

后面那个也是

所以加起来是

根据一些性质，函数极值点斜率为0

所以就要找到极值点

后面那个相当于前面一段斜率-j,后面那段+j

找到原函数上斜率为-j,j的点为x1,x2

如果x<x1，最终极值点会在x1上，f2(x)=f(x1)+j*(x1-x)

如果x1<=x<=x2，则极值点会为x,f2(x)=f(x)

如果x>x2，最终极值点会在x2上，f2(x)=f(x2)+j*(x-x2)

如果维护斜率，相当于两次区间赋值

考虑后半段

要加上从河流两边出发经过这座桥或者经过这里到达两边的人对这座桥的贡献

也就是一堆abs(k-x)

直接加上去，维护区间加

然后拿个set维护所有k

然后就￥%￥……%%￥#@￥%……&O*&^%$#@#%&*O(O&^%$EW#R#JI

##### 代码

调一天

```c++
#include<cstdio>
#include<set>
#include<algorithm> 
using namespace std;
#define N 200055
#pragma GCC optimize("-Ofast")
set<int> s1;
multiset<int> tle[N];
int st[N],n,m,a,b,c,d,ct,tr[N];
long long k,y,as;
inline int Max(int a,int b){return a>b?a:b;}
void add(int x,int a){for(int i=x;i<=m+1;i+=i&-i)tr[i]+=a;}
int que(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
struct segt{long long l,r,lb,rb,l1,l2,mx;long long sum;}e[N*4];
void pushup(int x){if(x>ct*2)return;e[x].sum=e[x<<1].sum+e[x<<1|1].sum;e[x].mx=Max(e[x<<1].mx,e[x<<1|1].mx);}
void pushdown(int x){if(x>ct*2)return;if(e[x].l2!=-233333333)e[x<<1].mx=e[x<<1|1].mx=e[x<<1].l2=e[x<<1|1].l2=e[x].l2,e[x<<1].sum=1ll*(e[x<<1].rb-e[x<<1].lb+1)*e[x].l2,e[x<<1|1].sum=1ll*(e[x<<1|1].rb-e[x<<1|1].lb+1)*e[x].l2,e[x].l2=-233333333,e[x<<1].l1=e[x<<1|1].l1=0;
if(e[x].l1)e[x<<1].mx+=e[x].l1,e[x<<1|1].mx+=e[x].l1,e[x<<1].l1+=e[x].l1,e[x<<1|1].l1+=e[x].l1,e[x<<1].sum+=1ll*(e[x<<1].rb-e[x<<1].lb+1)*e[x].l1,e[x<<1|1].sum+=1ll*(e[x<<1|1].rb-e[x<<1|1].lb+1)*e[x].l1,e[x].l1=0;} 
void build(int x,int l,int r)
{
    e[x].l=l;e[x].r=r;e[x].l2=-233333333;
    e[x].lb=st[l],e[x].rb=st[r+1]-1;
    if(l==r)return;
    int mid=(l+r)>>1;
    build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void add(int x,int l,int r,int s)
{
    pushdown(x);
    if(l>r)return;
    if(e[x].l==l&&e[x].r==r)
    {
        e[x].l1+=s,e[x].sum+=1ll*(e[x].rb-e[x].lb+1)*s;
        e[x].mx+=s;
        return;
    }
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)add(x<<1,l,r,s);
    else if(mid<l)add(x<<1|1,l,r,s);
    else add(x<<1,l,mid,s),add(x<<1|1,mid+1,r,s);
    pushup(x);
}
void modify(int x,int l,int r,int s)
{
    if(l>r)return;
    pushdown(x);
    if(e[x].l==l&&e[x].r==r)
    {
        e[x].l2=s,e[x].sum=1ll*(e[x].rb-e[x].lb+1)*s;
        e[x].mx=s;e[x].l1=0;
        return;
    }
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)modify(x<<1,l,r,s);
    else if(mid<l)modify(x<<1|1,l,r,s);
    else modify(x<<1,l,mid,s),modify(x<<1|1,mid+1,r,s);
    pushup(x);
}
pair<int,long long> query(int x,int s)
{
    pushdown(x);
    if(e[x].l==e[x].r)return make_pair(e[x].r,0);
    pair<int,long long> tmp;
    if(e[x<<1].mx>=s)return query(x<<1,s);
    else {tmp=query(x<<1|1,s);return make_pair(tmp.first,e[x<<1].sum+tmp.second);}
}
int read(){char c=getchar();while(c<'0'||c>'9')c=getchar();int ans=0;while(c>='0'&&c<='9')ans=ans*10+c-'0',c=getchar();return ans;}
int main()
{
    freopen("bridge.in","r",stdin);
    freopen("bridge.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)
    {
        a=read();b=read();c=read();d=read();
        if(a>c)a^=c^=a^=c,b^=d^=b^=d;
        if(a==c){as+=b>d?b-d:d-b;continue;}
        add(a+1,1);add(c,-1);
        s1.insert(b);s1.insert(d);tle[a].insert(b);tle[c-1].insert(d);
    }
    for(set<int>::iterator it=s1.begin();it!=s1.end();it++)
    st[++ct]=*it;
    build(1,1,ct-1);
    for(int i=m;i>=1;i--)
    {
        int tp3=que(i+1);
        pair<int,long long> tp1=query(1,-tp3),tp2=query(1,tp3);
        long long fx1=tp1.second+k,fx2=tp2.second+k;
        k=1ll*(st[tp1.first]-st[1])*tp3+fx1;
        modify(1,1,tp1.first-1,-tp3);
        modify(1,tp2.first,ct-1,tp3);
        for(multiset<int>::iterator it=tle[i].begin();it!=tle[i].end();it++)
        k+=(*it-st[1]),add(1,1,lower_bound(st+1,st+ct+1,*it)-st-1,-1),add(1,lower_bound(st+1,st+ct+1,*it)-st,ct-1,1);
    }
    pair<int,long long> tp4=query(1,0);
    printf("%lld\n",tp4.second+k+as);
}
```

### R9

#### R9T2 游戏(game)

随机生成n个点的有根树，从根开始，两人轮流向下走，不能走的人输，求先手赢的概率 mod998244353

部分分：

n<=10

n<=18

n<=3000

n<=1e5

多组询问

##### ？？？

暴搜过18（逃

考虑SG值做dp可以过200-3000，打表







设一种方案中有x棵子树，大小是s1,s2,s3...sx

考虑建EGF，则转移是将后手胜的EGF乘x次

但是一种方案会被计算x!/(\prod sumi!)次，sumi是sx=i的次数

考虑EGF乘起来时的那个多重组合

它会正好多算(\prod  sumi!)次

所以最后除以x!就行了

最后算上根

所以是g(x)/x=f(x)+f(x)^2/2!+f(x) ^3/3!+...=exp f(x)

又因为不是赢就是输

所以f(x)+g(x)=x+x^2/2+x ^3/3+...=-ln(1-x)

答案是g(x)第i项系数乘i

经过解(chao)方(biao)程(cheng)可以得到g(x)=ln(1-ln(1-x))

多项式ln

```c++
#include<cstdio>
using namespace std;
#define N 524500
#define mod 998244353
int fft[N],s[N],s1[N],s2[N],as[N],v[N],n,rev[N],v2[N],tp[N],ans[N],q;
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)rev[i]=rev[i/2]/2+(i&1)*s/2;
    for(int i=0;i<s;i++)
    fft[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(3,(mod-1)/l);
        if(t-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        {
            int st=1;
            for(int k=j;k<j+(l>>1);k++,st=1ll*st*s1%mod)
            {
                int v1=fft[k],v2=1ll*fft[k+(l>>1)]*st%mod;
                fft[k]=(v1+v2)%mod;
                fft[k+(l>>1)]=(v1-v2+mod)%mod;
            }
        }
    }
    int inv=pw(s,mod-2);
    if(t-1)for(int i=0;i<s;i++)fft[i]=1ll*fft[i]*inv%mod;
    for(int i=0;i<s;i++)a[i]=fft[i];
}
void getinv(int x)
{
    if(x==1){as[0]=pw(v[0],mod-2);return;}
    getinv((x+1)/2);
    int e=1;while(e<=x*3)e<<=1;
    for(int i=0;i<e;i++)s[i]=v2[i]=0;
    for(int i=0;i<x;i++)s[i]=as[i],v2[i]=v[i];
    for(int i=0;i<e;i++)rev[i]=rev[i/2]/2+(i&1)*e/2;
    dft(e,s,1);dft(e,v2,1);
    for(int i=0;i<e;i++)s1[i]=1ll*s[i]*(2-1ll*s[i]%mod*v2[i]%mod)%mod;
    dft(e,s1,-1);
    for(int i=0;i<x;i++)
    as[i]=(s1[i]+mod)%mod;
}
int main()
{
    freopen("game.in","r",stdin);
    freopen("game.out","w",stdout);
    n=100005;
    for(int i=0;i<n;i++)v[i]=i==0?1:pw(i,mod-2);
    for(int i=0;i<n;i++)i?tp[i-1]=1ll*v[i]*i%mod:1;
    getinv(n);
    int s=1;while(s<=n*2)s*=2;
    dft(s,tp,1);dft(s,as,1);for(int i=0;i<s;i++)as[i]=1ll*as[i]*tp[i]%mod;dft(s,as,-1);
    scanf("%d",&q);
    while(q--)
    scanf("%d",&n),printf("%d\n",(mod+1-as[n-1])%mod);
}
```



#### R9T3 膜法阵(magic)

##### 题面

给出序列A,B

$C_i=max(C_{i-1},A_i)$

多次询问，每次

1.增加一个A值

2.修改一个B值

然后求$\prod min(B_i,C_i)$

部分分：

n<=1000

n<=8e4

n<=1e5

##### 分析

本场最简单的题

一看前面两道看不懂题就直接写T3

考虑暴力分块

每次A增加相当于C区间覆盖

块内对B排个序

整块覆盖可以排好序lower_bound，前面的min是B，后面是C，处理B的前缀积，C的部分快速幂，一个log

边界直接推标记暴力没有log

B重构暴力排序是nsqrtnlogn的，3s能过，但有可能被卡成30-80

然而重构可以插入一次删除一次就没有log了，再改大小可以nsqrt(nlogn)800ms

还有更优秀的做法

考虑区间查小于它的数的乘积

可以树套树

nlog^2n

然而常数大代码长

所以我选择写分块

##### 代码

```c++
#include<cstdio>
#include<set>
#include<algorithm>
using namespace std;
#define N 100055
#define mod 1000000007
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int sz=300,lb[N],rb[N],bl[N],lz[N],s[N],b[N],sortb[N],lmul[N],ans[N],n,q,a,d,c,tr[N];
void rebuildb(int x,int f,int t)
{
    int s1=lower_bound(sortb+lb[x],sortb+rb[x]+1,f)-sortb;
    for(int i=s1;i<rb[x];i++)sortb[i]=sortb[i+1];
    int s2=lower_bound(sortb+lb[x],sortb+rb[x],t)-sortb;
    for(int i=rb[x];i>s2;i--)sortb[i]=sortb[i-1];
    sortb[s2]=t;
    lmul[lb[x]]=sortb[lb[x]];
    for(int i=lb[x]+1;i<=rb[x];i++)lmul[i]=1ll*lmul[i-1]*sortb[i]%mod;
}
void buildb(int x)
{
    for(int i=lb[x];i<=rb[x];i++)
    sortb[i]=b[i];
    sort(sortb+lb[x],sortb+rb[x]+1);
    lmul[lb[x]]=sortb[lb[x]];
    for(int i=lb[x]+1;i<=rb[x];i++)lmul[i]=1ll*lmul[i-1]*sortb[i]%mod;
}
void calans(int x)
{
    if(lz[x])
    {
        int s1=lower_bound(sortb+lb[x],sortb+rb[x]+1,lz[x])-sortb-1;
        ans[x]=1ll*(s1<lb[x]?1:lmul[s1])*pw(lz[x],rb[x]-s1)%mod;
    }
    else
    {
        ans[x]=1;
        for(int i=lb[x];i<=rb[x];i++)
        ans[x]=1ll*ans[x]*(b[i]>s[i]?s[i]:b[i])%mod;
    }
}
void modify(int l,int r,int v)
{
    if(bl[r]-bl[l]<1)
    {
        if(lz[bl[l]])
        {
            for(int i=lb[bl[l]];i<=rb[bl[l]];i++)
            s[i]=lz[bl[l]];
            lz[bl[l]]=0;
        }
        for(int i=l;i<=r;i++)s[i]=v;
        calans(bl[l]);return;
    }
    if(lz[bl[l]])
    {
        for(int i=lb[bl[l]];i<=rb[bl[l]];i++)
        s[i]=lz[bl[l]];
        lz[bl[l]]=0;
    }
    for(int i=l;i<=rb[bl[l]];i++)s[i]=v;
    calans(bl[l]);
    if(lz[bl[r]])
    {
        for(int i=lb[bl[r]];i<=rb[bl[r]];i++)
        s[i]=lz[bl[r]];
        lz[bl[r]]=0;
    }
    for(int i=lb[bl[r]];i<=r;i++)s[i]=v;
    calans(bl[r]);
    for(int i=bl[l]+1;i<bl[r];i++)
    lz[i]=v,calans(i);
}
set<int> tmp;
int getas()
{
    int s=1;for(int i=1;i<=bl[n];i++)s=1ll*s*ans[i]%mod;return s;
}
void ad(int x,int s){for(int i=x;i<=n;i+=i&-i)tr[i]=tr[i]>s?tr[i]:s;}
int qu(int x){int as=0;for(int i=x;i;i-=i&-i)as=as>tr[i]?as:tr[i];return as;}
int main()
{
    freopen("magic.in","r",stdin);
    freopen("magic.out","w",stdout);
    scanf("%d%d",&n,&q);
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&s[i]);ad(i,s[i]);
        if(s[i]>s[i-1])tmp.insert(i);
        else s[i]=s[i-1];
    }
    for(int i=1;i<=n;i++)
    scanf("%d",&b[i]);
    for(int i=1;i<=n;i++)bl[i]=(i-1)/sz+1;
    for(int i=1;i<=n;i++)lb[i]=(i-1)*sz+1,rb[i]=i*sz>n?n:i*sz;
    for(int i=1;i<=bl[n];i++)buildb(i),calans(i);
    while(q--)
    {
        scanf("%d%d%d",&a,&d,&c);
        if(a==0)
        {
            int rb=-1;
            if(c>qu(d))
            while(1)
            {
                set<int>::iterator tp=tmp.lower_bound(d);
                if(tp==tmp.end()){rb=n;break;}
                int st=*tp;
                if(c<s[st]){rb=st-1;break;}
                tmp.erase(tp);
            }
            if(rb==-1){printf("%d\n",getas());continue;}
            tmp.insert(d);
            modify(d,rb,c);s[d]=c;ad(d,c);
            printf("%d\n",getas());
        }
        else
        rebuildb(bl[d],b[d],c),b[d]=c,calans(bl[d]),printf("%d\n",getas());
    }
}
```





### Part 2 JZ Online

#### 4.19 T2 炮塔(tower)

##### 题面

![](/pic\13.png)

![](/pic\14.png)

##### 分析

如果出现连续的 ### 或者 ##. ,那么无论如何都走不过去 

否则,如果当前手上至少有两个,那么

1.对于 .#.,可以放下一个,向右两步,放下另一个,向左两步,拿起第一个,再向右两步,拿起第二个

2.对于.##*,可以放下,向右三步,拿起右边那个,变为 *##.

这样的话,只要不出现无论如何都不能走的情况,都一定可以走,并且可以拿到所有的*(每一个.##.需要留下一个 *)

如果当前手上只有一个

在遇到情况2时,可以照常

在遇到情况1时,会减少1个,并使其变为*#.

在手上有一个时,可以反向进行情况2,因此可以忽略情况2

如果手上有一个,且前面有一个*#.,那么可以回去拿到变为2个

如果手上当前没有,可以证明这种情况如果可以向回走拿到两个,它一定在前面某一次手上有一个的时候被考虑了,可以不用考虑

因此,可以从1开始向后扫,模拟上面的情况,同时记录*#.的个数和手上的个数,当手上个数>1时进行贪心

复杂度$O(n)$

##### 代码

特别短

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 1000040
char s[N],s2[N];
int T,l,nw,as,mx,bf;//as指手上个数,bf指前面的*#.个数
bool isok(int x){return !(s[x-1]!='*'&&s[x+1]!='*'&&s[x]=='#');}
bool sth()
{
    int ans=0;
    for(int i=1;i<=l;i++)
    {
        if(s2[i]=='*')ans++;
        if(s2[i]=='#'&&s2[i+1]=='#'&&s2[i+2]!='*')break;
        if(s2[i]=='#'&&s2[i+1]=='#')ans--;
        if(mx<ans)mx=ans;
    }
}
void dfs(int x)
{
    if(as>=2){sth();return;}
    if(bf+as>=2&&as>=1){sth();return;}
    if(as==1)mx=1;
    if(x>=l-2)return;
    //try x->x+1
    if(s[x+1]=='.')dfs(x+1);
    else if(s[x+1]=='*')s[x+1]='.',as=as+1,dfs(x+1);
    else
    {
        if(isok(x+1))bf=0,dfs(x+1);
        else if(isok(x+2))
        {
            if(as==0)return;
            as--;s[x]='*';bf++;
            dfs(x+1);
        }
        else return;
    }
}
int main()
{
    freopen("tower.in","r",stdin);
    freopen("tower.out","w",stdout);
    scanf("%d",&T);
    while(T--)
    {
        scanf("%s",s+1);
        l=strlen(s+1);
        for(int i=1;i<=5;i++)
        s[++l]='.';
        for(int i=1;i<=l;i++)s2[i]=s[i];
        nw=1,as=0,mx=0,bf=0;dfs(1);printf("%d\n",mx);
    }
}
```

#### 4.24 T1 密文(secret)

##### 题目

![](/pic\15.png)

![](/pic\16.png)

##### 分析

一次询问$[l,r]$相当于知道了$l-1,r$的异或前缀和的异或,相当于知道两者中任意一个就知道了另一个

那么相当于是要让0-n的所有点联通,选择$(l,r)$的代价为$a_{l+1} xor ... xor a_r$

记前缀异或为$s_i $,相当于代价为$s_r xor s_l$

那么就是异或最小生成树

$O(nlog^2n)$

##### 代码

这里用的是trie

```cpp
#include<cstdio>
#define N 3500050
int ch[N][2],sz[N],n,t,a,ct=1,as;
long long ans;
void insert(int x)
{
    int d=31,s=1;
    while((--d)>=0)
    {
        sz[s]++;
        bool t=x&(1<<d);
        if(!ch[s][t])ch[s][t]=++ct;
        s=ch[s][t];
    }
    sz[s]++;
}
int dfs2(int x,int d,int s)
{
    if(d==-1)return s;
    bool tp=(s&(1<<d));
    if(!ch[x][tp])
    {
        s|=1<<d;
        return dfs2(ch[x][!tp],d-1,s);
    }
    if(s&(1<<d))s^=1<<d;
    return dfs2(ch[x][tp],d-1,s);
}
void dfs1(int x,int d,int k,int s,int d2)
{
    if(d==-1)
    {
        int a=dfs2(k,d2,s);
        if(a<as)as=a;
        return;
    }
    if(ch[x][0])dfs1(ch[x][0],d-1,k,s,d2);
    if(ch[x][1])dfs1(ch[x][1],d-1,k,s+(1<<d),d2);
}
void dfs(int x,int d)
{
    if(!ch[x][1]&&!ch[x][0])return;
    if(!ch[x][1])dfs(ch[x][0],d-1);
    else if(!ch[x][0])dfs(ch[x][1],d-1);
    else
    {
        ans+=1<<d;
        if(sz[ch[x][0]]>sz[ch[x][1]])
        {
            as=1e9;
            dfs1(ch[x][1],d-1,ch[x][0],0,d-1);
            ans+=as;
        }
        else
        {
            as=1e9;
            dfs1(ch[x][0],d-1,ch[x][1],0,d-1);
            ans+=as;
        }
        dfs(ch[x][0],d-1);dfs(ch[x][1],d-1);
    }
}
int main()
{
    freopen("secret.in","r",stdin);
    freopen("secret.out","w",stdout);
    insert(0);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    scanf("%d",&a),t^=a,insert(t);
    dfs(1,30);printf("%lld\n",ans);
}
```

#### 4.24 T2 最短路(min)

![](/pic\17.png)

n<=100,m<=10000

##### 分析

官方题解:

![](/pic\18.png)

另一种做法：线性规划(我不能证明,但也找不到反例)

##### 代码

(线性规划)

```cpp
#include<cstdio>
#include<cmath>
using namespace std;
#define N 307
#define M 20000
int ct,n,m,k,a,b;
double f[M],s[M][N];
#define eps 1e-10
void pivot(int x,int y,int n,int m)
{
    double tp=s[y][x];s[y][x]=0;
    for(int i=1;i<=n+1;i++)s[y][i]/=-tp;
    s[y][x]=1/tp;
    for(int i=1;i<=m;i++)
    if(i!=y)
    {
        double s1=s[i][x];s[i][x]=0;
        if(s1<=eps&&s1>=-eps)continue;
        for(int j=1;j<=n+1;j++)
        s[i][j]+=s1*s[y][j];
    }
    double s1=f[x];f[x]=0;
    for(int j=1;j<=n+1;j++)f[j]+=s1*s[y][j];
}
double LP(int n,int m)
{
//	f[0]=-1;
    while(1)
    {
        ct++;
        int tp=0;
        for(int i=1;i<=n;i++)
        if(f[i]>f[tp])tp=i;
        if(tp==0)return f[n+1];
        int as=0;double t=1e9;
        for(int i=1;i<=m;i++)
        {
            if(s[i][tp]>=-eps)continue;
            double tmp=s[i][n+1]/-s[i][tp];
            if(tmp<t)t=tmp,as=i;
        }
        if(as==0)
        return 1e100;
        pivot(tp,as,n,m);
    }
}
void adde(int f,int t)
{
    s[++ct][f]=1;s[ct][t]=-1;s[ct][n*3+1]=1;
}
void adde2(int f,int t)
{
    s[++ct][f]=1;s[ct][t]=-1;s[ct][n*2+t]=1;
}
int main()
{
    freopen("min.in","r",stdin);
    freopen("min.out","w",stdout);
    scanf("%d%d%d",&n,&m,&k);
    for(int i=1;i<=n;i++)s[ct=1][n*2+i]=-1;s[1][n*3+1]=k;
    s[ct=2][n]=1;s[ct=3][n]=-1;
    for(int i=1;i<=n;i++)adde2(i+n,i);
    while(m--)scanf("%d%d",&a,&b),a=a+1,b=b+1,adde(a,b+n),adde(b,a+n);
    for(int i=1;i<=n;i++)
    {
        s[++ct][n*2+i]=-1;
        s[ct][n*3+1]=((i==1||i==n)?0:1);
    }
    f[1]=1;
    printf("%.0lf\n",floor(LP(n*3,ct)+eps));
}
```

#### 4.24 T3 特技飞行(aerobatics)

##### 题目

![](/pic\19.png)

![](/pic\20.png)

![](/pic\21.png)

![](/pic\22.png)

![](/pic\23.png)

##### 分析

首先处理c的分数，因为它和转向方式没有关系

第一个问题是处理出所有交点,将直线按st处位置插入,每次增加的交点个数即为逆序对数,可以用set维护

然后将坐标旋转45度,离散化一下,用BIT+扫描线可以得到每一个点有没有被观测到

然后是a和b的问题

显然全部选择a是合法的

只要求出尽量多的b的方式即可

可以发现这东西可以看成是一个置换,因此最少a次数为n-置换环个数

事实上,这个下界是合法的

![1564459692903](/pic\23.png)

复杂度$O(mlogm)$,m为交点数

##### 代码

场上写的,巨丑

```cpp
#include<cstdio>
#include<algorithm>
#include<set>
using namespace std;
#define N 200050
#define M 2000050
int n,a,b,c,x,xx,k,s[N],t[N],ls[N],lt[N],r[N][3],s1[N],t1[N],vis[N],head[N],cnt,ct,ct2,tr[M],a1,b1,c1;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
multiset<pair<int,int> > tp1,tp2;
struct pt{double x,y;}q[M];
struct qu{double x,y;int p,as,xx,yy;}st[M];
double lh[M],lh2[M];
void dfs(int u)
{
    vis[u]=1;
    for(int i=head[u];i;i=ed[i].next)
    if(!vis[ed[i].t])dfs(ed[i].t);
}
void ad(int x,int k){for(int i=x;i;i-=i&-i)tr[i]+=k;}
int que(int x){int ans=0;for(int i=x;i<=ct2;i+=i&-i)ans+=tr[i];return ans;}
bool cmp2(qu a,qu b){if(a.xx==b.xx)return a.p>b.p;return a.xx>b.xx;}
int main()
{
    freopen("aerobatics.in","r",stdin);
    freopen("aerobatics.out","w",stdout);
    scanf("%d%d%d%d%d%d",&n,&a1,&b1,&c1,&x,&xx);
    for(int i=1;i<=n;i++)scanf("%d",&s[i]),s1[i]=s[i];
    for(int i=1;i<=n;i++)scanf("%d",&t[i]),t1[i]=t[i],tp1.insert(make_pair(s[i],t[i]));
    sort(s1+1,s1+n+1);
    sort(t1+1,t1+n+1);
    for(int i=1;i<=n;i++)ls[i]=lower_bound(s1+1,s1+n+1,s[i])-s1,lt[i]=lower_bound(t1+1,t1+n+1,t[i])-t1,adde(ls[i],lt[i]);
    for(multiset<pair<int,int> >::iterator it=tp1.begin();it!=tp1.end();it++)
    {
        pair<int,int> s2=*it;
        swap(s2.first,s2.second);
        for(multiset<pair<int,int> >::reverse_iterator it2=tp2.rbegin();it2!=tp2.rend();it2++)
        {
            pair<int,int> s3=*it2;
            if(s2>s3)break;
            double fx=x+1.0*(xx-x)/(s3.second-s2.second+s2.first-s3.first)*(-s2.second+s3.second);
            double fy=s2.second+1.0*(-s2.second+s2.first)*(fx-x)/(xx-x);
            q[++ct]=(pt){fx+fy,fx-fy};
        }
        tp2.insert(s2);
    }
    int mxd=ct,mnd=n;
    for(int i=1;i<=n;i++)if(!vis[i])mnd--,dfs(i);
    long long as1=mxd*a1,as2=mnd*a1+(mxd-mnd)*b1;
    if(as1>as2)as1^=as2^=as1^=as2;
    for(int i=1;i<=ct;i++)st[++ct2]=(qu){q[i].x,q[i].y,1,0};
    scanf("%d",&k);
    while(k--)
    {
        scanf("%d%d%d",&a,&b,&c);
        st[++ct2]=(qu){(double)a+b+c,(double)a-b+c,2,1};
        st[++ct2]=(qu){(double)a+b-c-1e-6,(double)a-b+c,2,-1};
        st[++ct2]=(qu){(double)a+b+c,(double)a-b-c-1e-6,2,-1};
        st[++ct2]=(qu){(double)a+b-c-1e-6,(double)a-b-c-1e-6,2,1};
    }
    for(int i=1;i<=ct2;i++)lh[i]=st[i].x,lh2[i]=st[i].y;
    sort(lh+1,lh+ct2+1);sort(lh2+1,lh2+ct2+1);
    for(int i=1;i<=ct2;i++)st[i].xx=lower_bound(lh+1,lh+ct2+1,st[i].x-1e-7)-lh,st[i].yy=lower_bound(lh2+1,lh2+ct2+1,st[i].y-1e-7)-lh2;
    sort(st+1,st+ct2+1,cmp2);
    for(int i=1;i<=ct2;i++)
    {
        if(st[i].p==2)
        ad(st[i].yy,st[i].as);
        else
        {
            int as=que(st[i].yy);
            if(as>0)
            as1+=c1,as2+=c1;
        }
    }
    printf("%lld %lld\n",as1,as2);
}
```

#### 4.24 T4 吃(eat)

##### 题目

![](/pic\26.png)

$m=n-1$ 或者$m=n$

n<=1e5

##### 分析

首先考虑一棵树

对于i和j,如果在删掉i时i与j联通,则会有1的贡献,而这种情况出现的概率为1/(i,j路径间点数),代表i为路径上第一个被删除的点

点分治+fft统计路径条数,复杂度$O(nlog^2n) $

对于基环树的情况,可以先处理掉没有经过环的部分

考虑怎么计算经过环的部分

设路径上必经点数为x,环两边点数为a,b

概率为一个点是$x+a$个中最先的或$x+b$个中最先的

容斥可得答案为$1/(x+a)+1/(x+b)-1/(x+a+b)$

对于前两种,将环转为链,对没有跨过分割点的和跨过的分别做分治+fft,复杂度$O(nlog^2n)$

对于第三种,可以直接统计

$O(nlog^2n)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<stack>
using namespace std;
#define N 333000
#define M 2220000
#define mod 998244353
stack<int> st;
int head[N],cnt,ins[N],is[N],c[N],id[N],le,vis[N],ntt[N],s1[N],s2[N],s3[N],s4[N],s5[N],sth[M*5],lb[N],len[N],rev[N],n,m,a,b,ds[N],mx1,vl,as,sz[N],mx2,ct2;
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+(i&1)*s/2;
    for(int i=0;i<s;i++)ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(3,(mod-1)/l);
        if(t==-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        for(int k=j,st=1;k<j+(l>>1);k++,st=1ll*st*s1%mod)
        {
            int a1=ntt[k],a2=1ll*ntt[k+(l>>1)]*st%mod;
            ntt[k]=a1+a2-(a1+a2>=mod?mod:0);
            ntt[k+(l>>1)]=a1-a2+(a1<a2?mod:0);
        }
    }
    int inv=pw(s,t==-1?mod-2:0);
    for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void dfs0(int u,int fa)
{
    ins[u]=1;st.push(u);
    for(int i=head[u];i&&!le;i=ed[i].next)
    if(ed[i].t!=fa)
    {
        if(ins[ed[i].t])
        {
            while(1)
            {
                int a=st.top();st.pop();
                c[++le]=a;
                if(a==ed[i].t)break;
            }
        }
        else
        dfs0(ed[i].t,u);
    }
    if(!st.empty()&&st.top()==u)st.pop();
}
void dfs1(int u,int fa)
{
    sz[u]=1;
    int mx=0;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&!vis[ed[i].t])
    dfs1(ed[i].t,u),mx=max(mx,sz[ed[i].t]),sz[u]+=sz[ed[i].t];
    mx=max(mx,vl-sz[u]);
    if(mx<mx1)mx1=mx,as=u;
}
void dfs2(int u,int fa)
{
    ds[u]=ds[fa]+1;
    s2[ds[u]]++;
    if(mx2<ds[u])mx2=ds[u];
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&!vis[ed[i].t])
    dfs2(ed[i].t,u);
}
void work(int u)
{
    int mx=0;ds[u]=0;
    for(int i=head[u];i;i=ed[i].next)
    if(!vis[ed[i].t])
    {
        mx2=0;
        dfs2(ed[i].t,u);
        for(int i=1;i<=mx2;i++)s1[i]+=s2[i],s3[i]=s2[i],s2[i]=0;
        int l=1;while(l<=mx2*2)l<<=1;
        for(int i=mx2+1;i<l;i++)s3[i]=0;
        dft(l,s3,1);for(int i=0;i<l;i++)s3[i]=1ll*s3[i]*s3[i]%mod;dft(l,s3,-1);
        for(int i=0;i<l;i++)s4[i]=(s4[i]-s3[i]+mod)%mod;
        if(mx<mx2)mx=mx2;
    }
    int l=1;while(l<=mx*2)l<<=1;
    for(int i=mx+1;i<l;i++)s1[i]=0;
    dft(l,s1,1);for(int i=0;i<l;i++)s1[i]=1ll*s1[i]*(s1[i]+2)%mod;dft(l,s1,-1);
    for(int i=0;i<l;i++)s4[i]=(s4[i]+s1[i])%mod,s1[i]=0;
    return;
}
void dfs3(int u)
{
    vis[u]=1;work(u);
    for(int i=head[u];i;i=ed[i].next)
    if(!vis[ed[i].t])
    {
        vl=sz[ed[i].t],mx1=1e9;
        dfs1(ed[i].t,u);
        dfs3(as);
    }
}
void fz1(int l,int r)
{
    if(l>=r)return;
    int mid=(l+r)>>1;
    int m1=0,m2=0;
    for(int i=l;i<=mid;i++)m1=max(len[i]+mid-i,m1);
    for(int i=mid+1;i<=r;i++)m2=max(len[i]+i-mid-1,m2);
    for(int i=0;i<=m1;i++)s1[i]=0;
    for(int i=0;i<=m2;i++)s2[i]=0;
    for(int i=l;i<=mid;i++)
    for(int j=0;j<=len[i];j++)s1[mid-i+j]+=sth[j+lb[i]];
    for(int i=mid+1;i<=r;i++)
    for(int j=0;j<=len[i];j++)s2[i-mid-1+j]+=sth[j+lb[i]];
    int s=1;while(s<=m1+m2)s<<=1;
    for(int i=m1+1;i<s;i++)s1[i]=0;
    for(int i=m2+1;i<s;i++)s2[i]=0;
    dft(s,s1,1);dft(s,s2,1);for(int i=0;i<s;i++)s1[i]=2ll*s1[i]*s2[i]%mod;dft(s,s1,-1);
    for(int i=0;i<s;i++)s4[i+1]=(s4[i+1]+s1[i])%mod;
    fz1(l,mid);fz1(mid+1,r);
}
void fz2(int l,int r)
{
    if(l>=r)return;
    int mid=(l+r)>>1;
    int m1=0,m2=0;
    for(int i=l;i<=mid;i++)m1=max(len[i]+i-l,m1);
    for(int i=mid+1;i<=r;i++)m2=max(len[i]+r-i,m2);
    for(int i=0;i<=m1;i++)s1[i]=0;
    for(int i=0;i<=m2;i++)s2[i]=0;
    for(int i=l;i<=mid;i++)
    for(int j=0;j<=len[i];j++)s1[i-l+j]+=sth[j+lb[i]];
    for(int i=mid+1;i<=r;i++)
    for(int j=0;j<=len[i];j++)s2[r-i+j]+=sth[j+lb[i]];
    int s=1;while(s<=m1+m2)s<<=1;
    for(int i=m1+1;i<s;i++)s1[i]=0;
    for(int i=m2+1;i<s;i++)s2[i]=0;
    dft(s,s1,1);dft(s,s2,1);for(int i=0;i<s;i++)s1[i]=2ll*s1[i]*s2[i]%mod;dft(s,s1,-1);
    for(int i=0;i<s;i++)s4[i+le-r+l]=(s4[i+le-r+l]+s1[i])%mod;
    fz2(l,mid);fz2(mid+1,r);
}
int main()
{
    freopen("eat.in","r",stdin);
    freopen("eat.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b);
    if(m==n-1)
    {
        dfs3(1);
        int ans=0;
        for(int i=1;i<=n;i++)
        ans=(ans+1ll*pw(i+1,mod-2)*s4[i])%mod;
        printf("%d\n",(ans+n)%mod);
    }
    else
    {
        dfs0(1,0);
        for(int i=1;i<=le;i++)
        {
            vis[c[i==1?le:i-1]]=1,vis[c[i==le?1:i+1]]=1;vis[c[i]]=0;
            ds[0]=-1;mx2=0;
            dfs2(c[i],0);
            lb[i]=++ct2;
            len[i]=mx2;
            ct2+=mx2;
            for(int j=0;j<=mx2;j++)sth[j+lb[i]]=s2[j],s5[j]+=s2[j],s3[j]=s2[j],s2[j]=0;
            int l=1;while(l<=mx2*2)l<<=1;
            for(int j=mx2+1;j<l;j++)s3[j]=0;
            dft(l,s3,1);for(int j=0;j<l;j++)s3[j]=1ll*s3[j]*s3[j]%mod;dft(l,s3,-1);
            for(int j=0;j<l;j++)s4[j+le-1]=(s4[j+le-1]+s3[j])%mod,s3[j]=0;
            dfs3(c[i]);
        }
        fz1(1,le);fz2(1,le);
        int l=1;while(l<=n*2)l<<=1;
        dft(l,s5,1);for(int i=0;i<l;i++)s5[i]=1ll*s5[i]*s5[i]%mod;dft(l,s5,-1);
        for(int j=0;j<l;j++)s4[j+le-1]=(s4[j+le-1]-s5[j]+mod)%mod;
        int ans=0;
        for(int i=1;i<=n;i++)
        ans=(ans+1ll*pw(i+1,mod-2)*s4[i])%mod;
        printf("%d\n",(ans+n)%mod);
    }
}
```

#### 5.6 T1 麻雀(sparrow)

##### 题目

![](/pic/27.png)

##### 分析

因为所有数非负,所以大区间一定优于小区间

对于每一个l,维护所有左端点<=l的区间中最大的r,设其为$s_i$,可以只考虑区间$(i,s_i)$

对于加入区间，相当于区间取max，因为s单调不降，所以相当于区间赋值

维护前缀和，那么区间和相当于两个前缀$sum_{s_i},sum_{i-1}$相减

对于后一个，可以在修改时区间加

对于前一个，修改时二分出区间，两个值分开维护，插入时直接区间赋值

$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500500
struct segt{int l,r,mxr,m2,l1,lz,mx;}e[N*4];
int n,m,a,b,c,tr[N];
void ad(int x,int k){for(int i=x;i<=n;i+=i&-i)tr[i]+=k;}
int qu(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
void pushdown(int x)
{
    if(e[x].l1)e[x<<1].mxr=e[x<<1|1].mxr=e[x<<1].l1=e[x<<1|1].l1=e[x<<1|1].m2=e[x<<1].m2=e[x].l1,e[x].l1=0;
    if(e[x].lz)e[x<<1].lz+=e[x].lz,e[x<<1|1].lz+=e[x].lz,e[x<<1].mx+=e[x].lz,e[x<<1|1].mx+=e[x].lz,e[x].lz=0;
}
void pushup(int x){e[x].mxr=max(e[x<<1].mxr,e[x<<1|1].mxr);e[x].m2=min(e[x<<1].m2,e[x<<1|1].m2);e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);}
void build(int x,int l,int r)
{
    e[x].l=l,e[x].r=r,e[x].mx=-2e9;
    if(l==r)return;
    int mid=(l+r)>>1;
    build(x<<1,l,mid);build(x<<1|1,mid+1,r);
}
void modify1(int x,int l,int r,int a)
{
    if(e[x].l==l&&e[x].r==r){e[x].mxr=e[x].l1=e[x].m2=a;return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)modify1(x<<1,l,r,a);
    else if(mid<l)modify1(x<<1|1,l,r,a);
    else modify1(x<<1,l,mid,a),modify1(x<<1|1,mid+1,r,a);
    pushup(x);
}
void modify2(int x,int l,int r,int a)
{
    if(e[x].l==l&&e[x].r==r){e[x].lz+=a,e[x].mx+=a;return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)modify2(x<<1,l,r,a);
    else if(mid<l)modify2(x<<1|1,l,r,a);
    else modify2(x<<1,l,mid,a),modify2(x<<1|1,mid+1,r,a);
    pushup(x);
}
void modify3(int x,int l,int a)
{
    if(e[x].l==e[x].r){e[x].lz=e[x].mx=a;return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid<l)modify3(x<<1|1,l,a);
    else modify3(x<<1,l,a);
    pushup(x);
}
int get4(int x,int s)
{
    if(e[x].l==e[x].r)return e[x].l;
    pushdown(x);
    if(e[x<<1].mxr<s)return get4(x<<1|1,s);
    else return get4(x<<1,s);
}
int get5(int x,int s)
{
    if(e[x].l==e[x].r)return e[x].l;
    pushdown(x);
    if(e[x<<1|1].m2<=s)return get5(x<<1|1,s);
    else return get5(x<<1,s);
}
int main()
{
    freopen("sparrow.in","r",stdin);
    freopen("sparrow.out","w",stdout);
    scanf("%d%d",&n,&m);
    build(1,0,n);
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d%d",&a,&b,&c);
        if(a==1) 
        {
            int lb=get4(1,b),rb=b;
            ad(b,c);
            if(lb>rb){printf("%d\n",e[1].mx);continue;}
            modify2(1,lb,rb,c);
        }
        if(a==2)
        {
            int rb=get5(1,c-1),lb=b;
            if(lb>rb){printf("%d\n",e[1].mx);continue;}
            int as=qu(c)-qu(b-1);
            modify1(1,lb,rb,c);
            modify3(1,b,as);
        }
        printf("%d\n",e[1].mx);
    }
}
```

#### 5.6 T2 字符串匹配(ricerca)

##### 题目

有两个字符串，每一个位置上都是一个字符集合，两个位置匹配仅当两个位置的集合有交，求所有匹配的位置

n<=1e5,字符集元素种数<=5,时限3s-8s

##### 分析

可以对于每一个位置差记录匹配位置对数，答案即为所有匹配位置对数量为匹配串长度的差对应的开头

枚举一个串的字符集，这样可以求出另一个串中有哪些位置匹配，做一个相差的fft

$O(2^5nlogn)$

~~实测bitset比fft快~~

~~实测手写bitset更快-----linli~~

##### 代码

```cpp
#include<cstdio>
#include<cstring>
using namespace std;
#define N 524300
#define mod 998244353
int a[N],b[N],c[N],s[N],t[N],as[N],n,m,rev[N],ntt[N],fc[2][20][N],lg[N];
char q[10];
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
    t=(t+1)>>1;
    for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int q=lg[l];
        for(int j=0;j<s;j+=l)
        for(int k=j,st=0;k<j+(l>>1);k++,st++)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*fc[t][q][st]%mod;
            ntt[k]=s1+s2-(s1+s2>=mod?mod:0);
            ntt[k+(l>>1)]=s1-s2+(s1<s2?mod:0);
        }
    }
    int inv=pw(s,t?0:mod-2);
    for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void solve(int x,int ct)
{
    for(int i=0;i<n;i++)
    if((s[i]&x)==x)a[i]=1;
    else a[i]=0;
    for(int i=0;i<m;i++)
    if((t[i]&x)==x)b[m-i]=1;
    else b[m-i]=0;
    int l=1;while(l<=n+m+1)l<<=1;
    for(int i=n;i<l;i++)a[i]=0;
    for(int i=m+1;i<l;i++)b[i]=0;b[0]=0;
    dft(l,a,1);dft(l,b,1);
    for(int i=0;i<l;i++)c[i]=1ll*a[i]*b[i]%mod;
    dft(l,c,-1);
    for(int i=0;i<n;i++)as[i]+=ct*c[i+m];
}
int main()
{
    freopen("ricerca.in","r",stdin);
    freopen("ricerca.out","w",stdout);
    scanf("%d",&n);
    for(int i=0;i<n;i++)
    {
        scanf("%s",q+1);
        int l=strlen(q+1);
        for(int j=1;j<=l;j++)
        s[i]+=(1<<(q[j]-'a'));
    }
    scanf("%d",&m);
    for(int i=0;i<m;i++)
    {
        scanf("%s",q+1);
        int l=strlen(q+1);
        for(int j=1;j<=l;j++)
        t[i]+=(1<<(q[j]-'a'));
    }
    for(int i=2;i<=524290;i++)lg[i]=lg[i>>1]+1;
    for(int i=1;i<=19;i++)
    {
        int st=pw(3,(mod-1)>>i),st2=pw(st,mod-2);
        fc[1][i][0]=1,fc[0][i][0]=1;
        for(int j=1;j<1<<i;j++)fc[1][i][j]=1ll*fc[1][i][j-1]*st%mod,fc[0][i][j]=1ll*fc[0][i][j-1]*st2%mod;
    }
    for(int i=1;i<32;i++)
    {
        int ct=-1;
        for(int j=0;j<5;j++)if(i&(1<<j))ct=-ct;
        solve(i,ct);
    }
    int ans=0;
    for(int i=0;i<n;i++)
    if(as[i]==m)ans++;
    printf("%d\n",ans);
    for(int i=0;i<n;i++)
    if(as[i]==m)printf("%d ",i);
}
```

#### 5.6 T3 图论原题(mincost)

给一张图，每个点有两个权值$a_i,b_i$，一个点集是合法的当且仅当

1.点集大小大于等于k

2.点集导出子图联通

一个点集的权值为$max a_i +max b_i$

求最小的合法点集权值

n<=1e5

##### 分析

如果是边权的话，可以直接扫描线+lct维护生成树

但是因为是点权，一次撤销复杂度非常大，显然是不行的 

但是对于$(x,y)$,可以将它的权值变为$(max(a_x,a_y),max(b_x,b_y))$

可以看出，在点集中任取一个生成树，它的权值都等于点集的权值

所以直接做，复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500020
int f[N],ch[N][2],sz[N],lsz[N],l[N],st[N],rb,n,m,k,mx[N],s[N][2],a[N],b[N],sum[N],su,q[N][2],t[N][2];
bool nroot(int x){return ch[f[x]][0]==x||ch[f[x]][1]==x;}
void pushdown(int x){if(l[x])swap(ch[ch[x][0]][0],ch[ch[x][0]][1]),swap(ch[ch[x][1]][0],ch[ch[x][1]][1]),l[ch[x][0]]^=1,l[ch[x][1]]^=1,l[x]=0;}
void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1+lsz[x];mx[x]=s[mx[ch[x][0]]][1]>s[mx[ch[x][1]]][1]?mx[ch[x][0]]:mx[ch[x][1]];if(mx[x]==0)mx[x]=x;if(s[mx[x]][1]<s[x][1])mx[x]=x;}
void rotate(int x)
{
    int a=f[x],b=f[a],tmp=ch[a][1]==x;
    if(nroot(a))ch[b][ch[b][1]==a]=x;f[x]=b;
    f[a]=x;f[ch[x][!tmp]]=a;
    ch[a][tmp]=ch[x][!tmp];ch[x][!tmp]=a;
    pushup(a);pushup(x);
}
void splay(int x)
{
    int s=x;
    while(nroot(s))st[++rb]=s,s=f[s];
    pushdown(s);while(rb)pushdown(st[rb--]);
    while(nroot(x))
    {
        int a=f[x],b=f[a];
        if(nroot(a))rotate((ch[b][1]==a)^(ch[a][1]==x)?x:a);
        rotate(x);
    }
}
void access(int x)
{
    int tmp=0;
    while(x)
    {
        splay(x);
        if(ch[x][1])lsz[x]+=sz[ch[x][1]];
        if(tmp)lsz[x]-=sz[tmp];
        ch[x][1]=tmp;
        pushup(x);
        tmp=x;x=f[x];
    }
}
void makeroot(int x){access(x);splay(x);l[x]^=1;swap(ch[x][0],ch[x][1]);}
void link(int x,int y)
{
    makeroot(x);
    access(y);splay(y);
    if(f[x])
    {
        if(mx[y]==x)
        return;
        int s=mx[y];
        splay(s);
        if(ch[s][0]){f[ch[s][0]]=0;ch[s][0]=0;pushup(s);}
        makeroot(x);makeroot(y);
        f[x]=y,lsz[y]+=sz[x];pushup(y);
    }
    else
    {
        makeroot(y);
        if(sz[x]>=k)su--;
        if(sz[y]>=k)su--;
        f[x]=y;lsz[y]+=sz[x];pushup(y);
        if(sz[y]>=k)su++;
    }
}
int cut(int x,int y)
{
    if(!su)return 0;
    makeroot(x);
    access(y);
    splay(y);
    if(!f[x])return 1;
    if(ch[y][0]!=x||ch[x][1])return 1;
    int sa=sz[x],sb=sz[y]-sz[x];
    if(sz[y]<k||sa>=k||sb>=k||su>=2)
    {
        su-=sz[y]>=k;
        su+=sa>=k;su+=sb>=k;
        ch[y][0]=f[x]=0;pushup(y);
        return 1;
    }
    return 0;
}
bool cmp1(int x,int y){return t[x][0]<t[y][0];}
bool cmp2(int x,int y){return t[x][1]<t[y][1];}
int main()
{
    freopen("mincost.in","r",stdin);
    freopen("mincost.out","w",stdout);
    scanf("%d%d%d",&n,&m,&k);
    for(int i=1;i<=n;i++)scanf("%d%d",&s[i][0],&s[i][1]),sz[i]=1,mx[i]=i;
    if(k==1)
    {
        int ans=2e9;
        for(int i=1;i<=n;i++)
        ans=min(ans,s[i][0]+s[i][1]);
        printf("%d\n",ans); 
        return 0;
    }
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d",&q[i][0],&q[i][1]);
        t[i][0]=max(s[q[i][0]][0],s[q[i][1]][0]);
        t[i][1]=max(s[q[i][0]][1],s[q[i][1]][1]);
        a[i]=b[i]=i;
    }
    sort(a+1,a+m+1,cmp1);
    sort(b+1,b+m+1,cmp2);
    int rb=m,ans=2.1e9;
    for(int i=1;i<=m;i++)
    {
        if(t[a[i]][1]>=t[b[rb]][1])continue;
        link(q[a[i]][0],q[a[i]][1]);
        while(rb>1&&cut(q[b[rb]][1],q[b[rb]][0]))rb--;
        if(su)ans=min(ans,t[b[rb]][1]+t[a[i]][0]);
    }
    if(ans>2000000500)printf("no solution");
    else printf("%d\n",ans);
}
```

#### 5.8 T1 suffixarray

##### 题目

![](/pic/28.png)

![](/pic/29.png)

##### 分析

神仙题

考虑后缀都是形如AAB...,AB...,BA...,BBA...

从2^n-1到2^n相当于将$(A,B)$换为$(AB,BA)$

那么会有ABABBA...,ABBA...,BAAB...,BABAAB...

总共有ABABBA...,BABBA...,ABBA...,BBA...,BAAB...,AAB...,BABAAB...,ABAAB...

特判当前sa对应的后缀为1的情况

对于剩下的情况，将上面的八种排序后一种一种考虑，处理每一个2的次幂时4种后缀的方案数，将问题递归求解

具体来说，先找到它属于八种中的哪一种，然后找到对应的四种之一，然后算出它在这一种中需要排在第几，再加上字典序在它前面的几种的数量

$O(n)$

##### 代码

```cpp
#include<cstdio>
using namespace std;
long long s,n,t,dp[61][4];
//1==AB
//2==AAB
//3==BA
//4==BBA
//ABBABAABBAABABBA
long long solve(int x,long long k)
{
    //AB-> ABBA4 BBA8
    //AAB-> ABABBA3 BABBA7
    //BA-> BAAB5 AAB1
    //BBA-> BABAAB6 ABAAB2
    int st=0;
    if(x&1)st++;
    if(x==0)return 1;
    if(x==1)return k;
    if(k==1)if(x&1)return (1ll<<x)-2;else return 1ll<<x;
    if(k==(1ll<<x-1)+1)if(x&1)return 1ll<<x;else return (1ll<<x)-1;
    if(~x&1)if(k==1)return (1ll<<x);else k--;
    if(k<=dp[x-1][2])return solve(x-1,k+1+dp[x-1][0]+dp[x-1][1])*2;else k-=dp[x-1][2];
    if(x&1)if(k==1)return (1ll<<x)-1;else k--;
    if(k<=dp[x-1][3])return solve(x-1,k+1+dp[x-1][2]+dp[x-1][0]+dp[x-1][1])*2;else k-=dp[x-1][3];
    if(k<=dp[x-1][1])return solve(x-1,k+st)*2-1;else k-=dp[x-1][1];
    if(k<=dp[x-1][0])return solve(x-1,k+st+dp[x-1][1])*2-1;else k-=dp[x-1][0];
    if(x&1)if(k==1)return (1ll<<x);else k--;
    if(~x&1)if(k==1)return (1ll<<x)-1;else k--;
    if(k<=dp[x-1][2])return solve(x-1,k+1+dp[x-1][0]+dp[x-1][1])*2-1;else k-=dp[x-1][2];
    if(k<=dp[x-1][3])return solve(x-1,k+1+dp[x-1][0]+dp[x-1][1]+dp[x-1][2])*2-1;else k-=dp[x-1][3];
    if(k<=dp[x-1][1])return solve(x-1,k+st)*2;else k-=dp[x-1][1];
    if(k<=dp[x-1][0])return solve(x-1,k+st+dp[x-1][1])*2;else k-=dp[x-1][0];
}
int main()
{
    freopen("a.in","r",stdin);
    freopen("a.out","w",stdout);
    dp[1][0]=1;
    for(int i=2;i<=60;i++)
    {
        if(i&1)
        dp[i][0]=dp[i-1][0]+dp[i-1][2]+1,
        dp[i][1]=dp[i-1][1]+dp[i-1][3],
        dp[i][2]=dp[i-1][0]+dp[i-1][2],
        dp[i][3]=dp[i-1][1]+dp[i-1][3];
        else
        dp[i][0]=dp[i-1][0]+dp[i-1][2],
        dp[i][1]=dp[i-1][1]+dp[i-1][3],
        dp[i][2]=dp[i-1][0]+dp[i-1][2],
        dp[i][3]=dp[i-1][1]+dp[i-1][3]+1;
    }
    scanf("%lld%lld",&s,&n);
    while(n--)scanf("%lld",&t),printf("%lld\n",solve(s,t));
}
```

#### 5.8 T2 infinitesequence

##### 题目

![](/pic/30.png)

![](/pic/31.png)

![](/pic/32.png)

##### 分析

对于二次线性递推，首先需要特征根方程

特征根方程为$x^2-Ax-1=0$,判别式为$A^2+4$

1.$A^2+4=0(mod p)$

解得的根为$x_1=x_2=A/2$

根据特征根方程，$c_1=0,(c_1+c_2)*A/2=1$,可得$c_1=0,c_2=2/A$

因此$a_i=n*(A/2)^{n-1}(mod p)$

注意到$A^2=-4 (mod p)$,所以$(A/2)^4=1 (mod p)$

按照nmod4余数分类，对于每一类解出方程然后crt最后把答案加起来

2.$A^2+4$存在二次剩余

设$z^2=A^2+4(mod p)$

根为$x_1=(A+z)/2,x_2=(A-z)/2$

$c_1+c_2=0,c_1x_1+c_2x_2=1$

$c1=1/z,c2=-1/z$

所以$a_n=1/z((A+z)/2)^n-1/z((A-z)/2)^n(mod p)$

$1/z((A+z)/2)^n-1/z((A-z)/2)^n=x (mod p)$

$((A+z)/2)^n-((A-z)/2)^n=xz (mod p)$

因为$(A+z)/2*(A-z)/2=1/4(A^2-z^2)=-1 (mod p)$

按照n的奇偶性分类，可以得到$(((A+z)/2)^n)^2-xz((A+z)/2)^n-1=0 (mod p)$

或者$(((A+z)/2)^n)^2-xz((A+z)/2)^n+1=0 (mod p)$

可以用配方+cipolla解出方程，然后问题变为知道$((A+z)/2)^n$，求所有合法的n

在bsgs时求出第二小的解即可

需要在前根号p项里面判一下重

然后带入crt解，再加起来

3.$A^2+4$不存在二次剩余

可以将所有数表示为$a+b\sqrt {A^2+4}$,然后使用和前面一样的方法，复数求倒数很简单

复杂度$O(T\sqrt n log n)$,使用手写hash表差不多0.6s

##### 代码

```cpp
#include<cstdio>
#include<cstdlib>
#include<algorithm>
#include<cmath>
#include<cstring>
#include<queue>
using namespace std;
long long a,p,x,l,r,T,as,tp1,tp2,s1;
int pw(int a,int b){int ans=1;while(b){if(b&1)ans=1ll*ans*a%p;a=1ll*a*a%p;b>>=1;}return ans;}
struct comp{int a,b;};
comp operator *(comp a,comp b){return (comp){(1ll*a.a*b.a+1ll*a.b*b.b%p*tp1)%p,(1ll*a.a*b.b+1ll*a.b*b.a)%p};}
comp pw(comp a,int b){comp ans=(comp){1,0};while(b){if(b&1)ans=ans*a;a=a*a;b>>=1;}return ans;}
int cipolla(int a)
{
    if(a==0)return 0;
    if(pw(a,(p-1)/2)==p-1)return -1;
    while(1)
    {
        tp1=(rand()<<15|rand())%p+1;
        tp2=tp1;
        tp1=(1ll*tp2*tp2-a+p)%p;
        if(pw(tp1,(p-1)/2)!=p-1)continue;
        break;
    }
    comp as=pw((comp){tp2,1},(p+1)/2);
    return (min(1ll*as.a,p-as.a)+p)%p;
}
long long gcd(long long a,long long b){return b?gcd(b,a%b):a;}
struct hashtable{
    int hd[1080050],nt[1080050],vl[1080050],ct;
    long long v[1080050];
    queue<int> st;
    void clear(){ct=0;while(!st.empty())hd[st.front()]=0,st.pop();}
    void ins(long long x,int v1)
    {
        int s=x&1048575;
        if(!hd[s])st.push(s);
        vl[++ct]=v1;v[ct]=x;nt[ct]=hd[s];hd[s]=ct;
    }
    int que(long long x)
    {
        int s=x&1048575;
        for(int i=hd[s];i;i=nt[i])if(v[i]==x)return vl[i];
        return -1;
    }
}tp;
struct subtask3{
    struct scomp{int a,b;}t1;
    friend bool operator <(scomp a,scomp b){return a.a==b.a?a.b<b.b:a.a<b.a;}
    friend scomp operator +(scomp a,scomp b){return (scomp){(a.a+b.a)-(a.a+b.a>=p?p:0),a.b+b.b-(a.b+b.b>=p?p:0)};}
    friend scomp operator -(scomp a,scomp b){return (scomp){(a.a-b.a)+(a.a<b.a?p:0),a.b-b.b+(a.b<b.b?p:0)};}
    friend scomp operator *(scomp a,scomp b){return (scomp){(1ll*a.a*b.a+1ll*a.b*b.b%p*s1)%p,(1ll*a.a*b.b+1ll*a.b*b.a)%p};}
    friend scomp inv(scomp a)
    {
        if(a.b==0)return (scomp){pw(a.a,p-2),0};
        //a*x+b*y*s1=1
        //a*y+b*x=0
        //a^2/b*y+a*x=0
        //(a^2/b+b*s1)y=1
        int as1=(p-1ll*a.a*a.a%p*pw(a.b,p-2)%p+1ll*a.b*s1)%p;
        return (scomp){(int)(p-1ll*a.a*pw(a.b,p-2)%p*pw(as1,p-2)%p)%p,pw(as1,p-2)};
    }
    friend scomp pw(scomp a,int b){scomp ans=(scomp){1,0};while(b){if(b&1)ans=ans*a;a=a*a;b>>=1;}return ans;}
    long long solvebsgs(scomp x,scomp a,long long r,long long k1)
    {
        tp.clear();
        int k=sqrt(p*5)+10;
        scomp s=(scomp){1,0};
        int fu=0;
        for(int i=0;i<k;i++)
        {
            if(tp.que(s.a*10000000000ll+s.b)!=-1)fu=i-tp.que(s.a*10000000000ll+s.b);
            tp.ins(s.a*10000000000ll+s.b,i);
            s=s*x;
        }
        scomp s2=s,st=inv(a);
        long long as=-1,as2=-1;
        long long ans=0;
        for(int i=1;i<=k;i++)
        {
            scomp t=s2*st;
            int tp2=tp.que(t.a*10000000000ll+t.b);
            if(tp2!=-1)
            {
                if(as2!=-1)break;
                if(as!=-1)as2=as,as=1ll*i*k-tp2,fu=gcd(fu,as-as2);
                else as=1ll*i*k-tp2;
            }
            s2=s2*s;
        }
        if(as==-1)return 0;
        as%=fu;
        if(fu%2)fu*=2;
        else if(as%2!=k1)return 0;
        if(as%2!=k1&&fu/2%2)as+=fu/2;
        return r/fu+(r%fu>=as);
    }
    long long solve2(long long r)
    {
        s1=(a*a+4)%p;
        scomp as1=(scomp){a*(p+1)/2%p,(p+1)/2%p},as2=(scomp){a*(p+1)/2%p,(p-1)/2%p};
        scomp t=(scomp){0,x};
        scomp f1=(scomp){1,0},f2=(scomp){1,0};
        long long s1=0;
        {
            scomp a=(scomp){1,0},b=(scomp){0,0}-t,c=(scomp){p-1,0};
            b=b*(scomp){(p-1)/2,0};
            c=c+b*b;
            int as3=cipolla(c.a);
            if(as3!=-1)
            {
                if(as3==0)s1+=solvebsgs(as1,b,r,1);
                else s1+=solvebsgs(as1,(scomp){as3,b.b},r,1)+solvebsgs(as1,(scomp){(p-as3)%p,b.b},r,1);
            }
        }
        {
            scomp a=(scomp){1,0},b=(scomp){0,0}-t,c=(scomp){1,0};
            b=b*(scomp){(p-1)/2,0};
            c=c+b*b;
            int as3=cipolla(c.a);
            if(as3!=-1)
            {
                if(as3==0)s1+=solvebsgs(as1,b,r,0);
                else s1+=solvebsgs(as1,(scomp){as3,b.b},r,0)+solvebsgs(as1,(scomp){(p-as3)%p,b.b},r,0);
            }
        }
        return s1;
    }
}fk;
long long solvebsgs(long long x,long long a,long long r,long long k1)
{
    tp.clear();
    int k=sqrt(p*2)+1;
    int s=1,fu=0;
    s=1;
    for(int i=0;i<k;i++)
    {
        if(tp.que(s)!=-1)fu=i-tp.que(s);
        tp.ins(s,i);
        s=1ll*s*x%p;
    }
    int s2=s,as=-1,as2=-1,inv=pw(a,p-2);
    long long ans=0;
    for(int i=1;i<=k;i++)
    {
        int t=1ll*s2*inv%p,tp2=tp.que(t);
        if(tp2!=-1)
        if(as!=-1)
        {
            if(as2!=-1)break;
            as2=as,as=i*k-tp2,fu=gcd(fu,as-as2);	
        }
        else as=i*k-tp2;
        s2=1ll*s2*s%p;
    }
    if(as==-1)return 0;
    as%=fu;
    if(fu%2)fu*=2;
    else if(as%2!=k1)return 0;
    if(as%2!=k1&&fu/2%2)as+=fu/2;
    return r/fu+(r%fu>=as);
}
long long solve3(long long a,long long b,long long r)
{
    long long lcm=p*4;
    long long as=-1;
    if(a%4==b)as=a;
    if((a+p)%4==b)as=a+p;
    if((a+p*2)%4==b)as=a+p*2;
    if((a+p*3)%4==b)as=a+p*3;
    if(as==-1)
    return 0;
    as%=lcm;
    return r/lcm+(r%lcm>=as);
}
long long solve2(long long r)
{
    int ans=cipolla((a*a+4)%p);
    if(ans==-1)return fk.solve2(r);
    int as1=1ll*(a+ans)*(p+1)/2%p,as2=1ll*(a-ans+p)*(p+1)/2%p;
    if(as1==as2)
    {
        long long as3=solve3(x,1,r)+solve3(p-(1ll*x*pw(as1,p-2))%p,0,r)+solve3(p-x,3,r)+solve3(1ll*x*pw(as1,p-2)%p,2,r);
        return as3;
    }
    else
    {
        int t=1ll*ans*x%p;
        long long s1=0;
        {
            int a=1,b=p-t,c=1;
            int ans2=cipolla(((1ll*b*b-4)%p+p)%p);
            if(ans2!=-1)
            {
                int a3=1ll*(ans2-b+p)%p*pw(2,p-2)%p,a4=1ll*(2*p-ans2-b)%p*pw(2,p-2)%p;
                if(a3==a4)s1+=solvebsgs(as1,a3,r,1);
                else s1+=solvebsgs(as1,a3,r,1)+solvebsgs(as1,a4,r,1);
            }
        }
        {
            int a=1,b=p-t,c=p-1;
            int ans2=cipolla(((1ll*b*b+4)%p+p)%p);
            if(ans2!=-1)
            {
                int a3=1ll*(ans2-b+p)%p*pw(2,p-2)%p,a4=1ll*(2*p-ans2-b)%p*pw(2,p-2)%p;
                if(a3==a4)s1+=solvebsgs(as1,a3,r,0);
                else s1+=solvebsgs(as1,a3,r,0)+solvebsgs(as1,a4,r,0);
            }
        }
        return s1;
    }
}
int main()
{
    srand(2332332333ll);
    freopen("b.in","r",stdin);
    freopen("b.out","w",stdout);
    scanf("%lld",&T);
    while(T--)
    {
        as=0;
        scanf("%lld%lld%lld%lld%lld",&a,&p,&x,&l,&r);
        a%=p;
        printf("%lld\n",solve2(r)-solve2(l-1));
    }
}
```

#### 5.10 T1 排序二叉树(tree)

##### 题目

![](/pic/33.png)

![](/pic/34.png)

![](/pic/35.png)

##### 分析

首先是$n=1$的情况

考虑一个点的父亲，一定是权值小于它的里面时间小于等于它且最大的和权值大于于它的里面时间小于等于它且最大的两者中时间最大的

可以发现，如果按照权值排序，时间作为值，那么所有的祖先即为从它向左的单调栈和向右的单调栈上的所有元素

考虑线段树维护单调栈，设$fl(l,r,k)$表示在区间中从左向右，第一个值大于等于k的单调栈和，那么

设$mx$为$l,mid$的最大值，如果$mx<=k$,那么$fl(l,r,k)=fl(mid+1,r,k)$

否则$fl(l,r,k)=fl(l,mid,k)+fl(mid+1,r,mx)$

在线段树上维护所有的$fl(mid+1,r,mx)$,每一次修改时会修改log个，每问一次$fl(mid+1,r,mx)$复杂度log，总复杂度$O(mlog^2m)$

对于另一个方向同样维护

对于n不为1的情况，使用扫描线即可，复杂度不变

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 400500
struct st{int k,a,b,t;}s[N];
bool cmp(st a,st b){return a.t<b.t;}
struct qu{int a,b,t;};
vector<qu> q[N];
int n,m,a,b,c,d,ct,ct2,sq[N],cnt,ct3,is[N],t2[N];
long long as[N];
struct segt{int l,r;long long mx,las,ras,vl;}e[N*4];
long long queryl(int x,int tp)
{
    if(e[x].l==e[x].r)return e[x].mx>tp?e[x].vl:0;
    if(tp>=e[x<<1|1].mx)return queryl(x<<1,tp);
    else return e[x].las+queryl(x<<1|1,tp);
}
long long queryr(int x,int tp)
{
    if(e[x].l==e[x].r)return e[x].mx>tp?e[x].vl:0;
    if(tp>=e[x<<1].mx)return queryr(x<<1|1,tp);
    else return e[x].ras+queryr(x<<1,tp);
}
void pushup(int x)
{
    e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);
    e[x].las=queryl(x<<1,e[x<<1|1].mx);
    e[x].ras=queryr(x<<1|1,e[x<<1].mx);
}
void build(int x,int l,int r)
{
    e[x].l=l;e[x].r=r;
    if(e[x].l==e[x].r){e[x].vl=sq[e[x].l];return;}
    int mid=(l+r)>>1;
    build(x<<1,l,mid);
    build(x<<1|1,mid+1,r);
}
void modify(int x,int s,int a)
{
    if(e[x].l==e[x].r){e[x].mx=a;return;}
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=s)modify(x<<1,s,a);
    else modify(x<<1|1,s,a);
    pushup(x);
} 
int querymx(int x,int l,int r)
{
    if(e[x].l==l&&e[x].r==r)return e[x].mx;
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return querymx(x<<1,l,r);
    else if(mid<l)return querymx(x<<1|1,l,r);
    else return max(querymx(x<<1,l,mid),querymx(x<<1|1,mid+1,r));
}
pair<long long,int> query2l(int x,int l,int r,int v)
{
    if(l>r)return make_pair(0ll,0);
    if(e[x].l==l&&e[x].r==r)return make_pair(queryl(x,v),max(v,querymx(1,l,r)));
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return query2l(x<<1,l,r,v);
    else if(mid<l)return query2l(x<<1|1,l,r,v);
    else
    {
        pair<long long,int> st=query2l(x<<1|1,mid+1,r,v);
        return make_pair(query2l(x<<1,l,mid,st.second).first+st.first,max(v,querymx(1,l,r)));
    }
}
pair<long long,int> query2r(int x,int l,int r,int v)
{
    if(l>r)return make_pair(0ll,0);
    if(e[x].l==l&&e[x].r==r)return make_pair(queryr(x,v),max(v,querymx(1,l,r)));
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return query2r(x<<1,l,r,v);
    else if(mid<l)return query2r(x<<1|1,l,r,v);
    else
    {
        pair<long long,int> st=query2r(x<<1,l,mid,v);
        return make_pair(query2r(x<<1|1,mid+1,r,st.second).first+st.first,max(v,querymx(1,l,r)));
    }
}
int main()
{
    freopen("tree.in","r",stdin);
    freopen("tree.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++)
    {
        scanf("%d",&a);
        if(a==1)scanf("%d%d%d",&b,&c,&d),s[++cnt]=(st){1,d,i,b},s[++cnt]=(st){2,d,i,c+1},sq[++ct3]=d;
        else scanf("%d%d",&b,&c),q[b].push_back((qu){c,++ct2,m-i+1});
    }
    sort(sq+1,sq+ct3+1);
    for(int i=1;i<=cnt;i++)s[i].a=lower_bound(sq+1,sq+ct3+1,s[i].a)-sq;
    build(1,1,ct3);
    sort(s+1,s+cnt+1,cmp);
    int nt=1;
    for(int i=1;i<=n;i++)
    {
        while(nt<=cnt&&s[nt].t<=i)
        {
            if(s[nt].k==1)modify(1,s[nt].a,m-s[nt].b+1),is[s[nt].a]=1,t2[s[nt].a]=m-s[nt].b+1;
            else modify(1,s[nt].a,0),is[s[nt].a]=0;
            nt++;
        }
        int sz1=q[i].size();
        for(int j=0;j<sz1;j++)
        {
            qu t=q[i][j];
            int lb=lower_bound(sq+1,sq+ct3+1,t.a)-sq-1,rb=lb+1;
            if(sq[rb]==t.a)
            as[t.b]=query2l(1,1,rb,t.t).first+query2r(1,rb,ct3,t.t).first-(is[rb]&&t2[rb]>=t.t?t.a:0);
            else 
            as[t.b]=query2l(1,1,lb,t.t).first+query2r(1,rb,ct3,t.t).first;
        }
    }
    for(int i=1;i<=ct2;i++)printf("%lld\n",as[i]);
}
```

#### 5.21 T1 送你一道签到题(count)

##### 题目

![](/pic/36.png)

![](/pic/37.png)

##### 分析

设$f(i)=i^k\sum_{d_1d_2d_3...d_m=i}g(d_1)g(d_2)...g(d_m)$(我忘了约数个数和怎么打了)

则$f(i)=\sum_{d_1d_2d_3...d_m=i}h(d_1)h(d_2)...h(d_m),h(i)=i^kg(i)$

发现f是m个积性函数的卷积，因此f为积性函数

显然$f(p)=p^k$

对于$f(p^i)$,可以发现是把i个p分到m个位置中，dp非零的位置再乘上组合数即可

于是使用min_25筛

O(min_25)

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<map>
using namespace std;
#define N 1105000
#define mod 998244353
int ch[N],pr[N],cnt,f[N],f2[N],g[N],s1[N],s2[N],id[N],q,p1[N],as3[N];
long long n;
int m,k,st[N],fr[N],ifr[N],pre[N],suf[N],pwi[N];
int pw(int a,int p){int as=1;while(p){if(p&1)as=1ll*as*a%mod;a=1ll*a*a%mod;p>>=1;}return as;}
map<long long,int> tp;
int getpw(long long s)
{
    if(s<=1e6)return st[s];
    if(tp.count(s))return tp[s];
    pre[0]=1,suf[k+3]=1;
    for(int i=1;i<=k+2;i++)pre[i]=1ll*pre[i-1]*((s-i)%mod)%mod;
    for(int i=k+2;i>=1;i--)suf[i]=1ll*suf[i+1]*((s-i)%mod)%mod;
    int ans=0;
    for(int i=1;i<=k+2;i++)
    ans=(ans+1ll*st[i]*((k+2-i)&1?-1:1)*ifr[i-1]%mod*ifr[k+2-i]%mod*pre[i-1]%mod*suf[i+1])%mod;
    return tp[s]=ans;
}
void prime(int n)
{
    for(int i=2;i<=n;i++)
    {
        if(!ch[i])pr[++cnt]=i;
        for(int j=1;i*pr[j]<=n&&j<=cnt;j++)
        {
            ch[i*pr[j]]=1;
            if(i%pr[j]==0)break;
        }
    }
}
long long getid(int x)
{
    if(x<=q)return x;
    return n/(q-(n/q==q)+q-x+1);
}
int gettid(long long x)
{
    if(x<=q)return x;
    return q-(n/q==q)+q-(n/x)+1;
}
int solvegv2(long long n){return 2ll*m*f[gettid(n)]%mod;}
int C(int x,int y)
{
    int ans=1;
    for(int i=1;i<=y;i++)ans=1ll*ans*(x-y+i)%mod*pw(i,mod-2)%mod;
    return ans;
}
int solve3(int x)
{
    if(as3[x])return as3[x];
    int dp[43][43]={0};
    dp[0][0]=1;
    for(int i=1;i<=x;i++)
    for(int j=1;j<=x;j++)
    for(int k=0;k<j;k++)
    dp[i][j]=((dp[i][j]+1ll*dp[i-1][k]*(j-k+1)))%mod;
    int ans=0;
    for(int i=1;i<=x;i++)ans=(ans+1ll*dp[i][x]*C(m,i))%mod;
    return as3[x]=ans;
}
int solves(long long n,long long p)
{
    if(n<=1||pr[p]>n)return 0;
    int ans=solvegv2(n);
    ans=(ans-1ll*2*m*s1[p-1])%mod;
    for(int j=p;j<=cnt&&1ll*pr[j]*pr[j]<=n;j++)
    for(long long s=pr[j],st=1;s<=n;s=s*pr[j],st++)
    ans=(ans+1ll*pw(s%mod,k)*solve3(st)%mod*(solves(n/s,j+1)+(s!=pr[j])))%mod;
    return ans;
}
void init()
{
    int ct=q*2;
    fr[0]=ifr[0]=1;
    for(int i=1;i<=1e6;i++)st[i]=(st[i-1]+pw(i,k))%mod,fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2),pwi[i]=pw(i,k);
    if(n/q==q)ct--;
    for(int i=1;i<=ct;i++)
    {
        long long st=getid(i);
        f[i]=getpw(st);
    }
    for(int i=1;i<=cnt;i++)
    for(int j=ct;j>=1;j--)
    {
        if(1ll*pr[i]*pr[i]>getid(j))break;
        f[j]=(f[j]-1ll*p1[i]%mod*(f[gettid(getid(j)/pr[i])]-s1[i-1]))%mod;
    }
}
int main()
{
    freopen("count.in","r",stdin);
    freopen("count.out","w",stdout);
    scanf("%lld%d%d",&n,&m,&k);
    prime(q=sqrt(n));
    s1[0]=s2[0]=1;
    for(int i=1;i<=cnt;i++)
    s1[i]=(s1[i-1]+(p1[i]=pw(pr[i],k)))%mod;
    init();
    printf("%d\n",(solves(n,1)+1+mod)%mod);
}
```

#### 5.22 T2 连续段

##### 题目

![](/pic/38.png)

![](/pic/39.png)

##### 分析

析合树计数

参考WC2019LCA课件

设f为答案生成函数，则会有

$f=x+f^2+f^3+2f^4+2f^5+...+2f^n+...$

(合点度数>=2,析点度数>=4)

所以$f=x+2f/(1-f)-f^2-f^3$

~~拿一个分治fft维护f,f/1-f,f^2,f^3就可以了,nlog^2n~~

可以使用牛顿迭代做到一个log($f^4+2f^2-(1+x)f+x=0$)

##### 代码

分治fft(dp=f/(1-f),as=f,son2=f^2,son3=f^3)

```cpp
#include<cstdio>
using namespace std;
#define N 263000
int dp[N],n,mod,son2[N],son3[N],as[N],g,ntt[N],a[N],b[N],c[N],rev[N],t1[N],t2[N],t3[N],fc[2][19][N],lg[N];
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
bool check(int x)
{
    for(int i=2;i*i<=mod-1;i++)
    if((mod-1)%i==0)
    if(pw(x,(mod-1)/i)==1)return 0;
    return 1;
}
int getas()
{
    for(int i=2;i<mod;i++)
    if(check(i))return i;
}
void dft(int s,int *a,int t)
{
    t=(t+1)>>1;
    for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+(i&1)*(s>>1),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int q=lg[l];
        for(int j=0;j<s;j+=l)
        for(int k=j,st=0;k<j+(l>>1);k++,st++)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*fc[t][q][st]%mod;
            ntt[k]=s1+s2-(s1+s2>=mod?mod:0);
            ntt[k+(l>>1)]=s1-s2+(s1<s2?mod:0);
        }
    }
    int inv=pw(s,t?0:mod-2);
    for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
void cdq(int l,int r)
{
    if(l==r)
    {
        if(l==1){as[1]=dp[1]=1;return;}
        as[l]=((1ll*dp[l]*2-son2[l]-son3[l])%mod+mod)%mod;
        dp[l]=(dp[l]+as[l])%mod;
        return;
    }
    int mid=(l+r)>>1;
    cdq(l,mid);
    //as*dp->dp
    int s=1;while(s<=r-l+mid-l)s<<=1;
    for(int i=0;i<s;i++)t1[i]=t2[i]=a[i]=b[i]=0;
    for(int i=l;i<=mid;i++)t1[i-l]=dp[i];
    for(int i=1;i<=r-l&&i<l;i++)b[i]=as[i];
    dft(s,t1,1);dft(s,b,1);for(int i=0;i<s;i++)c[i]=1ll*t1[i]*b[i]%mod;
    for(int i=0;i<s;i++)a[i]=b[i]=0;
    for(int i=l;i<=mid;i++)t2[i-l]=as[i];
    for(int i=1;i<=r-l&&i<l;i++)b[i]=dp[i];
    dft(s,t2,1);dft(s,b,1);for(int i=0;i<s;i++)c[i]=(c[i]+1ll*t2[i]*b[i])%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)dp[i]=(dp[i]+c[i-l])%mod;
    for(int i=0;i<s;i++)c[i]=1ll*t1[i]*t2[i]%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)if(i>=l*2)dp[i]=(dp[i]+c[i-l-l])%mod;
    //as*as->son2
    for(int i=0;i<s;i++)a[i]=b[i]=0;
    for(int i=1;i<=r-l&&i<l;i++)b[i]=as[i];
    dft(s,b,1);for(int i=0;i<s;i++)c[i]=1ll*t2[i]*b[i]%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)son2[i]=(son2[i]+1ll*c[i-l]*2)%mod;
    for(int i=0;i<s;i++)c[i]=1ll*t2[i]*t2[i]%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)if(i>=l*2)son2[i]=(son2[i]+c[i-l-l])%mod;
    //as*son2->son3
    for(int i=0;i<s;i++)t3[i]=a[i]=b[i]=0;
    for(int i=l;i<=mid;i++)t3[i-l]=son2[i];
    for(int i=1;i<=r-l&&i<l;i++)b[i]=as[i];
    dft(s,t3,1);dft(s,b,1);for(int i=0;i<s;i++)c[i]=1ll*t3[i]*b[i]%mod;
    for(int i=0;i<s;i++)a[i]=b[i]=0;
    for(int i=1;i<=r-l&&i<l;i++)b[i]=son2[i];
    dft(s,b,1);for(int i=0;i<s;i++)c[i]=(c[i]+1ll*t2[i]*b[i])%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)son3[i]=(son3[i]+c[i-l])%mod;
    for(int i=0;i<s;i++)a[i]=b[i]=0;
    for(int i=0;i<s;i++)c[i]=1ll*t2[i]*t3[i]%mod;dft(s,c,-1);
    for(int i=mid+1;i<=r;i++)if(i>=l*2)son3[i]=(son3[i]+c[i-l-l])%mod;
    cdq(mid+1,r);
}
void init()
{
    g=getas();
    for(int i=2;i<=262200;i++)lg[i]=lg[i>>1]+1;
    for(int i=1;i<=18;i++)
    {
        int st=pw(g,(mod-1)>>i),st2=pw(st,mod-2);
        fc[1][i][0]=1,fc[0][i][0]=1;
        for(int j=1;j<1<<i;j++)fc[1][i][j]=1ll*fc[1][i][j-1]*st%mod,fc[0][i][j]=1ll*fc[0][i][j-1]*st2%mod;
    }
}
int main()
{
    freopen("b.in","r",stdin);
    freopen("b.out","w",stdout);
    scanf("%d%d",&n,&mod);
    init();cdq(1,n);
    for(int i=1;i<=n;i++)printf("%d\n",as[i]);
}
```

#### 5.30 T1 duliu

##### 题目

![](/pic/40.png)

![](/pic/41.png)

##### 分析

设$g(i,j)=\sum_{l_i<=a,b<=r_i}d_i+d_j$,那么$g(i,j)=(r_i-l_i+1)*(sum_{r_i}-sum_{l_i-1})$，可以快速算

这样的话大区间值一定大于小区间值

考虑如果一个区间端点一侧的点的d小于等于当前区间max，则显然可以扩展

那么，只会有两种区间

1.至少有一个端点为边界

2.左右端点两侧值都大于区间max

对于1，直接二分出$l=l_i$时最小右端点$mr$和$r=r_i$时最大左端点$ml$

对于区间2，考虑把在区间max位置统计。可以发现以一个点为max的区间只会存在一个，因此这样的区间只有n个

先算出n个区间，询问按照$x_i$排序

如果一个区间$[l_a,r_a]$满足$l_i<=l_a<=r_i<r_a$,那么显然它的值大于等于$[l_a,r_i]$

将所有区间按左端点插入，每次询问查询$[l_i,ml]$中的min

$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 300050
int v[N],n,m,l[N],r[N],st[N],rb,ct,a,b,as[N],rm[N][21],lg[N];
long long su[N],c;
struct query{int l,r,k,id;long long v;}s[N*2];
bool cmp(query a,query b){return a.v==b.v?a.k<b.k:a.v>b.v;}
struct segt{int l,r;long long mn;}e[N*4];
void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);}
void build(int x,int l,int r)
{
    e[x].mn=998244853;
    e[x].l=l;e[x].r=r;
    if(l==r)return;
    int mid=(e[x].l+e[x].r)>>1;
    build(x<<1,l,mid);build(x<<1|1,mid+1,r);
    pushup(x);
}
void modify1(int x,int s,long long v)
{
    if(e[x].l==e[x].r){e[x].mn=min(e[x].mn,v);return;}
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=s)modify1(x<<1,s,v);
    else modify1(x<<1|1,s,v);
    pushup(x);
}
long long getmn(int x,int l,int r)
{
    if(l>r)return 0;
    if(e[x].l==l&&e[x].r==r)return e[x].mn;
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return getmn(x<<1,l,r);
    else if(mid<l)return getmn(x<<1|1,l,r);
    else return min(getmn(x<<1,l,mid),getmn(x<<1|1,mid+1,r));
}
long long getas(int l,int r){return 1ll*(su[r]-su[l-1])*(r-l+1);}
int rmq(int a,int b)
{
    return max(rm[a][lg[b-a+1]],rm[b-(1<<lg[b-a+1])+1][lg[b-a+1]]);
}
int main()
{
    freopen("duliu.in","r",stdin);
    freopen("duliu.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)scanf("%d",&v[i]),su[i]=su[i-1]+v[i];
    v[n+1]=v[0]=998244352;
    for(int i=1;i<=n+1;i++)
    {
        while(v[i]>v[st[rb]]&&rb)r[st[rb]]=i-1,rb--;
        st[++rb]=i;
    }
    rb=0;st[1]=0;
    for(int i=n;i>=0;i--)
    {
        while(v[i]>=v[st[rb]]&&rb)l[st[rb]]=i+1,rb--;
        st[++rb]=i;
    }
    lg[0]=-1;
    for(int i=1;i<=n;i++)rm[i][0]=v[i],lg[i]=lg[i>>1]+1;
    for(int i=1;i<=20;i++)
    for(int j=1;j+(1<<i)-1<=n;j++)
    rm[j][i]=max(rm[j][i-1],rm[j+(1<<i-1)][i-1]);
    for(int i=1;i<=n;i++)s[++ct]=(query){l[i],r[i],1,i,getas(l[i],r[i])};
    for(int i=1;i<=m;i++)scanf("%d%d%lld",&a,&b,&c),s[++ct]=(query){a,b,2,i,(c+1)/2};
    sort(s+1,s+ct+1,cmp);
    build(1,1,n);
    for(int i=1;i<=ct;i++)
    {
        if(s[i].k==1)modify1(1,s[i].l,v[s[i].id]);
        else
        {
            if(getas(s[i].l,s[i].r)<s[i].v){as[s[i].id]=-1;continue;}
            int mn=1e8,la=s[i].r,ra=s[i].l;
            {
                int lb=s[i].l,rb=s[i].r,ans=rb;
                while(lb<=rb)
                {
                    int mid=(lb+rb)>>1;
                    if(getas(s[i].l,mid)>=s[i].v)ans=mid,rb=mid-1;
                    else lb=mid+1;
                }
                la=ans;
                mn=min(mn,rmq(s[i].l,ans));
            }
            {
                int lb=s[i].l,rb=s[i].r,ans=rb;
                while(lb<=rb)
                {
                    int mid=(lb+rb)>>1;
                    if(getas(mid,s[i].r)>=s[i].v)ans=mid,lb=mid+1;
                    else rb=mid-1;
                }
                ra=ans;
                mn=min(mn,rmq(ans,s[i].r));
            }
            as[s[i].id]=min(mn,(int)getmn(1,s[i].l,ra));
        }
    }
    for(int i=1;i<=m;i++)printf("%d\n",as[i]);
}
```

#### 5.30 T2 gre

##### 题目

![](/pic/42.png)

多组数据,T<=100,n<=1e5,k<=26

##### 分析

考虑字典序相邻两种字符x,y

如果当前出现了a个y，一种最优的构造方式是在前面塞至少2个x，然后每一个y后面一个x，最后一个y后面再加一个x

例如:bbbb->a...ababababaa

按照贪心的思路，除了a以外前面都只塞2个，最后如果构造出来长度>n则无解，否则往前面放a

构造的部分长度只有26^2*3/2,所以暴力复杂度$O(T(n+k^4))$，可以通过

##### 代码

```cpp
#include<cstdio>
using namespace std;
int T,n,q,le;
char as[23333];
void ins(int a,char s)
{
    for(int i=le;i>=a;i--)as[i+1]=as[i];
    as[a]=s;le++;
}
int main()
{
    freopen("gre.in","r",stdin);
    freopen("gre.out","w",stdout);
    scanf("%d",&T);
    while(T--)
    {
        le=0;
        scanf("%d%d",&n,&q);
        if(q==1)
        while(n--)printf("a");
        else
        {
            le=1;as[1]='a'+q-1;
            for(int i=q-1;i>=1;i--)
            {
                ins(1,i+'a'-1);
                ins(1,i+'a'-1);
                for(int j=1;j<=le;j++)
                if(as[j]==i+'a')ins(j+1,i+'a'-1);
                ins(le,i+'a'-1);
            }
            if(n<le)printf("CiYe");
            else
            {
                n-=le;while(n--)printf("a");
                printf("%s",as+1);
            }
            for(int i=1;i<=le;i++)as[i]=0;
        }
        printf("\n");
    }
}
```

### Part 3 JZ Revisit

#### 6.17 T1 担心(worry)

##### 题目

n个人站成一排，每个人有一个di每一时刻会随机有两个相邻的人进行比赛，i胜j的概率为di/(di+dj),输的人离开队伍，求最后第k个人留下来的概率模998244353

n<=200

##### 分析

设$dp[i][j][k]$表示$(i,j)$中k留下的概率

可以枚举最后另外一个人，以及他打败的区间

$dp[i][j][k]=\sum_{s=i}^{k-1}\sum_{t=s}^{k-1}dp[t+1][j][k]*dp[i][t][s]*1/(j-i)*d_k/(d_k+d_s)+\sum_{s=k+1}^{j}\sum_{t=k+1}^sdp[i][t-1][k]*dp[t][j][s]*1/(j-i)*d_k/(d_k+d_s)$

乘上1/(j-i)是因为这一场必须是最后一场

复杂度$O(n^5)$

考虑它左右两边是相互独立的，所以可以得到$dp[i][j][k]=dp[i][k][k]*dp[k][j][k]$

设$f[i][j]=dp[i][j][i],g[i][j]=dp[i][j][j]$

那么$f[i][j]=\sum_{s=i+1}^j\sum_{t=i+1}^s f[i][t-1]*g[t][s]*f[s][j]*1/(j-i)*d_i/(d_i+d_s)$

复杂度$O(n^4)$

$f[i][j]=\sum_{s=i+1}^j\sum_{t=i+1}^s f[i][t-1]*g[t][s]*f[s][j]*1/(j-i)*d_i/(d_i+d_s)$

$f[i][j]=1/(j-i)\sum_{s=i+1}^jf[i][t-1]*g[t][s]*d_i/(d_i+d_s)\sum_{t=i+1}^s f[s][j]$

前缀和即可做到$O(n^3)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#define N 501
#define mod 998244353
int dp[N][N][N],v[N],is[N][N],inv[N],k,n,h[N][N];
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int main()
{
    freopen("worry.in","r",stdin);
    freopen("worry.out","w",stdout);
    scanf("%d%d",&n,&k);
    for(int i=1;i<=n;i++)scanf("%d",&v[i]);
    for(int i=1;i<=n;i++)inv[i]=pw(i,mod-2);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=n;j++)
    if(i!=j)is[i][j]=1ll*v[i]*pw(v[i]+v[j]%mod,mod-2)%mod;
    for(int i=1;i<=n;i++)dp[i][i][i]=h[i][i]=1;
    for(int l=1;l<n;l++)
    for(int i=1;i+l<=n;i++)
    {
        int j=i+l;
        for(int k=i;k<j;k++)h[i][j]=(h[i][j]+1ll*dp[i][k][i]*dp[k+1][j][j])%mod;
        for(int t=i+1;t<=j;t++)
        dp[i][j][i]=(dp[i][j][i]+1ll*is[i][t]*h[i][t]%mod*dp[t][j][t]%mod*inv[l])%mod;
        for(int s=i;s<j;s++)
        dp[i][j][j]=(dp[i][j][j]+1ll*is[j][s]*dp[i][s][s]%mod*h[s][j]%mod*inv[l])%mod;
    }
    printf("%d\n",1ll*dp[1][k][k]*dp[k][n][k]%mod);
}
```

#### 6.17 T2 可爱(lovely)

##### 题目

给一个长度为n的串，求对于每一个长度为m的字串有多少长度为m的字串和它相差最多一个字符（不包括自己）

n,m<=1e5

##### 分析

首先可以SA处理完全相等的情况

对于不等的情况，设两个串为$(i,j),(k,l)$，那么$lcp(i...n,k...n)+lcp(j...1,l...1)=m-1$,其中后面两个是反串

对正串和反串分别建SA，从大到小枚举正串的lcp，若当前枚举到k，则只有正串sa上之间的height都大于等于k的是合法的，这个可以看成每次合并两个集合

在合并的时候，可以每一个集合维护集合内部所有位置对应反串位置上的rank，每一次合并增加的对即为一段rank连续的后缀，可以二分出区间

采用启发式合并，每一次插入一个点时先求出它新增的方案数，然后对对应区间区间+1，使用splay维护，复杂度$O(nlog^2n)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 200050
struct SA{
    int sa[N],rk[N],he[N],a[N],b[N],ct[N],n,m,mn[N][19],lg[N];
    char s[N];
    void pre()
    {
        m=n+255;
        for(int i=1;i<=n;i++)ct[a[i]=s[i]]++;
        for(int i=1;i<=m;i++)ct[i]+=ct[i-1];
        for(int i=n;i>=1;i--)
        sa[ct[a[i]]--]=i;
        for(int l=1;l<=n;l<<=1)
        {
            int cnt=0;
            for(int i=n;i>n-l;i--)b[++cnt]=i;
            for(int i=1;i<=n;i++)if(sa[i]>l)b[++cnt]=sa[i]-l;
            for(int i=1;i<=m;i++)ct[i]=0;
            for(int i=1;i<=n;i++)ct[a[i]]++;
            for(int i=1;i<=m;i++)ct[i]+=ct[i-1];
            for(int i=n;i>=1;i--)sa[ct[a[b[i]]]--]=b[i];
            for(int i=1;i<=n;i++)b[i]=a[i];
            int tp=2;a[sa[1]]=1;
            for(int i=2;i<=n;i++)a[sa[i]]=b[sa[i]]==b[sa[i-1]]&&b[sa[i]+l]==b[sa[i-1]+l]?tp-1:tp++;
            m=tp-1;
        }
        for(int i=1;i<=n;i++)rk[sa[i]]=i;
        int tp=0;
        for(int i=1;i<=n;i++)
        {
            if(tp)tp--;if(rk[i]==1)continue;
            while(s[i+tp]==s[sa[rk[i]-1]+tp])tp++;
            he[rk[i]]=tp;
        }
        for(int i=2;i<=n;i++)mn[i][0]=he[i];
        for(int j=1;j<=18;j++)
        for(int i=2;i+(1<<j)-1<=n;i++)
        mn[i][j]=min(mn[i][j-1],mn[i+(1<<j-1)][j-1]);
        for(int i=2;i<=n;i++)lg[i]=lg[i>>1]+1;
    }
    int lcp(int i,int j)
    {
        if(i>j)i^=j^=i^=j;if(i==j)return 1e8;
        i++;
        
        int tp=lg[j-i+1];
        return min(mn[i][tp],mn[j-(1<<tp)+1][tp]);
    }
}s,rs;
#define M 6000070
int ch[M][2],lz[M],ct,rt[M],sz[M],vl[M],st,fa[N],n,m,sp[N],as[N];
void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]];}
void pushdown(int x){if(lz[x])lz[ch[x][0]]+=lz[x],lz[ch[x][1]]+=lz[x],lz[x]=0;lz[0]=0;}
bool cmp(int a,int b){return s.he[a]>s.he[b];}
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
void treeinit()
{
    for(int i=m;i<=n;i++)
    {
        int lb=1,rb=n;rt[i]=ct+1;
        while(1)
        {
            int st=++ct,mid=(lb+rb)>>1;
            sz[st]=1;if(lb==rb)break;
            if(mid>=rs.rk[n-i+1])ch[st][0]=st+1,rb=mid;
            else ch[st][1]=st+1,lb=mid+1;
        }
        vl[ct]=rs.rk[n-i+1];
    }
}
int modify1(int x,int l,int r,int l1,int r1)
{
    if(!x)return 0;
    if(l==l1&&r==r1){lz[x]++;return sz[x];}
    pushdown(x);
    int mid=(l+r)>>1;
    if(mid>=r1)return modify1(ch[x][0],l,mid,l1,r1);
    else if(mid<l1)return modify1(ch[x][1],mid+1,r,l1,r1);
    else return modify1(ch[x][0],l,mid,l1,mid)+modify1(ch[x][1],mid+1,r,mid+1,r1);
}
void dfs(int x,int t)
{
    if(ch[x][0]+ch[x][1])
    {
        pushdown(x);
        if(ch[x][0])dfs(ch[x][0],t);
        if(ch[x][1])dfs(ch[x][1],t);
        return;
    }
    int tp=vl[x];
    int lb=1,rb=tp,ans=-1;
    while(lb<=rb)
    {
        int mid=(lb+rb)>>1;
        if(rs.lcp(mid,tp)>=st)ans=mid,rb=mid-1;
        else lb=mid+1;
    }
    int lb2=tp,rb2=n,ans2=-1;
    while(lb2<=rb2)
    {
        int mid=(lb2+rb2)>>1;
        if(rs.lcp(tp,mid)>=st)ans2=mid,lb2=mid+1;
        else rb2=mid-1;
    }
    if(ans==-1||ans2==-1)return;
    int tp1=modify1(rt[t],1,n,ans,ans2);
    lz[x]+=tp1;
    return;
}
int treeun(int a,int b)
{
    if(!a)return b;
    if(!b)return a;
    pushdown(a);pushdown(b);
    ch[a][0]=treeun(ch[a][0],ch[b][0]);
    ch[a][1]=treeun(ch[a][1],ch[b][1]);
    pushup(a);
    return a;
}
void un(int a,int b)
{
    if(a==b)return;
    a=finds(a),b=finds(b);
    if(sz[rt[a]]>sz[rt[b]])a^=b^=a^=b;
    fa[a]=b;
    dfs(rt[a],b);
    rt[b]=treeun(rt[a],rt[b]);
}
void dfs2(int x)
{
    if(!x)return;
    if(ch[x][0]+ch[x][1])pushdown(x),dfs2(ch[x][0]),dfs2(ch[x][1]);
    else as[n-rs.sa[vl[x]]-m+2]=lz[x];
}
int main()
{
    freopen("lovely.in","r",stdin);
    freopen("lovely.out","w",stdout);
    scanf("%d%d%s",&n,&m,s.s+1);s.n=rs.n=n;
    for(int i=1;i<=n;i++)rs.s[i]=s.s[n-i+1];
    s.pre(),rs.pre();
    treeinit();
    for(int i=1;i<n;i++)sp[i]=i+1;
    sort(sp+1,sp+n,cmp);
    int st1=1;
    for(int i=1;i<=n+m;i++)fa[i]=i;
    for(int i=m;i>=0;i--)
    {
        st=m-i-1;
        while(st1<=n&&s.he[sp[st1]]>=i)
        un(s.sa[sp[st1]]+m-1,s.sa[sp[st1]-1]+m-1),st1++;
    }
    dfs2(rt[finds(m)]);
    for(int i=1;i<=n-m+1;i++)printf("%d ",as[i]);
}
```

#### 6.18 T1 正方形(square)

##### 题目

有一个n*m矩阵，上面有一些障碍，有一个正方形，边长为k，它的左上角坐标为(xi,yi),求它能否移动到(si,ti),多组询问（如果初始位置不合法则为no）

n,m<=1000,q<=1e5

##### 分析

只考虑左上角的移动

可以发现，一个点作为左上角时合法的k一定是小于某个数，可以前缀和+二分算出

按k从大到小排序，每次相当于加入一些点，求联通性，可以使用并查集

注意讨论起点终点相等且不合法的情况（否则70）

$O(nmlogn+q)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1011
#define M 100040
#define K 1002333
int fa[K],su[N][N],tp[N][N],n,m,q,a,b,c,d,e,is[N][N],as[M];
struct st{int i,j;friend bool operator <(st a,st b){return tp[a.i][a.j]>tp[b.i][b.j];}};
char t[N][N];
struct que{int a,b,c,d,e,id;friend bool operator <(que a,que b){return a.e>b.e;}};
st s1[K];
que qu[M];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int main()
{
    freopen("square.in","r",stdin);
    freopen("square.out","w",stdout);
    scanf("%d%d%d",&n,&m,&q);
    for(int i=1;i<=n;i++)scanf("%s",t[i]+1);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    su[i][j]=su[i-1][j]+su[i][j-1]-su[i-1][j-1]+t[i][j]-'0';
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    {
        int lb=1,rb=min(n-i+1,m-j+1),ans=0;
        while(lb<=rb)
        {
            int mid=(lb+rb)>>1;
            if(su[i+mid-1][j+mid-1]-su[i+mid-1][j-1]-su[i-1][j+mid-1]+su[i-1][j-1]==0)
            ans=mid,lb=mid+1;
            else rb=mid-1;
        }
        tp[i][j]=ans;
        s1[i*m-m+j]=(st){i,j};
        fa[i*m-m+j]=i*m-m+j;
    }
    for(int i=1;i<=q;i++)scanf("%d%d%d%d%d",&a,&b,&c,&d,&e),qu[i]=(que){a-e+1,b-e+1,c-e+1,d-e+1,e,i};
    sort(s1+1,s1+n*m+1);sort(qu+1,qu+q+1);
    int l1=1,l2=1;
    for(int i=min(n,m);i>=1;i--)
    {
        while(l1<=n*m&&tp[s1[l1].i][s1[l1].j]>=i)
        {
            int x=s1[l1].i,y=s1[l1].j;is[x][y]=1;
            if(is[x][y-1])fa[finds(x*m-m+y)]=finds(x*m-m+y-1);
            if(is[x][y+1])fa[finds(x*m-m+y)]=finds(x*m-m+y+1);
            if(is[x-1][y])fa[finds(x*m-m+y)]=finds(x*m-2*m+y);
            if(is[x+1][y])fa[finds(x*m-m+y)]=finds(x*m+y);
            l1++;
        }
        while(l2<=q&&qu[l2].e>=i)
        {
            if(is[qu[l2].a][qu[l2].b]&&finds(qu[l2].a*m-m+qu[l2].b)==finds(qu[l2].c*m-m+qu[l2].d))as[qu[l2].id]++;
            l2++;
        }
    }
    for(int i=1;i<=q;i++)printf("%s\n",as[i]?"Yes":"No");
}
```

#### 6.18 T2 计数(count)

##### 题目

![](/pic/43.png)

n<=250000

##### 分析

考虑每一对的贡献

设两个位置为i,j，则贡献为$\sum_{k=0}^{min(i-1,n-j)}C_{i-1}^kC_{m-j}^k*2^{j-i}$

第一个代表枚举两侧放几个，第二个相当于枚举中间的放置

根据基本组合知识，可以得到$\sum_{k=0}^{min(i-1,n-j)}C_{i-1}^kC_{m-j}^k=\sum_{k=0}^{min(i-1,n-j)}C_{i-1}^kC_{m-j}^{m-j-k}=C_{m-j+i-1}^{i-1}$

所以相当于$\sum_{i=1}^n \sum_{j=i+1}^n[s_i=s_j]*C_{m-j+i-1}^{i-1}*2^{j-i}$

等于$\sum_{i=1}^n \sum_{j=i+1}^n[s_i=s_j]*(m-j+i-1)!/(m-j)!/(i-1)!/2^{m-j}/2^{i-1}*2^{m-1}$

对于m-j,i-1 fft

复杂度$O(nlogn)$

#### 6.18 T3 纳什均衡(nash)

##### 题目

![](/pic/44.png)

60% k<=3

80% n<=500

n<=2000,k<=20

##### 分析

考虑一个dp

$dp[i][a][b][c][d]$表示考虑i为根的子树，第一个人当前值为a，他只改变自己可以得到的最优值为b，第二个人当前值为c，他只改变自己可以得到的最优值为d的方案数

转移时比较显然，若当前为第一个人选，则$dp[i][a1][max(b1,b2)][c1][d1]+=dp[lson[i]][a1][b1][c1][d1]*dp[rson[i]][a2][b2][c2][d2]$

$dp[i][a2][max(b1,b2)][c1][d1]+=dp[lson[i]][a1][b1][c1][d1]*dp[rson[i]][a2][b2][c2][d2]$

第二个人同理，复杂度$O(nk^8)$

考虑最终答案只与a,b和c,d是否相等有关，所以可以设$dp[i][b][d][0/1][0/1]$后两维表示a,b是否相等，c,d是否相等

复杂度$O(nk^4)$

事实上，这个转移可以很方便地使用前缀和优化，可以做到$O(nk^3)$

具体参考代码

##### 代码

(只找到了nk^4的代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 5050
#define M 23
#define mod 998244353
int dp[N][M][M][2][2],ch[N][2],n,k,a,dep[N];
int main()
{
    freopen("nash.in","r",stdin);
    freopen("nash.out","w",stdout);
    scanf("%d%d",&n,&k);
    for(int i=2;i<=n;i++)
    {
        scanf("%d",&a);a++;dep[i]=dep[a]+1;
        if(!ch[a][0])ch[a][0]=i;
        else ch[a][1]=i;
    }
    for(int i=n;i>=1;i--)
    {
        if(!ch[i][0])
        {
            for(int j=1;j<=k;j++)
            for(int l=1;l<=k;l++)
            dp[i][j][l][1][1]=1;
        }
        else
        {
            for(int a=1;a<=k;a++)
            for(int b=1;b<=k;b++)
            for(int c=1;c<=k;c++)
            for(int d=1;d<=k;d++)
            {
                int s1=((long long)dp[ch[i][0]][a][b][0][0]+dp[ch[i][0]][a][b][1][0]+dp[ch[i][0]][a][b][0][1]+dp[ch[i][0]][a][b][1][1])%mod,
                s2=((long long)dp[ch[i][1]][c][d][0][0]+dp[ch[i][1]][c][d][1][0]+dp[ch[i][1]][c][d][0][1]+dp[ch[i][1]][c][d][1][1])%mod;
                if(dep[i]&1)
                {
                    int tp=max(a,c);
                    dp[i][tp][b][a>=c][1]=(dp[i][tp][b][a>=c][1]+1ll*s2*dp[ch[i][0]][a][b][1][1])%mod;
                    dp[i][tp][b][a>=c][0]=(dp[i][tp][b][a>=c][0]+1ll*s2*dp[ch[i][0]][a][b][1][0])%mod;
                    dp[i][tp][b][0][1]=(dp[i][tp][b][0][1]+1ll*s2*dp[ch[i][0]][a][b][0][1])%mod;
                    dp[i][tp][b][0][0]=(dp[i][tp][b][0][0]+1ll*s2*dp[ch[i][0]][a][b][0][0])%mod;
                    dp[i][tp][d][c>=a][1]=(dp[i][tp][d][c>=a][1]+1ll*s1*dp[ch[i][1]][c][d][1][1])%mod;
                    dp[i][tp][d][c>=a][0]=(dp[i][tp][d][c>=a][0]+1ll*s1*dp[ch[i][1]][c][d][1][0])%mod;
                    dp[i][tp][d][0][1]=(dp[i][tp][d][0][1]+1ll*s1*dp[ch[i][1]][c][d][0][1])%mod;
                    dp[i][tp][d][0][0]=(dp[i][tp][d][0][0]+1ll*s1*dp[ch[i][1]][c][d][0][0])%mod;
                }
                else
                {
                    int tp=max(b,d);
                    dp[i][a][tp][1][b>=d]=(dp[i][a][tp][1][b>=d]+1ll*s2*dp[ch[i][0]][a][b][1][1])%mod;
                    dp[i][a][tp][0][b>=d]=(dp[i][a][tp][0][b>=d]+1ll*s2*dp[ch[i][0]][a][b][0][1])%mod;
                    dp[i][a][tp][1][0]=(dp[i][a][tp][1][0]+1ll*s2*dp[ch[i][0]][a][b][1][0])%mod;
                    dp[i][a][tp][0][0]=(dp[i][a][tp][0][0]+1ll*s2*dp[ch[i][0]][a][b][0][0])%mod;
                    dp[i][c][tp][1][d>=b]=(dp[i][c][tp][1][d>=b]+1ll*s1*dp[ch[i][1]][c][d][1][1])%mod;
                    dp[i][c][tp][0][d>=b]=(dp[i][c][tp][0][d>=b]+1ll*s1*dp[ch[i][1]][c][d][0][1])%mod;
                    dp[i][c][tp][1][0]=(dp[i][c][tp][1][0]+1ll*s1*dp[ch[i][1]][c][d][1][0])%mod;
                    dp[i][c][tp][0][0]=(dp[i][c][tp][0][0]+1ll*s1*dp[ch[i][1]][c][d][0][0])%mod;
                }
            }
        }
    }
    int ans=0;
    for(int i=1;i<=k;i++)
    for(int j=1;j<=k;j++)
    ans=(ans+dp[1][i][j][1][1])%mod;
    printf("%d\n",ans);
}
```

#### 6.20 T1 running

##### 题目

![](/pic/46.png)

![](/pic/47.png)

![](/pic/48.png)

n,m<=1e5

**10s**,512MB

简要题意：带linkcut，修改的天天爱跑步

##### 分析

这是一道说起来还行写起来爆炸的题

把操作分块，每一次处理$\sqrt n $操作

将所有在这些操作中没有被修改的边缩起来，每一次2操作用lct维护，额外记录每条lct上的边对应原图的哪一条边

这样的话，每次1操作在lct上split可以得到$\sqrt n$条路径，且这些路径与2操作无关，并且包含了所有点

考虑怎么处理3操作

因为3操作只影响$\sqrt n$个点，先对$\sqrt n*\sqrt n$段做一次原题，然后可以发现每一个缩点以后的点只有$\sqrt n$条路径上，对于每一个修改的点，暴力算它是否可以观察到每一条路径上的人

如果使用tarjanLCA，总复杂度$O(n\sqrt n)$

如果使用rmqLCA，可以做到$O(n\sqrt {nlogn})$,卡一下可以过

从提交结果来看，tarjanLCA3-4s，rmqLCA9.5s

##### 代码

rmqLCA

//7KB

```cpp
#include<cstdio>
#include<set>
#include<vector>
#include<queue>
#include<map>
using namespace std;
#define N 230050
#define M 2100
int sz=1300,n,m,q[N][5],s1[N*2][3],fa[N],a,b,c,d,e,ct,as2[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
struct lct{
    int fa[N],ch[N][2],l[N],st[N],rb,sz[N];
    vector<int> tp;
    void clear(){for(int i=1;i<=n;i++)fa[i]=ch[i][0]=ch[i][1]=l[i]=sz[i]=0;}
    bool nroot(int x){return ch[fa[x]][0]==x||ch[fa[x]][1]==x;}
    void pushdown(int x){if(l[x])swap(ch[ch[x][0]][0],ch[ch[x][0]][1]),swap(ch[ch[x][1]][0],ch[ch[x][1]][1]),l[ch[x][0]]^=1,l[ch[x][1]]^=1,l[x]=0;}
    void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1;}
    void rotate(int x){int f=fa[x],g=fa[f],tmp=ch[f][1]==x;if(nroot(f))ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tmp]=ch[x][!tmp];fa[ch[x][!tmp]]=f;ch[x][!tmp]=f;fa[f]=x;pushup(f);pushup(x);}
    void splay(int x)
    {
        int tp=x;while(nroot(tp))st[++rb]=tp,tp=fa[tp];
        pushdown(tp);while(rb)pushdown(st[rb--]);
        while(nroot(x))
        {
            int f=fa[x],g=fa[f];
            if(nroot(f))rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);
            rotate(x);
        }
    }
    void access(int x){int tmp=0;while(x){splay(x);ch[x][1]=tmp;pushup(x);tmp=x;x=fa[x];}}
    int makeroot(int x){access(x);splay(x);l[x]^=1;swap(ch[x][0],ch[x][1]);}
    void split(int x,int y){
    makeroot(x);
    access(y);
    splay(y);}
    void link(int x,int y){makeroot(x);makeroot(y);fa[x]=y;pushup(y);}
    void cut(int x,int y){split(x,y);ch[y][0]=fa[x]=0;pushup(y);}
    int kth(int x,int k){pushdown(x);if(sz[ch[x][0]]>=k)return kth(ch[x][0],k);else if(sz[ch[x][0]]+1<k)return kth(ch[x][1],k-sz[ch[x][0]]-1);return x;}
    void dfs(int x){if(!x)return;pushdown(x);dfs(ch[x][0]);tp.push_back(x);dfs(ch[x][1]);}
    vector<int> sol(int y){tp.clear();dfs(y);return tp;}
}t;
struct rmqlca{
    struct edge{int t,next;}ed[N*2];
    int mn[N*2][19],st[N*2],lb[N],rb[N],ct,dep[N],head[N],cnt,lg[N*2];
    void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
    void init(){ct=0;for(int i=1;i<=n;i++)head[i]=0;cnt=ct=0;for(int i=1;i<n;i++)adde(s1[i][0],s1[i][1]);}
    void dfs(int u,int fa)
    {
        dep[u]=dep[fa]+1;st[++ct]=u;lb[u]=ct;
        for(int i=head[u];i;i=ed[i].next)
        if(ed[i].t!=fa)dfs(ed[i].t,u),st[++ct]=u;
        rb[u]=ct;
    }
    int lca(int a,int b)
    {
        int l1,r1;
        if(lb[a]>lb[b])l1=lb[b],r1=rb[a];
        else l1=lb[a],r1=rb[b];
        int tp=lg[r1-l1+1];
        int a1=mn[l1][tp],b1=mn[r1-(1<<tp)+1][tp];
        if(dep[a1]<dep[b1])return a1;
        else return b1;
    }
    void pre()
    {
        init();dfs(1,0);
        for(int i=1;i<=ct;i++)mn[i][0]=st[i];
        for(int i=2;i<=ct;i++)lg[i]=lg[i>>1]+1;
        for(int i=1;i<=18;i++)
        for(int j=1;j+(1<<i)-1<=ct;j++)
        {
            int a=mn[j][i-1],b=mn[j+(1<<i-1)][i-1];
            if(dep[a]<dep[b])mn[j][i]=a;else mn[j][i]=b;
        }
    }
}rm;
struct noip2016{
    int q[N][5],st[N*2][2],dep[N],w[N],head[N],cnt,as[N],f[N],vl[N],ct;
    vector<int> is;
    queue<int> s[N],s2[N];
    map<long long,long long> fu2;
    struct sth{int a,b,c,d;};
    vector<sth> tp1[N];
    struct edge{int t,next,v;}ed[N*2];
    void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
    void dfs2(int u,int fa)
    {
        dep[u]=dep[fa]+1;f[u]=fa;
        for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)vl[ed[i].t]=ed[i].v,dfs2(ed[i].t,u);
    }
    void dfs(int u,int fa)
    {
        as[u]-=st[w[u]+dep[u]][1]+st[w[u]-dep[u]+100000][0];
        for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa)dfs(ed[i].t,u);
        while(!s[u].empty())
        {
            int tp=s[u].front();s[u].pop();
            int s=tp/2000000,q=tp/1000000%2*2-1,t=tp%1000000;
            if(s==1)st[t+dep[u]][1]+=q;
            else st[t-dep[u]+100000][0]+=q;
        }
        as[u]+=st[w[u]+dep[u]][1]+st[w[u]-dep[u]+100000][0];
        while(!s2[u].empty())
        {
            int tp=s2[u].front();s2[u].pop();
            int s=tp/2000000,q=tp/1000000%2*2-1,t=tp%1000000;
            st[t+dep[u]][1]+=q;
            st[t-dep[u]+100000][0]+=q;
        }
    }
    void addq(int x,int y,int t)
    {
        tp1[finds(x)].push_back((sth){1,x,y,t});
        int l=rm.lca(x,y);
        s[x].push(t+3000000);
        s2[l].push(t+dep[x]-dep[l]+2000000);
        s[y].push(t+dep[x]+dep[y]-dep[l]*2+1000000);
        if(t+dep[x]-dep[l]==w[l])as[l]--;
    }
    void init()
    {
        for(int i=0;i<=n*2;i++)st[i][0]=st[i][1]=0;
        for(int i=1;i<=n;i++)head[i]=as[i]=f[i]=dep[i]=0,tp1[i].clear();
        t.clear();fu2.clear();is.clear();
        cnt=0;ct=0;
    }
    int getsz(int i,int j){return dep[i]+dep[j]-2*dep[rm.lca(i,j)]+1;}
    void sth1(int su)
    {
        init();
        for(int i=1;i<=n;i++)fa[i]=i;
        for(int i=1;i<n;i++)adde(s1[i][0],s1[i][1],i);
        dfs2(1,0);
        for(int i=1;i<=su;i++)
        if(q[i][0]==2)
        {
            int a=q[i][1],b=q[i][2];
            if(dep[a]>dep[b])a^=b^=a^=b;
            if(f[b]==a)s1[vl[b]][2]=1;
        }
        for(int i=1;i<n;i++)if(!s1[i][2])fa[finds(s1[i][0])]=finds(s1[i][1]);
        for(int i=1;i<n;i++)if(s1[i][2])t.link(finds(s1[i][0]),finds(s1[i][1])),fu2[1ll*finds(s1[i][0])*100050+finds(s1[i][1])]=fu2[1ll*finds(s1[i][1])*100050+finds(s1[i][0])]=1ll*100050*s1[i][0]+s1[i][1];
        rm.pre();
        for(int i=1;i<=su;i++)
        if(q[i][0]==1)
        {
            int st=0;
            t.split(finds(q[i][1]),finds(q[i][2]));
            vector<int> as=t.sol(finds(q[i][2])),as2;
            int nw=q[i][1];
            int sz=as.size();
            as2.push_back(q[i][1]);
            for(int i=0;i<sz-1;i++)
            {
                long long tp=fu2[1ll*as[i]*100050+as[i+1]];
                int s1=tp/100050,s2=tp%100050;
                if(finds(s1)!=as[i])s1^=s2^=s1^=s2;
                as2.push_back(s1),as2.push_back(s2);
            }
            as2.push_back(q[i][2]);
            sz=as2.size();
            for(int i=0;i<sz;i+=2)
            {
                addq(as2[i],as2[i+1],st);
                st+=getsz(as2[i],as2[i+1]);
            }
        }
        else if(q[i][0]==2)
        {
            t.cut(finds(q[i][1]),finds(q[i][2]));
            t.link(finds(q[i][3]),finds(q[i][4]));
            fu2[1ll*finds(q[i][3])*100050+finds(q[i][4])]=fu2[1ll*finds(q[i][4])*100050+finds(q[i][3])]=1ll*q[i][3]*100050+q[i][4];
        }
        else tp1[finds(q[i][1])].push_back((sth){3,q[i][1],q[i][2]});
        dfs(1,0);
        for(int i=1;i<=su;i++)
        if(q[i][0]==3)
        if(as[q[i][1]]!=-1)as[q[i][1]]=-1,is.push_back(q[i][1]);
        int sz=is.size();
        for(int i=0;i<sz;i++)as[is[i]]=0;
        for(int i=0;i<sz;i++)
        {
            int tp=is[i],sp=finds(tp);
            int sz=tp1[sp].size();
            for(int j=0;j<sz;j++)
            {
                sth sp1=tp1[sp][j];
                if(sp1.a==1)
                {
                    int s1=getsz(sp1.b,tp),s2=getsz(sp1.c,tp),s3=getsz(sp1.b,sp1.c);
                    if(s1+s2==s3+1&&s1+sp1.d-1==w[tp])as[tp]++;
                }
                else if(sp1.b==tp)
                w[tp]=sp1.c;
            }
        }
        int lb=1;
        for(int i=1;i<=su;i++)
        if(q[i][0]==2)
        {
            int fg=1;
            for(int j=i+1;j<=su;j++)
            if(q[j][0]==2&&((q[j][1]==q[i][3]&&q[j][2]==q[i][4])||(q[j][1]==q[i][4]&&q[j][2]==q[i][3])))fg=0;
            if(fg)
            while(lb<n)
            {
                if(s1[lb][2]==1){s1[lb][2]=0,s1[lb][0]=q[i][3],s1[lb][1]=q[i][4];break;}
                lb++;
            }
        }
        for(int i=1;i<=n;i++)as2[i]+=as[i];
    }
}fuc;
int main()
{
    freopen("running.in","r",stdin);
    freopen("running.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<n;i++)scanf("%d%d",&s1[i][0],&s1[i][1]);
    for(int i=1;i<=n;i++)scanf("%d",&fuc.w[i]);
    for(int i=1;i<=m;i++)
    {
        scanf("%d",&q[i][0]);
        if(q[i][0]!=2)scanf("%d%d",&q[i][1],&q[i][2]);
        else scanf("%d%d%d%d",&q[i][1],&q[i][2],&q[i][3],&q[i][4]);
    }
    for(int i=1;i<=m;i+=sz)
    {
        int lb=i,rb=min(i+sz-1,m);
        for(int j=lb;j<=rb;j++)
        for(int k=0;k<5;k++)fuc.q[j-lb+1][k]=q[j][k];
        fuc.sth1(rb-lb+1);
    }
    for(int i=1;i<=n;i++)printf("%d ",as2[i]);
}
```

#### 6.20 T2 sum

##### 题目

给定n，求$\sum_{i=1}^n\sum_{j=1}^i\sum_{k=1}^i[(i,j),(i,k)] \mod 2^{32}$

n<=1e10,3s

##### 分析

设$f(i)=\sum_{j=1}^i\sum_{k=1}^i[(i,j),(i,k)]$

那么$f(i)=\sum_{j|i}\sum_{k|i}lcm(j,k)phi(i/j)phi(i/k)$

$f(i)=\sum_{j|i}\sum_{k|i}i/gcd(j,k)*phi(j)phi(k) $

可以看出各因子间独立，这是一个积性函数

又因为，$f(p)=3p^2-3p+1$,$f(p^k)$可以容斥算出

因此使用min_25即可

##### 代码

```cpp
#include<cstdio>
#include<cmath>
using namespace std;
#define N 200050
#define ul unsigned int
#define ll long long
int ch[N],pr[N],ct,p;
ul f[N],su[N],g[N],s2[N],h[N],s3[N];
ll n;
int prime(int n)
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
int getid(ll x){return x<=p?x:p*2-(1ll*p*p==n)-n/x+1;}
ll gettid(int x){if(x<=p)return x;x=2*p-(1ll*p*p==n)+1-x;return n/x;}
void init()
{
    int tp=p*2-(1ll*p*p==n);
    for(int i=1;i<=tp;i++)
    {
        ll s1=gettid(i),s2=s1+1;
        if(s1%2)s2/=2;
        else s1/=2;
        f[i]=(ul)s1*(ul)s2-1;
    }
    for(int i=1;i<=tp;i++)
    {
        ll s1=gettid(i),s2=s1+1,s3=s1*2+1;
        if(s1%2)s2/=2;
        else s1/=2;
        if(s1%3)if(s2%3)s3/=3;
        else s2/=3;else s1/=3;
        g[i]=(ul)s1*(ul)s2*s3-1;
    }
    for(int i=1;i<=tp;i++)
    {
        ll s1=gettid(i);
        h[i]=(ul)s1-1;
    }
    for(int i=1;i<=ct;i++)su[i]=su[i-1]+pr[i],s2[i]=s2[i-1]+(ll)pr[i]*pr[i],s3[i]=s3[i-1]+1;
    for(int i=1;i<=ct;i++)
    for(int j=tp;j>=1;j--)
    {
        if(1ll*pr[i]*pr[i]>gettid(j))break;
        f[j]=f[j]-f[getid(gettid(j)/pr[i])]*pr[i]+su[i-1]*pr[i];
    }
    for(int i=1;i<=ct;i++)
    for(int j=tp;j>=1;j--)
    {
        if(1ll*pr[i]*pr[i]>gettid(j))break;
        g[j]=g[j]-g[getid(gettid(j)/pr[i])]*pr[i]*pr[i]+s2[i-1]*pr[i]*pr[i];
    }
    for(int i=1;i<=ct;i++)
    for(int j=tp;j>=1;j--)
    {
        if(1ll*pr[i]*pr[i]>gettid(j))break;
        h[j]=h[j]-h[getid(gettid(j)/pr[i])]+s3[i-1];
    }
}
ll pw(ll a,ll p){ll ans=1;while(p){if(p&1)ans*=a;a*=a;p>>=1;}return ans;}
ul solve2(ll n){return g[getid(n)]*3-3*f[getid(n)]+h[getid(n)];}
ul solve(ll a,int p)
{
    ul f[45]={0},as=0;long long tp=1,tp2=pw(a,p);
    for(int i=0;i<=40&&tp<=n;i++)f[i]=(ul)(tp2/tp)*(ul)tp2*2-(ul)(tp2/tp)*(ul)(tp2/tp),tp*=a;
    for(int i=0;i<=40;i++)f[i]-=f[i+1];
    tp=1;
    for(int i=0;i<=40&&tp<=n;i++)as+=f[i]*tp,tp*=a;
    return as;
}
ul solveas(ll n,ll p)
{
    if(n<=1||pr[p]>n||(p>ct&&pr[ct]>=n))return 0;
    ul ans=solve2(n);
    ans=ans+3*su[p-1]-3*s2[p-1]-s3[p-1];
    for(int j=p;j<=ct&&1ll*pr[j]*pr[j]<=n;j++)
    for(ll s=pr[j],tp=1;s<=n;s*=pr[j],tp++)
    ans=(ans+((s!=pr[j])+solveas(n/s,j+1))*solve(pr[j],tp));
    return ans;
}
int main()
{
    freopen("sum.in","r",stdin);
    freopen("sum.out","w",stdout);
    scanf("%lld",&n);
    prime(p=sqrt(n));
    init();
    printf("%u\n",solveas(n,1)+1);
}
```

#### 6.20 T3 number

##### 题意

定义f(i)表示i的约数个数，给定n，求所有满足f(x)=n的x的和模p，如果有无数个x输出-1

多组数据,T<=10,n,p<=1e18

##### 分析

如果$x=p_1^{k_1}*p_2^{k_2}*...*p_i^{k_i}$,那么$f(x)=(k_1+1)*(k_2+1)*...*(k_i+1)$

考虑对于n的一个质因子$s^k$,可以构造出$s^{s^k-1}$,显然有$f(s^{s^k-1})=s^k$

如果存在一个$s^k$使得$s>=3,k>=2$，那么

显然$s>=3$

$s^k-s^{k-1}>=1$

所以$s^{k-1}-1>=k$

因此可以构造$s^{s^{k-1}-1}*p^{s-1}$,它的f为$s^k$

因为p可以随便取，所以答案为-1，否则

如果不存在$k>=2$,可以发现n的每一个因子对应f里面一个乘积

所以只可能是$p_i^{p_{s_i}-1}$的乘积，其中$s$为任意排列

因为1e18里面最多的质因子只有15个，可以状压dp

否则，存在$s=2,k>=2$

如果$k>=3$

$2^2>=3+1$

所以可以一样构造，答案为-1

否则$s=k=2$

如果存在另外一个p，那么

$f(p*2^{p-1}*p_1)=2^2p$

因为$p_1$可以任意取，所以答案为-1

否则，$n=4$.可以得到$ans=8$

(事实证明所有数据都有$n=4$)

1e18分解质因数用pollard-rho

需要使用~~快速乘~~__int128

##### 代码

```cpp
#include<cstdio>
#include<cstdlib>
#include<algorithm>
using namespace std;
#define N 71
#define M 16
long long T,n,p,f[N],mrt[N]={2,3,7,11,13,61,10007,24251},ct,dp[1<<M],bitc[1<<M],pw2[M][M];
#define mod 10007
#define ll __int128
ll pw(ll a,ll p,ll md){ll ans=1;while(p){if(p&1)ans=ans*a%md;a=a*a%md;p>>=1;}return ans;}
bool mrtest(ll a,ll p){for(ll i=p-1;;i>>=1){ll tp=pw(a,i,p);if(tp==p-1)return 0;if(tp!=1)return 1;if(i&1)return 0;}}
bool mr(ll a){for(int i=0;i<8&&mrt[i]<a;i++)if(mrtest(mrt[i],a))return 0;return 1;}
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll pr(ll a)
{
    ll s1=1ll*rand()*1000000000+1ll*rand()*50000+rand(),s2,ad=rand()%(a-1)+1,tp=1;
    s2=s1%a,s1=(s2*s2+ad)%a;
    ll g=s1>s2?s1-s2:s2-s1;
    if(gcd(a,g)!=1&&gcd(a,g)!=a)return gcd(a,g);
    while(1)
    {
        if(s1==s2)return pr(a);
        if((tp&-tp)==tp)s2=s1;
        s1=(s1*s1+ad)%a;tp++;
        ll g=s1>s2?s1-s2:s2-s1;
        if(gcd(a,g)!=1&&gcd(a,g)!=a)
        return gcd(a,g);
    }
}
void dfs(ll x){if(mr(x)){f[++ct]=x;return;}if(x%2==0){dfs(2),dfs(x/2);return;}ll tp=pr(x);dfs(tp),dfs(x/tp);}
int main()
{
    freopen("number.in","r",stdin);
    freopen("number.out","w",stdout);
    for(int i=1;i<=65535;i++)bitc[i]=bitc[i-(i&-i)]+1;
    scanf("%lld",&T);
    while(T--)
    {
        scanf("%lld%lld",&n,&p);
        if(n==1){printf("%d\n",1%p);continue;}
        if(n==4){printf("%d\n",8%p);continue;}
        ct=0;dfs(n);
        sort(f+1,f+ct+1);
        int fg=0;
        for(int i=1;i<=ct;i++)if(f[i]==f[i-1])fg=1;
        if(fg==1){printf("-1\n");continue;}
        for(int i=0;i<1<<ct;i++)dp[i]=0;
        for(int i=1;i<=ct;i++)
        for(int j=1;j<=ct;j++)
        pw2[i][j]=pw(f[i],f[j]-1,p);
        dp[0]=1;
        for(int i=0;i<1<<ct;i++)
        {
            int tp=bitc[i]+1;
            if(tp>ct)continue;
            for(int j=1;j<=ct;j++)
            if(!(i&(1<<j-1)))
            dp[i|(1<<j-1)]=(dp[i|(1<<j-1)]+(__int128)pw2[tp][j]*dp[i])%p;
        }
        printf("%lld\n",dp[(1<<ct)-1]);
    }
}
```

#### 6.21 T1 ichi

##### 题目

给一棵带边权点权的以1为根的树，支持

1.询问一个点的值

2.给x子树内到x路径上边权都大于等于d的点点权+s，每一次的d不一样

强制在线，n<=1e5,3s,256MB

##### 分析

首先建kruskal重构树，那么边权限制就变成了这上面的一个子树

然后原树和kruskal重构树按dfs序排列以后就变成了二维区间加单点查询

直接使用线段树+splay维护，时间$O(nlog^2n)$,空间$O(nlogn)$，可能需要卡一下空间（反正我是250MB过的）

(貌似n>5000没有强制在线的，然后被水过了)

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 100050
int n,m,v[N],a,b,c,d,k;
long long lastans;
struct dfs_tree{
    struct edge{int t,next;}ed[N*2];
    int head[N],cnt,id[N],tid[N],ct,lb[N],rb[N];
    void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
    void dfs(int u,int fa)
    {
        lb[u]=id[u]=++ct;
        tid[ct]=u;
        for(int i=head[u];i;i=ed[i].next)
        if(ed[i].t!=fa)dfs(ed[i].t,u);
        rb[u]=ct;
    }
}t;
struct edg{int f,t,v;}e[N];
bool cmp(edg a,edg b){return a.v>b.v;}
struct kruskal_tree{
    int f[N*2][19],vl[N*2],ct,fa[N],v2[N*2],ch[N*2][2],lb[N*2],rb[N*2],id[N],ct2;
    int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
    void dfs(int u)
    {
        lb[u]=ct2+1;
        for(int i=1;i<=18;i++)f[u][i]=f[f[u][i-1]][i-1];
        if(u>n)dfs(ch[u][0]),dfs(ch[u][1]);
        else id[u]=++ct2;
        rb[u]=ct2;
    }
    int build()
    {
        sort(e+1,e+n+1,cmp);ct=n;
        for(int i=1;i<=n;i++)fa[i]=vl[i]=i;
        for(int i=1;i<=n;i++)
        if(finds(e[i].f)!=finds(e[i].t))
        {
            int a=finds(e[i].f),b=finds(e[i].t);
            f[vl[a]][0]=++ct;f[vl[b]][0]=ct;ch[ct][0]=vl[a],ch[ct][1]=vl[b];v2[ct]=e[i].v;
            vl[b]=ct;fa[a]=b;
        }
        dfs(ct);
    }
    int que(int x,int y)
    {
        for(int i=18;i>=0;i--)
        if(v2[f[x][i]]>=y&&f[x][i])
        x=f[x][i];
        return x;
    }
}kt;
#define M 7200050
int ch[M][2],fa[M],v2[M],ct;
long long su[M],vl[M];
struct Splay{
    int rt;
    void pushup(int x){su[x]=su[ch[x][0]]+su[ch[x][1]]+vl[x];}
    void rotate(int x){int f=fa[x],g=fa[f],tmp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;fa[ch[x][!tmp]]=f;ch[f][tmp]=ch[x][!tmp];ch[x][!tmp]=f;fa[f]=x;pushup(f);pushup(x);}
    void splay(int x){while(fa[x]){int f=fa[x],g=fa[f];if(g)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);rotate(x);}rt=x;}
    void insert(int x,int v,int s)
    {
        if(!rt){rt=++ct;su[ct]=vl[ct]=s;v2[ct]=v;return;}
        if(v2[x]==v){vl[x]+=s;su[x]+=s;splay(x);return;}
        bool tp=v2[x]<v;
        if(!ch[x][tp])
        {
            ch[x][tp]=++ct;
            fa[ct]=x;su[ct]=vl[ct]=s;v2[ct]=v;
            splay(ct);return;
        }
        insert(ch[x][tp],v,s);
    }
    long long query(int x,int v)
    {
        if(!x)return 0;
        if(v2[x]>v){if(ch[x][0])return query(ch[x][0],v);else {splay(x);return 0;}}
        if(v2[x]==v){long long ans=su[ch[x][0]]+vl[x];splay(x);return ans;}
        if(v2[x]<v){long long ans=su[ch[x][0]]+vl[x];if(ch[x][1])return ans+query(ch[x][1],v);else{splay(x);return ans;}}
    }
};
struct tree_tree{
    struct node{int l,r;Splay t;}st[N*4];
    void build(int x,int l,int r)
    {
        st[x].l=l;st[x].r=r;
        if(l==r)return;
        int mid=(l+r)>>1;
        build(x<<1,l,mid);build(x<<1|1,mid+1,r);
    }
    void add(int x,int r,int s,int v)
    {
        if(r>n||s>n)return;
        st[x].t.insert(st[x].t.rt,s,v);
        if(st[x].l==st[x].r)return;
        int mid=(st[x].l+st[x].r)>>1;
        if(mid>=r)add(x<<1,r,s,v);
        else add(x<<1|1,r,s,v);
    }
    long long que(int x,int a,int b)
    {
        if(st[x].l==st[x].r)return st[x].t.query(st[x].t.rt,b);
        int mid=(st[x].l+st[x].r)>>1;
        if(mid>=a)return que(x<<1,a,b);
        else return st[x<<1].t.query(st[x<<1].t.rt,b)+que(x<<1|1,a,b);
    }
}as;
int main()
{
    freopen("ichi.in","r",stdin);
    freopen("ichi.out","w",stdout);
    scanf("%d%d%d",&n,&m,&k);
    for(int i=1;i<=n;i++)scanf("%d",&v[i]);
    for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),t.adde(a,b),e[i]=(edg){a,b,c};
    t.dfs(1,0);kt.build();as.build(1,1,n);
    while(m--)
    {
        scanf("%d",&a);
        if(a==1)
        {
            scanf("%d",&b);
            if(k==1)b=(b+lastans)%n+1;
            printf("%lld\n",lastans=as.que(1,t.id[b],kt.id[b])+v[b]);
        }
        else
        {
            scanf("%d%d%d",&d,&c,&b);
            if(k==1)b=(b+lastans)%n+1;
            int l1=t.lb[b],r1=t.rb[b];
            int tp=kt.que(b,c);
            int l2=kt.lb[tp],r2=kt.rb[tp];
            as.add(1,l1,l2,d);
            as.add(1,r1+1,l2,-d);
            as.add(1,l1,r2+1,-d);
            as.add(1,r1+1,r2+1,d);
        }
    }
}
```

#### 6.21 T2 ni

##### 题目

![](/pic/49.png)

n<=500000

30% $E_i>=0$

##### 分析

首先考虑非负的做法

显然最优解为从小到大排序之后传递

设$g(i)=E_i+n-i$,表示i之后所有人都+1的最终值

显然存在一个位置使得他选的是0，从他往后所有人都+1（可能是第0个人）

如果到第i个人时数字小于$E_i$,那么最终值小于$g(i)$

如果第i个人之后有人选0，那么最终值小于$g(i)$

所以答案为$min_{i=0}^n g(i)$

所以直接线段树维护g即可

对于有负数的情况，可以发现一定是开始有一些人选-1，然后变成上一种情况，二分到那个位置即可

复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 500050
int n,v[N],t[N],tr[N],ct[N*2];
inline int ad(int x){for(int i=x;i<=n;i+=i&-i)tr[i]++;}
inline int qu(int x){int as=0;for(int i=x;i;i-=i&-i)as+=tr[i];return as;}
inline int Min(int a,int b){return a<b?a:b;}
struct node{int l,r,l1,l2,mn,mx;}e[N*4];
inline void pushdown(int x){if(e[x].l1)e[x<<1].l1+=e[x].l1,e[x<<1].mn+=e[x].l1,e[x<<1|1].l1+=e[x].l1,e[x<<1|1].mn+=e[x].l1,e[x].l1=0;if(e[x].l2)e[x<<1].mx+=e[x].l2,e[x<<1].l2+=e[x].l2,e[x<<1|1].mx+=e[x].l2,e[x<<1|1].l2+=e[x].l2,e[x].l2=0;}
inline void pushup(int x){e[x].mx=Min(e[x<<1].mx,e[x<<1|1].mx);e[x].mn=Min(e[x<<1].mn,e[x<<1|1].mn);}
void build(int x,int l,int r){e[x].l=l;e[x].r=r;e[x].mn=e[x].mx=998244353;if(l==r)return;int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);}
void modify(int x,int a,int p)
{
    if(e[x].l==e[x].r){e[x].mn=p+e[x].l1;e[x].mx=p+e[x].l2;return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=a)modify(x<<1,a,p);
    else modify(x<<1|1,a,p);
    pushup(x);
}
int que1(int x)
{
    if(e[x].l==e[x].r)return e[x].mx>0?e[x].l-1:e[x].l;
    pushdown(x);
    if(e[x<<1|1].mx>0)return que1(x<<1);
    else return que1(x<<1|1);
}
int que2(int x,int l,int r)
{
    if(e[x].l==l&&r==e[x].r)return e[x].mn;
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return que2(x<<1,l,r);
    else if(mid<l)return que2(x<<1|1,l,r);
    else return Min(que2(x<<1,l,mid),que2(x<<1|1,mid+1,r));
}
void add2(int x,int l,int r)
{
    if(e[x].l==l&&e[x].r==r){e[x].mx++;e[x].l2++;return;}
    int mid=(e[x].l+e[x].r)>>1;
    pushdown(x);
    if(mid>=r)add2(x<<1,l,r);
    else if(mid<l)add2(x<<1|1,l,r);
    else add2(x<<1,l,mid),add2(x<<1|1,mid+1,r);
    pushup(x);
}
void add1(int x,int l,int r)
{
    if(e[x].l==l&&e[x].r==r){e[x].mn++;e[x].l1++;return;}
    int mid=(e[x].l+e[x].r)>>1;
    pushdown(x);
    if(mid>=r)add1(x<<1,l,r);
    else if(mid<l)add1(x<<1|1,l,r);
    else add1(x<<1,l,mid),add1(x<<1|1,mid+1,r);
    pushup(x);
}
int main()
{
    freopen("ni.in","r",stdin);
    freopen("ni.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&v[i]),ct[v[i]+500000]++;
    for(int i=1;i<=1e6;i++)ct[i]+=ct[i-1];
    for(int i=1;i<=n;i++)t[i]=ct[v[i]+500000]--;
    build(1,1,n);
    for(int i=1;i<=n;i++)
    {
        modify(1,t[i],v[i]);
        add2(1,t[i],n);
        if(t[i]>1)add1(1,1,t[i]-1);
        ad(t[i]);
        int tp=que1(1);
        printf("%d\n",Min(que2(1,tp+1,n),i-qu(tp)*2));
    }
}

```

#### 6.21 T3 san

##### 题目

![](/pic/50.png)

n<=50,-200<=ai<=200

##### 分析

条件等价于如果选i,j，那么需要选i,j所有路径上的点

可以想到最小割

建两张图，如果$a_i>=0$,那么连$(s,i,a_i),(i_2,t,a_i)$

否则连$(i,i_2,-a_i)$

原图中的边权值为inf

然后最小割

##### 代码

```cpp
#include<cstdio>
#include<queue>
using namespace std;
#define N 105
#define M 5210
int n,m,v[N],a,b;
int head[N],cnt,dep[N];
struct edge{int t,next,v;}ed[M];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0};head[t]=cnt;}
bool bfs(int s,int t)
{
    queue<int> tp;
    for(int i=1;i<=n*2+2;i++)dep[i]=-1;
    tp.push(s);dep[s]=1;
    while(!tp.empty())
    {
        int s=tp.front();tp.pop();
        for(int i=head[s];i;i=ed[i].next)
        if(ed[i].v&&dep[ed[i].t]==-1)
        {
            dep[ed[i].t]=dep[s]+1,tp.push(ed[i].t);
            if(ed[i].t==t)return 1;
        }
    }
    return 0;
}
int dfs(int u,int f,int t)
{
    if(u==t)return f;
    if(!f)return 0;
    int ans=0,res=f,tmp;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].v&&dep[ed[i].t]==dep[u]+1&&(tmp=dfs(ed[i].t,min(res,ed[i].v),t)))
    {
        ed[i].v-=tmp;
        ed[i^1].v+=tmp;
        ans+=tmp;
        res-=tmp;
        if(!res)return ans;
    }
    return ans;
}
int dinic(int x,int y)
{
    int ans=0;
    while(bfs(x,y))
    ans+=dfs(x,123123123,y);
    return ans;
}
int main()
{
    freopen("san.in","r",stdin);
    freopen("san.out","w",stdout);
    scanf("%d%d",&n,&m);
    int su=0;cnt=1;
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&v[i]);
        if(v[i]>=0)adde(n*2+1,i,v[i]),adde(i+n,n*2+2,v[i]),su+=v[i];
        else adde(i,i+n,-v[i]);
    }
    for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),adde(a,b,1e9),adde(a+n,b+n,1e9);
    printf("%d\n",su-dinic(n*2+1,n*2+2));
}

```

#### 6.24 T1 旅途(journey)

##### 题目

给一张带权图，对于每一个k，定义一条路径的费用为边权前k大和，求最小费用从1到n路径

n<=3000

##### 分析

首先可以枚举前k大中最短边，写一个暴力分层dijkstra，复杂度$O(n^2mlogn)$

考虑所有答案一起算

先用最短路更新所有答案

枚举最短边s，将所有边权改为$max(v-s,0)$,跑最短路，然后用$k*s+mindis$更新所有k

正确性证明：

如果最短路上边数小于k，显然不优于直接走这条路

如果最短路上非0边数大于k，显然多计算了一些值

如果等于k，显然是合法的

如果小于k，会有小于s的边计算为s，不优

复杂度$O(nmlogn)$

##### 代码

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 3003
struct edge{int t,next,v;}ed[N*2];
int head[N],cnt,n,m,s[N],a,b,c,tp1;
long long dis[N],ans[N];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
void dij()
{
    priority_queue<pair<long long,int> > tp;
    for(int i=1;i<=n;i++)dis[i]=1e17;
    dis[1]=0;
    tp.push(make_pair(0,1));
    while(!tp.empty())
    {
        pair<long long,int> s=tp.top();tp.pop();
        int a=s.second;
        for(int i=head[a];i;i=ed[i].next)
        {
            int nb=b;
            long long ds=dis[a];
            if(ed[i].v>=tp1)ds+=ed[i].v-tp1;
            if(ds<dis[ed[i].t])
            {
                dis[ed[i].t]=ds;
                tp.push(make_pair(-dis[ed[i].t],ed[i].t));
            }
        }
    }
}
int main()
{
    freopen("journey.in","r",stdin);
    freopen("journey.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++)scanf("%d%d%d",&a,&b,&c),s[i]=c,adde(a,b,c);
    sort(s+1,s+m+1);
    for(int i=0;i<=n;i++)ans[i]=1e17;
    dij();
    for(int j=1;j<=n;j++)ans[j]=min(ans[j-1],dis[n]);
    for(int i=1;i<=m;i++)
    {
        tp1=s[i];
        dij();
        for(int j=1;j<=n;j++)ans[j]=min(ans[j],dis[n]+1ll*tp1*j);
    }
    for(int j=n;j>=1;j--)printf("%lld\n",ans[j]);
}

```

#### 6.24 T2 翻折(duplicate)

##### 题目

给定字符串s，定义双倍子串为形如$ss$的字串，求长度为偶数且不为双倍子串的子串数量

n<=1e5

##### 分析

只需求出双倍子串数量

建sa，枚举长度2l，在串上取n/l个位置，分别为l,2l,3l,4l,...,kl，对相邻两个求最长公共前缀和后缀，大于等于l则表示存在一段双倍子串，具体来说，设反串lcp为rlcp，则如果$lcp+rlcp+1>=l$,那么存在开头在区间$[x-lcp,x+rlcp-l]$,长度为2l的双倍子串

对于每一个长度扫一遍所有合法区间，复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 255000
int T,n,ct;
struct suffixarray{
    int sa[N],rk[N],he[N],mn[N][18],n,a[N],b[N],m,lg[N],ct[N];
    char v[N];
    void init(int s){n=s;for(int i=1;i<=n*2;i++)sa[i]=rk[i]=he[i]=a[i]=b[i]=v[i]=ct[i]=0;}
    void SA()
    {
        m=n+255;
        for(int i=1;i<=n;i++)ct[a[i]=v[i]]++;
        for(int i=1;i<=m;i++)ct[i]+=ct[i-1];
        for(int i=n;i>=1;i--)sa[ct[a[i]]--]=i;
        for(int k=1;k<=n;k<<=1)
        {
            int cnt=0;
            for(int i=n;i>n-k;i--)b[++cnt]=i;
            for(int i=1;i<=n;i++)if(sa[i]>k)b[++cnt]=sa[i]-k;
            for(int i=1;i<=m;i++)ct[i]=0;
            for(int i=1;i<=n;i++)ct[a[i]]++;
            for(int i=1;i<=m;i++)ct[i]+=ct[i-1];
            for(int i=n;i>=1;i--)sa[ct[a[b[i]]]--]=b[i];
            for(int i=1;i<=n;i++)b[i]=a[i];
            int tp=2;a[sa[1]]=1;
            for(int i=2;i<=n;i++)a[sa[i]]=b[sa[i]]==b[sa[i-1]]&&b[sa[i]+k]==b[sa[i-1]+k]?tp-1:tp++;
            m=tp-1;
        }
        for(int i=1;i<=n;i++)rk[sa[i]]=i;
        int as=0;
        for(int i=1;i<=n;i++)
        {
            if(as>0)as--;
            if(rk[i]==1)continue;
            while(v[i+as]==v[sa[rk[i]-1]+as])as++;
            he[rk[i]]=as;if(as)as--;
        }
        for(int i=2;i<=n;i++)mn[i][0]=he[i],lg[i]=lg[i>>1]+1;
        for(int j=1;j<=17;j++)
        for(int i=2;i+(1<<j)-1<=n;i++)
        mn[i][j]=min(mn[i][j-1],mn[i+(1<<j-1)][j-1]);
    }
    int lcp(int i,int j)
    {
        if(i>j)i^=j^=i^=j;
        if(i==j)return 1e7;i++;
        int tp=lg[j-i+1];
        return min(mn[i][tp],mn[j-(1<<tp)+1][tp]);
    }
}s,rs;
struct sth{int a,b;friend bool operator <(sth a,sth b){return a.a<b.a;}};
sth r[N];
long long as=0;
int main()
{
    freopen("duplicate.in","r",stdin);
    freopen("duplicate.out","w",stdout);
    scanf("%d",&T);
    while(T--)
    {
        as=0;
        scanf("%d",&n);s.init(n);rs.init(n);
        scanf("%s",s.v+1);for(int i=1;i<=n;i++)rs.v[n-i+1]=s.v[i];
        s.SA();rs.SA();
        for(int l=1;l<=n;l++)
        {
            ct=0;
            for(int k=1;k+l<=n;k+=l)
            {
                int nxt=k+l;
                int rp=s.lcp(s.rk[k],s.rk[nxt]),lp=rs.lcp(rs.rk[n-k+1],rs.rk[n-nxt+1]);
                if(lp+rp-1>=l)r[++ct]=(sth){k-lp+1,1},r[++ct]=(sth){k+rp-l+1,-1};
            }
            sort(r+1,r+ct+1);
            int su=0;
            r[++ct]=(sth){n-l*2+1,0};
            for(int i=1;i<=ct;i++)
            {
                if(su)as-=r[i].a-r[i-1].a;
                su+=r[i].b;
            }
        }
        for(int i=2;i<=n;i+=2)as+=n-i+1;
        printf("%lld\n",as);
    }
}

```

#### 6.24 T3 迷(maze)

##### 题目

一个二分图，两边有n,m个点

给出一个n*m矩阵代表两边连接情况

额外给定p行q列

k次询问，每次询问如果用一行\列去替换矩阵中的一行\列，最后二分图完美匹配数 mod 2的值

n,m,p,q<=1000,k<=100000

##### 分析

由线性代数知识可得，完美匹配数 mod 2 =矩阵行列式 mod 2

具体可以参考行列式的式子，把-1的次方删掉

~~（虽然猜个结论就出来了~~

然后横竖分开做，因为mod 2 可以交换行，所以可以求出其它行消完后的情况，消成对角矩阵后可以$O(n)$算出加入最后一行后是否满秩

使用线段树分治，总复杂度$O(n^3logn+nq)$

加上bitset，复杂度$O((n^3logn+nq)/32)$,可以卡过去

~~如果用伴随矩阵可以把log去掉~~

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<bitset>
using namespace std;
#define N 1020
#define M 100050
struct que1{int x,y,id;friend bool operator <(que1 a,que1 b){return a.x<b.x;};};
que1 tp[M];
struct que2{int x,y,id;friend bool operator <(que2 a,que2 b){return a.x<b.x;};};
que2 tp2[M];
int n,m,k,q,a,b,c,l1,l2,as[M],t1,t2;
char s[N][N],v1[N][N],v2[N][N];
bitset<N> las[N][12];
bitset<N> v[N],s1[N],rs[N],v11[N],v21[N];
void ins(bitset<N> t)
{
    for(int i=1;i<=n;i++)
    {
        if(!t[i])continue;
        if(v[i]==0){for(int j=i+1;j<=n;j++)if(t[j])t^=v[j];v[i]=t;for(int j=1;j<i;j++)if(v[j][i])v[j]^=t;return;}
        t^=v[i];
    }
}
void solve1(int l,int r,int dep)
{
    if(l==r)
    {
        int tp1=0,fg=0;
        for(int i=1;i<=n;i++)
        if(v[i][i]==0)
        {
            if(!tp1)tp1=i;
            else fg=1;
        }
        if(fg)
        {
            while(tp[l1].x==l&&l1<=t1)l1++;
            return;
        }
        bitset<N> su;
        for(int i=1;i<=n;i++)su[i]=v[i][tp1];
        while(tp[l1].x==l&&l1<=t1)
        {
            int su1=(v11[tp[l1].y]&su).count();
            if(su1%2==0&&v11[tp[l1].y][tp1]==1)as[tp[l1].id]=1;
            if(su1%2==1&&v11[tp[l1].y][tp1]==0)as[tp[l1].id]=1;
            l1++;
        }
        return;
    }
    for(int i=1;i<=n;i++)las[i][dep]=v[i];
    int mid=(l+r)>>1;
    for(int i=mid+1;i<=r;i++)ins(s1[i]);
    solve1(l,mid,dep+1);
    for(int i=1;i<=n;i++)v[i]=las[i][dep];
    for(int i=l;i<=mid;i++)ins(s1[i]);
    solve1(mid+1,r,dep+1);
}
void solve2(int l,int r,int dep)
{
    if(l==r)
    {
        int tp1=0,fg=0;
        for(int i=1;i<=n;i++)
        if(v[i][i]==0)
        {
            if(!tp1)tp1=i;
            else fg=1;
        }
        if(fg)
        {
            while(tp2[l2].x==l&&l2<=t2)l2++;
            return;
        }
        bitset<N> su;
        for(int i=1;i<=n;i++)su[i]=v[i][tp1];
        while(tp2[l2].x==l&&l2<=t2)
        {
            int su1=(v21[tp2[l2].y]&su).count();
            if(su1%2==0&&v21[tp2[l2].y][tp1]==1)as[tp2[l2].id]=1;
            if(su1%2==1&&v21[tp2[l2].y][tp1]==0)as[tp2[l2].id]=1;
            l2++;
        }
        return;
    }
    for(int i=1;i<=n;i++)las[i][dep]=v[i];
    int mid=(l+r)>>1;
    for(int i=mid+1;i<=r;i++)ins(rs[i]);
    solve2(l,mid,dep+1);
    for(int i=1;i<=n;i++)v[i]=las[i][dep];
    for(int i=l;i<=mid;i++)ins(rs[i]);
    solve2(mid+1,r,dep+1);
}
int main()
{
    freopen("maze.in","r",stdin);
    freopen("maze.out","w",stdout);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=n;j++)
    s1[i][j]=rs[j][i]=s[i][j]-'0';
    scanf("%d%d",&m,&k);
    for(int i=1;i<=m;i++)scanf("%s",v1[i]+1);
    for(int i=1;i<=m;i++)
    for(int j=1;j<=n;j++)
    v11[i][j]=v1[i][j]-'0';
    for(int i=1;i<=k;i++)scanf("%s",v2[i]+1);
    for(int i=1;i<=k;i++)
    for(int j=1;j<=n;j++)
    v21[i][j]=v2[i][j]-'0';
    for(int i=1;i<=n;i++)ins(s1[i]);
    int fg=1;
    for(int i=1;i<=n;i++)if(v[i][i]==0)fg=0;
    for(int i=1;i<=n;i++)v[i].reset();
    scanf("%d",&q);
    for(int i=1;i<=q;i++)
    {
        scanf("%d%d%d",&a,&b,&c);
        if(a==0)tp[++t1]=(que1){b,c,i};
        else tp2[++t2]=(que2){b,c,i};
    }
    sort(tp+1,tp+t1+1);
    sort(tp2+1,tp2+t2+1);
    l1=l2=1;
    solve1(1,n,1);
    for(int i=1;i<=n;i++)v[i].reset();
    solve2(1,n,1);
    printf("%d\n",fg);
    for(int i=1;i<=q;i++)printf("%d\n",as[i]);
}

```

#### 6.25 T1 梦批糼 (dream)

##### 题意

```
现在给定一个长宽高分别为n,m,k 的长方体，其中每个整点从(1,1,1) 开始编号。
定义子长方体是一个所有点都在这个长方体范围内且不退化的长方体。显然，这个长方
体有1/8n(n+1)m(m+1)k(k+1)个子长方体。
神树大人每次会随机选择一个子长方体并选中这个子长方体里的所有点。所有子
长方体被选择的概率相等。有些点被称为障碍，这些点不能被选中。其他点可以被选中
无数次。神树大人会随机选择w次。每个点有个权值，求最后没有一个障碍被选中时
至少被选中一次的点的期望权值和
n,m,k<=60,2s

```

##### 分析

每个点的贡献独立

如果可以算出一个点被覆盖且子长方体合法的方案数，那么它被覆盖且合法的概率为(1-((不覆盖障碍方案数-点被覆盖且子长方体合法的方案数)/不覆盖障碍方案数)^k)*(子长方体合法的方案数/总方案数)^k

这个式子表示先计算每一次都不覆盖它的方案，再用1减

因为覆盖障碍贡献为0，只计算不覆盖障碍的

所以只需要计算一个点被不覆盖障碍的子长方体覆盖了多少次

显然暴力$O(n^9)$,暴力+差分$O(n^6)$

考虑记录差分时只关心每个点作为合法长方体的8个角的次数

一个想法是先枚举两维，一维上可以很方便统计它作为左端点和右端点的次数

复杂度$O(n^5)$

以上一个算法为基础，枚举一维，问题变为二维上一个点作为矩形四个角的方案数

只需求出每一个位置向上，向下有多少个连续的非障碍点，然后同一排间扫描线+单调栈即可求出

非障碍点个数可以做到总复杂度$O(n^2)$,总复杂度$O(n^4)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 65
#define mod 998244353
int s[N][N][N],v[N][N][N],as[N][N][N],su,f[N][N],st[N],rb,tp[N],n,m,k,w,ct[N][N][4],ans;
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int getnxt(int x,int y)
{
    int as=1;
    if(!f[x][y])return 0;
    while(f[x][y+1]&&y+1<=k)y++,as++;
    return as;
}
void dp()
{
    for(int i=1;i<=m;i++)tp[i]=0;
    for(int i=1;i<=k;i++)
    {
        for(int j=1;j<=m;j++)
        if(tp[j]==0)tp[j]=getnxt(j,i);
        else tp[j]--;
        rb=0;st[0]=0;
        int su=0;
        for(int j=1;j<=m;j++)
        {
            while(rb&&tp[st[rb]]>=tp[j])
            {
                su-=(st[rb]-st[rb-1])*tp[st[rb]];
                rb--;
            }
            st[++rb]=j;
            su+=tp[j]*(j-st[rb-1]);
            ct[j][i][1]=su;
        }
        rb=0;su=0;st[0]=m+1;
        for(int j=m;j>=1;j--)
        {
            while(rb&&tp[st[rb]]>=tp[j])
            {
                su-=(-st[rb]+st[rb-1])*tp[st[rb]];
                rb--;
            }
            st[++rb]=j;
            su+=tp[j]*(st[rb-1]-j);
            ct[j][i][0]=su;
        }
    }
    for(int i=1;i<=m;i++)tp[i]=0;
    for(int i=1;i<=k/2;i++)
    for(int j=1;j<=m;j++)swap(f[j][i],f[j][k-i+1]);
    for(int i=1;i<=k;i++)
    {
        for(int j=1;j<=m;j++)
        if(tp[j]==0)tp[j]=getnxt(j,i);
        else tp[j]--;
        rb=0;st[0]=0;
        int su=0;
        for(int j=1;j<=m;j++)
        {
            while(rb&&tp[st[rb]]>=tp[j])
            {
                su-=(st[rb]-st[rb-1])*tp[st[rb]];
                rb--;
            }
            st[++rb]=j;
            su+=tp[j]*(j-st[rb-1]);
            ct[j][k-i+1][3]=su;
        }
        rb=0;su=0;st[0]=m+1;
        for(int j=m;j>=1;j--)
        {
            while(rb&&tp[st[rb]]>=tp[j])
            {
                su-=(-st[rb]+st[rb-1])*tp[st[rb]];
                rb--;
            }
            st[++rb]=j;
            su+=tp[j]*(st[rb-1]-j);
            ct[j][k-i+1][2]=su;
        }
    }
    for(int i=1;i<=k/2;i++)
    for(int j=1;j<=m;j++)swap(f[j][i],f[j][k-i+1]);
}
int main()
{
    freopen("dream.in","r",stdin);
    freopen("dream.out","w",stdout);
    scanf("%d%d%d%d",&n,&m,&k,&w);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    for(int l=1;l<=k;l++)
    scanf("%d",&s[i][j][l]);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    for(int l=1;l<=k;l++)
    scanf("%d",&v[i][j][l]);
    for(int l=1;l<=n;l++)
    {
        for(int i=1;i<=m;i++)
        for(int j=1;j<=k;j++)
        f[i][j]=1;
        for(int r=l;r<=n;r++)
        {
            for(int i=1;i<=m;i++)
            for(int j=1;j<=k;j++)
            f[i][j]&=s[r][i][j];
            dp();
            for(int i=1;i<=m;i++)
            for(int j=1;j<=k;j++)
            as[l][i][j]+=ct[i][j][0],as[l][i+1][j]-=ct[i][j][1],as[l][i][j+1]-=ct[i][j][2],as[l][i+1][j+1]+=ct[i][j][3],
            as[r+1][i][j]-=ct[i][j][0],as[r+1][i+1][j]+=ct[i][j][1],as[r+1][i][j+1]+=ct[i][j][2],as[r+1][i+1][j+1]-=ct[i][j][3],
            su=(su+ct[i][j][0])%mod;
        }
    }
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    for(int l=1;l<=k;l++)
    as[i][j][l]=(((long long)as[i][j][l]+as[i-1][j][l]+as[i][j-1][l]+as[i][j][l-1]-as[i-1][j-1][l]-as[i-1][j][l-1]-as[i][j-1][l-1]+as[i-1][j-1][l-1])%mod+mod)%mod;
    int su2=1ll*n*m*k*(n+1)*(m+1)*(k+1)%mod*873463809%mod;
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    for(int l=1;l<=k;l++)
    ans=(ans+1ll*(pw(1ll*su*pw(su2,mod-2)%mod,w)-pw(1ll*(su-as[i][j][l])*pw(su2,mod-2)%mod,w))*v[i][j][l])%mod;
    printf("%d\n",(ans+mod)%mod);
}

```



#### 6.25 T2 等你哈苏德(wait)

##### 题目

```
Joker 有m个黑白区间[li,ri],有些区间已经被指定了颜色，有些却没有。你要指定
这些未染色区间的颜色，使得数轴上对于每个点，覆盖他的黑区间个数和白区间个数差
的绝对值小于等于1
m<=3e4,1<=li<=ri<=1e9

```

##### 分析

首先考虑没有边已经被染色的情况

把区间看成边，对于每一个端点记录两边黑白差的差，可以发现区间染色相当于一个值+1，一个值-1，因此可以看成给边定向

将奇数度数点排序后相邻两个连起来，这样相邻两个一定是一个+1一个-1，这样前缀和以后都在-1到1之间

然后问题变为求一个边染色使得每个点出度=入度，跑网络流即可

如果有边被染色，问题变为一个上下界网络流，使用同样方法

~~然后WA52~~

因为可能出现不连通+两条重边染相同颜色这种情况，连相邻两个奇数时把中间的偶数点也连上即可

复杂度$O(m\sqrt m)$(单位流量dinic)

##### 代码

```cpp
#include<cstdio>
#include<queue>
#include<algorithm>
using namespace std;
#define N 60500
#define M 290000
int n,m,v[N],a,b,in[N],s[N][3],q[N],ct,lin[N],rin[N],as[N];
int head[N],cnt,dep[N];
struct edge{int t,next,v,id;}ed[M];
void adde(int f,int t,int v,int id){ed[++cnt]=(edge){t,head[f],v,id};head[f]=cnt;ed[++cnt]=(edge){f,head[t],0,0};head[t]=cnt;}
bool bfs(int s,int t)
{
    queue<int> tp;
    for(int i=1;i<=ct+2;i++)dep[i]=-1;
    tp.push(s);dep[s]=1;
    while(!tp.empty())
    {
        int s=tp.front();tp.pop();
        for(int i=head[s];i;i=ed[i].next)
        if(ed[i].v&&dep[ed[i].t]==-1)
        {
            dep[ed[i].t]=dep[s]+1,tp.push(ed[i].t);
            if(ed[i].t==t)return 1;
        }
    }
    return 0;
}
int dfs(int u,int f,int t)
{
    if(u==t)return f;
    if(!f)return 0;
    int ans=0,res=f,tmp;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].v&&dep[ed[i].t]==dep[u]+1&&(tmp=dfs(ed[i].t,min(res,ed[i].v),t)))
    {
        ed[i].v-=tmp;
        ed[i^1].v+=tmp;
        ans+=tmp;
        res-=tmp;
        if(!res)return ans;
    }
    return ans;
}
int dinic(int x,int y)
{
    int ans=0;
    while(bfs(x,y))
    ans+=dfs(x,123123123,y);
    return ans;
}
int main()
{
    freopen("wait.in","r",stdin);
    freopen("wait.out","w",stdout);
    scanf("%d%d",&n,&m);cnt=1;
    for(int i=1;i<=n;i++)scanf("%d%d%d",&s[i][0],&s[i][1],&s[i][2]),s[i][1]++,q[++ct]=s[i][0],q[++ct]=s[i][1];
    sort(q+1,q+ct+1);
    ct=unique(q+1,q+ct+1)-q-1;
    for(int i=1;i<=n;i++)
    {
        s[i][0]=lower_bound(q+1,q+ct+1,s[i][0])-q;
        s[i][1]=lower_bound(q+1,q+ct+1,s[i][1])-q;
        if(s[i][2]==0)
        {
            as[i]=0;
            lin[s[i][1]]++;rin[s[i][0]]++;
        }
        else if(s[i][2]==1)
        {
            as[i]=1;
            lin[s[i][0]]++;rin[s[i][1]]++;
        }
        else
        {
            lin[s[i][1]]++;rin[s[i][0]]++;
            adde(s[i][0],s[i][1],1,i);
        }
        in[s[i][0]]++;in[s[i][1]]++;
    }
    int las=0;
    for(int i=1;i<=ct;i++)
    if(in[i]&1)
    {
        if(las)
        {
            lin[i]++;rin[las]++;
            adde(las,i,1,0);
            las=0;
        }
        else las=i;
    }
    else
    if(las)lin[i]++,rin[las]++,adde(las,i,1,0),las=i;
    int su=0;
    for(int i=1;i<=ct;i++)
    {
        int tp=(rin[i]-lin[i])/2;
        if(tp>=0)adde(ct+1,i,tp,0),su+=tp;
        else adde(i,ct+2,-tp,0);
    }
    int ans=dinic(ct+1,ct+2);
    if(ans!=su){printf("-1\n");return 0;}
    for(int i=1;i<=cnt;i++)
    if(ed[i].id)as[ed[i].id]=!ed[i].v;
    for(int i=1;i<=n;i++)printf("%d ",as[i]);
}

```

#### 6.25 T3 喜欢最最痛(love)

##### 题目

```
神树大人种了一棵有边权的树，由于这是神树大人种的树，所以这棵树被命名为神神树。
神神树的边权为正整数。神树大人命令龚诗锋从 1 号点开始走一个路径并最终回到1号点，且这条路径经过了所有的边。一条路径的代价就是它经过的边的边权之和。
龚诗锋可以加若干条额外边，第i条加的额外边的边权为正整数Ai。注意,龚诗锋不一定要经过所有的额外边。
由于龚诗锋喜欢最最痛，所以对于所有的0<=K<=m，你需要输出允许加K条额外边的最小路径代价。
n<=1e5,5s,512MB

```

##### 分析

走i条额外边相当于在原树上删去i条边不相交的路径

可以看出(cai)对于每一个k，走i条额外边的代价是一个凸函数，三分即可

区间前k大可以主席树(对于这道题可以直接树状数组)

对于求i条边不相交的路径和最大值，考虑带撤销贪心，每次求带权直径，然后路径边权*-1

使用lct维护动态dp，分别维护取反前和取反后的矩阵，复杂度大概是$O(3^4nlogn)$（也许只有我写的这么大常数

##### 代码

这里只有一份树链剖分2个logTLEMLE还WA72.5的代码

(10kb)

```cpp
#include<cstdio>
#include<algorithm>
#include<queue>
using namespace std;
#define N 200050
struct sth{
    long long a,b,c;
    friend sth operator +(sth a,sth b){if(a.a>=b.a)return a;return b;}
    friend sth operator *(sth a,sth b){if(!a.b)a.b=b.c;if(!b.b)b.b=a.c;return (sth){a.a+b.a,max(a.b,b.b),min(a.b,b.b)};}
    friend bool operator <(sth a,sth b){return a.a<b.a;}
    friend bool operator ==(sth a,sth b){return a.a==b.a&&min(a.b,a.c)==min(b.b,b.c)&&max(a.b,a.c)==max(b.b,b.c);}
}inf;
struct dp{
    sth f[3][3];
    friend dp operator *(dp a,dp b)
    {
        dp c;
        for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)c.f[i][j].a=-1e17,c.f[i][j].b=0,c.f[i][j].c=0;
        for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        if(a.f[i][j].a>=-1e16)
        for(int k=0;k<3;k++)
        c.f[i][k]=(c.f[i][k]+a.f[i][j]*b.f[j][k]);
        return c;
    }
};
long long mxvl[N],mx[N],lasmx[N];
int n,m,tp[N],sz[N],son[N],fr[N],mxv[N],edv[N],id[N],tid[N],head[N],cnt,ct,dep[N],f[N],edv2[N],en[N],a,b,c,isdel[N],fg[N];
struct edge{int t,next,v;}ed[N*2];
void adde(int f,int t,int v){ed[++cnt]=(edge){t,head[f],v};head[f]=cnt;ed[++cnt]=(edge){f,head[t],v};head[t]=cnt;}
priority_queue<pair<long long,int> > sp[N],sp2[N];
priority_queue<sth> fs,fs2;
sth as1[N],as2[N];
struct node{int l,r,lz;dp s,s2;}e[N*4];
void pushup(int x){e[x].s=e[x<<1|1].s*e[x<<1].s;e[x].s2=e[x<<1|1].s2*e[x<<1].s2;}
void pushdown(int x){if(e[x].lz)e[x<<1].lz^=1,e[x<<1|1].lz^=1,swap(e[x<<1].s,e[x<<1].s2),swap(e[x<<1|1].s,e[x<<1|1].s2),e[x].lz=0;}
void build(int x,int l,int r)
{
    e[x].l=l;e[x].r=r;
    if(l==r)
    {
        l=tid[l];
        e[x].s.f[0][1]=inf;
        e[x].s.f[0][2]=inf;
        e[x].s.f[1][2]=inf;
        e[x].s.f[2][0]=inf;
        e[x].s.f[1][0]=(sth){mxvl[l]+edv[l],mxv[l],0};
        e[x].s.f[2][1]=(sth){mxvl[l],mxv[l],0};
        e[x].s.f[1][1]=(sth){edv[l],0,0};
        e[x].s2.f[0][1]=inf;
        e[x].s2.f[0][2]=inf;
        e[x].s2.f[1][2]=inf;
        e[x].s2.f[2][0]=inf;
        e[x].s2.f[1][0]=(sth){mxvl[l]-edv[l],mxv[l],0};
        e[x].s2.f[2][1]=(sth){mxvl[l],mxv[l],0};
        e[x].s2.f[1][1]=(sth){-edv[l],0,0};
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
        e[x].s.f[0][1]=inf;
        e[x].s.f[0][2]=inf;
        e[x].s.f[1][2]=inf;
        e[x].s.f[2][0]=inf;
        e[x].s.f[1][0]=(sth){mxvl[tid[s]]+edv[tid[s]],mxv[tid[s]],0};
        e[x].s.f[2][1]=(sth){mxvl[tid[s]],mxv[tid[s]],0};
        e[x].s.f[1][1]=(sth){edv[tid[s]],0,0};
        e[x].s2.f[0][1]=inf;
        e[x].s2.f[0][2]=inf;
        e[x].s2.f[1][2]=inf;
        e[x].s2.f[2][0]=inf;
        e[x].s2.f[1][0]=(sth){mxvl[tid[s]]-edv[tid[s]],mxv[tid[s]],0};
        e[x].s2.f[2][1]=(sth){mxvl[tid[s]],mxv[tid[s]],0};
        e[x].s2.f[1][1]=(sth){-edv[tid[s]],0,0};
        if(e[x].lz)swap(e[x].s,e[x].s2);
        return;
    }
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=s)modify(x<<1,s);
    else modify(x<<1|1,s);
    pushup(x);
}
dp query(int x,int l,int r)
{
    if(l>r)
    {
        sth st;st.a=st.b=st.c=0;
        dp fuc;
        fuc.f[0][0]=fuc.f[1][1]=fuc.f[2][2]=st;
        fuc.f[1][0]=fuc.f[0][1]=fuc.f[0][2]=fuc.f[2][0]=fuc.f[1][2]=fuc.f[2][1]=inf;
        return fuc;
    }
    if(e[x].l==l&&e[x].r==r)return e[x].s;
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)return query(x<<1,l,r);
    else if(mid<l)return query(x<<1|1,l,r);
    else return query(x<<1|1,mid+1,r)*query(x<<1,l,mid);
}
void modify(int x,int l,int r)
{
    if(l>r)return;
    if(e[x].l==l&&e[x].r==r){e[x].lz^=1;swap(e[x].s,e[x].s2);return;}
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=r)modify(x<<1,l,r);
    else if(mid<l)modify(x<<1|1,l,r);
    else modify(x<<1,l,mid),modify(x<<1|1,mid+1,r);
    pushup(x);
}
int qu2(int x,int s)
{
    if(e[x].l==e[x].r)return e[x].lz==1?-1:1;
    pushdown(x);
    int mid=(e[x].l+e[x].r)>>1;
    if(mid>=s)return qu2(x<<1,s);
    else return qu2(x<<1|1,s);
}
int qu2(int s)
{
    if(son[f[s]]==s)return qu2(1,id[f[s]]);
    else return fg[s];
}
sth getdp(int x)
{
    dp tp1=query(1,id[x],id[en[x]]-1);
    sth a=(sth){0,en[x],en[x]},b=a,c=(sth){0,0,0};
    return (a*tp1.f[0][1]+b*tp1.f[1][1]+c*tp1.f[2][1]);
}
sth getfdp(int x)
{
    dp tp1=query(1,id[x],id[en[x]]-1);
    sth a=(sth){0,en[x],en[x]},b=a,c=(sth){0,0,0};
    return (a*tp1.f[0][0]+b*tp1.f[1][0]+c*tp1.f[2][0]);
}
sth getvdp(int x)
{
    while(sp[x].size()&&sp2[x].size()&&sp[x].top().second==sp2[x].top().second)sp[x].pop(),sp2[x].pop();
    pair<long long,int> as=sp[x].top();sp[x].pop();
    while(sp[x].size()&&sp2[x].size()&&sp[x].top().second==sp2[x].top().second)sp[x].pop(),sp2[x].pop();
    if(!sp[x].size()){sp[x].push(as);return (sth){0,x,x};}
    pair<long long,int> as2=sp[x].top();sp[x].push(as);
    return (sth){as.first+as2.first,as.second,as2.second};
}
void pushupq(int x)
{
    if(!x)return;
    fs2.push(as1[x]);
    as1[x]=getvdp(x);
    fs.push(as1[x]);
}
void pushupt(int x)
{
    if(!x)return;
    fs2.push(as2[x]);
    as2[x]=getfdp(x);
    fs.push(as2[x]);
}
void dfs(int u,int fa)
{
    sz[u]=1;dep[u]=dep[fa]+1;f[u]=fa;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)
    {
        dfs(ed[i].t,u);edv2[ed[i].t]=ed[i].v;sz[u]+=sz[ed[i].t];
        if(sz[ed[i].t]>=sz[son[u]])son[u]=ed[i].t;
    }
}
void dfs2(int u,int v,int fa)
{
    mxvl[u]=-1e17;mx[u]=-1e17;
    sp[u].push(make_pair((long long)-1e17,0));
    tp[u]=v;id[u]=++ct;tid[ct]=u;en[u]=-1;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t==son[u])
    {
        edv[u]=ed[i].v;
        dfs2(ed[i].t,v,u);
        mx[u]=mx[ed[i].t]+ed[i].v;
        fr[u]=fr[ed[i].t];
        en[u]=en[ed[i].t];
    }
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&ed[i].t!=son[u])
    {
        dfs2(ed[i].t,ed[i].t,u);
        if(mx[ed[i].t]+ed[i].v>mxvl[u])mxvl[u]=mx[ed[i].t]+ed[i].v,mxv[u]=fr[ed[i].t];
        if(mx[ed[i].t]+ed[i].v>mx[u])mx[u]=mx[ed[i].t]+ed[i].v,fr[u]=fr[ed[i].t];
        sp[u].push(make_pair(mx[ed[i].t]+ed[i].v,fr[ed[i].t]));
    }
    if(mx[u]<0)mx[u]=0,fr[u]=u;
    sp[u].push(make_pair(0,u));
    as1[u]=getvdp(u);
    fs.push(as1[u]);
    if(mxvl[u]<0)mxvl[u]=0,mxv[u]=u;
    if(en[u]==-1)en[u]=u;
}
void rev(int a,int b)
{
    while(tp[a]!=tp[b])
    {
        if(dep[tp[a]]<dep[tp[b]])a^=b^=a^=b;
        sp2[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
        modify(1,id[a]);
        modify(1,id[tp[a]],id[a]-1);
        fg[tp[a]]*=-1;
        pushupt(tp[a]);
        sth fu=getdp(tp[a]);
        lasmx[tp[a]]=mx[tp[a]];
        mx[tp[a]]=fu.a,fr[tp[a]]=fu.b;
        sp[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
        while(!sp[f[tp[a]]].empty()&&!sp2[f[tp[a]]].empty()&&sp[f[tp[a]]].top().second==sp2[f[tp[a]]].top().second&&sp[f[tp[a]]].top().first==sp2[f[tp[a]]].top().first)
        sp[f[tp[a]]].pop(),sp2[f[tp[a]]].pop();
        mxvl[f[tp[a]]]=sp[f[tp[a]]].top().first,mxv[f[tp[a]]]=sp[f[tp[a]]].top().second;
        pushupq(f[tp[a]]);a=f[tp[a]];
    }
    if(id[a]<id[b])a^=b^=a^=b;
    sp2[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
    modify(1,id[a]);modify(1,id[b]);
    modify(1,id[b],id[a]-1);
    sth fu=getdp(tp[a]);pushupt(tp[a]);
    lasmx[tp[a]]=mx[tp[a]];mx[tp[a]]=fu.a,fr[tp[a]]=fu.b;
    sp[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
    while(!sp[f[tp[a]]].empty()&&!sp2[f[tp[a]]].empty()&&sp[f[tp[a]]].top().second==sp2[f[tp[a]]].top().second&&sp[f[tp[a]]].top().first==sp2[f[tp[a]]].top().first)
    sp[f[tp[a]]].pop(),sp2[f[tp[a]]].pop();
    mxvl[f[tp[a]]]=sp[f[tp[a]]].top().first,mxv[f[tp[a]]]=sp[f[tp[a]]].top().second;
    pushupq(f[tp[a]]);a=f[tp[a]];
    while(a)
    {
        sp2[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
        modify(1,id[a]);
        sth fu=getdp(tp[a]);pushupt(tp[a]);
        lasmx[tp[a]]=mx[tp[a]];mx[tp[a]]=fu.a,fr[tp[a]]=fu.b;
        sp[f[tp[a]]].push(make_pair(qu2(tp[a])*edv2[tp[a]]+mx[tp[a]],fr[tp[a]]));
        while(!sp[f[tp[a]]].empty()&&!sp2[f[tp[a]]].empty()&&sp[f[tp[a]]].top().second==sp2[f[tp[a]]].top().second&&sp[f[tp[a]]].top().first==sp2[f[tp[a]]].top().first)
        sp[f[tp[a]]].pop(),sp2[f[tp[a]]].pop();
        mxvl[f[tp[a]]]=sp[f[tp[a]]].top().first,mxv[f[tp[a]]]=sp[f[tp[a]]].top().second;
        pushupq(f[tp[a]]);a=f[tp[a]];
    }
}
sth qu(int x)
{
    dp tp1=query(1,id[tp[x]],id[x]-1);
    sth a=(sth){0,x,x},b=a,c=(sth){0,0,0};
    return a*tp1.f[0][0]+b*tp1.f[1][0]+c*tp1.f[2][0];
}
long long v[N],as[N];
struct Splay{
    int ch[N][2],sz[N],rt,fa[N],ct;
    long long vl[N],v[N];
    void init(int s){rt=ct=1;vl[1]=v[1]=s;}
    void pushup(int x){sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+1,vl[x]=vl[ch[x][0]]+vl[ch[x][1]]+v[x];}
    void rotate(int x){int f=fa[x],g=fa[f],tmp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tmp]=ch[x][!tmp];fa[ch[x][!tmp]]=f;ch[x][!tmp]=f;fa[f]=x;pushup(f);pushup(x);}
    void splay(int x){while(fa[x]){int f=fa[x],g=fa[f];if(g)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);rotate(x);}rt=x;}
    void insert(int x,int s){bool tp=v[x]<=s;if(!ch[x][tp]){ch[x][tp]=++ct;vl[ct]=v[ct]=s;fa[ct]=x;splay(ct);return;}insert(ch[x][tp],s);}
    long long query(int x,int k){if(!k){splay(x);return 0;}if(sz[ch[x][0]]>=k)return query(ch[x][0],k);if(sz[ch[x][0]]+1<k)return v[x]+vl[ch[x][0]]+query(ch[x][1],k-sz[ch[x][0]]-1);long long ans=v[x]+vl[ch[x][0]];splay(x);return ans;}
}st;
long long getans(int k){return as[k]+st.query(st.rt,k);}
int main()
{
    freopen("love.in","r",stdin);
    freopen("love.out","w",stdout);
    scanf("%d%d",&n,&m);inf.a=-1e17;inf.b=inf.c=0;
    sp[0].push(make_pair(0ll,1));
    for(int i=1;i<=n;i++)fg[i]=1;
    for(int i=1;i<n;i++)scanf("%d%d%d",&a,&b,&c),adde(a,b,c),as[0]+=c*2;
    dfs(1,0);dfs2(1,1,0);build(1,1,n);
    for(int i=1;i<=n;i++)lasmx[i]=mx[i];
    for(int i=1;i<=n;i++)if(tp[i]==i)as2[i]=getfdp(i),fs.push(as2[i]);
    for(int i=1;i<=m;i++)scanf("%d",&v[i]);
    for(int i=1;i<=m;i++)
    {
        sth fu=qu(en[1]);
        while(fs.size()&&fs2.size()&&fs.top()==fs2.top())
        fs.pop(),fs2.pop();
        if(fu<fs.top())fu=fs.top();
        as[i]=as[i-1]-fu.a;
        rev(fu.b,fu.c);
    }
    for(int i=0;i<=m;i++)
    {
        int lb=0,rb=i;
        while(lb<rb)
        {
            int mid1=(lb+rb)>>1,mid2=mid1+1;
            if(getans(mid1)<getans(mid2))rb=mid2-1;
            else lb=mid1+1;
        }
        printf("%lld ",getans(lb));
        if(i)st.insert(st.rt,v[i+1]);
        else st.init(v[1]);
    }
}

```

#### 6.28 T1 启程的日子(bitbit)

##### 题目

给一个01矩阵,每一次操作可以选一个联通的区域，将其+1/-1，求最少操作次数使得矩阵变为全零并输出方案

n,m<=500

##### 分析

对于答案是1的情况很容易判断

答案是2的情况有两种情况

1.删两个连通块

2.补一个连通块，删一个大连通块

可以发现情况2包含情况1

枚举0连通块然后算，复杂度$O(nm)$

对于n,m至少有一个为1的情况，答案显然是1段数

否则，考虑以下构造

第一个矩阵系数为1，包含第2~n-1行的奇数列以及第一行

第二个矩阵系数为1，包含第2~n-1行的偶数列以及最后一行

第三个矩阵系数为-1，包含第2~n-1行

对于第2~n-1行中的1，在第1或第2个矩阵中加

对于第一行和最后一行的0，在第3个矩阵中减

复杂度$O(nm)$

##### 代码

特判题（确信

```cpp
#include<cstdio>
#include<queue>
using namespace std;
#define N 505
#define M 250050
int id[N][N],fa[M],ok[M],f[4][2]={-1,0,1,0,0,1,0,-1},vis[N][N],ct,n,m,isrev,f1[N][N],f2[N][N],f3[N][N];
char mp[N][N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
queue<int> tp,tp2;
void dfs1(int x,int y)
{
    if(mp[x][y]=='1')ct++;
    vis[x][y]=1;
    for(int i=0;i<4;i++)
    {
        int nx=x+f[i][0],ny=y+f[i][1];
        if(nx<1||nx>n||ny<1||ny>m||vis[nx][ny]||mp[nx][ny]=='0')continue;
        dfs1(nx,ny);
    }
}
void dfs2(int x,int y)
{
    vis[x][y]=1;
    for(int i=0;i<4;i++)
    {
        int nx=x+f[i][0],ny=y+f[i][1];
        if(nx<1||nx>n||ny<1||ny>m||vis[nx][ny])continue;
        if(mp[nx][ny]=='0')ok[finds(id[nx][ny])]=1;
        else dfs2(nx,ny);
    }
}
int main()
{
    freopen("bitbit.in","r",stdin);
    freopen("bitbit.out","w",stdout);
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)scanf("%s",mp[i]+1);
    if(n>m)n^=m^=n^=m,isrev=1;
    if(isrev)
    {
        for(int i=1;i<=m;i++)
        for(int j=1;j<i;j++)
        swap(mp[i][j],mp[j][i]);
    }
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    if(mp[i][j]=='1')ct--;
    if(ct==0){printf("0\n");return 0;}
    int fg=0;
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    if(mp[i][j]=='1'&&!fg)dfs1(i,j),fg=1;
    if(ct==0)
    {
        printf("1\n+\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",mp[j][i]);
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",mp[i][j]);
        return 0;
    }
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    vis[i][j]=0;
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    id[i][j]=i*m-m+j,fa[id[i][j]]=id[i][j];
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    {
        if(mp[i][j]==mp[i][j-1]&&mp[i][j]=='0')fa[finds(id[i][j])]=finds(id[i][j-1]);
        if(mp[i][j]==mp[i-1][j]&&mp[i][j]=='0')fa[finds(id[i][j])]=finds(id[i-1][j]);
    }
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    if(mp[i][j]=='0'&&fa[id[i][j]]==id[i][j])
    tp.push(id[i][j]);
    for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
    if(mp[i][j]=='1'&&!vis[i][j])
    {
        dfs2(i,j);
        while(!tp.empty())
        {
            int st=tp.front();tp.pop();
            if(ok[st])ok[st]=0,tp2.push(st);
        }
        while(!tp2.empty())tp.push(tp2.front()),tp2.pop();
    }
    if(tp.size())
    {
        printf("2\n+\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",(mp[j][i]=='1'||finds(id[j][i])==tp.front())?'1':'0');
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",(mp[i][j]=='1'||finds(id[i][j])==tp.front())?'1':'0');
        printf("-\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",finds(id[j][i])==tp.front()?'1':'0');
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",finds(id[i][j])==tp.front()?'1':'0');
        return 0;
    }
    if(n>=2&&m>=3)
    {
        for(int i=1;i<=n;i+=2)
        for(int j=1;j<m;j++)
        f1[i][j]=1;
        for(int i=2;i<=n;i+=2)
        for(int j=2;j<=m;j++)
        f2[i][j]=1;
        for(int i=1;i<=n;i++)f1[i][1]=f2[i][m]=1;
        for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        f3[i][j]=1;
        for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            if((j==1||j==m)&&mp[i][j]=='1')f3[i][j]=0;
            else if(mp[i][j]=='1')
            if(i&1)f2[i][j]=1;
            else f1[i][j]=1;
        }
        printf("3\n+\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",f1[j][i]+'0');
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",f1[i][j]+'0');
        printf("+\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",f2[j][i]+'0');
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",f2[i][j]+'0');
        printf("-\n");
        if(isrev)
        for(int i=1;i<=m;i++,printf("\n"))
        for(int j=1;j<=n;j++)
        printf("%c",f3[j][i]+'0');
        else
        for(int i=1;i<=n;i++,printf("\n"))
        for(int j=1;j<=m;j++)
        printf("%c",f3[i][j]+'0');
        return 0;
    }
    else
    {
        int ans=0,st=1;
        for(int i=1;i<=m;i++)if(mp[1][i]=='1'&&mp[1][i+1]!='1')ans++;
        printf("%d\n",ans);
        while(ans)
        {
            printf("+\n");
            int lb,rb;
            while(mp[1][st]!='1')st++;
            lb=st;
            while(mp[1][st+1]=='1')st++;
            rb=st;
            if(isrev)
            for(int i=1;i<=m;i++,printf("\n"))
            printf("%c",(i>=lb&&i<=rb)?'1':'0');
            else
            {
                for(int i=1;i<=m;i++)
                printf("%c",(i>=lb&&i<=rb)?'1':'0');
                printf("\n");
            }
            st++;ans--;
        }
        return 0;
    }
}

```

#### 6.29 T1 智慧树(tree)

##### 题目

给一个带点权的树，对于每一个i求权值和 mod m = i 的连通块个数 mod 950009857

n<=8000,m<=60000,m|mod-1,5s,**32MB**

90% m=2^k

##### 分析

先考虑m=2^k

显然可以fft维护以每一个点为根的连通块的方案数，复杂度$O(nmlogm)$

因为fft为循环卷积，所以直接维护点值，x^k的点值可以根据定义直接预处理单位根$O(m)$，可以做到$O(nm)$

然后空间开不下

考虑树剖，对于重链直接父亲继承儿子的数组，因为路径上最多有logn条重链，所以空间可以$O(mlogn)$

现在考虑m不是2次幂的情况

因为IDFT相当于$A_i=\sum_{j=0}^{m-1}a_j*\omega_m^{-ij}$

-ij=((i-j)^2-i^2-j^2)/2

所以$A_i=\sum_{j=1}^{m-1}a_j*\omega_{2m}^{(i-j)^2}*\omega_{2m}^{-i^2}*\omega_{2m}^{-j^2}$

$A_i=\omega_{2m}^{-i^2}*\sum_{j=1}^{m-1}a_j*\omega_{2m}^{(i-j)^2}*\omega_{2m}^{-j^2}$

然后可以一次fft解决问题

这个算法叫做bluestein算法

时间复杂度$O(nm)$，空间复杂度$O(mlogn)$,需要卡常

##### 代码

```cpp
#include<cstdio>
using namespace std;
#define N 60050
#define M 263000
#define mod 950009857
int st[21][N],n,m,head[N],cnt,id[N],sz[N],son[N],v[N],as[N],a[M],b[M],c[M],ntt[M],rev[M],s,t,dwg[N];
struct edge{int t,next;}ed[N];
void adde(int f,int t){ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dfs1(int u,int fa)
{
    sz[u]=1;
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa)
    {
        dfs1(ed[i].t,u);
        sz[u]+=sz[ed[i].t];
        if(sz[ed[i].t]>sz[son[u]])son[u]=ed[i].t;
    }
}
void dfs2(int u,int fa,int id)
{
    if(son[u])
    {
        dfs2(son[u],u,id);
        for(int i=0,j=0;i<m;i++,j=j+v[u]-(j+v[u]>=m?m:0))
        st[id][i]=1ll*(st[id][i]+1)*dwg[j]%mod;
    }
    else
    for(int i=0,j=0;i<m;i++,j=j+v[u]-(j+v[u]>=m?m:0))
    st[id][i]=dwg[j];
    for(int i=head[u];i;i=ed[i].next)
    if(ed[i].t!=fa&&ed[i].t!=son[u])
    {
        dfs2(ed[i].t,u,id+1);
        for(int j=0;j<m;j++)st[id][j]=1ll*st[id][j]*(st[id+1][j]+1)%mod;
    }
    for(int j=0;j<m;j++)as[j]=(as[j]+st[id][j])%mod;
}
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+((i&1)*(s>>1)),ntt[rev[i]]=a[i];
    for(int l=2;l<=s;l<<=1)
    {
        int s1=pw(7,(mod-1)/l);
        if(t==-1)s1=pw(s1,mod-2);
        for(int j=0;j<s;j+=l)
        for(int k=j,st=1;k<j+(l>>1);k++,st=1ll*st*s1%mod)
        {
            int s1=ntt[k],s2=1ll*ntt[k+(l>>1)]*st%mod;
            ntt[k]=(s1+s2)%mod;ntt[k+(l>>1)]=(s1-s2+mod)%mod;
        }
    }
    int inv=t==-1?pw(s,mod-2):1;
    for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
    freopen("tree.in","r",stdin);
    freopen("tree.out","w",stdout);
    scanf("%d%d",&n,&m);for(int i=1;i<=n;i++)scanf("%d",&v[i]);
    dwg[0]=1;for(int i=1;i<m;i++)dwg[i]=1ll*dwg[i-1]*pw(7,(mod-1)/m)%mod;
    for(int i=1;i<n;i++)scanf("%d%d",&s,&t),adde(s,t);
    dfs1(1,0);for(int i=0;i<m;i++)st[1][i]=1;dfs2(1,0,1);
    for(int i=0;i<m;i++)a[i]=1ll*as[i]*pw(pw(7,mod-2),1ll*i*i%(mod-1)*(mod-1)/m/2%(mod-1))%mod;
    for(int i=0;i<m*2;i++)b[i]=pw(7,1ll*(mod+i-m)*(mod+i-m)%(mod-1)*(mod-1)/m/2%(mod-1));
    int l=1;while(l<=m*3)l<<=1;
    dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)c[i]=1ll*a[i]*b[i]%mod;dft(l,c,-1);
    for(int i=0;i<m;i++)printf("%d ",1ll*c[i+m-1]*pw(pw(7,mod-2),1ll*i*i%(mod-1)*(mod-1)/m/2%(mod-1))%mod*pw(m,mod-2)%mod);
}

```

#### 6.29 T2 组合数(combination)

##### 题目

![](/pic/52.png)

n<=7,m<=1e18,p<=7,p为质数,5s

##### 分析

考虑什么情况下mod p 会有值

分析一下可以得出

1.不为0当且仅当p进制下$i_1+i_2+...i_n$的每一位小于等于m

2.如果不为0，值只与p进制下最后一位有关

考虑$C_i^j=i!/j!/(i-j)!$,可以计算三个阶乘中p的次数

如果有一位大于，那么i-j会有退位，那么会减少里面p的次数，这样的话至少有一个p

第二个结论可以每一次j减去p，发现只不变

容斥为只有上界，从最后一位开始数位dp，$f(i,j,2^n,a,b)$表示记录到第i位，考虑了前j个数的第i位，当前哪些数后面的位大于限制，当前有多少进位，当前最后一位是多少

复杂度$O(4^n*logm*p^4)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define ll long long
int c1[11][11],s[8][67],ct,v[67],inv[7],pw[7][7],p,dp[67][8][130][8][8],ti[67][9][130][8][8],ct2;
ll n,m,l[10],r[10];
void predo(int t,ll x,ll mbit)
{
    int ct2=0;if(x>m)x=m;
    for(int i=1;i<=64;i++)s[t][i]=0;
    while(x)
    s[t][++ct2]=x/mbit,x%=mbit,mbit/=p;
}
int dfs(int t,int x,int y,int st,int a,int b)
{
    if(ti[x][y][st][a][b]==t)return dp[x][y][st][a][b];
    if(y==n+1)return dfs(t,x-1,1,st,b%p,b/p)*c1[v[x]][a]%p;
    if(x==0)
    return !st&&!a;
    dp[x][y][st][a][b]=0;ti[x][y][st][a][b]=t;
    for(int i=0;i<p;i++)
    {
        int s1=st,a1=a+i,b1=b;
        if(a1>=p)a1-=p,b1++;
        if(i>s[y][x])s1|=(1<<y-1);
        if(i<s[y][x])s1&=~(1<<y-1);
        dp[x][y][st][a][b]=(dp[x][y][st][a][b]+dfs(t,x,y+1,s1,a1,b1))%p;
    }
    return dp[x][y][st][a][b];
}
int main()
{
    freopen("combination.in","r",stdin);
    freopen("combination.out","w",stdout);
    scanf("%lld%lld%d",&n,&m,&p);
    for(int i=1;i<=n;i++)scanf("%lld%lld",&l[i],&r[i]);
    ll m2=m,mbit=1;while(m>=mbit*p)mbit*=p,ct++;
    int ct2=0;
    while(m)
    v[++ct2]=m/mbit,m%=mbit,mbit/=p;
    m=m2;mbit=1;ct=1;
    while(m>=mbit*p)mbit*=p,ct++;
    for(int i=1;i<p;i++)
    {
        pw[i][0]=1;
        for(int j=1;j<p;j++)
        pw[i][j]=pw[i][j-1]*i%p;
    }
    for(int i=1;i<p;i++)
    for(int j=1;j<p;j++)
    if(i*j%p==1)inv[i]=j;
    for(int i=0;i<p;i++)c1[i][i]=c1[i][0]=1;
    for(int i=2;i<p;i++)
    for(int j=1;j<i;j++)
    c1[i][j]=(c1[i-1][j]+c1[i-1][j-1])%p;
    int as=0;
    for(int i=0;i<1<<n;i++)
    {
        int fg=1;
        for(int j=1;j<=n;j++)
        if(i&(1<<j-1))
        predo(j,l[j]-1,mbit),fg*=-1;
        else predo(j,r[j],mbit);
        as+=fg*dfs(++ct2,ct,1,0,0,0);
    }
    printf("%d\n",(as%p+p)%p);
}
```

### Part 4 ZR Onsite

#### D1T1 挖矿题

##### 题目

```
帕里桑是一个喜欢看沙雕视频的兔子。突然有一天，帕里桑被抓走扔进了矿洞里带领兔子们挖矿。地下有n层节点可供挖矿，第i层有i个节点（从1开始编号），第一层是地面且只有一个节点。每个节点有一个收益（收益可能为负数），保证地面上的节点的收益为0。一个节点可以通向它左下或右下的节点，也就是说第x层第y个节点可以通向第x+1层第y个节点和第x+1层第y+1个节点。兔子们可以在一些节点建设基地，初始只有地面上的节点是基地。每一天兔子们可以向成为基地的节点的左下或右下方挖若干个节点，获得收益后这些节点将在第二天变为基地。注意一个基地可以同时向左下和右下方挖，也可以不向任何一个方向挖。多个基地可以在同一天中同时挖矿，唯一的要求是一个基地向某个方向挖矿时不可以中途改变方向。一个节点只能恰好被挖掘一次，被挖掘过的节点不可被再次挖掘，所以向某个方向挖掘时无法穿过被挖掘过的节点。注意一天之内也无法挖掘同一个节点两次。兔子们想知道K天之内可以获得的最大收益是多少。

为了防止帕里桑变成红烧兔，请你帮帮他吧！
多组数据,T<=50,n<=50,k<=2
```

##### 分析

因为k<=2，所以情况很简单

k=1时只需枚举两边挖多少，复杂度$O(n)$

考虑k=2时

如果两边第一天挖了a,b，那么第二天可以只考虑左边向右边挖，右边向左边挖

显然，以下两种情况最多出现一种

1.左边有一个挖的长度>=b

2.右边有一个挖的长度>=a

对没有超过的那一边dp，$dp[i][j]$表示考虑那一侧的前i个位置，目前最多挖深度为j，且计算了左边前j个位置的最优值

处理出最大前缀和最大前缀的前缀和即可快速转移

转移复杂度$O(n^3)$

然后枚举a,b，设左侧没有超过，则最优答案为$min(dp[a][i]+(i+1到b向下最大收益))$

然后反过来再做一次

复杂度$O(Tn^3)$

##### 代码

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define N 115
int T,n,s;
long long v[N][N],dp[N][N],mx[N][N],su[N][N],su2[N][N],su3[N][N],su4[N];
int main()
{
    scanf("%d",&T);
    while(T--)
    {
        memset(v,0,sizeof(v));
        memset(dp,-0x3f,sizeof(dp));
        memset(mx,0,sizeof(mx));
        memset(su,0,sizeof(su));
        memset(su2,0,sizeof(su2));
        memset(su3,0,sizeof(su3));
        memset(su4,0,sizeof(su4));
        scanf("%d%d",&n,&s);
        for(int i=2;i<=n;i++)
        for(int j=1;j<=i;j++)
        scanf("%lld",&v[i][j]);
        for(int i=2;i<=n;i++)
        for(int j=2;j<=i;j++)
        su[i][j]=su[i-1][j-1]+v[i][j];
        for(int i=2;i<=n;i++)
        for(int j=1;j<i;j++)
        su2[i][j]=su2[i-1][j]+v[i][j];
        for(int i=2;i<=n;i++)
        for(int j=2;j<=n;j++)
        mx[i][j]=max(mx[i][j-1],su[i+j-1][j]);
        for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
        su3[i][j]=su3[i][j-1]+mx[j][i];
        for(int i=1;i<=n;i++)su4[i]=su4[i-1]+mx[i][n];
        dp[1][1]=0;
        for(int i=2;i<=n;i++)
        for(int j=1;j<=n;j++)
        for(int k=1;k<=n-i+1;k++)
        if(j>=k)
        dp[i][j]=max(dp[i][j],dp[i-1][j]+su2[i+k-1][i]);
        else
        dp[i][k]=max(dp[i][k],dp[i-1][j]+su2[i+k-1][i]+su3[i-1][k]-su3[i-1][j]);
        long long as=0;
        for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
        {
            long long su=0;
            for(int k=1;k<=i;k++)su+=v[k][1];
            for(int k=2;k<=j;k++)su+=v[k][k];
            if(s==1)as=max(as,su);
            else
            for(int k=1;k<=i;k++)
            as=max(as,dp[j][k]+su4[i]-su4[k]+su);
        }
        memset(dp,-0x3f,sizeof(dp));
        memset(mx,0,sizeof(mx));
        memset(su,0,sizeof(su));
        memset(su2,0,sizeof(su2));
        memset(su3,0,sizeof(su3));
        memset(su4,0,sizeof(su4));
        for(int i=2;i<=n;i++)
        for(int j=1;j*2<=i;j++)
        v[i][j]^=v[i][i-j+1]^=v[i][j]^=v[i][i-j+1];
        for(int i=2;i<=n;i++)
        for(int j=2;j<=i;j++)
        su[i][j]=su[i-1][j-1]+v[i][j];
        for(int i=2;i<=n;i++)
        for(int j=1;j<i;j++)
        su2[i][j]=su2[i-1][j]+v[i][j];
        for(int i=2;i<=n;i++)
        for(int j=2;j<=n;j++)
        mx[i][j]=max(mx[i][j-1],su[i+j-1][j]);
        for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
        su3[i][j]=su3[i][j-1]+mx[j][i];
        for(int i=1;i<=n;i++)su4[i]=su4[i-1]+mx[i][n];
        dp[1][1]=0;
        for(int i=2;i<=n;i++)
        for(int j=1;j<=n;j++)
        for(int k=1;k<=n-i+1;k++)
        if(j>=k)
        dp[i][j]=max(dp[i][j],dp[i-1][j]+su2[i+k-1][i]);
        else
        dp[i][k]=max(dp[i][k],dp[i-1][j]+su2[i+k-1][i]+su3[i-1][k]-su3[i-1][j]);
        for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
        {
            long long su=0;
            for(int k=1;k<=i;k++)su+=v[k][1];
            for(int k=2;k<=j;k++)su+=v[k][k];
            if(s==1)as=max(as,su);
            else
            for(int k=1;k<=i;k++)
            as=max(as,dp[j][k]+su4[i]-su4[k]+su);
        }
        printf("%lld\n",as);
    }
}
```

#### D1T2 大水题

##### 题目

```
阿夸是海之女仆，是一个喜欢水的女孩子。她生活在虚拟世界的水中，但是她发现她从一片水域到达另一片水域时可能需要离开水中，她觉得这样有些不方便，于是想连通虚拟世界中的水域。为了简化问题，我们假设这个世界是二维的，我们只考虑这个世界中的土和水。这个世界可以被划分为n个等宽的列，每一列下层有无限体积的土，上层有一些体积的水。我们规定以某个高度为地平线，保证水一定在地平面上，这样第i列的土在地平线上的高度为hi，第i列的水的深度为di。注意由于等宽，我们可以将宽度视为1，所以这里高度和体积是相等的。 我们定义连通水域为水静止时，一个极长的区间[l,r]，区间内每一列都有水(即di>0)。由于这个世界存在垂直向下的重力，所以若相邻两列满足 di>0,hi+di>hi+1+di+1，那么水会从 i 流向 i+1 。类似可以定义从 i 流到 i−1 的条件。注意到经过一段时间后不再有水流动，此时若相邻两列都有水那么他们高度一定相同（hi+di=hi+1+di+1），即连通水域里的高度一定相同。 阿夸想挖去尽量少的土，使得最后只剩下一片连通水域。她只能挖去最上层的土（即让hi减去任意非负实数）。注意水是无法流出这个世界的边界的，即可以将这个世界的第0列和第n+1列的高于地平线的土的高度视为无限大。注意地平线下有无限体积的土，所以阿夸可以把hi减到负数。

你能帮助阿夸挖去最少体积的土使得虚拟世界中最终只剩下一片连通水域吗？（挖去的土的体积为所有hi减去的非负实数之和）
多组数据,T<=10,n<=5000,hi,di<=1000,3s,保证开始时水静止
```

##### 分析

枚举最后水所在的区间，则答案为将两侧的水流过来的代价+将中间的位置向下挖使得水不会流出的代价

前面的问题可以单调栈解决

考虑后面那个问题，首先无视两侧高度对水高度的限制，考虑最终水高度，显然可以将区域内高度高于水高的降低至水高，剩下的不变，显然可以二分出一个点，使得这样之后水容量正好为总水量

如果最后水高>这个值，说明不合法

如果小于，那么一定不优

最后将水量和两侧高度取min，计算时通过前缀和可以做到$O(T(n^2logV+nV))$

可以发现，函数相邻两个整点之间是一个一次函数，并且如果固定左端点，随着右端点增加，水高度会降低

显然这个函数单调，所以复杂度为$O(T(n^2+nV))$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;
#define N 5050
#define M 2010
int n,v1[N],v2[N],su[N][M],su2[N][M],f[N],f2[N],T,st1,lb,rb,mn1[N],mn2[N];
double solve(int i,int j,int tp)
{
    double lb=0;
    if(tp)
    {
        lb=1.0*(st1+su[j][tp-1]-su[i-1][tp-1])/(su2[j][tp-1]-su2[i-1][tp-1]);
        if(su2[j][tp-1]-su2[i-1][tp-1]==0)lb=tp-1;
        if(lb>tp)lb=tp;
    }
    if(tp)tp--;
    return su[j][2000]-su[j][tp]-su[i-1][2000]+su[i-1][tp]-(su2[j][2000]-su2[j][tp]-su2[i-1][2000]+su2[i-1][tp])*lb+st1-lb*(su2[j][tp]-su2[i-1][tp])+su[j][tp]-su[i-1][tp];
}
int main()
{
    scanf("%d",&T);
    while(T--)
    {
        scanf("%d",&n);st1=0;mn1[0]=mn2[n+1]=2000;
        memset(su,0,sizeof(su));memset(su2,0,sizeof(su2));
        memset(f,0,sizeof(f));
        memset(f2,0,sizeof(f2));
        for(int i=1;i<=n;i++)scanf("%d",&v1[i]),mn1[i]=mn2[i]=v1[i];
        for(int i=1;i<=n;i++)scanf("%d",&v2[i]),st1+=v2[i];
        lb=rb=0;
        for(int i=1;i<=n;i++)
        if(v2[i])
        {
            if(!lb)lb=i;
            rb=i;
        }
        int mn=1e9,as=0;
        for(int i=lb;i<=n;i++)
        {
            if(i==lb)mn=v1[i];
            if(mn<v1[i])as+=v1[i]-mn;
            else mn=v1[i];
            f[i]=as;
            mn1[i]=mn;
        }
        mn=1e9,as=0;
        for(int i=rb;i>=1;i--)
        {
            if(i==rb)mn=v1[i];
            if(mn<v1[i])as+=v1[i]-mn;
            else mn=v1[i];
            f2[i]=as;
            mn2[i]=mn;
        }
        for(int i=1;i<=n;i++)
        {
            for(int j=0;j<=2000;j++)su[i][j]=su[i-1][j];
            for(int j=v1[i];j<=2000;j++)su[i][j]+=v1[i];
        }
        for(int i=1;i<=n;i++)
        {
            for(int j=0;j<=2000;j++)su2[i][j]=su2[i-1][j];
            for(int j=v1[i];j<=2000;j++)su2[i][j]++;
        }
        double as2=1e17;
        for(int i=1;i<=n;i++)
        {
            int rb=mn1[i-1];
            for(int j=i;j<=n;j++)
            {
                if(rb==0)continue;
                while(rb>0&&(rb-1)*(su2[j][rb-1]-su2[i-1][rb-1])-su[j][rb-1]+su[i-1][rb-1]>st1)rb--;
                as2=min(as2,f[i-1]+f2[j+1]+solve(i,j,min(rb,mn2[j+1])));
            }
        }
        printf("%.10lf\n",as2);
    }
}
```

#### D2T1 范

##### 题目

给一个n*m的点阵，所有nm个整点上都有一个点，求满足下列条件的点集数 mod 323232323

1.大小为k

2.所有点在一条直线上

n,m,k<=1e5

##### 分析

先特判k=1

对于直线平行坐标轴的，可以直接算

首先枚举直线两端的点组成的向量，那么

$ans=2\sum_{i=1}^n\sum_{j=1}^mC_{gcd(i,j)-1}^{k-2}*(n-i)*(m-j)+n*C_m^k+m*C_n^k$

乘上2是因为即使规定从左向右还有向上和向下

按照莫比乌斯反演套路，得到

$ans=2\sum_{g=k}^{min(n,m)}\sum_{i=1}^{n/g}\sum_{j=1}^{m/g}[gcd(i,j)==1](n-ig)(m-jg)+n*C_m^k+m*C_n^k$

$ans=2\sum_{g=k}^{min(n,m)}\sum_{s=1}^{min(n,m)/g}\mu(s)\sum_{i=1}^{n/gs}\sum_{j=1}^{m/gs}(n-igs)(m-jgs)+n*C_m^k+m*C_n^k$

显然右边是两个等差数列和的乘积

枚举前两重，复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
using namespace std;
#define mod 323232323
#define N 100050
int n,m,k,as,fr[N],ifr[N],mu[N],pr[N],ch[N],ct;
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
int gcd(int i,int j){return j?gcd(j,i%j):i;}
void prime(int n)
{
    mu[1]=1;
    for(int i=2;i<=n;i++)
    {
        if(!ch[i])mu[i]=-1,pr[++ct]=i;
        for(int j=1;i*pr[j]<=n&&j<=ct;j++)
        {
            mu[i*pr[j]]=-mu[i];ch[i*pr[j]]=1;
            if(i%pr[j]==0){mu[i*pr[j]]=0;break;}
        }
    }
}
int solve(int x)
{
    int as=0;
    for(int g=1;g*x<=n&&g*x<=m;g++)
    {
        int s1=1;
        int rb=n-(g*x),lb=n-(n-1)/(g*x)*(g*x),su=(rb-lb)/(g*x)+1;
        s1=1ll*s1*(rb+lb)%mod*su%mod;
        rb=m-(g*x),lb=m-(m-1)/(g*x)*(g*x),su=(rb-lb)/(g*x)+1;
        s1=1ll*s1*(rb+lb)%mod*su%mod*(mod+1)/2%mod;
        as=(as+1ll*mu[g]*s1)%mod;
    }
    return (as+mod)%mod;
}
int main()
{
    scanf("%d%d%d",&n,&m,&k);prime(n);
    if(k==1){printf("%d\n",1ll*n*m%mod);return 0;}
    fr[0]=ifr[0]=1;
    for(int i=1;i<=100000;i++)fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
    as=(as+1ll*n*fr[m]%mod*ifr[m-k]%mod*ifr[k]+1ll*m*fr[n]%mod*ifr[n-k]%mod*ifr[k])%mod;
    for(int i=k;i<=n;i++)as=(as+1ll*fr[i-2]*ifr[i-k]%mod*ifr[k-2]%mod*solve(i-1))%mod;
    printf("%d\n",as);
}
```

#### D2T2 老

##### 题目

```
数轴上有 N 个区间，第 i 个为 [Li, Ri]。
我们称点 x 能覆盖区间 [Li, Ri]，当且仅当 Li ≤ x ≤ Ri，即，其在区间内。为了维持如此的覆盖，需要
支付 abs((x − Li) − (Ri − x)) 的代价，即，其到区间两端的距离之差。
你需要在数轴上选中若干个点，对于每个区间需要选择恰好一个选中的点覆盖它。
问：
1.至少需要选中多少个点。
2.在满足选中的点最少的前提下，覆盖的代价之和至少是多少。
n<=5e5 Li,Ri<=n
```

##### 分析

关于第一问，可以贪心解决，每次找到当前所有区间中ri最小的并放下去

考虑怎么解决第二问

贪心的放置方案将整个区域划分成了一些区间

如果一个区间里面没有放，根据贪心的思路这样不合法

如果一个区间放了多个，一定不优

所以一定是一个区间放一个

显然一个区间选择的点一定是中点两侧最近点中的一个

设$f(i,j)$表示选中点i,j，中间没有选点，当前所有中点在i到j之间的区间的代价和

考虑每个中点的贡献可以得到$f(i+1,j+1)+f(i,j)>=f(i,j+1)+f(i+1,j)$

因此f满足四边型不等式

因此直接进行决策单调性优化dp即可

通过记录前缀和，计算f可以做到$O(1)$，总复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 1000050
int l[N],r[N],v[N],l1[N],r1[N],n,mxr=0,mn[N],mx[N],st[N],ct1;
long long su[N],ct[N],dp[N];
bool cmp(int i,int j){return l1[i]<l1[j];}
long long solve(int l,int r)
{
    if(!l)return 1ll*r*(ct[r]-ct[l])-(su[r]-su[l]);
    if(r==mxr+1)return 1ll*(su[r]-su[l])-1ll*l*(ct[r]-ct[l]);
    return 1ll*r*(ct[r]-ct[(l+r)>>1])-(su[r]-su[(l+r)>>1])+1ll*(su[(l+r)>>1]-su[l])-1ll*l*(ct[(l+r)>>1]-ct[l]);
}
void solve2(int l,int r,int lb,int rb)
{
    if(l>r)return;
    int mid=(l+r)>>1;
    long long mn1=1e18,as;
    for(int j=max(mx[mid],lb);j<=rb;j++)
    {
        long long tp=solve(j,mid)+dp[j];
        if(tp<mn1)mn1=tp,as=j;
    }
    dp[mid]=mn1;
    solve2(l,mid-1,lb,as);
    solve2(mid+1,r,as,rb);
}
int main()
{
    srand(998244353);
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d%d",&l1[i],&r1[i]),su[l1[i]+r1[i]]+=l1[i]+r1[i],ct[l1[i]+r1[i]]++,l1[i]*=2,r1[i]*=2,mxr=max(mxr,r1[i]),v[i]=i;
    sort(v+1,v+n+1,cmp);
    for(int i=1;i<=n;i++)l[i]=l1[v[i]],r[i]=r1[v[i]];
    mn[n]=r[n];
    for(int i=n-1;i>=1;i--)mn[i]=min(mn[i+1],r[i]);
    int lb=1,as=0;
    while(lb<=n)
    {
        int tp=mn[lb];
        st[++ct1]=tp;
        as++;
        while(lb<=n&&l[lb]<=tp)
        lb++;
    }
    st[++ct1]=mxr+1;
    printf("%d ",as);
    for(int i=1;i<=n;i++)mx[r[i]+1]=max(mx[r[i]+1],l[i]);
    for(int i=1;i<=mxr+1;i++)mx[i]=max(mx[i],mx[i-1]),su[i]+=su[i-1],ct[i]+=ct[i-1];
    for(int i=1;i<=st[1];i++)dp[i]=solve(0,i);
    for(int i=1;i<ct1;i++)solve2(st[i]+1,st[i+1],st[i-1]+1,st[i]);
    printf("%lld\n",dp[mxr+1]);
}
```

#### D2T3 板

##### 题目

求满足下面条件的树个数 mod 323232323

1.有n个点

2.dfs序为1,2,3,...,n

3.第i个点子树大小大于等于di

n<=10000

##### 分析

显然如果b在a的子树中，c在b的子树中，那么c在a的子树中

因此可以对原条件进行一定改变，使得所有限制在dfs序上不相交或者包含

这显然是一棵树的结构

设$f[i][j]$表示i限制的点都在i子树内，且dfs序为i限制右端点的点的深度比i深度大j的方案数

考虑插入一个子树，它的父亲可以是dfs序最大点到根链上的任意一点，那么

$f_1[i][k]=\sum_{l=0}^k(\sum_{s=l}^{sz[i]}f[i][l])*f[j][k-l]$

最后答案为$\sum_{i=0}^n f[1][i]$

显然,dp第二维大小小于当前考虑的子树size，因此总复杂度$O(n^2)$

##### 代码

这里的第二维是从1开始的

```cpp
#include<cstdio>
using namespace std;
#define N 10050
#define mod 323232323
int dp[N][N],f[N],sz[N],su[N],a[N],b[N],n;
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    a[1]=n;
    for(int i=n;i>=1;i--)
    {
        a[i]=a[i]+i-1;if(a[i]>n){printf("0\n");return 0;}
        b[i]=i;
        a[i]=a[b[a[i]]];
        sz[i]=1;dp[i][1]=1;
        for(int j=i+1;j<=a[i];j=a[b[j]]+1)
        {
            su[sz[i]+1]=0;
            for(int k=sz[i];k>=0;k--)su[k]=(su[k+1]+dp[i][k])%mod,dp[i][k]=0;
            for(int k=1;k<=sz[i];k++)
            for(int l=1;l<=sz[j];l++)
            dp[i][k+l]=(1ll*su[k]*dp[j][l]+dp[i][k+l])%mod;
            sz[i]+=sz[j];
        }
        for(int j=i;j<=a[i];j++)b[j]=i;
    }
    int as=0;
    for(int i=1;i<=n;i++)as=(as+dp[1][i])%mod;
    printf("%d\n",as);
}
```

#### D3T1 图

##### 题目

```
一个简单图是好的当且仅当它至少满足下列条件之一:
1.它是一个点
2.它可以被分成若干个不连通的部分，且每一部分是好的
3.它是一个好图的补图
给一个好图，q次询问，每一次给一些点，询问这些点的导出子图是否连通，如果连通，则需要回答这些点两点间最短路的最大值
点数为n，边数为m，总询问点数为k
n<=1e4,m<=3e5,q<=5e4,k<=1e5

```

##### 分析

对于连通的两点，考虑它们的最短路

有三种情况

1.这张图所有好图连通，且两点在不同好图内，则答案为1

2.这张图所有好图连通，且两点在相同好图内，则答案最大为2

3.这张图所有好图不连通，则可以递归下去做

因此，答案<=2

答案为1当且仅当这是一个团

考虑找出所有点集间的边，这样就可以判断连通和团

设询问点数为s

$s<=\sqrt m$时，可以$O(s^2)$枚举

否则，可以$O(m)$枚举

容易发现复杂度为$O(k\sqrt m)$,可以通过

##### 代码

```cpp
#include<cstdio>
#include<cmath>
#include<algorithm>
#include<bitset>
using namespace std;
#define N 10100
#define M 300050
int n,m,q,k,a,b,fa[N],s[N],f2[N],g[M][2],is[N];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int finds2(int x){return f2[x]==x?x:f2[x]=finds2(f2[x]);}
bitset<N> f[N],s1;
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)fa[i]=i,f[i][i]=1;
    for(int i=1;i<=m;i++)scanf("%d%d",&a,&b),f[a][b]=f[b][a]=1,fa[finds(a)]=finds(b),g[i][0]=a,g[i][1]=b;
    scanf("%d",&q);
    while(q--)
    {
        scanf("%d",&k);s1.reset();
        for(int i=1;i<=k;i++)scanf("%d",&s[i]),s1[s[i]]=1,is[s[i]]=1,f2[s[i]]=s[i];
        sort(s+1,s+k+1);
        int fg=0;
        for(int i=1;i<k;i++)if(finds(s[i])!=finds(s[i+1]))fg=1;
        if(fg)printf("-1\n");
        else
        {
            if(k>=300)
            {
                for(int i=1;i<=m;i++)
                if(is[g[i][0]]&&is[g[i][1]])f2[finds2(g[i][0])]=finds2(g[i][1]);
                for(int i=1;i<k;i++)if(finds2(s[i])!=finds2(s[i+1]))fg=1;
            }
            else
            {
                for(int i=1;i<=k;i++)
                for(int j=1;j<=k;j++)
                if(f[s[i]][s[j]])f2[finds2(s[i])]=finds2(s[j]);
                for(int i=1;i<k;i++)if(finds2(s[i])!=finds2(s[i+1]))fg=1;
            }
            if(fg)printf("-1\n");
            else
            {
                for(int i=1;i<=k;i++)
                {
                    int tp=((~f[s[i]])&s1).any();
                    if(tp){fg=1;break;}
                }
                printf("%d\n",fg+1);
            }
        }
        for(int i=1;i<=k;i++)is[s[i]]=0;
    }
}

```

#### D3T2 递归

##### 题目

```cpp
对于一个长度为 n = 2^k 的 0/1 序列 A[0,...,n−1]，定义如下递归函数：
void work (int l,int r){
    bool flag[2]={false,false};
    for(int i=l;i<r;++i)flag[a[i]]=true;
    if(flag[0]&&flag[1]){
        ++count;
        int mid=(l+r)/2;
        work(l,mid);
        work(mid,r);
    }
    return;
}
其中 count 为全局变量。
初始时 A 中所有元素均为 0。现在有 q 次操作，每次操作为给定 l, r (0 ≤
l ≤ r ≤ n − 1)，让 A[i] = 1 xor A[i], (l ≤ i ≤ r)。对于每一种操作可以选择用
或者不用，总共会产生 2^q 个操作集合（这意味着可以不选任何操作，也可以全
选所有操作）。对这 2^q 个操作集合得到的 2^q 个序列 A （可能相同），分别调用
work(0, n) 函数，问最后 count 变量总和为多少。由于答案可能较大，请输出这
个数字对 998244353 取模的结果，即本题不考虑 C++ 代码中 int 溢出对 count 值
的影响，所有对于 count 运算均在对 998244353 取模意义下进行。
n,q<=600000

```

##### 分析

可以发现如果一个区间不会递归，那即使递归下去所有子区间贡献也是0

所以可以算每一个区间的贡献

首先差分，问题变为一次操作修改两个位置，区间全部相同可以看成一段差分为0

考虑怎么处理这个问题

修改看成连边，那么有三种边

1.内部到内部的边

2.内部到外部的边

3.外部到外部的边

对于第三种直接乘，考虑只有第一种的情况

对于一个连通块，先处理出一个生成树，然后对于每一条非树边，都可以将对应的环全部翻转，那么答案为2^(边数-点数+1)

对于多个连通块，直接乘起来即可

再考虑带第二种的情况

如果选了一些第二种边，相当于一些点一开始就是1

如果有偶数个1点，那么可以在生成树上配对然后路径异或，答案不变

否则，一定无解

因此，如果连通块有k条(k>0)第二种边，那么需要乘上$C_k^0+C_k^2+C_k^4+...=2^{k-1}$

复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
#include<vector>
using namespace std;
#define N 600050
#define mod 998244353
int fa[N],vl[N],n,m,a,b,is[N],c2[N],c3[N],as;
struct edg{int a,b;};
vector<edg> tp[N*2];
int finds(int x){return fa[x]==x?x:fa[x]=finds(fa[x]);}
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void solve(int l,int r,int id,int st1)
{
    if(l==r)return;
    for(int i=l;i<=r;i++)fa[i]=i,is[i]=0,c2[i]=c3[i]=0;
    int ct1=0,su=0;
    for(int i=l;i<r;i++)su+=vl[i];
    su+=tp[id].size();
    int sz=tp[id].size();
    for(int i=0;i<sz;i++)
    if(finds(tp[id][i].a)==finds(tp[id][i].b))ct1++;
    else fa[finds(tp[id][i].a)]=finds(tp[id][i].b);
    for(int i=l;i<r;i++)if(vl[i]){c2[finds(i)]++,c3[finds(i)]+=vl[i]-1;}
    for(int i=l;i<r;i++)if(c2[i])ct1+=c3[i]+c2[i]-1;
    as=(as-pw(2,ct1+m-su)+pw(2,m))%mod;
    int mid=(l+r)>>1,ls=st1,rs=st1;
    for(int i=l;i<=mid;i++)if(vl[i])ls+=vl[i];
    for(int i=mid;i<r;i++)if(vl[i])rs+=vl[i];
    for(int i=0;i<sz;i++)
    {
        edg sb=tp[id][i];
        if(sb.a<mid&&sb.b<mid)tp[id*2].push_back(sb),rs++;
        else if(sb.a>mid&&sb.b>mid)tp[id*2+1].push_back(sb),ls++;
        else vl[sb.a]++,vl[sb.b]++;
    }
    solve(l,mid,id*2,ls);solve(mid+1,r,id*2+1,rs);
}
int main()
{
    scanf("%d%d",&n,&m);
    int fu=0;
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d",&a,&b);
        if(a==0&&b==n-1)fu++;
        else if(a==0)vl[b+1]++;
        else if(b==n-1)vl[a]++;
        else tp[1].push_back((edg){a,b+1});
    }
    solve(1,n,1,fu);
    printf("%d\n",(as+mod)%mod);
}

```

#### D3T3 电阻

##### 题目

```
有一个deque，维护以下操作
1.加入一个t时刻从左边或右边插入的操作
2.加入一个t时刻从左边或右边删除的操作
3.删除t时刻的操作
4.询问对于当前的操作，在t时刻deque左侧第i个位置的数
n<=100000,强制在线

```

##### 分析

[模板]可追溯化双端队列

考虑手写单调队列的写法

初始l=1,r=0

左插入s[--l]=v

右插入s[++r]=v

左删除l++

右删除r--

使用平衡树维护每一个时刻的l和r

询问首先查询当前的l和r，然后就可以知道对应元素的l和r，然后相当于查询最后一个值等于一个给定值的插入操作，可以直接维护

复杂度$O(nlogn)$

##### 代码

```cpp
#include<algorithm>
#include<cstdio>
#include<map>
#include<set>
using namespace std;
#define N 100050
int n,q,a,b,c,d,id[N],lstans,s[N][5],ct;
map<int,int> tp;
char e[11];
struct Splay {
    int ch[N][2],vl[N],mx[N],mn[N],lz[N],v[N],v2[N],fa[N],rt,ct,is[N];
    void pushup(int x) {
        mx[x]=max(max(mx[ch[x][0]],mx[ch[x][1]]),is[x]?-998244353:vl[x]);
        mn[x]=min(min(mn[ch[x][0]],mn[ch[x][1]]),is[x]?998244353:vl[x]);
    }
    void pushdown(int x) {
        if(lz[x])lz[ch[x][0]]+=lz[x],lz[ch[x][1]]+=lz[x],mn[ch[x][0]]+=lz[x],mn[ch[x][1]]+=lz[x],mx[ch[x][0]]+=lz[x],mx[ch[x][1]]+=lz[x],vl[ch[x][0]]+=lz[x],vl[ch[x][1]]+=lz[x],lz[x]=0;
        mx[0]=-1e9;mn[0]=1e9;
    }
    void rotate(int x) {
        int f=fa[x],g=fa[f],tp=ch[f][1]==x;
        pushdown(f);
        pushdown(x);
        ch[g][ch[g][1]==f]=x;
        fa[x]=g;
        ch[f][tp]=ch[x][!tp];
        fa[ch[x][!tp]]=f;
        ch[x][!tp]=f;
        fa[f]=x;
        pushup(f);
        pushup(x);
    }
    void splay(int x,int y) {
        while(fa[x]!=y) {
            int f=fa[x],g=fa[f];
            if(g!=y)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);
            rotate(x);
        }
        if(!y)rt=x;
    }
    void init() {
        rt=1;
        ct=3;
        v[1]=2e9;
        v[2]=2e9+1;v[3]=-1;
        ch[1][1]=2,fa[2]=1;
        ch[1][0]=3,fa[3]=1;
        mx[1]=mx[2]=-1e9;
        mn[1]=mn[2]=1e9;
        mx[0]=-1e9;mn[0]=1e9;
        is[1]=is[2]=1;
    }
    void insert(int x,int s,int s2,int s3) {
        int tp=v[x]<s;
        pushdown(x);
        if(!ch[x][tp]) {
            ch[x][tp]=++ct;
            v[ct]=s;
            fa[ct]=x;
            v2[ct]=s2;
            if(s3<-100000)mx[ct]=-99898989,mn[ct]=9989898,is[ct]=1,vl[ct]=-s3-1000000;
            else mx[ct]=mn[ct]=vl[ct]=s3;
            splay(ct,0);
            return;
        }
        insert(ch[x][tp],s,s2,s3);
    }
    int find(int x,int s) {
        if(!x)return 0;
        if(v[x]==s) {
            splay(x,0);
            return x;
        }
        int tp=v[x]<s;
        return find(ch[x][tp],s);
    }
    void modify(int l,int r,int v) {
        int s1=find(rt,l),s2=find(rt,r);
        int t1=ch[s1][0],t2=ch[s2][1];
        while(ch[t1][1])t1=ch[t1][1];
        while(ch[t2][0])t2=ch[t2][0];
        splay(t1,0);
        splay(t2,t1);
        int s3=ch[t2][0];
        lz[s3]+=v;
        mn[s3]+=v;
        mx[s3]+=v;
        vl[s3]+=v;
        splay(s3,0);
    }
    pair<int,int> query1(int x,int s) {
        if(!x)return make_pair(-1,-1);
        pushdown(x);
        if(ch[x][1])if(mn[ch[x][1]]<=s&&mx[ch[x][1]]>=s)return query1(ch[x][1],s);
        if(vl[x]==s&&!is[x])return make_pair(v2[x],v[x]);
        return query1(ch[x][0],s);
    }
    int getnxt(int x,int s) {
        if(!x)return x;
        if(v[x]<s)return getnxt(ch[x][1],s);
        else {
            int tp=getnxt(ch[x][0],s);
            if(!tp)return x;
            return tp;
        }
    }
    int getpre(int x,int s) {
        if(!x)return x;
        if(v[x]>s)return getpre(ch[x][0],s);
        else {
            int tp=getpre(ch[x][1],s);
            if(!tp)return x;
            return tp;
        }
    }
    pair<int,int> query(int l,int r,int s) {
        int s1=getnxt(rt,r+1);
        splay(s1,0);
        return query1(ch[s1][0],s);
    }
    void del(int x) {
        int st=find(rt,x);splay(st,0);
        int lb=ch[st][0],rb=ch[st][1];
        while(ch[lb][1])lb=ch[lb][1];
        while(ch[rb][0])rb=ch[rb][0];
        splay(lb,0);
        splay(rb,lb);
        ch[rb][0]=fa[st]=0;
        pushup(rb);
        pushup(lb);
    } 
    int query2(int x,int s) {
        if(v[x]==s)
        return vl[x];
        pushdown(x);
        return query2(ch[x][v[x]<s],s);
    }
} t[2];
int solve(int s,int x) {
    int l1=t[0].getpre(t[0].rt,s),l2=t[1].getpre(t[1].rt,s);
    int lsz=t[0].query2(t[0].rt,t[0].v[l1]),rsz=t[1].query2(t[1].rt,t[1].v[l2]);
    int tp1=lsz-x+1,tp2=x-lsz;
    pair<int,int> as1=t[0].query(1,s,tp1),as2=t[1].query(1,s,tp2);
    if(as1.second<as2.second)return as2.first;return as1.first;
}
int main() {
    scanf("%d%d",&n,&q);
    t[0].init();
    t[1].init();
    while(n--) {
        scanf("%s",e);
        if(e[0]=='I')
        scanf("%d%d%d",&a,&c,&b),t[c].insert(t[c].rt,a^(q*lstans),b,t[c].query2(t[c].rt,t[c].v[t[c].getpre(t[c].rt,a^(q*lstans))])),t[c].modify(a^(q*lstans),2e9,1),tp[a^(q*lstans)]=++ct,s[ct][0]=1,s[ct][1]=a^(q*lstans),s[ct][2]=b,s[ct][3]=c;
        if(e[0]=='E')
        scanf("%d%d",&a,&b),t[b].insert(t[b].rt,a^(q*lstans),0,-t[b].query2(t[b].rt,t[b].v[t[b].getpre(t[b].rt,a^(q*lstans))])-1000000),t[b].modify(a^(q*lstans),2e9,-1),tp[a^(q*lstans)]=++ct,s[ct][0]=2,s[ct][1]=a^(q*lstans),s[ct][2]=b;
        if(e[0]=='D') {
            scanf("%d",&a);
            a^=lstans*q;
            int su=tp[a];
            if(s[su][0]==1)t[s[su][3]].modify(s[su][1],2e9,-1),t[s[su][3]].del(s[su][1]);
            else t[s[su][2]].modify(s[su][1],2e9,1),t[s[su][2]].del(s[su][1]);
        }
        if(e[0]=='Q')scanf("%d%d",&a,&b),printf("%d\n",lstans=solve(a^(q*lstans),b));
    }
}

```

#### D4T1 合并果子

##### 题目

```
小象有n堆果子排成一列，每堆果子有个权值。小象一开始可以选择一堆果子。接下来每一轮，小象可以选择将这堆果子与左边或者右边的果子合并，形成一堆新的果子，在新的果子上继续进行上面的操作。进行n−1轮后，合并成一堆果子。每次合并两堆果子的代价为两堆果子的重量之和。

小象想知道，对于每堆果子，如果小象选择以这堆果子作为初始选择，那么合并的最小代价是什么。
n<=200000

```

##### 分析

改变原问题：设第i个合并选择的是ai，最大化ai*i的和

考虑一侧相邻的两个数a,b,a在b前面

如果a>b，那么显然选了a以后接下来立刻会选b

如果把这样的a,b缩起来，考虑两段，如果前面那一段的平均数大于等于后面那一段，那么可以合并

合并完以后，因为一侧里面的平均数单调递增，所以可以直接贪心

扫描线，显然总共只会添加/删除段n次，使用平衡树维护贪心，复杂度$O(nlogn)$

##### 代码

暴躁代码警告

```cpp
#include<cstdio>
#include<algorithm>
#include<cmath>
using namespace std;
#define N 800600
struct Splay
{
    int ch[N][2],fa[N],rt,ct;
    long long sz[N],vl[N],s1[N],v1[N],su1[N],vu[N],vd[N];
    void init(){ct=2,rt=1;ch[1][1]=2;fa[2]=1;vu[2]=1198244343;vd[2]=1;vd[1]=1;vd[0]=1;}
    void pushup(int x){vl[x]=vl[ch[x][0]]+vl[ch[x][1]]+v1[x]+(s1[x]*vu[x]/vd[x])*sz[ch[x][0]]+su1[ch[x][1]]*(sz[ch[x][0]]+s1[x]);sz[x]=sz[ch[x][0]]+sz[ch[x][1]]+s1[x];su1[x]=su1[ch[x][0]]+su1[ch[x][1]]+s1[x]*vu[x]/vd[x];}
    void rotate(int x){int f=fa[x],g=fa[f],tp=ch[f][1]==x;ch[g][ch[g][1]==f]=x;fa[x]=g;ch[f][tp]=ch[x][!tp];fa[ch[x][!tp]]=f;ch[x][!tp]=f;fa[f]=x;pushup(f);pushup(x);}
    void splay(int x,int y=0){while(fa[x]!=y){int f=fa[x],g=fa[f];if(g!=y)rotate((ch[f][1]==x)^(ch[g][1]==f)?x:f);rotate(x);}if(!y)rt=x;}
    void insert(int x,long long t1,long long t2,long long x2,long long x3){bool tp=vu[x]*t2==vd[x]*t1?s1[x]<x2:vu[x]*t2<vd[x]*t1;if(!ch[x][tp]){ch[x][tp]=++ct;fa[ct]=x;vu[ct]=t1;vd[ct]=t2;s1[ct]=x2;v1[ct]=x3;su1[ct]=t1*x2/t2;splay(ct);return;}insert(ch[x][tp],t1,t2,x2,x3);}
    int finds(int x,long long t1,long long t2,long long x2,long long x3){bool tp=vu[x]*t2==vd[x]*t1?s1[x]<x2:vu[x]*t2<vd[x]*t1;if(vu[x]*t2==vd[x]*t1&&s1[x]==x2)return x;return finds(ch[x][tp],t1,t2,x2,x3);}
    void del(long long t1,long long t2,long long x2,long long x3){int tp=finds(rt,t1,t2,x2,x3);splay(tp);int lb=ch[tp][0];while(ch[lb][1])lb=ch[lb][1];int rb=ch[tp][1];while(ch[rb][0])rb=ch[rb][0];splay(lb);splay(rb,lb);fa[ch[rb][0]]=0;ch[rb][0]=0;pushup(rb);pushup(lb);splay(rb);}
}s;
int n,v[N],rb,ct1;long long su=0;
struct seg{long long su,sz,tp;}que[N];
struct modify{int ti,tp;long long t,x,y,z;friend bool operator <(modify a,modify b){return a.ti==b.ti?a.tp>b.tp:a.ti<b.ti;}}st[N];
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&v[i]),su+=v[i];
    for(int i=1;i<n;i++)
    {
        que[++rb]=(seg){v[i],1,v[i]};
        while(rb>1&&que[rb-1].su*que[rb].sz<=que[rb].su*que[rb-1].sz)
        {
            st[++ct1]=(modify){i+1,-1,que[rb-1].su,que[rb-1].sz,que[rb-1].sz,que[rb-1].tp};
            que[rb-1]=(seg){que[rb-1].su+que[rb].su,que[rb-1].sz+que[rb].sz,que[rb-1].tp+que[rb].tp+que[rb-1].su*que[rb].sz};
            rb--;
        }
        st[++ct1]=(modify){i+1,1,que[rb].su,que[rb].sz,que[rb].sz,que[rb].tp};
    }
    rb=0;
    for(int i=n;i>=2;i--)
    {
        que[++rb]=(seg){v[i],1,v[i]};
        while(rb>1&&que[rb-1].su*que[rb].sz<=que[rb].su*que[rb-1].sz)
        {
            st[++ct1]=(modify){i,1,que[rb-1].su,que[rb-1].sz,que[rb-1].sz,que[rb-1].tp};
            que[rb-1]=(seg){que[rb-1].su+que[rb].su,que[rb-1].sz+que[rb].sz,que[rb-1].tp+que[rb].tp+que[rb-1].su*que[rb].sz};
            rb--;
        }
        st[++ct1]=(modify){i,-1,que[rb].su,que[rb].sz,que[rb].sz,que[rb].tp};
    }
    while(rb)st[++ct1]=(modify){0,1,que[rb].su,que[rb].sz,que[rb].sz,que[rb].tp},rb--;
    sort(st+1,st+ct1+1);
    s.init();
    int lb=1;
    for(int i=1;i<=n;i++)
    {
        while(lb<=ct1&&st[lb].ti<=i)
        {
            if(st[lb].tp==1)
            s.insert(s.rt,st[lb].t,st[lb].x,st[lb].y,st[lb].z);
            else 
            s.del(st[lb].t,st[lb].x,st[lb].y,st[lb].z);
            lb++;
        }
        printf("%lld\n",su*n-s.vl[s.rt]-v[i]);
    }
}

```

#### day 5

数竞原题加强版+abel群计数，我不会证

#### D6T1 三角函数

##### 题目

求$\sum_{i=1}^nsin(x)/x^i$的m阶导数，分别输出sin和cos部分,模998244353

n<=1e5

##### 分析

$(sin(x)/x^i)^`=(cos(x)x^i+isin(x)x^{i-1})/x^{2i}=cos(x)/x^i+isin(x)/x^{i+1}$

cos同理

可以发现，它只有两种转移

1.系数，次数不变，sin(x),cos(x),-sin(x),-cos(x)间转换

2.sin与cos不变，系数乘上次数，次数+1

可以发现，如果多次求导后次数从i变为j，那么乘的系数为$C_m^{j-i}*(j-1)!/i!$

第一个C表示m次中选j-i次2转移的方案，第二个相当于$i*(i+1)*(i+2)*...*(j-1)$

于是按照转移1次数模4(或者模2)分类，进行ntt即可

复杂度$O(nlogn)$

##### 代码

```cpp
#include<cstdio>
using namespace std;
#define N 263000
#define mod 998244353
int n,m,s[N],as[N],a[N],b[N],c[N],ntt[N],rev[N],t,inv[N],st[2][19][N],lg[N],fr[N],ifr[N];
int pw(int a,int p){int ans=1;while(p){if(p&1)ans=1ll*ans*a%mod;a=1ll*a*a%mod;p>>=1;}return ans;}
void dft(int s,int *a,int t)
{
    for(int i=0;i<s;i++)rev[i]=(rev[i>>1]>>1)+(i&1)*s/2,ntt[rev[i]]=a[i];
    for(int i=2;i<=s;i<<=1)
    {
        for(int j=0;j<s;j+=i)
        for(int k=j,s1=0;k<j+(i>>1);k++,s1++)
        {
            int a1=ntt[k],a2=1ll*ntt[k+(i>>1)]*st[(t+1)>>1][lg[i]][s1]%mod;
            ntt[k]=a1+a2-(a1+a2>=mod?mod:0);
            ntt[k+(i>>1)]=a1-a2+(a1<a2?mod:0);
        }
    }
    int inv=pw(s,t==-1?mod-2:0);
    for(int i=0;i<s;i++)a[i]=1ll*ntt[i]*inv%mod;
}
int main()
{
    fr[0]=ifr[0]=1;
    for(int i=1;i<=262800;i++)inv[i]=pw(i,mod-2),fr[i]=1ll*fr[i-1]*i%mod,ifr[i]=pw(fr[i],mod-2);
    for(int i=2;i<=262800;i++)lg[i]=lg[i>>1]+1;
    for(int i=1;i<=18;i++)
    {
        st[0][i][0]=st[1][i][0]=1;
        int tp1=pw(3,(mod-1)>>i),tp2=pw(tp1,mod-2);
        for(int j=1;j<1<<i;j++)
        st[0][i][j]=1ll*st[0][i][j-1]*tp2%mod,st[1][i][j]=1ll*st[1][i][j-1]*tp1%mod;
    }
    scanf("%d%d",&m,&n);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]),a[i]=1ll*a[i]*ifr[i-1]%mod;
    for(int i=0;i<=m;i++)
    if((m-i+1)&1)
    b[i]=1ll*fr[m]*ifr[i]%mod*ifr[m-i]%mod*(((m-i)&2)?mod-1:1)%mod*(i&1?mod-1:1)%mod;
    int l=1;while(l<=n+m)l<<=1;
    dft(l,a,1);dft(l,b,1);for(int i=0;i<l;i++)c[i]=1ll*a[i]*b[i]%mod;dft(l,c,-1);
    for(int i=1;i<=n+m;i++)printf("%d ",1ll*c[i]*fr[i-1]%mod);
    printf("\n");
    for(int i=0;i<l;i++)b[i]=c[i]=0;
    for(int i=0;i<=m;i++)
    if((m-i)&1)b[i]=1ll*fr[m]*ifr[i]%mod*ifr[m-i]%mod*(((m-i+3)&2)?mod-1:1)%mod*(i&1?mod-1:1)%mod;
    dft(l,b,1);for(int i=0;i<l;i++)c[i]=1ll*a[i]*b[i]%mod;dft(l,c,-1);
    for(int i=1;i<=n+m;i++)printf("%d ",1ll*c[i]*fr[i-1]%mod);
}

```

#### D6T3 简单字符串

##### 题目

对于字符串 $s$ 和整数 $k$ ，定义 $f(s,k)$ 为，将 $s$ 划分为**至多** $k$ 段 $u_1,u_2,…u_l$ ，最小化 $max_{1≤i≤l}u_i$ (比较按照字典序) ，求最小化的结果。

有一个字符串 $S$ ， $q$ 次询问 $f(s[l_i…|s|],k_i)$ 的值，对于每个询问输出 $a_i,b_i$ 表示 $f(s[l_i…|s|],k_i)=S[a_i…b_i]$，其中要求$S[ai…bi]$在一个可能的划分中。

如果有多个，输出 $ai$ 最小的解，要求$ai≥li$。

n,q<=1e5

因为不会证lyndon相关，所以咕咕咕

#### D7T1 开源

##### 题目

```
蔡德仁发现 OI Diary 的预算不够用了，于是造了一些题来卖。

他把题分成了 n 堆，第 i 堆有 ai 道题。蔡德仁和艾莉芬轮流取题（选一堆，然后在其中取若干个，不能不取），蔡德仁先手。蔡德仁第一次可以取至多 K 个，之后每个人取的题数不能超过上一个人刚刚取的题数。不能取的人输。

请你求出蔡德仁是否必胜。如果是，你还需要求出他第一步的所有必胜策略。
n<=50000,a<=1e9

```

##### 分析

打表找规律

首先如果总和为奇数，那么先手取1一定获胜

否则，双方都只能取偶数，第一个不能取偶数的人输，所以可以把所有数除以2向下取整

可以得到结论，先手必胜当且仅当每一堆异或起来不为0且异或的lowbit小于等于k

枚举取哪一堆，枚举取完的lowbit，可以发现，对于一个lowbit最多有一种答案，因此复杂度$O(nloga)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
using namespace std;
#define N 50050
int n,v,s[N],su,as[N],ct,fg;
int main()
{
    scanf("%d %d",&n,&v);
    for(int i=1;i<=n;i++)scanf("%d",&s[i]),su^=s[i];
    if(!su){printf("0\n");return 0;}
    for(int i=1;i<=n;i++)
    {
        ct=0;
        int fk=s[i]^su;
        if(fk<=s[i])as[++ct]=s[i]-fk;
        for(int j=1;j<=30;j++)
        {
            int as1=((s[i]>>j)<<j)+((su^s[i])&((1<<j)-1));
            if(as1>s[i])as1-=1<<j;
            if(s[i]-as1!=0&&s[i]-as1<1<<j&&as1>=0)as[++ct]=s[i]-as1;
        }
        sort(as+1,as+ct+1);
        for(int j=1;j<=ct;j++)if(as[j]!=as[j-1]&&as[j]<=v)
        {
            if(!fg){printf("1\n");fg=1;}
            printf("%d %d\n",i,as[j]);
        }
    }
    if(!fg)printf("0\n");
}

```

#### D7T3 灌水

##### 题目

```
蔡德仁发现 OI Diary 的预算不够用了，于是水了一些周边。

他有 n 个手办，高度分别为 1 到 n。现在它们按某种顺序排成一排，第 i 个手办的高度为 pi。卖掉第 i 个手办可以获得 ai 的利润，ai 可以为负，表示亏损。

每天会发生以下两件事之一：

1.客人选定第 k 个手办，准备买包含它在内的一些位置和高度都连续的手办。也即，客人要买一个区间 [l,r](l≤k≤r) 中的所有手办，并且满足 p_l,p_l+1,…,p_r 在排序后相邻值之差均为 1。蔡德仁希望你帮他算出卖出这些手办后可能获得的最小利润。
2.行情变化，第 k 个手办的利润变成 v。
n<=200000,3s

```

##### 分析

首先建析合树（参考WCppt

只要建出析合树，就可以扫描线处理离线的问题

在修改的时候，一个析点的答案是很好维护的

然而对于合点，每一次都要重新计算，复杂度过大

首先，对于每一个合点使用线段树维护前缀后缀和的min和max用来快速计算合点的答案

考虑析合树上树链剖分，对于每一个合点，记录重儿子的答案，这样询问复杂度降为两个log

考虑修改，对于一个点，如果修改的是它的重儿子，那么可以直接加，否则在对应线段树上修改，复杂度两个log

为了保证询问时节点重儿子的权值是正确的，每次询问跳轻链时需要先查一下重儿子权值

具体实现有较多细节

总复杂度$O(nlog^2n)$

##### 代码

```cpp
#include<cstdio>
#include<algorithm>
#include<vector>
using namespace std;
#define N 400050
int v[N][2],fa[N],head[N],cnt,dep[N],ct,st[N],rb,mx[N],mn[N],sz[N],n,m,a,b,c,is[N],lasch[N],l[N];
long long su[N];
vector<int> ch[N];
struct edge{int t,next;}ed[N*2];
void adde(int f,int t){if(!f)return;ed[++cnt]=(edge){t,head[f]};head[f]=cnt;ed[++cnt]=(edge){f,head[t]};head[t]=cnt;}
void solve(int x)
{
    if(!rb){st[++rb]=x;return;}
    int tp=st[rb];
    if(!lasch[tp])st[rb]=++ct,lasch[ct]=tp,fa[tp]=ct,ch[ct].push_back(tp),su[ct]+=su[tp],is[ct]=1,sz[ct]=sz[tp],v[ct][0]=v[tp][0],v[ct][1]=v[tp][1],mn[ct]=mn[tp],mx[ct]=mx[tp],tp=ct;
    else tp=lasch[tp];
    if(mx[x]==mn[tp]-1||mn[x]-1==mx[tp])
    {
        tp=st[rb];
        ch[tp].push_back(x);su[tp]+=su[x];
        fa[x]=tp;lasch[tp]=x;mn[tp]=min(mn[tp],mn[x]);mx[tp]=max(mx[tp],mx[x]);
        v[tp][0]=min(v[tp][0],v[x][0]);v[tp][1]=max(v[tp][1],v[x][1]);sz[tp]+=sz[x];
        rb--;solve(tp);
        return;
    }
    int m1=mx[x],m2=mn[x],s=sz[x];
    if(v[st[rb]][0]>=l[v[x][1]])
    for(int i=rb;i>=1;i--)
    {
        m1=max(m1,mx[st[i]]),m2=min(m2,mn[st[i]]),s+=sz[st[i]];
        if(m1-m2+1==s)
        {
            int s=++ct;
            v[s][0]=v[x][0],v[s][1]=v[x][1],mn[s]=mn[x],mx[s]=mx[x],sz[s]=sz[x];fa[x]=s;lasch[s]=x;
            if(i==rb)is[s]=1;
            for(int j=rb;j>=i;j--)
            {
                v[s][0]=min(v[s][0],v[st[j]][0]);v[s][1]=max(v[s][1],v[st[j]][1]);mx[s]=max(mx[s],mx[st[j]]);mn[s]=min(mn[s],mn[st[j]]);sz[s]+=sz[st[j]];
                fa[st[j]]=s;
            }
            for(int j=i;j<=rb;j++)ch[s].push_back(st[j]),su[s]+=su[st[j]];
            ch[s].push_back(x);su[s]+=su[x];
            rb=i-1;
            solve(s);
            return;
        }
    }
    st[++rb]=x;
}
struct segt{
    struct node{int l,r;long long lz,mn,mx;}e[N*4];
    void pushup(int x){e[x].mn=min(e[x<<1].mn,e[x<<1|1].mn);e[x].mx=max(e[x<<1].mx,e[x<<1|1].mx);}
    void pushdown(int x){if(e[x].lz)e[x<<1].lz+=e[x].lz,e[x<<1].mn+=e[x].lz,e[x<<1|1].mn+=e[x].lz,e[x<<1|1].lz+=e[x].lz,e[x<<1].mx+=e[x].lz,e[x<<1|1].mx+=e[x].lz,e[x].lz=0;}
    void build(int x,int l,int r){e[x].l=l;e[x].r=r;if(e[x].l==e[x].r){e[x].mn=e[x].mx=e[x].l;e[x].lz=0;return;}int mid=(l+r)>>1;build(x<<1,l,mid);build(x<<1|1,mid+1,r);pushup(x);}
    void modify(int x,int l,int r,int s){if(e[x].l==l&&e[x].r==r){e[x].lz+=s;e[x].mn+=s;e[x].mx+=s;return;}pushdown(x);int mid=(e[x].l+e[x].r)>>1;if(mid>=r)modify(x<<1,l,r,s);else if(mid<l)modify(x<<1|1,l,r,s);else modify(x<<1,l,mid,s),modify(x<<1|1,mid+1,r,s);pushup(x);}
    int query(int x){if(e[x].l==e[x].r)return e[x].mn==0?e[x].l:-1;pushdown(x);if(e[x<<1].mn<=0&&e[x<<1].mx>=0)return query(x<<1);else return query(x<<1|1);}
}t1;
int st2[N],sr=0,st3[N],sr2=0,vl[N],vl2[N];
void getl()
{
    t1.build(1,1,n);
    for(int i=1;i<=n;i++)
    {
        t1.modify(1,1,n,-1);
        int ls=i-1;
        while(sr>0&&mx[i]>mx[st2[sr]])
        {
            t1.modify(1,vl[sr],ls,-mx[st2[sr]]+mx[i]);
            ls=vl[sr]-1,sr--;
        }
        st2[++sr]=i;vl[sr]=vl[sr]?vl[sr]:i;vl[sr+1]=i+1;
        ls=i-1;
        while(sr2>0&&mx[i]<mx[st3[sr2]])
        {
            t1.modify(1,vl2[sr2],ls,mx[st3[sr2]]-mx[i]);
            ls=vl2[sr2]-1,sr2--;
        }
        st3[++sr2]=i;vl2[sr2]=vl2[sr2]?vl2[sr2]:i;vl2[sr2+1]=i+1;
        l[i]=t1.query(1);
    }
}
struct node2{int l,r;long long lz,mn,mx;}e[N*40];
int ct2=0;
struct segt2{
    int lb;
    void init(int n){if(!n)return;lb=ct2+1;ct2+=n*4;build(1,1,n);}
    void pushup(int x){e[x+lb].mn=min(e[x*2+lb].mn,e[x*2+1+lb].mn);e[x+lb].mx=max(e[x*2+lb].mx,e[x*2+1+lb].mx);}
    void pushdown(int x){if(e[x+lb].lz)e[x*2+lb].lz+=e[x+lb].lz,e[x*2+lb].mn+=e[x+lb].lz,e[x*2+1+lb].mn+=e[x+lb].lz,e[x*2+1+lb].lz+=e[x+lb].lz,e[x*2+lb].mx+=e[x+lb].lz,e[x*2+1+lb].mx+=e[x+lb].lz,e[x+lb].lz=0;}
    void build(int x,int l,int r){e[x+lb].l=l;e[x+lb].r=r;if(e[x+lb].l==e[x+lb].r){e[x+lb].mn=e[x+lb].mx=0;e[x+lb].lz=0;return;}int mid=(l+r)>>1;build(x*2,l,mid);build(x*2+1,mid+1,r);pushup(x);}
    void modify(int x,int l,int r,int s){if(e[x+lb].l==l&&e[x+lb].r==r){e[x+lb].lz+=s;e[x+lb].mn+=s;e[x+lb].mx+=s;return;}pushdown(x);int mid=(e[x+lb].l+e[x+lb].r)>>1;if(mid>=r)modify(x*2,l,r,s);else if(mid<l)modify(x*2+1,l,r,s);else modify(x*2,l,mid,s),modify(x*2+1,mid+1,r,s);pushup(x);}
    int query1(int x,int l,int r){if(l>r||l<e[x+lb].l||r>e[x+lb].r)return 0;if(e[x+lb].l==l&&e[x+lb].r==r)return e[x+lb].mx;pushdown(x);int mid=(e[x+lb].l+e[x+lb].r)>>1;if(mid>=r)return query1(x*2,l,r);else if(mid<l)return query1(x*2+1,l,r);else return max(query1(x*2,l,mid),query1(x*2+1,mid+1,r));}
}hldt,tr[N],hldt2,tr2[N];
struct HLD{
    int sz[N],son[N],tp[N],id[N],tid[N],ct,son2[N],id2[N],las[N],fa[N];
    void dfs1(int u,int f){fa[u]=f;sz[u]=1;for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=f){dfs1(ed[i].t,u);if(sz[ed[i].t]>sz[son[u]])son[u]=ed[i].t;sz[u]+=sz[ed[i].t];}}
    void dfs2(int u,int fa,int v){tp[u]=v;id[u]=++ct;tid[ct]=u;if(son[u])dfs2(son[u],u,v);for(int i=head[u];i;i=ed[i].next)if(ed[i].t!=fa&&ed[i].t!=son[u])dfs2(ed[i].t,u,ed[i].t);}
    void init(int n)
    {
        dfs1(n,0);dfs2(n,0,n);is[0]=1;
        hldt.init(n);hldt2.init(n);
        for(int i=1;i<=n;i++)
        {
            las[i]=su[i];
            hldt2.modify(1,id[i],id[i],su[i]);
            tr[i].init(ch[i].size());
            tr2[i].init(ch[i].size());
            for(int j=1;j<=ch[i].size();j++)
            if(ch[i][j-1]==son[i])son2[i]=j;
            for(int j=1;j<=ch[i].size();j++)
            {
                id2[ch[i][j-1]]=j;
                tr[i].modify(1,j,ch[i].size(),su[ch[i][j-1]]);
                tr2[i].modify(1,1,j,su[ch[i][j-1]]);
            }
            if(ch[i].size())
            if(is[i])
            hldt.modify(1,id[i],id[i],-(su[son[i]]+tr[i].query1(1,son2[i]-1,son2[i]-1)-max(tr[i].query1(1,1,son2[i]-1),0)+tr2[i].query1(1,son2[i]+1,son2[i]+1)-max(tr2[i].query1(1,son2[i]+1,ch[i].size()),0)));
            else 
            hldt.modify(1,id[i],id[i],-(tr2[i].query1(1,1,1)));
        }
    }
    int getas(int x)
    {
        int i=x;
        if(x<=n)return su[x];
        if(is[x])return tr[i].query1(1,son2[i]-1,son2[i]-1)-max(tr[i].query1(1,1,son2[i]-1),0)+tr2[i].query1(1,son2[i]+1,son2[i]+1)-max(tr2[i].query1(1,son2[i]+1,ch[i].size()),0);
        return tr2[x].query1(1,1,1);
    }
    int getas2(int x)
    {
        int f=fa[x];
        int tp=hldt2.query1(1,id[son[f]],id[son[f]]);
        tr[f].modify(1,id2[son[f]],ch[f].size(),tp-las[son[f]]);
        tr2[f].modify(1,1,id2[son[f]],tp-las[son[f]]);
        las[son[f]]=tp;
        if(is[fa[x]])return hldt2.query1(1,id[x],id[x])+tr[f].query1(1,id2[x]-1,id2[x]-1)-max(tr[f].query1(1,1,id2[x]-1),0)+tr2[f].query1(1,id2[x]+1,id2[x]+1)-max(tr2[f].query1(1,id2[x]+1,ch[f].size()),0);
        else return tr2[f].query1(1,1,1);
    }
    void modify(int x,int ad)
    {
        int ad2=ad;
        while(x)
        {
            if(x!=tp[x])
            hldt.modify(1,id[tp[x]],id[fa[x]],-ad);
            hldt2.modify(1,id[tp[x]],id[x],ad);
            if(fa[tp[x]])
            {
                int sth=getas(fa[tp[x]]),id3=id2[tp[x]];
                tr[fa[tp[x]]].modify(1,id3,ch[fa[tp[x]]].size(),ad);
                tr2[fa[tp[x]]].modify(1,1,id3,ad);
                int sth2=getas(fa[tp[x]]);
                ad2=sth2-sth;
                hldt.modify(1,id[fa[tp[x]]],id[fa[tp[x]]],-ad2);
            }
            x=fa[tp[x]];
        }
    }
    int query(int x)
    {
        int as=su[x];
        while(x)
        {
            if(x!=tp[x])as=min(as,-hldt.query1(1,id[tp[x]],id[fa[x]]));
            if(fa[tp[x]])as=min(as,getas2(tp[x]));
            x=fa[tp[x]];
        }
        return as;
    }
}hld;
int main()
{
    scanf("%d",&n);ct=n;
    for(int i=1;i<=n;i++)scanf("%d",&mx[i]);getl();
    for(int i=1;i<=n;i++)
    mn[i]=mx[i],v[i][0]=v[i][1]=i,sz[i]=1,is[i]=1,scanf("%lld",&su[i]),solve(i);
    int tp2=++ct;for(int i=1;i<=rb;i++)fa[st[i]]=tp2,su[tp2]+=su[st[i]],ch[tp2].push_back(st[i]);v[tp2][0]=1;v[tp2][1]=n;
    for(int i=1;i<=tp2;i++)adde(i,fa[i]);
    hld.init(tp2);
    scanf("%d",&m);
    while(m--)
    {
        scanf("%d%d",&a,&b);
        if(a==1)printf("%d\n",hld.query(b));
        else scanf("%d",&c),hld.modify(b,c-su[b]),su[b]=c;
    }
}

```



