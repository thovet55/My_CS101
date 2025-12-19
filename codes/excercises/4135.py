n,m=map(int,input().split())
lst=[]
for _ in range(n):
    lst.append(int(input()))
maximum=sum(lst)
minimum=max(lst)
def check(money):
    global maximum,lst,m,n
    cur=0
    month=1
    for i in lst:
        if i>money:
            return False
        if money<cur+i:
            cur=i
            month+=1
        else:
            cur+=i
    return month<=m
l=minimum
r=maximum
ans=10**18
while l<=r:
    mid=(l+r)//2
    if check(mid):
        r=mid-1
        ans=min(ans,mid)
    else:
        l=mid+1
print(ans)
