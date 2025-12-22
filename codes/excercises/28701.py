n,k=map(int,input().split())
lst=list(map(int,input().split()))
lst.sort(reverse=True)
ans=0
s=sum(lst)
for i in range(n):
    if lst[i]>s/k:
        s-=lst[i]
        k-=1
    else:
        ans=s/k
print(f'{ans:.3f}')