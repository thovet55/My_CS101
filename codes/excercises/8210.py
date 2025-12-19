l,n,m=map(int,input().split())
from collections import deque
rock=deque()
for i in range(n):
    rock.append(int(input()))
rock.append(l)
def check(distance):
    cur=0
    d=0
    for x in rock:
        dist=x-cur
        if dist<distance:
            d+=1
            if d > m:
                return False
        else:
            cur=x
    return True
left=1
right=l
ans=0
while left<=right:
    mid=(left+right)//2
    if check(mid):
        left=mid+1
        ans=max(ans,mid)
    else:
        right=mid-1
print(ans)




