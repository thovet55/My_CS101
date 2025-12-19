m=int(input().strip())
n=int(input().strip())
from functools import cmp_to_key
numbers=list(map(str,input().split()))
def compare(x, y):
    if x + y > y + x:
        return 1
    elif x + y < y + x:
        return -1
    else:
        return 0
numbers.sort(key=cmp_to_key(compare), reverse=True)
dp=[0]*(m+1)
for number in numbers:
    length=len(number)
    for i in range(m,length-1,-1):
        if dp[i-length]==0 and i-length!=0:
           continue
        dp[i]=max(dp[i],int(str(dp[i-length])+number))
print(max(dp))