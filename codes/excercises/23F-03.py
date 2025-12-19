import math
s=str(input().strip())
m=math.floor(math.log2(len(s)))
l=0
r=m
ans=[]
while l<r:
    ans.append(s[2**l-1])
    ans.append(s[2**r-1])
    l+=1
    r-=1
if l==r:
    ans.append(s[2**l-1])
print(''.join(ans))