lst=list(map(int,input().split(',')))
n=len(lst)
dp1=[float('-inf')]*n
dp2=[float('-inf')]*n
dp1[0]=lst[0]
dp2[0]=0
stata=False
for i in range(1,n):
    item=lst[i]
    if item>=0:
        stata=True
    dp1[i]=max(item,dp1[i-1]+item)
    dp2[i]=max(dp2[i-1]+item,dp1[i-1])
if stata:
   print(max(max(dp1),max(dp2)))
else:
    print(max(lst))