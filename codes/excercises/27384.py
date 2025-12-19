n,k=map(int,input().split())
input_=list(map(int,input().split()))
vote=[]
for i in range(n):
    t=input_[2*i]
    c=input_[2*i+1]
    vote.append([t,c])
vote.sort(key=lambda x:x[0])
lst=[int(x)-1 for x in input().split()]
s=set(lst)
total=[0]*314200
avail=False
if n!=k:
  max_others=0
else:
  max_others=-1
  avail=True
min_in=0
ans=0
time_before=0
min_s=[0]*(n+1)
min_s[0]=k
for i in range(n):
    t, c = vote[i][0], vote[i][1]
    c -= 1
    total[c] += 1
    if avail:
        ans+=(t-time_before)
    time_before=t
    if c not in s:
        max_others = max(max_others, total[c])
    else:
        min_s[total[c]-1]-=1
        min_s[total[c]]+=1
        if min_s[min_in]==0:
            min_in+=1
    if min_in>max_others:
        avail=True
    else:
        avail=False
print(ans)