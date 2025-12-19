n,a,b=map(int,input().split())
plants=list(map(int,input().split()))
l=0
la=a
ltime=0
r=n-1
rb=b
rtime=0
def avail(i,cur):
    global plants
    return plants[i]<=cur
while l<=r:
    if l==r:
        if not avail(l,la) and not avail(r,rb):
            if la>=rb:
                ltime+=1
            else:
                rtime+=1
    else:
        if not avail(l,la):
            ltime+=1
            la=a
        if not avail(r,rb):
            rtime+=1
            rb=b
    la-=plants[l]
    rb-=plants[r]
    l+=1
    r-=1
print(ltime+rtime)


