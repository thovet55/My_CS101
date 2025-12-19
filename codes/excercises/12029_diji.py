n=int(input().strip())
import heapq
def diji(start_y,start_x,matrix,y,x):
    pq=[]
    INF=float('inf')
    ans=[[INF]*x for _ in range(y)]
    height=matrix[start_y][start_x]
    dy=[-1,1,0,0]
    dx=[0,0,-1,1]
    ans[start_y][start_x]=height
    heapq.heappush(pq,(height,start_y,start_x))
    while pq:
          h,Y,X=heapq.heappop(pq)
          if h>ans[Y][X]:
              continue
          for i in range(4):
              ny,nx=Y+dy[i],X+dx[i]
              if 0<=ny<y and 0<=nx<x:
                  newh=max(h,matrix[ny][nx])
                  if newh<ans[ny][nx]:
                     ans[ny][nx]=newh
                     heapq.heappush(pq,(newh,ny,nx))
    return ans
def solve():
    y,x=map(int,input().strip().split())
    matrix=[]
    for i in range(y):
        matrix.append(list(map(int,input().strip().split())))
    cordy,cordx=map(int,input().strip().split())
    cordy-=1
    cordx-=1
    safe=matrix[cordy][cordx]
    p=int(input().strip())
    water=[]
    for i in range(p):
        a,b=map(int,input().strip().split())
        water.append((a,b))
    ans=diji(cordy,cordx,matrix,y,x)
    for waters in water:
        a,b=waters
        a-=1
        b-=1
        if ans[a][b]<matrix[a][b]:
            print('Yes')
            return
    print('No')
for _ in range(n):
    solve()