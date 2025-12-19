'''n=int(input().strip())
from collections import deque
def flood(starty,startx,height,matrix,y,x,cmd_y,cmd_x):
    if matrix[cmd_y][cmd_x]>=height:
        return False
    q=deque()
    q.append((starty,startx))
    visited=set()
    visited.add((starty,startx))
    dy=[-1,1,0,0]
    dx=[0,0,1,-1]
    while q:
        cury,curx=q.popleft()
        if (cury,curx)==(cmd_y,cmd_x):
            return True
        for i in range(4):
            ny,nx=cury+dy[i],curx+dx[i]
            if 0 <= ny <y and 0<= nx <x:
                if matrix[ny][nx]<height and (ny,nx) not in visited:
                    visited.add((ny,nx))
                    q.append((ny,nx))
    return False
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
    state=False
    for points in water:
        a,b=points
        a-=1
        b-=1
        height=matrix[a][b]
        state=flood(a,b,height,matrix,y,x,cordy,cordx)
        if state:
           print("Yes")
           return
    print("No")

for _ in range(n):
    solve()
'''