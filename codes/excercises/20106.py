m,n,p=map(int,input().split())
import heapq
dy=[1,0,-1,0]
dx=[0,1,0,-1]
matrix=[]
for i in range(m):
    matrix.append(list(map(str,input().split())))
def dij(start_y,start_x,end_y,end_x):
    global matrix,dy,dx,m,n
    distance=[[float('inf')]*n for _ in range(m)]
    pq=[]
    heapq.heappush(pq,(0,start_y,start_x))
    distance[start_y][start_x] = 0
    while pq:
        step,y,x=heapq.heappop(pq)
        if step>distance[y][x]:
            continue
        if (y,x)==(end_y,end_x):
            print(step)
            return
        for i in range(4):
            ny=y+dy[i]
            nx=x+dx[i]
            if 0<=ny<m and 0<=nx<n and matrix[ny][nx]!='#':
                height = abs(int(matrix[ny][nx]) - int(matrix[y][x]))
                if distance[ny][nx]>distance[y][x]+height:
                    distance[ny][nx]=distance[y][x]+height
                    heapq.heappush(pq,(distance[ny][nx],ny,nx))
    print('NO')
    return
for i in range(p):
    starty,startx,endy,endx=map(int,input().split())
    if matrix[starty][startx]=='#' or matrix[endy][endx]=='#':
        print('NO')
    else:
        dij(starty,startx,endy,endx)