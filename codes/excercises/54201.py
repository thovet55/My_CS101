class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        mat=[[0,0,0],[0,1,0],[1,1,1]]
        Y=len(mat)
        X=len(mat[0])
        INF=float('inf')
        import heapq
        def bfs(y,x):
            pq=[]
            matrix = [[INF] * X for _ in range(Y)]
            heapq.heappush(pq,(0,y,x))
            dy=[1,0,-1,0]
            dx=[0,1,0,-1]
            ans=INF
            while pq:
                time,cur_y,cur_x=heapq.heappop(pq)
                if mat[cur_y][cur_x]==0:
                    ans=min(ans,time)
                if time>=matrix[cur_y][cur_x] or time>=ans:
                    continue
                else:
                    matrix[cur_y][cur_x]=time
                for i in range(4):
                    ny,nx=cur_y+dy[i],cur_x+dx[i]
                    if 0<=nx<X and 0<=ny<Y:
                        heapq.heappush(pq,(time+1,ny,nx))
            return ans
        ANS=[[INF]*X for _ in range(Y)]
        for i in range(Y):
            for j in range(X):
                ANS[i][j]=bfs(i,j)
        return ANS