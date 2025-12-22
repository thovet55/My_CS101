from collections import deque
def solve(n,m):
    q=deque([(n,'')])
    visited={n}
    while q:
        cur_n,cur_route=q.popleft()
        if cur_n==m:
            print(len(cur_route))
            print(cur_route)
            return
        #h:
        h_n=cur_n*3
        o_n=cur_n//2
        if h_n not in visited:
           q.append((h_n,cur_route+'H'))
           visited.add(h_n)
        if o_n not in visited:
            q.append((o_n,cur_route+'O'))
            visited.add(o_n)
a,b=map(int,input().split())
while a!=0 and b!=0:
    solve(a,b)
    a,b=map(int,input().split())