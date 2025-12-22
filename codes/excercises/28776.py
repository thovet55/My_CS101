n=int(input().strip())
lst=[]
king_l,king_r=map(int,input().split())
for i in range(n):
    lst.append(list(map(int,input().strip().split())))
lst.sort(key=lambda x:x[0]*x[1])
times=king_l
max_=0
for item in lst:
    max_=max(max_,times//item[1])
    times=times*item[0]
print(max_)