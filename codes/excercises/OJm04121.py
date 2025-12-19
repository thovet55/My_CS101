num = int(input().strip())
ans =[]

def stocking(n,p):
    max_left =[0]*n
    min_left = p[0]
    for i in range(0,n):
        min_left = min(min_left,p[i])
        max_left[i] = max(max_left[i],p[i]-min_left)
    max_earn_right = [0]*n
    max_right = p[n-1]
    for i in range(n-1,-1,-1):
        max_right = max(max_right,p[i])
        max_earn_right[i] = max(max_earn_right[i],max_right-p[i])
    earned = 0
    for i in range(n):
        earned=max(earned,max_left[i]+max_earn_right[i])
    return(earned)

for _ in range(num):
    days=int(input().strip())
    stock=[int(x) for x in input().split()]
    ans.append(stocking(days,stock))

for answer in ans:
    print(answer)