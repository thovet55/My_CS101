n=int(input().strip())
dices=[]
words=[]
for _ in range(4):
    dices.append(str(input().strip()))
for i in range(n):
    words.append(str(input().strip()))
ans=[False]*n
def spell(string,lst,idx):
    global dices,ans
    if len(string)>1:
        letter=string[0]
        for i in range(4):
            if letter in dices[i] and lst[i]:
                 lst[i]=False
                 spell(string[1:],lst,idx)
                 lst[i]=True
    else:
        for i in range(4):
            if string in dices[i] and lst[i]:
                 ans[idx]=True
                 return
for j in range(n):
        spell(words[j],[True]*4,j)
for _ in ans:
    if _:
        print('YES')
    else:
        print('NO')

