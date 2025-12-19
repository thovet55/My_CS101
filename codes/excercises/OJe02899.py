line = []

for _ in range(5):
    l = input().strip()
    line.append(list(map(int,l.split())))
n,m = map(int,input().split())

if m>=5 or n>=5:
    print('error')
def search(s):
    t = int(-1)
    for j in range(5):
        if s in line[j]:
            t +=1
            break
    return t

if search(n)==-1 or search(m)==-1:
    print('error')
else:
    line[m],line[n] = line[n],line[m]
    for k in range(5):
        print(" ".join(map(str,line[k])))