import math
mi=25
def H(num):
    return num*3
def O(num):
    return math.floor(num/2)
def find(n,m,string):
    global mi,ans
    if len(string)>mi:
        return
    elif n==m:
        ans.append(string)
        mi=min(mi,len(string))
        return
    else:
        if n!=1:
           find(O(n),m,string+'O')
        find(H(n),m,string+'H')
a,b=map(int,input().split())
while (a,b)!=(0,0):
    mi=25
    ans=[]
    if a==b:
        print(0)
    else:
      find(a,b,'')
      print(mi)
      ans.sort(key=lambda x:len(x),reverse=True)
      cur=ans.pop()
      ans_=cur
      while len(cur)<=mi:
          if cur<ans_:
            ans_=cur
          cur=ans.pop()
      print(ans_)
    a,b=map(int,input().split())
