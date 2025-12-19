lst=[]
mi=[]
def pop():
    global lst
    if lst:
        a=lst.pop()
        if mi and mi[-1]==a:
            mi.pop()
def push(n):
    global lst,mi
    lst.append(n)
    if mi and n<=mi[-1]:
        mi.append(n)
    if not mi:
        mi.append(n)
def m():
    global lst,mi
    if mi and lst:
      print(mi[-1])
import sys
input=sys.stdin.readline
string=str(input().strip())
while string!='':
    if string=='pop':
        pop()
    elif string=='min':
        m()
    else:
        ip=list(string.split())
        b=int(ip[-1])
        push(b)
    string=input().strip()