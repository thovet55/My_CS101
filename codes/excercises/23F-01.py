n=int(input())
for _ in range(n):
    num = int(input())
    two_power=len(bin(num))-3
    add=(1+num)*num//2
    minus=2**(two_power+1)-1
    print(add-2*minus)