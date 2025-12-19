t,m = map(int,input().split())
herbs = []
total = []
s = t
for i in range(m):
    data = input().split()
    herbs.append({'t':int(data[0]),'p':int(data[1])})

herbs.sort(key=lambda x:x['p'])

dp = [0]*(t+1)

for i in range(m):
    current_t = herbs[i]['t']
    current_p = herbs[i]['p']
    for j in range(t,current_t-1,-1):
        if dp[j]<dp[j-current_t]+current_p:
            dp[j] = dp[j-current_t]+current_p


max_value = dp[t]
print(max_value)