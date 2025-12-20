# Review & Basics

## 1. 高级迭代与循环技巧 (Iteration)

### Enumerate (带索引遍历)

在遍历列表时同时获取索引和值，比 `range(len(lst))` 更 Pythonic。

```python
items = ['a', 'b', 'c']

# 基础用法
for index, value in enumerate(items):
    print(f"{index}: {value}")

# start 参数：索引从 1 开始计数（常用）
for index, value in enumerate(items, start=1):
    print(f"第 {index} 个是 {value}")
```

### Zip (并行遍历)

同时遍历多个序列。

```python
names = ['Alice', 'Bob']
scores = [95, 88]

# 组合遍历
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# 解压缩 (Unzip)
zipped = [(1, 'a'), (2, 'b')]
nums, chars = zip(*zipped)  # nums=(1, 2), chars=('a', 'b')
```

### 列表推导式 (List Comprehension)

考试中快速生成列表的神器。

```python
# [表达式 for 变量 in 列表 if 条件]
nums = [1, 2, 3, 4, 5]
squares = [x**2 for x in nums if x % 2 == 0] # [4, 16]

# 二维数组初始化 (避免浅拷贝陷阱！)
# 创建 m 行 n 列的 0 矩阵
matrix = [[0] * n for _ in range(m)] 
```

## 2. 核心数据结构模块

### Heapq (堆/优先队列)

注意：Python 的 heapq 默认是小顶堆 (Min-Heap)。

如果要用大顶堆，通常存入负数 (-value, item)。

```python
import heapq

nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # O(n) 建堆，原地修改

# 常用操作 O(log n)
min_val = heapq.heappop(nums)  # 弹出最小值 (1)
heapq.heappush(nums, 2)        # 插入新值

# 访问堆顶 (不弹出)
top = nums[0]

# 获取最大的 n 个元素 / 最小的 n 个元素
largest_3 = heapq.nlargest(3, [3, 1, 4, 1, 5]) # [5, 4, 3]
smallest_3 = heapq.nsmallest(3, [3, 1, 4, 1, 5])
```

### Deque (双端队列)

来自 `collections` 模块。比 list 更高效的头尾操作。

- List 在头部 insert/pop 是 $O(n)$。
    
- Deque 在头尾操作都是 $O(1)$。
    

```python
from collections import deque

d = deque([1, 2, 3])

d.append(4)      # 尾部添加
d.appendleft(0)  # 头部添加 [0, 1, 2, 3, 4]

pop_val = d.pop()      # 尾部弹出
pop_left = d.popleft() # 头部弹出 (BFS常用)

# 旋转 (循环移位)
d.rotate(1)  # 向右轮转 1 位
d.rotate(-1) # 向左轮转 1 位
```

### Counter (词频统计)

快速统计元素出现次数。

```python
from collections import Counter

s = "abracadabra"
cnt = Counter(s) 
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# 获取出现频率最高的 n 个元素
print(cnt.most_common(2)) # [('a', 5), ('b', 2)]

# 计数器相减
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 - c2) # Counter({'a': 2}) -> b消失因为 1-2 < 0
```

### Defaultdict (默认字典)

处理字典键不存在的情况（图论建图常用）。

```python
from collections import defaultdict

# 默认值为 int 的 0
freq = defaultdict(int)
freq['apple'] += 1 

# 默认值为列表 (构建邻接表)
graph = defaultdict(list)
edges = [(1, 2), (1, 3), (2, 4)]
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u) # 无向图
```

## 3. 排序与 Lambda (Sorting)

### 自定义排序

`sort()` 是原地排序，`sorted()` 返回新列表。

```python
data = [(1, 'apple'), (3, 'banana'), (2, 'cherry')]

# 基础排序
data.sort() 

# Lambda 排序：按第二个元素长度排序
data.sort(key=lambda x: len(x[1]))

# 多重排序：先按分数降序，分数相同按名字升序
students = [('Alice', 90), ('Bob', 80), ('Charlie', 90)]
#技巧：数值加负号实现降序
students.sort(key=lambda x: (-x[1], x[0])) 
```

## 4. 字符串与常用函数

### 字符串魔法

```python
text = " Hello World "

# 去除首尾空白
clean = text.strip()

# 拆分与合并
parts = clean.split()  # 默认按空白字符切分
joined = "-".join(parts) # "Hello-World"

# 字符判断
"123".isdigit() # True
"abc".isalpha() # True
"a12".isalnum() # True (字母或数字)

# 格式化填充 (例如：输出固定宽度数字)
x = 5
print(f"{x:03d}") # "005"
```

### 数学与进制

```python
# 进制转换
bin(10)  # '0b1010'
hex(15)  # '0xf'
int('1010', 2) # 二进制转十进制 -> 10

# 除法
q, r = divmod(10, 3) # 商3，余1

# 快速幂 (计算 base^exp % mod)
res = pow(base, exp, mod) 

# 绝对值与最值
abs(-5)
max(1, 2, key=abs) # 按绝对值比较
```

## 5. 考试常用技巧 (Tips)

### 输入输出 (IO)

题目输入量大时，使用 `sys.stdin` 比 `input()` 快。

```python
import sys

# 读取所有行
lines = sys.stdin.readlines()

# 循环读取
for line in sys.stdin:
    line = line.strip()
    if not line: break
    # process...
```

### 递归深度

Python 默认递归深度限制为 1000，DFS 题目容易 RE (Runtime Error)。

```python
import sys
sys.setrecursionlimit(20000) # 调大限制
```

### Lru Cache

对于可哈希的函数，用内存省时间

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def function(n):
    #自定义函数
```

### 常用常量

```python
float('inf')  # 正无穷 (用于找最小值初始值)
float('-inf') # 负无穷
```

## 6. 正则表达式 (Regex)

### 1. 基础语法
| 符号 | 描述 | 示例 |
| :--- | :--- | :--- |
| `.` | 匹配**除换行符外**的任意字符 | `a.c` -> abc |
| `^` / `$` | 匹配字符串的**开头** / **结尾** | `^Hi` / `bye$` |
| `*` / `+` | 重复 **0次及以上** / **1次及以上** | `ab*`, `ab+` |
| `?` | 重复 **0次或1次**（非贪婪模式的基础） | `ab?` |
| `{n,m}` | 重复 **n 到 m 次** | `\d{3,5}` |
| `[]` | 字符集，匹配括号内任意一个字符 | `[a-zA-Z]` |
| `|` | 或逻辑 | `cat|dog` |
| `()` | **分组**，用于提取特定部分 | `(\d+)-(.*)` |
| `\` | 转义字符 | `\.`, `\*`, `\\` |

### 2. 预定义序列 (常用)
- `\d`: 数字 `[0-9]` (Digit)
- `\D`: 非数字
- `\w`: 单词字符 `[a-zA-Z0-9_]` (Word)
- `\W`: 非单词字符
- `\s`: 空白字符（空格、Tab、换行） (Space)
- `\S`: 非空白字符
- `\b`: 单词边界 (Boundary)

---

### 3. Python `re` 模块函数

> [!WARNING] **match vs search**
> - `re.match()`: **必须从字符串第一个字符开始匹配**，否则返回 None。
> - `re.search()`: **扫描整个字符串**，返回第一个成功的匹配。

```python
import re

text = "Rank 01: Chemistry"

# 1. re.search(pattern, string) -> 返回 Match 对象或 None
m = re.search(r'\d+', text)
if m:
    print(m.group())  # '01'

# 2. re.findall(pattern, string) -> 返回 字符串列表 (最常用)
items = re.findall(r'\w+', text) # ['Rank', '01', 'Chemistry']

# 3. re.sub(pattern, repl, string) -> 替换
new_text = re.sub(r'\d+', '99', text) # "Rank 99: Chemistry"

# 4. re.split(pattern, string) -> 分割
parts = re.split(r':\s*', text) # ['Rank 01', 'Chemistry']
```

### 4. 高级技巧
#### A. 分组提取 (Groups)
```python
s = "2025-12-20"
m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
if m:
    year = m.group(1)   # '2025'
    all_parts = m.groups() # ('2025', '12', '20')
```
#### B. 贪婪 vs 非贪婪
- **贪婪 (默认)**: `<a><b>` 用 `<.*>` 匹配结果为 `<a><b>`
- **非贪婪 (`?`)**: `<a><b>` 用 `<.*?>` 匹配结果为 `<a>` (遇到第一个符合条件的就停止)
#### C. 修饰符 (Flags)
- `re.I`: 忽略大小写 (Ignorecase)
- `re.S`: 让 `.` 匹配包括换行符在内的所有字符 (Dotall)
- `re.M`: 多行模式 (Multiline)
```python
# 示例：忽略大小写搜索
re.findall(r'pku', 'PKU pku Pku', re.I) # ['PKU', 'pku', 'Pku']
```

### 5.  考试小贴士
1. **Raw String**: 写正则表达式时，务必在字符串前加 `r`（例如 `r'\d+'`），防止 Python 自身的转义干扰。 
2. **返回值检查**: `re.search` 和 `re.match` 返回的是对象，直接 `print` 会显示 `<re.Match object...>`，记得加 `.group()` 拿结果。 
3. **空匹配**: 注意 `*` 可以匹配 0 次，有时会导致 `findall` 返回一堆空字符串，考试时如果发现结果不对，检查是否应该用 `+`。

# Algorithm & Examples
## 第一部分：核心数据结构 (Core Data Structures)

### 1. 单调栈 (Monotonic Stack)

应用场景：寻找数组中左边/右边第一个比当前元素大/小的位置。

复杂度：$O(N)$

```python
def monotonic_stack(nums):
    # 示例：找右边第一个比自己大的元素索引
    res = [-1] * len(nums)
    stack = [] # 存储索引，保持栈内元素对应的值单调递减
    
    for i, x in enumerate(nums):
        # 当当前元素 x 大于栈顶元素时，说明栈顶元素遇到了右边第一个比它大的
        while stack and x > nums[stack[-1]]:
            idx = stack.pop()
            res[idx] = i
        stack.append(i)
    return res
```

### 2. 单调队列 (Monotonic Queue)

应用场景：滑动窗口中的最大值/最小值。

复杂度：$O(N)$

```python
from collections import deque

def max_sliding_window(nums, k):
    q = deque() # 存储索引，保持队内元素对应的值单调递减
    res = []
    
    for i, x in enumerate(nums):
        # 1. 入队：维护单调性，比当前元素小的都移出（因为它们不再可能是最大值）
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        
        # 2. 出队：判断队首是否滑出窗口
        if i - q[0] >= k:
            q.popleft()
            
        # 3. 记录：当窗口完全形成后开始记录
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```

### 3. 并查集 (Disjoint Set Union - DSU)

应用场景：动态连通性、最小生成树(Kruskal)、判断图中是否有环。

优化：路径压缩 (Path Compression) + 按秩合并 (Union by Rank)。

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.size = [1] * (n + 1) # 可选：维护集合大小

    def find(self, x):
        if self.parent[x] != x:
            # 路径压缩：直接挂到根节点下
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # 可选：按大小合并，小的挂大的
            if self.size[rootX] < self.size[rootY]:
                rootX, rootY = rootY, rootX
            self.parent[rootY] = rootX
            self.size[rootX] += self.size[rootY]
            return True # 合并成功
        return False # 已经在同一集合
```

### 4. 树状数组 (Binary Indexed Tree / Fenwick Tree)

应用场景：单点修改，区间查询前缀和。

复杂度：$O(\log N)$

```python
class FenwickTree:
    def __init__(self, n):
        self.tree = [0] * (n + 1)
    
    def lowbit(self, x):
        return x & (-x)
    
    def update(self, i, delta):
        while i < len(self.tree):
            self.tree[i] += delta
            i += self.lowbit(i)
            
    def query(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= self.lowbit(i)
        return s
```

## 第二部分：搜索算法 (Search Algorithms)

### 1. 深度优先搜索 (DFS) & 回溯 (Backtracking)

应用场景：全排列、组合、子集、棋盘问题（N皇后）、连通区域。

核心：选择 -> 递归 -> 撤销。

```python
def backtrack(path, start_index):
    # 1. 终止条件
    if is_goal(path):
        res.append(path[:]) # 注意深拷贝
        return

    # 2. 遍历选择列表
    for i in range(start_index, len(choices)):
        choice = choices[i]
        
        # 剪枝 (Pruning)
        if not is_valid(choice): continue
            
        path.append(choice) # 做选择
        backtrack(path, i + 1) # 递归
        path.pop() # 撤销选择 (回溯)
```

### 2. 广度优先搜索 (BFS)

应用场景：无权图最短路径、层序遍历、拓扑排序。

变体：

- **多源 BFS**：初始化队列包含所有起点。
    
- **双向 BFS**：起点终点同时搜，相遇即停。
    

```pyhon
def bfs(start, target):
    q = deque([(start, 0)]) # (node, step)
    visited = {start}
    
    while q:
        cur, step = q.popleft()
        if cur == target: return step
        
        for neighbor in get_neighbors(cur):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, step + 1))
    return -1
```

### 3. Dijkstra 算法

应用场景：非负权图的最短路径。

核心：贪心 + 优先队列 (Heap)。

```python
import heapq

def dijkstra(graph, start, n):
    # graph: 邻接表 {u: [(v, weight), ...]}
    dist = {node: float('inf') for node in range(1, n + 1)}
    dist[start] = 0
    pq = [(0, start)] # (cost, node) 小顶堆

    while pq:
        d, u = heapq.heappop(pq)
        
        # 懒删除：如果当前距离大于已记录的最短距离，跳过
        if d > dist[u]: continue
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

### 4. 拓扑排序 (Topological Sort)

应用场景：任务调度、课程表（判断有向图是否有环）。

方法：Kahn 算法 (入度表 + BFS)。

```python
def topological_sort(n, edges):
    in_degree = [0] * n
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        
    q = deque([i for i in range(n) if in_degree[i] == 0])
    res = []
    
    while q:
        u = q.popleft()
        res.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    return res if len(res) == n else [] # 有环返回空
```

## 第三部分：动态规划 (Dynamic Programming)

### 1. 背包问题 (Knapsack)

- **0-1 背包** (每件物品限用1次)：**倒序**遍历容量。
    
- **完全背包** (物品无限用)：正序遍历容量。
    

```python
# 0-1 背包模板
# dp[j] 表示容量为 j 时的最大价值
dp = [0] * (capacity + 1)
for i in range(num_items):
    for w in range(capacity, weights[i] - 1, -1): # 倒序！
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

### 2. 区间 DP

**应用场景**：石子合并、回文子序列。从小区间推到大区间。

```python
# 模板：枚举长度 -> 枚举左端点 -> 枚举分割点
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = ... # 状态转移
```

### 3. 最长公共子序列 (LCS)

```python
# dp[i][j] 表示 text1[:i] 和 text2[:j] 的 LCS
if text1[i-1] == text2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

### 4. 最长上升子序列 (LIS)

**优化**：贪心 + 二分查找，复杂度 $O(N \log N)$。

```python
from bisect import bisect_left
def length_of_lis(nums):
    tails = []
    for num in nums:
        # 找到第一个 >= num 的位置
        idx = bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num # 贪心：让上升序列增长得更慢
    return len(tails)
```

## 第四部分：贪心算法 (Greedy)

### 1. 区间调度

**策略**：按**结束时间**排序，优先选结束早的，留给后面的空间更大。

```python
intervals.sort(key=lambda x: x[1])
end, count = float('-inf'), 0
for s, e in intervals:
    if s >= end:
        count += 1
        end = e
```

### 2. 反悔贪心 (Regret Greedy)

场景：做选择时无法确定全局最优，先选，若后面遇到更好的，则“反悔”撤销之前最差的选择。

工具：堆 (Heap)。

```python
# 例子：做任务赚积分，任务有截止时间
# 按截止时间排序
tasks.sort(key=lambda x: x[0])
pq = [] # 小顶堆存任务价值
for deadline, profit in tasks:
    heapq.heappush(pq, profit)
    if len(pq) > deadline: # 任务数超过了当前截止时间，做不完了
        heapq.heappop(pq) # 放弃价值最小的任务
```

## 第五部分：二分与双指针 (Binary Search & Two Pointers)

### 1. 二分答案 (Binary Search on Answer)

**场景**：答案具有单调性（如：求最大化最小值，最小化最大值）。

```python
def check(mid):
    # 验证 mid 是否可行，通常是一个贪心过程
    # ...
    return True

l, r = min_val, max_val
ans = -1
while l <= r:
    mid = (l + r) // 2
    if check(mid):
        ans = mid
        l = mid + 1 # 尝试更大的解（根据题意调整）
    else:
        r = mid - 1
```

### 2. 滑动窗口 (Sliding Window)

**模板**：右指针进，条件不满足时左指针缩。

```python
l = 0
for r in range(len(s)):
    # 1. 进窗：加入 s[r]
    # ...
    
    # 2. 缩窗：当窗口不再满足条件时
    while not valid():
        # 移出 s[l]
        l += 1
        
    # 3. 统计：更新结果
    ans = max(ans, r - l + 1)
```

## 第六部分：常用数学与技巧

### 1. 前缀和与差分

- **前缀和**：$P[i] = P[i-1] + A[i]$。用于 $O(1)$ 求区间和。
    
- **差分数组**：$D[i] = A[i] - A[i-1]$。用于 $O(1)$ 区间修改（$[l, r]$ 加上 $v$ $\to$ $D[l]+=v, D[r+1]-=v$）。
    

### 2. 位运算技巧

- `x & (x - 1)`: 消除二进制最后一位的 1。
    
- `x & -x`: 获取二进制最后一位的 1 (lowbit)。
    
- `a ^ b`: 无进位加法（异或）。
    
- `x << 1`: 乘 2； `x >> 1`: 除 2。
    

### 3. 快速幂 (Fast Power)

**场景**：求 $a^b \pmod m$。

```python
def fast_pow(base, power, mod):
    res = 1
    while power > 0:
        if power % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        power //= 2
    return res
```

### 4. 字符串匹配 (KMP)

**核心**：Next 数组（最长前后缀匹配长度）。

```python
def get_next(p):
    nxt = [0] * len(p)
    j = 0
    for i in range(1, len(p)):
        while j > 0 and p[i] != p[j]:
            j = nxt[j-1]
        if p[i] == p[j]:
            j += 1
        nxt[i] = j
    return nxt
```

## 第七部分：考场策略总结

1. **看数据范围**：
    
    - $N \le 20$: DFS / 状态压缩 DP ($O(2^N)$)
        
    - $N \le 100$: Floyd / DP ($O(N^3)$)
        
    - $N \le 1000$: DP / Dijkstra ($O(N^2)$)
        
    - $N \le 10^5$: 二分 / 贪心 / 堆 / 单调栈 ($O(N \log N)$ 或 $O(N)$)
        
2. **由易到难**：先想暴力搜索，再想记忆化搜索，最后推 DP 状态方程。
    
3. **边界检查**：空数组、只有一个元素、最大值/最小值溢出、除零错误。

----
# Credit
Credit to **Gemini 3 Pro**