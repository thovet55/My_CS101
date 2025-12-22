import sys


def solve():
    # 使用 split() 自动处理所有空格和换行
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    # 将剩下的数据两两分组
    processes = []
    for i in range(n):
        c = int(input_data[1 + 2 * i])
        w = int(input_data[2 + 2 * i])
        processes.append((c, w))

    # 核心贪心：按写文件时间 w 降序排列
    # 注意：千万不要按 compute 排序，那是干扰项
    processes.sort(key=lambda x: x[1], reverse=True)

    max_finish_time = 0
    current_cpu_sum = 0

    for c, w in processes:
        # CPU 时间是累加的（串行）
        current_cpu_sum += c
        # 当前进程完工时刻 = 它结束计算的时刻 + 它的写文件耗时
        finish_i = current_cpu_sum + w
        # 答案是所有进程完工时刻的最大值
        if finish_i > max_finish_time:
            max_finish_time = finish_i

    print(max_finish_time)


if __name__ == "__main__":
    solve()