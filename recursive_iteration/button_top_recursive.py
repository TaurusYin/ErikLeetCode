from functools import cache
from typing import Optional, List


class RecursiveFormat:
    def __init__(self):
        self.memo = {}
        return

    def button_top_sum(self, n):  # 返回值 f(参数)
        if n < 2: return n  # if (基本情况条件) return 基本情况的结果;
        child_sum = self.button_top_sum(n - 1)  # 修改参数， 返回值 = f(参数);
        res = child_sum + n  # 最终结果 = 根据参数与返回值计算
        return res  # return 最终结果;

    def top_button_sum(self, n, state_sum):
        if n < 2: return state_sum + 1  # if (基本情况条件) return 基本情况的结果;
        state_sum += n  # 中间变量 = 根据参数与中间变量重新计算
        return self.top_button_sum(n - 1, state_sum)

    def top_button_sum_iter(self, n, state_sum):
        """
        尾递归-> 转换成迭代
        :param n:
        :param state_sum:
        :return:
        """
        while True:
            if n < 2: return state_sum + 1  # if (基本情况条件) return 基本情况的结果;
            n -= 1
            state_sum += n  # 参数变化

    def multiply_iter(self, a, b):
        sum_value = 0
        while a > 0:
            a -= 1
            sum_value += b
        return sum_value

    def multiply_tb(self, a, b):
        if a == 0: return 0  # if (基本情况条件) return 基本情况的结果;
        a -= 1  # 修改参数
        child_multiply = self.multiply_tb(a, b)  # 返回值 = f(参数);
        return b + child_multiply  # 最终结果 = 根据参数与返回值计算

    def multiply_bt(self, a, b, state_value):
        if a == 0: return state_value  # if (基本情况条件) return 基本情况的结果;
        state_value += b;
        a -= 1
        return self.multiply_bt(a, b, state_value)  # 中间变量 = 根据参数与中间变量重新计算

    @cache
    def num_ways_bt(self, n):
        if n == 1: return 1
        if n == 2: return 2
        count = self.num_ways_bt(n - 1) + self.num_ways_bt(n - 2)
        return count

    def num_ways_memo_bt(self, n):
        if n == 1: return 1
        if n == 2: return 2
        if n in self.memo: return self.memo[n]  # 记忆化搜索
        count = self.num_ways_bt(n - 1) + self.num_ways_bt(n - 2)
        self.memo[n] = count  # 记忆化搜索
        return count

    def num_ways_tb(self, n, a, b):
        if n == 1: return 1
        if n == 2: return 2
        if n == 3: return a + b
        return self.num_ways_tb(n - 1, a + b, a)

    def num_ways_tb_for(self, n):
        a = 1; b = 1
        count = a + b
        while n >= 2:
            count = a + b
            b = a
            a = count
            n -= 1
        return count



if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5]
    s = RecursiveFormat()
    print(s.button_top_sum(n=5))
    print(s.top_button_sum(n=5, state_sum=0))
    print(s.multiply_iter(5, 4))
    print(s.multiply_tb(5, 4))
    print(s.multiply_bt(5, 4, state_value=0))
    print(s.num_ways_bt(10))
    print(s.num_ways_memo_bt(10))
    print(s.num_ways_tb(10, 2, 1))
    print(s.num_ways_tb_for(10))


