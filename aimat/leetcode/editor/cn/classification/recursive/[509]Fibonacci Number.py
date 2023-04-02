# The Fibonacci numbers, commonly denoted F(n) form a sequence, called the 
# Fibonacci sequence, such that each number is the sum of the two preceding ones, 
# starting from 0 and 1. That is, 
# 
#  
# F(0) = 0, F(1) = 1
# F(n) = F(n - 1) + F(n - 2), for n > 1.
#  
# 
#  Given n, calculate F(n). 
# 
#  
#  Example 1: 
# 
#  
# Input: n = 2
# Output: 1
# Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.
#  
# 
#  Example 2: 
# 
#  
# Input: n = 3
# Output: 2
# Explanation: F(3) = F(2) + F(1) = 1 + 1 = 2.
#  
# 
#  Example 3: 
# 
#  
# Input: n = 4
# Output: 3
# Explanation: F(4) = F(3) + F(2) = 2 + 1 = 3.
#  
# 
#  
#  Constraints: 
# 
#  
#  0 <= n <= 30 
#  
# 
#  Related Topics 递归 记忆化搜索 数学 动态规划 👍 617 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from functools import lru_cache
class Solution:
    def __init__(self):
        self.memo = {}
    @lru_cache()
    def fib_recursive(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        if n == 3:
            return 2
        return self.fib(n - 1) + self.fib(n - 2)

    def fib(self, n: int) -> int:
        if n in self.memo:
            return self.memo[n]
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        if n == 3:
            return 2
        f_n_1 = self.fib(n - 1)
        f_n_2 = self.fib(n - 2)
        return f_n_1 + f_n_2

# leetcode submit region end(Prohibit modification and deletion)
