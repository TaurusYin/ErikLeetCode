# You are climbing a staircase. It takes n steps to reach the top. 
# 
#  Each time you can either climb 1 or 2 steps. In how many distinct ways can 
# you climb to the top? 
# 
#  
#  Example 1: 
# 
#  
# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
#  
# 
#  Example 2: 
# 
#  
# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= n <= 45 
#  
# 
#  Related Topics 记忆化搜索 数学 动态规划 👍 2925 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from functools import lru_cache


class Solution:
    def __init__(self):
        self.memo = {}

    @lru_cache()
    def climbStairs_recursive(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        # n = 2 : (1,1) + (2)
        # n = 3 : (1,1,1) + (2,1) + (1,2)
        # n = 4 ： f(n-2) + f(n-1)
        return self.climbStairs(n - 2) + self.climbStairs(n - 1)

    def climbStairs_memo(self, n: int) -> int:
        if n in self.memo:
            return self.memo[n]
        if n == 1:
            return 1
        if n == 2:
            return 2
        f_n_2 = self.climbStairs(n - 2)
        f_n_1 = self.climbStairs(n - 1)
        self.memo[n] = f_n_2 + f_n_1
        return self.memo[n]

    def climbStairs(self, n: int) -> int:
        # 用一个数组 dp 来记录到达每个阶梯的方案数
        dp = [0] * (n + 1)
        if n == 1:
            return 1
        # 初始状态：到达第一个阶梯的方案数为 1，到达第二个阶梯的方案数为 2
        dp[1] = 1
        dp[2] = 2
        # 状态转移方程：dp[i] = dp[i-1] + dp[i-2]
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        # 返回结果
        return dp[n]

# leetcode submit region end(Prohibit modification and deletion)
