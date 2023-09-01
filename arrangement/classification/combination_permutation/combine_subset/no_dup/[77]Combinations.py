# Given two integers n and k, return all possible combinations of k numbers 
# chosen from the range [1, n]. 
# 
#  You may return the answer in any order. 
# 
#  
#  Example 1: 
# 
#  
# Input: n = 4, k = 2
# Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
# Explanation: There are 4 choose 2 = 6 total combinations.
# Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to 
# be the same combination.
#  
# 
#  Example 2: 
# 
#  
# Input: n = 1, k = 1
# Output: [[1]]
# Explanation: There is 1 choose 1 = 1 total combination.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= n <= 20 
#  1 <= k <= n 
#  
# 
#  Related Topics 回溯 👍 1467 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(start=1, current=[]):
            if len(current) == k:
                result.append(current[:])
                return

            for i in range(start, n + 1):
                current.append(i)
                backtrack(i + 1, current)
                current.pop()

        result = []
        backtrack()
        return result
# leetcode submit region end(Prohibit modification and deletion)
