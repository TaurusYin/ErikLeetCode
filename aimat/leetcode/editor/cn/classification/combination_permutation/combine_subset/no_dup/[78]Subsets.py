# Given an integer array nums of unique elements, return all possible subsets (
# the power set). 
# 
#  The solution set must not contain duplicate subsets. Return the solution in 
# any order. 
# 
#  
#  Example 1: 
# 
#  
# Input: nums = [1,2,3]
# Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
#  
# 
#  Example 2: 
# 
#  
# Input: nums = [0]
# Output: [[],[0]]
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 10 
#  -10 <= nums[i] <= 10 
#  All the numbers of nums are unique. 
#  
# 
#  Related Topics 位运算 数组 回溯 👍 2120 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start, path):
            result.append(list(path))
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1, path)
                path.pop()
        result = []
        backtrack(0, [])
        return result
        #return sorted(result, key=lambda x: (len(x), x))

# leetcode submit region end(Prohibit modification and deletion)
