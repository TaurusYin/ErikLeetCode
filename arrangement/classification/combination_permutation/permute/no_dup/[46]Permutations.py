# Given an array nums of distinct integers, return all the possible 
# permutations. You can return the answer in any order. 
# 
#  
#  Example 1: 
#  Input: nums = [1,2,3]
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
#  
#  Example 2: 
#  Input: nums = [0,1]
# Output: [[0,1],[1,0]]
#  
#  Example 3: 
#  Input: nums = [1]
# Output: [[1]]
#  
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 6 
#  -10 <= nums[i] <= 10 
#  All the integers of nums are unique. 
#  
# 
#  Related Topics 数组 回溯 👍 2654 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def _permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(current=[]):
            # If the current permutation is of length `len(nums)`, add it to the result list.
            if len(current) == len(nums):
                result.append(current[:])
                return
            for num in nums:
                # Only consider numbers not already used in the current permutation.
                if num in current:
                    continue
                current.append(num)
                backtrack(current)
                current.pop()
        result = []
        backtrack()
        return result

    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(current=[]):
            if len(current) == len(nums):
                result.append(current[:])
                return

            for i in range(len(nums)):
                # If the number at index `i` is already used, skip it.
                if used[i]:
                    continue

                current.append(nums[i])
                used[i] = True  # Mark the number as used
                backtrack(current)
                used[i] = False  # Unmark the number after backtracking
                current.pop()

        result = []
        used = [False] * len(nums)  # Initialize the tracking array
        backtrack()
        return result

"""
nums = [1,2,3]
print(permute(nums))  # [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
# leetcode submit region end(Prohibit modification and deletion)
