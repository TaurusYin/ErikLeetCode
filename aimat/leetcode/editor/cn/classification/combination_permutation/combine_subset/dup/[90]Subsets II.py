# Given an integer array nums that may contain duplicates, return all possible 
# subsets (the power set). 
# 
#  The solution set must not contain duplicate subsets. Return the solution in 
# any order. 
# 
#  
#  Example 1: 
#  Input: nums = [1,2,2]
# Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
#  
#  Example 2: 
#  Input: nums = [0]
# Output: [[],[0]]
#  
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 10 
#  -10 <= nums[i] <= 10 
#  
# 
#  Related Topics 位运算 数组 回溯 👍 1135 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start=0, current=[]):
            # We add all subsets to the result, not just those of certain lengths.
            result.append(current[:])

            for i in range(start, len(nums)):
                # If the current number is the same as the previous one, and we're not considering
                # the first element, then we skip to avoid duplicates.
                if i > start and nums[i] == nums[i - 1]:
                    continue

                current.append(nums[i])
                backtrack(i + 1, current)
                current.pop()

        nums.sort()  # Sort the array first to handle duplicates
        result = []
        backtrack()
        return result

"""
# Example usage
nums = [1,2,2]
print(subsetsWithDup(nums))  # [[],[1],[1,2],[1,2,2],[2],[2,2]]
The main idea behind skipping duplicates is this: After sorting the list, whenever we encounter a number that's the same as the previous number and we're not considering the first element, we can safely skip the current number. This is because the subset that would be generated using the current number would already have been generated using the previous number.
"""
# leetcode submit region end(Prohibit modification and deletion)
