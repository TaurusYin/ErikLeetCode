# You are given a 0-indexed array of non-negative integers nums. For each 
# integer in nums, you must find its respective second greater integer. 
# 
#  The second greater integer of nums[i] is nums[j] such that: 
# 
#  
#  j > i 
#  nums[j] > nums[i] 
#  There exists exactly one index k such that nums[k] > nums[i] and i < k < j. 
#  
# 
#  If there is no such nums[j], the second greater integer is considered to be -
# 1. 
# 
#  
#  For example, in the array [1, 2, 4, 3], the second greater integer of 1 is 4,
#  2 is 3, and that of 3 and 4 is -1. 
#  
# 
#  Return an integer array answer, where answer[i] is the second greater 
# integer of nums[i]. 
# 
#  
#  Example 1: 
# 
#  
# Input: nums = [2,4,0,9,6]
# Output: [9,6,6,-1,-1]
# Explanation:
# 0th index: 4 is the first integer greater than 2, and 9 is the second integer 
# greater than 2, to the right of 2.
# 1st index: 9 is the first, and 6 is the second integer greater than 4, to the 
# right of 4.
# 2nd index: 9 is the first, and 6 is the second integer greater than 0, to the 
# right of 0.
# 3rd index: There is no integer greater than 9 to its right, so the second 
# greater integer is considered to be -1.
# 4th index: There is no integer greater than 6 to its right, so the second 
# greater integer is considered to be -1.
# Thus, we return [9,6,6,-1,-1].
#  
# 
#  Example 2: 
# 
#  
# Input: nums = [3,3]
# Output: [-1,-1]
# Explanation:
# We return [-1,-1] since neither integer has any integer greater than it.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 10⁵ 
#  0 <= nums[i] <= 10⁹ 
#  
# 
#  Related Topics 栈 数组 二分查找 排序 单调栈 堆（优先队列） 👍 23 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from collections import defaultdict
from collections import deque
from typing import List


class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        """
        第二个栈仿照第一个栈，需要tmp数组一次放到第二个栈里
        :param nums:
        :return:
        """
        stack = []
        second_stack = deque([])
        next_greater = defaultdict()
        second_greater = defaultdict()
        for index, val in enumerate(nums):
            tmp = deque([])
            while stack and stack[-1][1] < val:
                top_index, top_val = stack.pop()
                tmp.appendleft((top_index, top_val))
                next_greater[top_index] = val
            stack.append((index, val))
            # print(f"second_stack: {second_stack}")

            while second_stack and second_stack[-1][1] < val:
                second_top_index, second_top_val = second_stack.pop()
                second_greater[second_top_index] = val
            second_stack.extend(list(tmp))
            # print(second_greater)
        return [second_greater.get(index, -1) for index, num in enumerate(nums)]

Solution().secondGreaterElement(nums=[2, 4, 0, 9, 6])

# leetcode submit region end(Prohibit modification and deletion)
