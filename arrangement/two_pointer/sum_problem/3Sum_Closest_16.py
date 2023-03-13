"""
Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.



Example 1:

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
Example 2:

Input: nums = [0,0,0], target = 1
Output: 0
Explanation: The sum that is closest to the target is 0. (0 + 0 + 0 = 0).


Constraints:

3 <= nums.length <= 500
-1000 <= nums[i] <= 1000
-104 <= target <= 104
"""


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()  # 排序
        closest = float('inf')
        for i in range(len(nums) - 2):
            # 如果当前数字和上一个数字相同，跳过
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            # 定义双指针l, r
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                diff = abs(s - target)
                if diff < abs(closest - target):
                    closest = s
                if s < target:  # 如果三数之和小于target，左指针右移
                    l += 1
                elif s > target:  # 如果三数之和大于target，右指针左移
                    r -= 1
                else:  # 如果三数之和等于target，直接返回
                    return target
        return closest
