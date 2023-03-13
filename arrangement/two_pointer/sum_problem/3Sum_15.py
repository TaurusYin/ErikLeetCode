"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation:
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.
Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.


Constraints:

3 <= nums.length <= 3000
-105 <= nums[i] <= 105
"""
from typing import List, Any
from collections import Counter


class Solution:
    # 定义一个函数，接受一个整数列表，返回一个二维列表
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()  # 排序
        res = []
        for i in range(len(nums) - 2):
            # 如果当前数字已经大于0，后面的数字都比它大，三数之和一定大于0，直接退出
            if nums[i] > 0:
                break
            # 如果当前数字和上一个数字相同，跳过
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            # 定义双指针l, r
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:  # 如果三数之和小于0，左指针右移
                    l += 1
                elif s > 0:  # 如果三数之和大于0，右指针左移
                    r -= 1
                else:  # 如果三数之和等于0，添加到结果集合中，并跳过相同的数字
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res

Solution().threeSum(nums=[-1, 0, 1, 2, -1, -4])
