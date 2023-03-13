"""
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

Example 1:
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
Example 2:
Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]
"""


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()  # 对数组进行排序
        res = []
        n = len(nums)
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:  # 跳过重复元素
                continue
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:  # 跳过重复元素
                    continue
                left, right = j + 1, n - 1  # 左右指针
                while left < right:
                    s = nums[i] + nums[j] + nums[left] + nums[right]
                    if s < target:
                        left += 1
                        while left < right and nums[left] == nums[left - 1]:  # 跳过重复元素
                            left += 1
                    elif s > target:
                        right -= 1
                        while left < right and nums[right] == nums[right + 1]:  # 跳过重复元素
                            right -= 1
                    else:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]:  # 跳过重复元素
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:  # 跳过重复元素
                            right -= 1
        return res


class KSumSolution:
    from typing import List

    def twoSum(self, nums, start, target):
        res = []
        lo, hi = start, len(nums) - 1
        while lo < hi:
            left, right = nums[lo], nums[hi]
            s = left + right
            if s < target:
                while lo < hi and nums[lo] == left:
                    lo += 1
            elif s > target:
                while lo < hi and nums[hi] == right:
                    hi -= 1
            else:
                res.append([left, right])
                while lo < hi and nums[lo] == left:
                    lo += 1
                while lo < hi and nums[hi] == right:
                    hi -= 1
        return res

    def kSum(self, nums, start, k, target):
        n = len(nums)
        if k == 2:
            return self.twoSum(nums, start, target)
        res = []
        for i in range(start, n - k + 1):
            if i > start and nums[i] == nums[i - 1]:
                continue
            for sub in self.kSum(nums, i + 1, k - 1, target - nums[i]):
                res.append([nums[i]] + sub)
        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        return self.kSum(nums, 0, 4, target)