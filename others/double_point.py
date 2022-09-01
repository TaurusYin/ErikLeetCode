# https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
from typing import List


# https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
def removeDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    if len(nums) < 2: return len(nums)

    i, j = 0, 1
    while j < len(nums):
        if nums[i] == nums[j]:
            j += 1
        else:
            i += 1
            nums[i] = nums[j]
            j += 1
    return i + 1


# https://leetcode.cn/problems/remove-element/
def removeElement(self, nums: List[int], val: int) -> int:
    a = 0
    b = 0

    while a < len(nums):
        if nums[a] != val:
            nums[b] = nums[a]
            b += 1
        a += 1

    return b


# https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    low, high = 0, len(numbers) - 1
    while low < high:
        total = numbers[low] + numbers[high]
        if total == target:
            return [low + 1, high + 1]
        elif total < target:
            low += 1
        else:
            high -= 1

    return [-1, -1]


# https://leetcode.cn/problems/3sum/solution/by-s1ne-p3qs/
def threeSum(self, nums: List[int]) -> List[List[int]]:
    ans = []
    nums.sort()
    n = len(nums)
    for i in range(n):
        if nums[i] > 0: break
        if i and nums[i] == nums[i - 1]: continue
        l, r = i + 1, n - 1
        while l < r:
            sm = nums[i] + nums[l] + nums[r]
            if sm > 0:
                r -= 1
            elif sm < 0:
                l += 1
            else:
                ans.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]: l += 1
    return ans


def fourSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[List[int]]

    """
    return self.nSumTarget(sorted(nums), 4, 0, target)  # 调用nSumTarget函数，仅需要一行就可以搞定啦~这里n=4，所以可以得到四数之和的结果集


def nSumTarget(self, nums, n, start, target):
    sz = len(nums)
    res = []

    if n == 2:  # 基础是两数之和问题
        lo = start  # 因为我们已经对nums列表进行的递增排列，所以该操作可以避免取到之前取过的值
        hi = len(nums) - 1
        while lo < hi:  # 左右双指针模型，双指针相向而行
            left, right = nums[lo], nums[hi]
            sums = nums[lo] + nums[hi]
            if sums < target:
                while lo < hi and nums[lo] == left:  # 避免因列表中存在多个相同数字，而使得结果集重复
                    lo += 1
            elif sums > target:
                while lo < hi and nums[hi] == right:
                    hi -= 1
            else:
                res.append([left, right])
                while lo < hi and nums[lo] == left:
                    lo += 1
                while lo < hi and nums[hi] == right:
                    hi -= 1
    else:
        for i in range(start, sz):  # 这里的start是为了避免取到之前取过的值
            for j in self.nSumTarget(nums, n - 1, i + 1, target - nums[i]):  # 递归至（n-1）Sum问题
                j.append(nums[i])
                if j not in res:  # 防止向res列表中添加已经找到的结果，避免重复
                    res.append(j)

    return res
