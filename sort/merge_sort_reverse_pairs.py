from typing import List

# https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/solution/4chong-jie-fa-yi-wang-da-jin-you-xu-shu-0nia5/
from sortedcontainers import SortedList


class Solution:
    def __init__(self):
        self.count = 0

    def reversePairs(self, nums: List[int]) -> int:
        def mergeSort(nums, low, high):
            if low == high:
                return
            mid = (low + high) >> 1
            mergeSort(nums, low, mid)
            mergeSort(nums, mid + 1, high)
            merge(nums, low, mid, high)

        def merge(nums, low, mid, high):
            i, j, tmp = low, mid + 1, []
            while i <= mid and j <= high:
                # print('{},{},{}'.format(nums[l:mid+1],nums[mid+1:r+1],tmp))
                if nums[i] <= nums[j]:
                    tmp.append(nums[i])
                    i += 1
                else:
                    tmp.append(nums[j])
                    j += 1
                    self.count += ((mid - i) + 1)
            if i <= mid:
                tmp.extend(nums[i:mid + 1])
            if j <= high:
                tmp.extend(nums[j:high + 1])
            nums[low:high + 1] = tmp

        if nums:
            mergeSort(nums, 0, len(nums) - 1)
            return self.count
        else:
            return 0

    def bruteForce(self, nums):
        count = 0
        for i in range(0, len(nums)):
            for j in range(i + 1, len(nums)):
                print('i:{},j:{}'.format(i, j))
                if nums[i] > nums[j]:
                    count += 1
        return count

    def reversePairs(self, nums: List[int]) -> int:
        from sortedcontainers import SortedList
        n = len(nums)
        sl = SortedList()
        ans = 0
        for i in range(n - 1, -1, -1):
            insert_value = nums[i]  # 反向遍历
            cnt = sl.bisect_left(insert_value)  # 找到右边比当前值小的元素个数
            ans += cnt  # 记入答案
            sl.add(nums[i])  # 将当前值加入有序数组中
        return ans

    """
    给你一个整数数组 nums ，按要求返回一个新数组 counts 。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
链接：https://leetcode.cn/problems/count-of-smaller-numbers-after-self
    """
    def countSmaller(self, nums: List[int]) -> List[int]:
        lst = SortedList()
        n = len(nums)
        for i in range(n - 1, -1, -1):
            lst.add(nums[i]) # 从小到大，找左边小的
            nums[i] = lst.bisect_left(nums[i])
        return nums


if __name__ == '__main__':
    case_set = [
        [5, 4, 3, 2, 1],
        [7, 5, 6, 4],
        []
    ]
    for case in case_set:
        x = Solution().reversePairs(nums=case)
        print(x)
