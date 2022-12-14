import collections
import heapq
from collections import defaultdict
from typing import List


def heap_sort(self, nums):
    i, l = 0, len(nums)
    self.nums = nums
    # 构造大顶堆，从非叶子节点开始倒序遍历，因此是l//2 -1 就是最后一个非叶子节点
    for i in range(l // 2 - 1, -1, -1):
        self.build_heap(i, l - 1)
    # 上面的循环完成了大顶堆的构造，那么就开始把根节点跟末尾节点交换，然后重新调整大顶堆
    for j in range(l - 1, -1, -1):
        nums[0], nums[j] = nums[j], nums[0]
        self.build_heap(0, j - 1)

    return nums


def build_heap(self, i, l):
    """构建大顶堆"""
    nums = self.nums
    left, right = 2 * i + 1, 2 * i + 2  ## 左右子节点的下标
    large_index = i
    if left <= l and nums[i] < nums[left]:
        large_index = left

    if right <= l and nums[left] < nums[right]:
        large_index = right

    # 通过上面跟左右节点比较后，得出三个元素之间较大的下标，如果较大下表不是父节点的下标，说明交换后需要重新调整大顶堆
    if large_index != i:
        nums[i], nums[large_index] = nums[large_index], nums[i]
        self.build_heap(large_index, l)


"""
拆顺子
给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个长度至少为 3 的子序列，其中每个子序列都由连续整数组成。
O(nlogn)使用最小堆，每次把x放到长度最小的x-1序列后面，最后判断，所有的最小值是否都大于等于3
字典的键为序列结尾数值，值为结尾为该数值的所有序列长度（以堆存储）。
更新方式：每遍历一个数，将该数加入能加入的长度最短的序列中，不能加入序列则新建一个序列；然后更新字典。
链接：https://leetcode.cn/problems/split-array-into-consecutive-subsequences/solution/zui-hao-li-jie-de-pythonban-ben-by-user2198v/
"""
def isPossible(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    chains = defaultdict(list)
    x = heapq
    for i in nums:
        if not chains[i - 1]:
            heapq.heappush(chains[i], 1)
        else:
            min_len = heapq.heappop(chains[i - 1])
            heapq.heappush(chains[i], min_len + 1)
        # print(chains)

    for _, chain in chains.items():
        if chain and chain[0] < 3:
            return False
    return True


isPossible(nums=[1, 2, 3, 3, 4, 5])

"""
O(n)
https://leetcode.cn/problems/split-array-into-consecutive-subsequences/solution/fen-ge-shu-zu-wei-lian-xu-zi-xu-lie-by-l-lbs5/
"""
def isPossible(self, nums: List[int]) -> bool:
    countMap = collections.Counter(nums) # 数组中的每个数字的剩余次数
    endMap = collections.Counter() # 数组中的每个数字作为结尾的子序列的数量

    for x in nums:
        if (count := countMap[x]) > 0:
            if (prevEndCount := endMap.get(x - 1, 0)) > 0:
                countMap[x] -= 1
                endMap[x - 1] = prevEndCount - 1
                endMap[x] += 1
            else:
                if (count1 := countMap.get(x + 1, 0)) > 0 and (count2 := countMap.get(x + 2, 0)) > 0:
                    countMap[x] -= 1
                    countMap[x + 1] -= 1
                    countMap[x + 2] -= 1
                    endMap[x + 2] += 1
                else:
                    return False

    return True


import heapq
from pprint import pprint

portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
pprint(cheap)
pprint(expensive)

import heapq

num1 = [32, 3, 5, 34, 54, 23, 132]
num2 = [23, 2, 12, 656, 324, 23, 54]
num1 = sorted(num1)
num2 = sorted(num2)
res = heapq.merge(num1, num2)
print(list(res))
