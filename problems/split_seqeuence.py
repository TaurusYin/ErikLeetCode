
"""
拆顺子
给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个长度至少为 3 的子序列，其中每个子序列都由连续整数组成。
O(nlogn)使用最小堆，每次把x放到长度最小的x-1序列后面，最后判断，所有的最小值是否都大于等于3
字典的键为序列结尾数值，值为结尾为该数值的所有序列长度（以堆存储）。
更新方式：每遍历一个数，将该数加入能加入的长度最短的序列中，不能加入序列则新建一个序列；然后更新字典。
链接：https://leetcode.cn/problems/split-array-into-consecutive-subsequences/solution/zui-hao-li-jie-de-pythonban-ben-by-user2198v/
"""
import collections
import heapq
from typing import List


def isPossible(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    chains = collections.defaultdict(list)
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
