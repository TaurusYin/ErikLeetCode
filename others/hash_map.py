import collections
import heapq
from collections import defaultdict
from random import random, randint
from typing import List

#https://leetcode.cn/problems/insert-delete-getrandom-o1/
class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.valToIndex = {}

    def insert(self, val: int) -> bool:
        if val in self.valToIndex:
            return False
        self.valToIndex[val] = len(self.nums)
        self.nums.append(val)
        return True

    # 交换要移除的元素到末尾
    # 然后pop掉
    # 这里需要注意，除了在nums进行交换，还需要修改原数组中末尾元素在valToIndex的值（因为原末尾元素被交换到val所在的位置了）
    def remove(self, val: int) -> bool:
        if val not in self.valToIndex:
            return False
        # 记录val的索引
        index = self.valToIndex[val]
        # 修改原数组中末尾元素在valToIndex的索引
        self.valToIndex[self.nums[-1]] = index
        # 交换要删除的元素和末尾元素在nums中的位置
        self.nums[self.valToIndex.get(val)], self.nums[-1] = self.nums[-1], self.nums[self.valToIndex.get(val)]
        self.nums.pop()
        self.valToIndex.pop(val)
        return True

    def getRandom(self) -> int:
        # 下面两个都可以实现O(1)采样
        return random.choice(self.nums)

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# https://leetcode.cn/problems/random-pick-with-blacklist/submissions/
class BlackList:
    def __init__(self, n: int, blacklist: List[int]):
        self.dict = {}
        # 白名单长度
        self.white = n - len(blacklist)
        # 将黑名单的值先添加到字典
        for b in blacklist:
            self.dict[b] = 0  # 可以选取为任何值

        # 在黑名单区 要映射的指针
        last = n - 1
        for b in blacklist:
            # 黑名单中的值 已经在 黑名单的区间, 那么可以忽略
            if b >= self.white:
                continue
            # last对应的值已经在黑名单中
            while last in self.dict:
                last -= 1
            self.dict[b] = last
            last -= 1

    def pick(self) -> int:
        index = randint(0, self.white - 1)  # 在白名单部分随机挑选
        if index in self.dict:  # 如果在黑名单中, 那么就映射为白名单的值
            return self.dict[index]
        return index


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

