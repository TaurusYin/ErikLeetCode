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

