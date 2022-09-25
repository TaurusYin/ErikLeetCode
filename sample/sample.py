from bisect import bisect
from collections import defaultdict
from random import randrange, choice, random, randint
from typing import List

"""
打乱数组：Fisher-Yates 洗牌算法
考虑通过调整 \textit{waiting}waiting 的实现方式以优化方法一。
我们可以在移除 \textit{waiting}waiting 的第 kk 个元素时，将第 kk 个元素与数组的最后 11 个元素交换，然后移除交换后数组的最后 11 个元素，这样我们只需要 O(1)O(1) 的时间复杂度即可完成移除第 kk 个元素的操作。此时，被移除的交换后数组的最后 11 个元素即为我们根据随机下标获取的元素。
链接：https://leetcode.cn/problems/shuffle-an-array/solution/da-luan-shu-zu-by-leetcode-solution-og5u/
"""

def shuffle(self) -> List[int]:
    for i in range(len(self.nums)):
        j = randrange(i, len(self.nums))
        self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
    return self.nums


# https://leetcode.cn/problems/random-pick-index/
class ReservoirSampling:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        ans = cnt = 0
        for i, num in enumerate(self.nums):
            if num == target:
                cnt += 1  # 第 cnt 次遇到 target
                if randrange(cnt) == 0:
                    ans = i
        return ans

    # linked list https://leetcode.cn/problems/linked-list-random-node/solution/lian-biao-sui-ji-jie-dian-by-leetcode-so-x6it/
    def getRandom(self) -> int:
        node, i, ans = self.head, 1, 0
        while node:
            if randrange(i) == 0:  # 1/i 的概率选中（替换为答案）
                ans = node.val
            i += 1
            node = node.next
        return ans


class HashSampling:
    def __init__(self, nums: List[int]):
        self.pos = defaultdict(list)
        for i, num in enumerate(nums):
            self.pos[num].append(i)

    def pick(self, target: int) -> int:
        return choice(self.pos[target])

    def getRandom(self) -> int:
        self.arr = []
        while head:
            self.arr.append(head.val)
            head = head.next
        return choice(self.arr)


"""
按权重随机选择
对于 w = [1, 3]，挑选下标 0 的概率为 1 / (1 + 3) = 0.25 （即，25%），而选取下标 1 的概率为 3 / (1 + 3) = 0.75（即，75%）。
链接：https://leetcode.cn/problems/random-pick-with-weight
"""


class WeightedSampling:
    def __init__(self, w: List[int]):
        self.pre = [0]
        for num in w:
            self.pre.append(self.pre[-1] + num)

    def pickIndex(self) -> int:
        k = random.randint(1, self.pre[-1])
        return bisect.bisect_left(self.pre, k) - 1


"""
 如果在黑名单中, 那么就映射为白名单的值
给定一个整数 n 和一个 无重复 黑名单整数数组 blacklist 。设计一种算法，从 [0, n - 1] 范围内的任意整数中选取一个 未加入 黑名单 blacklist 的整数。任何在上述范围内且不在黑名单 blacklist 中的整数都应该有 同等的可能性 被返回。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/random-pick-with-blacklist
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


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


# https://leetcode.cn/problems/implement-rand10-using-rand7/
rand_list = [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]
# 转盘
num = 3


# 指针
def rand7():
    pass


def rand10(self):
    """
    :rtype: int
    """
    global num
    num = (num + rand7()) % 10
    # 随机前进rand7()步
    return rand_list[num]


def rand10(self):
    """
    :rtype: int
    """
    num = (rand7() - 1) * 7 + rand7()
    while num > 40:
        num = (rand7() - 1) * 7 + rand7()
    return 1 + (num - 1) % 10


import random


