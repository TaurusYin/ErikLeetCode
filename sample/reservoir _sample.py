import random
from collections import Counter


class ReservoirSample(object):
    def __init__(self, size):
        self._size = size
        self._counter = 0
        self._sample = []

    def feed(self, item):
        self._counter += 1
        # 第i个元素（i <= k），直接进入池中
        if len(self._sample) < self._size:
            self._sample.append(item)
            return self._sample
        # 第i个元素（i > k），以k / i的概率进入池中
        rand_int = random.randint(1, self._counter)
        if rand_int <= self._size:
            self._sample[rand_int - 1] = item
        return self._sample

    def test_reservoir_sample(self):
        samples = []
        for i in range(10000):
            sample = []
            rs = ReservoirSample(3)
            for item in range(1, 11):
                sample = rs.feed(item)
            samples.extend(sample)
        r = Counter(samples)
        print(r)


"""
给你一个单链表，随机选择链表的一个节点，并返回相应的节点值。每个节点 被选中的概率一样 。
输入
["Solution", "getRandom", "getRandom", "getRandom", "getRandom", "getRandom"]
[[[1, 2, 3]], [], [], [], [], []]
输出
[null, 1, 3, 2, 2, 3]
Solution(ListNode head) 使用整数数组初始化对象。
int getRandom() 从链表中随机选择一个节点并返回该节点的值。链表中所有节点被选中的概率相等。
链接：https://leetcode.cn/problems/linked-list-random-node
"""
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def getRandom(self) -> int:
        node, i, ans = self.head, 1, 0
        while node:
            if randrange(i) == 0:  # 1/i 的概率选中（替换为答案）
                ans = node.val
            i += 1
            node = node.next
        return ans


"""
给你一个可能含有 重复元素 的整数数组 nums ，请你随机输出给定的目标数字 target 的索引。你可以假设给定的数字一定存在于数组中。
实现 Solution 类：
Solution(int[] nums) 用数组 nums 初始化对象。
int pick(int target) 从 nums 中选出一个满足 nums[i] == target 的随机索引 i 。如果存在多个有效的索引，则每个索引的返回概率应当相等。
链接：https://leetcode.cn/problems/random-pick-index
"""
class Solution:
    def __init__(self, nums: List[int]):
        self.len = len(nums)
        x = defaultdict()
        for idx, num in enumerate(nums):
            if num not in x:
                x[num] = []
            x[num].append(idx)
        self.hash_map = x
        print(x)

    def pick(self, target: int) -> int:
        arr = self.hash_map[target]
        length = len(arr)
        choice_idx = random.randint(0, length - 1)
        return arr[choice_idx]


if __name__ == '__main__':
    rs = ReservoirSample(size=3)
    rs.test_reservoir_sample()