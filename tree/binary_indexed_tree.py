'''BIT：Binary Indexed Tree 树状数组【模板类】'''
from typing import List

'''307模板：https://leetcode-cn.com/problems/range-sum-query-mutable/solution/shu-zhuang-shu-zu-xian-duan-shu-pythonsh-tjn5/'''


class BIT:
    def __init__(self, n):
        self.tree = [0] * (n + 1)  # 比原数组多1个元素

    def lowbit(self, x):
        return x & (-x)

    def add(self, i, delta):  # 单点更新：执行+delta
        i += 1  # 原数组下标转换到树状数组下标
        while i < len(self.tree):
            self.tree[i] += delta
            i += self.lowbit(i)

    def query(self, i):  # 前缀和查询
        i += 1  # 原数组下标转换到树状数组下标
        summ = 0
        while i > 0:
            summ += self.tree[i]
            i -= self.lowbit(i)
        return summ



class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)

        # 离散化：绝对数值转秩次【rank从0开始】
        uniques = sorted(set(nums))
        rank_map = {v: i for i, v in enumerate(uniques)}

        # 构建树状数组
        tree = BIT(len(uniques))

        # 从右往左查询
        ans = 0
        for i in range(n - 1, -1, -1):
            rank = rank_map[nums[i]]  # 当前值的排名
            tree.add(rank, 1)  # 单点更新+1
            ans += tree.query(rank - 1)  # 查询 当前排名之前 的元素有多少个

        return ans

if __name__ == '__main__':
    case_set = [
        [5, 4, 3, 2, 1],
        [7, 5, 6, 4],
        []
    ]
    for case in case_set:
        x = Solution().reversePairs(nums=case)
        print(x)