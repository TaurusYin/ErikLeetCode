''''ST：Segment Tree 线段树【模板类】'''
from typing import List

'''307模板：https://leetcode-cn.com/problems/range-sum-query-mutable/solution/shu-zhuang-shu-zu-xian-duan-shu-pythonsh-tjn5/'''




class ST:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (2 * n)  # 元素数目为原数组的2倍

    def build_tree(self, tree, root, start, end):
        mid = (start + end) >> 1
        left_node = 2 * root + 1
        right_node = 2 * root + 2
        


    def add(self, i, delta):
        i += self.n  # 原数组下标转换到线段树下标
        while i > 0:
            self.tree[i] += delta
            i //= 2

    def rangeSum(self, i, j):  # 区间和【自下而上统计】
        i, j = i + self.n, j + self.n  # 原数组下标转换到线段树下标
        summ = 0
        while i <= j:
            if i & 1 == 1:  # 右子节点
                summ += self.tree[i]
                i += 1
            if j & 1 == 0:  # 左子节点
                summ += self.tree[j]
                j -= 1
            i //= 2
            j //= 2
        return summ


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)

        # 离散化：绝对数值转秩次【rank从0开始】
        uniques = sorted(list(set(nums)))
        rank_map = {v: i for i, v in enumerate(uniques)}

        # 构建线段树
        tree = ST(len(uniques))

        # 从右往左查询
        ans = 0
        for i in range(n - 1, -1, -1):
            rank = rank_map[nums[i]]  # 当前值的排名
            tree.add(rank, 1)  # 单点更新+1
            ans += tree.rangeSum(0, rank - 1)  # 查询 当前排名之前 的元素有多少个

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