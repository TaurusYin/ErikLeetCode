"""
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
"""

from collections import Counter


def solve(nums, k):
    stats = dict(Counter(nums))
    sort_res = sorted(stats.items(), key=lambda x:x[1], reverse=True)
    res = []
    for i in range(0, k):
        res.append(sort_res[i][0])
    # n = len(nums)
    # O(n)
    return res

if __name__ == '__main__':
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print(solve(nums, k))
