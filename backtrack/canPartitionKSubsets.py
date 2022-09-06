# https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/by-guang-ying-7-cs08/
"""
给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
"""
from typing import List


def canPartitionKSubsets(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    每个元素可能放到k个不同的子集里面
    """
    # 回溯，时间复杂福O(k^n)，每个元素可以选一个桶放入，关键在于树层去重剪枝
    sums = sum(nums)
    n = len(nums)
    if sums % k != 0:
        return False  # 不能整除直接返回
    else:
        # 1.1 能整除先求target，设置初始数组
        target = sums // k
        nums.sort(reverse=True)
        path = [target] * k  # 初始为target
        if nums[-1] > target: return False

        def backtrack(startIndex):
            # 1.1 输入startIndex对应当前数组索引
            # 1.2 返回条件，当数组遍历完时
            if startIndex > n - 1:
                return True
            # 1.3 遍历不同的桶
            for i in range(k):
                if path[i] - nums[startIndex] >= 0:
                    # 1.4.1 剪枝1：只有相减后path[i]>=0才能继续
                    if i > 0 and path[i] == path[i - 1]:
                        # 1.4.2 剪枝2：关键剪枝，树层去重，当前层path[i]和path[i-1]相等时直接跳过，为重复计算
                        continue
                    path[i] -= nums[startIndex]
                    if backtrack(startIndex + 1):
                        return True
                    path[i] += nums[startIndex]
            return False

        return backtrack(0)


# https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/pythonzhi-guan-de-zan-shu-si-lu-hui-su-fa-jia-jian/
# k* 2^n
def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
    if not nums or len(nums) < k:  # 为空或不够分
        return False
    avg, mod = divmod(sum(nums), k)
    if mod:  # 不能整除
        return False
    nums.sort(reverse=True)  # 倒序排列
    if nums[0] > avg:  # 有超过目标的元素
        return False
    used = set()  # 记录已使用的数

    def dfs(k, start=0, tmpSum=0):  # 当前还需要凑的avg个数，当前从哪个数开始考虑，以及当前已凑够的和
        if tmpSum == avg:  # 如果已凑满一个
            return dfs(k - 1, 0, 0)  # 那么从最大数重新开始考虑，凑下一个
        if k == 1:  # 只剩最后一个，那么剩下的没使用的数加起来肯定凑满
            return True
        for i in range(start, len(nums)):  # 优先用大的数的凑
            if i not in used and nums[i] + tmpSum <= avg:  # 如果该数未使用并且可以用来凑
                used.add(i)  # 使用该数
                if dfs(k, i + 1, nums[i] + tmpSum):  # 继续用比该数小的数来凑
                    return True
                used.remove(i)  # 没有得到可用方案，则换个数来凑
        return False
