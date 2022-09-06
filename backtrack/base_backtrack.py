from typing import List
from utils.base_decorator import CommonLogger


class Solution:
    # 元素无重不可复选
    """
    形式一、元素无重不可复选，即 nums 中的元素都是唯一的，每个元素最多只能被使用一次，这也是最基本的形式。
    以组合为例，如果输入 nums = [2,3,6,7]，和为 7 的组合应该只有 [7]。
    """

    def subsets(self, nums: List[int]) -> List[List[int]]:
        @CommonLogger()
        def backtrack(nums, startIndex):
            result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
            print('result:{}, path:{}'.format(result, path[:]))
            for i in range(startIndex, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                print('local path:{}'.format(path))
                path.append(nums[i])
                backtrack(nums, i + 1)  # 递归
                path.pop()  # 回溯

        result = []
        path = []
        backtrack(nums, 0)
        return result

    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []
        path = []

        @CommonLogger()
        def backtracking(n, k, start_index):
            if len(path) == k:
                result.append(path[:])
                print('result:{}'.format(result))
                return

            end_index = n + 1 - (k - len(path))  # (k - len(path)) 还需要取多少个数
            for i in range(start_index, end_index + 1):  # cut from n+1 -> n - (k - len(path)) + 2
                print('path:{}'.format(path))
                path.append(i)
                backtracking(n, k, i + 1)
                path.pop()

        backtracking(n, k, 1)
        return result

    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []  # 存放符合条件结果的集合
        path = []  # 用来存放符合条件的结果
        used = []  # 用来存放已经用过的数字

        def backtrack(nums, used):
            if len(path) == len(nums):
                return res.append(path[:])  # 此时说明找到了一组
            for i in range(0, len(nums)):
                if nums[i] in used:
                    continue  # used里已经收录的元素，直接跳过
                path.append(nums[i])
                used.append(nums[i])
                backtrack(nums, used)
                used.pop()
                path.pop()

        backtrack(nums, used)
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums):
            if not nums:
                res.append(path.copy())
                return

            for i in range(0, len(nums)):
                path.append(nums[i])
                l = nums[:i]
                r = nums[i + 1:]
                backtrack(l + r)
                path.pop()

        path = []
        res = []
        backtrack(nums)

        return res

    """
    形式二、元素可重不可复选，即 nums 中的元素可以存在重复，每个元素最多只能被使用一次。
以组合为例，如果输入 nums = [2,5,2,1,2]，和为 7 的组合应该有两种 [2,2,2,1] 和 [5,2]。
    先sort used数组
    """

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []  # 存放符合条件结果的集合
        path = []  # 用来存放符合条件结果

        def backtrack(nums, startIndex):
            res.append(path[:])
            for i in range(startIndex, len(nums)):
                if i > startIndex and nums[i] == nums[i - 1]:  # 我们要对同一树层使用过的元素进行跳过
                    continue
                path.append(nums[i])
                backtrack(nums, i + 1)  # 递归
                path.pop()  # 回溯

        nums = sorted(nums)  # 去重需要排序
        backtrack(nums, 0)
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        path = []
        res = []
        candidates.sort()

        def backtrack(candidates, start_index):
            sum_val = sum(path)
            if len(path) > len(candidates) or sum_val > target:
                return
            if sum_val == target:
                res.append(path[:])

            for i in range(start_index, len(candidates)):
                # i > start_index, aviod candidates[0], candidates[-1]
                if i - 1 >= start_index and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                backtrack(candidates, i + 1)
                path.pop()

        backtrack(candidates, 0)
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # res用来存放结果
        if not nums: return []
        res = []
        used = [0] * len(nums)

        def backtracking(nums, used, path):
            # 终止条件
            if len(path) == len(nums):
                res.append(path.copy())
                return
            for i in range(len(nums)):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue
                    used[i] = 1
                    path.append(nums[i])
                    backtracking(nums, used, path)
                    path.pop()
                    used[i] = 0

        # 记得给nums排序
        backtracking(sorted(nums), used, [])
        return res

    """
    形式三、元素无重可复选，即 nums 中的元素都是唯一的，每个元素可以被使用若干次。
    i+1 改成i
    """

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []

        def backtrack(candidates, target, sum, startIndex):
            if sum > target: return
            if sum == target: return res.append(path[:])
            for i in range(startIndex, len(candidates)):
                if sum + candidates[i] > target: return  # 如果 sum + candidates[i] > target 就终止遍历
                sum += candidates[i]
                path.append(candidates[i])
                backtrack(candidates, target, sum, i)  # startIndex = i:表示可以重复读取当前的数
                sum -= candidates[i]  # 回溯
                path.pop()  # 回溯

        candidates = sorted(candidates)  # 需要排序
        backtrack(candidates, target, 0, 0)
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        if digits == "":
            return []
        num_map = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        path = []
        group = []
        res = []
        for x in list(digits):
            arr = num_map[x]
            group.append(arr)
        print(group)

        def backtrack(n):
            if len(path) == len(group):
                tmp_str = ''.join(path)
                res.append(tmp_str)
                return
            for x in group[n]:
                path.append(x)
                backtrack(n + 1)
                path.pop()

        backtrack(0)
        return res

    # https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/by-guang-ying-7-cs08/
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

            def huisu(startIndex):
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
                        if huisu(startIndex + 1):
                            return True
                        path[i] += nums[startIndex]
                return False

            return huisu(0)

    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n: return []
        board = [['.'] * n for _ in range(n)]
        res = []

        def isVaild(board, row, col):
            # 判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
            # 判断左上角是否冲突
            i = row - 1
            j = col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i = row - 1
            j = col + 1
            while i >= 0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def backtracking(board, row, n):
            # 如果走到最后一行，说明已经找到一个解
            if row == n:
                temp_res = []
                for temp in board:
                    temp_str = "".join(temp)
                    temp_res.append(temp_str)
                res.append(temp_res)
            for col in range(n):
                if not isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(board, row + 1, n)
                board[row][col] = '.'

        backtracking(board, 0, n)
        return res

    """
    https://leetcode.cn/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-leetcode-solution/
    例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/next-permutation
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


if __name__ == '__main__':
    s = Solution()
    res = s.subsets(nums=[1, 2, 3])
    res = s.combine(n=4, k=2)
    res = s.letterCombinations(digits='23')
