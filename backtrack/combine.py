from typing import List
from utils.base_decorator import CommonLogger


class Solution:
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



if __name__ == '__main__':
    s = Solution()
    res = s.subsets(nums=[1, 2, 3])
    res = s.combine(n=4, k=2)
    res = s.letterCombinations(digits='23')
