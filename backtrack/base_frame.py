from functools import cache


class Solutions:
    def __init__(self):
        return

    def combine(self, nums):
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

    def permute(self, nums):
        def backtrack(nums, startIndex):
            if len(path) == len(nums):
                result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
            # print('result:{}, path:{}'.format(result, path[:]))
            for i in range(0, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                # print('local path:{}'.format(path))
                if used[i]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack(nums, i + 1)  # 递归
                path.pop()  # 回溯
                used[i] = False

        result = []
        path = []
        used = [False] * len(nums)
        backtrack(nums, 0)
        print(used)
        return result

    def combine_duplicated(self, nums):
        def backtrack(nums, startIndex):
            result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
            print('result:{}, path:{}'.format(result, path[:]))
            for i in range(startIndex, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                # 剪枝逻辑，跳过值相同的相邻树枝
                if i > startIndex and nums[i] == nums[i - 1]:
                    continue
                print('local path:{}'.format(path))
                path.append(nums[i])
                backtrack(nums, i + 1)  # 递归
                path.pop()  # 回溯

        result = []
        path = []
        nums.sort()
        backtrack(nums, 0)
        return result

    def permute_duplicated(self, nums):
        def backtrack(nums, startIndex):
            if len(path) == len(nums):
                result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
            # print('result:{}, path:{}'.format(result, path[:]))
            for i in range(0, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                # print('local path:{}'.format(path))
                if used[i]:
                    continue
                # 剪枝逻辑，固定相同的元素在排列中的相对位置
                if i > 0 and nums[i] == nums[i - 1] and not used[i-1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack(nums, i + 1)  # 递归
                path.pop()  # 回溯
                used[i] = False

        result = []
        path = []
        used = [False] * len(nums)
        nums.sort()
        backtrack(nums, 0)
        print(used)
        return result

    def combine_multicheck(self, nums):
        def backtrack(nums, startIndex):
            if len(path) == len(nums):
                result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
                return
            print('result:{}, path:{}'.format(result, path[:]))
            for i in range(startIndex, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                print('local path:{}'.format(path))
                path.append(nums[i])
                backtrack(nums, i)  # 递归
                path.pop()  # 回溯

        result = []
        path = []
        backtrack(nums, 0)
        return result

    def permute_multicheck(self, nums):
        def backtrack(nums, startIndex):
            if len(path) == len(nums):
                result.append(path[:])  # 收集子集，要放在终止添加的上面，否则会漏掉自己
                return
            print('result:{}, path:{}'.format(result, path[:]))
            for i in range(0, len(nums)):  # 当startIndex已经大于数组的长度了，就终止了，for循环本来也结束了，所以不需要终止条件
                print('local path:{}'.format(path))
                path.append(nums[i])
                backtrack(nums, i)  # 递归
                path.pop()  # 回溯
        result = []
        path = []
        backtrack(nums, 0)
        return result





if __name__ == '__main__':
    s = Solutions()
    nums = [2, 3, 6, 7]
    res = s.combine(nums=nums)
    res = s.permute(nums=nums)
    nums = [2, 5, 2, 1, 2]
    res = s.combine_duplicated(nums=nums)
    res = s.permute_duplicated(nums=nums)
    nums = [2, 3, 6, 7]
    res = s.combine_multicheck(nums=nums)
    res = s.permute_multicheck(nums=nums)

    print()
