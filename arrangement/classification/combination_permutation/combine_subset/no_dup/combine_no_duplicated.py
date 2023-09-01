"""
@File    :   base.py   
@Contact :   yinjialai 
"""
"""
形式一、元素无重不可复选，
即 nums 中的元素都是唯一的，每个元素最多只能被使用一次，这也是最基本的形式。

"""
def combinations_subset(nums):
    def backtrack(start, path):
        result.append(list(path))
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    result = []
    backtrack(0, [])
    return result

nums = [2,3,6,7]
nums = [1,2,3]
print(combinations_subset(nums))
# [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]


"""
以组合为例，如果输入 nums = [2,3,6,7]，和为 7 的组合应该只有 [7]。
"""
def combinationSum(nums, target):
    def backtrack(start, path, current_sum):
        if current_sum == target:
            result.append(list(path))
            return
        for i in range(start, len(nums)):
            if current_sum + nums[i] > target:  # 提前结束循环以优化性能
                continue
            path.append(nums[i])
            backtrack(i + 1, path, current_sum + nums[i])
            path.pop()

    result = []
    backtrack(0, [], 0)
    return result

nums = [2,3,6,7]
print(combinationSum(nums, 7))  # 输出：[[7]]


