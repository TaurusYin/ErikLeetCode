"""
@File    :   full_permute.py   
@Contact :   yinjialai 
"""
def generate_permutations(nums):
    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            result.append(current_permutation.copy())
            return

        for num in nums:
            current_permutation.append(num)
            backtrack(current_permutation)
            current_permutation.pop()

    result = []
    backtrack([])
    return result

# 示例
nums = [1,2,3]
print(generate_permutations(nums))