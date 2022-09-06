# envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))


"""
示例 1：
输入：nums = [10,2]
输出："210"
示例 2：
输入：nums = [3,30,34,5,9]
输出："9534330"
https://leetcode.cn/problems/largest-number/solution/python3-san-chong-fang-fa-qiu-zui-da-shu-cpi4/
"""
def largestNumber(self, nums: List[int]) -> str:
    def cmp(x, y):
        return 1 if x + y < y + x else -1

    nums = list(map(str, nums))
    nums.sort(key=cmp_to_key(cmp))
    res = str(int("".join(nums)))
    return res


arr = [[1, 2, 3], [3, 2, 1], [4, 2, 1], [6, 4, 3]]
indices = [[2, 0], [0, 1]]
def custom_sort(x):
    tmp_sort_method = []
    print(x)
    for indice in indices:
        idx = indice[0]
        if indice[1] == 1:
            res = -x[idx]
        else:
            res = x[idx]
        tmp_sort_method.append(res)
    return tmp_sort_method
arr.sort(key=lambda x: custom_sort(x))

print()


def compare(a, b):
    if a[2] > b[2]:
        return 1
    elif a[2] < b[2]:
        return -1
    else:
        return 0
    return


x = arr.sort(key=functools.cmp_to_key(compare))
print(x)
