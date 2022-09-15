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

"""
https://leetcode.cn/problems/sub-sort-lcci/solution/python3-pai-xu-bi-jiao-bu-tong-by-qing-b-a0ay/
给定一个整数数组，编写一个函数，找出索引m和n，只要将索引区间[m,n]的元素排好序，整个数组就是有序的。注意：n-m尽量最小，也就是说，找出符合条件的最短序列。函数返回值为[m,n]，若不存在这样的m和n（例如整个数组是有序的），请返回[-1,-1]。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sub-sort-lcci
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def subSort(self, array: List[int]) -> List[int]:
    n = len(array)
    maxx = float('-inf')
    minn = float('inf')
    l = -1
    r = -1
    #   正向遍历，确定右边界
    for i in range(n):
        #   小于当前最大值，则说明需要参与排序
        if array[i] < maxx:
            r = i
        #   更新最大值
        else:
            maxx = array[i]

    #   反向遍历，确定左边界
    for i in range(n - 1, -1, -1):
        #   大于当前最小值，则说明需要参与排序
        if array[i] > minn:
            l = i
        #   更新最小值
        else:
            minn = array[i]

    return [l, r]



def subSort(self, array: List[int]) -> List[int]:
    # 如何比较字典序，没有比排序更有效的了
    array_1 = array[:]
    array_1.sort()
    # 如果字典序最小，那么肯定没答案了，而排序后的字典序最小，所以直接比较原字符串和字典序的不同
    if array_1 == array:
        return [-1, -1]
    # 正序获取第一个不同的字符
    for i in range(len(array_1)):
        if array[i] != array_1[i]:
            # 记住break啊！
            break
    # 逆序获取最后一个不同的字符
    for j in reversed(range(len(array_1))):
        if array[j] != array_1[j]:
            break
    # 排序NlogN，比较一次遍历N，共NlogN + N
    return [i, j]


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
