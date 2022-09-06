from typing import List

"""
O(n2)
排序好的列表中，索引为n的位置对应数字为n+1
从最大的数字开始，判断该数字在不在需要在的位置（找到索引）
如果在的话就把最大数字和该数字的位置减一，继续循环
如果不在的话，先把包含该数字的列表arr[:index+1]翻转一次，再把包含其需要放到的索引的列表arr[:last+1]翻转一次，就能把这个数字放到该放的地方了
作者：bluegreenred
链接：https://leetcode.cn/problems/pancake-sorting/solution/969-jian-bing-pai-xu-python-by-bluegreen-gobh/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""


def pancakeSort(self, arr: List[int]) -> List[int]:
    ans = []
    # 目前需要翻转的最大数字，以及该数字需要放的位置
    most, last = len(arr), len(arr) - 1
    while most > 0:
        index = arr.index(most)
        if index != last:
            ans.append(index + 1)
            arr[:index + 1] = arr[index::-1]  # 翻转一次
            ans.append(last + 1)
            arr[:last + 1] = arr[last::-1]  # 翻转第二次
        most -= 1
        last -= 1
    return ans


def pancakeSort(self, arr: List[int]) -> List[int]:
    n = len(arr)  # arr 中的元素为 1 到 n 的所有自然数

    res = []

    # 从大 (n) 到小 (1)，将元素 i (i = n, n - 1, ……, 1) 放置到最终的排序位置 i - 1
    for i in range(n, 0, -1):

        # 1. 先找到元素 i 的索引
        idx = arr.index(i)

        # 2. 再判断元素 i 是否在最终排序的位置 i - 1

        # 如果如果元素 i 的索引为 i - 1
        # 说明元素 i 已经在最终的排序位置，继续对下一个元素排序
        if idx == i - 1:
            continue

        # 否则，即元素 i 不在位置 i - 1
        # 则，经过两次翻转，将元素 i 放置在最终的排序位置 i - 1
        # 第一次翻转，k = idx + 1，将元素 i 及其之前的所有元素进行翻转，此时，元素 i 位于数组的开始位置 0
        # 第二次翻转，k = i，将前 i 个元素进行翻转，此时元素 i 位于索引 i - 1 处，即元素 i 的最终排序位置~

        # 第一次翻转，k = idx + 1，将元素 i 及其之前的所有元素进行翻转，此时，元素 i 位于数组的开始位置 0
        res.append(idx + 1)

        start, end = 0, idx
        while start < end:
            tmp = arr[start]
            arr[start] = arr[end]
            arr[end] = tmp

            start += 1
            end -= 1

        # 第二次翻转，k = i，将前 i 个元素进行翻转，此时元素 i 位于索引 i - 1 处，即元素 i 的最终排序位置~
        res.append(i)

        start, end = 0, i - 1
        while start < end:
            tmp = arr[start]
            arr[start] = arr[end]
            arr[end] = tmp

            start += 1
            end -= 1

    return res
