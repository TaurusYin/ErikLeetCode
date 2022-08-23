from collections import Counter
from typing import List


def uniqueOccurrences(self, arr: List[int]) -> bool:
    ans = []
    # 直接分析出现次数即可
    for k, v in Counter(arr).items():
        if v in ans:
            return False
        else:
            ans.append(v)
    return True


def uniqueOccurrences(self, arr: List[int]) -> bool:
    ans = []
    dict1 = {}
    for i in arr:
        if i not in dict1:
            dict1[i] = 1
        else:
            dict1[i] = dict1[i] + 1
    for k, v in dict1.items():
        if v in ans:
            return False
        else:
            ans.append(v)
    return True


def uniqueOccurrences(self, arr: List[int]) -> bool:
    set1 = set(arr)
    set2 = set()
    for i in set1:
        if arr.count(i) in set2:
            return False
        else:
            set2.add(arr.count(i))
    return True
