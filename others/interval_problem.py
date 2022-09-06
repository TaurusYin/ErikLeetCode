from typing import List

"""
https://leetcode.cn/problems/remove-covered-intervals/solution/
双元素排序，覆盖
[[1, 4], [1, 2], [3, 4]]
[[1, 4], [1, 4], [3, 4]]
[[1, 4], [1, 4], [1, 4]]
时间复杂度：O(N \log N)O(NlogN)，其中 NN 是区间的个数。
空间复杂度：O(\log N)O(logN)，为排序需要的空间。
"""


def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
    intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    val = len(intervals)
    for i, item in enumerate(intervals):
        if i >= 1:
            start, end = item[0], item[1]
            pre_start, pre_end = intervals[i - 1][0], intervals[i - 1][1]
            if start >= pre_start and end <= pre_end:
                intervals[i][0], intervals[i][1] = pre_start, pre_end
                val -= 1
    return val


"""
https://leetcode.cn/problems/merge-intervals/
区间合并
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
时间复杂度：O(n\log n)O(nlogn)，其中 nn 为区间的数量。除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 O(n\log n)O(nlogn)。
空间复杂度：O(\log n)O(logn)，其中 nn 为区间的数量。这里计算的是存储答案之外，使用的额外空间。O(\log n)O(logn) 即为排序所需要的空间复杂度。
"""
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])

    merged = []
    for interval in intervals:
        # 如果列表为空，或者当前区间与上一区间不重合，直接添加
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 否则的话，我们就可以与上一区间进行合并
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged
