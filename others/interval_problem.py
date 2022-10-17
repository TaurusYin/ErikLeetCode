from typing import List

"""
https://leetcode.cn/problems/remove-covered-intervals/solution/
双元素排序，覆盖
[[1, 4], [1, 2], [3, 4]]
[[1, 4], [1, 4], [3, 4]]
[[1, 4], [1, 4], [1, 4]]
时间复杂度：O(N \log N)O(NlogN)，其中 NN 是区间的个数。
空间复杂度：O(\log N)O(logN)，为排序需要的空间。
输入：intervals = [[1,4],[3,6],[2,8]]
输出：2
解释：区间 [3,6] 被区间 [2,8] 覆盖，所以它被删除了。
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

"""
输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/interval-list-intersections
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
# https://leetcode.cn/problems/interval-list-intersections/solution/qu-jian-lie-biao-de-jiao-ji-by-leetcode/
def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    ans = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Let's check if A[i] intersects B[j].
        # lo - the startpoint of the intersection
        # hi - the endpoint of the intersection
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            ans.append([lo, hi])

        # Remove the interval with the smallest endpoint
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return ans


"""
输入：tasks = ["A","A","A","B","B","B"], n = 2
输出：8
解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B
     在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。 

https://leetcode.cn/problems/task-scheduler/solution/python-xiang-jie-by-jalan/
"""
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        length = len(tasks)
        if length <= 1:
            return length

        # 用于记录每个任务出现的次数
        task_map = dict()
        for task in tasks:
            task_map[task] = task_map.get(task, 0) + 1
        # 按任务出现的次数从大到小排序
        task_sort = sorted(task_map.items(), key=lambda x: x[1], reverse=True)

        # 出现最多次任务的次数
        max_task_count = task_sort[0][1]
        # 至少需要的最短时间
        res = (max_task_count - 1) * (n + 1)

        for sort in task_sort:
            if sort[1] == max_task_count:
                res += 1

        # 如果结果比任务数量少，则返回总任务数
        return res if res >= length else length


