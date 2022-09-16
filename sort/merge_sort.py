from typing import List


def merge_sort(nums, low, high):
    if low == high:
        return
    mid = (low + high) // 2
    merge_sort(nums, low, mid)
    merge_sort(nums, mid + 1, high)
    merge(nums, low, mid, high)
    return nums


def merge(nums, low, mid, high):
    i, j, tmp = low, mid + 1, []
    while i <= mid and j <= high:
        print('{},{}'.format(nums[i:mid + 1], nums[mid + 1:j + 1]))
        if nums[i] <= nums[j]:  # 左半区第一个剩余元素更小
            tmp.append(nums[i])
            i += 1
        else:
            tmp.append(nums[j])
            j += 1
        print('tmp:{}'.format(tmp))
    if i <= mid: # 右半区用完了，左半区直接搬过去
        tmp.extend(nums[i:mid + 1])
    if j <= high: # 左半区用完了，右半区直接搬过去
        tmp.extend(nums[j:high + 1])
    nums[low:high + 1] = tmp # 把合并后的数组拷回原来的数组

"""
归并区间
https://leetcode.cn/problems/interval-list-intersections/submissions/
"""
def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    ans = []
    A = firstList;
    B = secondList
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
田忌赛马
https://leetcode.cn/problems/advantage-shuffle/
"""
def advantageCount(A, B):
        sortedA = sorted(A)
        sortedB = sorted(B)

        # assigned[b] = list of a that are assigned to beat b
        # remaining = list of a that are not assigned to any b
        assigned = {b: [] for b in B}
        remaining = []

        # populate (assigned, remaining) appropriately
        # sortedB[j] is always the smallest unassigned element in B
        j = 0
        for a in sortedA:
            if a > sortedB[j]:
                assigned[sortedB[j]].append(a)
                j += 1
            else:
                remaining.append(a)

        # Reconstruct the answer from annotations (assigned, remaining)
        return [assigned[b].pop() if assigned[b] else remaining.pop()for b in B]
nums1 = [2,7,11,15] ; nums2 = [1,10,4,11]
advantageCount(nums1,nums2)

"""
输入:
  [
    [1, 3, 5, 7],
    [2, 4, 6],
    [0, 8, 9, 10, 11]
  ]
输出: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
"""
def mergekSortedArrays(self, arrays):
    # write your code here
    import heapq
    result = []
    heap = []
    for index, array in enumerate(arrays):
        if len(array) == 0:
            continue
        heapq.heappush(heap, (array[0], index, 0))

    while len(heap):
        val, x, y = heap[0]
        heapq.heappop(heap)
        result.append(val)
        if y + 1 < len(arrays[x]):
            heapq.heappush(heap, (arrays[x][y + 1], x, y + 1))

    return result


"""
给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。
类似归并排序, 不过空间规定好了在nums1里, 所以为了防止数据被覆盖, 倒序归并即可
利用哨兵可以简化代码的写法
https://leetcode.cn/problems/merge-sorted-array/solution/by-isuxiz-2knu/
O(m+n)
O(1)
注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。
"""

def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    i, j = m - 1, n - 1
    curr = m + n - 1
    while curr >= 0:
        num1 = nums1[i] if i >= 0 else float('-inf')
        num2 = nums2[j] if j >= 0 else float('-inf')
        if num1 >= num2:
            nums1[curr] = num1
            i -= 1
        else:
            nums1[curr] = num2
            j -= 1
        curr -= 1
    return





if __name__ == '__main__':
    nums = [100, 2, 3, 4, 10, 40]
    nums = [1, 3, 2, 3, 1]
    print(merge_sort(nums, 0, len(nums) - 1))
