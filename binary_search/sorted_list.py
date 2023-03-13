
from sortedcontainers import SortedList

sl = SortedList([7,1,3,3,4,5,3,6])
print(sl)
# [1, 3, 3, 3, 4, 5, 6, 7]
left = sl.bisect_left(3) # index:1 第一个元素的位置
right = sl.bisect_right(3) # index:4 最后一个元素后一个位置
print()


"""
https://leetcode.cn/problems/count-of-range-sum/solution/327-qu-jian-he-de-ge-shu-by-shikata-akik-hhba/
Example 1:

Input: nums = [-2,5,-1], lower = -2, upper = 2
Output: 3
Explanation: The three ranges are: [0,0], [2,2], and [0,2] and their respective sums are: -2, -1, 2.

"""
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        """
        前缀和：prefix[i]: sum[0...i]
        区间和：sum_range[i...j] = sum[i...j] = sum[0...j] - sum[0...i] + nums[i]
                                 = prefix[j] - prefix[i] + nums[i]

        于是，对于区间 [lo...hi]：
            lower <= sum_range[lo...hi] <= upper
        即，
            lower <= prefix[hi] - prefix[lo] + nums[lo] <= upper

        对上述不等式变形，有：
            prefix[hi] - upper <= prefix[lo] - nums[lo] <= prefix[hi] - lower

        于是，我们可以用一个有序列表 sl 保存 prefix[lo] - nums[lo]，
        枚举右端点 hi，使用二分查找，得到满足上述不等式的元素个数
        注意有序列表 sl 中的元素是动态增加的，要确保区间端点 lo <= hi

        """

        n = len(nums)

        prefix = [0 for _ in range(n)]
        prefix[0] = nums[0]
        for i in range(1, n):
            prefix[i] = prefix[i - 1] + nums[i]

        sl = SortedList()

        res = 0

        # 枚举区间右端点 hi
        # prefix[lo] - nums[lo] <= prefix[hi] - lower
        # prefix[lo] - nums[lo] >= prefix[hi] - upper
        for i in range(n):
            sl.add(prefix[i] - nums[i])

            right = sl.bisect_right(prefix[i] - lower)
            left = sl.bisect_left(prefix[i] - upper)

            res += right - left

        return res


