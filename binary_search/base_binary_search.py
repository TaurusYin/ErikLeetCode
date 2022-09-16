class Solution:
    def common_binary_search(self, nums, target: int):
        # [4, 5, 10, 10, 10, 13, 17, 21] target:10
        left, right = -1, len(nums)
        while (left + 1) != right:
            mid = (left + right) >> 1
            # if nums[mid] isBlue()
            if nums[mid] <= target:  # search last 10
                left = mid
            else:
                right = mid
        # return left or right
        return left

    '''
    该系列模板与标准模板有以下不同：

初始要判断空数组。
循环条件是：left < right，没有 = 号。所以退出循环后，left = right。只是我习惯使用 right。
mid 的取值以及区间的选取不同：某个区间可能包含 mid 的值。
往往排除某个区间比较容易思考，所以通过不断排除区间的方法得到最终值。剩余条件直接让 left 或 right 等于 mid 即可。
若是结果隐含了第一、首次、左边、最小等条件，则 mid 取下整，即 mid = left + right >> 1。
考虑收缩左边界，使用 left = mid + 1。
若是结果隐含了最后、右边、最大等条件，则 mid 取上整，即 mid = left + right + 1 >> 1。
考虑收缩右边界，使用 right = mid - 1。
循环结束后，还要进行条件的判断。

作者：繁星满天
链接：https://leetcode.cn/circle/article/bNaUjl/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    '''

    # 查找可以通过访问无重复数组中的单个索引来确定的元素或条件
    def binary_target_search(self, nums, target: int):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
            if target == nums[mid]:
                return mid
            elif target > nums[mid]:
                left = mid + 1
            elif target < nums[mid]:
                right = mid - 1
        return -1

    def first_search(self, nums, target):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left < right:  # <= -> <
            mid = left + right >> 1
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid  # right = mid - 1 -> right = mid
        if nums[right] == target:
            return right
        return -1

    def searchInsert(self, nums: list, target: int) -> int:
        # 返回大于等于 target 的索引，有可能是最后一个
        size = len(nums)
        # 特判
        if size == 0:
            return 0
        left = 0
        # 如果 target 比 nums里所有的数都大，则最后一个数的索引 + 1 就是候选值，因此，右边界应该是数组的长度
        right = size
        # 二分的逻辑一定要写对，否则会出现死循环或者数组下标越界
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                assert nums[mid] >= target
                # [1,5,7] 2
                right = mid
            # 调试语句
            print('left = {}, right = {}, mid = {}'.format(left, right, mid))
        return left

    """
    https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/solution/by-huan-huan-20-3ein/
    给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
    在排序数组中查找元素的第一个和最后一个位置
    输入：nums = [5,7,7,8,8,10], target = 8
    输出：[3,4]
    """

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def searchLeft(nums, target):
            left, right = 0, len(nums)
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    right = mid
                elif nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid
            if left == len(nums):
                return -1
            return left if nums[left] == target else -1

        def searchRight(nums, target):
            left, right = 0, len(nums)
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                elif nums[mid] > target:
                    right = mid
            if right == 0:
                return -1
            return right - 1 if nums[right - 1] == target else -1

        return [searchLeft(nums, target), searchRight(nums, target)]

    """
    https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/
    寻找旋转排序数组中的最小值
    输入：nums = [3,4,5,1,2]
    输出：1
    解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
    """

    def findMin(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            pivot = low + (high - low) // 2
            if nums[pivot] < nums[high]:
                high = pivot
            else:
                low = pivot + 1
        return nums[low]


"""
https://leetcode.cn/problems/find-peak-element/solution/
https://leetcode.cn/problems/find-peak-element/solution/162-xun-zhao-feng-zhi-by-likeinsane-zfld/
我们知道二分查找适用于严格单调函数上找特定值；
而三分查找则适用于在单峰函数上找极大值（或单谷函数的极小值），也适用于求函数局部的极大/极小值。
如图，在单峰函数f，范围[l, r]内任取两点lmid，rmid为例：
若f(lmid)<f(rmid)，则f必在lmid处单调递减，则极大值在[lmid, r]上
若f(lmid)>f(rmid)，则f必在rmid处单调递减，则极大值在[l, rmid]上
"""


def findPeakElement(self, nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left

"""
https://leetcode.cn/problems/squares-of-a-sorted-array/solution/dai-ma-sui-xiang-lu-shu-zu-ti-mu-zong-ji-1rtz/
双指针法
数组其实是有序的， 只不过负数平方之后可能成为最大数了。
那么数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。
此时可以考虑双指针法了，i指向起始位置，j指向终止位置。
定义一个新数组result，和A数组一样的大小，让k指向result数组终止位置
时间复杂度为O(n)，相对于暴力排序的解法O(n + nlogn)还是提升不少的
"""
def sortedSquares(self, nums: List[int]) -> List[int]:
    n = len(nums)
    i, j, k = 0, n - 1, n - 1
    ans = [-1] * n
    while i <= j:
        lm = nums[i] ** 2
        rm = nums[j] ** 2
        if lm > rm:
            ans[k] = lm
            i += 1
        else:
            ans[k] = rm
            j -= 1
        k -= 1
    return ans


if __name__ == '__main__':
    nums = [4, 5, 10, 10, 10, 13, 17, 21]
    x = Solution().common_binary_search(nums=nums, target=10)
    x = Solution().binary_target_search(nums=nums, target=10)
    x = Solution().first_search(nums=nums, target=10)

    print(x)
