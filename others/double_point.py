# https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
import heapq
from typing import List

"""
初始化： 双指针 ii , jj 分列水槽左右两端；循环收窄： 直至双指针相遇时跳出；更新面积最大值 resres ；
选定两板高度中的短板，向中间收窄一格；返回值： 返回面积最大值 resres 即可；
链接：https://leetcode.cn/problems/container-with-most-water/solution/container-with-most-water-shuang-zhi-zhen-fa-yi-do/
"""


def maxArea(self, height: List[int]) -> int:
    i, j, res = 0, len(height) - 1, 0
    while i < j:
        if height[i] < height[j]:
            res = max(res, height[i] * (j - i))
            i += 1
        else:
            res = max(res, height[j] * (j - i))
            j -= 1
    return res


"""
https://leetcode.cn/problems/maximum-product-subarray/solution/shuang-zhi-zhen-by-wanglongjiang-ier1/
示例 1:
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
示例 2:
输入: nums = [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-product-subarray
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
0与任何数的乘积都是0，因此可以将数组看成被0分割的子数组，在各个子数组中查找乘积最大的值。
在一个非0的数组中求最大乘积，需要分析正负性。
- 没有负数或者负数为偶数个，最大乘积就是整个数组的乘积
- 有奇数个负数，如果第i个元素为负数，则[start,i-1]，[i+1,end]这2个区间的乘积都是最大乘积的候选。
通过下面2个指针交替移动算法可以计算所有[start,i-1]和[i+1,end]的乘积。

right指针向右移动，mul累计left至right指针之间的乘积，直至right遇到0或末尾。
向右移动left指针，mul除以被移出子数组的元素。
重复以上过程直至left指针移动到末尾。
时间复杂度：O(n)
空间复杂度：O(1)
作者：wanglongjiang
链接：https://leetcode.cn/problems/maximum-product-subarray/solution/shuang-zhi-zhen-by-wanglongjiang-ier1/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""


def maxProduct(self, nums: List[int]) -> int:
    left, right, n = 0, 0, len(nums)
    mul, product = 1, float('-inf')
    while left < n:
        while right < n and nums[right] != 0:  # 移动right指针直至遇到0，这中间用mul累计乘积，product记录最大的乘积
            mul *= nums[right]
            right += 1
            product = max(product, mul)
        while left + 1 < right:  # 移动left指针，这中间用mul累计乘积，product记录最大的乘积
            mul /= nums[left]
            left += 1
            product = max(product, mul)
        while right < n and nums[right] == 0:  # 跳过0
            product = max(product, 0)  # 有可能所有子数组的乘积都小于0，所以0也是候选
            right += 1
        left = right
        mul = 1
    return int(product)


# https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
"""
输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
"""


def removeDuplicates(self, nums: List[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    fast = slow = 1
    while fast < n:
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    return slow


"""
将元素依次入栈并统计元素数量。每次入栈判断是否和栈顶元素相同：如果与栈顶元素相同，那么将栈顶元素的数量加 1；如果栈顶元素数量达到 3，则将栈顶元素出栈；如果待入栈元素与栈顶元素不同，那么直接入栈并将该元素个数置为 1。遍历完字符串之后，将栈中剩余元素拼接即为答案。
链接：https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string-ii/solution/zhan-python3-by-smoon1989/
https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string-ii/solution/zhan-python3-by-smoon1989/
输入：s = "deeedbbcccbdaa", k = 3
输出："aa"
解释： 
先删除 "eee" 和 "ccc"，得到 "ddbbbdaa"
再删除 "bbb"，得到 "dddaa"
最后删除 "ddd"，得到 "aa"
"""


def removeDuplicates(self, s: str, k: int) -> str:
    n = len(s)
    stack = []
    for c in s:
        if not stack or stack[-1][0] != c:
            stack.append([c, 1])
        elif stack[-1][1] + 1 < k:
            stack[-1][1] += 1
        else:
            stack.pop()
    ans = ""
    for c, l in stack:
        ans += c * l
    return ans


# https://leetcode.cn/problems/remove-element/
def removeElement(self, nums: List[int], val: int) -> int:
    a = 0
    b = 0

    while a < len(nums):
        if nums[a] != val:
            nums[b] = nums[a]
            b += 1
        a += 1

    return b


"""
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。返回 [1, 2] 。
有序
链接：https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted
"""


def twoSum(self, numbers: List[int], target: int) -> List[int]:
    low, high = 0, len(numbers) - 1
    while low < high:
        total = numbers[low] + numbers[high]
        if total == target:
            return [low + 1, high + 1]
        elif total < target:
            low += 1
        else:
            high -= 1
    return [-1, -1]


"""
https://leetcode.cn/problems/two-sum/solution/liang-shu-zhi-he-by-leetcode-solution/
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

链接：https://leetcode.cn/problems/two-sum
"""


def twoSum(self, nums: List[int], target: int) -> List[int]:
    hashtable = dict()
    for i, num in enumerate(nums):
        if target - num in hashtable:
            return [hashtable[target - num], i]
        hashtable[nums[i]] = i
    return []


# https://leetcode.cn/problems/3sum/solution/by-s1ne-p3qs/
def threeSum(self, nums: List[int]) -> List[List[int]]:
    ans = []
    nums.sort()
    n = len(nums)
    for i in range(n):
        if nums[i] > 0: break
        if i and nums[i] == nums[i - 1]: continue
        l, r = i + 1, n - 1
        while l < r:
            sm = nums[i] + nums[l] + nums[r]
            if sm > 0:
                r -= 1
            elif sm < 0:
                l += 1
            else:
                ans.append([nums[i], nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]: l += 1
    return ans


def fourSum(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[List[int]]

    """
    return self.nSumTarget(sorted(nums), 4, 0, target)  # 调用nSumTarget函数，仅需要一行就可以搞定啦~这里n=4，所以可以得到四数之和的结果集


def nSumTarget(self, nums, n, start, target):
    sz = len(nums)
    res = []

    if n == 2:  # 基础是两数之和问题
        lo = start  # 因为我们已经对nums列表进行的递增排列，所以该操作可以避免取到之前取过的值
        hi = len(nums) - 1
        while lo < hi:  # 左右双指针模型，双指针相向而行
            left, right = nums[lo], nums[hi]
            sums = nums[lo] + nums[hi]
            if sums < target:
                while lo < hi and nums[lo] == left:  # 避免因列表中存在多个相同数字，而使得结果集重复
                    lo += 1
            elif sums > target:
                while lo < hi and nums[hi] == right:
                    hi -= 1
            else:
                res.append([left, right])
                while lo < hi and nums[lo] == left:
                    lo += 1
                while lo < hi and nums[hi] == right:
                    hi -= 1
    else:
        for i in range(start, sz):  # 这里的start是为了避免取到之前取过的值
            for j in self.nSumTarget(nums, n - 1, i + 1, target - nums[i]):  # 递归至（n-1）Sum问题
                j.append(nums[i])
                if j not in res:  # 防止向res列表中添加已经找到的结果，避免重复
                    res.append(j)

    return res


"""
滑动窗口左移动 右移动
https://leetcode.cn/problems/minimum-size-subarray-sum/solution/chang-du-zui-xiao-de-zi-shu-zu-by-leetcode-solutio/
O(n)
长度最小子数组
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
"""


def minSubArrayLen(self, s: int, nums: List[int]) -> int:
    if not nums:
        return 0

    n = len(nums)
    ans = n + 1
    start, end = 0, 0
    total = 0
    while end < n:
        total += nums[end]
        while total >= s:
            ans = min(ans, end - start + 1)
            total -= nums[start]
            start += 1
        end += 1

    return 0 if ans == n + 1 else ans


"""
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
算法的时间复杂度应该为 O(log (m+n)) 。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/median-of-two-sorted-arrays
"""


def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    num = nums1 + nums2
    num.sort()
    n = len(num)

    slow, fast = 0, 0
    while fast < n - 1:
        fast += 1
        slow += 1
        if fast < n - 1:
            fast += 1

    if n % 2 == 0:
        return (num[slow] + num[slow - 1]) / 2
    else:
        return float(num[slow])


"""
中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
示例：

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2

链接：https://leetcode.cn/problems/find-median-from-data-stream
"""
from sortedcontainers import SortedList


class MedianFinder:

    def __init__(self):
        self.nums = SortedList()
        self.left = self.right = None
        self.left_value = self.right_value = None

    def addNum(self, num: int) -> None:
        nums_ = self.nums

        n = len(nums_)
        nums_.add(num)

        if n == 0:
            self.left = self.right = 0
        else:
            # 模拟双指针，当 num 小于 self.left 或 self.right 指向的元素时，num 的加入会导致对应指针向右移动一个位置
            if num < self.left_value:
                self.left += 1
            if num < self.right_value:
                self.right += 1

            if n & 1:
                if num < self.left_value:
                    self.left -= 1
                else:
                    self.right += 1
            else:
                if self.left_value < num < self.right_value:
                    self.left += 1
                    self.right -= 1
                elif num >= self.right_value:
                    self.left += 1
                else:
                    self.right -= 1
                    self.left = self.right

        self.left_value = nums_[self.left]
        self.right_value = nums_[self.right]

    def findMedian(self) -> float:
        return (self.left_value + self.right_value) / 2


class MedianFinder:

    def __init__(self):
        self.small_heap = []  # 最大堆
        self.large_heap = []  # 最小堆

    def addNum(self, num: int) -> None:
        if len(self.small_heap) < len(self.large_heap):  # 加到small堆中
            # 先将num加到large中，再将large中的最小值弹出加入到small
            small_num = heapq.heappushpop(self.large_heap, num)
            heapq.heappush(self.small_heap, -small_num)
        else:
            # 先将num加到small中，再将small中的最小值弹出加入到large
            large_num = -heapq.heappushpop(self.small_heap, -num)
            heapq.heappush(self.large_heap, large_num)

    def findMedian(self) -> float:
        if len(self.small_heap) == len(self.large_heap):
            small = -self.small_heap[0]
            large = self.large_heap[0]
            return small + (large - small) / 2
        else:
            mid = self.large_heap[0]
            return mid


"""
https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/solution/581zui-duan-wu-xu-lian-xu-zi-shu-zu-pai-4dlft/
输入：nums = [2,6,4,8,10,9,15]
输出：5
解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
O(n)
"""
class Solution:
    def findUnsortedSubarray(self, nums):
        left, right, min_num, max_num = 0, 0, float("inf"), float("-inf")

        for i, j in enumerate(nums):
            if j < max_num:
                right = i
            max_num = max(max_num, j)

        for i in range(len(nums) - 1, -1, -1):
            if nums[i] > min_num:
                left = i
            min_num = min(min_num, nums[i])
        return 0 if left == right else right - left + 1

