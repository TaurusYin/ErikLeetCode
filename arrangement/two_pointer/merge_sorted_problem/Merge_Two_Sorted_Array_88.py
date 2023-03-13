"""
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
Merge nums1 and nums2 into a single array sorted in non-decreasing order.
The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

Example 2:
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].

Example 3:
Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.
"""


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        这个题的关键思路是通过双指针来实现两个有序数组的合并，从而得到一个新的有序数组。
具体来说，我们可以定义两个指针 p1 和 p2 分别指向 nums1 和 nums2 数组的最后一个元素，然后从后往前遍历两个数组，并比较 nums1[p1] 和 nums2[p2] 的大小关系。如果 nums1[p1] 大于 nums2[p2]，就将 nums1[p1] 插入到 nums1 的末尾，并将 p1 向前移动一个位置，否则将 nums2[p2] 插入到 nums1 的末尾，并将 p2 向前移动一个位置。
当 p1 和 p2 中任意一个指针遍历完其对应的数组时，我们就可以将另一个数组中未被遍历的元素直接添加到 nums1 的开头，从而得到一个新的有序数组。通过这个思路，我们可以避免频繁地对 nums1 数组进行元素移动，从而提高算法的效率。
        """
        p1, p2 = m - 1, n - 1
        p = m + n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
            p -= 1
        # 未合并的一并合并过去。左闭右开
        nums1[:p2 + 1] = nums2[:p2 + 1]


