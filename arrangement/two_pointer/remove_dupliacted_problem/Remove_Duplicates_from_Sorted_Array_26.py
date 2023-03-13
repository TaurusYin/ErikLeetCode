"""
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.


Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).

"""


class Solution:
    """
    给定一个按升序排列的整数数组 nums ，删除重复出现的元素，使每个元素只出现一次。要求在原地修改，空间复杂度为 O(1)，即不能使用额外的数组空间。
    双指针解法 由于题目要求在原地修改，不能使用额外的数组空间，因此我们可以使用双指针的方法来解决这个问题。
    定义两个指针 i 和 j ，其中 i 是慢指针，j 是快指针。快指针 j 指向下一个要比较的元素，慢指针 i 指向当前已经处理好的不重复元素的最后一个位置。
    初始时，i 指向数组的第一个位置，j 指向数组的第二个位置。 当 nums[i]=nums[j] 时，表示出现了重复元素，j 指针向后移动一位，继续比较下一个元素。
    当 nums[i]≠nums[j] 时，表示出现了新的不重复元素，将这个元素复制到 nums[i+1] 的位置上，并将 i 向后移动一位，继续比较下一个元素。重复上述过程，
    直到 j 指针到达数组的末尾。 最终，i+1 的值就是不重复元素的个数。我们可以返回 i+1 ，并将数组中前 i+1 个元素设置为不重复的元素。
    时间复杂度 时间复杂度为 O(n) ，其中 n 是数组的长度。双指针的方法最多遍历整个数组一次。 空间复杂度 空间复杂度为 O(1) ，由于题目要求在原地修改，因此没有使用额外的数组空间。
    """

    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]

        return i + 1
