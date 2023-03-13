"""
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].
The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

For example, the next permutation of arr = [1,2,3] is [1,3,2].
Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.
Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.



Example 1:

Input: nums = [1,2,3]
Output: [1,3,2]
Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]
Example 3:

Input: nums = [1,1,5]
Output: [1,5,1]


Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 100

"""

from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
            +---+---+---+---+---+---+
    | 5 | 4 | 7 | 5 | 3 | 2 |
    +---+---+---+---+---+---+
          |
          v
    +---+---+---+---+---+---+
    | 5 | 5 | 7 | 4 | 3 | 2 |
    +---+---+---+---+---+---+
                      |
                      v
    +---+---+---+---+---+---+
    | 5 | 5 | 2 | 3 | 4 | 7 |
    +---+---+---+---+---+---+

        """

        """
        Do not return anything, modify nums in-place instead.
        """
        # Initialize index of first element to -1
        first_index = -1
        # Start from the end of the list and find the first element that is less than its next element
        right = len(nums) - 1
        while right > 0:
            if nums[right - 1] < nums[right]:
                first_index = right - 1
                break
            right -= 1
        # If no such element is found, the entire list is in descending order, so reverse the list
        if first_index == -1:
            nums.reverse()
            return
        # Find the smallest element in the list that is greater than the element at the first index
        right = len(nums) - 1
        while right > first_index:
            if nums[right] > nums[first_index]:
                # Swap the two elements
                nums[right], nums[first_index] = nums[first_index], nums[right]
                break
            right -= 1
        # Reverse the portion of the list to the right of the first index
        nums[first_index + 1:] = sorted(nums[first_index + 1:])
        return nums


Solution().nextPermutation(nums=[5, 4, 7, 5, 3, 2])
