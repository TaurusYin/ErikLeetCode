"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9


Constraints:

n == height.length
1 <= n <= 2 * 104
0 <= height[i] <= 105

"""

class Solution:
    """
    我们可以通过观察，发现一根柱子上方能够盛放的水的量，取决于它左右两边最高的柱子中的较矮者。
那么，对于某个柱子i来说，它上方所能盛放的水的量，就是它左右两边最高的柱子中的较矮者减去它本身的高度。
因此，我们可以维护两个指针left和right，以及两个变量left_max和right_max，分别表示从左到右遍历过程中左侧已经遍历过的柱子中高度的最大值，以及右侧已经遍历过的柱子中高度的最大值。
对于每一个柱子i，如果left_max < right_max，那么对于它上方所能盛放的水的量，取决于它左侧的柱子中高度的最大值left_max。
反之，如果left_max >= right_max，那么对于它上方所能盛放的水的量，取决于它右侧的柱子中高度的最大值right_max。
维护left和right指针，不断地移动它们，直到left和right重合为止，就可以得到所有柱子上方所能盛放的水的总量。
    """

    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n < 3:
            return 0
        left, right = 0, n - 1
        left_max, right_max = height[0], height[n - 1]
        ans = 0
        while left <= right:
            left_max = max(left_max, height[left])
            right_max = max(right_max, height[right])
            if left_max < right_max:
                ans += left_max - height[left]
                left += 1
            else:
                ans += right_max - height[right]
                right -= 1
        return ans