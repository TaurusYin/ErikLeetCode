# You are given an array of integers nums, there is a sliding window of size k 
# which is moving from the very left of the array to the very right. You can only 
# see the k numbers in the window. Each time the sliding window moves right by one 
# position. 
# 
#  Return the max sliding window. 
# 
#  
#  Example 1: 
# 
#  
# Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
# Output: [3,3,5,5,6,7]
# Explanation: 
# Window position                Max
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7
#  
# 
#  Example 2: 
# 
#  
# Input: nums = [1], k = 1
# Output: [1]
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 10⁵ 
#  -10⁴ <= nums[i] <= 10⁴ 
#  1 <= k <= nums.length 
#  
# 
#  Related Topics 队列 数组 滑动窗口 单调队列 堆（优先队列） 👍 2450 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        if not nums:
            return []

        # Initialize deque and result list
        window, res = deque(), []

        for i, n in enumerate(nums):
            # Remove indices that are out of the current window
            while window and window[0] < i - k + 1:
                window.popleft()

            # Remove numbers that are smaller than the current number
            # from the back of the deque.
            while window and nums[window[-1]] < n:
                window.pop()

            # Add the current number's index to the back of the deque
            window.append(i)

            # The first number of the window is always the largest number.
            # So we add to result when we've passed at least k numbers into window.
            if i >= k - 1:
                res.append(nums[window[0]])

        return res

"""
# Example Usage
nums1 = [1,3,-1,-3,5,3,6,7]
k1 = 3
print(maxSlidingWindow(nums1, k1)) # Expected output: [3,3,5,5,6,7]

nums2 = [1]
k2 = 1
print(maxSlidingWindow(nums2, k2)) # Expected output: [1]
"""
# leetcode submit region end(Prohibit modification and deletion)
