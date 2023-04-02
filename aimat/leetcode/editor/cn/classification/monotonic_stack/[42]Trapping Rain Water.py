# Given n non-negative integers representing an elevation map where the width 
# of each bar is 1, compute how much water it can trap after raining. 
# 
#  
#  Example 1: 
#  
#  
# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
# Explanation: The above elevation map (black section) is represented by array [
# 0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) 
# are being trapped.
#  
# 
#  Example 2: 
# 
#  
# Input: height = [4,2,0,3,2,5]
# Output: 9
#  
# 
#  
#  Constraints: 
# 
#  
#  n == height.length 
#  1 <= n <= 2 * 10⁴ 
#  0 <= height[i] <= 10⁵ 
#  
# 
#  Related Topics 栈 数组 双指针 动态规划 单调栈 👍 4234 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def _trap(self, height: List[int]) -> int:
        size_res = []
        stack = []
        result = defaultdict()
        for index, val in enumerate(height):
            while stack and stack[-1][1] < val:
                # print(f"stack: {stack}, val:{val}")
                top_index, top_val = stack.pop()
                result[top_index] = val
                if stack:
                    distance = index - stack[-1][0] - 1
                    min_height = min(val, stack[-1][1])
                    size_res.append(distance * (min_height - top_val))
                    # print(f"val:{val} index:{index}, top_index:{top_index}, min_height:{min_height}, top_val:{top_val}")
                else:
                    size_res.append(0)

            stack.append((index, val))
        # print(size_res)
        return sum(size_res)

    def trap(self, height: List[int]) -> int:
        """
        这个方法利用了双指针的思想，从左右两端开始向中间遍历，记录左边和右边遍历过的柱子中的最大高度，同时从矮的一端开始计算蓄水量。因为如果矮的一端往高的一端走，即使后面有更高的柱子，也无法形成蓄水，只有往低的一端走才有可能形成蓄水。所以，每次选择左右两端中较矮的一端，计算它与它左右两侧的最大高度的差值，即为当前位置能够蓄水的高度，累加到总蓄水量中即可。这样遍历完整个数组后，即可得到总的蓄水量。这个方法的时间复杂度为 O(n)，空间复杂度为 O(1)。
        :param height:
        :return:
        """
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
# leetcode submit region end(Prohibit modification and deletion)
