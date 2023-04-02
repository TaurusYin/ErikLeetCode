# Koko loves to eat bananas. There are n piles of bananas, the iᵗʰ pile has 
# piles[i] bananas. The guards have gone and will come back in h hours. 
# 
#  Koko can decide her bananas-per-hour eating speed of k. Each hour, she 
# chooses some pile of bananas and eats k bananas from that pile. If the pile has less 
# than k bananas, she eats all of them instead and will not eat any more bananas 
# during this hour. 
# 
#  Koko likes to eat slowly but still wants to finish eating all the bananas 
# before the guards return. 
# 
#  Return the minimum integer k such that she can eat all the bananas within h 
# hours. 
# 
#  
#  Example 1: 
# 
#  
# Input: piles = [3,6,7,11], h = 8 sum=27
# Output: 4
#  
# 
#  Example 2: 
# 
#  
# Input: piles = [30,11,23,4,20], h = 5, sum=88  18
# Output: 30
#  
# 
#  Example 3: 
# 
#  
# Input: piles = [30,11,23,4,20], h = 6
# Output: 23
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= piles.length <= 10⁴ 
#  piles.length <= h <= 10⁹ 
#  1 <= piles[i] <= 10⁹ 
#  
# 
#  Related Topics 数组 二分查找 👍 472 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 向下取证
        def cal(sum_value, div):
            start = int(sum_value / div)
            mod = sum_value % div
            start = start + 1 if mod > 0 else start
            return start

        """
        sum_value = sum(piles)
        start = cal(sum_value, h)
        end = max(piles)
        arr = list(range(start, end + 1))
        """

        # 定义左右指针
        left, right = 1, max(piles)
        while left <= right:
            # 每次查找的中间元素
            mid = (left + right) // 2
            total_hour = sum(cal(pile, mid) for pile in piles)
            if total_hour > h:
                left = mid + 1
            elif total_hour <= h:
                right = mid - 1
        return left

    def _minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 左右边界
        left, right = 1, max(piles)
        while left <= right:
            mid = (left + right) // 2
            total_hour = sum((pile - 1) // mid + 1 for pile in piles)
            if total_hour <= h:
                # 如果总时间小于等于规定时间，说明速度过快或正好，缩小右边界
                right = mid - 1
            else:
                # 如果总时间大于规定时间，说明速度过慢，增大左边界
                left = mid + 1
        return left


piles = [3, 6, 7, 11]
h = 8
piles = [30, 11, 23, 4, 20]
h = 5
piles = [30, 11, 23, 4, 20]
h = 6
res = Solution().minEatingSpeed(piles=piles, h=h)
# leetcode submit region end(Prohibit modification and deletion)
