from bisect import bisect
from cmath import inf
from functools import cache
from typing import List


class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        """
                dp[i+1][0] = costs[i][0] + min(dp[i][1],dp[i][2])
                dp[i+1][1] = costs[i][1] + min(dp[i][0],dp[i][2])
                dp[i+1][2] = costs[i][2] + min(dp[i][0],dp[i][1])
        """
        dp = [[inf] * len(costs[0]) for _ in range(len(costs))]
        dp[0] = costs[0]
        print(dp)
        for i in range(0, len(costs) - 1):
            dp[i + 1][0] = costs[i + 1][0] + min(dp[i][1], dp[i][2])
            dp[i + 1][1] = costs[i + 1][1] + min(dp[i][0], dp[i][2])
            dp[i + 1][2] = costs[i + 1][2] + min(dp[i][0], dp[i][1])
        return min(dp[-1])

    """
    https://leetcode.cn/problems/russian-doll-envelopes/
    请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面。
    """

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        # 基于w升序，h降序排序
        envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))
        # 寻找以h升序的最长子序列
        heights = [envelope[1] for envelope in envelopes]
        return self.lengthOfLIS(heights)

    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        dp = [1] * len(envelopes)
        dp[0] = 1
        envelopes.sort()
        envelopes.sort(key=lambda x: (x[0], -x[1]))  # 先正排序，后逆排序
        for i in range(1, len(envelopes)):
            for j in range(0, i):
                if envelopes[i][0] > envelopes[j][0] and envelopes[i][1] > envelopes[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        扑克牌每次选一个插入二分偏左位置
        """
        n = len(nums)
        d = []
        for i, num in enumerate(nums):
            if not d or num > d[-1]:
                d.append(num)
            else:
                pos = bisect.bisect_left(d, num)  # 这个库函数用来查找num应该插入的位置，使其仍然有序
                d[pos] = num
        return len(d)

    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        以i为结尾的递增子序列 dp[i] = max(dp[i], dp[0..j] + 1) if nums[i] > nums[j]  j<i
        """
        dp = [1] * len(nums)
        dp[0] = 1
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def LengthOfLCIS(self, nums: List[int]) -> int:
        """
        最长连续递增
        dp[i + 1] = dp[i] + 1 if nums[i + 1] > nums[i] else 1
        :param nums:
        :return:
        """
        dp = [1] * (len(nums) + 1)
        for i in range(len(nums) - 1):
            dp[i + 1] = dp[i] + 1 if nums[i + 1] > nums[i] else 1
        return max(dp)

    def maxSubArray(self, nums: List[int]) -> int:
        """
        最大子数组的和， dp[i + 1] = dp[i] + nums[i] if dp[i] > 0 else nums[i]
        输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
        输出: 6
        解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
        :param nums:
        :return:
        """
        dp = [-inf] * (len(nums) + 1)
        for i in range(len(nums)):
            dp[i + 1] = dp[i] + nums[i] if dp[i] > 0 else nums[i]
        print(dp)
        return max(dp)

    """
    最长重复子数组
    输入：nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
    输出：3
    解释：长度最长的公共子数组是 [3,2,1] 。
    """
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # dp[i+1][j+1] = dp[i][j] + 1 if nums1[i] == nums[j] else 0
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        result = 0
        for i in range(0, len(nums1)):
            for j in range(0, len(nums2)):
                if nums1[i] == nums2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                result = max(result, dp[i + 1][j + 1])
        return result

    def longest_common_subarray(self, nums1: List[int], nums2: List[int]) -> int:
        """
        需要连续： dp[i+1][j+1] = dp[i][j] + 1 if nums1[i] == nums[j] else 0
        """
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]
        result = 0
        for i in range(0, len(nums1)):
            for j in range(0, len(nums2)):
                if nums1[i] == nums2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                result = max(result, dp[i + 1][j + 1])
        return result

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        不连续就可以：dp[i+1][j+1] = dp[i][j] + 1 if nums1[i] == nums[j] else max(dp[i+1][j], dp[i][j+1])
        """
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        max_value = 0
        for i in range(len(text1)):
            for j in range(len(text2)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if text1[i] == text2[j] else max(dp[i + 1][j], dp[i][j + 1])
                max_value = max(max_value, dp[i + 1][j + 1])
        print(dp)
        return max_value

    def lengthOfLongestSubstring(self, s: str) -> int:
        # 判断第j个字符是否在dp[i-1]的区间之内 https://leetcode.cn/problems/longest-substring-without-repeating-characters/submissions/
        n = len(s)
        if n <= 1:
            return n
        ans = 0
        dp = [0] * n
        dp[0] = 1
        for i in range(n):
            j = i - 1
            while j >= 0 and s[i] != s[j]:
                j -= 1
                # 判断第j个字符是否在dp[i-1]的区间之内
            if i - j > dp[i - 1]:
                dp[i] = dp[i - 1] + 1
            else:
                dp[i] = i - j
            ans = max(ans, dp[i])
        return ans

    def isSubsequence(self, s: str, t: str) -> bool:
        """
          依赖下方，因为s在t里面，与s和t公共子序列不同
          dp[i+1][j+1] = dp[i][j] + 1 if s[i] == t[j] else dp[i+1][j]
        """
        max_value = 0
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
        for i in range(len(s)):
            for j in range(len(t)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if s[i] == t[j] else dp[i + 1][j]
                max_value = max(dp[i + 1][j + 1], max_value)
        return max_value == len(s)

    def _numDistinct(self, s: str, t: str) -> int:
        # rabbit
        """
           " r a b b b i t  = s             b a b g b a g
        "  1 1 1 1 1 1 1 1
        r  0 1 1 1 1 1 1 1               b  1 1 1 1 1 1 1
        a  0 0 1 1 1 1 1 1               a  1 2 2 2 2 1 1
        b  0 0 0 1 2 3 3 3               g  1 1 2 3 1 1 1
        b  0 0 0 0 1 3 3 3
        i  0 0 0 0 0 0 3 3
        t  0 0 0 0 0 0 0 3

        """
        n, m = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(n + 1):
            dp[0][i] = 1
        for i in range(0, m):
            for j in range(0, n):
                if t[i] == s[j]:
                    dp[i + 1][j + 1] = dp[i][j] + dp[i + 1][j]
                else:
                    dp[i + 1][j + 1] = dp[i + 1][j]
        return dp[-1][-1]

    def numDistinct(self, s: str, t: str) -> int:
        # s = "rabbbit", t = "rabbit"
        @cache
        def help(s, t):
            if len(s) < len(t):
                return 0
            if not t:
                return 1
            if not s:
                return 0
            # 相当于双指针，如果当前相等，则可以同时右移（匹配了一个字符），也可以只右移s的指针
            if s[0] == t[0]:
                return help(s[1:], t[1:]) + help(s[1:], t)
            # 如果当前不等，说明不匹配，只能只右移s的指针（舍弃s）
            else:
                return help(s[1:], t)

        return help(s, t)

    """
    给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock
    """

    def _maxProfit(self, prices: List[int]) -> int:
        # dp[i] = dp[i-1]
        dp = []
        dp.append(prices[0])
        profit = 0
        for i in range(1, len(prices)):
            dp.append(min(dp[i - 1], prices[i]))
            profit = max(profit, prices[i] - dp[i])
        return profit

    def _maxProfit(self, prices: List[int]) -> int:
        low = float("inf")
        result = 0
        for i in range(len(prices)):
            low = min(low, prices[i])  # 取最左最小价格
            result = max(result, prices[i] - low)  # 直接取最大区间利润
        return result

    def _maxProfit(self, prices: List[int]) -> int:
        # dp[i] = dp[i-1] + max(prices[i] - prices[i-1], 0)
        # dp[i] = min(dp[i-1], prices[i])
        res = [0]
        mem = []
        mem.append(prices[0])
        for i in range(1, len(prices)):
            min_value = min(mem[i - 1], prices[i])
            mem.append(min_value)
            res.append(prices[i] - min_value)
        return max(res)

    """
    输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        buy1 = buy2 = -prices[0]
        sell1 = sell2 = 0
        for i in range(1, n):
            buy1 = max(buy1, -prices[i])
            sell1 = max(sell1, buy1 + prices[i])
            buy2 = max(buy2, sell1 - prices[i])
            sell2 = max(sell2, buy2 + prices[i])
        return sell2

    """
    EDit distance
    输入：word1 = "horse", word2 = "ros"
    输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/edit-distance
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """

    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)

        # 有一个字符串为空串
        if n * m == 0:
            return n + m

        # DP 数组
        D = [[0] * (m + 1) for _ in range(n + 1)]

        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i
        for j in range(m + 1):
            D[0][j] = j

        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1]
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)

        return D[n][m]


if __name__ == '__main__':
    s = Solution()
    nums1 = [1, 2, 3, 2, 1]
    nums2 = [3, 2, 1, 4, 7, 8]
    s.longest_common_subarray(nums1=nums1, nums2=nums2)
    s.numDistinct(s="rabbbit", t="rabbit")
    s.minCost(costs=[[17, 2, 17], [16, 16, 5], [14, 3, 19]])
