import functools
from bisect import bisect
from cmath import inf
from functools import cache
from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [inf] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin]) + 1
        return dp[-1] if dp[-1] != inf else -1

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
        """
        无重复最长字串
        :param s:
        :return:
        """
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

    """
    字符串成为回文串的最少插入次数
    给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。
    请你返回让 s 成为回文串的 最少操作次数 。
    「回文串」是正读和反读都相同的字符串。
    输入：s = "zzazz"
    输出：0
    解释：字符串 "zzazz" 已经是回文串了，所以不需要做任何插入操作。
    我们用 dp[i][j] 表示对于字符串 s 的子串 s[i:j]（这里的下标从 0 开始，并且 s[i:j] 包含 s 中的第 i 和第 j 个字符），最少添加的字符数量，使得 s[i:j] 变为回文串。
    我们从外向内考虑 s[i:j]：
    如果 s[i] == s[j]，那么最外层已经形成了回文，我们只需要继续考虑 s[i+1:j-1]；
    如果 s[i] != s[j]，那么我们要么在 s[i:j] 的末尾添加字符 s[i]，要么在 s[i:j] 的开头添加字符 s[j]，才能使得最外层形成回文。如果我们选择前者，那么需要继续考虑 s[i+1:j]；如果我们选择后者，那么需要继续考虑 s[i:j-1]。
    dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1)                     if s[i] != s[j]
    dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1, dp[i + 1][j - 1])   if s[i] == s[j]
    O(N**2)，其中 NN 是字符串 s 的长度
    链接：https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/solution/rang-zi-fu-chuan-cheng-wei-hui-wen-chuan-de-zui--2/
    """

    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span - 1
                dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1
                if s[i] == s[j]:
                    dp[i][j] = min(dp[i][j], dp[i + 1][j - 1])
        return dp[0][n - 1]


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


"""
https://leetcode.cn/problems/climbing-stairs/solution/zhi-xin-hua-shi-pa-lou-ti-zhi-cong-bao-l-lo1t/
"""
# 直接递归解法，容易超时，python可以加个缓存装饰器，这样也算是将递归转换成迭代的形式了
# 除了这种方式，还有增加步长来递归，变相的减少了重复计算
# 还有一种方法，在递归的同时，用数组记忆之前得到的结果，也是减少重复计算

class Solution:
    @functools.lru_cache(100)  # 缓存装饰器
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)

    # 直接DP，新建一个字典或者数组来存储以前的变量，空间复杂度O(n)


class Solution:
    def climbStairs(self, n: int) -> int:
        dp = {}
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    # 还是DP，只不过是只存储前两个元素，减少了空间，空间复杂度O(1)


class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1 or n == 2: return n
        a, b, temp = 1, 2, 0
        for i in range(3, n + 1):
            temp = a + b
            a = b
            b = temp
        return temp

    # 直接斐波那契数列的计算公式喽


class Solution:
    def climbStairs(self, n: int) -> int:
        import math
        sqrt5 = 5 ** 0.5
        fibin = math.pow((1 + sqrt5) / 2, n + 1) - math.pow((1 - sqrt5) / 2, n + 1)
        return int(fibin / sqrt5)


"""
https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg
圆环上有10个点，编号为0~9。从0点出发，每次可以逆时针和顺时针走一步，问走n步回到0点共有多少种走法。

输入: 2
输出: 2
解释：有2种方案。分别是0->1->0和0->9->0
如果你之前做过leetcode的70题爬楼梯，则应该比较容易理解：走n步到0的方案数=走n-1步到1的方案数+走n-1步到9的方案数。
因此，若设dp[i][j]为从0点出发走i步到j点的方案数，则递推式为：
Image
ps:公式之所以取余是因为j-1或j+1可能会超过圆环0~9的范围
"""


def backToOrigin(self, n):
    # 点的个数为10
    length = 10
    dp = [[0 for i in range(length)] for j in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(length):
            # dp[i][j]表示从0出发，走i步到j的方案数
            dp[i][j] = dp[i - 1][(j - 1 + length) % length] + dp[i - 1][(j + 1) % length]
    return dp[n][0]


"""
https://leetcode.cn/problems/house-robber/
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
链接：https://leetcode.cn/problems/house-robber
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

"""


def rob(self, nums: List[int]) -> int:
    if not nums:
        return 0
    size = len(nums)
    if size == 1:
        return nums[0]
    dp = [0] * size
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, size):
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    print(dp)
    return dp[size - 1]


"""
https://leetcode.cn/problems/house-robber-ii/solution/ji-yu-da-jia-jie-she-1-by-limtcyt-4-hztm/
情况一：考虑不包含首尾元素
情况二：考虑包含首元素，不包含尾元素
情况三：考虑包含尾元素，不包含首元素
"""


def rob(self, nums: List[int]) -> int:
    '''基于打家劫舍1'''
    """
    分情况讨论：
    plan1, 去掉0，剩余部分按不成环考虑
    plan2，去掉-1，剩余部分按不成环考虑
    比较二者取其优
    """

    def rob_noncicle(cost: list) -> int:
        '''不成环情况的打家劫舍'''
        n = len(cost)
        if n == 0:
            return 0
        elif n == 1:
            return cost[0]
        elif n == 2:
            return max(cost)
        dp = [0] * n
        dp[0] = cost[0]
        dp[1] = max(cost[0], cost[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + cost[i])
        return dp[n - 1]

    if len(nums) == 1:
        return nums[0]
    # 算出两个方案并比较，取其优
    return max(rob_noncicle(nums[1:]), rob_noncicle(nums[:-1]))


"""
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
O(n*target)
https://leetcode.cn/problems/partition-equal-subset-sum/solution/fen-ge-deng-he-zi-ji-by-leetcode-solution/
"""
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2:
            return False

        total = sum(nums)
        if total % 2 != 0:
            return False

        target = total // 2
        dp = [True] + [False] * target
        for i, num in enumerate(nums):
            for j in range(target, num - 1, -1):
                dp[j] |= dp[j - num]

        return dp[target]

"""
https://leetcode.cn/problems/target-sum/solution/494-mu-biao-he-dong-tai-gui-hua-zhi-01be-78ll/
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
"""
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sumAll = sum(nums)
        if S > sumAll or (S + sumAll) % 2:
            return 0
        target = (S + sumAll) // 2

        dp = [0] * (target + 1)
        dp[0] = 1

        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] + dp[j - num]
        return dp[-1]





if __name__ == '__main__':
    s = Solution()
    nums1 = [1, 2, 3, 2, 1]
    nums2 = [3, 2, 1, 4, 7, 8]
    s.longest_common_subarray(nums1=nums1, nums2=nums2)
    s.numDistinct(s="rabbbit", t="rabbit")
    s.minCost(costs=[[17, 2, 17], [16, 16, 5], [14, 3, 19]])
