from typing import List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def bt(cur_index):
            nonlocal max_sum
            if cur_index < 0:
                return 0

            pre_sum = bt(cur_index - 1)
            cur_max_sum = max(pre_sum + nums[cur_index], nums[cur_index])
            max_sum = max(max_sum, cur_max_sum)
            return cur_max_sum

        max_sum = nums[0]
        bt(len(nums) - 1)
        return max_sum

    def _maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]

        def bt(nums, cur_index, pre_sum):
            nonlocal max_sum
            if (cur_index >= len(nums)):
                return
            cur_sum = pre_sum + nums[cur_index]
            max_sum = max(max_sum, cur_sum)
            # not choose
            if cur_sum < 0:
                bt(nums, cur_index + 1, 0)
            # choose
            else:
                bt(nums, cur_index + 1, pre_sum + nums[cur_index])

        bt(nums, 0, 0)
        return max_sum

    def maxProfit(self, prices: List[int]) -> int:
        """
        输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
        :param prices:
        :return:
        """
        profit = 0
        for i in range(0, len(prices) - 1):
            profit += max(prices[i + 1] - prices[i], 0)
        return profit

    """
    输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """
    def maxProfit(self, prices: List[int]) -> int:
        def bt(prices, start_index, cash, status):
            if start_index == len(prices):
                res.append(cash)
                return
            diff = 0
            if start_index + 1 < len(prices):
                diff = prices[start_index + 1] - prices[start_index]
                print(diff)
            if status == 0:
                if diff > 0:
                    # buy
                    bt(prices, start_index + 1, cash - prices[start_index], 1)
                else:
                    # not operations
                    bt(prices, start_index + 1, cash, status)
            elif status == 1:
                if diff <= 0:
                    # sell
                    bt(prices, start_index + 1, cash + prices[start_index], 0)
                else:
                    # not operations
                    bt(prices, start_index + 1, cash, status)

        cash = 0
        res = []
        bt(prices, 0, cash, 0)
        return max(res)

    def _maxProfit(self, prices: List[int]) -> int:
        dp = [0]
        # dp[i] = dp[i-1] + max(prices[i] - prices[i-1], 0)
        for i in range(1, len(prices)):
            dp.append(dp[i - 1] + max(prices[i] - prices[i - 1], 0))
        return max(dp)

    def _maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(0, len(prices) - 1):
            profit += max(prices[i + 1] - prices[i], 0)
        return profit



if __name__ == '__main__':
    s = Solution()
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    s.maxSubArray(nums=nums)













