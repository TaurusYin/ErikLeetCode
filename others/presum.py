from typing import List


def diff(nums):
    n = len(nums)
    diff = [0] * n
    diff[0] = nums[0]
    for i in range(n - 1):
        diff[i + 1] = nums[i + 1] - nums[i]
    return diff


def inc(diff_nums, start, end, val):
    diff_nums[start] += val
    if end + 1 < len(diff_nums):
        diff_nums[end + 1] -= val
    return diff_nums


def diff_to_res(diff_nums):
    res = [0] * len(diff_nums)
    res[0] = diff_nums[0]
    for i in range(1, len(diff_nums)):
        res[i] = res[i - 1] + diff_nums[i]
    return res


class NumArray:
    def __init__(self, nums: List[int]):
        # self.preSum里先放一个0，即列表里总共放n+1个元素
        N = len(nums)
        self.preSum = [0] * (N + 1)
        for i in range(N):
            self.preSum[i + 1] = self.preSum[i] + nums[i]

    def sumRange(self, left: int, right: int) -> int:
        # 查询闭区间的累加和
        # 在self.preSum里，index=right + 1时，是加了index=right这个元素的值
        # 因此闭区间[left, right]的累加和等于下面
        return self.preSum[right + 1] - self.preSum[left]


class NumMatrix:
    # https://leetcode.cn/problems/O4NDxx/
    """
    [i+r][j+r] + [i - r - 1][j + r] + [i + r][j - r - 1] - [i - r - 1][j - r - 1]
    其实最优化的算法是遍历一遍所有的cell，用每个cell记录该点和[0,0]构成的长方形的和，
    那么算其中每个点的值就可以用[i+r][j+r] + [i - r - 1][j + r] + [i + r][j - r - 1] - [i - r - 1][j - r - 1]
    来求得，注意超出边界的部分需要特殊处理一下。按楼主的解法时间复杂度还是M*N*R，下面这个解法是M*N的复杂度。
    preSum[r][c]=preSum[r][c−1]+preSum[r−1][c]−preSum[r−1][c−1]+mat[r][c]
    """

    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix
        m = len(matrix)
        n = len(matrix[0])
        self.pre_sum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.pre_sum[i + 1][j + 1] = self.pre_sum[i + 1][j] + self.pre_sum[i][j + 1] - self.pre_sum[i][j] + \
                                             self.matrix[i][j]
        print(self.pre_sum)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.pre_sum[row2 + 1][col2 + 1] + self.pre_sum[row1][col1] - self.pre_sum[row2 + 1][col1] - \
               self.pre_sum[row1][col2 + 1]

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        """
            [i+r][j+r] + [i - r - 1][j + r] + [i + r][j - r - 1] - [i - r - 1][j - r - 1]
            其实最优化的算法是遍历一遍所有的cell，用每个cell记录该点和[0,0]构成的长方形的和，
            那么算其中每个点的值就可以用[i+r][j+r] + [i - r - 1][j + r] + [i + r][j - r - 1] - [i - r - 1][j - r - 1]
            来求得，注意超出边界的部分需要特殊处理一下。按楼主的解法时间复杂度还是M*N*R，下面这个解法是M*N的复杂度。
            preSum[r][c]=preSum[r][c−1]+preSum[r−1][c]−preSum[r−1][c−1]+mat[r][c]
        """

        m, n = len(mat), len(mat[0])
        p = [[0] * (1 + n) for _ in range(1 + m)]
        # 二维前缀和
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                p[i][j] = p[i - 1][j] + p[i][j - 1] - p[i - 1][j - 1] + mat[i - 1][j - 1]

        # 任意矩形和
        def get(x1, y1, x2, y2):
            return p[x2][y2] - p[x2][y1 - 1] - p[x1 - 1][y2] + p[x1 - 1][y1 - 1]

        # 计算基于i,j的区域和
        res = []
        for i in range(1, m + 1):
            tmp = []
            for j in range(1, n + 1):
                q = get(max(i - k, 1), max(j - k, 1), min(i + k, m), min(j + k, n))
                tmp.append(q)
            res.append(tmp)
        return res


class Solution:
    # https://leetcode.cn/problems/car-pooling/submissions/
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        trips = [[2,1,5],[3,3,7]], capacity = 4
        给定整数capacity和一个数组 trips , trip[i] = [numPassengersi, fromi, to
        表示第 i 次旅行numPassengers乘客，接他们和放他们的位置分别是fromi和toi。这些位置是从汽车的初始位置向东的公里数。
        :param trips:
        :param capacity:
        :return:
        """
        dp = [0] * 1000
        for trip in trips:
            num_passengers, start, end = trip[0], trip[1], trip[2]
            for i in range(start, end):
                dp[i] += num_passengers
        return capacity >= max(dp)


if __name__ == '__main__':
    s = Solution()
    x = diff([8, 2, 6, 3, 1])
    y = diff_to_res(x)
    print()
