"""
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
"""


def _rotate(matrix: List[List[int]]) -> None:
    # https://leetcode.cn/problems/rotate-image/submissions/
    n = len(matrix)
    for i in range(n // 2):
        for j in range((n + 1) // 2):
            matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] \
                = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]


def _rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    # 水平翻转
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
    # 主对角线翻转
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


#
def rotate_clockwise(matrix: List[List[int]]) -> None:
    matrix = list(zip(*matrix[::-1]))
    return matrix


def rotate_non_clockwise(matrix: List[List[int]]) -> None:
    matrix = list(zip(*matrix))[::-1]
    return matrix


"""
二分查找 搜索二维矩阵
https://leetcode.cn/problems/search-a-2d-matrix-ii/solution/sou-suo-er-wei-ju-zhen-ii-by-leetcode-so-9hcx/
"""


def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    for row in matrix:
        idx = bisect.bisect_left(row, target)
        if idx < len(row) and row[idx] == target:
            return True
    return False


"""
Z 字形查找
"""


def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    x, y = 0, n - 1
    while x < m and y >= 0:
        if matrix[x][y] == target:
            return True
        if matrix[x][y] > target:
            y -= 1
        else:
            x += 1
    return False


def generateMatrix(self, n: int) -> List[List[int]]:
    # https://leetcode.cn/problems/spiral-matrix-ii/
    left, right, up, down = 0, n - 1, 0, n - 1
    matrix = [[0] * n for _ in range(n)]
    num = 1
    while left <= right and up <= down:
        # 填充左到右
        for i in range(left, right + 1):
            matrix[up][i] = num
            num += 1
        up += 1
        # 填充上到下
        for i in range(up, down + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        # 填充右到左
        for i in range(right, left - 1, -1):
            matrix[down][i] = num
            num += 1
        down -= 1
        # 填充下到上
        for i in range(down, up - 1, -1):
            matrix[i][left] = num
            num += 1
        left += 1
    return matrix

"""
用原矩阵的第一行和第一列标记哪些行列含有0
空间复杂度为O(1)
https://leetcode.cn/problems/set-matrix-zeroes/solution/73-ju-zhen-zhi-ling-python-dfs-by-bluegr-sel7/
"""
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        n = len(matrix[0])
        flag_col0 = any(matrix[i][0] == 0 for i in range(m))  # 是否原本有0
        flag_row0 = any(matrix[0][j] == 0 for j in range(n))

        for i in range(1, len(matrix)):  # 用第一行和第一列来标记哪些行列有0
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        for i in range(1, m):  # 按列处理
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0

        for j in range(1, n):  # 按行处理
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0

        if flag_col0:  # 处理本来就有0的情况
            for i in range(m):
                matrix[i][0] = 0

        if flag_row0:  # 处理本来就有0的情况
            for j in range(n):
                matrix[0][j] = 0


