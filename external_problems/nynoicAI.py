"""
@File    :   ai_test.py   
@Contact :   yinjialai 
"""
from collections import deque
from typing import List

mat = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 1]
]

output = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 2, 1]
]


class Solution:
    def nearest_distance_dp(self, mat: List[List]) -> List[List]:
        """
        dp method
        :param mat:
        :return:
        """
        rows = len(mat)
        cols = len(mat[0])
        dp = [[float('inf') for _ in range(cols)] for _ in range(rows)]

        # dp init
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    dp[i][j] = 0
        # left -> right, top -> button
        for i in range(rows):
            for j in range(cols):
                if i > 0 and i < rows:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
                if j > 0 and j < cols:
                    dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)

        for i in range(rows - 1, -1, -1):
            for j in range(cols - 1, -1, -1):
                if j < cols - 1:
                    dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)
                if i < rows - 1:
                    dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)

        return dp

    def nearest_distance_bfs(self, mat: List[List]) -> List[List]:
        """
        BFS method
        :param mat:
        :return:
        """
        # O(n * m)
        # initialization
        rows = len(mat)
        cols = len(mat[0])
        result = [[0 for _ in range(cols)] for _ in range(rows)]
        queue = deque()
        directions = (0, 1), (1, 0), (0, -1), (-1, 0)

        # queue
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    queue.append((i, j))

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # condition
                if 0 <= nx < rows and 0 <= ny < cols and mat[nx][ny] == 1 and result[nx][ny] == 0:
                    result[nx][ny] = result[x][y] + 1
                    queue.append((nx, ny))
        return result

if __name__ == '__main__':
    s = Solution()
    res = s.nearest_distance_dp(mat=mat)
    # result = nearest_distance(mat=mat)
    # print(result)
    print()
