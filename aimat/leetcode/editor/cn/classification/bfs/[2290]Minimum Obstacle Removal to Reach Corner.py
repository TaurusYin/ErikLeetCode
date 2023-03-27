# You are given a 0-indexed 2D integer array grid of size m x n. Each cell has 
# one of two values: 
# 
#  
#  0 represents an empty cell, 
#  1 represents an obstacle that may be removed. 
#  
# 
#  You can move up, down, left, or right from and to an empty cell. 
# 
#  Return the minimum number of obstacles to remove so you can move from the 
# upper left corner (0, 0) to the lower right corner (m - 1, n - 1). 
# 
#  
#  Example 1: 
#  
#  
# Input: grid = [[0,1,1],[1,1,0],[1,1,0]]
# Output: 2
# Explanation: We can remove the obstacles at (0, 1) and (0, 2) to create a 
# path from (0, 0) to (2, 2).
# It can be shown that we need to remove at least 2 obstacles, so we return 2.
# Note that there may be other ways to remove 2 obstacles to create a path.
#  
# 
#  Example 2: 
#  
#  
# Input: grid = [[0,1,0,0,0],[0,1,0,1,0],[0,0,0,1,0]]
# Output: 0
# Explanation: We can move from (0, 0) to (2, 4) without removing any obstacles,
#  so we return 0.
#  
# 
#  
#  Constraints: 
# 
#  
#  m == grid.length 
#  n == grid[i].length 
#  1 <= m, n <= 10⁵ 
#  2 <= m * n <= 10⁵ 
#  grid[i][j] is either 0 or 1. 
#  grid[0][0] == grid[m - 1][n - 1] == 0 
#  
# 
#  Related Topics 广度优先搜索 图 数组 矩阵 最短路 堆（优先队列） 👍 40 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import deque


class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        def bfs(x, y):
            queue = deque([(x, y, 0)])
            while queue:
                xx, yy, level = queue.popleft()
                for new_x, new_y in [(xx + 1, yy), (xx - 1, yy), (xx, yy + 1), (xx, yy - 1)]:
                    if 0 <= new_x < m and 0 <= new_y < n and not visited[new_x][new_y]:
                        visited[new_x][new_y] = True
                        if grid[new_x][new_y] == 0:
                            queue.appendleft((new_x, new_y, level))
                        if grid[new_x][new_y] == 1:
                            queue.append((new_x, new_y, level + 1))
                        if new_x == m - 1 and new_y == n - 1:
                            return level

        m = len(grid)
        n = len(grid[0])
        visited = [[False] * n for _ in range(m)]
        return bfs(0, 0)


grid = [[0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0]]
Solution().minimumObstacles(grid=grid)
# leetcode submit region end(Prohibit modification and deletion)
