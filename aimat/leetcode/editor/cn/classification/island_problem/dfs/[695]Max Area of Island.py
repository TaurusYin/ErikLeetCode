# You are given an m x n binary matrix grid. An island is a group of 1's (
# representing land) connected 4-directionally (horizontal or vertical.) You may assume 
# all four edges of the grid are surrounded by water. 
# 
#  The area of an island is the number of cells with a value 1 in the island. 
# 
#  Return the maximum area of an island in grid. If there is no island, return 0
# . 
# 
#  
#  Example 1: 
#  
#  
# Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,
# 0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,
# 0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]
# ]
# Output: 6
# Explanation: The answer is not 11, because the island must be connected 4-
# directionally.
#  
# 
#  Example 2: 
# 
#  
# Input: grid = [[0,0,0,0,0,0,0,0]]
# Output: 0
#  
# 
#  
#  Constraints: 
# 
#  
#  m == grid.length 
#  n == grid[i].length 
#  1 <= m, n <= 50 
#  grid[i][j] is either 0 or 1. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 并查集 数组 矩阵 👍 943 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def maxAreaOfIsland_dfs(self, grid: List[List[int]]) -> int:
        def dfs(x, y, path=[], area=0):
            if visited[x][y]:
                return area
            visited[x][y] = True
            path.append([x, y])
            area += 1
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if new_x < 0 or new_x >= m or new_y < 0 or new_y >= n:
                    continue
                if grid[new_x][new_y] == 1:
                    area = dfs(new_x, new_y, path, area)
            return area

        m = len(grid)
        n = len(grid[0])
        num_res = 0
        visited = [[False] * n for _ in range(m)]
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    num_res = max(num_res, dfs(x, y, [], 0))
        return num_res

    def maxAreaOfIsland_bfs(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]  # 初始化 visited 数组
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 定义上下左右四个方向

        max_area = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    area = 1  # 记录岛屿面积
                    visited[i][j] = True  # 标记该位置已访问
                    queue = deque([(i, j)])  # 将该位置加入队列

                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in directions:
                            next_x, next_y = x + dx, y + dy
                            if 0 <= next_x < m and 0 <= next_y < n and grid[next_x][next_y] == 1 and not \
                                visited[next_x][next_y]:
                                visited[next_x][next_y] = True
                                area += 1
                                queue.append((next_x, next_y))

                    max_area = max(max_area, area)

        return max_area

    def _maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 0
        rows, cols = len(grid), len(grid[0])

        def dfs(r, c, area):
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 1:
                grid[r][c] = 0  # 标记为已访问
                area += 1
                area = dfs(r + 1, c, area)
                area = dfs(r - 1, c, area)
                area = dfs(r, c + 1, area)
                area = dfs(r, c - 1, area)
            return area

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    area = dfs(r, c, 0)
                    max_area = max(max_area, area)

        return max_area

grid = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
Solution().maxAreaOfIsland(grid=grid)
# leetcode submit region end(Prohibit modification and deletion)
