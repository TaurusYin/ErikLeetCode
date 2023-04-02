# Given an m x n 2D binary grid grid which represents a map of '1's (land) and 
# '0's (water), return the number of islands. 
# 
#  An island is surrounded by water and is formed by connecting adjacent lands 
# horizontally or vertically. You may assume all four edges of the grid are all 
# surrounded by water. 
# 
#  
#  Example 1: 
# 
#  
# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1
#  
# 
#  Example 2: 
# 
#  
# Input: grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3
#  
# 
#  
#  Constraints: 
# 
#  
#  m == grid.length 
#  n == grid[i].length 
#  1 <= m, n <= 300 
#  grid[i][j] is '0' or '1'. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 并查集 数组 矩阵 👍 2120 👎 0
from typing import List


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def _numIslands(self, grid: List[List[str]]) -> int:
        def dfs(x, y):
            if visited[x][y]:
                return 0
            visited[x][y] = True
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if new_x < 0 or new_x >= m or new_y < 0 or new_y >= n:
                    continue
                if grid[new_x][new_y] == "1":
                    dfs(new_x, new_y)
            return 1

        m = len(grid)
        n = len(grid[0])
        num_res = 0
        visited = [[False] * n for _ in range(m)]
        for x in range(m):
            for y in range(n):
                if grid[x][y] == "1":
                    num_res += dfs(x, y)
        return num_res

    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        visited = set()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and (i, j) not in visited:
                    count += 1
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1' and (nx, ny) not in visited:
                                queue.append((nx, ny))
                                visited.add((nx, ny))
        return count


# leetcode submit region end(Prohibit modification and deletion)
grid = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]
Solution().numIslands(grid=grid)
