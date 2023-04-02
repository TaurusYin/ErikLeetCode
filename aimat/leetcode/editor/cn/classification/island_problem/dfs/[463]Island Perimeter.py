# You are given row x col grid representing a map where grid[i][j] = 1 
# represents land and grid[i][j] = 0 represents water. 
# 
#  Grid cells are connected horizontally/vertically (not diagonally). The grid 
# is completely surrounded by water, and there is exactly one island (i.e., one or 
# more connected land cells). 
# 
#  The island doesn't have "lakes", meaning the water inside isn't connected to 
# the water around the island. One cell is a square with side length 1. The grid 
# is rectangular, width and height don't exceed 100. Determine the perimeter of 
# the island. 
# 
#  
#  Example 1: 
#  
#  
# Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
# Output: 16
# Explanation: The perimeter is the 16 yellow stripes in the image above.
#  
# 
#  Example 2: 
# 
#  
# Input: grid = [[1]]
# Output: 4
#  
# 
#  Example 3: 
# 
#  
# Input: grid = [[1,0]]
# Output: 4
#  
# 
#  
#  Constraints: 
# 
#  
#  row == grid.length 
#  col == grid[i].length 
#  1 <= row, col <= 100 
#  grid[i][j] is 0 or 1. 
#  There is exactly one island in grid. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 数组 矩阵 👍 645 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def __init__(self):
        self.perimeter = 0
    # 判断前后左右 是边界外或者是0|1交界处
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        def dfs(x, y, path=[]):
            if visited[x][y]:
                return path
            path.append((x, y))
            # print(f"append : {path}")
            visited[x][y] = True
            if (x - 1 < 0 or grid[x - 1][y] == 0) and grid[x][y] == 1:
                self.perimeter += 1
            if (x + 1 >= m or grid[x + 1][y] == 0) and grid[x][y] == 1:
                self.perimeter += 1
            if (y - 1 < 0 or grid[x][y - 1] == 0) and grid[x][y] == 1:
                self.perimeter += 1
            if (y + 1 >= n or grid[x][y + 1] == 0) and grid[x][y] == 1:
                self.perimeter += 1
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] == 1:
                    dfs(new_x, new_y, path)
            return path

        m = len(grid)
        n = len(grid[0])
        visited = [[False] * n for _ in range(m)]
        total_path = []
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    cur_path = dfs(x, y, [])
                    if cur_path:
                        total_path.append(cur_path)
        print(total_path)
        return self.perimeter


grid = [[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
Solution().islandPerimeter(grid=grid)
# leetcode submit region end(Prohibit modification and deletion)
