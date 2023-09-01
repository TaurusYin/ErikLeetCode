# Given a 2D grid consists of 0s (land) and 1s (water). An island is a maximal 4
# -directionally connected group of 0s and a closed island is an island totally (
# all left, top, right, bottom) surrounded by 1s. 
# 
#  Return the number of closed islands. 
# 
#  
#  Example 1: 
# 
#  
# 
#  
# Input: grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,
# 0,1,0,1],[1,1,1,1,1,1,1,0]]
# Output: 2
# Explanation: 
# Islands in gray are closed because they are completely surrounded by water (
# group of 1s). 
# 
#  Example 2: 
# 
#  
# 
#  
# Input: grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
# Output: 1
#  
# 
#  Example 3: 
# 
#  
# Input: grid = [[1,1,1,1,1,1,1],
#                [1,0,0,0,0,0,1],
#                [1,0,1,1,1,0,1],
#                [1,0,1,0,1,0,1],
#                [1,0,1,1,1,0,1],
#                [1,0,0,0,0,0,1],
#                [1,1,1,1,1,1,1]]
# Output: 2
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= grid.length, grid[0].length <= 100 
#  0 <= grid[i][j] <=1 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 并查集 数组 矩阵 👍 187 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    # 先遍历边界，再遍历中间
    def closedIsland(self, grid: List[List[int]]) -> int:
        # 深度优先搜索函数
        def dfs(i, j):
            # 边界条件判断
            if i < 0 or i >= m or j < 0 or j >= n:
                return
            # 如果当前点是水域或者已经遍历过了，直接返回
            if grid[i][j] == 1 or visited[i][j]:
                return
            # 标记当前点已经遍历过了
            visited[i][j] = True
            # 沿着四个方向继续搜索
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        m, n = len(grid), len(grid[0])
        # 初始化所有的点都没有被遍历过
        visited = [[False] * n for _ in range(m)]
        # 遍历边界上的点，标记为已经遍历过了
        for i in range(m):
            dfs(i, 0)
            dfs(i, n - 1)
        for j in range(n):
            dfs(0, j)
            dfs(m - 1, j)

        res = 0
        # 统计所有没有被遍历过的点的数量
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if grid[i][j] == 0 and not visited[i][j]:
                    dfs(i, j)
                    res += 1

        return res

    def closedIsland(self, grid: List[List[int]]) -> int:
        def dfs(x, y, path=[]):
            if visited[x][y]:
                return path
            path.append([x, y])
            visited[x][y] = True
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] == 0:
                    dfs(new_x, new_y, path)
            return path

        m = len(grid)
        n = len(grid[0])
        total_path = []
        visited = [[False] * n for _ in range(m)]
        for x in range(1, m - 1):
            for y in range(1, n - 1):
                if grid[x][y] == 0:
                    cur_path = dfs(x, y, [])
                    flag = False
                    for xx, yy in cur_path:
                        if grid[xx][yy] == 0 and (xx == 0 or yy == 0 or xx == m - 1 or yy == n - 1):
                            flag = True
                            break
                    if not flag and cur_path:
                        total_path.append(cur_path)
        return len(total_path)



grid = [[0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0]]


grid = [[1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]]


grid = [
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]]
grid = [[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]]


Solution().closedIsland(grid=grid)

# leetcode submit region end(Prohibit modification and deletion)
