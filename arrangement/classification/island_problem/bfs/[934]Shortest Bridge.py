# You are given an n x n binary matrix grid where 1 represents land and 0
# represents water. 
# 
#  An island is a 4-directionally connected group of 1's not connected to any 
# other 1's. There are exactly two islands in grid. 
# 
#  You may change 0's to 1's to connect the two islands to form one island. 
# 
#  Return the smallest number of 0's you must flip to connect the two islands. 
# 
#  
#  Example 1: 
# 
#  
# Input: grid = [[0,1],[1,0]]
# Output: 1
#  
# 
#  Example 2: 
# 
#  
# Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
# Output: 2
#  
# 
#  Example 3: 
# 
#  
# Input: grid = [
# [1,1,1,1,1],
# [1,0,0,0,1],
# [1,0,1,0,1],
# [1,0,0,0,1],
# [1,1,1,1,1]
# ]
# Output: 1
#  
# 
#  
#  Constraints: 
# 
#  
#  n == grid.length == grid[i].length 
#  2 <= n <= 100 
#  grid[i][j] is either 0 or 1. 
#  There are exactly two islands in grid. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 数组 矩阵 👍 438 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import deque


class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        """
        找到道第一个"1"， 再BFS, BFS为1时候且是第0层，left append
        :param grid:
        :return:
        """

        def bfs(x, y):
            queue = deque([(x, y, 0)])
            while queue:
                xx, yy, level = queue.popleft()
                for new_x, new_y in [(xx + 1, yy), (xx - 1, yy), (xx, yy + 1), (xx, yy - 1)]:
                    if 0 <= new_x < n and 0 <= new_y < n and not visited[new_x][new_y]:
                        visited[new_x][new_y] = True
                        if grid[new_x][new_y] == 1:
                            if level == 0:
                                queue.appendleft((new_x, new_y, level))
                            else:
                                return level
                        if grid[new_x][new_y] == 0:
                            queue.append((new_x, new_y, level + 1))

        n = len(grid)
        visited = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    visited[i][j] = True
                    return bfs(i, j)


def __shortestBridge(self, grid: List[List[int]]) -> int:
    def dfs(x, y):
        if visited[x][y]:
            return
        visited[x][y] = True
        queue_path.append((x, y, 0))
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        for delta_x, delta_y in directions:
            new_x = x + delta_x
            new_y = y + delta_y
            if new_x < 0 or new_x >= n or new_y < 0 or new_y >= n:
                continue
            if grid[new_x][new_y] == 1:
                dfs(new_x, new_y)
        return

    # DFS 找第一个岛的所有坐标
    n = len(grid)
    visited = [[False] * n for _ in range(n)]
    queue_path = deque()
    flag = True
    for x in range(n):
        for y in range(n):
            if grid[x][y] == 1 and flag:
                dfs(x, y)
                flag = False
                break

    # BFS 一层一层向外拓展知道找到另一个岛屿
    while queue_path:
        x, y, level = queue_path.popleft()
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        for delta_x, delta_y in directions:
            new_x = x + delta_x
            new_y = y + delta_y
            if new_x < 0 or new_x >= n or new_y < 0 or new_y >= n:
                continue
            if grid[new_x][new_y] == 0 and not visited[new_x][new_y]:
                visited[new_x][new_y] = True
                grid[new_x][new_y] = 1
                queue_path.append((new_x, new_y, level + 1))
            if grid[new_x][new_y] == 1 and not visited[new_x][new_y]:
                return level


grid = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]

grid = [[0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]]
Solution().shortestBridge(grid=grid)
# leetcode submit region end(Prohibit modification and deletion)
