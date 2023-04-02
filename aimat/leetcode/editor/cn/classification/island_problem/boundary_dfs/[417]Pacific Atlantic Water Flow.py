# There is an m x n rectangular island that borders both the Pacific Ocean and 
# Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and 
# the Atlantic Ocean touches the island's right and bottom edges. 
# 
#  The island is partitioned into a grid of square cells. You are given an m x 
# n integer matrix heights where heights[r][c] represents the height above sea 
# level of the cell at coordinate (r, c). 
# 
#  The island receives a lot of rain, and the rain water can flow to 
# neighboring cells directly north, south, east, and west if the neighboring cell's height 
# is less than or equal to the current cell's height. Water can flow from any cell 
# adjacent to an ocean into the ocean. 
# 
#  Return a 2D list of grid coordinates result where result[i] = [ri, ci] 
# denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic 
# oceans. 
# 
#  
#  Example 1: 
#  
#  
# Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
# 
# Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
# Explanation: The following cells can flow to the Pacific and Atlantic oceans, 
# as shown below:
# [0,4]: [0,4] -> Pacific Ocean 
#        [0,4] -> Atlantic Ocean
# [1,3]: [1,3] -> [0,3] -> Pacific Ocean 
#        [1,3] -> [1,4] -> Atlantic Ocean
# [1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
#        [1,4] -> Atlantic Ocean
# [2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
#        [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
# [3,0]: [3,0] -> Pacific Ocean 
#        [3,0] -> [4,0] -> Atlantic Ocean
# [3,1]: [3,1] -> [3,0] -> Pacific Ocean 
#        [3,1] -> [4,1] -> Atlantic Ocean
# [4,0]: [4,0] -> Pacific Ocean 
#        [4,0] -> Atlantic Ocean
# Note that there are other possible paths for these cells to flow to the 
# Pacific and Atlantic oceans.
#  
# 
#  Example 2: 
# 
#  
# Input: heights = [[1]]
# Output: [[0,0]]
# Explanation: The water can flow from the only cell to the Pacific and 
# Atlantic oceans.
#  
# 
#  
#  Constraints: 
# 
#  
#  m == heights.length 
#  n == heights[r].length 
#  1 <= m, n <= 200 
#  0 <= heights[r][c] <= 10⁵ 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 数组 矩阵 👍 582 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List


class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        m = len(heights)
        n = len(heights[0])
        visited_pacific = [[False] * n for _ in range(m)]
        visited_atlantic = [[False] * n for _ in range(m)]
        print(visited_pacific)

        def dfs(x, y, visited):
            visited[x][y] = True
            directions = [[-1, 0], [0, -1], [1, 0], [0, 1]]
            for dx, dy in directions:
                new_x = x + dx
                new_y = y + dy
                if new_x < 0 or new_y < 0:
                    continue
                if new_x >= m or new_y >= n:
                    continue
                if visited[new_x][new_y]:
                    continue
                if heights[new_x][new_y] >= heights[x][y]:
                    dfs(new_x, new_y, visited)

        # 按照行遍历左侧和右侧边界
        for i in range(m):
            dfs(i, 0, visited_pacific)
            dfs(i, n - 1, visited_atlantic)
        # 按列遍历上侧和下侧边界
        for i in range(n):
            dfs(0, i, visited_pacific)
            dfs(m - 1, i, visited_atlantic)

        result = []
        for i in range(m):
            for j in range(n):
                if visited_atlantic[i][j] and visited_pacific[i][j]:
                    result.append([i, j])
        return result

    def _pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []

        m, n = len(heights), len(heights[0])
        pacific_visited = [[False] * n for _ in range(m)]
        atlantic_visited = [[False] * n for _ in range(m)]

        def dfs(x, y, visited):
            visited[x][y] = True
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if new_x < 0 or new_x >= m or new_y < 0 or new_y >= n:
                    continue
                if visited[new_x][new_y]:
                    continue
                if heights[new_x][new_y] >= heights[x][y]:
                    dfs(new_x, new_y, visited)

        for i in range(m):
            dfs(i, 0, pacific_visited)
            dfs(i, n - 1, atlantic_visited)
        for i in range(n):
            dfs(0, i, pacific_visited)
            dfs(m - 1, i, atlantic_visited)

        res = []
        for i in range(m):
            for j in range(n):
                if pacific_visited[i][j] and atlantic_visited[i][j]:
                    res.append([i, j])

        return res


heights = [[1, 2, 2, 3, 5],
           [3, 2, 3, 4, 4],
           [2, 4, 5, 3, 1],
           [6, 7, 1, 4, 5],
           [5, 1, 1, 2, 4]]

heights = [[1, 1], [1, 1], [1, 1]]
Solution().pacificAtlantic(heights=heights)
# leetcode submit region end(Prohibit modification and deletion)
