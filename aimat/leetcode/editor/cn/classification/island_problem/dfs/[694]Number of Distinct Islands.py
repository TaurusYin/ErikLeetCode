# You are given an m x n binary matrix grid. An island is a group of 1's (
# representing land) connected 4-directionally (horizontal or vertical.) You may assume 
# all four edges of the grid are surrounded by water. 
# 
#  An island is considered to be the same as another if and only if one island 
# can be translated (and not rotated or reflected) to equal the other. 
# 
#  Return the number of distinct islands. 
# 
#  
#  Example 1: 
#  
#  
# Input: grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
# Output: 1
#  
# 
#  Example 2: 
#  
#  
# Input: grid = [[1,1,0,1,1],[1,0,0,0,0],[0,0,0,0,1],[1,1,0,1,1]]
# Output: 3
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
#  Related Topics 深度优先搜索 广度优先搜索 并查集 哈希表 哈希函数 👍 155 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from collections import defaultdict
from typing import List


# 把direction编码
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        def dfs(x, y, path=[], dirs=[]):
            if visited[x][y]:
                return path, dirs
            path.append((x, y))
            visited[x][y] = True
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for index, (delta_x, delta_y) in enumerate(directions):
                new_x = x + delta_x
                new_y = y + delta_y
                if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] == 1:
                    dirs.append(index)
                    dfs(new_x, new_y, path, dirs)
                    dirs.append(-index)
            return path, dirs

        m = len(grid)
        n = len(grid[0])
        visited = [[False] * n for _ in range(m)]
        total_path = []
        total_dirs = []
        hash_map = defaultdict(list)
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    cur_path, dirs = dfs(x, y, [], ['null'])
                    if cur_path:
                        total_path.append(cur_path)
                        hash_map[str(dirs)].append(len(path))

        print(total_path)
        print(total_dirs)
        print(hash_map)
        return len(hash_map.keys())

# leetcode submit region end(Prohibit modification and deletion)
