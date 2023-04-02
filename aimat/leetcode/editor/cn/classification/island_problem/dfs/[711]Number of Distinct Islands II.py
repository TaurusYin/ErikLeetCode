# You are given an m x n binary matrix grid. An island is a group of 1's (
# representing land) connected 4-directionally (horizontal or vertical.) You may assume 
# all four edges of the grid are surrounded by water. 
# 
#  An island is considered to be the same as another if they have the same 
# shape, or have the same shape after rotation (90, 180, or 270 degrees only) or 
# reflection (left/right direction or up/down direction). 
# 
#  Return the number of distinct islands. 
# 
#  
#  Example 1: 
#  
#  
# Input: grid = [[1,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1]]
# Output: 1
# Explanation: The two islands are considered the same because if we make a 180 
# degrees clockwise rotation on the first island, then two islands will have the 
# same shapes.
#  
# 
#  Example 2: 
#  
#  
# Input: grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
# Output: 1
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
#  Related Topics 深度优先搜索 广度优先搜索 并查集 哈希表 哈希函数 👍 54 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
import math
from collections import defaultdict
from typing import List


class Solution:
    # 欧式距离编码需要注意编码方式，小数点保留位数
    def numDistinctIslands2(self, grid: List[List[int]]) -> int:
        def calculate_distance(x1, y1, x2, y2):
            delta_x = abs(x1 - x2)
            delta_y = abs(y1 - y2)
            return math.sqrt(delta_x ** 2 + delta_y ** 2)

        def calculate_hash(arr):
            if not arr:
                return
            if len(arr) == 1:
                hash_map[-1].append(1)
                return
            if len(arr) == 2:
                hash_map[-2].append(2)
                return

            sum_val = 0
            n = len(arr)
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = arr[i]
                    x2, y2 = arr[j]
                    dist = calculate_distance(x1, y1, x2, y2)
                    sum_val += dist
            hash_map[round(sum_val, 6)].append((len(arr)))

        def dfs(x, y, path=[]):
            if visited[x][y]:
                return path
            path.append((x, y))
            visited[x][y] = True
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
        hash_map = defaultdict(list)
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    cur_path = dfs(x, y, [])
                    if cur_path:
                        calculate_hash(cur_path)
                        total_path.append(cur_path)

        print(total_path)
        print(hash_map)
        return len(hash_map.keys())


grid = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]
grid = [
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]]

Solution().numDistinctIslands2(grid=grid)
# leetcode submit region end(Prohibit modification and deletion)
