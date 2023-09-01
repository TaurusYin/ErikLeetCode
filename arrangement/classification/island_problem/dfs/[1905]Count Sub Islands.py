# You are given two m x n binary matrices grid1 and grid2 containing only 0's (
# representing water) and 1's (representing land). An island is a group of 1's 
# connected 4-directionally (horizontal or vertical). Any cells outside of the grid 
# are considered water cells. 
# 
#  An island in grid2 is considered a sub-island if there is an island in grid1 
# that contains all the cells that make up this island in grid2. 
# 
#  Return the number of islands in grid2 that are considered sub-islands. 
# 
#  
#  Example 1: 
#  
#  
# Input: grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]], 
# grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]
# Output: 3
# Explanation: In the picture above, the grid on the left is grid1 and the grid 
# on the right is grid2.
# The 1s colored red in grid2 are those considered to be part of a sub-island. 
# There are three sub-islands.
#  
# 
#  Example 2: 
#  
#  
# Input: grid1 = [[1,0,1,0,1],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[1,0,1,0,1]], 
# grid2 = [[0,0,0,0,0],[1,1,1,1,1],[0,1,0,1,0],[0,1,0,1,0],[1,0,0,0,1]]
# Output: 2 
# Explanation: In the picture above, the grid on the left is grid1 and the grid 
# on the right is grid2.
# The 1s colored red in grid2 are those considered to be part of a sub-island. 
# There are two sub-islands.
#  
# 
#  
#  Constraints: 
# 
#  
#  m == grid1.length == grid2.length 
#  n == grid1[i].length == grid2[i].length 
#  1 <= m, n <= 500 
#  grid1[i][j] and grid2[i][j] are either 0 or 1. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 并查集 数组 矩阵 👍 93 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def __init__(self):
        self.status = True

    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        def dfs(x, y, path=[]):
            if visited[x][y]:
                return path
            path.append([x, y])
            if grid2[x][y] == 1 and grid1[x][y] == 0:
                self.status = False
                print(f"({x},{y}) status: {self.status}")
                return []
            visited[x][y] = True
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if 0 <= new_x < m and 0 <= new_y < n and grid2[new_x][new_y] == 1:
                    dfs(new_x, new_y, path)
            return path

        m = len(grid2)
        n = len(grid2[0])
        visited = [[False] * n for _ in range(m)]
        total_path = []
        for i in range(m):
            for j in range(n):
                self.status = True
                if grid1[i][j] == 1 and grid2[i][j] == 1:
                    cur_path = dfs(i, j, [])
                    if cur_path and self.status:
                        print(f"({i},{j}), cur_path:{cur_path}")
                        total_path.append(cur_path)


        print(total_path)
        return len(total_path)

# leetcode submit region end(Prohibit modification and deletion)
