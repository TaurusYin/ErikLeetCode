# Given an m x n integers matrix, return the length of the longest increasing 
# path in matrix. 
# 
#  From each cell, you can either move in four directions: left, right, up, or 
# down. You may not move diagonally or move outside the boundary (i.e., wrap-
# around is not allowed). 
# 
#  
#  Example 1: 
#  
#  
# Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
# Output: 4
# Explanation: The longest increasing path is [1, 2, 6, 9].
#  
# 
#  Example 2: 
#  
#  
# Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
# Output: 4
# Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally 
# is not allowed.
#  
# 
#  Example 3: 
# 
#  
# Input: matrix = [[1]]
# Output: 1
#  
# 
#  
#  Constraints: 
# 
#  
#  m == matrix.length 
#  n == matrix[i].length 
#  1 <= m, n <= 200 
#  0 <= matrix[i][j] <= 2³¹ - 1 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 图 拓扑排序 记忆化搜索 数组 动态规划 矩阵 👍 749 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def _longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        """
        dfs 构建邻接表
        :param matrix:
        :return:
        """

        def dfs(x, y):
            if dp[x][y] != 0:
                return dp[x][y]

            longest_path = 0
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for delta_x, delta_y in directions:
                new_x = x + delta_x
                new_y = y + delta_y
                if 0 <= new_x < m and 0 <= new_y < n and matrix[new_x][new_y] > matrix[x][y]:
                    longest_path = max(longest_path, dfs(new_x, new_y))

            dp[x][y] = 1 + longest_path
            return dp[x][y]

        m = len(matrix)
        n = len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        max_length = 0

        for i in range(m):
            for j in range(n):
                max_length = max(max_length, dfs(i, j))

        return max_length

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        outdegree = [[0] * n for _ in range(m)]
        longest_path = [[1] * n for _ in range(m)]

        # Compute outdegrees for each cell
        for i in range(m):
            for j in range(n):
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[i][j]:
                        outdegree[i][j] += 1

        # Topological sorting using BFS
        queue = deque([(i, j) for i in range(m) for j in range(n) if outdegree[i][j] == 0])
        max_length = 0

        while queue:
            x, y = queue.popleft()
            max_length = max(max_length, longest_path[x][y])

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] < matrix[x][y]:
                    outdegree[nx][ny] -= 1
                    if outdegree[nx][ny] == 0:
                        queue.append((nx, ny))
                    longest_path[nx][ny] = max(longest_path[nx][ny], longest_path[x][y] + 1)

        return max_length


# leetcode submit region end(Prohibit modification and deletion)
