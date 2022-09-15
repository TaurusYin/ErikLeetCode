"""
https://leetcode.cn/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/number-of-islands
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def numIslands(self, grid: List[List[str]]) -> int:
    if not grid:
        return 0
    row = len(grid)
    col = len(grid[0])
    landNum = 0

    def Dfs(grid, i, j):
        grid[i][j] = 0  # 搜索过的地方置0
        for x, y in [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]:  # 搜索前后左右
            if 0 <= x < row and 0 <= y < col and grid[x][y] == '1':
                Dfs(grid, x, y)
        return

    for i in range(row):
        for j in range(col):
            if grid[i][j] == '1':
                landNum += 1
                Dfs(grid, i, j)
    return landNum


"""
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。
https://leetcode.cn/problems/minimum-path-sum/
"""


def minPathSum(self, grid: List[List[int]]) -> int:
    if not grid or not grid[0]:
        return 0

    rows, columns = len(grid), len(grid[0])
    dp = [[0] * columns for _ in range(rows)]
    dp[0][0] = grid[0][0]
    for i in range(1, rows):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, columns):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, rows):
        for j in range(1, columns):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

    return dp[rows - 1][columns - 1]


"""
https://leetcode.cn/problems/max-area-of-island/solution/695-dao-yu-de-zui-da-mian-ji-dfs-bfsshua-oe8m/
遍历二维数组，对于每块土地，去其前后左右找相邻土地，再去前后左右的土地找其前后左右的土地，直到周围没有土地
对于每一块已找过的土地，为避免重复计算，将其置为0
其实就是遍历了所有的岛屿，然后取这些岛屿的最大面积res = max(res, dfs(i, j))
O(m*n)
"""


def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j]:
            grid[i][j] = 0
            return 1 + dfs(i - 1, j) + dfs(i + 1, j) + dfs(i, j - 1) + dfs(i, j + 1)
        return 0

    res = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                res = max(res, dfs(i, j))
    return res


"""
遍历二维数组，对于每一块土地，也是去前后左右找相邻土地，只不过把找到的土地放到队列中
新土地加入队列的同时，岛屿面积area + 1，grid[nx][ny]置0
"""


def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])

    def bfs(i, j):
        queue = [(i, j)]
        grid[i][j] = 0
        area = 1
        while queue:
            x, y = queue.pop(0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                    grid[nx][ny] = 0
                    area += 1
                    queue.append((nx, ny))
        return area

    res = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j]:
                res = max(res, bfs(i, j))
    return res
