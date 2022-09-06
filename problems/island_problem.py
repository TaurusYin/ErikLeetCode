
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
