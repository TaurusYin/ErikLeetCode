def findIslandPath(self, grid: List[List[int]]) -> int:
    def dfs(x, y, path=[]):
        if visited[x][y]:
            return path
        path.append([x, y])
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
    total_path = []
    visited = [[False] * n for _ in range(m)]
    for x in range(1, m - 1):
        for y in range(1, n - 1):
            if grid[x][y] == 0:
                # dfs dfs(x, y, []) path 必须初始化
                cur_path = dfs(x, y, [])
                flag = False
                for xx, yy in cur_path:
                    if grid[xx][yy] == 0 and (xx == 0 or yy == 0 or xx == m - 1 or yy == n - 1):
                        flag = True
                        break
                if not flag and cur_path:
                    total_path.append(cur_path)
    return len(total_path)
