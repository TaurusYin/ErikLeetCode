import collections
from typing import List

"""
Union-Find 算法的复杂度可以这样分析：构造函数初始化数据结构需要 O(N) 的时间和空间复杂度；连通两个节点 union、判断两个节点的连通性 connected、计算连通分量 count 所需的时间复杂度均为 O(1)。
到这里，相信你已经掌握了 Union-Find 算法的核心逻辑，总结一下我们优化算法的过程：
1、用 parent 数组记录每个节点的父节点，相当于指向父节点的指针，所以 parent 数组内实际存储着一个森林（若干棵多叉树）。
2、用 size 数组记录着每棵树的重量，目的是让 union 后树依然拥有平衡性，保证各个 API 时间复杂度为 O(logN)，而不会退化成链表影响操作效率。
3、在 find 函数中进行路径压缩，保证任意树的高度保持在常数，使得各个 API 时间复杂度为 O(1)。使用了路径压缩之后，可以不使用 size 数组的平衡优化。
下面我们看一些具体的并查集题目。
"""
class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n  # 联通分量个数

    # 实现隔代压缩 路径压缩的时间复杂度为O(logN)
    def find(self, x):
        if x != self.root[x]:
            origin = self.root[x]
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:  # 找到环
            return True
        # parent_id 小的为父节点， 默认 y是x的父节点
        if root_x <= root_y:  # 没找到环，更新两个节点的parent为其中最小值
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        return False

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_root_part(self):
        # 获取每个根节点对应的组
        part = collections.defaultdict(list)
        n = len(self.root)
        for i in range(n):
            part[self.find(i)].append(i)
        return part

    def get_root_size(self):
        # 获取每个根节点对应的组大小
        size = collections.defaultdict(int)
        n = len(self.root)
        for i in range(n):
            size[self.find(i)] = self.size[self.find(i)]
        return size

    """
    冗余连接
    https://leetcode.cn/problems/redundant-connection/submissions/
    输入: edges = [[1,2], [1,3], [2,3]]
    输出: [2,3]
    O(NlogN)
    其中 NN 是图中的节点个数。需要遍历图中的 NN 条边，对于每条边，需要对两个节点查找祖先，如果两个节点的祖先不同则需要进行合并，需要进行 22 次查找和最多 11 次合并。一共需要进行 2N2N 次查找和最多 NN 次合并，因此总时间复杂度是 O(2N \log N)=O(N \log N)O(2NlogN)=O(NlogN)。这里的并查集使用了路径压缩，但是没有使用按秩合并，最坏情况下的时间复杂度是 O(N \log N)O(NlogN)，平均情况下的时间复杂度依然是 O(N \alpha (N))O(Nα(N))，其中 \alphaα 为阿克曼函数的反函数，\alpha (N)α(N) 可以认为是一个很小的常数。
    """

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        for i in range(len(edges)):
            if self.is_connected(edges[i][0], edges[i][1]):
                return edges[i]
            else:
                self.union(edges[i][0], edges[i][1])
        return []

    """
    https://leetcode.cn/problems/satisfiability-of-equality-equations/
    给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
    只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
    O(n+Clog C) C=26
    """

    def equationsPossible(self, equations: List[str]) -> bool:
        uf = UnionFind(1000)
        false_list = []
        for u, x, y, v in equations:
            if x == '=' and y == '=':
                uf.union(ord(u), ord(v))
            if x == '!':
                false_list.append((u, x, y, v))
        for u, x, y, v in false_list:
            if uf.is_connected(ord(u), ord(v)) == True:
                return False
        return True

    """
    https://leetcode.cn/problems/number-of-provinces/solution/
    isConnected = [[1,1,0],[1,1,0],[0,0,1]]
    最大联通分量个数
    O(n2logn)
    """

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # print(isConnected)
        m = len(isConnected)
        n = len(isConnected[0])
        uf = UnionFind(m)
        for i in range(0, m):
            for j in range(0, n):
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        return uf.part
        '''
        count = 0
        for i in range(len(uf.root)):
            if uf.root[i] == i:
                count += 1
        return count
        '''

    """
    
    """

    """
    连接n个联通分量需要n-1跟绳子，求联通分量-1
    https://leetcode.cn/problems/number-of-operations-to-make-network-connected/solution/
    """

    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n - 1:
            return -1
        uf = UnionFind(n)
        count = 0
        for conn in connections:
            u, v = conn[0], conn[1]
            uf.union(u, v)
        return uf.part - 1
        '''
        print(uf.root)
        count = 0
        for i in range(len(uf.root)):
            if uf.root[i] == i:
                count += 1
        return count - 1
        '''

    """
    https://leetcode.cn/problems/couples-holding-hands/solution/by-be_a_better_coder-jacj/
    人和座位由一个整数数组 row 表示，其中 row[i] 是坐在第 i 个座位上的人的 ID。情侣们按顺序编号，第一对是 (0, 1)，第二对是 (2, 3)，以此类推，最后一对是 (2n-2, 2n-1)。
    输入: row = [0,2,1,3]
    输出: 1
    解释: 只需要交换row[1]和row[2]的位置即可。
    """

    def minSwapsCouples(self, row: List[int]) -> int:
        n = len(row)
        couple = int(n / 2)  # 座位
        uf = UnionFind(couple)
        print(uf.root)
        for i in range(couple):
            uf.union(row[2 * i] // 2, row[2 * i + 1] // 2)  # 把座位相连
        return couple - uf.part

    """
    https://leetcode.cn/problems/graph-valid-tree/submissions/
    以图判树，判断是否有冗余边，最大联通分量=1
    """
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        uf = UnionFind(n)
        for i in range(len(edges)):
            if uf.is_connected(edges[i][0], edges[i][1]):
                return False
            else:
                uf.union(edges[i][0], edges[i][1])
        return True if uf.part == 1 else False

    """
    https://leetcode.cn/problems/surrounded-regions/solution/dfs-bfs-bing-cha-ji-by-powcai/
    输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

    """

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        row = len(board)
        col = len(board[0])
        dummy = row * col
        uf = UnionFind(dummy + 1)

        for i in range(row):
            for j in range(col):
                if board[i][j] == "O":
                    if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                        uf.union(i * col + j, dummy)
                    else:
                        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            if board[i + x][j + y] == "O":
                                uf.union(i * col + j, (i + x) * col + (j + y))
        for i in range(row):
            for j in range(col):
                if uf.find(dummy) == uf.find(i * col + j):
                    board[i][j] = "O"
                else:
                    board[i][j] = "X"

    """
    在由 1 x 1 方格组成的 n x n 网格 grid 中，每个 1 x 1 方块由 '/'、'\' 或空格构成。这些字符会将方块划分为一些共边的区域。给定网格 grid 表示为一个字符串数组，返回 区域的数量 
    输入：grid = [" /","/ "]
    输出：2
    """

    def regionsBySlashes(self, grid: List[str]) -> int:
        size = len(grid)
        n = 4 * size * size
        uf = UnionFind(n)
        for i in range(size):
            for j in range(size):
                # 每个方格都拆分为 4 份，
                # 将二维网格转换为一维表格
                idx = 4 * (i * size + j)
                ch = grid[i][j]
                # 方块只能由 /、\ 或空格构成，针对三种的情况进行方块内合并
                # 当方格为 / 时，合并 0 和 3 区域，以及 1 和 2 区域
                if ch == '/':
                    uf.union(idx, idx + 3)
                    uf.union(idx + 1, idx + 2)
                # 当方格为 \ 时，合并 0 和 1 区域，以及 2 和 3 区域
                elif ch == '\\':
                    uf.union(idx, idx + 1)
                    uf.union(idx + 2, idx + 3)
                # 当方格为空格时，合并所有区域
                else:
                    uf.union(idx, idx + 1)
                    uf.union(idx + 1, idx + 2)
                    uf.union(idx + 2, idx + 3)

                # 方块间合并
                if i < size - 1:
                    bottom_idx = 4 * ((i + 1) * size + j)
                    uf.union(idx + 2, bottom_idx)
                if j < size - 1:
                    right_idx = 4 * (i * size + j + 1)
                    uf.union(idx + 1, right_idx + 3)
        return uf.part


'''
class UnionFind:
    def __init__(self):
        """
        初始化
        """
        self.n = 1005
        self.father = [i for i in range(self.n)]

    def find(self, u):
        """
        并查集里寻根的过程
        """
        if u == self.father[u]:
            return u
        self.father[u] = self.find(self.father[u])
        return self.father[u]

    def join(self, u, v):
        """
        将v->u 这条边加入并查集
        """
        u = self.find(u)
        v = self.find(v)
        if u == v: return
        self.father[v] = u
        pass

    def same(self, u, v):
        """
        判断 u 和 v是否找到同一个根，本题用不上
        """
        u = self.find(u)
        v = self.find(v)
        return u == v
'''
if __name__ == '__main__':
    uf = UnionFind()
    uf.findRedundantConnection(edges=[[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]])
