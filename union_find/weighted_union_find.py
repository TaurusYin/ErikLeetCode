import collections
from typing import List


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n  # 联通分量个数
        # weighted case
        self.weight = [1.0] * n

    # 实现隔代压缩
    def find(self, x):
        if x != self.root[x]:
            # 在查询的时候合并到顺带直接根节点
            root_x = self.find(self.root[x])
            self.root[x] = root_x

            # weighted case
            self.weight[x] *= self.weight[self.root[x]]
            return root_x
        return x

    def union(self, x, y, value=1):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:  # 找到环
            return True
        if root_x <= root_y:  # 没找到环，更新两个节点的parent为其中最小值
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        # 将非根节点的秩赋0
        self.size[root_x] = 0
        self.part -= 1
        # weighted case
        self.weight[root_x] = value * self.weight[y] / self.weight[x]
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

    def query(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return self.weight[x] / self.weight[y]
        else:
            return -1.0


"""
https://leetcode.cn/problems/evaluate-division/
输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
时间复杂度：O((N+Q)log⁡A)O((N + Q)\log A)O((N+Q)logA)，
    构建并查集 O(Nlog⁡A)O(N \log A)O(NlogA) ，这里 NNN 为输入方程 equations 的长度，每一次执行合并操作的时间复杂度是 O(log⁡A)O(\log A)O(logA)，这里 AAA 是 equations 里不同字符的个数；
    查询并查集 O(Qlog⁡A)O(Q \log A)O(QlogA)，这里 QQQ 为查询数组 queries 的长度，每一次查询时执行「路径压缩」的时间复杂度是 O(log⁡A)O(\log A)O(logA)。
空间复杂度：O(A)O(A)O(A)：创建字符与 id 的对应关系 hashMap 长度为 AAA，并查集底层使用的两个数组 parent 和 weight 存储每个变量的连通分量信息，parent 和 weight 的长度均为 AAA。

"""

#a/b=2 ---> a=2b --->
#           x=2y
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        id = 0
        d = dict()

        # string to int id mapping
        for equ in equations:
            x = equ[0]
            y = equ[1]

            if x not in d:
                d[x] = id
                id += 1

            if y not in d:
                d[y] = id
                id += 1

        uf = UnionFind(id)
        for i in range(len(equations)):
            x = equations[i][0]
            y = equations[i][1]

            uf.union(d[x], d[y], values[i])

        ans = []
        for i in range(len(queries)):
            query = queries[i]
            x = d.get(query[0], None)
            y = d.get(query[1], None)

            if x is None or y is None:
                ans.append(-1.0)
            else:
                ans.append(uf.query(x, y))

        return ans

    """
    https://leetcode.cn/problems/evaluate-division/solution/floyd-duo-yuan-by-s1ne-cuxx/
    """
    def calcEquation(self, equations, values, queries):
        g = collections.defaultdict(dict)
        for (a, b), v in zip(equations, values): g[a][b], g[b][a] = v, 1 / v

        for k in g:
            for i in g:
                for j in g:
                    if k in g[i] and j in g[k]:
                        g[i][j] = g[i][k] * g[k][j]

        return [g[a][b] if a in g and b in g[a] else -1.0 for a, b in queries]

    """
    转成加权图+floyd求任意两点路径
    """

    class Solution:
        def calcEquation(self, equations: List[List[str]], values: List[float],
                         queries: List[List[str]]) -> List[float]:
            # 方法2: 转成加权图+floyd求任意两点路径
            maps = {}
            v = set()
            for i in range(len(values)):
                # 记录有向边的权重
                x, y, multiple = equations[i][0], equations[i][1], values[i]
                maps[x, y] = multiple
                maps[y, x] = 1 / multiple
                v.add(x)
                v.add(y)
            v = list(v)
            # floyd算法求任意两点路径
            for k in range(len(v)):
                for i in range(len(v)):
                    for j in range(len(v)):
                        vi, vj, vk = v[i], v[j], v[k]
                        if (vi, vk) in maps and (vk, vj) in maps:
                            if (vi, vj) not in maps:
                                maps[vi, vj] = maps[vi, vk] * maps[vk, vj]
                            else:
                                maps[vi, vj] = min(maps[vi, vj],
                                                   maps[vi, vk] * maps[vk, vj])
            v = set(v)
            res = []
            for q in queries:
                if q[0] not in v or q[1] not in v or (q[0], q[1]) not in maps:
                    # 某个变量不存在, 或者两点之间不存在路径, 返回-1
                    res.append(-1.0)
                else:
                    res.append(maps[q[0], q[1]])
            return res

