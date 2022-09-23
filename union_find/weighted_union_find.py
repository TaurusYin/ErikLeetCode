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
