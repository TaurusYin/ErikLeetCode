from typing import List


class Solution:
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

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        for i in range(len(edges)):
            if self.same(edges[i][0], edges[i][1]):
                return edges[i]
            else:
                self.join(edges[i][0], edges[i][1])
        return []
