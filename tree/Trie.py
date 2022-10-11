import heapq


class Node(object):
    def __init__(self, val: int):
        self.val = val

    def __repr__(self):
        return f'Node value: {self.val}'

    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        return self.val == other.val

    def __gt__(self, other):
        return self.val > other.val


heap = [Node(2), Node(0), Node(1), Node(4), Node(2)]
heap = [2, 0, 1, 4, 2]
heapq.heapify(heap)
print(heap)  # output: [Node value: 0, Node value: 2, Node value: 1, Node value: 4, Node value: 2]

heapq.heappop(heap)
print(heap)  # output: [

"""
https://leetcode.cn/problems/implement-trie-prefix-tree/

"""


class Trie:
    def __init__(self):
        # 初始化字典树
        self.alpha_dict = {}
        # 字符串结束标记
        self.end_of_string = -1

    def insert(self, word: str) -> None:
        node = self.alpha_dict
        # 迭代建立字典树
        for s in word:
            if s not in node:
                node[s] = {}
            node = node[s]
        # 该字符串最后一位进行标记
        node[self.end_of_string] = True

    def search(self, word: str) -> bool:
        node = self.alpha_dict
        for s in word:
            if s not in node:
                return False
            node = node[s]
        # 当且仅当当前word的每一位都可以在字典树中找到且存在end_of_string
        return self.end_of_string in node

    def startsWith(self, prefix: str) -> bool:
        node = self.alpha_dict
        for s in prefix:
            if s not in node:
                return False
            node = node[s]
        # 只要当前prefix中每一位都可以在字典树中找到就可以
        return True
