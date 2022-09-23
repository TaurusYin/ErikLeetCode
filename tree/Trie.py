
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
heap = [2,0,1,4,2]
heapq.heapify(heap)
print(heap)  # output: [Node value: 0, Node value: 2, Node value: 1, Node value: 4, Node value: 2]

heapq.heappop(heap)
print(heap)  # output: [

"""
https://leetcode.cn/problems/implement-trie-prefix-tree/

"""
class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def searchPrefix(self, prefix: str) -> "Trie":
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def search(self, word: str) -> bool:
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str) -> bool:
        return self.searchPrefix(prefix) is not None


