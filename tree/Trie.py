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


class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {'end':True}

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['end'] = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for c in word:
            if c not in node or 'end' not in node:
                return False
            node = node[c]
        return 'end' in node


    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for c in prefix:
            if c not in node:
                return False
            node = node[c]
        return True


"""
https://leetcode.cn/problems/longest-word-in-dictionary/solution/ci-dian-zhong-zui-chang-de-dan-ci-by-lee-k5gj/
O(sum(Li单词长度))
O(sum(Li单词长度))
输入：words = ["w","wo","wor","worl", "world"]
输出："world"
解释： 单词"world"可由"w", "wo", "wor", 和 "worl"逐步添加一个字母组成。
"""
def longestWord(self, words: List[str]) -> str:
    t = Trie()
    for word in words:
        t.insert(word)
    longest = ""
    for word in words:
        if t.search(word) and (len(word) > len(longest) or len(word) == len(longest) and word < longest):
            longest = word
    return longest
