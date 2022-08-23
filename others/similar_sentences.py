import collections
from typing import List


class UnionFind:
    def __init__(self, n):
        self.root = [i for i in range(n)]
        self.size = [1] * n
        self.part = n

    def find(self, x):
        if x != self.root[x]:
            # 在查询的时候合并到顺带直接根节点
            root_x = self.find(self.root[x])
            self.root[x] = root_x
            return root_x
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return True
        if root_x <= root_y:
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


class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        if len(words1) != len(words2): return False
        similar_words = {}
        for w1, w2 in pairs:
            if not w1 in similar_words: similar_words[w1] = set()
            if not w2 in similar_words: similar_words[w2] = set()
            similar_words[w1].add(w2)
            similar_words[w2].add(w1)
        for w1, w2 in zip(words1, words2):
            if w1 == w2: continue
            if w1 not in similar_words: return False
            if w2 not in similar_words[w1]: return False
        return True

    def _areSentencesSimilarTwo(self, sentence1, sentence2, similarPairs):
        if len(sentence1) != len(sentence2): return False
        graph = collections.defaultdict(list)
        for w1, w2 in similarPairs:
            graph[w1].append(w2)
            graph[w2].append(w1)

        for w1, w2 in zip(sentence1, sentence2):
            stack, seen = [w1], {w1}
            while stack:
                word = stack.pop()
                if word == w2: break
                for nei in graph[word]:
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
            else:
                return False
        return True

    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        if len(sentence1) != len(sentence2):
            return False
        words = set(sentence1).union(set(sentence2))
        for word1, word2 in similarPairs:
            words.add(word1)
            words.add(word2)
        words = list(words)
        dct = {word: i for i, word in enumerate(words)}

        uf = UnionFind(len(words))
        for word1, word2 in similarPairs:
            uf.union(dct[word1], dct[word2])
        m = len(sentence1)
        for i in range(m):
            if not uf.is_connected(dct[sentence1[i]], dct[sentence2[i]]):
                return False
        return True

    def numSimilarGroups(self, strs: List[str]) -> int:
        # https://leetcode.cn/problems/similar-string-groups/solution/xiang-si-zi-fu-chuan-zu-by-leetcode-solu-8jt9/
        def check():
            cnt = 0
            for k in range(m):
                if strs[i][k] != strs[j][k]:
                    cnt += 1
                if cnt > 2:
                    return False
            return cnt in [0, 2]

        n = len(strs)
        m = len(strs[0])
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if check():
                    uf.union(i, j)
        return uf.part


if __name__ == '__main__':
    words1 = ['great', 'acting', 'skills']
    words2 = ['fine', 'drama', 'talent']
    pairs = [["great", "fine"], ["acting", "drama"], ["skills", "talent"]]
    words1 = ["I", "love", "leetcode"]
    words2 = ["I", "love", "onepiece"]
    pairs = [["manga", "onepiece"], ["platform", "anime"], ["leetcode", "platform"], ["anime", "manga"]]
    # pairs = [["manga","hunterXhunter"],["platform","anime"],["leetcode","platform"],["anime","manga"]]

    s = Solution()
    s.areSentencesSimilar(words1=words1, words2=words2, pairs=pairs)
    s.areSentencesSimilarTwo(sentence1=words1, sentence2=words2, similarPairs=pairs)
