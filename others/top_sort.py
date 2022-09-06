from typing import List
# from itertools import pairwise
import collections


# https://leetcode.cn/problems/alien-dictionary/solution/huo-xing-ci-dian-by-leetcode-solution-nr0l/

def alienOrder(words: List[str]) -> str:
    edge = collections.defaultdict(list)
    degree = collections.defaultdict(int)
    n = len(words)
    string = set()
    for word in words:
        for w in word:
            string.add(w)

    for i in range(n - 1):
        for j, w in enumerate(words[i]):
            if j >= len(words[i + 1]):
                return ""
            if j < len(words[i + 1]) and w != words[i + 1][j]:
                edge[w].append(words[i + 1][j])
                degree[words[i + 1][j]] += 1
                break

    ans = []
    stack = [w for w in string if not degree[w]]
    while stack:
        ans.extend(stack)
        nex = []
        for w in stack:
            for v in edge[w]:
                degree[v] -= 1
                if not degree[v]:
                    nex.append(v)
        stack = nex
    return "".join(ans) if len(ans) == len(string) else ""


def _alienOrder(words: List[str]) -> str:
    # 1 构造有向图
    from collections import defaultdict
    edges = defaultdict(list)
    indegree = {c: 0 for c in words[0]}
    for word1, word2 in pairwise(words):
        # 自动获取连续的重叠对: [a,b,c] -> a,b ; b,c
        for c in word2:
            # 入度词典中, 字母c如果不存在,就设置为0
            indegree.setdefault(c, 0)

        for c1, c2 in zip(word1, word2):
            # 返回一个元组列表,每个元组由word1和word2中的对应字母组成
            # 以较短的可迭代序列为zhun
            if c1 != c2:
                edges[c1].append(c2)
                indegree[c2] += 1
                break

        else:
            # 如果比较字符串的循环自然结束
            if len(word1) > len(word2):
                # 前一个单词和后一个单词的前缀一致,且前面的单词长度更长
                # 不符合题意返回""
                return ""

    # 2 寻找拓扑排序
    from collections import deque
    q = deque([x for x in indegree if indegree[x] == 0])
    topology = []
    while q:
        c = q.popleft()
        topology.append(c)
        for neighbour in edges[c]:
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                q.append(neighbour)

    # 3 判断图中是否有环: 如果拓扑排序的长度和图中顶点个数不同,则有环
    return "" if len(topology) != len(indegree) else "".join(topology)

"""
给你一个整数 n ，按字典序返回范围 [1, n] 内所有整数。字典序
https://leetcode.cn/problems/lexicographical-numbers/solution/zi-dian-xu-pai-shu-by-leetcode-solution-98mz/
你必须设计一个时间复杂度为 O(n) 且使用 O(1) 额外空间的算法。
"""
def lexicalOrder(self, n: int) -> List[int]:
    ans = [0] * n
    num = 1
    for i in range(n):
        ans[i] = num
        if num * 10 <= n:
            num *= 10
        else:
            while num % 10 == 9 or num + 1 > n:
                num //= 10
            num += 1
    return ans


if __name__ == '__main__':
    alienOrder(words=["wrt", "wrf", "er", "ett", "rftt"])
