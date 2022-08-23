import collections
# https://leetcode.cn/problems/word-ladder/solution/python3-bfshe-shuang-xiang-bfsshi-xian-dan-ci-jie-/
from typing import List


def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    st = set(wordList)
    if endWord not in st:
        return 0
    m = len(beginWord)

    queue = collections.deque()
    queue.append((beginWord, 1))

    visited = set()
    visited.add(beginWord)

    while queue:
        cur, step = queue.popleft()
        if cur == endWord:
            return step

        for i in range(m):
            for j in range(26):
                tmp = cur[:i] + chr(97 + j) + cur[i + 1:]
                if tmp not in visited and tmp in st:
                    queue.append((tmp, step + 1))
                    visited.add(tmp)

    return 0


def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    st = set(wordList)
    if endWord not in st:
        return 0
    m = len(beginWord)
    lvisited = set()
    rvisited = set()
    lqueue = collections.deque()
    rqueue = collections.deque()

    lqueue.append(beginWord)
    rqueue.append(endWord)

    lvisited.add(beginWord)
    rvisited.add(endWord)
    step = 0

    while lqueue and rqueue:
        if len(lqueue) > len(rqueue):
            lqueue, rqueue = rqueue, lqueue
            lvisited, rvisited = rvisited, lvisited
        step += 1
        for k in range(len(lqueue)):
            cur = lqueue.popleft()
            if cur in rvisited:
                return step

            for i in range(m):
                for j in range(26):
                    tmp = cur[:i] + chr(97 + j) + cur[i + 1:]
                    if tmp not in lvisited and tmp in st:
                        lqueue.append(tmp)
                        lvisited.add(tmp)

    return 0
