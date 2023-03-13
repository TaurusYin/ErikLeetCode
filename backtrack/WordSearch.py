import copy
from collections import defaultdict
from functools import reduce
from typing import List

words = ["oath", "pea", "eat", "rain"]
Tree = lambda: defaultdict(Tree)
tree = Tree()
for w in words: reduce(dict.__getitem__, w + "#", tree)
print()




class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def backtrack(pos, start_index):
            nonlocal ans
            if path == word_dq:
                ans = True
                return True
            x, y = pos[0], pos[1]
            if start_index == len(word):
                return
            res.append(path[:])
            if board[x][y] != word_dq[start_index]:
                return
            for px, py in positions:
                # print('coordinate:{},{}'.format(px, py))
                if 0 <= x + px <= n - 1 and 0 <= y + py <= m - 1 and visit[x + px][y + py] == False:
                    path.append(board[x + px][y + py])
                    print(x + px, y + py, board[x][y], path[:])
                    visit[x + px][y + py] = True
                    backtrack([x + px, y + py], start_index + 1)
                    path.pop()
                    visit[x + px][y + py] = False

        n = len(board)
        m = len(board[0])
        positions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        word_dq = list(word)
        ans = False
        o_visit = [[False] * m for i in range(n)]
        for i in range(n):
            for j in range(m):
                visit = copy.copy(o_visit)
                visit[i][j] = True
                path, res = [board[i][j]], []
                backtrack([i, j], 0)
                visit[i][j] = False
                if ans:
                    return True
        return False

    from typing import List

    class Solution:
        def exist(self, board: List[List[str]], word: str) -> bool:
            # 判断坐标是否合法
            def inArea(x, y):
                return x >= 0 and x < m and y >= 0 and y < n

            def backtrack(index, startx, starty):
                if index == len(word) - 1:
                    return board[startx][starty] == word[index]
                if board[startx][starty] == word[index]:
                    # 标记该坐标已被访问
                    used[startx][starty] = True
                    # 在该坐标的上右下左四个方向依次寻找
                    for k in range(4):
                        newx = startx + d[k][0]
                        newy = starty + d[k][1]
                        if inArea(newx, newy) and not used[newx][newy] and backtrack(index + 1, newx, newy):
                            return True
                    used[startx][starty] = False

            m = len(board)
            n = len(board[0])
            d = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            used = [[False] * n for i in range(m)]
            for i in range(m):
                for j in range(n):
                    if backtrack(0, i, j):
                        return True
            return False

        """
        https://leetcode.cn/problems/word-search-ii/solution/pythondai-ma-jie-jue-zi-dian-shu-by-simo-s9k7/
        输入：board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
        输出：["eat","oath"]
        """
        def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
            # 网格的大小是12*12=144，单词长度为10，单词数量为3*10^4，三个数字相乘为4.32*10^7
            # 看这个级别的数字感觉在超时与不超时之间徘徊，写了一个朴素的dfs，果真超了
            # 分析超时的用例，发现是有大量的前缀相同的words，于是自然的想到用字典树做字符串的合并
            # 结果是字典树+dfs不超时

            class node:
                def __init__(self, val):
                    self.val = val
                    self.childs = dict()
                    self.leaf = False
                    self.word = ''

            from collections import deque
            from copy import deepcopy as dp
            m, n = len(board), len(board[0])
            ops = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            ans = set()

            def dfs(x, y, vv, cur):  # 深度优先搜索
                if cur.leaf:
                    nonlocal ans
                    ans.add(cur.word)
                for xx, yy in ops:
                    xxx, yyy = x + xx, y + yy
                    if 0 <= xxx < m and 0 <= yyy < n:
                        if board[xxx][yyy] in cur.childs and vv[xxx][yyy] == 0:
                            vv[xxx][yyy] = 1
                            dfs(xxx, yyy, vv, cur.childs[board[xxx][yyy]])
                            vv[xxx][yyy] = 0
                return

            def build_tree(words):  # 构建字典树
                root = node(-1)
                for w in words:
                    cur = root
                    for i, ww in enumerate(w):
                        if ww not in cur.childs:
                            nex = node(ww)
                            cur.childs[ww] = nex
                        else:
                            nex = cur.childs[ww]
                        cur = nex
                    cur.leaf = True
                    cur.word = w
                return root

            root = build_tree(words)
            v = [[0 for _ in range(n)] for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    if board[i][j] in root.childs:
                        vis = dp(v)
                        vis[i][j] = 1
                        dfs(i, j, vis, root.childs[board[i][j]])
            return list(ans)


board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "ABCCED"
board = [["a", "a"]]
word = "aaa"
board = [["a", "b"]]
word = "ba"
# board = [["a","b"],["c","d"]]
# word = "acdb"
Solution().exist(board=board, word=word)
