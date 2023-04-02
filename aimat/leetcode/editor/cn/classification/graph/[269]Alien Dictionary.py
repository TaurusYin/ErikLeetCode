# There is a new alien language that uses the English alphabet. However, the 
# order among the letters is unknown to you. 
# 
#  You are given a list of strings words from the alien language's dictionary, 
# where the strings in words are sorted lexicographically by the rules of this new 
# language. 
# 
#  Return a string of the unique letters in the new alien language sorted in 
# lexicographically increasing order by the new language's rules. If there is no 
# solution, return "". If there are multiple solutions, return any of them. 
# 
#  
#  Example 1: 
# 
#  
# Input: words = ["wrt","wrf","er","ett","rftt"]
# Output: "wertf"
#  
# 
#  Example 2: 
# 
#  
# Input: words = ["z","x"]
# Output: "zx"
#  
# 
#  Example 3: 
# 
#  
# Input: words = ["z","x","z"]
# Output: ""
# Explanation: The order is invalid, so return "".
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= words.length <= 100 
#  1 <= words[i].length <= 100 
#  words[i] consists of only lowercase English letters. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 图 拓扑排序 数组 字符串 👍 265 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from collections import deque
from collections import defaultdict
from typing import List


class Solution:
    """
    1. ["z", "z"] -> "z", ["abc", "ab"] -> "" 无order 情况需要考虑，加标识位
       if not found_order and len(word_a) > len(word_b):
    2. # 添加所有唯一字母到图中
        for word in words:
            for letter in word:
                graph[letter] = set()
    3. 重复字母用defaultdict(set)
    """

    def alienOrder_dfs(self, words: List[str]) -> str:
        print(words)
        graph = defaultdict(set)
        # 添加所有唯一字母到图中
        for word in words:
            for letter in word:
                graph[letter] = set()

        for i, word_a in enumerate(words):
            for j, word_b in enumerate(words):
                if j > i:
                    queue_a = deque(list(word_a))
                    queue_b = deque(list(word_b))
                    print(f"{word_a}, {word_b}")
                    # ["z", "z"], ["abc", "ab"] 无order 情况需要考虑，加标识位
                    found_order = False
                    while len(queue_a) != 0 and len(queue_b) != 0:
                        a = queue_a.popleft()
                        b = queue_b.popleft()
                        if a == b:
                            continue
                        else:
                            graph[a].add(b)
                            found_order = True
                            break
                    # 如果没有发现顺序关系且 word_a 长度大于 word_b，则返回空字符串
                    if not found_order and len(word_a) > len(word_b):
                        return ""

        print(graph)
        result = self.topo_sort_dfs(graph)
        print(result)
        return "".join(list(result))

    def topo_sort_dfs(self, graph):
        def dfs(node):
            if node in visiting:
                # 发现一个访问中的节点，说明存在环
                return False
            if node in visited:
                # 已访问节点，直接返回
                return True
            visiting.add(node)
            if node in graph:
                for neighbor in graph[node]:
                    if not dfs(neighbor):
                        return False
            visiting.remove(node)
            visited.add(node)
            post_order.append(node)
            return True

        visited = set()
        visiting = set()
        post_order = []
        for node in graph:
            if node not in visited:
                if not dfs(node):
                    # 发现环，返回空列表
                    return []

        # 将所有节点按照后序遍历序号从大到小排序
        result = list(reversed(post_order))
        print(f"post_order: {post_order}")
        return result

    def alienOrder(self, words: List[str]) -> str:
        graph = defaultdict(set)
        indegree = defaultdict(int)

        # 添加所有唯一字母到图中
        for word in words:
            for letter in word:
                graph[letter] = set()
                indegree[letter] = 0

        # 构建图
        for i in range(len(words) - 1):
            word_a = words[i]
            word_b = words[i + 1]
            found_order = False
            for a, b in zip(word_a, word_b):
                if a != b:
                    if b not in graph[a]:
                        graph[a].add(b)
                        indegree[b] += 1
                    found_order = True
                    break

            # 如果没有发现顺序关系且 word_a 长度大于 word_b，则返回空字符串
            if not found_order and len(word_a) > len(word_b):
                return ""

        # 拓扑排序
        queue = deque([node for node in graph if indegree[node] == 0])
        topo_order = []

        while queue:
            current = queue.popleft()
            topo_order.append(current)

            for neighbor in graph[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果排序后的字母数量与图中字母数量不匹配，说明存在环，返回空字符串
        if len(topo_order) != len(graph):
            return ""

        return "".join(topo_order)


Solution().alienOrder(words=["z", "z"])
# leetcode submit region end(Prohibit modification and deletion)
