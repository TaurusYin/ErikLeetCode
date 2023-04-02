# You are given an integer n, which indicates that there are n courses labeled 
# from 1 to n. You are also given an array relations where relations[i] = [
# prevCoursei, nextCoursei], representing a prerequisite relationship between course 
# prevCoursei and course nextCoursei: course prevCoursei has to be taken before course 
# nextCoursei. 
# 
#  In one semester, you can take any number of courses as long as you have 
# taken all the prerequisites in the previous semester for the courses you are taking. 
# 
# 
#  Return the minimum number of semesters needed to take all courses. If there 
# is no way to take all the courses, return -1. 
# 
#  
#  Example 1: 
#  
#  
# Input: n = 3, relations = [[1,3],[2,3]]
# Output: 2
# Explanation: The figure above represents the given graph.
# In the first semester, you can take courses 1 and 2.
# In the second semester, you can take course 3.
#  
# 
#  Example 2: 
#  
#  
# Input: n = 3, relations = [[1,2],[2,3],[3,1]]
# Output: -1
# Explanation: No course can be studied because they are prerequisites of each 
# other.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= n <= 5000 
#  1 <= relations.length <= 5000 
#  relations[i].length == 2 
#  1 <= prevCoursei, nextCoursei <= n 
#  prevCoursei != nextCoursei 
#  All the pairs [prevCoursei, nextCoursei] are unique. 
#  
# 
#  Related Topics 图 拓扑排序 👍 60 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import defaultdict


class Solution:
    def _minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        def dfs(node):
            if node in visiting:
                # 发现一个访问中的节点，说明存在环
                return -1
            if node in memo:
                # 已访问节点，直接返回
                return memo[node]
            visiting.add(node)
            max_depth = 0
            if node in graph:
                for neighbor in graph[node]:
                    depth = dfs(neighbor)
                    if depth == -1:
                        return -1
                    max_depth = max(max_depth, depth)
            visiting.remove(node)
            memo[node] = max_depth + 1
            return memo[node]

        graph = defaultdict(list)
        for item in relations:
            graph[item[0]].append(item[1])

        memo = {}
        visiting = set()
        max_path = 0
        for node in range(1, n + 1):
            path_len = dfs(node)
            if path_len == -1:
                return -1
            max_path = max(max_path, path_len)
        return max_path

    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        """
        这个实现首先计算每个节点的入度（即前置课程数量），然后使用 memo 存储已经计算过的节点的最大深度。队列 queue 存储入度为 0 的节点，这些节点可以在第一个学期学习。
        迭代过程中，我们从队列中取出节点，处理其所有邻居节点（后续课程），将邻居节点的入度减 1。如果邻居节点的入度变为 0，表示其前置课程已经完成，我们可以计算其深度（当前节点深度 + 1），将邻居节点添加到队列中。
        memo[neighbor] = memo[node] + 1
        :param n:
        :param relations:
        :return:
        """
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for prev, next in relations:
            graph[prev].append(next)
            in_degree[next] += 1

        start_nodes = [node for node in range(1, n + 1) if in_degree[node] == 0]
        # memo存储每个节点的最大深度
        memo = {node: 1 for node in start_nodes}

        queue = deque(start_nodes)

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    memo[neighbor] = memo[node] + 1
                    queue.append(neighbor)

        if len(memo) != n:
            return -1

        return max(memo.values())

    def __minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        graph = defaultdict(list)
        for item in relations:
            graph[item[0]].append(item[1])
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        # 将所有入度为0的节点加入队列
        queue = deque([node for node in range(1, n + 1) if in_degree[node] == 0])
        semester = 0
        taken_courses = 0
        while queue:
            next_queue = deque()
            while queue:
                node = queue.popleft()
                taken_courses += 1
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            queue = next_queue
            semester += 1

        # 如果图中存在环或无法完成所有课程，返回-1
        if taken_courses != n:
            return -1
        return semester


n = 3
relations = [[1, 3], [2, 3]]
Solution().minimumSemesters(n=n, relations=relations)
# leetcode submit region end(Prohibit modification and deletion)
