# There are a total of numCourses courses you have to take, labeled from 0 to 
# numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai,
#  bi] indicates that you must take course bi first if you want to take course ai.
#  
# 
#  
#  For example, the pair [0, 1], indicates that to take course 0 you have to 
# first take course 1. 
#  
# 
#  Return true if you can finish all courses. Otherwise, return false. 
# 
#  
#  Example 1: 
# 
#  
# Input: numCourses = 2, prerequisites = [[1,0]]
# Output: true
# Explanation: There are a total of 2 courses to take. 
# To take course 1 you should have finished course 0. So it is possible.
#  
# 
#  Example 2: 
# 
#  
# Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
# Output: false
# Explanation: There are a total of 2 courses to take. 
# To take course 1 you should have finished course 0, and to take course 0 you 
# should also have finished course 1. So it is impossible.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= numCourses <= 2000 
#  0 <= prerequisites.length <= 5000 
#  prerequisites[i].length == 2 
#  0 <= ai, bi < numCourses 
#  All the pairs prerequisites[i] are unique. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 图 拓扑排序 👍 1544 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import defaultdict


class Solution:
    def _canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(course, visited, graph):
            if visited[course] == -1:  # If the course is marked as temporary, it means there is a cycle
                return False
            if visited[course] == 1:  # If the course is marked as permanent, there is no need to visit it again
                return True

            visited[course] = -1  # Mark the course as temporary

            for neighbor in graph[course]:
                if not dfs(neighbor, visited, graph):
                    return False

            visited[course] = 1  # Mark the course as permanent
            return True

        # Create an adjacency list to represent the graph
        graph = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            graph[a].append(b)

        visited = [0] * numCourses  # 0 for not visited, -1 for temporary, 1 for permanent

        for course in range(numCourses):
            if visited[course] == 0:
                if not dfs(course, visited, graph):
                    return False

        return True

    def has_cycle(self, graph: defaultdict):
        def dfs(node, visited, visiting):
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if node in graph:
                for neighbor in graph[node]:
                    if dfs(neighbor, visited, visiting):
                        return True
            visiting.remove(node)
            visited.add(node)
            return False

        visited = set()
        visiting = set()
        for node in graph:
            if node not in visited:
                if dfs(node, visited, visiting):
                    return True
        return False

    def has_cycle_bfs(self, graph: defaultdict):
        in_degree = defaultdict(int)
        for node in graph.keys():
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        queue = deque([node for node in graph.keys() if in_degree[node] == 0])

        while queue:
            cur_node = queue.popleft()
            for neighbor in graph[cur_node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return any(in_degree[node] > 0 for node in graph.keys())

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def create_graph(numCourses, prerequisites):
            graph = defaultdict(list)
            for a, b in prerequisites:
                graph[b].append(a)
            return graph
        graph = create_graph(numCourses, prerequisites)
        return not self.has_cycle_bfs(graph)

numCourses = 2
prerequisites = [[1, 0]]
Solution().canFinish(numCourses=numCourses, prerequisites=prerequisites)

# leetcode submit region end(Prohibit modification and deletion)
