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
#  Return the ordering of courses you should take to finish all courses. If 
# there are many valid answers, return any of them. If it is impossible to finish all 
# courses, return an empty array. 
# 
#  
#  Example 1: 
# 
#  
# Input: numCourses = 2, prerequisites = [[1,0]]
# Output: [0,1]
# Explanation: There are a total of 2 courses to take. To take course 1 you 
# should have finished course 0. So the correct course order is [0,1].
#  
# 
#  Example 2: 
# 
#  
# Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
# Output: [0,2,1,3]
# Explanation: There are a total of 4 courses to take. To take course 3 you 
# should have finished both courses 1 and 2. Both courses 1 and 2 should be taken 
# after you finished course 0.
# So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3
# ].
#  
# 
#  Example 3: 
# 
#  
# Input: numCourses = 1, prerequisites = []
# Output: [0]
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= numCourses <= 2000 
#  0 <= prerequisites.length <= numCourses * (numCourses - 1) 
#  prerequisites[i].length == 2 
#  0 <= ai, bi < numCourses 
#  ai != bi 
#  All the pairs [ai, bi] are distinct. 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 图 拓扑排序 👍 755 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from collections import defaultdict
from typing import List

class Solution:
    """
    把没有依赖关系的课程也要加到graph里面，是这道题和模版拓扑排序的区别
    """
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        def dfs(node):
            if node in visiting:
                # 发现一个访问中的节点，说明存在环
                return False
            if node in visited:
                # 已访问节点，直接返回
                return True
            visiting.add(node)
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            visiting.remove(node)
            visited.add(node)
            post_order.append(node)
            return True

        graph = defaultdict(list)
        for a, b in prerequisites:
            graph[b].append(a)

        for node in range(numCourses):
            if node not in graph:
                graph[node] = []

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
        return result

    def findOrder_indegree(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        if not prerequisites:
            return list(reversed(range(numCourses)))
        # 统计每个节点的入度
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        for a, b in prerequisites:
            graph[b].append(a)

        # 在这个题目中，我们需要考虑的是课程编号从0到numCourses - 1，因此我们需要确保处理了所有的课程。
        # 把有些课程没有任何先修课程的课程加进去，
        for node in range(numCourses):
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        # 将所有入度为0的节点加入队列
        queue = deque([node for node in graph if in_degree[node] == 0])
        # 从队列中依次取出节点，并更新其邻居节点的入度
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果图中存在环，返回空列表
        if len(result) != len(graph):
            return []
        return result


numCourses = 2;
prerequisites = [[0, 1], [1, 0]]
Solution().findOrder(numCourses=numCourses, prerequisites=prerequisites)
# leetcode submit region end(Prohibit modification and deletion)
