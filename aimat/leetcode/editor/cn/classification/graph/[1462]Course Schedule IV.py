# There are a total of numCourses courses you have to take, labeled from 0 to 
# numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai,
#  bi] indicates that you must take course ai first if you want to take course bi.
#  
# 
#  
#  For example, the pair [0, 1] indicates that you have to take course 0 before 
# you can take course 1. 
#  
# 
#  Prerequisites can also be indirect. If course a is a prerequisite of course 
# b, and course b is a prerequisite of course c, then course a is a prerequisite 
# of course c. 
# 
#  You are also given an array queries where queries[j] = [uj, vj]. For the jᵗʰ 
# query, you should answer whether course uj is a prerequisite of course vj or 
# not. 
# 
#  Return a boolean array answer, where answer[j] is the answer to the jᵗʰ 
# query. 
# 
#  
#  Example 1: 
#  
#  
# Input: numCourses = 2, prerequisites = [[1,0]], queries = [[0,1],[1,0]]
# Output: [false,true]
# Explanation: The pair [1, 0] indicates that you have to take course 1 before 
# you can take course 0.
# Course 0 is not a prerequisite of course 1, but the opposite is true.
#  
# 
#  Example 2: 
# 
#  
# Input: numCourses = 2, prerequisites = [], queries = [[1,0],[0,1]]
# Output: [false,false]
# Explanation: There are no prerequisites, and each course is independent.
#  
# 
#  Example 3: 
#  
#  
# Input: numCourses = 3, prerequisites = [[1,2],[1,0],[2,0]], queries = [[1,0],[
# 1,2]]
# Output: [true,true]
#  
# 
#  
#  Constraints: 
# 
#  
#  2 <= numCourses <= 100 
#  0 <= prerequisites.length <= (numCourses * (numCourses - 1) / 2) 
#  prerequisites[i].length == 2 
#  0 <= ai, bi <= n - 1 
#  ai != bi 
#  All the pairs [ai, bi] are unique. 
#  The prerequisites graph has no cycles. 
#  1 <= queries.length <= 10⁴ 
#  0 <= ui, vi <= n - 1 
#  ui != vi 
#  
# 
#  Related Topics 深度优先搜索 广度优先搜索 图 拓扑排序 👍 104 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
from typing import List
from collections import defaultdict


class Solution:
    """
    首先，根据给定的先修关系（prerequisites）列表构建一个有向图，可以使用邻接表（如defaultdict(list)）来表示这个图。
    对于每个查询（queries），需要检查课程uj是否是课程vj的先修课程。这里可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来遍历图。从课程uj开始遍历，如果在遍历过程中找到课程vj，那么课程uj就是课程vj的先修课程，返回True。如果遍历完成后没有找到课程vj，那么课程uj不是课程vj的先修课程，返回False。
    无需考虑环的情况，注意每次查询时候初始化visit数组
    将所有查询的结果组合成一个布尔数组返回。
    """
    def __init__(self):
        self.flag = False

    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List:
        def dfs(src, dst):
            if src in visited:
                return
            visited.add(src)
            if src in graph:
                for node in graph[src]:
                    if node == dst:
                        self.flag = True
                        return
                    dfs(node, dst)
            return

        graph = defaultdict(list)
        for a, b in prerequisites:
            graph[a].append(b)
        for node in range(numCourses):
            if node not in graph:
                graph[node] = []
        result = []

        for src, dst in queries:
            visited = set()
            self.flag = False
            print(f"src:{src}")
            print(f"dst:{dst}")
            dfs(src, dst)
            result.append(self.flag)
        return result
# leetcode submit region end(Prohibit modification and deletion)
