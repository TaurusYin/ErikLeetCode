import collections
from typing import List

row = 4
col = 3
matrix = [[0 for j in range(col)] for i in range(row)]
print()

"""
请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）
 graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph[i][j]存在一条有向边）。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/all-paths-from-source-to-target
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def allPathsSourceTarget(graph: List[List[int]]) -> List[List[int]]:
    res = []

    def traverse(graph, s, path):
        # 添加节点到路径path
        path.append(s)
        # 如果是节点n-1 把当前路径添加到最终结果res
        n = len(graph)
        if s == n - 1:
            # 注意此处python的list浅拷贝 如果直接添加path 最后执行完res为空
            # 此处使用切片 切片第一层是深拷贝
            res.append(path[:])
            # 由于题目明确说明是有向无环图 所以不会无限递归 可以不return 如果要return需要在此处将s从path中移除
        # 递归的遍历每个相邻节点
        for v in graph[s]:
            traverse(graph, v, path)
        # 从path将节点s移除 s此时在path的末尾
        path.pop()

    traverse(graph, 0, [])
    return res


def _allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    res = []

    def backtrack(graph, node, path, target):
        print('current_path: {} '.format(path[:]))
        # 结束条件 到达目标节点
        if node == target:
            res.append(path[:])
            print('final_path: {} '.format(path[:]))
            return
        # 选择列表是当前node的所有邻居节点
        for neighbor in graph[node]:
            # 做选择
            path.append(neighbor)
            # 递归
            backtrack(graph, neighbor, path, target)
            # 撤销选择
            x = path.pop()
            print('pop_elem: {} '.format(x))

    # 初始时path包括起点节点0
    backtrack(graph, 0, [0], len(graph) - 1)
    return res


graph = [[1, 2], [3], [3], []]


_allPathsSourceTarget(graph)


def __allPathsSourceTarget(graph: List[List[int]]) -> List[List[int]]:
    res = []
    # 使用队列完成BFS 将路径入队 初始路径是[0]
    q = collections.deque([[0]])
    while q:
        path = q.popleft()
        # 取path的最后一个元素 看是不是目标len(graph)-1
        if path[-1] == len(graph) - 1:
            # 此路径已经完成 加入到结果
            res.append(path[:])
        else:
            # 未完成 遍历path最后一个元素的邻居节点
            for neighbor in graph[path[-1]]:
                # 将原path拼接当前邻居得到的新路径入队
                q.append(path + [neighbor])
    return res


graph = [[1, 2], [3], [3], []]
__allPathsSourceTarget(graph)

# https://leetcode.cn/problems/course-schedule/
"""
O(m+n)
判断图是否有环
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
示例 2：

输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/course-schedule
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


class Solution:
    def __init__(self):
        self.hascycle = False

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 先根据依赖关系建图
        graph = [[] for _ in range(numCourses)]
        for edge in prerequisites:
            fr = edge[1]
            to = edge[0]
            # 建立课程从fr到to的依赖
            graph[fr].append(to)

        # 记录当前 traverse 经过的路径
        onpath = [False for _ in range(numCourses)]
        # 标记节点被遍历的情况
        visited = [False for _ in range(numCourses)]
        self.hascycle = False

        # DFS遍历函数
        def traverse(graph, v):
            # 如果当前traverse经过的路径已经遍历过了 说明有环
            if onpath[v]:
                self.hascycle = True
            if visited[v] or self.hascycle:
                # 如果节点已经被遍历或找到了环 就不继续遍历了
                return
            # 标记节点v为已访问
            visited[v] = True
            # 前序位置 标记进入节点v的遍历
            onpath[v] = True
            for neighbor in graph[v]:
                traverse(graph, neighbor)
            # 后序位置 离开节点v的遍历
            onpath[v] = False

        # 由于图可能不是全联通的 所以需要对每个节点都调用一次遍历函数
        for i in range(numCourses):
            # 遍历每个节点
            traverse(graph, i)

        # 只要没有循环依赖（环）就说明可以完成所有课程，否则不能
        return not self.hascycle


Solution().canFinish(numCourses=2, prerequisites=[[1, 0], [0, 1]])


def _canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    indegrees = [0 for _ in range(numCourses)]
    adjacency = [[] for _ in range(numCourses)]
    queue = collections.deque()
    # Get the indegree and adjacency of every course.
    for cur, pre in prerequisites:
        indegrees[cur] += 1
        adjacency[pre].append(cur)
    # Get all the courses with the indegree of 0.
    for i in range(len(indegrees)):
        if not indegrees[i]: queue.append(i)
    # BFS TopSort.
    while queue:
        pre = queue.popleft()
        numCourses -= 1
        for cur in adjacency[pre]:
            indegrees[cur] -= 1
            if not indegrees[cur]: queue.append(cur)
    return not numCourses


"""
现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。

 

示例 1：

输入：numCourses = 2, prerequisites = [[1,0]]
输出：[0,1]
解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
示例 2：

输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
输出：[0,2,1,3]
解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/course-schedule-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # 存储有向图
    edges = collections.defaultdict(list)
    # 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    visited = [0] * numCourses
    # 用数组来模拟栈，下标 0 为栈底，n-1 为栈顶
    result = list()
    # 判断有向图中是否有环
    valid = True

    for info in prerequisites:
        edges[info[1]].append(info[0])

    def dfs(u: int):
        nonlocal valid
        # 将节点标记为「搜索中」
        visited[u] = 1
        # 搜索其相邻节点
        # 只要发现有环，立刻停止搜索
        for v in edges[u]:
            # 如果「未搜索」那么搜索相邻节点
            if visited[v] == 0:
                dfs(v)
                if not valid:
                    return
            # 如果「搜索中」说明找到了环
            elif visited[v] == 1:
                valid = False
                return
        # 将节点标记为「已完成」
        visited[u] = 2
        # 将节点入栈
        result.append(u)

    # 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
    for i in range(numCourses):
        if valid and not visited[i]:
            dfs(i)

    if not valid:
        return list()

    # 如果没有环，那么就有拓扑排序
    # 注意下标 0 为栈底，因此需要将数组反序输出
    return result[::-1]


def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    indegrees = [0 for _ in range(numCourses)]
    adjacency = [[] for _ in range(numCourses)]
    queue = deque()
    l = []
    # Get the indegree and adjacency of every course.
    for cur, pre in prerequisites:
        indegrees[cur] += 1
        adjacency[pre].append(cur)
    # Get all the courses with the indegree of 0.
    num = 0
    for i in range(len(indegrees)):
        if not indegrees[i]:
            queue.append(i)
            l.append(i)
            num += 1
    # BFS TopSort.
    while queue:
        pre = queue.popleft()
        for cur in adjacency[pre]:
            indegrees[cur] -= 1
            if not indegrees[cur]:
                queue.append(cur)
                l.append(cur)
                num += 1
    return l if num >= numCourses else []


"""
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
https://leetcode.cn/problems/reconstruct-itinerary/
"""


def findItinerary(self, tickets: List[List[str]]) -> List[str]:
    from collections import defaultdict
    import copy
    tickets_table = defaultdict(list)
    for ticket in tickets:
        start_city, end_city = ticket[0], ticket[1]
        tickets_table[start_city].append(end_city)
    path = ['JFK']
    res = []
    print(tickets_table)

    def bt(start_city):
        if len(res) > 0:
            return
        if len(path) - 1 == len(tickets):
            res.append(path.copy())
            return
        if start_city not in tickets_table or tickets_table[start_city] == []:
            return
        end_city_list = list(set(tickets_table[start_city]))
        # set()
        end_city_list.sort()
        for end_city in end_city_list:
            path.append(end_city)
            # print(path)
            tickets_table[start_city].remove(end_city)
            bt(end_city)
            tickets_table[start_city].append(end_city)
            path.pop()

    bt('JFK')
    # print(path)
    res.sort()
    print('res:{}'.format(res))
    return res[0]
