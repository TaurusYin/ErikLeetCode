"""
Dijkstra 算法的时间复杂度是多少？你去网上查，可能会告诉你是 O(ElogV)，其中 E 代表图中边的条数，V 代表图中节点的个数。

因为理想情况下优先级队列中最多装 V 个节点，对优先级队列的操作次数和 E 成正比，所以整体的时间复杂度就是 O(ElogV)。

不过这是理想情况，Dijkstra 算法的代码实现有很多版本，不同编程语言或者不同数据结构 API 都会导致算法的时间复杂度发生一些改变。

比如本文实现的 Dijkstra 算法，使用了 Java 的 PriorityQueue 这个数据结构，这个容器类底层使用二叉堆实现，但没有提供通过索引操作队列中元素的 API，所以队列中会有重复的节点，最多可能有 E 个节点存在队列中。

所以本文实现的 Dijkstra 算法复杂度并不是理想情况下的 O(ElogV)，而是 O(ElogE)，可能会略大一些，因为图中边的条数一般是大于节点的个数的。

不过就对数函数来说，就算真数大一些，对数函数的结果也大不了多少，所以这个算法实现的实际运行效率也是很高的，以上只是理论层面的时间复杂度分析，供大家参考。

"""




"""
根据题意，从节点 kk 发出的信号，到达节点 xx 的时间就是节点 kk 到节点 xx 的最短路的长度。因此我们需要求出节点 kk 到其余所有点的最短路，其中的最大值就是答案。若存在从 kk 出发无法到达的点，则返回 -1−1。

下面的代码将节点编号减小了 11，从而使节点编号位于 [0,n-1][0,n−1] 范围。
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
链接：https://leetcode.cn/problems/network-delay-time/solution/wang-luo-yan-chi-shi-jian-by-leetcode-so-6phc/

"""
import heapq
from typing import List


class Solution:

    # 首先将边转化为邻接表
    # 然后从k节点出发遍历节点，每遍历一个节点，将其到达时间进行更新。
    # 因图中节点可能有多条路径到达，或者有环路，只有到达时间比之前记录的时间小才重复遍历
    #
    # 作者：wanglongjiang
    # 链接：https://leetcode.cn/problems/network-delay-time/solution/dfs-by-wanglongjiang-3n7q/
    # 来源：力扣（LeetCode）
    # 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 输入转为邻接表
        graph = [[] for _ in range(n + 1)]
        for time in times:
            graph[time[0]].append((time[1], time[2]))
        arrivalTimes = [float('inf')] * (n + 1)

        # 遍历节点
        def dfs(nodeid, time):
            if arrivalTimes[nodeid] <= time:
                return
            arrivalTimes[nodeid] = time
            for nextid, nexttime in graph[nodeid]:
                dfs(nextid, time + nexttime)

        dfs(k, 0)
        # 找到最大的到达时间
        ans = float('-inf')
        for time in arrivalTimes[1:]:
            if time == float('inf'):
                return -1
            if ans < time:
                ans = time
        return ans


    # 思路2、Dijkstra算法
    #
    # 首先将边转化为邻接表
    # 设arrivalTimes数组，arrivalTimes[i]的含义是从i节点出发到达k需要的时间
    # 将k节点加入堆，从k节点出发遍历节点，每遍历一个节点，将其到达的其他节点加入最小堆（只有小于arrivalTimes中旧值的才加入）。
    #
    # https://leetcode.cn/problems/network-delay-time/solution/dfs-by-wanglongjiang-3n7q/
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # 输入转为邻接表
        graph = [[] for _ in range(n + 1)]
        for time in times:
            # time[0]: current_node, time[1]: next_node, time[2]: weight
            graph[time[0]].append((time[1], time[2]))
        # 用Dijkstra算法更新到达时间
        arrivalTimes = [float('inf')] * (n + 1)  # 到达时间
        # 起始点k
        arrivalTimes[k] = 0
        heap = [(0, k)]  # (weight, node)
        while heap:
            time, node = heapq.heappop(heap)
            for nextNode, nextTime in graph[node]:
                if (d := time + nextTime) < arrivalTimes[nextNode]:  # 如果下一节点的到达时间>当前节点到达时间+当前到下一节点时间
                    arrivalTimes[nextNode] = d  # 更新下一节点的到达时间
                    heapq.heappush(heap, (d, nextNode))  # 下一节点连结的节点也可能会更新，加入最小堆
        # 找到最大的到达时间
        ans = float('-inf')
        for time in arrivalTimes[1:]:
            if time == float('inf'):
                return -1
            if ans < time:
                ans = time
        return ans

    # O(n2 + m) m:times 长度 n:节点个数  O(n2),
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[float('inf')] * n for _ in range(n)]
        for x, y, time in times:
            g[x - 1][y - 1] = time

        dist = [float('inf')] * n
        dist[k - 1] = 0
        used = [False] * n
        for _ in range(n):
            x = -1
            for y, u in enumerate(used):
                if not u and (x == -1 or dist[y] < dist[x]):
                    x = y
            used[x] = True
            for y, time in enumerate(g[x]):
                dist[y] = min(dist[y], dist[x] + time)

        ans = max(dist)
        return ans if ans < float('inf') else -1

    # O(mlogm) m:times 长度  n:节点个数 O(m+n)
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[] for _ in range(n)]
        for x, y, time in times:
            g[x - 1].append((y - 1, time))

        dist = [float('inf')] * n
        dist[k - 1] = 0
        q = [(0, k - 1)]
        while q:
            time, x = heapq.heappop(q)
            if dist[x] < time:
                continue
            for y, time in g[x]:
                if (d := dist[x] + time) < dist[y]:
                    dist[y] = d
                    heapq.heappush(q, (d, y))

        ans = max(dist)
        return ans if ans < float('inf') else -1





"""
也就是说，到达某一个点，所有需要的总的体力值，是所有相邻格子间高度差绝对值的中最大一个，并不是求和。
所以，我们可以从起点开始搜索邻近节点，计算出到达每一个点需要的最小体力值，同时利用优先级队列，确定下一个需要搜索的点。
链接：https://leetcode.cn/problems/path-with-minimum-effort/solution/1631-dui-zui-xiao-ti-li-xiao-hao-lu-jing-j7ae/
O(mn*logmn)
"""
import heapq
from typing import List

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = len(heights), len(heights[0])
        # 用于记录已经访问过的点
        visited = set()
        # 用于记录到达每个点消耗的最小体力值
        energies = [[float("INF") for _ in range(n)] for _ in range(m)]
        # 小根堆：用于记录运动到点（i,j）需要消耗的最小能量和当前坐标（i,j）
        heap = list()
        # 将起点放入堆中，起点消耗的体力值是零
        energies[0][0] = 0
        heapq.heappush(heap, (0, 0, 0))
        # 从起点开始遍历周围的每一个点，找到每个点消耗的最小体力值
        while heap:
            # 到达点（i,j）最少需要的能量为energy
            energy, i, j = heapq.heappop(heap)
            if (i, j) in visited:
                continue

            # 如果遍历到最右下角的点，则停止搜索
            if i == m - 1 and j == n - 1:
                break

            visited.add((i, j))
            # 计算当前点（i,j）四周邻近的点需要消耗的最少能量
            for direction in DIRECTIONS:
                x, y = i + direction[0], j + direction[1]
                if 0 <= x < m and 0 <= y < n:
                    # 从当前点（i，j）到邻近的下一个点（x, y）需要消耗的能量
                    needed_energy = abs(heights[x][y] - heights[i][j])
                    # 从起点（0, 0）到（x,y）需要消耗的总能量
                    total_energy = max(energy, needed_energy)
                    # 如果从起点（0, 0）到达点（x,y）消耗的能量，小于从其他路径到达（x,y）消耗的能量，则更新到达点（x,y）的能量消耗值
                    if total_energy <= energies[x][y]:
                        energies[x][y] = total_energy
                        # 并将所消耗的总能量值和对应坐标放入堆中
                        heapq.heappush(heap, (energies[x][y], x, y))

        return energies[-1][-1]

'''
输入：n = 3, edges = [[0,1],[1,2],[0,2]], succProb = [0.5,0.5,0.2], start = 0, end = 2
输出：0.25000
解释：从起点到终点有两条路径，其中一条的成功概率为 0.2 ，而另一条为 0.5 * 0.5 = 0.25
'''

def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        e = [[] for i in range(n)]
        # 建边，无向图，先存指向再存边权值
        for i, [x, y] in enumerate(edges):
            e[x].append([y, succProb[i]])
            e[y].append([x, succProb[i]])
        # l存start到各边的最大值，初始化为零，最短路中初始化为inf，存start到各边的最小值
        # q为优先队列，队首为与start相连边的最大值，最短路中为最小值
        l, q = [0] * n, []
        # 将start入队
        heapq.heappush(q, [1, start])
        while q:
            x, y = heapq.heappop(q)
            # 因为构建的是大顶堆，存的第一个为边的权值，为负数，如果改变的权值的绝对值小于l中的，不需要对该点的边进行更新
            if abs(x) < l[y]: continue
            # 遍历该点的边，将边的值进行更新，并加入到优先队列中
            for i, j in e[y]:
                if abs(j * x) > l[i]:
                    l[i] = abs(j * x)
                    heapq.heappush(q, [-abs(j * x), i])
        return l[end]



