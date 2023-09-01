from collections import defaultdict, deque
from collections import defaultdict


def adj_matrix_to_graph(adj_matrix):
    graph = defaultdict(list)
    for i, row in enumerate(adj_matrix):
        for j, col in enumerate(row):
            if col:  # ？ 可以去掉
                graph[i].append(j)
    return graph


adj_matrix = [
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
graph = adj_matrix_to_graph(adj_matrix)


def adj_graph_to_matrix(graph: defaultdict):
    # 获取最大的节点编号
    max_node = max(max(graph.keys()), max(max(val) for val in graph.values()))
    # 初始化邻接矩阵
    adj_matrix = [[0] * (max_node + 1) for _ in range(max_node + 1)]
    # 填充邻接矩阵
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1
    return adj_matrix


graph = defaultdict(list, {
    0: [1, 2],
    1: [3, 4],
    2: [3, 5],
    3: [4, 5],
    4: [],
    5: []
})


# adj_matrix = adj_graph_to_matrix(graph)
# for row in adj_matrix:
#  print(row)


def topo_sort(graph):
    """
    拓扑排序的算法流程如下：
    找出所有入度为0的节点，将它们加入一个队列中。
    对于队列中的每个节点，将它从队列中移除，并将它的所有邻居节点的入度减1。如果邻居节点的入度变为0，则将邻居节点加入队列中。
    重复步骤2直到队列为空。排序的结果就是每个节点被移除队列的顺序。
    defaultdict 没有key value 返回0 访问a后，就默认加上了 'a':0
    defaultdict(<class 'int'>, {'b': 1, 'c': 1, 'd': 2, 'e': 2, 'f': 2, 'a': 0})
    :param graph:
    :return:
    """
    # 统计每个节点的入度
    in_degree = defaultdict(int)
    for node in graph:
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


def topo_sort_dfs(graph):
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


# 测试
graph = defaultdict(list, {
    'a': ['b', 'c'],
    'b': ['d', 'e'],
    'c': ['d', 'f'],
    'd': ['e', 'f'],
    'e': [],
    'f': []
})

graph = defaultdict(list, {'t': {'f'}, 'w': {'r', 'e'}, 'r': {'t'}, 'e': {'r'}})

print(topo_sort(graph))  # ['a', 'c', 'b', 'd', 'e', 'f']
print(topo_sort_dfs(graph))  # ['a', 'c', 'b', 'd', 'e', 'f']
print()

"""
def top_sort(graph: defaultdict):
    from collections import defaultdict
    from collections import deque
    in_degree = defaultdict(int)
    for node in graph.keys():
        neighbours = graph[node]
        for neighbour in neighbours:
            in_degree[neighbour] += 1

    print(in_degree)
    queue = deque([node for node in graph.keys() if in_degree[node] == 0])
    print(queue)
    print()
    result = []
    while queue:
        cur_node = queue.popleft()
        result.append(cur_node)
        print(result)
        for neighbour in graph[cur_node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
    if len(result) != len(graph):
        return []
    return result


def top_dfs(graph: defaultdict):
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        neighbours = graph[node]
        for neighbour in neighbours:
            dfs(neighbour)
        post_order.append(node)

    visited = set()
    post_order = []
    for node in graph.keys():
        dfs(node)
    if len(post_order) != len(graph):
        return []
    return list(reversed(post_order))
"""
