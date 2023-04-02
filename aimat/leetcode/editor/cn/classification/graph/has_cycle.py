"""
@File    :   has_cycle.py   
@Contact :   yinjialai 
"""
from collections import defaultdict


def has_cycle(graph: defaultdict):
    def dfs(node, visited, visiting):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if node in graph:  # 规避访问不存在元素自动改变defaultdict的问题
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


# 测试用例
graph1 = defaultdict(list, {
    'a': ['b'],
    'b': ['c'],
    'c': []
})

graph2 = defaultdict(list, {
    'a': ['b'],
    'b': ['c'],
    'c': ['a']
})

print(has_cycle(graph1))  # 输出：False
print(has_cycle(graph2))  # 输出：True

from collections import defaultdict, deque


def has_cycle(graph: defaultdict):
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


# 测试用例
graph1 = defaultdict(list, {
    'a': ['b'],
    'b': ['c'],
    'c': []
})

graph2 = defaultdict(list, {
    'a': ['b'],
    'b': ['c'],
    'c': ['a']
})

print(has_cycle(graph1))  # 输出：False
print(has_cycle(graph2))  # 输出：True
