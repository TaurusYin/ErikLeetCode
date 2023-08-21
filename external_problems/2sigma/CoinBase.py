"""
Suppose you are helping Two Sigma develop a build farm scheduler. The build
farm should build codebases according to their dependency constraints, and
should try to parallelize the build process as much as possible. Your job is to
make a build schedule that minimizes the total build time.
Input
You are given a list of codebases (String) with estimated build times (Positive
Integer). For example:
A:1, B:2, c:3, D:3, E:3
This means that codebase A takes 1 time unit to build, B takes 2, etc.
You are also given the codebase dependencies. For example:
B:A
C:A
D:A,B
E:B,C
This means codebase B depends on A, C depends on A, D depends on A and B,
etc. A is not listed, so it doesn't depend on any other codebases.
Output
Your function should return the optimal schedule as a list of strings in the
below format:
${execution_time},${codebase1},${codebase}.
*Note that the codebases don't need to follow any specific order. We

provide a helper function to ignore the ordering while testing.
The output should be :
1,A step 1: build[A] for 1 time unit
2,B,C step 2: build[B, C] for 2 time unit
1,C,D step 3: build[C, D] for 1 time unit
2,D,E step 4: build[D, E] for 2 time unit
1,E  step 5: build[E] for 1 time unit

他们家老生常谈的build task scheduler。要注意dependency graph存在环的情况，需要先判环
"""

import unittest
from typing import List, Tuple, Set
"""
write a hello world function
"""



class Task:
    def __init__(self, name: str, timeRemain: int):
        self.name = name
        self.timeRemain = timeRemain


def schedule(codebases: dict[str, int], deps: dict[str, Set[str]]) -> List[str]:
    tasksAfter = {}
    tasksHaveNoDep = []
    result = []

    # populating dependncies of a task
    for before, time in codebases.items():
        if before not in deps:
            task = Task(before, time)
            tasksHaveNoDep.append(task)

    # populate the tasks that have dependencies
    for after, befores in deps.items():
        for before in befores:
            if before not in tasksAfter:
                tasksAfter[before] = []

            tasksAfter[before].append(after)

    while tasksHaveNoDep:
        minTime = float("inf")
        clearedTasks = []

        while tasksHaveNoDep:
            t = tasksHaveNoDep.pop(0)
            minTime = min(t.timeRemain, minTime)
            clearedTasks.append(t)

        currRound = [str(minTime)]
        for clearedTask in clearedTasks:
            currRound.append(clearedTask.name)
            clearedTask.timeRemain = max(0, clearedTask.timeRemain - minTime)

            if clearedTask.timeRemain > 0:
                tasksHaveNoDep.append(clearedTask)
            else:
                if clearedTask.name not in tasksAfter:
                    continue

                tasksToBeCleared = tasksAfter[clearedTask.name]

                for taskToBeCleared in tasksToBeCleared:
                    tasks = deps[taskToBeCleared]
                    tasks.remove(clearedTask.name)
                    deps[taskToBeCleared] = tasks

                    if not deps[taskToBeCleared]:
                        newClearedTask = Task(taskToBeCleared, codebases[taskToBeCleared])
                        tasksHaveNoDep.append(newClearedTask)

        result.append(",".join(currRound))

    return result

from collections import deque

def top_schedule(codebases, deps):
    # Step 1: Find all nodes that don't have any incoming edges
    in_degree = {node: 0 for node in codebases}
    for befores in deps.values():
        for before in befores:
            in_degree[before] += 1

    queue = deque()
    for node, degree in in_degree.items():
        if degree == 0:
            queue.append(node)

    # Step 2: Perform topological sort
    result = []
    while queue:
        # Get the node with no incoming edges
        node = queue.popleft()
        # Add the node to the result
        result.append(node)
        # Decrease the in-degree of its neighbors
        for after in deps.get(node, []):
            in_degree[after] -= 1
            # If a neighbor has no more incoming edges, add it to the queue
            if in_degree[after] == 0:
                queue.append(after)

    # Step 3: Compute the completion time for each task
    task_time = {node: time for node, time in codebases.items()}
    completion_times = []
    for node in result:
        # Compute the completion time for the current node
        completion_time = task_time[node]
        for before in deps.get(node, []):
            completion_time = max(completion_time, task_time[before])
        completion_times.append(str(completion_time) + ',' + node)
        # Update the completion time for the neighbors of the current node
        for after in deps.get(node, []):
            task_time[after] = max(task_time[after], completion_time + codebases[after])

    return completion_times

'''
class TestSchedule(unittest.TestCase):
    def test_sample(self):
        codebases = {"A": 1, "B": 2, "C": 3, "D": 3, "E": 3}
        deps = {"B": {"A"}, "C": {"A"}, "D": {"A", "B"}, "E": {"B", "C"}}
        expected = ["1,A", "2,B,C", "1,C,D", "2,D,E", "1,E"]
        result = top_schedule(codebases, deps)
        self.assertEqual(result, expected)
'''

if __name__ == '__main__':
    from collections import deque


    def topological_sort(graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1

        queue = deque(node for node in graph if indegrees[node] == 0)
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) == len(graph):
            return result
        else:
            raise ValueError("The graph contains a cycle.")


    # example graph
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D", "E"],
        "D": ["F"],
        "E": ["F"],
        "F": []
    }

    print(topological_sort(graph))  # ['A', 'C', 'B', 'E', 'D', 'F']

    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
