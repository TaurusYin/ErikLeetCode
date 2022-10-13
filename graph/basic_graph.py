from collections import defaultdict

from collections import defaultdict

def matrix2table(matrix):
    result = []
    N = len(matrix)
    for i in range(N):
        tmp1 = []
        for j in range(N):
            if matrix[i][j]:
                tmp1.append(j)
        result.append(tmp1)
    return result


def table2matrix(table):
    ret = []
    N = len(table)
    for i in range(N):
        tmp = [0] * N
        for j in table[i]:
            tmp[j] = 1
        ret.append(tmp)
    return ret


matrix = [
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0]
]

tb = matrix2table(matrix)
print(tb)

mat = table2matrix(tb)
print(mat)



ticket_dict = defaultdict(list)
for item in tickets:
    ticket_dict[item[0]].append(item[1])


class AdjNode:
    def __init__(self, data):
        self.vertex = data
        self.next = None


# A class to represent a graph. A graph
# is the list of the adjacency lists.
# Size of the array will be the no. of the
# vertices "V"
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [None] * self.V

    # Function to add an edge in an undirected graph
    def add_edge(self, src, dest):
        # Adding the node to the source node
        node = AdjNode(dest)
        node.next = self.graph[src]
        self.graph[src] = node

        # Adding the source node to the destination as
        # it is the undirected graph
        node = AdjNode(src)
        node.next = self.graph[dest]
        self.graph[dest] = node

    # Function to print the graph
    def print_graph(self):
        for i in range(self.V):
            print("Adjacency list of vertex {}\n head".format(i), end="")
            temp = self.graph[i]
            while temp:
                print(" -> {}".format(temp.vertex), end="")
                temp = temp.next
            print(" \n")


V = 5
graph = Graph(V)
graph.add_edge(0, 1)
graph.add_edge(0, 4)
graph.add_edge(1, 2)
graph.add_edge(1, 3)
graph.add_edge(1, 4)
graph.add_edge(2, 3)
graph.add_edge(3, 4)
graph.print_graph()


class graph(object):

    def __init__(self, v):

        self.v = v
        self.graph = defaultdict(set)

    # Adds an edge to undirected graph
    def addEdge(self, source, destination):

        # Add an edge from source to destination.
        # If source is not present in dict add source to dict
        # self.graph.add(destination)

        # Add an dge from destination to source.
        # If destination is not present in dict add destination to dict
        self.graph[destination].add(source)

    # A utility function to print the adjacency
    # list representation of graph
    def print(self):

        for i, adjlist in sorted(self.graph.items()):
            print("Adjacency list of vertex ", i)
            for j in sorted(adjlist, reverse=True):
                print(j, end=" ")
            print('\n')


g = graph(5)

g.addEdge(0, 1)
g.addEdge(0, 4)
g.addEdge(1, 2)
g.addEdge(1, 3)
g.addEdge(1, 4)
g.addEdge(2, 3)
g.addEdge(3, 4)

# Print adjacenecy list
# representation of graph
g.print()
