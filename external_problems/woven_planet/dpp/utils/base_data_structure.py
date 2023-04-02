import heapq
from collections import defaultdict, OrderedDict


class ListItem:
    def __init__(self, value: list):
        self.value = value

    def __lt__(self, other: list):
        return int(self.value[1]) < int(other.value[1])

    def __eq__(self, other): return int(self.value[1]) == int(other.value[1])

    def __str__(self): return str(self.value[1])


class MinHeap:
    def __init__(self, k):
        self.k = k
        self.heap = []
        return

    def __str__(self):
        records = []
        for list_item in self.heap:
            records.append(list_item.value[0])
        output_string = '\n'.join(records)
        return str(output_string)

    def add_item(self, item: ListItem):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        else:
            top_item = self.heap[0]
            if int(item.value[1]) > int(top_item.value[1]):
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, item)
        return
