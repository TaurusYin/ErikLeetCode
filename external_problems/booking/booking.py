"""
Given a list of hotelId, parentHotelId and a score retrieve the top k root parentHotelIds with the highest scores:
[{0, 1, 10}, {1, 2, 20}, {3, 4, 10}, {7, 8, 5}] K = 2
Result: [[2, 20], [4,10]]
"""
import heapq
from collections import defaultdict, OrderedDict

from union_find.base_union_find import UnionFind

input = [[0, 1, 10], [1, 2, 20], [3, 4, 10], [7, 8, 5]]
K = 2

input = [
    (3, 0, 14),
    (0, None, 10),
    (4, 0, 44),
    (6, None, 7),
    (10, 6, 13),
    (7, 6, 17),
    (2, None, 2),
    (14, 2, 9),
    (25, 14, 10),
    (12, 2, 10),
    (13, 0, 1),
    (14, 2, 9),
]


class Node:
    def __init__(self, value: list):
        self.value = value

    def __lt__(self, other: list):
        return self.value[2] > other.value[2]

    def __eq__(self, other): return self.value[2] == other.value[2]

    def __str__(self): return str(self.value[2])


def k_largest_scores(input, k):
    uf = UnionFind(100)
    parent_list = []
    for id, parent_id, score in input:
        if id is not None and parent_id is not None:
            uf.union(parent_id, id)
    for id, parent_id, score in input:
        root_id = uf.find(id)
        if root_id == id:
            parent_list.append([id, parent_id, score])
    parent_list = sorted(parent_list, key=lambda x: -x[2])
    parent_list = [Node(x) for x in parent_list]
    heap = []

    heap = []
    for item in parent_list:
        # 如果堆长度没到k，无脑塞
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            # 如果长度到k了，且当前元素比堆顶要大，我们才加进去，当然要先把最小的pop出来再加！
            x = heap[0]
            if item.value[2] > heap[0].value[2]:
                heapq.heappop(heap)
                heapq.heappush(heap, item)

    res = list(map(lambda x:[x.value[0],x.value[2]], heap))
    return res


def _k_largest_scores(input, k):
    hotel_ids = {}
    parent_hotels = defaultdict()
    parent_scores = OrderedDict()
    scores = []
    for id, parent_id, score in input:
        hotel_ids[id] = [id, parent_id, score]

    for id, parent_id, score in input:
        current_id = parent_id
        while hotel_ids[current_id]:
            current_id = hotel_ids[parent_id][1]
            if current_id is None:
                current_id = parent_id
                break
        parent_hotels[current_id] = hotel_ids[current_id]

    scores = list(parent_scores.keys())
    heap = [x for x in scores[:k]]
    heapq.heapify(heap)
    n = len(scores)
    for i in range(k, n):
        if scores[i] > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, scores[i])
    res = [parent_scores[score] for score in heap]
    return res


res = k_largest_scores(input=input, k=K)
print()
