from collections import OrderedDict
from heapq import heappush, heappop


class UtilsDS:
    def __init__(self):
        return

    def sort_utils(self):
        nums = [[0, 1, 10], [1, 2, 20], [3, 4, 10], [7, 8, 5]]
        nums = sorted(nums, key=lambda x: -x[2])
        d = OrderedDict()
        for item in nums:
            d[item[2]] = [item[1], item[2]]
        print(d)

    def heapsort(self, iterable):
        h = []
        for value in iterable:
            heappush(h, value)
        res = [heappop(h) for i in range(len(h))]
        return res




if __name__ == '__main__':
    uds = UtilsDS()
    uds.sort_utils()
    uds.heapsort(iterable=[32,4,62,3,7,5])