import heapq


class MaxStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._data = DoubleLinkedList()
        self._max = TreeMap()

    def push(self, x: int) -> None:
        node = Node(x)
        self._data.add(node)
        if x in self._max:
            self._max[x].append(node)
        else:
            self._max[x] = [node]

    def pop(self) -> int:
        node = self._data.head.next
        self._data.remove(node)
        val = node.val
        self._max[val].pop()
        if len(self._max[val]) == 0:
            del self._max[val]
        return val

    def top(self) -> int:
        return self._data.head.next.val

    def peekMax(self) -> int:
        li = list(self._max.keys())
        return self._max[li[-1]][-1].val

    def popMax(self) -> int:
        li = list(self._max.keys())
        max_key = li[-1]
        node = self._max[max_key].pop()
        self._data.remove(node)
        val = node.val
        if len(self._max[val]) == 0:
            del self._max[val]
        return val

class Node:
    def __init__(self, val=None, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


class DoubleLinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def add(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next = node
        node.next.prev = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev


class TreeMap(collections.OrderedDict):
    def __init__(self):
        super(TreeMap, self).__init__()

    def __setitem__(self, key, val):
        if key in self:
            super(TreeMap, self).__setitem__(key, val)
        else:
            super(TreeMap, self).__setitem__(key, val)
            tmp = dict(self)
            self.clear()
            for key in sorted(tmp.keys()):
                super(TreeMap, self).__setitem__(key, tmp[key])


class MaxStack:
# 自增序列
# 最大堆
# soft delete
    def __init__(self):
        self.heap = []
        self.stack = []
        self.popped_set = set()
        self.count = 0

    def push(self, x: int) -> None:
        item = (-x, -self.count)
        self.stack.append(item)
        heapq.heappush(self.heap, item)
        self.count += 1

    def pop(self) -> int:
        self.clear_pop_in_stack() # 先清理标记
        item = self.stack.pop() # 弹出当前元素，加入删除标记
        self.popped_set.add(item)
        return -item[0]

    def top(self) -> int:
        self.clear_pop_in_stack() # 先清理标记
        item = self.stack[-1]
        return -item[0]

    def peekMax(self) -> int:
        self.clear_pop_in_heap() # 先清理标记
        item = self.heap[0]
        return -item[0]

    def popMax(self) -> int:
        self.clear_pop_in_heap()
        item = heapq.heappop(self.heap)
        self.popped_set.add(item) #标记堆中pop的元素
        return -item[0]

    # 每次stack操作时先清空stack里标记删除的元素
    def clear_pop_in_stack(self):
        while self.stack and self.stack[-1] in self.popped_set:
            self.popped_set.remove(self.stack[-1])
            self.stack.pop()
    # 每次heap操作时先清空heap里标记删除的元素
    def clear_pop_in_heap(self):
        while self.heap and self.heap[0] in self.popped_set:
            self.popped_set.remove(self.heap[0])
            heapq.heappop(self.heap)



# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()


