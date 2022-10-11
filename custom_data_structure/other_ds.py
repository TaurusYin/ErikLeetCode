import collections


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




