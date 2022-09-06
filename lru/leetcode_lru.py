import collections


class Node(object):
    def __init__(self, key, val, prev=None, next=None, **kw):
        self.prev, self.next = prev, next
        self.key, self.val = key, val
        for k, v in kw.items():
            setattr(self, k, v)

class DoubleLinkedList(object):
    def __init__(self):
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    # remove_first
    def remove_first(self):
        if self.head.next == self.tail:
            return
        first = self.head.next
        self.remove(first)
        return first

    # add_last
    def add_last(self, x: Node):
        # self.tail表示尾部的dummy节点, self.tail.prev表示尾结点, x.prev表示x的前驱为尾结点
        x.prev = self.tail.prev
        x.next = self.tail
        # handle tail.prev
        self.tail.prev.next = x
        self.tail.prev = x
        self.size += 1

    # remove node
    def remove(self, x: Node):
        x.prev.next = x.next
        x.next.prev = x.prev
        self.size -= 1

    def get_size(self):
        return self.size


class LRUCache(DoubleLinkedList):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.map = {}

    def get(self, key):
        if key in self.map:
            self.make_recent(key)
            return self.map[key].val
        else:
            return -1

    def put(self, key, val):
        if key in self.map:
            self.del_key(key)
            self.add_recent_key(key, val)
            return

        else:
            if self.capacity == self.size:
                self.remove_least_key()
        self.add_recent_key(key, val)
        return
        # adjust cache

    def make_recent(self, key):
        node = self.map[key]
        self.remove(node)
        self.add_last(node)
        return

    # add last(recent) k, v
    def add_recent_key(self, key, val):
        node = Node(key=key, val=val)
        self.add_last(node)
        self.map[key] = node
        return

    # remove first(least)
    def remove_least_key(self):
        deleted_node = self.remove_first()
        self.map.pop(deleted_node.key)
        return

    # remove key
    def del_key(self, key):
        x = self.map[key]
        self.remove(x)
        self.map.pop(key)


class OrderDictLRUCache(collections.OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)