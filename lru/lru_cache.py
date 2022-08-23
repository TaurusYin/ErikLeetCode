import collections
from typing import *

"""
1. Implement the interface: 这里的要求是改写出一个接口类，用以适用于任何的缓存替换算法（而不是地里其他面经说的仅仅LRU和MRU）
2. Optimize it: 这里要求你优化代码结构，但不能改变代码数据类型——我一开始想把Array换成Map，被告知这是一个好办法，但不希望我在这里这么做，地里其他面经里的优化还不够到位，可以最终达到O(n)
3. Make it thread-safe: 要求把代码改成多线程环境下也是安全的形式，这里可以使用别的数据类型，性能可以达到O(1)
4. Implement LRU & MRU: 要求基于以上更改，实现LRU和MRU
5. Optimize MRU to O(1): 要求把类似于LRU实现的MRU优化成O(1)的另一种实现
https://blog.csdn.net/qq_25800311/article/details/89074599
"""
from abc import abstractmethod, ABCMeta


class ICacheAlgorithms(metaclass=ABCMeta):
    @abstractmethod
    def get(self, key):
        """
        :param key: returns the corresponding value to the key
        :return: Key to the value to return
        """
        pass

    @abstractmethod
    def put(self, key, val):
        """
        puts key-value pair into setData and sets its DNode to head
        :param key:
        :param val:
        :return:
        """
        pass

    """
    @abstractmethod
    def get_map_data(self):
        pass

    @abstractmethod
    def __contains__(self, item) -> bool:
        pass
    
    @abstractmethod
    def is_full(self) -> bool:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass
    """

    @abstractmethod
    def del_key(self, key):
        """
        Removes node from doubly-linked list (DLL)
        :param key:
        :param val:
        :return:
        """

    pass


class LRUCache(ICacheAlgorithms):
    def __init__(self, capacity, n_sets=0):
        self.capacity = capacity
        self.map = {}  # k:key, v:node
        self.cache = DoubleLinkedList()
        self.n_sets = n_sets

    def get(self, key):
        if key in self.map:  # isExist()
            self.make_recent(key)  # get_algorithm_handler()
            return self.map[key].val
        else:
            return -1

    def put(self, key, val):
        if key in self.map:  # isExist()
            self.del_key(key)  # remove(key)
            # self.add_recent_key(key, val)
            # return

        else:
            if self.capacity == self.cache.size:  # isFull()
                self.remove_least_key()  # remove_algorithm_handler()
        self.add_recent_key(key, val)  # put_algorithm_handler()
        return
        # adjust cache

    def get_map_data(self):
        return self.map

    def __contains__(self, item) -> bool:
        if self.map.get(item):
            return True
        else:
            return False

    def make_recent(self, key):
        node = self.map[key]
        self.cache.remove(node)
        self.cache.add_last(node)
        return

    # add last(recent) k, v
    def add_recent_key(self, key, val):
        node = Node(key=key, val=val)
        self.cache.add_last(node)
        self.map[key] = node
        return

    # remove first(least)
    def remove_least_key(self):
        deleted_node = self.cache.remove_first()
        self.map.pop(deleted_node.key)
        return

    # remove key
    def del_key(self, key):
        x = self.map[key]
        self.cache.remove(x)
        self.map.pop(key)


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

    def remove_last(self):
        return

    def add_first(self, x: Node):
        return

    # remove node
    def remove(self, x: Node):
        x.prev.next = x.next
        x.next.prev = x.prev
        self.size -= 1

    def get_size(self):
        return self.size

    def is_empty(self):
        return

    def is_full(self):
        return


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


class NewLRUCache(ICacheAlgorithms, DoubleLinkedList):
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


if __name__ == '__main__':
    obj = LRUCache(capacity=2)
    obj.put(key=2, val=111)
    param_1 = obj.get(key=2)
    res = obj.__contains__(1)
    print()

    node = Node(prev='x', next='y', key='k', val='v')
    print()
