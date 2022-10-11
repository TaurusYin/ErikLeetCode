class Node:
    def __init__(self, key, val, pre=None, nex=None, freq=0):
        self.pre = pre
        self.nex = nex
        self.freq = freq
        self.val = val
        self.key = key

    def insert(self, nex):
        nex.pre = self
        nex.nex = self.nex
        self.nex.pre = nex
        self.nex = nex


def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.nex = tail
    tail.pre = head
    return (head, tail)


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.keyMap = {}

    def delete(self, node):
        if node.pre:
            node.pre.nex = node.nex
            node.nex.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
        return node.key

    def increase(self, node):
        node.freq += 1
        self.delete(node)
        self.freqMap[node.freq][-1].pre.insert(node)
        if node.freq == 1:
            self.minFreq = 1
        elif self.minFreq == node.freq - 1:
            head, tail = self.freqMap[node.freq - 1]
            if head.nex is tail:
                self.minFreq = node.freq

    def get(self, key: int) -> int:
        if key in self.keyMap:
            self.increase(self.keyMap[key])
            return self.keyMap[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity != 0:
            if key in self.keyMap:
                node = self.keyMap[key]
                node.val = value
            else:
                node = Node(key, value)
                self.keyMap[key] = node
                self.size += 1
            if self.size > self.capacity:
                self.size -= 1
                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
                self.keyMap.pop(deleted)
            self.increase(node)

"""
解法
为了达成get和put的O(1)复杂度，需要用字典、有序字典这两种数据结构。
由于需要根据频率f来删除最不常用的键值对，
因此需要一个f2kv的哈希字典，且这个哈希字典的元素都是有序字典OrderedDict。
因此需要一个minFreq记录最小的频率，免得删除时要遍历取最小freq
因此需要k2f的哈希字典，以取出key对应的freq，以及根据freq去OrderedDict中删除kv键值对
get方法设计：
当键不存在时，直接返回-1
当键存在时，增加key出现的频次
put方法设计：
当键存在时，更新key的值和频次
当键不存在时，插入k2f和f2kv，检查容量越界，删除最不常用的键值对

作者：jiatelin
链接：https://leetcode.cn/problems/lfu-cache/solution/lfuhuan-cun-by-jiatelin-361i/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""
class LFUCache:

    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self.f2kv = {}  # 这个需要时序
        self.k2f = {}
        self.min_f = -1
        self.nums = 0

    def get(self, key: int) -> int:
        # 当key存在时，增加k2f频率，更改f2kv中f键
        if key in self.k2f:
            freq = self.k2f[key]
            val = self.f2kv[freq][key]
            self.increaseFreq(key)
            return val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:  # 检查capacity的边界值
            return None
        # 当key存在时，增加k2f频率，更改f2kv中的f值，并且更改v值
        if key in self.k2f:
            freq = self.k2f[key]
            self.increaseFreq(key, value)
        else:
            # key不存在时，新增key-value对到k2f和f2kv中，并且检查是否超过最大容量
            self.nums += 1
            self.f2kv[1] = self.f2kv.get(1, OrderedDict())
            self.f2kv[1][key] = value
            self.k2f[key] = 1
            self.removeMinFreq()
            self.min_f = 1

    def increaseFreq(self, key, value=None):
        # 增加k2f记录的freq值
        # 更改f2kv中kv的键
        freq = self.k2f[key]
        self.k2f[key] += 1
        self.f2kv[freq + 1] = self.f2kv.get(freq + 1, OrderedDict())
        if value is None:
            self.f2kv[freq + 1][key] = self.f2kv[freq][key]
        else:
            self.f2kv[freq + 1][key] = value
        del self.f2kv[freq][key]
        if not self.f2kv[freq]:
            del self.f2kv[freq]
        if self.min_f == freq and not self.f2kv.get(freq, None):
            self.min_f += 1

    def removeMinFreq(self):
        # 删除f2kv中kv的键值对
        # 删除k2f中kf键值对
        if self.nums > self.capacity:
            k, v = self.f2kv[self.min_f].popitem(last=False)
            if not self.f2kv[self.min_f]:
                del self.f2kv[self.min_f]
            del self.k2f[k]
            self.nums = self.capacity


# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

