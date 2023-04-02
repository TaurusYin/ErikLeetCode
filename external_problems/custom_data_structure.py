# 代码实现单向链表by Fan
class Node(object):
    # 节点,数据区加链接区
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    # 单向链表，定义头
    def __init__(self, node=None):
        self._head = node

    # 判断链表是否为空
    def is_empty(self):
        print(self._head == None)

    # 获取链表的长度
    def length(self):
        cur = self._head
        count = 0
        while cur != None:
            cur = cur.next
            count += 1
        return count

    # 遍历链表
    def travel(self):
        cur = self._head
        while cur != None:
            print(cur.elem, end="")
            cur = cur.next

    # 头部插入
    def add(self, item):
        node = Node(item)
        node.next = self._head
        self._head = node

    # 尾部插入
    def append(self, item):
        node = Node(item)
        if self._head == None:
            self._head = node
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    # 指定位置插入
    def insert(self, pos, item):
        if pos < 1:
            self.add(item)
        elif pos > self.length() - 1:
            self.append(item)
        else:
            node = Node(item)
            pre = None
            cur = self._head
            count = 0
            while count < pos:
                pre = cur
                cur = cur.next
                count += 1
            pre.next = node
            node.next = cur

    # 查找特定元素是否存在
    def search(self, item):
        cur = self._head
        while cur != None:
            if cur.elem == item:
                print("在")
                return True
            else:
                cur = cur.next
        print("不在")
        return False

    # 删除某个特定元素
    def shanchu(self, item):
        pre = None
        cur = self._head
        while cur != None:
            if cur.elem == item:
                # 判断该节点是否为头节点
                if cur == self._head:
                    self._head = cur.next
                    return True
                else:
                    pre.next = cur.next
                    return True
            else:
                pre = cur
                cur = cur.next
        print("该元素不存在")

    # 删除某个位置的元素
    def delete(self, pos):
        pre = None
        cur = self._head
        i = 0
        j = 0
        if cur == None:
            print("删除请求不合法")
        elif pos == 0:
            self._head = cur.next
        elif pos == -1:
            if cur.next == None:
                self._head = None
            else:
                while cur.next != None:
                    pre = cur
                    cur = cur.next
                pre.next = None
        else:
            if pos > self.length() - 1:
                print("删除请求不合法")
            else:
                while i <= pos:
                    if j < pos:
                        pre = cur
                        cur = cur.next
                        j += 1
                    else:
                        pre.next = cur.next
                    i += 1


class Deque(object):
    # 创建一个空的双端队列Deque
    def __init__(self):
        self._list = SingleLinkList()

    # add_front()从队列头加入一个元素
    def add_front(self, item):
        self._list.add(item)

    # add_rear()从队尾加入一个元素
    def add_rear(self, item):
        self._list.append(item)

    # remove_front()从队头删除一个元素
    def remove_front(self):
        self._list.delete(0)

    # remove_rear()从队尾删除一个元素
    def remove_rear(self):
        self._list.delete(-1)

    # 判空函数
    def is_empty(self):
        print(self._list._head == None)

    # 返回队列的大小
    def size(self):
        return self._list.length()

    def travel(self):
        self._list.travel()


class Queue(object):

    def __init__(self):
        self._list = SingleLinkList()

    # 添加一个新的元素
    def enqueue(self, item):
        self._list.append(item)

    # 从队列头部删除一个元素
    def dequeue(self):
        self._list.delete(0)

    # 判空
    def is_empty(self):
        print(self._list._head == None)

    # 返回Queue中元素的个数
    def size(self):
        print(self._list.length())


class Stack(object):

    def __init__(self):
        self._list = SingleLinkList()

    # 添加一个新的元素
    def push(self, item):
        self._list.add(item)

    # 弹出栈顶元素
    def pop(self):
        self._list.delete(0)

    # 返回栈顶元素
    def peek(self):
        a = self._list._head.elem
        print(a)

    # 判空
    def is_empty(self):
        print(self._list._head == None)

    # 返回栈中元素的个数
    def size(self):
        print(self._list.length())
