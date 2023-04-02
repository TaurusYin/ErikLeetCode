class Node(object):
    """声明节点"""

    def __init__(self, element):
        self.element = element  # 给定一个元素
        self.next = None  # 初始设置下一节点为空


class Singly_linked_list:
    """Python实现单链表"""

    def __init__(self):
        self.__head = None  # head设置为私有属性，禁止外部访问
        self.__tail = None

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None

    def length(self):
        """返回链表长度"""
        cur = self.__head  # cur游标，用来移动遍历节点
        count = 0  # count记录节点数量
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def travel_list(self):
        """遍历整个链表，打印每个节点的数据"""
        cur = self.__head
        while cur is not None:
            print(cur.element, end=" ")
            cur = cur.next
        print("\n")

    def insert_head(self, element):
        """头插法：在单链表头部插入一个节点"""
        newest = Node(element)  # 创建一个新节点
        if self.__head is not None:  # 如果初始不为空，就将新节点的"next"指针指向head
            newest.next = self.__head
        self.__head = newest  # 把新节点设置为head

    def insert_tail(self, element):
        """尾插法：在单链表尾部增加一个节点"""
        if self.__head is None:
            self.insert_head(element)  # 如果这是第一个节点，调用insert_head函数
        else:
            cur = self.__head
            while cur.next is not None:  # 遍历到最后一个节点
                cur = cur.next
            cur.next = Node(element)  # 创建新节点并连接到最后

    def insert(self, pos, element):
        """指定位置插入元素"""

        # 如果位置在0或者之前，调用头插法
        if pos < 0:
            self.insert_head(element)
        # 如果位置在原链表长度之后，调用尾插法
        elif pos > self.length() - 1:
            self.insert_tail(element)
        else:
            cur = self.__head
            count = 0
            while count < pos - 1:
                count += 1
                cur = cur.next
            newest = Node(element)
            newest.next = cur.next
            cur.next = newest

    def delete_head(self):
        """删除头结点"""
        cur = self.__head
        if self.__head is not None:
            self.__head = self.__head.next
            cur.next = None
        return cur

    def delete_tail(self):
        """删除尾节点"""
        cur = self.__head
        if self.__head is not None:
            if self.__head.next is None:  # 如果头结点是唯一的节点
                self.__head = None
            else:
                while cur.next.next is not None:
                    cur = cur.next
                cur.next, cur = (None, cur.next)
        return cur

    def remove(self, element):
        """删除指定元素"""
        cur, prev = self.__head, None
        while cur is not None:
            if cur.element == element:
                if cur == self.__head:  # 如果该节点是头结点
                    self.__head = cur.next
                else:
                    prev.next = cur.next
                break
            else:
                prev, cur = cur, cur.next
        return cur.element

    def modify(self, pos, element):
        """修改指定位置的元素"""
        cur = self.__head
        if pos < 0 or pos > self.length():
            return False
        for i in range(pos - 1):
            cur = cur.next
        cur.element = element
        return cur

    def search(self, element):
        """查找节点是否存在"""
        cur = self.__head
        while cur:
            if cur.element == element:
                return True
            else:
                cur = cur.next
        return False

    def reverse_list(self):
        """反转整个链表"""
        cur, prev = self.__head, None
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        self.__head = prev


def main():
    List1 = Singly_linked_list()
    print("链表是否为空", List1.is_empty())

    List1.insert_head(1)
    List1.insert_head(2)
    List1.insert_tail(3)
    List1.insert_tail(4)
    List1.insert_tail(5)

    length_of_list1 = List1.length()
    print("插入节点后，List1 的长度为：", length_of_list1)

    print("遍历并打印整个链表: ")
    List1.travel_list()

    print("反转整个链表: ")
    List1.reverse_list()
    List1.travel_list()

    print("删除头节点: ")
    List1.delete_head()
    List1.travel_list()

    print("删除尾节点: ")
    List1.delete_tail()
    List1.travel_list()

    print("在第二个位置插入5: ")
    List1.insert(1, 5)
    List1.travel_list()

    print("在第-1个位置插入100：")
    List1.insert(-1, 100)
    List1.travel_list()

    print("在第100个位置插入2：")
    List1.insert(100, 2)
    List1.travel_list()

    print("删除元素5：")
    print(List1.remove(5))
    List1.travel_list()

    print("修改第5个位置的元素为7: ")
    List1.modify(5, 7)
    List1.travel_list()

    print("查找元素1:")
    print(List1.search(1))


if __name__ == "__main__":
    main()

