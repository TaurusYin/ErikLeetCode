from typing import List


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def hasCycle(self, head: ListNode) -> bool:
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next

    return True


def detectCycle(self, head):
    fast, slow = head, head
    while True:
        if not (fast and fast.next): return
        fast, slow = fast.next.next, slow.next
        if fast == slow: break
    fast = head
    while fast != slow:
        fast, slow = fast.next, slow.next
    return fast


# https://leetcode.cn/problems/intersection-of-two-linked-lists/solution/
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    A, B = headA, headB
    while A != B:
        A = A.next if A else headB
        B = B.next if B else headA
    return A


# https://leetcode.cn/problems/remove-nth-node-from-end-of-list/solution/by-jam007-gdxa/
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    dummy = ListNode(-1)
    dummy.next, slow, fast = head, dummy, head
    # 快指针先移动n步
    for _ in range(n):
        fast = fast.next

    # 快慢指针同时移动， 快指针移动到结尾结束， 慢指针当前在倒数第n-1个节点
    while fast:
        fast, slow = fast.next, slow.next

    # 跳过倒数第n个节点
    slow.next = slow.next.next
    return dummy.next


# https://leetcode.cn/problems/merge-two-sorted-lists/solution/he-bing-liang-ge-you-xu-lian-biao-by-leetcode-solu/
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    prehead = ListNode(-1)

    prev = prehead
    while l1 and l2:
        if l1.val <= l2.val:
            prev.next = l1
            l1 = l1.next
        else:
            prev.next = l2
            l2 = l2.next
        prev = prev.next

    # 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
    prev.next = l1 if l1 is not None else l2

    return prehead.next


def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    if not lists: return None
    res = None  # 设置初始结果为空
    for listi in lists:  # 逐个遍历 两两合并
        res = self.mergeTwoLists(res, listi)
    return res


# 两两合并 O(NK)
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)  # 构造虚节点
    move = dummy  # 设置移动节点等于虚节点
    while l1 and l2:  # 都不空时
        if l1.val < l2.val:
            move.next = l1  # 移动节点指向数小的链表
            l1 = l1.next
        else:
            move.next = l2
            l2 = l2.next
        move = move.next
    move.next = l1 if l1 else l2  # 连接后续非空链表
    return dummy.next  # 虚节点仍在开头


# 归并
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    if not lists: return None
    n = len(lists)  # 记录子链表数量
    return self.mergeSort(lists, 0, n - 1)  # 调用归并排序函数


def mergeSort(self, lists: List[ListNode], l: int, r: int) -> ListNode:
    if l == r:
        return lists[l]
    m = (l + r) // 2
    L = self.mergeSort(lists, l, m)  # 循环的递归部分
    R = self.mergeSort(lists, m + 1, r)
    return self.mergeTwoLists(L, R)  # 调用两链表合并函数


def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)  # 构造虚节点
    move = dummy  # 设置移动节点等于虚节点
    while l1 and l2:  # 都不空时
        if l1.val < l2.val:
            move.next = l1  # 移动节点指向数小的链表
            l1 = l1.next
        else:
            move.next = l2
            l2 = l2.next
        move = move.next
    move.next = l1 if l1 else l2  # 连接后续非空链表
    return dummy.next  # 虚节点仍在开头


# heap O(n∗log(k))
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    import heapq  # 调用堆
    minHeap = []
    for listi in lists:
        while listi:
            heapq.heappush(minHeap, listi.val)  # 把listi中的数据逐个加到堆中
            listi = listi.next
    dummy = ListNode(0)  # 构造虚节点
    p = dummy
    while minHeap:
        p.next = ListNode(heapq.heappop(minHeap))  # 依次弹出最小堆的数据
        p = p.next
    return dummy.next

# https://leetcode.cn/problems/partition-list/solution/python3-shuang-lian-biao-he-bing-by-qzxj-p0bs/
def partition(self, head: ListNode, x: int) -> ListNode:
    p, q = left, right = ListNode(), ListNode()
    while head:
        if head.val < x:
            left.next = head
            left = left.next
        else:
            right.next = head
            right = right.next
        head = head.next

    right.next = None
    left.next = q.next

    return p.next

# 走一步 走两步
# https://leetcode.cn/problems/middle-of-the-linked-list/solution/by-jyd-aphd/
def middleNode(self, head: ListNode) -> ListNode:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow

# https://leetcode.cn/problems/reverse-linked-list/solution/
# https://leetcode.cn/problems/reverse-linked-list/solution/yi-bu-yi-bu-jiao-ni-ru-he-yong-di-gui-si-67c3/
def reverseListIteration(self, head: ListNode) -> ListNode:
    cur = head
    prev = None
    while cur:
        tmp = cur.next
        cur.next = prev
        prev = cur
        cur = tmp
    return prev

# https://leetcode.cn/problems/reverse-linked-list-ii/solution/fan-zhuan-lian-biao-ii-by-leetcode-solut-teyq/
def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        # 设置 dummyNode 是这一类问题的一般做法
        dummy_node = ListNode(-1)
        dummy_node.next = head
        pre = dummy_node
        for _ in range(left - 1):
            pre = pre.next

        cur = pre.next
        for _ in range(right - left):
            next = cur.next
            cur.next = next.next
            next.next = pre.next
            pre.next = next
        return dummy_node.next


def reverse(self, head: ListNode, tail: ListNode):
    prev = tail.next
    p = head
    while prev != tail:
        nex = p.next
        p.next = prev
        prev = p
        p = nex
    return tail, head

# https://leetcode.cn/problems/reverse-nodes-in-k-group/solution/k-ge-yi-zu-fan-zhuan-lian-biao-by-leetcode-solutio/
def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    hair = ListNode(0)
    hair.next = head
    pre = hair

    while head:
        tail = pre
        # 查看剩余部分长度是否大于等于 k
        for i in range(k):
            tail = tail.next
            if not tail:
                return hair.next
        nex = tail.next
        head, tail = self.reverse(head, tail)
        # 把子链表重新接回原链表
        pre.next = head
        tail.next = nex
        pre = tail
        head = tail.next

    return hair.next


