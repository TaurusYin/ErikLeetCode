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


# https://leetcode.cn/problems/linked-list-cycle-ii/
# 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
"""
我们使用两个指针，\textit{fast}fast 与 \textit{slow}slow。它们起始都位于链表的头部。随后，\textit{slow}slow 指针每次向后移动一个位置，而 \textit{fast}fast 指针向后移动两个位置。如果链表中存在环，则 \textit{fast}fast 指针最终将再次与 \textit{slow}slow 指针在环中相遇。
如下图所示，设链表中环外部分的长度为 aa。\textit{slow}slow 指针进入环后，又走了 bb 的距离与 \textit{fast}fast 相遇。此时，\textit{fast}fast 指针已经走完了环的 nn 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nca+n(b+c)+b=a+(n+1)b+nc。
链接：https://leetcode.cn/problems/linked-list-cycle-ii/solution/huan-xing-lian-biao-ii-by-leetcode-solution/
"""


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
"""
给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。
你应当 保留 两个分区中每个节点的初始相对位置。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/partition-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


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


"""
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
https://leetcode.cn/problems/swap-nodes-in-pairs/
输入：head = [1,2,3,4]
输出：[2,1,4,3]
"""


def swapPairs(self, head: ListNode) -> ListNode:
    dummyHead = ListNode(0)
    dummyHead.next = head
    temp = dummyHead
    while temp.next and temp.next.next:
        node1 = temp.next
        node2 = temp.next.next
        temp.next = node2
        node1.next = node2.next
        node2.next = node1
        temp = node1
    return dummyHead.next


"""
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
https://leetcode.cn/problems/sort-list/solution/sort-list-gui-bing-pai-xu-lian-biao-by-jyd/
1. 分割 cut 环节： 找到当前链表中点 
2. 合并 merge 环节: 将两个排序链表合并，转化为一个排序链表。

"""


def sortList(self, head: ListNode) -> ListNode:
    if not head or not head.next: return head  # termination.
    # cut the LinkedList at the mid index.
    slow, fast = head, head.next
    while fast and fast.next:
        fast, slow = fast.next.next, slow.next
    mid, slow.next = slow.next, None  # save and cut.
    # recursive for cutting.
    left, right = self.sortList(head), self.sortList(mid)
    # merge `left` and `right` linked list and return it.
    h = res = ListNode(0)
    while left and right:
        if left.val < right.val:
            h.next, left = left, left.next
        else:
            h.next, right = right, right.next
        h = h.next
    h.next = left if left else right
    return res.next


# 走一步 走两步
# https://leetcode.cn/problems/middle-of-the-linked-list/solution/by-jyd-aphd/
"""
示例 1：

输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/middle-of-the-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


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


"""
输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]
示例 2：

输入：head = [5], left = 1, right = 1
输出：[5]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-linked-list-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


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


"""
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-nodes-in-k-group
"""


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


"""
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/copy-list-with-random-pointer
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def copyRandomList(self, head: 'Node') -> 'Node':
    Mydic = dict()

    def recursion(node: 'Node') -> 'Node':
        if node is None: return None
        if node in Mydic: return Mydic.get(node)
        root = Node(node.val)
        Mydic[node] = root
        root.next = recursion(node.next)
        root.random = recursion(node.random)
        return root

    return recursion(head)


"""
https://leetcode.cn/problems/reorder-list/
给定一个单链表 L 的头节点 head ，单链表 L 表示为：

L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
O(N)
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reorder-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return

        mid = self.middleNode(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = self.reverseList(l2)
        self.mergeList(l1, l2)

    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            nextTemp = curr.next
            curr.next = prev
            prev = curr
            curr = nextTemp
        return prev

    def mergeList(self, l1: ListNode, l2: ListNode):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp


"""
给定一个奇数位升序，偶数位降序的链表，将其重新排序。

输入: 1->8->3->6->5->4->7->2->NULL
输出: 1->2->3->4->5->6->7->8->NULL

1. 按奇偶位置拆分链表，得1->3->5->7->NULL和8->6->4->2->NULL
2. 反转偶链表，得1->3->5->7->NULL和2->4->6->8->NULL
3. 合并两个有序链表，得1->2->3->4->5->6->7->8->NULL
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def sortOddEvenList(self, head):
        if not head or not head.next:
            return head
        oddList, evenList = self.partition(head)
        evenList = self.reverse(evenList)
        return self.merge(oddList, evenList)

    def partition(self, head: ListNode) -> ListNode:
        evenHead = head.next
        odd, even = head, evenHead
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = None
        return [head, evenHead]

    def reverse(self, head):
        dumpy = ListNode(-1)
        p = head
        while p:
            temp = p.next
            p.next = dumpy.next
            dumpy.next = p
            p = temp
        return dumpy.next

    def merge(self, p, q):
        head = ListNode(-1)
        r = head
        while p and q:
            if p.val <= q.val:
                r.next = p
                p = p.next
            else:
                r.next = q
                q = q.next
            r = r.next
        if p:
            r.next = p
        if q:
            r.next = q
        return head.next


"""
运行条件：链表从头遍历到位，逐位相加
（1）需要保存进位
（2）需要保存结果
结束时：
（1）两个链表只要有一个非空就需要往后进行
（2）🚩如果链表遍历结束，进位不为0，需要把进位项添加在链表后面
链接：https://leetcode.cn/problems/add-two-numbers/solution/si-kao-guo-cheng-pythondai-ma-zhu-yi-by-4fl4i/
"""
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    # 当前指针，结果链表
    result = curr = ListNode()
    # 进位项
    remainder = 0
    # 非空满足循环条件
    while l1 or l2:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0
        total = x + y + remainder
        curr.next = ListNode(total % 10)
        remainder = total // 10
        # 防止某一链表已经为空，空链表.next会报错
        if l1: l1 = l1.next
        if l2: l2 = l2.next
        curr = curr.next

    if remainder: curr.next = ListNode(remainder)
    return result.next

"""
https://leetcode.cn/problems/copy-list-with-random-pointer/
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

"""
def copyRandomList(self, head: 'Node') -> 'Node':
        lookup = {}

        def dfs(head):
            if not head: return None
            if head in lookup: return lookup[head]
            clone = Node(head.val, None, None)
            lookup[head] = clone
            clone.next, clone.random = dfs(head.next), dfs(head.random)
            return clone

        return dfs(head)

"""
https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/solution/er-cha-shu-zhan-kai-wei-lian-biao-by-leetcode-solu/
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
"""
def flatten(self, root: TreeNode) -> None:
    preorderList = list()

    def preorderTraversal(root: TreeNode):
        if root:
            preorderList.append(root)
            preorderTraversal(root.left)
            preorderTraversal(root.right)

    preorderTraversal(root)
    size = len(preorderList)
    for i in range(1, size):
        prev, curr = preorderList[i - 1], preorderList[i]
        prev.left = None
        prev.right = curr



