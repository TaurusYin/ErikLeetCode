import heapq


# Merge function merge two arrays
# of different or same length
# if n = max(n1, n2)
# time complexity of merge is (o(n log(n)))
# O(N Log k)
# Function for meging k arrays
def mergeK(arr, k):
    res = []
    # Declaring min heap
    h = []
    # Inserting the first elements of each row
    for i in range(len(arr)):
        heapq.heappush(h, (arr[i][0], i, 0))
    # Loop to merge all the arrays
    while h:
        # ap stores the row number,
        # vp stores the column number
        val, ap, vp = heapq.heappop(h)
        res.append(val)
        if vp + 1 < len(arr[ap]):
            heapq.heappush(h, (arr[ap][vp + 1], ap, vp + 1))
    return res

arr =[[2, 6, 12 ],
          [ 1, 9 ],
          [23, 34, 90, 2000 ]]
k = 3
l = mergeK(arr, k)





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



# 双指针-将2个数组元素合并到新的数组中去 O(n1 + n2)
def mergeArrays(arr1, arr2):
    n1, n2 = len(arr1), len(arr2)
    arr3 = [None] * (n1 + n2)
    i, j, k = 0, 0, 0
    # Traverse both array
    while i < n1 and j < n2:
        # Check if current element of first array is smaller than current element of second array. If yes, store first array element and increment first array index. Otherwise do same with second array
        if arr1[i] < arr2[j]:
            arr3[k] = arr1[i]
            k = k + 1
            i = i + 1
        else:
            arr3[k] = arr2[j]
            k = k + 1
            j = j + 1
    # Store remaining elements of first array
    while i < n1:
        arr3[k] = arr1[i]
        k = k + 1
        i = i + 1
    # Store remaining elements of second array
    while j < n2:
        arr3[k] = arr2[j];
        k = k + 1
        j = j + 1
    print("Array after merging")
    for i in range(n1 + n2):
        print(str(arr3[i]), end=" ")
    return arr3
