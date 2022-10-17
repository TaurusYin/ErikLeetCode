from typing import Optional

from tree.binary_tree_construct import TreeNode


class Solution:
    """
    https://leetcode.cn/problems/delete-node-in-a-bst/
    """

    def _deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        def traversal(root):
            if not root:
                return root
            if root.val > key:
                root.left = traversal(root.left)
            elif root.val < key:
                root.right = traversal(root.right)
            elif root.val == key:
                if not root.left and not root.right:
                    return None
                if not root.left and root.right:
                    return root.right
                if not root.right and root.left:
                    return root.left
                if root.left and root.right:
                    temp = root.left
                    while (temp.right != None):
                        temp = temp.right
                    temp.right = root.right
                    return root.left
            return root

        return traversal(root)

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def traversal(root, parent):
            if not root:
                return
            if root.val > val:
                if not root.left:
                    root.left = TreeNode(val)
                    return
                traversal(root.left, root)
            else:
                if not root.right:
                    root.right = TreeNode(val)
                    return
                traversal(root.right, root)
            return

        if not root:
            return TreeNode(val)
        traversal(root, root)
        return root

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # 输入：root = [4,2,7,1,3], val = 2  [2,1,3] ?  什么时候设置全局变量不返回
        # https://leetcode.cn/problems/search-in-a-binary-search-tree/
        if not root:
            return None
        if val == root.val:
            self.ans = root
            return root
        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    # https://leetcode.cn/problems/validate-binary-search-tree/
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)  # 左
            if result and root.val <= result[-1]:
                print('{},{}'.format(root.val, result[-1]))
                self.ans = False
            result.append(root.val)  # 中序
            traversal(root.right)  # 右

        result = []
        res = traversal(root)
        print(result)
        return self.ans

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)  # 左
            if result and root.val:
                diff = abs(root.val - result[-1])
                if diff < self.ans:
                    self.ans = diff
            result.append(root.val)  # 中序

            traversal(root.right)  # 右

        result = []
        traversal(root)
        return self.ans

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        def traversal(root: TreeNode):
            nonlocal cnt, max_count, ans, pre
            if root == None:
                return
            traversal(root.left)  # 左

            if root.val == pre:  # 连续
                cnt += 1
            else:  # 不连续
                cnt = 1
            if cnt > max_count:  # 新众数
                max_count = cnt
                ans = [root.val]
            elif cnt == max_count:  # 同等众数
                ans.append(root.val)
            pre = root.val

            traversal(root.right)  # 右

        cnt, max_count, ans, pre = 1, 0, [], None
        traversal(root)
        return ans

    @cl
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        '''
        确认递归函数参数以及返回值：返回更新后剪枝后的当前root节点
        '''
        # Base Case
        if root is None or root.val is None:
            return
        # 单层递归逻辑
        if root.val < low:
            # 若当前root节点小于左界：只考虑其右子树，用于替代更新后的其本身，抛弃其左子树整体
            return self.trimBST(root.right, low, high)
        if high < root.val:
            # 若当前root节点大于右界：只考虑其左子树，用于替代更新后的其本身，抛弃其右子树整体
            return self.trimBST(root.left, low, high)
        if low <= root.val <= high:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
            # 返回更新后的剪枝过的当前节点root
        return root

    """
    二叉搜索树后序遍历
    后序遍历倒序： [ 根节点 | 右子树 | 左子树 ] 。类似 先序遍历的镜像 ，即先序遍历为 “根、左、右” 的顺序，而后序遍历的倒序为 “根、右、左” 顺序。
    单调栈
    时间复杂度 O(N)O(N) ： 遍历 postorderpostorder 所有节点，各节点均入栈 / 出栈一次，使用 O(N)O(N) 时间。
    https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/solution/mian-shi-ti-33-er-cha-sou-suo-shu-de-hou-xu-bian-6/
    """

    def verifyPostorder(self, postorder: [int]) -> bool:
        stack, root = [], float("+inf")
        for i in range(len(postorder) - 1, -1, -1):
            if postorder[i] > root: return False
            while (stack and postorder[i] < stack[-1]):
                root = stack.pop()
            stack.append(postorder[i])
        return True


"""
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
"""


def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    if not nums:
        return None
    if len(nums) == 1:
        return TreeNode(nums[0])
    mid = int(len(nums) / 2)
    left_node = self.sortedArrayToBST(nums[:mid])
    right_node = self.sortedArrayToBST(nums[mid + 1:])
    root = TreeNode(nums[mid])
    root.left = left_node
    root.right = right_node
    return root


"""
    输入: head = [-10,-3,0,5,9]
输出: [0,-3,9,-10,null,5]
解释: 一个可能的答案是[0，-3,9，-10,null,5]，它表示所示的高度平衡的二叉搜索树。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/convert-sorted-list-to-binary-search-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """


def sortedListToBST(self, head: ListNode) -> TreeNode:
    def getLength(head: ListNode) -> int:
        ret = 0
        while head:
            ret += 1
            head = head.next
        return ret

    def buildTree(left: int, right: int) -> TreeNode:
        if left > right:
            return None
        mid = (left + right + 1) // 2
        root = TreeNode()
        root.left = buildTree(left, mid - 1)
        nonlocal head
        root.val = head.val
        head = head.next
        root.right = buildTree(mid + 1, right)
        return root

    length = getLength(head)
    return buildTree(0, length - 1)


"""
作者：fuxuemingzhu
链接：https://leetcode.cn/problems/binary-search-tree-iterator/solution/fu-xue-ming-zhu-dan-diao-zhan-die-dai-la-dkrm/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
"""


class BSTIterator(object):

    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        cur = self.stack.pop()
        node = cur.right
        while node:
            self.stack.append(node)
            node = node.left
        return cur.val

    def hasNext(self):
        return len(self.stack) > 0


"""
长度为 nn 的序列能构成的不同二叉搜索树的个数
https://leetcode.cn/problems/unique-binary-search-trees/solution/bu-tong-de-er-cha-sou-suo-shu-by-leetcode-solution/
以 ii 为根、序列长度为 nn 的不同二叉搜索树个数 (1 \leq i \leq n)(1≤i≤n)
"""


def numTrees(self, n):
    """
    :type n: int
    :rtype: int
    """
    G = [0] * (n + 1)
    G[0], G[1] = 1, 1

    for i in range(2, n + 1):
        for j in range(1, i + 1):
            G[i] += G[j - 1] * G[i - j]

    return G[n]
