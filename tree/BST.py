from functools import cache
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

    """
    
    观察例子 [1,2,3,4,5,6,7] 假如2跟6互换了变为 [1,6,3,4,5,2,7]。观察可以得知只有(x1 = 6, y1 = 3) 和(x2 = 5, y2 = 2)的位置是逆序的。我们其实就是要找这两对逆序的位置x1和y2。然后交换其值即可。具体实现如下：
    定义pre，first，second指针。中序遍历树。
    如果pre为空，将其设置为当前root。
    否则pre不空，如果发现逆序对(pre.val >= root.val) 则做如下操作：
    如果first为空记录x1的位置到first，second记录当前的root,第二次发现逆序对会覆盖第一次的值，保证最后second记录的是y2的位置。
    中序遍历后，交换x1的val和y2的val即可。
    链接：https://leetcode.cn/problems/recover-binary-search-tree/solution/99-hui-fu-er-cha-sou-suo-shu-by-jyj407-jajj/
    """

    def recoverTree(self, root: Optional[TreeNode]) -> None:
        self.pre, self.first, self.second = None, None, None

        def inOrder(root):
            if not root:
                return

            inOrder(root.left)
            if not self.pre:
                self.pre = root
            else:
                if self.pre.val >= root.val:
                    if not self.first:
                        self.first = self.pre  # record x1
                    self.second = root  # record y2 eventually
                self.pre = root
            inOrder(root.right)

        inOrder(root)
        self.first.val, self.second.val = self.second.val, self.first.val

    """
    返回 树中任意两不同节点值之间的最小差值 
    """

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
    输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。
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
96. 不同的二叉搜索树
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

"""
要求结点数量为3的二叉搜索树的种数：左子树有0个结点，右子树有2个结点,左子树有1个结点，右子树有1个结点,左子树有2个结点，右子树有0个结点
f(n) = f(i) * f(n-i-1)
"""
@cache
def numTrees(self, n: int) -> int:
    if n <= 1:
        return 1
    num_trees = 0
    for i in range(n):
        left_num_trees = self.numTrees(i)
        right_num_trees = self.numTrees(n - 1 - i)
        num_trees += (left_num_trees * right_num_trees)
    return num_trees

"""
95. 不同的二叉搜索树 II
https://leetcode.cn/problems/unique-binary-search-trees-ii/solution/by-focused-antonellibcp-m77k/
"""
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        return self.generate_tree(1, n) if n else []

    def generate_tree(self, start, end):
        '''
        解题思路：递归分别寻找每层中的每个元素可能的左子树与右子树，然后由当前元素分别与左右子树组成的所有结果，再层层往上返回结果
        :param start: 起始数字
        :param end: 终止数字
        :return: 返回所有符合条件的二叉搜索树
        '''
        if start > end:
            return [None]
        allTrees = []
        for i in range(start, end + 1):  # 枚举可行根节点
            # 获得所有可行的左子树集合
            leftTrees = self.generate_tree(start, i - 1)
            # 获得所有可行的右子树集合
            rightTrees = self.generate_tree(i + 1, end)
            # 从每一层的左子树与右子树集合中各选出一棵，拼接到当前遍历元素的根节点上
            # 其中左子树列表元素的个数，取决于当前层的当前元素可以取的左孩子节点的种类数.如：很明显，当前节点为1时，左孩子只有1种，即为None
            # 同理,其中右子树列表元素的个数，取决于当前层的当前元素可以取的右孩子节点的种类数. 如：很明显，当前节点为1时，右孩子只有2种，即为3->2 or 2->3
            for l in leftTrees:
                for r in rightTrees:
                    currTree = TreeNode(i)
                    currTree.left = l
                    currTree.right = r
                    allTrees.append(currTree)  # 将这一层当前元素所有可能组成的搜索树放到列表中，如果已经是最外层，即表示当前元素所生成的所有的二叉搜索树结果放入list中
        return allTrees

    """
    426. 将二叉搜索树转化为排序的双向链表
    其实就是左右节点指向改变，右节点指向下一个比它大的数，左节点指向比它小的数。又是二叉搜索树，自然想到中序遍历
    将一个 二叉搜索树 就地转化为一个 已排序的双向循环链表.
    """
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        def traversal(root: TreeNode):
            nonlocal prev
            if root == None:
                return
            traversal(root.left)  # 左
            prev.right = root
            root.left = prev
            prev = prev.right
            print(root.val)
            traversal(root.right)  # 右

        if not root:
            return root
        dummy = Node(-1, None, None)
        prev = dummy
        traversal(root)
        prev.right = dummy.right
        dummy.right.left = prev
        print(root)
        return dummy.right

