"""
关键点：结构同满二叉树。

1.对于满二叉树，从根节点开始可以对节点编号1，2，...，某节点p的左子节点的序号为2p，右子节点的序号为2p+1；
2.若令根节点的序号p为0，且左子节点的序号为2p，右子节点的序号为2p+1，则每层节点中，节点的序号即代表节点在这一层中的位置索引
链接：https://leetcode.cn/problems/maximum-width-of-binary-tree/solution/662-er-cha-shu-zui-da-kuan-du-by-jue-qia-djxn/
O(N)
"""
from tree.binary_tree_construct import TreeNode


def widthOfBinaryTree(self, root: TreeNode) -> int:
    # bfs，队列中记录每个节点的root，pos，按层更新max_width
    if not root:
        return 0
    max_width = 0
    queue = [(root, 0)]
    while queue:
        width = queue[-1][1] - queue[0][1] + 1
        if max_width < width:
            max_width = width
        for _ in range(len(queue)):
            node, pos = queue.pop(0)
            if node.left:
                queue.append((node.left, pos * 2))
            if node.right:
                queue.append((node.right, pos * 2 + 1))
    return max_width


# https://leetcode.cn/problems/binary-tree-right-side-view/
def rightSideView(self, root) -> List[int]:
    def traversal(root, depth):
        if root.val == None: return []
        if len(final_res) == depth: final_res.append(None)
        final_res[depth] = root.val
        if root.left: traversal(root.left, depth + 1)
        if root.right: traversal(root.right, depth + 1)

    final_res = []
    traversal(root, 0)
    return final_res


def averageOfLevels(self, root) -> List[float]:
    import numpy as np

    def traversal(root, depth):
        if root.val == None:
            return []
        if len(res) == depth:
            res.append([])  # start the current depth
        res[depth].append(root.val)  # fulfil the current depth
        if root.left: traversal(root.left, depth + 1)  # process child nodes for the next depth
        if root.right: traversal(root.right, depth + 1)

    res = []
    traversal(root, 0)
    final_res = list(map(lambda x: np.mean(x), res))
    return final_res


def isSymmetric_recursion(self, root) -> bool:
    def traversal(root, depth):
        if len(self.res) == depth:
            print()
            self.res.append([])  # start the current depth
        if root == None:
            self.res[depth].append('null')
            return []
        else:
            self.res[depth].append(root.val)  # fulfil the current depth
        traversal(root.left, depth + 1)  # process child nodes for the next depth
        traversal(root.right, depth + 1)
        return

    traversal(root, 0)
    flag = True
    for item in self.res:
        if item != item[::-1]:
            flag = False
            break
    return flag


def isSymmetric_iteration(self, root) -> bool:
    result = []
    if root.val == None:
        return result
    from collections import deque
    que = deque([root])
    while que:
        size = len(que)
        st = []
        for _ in range(size):
            cur = que.popleft()
            if cur:
                st.append(cur.val)
                que.append(cur.left)
                que.append(cur.right)
            else:
                st.append(None)

        result.append(st)
        if st != st[::-1]:
            return False
        print(st)
    return True


def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def traversal(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val) and traversal(left.left, right.right) and traversal(left.right, right.left)

    return traversal(root.left, root.right)


def sameStructure(self, A, B):
    # 若把B搜索了一个遍，则返回True
    if not B:
        return True
    # 若B还没搜索完，但是A以及到头了，返回False
    if not A:
        return False
    # 若A,B均不为None,若两节点值不相等，则返回False
    if A.val != B.val:
        return False
    # 递归寻找左子树和右子树是否相同
    return self.sameStructure(A.left, B.left) and \
           self.sameStructure(A.right, B.right)


def isSameTree(self, s, t):
    if not s and not t:
        return True
    if not s or not t:
        return False
    return s.val == t.val and self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)


def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
    if not root and not subRoot:
        return True
    if not root or not subRoot:
        return False
    return self.isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)


"""
951. 翻转等价二叉树 二叉树是否是翻转 等价(翻转操作) 的函数
https://leetcode.cn/problems/flip-equivalent-binary-trees/solution/fan-zhuan-deng-jie-er-cha-shu-by-leetcode/
time complexity: O(min(N1,N2))  树大小
space complexity : O(min(H1,H2)) 树高度
"""


def isflipEquiv(self, root1, root2):
    if root1 is root2:
        return True
    if not root1 or not root2 or root1.val != root2.val:
        return False

    return (self.isflipEquiv(root1.left, root2.left) and
            self.isflipEquiv(root1.right, root2.right) or
            self.isflipEquiv(root1.left, root2.right) and
            self.isflipEquiv(root1.right, root2.left))


"""
输入：head = [4,2,8], root = [1,4,4,null,2,2,null,1,null,6,8,null,null,null,null,1,3]
输出：true
是不是链表的子路径
链接：https://leetcode.cn/problems/linked-list-in-binary-tree
"""


def dfs(self, head: ListNode, rt: TreeNode) -> bool:
    if not head:
        return True
    if not rt:
        return False
    if rt.val != head.val:
        return False
    return self.dfs(head.next, rt.left) or self.dfs(head.next, rt.right)


def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
    if not root:
        return False
    return self.dfs(head, root) or self.isSubPath(head, root.left) or self.isSubPath(head, root.right)


def minDepth(self, root: TreeNode) -> int:
    if not root:
        return 0
    if not root.left:
        return self.minDepth(root.right) + 1
    if not root.right:
        return self.minDepth(root.left) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1


def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


def isBalanced(self, root: TreeNode) -> bool:
    def maxDepth(root):
        if not root:
            return 0
        return max(maxDepth(root.left), maxDepth(root.right)) + 1

    if not root:
        return True
    else:
        return abs(maxDepth(root.left) - maxDepth(root.right)) <= 1 and self.isBalanced(
            root.left) and self.isBalanced(root.right)


"""
655. 输出二叉树
计算最大深度，分配位置
输入：root = [1,2,3,null,4]
输出：
[["","","","1","","",""],
 ["","2","","","","3",""],
 ["","","4","","","",""]]
链接：https://leetcode.cn/problems/print-binary-tree
"""


def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
    def cal_depth(node):
        if node.left:
            l_depth = cal_depth(node.left) + 1
        else:
            l_depth = 0
        if node.right:
            r_depth = cal_depth(node.right) + 1
        else:
            r_depth = 0
        return max(l_depth, r_depth)

    height = cal_depth(root)
    m = height + 1
    n = 2 ** m - 1
    ans = [[''] * n for _ in range(m)]

    def dfs(node, r, c):
        ans[r][c] = str(node.val)
        if node.left:
            dfs(node.left, r + 1, c - 2 ** (height - r - 1))
        if node.right:
            dfs(node.right, r + 1, c + 2 ** (height - r - 1))

    dfs(root, 0, (n - 1) // 2)
    return ans


"""
https://leetcode.cn/problems/check-completeness-of-a-binary-tree/solution/er-cha-shu-de-wan-quan-xing-jian-yan-by-leetcode/
是否完全二叉树
输入：root = [1,2,3,4,5,6]
输出：true
解释：最后一层前的每一层都是满的（即，结点值为 {1} 和 {2,3} 的两层），且最后一层中的所有结点（{4,5,6}）都尽可能地向左。
"""


def isCompleteTree(self, root):
    nodes = [(root, 1)]
    i = 0
    while i < len(nodes):
        node, v = nodes[i]
        i += 1
        if node:
            nodes.append((node.left, 2 * v))
            nodes.append((node.right, 2 * v + 1))

    return nodes[-1][1] == len(nodes)


def countNodes(self, root: TreeNode) -> int:
    if not root:
        return 0
    return 1 + self.countNodes(root.left) + self.countNodes(root.right)


def _diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    def traversal(root: Optional[TreeNode]):
        if not root:
            return 0
        left_depth = traversal(root.left)
        right_depth = traversal(root.right)
        current_depth = max(left_depth, right_depth)

        my_depth = left_depth + right_depth
        self.max_depth = max(my_depth, self.max_depth)
        return current_depth + 1

    traversal(root)
    return self.max_depth


def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root:
        return None
    if root == p or root == q:
        return root
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    if left:
        return left
    return right


def lowestCommonAncestor_BST(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if not root:
        return None
    if root.val == p or root.val == q:
        return root
    if max(p.val, q.val) < root.val:
        return self.lowestCommonAncestor_BST(root.left, p, q)
    elif min(p.val, q.val) > root.val:
        return self.lowestCommonAncestor_BST(root.right, p, q)
    else:
        # p < root < q  一定是最近公共祖先
        # left = self.lowestCommonAncestor(root.left, p, q)
        # right = self.lowestCommonAncestor(root.right, p, q)
        return root



"""
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
https://leetcode.cn/problems/sum-root-to-leaf-numbers/solution/qiu-gen-dao-xie-zi-jie-dian-shu-zi-zhi-he-by-leetc/
"""


def sumNumbers(self, root: TreeNode) -> int:
    def dfs(root: TreeNode, prevTotal: int) -> int:
        if not root:
            return 0
        total = prevTotal * 10 + root.val
        if not root.left and not root.right:
            return total
        else:
            return dfs(root.left, total) + dfs(root.right, total)

    return dfs(root, 0)


"""
https://leetcode.cn/problems/house-robber-iii/solution/python3-hou-xu-bian-li-by-accsrd-37t9/
输入: root = [3,2,3,null,3,null,1]
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
"""


def rob(self, root: TreeNode) -> int:
    def DFS(root):
        if not root:
            return 0, 0
        # 后序遍历
        leftchild_steal, leftchild_nosteal = DFS(root.left)
        rightchild_steal, rightchild_nosteal = DFS(root.right)

        # 偷当前node，则最大收益为【投当前节点+不偷左右子树】
        steal = root.val + leftchild_nosteal + rightchild_nosteal
        # 不偷当前node，则可以偷左右子树
        nosteal = max(leftchild_steal, leftchild_nosteal) + max(rightchild_steal, rightchild_nosteal)
        return steal, nosteal

    return max(DFS(root))
