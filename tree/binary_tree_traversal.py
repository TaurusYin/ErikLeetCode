from typing import List, Optional
from utils.base_decorator import CommonLogger
a = [1,3,4]
b = a
b.append('x')
# 1->2->3->4->5
# 5->4->3->2->1

inorder = [1,3,2]
x = {idx: val for idx, val in enumerate(inorder)}
print(x)

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.val == other
        else:
            return False

    '''
    def __repr__(self):
        return self.val
    '''


class Solution:
    def __init__(self):
        self.res = []
        return

    def create_tree_from_level_order_list(self, alist):
        """
        :param alist: alist = [1, 0, None, 4, 5, None, 6, 7]
        :return:
        """
        tree_node_list = [TreeNode(a) for a in alist]
        parent_num = len(alist) // 2
        for i in range(0, parent_num):
            left_index = 2 * i + 1
            right_index = 2 * i + 2
            if left_index < len(alist):
                if tree_node_list[left_index]:
                    tree_node_list[i].left = tree_node_list[left_index]
            if right_index < len(alist):
                if tree_node_list[right_index]:
                    tree_node_list[i].right = tree_node_list[right_index]
            print('root: {} left: {} right: {}'.format(i, 2 * i + 1, 2 * i + 2))
        return tree_node_list[0]

    def preorder_traversal(self, root: TreeNode) -> List[int]:

        @CommonLogger()
        def traversal(root: TreeNode):
            if root == None:
                return
            result.append(root.val)  # 前序
            traversal(root.left)  # 左
            traversal(root.right)  # 右

        result = []
        traversal(root)
        return result

    def inorder_traversal(self, root: TreeNode) -> List[int]:

        @CommonLogger()
        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)  # 左
            result.append(root.val)  # 中序
            traversal(root.right)  # 右

        result = []
        traversal(root)
        return result

    def postorder_traversal(self, root: TreeNode) -> List[int]:

        @CommonLogger()
        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)  # 左
            traversal(root.right)  # 右
            result.append(root.val)  # 后序

        result = []
        traversal(root)
        return result

    # pre and post
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def traversal(root, path):
            path.append(str(root.val))
            if root.left is None and root.right is None:
                print('leaf:{},path:{}'.format(root.val, path))
                res.append('->'.join(path))
                return
            if root.left:
                traversal(root.left, path)
                path.pop()
            if root.right:
                traversal(root.right, path)
                path.pop()

        res = []
        traversal(root, [])
        return res

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        def traversal(root: TreeNode, tag):
            if root.left == None and root.right == None:
                print('leaf:{}, tag:{}'.format(root.val, tag))
                if tag == 'left':
                    self.left_sum += root.val
                return
            if root.left:
                traversal(root.left, 'left')  # 左
            if root.right:
                traversal(root.right, 'right')  # 右

        res = 0
        traversal(root, '')
        return self.left_sum

    def preorder_traversal_iter(self, root):
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right:  # 右
                    st.append(node.right)
                if node.left:  # 左
                    st.append(node.left)
                st.append(node)  # 中
                st.append(None)  # 堆栈反向对应递归的推出条件标记
            else:
                node = st.pop()
                result.append(node.val)
        return result

    def inorder_traversal_iter(self, root):
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                if node.right:  # 添加右节点（空节点不入栈）
                    st.append(node.right)

                st.append(node)  # 添加中节点
                st.append(None)  # 中节点访问过，但是还没有处理，加入空节点做为标记。

                if node.left:  # 添加左节点（空节点不入栈）
                    st.append(node.left)
            else:  # 只有遇到空节点的时候，才将下一个节点放进结果集
                node = st.pop()  # 重新取出栈中元素
                result.append(node.val)  # 加入到结果集
        return result

    def postorder_traversal_iter(self, root):
        result = []
        st = []
        if root:
            st.append(root)
        while st:
            node = st.pop()
            if node != None:
                st.append(node)  # 中
                st.append(None)

                if node.right:  # 右
                    st.append(node.right)
                if node.left:  # 左
                    st.append(node.left)
            else:
                node = st.pop()
                result.append(node.val)
        return result

    def iter_template(self, root):
        result = []
        st = []
        while st:
            node = st.pop()
            if node != None:
                """
                left / right / in
                if node.right:
                    st.append(node.right)
                if node.left:
                    st.append(node.left)
                st.append(node)
                st.append(None)
                """
            else:
                node = st.pop()
                result.append(node.val)

    def level_order(self, root: TreeNode) -> List[List[int]]:
        # [3, 9, 20, None, None, 15, 7]
        @CommonLogger()
        def traversal(root, depth):
            if root.val==None: return []
            if len(res) == depth:
                res.append([])  # start the current depth
            res[depth].append(root.val)  # fulfil the current depth
            if root.left: traversal(root.left, depth + 1)  # process child nodes for the next depth
            if root.right: traversal(root.right, depth + 1)

        res = []
        traversal(root, 0)
        return res

    def level_order_complete(self, root:TreeNode):
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

    def level_order_iter(self, root: TreeNode) -> List[List[int]]:
        result = []
        if root.val ==None:
            return result
        from collections import deque
        que = deque([root])
        while que:
            size = len(que)
            st = []
            for _ in range(size):
                cur = que.popleft()
                st.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            result.append(st)

        return result

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        from collections import deque
        res = deque()

        def traversal(root, depth):
            print('root.val:{}'.format(root.val))
            if root.val==None: return
            if len(res) == depth: res.appendleft([])  # start the current depth
            res[-(depth + 1)].append(root.val)
            if root.left: traversal(root.left, depth + 1)
            if root.right: traversal(root.right, depth + 1)

        traversal(root, 0)
        res = list(res)
        return res

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

    def levelOrderChildren(self, root: 'Node') -> List[List[int]]:
        def traversal(root, depth):
            if root == None: return []
            if len(res) == depth: res.append([])
            res[depth].append(root.val)
            if root.children:
                for child in root.children:
                    traversal(child, depth + 1)
            else:
                return []
        res = []
        traversal(root, 0)
        return res


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


if __name__ == '__main__':
    alist = []
    alist = [1, 0, None, 4, 5, None, 6, 7]
    alist = [5, 4, 6, 1, 2, 7, 8]
    alist = [1, 2, 3, 4, 5, None, 6, None, None, 7, 8]

    s = Solution()
    target_bitree = s.create_tree_from_level_order_list(alist)
    pre_res = s.preorder_traversal(target_bitree)
    in_res = s.inorder_traversal(target_bitree)
    post_res = s.postorder_traversal(target_bitree)
    pre_res = s.preorder_traversal_iter(target_bitree)
    alist = [3, 9, 20, None, None, 15, 7]
    target_bitree = s.create_tree_from_level_order_list(alist)
    level_order_res = s.level_order(target_bitree)
    level_order_res = s.level_order_iter(target_bitree)
    alist = [3, 9, 20, None, None, 15, 7]
    target_bitree = s.create_tree_from_level_order_list(alist)
    level_order_res = s.levelOrderBottom(target_bitree)
    alist = [1, 2, 3, 4]
    target_bitree = s.create_tree_from_level_order_list(alist)
    level_order_res = s.rightSideView(target_bitree)
    alist = [0]
    target_bitree = s.create_tree_from_level_order_list(alist)
    level_order_res = s.averageOfLevels(target_bitree)
    alist = [2,3,3,4,None,None,4,None,5,5,None,None,6,6,None,7,8,8,7,9,0,0,1,1,0,0,9]
    target_bitree = s.create_tree_from_level_order_list(alist)
    isSymmetric_res = s.isSymmetric_recursion(target_bitree)
    alist = [1,2,2,None,3,None,3]
    target_bitree = s.create_tree_from_level_order_list(alist)
    isSymmetric_res = s.isSymmetric(target_bitree)

    print()
