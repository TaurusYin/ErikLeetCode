class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        '''
        确认递归函数参数以及返回值：返回更新后剪枝后的当前root节点
        '''
        # Base Case
        if not root: return None
        # 单层递归逻辑
        if root.val < key:
            # 若当前root节点小于左界：只考虑其右子树，用于替代更新后的其本身，抛弃其左子树整体
            return self.deleteNode(root.right, key)
        if key < root.val:
            # 若当前root节点大于右界：只考虑其左子树，用于替代更新后的其本身，抛弃其右子树整体
            return self.deleteNode(root.left, key)
        if root.val == key:
            root.left = self.deleteNode(root.left, key)
            root.right = self.deleteNode(root.right, key)
            # 返回更新后的剪枝过的当前节点root
        return root

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