from typing import List, Optional
from utils.base_decorator import CommonLogger

cl = CommonLogger()
a = [0, 1, 2, 3, 4, 5, 6]
idx = a.index(4)

x = a[:4]
y = a[4:-1]
print()


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def __init__(self):
        self.ans = None

    @CommonLogger()
    def buildTree_pre_in(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 第一步: 特殊情况讨论: 树为空. 或者说是递归终止条件
        if not preorder:
            return None
        # 第二步: 前序遍历的第一个就是当前的中间节点.
        root_val = preorder[0]
        root = TreeNode(root_val)
        # 第三步: 找切割点.
        separator_idx = inorder.index(root_val)
        # 第四步: 切割inorder数组. 得到inorder数组的左,右半边.
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]
        # 第五步: 切割preorder数组. 得到preorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟前序数组大小是相同的.
        preorder_left = preorder[1:1 + len(inorder_left)]
        preorder_right = preorder[1 + len(inorder_left):]
        # 第六步: 递归
        root.left = self.buildTree_pre_in(preorder_left, inorder_left)
        root.right = self.buildTree_pre_in(preorder_right, inorder_right)
        return root

    def buildTree_in_post(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 第一步: 特殊情况讨论: 树为空. (递归终止条件)
        if not postorder:
            return None
        # 第二步: 后序遍历的最后一个就是当前的中间节点.
        root_val = postorder[-1]
        root = TreeNode(root_val)
        # 第三步: 找切割点.
        separator_idx = inorder.index(root_val)
        # 第四步: 切割inorder数组. 得到inorder数组的左,右半边.
        inorder_left = inorder[:separator_idx]
        inorder_right = inorder[separator_idx + 1:]
        # 第五步: 切割postorder数组. 得到postorder数组的左,右半边.
        # ⭐️ 重点1: 中序数组大小一定跟后序数组大小是相同的.
        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left): len(postorder) - 1]
        # 第六步: 递归
        root.left = self.buildTree_in_post(inorder_left, postorder_left)
        root.right = self.buildTree_in_post(inorder_right, postorder_right)
        return root

    def buildTree_pre_post(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        # 1.
        if not preorder:
            return None
        # 2.
        root_val = preorder[0]
        root = TreeNode(root_val)

        # 3.
        if len(preorder) > 1:
            left_root_val = preorder[1]
            separator_idx = postorder.index(left_root_val)
        else:
            separator_idx = 0

        # 4.
        postorder_left = postorder[:separator_idx + 1]
        postorder_right = postorder[separator_idx + 1:-1]

        # 5.
        preorder_left = preorder[1:(1 + len(postorder_left))]
        preorder_right = preorder[(1 + len(postorder_left)):]
        # 6.
        root.left = self.buildTree_pre_post(preorder_left, postorder_left)
        root.right = self.buildTree_pre_post(preorder_right, postorder_right)
        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        # nums = [3,2,1,6,0,5] [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
        if not nums:
            return None
        max_value = max(nums)
        root = TreeNode(max_value)
        idx = nums.index(max_value)
        left = nums[:idx]
        right = nums[idx + 1:]
        print('max:{} left:{},right:{}'.format(max_value, left, right))
        root.left = self.constructMaximumBinaryTree(left)
        root.right = self.constructMaximumBinaryTree(right)
        return root

    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        max_value = max(nums)
        root = TreeNode(max_value)
        idx = nums.index(max_value)
        left = nums[:idx]
        right = nums[idx + 1:]
        print('max:{} left:{},right:{}'.format(max_value, left, right))
        root.left = self.constructMaximumBinaryTree(left)
        root.right = self.constructMaximumBinaryTree(right)
        return root

    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        # root1 = [1, 3, 2, 5], root2 = [2, 1, 3, null, 4, null, 7]
        if not root1:
            return root2
        if not root2:
            return root1
        if root1 and root2:
            root1.val = root1.val + root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1

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


if __name__ == '__main__':
    s = Solution()
    inorder = [9, 3, 15, 20, 7]
    postorder = [9, 15, 7, 20, 3]
    res = s.buildTree_in_post(inorder=inorder, postorder=postorder)
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    res = s.buildTree_pre_in(preorder=preorder, inorder=inorder)
    preorder = [1, 2, 4, 5, 3, 6, 7]
    postorder = [4, 5, 2, 6, 7, 3, 1]
    res = s.buildTree_pre_post(preorder=preorder, postorder=postorder)
    from tree.binary_tree_traversal import Solution as ts

    root = ts().create_tree_from_level_order_list(alist=[3, 0, 4, None, 2, None, None, None, None, 1])
    res = s.trimBST(root=root, low=1, high=3)
    print(cl.output)
    print()
    print()
