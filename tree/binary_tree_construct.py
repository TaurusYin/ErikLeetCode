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
    """
    给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:创建一个根节点，其值为 nums 中的最大值。
递归地在最大值 左边 的 子数组前缀上 构建左子树。
递归地在最大值 右边 的 子数组后缀上 构建右子树。
链接：https://leetcode.cn/leetbook/read/pu-tong-shu/nsyblm/
    """
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


    """
    case1：如果val大于root.val，则root就是val节点的左子树节点。
    case2：如果val大于非root.val，则非root就是val节点的左子树节点，并且非root节点的原父节点的右子树更新为val节点。
    case3：如果val小于最底层的叶子节点，则val节点就作为该叶子节点的右子树节点。
链接：https://leetcode.cn/problems/maximum-binary-tree-ii/solution/by-muse-77-zkfg/
    """
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        if root.val < val:
            return TreeNode(val, root, None)
        root.right = self.insertIntoMaxTree(root.right, val)
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
