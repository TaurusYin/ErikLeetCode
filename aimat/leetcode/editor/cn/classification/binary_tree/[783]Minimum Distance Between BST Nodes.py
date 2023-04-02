# Given the root of a Binary Search Tree (BST), return the minimum difference 
# between the values of any two different nodes in the tree. 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [4,2,6,1,3]
# Output: 1
#  
# 
#  Example 2: 
#  
#  
# Input: root = [1,0,48,null,null,12,49]
# Output: 1
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [2, 100]. 
#  0 <= Node.val <= 10⁵ 
#  
# 
#  
#  Note: This question is the same as 530: https://leetcode.com/problems/
# minimum-absolute-difference-in-bst/ 
# 
#  Related Topics 树 深度优先搜索 广度优先搜索 二叉搜索树 二叉树 👍 250 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.min_value = float('inf')

    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        def traversal(root: TreeNode):
            if root == None:
                return
            traversal(root.left)  # 左
            if result:
                self.min_value = min(self.min_value, abs(result[-1] - root.val))
            result.append(root.val)  # 中序
            traversal(root.right)  # 右

        result = []
        traversal(root)
        return self.min_value

# leetcode submit region end(Prohibit modification and deletion)
