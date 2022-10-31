import collections
from typing import Optional

from tree.binary_tree_construct import TreeNode

"""
112. 路径总和
https://leetcode.cn/problems/path-sum/
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
解释：等于目标和的根节点到叶节点路径如上图所示。 存不存在
"""


def hasPathSum(self, root: TreeNode, sum: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:
        return sum == root.val
    return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    def traversal(root, path):
        nonlocal ans
        if not root:
            return
        path.append(root.val)
        if root.left is None and root.right is None:
            print('leaf:{},path:{}'.format(root.val, path))
            if sum(path) == targetSum:
                ans = True
            return
        if root.left:
            traversal(root.left, path)
            path.pop()
        if root.right:
            traversal(root.right, path)
            path.pop()

    res, ans = [], False
    traversal(root, [])
    return ans


"""
https://leetcode.cn/problems/path-sum-ii/
返回路径
给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
叶子节点 是指没有子节点的节点。
"""


def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
    def dfs(root, sum_value, res, path):
        if not root:  # 空节点，不做处理
            return
        if not root.left and not root.right:  # 叶子节点
            if sum_value == root.val:  # 剩余的「路径和」恰好等于叶子节点值
                res.append(path + [root.val])  # 把该路径放入结果中
                print(path + [root.val])

        dfs(root.left, sum_value - root.val, res, path + [root.val])  # 左子树
        dfs(root.right, sum_value - root.val, res, path + [root.val])  # 右子树

    res = []
    dfs(root, targetSum, res, [])
    return res


def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    def traversal(root, path):
        nonlocal ans
        if not root:
            return
        path.append(root.val)
        if root.left is None and root.right is None:
            print('leaf:{},path:{}'.format(root.val, path))
            if sum(path) == targetSum:
                ans = True
                res.append(path.copy())
        if root.left:
            traversal(root.left, path)
            path.pop()
        if root.right:
            traversal(root.right, path)
            path.pop()

    res, ans = [], False
    traversal(root, [])
    return res


"""
437. 路径总和 III (任意路径和)
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
链接：https://leetcode.cn/problems/path-sum-iii
"""

# O(N*N)
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
    def dfs(root, sum_value):
        count = 0
        if not root:
            return 0
        if sum_value == root.val:
            count += 1
        count_l = dfs(root.left, sum_value - root.val)
        count_r = dfs(root.right, sum_value - root.val)
        return count + count_l + count_r

    if not root:
        return 0
    return dfs(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)

# O(N)
def pathSum(self, root: TreeNode, targetSum: int) -> int:
    prefix = collections.defaultdict(int)
    prefix[0] = 1

    def dfs(root, curr):
        if not root:
            return 0

        ret = 0
        curr += root.val
        ret += prefix[curr - targetSum]
        prefix[curr] += 1
        ret += dfs(root.left, curr)
        ret += dfs(root.right, curr)
        prefix[curr] -= 1

        return ret

    return dfs(root, 0)

"""
666. 路径总和 IV
输入: nums = [113, 215, 221]
输出: 12
解释: 列表所表示的树如上所示。 给定一个包含三位整数的 升序 数组 nums ，表示一棵深度小于 5 的二叉树，
请你返回 从根到所有叶子结点的路径之和 
路径和 = (3 + 5) + (3 + 1) = 12.
https://leetcode.cn/problems/path-sum-iv/solution/lu-jing-he-iv-by-leetcode/
"""
class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

class Solution(object):
    def pathSum(self, nums):
        self.ans = 0
        root = Node(nums[0] % 10)
        # 根据二叉树位置构造二叉树
        for x in nums[1:]:
            depth, pos, val = int(x/100), x/10 % 10, x % 10
            pos -= 1
            cur = root
            for d in range(depth - 2, -1, -1):
                if pos < 2**d:
                    cur.left = cur = cur.left or Node(val)
                else:
                    cur.right = cur = cur.right or Node(val)

                pos %= 2**d

        def dfs(node, running_sum = 0):
            if not node: return
            running_sum += node.val
            if not node.left and not node.right:
                self.ans += running_sum
            else:
                dfs(node.left, running_sum)
                dfs(node.right, running_sum)

        dfs(root)
        return self.ans




"""
https://leetcode.cn/problems/binary-tree-maximum-path-sum/solution/er-cha-shu-zhong-de-zui-da-lu-jing-he-by-leetcode-/
"""


class Solution:
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)

            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain

            # 更新答案
            self.maxSum = max(self.maxSum, priceNewpath)

            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)

        maxGain(root)
        return self.maxSum
