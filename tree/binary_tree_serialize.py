import collections

from tree.binary_tree_construct import TreeNode

#DFS

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        queue = collections.deque([root])
        res = []
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('None')
        return '[' + ','.join(res) + ']'

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        dataList = data[1:-1].split(',')
        root = TreeNode(int(dataList[0]))
        queue = collections.deque([root])
        i = 1
        while queue:
            node = queue.popleft()
            if dataList[i] != 'None':
                node.left = TreeNode(int(dataList[i]))
                queue.append(node.left)
            i += 1
            if dataList[i] != 'None':
                node.right = TreeNode(int(dataList[i]))
                queue.append(node.right)
            i += 1
        return root


# BFS
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return 'None'
        return str(root.val) + ',' + str(self.serialize(root.left)) + ',' + str(self.serialize(root.right))

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def dfs(dataList):
            val = dataList.pop(0)
            if val == 'None':
                return None
            root = TreeNode(int(val))
            root.left = dfs(dataList)
            root.right = dfs(dataList)
            return root

        dataList = data.split(',')
        return dfs(dataList)

"""
设计一个序列化和反序列化 N 叉树的算法。一个 N 叉树是指每个节点都有不超过 N 个孩子节点的有根树。序列化 / 反序列化算法的算法实现没有限制。你只需要保证 N 叉树可以被序列化为一个字符串并且该字符串可以被反序列化成原树结构即可。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/serialize-and-deserialize-n-ary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""

class Codec:
    def serialize(self, root: 'Node') -> list:
        if not root: return []
        data = [root.val]
        if root.children:
            for child in root.children:
                data.append(self.serialize(child))
        return data

    def deserialize(self, data: list) -> 'Node':
        if not data: return None
        root = Node(data[0])
        root.children = []
        for i in range(1, len(data)):
            root.children.append(self.deserialize(data[i]))
        return root


"""
给定一棵二叉树的「先序遍历」和「中序遍历」可以恢复这颗二叉树。给定一棵二叉树的「后序遍历」和「中序遍历」也可以恢复这颗二叉树。而对于二叉搜索树，给定「先序遍历」或者「后序遍历」，对其经过排序即可得到「中序遍历」。因此，仅对二叉搜索树做「先序遍历」或者「后序遍历」，即可达到序列化和反序列化的要求。此题解采用「后序遍历」的方法。
序列化时，只需要对二叉搜索树进行后序遍历，再将数组编码成字符串即可。
反序列化时，需要先将字符串解码成后序遍历的数组。在将后序遍历的数组恢复成二叉搜索树时，不需要先排序得到中序遍历的数组再根据中序和后序遍历的数组来恢复二叉树，而可以根据有序性直接由后序遍历的数组恢复二叉搜索树。后序遍历得到的数组中，根结点的值位于数组末尾，左子树的节点均小于根节点的值，右子树的节点均大于根节点的值，可以根据这些性质设计递归函数恢复二叉搜索树。
"""
class Codec:
    def serialize(self, root: TreeNode) -> str:
        arr = []
        def postOrder(root: TreeNode) -> None:
            if root is None:
                return
            postOrder(root.left)
            postOrder(root.right)
            arr.append(root.val)
        postOrder(root)
        return ' '.join(map(str, arr))

    def deserialize(self, data: str) -> TreeNode:
        arr = list(map(int, data.split()))
        def construct(lower: int, upper: int) -> TreeNode:
            if arr == [] or arr[-1] < lower or arr[-1] > upper:
                return None
            val = arr.pop()
            root = TreeNode(val)
            root.right = construct(val, upper)
            root.left = construct(lower, val)
            return root
        return construct(-inf, inf)



