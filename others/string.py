# s[::-1]
from collections import Counter


def uniqueOccurrences(self, arr: List[int]) -> bool:
    ans = []
    # 直接分析出现次数即可
    for k, v in Counter(arr).items():
        if v in ans:
            return False
        else:
            ans.append(v)
    return True

def isPalindrome(self, s: str) -> bool:
    sgood = "".join(ch.lower() for ch in s if ch.isalnum())
    return sgood == sgood[::-1]


def isPalindrome(self, s: str) -> bool:
    n = len(s)
    left, right = 0, n - 1

    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if left < right:
            if s[left].lower() != s[right].lower():
                return False
            left, right = left + 1, right - 1

    return True


def canPermutePalindrome(self, s: str) -> bool:
    return sum(x % 2 for x in collections.Counter(s).values()) <= 1


def canPermutePalindrome(self, s: str) -> bool:
    seen = set()
    for char in s:
        if char in seen:
            seen.remove(char)
        else:
            seen.add(char)
    return len(seen) <= 1


def reverseParentheses(self, s):
    stack = []
    for i in s:
        if i != ')':
            stack.append(i)
            continue
        tmp = []
        while stack[-1] != '(':
            tmp.append(stack.pop())
        stack.pop()
        if tmp:
            stack.extend(tmp)
    return ''.join(stack)


# https://leetcode.cn/problems/maximum-average-subtree/solution/hou-xu-bian-li-ji-lu-lei-jia-zhi-python-p25tr/
def maximumAverageSubtree(self, root):
    self.res = -float('inf')

    def recur(root):
        if not root:
            return 0, 0
        ls, ln = recur(root.left)
        rs, rn = recur(root.right)

        self.res = max(self.res, 1.0 * (ls + rs + root.val) / (ln + rn + 1))

        return ls + rs + root.val, ln + rn + 1

    recur(root)
    return self.res


class Solution:
    def rob(self, root: TreeNode) -> int:
        result = self.rob_tree(root)
        return max(result[0], result[1])

    def rob_tree(self, node):
        if node is None:
            return (0, 0)  # (偷当前节点金额，不偷当前节点金额)
        left = self.rob_tree(node.left)
        right = self.rob_tree(node.right)
        val1 = node.val + left[1] + right[1]  # 偷当前节点，不能偷子节点
        val2 = max(left[0], left[1]) + max(right[0], right[1])  # 不偷当前节点，可偷可不偷子节点
        return (val1, val2)
