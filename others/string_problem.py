# s[::-1]
import collections
from collections import Counter
from typing import List

from tree.binary_tree_traversal import TreeNode


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


def _rotate(matrix: List[List[int]]) -> None:
    # https://leetcode.cn/problems/rotate-image/submissions/
    n = len(matrix)
    for i in range(n // 2):
        for j in range((n + 1) // 2):
            matrix[i][j], matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1] \
                = matrix[n - j - 1][i], matrix[n - i - 1][n - j - 1], matrix[j][n - i - 1], matrix[i][j]


def _rotate(matrix: List[List[int]]) -> None:
    n = len(matrix)
    # 水平翻转
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
    # 主对角线翻转
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


#
def rotate_clockwise(matrix: List[List[int]]) -> None:
    matrix = list(zip(*matrix[::-1]))
    return matrix


def rotate_non_clockwise(matrix: List[List[int]]) -> None:
    matrix = list(zip(*matrix))[::-1]
    return matrix


def generateMatrix(self, n: int) -> List[List[int]]:
    # https://leetcode.cn/problems/spiral-matrix-ii/
    left, right, up, down = 0, n - 1, 0, n - 1
    matrix = [[0] * n for _ in range(n)]
    num = 1
    while left <= right and up <= down:
        # 填充左到右
        for i in range(left, right + 1):
            matrix[up][i] = num
            num += 1
        up += 1
        # 填充上到下
        for i in range(up, down + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        # 填充右到左
        for i in range(right, left - 1, -1):
            matrix[down][i] = num
            num += 1
        down -= 1
        # 填充下到上
        for i in range(down, up - 1, -1):
            matrix[i][left] = num
            num += 1
        left += 1
    return matrix


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


def reverseString(self, s: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1


# https://leetcode.cn/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/
"""
"""


def longestPalindrome(self, s: str) -> str:
    n = len(s)
    if n < 2:
        return s

    max_len = 1
    begin = 0
    # dp[i][j] 表示 s[i..j] 是否是回文串
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True

    # 递推开始
    # 先枚举子串长度
    for L in range(2, n + 1):
        # 枚举左边界，左边界的上限设置可以宽松一些
        for i in range(n):
            # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
            j = L + i - 1
            # 如果右边界越界，就可以退出当前循环
            if j >= n:
                break

            if s[i] != s[j]:
                dp[i][j] = False
            else:
                if j - i < 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]

            # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
            if dp[i][j] and j - i + 1 > max_len:
                max_len = j - i + 1
                begin = i
    return s[begin:begin + max_len]


def expandAroundCenter(self, s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return left + 1, right - 1


def longestPalindrome(self, s: str) -> str:
    start, end = 0, 0
    for i in range(len(s)):
        left1, right1 = self.expandAroundCenter(s, i, i)
        left2, right2 = self.expandAroundCenter(s, i, i + 1)
        if right1 - left1 > end - start:
            start, end = left1, right1
        if right2 - left2 > end - start:
            start, end = left2, right2
    return s[start: end + 1]


# https://leetcode.cn/problems/add-strings/solution/add-strings-shuang-zhi-zhen-fa-by-jyd/
def addStrings(self, num1: str, num2: str) -> str:
    res = ""
    i, j, carry = len(num1) - 1, len(num2) - 1, 0
    while i >= 0 or j >= 0:
        n1 = int(num1[i]) if i >= 0 else 0
        n2 = int(num2[j]) if j >= 0 else 0
        tmp = n1 + n2 + carry
        carry = tmp // 10
        res = str(tmp % 10) + res
        i, j = i - 1, j - 1
    return "1" + res if carry else res


"""
字符串相乘
https://leetcode.cn/problems/multiply-strings/solution/zi-fu-chuan-xiang-cheng-by-leetcode-solution/
"""


def multiply(self, num1: str, num2: str) -> str:
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    ansArr = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        x = int(num1[i])
        for j in range(n - 1, -1, -1):
            ansArr[i + j + 1] += x * int(num2[j])

    for i in range(m + n - 1, 0, -1):
        ansArr[i - 1] += ansArr[i] // 10
        ansArr[i] %= 10

    index = 1 if ansArr[0] == 0 else 0
    ans = "".join(str(x) for x in ansArr[index:])
    return ans


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


if __name__ == '__main__':
    rotate_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rotate_non_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
