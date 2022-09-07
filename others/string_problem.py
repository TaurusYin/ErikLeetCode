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

"""
https://leetcode.cn/problems/rotate-array/solution/by-codehard_livefun-ucxj/
轮转数组
"""
def rotate_array(self, nums: List[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    if k := (k % len(nums)):
        nums[:k], nums[k:] = nums[-k:], nums[:-k]



"""
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
"""


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


"""
二分查找 搜索二维矩阵
https://leetcode.cn/problems/search-a-2d-matrix-ii/solution/sou-suo-er-wei-ju-zhen-ii-by-leetcode-so-9hcx/
"""


def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    for row in matrix:
        idx = bisect.bisect_left(row, target)
        if idx < len(row) and row[idx] == target:
            return True
    return False


"""
Z 字形查找
"""


def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    x, y = 0, n - 1
    while x < m and y >= 0:
        if matrix[x][y] == target:
            return True
        if matrix[x][y] > target:
            y -= 1
        else:
            x += 1
    return False


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


"""
输入：s = "the sky is blue"
输出："blue is sky the"
https://leetcode.cn/problems/reverse-words-in-a-string/submissions/
"""


def reverseWords(self, s: str) -> str:
    return " ".join(reversed(s.split()))


def reverseWords(self, s: str) -> str:
    left, right = 0, len(s) - 1
    # 去掉字符串开头的空白字符
    while left <= right and s[left] == ' ':
        left += 1

    # 去掉字符串末尾的空白字符
    while left <= right and s[right] == ' ':
        right -= 1

    d, word = collections.deque(), []
    # 将单词 push 到队列的头部
    while left <= right:
        if s[left] == ' ' and word:
            d.appendleft(''.join(word))
            word = []
        elif s[left] != ' ':
            word.append(s[left])
        left += 1
    d.appendleft(''.join(word))

    return ' '.join(d)


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


"""
字符串相加：：
输入：num1 = "11", num2 = "123"
输出："134"
链接：https://leetcode.cn/problems/add-strings
"""
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


"""
输入：s = "42"
输出：42
解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
https://leetcode.cn/problems/string-to-integer-atoi/
"""
def myAtoi(self, s: str) -> int:
    flag = 1  # 标记正负号，默认为正
    n = len(s)
    num = 0
    i = 0
    if not s:  # 字符串为空直接返回0
        return 0
    for i in range(n):  # 跳过空格
        if s[i] != ' ':
            break
    if s[i] == '-' and i < n - 1:  # 空格后的第一个字符为'-'，并且'-'不是最后一个字符
        flag = -1
        i += 1
    elif s[i] == '+' and i < n - 1:  # 空格后的第一个字符为'+'，并且'+'不是最后一个字符
        flag = 1
        i += 1
    else:
        if not s[i].isdigit():  # 空格后的第一个字符不是'+、-'也不是数字
            return 0

    while s[i].isdigit() and i < n:
        num = num * 10 + int(s[i])
        if i == n - 1 or not s[i + 1].isdigit():  # 遍历到最后一个字符，或下一个字符不是数字，跳出循环
            break
        i += 1
    num = num * flag
    if num < -1 * 2 ** 31:  # 判断是否越界
        return -1 * 2 ** 31
    elif num > 2 ** 31 - 1:
        return 2 ** 31 - 1
    else:
        return num

"""
https://leetcode.cn/problems/decode-ways/solution/jie-ma-fang-fa-by-leetcode-solution-p8np/
示例 1：
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2：
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/decode-ways
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
def numDecodings(self, s: str) -> int:
    n = len(s)
    f = [1] + [0] * n
    for i in range(1, n + 1):
        if s[i - 1] != '0':
            f[i] += f[i - 1]
        if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) <= 26:
            f[i] += f[i - 2]
    return f[n]


"""
给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。
Excel
例如：
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...
 
示例 1：
输入：columnNumber = 1
输出："A"
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/excel-sheet-column-title
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def convertToTitle(self, columnNumber: int) -> str:
    ans = list()
    while columnNumber > 0:
        a0 = (columnNumber - 1) % 26 + 1
        ans.append(chr(a0 - 1 + ord("A")))
        columnNumber = (columnNumber - a0) // 26
    return "".join(ans[::-1])


"""
整数转换英文表示
示例 1：
输入：num = 123
输出："One Hundred Twenty Three"
示例 2：
输入：num = 12345
输出："Twelve Thousand Three Hundred Forty Five"
来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/integer-to-english-words
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""
class Solution:
    def __init__(self):
        self.nt = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
        "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        self.tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        self.t = ["Thousand", "Million", "Billion"]

    def numberToWords(self, num: int) -> str:
        def helper(num) -> list[str]:
            if num < 20:
                return [self.nt[num]]
            elif num < 100:
                res = [self.tens[num//10]]
                if num % 10:
                    res += helper(num % 10)
                return res
            elif num < 1000:
                res = [self.nt[num//100], "Hundred"]
                if num % 100:
                    res += helper(num%100)
                return res
            for p, w in enumerate(self.t, 1):
                if num < 1000 ** (p + 1):
                    return helper(num // 1000 ** p) + [w] + helper(num % 1000 ** p) if num % 1000 ** p else helper(num // 1000 ** p) + [w]
        return " ".join(helper(num))

"""
https://leetcode.cn/problems/decode-string/
示例 1：

输入：s = "3[a]2[bc]"
输出："aaabcbc"
示例 2：

输入：s = "3[a2[c]]"
输出："accaccacc"
"""


def decodeString(self, s: str) -> str:
    stack, res, multi = [], "", 0
    for c in s:
        if c == '[':
            stack.append([multi, res])
            res, multi = "", 0
        elif c == ']':
            cur_multi, last_res = stack.pop()
            res = last_res + cur_multi * res
        elif '0' <= c <= '9':
            multi = multi * 10 + int(c)
        else:
            res += c
    return res



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

def maxProduct(self, nums: List[int]) -> int:
    left, right, n = 0, 0, len(nums)
    mul, product = 1, float('-inf')
    while left < n:
        while right < n and nums[right] != 0:  # 移动right指针直至遇到0，这中间用mul累计乘积，product记录最大的乘积
            mul *= nums[right]
            right += 1
            product = max(product, mul)
        while left + 1 < right:  # 移动left指针，这中间用mul累计乘积，product记录最大的乘积
            mul /= nums[left]
            left += 1
            product = max(product, mul)
        while right < n and nums[right] == 0:  # 跳过0
            product = max(product, 0)  # 有可能所有子数组的乘积都小于0，所以0也是候选
            right += 1
        left = right
        mul = 1
    return int(product)


if __name__ == '__main__':
    rotate_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rotate_non_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
