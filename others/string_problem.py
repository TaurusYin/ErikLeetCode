# s[::-1]
import collections
from bisect import bisect
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
给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。回文字符串 是正着读和倒过来读一样的字符串。子字符串 是字符串中的由连续字符组成的一个序列。
链接：https://leetcode.cn/problems/palindromic-substrings
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
"""
class Solution:
    def countSubstrings(self, s: str) -> int:
        """
        （1）思路：动态规划
                我们以dp[i][j]表示区间[i, j]之间的子串是否为回文子串，这样可以思考这样三种情况的回文子串：
                    - 子串长度为1，例如 a 一定为回文子串，即 i=j 的情况
                    - 子串长度为2，且字符相同，例如 aa 一定为回文自传，即 s[i] = s[j] and j-i = 1
                    - 子串长度大于2，符合 abcba 形式的为回文子串，根据回文子串的定义，那么 abcba 去掉两边字符，仍为回文
                    子串，即bcb，转换成方程形式即 dp[i][j] = dp[i+1][j-1] and j-i > 1 and s[i] = s[j]
                剩下的均为不符合条件，即非回文子串。

        （2）复杂度：
            - 时间复杂度：O（N^2）
            - 空间复杂度：O（N^2）
        """
        # 处理特殊情况
        str_len = len(s)
        if str_len == 0 or s is None:
            return 0
        # 定义变量储存结果
        res = 0
        # 定义和初始化dp数组
        dp = [[False for _ in range(str_len)] for _ in range(str_len)]
        # 直接先给对角线赋值为True，防止出现 dp[i][j] = dp[i + 1][j - 1] 时，前值没有，例如，i=0，j=2的时候
        for i in range(str_len):
            dp[i][i] = True
        # 遍历字符串，更新dp数组
        # 注意，由于状态转义方程第三种情况是 dp[i][j] = dp[i + 1][j - 1] ，dp取决于 i+1的状态，但是正常遍历
        # 我们肯定是先有了i的状态才能有i+1的 状态，所以，此处我们遍历以 j 为主
        for j in range(str_len):
            # 因为对角线已经赋初始值，所以直接从i+1开始遍历
            for i in range(0, j):
                # 第一种情况，子串长度为1，例如 a 一定为回文子串，因为已经处理了对角线
                # 这里可以注释
                if j - i == 0:
                    dp[i][j] = True
                # 第二种和第三种可以合并，因为对于s[i]=s[j],中间加一个字符是没有影响的，即aba肯定也是回文子串
                # 所以可以合并为 j-i >= 1 and s[i] == s[j]

                # 第二种情况，子串长度为2，且字符相同，例如 aa 一定为回文自传，即 s[i] = s[j] and j-i = 1
                elif j - i == 1:
                    if s[i] == s[j]:
                        dp[i][j] = True
                # 第三种情况，子串长度大于2，符合 abcba 形式的为回文子串否则不是，即dp[i][j]取决于dp[i + 1][j - 1] 是否
                # 是回文子串
                else:
                    if s[i] == s[j]:
                        dp[i][j] = dp[i + 1][j - 1]
        # 遍历dp数组，数True的个数
        for i in range(str_len):
            for j in range(i, str_len):
                if dp[i][j] is True:
                    res += 1
        return res




# https://leetcode.cn/problems/palindrome-partitioning/
"""
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
"""
def partition(self, s: str) -> List[List[str]]:
    result = []
    path = []

    # 判断是否是回文串
    def pending_s(s):
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

    # 回溯函数，这里的index作为遍历到的索引位置，也作为终止判断的条件
    def back_track(s, index):
        # 如果对整个字符串遍历完成，并且走到了这一步，则直接加入result
        if index == len(s):
            result.append(path[:])
            return
        # 遍历每个子串
        for i in range(index, len(s)):
            # 剪枝，因为要求每个元素都是回文串，那么我们只对回文串进行递归，不是回文串的部分直接不care它
            # 当前子串是回文串
            if pending_s(s[index: i + 1]):
                # 加入当前子串到path
                path.append(s[index: i + 1])
                # 从当前i+1处重复递归
                back_track(s, i + 1)
                # 回溯
                path.pop()

    back_track(s, 0)
    return result


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
        self.nt = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven",
                   "Twelve",
                   "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        self.tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        self.t = ["Thousand", "Million", "Billion"]

    def numberToWords(self, num: int) -> str:
        def helper(num) -> list[str]:
            if num < 20:
                return [self.nt[num]]
            elif num < 100:
                res = [self.tens[num // 10]]
                if num % 10:
                    res += helper(num % 10)
                return res
            elif num < 1000:
                res = [self.nt[num // 100], "Hundred"]
                if num % 100:
                    res += helper(num % 100)
                return res
            for p, w in enumerate(self.t, 1):
                if num < 1000 ** (p + 1):
                    return helper(num // 1000 ** p) + [w] + helper(num % 1000 ** p) if num % 1000 ** p else helper(
                        num // 1000 ** p) + [w]

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


"""
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
     注意，你可以重复使用字典中的单词。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/word-break
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    import functools
    @functools.lru_cache(None)
    def back_track(s):
        if (not s):
            return True
        res = False
        for i in range(1, len(s) + 1):
            if (s[:i] in wordDict):
                res = back_track(s[i:]) or res
        return res

    return back_track(s)

# https://leetcode.cn/problems/compare-version-numbers/
def compareVersion(self, version1: str, version2: str) -> int:
    v1, v2 = version1.split('.'), version2.split('.')
    v1 = [int(c) for c in v1]
    v2 = [int(c) for c in v2]
    l = min(len(v1), len(v2))
    # 对共同长度部分比较
    for i in range(l):
        if v1[i] < v2[i]:
            return -1
        elif v1[i] > v2[i]:
            return 1
    # 到此步说明共同长度部分两者都相同 所以v1之后如果有非零的 则v1版本更老
    if len(v1) > l and any(v1[l:]):
        return 1
    if len(v2) > l and any(v2[l:]):
        return -1
    # 还没有不同之处 则返回0
    return 0


"""
字符串成为回文串的最少插入次数
给你一个字符串 s ，每一次操作你都可以在字符串的任意位置插入任意字符。
请你返回让 s 成为回文串的 最少操作次数 。
「回文串」是正读和反读都相同的字符串。
输入：s = "zzazz"
输出：0
解释：字符串 "zzazz" 已经是回文串了，所以不需要做任何插入操作。
我们用 dp[i][j] 表示对于字符串 s 的子串 s[i:j]（这里的下标从 0 开始，并且 s[i:j] 包含 s 中的第 i 和第 j 个字符），最少添加的字符数量，使得 s[i:j] 变为回文串。
我们从外向内考虑 s[i:j]：
如果 s[i] == s[j]，那么最外层已经形成了回文，我们只需要继续考虑 s[i+1:j-1]；
如果 s[i] != s[j]，那么我们要么在 s[i:j] 的末尾添加字符 s[i]，要么在 s[i:j] 的开头添加字符 s[j]，才能使得最外层形成回文。如果我们选择前者，那么需要继续考虑 s[i+1:j]；如果我们选择后者，那么需要继续考虑 s[i:j-1]。
dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1)                     if s[i] != s[j]
dp[i][j] = min(dp[i + 1][j] + 1, dp[i][j - 1] + 1, dp[i + 1][j - 1])   if s[i] == s[j]
O(N**2)，其中 NN 是字符串 s 的长度
链接：https://leetcode.cn/problems/minimum-insertion-steps-to-make-a-string-palindrome/solution/rang-zi-fu-chuan-cheng-wei-hui-wen-chuan-de-zui--2/
"""
class Solution:
    def minInsertions(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span - 1
                dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1
                if s[i] == s[j]:
                    dp[i][j] = min(dp[i][j], dp[i + 1][j - 1])
        return dp[0][n - 1]




if __name__ == '__main__':
    rotate_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    rotate_non_clockwise(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
